/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "app_mesodiffusion.h"
#include "solve.h"
#include "domain.h"
#include "random_park.h"
#include "memory.h"
#include "error.h"

using namespace SPPARKS_NS;

enum{ZERO,VACANT,OCCUPIED,TOP};
//enum{NO_ENERGY,LINEAR,NONLINEAR};
enum{DEPOSITION,NNHOP,SCHWOEBEL};
enum{NOSWEEP,RANDOM,RASTER,COLOR,COLOR_STRICT};  // from app_lattice.cpp

#define DELTAEVENT 100000

/* ---------------------------------------------------------------------- */

AppMesoDiffusion::AppMesoDiffusion(SPPARKS *spk, int narg, char **arg) : 
  AppLattice(spk,narg,arg)
{
  // these can be changed by model choice, see below

  ninteger = 1;
  ndouble = 0;
  delpropensity = 2;
  delevent = 1;
  allow_kmc = 1;
  allow_rejection = 0;
  allow_masking = 0;
  numrandom = 1;
  hopstyle = SCHWOEBEL;

  // parse arguments

  if (narg < 6) error->all(FLERR,"Illegal app_style mesodiffusion command");
// By CWP, Apr. 24, 2017  
// app_style mesodiffusion D a \gamma Nmax Nmin
//  D: diffusion coeff; a: lattice constant; \gamma: surface energy
// Nmax, Nmin: see Schwoebel 

  D = atof(arg[1]);
  a = atof(arg[2]);
  gamma = atof(arg[3]);
  Gamma = 4*D/(a*a);  // determine mesoscale diffusion hopping rate
    
    nsmax = atoi(arg[3]);
    nsmin = atoi(arg[4]);
  
  // increment delpropensity by 1 for nonlinear energy
  // increment delpropensity and delevent by 1 for Schwoebel hops
  // change allow_rejection to 1 for linear energy and non-Schwoebel hops

  
  if (hopstyle == SCHWOEBEL) delpropensity++;
  if (hopstyle == SCHWOEBEL) delevent++;
  allow_rejection = 1;

  create_arrays();

  esites = psites = NULL;
  echeck = pcheck = NULL;
  maxevent = 0;
  events = NULL;
  firstevent = NULL;

  //hbarrier = sbarrier = NULL;
  ecoord = NULL;
  hopsite = NULL;
  marklist = NULL;
  mark = NULL;

  allocated = 0;

  // default settings for app-specific commands

  depflag = 0;
  //barrierflag = 0;

  // statistics

  ndeposit = ndeposit_failed = 0;
  nfirst = nsecond = 0;
}

/* ---------------------------------------------------------------------- */

AppMesoDiffusion::~AppMesoDiffusion()
{
  delete [] esites;
  delete [] psites;
  delete [] echeck;
  delete [] pcheck;
  memory->sfree(events);
  memory->destroy(firstevent);

 // memory->destroy(hbarrier);
 // memory->destroy(sbarrier);
  delete [] ecoord;

  delete [] hopsite;
  delete [] marklist;
  memory->destroy(mark);
}

/* ----------------------------------------------------------------------
   input script commands unique to this app
------------------------------------------------------------------------- */

void AppMesoDiffusion::input_app(char *command, int narg, char **arg)
{
  if (sites_exist == 0) {
    char str[128];
    sprintf(str,"Cannot use %s command until sites exist",command);
    error->all(FLERR,str);
  }

  if (!allocated) allocate_data();
  allocated = 1;

  if (strcmp(command,"deposition") == 0) {
    if (narg < 1) error->all(FLERR,"Illegal deposition command");
    if (strcmp(arg[0],"off") == 0) {
      if (narg != 1 ) error->all(FLERR,"Illegal deposition command");
      depflag = 0;
      return;
    }

    if (narg != 7) error->all(FLERR,"Illegal deposition command");
    depflag = 1;
    deprate = atof(arg[0]);
    dir[0] = atof(arg[1]);
    dir[1] = atof(arg[2]);
    dir[2] = atof(arg[3]);
    d0 = atof(arg[4]);
    coordlo = atoi(arg[5]);
    coordhi = atoi(arg[6]);
    if (deprate < 0.0) error->all(FLERR,"Illegal deposition command");
    if (domain->dimension == 2 && (dir[1] >= 0.0 || dir[2] != 0.0))
      error->all(FLERR,"Illegal deposition command");
    if (domain->dimension == 3 && dir[2] >= 0.0)
      error->all(FLERR,"Illegal deposition command");
    if (d0 < 0.0) error->all(FLERR,"Illegal deposition command");
    if (coordlo < 0 || coordhi > maxneigh || coordlo > coordhi)
      error->all(FLERR,"Illegal deposition command");

    double len = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    dir[0] /= len;
    dir[1] /= len;
    dir[2] /= len;

  }  else error->all(FLERR,"Unrecognized command");
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppMesoDiffusion::grow_app()
{
  lattice = iarray[0];
}

/* ----------------------------------------------------------------------
   initialize before each run
   check validity of site values
------------------------------------------------------------------------- */

void AppMesoDiffusion::init_app()
{
  if (depflag && nprocs > 1)
    error->all(FLERR,"Cannot perform deposition in parallel");
  if (depflag && nsector > 1)
    error->all(FLERR,"Cannot perform deposition with multiple sectors");

  if (!allocated) allocate_data();
  allocated = 1;

  dimension = domain->dimension;
  dt_sweep = 1.0/maxneigh;

  // site validity

  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    if (lattice[i] < VACANT || lattice[i] > TOP) flag = 1;
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");
}

/* ----------------------------------------------------------------------
   setup before each run
------------------------------------------------------------------------- */

void AppMesoDiffusion::setup_app()
{
  for (int i = 0; i < nlocal+nghost; i++) echeck[i] = pcheck[i] = 0;

  // clear event list

  nevents = 0;
  for (int i = 0; i < nlocal; i++) firstevent[i] = -1;
  for (int i = 0; i < maxevent; i++) events[i].next = i+1;
  freeevent = 0;
}

/* ----------------------------------------------------------------------
   compute energy of site
------------------------------------------------------------------------- */

double AppMesoDiffusion::site_energy(int i)
{
  // energy only non-zero for OCCUPIED sites when energy included in model

  if (lattice[i] != OCCUPIED) return 0.0;


  // energy is a linear function of coordination number, just count bonds

  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (lattice[neighbor[i][j]] == VACANT) eng++;
  return (double) gamma*eng;
}



/* ----------------------------------------------------------------------
   KMC method
   compute total propensity of owned site summed over possible events
------------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */


double AppMesoDiffusion::site_propensity(int i)
{
  int j,ihop,nhop1,nhop2,eflag;
  double einitial,edelta,probone,proball;

  // events = OCCUPIED site exchanges with adjacent VACANT site
  // if engstyle, compute edelta between initial and final state
  // factor of 2 in edelta accounts for energy change of neighbors of I,J
  // if barrierflag, compute coordination of sites I,J
  // for 1st neigh hop, set delta = 1 to remove hopping atom itself
  // propensity is function of edelta, barrier, up/down hill, temperature

  clear_events(i);

  if (lattice[i] != OCCUPIED) {
    if (depflag && i == 0) {
      add_event(i,-1,deprate,DEPOSITION);
      return deprate;
    } else return 0.0;
  }

  // nhop1 = 1st neigh hops, nhop2 = 2nd neigh hops
  // hopsite = all possible hop sites

  nhop1 = 0;
  for (j = 0; j < numneigh[i]; j++)
    if (lattice[neighbor[i][j]] == VACANT) hopsite[nhop1++] = neighbor[i][j];
  nhop2 = nhop1;
  if (hopstyle == SCHWOEBEL) 
    nhop2 += schwoebel_enumerate(i,&hopsite[nhop1]);

  // loop over all possible hops

  einitial = site_energy(i);

  proball = 0.0;
 // double **barrier = hbarrier;
  int delta = 1;

  for (ihop = 0; ihop < nhop2; ihop++) {
    j = hopsite[ihop];
    if (ihop == nhop1) {
      delta = 0;
    }

    lattice[i] = VACANT;
    lattice[j] = OCCUPIED;
    probone = 0.0;

    edelta = site_energy(j) - einitial;
    
    
      if (edelta <= 0.0)
	probone = Gamma;  // regular hopping with rate Gamma
      else
	probone = exp((-2.0*edelta) * 
		      t_inverse) * Gamma;
    	
    
    if (probone > 0.0) {
      eflag = (ihop < nhop1) ? NNHOP : SCHWOEBEL;
      add_event(i,j,probone,eflag);
      proball += probone;
    }
    
    lattice[i] = OCCUPIED;
    lattice[j] = VACANT;
  }

  // add in single deposition event, stored by site 0

  if (depflag && i == 0) {
    add_event(i,-1,deprate,DEPOSITION);
    proball += deprate;
  }
  
  return proball;
}

/* ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   KMC method
   choose and perform an event for site
------------------------------------------------------------------------- */


void AppMesoDiffusion::site_event(int i, class RandomPark *random)
{
  int j,m,isite;
  
  // pick one event from total propensity by accumulating its probability
  // compare prob to threshhold, break when reach it to select event
  // perform event

  double threshhold = random->uniform() * propensity[i2site[i]]; //CWP: propensity here is the propensity of each set
  double proball = 0.0;                                          // Hence, i is index of lattice, and i2site 
                                                                 // is the mapping between lattice index and that of the sublattice
  int ievent = firstevent[i]; 
  while (1) {
    proball += events[ievent].propensity;
    if (proball >= threshhold) break;
    ievent = events[ievent].next;
    if (ievent < 0) error->one(FLERR,"Did not reach event propensity threshhold");
  }

  // deposition or hop event
  // for deposition event, find site to deposit on
  // after deposition, reset i and j to that site
  // so propensity around it is updated correctly

  if (events[ievent].style == DEPOSITION) {
    m = find_deposition_site(random);
    if (m < 0) return;
    lattice[m] = OCCUPIED;
    i = j = m;
  } else {
    j = events[ievent].destination;
    if (events[ievent].style == NNHOP) nfirst++;
    else nsecond++;
    lattice[i] = VACANT;
    lattice[j] = OCCUPIED;
  }

  // compute propensity changes for self and swap site and their neighs
  // 1,2 neighs for NNHOP and 1,2,3 neighs for SCHWOEBEL
  // ignore update of sites with isite < 0
  // use echeck[] to avoid resetting propensity of same site

  int nsites = 0;

  isite = i2site[i];
  propensity[isite] = site_propensity(i);
  esites[nsites++] = isite;
  echeck[isite] = 1;

  isite = i2site[j];
  if (isite >= 0) {
    propensity[isite] = site_propensity(j);
    esites[nsites++] = isite;
    echeck[isite] = 1;
  }

  if (hopstyle == NNHOP) {
    nsites += neighbor2(i,&esites[nsites]);
    nsites += neighbor2(j,&esites[nsites]);
  } else {
    nsites += neighbor3(i,&esites[nsites]);
    nsites += neighbor3(j,&esites[nsites]);
  }
  
  solve->update(nsites,esites,propensity);

  // sanity check on all propensity values

  /*
  printf("EVENT %d %d\n",i,j);
  for (m = 0; m < nlocal; m++) {
    if (fabs(propensity[m]-site_propensity(m)) > 1.0e-6) {
      printf("BAD PROP = %d %d %d %g %g\n",
	     id[i],id[j],id[m],propensity[m],site_propensity(m));
      error->one(FLERR,"BAD DONE");
    }
  }
  */

  // clear echeck array

  for (m = 0; m < nsites; m++) echeck[esites[m]] = 0;
}


void AppMesoDiffusion::site_event_rejection(int i, class RandomPark *random)
{
}



/* ----------------------------------------------------------------------
   re-compute propensities out to 2nd neighbors of site I
------------------------------------------------------------------------- */

int AppMesoDiffusion::neighbor2(int i, int *sites)
{
  int k,kk,m,mm,isite;

  int nsites = 0;

  for (k = 0; k < numneigh[i]; k++) {
    m = neighbor[i][k];
    isite = i2site[m];
    if (isite >= 0 && echeck[isite] == 0) {
      propensity[isite] = site_propensity(m);
      sites[nsites++] = isite;
      echeck[isite] = 1;
    }
    for (kk = 0; kk < numneigh[m]; kk++) {
      mm = neighbor[m][kk];
      isite = i2site[mm];
      if (isite >= 0 && echeck[isite] == 0) {
	propensity[isite] = site_propensity(mm);
	sites[nsites++] = isite;
	echeck[isite] = 1;
      }
    }
  }

  return nsites;
}

/* ----------------------------------------------------------------------
   re-compute propensities out to 3rd neighbors of site I
------------------------------------------------------------------------- */

int AppMesoDiffusion::neighbor3(int i, int *sites)
{
  int k,kk,kkk,m,mm,mmm,isite;

  int nsites = 0;

  for (k = 0; k < numneigh[i]; k++) {
    m = neighbor[i][k];
    isite = i2site[m];
    if (isite >= 0 && echeck[isite] == 0) {
      propensity[isite] = site_propensity(m);
      sites[nsites++] = isite;
      echeck[isite] = 1;
    }
    for (kk = 0; kk < numneigh[m]; kk++) {
      mm = neighbor[m][kk];
      isite = i2site[mm];
      if (isite >= 0 && echeck[isite] == 0) {
	propensity[isite] = site_propensity(mm);
	sites[nsites++] = isite;
	echeck[isite] = 1;
      }
      for (kkk = 0; kkk < numneigh[mm]; kkk++) {
	mmm = neighbor[mm][kkk];
	isite = i2site[mmm];
	if (isite >= 0 && echeck[isite] == 0) {
	  propensity[isite] = site_propensity(mmm);
	  sites[nsites++] = isite;
	  echeck[isite] = 1;
	}
      }
    }
  }

  return nsites;
}

/* ----------------------------------------------------------------------
   re-compute propensities out to 4th neighbors of site I
------------------------------------------------------------------------- */

int AppMesoDiffusion::neighbor4(int i, int *sites)
{
  int k,kk,kkk,kkkk,m,mm,mmm,mmmm,isite;

  int nsites = 0;

  for (k = 0; k < numneigh[i]; k++) {
    m = neighbor[i][k];
    isite = i2site[m];
    if (isite >= 0 && echeck[isite] == 0) {
      propensity[isite] = site_propensity(m);
      sites[nsites++] = isite;
      echeck[isite] = 1;
    }
    for (kk = 0; kk < numneigh[m]; kk++) {
      mm = neighbor[m][kk];
      isite = i2site[mm];
      if (isite >= 0 && echeck[isite] == 0) {
	propensity[isite] = site_propensity(mm);
	sites[nsites++] = isite;
	echeck[isite] = 1;
      }
      for (kkk = 0; kkk < numneigh[mm]; kkk++) {
	mmm = neighbor[mm][kkk];
	isite = i2site[mmm];
	if (isite >= 0 && echeck[isite] == 0) {
	  propensity[isite] = site_propensity(mmm);
	  sites[nsites++] = isite;
	  echeck[isite] = 1;
	}
	for (kkkk = 0; kkkk < numneigh[mmm]; kkkk++) {
	  mmmm = neighbor[mmm][kkkk];
	  isite = i2site[mmmm];
	  if (isite >= 0 && echeck[isite] == 0) {
	    propensity[isite] = site_propensity(mmmm);
	    sites[nsites++] = isite;
	    echeck[isite] = 1;
	  }
	}
      }
    }
  }

  return nsites;
}

/* ---------------------------------------------------------------------- */

int AppMesoDiffusion::ncoord(int i)
{
  int count = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (lattice[neighbor[i][j]] == OCCUPIED) count++;
  return count;
}

/* ----------------------------------------------------------------------
   clear all events out of list for site I
   add cleared events to free list
------------------------------------------------------------------------- */

void AppMesoDiffusion::clear_events(int i)
{
  int next;
  int index = firstevent[i];
  while (index >= 0) {
    next = events[index].next;
    events[index].next = freeevent;
    freeevent = index;
    nevents--;
    index = next;
  }
  firstevent[i] = -1;
}

/* ----------------------------------------------------------------------
   add an event to list for site I
   event = exchange with site J with probability = propensity
------------------------------------------------------------------------- */

void AppMesoDiffusion::add_event(int i, int destination, 
			      double propensity, int eventflag)
{
  // grow event list and setup free list

  if (nevents == maxevent) {
    maxevent += DELTAEVENT;
    events = 
      (Event *) memory->srealloc(events,maxevent*sizeof(Event),"app:events");
    for (int m = nevents; m < maxevent; m++) events[m].next = m+1;
    freeevent = nevents;
  }

  int next = events[freeevent].next;
  events[freeevent].propensity = propensity;
  events[freeevent].destination = destination;
  events[freeevent].style = eventflag;
  events[freeevent].next = firstevent[i];
  firstevent[i] = freeevent;
  freeevent = next;
  nevents++;
}

/* ----------------------------------------------------------------------
   enumerate Schwoebel hop events centered around OCCUPIED site I
   assume mark array is currently cleared, use it, clear it when done
------------------------------------------------------------------------- */

int AppMesoDiffusion::schwoebel_enumerate(int i, int *site)
{
  int j,k,m,jneigh,kneigh,count;

  int nhop = 0;

  // if coord(I) > Nmax, no hops possible

  count = 0;
  for (j = 0; j < numneigh[i]; j++)
    if (lattice[neighbor[i][j]] == OCCUPIED) count++;
  if (count > nsmax) return nhop;

  // mark first neighbors of site I as vacant = 1, occupied = 2

  for (jneigh = 0; jneigh < numneigh[i]; jneigh++) {
    j = neighbor[i][jneigh];
    if (lattice[j] == VACANT) mark[j] = 1;
    else if (lattice[j] == OCCUPIED) mark[j] = 2;
  }

  // loop over 1st and 2nd neighbors of site I to find possible hops to K
  // if K not vacant, no hop
  // if mark(K) = 1 or 2, K is a 1st neigh, not a 2nd neigh
  // if mark(K) = 30, already seen as a possible hop
  // mark(K) = 10 or 20, it has a 1st neigh that is vacant or occupied
  // mark(K) = 30, has both a vacant/occupied 1st neigh, so consider it
  // if coord(K) < Nmin, no hop possible
  // if all criteria met, then it is a candidate hop, add to site[]
  
  int nlist = 0;
  for (jneigh = 0; jneigh < numneigh[i]; jneigh++) {
    j = neighbor[i][jneigh];
    for (kneigh = 0; kneigh < numneigh[j]; kneigh++) {
      k = neighbor[j][kneigh];
      if (lattice[k] != VACANT) continue;
      if (mark[k] == 1 || lattice[k] == 2) continue;
      if (mark[k] == 30) continue;
      if (mark[k] == 10*mark[j]) continue;
      if (mark[k] == 0) marklist[nlist++] = k;
      mark[k] += 10*mark[j];
      if (mark[k] != 30) continue;

      count = 0;
      for (m = 0; m < numneigh[k]; m++)
	if (lattice[neighbor[k][m]] == OCCUPIED) count++;
      if (count < nsmin) continue;

      site[nhop++] = k;
    }
  }

  // clear marked sites, 1st and 2nd neighbors

  for (j = 0; j < numneigh[i]; j++) mark[neighbor[i][j]] = 0;
  for (k = 0; k < nlist; k++) mark[marklist[k]] = 0;

  return nhop;
}

/* ----------------------------------------------------------------------
   identify a VACANT site to deposit an atom
   return -1 if could not find a suitable site
------------------------------------------------------------------------- */

int AppMesoDiffusion::find_deposition_site(RandomPark *random)
{
  // pick a random position at top of box

  double start[3];
  start[0] = domain->boxxlo + domain->xprd*random->uniform();
  if (dimension == 2) {
    start[1] = domain->boxyhi;
    start[2] = 0.0;
  } else {
    start[1] = domain->boxylo + domain->yprd*random->uniform();
    start[2] = domain->boxzhi;
  }

  // for each vacant site:
  // discard site if neighbor count not between coordlo and coordhi
  // find site whose projected distance is closest to start point

  int i,ncount;
  double dist2start;

  int closesite = -1;
  double closedist = 1.0e20;

  for (i = 0; i < nlocal; i++) {
    if (lattice[i] != VACANT) continue;
    ncount = 0;
    for (int j = 0; j < numneigh[i]; j++)
      if (lattice[neighbor[i][j]] == OCCUPIED) ncount++;
    if (ncount < coordlo || ncount > coordhi) continue;

    if (exceed_limit(i,start,dist2start)) continue;
    if (dist2start < closedist) {
      closedist = dist2start;
      closesite = i;
    }
  }

  if (closesite < 0) ndeposit_failed++;
  else ndeposit++;

  return closesite;
}

/* ----------------------------------------------------------------------
   test if site M is within normal distance d0 from incident line
   if so, return 0 and dist2start, else return 1
   site M really becomes a periodic image in XY of M, adjusted via iprd/jprd
   dist2start = dist from site M to starting point of incident line
   dist2start is dist along incident line from start point to
     normal projection point of M
------------------------------------------------------------------------- */

int AppMesoDiffusion::exceed_limit(int m, double *start, double &dist2start)
{
  int increment,iprd,jprd;

  iprd = jprd = 0;
  double d0sq = d0*d0;

  double distsq = distsq_to_line(m,start,iprd,jprd,dist2start);
  double newdistsq = distsq_to_line(m,start,iprd-1,jprd,dist2start);
  if (newdistsq < distsq) increment = -1;
  else increment = 1;

  iprd += increment;
  newdistsq = distsq_to_line(m,start,iprd,jprd,dist2start);
  while (newdistsq < distsq) {
    distsq = newdistsq;
    iprd += increment;
    newdistsq = distsq_to_line(m,start,iprd,jprd,dist2start);
  }
  iprd -= increment;

  if (dimension == 3) {
    newdistsq = distsq_to_line(m,start,iprd,jprd-1,dist2start);
    if (newdistsq < distsq) increment = -1;
    else increment = 1;

    jprd += increment;
    newdistsq = distsq_to_line(m,start,iprd,jprd,dist2start);
    while (newdistsq < distsq) {
      distsq = newdistsq;
      jprd += increment;
      newdistsq = distsq_to_line(m,start,iprd,jprd,dist2start);
    }
  }
  jprd -= increment;

  if (distsq > d0sq) return 1;
  distsq = distsq_to_line(m,start,iprd,jprd,dist2start);
  return 0;
}

/* ----------------------------------------------------------------------
   compute normal distsq from site M to incident line of deposition
   site M really becomes a periodic image in XY of M, adjusted via iprd/jprd
   also compute and return dist2start
   dist2start = dist from site M to starting point of incident line
   dist2start is dist along incident line from start point to
     normal projection point of M
------------------------------------------------------------------------- */

double AppMesoDiffusion::distsq_to_line(int m, double *start,
				    int iprd, int jprd, double &dist2start)
{
  double delta[3],projection[3],offset[3];

  delta[0] = xyz[m][0] + iprd*domain->xprd - start[0];
  delta[1] = xyz[m][1] + jprd*domain->yprd - start[1];
  delta[2] = xyz[m][2] - start[2];
    
  dist2start = dir[0]*delta[0] + dir[1]*delta[1] + dir[2]*delta[2];
  projection[0] = dist2start*dir[0];
  projection[1] = dist2start*dir[1];
  projection[2] = dist2start*dir[2];
  
  offset[0] = delta[0] - projection[0];
  offset[1] = delta[1] - projection[1];
  offset[2] = delta[2] - projection[2];
  return offset[0]*offset[0] + offset[1]*offset[1] + offset[2]*offset[2];
}

/* ----------------------------------------------------------------------
   allocate data structs that have to wait until sites exist
   so that nlocal,nghost,maxneigh are set
------------------------------------------------------------------------- */

void AppMesoDiffusion::allocate_data()
{
  // for no_energy or linear:
  //   make esites large enough for 2 sites and their 1,2 neighbors
  //   do not need psites
  // for nonlinear:
  //   make esites large enough for 2 sites and their 1,2,3 neighbors
  //   make psites large enough for 2 sites and their 1st neighbors
  // Schwoebel hops add one level of neighbor dependence to esites

 
    int emax = 1 + maxneigh + maxneigh*maxneigh + maxneigh*maxneigh*maxneigh;
    esites = new int[2*emax];
    psites = NULL;
  
  echeck = new int[nlocal+nghost];
  pcheck = new int[nlocal+nghost];

  memory->create(firstevent,nlocal,"app:firstevent");

  ecoord = new double[maxneigh+1];
  for (int i = 0; i <= maxneigh; i++) ecoord[i] = 0.0;

//  memory->create(hbarrier,maxneigh+1,maxneigh+1,"app:hbarrier");
//  memory->create(sbarrier,maxneigh+1,maxneigh+1,"app:sbarrier");

 // for (int i = 0; i <= maxneigh; i++)
 //   for (int j = 0; j <= maxneigh; j++)
 //     hbarrier[i][j] = sbarrier[i][j] = 0.0;

  hopsite = new int[maxneigh*maxneigh + maxneigh];
  marklist = new int[maxneigh*maxneigh];

  mark = NULL;
  if (hopstyle == SCHWOEBEL) memory->create(mark,nlocal+nghost,"app:mark");
  if (mark)
    for (int i = 0; i < nlocal+nghost; i++) mark[i] = 0;
}
