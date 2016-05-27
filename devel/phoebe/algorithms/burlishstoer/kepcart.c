#ifndef _kepcart_c
#define _kepcart_c
#include <string.h>
#include <stdio.h>
#include <math.h>
#define MAX_CART_ITER 100

typedef struct {
  double x, y, z, xd, yd, zd;
} State;

extern double machine_epsilon;

void keplerian(double gm, State state, 
	  double *a, double *e, double *i, double *longnode, double *argperi, double *meananom)
{
  double rxv_x, rxv_y, rxv_z, hs, h, parameter;
  double e_x, e_y, e_z;
  double ecosargperi, esinargperi;
  double rhat_x, rhat_y, rhat_z;
  double vxh_x, vxh_y, vxh_z;
  double r, vs, rdotv, rdot, ecostrueanom, esintrueanom, cosnode, sinnode;
  double rcosu, rsinu, u, trueanom, eccanom;
  double n, tp;

  /* find direction of angular momentum vector */
  rxv_x = state.y * state.zd - state.z * state.yd;
  rxv_y = state.z * state.xd - state.x * state.zd;
  rxv_z = state.x * state.yd - state.y * state.xd;
  hs = rxv_x * rxv_x + rxv_y * rxv_y + rxv_z * rxv_z;
  h = sqrt(hs);

  r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
  vs = state.xd * state.xd + state.yd * state.yd + state.zd * state.zd;
  /* v = sqrt(vs);  unnecessary */
  rdotv = state.x * state.xd + state.y * state.yd + state.z * state.zd;
  rdot = rdotv / r;

  rhat_x = state.x/r; rhat_y = state.y/r; rhat_z = state.z/r;

  vxh_x = state.yd * rxv_z - state.zd * rxv_y;
  vxh_y = state.zd * rxv_x - state.xd * rxv_z;
  vxh_z = state.xd * rxv_y - state.yd * rxv_x;

  e_x = vxh_x/gm - rhat_x;
  e_y = vxh_y/gm - rhat_y;
  e_z = vxh_z/gm - rhat_z;

  //printf("e: %le\n", sqrt(e_x*e_x + e_y*e_y + e_z*e_z));

  parameter = hs / gm;

  *i = acos(rxv_z / h);

  if(rxv_x!=0.0 || rxv_y!=0.0) {
    *longnode = atan2(rxv_x, -rxv_y);
  } else {
    *longnode = 0.0;
  }

/*   ecosargperi = e_x*cos(*longnode) + e_y*sin(*longnode); */

/*   if(sin(*i) != 0.0){ */
/*     esinargperi = e_z/sin(*i); */
/*   }else{ */
/*     esinargperi = 0.0; */
/*   } */


  ecostrueanom = parameter / r - 1.0;
  esintrueanom = rdot * h / gm;
  *e = sqrt(ecostrueanom * ecostrueanom + esintrueanom * esintrueanom);

  if(esintrueanom!=0.0 || ecostrueanom!=0.0) {
    trueanom = atan2(esintrueanom, ecostrueanom);
  } else {
    trueanom = 0.0;
  }

  //printf("trueanom: %.16lf\n", trueanom);

  cosnode = cos(*longnode);
  sinnode = sin(*longnode);

  /* u is the argument of latitude */
  rcosu = state.x * cosnode + state.y * sinnode;
  rsinu = (state.y * cosnode - state.x * sinnode)/cos(*i);
  //rsinu = (state.y * cosnode*h - state.x * sinnode*h)/rxv_z;

  //fprintf(stderr, "# %le %le %le %le\n", rcosu, rsinu, (state.y * cosnode - state.x * sinnode), cos(*i));

  if(rsinu!=0.0 || rcosu!=0.0) {
    u = atan2(rsinu, rcosu);
  } else {
    u = 0.0;
  }

  *argperi = u - trueanom;

  *a = 1.0 / (2.0 / r - vs / gm);

  eccanom = 2.0 * atan(sqrt((1.0 - *e)/(1.0 + *e)) * tan(trueanom/2.0));
  *meananom = eccanom - *e * sin(eccanom);
  
  /* if(*a > 0.0){
    n = sqrt(gm/pow(*a, 3.0));
    tp = - *meananom/n;
    printf("tp: %.16le\n", tp);
  }

  if(*a < 0.0){
    double Z, F;

    n = sqrt(-gm/pow(*a, 3.0));

    Z = (1.0 - r/(*a))/(*e);
    F = log(Z + sqrt(Z*Z - 1.0));
    if(rdotv < 0.0){
      F = -F;
    }
    
    tp = -(*e * sinh(F) - F)/n;
    printf("tp: %.16le\n", tp);

  }

  if(*a == 0.0){
    double q, tau;
    
    q = 0.5*hs/gm;
    n = sqrt(gm/(2.0*pow(q, 3.0)));
    tau = sqrt(r/q - 1.0);
    if(rdotv < 0.0){
      tau = -tau;
    }
    
    tp = -(tau*tau*tau/3.0 + tau)/n;
    printf("tp: %.16le\n", tp);
  }

  printf("ecosw: %le %le\n", ecosargperi, *e * cos(*argperi));
  printf("argperi: %le %le\n", *argperi, atan2(esinargperi, ecosargperi));*/


  return;
}



void cartesian(double gm, 
	       double a, double e, double i, double longnode, double argperi, double meananom, 
	       State *state, int * icount)
{
  double meanmotion, cosE, sinE, foo;
  double x, y, z, xd, yd, zd;
  double xp, yp, zp, xdp, ydp, zdp;
  double cosw, sinw, cosi, sini, cosnode, sinnode;
  double E0, E1, E2, den;
  int count;

  count=0;
  /* first compute eccentric anomaly */
  E0 = meananom; 
  do {
    E1 = meananom + e * sin(E0);
    E2 = meananom + e * sin(E1);

    den = E2 - 2.0*E1 + E0;
    count++;
    if(fabs(den) > machine_epsilon) {
      E0 = E0 - (E1-E0)*(E1-E0)/den;
    }
    else {
      E0 = E2;
      E2 = E1;
    }
  } while(fabs(E0-E2) > machine_epsilon &&
	  count<MAX_CART_ITER);

  *icount = count;

  cosE = cos(E0);
  sinE = sin(E0);

  /* compute unrotated positions and velocities */
  foo = sqrt(1.0 - e*e);
  meanmotion = sqrt(gm/(a*a*a));
  x = a * (cosE - e);
  y = foo * a * sinE;
  z = 0.0;
  xd = -a * meanmotion * sinE / (1.0 - e * cosE);
  yd = foo * a * meanmotion * cosE / (1.0 - e * cosE);
  zd = 0.0;

  /* rotate by argument of perihelion in orbit plane*/
  cosw = cos(argperi);
  sinw = sin(argperi);
  xp = x * cosw - y * sinw;
  yp = x * sinw + y * cosw;
  zp = z;
  xdp = xd * cosw - yd * sinw;
  ydp = xd * sinw + yd * cosw;
  zdp = zd;

  /* rotate by inclination about x axis */
  cosi = cos(i);
  sini = sin(i);
  x = xp;
  y = yp * cosi - zp * sini;
  z = yp * sini + zp * cosi;
  xd = xdp;
  yd = ydp * cosi - zdp * sini;
  zd = ydp * sini + zdp * cosi;

  /* rotate by longitude of node about z axis */
  cosnode = cos(longnode);
  sinnode = sin(longnode);
  state->x = x * cosnode - y * sinnode;
  state->y = x * sinnode + y * cosnode;
  state->z = z;
  state->xd = xd * cosnode - yd * sinnode;
  state->yd = xd * sinnode + yd * cosnode;
  state->zd = zd;

  return;
}

#endif

