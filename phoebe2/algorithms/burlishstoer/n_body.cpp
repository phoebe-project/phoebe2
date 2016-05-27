#include <stdlib.h>
#include <math.h>
#include "n_body.h"
#include <omp.h>

using namespace std;

#define FRAC_IMP 3. // factor divided into H at each improvement step

#define KMAX 8 // DO NOT CHANGE

#define COPY(a,b,N) for (int i=0; i<N; i++) for (int j=0; j<3; j++) {b[i][j] = a[i][j];}
#define SGN(a) (a < 0 ? -1 : 1)

#define MM3_M0(a,b,h,N) for (int i=0; i < N; i++) for (int j=0; j<3; j++) {a[i][j] = b[i][j]+h*a[i][j];}
#define MM3_RES(r,a,b,c,h,N) for (int i=0; i < N; i++) for (int j=0; j<3; j++) {r[i][j] = 0.5*(a[i][j]+b[i][j]+h*c[i][j]);}
#define COPY_T(a,T,N,k1,k2) for (int i=0; i< N; i++) for (int j=0; j < 3; j++) {T[i][j][k1][k2] = a[i][j];}
#define T_COPY(T,a,N,k1,k2) for (int i=0; i< N; i++) for (int j=0; j < 3; j++) {a[i][j] = T[i][j][k1][k2];}

#define A3(r2,r1) for (int LOOP = 0; LOOP < 3; LOOP++) { r2[LOOP] = r1[LOOP]; }
#define A30(r) for (int LOOP=0; LOOP < 3; LOOP++) { r[LOOP] = 0;}
#define A3P(r2,r1) for (int LOOP = 0; LOOP < 3; LOOP++) { r2[LOOP] += r1[LOOP];}

static const double rat2Mat[KMAX][KMAX] = 
  {{-1, -1.3333333333333332593, -1.125, -1.0666666666666666519, -1.0416666666666667407, -1.028571428571428692, -1.0208333333333332593, -1.0158730158730158166},
   {0.33333333333333331483, -1, -1.7999999999999998224, -1.3333333333333332593, -1.1904761904761904656, -1.125, -1.0888888888888887951, -1.0666666666666666519},
   {0.125, 0.80000000000000004441, -1, -2.2857142857142855874, -1.5625, -1.3333333333333332593, -1.2249999999999998668, -1.1636363636363635798},
   {0.066666666666666665741, 0.33333333333333331483, 1.2857142857142858094, -1, -2.7777777777777785673, -1.7999999999999998224, -1.4848484848484846399, -1.3333333333333332593},
   {0.041666666666666664354, 0.19047619047619046562, 0.56249999999999988898, 1.7777777777777776791, -1, -3.2727272727272738173, -2.0416666666666665186, -1.6410256410256409687},
   {0.028571428571428570536, 0.125, 0.33333333333333331483, 0.80000000000000004441, 2.2727272727272729291, -1, -3.7692307692307682743, -2.2857142857142855874},
   {0.020833333333333332177, 0.088888888888888892281, 0.22499999999999995004, 0.48484848484848486194, 1.0416666666666669627, 2.769230769230766942, -1, -4.2666666666666666075},
   {0.015873015873015872135, 0.066666666666666665741, 0.16363636363636363535, 0.33333333333333331483, 0.64102564102564085768, 1.2857142857142858094, 3.2666666666666679397, -1}};

inline
int seqBS(int n) { return 2*(n+1); }

double neville(double Tr[][3][KMAX][KMAX],double Tv[][3][KMAX][KMAX], int N, int k) {

  double rat, error = 0.0;

  for (int j=0; j <k; j++) {
    for (int i =0; i< N; i++) {
      for (int l=0; l < 3; l++) {
	Tr[i][l][k][j+1] = Tr[i][l][k][j]+(Tr[i][l][k][j]-Tr[i][l][k-1][j])*rat2Mat[k][k-(j+1)];
	Tv[i][l][k][j+1] = Tv[i][l][k][j]+(Tv[i][l][k][j]-Tv[i][l][k-1][j])*rat2Mat[k][k-(j+1)];
      }
    }
  }

  if (k > 0) {
    for (int i=0;i<N;i++) {
      for (int l=0; l < 3;l++) {
	error += (Tr[i][l][k][k]-Tr[i][l][k][k-1])*(Tr[i][l][k][k]-Tr[i][l][k][k-1]);
	error += (Tv[i][l][k][k]-Tv[i][l][k][k-1])*(Tv[i][l][k][k]-Tv[i][l][k][k-1]);
      }
    }
  } else {
    error = 1.0;
  }
  
  return error;
}

inline
void rij(double r[][3], double * mass, double * eta, int i, int j, double ro[3]) {
  double sgn = (i < j ? 1 : -1),tem;
  register int k;


  if (i > j) {
    tem = i;
    i = j;
    j = tem;
  }
  
  if (i == 0) { 
    ro[0] = -sgn*r[j][0];
    ro[1] = -sgn*r[j][1];
    ro[2] = -sgn*r[j][2];
    for (k = 1; k <= j-1; k++) {
      ro[0] += -sgn*mass[k]*(1/eta[k])*r[k][0];
      ro[1] += -sgn*mass[k]*(1/eta[k])*r[k][1];
      ro[2] += -sgn*mass[k]*(1/eta[k])*r[k][2];
    }
    return;
  }

  ro[0] = sgn*(eta[i-1]*(1/eta[i]))*r[i][0];
  ro[1] = sgn*(eta[i-1]*(1/eta[i]))*r[i][1];
  ro[2] = sgn*(eta[i-1]*(1/eta[i]))*r[i][2];

  ro[0] += -sgn*r[j][0];
  ro[1] += -sgn*r[j][1];
  ro[2] += -sgn*r[j][2];

  for (k = i+1; k<=j-1; k++) {
 
    ro[0] += -sgn*(mass[k]*(1/eta[k]))*r[k][0];
    ro[1] += -sgn*(mass[k]*(1/eta[k]))*r[k][1];
    ro[2] += -sgn*(mass[k]*(1/eta[k]))*r[k][2];
  }
}

// 1/sqrt(temp)

inline
double norm3(double r[3]) {
  register double temp = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
  return sqrt(temp)*temp;

}

void rhs(double r[][3], double v[][3], double ro[][3], double vo[][3],double * mass, double * eta, int N)
{

  register int i,j,k,m;
  double rr[N][N][3],rp[3],temp,temp2,temp3;


  for (j = 0; j < N; j++) {
    for (i = 0; i < j; i++) {
      rij(r,mass,eta,i,j,rp);
      temp = norm3(rp);

      // rp[0]*temp^3
      rr[i][j][0] = (1/temp)*rp[0];
      rr[i][j][1] = (1/temp)*rp[1];
      rr[i][j][2] = (1/temp)*rp[2];
    }
  }
    
  A3(ro[0],v[0]);
  vo[0][0] = 0; vo[0][1] = 0; vo[0][2] = 0;

  for (i=1; i < N; i++) {
    temp = (eta[i]*(1/eta[i-1]));
    temp3 = 1/eta[i-1];

    ro[i][0] = v[i][0];//A3(ro[i],v[i]);
    vo[i][0] = (temp*mass[0])*rr[0][i][0];//A3(vo[i],(eta[i]/eta[i-1]*mass[0])*rr[0][i]);
    ro[i][1] = v[i][1];//A3(ro[i],v[i]);
    vo[i][1] = (temp*mass[0])*rr[0][i][1];
    ro[i][2] = v[i][2];//A3(ro[i],v[i]);
    vo[i][2] = (temp*mass[0])*rr[0][i][2];
    
    for (j = 1; j <= i-1; j++) {
      vo[i][0] += (temp*mass[j])*rr[j][i][0];//A3P(vo[i],temp*mass[j]*rr[j][i]);
      vo[i][1] += (temp*mass[j])*rr[j][i][1];
      vo[i][2] += (temp*mass[j])*rr[j][i][2];
    }
    
    for (j=i+1; j<=N-1; j++) {
      vo[i][0] += -(mass[j])*(rr[i][j][0]); //A3P(vo[i],-mass[j]*rr[i][j]);
      vo[i][1] += -(mass[j])*(rr[i][j][1]);
      vo[i][2] += -(mass[j])*(rr[i][j][2]);
    }
    
    for (j=0; j <= i-1; j++) {
      for (k=i+1; k <= N-1; k++) {
	vo[i][0] += (mass[k]*mass[j]*temp3)*(rr[j][k][0]);//A3P(vo[i],mass[j]*mass[k]*temp*rr[j][k]);
	vo[i][1] += (mass[k]*mass[j]*temp3)*(rr[j][k][1]);
	vo[i][2] += (mass[k]*mass[j]*temp3)*(rr[j][k][2]);
      }
    }
  } 
}

void mod_mid_3(double r[][3],double v[][3], int N,  int n, double H,double ro[][3],double vo[][3], double * mass, double * eta) {
  double zp_r[N][3],zm_r[N][3],zc_r[N][3], h = H/((double)n),twoh = 2.0*h;
  double zp_v[N][3],zm_v[N][3],zc_v[N][3];

  COPY(r,zp_r,N);
  COPY(zp_r,zc_r,N);
  COPY(zc_r,zm_r,N);

  COPY(v,zp_v,N);
  COPY(zp_v,zc_v,N);
  COPY(zc_v,zm_v,N);

  for (int m = 0; m < n; m++) {
    COPY(zp_r,zm_r,N);
    COPY(zc_r,zp_r,N);
   
    COPY(zp_v,zm_v,N);
    COPY(zc_v,zp_v,N);
    
    rhs(zp_r,zp_v,zc_r,zc_v,mass,eta,N);
     
    if (m == 0) { 
      MM3_M0(zc_r,zp_r,h,N);
      MM3_M0(zc_v,zp_v,h,N);  
      
    } else {
      MM3_M0(zc_r,zm_r,twoh,N);
      MM3_M0(zc_v,zm_v,twoh,N);
    }
  }
  rhs(zc_r,zc_v,zm_r,zm_v,mass,eta,N);
  
  MM3_RES(ro,zp_r,zc_r,zm_r,h,N);
  MM3_RES(vo,zp_v,zc_v,zm_v,h,N);
}
  
void evolve(double ** r, double ** v, double ** a,double * mass, double * eta, int N,
	    double time0, double time, double HMAX, int & status, double ORBIT_ERROR, double HLIMIT) {

  double ru[N][3],ru_w[N][3],ru_k[N][3],vu[N][3],vu_w[N][3],vu_k[N][3];
  double Tr[N][3][KMAX][KMAX],Tv[N][3][KMAX][KMAX],tcurr, H, error;
  int k;

  
  COPY(r,ru,N);
  COPY(v,vu,N);

  tcurr = time0;
  
  H = (fabs(time-tcurr) < HMAX ? time-tcurr : (time-tcurr < 0 ? -1: 1)*HMAX);
    
  while (H != 0) {
    
    COPY(ru,ru_w,N);
    COPY(vu,vu_w,N);
    
    k = 0;
    error = 1.0;
    while (error > ORBIT_ERROR && k < KMAX) {
	
      mod_mid_3(ru_w,vu_w,N,seqBS(k),H,ru_k,vu_k,mass,eta);

      
      COPY_T(ru_k,Tr,N,k,0);
      COPY_T(vu_k,Tv,N,k,0);

      error = neville(Tr,Tv,N,k);

      k++;
    }
      
    if ( k == KMAX) { 
      H /= FRAC_IMP;
      if (fabs(H) < HLIMIT) { 
	status = -1; 
	return;}
    } else {
      tcurr += H;
      H = (fabs(time-tcurr) < HMAX ? time-tcurr : (time-tcurr < 0 ? -1: 1)*HMAX);
	
      T_COPY(Tr,ru_w,N,k-1,k-1);
      T_COPY(Tv,vu_w,N,k-1,k-1);
      COPY(ru_w,ru,N);
      COPY(vu_w,vu,N);
    } 
  }

  //rhs(ru,vu,ru_w,vu_w,mass,eta,N);
    
  COPY(ru,r,N);
  COPY(vu,v,N);
  COPY(vu_w,a,N);
}
