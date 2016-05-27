#ifndef _N_BODY_STATE_
#define _N_BODY_STATE_

/*
  n_body_state.h

  Josh Carter, 2013

  Class defining NBodyState object.  Calls evolve described in n_body.h and coded in n_body.cpp  

*/

class NBodyState{
 private:
  double * mass;
  double * eta;
  double ** rj; //check on this (jacobian)
  double ** vj; 
  double ** aj;
  double ** rb; // barycentric
  double ** vb; 
  double ** rb_lt; // barycentric, lite-time corrected (observer at positive z)
  double time;
  int status;
  int N;
  
  void calcEta();
  
  void bary_coords(); 
  double bary_coords_lt(); 
  void calc_jac(double * a,double * e,double * in,double * o,double * ln,double * m); 

 public:
  // Constructor 1: initialize NBodyState with Jacobian coordinates
  NBodyState(double * m, double posj[][3], double velj[][3], int NN, double t0);
  // Constructor 2: initialize NBodyState with osculating elements for Jacobian coordinates.
  NBodyState(double * ms, double * a, double * e, double * in, double * o,double * ln, double * m, int NN, double t0);
 
  // Evolution overloaded operator.  t is time to evolve to, H is step size in BS integrator, ORBIT_ERROR is
  // orbit error tolerance and HLIMIT is minimum step size.
  // Example, for initialized state NBodyState state;
  //
  // state(100,1e-8,1e-16,1e-9); //evolves state to t = 0.
  double operator() (double t, double H, double ORBIT_ERROR, double HLIMIT);

  // Cruise operator moves according to velocity and acceleration and time t.  Returns new NBodyState.
  NBodyState * cruise(double t); //v,a time not reset

  // Returns current osculating keplerian elements for state
  void kep_elements(double * mj, double * a, double * e, double * in, double * o,double * ln, double * m); // return instantaneous keplerian elements

  // Returns mass of object obj
  double getMass(int obj);

  // Gets current time
  double getTime();

  // Gets the number of bodies in the state.
  int getN();

  // Returns NX3 matrix of positions for N objects corrected for the finite speed of light.
  double ** getBaryLT();

  // Returns barycentric coordinates for object obj.
  double X_B(int obj);
  double Y_B(int obj);
  double Z_B(int obj);

  // Returns barycentric velocities for object obj.
  double V_X_B(int obj);
  double V_Y_B(int obj);
  double V_Z_B(int obj);

  // Returns Jacobian coordinates for object obj.
  double X_J(int obj);
  double Y_J(int obj);
  double Z_J(int obj);

  // Returns Jacobian velocities for object obj.
  double V_X_J(int obj);
  double V_Y_J(int obj);
  double V_Z_J(int obj);

  // Returns barycentric, light-time corrected coordinates for object obj.
  double X_LT(int obj);
  double Y_LT(int obj);
  double Z_LT(int obj);

  // Returns barycentric, light-time corrected velocities for object obj.
  double V_X_LT(int obj);
  double V_Y_LT(int obj);
  double V_Z_LT(int obj);

  // Returns energy
  double getE();

  // Returns components of total angular momentum
  void getL(double * lx, double * ly, double * lz);
  
  // Destroys the state.
  ~NBodyState();

};

#endif
