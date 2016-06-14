#if !defined(__bodies_h)
#define __bodies_h

#include "../utils/utils.h"

/*
  Examples of implicitely defined closed and connected surfaces in 3D 
  using standard Euclidian coordinate system (x,y,z).
   
  The gradient should be outward from the surface, but it is not
  an requirement.

  Author: Martin Horvat, March, April 2016
*/

/*
  Torus
  
  Defined implicitly by the constrain and the gradient of it
  
    (x^2 + y^2 + z^2 + R^2 - A^2)^2 - 4 R^2 (x^2 + y^2) = 0
  
  Explicitely it is defined as
    
    [x,y,z] = R_z(phi_1) [R + A cos(phi_2), 0, A cos(phi_2) ]
  
    R_z(phi)= [cos(phi), -sin(phi), 0; sin(phi), cos(phi), 0; 0, 0, 1]

*/ 

template <class T>
struct Ttorus {
  
  T R, A;
  
  /*
    Reading and storing the parameters
    params [0] = R
    params [1] = A
  */
  
  Ttorus(void *params){
    T *p = (T*)params;
    R = p[0];
    A = p[1];
  }
  
  /* 
    Definition of the constrain and the gradient of it
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[4]:
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value 

  */
  void grad(T r[3], T ret[4]){
    
    T r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2],
      R2 = R*R,
      A2 = A*A,
      f1 = - A2 - R2 + r2,
      f2 = - A2 + R2 + r2;
    
      ret[0] = 4*r[0]*f1;
      ret[1] = 4*r[1]*f1;
      ret[2] = 4*r[2]*f2;
          
      ret[3] = -4*R2*(r[0]*r[0] + r[1]*r[1]) + f2*f2;
  }
  
  /*
    Initial point
  */ 
  void init(T r[3], T n[3]){
     
    r[0] = R + A;
    r[1] = r[2] = 0;
    
    n[0] = 8*A*R*(A + R);
    n[1] = n[2] = 0;
  } 
};


/*
  Sphere
  
  Defined of implicitly by a constrain

    x^2 + y^2 + z^2 - R^2 = 0
*/ 

template <class T>
struct Tsphere {
  
  double R;
  
  /*
    Reading and storing the parameters
    params [0] = R  -- radius
  */
  
  Tsphere(void *params){ 
    T *p = (T*) params;
    R = *p; 
  }
  
  /* 
    Definition of the constrain and the gradient of it
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[4]:
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value 

  */
  void grad(T r[3], T ret[4]){
    
    for (int i = 0; i < 3; ++i) ret[i] = 2*r[i];
     
    ret[3] = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - R*R;
  }

  /*
    Initial point
  */   
  void init(T r[3], T n[3]){
    r[0] = R;
    r[1] = r[2] = 0;
    
    n[0] = 2*R;
    n[1] = n[2] = 0;
  } 
};


/*
   Heart
   
   Defined implicitly
    
   (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3 = 0
   
   The normal vector field has singularity at (0,0,+-1).
*/ 

template <class T>
struct Theart {
  
  /*
    Reading and storing the parameters
  */
  
  Theart(void *params){ }
  
  /* Definition of constrain and the gradient of it
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      choice = 0: ret[4]
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value 
   
      choice = 1: ret[3]      
        {ret[0], ret[1], ret[2]} = grad-potential

  */
  void grad(T r[3], T ret[4]){
    
    T r02 = r[0]*r[0], 
      r12 = r[1]*r[1],
      r22 = r[2]*r[2],
      r23 = r22*r[2], 
      t = r02 + 9*r12/4 + r22 - 1,
      a = 3*t*t,
      b = t*t*t;
    
    ret[0] = 2*r[0]*(a - r23);
    ret[1] = 9*r[1]*(a - r23/20)/2;
    ret[2] = r[2]*(2*a - 3*(r02 + 9*r12/80)*r[2]);
    ret[3] = b - (r02 + 9*r12/80)*r23;
  }
  
  /*
    Initial point
  */
  void init(T r[3], T n[3]){

    r[0] = r[1] = 0;
    r[2] = -1;
    
    n[0] = n[1] = 0;
    n[2] = -1;
  } 
};

/*
   Generalized Rcohe/Kopal potential
      
    Omega = 
      1/rho 
      + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
      + 1/2 F^2(1 + q) rho^2 (1 - nu^2)
      
      rho = sqrt(x^2 + y^2 + z^2)
      
      x = rho lambda  lambda = sin(theta) cos(phi)
      y = rho mu      mu = sin(theta) sin(phi)
      z = rho nu      nu = cos(theta)
      
    implicitly defining Roche lobe
    
      F =  Omega0 - Omega(x,y,z)  == 0
      
    The initial point is on the x-axis in the point (x0,0,0).
  
   params = {q, F, delta, Omega0, x0)
*/ 

template <class T>
struct Tgen_roche {
  
  T q, F, delta, Omega0, x0, 
    b, f0; // derived constants
  
  
  Tgen_roche() {}
  
  /*
    Reading and storing the parameters
  */
  
  Tgen_roche(void *params, bool init_param = true){ 
    
    T *p = (T*) params;
    
    q = p[0];
    F = p[1];  
    delta = p[2];
    Omega0 = p[3];
    if (init_param) x0 = p[4];
    
    b = (1 + q)*F*F; 
    f0 = 1/(delta*delta);
  }
  
  /*
    Definition of the potential minus the reference and the 
    gradient of it.
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[4]:
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value 
  
  */
  #if 0
  void grad(T r[3], T ret[4]){
    
    T x1 = r[0], 
      x2 = r[0] - delta, 
      y = r[1], 
      z = r[2], 
      s = y*y + z*z,
      r12 = 1/(x1*x1 + s),
      r22 = 1/(x2*x2 + s),
      r1 = std::sqrt(r12), 
      r2 = std::sqrt(r22), 
      f1 = r1*r12,
      f2 = r2*r22;
    
    ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
    ret[1] = y*(f1 + q*f2 - b);
    ret[2] = z*(f1 + q*f2);
    ret[3] = Omega0 - (r1 + q*(r2 - f0*x1) + b*(x1*x1 + y*y)/2);
  }
  #else
  //
  // Slower and preciser version due to using hypot function
  //
  void grad(T r[3], T ret[4]){
    
    T x1 = r[0], 
      x2 = r[0] - delta, 
      y = r[1], 
      z = r[2], 
      r1 = 1/utils::hypot3(x1, y, z), 
      r2 = 1/utils::hypot3(x2, y, z), 
      s = y*y + z*z,
      f1 = r1/(s + x1*x1),
      f2 = r2/(s + x2*x2);
    
    ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
    ret[1] = y*(f1 + q*f2 - b);
    ret[2] = z*(f1 + q*f2);
    ret[3] = Omega0 - (r1 + q*(r2 - f0*x1) + b*(x1*x1 + y*y)/2);
  }
  
  //
  // Slower and preciser version due to using hypot function
  //
  void grad_only(T r[3], T ret[3]){
    
    T x1 = r[0], 
      x2 = r[0] - delta, 
      y = r[1], 
      z = r[2], 
      r1 = 1/utils::hypot3(x1, y, z), 
      r2 = 1/utils::hypot3(x2, y, z), 
      s = y*y + z*z,
      f1 = r1/(s + x1*x1),
      f2 = r2/(s + x2*x2);
    
    ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
    ret[1] = y*(f1 + q*f2 - b);
    ret[2] = z*(f1 + q*f2);
  }
  #endif
  
  /*
    Initial point: it is on the x-axis
  */ 
  void init(T r[3], T n[3]){
    
    T x = r[0] = x0;
    r[1] = r[2] = 0;
    
    
    T x1 = x0 - delta;
    
    n[0] = - x*b 
           + (x > 0 ? 1/(x*x) : (x < 0 ? -1/(x*x) : 0)) 
           + q*(f0 + (x1 > 0 ? 1/(x1*x1) : (x1 < 0 ? -1/(x1*x1) : 0))); 
   
    n[1] = n[2] = 0;
  } 

};

#endif // #if !defined(__bodies_h)
