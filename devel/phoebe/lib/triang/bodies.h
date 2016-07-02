#if !defined(__bodies_h)
#define __bodies_h

#include "../utils/utils.h"

/*
  Examples of implicitely defined closed and connected surfaces in 3D 
  using standard Euclidian coordinate system (x,y,z).
   
  The gradient should be outward from the surface, but it is not
  an requirement.
  
  Supported models:
  
  * Torus -- Ttorus <-- not simply connected, trouble for simple marching
  * Heart -- Theart <-- not entirely smooth, trouble for simple marching
  * Generalized Roche -- Tgen_roche
  * Rotating star -- Trotating_star
  * 
  Author: Martin Horvat, March, April, June 2016
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
  
  Tsphere(void *params, bool init_params =  true){ 
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
   Generalized Roche/Kopal potential
      
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
  
  Tgen_roche(void *params, bool init_param = true) 
  : q(((T*)params)[0]),
    F(((T*)params)[1]),
    delta(((T*)params)[2]),
    Omega0(((T*)params)[3])
  { 
    
    if (init_param) x0 = ((T*)params)[4];
    
    b = (1 + q)*F*F; 
    f0 = 1/(delta*delta);
  }
  
  /*
    Definition of the potential minus the reference.
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
      Omega0 - Omega(x,y,z)
  */
  
  T constrain(T r[3]) {
    return Omega0 - (
      1/utils::hypot3(r[0], r[1], r[2]) + 
      q*(1/utils::hypot3(r[0] - delta, r[1], r[2]) - f0*r[0]) + b*(r[0]*r[0] + r[1]*r[1])/2
    ); 
  }
  
  /*
    Definition of the potential minus the reference and the 
    gradient of it:
      
      -grad-potential Omega
    
    Minus guaranties that the normal points outward from the 
    iso-potential surfaces. 
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega 
  */

  
  void grad(T r[3], T ret[4]){
    
   T  x1 = r[0], 
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
    
  /*
    Definition of the gradient of the negative potential
     
      -grad-potential Omega
    
    Minus guaranties that the normal points outward from the 
    iso-potential surfaces. 
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */
  
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
  
  /*
    Initial point: it is on the x-axis
  */ 
  void init(T r[3], T g[3]){
    
    T x = r[0] = x0;
    r[1] = r[2] = 0;
    
    
    T x1 = x0 - delta;
    
    g[0] = - x*b 
           + (x > 0 ? 1/(x*x) : (x < 0 ? -1/(x*x) : 0)) 
           + q*(f0 + (x1 > 0 ? 1/(x1*x1) : (x1 < 0 ? -1/(x1*x1) : 0))); 
   
    g[1] = g[2] = 0;
  } 

  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega
    
  */
  void hessian (T r[3], T H[3][3]){
    
    T x1 = r[0], x2 = r[0] - delta, y = r[1], z = r[2], 
      x12 = x1*x1,
      x22 = x2*x2,
      y2 = y*y, 
      z2 = z*z,
    
      f11 = 1/utils::hypot3(x1, y, z), 
      f21 = 1/utils::hypot3(x2, y, z), 
      f12 = 1/(y2 + z2 + x12), 
      f22 = 1/(y2 + z2 + x22),
      
      f13 = f12*f11, 
      f23 = f22*f21,
      f15 = f12*f13,
      f25 = f22*f23;
      
    H[0][0] = -b + f13*(1 + q) - 3*(f15*x12 + f25*q*x22);
    H[0][1] = H[1][0] = -3*(f15*x1 + f25*q*x2)*y;
    H[0][2] = H[2][0] = -3*(f15*x1 + f25*q*x2)*z;
    H[1][1] = -b + f13*(1 + q) - 3*(f15 + f25*q)*y2;
    H[1][2] = H[2][1] = -3*(f15 + f25*q)*y*z;
    H[2][2] = -f13*(1 + q) - 3*(f15 + f25*q)*z2;
  }


  /* 
    Find the point the point on the horizon around individual lobes.
  
    Input:
      view - direction of the view
      choice - searching  
        0 - left lobe -- around (0, 0, 0)
        1 - right lobe -- around (delta, 0, 0)
        2 - overcontact ??
      max_iter - maximal number of iteration in search algorithm
    
    Output:
      p - point on the horizon
  */
  
  bool point_on_horizon(
    T r[3], 
    T view[3], 
    int choice = 0,
    int max_iter = 1000){
   
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Starting points (TODO: need to optimize its choice)
    //
    
    if (choice == 0) {
     r[0] = 0; 
     r[1] = 0;
     r[2] = 1e-6;
    } else if (choice == 1) {
     r[0] = delta; 
     r[1] = 0; 
     r[2] = 1e-6;  
    } else {
      std::cerr 
        << "point_on_horizon:: choices != 0,1 not supported yet\n";
      return false;
    }
    
    // Solving both constrains at the same time
    //  Omega_0 - Omega(r) = 0
    //  grad(Omega) n = 0
    
    int i, it = 0;
    
    T dr_max, r_max, t, f, H[3][3], 
      A[2][2], a[4], b[3], u[2], x[2];
    
    do {

      // a = {grad constrain, constrain}   
      grad(r, a);
      
      // get the hessian on the constrain
      hessian(r, H);
      
      utils::dot3D(H, view, b);
      
      // define the matrix of direction that constrains change
      A[0][0] = utils::dot3D(a,a);
      A[0][1] = A[1][0] = utils::dot3D(a,b);
      A[1][1] = utils::dot3D(b,b);
      
      // negative remainder in that directions
      u[0] = -a[3];
      u[1] = -utils::dot3D(a, view);
      
      // solving 2x2 system: 
      //  A x = u
      // and  
      //  making shifts 
      //  calculating sizes for stopping criteria 
      //
      
      dr_max = r_max = 0;
    
      if (utils::inverse2D(A, u, x)){ 
        
        //shift along the directions that the constrains change
        for (i = 0; i < 3; ++i) {
          r[i] += (t = x[0]*a[i] + x[1]*b[i]);
          
          // max of dr, max of r
          if ((t = std::abs(t)) > dr_max) dr_max = t;
          if ((t = std::abs(r[i])) > r_max) r_max = t;
        }
        
      } else {
        
        //alterantive path in direction of grad(Omega)
        f = u[0]/(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        
        for (i = 0; i < 3; ++i) {
          r[i] += (t = f*a[i]); 
          
          // max of dr, max of r
          if ((t = std::abs(t)) > dr_max) dr_max = t;
          if ((t = std::abs(r[i])) > r_max) r_max = t;
        }
      }
      
      //std::cout << "P----\n";
      //std::cout << r[0] << '\t' << r[1] << '\t' << r[2] << '\n';
      //std::cout << it << '\t' << dr_max << '\n';
      
    } while (dr_max > eps*r_max + min && ++it < max_iter);
    
    
    return (it < max_iter);
  }

};

/* 
  Rotating star
  
  Defined of implicitly by a constrain

    1/r + 1/2 omega^2 (x^2 + y^2) = Omega_0
*/ 

template <class T>
struct Trot_star {
  
  T omega, Omega0, w2;
  
  /*
    Reading and storing the parameters
    params[0] = omega0  
    params[1] = Omega0
    
  */
  
  Trot_star(void *params, bool init_param = true)
  : omega(((T*)params)[0]), Omega0(((T*)params)[1]) { 
    
    w2 = omega*omega;
  }
  
  /*
    Definition of the potential minus the reference.
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
      Omega0 - Omega(x,y,z)
  */
  
  T constrain(T r[3]) {
    return 
      Omega0 - 
      (1/utils::hypot3(r[0], r[1], r[2]) + w2*(r[0]*r[0] + r[1]*r[1])/2); 
  }
  /*
    Definition of the potential minus the reference and the 
    gradient of it:
      
      -grad-potential Omega
    
    Minus guaranties that the normal points outward from the 
    iso-potential surfaces. 
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega 
  */

  
  void grad(T r[3], T ret[4]){
    
    T x = r[0], 
      y = r[1],
      z = r[2],
      f = 1/utils::hypot3(x, y, z),
      r1 = std::pow(f, 3);
      
    ret[0] = (-w2 + r1)*x; 
    ret[1] = (-w2 + r1)*y;
    ret[2] = z*r1;
    ret[3] = Omega0 - (f + w2*(x*x + y*y)/2);
  }
    
  /*
    Definition of the gradient of the negative potential
     
      -grad-potential Omega
    
    Minus guaranties that the normal points outward from the 
    iso-potential surfaces. 
    
    Input:
      r[3] = {x, y, z}
      
    Output: 
    
      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */
  
  void grad_only(T r[3], T ret[3]){
    
    T x = r[0], 
      y = r[1],
      z = r[2],
      f = 1/utils::hypot3(x, y, z),
      r1 = std::pow(f, 3);
      
    ret[0] = (-w2 + r1)*x; 
    ret[1] = (-w2 + r1)*y;
    ret[2] = z*r1;
  }
  
  /*
    Initial point: it is on the z-axis, z>0
  */ 
  void init(T r[3], T g[3]){
    
    r[0] = r[1] = 0;
    r[2] = 1/Omega0;
    
    g[0] = g[1] = 0;
    g[2] = Omega0*Omega0;
  }
  
  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega
    
  */
  void hessian (T r[3], T H[3][3]){
    
    T x = r[0], y = r[1], z = r[2], 
      x2 = x*x,
      y2 = y*y, 
      z2 = z*z,
    
      f = 1/utils::hypot3(x, y, z), 
      f2 = 1/(y2 + z2 + x2), 
      
      f3 = f2*f, 
      f5 = f2*f3;
      
    H[0][0] = f3 - w2 - 3*f5*x2;
    H[0][1] = H[1][0] = -3*f5*x*y;
    H[0][2] = H[2][0] = -3*f5*x*z;
    H[1][1] = f3 - w2 - 3*f5*y2;
    H[1][2] = H[2][1] = -3*f5*y*z;
    H[2][2] = f3 - 3*f5*z2;
  }
  
};


#endif // #if !defined(__bodies_h)
