#if !defined(__misaligned_roche_h)
#define __misaligned_roche_h

/*
  Library dealing with the generalizd Roche potential with misaligned 
  binary system.

  THERE IS NO SUPPORT FOR (OVER-)CONTACT CASE, AS IT IS NOT PHYSICAL.
  STILL NOTATION OF TWO LOBES REMAINS.
  
  The surface is defined implicitely with potential
  
    Omega(x,y,z,params) = 
      1/r1 + q(1/r2 - x/delta^2) + 
      1/2 (1 + q) F^2 [(x cos theta' - z sin theta')^2 + y^2]
      
    r1 = sqrt(x^2 + y^2 + z^2)
    r2 = sqrt((x-delta)^2 + y^2 + z^2)
    
  and constrain
  
    Omega(x,y,z,params) == Omega0
 
  
  Parameters:
    q - mass ratio M2/M1 > 0
    F - synchronicity parameter in R
    delta - separation between the two objects > 0
    theta' - angle between z axis in spin of the object in [0, pi]
             spin in plane (x, z)
  
  Author: Martin Horvat, December 2016 
*/

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

// General rotines
#include "utils.h"                  // Misc routines (sqr, solving poly eq,..)

// Definition of bodies
#include "bodies.h"

#include "gen_roche.h"

namespace misaligned_roche {
  
  /*
    Calculating the pole height of the Roche is positive (sign = 1) or 
    negative direction  (sign = -1) of the axis of rotation.
   
    Solving
    
      1/tp + q (1/sqrt(tp^2 - 2*ss*tp + 1) - ss*tp) = delta Omega
      
      for tp > 0, ss = sign*s, s = sin(theta) in [-1,1]
      
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      sintheta - sinus of the angle of spin w.r.t. z axis in rotated system 
              
    Return:
      height = delta*tp
  */
  
  template <class T>
  T poleL_height(
    const T & Omega0,
    const T & q,
    const T & F,
    const T & delta,
    const T & sintheta, 
    const T & sign = 1
  ) {
    
    if (sintheta == 0)
      return gen_roche::poleL(Omega0, q, F, delta);

    if (Omega0 < 0 || q < 0)  return -1;
   
    T w = Omega0*delta, 
      s = sintheta, 
      t;
    
    if (sign < 0) s = -s;
      
    // calculate the estimate of the pole (in direction of the spin)
    // note: there is no symmetry across the equator 
    if (w >= 10 && w > q) {  
      t = 1/w;
      t *= 1 + q*t*(1 + t*(q + (-1 + 2*q*q + 3*s*s)*t/2));
    } else if (q > 10 && q > w) {
      t = (std::sqrt(w*w + 4*(1+q)*s*q) - w)/(2*q*s); 
    } else { 
      // using RK4 integration to get an estimate of r-pole      
      int n = 20;
      
      T du = 1.0/(n*w), t1, r, r2, r3, k[4];
      
      t = 0;
      
      do {
        t1 = t, r2 = 1 - (2*s - t1)*t1, r = std::sqrt(r2), r3 = r*r2;
        k[0] = du*(r*utils::sqr(r + q*t1*(1 - r*s*t1)))/(r3 + q*t1*t1*(s*(r3 - 1) + t1));
       
        t1 = t + 0.5*k[0], r2 = 1 - (2*s - t1)*t1, r = std::sqrt(r2), r3 = r*r2;
        k[1] = du*(r*utils::sqr(r + q*t1*(1 - r*s*t1)))/(r3 + q*t1*t1*(s*(r3 - 1) + t1));  
        
        t1 = t + 0.5*k[1], r2 = 1 - (2*s - t1)*t1, r = std::sqrt(r2), r3 = r*r2;
        k[2] = du*(r*utils::sqr(r + q*t1*(1 - r*s*t1)))/(r3 + q*t1*t1*(s*(r3 - 1) + t1));  
       
        t1 = t + k[2], r2 = 1 - (2*s - t1)*t1, r = std::sqrt(r2), r3 = r*r2;
        k[3] = du*(r*utils::sqr(r + q*t1*(1 - r*s*t1)))/(r3 + q*t1*t1*(s*(r3 - 1) + t1));  
         
        t += (k[0] + 2*(k[1] + k[2]) + k[3])/6;
        
      } while (--n);
    }
    
    // Newton-Raphson iteration based on polynomial
    const int iter_max = 100;
    const T eps = 4*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();
    
    int it = 0;
    
    T f, df;
    
    #if 0
    {
      const int n = 6;
      
      T s2 = s*s,
        q2 = q*q,
        a[7] = {
          1, 
          -2*(w + s), 
          1 + w*(w + 4*s) - (2*s + q)*q, 
          -2*(w*(1 + w*s) - s*q*(w + 2*s)), 
          w*w - q*s*(2 + (4*w - q)*s),
          2*q*s*(w - s2*q),
          q2*s2
        };
        
      do {
        f = a[n], df = 0;
        for (int i = n - 1; i >= 0; --i) { 
          df = f + t*df;
          f  = a[i] + t*f;
        }
    
        // Newton-Raphson step
        t -= (dt = f/df);
      } while (std::abs(dt) > eps*std::abs(t) + min && (++it) < iter_max);
    }
    #else
    // Newton-Raphson iteration based on original equation   
    {

      T dt, t2, tmp1, tmp2;
      
      do {
        t2 = t*t;
        tmp1 = 1 - 2*s*t + t2;
        tmp2 = std::sqrt(tmp1);
         
        f = 1/t + q*(1/tmp2 - s*t) - w;
        df = -1/t2 - q*(s + (t-s)/(tmp1*tmp2));
        
        // Newton-Raphson step
        t -= (dt = f/df);
      } while (std::abs(dt) > eps*std::abs(t) + min && (++it) < iter_max);
      
    }
    #endif
    
    return delta*t;
  }

  /*
    Calculating the pole height of the Roche is positive (sign = 1) or 
    negative direction  (sign = -1) of the axis of rotation.
   
    Solving
    
      1/tp + q (1/sqrt(1 - 2*sign*tp*sx + tp^2) - sx*sign*tp) = delta Omega
      
      for tp > 0.
      
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      s - direction of rotating axis = (sx, sy, sz) 
    
    Return:
      height = delta*tp
  */
 template <class T>
  T poleL_height(
    const T & Omega0,
    const T & q,
    const T & F ,
    const T & delta,
    T s[3], 
    const T & sign = 1
  ) {
    return poleL_height(Omega0, q, F, delta, s[0], sign);
  }

  /*
    Pole of the first star h = delta*tp in rotated coordinate system:
  
      Omega(tp*s, 0, tp*c) = Omega0
      
      s = sin(theta')             theta' in [-pi/2, pi/2]
      c = cos(theta')
      
      tp is height of the pole
      
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      theta - angle of spin w.r.t. z axis in rotated system
      
    Output:
      p[3] = (d tp s, 0, d*tp*c)
    
    Return: 
      true: if pole is not found, otherwise false
  */
  
  template <class T>
  bool poleL(
    T p[3], 
    const T & Omega0,
    const T & q,
    const T & F,
    const T & delta,
    const T & theta,
    const T & sign = 1 
  ) {
    
    T s, c;
    
    utils::sincos(theta, &s, &c);
    
    T tp = poleL_height(Omega0, q, F, delta, s, sign);
  
    if (tp > 0) {
    
      p[0] = s*tp*sign;
      p[1] = 0;
      p[2] = c*tp*sign;
      return true;
    }
    
    return false;
  }
  
    /*
    Pole of the first star h = delta*tp in canonical coordinate system:
  
      Omega(sign*delta*tp*s) = Omega0
      
      s = (sx,sy,sz)  tp > 0
       
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      s[3] - direction of spin w.r.t. z axis
      
    Output:
      p[3] = sign*delta*tp*s
    
    Return: 
      true: if pole is not found, otherwise false
  */
  
  template <class T>
  bool poleL(
    T p[3], 
    const T & Omega0,
    const T & q,
    const T & F,
    const T & delta,
    T s[3],
    const T & sign = 1 
  ) {
    
    T tp = poleL_height(Omega0, q, F, delta, s[0], sign);
  
    if (tp > 0) {
      tp *= sign;
      for (int i = 0; i < 3; ++i) p[i] = s[i]*tp;
      return true;
    }
    
    return false;
  }
  
  /* 
    Find the point on the horizon around individual lobes. Currently only
    primary lobe is supported, as this is physically only interesting case.
  
    Input:
      v - direction of the view
      choice :
        0 - left -- around (0, 0, 0)
        1 - right --  not supported
        2 - overcontact: not supported       
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      theta'- angle between z axis in spin of the object in [0, pi]
              spin in plane (x, z) 
      max_iter - maximal number of iteration in search algorithm
    
    Output:
      p - point on the horizon
    
    Return:
      true: if everything is OK
  */


  template<class T> 
  bool point_on_horizon(
    T r[3], 
    T v[3], 
    int choice,
    const T & Omega0, 
    const T & q, 
    const T & F = 1, 
    const T & delta = 1,
    const T & theta = 0,  
    int max_iter = 1000){
    
    if (theta == 0)
      return gen_roche::point_on_horizon(r, v, choice, Omega0, q, F, delta, max_iter);
    
    //
    // Starting points 
    //
      
    if (choice != 0) {
      std::cerr 
        << "point_on_horizon:: choices != 0 not supported yet\n";
      return false;
    }
   
    // estimate of the radius of sphere that is 
    // inside the lobe
    
    T fac = 0.5*poleL_height(Omega0, q, F, delta, std::sin(theta));
     
    // determine direction of initial point
    if (std::abs(v[0]) >= 0.5 || std::abs(v[1]) >= 0.5){
      fac /= std::hypot(v[0], v[1]);
      r[0] = fac*v[1];
      r[1] = -fac*v[0];
      r[2] = 0.0;
    } else {
      fac /= std::hypot(v[0], v[2]);
      r[0] = -fac*v[2];
      r[1] = 0.0;
      r[2] = fac*v[0];
    }

    //
    // Initialize body class
    //
    
    T params[] = {q, F, delta, theta, Omega0};
    
    Tmisaligned_rotated_roche<T> misaligned(params);
    
    // Solving both constrains at the same time
    //  Omega_0 - Omega(r) = 0
    //  grad(Omega) n = 0
    
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();
 
    int i, it = 0;
    
    T dr_max, r_max, t, f, H[3][3], 
      A[2][2], a[4], b[3], u[2], x[2];
    
    do {

      // a = {grad constrain, constrain}   
      misaligned.grad(r, a);
      
      // get the hessian on the constrain
      misaligned.hessian(r, H);
      
      utils::dot3D(H, v, b);
      
      // define the matrix of direction that constrains change
      A[0][0] = utils::dot3D(a, a);
      A[0][1] = A[1][0] = utils::dot3D(a, b);
      A[1][1] = utils::dot3D(b, b);
      
      // negative remainder in that directions
      u[0] = -a[3];
      u[1] = -utils::dot3D(a, v);
            
      // solving 2x2 system: 
      //  A x = u
      // and  
      //  making shifts 
      //  calculating sizes for stopping criteria 
      //
      
      dr_max = r_max = 0;
    
      if (utils::solve2D(A, u, x)){ 
        
        //shift along the directions that the constrains change
        for (i = 0; i < 3; ++i) {
          r[i] += (t = x[0]*a[i] + x[1]*b[i]);
          
          // max of dr, max of r
          if ((t = std::abs(t)) > dr_max) dr_max = t;
          if ((t = std::abs(r[i])) > r_max) r_max = t;
        }
        
      } else {
        
        //alternative path in direction of grad(Omega)
        f = u[0]/(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        
        for (i = 0; i < 3; ++i) {
          r[i] += (t = f*a[i]); 
          
          // max of dr, max of r
          if ((t = std::abs(t)) > dr_max) dr_max = t;
          if ((t = std::abs(r[i])) > r_max) r_max = t;
        }
      }
          
    } while (dr_max > eps*r_max + min && ++it < max_iter);
        
    return (it < max_iter);
  }

    /* 
    Find the point on the horizon around individual lobes. Currently only
    primary lobe is supported for theta != 0, as this is physically 
    only interesting case.
  
    Input:
      v - direction of the view
      choice :
        0 - left -- around (0, 0, 0)
        1 - right --  not supported
        2 - overcontact: not supported       
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      s - vector of the spin of the object |s| = 1
      max_iter - maximal number of iteration in search algorithm
    
    Output:
      p - point on the horizon
    
    Return:
      true: if everything is OK
  */

  
template<class T> 
  bool point_on_horizon(
    T r[3], 
    T v[3], 
    int choice,
    const T & Omega0, 
    const T & q, 
    const T & F, 
    const T & delta,
    T s[3],  
    int max_iter = 1000){
  
    // if no misalignment 
    if (std::abs(s[2]) == 1)
      return gen_roche::point_on_horizon(r, v, choice, Omega0, q, F, delta, max_iter);
  
    //
    // Starting points 
    //
      
    if (choice != 0) {
      std::cerr 
        << "point_on_horizon:: choices != 0 not supported yet\n";
      return false;
    }
   
    // estimate of the radius of sphere that is 
    // inside the lobe
    
    T fac = 0.5*poleL_height(Omega0, q, F, delta, s);
     
    // determine direction of initial point
    if (std::abs(v[0]) >= 0.5 || std::abs(v[1]) >= 0.5){
      fac /= std::hypot(v[0], v[1]);
      r[0] = fac*v[1];
      r[1] = -fac*v[0];
      r[2] = 0.0;
    } else {
      fac /= std::hypot(v[0], v[2]);
      r[0] = -fac*v[2];
      r[1] = 0.0;
      r[2] = fac*v[0];
    }

    //
    // Initialize body class
    //
    
    T params[] = {q, F, delta, s[0], s[1], s[2], Omega0};
    
    Tmisaligned_roche<T> misaligned(params);
    
    // Solving both constrains at the same time
    //  Omega_0 - Omega(r) = 0
    //  grad(Omega) n = 0
    
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();
 
    int i, it = 0;
    
    T dr_max, r_max, t, f, H[3][3], 
      A[2][2], a[4], b[3], u[2], x[2];
    
    do {

      // a = {grad constrain, constrain}   
      misaligned.grad(r, a);
      
      // get the hessian on the constrain
      misaligned.hessian(r, H);
      
      utils::dot3D(H, v, b);
      
      // define the matrix of direction that constrains change
      A[0][0] = utils::dot3D(a, a);
      A[0][1] = A[1][0] = utils::dot3D(a, b);
      A[1][1] = utils::dot3D(b, b);
      
      // negative remainder in that directions
      u[0] = -a[3];
      u[1] = -utils::dot3D(a, v);
            
      // solving 2x2 system: 
      //  A x = u
      // and  
      //  making shifts 
      //  calculating sizes for stopping criteria 
      //
      
      dr_max = r_max = 0;
    
      if (utils::solve2D(A, u, x)){ 
        
        //shift along the directions that the constrains change
        for (i = 0; i < 3; ++i) {
          r[i] += (t = x[0]*a[i] + x[1]*b[i]);
          
          // max of dr, max of r
          if ((t = std::abs(t)) > dr_max) dr_max = t;
          if ((t = std::abs(r[i])) > r_max) r_max = t;
        }
        
      } else {
        
        //alternative path in direction of grad(Omega)
        f = u[0]/(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        
        for (i = 0; i < 3; ++i) {
          r[i] += (t = f*a[i]); 
          
          // max of dr, max of r
          if ((t = std::abs(t)) > dr_max) dr_max = t;
          if ((t = std::abs(r[i])) > r_max) r_max = t;
        }
      }
          
    } while (dr_max > eps*r_max + min && ++it < max_iter);
        
    return (it < max_iter);
  }
    
  /*
    Starting point for meshing the misaligned Roche lobe. Currently only
    primary lobe is supported for theta != 0, as this is physically 
    only interesting case.
    
    Input:
      choice :
        0 - left
        1 - right
        2 - overcontact        
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      theta' - angle between z axis in spin of the object in [0, pi]
              spin in plane (x, z) 

    Output:
       r - position
       g - gradient
  */
  
  template <class T>
  bool meshing_start_point(
    T r[3], 
    T g[3],
    int choice,
    const T & Omega0, 
    const T & q, 
    const T & F = 1,
    const T & delta = 1,
    const T & theta = 0
  ){
    
    if (theta == 0)
      return gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, delta);
      
    // only primary lobes for theta != 0 are supported
    if (choice != 0) {
      std::cerr 
        << "meshing_start_point:: choices != 0 not supported yet\n";
      return false;
    }
   
    // for cases which are not strongy deformed the pole is also 
    // good point to starts
    
    if (!poleL(r, Omega0, q, F, delta, theta)) return false;
  
    // calculating the gradient
    T params[] = {q, F, delta, theta, Omega0};
    
    Tmisaligned_rotated_roche<T> misaligned(params);
    
    misaligned.grad_only(r, g);
  
    return true;
  } 
  
  /*
    Starting point for meshing the misaligned Roche lobe. Currently only
    primary lobe is supported for theta != 0, as this is physically 
    only interesting case.
    
    Input:
      choice :
        0 - left
        1 - right
        2 - overcontact        
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      s - axis of rotation |s| = 1 

    Output:
       r - position
       g - gradient
  */
  
  template <class T>
  bool meshing_start_point(
    T r[3], 
    T g[3],
    int choice,
    const T & Omega0, 
    const T & q, 
    const T & F,
    const T & delta,
    T s[3]
  ){
    
    if (std::abs(s[2]) == 1)
      return gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, delta);
    
    // only primary lobes for theta != 0 are supported
    if (choice != 0) {
      std::cerr 
        << "meshing_start_point:: choices != 0 not supported yet\n";
      return false;
    }
   
    // for cases which are not strongy deformed the pole is also 
    // good point to start
    
    if (!poleL(r, Omega0, q, F, delta, s)) return false;
  
    // calculating the gradient
    T params[] = {q, F, delta, s[0], s[1], s[2], Omega0};
    
    Tmisaligned_roche<T> misaligned(params);
    
    misaligned.grad_only(r, g);
  
    return true;
  } 
  
  /*
    Calculate the partical differentials of the Omega w.r.t. to spherical 
    coordinates r, theta, phi:
    
    W = 1/r 
        - q r(Cos[theta] Sin[th] - Cos[phi] Cos[th] Sin[theta]) 
        + 1/2 b r^2 Sin[theta]^2 
        + q/Sqrt[1 + r^2 - 2 r (Cos[theta] Sin[th] - Cos[phi] Cos[th] Sin[theta])]
    
    Input:
      r - radial distance 
      theta - azimuthal angle 
      phi - polar angle
      q - mass ratio M2/M1
      b = F^2 d^3 - 
      th - angle between z axis in spin of the object in [0, pi]
           spin in plane (x, z) 
    Output:
      W[3] = {Wr, Wtheta, Wphi}
      
  */ 
  template <class T> void calc_dOmega(
    unsigned mask,
    T W[3],
    const T &r,
    const T &theta,
    const T &phi,
    const T &q,
    const T &b,
    const T &th
  ){
    
    T st, ct, 
      sp, cp,
      s1, c1, 
      r2 = r*r;
    
    utils::sincos(theta, &st, &ct);
    utils::sincos(phi, &sp, &cp);
    utils::sincos(th, &s1, &c1);
    
    T t1 = c1*cp*st - ct*s1, 
      t2 = c1*cp*ct + st*s1, 
      t3 = std::pow(1 + 2*r*t1 + r2, -1.5);
      
    // partial_r W
    if ((mask & 1U) == 1U) 
      W[0] = -1/r2 + b*r*st*st + q*(t1 - t3*(r + t1));
        
    // partial_theta W
    if ((mask & 2U) == 2U) 
      W[1] = r*(r*b*ct*st - q*t2*(-1 + t3));
   
    // partial_phi W
    if ((mask & 4U) == 4U) 
      W[2] = q*r*c1*sp*st*(-1 + t3);
  
  }
  
  //
  //  Gauss-Lagrange quadrature (GLQ) in [0,Pi]
  //
  
  #if 1
  const int glq_n = 10; 

  // Gauss-Lagrange nodes x_i in [0, Pi] 
  const double glq_phi[]={0.04098752915855404645614279849,0.21195796795501301950049770970,0.50358227252148264361980251927,0.890020433646848629194248947918,1.336945061968532022939709383344,1.804647591621261215522933999936,2.251572219942944609268394435362,2.63801038106831059484284086401,2.92963468563478021896214567357,3.10060512443123919200650058479};

  // Gauss-Lagrange weights
  const double glq_weights[]={0.104727102742565162602343989,0.23475763028027358865964263,0.344140053490959719873798523,0.422963173620254734501715992,0.46420836666084341359382055662,0.46420836666084341359382055662,0.422963173620254734501715992,0.344140053490959719873798523,0.23475763028027358865964263,0.104727102742565162602343989};

  #else
  const int glq_n = 15;

  // Gauss-Lagrange nodes x_i in [0, Pi] 
  const double glq_phi[]={0.01886130858747740302305159111,0.09853072480927601402339328008,0.23843654121054845237023696531,0.43288361530924932044530786709,0.673915335359302111609881595248,0.951664838604199667507182945785,1.25476138297089931304806023055,1.57079632679489661923132169164,1.88683127061889392541458315273,2.18992781498559357095546043749,2.46767731823049112685276178803,2.70870903828054391801733551619,2.90315611237924478609240641797,3.04306192878051722443925010320,3.12273134500231583543959179217};
  
  // Gauss-Lagrange weights
  const double glq_weights[]={0.0483070795645355594897227,0.1105307289253955042410349,0.1683253098920381811957743,0.2192371082146767564709842,0.26117505775643872877586132,0.292421015016909813450427436,0.3116954482722822893249093754,0.318209158305239572565215093812,0.3116954482722822893249093754,0.292421015016909813450427436,0.26117505775643872877586132,0.2192371082146767564709842,0.1683253098920381811957743,0.1105307289253955042410349,0.0483070795645355594897227};
  #endif
 
  /*
    Computing area of the surface and the volume of the primary 
    generalized Roche lobes with misaligned spin and orbital velocity 
    vector.
    
    Input:
      pole - pole of the lobe
      Omega0 - value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      th - angle between z axis in spin of the object in [0, pi]
              spin in plane (x, z) 
      m - number of steps in x - direction
      
      choice - using as a mask
        1U  - Area , stored in v[0]
        2U  - Volume, stored in v[1]
        4U  - dVolume/dOmega, stored in v[2]
      
    Using: Integrating surface in spherical coordiantes
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in direction of theta
                      
    Output:
      v[3] = {area, volume, d{volume}/d{Omega}}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
  
  template<class T> 
  void area_volume_integration( 
    T v[3],
    const unsigned & choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    const T & th = 0,
    const int & m = 1 << 14)
  {
    
    //
    // What is calculated
    //
    
    bool 
      b_area = (choice & 1u) == 1u,
      b_vol  = (choice & 2u) == 2u,
      b_dvol = (choice & 4u) == 4u;
    
    if (!b_area && !b_vol && !b_dvol) return;
    
    unsigned mask = 3;
    if (b_dvol) mask += 4; 
    
    const int dim = glq_n + 3;
    
    T W[3], w[glq_n], y[dim], k[4][dim],
      d2 = delta*delta, 
      d3 = delta*d2, 
      d4 = d2*d2,
      b = (1 + q)*F*F*d3,
      dtheta = utils::m_pi/m,
      rt, rp, r,
      s, theta, theta_, r2;
    
    //
    // Setting initial values
    //
    
    { 
      T tp = poleL_height(Omega0, q, F, delta, std::sin(th))/delta;
      for (int i = 0; i < glq_n; ++i) {
        y[i] = tp;
        w[i] = dtheta*glq_weights[i];
      }
      y[glq_n] = y[glq_n + 1] = y[glq_n + 2] = 0;
    }
    
    //
    // Rk integration 
    //
    
    theta = 0;
    for (int i = 0; i < m; ++i) {
      
      // 1. step
      s = std::sin(theta);
      k[0][glq_n] = k[0][glq_n + 1] = k[0][glq_n + 2] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j]; 
        calc_dOmega(mask, W, r, theta, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        k[0][j] = dtheta*rt;
        
        r2 = r*r;
        
        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          k[0][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        }
        if (b_vol)  k[0][glq_n + 1] += w[j]*r*r2;
        if (b_dvol) k[0][glq_n + 2] += w[j]*r2/W[0];
      }
      if (b_vol)  k[0][glq_n + 1] *= s;
      if (b_dvol) k[0][glq_n + 2] *= s;
      
      // 2. step
      theta_ = theta + 0.5*dtheta;
      s = std::sin(theta_);
      k[1][glq_n] = k[1][glq_n + 1] = k[1][glq_n + 2] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j] + 0.5*k[0][j];
        calc_dOmega(mask, W, r, theta_, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        k[1][j] = dtheta*rt;
    
        r2 = r*r;
        
        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          k[1][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        }
        if (b_vol)  k[1][glq_n + 1] += w[j]*r*r2;
        if (b_dvol) k[1][glq_n + 2] += w[j]*r2/W[0];
      }
      if (b_vol)  k[1][glq_n + 1] *= s;
      if (b_dvol) k[1][glq_n + 2] *= s;
      
      
      // 3. step
      //theta_ = theta + 0.5*dtheta;
      //s = std::sin(theta_);
      k[2][glq_n] = k[2][glq_n + 1] = k[2][glq_n + 2] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j] + 0.5*k[1][j];
        calc_dOmega(mask, W, r, theta_, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        k[2][j] = dtheta*rt;

        r2 = r*r;
        
        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          k[2][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        }
        if (b_vol)  k[2][glq_n + 1] += w[j]*r*r2;
        if (b_dvol) k[2][glq_n + 2] += w[j]*r2/W[0];
      }
      if (b_vol)  k[2][glq_n + 1] *= s;
      if (b_dvol) k[2][glq_n + 2] *= s;
      
      
      // 4. step
      theta_ = theta + dtheta;
      s = std::sin(theta_);
      k[3][glq_n] = k[3][glq_n + 1] = k[3][glq_n + 2] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j] + k[2][j];
        calc_dOmega(mask, W, r, theta_, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        k[3][j] = dtheta*rt;

        r2 = r*r;
        
        if (b_area) {
         rp = -W[2]/W[0];      // partial_phi r
         k[3][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        }
        if (b_vol)  k[3][glq_n + 1] += w[j]*r*r2;
        if (b_dvol) k[3][glq_n + 2] += w[j]*r2/W[0];
      }
      if (b_vol)  k[3][glq_n + 1] *= s;
      if (b_dvol) k[3][glq_n + 2] *= s;
      
      // final step
      for (int j = 0; j < dim; ++j)
        y[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;  
            
      theta += dtheta;
    }
    
    if (b_area) v[0] = 2*d2*y[glq_n];
    if (b_vol)  v[1] = 2*d3*y[glq_n + 1]/3;
    if (b_dvol) v[2] = 2*d4*y[glq_n + 2];
  }
  
/*
    Computing volume of the primary generalized Roche lobes with 
    misaligned spin and orbital velocity vector and derivatives of 
    the volume w.r.t. to Omega, ...
    
    Input:
      pole - pole of the lobe
      Omega0 -value of the potential 
      q - mass ratio M2/M1
      F - synchronicity parameter
      d - separation between the two objects
      th - angle between z axis in spin of the object in [0, pi]
              spin in plane (x, z) 
      choice : composing from mask
        1U  - area , stored in v[0]
        2U  - volume, stored in v[1]
        4U  - dVolume/dOmega, stored in v[2]
                
    Output:
      v = {Area, Volume, dVolume/dOmega, ...}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */  
  
  template<class T> 
  void area_volume_asymp(
    T *v,
    const unsigned & choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & d = 1,
    const T & th = 0) {
    
    
    //
    // What is calculated
    //
    
    bool 
      b_area = (choice & 1u) == 1u,
      b_vol  = (choice & 2u) == 2u,
      b_dvol = (choice & 4u) == 4u;
    
    if (!b_area && !b_vol && !b_dvol) return;
    
    T w = d*Omega0, s = 1/w,
      b = (1 + q)*d*d*d*F*F,
      q2 = q*q, 
      b2 = b*b, 
      b3 = b2*b,
      cc = std::cos(2*th);
      
    // calculate area
    if (b_area) {
      
      T a[10] = {
          1, 2*q,3*q2, 2*b/3. + 4*q2, q*(10*b/3. + 5*q2), q2*(10*b + 6*q2),
          b2 + q*(-b/3. + b*cc + q*(2 + q*(70*b/3. + 7*q2))), 
          q*(8*b2 + q*(-8*b/3. + 8*b*cc + q*(16 + q*(140*b/3. + 8*q2)))),
          q2*(15./7 + 36*b2 + q*(-12*b + 36*b*cc + q*(72 + q*(84*b + 9*q2)))),
          68*b3/35. + q*(-41*b2/35. + 123*b2*cc/35. + q*(54*b/7. + 72*b*cc/35. + q*(24.17142857142857 + 120*b2 + q*(-40*b + 120*b*cc + q*(240 + q*(140*b + 10*q2))))))},
      
      sumA = a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*(a[7] + s*(a[8] + s*a[9]))))))));
      
      v[0] = utils::m_4pi/(Omega0*Omega0)*sumA;
    }
    
    if (b_vol || b_dvol) {
      
      T a[10] = {
          1, 3*q, 6*q2, b + 10*q2, q*(6*b + 15*q2), q2*(21*b + 21*q2),
          8*b2/5. + q*(-2*b/5. + 6*b*cc/5. + q*(2.4 + q*(56*b + 28*q2))),
          q*(72*b2/5. + q*(-18*b/5. + 54*b*cc/5. + q*(21.6 + q*(126*b + 36*q2)))),
          q2*(15./7 + 72*b2 + q*(-18*b + 54*b*cc + q*(108 + q*(252*b + 45*q2)))),
          22*b3/7. + q*(-11*b2/7. + 33*b2*cc/7. + q*(143*b/14. + 33*b*cc/14. + q*(26.714285714285715 + 264*b2 + q*(-66*b + 198*b*cc + q*(396 + q*(462*b + 55*q2))))))};
      
      // calculate volume 
      if (b_vol) {
        T sumV = a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*(a[7] + s*(a[8] + s*a[9]))))))));
        v[1] = utils::m_4pi/(3*Omega0*Omega0*Omega0)*sumV;
      }
      
      // calculate d(volume)/dOmega0
      if (b_dvol) {
        T sumdV = 3*a[0] + s*(4*a[1] + s*(5*a[2] + s*(6*a[3] + s*(7*a[4] + s*(8*a[5] + s*(9*a[6] + s*(10*a[7] + s*(11*a[8] + 12*s*a[9]))))))));
        v[2] = -utils::m_4pi/(3*Omega0*Omega0*Omega0*Omega0)*sumdV;
      }
    }
    
  }
  
  /*
    Calculate the value of the Kopal potential of misaligned Roche lobes 
    at given point r.
  
    Omega (r, q,f,d, theta) = 
      1/Sqrt[x^2 + y^2 + z^2] + 
      q (1/Sqrt[(d - x)^2 + y^2 + z^2] - x/d^2) + 
      1/2 F^2 (1 + q) (y^2 + (x Cos[theta] - z Sin[theta])^2)
  
    Input:
      r - point
      q - mass ratio M2/M1
      F - synchronicity parameter
      d - separation between the two objects
      th - angle between z axis in spin of the object
    Return:
      value of Omega
  */
  template <class T> T calc_Omega(
    T r[3],
    const T &q,
    const T &F,
    const T &d,
    const T &th
  ){
    T s, c;
    utils::sincos(th, &s, &c);
    
    T x = r[0], y = r[1], z = r[2], x1 = x*c - s*z,
      r1 = utils::hypot3(x, y, z),
      r2 = utils::hypot3(x - d, y, z);
      
    return 1/r1 + q*(1/r2 - x/(d*d)) + 0.5*F*F*(1 + q)*(x1*x1 + y*y);
  }
  
  /*
    Calculate 
      choice = 0:
        derivative of Lagrange point w.r.t. to theta: 
        dx/(dtheta)= (nabla g)^-1(x) .dg/dtheta(x)
      choice = 1:
        Newton-Rapson step 
        dx(x) = -(nabla g)^-1(x) .g(x)
    
    where g = nabla Omega 
    
    Input:
      choice: 0 or 1
      q - mass ratio M2/M1
      F - synchronicity parameter
      d - separation between the two objects
      th - value of the parameter theta
      x[2] - point (x,z)

    Output:
       k[2] = dx/(dtheta) or dx(x)
    
    Return:
      true - success, false - othewise
  */
  template <class T> 
  bool lag_point_deriv(
    const int & choice,
    const T & q, 
    const T & F,
    const T & d, 
    const T & th, 
    T x[2], 
    T k[2]){
    
    T a = (1 + q)*F*F,
      t1, t2, f03, f05, f13, f15, h[2][2];
  
    t1 = 1/(x[0]*x[0] + x[1]*x[1]), 
    f03 = t1*std::sqrt(t1),
    f05 = f03*t1;

    t1 = 1/((x[0] - d)*(x[0] - d) + x[1]*x[1]), 
    f13 = t1*std::sqrt(t1), 
    f15 = f13*t1;
    
    // Hessian h = nabla_{xz} o nabla_{xz} Omega
    T s, c;
    utils::sincos(th, &s, &c);
    
    t1 = f03 + q*f13, t2 = 3*(f05 + f15*q)*x[1]*x[1];
    
    h[0][0] = 2*t1 - t2 + a*c*c;
    h[0][1] = 3*(f05*x[0] + f15*q*(x[0] - d))*x[1] - a*s*c;
    h[1][1] = -t1 + t2 + a*s*s;
    
    T det = h[0][0]*h[1][1] - utils::sqr(h[0][1]); 
    
    if (det == 0) return false; // TODO: this should be more robust
    
    T g[2];
      
    if (choice == 0){
      utils::sincos(2*th, &s, &c);
      
      // dg = d/d(theta) nabla_{xz} Omega
      g[0] = -a*(x[1]*c + x[0]*s);
      g[1] = -a*(x[0]*c - x[1]*s);
    } else {
      // g = nabla_{xz} Omega
      t1 = f03 + f13*q, t2 = a*(x[0]*c - x[1]*s);
      g[0] = (-1/(d*d) + d*f13)*q - t1*x[0] + c*t2;
      g[1] = -t1*x[1] - s*t2;
    }
    
    // k = - h^-1 g  
    k[0] = (-g[0]*h[1][1] + g[1]*h[0][1])/det;
    k[1] = (+g[0]*h[0][1] - g[1]*h[0][0])/det;
    
    return true;
  }
  
  /*
  Calculate Lagrange points of misaligned Roche lobes in xz-plane
  as analytical continuations of Lagrange points of aligned Roche lobes 
  located on x-axis:
    L1 in [0,d],  L2 < 0,   L3 > d
  
  Input:
    choice - 1 for L1, 2 for L2 and 3 for L3 
    q - mass ratio M2/M1
    F - synchronicity parameter
    d - separation between the two objects
    theta - angle between z axis in spin of the object
  
  Output:
    L - point in xy plane
  Return:
    true - if calculation succeeded, false - otherwise
  */
  
  template<class T> 
  bool lagrange_point( 
    int choice,
    const T & q,
    const T & F,
    const T & d,
    const T & theta,
    T x[2]){
    
    //
    // Lagrange points for aligned case
    //
    T L0;
      
    switch (choice) {
      case 1: L0 = gen_roche::lagrange_point_L1(q, F, d); break;
      case 2: L0 = gen_roche::lagrange_point_L2(q, F, d); break;
      case 3: L0 = gen_roche::lagrange_point_L3(q, F, d); break;
      default: return false;
    }
    
    //
    // Approximating fixed point using RK4 integration from 
    // position at aligned case
    //
    x[0] = L0;
    x[1] = 0;
    
    if (theta == 0) return true;
      
    {
      int n = int(theta/0.1);
      
      bool ok = true; 
         
      T th = 0, dth = theta/n, k[4][2], x1[2];
        
      for (int i = 0; i < n; ++i) {
        
        if (!(ok = lag_point_deriv(0, q, F, d, th, x, k[0]))) break;
        
        for (int j = 0; j < 2; ++j) x1[j] = x[j] + 0.5*(k[0][j] *= dth);
        if (!(ok = lag_point_deriv(0, q, F, d, th + 0.5*dth, x1, k[1]))) break;
        
        for (int j = 0; j < 2; ++j) x1[j] = x[j] + 0.5*(k[1][j] *= dth);
        if (!(ok = lag_point_deriv(0, q, F, d, th + 0.5*dth, x1, k[2]))) break;

        for (int j = 0; j < 2; ++j) x1[j] = x[j] + (k[2][j] *= dth);
        if (!(ok = lag_point_deriv(0, q, F, d, th + dth, x1, k[3]))) break;
       
        for (int j = 0; j < 2; ++j) {
          k[3][j] *= dth;
          x[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;
        }
        th += dth;
      }
      
      if (!ok){
        std::cerr 
          << "misaligned_roche::lagrange_point: hit singularity\n";
        return false;
      }    
    }
    
    //
    // Polish the value of the fixed point via 2D Newton-Raphson
    //
    {
      const T epsR = 100*std::numeric_limits<T>::epsilon();
      const T epsA = 100*std::numeric_limits<T>::min();
      
      T dx[2];
      
      do {
                
        if (!lag_point_deriv(1, q, F, d, theta, x, dx)) {
          std::cerr 
            << "misaligned_roche::lagrange_point: hit singularity2\n";
          return false;
        }

        for (int i = 0; i < 2; ++i) x[i] += dx[i];

      } while (std::abs(dx[0]) > epsR*std::abs(x[0]) + epsA ||
               std::abs(dx[1]) > epsR*std::abs(x[1]) + epsA);
    }
    return true;  
  }
  
  /*
    Calculate the minimal value of the Kopal potential for which the 
    primary Roche lobe exists.
    
    Input:
      q - mass ratio M2/M1
      F - synchronicity parameter
      d - separation between the two objects
      th - angle between z axis in spin of the object
    
    Return:
      minimal permitted Omega, if NaN is returned we have some error
  */
  
  template<class T> 
  T calc_Omega_min( 
    const T & q,
    const T & F,
    const T & d, 
    const T & th = 0){
    
    T W[2], r[3], 
      th1 = utils::m_pi*std::abs(std::fmod(th/utils::m_pi + 0.5, 1) - 0.5);
     
    for (int i = 0; i < 2; ++i) {
      
      if (!lagrange_point(i + 1, q, F, d, th1, r)) return std::nan("");
      
      r[2] = r[1];
      r[1] = 0;
      
      W[i] = calc_Omega(r, q, F, d, th1);
    }
    
    
    return std::max(W[0], W[1]);  
  }


} // namespace misaligned_roche

#endif //#if !defined(__misaligned_roche_h)
