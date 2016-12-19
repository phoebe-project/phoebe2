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

  template <class T>
  T poleL_height(
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    const T & theta = 0
  ) {
    
    if (theta == 0)
      return gen_roche::poleL(Omega0, q, F, delta);

    if (Omega0 < 0 || q < 0)  return -1;
   
    T w = Omega0*delta, s, c, t;
    
    utils::sincos(theta, &s, &c);
      
    // calculate the estimate of the pole (in direction of the spin)
    // note: there is no symmetry across the equator 
    // TODO: improve the estimate of the poles
    if (w >= 10 && w > q) {  
      t = 1/w;
      t *= 1 + q*t*(1 + t*(q + (-1 + 2*q*q + 3*s*s)*t/2));
    } else if (q > 10 && q > w) {
      t = (std::sqrt(w*w + 4*(1+q)*s*q) - w)/(2*q*s); 
    } else { 
      t = 1;
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
    Pole of the first star h = delta*tp: 
  
      Omega(d tp s, 0, d*tp*c) = Omega0
      
      s = sin(theta')
      c = cos(theta')
       
    Solving
    
      1/tp + q (1/sqrt(tp^2 - 2 tp*s + 1) - s*tp) = delta Omega
      
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      theta - angle of spin w.r.t. z axis
      
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
    const T & F = 1,
    const T & delta = 1,
    const T & theta = 0
  ) {
    
    T tp = poleL_height(Omega0, q, F, delta, theta);
  
    if (tp > 0) {
      T s, c;
      
      utils::sincos(theta, &s, &c);
      
      p[0] = s*tp;
      p[1] = 0;
      p[2] = c*tp;
      return true;
    }
    
    return false;
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
    
    T fac = 0.5*poleL_height(Omega0, q, F, delta, theta);
     
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
    
    #if 0  
    
    // searching point with largest x^2 + z^2 in y-z plane
    // eventually I can implement, I have equations
          
    //??????

    #else
  
    // only primary lobes for theta != 0 are supported
    if (choice != 0) {
      std::cerr 
        << "meshing_start_point:: choices != 0 not supported yet\n";
      return false;
    }
   
    // for cases which are not strongy deformed the pole is also 
    // good point to starts
    
    if (!poleL(r, Omega0, q, F, delta, theta)) return false;
  
    #endif
    
    // calculating the gradient
    T params[] = {q, F, delta, theta, Omega0};
    
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
      
      choice -
        1  - area
        2  - volume
        3  - both
      
    Using: Integrating surface in spherical coordiantes
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in direction of theta
          
    Precision:
      ???
             
    Output:
      av[2] = {area, volume}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
  
  template<class T> 
  void area_volume( 
    T av[2],
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
      b_volume = (choice & 2u) == 2u;
    
    if (!b_area && !b_volume) return;
    
    const int dim = glq_n + 2;
    
    T W[3], w[glq_n], y[dim], k[4][dim],
      d2 = delta*delta,
      d3 = delta*d2,
      b = (1 + q)*F*F*d3,
      dtheta = utils::m_pi/m,
      rt, rp, r,
      s, theta, theta_, r2;

    { 
      T tp = poleL_height(Omega0, q, F, delta, th)/delta;
      for (int i = 0; i < glq_n; ++i) {
        y[i] = tp;
        w[i] = dtheta*glq_weights[i];
      }
      y[glq_n] = y[glq_n + 1] = 0;
    }
    
    theta = 0;
    for (int i = 0; i < m; ++i) {
      
      // 1. step
      s = std::sin(theta);
      k[0][glq_n] = k[0][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j]; 
        calc_dOmega(7, W, r, theta, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        rp = -W[2]/W[0];      // partial_phi r
        
        k[0][j] = dtheta*rt;
        
        r2 = r*r;
        
        if (b_area) k[0][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        if (b_volume) k[0][glq_n + 1] += w[j]*r*r2;
      }
      if (b_volume) k[0][glq_n + 1] *= s;
    
      // 2. step
      theta_ = theta + 0.5*dtheta;
      s = std::sin(theta_);
      k[1][glq_n] = k[1][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j] + 0.5*k[0][j];
        calc_dOmega(7, W, r, theta_, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        rp = -W[2]/W[0];      // partial_phi r
        
        k[1][j] = dtheta*rt;
    
        r2 = r*r;
        
        if (b_area) k[1][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        if (b_volume) k[1][glq_n + 1] += w[j]*r*r2;
      }
      if (b_volume) k[1][glq_n + 1] *= s;
      
      
      // 3. step
      theta_ = theta + 0.5*dtheta;
      s = std::sin(theta_);
      k[2][glq_n] = k[2][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        
        r = y[j] + 0.5*k[1][j];
        calc_dOmega(7, W, r, theta_, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        rp = -W[2]/W[0];      // partial_phi r
        
        k[2][j] = dtheta*rt;

        r2 = r*r;
        
        if (b_area) k[2][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        if (b_volume) k[2][glq_n + 1] += w[j]*r*r2;
      }
      if (b_volume) k[2][glq_n + 1] *= s;
      
      
      // 4. step
      theta_ = theta + dtheta;
      s = std::sin(theta_);
      k[3][glq_n] = k[3][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        r = y[j] + k[2][j];
        calc_dOmega(7, W, r, theta_, glq_phi[j], q, b, th);
        
        rt = -W[1]/W[0];      // partial_theta r
        rp = -W[2]/W[0];      // partial_phi r
        
        k[3][j] = dtheta*rt;

        r2 = r*r;
        
        if (b_area) k[3][glq_n] += w[j]*r*std::sqrt(rp*rp + s*s*(r2 + rt*rt));
        if (b_volume) k[3][glq_n + 1] += w[j]*r*r2;
      }
      if (b_volume) k[3][glq_n + 1] *= s;
      
      // final step
      for (int j = 0; j < dim; ++j)
        y[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;  
      
      /*
        std::cerr << "theta=" << theta << '\n';
      
      for (int j = 0; j < dim; ++j)
        std::cerr << j << '\t' << y[j] << '\n';
      std::cerr << '\n';
      */
      
      theta += dtheta;
    }
    
    if (b_area) av[0] = 2*d2*y[glq_n];
    if (b_volume) av[1] = 2*d3*y[glq_n + 1]/3;
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
      delta - separation between the two objects
      theta - angle between z axis in spin of the object in [0, pi]
              spin in plane (x, z) 
      choice : composing from mask
        1  - Volume, stored in av[0]
        2  - dVolume/dOmega, stored in av[1]
      
      m - number of steps in x - direction
                 
    Using: Integrating surface in cylindric geometry
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in the direction of theta
    
    Precision:
      ????
       
    Output:
      v = {Volume, dVolume/dOmega, ...}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
  
  template<class T> 
  void volume(
    T *v,
    const unsigned & choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    const T & th = 1,
    const int & m = 1 << 14)
  {
    
    //
    // What is calculated
    //
    
    bool 
      b_vol        = (choice & 1u) == 1u, // calc. Volume
      b_dvoldomega = (choice & 2u) == 2u; // calc. dVolume/dOmega
    
    if (!b_vol && !b_dvoldomega) return;
    
    const int dim = glq_n + 2;
    
    T W[3], w[glq_n], y[dim], k[4][dim],
      d3 = delta*delta*delta,
      d4 = delta*d3,
      b = (1 + q)*F*F*d3,
      dtheta = utils::m_pi/m,
      s, theta, theta_, r, r2;

    { 
      T tp = poleL_height(Omega0, q, F, delta, th)/delta;
      for (int i = 0; i < glq_n; ++i) {
        y[i] = tp;
        w[i] = dtheta*glq_weights[i];
      }
      y[glq_n] = y[glq_n + 1] = 0;
    }
    
    theta = 0;
    for (int i = 0; i < m; ++i) {
      
      // 1. step
      s = std::sin(theta);
      k[0][glq_n] = k[0][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        r = y[j];
        calc_dOmega(3, W, r, theta, glq_phi[j], q, b, th);
        
        k[0][j] = dtheta*(-W[1]/W[0]);
        
        r2 = r*r;
        
        if (b_vol) k[0][glq_n] += w[j]*r*r2;
        if (b_dvoldomega) k[0][glq_n+1] += w[j]*r2/W[0];
      }
      if (b_vol) k[0][glq_n] *= s;
      if (b_dvoldomega) k[0][glq_n+1] *= s;
    
      // 2. step
      theta_ = theta + 0.5*dtheta;
      s = std::sin(theta_);
      k[1][glq_n] = k[1][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        r = y[j] + 0.5*k[0][j];
        calc_dOmega(3, W, r, theta_, glq_phi[j], q, b, th);
        
        k[1][j] = dtheta*(-W[1]/W[0]);
        
        r2 = r*r;
        
        if (b_vol) k[1][glq_n] += w[j]*r*r2;
        if (b_dvoldomega) k[1][glq_n+1] += w[j]*r2/W[0];
      }
      if (b_vol) k[1][glq_n] *= s;
      if (b_dvoldomega) k[1][glq_n+1] *= s;

      // 3. step
      theta_ = theta + 0.5*dtheta;
      s = std::sin(theta_);
      k[2][glq_n] = k[2][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        r = y[j] + 0.5*k[1][j];
        calc_dOmega(3, W, r, theta_, glq_phi[j], q, b, th);
        
        k[2][j] = dtheta*(-W[1]/W[0]);
        
        r2 = r*r;
        
        if (b_vol) k[2][glq_n] += w[j]*r*r2;
        if (b_dvoldomega) k[2][glq_n+1] += w[j]*r2/W[0];
      }
      if (b_vol) k[2][glq_n] *= s;
      if (b_dvoldomega) k[2][glq_n+1] *= s;
      
      // 4. step
      theta_ = theta + dtheta;
      s = std::sin(theta_);
      k[3][glq_n] = k[3][glq_n + 1] = 0;
      for (int j = 0; j < glq_n; ++j){
        r = y[j] + k[2][j];
        calc_dOmega(3, W, r, theta_, glq_phi[j], q, b, th);
        
        k[3][j] = dtheta*(-W[1]/W[0]);
        
        r2 = r*r;
        
        if (b_vol) k[3][glq_n] += w[j]*r*r2;
        if (b_dvoldomega) k[3][glq_n+1] += w[j]*r2/W[0];
      }
      if (b_vol) k[3][glq_n] *= s;
      if (b_dvoldomega) k[3][glq_n+1] *= s;
      
      // final step
      for (int j = 0; j < dim; ++j)
        y[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;  
      
      theta += dtheta;
    }
    
    if (b_vol) v[0] = 2*d3*y[glq_n]/3;
    if (b_dvoldomega) v[1] = 4*d4*y[glq_n + 1]/3;
  }
  
} // namespace misaligned_roche

#endif //#if !defined(__misaligned_roche_h)
