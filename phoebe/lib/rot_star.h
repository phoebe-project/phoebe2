#if !defined(__rot_star_h)
#define __rot_star_h

/*
  Library dealing with rotating stars defined by the potential 

  Omega = 1/sqrt(x^2 + y^2 + z^2) + 1/2 omega^2 (x^2+ y^2) 

  Author: Martin Horvat,  June 2016 
*/

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

// general routines
#include "utils.h"      // Misc routines (sqr, solving poly eq,..)
#include "bodies.h"     // Definitions of different potentials

namespace rot_star {
  
  /*
    "Lagrange point" of the rotating star potential 
      
        (L_1, 0, 0)     (-L_1, 0, 0)
  
    Input:
      omega - parameter of the potential
      
    Return 
      value Lagrange point L_1: if (omega == 0) return NaN
    
  */ 
  template <class T>
  T lagrange_point(const T & omega) {
    if (omega == 0) return std::nan("");
    return std::cbrt(omega*omega);
  }
  
  
  /*  
    Potential on the x-axis
    
    Input:
      x - position on the x-axis
      omega - parameter of the potential
    
    Return 
      value of the potential
  */
 
  template<class T> 
  T potential_on_x_axis( const T & x, const T & omega) {
    return 1/std::abs(x) + omega*omega*x*x/2;
  }
  
  /*
    Points of on the x-axis satisfying
    
      Omega(x,0,0) = Omega_0
    
    The Roche lobes are limited by these points.
    
    Input:
      Omega0 - reference value of the potential
      omega - parameter of the potential
            
      trimming: 
        if false : all solutions
        if true  : trims solutions which bounds Roche lobes,
                   there are even number bounds
        
    Output:
      p - x-values of the points on x-axis sorted
  */
  template<class T> 
  void points_on_x_axis(
    std::vector<T> & points,
    const T & Omega0,
    const T & omega,
    const bool & trimming = true){
    
    if (omega == 0) return 1/Omega0;
    
    std::vector<T> roots;
    
    T a[4] = {1, -Omega0, 0, omega*omega/2};
    
    utils::solve_cubic(a, roots);
    
    for (auto && x: roots)
      if (x > 0) {
          points.push_back(x);
          points.push_back(-x);
      }
    
    std::sort(points.begin(), points.end());
    
    if (points.size() >= 2 && trimming) {
      points.pop_back();
      points.erase(points.begin());
    }
  }
  
  /*
    Pole of the star i.e.  smallest z > 0 such that
  
      Omega(0,0,z) = Omega0
    
    Input:
      Omega0 - value of potential
      omega - parameter of the potential connected to rotation
      
    Return:
      height of pole > 0
  */
  
  template <class T>
  T pole(const T & Omega0, const T & omega) {
    return 1/Omega0; 
  }
  
  
  /*
    Radious of the equator of the star i.e. such  that
  
      Omega(r,0,0) = Omega0
    
    solving
       1/r + 1/2 omega^2 r = Omega0
     
    The rescaled version with variable r = 1/Omega v is 
  
       1/v + 1/2 t v^2 = 1      or 
        
      1/2 t v^3 - v + 1 = 0
    
    with parameter 
      
      t = omega^2/Omega0^3
    
    Input:
      Omega0 - value of potential
      omega - parameter of the potential connected to rotation > 0
      
    Return:
      radius of the equator
    
    Comment:
      if equator does not exist return std::nan()
  */
  template <class T>
  T equator(const T & Omega0, const T & omega) {
    
    // if we are discussing sphere
    if (omega == 0) return 1/Omega0;
        
    //
    // Testing if the solution exists
    //  
    
    T t = omega*omega/(Omega0*Omega0*Omega0);
    
    if (t > 8./27) {   // critical value
      std::cerr << "equator::area_volume:There is no solution for equator.\n";
      return std::nan("");
    }
   
    // Approximation: series are generated in rot_star.nb
    
    T r;
    if (t < 0.2) {
      r = 1 + t*(0.5 + t*(0.75 + t*(1.5 + t*(3.4375 + t*(8.53125 + t*(22.3125 + t*(60.5625 + 168.99609375*t)))))));
    } else {
      T u = 8./27 - t;
      r = 1.5*(1. + u*(-1.0606601717798214 + u*(1.5 +u*(-2.3201941257683596 + u*(3.75 + u*(-6.221020499716413 + u*(10.5 + u*(-17.940978762575014 + 30.9375*u))))))));
    }
    
    //
    // Newton-Raphson iteration
    // as approximations are so good it should not fail
    //
    
    const int max_iter = 10;
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    int it = 0;
    
    T f, dr;
    
    do {
      f = r*r*t;
      r -= (dr = (2 + r*(-2 + f))/(-2 + 3*f)); // = poly/Dpoly
    } while (std::abs(dr) > eps*r + min && ++it < max_iter);
    
    /*
      std::vector<T> roots;
      
      T a[4] = {1, -1, 0, t};
      
      utils::solve_cubic(a, roots);
      
      for (auto && x: roots) if (x > 0) return x/Omega0;
    */
    
    return r/Omega0; 
  }
    

  
  /*
    Returning the critical values of the star potential.
    
    Input:
      omega - potential parameter
    
    Return:
      critical value of the potential 
      if it does not exist returns NaN
  */

  template<class T> 
  T critical_potential(const T & omega) {
  
    if (omega == 0) return std::nan("");
    
    return 3*std::pow(omega, 2./3)/2;
  }
  
 /*
    Computing area of the surface and the volume of the rotating star 
    lobe.
  
    Input:
      omega - parameter of the potential
      Omega0 - value of the potential 
      choice -
        1  - area
        2  - volume
        3  - both

    Using: Gauss-Lagrange integration along z direction
       
    Output:
      av[2] = {area, volume}
    
    Precision is at least 10^-5.
    
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
 
  template<class T> 
  void area_volume(
    double *av, 
    const unsigned &choice, 
    const T & Omega0, 
    const T & omega) {

    //
    // What is calculated
    //
    
    bool 
      b_area = (choice & 1u) == 1u,
      b_volume = (choice & 2u) == 2u;

    if (!b_area && !b_volume) return;
    
    T Omega2 = Omega0*Omega0,
      Omega3 = Omega0*Omega2;
      
    if (omega == 0) {
      
      if (b_area) av[0] = utils::m_4pi/Omega2;
      if (b_volume) av[1] = utils::m_4pi/(3*Omega3); 
        
      return;
    }  
    
    //
    // Testing if the solution exists
    //  
    
    T t = omega*omega/Omega3;
    
    if (t > 8./27) {   // critical value
      std::cerr << "rotstar::area_volume:There is no solution for equator.\n";
      return;
    }

    //
    // Analytic approximation (generated in rot_star.nb)
    // relative precision at least 1e-5 for t < 0.1
    
    if (t < 0.1) {
      
      T f = utils::m_4pi/Omega2;
      
      if (b_area) {
        
        // coefficients generated by volume_rotstar_series.m in spherical coordinates
        // A(t) = series up to degree m = 10
        const T a[] = {
          1., 2./3, 1., 1.9428571428571428571, 4.3142857142857142857, 
          10.398268398268398268, 26.488777888777888778, 70.225419025419025419, 
          191.86864508040978629, 536.69475993562680869, 1529.8380707699593149};
              
        T A = a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*(a[5] + t*(a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t*a[10])))))))));
        
        av[0] = f*A;
      }
      
      if (b_volume) {
        
        // coefficients generated by volume_rotstar_series.m in spherical coordinates
        // V(t) = series up to degree m=10;

        const T a[] = {
          1., 1, 1.6, 3.1428571428571428571, 6.9333333333333333333, 
          16.484848484848484848, 41.302697302697302697, 107.56923076923076923, 
          288.62745098039215686, 792.96594427244582043, 2220.8419444332757026};
      
        
        T V = a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*(a[5] + t*(a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t*a[10])))))))));
        
        av[1] = f*V/(3*Omega0);
      }
      
      return;
    }
    
    //
    // Integrate using new variables
    //  v = Omega z, u = Omega rho  rho^2 = x^2 + y^2 
    // we additionally introduce
    //  s = u^2,
    // and rewrite equation for potential:
    //  1 = 1/sqrt(s + v^2) + 1/2 t s     v in [0,1]   
    // where
    //  v=1 is pole, v=0 is equator
    // with t = omega^2/Omega^3
    
    // u du/dv = 1/2 ds/dv
    
    const int m = 1 << 16;
        
    T dv = 1./m, v = 1,
      s = 0, A = 0, V = 0, k[4][3],
      v1, s1, q, F; // auxiliary variables  
    
    
    // FIX: just that I don't get may be used uninitialized in this function 
    for (int i = 0; i < 4; ++i) k[i][0] = k[i][1] = k[i][2] = 0;
  
    for (int i = 0; i < m; ++i) {
      
      //
      // 1. step
      //
      v1 = v; 
      s1 = s; 
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));              // = ds/dv
      
      k[0][0] = dv*F;                                 // = dv*ds/dv
      if (b_area) k[0][1] = dv*std::sqrt(s1 + F*F/4); // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[0][2] = dv*s1;                  // dv u^2
        
      // prepare: y1 = y + k0/2
      s1 = s + k[0][0]/2;
      
      //
      // 2. step 
      //
      v1 = v - dv/2;
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));               // =ds/dv
       
      k[1][0] = dv*F;                                  // = dv*ds/dv
      if (b_area) k[1][1] = dv*std::sqrt(s1 + F*F/4);  // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[1][2] = dv*s1;                   // dv u^2
      
      // prepare: y1 = y + k1/2
      s1 = s + k[1][0]/2;
      
      //
      // 3. step
      //
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));                // =ds/dv
      
      k[2][0] = dv*F;                                   // = dv*ds/dv
      if (b_area) k[2][1] = dv*std::sqrt(s1 + F*F/4);   // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[2][2] = dv*s1;                    // dv u^2
      
      // prepare: y1 = y + k1/2
      s1 = s + k[2][0];
      
      // 4. step
      v1 = v - dv;
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));                // =ds/dv
      
      k[3][0] = dv*F;                                   // = dv*ds/dv
      if (b_area) k[3][1] = dv*std::sqrt(s1 + F*F/4);   // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[3][2] = dv*s1;                    // dv u^2
    
      s += (k[0][0] + 2*(k[1][0] + k[2][0]) + k[3][0])/6;
      if (b_area) A += (k[0][1] + 2*(k[1][1] + k[2][1]) + k[3][1])/6;
      if (b_volume) V += (k[0][2] + 2*(k[1][2] + k[2][2]) + k[3][2])/6;
          
      v -= dv;
    }
    
    if (b_area) av[0] = utils::m_4pi*A/Omega2;
    if (b_volume) av[1] = utils::m_2pi*V/Omega3;
  }
  
  /*
    Computing the volume of the rotating star and its derivatives w.r.t. to Omega, ...
    
    The range on x-axis is [x0, x1].
  
    Input:
      Omega0 - value of the potential
      omega - parameter of th potential
      choice : composing from mask
        1  - Volume, stored in av[0]
        2  - dVolume/dOmega, stored in av[1]
                 
    Using: Integrating surface in cylindric geometry
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in x direction
    
    Precision:
      At the default setup the relative precision should be better 
      than 1e-5.
             
    Output:
      res = {Volume, dVolume/dOmega, ...}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
  
  template<class T> 
  void volume(
    T *res,
    const unsigned & choice,
    const T & Omega0,
    const T & omega)
  {
  
    //
    // What is calculated
    //
    
    bool 
      b_Vol= (choice & 1u) == 1u,
      b_dVoldOmega = (choice & 2u) == 2u;

    if (!b_Vol && !b_dVoldOmega) return;
    
    T Omega2 = Omega0*Omega0,
      Omega3 = Omega0*Omega2;
      
    if (omega == 0) {
      
      T Vol =  utils::m_4pi/(3*Omega3);
      
      if (b_Vol) res[0] = Vol;
      if (b_dVoldOmega) res[1] = -3*Vol/Omega0; 
        
      return;
    }  

    //
    // Testing if the solution exists
    //
    
    T t = omega*omega/Omega3; // t in [t_crit], t_crit = 8/27
    
    if (t > 8./27) {   // critical value
      std::cerr << "rotstar::area_volume:There is no solution for equator.\n";
      return;
    }

    //
    // Analytic approximation (generated in rot_star.nb)
    // relative precision at least 1e-5 for t < 0.1
    if (t < 0.1) {
      
      const T a[] = {1, 1, 1.6, 3.1428571428571428571, 6.9333333333333333333, 16.484848484848484848, 41.302697302697302697, 107.56923076923076923, 288.62745098039215686, 792.96594427244582043, 2220.8419444332757026};
      
      T Vol = 
          a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*(a[5] + 
          t*(a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t*a[10]))))))))),
        f = utils::m_4pi/(3*Omega3);
      
      if (b_Vol) res[0] = f*Vol;
      
      if (b_dVoldOmega) {
        T dVoldt = 
            a[1] + t*(2*a[2] + t*(3*a[3] + t*(4*a[4] + t*(5*a[5] + 
            t*(6*a[6] + t*(7*a[7] + t*(8*a[8] + t*(9*a[9] + t*10*a[10]))))))));
        
        res[1] = -3*f*(Vol + t*dVoldt)/Omega0;
      }
      
      return;
    }
    
    //
    // Integrate using new variables
    //  v = Omega z, u = Omega rho  rho^2 = x^2 + y^2 
    // we additionally introduce
    //  s = u^2,
    // and rewrite equation for potential:
    //  1 = 1/sqrt(s + v^2) + 1/2 t s     v in [0,1]   
    // where
    //  v=1 is pole, v=0 is equator
    // with t = omega^2/Omega^3
    
    // u du/dv = 1/2 ds/dv
    
    const int m = 1 << 16;
        
    T dv = 1./m, v = 1,
      s = 0, Vol = 0, dVoldOmega = 0, k[4][3],
      v1, s1, q, f, F; // auxiliary variables  
    
    
    // FIX: just that I don't get may be used uninitialized in this function 
    for (int i = 0; i < 4; ++i) k[i][0] = k[i][1] = k[i][2] = 0;
  
    for (int i = 0; i < m; ++i) {
      
      //
      // 1. step
      //
      v1 = v; 
      s1 = s; 
      q = v1*v1 + s1;
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);              // = ds/dv
      
      k[0][0] = dv*F;                  // = dv*ds/dv
      if (b_Vol) k[0][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[0][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
        
      // prepare: y1 = y + k0/2
      s1 = s + k[0][0]/2;
      
      //
      // 2. step 
      //
      v1 = v - dv/2;
      q = v1*v1 + s1; 
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);               // = ds/dv
       
      k[1][0] = dv*F;                   // = dv*ds/dv
      if (b_Vol) k[1][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[1][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
       
      // prepare: y1 = y + k1/2
      s1 = s + k[1][0]/2;
      
      //
      // 3. step
      //
      q = v1*v1 + s1; 
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);               // = ds/dv
      
      k[2][0] = dv*F;                   // = dv*ds/dv
      if (b_Vol) k[2][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[2][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
   
      // prepare: y1 = y + k1/2
      s1 = s + k[2][0];
      
      // 4. step
      v1 = v - dv;
      q = v1*v1 + s1; 
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);               // = ds/dv
      
      k[3][0] = dv*F;                   // = dv*ds/dv
      if (b_Vol) k[3][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[3][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
      
      s += (k[0][0] + 2*(k[1][0] + k[2][0]) + k[3][0])/6;
      if (b_Vol) Vol += (k[0][1] + 2*(k[1][1] + k[2][1]) + k[3][1])/6;
      if (b_dVoldOmega) dVoldOmega += (k[0][2] + 2*(k[1][2] + k[2][2]) + k[3][2])/6;
          
      v -= dv;
    }
    
    f = utils::m_2pi/Omega3;
    
    if (b_Vol) res[0] = f*Vol;
    if (b_dVoldOmega) res[1] = -3*f*dVoldOmega/Omega0;  
  }
  
    
  /* 
    Find a point on the horizon around a lobe of the rotating star.
  
    Input:
      view - direction of the view
      Omega0 - value of the potential
      omega - parameter of th potential
    
    Output:
      p - point on the horizon
  */
  template<class T> 
  bool point_on_horizon(
    T r[3], 
    T view[3],
    const T & Omega0,
    const T & omega
  ){
    
    T r0 = equator(Omega0, omega);
    
    if (view[0] == 0 && view[1] == 0) {
      r[0] = r0;
      r[1] = r[2] = 0;
    }
    
    if (std::isnan(r0)) return false;
    
    T f = r0/std::sqrt(view[0]*view[0] + view[1]*view[1]);
    r[0] = -f*view[1];
    r[1] = f*view[0];
    r[2] = 0;
    
    
    return true;
  }
  
  /*
    Initial point: it is on the z-axis, z>0
    
    Input:
      Omega0 - value of the potential
      omega - parameter of th potential
  
    Output:
      r - position 
      g - gradient
  */ 
  template <class T>
  void meshing_start_point(
    T r[3], 
    T g[3],
    const T & Omega0,
    const T & omega
  ){
    r[0] = r[1] = 0;
    r[2] = 1/Omega0;
    
    g[0] = g[1] = 0;
    g[2] = Omega0*Omega0;
  }
  
} // namespace rot_star

#endif // #if !defined(__rot_star_h) 
