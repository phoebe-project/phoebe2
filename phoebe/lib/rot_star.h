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
#include "utils/utils.h"                  // Misc routines (sqr, solving poly eq,..)
#include "triang/triang_marching.h"       // Maching triangulation
#include "eclipsing/eclipsing.h"          // Eclipsing/Hidden surface removal
#include "triang/bodies.h"                // Definitions of different potentials

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
    
    Comment:
      if pole is not found it return -1
  */
  template <class T>
  T pole(const T & Omega0, const T & omega) {
    return 1/Omega0; 
  }
  
  
  /*
    Returning the critical values of the star potential.
    
    Input:
      omega - potential parameter
    
    Return:
      critical value of the potential : if (omega == 0) return NaN
  */

  template<class T> 
  T critical_potential(const T & omega) {
    if (omega == 0) return std::nan("");
    //return potential_on_x_axis(lagrange_point(omega), omega);
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
      
      if (b_area) av[0] = utils::M_4PI/Omega2;
      if (b_volume) av[1] = utils::M_4PI/(3*Omega3); 
        
      return;
    }  
    
    //
    // Testing if the solution exists
    //
    
    T b = omega*omega/(2*Omega3);
    
    if (b > 4/27.) { // critical b = 4/27
      std::cerr << "rotstar::area_volume:There is no solution for equator.\n";
      return;
    }

    //
    // Integrate using new variables
    //  v = Omega z, u = Omega rho  rho^2 = x^2 + y^2 
    // we additionally introduce
    //  s = u^2, t = v^2  dt = 2v dv
    // and rewrite equation for potential:
    //  1 = 1/sqrt(s + t) + b s     t in [0,1]   
    // where
    //  t=1 is pole, t=0 is equator
    // with b = 1/2 omega^2/Omega^3
    
    // u du/dv = 1/2 ds/dv = v*ds/dt
    
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
      F = 2*v1/(1 - 2*b*q*std::sqrt(q));              // = ds/dv
      
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
      F = 2*v1/(1 - 2*b*q*std::sqrt(q));               // =ds/dv
       
      k[1][0] = dv*F;                                  // = dv*ds/dv
      if (b_area) k[1][1] = dv*std::sqrt(s1 + F*F/4);  // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[1][2] = dv*s1;                   // dv u^2
      
      // prepare: y1 = y + k1/2
      s1 = s + k[1][0]/2;
      
      //
      // 3. step
      //
      q = v1*v1 + s1; 
      F = 2*v1/(1 - 2*b*q*std::sqrt(q));                // =ds/dv
      
      k[2][0] = dv*F;                                   // = dv*ds/dv
      if (b_area) k[2][1] = dv*std::sqrt(s1 + F*F/4);   // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[2][2] = dv*s1;                    // dv u^2
      
      // prepare: y1 = y + k1/2
      s1 = s + k[2][0];
      
      // 4. step
      v1 = v - dv;
      q = v1*v1 + s1; 
      F = 2*v1/(1 - 2*b*q*std::sqrt(q));                // =ds/dv
      
      k[3][0] = dv*F;                                   // = dv*ds/dv
      if (b_area) k[3][1] = dv*std::sqrt(s1 + F*F/4);   // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[3][2] = dv*s1;                    // dv u^2
    
      s += (k[0][0] + 2*(k[1][0] + k[2][0]) + k[3][0])/6;
      if (b_area) A += (k[0][1] + 2*(k[1][1] + k[2][1]) + k[3][1])/6;
      if (b_volume) V += (k[0][2] + 2*(k[1][2] + k[2][2]) + k[3][2])/6;
          
      v -= dv;
    }
    
    if (b_area) av[0] = utils::M_4PI*A/Omega2;
    if (b_volume) av[1] = utils::M_2PI*V/Omega3;
  }
  
  
} // namespace rot_star

#endif // #if !defined(__rot_star_h) 
