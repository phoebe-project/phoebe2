#if !defined(__gen_roche_h)
#define __gen_roche_h

/*
  Library dealing with generalized Roche lobes/Kopal potential 

  Omega = 
    1/rho 
    + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
    + 1/2 F^2(1 + q) rho^2 (1 - nu^2)

  where position in spherical coordinates is given as

  x = rho lambda      lambda = sin(theta) cos(phi)
  y = rho  mu         mu = sin(theta) sin(phi)
  z = rho nu          nu = cos(theta)

  Author: Martin Horvat,  March 2016 
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

// Roche specific routines and part of gen_roche namespace
#include "lagrange/gen_roche_lagrange.h"  // Lagrange points for gen Kopal potential
#include "volume/gen_roche_area_volume.h" // Roche lobes volume

namespace gen_roche {

  /*  
    Generalized Kopal potential on the x-axis
    
    Input:
      x - position on the x-axis
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
    
    Return 
      value of the potential
  */
 
  template<class T> 
  T potential_on_x_axis(
    const T & x,
    const T & q,
    const T & F = 1,
    const T & delta = 1)
  {      
    return 1/std::abs(x)
           + q*(1/std::abs(delta - x) - x/(delta*delta))
           + F*F*(1 + q)*x*x/2;
  }

  /*
    Returning the critical values of the generalized Kopal potential, i.e.
    the values at the L1, L2, L3 point
    
    Input:
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
    
    Output:
      omega_crit[3] - critical value of the potential at L1, L2, L3 point
  */

  template<class T> 
  void critical_potential(
    T omega_crit[3],
    const T & q, const T & F = 1, const T & delta = 1) {
    
    // note: x in [0, delta]
    omega_crit[0] = 
      potential_on_x_axis(lagrange_point_L1(q, F, delta), q, F, delta);
    // note: x < 0
    omega_crit[1] = 
      potential_on_x_axis(lagrange_point_L2(q, F, delta), q, F, delta);
    // note : x > delta
    omega_crit[2] = 
      potential_on_x_axis(lagrange_point_L3(q, F, delta), q, F, delta);
  }
  
  /*
    Returning the critical values of the generalized Kopal potential, i.e.
    the values at the L1, L2, L3 point
    
    Input:
      choice - 
        if 1st bit is set : L1 is stored in omega_crit[0]
        if 2nd bit is set : L2 is stored in omega_crit[1]
        if 3ed bit is set : L3 is stored in omega_crit[2]
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
    
    Output:
      omega_crit[3] - critical value of the potential at L1, L2, L3 point
      L_points[3] - value of  L1, L2, L3 point
  */
  template<class T> 
  void critical_potential(
    T omega_crit[3], 
    T L_points[3], 
    unsigned choice, 
    const T & q, const T & F = 1, const T & delta = 1) {
    
    // note: x in [0, delta]
    if ((choice & 1u) == 1u )
      omega_crit[0] = 
        potential_on_x_axis(
          L_points[0] = lagrange_point_L1(q, F, delta), 
          q, F, delta);
      
    // note: x < 0
    if ((choice & 2u) == 2u)
      omega_crit[1] = 
        potential_on_x_axis(
          L_points[1] = lagrange_point_L2(q, F, delta), 
          q, F, delta);
  
    // note : x > delta
    if ((choice & 4u) == 4u)
      omega_crit[2] = 
        potential_on_x_axis(
          L_points[2] = lagrange_point_L3(q, F, delta), 
          q, F, delta);
  }
  
  /*
    Unified treatment of the common equation for the poles of the 
    Roche lobes basicaly solving:
    
      q/sqrt(1 + h^2) + 1/h = w
  */
  
  template<class T>
  T poleLR(const T &w, const T &q){
    
    if (w < 0 || q < 0)  return -1;
    
    T s, h;
    
    bool iterative = true;
    
    if (w >= 10 && q < 4*w) {                //  w -> infty
      s = 1/w;
      h = s*(1 + q*s*(1 + q*s));
    } else if (q >= 10 && w < 4*q) {         // q -> infty
      s = 1/q;
      h = 1/(s*w*(1 + s*(-1 + s*(1 + w*w/2))));
    } else if (w < 2 + 2./3*q){              // w -> 0
      s = w/(1 + q); 
      h = 1/(s*(1 + q*s*s/(2*(1 + q))));
    } else if (2*q < w + 2) {                // q -> 0
      T t = 1 + w*w, t2 = t*t, t4 = t2*t2,
        a = 1/(w*std::sqrt(t)),
        b = w/t2,
        c = (2*w*w - 3)/(2*t4*a);
        
      h = 1/w + q*(a + q*(b + c*q));
    } else iterative = false;
    
    if (iterative) {  
    
      const int iter_max = 100;
      const T eps = 4*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      int it = 0;
      
      T t1, t2, h2, f, dh, df;
      
      do {
        h2 = h*h;
        t1 = 1 + h2;
        t2 = std::sqrt(t1);
        
        f = 1/h + q/t2 - w;
        df = -1/h2 - h*q/(t1*t2); 
        
        h -= (dh = f/df);
        
      } while (std::abs(dh) <  eps*std::abs(h) + min && (++it) < iter_max);
    
    } else {  // some generic regime
     
    
      T a[5] = {1, -2*w, 1 + (w + q)*(w - q), -2*w, w*w};
      
      std::vector<T> roots;
    
      utils::solve_quartic(a, roots);
            
      for (auto && z : roots) if (z > 0 && w*z >= 1) return z;
      
      return -1;
    }
    
    return h;  
  }
  
  
  /*
    Pole of the first star at (0,0,z), z > 0, i.e. 
    smallest z > 0 such that
  
      Omega(0,0,z) = Omega0
    
    Solving
      q/sqrt(h^2+1) + 1/h = delta Omega
      
      z = delta h
      
    
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      
    Return:
      height of pole > 0
    
    Comment:
      if pole is not found it return -1
  */
  template <class T>
  T poleL(
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1
  ) {

    return delta*poleLR(Omega0*delta, q);
  }

  
  /*
    Pole of the second star at (delta,0,z), z > 0, i.e. 
    smallest z > 0 such that
  
      Omega(1,0,z) = Omega0
    
    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      
    Return:
      height of pole > 0
    
    Comment:
      if pole is not found it return -1
  */
  template <class T>
  T poleR(
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1
  ) {
  
    T p = 1/q,
      nu = 1 + Omega0*delta*p - F*F*delta*delta*delta*(1 + p)/2;
     
    return delta*poleLR(nu, p);
  }

   
  /*
    Based on the critical omegas determine the type of system we can 
    have at that potential.
    
    Input:
      omega - value of the potential
      omega_crit[3] -  critical value of the potential at L1, L2, L3 point
    Return:
      type of the system
    Output:
      number of bodies
  */

  enum Tsystem_type {
    detached,
    solo_left,
    solo_right,
    overcontact,
    overflow
  };

  template<class T> 
  Tsystem_type determine_type(
    const T &omega,
    T omega_crit[3],
    int & nr) {
    
    T oL1 = omega_crit[0],  // central
      oL2 = omega_crit[1],  // left
      oL3 = omega_crit[2];  // right
    
    // omega > all omegas
    if (omega > oL1 && omega > oL2 && omega > oL3){
      nr = 2;
      return detached;
    } 
    
    // omega < all omegas
    if (omega < oL1 && omega < oL2 && omega < oL3 ){
      nr = 0;
      return overflow;
    } 
    
    
    if (oL1 > oL2 && oL1 > oL3) { // omega(L1) > omega(L_{2,3})  
      
      if (oL2 >= oL3) {
        if (omega >= oL2) {
          nr = 1;
          return overcontact;  
        } 
      } else { // oL2 < oL3
        if (omega >= oL3) {
          nr = 1;
          return overcontact;
        } 
      }
    } else if (oL1 < oL2 && oL1 < oL3) { // omega(L1) < omega(L_{2,3})
      
      if (oL2 >= oL3) {
        if (omega >= oL3) {
          nr = 1;
          return solo_right;
        } 
      } else { // oL2 < oL3
        if (omega >= oL2) {
          nr = 1;
          return solo_left;
        } 
      }   
    } else if (oL2 > oL1 && oL1 > oL3) { // omega(L_2) > omega(L1) > omega(L_3)
      if (omega >= oL1) {
        nr = 1;
        return solo_right; 
      } 
    } else if (oL2 > oL1 && oL1 > oL3) {  // omega(L_3) > omega(L1) > omega(L_2)
      if (omega >= oL1) {
        nr = 1;
        return solo_left; 
      }  
    }

    nr = 0;
    return overflow;
  }
  
  /*
    Points of on the x-axis satisfying
    
      Omega(x,0,0) = Omega_0
    
    The Roche lobes are limited by these points.
    
    Input:
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      
      trimming: 
        if false : all solutions
        if true  : trims solutions which bounds Roche lobes,
                   there are even number bounds
        
    Output:
      p - x-values of the points on x-axis
  */
  template<class T> 
  void points_on_x_axis(
    std::vector<T> & points,
    const T & Omega0, 
    const T & q, 
    const T & F = 1, 
    const T & delta = 1,
    const bool & trimming = true){
    
    T fa = F*F*delta*delta*delta, // rescaled F^2
      rO = delta*Omega0,          // rescaled pontential 
      p = 1/q,
      b1 = fa*(1 + q),
      b2 = fa*(1 + p);
      
    points.clear();
    
    std::vector<T> roots;
    
    //  
    // left of origin: x < 0
    //
    {
      T a[5] = {2, 2*(1 + q - rO), 2*(q - rO), b1 + 2*q, b1};
      
      utils::solve_quartic(a, roots);
    
      for (auto && v : roots) if (v > 0) points.push_back(-delta*v);
    }
    
    //
    // center part: x in [0, delta]
    //
    {
      T a[5] = {-2, 2*(1 - q + rO), 2*(q - rO), -b1 - 2*q, b1};
    
      utils::solve_quartic(a, roots);
            
      for (auto && v : roots) if (0 < v && v < 1) points.push_back(delta*v);
                
    }
    
    //
    // right of center: x > delta
    //
    {
      T a[5] = {2, b2 - 2*p*(-1 + rO), -4 + 3*b2 - 2*p*rO, -2 + 3*b2, b2};
    
      utils::solve_quartic(a, roots);
      
      for (auto && v : roots) if (v > 0) points.push_back(delta*(1 + v));
    }
    
    std::sort(points.begin(), points.end());
    
    if (trimming) {
      
      if (points.size() > 1) {
         auto it = points.begin();
         if (*it < 0 && *(it + 1) < 0) points.erase(it);
      }
      
      if (points.size() > 1) {
        auto it = points.end()-1; 
        if (*it > delta && *(it-1) > delta) points.erase(it);
      }  
      
      if (points.size() != 0) {
        
        auto it = points.begin(), 
             it_end = points.end();
        
        unsigned len = (points.size() >> 1) << 1;
        
        if (points.size() != len) { 
          if (len == 0) 
            points.clear();
          else if (*it < 0) 
            points.erase(it + len, it_end); 
          else if (*(it - 1) > delta)
            points.erase(it, it_end - len);
        }
      }
    }
  }
  

  /*
    Calculate the upper and lowe limit of Roche lobes on x-axis
    satisfying
    
      Omega(x,0,0) = Omega_0

    Input:
      Omega0 - reference value of the potential
      choice :
        0 - left
        1 - right
        2 - overcontact        
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects

    Output:
      p - x-values of the points on x-axis
  */
  template<class T> 
  bool lobe_x_points(
    T xrange[2],
    int choice,
    const T & Omega0, 
    const T & q, 
    const T & F = 1, 
    const T & delta = 1,
    bool enable_checks = false
    ){
    
    T omega_crit[3], L_points[3],
      fa = F*F*delta*delta*delta, // rescaled F^2
      rO = delta*Omega0,          // rescaled pontential 
      p = 1/q,
      b1 = fa*(1 + q),
      b2 = fa*(1 + p);
      
    std::vector<T> roots;
    
    
    if (choice < 0 || choice > 2) return false;
     
    //
    //  left lobe
    //
    if (choice == 0) {

      if (enable_checks) { 
        // omega_crit[0] = Omega(L1), omega_crit[1] = Omega(L2)
        critical_potential(omega_crit, L_points, 1+2, q, F, delta);
       
        if (!(omega_crit[0] < Omega0 && omega_crit[1] < Omega0)) {
          std::cerr << "lobe_x_points::left lobe does not seem to exist\n";
          return false;
        }
      }
      
      // left side, x < 0
      T aL[5] = {2, 2*(1 + q - rO), 2*(q - rO), b1 + 2*q, b1};
      
      utils::solve_quartic(aL, roots);
      
      // grab smallest root positive
      for (auto && v : roots) 
        if (v > 0) { xrange[0] = - delta*v; break;}
      
      
      // central part
      T aC[5] = {-2, 2*(1 - q + rO), 2*(q - rO), -b1 - 2*q, b1};
    
      utils::solve_quartic(aC, roots);
     
      // grab the smallest/first root in [0,1]
      
      for (auto && v : roots) 
        if (0 < v && v < 1) {xrange[1] = delta*v; break;}
    }
    
    //
    //  right lobe
    //
    
    if (choice == 1) {
      
      if (enable_checks){
        
        // omega_crit[0] = Omega(L1), omega_crit[2] = Omega(L3)
        critical_potential(omega_crit, L_points, 1+4, q, F, delta);
       
        if (!(omega_crit[0] < Omega0 && omega_crit[2] < Omega0)) {
          std::cerr << "lobe_x_points::right lobe does not seem to exist\n";
          return false;
        }
      }

      // central part
      T aC[5] = {-2, 2*(1 - q + rO), 2*(q - rO), -b1 - 2*q, b1};
    
      utils::solve_quartic(aC, roots);
      
      // grab the largest/second root in [0,1] 
      int i = 0;     
      for (auto && v : roots) 
        if (0 < v && v < 1) if (i++ == 1) {xrange[0] = delta*v; break;}  
      
      // right side, x > 0
      T aR[5] = {2, b2 - 2*p*(-1 + rO), -4 + 3*b2 - 2*p*rO, -2 + 3*b2, b2};
    
      utils::solve_quartic(aR, roots);
      // grab smallest root positive 
      for (auto && v : roots) if (v > 0) {xrange[1] = delta*(1 + v); break;};
    }
    
    
    //
    // overcontact
    //
    
    if (choice == 2){
      
      if (enable_checks) {
        // omega_crit[0] = Omega(L1), omega_crit[1] = Omega(L2), omega_crit[2] = Omega(L3)
        critical_potential(omega_crit, L_points, 1+2+4, q, F, delta);
       
        if (!(Omega0 < omega_crit[0] && Omega0 > omega_crit[1] && Omega0 > omega_crit[2])) {
          std::cerr << "lobe_x_points::overcontact lobe does not seem to exist\n";
          return false;
        }
      }
      
      // left side, x < 0
      T aL[5] = {2, 2*(1 + q - rO), 2*(q - rO), b1 + 2*q, b1};
      
      utils::solve_quartic(aL, roots);
      
      // grab smallest root positive
      for (auto && v : roots) 
        if (v > 0) { xrange[0] = -delta*v; break;} 
      
      
      T aR[5] = {2, b2 - 2*p*(-1 + rO), -4 + 3*b2 - 2*p*rO, -2 + 3*b2, b2};
    
      utils::solve_quartic(aR, roots);
      
      // grab smallest root positive
      for (auto && v : roots) 
        if (v > 0) {xrange[1] = delta*(1 + v); break;};
    }

    return true;
  }

} // namespace gen_roche

#endif
