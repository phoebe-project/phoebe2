#if !defined(__gen_roche_h)
#define __gen_roche_h

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#include "lagrange/gen_roche_lagrange.h"
#include "utils/utils.h"
#include "volume/volume.h"

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
    
    T d2 = delta*delta,
      x2 = x*x; 
      
    return 1/std::abs(x) + q*(1/std::abs(delta - x) - x/d2) + F*F*(1 + q)*x2/2;
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
      potential_on_x_axis(langrange_point_L1(q, F, delta), q, F, delta);
    // note: x < 0
    omega_crit[1] = 
      potential_on_x_axis(langrange_point_L2(q, F, delta), q, F, delta);
    // note : x > delta
    omega_crit[2] = 
      potential_on_x_axis(langrange_point_L3(q, F, delta), q, F, delta);
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
        
        int len = (points.size() >> 1) << 1;
        
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

} // namespace gen_roche

#endif
