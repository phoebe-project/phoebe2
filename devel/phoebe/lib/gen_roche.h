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

// General rotines
#include "utils/utils.h"                  // Misc routines (sqr, solving poly eq,..)

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
    Rescaled potential on x - axis
    
      Tilde{Omega} = delta Omega 
                 = 1/|t| + q(1/|t-1| -t) + 1/2 b t^2
    Input:
      t = x/delta
      q - mass ratio M2/M1
      b = F^2 delta^3 (1+q) - parameter of the rescaled potential
      choice:
        1st bit set: v[0] = value  
        2nd bit set: v[1] = dTilde{Omega}/dt 
        3rd bit set: v[2] = d^2Tilde{Omega}/dt^2 
    Output:
      v = {value, dTilde{Omega}/dt, d^2Tilde{Omega}/dt^2}
   
  */ 

  template<class T> 
  void rescaled_potential_on_x_axis(
    T *v,
    unsigned choice,
    const T & t,
    const T & q,
    const T & b)
  {      
    int 
      s1 = (t > 0 ? 1 : -1),
      s2 = (t > 1 ? 1 : -1);
      
    T
      t1 = std::abs(t),
      t2 = std::abs(t - 1);
    
    if ( (choice & 1u) == 1u)
      v[0] = 1/t1 + q*(1/t2 - t) + b*t*t/2;
    
    if ( (choice & 2u) == 2u)
      v[1] = -s1/(t1*t1) - q*(s2/(t2*t2) + 1) + b*t;
    
    if ( (choice & 4u) == 4u)
      v[2] = 2*(1/(t1*t1*t1) + q/(t2*t2*t2)) + b;
  }
  
  /*
    Finding range on x-axis for each lobe separately
    
      w = delta Omega
      q = M2/M1
      b = delta^3 F^2(1 + q)
    
    by solving 
      
      Tilde{Omega} = w = 1/|t| + q(1/|t-1| -t) + 1/2 b t'^2
    
    Solving: 
      1/t + 1/2 b t^2 + q (t + 1/(1 + t)) = w
      solution = -t
  */ 
  
  template <class T>
  T left_lobe_left_xborder(
    const T & w, 
    const T & q, 
    const T & b,
    const T & tL2
  ) {

    if (w > 100 && w > 2*q){  // w->infty
      
      const int iter_max = 10;
      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      T q2 = q*q,
        s = 1/w,
        a[8] = {1, q, q2, b/2 + q*(1 + q2), 
          q*(-1 + 2*b + q*(4 + q2)),
          q*(1 + q*(-5 + 5*b + q*(10 + q2))), 
          b*(3*b/4 + q*(3 + 10*q2)) + q*(-1 + q*(9 + q*(-15 + q*(20 + q2)))),
          q*(1 + b*(-3.5 + 21*b/4) + q*(-14 + 21*b + q*(42 + q*(-35 + 35*b/2 + q*(35 + q2)))))
        },
        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));
        
       int it = 0;
       
       T dt, v[2]; 
       
       t = -t;
       
       do {
          // note: working with 
          rescaled_potential_on_x_axis(v, 3, t, q, b);
          
          t -= (dt = (v[0] - w)/v[1]); 
          
          //std::cerr << it << '\t' << t << '\t' << dt << '\n';
          
       } while ( std::abs(dt) > eps*std::abs(t) + min && ++it < iter_max);
      
      if (!(it < iter_max))
        std::cerr << "left_lobe_left_xborder::slow convergence\n";

      return t;
    }
    
    std::vector<T> roots;

    T a[5] = {2, 2*(1 + q - w), 2*(q - w), b + 2*q, b};
    
    utils::solve_quartic(a, roots);
    
    // grab smallest root positive
    for (auto && v : roots) if (v > 0) return  -v;
    
    return std::nan("");
  }

  /* 
    Solving:

      q (1/(1 - t) - t) + 1/t + 1/2 b t^2 = w
      
      solution = t
  */

  template <class T>
  T left_lobe_right_xborder(
    const T & w, 
    const T & q, 
    const T & b,
    const T & tL1
  ) {
  
    if (w > 100 && w > 2*q){  // w->infty
      
      const int iter_max = 10;
      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      T q2 = q*q,
        s = 1/w,
        a[8] = {1, q, q2, b/2 + q*(1 + q2), 
          q*(1 + 2*b + q*(4 + q2)), 
          q*(1 + q*(5 + 5*b + q*(10 + q2))), 
          b*(3*b/4 + q*(3 + 10*q2)) + q*(1 + q*(9 + q*(15 + q*(20 + q2)))),
          q*(1 + b*(3.5 + 21*b/4) + q*(14 + 21*b + q*(42 + q*(35 + 35*b/2 + q*(35 + q2)))))
        },
        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));
        
       int it = 0;
       
       T dt, v[2]; 

       do {
          rescaled_potential_on_x_axis(v, 3, t, q, b);
          
          t -= (dt = (v[0] - w)/v[1]); 
          
          //std::cerr << it << '\t' << t << '\t' << dt << '\n';
          
       } while (std::abs(dt) > eps*std::abs(t) + min && ++it < iter_max);
      
      if (!(it < iter_max))
        std::cerr << "left_lobe_right_xborder::slow convergence\n";

      return t;
    }
    
    std::vector<T> roots;

    T a[5] = {2, 2*(-1 + q - w), 2*(-q + w), b + 2*q, -b};
    
    //for (int i = 0; i < 5; ++i) std::cout << "a=" << a[i] << '\n';
    utils::solve_quartic(a, roots);

    // grab the smallest/first root in [0,1]

    for (auto && v : roots) if (0 < v && v < 1) return v;
    
    return std::nan("");
  }
  
  /* 
    Solving:
    
      p = 1/q,  c = p b, r = (w + q - b/2)/q = p(w - b/2) + 1
     
      p/(1 - t) + 1/t + t(1 - c) + 1/2 c t^2 = r
    
      solution = 1 - t
  */
  template <class T>
  T right_lobe_left_xborder(
    const T & w, 
    const T & q, 
    const T & b,
    const T & tL1
  ) {
    
     T p = 1/q,
       c = p*b,
       r = p*(w - b/2) + 1;
     
     if (r > 100 && r > 2*p){  // w->infty
      
      const int iter_max = 10;
      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      T p2 = p*p,
        s = 1/r,
        a[8] = {1, p, 1 - c + p*(1 + p), c*(0.5 - 3*p) + p*(4 + p*(3 + p)),
          2 + c*(-4 + 2*c + (-2 - 6*p)*p) + p*(5 + p*(12 + p*(6 + p))),
          c*(2.5 + c*(-2.5 + 10*p) + p*(-22.5 + (-15 - 10*p)*p)) + p*(16 + p*(30 + p*(30 + p*(10 + p)))),
          5 + c*(-15 + c*(15.75 - 5*c + 30*p2) + p*(-18 + p*(-90 + (-50 - 15*p)*p))) + p*(22 + p*(90 + p*(110 + p*(65 + p*(15 + p))))),
          c*(10.5 + c*(-21 + c*(10.5 - 35*p) + p*(110.25 + p*(52.5 + 70*p))) + p*(-129.5 + p*(-210 + p*(-297.5 + (-122.5 - 21*p)*p)))) + p*(64 + p*(210 + p*(385 + p*(315 + p*(126 + p*(21 + p))))))
        },
        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));
        
       int it = 0;
       
       T dt, v[2]; 
       
       t = 1 - t;
       
       do {
          // note: working with 
          rescaled_potential_on_x_axis(v, 3, t, q, b);
          
          t -= (dt = (v[0] - w)/v[1]); 
          
          //std::cerr << it << '\t' << t << '\t' << dt << '\n';
          
       } while ( std::abs(dt) > eps*std::abs(t) + min && ++it < iter_max);
      
      if (!(it < iter_max))
        std::cerr << "right_lobe_left_xborder::slow convergence\n";

      return t;
    }
    
  
    std::vector<T> roots;
    
    T a[5] = {2, 2*(-1 + p - r), 2*(1 - c + r), 2 + 3*c, -c};
   
    utils::solve_quartic(a, roots);
      
    // grab the smallest root in [0,1] 
    for (auto && v : roots) if (0 < v && v < 1) return 1 - v;
    
    return std::nan("");
  }

  /* 
    Solving:
      p = 1/q,  c = p b, r = p(w - b/2) + 1,
    
      1/t + (-1 + c) t + (c t^2)/2 + p/(1 + t)  = r
      
      solution = 1 + t 
  */

  template <class T>
  T right_lobe_right_xborder(
    const T & w, 
    const T & q, 
    const T & b,
    const T & tL3
  ) {
      
     T p = 1/q,
       c = p*b,
       r = p*(w - b/2) + 1;
     
     if (r > 100 && r > 2*p){  // w->infty
      
      const int iter_max = 10;
      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      T p2 = p*p,
        s = 1/r,
        a[8] = {1, p, -1 + c + (-1 + p)*p, c*(0.5 + 3*p) + p*(-2 + (-3 + p)*p),
          2 + p*(3 + (-6 + p)*p2) + c*(-4 + 2*c + p*(-2 + 6*p)), 
          c*(-2.5 + c*(2.5 + 10*p) + p*(-17.5 + p*(-15 + 10*p))) + p*(6 + p*(10 + p*(10 + (-10 + p)*p))),
          -5 + p*(-10 + p2*(10 + p*(35 + (-15 + p)*p))) + c*(15 + c*(-14.25 + 5*c + 30*p2) + p*(12 + p*(-30 + p*(-50 + 15*p)))),
          c*(10.5 + c*(-21 + c*(10.5 + 35*p) + p*(-99.75 + p*(-52.5 + 70*p))) + p*(87.5 + p*(105 + p*(17.5 + p*(-122.5 + 21*p))))) + p*(-20 + p*(-42 + p*(-35 + p*(-35 + p*(84 + (-21 + p)*p)))))
        },
        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));
        
       int it = 0;
       
       T dt, v[2]; 
       
       t = 1 + t;
       
       do {
          // note: working with 
          rescaled_potential_on_x_axis(v, 3, t, q, b);
          
          t -= (dt = (v[0] - w)/v[1]); 
          
          //std::cerr << it << '\t' << t << '\t' << dt << '\n';
          
       } while ( std::abs(dt) > eps*std::abs(t) + min && ++it < iter_max);
      
      if (!(it < iter_max))
        std::cerr << "right_lobe_right_xborder::slow convergence\n";

      return t;
    }
    
  
    std::vector<T> roots;
    
    T a[5] = {2, 2*(1 + p - r), 2*(-1 + c - r), -2 + 3*c, c};
   
    utils::solve_quartic(a, roots);
      
    // grab the smallest root in [0,1] 
    for (auto && v : roots) if (0 < v && v < 1) return 1 + v;
      
    return std::nan("");
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
    
    T omega[3], L[3],
      w = Omega0*delta,                   // rescaled potential 
      b = F*F*delta*delta*delta*(1 + q);  // rescaled F^2
      
    
    if (choice < 0 || choice > 2) return false;
     
    //
    //  left lobe
    //
    
    if (choice == 0) {
      
      // omega[0] = Omega(L1), omega[1] = Omega(L2)
      critical_potential(omega, L, 1+2, q, F, delta);
 
      if (enable_checks && !(omega[0] < Omega0 && omega[1] < Omega0)) {
        
        std::cerr 
          << "lobe_x_points::left lobe does not seem to exist\n"
          << "omegaL1=" << omega[0] << " omegaL2=" << omega[1] << '\n';
        return false;
      }
      
      xrange[0] = delta*left_lobe_left_xborder(w, q, b, L[1]/delta);
      xrange[1] = delta*left_lobe_right_xborder(w, q, b, L[0]/delta);
    }
    
    //
    //  right lobe
    //
    
    if (choice == 1) {
      
      // omega[0] = Omega(L1), omega[2] = Omega(L3)
      critical_potential(omega, L, 1+4, q, F, delta);
    
      if (enable_checks && !(omega[0] < Omega0 && omega[2] < Omega0)) {
        std::cerr 
          << "lobe_x_points::right lobe does not seem to exist\n"
          << "omegaL1=" << omega[0] << " omegaL3=" << omega[2] << '\n';
        return false;      
      }
      
      xrange[0] = delta*right_lobe_left_xborder(w, q, b, L[0]/delta);
      xrange[1] = delta*right_lobe_right_xborder(w, q, b, L[2]/delta);  
    }
    
    //
    // overcontact
    //
    
    if (choice == 2){
      
      // omega[0] = Omega(L1), omega[1] = Omega(L2), omega[2] = Omega(L3)
      critical_potential(omega, L, 1+2+4, q, F, delta);
       
      if (enable_checks && 
          !(Omega0 < omega[0] && Omega0 > omega[1] && Omega0 > omega[2])) {
        std::cerr 
          << "lobe_x_points::overcontact lobe does not seem to exist\n"
          << "omegaL1=" << omega[0] << " omegaL2=" << omega[1] << " omegaL3=" << omega[2] << '\n';
        return false;
      }
      
      xrange[0] = delta*left_lobe_left_xborder(w, q, b, L[1]/delta);
      xrange[1] = delta*right_lobe_right_xborder(w, q, b, L[2]/delta);
    }


    if (std::isnan(xrange[0])) {
      std::cerr << "lobe_x_points::problems with left boundary\n";
      return false;
    }  

    if (std::isnan(xrange[1])) {
      std::cerr << "lobe_x_points::problems with right boundary\n";
      return false;
    }

    return true;
  }

} // namespace gen_roche

#endif
