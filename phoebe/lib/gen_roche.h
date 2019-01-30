#pragma once

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
#include "utils.h"                  // Misc routines (sqr, solving poly eq,..)

// Definition of bodies
#include "bodies.h"

// Roche specific routines and part of gen_roche namespace

// Lagrange fixed points L1, L2, L3

#include "gen_roche_lagrange_L1.h"
#include "gen_roche_lagrange_L2.h"
#include "gen_roche_lagrange_L3.h"

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
    if ((choice & 1U) == 1U )
      omega_crit[0] =
        potential_on_x_axis(
          L_points[0] = lagrange_point_L1(q, F, delta),
          q, F, delta);

    // note: x < 0
    if ((choice & 2U) == 2U)
      omega_crit[1] =
        potential_on_x_axis(
          L_points[1] = lagrange_point_L2(q, F, delta),
          q, F, delta);

    // note : x > delta
    if ((choice & 4U) == 4U)
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

      } while (std::abs(dh) > eps*std::abs(h) + min && (++it) < iter_max);

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

      Omega(delta, 0,z) = Omega0

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
    contact_binary,
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
          return contact_binary;
        }
      } else { // oL2 < oL3
        if (omega >= oL3) {
          nr = 1;
          return contact_binary;
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

  template <class T, class F>
  T polish_xborder(
    const T & w1,
    const T & q1,
    const T & b1,
    const T & t1) {

    const int max_iter = 10;
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    int it = 0;

    F t = t1, w = w1, q = q1, b = b1, dt, v[2];

    do {
      rescaled_potential_on_x_axis(v, 3, t, q, b);

      t -= (dt = (v[0] - w)/v[1]);

    } while (std::abs(dt) > eps*std::abs(t) + min && ++it < max_iter);

    if (it >= max_iter){
      std::cerr.precision(std::numeric_limits<F>::digits10+1);
      std::cerr
        << "polish_xborder:" << '\n'
        <<  "w=" << w << " q=" << q << " b=" << b << " t=" << t
        << std::endl;
    }

    return T(t);
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
    const T & b
  ) {

    const char *fname = "left_lobe_left_xborder";

    const int max_iter = 100;
    const T eps = 2*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Is solution is near to Lagrange point?
    //

    T l = lagrange_point_L2(q, std::sqrt(b/(1 + q)), 1.);

    if (q*(1/(1 - l) - l) - 1/l + b*l*l/2 == w) return l;

    //
    // Cases away from Lagrange point
    //

    T t;

    if (w > 100) {

      if (2*q < w){  // w->infty

        T q2 = q*q,
          s = 1/w,
          a[8] = {1, q, q2, b/2 + q*(1 + q2),
            q*(-1 + 2*b + q*(4 + q2)),
            q*(1 + q*(-5 + 5*b + q*(10 + q2))),
            b*(3*b/4 + q*(3 + 10*q2)) + q*(-1 + q*(9 + q*(-15 + q*(20 + q2)))),
            q*(1 + b*(-3.5 + 21*b/4) + q*(-14 + 21*b + q*(42 + q*(-35 + 35*b/2 + q*(35 + q2)))))
          };

        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));

        t = -t;
      } else if (q < w) { // w->infty, q ~ w

        T a = b/(1 + q),
          s = 1/w,
          f = q*s,
          f1 = 1 - f, f12 = f1*f1, f13 = f12*f1,
          s1 = 1/(w - q),

          // denominator
          D[8] = {1, 1, 2*f1, 2*f1, 4*f12, 2*f12, 4*f13, 4*f13},

          // numerator
          N[8] = {1, 0, (-2 - a)*f, -a + (2 + a)*f, f*(4 + (8 + a*(12 + 3*a))*f),
            f*(-2 + a*(6 + 3*a) + (-12 + (-13 - 3*a)*a)*f),
            -3*a*a + f*(-4 + a*(14 + 9*a) + f*(-40 + (-44 - 9*a)*a + (-4 + a*(-42 + (-33 - 6*a)*a))*f)),
            f*(4 - 16*a + f*(64 + a*(-22 + (-72 - 18*a)*a) + (112 + a*(218 + a*(117 + 18*a)))*f))},
          C[8];

        for (int i = 0; i < 8; ++i) C[i] = N[i]/D[i];

        t = s/f1*(C[0] + s1*(C[1] + s1*(C[2] + s1*(C[3] + s1*(C[4] + s1*(C[5] + s1*(C[6] + s1*C[7])))))));

        t = -t;
      }

      return polish_xborder<T,long double>(w, q, b, t);
    }

    const int method = 0;

    if (method == 0) {  // Bisection on [-|l|,0]

      int it = 0;

      long double f, x[2] = {l, 0};

      do {
        t = (x[0] + x[1])/2;

        f = q*(1/(1 - t) - t) - 1/t + b*t*t/2 - w;


        if (f == 0) return t;

        if (f > 0) x[1] = t; else x[0] = t;

      } while (std::abs(x[1] - x[0]) > eps*std::max(std::abs(x[0]), std::abs(x[1])) + min && ++it < max_iter);

      if (it >= max_iter)
        std::cerr
          << fname << "::too many iterations\n"
          << "x0=" << x[0] << " x1=" << x[1] << " l=" << l << '\n'
          << "w=" << w << " q=" << q << " b=" << b  << '\n';
      else
        return t;

    } else {  // Solving general quartic eq.

      std::vector<long double> roots;

      long double a[5] = {2, 2*(1 + q - w), 2*(q - w), b + 2*q, b};

      utils::solve_quartic(a, roots);

      // grab smallest root positive
      for (auto && v : roots) if (v > 0) return  -v;

    }

    return std::numeric_limits<T>::quiet_NaN();
  }

  /*
    Solving:

      q (1/(1 - t) - t) + 1/t + 1/2 b t^2 = w

    with
      w = delta Omega
      q = M2/M1
      b = delta^3 F^2(1 + q)

    Return:
      t
  */
  //#define DEBUG
  template <class T>
  T left_lobe_right_xborder(
    const T & w,
    const T & q,
    const T & b
  ) {

    const char *fname = "left_lobe_right_xborder";

    const int max_iter = 100;
    const T eps = 2*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Is solution is near to Lagrange point?
    //

    T l = lagrange_point_L1(q, std::sqrt(b/(1 + q)), 1.), t = l;

    if (q*(1/(1 - t) - t) + 1/t + b*t*t/2 == w) return t;

    //
    // Cases away from Lagrange point
    //

    if (w > 100) {  // w->infty

      if (2*q < w){

        T q2 = q*q,
          s = 1/w,
          a[8] = {1, q, q2, b/2 + q*(1 + q2),
            q*(1 + 2*b + q*(4 + q2)),
            q*(1 + q*(5 + 5*b + q*(10 + q2))),
            b*(3*b/4 + q*(3 + 10*q2)) + q*(1 + q*(9 + q*(15 + q*(20 + q2)))),
            q*(1 + b*(3.5 + 21*b/4) + q*(14 + 21*b + q*(42 + q*(35 + 35*b/2 + q*(35 + q2)))))
          };

        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));

      } else if (q < w) {

        T a = b/(1 + q),
          s = 1/w,
          f = q*s,
          f1 = 1 - f, f12 = f1*f1, f13 = f12*f1,
          s1 = 1/(w - q),

          // denominator
          D[8] = {1, 1, 2*f1, 2*f1, 4*f12, 2*f12, 4*f13, 4*f13},

          // numerator
          N[8] ={1, 0, (-2 - a)*f, -a + (-2 + a)*f, f*(4 + (8 + a*(12 + 3*a))*f),
            f*(2 + a*(6 + 3*a) + (12 + (1 - 3*a)*a)*f),
            -3*a*a + f*(-4 + a*(-14 + 9*a) + f*(-40 + (12 - 9*a)*a + (-4 + a*(-70 + (-33 - 6*a)*a))*f)),
            f*(-4 - 16*a + f*(-64 + a*(-58 + (-72 - 18*a)*a) + (-112 + a*(-106 + a*(27 + 18*a)))*f))},
          C[8];

        for (int i = 0; i < 8; ++i) C[i] = N[i]/D[i];

        t = s/f1*(C[0] + s1*(C[1] + s1*(C[2] + s1*(C[3] + s1*(C[4] + s1*(C[5] + s1*(C[6] + s1*C[7])))))));
      }

      return polish_xborder<T,long double>(w, q, b, t);
    }

    const int method  = 0;

    if (method  == 0) { // Bisection on [0,l]

      int it = 0;

      long double f, x[2] = {0, l};

      do {
        t = (x[0] + x[1])/2;

        f = q*(1/(1 - t) - t) + 1/t + b*t*t/2 - w;

        if (f == 0) return t;

        if (f < 0) x[1] = t; else x[0] = t;

      } while (std::abs(x[1] - x[0]) > eps*std::max(x[0], x[1]) + min && ++it < max_iter );

      if (it >= max_iter)
        std::cerr
          << fname << "::too many iterations\n"
          << "x0=" << x[0] << " x1=" << x[1] << " l=" << l << '\n'
          << "w=" << w << " q=" << q << " b=" << b  << '\n';
      else
        return t;

    } else { // Solving general quartic eq.

      std::vector<long double> roots;

      long double a[5] = {2, 2*(-1 + q - w), 2*(-q + w), b + 2*q, -b};

      #if defined(DEBUG)
      std::cerr.precision(16);
      for (int i = 0; i < 5; ++i) std::cerr << "a[" << i << "]=" << a[i] << '\n';
      #endif

      utils::solve_quartic(a, roots);

      for (auto && v : roots) if (0 < v && v < 1) return v;
    }

    return std::numeric_limits<T>::quiet_NaN();
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif

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
    const T & b
  ) {

    const char *fname = "right_lobe_left_xborder";

    const int max_iter = 100;
    const T eps = 2*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Is solution is near to Lagrange point?
    //

    T l = lagrange_point_L1(q, std::sqrt(b/(1 + q)), 1.), t = l;

    if (q*(1/(1 - t) - t) + 1/t + b*t*t/2 == w) return t;

    //
    // Cases away from Lagrange point
    //

    T p = 1/q,
      c = p*b,
      r = p*(w - b/2) + 1;

    if (r > 100 && r > 2*p){  // w->infty
      T p2 = p*p,
        s = 1/r,
        a[8] = {1, p, 1 - c + p*(1 + p), c*(0.5 - 3*p) + p*(4 + p*(3 + p)),
          2 + c*(-4 + 2*c + (-2 - 6*p)*p) + p*(5 + p*(12 + p*(6 + p))),
          c*(2.5 + c*(-2.5 + 10*p) + p*(-22.5 + (-15 - 10*p)*p)) + p*(16 + p*(30 + p*(30 + p*(10 + p)))),
          5 + c*(-15 + c*(15.75 - 5*c + 30*p2) + p*(-18 + p*(-90 + (-50 - 15*p)*p))) + p*(22 + p*(90 + p*(110 + p*(65 + p*(15 + p))))),
          c*(10.5 + c*(-21 + c*(10.5 - 35*p) + p*(110.25 + p*(52.5 + 70*p))) + p*(-129.5 + p*(-210 + p*(-297.5 + (-122.5 - 21*p)*p)))) + p*(64 + p*(210 + p*(385 + p*(315 + p*(126 + p*(21 + p))))))
        },
        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));

      return polish_xborder<T,long double>(w, q, b, 1 - t);
    }

    const int method = 0;

    if (method == 0) {  // Bisection on [l, 1]

      int it = 0;

      long double f, x[2] = {l, 1};

      do {
        t = (x[0] + x[1])/2;

        f = q*(1/(1 - t) - t) + 1/t + b*t*t/2 - w;

        if (f == 0) return t;

        if (f < 0) x[0] = t; else x[1] = t;

      } while (std::abs(x[1] - x[0]) > eps*std::max(x[0], x[1]) + min && ++it < max_iter);

      if (it >= max_iter)
        std::cerr
          << fname << "::too many iterations\n"
          << "x0=" << x[0] << " x1=" << x[1] << " l=" << l << '\n'
          << "w=" << w << " q=" << q << " b=" << b  << '\n';
      else
        return t;

    } else { // Solving general quartic eq.

      std::vector<long double> roots;

      long double a[5] = {2, 2*(-1 + p - r), 2*(1 - c + r), 2 + 3*c, -c};

      utils::solve_quartic(a, roots);

      // grab the smallest root in [0,1]
      for (auto && v : roots) if (0 < v && v < 1) return 1 - v;
    }

    return std::numeric_limits<T>::quiet_NaN();
  }

  /*
    Solving:
      a) p = 1/q,  c = p b, r = p(w - b/2) + 1,

      1/t + (-1 + c) t + (c t^2)/2 + p/(1 + t)  = r

      b)
      1/(t + 1) + q (1/t - t - 1) + 1/2 a (1 + q) (t + 1)^2 = w

    with
      w = delta Omega
      q = M2/M1
      b = delta^3 F^2(1 + q)

    Return:
      1 + t
  */
// #define DEBUG
  template <class T>
  T right_lobe_right_xborder(
    const T & w,
    const T & q,
    const T & b
  ) {

    #if defined(DEBUG)
    std::cerr << fname << "::START" << std::endl;
    #endif

    const char *fname = "right_lobe_right_xborder";

    const int max_iter = 100;
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Check if it is on the Lagrange point
    //

    T l = lagrange_point_L3(q, std::sqrt(b/(1 + q)), 1.), t = l;

    if (q*(1/(t-1) - t) + 1/t + b*t*t/2 == w) return t;

    //
    // Cases away from Lagrange point
    //

    T p = 1/q,
      c = p*b,
      r = p*(w - b/2) + 1;

    if (r > 100 && r > 2*p){  // w->infty

      T p2 = p*p,
        s = 1/r,
        a[8] = {1, p, -1 + c + (-1 + p)*p, c*(0.5 + 3*p) + p*(-2 + (-3 + p)*p),
          2 + p*(3 + (-6 + p)*p2) + c*(-4 + 2*c + p*(-2 + 6*p)),
          c*(-2.5 + c*(2.5 + 10*p) + p*(-17.5 + p*(-15 + 10*p))) + p*(6 + p*(10 + p*(10 + (-10 + p)*p))),
          -5 + p*(-10 + p2*(10 + p*(35 + (-15 + p)*p))) + c*(15 + c*(-14.25 + 5*c + 30*p2) + p*(12 + p*(-30 + p*(-50 + 15*p)))),
          c*(10.5 + c*(-21 + c*(10.5 + 35*p) + p*(-99.75 + p*(-52.5 + 70*p))) + p*(87.5 + p*(105 + p*(17.5 + p*(-122.5 + 21*p))))) + p*(-20 + p*(-42 + p*(-35 + p*(-35 + p*(84 + (-21 + p)*p)))))
        },
        t = s*(a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*a[7])))))));

      return polish_xborder<T,long double>(w, q, b, 1 + t);
    }

    const int method = 0;

    if (method == 0) { // Bisection on [1, l]

      int it = 0;

      long double f, x[2] = {1, l};   // signs on boundary: +, -

      do {
        t = (x[0] + x[1])/2;

        f = q*(1/(t - 1) - t) + 1/t + b*t*t/2 - w;

        if (f == 0) return t;

        if (f > 0) x[0] = t; else x[1] = t;

      } while (std::abs(x[1] - x[0]) > eps*std::max(x[0], x[1]) + min && ++it < max_iter);

      if (it >= max_iter)
        std::cerr
          << fname << "::too many iterations\n"
          << "x0=" << x[0] << " x1=" << x[1] << " l=" << l << '\n'
          << "w=" << w << " q=" << q << " b=" << b  << '\n';
      else
        return t;

    } else { // Solving general quartic eq.

      std::vector<long double> roots;

      long double a[5] = {2, 2*(1 + p - r), 2*(-1 + c - r), -2 + 3*c, c};

      utils::solve_quartic(a, roots);

      // grab the smallest root in [0,1]
      for (auto && v : roots) if (0 < v && v < 1) return 1 + v;
    }

    #if defined(DEBUG)
    std::cerr << fname << "::END" << std::endl;
    #endif

    return std::numeric_limits<T>::quiet_NaN();
  }

  /*
    Calculate the upper and lower limit of Roche lobes on x-axis
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
  // #define DEBUG
  template<class T>
  bool lobe_xrange(
    T xrange[2],
    int choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    bool enable_checks = false
    ){

    const char *fname = "lobe_xrange";

    #if defined(DEBUG)
    std::cerr << fname << "::START" << std::endl;
    #endif

    T omega[3], L[3],
      w = Omega0*delta,                   // rescaled potential
      b = F*F*delta*delta*delta*(1 + q);  // rescaled F^2

    if (choice < 0 || choice > 2) return false;

    #if defined(DEBUG)
    std::cerr.precision(16);
    std::cerr << std::scientific
      << "choice=" << choice << '\n'
      << "w=" << w << '\n'
      << "q=" << q << '\n'
      << "b=" << b << std::endl;
    #endif


    //
    // Checking if we discuss semi-detached
    //

    if (choice != 2) {

      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();

      T w, L;

      critical_potential(&w, &L, 1, q, F, delta);

      if (std::abs(w - Omega0) < eps*std::max(std::abs(w), std::abs(Omega0)) + min) {

        if (choice == 0) {
          xrange[0] = delta*left_lobe_left_xborder(w, q, b);
          xrange[1] = L;
        } else {
          xrange[0] = L;
          xrange[1] = delta*right_lobe_left_xborder(w, q, b);
        }

        return true;
      }
    }


    //
    //  left lobe
    //

    if (choice == 0) {

      if (enable_checks) {

        // omega[0] = Omega(L1), omega[1] = Omega(L2)
        critical_potential(omega, L, 1+2, q, F, delta);

        if (!(omega[0] <= Omega0 && omega[1] <= Omega0)) {
          std::cerr
            << fname << "::left lobe does not seem to exist\n"
            << "omegaL1=" << omega[0] << " omegaL2=" << omega[1] << '\n'
            << "Omega0=" << Omega0 << " q=" << q << " F=" << F << " delta=" << delta << '\n';
          return false;
        }
      }

      xrange[0] = delta*left_lobe_left_xborder(w, q, b);
      xrange[1] = delta*left_lobe_right_xborder(w, q, b);
    }

    //
    //  right lobe
    //

    if (choice == 1) {

      if (enable_checks) {

        // omega[0] = Omega(L1), omega[2] = Omega(L3)
        critical_potential(omega, L, 1+4, q, F, delta);

        if (!(omega[0] <= Omega0 && omega[2] <= Omega0)) {
          std::cerr
            << fname << "::right lobe does not seem to exist\n"
            << "omegaL1=" << omega[0] << " omegaL3=" << omega[2] << '\n'
            << "Omega0=" << Omega0 << " q=" << q << " F=" << F << " delta=" << delta << '\n';
          return false;
        }
      }

      xrange[0] = delta*right_lobe_left_xborder(w, q, b);
      xrange[1] = delta*right_lobe_right_xborder(w, q, b);
    }

    //
    // overcontact
    //

    if (choice == 2){

      if (enable_checks) {

        // omega[0] = Omega(L1), omega[1] = Omega(L2), omega[2] = Omega(L3)
        critical_potential(omega, L, 1+2+4, q, F, delta);

        if (!(Omega0 <= omega[0] && Omega0 >= omega[1] && Omega0 >= omega[2])) {
          std::cerr
            << fname << "::contact binary lobe does not seem to exist\n"
            << "omegaL1=" << omega[0] << " omegaL2=" << omega[1] << " omegaL3=" << omega[2] << '\n'
            << "Omega0=" << Omega0 << " q=" << q << " F=" << F << " delta=" << delta << '\n';
          return false;
        }
      }

      xrange[0] = delta*left_lobe_left_xborder(w, q, b);
      xrange[1] = delta*right_lobe_right_xborder(w, q, b);
    }

    //std::cerr << "BU:" << xrange[0] << '\t' << xrange[1] << std::endl;

    if (std::isnan(xrange[0])) {
      std::cerr << fname << "::problems with left boundary\n";
      return false;
    }

    if (std::isnan(xrange[1])) {
      std::cerr << fname << "::problems with right boundary\n";
      return false;
    }

    #if defined(DEBUG)
    std::cerr << fname << "::END" << std::endl;
    #endif

    return true;
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif

  /*
    Find the point on the horizon around individual lobes.

    Input:
      view - direction of the view
      choice :
        0 - left -- around (0, 0, 0)
        1 - right -- around (delta, 0, 0)
        2 - overcontact  ???
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      max_iter - maximal number of iteration in search algorithm

    Output:
      p - point on the horizon
  */
  //#define DEBUG
  template<class T>
  bool point_on_horizon(
    T p[3],
    T view[3],
    int choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    int max_iter = 1000){

    //
    // Starting points
    //

    if (choice != 0 && choice != 1) {
      std::cerr
        << "point_on_horizon:: choices != 0,1 not supported yet\n";
      return false;
    }

    typedef T real;

    real r[3], v[3];

    for (int i = 0; i < 3; ++i) v[i] = view[i];

    #if 0

    if (choice == 0) {
      r[0] = 0;
      r[1] = 0;
      r[2] = 1e-6;
    } else {  // choice == 1
      r[0] = delta;
      r[1] = 0;
      r[2] = 1e-6;
    }
    #else

    // determine direction of initial point
    real fac;
    if (std::abs(v[0]) >= 0.5 || std::abs(v[1]) >= 0.5){
      fac = 1/std::hypot(v[0], v[1]);
      r[0] = fac*v[1];
      r[1] = -fac*v[0];
      r[2] = 0.0;
    } else {
      fac = 1/std::hypot(v[0], v[2]);
      r[0] = -fac*v[2];
      r[1] = 0.0;
      r[2] = fac*v[0];
    }

    // estimate of the radius of sphere that is
    // inside the Roche lobe

    real r0 = 0.5*(choice == 0 ?
                poleL(Omega0, q, F, delta):
                poleR(Omega0, q, F, delta));

    // rescaled the vector fo that the point on the sphere
    for (int i = 0; i < 3; ++i) r[i] *= r0;

    // shift if we discuss the right lobe
    if (choice == 1) r[0] += delta;
    #endif

    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Initialize body class
    //

    real params[] = {q, F, delta, Omega0};

    Tgen_roche<real> roche(params);

    // Solving both constrains at the same time
    //  Omega_0 - Omega(r) = 0
    //  grad(Omega) n = 0

    int i, it = 0;

    real
      dr_max, r_max, t, f, H[3][3],
      A[2][2], a[4], b[3], u[2], x[2];

    do {

      // a = {grad constrain, constrain}
      roche.grad(r, a);

      // get the hessian on the constrain
      roche.hessian(r, H);

      utils::dot3D(H, v, b);

      // define the matrix of direction that constrains change
      A[0][0] = utils::dot3D(a, a);
      A[0][1] = A[1][0] = utils::dot3D(a, b);
      A[1][1] = utils::dot3D(b, b);

      // negative remainder in that directions
      u[0] = -a[3];
      u[1] = -utils::dot3D(a, v);

      #if defined(DEBUG)
      std::cerr.precision(16);
      std::cerr << std::scientific;
      std::cerr
        << "Omega0=" << Omega0
        << "\nq=" << q
        << "\nF=" << F
        << "\nd=" << delta << '\n';

      std::cerr
        << "r=\n"
        << r[0] << '\t' << r[1] << '\t' << r[2] << '\n'
        << "H=\n"
        << H[0][0] << '\t' << H[0][1] << '\t' << H[0][2] << '\n'
        << H[1][0] << '\t' << H[1][1] << '\t' << H[1][2] << '\n'
        << H[2][0] << '\t' << H[2][1] << '\t' << H[2][2] << '\n'
        << "a=\n"
        << a[0] << '\t' << a[1] << '\t' << a[2] << '\t' << a[3] << '\n'
        << "v=\n"
        << v[0] << '\t' << v[1] << '\t' << v[2] << '\n'
        << "A=\n"
        << A[0][0] << '\t' << A[0][1] << '\n'
        << A[1][0] << '\t' << A[1][1] << '\n'
        << "u=\n"
        << u[0] << '\t' << u[1] << '\n';
      #endif

      // solving 2x2 system:
      //  A x = u
      // and
      //  making shifts
      //  calculating sizes for stopping criteria
      //

      dr_max = r_max = 0;

      if (utils::solve2D(A, u, x)){

        #if defined(DEBUG)
        std::cerr << "x=\n" << x[0] << '\t' << x[1] << '\n';
        #endif

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

      #if defined(DEBUG)
      std::cerr.precision(16);
      std::cerr << std::scientific;
      std::cerr << "point_on_horizon:\n";
      std::cerr << "rnew=" << r[0] << '\t' << r[1] << '\t' << r[2] << '\n';
      std::cerr << "it=" << it << " dr_max=" << dr_max << " r_max=" << r_max << '\n';
      #endif

    } while (dr_max > eps*r_max + min && ++it < max_iter);

    /*
    roche.grad(r, a);

    std::cout.precision(16);
    std::cout << std::scientific;
    std::cout << "dOmega=" << a[3] << "\n";

    T sum = 0;
    for (int i = 0; i < 3; ++i) sum += a[i]*view[i];

    std::cout << "dOmega=" << a[3] << " sum=" << sum << "\n";
    */

    for (int i = 0; i < 3; ++i) p[i] = r[i];

    return (it < max_iter);
  }
  //#undef DEBUG

  /*
    Starting point for meshing the Roche lobe is on x-axis.

    Input:

      choice :
        0 - left
        1 - right
        2 - overcontact
      Omega0 - reference value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects

    Output:
       r - position
       g - gradient
  */
  //#define DEBUG
  template <class T>
  bool meshing_start_point(
    T r[3],
    T g[3],
    int choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1
  ){

    //
    // Checking if we discuss semi-detached
    //

    if (choice != 2) {

      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();

      T w, L;

      critical_potential(&w, &L, 1, q, F, delta);

      if (std::abs(w - Omega0) < eps*std::max(std::abs(w), std::abs(Omega0)) + min) {

        g[0] = (choice == 0 ? eps : -eps);  // TODO: don't know if this a good solution
        g[1] = g[2] = 0;

        r[0] = L;
        r[1] = r[2] = 0;
        return true;
      }
    }

    T xrange[2];

    if (!lobe_xrange(xrange, choice, Omega0, q, F, delta, true)) return false;

    #if defined(DEBUG)
    std::cerr
      << "meshing_start_point: q=" << q
      << " F=" << F
      << " d=" << delta
      << " Omega=" << Omega0 << '\n';

    std::cerr << "xrange=" << xrange[0] << '\t' << xrange[1] << '\n';

    T omega[3], L[3];

    critical_potential(omega, L,1+2+4, q, F, delta);

    std::cerr << "L=" << L[0] << '\t' << L[1] << '\t' << L[2] << '\n';
    #endif

    T x;

    if (choice != 1)
      x = xrange[1];
    else
      x = xrange[0];

    T b = (1 + q)*F*F,
      f0 = 1/(delta*delta),
      x1 = x - delta;

    r[0] = x;
    r[1] = r[2] = 0;


    g[0] = - x*b
           + (x > 0 ? 1/(x*x) : (x < 0 ? -1/(x*x) : 0))
           + q*(f0 + (x1 > 0 ? 1/(x1*x1) : (x1 < 0 ? -1/(x1*x1) : 0)));

    g[1] = g[2] = 0;

    return true;
  }
  #if  defined(DEBUG)
  #undef DEBUG
  #endif

  /*
    Solving 2x2 system of nonlinear equations

      delta Omega(delta x,delta y, 0) = w
      d/dx Omega(delta x,delta y, 0) = 0

    via Newton-Raphson iteration.

    Input:
      u - initial estimate (starting position)
      w = Omega0*delta - rescaled value of the potential
      q - mass ratio M2/M1
      b = F^2 delta^3(1+q) - synchronicity parameter
      epsA - absolute precision aka accuracy
      epsR - relative precison aka precision
      max_iter - maximal number of iterations

    Output:
      u - solution
  */
  template <class T>
  bool lobe_ymax_internal(
    T u[2],
    const T & w,
    const T & q,
    const T & b,
    const T & epsA = 1e-12,
    const T & epsR = 1e-12,
    const int max_iter = 100
  ){

    int it = 0;

    T du_max = 0, u_max = 0;

    do {

      T x = u[0], x1 = x - 1, y = u[1], y2 = y*y,

        r12 = x*x + y*y, r1 = std::sqrt(r12),
        f1 = 1/r1, f12 = f1*f1, f13 = f12*f1, f15 = f13*f12,

        r22 = x1*x1 + y*y, r2 = std::sqrt(r22),
        f2 = 1/r2, f22 = f2*f2, f23 = f22*f2, f25 = f23*f22,

        // F - both equations that need to be solved (F=0)
        F[2] = {
          b*r12/2 + f1 + (f2 - x)*q - w,
          (b - f13)*x - q*(1 + f23*x1)
        },

        // derivative of F: M = [ [F1,x, F1,y],  [F2,x, F2,y]]
        M[2][2] = {
          {
            (b - f13)*x - q*(1 + f23*x1),
            (b - f13 - f23*q)*y
          },
          {
            b + 2*f13 - 3*f15*y2 + f23*q*(2 - 3*f22*y2),
            3*(f15*x + f25*q*x1)*y
          }
        },

        // inverse of M
        det = M[0][0]*M[1][1] - M[0][1]*M[1][0],
        N[2][2] = {{M[1][1], -M[0][1]}, {-M[1][0], M[0][0]}};

      // calculate the step in x and y
      T t;
      du_max = u_max = 0;
      for (int i = 0; i < 2; ++i) {
        u[i] += (t = -(N[i][0]*F[0] + N[i][1]*F[1])/det);

        t = std::abs(t);
        if (t > du_max) du_max = t;

        t = std::abs(u[i]);
        if (t > u_max) u_max = t;
      }

    } while (du_max > epsA + epsR*u_max && ++it < max_iter);


    return it < max_iter;
  }

  /*
    The point with the largest y components of the primary star:

      Omega(x,y,0) = Omega0

    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects

    Output: (optionaly)
      r = {x, y}

    Return:
      y: if the point is not found return -1
  */
  template <class T>
  T lobe_ybound_L(
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    T *r = 0
  ) {

    T w = Omega0*delta,
      b = (1 + q)*F*F*delta*delta*delta,
      u[2] = {0, 0.5*poleL(Omega0, q, F, delta)/delta};


    if (!lobe_ymax_internal(u, w, q, b)) {
      std::cerr << "lobe_ybound_L::Newton-Raphson did not converge\n";
      return -1;
    }

    if (r) for (int i = 0; i < 2; ++i) r[i] = delta*u[i];

    return delta*u[1];
  }

  /*
    The point with the largest y components of the secondary star:

      Omega(delta,y,x) = Omega0

    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects

    Output: (optionaly)
      r = {x, y}

    Return:
      y: if the point is not found return -1
  */
  template <class T>
  T lobe_ybound_R(
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    T *r = 0
  ) {

    T w = Omega0*delta,
      b = (1 + q)*F*F*delta*delta*delta,
      u[2] = {1, 0.5*poleR(Omega0, q, F, delta)/delta};

    if (!lobe_ymax_internal(u, w, q, b)) {
      std::cerr << "lobe_ybound_R::Newton-Raphson did not converge.\n";
      return -1;
    }

    if (r) for (int i = 0; i < 2; ++i) r[i] = delta*u[i];

    return delta*u[1];
  }


  //
  //  Gauss-Lagrange quadrature (GLQ) in [0,Pi/4]
  //

  template <class T, int N> struct glq {
    static const int n = N;
    static const T c2_phi[];        // cos()^2 of Gauss-Lagrange nodes x_i in [0, Pi/2]
    static const T s2c2_phi[];      // (sin(phi)cos(phi))^2
    static const T sc_phi[];        // sin()^2 of Gauss-Lagrange nodes x_i in [0, Pi/2]
    static const T weights[];   // Gauss-Lagrange weights * Pi
  };

  /*
    Gauss-Lagrange: double, N=10
  */
  template <>
  const double glq<double, 10>::c2_phi[]={0.999580064408504868392251519574,0.988810441236620370626267163745,0.93792975088501989705468221942,0.814698074016316269208833703789,0.615862835349225523334990208675,0.384137164650774476665009791325,0.185301925983683730791166296211,0.06207024911498010294531778058,0.011189558763379629373732836255,0.0004199355914951316077484804259};
  
  template <>
  const double glq<double, 10>::s2c2_phi[]=
  {0.0004197592455941272416985899940,0.011064352538060503513192603217,0.058217533289784414692374340457,0.150965122210421127009962838622,0.236575803384838256502140293303,0.236575803384838256502140293303,0.150965122210421127009962838622,0.058217533289784414692374340457,0.011064352538060503513192603217,0.0004197592455941272416985899940};
  
  template <>
  const double glq<double, 10>::sc_phi[]={0.020492330065054378968043365567,0.9997900101563852235169839221736,0.105780710733950116401421690636,0.9943894816602900746042204870849,0.249139015641830179695805173292,0.968467733528081924117038032924,0.430467102092231585658617450885,0.902606267436868770388218111745,0.619787999763446917824362871368,0.784769287975278453643040025427,0.784769287975278453643040025427,0.619787999763446917824362871368,0.902606267436868770388218111745,0.43046710209223158565861745088,0.968467733528081924117038032924,0.24913901564183017969580517329,0.994389481660290074604220487085,0.10578071073395011640142169064,0.9997900101563852235169839221736,0.02049233006505437896804336557};
  template <>
  const double glq<double, 10>::weights[]={0.209454205485130325204687978,0.46951526056054717731928526,0.68828010698191943974759705,0.845926347240509469003431984,0.92841673332168682718764111324,0.92841673332168682718764111324,0.845926347240509469003431984,0.68828010698191943974759705,0.46951526056054717731928526,0.209454205485130325204687978};
  
  /*
    Gauss-Lagrange: long double, N=10
  */
  template <>
  const long double glq<long double, 10>::c2_phi[]={0.999580064408504868392251519574L,0.988810441236620370626267163745L,0.93792975088501989705468221942L,0.814698074016316269208833703789L,0.615862835349225523334990208675L,0.384137164650774476665009791325L,0.185301925983683730791166296211L,0.06207024911498010294531778058L,0.011189558763379629373732836255L,0.0004199355914951316077484804259L};
  
  template <>
  const long double glq<long double, 10>::s2c2_phi[]=
  {0.0004197592455941272416985899940L,0.011064352538060503513192603217L,0.058217533289784414692374340457L,0.150965122210421127009962838622L,0.236575803384838256502140293303L,0.236575803384838256502140293303L,0.150965122210421127009962838622L,0.058217533289784414692374340457L,0.011064352538060503513192603217L,0.0004197592455941272416985899940L};
  
  template <>
  const long double glq<long double, 10>::sc_phi[]={0.020492330065054378968043365567L,0.9997900101563852235169839221736L,0.105780710733950116401421690636L,0.9943894816602900746042204870849L,0.249139015641830179695805173292L,0.968467733528081924117038032924L,0.430467102092231585658617450885L,0.902606267436868770388218111745L,0.619787999763446917824362871368L,0.784769287975278453643040025427L,0.784769287975278453643040025427L,0.619787999763446917824362871368L,0.902606267436868770388218111745L,0.43046710209223158565861745088L,0.968467733528081924117038032924L,0.24913901564183017969580517329L,0.994389481660290074604220487085L,0.10578071073395011640142169064L,0.9997900101563852235169839221736L,0.02049233006505437896804336557L};
  template <>
  const long double glq<long double, 10>::weights[]={0.209454205485130325204687978L,0.46951526056054717731928526L,0.68828010698191943974759705L,0.845926347240509469003431984L,0.92841673332168682718764111324L,0.92841673332168682718764111324L,0.845926347240509469003431984L,0.68828010698191943974759705L,0.46951526056054717731928526L,0.209454205485130325204687978L};

  /*
    Gauss-Lagrange: double, N=15
  */
  template <>
  const double glq<double, 15>::c2_phi[]={0.999911065396171632736007253518,0.997574886997681325232904530099,0.985854212895255771578898079594,0.953879938591329487419660590834,0.890692147886632646280933330332,0.790164039217058712341619458721,0.655400162984313217687403793532,0.5,0.344599837015686782312596206468,0.209835960782941287658380541279,0.109307852113367353719066669668,0.046120061408670512580339409166,0.014145787104744228421101920406,0.0024251130023186747670954699007,0.0000889346038283672639927464819};
  
  template <>
  const double glq<double, 15>::s2c2_phi[] ={0.0000889266944646091553555375073,0.0024192318292446596704491832554,0.013945683811931480320682058875,0.043993001344330973475114779410,0.097359645579729565862303734566,0.165804830345241213606518615509,0.225850789344448888056399859737,0.250000000000000000000000000000,0.225850789344448888056399859737,0.165804830345241213606518615509,0.097359645579729565862303734566,0.043993001344330973475114779410,0.013945683811931480320682058875,0.002419231829244659670449183255,0.0000889266944646091553555375073};
  
  template <>
  const double glq<double, 15>::sc_phi[]={0.009430514504965636496719714287,0.99995553170937138065232597719866,0.049245436360323529649147830490,0.9987867074594461808460040593058,0.118936063095867724375201909234,0.9929019150425966277731892615479,0.214755818102026080080238124559,0.976667772884582020710505864562,0.330617380234868102916887859767,0.943764879557738494192185923899,0.458078553070258123013941167236,0.888911716210928739911629494937,0.587026266035589571311728655155,0.809567886581670936826823880946,0.707106781186547524400844362105,0.707106781186547524400844362105,0.809567886581670936826823880946,0.587026266035589571311728655155,0.888911716210928739911629494937,0.458078553070258123013941167236,0.943764879557738494192185923899,0.33061738023486810291688785977,0.976667772884582020710505864562,0.21475581810202608008023812456,0.992901915042596627773189261548,0.11893606309586772437520190923,0.9987867074594461808460040593058,0.04924543636032352964914783049,0.9999555317093713806523259771987,0.00943051450496563649671971429};
  template <>
  const double glq<double, 15>::weights[]={0.0966141591290711189794453,0.22106145785079100848207,0.336650619784076362391549,0.4384742164293535129419684,0.5223501155128774575517226,0.584842030033819626900854873,0.6233908965445645786498187508,0.636418316610479145130430187624,0.6233908965445645786498187508,0.584842030033819626900854873,0.5223501155128774575517226,0.4384742164293535129419684,0.336650619784076362391549,0.22106145785079100848207,0.0966141591290711189794453};

   /*
    Gauss-Lagrange: long double, N=15
  */
  template <>
  const long double glq<long double, 15>::c2_phi[]={0.999911065396171632736007253518L,0.997574886997681325232904530099L,0.985854212895255771578898079594L,0.953879938591329487419660590834L,0.890692147886632646280933330332L,0.790164039217058712341619458721L,0.655400162984313217687403793532L,0.5L,0.344599837015686782312596206468L,0.209835960782941287658380541279L,0.109307852113367353719066669668L,0.046120061408670512580339409166L,0.014145787104744228421101920406L,0.0024251130023186747670954699007L,0.0000889346038283672639927464819L};
  template <>
  const long double glq<long double, 15>::s2c2_phi[] = {0.0000889266944646091553555375073L,0.0024192318292446596704491832554L,0.013945683811931480320682058875L,0.043993001344330973475114779410L,0.097359645579729565862303734566L,0.165804830345241213606518615509L,0.225850789344448888056399859737L,0.250000000000000000000000000000L,0.225850789344448888056399859737L,0.165804830345241213606518615509L,0.097359645579729565862303734566L,0.043993001344330973475114779410L,0.013945683811931480320682058875L,0.002419231829244659670449183255L,0.0000889266944646091553555375073L};
  
  template <>
  const long double glq<long double, 15>::sc_phi[]={0.009430514504965636496719714287L,0.99995553170937138065232597719866L,0.049245436360323529649147830490L,0.9987867074594461808460040593058L,0.118936063095867724375201909234L,0.9929019150425966277731892615479L,0.214755818102026080080238124559L,0.976667772884582020710505864562L,0.330617380234868102916887859767L,0.943764879557738494192185923899L,0.458078553070258123013941167236L,0.888911716210928739911629494937L,0.587026266035589571311728655155L,0.809567886581670936826823880946L,0.707106781186547524400844362105L,0.707106781186547524400844362105L,0.809567886581670936826823880946L,0.587026266035589571311728655155L,0.888911716210928739911629494937L,0.458078553070258123013941167236L,0.943764879557738494192185923899L,0.33061738023486810291688785977L,0.976667772884582020710505864562L,0.21475581810202608008023812456L,0.992901915042596627773189261548L,0.11893606309586772437520190923L,0.9987867074594461808460040593058L,0.04924543636032352964914783049L,0.9999555317093713806523259771987L,0.00943051450496563649671971429L};
  template <>
  const long double glq<long double, 15>::weights[]={0.0966141591290711189794453L,0.22106145785079100848207L,0.336650619784076362391549L,0.4384742164293535129419684L,0.5223501155128774575517226L,0.584842030033819626900854873L,0.6233908965445645786498187508L,0.636418316610479145130430187624L,0.6233908965445645786498187508L,0.584842030033819626900854873L,0.5223501155128774575517226L,0.4384742164293535129419684L,0.336650619784076362391549L,0.22106145785079100848207L,0.0966141591290711189794453L};

  /*
    Computing area of the surface, the volume and derivative of the
    volume w.r.t. potential of the Roche lobes that intersects x-axis

      {x0,0,0}  {x1,0,0}

    The range on x-axis is [x0, x1]. We assume
      if dir == +1  => radius(x0) = 0
      if dir == -1  => radisu(x1) = 0

    Input:
      x_bounds[2] = {x0,x1}
      Omega0 - value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      m - number of steps in x - direction
      choice - using as a mask
        1U  - area , stored in v[0]
        2U  - volume, stored in v[1]
        4U  - d(volume)/dOmega, stored in v[2]
      dir -
        +1 - integrating from x0 to x1
        -1 - integrating from x1 to x0
      polish - if true than after each RK step
                we perform reprojection onto surface

    Using: Integrating surface in cylindrical coordinate system
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in x direction

    Precision:
      At the default setup the relative precision should be better
      than 1e-4. NOT SURE

    Stability:
      It works for overcontact and well detached cases, but could be problematic
      d(polar radius)/dz is too small.

    Output:
      v[3] = {area, volume, d{volume}/dOmega}

    Ref:
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better
  */

  template<class T>
  void area_volume_directed_integration(
    T v[3],
    const unsigned & choice,
    const int & dir,
    T xrange[2],
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & delta = 1,
    const int & m = 1 << 14,
    const bool polish = false)
  {

    //
    // What is calculated
    //

    bool
      b_area = (choice & 1u) == 1u,
      b_vol  = (choice & 2u) == 2u,
      b_dvol = (choice & 4u) == 4u;

    if (!b_area && !b_vol && !b_dvol) return;
  
    using real = long double;

    using G = glq<real, 10>;
  
    const int dim = G::n + 3;
    
    real d2 = delta*delta,
      d3 = d2*delta,
      d4 = d2*d2,
      b = d3*F*F*(1 + q),
      omega = delta*Omega0;

    real
      y[dim], k[4][dim], w[G::n],
      t = (dir > 0 ? xrange[0]/delta : xrange[1]/delta),
      dt = (xrange[1] - xrange[0])/(m*delta);

    //
    // Integration over the surface with RK4
    //  Def:
    //    y = R^2, F = dR/dt, G = dR/dphi,
    //    x = delta t, c = cos(phi)^2
    //  Eq:
    //    y_i(t)' = F(t, y_i(t), c_i) i = 0, .., n -1
    //    A'(t) = delta^2 sqrt(R + G^2/(4R) + F^2/4) df dt
    //    V'(t) = delta^3 1/2 R df dx

    for (int i = 0; i < G::n; ++i) w[i] = dt*G::weights[i];

    dt *= dir;

    // init point
    for (int i = 0; i < dim; ++i) y[i] = 0;

    for (int j = 0; j < m; ++j){

      {
        // auxiliary variables
        real t1, f, g, f1, f2, s, s1, s2;

        //
        // RK iteration
        //

        // 1. step
        k[0][G::n] = k[0][G::n+1] = k[0][G::n+2] = 0;
        s1 = t*t, s2 = (t-1)*(t-1);
        for (int i = 0; i < G::n; ++i) {
          s = y[i];

          // f1 = (R + t^2)^(-3/2)
          f1 = s + s1;
          f1 = 1/(f1*std::sqrt(f1));

          // f1 = (R + (t-1)^2)^(-3/2)
          f2 = s + s2;
          f2 = 1/(f2*std::sqrt(f2));

          // k0 = -dx F_t/F_R
          g = 1/(b*G::c2_phi[i] - f1 - q*f2);
          f = (q*(1 + (t - 1)*f2) + t*(f1 - b))*g; // = 1/2 dR/dt, x = delta*t
          k[0][i] = dt*2*f;

          if (b_area) {
            g *= b; // = -(dR/dc)/R   Note: dR/dphi = -dR/dc*2*sqrt(c(1-c))
            k[0][G::n] += w[i]*std::sqrt(s*(1 + g*g*G::s2c2_phi[i]) + f*f); // = dA
          }

          if (b_vol)  k[0][G::n+1] += w[i]*s;  // = 2*dV
          if (b_dvol) k[0][G::n+2] += w[i]/(-f1 - q*f2 + b*G::c2_phi[i]); // d(dV/dOmega)/dt
        }


        // 2. step
        t1 = t + dt/2;

        k[1][G::n] = k[1][G::n+1] = k[1][G::n+2] = 0;
        s1 = t1*t1, s2 = (t1-1)*(t1-1);
        for (int i = 0; i < G::n; ++i){
          s = y[i] + 0.5*k[0][i];

          // f1 = (R + t1^2)^(-3/2)
          f1 = s + s1;
          f1 = 1/(f1*std::sqrt(f1));

          // f2 = (R + (t1-1)^2)^(-3/2)
          f2 = s + s2;
          f2 = 1/(f2*std::sqrt(f2));

          // k0 = -dx F_t/F_R
          g = 1/(b*G::c2_phi[i] - f1 - q*f2);
          f = (q*(1 + (t1 - 1)*f2) + t1*(f1 - b))*g; // = 1/2 dR/dt, x = delta*t
          k[1][i] = dt*2*f;

          if (b_area) {
            g *= b; // = -(dR/dc)/R   Note: dR/dphi = -dR/dc*2*sqrt(c(1-c))
            k[1][G::n] += w[i]*std::sqrt(s*(1 + g*g*G::s2c2_phi[i]) + f*f);  // = dA
          }

          if (b_vol) k[1][G::n+1] += w[i]*s; // = 2*dV
          if (b_dvol) k[1][G::n+2] += w[i]/(-f1 - q*f2 + b*G::c2_phi[i]); // d(dV/dOmega)/dt
        }

        // 3. step
        k[2][G::n] = k[2][G::n+1] = k[2][G::n+2] = 0;
        for (int i = 0; i < G::n; ++i) {
          s = y[i] + 0.5*k[1][i];

          // f1 = (R + t1^2)^(-3/2)
          f1 = s + s1;
          f1 = 1/(f1*std::sqrt(f1));

          // f2 = (R + (t1-1)^2)^(-3/2)
          f2 = s + s2;
          f2 = 1/(f2*std::sqrt(f2));

          // k0 = -dx F_t/F_R
          g = 1/(b*G::c2_phi[i] - f1 - q*f2);
          f = (q*(1 + (t1 - 1)*f2) + t1*(f1 - b))*g; // = 1/2 dR/dt, x = delta*t
          k[2][i] = dt*2*f;

          if (b_area) {
            g *= b; // = -(dR/dc)/R   Note: dR/dphi = -dR/dc*2*sqrt(c(1-c))
            k[2][G::n] += w[i]*std::sqrt(s*(1 + g*g*G::s2c2_phi[i]) + f*f); // = dA
          }
          if (b_vol) k[2][G::n+1] += w[i]*s; // = 2*dV
          if (b_dvol) k[2][G::n+2] += w[i]/(-f1 - q*f2 + b*G::c2_phi[i]); // d(dV/dOmega)/dt
        }

        // 4. step
        t1 = t + dt;
        k[3][G::n] = k[3][G::n+1] = k[3][G::n+2] = 0;
        s1 = t1*t1, s2 = (t1-1)*(t1-1);
        for (int i = 0; i < G::n; ++i){
          s = y[i] + k[2][i];

          // f1 = (R + t1^2)^(-3/2)
          f1 = s + s1;
          f1 = 1/(f1*std::sqrt(f1));

          // f2 = (R + (t1-1)^2)^(-3/2)
          f2 = s + s2;
          f2 = 1/(f2*std::sqrt(f2));

          // k0 = -dx F_t/F_R
          g = 1/(b*G::c2_phi[i] - f1 - q*f2);
          f = (q*(1 + (t1 - 1)*f2) + t1*(f1 - b))*g; // = 1/2 dR/dt, x = delta*t
          k[3][i] = dt*2*f;

          if (b_area) {
            g *= b; // = (dR/dc)/R   Note: dR/dphi = dR/dc*2*sqrt(c(1-c))
            k[3][G::n] += w[i]*std::sqrt(s*(1 + g*g*G::s2c2_phi[i]) + f*f); // = dA
          }

          if (b_vol) k[3][G::n+1] += w[i]*s; // = 2*dV
          if (b_dvol) k[3][G::n+2] += w[i]/(-f1 - q*f2 + b*G::c2_phi[i]); // d(dV/dOmega)/dt
        }

      }


      for (int i = 0; i < dim; ++i)
        y[i] += (k[0][i] + 2*(k[1][i] + k[2][i]) + k[3][i])/6;

      //std::cerr << "t =" << t << " y[G::n]=" << y[G::n] << '\n';

      t += dt;

      //
      // Polishing solutions with Newton-Rapson iteration
      //
      if (polish){

        const int it_max = 10;
        const real eps = 10*std::numeric_limits<T>::epsilon();
        const real min = 10*std::numeric_limits<T>::min();

        int it;

        real 
          s1 = t*t, s2 = (t - 1)*(t - 1),
          s, f, df, ds, f1, f2, g1, g2;

        for (int i = 0; i < G::n; ++i) {

          s = y[i];

          it = 0;
          do {

            // g1 = (R + (t-1)^2)^(-1/2), f1 = (R + t^2)^(-3/2)
            f1 = 1/(s + s1);
            g1 = std::sqrt(f1);
            f1 *= g1;

            // g2 = (R + (t-1)^2)^(-1/2),  f2 = (R + (t-1)^2)^(-3/2)
            f2 = 1/(s + s2);
            g2 = std::sqrt(f2);
            f2 *= g2;

            df = -(b*G::c2_phi[i] - f1 - q*f2)/2;                     // = dF/dR
            f = omega - q*(g2 - t) - g1 - b*(G::c2_phi[i]*s + s1)/2;  // =F

            s -= (ds = f/df);

          } while (std::abs(f) > eps &&
                  std::abs(ds) > eps*std::abs(s) + min &&
                  ++it < it_max);

          if (!(it < it_max))
            std::cerr << "Volume: Polishing did not succeed\n";

          y[i] = s;
        }
      }
    }

    if (b_area) v[0] = d2*y[G::n];
    if (b_vol)  v[1] = d3*y[G::n+1]/2;
    if (b_dvol) v[2] = d4*y[G::n+2];
  }
  
  /*
    A simplified version of directed integrated 
      
      void area_volume_directed_integration()
  */ 
  template<class T>
  void area_volume_integration(
    T v[3],
    const unsigned & choice,
    T xrange[2],
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & d = 1,
    const int & m = 1 << 14,
    const bool polish = false) {

    #if 1
    T
      xrange1[2] = {xrange[0], (xrange[0] + xrange[1])/2},
      xrange2[2] = {xrange1[1], xrange[1]},

      v1[3]= {0,0,0}, v2[3] = {0,0,0};

    area_volume_directed_integration(v1, choice, +1, xrange1, Omega0, q, F, d, m, polish);
    area_volume_directed_integration(v2, choice, -1, xrange2, Omega0, q, F, d, m, polish);

    if ((choice & 1u) == 1u) v[0] = v1[0] + v2[0]; // area
    if ((choice & 2u) == 2u) v[1] = v1[1] + v2[1]; // vol
    if ((choice & 4u) == 4u) v[2] = v1[2] + v2[2]; // dvol/dOmega

    #else
    area_volume_directed_integration(v, choice, -1, xrange, Omega0, q, F, d, m, polish);
    #endif
  }
  
  
   /*
    Parameterize dimensionless potential (omega) of aligned Roche lobe

      W
        = delta Omega
        = 1/r + q(1/|vec(r) - vec{i}|
          - i.vec{r}) + 1/2 b ( vec{r}^2 - (vec{r}.vec{k})^2 )

    in spherical coordinates (r, nu, phi) with

      vec{r} = r [cos(nu), sin(nu) cos(phi), sin(nu) sin(phi)]

    with the

      b = F^2 (1+q) delta^3

    The routine returns

      {dW/d(r), dW/d(nu), dW/dphi}

    Input:
      mask :
        1. bit set for  W[0] = partial_r W
        2. bit set for  W[1] = partial_nu W
        3. bit set fot  W[2] = partial_phi W
      r - radius
      sc_nu[2] - [sin(nu), cos(nu)] - sinus and cosine of azimuthal angle
      q - mass ratio
      b - parameter of the potential
      c2_phi - cos^2 phi

    Output:
      W[2] =   {dW/d(r), dW/d(nu), dW/dphi}
  */
  template <class T, class F>
  void calc_dOmega2(
    T W[2],
    const unsigned &mask,
    const T r[2],
    const T sc_nu[2],
    const F sc_phi[2],
    const T &q,
    const T &b){

    T A = sc_nu[1],                   // cos(nu)
      B = sc_nu[0]*sc_phi[0],         // sin(nu) sin(phi)
      C = -sc_nu[0],                  // = partial_nu A = -sin(nu)
      D = sc_nu[1]*sc_phi[0],         // = partial_nu B = cos(nu)sin(phi)
//      G = 0,                          // = partial_phi A
      H = sc_nu[0]*sc_phi[1];         // = partial_phi B

    T t = 1/(1 + r[1] - 2*r[0]*A);
    t *= std::sqrt(t);                // std::pow(.., -1.5);

    if ((mask & 1U) == 1U) W[0] = q*(t*(A - r[0]) - A) + b*r[0]*(1 - B*B) - 1/r[1]; // dOmega/dr
    if ((mask & 2U) == 2U) W[1] = q*C*r[0]*(t - 1) - b*r[1]*B*D;                    // dOmega/d(nu)
    if ((mask & 4U) == 4U) W[2] = /* q*G*r[0]*(t - 1) */ - b*r[1]*B*H;              // dOmega/d(phi)
  }

  template <class T, class F>
  void calc_dOmega2_pole(
    T v[2],
    const unsigned & mask,
    const T r[2],
    const F sc_phi[2],
    const T &q,
    const T &b){

    T A[2] = {1, 0},          //  A, partial_nu A at pole
      B[2] = {0, sc_phi[0]},  //  B, partial_nu B at pole

      B00 = B[0]*B[0],

      t2 = 1/(1 + r[1] - 2*r[0]*A[0]), t3 = t2*std::sqrt(t2), t5 = t2*t3,

      Wrr = b + 2/(r[0]*r[1]) + q*t5*(-1 + 2*r[1] + A[0]*(3*A[0] - 4*r[0])) - b*B00,    // d^2Omega/dr^2
      Wnn = r[0]*(q*(A[0]*(1 - t3) + 3*r[0]*t5*A[1]*A[1]) + b*r[0]*(B00 - B[1]*B[1])),  // d^2Omega/dnu^2
      Wnr = q*A[1]*(-1 + t5*(1 + r[0]*(A[0] - 2*r[0]))) - 2*b*r[0]*B[0]*B[1];           // d^2Omega/drdnu

     if ((mask & 1U) == 1U) v[0] = (-Wnr-std::sqrt(Wnr*Wnr-Wrr*Wnn))/Wrr;    // dr/dnu
     if ((mask & 2U) == 2U) v[1] = 1/(Wrr*v[0] + Wnr);                       // d^2(dV/dOmega)/(dnu dphi)
  }
  
  /*
    Computing area of the surface, the volume and derivative of the
    volume w.r.t. potential of the semi-detached Roche.

    Input:
      L1 - Lagrange point L1
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      m - number of steps in x - direction
      choice - using as a mask
        1U  - area , stored in v[0]
        2U  - volume, stored in v[1]
        4U  - d(volume)/dOmega, stored in v[2]

    Using: Integrating surface in spherical system
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in x direction

    Precision:
      At the default setup the relative precision should be better
      than 1e-4. NOT SURE

    Output:
      v[3] = {area, volume, d{volume}/dOmega}

    Ref:
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
  */
 
  
  template<class T>
  void critical_area_volume_integration(
    T v[3],
    const unsigned & choice,
    const T & L1,
    const T & q,
    const T & F = 1,
    const T & d = 1,
    const int & m = 1 << 14) {

    //
    // What is calculated
    //

    bool
      b_area = (choice & 1U) == 1U,
      b_vol  = (choice & 2U) == 2U,
      b_dvol = (choice & 4U) == 4U;

    if (!b_area && !b_vol && !b_dvol) return;

    unsigned
      mask = 3,   // = 011b, default
      mask2 = 1;  // = 001b, default for the pole

    if (b_area) mask |= 4;     // + 100b
    if (b_dvol) mask2 |= 2;    // + 010b
    
    using real = long double;
    
    using G = glq<real, 10>;
    
    const int dim = G::n + 3;
    
    real 
      y[dim], k[4][dim], w[G::n], W[3], sc_nu[2], sum[3], 
      r[2], rt, rp, nu, 
      q_ = q, d2 = d*d, d3 = d2*d, b = d3*F*F*(1 + q),
      dnu = utils::m_pi/m;
  
    
    //
    // Setup init point
    //
    {
      real tp = L1/d;
      for (int i = 0; i < G::n; ++i) {
        w[i] = dnu*G::weights[i];
        y[i] = tp;
      }
      y[G::n] = y[G::n+1] = y[G::n+2] = 0;
   }
 
    //
    // Integration
    // 
    
    nu = 0;
    
    for (int i = 0; i < m; ++i) {
   
      // 1. step
      sum[0] = sum[1] = sum[2] = 0;
      if (i == 0) {   // discussing pole = L1 separately, sin(nu) = 0

        for (int j = 0; j < G::n; ++j){
          r[0] = y[j], r[1] = r[0]*r[0];
          calc_dOmega2_pole(W, mask2, r, G::sc_phi + 2*j, q_, b);
          k[0][j] = dnu*W[0];

          if (b_dvol) sum[2] += w[j]*r[1]*W[1];
        }

        if (b_area) k[0][G::n] = 0;
        if (b_vol)  k[0][G::n+1] = 0;
        if (b_dvol) k[0][G::n+2] = sum[2];

      } else {

        utils::sincos(nu, sc_nu, sc_nu+1);
        for (int j = 0; j < G::n; ++j){
          r[0] = y[j], r[1] = r[0]*r[0];

          calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b);
          rt = -W[1]/W[0];          // partial_nu r
          k[0][j] = dnu*rt;

          if (b_area) {
            rp = -W[2]/W[0];      // partial_phi r
            sum[0] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
          }

          if (b_vol)  sum[1] += w[j]*r[0]*r[1];
          if (b_dvol) sum[2] += w[j]*r[1]/W[0];
        }
        if (b_area) k[0][G::n] = sum[0];
        if (b_vol)  k[0][G::n+1] = sum[1]*sc_nu[0];
        if (b_dvol) k[0][G::n+2] = sum[2]*sc_nu[0];
      }

      // 2. step
      sum[0] = sum[1] = sum[2] = 0;
      utils::sincos(nu + 0.5*dnu, sc_nu, sc_nu+1);
      for (int j = 0; j < G::n; ++j){
        r[0] = y[j] + 0.5*k[0][j], r[1] = r[0]*r[0];

        calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b);
        rt = -W[1]/W[0];        // partial_nu r
        k[1][j] = dnu*rt;

        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          sum[0] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  sum[1] += w[j]*r[0]*r[1];
        if (b_dvol) sum[2] += w[j]*r[1]/W[0];
      }
      if (b_area) k[1][G::n] = sum[0];
      if (b_vol)  k[1][G::n+1] = sum[1]*sc_nu[0];
      if (b_dvol) k[1][G::n+2] = sum[2]*sc_nu[0];

      // 3. step
      sum[0] = sum[1] = sum[2] = 0;
      for (int j = 0; j < G::n; ++j){
        r[0] = y[j] + 0.5*k[1][j], r[1] = r[0]*r[0];

        calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b);
        rt = -W[1]/W[0];        // partial_nu r
        k[2][j] = dnu*rt;

        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          sum[0] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  sum[1] += w[j]*r[0]*r[1];
        if (b_dvol) sum[2] += w[j]*r[1]/W[0];
      }
      if (b_area) k[2][G::n] = sum[0];
      if (b_vol)  k[2][G::n+1] = sum[1]*sc_nu[0];
      if (b_dvol) k[2][G::n+2] = sum[2]*sc_nu[0];

      // 4. step
      sum[0] = sum[1] = sum[2] = 0;
      utils::sincos(nu + dnu, sc_nu, sc_nu+1);
      for (int j = 0; j < G::n; ++j){
        r[0] = y[j] + k[2][j], r[1] = r[0]*r[0];

        calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b);
        rt = -W[1]/W[0];         // partial_nu r
        k[3][j] = dnu*rt;

        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          sum[0] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  sum[1] += w[j]*r[0]*r[1];
        if (b_dvol) sum[2] += w[j]*r[1]/W[0];
      }
      if (b_area) k[3][G::n] = sum[0];
      if (b_vol)  k[3][G::n+1] = sum[1]*sc_nu[0];
      if (b_dvol) k[3][G::n+2] = sum[2]*sc_nu[0];

      // final step
      for (int j = 0; j < dim; ++j)
        y[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;

      nu += dnu;
    }

    if (b_area) v[0] = d2*y[G::n];
    if (b_vol)  v[1] = d3*y[G::n + 1]/3;
    if (b_dvol) v[2] = d3*d*y[G::n + 2];
  }

  /*
    Calculating volume and value of the potential of the critical
    (semi-detached) case.

    Input:
      choice: calculate
        1U  - Area , stored in v[0]
        2U  - Volume, stored in v[1]
        4U  - dVolume/dOmega, stored in v[2]
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects

    Output:
      OmegaC - value of the Kopal potential
      av[3]   - volume and dvolume/dOmega of the critical volume

    Return:
      true - if there are no problem and false otherwise
  */
  //#define DEBUG
  template <class T>
  bool critical_area_volume(
    const unsigned &choice, 
    const T & q,
    const T & F,
    const T & delta,
    T & OmegaC,
    T av[3]) {
    
    #if defined(DEBUG)
    const char *fname = "critical_area_volume";
    #endif
    
    T L1 = lagrange_point_L1(q, F, delta);
    
    OmegaC = potential_on_x_axis(L1, q, F, delta);
     
    #if defined(DEBUG)
    std::cerr.precision(16);
    std::cerr << fname << "::OmegaC=" << OmegaC << " L1=" << L1 << '\n'; 
    #endif
    
    // compute volume and d(volume)/dOmega
    critical_area_volume_integration(av, choice, L1, q, F, delta, 1<<10);
    
    #if defined(DEBUG)
    std::cerr << fname << "::av=" << av[0] << ':' << av[1] << ':' << av[2] << '\n'; 
    #endif 
    return true;
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif
  

  /*
    Computing surface area and the volume of the primary Roche lobe in the limit of high w=delta*Omega. It should precise at least up to 5.5 digits for

      w > 10 &&  w > 5(q + b^(2/3)/4)  (empirically found bound)

     Analysis available in volume.nb

    Input:
      Omega0
      w - Omega0*delta - rescaled Omega
      q - mass ratio M2/M1
      b = (1+q)F^2 delta^3 - rescalted synchronicity parameter
      choice - using as a mask
        1U  - area , stored in v[0]
        2U  - volume, stored in v[1]
        4U  - d(volume)/dOmega, stored in v[2]

    Using: series approximation generated in volume.nb

    Output:
      v[3] = {area, volume, d{volume}/dOmega}


    Ref:
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better
  */

  template<class T>
  void area_volume_primary_asymp(
    T v[3],
    const unsigned & choice,
    const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & d = 1
  ) {

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
      q2 = q*q, q3 = q2*q,
      b2 = b*b, b3 = b2*b;


    // calculate area
    if (b_area) {

      T a[10] = {1, 2*q, 3*q2, 2*b/3. + 4*q3, q*(10*b/3. + 5*q3),
          q2*(10*b + 6*q3), b2 + q*(2*b/3. + q*(2 + q*(70*b/3. + 7*q3))),
          q*(8*b2 + q*(16*b/3. + q*(16 + q*(140*b/3. + 8*q3)))),
          q2*(2.142857142857143 + 36*b2 + q*(24*b + q*(72 + q*(84*b + 9*q3)))),
          68*b3/35. + q*(82*b2/35. + q*(342*b/35. + q*(24.17142857142857 + 120*b2 + q*(80*b + q*(240 + q*(140*b + 10*q3))))))
        },
        sumA = a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*(a[7] + s*(a[8] + s*a[9]))))))));

      v[0] = utils::m_4pi/(Omega0*Omega0)*sumA;
    }

    if (b_vol || b_dvol) {

      T a[10] = {1, 3*q, 6*q2, b + 10*q3, q*(6*b + 15*q3),
          q2*(21*b + 21*q3), 8*b2/5. + q*(4*b/5. + q*(2.4 + q*(56*b + 28*q3))),
          q*(72*b2/5. + q*(36*b/5. + q*(21.6 + q*(126*b + 36*q3)))),
          q2*(2.142857142857143 + 72*b2 + q*(36*b + q*(108 + q*(252*b + 45*q3)))),
          22*b3/7. + q*(22*b2/7. + q*(88*b/7. + q*(26.714285714285715 + 264*b2 + q*(132*b + q*(396 + q*(462*b + 55*q3))))))
        };

      // calculate volume
      if (b_vol) {
        T sumV = a[0] + s*(a[1] + s*(a[2] + s*(a[3] + s*(a[4] + s*(a[5] + s*(a[6] + s*(a[7] + s*(a[8] + s*a[9]))))))));
        v[1] = utils::m_4pi/(3*Omega0*Omega0*Omega0)*sumV;
      }

      // calculate d(volume)/d(Omega)
      if (b_dvol) {
        T sumdV = 3*a[0] + s*(4*a[1] + s*(5*a[2] + s*(6*a[3] + s*(7*a[4] + s*(8*a[5] + s*(9*a[6] + s*(10*a[7] + s*(11*a[8] + 12*s*a[9]))))))));
        v[2] = -utils::m_4pi/(3*Omega0*Omega0*Omega0*Omega0)*sumdV;
      }
    }
  }
} // namespace gen_roche
