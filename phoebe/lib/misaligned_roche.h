#pragma once

/*
  Library dealing with the generalized Roche potential with misaligned
  binary system.

  THERE IS NO SUPPORT FOR (OVER-)CONTACT CASE, AS IT IS NOT PHYSICAL.
  STILL NOTATION OF TWO LOBES REMAINS.

  The surface is defined implicitly with potential

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
    theta' - angle between z axis and spin of the object in [0, pi]
             spin in plane (x, z)

  Author: Martin Horvat, December 2016
*/

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <fstream>

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

      if sign = 0: return (tp (s > 0) + tp(s < 0))/2

    Input:
      Omega0 - value of potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      sintheta - sinus of the angle of spin w.r.t. z axis in rotated system

    Return:
      height = delta*tp
  */
  //#define DEBUG
  template <class T>
  T poleL_height(
    const T & Omega0,
    const T & q,
    const T & F,
    const T & delta,
    const T & sintheta,
    const int & sign = 1
  ) {

    #if defined(DEBUG)
    std::cerr << "poleL_height:\n"
      << "Omega0=" << Omega0
      << " q=" << q
      << " F=" << F
      << " delta=" << delta
      << " sintheta=" << sintheta
      << " sign=" << sign << '\n';
    #endif

    if (sintheta == 0)
      return gen_roche::poleL(Omega0, q, F, delta);

    if (Omega0 < 0 || q < 0)  return -1;


    if (sign == 0) {

      T p1 = poleL_height(Omega0, q, F, delta, sintheta, 1),
        p2 = poleL_height(Omega0, q, F, delta, sintheta, -1);

      if (p1 > 0 && p2 > 0) return (p1 + p2)/2;

      return -1;
    }

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
    const int & sign = 1
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
    if (s[0] == 0 && s[1] == 0)
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
        2 - over-contact
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

    // for cases which are not strongly deformed the pole is also
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

    if (s[0] == 0 && s[1] == 0)
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
      mask :
        1. bit set for  W[0] = partial_r W
        2. bit set for  W[1] = partial_theta W
        3. bit set fot  W[2] = partial_phi W
      r - radial distance
      sc_theta - sinus and cosine azimuthal angle
      sc_phi - sinus and cosine polar angle
      q - mass ratio M2/M1
      b = F^2 d^3 -
      sc_th - sinus and cosine angle between z axis in spin of the object in [0, pi]
              spin in plane (x, z)
    Output:
      W[3] = {Wr, Wtheta, Wphi}

  */

  template <class T, class F> void calc_dOmega(
    T W[3],
    const unsigned & mask,
    const T r[2],
    const T sc_nu[2],
    const F sc_phi[2],
    const T &q,
    const T &b,
    const T sc_th[2]
  ){

    T t = sc_th[1]*sc_phi[1],
      t1 = t*sc_nu[0] - sc_nu[1]*sc_th[0],
      t2 = t*sc_nu[1] + sc_nu[0]*sc_th[0],
      t3 = 1/(1 + 2*r[0]*t1 + r[1]);

    t3 *= std::sqrt(t3);

    // partial_r W
    if ((mask & 1U) == 1U)
      W[0] = -1/r[1] + b*r[0]*sc_nu[0]*sc_nu[0] + q*(t1 - t3*(r[0] + t1));

    // partial_theta W
    if ((mask & 2U) == 2U)
      W[1] = r[0]*(r[0]*b*sc_nu[1]*sc_nu[0] - q*t2*(-1 + t3));

    // partial_phi W
    if ((mask & 4U) == 4U)
      W[2] = q*r[0]*sc_th[1]*sc_phi[0]*sc_nu[0]*(-1 + t3);
  }

  //
  //  Gauss-Lagrange quadrature (GLQ) in [0,Pi]
  //

  template <class T, int N> struct glq {
    static const int n = N;
    static const T phi[];       // Gauss-Lagrange nodes x_i in [0, Pi]
    static const T weights[];   // Gauss-Lagrange weights
    static const T sc_phi[];    // sin and cos of nodes  x_i
  };

  /*
    Gauss-Lagrange: double, N=10
  */
  template <>
  const double glq<double, 10>::phi[] = {0.04098752915855404645614279849,0.21195796795501301950049770970,0.50358227252148264361980251927,0.890020433646848629194248947918,1.336945061968532022939709383344,1.804647591621261215522933999936,2.251572219942944609268394435362,2.63801038106831059484284086401,2.92963468563478021896214567357,3.10060512443123919200650058479};
  template <>
  const double glq<double, 10>::weights[] = {0.104727102742565162602343989,0.23475763028027358865964263,0.344140053490959719873798523,0.422963173620254734501715992,0.46420836666084341359382055662,0.46420836666084341359382055662,0.422963173620254734501715992,0.344140053490959719873798523,0.23475763028027358865964263,0.104727102742565162602343989};
  template <>
  const double glq<double, 10>::sc_phi[]={0.04097605376773743163322240975,0.9991601288170097367845030391481,0.2103744522327794774210133837,0.977620882473240741252534327489,0.482566195624121249707483164557,0.87585950177003979410936443884,0.777084608547669349628856397297,0.629396148032632538417667407578,0.9727811745399645767183604177668,0.231725670698451046669980417349,0.9727811745399645767183604177668,-0.231725670698451046669980417349,0.777084608547669349628856397297,-0.629396148032632538417667407578,0.482566195624121249707483164557,-0.87585950177003979410936443884,0.2103744522327794774210133837,-0.977620882473240741252534327489,0.04097605376773743163322240975,-0.9991601288170097367845030391481};

  /*
    Gauss-Lagrange: long double, N=10
  */
  template <>
  const long double glq<long double, 10>::phi[] = {0.04098752915855404645614279849L,0.21195796795501301950049770970L,0.50358227252148264361980251927L,0.890020433646848629194248947918L,1.336945061968532022939709383344L,1.804647591621261215522933999936L,2.251572219942944609268394435362L,2.63801038106831059484284086401L,2.92963468563478021896214567357L,3.10060512443123919200650058479L};
  template <>
  const long double glq<long double, 10>::weights[] = {0.104727102742565162602343989L,0.23475763028027358865964263L,0.344140053490959719873798523L,0.422963173620254734501715992L,0.46420836666084341359382055662L,0.46420836666084341359382055662L,0.422963173620254734501715992L,0.344140053490959719873798523L,0.23475763028027358865964263L,0.104727102742565162602343989L};
  template <>
  const long double glq<long double, 10>::sc_phi[]={0.04097605376773743163322240975L,0.9991601288170097367845030391481L,0.2103744522327794774210133837L,0.977620882473240741252534327489L,0.482566195624121249707483164557L,0.87585950177003979410936443884L,0.777084608547669349628856397297L,0.629396148032632538417667407578L,0.9727811745399645767183604177668L,0.231725670698451046669980417349L,0.9727811745399645767183604177668L,-0.231725670698451046669980417349L,0.777084608547669349628856397297L,-0.629396148032632538417667407578L,0.482566195624121249707483164557L,-0.87585950177003979410936443884L,0.2103744522327794774210133837L,-0.977620882473240741252534327489L,0.04097605376773743163322240975L,-0.9991601288170097367845030391481L};



  /*
    Gauss-Lagrange: double, N=15
  */

  template <>
  const double glq<double, 15>::phi[] ={0.01886130858747740302305159111,0.09853072480927601402339328008,0.23843654121054845237023696531,0.43288361530924932044530786709,0.673915335359302111609881595248,0.951664838604199667507182945785,1.25476138297089931304806023055,1.57079632679489661923132169164,1.88683127061889392541458315273,2.18992781498559357095546043749,2.46767731823049112685276178803,2.70870903828054391801733551619,2.90315611237924478609240641797,3.04306192878051722443925010320,3.12273134500231583543959179217};
  template <>
  const double glq<double, 15>::weights[] ={0.0483070795645355594897227,0.1105307289253955042410349,0.1683253098920381811957743,0.2192371082146767564709842,0.26117505775643872877586132,0.292421015016909813450427436,0.3116954482722822893249093754,0.318209158305239572565215093812,0.3116954482722822893249093754,0.292421015016909813450427436,0.26117505775643872877586132,0.2192371082146767564709842,0.1683253098920381811957743,0.1105307289253955042410349,0.0483070795645355594897227};
  template <>
  const double glq<double, 15>::sc_phi[] ={0.01886019029221170454895920332,0.999822130792343265472014507036,0.09837137447946246258546963059,0.995149773995362650465809060199,0.23618368963102833465466676737,0.971708425790511543157796159189,0.41949017315942443170777562913,0.907759877182658974839321181669,0.624050144074110653072553853434,0.781384295773265292561866660664,0.814382785538204297341827158319,0.580328078434117424683238917443,0.950475227124723936234712290234,0.310800325968626435374807587065,1.,0,0.950475227124723936234712290234,-0.310800325968626435374807587065,0.814382785538204297341827158319,-0.580328078434117424683238917443,0.624050144074110653072553853434,-0.781384295773265292561866660664,0.41949017315942443170777562913,-0.907759877182658974839321181669,0.23618368963102833465466676737,-0.971708425790511543157796159189,0.09837137447946246258546963059,-0.995149773995362650465809060199,0.01886019029221170454895920332,-0.999822130792343265472014507036};


  /*
    Gauss-Lagrange: long double, N=15
  */
  template <>
  const long double glq<long double, 15>::phi[] ={0.01886130858747740302305159111L,0.09853072480927601402339328008L,0.23843654121054845237023696531L,0.43288361530924932044530786709L,0.673915335359302111609881595248L,0.951664838604199667507182945785L,1.25476138297089931304806023055L,1.57079632679489661923132169164L,1.88683127061889392541458315273L,2.18992781498559357095546043749L,2.46767731823049112685276178803L,2.70870903828054391801733551619L,2.90315611237924478609240641797L,3.04306192878051722443925010320L,3.12273134500231583543959179217L};
  template <>
  const long double glq<long double, 15>::weights[] = {0.0483070795645355594897227L,0.1105307289253955042410349L,0.1683253098920381811957743L,0.2192371082146767564709842L,0.26117505775643872877586132L,0.292421015016909813450427436L,0.3116954482722822893249093754L,0.318209158305239572565215093812L,0.3116954482722822893249093754L,0.292421015016909813450427436L,0.26117505775643872877586132L,0.2192371082146767564709842L,0.1683253098920381811957743L,0.1105307289253955042410349L,0.0483070795645355594897227L};
  template <>
  const long double glq<long double, 15>::sc_phi[] ={0.01886019029221170454895920332L,0.999822130792343265472014507036L,0.09837137447946246258546963059L,0.995149773995362650465809060199L,0.23618368963102833465466676737L,0.971708425790511543157796159189L,0.41949017315942443170777562913L,0.907759877182658974839321181669L,0.624050144074110653072553853434L,0.781384295773265292561866660664L,0.814382785538204297341827158319L,0.580328078434117424683238917443L,0.950475227124723936234712290234L,0.310800325968626435374807587065L,1.L,0,0.950475227124723936234712290234L,-0.310800325968626435374807587065L,0.814382785538204297341827158319L,-0.580328078434117424683238917443L,0.624050144074110653072553853434L,-0.781384295773265292561866660664L,0.41949017315942443170777562913L,-0.907759877182658974839321181669L,0.23618368963102833465466676737L,-0.971708425790511543157796159189L,0.09837137447946246258546963059L,-0.995149773995362650465809060199L,0.01886019029221170454895920332L,-0.999822130792343265472014507036L};


  /*
    Gauss-Lagrange: double, N=20
  */
  template <>
  const double glq<double, 20>::phi[] ={0.01079357115998835146186400956,0.05659276429335242893622568720,0.13786183772187121870223765916,0.25271446970529852223404662382,0.39846090955745937951714246968,0.571685541465312886007064052145,0.768328316649816128297271224881,0.983780175339827807272301539548,1.21299114852575142200527236462,1.45058874849600697890818542853,1.69100390509378625955445795475,1.92860150506404181645737101866,2.15781247824996543119034184373,2.37326433693997711016537215840,2.56990711212448035245557933113,2.74313174403233385894550091360,2.88887818388449471622859675946,3.00373081586792201976040572412,3.08499988929644080952641769608,3.13079908242980488700077937372};
  template <>
  const double glq<double, 20>::weights[] = {0.027668017714319232925418,0.06377657679306866010546,0.09844502331593073687937,0.13081079977613566682353,0.16011145779868497445494,0.185659536652395137163457,0.2068560295565878367516543,0.22320404656916060739551142,0.2343203792081907677358680836,0.23994445941042299899612266552,0.23994445941042299899612266552,0.2343203792081907677358680836,0.22320404656916060739551142,0.2068560295565878367516543,0.185659536652395137163457,0.16011145779868497445494,0.13081079977613566682353,0.09844502331593073687937,0.06377657679306866010546,0.027668017714319232925418};
  template <>
  const double glq<double, 20>::sc_phi[] ={0.01079336158391596000681734331,0.999941749976326794615012047902,0.05656256046970460170211787674,0.998399056867098112318033963413,0.13742555479002748434059238656,0.990512098306049543001505966660,0.25003312796205107259814091063,0.968237282344319043647953517531,0.38800028546857557230688753527,0.921659252910913851897432015992,0.541050340504713439907471636237,0.840990207457693078705949304014,0.694934147189494251761901518246,0.719073383647323118290100964366,0.832597074589442745633198713268,0.553879148718474428402652823371,0.936667747979542417590368396156,0.350219259742996446317428054187,0.992783764808257789520375003351,0.119918290236068168443178536366,0.992783764808257789520375003351,-0.119918290236068168443178536366,0.936667747979542417590368396156,-0.350219259742996446317428054186,0.832597074589442745633198713268,-0.553879148718474428402652823371,0.694934147189494251761901518246,-0.719073383647323118290100964366,0.541050340504713439907471636237,-0.840990207457693078705949304014,0.38800028546857557230688753527,-0.921659252910913851897432015992,0.25003312796205107259814091063,-0.968237282344319043647953517531,0.13742555479002748434059238656,-0.990512098306049543001505966660,0.05656256046970460170211787674,-0.998399056867098112318033963413,0.01079336158391596000681734331,-0.999941749976326794615012047902};

  /*
    Gauss-Lagrange: long double, N=20
  */
  template <>
  const long double glq<long double, 20>::phi[] ={0.01079357115998835146186400956L,0.05659276429335242893622568720L,0.13786183772187121870223765916L,0.25271446970529852223404662382L,0.39846090955745937951714246968L,0.571685541465312886007064052145L,0.768328316649816128297271224881L,0.983780175339827807272301539548L,1.21299114852575142200527236462L,1.45058874849600697890818542853L,1.69100390509378625955445795475L,1.92860150506404181645737101866L,2.15781247824996543119034184373L,2.37326433693997711016537215840L,2.56990711212448035245557933113L,2.74313174403233385894550091360L,2.88887818388449471622859675946L,3.00373081586792201976040572412L,3.08499988929644080952641769608L,3.13079908242980488700077937372L};
  template <>
  const long double glq<long double, 20>::weights[] = {0.027668017714319232925418L,0.06377657679306866010546L,0.09844502331593073687937L,0.13081079977613566682353L,0.16011145779868497445494L,0.185659536652395137163457L,0.2068560295565878367516543L,0.22320404656916060739551142L,0.2343203792081907677358680836L,0.23994445941042299899612266552L,0.23994445941042299899612266552L,0.2343203792081907677358680836L,0.22320404656916060739551142L,0.2068560295565878367516543L,0.185659536652395137163457L,0.16011145779868497445494L,0.13081079977613566682353L,0.09844502331593073687937L,0.06377657679306866010546L,0.027668017714319232925418L};
  template <>
  const long double glq<long double, 20>::sc_phi[] ={0.01079336158391596000681734331L,0.999941749976326794615012047902L,0.05656256046970460170211787674L,0.998399056867098112318033963413L,0.13742555479002748434059238656L,0.990512098306049543001505966660L,0.25003312796205107259814091063L,0.968237282344319043647953517531L,0.38800028546857557230688753527L,0.921659252910913851897432015992L,0.541050340504713439907471636237L,0.840990207457693078705949304014L, 0.694934147189494251761901518246L,0.719073383647323118290100964366L,0.832597074589442745633198713268L,0.553879148718474428402652823371L,0.936667747979542417590368396156L,0.350219259742996446317428054187L,0.992783764808257789520375003351L,0.119918290236068168443178536366L,0.992783764808257789520375003351L,-0.119918290236068168443178536366L,0.936667747979542417590368396156L,-0.350219259742996446317428054186L,0.832597074589442745633198713268L,-0.553879148718474428402652823371L,0.694934147189494251761901518246L,-0.719073383647323118290100964366L,0.541050340504713439907471636237L,-0.840990207457693078705949304014L,0.38800028546857557230688753527L,-0.921659252910913851897432015992L,0.25003312796205107259814091063L,-0.968237282344319043647953517531L,0.13742555479002748434059238656L,-0.990512098306049543001505966660L,0.05656256046970460170211787674L,-0.998399056867098112318033963413L,0.01079336158391596000681734331L,-0.999941749976326794615012047902L};


  /*
    Computing area of the surface and the volume of the primary
    generalized Roche lobes with misaligned spin and orbital velocity
    vector.

    Input:
      choice - using as a mask
        1U  - Area , stored in v[0]
        2U  - Volume, stored in v[1]
        4U  - dVolume/dOmega, stored in v[2]
      pole - pole of the lobe
      Omega0 - value of the potential
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
      th - angle between z axis in spin of the object in [0, pi/2]
              spin in plane (x, z)
      m - number of steps in x - direction



    Using: Integrating surface in spherical coordinates
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in direction of theta

    Output:
      v[3] = {area, volume, d{volume}/d{Omega}}

    Notes: for Omega > Omega_critical

      A, V, dV/dOmega converge as O(m^-4)

    Ref:
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better
  */

  template<class T>
  void area_volume_integration(
    T v[3],
    const unsigned & choice,
    const T & pole,
    [[maybe_unused]] const T & Omega0,
    const T & q,
    const T & F = 1,
    const T & d = 1,
    const T & th = 0,
    const int & m = 1 << 14)
  {

    //
    // What is calculated
    //

    bool
      b_area = (choice & 1U) == 1U,
      b_vol  = (choice & 2U) == 2U,
      b_dvol = (choice & 4U) == 4U;

    if (!b_area && !b_vol && !b_dvol) return;

    unsigned mask = 3;        // = 011b
    if (b_area) mask |= 4U;   // += 100b

    using real = long double;

    using G = glq<real, 15>;

    const int dim = G::n + 3;

    real
      W[3], w[G::n], y[dim], k[4][dim], sc_nu[2], sc_th[2],
      q_ = q, d2 = d*d, d3 = d*d2,
      b = (1 + q)*F*F*d3,
      dnu = utils::pi<real>()/m,
      rt, rp, r[2], nu;

    //
    // Setting initial values
    //

    {
      real tp = pole/d;
      for (int i = 0; i < G::n; ++i) {
        y[i] = tp;
        w[i] = dnu*G::weights[i];
      }
      y[G::n] = y[G::n + 1] = y[G::n + 2] = 0;

      utils::sincos(real(th), sc_th, sc_th + 1);
    }

    //
    // Rk integration
    //

    nu = 0;
    for (int i = 0; i < m; ++i) {

      // 1. step
      utils::sincos(nu, sc_nu, sc_nu + 1);
      k[0][G::n] = k[0][G::n + 1] = k[0][G::n + 2] = 0;
      for (int j = 0; j < G::n; ++j){

        r[0] = y[j], r[1] = r[0]*r[0];
        calc_dOmega(W, mask, r, sc_nu, G::sc_phi+2*j, q_, b, sc_th);

        rt = -W[1]/W[0];      // partial_theta r
        k[0][j] = dnu*rt;

        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          k[0][G::n] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  k[0][G::n + 1] += w[j]*r[1]*r[0];
        if (b_dvol) k[0][G::n + 2] += w[j]*r[1]/W[0];
      }
      if (b_vol)  k[0][G::n + 1] *= sc_nu[0];
      if (b_dvol) k[0][G::n + 2] *= sc_nu[0];

      // 2. step
      utils::sincos(nu + 0.5*dnu, sc_nu, sc_nu + 1);
      k[1][G::n] = k[1][G::n + 1] = k[1][G::n + 2] = 0;
      for (int j = 0; j < G::n; ++j){

        r[0] = y[j] + 0.5*k[0][j], r[1] = r[0]*r[0];
        calc_dOmega(W, mask, r, sc_nu, G::sc_phi+2*j, q_, b, sc_th);

        rt = -W[1]/W[0];      // partial_theta r
        k[1][j] = dnu*rt;

        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          k[1][G::n] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  k[1][G::n + 1] += w[j]*r[1]*r[0];
        if (b_dvol) k[1][G::n + 2] += w[j]*r[1]/W[0];
      }
      if (b_vol)  k[1][G::n + 1] *= sc_nu[0];
      if (b_dvol) k[1][G::n + 2] *= sc_nu[0];


      // 3. step
      k[2][G::n] = k[2][G::n + 1] = k[2][G::n + 2] = 0;
      for (int j = 0; j < G::n; ++j){

        r[0] = y[j] + 0.5*k[1][j], r[1] = r[0]*r[0];
        calc_dOmega(W, mask, r, sc_nu, G::sc_phi+2*j, q_, b, sc_th);

        rt = -W[1]/W[0];      // partial_theta r
        k[2][j] = dnu*rt;

        if (b_area) {
          rp = -W[2]/W[0];      // partial_phi r
          k[2][G::n] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  k[2][G::n + 1] += w[j]*r[1]*r[0];
        if (b_dvol) k[2][G::n + 2] += w[j]*r[1]/W[0];
      }
      if (b_vol)  k[2][G::n + 1] *= sc_nu[0];
      if (b_dvol) k[2][G::n + 2] *= sc_nu[0];


      // 4. step
      utils::sincos(nu + dnu, sc_nu, sc_nu + 1);
      k[3][G::n] = k[3][G::n + 1] = k[3][G::n + 2] = 0;
      for (int j = 0; j < G::n; ++j){

        r[0] = y[j] + k[2][j], r[1] = r[0]*r[0];
        calc_dOmega(W, mask, r, sc_nu, G::sc_phi+2*j, q_, b, sc_th);

        rt = -W[1]/W[0];      // partial_theta r
        k[3][j] = dnu*rt;

        if (b_area) {
         rp = -W[2]/W[0];      // partial_phi r
         k[3][G::n] += w[j]*r[0]*std::sqrt(rp*rp + sc_nu[0]*sc_nu[0]*(r[1] + rt*rt));
        }
        if (b_vol)  k[3][G::n + 1] += w[j]*r[1]*r[0];
        if (b_dvol) k[3][G::n + 2] += w[j]*r[1]/W[0];
      }
      if (b_vol)  k[3][G::n + 1] *= sc_nu[0];
      if (b_dvol) k[3][G::n + 2] *= sc_nu[0];


      // final step
      for (int j = 0; j < dim; ++j)
        y[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;

      nu += dnu;
    }

    if (b_area) v[0] = 2*d2*y[G::n];
    if (b_vol)  v[1] = 2*d3*y[G::n + 1]/3;
    if (b_dvol) v[2] = 2*d3*d*y[G::n + 2];
  }

  /*
    Parameterize dimensionless potential (omega)

      W
        = delta Omega
        = 1/r + q(1/|vec(r) - vec{i}|
          - i.vec{r}) + 1/2 b | vec{r} - vec{S}(vec{r}.vec{S})|^2

    in spherical coordinates (r, nu, phi) with

      vec{r} = r [sin(nu) cos(phi), sin(nu) sin(phi), cos(nu)]

    with the

      x - axis in the plane of vec{S} and line connecting centers
      z - axis connecting origin and L1 point
      b = F^2 (1+q) delta^3

      vec{i} = (a', 0, b')
      vec{S} = (c', 0, d')

    The routine returns

      {dW/d(r), dW/d(nu), dW/dphi}

    Input:
      mask :
        1. bit set for  W[0] = partial_r W
        2. bit set for  W[1] = partial_nu W
        3. bit set fot  W[2] = partial_phi W
      r - radius
      sc_nu[2] - [sin(nu), cos(nu)] - sinus and cosine of azimuthal angle
      sc_phi[2] - [sin(phi), cos(phi)] - sinus and cosine of polar angle
      q - mass ratio
      b - parameter of the potential
      p[4] = {a', b', c', d') - parameters of vectors

    Output:
      W[2] =   {dW/d(r), dW/d(nu), dW/dphi}
  */
  template <class T>
  void calc_dOmega2(
    T W[2],
    const unsigned &mask,
    const T r[2],
    const T sc_nu[2],
    const T sc_phi[2],
    const T &q,
    const T &b,
    T p[4]){

    T t = sc_nu[0]*sc_phi[1],         // sin(nu) cos(phi)
      A = p[0]*t + p[1]*sc_nu[1],     // a' sin(nu) cos(phi) + b' cos(nu)
      B = p[2]*t + p[3]*sc_nu[1];     // c' sin(nu) cos(phi) + d' cos(nu)

    t = sc_nu[1]*sc_phi[1];           // cos(nu) cos(phi)

    T C = p[0]*t - p[1]*sc_nu[0],     // = partial_nu A
      D = p[2]*t - p[3]*sc_nu[0];     // = partial_nu B

    t = -sc_nu[0]*sc_phi[0];          // -sin(nu) sin(phi)

    T G = p[0]*t,                     // = partial_phi A
      H = p[2]*t;                     // = partial_phi B

    t = 1/(1 + r[1] - 2*r[0]*A);
    t *= std::sqrt(t);                // std::pow(.., -1.5);

    if ((mask & 1U) == 1U) W[0] = q*(t*(A - r[0]) - A) + b*r[0]*(1 - B*B) - 1/r[1]; // dOmega/dr
    if ((mask & 2U) == 2U) W[1] = q*C*r[0]*(t - 1) - b*r[1]*B*D;                    // dOmega/d(nu)
    if ((mask & 4U) == 4U) W[2] = q*G*r[0]*(t - 1) - b*r[1]*B*H;                    // dOmega/d(phi)
  }

  template <class T>
  bool calc_dOmega2_pole(
    T v[2],
    const unsigned & mask,
    const T r[2],
    const T sc_phi[2],
    const T &q,
    const T &b,
    T p[4]){

    T A[2] = {p[1], p[0]*sc_phi[1]},
      B[2] = {p[3], p[2]*sc_phi[1]},

      B00 = B[0]*B[0],

      t2 = 1/(1 + r[1] - 2*r[0]*A[0]), t3 = t2*std::sqrt(t2), t5 = t2*t3,

      Wrr = b + 2/(r[0]*r[1]) + q*t5*(-1 + 2*r[1] + A[0]*(3*A[0] - 4*r[0])) - b*B00,    // d^2Omega/dr^2
      Wnn = r[0]*(q*(A[0]*(1 - t3) + 3*r[0]*t5*A[1]*A[1]) + b*r[0]*(B00 - B[1]*B[1])),  // d^2Omega/dnu^2
      Wnr = q*A[1]*(-1 + t5*(1 + r[0]*(A[0] - 2*r[0]))) - 2*b*r[0]*B[0]*B[1];           // d^2Omega/drdnu


    if ((mask & 1U) == 1U) {
      T det = Wnr*Wnr-Wrr*Wnn;
      if (det < 0) {
         std::cerr << "calc_dOmega2_pole::det=" << det << "\n";
         return false;
      }
      v[0] = (-Wnr-std::sqrt(det))/Wrr;    // dr/dnu
    }

    if ((mask & 2U) == 2U) v[1] = 1/(Wrr*v[0] + Wnr);                       // d^2(dV/dOmega)/(dnu dphi)
    return false;
  }

  /*
    Computing the volume of the critical generalized Roche lobes
    with misaligned spin and orbital velocity vector.

    Input:
      choice - using as a mask
        1U  - Area , stored in v[0]
        2U  - Volume, stored in v[1]
        4U  - dVolume/dOmega, stored in v[2]

      x[2]  = {x_0,z_0} - critical point in xz plane
      q - mass ratio M2/M1
      F - synchronicity parameter
      d - separation between the two objects
      th - angle between z axis in spin of the object in [0, pi/2]
              spin in plane (x, z)
      m - number of steps in x - direction

    Using: Integrating surface in spherical coordinates

      a. Gauss-Lagrange integration in phi direction
      b. RK4 in direction of theta

    with the pole at L1 or L2 point.

    Additionally taking care of the pole.

    Output:
      v[3] = {area, volume, d{volume}/d{Omega}}

    Notes:
      A converge as O(m^-3)
      V, dV/dOmega converge as O(m^-4)

    Ref:
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better
  */
  //#define DEBUG
  template<class T>
  void critical_area_volume_integration(
    T v[3],
    const unsigned & choice,
    const T x[2],
    const T & q,
    const T & F = 1,
    const T & d = 1,
    const T & th = 0,
    const int & m = 1 << 14)
  {

    #if defined(DEBUG)
    std::cerr.precision(16);
    const char *fname = "critical_area_volume_integration";
    #endif

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
    using G = glq<real, 15>;

    const int dim = G::n + 3;

    real
      w[G::n], y[dim], k[4][dim], sc_nu[2], p[4], W[3], sum[3],
      r[2], nu, rt, rp,
      q_ = q, d2 = d*d, d3 = d2*d,
      b = (1 + q)*F*F*d3,
      dnu = utils::pi<real>()/m;

    #if defined(DEBUG)
    //
    // Checking if the x is a good critical point
    //

    real params[] =  {q, F, d, th, 0};

    Tmisaligned_rotated_roche<real> body(params);

    real B[3][3], g[3], Omega0;

    B[2][0]= x[0];
    B[2][1] = 0;
    B[2][2] = x[1];

    Omega0 = -body.constrain(B[2]);

    body.grad_only(B[2], g);

    std::cerr.precision(16);
    std::cerr
      << fname << "::Omega0=" << Omega0 << " grad="
      << g[0] << '\t' << g[1] << '\t' <<  g[2] << '\n';

    //
    // Basis
    //
    real t = std::hypot(real(x[0]), real(x[1]));

    B[2][0] /= t;
    B[2][2] /= t;

    B[0][0] = B[2][2];
    B[0][1] = 0;
    B[0][2] = -B[2][0];

    B[1][0] = 0;
    B[1][1] = 1;
    B[1][2] = 0;
    #endif

    //
    // Setting initial values
    //

    {
      real tp = std::hypot(real(x[0]), real(x[1]));

      utils::sincos(real(th), sc_nu, sc_nu+1);

      p[0] = x[1]/tp;                 // a'
      p[1] = x[0]/tp;                 // b'
      p[2] = p[0]*sc_nu[0] - p[1]*sc_nu[1]; // c'
      p[3] = p[0]*sc_nu[1] + p[1]*sc_nu[0]; // d'

      tp /= d;
      for (int i = 0; i < G::n; ++i) {
        y[i] = tp;
        w[i] = dnu*G::weights[i];
      }
      y[G::n] = y[G::n + 1] = y[G::n + 2] = real(0);

      #if defined(DEBUG)
      for (int i = 0; i < 4; ++i)
        k[i][G::n] = k[i][G::n + 1] = k[i][G::n + 2] = real(0);
      #endif
    }

    //
    // Rk integration
    //

    nu = 0;
    for (int i = 0; i < m; ++i) {

      // 1. step
      #if defined(DEBUG)
      W[0]= W[1] = W[2] = 0;
      #endif
      sum[0] = sum[1] = sum[2] = 0;
      if (i == 0) {   // discussing pole = L1 separately, sin(nu) = 0

        for (int j = 0; j < G::n; ++j){
          r[0] = y[j], r[1] = r[0]*r[0];
          calc_dOmega2_pole(W, mask2, r, G::sc_phi + 2*j, q_, b, p);
          k[0][j] = dnu*W[0];

          #if defined(DEBUG)
          std::cerr << fname << "::k00[" << j << "]=" << k[0][j] << " W=" << W[0] << ':' << W[1] << "\n";
          #endif

          if (b_dvol) sum[2] += w[j]*r[1]*W[1];
        }

        if (b_area) k[0][G::n] = 0;
        if (b_vol)  k[0][G::n+1] = 0;
        if (b_dvol) k[0][G::n+2] = sum[2];

      } else {

        utils::sincos(nu, sc_nu, sc_nu+1);
        for (int j = 0; j < G::n; ++j){
          r[0] = y[j], r[1] = r[0]*r[0];

          calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b, p);
          rt = -W[1]/W[0];          // partial_nu r
          k[0][j] = dnu*rt;

          #if defined(DEBUG)
          std::cerr << fname << "::k0[" << j << "]=" << k[0][j] <<" W=" << W[0] << ':' << W[1] << ':' << W[2] << "\n";
          #endif

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
      #if defined(DEBUG)
      W[0]= W[1] = W[2] = 0;
      #endif
      sum[0] = sum[1] = sum[2] = 0;
      utils::sincos(nu + 0.5*dnu, sc_nu, sc_nu+1);
      for (int j = 0; j < G::n; ++j){
        r[0] = y[j] + 0.5*k[0][j], r[1] = r[0]*r[0];

        calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b, p);
        rt = -W[1]/W[0];        // partial_nu r
        k[1][j] = dnu*rt;

        #if defined(DEBUG)
        std::cerr << fname << "::k1[" << j << "]=" << k[1][j] << " W=" << W[0] << ':' << W[1] << ':' << W[2] << "\n";
        #endif

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
      #if defined(DEBUG)
      W[0]= W[1] = W[2] = 0;
      #endif
      sum[0] = sum[1] = sum[2] = 0;
      for (int j = 0; j < G::n; ++j){
        r[0] = y[j] + 0.5*k[1][j], r[1] = r[0]*r[0];

        calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b, p);
        rt = -W[1]/W[0];        // partial_nu r
        k[2][j] = dnu*rt;

        #if defined(DEBUG)
        std::cerr << fname << "::k2[" << j << "]=" << k[2][j] << " W=" << W[0] << ':' << W[1] << ':' << W[2] << "\n";
        #endif

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
      #if defined(DEBUG)
      W[0]= W[1] = W[2] = 0;
      #endif
      sum[0] = sum[1] = sum[2] = 0;
      utils::sincos(nu + dnu, sc_nu, sc_nu+1);
      for (int j = 0; j < G::n; ++j){
        r[0] = y[j] + k[2][j], r[1] = r[0]*r[0];

        calc_dOmega2(W, mask, r, sc_nu, G::sc_phi + 2*j, q_, b, p);
        rt = -W[1]/W[0];         // partial_nu r
        k[3][j] = dnu*rt;

        #if defined(DEBUG)
        std::cerr << fname << "::k3[" << j << "]=" << k[3][j] << " W=" << W[0] << ':' << W[1] << ':' << W[2] << "\n";
        #endif

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

      #if defined(DEBUG)
      for (int j = 0; j < G::n; ++j) {

        real f, a[3] = {G::sc_phi[2*j+1]*sc_nu[0], G::sc_phi[2*j]*sc_nu[0], sc_nu[1]}, u[3];

        f = y[j]*d;
        for (int k = 0; k < 3; ++k) {
          u[k] = 0;
          for (int l = 0; l < 3; ++l) u[k]+= f*a[l]*B[l][k];
        }

        std::cerr << fname
          << "::y[" << j << "]=" << y[j]
          << " dOmega=" <<
          Omega0 + body.constrain(u)
          << "\n";
      }


      for (int j = G::n; j < dim; ++j)
        std::cerr << << fname << "::y[" << j << "]=" << y[j] << "\n";
      #endif

      nu += dnu;
    }

    if (b_area) v[0] = 2*d2*y[G::n];
    if (b_vol)  v[1] = 2*d3*y[G::n + 1]/3;
    if (b_dvol) v[2] = 2*d3*d*y[G::n + 2];

    #if defined(DEBUG)
    if (b_dvol && y[G::n + 2] > 0) {

      std::ofstream f("params.txt",std::ofstream::binary);

      int len = 9;
      T *buf = new T [len];

      buf[0]=v[0];
      buf[1]=v[1];
      buf[2]=v[2];
      buf[3]=x[0];
      buf[4]=x[1];
      buf[5]=q;
      buf[6]=F;
      buf[7]=d;
      buf[8]=th;

      std::cerr << fname << "::choice=" << choice << " m=" << m << '\n';

      f.write((char *)buf, sizeof(T)*len);

      f.close();

      delete [] buf;
    }
    #endif
  }

  #if defined(DEBUG)
  #undef DEBUG
  #endif

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
  //#define DEBUG
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

    #if defined(DEBUG)
    std::cerr << "L0=" << L0 << " theta=" << theta << '\n';
    #endif

    //
    // Approximating fixed point using RK4 integration from
    // position at aligned case
    //
    x[0] = L0;
    x[1] = 0;

    if (theta == 0) return true;

    {
      int n = int(std::abs(theta)/0.1);

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
  #if defined(DEBUG)
  #undef DEBUG
  #endif

  /*
    Calculate the minimal value of the Kopal potential for which the
    primary Roche lobe exists.

    Input:
      q - mass ratio M2/M1
      F - synchronicity parameter
      d - separation between the two objects
      th - angle between z axis in spin of the object
      L[2] - Lagrange point limiting the lobe

    Return:
      minimal permitted Omega, if NaN is returned we have some error
  */

  template<class T>
  T calc_Omega_min(
    const T & q,
    const T & F,
    const T & d,
    const T & th = 0,
    T *L = 0,
    T *pth = 0){

    T W[2], r[2][3],
      th1 = utils::m_pi*std::abs(std::fmod(th/utils::m_pi + 0.5, 1) - 0.5);

    //std::cerr << "calc_Omega_min::th1=" << th1 << '\n';

    for (int i = 0; i < 2; ++i) {

      if (!lagrange_point(i + 1, q, F, d, th1, r[i])) return std::numeric_limits<T>::quiet_NaN();

      r[i][2] = r[i][1];
      r[i][1] = 0;

      W[i] = calc_Omega(r[i], q, F, d, th1);
      //std::cerr << "calc_Omega_min::W[" << i << "]=" << W[i] << " x=" << r[i][0] << ":" << r[i][2] << '\n';
    }

    int ind = (W[0] > W[1] ? 0 : 1);

    if (L) {
      L[0] = r[ind][0];
      L[1] = r[ind][2];
    }

    if (pth) *pth = th1;

    return W[ind];
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
      d - separation between the two objects
      th - angle between z axis in spin of the object

    Output:
      OmegaC - value of the Kopal potential
      av[3] - area, volume and dvolume/dOmega of the critical lobe

    Return:
      true - if there are no problem and false otherwise
  */
  //#define DEBUG
  template <class T>
  bool critical_area_volume(
    const unsigned &choice,   // inputs
    const T & q,
    const T & F,
    const T & d,
    const T & th,
    T & OmegaC,             // outputs
    T av[3]) {

    const char *fname =  "critical_area_volume";

    #if defined(DEBUG)
    std::cerr.precision(16);
    std::cerr
      << fname << "::START\n"
      << fname << "::q=" << q << " F=" << F << " d=" << d << " th=" << th << '\n';
    #endif

    if (th == 0)
      return gen_roche::critical_area_volume(choice, q, F, d, OmegaC, av);

    T x[2], th1;

    OmegaC = calc_Omega_min(q, F, d, th, x, &th1);

    if (std::isnan(OmegaC)) {
      std::cerr << fname << "::Calculation of Omega_min failed\n";
      return false;
    }

    #if defined(DEBUG)
    std::cerr << fname << "::x=" << x[0] << ' ' << x[1] << " OmegaC=" << OmegaC << '\n';
    #endif

    critical_area_volume_integration(av, choice, x, q, F, d, th1, 1<<10);

    #if defined(DEBUG)
    std::cerr
      << fname << "::av=" << av[0] << ' ' << av[1] << ' ' << av[2] << '\n'
      << fname << "::critical_volume::END\n";
    #endif

    return true;
  }

  #if defined(DEBUG)
  #undef DEBUG
  #endif
} // namespace misaligned_roche

