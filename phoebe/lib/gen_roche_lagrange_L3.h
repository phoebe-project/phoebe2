#pragma once
/*
  Library dealing with L3 lagrange point of the generalized Roche
  lobes/Kopal potential

  Omega =
    1/rho
    + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
    + 1/2 F^2(1 + q) rho^2 (1 - nu^2)

  where position in spherical coordinates is given as

  x = rho lambda      lambda = sin(theta) cos(phi)
  y = rho  mu         mu = sin(theta) sin(phi)
  z = rho nu          nu = cos(theta)


  L3 is positioned on the x-axis:

    (x, 0, 0)

  with x > delta.

  Author: Martin Horvat,  March 2016
*/

#include <iostream>
#include <cmath>
#include <limits>

#include "utils.h"

namespace gen_roche {

  /*
    Solving (a > 0)
      -1 + (-1+a) t^2 + a t^3 == 0
    as a cubic equation
      x^3+b x^2+c x + d = 0
  */

  template <class T> T solve_cubic2(const T & a){

    // with Tschirnhaus transformation : t = x - b/3
    //    x^3 + p x + q = 0

    T b = (a - 1)/a,
      c = 0,
      d = -1/a,

      p = c - b*b/3, // -(1-1/a)^2/3 < 0 !
      q = b*(2*b*b/27 - c/3) + d,

      D = p*p*p/27 + q*q/4,

      A = 2*std::sqrt(std::abs(p)/3), phi;


    if (D <= 0){ // 3 real roots
      phi = std::acos(3*q/(A*p)); // biggest
      return A*std::cos(phi/3) - b/3;
    }

    // D > 0, only one real root
    if (p < 0){
      phi = acosh(-3*std::abs(q)/(A*p));
      return (q > 0 ? -A : A)*std::cosh(phi/3) - b/3;
    }

    // It not come to here!!!!

    // D > 0 && p > 0, only one real root
    phi = asinh(3*q/(A*p));
    return -A*std::sinh(phi/3) - b/3;
  }

/*
  Lagrange L3 point for the generalized potential on the line connecting
  both stars

      x > delta, y = 0, z = 0

  satisfying nabla omega = 0. The routine is tested in the regime of

    (q,F) in [0.01, 20] x [0.01 x 20]

  I. It is obtained by fining the root P(t) = 0 on interval t > 0 of
     the polynomial

    P(t) = 1 + 2 t + (2 + p - b)t^2 + (2 - 3b) t^3 + (1 - 3b) t^4 - bt^5

    a = F^2 delta^3
    b = a (1 + p)
    q = 1/p

  and solution is

    x = delta (1 + t)

  II. The other way is to transform polynomial

    P(s/(1-s)) = Q(s)/(1-s)^5

  with ( t= s/(1-s), s= t/(1+t) )

    Q(s) = 1 - 3 s + (4 + p - b)s^2 - (2 + 3p)s^3 + 3 p s^4 - p s^5

  and solve

    Q(s) = 0

  Then
    x = delta (1 + s/(1-s))

  Input:
    q - mass ratio M2/M1
    F - synchronicity parameter
    delta - separation between the two objects
  Returned:
    x value of the L3
*/

#if defined(DEBUG)
double lagrange_point_L3_x; // approximation going into N-R iteration
int lagrange_point_L3_n;    // number of interations in Newton-Raphson
#endif

template <class T>
T lagrange_point_L3(
  const T & q,
  const T & F = 1,
  const T & delta = 1
) {

  T p = 1/q;

  //
  // Discussing F = 0
  //

  if (F == 0)  return std::numeric_limits<T>::quiet_NaN();

  T x,
    a = F*F*delta*delta*delta;

  const T eps = 10*std::numeric_limits<T>::epsilon();
  const T min = 10*std::numeric_limits<T>::min();

  //
  // Discussing the case a = 1
  //

  if (a == 1) {  // t in [0,1]

    if (p > 1.5){ // using P

      T w = 1./std::cbrt(3*(1 + p)),
        c[8] ={1.,
          0.3333333333333333,
          -0.1111111111111111,
          0.6172839506172839,
          0.17695473251028807,
          -0.4444444444444444,
          0.9399481786313062,
          -0.027485647513082356};

        x = w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

    } else if (p < 0.5) {
      T c[8] = {
          0.5833333333333334,
          -0.24305555555555555,
          0.15562307098765432,
          -0.1101345486111111,
          0.07755659440907922,
          -0.049693552536710775,
          0.023596627510085143,
          0.0017260465555892458
        },
        s = p*(c[0] + p*(c[1] + p*(c[2] + p*(c[3] + p*(c[4] + p*(c[5] + p*(c[6] + p*c[7])))))));

       x = 1/(1 + s);

    } else { // q in [0.5, 1.5]
      T t0 = 0.69840614455492,
        w = p - 1,
        c[8]={
          -0.16326993510260143,
          0.06953110611033309,
          -0.033430671075654735,
          0.01687940218811356,
          -0.008734076561902074,
          0.004580958503437698,
          -0.0024213475610572683,
          0.0012854157986699256
        },
      s =  w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

      x = t0 + s;
    }

    //
    // Polishing: using Newton-Raphson
    //

    #if defined(DEBUG)
    lagrange_point_L3_x = x;
    #endif

    int n = 0;    // counting steps

    bool turn = (x > 0.5);

    if (turn) x = 1 - x;

    T dx, P, dP;

    do {

      if (turn){
        P = -7*p + x*(12 + 26*p + x*(-24 - 37*p + x*(19 + 25*p + x*(-7 - 8*p + (1 + p)*x))));
        dP = 12 + 26*p + x*(-48 - 74*p + x*(57 + 75*p + x*(-28 - 32*p + 5*(1 + p)*x)));
      } else {
        P = 1 + x*(2 + x*(1 + x*(-1 - 3*p + x*(-2 - 3*p - (1 + p)*x))));
        dP = 2 + x*(2 + x*(-3 - 9*p + x*(-8 - 12*p - 5*(1 + p)*x)));
      }

      dx = -P/dP;

      x += dx;

      if (++n > 10) {
        std::cerr << "Slow convergence at F=" << F << " q=" << q << " !\n";
        #if defined(DEBUG)
        lagrange_point_L3_n += n;
        #endif
        return delta*(turn ? 2 - x : 1 + x);
      }

    } while (P != 0 && std::abs(dx) > std::abs(x)*eps + min);


    if (turn) x = 1 - x;

    #if defined(DEBUG)
    lagrange_point_L3_n += n;
    #endif

    return delta*(1 + x);
  }

  //
  //  a != 1
  //

  //
  // Using a lot of iterations as the analytical series approaches q->0
  // Very badly cover the whole domain.
  //

  if (q == 1) {

    if (a > 1) {  // using P: for large a
      T b = std::sqrt(2*a),
        t = 1/b,
        g;

      // simple iteration
      for (int i = 0; i < 4; ++i) {
        g = 1/(b*std::sqrt(1 + t)); // g < 1
        t = 2*g/(1 - g + std::sqrt(1 + g*(2 -3*g)));
      }

      x = t;
    } else if (a == 1){

      x = 0.69840614455492;

    } else {   // using P(1/u): for small a

      T b = 2*a,
        u = b,
        g;

      // simple iteration
      for (int i = 0; i < 4; ++i) {

        g = b*utils::sqr((1 + u*(2 + u))/(1 + u*(1 + u)));

        u = 2*g/(std::sqrt(1 + 4*g) + 1);
      }

      x = 1/u;
    }

  } else {


    if (a > 1) { // using P(t): for large a and all p

      //
      // Works well for all a (in particular large a) and
      // not to large p
      //

      T b = a*(1 + p),
        t = 1/std::sqrt(b),
        t1, g, e;

      // simple iteration
      for (int i = 0; i < 4; ++i) {

        t1 = utils::sqr(1 + t);
        g = (1 + t*t*(1 + p/t1))/t1;
        e = g/b;

        t = (e + std::sqrt(e*e + 4*e))/2;
      }

     x = t;

    } else if (p < 0.1) { // using P : small p

      //
      //  R(t) = (1+t)^2 (1 + (a-1) t^2 - at^3)/(t^2 (-1 + a (1 + t)^3)
      //  R(t0 + u) = p => u(p)
      //  Works for small p and all a
      //

      T t = solve_cubic2(a),   // -1 + (-1 + a) t^2 + a t^3 == 0
        t2 = t*t,
        t3 = t2*t,
        t4 = t2*t2,
        t_1 = 1 + t,
        t_2 = t_1*t_1,
        t_3 = t_2*t_1,
        t_4 = t_2*t_2,
        t_6 = t_4*t_2,

        a2 = a*a;

      // derivatives of
      T r[3]={
          -t_1*(-2 + 2*t3 + a*t_1*(2 + t*(7 + t*(8 + t*t_2)))),
          3 + 2*t + t4 + a2*t_4*(3 + t*(14 + t*(25 + t*(20 + t*t_2)))) + a*(1 + t)*(-6 + t*(-22 + t*(-23 + t*(-4 + 7*t*t_2)))),
          2*(2 + t) - a*(12 + t*(51 + 2*t*(36 + t*(20 + t*(4 + 5*t*t_2)))) + a2*t_6*(4 + t*(23 + t*(54 + t*(65 + t*(40 + t*t_2))))) +
          4*a*t_3*(-3 + t*(-15 + t*(-27 + t*(-17 + 2*t*(1 + 2*t*t_2))))))
        },
        f = t3*utils::sqr(1 - a*t_3),
        fac = t*(-1 + a*t3);

        r[0] /= f; f *= fac;
        r[1] /= f; f *=  fac;
        r[2] /= f;

        r[1] /= r[0];
        r[2] /= r[0];

      // construct the inverse series
       T w = p/r[0],
         c1 = -r[1],
         c2 = 2*r[1]*r[1] - r[2],
         u = w*(1 + w*(c1 + w*c2));

      x = t + u;

    } else if (p > std::pow(2*a, -2./3) && a < 0.1) {

        // Using P expanded arond 1/c -1 in the series of 1/(p+2)
        // Works well for p > (2a)^(-2/3) and a > 0.1
        // with relative precision approx 0.02

        T c = std::pow(a, 1./3),
          w = 1/(3*utils::sqr(1 - c)*a*(p + 2)),
          r[3] ={
            1 + c*(-2 + c*(1 + (2 - c)*c)),
            (c*(-1 + c*(5 + c*(-13 + c*(27 + c*(-39 + c*(27 + c*(14 + c*(-34 + (20 - 4*c)*c)))))))))/(-1 + c),
            (c*c*(2 + c*(-16 + c*(74 + c*(-262 + c*(719 + c*(-1516 + c*(2531 + c*(-3184 + c*(2489 + c*(-164 + c*(-2036 + c*(2380 + c*(-1346 + (400 - 50*c)*c))))))))))))))/(3 + c*(-6 + 3*c))
          },
          u = w*(r[0] + w*(r[1] + w*r[2]));


        x = 1/c - 1 + u;

    } else { // using P(1/u): for small a

      //
      // Works well for all a (in particular small a) and
      // not for too large p
      //

      T b = a*(1 + p),
        u = b,
        u1, g, e;

      // simple iteration
      for (int i = 0; i < 10; ++i) {

        u1 = utils::sqr(1 + u);
        g = (1 + u*u*(1 + p/u1))/u1;
        e = b/g;

        u = 2*e/(std::sqrt(1 + 4*e) + 1);

      }

      x = 1/u;
    }
  }



  //
  // Polishing: using Newton-Rapson
  //

  #if defined(DEBUG)
  lagrange_point_L3_x = x;
  #endif

  int n = 0;  // counting steps

  T t, dt, f, df, c;

  #if 1
  // Using P:
  c = a*(1 + p);
  t = x;
  #else
  // Using Q:
  c = 4 + p - a*(1 + p);
  t = x/(1 + x);
  #endif

  do {

    #if 1 // using P
      f = 1 + t*(2 + t*(2 - c + p + t*(2 - 3*c + t*(1 - c*(3+t)))));
      df = 2 + t*(4 - 2*c + 2*p + t*(6 - 9*c + t*(4 - c*(12 + 5*t))));
    #else // using Q
      f = 1 + t*(-3 + t*(c + t*(-2 - 3*p + t*p*(3 - t))));
      df = -3 + t*(2*c + t*(-6 - 9*p + t*p*(12 - 5*t)));
    #endif

    dt = -f/df;

    t += dt;

    if (++n > 10) {
      std::cerr << "Slow convergence at F=" << F << " q=" << q << " !\n";

      #if 1
      x = t;
      #else
      x = t/(1 - t);
      #endif

      #if defined(DEBUG)
      lagrange_point_L3_n += n;
      #endif

      return delta*(1 + x);
    }

  } while (f != 0 && std::abs(dt) > std::abs(t)*eps + min);

  #if 1
  x = t;
  #else
  x = t/(1 - t);
  #endif

  #if defined(DEBUG)
  lagrange_point_L3_n += n;
  #endif

  return delta*(1 + x);
}

} // namespace gen_roche

