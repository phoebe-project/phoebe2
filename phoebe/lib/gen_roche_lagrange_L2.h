#pragma once
/*
  Library dealing with L2 lagrange point of the generalized Roche
  lobes/Kopal potential

  Omega =
    1/rho
    + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
    + 1/2 F^2(1 + q) rho^2 (1 - nu^2)

  where position in spherical coordinates is given as

  x = rho lambda      lambda = sin(theta) cos(phi)
  y = rho  mu         mu = sin(theta) sin(phi)
  z = rho nu          nu = cos(theta)


  L2 is positioned on the x-axis:

    (x, 0, 0)

  with x < 0

  Author: Martin Horvat,  March 2016
*/

#include <iostream>
#include <cmath>
#include <limits>

#include "utils.h"


namespace gen_roche {

/*
  Lagrange L2 point for the generalized potential on the line connecting
  both stars

      x < 0, y = 0, z = 0

  satisfying nabla omega = 0. The routine is tested in the regime of

    (q,F) in [0.01, 20] x [0.01 x 20]

  I. It is obtained by fining the root P(t) = 0 on interval t > 0 of
     the polynomial

    P(t) = 1 + 2 t + t^2 - (2q + b) t^3 - (q + 2 b) t^4 - b t^5

    a = F^2 delta^3
    b = a (1 + q)

  and solution is

    x = -t delta


  II. The other way is to transform polynomial

    P(s/(1-s)) = Q(s)/(1-s)^5

  with

    Q(s) = 1 - 3 s + 3 s^2 - (1 + 2q + a(1 + q)) s^3 + 3 q s^4 - q s^5
    a = F^2 delta^3

  and solve

    Q(s) = 0

  Then
    x = -delta s/(1-s)

  Input:
    q - mass ratio M2/M1
    F - synchronicity parameter
    delta - separation between the two objects
  Returned:
    x value of the L2
*/

#if defined(DEBUG)
double lagrange_point_L2_x; // approximation going into N-R iteration
int lagrange_point_L2_n;    // number of interations in Newton-Raphson
#endif

template <class T>
T lagrange_point_L2(
  const T & q,
  const T & F = 1,
  const T & delta = 1
) {

  //
  // Discussing F = 0
  //

  if (F == 0) {

    T s;

    if (q > 1e-4) {
      // Using Q: Solving  (1-s)^2 - q(2-s) = 0
      const T sqrt3 = std::sqrt(3);

      T q2 = q*q,
        f1 = std::cbrt(1 + 54*q2 - 6*sqrt3*q*std::sqrt(1 + 27*q2)),
        f2 = (3 - 2/q + (1/f1 + f1)/q)/3,
        f2s = std::sqrt(f2);

      s = (1 + f2s - std::sqrt(3 - 2/q - f2 + 2*(1 + 1/q)/f2s))/2;
    } else {
      T u = std::sqrt(q);
      s = 1 + u*(-1 + u*(1 + u*(-0.5 + u*(-1 + u*(3.625 + u*(-6 + u*(3.6875 + 11*u)))))));
    }

    return -delta*s/(1 - s);
  }

  T x,
    a = F*F*delta*delta*delta;

  const T eps = 10*std::numeric_limits<T>::epsilon();
  const T min = 10*std::numeric_limits<T>::min();

  //
  // Discussing the case a = 1
  //

  if (a == 1){  // t in [0,1]

    if (q > 1.5){ // using P
      T w = std::pow(3*(q + 1), -1./3),
        c[8] ={1.,
          0.3333333333333333,
          -0.1111111111111111,
          0.6172839506172839,
          0.17695473251028807,
          -0.4444444444444444,
          0.9399481786313062,
          -0.027485647513082356};

        x = w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

    } else if (q < 0.5) {
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
        s = q*(c[0] + q*(c[1] + q*(c[2] + q*(c[3] + q*(c[4] + q*(c[5] + q*(c[6] + q*c[7])))))));

       x = 1/(1 + s);

    } else { // q in [0.5, 1.5]

      T t0 = 0.69840614455492,
        w = q - 1,
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
    lagrange_point_L2_x = x;
    #endif

    int n = 0;    // counting steps

    bool turn = (x > 0.5);

    if (turn) x = 1 - x;

    T dx, P, dP;

    do {

      if (turn){
        P = -7*q + x*(12 + 26*q + x*(-24 - 37*q + x*(19 + 25*q + x*(-7 - 8*q + (1 + q)*x))));
        dP = 12 + 26*q + x*(-48 - 74*q + x*(57 + 75*q + x*(-28 - 32*q + 5*(1 + q)*x)));
      } else {
        P = 1 + x*(2 + x*(1 + x*(-1 - 3*q + x*(-2 - 3*q - (1 + q)*x))));
        dP = 2 + x*(2 + x*(-3 - 9*q + x*(-8 - 12*q - 5*(1 + q)*x)));
      }

      dx = -P/dP;

      x += dx;

      if (++n > 10) {
        std::cerr << "Slow convergence at F=" << F << " q=" << q << " !\n";
        #if defined(DEBUG)
        lagrange_point_L2_n += n;
        #endif
        return -delta*(turn ? 1 - x : x);
      }

    } while (P != 0 && std::abs(dx) > std::abs(x)*eps + min);


    if (turn) x = 1 - x;

    #if defined(DEBUG)
    lagrange_point_L2_n += n;
    #endif

    return -delta*x;
  }

  //
  //  a != 1
  //


  if (q == 1){

    if (a > 2) { // using Q
      T w = std::pow(2*a + 5,-1./3),
        s = w*(1 + (w-1)*w);

      x = s/(1 - s);
    } else if (a < 0.1){ // using Q
      T c[8] = {
          -0.3144040531308815,
          0.7451796412511179,
          -2.5574013258834385,
          10.393413194099757,
          -46.480222635440576,
          220.992329719314,
          -1096.31143523615,
          5610.772061807706
        },
        s0 = 0.5310100564595691,
        s = s0 + a*(c[0] + a*(c[1] + a*(c[2] + a*(c[3] + a*(c[4] + a*(c[5] + a*(c[6] + a*c[7])))))));

      x = s/(1 - s);
    } else {  // using P
      T c[8] = {
          -0.1687145366373403,
          0.08534802424313279,
          -0.051605226945983546,
          0.03395282072634135,
          -0.02348777505462479,
          0.016802697645681368,
          -0.012314365592402357,
          0.009191764021482736
        },
        w = a - 1,
        x0 = 0.69840614455492,
        u = w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

      x = x0 + u;
    }

  } else {          // q != 1

    if (a > 2){ // using Q

      T w = std::pow(3 + 2*q + a*(1 + q), -1./3),
        s = w*(1 + (w - 1)*w);

      x = s/(1 - s);

    } else if (q > 2) {  // using Q

      T w = std::pow((2 + a)*q, -1./3)/(3*(2 + a)),
        c[8]= {
          3,
          -9 - 9*a,
          9 + a*(18 + 27*a),27 + a*(-135 + a*(-216 + (-243 - 27*a)*a)),
          -54 + a*(-864 + a*(-216 + a*(162 + a*(1134 + 162*a)))),
          -243 + a*(729 + a*(15552 + a*(15552 + a*(11664 + (-3159 - 729*a)*a)))),
          540 + a*(12960 + a*(92016 + a*(27162 + a*(8505 + a*(-47628 + a*(28917 + a*(8748 + 486*a))))))),
          2835 + a*(-7371 + a*(244944 + a*(-339147 + a*(228744 + a*(177390 + a*(655614 + a*(-57591 + (-47385 - 3645*a)*a)))))))
        },
        s = (2 + a)*w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

      x = s/(1 - s);

    } else { // using P, small q and a < 2
      T b = a*(1 + q),
        t = std::pow(b, -1./3);

      // few steps of a simple iteration
      for (int i = 0; i < 4; ++i)
        t = std::pow(b + q*(2 + t)/((1 + t)*(1 + t)), -1./3);

      x = t;
    }

    //
    // I have another approximation for a very near to 0 (a < 0.1) and finite size q
    // As the range of convergence is not well defined is has limited use.

  }

  //
  // Polishing: using Newton-Rapson
  //

  #if defined(DEBUG)
  lagrange_point_L2_x = x;
  #endif

  int n = 0;  // counting steps

  T t, dt, f, df, c;

  #if 1
  // Using P:
  c = a*(1 + q);
  t = x;
  #else
  // Using Q:
  c = 1 + 2*q + a*(1 + q);
  t = x/(1 + x);
  #endif

  do {

    #if 1 // using P
      f = 1 + t*(2 + t*(1 + t*(-c - 2*q + t*(-2*c - q - c*t))));
      df = 2 + t*(2 + t*(-3*c - 6*q + t*(-8*c - 4*q - 5*c*t)));
    #else // using Q
      f = 1 + t*(-3 + t*(3 + t*(-c + t*(3*q - q*t))));
      df = -3 + t*(6 + t*(-3*c + t*(12*q - 5*q*t)));
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
      lagrange_point_L2_n += n;
      #endif

      return -delta*x;
    }

  } while (f != 0 && std::abs(dt) > std::abs(t)*eps + min);

  #if 1
  x = t;
  #else
  x = t/(1 - t);
  #endif

  #if defined(DEBUG)
  lagrange_point_L2_n += n;
  #endif

  return -delta*x;
}

} // namespace gen_roche

