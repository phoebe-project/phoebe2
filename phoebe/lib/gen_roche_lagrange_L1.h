#pragma once

/*
  Library dealing with L1 lagrange point of the generalized Roche
  lobes/Kopal potential

  Omega =
    1/rho
    + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
    + 1/2 F^2(1 + q) rho^2 (1 - nu^2)

  where position in spherical coordinates is given as

  x = rho lambda      lambda = sin(theta) cos(phi)
  y = rho  mu         mu = sin(theta) sin(phi)
  z = rho nu          nu = cos(theta)

  L1 is positioned on the x-axis:

    (x, 0, 0)

  with x in [0, delta].

  Author: Martin Horvat,  March 2016
*/


#include <iostream>
#include <cmath>
#include <limits>

#include "utils.h"

namespace gen_roche {

/*
  First real solution of (1-t)^2 (1-st) = z, s>0, z>0

    s t^3 - (1+ 2s) t^2 + (2+s) t + z-1 = 0

  write it as 3 degree polynomial

    t^3  + b t^2 + c t + d = 0
    b = - (1+ 2s)/s, c = (2+s)/s, d = (z-1)/s

  Ref: http://www.helioscorner.com/numerical-solution-of-a-cubic-equation-which-is-the-fastest-way/
*/

template <class T> T solve_cubic1(const T & s, const T & z){

  if (s == 1) return 1 - std::pow(z, 1./3);

  // with Tschirnhaus transformation : t = x - b/3
  //    x^3 + p x + q = 0
  T b = -2 - 1/s,
    c = 1 + 2/s,
    d = (z - 1)/s,

    p = c - b*b/3, // -(s-1)^2/(3 s^2) < 0 !
    q = b*(2*b*b/27 - c/3) + d, // (2 (-1 + s)^3)/(27 s^3) + z/s

    D = p*p*p/27 + q*q/4,// (z (4 (-1 + s)^3 + 27 s^2 z))/(108 s^4)

    A = 2*std::sqrt(std::abs(p)/3), phi;


  if (D <= 0){ // 3 real roots
    phi = std::acos(3*q/(A*p)) - 4*utils::m_pi; // smallest
    return A*std::cos(phi/3) - b/3;
  }

  // D > 0, only one real root
  if (p < 0){
    phi = acosh(-3*std::abs(q)/(A*p));
    return (q > 0 ? -A : A)*std::cosh(phi/3) - b/3;
  }

  // NEVER REACHES THIS PART

  // D > 0 && p > 0, only one real root
  phi = asinh(3*q/(A*p));
  return -A*std::sinh(phi/3) - b/3;
}

/*
  Lagrange L1 point for the generalized potential on the line connecting
  both stars

      x in [0, delta], y = 0, z = 0

  satisfying nabla omega = 0. The routine is tested in the regime of

    (q,F) in [0.01, 20] x [0.01 x 20]

  I. It is obtained by fining the root P(t) = 0 on interval t in (0,1)
     of the polynomial

    P(t) = 1 - 2t + t^2 - (2q + b) t^3 + (q + 2b) t^4 - b t^5

    b = a(1 + q)
    a = F^2 delta^3

  and solution is

    x = t delta

  II. The other way is to transform polynomial

    P(s/(1+s)) = Q(s)/(1+s)^5

  with

    Q(s) = 1 + 3s + 3 s^2 - (a - 1 + q(2 + a)) s^3 - 3q s^4 + q s^5
    a = F^2 delta^3

  and solve

    Q(s) = 0

  Then
    x = delta s/(1+s)

  Input:
    q - mass ratio M2/M1
    F - synchronicity parameter
    delta - separation between the two objects
  Returned:
    x value of the L1
*/

#if defined(DEBUG)
double lagrange_point_L1_x; // approximation going into N-R iteration
int lagrange_point_L1_n;    // number of interations in Newton-Raphson
#endif

template <class T>
T lagrange_point_L1(
  const T & q,
  const T & F = 1,
  const T & delta = 1
)
{

  //
  // Discussing F = 0
  //


  if (F == 0){

    //
    // t0 is exact root of P at a = 0, i.e. (1-t)^2+qt^3(t-2) = 0
    //
    const T sqrt3 = std::sqrt(3);

    T q2 = q*q,
      f1 = std::cbrt(1 + 54*q2 + 6*sqrt3*q*std::sqrt(1 + 27*q2)),
      f2 = std::sqrt((1 + f1*(3*q - 2 + f1))/(f1*q)),
      t0 = (3 + sqrt3*f2 - sqrt3*std::sqrt(6 - (1/f1 + f1 + 4)/q + 6*sqrt3*(1+q)/(q*f2)))/6;

    return delta*t0;
  }

  //
  // Discussing the case a = 1
  //

  T x,
    a = F*F*delta*delta*delta;

  const T eps = 10*std::numeric_limits<T>::epsilon();
  const T min = 10*std::numeric_limits<T>::min();


  if (a == 1) {   // simple Roche lobe/Kopal potential

    if (q == 1) return delta/2;

    if (0.5 <= q && q <= 1.5) {

      //
      // Approximation: using P and expansion around q=1
      // rel. precision for q in [0.5, 1.5]: <  1e-4
      //
      T w = q - 1,
        c[8] = {-0.10294117647058823,
                0.051470588235294115,
                -0.03248210031010165,
                0.022987856347505418,
                -0.017434752571749434,
                0.013852218889413575,
                -0.011379022393005212,
                0.009584546628778036
        };

      x = 0.5 + w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

    } else {

      //
      // Approximation: using Q and expanding around q->oo and q->0
      // rel. precision for q in [0.01, 100]: <  0.01
      //
      T w = (q > 1 ? std::pow(3*q,-1./3) : std::pow(q/3,1./3));

      // series in Horner form
      // w + (2 w^2)/3 + (2 w^3)/9 - (32 w^4)/81 - (50 w^5)/243 + (2 w^6)/9 + (3080 w^7)/6561 - (5330 w^8)/19683
      T s = w*(1 + w*(2./3 + w*(2./9 + w*(-32./81 + w*(-50./243 + w*(2./9 + w*(3080./6561 - 5330.*w/19683)))))));

      x = (q > 1 ? s : 1)/(1 + s);
    }

    //
    // Polishing: using Newton-Raphson
    // accuracy for q in [0.01, 100] < 10^-15, with max n ~ 4
    //
    #if defined(DEBUG)
    lagrange_point_L1_x = x;
    #endif

    bool turn = (q<1);

    if (turn) x = 1 - x;

    int n = 0;    // counting steps

    T dx, P, dP;

    do {

      if (turn) {   // working with P(1-x)
        P = -q + x*(2*q + x*(-q + x*(3 + q + x*(-3 - 2*q + (1 + q)*x))));
        dP = 2*q + x*(-2*q + x*(9 + 3*q + x*(-12 - 8*q + (5 + 5*q)*x)));
      } else {      // working with P(x)
        P = 1 + x*(-2 + x*(1 + x*(-1 - 3*q + x*(2 + 3*q + (-1 - q)*x))));
        dP = -2 + x*(2 + x*(-3 - 9*q + x*(8 + 12*q + (-5 - 5*q)*x)));
      }

      dx = -P/dP;

      x += dx;

      if (++n > 10) {
        std::cerr << "Slow convergence at a=" << 1 << " q=" << q << " !\n";
        #if defined(DEBUG)
        lagrange_point_L1_n += n;
        #endif
        return delta*(turn ? 1 - x : x);;
      }

    } while (P != 0 && std::abs(dx) > std::abs(x)*eps + min);

    if (turn) x = 1 - x;

    #if defined(DEBUG)
    lagrange_point_L1_n += n;
    #endif

    return delta*x;
  }

  //
  // Discussing the case a != 1
  //

  if (q == 1) {

    if (a < 0.5) {
      //
      // Approximation: expansion of P at q = 1 for a->0
      // rel. precision for a < 0.5: < 1e-10
      //
      T t0 = 0.5310100564595691,
        c[8]= {
          -0.032432373069687326,
          0.0012009812247423704,
          0.00026677335735642835,
          -0.000044242336209068785,
          -2.6016936984989166e-6,
          1.5097474266330576e-6,
          -6.688040686774399e-8,
          -4.438264579211867e-8
        };

      x = t0 + a*(c[0] + a*(c[1] + a*(c[2] + a*(c[3] + a*(c[4] + a*(c[5] + a*(c[6] + a*c[7])))))));
    } else if (0.5 <= a && a <= 6) {
      //
      // Approximation: expansion of P at q = 1 around a = 1
      // rel. precision for a < 6: < 1e-3
      //
      T w = a - 1,
        c[8] = {-0.029411764705882353,
              0.0017301038062283738,
              0.00008979777540977718,
              -0.00003908844341366772,
              3.5275739169727247e-6,
              4.3127933065985745e-7,
              -1.610562908586607e-7,
              1.3085721781321937e-8
        };

      x = 0.5 + w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));
    } else {
      //
      // Approximation: expansion of Q at q = 1 for a->oo
      // rel. precision for a > 6: < 0.015
      //
      T w = std::pow(2*a,-1./3),
        s = w*(1 + w*(1 + w*(1 + w*(1./3 + w*(-4./3 + w*(-13./3 + (-73 - 91*w)*w/9))))));

      x = s/(1 + s);
    }

  } else {

    //
    // First dealing with large arguments
    //

    if (a > 4) {         // large a and arbitrary q
      //
      // Approximation: using Q and expanding around a->oo
      // rel. precision for a > 10 for all q: < 0.002
      // A very good approximation
      //
      T w = std::pow(a - 1 + q*(2 + a), -1./3),
        s = w*(1 + w*(1 + w*(1 + w*(2./3 + w*(1./3 - q + w*(-10*q/3. + w*(-1./9 - 22*q/3. -(1./9 + 35*q/3.)*w)))))));

      x = s/(1 + s);

    } else if (q > 2) {   // large q, finite a
      //
      // Approximation: using Q and expanding around q->oo
      // rel. precision for q > 2 and a < 100^2: < 0.02
      //

      T w = std::pow((2 + a)*q, -1./3)/(2 + a),
        c[8] = {
          1,
          1 + a,
          (1 + a*(2 + 3*a))/3,
          (-1 + a*(-11 + a*(-16 + (-3 - a)*a)))/3,
          -2*(1 + a*(-8 + a*(16 + a*(51 + a*(12 + 3*a)))))/9,
          (1 + a*(13 + a*(88 + a*(92 + a*(-22 + (-7 - 3*a)*a)))))/3,
          (20 + a*(-240 + a*(-264 + a*(8386 + a*(14265 + a*(4176 + a*(1251 + a*(108 + 18*a))))))))/81,
          (-35 + a*(-581 + a*(-10176 + a*(-36445 + a*(-25084 + a*(14898 + a*(6402 + a*(2691 + a*(315 + 45*a)))))))))/81
        },
        s = (2 + a)*w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

      x = s/(1 + s);

    }  else if (q < 1 && a > 1) {  // small q, finite a

      //
      // Approximation: using P=0 in limit q->0 expressed in the form
      //  q = (1-t)^2 (1-ct) R(t)  R(t) = (1 + (ct)^2 + (ct)^2)/(t^3 (2-t))
      //  c = (a(1+q))^(1/3)
      // and solved by iteration
      //    q/R(t_{n-1}) = (1-t_n)^2 (1-c t_n)
      // Convergence is usually good for small q < 0.5 and best for c > 1

      T c = std::cbrt(a*(1 + q)),
        t = (c > 1 ? 1/c : 1),
        R;

      for (int i = 0; i < 4; ++i) {
        R = (1 + c*t*(1 + c*t))/(t*t*t*(2 - t));
        t = solve_cubic1(c, q/R);
      }

      x = t;

    } else if (a < 1) {  // small a and finite size q

      //
      // Working with P=0 in the limit a->0 expressed in the form
      //   R(t) := ((1-t)^2+qt^3(t-2))/t^3(1-t)^2 = b := a(1+q)
      // Works good for small a and q > 0.1

      T t0 = lagrange_point_L1(q, T(0), T(1));

      //
      // Derivaties of R(t) at t0 /n!
      //
      int A = -1,
          B = 3;

      T r[8],
        t1 = std::pow(t0 - 1, 2),
        t2 = std::pow(t0, 3);

      for (int i = 0; i < 8; ++i) {
        t1 *= t0 - 1;
        t2 *= t0;

        r[i] = A*(q*(t0 - (i + 3))/t1 + B/t2);

        A = -A;
        B += i + 3;
      }

      //
      // Building the inverse (Taylor) series
      //
      for (int i = 1; i < 8; ++i) r[i] /= r[0];

      T w = a*(1 + q)/r[0],
        c[8] = {
          1,
          -r[1],
          2*r[1]*r[1] - r[2],
          r[1]*(-5*r[1]*r[1] + 5*r[2]) - r[3],
          3*r[2]*r[2] + r[1]*(r[1]*(14*r[1]*r[1] - 21*r[2]) + 6*r[3]) - r[4],
          7*r[2]*r[3] + r[1]*(-28*r[2]*r[2] + r[1]*(r[1]*(-42*r[1]*r[1] + 84*r[2]) - 28*r[3]) + 7*r[4]) - r[5],
          4*r[3]*r[3] + r[2]*(-12*r[2]*r[2] + 8*r[4]) + r[1]*(-72*r[2]*r[3] + r[1]*(180*r[2]*r[2] + r[1]*(r[1]*(132*r[1]*r[1] - 330*r[2]) + 120*r[3]) - 36*r[4]) + 8*r[5]) - r[6],
          9*r[3]*r[4] + r[2]*(-45*r[2]*r[3] + 9*r[5]) + r[1]*(-45*r[3]*r[3] + r[2]*(165*r[2]*r[2] - 90*r[4]) + r[1]*(495*r[2]*r[3] + r[1]*(-990*r[2]*r[2] + r[1]*(r[1]*(-429*r[1]*r[1] + 1287*r[2]) - 495*r[3]) + 165*r[4]) - 45*r[5]) + 9*r[6]) - r[7]
        },
        s = w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

      x = t0 + s;

    } else { //  a in [1, 4.5] and q in [0.5, 2]

      //
      // Working with P in the form in the limit a->1:
      //   R(t) := P(t,q,a=1)/t^3(1-t)^2 = (1 + q)(a - 1)
      // Good approximation for a large range around a = 1 and q > 0.1

      //
      // t0 is exact root of P at a = 1
      //

      T t0 = lagrange_point_L1(q);

      //
      // Derivaties of R(t) at t0 /n!
      //
      int A = -1,
          B = 3;

      T r[8],
        t1 = std::pow(t0 - 1, 2),
        t2 = std::pow(t0, 3);

      for (int i = 0; i < 8; ++i) {
        t1 *= t0 - 1;
        t2 *= t0;

        r[i] = A*(q*(t0 - (i + 3))/t1 + B/t2);

        A = -A;
        B += i + 3;
      }

      //
      // Building the inverse (Taylor) series
      //
      for (int i = 1; i < 8; ++i) r[i] /= r[0];

      T w = (1 + q)*(a - 1)/r[0],
        c[8] = {
          1,
          -r[1],
          2*r[1]*r[1] - r[2],
          r[1]*(-5*r[1]*r[1] + 5*r[2]) - r[3],
          3*r[2]*r[2] + r[1]*(r[1]*(14*r[1]*r[1] - 21*r[2]) + 6*r[3]) - r[4],
          7*r[2]*r[3] + r[1]*(-28*r[2]*r[2] + r[1]*(r[1]*(-42*r[1]*r[1] + 84*r[2]) - 28*r[3]) + 7*r[4]) - r[5],
          4*r[3]*r[3] + r[2]*(-12*r[2]*r[2] + 8*r[4]) + r[1]*(-72*r[2]*r[3] + r[1]*(180*r[2]*r[2] + r[1]*(r[1]*(132*r[1]*r[1] - 330*r[2]) + 120*r[3]) - 36*r[4]) + 8*r[5]) - r[6],
          9*r[3]*r[4] + r[2]*(-45*r[2]*r[3] + 9*r[5]) + r[1]*(-45*r[3]*r[3] + r[2]*(165*r[2]*r[2] - 90*r[4]) + r[1]*(495*r[2]*r[3] + r[1]*(-990*r[2]*r[2] + r[1]*(r[1]*(-429*r[1]*r[1] + 1287*r[2]) - 495*r[3]) + 165*r[4]) - 45*r[5]) + 9*r[6]) - r[7]
        },
        s = w*(c[0] + w*(c[1] + w*(c[2] + w*(c[3] + w*(c[4] + w*(c[5] + w*(c[6] + w*c[7])))))));

      x = t0 + s;
    }
  }

  //
  // Polishing: using Newton-Rapson
  // accuracy for q in [0.01, 100] < 10^-15, with max n ~ 4
  //

  #if defined(DEBUG)
  lagrange_point_L1_x = x;
  #endif

  bool turn = (x > 0.5);

  if (turn) x = 1 - x;

  int n = 0;  // counting steps

  T dx, P, dP, b = a*(1 + q);

  do {

    if (turn){   // working with P(1-x)
      P = -q + x*(2*q + x*(1 - b + x*(3*b - 2*q + x*(-3*b + q + b*x))));
      dP = 2*q + x*(2 - 2*b + x*(9*b - 6*q + x*(-12*b + 4*q + 5*b*x)));
    } else {      // working with P(x)
      P = 1 + x*(-2 + x*(1 + x*(-b - 2*q + x*(2*b + q - b*x))));
      dP = -2 + x*(2 + x*(-3*b - 6*q + x*(8*b + 4*q - 5*b*x)));
    }

    dx = -P/dP;

    x += dx;

    if (++n > 10) {
      std::cerr << "Slow convergence at a=" << a << " q=" << q << " !\n";
      #if defined(DEBUG)
      lagrange_point_L1_n += n;
      #endif
      return delta*(turn ? 1 - x : x);
    }

  } while (P != 0 && std::abs(dx) > std::abs(x)*eps + min);

  #if defined(DEBUG)
  lagrange_point_L1_n += n;
  #endif

  if (turn) x = 1 - x;

  return delta*x;
}

} // namespace gen_roche
