#pragma once

/*
  Commonly used routines.

  Author: Martin Horvat, April 2016
*/

#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>

#include "sincos.h"

// macro to make aliases of functions
// usage: ALIAS_TEMPLATE_FUNCTION(fun_alias, fun)
#define ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF) \
template<typename... Args> \
inline auto highLevelF(Args&&... args) -> decltype(lowLevelF(std::forward<Args>(args)...)) \
{ \
    return lowLevelF(std::forward<Args>(args)...); \
}

namespace utils {

  const double m_pi = 3.14159265358979323846264338327950419984;  // pi
  const double m_2pi = 6.2831853071795864769252867665590083999;  // 2 pi
  const double m_4pi = 12.5663706143591729538505735331180167998; // 4 pi
  const double m_4pi3 = 4.18879020478639098461685784437267226645; // 4 pi/3
  const double m_pi3 = 1.04719755119659774615421446109316806665; // pi/3
  const double m_pi2 = 1.57079632679489661923132169163975209992; // pi/2
  const double m_e = 2.71828182845904523533978449066641588615; // e
  const double m_1_e = 0.36787944117144232159552377016146086744; // 1/e

  template<class T>
  constexpr T pi() {return T(3.14159265358979323846264338327950419984L);};

  /*
    Approximation formula for ArcCos

    Input:
      x in [-1,1]
    Return:
      arccos(x) in [0,pi]

    Ref:
      p.81 Handbook of Mathematical Functions, by M. Abramowitz and I. Stegun
  */
  float __acosf(const float & x) {

    if (x ==  0) return 1.57079632679489;
    if (x >=  1) return 0;
    if (x <= -1) return 3.14159265358979;

    float
      t = std::abs(x),
      s = std::sqrt(1-t)*(1.5707288 + t*(-0.2121144 + (0.074261 - 0.0187293*t)*t));

    return (x > 0 ? s : 3.14159265358979 - s);
  }

  /*
    Return square of the value.

    Input: x

    Return: x^2
  */
  template <class T>
  T sqr(const T &x){ return x*x; }

  /*
    Return cube of the value.

    Input: x

    Return: x^3
  */
  template <class T>
  T cube(const T &x){ return x*x*x; }


  /*
    Calculate the max of 3D vector.

    Input:
      x, y, z

    Return:
      max(x,y,z}
  */

  template <class T> T max3(const T & x, const  T & y, const T & z){

    T t;

    if (x > y) t = x; else t = y;

    if (z > t) return z;

    return t;
  }


  /*
    Calculate the max of 3D vector.

    Input:
      x[3] -- vector of 3 values

    Return:
      max(x,y,z}
  */
  template <class T> T max3(T x[3]){

    T t;

    if (x[0] > x[1]) t = x[0]; else t = x[1];

    if (x[2] > t) return x[2];

    return t;
  }

  /*
    Calculate the min of 3D vector.

    Input:
      x, y, z

    Return:
      min(x,y,z}
  */

  template <class T>
  T min3(const T & x, const T & y, const T & z){

    T t;

    if (x < y) t = x; else t = y;

    if (z < t) return z;

    return t;
  }

  /*
    Calculate the min of 3D vector.

    Input:
      x[3]

    Return:
      min(x,y,z}
  */

  template <class T> T min3(T x[3]){

    T t;

    if (x[0] < x[1]) t = x[0]; else t = x[1];

    if (x[2] < t) return x[2];

    return t;
  }

  /*
    Calculate the min and max of 3D vector.

    Input:
      x, y, z

    Output:
      mm[2] = {min, max}
  */

  template <class T>
  void minmax3(const T& x, const T & y, const T & z, T mm[2]){

    if (x > y) {
      mm[0] = y;
      mm[1] = x;
    } else {
      mm[0] = x;
      mm[1] = y;
    }

    if (mm[0] > z)
      mm[0] = z;
    else if (mm[1] < z)
      mm[1] = z;
  }

  /*
    Calculate the min and max of 3D vector.

    Input:
      x[3]

    Output:
      mm[2] = {min, max}
  */

  template <class T> void minmax3(T x[3], T mm[2]){


    if (x[0] > x[1]) {
      mm[0] = x[1];
      mm[1] = x[0];
    } else {
      mm[0] = x[0];
      mm[1] = x[1];
    }

    if (mm[0] > x[2])
      mm[0] = x[2];
    else if (mm[1] < x[2])
      mm[1] = x[2];
  }

  // y  = A x
  template <class T> void dot3D(T A[3][3], T x[3], T y[3]) {
    for (int i = 0; i < 3; ++i)
      y[i] = A[i][0]*x[0] + A[i][1]*x[1] + A[i][2]*x[2];
  }

  // y^T  = x^T A
  template <class T> void dot3D(T x[3], T A[3][3], T y[3]) {
    for (int i = 0; i < 3; ++i)
      y[i] = A[0][i]*x[0] + A[1][i]*x[1] + A[2][i]*x[2];
  }

  // x^T.y
  template <class T, class F> T inline dot3D(T x[3], F y[3]) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }

  // z = x cross y
  template <class T> void cross3D(T x[3], T y[3], T z[3]) {
    z[0] = x[1]*y[2] - x[2]*y[1];
    z[1] = x[2]*y[0] - x[0]*y[2];
    z[2] = x[0]*y[1] - x[1]*y[0];
  }

    // z = x cross y
  template <class T> T cross2D(T x[2], T y[2]) {
    return x[0]*y[1] - x[1]*y[0];
  }


  // solve for x: A x = b
  template <class T> bool solve2D(T A[2][2], T b[2], T x[2]){
    T det = A[0][0]*A[1][1] - A[1][0]*A[0][1];

    if (det == 0) return false;

    x[0] = (A[1][1]*b[0] - A[0][1]*b[1])/det;
    x[1] = (A[0][0]*b[1] - A[1][0]*b[0])/det;
    return true;
  }

  // solve for x: A x = b
  template <class T> bool solve3D(T A[3][3], T b[3], T x[3]){

    T det =
      A[0][2]*(-A[1][1]*A[2][0]+A[1][0]*A[2][1])+
      A[0][1]*(+A[1][2]*A[2][0]-A[1][0]*A[2][2])+
      A[0][0]*(-A[1][2]*A[2][1]+A[1][1]*A[2][2]);

    if (det == 0) return false;

    det = 1/det;

    x[0] = det*(b[2]*(-A[0][2]*A[1][1]+A[0][1]*A[1][2])+b[1]*(+A[0][2]*A[2][1]-A[0][1]*A[2][2])+b[0]*(-A[1][2]*A[2][1]+A[1][1]*A[2][2]));
    x[1] = det*(b[2]*(+A[0][2]*A[1][0]-A[0][0]*A[1][2])+b[1]*(-A[0][2]*A[2][0]+A[0][0]*A[2][2])+b[0]*(+A[1][2]*A[2][0]-A[1][0]*A[2][2]));
    x[2] = det*(b[2]*(-A[0][1]*A[1][0]+A[0][0]*A[1][1])+b[1]*(+A[0][1]*A[2][0]-A[0][0]*A[2][1])+b[0]*(-A[1][1]*A[2][0]+A[1][0]*A[2][1]));

    return true;
  }


  // solve for x: x^t A = b^t
  template <class T> bool solve3D(T b[3], T A[3][3], T x[3]){

    T det =
      A[0][2]*(-A[1][1]*A[2][0]+A[1][0]*A[2][1])+
      A[0][1]*(+A[1][2]*A[2][0]-A[1][0]*A[2][2])+
      A[0][0]*(-A[1][2]*A[2][1]+A[1][1]*A[2][2]);

    if (det == 0) return false;

    det = 1/det;

    x[0] = det*(b[2]*(-A[1][1]*A[2][0]+A[1][0]*A[2][1])+b[1]*(+A[1][2]*A[2][0]-A[1][0]*A[2][2])+b[0]*(-A[1][2]*A[2][1]+A[1][1]*A[2][2]));
    x[1] = det*(b[2]*(+A[0][1]*A[2][0]-A[0][0]*A[2][1])+b[1]*(-A[0][2]*A[2][0]+A[0][0]*A[2][2])+b[0]*(+A[0][2]*A[2][1]-A[0][1]*A[2][2]));
    x[2] = det*(b[2]*(-A[0][1]*A[1][0]+A[0][0]*A[1][1])+b[1]*(+A[0][2]*A[1][0]-A[0][0]*A[1][2])+b[0]*(-A[0][2]*A[1][1]+A[0][1]*A[1][2]));

    return true;
  }

  // z = x +  a*y
  template <class T, class F> inline void fma3D(T x[3], F y[3], const T & a, T z[3]) {
    for (int i = 0; i < 3; ++i) z[i] = x[i] + a*y[i];
  }


  // z = x - y
  template <class T> inline void sub3D(T x[3], T y[3], T z[3]) {
    for (int i = 0; i < 3; ++i) z[i] = x[i] - y[i];
  }


  // return x.x
  template <class T> inline T norm2(T x[3]) {
    // for higher precision we can sort and
    // do Kahan's compensation
    return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
  }

  /*
    Calculate L2 norm of 3D vector

    Input:
      x, y, z

    Return:
      std::sqrt(x*x + y*y + z*z)

    Ref: fallowing idea from
      BLAS  http://www.netlib.org/blas/snrm2.f
  */
  template <class T>
  T hypot3 (const T & x, const T& y, const T& z){

    T a[3] = {std::abs(x), std::abs(y), std::abs(z)};

    if (a[0] < a[1]) std::swap(a[0], a[1]);
    if (a[0] < a[2]) std::swap(a[0], a[2]);

    a[1] /= a[0];
    a[2] /= a[0];

    T t = a[1]*a[1] + a[2]*a[2];

    return a[0]*std::sqrt(1 + t);
  }


  /*
    Calculate L2 norm of 3D vector

    Input:
      x[3]

    Return:
      std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])

    Ref: fallowing idea from
      BLAS  http://www.netlib.org/blas/snrm2.f
  */
  #if 0
  template <class T>  T hypot3 (T x[3]){

    T a[3] = {std::abs(x[0]), std::abs(x[1]), std::abs(x[2])}, t;

    if (a[0] < a[1]) { t = a[0]; a[0] = a[1]; a[1] = t;}
    if (a[0] < a[2]) { t = a[0]; a[0] = a[2]; a[2] = t;}

    a[1] /= a[0];
    a[2] /= a[0];

    t = a[1]*a[1] + a[2]*a[2];

    return a[0]*std::sqrt(1 + t);
  }

  #else
  template <class T> T hypot3 (T x[3]){

    T a[3] = {std::abs(x[0]), std::abs(x[1]), std::abs(x[2])},
      *p[3] = {a, a + 1, a + 2};

    if (*(p[0]) < *(p[1])) std::swap(p[0], p[1]);
    if (*(p[0]) < *(p[2])) std::swap(p[0], p[2]);

    *(p[1]) /= *(p[0]);
    *(p[2]) /= *(p[0]);

    T t = (*p[1])*(*p[1]) + (*p[2])*(*p[2]);

    return (*(p[0]))*std::sqrt(1 + t);
  }

  #endif


  /* Swap two elements

    Input:
    x, y

    Output
    x, y = (y,x)

  */
  template <class T> void swap (T & x, T & y) {
    T z = x;
    x = y;
    y = z;
  }

  /*
    Swap elements of vectors

    Input:
    x, y

    Output
    x, y = (y,x)

  */
  template <class T> void swap_array (T *x, T *y, int n) {
    if (x != y) {
      T z;
      for (T *xe = x + n; x != xe; ++x, ++y) z = *x, *x = *y, *y = z;
    }
  }


  template <class T, int n> inline void swap_array (T *x, T *y) {
    if (x != y) {
      T z;
      for (T *xe = x + n; x != xe; ++x, ++y) z = *x, *x = *y, *y = z;
    }
  }


  /*
  Sort 3D vector in accending order and return the new index order.

  Input:
    x[3] - 3 values

  Output:
    ind[3] - indices of elements of x in ordered state
  */
  template <class T> void sort3ind(T x[3], int ind[3]){

    T a[3] = {x[0], x[1], x[2]};

    for (int i = 0; i < 3; ++i) ind[i] = i;

    if (a[0] > a[1]) {
      swap(a[0], a[1]);
      swap(ind[0], ind[1]);
    }

    if (a[1] > a[2]) {
      swap(a[1], a[2]);
      swap(ind[1], ind[2]);
    }

    if (a[0] > a[1]) {
      //swap(a[0], a[1]); // Not necessary.
      swap(ind[0], ind[1]);
    }
  }

  /*
    Rolish real roots of n-degree polynomial

      a[0] + a[1]x + a[2]x^2 + ...+a[n]x^n

    with a Newton-Raphson iteration.

    Input:
      n - degree of polynomial
      a[n+1] - vector of coefficients
      roots - vector of roots

    Output:
      roots - vector of polished roots
  */
   template <class T>
   void polish(const int & n, T *a, std::vector<T> & roots, const bool multi_roots = true){

      const int iter_max = 20;

      const T eps_2 = std::numeric_limits<T>::epsilon()/2;
      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();

      int  ir = 0;

      for (auto && xi: roots) {

        int it = 0;

        long double dx, x = xi, f, df, d2f, e;

        do {

          if (multi_roots) {

            // Horner algorithm to compute value, derivative and second derivative
            // http://www.ece.rice.edu/dsp/software/FVHDP/horner2.pdf
            f = a[n], df = d2f = e = 0;
            for (int i = n - 1; i >= 0; --i) {
              d2f = df + x*d2f;
              df = f + x*df;

              // maximal expected error
              e = e*std::abs(x) + eps_2*(2*std::abs(x*f) + std::abs(a[i]));
              f  = a[i] + x*f;
            }
            d2f *= 2;

            // Newton-Raphson step for multiple roots
            // https://www.math.uwaterloo.ca/~wgilbert/Research/GilbertNewtonMultiple.pdf
            x -= (dx = f*df/(df*df - f*d2f));
          } else {

            // Horner algorithm to compute value and derivative
            // http://www.physics.utah.edu/~detar/lessons/c++/array/node4.html
            f = a[n], df = e = 0;
            for (int i = n - 1; i >= 0; --i) {
              df = f + x*df;
              // maximal expected error
              e = e*std::abs(x) + eps_2*(2*std::abs(x*f) + std::abs(a[i]));
              f  = a[i] + x*f;
            }

            // Newton-Raphson step
            x -= (dx = f/df);
          }

          #if 0
          std::cerr.precision(16);
          std::cerr << std::scientific;
          std::cerr
            << "I=:" << it << '\t' << x << '\t'
            << dx << '\t' << f << '\t' << df << '\t' << e << '\n';
          #endif


        } while (std::abs(f) > 0.5*e && std::abs(dx) > eps*std::abs(x) + min && (++it) < iter_max);

        //std::cout << "-----\n";

        if (it == iter_max) {
          std::cerr << "Warning: Root polishing did not succeed\n";

          std::cerr.precision(std::numeric_limits<T>::digits10);
          std::cerr << std::scientific;

          std::cerr
            << "i=" << ir << '\n'
            << "n=" << n << '\n'
            << "x=" << x << '\n'
            << "xi=" << xi << '\n'
            << "dx=" << dx << '\n'
            << "f=" << f << '\n'
            << "eps=" << eps << '\n'
            << "min="<< min << '\n';

          for (int i = 0; i <= n; ++i) std::cerr << a[i] << '\t';
          std::cerr << '\n';
        }

        xi = x;
        ++ir;
      }
   }

  /*
    Real roots of the quadratic equation

      a[2] x^2 + a[1] x + a[0] = 0

    Input:
      a[3] -- cofficients of the polynomial

    Output:
      roots -- list of real roots sorted ascending order
  */
  template <class T>
  void solve_quadratic(T a[3], std::vector<T> & roots){

    roots.clear();

    if (a[2] != 0){

      //
      // Solving quadratic equation
      // x^2 + b x + c = 0
      //

      T b = a[1]/a[2],
        c = a[0]/a[2];

      T D = b*b - 4*c;

      if (D >= 0) {

        if (D == 0)

          roots.push_back(-b/2);

        else {
          D = std::sqrt(D);

          T x1 = -(b + (b > 0 ? D : -D))/2,
            x2 = c/x1;

          if (x1 < x2) {
            roots.push_back(x1);
            roots.push_back(x2);
          } else {
            roots.push_back(x2);
            roots.push_back(x2);
          }
        }
      }

    } else {
      //
      // Solving linear equation
      //
      roots.push_back(-a[0]/a[1]);
    }
  }

  /*
    Real roots of the cubic equation
      a[3] x^3 + a[2] x^2 + a[1] x + a[0] = 0

    Input:
      a[4] -- cofficients of the polynomial

    Output:
      roots -- vector of real roots sorted ascending order

    Using: Trigonometric method

    Refs:
      https://en.wikipedia.org/wiki/Cubic_function
      http://mathworld.wolfram.com/CubicFormula.html
  */

  template <class T>
  void solve_cubic(T a[4], std::vector<T> & roots) {

    roots.clear();

    const T eps = std::numeric_limits<T>::epsilon();

    if (a[3] != 0) {

      //
      // Working with a cubic equation
      //

      // rewritten into polynomial
      // x^3 + b x^2 + c x + d = 0
      T b = a[2]/a[3],
        c = a[1]/a[3],
        d = a[0]/a[3];


      // Tschirnhaus transformation : t = x - b/3
      //    x^3 + p x + q = 0
      T p = c - b*b/3,
        q = b*(2*b*b/9 - c)/3 + d,

        D = p*p*p/27 + q*q/4,

        A = 2*std::sqrt(std::abs(p)/3), phi;

      #if 0
      std::cerr.precision(16);
      std::cerr << std::scientific;

      std::cerr
        << "cubic::\n"
        << "a=" << d  << ' ' << c << ' ' << b << ' ' << 1 << '\n'
        << "D=" << D << " A=" << A  << '\n'
        << "q=" << q << " p=" << p << '\n';
      #endif

      if (D <= 0 || std::abs(D) < eps){ // 3 real roots, of 1 real roots if (p=q=0)

        if (p == 0 && q == 0)
          roots.push_back(-b/3);
        else {

          T r = 3*q/(A*p);

          #if 0
          std::cerr
            << "cubic::\n"
            << "r=" << r << '\n';
          #endif

          phi = (std::abs(r) > 1 ? 0 : std::acos(r));

          for (int i = 2; i >= 0; --i) {
            #if 0
            T t;
            roots.push_back(t = A*std::cos((phi - m_2pi*i)/3) - b/3);
            std::cerr << "cubic::x=" << t << '\n';
            #else
            roots.push_back(A*std::cos((phi - m_2pi*i)/3) - b/3);
            #endif
          }
        }
      } else {

        // D > 0, only one real root
        if (p < 0){

          phi = acosh(-3*std::abs(q)/(A*p));
          roots.push_back((q > 0 ? -A : A)*std::cosh(phi/3) - b/3);

        } else if (p == 0) {

          roots.push_back( std::cbrt(q) - b/3);

        } else {  // p > 0
          phi = asinh(3*q/(A*p));
          roots.push_back(-A*std::sinh(phi/3) - b/3);
        }
      }

      polish(3, a, roots);

    } else {
      //
      // Working with a quadratic equation
      //  a[2] x^2 + a[1] x + a[0] = 0
      //
      solve_quadratic(a, roots);
    }
  }

  /*
    Real roots of the quartic equation

      a[4] x^4 + a[3] x^3 + a[2] x^2 + a[1] x + a[0] = 0

      Input:
        a[5] -- coefficients of the polynomial

      Output:
        roots -- vector of real roots sorted ascending order

    Using: Ferrari's solutions

    Ref:
      https://en.wikipedia.org/wiki/Quartic_function
      http://mathworld.wolfram.com/QuarticEquation.html
      Olver F W, et. al - NIST Handbook Mathematical Functions (CUP, 2010)
  */
  template <class T>
  void solve_quartic(T a[5], std::vector<T> & roots)
  {
    roots.clear();

    if (a[4] != 0) {

      //
      // Working with a quartic equation
      //

      //  x^4 + b x^3 + c x^2 + d x + e = 0

      T b = a[3]/a[4],
        c = a[2]/a[4],
        d = a[1]/a[4],
        e = a[0]/a[4];

      // getting depressed quartic: x = y - b/4
      // y^4 + p y^2  + q y + r = 0

      T  b2= b*b,
         p = c - 3*b2/8,
         q = b*(b2/8 - c/2) + d,
         r = b*(b*(-3*b2/256 + c/16) - d/4) + e;

      if (q == 0) { // Biquadratic equations

        T s[3] = {r, p, 1};

        std::vector<T> roots1;
        solve_quadratic(s, roots1);

        for (auto && v : roots1)
          if (v >= 0) {
            roots.push_back(std::sqrt(v) - b/4);
            roots.push_back(-std::sqrt(v) - b/4);
          }

      } else {
        //
        // creating a resolvent cubic
        //  m^3 + 5/2 p m^2 + (2p^2-r)m + (p^3/2 - pr/2 -q^2/8) = 0
        //

        T s[4] = { p*(p*p -r)/2 -q*q/8, 2*p*p - r, 5*p/2, 1};

        std::vector<T> roots1;
        solve_cubic(s, roots1);

        // using the one which is positive
        T t = -1;
        for (auto && r: roots1) if ((t = 2*r + p) > 0) break;

        if (t > 0) {

          T st = std::sqrt(t)/2, b_ = b/4, t1;

          for (int s1 = -1; s1 <= 1; s1 += 2){

            t1 = -(2*p + t + s1*q/st);

            if (t1 >= 0) {
              t1 = std::sqrt(std::abs(t1))/2;
              for (int s2 = -1; s2 <= 1; s2 += 2)
                roots.push_back(s1*st + s2*t1 - b_);
            }
          }
        }
      }

      //
      // polish roots with Newton-Raphson iteration as
      // rounding error errors in Ferrari's method are significant
      //
      polish(4, a, roots);

      std::sort(roots.begin(), roots.end());
    } else {
      //
      // Working with a cubic equation
      //  a[3] x^3 + a[2] x^2 + a[1] x + a[0] = 0
      //
      solve_cubic(a, roots);
    }
  }

  /*
    Create/reserve C-style matrix

    Input:
      nrow - number of rows
      ncol - number of columns

    Return:
      m[nrow][ncol]
  */
  template <class T> T** matrix(const int & nrow, const int & ncol) {

    T **m = new T* [nrow];

    m[0] = new T [nrow*ncol];

    for (int i = 1; i < nrow; ++i) m[i] = m[i-1]+ ncol;

    return m;
  }

  /*
   Free the C-style matrix

    Input:
      m - pointer to the matrix
  */
  template <class T> void free_matrix(T **m) {
    delete [] m[0];
    delete [] m;
  }


  /*
    The name 'flt' stands for first larger than:
    First element in an sorted array in ascending order larger than target value.

    Input:
      t: comparison value
      a: array to be searched
      n: array length

    Return:
      index of the first element larger than target, or
      -1 if target is out of bounds.
      0  if the lower boundary is breached 0
  */
  template <class T>
  int flt(const T & t, const T *a, const int &n) {

    /* We only need to test the upper boundary; if the lower boundary is
     * breached if 0 is returned. The calling functions thus must test
     * against flt index < 1. */

    if (t > a[n-1]) return -1;

    if (t < a[0]) return 0;

    int low = 0, high = n, mid;

    while (low != high) {

      mid = (low + high) >> 1;

      if (a[mid] <= t)
        low = mid + 1;
      else
        high = mid;
    }

    return low;
  }


  /*
    Linear interpolation of points

      (x_i,y_i) i = 0, ..., n -1

    Input:
      t: target value
      n: number of points
      x: array of independent values sorted in ascending order
      y: array of independent values

    Return:
      linearly interpolated value of y at t
  */
  template <class T>
  T lin_interp(const T & t, const int & n, const T *x, const T *y){

    int i  = flt(t, x, n);

    if (i == 0 || i == -1) return std::numeric_limits<T>::quiet_NaN();

    return (y[i]*(x[i] - t) + y[i-1]*(t - x[i-1]))/(x[i] - x[i-1]);
  }

  /*
   Giving the principal solution -- upper branch of the solution
   of the Lambert equation

        W e^W = x

    for x in [-1/e, infty) and W > -1.

    The solution is called Lambert W function or ProductLog

   Ref:
    * http://mathworld.wolfram.com/LambertW-Function.html
    * Robert M. Corless, et.al, A Sequence of Series for The Lambert W Function

  */

  template <class T>
  T lambertW(const T & x){

    // checking limits
    if (x == T(0)) return  T(0);

    if (std::isinf(x) ) {
      if (x > 0)
        return std::numeric_limits<T>::infinity();
      else
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (x < -m_1_e || std::isnan(x)) return std::numeric_limits<T>::quiet_NaN();

    //
    // calculating approximation
    //

    T p, w;


    if (x < -0.25*m_1_e) {
      // around branching point
      p = std::sqrt(2.0*(m_e*x + 1.0));
      w = -1. + p*(1 + p*(-0.3333333333333333 + p*(0.1527777777777778 +
           p*(-0.07962962962962963 + p*(0.044502314814814814 + p*(-0.02598471487360376 +
           p*(0.01563563253233392 - 0.009616892024299432*p)))))));

      if (x + m_1_e < 1e-3) return w;

    } else if ( x < -0.5*m_1_e){
      // series around -0.75*m_1_e
      p = x + 3*m_1_e/4;
      w = -0.41986860097402295 + p*(2.6231325990041836 + p*(-9.370814155554825 +
          p*(53.573090925650874 + p*(-371.14484652831385 + p*(2852.957668395053 +
          p*(-23404.79831089446 + p*(200748.5156249781 - 1.7785330276273934e6*p)))))));

      if (std::abs(p) < 1e-3) return w;
    } else if (std::abs(x) <= 0.5*m_1_e) {
      w = x*(1 + x*(-1. + x*(1.5 + x*(-2.6666666666666665 + x*(5.208333333333333 +
          x*(-10.8 + (23.343055555555555 - 52.01269841269841*x)*x))))));

      if (std::abs(x) < 1e-3) return w;
    } else if (x <= 1.5*m_1_e) {
      // series around m_1_e
      p = x - m_1_e;
      w = 0.2784645427610738 + p*(0.5920736016838016 + p*(-0.31237407786966215 +
         p*(0.24090600442965682 + p*(-0.2178832755815021 + p*(0.21532401351646263 +
         p*(-0.22520037555300257 + p*(0.24500015392074573 - 0.27439507212836256*p)))))));

      if (std::abs(p) < 1e-3) return w;
   } else if (x <= 2.5*m_1_e) {
      // series around 2*m_1_e
      p = x - 2*m_1_e;
      w = 0.46305551336554884 + p*(0.4301666532987023 + p*(-0.1557603379726202 +
          p*(0.08139743653170319 + p*(-0.049609658385920324 + p*(0.032938686466159176 +
          p*(-0.02310194185417815 + p*(0.016833472598465127 - 0.012616316325209298*p)))))));

      if (std::abs(p) < 1e-3) return w;
    } else if (x <= 20*m_1_e) {
      // expanding W(exp(p+1)) around p=0
      p = std::log(x/m_e);
      w = 1. + p*(0.5 + p*(0.0625 + p*(-0.005208333333333333 + p*(-0.0003255208333333333 +
          p*(0.00021158854166666667 + p*(-0.00003187391493055556 +
          p*(-1.7680819072420636e-6 + 1.8520960732111855e-6*p)))))));

    } else { // asymptotic approximation
      T L1 = std::log(x),
        L2 = std::log(L1);
      w = L1 - L2 + L2/L1*(1 + (L2/2 - 1)/L1);
    }

    //
    // Halley’s method
    // x > 0: log(w) + w = log(x) := z
    // x < 0: log(q) - q = log(-x) := z, q = -w
    // Ref:
    //  *  https://en.wikipedia.org/wiki/Halley%27s_method

    const int max_iter = 20;
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    int iter = 0;

    T z = std::log(std::abs(x)), dw;

    if (x > 0) {
      do {
        p = std::log(w) - z;
        dw = (2*w*(1 + w)*(w + p))/((2 + w)*(1 + 2*w) + p);
        w -= dw;
      } while (std::abs(dw) < eps*std::abs(w) + min && ++iter < max_iter );
    } else {
      w = -w;
      do {
        p = std::log(w) - z;
        dw = (2*w*(-1 + w)*(w - p))/((-2 + w)*(-1 + 2*w) + p);
        w -= dw;
      } while (std::abs(dw) < eps*std::abs(w) + min && ++iter < max_iter );
      w = -w;
    }

    return w;
  }
} // namespace utils

