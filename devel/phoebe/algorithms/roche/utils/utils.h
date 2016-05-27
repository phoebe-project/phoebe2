#if !defined(__utils_h)
#define __utils_h

/*
  Commonly used routines.
  
  Author: Martin Horvat, April 2016
*/ 

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace utils {
  
  // 2 pi
  const double M_2PI = 6.2831853071795864769252867665590083999;
  
  /*
    Returning the square of the argument.
  */ 
  template <class T> T sqr(const T & x ) {
    return x*x;
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
   void polish(const int & n, T *a, std::vector<T> & roots){
      
      const int iter_max = 10;
      const T eps = 4*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      int it;
      
      T dx, f, df;
      
      for (auto && x : roots) {
        
        it = 0;  
        
        do {
          
          // Horner algorithm to compute value and derivative
          f = a[n], df = 0;
          for (int i = n - 1; i >= 0; --i) { 
            df = f + x*df;
            f  = a[i] + x*f;
          }
          
          // Newton-Raphson step
          x -= (dx = f/df); 
          
        } while (f != 0 && std::abs(dx) > eps*std::abs(x) + min && (++it) < iter_max); 
        
        if (it == iter_max) std::cerr << "Warning: Root polishing did not succeed\n";
        
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
  void solve_quadratic(T a[3], std::vector<T> &roots)
  {
    
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
  void solve_cubic(T a[4], std::vector<T> & roots)
  {
    
    roots.clear();
    
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
      
      if (D <= 0){ // 3 real roots, of 1 real roots if (p=q=0)
        
        if (p == 0 && q == 0) 
          roots.push_back(-b/3);
        else {
          phi = std::acos(3*q/(A*p));
        
          for (int i = 2; i >= 0; --i)
            roots.push_back(A*std::cos((phi - M_2PI*i)/3) - b/3);
        }
      } else {
        
        // D > 0, only one real root
        if (p < 0){
       
          phi = acosh(-3*std::abs(q)/(A*p));
          roots.push_back((q > 0 ? -A : A)*std::cosh(phi/3) - b/3);
       
        } else if (p == 0) {
       
          roots.push_back( (q > 0 ? -1 : 1)*std::pow(std::abs(q), 1./3) - b/3);
          
        } else {  // p > 0
          phi = asinh(3*q/(A*p));
          roots.push_back(-A*std::sinh(phi/3) - b/3);
        }
      }
      
    } else { 
      //
      // Working with a quadatic equation 
      //  a[2] x^2 + a[1] x + a[0] = 0
      //
      solve_quadratic(a, roots);
    }
  }
  
  /*
    Real roots of the quartic equation
      
      a[4] x^4 + a[3] x^3 + a[2] x^2 + a[1] x + a[0] = 0 
    
      Input:
        a[5] -- cofficients of the polynomial
      
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
         
      T  p = c - 3*b*b/8,
         q = b*(b*b/8 - c/2) + d,
         r = b*(b*(-3*b*b/256 + c/16) - d/4) + e;
        
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
        
        // using the first 
        T t  = 2*roots1.front() + p;
        
        if (t >= 0){
          
          T st = std::sqrt(t)/2, b_ = b/4, t1;
          
          for (int s1 = -1; s1 <= 1; s1 += 2){
            
            t1 = -(2*p + t + s1*q/st);

            if (t1 >= 0) {
              t1 = std::sqrt(t1)/2;
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
} // namespace utils

#endif  // #if !defined(__utils_h)
