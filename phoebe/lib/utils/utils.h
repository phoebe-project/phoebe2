#if !defined(__utils_h)
#define __utils_h

/*
  Commonly used routines.
  
  Author: Martin Horvat, April 2016
*/ 

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>

namespace utils {
  
  const double M_2PI = 6.2831853071795864769252867665590083999;  // 2 pi
  const double M_4PI = 12.5663706143591729538505735331180167998; // 4 pi
  
  
  /*
    Calculate the max of 3D vector.
    
    Input:
      x, y, z
    
    Return:
      max(x,y,z}
  */
  
  template <class T>  T max3(const T & x, const  T & y, const T & z){
    
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
  template <class T> T dot3D(T x[3], T y[3]) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }
  
  // z = x cross y  
  template <class T> void cross3D(T x[3], T y[3], T z[3]) {
    z[0] = x[1]*y[2] - x[2]*y[1];
    z[1] = x[2]*y[0] - x[0]*y[2];
    z[2] = x[0]*y[1] - x[1]*y[0];
  }
  
  // solve for x: A x = b
  template <class T> bool inverse2D(T A[2][2], T b[2], T x[2]){
    T det = A[0][0]*A[1][1] - A[1][0]*A[0][1];
    
    if (det == 0) return false;
    
    x[0] = (A[1][1]*b[0] - A[0][1]*b[1])/det;
    x[1] = (A[0][0]*b[1] - A[1][0]*b[0])/det;
    return true;
  }
  
  /*
    Calculate L2 norm of 3D vector
    
    Input:
      x, y, z
       
    Return:
      std::sqrt(x*x + y*y + z*z)
  */ 
  template <class T>
  T hypot3 (const T & x, const T& y, const T& z){
    
    T a[3] = {std::abs(x), std::abs(y), std::abs(z)}, t;
    
    if (a[0] < a[1]) { t = a[0]; a[0] = a[1]; a[1] = t;}
    if (a[0] < a[2]) { t = a[0]; a[0] = a[2]; a[2] = t;}
      
    a[1] /= a[0]; 
    a[2] /= a[0];
        
    t = a[1]*a[1] + a[2]*a[2];
    
    return a[0]*std::sqrt(1 + t);
  }
  
  
  /*
    Calculate L2 norm of 3D vector
    
    Input:
      x[3]
       
    Return:
      std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
  */ 
  template <class T>
  T hypot3 (T x[3]){
    
    T a[3] = {std::abs(x[0]), std::abs(x[1]), std::abs(x[2])}, t;
    
    if (a[0] < a[1]) { t = a[0]; a[0] = a[1]; a[1] = t;}
    if (a[0] < a[2]) { t = a[0]; a[0] = a[2]; a[2] = t;}
      
    a[1] /= a[0]; 
    a[2] /= a[0];
        
    t = a[1]*a[1] + a[2]*a[2];
    
    return a[0]*std::sqrt(1 + t);
  }
  
  
  
  /*
    Returning the square of the argument.
  */ 
  template <class T> T sqr(const T & x ) {
    return x*x;
  }
  
  
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
   void polish(const int & n, T *a, std::vector<T> & roots){
      
      const int iter_max = 10;
      const T eps = 10*std::numeric_limits<T>::epsilon();
      const T min = 10*std::numeric_limits<T>::min();
      
      int it;
      
      T dx;
      
      for (auto && x : roots) {
        
        it = 0;  
        
        do {
          #if 0
          // Horner algorithm to compute value and derivative
          // http://www.physics.utah.edu/~detar/lessons/c++/array/node4.html
          T f = a[n], df = 0;
          for (int i = n - 1; i >= 0; --i) { 
            df = f + x*df;
            f  = a[i] + x*f;
          }
          
          // Newton-Raphson step
          x -= (dx = f/df); 
          #else
          // Horner algorithm to compute value, derivative and second derivative
          // http://www.ece.rice.edu/dsp/software/FVHDP/horner2.pdf
          T f = a[n], df = 0, d2f = 0;
          for (int i = n - 1; i >= 0; --i) { 
            d2f = df + x*d2f;
            df = f + x*df;
            f  = a[i] + x*f;
          }
          d2f *= 2;
          
          // Newton-Raphson step for multiple roots
          // https://www.math.uwaterloo.ca/~wgilbert/Research/GilbertNewtonMultiple.pdf
          x -= (dx = f*df/(df*df - f*d2f)); 
          #endif
          /*
          std::cout.precision(16);
          std::cout << std::scientific;
          std::cout << x << '\t' << dx << '\t' << f  << '\n';
          */
            
        } while (std::abs(dx) > eps*std::abs(x) + min && (++it) < iter_max); 
        
        //std::cout << "-----\n";
        
        if (it == iter_max) 
          std::cerr << "Warning: Root polishing did not succeed\n";
        
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
      
      if (D <= 0 || std::abs(D) < eps){ // 3 real roots, of 1 real roots if (p=q=0)
        
        if (p == 0 && q == 0) 
          roots.push_back(-b/3);
        else { 
          
          T r = 3*q/(A*p);
          
          phi = (std::abs(r) > 1 ? 0 : std::acos(r));
        
        
          for (int i = 2; i >= 0; --i)
            roots.push_back(A*std::cos((phi - M_2PI*i)/3) - b/3);
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
            
        #if 0
        // using the first 
        T t = 2*roots1.front() + p;
        #else
        // using the one which is positive
        T t = -1;
        for (auto && r: roots1) if ((t = 2*r + p) >=0) break;
        #endif
         
        if (t >= 0){
          
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
 
  
  
} // namespace utils

#endif  // #if !defined(__utils_h)
