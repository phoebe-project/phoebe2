#pragma once

/*
  Defining sincos function for different types in utils namespace:
    float
    double
    long double

  Double is enabled by pseudo-asm code and TESTED on AMD64 system!


  Author: Martin Horvat, July 2016

  Refs:
  * https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
  * http://www.tptp.cc/mirrors/siyobik.info/instruction/FSINCOS.html
  * x86 Assembly Language Reference Manual - Oracle
    https://docs.oracle.com/cd/E19641-01/802-1948/802-1948.pdf
  * http://stackoverflow.com/questions/11165379/loading-double-to-fpu-with-gcc-inline-assembler
  * http://www.willus.com/mingw/x87inline.h
*/

#include <cmath>

// TODO: How to test is architecture has sincos as part of assembly language
// Tested using g++ 5.4 and Intel icpc (ICC) 16.0.2 20160204 on
// Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz

#if defined(__GNUC__) || defined(__clang__)
#define TARGET_HAS_SINCOS 1
#else
#define TARGET_HAS_SINCOS 0
#endif

namespace utils {

  #if TARGET_HAS_SINCOS
  template <class T>
  inline void sincos(const T &angle, T *s, T *c){
    // works with gcc
    //asm volatile("fsincos" : "=t" (*c), "=u" (*s) : "0" (angle) : "st(7)");

    // works with gcc and clang
    asm volatile("fsincos" : "=t" (*c), "=u" (*s) : "0" (angle));
  }
  #else
  template<class T>
  inline void sincos(const T &angle, T *s, T *c){
    *s = std::sin(angle);
    *c = std::cos(angle);
  }
  #endif

  /*
    Calculate array of scaled sinus and cosinus

    Input:
      n >= 0 - number of angles
      f - elementary angle
      scale - prefactor

    Output:
      sa[n+1] = {0, sin(f), sin(2*f), ..., sin(n*f) }
      ca[n+1] = {1, cos(f), cos(2*f), ..., cos(n*f) }
  */
  template <class T>
  void sincos_array(const int & n, const T &f, T *sa, T *ca, const T & scale = 1){

    sa[0] = 0;
    ca[0] = scale;

    if (n == 0) return;

    #if 1

    T s, c;

    utils::sincos(f, &s, &c);

    sa[1] = s*scale;
    ca[1] = c*scale;

    for (int i = 1; i < n; ++i) {
      ca[i+1] = ca[i]*c - sa[i]*s;
      sa[i+1] = ca[i]*s + sa[i]*c;
    }
    #else // slower version, but preciser

    for (int i = 1; i <= n; ++i) {
      utils::sincos(i*f, sa + i, ca + i);
      ca[i] *= scale;
      sa[i] *= scale;
    }
    #endif
  }

}

