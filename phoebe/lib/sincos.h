#pragma once

/*
  Defining sincos function for different types in utils namespace:
    float
    double
    long double


  Author: Martin Horvat, July 2016, April 2022
*/

#include <cmath>

namespace utils {

  template<class T>
  inline void sincos(const T &angle, T *s, T *c){
    #if defined(TARGET_HAS_SINCOS)
    asm volatile("fsincos" : "=t" (*c), "=u" (*s) : "0" (angle));
    #else
    *s = std::sin(angle);
    *c = std::cos(angle);
    #endif
  }

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

