/*
  Testing idea of templated pseudo-assembler code with gcc and icpc.
  
  Author: Martin Horvat, July 2016
*/

#include <iostream>
#include <cmath>
#include <limits>

template <class T>
void my_sincos(const T &angle, T *s, T *c){
  asm volatile("fsincos" : "=t" (*c), "=u" (*s) : "0" (angle) : "st(7)");
}

int main(){

  {
  typedef float real;
  
  real angle = 1, s, c;
  
  std::cout.precision(std::numeric_limits<real>::digits10);
  std::cout<< std::scientific;
  
  my_sincos(angle, &s, &c);
  
  std::cout 
    << s << '\t' << c << '\n'
    << std::sin(angle) << '\t' << std::cos(angle) << '\n';
  }
  
  
  
  {
  typedef double real;
  
  real angle = 1, s, c;
  
  std::cout.precision(std::numeric_limits<real>::digits10);
  std::cout<< std::scientific;
  
  my_sincos(angle, &s, &c);
  
  std::cout 
    << s << '\t' << c << '\n'
    << std::sin(angle) << '\t' << std::cos(angle) << '\n';
  }
  
  
  
  {
  typedef long double real;
  
  real angle = 1, s, c;
  
  std::cout.precision(std::numeric_limits<real>::digits10);
  std::cout<< std::scientific;
  
  my_sincos(angle, &s, &c);
  
  std::cout 
    << s << '\t' << c << '\n'
    << std::sin(angle) << '\t' << std::cos(angle) << '\n';
  }
    
  return 0;
}
