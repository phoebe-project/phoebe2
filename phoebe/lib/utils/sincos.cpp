/*
  Testing idea of templated pseudo-assembler code with gcc and icpc.
  
  Author: Martin Horvat, July 2016
*/

#include <iostream>
#include <cmath>
#include <limits>

#include "sincos.h"

int main(){

  {
  typedef float real;
  
  real angle = 1, s, c;
  
  std::cout.precision(std::numeric_limits<real>::digits10);
  std::cout<< std::scientific;
  
  utils::sincos(angle, &s, &c);
  
  std::cout 
    << s << '\t' << c << '\n'
    << std::sin(angle) << '\t' << std::cos(angle) << '\n';
  }
  
  
  
  {
  typedef double real;
  
  real angle = 1, s, c;
  
  std::cout.precision(std::numeric_limits<real>::digits10);
  std::cout<< std::scientific;
  
  utils::sincos(angle, &s, &c);
  
  std::cout 
    << s << '\t' << c << '\n'
    << std::sin(angle) << '\t' << std::cos(angle) << '\n';
  }
  
  
  
  {
  typedef long double real;
  
  real angle = 1, s, c;
  
  std::cout.precision(std::numeric_limits<real>::digits10);
  std::cout<< std::scientific;
  
  utils::sincos(angle, &s, &c);
  
  std::cout 
    << s << '\t' << c << '\n'
    << std::sin(angle) << '\t' << std::cos(angle) << '\n';
  }
  
  
  std::cout 
  << "Wolfram Mathematica\n"
  << "sin(1)=0.8414709848078965066525023216302989996225630607983710657554\n"
  << "cos(1)=0.5403023058681397174009366074429766037323104206179222275411\n";
  
  return 0;
}
