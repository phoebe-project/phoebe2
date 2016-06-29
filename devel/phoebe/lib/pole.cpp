/*
  Testing the calculation of the poles
  
  Author: Martin Horvat, April 2016
  
  Compile: run make 
*/ 

#include <iostream>
#include <cmath>

#include "gen_roche.h"

int main(){
  std::cout.precision(16);
  std::cout <<std::scientific;
  
  #if 0
  double 
    q = 0.5,
    F = 0.5,
    deltaR = 1,
    Omega0 = 2.65; 
  
  
  double 
    zL = gen_roche::poleL(Omega0, q, F, deltaR),
    zR = gen_roche::poleR(Omega0, q, F, deltaR);
  
  std::cout << zL << '\t' << zR << '\n';
  #endif
  
  
  #if 1
  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 27092.1846036; 
  
  
  double 
    zL = gen_roche::poleL(Omega0, q, F, deltaR),
    zR = gen_roche::poleR(Omega0, q, F, deltaR);
  
  std::cout << zL << '\t' << zR << '\n';
  #endif
  
  
  return 0;
}
