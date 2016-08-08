/*
  Testing volume of Roche lobes and its derivatives
  
  Author: Martin Horvat, June 2016
*/

#include <iostream>
#include <cmath>

#include "gen_roche_area_volume.h"
#include "../gen_roche.h" 

int main(){
  
  #if 1
  //
  // overcontact case
  //
  int choice = 2;
  double 
    q = 0.5,
    F = 0.5,
    deltaR = 1,
    Omega0 = 2.65;
  #endif
  
  
  #if 0
  //
  // Phoebe generic case: detached case
  //
  int choice = 0;
  double
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 10;
  #endif
   
  
  std::cout.precision(16);
  
  std::cout << std::scientific;
  
  //
  // Direct calculation of derivative
  //   
  
  double xrange[2];

  gen_roche::lobe_x_points(xrange, choice, Omega0, q, F, deltaR);
   
  std::cout << "x0=" << xrange[0] << " x1=" << xrange[1] << '\n';
      
  double v[2]; 
  
  gen_roche::volume(v, 3, xrange, Omega0, q, F, deltaR);
  
  std::cout << "V=" << v[0] << "\tdV/dOmega=" << v[1] << '\n';
  
  //
  // Numerics calculation of derivative
  //
  
  double u[2], dOmega = 0.001, Omega1 = Omega0 + dOmega;
  
  gen_roche::lobe_x_points(xrange, choice, Omega1, q, F, deltaR);
  
  std::cout << "x0=" << xrange[0] << " x1=" << xrange[1] << '\n';
    
  gen_roche::volume(u, 3, xrange, Omega1, q, F, deltaR);
  
  std::cout << "V=" << u[0] << "\tdV/dOmega=" << u[1] << '\n';
  
  std::cout << "numerical:\t\t\tdV/dOmega=" << (u[0] - v[0])/dOmega << '\n';
  
  return 0;
}
