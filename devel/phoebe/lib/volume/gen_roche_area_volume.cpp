/*
  Testing area and volume of Roche lobes
  
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
  
  double xrange[2];

  gen_roche::lobe_x_points(xrange, choice, Omega0, q, F, deltaR);
  
  
  std::cout << "x0=" << xrange[0] << " x1=" << xrange[1] << '\n';
  
  double av[2]; 
     
  gen_roche::area_volume_integration(av, 3, xrange, Omega0, q, F, deltaR);
  
  std::cout << "A=" << av[0] << " V=" << av[1] << '\n';
  
  
  return 0;
}
