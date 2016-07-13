/*
  Testing the points on x-axis of the generalized Roche lobes, i.e. implicitely defined surface
  
    Omega(x,y,z) = Omega0
  
  where Omega is the generalized Kopal potential.
  
  Author: Martin Horvat, April 2016
   
  Compile:
    g++ points_on_x_axis.cpp -o points_on_x_axis -O3 -Wall -std=c++11
*/

#include <iostream>
#include <cmath>
#include <list>

#include "gen_roche.h"

int main(){

  
  #if 0
  // Left lobe
  // Expected result: failed
  int choice = 0;
  double 
    Omega0 = 3, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
   
    
  #if 0
  // Left lobe
  // Expected result: -1.1140295269542350e-01	1.1143792464110042e-01
  int choice = 0;
  double 
    Omega0 = 10, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
  
  
  #if 0
  // Right lobe
  // Expected result: 8.8856207535889964e-01	1.1114029526954234e+00
  int choice = 1;
  double 
    Omega0 = 10, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
  
  
  #if 0
  // Overcontact
  // Expected result: -4.6961743405803708e-01	1.4696174340580370e+00
  int choice = 2;  
  double 
    Omega0 = 3.5, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
  
  #if 0
  // Overcontact
  // Expected result: -4.6961743405803708e-01	1.4696174340580370e+00
  int choice = 2;  
  double 
    Omega0 = 3.7, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
  
  #if 0
  // Left lobe, large Omega, small q
  // Expected result: 
  int choice = 0;  
  double 
    Omega0 = 27092.1846036, 
    q = 3.00348934885e-06,
    F = 365.25,
    delta = 1;
  #endif
  
  #if 0
  // Left lobe, large Omega, small q
  // Expected result: 
  int choice = 0;  
  double 
    Omega0 = 30000, 
    q = 3.00348934885e-06,
    F = 365.25,
    delta = 1;
  #endif
  
  
  #if 0
  // Left lobe, large Omega, small q
  // Expected result: 
  int choice = 0;  
  double 
    Omega0 = 30000, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
  
  #if 0
  // Right lobe, large Omega, small q
  // Expected result: 
  int choice = 1;  
  double 
    Omega0 = 30000, 
    q = 1,
    F = 1,
    delta = 1;
  #endif
  
  #if 1
  // Right lobe, large Omega, small q
  // Expected result: 
  int choice = 0;  
  double 
    Omega0 = 8.992277876713667,
    q = 1,
    F = 1,
    delta = 1.0000000000000002;
  #endif
  
  
  double xrange[2];
  
  if (!gen_roche::lobe_x_points(xrange, choice, Omega0, q, F, delta, true)){
    std::cerr << "Problem obtaining boundaries\n";
    return EXIT_FAILURE;
  }

  std::cout.precision(16);
  std::cout << std::scientific;
  
  std::cout << xrange[0] << '\t' << xrange[1] << '\n';

  return 0;
}
