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
  double 
    Omega0 = 4, 
    q = 0.7,
    F = 0.5,
    delta = 3;
  
  #endif
  
  
  #if 0
  // Expected result: no x points after trim
  double 
    Omega0 = 3, 
    q = 1,
    F = 1,
    delta = 1;
  
  #endif
  
  
  
  #if 1
  // Expected result: overcontact, 2 x points
  double 
    Omega0 = 3.5, 
    q = 1,
    F = 1,
    delta = 1;
  
  #endif
    
  std::vector<double> points;
  
  gen_roche::points_on_x_axis(points, Omega0, q, F, delta);

  std::cout.precision(16);
  
  std::cout << "Values on x-axis\n";
  
  for (auto && v: points) std::cout << v << '\n';

  return 0;
}
