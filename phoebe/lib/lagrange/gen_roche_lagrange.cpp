/*
  Testing the Lagrange point calculation for generalized Roche/Kopal potential
  Author: Martin Horvat, March 2016
  
  Compile: g++ gen_roche_lagrange.cpp -o gen_roche_lagrange -O3 -Wall -DDEBUG -std=c++11
  
  Run: 
  
  time ./gen_roche_lagrange

  real	0m5.890s
  user	0m5.767s
  sys	0m0.116s

  This generates L1.dat, L2.dat and L3.dat.
*/

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

#include "gen_roche_lagrange.h"

int main(){
  
  #if 0 
  std::string s;
  
  double x, q, F;
  
  std::ofstream f;

  
  for (int l = 0; l < 3; ++l) {
    
    s = std::to_string(l+1);
    
    f.open("L" + s+".dat");
    
    f.precision(16);
    
    for (int i = 1; i <= 1000; ++i) {
    
      for (int j = 1; j <= 1000; ++j) { 
        
        q = 0.01*i;
        
        F = 0.01*j;
        
        switch (l) {
          case 0:
            x = gen_roche::lagrange_point_L1<double>(q, F);
            f 
            << q << '\t'
            << F << '\t' 
            << x << '\t'
            << gen_roche::lagrange_point_L1_x << '\t'
            << gen_roche::lagrange_point_L1_n << '\n';
            
            gen_roche::lagrange_point_L1_n = 0;
            break;
          
          case 1:
            x = gen_roche::lagrange_point_L2<double>(q, F);
            f 
            << q << '\t'
            << F << '\t' 
            << x << '\t'
            << gen_roche::lagrange_point_L2_x << '\t'
            << gen_roche::lagrange_point_L2_n << '\n';
        
            gen_roche::lagrange_point_L2_n = 0;
            break;
       
          case 2:
            x = gen_roche::lagrange_point_L1<double>(q, F);
            f 
            << q << '\t'
            << F << '\t' 
            << x << '\t'
            << gen_roche::lagrange_point_L3_x << '\t'
            << gen_roche::lagrange_point_L3_n << '\n';
            
            gen_roche::lagrange_point_L3_n = 0;
            break;
        }
      }
      f << '\n';
    }
    
    f.close();
  }
  #endif
  
  #if 0
  
  double 
    q = 3.00348934885e-06, 
    F = 365.25;
  
  std::cout.precision(16);
  std::cout << std::scientific;
  
  std::cout  
    << gen_roche::lagrange_point_L1<double>(q, F) << '\t'
    << gen_roche::lagrange_point_L2<double>(q, F) << '\t'
    << gen_roche::lagrange_point_L3<double>(q, F) << '\n';
   
  #endif
  
  
  
  #if 1
  
  double 
    q = 1e-6, 
    F = 1e-12;
  
  std::cout.precision(16);
  std::cout << std::scientific;
  
  std::cout  
    << gen_roche::lagrange_point_L1<double>(q, F) << '\t'
    << gen_roche::lagrange_point_L2<double>(q, F) << '\t'
    << gen_roche::lagrange_point_L3<double>(q, F) << '\n';
   
  #endif
  return 0;
}
