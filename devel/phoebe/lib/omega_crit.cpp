/*
  Creating a map of the critical potential configurations.

  Author: Martin Horvat, March 2016
   
  Compile: 
    g++ omega_crit.cpp -o omega_crit -O3 -Wall -std=c++11
  
  Memory leak test:
    valgrind --leak-check=full -show-leak-kinds=all -v ./omega_crit
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#include "gen_roche.h"
      
int main(){
  
  #if 1
  int ind[3];
  
  std::string conf;
  
  double omega[3], q, F; 
  
  std::ofstream f("omega_crit.dat");
  
  f.precision(16);
  
  for (int i = 1; i <= 400; ++i) {
    
    q  = 0.01*i;
      
    for (int j = 1; j <= 400; ++j) {
      

      F  = 0.005*j;
      
      gen_roche::critical_potential(omega, q, F); 
             
      utils::sort3ind(omega, ind);
      
      conf = std::to_string(ind[0]+1) + 
             std::to_string(ind[1]+1) + 
             std::to_string(ind[2]+1);
      
      f << q << '\t' << F << '\t' << conf << '\n';
      //  << omega[0] << '\t' << omega[1] << '\t' << omega[2] << '\n';
    }
    f << '\n';
  }
  #else
  
  double 
    omega[3], 
    q = 1, 
    F = 1,
    delta = 1; 
    
  gen_roche::critical_potential(omega, q, F, delta); 
  
  std::cout  << omega[0] << ' ' << omega[1] << ' ' << omega[2] << '\n';
  
  #endif
  
  return 0;
}
