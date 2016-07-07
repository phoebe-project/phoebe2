/*
  Testing new algorithm for eclipsing.
  
  Profiling:

  g++ -O3 -Wall -std=c++11 eclipsing_new.cpp -o eclipsing_new -pg 
  
  ./eclipsing_new
   
  gprof  eclipsing_new gmon.out > analysis.txt
  
  Reading analysis.txt
   
  Author: Martin Horvat, May 2016
*/ 

#include <iostream>
#include <cmath>
#include <fstream>
#include <limits>
#include <ctime>

#include "../gen_roche.h"

int main(){
  
  //
  // Reading data
  //
  
  std::vector<T3Dpoint<int>> Tr; 
 
  {
    std::ifstream f("T.dat");
   
    int indices[3];
    
    double data[3];
    
    while (f >> data[0] >> data[1] >> data[2]) {
      for (int i = 0; i < 3; ++i) indices[i] = data[i];
      Tr.emplace_back(indices);
    }
  }
  
  std::vector<T3Dpoint<double>> V; 
 
  {
    std::ifstream f("V.dat");
   
    double data[3];
    
    while (f >> data[0] >> data[1] >> data[2]) V.emplace_back(data);
  }
  
  
  std::vector<T3Dpoint<double>> NatT; 
 
  {
    std::ifstream f("TN.dat");
   
    double data[3];
    
    while (f >> data[0] >> data[1] >> data[2]) NatT.emplace_back(data);
  }
   
  std::cout 
    << "Tr.size()=" << Tr.size() 
    << "\tV.size()=" << V.size() 
    << "\tN.size()=" << NatT.size() << '\n';
  

  
  //std::vector<T3Dpoint<double>> W; 
      
  double 
    theta = 20./180*M_PI, 
    view[3] = {std::cos(theta), 0, std::sin(theta)};
  
  #if 1
  std::vector<double> M;    
  triangle_mesh_visibility(view, V, Tr, NatT, &M);
  #else
  std::vector<Tvisibility> M;  
  triangle_mesh_rough_visibility(view, V, Tr, NatT, M);
  #endif
  
  //
  // Saving mask
  //
  
  {
    std::ofstream fr("mask.dat");
    fr.precision(16);
    fr << std::scientific;
    for (auto && m : M) fr << m  << '\n';
  }
 
 /*
  //
  // Saving weights
  //
  {
    std::ofstream fr("weights.dat");
    for (auto && w : W) fr << w << '\n';
  }
*/
  return 0;
}
