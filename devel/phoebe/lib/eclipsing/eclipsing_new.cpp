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

#include "eclipsing.h"
#include "../gen_roche.h"
#include "../mesh.h"

int main(){
 
 
  clock_t start, end;
  int  max_triangles = 10000000;
  
  #if 0   
  //
  // Overcontact case
  //
  int choice = 2;
  
  double 
    q = 0.5,
    F = 0.5,
    deltaR = 1,
    Omega0 = 2.65,
    
    delta = 0.01;
  #endif
  
  #if 1
  //
  // Generic case
  //
  int choice = 0;
  
  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 10,
    
    delta = 0.01;
  #endif

  double r[3], g[3];
   
  if (!gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, deltaR)) {
    std::cerr << "Don't fiding the starting point\n";
    return EXIT_FAILURE;
  }
  
  //
  // Make triangulation of the surface
  //
  
  std::cout << "Surface triagulation:";
  
  start = clock();

  double params[4] = {q, F, deltaR, Omega0};   
  
  Tmarching<double, Tgen_roche<double> > march(params);

  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr; 
  
  
  if (!march.triangulize(r, g, delta, max_triangles, V, NatV, Tr)){
    std::cerr << "There is too much triangles\n";
      return EXIT_FAILURE;
  }
  
  end = clock();
  
  std::cout << " time=" << end - start << " um\n";
  std::cout << "V.size=" << V.size() << " T.size=" << Tr.size() << '\n';
  

  //
  // Calc triangle properties
  //
  
  std::cout << "Triangle properties:";
  
  start = clock();
    
  std::vector<T3Dpoint<double>> N;
    
  mesh_attributes(V, NatV, Tr, (std::vector<double>*)0, &N);
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  std::cout << "N.size=" << N.size() << '\n';
  
  //
  //  Testing the new eclipsing algorithm
  //
  
  std::cout << "New eclipsing:";
  
  start = clock();  
  
  std::vector<double> M;
  std::vector<T3Dpoint<double>> W;
  std::vector<std::vector<int>> H;
  
  double 
    theta = 20./180*M_PI, 
    view[3] = {std::cos(theta), 0, std::sin(theta)};
    
  triangle_mesh_visibility(view, V, Tr, N, &M, &W, &H);

  end = clock();
  
  std::cout << " time= " << end - start << " um\n";


  //
  // Storing results
  //
  
  std::cout << "Storing results:";
  
  start = clock();
  
  //
  // Viewer direction
  //
  std::ofstream fr("view_new.dat");
  fr.precision(16);
  for (int i = 0; i < 3; ++i) fr << view[i] << '\n';  
  fr.close();
  
  //
  // Save triangles
  //
  fr.open("triangles_new.dat");
  for (auto && t: Tr)
    for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';
  fr.close();
  
  //
  // Saving mask
  //
  fr.open("mask_new.dat");
  for (auto && m : M) fr << m << '\n';
  fr.close();
 
 
  //
  // Saving weights
  //
  fr.open("mask_weights.dat");
  for (auto && w : W) fr << w << '\n';
  fr.close();
  
  //
  // Saving horizon
  //
  fr.open("horizon.dat");
  for (auto && h : H) {
    fr << h.size() << '\n';
    for (auto && index : h) fr << V[index] << '\n';
  }
  fr.close();
  

  end = clock();
  std::cout << " time= " << end - start << " um\n";



  
  return 0;
}
