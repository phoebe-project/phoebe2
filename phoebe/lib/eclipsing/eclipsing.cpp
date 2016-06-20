/*
  Testing eclipsing algorithm -- classification of visibility 
  (visibile, partial, hidden)
  
  Author: Martin Horvat, May 2016
*/ 

#include <iostream>
#include <cmath>
#include <fstream>

#include "eclipsing.h"
#include "../gen_roche.h"
#include "../triang/triang_marching.h"
#include "../triang/bodies.h"


#include <ctime>

int main(){
 
 
  clock_t start, end;
     
  //
  // Overcontact case
  //
  
  int  max_triangles = 10000000;
    
  double 
    q = 0.5,
    F = 0.5,
    deltaR = 1,
    Omega0 = 2.65,
    
    delta = 0.01;

  std::vector<double> x_points;
    
  gen_roche::points_on_x_axis(x_points, Omega0, q, F, deltaR);
    
  double  
    x0 = x_points.front(),
    params[5] = {q, F, deltaR, Omega0, x0};   
  
  //
  // make triangulation of the surface
  //
  
  std::cout << "Surface triagulation:";
  
  start = clock();
  
  Tmarching<double, Tgen_roche<double> > march(params);
  
  std::vector<T3Dpoint<double> > V, NatV;
  std::vector<T3Dpoint<int>> T; 
  
  if (!march.triangulize(delta, max_triangles, V, NatV, T)){
    std::cerr << "There is too much triangles\n";
  }
  
  end = clock();
  
  std::cout << " time=" << end - start << " um\n";
   std::cout << "V.size=" << V.size() << " T.size=" << T.size() << '\n';


  //
  // Calc triangle properties
  //
  
  std::cout << "Triangle properties:";
  
  start = clock();
    
  std::vector<T3Dpoint<double>> N;
    
  mesh_attributes(V, NatV, T, (std::vector<double>*)0, &N);
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  std::cout << "N.size=" << V.size() << '\n';
   
  //
  // Eclipsing
  //
  
  std::cout << "Eclipsing:";
  
  start = clock();
    
  double 
    theta = 20./180*M_PI, 
    view[3] = {std::cos(theta), 0, std::sin(theta)};
    
  std::vector<Tvisibility> M;
      
  triangle_mesh_rough_visibility(view, V, T, N, M);
  //triangle_mesh_rough_visibility_elegant(view, V, T, N, M);
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  
  //
  // Storing results
  //
  
  std::cout << "Storing results:";
  
  start = clock();
  
  //
  // Save view
  //
  std::ofstream fr("view.dat");
  fr.precision(16);
  for (int i = 0; i < 3; ++i) fr << view[i] << '\n';
  fr.close();
  
  
  //
  // Save triangles
  //
  fr.open("triangles.dat");
  for (auto && t: T)
    for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';  
  fr.close();
  
  //
  //  map
  //
  fr.open("mask.dat");
  for (auto && t: M) fr << t << '\n';
  fr.close();
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  
  return 0;
}
