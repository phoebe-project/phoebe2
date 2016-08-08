/*
  Testing the Marching triangulization method. 

  Author: Martin Horvat, April 2016
  
  Compile: 
    g++ triang_marching.cpp -o triang_marching -O3 -Wall -std=c++11
*/ 

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <set>
#include <list>

#include "triang_marching.h"
#include "bodies.h"

#include "../gen_roche.h" 

int main(){
  

  int  max_triangles = 10000000;
    
  #if 0
  
  //
  // Some simple Roche lobe: large Omega limit
  //
  
  int  choice = 0;
  
  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 27092.1846036,
    
    delta = 0.01;
    
  #endif
  
  
  #if 1
  
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

  // delta = 0.1
  // Nr. of vertices =524
  // Nr. of triangles=1044
  // real	0m0.002s (no ouput)
  
  // delta = 0.01
  // Nr. of vertices =48496
  // Nr. of triangles=96988
  // real	0m0.060s (no ouput)

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
    Omega0 = 10,

    delta = 0.01;
  #endif


  std::cout.precision(16);
  std::cout << std::scientific;

  double params[4] = {q, F, deltaR, Omega0};   

  Tmarching<double, Tgen_roche<double> > march(params);

  double r[3], g[3];
   
  if (!gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, deltaR)) {
    std::cerr << "Don't fiding the starting point\n";
    return EXIT_FAILURE;
  }
  
  std::vector <T3Dpoint<double> > V;
  std::vector <T3Dpoint<int>> T; 
  std::vector <T3Dpoint<double> >NatV;
    
  if (!march.triangulize(r, g, delta, max_triangles, V, NatV, T)){
    std::cerr << "There is too much triangles\n";
    return EXIT_FAILURE;
  }

  std::cout 
    << "Nr. of vertices =" << V.size()  << '\n'
    << "Nr. of triangles=" << T.size() << '\n';
  
  //
  // Area and volume
  //
  double av[2];
  
  mesh_area_volume(V, NatV, T, av);
  std::cout << "Area=" << av[0] << "\nVolume=" << av[1] << '\n';
 
  #if 0
  //
  // Storing triagulation results
  //
  {
    
    std::ofstream fr("vertices.dat");
    fr.precision(16);
    fr << std::scientific;
    for (auto && v: V) fr << v << '\n';
    fr.close();
    
    
    fr.open("triangles1.dat");
    for (auto && t: T) fr << t << '\n';
    fr.close();
    
    fr.open("triangles2.dat");
    for (auto && t: T) {
      for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';
      fr << '\n';
    }
    fr.close();
  }

  //
  // Storing all indices in triangles
  //
  {
    std::set<int> ti;
    
    for (auto && t: T) for (int i = 0; i < 3; ++i) ti.insert(t.data[i]);
    
    std::ofstream fti("triangle_indices.dat");
    fti.precision(16);
    for (auto && t : ti) fti << t << '\n';
  }
  
  //
  // Triangle properties
  //
  {
    int choice = 0;
    double area, volume;
    std::vector <T3Dpoint<double>> C, N;
    std::vector <double> A;
    
    mesh_attributes(V, NatV, T, &A, &N, &area, &volume, choice);
  
    std::ofstream fa("triangle_areas.dat");
    fa.precision(16);
    for (auto && t : A) fa << t << '\n';
    fa.close();
    
    fa.open("triangle_normals.dat");
    fa.precision(16);
    for (auto && t : N) fa << t << '\n';
    fa.close();    
  }
  
  //
  // Central points
  //
  {  
    
    std::vector <T3Dpoint<double>> C, NatC;
  
    march.central_points(V, T, &C, &NatC);
   
    std::ofstream fc("triangle_central.dat");
    fc.precision(16);
    for (auto && t : C) fc << t << '\n';
    fc.close();
    
    
    fc.open("triangle_central_normals.dat");
    for (auto && t : C) fc << t << '\n';
    fc.close();
  }
  
  #endif 
  
  
  return 0;
}
