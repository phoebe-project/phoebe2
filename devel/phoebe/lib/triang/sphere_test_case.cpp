/*
  Testing the Marching triangulization method on Sphere:
    area and volume of the mesh VS known quantities
    
  S = 4pi R^2
  V = 4pi/3 R^3 

  Author: Martin Horvat, June 2016
  
  Compile: 
    g++ sphere_area_volume.cpp -o sphere_area_volume -O3 -Wall -std=c++11
*/ 

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <set>
#include <list>

#include "triang_mesh.h"
#include "triang_marching.h"
#include "bodies.h"
#include "../gen_roche.h" 

int main(){
  
  
  //
  // Sphere
  //
  
  int 
    max_triangles = 10000000;
    
  double
    R = 1,
    delta = 3.2903174688109063e-02;
  
  std::cout.precision(16);
  std::cout << std::scientific;
  
  double r[3], g[3], av[2];

  Tmarching<double, Tsphere<double> > march(&R);
  
  march.init(r, g);
      
  std::vector <T3Dpoint<double> > V, NatV;
  std::vector <T3Dpoint<int>> Tr; 
    
  if (!march.triangulize(r, g, delta, max_triangles, V, NatV, Tr)){
    std::cerr << "There is too much triangles\n";
  }
 
  mesh_area_volume(V, NatV, Tr, av);
 
  double dR = 0;
  
  for (auto && v : V) dR = std::max(dR, v[0]*v[0]  + v[1]*v[1] + v[2]*v[2] - 1);
   
  std::cout
    << delta << '\t'
    << V.size() << '\t' 
    << Tr.size() << '\t'
    << av[0] << '\t'
    << av[1] << '\t'
    << dR << std::endl;


  std::ofstream f("sphere_triangles.dat");
  f.precision(16);
  f << std::scientific;
  for (auto && t : Tr) 
    for (int i = 0; i < 3; ++i) f << V[t[i]] << '\n';
  f.close();
  
  return 0;
}
