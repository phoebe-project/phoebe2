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
    max_triangles = 10000000,
    steps = 100;
    
  double
    R = 1,
    delta0 = 0.25,
    delta1 = 0.01,
    fac = exp(log(delta1/delta0)/steps),
    
    delta = delta0;
  
  std::cout.precision(16);
  std::cout << std::scientific;
  
  double r[3], g[3];
    
  for (int i = 0; i < steps; ++i){
    
    Tmarching<double, Tsphere<double> > march(&R);
    
    march.init(r, g);
        
    std::vector <T3Dpoint<double> > V;
    std::vector <T3Dpoint<int>> T; 
    std::vector <T3Dpoint<double> >NatV;
      
    if (!march.triangulize(r, g, delta, max_triangles, V, NatV, T)){
      std::cerr << "There is too much triangles\n";
    }

    double av[2];
    
    mesh_area_volume(V, NatV, T, av);
   
    double dR = 0;
    
    for (auto && v : V) dR = std::max(dR, v[0]*v[0]  + v[1]*v[1] + v[2]*v[2] - 1);
     
    std::cout
      << delta << '\t'
      << V.size() << '\t' 
      << T.size() << '\t'
      << av[0] << '\t'
      << av[1] << '\t'
      << dR << std::endl;
  
    delta *= fac;
  }
  
  return 0;
}
