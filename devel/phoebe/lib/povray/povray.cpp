/*
  Testing the povray export

  mkdir -p ${HOME}/.povray/3.7
  touch ${HOME}/.povray/3.7/povray.conf

  Author: Martin Horvat, July 2016

*/ 

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <set>
#include <list>

#include "../gen_roche.h" 
#include "../mesh.h"
#include "povray.h"

int main(){
  
  int  max_triangles = 10000000;
  
  #if 0
  
  //
  // Some simple Roche lobe
  //
  
  int choice = 0;
  
  double 
    q = 0.5,
    F = 1.5,
    deltaR = 1,
    Omega0 = 4,
    params[4] = {q, F, deltaR, Omega0},
    
    delta = 0.01;   

  #endif
  
  
  #if 0
  
  //
  // Some simple Roche lobe: large Omega limit
  //

  int choice = 0;   

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
    
  #endif
  

  #if 0
  
  //
  // Phoebe generic case: detached case
  //
  
  int choice  = 0;

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
  std::vector <T3Dpoint<int>> Tr; 
  std::vector <T3Dpoint<double> >NatV;
    
  if (!march.triangulize(r, g, delta, max_triangles, V, NatV, Tr)){
    std::cerr << "There is too much triangles\n";
    return EXIT_FAILURE;
  }

  std::cout 
    << "Nr. of vertices =" << V.size()  << '\n'
    << "Nr. of triangles=" << Tr.size() << '\n';
  
  //
  // Example 1
  //
  
  #if 0
  std::ofstream file("scene.pov");
  
  // povray +R2 +A0.1 +J1.2 +Am2 +Q9 +H480 +W640 scene.pov
  
  file.precision(16);
  
  std::string body_color("Red");
  
  T3Dpoint<double> 
    camera_location(0,2,2),
    camera_look_at(0,0,0),
    light_source(100, 100, 100);
  
  double plane_z = -1;
  
  triangle_mesh_export_povray (
    file, 
    V, NatV, Tr,
    body_color,
    camera_location, 
    camera_look_at, 
    light_source,
    &plane_z); 
  
  #endif
  
  
  //
  // Example 2
  //
  
  #if 1
  std::ofstream file("scene2.pov");
  
  // povray +R2 +A0.1 +J1.2 +Am2 +Q9 +H480 +W640 scene.pov
  
  file.precision(16);
  
  std::string body_color("Red");
  
  T3Dpoint<double> 
    camera_location(0,2,2),
    camera_look_at(0,0,0),
    light_source(100, 100, 100);
  
  double plane_z = -1;
  
  int Nv = V.size();
  
  std::vector <T3Dpoint<double> > V1(V);
  std::vector <T3Dpoint<int>> Tr1(Tr); 
  std::vector <T3Dpoint<double> >NatV1(NatV);
  
  for (auto  && v : V)  {
    T3Dpoint<double> v1(v);
    v[1] -= 2;
    V1.push_back(v);
  }
  
  for (auto  && n : NatV)  NatV1.push_back(n);
  
  for (auto  && t : Tr) {
    T3Dpoint<int> t1(t);
    for (int i = 0; i < 3; ++i) t1[i] += Nv;
    Tr1.push_back(t1);
  }
  
  triangle_mesh_export_povray (
    file, 
    V1, NatV1, Tr1,
    body_color,
    camera_location, 
    camera_look_at, 
    light_source,
    &plane_z);
     
  
  #endif
  
  
  return 0;
}
