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
#include "povray.h"

int main(){
  
  #if 0
  
  //
  // Sphere
  //
  
  int  max_triangles = 10000000;
    
  double
    R = 1,
    delta = 0.01;  
  
  Tmarching<double, Tsphere<double> > march(&R);
  #endif

  #if 0
  
  //
  // Some simple Roche lobe
  //
  
  int  max_triangles = 10000000;
    
  double 
    q = 0.5,
    F = 1.5,
    deltaR = 1,
    Omega0 = 4,
    x0 =-0.3020194679312359,
    params[5] = {q, F, deltaR, Omega0, x0},
    
    delta = 0.01;   
  
  Tmarching<double, Tgen_roche<double>> march(params);
  #endif
  
  
  #if 0
  
  //
  // Some simple Roche lobe: large Omega limit
  //
  
  int  max_triangles = 10000000;
    
  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 27092.1846036;
    
  double xrange[2];
  
  if (!gen_roche::lobe_x_points(xrange, 0, Omega0, q, F, deltaR, true)){
    std::cerr << "Determing lobe's boundaries failed\n";
    return EXIT_FAILURE;
  }
  std::cout.precision(16);
  std::cout << std::scientific;
  
  std::cout << xrange[0] << '\t' << xrange[1] << '\n';   
      
  double 
    delta = std::abs(xrange[0])/10,
    params[5] = {q, F, deltaR, Omega0, xrange[0]};   
    
  Tmarching<double, Tgen_roche<double>> march(params);
  #endif
  
  
  #if 1
  
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

  double xrange[2];
  
  if (!gen_roche::lobe_x_points(xrange, 2, Omega0, q, F, deltaR, true)){
    std::cerr << "Determing lobe's boundaries failed\n";
    return EXIT_FAILURE;
  }
  
  std::cout.precision(16);
  std::cout << std::scientific;
  
  std::cout << xrange[0] << '\t' << xrange[1] << '\n';
  
  double  params[5] = {q, F, deltaR, Omega0, xrange[0]};   
  
  Tmarching<double, Tgen_roche<double> > march(params);
  #endif
  

  #if 0
  
  //
  // Phoebe generic case: detached case
  //
  
  int  max_triangles = 10000000;

  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 10,

    delta = 0.01;
  
  double xrange[2];
  
  if (!gen_roche::lobe_x_points(xrange, 2, Omega0, q, F, deltaR, true)){
    std::cerr << "Determing lobe's boundaries failed\n";
    return EXIT_FAILURE;

  double params[5] = {q, F, deltaR, Omega0, xrange[0]};   

  std::cout.precision(16);

  Tmarching<double, Tgen_roche<double> > march(params);
  #endif

  std::vector <T3Dpoint<double> > V;
  std::vector <T3Dpoint<int>> Tr; 
  std::vector <T3Dpoint<double> >NatV;
    
  if (!march.triangulize(delta, max_triangles, V, NatV, Tr)){
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
  
  std::string color_body("Red");
  
  T3Dpoint<double> 
    camera_location(0,2,2),
    camera_look_at(0,0,0),
    light_source(100, 100, 100);
  
  double plane_z = -1;
  
  triangle_mesh_export_povray (
    color_body,
    camera_location, 
    camera_look_at, 
    light_source,
    plane_z,
    file, 
    V, NatV, Tr); 
  
  #endif
  
  
  //
  // Example 2
  //
  
  #if 1
  std::ofstream file("scene2.pov");
  
  // povray +R2 +A0.1 +J1.2 +Am2 +Q9 +H480 +W640 scene.pov
  
  file.precision(16);
  
  std::string color_body("Red");
  
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
    color_body,
    camera_location, 
    camera_look_at, 
    light_source,
    plane_z,
    file, 
    V1, NatV1, Tr1); 
  
  #endif
  
  
  return 0;
}
