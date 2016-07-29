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

  // delta = 0.1
  // Nr. of vertices =524
  // Nr. of triangles=1044
  // real	0m0.002s (no ouput)
  
  // delta = 0.01
  // Nr. of vertices =48496
  // Nr. of triangles=96988
  // real	0m0.060s (no ouput)


  
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
  
  //delta = 0.01
  //Nr. of vertices =2063
  //Nr. of triangles=4122
  //time: real 0m0.008s  (no output)

  std::vector<double> xp;

  gen_roche::points_on_x_axis(xp, Omega0, q, F, deltaR);

  if (xp.size() == 0) {
    std::cout << "Error: Not init points exists.\n";
    return 0;
  }

  double  
    x0 = xp.front(),    // Left lobe          
    params[5] = {q, F, deltaR, Omega0, x0};   

  std::cout.precision(16);

  Tmarching<double, Tgen_roche<double> > march(params);
  #endif


  #if 0
  //
  // Torus
  // should not work with the current algorithm 
  //
  
  int  max_triangles = 10000000;
  
  double 
    params[2] = {1, 0.3}, // R, A
    delta = 0.01;
    
  Tmarching<double, Ttorus<double> > march(params);
  #endif

  #if 0
  //
  // Heart
  // is problematic as the surface is not smooth 
  //
  
  int  max_triangles = 10000000;
  
  double  delta = 0.01;
  
  Tmarching<double, Theart<double> > march(0);
  #endif
  
  
  std::vector <T3Dpoint<double> > V;
  std::vector <T3Dpoint<int>> T; 
  std::vector <T3Dpoint<double> >NatV;
    
  if (!march.triangulize(delta, max_triangles, V, NatV, T)){
    std::cerr << "There is too much triangles\n";
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
