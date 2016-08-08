/*
  Testing the offseting the mesh to match the reference area.
  
  Author: Martin Horvat, June 2016
*/

#include <iostream>
#include <fstream>
#include <cmath>

#include "triang_marching.h"
#include "bodies.h"

#include "../gen_roche.h" 

int main(){
  
  
  //
  // Phoebe generic case: detached case
  //

  int
    max_triangles = 10000000,
    choice = 0;
    
  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 10,
    
    delta = 0.01;
  
  std::cout.precision(16);
  std::cout << std::scientific;

  double params[4] = {q, F, deltaR, Omega0};   

  Tmarching<double, Tgen_roche<double> > march(params);

  double r[3], g[3];
   
  if (!gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, deltaR)) {
    std::cerr << "Don't fiding the starting point\n";
    return EXIT_FAILURE;
  }
  
  //
  //  Generate the mesh
  //
  
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
  // Mesh offseting
  //  
  
  double A, xrange[2];
 
  gen_roche::lobe_x_points(xrange, choice, Omega0, q, F, deltaR, true);
  
  gen_roche::area_volume_integration(&A, 1, xrange, Omega0, q, F, deltaR);
  
  std::cout << "ref A=" << A << '\n';
  
  std::vector <T3Dpoint<double> > Vnew(V);
   
  if (!mesh_offseting_matching_area( A, Vnew, NatV, Tr)){
    std::cerr << "Offseting failed\n";
    return EXIT_FAILURE;
  }
  
  std::ofstream fr("o_vertices1.dat");
  fr.precision(16);
  fr << std::scientific;
  for (auto && v: V) fr << v << '\n';
  fr.close();


  fr.open("o_vertices2.dat");
  fr.precision(16);
  fr << std::scientific;
  for (auto && v: Vnew) fr << v << '\n';
  fr.close();

  
  fr.open("o_triangles1.dat");
  for (auto && t: Tr) {
    for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';
    fr << '\n';
  }
  fr.close();

  fr.open("o_triangles2.dat");
  for (auto && t: Tr) {
    for (int i = 0; i < 3; ++i) fr << Vnew[t.data[i]] << '\n';
    fr << '\n';
  }
  fr.close();
  
  return 0;
}
