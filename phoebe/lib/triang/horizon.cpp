/*
  Testing the calculating of the horizons
  
  Author: Martin Horvat, June 2016
*/

#include <iostream>
#include <cmath>

#include "../gen_roche.h"

int main(){
  
  
  //
  // Phoebe generic case: detached case
  //
  
  int  max_triangles = 10000000;

  double 
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 10,
    delta = 0.01,
    xrange[2] = {0, 0};
  
  if (!gen_roche::lobe_x_points(xrange, 0, Omega0, q, F, deltaR, true)){
    std::cerr << "Determing lobe's boundaries failed\n";
    return EXIT_FAILURE;
  }
  
  double params[5] = {q, F, deltaR, Omega0, xrange[0]};   

  std::cout.precision(16);
  std::cout << std::scientific;
  
  Tmarching<double, Tgen_roche<double> > march(params);
  
  //
  //  Generate the mesh
  //
  
  std::vector <T3Dpoint<double> > V;
  std::vector <T3Dpoint<int>> T; 
  std::vector <T3Dpoint<double> >NatV;
    
  if (!march.triangulize(delta, max_triangles, V, NatV, T)){
    std::cerr << "There is too much triangles\n";
  }

  std::cout 
    << "Nr. of vertices =" << V.size()  << '\n'
    << "Nr. of triangles=" << T.size() << '\n';
  
  
  int 
    max_iter = 100,
    length = 10000;
  
  unsigned choice = 0;
  
  double 
    p[3],
    theta = 20./180*M_PI, 
    view[3] = {std::cos(theta), 0, std::sin(theta)};
  

  //
  //  Find a point on horizon
  //
  
  if (!march.point_on_horizon(p, view, choice, max_iter)) {
    std::cerr 
    << "roche_horizon::Convergence to the point on horizon failed\n";
    return 0;
  }
  
  //
  // Estimate the step size
  //
  
  double dt = 0;
  
  if (choice == 0 || choice == 1)
    dt = utils::M_2PI*utils::hypot3(p)/length;
  else
    dt = 2*utils::M_2PI*utils::hypot3(p)/length;
  
  //
  //  Find the horizon
  //
  
  std::vector<T3Dpoint<double>> H;
 
  if (!march.horizon(H, view, p, dt)) {
   std::cerr 
    << "roche_horizon::Calculation of the horizon failed\n";
    return 0;
  }
  
  std::ofstream fr("h_vertices.dat");
  fr.precision(16);
  fr << std::scientific;
  for (auto && v: V) fr << v << '\n';
  fr.close();

    
  fr.open("h_triangles1.dat");
  for (auto && t: T) fr << t << '\n';
  fr.close();
  
  fr.open("h_triangles2.dat");
  for (auto && t: T) {
    for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';
    fr << '\n';
  }
  fr.close();

  
  fr.open("h_horizon.dat");
  for (auto && h: H) fr << h << '\n';
  fr.close();
  
  return 0;
}
