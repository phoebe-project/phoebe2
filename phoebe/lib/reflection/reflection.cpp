/*
  Testing including reflection effects to star lobes. 

  Author: Martin Horvat, July 2016
*/ 

#include <iostream>
#include <cmath>

#include "../gen_roche.h"
#include "../mesh.h"
#include "reflection.h"

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
    
    delta = 0.02;

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
  std::vector<T3Dpoint<int>> Tr; 
  
  if (!march.triangulize(delta, max_triangles, V, NatV, Tr)){
    std::cerr << "There is too much triangles\n";
  }
  
  end = clock();
  
  std::cout << " time=" << end - start << " um\n";
  std::cout << "V.size=" << V.size() << " Tr.size=" << Tr.size() << '\n';


  //
  // Calc triangle properties
  //
  
  std::cout << "Triangle properties:";
  
  start = clock();
    
  std::vector<T3Dpoint<double>> NatT;
  std::vector<double> A;
    
  mesh_attributes(V, NatV, Tr, &A, &NatT);
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  std::cout << "A.size=" << A.size() << '\n';
  std::cout << "NatT.size=" << NatT.size() << '\n';
   
  //
  // Reflection effect/radiosity problem Wilson
  //
  
  std::cout << "Reflection -- determine matrix:";
  start = clock();
   
  std::vector<TLDmodel<double>*> LDmodels;    // LD models

  LDmodels.push_back(new TLDlinear<double>(0.3));
  
  std::vector<int> 
    LDidx(Tr.size(), 0);  // indices of the LD models in use
  
  std::vector<double> 
    R(Tr.size(), 0.75),   // reflection coefficients
    M0(Tr.size(), 1),     // intrinsic radient exitance 
    M;                    // output radient radiosities
  
  std::vector<Tmat_elem<double>> Fmat;
   
  triangle_mesh_radiosity_wilson(V, Tr, NatT, A, LDmodels, LDidx, Fmat);
   
  end = clock();
   
  std::cout << " time= " << end - start << " um\n";
  std::cout << "Fmat.size=" << Fmat.size() << '\n';   
   
  std::cout << "Reflection -- solving equation:";
  start = clock(); 
  
  if (!solve_radiosity_equation(Fmat, R, M0, M)){
    std::cerr << "reflection:we did not converge\n";
  }
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  
  //
  // Storing results
  //
  
  std::cout << "Storing results:";
  
  start = clock();
  
  
  //
  // Save triangles
  //
  std::ofstream fr("triangles.dat");
  fr.precision(16);
  for (auto && t: Tr)
    for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';  
  fr.close();
 
  //
  // Normals
  //
  fr.open("normals.dat");
  
  for (auto && n: NatT)  fr << n << '\n';
  fr.close();
  
  //
  //  Intensities
  //
  fr.open("intensity.dat");
  {
    int Nt = M.size();
    for (int i = 0; i < Nt; ++i) fr << M0[i] << '\t' << M[i] << '\n';
  }
  fr.close();
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  
  
  return EXIT_FAILURE;
}
