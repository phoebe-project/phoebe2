/*
  Testing including reflection effects to star lobes. 

  Author: Martin Horvat, July 2016
*/ 

#include <iostream>
#include <cmath>

#include "../gen_roche.h"
#include "../mesh.h"
#include "reflection.h"


#define PER_VERTEX

int main(){
  
  
  clock_t start, end;
     
  //
  // Detached case
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

  double r[3], g[3];
   
  if (!gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, deltaR)) {
    std::cerr << "Don't fiding the starting point\n";
    return EXIT_FAILURE;
  }


  //
  // make triangulation of the surface
  //
  
  std::cout << "Surface triagulation:";
  
  start = clock();
  
  double params[4] = {q, F, deltaR, Omega0};
  
  Tmarching<double, Tgen_roche<double> > march(params);
  
  std::vector<T3Dpoint<double> > V0, NatV0;
  std::vector<T3Dpoint<int>> Tr0; 
  
  if (!march.triangulize(r, g, delta, max_triangles, V0, NatV0, Tr0)){
    std::cerr << "There is too much triangles\n";
    return EXIT_FAILURE;
  }
  
  end = clock();
  
  std::cout << " time=" << end - start << " um\n";
  std::cout << "V.size=" << V0.size() << " Tr.size=" << Tr0.size() << '\n';

  std::vector<T3Dpoint<double>> NatT0;
  std::vector<double> A0;
  
  
  //
  // Calc triangle properties
  //
  
  std::cout << "Triangle properties:";
    
  start = clock();
  
  mesh_attributes(V0, NatV0, Tr0, &A0, &NatT0);
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  std::cout << "A.size=" << A0.size() << '\n';
  std::cout << "NatT.size=" << NatT0.size() << '\n';
  
  //
  // Make another body
  //
  
  std::vector<T3Dpoint<double> > V1(V0), NatT1(NatT0), NatV1(NatV0);
  std::vector<T3Dpoint<int>> Tr1(Tr0); 
  std::vector<double> A1(A0);

  double shift = 2;
  for (auto && v : V1) v[0] += shift;
  
  //
  // Compose a n-body case
  //
  
  std::vector<std::vector<T3Dpoint<double>>> Vs{V0, V1}, NatTs{NatT0, NatT1}, NatVs{NatV0, NatV1};
  std::vector<std::vector<T3Dpoint<int>>> Trs{Tr0, Tr1}; 
  std::vector<std::vector<double>> As{A0, A1};
    
  //
  // Reflection effect/radiosity problem Wilson
  //
  
  std::cout << "Reflection -- determine matrix:";
   
  std::vector<TLDmodel<double>*> LDmodels;    // LD models
  LDmodels.push_back(new TLDlinear<double>(0.3));
  LDmodels.push_back(new TLDlinear<double>(0.3));
  //LDmodels.push_back(new TLDquadratic<double>(0.3, 0.3));
  
  std::vector<Tmat_elem_nbody<double>> Fmat;
      
  start = clock();
  
  #if defined(PER_VERTEX)
  triangle_mesh_radiosity_wilson_vertices_nbody_convex(Vs, Trs, NatVs, As, LDmodels, Fmat);
  #else
  triangle_mesh_radiosity_wilson_triangles_nbody_convex(Vs, Trs, NatTs, As, LDmodels, Fmat);
  #endif
  
     
  end = clock();
     
  std::cout << " time= " << end - start << " um\n";
  std::cout << "Fmat.size=" << Fmat.size() << '\n';   
   
  std::cout << "Reflection -- solving equation:";
 
  std::vector<std::vector<double>> Rs(2), M0s(2), Ms;
  
  #if defined(PER_VERTEX)
  for (int i = 0; i < 2; ++i) {
    int N = Vs[i].size(); 
    Rs[i].assign(N, 0.3);
    M0s[i].assign(N, 1);
  }
  #else 
  for (int i = 0; i < 2; ++i) {
    int N = Trs[i].size(); 
    Rs[i].assign(N, 0.3);
    M0s[i].assign(N, 1);
  }
  #endif
  
  start = clock(); 
  
  if (!solve_radiosity_equation_nbody(Fmat, Rs, M0s, Ms)){
    std::cerr << "reflection:we did not converge\n";
    return EXIT_FAILURE;
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
  std::ofstream fr("triangles_nbody.dat");
  fr.precision(16);
  for (int i = 0; i < 2; ++i)
    for (auto && t : Trs[i])
      for (int j = 0; j < 3; ++j) fr << Vs[i][t[j]] << '\n';  
  fr.close();
 
  //
  // Normals
  //
  fr.open("normals_nbody.dat");
  
  for (auto && B : NatTs) for (auto && n : B) fr << n << '\n';
  fr.close();
  
  //
  // Save matrix
  //
  
  #if defined(PER_VERTEX)
  fr.open("matrix_v_nbody.dat");
  #else // PER_TRIANGLE
  fr.open("matrix_t_nbody.dat");
  #endif
  
  {
    int d = Trs[0].size();
    
    for (auto && f : Fmat) 
      fr << f.b1*d + f.i1 << ' ' << f.b2*d + f.i2 << ' ' << f.F << '\n';
  }
  fr.close();
  
  //
  //  Intensities
  //
  
  #if defined(PER_VERTEX)
  fr.open("intensity_v_nbody.dat");
  #else // PER_TRIANGLE
  fr.open("intensity_t_nbody.dat");
  #endif  
  {
    for (int i = 0; i < 2; ++i)
      for (int j = 0, N = M0s[i].size(); j < N; ++j) 
        fr << M0s[i][j] << '\t' << Ms[i][j] << '\n';
  }
  fr.close();
  
  end = clock();
  
  #if defined(PER_VERTEX)
  // Intensities per triangles
  {
    
    std::vector<std::vector<double>> Mts(2);
    
    for (int b = 0; b < 2; ++b) {
      
      Mts[b].resize(Trs[b].size(), 0);
      
      int i = 0;
      for (auto && t: Trs[b]) {
        for (int j = 0; j < 3; ++j) Mts[b][i] += Ms[b][t[j]]/3;
        ++i;
      }
    }
    
    fr.open("intensity_ct_nbody.dat");
    {
      for (int b = 0; b < 2; ++b) {
        int N = Mts[b].size();
        for (int i = 0; i < N; ++i) fr << Mts[b][i] << '\n';
      }
    }
    fr.close();
  }
  #endif
  
  
  std::cout << " time= " << end - start << " um\n";
  
  return EXIT_FAILURE;
}
