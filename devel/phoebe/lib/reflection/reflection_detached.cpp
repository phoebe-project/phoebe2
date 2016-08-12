/*
  Testing including reflection effects to star lobes. 

  Author: Martin Horvat, July 2016
*/ 

#include <iostream>
#include <cmath>

#include "../gen_roche.h"
#include "../mesh.h"
#include "reflection.h"


//#define PER_VERTEX

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
  
  std::vector<T3Dpoint<double> > V, NatV;
  std::vector<T3Dpoint<int>> Tr; 
  
  if (!march.triangulize(r, g, delta, max_triangles, V, NatV, Tr)){
    std::cerr << "There is too much triangles\n";
    return EXIT_FAILURE;
  }
  
  end = clock();
  
  std::cout << " time=" << end - start << " um\n";
  std::cout << "V.size=" << V.size() << " Tr.size=" << Tr.size() << '\n';


  //
  // Make a two body case -- changing:
  //  V, NatV, Tr
  //
  {  
    std::vector<T3Dpoint<double> > Vs(V), NatVs(NatV);
    std::vector<T3Dpoint<int>> Trs(Tr); 

    int Nv = V.size();
    
    double shift = 2;
    
    for (auto && v : V) Vs.emplace_back(v[0] + shift, v[1], v[2]);
    
    for (auto && n : NatV) NatVs.push_back(n);
        
    for (auto && t : Tr) Trs.emplace_back(t[0] + Nv, t[1] + Nv, t[2] + Nv);
    
  
    V = Vs;
    NatV = NatVs;
    Tr = Trs;
    
    std::cout 
      << "V.size=" << V.size() 
      << " NatV.size=" << NatV.size()
      << " Tr.size=" << Tr.size() << std::endl; 
  } 

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
   
  std::vector<TLDmodel<double>*> LDmodels;    // LD models

  LDmodels.push_back(new TLDlinear<double>(0.3));
 
  start = clock();

  
  #if defined(PER_VERTEX)

  int Nv = V.size();
  
  std::vector<int> 
    LDidx(Nv, 0);  // indices of the LD models in use
  
  std::vector<double> 
    R(Nv, 0.3),   // reflection coefficients
    M0(Nv, 1),     // intrinsic radient exitance 
    M;             // output radient radiosities
  
  std::vector<Tmat_elem<double>> Fmat;
 
  triangle_mesh_radiosity_wilson_vertices(V, Tr, NatV, A, LDmodels, LDidx, Fmat);

  #else
    
  int Nt = Tr.size();
  
  std::vector<int> 
    LDidx(Nt, 0);  // indices of the LD models in use
  
  std::vector<double> 
    R(Nt, 0.3),   // reflection coefficients
    M0(Nt, 1),     // intrinsic radient exitance 
    M;             // output radient radiosities
  
  std::vector<Tmat_elem<double>> Fmat;

  triangle_mesh_radiosity_wilson_triangles(V, Tr, NatT, A, LDmodels, LDidx, Fmat);
    
  #endif
     
  end = clock();
     
  std::cout << " time= " << end - start << " um\n";
  std::cout << "Fmat.size=" << Fmat.size() << '\n';   
   
  std::cout << "Reflection -- solving equation:";
  start = clock(); 
  
  if (!solve_radiosity_equation(Fmat, R, M0, M)){
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
  std::ofstream fr("triangles_detached.dat");
  fr.precision(16);
  for (auto && t: Tr)
    for (int i = 0; i < 3; ++i) fr << V[t.data[i]] << '\n';  
  fr.close();
 
  //
  // Normals
  //
  fr.open("normals_detached.dat");
  
  for (auto && n: NatT)  fr << n << '\n';
  fr.close();
  
  //
  //  Intensities
  //
  #if defined(PER_VERTEX)
  fr.open("intensity_v_detached.dat");
  #else
  fr.open("intensity_t_detached.dat");
  #endif
  {
    int N = M.size();
    for (int i = 0; i < N; ++i) fr << M0[i] << '\t' << M[i] << '\n';
  }
  fr.close();
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
    

  #if defined(PER_VERTEX)
  // Intensities per triangles
  {
    
    std::vector<double> Mt(Tr.size(), 0);
    
    int i = 0;
    for (auto && t: Tr) {
      for (int j = 0; j < 3; ++j) Mt[i] += M[t[j]]/3;
      ++i;
    }
    
    fr.open("intensity_ct_detached.dat");
    {
      int N = Mt.size();
      for (int i = 0; i < N; ++i) fr << Mt[i] << '\n';
    }
    fr.close();
  }
  #endif
  
  
  end = clock();
  
  std::cout << " time= " << end - start << " um\n";
  
  return EXIT_FAILURE;
}
