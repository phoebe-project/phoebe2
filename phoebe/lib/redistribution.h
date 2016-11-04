#if !defined(__redistribution_h)
#define __redistribution_h

/*
  Library discussing the process of reflection with (quasi) stationary
  redistribution of the incoming flux over the surface of the body.

  Author: Martin Horvat, October 2016
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <utility>
#include <map>
#include <cmath>
#include <string>

#include "hash.h"
#include "utils.h"
#include "redistribution.h"


/*
  Matrix element of the sparse-matrix
*/
template <class T>
struct Tsparse_mat_elem {  
  int i, j;
  T value;
  
  Tsparse_mat_elem() {}
  Tsparse_mat_elem(int i, int j, const T &value) 
  : i(i), j(j), value(value) {}
};


/*
  Fit sphere to the vertices
  
  Input:
    V - vector of vertices
  
  Output:
    R - radious of the sphere
    x - center of the sphere
  
  Return:
    true - if unique solution exists, 
    false - otherwise
*/

template <class T>
bool fit_sphere(std::vector<T3Dpoint<T>> & V, T & R, T x[3]){
  
  int i, j, k, n = V.size();
  
  T a[3], b, c[3][3], d[3];
  
  b = 0; 
  for (i = 0; i < 3; ++i) {
    a[i] = d[i] = 0; 
    for (j = 0; j < 3; ++j) c[i][j] = 0;  
  }

  // a = Sum[data[[i]], {i, n}]/n;
  // b = Sum[data[[i]].data[[i]], {i, n}]/n;
  // c = Sum[Outer[Times, data[[i]], data[[i]]], {i, n}]/n;
  // d = Sum[data[[i]]*(data[[i]].data[[i]]), {i, n}]/n;
  
  T t, sum, *v;
 
  for (auto && p: V) {
    
    v = p.data;
    
    sum = 0;
    for (j = 0; j < 3; ++j) {
      a[j] += (t = v[j]);
      sum += t*t; 
    }
    
    b += sum;
    
    for (j = 0; j < 3; ++j) {
      d[j] += (t = v[j])*sum;
      c[j][j] += t*t;
      for (k = 0; k < j; ++k) c[j][k] += t*v[k];
    }
  }
  
  b /= n;
  for (i = 0; i < 3; ++i) {
    a[i] /= n; 
    d[i] /= n; 
    c[i][i] /= n;
    for (j = 0; j < i; ++j) c[i][j] /= n;
  }
  
  // f = Outer[Times, a, a] - c;
  T f[3][3];
  for (i = 0; i < 3; ++i) { 
    f[i][i] = a[i]*a[i] - c[i][i];
    for (j = 0; j < i; ++j) f[j][i] = f[i][j] = a[i]*a[j] - c[i][j];
  }

  // e = b*a - d;
  T e[3];
  for (i = 0; i < 3; ++i) e[i] = b*a[i] - d[i]; 
  
  // x = Inverse[f].e/2;
  if (!utils::solve3D(f, e, x)) {
    std::cerr << "fit_sphere::The matrix is singular\n";
    return false;
  }
  
  for (i = 0; i < 3; ++i) x[i] *= 0.5;
  
  // R = sqrt(x.x - 2 x.a + b);
  sum = b;
  for (i = 0; i < 3; ++i) sum += x[i]*(x[i] - 2*a[i]);
  
  R = std::sqrt(sum);

  return true;
}
/*
  Calculating barycenters

  Input:
    V - vector of vertices
    Tr - vector of triangles 

  Output:
    B - vector of barycenters associated to triangles
*/ 
template<class T> 
void calc_barycenters(
  std::vector<T3Dpoint<T>>   & V,     // inputs 
  std::vector<T3Dpoint<int>> & Tr,
  std::vector<T3Dpoint<T>> & B        // output                
){
  
  B.resize(Tr.size());

  int i, *t;
  
  T *v[3], *c;

  auto it = B.begin();

  for (auto && tr : Tr) {

    t = tr.data;
    
    // get pointers to vertices of triangle tr
    for (i = 0; i < 3; ++i) v[i] = V[t[i]].data;
    
    // calc barycenters
    c = (it++)->data;
    for (i = 0; i < 3; ++i) 
      c[i] = (v[0][i] + v[1][i] + v[2][i])/3;
  }  
}

/*
  Transform points into directions:
    
    p -> (p - x)/|p -x|  on unit sphere
  
  with x center of the sphere.
  
  Input:
    x - center of unit sphere
    P - vector of points  
  Output:
    P - vector on unit sphere
*/
template<class T>
void calc_directions(
  T x[3],                             // input
  std::vector<T3Dpoint<T>> & P        // input & output
){
  
  int i;
  
  T *v, t, f;
  
  for (auto && c : P) {
  
    v = c.data;
    
    f = 0;
    for (i = 0; i < 3; ++i) {
      t = (v[i] -= x[i]);
      f += t*t;
    }
    
    f = 1/std::sqrt(f);
    for (i = 0; i < 3; ++i) v[i] *= f;
  }
}

/*
  Calculating connectivity matrix based the rule
    
    C_{i,j} = 1 if distance is < thresh
              0 otherwise
              
  Input:
    thresh - threshold value of distance
    P - vector of points on the unit sphere
    
  Output:
    C - matrix of non-zero connections C[i] = {j_1, j_2, ..., j_k}
        equal to non-zero:
          C_{i,j_1}, ..., C_{i,j_k}
*/
template<class T>
void calc_connectivity(
  const T & thresh, 
  std::vector<T3Dpoint<T>> & P, 
  std::vector<std::vector<int>> & C)
{ 
  
  int i, j, N = P.size();
  
  C.resize(N);
  
  T *v1, *v2, tmp;
                        
  for (i = 0; i < N; ++i) {
  
    v1 = P[i].data;
    
    for (j = 0; j < i; ++j) {
    
      v2 = P[j].data;

      tmp = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
      
      if (std::acos(tmp) <= thresh) {
        C[i].push_back(j);
        C[j].push_back(i);
      }
    }
    
    C[i].push_back(i);
  }
}

void print_connectivity(
  const char *filename,
  std::vector<std::vector<int>> & C
) {
  
  std::ofstream f(filename);

  int i = 0;
  for (auto && c : C) {
    for (auto && j : c) f << i << '\t' << j << '\n';   
    ++i;
  }
}


/*
  Calculating connectivity matrix based the rule
    
    C_{i,j} = 1 if distance is < thresh
              0 otherwise
              
  Input:
    thresh - threshold value of distance
    o - axis  (|o| = 1)
    P - vector of points on the unit sphere
    
  Output:
    C - matrix of non-zero connections C[i] = {j_1, j_2, ..., j_k}
        equal to non-zero:
          C_{i,j_1}, ..., C_{i,j_k}
*/
template<class T>
void calc_connectivity(
  const T & thresh, 
  T o[3],
  std::vector<T3Dpoint<T>> & P, 
  std::vector<std::vector<int>> & C)
{ 
  
  int i, j, N = P.size();
  
  C.resize(N);
  
  T *v1, *v2, c1, c2, tmp;
              
  for (i = 0; i < N; ++i) {

    v1 = P[i].data;
    c1 = v1[0]*o[0] + v1[1]*o[1] + v1[2]*o[2];
     
    for (j = 0; j < i; ++j) {
    
      v2 = P[j].data;
      c2 = v2[0]*o[0] + v2[1]*o[1] + v2[2]*o[2];

      tmp = std::sqrt((1 - c1*c1)*(1 - c2*c2)) + c1*c2;
      
      if (std::acos(tmp) <= thresh) {
        C[i].push_back(j);
        C[j].push_back(i);
      }
    }
    
    C[i].push_back(i);
  }
}

/*
  Calculating redistribution matrix D_{i,j} based on the 
  connectivity matrix C_{i,j}:

    D_{i,j} = A_j C_{i,j}/(sum_k A_k C_{k,j})
              
  Input:
    C - matrix of non-zero connections C[i] = {j_1, j_2, ..., j_k}
        equal to non-zero:
          C_{i,j_1}, ..., C_{i,j_k}
    A - vector of areas
    
  Output:
    Dmat - sparse matrix
*/

template<class T>
void calc_redistrib_matrix(
  std::vector<std::vector<int>> &C, 
  std::vector<T> &A, 
  std::vector<Tsparse_mat_elem<T>> & Dmat)
{

  T f;
  
  int j = 0;
  
  auto it = A.begin();
  
  for (auto && c : C) {
    f = 0;
    for (auto && i : c) f += A[i];
    f = 1/f;
    
    for (auto && i : c) Dmat.emplace_back(i, j, f*(*it)); 
    
    ++j;
    ++it;
  }
}

/*
  Calculating redistribution matrix D_{i,j} for uniform redistribution:
  
    D_{i,j} = A_j/A
              
  Input:
    A - vector of areas
    
  Output:
    Dmat - sparse matrix
*/

template<class T>
void calc_redistrib_matrix(
  std::vector<T> &A, 
  std::vector<Tsparse_mat_elem<T>> & Dmat)
{
  T f = 0;
  for (auto && a: A) f += a;
  f = 1/f;

  int i, j = 0, N = A.size();
  
  Dmat.reserve(N*N);
  
  T ra;
  
  for (auto && a : A){
    ra = a*f;
    for (i = 0; i < N; ++i) Dmat.emplace_back(i, j, ra);
    ++j;
  }
}

/*
  Calculate areas at vertices.

  Input:
    Nv - number of vertices
    Tr - vectors of triangles
    A - vector of areas of triangles

  Output:
    AatV - vector of areas at vertices
*/
template <class T>
void calc_area_at_vertices(
  int Nv, 
  std::vector<T3Dpoint<int>> & Tr,
  std::vector<T> & A,
  std::vector<T> & AatV
){
  AatV.clear();
  AatV.resize(Nv, T(0));
  
  int *t;
  
  auto it = A.begin();
  
  T a;
  
  for (auto && tr: Tr) {
    t = tr.data;
    a = (*(it++))/3;
    for (int j = 0; j < 3; ++j) AatV[t[j]] += a;
  }
}

/*
  Calculate redistribution matrices for a given body in the per-triangle 
  discretization. Surface represents a single body.

  Input:

    V - vectors of vertices
    Tr - vectors of triangles
    NatT - vectors of normals at triangles
    A - vector of areas of triangles
    Dpars - map of redistribution model parameters
            which are used to calculate distribution matrices
          
          Models supported:
            global  0 parameters
            local   1 parameter [h] [h is angle in radians]
                        h = 0: flux is reflected back from element that
                        accepted it 
            horiz   4 parameters: [o_x, o_y, o_z, h]
            
          Example:
             Dpar["global"] = std::vector<T>();
             Dpar["local"] = std::vector<T>(1);
             Dpar["horiz"] = std::vector<T>(4);
             
  
  Output:
    Dmats - map of redistribution matrices in sparse format
  
  Note:
  * distances on the body area calculated by reprojecting mesh onto
    tightest fitted sphere
  * using sharp edge in definion of the neighborhood of width h 
*/
template <class T> 
bool triangle_mesh_redistribution_matrix_triangles(
  std::vector<T3Dpoint<T>>   & V,                                 // inputs 
  std::vector<T3Dpoint<int>> & Tr,
  std::vector<T3Dpoint<T>>   & NatT,
  std::vector<T> & A,
  std::map<fnv1a_32::hash_t, std::vector<T>> & Dpars,
  std::map<fnv1a_32::hash_t, std::vector<Tsparse_mat_elem<T>>> & Dmats  // output
){
  
  const char *fname = "triangle_mesh_redistribution_matrix_triangles";
   
  int i, Nt = Tr.size();
  
  bool st = true;
  
  std::vector<T3Dpoint<T>> P;
  
  Dmats.clear();
    
  T x[3], r;
  
  for (auto && p : Dpars){
    
    auto & par = p.second;
    auto & Dmat = Dmats[p.first];
    
    switch (p.first) {
      
      case "none"_hash32: break;
      
      case "global"_hash32: 
        calc_redistrib_matrix(A, Dmat);
        break;
      
      case "local"_hash32:
      {
        if (par.size() != 1) {
          std::cerr << fname << "::Params wrongs size\n";
          return false;
        }
        
        T h = par[0]; // neighborhood size

        if (h == 0) {  // h == 0
          Dmat.reserve(Nt);
          for (i = 0; i < Nt; ++i) Dmat.emplace_back(i, i, T(1));
        } else {        // h > 0
          
          if (st) {
            // calculate of characteristic points: barycenters
            calc_barycenters(V, Tr, P);

            // fitting sphere to barycenters
            if (!fit_sphere(P, r, x)) {
              std::cerr << fname << "::Fitting sphere failed\n";
              return false;
            }
            
            // project characteristic points onto unit sphere
            // v -> (v - x)/|v - x|          
            calc_directions(x, P);
            
            st = false;
          }
          
          // creating "connectivity" matrix 
          std::vector<std::vector<int>>C;
          
          calc_connectivity(h, P, C);
          
          // creating re-distribution matrix from connectivity matrix          
          calc_redistrib_matrix(C, A, Dmat);
        }
      }
      
      break;
      
      case "horiz"_hash32:
      {
        
        if (par.size() != 4) {
          std::cerr << fname << "::Params wrongs size\n";
          return false;
        }
        
        T o[3] = {par[0], par[1], par[2]},   // axis
          h = par[3];                        // neighborhood size

        if (h == 0) {   // h == 0
          Dmat.reserve(Nt);
          for (i = 0; i < Nt; ++i) Dmat.emplace_back(i, i, T(1));
        } else {        // h > 0
          
          if (st) {            
            // calculate characteristic points: barycenters
            calc_barycenters(V, Tr, P);
            
            // fitting sphere to barycenters          
            T x[3], r;
            
            if (!fit_sphere(P, r, x)) {
              std::cerr << fname << "::Fitting sphere failed\n";
              return false;
            }
            
            // project characteristic points onto unit sphere
            // v -> (v - x)/|v - x|          
            calc_directions(x, P);
            
            st = false;
          }
          
          // creating "connectivity" matrix 
          std::vector<std::vector<int>>C;
          
          calc_connectivity(h, o, P, C);
           
          // creating re-distribution matrix from connectivity matrix
          calc_redistrib_matrix(C, A, Dmat);
        }
      }
      break;
      
      default:
        std::cerr 
          << fname 
          << "::\nThis type of redistribution is not supported\n";
        return false;
    }
  }
  
  return true;
}

/*
  Calculate redistribution matrices for a given body in the per-vertices 
  discretization. Surface represents a single body.

  Input:

    V - vector of n vectors of vertices
    Tr - vector of n vectors of triangles
    NatV - vector of n vectors of normals at vertices
    A - vector of n vector of areas of triangles
    Dpars - map of redistribution model parameters
            which are used to calculate distribution matrices
          
          Models supported:
            none    0 paramters
            global  0 parameters
            local   1 parameter [h is angle in radians]
                        h = 0: flux is reflected back from element that
                        accepted it 
            horiz   4 parameters: [o_x, o_y, o_z, h]
            
          Example:
             Dpar["global"] = std::vector<T>();
             Dpar["local"] = std::vector<T>(1);
  
  Output:
    Dmats - map of redistribution matrices in sparse format
  
  Note:
  * distances on the body area calculated by reprojecting mesh onto
    tightest fitted sphere
  * using sharp edge in definion of the neighborhood of width h  
*/ 
template <class T>
bool triangle_mesh_redistribution_matrix_vertices(
  std::vector<T3Dpoint<T>>   & V,                                 // inputs 
  std::vector<T3Dpoint<int>> & Tr,
  std::vector<T3Dpoint<T>>   & NatV,
  std::vector<T> & A,
  std::map<fnv1a_32::hash_t, std::vector<T>> & Dpars,
  std::map<fnv1a_32::hash_t, std::vector<Tsparse_mat_elem<T>>> & Dmats  // output
){
  
  const char *fname = "triangle_mesh_redistribution_matrix_vertices";
  
  // calculate areas associates with vertices
  int i, Nv = V.size();
    
  // variables to prevent multiple calculations
  bool st[2] = {true, true};
  
  std::vector<T3Dpoint<T>> P;
  
  std::vector<T> AatV;
  
  Dmats.clear();
  
  T x[3], r;
 
  for (auto && p : Dpars){
    
    auto & par = p.second;
    auto & Dmat = Dmats[p.first];
    
    switch (p.first) {
      
      case "none"_hash32: break;
      
      case "global"_hash32:
      
      if (st[0]) {
        calc_area_at_vertices(Nv, Tr, A, AatV);
        st[0] = false;
      }
      
      calc_redistrib_matrix(AatV, Dmat);
      break;
        
      case "local"_hash32:
      {
        
        if (par.size() != 1) {
          std::cerr << fname << "::Params wrongs size\n";
          return false;
        }
        
        T h = par[0]; // neighborhood size

        if (h == 0) {  // h == 0
          Dmat.reserve(Nv);
          for (i = 0; i < Nv; ++i) Dmat.emplace_back(i, i, T(1));
        } else {        // h > 0
          
          if (st[0]) {
            calc_area_at_vertices(Nv, Tr, A, AatV);
            st[0] = false;
          }
          
          if (st[1]) {
            P = V;
            
            // fitting sphere to vertices
            if (!fit_sphere(P, r, x)) {
              std::cerr << fname << "::Fitting sphere failed\n";
              return false;
            }

            // project characteristic points onto unit sphere
            // v -> (v - x)/|v - x|
            calc_directions(x, P);
      
            st[1] = false;
          }
          
          // creating "connectivity" matrix 
          std::vector<std::vector<int>>C;          
          calc_connectivity(h, P, C);
          
          //print_connectivity("c_local.txt", C);
            
          // creating re-distribution matrix from connectivity matrix
          calc_redistrib_matrix(C, AatV, Dmat);
        }
      }
      
      break;
      
      case "horiz"_hash32:
      {
        if (par.size() != 4) {
          std::cerr << fname << "::Params wrongs size\n";
          return false;
        }
        
        T o[3] = {par[0], par[1], par[2]},   // axis
          h = par[3];                        // neighborhood size
        
//        std::cerr 
//          << "o=" << o[0] << ' ' << o[1] << ' ' << o[2] << '\n'
//          << " h=" << h << '\n';
          
        if (h == 0) {  // h == 0
          Dmat.reserve(Nv);
          for (i = 0; i < Nv; ++i) Dmat.emplace_back(i, i, T(1));
        } else {        // h > 0
          
          if (st[0]) {
            calc_area_at_vertices(Nv, Tr, A, AatV);
            st[0] = false;
          }         
          
          if (st[1]) {
            P = V;
            
            // fitting sphere to vertices
            if (!fit_sphere(P, r, x)) {
              std::cerr << fname << "::Fitting sphere failed\n";
              return false;
            }

            // project characteristic points onto unit sphere
            // v -> (v - x)/|v - x|
            calc_directions(x, P);
           
            st[1] = false;
          }
          
          // creating "connectivity" matrix 
          std::vector<std::vector<int>>C;
          calc_connectivity(h, o, P, C);

          //print_connectivity("c_horiz.txt", C);
            
          // creating re-distribution matrix from connectivity matrix
          calc_redistrib_matrix(C, AatV, Dmat);
        }
      }
      break;
      
      default:
        std::cerr 
          << fname 
          << "::\nThis type of redistribution is not supported\n";
        return false;
    }
  }
  
  return true;  
}


/*
  Solving the radiosity-redistribution model -- a combination of 
  Wilson's reflection model and redistribution framework:
  
    F1 = F0 + D (id - diag(R)) Fin
    Fin = L_{LD} Fout
    Fout = F1 + diag(R) Fin 

  or
    Fout 
      = F0 + D (id - diag(R)) Fin + diag(R) Fin 
      = F0 + {D(id - diag(R)) + diag(R)} L_{LD} Fout
    
    F1 = F0 + D(1-diag(R) L_{LD} Fout
    
  where 
    L_{LD} - matrix of limb-darkened view-factors
    D - redistribution matrix 
    
    R is vector of reflection coefficients, 
    F0 is vector of intrinsic radiant exitances from triangles/vertices
    F1 updated intrinsic radiant exitances from triangles/vertices
    Fout is vector of radiosity (intrinsic and reflection) of triangles/vertices
  
  Method: Simple iteration 

  Input:
    Lmat - matrix of view factors 
    Dmat - matrix of redistribution view factor 
    
    R - vector of albedo/reflection of triangles/of vertices
    F0 - vector of intrisic radiant exitance of triangles/of vertices
    epsF - relative precision of radiosity    
    max_iter - maximal number of iterations
 
  Output:
    F1   - updated vector of intrisic radiant exitance of triangles/of vertices
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices
     
  Returns:
    true if we reached wanted relative precision, false otherwise
*/


template <class T>
void solve_radiosity_equation_with_redistribution_Wilson(
  std::vector<Tview_factor<T>> &Lmat,        // input
  std::vector<Tsparse_mat_elem<T>> &Dmat,      
  std::vector<T> &R,  
  std::vector<T> &F0, 
  std::vector<T> &F1,                       // output
  std::vector<T> &Fout,                 
  const T & epsF = 1e-12,                   // optional params
  const T & max_iter = 100){

  int N = R.size();

  T *M0 = new T [2*N], *M1 = M0 + N;

  //
  // Iteration: 
  //   Fout_{k+1} = F0 + {D(id - diag(R)) + diag(R)} L_{LD} Fout_{k}
  //
  
  int iter = 0;
  
  T t, dt, dF, Fmax;

  Fout = F0;
  
  do {
    
    // M0 =  L_{LD} Fout
    memset(M0, 0, sizeof(T)*N);
    for (auto && f: Lmat) M0[f.i] += f.F*Fout[f.j];

    // M1 = diag(R) M0
    for (int i = 0; i < N; ++i) M1[i] = R[i]*M0[i];
    
    // F1 = F0 + D(id - diag(R)) M0 
    F1 = F0;
    for (auto && f: Dmat) F1[f.i] += f.value*(1- R[f.j])*M0[f.j]; 

    // Fout = F1 + M1
    Fmax = dF = 0;
    for (int i = 0; i < N; ++i) {
      t = F1[i] + M1[i];
      
      if (t > Fmax) Fmax = t;
      
      dt = std::abs(Fout[i] - t);
      
      if (dt > dF) dF = dt;
      
      Fout[i] = t;
    }
    
  } while (dF >= Fmax*epsF && ++iter < max_iter);

  delete [] M0;
  
  return iter < max_iter;
}

/*
  Solving the radiosity-redistribution model -- a combination of 
  Horvat's reflection model and redistribution framework:
   
    F1 = F0 + D (id - diag(R)) Fin
    
    Fin = L_{LD} F1 + L0 diag(R) Fin 
   
    Fout = F1 + diag(R) Fin 

  or
    Fin = L_{LD} F0 + [L_{LD} D (1- diag(R)) + L0 diag(R)] Fin
    F1 = F0 + D(1-diag(R)) Fin
    Fout = F1 + diag(R) Fin 
  
  where 
    L_{LD} - matrix of limb-darkened view-factors
    L_{LD} - matrix of Lambertian view-factors
    D - redistribution matrix 
  
    R is vector of reflection coefficients, 
    F0 is vector of intrinsic radiant exitances from triangles/vertices
    F1 updated intrinsic radiant exitances from triangles/vertices
    Fout is vector of radiosity (intrinsic and reflection) of triangles/vertices
  
  Method: Simple iteration 

  Input:
    Lmat - matrix of view factors 
    Dmat - matrix of redistribution view factor 
    
    R - vector of albedo/reflection of triangles/of vertices
    F0 - vector of intrisic radiant exitance of triangles/of vertices
    epsF - relative precision of radiosity    
    max_iter - maximal number of iterations
 
  Output:
    F1   - updated vector of intrisic radiant exitance of triangles/of vertices
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices
     
  Returns:
    true if we reached wanted relative precision, false otherwise
*/

template <class T>
void solve_radiosity_equation_with_redistribution_Horvat(
  std::vector<Tview_factor<T>> &Lmat,        // input
  std::vector<Tsparse_mat_elem<T>> &Dmat,      
  std::vector<T> &R,  
  std::vector<T> &F0, 
  std::vector<T> &F1,                       // output
  std::vector<T> &Fout,                 
  const T & epsF = 1e-12,                   // optional params
  const T & max_iter = 100
){

  int N = R.size();
  
  //
  //  S0 = L_{LD} F0
  //

  std::vector<T> S0(N, T(0));
 
  for (auto && f: Lmat) S0[f.i] += f.F*F0[f.j];
  
  //
  // Iteration: 
  //   Fin_{k+1} = S0 + [L_{LD} D (1- diag(R)) + L0 diag(R)] Fin_k
  //
   
  int iter = 0;
  
  T t, dt, dF, Fmax,
    *M0 = new T [3*N], 
    *M1 = M0 + N,
    *M2 = M1 + N;
  
  std::vector<T> Fin(S0);
  
  do {
    
    // M0 =  L0 diag(R) Fin
    memset(M0, 0, sizeof(T)*3*N);
    for (auto && f: Lmat) M0[f.i] += f.F0*R[f.j]*Fin[f.j];

    // M1 = D (1- diag(R))Fin
    for (auto && f: Dmat) M1[f.i] += f.value*(1 - R[f.j])*Fin[f.j];
    
    // M2 = L_{LD} M1
    for (auto && f: Lmat) M2[f.i] += f.F*M1[f.j];
     
    // Fin = S0 + M0 + M2
    Fmax = dF = 0;
    for (int i = 0; i < N; ++i) {
      t = S0[i] + M0[i] + M2[i];
      
      if (t > Fmax) Fmax = t;
      
      dt = std::abs(Fin[i] - t);
      
      if (dt > dF) dF = dt;
      
      Fin[i] = t;
    }
    
  } while (dF >= Fmax*epsF && ++iter < max_iter);

  // F1 = F0 + M1   M1 = D(1-diag(T)) Fin
  F1 = F0;
  for (int i = 0; i < N; ++i) F1[i] += M1[i];
  
  // Fout = F1 + diag(R) Fin
  Fout = F1;
  for (int i = 0; i < N; ++i) Fout[i] += R[i]*Fin[i];
  
  delete [] M0;
  
  return iter < max_iter;  
}

/*
  Solving the radiosity-redistribution model -- a combination of 
  Wilson's reflection model and redistribution framework:
  
    F1 = F0 + D (id - diag(R)) Fin
    Fin = L_{LD} Fout
    Fout = F1 + diag(R) Fin 

  or
    Fout 
      = F0 + D (id - diag(R)) Fin + diag(R) Fin 
      = F0 + {D(id - diag(R)) + diag(R)} L_{LD} Fout
    
    F1 = F0 + D(1-diag(R) L_{LD} Fout
    
  where 
    L_{LD} - matrix of limb-darkened view-factors
    D - redistribution matrix 
    
    R is vector of reflection coefficients, 
    F0 is vector of intrinsic radiant exitances from triangles/vertices
    F1 updated intrinsic radiant exitances from triangles/vertices
    Fout is vector of radiosity (intrinsic and reflection) of triangles/vertices
  
  Method: Simple iteration 

  Input:
    Lmat - matrix of view factors 
    Dmat - matrix of redistribution view factor 
    
    R - vector of albedo/reflection of triangles/of vertices
    F0 - vector of intrisic radiant exitance of triangles/of vertices
    epsF - relative precision of radiosity    
    max_iter - maximal number of iterations
 
  Output:
    F1   - updated vector of intrisic radiant exitance of triangles/of vertices
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices
     
  Returns:
    true if we reached wanted relative precision, false otherwise
*/


template <class T>
bool solve_radiosity_equation_with_redistribution_Wilson_nbody(
  std::vector<Tview_factor_nbody<T>> &Lmat,     // input
  std::vector<std::vector<Tsparse_mat_elem<T>>> &Dmat,      
  std::vector<std::vector<T>> &R,  
  std::vector<std::vector<T>> &F0, 
  std::vector<std::vector<T>> &F1,                      // output
  std::vector<std::vector<T>> &Fout,                 
  const T & epsF = 1e-12,                               // optional 
  const T & max_iter = 100){

  // number of bodies
  int i, b, nb = F0.size();
  
  // lengths of sub-vectors
  std::vector<int> N(nb);
  for (b = 0; b < nb; ++b) N[b] = F0[b].size();

  std::vector<std::vector<T>> M0(nb), M1(nb);
  
  for (b = 0; b < nb; ++b) {
    M0[b].resize(N[b], T(0));
    M1[b].resize(N[b], T(0));
  } 
   
  //
  // Iteration: 
  //   Fout_{k+1} = F0 + {D(id - diag(R)) + diag(R)} L_{LD} Fout_{k}
  //
  
  int iter = 0;
  
  T t, dt, dF, Fmax;

  Fout = F0;
  
  do {
    
    // M0 =  L_{LD} Fout
    for (b = 0; b < nb; ++b) memset(M0[b].data(), 0, sizeof(T)*N[b]);   
    for (auto && f: Lmat) M0[f.b1][f.i1] += f.F*Fout[f.b2][f.i2];

    // M1 = diag(R) M0
    for (b = 0; b < nb; ++b)
      for (i = 0; i < N[b]; ++i) 
        M1[b][i] = R[b][i]*M0[b][i];
    
    // F1 = F0 + D(id - diag(R)) M0 
    F1 = F0;
    for (b = 0; b < nb; ++b)
      for (auto && f: Dmat[b]) 
        F1[b][f.i] += f.value*(1- R[b][f.j])*M0[b][f.j]; 

    // Fout = F1 + M1
    Fmax = dF = 0;
    for (b = 0; b < nb; ++b)
      for (i = 0; i < N[b]; ++i) {
        t = F1[b][i] + M1[b][i];
        
        if (t > Fmax) Fmax = t;
        
        dt = std::abs(Fout[b][i] - t);
        
        if (dt > dF) dF = dt;
        
        Fout[b][i] = t;
      }
    
  } while (dF >= Fmax*epsF && ++iter < max_iter);

  
  return iter < max_iter;
}

/*
  Solving the radiosity-redistribution model -- a combination of 
  Horvat's reflection model and redistribution framework:
   
    F1 = F0 + D (id - diag(R)) Fin
    
    Fin = L_{LD} F1 + L0 diag(R) Fin 
   
    Fout = F1 + diag(R) Fin 

  or
    Fin = L_{LD} F0 + [L_{LD} D (1- diag(R)) + L0 diag(R)] Fin
    F1 = F0 + D(1-diag(R)) Fin
    Fout = F1 + diag(R) Fin 
  
  where 
    L_{LD} - matrix of limb-darkened view-factors
    L_{LD} - matrix of Lambertian view-factors
    D - redistribution matrix 
  
    R is vector of reflection coefficients, 
    F0 is vector of intrinsic radiant exitances from triangles/vertices
    F1 updated intrinsic radiant exitances from triangles/vertices
    Fout is vector of radiosity (intrinsic and reflection) of triangles/vertices
  
  Method: Simple iteration 

  Input:
    Lmat - matrix of view factors 
    Dmat - matrix of redistribution view factor 
    
    R - vector of albedo/reflection of triangles/of vertices
    F0 - vector of intrisic radiant exitance of triangles/of vertices
    epsF - relative precision of radiosity    
    max_iter - maximal number of iterations
 
  Output:
    F1   - updated vector of intrisic radiant exitance of triangles/of vertices
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices
     
  Returns:
    true if we reached wanted relative precision, false otherwise
*/

template <class T>
bool solve_radiosity_equation_with_redistribution_Horvat_nbody(
  std::vector<Tview_factor_nbody<T>> &Lmat,     // input
  std::vector<std::vector<Tsparse_mat_elem<T>>> &Dmat,      
  std::vector<std::vector<T>> &R,  
  std::vector<std::vector<T>> &F0, 
  std::vector<std::vector<T>> &F1,                      // output
  std::vector<std::vector<T>> &Fout,                 
  const T & epsF = 1e-12,                               // optional 
  const T & max_iter = 100){

  // number of bodies
  int i, b, nb = F0.size();
  
  // lengths of sub-vectors
  std::vector<int> N(nb);
  for (b = 0; b < nb; ++b) N[b] = F0[b].size();

  std::vector<std::vector<T>> S0(nb), M0(nb), M1(nb), M2(nb);
  
  for (b = 0; b < nb; ++b) {
    S0[b].resize(N[b], T(0));
    M0[b].resize(N[b], T(0));
    M1[b].resize(N[b], T(0));
    M2[b].resize(N[b], T(0));
  } 

  //
  //  S0 = L_{LD} F0
  //

  for (auto && f: Lmat) S0[f.b1][f.i1] += f.F*F0[f.b2][f.i2];
  
  //
  // Iteration: 
  //   Fin_{k+1} = S0 + [L_{LD} D (1- diag(R)) + L0 diag(R)] Fin_k
  //
   
  int iter = 0;
  
  T t, dt, dF, Fmax;
  
  std::vector<std::vector<T>> Fin = S0;
  
  do {
    
    // M0 =  L0 diag(R) Fin
    for (b = 0; b < nb; ++b) memset(M0[b].data(), 0, sizeof(T)*N[b]);
    for (auto && f: Lmat) 
      M0[f.b1][f.i1] += f.F0*R[f.b2][f.i2]*Fin[f.b2][f.i2];

    // M1 = D (1- diag(R))Fin
    for (b = 0; b < nb; ++b) {
      memset(M1[b].data(), 0, sizeof(T)*N[b]);
      for (auto && f : Dmat[b]) 
        M1[b][f.i] += f.value*(1 - R[b][f.j])*Fin[b][f.j];
    }
    
    // M2 = L_{LD} M1
    for (b = 0; b < nb; ++b) memset(M2[b].data(), 0, sizeof(T)*N[b]);
    for (auto && f: Lmat) M2[f.b1][f.i1] += f.F*M1[f.b2][f.i2];
     
    // Fin = S0 + M0 + M2
    Fmax = dF = 0;
    for (b = 0; b < nb; ++b)
      for (i = 0; i < N[b]; ++i) {
        t = S0[b][i] + M0[b][i] + M2[b][i];
      
        if (t > Fmax) Fmax = t;
      
        dt = std::abs(Fin[b][i] - t);
      
        if (dt > dF) dF = dt;
      
        Fin[b][i] = t;
      }
    
  } while (dF >= Fmax*epsF && ++iter < max_iter);

  // F1 = F0 + M1   M1 = D(1-diag(T)) Fin
  F1 = F0;
  for (b = 0; b < nb; ++b)
    for (i = 0; i < N[b]; ++i) 
      F1[b][i] += M1[b][i];
  
  // Fout = F1 + diag(R) Fin
  Fout = F1;
  for (b = 0; b < nb; ++b)
    for (i = 0; i < N[b]; ++i) 
      Fout[b][i] += R[b][i]*Fin[b][i];
  
  return iter < max_iter;  
}

/*
  Calculating the sum of sparse matrices:
    B = sum_i  w_i A_i
  
  Input:
    w - weights
    A - array of sparse matrices
  
  Output:
    B - sparse matrix
*/
template <class T>
void add_sparse_matrices(
  std::vector<T> &w,
  std::vector<std::vector<Tsparse_mat_elem<T>>> &A,
  std::vector<Tsparse_mat_elem<T>> &B
){
  int n = w.size();
  
  if (A.size() != n) {
    std::cerr << "add_sparse_matrices::Sizes do not match.";
    return;
  }
  
  typedef std::pair<int, int> Tpair;
  
  std::map<Tpair, T> C;    
  
  T t;
  for (int i = 0; i < n; ++i) {
    t = w[i];
    for (auto && a : A[i]) C[Tpair(a.i, a.j)] += t*a.value; 
  }
  
  B.clear();
  
  for (auto && c : C) 
    B.emplace_back(c.first.first, c.first.second, c.second);
}

/*
  Calculating the sum of sparse matrices:
    B = sum_i  w_i A_i
  
  Input:
    w - map of weights
    A - map of sparse matrices
  
  Output:
    B - sparse matrix
*/
template <class T>
void add_sparse_matrices(
  std::map<fnv1a_32::hash_t, T> &W,
  std::map<fnv1a_32::hash_t, std::vector<Tsparse_mat_elem<T>> > &A,
  std::vector<Tsparse_mat_elem<T>> &B
){
  
  typedef std::pair<int, int> Tpair;
  
  std::map<Tpair, T> C;    
  
  T t;  
  for (auto && w : W) {
    t = w.second;
    for (auto && a : A[w.first]) C[Tpair(a.i, a.j)] += t*a.value; 
  }

  B.clear();
  
  for (auto && c : C) 
    B.emplace_back(c.first.first, c.first.second, c.second);
}

/*
  Calculating the sum of sparse matrices:
    B = sum_i  w_i A_i
  
  Input:
    d - dimension
    w - weights
    A - array of sparse matrices
  
  Output:
    B - sparse matrix
*/
template <class T>
void add_sparse_matrices(
  const int &d,
  std::map<fnv1a_32::hash_t, T> &W,
  std::map<fnv1a_32::hash_t, std::vector<Tsparse_mat_elem<T>> > &A,
  std::vector<Tsparse_mat_elem<T>> &B
){
  
  // number of distribution matrices different then none and non-zero
  int n = 0;
  
  auto h_none = "none"_hash32;
  
  for (auto && w : W) 
    if (w.first != h_none && w.second != 0) 
      ++n; 

  if (n > 1) {
    // there are several non-zero matrices
    // so it is wise to reserve dxd matrix to
    // speed up the calculations
    
    T **R = utils::matrix <T>(d, d);
    
    memset(R[0], 0, d*d*sizeof(T));    
    
    int size = 0;
    
    for (auto && w : W) {
      T t = w.second;
      if (t != 0) {
        auto & mat = A[w.first];
        for (auto && m : mat) {
          T & r = R[m.i][m.j];
          if (r == 0){
            ++size;
            r = t*m.value;
          } else r += t*m.value; 
        }
      }
    }

    B.clear();
    B.reserve(size);
    
    int i, j;
    
    T *r = R[0];
    
    for (i = 0; i < d; ++i)
      for (j = 0; j < d; ++j, ++r)
        if (*r) B.emplace_back(i, j, *r);
    
    utils::free_matrix(R);
    
  } else {
    // just one non-zero and non-none element
    for (auto && w : W) 
      if (w.first != h_none && w.second != 0) {
        B = A[w.first];
        T t = w.second;
        for (auto & b : B) b.value *= t;
        break;
      }
  }
}


#endif // #if !defined(__redistribution_h)
