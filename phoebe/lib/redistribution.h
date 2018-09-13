#pragma once

/*
  Library discussing the process of reflection with (quasi) stationary
  redistribution of the incoming flux over the surface of the body.

  Author: Martin Horvat, April 2017
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
  Class for handling redistribution in an efficient and a compact form.

  We decompose the redistribution matrix as

  R = w_{global} 1.p^T + w_{horiz} D_{horiz} + w_{local} D_{local} + ...
      -----------------  -----------------------------------------------
      D_{proj}              D_{sparse}
*/

enum Tsupport_type { triangles, vertices };

template <class T>
class Tredistribution{

  //
  //  Redistribution supports two type of contributions
  //
  bool trivial_redistr;                          // check if redistribution is enabled
  std::vector<T> p;                              // ~ D_{proj} = [1..1].p^T
  std::vector<std::vector<std::pair<int,T>>> S;  // ~ D_{sparse}


  //
  // adding diagonal to a temp. redistribution matrix
  //
  void add_diag_redistr_matrix(
    const T & weight,                                       // input
    int N,
    std::vector<std::vector<std::pair<int,T>>> & TS         // output
  ) {
    if (TS.size() == 0) TS.resize(N);
    for (int i = 0; i < N; ++i) TS[i].emplace_back(i, weight);
  }

  //
  // adding from conn. matrix calculated matric to tmp redistr. matrix
  //
  void add_redistr_matrix(
    const T & weight,                                       // input
    std::vector<std::vector<std::pair<int,T>>> &C,
    std::vector<T> &A,
    std::vector<std::vector<std::pair<int,T>>> & TS         // output
  ) {

    if (TS.size() == 0) TS.resize(A.size());

    int j = 0;

    auto it = A.begin();

    for (auto && c : C) {

      long double f = 0;
      for (auto && q : c) f += A[q.first]*q.second;
      f = weight*(*it)/f;

      for (auto && q : c) TS[q.first].emplace_back(j, q.second*f);

      ++j;
      ++it;
    }
  }

  //
  // Calculating weighted connectivity matrix for local redistribution
  //
  template <class F>
  void calc_local_connectivity(
    const T & thresh,                                       // input
    std::vector<T3Dpoint<T>> & P,
    std::vector<std::vector<std::pair<int,T>>> & C          // output
  ){

    int i, j, N = P.size();

    C.clear();

    C.resize(N);

    T t, t0 = F()(0.0, thresh), *v1, *v2, tmp;

    for (i = 0; i < N; ++i) {

      v1 = P[i].data;

      for (j = 0; j < i; ++j) {

        v2 = P[j].data;

        tmp = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

        t = F()(std::acos(tmp), thresh);

        if (t) {
          C[i].emplace_back(j, t);
          C[j].emplace_back(i, t);
        }
      }

      C[i].emplace_back(i, t0);
    }
  }

  //
  // Calculating weighted connectivity matrix for horizontal redistribution
  //
  template <class F>
  void calc_horiz_connectivity(
    const T & thresh,                                     // input
    T o[3],
    std::vector<T3Dpoint<T>> & P,
    std::vector<std::vector<std::pair<int,T>>> & C        // output
  ) {

    int i, j, N = P.size();

    C.clear();

    C.resize(N);

    T t, t0 = F()(0.0, thresh), *v1, *v2, c1, c2, tmp;

    for (i = 0; i < N; ++i) {

      v1 = P[i].data;
      c1 = v1[0]*o[0] + v1[1]*o[1] + v1[2]*o[2];

      for (j = 0; j < i; ++j) {

        v2 = P[j].data;
        c2 = v2[0]*o[0] + v2[1]*o[1] + v2[2]*o[2];

        tmp = std::sqrt((1 - c1*c1)*(1 - c2*c2)) + c1*c2;

        t = F()(std::acos(tmp), thresh);

        if (t) {
          C[i].emplace_back(j, t);
          C[j].emplace_back(i, t);
        }
      }

      C[i].emplace_back(i, t0);
    }
  }

  //
  // Calculate directions on sphere for all casese
  //
  bool calc_projection_to_sphere(
    const Tsupport_type & type,           // input
    std::vector<T3Dpoint<T>>  & V,
    std::vector<T3Dpoint<int>> & Tr,
    std::vector<T3Dpoint<T>> & P          // output
  ) {

    if (type == triangles)
      calc_barycenters(V, Tr, P);       // calc. of barycenters
    else
      P = V;

    T x[3], r;

    // fitting sphere to barycenters
    if (!fit_sphere(P, r, x)) return false;

    // project characteristic points onto unit sphere
    // v -> (v - x)/|v - x|
    calc_directions(x, P);

    return true;
  }

  //
  // Calculate areas at vertices and return pointer to array of areas
  //
  std::vector<T> * calc_areas (
    const Tsupport_type & type,       // input
    int Nv,
    std::vector<T3Dpoint<int>> & Tr,
    std::vector<T> & A,
    std::vector<T> & AatV             // ouput
  ) {

    if (type == vertices) {
      calc_area_at_vertices(Nv, Tr, A, AatV);
      return &AatV;
    }

    return &A;
  }

  //
  // Add second part of elements with identical first part
  //
  void add_identical(
    std::vector<std::pair<int, T>> &out,
    std::vector<std::pair<int, T>> &in
  ){

    if (in.size() <= 1) {
      out = in;
      return;
    }

    auto it = in.begin(), ite = in.end();

    std::sort(it, ite);

    int ind = it ->first;

    T sum = it->second;

    while (++it != ite) {
      if (ind == it->first)
        sum += it-> second;
      else {
        out.emplace_back(ind, sum);
        ind = it -> first;
        sum = it -> second;
      }
    }
    out.emplace_back(ind, sum);
  }

  public:

  Tredistribution():trivial_redistr(true) {}

  void clear() {
    p.clear();
    S.clear();
    trivial_redistr = true;
  }

  bool is_trivial() const { return trivial_redistr;}

  /*
  Calculate redistribution matrices for a given body. Surface represents
  a single body.

  Input:
    type - type of surface support
    V - vectors of vertices
    Tr - vectors of triangles
    N - vectors of normals at triangles/ at vertices
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

  Note:
    * distances on the body area calculated by reprojecting mesh onto
      tightest fitted sphere
    * using sharp edge in definion of the neighborhood of width h
  */
  template <class F>
  bool init(
    const Tsupport_type & type,
    std::vector<T3Dpoint<T>>   & V,
    std::vector<T3Dpoint<int>> & Tr,
    std::vector<T3Dpoint<T>>   & N, // vertices: NatV, triangle: NatT
    std::vector<T> & A,
    std::map<fnv1a_32::hash_t, std::vector<T>> & Dpars,
    std::map<fnv1a_32::hash_t, T> & W
  ) {

    const char * fname = "Tredistr::init";

    //
    // Destroy if they were used
    //
    if (p.size()) p.clear();
    if (S.size()) S.clear();

    //
    // auxilary variables so that we don't compute twice
    //
    std::vector<T3Dpoint<T>> P;    // projections on sphere

    std::vector<T> AatV, *A1 = 0;  // areas per vertices

    std::vector<std::vector<std::pair<int,T>>>
      C,    // connectivity matrix
      TS;   // temp. sparse matrix

    //
    // number of elements
    //
    int Ne = (type == vertices ? V.size() : A.size());

    //
    // generate redistribution matrices
    //
    for (auto && w : W) if (w.second != 0) {

      auto & par = Dpars[w.first];

      switch (w.first) {

        //
        // global/uniform redistribution
        //
        case "global"_hash32: {

          if (A1 == 0) A1 = calc_areas (vertices, Ne, Tr, A, AatV);

          T fac = T();
          for (auto && a: *A1) fac += a;

          fac = w.second/fac;

          p = *A1;

          for (auto && e : p) e *= fac;

          break;
        }

        //
        // local redistribution
        //

        case "local"_hash32: {

          if (par.size() != 1) {
            std::cerr << fname << "::Params wrongs size\n";
            return false;
          }

          T h = par[0]; // neighborhood size

          if (A1 == 0) A1 = calc_areas (vertices, Ne, Tr, A, AatV);

          if (P.size() == 0 && !calc_projection_to_sphere(type, V, Tr, P)) {
            std::cerr << fname <<"::Projections to sphere failed\n";
            return false;
          }


          if (h == 0)   // h == 0
            add_diag_redistr_matrix(w.second, Ne, TS);
          else {        // h > 0
            // creating "connectivity" matrix
            calc_local_connectivity<F>(h, P, C);

            // creating re-distribution matrix from connectivity matrix
            add_redistr_matrix(w.second, C, *A1, TS);
          }

          break;
        }

        //
        // horizontal redistribution
        //

        case "horiz"_hash32: {

          if (par.size() != 4) {
            std::cerr << fname << "::Params wrongs size\n";
            return false;
          }

          T o[3] = {par[0], par[1], par[2]},   // axis
            h = par[3];                        // neighborhood size

          if (A1 == 0) A1 = calc_areas (vertices, Ne, Tr, A, AatV);

          if (P.size() == 0 && !calc_projection_to_sphere(type, V, Tr, P)) {
            std::cerr << fname << "::Projections to sphere failed\n";
            return false;
          }

          if (h == 0) {   // h == 0
            add_diag_redistr_matrix(w.second, Ne, TS);
          } else {        // h > 0
            // creating "connectivity" matrix
            calc_horiz_connectivity<F>(h, o, P, C);

            // creating re-distribution matrix from connectivity matrix
            add_redistr_matrix(w.second, C, *A1, TS);
          }
          break;
        }
      }
    }

    // storing redistribution matrix
    if (TS.size()) {
      S.resize(TS.size());
      auto is = S.begin(), it = TS.begin(), et = TS.end();
      while (it != et) add_identical(*(is++), *(it++));
    }

    trivial_redistr =  p.size() == 0 && S.size() == 0;

    return true;
  }

  // a <- a + R b
  void mul_add (std::vector<T> & a, std::vector<T> & b){

    if (p.size()) {
      T sum = T();
      auto ib = b.begin(), eb = b.end(), ip = p.begin();
      while (eb != ib) sum += (*(ib++))*(*(ip++));

      for (auto && f : a) f += sum;
    }

    if (S.size()) {
      auto it = a.begin();
      for (auto && s : S) {
        T sum = T();
        for (auto && e: s) sum += e.second*b[e.first];
        *(it++) += sum;
      }
    }
  }
};


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

    F1 = F0 + D(1-diag(R)) L_{LD} Fout

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
    Dmat - redistribution matrices

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
  Tredistribution<T> &D,
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

    // (1) M1 <- diag(R) M0
    // (2) M0 <- (id - diag(R)) M0
    for (int i = 0; i < N; ++i) {
      M1[i] = R[i]*M0[i];
      M0[i] -= M1[i];
    }

    // F1 = F0 + D M0
    F1 = F0;
    D.mul_add(F1, M0);

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
  Tredistribution<T> &D,
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
    memset(M0, 0, sizeof(T)*N);
    for (auto && f: Lmat) M0[f.i] += f.F0*R[f.j]*Fin[f.j];

    // M2 = (1- diag(R))Fin
    for (int i = 0; i < N; ++i) M2[i] = (1 - R[i])*Fin[i];

    // M1 = D M2
    memset(M1, 0, sizeof(T)*N);

    D.mul_add(M1, M2);

    // M2 = M0 + L_{LD} M1
    M2 = M0;
    for (auto && f: Lmat) M2[f.i] += f.F*M1[f.j];

    // Fin = S0 + M0 + M2
    Fmax = dF = 0;
    for (int i = 0; i < N; ++i) {
      t = S0[i] + M2[i];

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
    Dmat - redistribution matrices

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
  std::vector<Tview_factor_nbody<T>> &Lmat,             // input
  std::vector<Tredistribution<T>> &D,
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

    // (1) M1 <- diag(R) M0
    // (2) M0 <- (id - R) M0
    for (b = 0; b < nb; ++b)
      for (i = 0; i < N[b]; ++i) {
        M1[b][i] = R[b][i]*M0[b][i];
        M0[b][i] -= M1[b][i];
      }

    // F1 = F0 + D.M0
    F1 = F0;
    for (b = 0; b < nb; ++b) D[b].mul_add(F1[b], M0[b]);

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
  std::vector<Tredistribution<T>> &D,
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

    // M2 = (1- diag(R))Fin
    for (b = 0; b < nb; ++b)
      for (i = 0; i < N[b]; ++i)
        M2[b][i] = (1 - R[b][i])*Fin[b][i];

    // M1 = D M2
    for (b = 0; b < nb; ++b){
      memset(M1[b].data(), 0, sizeof(T)*N[b]);
      D[b].mul_add(M1[b], M2[b]);
    }

    // M2 = M0 + L_{LD} M1
    M2 = M0;
    for (auto && f: Lmat) M2[f.b1][f.i1] += f.F*M1[f.b2][f.i2];

    // Fin = S0 + M2
    Fmax = dF = 0;
    for (b = 0; b < nb; ++b)
      for (i = 0; i < N[b]; ++i) {
        t = S0[b][i] + M2[b][i];

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

