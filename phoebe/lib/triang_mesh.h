#pragma once

/*
  Library for calculating properties and deformations of triangular meshes stored in the face-vertex format.

  Ref:
    * https://en.wikipedia.org/wiki/Surface_triangulation

  Author: August 2016
*/

#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <limits>

#include "utils.h"

/*
  Structure describing 3D point
*/
template <class T>
struct T3Dpoint {

  T data[3];     // normal vector of triangle

  T3Dpoint() {}
  
  T3Dpoint(const T & val):data{val, val, val} {}
  
  T3Dpoint(const T3Dpoint & v) : data{v.data[0],v.data[1],v.data[2]} { }

  T3Dpoint(const T &x1, const T &x2, const T &x3) : data{x1, x2, x3} {}

  T3Dpoint(T *p) : data{p[0], p[1], p[2]} {}

  T & operator[](const int &idx) { return data[idx]; }

  const T & operator[](const int &idx) const { return data[idx]; }

  T3Dpoint & operator*=(const T &fac) {
    data[0] *= fac;
    data[1] *= fac;
    data[2] *= fac;
    return *this;
  }

  T* operator & () const { return data; }

  void fill(const T & val){
    data[0] = data[1] = data[2] = val;
  }

  void assign(const T & x1, const T & x2, const T & x3) {
    data[0] = x1;
    data[1] = x2;
    data[2] = x3;
  }
};

/*
  Overloading printing of Tt3Dpoint
*/
template<class T>
std::ostream& operator<<(std::ostream& os, const T3Dpoint<T> & v)
{

  os << v[0] << ' ' << v[1] << ' ' << v[2];

  return os;
}

/*
  Calculate the area of a triangle defined by the three vertices AND
  optionally normal to the surface.

  Input:
    v1, v2, v3 - vertices of the triangle
    n - normal at of the vertices

  Output (optionally):
    n - normal to the surface in the same direction input n

  Return:
    area of the triangle

*/
template <class T>
T triangle_area (T v1[3], T v2[3], T v3[3], T *n = 0) {

  T a[3], b[3];

  // a = v2 - v1
  // b = v3 - v1
  for (int i = 0; i < 3; ++i) {
    a[i] = v2[i] - v1[i];
    b[i] = v3[i] - v1[i];
  }

  // Cross[{a[0], a[1], a[2]}, {b[0], b[1], b[2]}]
  // {-a[2] b[1] + a[1] b[2], a[2] b[0] - a[0] b[2], -a[1] b[0] + a[0] b[1]}

  T c[3] = {
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    },
    norm = utils::hypot3(c[0], c[1], c[2]); // std::hypot(,,) is comming in C++17

  if (n) {

    // copy c -> n and calculate scalar product c.n1
    T scalar = 0;
    for (int i = 0; i < 3; ++i) scalar += n[i]*c[i];

    // based on the scalar prodyct normalize normal n
    T fac = (scalar > 0 ? 1/norm : -1/norm);
    for (int i = 0; i < 3; ++i) n[i] = fac*c[i];
  }

  return norm/2;
}

/*
  Calculate the area of a triangle defined by the three vertices AND
  optionally normal to the surface.

  Input:
    r[3][3]- vertices of the triangle (r[0], r[1], r[2])
    n[3] - estimated normal vector (optional)

  Output (optional):
    n - normal to the surface in the same direction as input n

  Return:
    area of the triangle
*/
template <class T>
T triangle_area (T r[3][3], T *n = 0) {

  T a[3], b[3];

  // a = v2 - v1
  // b = v3 - v1
  for (int i = 0; i < 3; ++i) {
    a[i] = r[1][i] - r[0][i];
    b[i] = r[2][i] - r[0][i];
  }

  // Cross[{a[0], a[1], a[2]}, {b[0], b[1], b[2]}]
  // {-a[2] b[1] + a[1] b[2], a[2] b[0] - a[0] b[2], -a[1] b[0] + a[0] b[1]}

  T c[3] = {
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    },
    norm = utils::hypot3(c[0], c[1], c[2]); // std::hypot(,,) is comming in C++17

  if (n) {

    // copy c -> n and calculate scalar product c.n1
    T scalar = 0;
    for (int i = 0; i < 3; ++i) scalar += c[i]*n[i];

    // based on the scalar prodyct normalize normal n
    T fac = (scalar > 0 ? 1/norm : -1/norm);
    for (int i = 0; i < 3; ++i) n[i] = fac*c[i];
  }

  return norm/2;
}


/*
  Define an orthonormal basis {t1,t2,n} based on g:

    n proportional to g

    t1, t2 -- orthogonal to g

  Input:
    n[3] - normal vector on the surface
    norm (optional): if true normalize n

  Output:
    b[3][3] = {t1,t2, n}
*/

template <class T>
void create_basis(T g[3], T b[3][3], const bool & norm = false){

  //
  // Define screen coordinate system: b[3]= {t1,t2,view}
  //
  T *t1 = b[0], *t2 = b[1], *n = b[2], fac;

  if (norm){
    fac = 1/utils::hypot3(g[0], g[1], g[2]);
    for (int i = 0; i < 3; ++i) n[i] = fac*g[i];
  } else
    for (int i = 0; i < 3; ++i) n[i] = g[i];

  //
  // creating base in the tangent plane
  //

  // defining vector t1
  if (std::abs(n[0]) >= 0.5 || std::abs(n[1]) >= 0.5){
    //fac = 1/std::sqrt(n[1]*n[1] + n[0]*n[0]);
    fac = 1/std::hypot(n[0], n[1]);
    t1[0] = fac*n[1];
    t1[1] = -fac*n[0];
    t1[2] = 0.0;
  } else {
    //fac = 1/std::sqrt(n[0]*n[0] + n[2]*n[2]);
    fac = 1/std::hypot(n[0], n[2]);
    t1[0] = -fac*n[2];
    t1[1] = 0.0;
    t1[2] = fac*n[0];
  }

  // t2 = n x t1
  t2[0] = n[1]*t1[2] - n[2]*t1[1];
  t2[1] = n[2]*t1[0] - n[0]*t1[2];
  t2[2] = n[0]*t1[1] - n[1]*t1[0];
}

/*
  Transform coordinates of a vector r from standard cartesian basis

    r = sum_i u_i e_i   e_1= (1,0,0), e_2=(0,1,0), e_3=(0,0,1)

  into new orthonormal basis {b_1. b_2, b3}

    r = sum_i v_i b_i

  Input:
    u[3] -- coordinate of r in cartesian basis
    b[3][3] = {b_1, b_2, b_3} -- coordinates of the basis

  Output:
    v[3] -- coordinate of r in  basis b

*/

template <class T>
void trans_basis(T u[3], T v[3], T b[3][3]){

  int i, j;

  if (u != v) {

    for (i = 0; i < 3; ++i) {
      T s = 0;
      for (j = 0; j < 3; ++j) s += b[i][j]*u[j];
      v[i] = s;
    }

  } else {

    T s[3] = {0, 0, 0};

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) s[i] += b[i][j]*u[j];

    for (i = 0; i < 3; ++i) v[i] = s[i];
  }
}


/*
  Calculate area of the triangulated of surfaces and volume of the body
  that the surface envelopes.

  Input:
    V - vector of vertices
    NatV - vector of normals at vertices
    Tr - vector of triangles
    choice - index of vertex as reference normal

  Output:
    av[2] = {area, volume}

  Ref:
  * Cha Zhang and Tsuhan Chen, Efficient feature extraction for 2d/3d
    objects in mesh representation, Image Processing, 2001.

*/
template <class T>
void mesh_area_volume(
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<T>> & NatV,
  std::vector <T3Dpoint<int>> & Tr,
  T av[2],
  int choice = 0) {

  int i, j;

  T sumA = 0, sumV = 0,
    a[3], b[3], c[3], v[3][3],
    *pv, *pn, f1, f2;

  for (auto && t: Tr) {

    //
    // copy data
    //
    for (i = 0; i < 3; ++i) {
      pv = V[t[i]].data;
      for (j = 0; j < 3; ++j) v[i][j] = pv[j];
    }

    //
    // Computing surface element
    //

    for (i = 0; i < 3; ++i) {
      a[i] = v[1][i] - v[0][i];
      b[i] = v[2][i] - v[0][i];
    }

    // Cross[{a[0], a[1], a[2]}, {b[0], b[1], b[2]}]
    // {-a[2] b[1] + a[1] b[2], a[2] b[0] - a[0] b[2], -a[1] b[0] + a[0] b[1]}

    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];

    sumA += utils::hypot3(c[0], c[1], c[2]); // std::hypot(,,) is comming in C++17
    // sumA += std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2])/2;

    //
    // Computing volume of signed tetrahedron
    //

    // determine the sign of the normal

    pn = NatV[t[choice]].data;
    pv = v[choice];

    f1 = f2 = 0;
    for (i = 0; i < 3; ++i) {
      f1 += pn[i]*c[i];
      f2 += pv[i]*c[i];
    }

    // volume of the signed tetrahedron
    if (f1 != 0) {

      f2 = std::abs(
        (-v[2][0]*v[1][1] + v[1][0]*v[2][1])*v[0][2] +
        (+v[2][0]*v[0][1] - v[0][0]*v[2][1])*v[1][2] +
        (-v[1][0]*v[0][1] + v[0][0]*v[1][1])*v[2][2]
      );

      if (f1 > 0) sumV += f2; else sumV -= f2;
    }
  }

  av[0] = sumA/2;
  av[1] = sumV/6;
}


/*
  Calculate area of the triangulated of surfaces.

  Input:
    V - vector of vertices
    Tr - vector of triangles
    choice - index of vertex as reference normal

  Output:
    av[2] = {area, volume}

  Ref:
  * Cha Zhang and Tsuhan Chen, Efficient feature extraction for 2d/3d
    objects in mesh representation, Image Processing, 2001.

*/
template <class T>
T mesh_area(
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<int>> & Tr) {

  T a[3], b[3], c[3], *v[3];

  long double sumA = 0;

  for (auto && t: Tr) {

    //
    // link data data
    //

    for (int i = 0; i < 3; ++i) v[i] = V[t[i]].data;

    //
    // Computing surface element
    //

    for (int i = 0; i < 3; ++i) {
      a[i] = v[1][i] - v[0][i];
      b[i] = v[2][i] - v[0][i];
    }

    // Cross[{a[0], a[1], a[2]}, {b[0], b[1], b[2]}]
    // {-a[2] b[1] + a[1] b[2], a[2] b[0] - a[0] b[2], -a[1] b[0] + a[0] b[1]}

    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];

    sumA += utils::hypot3(c[0], c[1], c[2]); // std::hypot(,,) is comming in C++17
  }

  return sumA/2;
}

/*
  Calculate properties of triangles -- areas of triangles and a normal

  Input:
    V - vector of vertices
    NatV - vector of normals at vertices
    Tr - vector of triangles
    choice in {0,1,2} - which vertex is choosen as the reference
    reorientate - if Tr should be reorientated so that normal point outwards

  Output:
    A - triangle areas
    N - triangle normals
    area - total area of the triangles
    volume - volume of the body enclosed by the mesh

  Ref:
  * Cha Zhang and Tsuhan Chen, Efficient feature extraction for 2d/3d
    objects in mesh representation, Image Processing, 2001.
*/
template<class T>
void mesh_attributes(
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<T>> & NatV,
  std::vector <T3Dpoint<int>> & Tr,
  std::vector <T> *A = 0,
  std::vector <T3Dpoint<T>> * N = 0,
  T *area = 0,
  T *volume = 0,
  int choice = 0,
  bool reorientate = false
) {

  if (A == 0 && area == 0 && volume == 0 && N == 0) return;


  if (A) {
    A->clear();
    A->reserve(Tr.size());
  }

  if (N) {
    N->clear();
    N->reserve(Tr.size());
  }

  if (area)  *area = 0;
  if (volume) *volume = 0;

  int i, j;

  T a[3], b[3], c[3], v[3][3], *p, f, fv, norm;

  bool
    st_area = (area != 0) || (A != 0),
    st_N = (N != 0),
    st_area_N = st_area || st_N,
    st_volume = (volume != 0),
    st_N_volume = st_N || st_volume || reorientate;

  for (auto && t: Tr) {

    //
    // copy data
    //

    for (i = 0; i < 3; ++i) {
      p = V[t[i]].data;
      for (j = 0; j < 3; ++j) v[i][j] = p[j];
    }

    //
    // Computing surface element
    //

    for (i = 0; i < 3; ++i) {
      a[i] = v[1][i] - v[0][i]; // v1-v0
      b[i] = v[2][i] - v[0][i]; // v2-v0
    }

    // Cross[{a[0], a[1], a[2]}, {b[0], b[1], b[2]}]
    // {-a[2] b[1] + a[1] b[2], a[2] b[0] - a[0] b[2], -a[1] b[0] + a[0] b[1]}

    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];

    if (st_area_N) {

      norm = utils::hypot3(c[0], c[1], c[2]); // std::hypot(,,) is comming in C++17

      f  = norm/2;

      if (A) A->emplace_back(f);
      if (area) *area += f;
    }

    if (st_N_volume) {

      //
      // Compute normal to the surface element
      //

      // orienting normal vector along the normal at vertex t[choice]
      p = NatV[t[choice]].data,

      f = 0;
      for (i = 0; i < 3; ++i) f += p[i]*c[i];

      if (f < 0) {
        for (i = 0; i < 3; ++i) c[i] = -c[i];
        // change the order of indices
        if (reorientate) {i = t[1]; t[1] = t[2]; t[2] = i; }
      }

      // normalize the normal
      if (st_N) {
        f = 1/norm;
        for (i = 0; i < 3; ++i) c[i] *= f;
        N->emplace_back(c);
      }

      //
      // Computing volume of signed tetrahedron
      //

      if (st_volume) {

        // determine the sign tetrahedron
        p = v[0];
        f = 0;
        for (i = 0; i < 3; ++i) f += p[i]*c[i];

        if (f != 0) {
          // volume of tetrahedron
          fv = std::abs(
            (-v[2][0]*v[1][1] + v[1][0]*v[2][1])*v[0][2] +
            (+v[2][0]*v[0][1] - v[0][0]*v[2][1])*v[1][2] +
            (-v[1][0]*v[0][1] + v[0][0]*v[1][1])*v[2][2]
          )/6;

          if (f > 0) *volume += fv; else  *volume -= fv;
        }
      }
    }
  }
}


/*
  Calculate properties of triangles with assumption that normals taken in
  given order point triangles outwards

  Input:
    V - vector of vertices
    Tr - vector of triangles

  Output:
    A - triangle areas
    N - triangle normals
    area - total area of the triangles
    volume - volume of the body enclosed by the mesh

  Ref:
  * Cha Zhang and Tsuhan Chen, Efficient feature extraction for 2d/3d
    objects in mesh representation, Image Processing, 2001.
*/
template<class T>
void mesh_attributes(
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<int>> & Tr,
  std::vector <T> *A = 0,
  std::vector <T3Dpoint<T>> * N = 0,
  T *area = 0,
  T *volume = 0
) {

  if (A == 0 && area == 0 && volume == 0 && N == 0) return;


  if (A) {
    A->clear();
    A->reserve(Tr.size());
  }

  if (N) {
    N->clear();
    N->reserve(Tr.size());
  }

  if (area)  *area = 0;
  if (volume) *volume = 0;

  int i, j;

  T a[3], b[3], c[3], v[3][3], *p, f, fv, norm;

  bool
    st_area = (area != 0) || (A != 0),
    st_N = (N != 0),
    st_area_N = st_area || st_N,
    st_volume = (volume != 0),
    st_N_volume = st_N || st_volume;

  for (auto && t: Tr) {

    //
    // copy data
    //

    for (i = 0; i < 3; ++i) {
      p = V[t[i]].data;
      for (j = 0; j < 3; ++j) v[i][j] = p[j];
    }

    //
    // Computing surface element
    //

    for (i = 0; i < 3; ++i) {
      a[i] = v[1][i] - v[0][i]; // v1-v0
      b[i] = v[2][i] - v[0][i]; // v2-v0
    }

    // Cross[{a[0], a[1], a[2]}, {b[0], b[1], b[2]}]
    // {-a[2] b[1] + a[1] b[2], a[2] b[0] - a[0] b[2], -a[1] b[0] + a[0] b[1]}

    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];

    if (st_area_N) {

      // std::hypot(,,) is comming in C++17
      norm = utils::hypot3(c[0], c[1], c[2]);

      f  = norm/2;

      if (A) A->emplace_back(f);
      if (area) *area += f;
    }

    if (st_N_volume) {

      //
      // Compute normal to the surface element
      //

      // normalize the normal
      if (st_N) {
        f = 1/norm;
        for (i = 0; i < 3; ++i) c[i] *= f;
        N->emplace_back(c);
      }

      //
      // Computing volume of signed tetrahedron
      //

      if (st_volume) {

        // determine the sign tetrahedron
        p = v[0];
        f = 0;
        for (i = 0; i < 3; ++i) f += p[i]*c[i];

        if (f != 0) {
          // volume of tetrahedron
          fv = std::abs(
            (-v[2][0]*v[1][1] + v[1][0]*v[2][1])*v[0][2] +
            (+v[2][0]*v[0][1] - v[0][0]*v[2][1])*v[1][2] +
            (-v[1][0]*v[0][1] + v[0][0]*v[1][1])*v[2][2]
          )/6;

          if (f > 0) *volume += fv; else  *volume -= fv;
        }
      }
    }
  }
}

/*
  Offsetting the mesh to match the reference area by moving vertices along the normals in vertices so that the total area matches its reference value.

  Currently supporting only curvature independent.

  Input:
    A0 - reference area
    V - vector of vertices
    NatV - vector of normals
    Tr - vector of triangles
    max_iter - maximal number of iterator
  Output:
    Vnew - vector of new vertices

  Return:
    false - If somethings fails

*/
template <class T>
bool mesh_offseting_matching_area(
  const T &A0,
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<T>> & NatV,
  std::vector <T3Dpoint<int>> & Tr,
  const int max_iter = 100) {

  const T eps = 10*std::numeric_limits<T>::epsilon();

  int it = 0, Nv = V.size();

  T A[2], dt = 1e-12;

  A[0] = mesh_area(V, Tr);

  do {

    // shift of the vertices
    for (int i = 0; i < Nv; ++i)
      for (int j = 0; j < 3; ++j)
        V[i][j] += dt*NatV[i][j];

    // calculating area
    A[1] = A[0];
    A[0] = mesh_area(V, Tr);


    // secant step
    dt *= (A0 - A[0])/(A[0] - A[1]);

    /*
    std::cerr.precision(16);
    std::cerr <<std::scientific;
    std::cerr << dt << '\t' << A0 << '\t' << A[0] << '\t' << A[1] << '\n';
    */

  } while (std::abs(1 - A[0]/A0) > eps && ++it < max_iter);

  return it < max_iter;
}

/*
  Offseting the mesh to match the reference area by moving vertices
  along the normals in vertices so that the total area matches its
  reference value. This version takes into account the local
  curvature in more accurately predicting the shift of points.

  Method for estimating curvature tensor:

  Taubin G,
  Estimating the tensor of curvature of a surface from a polyhedral approximation,
  Conference: Computer Vision, 1995

  Input:
    A0 - reference area
    V - vector of vertices
    NatV - vector of normals
    Tr - vector of triangles
    max_iter - maximal number of iterator

  Output:
    Vnew - vector of new vertices

  Return:
    false - If somethings fails

*/
template <class T>
bool mesh_offseting_matching_area_curvature(
  const T &A0,
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<T>> & NatV,
  std::vector <T3Dpoint<int>> & Tr,
  const int max_iter = 100) {

  const T eps = 10*std::numeric_limits<T>::epsilon();

  //
  // Building polygons around the vertex ~ Cp
  //

  int Nv = V.size();

  std::vector<std::vector<int>> Cp(Nv);

  {

    //
    // Create connection vertex -> list of triangles
    //
    std::vector<std::vector<int>> Ct(Nv);

    int i = 0;
    for (auto && t : Tr) {

      // check how t[j]-th vertex is connected
      for (int j = 0; j < 3; ++j) {
        auto & r = Ct[t[j]];  // reference to row of Ct

        // check if i-th triangle is already included
        bool ok = true;
        for (auto && k : r) if (i == k) {
          ok = false;
          break;
        }

        // add i-th triangle as connected to t[j]-th vertex
        if (ok) r.push_back(i);
      }

      ++i;
    }

    //
    // Transform (vertex, list of triangles) -> (vertex, polygon of vertices)
    //
    {
      int i = 0, n, *pairs, *pe, last;

      auto
        it_poly = Cp.begin(), ite_poly = Cp.end(),
        it_triangle = Ct.begin();

      while (it_poly != ite_poly) {

        // number of triangles
        n = it_triangle->size();

        // gather pairs of connected vertices,
        // get other two points ! = i
        pairs = new int [2*n];
        pe = pairs + 2*n;

        {
          int *p = pairs, *t;
          for (auto && k: *it_triangle) {
            t = Tr[k].data;
            for (int j = 0; j < 3; ++j) if (t[j] != i) *(p++) = t[j];
          }
        }

        // connect the pairs of indices by resorting pairs
        // and gather points for closed polygon
        it_poly->reserve(n);

        {
          for (int *p = pairs; p != pe; p += 2) {

            it_poly->push_back(*p);
            last = p[1];

            for (int *q = p + 2, *pb = q; q != pe; q += 2)
              if (last == q[0]){
                utils::swap_array<int,2>(pb, q);
                break;
              } else if (last == q[1]) {
                std::swap(q[0], q[1]);
                utils::swap_array<int,2>(pb, q);
                break;
              }
          }
        }

        delete [] pairs;

        ++i, ++it_triangle, ++it_poly;
      }
    }
  }

  //
  // Calculating weights to shift at vertices
  //

  std::vector<T> W(Nv);
  {
    for (int i = 0; i < Nv; ++i){

      // polygon around the vertices
      auto & poly = Cp[i];
      auto itb = poly.begin(), ite = poly.end();
      int n = poly.size();

      // central vertex
      auto *v = V[i].data;

      T at = 0,          // total area
        *w = new T [n];  // starting as areas of triagles in polygon

      {
        T *p = w;
        for (auto it = itb, itp = ite - 1; it != ite; itp = it++)
          at += (*(p++) = triangle_area(v, V[*it].data, V[*itp].data));
      }

      // calculate triangles incident to (i, poly[j]) and using them
      // to define the avaraging weights stored in w[]
      {
        T t = w[0];
        for (int j = 0; j < n - 1; ++j) w[j] += w[j + 1];
        w[n - 1] += t;

        t = 1/(2*at);
        for (int j = 0; j < n; ++j) w[j] *= t;
      }

      // calculate average curvature as a trace of the Taubin's tensor M

      T h = 0;

      {
        T *p = w, *n = NatV[i].data, q[3];

        for (auto it = itb; it != ite; ++p, ++it) {

          // q = v - u
          utils::sub3D(v, V[*it].data, q);

          // estimate of directional curvature k = 2*(n.q)/ |q|^2
          // and perform the trace
          h += 2*(*p)*utils::dot3D(n, q)/utils::norm2(q);
        }
      }

      W[i] = at*h;

      delete [] w;
    }
  }

  //for ( auto && w : W) std::cerr << w << '\n';

  //
  // Match the mesh area to the reference area
  //

  int it = 0;

  T dt1, A[2], dt = 1e-3;

  A[0] = mesh_area(V, Tr);

  do {

    // shift of the vertices
    for (int i = 0; i < Nv; ++i) {
      dt1 = W[i]*dt;
      for (int j = 0; j < 3; ++j) V[i][j] += dt1*NatV[i][j];
    }

    // calculating area
    A[1] = A[0];
    A[0] = mesh_area(V, Tr);


    // secant step
    dt *= (A0 - A[0])/(A[0] - A[1]);

    /*
    std::cerr.precision(16);
    std::cerr <<std::scientific;
    std::cerr << dt << '\t' << A0 << '\t' << A[0] << '\t' << A[1] << '\n';
    */

  } while (std::abs(1 - A[0]/A0) > eps && ++it < max_iter);

  return it < max_iter;
}
