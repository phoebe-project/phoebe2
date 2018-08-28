#pragma once

/*
  Library for solving the stellar radiosity problem. Wilson called this reflection effects. Basically is problem of evaluating radiosity by taking into account intrinsic radiant exitance and reflection of it from surface of the stars.

  Author: Martin Horvat, July 2016

  Ref:

  Stars:
    * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect, Astrophysical Journal,  356, 613-622, 1990 June
    * Budaj J., The reflection effect in interacting binaries or in planet–star systems, The Astronomical Journal, 141:59 (12pp), 2011 February
    * Josef Kallrath, Eugene F. Milone (auth.)-Eclipsing Binary Stars - Modeling and Analysis (Springer-Verlag New York, 2009)

  Radiometry:
    * http://light-measurement.com/calculation-of-radiometric-quantities/
    * https://en.wikipedia.org/wiki/Radiometry
    * https://en.wikipedia.org/wiki/Radiosity_(radiometry)
*/


#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <cstring>

#include "utils.h"
#include "triang_mesh.h"
#include "ld_models.h"

/*
  Check if the line segment

    r = c1 + (c2 - c1)t      t in [0,1]

  cuts the triangles with vertices

    (v[0], v[1], v[2])
*/

template <class T>
bool triangle_cuts_line(T *v[3], T c1[3], T c2[3]){

  // define transpose matrix A^T = [r2-r1, r3-r1, c1 -c2]
  T A[3][3], b[3], x[3];

  for (int i = 0; i < 3; ++i){
    A[0][i] = v[1][i] - v[0][i];
    A[1][i] = v[2][i] - v[0][i];
    A[2][i] = c1[i] - c2[i];
    b[i] = c1[i] - v[0][i];
  }

  // solve system A^t x = b or x^t A = b^t
  if (!utils::solve3D(b, A, x)) {
    std::cerr << "triangle_cuts_line::This should not happen\n";
    return false;
  }

  // if 0<=x[i]<=1 then there is intersection
  for (int i = 0; i < 3; ++i)
    if (x[i] < 0 || x[i] > 1) return false;

  return (x[0] + x[1]<=1);
}

template <class T>
bool triangle_cuts_line(T n[3], T *v[3], T c1[3], T c2[3]){

  //
  // Is line cuts the plane in which the triangle lies
  //

  T f[3], g[3];

  for (int i = 0; i < 3; ++i) {
    f[i] = c1[i] - v[0][i];
    g[i] = c1[i] - c2[i];
  }

  T t3 = utils::dot3D(g,n);

  if (t3 == 0) return false;

  t3 = utils::dot3D(f,n)/t3;

  if (t3 < 0 || t3 > 1) return false;

  //
  // Check if the point of intesection is in the triangle
  //

  T a[3], b[3], c[3];

  for (int i = 0; i < 3; ++i) {
    a[i] = v[1][i] - v[0][i];
    b[i] = v[2][i] - v[0][i];
    c[i] = f[i] - t3*g[i];
  }

  // find largest/most pronounced components of the normal
  int l = 0;
  {
    T t, m = std::abs(n[0]);
    for (int i = 1; i < 3; ++i) if ((t = std::abs(n[i])) > m) {
      m = t;
      l = i;
    }
  }

  // define components that need to be taken into account
  int j, k;

  if (l == 0) {
    j = 1;
    k = 2;
  } else if (l == 1){
    j = 0;
    k = 2;
  } else { // l == 2
    j = 0;
    k = 1;
  }

  // solve equation:
  //  [ a[j] b[j] ] [t1] = [c[j]]
  //  [ a[k] b[k] ] [t2] = [c[k]]

  T det = 1/(a[j]*b[k] - a[k]*b[j]);

  T t1 = det*(b[k]*c[j] - b[j]*c[k]);
  if (t1 < 0 || t1 > 1) return false;

  T t2 = det*(a[j]*c[k] - a[k]*c[j]);
  if (t2 < 0 || t2 > 1) return false;

  return (t1 + t2 <= 1);
}

/*
  Matrix element of the sparse-matrix F_{i, j}
*/
template <class T>
struct Tview_factor {
  int i, j;
              // viewing factor F_{i <- j}
  T F0,       // Lambert viewing factor
    F;        // Limb darkend viewing factor

  Tview_factor() {}
  Tview_factor(int i, int j, const T &F0, const T &F) : i(i), j(j), F0(F0), F(F) {}
};

/*
  Calculating limb-darkened radiosity/view factor matrices with elements
  defined per TRIANGLE.

  Input:

    V - vector of vertices
    Tr - vector of triangles
    NatT - vector of normals at triangles
    A - vector of areas of triangles
    LDmodels - vector of limb darkening models in use
    LDidx - vector of indices of models used on each of triangles

    epsC - threshold for permitted cos(theta)
              cos(theta_i) > epsC to be considered in line-of-sight
           ideally epsC = 0, epsC=0.00872654 corresponds to 89.5deg
  Output:
    Fmat - matrix of LD view factors
            F0 - Lambert view-factor
            F - limb-darkened view-factor
              - cos(theta') if LDmodels[i] == 0, evaluation of LD is
                postponed to routines outside the routine
*/

template <class T>
void triangle_mesh_radiosity_matrix_triangles(
  std::vector <T3Dpoint<T>> & V,                  // inputs
  std::vector <T3Dpoint<int>> & Tr,
  std::vector <T3Dpoint<T>> & NatT,
  std::vector <T> & A,
  std::vector <TLDmodel<T>*> &LDmodels,
  std::vector <int> &LDidx,

  std::vector <Tview_factor<T>> & Fmat,              // output
  const T & epsC = 0.00872654) {

  //
  // Calculate the centroids of triangles
  //

  int Nt = Tr.size();

  T *CatT = new T [3*Nt];

  {
    T *c = CatT, *v[3];

    for (auto && t : Tr){

      // pointers to vertices
      for (int k = 0; k < 3; ++k) v[k] = V[t[k]].data;

      // centroid
      for (int k = 0; k < 3; ++k)
        c[k] = (v[0][k] + v[1][k] + v[2][k])/3;

      c += 3;
    }
  }

  //
  // Calculate depth and view-factor matrix DF thereby
  // using over-simplified check visibility, where only line-of-sight
  // between centroids of triangles is checked
  //

  struct Tp {

    int i;

    T h, F0, F;

    bool operator < (const Tp & rhs) const { return h < rhs.h; }
  };

  // depth and view-factor matrix
  std::vector<std::vector<Tp>> DF(Nt);

  {

    T tmp, tmp2, s, s2, *n[2], *c[2], a[3];

    Tp p[2];


    TLDmodel<T> *pld;

    // loop over triangles Ti
    for (p[0].i = 1, c[0] = CatT + 3; p[0].i < Nt; ++p[0].i, c[0] += 3) {

      n[0] = NatT[p[0].i].data;                  // normal of Ti

      // loop over triangles Tj
      for (p[1].i = 0, c[1] = CatT; p[1].i < p[0].i; ++p[1].i, c[1] += 3) {

        //
        // Check if it is possible to see the centroid of Tj from
        // the centroid of Ti and vice versa
        //

        // vector connected centroids c -> c1: a = c1 - c
        utils::sub3D(c[1], c[0], a);

        n[1] = NatT[p[1].i].data;            // normal of Tj

        // looking at Tj from Ti
        if ((p[0].h = utils::dot3D(n[0], a)) > 0 &&
            (p[1].h = -utils::dot3D(n[1], a)) > 0) {

          tmp = epsC*(s = std::sqrt(s2 = utils::norm2(a)));

          // throw away also all pairs with to large viewing angle
          if (p[0].h > tmp && p[1].h > tmp) {

            // conclusion: probably Tj illuminates Ti and vice versa

            //
            // calculate Lambert view factor
            //
            tmp = p[0].h*p[1].h/(s2*s2);

            //
            // calculate LD view factors
            //

            // looking at p[1] from p[0] and vice versa
            for (int k = 0; k < 2; ++k) {
              tmp2 = tmp*A[p[k].i];

              p[k].F0 = tmp2/utils::m_pi;

              if ((pld = LDmodels[LDidx[p[k].i]]))
                p[k].F = tmp2*pld->F(p[k].h/s);
              else
                p[k].F = p[k].h/s;
            }

            //
            // storing the results in depth and view-factor matrix
            //

            DF[p[0].i].push_back(p[1]);  // registering pair (p.i <- p1)
            DF[p[1].i].push_back(p[0]);  // registering pair (p1.i <- p)
          }
        }
      }
    }
  }

  //
  // Check if the line of sign from centroids of triangles is obstructed
  // and generate reduced depth-view factor matrix DF
  //

  {
    int *t, i = 0;

    bool ok_visible;

    T *c = CatT, *c1, *n, *v[3];

    for (auto && q : DF) {

        // if there is one element visible there is no obstruction possible
        if (q.size() > 1) {

        // sorting w.r.t. depth from triangle with index p.first
        std::sort(q.begin(), q.end());

        auto it_b = q.begin(), it = it_b + 1;

        // look over triangles and see is line-of-sight is obstructed
        while (it != q.end()) {

          // centroid of the triangle view from c
          c1 = CatT + 3*it->i;

          ok_visible = true;

          // check if line c1 <-> c is cut by triangle at less depth
          // from triangle with index p.first
          for (auto it1 = it_b; ok_visible && it1 != it; ++it1) {

            // pointers to vertices
            t = Tr[it1->i].data;
            for (int k = 0; k < 3; ++k) v[k] = V[t[k]].data;

            // normal of the triangle
            n = NatT[it1->i].data;

            // check if triangle cuts the line
            //ok_visible = !triangle_cuts_line(v, c, c1);
            ok_visible = !triangle_cuts_line(n, v, c, c1);
          }

          // line-of-sight of triangles with indices (p.first, it->i)
          // is obstructed, erasing *it element
          if (!ok_visible) {

            // erase conjugate pair (it->i, p.first) from DF[it->i]
            auto & z = DF[it->i];
            for (auto it1 = z.begin(), it1_e = z.end(); it1 != it1_e; ++it1)
              if (it1->i == i) {
                z.erase(it1);
                break;
              }

            // erase (u.first, it->i) from DF[p.first]
            it = q.erase(it);
          } else ++it;
        }
      }

      ++i;
      c += 3;
    }
  }

  delete [] CatT;

  //
  // Generate LD view factor matrix F by collecting data
  // from depth-view factor matrix
  //

  Fmat.clear();

  int i = 0;
  for (auto && p : DF){
    for (auto && q : p) Fmat.emplace_back(i, q.i, q.F0, q.F);
    ++i;
  }
}


/*
  Matrix element of the sparse-matrix for n-body case
  F_{(b_1, i_1), (b_2, i_2)}
*/
template <class T>
struct Tview_factor_nbody {

  int b1, i1,
      b2, i2;
                    // F_{ (b_2, i_2)->(b_1, i_1) }
  T F0,             // Lambert
    F;              // Limb darkend

  Tview_factor_nbody() {}
  Tview_factor_nbody(int b1, int i1, int b2, int i2, const T &F0, const T &F)
  : b1(b1), i1(i1), b2(b2), i2(i2), F0(F0), F(F) {}
};


/*
  Calculating limb-darkened radiosity/view-factor matrices with elements
  defined per TRIANGLE for a set of n convex bodies.

  Input:

    V - vector of n vectors of vertices
    Tr - vector of n vectors of triangles
    NatT - vector of n vectors of normals at triangles
    A - vector of n vector of areas of triangles
    LDmodels - vector of n limb darkening models in use
    epsC - threshold for permitted cos(theta)
              cos(theta_i) > epsC to be considered in line-of-sight
           ideally epsC = 0, epsC=0.00872654 corresponds to 89.5deg
  Output:
    Fmat - matrix of LD view factors
            F0 - Lambert view-factor
            F - limb-darkened view-factor
              - cos(theta') if LDmodels[i] == 0, evaluation of LD is
                postponed to routines outside the routine
*/


template <class T>
void triangle_mesh_radiosity_matrix_triangles_nbody_convex(
  std::vector <std::vector <T3Dpoint<T>>> & V,              // inputs
  std::vector <std::vector <T3Dpoint<int>>> & Tr,
  std::vector <std::vector <T3Dpoint<T>>> & NatT,
  std::vector <std::vector <T>> & A,
  std::vector <TLDmodel<T>*> & LDmodels,

  std::vector <Tview_factor_nbody<T>> & Fmat,                  // output
  const T & epsC = 0.00872654) {

  //
  // Check if the LDmodels are supplied
  //

  int nb = V.size();  // number of bodies

  if (nb == 1) return;

  //
  // Calculate nr. of triangles
  //

  std::vector<int> Nt(nb);

  for (int i = 0; i < nb; ++i) Nt[i] = Tr[i].size();

  //
  // Calculate the centroids of triangles
  //

  std::vector <std::vector<T3Dpoint<T>>> CatT(nb);

  {
    T c[3], *v[3];

    for (int i = 0; i < nb; ++i) {

      CatT[i].reserve(Nt[i]);

      for (auto && t : Tr[i]){

        // pointers to vertices
        for (int k = 0; k < 3; ++k) v[k] = V[i][t[k]].data;

        // centroid
        for (int k = 0; k < 3; ++k) c[k] = (v[0][k] + v[1][k] + v[2][k])/3;

        CatT[i].push_back(c);
      }
    }
  }

  //
  // For two bodies we can build matrix based on viewing angles
  //

  if (nb == 2) {

    T tmp, s, s2, F, *n[2], *c[2], a[3], h[2];

    TLDmodel<T> *pld;

    for (int i1 = 0, mi = Nt[0]; i1 < mi; ++i1) {

      c[0] = CatT[0][i1].data;
      n[0] = NatT[0][i1].data;

      for (int j1 = 0, mj = Nt[1]; j1 < mj; ++j1) {

        c[1] = CatT[1][j1].data;
        n[1] = NatT[1][j1].data;

        //
        // Check if it is possible to see the centroid of Tj from
        // the centroid of Ti and vice versa
        //

        // vector connected centroids c -> c1: a = c1 - c
        utils::sub3D(c[1], c[0], a);

        // looking at Tj from Ti
        if ((h[0] = utils::dot3D(n[0], a)) > 0 &&
            (h[1] = -utils::dot3D(n[1], a)) > 0) {

          tmp = epsC*(s = std::sqrt(s2 = utils::norm2(a)));

          // throw away also all pairs with to large viewing angle
          if (h[0] > tmp && h[1] > tmp) {

            // conclusion: probably Tj illuminates Ti and vice versa

            //
            // calculate Lambert view factor
            //

            F = h[0]*h[1]/(s2*s2);

            //
            // calculate LD view factors and store results in the matrix
            //

            // F_{(0, i1) -> (1,j1) } = F_{(1,j1), (0, i1)}
            tmp = F*A[0][i1];

            if ((pld = LDmodels[0]))
              Fmat.emplace_back(1, j1, 0, i1, tmp/utils::m_pi, tmp*pld->F(h[0]/s));
            else
              Fmat.emplace_back(1, j1, 0, i1, tmp/utils::m_pi, h[0]/s);

            // F_{(1,j1) -> (0,i1) } = F_{(0, i1), (1,j1)}
            tmp = F*A[1][j1];

            if ((pld = LDmodels[1]))
              Fmat.emplace_back(0, i1, 1, j1, tmp/utils::m_pi, tmp*pld->F(h[1]/s));
            else
              Fmat.emplace_back(0, i1, 1, j1, tmp/utils::m_pi, h[1]/s);
          }
        }
      }
    }

    return;
  }


  //
  // Calculate depth and view-factor matrix DF thereby
  // using over-simplified check visibility, where only line-of-sight
  // between centroids of triangles is checked
  //

  struct Tp {

    int b,    // index of the body
        i;    // index of the triangle

    T
      h,      // depth
      F0,     // Lambert view-factor
      F;      // LD view-factor

    bool operator < (const Tp & rhs) const { return h < rhs.h; }
  };

  // define depth and view-factor matrix
  std::vector<std::vector<std::vector<Tp>>> DF(nb);
  for (int i = 0; i < nb; ++i) DF[i].resize(Nt[i]);

  {
    int m[2];

    T tmp, tmp2, s, s2, *n[2], *c[2], a[3];

    Tp p[2];

    TLDmodel<T> *pld;

    for (p[0].b = 1; p[0].b < nb; ++p[0].b)
    for (p[0].i = 0, m[0] = Nt[p[0].b]; p[0].i < m[0]; ++p[0].i) {

      c[0] = CatT[p[0].b][p[0].i].data;
      n[0] = NatT[p[0].b][p[0].i].data;

      for (p[1].b = 0; p[1].b < p[0].b; ++p[1].b)
      for (p[1].i = 0, m[1] = Nt[p[1].b]; p[1].i < m[1]; ++p[1].i) {

        c[1] = CatT[p[1].b][p[1].i].data;
        n[1] = NatT[p[1].b][p[1].i].data;

        //
        // Check if it is possible to see the centroid of Tj from
        // the centroid of Ti and vice versa
        //

        // vector connected centroids c -> c1: a = c1 - c
        utils::sub3D(c[1], c[0], a);

        // looking at Tj from Ti
        if ((p[0].h = utils::dot3D(n[0], a)) > 0 &&
            (p[1].h = -utils::dot3D(n[1], a)) > 0) {

          tmp = epsC*(s = std::sqrt(s2 = utils::norm2(a)));

          // throw away also all pairs with to large viewing angle
          if (p[0].h > tmp && p[1].h > tmp) {

            // conclusion: probably Tj illuminates Ti and vice versa

            //
            // calculate Lambert view factor
            //
            tmp = p[0].h*p[1].h/(s2*s2);

            //
            // calculate LD view factors
            //

            // looking at Tj from Ti and vice versa
            for (int k = 0; k < 2; ++k){
              tmp2 = tmp*A[p[k].b][p[k].i];

              p[0].F0 = tmp2/utils::m_pi;

              if ((pld = LDmodels[p[k].b]))
                p[k].F = tmp2*pld->F(p[k].h/s);
              else
                p[k].F = p[k].h/s;
            }
            //
            // storing the results in depth and view-factor matrix
            //

            // registering pair p[1] -> p[0] : F_{p[1] -> p[0]}
            DF[p[0].b][p[0].i].push_back(p[1]);

            // registering pair p[0] -> p[1] : F_{p[0] -> p[1]}
            DF[p[1].b][p[1].i].push_back(p[0]);

          }
        }
      }
    }
  }

  //
  // Check if the line of sign from centroids of triangles is obstructed
  // and generate reduced depth-view factor matrix DF
  //

  {

    T *c, *c1, *n, *v[3];

    int b = 0, // index of the body
        i;     // index of a triangle on body


    for (auto && B : DF) {

      i = 0;
      for (auto && q : B) {

        // if there is one element visible there is no obstruction possible
        if (q.size() > 1) {

          // if there is only one body visible from a triangle then
          // there is no obstruction possible
          auto itb = q.begin(), ite = q.end(), it = itb;

          {
            int b1 = it->b;
            while (++it != ite) if (it->b != b1) break;
          }

          // no other bodies is being observed from triangle (i, j)
          if (it == ite) continue;

          c = CatT[b][i].data;

          // sorting w.r.t. depth from triangle with index p.first
          std::sort(itb, ite);

          int b1 = itb->b;

          it = itb + 1;

          // look over triangles and see is line-of-sight is obstructed
          while (it != q.end()) {

            // in convex bodies, as long we are looking at the same
            // one it can not bi obstructed

            if (it->b == b1) { ++it; continue; }

            // centroid of the triangle view from c
            c1 = CatT[it->b][it->i].data;

            // check if line c1 <-> c is cut by triangle at less depth
            // from triangle with index p.first
            auto it1 = itb;

            while (it1 != it) {

              // pointers to vertices
              int *t = Tr[it1->b][it1->i].data;
              for (int k = 0; k < 3; ++k) v[k] = V[it1->b][t[k]].data;

              // normal of the triangle
              n = NatT[it1->b][it1->i].data;

              // check if triangle cuts the line
              //ok_visible = !triangle_cuts_line(v, c, c1);
              if (triangle_cuts_line(n, v, c, c1)) break;

              ++it1;
            }

            // line-of-sight of triangles with indices (p.first, it->i)
            // is obstructed, erasing *it element
            if (it1 != it) {

              // erase conjugate pair (it->i, p.first) from DF[it->i]
              auto & z = DF[it->b][it->i];
              for (auto it2 = z.begin(), it2e = z.end(); it2 != it2e; ++it2)
                if (it2->b == b && it2->i == i) {
                  z.erase(it2);
                  break;
                }

              // erase (u.first, it->i) from DF[p.first]
              it = q.erase(it);
            } else ++it;
          }
        }
        ++i;
      }
      ++b;
    }
  }

  //
  // Generate LD view factor matrix F by collecting data
  // from depth-view factor matrix
  //

  Fmat.clear();

  {
    int b = 0, // index of the body
        i;     // index of the triagle on body

    for (auto && B : DF) {  // loop over bodies

      i = 0;
      for (auto && p : B) {    // loop over triangles
        for (auto && q : p) Fmat.emplace_back(q.b, q.i, b, i, q.F0, q.F);
        ++i;
      }
      ++b;
    }
  }
}



/*
  Check is the circle with center c and radius r laying on the plane
  with normal n cuts the line v[0] + (v[1] - v[0])t for t in [0,1].

  Input
    c - center
    n - normal vector
    r2 - square of the radius = r^2

    v - edges of the line section

  Return
    true if the circle intersects the line, false otherwise
*/

template <class T>
bool disk_cuts_line(T c[3], T n[3], const T & r2, T *v[2]){

  //
  // If the line cuts the plane of the circle
  //

  T s[2] = {0, 0}, a[3], b[3];

  for (int i = 0; i < 3; ++i) {
   a[i] = c[i] - v[0][i];
   b[i] = v[1][i] - v[0][i];

   s[0] += a[i]*n[i];
   s[1] += b[i]*n[i];
  }

  if (s[1] == 0) return false;

  s[0] /= s[1];

  if (s[0] < 0 || s[0] > 1) return false;

  //
  // If the point on the plane is inside the circle
  //

  s[1] = 0;

  T t;
  for (int i = 0; i < 3; ++i) {
    t = a[i] - s[0]*b[i];
    s[1] += t*t;
  }

  if (s[1] > r2) return false;

  return true;
}

/*
  Calculating limb-darkned radiosity matrix/view-factors with elements
  defined per VERTEX. A vertex is associated with a disk in the
  tangent space equal to 1/3 of neighboring triangles.

  Input:

    V - vector of vertices
    Tr - vector of triangles
    NatV - vector of normals at vertices
    A - vector of areas of triangles
    LDmodels - vector of limb darkening models in use
    LDidx - vector of indices of models used on each of vertices

    epsC - threshold for permitted cos(theta)
              cos(theta_i) > epsC to be considered in line-of-sight
           ideally epsC = 0, epsC=0.00872654 corresponds to 89.5deg
  Output:
    Fmat - matrix of LD view factors
            F0 - Lambert view-factor
            F - limb-darkened view-factor
              - cos(theta') if LDmodels[i] == 0, evaluation of LD is
                postponed to routines outside the routine
*/

template <class T>
void triangle_mesh_radiosity_matrix_vertices(
  std::vector <T3Dpoint<T>> & V,                  // inputs
  std::vector <T3Dpoint<int>> & Tr,
  std::vector <T3Dpoint<T>> & NatV,
  std::vector <T> & A,
  std::vector <TLDmodel<T>*> & LDmodels,
  std::vector <int> & LDidx,
  std::vector <Tview_factor<T>> & Fmat,              // output
  const T & epsC = 0.00872654) {

  //
  // Calculate the areas associated to vertices
  //

  int Nv = V.size();

  std::vector<T> AatV(Nv, 0);

  {
    T a;

    auto itA = A.begin();

    for (auto && t : Tr){
      a = (*itA)/3;
      for (int j = 0; j < 3; ++j) AatV[t[j]] += a;
      ++itA;
    }
  }

  //
  // Calculate depth and view-factor matrix DF thereby
  // using over-simplified check visibility, where only line-of-sight
  // between centroids of triangles is checked
  //

  struct Tp {

    int i;

    T h, F0, F;

    bool operator < (const Tp & rhs) const { return h < rhs.h; }
  };

  // depth and view-factor matrix
  std::vector<std::vector<Tp>> DF(Nv);

  {
    T tmp, tmp2, s, s2, a[3];

    Tp p[2];

    TLDmodel<T>* pld;

    std::vector<int>::iterator itLb = LDidx.begin(), itL[2];

    typename std::vector<T>::iterator itAb = AatV.begin(), itA[2];

    typename std::vector<T3Dpoint<T>>::iterator
      itVb = V.begin(), itVe = V.end(), itV[2],
      itNb = NatV.begin(), itN[2];

    p[0].i = 1;
    itV[0] = itVb+1;
    itA[0] = itAb+1;
    itN[0] = itNb+1;
    itL[0] = itLb+1;

    while (itV[0] != itVe) {

      p[1].i = 0;
      itV[1] = itVb;
      itA[1] = itAb;
      itN[1] = itNb;
      itL[1] = itLb;

      while (itV[1] != itV[0]) {

        //
        // Check if it is possible to see the vertex V1 from to V and
        // vice versa: itV pointing to V, itV1 pointing to V1
        //

        // vector connecting vertices  a = V1 - V
        utils::sub3D(itV[1]->data, itV[0]->data, a);

        // looking at V1 from V and vice versa
        if ((p[0].h = +utils::dot3D(itN[0]->data, a)) > 0 &&
            (p[1].h = -utils::dot3D(itN[1]->data, a)) > 0) {

          tmp = epsC*(s = std::sqrt(s2 = utils::norm2(a)));

          // throw away also all pairs with to large viewing angle
          if (p[0].h > tmp && p[1].h > tmp) {

            // conclusion: probably V illuminates V1 and vice versa

            //
            // calculate Lambert view factor
            //
            tmp = p[0].h*p[1].h/(s2*s2);

            //
            // calculate LD view factors
            //

            // looking at V1 from V and vice versa
            for (int i = 0; i < 2; ++i) {
              tmp2 = tmp*(*itA[i]);

              p[i].F0 = tmp2/utils::m_pi;

              if ((pld = LDmodels[*itL[i]]))
                p[i].F = tmp2*pld->F(p[i].h/s);
              else
                p[i].F = p[i].h/s;
            }
            //
            // storing the results in depth and view-factor matrix
            //

            DF[p[0].i].push_back(p[1]);  // registering pair (p[1].i -> p[0])
            DF[p[1].i].push_back(p[0]);  // registering pair (p[0].i -> p[1])
          }
        }

        ++p[1].i;
        ++itV[1];
        ++itN[1];
        ++itA[1];
        ++itL[1];
      }

      ++p[0].i;
      ++itV[0];
      ++itN[0];
      ++itA[0];
      ++itL[0];
    }
  }

  //
  // Divide areas associated to vertices by pi do get effective r^2
  //
  {
    T fac = 1/utils::m_pi;
    for (auto && a : AatV) a *= fac;
  }
  //
  // Check if the line of sign from centroids of triangles is obstructed
  // and generate reduced depth-view factor matrix DF
  //

  {
    int i = 0;

    bool ok_visible;

    T *v[2];

    typename std::vector<T3Dpoint<T>>::iterator itV = V.begin();

    typename std::vector<Tp>::iterator itb, it;

    for (auto && q : DF) {

        // if there is one element visible there is no obstruction possible
        if (q.size() > 1) {

        // sorting w.r.t. depth from triangle with index p.first
        std::sort(q.begin(), q.end());

        it = (itb = q.begin()) + 1;

        v[0] = itV->data;

        // look over triangles and see is line-of-sight is obstructed
        while (it != q.end()) {

          // centroid of the triangle view from c
          v[1] = V[it->i].data;

          ok_visible = true;

          // check if line v <-> it->V.data is cut by a circle at less depth
          for (auto it1 = itb; ok_visible && it1 != it; ++it1)
            ok_visible = !disk_cuts_line(V[it1->i].data, NatV[it1->i].data, AatV[it1->i], v);

          // line-of-sight between vertices with indices (i, it->i)
          // is obstructed, erasing these pairs
          if (!ok_visible) {

            // erase conjugate pair (it->i, i) from DF[it->i]
            auto & z = DF[it->i];
            for (auto it1 = z.begin(), it1e = z.end(); it1 != it1e; ++it1)
              if (it1->i == i) {
                z.erase(it1);
                break;
              }

            // erase (i, it->i) from DF[i]
            it = q.erase(it);
          } else ++it;
        }
      }

      ++i;
      ++itV;
    }
  }

  //
  // Generate LD view factor matrix F by collecting data
  // from depth-view factor matrix
  //

  Fmat.clear();

  int i = 0;
  for (auto && p : DF) {
    for (auto && q : p)
      Fmat.emplace_back(i, q.i, q.F0, q.F); // F_{i,q.i=j} = F_{i<-j}
    ++i;
  }
}


/*
  Calculating limb-darkened radiosity/view-factors matrix with matrix
  elements defined per VERTICES for a set of n convex bodies.

  Input:

    V - vector of n vectors of vertices
    Tr - vector of n vectors of triangles
    NatV - vector of n vectors of normals at vertices
    A - vector of n vector of areas of triangles
    LDmodels - vector of n limb darkening models in use

    epsC - threshold for permitted cos(theta)
              cos(theta_i) > epsC to be considered in line-of-sight
           ideally epsC = 0, epsC=0.00872654 corresponds to 89.5deg
  Output:
    Fmat - matrix of LD view factors
            F0 - Lambert view-factor
            F - limb-darkened view-factor
              - cos(theta') if LDmodels[i] == 0, evaluation of LD is
                postponed to routines outside the routine
*/


template <class T>
void triangle_mesh_radiosity_matrix_vertices_nbody_convex(
  std::vector <std::vector <T3Dpoint<T>>> & V,              // inputs
  std::vector <std::vector <T3Dpoint<int>>> & Tr,
  std::vector <std::vector <T3Dpoint<T>>> & NatV,
  std::vector <std::vector <T>> & A,
  std::vector <TLDmodel<T>*> & LDmodels,

  std::vector <Tview_factor_nbody<T>> & Fmat,                  // output
  const T & epsC = 0.00872654) {

  //
  // Check if the LDmodels are supplied
  //

  int nb = V.size();  // number of bodies

  if (nb == 1) return;

  //
  // Calculate nr. of vertices
  //

  std::vector<int> Nv(nb);

  for (int i = 0; i < nb; ++i) Nv[i] = V[i].size();

  //
  // Calculate the area associated to vertices
  //

  std::vector<std::vector<T>> AatV(nb);
  {
    T a;

    for (int i = 0; i < nb; ++i) {

      AatV[i].resize(Nv[i], 0);

      auto itA = A[i].begin();

      for (auto && t : Tr[i]) {
        a = (*itA)/3;
        for (int j = 0; j < 3; ++j) AatV[i][t[j]] += a;
        ++itA;
     }
    }
  }

  //
  // For two bodies we can build matrix based on viewing angles
  //

  if (nb == 2) {

    T tmp, s, s2, F, *n[2], *v[2], a[3], h[2];

    TLDmodel<T>* pld;

    for (int i1 = 0, mi = Nv[0]; i1 < mi; ++i1) {

      v[0] = V[0][i1].data;
      n[0] = NatV[0][i1].data;

      for (int j1 = 0, mj = Nv[1]; j1 < mj; ++j1) {

        v[1] = V[1][j1].data;
        n[1] = NatV[1][j1].data;

        //
        // Check if vertex v[0] is visible from v[1]
        //

        // vector connected centroids v -> v1: a = v1 - v
        utils::sub3D(v[1], v[0], a);

        // looking at Tj from Ti
        if ((h[0] = utils::dot3D(n[0], a)) > 0 &&
            (h[1] = -utils::dot3D(n[1], a)) > 0) {

          tmp = epsC*(s = std::sqrt(s2 = utils::norm2(a)));

          // throw away also all pairs with to large viewing angle
          if (h[0] > tmp && h[1] > tmp) {

            // conclusion: probably Tj illuminates Ti and vice versa

            //
            // calculate Lambert view factor
            //

            F = h[0]*h[1]/(s2*s2);

            //
            // calculate LD view factors and store results in the matrix
            //

            // F_{(0, i1) -> (1,j1) } = F_{(1,j1), (0, i1)}
            tmp = F*AatV[0][i1];
            if ((pld = LDmodels[0]))
              Fmat.emplace_back(1, j1, 0, i1, tmp/utils::m_pi, tmp*pld->F(h[0]/s));
            else
              Fmat.emplace_back(1, j1, 0, i1, tmp/utils::m_pi, h[0]/s);

            // F_{(1,j1) -> (0,i1) } = F_{(0, i1), (1,j1)}
            tmp = F*AatV[1][j1];
            if ((pld = LDmodels[1]))
              Fmat.emplace_back(0, i1, 1, j1, tmp/utils::m_pi, tmp*pld->F(h[1]/s));
            else
              Fmat.emplace_back(0, i1, 1, j1, tmp/utils::m_pi, h[1]/s);

          }
        }
      }
    }

    return;
  }


  //
  // Calculate depth and view-factor matrix DF thereby
  // using over-simplified check visibility, where only line-of-sight
  // between centroids of triangles is checked
  //

  struct Tp {

    int b,    // index of the body
        i;    // index of the triangle

    T
      h,      // depth
      F0,     // Lambert view factor
      F;      // LD view-factor

    bool operator < (const Tp & rhs) const { return h < rhs.h; }
  };

  // define depth and view-factor matrix
  std::vector<std::vector<std::vector<Tp>>> DF(nb);
  for (int i = 0; i < nb; ++i) DF[i].resize(Nv[i]);

  {
    int m[2];

    T tmp, tmp2, s, s2, *n[2], *v[2], a[3];

    Tp p[2];

    TLDmodel<T>* pld;

    for (p[0].b = 1; p[0].b < nb; ++p[0].b)
    for (p[0].i = 0, m[0] = Nv[p[0].b]; p[0].i < m[0]; ++p[0].i) {

      v[0] = V[p[0].b][p[0].i].data;
      n[0] = NatV[p[0].b][p[0].i].data;

      for (p[1].b = 0; p[1].b < p[0].b; ++p[1].b)
      for (p[1].i = 0, m[1] = Nv[p[1].b]; p[1].i < m[1]; ++p[1].i) {

        v[1] = V[p[1].b][p[1].i].data;
        n[1] = NatV[p[1].b][p[1].i].data;

        //
        // Check if it is possible to see the centroid of Tj from
        // the centroid of Ti and vice versa
        //

        // vector connected centroids c -> c1: a = c1 - c
        utils::sub3D(v[1], v[0], a);

        // looking at Tj from Ti
        if ((p[0].h = utils::dot3D(n[0], a)) > 0 &&
            (p[1].h = -utils::dot3D(n[1], a)) > 0) {

          tmp = epsC*(s = std::sqrt(s2 = utils::norm2(a)));

          // throw away also all pairs with to large viewing angle
          if (p[0].h > tmp && p[1].h > tmp) {

            // conclusion: probably Tj illuminates Ti and vice versa

            //
            // calculate Lambert view factor
            //
            tmp = p[0].h*p[1].h/(s2*s2);

            //
            // calculate LD view factors
            //

            // looking at Tj from Ti and vice versa
            for (int k = 0; k < 2; ++k) {
              tmp2 = tmp*AatV[p[k].b][p[k].i];

              p[k].F0 = tmp2/utils::m_pi;

              if ((pld = LDmodels[p[k].b]))
                p[k].F = tmp2*pld->F(p[k].h/s);
              else
                p[k].F = p[k].h/s;
            }

            //
            // storing the results in depth and view-factor matrix
            //

            // registering pair p[1] -> p[0] : F_{p[1] -> p[0]}
            DF[p[0].b][p[0].i].push_back(p[1]);

            // registering pair p[0] -> p[1] : F_{p[0] -> p[1]}
            DF[p[1].b][p[1].i].push_back(p[0]);

          }
        }
      }
    }
  }

  //
  // Calculate radii^2 of disks associated to vertices
  //

  {
    T fac = 1/utils::m_pi;

    for (auto && B : AatV) for (auto && q : B) q *= fac;
  }

  //
  // Check if the line of sign from centroids of triangles is obstructed
  // and generate reduced depth-view factor matrix DF
  //

  {

    T *v[2];

    int b = 0, // index of the body
        i;     // index of a triangle on body

    for (auto && B : DF) {

      i = 0;
      for (auto && q : B) {

        // if there is one element visible there is no obstruction possible
        if (q.size() > 1) {

          // if there is only one body visible from a triangle then
          // there is no obstruction possible
          auto itb = q.begin(), ite = q.end(), it = itb;

          {
            int b1 = it->b;
            while (++it != ite) if (it->b != b1) break;
          }

          // no other bodies is being observed from triangle (i, j)
          if (it == ite) continue;

          v[0] = V[b][i].data;

          // sorting w.r.t. depth from triangle with index p.first
          std::sort(itb, ite);

          int b1 = itb->b;

          it = itb + 1;

          // look over triangles and see is line-of-sight is obstructed
          while (it != q.end()) {

            // in convex bodies, as long we are looking at the same
            // one it can not bi obstructed

            if (it->b == b1) { ++it; continue; }

            // centroid of the triangle view from c
            v[1] = V[it->b][it->i].data;

            // check if line c1 <-> c is cut by triangle at less depth
            // from triangle with index p.first
            auto it1 = itb;

            while (it1 != it) {
              // check if triangle cuts the line
              if (disk_cuts_line(
                V[it1->b][it1->i].data,
                NatV[it1->b][it1->i].data,
                AatV[it1->b][it1->i], v)
              ) break;
              ++it1;
            }

            // line-of-sight of triangles with indices (p.first, it->i)
            // is obstructed, erasing *it element
            if (it1 != it) {

              // erase conjugate pair (it->i, p.first) from DF[it->i]
              auto & z = DF[it->b][it->i];
              for (auto it2 = z.begin(), it2e = z.end(); it2 != it2e; ++it2)
                if (it2->b == b && it2->i == i) {
                  z.erase(it2);
                  break;
                }

              // erase (u.first, it->i) from DF[p.first]
              it = q.erase(it);
            } else ++it;
          }
        }
        ++i;
      }
      ++b;
    }
  }

  //
  // Generate LD view factor matrix F by collecting data
  // from depth-view factor matrix
  //

  Fmat.clear();

  {
    int b = 0, // index of the body
        i;     // index of the triagle on body

    for (auto && B : DF) {  // loop over bodies

      i = 0;
      for (auto && p : B) {    // loop over triangles
        for (auto && q : p) Fmat.emplace_back(q.b, q.i, b, i, q.F0, q.F);
        ++i;
      }
      ++b;
    }
  }
}


/*
  Solving the radiosity model as given in (Wilson, 1990) using the
  limb-darkened view factors

    (1 - diag(R) F) M = M0

  where
    F - matrix of limb-darkened view-factors
    R is vector of reflection coefficients,
    M0 is vector of intrinsic radiant exitances from triangles/vertices
    M is vector of radiosity (intrinsic and reflection) of triangles/vertices

  Method:
    Simple iteration

      M_{k+1} = M0  + diag(R) F M_{k}

    with initial condition
      M_0 = M0

  Input:
    Fmat - matrix of view factor
    R - vector of albedo/reflection of triangles/of vertices
    M0 - vector of intrisic radiant exitance of triangles/of vertices
    epsM - relative precision of radiosity
    max_iter - maximal number of iterations

  Output:
    M - vector of radiosity (intrinsic and reflection) of triangles/of vertices

  Returns:
    true if we reached wanted relative precision, false otherwise
*/
template <class T>
bool solve_radiosity_equation_Wilson(
  std::vector<Tview_factor<T>> &Fmat,      // input
  std::vector<T> &R,
  std::vector<T> &M0,
  std::vector<T> &M,                    // output
  const T & epsM = 1e-12,
  const T & max_iter = 100) {

  int
    Nt = R.size(),        // number of triangles
    it = 0,               // number of iterations
    size = Nt*sizeof(T);  // size of vectors in bytes

  T *buf = new T [2*Nt], *S0 = buf, *S1 = buf + Nt,   // prepare buffer
    *pM = M0.data(), t, dS, Smax;

  // initial condition
  memcpy(S0, pM, size);

  do {

    // iteration step
    memcpy(S1, pM, size);

    for (auto && f: Fmat) S1[f.i] += R[f.i]*f.F*S0[f.j];

    // check convergence
    dS = Smax = 0;
    for (int j = 0; j < Nt; ++j) {
      if (S1[j] > Smax) Smax = S1[j];
      t = std::abs(S1[j] - S0[j]);
      if (t > dS)  dS = t;
    }

    //std::cerr << dS << '\t' << Smax << '\t' << dS/Smax << '\n';

    // swap pointers as to save previous step
    utils::swap(S0, S1);

  } while (dS > epsM*Smax && ++it < max_iter);


  // copy the results to the output
  M.assign(S0, S0 + Nt);

  delete [] buf;

  return it < max_iter;
}

/*
Solving the radiosity model proposed by M. Horvat for Phoebe 2b.
  Introducing radiocity matrices:

    L_{LD} - limb-darkened view-factors
    L_0 - Lambert view-factors

  and fluxes:

    F_{in} - incoming (from reflection) flux
    F_{out} - outwards (intrinsic and reflection) flux
    F_0 - intrinsic flux

  The equations are
    S0 =  L_{LD} F_0

    F_{in} = S0 + L_0 diag(R) F_{in}

    F_{out} = F_0 + diag(R) F_{in}

  Method:
    Simple iteration

      F_{in, k+1} =  S0 + L_0  diag(R) F_{in, k}   k = 0, 1, ...,

    with initial condition

      F_{in,0} = S0

  Input:
    Fmat - matrix of view factor
    R - vector of albedo/reflection of triangles/of vertices
    F0 - vector of intrisic radiant exitance of triangles/of vertices
    S0 - vector of LD reflected intrisic radiant exitance of triangles/of vertices
    epsF - relative precision of radiosity
    max_iter - maximal number of iterations

  Output:
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices

  Returns:
    true if we reached wanted relative precision, false otherwise

*/
template <class T>
bool solve_radiosity_equation_Horvat(
  std::vector<Tview_factor<T>> &Fmat,      // input
  std::vector<T> &R,
  std::vector<T> &F0,
  std::vector<T> &S0,
  std::vector<T> &Fout,                 // output
  const T & epsF = 1e-12,
  const T & max_iter = 100) {

  int
    Nt = R.size(),        // number of triangles/vertices
    it = 0,               // number of iterations
    size = Nt*sizeof(T);  // size of vectors in bytes


  T *buf = new T [3*Nt],                  // prepare buffer
    *pS0 = S0.data(),
    *S[2] = {buf, buf + Nt};

  //
  // do iteration:
  //   F_{in, k+1} =  S0 + L_0 diag(R) F_{in, k}
  // with S0 = L_{LD} F0

  // initial condition
  memcpy(S[0], pS0, size);

  T t, dS, Smax;

  do {

    // iteration step: Lambert reflection
    // S[1] =  S0 + L_0 diag(R) S[0]
    memcpy(S[1], pS0, size);
    for (auto && f: Fmat) S[1][f.i] += f.F0*R[f.j]*S[0][f.j];

    // check convergence
    dS = Smax = 0;
    for (int j = 0; j < Nt; ++j) {
      if (S[1][j] > Smax) Smax = S[1][j];
      t = std::abs(S[1][j] - S[0][j]);
      if (t > dS)  dS = t;
    }

    //std::cerr << dS << '\t' << Smax << '\t' << dS/Smax << '\n';

    // swap pointers as to save previous step
    utils::swap(S[0], S[1]);

  } while (dS > epsF*Smax && ++it < max_iter);

  //
  // S[0] contains best approximate of F_{in}
  // calculate F_{out} = F0  + diag(R)F_{in}
  //

  Fout = F0;
  for (int j = 0; j < Nt; ++j) Fout[j] += R[j]*S[0][j];

  delete [] buf;

  return it < max_iter;
}


/*
  Solving the radiosity model proposed by M. Horvat for Phoebe 2b.
  Introducing radiocity matrices:

    L_{LD} - limb-darkened view-factors
    L_0 - Lambert view-factors

  and fluxes:

    F_{in} - incoming (from reflection) flux
    F_{out} - outwards (intrinsic and reflection) flux
    F_0 - intrinsic flux

  The equations are
    S0 =  L_{LD} F_0

    F_{in} = S0 + diag(R) F_{in}

    F_{out} = F_0 + diag(R) F_{in}

  Method:
    Simple iteration

      F_{in, k+1} =  S0 + diag(R) L_0 F_{in, k}   k = 0, 1, ...,

    with initial condition

      F_{in,0} = S0

  Input:
    Fmat - matrix of view factor
    R - vector of albedo/reflection of triangles/of vertices
    F0 - vector of intrisic radiant exitance of triangles/of vertices
    epsF - relative precision of radiosity
    max_iter - maximal number of iterations

  Output:
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices

  Returns:
    true if we reached wanted relative precision, false otherwise
*/

template <class T>
bool solve_radiosity_equation_Horvat(
  std::vector<Tview_factor<T>> &Fmat,      // input
  std::vector<T> &R,
  std::vector<T> &F0,
  std::vector<T> &Fout,                // output
  const T & epsF = 1e-12,
  const T & max_iter = 100) {

  //
  // calculate limb-darkened emission
  //  S0 = L_{LD} F0
  //

  std::vector <T> S0(F0.size(), 0);
  for (auto && f: Fmat) S0[f.i] += f.F*F0[f.j];

  return solve_radiosity_equation_Horvat(Fmat, R, F0, S0, Fout, epsF, max_iter);

}

/*
  Solving the radiosity model for n-body case as given in
  (Wilson, 1990) using the limb-darkened view factors

    (1 - diag(R) F) M = M0

  where
    F is matrix of limb-darkened view-factors
    R is vector of reflection coefficients,
    M0 is vector of intrinsic radiant exitances from triangles/vertices
    M is vector of radiosity (intrinsic and reflection) of triangles/vertices

  Method:
    Simple iteration

      M_{k+1} = M0  + diag(R) F M_{k}

    with initial condition
      M_0 = M0

  Input:
    Fmat - matrix of view factor for n-body formalism
    R - vector of albedo/reflection of triangles/vertices
    M0 - vector of intrisic radiant exitance of triangles/vertices
    epsM - relative precision of radiosity
    max_iter - maximal number of iterations

  Output:
    M - vector of radiosity (intrinsic and reflection) of triangles/of vertices

  Returns:
    true if we reached wanted relative precision, false otherwise
*/

template <class T>
bool solve_radiosity_equation_Wilson_nbody(
  std::vector<Tview_factor_nbody<T>> &Fmat,      // input
  std::vector<std::vector<T>> &R,
  std::vector<std::vector<T>> &M0,
  std::vector<std::vector<T>> &M,             // output
  const T & epsM = 1e-12,
  const T & max_iter = 100) {

  // number of bodies
  int  nb = M0.size();

  // collect length of sub-vectors
  std::vector<int> N(nb);
  for (int i = 0; i < nb; ++i) N[i] = M0[i].size();

  std::vector<std::vector<T>> M1;

  // initial condition of the iteration
  M = M0;

  T t, dM, Mmax;

  int it = 0;

  do {

    // temporary store last result
    M1 = M;

    // iteration step

    M = M0;
    for (auto && f: Fmat) M[f.b1][f.i1] += R[f.b1][f.i1]*f.F*M1[f.b2][f.i2];

    // check convergence, compute L1 norms
    dM = Mmax = 0;
    for (int i = 0; i < nb; ++i) for (int j = 0, m = N[i]; j < m; ++j) {

      if (M[i][j] > Mmax) Mmax = M[i][j];
      t = std::abs(M[i][j] - M1[i][j]);
      if (t > dM)  dM = t;
    }

    //std::cerr << dM << '\t' << Mmax << '\t' << dM/Mmax << '\n';
  } while (dM > epsM*Mmax && ++it < max_iter);

  return it < max_iter;
}


/*
  Solving the radiosity model proposed by M. Horvat for Phoebe 2b.
  Introducing radiocity matrices:

    L_{LD} - limb-darkened view-factors
    L_0 - Lambert view-factors

  and fluxes:

    F_{in} - incoming (from reflection) flux
    F_{out} - outwards (intrinsic and reflection) flux
    F_0 - intrinsic flux

  The equations are
    S0 =  L_{LD} F_0

    F_{in} = S0 + L_0 diag(R) F_{in}

    F_{out} = F_0 + diag(R) F_{in}

  Method:
    Simple iteration

      F_{in, k+1} =  S0 + diag(R) L_0 F_{in, k}   k = 0, 1, ...,

    with initial condition

      F_{in,0} = S0

  Input:
    Fmat - matrix of view factor for n-body formalism
    R - vector of albedo/reflection of triangles/vertices
    F0 - vector of intrisic radiant exitance of triangles/vertices
    S0 - vector o LD diffusion of intrisic radiant exitance of triangles/vertices
    epsF - relative precision of radiosity
    max_iter - maximal number of iterations

  Output:
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices

  Returns:
    true if we reached wanted relative precision, false otherwise
*/

template <class T>
bool solve_radiosity_equation_Horvat_nbody(
  std::vector<Tview_factor_nbody<T>> &Fmat,      // input
  std::vector<std::vector<T>> &R,
  std::vector<std::vector<T>> &F0,
  std::vector<std::vector<T>> &S0,
  std::vector<std::vector<T>> &Fout,           // output
  const T & epsF = 1e-12,
  const T & max_iter = 100) {

  // number of bodies
  int nb = F0.size();

  // lengths of sub-vectors
  std::vector<int> N(nb);
  for (int i = 0; i < nb; ++i) N[i] = F0[i].size();

  //
  // do iteration:
  //   F_{in, k+1} =  S0 + L_0 diag(R) F_{in, k}
  // with S0 = L_{LD} F0

  // initial condition
  std::vector<std::vector<T>> Fin(S0), Ftmp;

  int it = 0;

  T t, dF, Fmax;

  do {

    // temporary store last result
    Ftmp = Fin;

    // iteration step
    Fin = S0;
    for (auto && f: Fmat)
      Fin[f.b1][f.i1] += f.F0*R[f.b2][f.i2]*Ftmp[f.b2][f.i2];

    // check convergence, compute L1 norms
    dF = Fmax = 0;
    for (int i = 0; i < nb; ++i) for (int j = 0, m = N[i]; j < m; ++j) {

      if (Fin[i][j] > Fmax) Fmax = Fin[i][j];
      t = std::abs(Fin[i][j] - Ftmp[i][j]);
      if (t > dF)  dF = t;
    }

  } while (dF > epsF*Fmax && ++it < max_iter);

  //
  // calculate F_{out} = F0  + diag(R)F_{in}
  //

  Fout = F0;
  for (int i = 0; i < nb; ++i)
    for (int j = 0, m = N[i]; j < m; ++j)
      Fout[i][j] += R[i][j]*Fin[i][j];

  return it < max_iter;
}

/*
  Solving the radiosity model proposed by M. Horvat for Phoebe 2b.
  Introducing radiocity matrices:

    L_{LD} - limb-darkened view-factors
    L_0 - Lambert view-factors

  and fluxes:

    F_{in} - incoming (from reflection) flux
    F_{out} - outwards (intrinsic and reflection) flux
    F_0 - intrinsic flux

  The equations are
    S0 =  L_{LD} F_0

    F_{in} = S0 + L_0 diag(R) F_{in}

    F_{out} = F_0 + diag(R) F_{in}

  Method:
    Simple iteration

      F_{in, k+1} =  S0 + diag(R) L_0 F_{in, k}   k = 0, 1, ...,

    with initial condition

      F_{in,0} = S0

  Input:
    Fmat - matrix of view factor for n-body formalism
    R - vector of albedo/reflection of triangles/vertices
    F0 - vector of intrisic radiant exitance of triangles/vertices
    epsF - relative precision of radiosity
    max_iter - maximal number of iterations

  Output:
    Fout - vector of radiosity (intrinsic and reflection) of triangles/of vertices

  Returns:
    true if we reached wanted relative precision, false otherwise
*/

template <class T>
bool solve_radiosity_equation_Horvat_nbody(
  std::vector<Tview_factor_nbody<T>> &Fmat,      // input
  std::vector<std::vector<T>> &R,
  std::vector<std::vector<T>> &F0,
  std::vector<std::vector<T>> &Fout,           // output
  const T & epsF = 1e-12,
  const T & max_iter = 100) {

  //
  // calculate limb-darkened emission of intrisic radiant exitance
  //  S0 = L_{LD} F0
  //
  int nb = F0.size();

  std::vector<std::vector<T>> S0(nb);

  for (int i = 0; i < nb; ++i) S0[i].resize(F0[i].size(), 0);

  for (auto && f: Fmat) S0[f.b1][f.i1] += f.F*F0[f.b2][f.i2];

  return solve_radiosity_equation_Horvat_nbody( Fmat, R, F0, S0, Fout, epsF, max_iter);
}
