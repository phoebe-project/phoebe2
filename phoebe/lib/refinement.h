#pragma once

/*
  Library for mesh refinement techniques. Currently supporting:

    Global mid-edge (midpoint)/Loops subdivision of triangles

  Author: Martin Horvat
*/

#include <iostream>
#include <cmath>
#include <vector>
#include <map>

#include "utils.h"
#include "triang_mesh.h"

/*
  Perform a mid-edge (midpoint)/Loops subdivision of all triangles
  and reproject onto surface along the normal field of implicitly
  defined body

  Input:
    nr_divs - number of subdivisions
    params - body parameters
    V - vector of vertices
    NatV - vector of normals at vertices
    G - vector of gradient norms
    Tr - vector of vertex indices forming triangle faces

  Output:
    V - vector of vertices
    NatV - vector of normals at vertices
    G - vector of gradient norms
    Tr - vector of vertex indices forming triangle faces

*/
template <class T, class Tbody>
bool mesh_refine_midedge_subdivision(
  int nr_divs,
  T *params,
  std::vector<T3Dpoint<T>> &V,
  std::vector<T3Dpoint<T>> &NatV,
  std::vector<T> &G,
  std::vector<T3Dpoint<int>> &Tr ) {

  int max_iter = 100;
  const T eps = 10*std::numeric_limits<T>::epsilon();
  const T min = 10*std::numeric_limits<T>::min();

  Tbody body(params);

  bool ok = true;

  T u[3][3], *v[3];

  std::vector<T3Dpoint<int>> Tr1;

  for (int i = 0; ok && i < nr_divs; ++i) {

    int Nv = V.size();

    Tr1.clear();

    // database of mid-points
    std::map<int, int> db;
    std::map<int,int>::iterator it;

    for (auto && tr : Tr) {

      // link the existing vertices of triangle
      for (int j = 0; j < 3; ++j) v[j] = V[tr[j]].data;

      // determine the index of midpoints
      int idx[3] = {-1,-1, -1};
      {

        int k = V.size(), index;

        // pair (tr[0],tr[1])
        index = (tr[0] < tr[1] ? tr[0]*Nv  + tr[1] : tr[1]*Nv  + tr[0]);
        if ((it = db.find(index)) != db.end()) {
          idx[0] = it->second;
        } else {
          db[index] = k++;
          for (int j = 0; j < 3; ++j) u[0][j] = (v[0][j] + v[1][j])/2;
        }

        // pair (tr[1],tr[2])
        index = (tr[1] < tr[2] ? tr[1]*Nv  + tr[2] : tr[2]*Nv  + tr[1]);
        if ((it = db.find(index)) != db.end()) {
          idx[1] = it->second;
        } else {
          db[index] = k++;
          for (int j = 0; j < 3; ++j) u[1][j] = (v[1][j] + v[2][j])/2;
        }

        // pair (tr[0],tr[2])
        index = (tr[0] < tr[2] ? tr[0]*Nv  + tr[2] : tr[2]*Nv  + tr[0]);
        if ((it = db.find(index)) != db.end()) {
          idx[2] = it->second;
        } else {
          db[index] = k++;
          for (int j = 0; j < 3; ++j) u[2][j] = (v[0][j] + v[2][j])/2;
        }

      }

      // project new vertices onto isosurface by
      // Newton-Raphson iteration to solve F(u_k - t grad(F))=0
      {
        int nr_iter;

        T g[4], *r, t, dr1, r1, fac;

        for (int j = 0; ok && j < 3; ++j) if (idx[j] < 0) {

          r = u[j];

          nr_iter = 0;

          do {

            // g = (grad F, F)
            body.grad(r, g);

            // fac = F/|grad(F)|^2
            fac = g[3]/utils::norm2(g);

            // dr = F/|grad(F)|^2 grad(F)
            // r' = r - dr
            dr1 = r1 = 0;
            for (int k = 0; k < 3; ++k) {

              r[k] -= (t = fac*g[k]);

              // calc. L_infty norm of vec{dr}
              if ((t = std::abs(t)) > dr1) dr1 = t;

              // calc L_infty of of vec{r'}
              if ((t = std::abs(r[k])) > r1) r1 = t;
            }

          } while (dr1 > eps*r1 + min && ++nr_iter < max_iter);

          ok = (nr_iter < max_iter);

          if (ok) {

            // insert new vertex
            idx[j] = V.size();
            V.emplace_back(r);

            // insert g-norm
            fac = utils::hypot3(g[0], g[1], g[2]);
            G.emplace_back(fac);

            // insert normal
            fac = 1/fac;
            for (int k = 0; k < 3; ++k) g[k] *= fac;
            NatV.emplace_back(g);
          }
        }
      }

      if (ok) {
        // insert triangles, note:
        // (0+1)/2 ~ idx[0], (1+2)/2 ~ idx[1], (0+2)/2 ~ idx[2]

        Tr1.emplace_back(tr[0],  idx[0],  idx[2]);
        Tr1.emplace_back(idx[0], tr[1],   idx[1]);
        Tr1.emplace_back(idx[2], idx[1],   tr[2]);
        Tr1.emplace_back(idx[0], idx[1],  idx[2]);
      }

    }

    Tr = Tr1;
  }

  return ok;
}
