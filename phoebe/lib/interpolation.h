#pragma once
/*

  Library for interpolation of data. Currently supporting:

  * multi-dimensional linear interpolation based on gridded data

  Author: Martin Horvat, September 2016
*/

#include <iostream>
#include <cmath>

#include "utils.h"

/*
  Class for multi-dimensional linear interpolation based on gridded data.
  Gridded data means that the values of a function are given on a
  grid defined by values on axes. We want to have a interpolating function

     Interpolation(x) in R^{Nv}

  for x in R^{Na}.

  Terminology:

    axis: vector of ticks on ith axis  (in ascending order)
      A_i = (a_{i,0},.., a_{i, L_i-})

    axes: vector of all axis
      A = (A_i: i = 0, .., Na)    aka "the axes"

    grid of points: set of points of dimension Na

      Gv = A_1 x A_2 x ... A_{Na} in R^{Na}

    a point from Gv is just

      r(i_0, .., i_{Na-1}) = {a_{0,i_0}, a_{1,i_1}, ..., a_{Na-1, i_{Na-1}}

    grid of indices: is a set of indices labeling

      Gi = I_1 x I_2 x ... x L_{Na-1}

    with interval

      I_i = [0, .., L_i-1]

    gridded data: is a tensor of given data at grid points

      D[u in Gi] = V_u in R^{Nv}
      D tensor of rank Na of vector value

    or

      G[(u,i) in Gg] = V_{u,i}  in R   aka "the grid"
      Gg = Gi x [0,..,Nv-1],  G tensor of rank Na+1

    where V_u is known values known at r(u) with u = (i_0, .., i_{Na-1})

  Example: we have the following vertices with corresponding values:

    v0 = (0, 2), f(v0) = 5
    v1 = (0, 3), f(v0) = 6
    v2 = (1, 3), f(v0) = 7
    v3 = (1, 2), f(v0) = 8

  This means
    A = ([0.0, 1.0], [2.0, 3.0])
    D = [[[5.0], [6.0]], [[7.0], [8.0]]]

  Notes:
  * !!!NOT-THREAD SAVE!!!
  * Algorithm was developed by A. Prsa for Phoebe 0.x
*/

template <class T>
struct Tlinear_interpolation {

  // borrowed data

  int
    Na,       // number of axes
    Nv,       // dimension of response
    *L;       // array of number of values per axis

  T **A, *G;

  // own data

  T *lo, *hi, *prod, **n, **fvv, *__ret;

  int *axelen, *axidx, Nf;

  /*
    Initialization of the interpolation.

    Input:
      Na - number of axes
      Nv - number of values in data points/dimension of interpolated values
      L - numbers of points on axes
      A - pointers to values on axes given in ascending order !!
      G - pointer the values of the grid (tensor

    Note:
    * data addressed by arguments is not copied
  */
  Tlinear_interpolation(const int &Na, const int &Nv, int *L, T **A, T *G)
   : Na(Na), Nv(Nv), L(L), A(A), G(G), Nf(1<< Na) {

    lo = new T [3*Na + Nv];
    hi = lo + Na,
    prod = hi + Na,
    __ret = prod + Na;

    axidx = new int [Na];

    n = utils::matrix<T>(Nf, Na);   // space to hold all the nodes
    fvv = utils::matrix<T>(Nf, Nv); // function value arrays
  }


  ~ Tlinear_interpolation() {
    delete [] lo;
    delete [] axidx;

    utils::free_matrix(n);
    utils::free_matrix(fvv);
  }

  /*
    Performing interpolation

      r = interpolation(x)

    Input:
      x - array of dimension Na

    Output:
      r - array of dimension Nv

    Return:
      false - out of bounds, true - otherwise
  */
  bool get(T *x, T *r) {

    T *g;

    int j, k, l, m, o, idx;

    // Run the axes first to make sure interpolation is possible.
    for (j = Na-1; j >= 0; --j) {
      axidx[j] = utils::flt(x[j], A[j], L[j]);

      // AN OUT-OF-BOUNDS SITUATION -- both sides handled.
      if (axidx[j] < 1) {
        for (l = 0; l < Nv; ++l) r[l] = std::numeric_limits<T>::quiet_NaN();
        return false;
      }
    }

    for (j = Na-1; j >= 0; --j) {
      lo[j] = A[j][axidx[j]-1];
      hi[j] = A[j][axidx[j]];
      prod[j] = (j == Na - 1) ? 1.0 : prod[j+1]*L[j+1];
    }

    for (k = 0; k < Nf; ++k) {

      for (j = idx = 0; j < Na; ++j)
        idx += (axidx[j] - 1 + ((k >> j) & 1) )*prod[j];

      g = G + idx*Nv;
      for (l = 0; l < Nv; ++l) fvv[k][l] = g[l];
    }

    // Populate the nodes:
    for (k = 0; k < Na; ++k)
      for (j = 0; j < Nf; ++j)
        n[j][k] = lo[k] + ((j >> k) & 1)*(hi[k] - lo[k]);

    for (k = 0, o = Na - 1, m = Nf >> 1; k < Na; ++k, --o, m >>=1)
      for (j = 0; j < m; ++j)
        for (l = 0; l < Nv; ++l)
          fvv[j][l] += (x[o] - n[j][o])*(fvv[j + m][l] - fvv[j][l])/(n[j + m][o] - n[j][o]);

    for (l = 0; l < Nv; ++l) r[l] = fvv[0][l];

    return true;
  }

  /*
    Performing interpolation

      r = interpolation(x)

    and returning only i-th elements

    Input:
      x - array of dimension Na

    Return:
      r[i]
  */

  T get(T *r, const int &i) {

    get(r, __ret);

    return __ret[i];
  }

};
