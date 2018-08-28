#pragma once

/*
  Support for calculations of "smoooth" horizon on the lobes by integration
  on the surface some initial point.


  Author: Martin Horvat, August 2016
*/

#include "cmath"
#include "utils.h"

template <class T, class Tbody>
struct Thorizon: public Tbody {


  Thorizon (T *param) : Tbody(param){ }


  /*
    Derivative the curve along the horizon
  */

  void derivative(T r[3], T F[3], T view[3]){

    T h[3], g[3], H[3][3];

    // calculate gradient
    this->grad_only(r, g);

    // calc hessian matrix H
    this->hessian(r, H);

    // calc h = H view
    utils::dot3D(H, view, h);

    // cross product F = h x g
    utils::cross3D(h, g, F);

    // normalize for normal parametrization
    T f = 1/utils::hypot3(F);
    for (int i = 0; i < 3; ++i) F[i] *= f;
  }

  /*
    RK4 step
  */

  void RK4step(T r[3], T dt, T view[3]){

    T r1[3], k[4][3];

    derivative(r, k[0], view);

    for (int i = 0; i < 3; ++i) r1[i] = r[i] + (k[0][i] *= dt)/2;

    derivative(r1, k[1], view);

    for (int i = 0; i < 3; ++i) r1[i] = r[i] + (k[1][i] *= dt)/2;

    derivative(r1, k[2], view);

    for (int i = 0; i < 3; ++i) r1[i] = r[i] + (k[2][i] *= dt);

    derivative(r1, k[3], view);

    for (int i = 0; i < 3; ++i) k[3][i] *= dt;

    for (int i = 0; i < 3; ++i)
      r[i] += (k[0][i] + 2*(k[1][i] + k[2][i]) + k[3][i])/6;
  }


  /*
    Calculate the horizon on the body in the light is coming from direction v.

    Input:
      view - direction of the viewer/or the light
      p - point on the horizon
      dt - step in length
      max_iter - maximal number of iterations

    Output:
      H - trajectory on surface of the body
  */

  bool calc(
    std::vector<T3Dpoint<T>> & H,
    T view[3],
    T p[3],
    const T &dt,
    const int max_iter = 100000) {

    T f[2] = {0,0}, r[2][3], F[3];

    derivative(p, F, view);

    for (int i = 0; i < 3; ++i)  r[0][i] = p[i];

    int it = 0;

    do {

      for (int i = 0; i < 3; ++i) r[1][i] = r[0][i];
      RK4step(r[0], dt/2, view);
      RK4step(r[0], dt/2, view);
      RK4step(r[1], dt, view);

      // 2 x RK4 => error of O(dt^6)
      for (int i = 0; i < 3; ++i) r[0][i] += (r[0][i] - r[1][i])/15;

      f[1] = f[0], f[0] = 0;
      for (int i = 0; i < 3; ++i) f[0] += (r[0][i] - p[i])*F[i];

      // check the crossing through SOC
      if (f[1] < 0 && f[0] > 0)
        break;
      else
        H.emplace_back(r[0]);

    } while (++it < max_iter);

    return (it < max_iter);
  }
};
