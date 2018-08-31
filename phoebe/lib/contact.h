#pragma once

/*
  Library dealing for dealing the !!contact case!! of Roche lobes defined as iso-surface of Kopal potential

  Omega =
    1/rho
    + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
    + 1/2 F^2(1 + q) rho^2 (1 - nu^2)
  
  with 
    F = 1 !!!!
  
  where position r=(x,y,z) in spherical coordinates is given as

    x = rho lambda      lambda = sin(theta) cos(phi)
    y = rho  mu         mu = sin(theta) sin(phi)
    z = rho nu          nu = cos(theta)

 
  Contact lobe exists for
  
    max(Omega(L2), Omega(L3)) <= Omega(r) <= Omega(L1) 
  
  Author: Martin Horvat, August 2018
*/

#include "gen_roche.h"

namespace contact {
  
  /*
    Calculate the minimal distance r of the neck from x axis of the contact
    Roche lobe at angle phi from y axis:

        Omega_0 = Omega(x, r cos(phi), r sin(phi))
    
    assuming F = 1.

    Input:
      cos_phi - cosine of the angle
      q - mass ratio
      d - separation between the two objects
      Omega0 - value potential
        for minimal distance in
          xy plane phi = 0
          xz plane phi = pi/2
      it_max - maximal number of iterations
  
    Output:
      u[2] = {xmin, rmin}
    
    Return:
      true if converged in given maximal number of iterations
      
  */ 
  template <class T>
  bool neck_min( 
    T u[2],
    const T & cos_phi,
    const T & q,
    const T & d,
    const T & Omega0,
    const int & it_max = 20) {
 
    const T eps = 4*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    int it = 0;
    
    T  b = (1+q)*d*d*d, W0 = Omega0*d, c = cos_phi*cos_phi, 
      t1, t2, s11, s12, s13, s15, s21, s22, s23, s25,
      x1, du[2], det, W, Wr, Wx, Wxx, Wrx;

    u[0] = gen_roche::lagrange_point_L1(q, T(1), d)/d;
    u[1] = 0;

    // Newton-Raphson iteration
    do {
      x1 = u[0]-1, t1 = u[0]*u[0], t2 = x1*x1;

      s12 = 1/(u[1] + t1), s11 = std::sqrt(s12), s13 = s11*s12, s15 = s13*s12;
      s22 = 1/(u[1] + t2), s21 = std::sqrt(s22), s23 = s21*s22, s25 = s23*s22;

      W = s11 + q*(s21 - u[0]) + (b*(t1 + c*u[1]))/2 - W0;
      Wx =-(q*(1 + s23*x1)) + b*u[0] - s13*u[0];
      Wr = (b*c - s13 - q*s23)/2;
      Wrx = 3*(q*s25*x1 + s15*u[0])/2;
      Wxx = b - s13 + 3*s15*t1 + q*s23*(-1 + 3*s22*t2);

      det = Wx*Wrx - Wr*Wxx;

      u[0] -= (du[0] = (Wrx*W - Wr*Wx)/det);
      u[1] -= (du[1] = (Wx*Wx - Wxx*W)/det);

    } while (
      std::abs(du[0]) > eps*std::abs(u[0]) + min &&
      std::abs(du[1]) > eps*std::abs(u[1]) + min &&
      ++it < it_max);

  
    u[0] *= d;
    u[1] = d*std::sqrt(u[1]);

    return it < it_max;  
  }
  
  
  
} // namespace contact
