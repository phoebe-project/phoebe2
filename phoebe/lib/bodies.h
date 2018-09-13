#pragma once

/*
  Examples of implicitely defined closed and connected surfaces in 3D
  using standard Euclidian coordinate system (x,y,z).

  The gradient should be outward from the surface, but it is not
  an requirement.

  Supported models:

  * Torus -- Ttorus <-- not simply connected, trouble for simple marching
  * Heart -- Theart <-- not entirely smooth, trouble for simple marching
  * Generalized Roche -- Tgen_roche
  * Rotating star -- Trotating_star

  Author: Martin Horvat, March, April, June 2016
*/

#include "utils.h"

/* ===================================================================
  Torus

  Defined implicitly by the constrain and the gradient of it

    (x^2 + y^2 + z^2 + R^2 - A^2)^2 - 4 R^2 (x^2 + y^2) = 0

  Explicitely it is defined as

    [x,y,z] = R_z(phi_1) [R + A cos(phi_2), 0, A cos(phi_2) ]

    R_z(phi)= [cos(phi), -sin(phi), 0; sin(phi), cos(phi), 0; 0, 0, 1]

 =================================================================== */

template <class T>
struct Ttorus {

  T R, A;

  /*
    Reading and storing the parameters
    params [0] = R
    params [1] = A
  */

  Ttorus(void *params){
    T *p = (T*)params;
    R = p[0];
    A = p[1];
  }

  /*
    Definition of the constrain and the gradient of it

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value

  */
  void grad(T r[3], T ret[4]){

    T r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2],
      R2 = R*R,
      A2 = A*A,
      f1 = - A2 - R2 + r2,
      f2 = - A2 + R2 + r2;

      ret[0] = 4*r[0]*f1;
      ret[1] = 4*r[1]*f1;
      ret[2] = 4*r[2]*f2;

      ret[3] = -4*R2*(r[0]*r[0] + r[1]*r[1]) + f2*f2;
  }

  /*
    Initial point
  */
  void init(T r[3], T n[3]){

    r[0] = R + A;
    r[1] = r[2] = 0;

    n[0] = 8*A*R*(A + R);
    n[1] = n[2] = 0;
  }
};


/* ===================================================================
  Sphere

  Defined of implicitly by a constrain

    x^2 + y^2 + z^2 - R^2 = 0
 =================================================================== */

template <class T>
struct Tsphere {

  double R, R2;

  /*
    Reading and storing the parameters
    params [0] = R  -- radius
  */

  Tsphere(void *params){
    T *p = (T*) params;
    R = *p;
    R2 = R*R;
  }

  /*
    Definition of the constrain and the gradient of it

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value

  */
  void grad(T r[3], T ret[4], const bool & precision = false){

    for (int i = 0; i < 3; ++i) ret[i] = 2*r[i];

    ret[3] = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - R2;
  }


  void grad_only(T r[3], T ret[3], const bool & precision = false){

    for (int i = 0; i < 3; ++i) ret[i] = 2*r[i];

  }

  /*
    Initial point
  */
  void init(T r[3], T n[3]){
    r[0] = R;
    r[1] = r[2] = 0;

    n[0] = 2*R;
    n[1] = n[2] = 0;
  }
};


/*  ===================================================================
   Heart

   Defined implicitly

   (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3 = 0

   The normal vector field has singularity at (0,0,+-1).
 =================================================================== */

template <class T>
struct Theart {

  /*
    Reading and storing the parameters
  */

  Theart(T *params){ }

  /* Definition of constrain and the gradient of it

    Input:
      r[3] = {x, y, z}

    Output:

      choice = 0: ret[4]
        {ret[0], ret[1], ret[2]} = grad-potential
        ret[3] = potential-value

      choice = 1: ret[3]
        {ret[0], ret[1], ret[2]} = grad-potential

  */
  void grad(T r[3], T ret[4]){

    T r02 = r[0]*r[0],
      r12 = r[1]*r[1],
      r22 = r[2]*r[2],
      r23 = r22*r[2],
      t = r02 + 9*r12/4 + r22 - 1,
      a = 3*t*t,
      b = t*t*t;

    ret[0] = 2*r[0]*(a - r23);
    ret[1] = 9*r[1]*(a - r23/20)/2;
    ret[2] = r[2]*(2*a - 3*(r02 + 9*r12/80)*r[2]);
    ret[3] = b - (r02 + 9*r12/80)*r23;
  }

  /*
    Initial point for surface triangulation
  */
  void init(T r[3], T n[3]){

    r[0] = r[1] = 0;
    r[2] = -1;

    n[0] = n[1] = 0;
    n[2] = -1;
  }
};

/*  ===================================================================
   Generalized Roche/Kopal potential

    Omega =
      1/rho
      + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
      + 1/2 F^2(1 + q) rho^2 (1 - nu^2)

      rho = sqrt(x^2 + y^2 + z^2)

      x = rho lambda  lambda = sin(theta) cos(phi)
      y = rho mu      mu = sin(theta) sin(phi)
      z = rho nu      nu = cos(theta)

    implicitly defining Roche lobe

      F =  Omega0 - Omega(x,y,z)  == 0

    The initial point is on the x-axis in the point (x0,0,0).

   params = {q, F, delta, Omega0, x0)
 =================================================================== */

template <class T>
struct Tgen_roche {

  T q, F, delta, Omega0,
    b, f0; // derived constants

  Tgen_roche() {}

  /*
    Reading and storing the parameters
  */

  Tgen_roche(T *params)
  : q(params[0]),
    F(params[1]),
    delta(params[2]),
    Omega0(params[3])
  {

    b = (1 + q)*F*F;
    f0 = 1/(delta*delta);
  }

  /*
    Definition of the reference minus potential.

    Input:
      r[3] = {x, y, z}

    Output:
      Omega0 - Omega(x,y,z)
  */

  T constrain(T r[3]) {
    return Omega0 - (
      1/utils::hypot3(r[0], r[1], r[2]) +
      q*(1/utils::hypot3(r[0] - delta, r[1], r[2]) - f0*r[0]) + b*(r[0]*r[0] + r[1]*r[1])/2
    );
  }

  /*
    Definition of the potential minus the reference and the
    gradient of it:

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega
  */


  void grad(T r[3], T ret[4], const bool & precision = false){

   if (precision) {

     long double
        x1 = r[0],
        x2 = x1 - delta,
        y = r[1],
        z = r[2],
        r1 = 1/utils::hypot3(x1, y, z),
        r2 = 1/utils::hypot3(x2, y, z),
        s = y*y + z*z,
        f1 = r1/(s + x1*x1),
        f2 = r2/(s + x2*x2);

      ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
      ret[1] = y*(f1 + q*f2 - b);
      ret[2] = z*(f1 + q*f2);
      ret[3] = Omega0 - (r1 + q*(r2 - f0*x1) + b*(x1*x1 + y*y)/2);
      return;
    }
    T x1 = r[0],
      x2 = r[0] - delta,
      y = r[1],
      z = r[2],
      r1 = 1/utils::hypot3(x1, y, z),
      r2 = 1/utils::hypot3(x2, y, z),
      s = y*y + z*z,
      f1 = r1/(s + x1*x1),
      f2 = r2/(s + x2*x2);

    ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
    ret[1] = y*(f1 + q*f2 - b);
    ret[2] = z*(f1 + q*f2);
    ret[3] = Omega0 - (r1 + q*(r2 - f0*x1) + b*(x1*x1 + y*y)/2);
  }

  /*
    Definition of the gradient of the negative potential

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */

  void grad_only(T r[3], T ret[3], const bool & precision = false){

    if (precision) {
      long double
        x1 = r[0],
        x2 = r[0] - delta,
        y = r[1],
        z = r[2],
        r1 = 1/utils::hypot3(x1, y, z),
        r2 = 1/utils::hypot3(x2, y, z),
        s = y*y + z*z,
        f1 = r1/(s + x1*x1),
        f2 = r2/(s + x2*x2);

      ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
      ret[1] = y*(f1 + q*f2 - b);
      ret[2] = z*(f1 + q*f2);
      return;
    }

    T x1 = r[0],
      x2 = r[0] - delta,
      y = r[1],
      z = r[2],
      r1 = 1/utils::hypot3(x1, y, z),
      r2 = 1/utils::hypot3(x2, y, z),
      s = y*y + z*z,
      f1 = r1/(s + x1*x1),
      f2 = r2/(s + x2*x2);

    ret[0] = -x1*(b - f1) + q*(f0 + f2*x2);
    ret[1] = y*(f1 + q*f2 - b);
    ret[2] = z*(f1 + q*f2);
  }


  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega

  */
  void hessian (T r[3], T H[3][3]){

    T x1 = r[0], x2 = x1 - delta, y = r[1], z = r[2],
      x12 = x1*x1,
      x22 = x2*x2,
      y2 = y*y,
      z2 = z*z,

      f11 = 1/utils::hypot3(x1, y, z),
      f21 = 1/utils::hypot3(x2, y, z),
      f12 = 1/(y2 + z2 + x12),
      f22 = 1/(y2 + z2 + x22),

      f13 = f12*f11,
      f23 = f22*f21,
      f15 = f12*f13,
      f25 = f22*f23;

    H[0][0] = -b + f13 + q*f23 - 3*(f15*x12 + f25*q*x22);
    H[0][1] = H[1][0] = -3*(f15*x1 + f25*q*x2)*y;
    H[0][2] = H[2][0] = -3*(f15*x1 + f25*q*x2)*z;
    H[1][1] = -b + f13 + q*f23 - 3*(f15 + f25*q)*y2;
    H[1][2] = H[2][1] = -3*(f15 + f25*q)*y*z;
    H[2][2] = f13 + q*f23 - 3*(f15 + f25*q)*z2;
  }

};

/* ===================================================================
  Rotating star

  Defined of implicitly by a constrain

    1/r + 1/2 omega^2 (x^2 + y^2) = Omega_0
 =================================================================== */

template <class T>
struct Trot_star {

  T omega, Omega0, w2;

  /*
    Reading and storing the parameters
    params[0] = omega0
    params[1] = Omega0

  */

  Trot_star(T *params) : omega(params[0]), Omega0(params[1]) {

    w2 = omega*omega;
  }

  /*
    Definition of the potential minus the reference.

    Input:
      r[3] = {x, y, z}

    Output:
      Omega0 - Omega(x,y,z)
  */

  T constrain(T r[3]) {
    return
      Omega0 - (1/utils::hypot3(r) + w2*(r[0]*r[0] + r[1]*r[1])/2);
  }
  /*
    Definition of the potential minus the reference and the
    gradient of it:

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega
  */


  void grad(T r[3], T ret[4], const bool & precision = false){

    if (precision) {

      long double
        x = r[0],
        y = r[1],
        z = r[2],
        f = 1/utils::hypot3(x, y, z),
        r1 = std::pow(f, 3);

      ret[0] = (-w2 + r1)*x;
      ret[1] = (-w2 + r1)*y;
      ret[2] = z*r1;
      ret[3] = Omega0 - (f + w2*(x*x + y*y)/2);

      return;
    }

    T x = r[0],
      y = r[1],
      z = r[2],
      f = 1/utils::hypot3(x, y, z),
      r1 = std::pow(f, 3);

    ret[0] = (-w2 + r1)*x;
    ret[1] = (-w2 + r1)*y;
    ret[2] = z*r1;
    ret[3] = Omega0 - (f + w2*(x*x + y*y)/2);
  }

  /*
    Definition of the gradient of the negative potential

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */

  void grad_only(T r[3], T ret[3], const bool & precision = false){

    if (precision){
      long double
        x = r[0],
        y = r[1],
        z = r[2],
        f = 1/utils::hypot3(x, y, z),
        r1 = std::pow(f, 3);

      ret[0] = (-w2 + r1)*x;
      ret[1] = (-w2 + r1)*y;
      ret[2] = z*r1;
      return;
    }

    T x = r[0],
      y = r[1],
      z = r[2],
      f = 1/utils::hypot3(x, y, z),
      r1 = std::pow(f, 3);

    ret[0] = (-w2 + r1)*x;
    ret[1] = (-w2 + r1)*y;
    ret[2] = z*r1;
  }


  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega

  */
  void hessian (T r[3], T H[3][3]){

    T x = r[0], y = r[1], z = r[2],
      x2 = x*x,
      y2 = y*y,
      z2 = z*z,

      f = 1/utils::hypot3(x, y, z),
      f2 = 1/(y2 + z2 + x2),

      f3 = f2*f,
      f5 = f2*f3;

    H[0][0] = f3 - w2 - 3*f5*x2;
    H[0][1] = H[1][0] = -3*f5*x*y;
    H[0][2] = H[2][0] = -3*f5*x*z;
    H[1][1] = f3 - w2 - 3*f5*y2;
    H[1][2] = H[2][1] = -3*f5*y*z;
    H[2][2] = f3 - 3*f5*z2;
  }

};


/* ===================================================================
  Rotating star with misaligned spin w.r.t. to x,y,z

    Omega(r) = 1/|r| + 1/2 omega^2 |r - s (r.s)|^2
      r = (x, y, z)
      s = (sx, sy, sz)      |s| = 1

    gradOmega = -r/|r|^3 + omega^2 (r - s(r.s))

  defined of implicitly by a constrain

    constrain = Omega_0 - Omega(r) == 0
 =================================================================== */

template <class T>
struct Tmisaligned_rot_star {

  T omega, Omega0, s[3], omega2;

  /*
    Reading and storing the parameters
    params[0] = omega0
    params[1] = spin[0]
    params[2] = spin[1]
    params[3] = spin[2]
    params[4] = Omega0

  */

  Tmisaligned_rot_star(T *params)
    : omega(params[0]),
      Omega0(params[4]) {

    omega2 = omega*omega;
    for (int i = 0; i < 3; ++i) s[i] = params[i+1];
  }

  /*
    Definition of the potential minus the reference.

    Input:
      r[3] = {x, y, z}

    Output:
      Omega0 - Omega(x,y,z)
  */

  T constrain(T r[3]) {

    T w = utils::dot3D(r, s),   // = r.s
      f = 1/utils::hypot3(r),   // 1/|r|
      g[3];

    utils::fma3D(r, s, -w, g);  // g = r - w*s

    return Omega0 - f - 0.5*omega2*utils::norm2(g);
  }
  /*
    Definition of the potential minus the reference and the
    gradient of it:

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega
  */


  void grad(T r[3], T ret[4], const bool & precision = false){

    if (precision) {
      long double rl[3] = {r[0], r[1], r[2]}, gl[3];

      // g = r - (r.s)*s
      utils::fma3D(rl, s, -utils::dot3D(rl, s), gl);

      long double fl = 1/utils::hypot3(rl);  // = 1/|r|

      // ret[3] = Omega0 - 1/|r| - 1/2 omega^2 |g|^2
      ret[3] = Omega0 - fl - 0.5*omega2*utils::norm2(gl);

      fl *= fl*fl;
      for (int i = 0; i < 3; ++i) ret[i] = fl*rl[i] - omega2*gl[i];

      return;
    }

    T g[3];

    // g = r - (r.s)*s
    utils::fma3D(r, s, -utils::dot3D(r, s), g);

    T f = 1/utils::hypot3(r);     // = 1/|r|

    // ret[3] = Omega0 - 1/|r| - 1/2 omega^2 |g|^2
    ret[3] = Omega0 - f - 0.5*omega2*utils::norm2(g);

    f *= f*f;

    // ret = r/|r|^3 - omega^2 g
    for (int i = 0; i < 3; ++i) ret[i] = f*r[i] - omega2*g[i];
  }

  /*
    Definition of the gradient of the negative potential

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */

  void grad_only(T r[3], T ret[3], const bool & precision = false){

    if (precision) {
      long double rl[3] = {r[0], r[1], r[2]}, gl[3];

      // g = r - (r.s)*s
      utils::fma3D(rl, s, -utils::dot3D(rl, s), gl);

      long double fl = 1/utils::hypot3(rl);  // = 1/|r|

      fl *= fl*fl;
      for (int i = 0; i < 3; ++i) ret[i] = fl*rl[i] - omega2*gl[i];

      return;
    }

    T g[3];

    // g = r - (r.s)*s
    utils::fma3D(r, s, -utils::dot3D(r, s), g);

    T f = 1/utils::hypot3(r);     // = 1/|r|

    f *= f*f;

    // ret = r/|r|^3 - omega^2 g
    for (int i = 0; i < 3; ++i) ret[i] = f*r[i] - omega2*g[i];
  }


  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega

  */
  void hessian (T r[3], T H[3][3]){

    T x = r[0], y = r[1], z = r[2],
      x2 = x*x, y2 = y*y, z2 = z*z,

      f = 1/utils::hypot3(x, y, z),
      f2 = f*f, f3 = f2*f, f5 = f2*f3;

    H[0][0] = f3 - 3*f5*x2 - omega2*(1 - s[0]*s[0]);
    H[0][1] = H[1][0] = -3*f5*x*y + omega2*s[0]*s[1];
    H[0][2] = H[2][0] = -3*f5*x*z + omega2*s[0]*s[2];
    H[1][1] = f3 - 3*f5*y2 - omega2*(1 - s[1]*s[1]);
    H[1][2] = H[2][1] = -3*f5*y*z + omega2*s[1]*s[2];
    H[2][2] = f3 - 3*f5*z2 - omega2*(1 - s[2]*s[2]);
  }

};


/* ===================================================================
  Generalizd Roche potential with misaligned binary system in rotated
  coordinate system.

  Defined of implicitly by a constrain

    Omega(x,y,z,params) =
      1/r1 + q(1/r2 - x/delta^2) + 1/2 (1+q) F^2 [(x cos theta' - z sin theta')^2 + y^2]

    constrain = Omega0 - Omega(x,y,z,params) = 0

    r1 = sqrt(x^2 + y^2 + z^2)
    r2 = sqrt((x - delta)^2 + y^2+z^2)

 Ref:
  * Avni Y and Schiller N,
    Generalized Roche potential for misaligned binary systems -
    Properties of the critical lobe,
    Astrophysical Journal, Part 1, vol. 257, June 15, 1982, p. 703-714.
 =================================================================== */

template <class T>
struct Tmisaligned_rotated_roche {

  T q, F, delta, theta, Omega0,
    b, f0, s, c;     // derived

  /*
    Reading and storing the parameters
    params[0] = q
    params[1] = F
    params[2] = delta
    params[3] = theta
    params[4] = Omega0
  */

  Tmisaligned_rotated_roche(T *params) {

    q = params[0];
    F = params[1];
    delta = params[2];
    theta = params[3];
    Omega0 = params[4];

    f0 = 1/(delta*delta);
    b = (1 + q)*F*F;
    utils::sincos(theta, &s, &c);
  }

  /*
    Definition of the potential minus the reference.

    Input:
      r[3] = {x, y, z}

    Output:
      Omega0 - Omega(x,y,z)
  */

  T constrain(T r[3]) {

    T x = r[0], y = r[1], z = r[2],
      x_ = x*c - z*s,
      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x - delta, y, z);

    return Omega0 - (1/r1 + q*(1/r2 - f0*x) + 0.5*b*(x_*x_ + y*y));
  }

  /*
    Definition of the potential minus the reference and the
    gradient of it:

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega
  */


  void grad(T r[3], T ret[4], const bool & precision = false){

    if (precision) {

      long double
        x = r[0], y = r[1], z = r[2],
        x1 = x - delta,
        x_ = x*c - z*s,
        r1 = utils::hypot3(x, y, z),
        r2 = utils::hypot3(x1, y, z),
        f1 = 1/r1, f13 = f1*f1*f1,
        f2 = 1/r2, f23 = f2*f2*f2,
        tmp = f13 + f23*q;


      ret[0] = f13*x + q*(f0 + f23*x1) - b*c*x_,
      ret[1] = (tmp - b)*y;
      ret[2] = tmp*z + b*s*x_;
      ret[3] = Omega0 - (f1 + q*(f2 - f0*x) + 0.5*b*(x_*x_ + y*y));

      return;
    }

    T x = r[0], y = r[1], z = r[2],
      x1 = x - delta,
      x_ = x*c - z*s,
      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x1, y, z),
      f1 = 1/r1, f13 = f1*f1*f1,
      f2 = 1/r2, f23 = f2*f2*f2,
      tmp = f13 + f23*q;

    ret[0] = f13*x + q*(f0 + f23*x1) - b*c*x_,
    ret[1] = (tmp - b)*y;
    ret[2] = tmp*z + b*s*x_;
    ret[3] = Omega0 - (f1 + q*(f2 - f0*x) + 0.5*b*(x_*x_ + y*y));
  }

  /*
    Definition of the gradient of the negative potential

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */

  void grad_only(T r[3], T ret[3], const bool & precision = false){

    if (precision) {
      long double
        x = r[0], y = r[1], z = r[2],
        x1 = x - delta,
        x_ = x*c - z*s,
        r1 = utils::hypot3(x, y, z),
        r2 = utils::hypot3(x1, y, z),
        f13 = 1/(r1*r1*r1),
        f23 = 1/(r2*r2*r2),
        tmp = f13 + f23*q;

      ret[0] = f13*x + q*(f0 + f23*x1) - b*c*x_,
      ret[1] = (tmp - b)*y;
      ret[2] = tmp*z + b*s*x_;
      return;
    }

    T x = r[0], y = r[1], z = r[2],
      x1 = x - delta,
      x_ = x*c - z*s,
      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x1, y, z),
      f13 = 1/(r1*r1*r1),
      f23 = 1/(r2*r2*r2),
      tmp = f13 + f23*q;

    ret[0] = f13*x + q*(f0 + f23*x1) - b*c*x_,
    ret[1] = (tmp - b)*y;
    ret[2] = tmp*z + b*s*x_;
  }


  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega

  */
  void hessian (T r[3], T H[3][3]){

    T x = r[0], y = r[1], z = r[2],
      x1 = x - delta,
      w = y*y + z*z,
      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x1, y, z),
      f1 = 1/r1, f13 = f1*f1*f1, f15 = f13*f1*f1,
      f2 = 1/r2, f23 = f2*f2*f2, f25 = f23*f2*f2,
      tmp1 = f15 + f25*q, tmp2 = f13 + f23*q;

    H[0][0] = -b*c*c - 2*tmp2  + 3*tmp1*w;
    H[0][1] = H[1][0] = -3*(delta*f15 + tmp1*x1)*y;
    H[0][2] = H[2][0] = b*c*s - 3*(delta*f15 + tmp1*x1)*z;
    H[1][1] =  -b + tmp2 - 3*tmp1*y*y;
    H[1][2] = H[2][1] = -3*tmp1*y*z;
    H[2][2] = tmp2 - b*s*s - 3*tmp1*z*z;
  }
};


/* ===================================================================
  Generalizd Roche potential with misaligned binary system in canonical
  coordinate system.

  Defined of implicitly by a constrain

    Omega(x,y,z,params) =
      1/r1 + q(1/r2 - x/delta^2) + 1/2 (1+q) F^2 [r - s(s.r)]

    constrain = Omega0 - Omega(x,y,z,params) = 0
    r = (x,y,z)
    r1 = sqrt(x^2 + y^2 + z^2)
    r2 = sqrt((x - delta)^2 + y^2+z^2)

 Ref:
  * Avni Y and Schiller N,
    Generalized Roche potential for misaligned binary systems -
    Properties of the critical lobe,
    Astrophysical Journal, Part 1, vol. 257, June 15, 1982, p. 703-714.
 =================================================================== */

template <class T>
struct Tmisaligned_roche {

  T q, F, delta, s[3], Omega0,
    b, f0;     // derived

  /*
    Reading and storing the parameters
    params[0] = q
    params[1] = F
    params[2] = delta
    params[3] = s[0]
    params[4] = s[1]
    params[5] = s[2]
    params[6] = Omega0
  */

  Tmisaligned_roche(T *params) {
   
    q = params[0];
    F = params[1];
    delta = params[2];
    s[0] = params[3];
    s[1] = params[4];
    s[2] = params[5];
    Omega0 = params[6];

    f0 = 1/(delta*delta);
    b = (1 + q)*F*F;
  }

  /*
    Definition of the potential minus the reference.

    Input:
      r[3] = {x, y, z}

    Output:
      Omega0 - Omega(x,y,z)
  */

  T constrain(T r[3]) {

    T x = r[0], y = r[1], z = r[2],
      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x - delta, y, z),
      sr = utils::dot3D(s, r),
      rp = 0;

    for (int i = 0; i < 3; ++i) rp += utils::sqr(r[i] - sr*s[i]);

    return Omega0 - (1/r1 + q*(1/r2 - f0*x) + 0.5*b*rp);
  }

  /*
    Definition of the potential minus the reference and the
    gradient of it:

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[4]:
        {ret[0], ret[1], ret[2]} = -grad-potential Omega
        ret[3] = Omega0 - Omega
  */


  void grad(T r[3], T ret[4], const bool & precision = false){

    if (precision) {

      long double
        x = r[0], y = r[1], z = r[2],
        sx = s[0], sy = s[1], sz = s[2],
        r1 = utils::hypot3(x, y, z),
        r2 = utils::hypot3(x - delta, y, z),
        sr = sx*x + sy*y + sz*z,
        f1 = 1/r1, f13 = f1*f1*f1,
        f2 = 1/r2, f23 = f2*f2*f2,
        tmp = f13 + f23*q,
        p = 0;

        for (int i = 0; i < 3; ++i) p += utils::sqr(r[i] - sr*s[i]);

      ret[0] = (f0 - delta*f23)*q + b*sr*sx + (-b + tmp)*x;
      ret[1] = b*sr*sy - b*y + tmp*y;
      ret[2] = b*sr*sz - b*z + tmp*z;
      ret[3] = Omega0 - (1/r1 + q*(1/r2 - f0*x) + 0.5*b*p);

      return;
    }

    T x = r[0], y = r[1], z = r[2],
      sx = s[0], sy = s[1], sz = s[2],

      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x - delta, y, z),

      sr = sx*x + sy*y + sz*z,

      f1 = 1/r1, f13 = f1*f1*f1,
      f2 = 1/r2, f23 = f2*f2*f2,

      tmp = f13 + f23*q,
      p = 0;

      for (int i = 0; i < 3; ++i) p += utils::sqr(r[i] - sr*s[i]);

      ret[0] = (f0 - delta*f23)*q + b*sr*sx + (-b + tmp)*x;
      ret[1] = b*sr*sy - b*y + tmp*y;
      ret[2] = b*sr*sz - b*z + tmp*z;
      ret[3] = Omega0 - (f1 + q*(f2 - f0*x) + 0.5*b*p);
  }

  /*
    Definition of the gradient of the negative potential

      -grad-potential Omega

    Minus guaranties that the normal points outward from the
    iso-potential surfaces.

    Input:
      r[3] = {x, y, z}

    Output:

      ret[3]:
        {ret[0], ret[1], ret[2]} = grad-potential
  */

  void grad_only(T r[3], T ret[3], const bool & precision = false){

    if (precision) {
      long double
        x = r[0], y = r[1], z = r[2],
        sx = s[0], sy = s[1], sz = s[2],

        r1 = utils::hypot3(x, y, z),
        r2 = utils::hypot3(x - delta, y, z),

        sr = sx*x + sy*y + sz*z,

        f1 = 1/r1, f13 = f1*f1*f1,
        f2 = 1/r2, f23 = f2*f2*f2,

        tmp = f13 + f23*q;

      ret[0] = (f0 - delta*f23)*q + b*sr*sx + (-b + tmp)*x;
      ret[1] = b*sr*sy - b*y + tmp*y;
      ret[2] = b*sr*sz - b*z + tmp*z;

      return;
    }

    T x = r[0], y = r[1], z = r[2],
      sx = s[0], sy = s[1], sz = s[2],

      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x - delta, y, z),

      sr = sx*x + sy*y + sz*z,

      f1 = 1/r1, f13 = f1*f1*f1,
      f2 = 1/r2, f23 = f2*f2*f2,

      tmp = f13 + f23*q;

      ret[0] = (f0 - delta*f23)*q + b*sr*sx + (-b + tmp)*x;
      ret[1] = b*sr*sy - b*y + tmp*y;
      ret[2] = b*sr*sz - b*z + tmp*z;
  }


  /*
    Calculate Hessian matrix of the constrain Omega0 - Omega:
    resulting:
      H_{ij} = - partial_i partial_j Omega

  */
  void hessian (T r[3], T H[3][3]){

    T x1 = r[0], y1 = r[1], z1 = r[2], x2 = x1 - delta,
      sx = s[0], sy = s[1], sz = s[2],

      x12 = x1*x1, y12 = y1*y1, z12 = z1*z1, x22 = x2*x2,
      sx2 = sx*sx, sy2 = sy*sy, sz2 = sz*sz,

      r1 = utils::hypot3(r),
      r2 = utils::hypot3(x2, y1, z1),

      f1 = 1/r1, f13 = f1*f1*f1, f15 = f13*f1*f1,
      f2 = 1/r2, f23 = f2*f2*f2, f25 = f23*f2*f2;

    H[0][0] = f13 + b*(-1 + sx2) - 3*f15*x12 + f25*q*(-2*x22 + y12 + z12);
    H[0][1] = H[1][0] = b*sx*sy - 3*(f15*x1 + f25*q*x2)*y1;
    H[0][2] = H[2][0] = b*sx*sz - 3*(f15*x1 + f25*q*x2)*z1;
    H[1][1] = b*(-1 + sy2) + f15*(x12 - 2*y12 + z12) + f25*q*(x22 - 2*y12 + z12);
    H[1][2] = H[2][1] = b*sy*sz - 3*(f15 + f25*q)*y1*z1;
    H[2][2] = b*(-1 + sz2) + f15*(x12 + y12 - 2*z12) + f25*q*(x22 + y12 - 2*z12);
  }
};

