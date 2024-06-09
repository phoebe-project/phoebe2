#pragma once

/*
  Library dealing with rotating stars defined by the potential

  Omega = 1/sqrt(x^2 + y^2 + z^2) + 1/2 omega^2 (x^2+ y^2)

  Author: Martin Horvat,  June 2016
*/

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

// general routines
#include "utils.h"      // Misc routines (sqr, solving poly eq,..)
#include "bodies.h"     // Definitions of different potentials

namespace rot_star {

  /* Solution of :

      1/rho + 1/2 x (8/27) rho^2 == 1  x in [0,1]

    is

      F[x_] := (3/Sqrt[x]) Sin[1/3 ArcSin[Sqrt[x]]] in [1, 1.5]

    for x < 1e-2:
      F[x] = 1. + x*(0.14814814814814814815 + x*(0.065843621399176954733 + x*(0.039018442310623380582 +
              x*(0.026494004038077604099 + x*(0.019482459535071207863 +
                    x*(0.015097518500112161079 + x*(0.012141919640301843831 + x*(0.01003890741502080552 + (0.0084799087976134159689 + 0.0072864401520233796473*x)*x))))))))
    for x < 1e-3:
    u = sqrt(1-x)
      F[x]= 1.5 + u*(-0.86602540378443864676 + u*(0.66666666666666666667 + u*(-0.56131276171213615994 +
              u*(0.49382716049382716049 + u*(-0.44593180513797483817 + u*
                     (0.40969364426154549611 + u*(-0.38104754777662929294 + u*(0.35766905451404765534 + (-0.33812089501784235099 + 0.32146058232867492974*u)*u))))))))

    Precision of both approximations is 10^-16.
  */
  template <class T> T radius_F(const T & x){

    if (x < 0.01)
      return 1. + x*(0.14814814814814814815 + x*(0.065843621399176954733 + x*(0.039018442310623380582 +
            x*(0.026494004038077604099 + x*(0.019482459535071207863 +
                  x*(0.015097518500112161079 + x*(0.012141919640301843831 + x*(0.01003890741502080552 + (0.0084799087976134159689 + 0.0072864401520233796473*x)*x))))))));
    T u;
    if (x > 1 - 1e-3) {
      u = std::sqrt(1 - x);
      return 1.5 + u*(-0.86602540378443864676 + u*(0.66666666666666666667 + u*(-0.56131276171213615994 +
            u*(0.49382716049382716049 + u*(-0.44593180513797483817 + u*
                   (0.40969364426154549611 + u*(-0.38104754777662929294 + u*(0.35766905451404765534 + (-0.33812089501784235099 + 0.32146058232867492974*u)*u))))))));
    }

    u = std::sqrt(x);
    return 3*std::sin(std::asin(u)/3)/u;
  }

  /*
    Derivative of F, defined as

    F[x] = (3/Sqrt[x]) Sin[1/3 ArcSin[Sqrt[x]]]

    and is written as

      1/2 (Cos[ArcSin[Sqrt[x]]/3]/(Sqrt[1-x] x)-(3 Sin[ArcSin[Sqrt[x]]/3])/x^(3/2))


  */
  template <class T> T radius_dF(const T & x){

    if (x < 0.01)
      return 0.14814814814814814815 + x*(0.13168724279835390947 + x*(0.11705532693187014175 +
         x*(0.1059760161523104164 + x*(0.097412297675356039314 + x*(0.090585111000672966473 +
                  x*(0.084993437482112906814 + x*(0.08031125932016644416 + x*(0.07631917917852074372 + (0.072864401520233796473 + 0.069837184838716836813*x)*x))))))));

    T s, c, u = std::sqrt(x), v = std::asin(u/3);

    utils::sincos(v, &s, &c);

    return (c/std::sqrt(1-x) - 3*s/u)/(2*x);
  }
  /*
    "Lagrange point" of the rotating star potential

        (L_1, 0, 0)     (-L_1, 0, 0)

    Input:
      omega - parameter of the potential

    Return
      value Lagrange point L_1: if (omega == 0) return NaN

  */
  template <class T>
  T lagrange_point(const T & omega) {
    if (omega == 0) return std::numeric_limits<T>::quiet_NaN();
    return std::cbrt(omega*omega);
  }

  /*
    Pole of the star i.e.  smallest z > 0 such that

      Omega(0,0,z) = Omega0

    Input:
      Omega0 - value of potential
      omega - parameter of the potential connected to rotation

    Return:
      height of pole > 0
  */

  template <class T>
  T pole(const T & Omega0, const T & omega) {
    return 1/Omega0;
  }

  /*
    Radius of the equator of the star i.e. such  that

      Omega(r,0,0) = Omega0

    solving
       1/r + 1/2 omega^2 r = Omega0

    The rescaled version with variable r = 1/Omega v is

       1/v + 1/2 t v^2 = 1      or

      1/2 t v^3 - v + 1 = 0

    with parameter

      t = omega^2/Omega0^3

    Input:
      Omega0 - value of potential
      omega - parameter of the potential connected to rotation > 0

    Return:
      radius of the equator

    Comment:
      if equator does not exist return std::quiet_nan()
  */
  template <class T>
  T equator(const T & Omega0, const T & omega) {

    // if we are discussing sphere
    if (omega == 0) return 1/Omega0;

    // Testing if the solution exists
    T t = 27*utils::sqr(omega/Omega0)/(8*Omega0);

    if (t > 1) {   // critical value
      std::cerr << "equator::area_volume:There is no solution for equator.\n";
      return std::numeric_limits<T>::quiet_NaN();
    }

    // return standard solution
    return radius_F(t)/Omega0;
  }

  /*
    Returning the critical values of the star potential.

    Input:
      omega - potential parameter

    Return:
      critical value of the potential
      if it does not exist returns NaN
  */

  template<class T>
  T critical_potential(const T & omega) {

    if (omega == 0) return std::numeric_limits<T>::quiet_NaN();

    return 3*std::pow(omega, 2./3)/2;
  }

  const int glq_n = 12;

  // Gauss-Lagrange nodes x_i in [0, 1]
  const double glq_x[] = {0.00921968287664037465472545492535958851992240009313,0.04794137181476257166076706694045190373120164539335,0.11504866290284765648155308339359096200753712499053,0.20634102285669127635164879052973285981545074297597,0.3160842505009099031236542316781412193718199293323,0.437383295744265542263779315268073435008301541847278,0.562616704255734457736220684731926564991698458152722,0.6839157494990900968763457683218587806281800706677,0.79365897714330872364835120947026714018454925702403,0.88495133709715234351844691660640903799246287500947,0.95205862818523742833923293305954809626879835460665,0.99078031712335962534527454507464041148007759990687};

  // transformation of Gauss-Lagrange nodes 1-x_i^2
  const double glq_c[] = {0.9999149974476541842661467965417882899415507647565282,0.997701624868518688886356111619185330729455844559929,0.98676380516426692544878948933295339949663441664057,0.95742338228645440826431277137704492238819033349748,0.90009074658527803765626345879402831515538079316031,0.80869585260388434398548169046921499122585881057871,0.683462444092415428513040321005361861242461894273266,0.5322592475870978439035719221503107538990206518249,0.37010542799983696096761035243651064201909181944942,0.21686113096996223841189565612013532351170866662164,0.09358436849804383220789024550008913819185913534663,0.0183543632009349335755977063925074669813955649428};

  //  Gauss-Lagrange nodes weight w_i
  const double glq_w[] = {0.023587668193255913597307980742508530158514537,0.0534696629976592154801273590969981121072850867,0.0800391642716731131673262647716795359360058652,0.10158371336153296087453222790489918825325907364,0.1167462682691774043804249494624390281297049861,0.12457352290670139250028121802147560541523045128481,0.12457352290670139250028121802147560541523045128481,0.1167462682691774043804249494624390281297049861,0.10158371336153296087453222790489918825325907364,0.0800391642716731131673262647716795359360058652,0.0534696629976592154801273590969981121072850867,0.023587668193255913597307980742508530158514537};


  /*
   Computing area of the surface and the volume of the rotating star lobe.
   The quantities are defined as

      A = fA*int_0^1 dx F[t (1-x^2)]^2
      V = fV*int_0^1 dx F[t (1-x^2)]^3

  with
      fA = 4 Pi/Omega0^2
      fV = 4Pi/(3 Omega0^3)
      t = 28/7 omega^2/Omega0^3

    Input:
      omega - parameter of the potential
      Omega0 - value of the potential
      choice - setting
        1 bit  - area
        2 bit  - volume

    Output:
      res[2] = {area, volume}

    Return:
      -1 if no results are computed
       0 if everything is OK else
       1 if the equator does not exist

    Ref:
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
  */

  template<class T>
  int area_volume(
    double res[2],
    const unsigned &choice,
    const T & Omega0,
    const T & omega) {

    //
    // What is calculated
    //

    bool
      b_area = (choice & 1u) == 1u,
      b_vol = (choice & 2u) == 2u;

    if (!b_area && !b_vol) return -1;

    T Omega2 = utils::sqr(Omega0),
      Omega3 = Omega0*Omega2,

      fA = utils::m_4pi/Omega2,
      fV = fA/(3*Omega0);

    if (omega == 0) {

      if (b_area) res[0] = fA;
      if (b_vol)  res[1] = fV;

      return 0;
    }

    //
    // Testing if the solution exists
    //

    T t = 27*omega*omega/(8*Omega3);

    if (t > 1) {   // critical value
      std::cerr << "rotstar::area_volume:There is no solution for equator.\n";
      return 1;
    }

    // maximal possible area and volume
   if (t == 1) {

      if (b_area) res[0] = fA*1.4738328557327723663;
      if (b_vol)  res[1] = fV*1.8262651430357240475;

      return 0;
    }

    //
    // Analytic approximation (generated in rot_star.nb)
    // relative precision at least 1e-20 for t < 0.1
    //
    if (t < 0.1) {
       if (b_area)
        res[0] = fA*(
              1. + t*(0.1975308641975308642 + t*(0.081938728852309099223 + t*(0.044592505497855292094 +
              t*(0.027991125536102306807 + t*(0.019191956290440516452 + t*(0.013973517907720004089 +
              t*(0.010626774112537682122 + t*(0.008352883235964515829 + (0.0067378382224733158155 +
              0.0055496612950397498048*t)*t)))))))));

      if (b_vol)
        res[1] = fV*(
              1. + t*(0.2962962962962962963 + t*(0.1404663923182441701 + t*(0.081752926746068035506 +
              t*(0.053437603296195312995 + t*(0.037645760415864089963 + t*(0.027947035815440008177 +
              t*(0.021566100404855884306 + t*(0.017145391905400848281 + (0.013956950603694725618 +
              0.011581901833126434375*t)*t)))))))));

    } else {

      T r[glq_n];
      for (int i = 0; i < glq_n; ++i) r[i] = radius_F(t*glq_c[i]);

      // = w.r^2
      if (b_area) {
        res[0] = 0;
        for (int i = 0; i < glq_n; ++i) res[0] += utils::sqr(r[i])*glq_w[i];
        res[0] *= fA;
      }

      // = w.r^3
      if (b_vol) {
        res[1] = 0;
        for (int i = 0; i < glq_n; ++i) res[1] += utils::cube(r[i])*glq_w[i];
        res[1] *= fV;
      }
    }

    return 0;
  }

  /*
    Computing value of the Kopal potential

      1/r + 1/2 omega^2 (x^2 +y^2) = Omega

    of the rotating star at given volume V given by

      V = 4Pi/(3 Omega^3) v(t)  v(t) = int_0^1 dx F[t(1-x^2)]^3

    Solving

      q = t v(t)        for   t in [0,1]

    with

      q = 3 V omega^2/(4 pi) (27/8)

      t = 27/8 omega^2/Omega^3

    Input:
      V - volume
      omega - parameter of the potential

    Return:
      Omega = (27/8 omega^2/t)^(1/3)

      if Omega is not found it return NaN
  */
  template <class T>
  T Omega_at_vol(const T & V, const T & omega) {

    const int max_iter = 20;
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T eps1 = 200*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    //
    // Without rotation case
    //
    if (omega == 0) return std::cbrt(utils::m_4pi/(3*V));

    //
    // Testing if the solution exists
    //
    T w2 = utils::sqr(omega),
      q = 0.80572189940272013733*V*w2;

    const T qmax = 1.8262651430357240475;

    if (q-qmax >= qmax*eps1) {
      std::cerr << "rotstar::Omega_at_vol::Volume is too large for given omega.\n";
      return std::numeric_limits<T>::quiet_NaN();
    }

    //
    // Limiting case: Largest volume, t = 1
    //
    if (std::abs(q - qmax) <= eps1*qmax) return 1.5*std::cbrt(w2);

    //
    // All other cases
    //
    const int data_n = 51;

    // sampled points in [0,1]
    const T data_t[]={
      0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,
      0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,
      0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,
      0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1.};

    // values of t v(t) for t in data_t
    const T data_tv[]={
      0,0.020119655503571336279,0.040483278841607347711,0.06109811031685671632,
      0.081971749286726935007,0.10311217969878136222,0.12452779802698813392,
      0.14622744389251451246,0.16822043369364172179,0.19051659761711118807,
      0.2131263204592856054,0.23606058675161540538,0.25933103076313888313,
      0.2829499920457121506,0.30693057729860478801,0.33128672946207243005,
      0.35603330510965270707,0.38118616140272611332,0.40676225410663677499,
      0.4327797484560449624,0.45925814501198852319,0.48621842309236820716,
      0.51368320490491044737,0.54167694419848036486,0.57022614411689890316,
      0.59935961004586970359,0.62910874466564005124,0.65950789426641189408,
      0.69059475779943684767,0.72241087333515090597,0.7550022008832237079,
      0.78841982633858218628,0.82272081930436687919,0.85796928868986375403,
      0.89423769579845619539,0.93160850747838837152,0.97017630562137649538,
      1.0100505201627646948,1.051359031494625967,1.0942530137644837284,
      1.1389135975604037053,1.1855612854393166937,1.2344696909894107234,
      1.285986380835111428,1.340566051023177422,1.3988266801184253404,
      1.461652617918166124,1.5304065938856312545,1.6074478031893582002,
      1.6978529467004852418,1.8262651430357240475};

    int it = 0;

    T t = utils::lin_interp<T>(q, data_n, data_tv, data_t),
      r[3][glq_n], dt, v, dv;

    do {

      if ( t < 0.1) {
        // calc v(t)
        v = 1. + t*(0.2962962962962962963 + t*(0.1404663923182441701 + t*(0.081752926746068035506 +
              t*(0.053437603296195312995 + t*(0.037645760415864089963 + t*(0.027947035815440008177 +
              t*(0.021566100404855884306 + t*(0.017145391905400848281 + (0.013956950603694725618 +
              0.011581901833126434375*t)*t))))))));
        // calc dv/dt(t)
        dv = 0.2962962962962962963 + t*(0.28093278463648834019 + t*(0.24525878023820410652 +
             t*(0.21375041318478125198 + t*(0.18822880207932044981 + t*(0.16768221489264004906 +
             t*(0.15096270283399119014 + t*(0.13716313524320678624 + (0.12561255543325253056 +
             0.11581901833126434375*t)*t)))))));
      } else {

        for (int i = 0; i < glq_n; ++i) {
          r[0][i] = radius_F(t*glq_c[i]);
          r[1][i] = r[0][i]*r[0][i];
          r[2][i] = r[1][i]*r[0][i];
        }

        // calc v(t) = int_0^1 dx F[t (1 - x^2)]^3
        v = 0;
        for (int i = 0; i < glq_n; ++i) v += r[2][i]*glq_w[i];

        // calc dv/dt(t) = 3 int_0^1 dx F[t (1 - x^2)]^2 dF[t (1 - x^2)] (1 - x^2)]
        dv = 0;
        for (int i = 0; i < glq_n; ++i) dv += r[1][i]*radius_dF(t*glq_c[i])*glq_c[i]*glq_w[i];
        dv *= 3;
      }

      // calc d(tv)
      dv = v + t*dv;

      t -= (dt = (t*v - q)/dv);

    } while (std::abs(dt) > min + eps*t && ++it < max_iter);


    if (it == max_iter) {
      std::cerr << "rotstar::Omega_at_vol::To many iterations.\n";
      return std::numeric_limits<T>::quiet_NaN();
    }

    return 1.5*std::cbrt(w2/t);
  }


  /*
    Find a point on the horizon around a lobe of the rotating star.

    Input:
      view - direction of the view
      Omega0 - value of the potential
      omega - parameter of the potential
      spin - vector of the spin
             if (spin == NULL) align cases is assumed
                spin = [0, 0, 1]
    Output:
      p - point on the horizon
  */
  template<class T>
  bool point_on_horizon(
    T r[3],
    T view[3],
    const T & Omega0,
    const T & omega,
    T *spin = 0
  ){

    T r0 = equator(Omega0, omega);

    if (std::isnan(r0)) return false;

    //
    // if spin is aligned with view => horizon == equator
    //
    const T eps = 10*std::numeric_limits<T>::epsilon();

    if ((spin == 0 && view[0] == 0 && view[1] == 0) ||
        (spin != 0 && std::abs(std::abs(utils::dot3D(spin, view)) - 1.) <  eps)
    ) {
      // just some point on equator
      r[0] = r0;
      r[1] = r[2] = 0;
      return true;
    }

    if (spin == 0 || (spin != 0 && spin[0] == 0 && spin[1] == 0)) {

      // intersection of horizon with equator
      T f = r0/std::sqrt(view[0]*view[0] + view[1]*view[1]);
      r[0] = -f*view[1];
      r[1] = f*view[0];
      r[2] = 0;

    } else {

      // create orthonormal basis, assuming spin = e_z
      T e[2][3];

      // e_x =  view x e_z
      utils::cross3D(view, spin,  e[0]);

      // e_x  needs normalization
      T f = 1/utils::hypot3(e[0]);
      for (int i = 0; i < 3; ++i) e[0][i] *=f;

      // e_y = e_z x e_x
      utils::cross3D(spin, e[0], e[1]);

      // project vector view onto this basis
      T v[2];
      for (int i = 0; i < 2; ++i) v[i] = utils::dot3D(view, e[i]);

      // intersection of horizon with equator
      f = r0/std::sqrt(v[0]*v[0] + v[1]*v[1]);
      T w[2] = {-f*v[1], f*v[0]};

      // transform back to original coordinate system
      for (int i = 0; i < 3; ++i) r[i] = 0;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) r[j] += w[i]*e[i][j];
    }

    return true;
  }

  /*
    Initial point: it is on the z-axis, z>0

    Input:
      Omega0 - value of the potential
      omega - parameter of the potential

    Output:
      r - position
      g - gradient
  */
  template <class T>
  void meshing_start_point(
    T r[3],
    T g[3],
    const T & Omega0,
    const T & omega
  ){
    r[0] = r[1] = 0;
    r[2] = 1/Omega0;

    g[0] = g[1] = 0;
    g[2] = Omega0*Omega0;
  }


  /*
    Point x on the surface

      Omega(x) = Omega_0

    in directon

      u = (sin(theta)cos(phi), sin(theta) sin(phi), cos(theta))
      x = r*u

    Input:
      theta - polar angle (from z axis)
      phi - azimulal angle
      Omega0 - value of the potential
      omega - parameter of the potential

    Output:
      x - point on the surface
      g - gradient on the surface
  */
  template <class T>
  void point_on_surface(
    const T & Omega0,
    const T & omega,
    const T& theta,
    const T& phi,
    T x[3],
    T *g = 0) {

    T st, ct, sp, cp;

    utils::sincos(theta, &st, &ct);
    utils::sincos(phi, &sp, &cp);

    T r = equator(Omega0, omega*st);

    x[0] = r*st*cp;
    x[1] = r*st*sp;
    x[2] = r*ct;

    if (g) {
      T w2 = omega*omega, r2 = r*r, r3 = r2*r, f = (1 - w2*r3)/r2;

      g[0] = f*cp*st;
      g[1] = f*st*sp;
      g[2] = ct/r2;
    }
  }

  /*
    Point x on the surface for rotating star with misalignment

      Omega(x) = Omega_0

    where

      Omega(r) = 1/|r| + 1/2 omega^2 |r - s (r.s)|^2
        r = (x, y, z)
        s = (sx, sy, sz)    |s| = 1

    in directon

      u = (sin(theta)cos(phi), sin(theta) sin(phi), cos(theta))
      x = r*u

    Input:
      theta - polar angle (from z axis)
      phi - azimulal angle
      Omega0 - value of the potential
      omega - parameter of the potential

    Output:
      x - point on the surface
      g - gradient on the surface (-grad Omega)
  */
  template <class T>
  void point_on_surface(
    const T & Omega0,
    const T & omega,
    T spin[3],
    const T& theta,
    const T& phi,
    T x[3],
    T *g = 0) {

    T st, ct, sp, cp;

    utils::sincos(theta, &st, &ct);
    utils::sincos(phi, &sp, &cp);

    // direction in which we want the point
    x[0] = st*cp;
    x[1] = st*sp;
    x[2] = ct;

    T t = utils::dot3D(x, spin),
      r = equator(Omega0, omega*std::sqrt(1-t*t));

    // calculate the gradient
    if (g) {
      T f1 = 1/(r*r), f2 = omega*omega*r;
      f1 -= f2;
      f2 *= t;
      for (int i = 0; i < 3; ++i) g[i] = f1*x[i] + f2*spin[i];
    }

    // calculate the point
    for (int i = 0; i < 3; ++i) x[i] *= r;
  }


} // namespace rot_star

