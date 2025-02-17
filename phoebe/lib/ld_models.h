#pragma once
/*
  Limb darkening models. Structures provide

    D(mu)
    F(mu) = D(mu)/D0        mu = cos(theta)

  and

    D0 = integral_{half-sphere} D(cos(theta)) cos(theta) dOmega

  in spherical coordinate system

    vec r = r (sin(theta) sin(phi), sin(theta) cos(phi), cos(theta))

    dOmega = sin(theta) dtheta dphi

  Supporting:

    * 'uniform': D(mu) = 1
    * 'linear':  D(mu) = 1 - x*(1 - mu)
    * 'quadratic': D(mu) = 1 - x(1 - mu) - y(1 - mu)^2
    * 'nonlinear': D(mu) = 1 - x(1 - mu) - y (1 - mu)^p
    * 'logarithmic': D(mu) = 1 - x*(1-mu) - y*mu*log(mu)
    * 'square_root': D(mu) = 1 - x(1 - mu) - y(1 - sqrt(mu))
    * 'power' ~ claret:
      D(mu) = 1 - a0(1 - mu^(1/2)) - a1(1 - mu) - a2(1-mu^(3/2)) - a3(1 - mu^2)

  Constraints:



  I. In strictly autonomous systems

    D(mu) in [0,1]

  where lower bound D(mu)>0 means that the surface is not energy sink and
  upper bound D(mu) < 1 means that emitted energy can not exceed generated one.

  II. In irradiated stars, without properly modeling irradiation:

    D(mu) >= 0 and D(mu) can be larger than 1

  [Claret 2004, Fig 4]



  Author: Martin Horvat, August 2016, August 2019

  Ref:

  * Claret, A., Diaz-Cordoves, J., & Gimenez, A., Linear and non-linear limb-darkening coefficients for the photometric bands R I J H K. Astronomy and Astrophysics Supplement, v.114, p.247, 1995.

  * Claret, A., 2000, A&A, 363, 1081

  * Kallrath, Josef, Milone, Eugene F., Eclipsing Binary Stars: Modeling and Analysis (Springer Verlag, 2009)

  * A. Claret, On the irradiated stellar atmospheres in close binary systems: Improvements and uncertainties, A&A 422, 665–673 (2004) DOI: 10.1051/0004-6361:20047056
*/


#include <cmath>
#include <limits>
#include "hash.h"

#include "utils.h"

// Enumeration of supported types
enum TLDmodel_type {
  UNIFORM,
  LINEAR,
  QUADRATIC,
  NONLINEAR,
  LOGARITHMIC,
  SQUARE_ROOT,
  POWER,
  NONE
};

/* ====================================================================
  Interface to the limb darkening models through structures each
  discussing separate model
 ==================================================================== */

// Abstract struct for limb darkening models
//
// Ref: http://stackoverflow.com/questions/301203/extract-c-template-parameters
template <class T>
struct TLDmodel {
  T D0;               // integrated LD view-factor
  TLDmodel_type type; // type of the model
  int nr_par;         // number of parameters of type T

  virtual ~TLDmodel() = default;
  virtual T D(const T & mu) const = 0;

  virtual bool check () const = 0;                   // check if D(mu) > 0
  virtual bool check_strict () const = 0;            // check if D(mu) in [0,1]

  T F(const T & mu) const { return D(mu)/D0; }
};


// Uniform limb darkening == plain Labertian (0 parameter)
// D(mu) = 1
template <class T>
struct TLDuniform: TLDmodel<T> {

  TLDuniform(T *p) {setup();}
  TLDuniform(){setup();}

  void setup(){
    this->D0 = utils::m_pi;
    this->type = UNIFORM;
    this->nr_par = 0;
  }

  T D([[maybe_unused]] const T & mu) const { return 1; }

  bool check() const { return true; }
  bool check_strict() const { return true; }
};

// Linear limb darkening (1 parameter)
// D(mu) = 1 - x*(1 - mu)
template <class T>
struct TLDlinear : TLDmodel<T> {

  T x;

  TLDlinear(T *p): x(*p) {setup();}
  TLDlinear(const T &x) : x(x) {setup();}

  void setup(){
    this->D0 = utils::m_pi*(1 - x/3);
    this->type = LINEAR;
    this->nr_par = 1;
  }

  T D(const T & mu) const {
    return  1 - x*(1 - mu);
  }

  bool check() const {
    return x <= 1;
  }

  bool check_strict () const {
    return x <= 1 && x >= 0;
  }
};

// Quadratic limb darkening (2 parameter)
// D(mu) = 1 - x(1 - mu) - y(1 - mu)^2
template <class T>
struct TLDquadratic: TLDmodel<T> {

  T x, y;

  TLDquadratic(T *p) : x(p[0]), y(p[1]) {
    setup();
  }

  TLDquadratic(const T &x, const T &y) : x(x), y(y) {
    setup();
  }

  void setup(){
    this->D0 = utils::m_pi*(1 - x/3 - y/6);
    this->type = QUADRATIC;
    this->nr_par = 2;
  }

  T D(const T &mu) const {
    T u = 1 - mu;
    return 1 - u*(x + y*u);
  }

  bool check() const {
    return y <= (x <= 2 ? 1 - x : -0.25*x*x);
  }

  bool check_strict() const {
    return x >= 0 && y >= -x && y <= (x <= 2 ? 1 - x : -0.25*x*x);
  }
};

// Nonlinear limb darkening (3 parameters)
// sometimes called power limb darkening
// D(mu) = 1 - x(1 - mu) - y (1 - mu)^p
template <class T>
struct TLDnonlinear: TLDmodel<T> {

  T x, y, p;

  TLDnonlinear(T *par) : x(par[0]), y(par[1]), p(par[2]) {
    setup();
  }
  TLDnonlinear(const T &x, const T & y, const T &p) : x(x), y(y), p(p) {
    setup();
  }

  void setup(){
    this->D0 = utils::m_pi*(1 - x/3 - y/(1 + p*(3 + p)/2));
    this->type = NONLINEAR;
    this->nr_par = 3;
  }

  T D(const T &mu) const {
    T u = 1 - mu;
    return 1 - x*u - y*std::pow(u, p);
  }

  bool check() const {
    T t;

    if (p > 1) {
      return y <= (x <= (t = p/(p - 1)) ? 1 - x : std::pow(x/t,p)/(1 - p));
    } else if (p < 1) {
      T q = 1/p;
      return x <= (y <= (t = q/(q-1)) ? 1 - y : std::pow(y/t,q)/(1 - q));
    }

    // p == 1 (linear case)
    t = x + y;
    return t <= 1;
  }

  bool check_strict() const {
    T t;

    if (p > 1) {
      return
        x >= 0 && y >= -x &&
        y <= (x <= (t = p/(p - 1)) ? 1 - x : std::pow(x/t,p)/(1 - p));
    } else if (p < 1) {
      T q = 1/p;

      return
        y >= 0 && x >= -y &&
        x <= (y <= (t = q/(q-1)) ? 1 - y : std::pow(y/t,q)/(1 - q));
    }

    // p == 1 (linear case)
    t = x + y;
    return t <= 1 && t >= 0;
  }
};

// Logarithmic limb darkening (2 parameters)
// D(mu) = 1 - x*(1-mu) - y*mu*log(mu)
template <class T>
struct TLDlogarithmic: TLDmodel<T> {

  T x, y;

  TLDlogarithmic(const T *p) : x(p[0]), y(p[1]) {
    setup();
  }

  TLDlogarithmic(const T &x, const T &y) : x(x), y(y) {
    setup();
  }

  void setup(){
    this->D0 = utils::m_pi*(1 - x/3  + 2*y/9);
    this->type = LOGARITHMIC;
    this->nr_par = 2;
  }

  T D(const T &mu) const {
    return 1 - x*(1 - mu) - y*mu*std::log(mu);
  }

  bool check() const {
    return
      x <= 1 &&
      y >= (x == 1 ? 0 : (x == 0 ? -utils::m_e : -x/utils::lambertW(x/((1 - x)*utils::m_e))));
  }

  bool check_strict() const {
    return
      x <= 1 && x >= 0 && y <= x &&
      y >= (x == 1 ? 0 : (x == 0 ? -utils::m_e : -x/utils::lambertW(x/((1 - x)*utils::m_e))));
  }
};


// Square-root limb darkening (2 parameters)
// D(mu) = 1 - x(1 - mu) - y(1 - sqrt(mu))
template <class T>
struct TLDsquare_root: TLDmodel<T> {

  T x, y;

  TLDsquare_root(T *p) : x(p[0]), y(p[1]) {
    setup();
  }

  TLDsquare_root(const T &x, const T &y) : x(x), y(y) {
    setup();
  }

  void setup(){
    this->D0 = utils::m_pi*(1 - x/3 - y/5);
    this->type = SQUARE_ROOT;
    this->nr_par = 2;
  }

  T D(const T & mu) const {
    return 1 - x*(1 - mu) - y*(1 - std::sqrt(mu));
  }

  bool check() const {
    return
      y <= (x <= 1 ? 1 - x : 2*(std::sqrt(x) -x));
  }

  bool check_strict() const {
    return
      x >= -1 && x <= 4 &&
      y >= -4 && y <= 2 &&
      y >= (x <= 0 ? -2*x : -x) &&
      y <= (x <= 1 ? 1 - x : 2*(std::sqrt(x) -x))
    ;
  }
};

// Claret's or Power limb darkening (4 parameters)
// D(mu) = 1 - a[0](1 - mu^(1/2)) - a[1](1 - mu) - a[2] (1-mu^(3/2)) - a[3] (1 - mu^2)
template <class T>
struct TLDpower: TLDmodel<T> {

  T a[4];

  TLDpower(T *p) {
    for (int i = 0; i < 4; ++i) a[i] = p[i];
    setup();
  }

  TLDpower(const T &a0, const T &a1, const T &a2, const T &a3)
  : a{a0, a1, a2, a3} {
    setup();
  }

  void setup(){
    this->D0 = utils::m_pi*(1 - (42*a[0] + 70*a[1] + 90*a[2] + 105*a[3])/210);
    this->type = POWER;
    this->nr_par = 4;
  }

  T D(const T & mu) const {
    T q = std::sqrt(mu);
    return
      1 - a[0]*(1 - q) - a[1]*(1 - mu) - a[2]*(1 - mu*q) - a[3]*(1 - mu*mu);
  }

  bool check() const {

    if (a[0] + a[1] + a[2] + a[3] > 1) return false;

    // empirical check based on some points
    T t, dmu = 0.01;

    for (T mu = 0; mu <= 1; mu += dmu) {
      t = D(mu);
      if (t < 0) return false;
    }

    return true;
  }

  bool check_strict() const {

    if (a[0] + a[1] + a[2] + a[3] > 1) return false;

    // empirical check based on some points
    T t, dmu = 0.01;

    for (T mu = 0; mu <= 1; mu += dmu) {
      t = D(mu);
      if (t < 0 || t > 1) return false;
    }

    return true;
  }

};

/* ====================================================================
  Interface to the limb darkening models through the function
 ==================================================================== */

namespace LD {

  /*
    Calculate the value of the LD factor D(mu) in the differential
    view-factor in spherical coordinates:

      vec r = r (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))

    Input:
      choice - determine LD model
      mu = cos(theta)
      p - pointer to parameters

    Return:
      D(mu)
  */

  template <class T>
  T D(TLDmodel_type choice, const T & mu, T *p) {


    switch (choice){
      case UNIFORM:
        return 1;
      case LINEAR:
        return 1 - p[0]*(1 - mu);
      case QUADRATIC:
      {
        T t = 1 - mu;
        return 1 - t*(p[0] + t*p[1]);
      }
      case NONLINEAR:
      {
        T t = 1 - mu;
        return 1 - p[0]*t - p[1]*std::pow(t, p[2]);
      }
      case LOGARITHMIC:
        return 1 - p[0]*(1 - mu) - p[1]*mu*std::log(mu);
      case SQUARE_ROOT:
        return 1 - p[0]*(1 - mu) - p[1]*(1 - std::sqrt(mu));
      case POWER:
      {
        T q = std::sqrt(mu);
        return 1 - p[0]*(1 - q) - p[1]*(1 - mu) - p[2]*(1 - mu*q) - p[3]*(1 - mu*mu);
      }
      default:
        std::cerr << "LD::D::This model is not supported\n";
        return std::numeric_limits<T>::quiet_NaN();
    }
  }


  /*
    Calculate the integrated LD factor D(mu) given by

      D0 = 2 Pi int_0^{pi/2} dtheta sin(theta) D(cos(theta)) cos(theta)

    Input:
      choice - determine LD model
      p - pointer to parameters

    Return:
      D0
  */

  template <class T>
  T D0(TLDmodel_type choice, T *p) {

    switch (choice){
      case UNIFORM:
        return utils::m_pi;
      case LINEAR:
        return utils::m_pi*(1 - p[0]/3);
      case QUADRATIC:
        return utils::m_pi*(1 - p[0]/3 - p[1]/6);
      case NONLINEAR:
        return utils::m_pi*(1 - p[0]/3 - p[1]/(1 + p[2]*(3 + p[2])/2));
      case LOGARITHMIC:
        return utils::m_pi*(1 - p[0]/3  + 2*p[1]/9);
      case SQUARE_ROOT:
        return utils::m_pi*(1 - p[0]/3 - p[1]/5);
      case POWER:
        return utils::m_pi*(1 - (42*p[0] + 70*p[1] + 90*p[2] + 105*p[3])/210);
      default:
        std::cerr << "LD::D0::This model is not supported\n";
        return std::numeric_limits<T>::quiet_NaN();
    }
  }

  /*
    Give the number of parameters used by different LD models.

    Input:
      choice - determine LD model

    Return:
     nr. parameters
  */

  int nrpar(TLDmodel_type choice) {

    switch (choice){
      case UNIFORM: return 0;
      case LINEAR: return 1;
      case QUADRATIC: return 2;
      case NONLINEAR: return 3;
      case LOGARITHMIC: return 2;
      case SQUARE_ROOT: return 2;
      case POWER: return 4;
      case NONE: return -1;
    }

    std::cerr << "LD::nrpar::This model is not supported\n";
    return -1;
  }


  /*
    Give the type based on the name of the LD models.

    Input:
      s - string on name of LD model

    Return:
      type as defined in TLDmodel_type
  */

  TLDmodel_type type(const char *s) {

    switch (fnv1a_32::hash(s)){
      case "uniform"_hash32: return UNIFORM;
      case "linear"_hash32:  return LINEAR;
      case "quadratic"_hash32: return QUADRATIC;
      case "nonlinear"_hash32: return NONLINEAR;
      case "logarithmic"_hash32: return LOGARITHMIC;
      case "square_root"_hash32: return SQUARE_ROOT;
      case "power"_hash32: return POWER;

      default:
        std::cerr << "LD::type::This model is not supported\n";
      return NONE;
    }
  }

  /*
    Calculate gradient of LD function D(mu) w.r.t. parameters. The D(mu) is
    given in spherical coordinates:

      vec r = r (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))

    Input:
      choice - determine LD model
      mu = cos(theta)
      p - pointer to parameters

    Output:
      g = grad_{params} D(mu)
  */

  template <class T>
  void gradparD(TLDmodel_type choice, const T & mu, T *p, T *g) {

    switch (choice){
      case UNIFORM: break;
      case LINEAR:
        g[0] = mu - 1;
        break;
      case QUADRATIC:
        g[0] = mu - 1;
        g[1] = -g[0]*g[0];
        break;
      case NONLINEAR:
        g[0] = mu - 1;
        g[1] = -std::pow(-g[0], p[2]);
        g[2] = p[1]*g[1]*std::log(-g[0]);
        break;
      case LOGARITHMIC:
        g[0] = mu - 1;
        g[1] = -mu*std::log(mu);
        break;
      case SQUARE_ROOT:
        g[0] = mu - 1;
        g[1] = std::sqrt(mu) - 1;
        break;
      case POWER:
      {
        T q = std::sqrt(mu);
        g[0] = q - 1;
        g[1] = mu - 1;
        g[2] = q*mu - 1;
        g[3] = mu*mu - 1;
      }
      break;

      default:
       std::cerr << "LD::gradparD::This model is not supported\n";
    }
  }


  /*
    Checking is the parameters yield D(mu) > 0 for all mu in [0,1].

    Input:
      choice - determine LD model
      p - pointer to parameters

    Output:
      true - is everything is ok
  */

  template <class T>
  bool check(TLDmodel_type choice, T *p) {

    switch (choice) {

      case UNIFORM:
        return true;

      case LINEAR:
        return p[0] <= 1;

      case QUADRATIC:
        return
          p[1] <= (p[0] <= 2 ? 1 - p[0] : -0.25*p[0]*p[0]);

      case NONLINEAR:
      {
        T t;

        if (p[2] > 1) {

          return
            p[1] <= (p[0] <= (t = p[2]/(p[2] - 1)) ? 1 - p[0] : std::pow(p[0]/t, p[2])/(1 - p[2]));

        } else if (p[2] < 1) {
          T q = 1/p[2];

          return
            p[0] <= (p[1] <= (t = q/(q - 1)) ? 1 - p[1] : std::pow(p[1]/t, q)/(1 - q));
        }

        // p[2] == 1, linear case
        t = p[0] + p[1];
        return t <= 1;
      }

      case LOGARITHMIC:
        return
          p[0] <= 1 &&
          p[1] >= (p[0] == 1 ? 0 : (p[0] == 0 ? -utils::m_e : -p[0]/utils::lambertW(p[0]/((1 - p[0])*utils::m_e))));

      case SQUARE_ROOT:
        return p[1] <= (p[0] <= 1 ? 1 - p[0] : 2*(std::sqrt(p[0]) - p[0]));


      case POWER:
      {
        if (p[0] + p[1] + p[2] + p[3] > 1) return false;

        // empirical check based on some points
        T q, t, dmu = 0.01;

        for (T mu = 0; mu <= 1; mu += dmu) {

          q = std::sqrt(mu);
          t = 1 - p[0]*(1 - q) - p[1]*(1 - mu) - p[2]*(1 - mu*q) - p[3]*(1 - mu*mu);

          if (t < 0) return false;
        }

        return true;
      }

      default:
       std::cerr << "LD::check::This model is not supported\n";
       return false;
    }
  }

  /*
    Checking is the parameters yield D(mu) in [0,1] for mu in [0,1].

    Input:
      choice - determine LD model
      p - pointer to parameters

    Output:
      true - is everything is ok
  */

  template <class T>
  bool check_strict(TLDmodel_type choice, T *p) {

    switch (choice) {

      case UNIFORM:
        return true;

      case LINEAR:
        return p[0] <= 1 && p[0] >= 0;

      case QUADRATIC:
        return
          p[0] >= 0 && p[1] >= -p[0] &&
          p[1] <= (p[0] <= 2 ? 1 - p[0] : -0.25*p[0]*p[0]);

      case NONLINEAR:
      {
        T t;

        if (p[2] > 1) {

          return
            p[0] >= 0 && p[1] >= -p[0] &&
            p[1] <= (p[0] <= (t = p[2]/(p[2] - 1)) ? 1 - p[0] : std::pow(p[0]/t, p[2])/(1 - p[2]));

        } else if (p[2] < 1) {
          T q = 1/p[2];

          return
            p[1] >= 0 && p[0] >= -p[1] &&
            p[0] <= (p[1] <= (t = q/(q - 1)) ? 1 - p[1] : std::pow(p[1]/t, q)/(1 - q));
        }

        // p[2] == 1, linear case
        t = p[0] + p[1];
        return t >= 0 && t <= 1;
      }

      case LOGARITHMIC:
        return
          p[0] <= 1 && p[0] >= 0 && p[1] <= p[0] &&
          p[1] >= (p[0] == 1 ? 0 : (p[0] == 0 ? -utils::m_e : -p[0]/utils::lambertW(p[0]/((1 - p[0])*utils::m_e))));

      case SQUARE_ROOT:
        return
          p[0] >= -1 && p[0] <= 4 &&
          p[1] >= -4 && p[1] <= 2 &&
          p[1] >= (p[0] <= 0 ? -2*p[0] : -p[0]) &&
          p[1] <= (p[0] <= 1 ? 1 - p[0] : 2*(std::sqrt(p[0]) - p[0]));

      case POWER:
      {
        if (p[0] + p[1] + p[2] + p[3] > 1) return false;

        // empirical check based on some points
        T q, t, dmu = 0.01;

        for (T mu = 0; mu <= 1; mu += dmu) {

          q = std::sqrt(mu);
          t = 1 - p[0]*(1 - q) - p[1]*(1 - mu) - p[2]*(1 - mu*q) - p[3]*(1 - mu*mu);

          if (t < 0 || t > 1) return false;
        }

        return true;
      }

      default:
       std::cerr << "LD::check::This model is not supported\n";
       return false;
    }
  }
}
