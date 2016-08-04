#if !defined(__ld_models_h)
#define __ld_models_h


/* 
  Limb darkening models. Structures provide
   
    D(mu)
    F(mu) = D(mu)/D0        mu = cos(theta)
  
  and
  
    D0 = integral_{half-sphere} D(cos(theta)) cos(theta) dOmega
   
  in spherical coordinate system
    
    vec r = r (sin(theta) sin(phi), sin(theta) cos(phi), cos(theta))
    
    dOmega = sin(theta) dtheta dphi
  
  
  Author: Martin Horvat, August 2016
  
  Ref:

  * Claret, A., Diaz-Cordoves, J., & Gimenez, A., Linear and non-linear limb-darkening coefficients for the photometric bands R I J H K.Astronomy and Astrophysics Supplement, v.114, p.247, 1995.

  * Kallrath, Josef, Milone, Eugene F., Eclipsing Binary Stars: Modeling and Analysis (Spinger Verlag, 2009)
*/

#include <cmath>
#include "hash.h"


// Enumeration of supported types
enum TLDmodel_type {
  UNIFORM,
  LINEAR,
  QUADRATIC, 
  NONLINEAR,
  LOGARITHMIC,
  SQUARE_ROOT,
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
  virtual bool check () const = 0;
  
  T F(const T & mu) const { return D(mu)/D0; }
};


// Uniform limb darkening == plain Labertian (0 parameter)
// D(x) = 1
template <class T>
struct TLDuniform: TLDmodel<T> {
  
  TLDuniform(T *p) {setup();}
  TLDuniform(){setup();}

  void setup(){
    this->D0 = M_PI; 
    this->type = UNIFORM;
    this->nr_par = 0;
  }

  T D(const T & mu) const { return 1; }
  
  bool check() const { return true; }
};
  
// Linear limb darkening (1 parameter)
// D(x) = 1 - x*(1 - mu)
template <class T> 
struct TLDlinear : TLDmodel<T> {
  
  T x;
  
  TLDlinear(T *p): x(*p) {setup();}
  TLDlinear(const T &x) : x(x) {setup();}
  
  void setup(){
    this->D0 = M_PI*(1 - x/3);
    this->type = LINEAR;
    this->nr_par = 1; 
  }
  
  T D(const T & mu) const {
    return  1 - x*(1 - mu);
  }
  
  bool check () const {
    return (x < 1 && x > 0);
  }
};

// Quadratic limb darkening (2 parameter)  
// D(x) = 1 - x(1 - mu) - y(1 - mu)^2  
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
    this->D0 = M_PI*(1 - x/3 - y/6);
    this->type = QUADRATIC;
    this->nr_par = 2;  
  }
  
  T D(const T &mu) const {
    T u = 1 - mu;
    return 1 - u*(x + y*u);
  }
  
  bool check() const {
    return (x + y < 1 && x > 0 && y > 0);
  }
};

// Nonlinear limb darkening (3 parameters)
// D(x) = 1 - x(1 - mu) - y (1 - mu)^p
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
    this->D0 = M_PI*(1 - x/3 - y/(1 + p*(3 + p)/2));
    this->type = NONLINEAR;
    this->nr_par = 3;
  }
  
  T D(const T &mu) const {
    T u = 1 - mu;
    return 1 - x*u - y*std::pow(u, p); 
  }
  
  bool check() const {
    return (x + y < 1 && x > 0 && y > 0);  
  }
};

// Logarithmic limb darkening (2 parameters)
// D(x) = 1 - x*(1-mu) - y*mu*log(mu)
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
    this->D0 = M_PI*(1 - x/3  + 2*y/9);
    this->type = LOGARITHMIC;
    this->nr_par = 2;
  }
  
  T D(const T &mu) const {
    return 1 - x*(1 - mu) - y*mu*std::log(mu);
  }
  
  bool check() const {
    return (x < 1 && x > 0 && y > 0); //???????
  }
};


// Square-root limb darkening (2 parameters)
// D(x) = 1 - x(1 - mu) - y(1 - sqrt(mu))
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
    this->D0 = M_PI*(1 - x/3 - y/5);
    this->type = SQUARE_ROOT;
    this->nr_par = 2;   
  }
  
  T D(const T & mu) const {
    return 1 - x*(1 - mu) - y*(1 - std::sqrt(mu));
  }
  
  bool check() const {
    return (x + y < 1 && x > 0 && y > 0);
  }
};

/* ====================================================================
  Interface to the limb darkening models through the function
 ==================================================================== */



namespace LD{


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
      case UNIFORM: return 1;
      case LINEAR:  return 1 - p[0]*(1 - mu);
      case QUADRATIC: return 1 - (1 - mu)*(p[0] + (1 - mu)*p[1]);
      case NONLINEAR: return 1 - p[0]*(1 - mu) - p[1]*std::pow(1 - mu, p[2]);
      case LOGARITHMIC: return 1 - p[0]*(1 - mu) - p[1]*mu*std::log(mu);
      case SQUARE_ROOT: return 1 - p[0]*(1 - mu) - p[1]*(1- std::sqrt(mu));
      default:
        std::cerr << "LD::D::This model is not supported\n";
        return std::nan("");
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
      case UNIFORM: return M_PI;
      case LINEAR:  return M_PI*(1 - p[0]/3);
      case QUADRATIC: return M_PI*(1 - p[0]/3 - p[1]/6);
      case NONLINEAR: return M_PI*(1 - p[0]/3 - p[1]/(1 + p[2]*(3 + p[2])/2));
      case LOGARITHMIC: return M_PI*(1 - p[0]/3  + 2*p[1]/9);
      case SQUARE_ROOT: return  M_PI*(1 - p[0]/3 - p[1]/5);
      default:
        std::cerr << "LD::D0::This model is not supported\n";
        return std::nan("");
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
      case LINEAR:  return 1;
      case QUADRATIC: return 2;
      case NONLINEAR: return 3;
      case LOGARITHMIC: return 2;
      case SQUARE_ROOT: return 2;
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
    
    switch (fnv1a_64::hash(s)){
      case "uniform"_hash: return UNIFORM;
      case "linear"_hash:  return LINEAR;
      case "quadratic"_hash: return QUADRATIC;
      case "nonlinear"_hash: return NONLINEAR;
      case "logarithmic"_hash: return LOGARITHMIC;
      case "square_root"_hash: return SQUARE_ROOT;
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
      case LINEAR: g[0] = mu - 1; break;
      case QUADRATIC: g[0] = mu - 1; g[1] = -g[0]*g[0]; break;
      case NONLINEAR:
        g[0] = mu - 1; 
        g[1] = -std::pow(-g[0], p[2]); 
        g[2] = p[1]*g[1]*std::log(-g[0]);
      break;
      case LOGARITHMIC: g[0] = mu - 1; g[1] = -mu*std::log(mu); break;
      case SQUARE_ROOT: g[0] = mu - 1; g[1] = std::sqrt(mu) - 1; break;
      
      default:
       std::cerr << "LD::gradparD::This model is not supported\n";
    }
    
  }
  
}

#endif
