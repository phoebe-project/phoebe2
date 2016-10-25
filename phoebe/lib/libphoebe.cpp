/* 
  Wrappers for calculations concerning the 
    
    * generalized Kopal potential or Roche lobes
    * rotating stars  
    * triangular meshes
  
  Need to install for Python.h header:
    apt-get install python-dev
  
  Ref:
  
  Python C-api: 
  * https://docs.python.org/2/c-api/index.html 

  Numpi C-api manual:
  * http://docs.scipy.org/doc/numpy/reference/c-api.html
  * http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
  * http://scipy.github.io/old-wiki/pages/Cookbook/C_Extensions/NumPy_arrays
  
  Wrapping tutorial:
  * http://intermediate-and-advanced-software-carpentry.readthedocs.io/en/latest/c++-wrapping.html
  
  Author: Martin Horvat, August 2016
*/

#include <iostream>
#include <vector>
#include <typeinfo>
#include <algorithm>
#include <cstring>
#include <cstdint>

#include "utils.h"                // General routines

#include "triang_mesh.h"           // Dealing with triangular meshes
#include "triang_marching.h"       // Maching triangulation
#include "bodies.h"                // Definitions of different potentials
#include "eclipsing.h"             // Eclipsing/Hidden surface removal
#include "povray.h"                // Exporting meshes to povray (minimalistic)
#include "reflection.h"            // Dealing with reflection effects/radiosity problem
#include "horizon.h"               // Calculation of horizons
    
#include "gen_roche.h"             // support for generalized Roche lobes 
#include "rot_star.h"              // support for rotating stars

#include "wd_atm.h"                // Wilson-Devinney atmospheres
#include "interpolation.h"         // Nulti-dimensional linear interpolation

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// providing the Python interface -I/usr/include/python2.7/
#include <Python.h>
#include <numpy/arrayobject.h>


// Porting to Python 3
// Ref: http://python3porting.com/cextensions.html
#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
          PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);

  // adding missing declarations and functions
  #define PyString_Type PyBytes_Type
  #define PyString_AsString PyBytes_AsString
  #define PyString_Check PyBytes_Check
  
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);
#endif

 
/*
  Getting the Python typename 
*/

template<typename T> NPY_TYPES PyArray_TypeNum();

template<>  NPY_TYPES PyArray_TypeNum<int>() { return NPY_INT;}
template<>  NPY_TYPES PyArray_TypeNum<double>() { return NPY_DOUBLE;}


/*
  Insert into dictionary and deferences the inserted object
  Ref:
  https://robinelvin.wordpress.com/2011/03/24/python-c-extension-memory-leak/
*/

int PyDict_SetItemStringStealRef(PyObject *p, const char *key, PyObject *val){
 
  int status = PyDict_SetItemString(p, key, val);
 
  Py_XDECREF(val);
 
  return status;
}

template <typename T>
PyObject *PyArray_FromVector(std::vector<T> &V){
  
  int N = V.size();
  
  T *p = new T [N];
  
  std::copy(V.begin(), V.end(), p);
  
  npy_intp dims[1] = {N};
  
  PyObject *pya = 
    PyArray_SimpleNewFromData(1, dims, PyArray_TypeNum<T>(), p); 

  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}


template<typename T>
void PyArray_ToVector(PyArrayObject *oV, std::vector<T> & V){
  
  T *V_begin = (T*) PyArray_DATA(oV);
  
  V.assign(V_begin, V_begin + PyArray_DIM(oV, 0));
}

template <typename T>
PyObject *PyArray_From3DPointVector(std::vector<T3Dpoint<T>> &V){
  
  // Note: p is interpreted in C-style contiguous fashion
  int N = V.size();
  
  T *p = new T [3*N], *b = p; 
  
  for (auto && v : V)
    for (int i = 0; i < 3; ++i) *(b++) = v[i];
  
  npy_intp dims[2] = {N, 3};
  
  PyObject *pya =
    PyArray_SimpleNewFromData(2, dims, PyArray_TypeNum<T>(), p); 
  
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}


template <typename T>
void PyArray_To3DPointVector(
  PyArrayObject *oV, 
  std::vector<T3Dpoint<T>> &V){
   
  // Note: p is interpreted in C-style contiguous fashion
  int N = PyArray_DIM(oV, 0);
  
  V.reserve(N);
  
  for (T *p = (T*) PyArray_DATA(oV), *p_e = p + 3*N; p != p_e; p += 3)
    V.emplace_back(p);
}

/*
  C++ wrapper for Python code:
  
  Calculate critical potential, that is the value of the generalized 
  Kopal potential in Lagrange points.
  
  Python:
    
    omega = roche_critical_potential(q, F, d, <keywords>=<value>)
  
  where parameters are
  
  positionals:
  
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
  
  keywords: optional
    
    L1: boolean, default true
      switch calculating value of the potential at L1 
    L2: boolean, default true
      switch calculating value of the potential at L2 
    L3: boolean, default true
      switch calculating value of the potential at L3 
    
  and returns dictionary with keywords:
  
    L1:
      float: value of the potential as L1
    L2:
      float: value of the potential as L2
    L3:
      float: value of the potential as L3

*/

static PyObject *roche_critical_potential(PyObject *self, PyObject *args, PyObject *keywds) {
  
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"L1",
    (char*)"L2",
    (char*)"L3",
    NULL};
         
  bool b_L[3] = {true, true, true};
     
  double q, F, delta;
  
  PyObject *o_L[3] = {0,  0, 0};
  
  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "ddd|O!O!O!", kwlist,
        &q, &F, &delta, 
        &PyBool_Type, o_L,
        &PyBool_Type, o_L + 1,
        &PyBool_Type, o_L + 2)
  ){
    std::cerr << "roche_critical_potential:Problem reading arguments\n";
    return NULL;
  }
  
  // reading selection
  for (int i = 0; i < 3; ++i)
    if (o_L[i]) b_L[i] = PyObject_IsTrue(o_L[i]);
  
  // create a binary version of selection
  unsigned choice = 0;
  for (int i = 0, j = 1; i < 3; ++i, j<<=1) 
    if (b_L[i]) choice += j;
  
  //
  // Do calculations
  //
  double omega[3], L[3];
  
  gen_roche::critical_potential(omega, L, choice, q, F, delta);

  PyObject *results = PyDict_New();
    
  //
  // Store results
  //  
  const char *labels[] = {"L1","L2", "L3"};
  
  for (int i = 0; i < 3; ++i)
    if (b_L[i]) 
      PyDict_SetItemStringStealRef(results, labels[i], PyFloat_FromDouble(omega[i]));
  
  return results;
}

/*
  C++ wrapper for Python code:
  
  Calculate critical potential of the rotating star potential.
  
  Python:
    
    Omega_crit = rotstar_critical_potential(omega)
  
  where parameters are
  
    omega: float - parameter of the potential
  
  and returns a float
  
    omega:  float
*/

static PyObject *rotstar_critical_potential(PyObject *self, PyObject *args) {
    
  // parse input arguments   
  double omega;
  
  if (!PyArg_ParseTuple(args, "d", &omega)){
    std::cerr << "rotstar_critical_potential:Problem reading arguments\n";
    return NULL;
  }
  
  if (omega == 0) return NULL; // there is no critical value

  return PyFloat_FromDouble(rot_star::critical_potential(omega));
}


/*
  C++ wrapper for Python code:
  
  Calculate height h the left (right) Roche lobe at the position 
    
    (0,0,h)   (  (d,0,h) )
    
  of the primary (secondary) star. Roche lobe(s) is defined as equipotential 
  of the generalized Kopal potential Omega:

      Omega_0 = Omega(x,y,z)
  
  Python:
    
    h = roche_pole(q, F, d, Omega0, <keywords> = <value>)
  
  where parameters are
  
  positionals:
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega: float - value potential 
  
  keywords:
    choice: integer, default 0
            0 for discussing left lobe, 1 for discussing right lobe
    
  and return float
  
    h : height of the lobe's pole
*/

static PyObject *roche_pole(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"choice",
    NULL};
       
  int choice = 0;
  
  double q, F, delta, Omega0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dddd|i", kwlist, &q, &F, &delta, &Omega0, &choice)){
    std::cerr << "roche_pole:Problem reading arguments\n";
    return NULL;
  }
  
  //
  // Compute the poles
  //   
  if (choice == 0)  // Left lobe
    return PyFloat_FromDouble(gen_roche::poleL(Omega0, q, F, delta));
  
  // Right lobe  
  return PyFloat_FromDouble(gen_roche::poleR(Omega0, q, F, delta));
}


/*
  C++ wrapper for Python code:
  
  Calculate height h the rotating star
    
    (0,0,h) 
    
  The lobe of the rotating star is defined as equipotential 
  of the potential Omega:

      Omega_0 = Omega(x,y,z) = 1/r + 1/2 omega^2 (x^2 + y^2)
  
  Python:
    
    h = rotstar_pole(omega, Omega0, <keywords> = <value>)
  
  where parameters are
  
  positionals:
    omega: float - parameter of the potential
    Omega: float - value potential 
      
  and return float
  
    h : height of the lobe's pole
*/

static PyObject *rotstar_pole(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"omega",
    (char*)"Omega0",
    NULL};
  
  double omega, Omega0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dd", kwlist, &omega, &Omega0)){
    std::cerr << "rotstar_pole:Problem reading arguments\n";
    return NULL;
  }
  
  return PyFloat_FromDouble(1/Omega0);
}


/*
  C++ wrapper for Python code:
  
  Calculate parameters of the rotating star from Roche binary model by 
  matching the poles and centrifugal force.
  
  Python:
    
   param_rotstar
      = rotstar_from_roche(q, F, d, Omega0, <keywords> = <value>)
  
  where parameters are
   
    positionals:
    
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      delta: float - separation between the two objects
      Omega0: float - value potential 

    keywords:

      choice: default 0
        0 - primary star
        1 - secondary star -- not yet supported
        2 - overcontact -- not permitted
      
  and returns vector of parameters float
  
    param_rotstar : 1-rank numpy array = (omega_rotstar, Omega0_rotstar) 
*/

static PyObject *rotstar_from_roche(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"delta",
    (char*)"Omega0",
    (char*)"choice",
    NULL};
  
  int choice = 0;
  
  double q, F, delta, Omega0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dddd|i", kwlist, 
      &q, &F, &delta, &Omega0, 
      &choice)
    ) {
    std::cerr << "rotstar_from_roche:Problem reading arguments\n";
    return NULL;
  }
  
  if (choice != 0) {
    std::cerr 
      << "rotstar_from_roche::Choice != 0 is not yet supported\n";
    return NULL;
  }
  
  double *data = new double [2];
  
  data[0] = F*std::sqrt(1 + q);
  data[1] = 1/gen_roche::poleL(Omega0, q, F, delta);
  
  npy_intp dims[1] = {2};
  
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);   
      
  return  pya;
}

/*
  C++ wrapper for Python code:
  
  Calculate area and volume of the Roche lobe(s) is defined as 
  equipotential of the generalized Kopal potential Omega:

      Omega_0 = Omega(x,y,z)
  
  Python:
    
    dict = roche_area_volume(q, F, d, Omega0, <keyword>=<value>)
  
  where parameters are
  
  positionals:
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega: float - value potential 
  
  keywords:
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe
            2 for discussing overcontact 
            
    lvolume: boolean, default True
    larea: boolean, default True
    
    epsA : float, default 1e-12
      relative precision of the area

    epsV : float, default 1e-12
      relative precision of the volume

  Returns:
  
    dictionary
  
  with keywords
  
    lvolume: volume of the left or right Roche lobe  
      float:  
      
    larea: area of the left or right Roche lobe
      float:
*/

//#define DEBUG
static PyObject *roche_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"choice",
    (char*)"larea",
    (char*)"lvolume",
    (char*)"epsA",
    (char*)"epsV",
    NULL};
       
  int choice = 0;
 
  double eps[2] = {1e-12, 1e-12};
  
  bool b_av[2] = {true, true};  // b_larea, b_lvolume
  
  PyObject *o_av[2] = {0, 0};    // *o_larea = 0, *o_lvolume = 0;
  
  double q, F, delta, Omega0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dddd|iO!O!dd", kwlist, 
      &q, &F, &delta, &Omega0, 
      &choice,
      &PyBool_Type, o_av,
      &PyBool_Type, o_av + 1,
      eps, eps + 1
      )
    ) {
    std::cerr << "roche_area_volume:Problem reading arguments\n";
    return NULL;
  }
  
  unsigned res_choice = 0;
 
  //
  // Read boolean variables and define result-choice
  //   
  for (int i = 0, j = 1; i < 2; ++i, j <<=1) {
    if (o_av[i]) b_av[i] = PyObject_IsTrue(o_av[i]);
    if (b_av[i]) res_choice += j;
  }
  
  if (res_choice == 0) return NULL;
  

  //
  // Posibility of using approximation
  //
  
  double 
    w = delta*Omega0,
    b = (1 + q)*F*F*delta*delta*delta,
    w0 = 5*(q + std::cbrt(b*b)/4) - 29.554 - 5.26235*std::log(std::min(eps[0], eps[1])),
    av[2];                          // for results
    
  if (choice == 0 && w >= std::max(10., w0)) {
    
    // Approximation by using the series 
    // with empirically derived criterion 
    
    gen_roche::area_volume_primary_approx_internal(av, res_choice, Omega0, w, q, b);
    
  } else { 
    
    // Approximation by integration over the surface
    // relative precision should be better than 1e-12
    
    //
    // Choosing boundaries on x-axis
    //
    
    double xrange[2];
      
    if (!gen_roche::lobe_xrange(xrange, choice, Omega0, q, F, delta, true)){
      std::cerr << "roche_area_volume:Determining lobe's boundaries failed\n";
      return NULL;
    }
    
    //
    // Calculate area and volume:
    //

    const int m_min = 1 << 6;  // minimal number of points along x-axis
    
    int m0 = m_min;            // starting number of points alomg x-axis  
        
    bool 
      polish = false,
      adjust = true;
      
    double p[2][2], e, t;
        
    #if defined(DEBUG)
    std::cerr.precision(16);
    std::cerr << std::scientific;
    #endif
    
    //
    // one step adjustment of precison for area and volume
    // calculation
    //
    
    do {
        
      for (int i = 0, m = m0; i < 2; ++i, m <<= 1) {
        gen_roche::area_volume_integration
          (p[i], res_choice, xrange, Omega0, q, F, delta, m, polish);
        
        #if defined(DEBUG)
        std::cerr << "P:" << p[i][0] << '\t' << p[i][1] << '\n';
        #endif
      }
      
      if (adjust) {
           
        // extrapolation based on assumption
        //   I = I0 + a_1 h^4
        // estimating errors
        
        int m0_next = m0;
        
        adjust = false;
        
        for (int i = 0; i < 2; ++i) if (b_av[i]) {
          // best approximation
          av[i] = t = (16*p[1][i] - p[0][i])/15;
          
          // relative error
          e = std::max(std::abs(p[0][i]/t - 1), 16*std::abs(p[1][i]/t - 1));
          
          #if defined(DEBUG)
          std::cerr << "err=" << e << " m0=" << m0 << '\n';
          #endif
          
          if (e > eps[i]) {
            int k = int(1.1*m0*std::pow(e/eps[i], 0.25));
            if (k > m0_next) {
              m0_next = k;
              adjust = true;
            }
          }
        }
        
        if (adjust) m0 = m0_next; else break;
      
      } else {
        // best approximation
        for (int i = 0; i < 2; ++i) 
          if (b_av[i]) av[i] = (16*p[1][i] - p[0][i])/15;
        break;
      }
      
      adjust = false;
      
    } while (1);
    
  }
   
  PyObject *results = PyDict_New();
  
  const char *str[2] =  {"larea", "lvolume"};
  
  for (int i = 0; i < 2; ++i) if (b_av[i])
    PyDict_SetItemStringStealRef(results, str[i], PyFloat_FromDouble(av[i]));

  return results;
}
//#undef DEBUG

/*
  C++ wrapper for Python code:
  
  Calculate area and volume of rotating star lobe is defined as 
  equipotential of the potential Omega:

      Omega_0 = Omega(x,y,z) = 1/r  + 1/2 omega^2  (x^2 + y^2)
  
  Python:
    
    dict = rotstar_area_and_volume(omega, Omega0, <keyword>=<value>)
  
  where parameters are
  
  positionals:
    omega: float - parameter of the potential
    Omega: float - value potential 
  
  keywords:
  
    lvolume: boolean, default True
    larea: boolean, default True
    
  Returns:
  
    dictionary
  
  with keywords
  
    lvolume: volume of the lobe of the rotating star  
      float:  
      
    larea: area of the lobe of the rotating star
      float:
*/

static PyObject *rotstar_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"omega",
    (char*)"Omega0",
    (char*)"larea",
    (char*)"lvolume",
    NULL};
  
  bool 
    b_larea = true,
    b_lvolume = true;
        
  PyObject
    *o_larea = 0,
    *o_lvolume = 0;
  
  double omega, Omega0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dd|O!O!", kwlist, 
      &omega, &Omega0, 
      &PyBool_Type, &o_larea,
      &PyBool_Type, &o_lvolume
      )
    ) {
    std::cerr << "rotstar_area_volume:Problem reading arguments\n";
    return NULL;
  }
  
  if (o_larea) b_larea = PyObject_IsTrue(o_larea);
  if (o_lvolume) b_lvolume = PyObject_IsTrue(o_lvolume);
  
  if (!b_larea && !b_lvolume) return NULL;
 
  //
  // Calculate area and volume
  //
  
  unsigned res_choice = 0;
  
  if (b_larea) res_choice |= 1u;
  if (b_lvolume) res_choice |= 2u;
  
  double av[2] = {0,0};
  
  rot_star::area_volume(av, res_choice, Omega0, omega);
    
  PyObject *results = PyDict_New();
      
  if (b_larea)
    PyDict_SetItemStringStealRef(results, "larea", PyFloat_FromDouble(av[0]));

  if (b_lvolume)
    PyDict_SetItemStringStealRef(results, "lvolume", PyFloat_FromDouble(av[1]));
  
  return results;
}

/*
  C++ wrapper for Python code:
  
  Calculate the value of the generalized Kopal potential Omega1 corresponding 
  to parameters (q,F,d) and the volume of the Roche lobes equals to vol.  
    
  The Roche lobe(s) is defined as equipotential of the generalized
  Kopal potential Omega:

      Omega_i = Omega(x,y,z; q, F, d)
  
  Python:
    
    Omega1 = roche_Omega_at_vol(vol, q, F, d, Omega0, <keyword>=<value>)
  
  where parameters are
  
  positionals:
    vol: float - volume of the Roche lobe
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega0: float - guess for value potential Omega1
    
  keywords: (optional)
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe
            2 for discussing overcontact
    precision: float, default 1e-12
      aka relative precision
    accuracy: float, default 1e-12
      aka absolute precision
    max_iter: integer, default 100
      maximal number of iterations in the Newton-Raphson
    
  Returns:
  
    Omega1 : float
      value of the Kopal potential for (q,F,d1) such that volume
      is equal to the case (q,F,d,Omega0)
*/

//#define DEBUG
static PyObject *roche_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"vol",
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"choice",
    (char*)"precision",
    (char*)"accuracy",
    (char*)"max_iter",
    NULL};
       
  int 
    choice = 0;
    
  double
    vol, 
    q, F, delta, 
    Omega0 = nan(""),
    precision = 1e-12,
    accuracy = 1e-12;
  
  int max_iter = 100;  
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddddd|iddi", kwlist, 
      &vol, &q, &F, &delta, &Omega0,
      &choice,
      &precision,
      &accuracy,
      &max_iter
      )
    ) {
    std::cerr << "roche_Omega_at_vol:Problem reading arguments\n";
    return NULL;
  }
    
  bool b_Omega0 = !std::isnan(Omega0);
  
  if (!b_Omega0) {
    std::cerr << "Currently not supporting lack of guessed Omega.\n";
    return NULL;
  }
     
  const int m_min = 1 << 6;  // minimal number of points along x-axis
    
  int 
    m0 = m_min,  // minimal number of points along x-axis
    it = 0;      // number of iterations

  double 
    Omega = Omega0, dOmega, 
    V[2], xrange[2], p[2][2];
  
  bool polish = false;
  
  #if defined(DEBUG)
  std::cerr.precision(16); 
  std::cerr << std::scientific;
  #endif
  
  // expected precisions of the integrals
  double eps = precision/2;
  
  do {

    if (!gen_roche::lobe_xrange(xrange, choice, Omega, q, F, delta, true)){
      std::cerr << "roche_area_volume:Determining lobe's boundaries failed\n";
      return NULL;
    }
    
    // adaptive calculation of the volume and its derivative
    bool adjust = true;
    
    do {
      
      #if defined(DEBUG)
      std::cerr << "it=" << it << '\n';
      #endif
       
      // calculate volume and derivate volume w.r.t to Omega
      for (int i = 0, m = m0; i < 2; ++i, m <<= 1) {
        gen_roche::volume(p[i], 3, xrange, Omega, q, F, delta, m, polish);
        
        #if defined(DEBUG)
        std::cerr << "V:" <<  p[i][0] << '\t' << p[i][1] << '\n';
        #endif
      }
     
      if (adjust) {
        
        // extrapolations based on the expansion
        // I = I0 + a1 h^4 + a2 h^5 + ...
        // result should have relative precision better than 1e-12 
        
        int m0_next = m0;
        
        double e, t;
        
        adjust = false;
        
        for (int i = 0; i < 2; ++i) {
          
          // best estimate
          V[i] = t = (16*p[1][i] - p[0][i])/15;
          
          // relative error
          e = std::max(std::abs(p[0][i]/t - 1), 16*std::abs(p[1][i]/t - 1));
          
          #if defined(DEBUG)
          std::cerr << "e=" << e << " m0 =" << m0 << '\n';
          #endif
          
          if (e > eps) {
            int k = int(1.1*m0*std::pow(e/eps, 0.25));
            if (k > m0_next) {
              m0_next = k;
              adjust = true;
            }  
          }
        }
        
        if (adjust) m0 = m0_next; else break;
        
      } else {
        // just calculate best estimate, 
        // as the precision should be adjusted
        for (int i = 0; i < 2; ++i) V[i] =(16*p[1][i] - p[0][i])/15;
        break;
      }
      
      adjust = false;
      
    } while (1); 
        
    Omega -= (dOmega = (V[0] - vol)/V[1]);
    
    #if defined(DEBUG) 
    std::cerr 
      << "Omega=" << Omega 
      << "\tvol=" << vol 
      << "\tV[0]= " << V[0] 
      << "\tdOmega=" << dOmega << '\n';
    #endif
    
  } while (std::abs(dOmega) > accuracy + precision*Omega && ++it < max_iter);
   
  if (!(it < max_iter)){
    std::cerr << "roche_Omega_at_vol: Maximum number of iterations exceeded\n";
    return NULL;
  }
  // We use the condition on the argument (= Omega) ~ constraining backward error, 
  // but we could also use condition on the value (= Volume) ~ constraing forward error
  
  return PyFloat_FromDouble(Omega);
}
#undef DEBUG

/*
  C++ wrapper for Python code:
  
  Calculate the value of the rotating star potential Omega1 at parameter
  omega and star's volume equal to vol.  
    
  The  rotating star is defined as equipotential of the generalized
  Kopal potential Omega:

      Omega_i = Omega(x,y,z; omega) = 1/r + 1/2 omega^2 (x^2 + y^2)
  
  Python:
    
    Omega1 = rotstar_Omega_at_vol(vol, omega, Omega0, <keyword>=<value>)
  
  where parameters are
  
  positionals:
    vol: float - volume of the star's lobe
    omega: float  - parameter of the potential
    Omega0 - guess for value potential Omega1
  
  keywords: (optional)
    precision: float, default 1e-12
      aka relative precision
    accuracy: float, default 1e-12
      aka absolute precision
    max_iter: integer, default 100
      maximal number of iterations in the Newton-Raphson
    
  Returns:
  
    Omega1 : float
      value of the Kopal potential for (q,F,d1) such that volume
      is equal to the case (q,F,d,Omega0)
*/

//#define DEBUG
static PyObject *rotstar_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //
  
  char *kwlist[] = {
    (char*)"vol",
    (char*)"omega",
    (char*)"Omega0",
    (char*)"precision",
    (char*)"accuracy",
    (char*)"max_iter",
    NULL};
    
  double
    vol, omega, 
    Omega0 = nan(""),
    precision = 1e-12,
    accuracy = 1e-12;
  
  int max_iter = 100;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddd|ddi", kwlist, 
      &vol, &omega, &Omega0,
      &precision,
      &accuracy,
      &max_iter
      )
    ) {
    std::cerr << "rotstar_Omega_at_vol:Problem reading arguments\n";
    return NULL;
  }
    
  bool b_Omega0 = !std::isnan(Omega0);
  
  if (!b_Omega0) {
    std::cerr << "Currently not supporting lack of guessed Omega.\n";
    return NULL;
  }
    
  int it = 0;
      
  double Omega = Omega0, dOmega, V[2] = {0,0};
  
  #if defined(DEBUG)
  std::cout.precision(16); std::cout << std::scientific;
  #endif
  do {

    rot_star::volume(V, 3, Omega, omega);
        
    Omega -= (dOmega = (V[0] - vol)/V[1]);
    
    #if defined(DEBUG) 
    std::cout 
      << "Omega=" << Omega 
      << "\tvol=" << vol 
      << "\tV[0]= " << V[0] 
      << "\tdOmega=" << dOmega << '\n';
    #endif
    
  } while (std::abs(dOmega) > accuracy + precision*Omega && ++it < max_iter);
   
  if (!(it < max_iter)){
    std::cerr << "rotstar_Omega_at_vol: Maximum number of iterations exceeded\n";
    return NULL;
  }
  // We use the condition on the argument (= Omega) ~ constraining backward error, 
  // but we could also use condition on the value (= Volume) ~ constraing forward error
  
  return PyFloat_FromDouble(Omega);
}

/*
  C++ wrapper for Python code:
  
  Calculate the gradient and the value of the potential of the generalized
  Kopal potential Omega at a given point

      -grad Omega (x,y,z)
  
  which is outwards the Roche lobe.
  
  
  Python:
    
    g = roche_gradOmega(q, F, d, r)
   
   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      r: 1-rank numpy array of length 3 = [x,y,z]
   
  
  and returns float
  
    g : 1-rank numpy array 
      = [-grad Omega_x, -grad Omega_y, -grad Omega_z, -Omega(x,y,z)]
*/


static PyObject *roche_gradOmega(PyObject *self, PyObject *args) {
    
  double p[4];  

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dddO!", p, p + 1, p + 2, &PyArray_Type, &X)) {
    std::cerr << "roche_gradOmega:Problem reading arguments\n";
    return NULL;
  }

  p[3] = 0;
  
  Tgen_roche<double> b(p);
  
  double *g = new double [4];

  b.grad((double*)PyArray_DATA(X), g);
  
  npy_intp dims[1] = {4};

  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}

/*
  C++ wrapper for Python code:
  
  Calculate the gradient and the value of the potential of the generalized
  Kopal potential Omega at a given point

      -grad Omega (x,y,z)
  
  which is outwards the Roche lobe.
  
  
  Python:
    
    g = rot_gradOmega(omega, r)
   
   with parameters
      omega: float - parameter of the potential
      r: 1-rank numpy array of length 3 = [x,y,z]
  
  and returns float
  
    g : 1-rank numpy array 
      = [-grad Omega_x, -grad Omega_y, -grad Omega_z, -Omega(x,y,z)]
*/


static PyObject *rotstar_gradOmega(PyObject *self, PyObject *args) {
    
  double p[2];  

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dO!", p, &PyArray_Type, &X)) {
    std::cerr << "rotstar_gradOmega:Problem reading arguments\n";
    return NULL;
  }

  p[1] = 0;
  
  Trot_star<double> b(p);
  
  double *g = new double [4];

  b.grad((double*)PyArray_DATA(X), g);
  
  npy_intp dims[1] = {4};

  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
    
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}
 
/*
  C++ wrapper for Python code:
  
  Calculate the gradient of the potential of the generalized
  Kopal potential Omega at a given point

      -grad Omega (x,y,z)
  
  Python:
    
    g = roche_gradOmega_only(q, F, d, r)
   
   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      r: 1-rank numpy array of length 3 = [x,y,z]
   
  
  and returns float
  
    g : 1-rank numpy array = -grad Omega (x,y,z)
*/

static PyObject *roche_gradOmega_only(PyObject *self, PyObject *args) {

  double p[4];

  PyArrayObject *X;  
  
  if (!PyArg_ParseTuple(args, "dddO!", p, p + 1, p + 2, &PyArray_Type, &X)) {
    std::cerr << "roche_gradOmega_only:Problem reading arguments\n";
    return NULL;
  }

  Tgen_roche<double> b(p);
  
  double *g = new double [3];

  b.grad_only((double*)PyArray_DATA(X), g);

  npy_intp dims[1] = {3};

  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);

  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}

/*
  C++ wrapper for Python code:
  
  Calculate the gradient of the potential of the rotating star potential

      -grad Omega (x,y,z)
  
  Python:
    
    g = rotstar_gradOmega_only(omega, r)
   
   with parameters
    
      omega: float - parameter of the potential
      r: 1-rank numpy array of length 3 = [x,y,z]
   
  
  and returns float
  
    g : 1-rank numpy array = -grad Omega (x,y,z)
*/

static PyObject *rotstar_gradOmega_only(PyObject *self, PyObject *args) {

  double p[2];

  PyArrayObject *X;  
  
  if (!PyArg_ParseTuple(args, "dO!", p, &PyArray_Type, &X)) {
    std::cerr << "rotstar_gradOmega_only:Problem reading arguments\n";
    return NULL;
  }
  
  Trot_star<double> b(p);
  
  double *g = new double [3];

  b.grad_only((double*)PyArray_DATA(X), g);

  npy_intp dims[1] = {3};

  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);

  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}


/*
  C++ wrapper for Python code:
  
  Calculate the value of the potential of the generalized
  Kopal potential Omega at a given point

      Omega (x,y,z; q, F, d)
  
  Python:
    
    Omega0 = roche_Omega(q, F, d, r)
   
   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      r: 1-rank numpy array of length 3 = [x,y,z]
   
  
  and returns a float
  
    Omega0 - value of the Omega at (x,y,z)
*/

static PyObject *roche_Omega(PyObject *self, PyObject *args) {

  double p[4];

  PyArrayObject *X;  
  
  if (!PyArg_ParseTuple(args, "dddO!", p, p + 1, p + 2, &PyArray_Type, &X)){
    std::cerr << "roche_Omega:Problem reading arguments\n";
    return NULL;
  }
  
  p[3] = 0; // Omega
  
  Tgen_roche<double> b(p);
  
  return PyFloat_FromDouble(-b.constrain((double*)PyArray_DATA(X)));
}

/*
  C++ wrapper for Python code:
  
  Calculate the value of the potential of the rotating star at 
  a given point

      Omega (x,y,z; omega)
  
  Python:
    
    Omega0 = rotstar_Omega(omega, r)
   
   with parameters
  
      omega: float - parameter of the potential
      r: 1-rank numpy array of length 3 = [x,y,z]
   
  
  and returns a float
  
    Omega0 - value of the Omega at (x,y,z)
*/

static PyObject *rotstar_Omega(PyObject *self, PyObject *args) {

  double p[2];

  PyArrayObject *X;  
  
  if (!PyArg_ParseTuple(args, "dO!", p, &PyArray_Type, &X)) {
    std::cerr << "rotstar_Omega:Problem reading arguments\n";
    return NULL;
  }

  p[1] = 0; // Omega
  
  Trot_star<double> b(p);

  return PyFloat_FromDouble(-b.constrain((double*)PyArray_DATA(X)));
}

/*
  C++ wrapper for Python code:

    Marching meshing of Roche lobes implicitely defined by generalized 
    Kopal potential:
    
      Omega_0 = Omega(x,y,z)
    
  Python:

    dict = marching_mesh(q, F, d, Omega0, delta, <keyword>=[true,false], ... )
    
  where parameters
  
    positional:
  
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      Omega0: float - value of the generalized Kopal potential
      delta: float - size of triangles edges projected to tangent space
    
    keywords: 
      choice: integer, default 0
          0 - primary lobe is exists
          1 - secondary lobe is exists
        for overcontacts choice is 0 or 1
        choice controls where is the begining the triangulation

      max_triangles:integer, default 10^7 
        maximal number of triangles
        if number of triangles exceeds max_triangles it returns NULL  
      
      full: boolean, default False
        using full version of marching method as given in the paper 
        by (Hartmann, 1998)
        
      vertices: boolean, default False
      vnormals: boolean, default False
      vnormgrads:boolean, default False
      triangles: boolean, default False
      tnormals: boolean, default False
      areas: boolean, default False
      area: boolean, default False
      volume: boolean, default False
      centers: boolean, default False
      cnormals: boolean, default False
      cnormgrads: boolean, default False
      init_phi: float, default 0

  Returns:
  
    dictionary
  
  with keywords
  
    vertices: 
      V[][3]    - 2-rank numpy array of vertices 
    
    vnormals:
      NatV[][3] - 2-rank numpy array of normals at vertices
 
    vnormgrads:
      GatV[]  - 1-rank numpy array of norms of the gradients at central points
 
    triangles:
      T[][3]    - 2-rank numpy array of 3 indices of vertices 
                composing triangles of the mesh aka connectivity matrix
    
    tnormals:
      NatT[][3] - 2-rank numpy array of normals of triangles
  
    areas:
      A[]       - 1-rank numpy array of areas of triangles
    
    area:
      area      - area of triangles of mesh
    
    volume:
      volume    - volume of body enclosed by triangular mesh
      
    centers:
      C[][3]    - 2-rank numpy array of central points of triangles
                  central points is  barycentric points projected to 
                  Roche lobes
    cnormals:
      NatC[][3]   - 2-rank numpy array of normals of central points
 
    cnormgrads:
      GatC[]      - 1-rank numpy array of norms of the gradients at central points
    
    
  Typically face-vertex format is (V, T) where
  
    V - vertices
    T - connectivity matrix with indices labeling vertices in 
        counter-clockwise orientation so that normal vector is pointing 
        outward
  
  Refs:
  * for face-vertex format see https://en.wikipedia.org/wiki/Polygon_mesh
  * http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
  * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays
  * https://docs.python.org/2.0/ext/parseTupleAndKeywords.html
  * https://docs.python.org/2/c-api/arg.html#c.PyArg_ParseTupleAndKeywords
*/

static PyObject *roche_marching_mesh(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"delta",
    (char*)"choice",
    (char*)"max_triangles",
    (char*)"full",
    (char*)"vertices", 
    (char*)"vnormals",
    (char*)"vnormgrads",
    (char*)"triangles", 
    (char*)"tnormals", 
    (char*)"centers", 
    (char*)"cnormals",
    (char*)"cnormgrads",
    (char*)"areas",
    (char*)"area",
    (char*)"volume",
    (char*)"init_phi",
    NULL};
  
  double q, F, d, Omega0, delta, 
            init_phi = 0;   
  
  int choice = 0,               
      max_triangles = 10000000; // 10^7
      
  bool
    b_full = false,
    b_vertices = false, 
    b_vnormals = false, 
    b_vnormgrads = false,
    b_triangles = false, 
    b_tnormals = false, 
    b_centers = false,
    b_cnormals = false,
    b_cnormgrads = false,
    b_areas = false,
    b_area = false,
    b_volume = false;
    
  // http://wingware.com/psupport/python-manual/2.3/api/boolObjects.html
  PyObject
    *o_full = 0,
    *o_vertices = 0, 
    *o_vnormals = 0, 
    *o_vnormgrads = 0,
    *o_triangles = 0, 
    *o_tnormals = 0, 
    *o_centers = 0,
    *o_cnormals = 0,
    *o_cnormgrads = 0,
    *o_areas = 0,
    *o_area = 0,
    *o_volume = 0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddddd|iiO!O!O!O!O!O!O!O!O!O!O!O!d", kwlist,
      &q, &F, &d, &Omega0, &delta, // neccesary 
      &choice,                     // optional ...
      &max_triangles,
      &PyBool_Type, &o_full,
      &PyBool_Type, &o_vertices, 
      &PyBool_Type, &o_vnormals,
      &PyBool_Type, &o_vnormgrads,
      &PyBool_Type, &o_triangles, 
      &PyBool_Type, &o_tnormals,
      &PyBool_Type, &o_centers,
      &PyBool_Type, &o_cnormals,
      &PyBool_Type, &o_cnormgrads,
      &PyBool_Type, &o_areas,
      &PyBool_Type, &o_area,
      &PyBool_Type, &o_volume,
      &init_phi
      )) {
    std::cerr << "roche_marching_mesh:Problem reading arguments\n";
    return NULL;
  }
  
  if (o_full) b_full = PyObject_IsTrue(o_full);
  if (o_vertices) b_vertices = PyObject_IsTrue(o_vertices);
  if (o_vnormals) b_vnormals = PyObject_IsTrue(o_vnormals);
  if (o_vnormgrads) b_vnormgrads = PyObject_IsTrue(o_vnormgrads);
  if (o_triangles) b_triangles = PyObject_IsTrue(o_triangles);
  if (o_tnormals)  b_tnormals = PyObject_IsTrue(o_tnormals);
  if (o_centers) b_centers = PyObject_IsTrue(o_centers);
  if (o_cnormals) b_cnormals = PyObject_IsTrue(o_cnormals);
  if (o_cnormgrads) b_cnormgrads = PyObject_IsTrue(o_cnormgrads);
  if (o_areas) b_areas = PyObject_IsTrue(o_areas);
  if (o_area) b_area = PyObject_IsTrue(o_area);
  if (o_volume) b_volume = PyObject_IsTrue(o_volume);
     
  //
  // Storing results in dictioonary
  // https://docs.python.org/2/c-api/dict.html
  //
  PyObject *results = PyDict_New();
  
  
  if (choice < 0 || choice > 2){
    std::cerr << 
      "roche_marching_mesh::This choice is not supported\n"; 
    return NULL;
  }
    
  //
  // Choosing the meshing initial point 
  //
  
  double r[3], g[3];
  
  if (!gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, d)){
    std::cerr << "roche_marching_mesh:Determining initial meshing point failed\n";
    return NULL;
  }
  
  //
  //  Marching triangulation of the Roche lobe 
  //
    
  double params[4] = {q, F, d, Omega0};
  
  Tmarching<double, Tgen_roche<double>> march(params);  
  
  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr; 
  std::vector<double> *GatV = 0;
     
  if (b_vnormgrads) GatV = new std::vector<double>;
  
  
  if ((b_full ? 
       !march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi) :
       !march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
      )){
    std::cerr << "roche_marching_mesh::There are too many triangles\n";
    return NULL;
  }
  
  //
  // Calculte the mesh properties
  //
  int vertex_choice = 0;
  
  double 
    area, volume, 
    *p_area = 0, *p_volume = 0;
  
  std::vector<double> *A = 0; 
  
  std::vector<T3Dpoint<double>> *NatT = 0;
  
  if (b_areas) A = new std::vector<double>;
  
  if (b_area) p_area = &area;
  
  if (b_tnormals) NatT = new std::vector<T3Dpoint<double>>;
  
  if (b_volume) p_volume = &volume;
   
  mesh_attributes(V, NatV, Tr, A, NatT, p_area, p_volume, vertex_choice, true);
 
  //
  // Calculte the central points
  // 

  std::vector<double> *GatC = 0;
  
  std::vector<T3Dpoint<double>> *C = 0, *NatC = 0;
  
  if (b_centers) C = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormals) NatC = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormgrads) GatC = new std::vector<double>;
 
 
  march.central_points(V, Tr, C, NatC, GatC);
  
  
  if (b_vertices)
    PyDict_SetItemStringStealRef(results, "vertices", PyArray_From3DPointVector(V));

  if (b_vnormals)
    PyDict_SetItemStringStealRef(results, "vnormals", PyArray_From3DPointVector(NatV));

  if (b_vnormgrads) {
    PyDict_SetItemStringStealRef(results, "vnormgrads", PyArray_FromVector(*GatV));
    delete GatV;
  }
  
  if (b_triangles)
    PyDict_SetItemStringStealRef(results, "triangles", PyArray_From3DPointVector(Tr));

  if (b_areas) {
    PyDict_SetItemStringStealRef(results, "areas", PyArray_FromVector(*A));
    delete A;  
  }
  
  if (b_area)
    PyDict_SetItemStringStealRef(results, "area", PyFloat_FromDouble(area));

  if (b_tnormals) {
    PyDict_SetItemStringStealRef(results, "tnormals", PyArray_From3DPointVector(*NatT));
    delete NatT;
  }

  if (b_volume)
    PyDict_SetItemStringStealRef(results, "volume", PyFloat_FromDouble(volume));
  
  if (b_centers) {
    PyDict_SetItemStringStealRef(results, "centers", PyArray_From3DPointVector(*C));
    delete C;  
  }

  if (b_cnormals) {
    PyDict_SetItemStringStealRef(results, "cnormals", PyArray_From3DPointVector(*NatC));
    delete NatC;
  }
  
  if (b_cnormgrads) {
    PyDict_SetItemStringStealRef(results, "cnormgrads", PyArray_FromVector(*GatC));
    delete GatC;
  }
  
  return results;
}

/*
  C++ wrapper for Python code:

  Marching meshing of rotating star implicitely defined by the potential
    
      Omega_0 = Omega(x,y,z) = 1/r + 1/2 omega^2 (x^2 + y^2)
    
  Python:

    dict = rotstar_marching_mesh(omega, Omega0, delta, <keyword>=[true,false], ... )
    
  where parameters
  
    positional:
      omega: float - parameter of the potential
      Omega0: float - value of the generalized Kopal potential
      delta: float - size of triangles edges projected to tangent space
    
    keywords: 
      max_triangles:integer, default 10^7 
            maximal number of triangles
            if number of triangles exceeds max_triangles it returns NULL  
  
      full: boolean, default False
        using full version of marching method as given in the paper 
        by (Hartmann, 1998)

      vertices: boolean, default False
      vnormals: boolean, default False
      vnormgrads:boolean, default False
      triangles: boolean, default False
      tnormals: boolean, default False
      areas: boolean, default False
      area: boolean, default False
      volume: boolean, default False
      centers: boolean, default False
      cnormals: boolean, default False
      cnormgrads: boolean, default False
      init_phi: float, default 0

  Returns:
  
    dictionary
  
  with keywords
  
    vertices: 
      V[][3]    - 2-rank numpy array of vertices 
    
    vnormals:
      NatV[][3] - 2-rank numpy array of normals at vertices
 
    vnormgrads:
      GatV[]  - 1-rank numpy array of norms of the gradients at vertices
 
    triangles:
      T[][3]    - 2-rank numpy array of 3 indices of vertices 
                composing triangles of the mesh aka connectivity matrix
    
    tnormals:
      NatT[][3] - 2-rank numpy array of normals of triangles
  
    areas:
      A[]       - 1-rank numpy array of areas of triangles
    
    area:
      area      - area of triangles of mesh
    
    volume:
      volume    - volume of body enclosed by triangular mesh
      
    centers:
      C[][3]    - 2-rank numpy array of central points of triangles
                  central points is  barycentric points projected to 
                  Roche lobes
    cnormals:
      NatC[][3]   - 2-rank numpy array of normals of central points
 
    cnormgrads:
      GatC[]      - 1-rank numpy array of norms of the gradients at central points
    
    
  Typically face-vertex format is (V, T) where
  
    V - vertices
    T - connectivity matrix with indices labeling vertices in 
        counter-clockwise orientation so that normal vector is pointing 
        outward
  
  Refs:
  * for face-vertex format see https://en.wikipedia.org/wiki/Polygon_mesh
  * http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
  * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays
  * https://docs.python.org/2.0/ext/parseTupleAndKeywords.html
  * https://docs.python.org/2/c-api/arg.html#c.PyArg_ParseTupleAndKeywords
*/

static PyObject *rotstar_marching_mesh(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"omega",
    (char*)"Omega0",
    (char*)"delta",
    (char*)"max_triangles",
    (char*)"full",
    (char*)"vertices", 
    (char*)"vnormals",
    (char*)"vnormgrads",
    (char*)"triangles", 
    (char*)"tnormals", 
    (char*)"centers", 
    (char*)"cnormals",
    (char*)"cnormgrads",
    (char*)"areas",
    (char*)"area",
    (char*)"volume",
    (char*)"init_phi",
    NULL};
  
  double omega, Omega0, delta, init_phi = 0;   
  
  int max_triangles = 10000000; // 10^7
      
  bool 
    b_full = false,
    b_vertices = false, 
    b_vnormals = false, 
    b_vnormgrads = false,
    b_triangles = false, 
    b_tnormals = false, 
    b_centers = false,
    b_cnormals = false,
    b_cnormgrads = false,
    b_areas = false,
    b_area = false,
    b_volume = false;
  
  // http://wingware.com/psupport/python-manual/2.3/api/boolObjects.html
  PyObject
    *o_full = 0,
    *o_vertices = 0, 
    *o_vnormals = 0, 
    *o_vnormgrads = 0,
    *o_triangles = 0, 
    *o_tnormals = 0, 
    *o_centers = 0,
    *o_cnormals = 0,
    *o_cnormgrads = 0,
    *o_areas = 0,
    *o_area = 0,
    *o_volume = 0; 

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddd|iO!O!O!O!O!O!O!O!O!O!O!O!d", kwlist,
      &omega, &Omega0, &delta, // neccesary 
      &max_triangles,
      &PyBool_Type, &o_full,       
      &PyBool_Type, &o_vertices, 
      &PyBool_Type, &o_vnormals,
      &PyBool_Type, &o_vnormgrads,
      &PyBool_Type, &o_triangles, 
      &PyBool_Type, &o_tnormals,
      &PyBool_Type, &o_centers,
      &PyBool_Type, &o_cnormals,
      &PyBool_Type, &o_cnormgrads,
      &PyBool_Type, &o_areas,
      &PyBool_Type, &o_area,
      &PyBool_Type, &o_volume,
      &init_phi)
  ){
    std::cerr << "rotstar_marching_mesh:Problem reading arguments\n";
    return NULL;
  }
  
  if (o_full) b_full = PyObject_IsTrue(o_full); 
  if (o_vertices) b_vertices = PyObject_IsTrue(o_vertices);
  if (o_vnormals) b_vnormals = PyObject_IsTrue(o_vnormals);
  if (o_vnormgrads) b_vnormgrads = PyObject_IsTrue(o_vnormgrads);
  if (o_triangles) b_triangles = PyObject_IsTrue(o_triangles);
  if (o_tnormals)  b_tnormals = PyObject_IsTrue(o_tnormals);
  if (o_centers) b_centers = PyObject_IsTrue(o_centers);
  if (o_cnormals) b_cnormals = PyObject_IsTrue(o_cnormals);
  if (o_cnormgrads) b_cnormgrads = PyObject_IsTrue(o_cnormgrads);
  if (o_areas) b_areas = PyObject_IsTrue(o_areas);
  if (o_area) b_area = PyObject_IsTrue(o_area);
  if (o_volume) b_volume = PyObject_IsTrue(o_volume);
     
  //
  // Storing results in dictioonary
  // https://docs.python.org/2/c-api/dict.html
  //
  PyObject *results = PyDict_New();

  //
  // Getting initial meshing point
  //
  double r[3], g[3];
  rot_star::meshing_start_point(r, g, Omega0, omega);
 
  //
  //  Marching triangulation of the Roche lobe 
  //
    
  double params[3] = {omega, Omega0};
  
  Tmarching<double, Trot_star<double>> march(params);  
  
  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr; 
  std::vector<double> *GatV = 0;
   
  if (b_vnormgrads) GatV = new std::vector<double>;
 
  
  if ((b_full ? 
      !march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi):
      !march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
      )){
    std::cerr << "There is too much triangles\n";
    return NULL;
  }
  
 
  //
  // Calculate the mesh properties
  //
  int vertex_choice = 0;
  
  double 
    area, volume, 
    *p_area = 0, *p_volume = 0;
  
  std::vector<double> *A = 0; 
  
  std::vector<T3Dpoint<double>> *NatT = 0;
  
  if (b_areas) A = new std::vector<double>;
  
  if (b_area) p_area = &area;
  
  if (b_tnormals) NatT = new std::vector<T3Dpoint<double>>;
  
  if (b_volume) p_volume = &volume;
  
  // We do reordering triangles, so that NatT is pointing out
  mesh_attributes(V, NatV, Tr, A, NatT, p_area, p_volume, vertex_choice, true);

  //
  // Calculte the central points
  // 

  std::vector<double> *GatC = 0;
  
  std::vector<T3Dpoint<double>> *C = 0, *NatC = 0;
  
  if (b_centers) C = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormals) NatC = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormgrads) GatC = new std::vector<double>;
 
  march.central_points(V, Tr, C, NatC, GatC);
 
  //
  // Returning results
  //
  
 if (b_vertices)
    PyDict_SetItemStringStealRef(results, "vertices", PyArray_From3DPointVector(V));

  if (b_vnormals)
    PyDict_SetItemStringStealRef(results, "vnormals", PyArray_From3DPointVector(NatV));

  if (b_vnormgrads) {
    PyDict_SetItemStringStealRef(results, "vnormgrads", PyArray_FromVector(*GatV));
    delete GatV;
  }
  
  if (b_triangles)
    PyDict_SetItemStringStealRef(results, "triangles", PyArray_From3DPointVector(Tr));

  
  if (b_areas) {
    PyDict_SetItemStringStealRef(results, "areas", PyArray_FromVector(*A));
    delete A;  
  }
  
  if (b_area)
    PyDict_SetItemStringStealRef(results, "area", PyFloat_FromDouble(area));

  if (b_tnormals) {
    PyDict_SetItemStringStealRef(results, "tnormals", PyArray_From3DPointVector(*NatT));
    delete NatT;
  }

  if (b_volume)
    PyDict_SetItemStringStealRef(results, "volume", PyFloat_FromDouble(volume));
    
  
  if (b_centers) {
    PyDict_SetItemStringStealRef(results, "centers", PyArray_From3DPointVector(*C));
    delete C;  
  }

  if (b_cnormals) {
    PyDict_SetItemStringStealRef(results, "cnormals", PyArray_From3DPointVector(*NatC));
    delete NatC;
  }
  
  if (b_cnormgrads) {
    PyDict_SetItemStringStealRef(results, "cnormgrads", PyArray_FromVector(*GatC));
    delete GatC;
  }
  
  return results;
}

/*
  C++ wrapper for Python code:

    Calculation of visibility of triangles
    
  Python:

    dict = mesh_visibility(v, V, T, N, method, <keyword> = <value>)
    
  with arguments
  
    viewdir[3] - 1-rank numpy array floats = 3 coordinates representing 3D point
    V[][3] - 2-rank numpy array of vertices  
    T[][3] - 2-rank numpy array of indices of vertices composing triangles
    N[][3] - 2-rank numpy array of normals of triangles (if using boolean method)
             2-rank numpy array of normals of vertices (if using boolean method)
    
    method = ["boolean", "linear"]
    
    (optional)   
    tvisibilities: boolean, default True
    taweights: boolean, default False 
    horizon: boolean, default False
    
  Returns: dictionary with keywords
   
  keywords:
  
    tvisibilities: triangle visibility mask
      M[] - 1-rank numpy array of the ratio of the surface that is visible
  
    taweights: triangle averaging weights
      W[][3] - 2-rank numpy array of three weight one for each vertex of triangle
 
    horizon: a list of horizons defined by indices of vertices
      list of 1-rank numpy arrays of indices
    
    Note: They are not sorted in depth, as in principle they can not be!  
  
  Ref:
  * http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
  * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays
  * http://folk.uio.no/hpl/scripting/doc/python/NumPy/Numeric/numpy-13.html
*/

static PyObject *mesh_visibility(PyObject *self, PyObject *args, PyObject *keywds){

  //
  // Reading arguments
  //

  static char *kwlist[] = {
    (char*)"viewdir",
    (char*)"V",
    (char*)"T",
    (char*)"N",
    (char*)"method",
    (char*)"tvisibilities",
    (char*)"taweights",
    (char*)"horizon",
    NULL};
       
  PyArrayObject *ov = 0, *oV = 0, *oT = 0, *oN = 0;
  
  PyObject 
    *o_method, 
    *o_tvisibilities = 0, 
    *o_taweights = 0, 
    *o_horizon = 0;

  bool 
    b_tvisibilities = true,
    b_taweights = false,
    b_horizon = false;
  
  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "O!O!O!O!O!|O!O!O!", kwlist,
        &PyArray_Type, &ov,
        &PyArray_Type, &oV, 
        &PyArray_Type, &oT,
        &PyArray_Type, &oN,
        &PyString_Type, &o_method,
        &PyBool_Type, &o_tvisibilities,
        &PyBool_Type, &o_taweights,
        &PyBool_Type, &o_horizon    
        )
      ){
    std::cerr << "mesh_visibility:Problem reading arguments\n";
    return NULL;
  }
    
  if (o_tvisibilities) b_tvisibilities = PyObject_IsTrue(o_tvisibilities);
  if (o_taweights) b_taweights = PyObject_IsTrue(o_taweights);
  if (o_horizon) b_horizon = PyObject_IsTrue(o_horizon);
      
  if (!b_tvisibilities && !b_taweights && !b_horizon) return NULL;
  
  if (!PyArray_ISCONTIGUOUS(ov)|| 
      !PyArray_ISCONTIGUOUS(oV)|| 
      !PyArray_ISCONTIGUOUS(oT)|| 
      !PyArray_ISCONTIGUOUS(oN)) {
        
    std::cerr << "mesh_visibility::Input numpy arrays are not C-contiguous\n";
    return NULL;
  }
  
  double *view = (double*)PyArray_DATA(ov);
  
  std::vector<T3Dpoint<double> > V;
  PyArray_To3DPointVector(oV, V);
  
  std::vector<T3Dpoint<int>> T;
  PyArray_To3DPointVector(oT, T);

  std::vector<T3Dpoint<double> > N;
  PyArray_To3DPointVector(oN, N);
  
  std::vector<double> *M = 0;
  if (b_tvisibilities) M = new std::vector<double>;
  
  std::vector<T3Dpoint<double>> *W = 0;
  if (b_taweights) W = new std::vector<T3Dpoint<double>>;
  
  std::vector<std::vector<int>> *H = 0;
  if (b_horizon) H = new std::vector<std::vector<int>>;
  
  //
  //  Calculate visibility
  //
  {
    char *s = PyString_AsString(o_method);
        
    switch (fnv1a_32::hash(s)) {
      
      case "boolean"_hash32:
        // N - normal of traingles
        triangle_mesh_visibility_boolean(view, V, T, N, M, W, H);
        break;
        
      case "linear"_hash32:
        // N - normals at vertices
        triangle_mesh_visibility_linear(view, V, T, N, M, W, H);
        break;
    }
  }
  //
  // Storing results in dictionary
  // https://docs.python.org/2/c-api/dict.html
  //
  
  PyObject *results = PyDict_New();
  
  if (b_tvisibilities) {
    PyDict_SetItemStringStealRef(results, "tvisibilities", PyArray_FromVector(*M));
    delete M;
  }
  
  if (b_taweights) {
    PyDict_SetItemStringStealRef(results,"taweights", PyArray_From3DPointVector(*W));
    delete W; 
  }

  if (b_horizon) {
    PyObject *list = PyList_New(H->size());
    
    int i = 0;
    for (auto && h : *H) PyList_SetItem(list, i++, PyArray_FromVector(h));
    
    PyDict_SetItemStringStealRef(results, "horizon", list);
    delete H;  
  }

  return results;
}

/*
  C++ wrapper for Python code:

    Calculation of rough visibility of triangles
    
  Python:

    M = mesh_rough_visibility(v, V, T, N)
    
  with arguments
    v[3] - 1-rank numpy array of 3 coordinates representing 3D point
    V[][3] - 2-rank numpy array of vertices  
    T[][3] - 2-rank numpy array of indices of vertices composing triangles
    N[][3] - 2-rank numpy array of normals of triangles
  
  Returns: 
    M[] - 1-rank numpy array of the "ratio" of the surface that is visible
  
      0: for hidden
      1/2: for partially
      1:  for fully visible
    
  Ref:
  * http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
  * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays
  * http://folk.uio.no/hpl/scripting/doc/python/NumPy/Numeric/numpy-13.html
*/

static PyObject *mesh_rough_visibility(PyObject *self, PyObject *args){

  //
  // Storing/Reading arguments
  //
  
  PyArrayObject *ov, *oV, *oT, *oN;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!O!O!", 
        &PyArray_Type, &ov,
        &PyArray_Type, &oV, 
        &PyArray_Type, &oT,
        &PyArray_Type, &oN)){
    std::cerr << "mesh_rough_visibility:Problem reading arguments\n";
    return NULL;
  }
  
  double *view = (double*)PyArray_DATA(ov);
    
  std::vector<T3Dpoint<double> > V;
  PyArray_To3DPointVector(oV, V);
  
  std::vector<T3Dpoint<int>> T;
  PyArray_To3DPointVector(oT, T);

  std::vector<T3Dpoint<double> > N;
  PyArray_To3DPointVector(oN, N);
  
  std::vector<Tvisibility> Mt;

  //
  //  Calculate visibility
  //
  
  triangle_mesh_rough_visibility(view, V, T, N, Mt);
   
  //
  // Storing result
  //
  int Nt = PyArray_DIM(oT, 0);
     
  npy_intp dims[1] = {Nt};

  double *M = new double [Nt], *p = M;
 
  for (auto && m: Mt) 
    *(p++) = (m == hidden ? 0 : (m == partially_hidden ? 0.5 : 1.0));
  
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, M);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya;
}

/*
  C++ wrapper for Python code:

    Offseting the mesh along the normals of vertices to match the area 
    of the mesh with the reference area.
    
  Python:

    dict = mesh_offseting(A, V, NatV, T, <keyword> = <value>)
    
  with arguments
    area : float - reference area
    V    : 2-rank numpy array of vertices  
    NatV : 2-rank numpy array of normals at vertices
    T    : 2-rank numpy array of indices of vertices composing triangles
  
  (optional)
    max_iter:integer, default 1000, maximal number of iterations
    vertices: boolean, default True
    tnormals: boolean, default False
    areas: boolean, default False
    volume: boolean, default False
    area : boolean, default False
    centers: boolean, default False 
    cnormals: boolean, default False 
    cnormgrads: boolean, default False
    
    curvature: boolean, default False
      Enabling curvature dependent offseting.
  
  Returns: dictionary with keywords
  
    vertices
      Vnew: 2-rank numpy array of new vertices
    
    tnormals: default false
      NatT: 2-rank numpy array of normals of triangles
  
    areas:  default false
      A: 1-rank numpy array of areas of triangles
        
    volume:  default false
      volume:float, volume of body enclosed by triangular mesh
  
    area:  default false
      area: float - area of triangles of mesh
*/

static PyObject *mesh_offseting(PyObject *self, PyObject *args,  PyObject *keywds){
  
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"area",
    (char*)"V",
    (char*)"NatV",
    (char*)"T",
    (char*)"max_iter",
    (char*)"vertices",
    (char*)"tnormals",
    (char*)"areas",
    (char*)"volume",
    (char*)"area",
    (char*)"curvature",
    NULL
  };

  double area;
  
  int max_iter = 100;
    
  bool 
    b_vertices = true,
    b_tnormals = false,
    b_areas = false,
    b_volume = false,
    b_area = false,
    b_curvature = false;
    
  PyArrayObject *oV, *oNatV, *oT;
  
  PyObject  
    *o_vertices = 0,
    *o_tnormals = 0,
    *o_areas = 0,
    *o_volume = 0,
    *o_area = 0,
    *o_curvature = 0;
    
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dO!O!O!|iO!O!O!O!O!O!", kwlist,
      &area,
      &PyArray_Type, &oV, 
      &PyArray_Type, &oNatV, 
      &PyArray_Type, &oT,
      &max_iter,                      // optionals ...
      &PyBool_Type, &o_vertices,
      &PyBool_Type, &o_tnormals,
      &PyBool_Type, &o_areas,
      &PyBool_Type, &o_volume,
      &PyBool_Type, &o_area,
      &PyBool_Type, &o_curvature
      )){
    std::cerr << "*mesh_offseting:Problem reading arguments\n";
    return NULL;
  }
  
  if (o_vertices) b_vertices = PyObject_IsTrue(o_vertices);
  if (o_tnormals) b_tnormals = PyObject_IsTrue(o_tnormals);
  if (o_areas) b_areas = PyObject_IsTrue(o_areas);
  if (o_volume) b_volume = PyObject_IsTrue(o_volume);
  if (o_area) b_area = PyObject_IsTrue(o_area);
  if (o_curvature) b_curvature = PyObject_IsTrue(o_curvature);
  
  //
  // Storing input data
  //
  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;

  PyArray_To3DPointVector(oV, V);
  PyArray_To3DPointVector(oNatV, NatV);
  PyArray_To3DPointVector(oT, Tr);

  //
  // Running mesh offseting
  //
  if (b_curvature ? 
       !mesh_offseting_matching_area_curvature(area, V, NatV, Tr, max_iter):
       !mesh_offseting_matching_area(area, V, NatV, Tr, max_iter) ){
         
    std::cerr << "mesh_offseting_matching_area::Offseting failed\n";
    return NULL;
  }
 
  //
  // Calculate properties of the mesh with new vertices 
  //
  
  std::vector<T3Dpoint<double>> *NatT = 0;
  if (b_tnormals) NatT = new std::vector<T3Dpoint<double>>; 

  std::vector<double> *A = 0;
  if (b_areas) A = new std::vector<double>;
 
  double area_new, *p_area = 0, volume, *p_volume = 0;
  
  if (b_area) p_area = &area_new;
  if (b_volume) p_volume = &volume;
  
  // note: no need to reorder
  mesh_attributes(V, NatV, Tr, A, NatT, p_area, p_volume);
  
  //
  // Returning results
  //  
  
  PyObject *results = PyDict_New();
  
  if (b_vertices)
    PyDict_SetItemStringStealRef(results, "vertices", PyArray_From3DPointVector(V));
  
  if (b_tnormals) {
    PyDict_SetItemStringStealRef(results,"tnormals", PyArray_From3DPointVector(*NatT));
    delete NatT; 
  }
  
  if (b_areas) {
    PyDict_SetItemStringStealRef(results,"areas", PyArray_FromVector(*A));
    delete A; 
  }

  if (b_volume)
    PyDict_SetItemStringStealRef(results,"volume", PyFloat_FromDouble(volume));

  if (b_area)
    PyDict_SetItemStringStealRef(results, "area", PyFloat_FromDouble(area_new));


  return results;
}

/*
  C++ wrapper for Python code:

  Calculate properties of the tringular mesh.
  
  Python:

    dict = mesh_properties(V, T, <keyword>=[true,false], ... )
    
  where positional parameters
  
    V[][3]: 2-rank numpy array of vertices 
    T[][3]: 2-rank numpy array of 3 indices of vertices 
            composing triangles of the mesh aka connectivity matrix

  and optional keywords:  
   
    tnormals: boolean, default False
    areas: boolean, default False
    area: boolean, default False
    volume: boolean, default False
    
  Returns:
  
    dictionary
  
  with keywords
  
    tnormals:
      NatT[][3] - 2-rank numpy array of normals of triangles
  
    areas:
      A[]       - 1-rank numpy array of areas of triangles
    
    area:
      area      - area of triangles of mesh
    
    volume:
      volume    - volume of body enclosed by triangular mesh
*/

static PyObject *mesh_properties(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"T", 
    (char*)"tnormals", 
    (char*)"areas",
    (char*)"area",
    (char*)"volume",
    NULL};
  
  bool 
    b_tnormals = false, 
    b_areas = false,
    b_area = false,
    b_volume = false;
  
  PyArrayObject *oV, *oT;
    
  PyObject
    *o_tnormals = 0, 
    *o_areas = 0,
    *o_area = 0,
    *o_volume = 0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!|O!O!O!O!", kwlist,
      &PyArray_Type, &oV, // neccesary 
      &PyArray_Type, &oT,
      &PyBool_Type, &o_tnormals,  // optional
      &PyBool_Type, &o_areas,
      &PyBool_Type, &o_area,
      &PyBool_Type, &o_volume 
      )){
    std::cerr << "mesh_properties:Problem reading arguments\n";
    return NULL;
  }
  
  
  if (o_tnormals) b_tnormals = PyObject_IsTrue(o_tnormals);
  if (o_areas) b_areas = PyObject_IsTrue(o_areas);
  if (o_area) b_area = PyObject_IsTrue(o_area);
  if (o_volume) b_volume = PyObject_IsTrue(o_volume);

  if (!b_tnormals && !b_areas && !b_area && !b_volume) return NULL;
  
  //
  // Storing input data
  //
  std::vector<T3Dpoint<double>> V;
  std::vector<T3Dpoint<int>> Tr;

  PyArray_To3DPointVector(oV, V);
  PyArray_To3DPointVector(oT, Tr);

  //
  // Calculte the mesh properties
  //
  
  double 
    area, volume, 
    *p_area = 0, *p_volume = 0;
  
  std::vector<double> *A = 0; 
  
  std::vector<T3Dpoint<double>> *NatT = 0;
  
  if (b_areas) A = new std::vector<double>;
  
  if (b_area) p_area = &area;
  
  if (b_tnormals) NatT = new std::vector<T3Dpoint<double>>;
  
  if (b_volume) p_volume = &volume;
  
  
  mesh_attributes(V, Tr, A, NatT, p_area, p_volume);

  //
  // Returning results
  //  
  
  PyObject *results = PyDict_New();
  

  if (b_areas) {
    PyDict_SetItemStringStealRef(results, "areas", PyArray_FromVector(*A));
    delete A;  
  }
  
  if (b_area)
    PyDict_SetItemStringStealRef(results, "area", PyFloat_FromDouble(area));

  if (b_tnormals) {
    PyDict_SetItemStringStealRef(results, "tnormals", PyArray_From3DPointVector(*NatT));
    delete NatT;
  }

  if (b_volume)
    PyDict_SetItemStringStealRef(results, "volume", PyFloat_FromDouble(volume));

  
  return results;
}

/*
  C++ wrapper for Python code:

  Export the mesh into povray file.
   
  Python:

    dict = mesh_export_povray("scene.pov", V, NatV T,  [..],[..],[..])
  
  Commandline: for povray
    povray +R2 +A0.1 +J1.2 +Am2 +Q9 +H480 +W640 scene.pov
  
  where positional parameters
    filename: string

    V[][3]: 2-rank numpy array of vertices 
    NatV[][3]: 2-rank numpy array of normals at vertices 
    T[][3]: 2-rank numpy array of 3 indices of vertices 
            composing triangles of the mesh aka connectivity matrix


    camera_location: 1-rank numpy array of floats -- location of the camera
    camera_look_at: 1-rank numpy array of floats -- point to which camera is pointing
    light_source: 1-rank numpy array of floats -- location of point light source
     
  optional:
   
    body_color: string, default Red
      color of the body, e.g. White, Red, Yellow, .., 
    
    plane_enable: boolean, default false
      enabling infinite horizontal plane
    
    plane_height: float, default zmin{mesh} - 25%(zmax{mesh} - zmin{mesh}) 
      height at which is the infinite horizontal plane
      
  Returns:
    
    None

*/

static PyObject *mesh_export_povray(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"filename",
    (char*)"V",
    (char*)"NatV",
    (char*)"T", 
    (char*)"camera_location",
    (char*)"camera_look_at",
    (char*)"light_source",
    (char*)"body_color",
    (char*)"plane_enable",
    (char*)"plane_height",
    NULL};
    
  PyArrayObject 
    *oV, *oNatV, *oT, *o_camera_location, 
    *o_camera_look_at, *o_light_source;
    
  PyObject 
    *o_filename, *o_body_color = 0, 
    *o_plane_enable = 0, *o_plane_height = 0;
   
  bool plane_enable = false;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!|O!O!O!", kwlist,
      &PyString_Type, &o_filename, // neccesary
      &PyArray_Type, &oV,  
      &PyArray_Type, &oNatV,
      &PyArray_Type, &oT, 
      &PyArray_Type, &o_camera_location,
      &PyArray_Type, &o_camera_look_at, 
      &PyArray_Type, &o_light_source,
      &PyString_Type, &o_body_color,   // optional
      &PyBool_Type, &o_plane_enable,
      &PyFloat_Type, &o_plane_height)){
    std::cerr << "mesh_export_povray:Problem reading arguments\n";
    return NULL;
  }
    
  //
  // Storing input data
  //
  
  char *filename = PyString_AsString(o_filename);
    
  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;

  PyArray_To3DPointVector(oV, V);
  PyArray_To3DPointVector(oNatV, NatV);
  PyArray_To3DPointVector(oT, Tr);

  T3Dpoint<double> 
    camera_location((double *)PyArray_DATA(o_camera_location)),
    camera_look_at((double *)PyArray_DATA(o_camera_look_at)),
    light_source((double *)PyArray_DATA(o_light_source));

  std::string body_color((o_body_color ? PyString_AsString(o_body_color):"Red"));
  
  double plane_height, *p_plane_height = 0;
    
  if (o_plane_enable) plane_enable = PyObject_IsTrue(o_plane_enable);

  if (plane_enable) {
    
    p_plane_height = &plane_height;
    
    if (o_plane_height)
      plane_height = PyFloat_AsDouble(o_plane_height);
    else {
      double t, zmin, zmax;
      
      zmax= -(zmin = std::numeric_limits<double>::max());
      
      for (auto && v: V) {
        t = v[2];
        if (t > zmax) zmax = t;
        if (t < zmin) zmin = t;
      }
    
      plane_height = zmin - 0.25*(zmax - zmin);
    }
  }

  std::ofstream file(filename);
    
  triangle_mesh_export_povray(
    file, 
    V, NatV, Tr,
    body_color,
    camera_location, 
    camera_look_at, 
    light_source,
    p_plane_height);


  Py_INCREF(Py_None);
  return Py_None;
}

/*
  Create a LD model from a tuple.
  
  Input:
    p - Tuple of the form ("name", 1-rank numpy array of floats)

    
  Return:
    pointer to the TLDmodel<double>, in case of error return NULL;
*/ 
bool LDmodelFromTuple(
  PyObject *p, 
  TLDmodel<double> * & pmodel) {

  if (!PyTuple_CheckExact(p)) {
    std::cerr 
      << "LDmodelFromTuple::LD model description is not a tuple.\n"; 
    return false;
  }
      
  if (PyTuple_Size(p) == 0) {     
    std::cerr << "LDmodelFromTuple::LD model tuple is empty.\n";
    return false;
  }
  
  PyObject *s = PyTuple_GetItem(p, 0);
      
  if (!PyString_Check(s)) {
    std::cerr << "LDmodelFromTuple::LD model name is not string.\n";
    return false;
  }
    
  double *par = 0;
  
  pmodel = 0;
  
  switch (fnv1a_32::hash(PyString_AsString(s))){

    case "uniform"_hash32: 
      pmodel = new TLDuniform<double>();
      return true;
      
    case "linear"_hash32: 
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDlinear<double>(par);
      return true;
    
    case "quadratic"_hash32:
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDquadratic<double>(par);
      return true;
    
    case "nonlinear"_hash32:
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDnonlinear<double>(par);
      return true;
      
    case "logarithmic"_hash32:
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDlogarithmic<double>(par);
      return true;
    
    case "square_root"_hash32:
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDsquare_root<double>(par);
      return true;
      
    case "claret"_hash32:
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDclaret<double>(par);
      return true;
   
    case "interp"_hash32:
      return true;
  }

  std::cerr << "LDmodelFromTuple::Don't know to handle this LD model.\n";
  return false;
}


/*
  Create a LD model from a tuple.
  
  Input:
    p - list of tuples of the form ("name", 1-rank numpy array of floats)
  
  Output:
    LDmod - vector of LDmodels
    
  Return:
    true if no error, false otherwise
*/

bool LDmodelFromListOfTuples(
  PyObject *p, 
  std::vector<TLDmodel<double>*> & LDmod) {

  int len = PyList_Size(p);
  
  TLDmodel<double> *ld_mod;
      
  for (int i = 0; i < len; ++i) {
    
    if (LDmodelFromTuple(PyList_GetItem(p, i), ld_mod)) {
      LDmod.push_back(ld_mod);
    } else {
      for (auto && ld: LDmod) if (ld) delete ld;
      return false;
    }
  }
  return true;
}

/*
  C++ wrapper for Python code:

  Calculate radiosity of triangles due to reflection according to a 
  chosen reflection model using triangles as support of the surface.
  
  Python:

    F = mesh_radiosity_problem_triangles(V, Tr, NatT, A, R, F0, LDmod, LDidx, model, <keyword>=<value>, ... )
    
  where positional parameters:
  
    V[][3]: 2-rank numpy array of vertices 
    Tr[][3]: 2-rank numpy array of 3 indices of vertices 
            composing triangles of the mesh aka connectivity matrix
    NatT[][3]: 2-rank numpy array of normals of face triangles
    A[]: 1-rank numpy array of areas of triangles
    R[]: 1-rank numpy array of albedo/reflection of triangles
    F0[]: 1-rank numpy array of intrisic radiant exitance/flux of triangles

    LDmod: list of tuples of the format 
            ("name", sequence of parameters)
            supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "claret"      4 parameters
              "interp"      interpolation data TODO !!!!
              
    LDidx[]: 1-rank numpy array of indices of LD models used on each of triangles
    
    model : string - name of the reflection model in use 
             method in {"Wilson", "Horvat"}
             
  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg) 
          threshold for permitted cos(view-angle)
    epsM: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.
 
  Returns:
    F[]: 1-rank numpy array of radiosities (intrinsic and reflection) of triangles
  
  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect, 
    Astrophysical Journal,  356, 613-622, 1990 June
*/

static PyObject *mesh_radiosity_problem_triangles(
  PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "mesh_radiosity_problem_triangles";
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr", 
    (char*)"NatT", 
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"LDidx",
    (char*)"model",
    (char*)"epsC",
    (char*)"epsM",
    (char*)"max_iter",
    NULL
  };
  
  int max_iter = 100;         // default value
  
  double 
    epsC = 0.00872654,        // default value
    epsM = 1e-12;             // default value
  
  PyArrayObject *oV, *oT, *oNatT, *oA, *oR, *oF0, *oLDidx;
     
  PyObject *oLDmod, *omodel;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!O!O!|ddi", kwlist,
      &PyArray_Type, &oV,         // neccesary 
      &PyArray_Type, &oT,
      &PyArray_Type, &oNatT,
      &PyArray_Type, &oA,
      &PyArray_Type, &oR,
      &PyArray_Type, &oF0,
      &PyList_Type, &oLDmod,
      &PyArray_Type, &oLDidx,
      &PyString_Type, &omodel,
      &epsC,                      // optional
      &epsM,
      &max_iter)){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
  
    
  //
  // Storing input data
  //  

  std::vector<TLDmodel<double>*> LDmod;
  
  if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
    std::cerr << fname << "::Not able to read LD models\n"; 
    return NULL;
  }
  
  //
  // Check is there is interpolation is used
  //
  
  bool st_interp = false;
  
  for (auto && pld : LDmod) if (pld == 0) {
    st_interp = true;
    break;
  }
  
  std::vector<int> LDidx;
  PyArray_ToVector(oLDidx, LDidx);
 
  std::vector<T3Dpoint<double>> V, NatT;
  std::vector<T3Dpoint<int>> Tr;

  std::vector<double> A;
  PyArray_ToVector(oA, A);

  PyArray_To3DPointVector(oV, V);
  PyArray_To3DPointVector(oT, Tr);
  PyArray_To3DPointVector(oNatT, NatT);
 
  //
  // Determine the LD view-factor matrix
  //

  std::vector<Tmat_elem<double>> Fmat;
    
  triangle_mesh_radiosity_matrix_triangles(V, Tr, NatT, A, LDmod, LDidx, Fmat);
  
  for (auto && ld: LDmod) delete ld;
  LDmod.clear();
  
  LDidx.clear();
  V.clear();
  Tr.clear();
  NatT.clear();
  A.clear();
  
  //
  // Solving the radiosity equation
  //
  
  std::vector<double> F0, F, R;
  
  PyArray_ToVector(oR, R);
  PyArray_ToVector(oF0, F0);
  
  {
    bool success = false;
    
    char *s = PyString_AsString(omodel);
      
    switch (fnv1a_32::hash(s)) {
      
      case "Wilson"_hash32:
        
        if (st_interp) {
          std::cerr  
            << fname 
            << "::Interpolation isn't supported with Wilson's reflection model\n";
          return NULL; 
        }
        
        success = solve_radiosity_equation_Wilson(Fmat, R, F0, F);
      break;
      
      case "Horvat"_hash32:
        if (st_interp) {
          #if 0
          int N = Tr.size();
        
          // calculate F0 
          for (int i = 0; i < N; ++i) 
            if (LDmod[LDidx[i]] == 0) F0[i] = Interp("F", LDidx[i], params[i]);
        
          // calculate and S0
          std::vector<double> S0(N, 0);
          
          for (auto && f : Fmat)
            if (LDmod[LDidx[f.j]] == 0)
              S0[f.i] += utils::m_pi*f.F0*Interp("I", LDidx[i], params[i], f.F);
            else 
              S0[f.i] += f.F*F0[j];
            
          success = solve_radiosity_equation_Horvat(Fmat, R, F0, S0, F);
          #endif
          
          std::cerr << fname  << "::Not yet implemented\n";
          return NULL;
        } else
          success = solve_radiosity_equation_Horvat(Fmat, R, F0, F);
      break;
      
      default:
        std::cerr 
          << fname << "::This radiosity model ="
          << s << " does not exist\n";
        return NULL;
    }
    
    if (!success)
      std::cerr << fname << "::slow convergence\n";
  }
  
   
  return PyArray_FromVector(F);
}


/*
  C++ wrapper for Python code:

  Calculate radiosity of triangles on n convex bodies due to reflection 
  according to a chosen reflection model using triangles as support of 
  the surface.
  
  Python:

    F = mesh_radiosity_problem_triangles_nbody_convex(
        V, Tr, NatT, A, R, F0, LDmod, model, <keyword> = <value>, ... )
    
  where positional parameters:
  
    V = {V1, V2, ...} : list of 2-rank numpy array of vertices V[][3],
                   length of the list is n, as number of bodies
    
    Tr = {Tr1, Tr2, ...} : list of 2-rank numpy array of 3 indices of vertices Tr[][3]
                    composing triangles of the mesh aka connectivity matrix
                    length of the list is n, as number of bodies
          
    NatT = {NatT1, NatT2, ...} : list of 2-rank numpy array of normals of face triangles NatT[][3]
    A = {A1, A2, ...} : list of 1-rank numpy array of areas of triangles A[]
    R = {R1, R2, ...} : list of 1-rank numpy array of albedo/reflection of triangles R[]
    F0 = {F0_0, F0_1, ...} : list of 1-rank numpy array of intrisic radiant exitance of triangles F0[]

    LDmod = {LDmod1, LDmod2,..}: list of tuples of the format
    
            ("name", sequence of parameters)
            with one model per body. Supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
               "claret"      4 parameters
              "interp"      interpolation data  TODO !!!!
              
               
     model : string - name of the reflection model in use 
             method in {"Wilson", "Horvat"}        
  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg) 
          threshold for permitted cos(view-angle)
    epsM: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.
 
  Returns:
    F = {F_0, F_1, ...} : list of 1-rank numpy array of total radiosities 
                      (intrinsic and reflection) of triangles
  
  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect, 
    Astrophysical Journal,  356, 613-622, 1990 June
*/

static PyObject *mesh_radiosity_problem_triangles_nbody_convex(
  PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "mesh_radiosity_problem_triangles_nbody_convex";
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr", 
    (char*)"NatT", 
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"model",
    (char*)"epsC",
    (char*)"epsM",
    (char*)"max_iter",
    NULL
  };
  
  int max_iter = 100;         // default value
  
  double 
    epsC = 0.00872654,        // default value
    epsM = 1e-12;             // default value
  
  PyObject *oLDmod, *omodel, *oV, *oTr, *oNatT, *oA, *oR, *oF0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!|ddi", kwlist,
      &PyList_Type, &oV,         // neccesary 
      &PyList_Type, &oTr,
      &PyList_Type, &oNatT,
      &PyList_Type, &oA,
      &PyList_Type, &oR,
      &PyList_Type, &oF0,
      &PyList_Type, &oLDmod,
      &PyString_Type, &omodel,
      &epsC,                     // optional
      &epsM,
      &max_iter)){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
  
  //
  // Storing input data
  //  

  std::vector<TLDmodel<double>*> LDmod;
  
  if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
    std::cerr << fname << "::Not able to read LD models\n"; 
    return NULL;
  }
 
  //
  // Check is there is interpolation is used
  //
  
  bool st_interp = false;
  
  for (auto && pld : LDmod) if (pld == 0) {
    st_interp = true;
    break;
  }
 
  // getting data from list of PyArrays
  int n = LDmod.size();   // number of bodies
  
  if (n <= 1){
    std::cerr << fname << "::There seem to just n=" << n << " bodies.\n";
    return NULL;
  }
   
  std::vector<std::vector<T3Dpoint<double>>> V(n), NatT(n);
  std::vector<std::vector<T3Dpoint<int>>> Tr(n);
  std::vector<std::vector<double>> A(n), R(n), F0(n), F;
 
  for (int i = 0; i < n; ++i){

    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oV, i), V[i]);
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oNatT, i), NatT[i]);
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oTr, i), Tr[i]);
 
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oA, i), A[i]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oR, i), R[i]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oF0, i), F0[i]);
  }
 
  //
  // Determine the LD view-factor matrix
  //

  std::vector<Tmat_elem_nbody<double>> Fmat;
  
  triangle_mesh_radiosity_matrix_triangles_nbody_convex(V, Tr, NatT, A, LDmod, Fmat);

  for (auto && ld: LDmod) delete ld;
  LDmod.clear();
    

  //
  // Solving the radiosity equation depending on the model
  //
  {
    bool success = false;
    
    char *s = PyString_AsString(omodel);
      
    switch (fnv1a_32::hash(s)) {
      
      case "Wilson"_hash32:
        if (st_interp) {
          std::cerr  << fname 
            << "::Interpolation isn't supported with Wilson's reflection model\n";
          return NULL; 
        }
        success = solve_radiosity_equation_Wilson_nbody(Fmat, R, F0, F);
      break;
      
      case "Horvat"_hash32:
       if (st_interp) {
          #if 0
          std::vector<std::vector<double>> S0;
          
          // calculating F0, S0
          
          success = solve_radiosity_equation_Horvat_nbody(Fmat, R, F0, S0, F);
          #endif
          std::cerr << fname << "::This is not yet implemented\n";
          return NULL;
        } else {
          success = solve_radiosity_equation_Horvat_nbody(Fmat, R, F0, F);
        }
      break;
      
      default:
        std::cerr 
        << fname << "::This radiosity model ="
        << s << " does not exist\n";
        return NULL;
    }
    
    if (!success)
      std::cerr << fname << "::slow convergence\n";
  }
  
  
  PyObject *results = PyList_New(n);
  
  for (int i = 0; i < n; ++i)
    PyList_SetItem(results, i, PyArray_FromVector(F[i]));

  // TODO: check the reference count ????
  return results;
}


/*
  C++ wrapper for Python code:

  Calculate radiosity of triangles on n convex bodies due to reflection 
  according to a chosen reflection model using vertices as support of 
  the surface.
  
  Python:

    F = mesh_radiosity_problem_vertices_nbody_convex(
        V, Tr, NatT, A, R, F0, LDmod, <keyword> = <value>, ... )
    
  where positional parameters:
  
    V = {V1, V2, ...} : list of 2-rank numpy array of vertices V[][3],
                   length of the list is n, as number of bodies
    
    Tr = {Tr1, Tr2, ...} : list of 2-rank numpy array of 3 indices of vertices Tr[][3]
                    composing triangles of the mesh aka connectivity matrix
                    length of the list is n, as number of bodies
          
    NatV = {NatV1, NatV2, ...} : list of 2-rank numpy array of normals at vertices NatV[][3]
     
    A = {A1, A2, ...} : list of 1-rank numpy array of areas at vertices A[]
    R = {R1, R2, ...} : list of 1-rank numpy array of albedo/reflection at vertices R[]
    F0 = {F0_0, F0_1, ...} : list of 1-rank numpy array of intrisic radiant exitance at vertices F0[]
    
    model : string - name of the reflection model in use 
             method in {"Wilson", "Horvat"}
    
    LDmod = {LDmod1, LDmod2,..}: list of tuples of the format
    
            ("name", sequence of parameters)
            with one model per body. Supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "claret"      4 parameters
              "interp"      interpolation data  TODO !!!!
  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg) 
          threshold for permitted cos(view-angle)
    epsF: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.
 
  Returns:
    F = {F_0, F_1, ...} : list of 1-rank numpy array of total radiosities 
                      (intrinsic and reflection) at vertices
  
  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect, 
    Astrophysical Journal,  356, 613-622, 1990 June
*/

static PyObject *mesh_radiosity_problem_vertices_nbody_convex(
  PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "mesh_radiosity_problem_vertices_nbody_convex";
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr", 
    (char*)"NatV", 
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"model",
    (char*)"epsC",
    (char*)"epsF",
    (char*)"max_iter",
    NULL
  };
  
  int max_iter = 100;         // default value
  
  double 
    epsC = 0.00872654,        // default value
    epsF = 1e-12;             // default value
  
  PyObject *oLDmod, *omodel, *oV, *oTr, *oNatV, *oA, *oR, *oF0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!O!|ddi", kwlist,
      &PyList_Type, &oV,         // neccesary 
      &PyList_Type, &oTr,
      &PyList_Type, &oNatV,
      &PyList_Type, &oA,
      &PyList_Type, &oR,
      &PyList_Type, &oF0,
      &PyList_Type, &oLDmod,
      &PyString_Type, &omodel,
      &epsC,                     // optional
      &epsF,
      &max_iter)){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
  
  
  //
  // Storing input data
  //  

  std::vector<TLDmodel<double>*> LDmod;
  
  if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
    std::cerr << fname << "::Not able to read LD models\n"; 
    return NULL;
  }
 
  //
  // Check is there is interpolation is used
  //
  
  bool st_interp = false;
  
  for (auto && pld : LDmod) if (pld == 0) {
    st_interp = true;
    break;
  }
 
  // getting data from list of PyArrays
  int n = LDmod.size(); // number of bodies
  
  if (n <= 1){
    std::cerr << fname << "::There seem to just n=" << n << " bodies.\n";
    return NULL;
  }
  
  std::vector<std::vector<T3Dpoint<double>>> V(n), NatV(n);
  std::vector<std::vector<T3Dpoint<int>>> Tr(n);
  std::vector<std::vector<double>> A(n), R(n), F0(n), F;
 
  for (int i = 0; i < n; ++i){

    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oV, i), V[i]);
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oNatV, i), NatV[i]);
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oTr, i), Tr[i]);
 
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oA, i), A[i]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oR, i), R[i]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oF0, i), F0[i]);
  }
 
  //
  // Determine the LD view-factor matrix
  //

  std::vector<Tmat_elem_nbody<double>> Fmat;
    
  triangle_mesh_radiosity_matrix_vertices_nbody_convex(
    V, Tr, NatV, A, LDmod, Fmat);
 
  std::cerr << "Fmat.size=" << Fmat.size() << " V.size=" << V.size() << '\n';
  for (int i = 0; i < 2; ++i) std::cerr << "V[i].size=" << V[i].size() << '\n';
 
  for (auto && ld: LDmod) delete ld;
  LDmod.clear();
  
  //
  // Solving the radiosity equation depending on the model
  //
  {
    bool success = false;
    
    char *s = PyString_AsString(omodel);
        
    switch (fnv1a_32::hash(s)) {
      
      case "Wilson"_hash32:
        
        if (st_interp) {
          std::cerr  << fname 
            << "::Interpolation isn't supported with Wilson's reflection model\n";
          return NULL; 
        }
        
        success = solve_radiosity_equation_Wilson_nbody(Fmat, R, F0, F);
      break;
      
      case "Horvat"_hash32:
        if (st_interp) {
          #if 0
          std::vector<std::vector<double>> S0;
          
          // ???? calculate F0, S0
          
          success = solve_radiosity_equation_Horvat_nbody(Fmat, R, F0, S0, F);
          #endif
          std::cerr << fname << "::This is not yet implemented\n";
          return NULL;
        } else {
          success = solve_radiosity_equation_Horvat_nbody(Fmat, R, F0, F);
        }
      break;
      
      default:
        std::cerr 
        << fname << "::This radiosity model ="
        << s << " does not exist\n";
        return NULL;
    }
    
    if (!success)
      std::cerr << fname << "::slow convergence\n";
  }
  
  PyObject *results = PyList_New(n);
  
  for (int i = 0; i < n; ++i)
    PyList_SetItem(results, i, PyArray_FromVector(F[i]));

  // TODO: check the reference count ????
  return results;
}

/*
  C++ wrapper for Python code:

  Calculate radiosity of triangles due to reflection according to a 
  choosen reflection model using VERTICES as support of the surface. 
  We image a disk in the tangent space associated to the vertices.
  
  Python:

    F = mesh_radiosity_problem_vertices(V, Tr, NatT, A, R, F0, LDmod, LDidx, model, <keyword>=<value>, ... )
    
  where positional parameters:
  
    V[][3]: 2-rank numpy array of vertices 
    Tr[][3]: 2-rank numpy array of 3 indices of vertices 
            composing triangles of the mesh aka connectivity matrix
    NatV[][3]: 2-rank numpy array of normals at vertices
    A[]: 1-rank numpy array of areas of triangles
    R[]: 1-rank numpy array of albedo/reflection at vertices
    F0[]: 1-rank numpy array of intrisic radiant exitance at vertices

    LDmod: list of tuples of the format 
            ("name", sequence of parameters)
            supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "claret"      4 parameters
              "interp"      interpolation data  TODO !!!!
              
    LDidx[]: 1-rank numpy array of indices of LD models used on each vertex

    model : string - name of the reflection model in use 
             method in {"Wilson", "Horvat"}
    
  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg) 
          threshold for permitted cos(view-angle)
    epsM: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.
 
  Returns:
    F[]: 1-rank numpy array of radiosities (intrinsic and reflection) at 
          vertices
  
  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect, 
    Astrophysical Journal,  356, 613-622, 1990 June
*/

static PyObject *mesh_radiosity_problem_vertices(
  PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "mesh_radiosity_problem_vertices";
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr", 
    (char*)"NatV", 
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"LDidx",
    (char*)"model",
    (char*)"epsC",
    (char*)"epsM",
    (char*)"max_iter",
    NULL
  };
  
  int max_iter = 100;         // default value
  
  double 
    epsC = 0.00872654,        // default value
    epsM = 1e-12;             // default value
  
  PyArrayObject *oV, *oT, *oNatV, *oA, *oR, *oF0, *oLDidx;
     
  PyObject *oLDmod, *omodel;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!O!O!|ddi", kwlist,
      &PyArray_Type, &oV,         // neccesary 
      &PyArray_Type, &oT,
      &PyArray_Type, &oNatV,
      &PyArray_Type, &oA,
      &PyArray_Type, &oR,
      &PyArray_Type, &oF0,
      &PyList_Type, &oLDmod,
      &PyArray_Type, &oLDidx,
      &PyString_Type, &omodel,
      &epsC,                      // optional
      &epsM,
      &max_iter)){
        
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
  
  //
  // Storing input data
  //  

  std::vector<TLDmodel<double>*> LDmod;
    
  if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
    std::cerr << fname << "::Not able to read LD models\n"; 
    return NULL;
  }
  
  //
  // Check is there is interpolation is used
  //
  
  bool st_interp = false;
  
  for (auto && pld : LDmod) if (pld == 0) {
    st_interp = true;
    break;
  }
  
  std::vector<int> LDidx;
  PyArray_ToVector(oLDidx, LDidx);
 
  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;

  std::vector<double> A;
  PyArray_ToVector(oA, A);

  PyArray_To3DPointVector(oV, V);
  PyArray_To3DPointVector(oT, Tr);
  PyArray_To3DPointVector(oNatV, NatV);
 
    
  //
  // Determine the LD view-factor matrix
  //

  std::vector<Tmat_elem<double>> Fmat;
    
  triangle_mesh_radiosity_matrix_vertices(V, Tr, NatV, A, LDmod, LDidx,  Fmat);
  
  for (auto && ld: LDmod) delete ld;
  LDmod.clear();
  
  // some clean up to reduce memory footprint
  LDidx.clear(); V.clear(); Tr.clear();
  NatV.clear();  A.clear();
  
  //
  // Solving the radiosity equation depending on the model
  //
  
  std::vector<double> F0, F, R;
  
  PyArray_ToVector(oR, R);
  PyArray_ToVector(oF0, F0);
  
  {
    bool success = false;
    
    char *s = PyString_AsString(omodel);
      
    switch (fnv1a_32::hash(s)) {
      
      case "Wilson"_hash32:
      
        if (st_interp) {
          std::cerr  << fname 
            << "::Interpolation isn't supported with Wilson's reflection model\n";
          return NULL; 
        }
      
        success = solve_radiosity_equation_Wilson(Fmat, R, F0, F);
      
      break;
      
      case "Horvat"_hash32:
      
        if (st_interp) {
          #if 0
          std::vector<double> S0;
          
          // calculate F0, S0 ?????
          
          success = solve_radiosity_equation_Horvat(Fmat, R, F0, S0, F);
          #endif
          
          std::cerr << fname << "::This is not yet implemented\n";
          return NULL;
        } else {
          success = solve_radiosity_equation_Horvat(Fmat, R, F0, F);
        }
      
      break;
      
      default:
        std::cerr 
        << fname << "::This radiosity model ="
        << s << " does not exist\n";
        return NULL;
    }
    
    if (!success)
      std::cerr << fname << "::slow convergence\n";
  }
  

  return PyArray_FromVector(F);
}

/*
  C++ wrapper for Python code:

  Calculate the central points corresponding to the mesh and Roche lobe.
  
  Python:

    dict = roche_centers(q, F, d, Omega0, V, T, <keyword>=[true,false])
    
  where parameters
  
    positional:
  
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      Omega0: float - value of the generalized Kopal potential
      V: 2-rank numpy array of vertices  
      T: 2-rank numpy array of indices of vertices composing triangles
       
    keywords: 
     
      centers: boolean, default False
      cnormals: boolean, default False
      cnormgrads: boolean, default False
      
  Returns:
  
    dictionary
  
  with keys
  
  centers:
    C: 2-rank numpy array of new centers

  cnormals: default false
    NatC: 2-rank numpy array of normals of centers

  cnormgrads: default false
    GatC: 1-rank numpy array of norms of gradients at centers
*/
static PyObject *roche_central_points(PyObject *self, PyObject *args,  PyObject *keywds){
  
  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"vertices",
    (char*)"triangles",
    (char*)"centers", 
    (char*)"cnormals",
    (char*)"cnormgrads",
    NULL};
  
  double q, F, d, Omega0;   

  bool 
    b_centers = false,
    b_cnormals = false,
    b_cnormgrads = false;
    
  PyArrayObject *oV,  *oT;
 
  PyObject
    *o_centers = 0,
    *o_cnormals = 0,
    *o_cnormgrads = 0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddddO!O!|O!O!O!", kwlist,
      &q, &F, &d, &Omega0,         // neccesary 
      &PyArray_Type, &oV, 
      &PyArray_Type, &oT,
      &PyBool_Type, &o_centers,    // optional
      &PyBool_Type, &o_cnormals,
      &PyBool_Type, &o_cnormgrads
      )){
    std::cerr << "roche_central_points::Problem reading arguments\n";
    return NULL;
  }
  
  if (o_centers) b_centers = PyObject_IsTrue(o_centers);
  if (o_cnormals) b_cnormals = PyObject_IsTrue(o_cnormals);
  if (o_cnormgrads) b_cnormgrads = PyObject_IsTrue(o_cnormgrads);


  if (!b_centers && !b_cnormals && !b_cnormgrads) return NULL;
  
  //
  // Storing data
  //
  
  std::vector<T3Dpoint<double>> V;
  PyArray_To3DPointVector(oV, V);
  
  std::vector<T3Dpoint<int>> Tr;
  PyArray_To3DPointVector(oT, Tr);
  
  //
  //  Init marching triangulation for the Roche lobe 
  //
    
  double params[4] = {q, F, d, Omega0};
  
  Tmarching<double, Tgen_roche<double>> march(params);  
  
  //
  // Calculte the central points
  // 

  std::vector<double> *GatC = 0;
  
  std::vector<T3Dpoint<double>> *C = 0, *NatC = 0;
  
  if (b_centers) C = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormals) NatC = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormgrads) GatC = new std::vector<double>;
 
  march.central_points(V, Tr, C, NatC, GatC);
  
  //
  // Returning results
  //
  
  PyObject *results = PyDict_New();
    
  if (b_centers) {
    PyDict_SetItemStringStealRef(results, "centers", PyArray_From3DPointVector(*C));
    delete C;  
  }

  if (b_cnormals) {
    PyDict_SetItemStringStealRef(results, "cnormals", PyArray_From3DPointVector(*NatC));
    delete NatC;
  }
  
  if (b_cnormgrads) {
    PyDict_SetItemStringStealRef(results, "cnormgrads", PyArray_FromVector(*GatC));
    delete GatC;
  }
  
  return results; 
}


/*
  C++ wrapper for Python code:

    Reprojecting the points into the Roche lobes.
    
  Python:

    dict = roche_reprojecting_vertices(V, q, F, d, Omega0, <keywords>=<value>)
    
  with arguments
  
  positionals: necessary
    V[][3] - 2-rank numpy array of vertices  
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega0: float - value of the generalized Kopal potential
  
  keywords:
    vertices: boolean, default False
    vnormals: boolean, default False
    vnormgrads:boolean, default False
    max_iter: integer, default 100
    
  Return: dictionary with keys
    
    vertices: 
      V1[][3]    - 2-rank numpy array of vertices 
    
    vnormals:
      NatV1[][3] - 2-rank numpy array of normals at vertices
 
    vnormgrads:
      GatV1[]    - 1-rank numpy array of norms of the gradients at central points
  
  
  Ref:
  * http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
  * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays
  * http://folk.uio.no/hpl/scripting/doc/python/NumPy/Numeric/numpy-13.html
*/

static PyObject *roche_reprojecting_vertices(PyObject *self, PyObject *args, PyObject *keywds) {

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"V",
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"vertices",
    (char*)"vnormals",
    (char*)"vnormgrads",
    (char*)"max_iter",
    NULL
  };
  
  PyArrayObject *oV;
  
  double q, F, d, Omega0;   
  
  bool 
    b_vertices = false, 
    b_vnormals = false, 
    b_vnormgrads = false;
  
  int max_iter = 100;
  
  PyObject
    *o_vertices = 0, 
    *o_vnormals = 0, 
    *o_vnormgrads = 0;
    
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!dddd|O!O!O!i", kwlist,
      &PyArray_Type, &oV, 
      &q, &F, &d, &Omega0,
      &PyBool_Type, &o_vertices, 
      &PyBool_Type, &o_vnormals,
      &PyBool_Type, &o_vnormgrads,
      &max_iter)){
    std::cerr << "roche_reprojecting_vertices::Problem reading arguments\n";
    return NULL;
  }
  
  if (o_vertices) b_vertices = PyObject_IsTrue(o_vertices);
  if (o_vnormals) b_vnormals = PyObject_IsTrue(o_vnormals);
  if (o_vnormgrads) b_vnormgrads = PyObject_IsTrue(o_vnormgrads);
  
  if (!o_vertices && !o_vnormals && !o_vnormgrads) return NULL;
  
  double params[] = {q, F, d, Omega0};  
  
  Tmarching<double, Tgen_roche<double>> march(params);
  
  double n[3], g, *pg = 0;
  
  std::vector<T3Dpoint<double>> V, *NatV = 0;

  std::vector<double> *GatV = 0;

  PyArray_To3DPointVector(oV, V);
  
  int Nv = V.size();
  
  if (b_vnormals) {
    NatV = new std::vector<T3Dpoint<double>>;
    NatV->reserve(Nv);
  }
  
  if (b_vnormgrads) {
    GatV = new std::vector<double>;
    GatV->reserve(Nv);
    pg = &g;
  }
  
  for (auto && v: V) { 
    march.project_onto_potential(v.data, v.data, n, max_iter, pg);
    
    if (b_vnormals) NatV->emplace_back(n);
    if (b_vnormgrads) GatV->emplace_back(g);
  }
  
  PyObject *results = PyDict_New();
  
  if (b_vertices)
    PyDict_SetItemStringStealRef(results, "vertices", PyArray_From3DPointVector(V));
  
  if (b_vnormals) {
    PyDict_SetItemStringStealRef(results, "vnormals", PyArray_From3DPointVector(*NatV));
    delete NatV;
  }    

  if (b_vnormgrads) {
    PyDict_SetItemStringStealRef(results, "vnormgrads", PyArray_FromVector(*GatV));
    delete GatV;
  }    

  return results;
}

/*
  C++ wrapper for Python code:

    Calculating the horizon on the Roche lobes.
    
  Python:

    H = roche_horizon(v, q, F, d, Omega0, <keywords>=<value>)
    
  with arguments
  
  positionals: necessary
    v[3] - 1-rank numpy array of floats: direction of the viewer  
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega0: float - value of the generalized Kopal potential
  
  keywords:
    length: integer, default 1000, 
      approximate number of points on a horizon
    choice: interr, default 0:
      0 - searching a point on left lobe
      1 - searching a point on right lobe
      2 - searching a point for overcontact case
  Return: 
    H: 2-rank numpy array of 3D point on a horizon
*/

static PyObject *roche_horizon(PyObject *self, PyObject *args, PyObject *keywds) {

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"v",
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"length",
    (char*)"choice",
    NULL
  };
  
  PyArrayObject *oV;
  
  double q, F, d, Omega0;   
    
  int 
    length = 1000,
    choice  = 0,
    max_iter = 100;
  
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!dddd|ii", kwlist,
      &PyArray_Type, &oV, 
      &q, &F, &d, &Omega0,
      &length,
      &choice)){
    std::cerr << "roche_horizon::Problem reading arguments\n";
    return NULL;
  }
  
  double 
    params[] = {q, F, d, Omega0},
    *view = (double*) PyArray_DATA(oV);
  
  double p[3];
  
  //
  //  Find a point on horizon
  //
  if (!gen_roche::point_on_horizon(p, view, choice, Omega0, q, F, d, max_iter)) {
    std::cerr 
    << "roche_horizon::Convergence to the point on horizon failed\n";
    return NULL;
  }
  
  //
  // Estimate the step
  //
  double dt = 0; 
  
  if (choice == 0 || choice == 1)
    dt = utils::m_2pi*utils::hypot3(p)/length;
  else
    dt = 2*utils::m_2pi*utils::hypot3(p)/length;
  
  //
  //  Find the horizon
  //
  
  Thorizon<double, Tgen_roche<double>> horizon(params);
    
  std::vector<T3Dpoint<double>> H;
 
  if (!horizon.calc(H, view, p, dt)) {
   std::cerr 
    << "roche_horizon::Convergence to the point on horizon failed\n";
    return NULL;
  }

  return PyArray_From3DPointVector(H);
}

/*
  C++ wrapper for Python code:

    Calculating the horizon on the rotating star.
    
  Python:

    H = rotstar_horizon(v, q, F, d, Omega0, <keywords>=<value>)
    
  with arguments
  
  positionals: necessary
    v[3] - 1-rank numpy array of floats: direction of the viewer  
    omega: float - parameter of the potential
    Omega0: float - value of the potential of the rotating star
    
  keywords:
    length: integer, default 1000, 
      approximate number of points on a horizon

  Return: 
    H: 2-rank numpy array of floats -- 3D points on a horizon
*/

static PyObject *rotstar_horizon(PyObject *self, PyObject *args, PyObject *keywds) {

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"v",
    (char*)"omega",
    (char*)"Omega0",
    (char*)"length",
    NULL
  };
  
  PyArrayObject *oV;
  
  double omega, Omega0;   
    
  int 
    length = 1000;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!dd|i", kwlist,
      &PyArray_Type, &oV, 
      &omega, &Omega0,
      &length)){
    std::cerr << "rotstar_horizon::Problem reading arguments\n";
    return NULL;
  }

  double 
    params[] = {omega, Omega0},
    *view = (double*) PyArray_DATA(oV);
  
  double p[3];
  
  //
  //  Find a point on horizon
  //
  if (!rot_star::point_on_horizon(p, view, Omega0, omega)) {
    std::cerr 
    << "rotstar_horizon::Convergence to the point on horizon failed\n";
    return NULL;
  }
  
  //
  // Estimate the step
  //
  
  double dt = utils::m_2pi*utils::hypot3(p)/length;
  
  //
  //  Find the horizon
  //
  
  Thorizon<double, Trot_star<double>> horizon(params);
    
  std::vector<T3Dpoint<double>> H;
 
  if (!horizon.calc(H, view, p, dt)) {
   std::cerr 
    << "rotstar_horizon::Convergence to the point on horizon failed\n";
    return NULL;
  }

  return PyArray_From3DPointVector(H);
}


/*
  C++ wrapper for Python code:

    Calculation of the ranges of Roche lobes on x-axix
    
  Python:

    xrange = roche_xrange(q, F, d, Omega0, <keywords>=<value>)
    
  with arguments
  
  positionals: necessary
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega0: float - value of the generalized Kopal potential
  
  keywords:
    choice: integer, default 0:
      0 - searching a point on left lobe
      1 - searching a point on right lobe
      2 - searching a point for overcontact case
  
  Return: 
    xrange: 1-rank numpy array of two numbers p
*/

static PyObject *roche_xrange(PyObject *self, PyObject *args, PyObject *keywds) {

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"choice",
    NULL
  };
    
  double q, F, d, Omega0;   
    
  int choice  = 0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dddd|i", kwlist,
      &q, &F, &d, &Omega0, &choice)){
    std::cerr << "roche_xrange::Problem reading arguments\n";
    return NULL;
  }
  
  if (choice < 0 || choice > 2) {
    std::cerr 
      << "roche_xrange::This choice of computation is not supported\n";
    return NULL;
  }
  
  double *xrange = new double [2];
  
  if (!gen_roche::lobe_xrange(xrange, choice, Omega0, q, F, d, true)){
      std::cerr << "roche_xrange::Determining lobe's boundaries failed\n";
      return NULL;
  }
  
  npy_intp dims = 2;
  
  PyObject *results = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, xrange);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)results, NPY_ARRAY_OWNDATA);
  
  return results;
}

/*
  C++ wrapper for Python code:

  Rastering of Roche lobes -- determining a square 3D grid inside a
  Roche lobe with points/vertices determined by formula
      
    vec{r}(i_1,i_2,i_3) = vec{r}_0 + (L_1 i_1, L_2 i_2, L_3 i_3)
    
  with indices
      
    i_k in [0, N_k - 1] : k = 1,2,3 
    
  and step sizes
    
    L_k : k = 1,2,3 
      
  Python:

    dict = roche_square_grid(q, F, d, dims, <keywords>=<value> )
    
  with arguments
  
  positionals: necessary
  
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega0: float - value of the generalized Kopal potential  
    dims: 1-rank numpy arrays of 3 integers = [N_0, N_1, N_2]
  
  keywords: optional
    choice: integer, default 0:
      0 - searching a point on left lobe
      1 - searching a point on right lobe
      2 - searching a point for overcontact case
    
    boundary_list: boolean, default false
      return the list of boundary points
  
    boundary_mark: boolean, default false
      mark boundary point by 2 in the mask
  
  Return: a dictionary with keyword
  
    bmask: 3-rank numpy array of uint8, representing binary mask
      b[i1,i2,i3] in {0,1,2} 
        b == 2: boundary point (if marking enabled)
        b == 1: point means that a point is in Roche lobe
        b == 0: othewise
      and indices are
        i_k in [0, N_k -1]
      
    bbox:
      2-rank numpy array of float = [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
  
    origin: origin of grid r0 = [x, y, z]
      1-rank numpy of 3 floats
      
    steps: step sizes/cell dimensions [L_1, L_2, L_3]
      1-rank numpy of 3 floats

    boundary:
      2-rank numpy array of int: vector of triples of indices
*/


static PyObject *roche_square_grid(PyObject *self, PyObject *args, PyObject *keywds) {

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"Omega0",
    (char*)"dims",
    (char*)"choice",
    (char*)"boundary_list",
    (char*)"boundary_mark",
    NULL
  };
  
  double q, F, d, Omega0;   
    
  int choice  = 0;
  
  bool 
    b_boundary_list = false,
    b_boundary_mark = false;
    
  PyArrayObject *o_dims;
  
  PyObject 
    *o_boundary_list = 0,
    *o_boundary_mark = 0;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddddO!|iO!O!", kwlist,
      &q, &F, &d, &Omega0,    // necessary
      &PyArray_Type, &o_dims, 
      &choice,                // optional
      &PyBool_Type, &o_boundary_list,
      &PyBool_Type, &o_boundary_mark
      )){
    std::cerr << "roche_square_grid::Problem reading arguments\n";
    return NULL;
  }

  if (choice < 0 || choice > 2) {
    std::cerr << "roche_square_grid::This choice is not supported\n";
    return NULL;
  }
  
  int dims[3];
  {
    void *p = PyArray_DATA(o_dims);
    
    int size = PyArray_ITEMSIZE(o_dims);
    
    if (size == sizeof(int))
      for (int i = 0; i < 3; ++i) dims[i] = ((int*)p)[i];
    else if (size == sizeof(long)) 
      for (int i = 0; i < 3; ++i) dims[i] = ((long*)p)[i];
    else  {
      std::cerr 
        << "roche_square_grid::This type of dims is not supported\n";
      return NULL;
    }
  }
  
  if (o_boundary_list) b_boundary_list = PyObject_IsTrue(o_boundary_list);
  if (o_boundary_mark) b_boundary_mark = PyObject_IsTrue(o_boundary_mark);
  
  //
  // Determining the ranges
  //
  
  double ranges[3][2];
  
  bool checks = true;
  // x - range
  if (!gen_roche::lobe_xrange(ranges[0], choice, Omega0, q, F, d, checks)) {
    std::cerr << "roche_square_grid::Failed to obtain xrange\n";
    return NULL;
  }
  
  switch (choice) {
    
    case 0: // left lobe 
      // y - range
      ranges[1][0] = -(ranges[1][1] = gen_roche::lobe_ybound_L(Omega0, q, F, d));
      // z - range
      ranges[2][0] = -(ranges[2][1] = gen_roche::poleL(Omega0, q, F, d));
      break;
      
    case 1: // right lobe
      // y - range
      ranges[1][0] = -(ranges[1][1] = gen_roche::lobe_ybound_R(Omega0, q, F, d));
      // z -range
      ranges[2][0] = -(ranges[2][1] = gen_roche::poleR(Omega0, q, F, d));
      break;
      
    default:
      // y - range    
      ranges[1][0] = -(ranges[1][1] = 
        std::max(
          gen_roche::lobe_ybound_L(Omega0, q, F, d), 
          gen_roche::lobe_ybound_R(Omega0, q, F, d)
        ));
        
      // z -range      
      ranges[2][0] = -(ranges[2][1] = 
        std::max(
          gen_roche::poleL(Omega0, q, F, d), 
          gen_roche::poleR(Omega0, q, F, d)
        ));      
  }
  
  //
  // Determining characteristics of the grid
  //
  double r0[3], L[3];
  for (int i = 0; i < 3; ++i){
    r0[i] = ranges[i][0];
    L[i] = (ranges[i][1] - ranges[i][0])/(dims[i]-1);
  }

  
  //
  // Scan over the Roche lobe
  //
  int size = dims[0]*dims[1]*dims[2];
  
  std::uint8_t *mask = new std::uint8_t [size];
  
  {
    double params[4] = {q, F, d, Omega0}, r[3]; 
    
    Tgen_roche<double> roche(params);
  
    std::uint8_t *m = mask;
    
    std::memset(m, 0, size);
    
    int u[3];
    
    for (u[0] = 0; u[0] < dims[0]; ++u[0])
      for (u[1] = 0; u[1] < dims[1]; ++u[1])
        for (u[2] = 0; u[2] < dims[2]; ++u[2], ++m) {
          
          for (int i = 0; i < 3; ++i) r[i] = r0[i] + u[i]*L[i];
          
          if (roche.constrain(r) <= 0) *m = 1;
        }  
    
  }
  
  //
  // Mark boundary points
  //
  
  std::vector<int> bpoints;
   
  if (b_boundary_list || b_boundary_mark) {
    
    int index, u[3];
    
    std::uint8_t b, *m_prev, *m = mask;
    

    // scan along z direction
    for (u[0] = 0; u[0] < dims[0]; ++u[0])
      for (u[1] = 0; u[1] < dims[1]; ++u[1]) {
        b = 0;
        for (u[2] = 0; u[2] < dims[2]; ++u[2], ++m) {
          
          if (b == 0 && *m != 0) {
            if (b_boundary_mark) *m = 2;
            if (b_boundary_list)
              bpoints.push_back(u[2] + dims[2]*(u[1] + u[0]*dims[1]));
            b = 1;
          } else if (b == 1 && *m == 0) {
            if (b_boundary_mark) *m_prev = 2;
            if (b_boundary_list)
              bpoints.push_back(u[2] + dims[2]*(u[1] + u[0]*dims[1]));
            b = 0;
          }
          
          m_prev = m;
        }
      }
      
    // scan along y direction
    for (u[0] = 0; u[0] < dims[0]; ++u[0])
      for (u[2] = 0; u[2] < dims[2]; ++u[2]) {
        b = 0;
        for (u[1] = 0; u[1] < dims[1]; ++u[1]) {
          
          m = mask + (index = u[2] + dims[2]*(u[1] + u[0]*dims[1]));

          if (b == 0 && *m != 0) {
            if (b_boundary_mark) *m = 2;
            if (b_boundary_list) bpoints.push_back(index); 
            b = 1;
          } else if (b == 1 && *m == 0) {
            if (b_boundary_mark) *m_prev = 2;
            if (b_boundary_list) bpoints.push_back(index);
            b = 0;
          }
          
          m_prev = m;
        }
      }
      
    // scan along x direction
    for (u[1] = 0; u[1] < dims[1]; ++u[1])
      for (u[2] = 0; u[2] < dims[2]; ++u[2]) {
        b = 0;
        for (u[0] = 0; u[0] < dims[0]; ++u[0]) {
          
          m = mask + (index = u[2] + dims[2]*(u[1] + u[0]*dims[1]));

          if (b == 0 && *m != 0) {
            if (b_boundary_mark) *m = 2;
            if (b_boundary_list) bpoints.push_back(index); 
            b = 1;
          } else if (b == 1 && *m == 0) {
            if (b_boundary_mark) *m_prev = 2;
            if (b_boundary_list) bpoints.push_back(index); 
            b = 0;
          }
          
          m_prev = m;
        }
      }
  }
   
  //
  // Returning results
  //
  
  double 
    *origin = new double [3],
    *steps = new double [3],
    *bbox = new double [6];


  for (int i = 0; i < 3; ++i){
    origin[i] = r0[i];
    steps[i] = L[i];
    bbox[2*i] = ranges[i][0];
    bbox[2*i + 1] = ranges[i][1];
  }
  
  npy_intp nd[3] = {3, 2, 0};
    
  PyObject 
    *o_origin = PyArray_SimpleNewFromData(1, nd, NPY_DOUBLE, origin),
    *o_steps = PyArray_SimpleNewFromData(1, nd, NPY_DOUBLE, steps),
    *o_bbox = PyArray_SimpleNewFromData(2, nd, NPY_DOUBLE, bbox);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)o_origin, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_steps, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_bbox, NPY_ARRAY_OWNDATA);
  
  // storing the mask
  for (int i = 0; i < 3; ++i) nd[i] = dims[i];
  PyObject *o_mask = PyArray_SimpleNewFromData(3, nd, NPY_UINT8, mask);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_mask, NPY_ARRAY_OWNDATA);
  
  PyObject *results = PyDict_New();
  PyDict_SetItemStringStealRef(results, "origin", o_origin);
  PyDict_SetItemStringStealRef(results, "steps", o_steps);
  PyDict_SetItemStringStealRef(results, "bbox", o_bbox);
  PyDict_SetItemStringStealRef(results, "mask", o_mask);
  
  
  if (b_boundary_list) {
    nd[0] = bpoints.size();
    nd[1] = 3;
    
    PyObject *o_blist = PyArray_SimpleNew(2, nd, NPY_INT);
    
    void *p = PyArray_DATA((PyArrayObject*)o_blist);
    
    int l[2] = {dims[1]*dims[2], dims[2]},
        size = PyArray_ITEMSIZE((PyArrayObject*)o_blist);
        
    // index = u[2] + dims[2]*(u[1] + u[0]*dims[1]));
  
    if (size == sizeof(int)) {

      int *q = (int*)p;
      
      for (auto && b : bpoints) {
        q[0] = b/l[0]; 
        
        b -= q[0]*l[0];
        
        q[1] = b/l[1];
        
        q[2] = b - q[1]*l[1];
        
        q += 3;
      } 
    
    } else if (size == sizeof(long)) {
      long *q = (long*)p;
      
      for (auto && b : bpoints) {
        q[0] = b/l[0]; 
        
        b -= q[0]*l[0];
        
        q[1] = b/l[1];
        
        q[2] = b - q[1]*l[1];
        
        q += 3;
      } 
    }
    
    PyDict_SetItemStringStealRef(results, "boundary", o_blist);
  }
      
  return results;
}


/*
  C++ wrapper for Python code:

    Calculating the limb darkening function D(mu) in speherical coordinates
    
    vec r = r (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
  
    and mu=cos(theta)
  
  Python:

    value = ld_D(mu, descr, params)
    
  with arguments

    mu: float
    descr: string
           supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "claret"      4 parameters
    params: 1-rank numpy array 
  Return: 
    value of D(mu) for a given LD model 
*/

static PyObject *ld_D(PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "ld_D";
  
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"mu",          
    (char*)"descr",
    (char*)"params",
    NULL
  };
  
  double mu;
  
  PyObject *o_descr;
  
  PyArrayObject *o_params;
  
  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "dO!O!", kwlist, 
        &mu, 
        &PyString_Type, &o_descr, 
        &PyArray_Type,  &o_params)
      ){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
 
  TLDmodel_type type = LD::type(PyString_AsString(o_descr));
  
  if (type == NONE) {
    std::cerr << fname << "::This model is not supported\n";
    return NULL;  
  }
  
  return PyFloat_FromDouble(LD::D(type, mu, (double*)PyArray_DATA(o_params)));
}



/*
  C++ wrapper for Python code:

    Calculating integral of limb darkening function D(mu) over the
    unit half sphere:
    
    int_0^pi 2pi cos(theta) sin(theta) D(cos(theta))
    
  Python:

    value = ld_D0(descr, params)
    
  with arguments

    descr: string
           supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "claret"      4 parameters
    params: 1-rank numpy array
     
  Return: 
    value of integrated D(mu) for a given LD model 
*/

static PyObject *ld_D0(PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "ld_D0";
  
  //
  // Reading arguments
  //

  char *kwlist[] = {   
    (char*)"descr",
    (char*)"params",
    NULL
  };
  
  PyObject *o_descr;
  
  PyArrayObject *o_params;
  
  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "O!O!", kwlist, 
        &PyString_Type, &o_descr, 
        &PyArray_Type,  &o_params)
      ){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
 
  TLDmodel_type type = LD::type(PyString_AsString(o_descr));
  
  if (type == NONE) {
    std::cerr << fname << "::This model is not supported\n";
    return NULL;  
  }
  
  return PyFloat_FromDouble(LD::D0(type, (double*)PyArray_DATA(o_params)));
}



/*
  C++ wrapper for Python code:

    Calculating the gradient fo the limb darkening function D(mu) 
    with respect to parameters at constant argument in speherical coordinates
    
    vec r = r (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
  
    and mu = cos(theta)
  
  Python:

    grad_{parameters} D = ld_gradparD(mu, descr, params)
    
  with arguments

    mu: float
    descr: string:
          "uniform"     0 parameters
          "linear"      1 parameters
          "quadratic"   2 parameters
          "nonlinear"   3 parameters
          "logarithmic" 2 parameters
          "square_root" 2 parameters
          "claret"      4 parameters
    
    params: 1-rank numpy array 
     
  Return: 
    1-rank numpy array of floats: gradient of the function D(mu) w.r.t. parameters
*/

static PyObject *ld_gradparD(PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "ld_gradparD";
   
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"mu",          
    (char*)"descr",
    (char*)"params",
    NULL
  };
  
  double mu;
  
  PyObject *o_descr;
  
  PyArrayObject *o_params;

  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "dO!O!", kwlist, 
        &mu, 
        &PyString_Type, &o_descr,
        &PyArray_Type,  &o_params)
      ) {
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
  
  TLDmodel_type type = LD::type(PyString_AsString(o_descr));
  
  if (type == NONE) {
    std::cerr << fname << "::This model is not supported\n";
    return NULL;  
  }
  
  int nr_par = LD::nrpar(type);
  
  double *g = new double [nr_par];
    
  LD::gradparD(type, mu, (double*)PyArray_DATA(o_params), g);
  
  // Return the results
  npy_intp dims = nr_par;

  PyObject *results = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, g);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)results, NPY_ARRAY_OWNDATA);

  return results;
}


/*
  C++ wrapper for Python code:

    Determining number of float parameters particula 
    limb darkening model
    
  Python:

    value = ld_nrpar(descr)
    
  with arguments

    descr: string (bytes)  
          supported ld models:
            "uniform"     0 parameters
            "linear"      1 parameters
            "quadratic"   2 parameters
            "nonlinear"   3 parameters
            "logarithmic" 2 parameters
            "square_root" 2 parameters
            "claret"      4 parameters
  Return: 
    int: number of parameters 
*/

static PyObject *ld_nrpar(PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "ld_nrpar";
  
  //
  // Reading arguments
  //

  char *kwlist[] = {         
    (char*)"descr",
    NULL
  };
   
  PyObject *o_descr;

  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "O!", kwlist, 
        &PyString_Type, &o_descr)
      ){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
 
  TLDmodel_type type = LD::type(PyString_AsString(o_descr));
  
  if (type == NONE) {
    std::cerr << fname << "::This model is not supported\n";
    return NULL;  
  }
    
  return PyInt_FromLong(LD::nrpar(type));
}

/*
  C++ wrapper for Python code:

    Check the parameters of the particular limb darkening model
    
  Python:

    value = ld_check(descr, params)
    
  with arguments

    descr: string (bytes)  
          supported ld models:
            "uniform"     0 parameters
            "linear"      1 parameters
            "quadratic"   2 parameters
            "nonlinear"   3 parameters
            "logarithmic" 2 parameters
            "square_root" 2 parameters
            "claret"      4 parameters
    params: 1-rank numpy array of float
  
  Return: 
    true: int: number of parameters 
*/

static PyObject *ld_check(PyObject *self, PyObject *args, PyObject *keywds) {
  
  const char *fname = "ld_check";
  
  //
  // Reading arguments
  //

  char *kwlist[] = {         
    (char*)"descr",
    (char*)"params",
    NULL
  };
   
  PyObject *o_descr;
  
  PyArrayObject *o_params;

  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "O!O!", kwlist, 
        &PyString_Type, &o_descr,
        &PyArray_Type,  &o_params)
      ){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }
 
  TLDmodel_type type = LD::type(PyString_AsString(o_descr));
  
  if (type == NONE) {
    std::cerr << fname << "::This model is not supported\n";
    return NULL;  
  }
  
  return PyBool_FromLong(LD::check(type, (double*)PyArray_DATA(o_params)));
}


/*
  C++ wrapper for Python code:

    Reading files with expansion coefficients needed for 
    Wilson-Devinney type of atmospheres.
  
  Python:

    dict = wd_readdata(filename_planck, filename_atm)
    
  with arguments
  
    filename: string - filename of the file loaded
    
  Returns: dictionary with keys
    
    planck_table: coefficients for calculating Planck intensity
      1-rank numpy array of floats
    
    atm_table: coefficients for calculating light intensity with atmospheres
      1-rank numpy array of floats
*/ 
static PyObject *wd_readdata(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"filename_planck",
    (char*)"filename_atm",
    NULL
  };
  
  PyObject *ofilename_planck, *ofilename_atm;
  
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!", kwlist,
      &PyString_Type, &ofilename_planck, 
      &PyString_Type, &ofilename_atm)
      ) {
    std::cerr << "wd_readdata::Problem reading arguments\n";
    return NULL;
  }
  
  double 
    *planck_table = new double[wd_atm::N_planck],
    *atm_table = new double[wd_atm::N_atm];
  
  //
  // Reading 
  //
  
  int len[2] = {
    wd_atm::read_data<double, wd_atm::N_planck>(PyString_AsString(ofilename_planck), planck_table),
    wd_atm::read_data<double, wd_atm::N_atm>(PyString_AsString(ofilename_atm), atm_table)
  };
  
  //
  // Checks
  //
  if (len[0] != wd_atm::N_planck || len[1] != wd_atm::N_atm) {
    std::cerr << "wd_readdata::Problem reading data\n";
    delete [] planck_table;
    delete [] atm_table;
    return NULL;
  } 
  
  //
  // Returning results
  //
  
  PyObject *results = PyDict_New();
  
  {
    npy_intp dims = wd_atm::N_planck;
    PyObject *pya = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, planck_table);
    PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
    PyDict_SetItemStringStealRef(results, "planck_table", pya);
  }

 {
    npy_intp dims = wd_atm::N_atm;
    PyObject *pya = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, atm_table);
    PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
    PyDict_SetItemStringStealRef(results, "atm_table", pya);
  }

  return results;
}
#if 0
/*
  C++ wrapper for Python code:

    Computing Planck central intensity and its logarithm. Works for 
    tempratues in the range [500,500300] K.

  Python:
   
    result = wd_planckint(t, ifil, planck_table)
   
  Input:
  
    t: float - temperature
    ifil: integer - index of the filter 1,2, ... 
    planck_table: 1-rank numpy array of floats - array of coefficients 
    
  Return:
    
    result : 1-rank numpy array of floats = [ylog, y] 

  with    
    ylog: float - log of Planck central intensity
    y: float - Planck central intensity

*/
static PyObject *wd_planckint(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"t",
    (char*)"ifil",
    (char*)"planck_table",
    NULL
  };
  
  int ifil;
 
  double t;
  
  PyArrayObject *oplanck_table;
  
  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "diO!", kwlist, 
        &t, &ifil, &PyArray_Type, &oplanck_table
      )) {
    std::cerr << "wd_planckint::Problem reading arguments\n";
    return NULL;
  }
  //
  // Calculate without checks
  //
  
  double *y = new double [2];
      
  wd_atm::planckint(t, ifil, 
                    (double*) PyArray_DATA(oplanck_table), 
                    y[0], y[1]);
  
  //
  // Returing result
  //
  
  npy_intp dims = 2;
  PyObject *res = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, y);
  PyArray_ENABLEFLAGS((PyArrayObject *)res, NPY_ARRAY_OWNDATA);
  
  return res;
}
#else
/*
  C++ wrapper for Python code:

    Computing the logarithm of the Planck central intensity. Works for 
    temperatures in the range [500,500300] K.

  Python:
   
    result = wd_planckint(t, ifil, planck_table)
   
  Input:
  
  positional: necessary
  
    t:  float - temperature or
        1-rank numpy array of float - temperatures
        
    ifil: integer - index of the filter 1,2, ... 
    planck_table: 1-rank numpy array of floats - array of coefficients 
     
  Return:
    
    result : 
      ylog: float - log of Planck central intensity or
      1- rank numpy array of floats - log of Planck central intensities
    
    Note:
      In the case of errors in calculations ylog/entry in numpy array is NaN.
*/
static PyObject *wd_planckint(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"t",
    (char*)"ifil",
    (char*)"planck_table",
    NULL
  };
  
  int ifil;
 
  PyObject *ot;

  PyArrayObject *oplanck_table;
  
  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "OiO!", kwlist, 
        &ot, &ifil, &PyArray_Type, &oplanck_table)
      ) {
    std::cerr << "wd_planckint::Problem reading arguments\n";
    return NULL;
  }
  
  double *planck_table = (double*) PyArray_DATA(oplanck_table);
    
  if (PyFloat_Check(ot)) { // argument if float
    
    double ylog, t = PyFloat_AS_DOUBLE(ot);
    
    //
    //  Calculate ylog and return
    //
    
    if (wd_atm::planckint_onlylog(t, ifil, planck_table, ylog))
      return PyFloat_FromDouble(ylog);
    else {
      std::cerr 
      << "wd_planckint::Failed to calculate Planck central intensity\n";
      return PyFloat_FromDouble(std::nan(""));
    }
  
  } else if (PyArray_Check(ot)) {  // argument is a numpy array
    
    int n = PyArray_DIM((PyArrayObject *)ot, 0);
     
    if (n == 0) {
      std::cerr << "wd_planckint::Arrays of zero length\n";
      return NULL;
    }
    
    double *t =  (double*) PyArray_DATA((PyArrayObject *)ot),
            *results = new double [n];
    
    //
    //  Calculate ylog for an array 
    //

    bool ok = true;
    
    for (double *r = results, *r_e = r + n; r != r_e;  ++r, ++t)
      if (!wd_atm::planckint_onlylog(*t, ifil, planck_table, *r)) { 
        *r = std::nan("");
        ok = false;
      }
    
    if (!ok) {
      std::cerr 
      << "wd_planckint::Failed to calculate Planck central intensity at least once\n";
    }
    
    //
    // Return results
    //
  
    npy_intp dims = n;
    PyObject *oresults = 
      PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, results);
      
    PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
    
    return oresults;
  } 
  
  std::cerr 
    << "wd_planckint:: This type of temperature input is not supported\n";
  return NULL;
}


#endif

#if 0
/*
  C++ wrapper for Python code:
    
    Calculation of the light intensity and their logarithm 
    for a star with a certain atmosphere model.
  
  Python:
  
    results = wd_atmint(t, logg, abunin, ifil, planck_table, atm_table)
  
  Input:
  
   t:float - temperature
   logg:float - logarithm of surface gravity
   abunin:float - abundance/metallicity
   ifil: integer - index of the filter 1,2, ... 
   planck_table: 1-rank numpy array of float - planck table
   atm_table: 1-rank numpy array of float - atmospheres table

  Return:
  
    result : 1-rank numpy array of floats = [xintlog, xint, abunin]
  
  with
  
    xintlog - log of intensity
    xint - intensity
    abunin -  the allowed value nearest to the input value.
*/
static PyObject *wd_atmint(PyObject *self, PyObject *args, PyObject *keywds) {
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"t",
    (char*)"logg",
    (char*)"abunin",
    (char*)"ifil",
    (char*)"planck_table",
    (char*)"atm_table",
    NULL
  };

  int ifil;

  double t, logg, abunin;
    
  PyArrayObject *oplanck_table, *oatm_table;
  
  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "dddiO!O!", kwlist, 
        &t, &logg, &abunin, &ifil, 
        &PyArray_Type, &oplanck_table, 
        &PyArray_Type, &oatm_table
      )) {
    std::cerr << "wd_atmint::Problem reading arguments\n";
    return NULL;
  }
  //
  // Calculate without checks
  //
  
  double *y = new double [3];
  
  y[2] = abunin;
      
  wd_atm::atmx(t, logg, y[2], ifil, 
              (double*)PyArray_DATA(oplanck_table), 
              (double*)PyArray_DATA(oatm_table), 
              y[0], y[1]);
  
  //
  // Returing result
  //
  
  npy_intp dims = 3;
  PyObject *res = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, y);
  PyArray_ENABLEFLAGS((PyArrayObject *)res, NPY_ARRAY_OWNDATA);
  
  return res;
}
#else
/*
  C++ wrapper for Python code:
    
    Calculation of logarithm the light intensity from a star with a certain 
    atmosphere model.
  
  Python:
  
    results = wd_atmint(t, logg, abunin, ifil, planck_table, atm_table, <keywords>=<value>)
  
  Input:
  
  positional: necessary
  
   t:float - temperature or 
     1-rank numpy array of floats - temperatures
    
   logg:float - logarithm of surface gravity or
        1-rank numpy array of floats - temperatures
        
   abunin:float - abundance/metallicity
          1-rank numpy of floats - abundances
        
   ifil: integer - index of the filter 1,2, ... 
   planck_table: 1-rank numpy array of float - planck table
   atm_table: 1-rank numpy array of float - atmospheres table

  keywords: optional
  
    return_abunin: boolean, default false
    if allowed value of abunin should be returned 
    
  Return:
    
    if t is float:
      if return_abunin == true:
        result : 1-rank numpy array of floats = [xintlog, abunin]
      else
        result: float = xintlog 
    else
      if return_abunin == true:
        result : 2-rank numpy array of float -- array of [xintlog, abunin]
      else
        result: 1-rank numpy array of floats -- xintlogs
    
  with
  
    xintlog - log of intensity
    abunin -  allowed value nearest to the input value.
*/
static PyObject *wd_atmint(PyObject *self, PyObject *args, PyObject *keywds) {
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"t",
    (char*)"logg",
    (char*)"abunin",
    (char*)"ifil",
    (char*)"planck_table",
    (char*)"atm_table",
    (char*)"return_abunin",
    NULL
  };
  

  int ifil;

  bool return_abunin = false;

  PyObject *ot, *ologg, *oabunin, *oreturn_abunin = 0;
  
  PyArrayObject *oplanck_table, *oatm_table;
  
  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "OOOiO!O!|O!", kwlist, 
        &ot, &ologg, &oabunin, &ifil, 
        &PyArray_Type, &oplanck_table, 
        &PyArray_Type, &oatm_table,
        &PyBool_Type, &oreturn_abunin
      )) {
    std::cerr << "wd_atmint::Problem reading arguments\n";
    return NULL;
  }

  if (oreturn_abunin) return_abunin = PyBool_Check(oreturn_abunin);
  
  //    
  // Check type of temperature argument and read them
  //
  
  double t, logg, abunin, *pt, *plogg, *pabunin;
  
  int n = -2;
  
  if (PyFloat_Check(ot)){ 
    n = -1;
    // arguments
    t = PyFloat_AS_DOUBLE(ot),
    logg = PyFloat_AS_DOUBLE(ologg),
    abunin = PyFloat_AS_DOUBLE(oabunin);
  
  } else if (PyArray_Check(ot)) {
    
    n = PyArray_DIM((PyArrayObject *)ot, 0);
    
    if (n == 0) {
      std::cerr << "wd_planckint::Arrays are of zero length\n";
      return NULL;
    }
        
    // arguments
    pt = (double*)PyArray_DATA((PyArrayObject *)ot),
    plogg = (double*)PyArray_DATA((PyArrayObject *)ologg),
    pabunin = (double*)PyArray_DATA((PyArrayObject *)oabunin);  

  } else {
    std::cerr 
      << "wd_planckint:: This type of temperature input is not supported\n";
    return NULL;
  }
  
  //
  // Do calculations and storing it in PyObject
  //
  
  PyObject *oresults;
    
  double *planck_table = (double*)PyArray_DATA(oplanck_table), 
         *atm_table = (double*)PyArray_DATA(oatm_table);

  if (return_abunin) {   // returning also abundances
    
    //
    //  Calculate yintlog and abundance
    //
    
    if (n == -1){ // single calculation
    
      double *r = new double[2]; // to store results

      r[1] = abunin;
      
      // do calculation    
      if (!wd_atm::atmx_onlylog(t, logg, r[1], ifil, planck_table, atm_table, r[0])) {
        std::cerr << "wd_atmint::Failed to calculate logarithm of intensity\n";
        r[0] = std::nan("");
      }
      
      // store results in numpy array
      npy_intp dims = 2;
      oresults = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, r);
      
    } else {  // calculation whole array
     
      double *results = new  double [2*n]; // to store results

             
      bool ok = true;
      for (double *r = results, *r_e = r + 2*n; r != r_e; r += 2, ++pt, ++plogg, ++pabunin){
        
        r[1] = *pabunin;
        
        if (!wd_atm::atmx_onlylog(*pt, *plogg, r[1], ifil, planck_table, atm_table, r[0])) {
          r[0] = std::nan("");
          ok = false;
        }
      }  
      
      if (!ok)
        std::cerr << "wd_atmint::Failed to calculate logarithm of intensity at least once\n";
      
      // store results in numpy array
      npy_intp dims[2] = {n, 2};
      oresults = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, results); 
    }    
    
    PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
    
  } else {                    // returning only logarithm of intensities

    //
    //  Calculate yintlogs 
    //
    
    if (n == -1){ // single calculation
        
      double r; // log of intensity
        
      if (wd_atm::atmx_onlylog(t, logg, abunin, ifil, planck_table, atm_table, r))
        oresults = PyFloat_FromDouble(r);
      else {
        std::cerr << "wd_atmint::Failed to calculate logarithm of intensity\n";
        oresults = PyFloat_FromDouble(std::nan(""));
      }
       
    } else { // calculation whole array
      
      double *results = new double [n],  // to store results
             tmp;
              
      bool ok = true;
      
      for (double *r = results, *r_e = r + n; r != r_e; ++r, ++pt, ++plogg, ++pabunin){
        
        tmp = *pabunin;
        
        if (!wd_atm::atmx_onlylog(*pt, *plogg, tmp, ifil, planck_table, atm_table, *r)) {       
          *r = std::nan("");
          ok = false;
        }
      }  
      
      if (!ok)
        std::cerr 
        << "wd_atmint::Failed to calculate logarithm of intensity at least once\n";
      
      npy_intp dims = n;
      oresults = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, results);
      PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
    }  
  }
  
  return oresults;
}

#endif

/*
  C++ wrapper for python code:
 
    Multi-dimensional linear interpolation of gridded data. Gridded data 
    means data ordered in a grid.
  
  Python:
  
    results = interp(req, axes, grid) 
  
  with arguments:
    req: 2-rank numpy array = MxN array (M rows, N columns) where 
        each column stores the value along the respective axis and 
        each row corresponds to a single point to be interpolated;
  
    axes: tuple of N numpy arrays, with each array holding all unique 
          vertices along its respective axis in ascending order 
    
    grid: N+1-rank numpy array =  N1xN2x...xNNxNv array, 
          where Ni are lengths of individual axes, and the last element
          is the vertex value of dimension Nv

  Example: we have the following vertices with corresponding values:

    v0 = (0, 2), f(v0) = 5
    v1 = (0, 3), f(v0) = 6
    v2 = (1, 3), f(v0) = 7
    v3 = (1, 2), f(v0) = 8

  We are interested in f(0.5, 2.5) and f(0.75, 2.25). Note that
  all values need to be floats, thus:

    req = np.array([[0.5, 2.5], [0.75, 2.25]])
    axes = (np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    grid = np.array([[[5.0], [6.0]], [[7.0], [8.0]]])

  Return: 
    2-rank numpy array = MxNv array of interpolated values
*/

static PyObject *interp(PyObject *self, PyObject *args, PyObject *keywds) {
  
  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"req",
    (char*)"axes",
    (char*)"grid",
    NULL
  };
       
  PyObject *o_axes;
 
  PyArrayObject *o_req, *o_grid;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds, "O!O!O!", kwlist, 
      &PyArray_Type, &o_req, 
      &PyTuple_Type, &o_axes, 
      &PyArray_Type, &o_grid)) {
      
    std::cerr 
      << "interp::argument type mismatch: req and grid need to be numpy "
      << "arrays and axes a tuple of numpy arrays.\n";
    
    return NULL;
  }
  
   PyArrayObject 
    *o_req1 = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)o_req, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
    *o_grid1 = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)o_grid, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  
  if (!o_req1 ||!o_grid1) {
    
    if (!o_req1) std::cerr << "interp::req failed transformation to IN_ARRAY\n";
    if (!o_grid1) std::cerr << "interp::grid failed transformation to IN_ARRAY\n";

    Py_DECREF(o_req1);
    Py_DECREF(o_grid1);
    return NULL;
  }

  int Na = PyTuple_Size(o_axes),      // number of axes
      Np = PyArray_DIM(o_req1, 0),     // number of points
      Nv = PyArray_DIM(o_grid1, Na),   // number of values interpolated
      Nr = Np*Nv;                     // number of returned values
  
  double
    *R = new double [Nr],                 // returned values
    *Q = (double *) PyArray_DATA(o_req1),  // requested values
    *G = (double *) PyArray_DATA(o_grid1); // grid of values
    

  // Unpack the axes
  int *L = new int [Na];      // number of values in axes
  double **A = new double* [Na]; // pointers to tuples in axes
  
  {
    PyArrayObject *p;
    for (int i = 0; i < Na; ++i) {
      p = (PyArrayObject*)PyTuple_GET_ITEM(o_axes, i); // no checks, borrows reference
      L[i] = (int) PyArray_DIM(p, 0);
      A[i] = (double *) PyArray_DATA(p);
    }
  }
  
  //
  // Do interpolation
  //
  
  Tlinear_interpolation<double> lin_iterp(Na, Nv, L, A, G);
  
  for (double *q = Q, *r = R, *re = r + Nr; r != re; q += Na, r += Nv) 
    lin_iterp.get(q, r);
  
  // clean copies of objects
  Py_DECREF(o_req1);
  Py_DECREF(o_grid1);
    
  // Clean data about axes
  delete [] L;  
  delete [] A;

  //
  // Return results
  //
   
  npy_intp dims[2] = {Np, Nv};
	PyObject *o_ret = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, R);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_ret, NPY_ARRAY_OWNDATA);
    
  return o_ret;
}

/*
  Define functions in module
   
  Some modification in declarations due to use of keywords
  Ref:
  * https://docs.python.org/2.0/ext/parseTupleAndKeywords.html
*/ 
static PyMethodDef Methods[] = {
  
  { "roche_critical_potential", 
    (PyCFunction)roche_critical_potential,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the critical potentials of Kopal potential for given "
    "values of q, F, and d."},
  
  { "rotstar_critical_potential", 
    rotstar_critical_potential,   
    METH_VARARGS, 
    "Determine the critical potentials of the rotating star potental "
    "for given values of omega."},
    
// --------------------------------------------------------------------
    
  { "roche_pole", 
    (PyCFunction)roche_pole,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the height of the pole of generalized Roche lobes for given "
    "values of q, F, d and Omega0"},
 
  { "rotstar_pole", 
    (PyCFunction)rotstar_pole,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the height of the pole of rotating star for given a omega."},

// --------------------------------------------------------------------
  { "rotstar_from_roche", 
    (PyCFunction)rotstar_from_roche,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine parameters of the rotating stars from parameters Roche "
    " by matching the poles"},

// --------------------------------------------------------------------
 
  { "roche_area_volume", 
    (PyCFunction)roche_area_volume,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the area and volume of the generalized Roche lobes for given "
    "values of q, F, d and Omega0."},
 
  { "rotstar_area_volume", 
    (PyCFunction)rotstar_area_volume,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the area and volume of the rotating star for given a omega "
    "and Omega0"},

// --------------------------------------------------------------------
 
  { "roche_Omega_at_vol", 
    (PyCFunction)roche_Omega_at_vol,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the value of the generalized Kopal potential at "
    "values of q, F, d and volume."},


  { "rotstar_Omega_at_vol", 
    (PyCFunction)rotstar_Omega_at_vol,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the value of the rotating star potential at "
    "values of omega and volume."},
// --------------------------------------------------------------------

  { "roche_gradOmega", 
    roche_gradOmega,   
    METH_VARARGS, 
    "Calculate the gradient and the value of the generalized Kopal potentil"
    " at given point [x,y,z] for given values of q, F and d."},  

    { "rotstar_gradOmega", 
    rotstar_gradOmega,   
    METH_VARARGS, 
    "Calculate the gradient and the value of the rotating star potential"
    " at given point [x,y,z] for given values of omega."},  

// --------------------------------------------------------------------

  { "roche_Omega", 
    roche_Omega,   
    METH_VARARGS, 
    "Calculate the value of the generalized Kopal potentil"
    " at given point [x,y,z] for given values of q, F and d."},  

    { "rotstar_Omega", 
    rotstar_Omega,   
    METH_VARARGS, 
    "Calculate the value of the rotating star potential"
    " at given point [x,y,z] for given values of omega."},  
 
// --------------------------------------------------------------------
    
  { "roche_gradOmega_only", 
    roche_gradOmega_only,   
    METH_VARARGS, 
    "Calculate the gradient of the generalized Kopal potential"
    " at given point [x,y,z] for given values of q, F and d."},   
  
  { "rotstar_gradOmega_only", 
    rotstar_gradOmega_only,   
    METH_VARARGS, 
    "Calculate the gradient of the rotating star potential"
    " at given point [x,y,z] for given values of omega."},   

// --------------------------------------------------------------------
  
  { "roche_marching_mesh", 
    (PyCFunction)roche_marching_mesh,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the triangular meshing of generalized Roche lobes for "
    "given values of q, F, d and value of the generalized Kopal potential "
    "Omega0. The edge of triangles used in the mesh are approximately delta."},
    
  { "rotstar_marching_mesh", 
    (PyCFunction)rotstar_marching_mesh,   
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the triangular meshing of a rotating star for given "
    "values of omega and value of the star potential Omega. The edge "
    "of triangles used in the mesh are approximately delta."},

// --------------------------------------------------------------------    
  
  { "mesh_visibility",
    (PyCFunction)mesh_visibility,
    METH_VARARGS|METH_KEYWORDS, 
    "Determine the ratio of triangle surfaces that are visible in a triangular mesh."},
  
  { "mesh_rough_visibility",
    mesh_rough_visibility,
    METH_VARARGS,
    "Classify the visibility of triangles of the mesh into hidden, partially hidden and visible"},
  
  { "mesh_offseting",
    (PyCFunction)mesh_offseting,
    METH_VARARGS|METH_KEYWORDS, 
    "Offset the mesh along the normals in vertices to match the area with reference area."},
    
  { "mesh_properties", 
    (PyCFunction)mesh_properties,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculate the properties of the triangular mesh."},

  { "mesh_export_povray",
    (PyCFunction)mesh_export_povray,
    METH_VARARGS|METH_KEYWORDS, 
    "Exporting triangular mesh into a Pov-Ray file."},
 
  { "mesh_radiosity_problem_triangles",
    (PyCFunction)mesh_radiosity_problem_triangles,
    METH_VARARGS|METH_KEYWORDS, 
    "Solving the radiosity problem with limb darkening as defined by a chosen reflection model."},
  
  { "mesh_radiosity_problem_triangles_nbody_convex",
    (PyCFunction)mesh_radiosity_problem_triangles_nbody_convex,
    METH_VARARGS|METH_KEYWORDS, 
    "Solving the radiosity problem with limb darkening as defined a chosen reflection model "
    "for n separate convex bodies using triangles as radiating surfaces."},
    
  { "mesh_radiosity_problem_vertices_nbody_convex",
    (PyCFunction)mesh_radiosity_problem_vertices_nbody_convex,
    METH_VARARGS|METH_KEYWORDS, 
    "Solving the radiosity problem with limb darkening as defined a chosen reflection model "
    "for n separate convex bodies using disks attached to vertices as "
    "radiating surfaces."},  
    
  { "mesh_radiosity_problem_vertices",
    (PyCFunction)mesh_radiosity_problem_vertices,
    METH_VARARGS|METH_KEYWORDS, 
    "Solving the radiosity problem with limb darkening as defined a chosen reflection model "
    "using disks attached to vertices as radiating surfaces."},
    
// --------------------------------------------------------------------    

  { "roche_reprojecting_vertices",
    (PyCFunction)roche_reprojecting_vertices,
    METH_VARARGS|METH_KEYWORDS, 
    "Reprojecting vertices onto the Roche lobe defined by q,F,d, and the value of"
    " generalized Kopal potential Omega."},
    
// --------------------------------------------------------------------    

  { "roche_central_points",
    (PyCFunction)roche_central_points,
    METH_VARARGS|METH_KEYWORDS, 
    "Determining the central points of triangular mesh on the Roche lobe"
    " defined by q,F,d, and the value of generalized Kopal potential Omega."},
    
// --------------------------------------------------------------------    
  { "roche_horizon",
    (PyCFunction)roche_horizon,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the horizon on the Roche lobe defined by view direction,"
    "q,F,d, and the value of generalized Kopal potential Omega."},


  { "rotstar_horizon",
    (PyCFunction)rotstar_horizon,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the horizon on the rotating star defined by view direction,"
    "omega, and the value of the potential"},

// --------------------------------------------------------------------
  { "roche_xrange",
    (PyCFunction)roche_xrange,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the range of the Roche lobes on x-axis at given"
    "q, F, d, and the value of generalized Kopal potential Omega."},

// --------------------------------------------------------------------
  { "roche_square_grid",
    (PyCFunction)roche_square_grid,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the square grid of the interior of the Roche lobes at given"
    "q, F, d, and the value of generalized Kopal potential Omega."},
// --------------------------------------------------------------------

  { "ld_D",
    (PyCFunction)ld_D,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the value of the limb darkening function."},

  { "ld_D0",
    (PyCFunction)ld_D0,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the integrated limb darkening function."},
        
  { "ld_gradparD",
    (PyCFunction)ld_gradparD,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating the gradient of the limb darkening function w.r.t. "
    "parameters."},

  { "ld_nrpar",
    (PyCFunction)ld_nrpar,
    METH_VARARGS|METH_KEYWORDS, 
    "Returns the number of required parameters."},
    
  { "ld_check",
    (PyCFunction)ld_check,
    METH_VARARGS|METH_KEYWORDS, 
    "Checking parameters if resulting D(mu) is in the range [0,1] for all mu."},
      
// --------------------------------------------------------------------

    { "wd_readdata",
    (PyCFunction)wd_readdata,
    METH_VARARGS|METH_KEYWORDS, 
    "Reading the file with WD coefficients."},
    
    
  { "wd_planckint",
    (PyCFunction)wd_planckint,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating Planck central intensity at given temperatures,"
    "filter index and array of coefficients"},

  { "wd_atmint",
    (PyCFunction)wd_atmint,
    METH_VARARGS|METH_KEYWORDS, 
    "Calculating intensity for a given atmospheres at given temperatures,"
    "filter index and array of coefficients"},

// --------------------------------------------------------------------

 {"interp", 
  (PyCFunction)interp, METH_VARARGS|METH_KEYWORDS, 
  "Multi-dimensional linear interpolation of arrays with gridded data."},  
  
  {NULL,  NULL, 0, NULL} // terminator record
};

static char const *Docstring =
  "Module wraps routines dealing with models of stars and "
  "triangular mesh generation and their manipulation.";



/* module initialization */
MOD_INIT(libphoebe) {
  
  PyObject *backend;
  
  MOD_DEF(backend, "libphoebe", Docstring, Methods)

  if (!backend) return MOD_ERROR_VAL;
    
  // Added to handle Numpy arrays
  // Ref: 
  // * http://docs.scipy.org/doc/numpy-1.10.1/user/c-info.how-to-extend.html
  import_array();
  
  return MOD_SUCCESS_VAL(backend);
}
