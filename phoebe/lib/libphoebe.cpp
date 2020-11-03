/*
  Wrappers for calculations concerning the

    * generalized Kopal potential or Roche lobes
    * rotating stars
    * triangular meshes

  Need to install for Python.h header:
    apt-get install python-dev


  Testing versions of python;
    PYTHON=python2 make
    PYTHON=python3 make

  Ref:

  Python C-api:
  * https://docs.python.org/2/c-api/index.html

  Numpi C-api manual:
  * http://docs.scipy.org/doc/numpy/reference/c-api.html
  * http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
  * http://scipy.github.io/old-wiki/pages/Cookbook/C_Extensions/NumPy_arrays

  Wrapping tutorial:
  * http://intermediate-and-advanced-software-carpentry.readthedocs.io/en/latest/c++-wrapping.html

  Porting to python 3:
  * https://docs.python.org/3/howto/cporting.html

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
#include "redistribution.h"        // Dealing with reflection-redistribution problem
#include "horizon.h"               // Calculation of horizons

#include "gen_roche.h"             // support for generalized Roche lobes
#include "contact.h"               // support for contact case of Roche lobes
#include "rot_star.h"              // support for rotating stars
#include "misaligned_roche.h"      // support for gen. Roche lobes with missaligned angular momenta

#include "wd_atm.h"                // Wilson-Devinney atmospheres
#include "interpolation.h"         // Nulti-dimensional linear interpolation

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// providing the Python interface -I/usr/include/python2.7/
#include <Python.h>
#include <numpy/arrayobject.h>

struct module_state {
    PyObject *error;
};

// Porting to Python 3
// Ref: http://python3porting.com/cextensions.html
#if PY_MAJOR_VERSION >= 3

  #define MOD_GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

  static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(MOD_GETSTATE(m)->error);
    return 0;
  }

  static int module_clear(PyObject *m){
    Py_CLEAR(MOD_GETSTATE(m)->error);
    return 0;
  }

  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
    static struct PyModuleDef moduledef = { \
      PyModuleDef_HEAD_INIT, \
      name, \
      doc, \
      sizeof(struct module_state), \
      methods,\
      NULL,\
      module_traverse,\
      module_clear,\
      NULL}; \
    ob = PyModule_Create(&moduledef);

  #define MOD_NEW_EXCEPTION(st_error, name)\
    st_error = PyErr_NewException(name, NULL, NULL);

  // adding missing declarations and functions
  #define PyString_Type PyBytes_Type
  #define PyString_AsString PyBytes_AsString
  #define PyString_Check PyBytes_Check
  #define PyInt_FromLong PyLong_FromLong
#else

  #define MOD_GETSTATE(m) (&_state)
  static struct module_state _state;

  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);


  #define MOD_NEW_EXCEPTION(st_error, name)\
    char _name[255];\
    sprintf(_name, "%s", name);\
    st_error = PyErr_NewException(_name, NULL, NULL);
#endif


//#define USING_SimpleNewFromData

/*
  Getting the Python typename
*/

template<typename T> NPY_TYPES PyArray_TypeNum();

template<>  NPY_TYPES PyArray_TypeNum<int>() { return NPY_INT;}
template<>  NPY_TYPES PyArray_TypeNum<double>() { return NPY_DOUBLE;}

/*
  Create string
*/
std::string operator "" _s (const char* m, std::size_t) {
  return std::string(m);
}

/*
 Verbosity of libphoebe.
*/
int verbosity_level = 0;


class TNullBuff : public std::streambuf {
public:
  int overflow(int c) { return c; }
};

TNullBuff null_buffer;

std::ostream report_stream(&null_buffer);

/*
  Report error with or without Python exception
*/
void raise_exception(const std::string & str){
  if (verbosity_level >= 1) report_stream << str << std::endl;
  PyErr_SetString(PyExc_TypeError, str.c_str());
}

/*
  Setting the verbosity of the library to std::cerr using level.

    level = 0: no output
    level = 1: output for python exception
    level = 2: output for python exception and
               additional explanation to exceptions
    level = 3: -- did not decide what this should be --
    level = 4: all possible output -- debug mode

  Input:
    level
*/
static PyObject *setup_verbosity(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "setup_verbosity"_s;

  char *kwlist[] = {
    (char*)"level",
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "i", kwlist, &verbosity_level)){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (verbosity_level == 0)
    report_stream.rdbuf(&null_buffer);
  else {
    report_stream.rdbuf(std::cerr.rdbuf());
    report_stream.precision(16);
    report_stream << std::scientific;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


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
  npy_intp dims[1] = {N};

  #if defined(USING_SimpleNewFromData)
  //T *p = new T [N];
  T *p = (T*) PyObject_Malloc(N*sizeof(T));
  std::copy(V.begin(), V.end(), p);
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, PyArray_TypeNum<T>(), p);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, PyArray_TypeNum<T>());
  std::copy(V.begin(), V.end(), (T*)PyArray_DATA((PyArrayObject *)pya));
  #endif

  return pya;
}

template <typename T>
PyObject *PyArray_FromVector(int N, T *V){

  npy_intp dims[1] = {N};

  #if defined(USING_SimpleNewFromData)
  //T *p = new T [N];
  T *p = (T*) PyObject_Malloc(N*sizeof(T));
  std::copy(V, V + N, p);
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, PyArray_TypeNum<T>(), p);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, PyArray_TypeNum<T>());
  std::copy(V, V + N, (T*)PyArray_DATA((PyArrayObject *)pya));
  #endif

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

  npy_intp dims[2] = {N, 3};

  #if defined(USING_SimpleNewFromData)
  //T *p = new T [3*N];
  T *p = (T*) PyObject_Malloc(3*N*sizeof(T));
  T *b = p;
  for (auto && v : V) for (int i = 0; i < 3; ++i) *(b++) = v[i];
  PyObject *pya = PyArray_SimpleNewFromData(2, dims, PyArray_TypeNum<T>(), p);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(2, dims, PyArray_TypeNum<T>());
  T *p = (T*)PyArray_DATA((PyArrayObject *)pya);
  for (auto && v : V) for (int i = 0; i < 3; ++i) *(p++) = v[i];
  #endif

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

    style: int, default 0

      0 - canonical - conventional:
          L3  -- heavier star -- L1 -- lighter star -- L2 --> x

      1 - native to definition of the potential
          L2  -- origin -- L1 -- object -- L3 --> x

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
    (char*)"style",
    NULL};

  bool b_L[3] = {true, true, true};

  int style = 0;

  double q, F, delta;

  PyObject *o_L[3] = {0,  0, 0};

  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "ddd|O!O!O!i", kwlist,
        &q, &F, &delta,
        &PyBool_Type, o_L,
        &PyBool_Type, o_L + 1,
        &PyBool_Type, o_L + 2,
        &style)
  ){
    raise_exception("roche_critical_potential::Problem reading arguments");
    return NULL;
  }

  int ind[3] = {0, 1, 2};

  if (style == 0 && q > 1) {      // case : M2 > M1
    ind[1] = 2;
    ind[2] = 1;
  }


  // reading selection
  for (int i = 0; i < 3; ++i)
    if (o_L[ind[i]]) b_L[i] = PyObject_IsTrue(o_L[ind[i]]);

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
  const char *labels[] = {"L1", "L2", "L3"};

  for (int i = 0; i < 3; ++i)
    if (b_L[i])
      PyDict_SetItemStringStealRef(results,
        labels[ind[i]],
         PyFloat_FromDouble(omega[i]));

  return results;
}

/*
  C++ wrapper for Python code:

  Calculate rotating parameter and interval theta' of the Roche lobes
  with misaligned spin (S) and orbital angular velocity vectors.

  Spin (S) is a angular velocity vector defined in the rotating reference
  frame attached to the binary system with the origin (o) in the center of
  the primary star and with the basis vectors
    B =(^i, ^j, ^k)

    vec S = |S| B.(sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))

  where
    ^i - in directon from primary star to secondary star
    ^k - in direction of orbital angular angular velocity vector
    ^j = ^k x ^i

  The potential of misaligned Roche lobes is given in reference frame

    o, B' = (^i, ^j', ^k')

  with

    ^j' = ^j cos(xi) + ^k sin(xi)
    ^k' = -^j sin(xi) + ^k cos(xi)

  and in this system

    vec S = |S| B'.(sin(theta'), 0, cos(theta'))

  Python:

    [xi, theta'] = misaligned_transf(type, S)

   with parameters

    type: string
    S : 1-rank numpy array

    if type="cartesian":
      S = np.array([Sx, Sy, Sz])  vec S = Sx*^i + Sy*^j + Sz*^k

    if type = "spherical":
      S = np.array([theta, phi])


  and returns

    [xi, theta']: 1-rank numpy array
      xi - rotation angle
      theta' - new angle between S and new z-axis ^k'
*/

static PyObject *roche_misaligned_transf(PyObject *self, PyObject *args) {

  auto fname = "roche_misaligned_trans"_s;

  PyObject *o_type;
  PyArrayObject *o_S;

  if (!PyArg_ParseTuple(args, "O!O!",
       &PyString_Type, &o_type,
       &PyArray_Type, &o_S)){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double s[3];

  if (PyArray_Check(o_S) && PyArray_TYPE((PyArrayObject *) o_S) == NPY_DOUBLE) {

    double *S = (double*) PyArray_DATA(o_S), t;


    switch (fnv1a_32::hash(PyString_AsString(o_type))) {

      case "cartesian"_hash32:
        t = 1/utils::hypot3(S);
        for (int i = 0; i < 3; ++i) s[i] = t*S[i];
      break;

      case "spherical"_hash32:
        s[0] = std::sin(S[0])*std::cos(S[1]);
        s[1] = std::sin(S[0])*std::sin(S[1]);
        s[2] = std::cos(S[0]);
      break;

      default:
        raise_exception(fname + "::This type is not supported");
        return NULL;
    }
  } else {
    raise_exception(fname + "::This type of misalignment is not supported");
    return NULL;
  }

  double res[2]={
    std::atan2(-s[1], s[2]),
    std::atan2(s[0], std::sqrt(1 - s[0]*s[0]))
  };

  return PyArray_FromVector(2, res);
}

/*
  C++ wrapper for Python code:

  Calculate critical potential of the rotating star potential, which
  is the minimal potential for which lobe exists.

  Python:

    Omega_crit = rotstar_critical_potential(omega)

  where parameters are

    omega: float - parameter of the potential

  and returns a float

    Omega_critical:  float
*/

static PyObject *rotstar_critical_potential(PyObject *self, PyObject *args) {

  // parse input arguments
  double omega;

  if (!PyArg_ParseTuple(args, "d", &omega)){
    raise_exception("rotstar_critical_potential::Problem reading arguments");
    return NULL;
  }

  if (omega == 0) return NULL; // there is no critical value

  return PyFloat_FromDouble(rot_star::critical_potential(omega));
}

/*
  C++ wrapper for Python code:

  Calculate critical potential of the rotating star with misaignment
  having potential function

    Omega(x,y,z; omega, s) = 1/|r| + 1/2 omega^2 |r - s (r.s)|^2

  with

    r = {x, y, z}
    s = {sx, sy, sz}

  Aligned case is

    s = {0, 0, 1}

  Critical potential is the minimal potential for which lobe exists.

  Python:

    Omega_crit = rotstar_misaligned_critical_potential(omega, misalignemnt)

  where parameters are

    omega: float - parameter of the potential
           Note:
           for comparison to Roche model (a=1) : omega = F sqrt(1+q),
           for independent star of mass M : omega = angular velocity/sqrt(G M)

    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
              s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1

  and returns a float

    Omega_critical:  float
*/

static PyObject *rotstar_misaligned_critical_potential(PyObject *self, PyObject *args) {

  auto fname = "rotstar_misaligned_critical_potential"_s;

  // parse input arguments
  double omega;

  PyObject *o_misalignment;

  if (!PyArg_ParseTuple(args, "dO",
      &omega,
      &o_misalignment)
  ){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (omega == 0) return NULL; // there is no critical value

  // Critical potential is independent of the spin orientation and is not read!

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
    raise_exception("roche_pole::Problem reading arguments");
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

  Calculate pole height of the primary star in the misaligned Roche potential.

  Python:

    h = roche_misaligned_pole(q, F, d, misalignment, Omega0, sign)

  where parameters are

  positionals:
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment: in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
    or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    Omega: float - value potential

  optional:
    sign: float - sign of the pole in {1, 0, -1}, default 0

  and return
    p: float - height of the pole:
      p_+   for sign = +1
      p_-   for sign = -1
      (p_+ + p_-)/2 for sign = 0

*/


static PyObject *roche_misaligned_pole(
  PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_pole"_s;

  if (verbosity_level>=4)
    report_stream  << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
    (char*)"Omega0",
    (char*)"sign",
    NULL};

  int sign = 0;

  double q, F, delta, Omega0;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "dddOd|i", kwlist,
        &q, &F, &delta, &o_misalignment, &Omega0, &sign)
      ){

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double p, s;

  if (PyFloat_Check(o_misalignment)) {
    s = std::sin(PyFloat_AsDouble(o_misalignment));
  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {
    s = ((double*) PyArray_DATA((PyArrayObject*)o_misalignment))[0];
  } else {
    raise_exception(fname + "::This type of misalignment is not supported.");
    return NULL;
  }

  p = misaligned_roche::poleL_height(Omega0, q, F, delta, s, sign);

  if (p < 0) {
    raise_exception(fname + "::Problems calculating poles.");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  return PyFloat_FromDouble(p);
}


/*
  C++ wrapper for Python code:

  Calculate the minimal value of the Kopal potential permitted in order to have
  compact primary Roche lobe:

      Omega_{min} (x,y,z; q, F, d, misalignment) =
        min {Omega(L1(misaligment)), Omega(L2(misalignment)) }

  Python:

    Omega_{min} = misaligned_Omega_min(q, F, delta, misalignment)

   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1

  and returns a float

    Omega0 - value of the Omega at (x,y,z)
*/

static PyObject *roche_Omega_min(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_Omega_min"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    NULL};

  double q, F, d;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "ddd", kwlist,
        &q, &F, &d)
      ){

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double omega[2], L[2];

  gen_roche::critical_potential(omega, L, 3, q, F, d);

  return PyFloat_FromDouble(std::min(omega[0], omega[1]));
}

/*
  C++ wrapper for Python code:

  Calculate the minimal value of the Kopal potential permitted in order to have
  compact primary Roche lobe:

      Omega_{min} (q, F, d, misalignment) =
        min {Omega(L1(q,F,d,misaligment)), Omega(L2(q,F,d,misalignment)) }

  Python:

    Omega_{min} = roche_misaligned_Omega_min(q, F, delta, misalignment)

   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1

  and returns a float

    Omega0 - value of the Omega at (x,y,z)
*/

static PyObject *roche_misaligned_Omega_min(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_Omega_min"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
    NULL};

  double q, F, d;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "dddO", kwlist,
        &q, &F, &d, &o_misalignment)
      ){

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double Omega_min;

  if (PyFloat_Check(o_misalignment)) {
    double theta = PyFloat_AsDouble(o_misalignment);
    Omega_min = misaligned_roche::calc_Omega_min(q, F, d, theta);
  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {
    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    Omega_min = misaligned_roche::calc_Omega_min(q, F, d, std::asin(s[0]));
  } else {
    raise_exception(fname + "::This type of misalignment is not supported");
    return NULL;
  }

  if (std::isnan(Omega_min)) {
    raise_exception(fname + "::Calculation of Omega_min failed.");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  return PyFloat_FromDouble(Omega_min);
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
    raise_exception("rotstar_pole::Problem reading arguments");
    return NULL;
  }

  return PyFloat_FromDouble(1/Omega0);
}

/*
  C++ wrapper for Python code:

  Calculate height h the rotating star with misalignment in the
  direction of the pole

    h*spin

  The lobe of the rotating star is defined as equipotential
  of the potential Omega:

      Omega_0 = Omega(x,y,z; omega, s)
              = 1/r + 1/2 omega^2 | r - r(s.r)|^2

  with
      r = {x, y, z}
      s = {sx, sy, sz}

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    h = rotstar_misaligned_pole(omega, misalignment, Omega0, <keywords> = <value>)

  where parameters are

  positionals:
    omega: float - parameter of the potential
           Note:
           for comparison to Roche model (a=1): omega = F sqrt(1+q),
           for independent star of mass M : omega = angular velocity/sqrt(G M)
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
      s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    Omega: float - value potential

  and return float

    h : height of the lobe's pole
*/

static PyObject *rotstar_misaligned_pole(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_misaligned_pole"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"omega",
    (char*)"misalignment",
    (char*)"Omega0",
    NULL};

  double omega, Omega0;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dOd", kwlist,
      &omega,
      &o_misalignment,
      &Omega0)
  ){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  // Pole height is independent of the spin orientation and is not read!

  return PyFloat_FromDouble(1/Omega0);
}

/*
  C++ wrapper (trivial) for Python code:

  Calculate height h of sphere's pole defined

    Omega0 = Omega(x, y, z) = 1/sqrt(x^2 + y^2 + z^2)

  Python:

    h = sphere_pole(Omega0)

  where parameters are

  positionals:

    Omega0: float

  and return float

    h : height of the lobe's pole = R
*/

static PyObject *sphere_pole(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "sphere_pole"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"Omega0",
    NULL};

  double Omega0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "d", kwlist, &Omega0)){
    raise_exception(fname + "::Problem reading arguments");
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
        2 - contact binary -- not permitted

  and returns vector of parameters float

    param_rotstar : 1-rank numpy array = (omega_rotstar, Omega0_rotstar)
*/

static PyObject *rotstar_from_roche(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_from_roche"_s;
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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice != 0) {
    raise_exception(fname + "::Choice != 0 is not yet supported");
    return NULL;
  }

  double data[2] = {
    F*std::sqrt(1 + q),
    1/gen_roche::poleL(Omega0, q, F, delta)
  };

  if (utils::sqr(data[0])/utils::cube(data[1])> 8./27) {
    raise_exception(fname + "::The lobe does not exist.");
    return NULL;
  }

  return PyArray_FromVector(2, data);
}


/*
  C++ wrapper for Python code:

  Calculate parameters of the rotating star from Roche binary model with
  misalignment by matching the poles and centrifugal force.

  Python:

   param_rotstar
      = rotstar_misaligned_from_roche_misaligned(q, F, d, misalignment, Omega0, <keywords> = <value>)

  where parameters are

    positionals:

      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      delta: float - separation between the two objects
      misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
                s = [sin(angle), 0, cos(angle)]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
      Omega0: float - value potential

    keywords:

      choice: default 0
        0 - primary star
        1 - secondary star -- not yet supported
        2 - contact binary -- not permitted

  Returns: for rotstar dictionary with keywords:

    "omega": angular frequency of rotating star
      float

    "misalignment": direction of rotation
      1-rank numpy array

    "Omega": value of the potential
      float

*/

static PyObject *rotstar_misaligned_from_roche_misaligned(
  PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_misaligned_from_roche_misaligned"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"delta",
    (char*)"misalignment",
    (char*)"Omega0",
    (char*)"choice",
    NULL};

  int choice = 0;

  double q, F, delta, Omega0;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dddOd|i", kwlist,
      &q,
      &F,
      &delta,
      &o_misalignment,
      &Omega0,
      &choice)
  ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice != 0) {
    raise_exception(fname + "::Choice != 0 is not yet supported");
    return NULL;
  }

  double spin[3] = {0, 0, 1.0};

  if (PyFloat_Check(o_misalignment)) {
    double s = std::sin(PyFloat_AsDouble(o_misalignment));

    spin[0] = s;
    spin[1] = 0;
    spin[2] = std::sqrt(1. - s*s);

  } else if (PyArray_Check(o_misalignment)) {

    double *s = (double*)PyArray_DATA((PyArrayObject*)o_misalignment);
    for (int i = 0; i < 3; ++i) spin[i] = s[i];

  } else {
    raise_exception(fname + ":: This type of misalignment if not supported");
    return NULL;
  }

  double
    r_omega = F*std::sqrt(1 + q),
    r_Omega = 1/misaligned_roche::poleL_height(Omega0, q, F, delta, spin, 0);


  if (utils::sqr(r_omega)/utils::cube(r_Omega)> 8./27) {
    raise_exception(fname + "::The lobe does not exist.");
    return NULL;
  }

  PyObject *results = PyDict_New();

  PyDict_SetItemStringStealRef(results, "omega", PyFloat_FromDouble(r_omega));
  PyDict_SetItemStringStealRef(results, "misalignment", PyArray_FromVector(3, spin));
  PyDict_SetItemStringStealRef(results, "Omega", PyFloat_FromDouble(r_Omega));

  return results;
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
            2 for discussing contact envelope

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


static PyObject *roche_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_area_volume"_s;

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

    raise_exception(fname + "::Problem reading arguments");
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

  if (verbosity_level>=4)
    report_stream
      << "q=" << q
      << " F=" << F
      << " d=" << delta
      << " Omega0=" << Omega0  << '\n';


  //
  // Posibility of using approximation
  //

  double
    b = (1 + q)*F*F*delta*delta*delta,
    w0 = 5*(q + std::cbrt(b*b)/4) - 29.554 - 5.26235*std::log(std::min(eps[0], eps[1])),
    av[2];                          // for results

  if (choice == 0 && delta*Omega0 >= std::max(10., w0)) {

    // Approximation by using the series
    // with empirically derived criterion

    gen_roche::area_volume_primary_asymp(av, res_choice, Omega0, q, F, delta);

    if (verbosity_level>=4)
      report_stream << "asymp:" << res_choice<<" " << av[0] << " " << av[1] << '\n';

  } else {

    // Approximation by integration over the surface
    // relative precision should be better than 1e-12

    //
    // Choosing boundaries on x-axis
    //

    double xrange[2];

    if (!gen_roche::lobe_xrange(xrange, choice, Omega0, q, F, delta, true)){
      raise_exception(fname + "Determining lobe's boundaries failed");
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

    //
    // one step adjustment of precison for area and volume
    // calculation
    //

    do {

      for (int i = 0, m = m0; i < 2; ++i, m <<= 1) {
        gen_roche::area_volume_integration
          (p[i], res_choice, xrange, Omega0, q, F, delta, m, polish);

        if (verbosity_level>=4)
          report_stream << "P:" << p[i][0] << '\t' << p[i][1] << '\n';
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

          if (verbosity_level>=4)
            report_stream << "err=" << e << " m0=" << m0 << '\n';


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
          if (b_av[i]) {
            av[i] = (16*p[1][i] - p[0][i])/15;

            if (verbosity_level>=4)
              report_stream << "B:" << i << ":" << av[i] << '\n';
          }
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


/*
  C++ wrapper for Python code:

  Calculate area and volume of rotating star lobe is defined as
  equipotential of the potential Omega:

      Omega_0 = Omega(x,y,z) = 1/r  + 1/2 omega^2  (x^2 + y^2)

  Python:

    dict = rotstar_area_volume(omega, Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    omega: float - parameter of the potential
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
      s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
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

  auto fname = "rotstar_area_volume"_s;

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
    raise_exception(fname + "::Problem reading arguments");
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

  switch (rot_star::area_volume(av, res_choice, Omega0, omega)) {
    case -1:
    raise_exception(fname + "::No calculations are requested");
    return NULL;

    case 1:
    raise_exception(fname +
      "::The lobe does not exist. t is not in [0,1]\n" +
      "Omega0=" + std::to_string(Omega0) +
      " omega=" + std::to_string(omega) +
      " t=" + std::to_string(27*omega*omega/(Omega0*Omega0*Omega0)/8)
      );
    return NULL;
  }

  PyObject *results = PyDict_New();

  if (b_larea)
    PyDict_SetItemStringStealRef(results, "larea", PyFloat_FromDouble(av[0]));

  if (b_lvolume)
    PyDict_SetItemStringStealRef(results, "lvolume", PyFloat_FromDouble(av[1]));

  return results;
}

/*
  C++ wrapper for Python code:

  Calculate area and volume of lobe of the rotating star with misalignment
  is defined as equipotential of the potential Omega:

    Omega_0 = Omega(x,y,z; omega, s)
            = 1/|r|  + 1/2 omega^2 | r - r (r.s)|^2
  with
    r = {x, y, z}
    s = {sx, sy, sz}

  Aligned case is
    s = {0, 0, 1}

  Python:

    dict = rotstar_misaligned_area_volume(omega, misalignment, Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    omega: float - parameter of the potential
      Note: for comparison to roche : omega = F sqrt(1+q),
          for independent star of mass M : omega = angular velocity/sqrt(G M)
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
              s = [sin(angle), 0, cos(angle)]
      or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
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

static PyObject *rotstar_misaligned_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_misaligned_area_volume"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"omega",
    (char*)"misalignment",
    (char*)"Omega0",
    (char*)"larea",
    (char*)"lvolume",
    NULL};

  bool
    b_larea = true,
    b_lvolume = true;

  PyObject
    *o_misalignment,
    *o_larea = 0,
    *o_lvolume = 0;

  double omega, Omega0;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "dOd|O!O!", kwlist,
        &omega,
        &o_misalignment,
        &Omega0,
        &PyBool_Type, &o_larea,
        &PyBool_Type, &o_lvolume)
  ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (o_larea) b_larea = PyObject_IsTrue(o_larea);
  if (o_lvolume) b_lvolume = PyObject_IsTrue(o_lvolume);

  if (!b_larea && !b_lvolume) return NULL;


  // Volume and area is independent of the spin orientation and is not read!

  //
  // Calculate area and volume
  //

  unsigned res_choice = 0;

  if (b_larea) res_choice |= 1u;
  if (b_lvolume) res_choice |= 2u;

  double av[2] = {0,0};

  switch (rot_star::area_volume(av, res_choice, Omega0, omega)) {
    case -1:
    raise_exception(fname + "::No calculations are requested");
    return NULL;

    case 1:
    raise_exception(fname +
      "::The lobe does not exist. t is not in [0,1]\n" +
      "Omega0=" + std::to_string(Omega0) +
      " omega=" + std::to_string(omega) +
      " t=" + std::to_string(27*omega*omega/(Omega0*Omega0*Omega0)/8)
      );
    return NULL;
  }

  PyObject *results = PyDict_New();

  if (b_larea)
    PyDict_SetItemStringStealRef(results, "larea", PyFloat_FromDouble(av[0]));

  if (b_lvolume)
    PyDict_SetItemStringStealRef(results, "lvolume", PyFloat_FromDouble(av[1]));

  return results;
}


/*
  C++ wrapper (trivial) for Python code:

  Calculate area and volume of sphere of radious R:

    Omega0 = 1/R

  Python:

    dict = sphere_area_volume(Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    Omega0: 1/radius

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

static PyObject *sphere_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "sphere_area_volume"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
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

  double Omega0;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "d|O!O!", kwlist,
        &Omega0,
        &PyBool_Type, &o_larea,
        &PyBool_Type, &o_lvolume
      )
     ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (o_larea) b_larea = PyObject_IsTrue(o_larea);
  if (o_lvolume) b_lvolume = PyObject_IsTrue(o_lvolume);

  if (!b_larea && !b_lvolume) return NULL;

  //
  // Calculate area and volume
  //

  PyObject *results = PyDict_New();

  double R = 1/Omega0, R2 = R*R;

  if (b_larea)
    PyDict_SetItemStringStealRef(
      results,
      "larea",
      PyFloat_FromDouble(utils::m_4pi*R2)
    );

  if (b_lvolume)
    PyDict_SetItemStringStealRef(
      results,
      "lvolume",
      PyFloat_FromDouble(utils::m_4pi3*R2*R)
    );

  return results;
}

/*
  C++ wrapper for Python code:

  Calculate the volume of the semi-detached Roche lobe with
  misaligned spin and orbit velocity vectors.

  Python:

    critical_volume = roche_misaligned_critical_volume(q, F, d, misalignment)

  where parameters are

  positionals:
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment: in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1

  Returns:

    critical_volume: float
*/


static PyObject *roche_misaligned_critical_volume(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_critical_volume"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
    NULL};

  double q, F, d;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "dddO", kwlist,
        &q, &F, &d, &o_misalignment)
      ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << "q:" << q << " F=" << F << " d=" << d << '\n';

  double theta;

  if (PyFloat_Check(o_misalignment)) {

    theta = std::abs(PyFloat_AsDouble(o_misalignment)); // in [0, pi/2]

    if (verbosity_level>=4)
      report_stream << fname + "::theta:" << theta << '\n';

  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    double *s = (double*)PyArray_DATA((PyArrayObject *)o_misalignment);

    if (verbosity_level>=4)
      report_stream << fname + "::spin:" << s[0] << ' ' << s[1] << ' ' << s[2] << '\n';

    if (s[0] == 0) {
      theta = 0;
    } else {
      theta = std::asin(std::abs(s[0])); // in [0, pi/2]
    }
  } else {
    raise_exception(fname + ":: This type of misalignment if not supported");
    return NULL;
  }

  //
  // Calculate critical volume
  //
  double OmegaC, buf[3];

  if (!misaligned_roche::critical_area_volume(2, q, F, d, theta, OmegaC, buf)){
    raise_exception(fname + "::Calculation of critical volume failed");
    return NULL;
  }

  return PyFloat_FromDouble(buf[1]);
}

/*
  C++ wrapper for Python code:

  Calculate area and volume of the generalied Roche lobes with
  misaligned spin and orbit velocity vectors.

  Omega_0 = Omega(x,y,z)

  Python:

    dict = roche_misaligned_area_volume(q, F, d, misalignment, Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment: in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    Omega: float - value potential

  keywords:
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe
            2 for discussing contact

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


static PyObject *roche_misaligned_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_area_volume"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
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

  double q, F, d, Omega0;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "dddOd|iO!O!dd", kwlist,
        &q, &F, &d, &o_misalignment, &Omega0,
        &choice,
        &PyBool_Type, o_av,
        &PyBool_Type, o_av + 1,
        eps, eps + 1)
      ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice != 0) {
    raise_exception(fname + "::choice != 0 is currently not supported");
    return NULL;
  }

  if (verbosity_level>=4)
      report_stream << fname + "::q:" << q << " F=" << F << " Omega=" << Omega0 << " d=" << d << '\n';

  bool aligned = false;

  double theta;

  if (PyFloat_Check(o_misalignment)) {

    theta = std::abs(PyFloat_AsDouble(o_misalignment)); // in [0, pi/2]

    aligned = (std::sin(theta) == 0); // theta ~0 => aligned

    if (verbosity_level>=4)
      report_stream << fname + "::theta:" << theta << '\n';


  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    double *s = (double*)PyArray_DATA((PyArrayObject *)o_misalignment);

    if (verbosity_level>=4)
      report_stream << fname << "::spin:" << s[0] << ' ' << s[1] << ' ' << s[2] << '\n';

    if (s[0] == 0) {
      aligned = true;
      theta = 0;
    } else {
      aligned = false;
      theta = std::asin(std::abs(s[0])); // in [0, pi/2]
    }

  } else {
    raise_exception(fname + ":: This type of misalignment if not supported");
    return NULL;
  }

  //
  // Read boolean variables and define result-choice
  //

  unsigned res_choice = 0;

  for (int i = 0, j = 1; i < 2; ++i, j <<=1) {
    if (o_av[i]) b_av[i] = PyObject_IsTrue(o_av[i]);
    if (b_av[i]) res_choice += j;
  }

  if (res_choice == 0) return NULL;

  if (verbosity_level>=4)
    report_stream << "res_choice=" << res_choice << '\n';

  //
  // Calculate area and volume:
  //

  double
    av[2],        // storing results
    OmegaC = misaligned_roche::calc_Omega_min(q, F, d, theta),
    dOmegaC = Omega0 - OmegaC,
    eps0 = 1e-12, eps1 = 1e-12;

  if (dOmegaC < -OmegaC*eps0){

    raise_exception(fname + ":: Object is not detached.");

    if (verbosity_level>=2)
      report_stream << fname + "::OmegaC=" << OmegaC << "  Omega0=" << Omega0 << '\n'
      << "q=" << q << " F=" << F << " delta=" << d << " theta=" << theta << '\n';

    return NULL;

  } else if (std::abs(dOmegaC) < OmegaC*eps1) {  // semi-detached case

    if (!misaligned_roche::critical_area_volume(res_choice, q, F, d, theta, OmegaC, av)){
      raise_exception(fname + "::Calculation of critical lobe failed");
      return NULL;
    }

  } else {                                       // detached case

    const int m_min = 1 << 8;  // minimal number of points along x-axis

    int m0 = m_min;            // starting number of points alomg x-axis

    bool adjust = true;

    double p[2][2], xrange[2], pole, e;

    //
    // Choosing boundaries on x-axis or calculating the pole
    //

    if (aligned) {      // Non-misaligned Roche lobes
      if (!gen_roche::lobe_xrange(xrange, choice, Omega0, q, F, d, true)){
        raise_exception(fname + "Determining lobe's boundaries failed");
        return NULL;
      }
    } else {            // mis-aligned Roche lobes
      pole = misaligned_roche::poleL_height(Omega0, q, F, d, std::sin(theta));
      if (pole < 0) {
        raise_exception(fname + "Determining pole failed");
        return NULL;
      }
    }
    //
    // one step adjustment of precison for area and volume
    // calculation
    //

    do {

      for (int i = 0, m = m0; i < 2; ++i, m <<= 1)
        if (theta == 0)
          gen_roche::area_volume_integration
            (p[i], res_choice, xrange, Omega0, q, F, d, m);
        else {
          misaligned_roche::area_volume_integration
            (p[i], res_choice, pole, Omega0, q, F, d, theta, m);

          if (verbosity_level>=4)
            report_stream << fname << "::m=" << m << " p[" << i  << "]=" << p[i][0] << ' ' << p[i][1] << '\n';

        }

      // best approximation
      for (int i = 0; i < 2; ++i)
        if (b_av[i]) av[i] = (16*p[1][i] - p[0][i])/15;

      if (adjust) {

        // extrapolation based on assumption
        //   I = I0 + a_1 h^4
        // estimating errors

        int m0_next = m0;

        adjust = false;

        for (int i = 0; i < 2; ++i) if (b_av[i]) {

          // relative error
          e = std::max(std::abs(p[0][i]/av[i] - 1), 16*std::abs(p[1][i]/av[i] - 1));

          if (e > eps[i]) {
            int k = int(1.1*m0*std::pow(e/eps[i], 0.25));
            if (k > m0_next) {
              m0_next = k;
              adjust = true;
            }
          }
        }

        if (adjust) m0 = m0_next;
      }
    } while (adjust);
  }

  PyObject *results = PyDict_New();

  const char *str[2] =  {"larea", "lvolume"};

  for (int i = 0; i < 2; ++i) if (b_av[i]) {

    if (verbosity_level>=4)
      report_stream << fname + "::av[" << i << "]=" << av[i] << '\n';

    PyDict_SetItemStringStealRef(results, str[i], PyFloat_FromDouble(av[i]));
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

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

  keywords: (optional)
    Omega0: float - guess for value potential Omega1
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe
            2 for discussing contact envelope
    precision: float, default 1e-12
      aka relative precision
    accuracy: float, default 1e-12
      aka absolute precision
    max_iter: integer, default 100
      maximal number of iterations in the Newton-Raphson

  Returns:

    Omega1 : float
      value of the Kopal potential for (q,F,d) at which the lobe has the given volume
*/


static PyObject *roche_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_Omega_at_vol"_s;

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
      args, keywds,  "dddd|diddi", kwlist,
      &vol, &q, &F, &delta, &Omega0,
      &choice,
      &precision,
      &accuracy,
      &max_iter
      )
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (std::isnan(Omega0)) {
    // equivalent radius
    double  r = std::cbrt(0.75*vol/utils::m_pi);

    if (verbosity_level>=4)
      report_stream << fname + "::r=" << r << '\n';

  /* Omega[x_, y_, z_, {q_, F_, d_, theta_}] = 1/Sqrt[x^2 + y^2 + z^2] +
      q (-(x/d^2) + 1/Sqrt[(d - x)^2 + y^2 + z^2]) +
      1/2 F^2 (1 + q) (y^2 + (x Cos[theta] - z Sin[theta])^2)
  */

    // = Omega[r,0,0]
    Omega0 =
      1/r  +
      q*(1/std::abs(r - delta) - (r/delta)/delta) +
      0.5*(1 + q)*utils::sqr(F*r);
  }

  const int m_min = 1 << 6;  // minimal number of points along x-axis

  int
    m0 = m_min,  // minimal number of points along x-axis
    it = 0;      // number of iterations

  double
    Omega = Omega0, dOmega,
    V[2], xrange[2], p[2][2];

  // expected precisions of the integrals
  double eps = precision/2;

  do {

    if (!gen_roche::lobe_xrange(xrange, choice, Omega, q, F, delta, true)){
      raise_exception(fname + "::Determining lobe's boundaries failed");
      return NULL;
    }

    // adaptive calculation of the volume and its derivative
    bool adjust = true;

    do {

      if (verbosity_level>=4)
        report_stream << fname + "::it=" << it << '\n';


      // calculate volume and derivate volume w.r.t to Omega
      for (int i = 0, m = m0; i < 2; ++i, m <<= 1) {
        gen_roche::area_volume_integration(p[i] - 1, 6, xrange, Omega, q, F, delta, m);

        if (verbosity_level>=4)
          report_stream << fname + "::V:" <<  p[i][0] << '\t' << p[i][1] << '\n';
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

          if (verbosity_level>=4)
            report_stream << fname + "::e=" << e << " m0 =" << m0 << '\n';

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

    if (verbosity_level>=4)
      report_stream
        << fname + ":: Omega=" << Omega << " vol=" << vol
        << " V[0]= " << V[0] << " dOmega=" << dOmega << '\n';

  } while (std::abs(dOmega) > accuracy + precision*Omega && ++it < max_iter);

  if (!(it < max_iter)){
    raise_exception(fname + "::Maximum number of iterations exceeded");
    return NULL;
  }
  // We use the condition on the argument (= Omega) ~ constraining backward error,
  // but we could also use condition on the value (= Volume) ~ constraing forward error

  return PyFloat_FromDouble(Omega);
}


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

  Returns:

    Omega1 : float
    value of the Kopal potential at omega and lobe volume
*/


static PyObject *rotstar_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_Omega_at_vol"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"vol",
    (char*)"omega",
    NULL};

  double vol, omega;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dd", kwlist,
      &vol, &omega)
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double Omega = rot_star::Omega_at_vol(vol, omega);

  if (std::isnan(Omega)){
    raise_exception(fname + "::Problem determining Omega. See cerr outputs.");
    return NULL;
  }

  return PyFloat_FromDouble(Omega);
}


/*
  C++ wrapper for Python code:

  Calculate the value of potential Omega1 of a rotating star with
  misalignment at parameter omega, spin s and star's volume equal to vol.

  The  rotating star is defined as equipotential of the generalized
  Kopal potential Omega:

    Omega1 == Omega(x,y,z; omega, s)
           = 1/|r| + 1/2 omega^2 | r - s*(r.s)|^2

  with
    r = {x, y, z}
    s = {sx, sy, sz}

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    Omega1 = rotstar_misaligned_Omega_at_vol(vol, omega, misalignment, Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    vol: float - volume of the star's lobe
    omega: float  - parameter of the potential
    Note: for comparison to roche : omega = F sqrt(1+q),
          for independent star of mass M : omega = angular velocity/sqrt(G M)

  keywords: (optional)
   misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
      s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    Note:
      misaligned does not affect the volume and therefore is ignored
  Returns:

    Omega1 : float
      value of the Kopal potential at omega and lobe volume
*/


static PyObject *rotstar_misaligned_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_misaligned_Omega_at_vol"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"vol",
    (char*)"omega",
    (char*)"misalignment",
    NULL};

  double vol, omega;

  PyObject *o_misalignment;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "dd|O", kwlist,
        &vol,
        &omega,
        &o_misalignment)
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double Omega = rot_star::Omega_at_vol(vol, omega);

  if (std::isnan(Omega)){
    raise_exception(fname + "::Problem determining Omega. See cerr outputs.");
    return NULL;
  }

  return PyFloat_FromDouble(Omega);
}



/*
  C++ wrapper for Python code:

  Calculate the value of the generalized Kopal potential Omega1 corresponding
  to parameters (q,F,d,theta) and the volume of the Roche lobes, with misaligned
  spin and orbital angular velocity vectors, equals to vol.

  The Roche lobe(s) is defined as equipotential of the generalized
  Kopal potential Omega:

      Omega_i = Omega(x,y,z; q, F, d, misalignment)

  Python:

    Omega1 = roche_misaligned_Omega_at_vol(vol, q, F, d, misalignment, Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    vol: float - volume of the Roche lobe
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
    or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1

  keywords: (optional)
    Omega0: float - guess for value potential Omega1
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe
            2 for discussing contact
    precision: float, default 1e-12
      aka relative precision
    accuracy: float, default 1e-12
      aka absolute precision
    max_iter: integer, default 100
      maximal number of iterations in the Newton-Raphson

  Returns:

    Omega1 : float
      value of the Kopal potential for (q,F,d1,spin) at which the lobe has the given volume
*/


static PyObject *roche_misaligned_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_Omega_at_vol"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"vol",
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
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
    q, F, d,
    Omega0 = nan(""),
    precision = 1e-12,
    accuracy = 1e-12;

  PyObject *o_misalignment;

  int max_iter = 10;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "ddddO|diddi", kwlist,
        &vol, &q, &F, &d, &o_misalignment, &Omega0,
        &choice,
        &precision,
        &accuracy,
        &max_iter
      )
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice != 0) {
    raise_exception(fname + "::choice != 0 is currently not supported");
    return NULL;
  }

  bool aligned = false;

  double theta = 0;

  if (PyFloat_Check(o_misalignment)) {

    theta = PyFloat_AsDouble(o_misalignment);
    aligned = (std::sin(theta) == 0); // theta ~0, pi => aligned

  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    double *s = (double*)PyArray_DATA((PyArrayObject *)o_misalignment);

    if (verbosity_level>=4)
      report_stream <<  fname << "::spin:" << s[0] << ' ' << s[1] << ' ' << s[2] << '\n';

    aligned = (s[0] == 0);
    theta = std::asin(s[0]);

  } else {
    raise_exception(fname + ":: This type of misalignment if not supported");
    return NULL;
  }

  //
  //  Check if the volume if larger than critical
  //

  double OmegaC, buf[3], volC[2];

  if (verbosity_level>=4)
    report_stream << fname << "::calculate critical volume ...\n";

  if (aligned)
    gen_roche::critical_area_volume(6, q, F, d, OmegaC, buf);
  else if (!misaligned_roche::critical_area_volume(6, q, F, d, theta, OmegaC, buf)) {
    raise_exception(fname + ":: Calculation of critical_volume failed");
  }

  volC[0] = buf[1];
  volC[1] = buf[2];

  if (verbosity_level>=4)
    report_stream << fname
      << "::OmegaC=" << OmegaC << " volC=" << volC[0] << ":" << volC[1]
      << " vol=" << vol << " aligned=" << aligned << '\n';

  if (std::abs(vol - volC[0]) <  precision*volC[0]){

    if (verbosity_level>=4)
      report_stream
        << fname + "::Potential identical to L1\n"
        << fname + "::END" << std::endl;

    return PyFloat_FromDouble(OmegaC);    // Omega at L1 point

  } else if (vol > volC[0]){
    raise_exception(fname + "::The volume is beyond critical");

    if (verbosity_level >=2)
      report_stream
        << fname + "::OmegaC=" << OmegaC << " volC=" << volC[0] << " dvolC/dOmega=" << volC[1] << '\n'
        << fname + "::vol=" << vol << " q=" << q << " F=" << F << " d=" << d << " theta=" << theta << '\n';

    return NULL;
  }

  // Omega very near to critical
  double dOmega1 = (vol - volC[0])/volC[1];

  if (std::abs(dOmega1) < precision*OmegaC){

    if (verbosity_level>=4)
      report_stream
        << fname + "::Potential is very near to critical\n"
        << fname + "::END" << std::endl;

    return PyFloat_FromDouble(OmegaC + dOmega1);
  }
  //
  // If Omega0 is not set, we estimate it
  //

  if (std::isnan(Omega0)) {
    // equivalent radius
    double  r = std::cbrt(0.75*vol/utils::m_pi);

    if (verbosity_level>=4)
      report_stream << fname << "::r=" << r << '\n';

    // = Omega[r,0,0]
    Omega0 =
      1/r  +
      q*(1/std::abs(r - d) - (r/d)/d) +
      0.5*(1 + q)*utils::sqr(F*r*std::cos(theta));

    // Newton-Raphson step
    if (Omega0 < OmegaC) Omega0 = OmegaC + (vol - volC[0])/volC[1];
  }

  //
  // Checking estimate of the Omega0
  //
  if (Omega0 < OmegaC) {
    raise_exception(fname + "::The estimated Omega is beyond critical.");
    return NULL;
  }

  if (verbosity_level>=4)
      report_stream
        << fname + "::vol=" << vol << " q=" << q <<  " F=" << F << " Omega0=" << Omega0
        << " d=" << d << " theta=" << theta << " choice=" << choice << std::endl;

  //
  // Trying to calculate Omega at given volume
  //
  const int m_min = 1 << 8;  // minimal number of points along x-axis

  int
    m0 = m_min,  // minimal number of points along x-axis
    it = 0;      // number of iterations

  double
    Omega = Omega0, dOmega,
    V[2], xrange[2], p[2][2],
    pole = -1;


  // expected precisions of the integrals
  double eps = precision/2;

  // adaptive calculation of the volume and its derivative,
  // permitting adjustment just once as it not necessary stable
  bool adjust = true;

  do {

    if (aligned) {      // Non-misaligned Roche lobes
      if (!gen_roche::lobe_xrange(xrange, choice, Omega, q, F, d, true)){
        raise_exception(fname + "::Determining lobe's boundaries failed");
        return NULL;
      }
    } else {
      pole = misaligned_roche::poleL_height(Omega, q, F, d, std::sin(theta));
      if (pole < 0) {
        raise_exception(fname + "Determining pole failed");
        return NULL;
      }

      if (verbosity_level>=4)
        report_stream << fname + "::pole=" << pole << '\n';
    }

    do {

      // calculate volume and derivate volume w.r.t to Omega
      for (int i = 0, m = m0; i < 2; ++i, m <<= 1)
        if (aligned)
          gen_roche::area_volume_integration(p[i]-1, 6, xrange, Omega, q, F, d, m);
        else
          misaligned_roche::area_volume_integration(p[i]-1, 6, pole, Omega, q, F, d, theta, m);


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

          if (e > eps) {
            int k = int(1.1*m0*std::pow(e/eps, 0.25));
            if (k > m0_next) {
              m0_next = k;
              adjust = true;
            }
          }

          if (verbosity_level>=4)
            report_stream << fname
              << "::m=" <<  m0 << " m0_next=" << m0_next
              << " V[" << i << "]=" << V[i] << " e =" << e << '\n';

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

    // Newton-Raphson iteration step
    Omega -= (dOmega = (V[0] - vol)/V[1]);

    // correction if the values are smaller than critical
    if (Omega < OmegaC) Omega = OmegaC - (dOmega = (volC[0] - vol)/volC[1]);

    if (verbosity_level>=4)
      report_stream << fname + "::Omega=" << Omega  << " dOmega=" << dOmega << '\n';

  } while (std::abs(dOmega) > accuracy + precision*Omega && ++it < max_iter);

  if (!(it < max_iter)){
    raise_exception(fname + "::Maximum number of iterations exceeded");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname + "::final:Omega=" << Omega  << " dOmega=" << dOmega << '\n';


  // We use the condition on the argument (= Omega) ~ constraining backward error,
  // but we could also use condition on the value (= Volume) ~ constraing forward error
  if (verbosity_level>=4)
      report_stream << fname << "::END" << std::endl;

  return PyFloat_FromDouble(Omega);
}


/*
  C++ wrapper for Python code:

  Calculate the value of potential Omega1 of spherical star with
  volume equal to vol. The star has the Kopal potential:

    Omega(x,y,z) = 1/|r|

  with
    r = {x, y, z}

  Note: Misalignment does not make sense in spherical stars.

  Python:

    Omega1 = sphere_Omega_at_vol(vol)

  where parameters are

  positionals:
    vol: float - volume of the star's lobe, vol = 4 Pi/3 r^3

  Returns:

    Omega1 : float
      value of the Kopal potential at volume vol
*/


static PyObject *sphere_Omega_at_vol(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "sphere_Omega_at_vol"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"vol",
    NULL};

  double vol;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "d", kwlist,
        &vol)
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double r = std::cbrt(0.75*vol/(utils::m_pi));   // radius = 3 V/(4 Pi)

  return PyFloat_FromDouble(1/r);
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
    raise_exception("roche_gradOmega::Problem reading arguments");
    return NULL;
  }
  p[3] = 0;

  Tgen_roche<double> b(p);
  npy_intp dims[1] = {4};

  #if defined(USING_SimpleNewFromData)
  double *g = new double [4];
  b.grad((double*)PyArray_DATA(X), g);
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  b.grad((double*)PyArray_DATA(X), (double*)PyArray_DATA((PyArrayObject *)pya));
  #endif

  return pya;
}

/*
  C++ wrapper for Python code:

  Calculate the gradient and the value of the rotenting star potential
  at a given point

      -grad Omega (x,y,z)

  which is outwards the lobe.


  Python:

    g = rotstar_gradOmega(omega, r)

  with parameters

    omega: float - parameter of the potential
    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array
      = [-grad Omega_x, -grad Omega_y, -grad Omega_z, -Omega(x,y,z)]
*/


static PyObject *rotstar_gradOmega(PyObject *self, PyObject *args) {

  auto fname = "rotstar_gradOmega"_s;

  double p[2];

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dO!", p, &PyArray_Type, &X)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  p[1] = 0;

  Trot_star<double> b(p);

  npy_intp dims[1] = {4};

  #if defined(USING_SimpleNewFromData)
  double *g = new double [4];
  b.grad((double*)PyArray_DATA(X), g);
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  b.grad((double*)PyArray_DATA(X), (double*)PyArray_DATA((PyArrayObject *)pya));
  #endif

  return pya;
}

/*
  C++ wrapper for Python code:

  Calculate the gradient and the value of the rotating star potential
  with misalignment at a given point

      -grad Omega (x,y,z; omega, s)

  where

      Omega(x,y,z; omega, s)
            = 1/|r|  + 1/2 omega^2 | r - r (r.s)|^2

  with
      r = {x, y, z}
      s = {sx, sy, sz}    |s| = 1

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    g = rotstar_misaligned_gradOmega(omega, misalignment, r)

  with parameters

    omega: float - parameter of the potential
          Note:
          for comparison to Roche model (a=1) : omega = F sqrt(1+q),
          for independent star of mass M : omega = angular velocity/sqrt(G M)
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
      s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array
      = [-grad Omega_x, -grad Omega_y, -grad Omega_z, -Omega(x,y,z)]
*/


static PyObject *rotstar_misaligned_gradOmega(PyObject *self, PyObject *args) {

  auto fname = "rotstar_misaligned_gradOmega"_s;

  double p[5];

  PyObject *o_misalignment;

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dOO!",
      p,
      &o_misalignment,
      &PyArray_Type, &X)
  ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  p[4] = 0;

  if (PyFloat_Check(o_misalignment)) {
    double s = std::sin(PyFloat_AsDouble(o_misalignment));

    p[1] = s;
    p[2] = 0;
    p[3] = std::sqrt(1. - s*s);

  } else if (PyArray_Check(o_misalignment)) {
    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);

    for (int i = 0; i < 3; ++i) p[i+1] = s[i];
  }

  Tmisaligned_rot_star<double> b(p);

  double g[4];

  b.grad((double*)PyArray_DATA(X), g);

  return PyArray_FromVector(4, g);
}


/*
  C++ wrapper for Python code:

  Calculate the gradient and the value of the potential spherical object
  at a given point

      -grad Omega (x,y,z) = r/|r|^3 r =[x,y,z]

  which is outwards the lobe.


  Python:

    g = sphere_gradOmega(r)

  with parameters

    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array
      = [-grad Omega_x, -grad Omega_y, -grad Omega_z, -Omega(x,y,z)]
*/


static PyObject *sphere_gradOmega(PyObject *self, PyObject *args) {

  auto fname = "sphere_gradOmega"_s;

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double
    *x = (double*) PyArray_DATA(X),
    R = utils::hypot3(x),
    F = 1/(R*R*R);

  npy_intp dims[1] = {4};

  #if defined(USING_SimpleNewFromData)
  double  *g = new double [4];
  for (int i = 0; i < 3; ++i) g[i] = F*x[i];
  g[3] = -1/R;

  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *g = (double*)PyArray_DATA((PyArrayObject *)pya);
  for (int i = 0; i < 3; ++i) g[i] = F*x[i];
  g[3] = -1/R;
  #endif

  return pya;
}

/*
  C++ wrapper for Python code:

  Calculate the gradient of the potential of the generalized
  Kopal potential Omega with misaligned spin and orbital angular
  velocity vectors at a given point

      -grad Omega (x,y,z)

  Python:

    g = roche_misaligned_gradOmega(q, F, d, misalignment, r)

  with parameters

    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array
        = [-grad Omega_x, -grad Omega_y, -grad Omega_z, -Omega(x,y,z)]
*/


static PyObject *roche_misaligned_gradOmega(PyObject *self, PyObject *args) {

  auto fname = "roche_misaligned_gradOmega"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  double p[7];

  PyObject *o_misalignment;

  PyArrayObject *o_x;

  if (!PyArg_ParseTuple(args, "dddOO!",
        p, p + 1, p + 2,
        &o_misalignment,
        &PyArray_Type, &o_x)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double
    *x = (double*) PyArray_DATA(o_x),
    g[4];

  if (PyFloat_Check(o_misalignment)) {

    p[3] = PyFloat_AsDouble(o_misalignment);
    p[4] = 0; // Omega0 = 0

    Tmisaligned_rotated_roche<double> b(p);
    b.grad(x, g);

  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);

    p[3] = s[0];
    p[4] = s[1];
    p[5] = s[2];
    p[6] = 0; // Omega0 = 0

    Tmisaligned_roche<double> b(p);
    b.grad(x, g);

  } else {
    raise_exception(fname + "::This type of misalignment is not supported");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  return PyArray_FromVector(4, g);
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

  auto fname = "roche_gradOmega_only"_s;

  double p[4];

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dddO!", p, p + 1, p + 2, &PyArray_Type, &X)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  Tgen_roche<double> b(p);

  npy_intp dims[1] = {3};

  #if defined(USING_SimpleNewFromData)
  double *g = new double [3];
  b.grad_only((double*)PyArray_DATA(X), g);
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  b.grad_only((double*)PyArray_DATA(X), (double*)PyArray_DATA((PyArrayObject *)pya));
  #endif

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

  auto fname = "rotstar_gradOmega_only"_s;

  double p[2];

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dO!", p, &PyArray_Type, &X)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  Trot_star<double> b(p);

  npy_intp dims[1] = {3};

  #if defined(USING_SimpleNewFromData)
  double *g = new double [3];
  b.grad_only((double*)PyArray_DATA(X), g);
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  b.grad_only((double*)PyArray_DATA(X), (double*)PyArray_DATA((PyArrayObject *)pya));
  #endif

  return pya;
}



/*
  C++ wrapper for Python code:

  Calculate the gradient of the potential of the rotating star potential
  with misalignment

      -grad Omega (x,y,z; omega, s)

   where

      Omega(x,y,z; omega, s)
            = 1/|r|  + 1/2 omega^2 | r - r (r.s)|^2

  with
      r = {x, y, z}
      s = {sx, sy, sz}    |s| = 1

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    g = rotstar_misaligned_gradOmega_only(omega, misalignment, r)

  with parameters

    omega: float - parameter of the potential
          Note:
          for comparison to Roche model (a=1): omega = F sqrt(1+q),
          for independent star of mass M : omega = angular velocity/sqrt(G M)
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
              s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array = -grad Omega (x,y,z)
*/

static PyObject *rotstar_misaligned_gradOmega_only(PyObject *self, PyObject *args) {

  auto fname = "rotstar_misaligned_gradOmega_only"_s;

  double p[5];

  PyObject *o_misalignment;

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dOO!",
      p,
      &o_misalignment,
      &PyArray_Type, &X)
  ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  Tmisaligned_rot_star<double> b(p);

  if (PyFloat_Check(o_misalignment)) {

    double s = std::sin(PyFloat_AsDouble(o_misalignment));

    p[1] = s;
    p[2] = 0;
    p[3] = std::sqrt(1. - s*s);

  } else if (PyArray_Check(o_misalignment)) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    for (int i = 0; i < 3; ++i) p[i+1] = s[i];
  }

  double g[3];

  b.grad_only((double*)PyArray_DATA(X), g);

  return PyArray_FromVector(3, g);
}

/*
  C++ wrapper for Python code:

  Calculate the gradient of the potential of the potential corresponding
  to the sphere

      -grad Omega (x,y,z)

  Python:

    g = sphere_gradOmega_only(r)

  with parameters

    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array = -grad Omega (x,y,z)
*/

static PyObject *sphere_gradOmega_only(PyObject *self, PyObject *args) {

  auto fname = "sphere_gradOmega_only"_s;

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }
  double
    *x = (double*)PyArray_DATA(X),
    R = utils::hypot3(x),
    F = 1/(R*R*R);

  npy_intp dims[1] = {3};

  #if defined(USING_SimpleNewFromData)
  double *g = new double [3];
  for (int i = 0; i < 3; ++i) g[i] = F*x[i];
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *g = (double*)PyArray_DATA((PyArrayObject *)pya);
  for (int i = 0; i < 3; ++i) g[i] = F*x[i];
  #endif

  return pya;
}

/*
  C++ wrapper for Python code:

  Calculate the gradient of the potential of the generalized
  Kopal potential Omega with misaligned spin and orbital angular
  velocity vectors at a given point

      -grad Omega (x,y,z)

  Python:

    g = roche_misaligned_gradOmega_only(q, F, d, misalignment, r)

   with parameters

    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
    or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns float

    g : 1-rank numpy array = -grad Omega (x,y,z)
*/

static PyObject *roche_misaligned_gradOmega_only(PyObject *self, PyObject *args) {

  auto fname = "roche_misaligned_gradOmega_only"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  double p[7];

  PyObject *o_misalignment;

  PyArrayObject *o_x;

  if (!PyArg_ParseTuple(args, "dddOO!",
        p, p + 1, p + 2,
        &o_misalignment,
        &PyArray_Type, &o_x)
      ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double
    *x = (double*) PyArray_DATA(o_x),
    g[3];

  if (PyFloat_Check(o_misalignment)) {

    p[3] = PyFloat_AsDouble(o_misalignment);
    p[4] = 0; // Omega0 = 0

    Tmisaligned_rotated_roche<double> b(p);
    b.grad_only(x, g);
  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);

    p[3] = s[0];
    p[4] = s[1];
    p[5] = s[2];
    p[6] = 0; // Omega0 = 0

    Tmisaligned_roche<double> b(p);
    b.grad_only(x, g);
  } else {
    raise_exception(fname + "::This type of misalignment is not supported");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  return PyArray_FromVector(3, g);
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
    raise_exception("roche_Omega::Problem reading arguments");
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
    raise_exception("rotstar_Omega::Problem reading arguments");
    return NULL;
  }

  p[1] = 0; // Omega

  Trot_star<double> b(p);

  return PyFloat_FromDouble(-b.constrain((double*)PyArray_DATA(X)));
}


/*
  C++ wrapper for Python code:

  Calculate the value of the potential of the rotating star with
  misaligned spin at a given point

    Omega (x,y,z; omega, s) = 1/|r| + 1/2 omega^2 |r - s(s*r)|^2

  with
    r = {x, y, z}
    s = {sx, sy, sz}

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    Omega0 = rotstar_misaligned_Omega(omega, misalignment, r)

  with parameters

    omega: float - parameter of the potential
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
              s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    r: 1-rank numpy array of length 3 = [x,y,z]

  and returns a float

    Omega0 - value of the Omega at (x,y,z)
*/

static PyObject *rotstar_misaligned_Omega(PyObject *self, PyObject *args) {

  auto fname = "rotstar_misaligned_Omega"_s;

  double p[5];

  PyObject *o_misalignment;

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "dOO!",
       p,
       &o_misalignment,
       &PyArray_Type, &X)
  ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  p[4] = 0;     // Omega

  if (PyFloat_Check(o_misalignment)) {
    double s = std::sin(PyFloat_AsDouble(o_misalignment));

    p[1] = s;
    p[2] = 0;
    p[3] = std::sqrt(1. - s*s);

  } else if (PyArray_Check(o_misalignment)) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    for (int i = 0; i < 3; ++i) p[i+1] = s[i];

  } else {
    raise_exception(fname + "::This type of misalignment is not supported.");
    return NULL;
  }

  Tmisaligned_rot_star<double> b(p);

  return PyFloat_FromDouble(-b.constrain((double*)PyArray_DATA(X)));
}

/*
  C++ wrapper for Python code:

  Calculate the value of the potential of the sphere at
  a given point

      Omega (x,y,z) = 1/sqrt(x^2+ y^2 + z^2)

  Python:

    Omega0 = rotstar_Omega(r)

   with parameters

      r: 1-rank numpy array of length 3 = [x,y,z]


  and returns a float

    Omega0 - value of the Omega at (x,y,z)
*/

static PyObject *sphere_Omega(PyObject *self, PyObject *args) {

  auto fname = "sphere_Omega"_s;

  double p[2];

  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", p, &PyArray_Type, &X)) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double R = utils::hypot3((double*)PyArray_DATA(X));

  return PyFloat_FromDouble(1/R);
}

/*
  C++ wrapper for Python code:

  Calculate the value of the potential of the generalized
  Kopal potential Omega for misaligned binaries at a given point

      Omega (x,y,z; q, F, d, misalignment)

  Python:

    Omega0 = roche_misaligned_Omega(q, F, delta, misalignment, r)

   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
      r: 1-rank numpy array of length 3 = [x,y,z]

  and returns a float

    Omega0 - value of the Omega at (x,y,z)
*/


static PyObject *roche_misaligned_Omega(PyObject *self, PyObject *args) {

  auto fname = "roche_misaligned_Omega"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  double p[7];

  PyObject *o_misalignment;

  PyArrayObject *o_x;

  if (!PyArg_ParseTuple(args, "dddOO!",
       p, p + 1, p + 2,
       &o_misalignment,
       &PyArray_Type, &o_x)){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double *x = (double*) PyArray_DATA(o_x);

  if (PyFloat_Check(o_misalignment)) {
    p[3] = PyFloat_AsDouble(o_misalignment);
    p[4] = 0; // Omega0 = 0

    if (verbosity_level>=4)
      report_stream << fname << "::END" << std::endl;

    Tmisaligned_rotated_roche<double> b(p);
    return PyFloat_FromDouble(-b.constrain(x));
  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);

    p[3] = s[0];
    p[4] = s[1];
    p[5] = s[2];
    p[6] = 0; // Omega0 = 0

    if (verbosity_level>=4)
      report_stream << fname << "::END" << std::endl;

    Tmisaligned_roche<double> b(p);
    return PyFloat_FromDouble(-b.constrain(x));
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  raise_exception(fname + "::This type of misalignment is not supported");
  return NULL;
}


/*
  C++ wrapper for Python code:

    Marching meshing of Roche lobes implicitely defined

    Omega_0 = Omega(x,y,z)

    by generalized Kopal potential:

    Omega(x,y,z) =  1/r1 + q [1/r2 - x/delta^2] + 1/2 F^2(1 + q) (x^2 + y^2)
    r1 = sqrt(x^2 + y^2 + z^2)
    r1 = sqrt((x-delta)^2 + y^2 + z^2)

  Python:

    dict = roche_marching_mesh(q, F, d, Omega0, delta, <keyword>=[true,false], ... )

  where parameters

    positional:

      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      Omega0: float - value of the generalized Kopal potential
      delta: float - size of triangles edges projected to tangent space

    keywords:
      choice: integer, default 0
          0 - primary lobe
          1 - secondary lobe
        for contacts choice is 0 or 1
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

  auto fname = "roche_marching_mesh"_s;

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
    b_full = true,
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

    raise_exception(fname + "::Problem reading arguments");
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
    raise_exception(fname + "::This choice is not supported");
    return NULL;
  }

  //
  // Choosing the meshing initial point
  //

  double r[3], g[3];

  if (!gen_roche::meshing_start_point(r, g, choice, Omega0, q, F, d)){
    raise_exception(fname + "::Determining initial meshing point failed");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream
      << fname <<  "::choice=" << choice << '\n'
      << "r=" << r[0] << " " <<  r[1] << " " << r[2] << '\n'
      << "g=" << g[0] << " " <<  g[1] << " " << g[2] << '\n';

  //
  //  Marching triangulation of the Roche lobe
  //

  double params[4] = {q, F, d, Omega0};

  if (verbosity_level>=4)
    report_stream
      << fname << "::q=" << q
      << " F=" << F << " d=" << d
      << " Omega0=" << Omega0 << " delta=" << delta
      << " full=" << b_full << " max_triangles=" << max_triangles <<'\n';

  Tmarching<double, Tgen_roche<double>> march(params);

  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;
  std::vector<double> *GatV = 0;

  if (b_vnormgrads) GatV = new std::vector<double>;

  int error =
    (b_full ?
      march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi) :
      march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
    );

  switch(error) {
    case 1:
      raise_exception("There are too many triangles!");
      return NULL;
    case 2:
      raise_exception("Projections are failing!");
      return NULL;
  }

  if (verbosity_level >=4)
    report_stream << fname
      << "::V.size=" << V.size()
      << " Tr.size=" << Tr.size() << '\n';

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
        orientation of the initial polygon front
      init_dir: 1-rank numpy array of floats = [theta, phi], default [0,0]
        direction of the initial point in marching given by spherical angles

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

  auto fname = "rotstar_marching_mesh"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

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
    (char*)"init_dir",
    NULL};

  double omega, Omega0, delta,
         init_phi = 0, init_dir[2] = {0., 0.};

  int max_triangles = 10000000; // 10^7

  bool
    b_full = true,
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

  PyArrayObject *o_init_dir = 0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddd|iO!O!O!O!O!O!O!O!O!O!O!O!dO!", kwlist,
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
      &init_phi,
      &PyArray_Type, &o_init_dir)
  ){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::Omega=" << Omega0 << " omega=" << omega << " delta=" << delta << std::endl;

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
  if (o_init_dir) {
    double *p = (double*)PyArray_DATA(o_init_dir);
    init_dir[0] = p[0];
    init_dir[1] = p[1];
  }

  //
  // Check if the lobe exists
  //
  if (27*utils::sqr(omega)/(8*utils::cube(Omega0)) > 1){
    raise_exception(fname + "::The lobe does not exist.");
    return NULL;
  }

  //
  // Storing results in dictioonary
  // https://docs.python.org/2/c-api/dict.html
  //

  PyObject *results = PyDict_New();

  //
  // Getting initial meshing point
  //

  if (verbosity_level>=4)
    report_stream << fname << "::Point on surface" << std::endl;

  double r[3], g[3];
  //rot_star::meshing_start_point(r, g, Omega0, omega);
  rot_star::point_on_surface(Omega0, omega, init_dir[0], init_dir[1], r, g);

  if (verbosity_level>=4)
    report_stream << fname
      << "::r=" << r[0] << " " << r[1] << " " << r[2]
      << "g=" << g[0] << " " << g[1] << " " << g[2]
      << std::endl;

  //
  //  Marching triangulation of the Roche lobe
  //

  if (verbosity_level>=4)
    report_stream << fname << "::Marching" << std::endl;

  double params[3] = {omega, Omega0};

  Tmarching<double, Trot_star<double>> march(params);

  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;
  std::vector<double> *GatV = 0;

  if (b_vnormgrads) GatV = new std::vector<double>;

  int error =
    (b_full ?
      march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi) :
      march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
    );

  switch(error) {
    case 1:
      raise_exception("There are too many triangles!");
      return NULL;
    case 2:
      raise_exception("Projections are failing!");
      return NULL;
  }

  if (verbosity_level >= 4)
    report_stream << fname << "::Outputing" << std::endl;

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

  if (verbosity_level>=4)
    report_stream << fname << "::END\n";

  return results;
}


/*
  C++ wrapper for Python code:

  Marching meshing of rotating star with misalignment implicitely defined
  by the  potential

    Omega (x,y,z; omega, s) = 1/|r| + 1/2 omega^2 |r - s(s*r)|^2

  with
    r = {x, y, z}
    s = {sx, sy, sz}    |s| = 1

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    dict = rotstar_misaligned_marching_mesh(omega, misalignment, Omega0, delta, <keyword>= ... )

  where parameters

    positional:
      omega: float - parameter of the potential
          Note:
          for comparison to Roche model (a=1): omega = F sqrt(1+q),
          for independent star of mass M : omega = angular velocity/sqrt(G M)
      misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
              s = [sin(angle), 0, cos(angle)]
      or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
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
        orientation of the initial polygon front
      init_dir: 1-rank numpy array of floats = [theta, phi], default [0,0]
        direction of the initial point in marching given by spherical angles

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

static PyObject *rotstar_misaligned_marching_mesh(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_misaligned_marching_mesh"_s;

  if (verbosity_level>=4)
    report_stream  << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"omega",
    (char*)"misalignment",
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
    (char*)"init_dir",
    NULL};

  double omega, Omega0, delta,
         init_phi = 0, init_dir[2] = {0., 0.};

  int max_triangles = 10000000; // 10^7

  bool
    b_full = true,
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

  PyObject *o_misalignment;

  PyArrayObject *o_init_dir = 0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dOdd|iO!O!O!O!O!O!O!O!O!O!O!O!dO!", kwlist,
      &omega, &o_misalignment, &Omega0, &delta, // neccesary
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
      &init_phi,
      &PyArray_Type, &o_init_dir)
  ){
    raise_exception(fname + "::Problem reading arguments");
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
  if (o_init_dir) {
    double *p = (double*)PyArray_DATA(o_init_dir);
    init_dir[0] = p[0];
    init_dir[1] = p[1];
  }

  //
  // Check if the lobe exists
  //
  if (27*utils::sqr(omega)/(8*utils::cube(Omega0)) > 1){
    raise_exception(fname + "::The lobe does not exist.");
    return NULL;
  }

  //
  // Reading spin
  //
  double spin[3];

  if (PyFloat_Check(o_misalignment)) {
    double s = std::sin(PyFloat_AsDouble(o_misalignment));

    spin[0] = s;
    spin[1] = 0;
    spin[2] = std::sqrt(1. - s*s);

  } else if (PyArray_Check(o_misalignment)) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    for (int i = 0; i < 3; ++i) spin[i] = s[i];

  } else {
    raise_exception(fname + "::This type of misalignment is not supported.");
    return NULL;
  }

  //
  // Storing results in dictioonary
  // https://docs.python.org/2/c-api/dict.html
  //

  PyObject *results = PyDict_New();

  //
  // Getting initial meshing point
  //

  double r[3], g[3];
  //rot_star::meshing_start_point(r, g, Omega0, omega);
  rot_star::point_on_surface(Omega0, omega, spin, init_dir[0], init_dir[1], r, g);

  //
  //  Marching triangulation of the Roche lobe
  //

  double params[5] = {omega, spin[0], spin[1], spin[2], Omega0};

  Tmarching<double, Tmisaligned_rot_star<double>> march(params);

  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;
  std::vector<double> *GatV = 0;

  if (b_vnormgrads) GatV = new std::vector<double>;


  int error =(b_full ?
      march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi):
      march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
      );

  switch(error) {
    case 1:
      raise_exception("There are too many triangles!");
      return NULL;
    case 2:
      raise_exception("Projections are failing!");
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

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  return results;
}



/*
  C++ wrapper for Python code:

    Marching meshing of sphere implicitely defined

      Omega0 = Omega(x,y,z) = 1/sqrt(x^2 + y^2 + z^2)

  Python:

    dict = sphere_marching_mesh(Omega0, delta, <keyword>=[true,false], ... )

  where parameters

    positional:
      Omega0: float - value of the potential
      delta: float - size of triangles edges projected to tangent space

    keywords:
      choice: integer, default 0
          0 - primary lobe
          1 - secondary lobe
        for contacts choice is 0 or 1
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

static PyObject *sphere_marching_mesh(PyObject *self, PyObject *args, PyObject *keywds) {

 auto fname = "sphere_marching_mesh"_s;

  //
  // Reading arguments
  //

 char *kwlist[] = {
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

  double Omega0, delta,
         init_phi = 0;

  int max_triangles = 10000000; // 10^7

  bool
    b_full = true,
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
      args, keywds,  "dd|iO!O!O!O!O!O!O!O!O!O!O!O!d", kwlist,
      &Omega0, &delta,                  // neccesary
      &max_triangles,                   // optional ...
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
    raise_exception(fname + "::Problem reading arguments");
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
  //  Marching triangulation of the Roche lobe
  //

  double R = 1/Omega0;

  Tmarching<double, Tsphere<double> > march(&R);

  double r[3], g[3];

  march.init(r, g);

  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr;
  std::vector<double> *GatV = 0;

  int error =
    (b_full ?
      march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi) :
      march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
    );

  switch(error) {
    case 1:
      raise_exception("There are too many triangles!");
      return NULL;
    case 2:
      raise_exception("Projections are failing!");
      return NULL;
  }

  if (b_vnormgrads)
    GatV = new std::vector<double>(V.size(), Omega0*Omega0);

  //
  // Calculte the mesh properties
  //
  int vertex_choice = 0;

  double
    area, volume,
    *p_area = 0,
    *p_volume = 0;

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

  if (b_centers || b_cnormals) {

    std::vector<T3Dpoint<double>>::iterator itC, itNatC;

    if (b_centers) {
      C = new std::vector<T3Dpoint<double>> (Tr.size());
      itC = C->begin();
    }

    if (b_cnormals){
      NatC = new std::vector<T3Dpoint<double>> (Tr.size());
      itNatC = NatC->begin();
    }

    double f, t, r[3];

    for (auto tr : Tr) {

      f = 0;
      for (int i = 0; i < 3; ++i) {
        r[i] = t = V[tr[0]][i] +  V[tr[1]][i] + V[tr[2]][i];
        f += t*t;
      }

      f = 1/std::sqrt(f);

      for (int i = 0; i < 3; ++i) r[i] *= f;

      // C
      if (b_centers) {
        for (int i = 0; i < 3; ++i) (*itC)[i] = R*r[i];
        ++itC;
      }

      // Cnorms
      if (b_cnormals) {
        for (int i = 0; i < 3; ++i) (*itNatC)[i] = r[i];
        ++itNatC;
      }
    }
  }

  if (b_cnormgrads)
    GatC = new std::vector<double>(V.size(), Omega0*Omega0);

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

    Marching meshing of generalized Roche lobes with misaligned spin and orbit angular
    velocity vector that is implicitely

      Omega(x,y,z) = Omega0

    defined by Avni's generalized Kopal potential:

      Omega(x,y,z,params) =
      1/r1 + q(1/r2 - x/delta^2) +
      1/2 (1 + q) F^2 [(x cos theta' - z sin theta')^2 + y^2]

    r1 = sqrt(x^2 + y^2 + z^2)
    r2 = sqrt((x-delta)^2 + y^2 + z^2)

  Python:

    dict = roche_misaligned_marching_mesh(q, F, d, misalignment, Omega0, delta, <keyword>=[true,false], ... )

  where parameters

    positional:

      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      misalignment:  in rotated coordinate system:
          float - angle between spin and orbital angular velocity vectors [rad]
      or in canonical coordinate system:
          1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
      Omega0: float - value of the generalized Kopal potential
      delta: float - size of triangles edges projected to tangent space

    keywords:
      choice: integer, default 0
          0 - primary lobe
          1 - secondary lobe
        for contacts choice is 0 or 1
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


static PyObject *roche_misaligned_marching_mesh(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_marching_mesh"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  //
  // Reading arguments
  //
  char *kwlist[] = {
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
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

  PyObject *o_misalignment;

  double
    q, F, d, Omega0, delta,
    init_phi = 0;

  int
    choice = 0,
    max_triangles = 10000000; // 10^7

  bool
    b_full = true,
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
        args, keywds,  "dddOdd|iiO!O!O!O!O!O!O!O!O!O!O!O!d", kwlist,
        &q, &F, &d, &o_misalignment, &Omega0, &delta,  // neccesary
        &choice,                              // optional ...
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

    raise_exception(fname + "::Problem reading arguments");
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
    raise_exception(fname + "::This choice is not supported.");
    return NULL;
  }

  //
  // Choosing the meshing initial point
  //
  bool rotated, ok, aligned = false;

  double r[3], g[3], theta, *s = 0;

  if (PyFloat_Check(o_misalignment)) {

    theta = PyFloat_AsDouble(o_misalignment);
    aligned = (std::sin(theta) == 0);     // theta ~0, pi => aligned

    ok = misaligned_roche::meshing_start_point(r, g, choice, Omega0, q, F, d, theta);
    rotated = true;

  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    aligned  = (s[0] == 0 && s[1] == 0);

    // we could work with s[0]==0, calculate aligned case make simple
    // rotation around x-axis

    if (verbosity_level>=4)
      report_stream << fname
        << "::spin:" << s[0] << ' ' << s[1] << ' ' << s[2]
        << " Omega:" <<  Omega0
        << " q=" << q
        << " F=" << F
        << " d=" << d
        << " delta =" << delta << '\n'
        << " full=" << b_full
        << " max_triangles=" << max_triangles <<'\n';

    ok = misaligned_roche::meshing_start_point(r, g, choice, Omega0, q, F, d, s);
    rotated = false;

    if (verbosity_level>=4)
      report_stream << fname
        << "::r=" << r[0] << ' ' << r[1] << ' ' << r[2]
        << " g=" << g[0] << ' ' << g[1] << ' ' << g[2] << '\n';

  } else {
    raise_exception(fname + "::This type of misalignment is not supported.");
    return NULL;
  }

  if (!ok || s == 0){
    raise_exception(fname + "::Determining initial meshing point failed.");
    return NULL;
  }

  //
  //  Marching triangulation of the Roche lobe and calculate central points
  //

  std::vector<double> *GatC = 0, *GatV = 0;
  std::vector<T3Dpoint<double>> V, NatV,  *C = 0, *NatC = 0;
  std::vector<T3Dpoint<int>> Tr;

  if (b_centers) C = new std::vector<T3Dpoint<double>>;
  if (b_cnormals) NatC = new std::vector<T3Dpoint<double>>;
  if (b_cnormgrads) GatC = new std::vector<double>;
  if (b_vnormgrads) GatV = new std::vector<double>;

  int error = 0;

  if (aligned) {
    double params[] = {q, F, d, Omega0};

    Tmarching<double, Tgen_roche<double>> march(params);

    error = (b_full ?
         march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi) :
         march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
        );

    if (error == 0 && !march.central_points(V, Tr, C, NatC, GatC)) error = 4;

  } else {
    if (rotated) {
      double params[] = {q, F, d, theta, Omega0};

      Tmarching<double, Tmisaligned_rotated_roche<double>> march(params);

      error = (
        b_full ?
          march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi):
          march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
        );

      if (error == 0 && !march.central_points(V, Tr, C, NatC, GatC)) error = 4;

    } else {
      double params[] = {q, F, d, s[0], s[1], s[2], Omega0};

      Tmarching<double, Tmisaligned_roche<double>> march(params);

      error = (
        b_full ?
          march.triangulize_full_clever(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi):
          march.triangulize(r, g, delta, max_triangles, V, NatV, Tr, GatV, init_phi)
        );

      if (error == 0 && !march.central_points(V, Tr, C, NatC, GatC)) error = 4;
    }
  }


  if (error && verbosity_level>=2) {
    report_stream << fname
      << "::q=" << q << " F=" << F
      << " d=" << d << " Omega0=" << Omega0
      << " delta=" << delta << " full=" << b_full
      << " max_triangles=" << max_triangles <<'\n';

    if (rotated)
      report_stream << fname << " theta=" << theta << '\n';
    else
      report_stream << fname << " s=(" << s[0] << ',' << s[1] << ',' << s[2] << ")\n";
  }

  switch(error) {
    case 1:
      raise_exception("There are too many triangles!");
      return NULL;
    case 2:
      raise_exception("Projections are failing!");
      return NULL;
    case 4:
      raise_exception("Central points did not converge!");
      return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::V.size=" << V.size() << " Tr.size=" << Tr.size() << '\n';

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

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

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

  auto fname = "mesh_visibility"_s;

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
    raise_exception(fname + "::Problem reading arguments");
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

    raise_exception(fname + "::Input numpy arrays are not C-contiguous");
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

    raise_exception("mesh_rough_visibility::Problem reading arguments");
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

  #if defined(USING_SimpleNewFromData)
  double *M = new double [Nt], *p = M;
  for (auto && m: Mt) *(p++) = (m == hidden ? 0 : (m == partially_hidden ? 0.5 : 1.0));
  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, M);
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  #else
  PyObject *pya = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *M = (double *) PyArray_DATA((PyArrayObject *)pya);
  for (auto && m: Mt) *(M++) = (m == hidden ? 0 : (m == partially_hidden ? 0.5 : 1.0));
  #endif

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

  auto fname = "mesh_offseting"_s;

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
    raise_exception(fname + "::Problem reading arguments");
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

    raise_exception(fname + "::Offseting failed");
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
    raise_exception("mesh_properties::Problem reading arguments");
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

    raise_exception("mesh_export_povray::Problem reading arguments");
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

  auto fname = "LDmodelFromTuple"_s;

  if (!PyTuple_CheckExact(p)) {
    if (verbosity_level >=2) report_stream << fname + "::LD model description is not a tuple.\n";
    return false;
  }

  if (PyTuple_Size(p) == 0) {
    if (verbosity_level >=2) report_stream << fname + "::LD model tuple is empty.\n";
    return false;
  }

  PyObject *s = PyTuple_GetItem(p, 0);

  if (!PyString_Check(s)) {
    if (verbosity_level >=2) report_stream << fname + "::LD model name is not string.\n";
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

    case "power"_hash32:
      par = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(p, 1));
      pmodel = new TLDpower<double>(par);
      return true;
  }

  if (verbosity_level >=2)
    report_stream << fname + "::Don't know to handle this LD model.\n";

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

    F = mesh_radiosity_problem_triangles(
        V, Tr, NatT, A, R, F0, LDmod, LDidx, model, support, <keyword>=<value>, ... )

  where positional parameters:

    V[][3]: 2-rank numpy array of vertices
    Tr[][3]: 2-rank numpy array of 3 indices of vertices
            composing triangles of the mesh aka connectivity matrix
    N[][3]: 2-rank numpy array of normals at triangles/vertices
    A[]: 1-rank numpy array of areas of triangles
    R[]: 1-rank numpy array of albedo/reflection at triangle/vertices
    F0[]: 1-rank numpy array of intrisic radiant exitance at triangle/vertices

    LDmod: list of tuples of the format
            ("name", sequence of parameters)
            supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "power"       4 parameters

    LDidx[]: 1-rank numpy array of indices of LD models used on each triangle/vertex

    model : string - name of the reflection model in use
             method in {"Wilson", "Horvat"}

    support: string
              {"triangles","vertices"}
  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg)
          threshold for permitted cos(view-angle)
    epsF: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.

  Returns:
    F[]: 1-rank numpy array of radiosities (intrinsic and reflection)
          at triangles/vertices

  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect,
    Astrophysical Journal,  356, 613-622, 1990 June
*/

static PyObject *mesh_radiosity_problem(
  PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "mesh_radiosity_problem"_s;

  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr",
    (char*)"N",
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"LDidx",
    (char*)"model",
    (char*)"support",
    (char*)"epsC",
    (char*)"epsF",
    (char*)"max_iter",
    NULL
  };

  int max_iter = 100;         // default value

  double
    epsC = 0.00872654,        // default value
    epsF = 1e-12;             // default value

  PyArrayObject *oV, *oT, *oN, *oA, *oR, *oF0, *oLDidx;

  PyObject *oLDmod, *omodel, *osupport;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!O!O!O!|ddi", kwlist,
      &PyArray_Type, &oV,         // neccesary
      &PyArray_Type, &oT,
      &PyArray_Type, &oN,
      &PyArray_Type, &oA,
      &PyArray_Type, &oR,
      &PyArray_Type, &oF0,
      &PyList_Type, &oLDmod,
      &PyArray_Type, &oLDidx,
      &PyString_Type, &omodel,
      &PyString_Type, &osupport,
      &epsC,                      // optional
      &epsF,
      &max_iter)){

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  //
  // Storing input data
  //

  std::vector<TLDmodel<double>*> LDmod;

  if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
    raise_exception(fname +  "::Not able to read LD models");
    return NULL;
  }


  std::vector<int> LDidx;
  PyArray_ToVector(oLDidx, LDidx);

  std::vector<T3Dpoint<double>> V, N;
  std::vector<T3Dpoint<int>> Tr;

  std::vector<double> A;
  PyArray_ToVector(oA, A);

  PyArray_To3DPointVector(oV, V);
  PyArray_To3DPointVector(oT, Tr);
  PyArray_To3DPointVector(oN, N);

  //
  // Determine the LD view-factor matrix
  //

  std::vector<Tview_factor<double>> Fmat;
  {
    char *s =  PyString_AsString(osupport);

    switch (fnv1a_32::hash(s)) {

      case "triangles"_hash32:
        triangle_mesh_radiosity_matrix_triangles(
          V, Tr, N, A, LDmod, LDidx,  Fmat);
      break;

      case "vertices"_hash32:
        triangle_mesh_radiosity_matrix_vertices(
          V, Tr, N, A, LDmod, LDidx,  Fmat);
      break;

      default:
        raise_exception(fname + "::This support type is not supported");
      return NULL;
    }
  }

  for (auto && ld: LDmod) delete ld;
  LDmod.clear();

  // some clean up to reduce memory footprint
  LDidx.clear(); V.clear(); Tr.clear(); N.clear();  A.clear();

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
        success = solve_radiosity_equation_Wilson(Fmat, R, F0, F);

        break;

      case "Horvat"_hash32:
        success = solve_radiosity_equation_Horvat(Fmat, R, F0, F);
        break;

      default:
        raise_exception(fname + "::This radiosity model =" + std::string(s) + " does not exist");
        return NULL;
    }

    if (!success)
      raise_exception(fname + "::slow convergence");
  }

  return PyArray_FromVector(F);
}

/*
  C++ wrapper for Python code:

  Calculate radiosity of triangles on n convex bodies due to reflection
  according to a chosen reflection model using triangles or vertices as
  support of the surface.

  Python:

    F = mesh_radiosity_problem_nbody_convex(
        V, Tr, N, A, R, F0, LDmod, model, support, <keyword> = <value>, ... )

  where positional parameters:

    V = {V1, V2, ...} :
      list of 2-rank numpy array of vertices V[][3],
      length of the list is n, as number of bodies

    Tr = {Tr1, Tr2, ...} :
      list of 2-rank numpy array of 3 indices of vertices Tr[][3]
      composing triangles of the mesh aka connectivity matrix
      length of the list is n, as number of bodies

    N = {N1, N2, ...} :
      list of 2-rank numpy array of normals at triangles or vertices N[][3]

    A = {A1, A2, ...} :
      list of 1-rank numpy array of areas of triangles A[]

    R = {R1, R2, ...} :
      list of 1-rank numpy array of albedo/reflection at triangles
      or vertices R[]

    F0 = {F0_0, F0_1, ...} :
      list of 1-rank numpy array of intrisic radiant exitance at
      triangles or vertices F0[]

    LDmod = {LDmod1, LDmod2,..}: list of tuples of the format

            ("name", sequence of parameters)
            with one model per body. Supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "power"       4 parameters


     model : string - name of the reflection model in use
             method in {"Wilson", "Horvat"}

    support: string
              {"triangles","vertices"}

  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg)
          threshold for permitted cos(view-angle)
    epsF: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.

  Returns:
    F = {F_0, F_1, ...} : list of 1-rank numpy array of total radiosities
                      (intrinsic and reflection) at triangles or vertices

  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect,
    Astrophysical Journal,  356, 613-622, 1990 June
*/

static PyObject *mesh_radiosity_problem_nbody_convex(
  PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "mesh_radiosity_problem_nbody_convex"_s;

  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr",
    (char*)"N",
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"model",
    (char*)"support",
    (char*)"epsC",
    (char*)"epsF",
    (char*)"max_iter",
    NULL
  };

  int max_iter = 100;         // default value

  double
    epsC = 0.00872654,        // default value
    epsF = 1e-12;             // default value

  PyObject *oLDmod, *omodel, *oV, *oTr, *oN, *oA, *oR, *oF0, *osupport;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "O!O!O!O!O!O!O!O!O!|ddi", kwlist,
        &PyList_Type, &oV,         // neccesary
        &PyList_Type, &oTr,
        &PyList_Type, &oN,
        &PyList_Type, &oA,
        &PyList_Type, &oR,
        &PyList_Type, &oF0,
        &PyList_Type, &oLDmod,
        &PyString_Type, &omodel,
        &PyString_Type, &osupport,
        &epsC,                     // optional
        &epsF,
        &max_iter)
      ){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  //
  // Storing input data
  //

  std::vector<TLDmodel<double>*> LDmod;

  if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
    raise_exception(fname + "::Not able to read LD models");
    return NULL;
  }

  //
  // Checking number of bodies
  //

  int n = LDmod.size();

  if (n <= 1){
    raise_exception(fname + "::There seem to just n=" + std::to_string(n) + " bodies.");
    return NULL;
  }

  //
  // Check is there is interpolation is used
  //
  std::vector<std::vector<T3Dpoint<double>>> V(n), N(n);
  std::vector<std::vector<T3Dpoint<int>>> Tr(n);
  std::vector<std::vector<double>> A(n), R(n), F0(n), F;

  for (int b = 0; b < n; ++b){
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oV, b), V[b]);
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oN, b), N[b]);
    PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oTr, b), Tr[b]);

    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oA, b), A[b]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oR, b), R[b]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oF0, b), F0[b]);
  }

  //
  // Determine the LD view-factor matrix
  //

  std::vector<Tview_factor_nbody<double>> Fmat;

  {
    char *s =  PyString_AsString(osupport);

    switch (fnv1a_32::hash(s)) {

      case "triangles"_hash32:
        triangle_mesh_radiosity_matrix_triangles_nbody_convex(
          V, Tr, N, A, LDmod, Fmat);
        break;

      case "vertices"_hash32:
        triangle_mesh_radiosity_matrix_vertices_nbody_convex(
          V, Tr, N, A, LDmod, Fmat);
        break;

      default:
        raise_exception(fname + "::This support type is not supported");
        return NULL;
    }
  }


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
        success = solve_radiosity_equation_Wilson_nbody(Fmat, R, F0, F);
      break;

      case "Horvat"_hash32:
      	success = solve_radiosity_equation_Horvat_nbody(Fmat, R, F0, F);
      break;

      default:
        raise_exception(fname + "::This radiosity model ="+ std::string(s) + " does not exist");
        return NULL;
    }

    if (!success) raise_exception(fname + "::slow convergence");
  }


  PyObject *results = PyList_New(n);

  for (int b = 0; b < n; ++b)
    PyList_SetItem(results, b, PyArray_FromVector(F[b]));

  return results;
}


/*
  C++ wrapper for Python code:

  Calculate radiosity of triangles on n convex bodies due to reflection
  according to a chosen reflection model using triangles or vertices
  as support of the surface.

  Python:

    dict = mesh_radiosity_redistrib_problem_nbody_convex(
        V, Tr, N, A, R, F0, LDmod, Dmod, Dweight, model, support, <keyword> = <value>, ... )

  where positional parameters:

    V = {V1, V2, ...} :
      list of 2-rank numpy array of vertices V[][3],
      length of the list is n, as number of bodies

    Tr = {Tr1, Tr2, ...} :
      list of 2-rank numpy array of 3 indices of vertices Tr[][3]
      composing triangles of the mesh aka connectivity matrix
      length of the list is n, as number of bodies

    N = {N1, N2, ...} :
      list of 2-rank numpy array of normals at triangles or vertices N[][3]
    A = {A1, A2, ...} :
      list of 1-rank numpy array of areas of triangles A[]
    R = {R1, R2, ...} :
      list of 1-rank numpy array of albedo/reflection of triangles R[]
    F0 = {F0_0, F0_1, ...} :
      list of 1-rank numpy array of intrisic radiant exitance F0[]

    LDmod = {LDmod1, LDmod2, ...}: list of tuples of the format

            ("name", sequence of parameters)

            with one model per body. Supported ld models:
              "uniform"     0 parameters
              "linear"      1 parameters
              "quadratic"   2 parameters
              "nonlinear"   3 parameters
              "logarithmic" 2 parameters
              "square_root" 2 parameters
              "power"       4 parameters
              "interp"      interpolation data  TODO !!!!

    Dmod = {Dmod1, Dmod2, ...}: list of dictionaries of element of the format
           {'redistr. name':  params = 1-rank numpy array}

            with one model per body. Supported redistribution models:
            "none"   0 paramters -> do only reflection
            "global" 0 paramaters
            "local"  1 parameter (h)   h - angle in radians
            "horiz"  4 parameters (o_x, o_y, o_z, h) o_i - unit vector

    Dweight = {Dw1, Dw2, ....}: list of dictionaries with weights of
              different models fo the format
              {'redistr. name': value of float type}

              Note: sum_i value_i = 1

    model : string - name of the reflection model in use
             method in {"Wilson", "Horvat"}

    support: string
              {"triangles","vertices"}
  optionally:

    epsC: float, default 0.00872654 = cos(89.5deg)
          threshold for permitted cos(view-angle)
    epsF: float, default 1e-12
          relative precision of radiosity vector in sense of L_infty norm
    max_iter: integer, default 100
          maximal number of iterations in the solver of the radiosity eq.

 Returns:

    dict - dictionary

  with keywords

    radiosity:
      {F_0, F_1, ...} :
        list of 1-rank numpy array of total radiosities at
        triangles or vertices

    update-exitance:
      {F1_0, F1_1, ...} :
        list of 1-rank numpy array of updated emittances at
        triangles or vertices

  Ref:
  * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect,
    Astrophysical Journal,  356, 613-622, 1990 June
*/

struct Tmesh_radiosity_redistrib_problem_nbody {

  bool
    use,
    stored,
    only_reflection;

  int nb;

  Tsupport_type support;

  std::vector<Tview_factor_nbody<double>> Lmat;

  std::vector<Tredistribution<double>> Dmat;

  Tmesh_radiosity_redistrib_problem_nbody() { clear();}


  void clear(bool _use = false) {

    use = _use;      // whether we use this structure to store

    stored = false;  // true - needs to be set, false - it is already set

    only_reflection = false;

    nb = 0;           // number of bodies

    support = triangles;

    Lmat.clear();

    Dmat.clear();
  }

} __redistrib_problem_nbody;

static PyObject *mesh_radiosity_redistrib_problem_nbody_convex_setup(
  PyObject *self, PyObject *args, PyObject *keywds){

  auto fname = "mesh_radiosity_redistrib_problem_nbody_convex_setup"_s;

  char *kwlist[] = {
    (char*)"use_stored",
    (char*)"reset",
    NULL
  };

 PyObject *o_use_stored, *o_reset;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "O!O!", kwlist,
        &PyBool_Type, &o_use_stored,      // neccesary
        &PyBool_Type, &o_reset)
      ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  bool
    b_use_stored = PyObject_IsTrue(o_use_stored),
    b_reset = PyObject_IsTrue(o_reset);

  if (b_reset) __redistrib_problem_nbody.clear(b_use_stored);

  Py_INCREF(Py_None);

  return Py_None;

}




static PyObject *mesh_radiosity_redistrib_problem_nbody_convex(
  PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "mesh_radiosity_redistrib_problem_nbody_convex"_s;

  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"V",
    (char*)"Tr",
    (char*)"N",
    (char*)"A",
    (char*)"R",
    (char*)"F0",
    (char*)"LDmod",
    (char*)"Dmod",
    (char*)"Dweight",
    (char*)"model",
    (char*)"support",
    (char*)"epsC",
    (char*)"epsF",
    (char*)"max_iter",
    NULL
  };

  int max_iter = 100;         // default value

  double
    epsC = 0.00872654,        // default value
    epsF = 1e-12;             // default value

  PyObject
    *oLDmod, *oDmod, *oDweight, *omodel, *osupport,
    *oV, *oTr, *oN, *oA, *oR, *oF0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!O!O!O!O!O!O!O!O!O!O!|ddi", kwlist,
      &PyList_Type, &oV,         // neccesary
      &PyList_Type, &oTr,
      &PyList_Type, &oN,
      &PyList_Type, &oA,
      &PyList_Type, &oR,
      &PyList_Type, &oF0,
      &PyList_Type, &oLDmod,
      &PyList_Type, &oDmod,
      &PyList_Type, &oDweight,
      &PyString_Type, &omodel,
      &PyString_Type, &osupport,
      &epsC,                     // optional
      &epsF,
      &max_iter)){

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  //
  // Number of bodies
  //

  int nb = PyList_Size(oF0);

  if (nb <= 1){
    raise_exception(fname + "::There seem to just n=" + std::to_string(nb) + " bodies.");
    return NULL;
  }

  //
  // Reading input intensities arrays and reflection : ALWAYS READ
  //

  std::vector<std::vector<double>> F0(nb), R(nb);

  for (int b = 0; b < nb; ++b) {
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oF0, b), F0[b]);
    PyArray_ToVector((PyArrayObject *)PyList_GetItem(oR, b), R[b]);
  }

  //
  // Matrices needed to reflection-redistribution processes
  //
  Tsupport_type support;

  bool only_reflection = true;

  std::vector<Tredistribution<double>> Dmat(nb);

  std::vector<Tview_factor_nbody<double>> Lmat;

  if (__redistrib_problem_nbody.use && __redistrib_problem_nbody.stored) {

    only_reflection = __redistrib_problem_nbody.only_reflection;

    Lmat = __redistrib_problem_nbody.Lmat;
    Dmat = __redistrib_problem_nbody.Dmat;

    support =  __redistrib_problem_nbody.support;

  } else {

    //
    // Reading support type
    //

    {
      char *s = PyString_AsString(osupport);

      switch (fnv1a_32::hash(s)) {
        case "triangles"_hash32: support = triangles; break;
        case "vertices"_hash32: support = vertices; break;

        default:
          raise_exception(fname + "::This support type = " + std::string(s) + "is not supported");
          return NULL;
      }
    }

    //
    // Reading geometry of the bodies
    //

    std::vector<std::vector<T3Dpoint<double>>> V(nb), N(nb);
    std::vector<std::vector<T3Dpoint<int>>> Tr(nb);
    std::vector<std::vector<double>> A(nb);

    for (int b = 0; b < nb; ++b){

      PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oV, b), V[b]);
      PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oN, b), N[b]);
      PyArray_To3DPointVector((PyArrayObject *)PyList_GetItem(oTr, b), Tr[b]);
      PyArray_ToVector((PyArrayObject *)PyList_GetItem(oA, b), A[b]);
    }

    //
    // Calculate redistribution matrices
    //

    struct Tlinear_edge {
      double operator()(const double &x , const double &thresh) const {
        if (std::abs(x) <= thresh) return 1.0 - std::abs(x)/thresh;
        return 0.0;
      }
    };

    {
      Py_ssize_t pos;

      PyObject *o, *key, *value;

      for (int b = 0; b < nb; ++b){

        std::map<fnv1a_32::hash_t, std::vector<double>> Dpars;
        std::map<fnv1a_32::hash_t, double> Dweights;

        // reading redistribution model parameters
        o = PyList_GetItem(oDmod, b);
        pos = 0;
        while (PyDict_Next(o, &pos, &key, &value))
          PyArray_ToVector((PyArrayObject*)value, Dpars[fnv1a_32::hash(PyString_AsString(key))]);

        // reading weights for redistribution model
        o = PyList_GetItem(oDweight, b);
        pos = 0;
        while (PyDict_Next(o, &pos, &key, &value))
          Dweights[fnv1a_32::hash(PyString_AsString(key))] = PyFloat_AsDouble(value);


        if (!Dmat[b].init<Tlinear_edge> (support, V[b], Tr[b], N[b], A[b], Dpars, Dweights)){
           raise_exception(fname + "::Redistribution matrix calculation failed");
           return NULL;
        }

        if (!Dmat[b].is_trivial()) only_reflection = false;
      }
    }
    //
    // Reading LD models
    //

    std::vector<TLDmodel<double>*> LDmod;

    if (!LDmodelFromListOfTuples(oLDmod, LDmod)){
      raise_exception(fname + "::Not able to read LD models");
      return NULL;
    }

    if (verbosity_level>=4){
      int i = 0;
      for (auto && pld: LDmod)
        report_stream  << fname << "::" << i++ << " LD:type=" << pld->type << '\n';
    }


    //
    // Calculate view-factor matrices
    //

    if (support == triangles)
      triangle_mesh_radiosity_matrix_triangles_nbody_convex(V, Tr, N, A, LDmod, Lmat);
    else
      triangle_mesh_radiosity_matrix_vertices_nbody_convex(V, Tr, N, A, LDmod, Lmat);


    for (auto && ld: LDmod) delete ld;
    LDmod.clear();

    //
    // storing matrices if needed
    //

    if (__redistrib_problem_nbody.use) {
      __redistrib_problem_nbody.Lmat = Lmat;
      __redistrib_problem_nbody.Dmat = Dmat;
      __redistrib_problem_nbody.only_reflection = only_reflection;
      __redistrib_problem_nbody.support = support;
      __redistrib_problem_nbody.nb = nb;
      __redistrib_problem_nbody.stored = true;
    }
  }

  //
  // Solving the irradiation equations depending on the model
  //

  std::vector<std::vector<double>> F1, Fout;
  {
    bool st = false;

    char *s = PyString_AsString(omodel);

    switch (fnv1a_32::hash(s)) {

      case "Wilson"_hash32:
        st = only_reflection ?
          solve_radiosity_equation_Wilson_nbody(Lmat, R, F0, Fout):
          solve_radiosity_equation_with_redistribution_Wilson_nbody(Lmat, Dmat, R, F0, F1, Fout);
      break;

      case "Horvat"_hash32:
         st = only_reflection ?
          solve_radiosity_equation_Horvat_nbody(Lmat, R, F0, Fout):
          solve_radiosity_equation_with_redistribution_Horvat_nbody(Lmat, Dmat, R, F0, F1, Fout);
      break;

      default:
        raise_exception(fname +
          "::This radiosity-redistribution model =" +
          std::string(s) + " does not exist");
        return NULL;
    }

    if (only_reflection) F1 = F0;  // nothing happens to exitance !!!!

    if (!st) raise_exception(fname + "::slow convergence");
  }

  PyObject *results = PyDict_New();

  PyObject *oFout = PyList_New(nb), *oF1 = PyList_New(nb);

  for (int b = 0; b < nb; ++b) {
    PyList_SetItem(oFout, b, PyArray_FromVector(Fout[b]));
    PyList_SetItem(oF1, b, PyArray_FromVector(F1[b]));
  }

  PyDict_SetItemString(results, "radiosity", oFout);
  PyDict_SetItemString(results, "update-emittance", oF1);

  Py_DECREF (oFout);
  Py_DECREF (oF1);

  return results;
}
/*
  Calculate an rough approximation of the surface average updated
  exitance F_{0,b}' and radiosity F_{out,b} for both bodies b=A, B
  in a binary system of two spheres separated by distance d

  Python:

    dict = radiosity_redistrib_1dmodel(d, radiusA, reflectA, redistr_typeA,
                                          radiusB, reflectB, redistr_typeB)
  where positional parameters:

    d: float - distance between stars
    radiusA: float - radius of the star A
    reflectA: float - reflection of star A
    F0A: average exitance of star A
    redistr_typeA: int - redistribution type of star A
      0: global - uniform global redistribution
      1: horiz  - horizontal redistribution
      2: local - local redistribution

    radiusB: float - radius of the star B
    reflectb: float - reflection of star B
    F0B: average exitance of star B
    redistr_typeB: int - redistribution type of star B
      0: global - uniform global redistribution
      1: horiz  - horizontal redistribution
      2: local - local redistribution

Returns:

    dict - dictionary

  with keywords

    radiosityA: float - Surface average of radiosity for body A
    update-exitanceA: float - Surface average of updated exitance for body A
    radiosityB: float - Surface average of radiosity for body B
    update-exitanceB: float - Surface average of updated exitance for body B

Example:
  import libphoebe

  d=5
  radiusA=2.
  reflectA=0.3
  F0A=1.0
  redistr_typeA=0

  radiusB=1.
  reflectB=0.7
  F0B=2.0
  redistr_typeB=0

  res= libphoebe.radiosity_redistrib_1dmodel(d,
                                        radiusA, reflectA, F0A, redistr_typeA,
                                        radiusB, reflectB, F0B, redistr_typeB)

 {'update-emittanceB': 2.0410763114593298, 'update-emittanceA': 1.0206982972948087, 'radiosityB': 2.012322893437799, 'radiosityA': 1.014488808106366}

*/
static PyObject *radiosity_redistrib_1dmodel(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "radiosity_redistrib_1dmodel"_s;

  //
  // Reading arguments
  //

 char *kwlist[] = {
    (char*)"d",
    (char*)"radiusA",
    (char*)"reflectA",
    (char*)"F0A",
    (char*)"redistr_typeA",
    (char*)"radiusB",
    (char*)"reflectB",
    (char*)"F0B",
    (char*)"redistr_typeB",
    NULL
  };

  int rtypeA, rtypeB;

  double d, rA, rhoA, F0A, rB, rhoB, F0B;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddddidddi", kwlist,
      &d,
      &rA,
      &rhoA,
      &F0A,
      &rtypeA,
      &rB,
      &rhoB,
      &F0B,
      &rtypeB)
    ){
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }

  double
    /* limb-darkended radosity operator coefficient */
    LldAB = utils::sqr(rA/d)*0.5,
    LldBA = utils::sqr(rB/d)*0.5,

    /* Lambertian radosity operator coefficient */
    LLAB = LldAB,
    LLBA = LldBA,

    DA = (rtypeA == 0 || rtypeA == 1 ? 0.5 : 1),
    DB = (rtypeB == 0 || rtypeB == 1 ? 0.5 : 1),

    /* auxiliary variables */
    GA = LldBA*F0B,
    GB = LldAB*F0A,

    TAB = LldAB*DA*(1 - rhoA) + LLAB*rhoA,
    TBA = LldBA*DB*(1 - rhoB) + LLBA*rhoB,

    det = 1 - TAB*TBA,

    FinA = (GA + TBA*GB)/det,
    FinB = (TAB*GA + GB)/det,

    /* update-exitance: body A */
    F1Ad = F0A + DA*(1 - rhoA)*FinA,
    F1An = F0A + (1 - DA)*(1 - rhoA)*FinA,

    /* update-exitance: body B */
    F1Bd = F0B + DB*(1 - rhoB)*FinB,
    F1Bn = F0B + (1 - DB)*(1 - rhoB)*FinB,

    /* radiosity: body A */
    FoutAd = F1Ad + rhoA*FinA,
    FoutAn = F1An,

    /* radiosity: body B */
    FoutBd = F1Bd + rhoB*FinB,
    FoutBn = F1Bn;

  PyObject *results = PyDict_New();

  PyDict_SetItemStringStealRef(results, "update-emittanceA", PyFloat_FromDouble((F1Ad + F1An)/2));
  PyDict_SetItemStringStealRef(results, "radiosityA", PyFloat_FromDouble((FoutAd + FoutAn)/2));

  PyDict_SetItemStringStealRef(results, "update-emittanceB", PyFloat_FromDouble((F1Bd + F1Bn)/2));
  PyDict_SetItemStringStealRef(results, "radiosityB", PyFloat_FromDouble((FoutBd + FoutBn)/2));

  return results;
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

  auto fname = "roche_central_points"_s;

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
    raise_exception(fname + "::Problem reading arguments.");
    return NULL;
  }

  if (o_centers) b_centers = PyObject_IsTrue(o_centers);
  if (o_cnormals) b_cnormals = PyObject_IsTrue(o_cnormals);
  if (o_cnormgrads) b_cnormgrads = PyObject_IsTrue(o_cnormgrads);


  if (!b_centers && !b_cnormals && !b_cnormgrads) {
     raise_exception(fname + "::Nothing to compute.");
     return NULL;
   }
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
  // Calculate the central points
  //

  std::vector<double> *GatC = 0;

  std::vector<T3Dpoint<double>> *C = 0, *NatC = 0;

  if (b_centers) C = new std::vector<T3Dpoint<double>>;

  if (b_cnormals) NatC = new std::vector<T3Dpoint<double>>;

  if (b_cnormgrads) GatC = new std::vector<double>;

  if (!march.central_points(V, Tr, C, NatC, GatC)){
    raise_exception(fname + "::Problem with projection onto surface.");
    return NULL;
  }

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

    raise_exception("roche_reprojecting_vertices::Problem reading arguments");
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
      2 - searching a point for contact binary case
  Return:
    H: 2-rank numpy array of 3D point on a horizon
*/

static PyObject *roche_horizon(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_horizon"_s;

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
    raise_exception(fname + "::Problem reading arguments");
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
    raise_exception(fname + "::Convergence to the point on horizon failed");
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
   raise_exception(fname + "::Convergence to the point on horizon failed");
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

  auto fname = "rotstar_horizon"_s;

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
    raise_exception(fname + "::Problem reading arguments");
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
    raise_exception(fname + "::Convergence to the point on horizon failed");
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
    raise_exception(fname + "::Convergence to the point on horizon failed");
    return NULL;
  }

  return PyArray_From3DPointVector(H);
}


/*
  C++ wrapper for Python code:

  Calculating the horizon on the rotating star with misalignment at reference value Omega0 of the potential

    Omega (x,y,z; omega, s) = 1/|r| + 1/2 omega^2 |r - s(s*r)|^2

  with
    r = {x, y, z}
    s = {sx, sy, sz}    |s| = 1

  Aligned case is

    s = { 0, 0, 1.}

  Python:

    H = rotstar_misaligned_horizon(v, omega, misalignment, Omega0, <keywords>=<value>)

  with arguments

  positionals: necessary
    v[3] - 1-rank numpy array of floats: direction of the viewer
    omega: float - parameter of the potential
           Note:
           for comparison to Roche model (a=1): omega = F sqrt(1+q),
           for independent star of mass M : omega = angular velocity/sqrt(G M)
    misalignment:  in rotated coordinate system:
      float - angle between spin and orbital angular velocity vectors [rad]
              s = [sin(angle), 0, cos(angle)]
    or in canonical coordinate system:
      1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1

    Omega0: float - value of the potential of the rotating star

  keywords:
    length: integer, default 1000,
      approximate number of points on a horizon

  Return:
    H: 2-rank numpy array of floats -- 3D points on a horizon
*/

static PyObject *rotstar_misaligned_horizon(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "rotstar_misaligned_horizon"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"v",
    (char*)"omega",
    (char*)"misalignment",
    (char*)"Omega0",
    (char*)"length",
    NULL
  };

  int length = 1000;

  PyObject *o_misalignment;

  PyArrayObject *oV;

  double pars[5];

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "O!dOd|i", kwlist,
      &PyArray_Type, &oV,
      pars,
      &o_misalignment,
      pars + 4,
      &length)
  ){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double *view = (double*) PyArray_DATA(oV);

  if (PyFloat_Check(o_misalignment)) {
    double s = std::sin(PyFloat_AsDouble(o_misalignment));

    pars[1] = s;
    pars[2] = 0;
    pars[3] = std::sqrt(1. - s*s);

  } else if (PyArray_Check(o_misalignment)) {

    double *s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    for (int i = 0; i < 3; ++i) pars[1+i] = s[i];

  } else {
    raise_exception(fname + "::This type of misalignment is not supported.");
    return NULL;
  }

  //
  //  Find a point on horizon
  //
  double p[3];

  if (!rot_star::point_on_horizon(p, view, pars[4], pars[0], pars+1)) {
    raise_exception(fname + "::Convergence to the point on horizon failed");
    return NULL;
  }

  //
  // Estimate the step
  //

  double dt = utils::m_2pi*utils::hypot3(p)/length;

  //
  //  Find the horizon
  //

  Thorizon<double, Tmisaligned_rot_star<double>> horizon(pars);

  std::vector<T3Dpoint<double>> H;

  if (!horizon.calc(H, view, p, dt)) {
    raise_exception(fname + "::Convergence to the point on horizon failed");
    return NULL;
  }

  return PyArray_From3DPointVector(H);
}

/*
  C++ wrapper for Python code:

    Calculating the horizon on the generalied Roche lobes with
    misaligned spin and orbit angular velocity vectors.

  Python:

    H = roche_misaligned_horizon(v, q, F, d, misalignment, Omega0, <keywords>=<value>)

  with arguments

  positionals: necessary
    v[3] - 1-rank numpy array of floats: direction of the viewer
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    misalignment:  in rotated coordinate system:
        float - angle between spin and orbital angular velocity vectors [rad]
    or in canonical coordinate system:
        1-rank numpy array of length 3 = [sx, sy, sz]  |s| = 1
    Omega0: float - value of the generalized Kopal potential

  keywords:
    length: integer, default 1000,
      approximate number of points on a horizon
    choice: interr, default 0:
      0 - searching a point on left lobe
      1 - searching a point on right lobe
      2 - searching a point for contact case
  Return:
    H: 2-rank numpy array of 3D point on a horizon
*/

static PyObject *roche_misaligned_horizon(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_misaligned_horizon"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"v",
    (char*)"q",
    (char*)"F",
    (char*)"d",
    (char*)"misalignment",
    (char*)"Omega0",
    (char*)"length",
    (char*)"choice",
    NULL
  };

  PyObject *o_misalignment;

  PyArrayObject *oV;

  double q, F, d, Omega0;

  int
    length = 1000,
    choice  = 0,
    max_iter = 100;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "O!dddOd|ii", kwlist,
        &PyArray_Type, &oV,
        &q, &F, &d, &o_misalignment, &Omega0,
        &length, &choice)
  ){
    raise_exception(fname + "::Problem reading arguments.");
    return NULL;
  }

  //
  // Determine misalignment and find a point on horizon
  //

  bool rotated, ok, aligned;

  double
    theta = 0, *s = 0, p[3],
    *view = (double*)PyArray_DATA(oV);

  if (PyFloat_Check(o_misalignment)) {

    theta = PyFloat_AsDouble(o_misalignment);
    aligned = (std::sin(theta) == 0); // theta ~0, pi => aligned
    ok = misaligned_roche::point_on_horizon(p, view, choice, Omega0, q, F, d, theta, max_iter);
    rotated = true;

  } else if (PyArray_Check(o_misalignment) &&
    PyArray_TYPE((PyArrayObject *) o_misalignment) == NPY_DOUBLE) {

    s = (double*) PyArray_DATA((PyArrayObject*)o_misalignment);
    aligned = (s[0] == 0 && s[1] == 0);
    // we could work with s[0]==0, calculate aligned case make simple
    // rotation around x-axis
    ok = misaligned_roche::point_on_horizon(p, view, choice, Omega0, q, F, d, s, max_iter);
    rotated = false;
  } else {
    raise_exception(fname + "::This type of misalignment is not supported.");
    return NULL;
  }

  if (!ok) {
    raise_exception(fname + "::Convergence to the point on horizon failed.");
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
  std::vector<T3Dpoint<double>> H;
  if (aligned) {
    double params[] = {q, F, d, Omega0};
    Thorizon<double, Tgen_roche<double>> horizon(params);
    ok = horizon.calc(H, view, p, dt);
  } else {
    if (rotated) {
      double params[] = {q, F, d, theta, Omega0};
      Thorizon<double, Tmisaligned_rotated_roche<double>> horizon(params);
      ok = horizon.calc(H, view, p, dt);
    } else {
      double params[] = {q, F, d, s[0], s[1], s[2], Omega0};
      Thorizon<double, Tmisaligned_roche<double>> horizon(params);
      ok = horizon.calc(H, view, p, dt);
    }
  }

  if (!ok) {
    raise_exception(fname + "::Finding horizon failed.");
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
      2 - searching a point for contact binary case

  Return:
    xrange: 1-rank numpy array of two numbers p
*/

static PyObject *roche_xrange(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_xrange"_s;

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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice < 0 || choice > 2) {
    raise_exception(fname + "::This choice of computation is not supported");
    return NULL;
  }

  double xrange[2];

  if (!gen_roche::lobe_xrange(xrange, choice, Omega0, q, F, d, true)){
    raise_exception(fname + "::Determining lobe's boundaries failed");
    return NULL;
  }

  return PyArray_FromVector(2, xrange);
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
      2 - searching a point for contact binary case

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

  auto fname = "roche_square_grid"_s;

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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice < 0 || choice > 2) {
    raise_exception(fname + "::This choice is not supported");
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
      raise_exception(fname + "::This type of dims is not supported");
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
    raise_exception(fname + "::Failed to obtain xrange");
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

    // extend the ranges for better subdivision

  for (int i = 0; i < 3; ++i){
    ranges[i][0] += 0.1*ranges[i][0];
    ranges[i][1] -= 0.1*ranges[i][0];
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

  npy_intp nd[3];
  for (int i = 0; i < 3; ++i) nd[i] = dims[i];

  #if defined(USING_SimpleNewFromData)
  std::uint8_t *mask = new std::uint8_t [size];
  PyObject *o_mask = PyArray_SimpleNewFromData(3, nd, NPY_UINT8, mask);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_mask, NPY_ARRAY_OWNDATA);
  #else
  PyObject *o_mask = PyArray_SimpleNew(3, nd, NPY_UINT8);
  std::uint8_t *mask = (uint8_t *)PyArray_DATA((PyArrayObject*)o_mask);
  #endif

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

  nd[0] = 3;
  nd[1] = 2;

  #if defined(USING_SimpleNewFromData)
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

  PyObject
    *o_origin = PyArray_SimpleNewFromData(1, nd, NPY_DOUBLE, origin),
    *o_steps = PyArray_SimpleNewFromData(1, nd, NPY_DOUBLE, steps),
    *o_bbox = PyArray_SimpleNewFromData(2, nd, NPY_DOUBLE, bbox);

  PyArray_ENABLEFLAGS((PyArrayObject *)o_origin, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_steps, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_bbox, NPY_ARRAY_OWNDATA);

  // storing the mask

  #else

  PyObject
    *o_origin = PyArray_SimpleNew(1, nd, NPY_DOUBLE),
    *o_steps = PyArray_SimpleNew(1, nd, NPY_DOUBLE),
    *o_bbox = PyArray_SimpleNew(2, nd, NPY_DOUBLE);

  double
    *origin = (double*) PyArray_DATA((PyArrayObject *)o_origin),
    *steps = (double*) PyArray_DATA((PyArrayObject *)o_steps),
    *bbox = (double*) PyArray_DATA((PyArrayObject *)o_bbox);

  for (int i = 0; i < 3; ++i){
    origin[i] = r0[i];
    steps[i] = L[i];
    bbox[2*i] = ranges[i][0];
    bbox[2*i + 1] = ranges[i][1];
  }
  #endif

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
              "power"       4 parameters
    params: 1-rank numpy array
  Return:
    value of D(mu) for a given LD model
*/

static PyObject *ld_D(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "ld_D"_s;

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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  TLDmodel_type type = LD::type(PyString_AsString(o_descr));

  if (type == NONE) {
    raise_exception(fname + "::This model is not supported");
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
              "power "      4 parameters
    params: 1-rank numpy array

  Return:
    value of integrated D(mu) for a given LD model
*/

static PyObject *ld_D0(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "ld_D0"_s;

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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  TLDmodel_type type = LD::type(PyString_AsString(o_descr));

  if (type == NONE) {
    raise_exception(fname + "::This model is not supported");
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
          "power"       4 parameters

    params: 1-rank numpy array

  Return:
    1-rank numpy array of floats: gradient of the function D(mu) w.r.t. parameters
*/

static PyObject *ld_gradparD(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "ld_gradparD"_s;

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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  TLDmodel_type type = LD::type(PyString_AsString(o_descr));

  if (type == NONE) {
    raise_exception(fname + "::This model is not supported");
    return NULL;
  }

  int nr_par = LD::nrpar(type);

  npy_intp dims = nr_par;

  #if defined(USING_SimpleNewFromData)
  double *g = new double [nr_par];
  LD::gradparD(type, mu, (double*)PyArray_DATA(o_params), g);
  PyObject *results = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, g);
  PyArray_ENABLEFLAGS((PyArrayObject *)results, NPY_ARRAY_OWNDATA);
  #else
  PyObject *results = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
  double *g = (double *)PyArray_DATA((PyArrayObject *)results);
  LD::gradparD(type, mu, (double*)PyArray_DATA(o_params), g);
  #endif

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
            "power"       4 parameters
  Return:
    int: number of parameters
*/

static PyObject *ld_nrpar(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "ld_nrpar"_s;

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
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  TLDmodel_type type = LD::type(PyString_AsString(o_descr));

  if (type == NONE) {
    raise_exception(fname + "::This model is not supported");
    return NULL;
  }

  return PyInt_FromLong(LD::nrpar(type));
}


/*
  C++ wrapper for Python code:

    Check the parameters of the particular limb darkening model

  Python:

    value = ld_check(descr, params, strict=False)

  with arguments

    descr: string (bytes)
          supported ld models:
            "uniform"     0 parameters
            "linear"      1 parameters
            "quadratic"   2 parameters
            "nonlinear"   3 parameters
            "logarithmic" 2 parameters
            "square_root" 2 parameters
            "power"       4 parameters

    params: 1-rank numpy array of float

  optional

    strict: Boolean, default False

    strict checking: if D(mu) in [0,1] for all mu in [0,1]
    loose (non-strict): if D(mu) >=0 for all mu in [0,1]

  Return:
    true: if parameters pass the checks, false otherwise

  Example:

  import numpy as np
  import libphoebe

  print ld_check("nonlinear", np.array([2.0, 0.2, 3]))
  print ld_check("logarithmic", np.array([0., 0.]))
  print ld_check("logarithmic", np.array([0.6, -1]))
  print ld_check("logarithmic", np.array([0.6, -2]))

  False
  True
  True
  False

*/

static PyObject *ld_check(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "ld_check"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"descr",
    (char*)"params",
    (char*)"strict",
    NULL
  };

  bool strict = false;

  PyObject *o_descr, *o_strict = 0;

  PyArrayObject *o_params;

  if (!PyArg_ParseTupleAndKeywords(args, keywds,  "O!O!|O!", kwlist,
        &PyString_Type, &o_descr,
        &PyArray_Type,  &o_params,
        &PyBool_Type,   &o_strict )
      ){
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (o_strict) strict = PyObject_IsTrue(o_strict);

  TLDmodel_type type = LD::type(PyString_AsString(o_descr));

  if (type == NONE) {
    raise_exception(fname + "::This model is not supported");
    return NULL;
  }

  if (strict)
    return PyBool_FromLong(LD::check_strict(type, (double*)PyArray_DATA(o_params)));

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
  auto fname ="wd_readdata"_s;
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

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  npy_intp planck_dims = wd_atm::N_planck, atm_dims = wd_atm::N_atm;

  #if defined(USING_SimpleNewFromData)
  double
    *planck_table = new double[wd_atm::N_planck],
    *atm_table = new double[wd_atm::N_atm];

  PyObject
    *py_planck = PyArray_SimpleNewFromData(1, &planck_dims, NPY_DOUBLE, planck_table),
    *py_atm = PyArray_SimpleNewFromData(1, &atm_dims, NPY_DOUBLE, atm_table);

  PyArray_ENABLEFLAGS((PyArrayObject *)py_planck, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)py_atm, NPY_ARRAY_OWNDATA);
  #else
  PyObject
    *py_planck = PyArray_SimpleNew(1, &planck_dims, NPY_DOUBLE),
    *py_atm = PyArray_SimpleNew(1, &atm_dims, NPY_DOUBLE);

  double
    *planck_table = (double*)PyArray_DATA((PyArrayObject *)py_planck),
    *atm_table = (double*)PyArray_DATA((PyArrayObject *)py_atm);
  #endif

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
    raise_exception(fname + "::Problem reading data");
    delete [] planck_table;
    delete [] atm_table;
    return NULL;
  }

  //
  // Returning results
  //

  PyObject *results = PyDict_New();
  PyDict_SetItemStringStealRef(results, "planck_table", py_planck);
  PyDict_SetItemStringStealRef(results, "atm_table", py_atm);

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
    raise_exception("wd_planckint::Problem reading arguments");
    return NULL;
  }

  //
  // Calculate without checks
  //

  double y[2];
  wd_atm::planckint(t, ifil,
                    (double*) PyArray_DATA(oplanck_table),
                    y[0], y[1]);


  return PyArray_FromVector(2,y);
}
#else
/*
  C++ wrapper for Python code:

    Computing the logarithm of the Planck central intensity. Works for
    temperatures in the range [500,500300] K.

  Python:

    result = wd_planckint(t, ifil, planck_table)

  Minimal testing script:

    import libphoebe as lph
    import numpy as np

    planck="..../wd/atmcofplanck.dat"
    atm="..../wd/atmcof.dat"

    d = lph.wd_readdata(planck, atm)

    temps = np.array([1000., 2000.])

    print lph.wd_planckint(temps, 1, d["planck_table"])

    returns:

    [-0.28885608  8.45013452]

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

  auto fname = "wd_planckint"_s;

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

    raise_exception(fname + "::Problem reading arguments");

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
      raise_exception(fname + "::Failed to calculate Planck central intensity");
      return PyFloat_FromDouble(std::numeric_limits<double>::quiet_NaN());
    }

  } else if (
    PyArray_Check(ot) &&
    PyArray_TYPE((PyArrayObject *) ot) == NPY_DOUBLE
    ) {  // argument is a numpy array of float(double)

    int n = PyArray_DIM((PyArrayObject *)ot, 0);

    if (n == 0) {
      raise_exception(fname + "::Arrays of zero length");
      return NULL;
    }

    double *t =  (double*) PyArray_DATA((PyArrayObject *)ot);

    //
    // Prepare space for results
    //

    npy_intp dims = n;

    #if defined(USING_SimpleNewFromData)
    double *results = new double [n];
    PyObject *oresults = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, results);
    PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
    #else
    PyObject *oresults = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    double *results = (double *)PyArray_DATA((PyArrayObject *)oresults);
    #endif

    //
    //  Calculate ylog for an array
    //

    bool ok = true;

    for (double *r = results, *r_e = r + n; r != r_e;  ++r, ++t)
      if (!wd_atm::planckint_onlylog(*t, ifil, planck_table, *r)) {
        *r = std::numeric_limits<double>::quiet_NaN();
        ok = false;
      }

    if (!ok)
      raise_exception(fname + "::Failed to calculate Planck central intensity at least once");

    return oresults;
  }

  raise_exception(fname + "::This type of temperature input is not supported");
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

    raise_exception("wd_atmint::Problem reading arguments\n");
    return NULL;
  }

  //
  // Calculate without checks
  //

  double y[3];

  y[2] = abunin;

  wd_atm::atmx(t, logg, y[2], ifil,
              (double*)PyArray_DATA(oplanck_table),
              (double*)PyArray_DATA(oatm_table),
              y[0], y[1]);

  return PyArray_FromVector(3, y);
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

  auto fname = "wd_atmint"_s;

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
    raise_exception(fname + "::Problem reading arguments\n");
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

  } else if (
    PyArray_Check(ot) &&
    PyArray_TYPE((PyArrayObject *) ot) == NPY_DOUBLE
  ) {

    n = PyArray_DIM((PyArrayObject *)ot, 0);

    if (n == 0) {
      raise_exception(fname + "::Arrays are of zero length");
      return NULL;
    }

    // arguments
    pt = (double*)PyArray_DATA((PyArrayObject *)ot),
    plogg = (double*)PyArray_DATA((PyArrayObject *)ologg),
    pabunin = (double*)PyArray_DATA((PyArrayObject *)oabunin);

  } else {
    raise_exception(fname + "::This type of temperature input is not supported");
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

      // prepare numpy array to store the results
      npy_intp dims = 2;

      #if defined(USING_SimpleNewFromData)
      double *r = new double[2];
      oresults = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, r);
      PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
      #else
      oresults = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
      double *r = (double*)PyArray_DATA((PyArrayObject *)oresults);
      #endif

      r[1] = abunin;

      // do calculation
      if (!wd_atm::atmx_onlylog(t, logg, r[1], ifil, planck_table, atm_table, r[0])) {
        raise_exception(fname + "::Failed to calculate logarithm of intensity");
        r[0] = std::numeric_limits<double>::quiet_NaN();
      }

    } else {  // calculation whole array


      // prepare numpy array to store the results
      npy_intp dims[2] = {n, 2};

      #if defined(USING_SimpleNewFromData)
      double *results = new  double [2*n]; // to store results
      oresults = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, results);
      PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
      #else
      oresults = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
      double *results = (double*)PyArray_DATA((PyArrayObject *)oresults);
      #endif

      bool ok = true;
      for (double *r = results, *r_e = r + 2*n; r != r_e; r += 2, ++pt, ++plogg, ++pabunin){

        r[1] = *pabunin;

        if (!wd_atm::atmx_onlylog(*pt, *plogg, r[1], ifil, planck_table, atm_table, r[0])) {
          r[0] = std::numeric_limits<double>::quiet_NaN();
          ok = false;
        }
      }

      if (!ok)
        raise_exception(fname + "::Failed to calculate logarithm of intensity at least once");
    }


  } else {                    // returning only logarithm of intensities

    //
    //  Calculate yintlogs
    //

    if (n == -1){ // single calculation

      double r; // log of intensity

      if (wd_atm::atmx_onlylog(t, logg, abunin, ifil, planck_table, atm_table, r))
        oresults = PyFloat_FromDouble(r);
      else {
        raise_exception(fname + "::Failed to calculate logarithm of intensity");
        oresults = PyFloat_FromDouble(std::numeric_limits<double>::quiet_NaN());
      }

    } else { // calculation whole array

      // prepare numpy array to store the results
      npy_intp dims = n;

      #if defined(USING_SimpleNewFromData)
      double *results = new double [n];
      oresults = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, results);
      PyArray_ENABLEFLAGS((PyArrayObject *)oresults, NPY_ARRAY_OWNDATA);
      #else
      oresults = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
      double *results = (double*)PyArray_DATA((PyArrayObject *)oresults);
      #endif

      double tmp;

      bool ok = true;

      for (double *r = results, *r_e = r + n; r != r_e; ++r, ++pt, ++plogg, ++pabunin){

        tmp = *pabunin;

        if (!wd_atm::atmx_onlylog(*pt, *plogg, tmp, ifil, planck_table, atm_table, *r)) {
          *r = std::numeric_limits<double>::quiet_NaN();
          ok = false;
        }
      }

      if (!ok)
        raise_exception(fname + "::Failed to calculate logarithm of intensity at least once");
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
    char *kwlist[] = {
        (char*)"req",
        (char*)"axes",
        (char*)"grid",
        NULL
    };

    PyObject *o_axes;

    // PyObject *o_req, *o_grid;
    PyArrayObject *o_req, *o_grid;

    if (!PyArg_ParseTupleAndKeywords(
          args, keywds, "O!O!O!", kwlist,
          &PyArray_Type, &o_req,
          &PyTuple_Type, &o_axes,
          &PyArray_Type, &o_grid))
        {
          raise_exception("interp::argument type mismatch: req and grid need to be numpy arrays and axes a tuple of numpy arrays.");
          return NULL;
        }

    // if (!PyArg_ParseTuple(args, "OOO", &o_req, &o_axes, &o_grid)) {
    //     raise_exception("arguments for interp(req, axes, grid) could not be parsed.");
    //     return NULL;
    // }

    // if (!PyArray_Check(o_req)) {
    //     raise_exception("argument `req` should be a numpy array.");
    // }

    // if (!PyArray_Check(o_grid)) {
    //     raise_exception("argument `grid` should be a numpy array.");
    // }

    // if (!PyArray_Check(o_axes) && !PyList_Check(o_axes) && !PyTuple_Check(o_axes)) {
    //     raise_exception("argument `axes` should be a numpy array, a list or a tuple.");
    // }

     PyArrayObject
      *o_req1 = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)o_req, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
      *o_grid1 = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)o_grid, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    // PyArrayObject
    //     *o_req1 = (PyArrayObject *) PyArray_FromObject((PyObject *) o_req, NPY_DOUBLE, 0, 0),
    //     *o_grid1 = (PyArrayObject *) PyArray_FromObject((PyObject *) o_grid, NPY_DOUBLE, 0, 0);

    if (!o_req1 ||!o_grid1) {
        if (!o_req1) raise_exception("argument `req` is not a correctly shaped numpy array.");
        if (!o_grid1) raise_exception("argument `grid` is not a correctly shaped numpy array.");

        Py_DECREF(o_req1);
        Py_DECREF(o_grid1);
        return NULL;
    }

    /* number of axes: */
    int Na;
    if (PyList_Check(o_axes))
        Na = PyList_Size(o_axes);
    else if (PyTuple_Check(o_axes))
        Na = PyTuple_Size(o_axes);
    else /* if (PyArray_Check(o_axes)) */
        Na = PyArray_DIM((PyArrayObject *) o_axes, 0);

    int Np = PyArray_DIM(o_req1, 0),     /* number of points */
        Nv = PyArray_DIM(o_grid1, Na),   /* number of values interpolated */
        Nr = Np*Nv;                      /* number of returned values */

    double
        *Q = (double *) PyArray_DATA(o_req1),  // requested values
        *G = (double *) PyArray_DATA(o_grid1); // grid of values

    // Unpack the axes
    int *L = new int [Na];      // number of values in axes
    double **A = new double* [Na]; // pointers to tuples in axes

    {
        PyArrayObject *p;
        for (int i = 0; i < Na; ++i) {
            if (PyList_Check(o_axes))
                p = (PyArrayObject *) PyList_GET_ITEM(o_axes, i); // no checks, borrows reference
            else if (PyTuple_Check(o_axes))
                p = (PyArrayObject *) PyTuple_GET_ITEM(o_axes, i); // no checks, borrows reference
            else /* if (PyArray_Check(o_axes)) */
                /* handle me */
                p = (PyArrayObject *) o_axes;

            L[i] = (int) PyArray_DIM(p, 0);
            A[i] = (double *) PyArray_DATA(p);
        }
    }

  //
  // Prepare for returned values
  //
  npy_intp dims[2] = {Np, Nv};

  #if defined(USING_SimpleNewFromData)
  double *R = new double [Nr];
	PyObject *o_ret = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, R);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_ret, NPY_ARRAY_OWNDATA);
  #else
	PyObject *o_ret = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  double *R = (double *) PyArray_DATA((PyArrayObject *)o_ret);
  #endif

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

  return o_ret;
}

/*
  Calculate cosine of the angle of scalar projections

    r[i] = x[i].y[i]/(|x[i]| |y[i]|)    i = 0, ..., n -1

  Input:
    x : 2-rank numpy array
    y : 2-rank numpy array

  Return:
    r: 1- rank numpy array
*/
static PyObject *scalproj_cosangle(PyObject *self, PyObject *args) {

  auto fname = "vec_proj"_s;

  PyArrayObject *o_x, *o_y;

  if  (!PyArg_ParseTuple(args,
        "O!O!",
        &PyArray_Type, &o_x,
        &PyArray_Type, &o_y)
      ){
    raise_exception(fname +  "::Problem reading arguments");
    return NULL;
  }

  int n = PyArray_DIM(o_x, 0);

  npy_intp dims = n;

  double
    s, x, y,
    *px = (double*)PyArray_DATA(o_x),
    *py = (double*)PyArray_DATA(o_y);

  #if defined(USING_SimpleNewFromData)
  double *r = new double [n];
  PyObject *o_r = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, r);
  PyArray_ENABLEFLAGS((PyArrayObject *)o_r, NPY_ARRAY_OWNDATA);
  #else
  PyObject *o_r = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
  double *r = (double *)PyArray_DATA((PyArrayObject *)o_r);
  #endif

  for (double *p = r, *pe = r + n; pe != p; ++p) {

    s = x = y = 0;
    for (int i = 0; i < 3; ++i, ++px, ++py) {
     s += (*px)*(*py);
     x += (*px)*(*px);
     y += (*py)*(*py);
    }

    *p = s/std::sqrt(x*y);
  }

  return o_r;
}


/*
  C++ wrapper for Python code:

  Calculate the area, volume and dvolume/dOmega of the left or right
  side of the "Roche" contact lobe defined as equipotential of the generalized
  Kopal potential Omega:

      Omega_0 = Omega(x,y,z)

  assuming F = 1. We divide the Roche contact lobe on sides by cutting
  it with a plane x=const.

  Python:

    dict = roche_contact_partial_area_volume(x, q, d, Omega0, <keyword>=<value>)

  where parameters are

  positionals:
    x: float - plane dividing left and right side
    q: float = M2/M1 - mass ratio
    d: float - separation between the two objects
    Omega: float - value potential

  keywords:
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe

    larea: boolean, default True
    lvolume: boolean, default True
    ldvolume: boolean, default True

    epsA : float, default 1e-12
      relative precision of the area

    epsV : float, default 1e-12
      relative precision of the volume

    epsdV : float, default 1e-12
      relative precision of the dvolume/dOmega

  Returns:

    dictionary

  with keywords
    larea: area of the left or right Roche lobe
      float:

    lvolume: volume of the left or right Roche lobe
      float:

    ldvolume: dvolume/dOmega of the left or right Roche lobe
      float:

    Example:
    import libphoebe

    x=0.7       # where we cut it
    choice = 0  # 0 for left and 1 for right
    q=0.1
    Omega0=1.9
    d=1.

    res=libphoebe.roche_contact_partial_area_volume(x, q, d, Omega0, choice)

    {'larea': 4.587028506379938, 'lvolume': 0.9331872042603445, 'ldvolume': -2.117861555286342}
*/


static PyObject *roche_contact_partial_area_volume(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_contact_partial_area_volume"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"x",
    (char*)"q",
    (char*)"d",
    (char*)"Omega0",
    (char*)"choice",
    (char*)"larea",
    (char*)"lvolume",
    (char*)"ldvolume",
    (char*)"epsA",
    (char*)"epsV",
    (char*)"epsdV",
    NULL};

  int choice = 0;

  double eps[3] = {1e-12, 1e-12, 1e-12};

  bool b_r[3] = {true, true, true}; // b_larea, b_lvolume, b_ldvolume

  PyObject *o_r[3] = {0, 0, 0};     // *o_larea = 0, *o_lvolume = 0, *o_ldvolume

  double x, q, d, Omega0;

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "dddd|iO!O!O!ddd", kwlist,
      &x, &q, &d, &Omega0,  // necessary
      &choice,
      &PyBool_Type, o_r,
      &PyBool_Type, o_r + 1,
      &PyBool_Type, o_r + 2,
      eps, eps + 1, eps + 2
      )
    ) {

    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  unsigned res_choice = 0;

  //
  // Read boolean variables and define result-choice
  //
  for (int i = 0, j = 1; i < 3; ++i, j <<= 1) {
    if (o_r[i]) b_r[i] = PyObject_IsTrue(o_r[i]);
    if (b_r[i]) res_choice += j;
  }

  if (res_choice == 0) {
    raise_exception(fname + "::Nothing is computed.");
    return NULL;
  }

  if (choice != 0 && choice != 1){
    raise_exception(fname + "::This choice of sides is not possible.");
    return NULL;
  }

  if (verbosity_level >=4)
    report_stream << fname
      << ":: q=" << q
      << " d=" << d
      << " Omega0=" << Omega0
      << " res_choice=" << res_choice << '\n';

  //
  // Choosing boundaries on x-axis
  //

  double xrange0[2], xrange[2];

  if (!gen_roche::lobe_xrange(xrange0, 2, Omega0, q, 1., d, true)){
    raise_exception(fname + "::Determining lobe's boundaries failed");
    return NULL;
  }

  if (verbosity_level >=4)
    report_stream << fname + "::xrange=" << xrange0[0] << ':' <<  xrange0[1] << '\n';

  if (x < xrange0[0] || xrange0[1] < x) {
    raise_exception(fname + "::Plane cutting lobe is outside xrange.");
    return NULL;
  }

  if (choice == 0) {
    xrange[0] = xrange0[0];
    xrange[1] = x;
  } else {
    xrange[0] = x;
    xrange[1] = xrange0[1];
  }

  //
  // Calculate area and volume:
  //

  const int m_min = 1 << 6;  // minimal number of points along x-axis

  int m0 = m_min,            // starting number of points alomg x-axis
      dir = (choice == 0 ? 1 : -1);

  bool
    polish = false,
    adjust = true;

  double r[3], p[2][3], e;

  //
  // one step adjustment of precison for area and volume
  // calculation
  //

  do {

    for (int i = 0, m = m0; i < 2; ++i, m <<= 1) {

      gen_roche::area_volume_directed_integration(p[i], res_choice, dir, xrange, Omega0, q, 1., d, m, polish);

      if (verbosity_level >=4)
        report_stream  << fname + "::m=" << m << " P:" << p[i][0] << '\t' << p[i][1] << '\t' << p[i][2] << '\n';
    }

    // best approximation
    for (int i = 0; i < 3; ++i) if (b_r[i]) {
      r[i] = (16*p[1][i] - p[0][i])/15;

      if (verbosity_level >=4)
        report_stream  << fname << "::B:" << i << ":" << r[i] << '\n';
    }

    if (adjust) {

      // extrapolation based on assumption
      //   I = I0 + a_1 h^4
      // estimating errors. This seems to be valid for well behaved functions.

      int m0_next = m0;

      adjust = false;

      for (int i = 0; i < 3; ++i) if (b_r[i]) {

        // relative error
        e = std::max(std::abs(p[0][i]/r[i] - 1), 16*std::abs(p[1][i]/r[i] - 1));

        if (verbosity_level >=4)
          report_stream  << fname << "::err=" << e << " m0=" << m0 << '\n';

        if (e > eps[i]) {
          int k = int(1.1*m0*std::pow(e/eps[i], 0.25));
          if (k > m0_next) {
            m0_next = k;
            adjust = true;
          }
        }
      }

      if (adjust) m0 = m0_next;
    }

  } while (adjust);

  PyObject *results = PyDict_New();

  const char *str[3] =  {"larea", "lvolume", "ldvolume"};

  for (int i = 0; i < 3; ++i) if (b_r[i])
    PyDict_SetItemStringStealRef(results, str[i], PyFloat_FromDouble(r[i]));

  return results;
}




/*
  C++ wrapper for Python code:

  Calculate the minimal distance r of the neck from x axis of the contact
  Roche lobe at angle phi from y axis:

      Omega_0 = Omega(x, r cos(phi), r sin(phi))


  assuming F = 1.

  Python:

    dict = roche_contact_neck_min(phi, q, d, Omega0)

  where parameters are

  positionals:
    q: float = M2/M1 - mass ratio
    d: float - separation between the two objects
    Omega: float - value potential
    phi: angle
      for minimal distance in
        xy plane phi = 0
        xz plane phi = pi/2

  Returns:
      dictionary

  with keywords

    rmin: minimal distance
      float:

    xmin: position of minimum
      float:

  Example:
    import numpy as np
    import libphoebe

    q=0.1
    d=1.
    Omega0 = 1.9

    for phi in np.linspace(0,np.pi/2, 4):
      print libphoebe.roche_contact_neck_min(phi, q, d, Omega0)


    {'xmin': 0.742892957853368, 'rmin': 0.14601804638933566}
    {'xmin': 0.7415676153921865, 'rmin': 0.14223055177447497}
    {'xmin': 0.7393157248476556, 'rmin': 0.13553553766381343}
    {'xmin': 0.7383492639142092, 'rmin': 0.13255145166593718}

*/


static PyObject *roche_contact_neck_min(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_contact_neck_min"_s;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"phi",
    (char*)"q",
    (char*)"d",
    (char*)"Omega0",
    NULL};


  double q, d, Omega0, phi;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "dddd", kwlist,
        &phi, &q, &d, &Omega0)
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  double u[2];

  if (!contact::neck_min(u, std::cos(phi), q, d, Omega0)) {
    raise_exception(fname + "::Slow convergence");
    return NULL;
  }

  PyObject *results = PyDict_New();

  PyDict_SetItemStringStealRef(results, "xmin", PyFloat_FromDouble(u[0]));
  PyDict_SetItemStringStealRef(results, "rmin", PyFloat_FromDouble(u[1]));

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

    Omega1 = roche_contact_Omega_at_partial_vol(vol, phi, q, d, <keyword>=<value>)

  where parameters are

  positionals:
    vol: float - volume of the Roche lobe
    q: float = M2/M1 - mass ratio
    d: float - separation between the two objects

  keywords: (optional)
    Omega0: float - guess for value potential Omega1
    choice: integer, default 0
            0 for discussing left lobe
            1 for discussing right lobe
    precision: float, default 1e-12
      aka relative precision
    accuracy: float, default 1e-12
      aka absolute precision
    max_iter: integer, default 100
      maximal number of iterations in the Newton-Raphson

  Returns:

    Omega1 : float
      value of the Kopal potential for (q,F,d) at which the lobe has the given volume
*/

static PyObject *roche_contact_Omega_at_partial_vol(PyObject *self, PyObject *args, PyObject *keywds) {

  auto fname = "roche_contact_Omega_at_partial_vol"_s;

  if (verbosity_level>=4)
    report_stream << fname << "::START" << std::endl;

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"vol",
    (char*)"phi",
    (char*)"q",
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
    vol, phi, q, d,
    Omega0 = nan(""),
    precision = 1e-12,
    accuracy = 1e-12;

  int max_iter = 20;

  if (!PyArg_ParseTupleAndKeywords(
        args, keywds,  "dddd|diddi", kwlist,
        &vol, &phi, &q, &d,  // necessary
        &Omega0,
        &choice,
        &precision,
        &accuracy,
        &max_iter
      )
    ) {
    raise_exception(fname + "::Problem reading arguments");
    return NULL;
  }

  if (choice != 0 && choice != 1){
    raise_exception(fname + "::This choice of sides is not possible.");
    return NULL;
  }

  //
  //  Getting minimal volume and maximal permitted Omega
  //
  if (verbosity_level>=4)
    report_stream << fname + "::calculate minimal critical volume ...\n";

  double buf[3], Omega_min, Omega_max, vol_min, vol_max;

  if (!gen_roche::critical_area_volume(2, (choice == 0 ? q : 1/q), 1., d, Omega_max, buf)){
    raise_exception(fname + "::Determining lobe's boundaries failed");
    return NULL;
  }

  vol_min = buf[1];

  //
  //  Getting maximal volume and minimal permitted Omega
  //

  if (verbosity_level>=4)
    report_stream << fname << "::calculate maximal critical volume ...\n";

  double OmegaC[3], L[3];

  gen_roche::critical_potential(OmegaC, L, 2+4, q, 1., d);

  Omega_min = std::max(OmegaC[1], OmegaC[2]);

  if (verbosity_level >=4)
    report_stream << fname + "::L2=" << L[1] << " L3=" <<  L[2] << '\n';

  double cos_phi = std::cos(phi), xrange[2], u[2], b = d*d*d*(1 + q);

  if (!contact::neck_min(u, cos_phi, q, d, Omega_min)){
    raise_exception(fname + "::Calculating neck minimum failed. 1.");

    if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

    return NULL;
  }

  if (u[0] < L[1] || L[2] < u[0]) {
    raise_exception(fname + "::Plane cutting lobe is outside [L2,L3].");

    if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

    return NULL;
  }

  if (choice == 0) {
    xrange[0] = (OmegaC[1] > OmegaC[2] ? L[1] : d*gen_roche::left_lobe_left_xborder(d*Omega_min, q, b));
    xrange[1] = u[0];
  } else {
    xrange[0] = u[0];
    xrange[1] = (OmegaC[1] < OmegaC[2] ? L[2] : d*gen_roche::right_lobe_right_xborder(d*Omega_min, q, b));
  }

  int dir = (choice == 0 ? 1 : -1);
  gen_roche::area_volume_directed_integration(buf, 2, dir, xrange, Omega_min, q, 1., d, 1 << 14);

  vol_max = buf[1];

  if (vol < vol_min  || vol > vol_max){
    raise_exception(fname + "::Volume is outside bounds.");

    if (verbosity_level >= 2)
      report_stream << fname + "::vol=" << vol << " vol_min=" << vol_min << " vol_max=" << vol_max << '\n';

    if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

    return NULL;
  }

  if (verbosity_level >= 4)
    report_stream << fname
      << "::Omega_min=" << Omega_min << " Omega_max=" << Omega_max
      << " vol_min=" << vol_min << " vol_max=" << vol_max << '\n';

  //
  // If Omega0 is not set, we estimate it
  //

  if (std::isnan(Omega0)) {

   double f = (vol - vol_min)/(vol_max - vol_min);

   Omega0 = Omega_min*f + Omega_max*(1 - f);
  }

  //
  // Checking estimate of the Omega0
  //
  if (Omega0 < Omega_min || Omega0 > Omega_max) {
    raise_exception(fname + "::The estimated Omega is outside bounds.");

    if (verbosity_level >= 2)
      report_stream << fname + "::Omega0=" << Omega0 << " Omega_min=" << Omega_min << " Omega_max=" << Omega_max << '\n';

    if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

    return NULL;
  }

  if (verbosity_level >= 4)
      report_stream
        << fname + "::vol=" << vol << " q=" << q << " Omega0=" << Omega0
        << " d=" << d << " choice=" << choice << std::endl;

  //
  // Trying to calculate Omega at given volume
  //
  const int m_min = 1 << 8;  // minimal number of points along x-axis

  int
    m0 = m_min,  // minimal number of points along x-axis
    it = 0;      // number of iterations

  // expected precisions of the integrals
  double eps = precision/2;

  // adaptive calculation of the volume
  // permitting adjustment just once as it not necessary stable
  bool adjust = true;

  double v[2], w[2], p[2], t;

   // first step of secant method
  if (Omega0 - Omega_min < Omega_max - Omega0) {
    v[0] = vol_max - vol;
    w[0] = Omega_min;
  } else {
    v[0] = vol_min - vol;
    w[0] = Omega_max;
  }

  w[1] = Omega0;

  do {

    if (!contact::neck_min(u, cos_phi, q, d, w[1])){
      raise_exception(fname + "::Calculating neck minimum failed.2");

      if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

      return NULL;
    }

    if (choice == 0) {
      xrange[0] = d*gen_roche::left_lobe_left_xborder(d*w[1], q, b);
      xrange[1] = u[0];
    } else {
      xrange[0] = u[0];
      xrange[1] = d*gen_roche::right_lobe_right_xborder(d*w[1], q, b);
    }

    if (std::isnan(xrange[0]) || std::isnan(xrange[1])) {
      raise_exception(fname + "::Determining lobe's boundaries failed");

      if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

      return NULL;
    }

    //
    // calculate the volume at w[1]
    //
    do {


      for (int i = 0, m = m0; i < 2; ++i, m <<= 1) {
        gen_roche::area_volume_directed_integration(buf, 2, dir, xrange, w[1], q, 1., d, m);

        p[i] = buf[1];
      }

      // extrapolations based on the expansion
      // I = I0 + a1 h^4 + a2 h^5 + ...
      // result should have relative precision better than 1e-12

      v[1] = (16*p[1] - p[0])/15;

      if (adjust) {

        // relative error
        double e = std::max(std::abs(p[0]/v[1] - 1), 16*std::abs(p[1]/v[1] - 1));

        if (e > eps)
          m0 = int(1.1*m0*std::pow(e/eps, 0.25));
        else adjust = false;

        if (verbosity_level>=4)
          report_stream << fname << "::m=" <<  m0 << " V=" << v[1] << " e =" << e << '\n';
      }
    } while (adjust);

    // interested only in volume - <target volume>
    v[1] -= vol;

    // secant method step
    t = w[1] - v[1]*(w[1] - w[0])/(v[1] - v[0]);

    v[0]  = v[1];
    w[0] = w[1];
    w[1] = t;

    if (verbosity_level>=4)
      report_stream
        << fname + "::Omega=" << w[0] << " dV=" << v[0]
        << " dOmega=" << w[1] - w[0]  << " Omega*=" << w[1]<< '\n';

  } while (std::abs(w[0] - w[1]) > accuracy + precision*w[1] && ++it < max_iter);

  if (it >= max_iter){
    raise_exception(fname + "::Maximum number of iterations exceeded");

    if (verbosity_level>=4) report_stream << fname + "::END" << std::endl;

    return NULL;
  }

  if (verbosity_level>=4)
    report_stream << fname << "::END" << std::endl;

  return PyFloat_FromDouble(w[1]);
}


/*
  Computes monochromatic blackbody intensity in W/m^3 using the
  Planck function:

    intensity = planck_function (lam, Teff)

    B_\lambda (\lambda ,T)= \frac {2hc^2}{\lambda^5}\frac {1}{e^{\frac {hc}{\lambda k_{\mathrm {B} }T}}-1}

  Input:
    lam : wavelength in m
      float
    or
      1- rank numpy array

    Teff: effective temperature in K
      float
    or
      1- rank numpy array

  Returns: monochromatic blackbody intensity:
      float : if lam and Teff are float
    or
      1-rank numpy array : if lam or Teff are 1-rank numpy arrays
    or
      2-rank numpy array : if lam and Teff are 1-rank numpy arrays
*/
static PyObject *planck_function(PyObject *self, PyObject *args) {

  const double A = 1.1910429526245747e-16; // = 2 h c^2 [m4 kg / s3];
  const double B = 0.014387773538277205;   // = hc/k [mK];

  const char *fname = "planck_function";

  //
  // Reading arguments
  //

  PyObject *o_lam, *o_Teff;

  if (!PyArg_ParseTuple(args, "OO", &o_lam, &o_Teff)) {
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }

  //
  // Read lambdas
  //
  int n_lam = -1;
  double *p_lam, lam;

  if (PyFloat_Check(o_lam)) {
    lam =  PyFloat_AS_DOUBLE(o_lam);
    p_lam = &lam;
  } else if (PyArray_Check(o_lam)) {
    n_lam = PyArray_DIM((PyArrayObject *)o_lam, 0);
    p_lam = (double*)PyArray_DATA((PyArrayObject *)o_lam);
  } else {
    std::cerr << fname << ":: This type of input of lambdas is not supported\n";
    return NULL;
  }

  //
  // Read tempeatures
  //

  int n_Teff = -1;
  double *p_Teff, Teff;

  if (PyFloat_Check(o_Teff)) {
    Teff = PyFloat_AS_DOUBLE(o_Teff);
    p_Teff = &Teff;
  } else if (PyArray_Check(o_Teff)) {
    n_Teff = PyArray_DIM((PyArrayObject*)o_Teff, 0);
    p_Teff = (double*)PyArray_DATA((PyArrayObject *)o_Teff);
  } else {
    std::cerr << fname << ":: This type of input of Teff is not supported\n";
    return NULL;
  }

  //
  // if both arguments are float the result if float
  //
  if (n_lam < 0 && n_Teff < 0)
    return (lam == 0 ? 0 : PyFloat_FromDouble(A/std::pow(lam,5)/(std::exp(B/(lam*Teff)) - 1)));

  //
  // At least one of the arguments is numpy array and
  // the result is numpy array
  //
  npy_intp dims[2];
  PyObject *o_r;

  if (n_lam < 0 && n_Teff > 0) {        // Teff is array => result is array
    n_lam = 1;
    dims[0] = n_Teff;
    o_r = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  } else if (n_lam > 0 && n_Teff < 0) { // lam is array => result is array
    n_Teff = 1;
    dims[0] = n_lam;
    o_r = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  } else { // both are arrays => => result is a matrix
    dims[0] = n_lam;
    dims[1] = n_Teff;
    o_r = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  }

  double
    tmp, tmp2,
    *r = (double *)PyArray_DATA((PyArrayObject *)o_r);

  for (int i = 0; i < n_lam; ++i) {
    lam = p_lam[i];
    if (lam != 0) {
      tmp = A/std::pow(lam, 5);
      tmp2 = B/lam;
      for (int j = 0; j < n_Teff; ++j)
        *(r++) = tmp/(std::exp(tmp2/p_Teff[j]) - 1);
    } else for (int j = 0; j < n_Teff; ++j) *(r++) = 0;
  }

  return o_r;
}

/*
  Computing CCM89 extinction value as a value of wavelength

  Input:
    lam: wavelength in m
      float
    or
      1- rank numpy array

  Returns: extinction coefficients:
      1-rank numpy array: two values
    or
      2-rank numpy array: array of two values
*/
static PyObject *CCM89_extinction(PyObject *self, PyObject *args) {

  const char *fname = "CCM89_extinction";

  //
  // Reading arguments
  //

  PyObject *o_lam;

  if (!PyArg_ParseTuple(args, "O", &o_lam)) {
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }

  //
  // Reading variables and reserving space for results
  //

  int n;
  npy_intp dims[2];
  PyObject *o_r;
  double *l, lam;

  if (PyFloat_Check(o_lam)) {
    n = 1;
    lam = PyFloat_AS_DOUBLE(o_lam);
    l = &lam;

    dims[0] = 2;
    o_r = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  } else if (PyArray_Check(o_lam)) {
    n = PyArray_DIM((PyArrayObject*)o_lam, 0);
    l = (double*)PyArray_DATA((PyArrayObject *)o_lam);

    dims[0] = n;
    dims[1] = 2;
    o_r = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

  } else {
    std::cerr << fname << ":: This type of input of lambdas is not supported\n";
    return NULL;
  }


  //
  // Calculating results
  //


  double
    x, y, y2,
    *r = (double *)PyArray_DATA((PyArrayObject *)o_r);

  do {

    x = 1e-6/(*(l++));

    if (0.3 <= x && x <= 1.1) {
      y = std::pow(x, 1.61);
      *(r++) = 0.574*y;
      *(r++) = -0.527*y;
    } else if (x <= 3.3) {
      y = x - 1.82;
      //ax = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
      *(r++) = 1 + y*(0.17699 + y*(-0.50447 + y*(-0.02427 + y*(0.72085 + y*(0.01979 + (-0.7753 + 0.32999*y)*y)))));
      //bx = 1.141338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
      *(r++) = y*(1.41338 + y*(2.28305 + y*(1.07233 + y*(-5.38434 + y*(-0.62251 + (5.3026 - 2.09002*y)*y)))));
    } else if (x <= 5.9) {
      *(r++) = 1.752 - 0.316*x - 0.104/(utils::sqr(x - 4.67) + 0.341);
      *(r++) = -3.090 + 1.825*x + 1.206/(utils::sqr(x - 4.62) + 0.263);
    } else if (x <= 8.0) {
      y = x - 5.9;
      y2 = y*y;
      *(r++) = 1.752 - 0.316*x - 0.104/(utils::sqr(x - 4.67) + 0.341) - (0.04473 + 0.009779*y)*y2;
      *(r++) =- 3.090 + 1.825*x + 1.206/(utils::sqr(x - 4.62) + 0.263) + (0.2130 + 0.1207*y)*y2;
    } else if (x <= 10) {
      y = x - 8;
      //ax = -1.073 - 0.628*y + 0.137*y**2 - 0.070*y**3;
      *(r++) = -1.073 + y*(-0.628 + (0.137 - 0.07*y)*y);
      //bx = 13.670 + 4.257*y + 0.420*y**2 + 0.374*y**3
      *(r++) = 13.67 + y*(4.257 + (0.42 + 0.374*y)*y);
    } else {
      std::cerr
        << fname
        << "Passband wavelength outside the range defined for CCM89 extinction (0.1-3.3 micron)\n";
      return NULL;
    }
  } while (--n);

  return o_r;
}


/*
  Computing Gordon et al. (2009) extinction value as a value of wavelength

  Input:
    lam: wavelength in m
      float
    or
      1- rank numpy array

  Returns: extinction coefficients:
      1-rank numpy array: two values
    or
      2-rank numpy array: array of two values
*/
static PyObject *gordon_extinction(PyObject *self, PyObject *args) {

  const char *fname = "gordon_extinction";

  //
  // Reading arguments
  //

  PyObject *o_lam;

  if (!PyArg_ParseTuple(args, "O", &o_lam)) {
    std::cerr << fname << "::Problem reading arguments\n";
    return NULL;
  }

  //
  // Reading variables and reserving space for results
  //

  int n;
  npy_intp dims[2];
  PyObject *o_r;
  double *l, lam;

  if (PyFloat_Check(o_lam)) {
    n = 1;
    lam = PyFloat_AS_DOUBLE(o_lam);
    l = &lam;

    dims[0] = 2;
    o_r = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  } else if (PyArray_Check(o_lam)) {
    n = PyArray_DIM((PyArrayObject*)o_lam, 0);
    l = (double*)PyArray_DATA((PyArrayObject *)o_lam);

    dims[0] = n;
    dims[1] = 2;
    o_r = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

  } else {
    std::cerr << fname << ":: This type of input of lambdas is not supported\n";
    return NULL;
  }


  //
  // Calculating results
  //


  double
    x, y, x59square,
    *r = (double *)PyArray_DATA((PyArrayObject *)o_r);

  do {

    x = 1e-6/(*(l++));

    if (0.3 <= x && x <= 1.1) {
      y = std::pow(x, 1.61);
      *(r++) = 0.574*y;
      *(r++) = -0.527*y;
    } else if (x <= 3.3) {
      y = x - 1.82;
      //ax = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
      *(r++) = 1 + y*(0.17699 + y*(-0.50447 + y*(-0.02427 + y*(0.72085 + y*(0.01979 + (-0.7753 + 0.32999*y)*y)))));
      //bx = 1.141338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
      *(r++) = y*(1.41338 + y*(2.28305 + y*(1.07233 + y*(-5.38434 + y*(-0.62251 + (5.3026 - 2.09002*y)*y)))));
    } else if (x <= 5.9) {
      *(r++) = 1.894 - 0.373*x - 0.0101/((x - 4.57)*(x - 4.57) + 0.0384);
      *(r++) = -3.490 + 2.057*x + 0.706/((x - 4.59)*(x - 4.59) + 0.0169);
    } else if (x <= 11.0) {
      x59square=(x - 5.9)*(x - 5.9);
      *(r++) = 1.894 - 0.373*x - 0.0101/((x - 4.57)*(x - 4.57) + 0.0384) - 0.110*x59square - 0.0100*x59square*(x - 5.9);
      *(r++) = -3.490 + 2.057*x + 0.706/((x - 4.59)*(x - 4.59) + 0.0160) + 0.531*x59square + 0.0544*x59square*(x - 5.9);
    } else {
      std::cerr
        << fname
        << "Passband wavelength outside the range defined for CCM89 and Gordon et al. (2009) extinction (0.1-3.3 micron)\n";
      return NULL;
    }
  } while (--n);

  return o_r;
}


/*
  Define functions in module

  Some modification in declarations due to use of keywords
  Ref:
  * https://docs.python.org/2.0/ext/parseTupleAndKeywords.html
*/
static PyMethodDef Methods[] = {

 { "roche_misaligned_transf",
    roche_misaligned_transf,
    METH_VARARGS,
    "Determine angle parameters of the misaligned Roche lobes from "
    "the spin angular velocity in the rotating binary system "},

// --------------------------------------------------------------------


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

  { "rotstar_misaligned_critical_potential",
    rotstar_misaligned_critical_potential,
    METH_VARARGS,
    "Determine the critical potentials of the rotating star potental "
    "with misalignment for given values of omega and spin"},

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

  { "rotstar_misaligned_pole",
    (PyCFunction)rotstar_misaligned_pole,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the height of the pole of rotating star with misalignment "
    "for given a omega and spin."},

  { "sphere_pole",
    (PyCFunction)sphere_pole,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the height of the pole of sphere for given a R."},

  { "roche_misaligned_pole",
    (PyCFunction)roche_misaligned_pole,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the postion of the pole of generalized Roche lobes with "
    "misaligned angular spin-orbital angular velocity vectors for given "
    "values of q, F, d, misalignment(theta or direction) and Omega0."},

// --------------------------------------------------------------------

  {"roche_Omega_min",
    (PyCFunction)roche_Omega_min,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the minimal posible value of the Kopal potential that"
    "permits existance of the compact Roche lobe for given "
    "values of q, F and d."},

  { "roche_misaligned_Omega_min",
    (PyCFunction)roche_misaligned_Omega_min,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the minimal posible value of the Kopal potential that"
    "permits existance of the compact Roche lobe for given "
    "values of q, F, d and misalignment (theta or direction)."},

// --------------------------------------------------------------------
  { "roche_misaligned_critical_volume",
    (PyCFunction)roche_misaligned_critical_volume,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the volume of the semi-detached case of the misaligned "
    "Roche lobe for given values of q, F, F and misalignment (theta or "
    "direction)"},
// --------------------------------------------------------------------

  { "rotstar_from_roche",
    (PyCFunction)rotstar_from_roche,
    METH_VARARGS|METH_KEYWORDS,
    "Determine parameters of the rotating stars from parameters Roche "
    " by matching the poles"},


  { "rotstar_misaligned_from_roche_misaligned",
    (PyCFunction)rotstar_misaligned_from_roche_misaligned,
    METH_VARARGS|METH_KEYWORDS,
    "Determine parameters of the rotating stars with misalignment from "
    "parameters Roche with misalignment by matching the poles."},

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

  { "rotstar_misaligned_area_volume",
    (PyCFunction)rotstar_misaligned_area_volume,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the area and volume of the rotating star with misalignment "
    "for given a omega and Omega0"},

  { "sphere_area_volume",
    (PyCFunction)sphere_area_volume,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the area and volume of the sphere for given a R."},

  { "roche_misaligned_area_volume",
    (PyCFunction)roche_misaligned_area_volume,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the area and volume of the generalized Roche lobes with "
    "misaligned spin and orbtal angular velocity vectors for given "
    "values of q, F, d, misalignment(theta or direction) and Omega0."},

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

   { "rotstar_misaligned_Omega_at_vol",
    (PyCFunction)rotstar_misaligned_Omega_at_vol,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the value of the rotating star potential with misalignment at "
    "values of omega and volume."},

   { "roche_misaligned_Omega_at_vol",
    (PyCFunction)roche_misaligned_Omega_at_vol,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the value of the generalized Kopal potential of "
    "Roche lobes with with misaligned spin and orbtal angular "
    "velocity vectors at values of q, F, d, misalignment(theta or direction) "
    "and volume."},

  { "sphere_Omega_at_vol",
    (PyCFunction)sphere_Omega_at_vol,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the value of the spherical star potential at given volume."},

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

  { "rotstar_misaligned_gradOmega",
    rotstar_misaligned_gradOmega,
    METH_VARARGS,
    "Calculate the gradient and the value of the rotating star potential "
    "with misalignment at given point [x,y,z] for given values of omega "
    "and spin."},

  { "sphere_gradOmega",
    sphere_gradOmega,
    METH_VARARGS,
    "Calculate the gradient of the potential of the sphere"
    " at given point [x,y,z]."},

  { "roche_misaligned_gradOmega",
    roche_misaligned_gradOmega,
    METH_VARARGS,
    "Calculate the gradient of the generalized Kopal potential with "
    " misaligned angular momenta at given point [x,y,z] for given "
    " values of q, F, d and misalignment(theta or direction)"},

// --------------------------------------------------------------------

  { "roche_Omega",
    roche_Omega,
    METH_VARARGS,
    "Calculate the value of the generalized Kopal potential"
    " at given point [x,y,z] for given values of q, F and d."},

  { "rotstar_Omega",
    rotstar_Omega,
    METH_VARARGS,
    "Calculate the value of the rotating star potential"
    " at given point [x,y,z] for given values of omega."},


   { "rotstar_misaligned_Omega",
    rotstar_misaligned_Omega,
    METH_VARARGS,
    "Calculate the value of the rotating star potential with misalignment"
    " at given point [x,y,z] for given values of omega and spin"},

  { "sphere_Omega",
    sphere_Omega,
    METH_VARARGS,
    "Calculate the value of the potential of the sphere "
    " at given point [x,y,z]."},

  { "roche_misaligned_Omega",
    roche_misaligned_Omega,
    METH_VARARGS,
    "Calculate the value of the generalized Kopal potential with "
    " misaligned angular velocity vectors at given point [x,y,z] for given "
    " values of q, F, d and misalignment(theta or direction)"},

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


  { "rotstar_misaligned_gradOmega_only",
    rotstar_misaligned_gradOmega_only,
    METH_VARARGS,
    "Calculate the gradient of the rotating star potential with misalignment"
    " at given point [x,y,z] for given values of omega and spin"},

  { "sphere_gradOmega_only",
    sphere_gradOmega_only,
    METH_VARARGS,
    "Calculate the gradient of the potential of the sphere"
    " at given point [x,y,z]."},

  { "roche_misaligned_gradOmega_only",
    roche_misaligned_gradOmega_only,
    METH_VARARGS,
    "Calculate the gradient of the generalized Kopal potential with "
    " misaligned angular momenta at given point [x,y,z] for given "
    " values of q, F, d and misalignment(theta or direction)"},

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

  { "rotstar_misaligned_marching_mesh",
    (PyCFunction)rotstar_misaligned_marching_mesh,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the triangular meshing of a rotating star with misalignment "
    "for given values of omega, spin and value of the star potential Omega."
    "The edge of triangles used in the mesh are approximately delta."},

  { "sphere_marching_mesh",
    (PyCFunction)sphere_marching_mesh,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the triangular meshing of a sphere for given radius R."
    "The edge of triangles used in the mesh are approximately delta."},

  { "roche_misaligned_marching_mesh",
    (PyCFunction)roche_misaligned_marching_mesh,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the triangular meshing of generalized Roche lobes with "
    "misaligned spin and orbital angular velocity vectors for "
    "given values of q, F, d, misalignment(theta or direction) and value"
    "of the generalized Kopal potential Omega0. The edge of triangles "
    "used in the mesh are approximately delta."},

// --------------------------------------------------------------------

  { "mesh_visibility",
    (PyCFunction)mesh_visibility,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the ratio of triangle surfaces that are visible "
    "in a triangular mesh."},

  { "mesh_rough_visibility",
    mesh_rough_visibility,
    METH_VARARGS,
    "Classify the visibility of triangles of the mesh into hidden, "
    "partially hidden and visible"},

  { "mesh_offseting",
    (PyCFunction)mesh_offseting,
    METH_VARARGS|METH_KEYWORDS,
    "Offset the mesh along the normals in vertices to match the "
    "area with reference area."},

  { "mesh_properties",
    (PyCFunction)mesh_properties,
    METH_VARARGS|METH_KEYWORDS,
    "Calculate the properties of the triangular mesh."},

  { "mesh_export_povray",
    (PyCFunction)mesh_export_povray,
    METH_VARARGS|METH_KEYWORDS,
    "Exporting triangular mesh into a Pov-Ray file."},

  { "mesh_radiosity_problem",
    (PyCFunction)mesh_radiosity_problem,
    METH_VARARGS|METH_KEYWORDS,
    "Solving the radiosity problem with limb darkening using "
    "a chosen reflection model."},

  { "mesh_radiosity_problem_nbody_convex",
    (PyCFunction)mesh_radiosity_problem_nbody_convex,
    METH_VARARGS|METH_KEYWORDS,
    "Solving the radiosity problem with limb darkening for n separate "
    "convex bodies using chosen reflection model."},

   { "mesh_radiosity_redistrib_problem_nbody_convex",
    (PyCFunction)mesh_radiosity_redistrib_problem_nbody_convex,
    METH_VARARGS|METH_KEYWORDS,
    "Solving the radiosity redistribution problem with limb darkening "
    "for n separate convex bodies using chosen reflection model."},

   { "mesh_radiosity_redistrib_problem_nbody_convex_setup",
    (PyCFunction)mesh_radiosity_redistrib_problem_nbody_convex_setup,
    METH_VARARGS|METH_KEYWORDS,
    "Background setup of radiosity redistribution problem with limb "
    "darkening for n separate convex bodies using chosen reflection model."},

{ "radiosity_redistrib_1dmodel",
    (PyCFunction)radiosity_redistrib_1dmodel,
    METH_VARARGS|METH_KEYWORDS,
    "Calculating a rough approximate of the surface average updated-exitance "
    "and radiosity for both bodies of a binary system composed of two spheres."},
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

    { "rotstar_misaligned_horizon",
    (PyCFunction)rotstar_misaligned_horizon,
    METH_VARARGS|METH_KEYWORDS,
    "Calculating the horizon on the rotating star with misalignment "
    "defined by view direction, omega, spin and the value of the potential"},

  { "roche_misaligned_horizon",
    (PyCFunction)roche_misaligned_horizon,
    METH_VARARGS|METH_KEYWORDS,
    "Calculating the horizon on the Roche lobe with misaligned spin and orbital "
    "angular velocity vectors defined by the view direction,"
    "q,F,d, theta and the value of generalized Kopal potential Omega."},

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
    (PyCFunction)interp,
    METH_VARARGS|METH_KEYWORDS,
    "Multi-dimensional linear interpolation of arrays with gridded data."},

// --------------------------------------------------------------------

  {"scalproj_cosangle",
    scalproj_cosangle,
    METH_VARARGS,
    "Calculate normalized projections of vectors."},

// --------------------------------------------------------------------

  {"roche_contact_partial_area_volume",
    (PyCFunction)roche_contact_partial_area_volume,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the area, volume and dvolume/dOmega of the Roche contact lobe"
    "at given mass ratio q, separation d and and Omega0"},

  {"roche_contact_neck_min",
    (PyCFunction)roche_contact_neck_min,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the minimal distance and position from x axis of the neck "
    "of the Roche contact lobe at given mass ratio q, separation d and Omega0"},

   {"roche_contact_Omega_at_partial_vol",
    (PyCFunction)roche_contact_Omega_at_partial_vol,
    METH_VARARGS|METH_KEYWORDS,
    "Determine the value of the potential at a partial volume for the contact "
    "Roche lobe at given mass ratio q and separation d."},

// --------------------------------------------------------------------


 {"planck_function",
  planck_function,
  METH_VARARGS,
  "Calculate monochromatic blackbody intensity at a given wavelength and "
  "temperature."},

   {"CCM89_extinction",
  CCM89_extinction,
  METH_VARARGS,
  "Calculate CCM89 extinction coefficients for a given wavelength"},

  {"gordon_extinction",
  gordon_extinction,
  METH_VARARGS,
  "Calculate Gordon et al. (2009, UV) and CCM89 (OPT-IR) extinction coefficients for a given wavelength"},


// --------------------------------------------------------------------

  {"setup_verbosity",
    (PyCFunction)setup_verbosity,
    METH_VARARGS|METH_KEYWORDS,
    "Setting the verbosity of libphoebe"},

  {NULL,  NULL, 0, NULL} // terminator record
};

static const char *Name = "libphoebe";

static const char *ExceptionName = "libphoebe.error";

static const char *Docstring =
  "Module wraps routines dealing with models of stars and "
  "triangular mesh generation and their manipulation.";


/* module initialization */
MOD_INIT(libphoebe) {

  PyObject *backend;

  MOD_DEF(backend, Name, Docstring, Methods)

  if (!backend) return MOD_ERROR_VAL;

  struct module_state *st = MOD_GETSTATE(backend);

  MOD_NEW_EXCEPTION(st->error, ExceptionName)

  if (st->error == NULL) {
    Py_DECREF(backend);
    return MOD_ERROR_VAL;
  }

  // Added to handle Numpy arrays
  // Ref:
  // * http://docs.scipy.org/doc/numpy-1.10.1/user/c-info.how-to-extend.html
  import_array();

  return MOD_SUCCESS_VAL(backend);
}
