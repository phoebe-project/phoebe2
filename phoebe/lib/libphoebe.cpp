/* 
  Wrappers for calculations concerning the generalized Kopal potential or
  Roche lobes.
   
  
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
  
  Author: Martin Horvat, June 2016
*/

#include <iostream>
#include <vector>
#include <typeinfo>
#include <algorithm>

#include "gen_roche.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
 
// providing the Python interface -I/usr/include/python2.7/
#include <Python.h>
#include <numpy/arrayobject.h>

/*
  Creating PyArray from std::vector.
  
  Ref: 
  http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
*/ 
template <class T>
PyObject  *PyArray_SimpleNewFromVector(
  int np, 
  npy_intp *dims, 
  int typenum, 
  void *V){
  
  int size = dims[0];
  for (int i = 1; i < np; ++i) size *= dims[i];
  size *= sizeof(T);
  
  // Note: p is interpreted in C-style contiguous fashion
  void *p = malloc(size);
  memcpy(p, V, size);
  
  return PyArray_SimpleNewFromData(np, dims, typenum, p); 
}

/*
  Getting the Python typename 
*/

template<typename T> NPY_TYPES PyArray_TypeNum();

template<>  NPY_TYPES PyArray_TypeNum<int>() { return NPY_INT;}
template<>  NPY_TYPES PyArray_TypeNum<double>() { return NPY_DOUBLE;}

template <typename T>
PyObject *PyArray_FromVector(std::vector<T> &V){
  
  int N = V.size();
  
  T *p = new T [N];
  
  std::copy(V.begin(), V.end(), p);
  
  npy_intp dims[1] = {N};
  
  return PyArray_SimpleNewFromData(1, dims, PyArray_TypeNum<T>(), p); 
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
  
  return PyArray_SimpleNewFromData(2, dims, PyArray_TypeNum<T>(), p); 
}


template <typename T>
void PyArray_To3DPointVector(PyArrayObject *oV, std::vector<T3Dpoint<T>> &V){
   
  // Note: p is interpreted in C-style contiguous fashion
  int N = PyArray_DIM(oV, 0);
  
  V.reserve(N);
  
  for (auto *p = (T*) PyArray_DATA(oV), p_e = p + 3*N; p != p_e; p += 3)
    V.emplace_back(p);
}

/*
  Python wrapper for C++ code:
  
  Calculate critical potential, that is the value of the generalized 
  Kopal potential in Lagrange points.
  
  Python:
    
    omega = roche_critical_potential(q, F, d)
  
  where parameters are
  
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
  
  and returns
  
    omega: 1-rank numpy array of 3 floats
*/

static PyObject *roche_critical_potential(PyObject *self, PyObject *args) {
    
  // parse input arguments   
  double q, F, delta;
  
  if (!PyArg_ParseTuple(args, "ddd", &q, &F, &delta)) return NULL;
      
  // calculate critical potential
  double *omega = new double[3];
  gen_roche::critical_potential(omega, q, F, delta);
  
  // return the results
  npy_intp dims[1] = {3};

  return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, omega);
}

/*
  Python wrapper for C++ code:
  
  Calculate point on x-axis 
  
    (x,0,0)
  
  of the generalized Roche lobes define by  generalized Kopal potential:
    
      Omega(x,y,z) = Omega0
  
  Python:
    
    points=roche_points_on_x_axis(q, F, d, Omega0)
  
  where parameters are
  
    q: float = M2/M1 - mass ratio
    F: float - synchronicity parameter
    d: float - separation between the two objects
    Omega0: float - separation between the two objects
     
  
  and returns
  
    points : 1-rank numpy array of floats
*/

static PyObject *roche_points_on_x_axis(PyObject *self, PyObject *args) {
    
  // parse input arguments   
  double q, F, delta, Omega0;
  
  if (!PyArg_ParseTuple(args, "dddd", &q, &F, &delta, &Omega0))
    return NULL;
      
  // calculate x points
  std::vector<double> points;
  gen_roche::points_on_x_axis(points, Omega0, q, F, delta);

  return PyArray_FromVector(points);
}

/*
  Python wrapper for C++ code:
  
  Calculate height h the left (right) Roche lobe at the position 
    
    (0,0,h)   (  (d,0,h) )
    
  of the primary (secondary) star. Roche lobe(s) is defined as equipotential 
  of the generalized Kopal potential Omega:

      Omega_0 = Omega(x,y,z)
  
  Python:
    
    h = roche_pole(q, F, d, Omega0, choice=0)
  
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
      args, keywds,  "dddd|i", kwlist, &q, &F, &delta, &Omega0, &choice))
      return NULL;
  
  //
  // Compute the poles
  //   
  if (choice == 0)  // Left lobe
    return PyFloat_FromDouble(gen_roche::poleL(Omega0, q, F, delta));
  
  // Right lobe  
  return PyFloat_FromDouble(gen_roche::poleR(Omega0, q, F, delta));
}

/*
  Python wrapper for C++ code:
  
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
    
  Tgen_roche<double> b;
   
  PyArrayObject *X;
  
  if (!PyArg_ParseTuple(args, "dddO!", &b.q, &b.F, &b.delta, &PyArray_Type, &X))
    return NULL;

  b.Omega0 = 0;

  double *g = new double [4];

  b.grad((double*)PyArray_DATA(X), g);
  
  npy_intp dims[1] = {4};

  return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
}
 
/*
  Python wrapper for C++ code:
  
  Calculate the gradient of the potential of the generalized
  Kopal potential Omega at a given point

      -grad Omega (x,y,z)
  
  Python:
    
    g = roche_gradOmega(q, F, d, r)
   
   with parameters
      q: float = M2/M1 - mass ratio
      F: float - synchronicity parameter
      d: float - separation between the two objects
      r: 1-rank numpy array of length 3 = [x,y,z]
   
  
  and returns float
  
    g : 1-rank numpy array = -grad Omega (x,y,z)
*/


static PyObject *roche_gradOmega_only(PyObject *self, PyObject *args) {

  Tgen_roche<double> b;
   
  PyArrayObject *X;
  
  if (!PyArg_ParseTuple(args, "dddO!", &b.q, &b.F, &b.delta, &PyArray_Type, &X))
    return NULL;

  double *g = new double [3];

  b.grad_only((double*)PyArray_DATA(X), g);

  npy_intp dims[1] = {3};

  return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, g);
}

/*
  Python wrapper for C++ code:

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
  
      vertices: boolean, default False
      vnormals: boolean, default False
      triangles: boolean, default False
      tnormals: boolean, default False
      areas: boolean, default False
      area: boolean, default False
      volume: boolean, default False
      centers: boolean, default False
      cnormals: boolean, default False
      cnormgrads: boolean, default False
      vnormgrads: boolean, default False
   
    
  Returns:
  
    dictionary
  
  with keywords
  
    vertices: 
      V[][3]    - 2-rank numpy array of a pairs of vertices 
    
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
    NULL};
  
  double q, F, d, Omega0, delta;   
  
  int choice = 0,               
      max_triangles = 10000000; // 10^7
      
  bool 
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

  if (!PyArg_ParseTupleAndKeywords(
      args, keywds,  "ddddd|iiiiiiiiiiiii", kwlist,
      &q, &F, &d, &Omega0, &delta, // neccesary 
      &choice,                     // optional ...
      &max_triangles,
      &b_vertices, 
      &b_vnormals,
      &b_vnormgrads,
      &b_triangles, 
      &b_tnormals,
      &b_centers,
      &b_cnormals,
      &b_cnormgrads,
      &b_areas,
      &b_area,
      &b_volume 
      ))
    return NULL;
     
  //
  // Storing results in dictioonary
  // https://docs.python.org/2/c-api/dict.html
  
  PyObject *results = PyDict_New();
  
        
  //
  // Points on the x - axis 
  //
  
  std::vector<double> x_points;
    
  gen_roche::points_on_x_axis(x_points, Omega0, q, F, d);

  if (x_points.size() == 0) return NULL;
  
  //
  // Chosing the lobe (left or right or overcontact) 
  //
  
  double x0; 
  
  if (choice == 0) {    // left lobe
    x0 = x_points.front();
    if (x0 > 0) return NULL;
  } else {              // right lobe
    x0 = x_points.back();
    if (x0 < d) return NULL;
  }
  
  //
  //  Marching triangulation of the Roche lobe 
  //
    
  double params[5] = {q, F, d, Omega0, x0};
  
  Tmarching<double, Tgen_roche<double>> march(params);  
  
  std::vector<T3Dpoint<double>> V, NatV;
  std::vector<T3Dpoint<int>> Tr; 
  std::vector<double> *GatV = 0;
   
  if (b_vnormgrads) GatV = new std::vector<double>;
  
  if (!march.triangulize(delta, max_triangles, V, NatV, Tr, GatV)){
    std::cerr << "There is too much triangles\n";
    return NULL;
  }
  
  if (b_vertices)
    PyDict_SetItemString(results, "vertices", PyArray_From3DPointVector(V));

  if (b_vnormals)
    PyDict_SetItemString(results, "vnormals", PyArray_From3DPointVector(NatV));

  if (b_vnormgrads) {
    PyDict_SetItemString(results, "vnormgrads", PyArray_FromVector(*GatV));
    delete GatV;
  }
  
  if (b_triangles)
    PyDict_SetItemString(results, "triangles", PyArray_From3DPointVector(Tr));

  
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

  if (b_areas) {
    PyDict_SetItemString(results, "areas", PyArray_FromVector(*A));
    delete A;  
  }
  
  if (b_area)
    PyDict_SetItemString(results, "area", PyFloat_FromDouble(area));

  if (b_tnormals) {
    PyDict_SetItemString(results, "tnormals", PyArray_From3DPointVector(*NatT));
    delete NatT;
  }

  if (b_volume)
    PyDict_SetItemString(results, "volume", PyFloat_FromDouble(volume));

  //
  // Calculte the central points
  // 

  std::vector<double> *GatC = 0;
  
  std::vector<T3Dpoint<double>> *C = 0, *NatC = 0;
  
  if (b_centers) C = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormals) NatC = new std::vector<T3Dpoint<double>>;
 
  if (b_cnormgrads) GatC = new std::vector<double>;
 
  march.central_points(V, Tr, C, NatC, GatC);
  
  if (b_centers) {
    PyDict_SetItemString(results, "centers", PyArray_From3DPointVector(*C));
    delete C;  
  }

  if (b_cnormals) {
    PyDict_SetItemString(results, "cnormals", PyArray_From3DPointVector(*NatC));
    delete NatC;
  }
  
  if (b_cnormgrads) {
    PyDict_SetItemString(results, "cnormgrads", PyArray_FromVector(*GatC));
    delete GatC;
  }
  
  return results;
}

/*
  Python wrapper for C++ code:

    Calculation of visibility of triangles
    
  Python:

    dict = triangle_mesh_visibility(v, V, T, N, <keyword> = <value>)
    
  with arguments
  
    viewdir[3] - 1-rank numpy array of 3 coordinates representing 3D point
    V[][3] - 2-rank numpy array of vertices  
    T[][3] - 2-rank numpy array of indices of vertices composing triangles
    N[][3] - 2-rank numpy array of normals of triangles
 
    (optional)   
    tvisibilities: boolean, default True
    taweights: boolean, default False 
          
  Returns: dictionary with keywords
   
  keywords:
  
    tvisibilities: triangle visibility mask
      M[] - 1-rank numpy array of the ratio of the surface that is visible
  
    taweights: triangle averaging weights
      W[][3] - 2-rank numpy array of three weight one for each vertex of triangle
  
  Ref:
  * http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
  * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays
  * http://folk.uio.no/hpl/scripting/doc/python/NumPy/Numeric/numpy-13.html
*/

static PyObject *mesh_visibility(PyObject *self, PyObject *args, PyObject *keywds){

  //
  // Reading arguments
  //

  char *kwlist[] = {
    (char*)"viewdir",
    (char*)"vertices",
    (char*)"triangles",
    (char*)"tnormals",
    (char*)"tvisibilities",
    (char*)"taweights",
    NULL};
       
  PyArrayObject *ov = 0, *oV = 0, *oT = 0, *oN = 0;
  
  bool 
    b_tvisibilities = true,
    b_taweights = false;
  
  #if defined(DEBUG)
  std::cout << "start" << std::endl;
  #endif
  
  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(
        args, keywds, "O!O!O!O!|ii", kwlist,
        &PyArray_Type, &ov,
        &PyArray_Type, &oV, 
        &PyArray_Type, &oT,
        &PyArray_Type, &oN,
        &b_tvisibilities,
        &b_taweights
        )
      )
    return NULL;
  
  //std::cout << ov << ' ' << oV << ' ' << oT << ' ' << oN << '\n';
  
  #if defined(DEBUG)
   
  std::cout 
    << "ov: NDim=" << PyArray_NDIM(ov) 
    << " Dim=" << PyArray_DIM(ov, 0) 
    << " Type=" << PyArray_TYPE(ov)
    << "\n";

  std::cout 
    << "oV: NDim=" << PyArray_NDIM(oV) 
    << " Dim=" << PyArray_DIM(oV, 0) << " " << PyArray_DIM(oV, 1)
    << " Type=" << PyArray_TYPE(oV)
    << "\n";


  std::cout 
    << "oT: NDim=" << PyArray_NDIM(oT) 
    << " Dim=" << PyArray_DIM(oT, 0) << " " << PyArray_DIM(oT, 1)
    << " Type=" << PyArray_TYPE(oT)
    << "\n";

  std::cout 
    << "oT: NDim=" << PyArray_NDIM(oN) 
    << " Dim=" << PyArray_DIM(oN, 0) << " " << PyArray_DIM(oN, 1)
    << " Type=" << PyArray_TYPE(oN)
    << "\n";
  
  
  std::cout.flush();
  #endif
  
  if (!b_tvisibilities && !b_taweights) return NULL;

  std::cout << "0" << std::endl; std::cout.flush();
  
  double *view = (double*)PyArray_DATA(ov);
 
  std::cout << "1" << std::endl; std::cout.flush();
  
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
  
  //
  //  Calculate visibility
  //
  triangle_mesh_visibility(view, V, T, N, M, W);
  
  //
  // Storing results in dictionary
  // https://docs.python.org/2/c-api/dict.html
  //
  
  PyObject *results = PyDict_New();
  
  if (b_tvisibilities) {
    PyDict_SetItemString(results, "tvisibilities", PyArray_FromVector(*M));
    delete M;
  }
  
  if (b_taweights) {
    PyDict_SetItemString(results,"taweights", PyArray_From3DPointVector(*W));
    delete W;  
  }

  return results;
}


/*
  Python wrapper for C++ code:

    Calculation of rough visibility of triangles
    
  Python:

    M = triangle_mesh_visibility(v, V, T, N)
    
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
        &PyArray_Type, &oN)) {
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
  
  return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, M);
}

/*  define functions in module */
/* 
  Some modification in declarations due to use of keywords
  Ref:
  * https://docs.python.org/2.0/ext/parseTupleAndKeywords.html
*/ 
static PyMethodDef Methods[] = {
    
    { "roche_critical_potential", 
      roche_critical_potential,   
      METH_VARARGS, 
      "Determine the critical potentials for given values of q, F, and d."},
    
      
    { "roche_pole", 
      (PyCFunction)roche_pole,   
      METH_VARARGS|METH_KEYWORDS, 
      "Determine the height of the pole of generalized Roche lobes for given "
      "values of q, F, d and Omega0"},
   
    { "roche_points_on_x_axis",
      roche_points_on_x_axis,
      METH_VARARGS, 
      "Calculate the points on x-axis of the generalized Roche lobes "
      "ar values of q, F, d and Omega0"},
      
   
    { "roche_gradOmega", 
      roche_gradOmega,   
      METH_VARARGS, 
      "Calculate the gradient and the value of the generalized Kopal potentil"
      " at given point [x,y,z] for given values of q, F and d"},  
      
    { "roche_gradOmega_only", 
      roche_gradOmega_only,   
      METH_VARARGS, 
      "Calculate the gradient of the generalized Kopal potentil"
      " at given point [x,y,z] for given values of q, F and d"},   
    

    { "roche_marching_mesh", 
      (PyCFunction)roche_marching_mesh,   
      METH_VARARGS|METH_KEYWORDS, 
      "Determine the triangular meshing of generalized Roche lobes for "
      "given values of q, F, d and value of the generalized Kopal potential "
      "Omega0. The edge of triangles used in the mesh are approximately delta."},
    
    { "mesh_visibility",
      (PyCFunction)mesh_visibility,
      METH_VARARGS|METH_KEYWORDS, 
      "Determine the ratio of triangle surfaces that are visible in a triangular mesh."},
    
    { "mesh_rough_visibility",
      mesh_rough_visibility,
      METH_VARARGS,
      "Classify the visibility of triangles of the mesh into hidden, partially hidden and visible"},
      
    {NULL,  NULL, 0, NULL} // terminator record
};

/* module initialization */
PyMODINIT_FUNC initlibphoebe (void)
{
  PyObject *backend = Py_InitModule("libphoebe", Methods);
  
  // Added to handle Numpy arrays
  // Ref: 
  // * http://docs.scipy.org/doc/numpy-1.10.1/user/c-info.how-to-extend.html
  import_array();
  
  if (!backend) return;
}
