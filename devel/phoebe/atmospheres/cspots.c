#include "Python.h"
#include "numpy/arrayobject.h"


static PyObject *detect_circular_spot(PyObject *self, PyObject *args)
{
  // Python calling signature:
  // >>> x,y,z = cgeometry.rotate_into_orbit(obj,euler,loc).reshape((3,-1))
  // with obj a 3XN array, euler and loc can be tuples of floats
  double spot_long,spot_colat,spot_angrad; // will hold spot parameters
  double x1,x2,x3,y1,y2,y3,z1,z2,z3; // Cartesian coordinates
  double phi1,theta1,phi2,theta2,phi3,theta3; // Spherical coordinates
  double d1,d2,d3; //distances
  double s1,s2,s3,p1,p2,p3; //shortcuts
  int inside,outside; // booleans
  int i,N;
  PyArrayObject *tri_array,*tri_in_on_spot; // triangle vertices coordinates
  double *tri; // keeps track of triangles on/in spots
  int dims[1];
  
  // see http://docs.python.org/2/c-api/arg.html for details on formatting strings
  // Python Scripting for Computational Science by Hans Petter Langtangen is
  // also very helpful
  if (!PyArg_ParseTuple(args, "O!(ddd)", &PyArray_Type, &tri_array, &spot_long, &spot_colat, &spot_angrad))
      return NULL;
 
  // error checks: first array should be two-dimensional, second one two-dimensional and 
  // thr type of both should be double
  if (tri_array->nd != 2 || tri_array->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "dim(obj)!=2 or dim(rot)!=2 or type not equal to double");
      return NULL;
  }
  
  if ((int)tri_array->dimensions[0]!=9){
      PyErr_SetString(PyExc_ValueError, "object array has incorrect size");
      return NULL;
  }
  
  // get dimensions of the array. it is also possible to access the, *tri_in_spot
  // elements of the array as if it was two-dimensional, but in this
  // case it is just easier to treat the arrays as one dimensional
  // with Nx3 elements
  N= (int)tri_array->dimensions[1];
  
  // we need to 'source' the data from the arrays and create a new 
  // return array
  tri = (double *)(tri_array->data);
  
  dims[0]=(int)tri_array->dimensions[1]*2;
  tri_in_on_spot = (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_INT);
  
  //////////// the original part
  for (i=0; i<N;i+=1)
  {
      // Cartesian coordinates
      x1 = tri[i];
      y1 = tri[i+N];
      z1 = tri[i+2*N];

      x2 = tri[i+3*N];
      y2 = tri[i+4*N];
      z2 = tri[i+5*N];
      
      x3 = tri[i+6*N];
      y3 = tri[i+7*N];
      z3 = tri[i+8*N];
      
      // Spherical coordinates
      //r1 = sqrt(x1*x1 + y1*y1 + z1*z1)
      //r2 = sqrt(x2*x2 + y2*y2 + z2*z2)
      //r3 = sqrt(x3*x3 + y3*y3 + z3*z3)
      
      phi1 = atan2(y1,x1);
      phi2 = atan2(y2,x2);
      phi3 = atan2(y3,x3);
      
      theta1 = atan2(sqrt(x1*x1+y1*y1),z1);
      theta2 = atan2(sqrt(x2*x2+y2*y2),z2);
      theta3 = atan2(sqrt(x3*x3+y3*y3),z3);
      
      // distances of all vertex coordinates to spot
      s1 = sin(0.5*(spot_colat-theta1));
      s2 = sin(0.5*(spot_colat-theta2));
      s3 = sin(0.5*(spot_colat-theta3));
      p1 = sin(0.5*(phi1-spot_long));
      p2 = sin(0.5*(phi2-spot_long));
      p3 = sin(0.5*(phi3-spot_long));
      
      d1 = 2.0*asin(sqrt(s1*s1+sin(spot_colat)*sin(theta1)*p1*p1));
      d2 = 2.0*asin(sqrt(s2*s2+sin(spot_colat)*sin(theta2)*p2*p2));
      d3 = 2.0*asin(sqrt(s3*s3+sin(spot_colat)*sin(theta3)*p3*p3));
      
      inside = (d1<=spot_angrad) && (d2<=spot_angrad) && (d3<=spot_angrad);
      outside = (d1>spot_angrad) && (d2>spot_angrad) && (d3>spot_angrad);
      
      ((int *)tri_in_on_spot->data)[i] = (1-inside) && (1-outside);
      ((int *)tri_in_on_spot->data)[i+N] = inside;
      
  }
  //////////// 
  
  // create a new numpy array and return it back to python.
  return PyArray_Return(tri_in_on_spot);
}









// register all functions
static PyMethodDef spotsMethods[] = {
  {"detect_circular_spot",  detect_circular_spot}, //python name, C name
  {NULL, NULL} //required ending
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cspots",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        spotsMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif



PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_cspots(void)
#else
initcspots(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  (void) PyModule_Create(&moduledef);
#else
  (void) Py_InitModule3("cspots", spotsMethods,"cspots doc");
  import_array();
#endif
}
