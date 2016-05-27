#include "Python.h"
#include "numpy/arrayobject.h"


static PyObject *rotate_into_orbit(PyObject *self, PyObject *args)
{
  // Python calling signature:
  // >>> x,y,z = cgeometry.rotate_into_orbit(obj,euler,loc).reshape((3,-1))
  // with obj a 3XN array, euler and loc can be tuples of floats
  double theta,longan,incl; // Euler angles
  double s1,c1,s2,c2,s3,c3; // will hold values of sin/cos Euler angles
  double x,y,z;             // will hold values of rotated coordinates
  double x0,y0,z0;          // translation vector
  int i,N;
  PyArrayObject *obj_array, *rot; // original and rotated coords
  double *obj;
  int dims[1];
  
  // see http://docs.python.org/2/c-api/arg.html for details on formatting strings
  // Python Scripting for Computational Science by Hans Petter Langtangen is
  // also very helpful
  if (!PyArg_ParseTuple(args, "O!(ddd)(ddd)", &PyArray_Type, &obj_array, &theta, &longan, &incl, &x0, &y0, &z0))
      return NULL;
 
  // error checks: first array should be two-dimensional, second one two-dimensional and 
  // thr type of both should be double
  if (obj_array->nd != 2 || obj_array->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "dim(obj)!=2 or dim(rot)!=2 or type not equal to double");
      return NULL;
  }
  
  if ((int)obj_array->dimensions[0]!=3){
	  PyErr_SetString(PyExc_ValueError, "object array has incorrect size");
	  return NULL;
  }
  
  // get dimensions of the array. it is also possible to access the
  // elements of the array as if it was two-dimensional, but in this
  // case it is just easier to treat the arrays as one dimensional
  // with Nx3 elements
  N= (int)obj_array->dimensions[1];
  
  // we need to 'source' the data from the arrays and create a new 
  // return array
  obj = (double *)(obj_array->data);
  
  dims[0]=(int)obj_array->dimensions[1]*3;
  rot = (PyArrayObject *)PyArray_FromDims(1, dims, NPY_DOUBLE);
  
  //////////// the original part
  s1 = sin(incl);
  c1 = cos(incl);
  s2 = sin(longan);
  c2 = cos(longan);
  s3 = sin(theta);
  c3 = cos(theta);
  for (i=0; i<N;i+=1)
  {
      x = (-c2*c3+s2*c1*s3)*obj[i] + (c2*s3+s2*c1*c3)*obj[i+N] - s2*s1*obj[i+2*N] + x0;
      y = (-s2*c3-c2*c1*s3)*obj[i] + (s2*s3-c2*c1*c3)*obj[i+N] + c2*s1*obj[i+2*N] + y0;
      z =           (s1*s3)*obj[i] +          (s1*c3)*obj[i+N] +    c1*obj[i+2*N] + z0;
      ((double *)rot->data)[i] = x;
      ((double *)rot->data)[i+N] = y;
      ((double *)rot->data)[i+2*N] = z;
      
  }
  //////////// 
  
  // create a new numpy array and return it back to python.
  return PyArray_Return(rot);
}















static PyObject *cos_theta(PyObject *self, PyObject *args)
{
  double bnorm, num, denom, anorm;
  int N,N2,i;
  PyArrayObject *a_array, *b_array, *cosangle;
  double *a, *b;
  int dims[1];
  
  // see http://docs.python.org/2/c-api/arg.html for details on formatting strings
  // Python Scripting for Computational Science by Hans Petter Langtangen is
  // also very helpful
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_array, &PyArray_Type, &b_array))
      return NULL;
 
  // error checks: first array should be two-dimensional, second one one-dimensional and 
  // thr type of both should be double
  if (a_array->nd != 2 || b_array->nd != 1 || a_array->descr->type_num != PyArray_DOUBLE || b_array->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "dim(a)!=2 or dim(b)!=1 or type not equal to double");
      return NULL;
  }
  
  if ((int)a_array->dimensions[1]!=3 || (int)b_array->dimensions[0]!=3){
      PyErr_SetString(PyExc_ValueError, "array size mismatch");
      return NULL;
  }
  
  // get dimensions of the array. it is also possible to access the
  // elements of the array as if it was two-dimensional, but in this
  // case it is just easier to treat the 'a' array as one dimensional
  // with Nx3 elements
  N=(int)a_array->dimensions[0]; 
  N2=(int)a_array->dimensions[0]*2; 
  
  // we need to 'source' the data from the arrays and create a new 
  // return array
  a = (double *)(a_array->data);
  b = (double *)(b_array->data);
  
  dims[0]=(int)a_array->dimensions[0];
  cosangle = (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_DOUBLE);
  
  //////////// the original part
  bnorm = sqrt( b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  for (i=0; i<N;i+=1)
  {
      num = a[i]*b[0] + a[i+N]*b[1] + a[i+N2]*b[2];
      anorm = sqrt(a[i]*a[i] + a[i+N]*a[i+N] + a[i+N2]*a[i+N2]);  
      denom = anorm*bnorm;

      ((double *)cosangle->data)[i] = num/denom;
  }
  //////////// 
  
  // create a new numpy array and return it back to python.
  return PyArray_Return(cosangle);
}








// register all functions
static PyMethodDef geometryMethods[] = {
  {"rotate_into_orbit",  (PyCFunction)rotate_into_orbit, METH_VARARGS, NULL}, //python name, C name
  {"cos_theta", (PyCFunction)cos_theta, METH_VARARGS, NULL}, 
  {NULL, NULL} //required ending
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cgeometry",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        geometryMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif





// module init function
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_cgeometry(void)
#else
initcgeometry(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  (void) PyModule_Create(&moduledef);
#else
  (void) Py_InitModule3("cgeometry", geometryMethods,"cgeometry docs");
  import_array(); //needed if numpy is used
#endif
}
