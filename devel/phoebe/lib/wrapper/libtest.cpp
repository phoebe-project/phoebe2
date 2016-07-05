/*
  Testing wrapping and memory leaks. 

  Author: Martin Horvat, Jul 2016
*/ 

#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *fun1(PyObject *self, PyObject *args) {
  
  double x;
  
  if (!PyArg_ParseTuple(args, "d", &x)) return NULL;
  
  double *a = new double[10];
  
  for (int i = 0; i < 10; ++i) a[i] = x + 10;
  
  npy_intp dims[1] = {10};

  PyObject *pya = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, a);
  
  PyArray_ENABLEFLAGS((PyArrayObject *)pya, NPY_ARRAY_OWNDATA);
  
  return pya; 
}


static PyObject *fun2(PyObject *self, PyObject *args) {
  
  double x;
  
  if (!PyArg_ParseTuple(args, "d", &x)) return NULL;
  
  PyObject *tuple = PyTuple_New(3);
  
  for (int i = 0; i< 3; ++i) 
    PyTuple_SetItem(tuple, i, PyFloat_FromDouble(x + i)); 
      
  return tuple; 
}
/*
  Define functions in module
   
  Some modification in declarations due to use of keywords
  Ref:
  * https://docs.python.org/2.0/ext/parseTupleAndKeywords.html
*/ 
static PyMethodDef Methods[] = {
    
    { "fun1", 
      fun1,   
      METH_VARARGS, 
      "Testing function 1."},

    { "fun2", 
      fun2,   
      METH_VARARGS, 
      "Testing function 2."},


    {NULL,  NULL, 0, NULL} // terminator record
};

/* module initialization */
PyMODINIT_FUNC initlibtest(void)
{
  PyObject *backend = Py_InitModule("libtest", Methods);
  
  // Added to handle Numpy arrays
  // Ref: 
  // * http://docs.scipy.org/doc/numpy-1.10.1/user/c-info.how-to-extend.html
  import_array();
  
  if (!backend) return;
}

