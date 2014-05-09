#include <iostream>
#include <vector>
#include <fstream>
#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *place_in_binary_orbit(PyObject *dummy, PyObject *args)
{
    /* Place a mesh in a binary orbit
     * 
     *
     * Output: no output
     */
    
    /* Variable declaration:
     * 
     */
    double theta,longan,incl; // Euler angles
    double s1,c1,s2,c2,s3,c3; // will hold values of sin/cos Euler angles
    double c1c3,c1s3;
    double u1,v1,w1,u2,v2,w2,u3,v3,w3;
    double x,y,z;             // will hold values of rotated coordinates
    double x0,y0,z0;          // translation vector
    double vx, vy, vz;        // temporary velocity vector
    double v0x, v0y, v0z;        // velocity offset vector
    int xp, yp, zp;
    
    int n_triangles;
    double *mu, *center, *ocenter, *polar_dir;
    double *triangle, *normal, *velo;
    PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL, *arg5=NULL, *arg6=NULL, *arg7=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL, *arr5=NULL, *arr6=NULL, *arr7=NULL;
    
    /* Argument parsing:
     * We need to parse the arguments, and then put the data
     * in arrays.
     */
    // Parse arguments
    if (!PyArg_ParseTuple(args, "OOOOOOO(ddd)(ddd)(ddd)", &arg1, &arg2, &arg3,
                                &arg4, &arg5, &arg6, &arg7,
                                &theta, &longan, &incl,
                                &x0, &y0, &z0, &v0x, &v0y, &v0z)) return NULL;

    // Put the arguments in arrays
    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr3 = PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr4 = PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr5 = PyArray_FROM_OTF(arg5, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr6 = PyArray_FROM_OTF(arg6, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr7 = PyArray_FROM_OTF(arg7, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    if (arr2 == NULL) return NULL;
    if (arr3 == NULL) return NULL;
    if (arr4 == NULL) return NULL;
    if (arr5 == NULL) return NULL;
    if (arr6 == NULL) return NULL;
    if (arr7 == NULL) return NULL;
    
    // Pointers to data
    mu = (double *)PyArray_DATA(arr1); // pointer to data.
    center = (double *)PyArray_DATA(arr2); // pointer to data.
    triangle = (double *)PyArray_DATA(arr3); // pointer to data.
    normal = (double *)PyArray_DATA(arr4); // pointer to data.
    velo = (double *)PyArray_DATA(arr5); // pointer to data.
    polar_dir = (double *)PyArray_DATA(arr6); // pointer to data.
    ocenter = (double *)PyArray_DATA(arr7); // pointer to data.
    
    // length
    n_triangles = PyArray_DIM(arr1, 0);
    
    s1 = sin(incl);
    c1 = cos(incl);
    s2 = sin(longan);
    c2 = cos(longan);
    s3 = sin(theta);
    c3 = cos(theta);
    c1s3 = c1*s3;
    c1c3 = c1*c3;
    
    u1 = -c2*c3+s2*c1s3;
    u2 = -s2*c3-c2*c1s3;
    u3 = s1*s3;
    v1 = c2*s3+s2*c1c3;
    v2 = s2*s3-c2*c1c3;
    v3 = s1*c3;
    w1 = -s2*s1;
    w2 = c2*s1;
    w3 = c1;
    
    for (int i=0;i<n_triangles;i++){
        xp = 3*i;
        yp = 3*i+1;
        zp = 3*i+2;
        
        
        // velocity [VECTOR]
        vx = ocenter[yp]*polar_dir[2] - ocenter[zp]*polar_dir[1];
        vy =-ocenter[xp]*polar_dir[2] + ocenter[zp]*polar_dir[0];
        vz = ocenter[xp]*polar_dir[1] - ocenter[yp]*polar_dir[0];
        velo[xp] = u1*vx + v1*vy + w1*vz + v0x;
        velo[yp] = u2*vx + v2*vy + w2*vz + v0y;
        velo[zp] = u3*vx + v3*vy + w3*vz + v0z;                
        
        // center coordinates
        x = center[xp];
        y = center[yp];
        z = center[zp];
        center[xp] = u1*x + v1*y + w1*z + x0;
        center[yp] = u2*x + v2*y + w2*z + y0;
        center[zp] = u3*x + v3*y + w3*z + z0;
        
        // normal [VECTOR]
        x = normal[xp];
        y = normal[yp];
        z = normal[zp];
        normal[xp] = u1*x + v1*y + w1*z;
        normal[yp] = u2*x + v2*y + w2*z;
        normal[zp] = u3*x + v3*y + w3*z;
        
        // limb angle
        // default los dir = [0,0,+1]
        mu[i] = normal[zp]/sqrt(normal[xp]*normal[xp] + normal[yp]*normal[yp] + normal[zp]*normal[zp]);
        
        // triangle 1 coordinates
        xp = 9*i;
        yp = 9*i+1;
        zp = 9*i+2;
        x = triangle[xp];
        y = triangle[yp];
        z = triangle[zp];
        triangle[xp] = u1*x + v1*y + w1*z + x0;
        triangle[yp] = u2*x + v2*y + w2*z + y0;
        triangle[zp] = u3*x + v3*y + w3*z + z0;
        
        // triangle 2 coordinates
        xp = xp+3;
        yp = yp+3;
        zp = zp+3;
        x = triangle[xp];
        y = triangle[yp];
        z = triangle[zp];
        triangle[xp] = u1*x + v1*y + w1*z + x0;
        triangle[yp] = u2*x + v2*y + w2*z + y0;
        triangle[zp] = u3*x + v3*y + w3*z + z0;
        
        // triangle 3 coordinates
        xp = xp+3;
        yp = yp+3;
        zp = zp+3;
        x = triangle[xp];
        y = triangle[yp];
        z = triangle[zp];
        triangle[xp] = u1*x + v1*y + w1*z + x0;
        triangle[yp] = u2*x + v2*y + w2*z + y0;
        triangle[zp] = u3*x + v3*y + w3*z + z0;
        
        
    }                    
        
    
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    Py_XDECREF(arr3);
    Py_XDECREF(arr4);
    Py_XDECREF(arr5);
    Py_XDECREF(arr6);
    Py_XDECREF(arr7);
    // We need to incref Py None, otherwise we'll get a fatal Python error
    // somewhere after executing this for many times...
    Py_INCREF(Py_None);
    return Py_None;
}




// register all functions
static PyMethodDef ctransMethods[] = {
  {"place_in_binary_orbit",  place_in_binary_orbit}, //python name, C name
  {NULL, NULL, 0, NULL} //required ending
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ctrans",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        ctransMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif


// module init function
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_ctrans(void)
#else
initctrans(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  (void) PyModule_Create(&moduledef);
#else
  (void) Py_InitModule3("ctrans", ctransMethods,"ctrans doc");
  import_array();
#endif
}    
