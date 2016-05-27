
#include <Python.h>
#include "gen_roche.h"

static PyObject *critical_potential(PyObject *self, PyObject *args) {
    // parse input arguments   
    double q, F, delta, omega[3];
    
    // printf("parsing arguments...\n");
    if (!PyArg_ParseTuple(args, "ddd", &q, &F, &delta))
        return NULL;
    
    gen_roche::critical_potential(omega, q, F, delta);

    PyObject *omega_o = PyTuple_New(3);
    PyTuple_SetItem(omega_o, 0, Py_BuildValue("d", omega[0]));
    PyTuple_SetItem(omega_o, 1, Py_BuildValue("d", omega[1]));
    PyTuple_SetItem(omega_o, 2, Py_BuildValue("d", omega[2]));

    return omega_o;
}

static PyMethodDef Methods[] = {
    {"critical_potential", critical_potential,   METH_VARARGS, "Determine the critical potentials for given values of q, F, and delta"},
    {NULL,               NULL,             0,            NULL}
};

PyMODINIT_FUNC initphoebe_roche (void)
{
    PyObject *backend = Py_InitModule("phoebe_roche", Methods);
    if (!backend)
        return;
}
