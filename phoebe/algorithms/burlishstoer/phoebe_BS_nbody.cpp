
#include <Python.h>
#include "n_body_state.h"
#include "n_body.h"

static PyObject *kep2cartesian(PyObject *self, PyObject *args) {
    // printf("entered: kep2cartesian\n");
    // parse input arguments
    PyObject *mass_o, *a_o, *e_o, *inc_o, *om_o, *ln_o, *ma_o;
    double *mass, *a, *e, *inc, *om, *ln, *ma, *mj, t0;
    int Nobjects, i, j;

    // printf("parsing arguments...\n");
    if (!PyArg_ParseTuple(args, "OOOOOOOd", &mass_o, &a_o, &e_o, &inc_o, &om_o, &ln_o, &ma_o, &t0))
        return NULL;
    // printf("arguments parsed!\n");

    // get lengths of lists
    Nobjects = PyTuple_Size(mass_o);

    // convert objects to C arrays
    //~ printf("converting to C arrays...\n");
    mass = new double[Nobjects];
    a = new double[Nobjects-1];
    e = new double[Nobjects-1];
    inc = new double[Nobjects-1];
    om = new double[Nobjects-1];
    ln = new double[Nobjects-1];
    ma = new double[Nobjects-1];
    mj = new double[Nobjects-1];  // NOT REALLY SURE WHAT THIS DOES

    for (i = 0; i < Nobjects; i++){
        mass[i] = PyFloat_AsDouble(PyTuple_GetItem(mass_o, i));
    }
    for (i = 0; i < Nobjects-1; i++){
        a[i] = PyFloat_AsDouble(PyTuple_GetItem(a_o, i));
        e[i] = PyFloat_AsDouble(PyTuple_GetItem(e_o, i));
        inc[i] = PyFloat_AsDouble(PyTuple_GetItem(inc_o, i));
        om[i] = PyFloat_AsDouble(PyTuple_GetItem(om_o, i));
        ln[i] = PyFloat_AsDouble(PyTuple_GetItem(ln_o, i));
        ma[i] = PyFloat_AsDouble(PyTuple_GetItem(ma_o, i));
    }

    // initialize output
    //~ printf("initializing output...\n");
    PyObject *dict = PyDict_New();

    PyObject *x = PyTuple_New(Nobjects);
    PyObject *y = PyTuple_New(Nobjects);
    PyObject *z = PyTuple_New(Nobjects);
    PyObject *vx = PyTuple_New(Nobjects);
    PyObject *vy = PyTuple_New(Nobjects);
    PyObject *vz = PyTuple_New(Nobjects);

    // create Nbody state
    NBodyState state(mass, a, e, inc, om, ln, ma, Nobjects, t0);

    state.kep_elements(mj,a,e,inc,om,ln,ma);

    for (j = 0; j < Nobjects; j++){
        PyTuple_SetItem(x, j, Py_BuildValue("d", state.X_B(j)));
        PyTuple_SetItem(y, j, Py_BuildValue("d", state.Y_B(j)));
        PyTuple_SetItem(z, j, Py_BuildValue("d", state.Z_B(j)));
        PyTuple_SetItem(vx, j, Py_BuildValue("d", state.V_X_B(j)));
        PyTuple_SetItem(vy, j, Py_BuildValue("d", state.V_Y_B(j)));
        PyTuple_SetItem(vz, j, Py_BuildValue("d", state.V_Z_B(j)));
    }

    PyDict_SetItem(dict, Py_BuildValue("s", "x"), x);
    PyDict_SetItem(dict, Py_BuildValue("s", "y"), y);
    PyDict_SetItem(dict, Py_BuildValue("s", "z"), z);
    PyDict_SetItem(dict, Py_BuildValue("s", "vx"), vx);
    PyDict_SetItem(dict, Py_BuildValue("s", "vy"), vy);
    PyDict_SetItem(dict, Py_BuildValue("s", "vz"), vz);

    return dict;
}


static PyObject *do_dynamics(PyObject *self, PyObject *args) {
    //~ printf("entered: do_dynamics\n");
    // parse input arguments
    PyObject *mass_o, *a_o, *e_o, *inc_o, *om_o, *ln_o, *ma_o, *times_o;
    double *mass, *a, *e, *inc, *om, *ln, *ma, *times, *mj;
    int Nobjects, Ntimes, i, j, ltte, return_keplerian;
    double t0, maxh, orbit_error;

    //~ printf("parsing arguments...\n");
    if (!PyArg_ParseTuple(args, "OOOOOOOOdddii", &times_o, &mass_o, &a_o, &e_o, &inc_o, &om_o, &ln_o, &ma_o, &t0, &maxh, &orbit_error, &ltte, &return_keplerian))
        return NULL;

    // get lengths of lists
    Nobjects = PyTuple_Size(mass_o);
    Ntimes = PyTuple_Size(times_o);

    // convert objects to C arrays
    //~ printf("converting to C arrays...\n");
    mass = new double[Nobjects];
    a = new double[Nobjects-1];
    e = new double[Nobjects-1];
    inc = new double[Nobjects-1];
    om = new double[Nobjects-1];
    ln = new double[Nobjects-1];
    ma = new double[Nobjects-1];
    mj = new double[Nobjects-1];  // NOT REALLY SURE WHAT THIS DOES
    times = new double[Ntimes];

    for (i = 0; i < Nobjects; i++){
        mass[i] = PyFloat_AsDouble(PyTuple_GetItem(mass_o, i));
    }
    for (i = 0; i < Nobjects-1; i++){
        a[i] = PyFloat_AsDouble(PyTuple_GetItem(a_o, i));
        e[i] = PyFloat_AsDouble(PyTuple_GetItem(e_o, i));
        inc[i] = PyFloat_AsDouble(PyTuple_GetItem(inc_o, i));
        om[i] = PyFloat_AsDouble(PyTuple_GetItem(om_o, i));
        ln[i] = PyFloat_AsDouble(PyTuple_GetItem(ln_o, i));
        ma[i] = PyFloat_AsDouble(PyTuple_GetItem(ma_o, i));
    }
    for (i = 0; i < Ntimes; i++){
        times[i] = PyFloat_AsDouble(PyTuple_GetItem(times_o, i));
    }

    // initialize output
    //~ printf("initializing output...\n");
    PyObject *dict = PyDict_New();

    PyObject *x = PyTuple_New(Ntimes);
    PyObject *y = PyTuple_New(Ntimes);
    PyObject *z = PyTuple_New(Ntimes);
    PyObject *vx = PyTuple_New(Ntimes);
    PyObject *vy = PyTuple_New(Ntimes);
    PyObject *vz = PyTuple_New(Ntimes);

    // if (return_keplerian > 0) {
    PyObject *kepl_a = PyTuple_New(Ntimes);
    PyObject *kepl_e = PyTuple_New(Ntimes);
    PyObject *kepl_in = PyTuple_New(Ntimes);
    PyObject *kepl_o = PyTuple_New(Ntimes);
    PyObject *kepl_ln = PyTuple_New(Ntimes);
    PyObject *kepl_m = PyTuple_New(Ntimes);
    // }

    // create Nbody state
    NBodyState state(mass, a, e, inc, om, ln, ma, Nobjects, t0);

    state.kep_elements(mj,a,e,inc,om,ln,ma);


    for (i = 0; i < Ntimes; i++) {
        // integrate to this time

        state.kep_elements(mj,a,e,inc,om,ln,ma);

        state(times[i],maxh,orbit_error,1e-16);
        //~ state.kep_elements(mj,a,e,inc,om,ln,ma);

        // need to (re)create tuples to hold positions for each object at this time
        PyObject *xi = PyTuple_New(Nobjects);
        PyObject *yi = PyTuple_New(Nobjects);
        PyObject *zi = PyTuple_New(Nobjects);
        PyObject *vxi = PyTuple_New(Nobjects);
        PyObject *vyi = PyTuple_New(Nobjects);
        PyObject *vzi = PyTuple_New(Nobjects);

        PyObject *kepl_ai = PyTuple_New(Nobjects);
        PyObject *kepl_ei = PyTuple_New(Nobjects);
        PyObject *kepl_ini = PyTuple_New(Nobjects);
        PyObject *kepl_oi = PyTuple_New(Nobjects);
        PyObject *kepl_lni = PyTuple_New(Nobjects);
        PyObject *kepl_mi = PyTuple_New(Nobjects);

        for (j = 0; j < Nobjects; j++){
            if (ltte > 0) {
                PyTuple_SetItem(xi, j, Py_BuildValue("d", state.X_LT(j)));
                PyTuple_SetItem(yi, j, Py_BuildValue("d", state.Y_LT(j)));
                PyTuple_SetItem(zi, j, Py_BuildValue("d", state.Z_LT(j)));
                PyTuple_SetItem(vxi, j, Py_BuildValue("d", state.V_X_LT(j)));
                PyTuple_SetItem(vyi, j, Py_BuildValue("d", state.V_Y_LT(j)));
                PyTuple_SetItem(vzi, j, Py_BuildValue("d", state.V_Z_LT(j)));
            } else {
                PyTuple_SetItem(xi, j, Py_BuildValue("d", state.X_B(j)));
                PyTuple_SetItem(yi, j, Py_BuildValue("d", state.Y_B(j)));
                PyTuple_SetItem(zi, j, Py_BuildValue("d", state.Z_B(j)));
                PyTuple_SetItem(vxi, j, Py_BuildValue("d", state.V_X_B(j)));
                PyTuple_SetItem(vyi, j, Py_BuildValue("d", state.V_Y_B(j)));
                PyTuple_SetItem(vzi, j, Py_BuildValue("d", state.V_Z_B(j)));
            }

            // if (return_keplerian > 0) {
                // state.kep_elements(mji, ai, ei, ini, oi, lni, mi);
            PyTuple_SetItem(kepl_ai, j, Py_BuildValue("d", a));
            PyTuple_SetItem(kepl_ei, j, Py_BuildValue("d", e));
            PyTuple_SetItem(kepl_ini, j, Py_BuildValue("d", inc));
            PyTuple_SetItem(kepl_oi, j, Py_BuildValue("d", om));
            PyTuple_SetItem(kepl_lni, j, Py_BuildValue("d", ln));
            PyTuple_SetItem(kepl_mi, j, Py_BuildValue("d", ma));
            // }
        }

        PyTuple_SetItem(x, i, xi);
        PyTuple_SetItem(y, i, yi);
        PyTuple_SetItem(z, i, zi);
        PyTuple_SetItem(vx, i, vxi);
        PyTuple_SetItem(vy, i, vyi);
        PyTuple_SetItem(vz, i, vzi);

        // if (return_keplerian > 0) {
        PyTuple_SetItem(kepl_a, i, kepl_ai);
        PyTuple_SetItem(kepl_e, i, kepl_ei);
        PyTuple_SetItem(kepl_in, i, kepl_ini);
        PyTuple_SetItem(kepl_o, i, kepl_oi);
        PyTuple_SetItem(kepl_ln, i, kepl_lni);
        PyTuple_SetItem(kepl_m, i, kepl_mi);
        // }
    }

    // prepare dictionary for returning output
    PyDict_SetItem(dict, Py_BuildValue("s", "t"), times_o);
    PyDict_SetItem(dict, Py_BuildValue("s", "x"), x);
    PyDict_SetItem(dict, Py_BuildValue("s", "y"), y);
    PyDict_SetItem(dict, Py_BuildValue("s", "z"), z);
    PyDict_SetItem(dict, Py_BuildValue("s", "vx"), vx);
    PyDict_SetItem(dict, Py_BuildValue("s", "vy"), vy);
    PyDict_SetItem(dict, Py_BuildValue("s", "vz"), vz);

    // if (return_keplerian > 0) {
    PyDict_SetItem(dict, Py_BuildValue("s", "kepl_a"), kepl_a);
    PyDict_SetItem(dict, Py_BuildValue("s", "kepl_e"), kepl_e);
    PyDict_SetItem(dict, Py_BuildValue("s", "kepl_in"), kepl_in);
    PyDict_SetItem(dict, Py_BuildValue("s", "kepl_o"), kepl_o);
    PyDict_SetItem(dict, Py_BuildValue("s", "kepl_ln"), kepl_ln);
    PyDict_SetItem(dict, Py_BuildValue("s", "kepl_m"), kepl_m);
    // }

    // return format (for a triple) will be something like
    // {'t': [...], 'x': [(t1, t2, t3), (t1, t2, t3), ...], 'y': ..., 'z': ...}

    return dict;

}


static PyMethodDef Methods[] = {
    {"do_dynamics",      do_dynamics,   METH_VARARGS, "Do N-body dynamics to get positions as a function of time"},
    {"kep2cartesian",    kep2cartesian, METH_VARARGS, "Convert from keplerian to cartesian initial conditions"},
    {NULL,               NULL,             0,            NULL}
};

PyMODINIT_FUNC initphoebe_burlishstoer (void)
{
    PyObject *backend = Py_InitModule("phoebe_burlishstoer", Methods);
    if (!backend)
        return;
}
