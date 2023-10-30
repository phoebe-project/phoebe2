#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>

#include "ndpolator.h"

array *new_array(int len)
{
    array *a = malloc(sizeof(*a));
    a->data = malloc(len * sizeof(a->data));
    return a;
}

int free_array(array *a)
{
    free(a->data);
    free(a);

    return NDPOLATOR_SUCCESS;
}

int find_first_geq_than(double *arr, int l, int r, double x, double tol)
{
    int m = l + (r - l) / 2;

    while (l != r) {
        if (arr[m] < x)
            l = m + 1;
        else
            r = m;

        m = l + (r - l) / 2;
    }

    if (fabs( (x-arr[l-1])/(arr[l]-arr[l-1]) ) < tol)
        return -(l-1);
    if (fabs( (arr[l]-x)/(arr[l]-arr[l-1]) ) < tol)
        return -l;

    return l;
}

int c_ndpolate(int N, int vdim, double *x, double *fv)
{
    int i, j, k;

    for (i = 0; i < N; i++) {
        // printf("N=%d x[%d]=%3.3f\n", N, i, x[i]);
        for (j = 0; j < (1 << (N - i - 1)); j++) {
            // printf("j=%d fv[%d]=%3.3f, fv[%d]=%3.3f, ", j, (1 << (N - i - 1)) + j, fv[(1 << (N - i - 1)) + j], j, fv[j]);
            for (k = 0; k < vdim; k++) {
                // fv[j] += (fv[(1 << (N - i - 1)) + j]-fv[j]) * x[i];
                fv[j*vdim+k] += (fv[((1 << (N - i - 1)) + j)*vdim+k] - fv[j*vdim+k]) * x[i];
            }
            // printf("corr=%3.3f\n", fv[j]);
        }
    }

    return 0;
}

PyObject *vectorized_find_first_geq_than(PyObject *axes, PyArrayObject *query_pts)
{
    int i, j;

    npy_intp *query_pts_shape = PyArray_SHAPE(query_pts);

    int naxes = PyTuple_Size(axes);
    int nelems = PyArray_DIM(query_pts, 0);

    int *ms = malloc(nelems * naxes * sizeof(*ms));

    double *query_pts_data = (double *) PyArray_DATA(query_pts);

    PyObject *indices;

    int debug = 0;

    if (debug) {
        printf("vecfind: naxes=%d, nelems=%d\n", naxes, nelems);
        for (i = 0; i < nelems; i++)
            for (j = 0; j < naxes; j++)
                printf("%d %d %lf\n", i, j, query_pts_data[i*naxes+j]);
    }

    for (i = 0; i < naxes; i++) {
        PyArrayObject *npaxis = (PyArrayObject *) PyTuple_GetItem(axes, i);
        double *axis = (double *) PyArray_DATA(npaxis);
        int axis_length = PyArray_SIZE(npaxis);

        for (j = 0; j < nelems; j++)
            ms[j * naxes + i] = find_first_geq_than(axis, 0, axis_length - 1, query_pts_data[j * naxes + i], 1e-3);
    }

    indices = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, ms);
    PyArray_ENABLEFLAGS((PyArrayObject *) indices, NPY_ARRAY_OWNDATA);

    return indices;
}

static PyObject *find(PyObject *self, PyObject *args)
{
    PyObject *tuple_of_axes, *indices;
    PyArrayObject *query_pts;

    if (!PyArg_ParseTuple(args, "OO", &tuple_of_axes, &query_pts))
        return NULL;

    indices = vectorized_find_first_geq_than(tuple_of_axes, query_pts);
    return indices;
}

void map_to_unit_cube(PyObject *axes, PyArrayObject *interpolants)
{
    int i, j;
    int naxes = PyTuple_Size(axes);
    int nelems = PyArray_DIM(interpolants, 0);

    double *grid = (double *) PyArray_DATA(interpolants);

    for (i = 0; i < naxes; i++) {
        PyArrayObject *pyaxis = (PyArrayObject *) PyTuple_GetItem(axes, i);
        int len = PyArray_DIM(pyaxis, 0);
        double *axis = (double *) PyArray_DATA(pyaxis);
        double min = axis[0], max = axis[len - 1];
        for (j = 0; j < len; j++)
            axis[j] = (axis[j] - min) / (max - min);
        for (j = 0; j < nelems; j++)
            grid[j * naxes + i] = (grid[j * naxes + i] - min) / (max - min);
    }

    return;
}

static PyObject *map(PyObject *self, PyObject *args)
{
    PyObject *tuple_of_axes;
    PyArrayObject *interpolants;

    if (!PyArg_ParseTuple(args, "OO", &tuple_of_axes, &interpolants))
        return NULL;

    map_to_unit_cube(tuple_of_axes, interpolants);
    return Py_BuildValue("i", 0);
}

PyObject *extract_hypercube(PyArrayObject *indices, PyArrayObject *grid)
{
    int i, j;

    int *index = (int *) PyArray_DATA(indices);

    int nelems = PyArray_DIM(indices, 0);
    int naxes = PyArray_DIM(indices, 1);

    PyObject *slices = PyTuple_New(naxes);
    PyObject *hypercubes = PyTuple_New(nelems);
    PyObject *extracted_hypercube;

    int debug = 0;

    for (i = 0; i < nelems; i++) {
        for (j = 0; j < naxes; j++) {
            if (debug)
                printf("nelem %d/%d: index[%d/%d]=%d\n", i, nelems, j, naxes, index[i*naxes+j]);
            PyObject *slice = index[i * naxes + j] > 0 ? PySlice_New(PyLong_FromLong(index[i * naxes + j] - 1), PyLong_FromLong(index[i * naxes + j] + 1), NULL) : PyLong_FromLong(-index[i * naxes + j]);
            PyTuple_SetItem(slices, j, slice);
        }

        /* Because of the slicing, the hypercube will not have its data aligned.
         * That is why we need to build a new array with aligned data; the added
         * benefit is that we can make the data F-contiguous (i.e., the first
         * dimension changes the fastest), which is what we need for the
         * interpolation.
         */
        extracted_hypercube = PyArray_FROM_OTF(PyObject_GetItem((PyObject *) grid, slices), NPY_DOUBLE, NPY_ARRAY_CARRAY);
        PyTuple_SetItem(hypercubes, i, extracted_hypercube);
    }

    return hypercubes;
}

static PyObject *hypercubes(PyObject *self, PyObject *args)
{
    PyArrayObject *indices, *grid;

    if (!PyArg_ParseTuple(args, "OO", &indices, &grid))
        return NULL;

    return extract_hypercube(indices, grid);
}

static PyObject *ainfo(PyObject *self, PyObject *args)
{
    int i, ndim, size;
    PyArrayObject *array;
    double *data;
    npy_intp *dims, *shape, *strides;

    if (!PyArg_ParseTuple(args, "O", &array))
        return NULL;

    ndim = PyArray_NDIM(array);
    size = PyArray_SIZE(array);

    printf("array->nd = %d\n", ndim);
    printf("array->flags = %d\n", PyArray_FLAGS(array));
    printf("array->type = %d\n", PyArray_TYPE(array));
    printf("array->itemsize = %ld\n", PyArray_ITEMSIZE(array));
    printf("array->size = %d\n", size);
    printf("array->nbytes = %ld\n", PyArray_NBYTES(array));

    dims = PyArray_DIMS(array);
    printf("array->dims = [");
    for (i=0; i<ndim-1; i++)
        printf("%ld, ", dims[i]);
    printf("%ld]\n", dims[i]);

    shape = PyArray_SHAPE(array);
    printf("array->shape = [");
    for (i=0; i<ndim-1; i++)
        printf("%ld, ", shape[i]);
    printf("%ld]\n", shape[i]);

    strides = PyArray_STRIDES(array);
    printf("array->strides = [");
    for (i=0; i<ndim-1; i++)
        printf("%ld, ", strides[i]);
    printf("%ld]\n", strides[i]);

    printf("array->is_c_contiguous: %d\n", PyArray_IS_C_CONTIGUOUS(array));
    printf("array->is_f_contiguous: %d\n", PyArray_IS_F_CONTIGUOUS(array));
    printf("array->is_fortran: %d\n", PyArray_ISFORTRAN(array));
    printf("array->is_writeable: %d\n", PyArray_ISWRITEABLE(array));
    printf("array->is_aligned: %d\n", PyArray_ISALIGNED(array));
    printf("array->is_behaved: %d\n", PyArray_ISBEHAVED(array));
    printf("array->is_behaved_ro: %d\n", PyArray_ISBEHAVED_RO(array));
    printf("array->is_carray: %d\n", PyArray_ISCARRAY(array));
    printf("array->is_farray: %d\n", PyArray_ISFARRAY(array));
    printf("array->is_carray_ro: %d\n", PyArray_ISCARRAY_RO(array));
    printf("array->is_farray_ro: %d\n", PyArray_ISFARRAY_RO(array));
    printf("array->is_isonesegment: %d\n", PyArray_ISONESEGMENT(array));

    data = (double *) PyArray_DATA(array);
    printf("data = [");
    for (i = 0; i < size-1; i++)
        printf("%lf, ", data[i]);
    printf("%lf]\n", data[i]);

    return Py_BuildValue("");
}

static PyObject *ndpolate(PyObject *self, PyObject *args)
{
    PyArrayObject *query_pts, *indices;
    PyObject *axes, *hypercubes;

    PyArrayObject *hypercube;
    PyObject *interpolated_array;

    double *x, *fv;
    int *index;
    double *grid, *query_pt;

    double *interpolated_values;

    int i, j, k, nelems, naxes, dim, vdim;

    int debug = 0;

    if (!PyArg_ParseTuple(args, "OOOO", &query_pts, &indices, &axes, &hypercubes))
        return NULL;

    nelems = PyArray_DIM(indices, 0);
    naxes = PyArray_DIM(indices, 1);

    /* allocate the N-D point of interest to the maximum possible
     * dimension of the hypercubes to avoid having to allocate/free
     * its memory for each hypercube in the loop below. */
    x = malloc(naxes*sizeof(*x));

    index = (int *) PyArray_DATA(indices);
    query_pt = (double *) PyArray_DATA(query_pts);

    /* In order to allocate memory for the interpolated array, we need
     * to know the dimensionality of the vertex values. This information
     * is stored in hypercubes, so we grab the first hypercube just to
     * assign vdim and then reuse it in the loop over all nelems. Note
     * that the dimensionality of the hypercube itself does not play
     * a role here, only the dimensionality of its vertices, which are
     * the same for all hypercubes. */
    hypercube = (PyArrayObject *) PyTuple_GetItem(hypercubes, 0);
    vdim = PyArray_DIM(hypercube, PyArray_NDIM(hypercube)-1);

    if (debug) {
        printf("nelems=%d\n", nelems);
        printf("naxes=%d\n", naxes);
        printf("vdim=%d\n", vdim);
    }

    interpolated_values = malloc(nelems*vdim*sizeof(*interpolated_values));

    for (i = 0; i < nelems; i++) {
        hypercube = (PyArrayObject *) PyTuple_GetItem(hypercubes, i);

        dim = PyArray_NDIM(hypercube) - 1;  /* hypercube dimensionality */
        grid = (double *) PyArray_DATA(hypercube);

        if (debug) {
            printf("hypercube %d, dim=%d\n", i, dim);
            printf("  query_pt = [");
            for (j = 0; j < naxes-1; j++) {
                printf("%lf, ", query_pt[i*naxes+j]);
            }
            printf("%lf]\n", query_pt[i*naxes+j]);

            printf("  indices = [");
            for (j = 0; j < naxes-1; j++) {
                printf("%d, ", index[i*naxes+j]);
            }
            printf("%d]\n", index[i*naxes+j]);

            printf("  grid = [\n");
            for (j = 0; j < (1 << dim); j++) {
                printf("         [");
                for (k = 0; k < vdim-1; k++) {
                    printf(" %lf, ", grid[j*vdim+k]);
                }
                printf("%lf ],\n", grid[j*vdim+k]);
            }
            printf("         ]\n");
        }

        /* handle the case where the vertex itself was requested: */
        if (dim == 0) {
            memcpy(interpolated_values + i*vdim, grid, vdim*sizeof(*interpolated_values));
            continue;
        }

        /* since the interpolator modifies grid values, we need to work on a copy. */
        fv = malloc((1 << dim) * vdim * sizeof(*fv));
        memcpy(fv, grid, (1 << dim) * vdim * sizeof(*fv));

        k = 0;
        for (j = 0; j < naxes; j++) {
            /* skip when interpolant coordinate coincides with vertex coordinate: */
            if (index[i * naxes + j] < 0)
                continue;

            PyArrayObject *ax = (PyArrayObject *) PyTuple_GetItem(axes, j);
            double *axis = PyArray_DATA(ax);

            x[k] = (query_pt[i * naxes + j] - axis[index[i * naxes + j] - 1]) / (axis[index[i * naxes + j]] - axis[index[i * naxes + j] - 1]);
            k++;
        }

        c_ndpolate(dim, vdim, x, fv);

        if (debug) {
            printf("    fv = [");
            for (j = 0; j < vdim-1; j++) {
                printf(" %lf, ", fv[j]);
            }
            printf("%lf ]\n", fv[j]);
        }

        memcpy(interpolated_values + i*vdim, fv, vdim*sizeof(*interpolated_values));

        free(fv);
    }

    /* done with points of interest, free its memory: */
    free(x);

    /* Do not be tempted to flatten the array here, it needs to remain {nelems, vdim} even
     * if vdim == 1, otherwise all hell breaks loose with vectorized operations on the
     * numpy side. */
    npy_intp adim[] = {nelems, vdim};
    interpolated_array = PyArray_SimpleNewFromData(2, adim, NPY_DOUBLE, interpolated_values);
    PyArray_ENABLEFLAGS((PyArrayObject *) interpolated_array, NPY_ARRAY_OWNDATA);

    return interpolated_array;
}

static PyMethodDef cndpolator_methods[] = {
    {"ndpolate", ndpolate, METH_VARARGS, "C version of N-dimensional interpolation"},
    {"map", map, METH_VARARGS, "map axes and interpolants to unit cube"},
    {"find", find, METH_VARARGS, "find first greater-or-equal than"},
    {"ainfo", ainfo, METH_VARARGS, "array information for internal purposes"},
    {"hypercubes", hypercubes, METH_VARARGS, "create a hypercube from the passed indices and a function value grid"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cndpolator_module = {
    PyModuleDef_HEAD_INIT,
    "cndpolator",
    NULL, /* documentation */
    -1,
    cndpolator_methods};

PyMODINIT_FUNC PyInit_cndpolator(void)
{
    import_array();
    return PyModule_Create(&cndpolator_module);
}
