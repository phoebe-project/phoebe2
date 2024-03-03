/**
 * @file ndp_types.c
 * @brief Ndpolator's type constructors and destructors.
 */

#include <stdlib.h>
#include <stdio.h>

/**
 * @def NO_IMPORT_ARRAY
 * Required by numpy C-API. It tells the wrapper that numpy's `import_array()` is not
 * invoked in this source file but that the API does use numpy arrays.
 */

#define NO_IMPORT_ARRAY

/**
 * @private
 * @def PY_ARRAY_UNIQUE_SYMBOL
 * Required by numpy C-API. It defines a unique symbol that is used in c/h source
 * files that do not call `import_array()`.
 */

#define PY_ARRAY_UNIQUE_SYMBOL cndpolator_ARRAY_API

#include "ndp_types.h"

/**
 * @brief #ndp_axis constructor.
 *
 * @details
 * Initializes a new #ndp_axis instance. It sets @p len to 0 and it NULLifies
 * the @p val array.
 *
 * @return An initialized #ndp_axis instance.
 */

ndp_axis *ndp_axis_new()
{
    ndp_axis *axis = malloc(sizeof(*axis));

    axis->len = 0;
    axis->val = NULL;

    return axis;
}

/**
 * @brief #ndp_axis constructor from passed data.
 *
 * @param len length of the @p val array
 * @param val array of vertices that span the axis 
 *
 * @details
 * Initializes a new #ndp_axis instance, sets @p axis->len to @p len and @p
 * axis->val to @p val. Note that the function **does not copy the array**, it
 * only assigns a pointer to it. Thus, the calling function needs to pass an
 * allocated copy if the array is (re)used elsewhere. Ndpolator treats the @p
 * val array as read-only and it will not change it.
 *
 * @return An initialized #ndp_axis instance.
 */

ndp_axis *ndp_axis_new_from_data(int len, double *val)
{
    ndp_axis *axis = malloc(sizeof(*axis));

    axis->len = len;
    axis->val = val;

    return axis;
}

/**
 * @brief #ndp_axis destructor.
 * 
 * @param axis a #ndp_axis instance to be freed
 * 
 * @details
 * Frees memory allocated for the #ndp_axis instance. That includes the
 * @p val array memory, and the #ndp_axis instance itself.
 * 
 * @return #ndp_status code.
 */

int ndp_axis_free(ndp_axis *axis)
{
    if (axis->val)
        free(axis->val);
    free(axis);

    return NDP_SUCCESS;
}

/**
 * @brief #ndp_axes constructor.
 * 
 * @details
 * Initializes a new #ndp_axes instance. It sets @p len to 0 and
 * it NULLifies the @p cplen array.
 * 
 * @return An initialized #ndp_axes instance.
 */

ndp_axes *ndp_axes_new()
{
    ndp_axes *axes = malloc(sizeof(*axes));

    axes->len = 0;
    axes->cplen = NULL;

    return axes;
}

/**
 * @brief #ndp_axes constructor from passed data.
 * 
 * @param naxes number of axes to be stored in the #ndp_axes structure
 * @param nbasic number of basic axes among the passed axes
 * @param axis a @p naxes -dimensional array of #ndp_axis instances
 * 
 * @details
 * Initializes a new #ndp_axes instance, sets @p axes->len to @p naxes, and @p
 * axes->axis to @p axis. Note that the function **does not copy the array**, it
 * only assigns a pointer to it. Thus, the calling function needs to pass an
 * allocated copy if the array is (re)used elsewhere. Ndpolator treats the @p
 * axis array as read-only and it will not change it.
 *
 * @return An initialized #ndp_axes instance.
 */

ndp_axes *ndp_axes_new_from_data(int naxes, int nbasic, ndp_axis **axis)
{
    ndp_axes *axes = ndp_axes_new();

    axes->len = naxes;
    axes->nbasic = nbasic;
    axes->axis = axis;

    /* add a cumulative product array: */
    axes->cplen = malloc(naxes*sizeof(*(axes->cplen)));
    for (int i = 0; i < naxes; i++) {
        axes->cplen[i] = 1.0;
        for (int j = i+1; j < naxes; j++)
            axes->cplen[i] *= axes->axis[j]->len;
    }

    return axes;
}

/**
 * @brief #ndp_axes constructor from the passed python object.
 *
 * @param py_axes a tuple of ndarrays, one for each axis
 * @param nbasic an integer, the number of basic (spanning) axes
 *
 * @details
 * Initializes a new #ndp_axes instance by translating python data into C and
 * then calling #ndp_axes_new_from_data(). The passed python object must be a
 * tuple of numpy arrays, one for each axis.
 *
 * @return An initialized #ndp_axes instance.
 */

ndp_axes *ndp_axes_new_from_python(PyObject *py_axes, int nbasic)
{
    ndp_axes *axes;

    int naxes = PyTuple_Size(py_axes);
    ndp_axis **axis = malloc(naxes*sizeof(*axis));

    if (nbasic == 0) nbasic = naxes;

    for (int i = 0; i < naxes; i++) {
        PyArrayObject *py_axis = (PyArrayObject *) PyTuple_GetItem(py_axes, i);
        int py_axis_len = PyArray_DIM(py_axis, 0);
        double *py_axis_data = (double *) PyArray_DATA(py_axis);
        axis[i] = ndp_axis_new_from_data(py_axis_len, py_axis_data);
    }

    axes = ndp_axes_new_from_data(naxes, nbasic, axis);

    return axes;
}

/**
 * @brief #ndp_axes destructor.
 * 
 * @param axes a #ndp_axes instance to be freed
 * 
 * @details
 * Frees memory allocated for the #ndp_axes instance. That includes the
 * @p cplen array memory, and the #ndp_axes instance itself.
 * 
 * @return #ndp_status code.
 */

int ndp_axes_free(ndp_axes *axes)
{
    if (axes->cplen)
        free(axes->cplen);
    
    free(axes);

    return NDP_SUCCESS;
}

/**
 * @brief #ndp_query_pts constructor.
 *
 * @details
 * Initializes a new #ndp_query_pts instance. It sets @p nelems and @p naxes
 * to 0 and it NULLifies all arrays.
 *
 * @return An initialized #ndp_query_pts instance.
 */

ndp_query_pts *ndp_query_pts_new()
{
    ndp_query_pts *qpts = malloc(sizeof(*qpts));

    qpts->nelems = 0;
    qpts->naxes = 0;
    qpts->indices = NULL;
    qpts->flags = NULL;
    qpts->requested = NULL;
    qpts->normed = NULL;

    return qpts;
}

ndp_query_pts *ndp_query_pts_new_from_data(int nelems, int naxes, int *indices, int *flags, double *requested, double *normed)
{
    ndp_query_pts *qpts = malloc(sizeof(*qpts));

    qpts->nelems = nelems;
    qpts->naxes = naxes;
    qpts->indices = indices;
    qpts->flags = flags;
    qpts->requested = requested;
    qpts->normed = normed;

    return qpts;
}

/**
 * @brief An #ndp_query_pts instance memory allocator.
 *
 * @param qpts an #ndp_query_pts instance
 * @param nelems number of query points
 * @param naxes query points dimension (number of axes)
 *
 * @details
 * Allocates memory for the #ndp_query_pts instance. Each array in the struct
 * has @p nelems x @p naxes elements.
 *
 * @return int an #ndp_status code.
 */

int ndp_query_pts_alloc(ndp_query_pts *qpts, int nelems, int naxes)
{
    qpts->nelems = nelems;
    qpts->naxes = naxes;

    qpts->indices = malloc(nelems * naxes * sizeof(*(qpts->indices)));
    qpts->flags = malloc(nelems * naxes * sizeof(*(qpts->flags)));
    qpts->requested = malloc(nelems * naxes * sizeof(*(qpts->requested)));
    qpts->normed = malloc(nelems * naxes * sizeof(*(qpts->normed)));

    return NDP_SUCCESS;
}

/**
 * @brief #ndp_query_pts destructor.
 *
 * @param qpts a #ndp_query_pts instance to be freed
 *
 * @details
 * Frees memory allocated for the #ndp_query_pts instance. That includes all
 * array memory, and the #ndp_query_pts instance itself.
 *
 * @return #ndp_status code.
 */

int ndp_query_pts_free(ndp_query_pts *qpts)
{
    if (qpts->indices)
        free(qpts->indices);
    if (qpts->flags)
        free(qpts->flags);
    if (qpts->requested)
        free(qpts->normed);
    if (qpts->normed)
        free(qpts->normed);
    
    free(qpts);

    return NDP_SUCCESS;
}

/**
 * @brief #ndp_table constructor.
 *
 * @details
 * Initializes a new #ndp_table instance. It sets @p vdim, @p ndefs and @p
 * hcdefs to 0, and it NULLifies all arrays.
 *
 * @return An initialized #ndp_axes instance.
 */

ndp_table *ndp_table_new()
{
    ndp_table *table = malloc(sizeof(*table));
    table->vdim = 0;
    table->axes = NULL;
    table->grid = NULL;

    table->nverts = 0;
    table->vmask = NULL;
    table->hcmask = NULL;

    return table;
}

/**
 * @brief #ndp_table constructor from passed data.
 *
 * @param axes an #ndp_axes instance that stores all axis information
 * @param vdim function value (vertex) length
 * @param grid full grid of all defined function values
 *
 * @details
 * Initializes a new #ndp_table instance from passed data. Note that the
 * function **does not copy the arrays**, it only assigns pointers to them.
 * Thus, the calling function needs to pass allocated copies if the arrays are
 * (re)used elsewhere. Ndpolator treats all arrays as read-only and it will
 * not change them.
 *
 * This constructor also initializes a list of all non-nan vertices in the
 * grid. It does so by traversing the grid and storing their count in @p ndefs
 * and their vertex positions in the @p defined_vertices array.
 *
 * Finally, the constructor initalizes a list of all fully defined hypercubes
 * in the grid. It does so by traverysing defined vertex positions and
 * checking whether all hypercube vertices are defined. Their count is stored
 * in @p hcdefs, while their list is stored in the @p defined_hypercubes
 * array.
 *
 * @return An initialized #ndp_table instance.
 */

ndp_table *ndp_table_new_from_data(ndp_axes *axes, int vdim, double *grid)
{
    int debug = 0;
    int pos;
    int ith_corner[axes->nbasic], cidx[axes->nbasic];

    ndp_table *table = ndp_table_new();

    table->axes = axes;
    table->vdim = vdim;
    table->grid = grid;

    /* count all vertices in the grid: */
    table->nverts = 1;
    for (int i = 0; i < axes->nbasic; i++)
        table->nverts *= axes->axis[i]->len;

    /* collect all non-nan vertices: */
    table->vmask = calloc(table->nverts, sizeof(*(table->vmask)));
    for (int i = 0; i < table->nverts; i++) {
        pos = i*axes->cplen[axes->nbasic-1]*vdim;
        if (grid[pos] == grid[pos])  /* false if nan */
            table->vmask[i] = 1;
    }

    table->hcmask = calloc(table->nverts, sizeof(*(table->hcmask)));
    for (int i = 0; i < table->nverts; i++) {
        int nan_encountered = 0;

        /* skip undefined vertices: */
        if (table->vmask[i] == 0)
            continue;

        /* convert running index to per-axis indices of the superior corner of the hypercube: */
        for (int k = 0; k < axes->nbasic; k++) {
            ith_corner[k] = (i / (axes->cplen[k] / axes->cplen[axes->nbasic-1])) % axes->axis[k]->len;
            if (debug)
                printf("i=%d k=%d cplen[k]=%d cplen[nbasic-1]=%d num=%d\n", i, k, axes->cplen[k], axes->cplen[axes->nbasic-1], i / (axes->cplen[k] / axes->cplen[axes->nbasic-1]));
            /* skip edge elements: */
            if (ith_corner[k] == 0) {
                nan_encountered = 1;
                break;
            }
        }

        if (nan_encountered)
            continue;

        if (debug) {
            printf("i=% 3d c=[", i);
            for (int k = 0; k < axes->nbasic; k++)
                printf("%d ", ith_corner[k]);
            printf("\b]\n");
        }

        /* loop over all basic hypercube vertices and see if they're all defined: */
        for (int j = 0; j < 1 << table->axes->nbasic; j++) {                
            for (int k = 0; k < table->axes->nbasic; k++)
                cidx[k] = ith_corner[k]-1+(j / (1 << (table->axes->nbasic-k-1))) % 2;

            if (debug) {
                printf("  c%d=[", j);
                for (int k = 0; k < table->axes->nbasic; k++)
                    printf("%d ", cidx[k]);
                printf("\b]\n");
            }

            /* convert per-axis indices to running index: */
            pos = 0;
            for (int k = 0; k < table->axes->nbasic; k++)
                pos += cidx[k] * axes->cplen[k] / axes->cplen[axes->nbasic-1];

            if (!table->vmask[pos]) {
                nan_encountered = 1;
                break;
            }
        }

        if (nan_encountered)
            continue;
        
        table->hcmask[i] = 1;
    }

    if (debug) {
        for (int i = 0, sum = 0; i < table->nverts; i++) {
            sum += table->hcmask[i];
            if (i == table->nverts-1)
                printf("%d fully defined hypercubes found.\n", sum);
        }
    }

    return table;
}

/**
 * @brief #ndp_table constructor from the passed python objects.
 *
 * @param py_axes a tuple of ndarrays, one for each axis
 * @param py_nbasic an integer, number of basic (spanning) axes
 * @param py_grid a numpy ndarray of all function values
 *
 * @details
 * Initializes a new #ndp_table instance by translating python data into C and
 * then calling #ndp_table_new_from_data(). The passed @p py_axes parameter
 * must be a tuple of numpy arrays, one for each axis; the passed @p py_nbasic
 * must be an integer that provides the number of basic axes (<=
 * len(py_axes)), and the passed @p py_grid parameter must be a numpy array of
 * the shape (n1, n2, ..., nk, ..., nN, vdim), where nk is the length of the
 * k-th axis.
 *
 * @return An initialized #ndp_table instance.
 */

ndp_table *ndp_table_new_from_python(PyObject *py_axes, int nbasic, PyArrayObject *py_grid)
{
    ndp_axes *axes = ndp_axes_new_from_python(py_axes, nbasic);

    int ndims = PyArray_NDIM(py_grid);
    int vdim = PyArray_DIM(py_grid, ndims-1);

    /* work around the misbehaved array: */
    PyArrayObject *py_behaved_grid = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_grid, NPY_DOUBLE, NPY_ARRAY_CARRAY);
    double *grid = (double *) PyArray_DATA(py_behaved_grid);

    return ndp_table_new_from_data(axes, vdim, grid);
}

/**
 * @brief #ndp_table destructor.
 * 
 * @param table a #ndp_table instance to be freed
 * 
 * @details
 * Frees memory allocated for the #ndp_table instance. That includes all
 * array memory, and the #ndp_table instance itself.
 * 
 * @return #ndp_status code.
 */

int ndp_table_free(ndp_table *table)
{
    if (table->axes)
        ndp_axes_free(table->axes);

    if (table->grid)
        free(table->grid);

    if (table->vmask)
        free(table->vmask);

    if (table->hcmask)
        free(table->hcmask);

    free(table);

    return NDP_SUCCESS;
}

/**
 * @brief #ndp_hypercube constructor.
 *
 * @details
 * Initializes a new #ndp_hypercube instance. It sets @p dim to 0, and it
 * NULLifies the @v array.
 *
 * @return An initialized #ndp_hypercube instance.
 */

ndp_hypercube *ndp_hypercube_new()
{
    ndp_hypercube *hc = malloc(sizeof(*hc));
    hc->dim = 0;
    hc->vdim = 0;
    hc->v = NULL;
    return hc;
}

/**
 * @brief #ndp_hypercube constructor from passed data.
 *
 * @param dim hypercube dimension, typically equal to the number of axes
 * @param vdim grid function value length (a.k.a. vertex dimension)
 * @param fdhc fully defined hypercube flag (1 for fully defined, 0 if there
 * are nans among the function values)
 * @param v hypercube function values in 2<sup>dim</sup> vertices, each
 * @p vdim long
 *
 * @details
 * Initializes a new #ndp_hypercube instance. It populates all fields from
 * passed arguments. Note that the function **does not copy the array**, it
 * only assigns a pointer to it. Thus, the calling function needs to pass an
 * allocated copy if the array is (re)used elsewhere. Ndpolator treats the @p
 * v array as read-only and it will not change it.
 *
 * @return An initialized #ndp_hypercube instance.
 */

ndp_hypercube *ndp_hypercube_new_from_data(int dim, int vdim, int fdhc, double *v)
{
    ndp_hypercube *hc = malloc(sizeof(*hc));

    hc->dim = dim;
    hc->vdim = vdim;
    hc->fdhc = fdhc;
    hc->v = v;

    return hc;
}

/**
 * @brief Helper function that prints hypercube values.
 *
 * @param hc a #ndp_hypercube instance
 * @param prefix a string to be prepended to each printed line, typically used
 * for indentation
 *
 * @details
 * Prints the contents on the #ndp_hypercube @p hc, optionally prefixing it
 * with @p prefix.
 *
 * @return #ndp_status.
 */

int ndp_hypercube_print(ndp_hypercube *hc, const char *prefix)
{
    printf("%shc->dim = %d\n", prefix, hc->dim);
    printf("%shc->vdim = %d\n", prefix, hc->vdim);
    printf("%shc->fdhc = %d\n", prefix, hc->fdhc);

    printf("%shc->v = [", prefix);
    for (int i = 0; i < (1<<hc->dim); i++) {
        printf("{");
        for (int j = 0; j < hc->vdim; j++) {
            printf("%f ", hc->v[i*hc->vdim+j]);
        }
        printf("\b}, ");
    }
    printf("\b\b]\n");

    return NDP_SUCCESS;
}

/**
 * @brief #ndp_hypercube destructor.
 * 
 * @param hypercube a #ndp_hypercube instance to be freed
 * 
 * @details
 * Frees memory allocated for the #ndp_hypercube instance. That includes the @p v
 * array memory, and the #ndp_hypercube instance itself.
 * 
 * @return #ndp_status code.
 */

int ndp_hypercube_free(ndp_hypercube *hc)
{
    free(hc->v);
    free(hc);
    return NDP_SUCCESS;
}

/**
 * @brief #ndp_query constructor.
 *
 * @details
 * Initializes a new #ndp_query instance. It sets @p nelems to 0, and it
 * NULLifies all arrays in the struct.
 *
 * @return An initialized #ndp_query instance.
 */

ndp_query *ndp_query_new()
{
    ndp_query *query = malloc(sizeof(*query));

    return query;
}

/**
 * @brief #ndp_query destructor.
 *
 * @param query a #ndp_query instance to be freed
 *
 * @details
 * Frees memory allocated for the #ndp_query instance. That includes all array
 * memory, and the #ndp_query instance itself.
 *
 * @return #ndp_status code.
 */

int ndp_query_free(ndp_query *query)
{
    free(query);

    return NDP_SUCCESS;
}
