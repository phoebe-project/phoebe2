/**
 * @file ndpolator.c
 * @brief Main functions and python bindings.
 */

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/**
 * @private
 * @def PY_ARRAY_UNIQUE_SYMBOL
 * Required by numpy C-API. It defines a unique symbol to be used in other
 * C source files and header files.
 */

#define PY_ARRAY_UNIQUE_SYMBOL cndpolator_ARRAY_API

#include "ndpolator.h"
#include "ndp_types.h"

/**
 * @def min(a,b)
 * Computes the minimum of @p a and @p b.
 */

#define min(a,b) (((a)<(b))?(a):(b))

/**
 * @def max(a,b)
 * Computes the maximum of @p a and @p b.
 */

#define max(a,b) (((a)>(b))?(a):(b))

/**
 * <!-- _ainfo() -->
 * @private
 * @brief Internal function for printing ndarray flags from C.
 *
 * @param array numpy ndarray to be analyzed
 * @param print_data boolean, determines whether array contents should be
 * printed.
 *
 * @details
 * The function prints the dimensions, types, flags, and (if @p print_data is
 * TRUE) array contents. This is an internal function that should not be used
 * for anything other than debugging.
 */

void _ainfo(PyArrayObject *array, int print_data)
{
    int i, ndim, size;
    npy_intp *dims, *shape, *strides;

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
    for (i = 0; i < ndim - 1; i++)
        printf("%ld, ", dims[i]);
    printf("%ld]\n", dims[i]);

    shape = PyArray_SHAPE(array);
    printf("array->shape = [");
    for (i = 0; i < ndim - 1; i++)
        printf("%ld, ", shape[i]);
    printf("%ld]\n", shape[i]);

    strides = PyArray_STRIDES(array);
    printf("array->strides = [");
    for (i = 0; i < ndim - 1; i++)
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

    if (print_data) {
        if (PyArray_TYPE(array) == 5) {
            int *data = (int *) PyArray_DATA(array);
            printf("data = [");
            for (i = 0; i < size - 1; i++)
                printf("%d, ", data[i]);
            printf("%d]\n", data[i]);
        } else {
            double *data = (double *) PyArray_DATA(array);
            printf("data = [");
            for (i = 0; i < size - 1; i++)
                printf("%lf, ", data[i]);
            printf("%lf]\n", data[i]);
        }
    }

    return;
}

/**
 * <!-- find_first_geq_than() -->
 * @brief Finds the superior hypercube vertex for the passed parameter.
 *
 * @param axis an #ndp_axis instance to be searched, must be sorted in
 * ascending order
 * @param l index of the left search boundary in the @p axis, normally 0, but
 * can be anything between 0 and @p r-1
 * @param r index of the right search boundary in the @p axis, normally len(@p
 * axis)-1, but can be anything between @p l+1 and len(@p axis)-1
 * @param x value to be found in @p axis
 * @param rtol relative (fractional) tolerance to determine if @p x coincides
 * with a vertex in @p axis
 * @param flag flag placeholder; it will be populated with one of
 * #NDP_ON_GRID, #NDP_ON_VERTEX, #NDP_OUT_OF_BOUNDS
 *
 * @details
 * Uses bisection to find the index in @p axis that points to the first value
 * that is greater or equal to the requested value @p x. Indices @p l and @p r
 * can be used to narrow the search within the array. When the suitable index
 * is found, a flag is set to #NDP_ON_GRID if @p x is in the array's value
 * span, #NDP_OUT_OF_BOUNDS is @p x is either smaller than @p axis[0] or
 * larger than @p axis[N-1], and #NDP_ON_VERTEX if
 * @p x is within @p rtol of the value in the array.
 *
 * @return index of the first value in the array that is greater-or-equal-to
 * the requested value @p x. It also sets the @p flag accordingly. 
 */

int find_first_geq_than(ndp_axis *axis, int l, int r, double x, double rtol, int *flag)
{
    int m = l + (r - l) / 2;

    while (l != r) {
        if (x > axis->val[m])
            l = m + 1;
        else
            r = m;

        m = l + (r - l) / 2;
    }

    *flag = (x < axis->val[0] || x > axis->val[axis->len-1]) ? NDP_OUT_OF_BOUNDS : NDP_ON_GRID;

    if ((l > 0 && fabs( (x - axis->val[l]) / (axis->val[l]-axis->val[l-1]) ) < rtol) ||
        (l < axis->len && fabs((x-axis->val[l]) / (axis->val[l+1] - axis->val[l])) < rtol))
        *flag |= NDP_ON_VERTEX;

    return l;
}

/**
 * <!-- idx2pos() -->
 * @brief Converts an array of indices into an integer position of the array.
 *
 * @param axes a ndp_axes structure that holds all ndpolator axes
 * @param vdim vertex length (number of function values per grid point)
 * @param index a naxes-dimensional array of indices
 * @param pos placeholder for the position index in the NDP grid that
 * corresponds to per-axis indices
 *
 * @details
 * For efficiency, all ndpolator arrays are 1-dimensional, where axes are
 * stacked in the usual C order (last axis runs first). Referring to grid
 * elements can be done either by position in the 1-dimensional array, or
 * per-axis indices. This function converts from the index representation to
 * position.
 *
 * @return #ndp_status.
 */

int idx2pos(ndp_axes *axes, int vdim, int *index, int *pos)
{
    *pos = axes->cplen[0]*index[0];
    for (int i = 1; i < axes->len; i++)
        *pos += axes->cplen[i]*index[i];
    *pos *= vdim;

    return NDP_SUCCESS;
}

/**
 * <!-- pos2idx() -->
 * @brief Converts position in the array into an array of per-axis indices.
 *
 * @param axes a ndp_axes structure that holds all ndpolator axes
 * @param vdim vertex length (number of function values per grid point)
 * @param pos position index in the grid
 * @param idx an array of per-axis indices; must be allocated
 *
 * @details
 * For efficiency, all ndpolator arrays are 1-dimensional, where axes are
 * stacked in the usual C order (last axis runs first). Referring to grid
 * elements can be done either by position in the 1-dimensional array, or
 * per-axis indices. This function converts from position index representation
 * to an array of per-axis indices.
 * 
 * @return #ndp_status.
 */

int pos2idx(ndp_axes *axes, int vdim, int pos, int *idx)
{
    int debug = 0;

    for (int i=0; i < axes->len; i++)
        idx[i] = pos / vdim / axes->cplen[i] % axes->axis[i]->len;

    if (debug) {
        printf("pos = %d idx = [", pos);
        for (int j = 0; j < axes->len; j++)
            printf("%d ", idx[j]);
        printf("\b]\n");
    }

    return NDP_SUCCESS;
}

/**
 * <!-- c_ndpolate() -->
 * @brief Linear interpolation and extrapolation on a fully defined hypercube.
 *
 * @param naxes ndpolator dimension (number of axes)
 * @param vdim vertex length (number of function values per grid point)
 * @param x point of interest
 * @param fv naxes-dimensional unit hypercube of function values
 *
 * @details
 * Interpolates (or extrapolates) function values on a @p naxes -dimensional
 * fully defined hypercube in a query point @p x. Function values are
 * @p vdim -dimensional. The hypercube is assumed normalized and @p x
 * components are relative to the hypercube. For example, if @p naxes = 3, the
 * hypercube will be 2<sup>3</sup> = 8-dimensional, with hypercube vertices at
 * {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1,
 * 0), (1, 1, 1)}, and @p x a 3-dimensional array w.r.t. the hypercube, i.e.
 * @p x = (0.3, 0.4, 0.6) for interpolation, or @p x = (-0.2, -0.3, 0.6) for
 * extrapolation.
 *
 * @warning
 * For optimization purposes, the function overwrites @p fv values. The user
 * is required to make a copy of the @p fv array if the contents are meant to
 * be reused.
 *
 * @return #ndp_status status code.
 */

int c_ndpolate(int naxes, int vdim, double *x, double *fv)
{
    int i, j, k;

    for (i = 0; i < naxes; i++) {
        // printf("naxes=%d x[%d]=%3.3f\n", naxes, i, x[i]);
        for (j = 0; j < (1 << (naxes - i - 1)); j++) {
            // printf("j=%d fv[%d]=%3.3f, fv[%d]=%3.3f, ", j, (1 << (naxes - i - 1)) + j, fv[(1 << (naxes - i - 1)) + j], j, fv[j]);
            for (k = 0; k < vdim; k++) {
                // fv[j] += (fv[(1 << (naxes - i - 1)) + j]-fv[j]) * x[i];
                fv[j * vdim + k] += (fv[((1 << (naxes - i - 1)) + j) * vdim + k] - fv[j * vdim + k]) * x[i];
            }
            // printf("corr=%3.3f\n", fv[j]);
        }
    }

    return NDP_SUCCESS;
}

/**
 * <!-- find_nearest() -->
 * @brief Finds the nearest defined value on the grid.
 *
 * @param normed_elem unit hypercube-normalized query point
 * @param elem_index superior corner of the containing/nearest hypercube
 * @param elem_flag flag per component of the query point
 * @param table a #ndp_table instance with full ndpolator definition
 * @param mask a boolean array that flags grid elements as defined
 *
 * @details
 * When extrapolating, we need to either find the nearest defined vertex or
 * the nearest defined hypercube.
 *
 * Parameter @p normed_elem provides coordinates of the query point in unit
 * hypercube space. For example, `normed_elem=(0.3, 0.8, 0.2)` provides
 * coordinates of the query point with respect to the inferior hypercube
 * corner, which in this case would be within the hypercube. On the other
 * hand, `normed_elem=(-0.2, 0.3, 0.4)` would lie outside of the hypercube.
 *
 * Parameter @p elem_index provides coordinates of the superior hypercube
 * corner (i.e., indices of each axis where the corresponding value is the
 * first value greater than the query point coordinate). For example, if the
 * query point (in index space) is `(4.2, 5.6, 8.9)`, then `elem_index=(5, 6,
 * 9)`.
 *
 * Parameter @p elem_flag flags each coordinate of the @p normed_elem. Flags
 * can be either #NDP_ON_GRID, #NDP_ON_VERTEX, or #NDP_OUT_OF_BOUNDS. This is
 * important because @p elem_index points to the nearest larger axis value if
 * the coordinate does not coincide with the axis vertex, and it points to
 * the vertex itself if it coincides with the coordinate. For example, if
 * `axis=[0,1,2,3,4]` and the requested element is `1.5`, then @p elem_index
 * will equal 2; if the requested element is `1.0`, then @p elem_index will
 * equal 1; if the requested element is `-0.3`, then @p elem_index will equal
 * 0. In order to correctly account for out-of-bounds and on-vertex requests,
 * the function needs to be aware of the flags.
 * 
 * Parameter @p table is an #ndp_table structure that defines all relevant
 * grid parameters. Of particular use here is the grid of function values and
 * all axis definitions.
 *
 * Finally, parameter @p mask flags basic grid points with defined values. Two
 * masks are precomputed during #ndp_table initialization: @p table->vmask and
 * @p table->hcmask. The first one masks defined vertices, and the second one
 * masks fully defined hypercubes. If a vertex (or a hypercube) is defined,
 * the value of the mask is set to 1; otherwise it is set to 0. These arrays
 * have @p table->nverts elements, which equals to the product of the lengths
 * of all basic axes.
 *
 * The function computes Euclidean square distances for each masked grid point
 * from the requested element and returns the pointer to the nearest function
 * value. The search is optimized by searching over basic axes first.
 *
 * @return allocated pointer to the nearest coordinates. The calling function
 * must free the memory once done.
 */

int *find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, int *mask)
{
    int min_pos;
    double dist, min_dist = 1e50;
    int *coords = malloc(table->axes->len * sizeof(*coords));

    for (int i = 0; i < table->nverts; i++) {
        /* skip if vertex is undefined: */
        if (!mask[i])
            continue;

        /* find the distance to the basic vertex: */
        dist = 0.0;
        for (int j = 0; j < table->axes->nbasic; j++) {
            /* converts running index to j-th coordinate: */
            int coord = i / (table->axes->cplen[j] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[j]->len;
            if ((NDP_ON_VERTEX & elem_flag[j]) || (NDP_OUT_OF_BOUNDS & elem_flag[j]))
                dist += (elem_index[j]+normed_elem[j]-coord)*(elem_index[j]+normed_elem[j]-coord);
            else
                dist += (elem_index[j]+normed_elem[j]-1-coord)*(elem_index[j]+normed_elem[j]-1-coord);
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_pos = i;
        }
    }

    /* Assemble the coordinates: */
    // printf("nearest = [");
    for (int i = 0; i < table->axes->nbasic; i++) {
        coords[i] = min_pos / (table->axes->cplen[i] / table->axes->cplen[table->axes->nbasic-1]) % table->axes->axis[i]->len;
        // printf("%d, ", coords[i]);
    }
    // printf("\b\b], dist=%f pos=", min_dist);

    for (int i = table->axes->nbasic; i < table->axes->len; i++)
        coords[i] = max(0, min(table->axes->axis[i]->len-1, round(elem_index[i]+normed_elem[i])));

    return coords;
}

/**
 * <!-- find_indices() -->
 * @brief Determines hypercube indices based on the passed query points.
 *
 * @param nelems number of query points
 * @param qpts query points, an @p nelems -by- @p naxes array of doubles
 * @param axes a @p qpdim -dimensional array of axes
 *
 * @details
 * Computes superior index of the n-dimensional hypercubes that contain query
 * points @p qpts. It does so by calling #find_first_geq_than() sequentially
 * for all @p axes.
 *
 * When any of the query point components coincides with the grid vertex, that
 * component will be flagged by #NDP_ON_VERTEX. This is used in
 * #find_hypercubes() to reduce the dimensionality of the corresponding
 * hypercube. Any query point components that fall outside of the grid
 * boundaries are flagged by #NDP_OUT_OF_BOUNDS. Finally, all components that
 * do fall within the grid are flagged by #NDP_ON_GRID.
 *
 * @return a #ndp_query_pts instance.
 */

ndp_query_pts *find_indices(int nelems, double *qpts, ndp_axes *axes)
{
    ndp_query_pts *query_pts = ndp_query_pts_new();
    double rtol = 1e-3;  /* relative tolerance for vertex matching */
    int debug = 1;

    ndp_query_pts_alloc(query_pts, nelems, axes->len);

    if (debug) {
        printf("find_indices():\n  number of query points=%d\n  query point dimension=%d\n", nelems, axes->len);
        for (int i = 0; i < axes->len; i++) {
            printf("  axis %d (length %d):\n    [", i, axes->axis[i]->len);
            for (int j = 0; j < axes->axis[i]->len; j++) {
                printf("%2.2f ", axes->axis[i]->val[j]);
            }
            printf("\b]\n");
        }
    }

    for (int i = 0; i < axes->len; i++) {
        for (int j = 0; j < nelems; j++) {
            int k = j*axes->len + i;
            double lo, hi;
            query_pts->requested[k] = qpts[k];
            query_pts->indices[k] = find_first_geq_than(axes->axis[i], 0, axes->axis[i]->len - 1, qpts[k], rtol, &query_pts->flags[k]);
            if ((query_pts->flags[k] & NDP_ON_VERTEX) == NDP_ON_VERTEX) {
                query_pts->normed[k] = 0.0;
                continue;
            }
            /* no need to worry about the upper boundary here, indices can't ever be above axlen-1 */
            lo = axes->axis[i]->val[max(0, query_pts->indices[k]-1)];
            hi = axes->axis[i]->val[max(1, query_pts->indices[k])];
            query_pts->normed[k] = (qpts[k] - lo)/(hi - lo);
        }
    }

    if (debug) {
        for (int i = 0; i < nelems; i++) {
            printf("  query_pt[%d] = [", i);
            for (int j = 0; j < axes->len; j++) {
                printf("%2.2f ", qpts[i*axes->len + j]);
            }
            printf("\b]");

            printf("  indices = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%d ", query_pts->indices[i*axes->len + j]);
            }
            printf("\b]");

            printf("  flags = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%d ", query_pts->flags[i*axes->len + j]);
            }
            printf("\b]");

            printf("  normed_query_pt = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%3.3f ", query_pts->normed[i*axes->len + j]);
            }
            printf("\b]\n");
        }
    }

    return query_pts;
}

/**
 * <!-- find_hypercubes() -->
 * @brief Determines n-dimensional hypercubes that contain (or are adjacent
 * to) the query points identified by indices.
 *
 * @param nelems number of query points
 * @param indices a @p nelems -by- @p naxes array of indices
 * @param flags a @p nelems -by- @p naxes array of flags
 * @param table a (@p naxes + 1)-dimensional grid of @p vdim -dimensional
 * arrays
 *
 * @details
 * Hypercubes are n-dimensional subgrids that contain the point of interest
 * (i.e., a query point). If the query point lies within the hypercube, the
 * ndpolator will interpolate based on the function values in the hypercube.
 * If the query point is adjacent to the hypercube, ndpolator will extrapolate
 * instead. The hypercubes here need not be fully defined (although this might
 * be a better place to do it rather than in the #ndpolate() function as it is
 * currently implemented).
 *
 * Depending on the @p flag, the dimension of the hypercube can be reduced. In
 * particular, if any query point component flag is set to #NDP_ON_VERTEX, then the
 * corresponding dimension is eliminated (there is no need to interpolate or
 * extrapolate when the value is already on the axis).
 *
 * @return a #hypercube_info struct of hypercubes.
 */

ndp_hypercube **find_hypercubes(ndp_query_pts *qpts, ndp_table *table)
{
    int fdhc, tidx, *iptr;
    int dim_reduction, hc_size;
    double *hc_vertices;

    ndp_axes *axes = table->axes;
    int cidx[axes->len];

    int nelems = qpts->nelems;
    int *indices = qpts->indices;
    int *flags = qpts->flags;

    ndp_hypercube **hypercubes = malloc(nelems * sizeof(*hypercubes));

    int debug = 1;

    for (int i = 0; i < nelems; i++) {
        /* assume the hypercube (or the relevant subcube) is fully defined: */
        fdhc = 1;

        /* if qpts are out of bounds, set fdhc to 0: */
        for (int j = 0; j < axes->len; j++) {
            int pos = i * axes->len + j;
            if (NDP_OUT_OF_BOUNDS & flags[pos])
                fdhc = 0;
        }

        /* point iptr to the i-th index multiplet: */
        iptr = indices + i*axes->len;

        // do not check whether the hypercube is fully defined before reducing
        // its dimension: it may happen that we don't need the undefined parts
        // of the hypercube!

        /* reduce hypercube dimension for each query point component that coincides with the grid vertex: */
        dim_reduction = 0;
        for (int j = 0; j < axes->len; j++)
            if ((NDP_ON_VERTEX & flags[i*axes->len+j]))
                dim_reduction++;

        hc_size = axes->len-dim_reduction;
        hc_vertices = malloc(table->vdim * (1 << hc_size) * sizeof(*hc_vertices));

        if (debug) {
            printf("hypercube %d:\n", i);
            printf("  basic indices: [");
            for (int j = 0; j < axes->nbasic; j++)
                printf("%d ", iptr[j]);
            printf("\b]\n");
            printf("  hypercube size: %d\n", hc_size);
        }

        for (int j = 0; j < (1 << hc_size); j++) {
            for (int k = 0, l = 0; k < axes->len; k++) {
                if (NDP_ON_VERTEX & flags[i*axes->len+k]) {
                    cidx[k] = iptr[k];
                    continue;
                }
                cidx[k] = max(iptr[k]-1+(j / (1 << (hc_size-l-1))) % 2, (j / (1 << (hc_size-l-1))) % 2);
                l++;
            }
            if (debug) {
                printf("    cidx = [");
                for (int k = 0; k < axes->len; k++)
                    printf("%d ", cidx[k]);
                printf("\b], ");
            }

            idx2pos(axes, table->vdim, cidx, &tidx);
            if (table->grid[tidx] != table->grid[tidx])  /* true if nan */
                fdhc = 0;

            if (debug)
                printf("  tidx = %d, table[tidx] = %f\n", tidx, table->grid[tidx]);

            memcpy(hc_vertices + j*table->vdim, table->grid + tidx, table->vdim*sizeof(*hc_vertices));
        }

        ndp_hypercube *hypercube = ndp_hypercube_new_from_data(hc_size, table->vdim, fdhc, hc_vertices);
        if (debug)
            ndp_hypercube_print(hypercube, "    ");

        hypercubes[i] = hypercube;
    }

    return hypercubes;
}

/**
 * <!-- ndpolate() -->
 * @brief Runs linear interpolation or extrapolation in n dimensions.
 *
 * @param nelems number of query points
 * @param query_pts the flattened array of query points: for an n-dimensional
 * space, the @p query_pts array needs to be of length @p nelems x @p n, where
 * the flattening is done in the C order (last axis runs first)
 * @param table an #ndp_table structure that has all identifying information
 * on the interpolating grid itself
 * @param extrapolation_method how extrapolation should be done; one of
 * #NDP_METHOD_NONE, #NDP_METHOD_NEAREST, or #NDP_METHOD_LINEAR.
 *
 * @details
 * This is the main workhorse on the ndpolator module. It assumes that the
 * main #ndp_table @p table structure has been set up. It takes the points of
 * interest @p query_pts and it calls #find_indices() and #find_hypercubes()
 * consecutively, to populate the #ndp_query structure. While at it, the
 * function also checks whether any of the query point components are out of
 * bounds (flag = #NDP_OUT_OF_BOUNDS) and it prepares those query points for
 * extrapolation, depending on the @p extrapolation_method parameter.
 *
 * Once the supporting structures are initialized and populated, #ndpolate()
 * will first handle the out-of-bounds elements. It will set the value of NAN
 * if @p extrapolation_method = #NDP_METHOD_NONE, find the nearest defined
 * grid vertex by using #find_nearest() and set the value to the found nearest
 * value if @p extrapolation_method = #NDP_METHOD_NEAREST, and lookup the
 * nearest fully defined hypercube for extrapolation if
 * @p extrapolation_method = #NDP_METHOD_LINEAR.
 *
 * Finally, the ndpolator will loop through all hypercubes and call
 * #c_ndpolate() to get the interpolated or extrapolated function values for
 * each query point. The results are stored in the #ndp_query structure.
 *
 * @return a #ndp_query structure that holds all information on the specific
 * ndpolator run.
 */

ndp_query *ndpolate(ndp_query_pts *qpts, ndp_table *table, ndp_extrapolation_method extrapolation_method)
{
    ndp_query *query = ndp_query_new();
    double selected[table->axes->len];
    ndp_hypercube *hypercube;

    int debug = 1;
    int nelems = qpts->nelems;

    query->hypercubes = find_hypercubes(qpts, table);

    if (debug) {
        for (int i = 0; i < qpts->nelems; i++) {
            ndp_hypercube *hypercube = query->hypercubes[i];
            printf("  hypercube %d: dim=%d vdim=%d fdhc=%d v=[", i, hypercube->dim, hypercube->vdim, hypercube->fdhc);
            for (int j = 0; j < 1 << hypercube->dim; j++) {
                printf("{");
                for (int k = 0; k < hypercube->vdim; k++)
                    printf("%2.2f, ", hypercube->v[j*hypercube->vdim+k]);
                printf("\b\b} ");
            }
            printf("\b] indices={");
            for (int j = 0; j < table->axes->len; j++)
                printf("%d, ", qpts->indices[i*table->axes->len+j]);
            printf("\b\b} flags={");
            for (int j = 0; j < table->axes->len; j++)
                printf("%d, ", qpts->flags[i*table->axes->len+j]);
            printf("\b\b}\n");
        }
    }

    query->interps = malloc(nelems * table->vdim * sizeof(*(query->interps)));
    for (int i = 0; i < nelems; i++) {
        /* handle out-of-bounds elements first: */
        if (!query->hypercubes[i]->fdhc) {
            switch (extrapolation_method) {
                case NDP_METHOD_NONE:
                    for (int j = 0; j < table->vdim; j++)
                        query->interps[i*table->vdim+j] = NAN;
                    continue;
                break;
                case NDP_METHOD_NEAREST: {
                    double *normed_elem = qpts->normed + i * table->axes->len;
                    int *elem_index = qpts->indices + i * table->axes->len;
                    int *elem_flag = qpts->flags + i * table->axes->len;
                    int pos;

                    int *coords = find_nearest(normed_elem, elem_index, elem_flag, table, table->vmask);
                    idx2pos(table->axes, table->vdim, coords, &pos);
                    memcpy(query->interps + i*table->vdim, table->grid + pos, table->vdim * sizeof(*(query->interps)));
                    free(coords);
                    continue;
                }
                break;
                case NDP_METHOD_LINEAR: {
                    double *normed_elem = qpts->normed + i * table->axes->len;
                    int *elem_index = qpts->indices + i * table->axes->len;
                    int *elem_flag = qpts->flags + i * table->axes->len;
                    int cidx[table->axes->len];
                    int pos;

                    int *coords = find_nearest(normed_elem, elem_index, elem_flag, table, table->hcmask);
                    double *hc_vertices = malloc(table->vdim * (1 << table->axes->len) * sizeof(*hc_vertices));

                    if (debug) {
                        printf("  coords = [");
                        for (int k = 0; k < table->axes->len; k++) {
                            printf("%d ", coords[k]);
                        }
                        printf("\b]\n");
                    }

                    for (int j = 0; j < (1 << table->axes->len); j++) {
                        printf("  cidx[%d] = [", j);
                        for (int k = 0; k < table->axes->len; k++) {
                            cidx[k] = max(coords[k]-1+(j / (1 << (table->axes->len-k-1))) % 2, (j / (1 << (table->axes->len-k-1))) % 2);
                            printf("%d ", cidx[k]);
                        }
                        printf("\b]\n");

                        idx2pos(table->axes, table->vdim, cidx, &pos);
                        memcpy(hc_vertices + j * table->vdim, table->grid + pos, table->vdim * sizeof(*hc_vertices));
                    }

                    ndp_hypercube_free(query->hypercubes[i]);
                    hypercube = query->hypercubes[i] = ndp_hypercube_new_from_data(table->axes->len, table->vdim, /* fdhc = */ 1, hc_vertices);
                    ndp_hypercube_print(hypercube, "    ");

                    /* shift indices and normed query points to account for the new hypercube: */
                    printf("  updated query_pt[%d] = [", i);
                    for (int j = 0; j < table->axes->len; j++) {
                        printf("%3.3f->", qpts->normed[i * table->axes->len + j]);
                        qpts->normed[i * table->axes->len + j] += qpts->indices[i * table->axes->len + j] - coords[j] + (qpts->flags[i * table->axes->len + j] == NDP_OUT_OF_BOUNDS) + ((NDP_ON_VERTEX & qpts->flags[i * table->axes->len + j]) == NDP_ON_VERTEX && qpts->indices[i * table->axes->len + j] > 0);
                        printf("%3.3f ", qpts->normed[i * table->axes->len + j]);
                    }
                    printf("\b]\n");

                    c_ndpolate(hypercube->dim, hypercube->vdim, &qpts->normed[i * table->axes->len], hypercube->v);
                    memcpy(query->interps + i*table->vdim, hypercube->v, table->vdim * sizeof(*(query->interps)));
                    free(coords);
                    continue;
                }
                break;
                default:
                    /* invalid extrapolation method */
                    return NULL;
                break;
            }
        }
        else {
            /* continue with regular interpolation: */
            hypercube = query->hypercubes[i];
        }

        printf("selected = [");
        for (int j=0, k=0; j < table->axes->len; j++) {
            /* skip when queried coordinate coincides with a vertex: */
            if (qpts->flags[i * table->axes->len + j] == NDP_ON_VERTEX)
                continue;
            selected[k] = qpts->normed[i * table->axes->len + j];
            printf("%lf ", selected[k]);
            k++;
        }
        printf("\b]\n");

        if (debug) {
            printf("  i=%d dim=%d vdim=%d nqpts=[", i, hypercube->dim, hypercube->vdim);
            for (int j = 0; j < table->axes->len; j++)
                printf("%2.2f ", qpts->normed[i*table->axes->len + j]);
            printf("\b]\n");
        }

        c_ndpolate(hypercube->dim, hypercube->vdim, selected, hypercube->v);
        memcpy(query->interps + i*table->vdim, hypercube->v, table->vdim * sizeof(*(query->interps)));
    }

    return query;
}

/**
 * <!-- py_find() -->
 * @brief Python wrapper to the #find_indices() function.
 * 
 * @param self reference to the module object
 * @param args tuple (axes, query_pts)
 * 
 * @details
 * The wrapper takes a tuple of axes and an ndarray of query points, and it
 * calls #find_indices() to compute the indices, flags, and unit-normalized
 * query points w.r.t. the corresponding hypercube. These are returned in a
 * tuple to the calling function.
 * 
 * @note: In most (if not all) practical circumstances this function should
 * not be used because of the C-python data translation overhead. Instead,
 * use #py_ndpolate() instead as all allocation is done in C.
 * 
 * @return a tuple of (indices, flags, normed_query_pts).
 */

static PyObject *py_find(PyObject *self, PyObject *args)
{
    PyObject *py_axes, *py_indices, *py_flags, *py_normed_query_pts, *py_combo;
    PyArrayObject *py_query_pts;

    npy_intp *query_pts_shape;

    int i, naxes, nelems, nbasic = 0;

    double *qpts;
    double *normed_qpts;

    ndp_axis **axis;
    ndp_axes *axes;
    ndp_query_pts *query_pts;

    if (!PyArg_ParseTuple(args, "OO|i", &py_axes, &py_query_pts, &nbasic))
        return NULL;

    naxes = PyTuple_Size(py_axes);
    nelems = PyArray_DIM(py_query_pts, 0);
    qpts = (double *) PyArray_DATA(py_query_pts);
    if (nbasic == 0) nbasic = naxes;

    normed_qpts = malloc(nelems * naxes * sizeof(*normed_qpts));

    query_pts_shape = PyArray_SHAPE(py_query_pts);

    axis = malloc(naxes*sizeof(*axis));

    for (i = 0; i < naxes; i++) {
        PyArrayObject *py_axis = (PyArrayObject *) PyTuple_GetItem(py_axes, i);
        axis[i] = ndp_axis_new_from_data(PyArray_SIZE(py_axis), (double *) PyArray_DATA(py_axis));
    }

    axes = ndp_axes_new_from_data(naxes, nbasic, axis);
    query_pts = find_indices(nelems, qpts, axes);

    /* clean up: */
    for (i = 0; i < naxes; i++)
        free(axes->axis[i]);
    ndp_axes_free(axes);

    py_indices = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, query_pts->indices);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_indices, NPY_ARRAY_OWNDATA);

    py_flags = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, query_pts->flags);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_flags, NPY_ARRAY_OWNDATA);

    py_normed_query_pts = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_DOUBLE, query_pts->normed);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_normed_query_pts, NPY_ARRAY_OWNDATA);

    py_combo = PyTuple_New(3);
    PyTuple_SET_ITEM(py_combo, 0, py_indices);
    PyTuple_SET_ITEM(py_combo, 1, py_flags);
    PyTuple_SET_ITEM(py_combo, 2, py_normed_query_pts);

    return py_combo;
}

/**
 * <!-- py_hypercubes() -->
 * @brief Python wrapper to the #find_hypercubes() function.
 *
 * @param self reference to the module object
 * @param args tuple (indices, axes, flags, grid)
 *
 * @details
 * The wrapper takes a tuple of indices, axes, flags and function value grid,
 * and it calls #find_hypercubes() to compute the hypercubes, reducing their
 * dimension when possible. Hypercubes are returned to the calling function.
 *
 * @note: In most (if not all) practical circumstances this function should
 * not be used because of the C-python data translation overhead. Instead, use
 * #py_ndpolate() instead as all allocation is done in C.
 *
 * @return an ndarray of hypercubes.
 */

static PyObject *py_hypercubes(PyObject *self, PyObject *args)
{
    PyArrayObject *py_indices, *py_flags, *py_grid;
    PyObject *py_axes;

    ndp_table *table;
    ndp_query_pts *qpts;

    int *indices, *flags;
    int nelems, naxes;
    int nbasic = 0;

    PyObject *py_hypercubes;
    ndp_hypercube **hypercubes;

    if (!PyArg_ParseTuple(args, "OOOO|i", &py_indices, &py_axes, &py_flags, &py_grid, &nbasic))
        return NULL;

    nelems = PyArray_DIM(py_indices, 0);
    naxes = PyArray_DIM(py_indices, 1);
    if (nbasic == 0) nbasic = naxes;

    indices = (int *) PyArray_DATA(py_indices);
    flags = (int *) PyArray_DATA(py_flags);

    qpts = ndp_query_pts_new_from_data(nelems, naxes, indices, flags, /* requested= */ NULL, /* normed= */ NULL);

    py_hypercubes = PyTuple_New(nelems);

    table = ndp_table_new_from_python(py_axes, nbasic, py_grid);

    hypercubes = find_hypercubes(qpts, table);

    for (int i = 0; i < nelems; i++) {
        npy_intp shape[hypercubes[i]->dim+1];
        PyObject *py_hypercube;
        int j;

        for (j = 0; j < hypercubes[i]->dim; j++)
            shape[j] = 2;
        shape[j] = hypercubes[i]->vdim;

        py_hypercube = PyArray_SimpleNewFromData(hypercubes[i]->dim+1, shape, NPY_DOUBLE, hypercubes[i]->v);
        PyArray_ENABLEFLAGS((PyArrayObject *) py_hypercube, NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(py_hypercubes, i, py_hypercube);
    }

    return py_hypercubes;
}

/**
 * <!-- py_ainfo -->
 * @private
 * @brief Python wrapper to the #_ainfo() function.
 * 
 * @param self reference to the module object
 * @param args tuple (ndarray | print_data)
 * 
 * @details
 * Prints information on the passed array (its dimensions, flags and content
 * if @p print_data = True).
 * 
 * @return None
 */

static PyObject *py_ainfo(PyObject *self, PyObject *args)
{
    int i, ndim, size, print_data = 1;
    PyArrayObject *array;
    npy_intp *dims, *shape, *strides;

    if (!PyArg_ParseTuple(args, "O|i", &array, &print_data))
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
    for (i = 0; i < ndim - 1; i++)
        printf("%ld, ", dims[i]);
    printf("%ld]\n", dims[i]);

    shape = PyArray_SHAPE(array);
    printf("array->shape = [");
    for (i = 0; i < ndim - 1; i++)
        printf("%ld, ", shape[i]);
    printf("%ld]\n", shape[i]);

    strides = PyArray_STRIDES(array);
    printf("array->strides = [");
    for (i = 0; i < ndim - 1; i++)
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

    if (print_data) {
        if (PyArray_TYPE(array) == 5) {
            int *data = (int *) PyArray_DATA(array);
            printf("data = [");
            for (i = 0; i < size - 1; i++)
                printf("%d, ", data[i]);
            printf("%d]\n", data[i]);
        } else {
            double *data = (double *) PyArray_DATA(array);
            printf("data = [");
            for (i = 0; i < size - 1; i++)
                printf("%lf, ", data[i]);
            printf("%lf]\n", data[i]);
        }
    }

    return Py_None;
}

/**
 * <!-- py_ndpolate() -->
 * @brief Python wrapper to the #ndpolate() function.
 *
 * @param self reference to the module object
 * @param args tuple (query_pts, axes, flags, grid | extrapolation_method)
 *
 * @details
 * The wrapper takes a tuple of query points, axes, flags and function value
 * grid, and it calls #ndpolate() to run interpolation and/or extrapolation in
 * all query points. Interpolated/extrapolated values are returned to the
 * calling function in a (nelems-by-vdim)-dimensional ndarray.
 *
 * @note This is the main (and probably the only practical) entry point from
 * python code to the C ndpolator module.
 * 
 * @return an ndarray of interpolated/extrapolated values.
 */

static PyObject *py_ndpolate(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *py_query_pts, *py_grid;
    PyObject *py_axes, *py_rv;
    int nbasic = 0;

    ndp_extrapolation_method extrapolation_method = NDP_METHOD_NONE;  /* default value */

    static char *kwlist[] = {"query_pts", "axes", "grid", "nbasic", "extrapolation_method", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ii", kwlist, &py_query_pts, &py_axes, &py_grid, &nbasic, &extrapolation_method))
        return NULL;

    ndp_table *table = ndp_table_new_from_python(py_axes, nbasic, py_grid);
    PyObject *py_capsule = PyCapsule_New((void *) table, NULL, NULL);

    int nelems = PyArray_DIM(py_query_pts, 0);
    double *qpts = PyArray_DATA(py_query_pts);

    ndp_query_pts *query_pts = find_indices(nelems, qpts, table->axes);

    ndp_query *query = ndpolate(query_pts, table, extrapolation_method);

    npy_intp adim[] = {nelems, table->vdim};
    PyObject *py_interps = PyArray_SimpleNewFromData(2, adim, NPY_DOUBLE, query->interps);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_interps, NPY_ARRAY_OWNDATA);

    py_rv = PyTuple_New(2);
    PyTuple_SetItem(py_rv, 0, py_interps);
    PyTuple_SetItem(py_rv, 1, py_capsule);

    return py_rv;
}

/**
 * <!-- _register_enum() -->
 * @private
 * @brief Helper function to transfer C enums to Python enums.
 * 
 * @param self reference to the module object
 * @param enum_name Python-side enum name string
 * @param py_enum Python dictionary that defines enumerated constants
 * 
 * @details
 * Registers an enumerated constant in Python.
 */

void _register_enum(PyObject *self, const char *enum_name, PyObject *py_enum)
{
    PyObject *py_enum_class = NULL;
    PyObject *py_enum_module = PyImport_ImportModule("enum");
    if (!py_enum_module)
        Py_CLEAR(py_enum);

    py_enum_class = PyObject_CallMethod(py_enum_module, "IntEnum", "sO", enum_name, py_enum);

    Py_CLEAR(py_enum);
    Py_CLEAR(py_enum_module);

    if (py_enum_class && PyModule_AddObject(self, enum_name, py_enum_class) < 0)
        Py_CLEAR(py_enum_class);
}

/**
 * <!-- ndp_register_enums() -->
 * @brief Translates and registers all C-side enumerated types into Python.
 * 
 * @param self reference to the module object
 * @return #ndp_status
 */

int ndp_register_enums(PyObject *self)
{
    PyObject* py_enum = PyDict_New();

    PyDict_SetItemString(py_enum, "NONE", PyLong_FromLong(NDP_METHOD_NONE));
    PyDict_SetItemString(py_enum, "NEAREST", PyLong_FromLong(NDP_METHOD_NEAREST));
    PyDict_SetItemString(py_enum, "LINEAR", PyLong_FromLong(NDP_METHOD_LINEAR));
    _register_enum(self, "ExtrapolationMethod", py_enum);

    return NDP_SUCCESS;
}

/**
 * @brief Standard python boilerplate code that defines methods present in this C module.
 */

static PyMethodDef cndpolator_methods[] =
{
    {"ndpolate", (PyCFunction) py_ndpolate, METH_VARARGS | METH_KEYWORDS, "C implementation of N-dimensional interpolation"},
    {"find", py_find, METH_VARARGS, "find first greater-or-equal than"},
    {"hypercubes", py_hypercubes, METH_VARARGS, "create a hypercube from the passed indices and a function value grid"},
    {"ainfo", py_ainfo, METH_VARARGS, "array information for internal purposes"},
    {NULL, NULL, 0, NULL}
};

/**
 * @brief Standard python boilerplate code that defines the ndpolator module.
 */

static struct PyModuleDef cndpolator_module = 
{
    PyModuleDef_HEAD_INIT,
    "cndpolator",
    NULL, /* documentation */
    -1,
    cndpolator_methods
};

/**
 * <!-- PyInit_cndpolator() -->
 * @private
 * @brief Initializes the ndpolator C module for Python.
 * 
 * @return PyMODINIT_FUNC 
 */

PyMODINIT_FUNC PyInit_cndpolator(void)
{
    PyObject *module;
    import_array();
    module = PyModule_Create(&cndpolator_module);
    ndp_register_enums(module);
    return module;
}
