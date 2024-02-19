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
 * @param axis array of scalars to be searched, must be sorted in ascending
 * order
 * @param l index of the left search boundary in the @p axis, normally 0, but
 * can be anything between 0 and @p r-1
 * @param r index of the right search boundary in the @p axis, normally
 * len(@p axis)-1, but can be anything between @p l+1 and len(@p axis)-1
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

int find_first_geq_than(double *axis, int l, int r, double x, double rtol, int *flag)
{
    int m = l + (r - l) / 2;

    while (l != r) {
        if (axis[m] < x)
            l = m + 1;
        else
            r = m;

        m = l + (r - l) / 2;
    }

    *flag = (x < axis[0] || axis[r] < x) ? NDP_OUT_OF_BOUNDS : NDP_ON_GRID;

    if (fabs((x - axis[l - 1]) / (axis[l] - axis[l - 1])) < rtol) {
        *flag |= NDP_ON_VERTEX;
        return l-1;
        // return -(l - 1);
    }
    if (fabs((axis[l] - x) / (axis[l] - axis[l - 1])) < rtol) {
        *flag |= NDP_ON_VERTEX;
        return l;
        // return -l;
    }

    return l;
}

/**
 * <!-- idx2pos() -->
 * @brief Converts an array of indices into an integer position of the array.
 *
 * @param axes a ndp_axes structure that holds all ndpolator axes
 * @param vdim vertex length (number of function values per grid point)
 * @param index a naxes-dimensional array of indices
 *
 * @details
 * For efficiency, all ndpolator arrays are 1-dimensional, where axes are
 * stacked in the usual C order (last axis runs first). Referring to grid
 * elements can be done either by position in the 1-dimensional array, or
 * per-axis indices. This function converts from the index representation to
 * position.
 * 
 * @return position index in the NDP grid that corresponds to per-axis
 * indices.
 */

int idx2pos(ndp_axes *axes, int vdim, int *index)
{
    int pos = axes->cplen[0]*index[0];
    for (int i = 1; i < axes->len; i++)
        pos += axes->cplen[i]*index[i];

    return vdim*pos;
}

/**
 * <!-- pos2idx() -->
 * @brief Converts position in the array into an array of per-axis indices.
 *
 * @param axes a ndp_axes structure that holds all ndpolator axes
 * @param vdim vertex length (number of function values per grid point)
 * @param pos position index in the grid
 *
 * @details
 * For efficiency, all ndpolator arrays are 1-dimensional, where axes are
 * stacked in the usual C order (last axis runs first). Referring to grid
 * elements can be done either by position in the 1-dimensional array, or
 * per-axis indices. This function converts from position index representation
 * to an array of per-axis indices.
 * 
 * @return an array of per-axis indices
 */

int *pos2idx(ndp_axes *axes, int vdim, int pos)
{
    int debug = 0;
    int *index = malloc(axes->len * sizeof(*index));

    for (int i=0; i < axes->len; i++)
        index[i] = pos / vdim / axes->cplen[i] % axes->axis[i]->len;

    if (debug) {
        printf("pos = %d idx = [", pos);
        for (int j = 0; j < axes->len; j++)
            printf("%d ", index[j]);
        printf("\b]\n");
    }

    return index;
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
 * <!-- find_nearest_defined_index() -->
 * @brief Finds the nearest defined index (i.e., with non-nan function values)
 * on the grid.
 *
 * @param normed_elem unit hypercube-normalized query point
 * @param elem_index indices of the nearest hypercube's inferior corner
 * @param table a #ndp_table instance with full ndpolator definition
 *
 * @details
 * When #ndp_extrapolation_method is set to #NDP_METHOD_NEAREST, this function
 * is called to find the nearest defined vertex on the grid. The passed @p
 * table stores a list of all defined basic vertices in a (private) array
 * @p table->defined_vertices; this function computes Euclidean square
 * distances for each defined vertex from the requested element and returns
 * the position of the nearest element. The search is further optimized by
 * searching only through basic axes first.
 * 
 * The reason we need to pass both @p normed_elem and @p elem_index is because
 * @p normed_elem is given relative to the hypercube that is determined by @p
 * elem_index. As the search is done in absolute index space rather than axis
 * space, we need to convert @p normed_elem from relative to absolute indices.
 *
 * @return int 
 */

int find_nearest_defined_index(double *normed_elem, int *elem_index, ndp_table *table)
{
    int min_pos;
    double dist, min_dist = 1e50;

    int *grid_index;

    for (int i = 0; i < table->ndefs; i++) {
        /* convert position to indices: */
        grid_index = pos2idx(table->axes, table->vdim, table->defined_vertices[i]);

        // printf("i=%d pos=%d index=[", i, table->defined_vertices[i]);
        // for (int j = 0; j < table->axes->len; j++)
        //     printf("%d ", grid_index[j]);
        // printf("\b] ");

        /* find the distance to the basic vertex: */
        dist = 0.0;
        for (int j = 0; j < table->axes->nbasic; j++) {
            dist += (elem_index[j]+normed_elem[j]-grid_index[j])*(elem_index[j]+normed_elem[j]-grid_index[j]);
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_pos = table->defined_vertices[i];
        }

        // printf("dist=%e\n", dist);

        free(grid_index);
    }

    // printf("min dist=%e, pos=%d\n", min_dist, min_pos);

    /* attached axes are guaranteed to be defined for all basic vertices, so
     * we can round the normed element to get the index value:
     */
    for (int j = table->axes->nbasic; j < table->axes->len; j++) {
        min_pos += table->axes->cplen[j] * max(0, min(table->axes->axis[j]->len-1, round(elem_index[j]+normed_elem[j])));
    }

    return min_pos;
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
 * @return an #index_info struct of indices, flags and normalized query
 * points.
 */

struct index_info find_indices(int nelems, double *qpts, ndp_axes *axes)
{
    struct index_info rv;
    int *index = malloc(nelems * axes->len * sizeof(*index));
    int *flag = calloc(nelems * axes->len, sizeof(*flag));
    double *normed_qpts = malloc(nelems * axes->len * sizeof(*normed_qpts));

    double rtol = 1e-3;  /* relative tolerance for vertex matching */

    int debug = 0;

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
            index[k] = find_first_geq_than(axes->axis[i]->val, 0, axes->axis[i]->len - 1, qpts[k], rtol, &flag[k]);
            if (flag[k] == NDP_ON_VERTEX)
                normed_qpts[k] = 0.0;  /* we need this for extrapolation-by-nearest */
            else
                normed_qpts[k] = (qpts[k] - axes->axis[i]->val[max(index[k] - 1, 0)])/(axes->axis[i]->val[max(index[k], 1)] - axes->axis[i]->val[max(index[k] - 1, 0)]);
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
                printf("%d ", index[i*axes->len + j]);
            }
            printf("\b]");

            printf("  flags = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%d ", flag[i*axes->len + j]);
            }
            printf("\b]");

            printf("  normed_query_pt = [");
            for (int j = 0; j < axes->len; j++) {
                printf("%2.2f ", normed_qpts[i*axes->len + j]);
            }
            printf("\b]\n");
        }
    }

    rv.index = index;
    rv.flag = flag;
    rv.normed_qpts = normed_qpts;

    return rv;
}

/**
 * <!-- find_hypercubes() -->
 * @brief Determines n-dimensional hypercubes that contain (or are adjacent
 * to) the query points identified by indices.
 *
 * @param nelems number of query points
 * @param index a @p nelems -by- @p naxes array of indices
 * @param flag a @p nelems -by- @p naxes array of flags
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

struct hypercube_info find_hypercubes(int nelems, int *index, int *flag, ndp_table *table)
{
    int i, j, k, l, tidx, *iptr;
    int dim_reduction, hc_size;
    struct hypercube_info rv;

    ndp_axes *axes = table->axes;

    ndp_hypercube **hypercubes = malloc(nelems*sizeof(*hypercubes));

    // printf("find_hypercubes:\n");

    for (i = 0; i < nelems; i++) {
        iptr = &index[i*axes->len];

        // printf("  flags = [");
        dim_reduction = 0;
        for (k = 0; k < axes->len; k++) {
            // printf("%d ", flag[i*naxes+k]);
            if (flag[i*axes->len+k] == NDP_ON_VERTEX)
                dim_reduction++;
        }
        // printf("\b]\n");
        hc_size = axes->len-dim_reduction;
        double *hc_vertices = malloc(table->vdim * (1 << hc_size) * sizeof(*hc_vertices));
        // printf("   hypercube size: %d\n", hc_size);

        for (j = 0; j < (1 << hc_size); j++) {
            int cidx[axes->len];  /* current index naxes-plet */
            l = 0;
            for (k = 0; k < axes->len; k++) {
                if (flag[i*axes->len+k] == NDP_ON_VERTEX) {
                    cidx[k] = iptr[k];
                    continue;
                }
                // cidx[k] = iptr[k]-1+(j / (1 << (hc_size-l-1))) % 2;
                cidx[k] = max(iptr[k]-1+(j / (1 << (hc_size-l-1))) % 2, (j / (1 << (hc_size-l-1))) % 2);
                l++;
            }
            // printf("    cidx = [");
            // for (k = 0; k < axes->len; k++)
                // printf("%d ", cidx[k]);
            // printf("\b], ");

            tidx = idx2pos(axes, table->vdim, cidx);
            // printf("tidx = %d, table[tidx] = %f\n", tidx, table->grid[tidx]);

            memcpy(hc_vertices + j*table->vdim, table->grid + tidx, table->vdim*sizeof(*hc_vertices));
        }

        ndp_hypercube *hypercube = ndp_hypercube_new_from_data(hc_size, table->vdim, hc_vertices);
        // ndp_hypercube_print(hypercube, "    ");
        hypercubes[i] = hypercube;
    }

    rv.hypercubes = hypercubes;
    return rv;
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
 * grid vertex by using #find_nearest_defined_index() and set the value to the
 * found nearest value if @p extrapolation_method = #NDP_METHOD_NEAREST, and
 * lookup the nearest fully defined hypercube for extrapolation if
 * @p extrapolation_method = #NDP_METHOD_LINEAR.
 *
 * Finally, the ndpolator will loop through all hypercubes and call
 * #c_ndpolate() to get the interpolated or extrapolated function values for
 * each query point. The results are stored in the #ndp_query structure.
 *
 * @return a #ndp_query structure that holds all information on the specific
 * ndpolator run.
 */

ndp_query *ndpolate(int nelems, double *query_pts, ndp_table *table, ndp_extrapolation_method extrapolation_method)
{
    int k;
    ndp_query *query = ndp_query_new();
    double selected[table->axes->len];
    struct index_info ii;
    struct hypercube_info hi;
    ndp_hypercube *hypercube;

    int debug = 0;

    query->nelems = nelems;
    query->elems = query_pts;
    query->out_of_bounds = calloc(nelems, sizeof(*(query->out_of_bounds)));

    ii = find_indices(nelems, query_pts, table->axes);
    query->normed_elems = ii.normed_qpts;
    query->indices = ii.index;
    query->flags = ii.flag;

    /* flag any out-of-bounds query points: */
    for (int i = 0; i < nelems; i++) {
        for (int j = 0; j < table->axes->len; j++) {
            if (query->flags[i*table->axes->len+j] == NDP_OUT_OF_BOUNDS) {
                query->out_of_bounds[i] = 1;
                if (debug)
                    printf("  query point %d is out of bounds.\n", i);
                break;
            }
        }
    }

    hi = find_hypercubes(nelems, query->indices, query->flags, table);
    query->hypercubes = hi.hypercubes;

    if (debug) {
        for (int i = 0; i < query->nelems; i++) {
            ndp_hypercube *hypercube = query->hypercubes[i];
            printf("  hypercube %d: dim=%d vdim=%d v=[", i, hypercube->dim, hypercube->vdim);
            for (int j = 0; j < 1 << hypercube->dim; j++) {
                printf("{");
                for (int k = 0; k < hypercube->vdim; k++)
                    printf("%2.2f, ", hypercube->v[j*hypercube->vdim+k]);
                printf("\b\b} ");
            }
            printf("\b]\n");
        }
    }

    query->interps = malloc(nelems * table->vdim * sizeof(*(query->interps)));
    for (int i = 0; i < nelems; i++) {
        /* handle out-of-bounds elements first: */
        if (query->out_of_bounds[i]) {
            switch (extrapolation_method) {
                case NDP_METHOD_NONE:
                    for (int j = 0; j < table->vdim; j++)
                        query->interps[i*table->vdim+j] = NAN;
                    continue;
                break;
                case NDP_METHOD_NEAREST: {
                    double *normed_elem = query->normed_elems + i * table->axes->len;
                    int *elem_index = query->indices + i * table->axes->len;
                    int nearest = find_nearest_defined_index(normed_elem, elem_index, table);

                    if (debug) {
                        printf("  i=%d dim=%d vdim=%d nqpts=[", i, query->hypercubes[i]->dim, query->hypercubes[i]->vdim);
                        for (int j = 0; j < table->axes->len; j++)
                            printf("%2.2f ", query->normed_elems[i*table->axes->len + j]);
                        printf("\b]\n");

                        printf("nearest: %d, value: %f\n", nearest, table->grid[table->vdim*nearest]);
                    }

                    memcpy(query->interps + i*table->vdim, table->grid + nearest*table->vdim, table->vdim*sizeof(*(query->interps)));
                    continue;
                }
                break;
                case NDP_METHOD_LINEAR:
                    if (debug)
                        printf("all ready if the hypercube is fully defined... otherwise lookup\n");
                break;
                default:
                    /* invalid extrapolation method */
                    return NULL;
                break;
            }
        }

        /* continue with regular interpolation: */
        hypercube = query->hypercubes[i];

        k = 0;
        for (int j = 0; j < table->axes->len; j++) {
            /* skip when queried coordinate coincides with a vertex: */
            if (query->flags[i * table->axes->len + j] == NDP_ON_VERTEX)
                continue;
            selected[k] = query->normed_elems[i * table->axes->len + j];
            k++;
        }

        if (debug) {
            printf("  i=%d dim=%d vdim=%d nqpts=[", i, hypercube->dim, hypercube->vdim);
            for (int j = 0; j < table->axes->len; j++)
                printf("%2.2f ", query->normed_elems[i*table->axes->len + j]);
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

    struct index_info rv;

    int i, naxes, nelems, nbasic = 0;

    double *qpts;

    double *normed_qpts;

    ndp_axis **axis;
    ndp_axes *axes;

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
    rv = find_indices(nelems, qpts, axes);

    /* clean up: */
    for (i = 0; i < naxes; i++)
        free(axes->axis[i]);
    ndp_axes_free(axes);

    py_indices = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, rv.index);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_indices, NPY_ARRAY_OWNDATA);

    py_flags = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_INT, rv.flag);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_flags, NPY_ARRAY_OWNDATA);

    py_normed_query_pts = PyArray_SimpleNewFromData(2, query_pts_shape, NPY_DOUBLE, rv.normed_qpts);
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

    int *indices, *flags;
    int nelems, naxes;
    int nbasic = 0;

    struct hypercube_info rv;
    PyObject *py_hypercubes;

    printf("hypercubes:\n");

    if (!PyArg_ParseTuple(args, "OOOO|i", &py_indices, &py_axes, &py_flags, &py_grid, &nbasic))
        return NULL;

    nelems = PyArray_DIM(py_indices, 0);
    naxes = PyArray_DIM(py_indices, 1);
    if (nbasic == 0) nbasic = naxes;

    indices = (int *) PyArray_DATA(py_indices);
    flags = (int *) PyArray_DATA(py_flags);

    py_hypercubes = PyTuple_New(nelems);

    printf("  nelems=%d\n", nelems);
    printf("  naxes=%d\n", naxes);

    table = ndp_table_new_from_python(py_axes, nbasic, py_grid);

    rv = find_hypercubes(nelems, indices, flags, table);

    for (int i = 0; i < nelems; i++) {
        npy_intp shape[rv.hypercubes[i]->dim+1];
        PyObject *py_hypercube;
        int j;

        for (j = 0; j < rv.hypercubes[i]->dim; j++)
            shape[j] = 2;
        shape[j] = rv.hypercubes[i]->vdim;

        py_hypercube = PyArray_SimpleNewFromData(rv.hypercubes[i]->dim+1, shape, NPY_DOUBLE, rv.hypercubes[i]->v);
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

static PyObject *py_ndpolate(PyObject *self, PyObject *args)
{
    PyArrayObject *py_query_pts, *py_grid;
    PyObject *py_axes;
    int nbasic = 0;

    ndp_extrapolation_method extrapolation_method = NDP_METHOD_NONE;  /* default value */

    if (!PyArg_ParseTuple(args, "OOO|ii", &py_query_pts, &py_axes, &py_grid, &nbasic, &extrapolation_method))
        return NULL;

    ndp_table *table = ndp_table_new_from_python(py_axes, nbasic, py_grid);

    int nelems = PyArray_DIM(py_query_pts, 0);
    double *query_pts = PyArray_DATA(py_query_pts);

    ndp_query *query = ndpolate(nelems, query_pts, table, extrapolation_method);

    npy_intp adim[] = {nelems, table->vdim};
    PyObject *py_interps = PyArray_SimpleNewFromData(2, adim, NPY_DOUBLE, query->interps);
    PyArray_ENABLEFLAGS((PyArrayObject *) py_interps, NPY_ARRAY_OWNDATA);

    return py_interps;
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
    {"ndpolate", py_ndpolate, METH_VARARGS, "C version of N-dimensional interpolation"},
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
