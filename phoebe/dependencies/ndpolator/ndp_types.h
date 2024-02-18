#ifndef NDP_TYPES_H
    #define NDP_TYPES_H 1

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/**
 * @enum ndp_status All ndpolator functions that do not allocate memory should
 * return #ndp_status. If no exceptions occurred, #NDP_SUCCESS should be
 * returned. Otherwise, a suitable #ndp_status should be returned.
 */

typedef enum {
    NDP_SUCCESS = 0,  /*!< normal exit, the function did not run into any exceptions */
    NDP_INVALID_TYPE  /*!< the passed argument type is invalid, the function cannot continue */
} ndp_status;

/**
 * <!-- typedef struct ndp_axis -->
 * @brief Ndpolator's single axis structure.
 *
 * @details
 * An axis, in ndpolator language, is an array of length @p len and vertices
 * @p val. Axes span ndpolator dimensions: for `N`-dimensional interpolation
 * and/or extrapolation, there need to be `N` axes. Note that axes themselves
 * do not have any function values associated to them; they only span the
 * `N`-dimensional grid.
 */

typedef struct ndp_axis {
    int len;      /*!< axis length (number of vertices) */
    double *val;  /*!< axis vertices */
} ndp_axis;

ndp_axis *ndp_axis_new();
ndp_axis *ndp_axis_new_from_data(int len, double *val);
int ndp_axis_free(ndp_axis *axis);

/**
 * <!-- typedef struct ndp_axes -->
 * @brief Ndpolator's complete axes structure.
 *
 * @details
 * This structure stores all axes that span the ndpolator grid. Each axis must
 * be of the #ndp_axis type. Function values are associated to each
 * combination (cartesian product) of axis indices.
 *
 * There are two types of axes that ndpolator recognizes: _basic_ and
 * _attached_. Basic axes span the sparse grid: function values can either be
 * defined, or null. Attached axes, on the other hand, are _guaranteed_ to
 * have function values defined for all combinations of basic indices that
 * have function values defined. For example, if `(i, j, k)` are basic indices
 * that have a defined function value, then `(i, j, k, l, m)` are guaranteed
 * to be defined as well, where `l` and `m` index attached axes.
 */

typedef struct ndp_axes {
    int len;          /*!< number of axes */
    int nbasic;       /*!< number of basic axes; basic axes must be given first in the @p axis array */
    ndp_axis **axis;  /*!< an array of #ndp_axis-type axes */
    int *cplen;       /*!< @private cumulative product of axis lengths, for example `[76, 11, 8, 42]` -> `[11*8*42, 8*42, 42, 1]` */
} ndp_axes;

ndp_axes *ndp_axes_new();
ndp_axes *ndp_axes_new_from_data(int naxes, int nbasic, ndp_axis **axis);
ndp_axes *ndp_axes_new_from_python(PyObject *py_axes, int nbasic);
int ndp_axes_free(ndp_axes *axes);

/**
 * <!-- typedef struct ndp_table -->
 * @brief Ndpolator's complete table structure.
 *
 * @details
 * Ndpolator uses #ndp_table to store all relevant parameters for
 * interpolation and/or extrapolation. It stores the axes that span the
 * interpolation hyperspace (in a #ndp_axes structure), the function values
 * across the interpolation hyperspace (@p grid), function value length (@p
 * vdim), and several private fields that further optimize interpolation.
 */

typedef struct ndp_table {
    int vdim;
    ndp_axes *axes;
    double *grid;
    int ndefs;                /*!< @private number of defined vertices */
    int *defined_vertices;    /*!< @private positions of defined vertices in @grid */
    int hcdefs;               /*!< @private number of fully defined hypercubes */
    int *defined_hypercubes;  /*!< @private positions of fully defined hypercubes */
} ndp_table;

ndp_table *ndp_table_new();
ndp_table *ndp_table_new_from_data(ndp_axes *axes, int vdim, double *grid);
ndp_table *ndp_table_new_from_python(PyObject *py_axes, int nbasic, PyArrayObject *py_grid);
int ndp_table_free(ndp_table *table);

/**
 * <!-- typedef struct ndp_hypercube -->
 * @brief Ndpolator's hypercube structure.
 *
 * @details
 * Hypercubes are subgrids that enclose (or are adjacent to, in the case of
 * extrapolation) the passed query points, one per query point. They are
 * qualified by their dimension (an N-dimensional hypercube has 2<sup>N</sup>
 * vertices) and their function value length. Note that hypercube dimension
 * can be less than the dimension of the grid itself: if a query point
 * coincides with any of the axes, that will reduce the dimensionality of the
 * hypercube. If all query point components coincide with the axes (i.e, the
 * vertex itself is requested), then the hypercube dimension equals 0, so
 * there is no interpolation at all -- only that vertex's function value is
 * returned.
 */

typedef struct ndp_hypercube {
    int dim;    /*!< dimension of the hypercube */
    int vdim;   /*!< function value length */
    double *v;  /*!< hypercube vertex function values, in C order (last axis runs fastest)*/
} ndp_hypercube;

ndp_hypercube *ndp_hypercube_new();
ndp_hypercube *ndp_hypercube_new_from_data(int dim, int vdim, double *v);
int ndp_hypercube_alloc(ndp_hypercube *hc, int dim, int vdim);
int ndp_hypercube_print(ndp_hypercube *hc, const char *prefix);
int ndp_hypercube_free(ndp_hypercube *hc);

/* defined in ndpolator.c: */
extern int idx2pos(ndp_axes *axes, int vdim, int *index);
extern int *pos2idx(ndp_axes *axes, int vdim, int pos);

/**
 * <!-- typedef struct ndp_query -->
 * @brief Ndpolator's query structure.
 *
 * @details
 * Query is ndpolator's main work structure. It stores the query points
 * (called elements in the structure), the corresponding axis indices,
 * flags, and hypercubes. Once interpolation/extrapolation is done (by
 * calling #ndpolate()), interpolated values are also stored in it.
 */

typedef struct ndp_query {
    int nelems;                  /*!< number of query points */
    int extrapolation_method;    /*!< a #ndp_extrapolation_method */

    double *elems;               /*!< an array of (flattened) query points, stacked in the C-order (last axis varies the fastest) */
    double *normed_elems;        /*!< @private an array of unit-normalized query points */
    int *out_of_bounds;          /*!< @private an array of out-of-bounds flags, per query point */
    int *indices;                /*!< an array of (flattened) indices of the superior hypercube vertex */
    int *flags;                  /*!< an array of (flattened) flags per query point component (see #ndp_vertex_flag)*/

    ndp_hypercube **hypercubes;  /*!< an array of hypercubes, one per query point */

    double *interps;             /*!< an array of interpolants -- results of interpolation/extrapolation */
} ndp_query;

ndp_query *ndp_query_new();
int ndp_query_free(ndp_query *query);

#endif
