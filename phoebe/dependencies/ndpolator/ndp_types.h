/**
 * @file ndp_types.h
 * @brief Ndpolator's type definitions and constructor/desctructor prototypes.
 */

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
 * <!-- typedef struct ndp_query_pts -->
 * @brief Ndpolator's structure for query points.
 *
 * @details
 * Query points (points of interest) are given by n coordinates that
 * correspond to n #ndp_axis instances stored in #ndp_axes. Their number is
 * given by the @p nelems field and their dimension by the @p naxes field. The
 * @p indices array provides superior corners of the hypercube that contains a
 * query point; the @p flags array tags each query point component with one of
 * the #ndp_vertex_flag flags: #NDP_ON_GRID, #NDP_ON_VERTEX, or
 * #NDP_OUT_OF_BOUNDS. The actual query points (as passed to ndpolator, in
 * axis units) are stored in the @p requested array, and the unit-hypercube
 * normalized units are stored in the @p normed array.
 */

typedef struct ndp_query_pts {
    int nelems;         /*!< number of query points */
    int naxes;          /*!< query point dimension (number of axes) */
    int *indices;       /*!< an array of superior hypercube indices */
    int *flags;         /*!< an array of flags, one per query point component */
    double *requested;  /*!< an array of absolute query points (in axis units) */
    double *normed;     /*!< an array of unit-hypercube normalized query points */
} ndp_query_pts;

ndp_query_pts *ndp_query_pts_new();
ndp_query_pts *ndp_query_pts_new_from_data(int nelems, int naxes, int *indices, int *flags, double *requested, double *normed);
int ndp_query_pts_alloc(ndp_query_pts *qpts, int nelems, int naxes);
int ndp_query_pts_free(ndp_query_pts *qpts);

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
    int vdim;        /*!< function value length (1 for scalars, >1 for arrays) */
    ndp_axes *axes;  /*!< an #ndp_axes instance that defines all axes */
    double *grid;    /*!< an array that holds all function values, in C-native order */
    int nverts;      /*!< @private number of basic grid points */
    int *vmask;      /*!< @private nverts-length mask of nodes (defined grid points) */
    int *hcmask;     /*!< @private nverts-length mask of fully defined hypercubes */
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
    int dim;      /*!< dimension of the hypercube */
    int vdim;     /*!< function value length */
    int fdhc;     /*!< flag that indicates whether the hypercube is fully defined */
    double *v;    /*!< hypercube vertex function values, in C order (last axis runs fastest)*/
} ndp_hypercube;

ndp_hypercube *ndp_hypercube_new();
ndp_hypercube *ndp_hypercube_new_from_data(int dim, int vdim, int fdhc, double *v);
int ndp_hypercube_alloc(ndp_hypercube *hc, int dim, int vdim);
int ndp_hypercube_print(ndp_hypercube *hc, const char *prefix);
int ndp_hypercube_free(ndp_hypercube *hc);

/* defined in ndpolator.c: */
extern int idx2pos(ndp_axes *axes, int vdim, int *index, int *pos);
extern int pos2idx(ndp_axes *axes, int vdim, int pos, int *idx);

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
    int extrapolation_method;    /*!< a #ndp_extrapolation_method */
    ndp_hypercube **hypercubes;  /*!< an array of hypercubes, one per query point */
    double *interps;             /*!< an array of interpolants -- results of interpolation/extrapolation */
} ndp_query;

ndp_query *ndp_query_new();
int ndp_query_free(ndp_query *query);






/**
 * <!-- struct index_info -->
 * @brief Stores all fields related to query point indexing.
 *
 * @details
 * Function #find_indices() computes three main deliverables that are stored
 * in this structure: (1) an array of indices, @p index, that correspond to
 * the superior hypercube corner that contains or is adjacent to the query
 * point; (2) an array of #ndp_vertex_flag flags, @p flag, for each component
 * of the query point; and (3) an array of unit hypercube-normalized query
 * points. Structure arrays are allocated by #find_indices() and need to be
 * freed once they are no longer required, typically by #ndp_query_free().
 */

/**
 * <!-- struct hypercube_info -->
 * @brief Stores all fields related to the hypercubes.
 *
 * @details
 * Function #find_hypercubes() computes an array of #ndp_hypercube @p
 * hypercubes that correspond to each query point, and sets the out-of-bounds
 * flag for any query points that are off grid.
 */

#endif
