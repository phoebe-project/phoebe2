#ifndef NDPOLATOR_H
    #define NDPOLATOR_H 1

#include "ndp_types.h"


/**
 * @enum ndp_extrapolation_method
 * Determines how the ndpolator should treat an out-of-bounds query points.
 */

typedef enum {
    NDP_METHOD_NONE = 0,     /*!< do not extrapolate; use NAN instead */
    NDP_METHOD_NEAREST,      /*!< extrapolate by finding the nearest defined vertex and use its face value */
    NDP_METHOD_LINEAR        /*!< extrapolate linearly by finding the nearest fully defined hypercube */
} ndp_extrapolation_method;

/**
 * @enum ndp_vertex_flag Flags each component of the query point whether it is
 * within the axis span, on one of the axis vertices, or if it is
 * out-of-bounds.
 */

enum ndp_vertex_flag {
    NDP_ON_GRID = 0,         /*!< default flag: the query point component is on-grid and can be interpolated */
    NDP_ON_VERTEX,           /*!< the query point component coincides with a vertex (within a specified tolerance) and can reduce hypercube dimensionality */
    NDP_OUT_OF_BOUNDS        /*!< the query point component is off-grid and will need to be extrapolated (see #ndp_extrapolation_method for possible extrapolation methods) */
};

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

struct index_info {
    int *index;              /*!< an ndarray of indices pointing to the superior corner of the hypercube that contains the query point */
    int *flag;               /*!< an ndarray of flags that correspond to each of the query point components */
    double *normed_qpts;     /*!< an ndarray of unit-normalized coordinates w.r.t. the parent hypercube */
};

/**
 * <!-- struct hypercube_info -->
 * @brief Stores all fields related to the hypercubes.
 *
 * @details
 * Function #find_hypercubes() computes an array of #ndp_hypercube @p
 * hypercubes that correspond to each query point, and sets the out-of-bounds
 * flag for any query points that are off grid.
 */

struct hypercube_info {
    int *out_of_bounds;
    ndp_hypercube **hypercubes;
};

/**
 * <!-- struct ndpolate_info -->
 * @brief Stores all fields related to the interpolation.
 * 
 * @details
 * Function #ndpolate() computes a single deliverable that is stored
 * in this structure: an array of interpolants, @p interps. In order to
 * allow for future expansion, the @p interps field is packed inside the
 * #ndpolate_info struct.
 */

struct ndpolate_info {
    double *interps;  /*!< an array of interpolated values */
};

struct index_info find_indices(int nelems, double *qpts, ndp_axes *axes);
struct hypercube_info find_hypercubes(int nelems, int *index, int *flag, ndp_table *table);
ndp_query *ndpolate(int nelems, double *query_pts, ndp_table *table, ndp_extrapolation_method extrapolation_method);

#endif
