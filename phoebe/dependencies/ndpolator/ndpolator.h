/**
 * @file ndpolator.h
 * @brief Enumerators and function prototypes for ndpolator.
 */

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
 * @enum ndp_vertex_flag
 * Flags each component of the query point whether it is within the axis span,
 * on one of the axis vertices, or if it is out-of-bounds.
 */

enum ndp_vertex_flag {
    NDP_ON_GRID = 0,         /*!< default flag: the query point component is on-grid and can be interpolated */
    NDP_ON_VERTEX,           /*!< the query point component coincides with a vertex (within a specified tolerance) and can reduce hypercube dimensionality */
    NDP_OUT_OF_BOUNDS        /*!< the query point component is off-grid and will need to be extrapolated (see #ndp_extrapolation_method for possible extrapolation methods) */
};

int *find_nearest(double *normed_elem, int *elem_index, int *elem_flag, ndp_table *table, int *mask);
ndp_query_pts *find_indices(int nelems, double *qpts, ndp_axes *axes);
ndp_hypercube **find_hypercubes(ndp_query_pts *qpts, ndp_table *table);
ndp_query *ndpolate(ndp_query_pts *qpts, ndp_table *table, ndp_extrapolation_method extrapolation_method);

#endif
