#ifndef NDPOLATOR_H
    #define NDPOLATOR_H 1

typedef enum ndpolator_error {
    NDPOLATOR_SUCCESS = 0,
    NDPOLATOR_INVALID_TYPE
} ndpolator_error;

typedef struct array {
    int len;
    double *data;
} array;

typedef enum ndp_extrapolation_method {
    NDP_METHOD_NONE = 0,
    NDP_METHOD_NEAREST,
    NDP_METHOD_LINEAR
} ndp_extrapolation_method;

enum {
    NDP_ON_GRID = 0,
    NDP_OUT_OF_BOUNDS = 1
};

#endif
