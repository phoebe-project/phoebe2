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

#endif
