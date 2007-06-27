#ifndef PHOEBE_NMS_H
#define PHOEBE_NMS_H 1

#include "phoebe_types.h"

typedef struct PHOEBE_nms_simplex {
	PHOEBE_matrix *corners;               /* simplex corner points            */
	PHOEBE_vector *values;                /* function values at corner points */
	PHOEBE_vector *ws1;                   /* workspace 1                      */
	PHOEBE_vector *ws2;                   /* workspace 2                      */
} PHOEBE_nms_simplex;

PHOEBE_nms_simplex *phoebe_nms_simplex_new      ();
int                 phoebe_nms_simplex_alloc    (PHOEBE_nms_simplex *simplex, int n);
int                 phoebe_nms_simplex_free     (PHOEBE_nms_simplex *simplex);

int                 phoebe_nms_set              (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *), PHOEBE_vector *x, double *size, PHOEBE_vector *step_size);
int                 phoebe_nms_iterate          (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *), PHOEBE_vector *x, double *size, double *fval);
double              phoebe_nms_size             (PHOEBE_nms_simplex *simplex);
double              phoebe_nms_move_corner      (double coeff, PHOEBE_nms_simplex *simplex, int corner, PHOEBE_vector *xc, double (*f) (PHOEBE_vector *));
int                 phoebe_nms_contract_by_best (PHOEBE_nms_simplex *simplex, int best, PHOEBE_vector *xc, double (*f) (PHOEBE_vector *));
int                 phoebe_nms_calc_center      (const PHOEBE_nms_simplex *simplex, PHOEBE_vector *mp);

#endif
