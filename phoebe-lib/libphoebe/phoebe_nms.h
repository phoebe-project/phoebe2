#ifndef PHOEBE_NMS_H
#define PHOEBE_NMS_H 1

#include "phoebe_parameters.h"
#include "phoebe_types.h"

typedef struct PHOEBE_nms_simplex {
	PHOEBE_matrix *corners;               /* simplex corner points            */
	PHOEBE_vector *values;                /* function values at corner points */
	PHOEBE_vector *ws1;                   /* workspace 1                      */
	PHOEBE_vector *ws2;                   /* workspace 2                      */
} PHOEBE_nms_simplex;

/* These are typedeffed structs used for passing arguments to minimizers: */

typedef struct PHOEBE_nms_parameters {
	PHOEBE_parameter_list *tba;
	int dim_tba;       /* The dimension of the subspace marked for adjustment */
	int lcno;                          /* The number of observed light curves */
	int rvno;                             /* The number of observed RV curves */
	bool rv1;
	bool rv2;
	int CALCHLA;
	int CALCVGA;
	PHOEBE_curve **obs;             /* An array of all transformed LC/RV data */
	double *average;
	PHOEBE_vector *chi2s;      /* A vector of individual passband chi2 values */
	PHOEBE_vector *weights;        /* A vector of individual passband weights */
} PHOEBE_nms_parameters;

PHOEBE_nms_simplex *phoebe_nms_simplex_new      ();
int                 phoebe_nms_simplex_alloc    (PHOEBE_nms_simplex *simplex, int n);
int                 phoebe_nms_simplex_free     (PHOEBE_nms_simplex *simplex);

int                 phoebe_nms_set              (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_vector *x, PHOEBE_nms_parameters *params, double *size, PHOEBE_vector *step_size);
int                 phoebe_nms_iterate          (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_vector *x, PHOEBE_nms_parameters *params, double *size, double *fval);
double              phoebe_nms_size             (PHOEBE_nms_simplex *simplex);
double              phoebe_nms_move_corner      (double coeff, PHOEBE_nms_simplex *simplex, int corner, PHOEBE_vector *xc, PHOEBE_nms_parameters *params, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *));
int                 phoebe_nms_contract_by_best (PHOEBE_nms_simplex *simplex, int best, PHOEBE_vector *xc, PHOEBE_nms_parameters *params, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *));
int                 phoebe_nms_calc_center      (const PHOEBE_nms_simplex *simplex, PHOEBE_vector *mp);

#endif
