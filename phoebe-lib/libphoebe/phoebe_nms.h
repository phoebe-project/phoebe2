#ifndef PHOEBE_NMS_H
#define PHOEBE_NMS_H 1

#include "phoebe_parameters.h"
#include "phoebe_types.h"

typedef struct PHOEBE_nms_simplex {
	PHOEBE_matrix *corners;                          /* simplex corner points */
	PHOEBE_vector *values;                /* function values at corner points */
	PHOEBE_vector *ws1;                                        /* workspace 1 */
	PHOEBE_vector *ws2;                                        /* workspace 2 */
	PHOEBE_vector *center;                                  /* simplex center */
	PHOEBE_vector *delta;                                     /* current step */
	PHOEBE_vector *xmc;                              /* x-center on workspace */
	double         S2;                                                      /**/
} PHOEBE_nms_simplex;

/* These are typedeffed structs used for passing arguments to minimizers: */

typedef struct PHOEBE_nms_parameters {
	PHOEBE_array *qualifiers;           /* A list of unconstrained qualifiers */
	PHOEBE_vector *l_bounds;              /* Lower parameter value boundaries */
	PHOEBE_vector *u_bounds;              /* Upper parameter value boundaries */
	int lcno;                          /* The number of observed light curves */
	int rvno;                             /* The number of observed RV curves */
	PHOEBE_curve **obs;             /* An array of all transformed LC/RV data */
	PHOEBE_vector *psigma;        /* A vector of passband standard deviations */
	PHOEBE_array *lexp;       /* An array of level-dependent weight exponents */
	PHOEBE_vector *chi2s;      /* A vector of individual passband chi2 values */
	bool autolevels;              /* Should levels be computed automatically? */
	bool autogamma;       /* Should gamma velocity be computed automatically? */
	PHOEBE_vector *l3;                 /* Values of third light, per passband */
	PHOEBE_el3_units l3units;                         /* Units of third light */
	PHOEBE_vector *levels;
	                  /* If levels are computed, their values are stored here */
	bool rv1;
	bool rv2;
	double *average;
} PHOEBE_nms_parameters;

PHOEBE_nms_simplex *phoebe_nms_simplex_new      ();
int                 phoebe_nms_simplex_alloc    (PHOEBE_nms_simplex *simplex, int n);
int                 phoebe_nms_simplex_free     (PHOEBE_nms_simplex *simplex);

int phoebe_nms_simplex_set     (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_nms_parameters *params, PHOEBE_vector *x, double *size, const PHOEBE_vector *step_size);
int phoebe_nms_simplex_iterate (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_nms_parameters *params, PHOEBE_vector *x, double *size, double *fval);

#endif
