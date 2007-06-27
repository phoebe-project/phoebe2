#ifndef PHOEBE_FITTING_H
	#define PHOEBE_FITTING_H

#include <stdarg.h>

#ifdef HAVE_LIBGSL
	#ifndef PHOEBE_GSL_DISABLED
		#include <gsl/gsl_rng.h>
		extern gsl_rng *PHOEBE_randomizer;
	#endif
#endif

#include "phoebe_types.h"

/* These are typedeffed structs used for passing arguments to minimizers:     */

typedef struct NMS_passed_parameters {
	int no_tba;             /* The number of parameters marked for adjustment */
	int dim_tba;       /* The dimension of the subspace marked for adjustment */
	int lcno;                          /* The number of observed light curves */
	int rvno;                             /* The number of observed RV curves */
	bool rv1;
	bool rv2;
	bool color_constraint;
	int CALCHLA;
	int CALCVGA;
	bool ASINI;
	int CC;
	PHOEBE_curve **obs;             /* An array of all transformed LC/RV data */
	double *weight;
	double *average;
	double *cindex;
	WD_LCI_parameters **pars;        /* Model parameters for all LC/RV curves */
	double ***pointers;           /* Translation table between GSL and PHOEBE */
	PHOEBE_vector **chi2s;
} NMS_passed_parameters;

#define wd_dc(atmtab,pltab,corrs,errors,chi2s,cfval) dc_(atmtab,pltab,corrs,errors,chi2s,cfval,strlen(atmtab),strlen(pltab))

int phoebe_minimize_using_dc  (FILE *dc_output, PHOEBE_minimizer_feedback *feedback);
int phoebe_minimize_using_nms (double accuracy, int iter_no, FILE *nms_output, PHOEBE_minimizer_feedback *feedback);

int kick_parameters (double sigma);

#endif
