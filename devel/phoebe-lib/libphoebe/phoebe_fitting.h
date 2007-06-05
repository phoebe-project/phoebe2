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
	int to_be_adjusted;
	int lcno;
	int rvno;
	bool rv1;
	bool rv2;
	bool color_constraint;
	int CALCHLA;
	int CALCVGA;
	bool ASINI;
	int CC;
	int *indices;
	PHOEBE_curve **obs;
	double *weight;
	double *average;
	double *cindex;
	WD_LCI_parameters **pars;
	double ***pointers;
	PHOEBE_vector **chi2s;
} NMS_passed_parameters;

#define wd_dc(atmtab,pltab,corrs,errors,chi2s) dc_(atmtab,pltab,corrs,errors,chi2s,strlen(atmtab),strlen(pltab))

int find_minimum_with_nms (double accuracy, int iter_no, FILE *nms_output, PHOEBE_minimizer_feedback *feedback);
int find_minimum_with_dc  (FILE *dc_output, PHOEBE_minimizer_feedback *feedback);
int kick_parameters (double sigma);
int nms_print (const char *fmt, ...);

#endif
