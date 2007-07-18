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

#define wd_dc(atmtab,pltab,corrs,errors,chi2s,cfval) dc_(atmtab,pltab,corrs,errors,chi2s,cfval,strlen(atmtab),strlen(pltab))

int phoebe_minimize_using_dc  (FILE *dc_output, PHOEBE_minimizer_feedback *feedback);
int phoebe_minimize_using_nms (double accuracy, int iter_no, FILE *nms_output, PHOEBE_minimizer_feedback *feedback);

int kick_parameters (double sigma);

#endif
