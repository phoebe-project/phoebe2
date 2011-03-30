#ifndef PHOEBE_FITTING_H
	#define PHOEBE_FITTING_H

#include <stdarg.h>

#include "phoebe_nms.h"
#include "phoebe_types.h"

#define wd_dc(atmtab,pltab,l3perc,knobs,indeps,fluxes,weights,nph,delph,corrs,errors,chi2s,cormat,params,cfval) dc_(atmtab,pltab,l3perc,knobs,indeps,fluxes,weights,nph,delph,corrs,errors,chi2s,cormat,params,cfval,strlen(atmtab),strlen(pltab))

double phoebe_chi2_cost_function (PHOEBE_vector *adjpars, PHOEBE_nms_parameters *params);

int phoebe_minimize_using_dc  (FILE *dc_output, PHOEBE_minimizer_feedback *feedback);
int phoebe_minimize_using_nms (FILE *nms_output, PHOEBE_minimizer_feedback *feedback);

#endif
