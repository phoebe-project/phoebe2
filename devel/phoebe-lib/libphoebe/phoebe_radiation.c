#include <math.h>

#include "phoebe_build_config.h"
#include "phoebe_error_handling.h"
#include "phoebe_types.h"

#if defined (HAVE_LIBGSL) && !defined (PHOEBE_GSL_DISABLED)
#include <gsl/gsl_spline.h>
#endif

int phoebe_compute_passband_intensity (double *intensity, PHOEBE_hist *SED, PHOEBE_hist *PTF, int mode)
{
	/**
	 * phoebe_compute_passband_intensity:
	 * @intensity: placeholder for the computed value
	 * @SED: spectral energy distribution to be integrated
	 * @PTF: passband transmission function to be integrated
	 * @mode: 1 for energy-weighted and 2 for photon flux-weighted integration
	 *
	 * This function computes passband intensity of the passed spectral
	 * energy distribution function and the passed passband transmission
	 * function. Passband intensity is defined as:
	 *
	 * I = \pi \int SED (lambda) PTF (lambda) dlambda / \int PTF (lambda) dlambda
	 *
	 * Returns: #PHOEBE_error_code
	 */

#if defined (HAVE_LIBGSL) && !defined (PHOEBE_GSL_DISABLED)
	int status, i;

	double intPTF;
	double *PTFindeps;
	double passband_flux;

	gsl_interp_accel *acc;
	gsl_spline       *spline;

	/* Error handling: */
	if (!SED) return ERROR_HIST_NOT_INITIALIZED;
	if (!PTF) return ERROR_HIST_NOT_INITIALIZED;

	/* If we are weighting the integrals with photon flux, we need to multiply
	 * the response function with \lambda:
	 */
	if (mode == 2)
		for (i = 0; i < PTF->bins; i++)
			PTF->val[i] *= 0.5*(PTF->range[i]+PTF->range[i+1]);
	
	/* Compute the integral over the passband transmission function: */
	status = phoebe_hist_integrate (&intPTF, PTF, PTF->range[0], PTF->range[PTF->bins]);
	if (status != SUCCESS) return status;

	/* Assemble an array of center-bin values: */
	PTFindeps = phoebe_malloc (PTF->bins * sizeof (*PTFindeps));
	for (i = 0; i < PTF->bins; i++)
		PTFindeps[i] = (PTF->range[i]+PTF->range[i+1])/2;

	/* Interpolate a cubic spline through the passband transmission function: */
	acc    = gsl_interp_accel_alloc ();
	spline = gsl_spline_alloc (gsl_interp_cspline, PTF->bins);
	gsl_spline_init (spline, PTFindeps, PTF->val, PTF->bins);

	/* Integrate spectral energy distribution function times this spline: */
	passband_flux = 0;
	i = 0;
	while (SED->range[i] < PTF->range[0]) i++;
	while (SED->range[i] < PTF->range[PTF->bins]) {
		passband_flux += SED->val[i] * gsl_spline_eval (spline, (SED->range[i]+SED->range[i+1])/2, acc) * (SED->range[i+1]-SED->range[i]);
		i++;
	}

	/* Compute the intensity: */
	*intensity = M_PI * passband_flux / intPTF;

	/* Free the array of center-bin values: */
	free (PTFindeps);

	/* Free the spline: */
	gsl_spline_free (spline);
	gsl_interp_accel_free (acc);

	return SUCCESS;
#endif
	phoebe_lib_warning ("GSL support needed to compute passband intensity, aborting.\n");
	return SUCCESS;
}

