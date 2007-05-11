#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_calculations.h"
#include "phoebe_error_handling.h"
#include "phoebe_spectra.h"
#include "phoebe_types.h"

#ifdef HAVE_LIBGSL
#ifndef PHOEBE_GSL_DISABLED

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_histogram.h>

#endif
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))
#define pdif(a)  ((a) > 0 ? (a) : 0)

int query_spectra_repository (char *rep_name, PHOEBE_specrep *spec)
{
	/*
	 * This function queries the passed directory location for spectra. The
	 * number of found spectra is reported through a spec.no parameter field.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 *   ERROR_SPECTRA_REPOSITORY_NOT_FOUND
	 *   ERROR_SPECTRA_REPOSITORY_INVALID_NAME
	 */

	DIR *repository;
	struct dirent *filelist;

	int RES, LAMIN, LAMAX, MET, TEMP, LOGG;
	char METSIGN;

	if (!rep_name)
		return ERROR_SPECTRA_REPOSITORY_INVALID_NAME;

	if (!filename_is_directory (rep_name))
		return ERROR_SPECTRA_REPOSITORY_NOT_DIRECTORY;

	repository = opendir (rep_name);
	if (!repository)
		return ERROR_SPECTRA_REPOSITORY_NOT_FOUND;

	/* Initialize the spectra database:                                       */
	spec->no = 0; spec->prop = NULL;

	while (filelist = readdir (repository)) {
		if (sscanf (filelist->d_name, "F%4d%dV000-R%d%c%dT%dG%dK2NOVER.ASC", &LAMIN, &LAMAX, &RES, &METSIGN, &MET, &TEMP, &LOGG) != 7) continue;
		if (METSIGN == 'M') MET = -MET;
		spec->no++;
		spec->prop = phoebe_realloc (spec->prop, spec->no * sizeof (*spec->prop));
		spec->prop[spec->no-1].resolution  = RES;
		spec->prop[spec->no-1].lambda_min  = LAMIN;
		spec->prop[spec->no-1].lambda_max  = LAMAX;
		spec->prop[spec->no-1].temperature = TEMP;
		spec->prop[spec->no-1].metallicity = MET;
		spec->prop[spec->no-1].gravity     = LOGG;
	}
	closedir (repository);

	if (spec->no == 0)
		return ERROR_SPECTRA_REPOSITORY_EMPTY;

	return SUCCESS;
}

PHOEBE_spectrum *phoebe_spectrum_new ()
{
	PHOEBE_spectrum *spectrum = phoebe_malloc (sizeof (*spectrum));
	spectrum->R   = 0;
	spectrum->Rs  = 0;
	spectrum->disp = PHOEBE_SPECTRUM_DISPERSION_NONE;
	spectrum->data = phoebe_hist_new ();
	return spectrum;
}

PHOEBE_spectrum *phoebe_spectrum_new_from_file (char *filename)
{
	FILE *input;
	PHOEBE_spectrum *spectrum;
	PHOEBE_vector *bin_centers;
	int linecount = 1;

	input = fopen (filename, "r");
	if (!input) return NULL;

	spectrum = phoebe_spectrum_new ();
	bin_centers = phoebe_vector_new ();

	while (!feof (input)) {
		double wl, flux;
		char line[255];
		char *strptr;
		char *lineptr = line;

		fgets (line, 254, input);
		if (feof (input)) break;

		/* Remove the trailing newline:                                       */
		if ( strptr = strchr (line, '\n') ) *strptr = '\0';

		/* Remove comments (if any):                                          */
		if ( strptr = strchr (line, '#') ) *strptr = '\0';

		/* Remove any leading whitespaces and empty lines:                        */
		while ( (lineptr[0] == ' ' || lineptr[0] == '\t') && lineptr[0] != '\0') lineptr++;
		if (*lineptr == '\0') {
			linecount++;
			continue;
		}

		if (lineptr[0] != '\0')
			if (sscanf (lineptr, "%lf %lf", &wl, &flux) == 2) {
				phoebe_vector_realloc (bin_centers, bin_centers->dim + 1);
				phoebe_spectrum_realloc (spectrum, spectrum->data->bins + 1);
				bin_centers->val[bin_centers->dim-1] = wl;
				spectrum->data->val[spectrum->data->bins-1] = flux;
			}

		linecount++;
	}

	fclose (input);

	phoebe_hist_set_ranges (spectrum->data, bin_centers);
	phoebe_vector_free (bin_centers);

	/* Guess the dispersion function: */
	phoebe_spectrum_dispersion_guess (&(spectrum->disp), spectrum);

	/* Guess a sampling power: */
	if (spectrum->disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR)
		spectrum->Rs = 1.0/(spectrum->data->range[1]-spectrum->data->range[0]);
	else
		spectrum->Rs = 0.5*(spectrum->data->range[0]+spectrum->data->range[1])/(spectrum->data->range[1]-spectrum->data->range[0]);

	return spectrum;
}

PHOEBE_spectrum *phoebe_spectrum_create (double ll, double ul, double R, PHOEBE_spectrum_dispersion disp)
{
	/*
	 * This function creates an empty spectrum of resolving power R, sampled on
	 * the wavelength interval [ll, ul]. If the dispersion type is log, the
	 * resolving power is constant and the dispersion changes throughout the
	 * spectrum. If the dispersion type is linear, the dispersion is constant
	 * and the resolving power changes. Dispersion relates to resolving power
	 * simply as \delta = 1/R.
	 */

	int status;
	PHOEBE_spectrum *spectrum;
	int N, i;
	double q;

	switch (disp) {
		case PHOEBE_SPECTRUM_DISPERSION_LINEAR:
			q = 1.0/R;
			N = (int) ((ul-ll)/q+1e-6);
		break;
		case PHOEBE_SPECTRUM_DISPERSION_LOG:
			q = (1+1./2./R)/(1-1./2./R);
			N = 1 + (int) (log (ul/ll) / log (q));
		break;
		default:
			phoebe_lib_error ("unsupported spectrum dispersion type, aborting.\n");
			return NULL;
		break;
	}

	spectrum = phoebe_spectrum_new ();

	spectrum->R    = R;
	spectrum->Rs   = R;
	spectrum->disp = disp;

	status = phoebe_spectrum_alloc (spectrum, N);

	if (status != SUCCESS) {
		phoebe_lib_error ("phoebe_spectrum_create: %s", phoebe_error (status));
		return NULL;
	}

	spectrum->data->range[0] = ll; spectrum->data->val[0] = 0.0;

	switch (disp) {
		case PHOEBE_SPECTRUM_DISPERSION_LINEAR:
			for (i = 1; i < N; i++) {
				spectrum->data->range[i] = q + spectrum->data->range[i-1];
				spectrum->data->val[i] = 0.0;
			}
			spectrum->data->range[N] = q + spectrum->data->range[N-1];
		break;
		case PHOEBE_SPECTRUM_DISPERSION_LOG:
			for (i = 1; i < N; i++) {
				spectrum->data->range[i] = q * spectrum->data->range[i-1];
				spectrum->data->val[i] = 0.0;
			}
			spectrum->data->range[N] = q * spectrum->data->range[N-1];
		break;
	}

	return spectrum;
}

PHOEBE_spectrum *phoebe_spectrum_duplicate (PHOEBE_spectrum *spectrum)
{
	/*
	 * This function makes a duplicate copy of the spectrum 'spectrum'. It
	 * is somewhat tricky to avoid memory leaks in this function: the call to
	 * phoebe_spectrum_new () initializes an empty histogram as convenience
	 * for a range of other functions, but in this case it is undesireable,
	 * so it has to be undone. The reason is that phoebe_hist_duplicate ()
	 * function also initializes an empty histogram and allocates space for
	 * the copy. Thus there should be no call to phoebe_spectrum_alloc () in
	 * this function. The current implementation has been tested against
	 * memory leaks and there are none (according to valgrind).
	 */

	PHOEBE_spectrum *copy;

	if (!spectrum) return NULL;

	copy = phoebe_spectrum_new ();

	copy->R  = spectrum->R;
	copy->Rs = spectrum->Rs;
	copy->disp = spectrum->disp;

	phoebe_hist_free (copy->data);
	copy->data = phoebe_hist_duplicate (spectrum->data);

	return copy;
}

PHOEBE_vector *phoebe_spectrum_get_column (PHOEBE_spectrum *spectrum, int col)
{
	/*
	 * This function allocates a vector and copies the contents of the col-th
	 * column in the spectrum 'spectrum' to it. The values of 'col' may be 1
	 * (usually a wavelength) or 2 (usually a flux). If an error occurs, NULL
	 * is returned.
	 */

	int i;
	PHOEBE_vector *out;

	if (!spectrum)                 return NULL;
	if (spectrum->data->bins == 0) return NULL;
	if (col != 1 && col != 2)      return NULL;

	out = phoebe_vector_new ();

	if (col == 1)
		phoebe_vector_alloc (out, spectrum->data->bins+1);
	else
		phoebe_vector_alloc (out, spectrum->data->bins);

	for (i = 0; i < out->dim; i++) {
		if (col == 1)
			out->val[i] = spectrum->data->range[i];
		else
			out->val[i] = spectrum->data->val[i];
	}

	return out;
}

int phoebe_spectrum_alloc (PHOEBE_spectrum *spectrum, int dim)
{
	/*
	 * This function allocates storage memory for a spectrum of 'dim'.
	 *
	 * Return values:
	 *
	 *   ERROR_SPECTRUM_NOT_INITIALIZED
	 *   ERROR_SPECTRUM_ALREADY_ALLOCATED
	 *   ERROR_SPECTRUM_INVALID_DIMENSION
	 *   SUCCESS
	 */

	if (!spectrum)
		return ERROR_SPECTRUM_NOT_INITIALIZED;

	if (spectrum->data->bins != 0)
		return ERROR_SPECTRUM_ALREADY_ALLOCATED;

	if (dim < 1)
		return ERROR_SPECTRUM_INVALID_DIMENSION;

	phoebe_hist_alloc (spectrum->data, dim);

	return SUCCESS;
}

int phoebe_spectrum_realloc (PHOEBE_spectrum *spectrum, int dim)
{
	/*
	 * This function reallocates storage memory for a spectrum of 'dim'.
	 *
	 * Return values:
	 *
	 *   ERROR_SPECTRUM_NOT_INITIALIZED
	 *   ERROR_SPECTRUM_INVALID_DIMENSION
	 *   SUCCESS
	 */

	if (!spectrum)
		return ERROR_SPECTRUM_NOT_INITIALIZED;

	if (dim < 1)
		return ERROR_SPECTRUM_INVALID_DIMENSION;

	phoebe_hist_realloc (spectrum->data, dim);
	return SUCCESS;
}

int phoebe_spectrum_free (PHOEBE_spectrum *spectrum)
{
	/*
	 * This function frees the allocated space for a spectrum.
	 */

	if (!spectrum) return SUCCESS;
	if (spectrum->data)
		phoebe_hist_free (spectrum->data);

	free (spectrum);

	return SUCCESS;
}

char *phoebe_spectrum_dispersion_type_get_name (PHOEBE_spectrum_dispersion disp)
{
	/*
	 * This function translates the enumerated value to a string.
	 */

	switch (disp) {
		case PHOEBE_SPECTRUM_DISPERSION_LINEAR:
			return "linear";
		case PHOEBE_SPECTRUM_DISPERSION_LOG:
			return "logarithmic";
		case PHOEBE_SPECTRUM_DISPERSION_NONE:
			return "none";
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_spectrum_dispersion_type_get_name (), please report this!\n");
			return "invalid";
	}
}

int phoebe_spectrum_dispersion_guess (PHOEBE_spectrum_dispersion *disp, PHOEBE_spectrum *s)
{
	/*
	 * This function guesses the type of the spectrum dispersion.
	 */

	int i;

	*disp = PHOEBE_SPECTRUM_DISPERSION_LINEAR;
	for (i = 0; i < s->data->bins-1; i++) {
		if (fabs (2*s->data->range[i+1]-s->data->range[i]-s->data->range[i+2]) > 1e-6) {
			*disp = PHOEBE_SPECTRUM_DISPERSION_NONE;
			break;
		}
	}
	if (*disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR)
		return SUCCESS;

	*disp = PHOEBE_SPECTRUM_DISPERSION_LOG;
	for (i = 0; i < s->data->bins-1; i++)
		if (fabs (s->data->range[i+1]/s->data->range[i] - s->data->range[i+2]/s->data->range[i+1]) > 1e-6) {
			*disp = PHOEBE_SPECTRUM_DISPERSION_NONE;
			break;
		}

	return SUCCESS;
}

int phoebe_spectrum_crop (PHOEBE_spectrum *spectrum, double ll, double ul)
{
	/*
	 * This function crops the passed spectrum to the [ll, ul] interval.
	 * All error handling is done by the called function, so we don't have
	 * to worry about anything here.
	 */

	return phoebe_hist_crop (spectrum->data, ll, ul);
}

int phoebe_spectrum_new_from_repository (PHOEBE_spectrum **spectrum, double R, int T, int g, int M, double ll, double ul)
{
	/*
	 * This function queries the spectra repository, takes closest gridded spe-
	 * ctra from it and linearly interpolates to the spectrum represented by
	 * the passed parameters.
	 *
	 * Input parameters:
	 *
	 *   R    ..  resolution (typically 50000)
	 *   T    ..  effective temperature
	 *   g    ..  log g/g0
	 *   M    ..  metallicity [M/H] in Solar units
	 *   ll   ..  lower queried wavelength interval limit
	 *   ul   ..  upper queried wavelength interval limit
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 *   ERROR_SPECTRA_REPOSITORY_EMPTY
	 *   ERROR_SPECTRA_REPOSITORY_INVALID_NAME
	 *   ERROR_SPECTRA_REPOSITORY_NOT_DIRECTORY
	 *   ERROR_SPECTRA_REPOSITORY_NOT_FOUND
	 */

	int i, j;
	int Tlow, Thigh, Mlow, Mhigh, glow, ghigh;
	char Mlostr[5], Mhistr[5];

	double square, minsquare;
	int    minindex;
	int    status;

	/* 3-D interpolation requires 2^3 = 8 nodes: */
	double x[3], lo[3], hi[3];
	PHOEBE_spectrum *fv[8];
	char filename[8][255];

	PHOEBE_specrep spec;

	status = query_spectra_repository (PHOEBE_KURUCZ_DIR, &spec);
	if (status != SUCCESS)
		return status;

	phoebe_debug ("\n");
	phoebe_debug ("Repository location: %s\n", PHOEBE_KURUCZ_DIR);
	phoebe_debug ("Synthetic spectra in the repository: %d\n\n", spec.no);

	/*
	 * Now we have to find the closest match from the grid; we shall use
	 * least squares for this ;) :
	 */

	minsquare = pow (T - spec.prop[0].temperature, 2) + 
                pow (g - spec.prop[0].gravity,     2) +
                pow (M - spec.prop[0].metallicity, 2);

	for (i = 1; i < spec.no; i++) {
		square = pow (T - spec.prop[i].temperature, 2) + 
                 pow (g - spec.prop[i].gravity,     2) +
		         pow (M - spec.prop[i].metallicity, 2);
		if (square < minsquare) {
			minsquare = square; minindex = i;
		}
	}

	phoebe_debug ("The closest spectrum is: R=%d at [%d, %d]\n                         T=%d, [M/H]=%d, logg=%d\n", spec.prop[minindex].resolution, 
		spec.prop[minindex].lambda_min, spec.prop[minindex].lambda_max, spec.prop[minindex].temperature,
		spec.prop[minindex].metallicity, spec.prop[minindex].gravity);

	/*
	 * Since we now know which is the closest spectrum, let's find the limiting
	 * values for all parameters:
	 */

	if (T >= spec.prop[minindex].temperature) {
		Tlow  = spec.prop[minindex].temperature;
		Thigh = 2 * spec.prop[minindex].temperature;  /* This should suffice! */
		for (i = 0; i < spec.no; i++)
			if ( (spec.prop[i].temperature - spec.prop[minindex].temperature > 0) &&
			     (spec.prop[i].lambda_min  == spec.prop[minindex].lambda_min)     &&
			     (spec.prop[i].lambda_max  == spec.prop[minindex].lambda_max)     &&
			     (spec.prop[i].metallicity == spec.prop[minindex].metallicity)    &&
			     (spec.prop[i].gravity     == spec.prop[minindex].gravity) )
				if (spec.prop[i].temperature - spec.prop[minindex].temperature < Thigh - Tlow)
					Thigh = spec.prop[i].temperature;
	}
	if (T < spec.prop[minindex].temperature) {
		Thigh = spec.prop[minindex].temperature;
		Tlow  = spec.prop[minindex].temperature / 2;  /* This should suffice! */
		for (i = 0; i < spec.no; i++)
			if ( (spec.prop[i].temperature - spec.prop[minindex].temperature < 0) &&
			     (spec.prop[i].lambda_min  == spec.prop[minindex].lambda_min)     &&
			     (spec.prop[i].lambda_max  == spec.prop[minindex].lambda_max)     &&
			     (spec.prop[i].metallicity == spec.prop[minindex].metallicity)    &&
			     (spec.prop[i].gravity     == spec.prop[minindex].gravity) )
				if (spec.prop[minindex].temperature - spec.prop[i].temperature < Thigh - Tlow)
					Tlow = spec.prop[i].temperature;
	}

	if (M >= spec.prop[minindex].metallicity) {
		Mlow  = spec.prop[minindex].metallicity;
		Mhigh = 5 + spec.prop[minindex].metallicity;  /* This should suffice! */
		for (i = 0; i < spec.no; i++)
			if ( (spec.prop[i].metallicity - spec.prop[minindex].metallicity > 0) &&
			     (spec.prop[i].metallicity - spec.prop[minindex].metallicity < Mhigh - Mlow) &&
				 (spec.prop[i].lambda_min  == spec.prop[minindex].lambda_min)     &&
			     (spec.prop[i].lambda_max  == spec.prop[minindex].lambda_max)     &&
			     (spec.prop[i].temperature == spec.prop[minindex].temperature)    &&
			     (spec.prop[i].gravity     == spec.prop[minindex].gravity) )
					Mhigh = spec.prop[i].metallicity;
	}
	if (M < spec.prop[minindex].metallicity) {
		Mhigh = spec.prop[minindex].metallicity;
		Mlow  = spec.prop[minindex].metallicity - 5;  /* This should suffice! */
		for (i = 0; i < spec.no; i++)
			if ( (spec.prop[i].metallicity - spec.prop[minindex].metallicity < 0) &&
			     (spec.prop[i].lambda_min  == spec.prop[minindex].lambda_min)     &&
			     (spec.prop[i].lambda_max  == spec.prop[minindex].lambda_max)     &&
			     (spec.prop[i].temperature == spec.prop[minindex].temperature)    &&
			     (spec.prop[i].gravity     == spec.prop[minindex].gravity) )
				if (spec.prop[minindex].metallicity - spec.prop[i].metallicity < Mhigh - Mlow)
					Mlow = spec.prop[i].metallicity;
	}

	if (g >= spec.prop[minindex].gravity) {
		glow  = spec.prop[minindex].gravity;
		ghigh = 2 * spec.prop[minindex].gravity;      /* This should suffice! */
		for (i = 0; i < spec.no; i++)
			if ( (spec.prop[i].gravity - spec.prop[minindex].gravity > 0) &&
			     (spec.prop[i].lambda_min  == spec.prop[minindex].lambda_min)     &&
			     (spec.prop[i].lambda_max  == spec.prop[minindex].lambda_max)     &&
			     (spec.prop[i].temperature == spec.prop[minindex].temperature)    &&
			     (spec.prop[i].metallicity == spec.prop[minindex].metallicity) )
				if (spec.prop[i].gravity - spec.prop[minindex].gravity < ghigh - glow)
					ghigh = spec.prop[i].gravity;
	}
	if (g < spec.prop[minindex].gravity) {
		ghigh = spec.prop[minindex].gravity;
		glow  = spec.prop[minindex].gravity / 2;      /* This should suffice! */
		for (i = 0; i < spec.no; i++)
			if ( (spec.prop[i].gravity - spec.prop[minindex].gravity < 0) &&
			     (spec.prop[i].lambda_min  == spec.prop[minindex].lambda_min)     &&
			     (spec.prop[i].lambda_max  == spec.prop[minindex].lambda_max)     &&
			     (spec.prop[i].temperature == spec.prop[minindex].temperature)    &&
			     (spec.prop[i].metallicity == spec.prop[minindex].metallicity) )
				if (spec.prop[minindex].gravity - spec.prop[i].gravity < ghigh - glow)
					glow = spec.prop[i].gravity;
	}

	phoebe_debug ("\n");
	phoebe_debug ("Temperature range: [%d, %d]\n", Tlow, Thigh);
	phoebe_debug ("Metallicity range: [%d, %d]\n", Mlow, Mhigh);
	phoebe_debug ("Gravity range:     [%d, %d]\n", glow, ghigh);

	/* We don't need the grid anymore, so let's free the memory:              */
	free (spec.prop);

	/* Let's build interpolation structures:                                  */
	 x[0] = T;      x[1] = g;      x[2] = M;
	lo[0] = Tlow;  lo[1] = glow;  lo[2] = Mlow;
	hi[0] = Thigh; hi[1] = ghigh; hi[2] = Mhigh;

	/* The filename changes depending on the sign of the metallicity: */
	if (Mlow < 0)
		sprintf (Mlostr, "M%02d", -Mlow);
	else
		sprintf (Mlostr, "P%02d", Mlow);

	if (Mhigh < 0)
		sprintf (Mhistr, "M%02d", -Mhigh);
	else
		sprintf (Mhistr, "P%02d", Mhigh);

	/* Node  0: (0, 0, 0) */
		sprintf (filename[ 0], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mlostr, Tlow, glow);
	/* Node  1: (1, 0, 0) */
		sprintf (filename[ 1], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mlostr, Thigh, glow);
	/* Node  2: (0, 1, 0) */
		sprintf (filename[ 2], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mlostr, Tlow, ghigh);
	/* Node  3: (1, 1, 0) */
		sprintf (filename[ 3], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mlostr, Thigh, ghigh);
	/* Node  4: (0, 0, 1) */
		sprintf (filename[ 4], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mhistr, Tlow, glow);
	/* Node  5: (1, 0, 1) */
		sprintf (filename[ 5], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mhistr, Thigh, glow);
	/* Node  6: (0, 1, 1) */
		sprintf (filename[ 6], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mhistr, Tlow, ghigh);
	/* Node  7: (1, 1, 1) */
		sprintf (filename[ 7], "%s/F250010500V000-R20000%sT%dG%dK2NOVER.ASC", PHOEBE_KURUCZ_DIR, Mhistr, Thigh, ghigh);

	/* Read in the node spectra; if readout fails, free memory and abort.     */
	for (i = 0; i < 8; i++) {
		fv[i] = phoebe_spectrum_new_from_file (filename[i]);
		if (!fv[i]) {
			phoebe_lib_error ("spectrum %s not found, aborting.\n", filename[i]);
			for (j = 0; j < i-1; j++)
				phoebe_spectrum_free (fv[j]);
			return ERROR_SPECTRUM_NOT_IN_REPOSITORY;
		}
	}

	/* Everything seems to be ok; proceed to the interpolation.               */
	phoebe_interpolate (3, x, lo, hi, TYPE_SPECTRUM, fv);

	/* Free all except the first spectrum: */
	for (i = 1; i < 8; i++)
		phoebe_spectrum_free (fv[i]);

	/* Assign the passed argument to the first spectrum: */
	*spectrum = fv[0];

	/* All spectra in the repository are in wavelength-space: */
	(*spectrum)->disp = PHOEBE_SPECTRUM_DISPERSION_LOG;

	return SUCCESS;
}

int phoebe_spectrum_apply_doppler_shift (PHOEBE_spectrum **dest, PHOEBE_spectrum *src, double velocity)
{
	/*
	 * This function applies a Doppler shift to the spectrum. It does that by
	 * first shifting the ranges according to the passed velocity and then
	 * rebins the obtained histogram to the original bins.
	 *
	 * Input parameters:
	 *
	 *   dest      ..  Doppler-shifted spectrum
	 *   src       ..  Original (unmodified) spectrum
	 *   velocity  ..  Radial velocity in km/s
	 */

	int i, status;

	*dest = phoebe_spectrum_duplicate (src);

	switch (src->disp) {
		case PHOEBE_SPECTRUM_DISPERSION_LOG:
			for (i = 0; i < src->data->bins+1; i++)
				src->data->range[i] *= 1.0 + velocity / 299791.0;
		break;
		case PHOEBE_SPECTRUM_DISPERSION_LINEAR:
			for (i = 0; i < src->data->bins+1; i++)
				src->data->range[i] += velocity / 299791.0;
		break;
	}

	status = phoebe_hist_rebin ((*dest)->data, src->data, PHOEBE_HIST_CONSERVE_VALUES);

	return status;
}

int phoebe_spectrum_rebin (PHOEBE_spectrum **src, PHOEBE_spectrum_dispersion disp, double ll, double ul, double R)
{
	/*
	 * This function resamples the passed spectrum to a resolution R. It works
	 * for both degrading and enhancing spectra.
	 *
	 * Return values:
	 *
	 *   ERROR_SPECTRA_NO_OVERLAP
	 *   SUCCESS
	 */

	int status;
	PHOEBE_spectrum *dest;

	if (!*src)
		return ERROR_SPECTRUM_NOT_INITIALIZED;
	if ((*src)->data->bins == 0)
		return ERROR_SPECTRUM_NOT_ALLOCATED;

	if (disp == PHOEBE_SPECTRUM_DISPERSION_NONE) disp = PHOEBE_SPECTRUM_DISPERSION_LINEAR;

	/* Create a new spectrum with a given dispersion type and sampling power: */
	dest = phoebe_spectrum_create (ll, ul, R, disp);

	/* Resample the histogram in the spectrum: */
	status = phoebe_hist_rebin (dest->data, (*src)->data, PHOEBE_HIST_CONSERVE_VALUES);
	if (status != SUCCESS) {
		phoebe_spectrum_free (dest);
		return status;
	}

	phoebe_spectrum_free (*src);
	*src = dest;

	return SUCCESS;
}

int phoebe_spectra_add (PHOEBE_spectrum **dest, PHOEBE_spectrum *src1, PHOEBE_spectrum *src2)
{
	/*
	 * This function adds the flux parts of the two spectra. It is primitive
	 * in a sense that it allows only spectra with identical wavelength parts
	 * (and thus equal dimensions) to be added.
	 */

	int i;

	/* Are the spectra valid? */
	if (!src1 || !src2)
		return ERROR_SPECTRUM_NOT_INITIALIZED;

	/* Are the spectra of the same dimension? */
	if (src1->data->bins != src2->data->bins)
		return ERROR_SPECTRA_DIMENSION_MISMATCH;

	/* Everything seems to be ok; let's allocate space: */
	phoebe_spectrum_alloc (*dest, src1->data->bins);

	/* Do the sumation, but only if wavelengths are aligned: */
	for (i = 0; i < src1->data->bins; i++) {
		if (fabs (src1->data->range[i]-src2->data->range[i]) < PHOEBE_NUMERICAL_ACCURACY) {
			(*dest)->data->range[i] = src1->data->range[i];
			(*dest)->data->val[i] = src1->data->val[i] + src2->data->val[i];
		} else {
			phoebe_spectrum_free (*dest);
			return ERROR_SPECTRA_NOT_ALIGNED;
		}
	}
	/* And, of course, don't forget the last bin: */
	(*dest)->data->range[i] = src1->data->range[i];

	return SUCCESS;
}

int phoebe_spectra_subtract (PHOEBE_spectrum **dest, PHOEBE_spectrum *src1, PHOEBE_spectrum *src2)
{
	/*
	 * This function subtracts the flux parts of the two spectra. It is
	 * primitive in a sense that it allows only spectra with identical
	 * wavelength parts (and thus equal dimensions) to be subtracted.
	 */

	int i;

	/* Are the spectra valid? */
	if (!src1 || !src2)
		return ERROR_SPECTRUM_NOT_INITIALIZED;

	/* Are the spectra of the same dimension? */
	if (src1->data->bins != src2->data->bins)
		return ERROR_SPECTRA_DIMENSION_MISMATCH;

	/* Everything seems to be ok; let's allocate space: */
	phoebe_spectrum_alloc (*dest, src1->data->bins);

	/* Do the subtraction, but only if wavelengths are aligned: */
	for (i = 0; i < src1->data->bins; i++) {
		if (fabs (src1->data->range[i]-src2->data->range[i]) < PHOEBE_NUMERICAL_ACCURACY) {
			(*dest)->data->range[i] = src1->data->range[i];
			(*dest)->data->val[i] = src1->data->val[i] - src2->data->val[i];
		} else {
			phoebe_spectrum_free (*dest);
			return ERROR_SPECTRA_NOT_ALIGNED;
		}
	}
	/* And, of course, don't forget the last bin: */
	(*dest)->data->range[i] = src1->data->range[i];

	return SUCCESS;
}

int phoebe_spectra_merge (PHOEBE_spectrum **dest, PHOEBE_spectrum *src1, PHOEBE_spectrum *src2, double w1, double w2, double ll, double ul, double Rs)
{
	/*
	 * This function merges the two spectra src1 and src2 into a newly
	 * created spectrum dest. The source spectra are copied and resampled to
	 * the passed wavelength interval [ll, ul] and resolution R, so the sizes
	 * may be different. The weighting of both spectra is normalized to the
	 * sum of both weights. Since the function works with copies of original
	 * spectra, their contents are not modified.
	 *
	 * I/O parameters:
	 *
	 *   dest  ..  convolved spectrum
	 *   src1  ..  original (unmodified) spectrum 1
	 *   src2  ..  original (unmodified) spectrum 2
	 *   w1    ..  weight of src1
	 *   w2    ..  weight of src2
	 *
	 * Output values:
	 *
	 *   ERROR_INVALID_WAVELENGTH_INTERVAL
	 *   ERROR_INVALID_SAMPLING_POWER
	 *   ERROR_SPECTRA_NO_OVERLAP
	 *   SUCCESS
	 */

	int i, status;
	PHOEBE_spectrum *s1, *s2;

	if (ll >= ul) return ERROR_INVALID_WAVELENGTH_INTERVAL;
	if (Rs <= 1)  return ERROR_INVALID_SAMPLING_POWER;

	s1 = phoebe_spectrum_duplicate (src1);
	status = phoebe_spectrum_rebin (&s1, PHOEBE_SPECTRUM_DISPERSION_LINEAR, s1->data->range[0], s1->data->range[s1->data->bins], Rs);
	if (status != SUCCESS) {
		phoebe_spectrum_free (s1);
		return status;
	}

	s2 = phoebe_spectrum_duplicate (src2);
	status = phoebe_spectrum_rebin (&s2, PHOEBE_SPECTRUM_DISPERSION_LINEAR, s2->data->range[0], s2->data->range[s2->data->bins], Rs);
	if (status != SUCCESS) {
		phoebe_spectrum_free (s1);
		phoebe_spectrum_free (s2);
		return status;
	}

	*dest = phoebe_spectrum_create (ll, ul, Rs, PHOEBE_SPECTRUM_DISPERSION_LINEAR);

	for (i = 0; i < (*dest)->data->bins; i++)
		(*dest)->data->val[i] = (w1 * s1->data->val[i] + w2 * s2->data->val[i])/(w1+w2);

	phoebe_spectrum_free (s1);
	phoebe_spectrum_free (s2);

	return SUCCESS;
}

int phoebe_spectra_multiply (PHOEBE_spectrum **dest, PHOEBE_spectrum *src1, PHOEBE_spectrum *src2, double ll, double ul, double R)
{
	/*
	 * This function multiplies the two spectra src1 and src2 into the newly
	 * created and allocated spectrum dest that spans on the wavelength
	 * interval [ll, ul] and is sampled with a resolution R.
	 *
	 * Input parameters:
	 *
	 *   dest    ..  newly created multiplied spectrum
	 *   src1    ..  input spectrum 1
	 *   src2    ..  input spectrum 2
	 *   ll      ..  lower wavelength interval limit
	 *   ul      ..  upper wavelength interval limit
	 *   R       ..  resolution
	 */

	int i;
	PHOEBE_spectrum *s1, *s2;

	s1 = phoebe_spectrum_duplicate (src1);
	s2 = phoebe_spectrum_duplicate (src2);

	phoebe_spectrum_rebin (&s1, PHOEBE_SPECTRUM_DISPERSION_LOG, s1->data->range[0], s1->data->range[s1->data->bins], R);
	phoebe_spectrum_rebin (&s2, PHOEBE_SPECTRUM_DISPERSION_LOG, s2->data->range[0], s2->data->range[s2->data->bins], R);

	*dest = phoebe_spectrum_create (ll, ul, R, PHOEBE_SPECTRUM_DISPERSION_LOG);

	for (i = 0; i < (*dest)->data->bins; i++) {
		(*dest)->data->val[i] = s1->data->val[i] * s2->data->val[i];
	}

	phoebe_spectrum_free (s1);
	phoebe_spectrum_free (s2);

	return SUCCESS;
}

int phoebe_spectrum_multiply_by (PHOEBE_spectrum **dest, PHOEBE_spectrum *src, double factor)
{
	/*
	 * This function multiplies spectrum src by factor.
	 */

	int i;

	/* Is the spectrum valid? */
	if (!src)
		return ERROR_SPECTRUM_NOT_INITIALIZED;

	/* Everything seems to be ok; let's allocate space: */
	phoebe_spectrum_alloc (*dest, src->data->bins);

	/* Do the multiplication, but only if wavelengths are aligned: */
	for (i = 0; i < src->data->bins; i++) {
		(*dest)->data->range[i] = src->data->range[i];
		(*dest)->data->val[i] = factor * src->data->val[i];
	}
	/* And, of course, don't forget the last bin: */
	(*dest)->data->range[i] = src->data->range[i];

	return SUCCESS;
}

int phoebe_spectrum_integrate (PHOEBE_spectrum *spectrum, double ll, double ul, double *result)
{
	/*
	 * This function integrates the spectrum on the wavelength interval
	 * [ll, ul]. The values ll and ul are optional; if they are absent,
	 * default values (the whole spectrum range) will be used.
	 *
	 * The function evaluates the following expression:
	 *
	 *   I = \int_ll^ul s (\lambda) d\lambda = \sum s (\lambda) \Delta \lambda
	 *
	 * The sum goes over all covered \Delta \lambda's, taking into account
	 * the partial coverage at interval borders.
	 *
	 * Input parameters:
	 *
	 *   spectrum  ..  input spectrum
	 *   ll        ..  lower wavelength interval limit
	 *   ul        ..  upper wavelength interval limit
	 *   result    ..  the value of the integral I
	 *
	 * Output values:
	 *
	 *   SUCCESS
	 */

	int l = 1;
	double sum = 0.0;

	/* Fast-forward to the first bin of wavelength coverage: */
	while (ll > spectrum->data->range[l]) {
		l++;
		if (l == spectrum->data->bins) {
			*result = 0.0;
			return SUCCESS;
		}
	}

	while (ul > spectrum->data->range[l]) {
		sum += (min (spectrum->data->range[l], ul) - max (ll, spectrum->data->range[l-1])) * spectrum->data->val[l-1];
		l++; if (l == spectrum->data->bins) break;
	}

	*result = sum;
	return SUCCESS;
}

double intern_spectrum_evaluate_gaussian (double l1, double l2, double fwhm)
{
	/*
	 * This is an auxiliary function to phoebe_spectrum_broaden () function.
	 * It computes a Gaussian profile centered at l1, with full-width-half-
	 * maximum (FWHM) passed as argument. FWHM is related to \sigma by the
	 * following relation:
	 *
	 *   FWHM = 2 \sqrt{2 \log 2} \sigma,
	 *
	 * where \log is the natural logarithm.
	 */

	return exp (-4*log(2.0)*(l2-l1)*(l2-l1)/fwhm/fwhm);
}

int phoebe_spectrum_broaden (PHOEBE_spectrum **dest, PHOEBE_spectrum *src, double R)
{
	/*
	 * This function broadens the spectrum to the width that is determined by
	 * the passed true resolution. The sampling of the outgoing spectrum is
	 * unchanged (sampling resolution is preserved).
	 *
	 * The operation itself is quite simple: it convolves the input spectrum
	 * with the Gaussian that has standard deviation \sigma_g such that the
	 * convolved spectrum has the resolution R:
	 *
	 *   \sigma_conv^2 = \sigma_orig^2 + \sigma_g^2
	 *
	 * Broadening is done for a constant passed resolution, thus for a variable
	 * sampling step. For low-res spectra (e.g. R <~ 10) the broadening quite
	 * noticeably deviates from the Gaussian profile.
	 *
	 * Return values:
	 *
	 *   ERROR_SPECTRA_DIMENSION_MISMATCH
	 *   ERROR_INVALID_SAMPLING_POWER
	 *   SUCCESS
	 */

	int i, k;
	double s1, s2, w;
	double fwhm;

	if (src->R < 1 || R < 1)
		return ERROR_INVALID_SAMPLING_POWER;

	*dest = phoebe_spectrum_duplicate (src);

	for (i = 0; i < src->data->bins; i++) {
		/* Left part of the Gaussian: */
		k = 0; s1 = 0.0; s2 = 0.0;
		fwhm = src->data->range[i] * sqrt (1./R/R - 1./src->R/src->R);
		while (i-k >= 0 && (w = intern_spectrum_evaluate_gaussian (src->data->range[i], src->data->range[i-k], fwhm)) > 0.01) {
			s1 += w * src->data->val[i-k];
			s2 += w;
			k++;
		}

		/* Right part of the Gaussian: */
		k = 1;
		fwhm = src->data->range[i] * sqrt (1./R/R - 1./src->R/src->R);
		while (i+k < src->data->bins && (w = intern_spectrum_evaluate_gaussian (src->data->range[i], src->data->range[i+k], fwhm)) > 0.01) {
			s1 += w * src->data->val[i+k];
			s2 += w;
			k++;
		}

		/* The correction: */
		(*dest)->data->val[i] = s1/s2;
		(*dest)->R = R;
	}

	return SUCCESS;
}

double intern_spectrum_rotational_broadening_function (double l1, double l2, double vsini, double ldx)
{
	/*
	 * This is the rotational broadening function that is used for Doppler
	 * broadening of the spectrum. It is adopted from Gray, Chapter 17, p.374.
	 */

	double dlambda = l2 - l1;
	double dlambda_max = vsini / 299791.0 * l1;
	double factor = 1.0 - dlambda*dlambda / (dlambda_max*dlambda_max);
	
	if (factor < 0) return 0.0;
	else return (2.0 * (1-ldx) * sqrt (factor) + M_PI/2.0 * ldx * factor) / M_PI / dlambda_max / (1.0-ldx/3.0);
}

int phoebe_spectrum_apply_rotational_broadening (PHOEBE_spectrum **dest, PHOEBE_spectrum *src, double vsini, double ldx)
{
	/*
	 * This function applies the rotational (Doppler) broadening to the src
	 * spectrum and stores the result in spectrum dest.
	 *
	 * Arguments:
	 *
	 *   vsini  ..  v_r \sin i in km/s
	 *   ldx    ..  linear limb darkening coefficient
	 *
	 * The broadening algorithm is adopted from Gray, Chapter 17, p.374.
	 *
	 * Return values:
	 *
	 *   ERROR_BROADENING_INADEQUATE_ACCURACY
	 *   SUCCESS
	 */

	int i, k;
	double s1, s2, G;

	/* Duplicate the source spectrum:                                         */
	*dest = phoebe_spectrum_duplicate (src);

	/* The sign of vsini does not matter for the broadening:                  */
	vsini = fabs (vsini);

	/* Error-handling:                                                        */
	if (vsini < 5.0) return ERROR_BROADENING_INADEQUATE_ACCURACY;

	for (i = 0; i < src->data->bins; i++) {
		k = 0; s1 = 0; s2 = 0;
		while (i-k >= 0 && (G = intern_spectrum_rotational_broadening_function (src->data->range[i], src->data->range[i-k], vsini, ldx)) > 1e-3) {
			s1 += G * src->data->val[i-k];
			s2 += G;
			k++;
		}
		
		k = 1;
		while (i+k < src->data->bins && (G = intern_spectrum_rotational_broadening_function (src->data->range[i], src->data->range[i+k], vsini, ldx)) > 1e-3) {
			s1 += G * src->data->val[i+k];
			s2 += G;
			k++;
		}

		(*dest)->data->val[i] = s1/s2;
	}

	return SUCCESS;
}

int phoebe_spectrum_set_sampling (PHOEBE_spectrum *spectrum, double Rs)
{
	/*
	 * This function sets the spectrum's sampling resolution.
	 */

	if (Rs < 1)
		return ERROR_SPECTRUM_INVALID_SAMPLING;

	spectrum->Rs = Rs;

	return SUCCESS;
}

int phoebe_spectrum_set_resolution (PHOEBE_spectrum *spectrum, double R)
{
	/*
	 * This function sets the spectrum's true resolution.
	 */

	if (R < 1)
		return ERROR_SPECTRUM_INVALID_RESOLUTION;

	spectrum->R = R;

	return SUCCESS;
}
