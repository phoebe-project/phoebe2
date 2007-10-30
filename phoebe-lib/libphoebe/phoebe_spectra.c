#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

/* ftw.h contains ftw (), a function for manipulating directory trees */
#include <ftw.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_calculations.h"
#include "phoebe_configuration.h"
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

#define pdif(a)  ((a) > 0 ? (a) : 0)

PHOEBE_specrep PHOEBE_spectra_repository;

int intern_spectra_repository_process (const char *filename, const struct stat *filestat, int flag)
{
	/**
	 *
	 */

	int i, argmatch;
	char *relative = strrchr (filename, '/');
	int type, T, logg, met, turb, res, alpenh;
	char metsign, alpha = 'S', odftype[3], respow[4], sptype;

	argmatch = sscanf (relative, "/T%5dG%2d%c%2dV000K%1dS%2cNV%3c%c.ASC", &T, &logg, &metsign, &met, &turb, odftype, respow, &sptype);
	phoebe_debug ("%2d matched; ", argmatch);
	if (argmatch == 8) {
		/* Handle metallicity sign: */
		if (metsign == 'M') met = -met;

		/* Commit the temperature to the array of temperature nodes (if new): */
		for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++) {
			if (PHOEBE_spectra_repository.Teffnodes->val.iarray[i] == T)
				break;
		}
		if (i == PHOEBE_spectra_repository.Teffnodes->dim) {
			phoebe_array_realloc (PHOEBE_spectra_repository.Teffnodes, PHOEBE_spectra_repository.Teffnodes->dim+1);
			PHOEBE_spectra_repository.Teffnodes->val.iarray[i] = T;
		}

		/* Commit the log(g) to the array of log(g) nodes (if new): */
		for (i = 0; i < PHOEBE_spectra_repository.loggnodes->dim; i++) {
			if (PHOEBE_spectra_repository.loggnodes->val.iarray[i] == logg)
				break;
		}
		if (i == PHOEBE_spectra_repository.loggnodes->dim) {
			phoebe_array_realloc (PHOEBE_spectra_repository.loggnodes, PHOEBE_spectra_repository.loggnodes->dim+1);
			PHOEBE_spectra_repository.loggnodes->val.iarray[i] = logg;
		}

		/* Commit the metallicity to the array of metallicity nodes (if new): */
		for (i = 0; i < PHOEBE_spectra_repository.metnodes->dim; i++) {
			if (PHOEBE_spectra_repository.metnodes->val.iarray[i] == met)
				break;
		}
		if (i == PHOEBE_spectra_repository.metnodes->dim) {
			phoebe_array_realloc (PHOEBE_spectra_repository.metnodes, PHOEBE_spectra_repository.metnodes->dim+1);
			PHOEBE_spectra_repository.metnodes->val.iarray[i] = met;
		}

		/* Reallocate the memory in blocks of 10000 spectra (for efficiency): */
		if (PHOEBE_spectra_repository.no % 10000 == 0)
			PHOEBE_spectra_repository.prop = phoebe_realloc (PHOEBE_spectra_repository.prop, 10000 * sizeof (*(PHOEBE_spectra_repository.prop)));
		PHOEBE_spectra_repository.no++;

		/* Add terminating characters to read strings: */
		odftype[2] = '\0'; respow[3] = '\0';

		/* Get a numeric value of the resolving power: */
		if      (strcmp (respow, "R20") == 0) res = 20000;
		else if (strcmp (respow, "RVS") == 0) res = 11500;
		else if (strcmp (respow, "RAV") == 0) res =  8500;
		else if (strcmp (respow, "SLN") == 0) res =  2000;
		else if (strcmp (respow, "D01") == 0) res =  2500;
		else if (strcmp (respow, "D10") == 0) res =   250;
		else {
			phoebe_lib_error ("resolving power string %s is invalid, assuming D01.\n", respow);
			res = 2500;
		}

		/* Get a numeric value of alpha enhancement: */
		if (alpha == 'A') alpenh = 4;
		else alpenh = 0;

		/* Get the spectrum type switch: */
		if (sptype == 'F') type = 1; /* flux */
		else               type = 0; /* normalized */

		/* Add the parsed data to the repository: */
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].filename = strdup (filename);
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].type = type;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].lambda_min = 2500;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].lambda_max = 10500;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].resolution = res;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].temperature = T;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].gravity = logg;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].metallicity = met;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].alpha = alpenh;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].microturbulence = turb;
		phoebe_debug ("parsed:  %s\n", relative+1);
	}
	else
		phoebe_debug ("skipped: %s\n", relative+1);

	return SUCCESS;
}

int phoebe_spectra_set_repository (char *rep_name)
{
	int i, j, k, s;

	if (PHOEBE_spectra_repository.no != 0) {
		phoebe_array_free (PHOEBE_spectra_repository.Teffnodes);
		phoebe_array_free (PHOEBE_spectra_repository.loggnodes);
		phoebe_array_free (PHOEBE_spectra_repository.metnodes);

		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			free (PHOEBE_spectra_repository.prop[i].filename);
		free (PHOEBE_spectra_repository.prop);
	}

	PHOEBE_spectra_repository.no = 0;
	PHOEBE_spectra_repository.prop = NULL;

	PHOEBE_spectra_repository.Teffnodes = phoebe_array_new (TYPE_INT_ARRAY);
	PHOEBE_spectra_repository.loggnodes = phoebe_array_new (TYPE_INT_ARRAY);
	PHOEBE_spectra_repository.metnodes  = phoebe_array_new (TYPE_INT_ARRAY);

	if (!rep_name)
		return ERROR_SPECTRA_REPOSITORY_INVALID_NAME;

	if (!filename_is_directory (rep_name))
		return ERROR_SPECTRA_REPOSITORY_NOT_DIRECTORY;

	ftw (rep_name, intern_spectra_repository_process, 10);

	/*
	 * The ftw function put all the T, log g, M/H values into arrays that we
	 * will use to generate a 3D grid. However, these values are added in
	 * order of appearance, whereas we need them sorted. That is why:
	 */

	qsort (PHOEBE_spectra_repository.Teffnodes->val.iarray, PHOEBE_spectra_repository.Teffnodes->dim, sizeof (*(PHOEBE_spectra_repository.Teffnodes->val.iarray)), diff_int);
	qsort (PHOEBE_spectra_repository.loggnodes->val.iarray, PHOEBE_spectra_repository.loggnodes->dim, sizeof (*(PHOEBE_spectra_repository.loggnodes->val.iarray)), diff_int);
	qsort (PHOEBE_spectra_repository.metnodes->val.iarray,  PHOEBE_spectra_repository.metnodes->dim,  sizeof (*(PHOEBE_spectra_repository.metnodes->val.iarray)),  diff_int);

	printf ("Temperature nodes:\n");
	for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++)
		printf ("\t%d\n", PHOEBE_spectra_repository.Teffnodes->val.iarray[i]);
	printf ("Gravity nodes:\n");
	for (i = 0; i < PHOEBE_spectra_repository.loggnodes->dim; i++)
		printf ("\t%d\n", PHOEBE_spectra_repository.loggnodes->val.iarray[i]);
	printf ("Metallicity nodes:\n");
	for (i = 0; i < PHOEBE_spectra_repository.metnodes->dim; i++)
		printf ("\t%d\n", PHOEBE_spectra_repository.metnodes->val.iarray[i]);

	/* Now that the nodes are in place, we create our dynamic 3D grid table: */
	PHOEBE_spectra_repository.table = phoebe_malloc (PHOEBE_spectra_repository.Teffnodes->dim * sizeof (*PHOEBE_spectra_repository.table));
	for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++) {
		PHOEBE_spectra_repository.table[i] = phoebe_malloc (PHOEBE_spectra_repository.loggnodes->dim * sizeof (**PHOEBE_spectra_repository.table));
		for (j = 0; j < PHOEBE_spectra_repository.loggnodes->dim; j++) {
			PHOEBE_spectra_repository.table[i][j] = phoebe_malloc (PHOEBE_spectra_repository.metnodes->dim * sizeof (***PHOEBE_spectra_repository.table));
			for (k = 0; k < PHOEBE_spectra_repository.metnodes->dim; k++)
				PHOEBE_spectra_repository.table[i][j][k] = NULL;
		}
	}

	/* Finally we go over all spectra and populate the table with pointers: */
	for (s = 0; s < PHOEBE_spectra_repository.no; s++) {
		for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++)
			if (PHOEBE_spectra_repository.prop[s].temperature == PHOEBE_spectra_repository.Teffnodes->val.iarray[i])
				break;
		for (j = 0; j < PHOEBE_spectra_repository.loggnodes->dim; j++)
			if (PHOEBE_spectra_repository.prop[s].gravity == PHOEBE_spectra_repository.loggnodes->val.iarray[j])
				break;
		for (k = 0; k < PHOEBE_spectra_repository.metnodes->dim; k++)
			if (PHOEBE_spectra_repository.prop[s].metallicity == PHOEBE_spectra_repository.metnodes->val.iarray[k])
				break;
		PHOEBE_spectra_repository.table[i][j][k] = &(PHOEBE_spectra_repository.prop[s]);
	}

	phoebe_debug ("Table:\n");
	for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++)
		for (j = 0; j < PHOEBE_spectra_repository.loggnodes->dim; j++)
			for (k = 0; k < PHOEBE_spectra_repository.metnodes->dim; k++) {
				if (PHOEBE_spectra_repository.table[i][j][k])
					phoebe_debug ("%d %d %d: %s\n", i, j, k, PHOEBE_spectra_repository.table[i][j][k]->filename);
				else
					phoebe_debug ("%d %d %d: %s\n", i, j, k, "n/a");
			}

	return SUCCESS;
}

int phoebe_spectra_free_repository ()
{
	int i, j;

	if (PHOEBE_spectra_repository.no == 0)
		return SUCCESS;

	/* The 'table' field holds pointers, so we need to remove only the table: */
	for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++) {
		for (j = 0; j < PHOEBE_spectra_repository.loggnodes->dim; j++)
			free (PHOEBE_spectra_repository.table[i][j]);
		free (PHOEBE_spectra_repository.table[i]);
	}
	free (PHOEBE_spectra_repository.table);

	/* Free memory occupied by arrays: */
	phoebe_array_free (PHOEBE_spectra_repository.Teffnodes);
	phoebe_array_free (PHOEBE_spectra_repository.loggnodes);
	phoebe_array_free (PHOEBE_spectra_repository.metnodes);

	/* Finally, free the actual spectra properties: */
	for (i = 0; i < PHOEBE_spectra_repository.no; i++)
		free (PHOEBE_spectra_repository.prop[i].filename);
	free (PHOEBE_spectra_repository.prop);

	return SUCCESS;
}

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

	while ( (filelist = readdir (repository)) ) {
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
	/*
	 * This function opens a two-column file 'filename' and reads its contents
	 * to a newly allocated spectrum. The columns are assumed to contain bin
	 * centers and fluxes.
	 */

	FILE *input;
	PHOEBE_spectrum *spectrum;
	PHOEBE_vector *bin_centers;
	int linecount = 1;

	char line[255];
	char *strptr;
	double wl, flux;

	input = fopen (filename, "r");
	if (!input) return NULL;

	spectrum = phoebe_spectrum_new ();
	bin_centers = phoebe_vector_new ();

	while (!feof (input)) {
		fgets (line, 254, input);
		if (feof (input)) break;

		/* Remove empty lines:                                                */
		if ( (strptr = strchr (line, '\n')) ) *strptr = '\0';

		/* Remove comments (if any):                                          */
		if ( (strptr = strchr (line, '#')) ) *strptr = '\0';

		if (sscanf (line, "%lf %lf", &wl, &flux) == 2) {
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
		default:
			/* fall through for PHOEBE_SPECTRUM_DISPERSION_NONE */
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
	 * memory leaks and there were none (according to valgrind).
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

int phoebe_spectrum_new_from_repository (PHOEBE_spectrum **spectrum, int T, int g, int M)
{
	/*
	 * This function queries the spectra repository, takes closest gridded spe-
	 * ctra from it and linearly interpolates to the spectrum represented by
	 * the passed parameters.
	 *
	 * Input parameters:
	 *
	 *   T    ..  effective temperature
	 *   g    ..  log g/g0 in cgs units
	 *   M    ..  metallicity [M/H] in Solar units
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
	int    minindex = 0;
	int    status;

	/* 3-D interpolation requires 2^3 = 8 nodes: */
	double x[3], lo[3], hi[3];
	PHOEBE_spectrum *fv[8];
	char filename[8][255];
	PHOEBE_vector *specvals;

	char *kuruczdir;

	phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &kuruczdir);

	if (PHOEBE_spectra_repository.no == 0)
		return ERROR_SPECTRA_REPOSITORY_EMPTY;

	phoebe_debug ("\n");
	phoebe_debug ("Synthetic spectra in the repository: %d\n\n", PHOEBE_spectra_repository.no);

	/*
	 * Now we have to find the closest match from the grid; we shall use
	 * least squares for this ;) :
	 */

	minsquare = pow (T - PHOEBE_spectra_repository.prop[0].temperature, 2) + 
                pow (g - PHOEBE_spectra_repository.prop[0].gravity,     2) +
                pow (M - PHOEBE_spectra_repository.prop[0].metallicity, 2);

	for (i = 1; i < PHOEBE_spectra_repository.no; i++) {
		square = pow (T - PHOEBE_spectra_repository.prop[i].temperature, 2) + 
                 pow (g - PHOEBE_spectra_repository.prop[i].gravity,     2) +
		         pow (M - PHOEBE_spectra_repository.prop[i].metallicity, 2);
		if (square < minsquare) {
			minsquare = square; minindex = i;
		}
	}

	phoebe_debug ("The closest spectrum is: R=%d at [%d, %d]\n                         T=%d, [M/H]=%d, logg=%d\n", PHOEBE_spectra_repository.prop[minindex].resolution, 
		PHOEBE_spectra_repository.prop[minindex].lambda_min, PHOEBE_spectra_repository.prop[minindex].lambda_max, PHOEBE_spectra_repository.prop[minindex].temperature,
		PHOEBE_spectra_repository.prop[minindex].metallicity, PHOEBE_spectra_repository.prop[minindex].gravity);

	/*
	 * Since we now know which is the closest spectrum, let's find the limiting
	 * values for all parameters:
	 */

	if (T >= PHOEBE_spectra_repository.prop[minindex].temperature) {
		Tlow  = PHOEBE_spectra_repository.prop[minindex].temperature;
		Thigh = 2 * PHOEBE_spectra_repository.prop[minindex].temperature;  /* This should suffice! */
		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			if ( (PHOEBE_spectra_repository.prop[i].temperature - PHOEBE_spectra_repository.prop[minindex].temperature > 0) &&
			     (PHOEBE_spectra_repository.prop[i].lambda_min  == PHOEBE_spectra_repository.prop[minindex].lambda_min)     &&
			     (PHOEBE_spectra_repository.prop[i].lambda_max  == PHOEBE_spectra_repository.prop[minindex].lambda_max)     &&
			     (PHOEBE_spectra_repository.prop[i].metallicity == PHOEBE_spectra_repository.prop[minindex].metallicity)    &&
			     (PHOEBE_spectra_repository.prop[i].gravity     == PHOEBE_spectra_repository.prop[minindex].gravity) )
				if (PHOEBE_spectra_repository.prop[i].temperature - PHOEBE_spectra_repository.prop[minindex].temperature < Thigh - Tlow)
					Thigh = PHOEBE_spectra_repository.prop[i].temperature;
	}
	if (T < PHOEBE_spectra_repository.prop[minindex].temperature) {
		Thigh = PHOEBE_spectra_repository.prop[minindex].temperature;
		Tlow  = PHOEBE_spectra_repository.prop[minindex].temperature / 2;  /* This should suffice! */
		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			if ( (PHOEBE_spectra_repository.prop[i].temperature - PHOEBE_spectra_repository.prop[minindex].temperature < 0) &&
			     (PHOEBE_spectra_repository.prop[i].lambda_min  == PHOEBE_spectra_repository.prop[minindex].lambda_min)     &&
			     (PHOEBE_spectra_repository.prop[i].lambda_max  == PHOEBE_spectra_repository.prop[minindex].lambda_max)     &&
			     (PHOEBE_spectra_repository.prop[i].metallicity == PHOEBE_spectra_repository.prop[minindex].metallicity)    &&
			     (PHOEBE_spectra_repository.prop[i].gravity     == PHOEBE_spectra_repository.prop[minindex].gravity) )
				if (PHOEBE_spectra_repository.prop[minindex].temperature - PHOEBE_spectra_repository.prop[i].temperature < Thigh - Tlow)
					Tlow = PHOEBE_spectra_repository.prop[i].temperature;
	}

	if (M >= PHOEBE_spectra_repository.prop[minindex].metallicity) {
		Mlow  = PHOEBE_spectra_repository.prop[minindex].metallicity;
		Mhigh = 5 + PHOEBE_spectra_repository.prop[minindex].metallicity;  /* This should suffice! */
		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			if ( (PHOEBE_spectra_repository.prop[i].metallicity - PHOEBE_spectra_repository.prop[minindex].metallicity > 0) &&
			     (PHOEBE_spectra_repository.prop[i].metallicity - PHOEBE_spectra_repository.prop[minindex].metallicity < Mhigh - Mlow) &&
				 (PHOEBE_spectra_repository.prop[i].lambda_min  == PHOEBE_spectra_repository.prop[minindex].lambda_min)     &&
			     (PHOEBE_spectra_repository.prop[i].lambda_max  == PHOEBE_spectra_repository.prop[minindex].lambda_max)     &&
			     (PHOEBE_spectra_repository.prop[i].temperature == PHOEBE_spectra_repository.prop[minindex].temperature)    &&
			     (PHOEBE_spectra_repository.prop[i].gravity     == PHOEBE_spectra_repository.prop[minindex].gravity) )
					Mhigh = PHOEBE_spectra_repository.prop[i].metallicity;
	}
	if (M < PHOEBE_spectra_repository.prop[minindex].metallicity) {
		Mhigh = PHOEBE_spectra_repository.prop[minindex].metallicity;
		Mlow  = PHOEBE_spectra_repository.prop[minindex].metallicity - 5;  /* This should suffice! */
		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			if ( (PHOEBE_spectra_repository.prop[i].metallicity - PHOEBE_spectra_repository.prop[minindex].metallicity < 0) &&
			     (PHOEBE_spectra_repository.prop[i].lambda_min  == PHOEBE_spectra_repository.prop[minindex].lambda_min)     &&
			     (PHOEBE_spectra_repository.prop[i].lambda_max  == PHOEBE_spectra_repository.prop[minindex].lambda_max)     &&
			     (PHOEBE_spectra_repository.prop[i].temperature == PHOEBE_spectra_repository.prop[minindex].temperature)    &&
			     (PHOEBE_spectra_repository.prop[i].gravity     == PHOEBE_spectra_repository.prop[minindex].gravity) )
				if (PHOEBE_spectra_repository.prop[minindex].metallicity - PHOEBE_spectra_repository.prop[i].metallicity < Mhigh - Mlow)
					Mlow = PHOEBE_spectra_repository.prop[i].metallicity;
	}

	if (g >= PHOEBE_spectra_repository.prop[minindex].gravity) {
		glow  = PHOEBE_spectra_repository.prop[minindex].gravity;
		ghigh = 2 * PHOEBE_spectra_repository.prop[minindex].gravity;      /* This should suffice! */
		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			if ( (PHOEBE_spectra_repository.prop[i].gravity - PHOEBE_spectra_repository.prop[minindex].gravity > 0) &&
			     (PHOEBE_spectra_repository.prop[i].lambda_min  == PHOEBE_spectra_repository.prop[minindex].lambda_min)     &&
			     (PHOEBE_spectra_repository.prop[i].lambda_max  == PHOEBE_spectra_repository.prop[minindex].lambda_max)     &&
			     (PHOEBE_spectra_repository.prop[i].temperature == PHOEBE_spectra_repository.prop[minindex].temperature)    &&
			     (PHOEBE_spectra_repository.prop[i].metallicity == PHOEBE_spectra_repository.prop[minindex].metallicity) )
				if (PHOEBE_spectra_repository.prop[i].gravity - PHOEBE_spectra_repository.prop[minindex].gravity < ghigh - glow)
					ghigh = PHOEBE_spectra_repository.prop[i].gravity;
	}
	if (g < PHOEBE_spectra_repository.prop[minindex].gravity) {
		ghigh = PHOEBE_spectra_repository.prop[minindex].gravity;
		glow  = PHOEBE_spectra_repository.prop[minindex].gravity / 2;      /* This should suffice! */
		for (i = 0; i < PHOEBE_spectra_repository.no; i++)
			if ( (PHOEBE_spectra_repository.prop[i].gravity - PHOEBE_spectra_repository.prop[minindex].gravity < 0) &&
			     (PHOEBE_spectra_repository.prop[i].lambda_min  == PHOEBE_spectra_repository.prop[minindex].lambda_min)     &&
			     (PHOEBE_spectra_repository.prop[i].lambda_max  == PHOEBE_spectra_repository.prop[minindex].lambda_max)     &&
			     (PHOEBE_spectra_repository.prop[i].temperature == PHOEBE_spectra_repository.prop[minindex].temperature)    &&
			     (PHOEBE_spectra_repository.prop[i].metallicity == PHOEBE_spectra_repository.prop[minindex].metallicity) )
				if (PHOEBE_spectra_repository.prop[minindex].gravity - PHOEBE_spectra_repository.prop[i].gravity < ghigh - glow)
					glow = PHOEBE_spectra_repository.prop[i].gravity;
	}

	phoebe_debug ("\n");
	phoebe_debug ("Temperature range: [%d, %d]\n", Tlow, Thigh);
	phoebe_debug ("Metallicity range: [%d, %d]\n", Mlow, Mhigh);
	phoebe_debug ("Gravity range:     [%d, %d]\n", glow, ghigh);

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
		sprintf (filename[ 0], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Tlow, glow, Mlostr);
	/* Node  1: (1, 0, 0) */
		sprintf (filename[ 1], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Thigh, glow, Mlostr);
	/* Node  2: (0, 1, 0) */
		sprintf (filename[ 2], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Tlow, ghigh, Mlostr);
	/* Node  3: (1, 1, 0) */
		sprintf (filename[ 3], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Thigh, ghigh, Mlostr);
	/* Node  4: (0, 0, 1) */
		sprintf (filename[ 4], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Tlow, glow, Mhistr);
	/* Node  5: (1, 0, 1) */
		sprintf (filename[ 5], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Thigh, glow, Mhistr);
	/* Node  6: (0, 1, 1) */
		sprintf (filename[ 6], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Tlow, ghigh, Mhistr);
	/* Node  7: (1, 1, 1) */
		sprintf (filename[ 7], "%s/T%05dG%2d%sV000K2SNWNVD01F.ASC", kuruczdir, Thigh, ghigh, Mhistr);

	/* Read in the node spectra; if the readout fails, free memory and abort. */
	for (i = 0; i < 8; i++) {
		fv[i] = phoebe_spectrum_create (2500, 10500, 2500, PHOEBE_SPECTRUM_DISPERSION_LINEAR);
		specvals = phoebe_vector_new_from_column (filename[i], 1);
		if (!specvals) {
			phoebe_lib_error ("spectrum %s not found, aborting.\n", filename[i]);
			for (j = 0; j < i-1; j++)
				phoebe_spectrum_free (fv[j]);
			return ERROR_SPECTRUM_NOT_IN_REPOSITORY;
		}

		status = phoebe_hist_set_values (fv[i]->data, specvals);
		if (status != SUCCESS)
			phoebe_lib_error ("%s", phoebe_error (status));
		phoebe_vector_free (specvals);
	}

	/* Everything seems to be ok; proceed to the interpolation.               */
	phoebe_interpolate (3, x, lo, hi, TYPE_SPECTRUM, fv);

	/* Free all except the first spectrum: */
	for (i = 1; i < 8; i++)
		phoebe_spectrum_free (fv[i]);

	/* Assign the passed argument to the first spectrum: */
	*spectrum = fv[0];

	/* All spectra in the repository are in pixel space: */
	(*spectrum)->disp = PHOEBE_SPECTRUM_DISPERSION_LINEAR;

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
		default:
			/* fall through for PHOEBE_SPECTRUM_DISPERSION_NONE */
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
	status = phoebe_hist_rebin (dest->data, (*src)->data, PHOEBE_HIST_CONSERVE_DENSITY);
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

	phoebe_spectrum_rebin (&s1, PHOEBE_SPECTRUM_DISPERSION_LOG, ll, ul, R);
	phoebe_spectrum_rebin (&s2, PHOEBE_SPECTRUM_DISPERSION_LOG, ll, ul, R);

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
