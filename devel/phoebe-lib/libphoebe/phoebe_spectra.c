#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#include "phoebe_build_config.h"

/*
 * ftw.h contains ftw (), a function for manipulating directory trees.
 * Unfortunately, this is a POSIX extension and is present in glibc only.
 * For windows we need to make workarounds.
 */

#ifdef HAVE_FTW_H
	#include <ftw.h>
#endif

#include "phoebe_accessories.h"
#include "phoebe_calculations.h"
#include "phoebe_configuration.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
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

PHOEBE_specrep PHOEBE_spectra_repository;

/**
 * SECTION:phoebe_spectra
 * @title: PHOEBE spectra
 * @short_description: functions that facilitate spectra manipulation
 *
 * These are the functions that facilitate spectra manipulation, including
 * the repository I/O.
 */

int intern_spectra_repository_process (const char *filename, const struct stat *filestat, int flag)
{
	/*
	 * intern_spectra_repository_process:
	 * @filename: filename (relative to dirpath) to be processed
	 * @filestat: a variable for all file properties
	 * @flag: file type passed by ftw(), which we don't use.
	 * 
	 * This is an internal function and should not be used by any function
	 * except phoebe_spectra_set_repository (). It parses a filename passed
	 * by the ftw() function and compares it to the following predefined
	 * format:
	 * 
	 *   T%%5dG%%2d%%c%%2dV000K%%1dS%%2cNV%%3c%%c.ASC
	 *
	 * Only 0 rotational velocities are matched because phoebe uses the internal
	 * function for rotational broadening.
	 *
	 * Returns: #PHOEBE_error_code.
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
			PHOEBE_spectra_repository.prop = phoebe_realloc (PHOEBE_spectra_repository.prop, (PHOEBE_spectra_repository.no/10000+1)*10000*sizeof (*(PHOEBE_spectra_repository.prop)));
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

int intern_spectra_alt_repository_process (const char *filename, const struct stat *filestat, int flag)
{
	/*
	 * intern_spectra_alt_repository_process:
	 * @filename: filename (relative to dirpath) to be processed
	 * @filestat: a variable for all file properties
	 * @flag: file type passed by ftw(), which we don't use.
	 * 
	 * This is an internal function and should not be used by any function
	 * except phoebe_spectra_set_repository (). It parses a filename passed
	 * by the ftw() function and compares it to the following predefined
	 * format:
	 * 
	 *   T%%5dG%%2d%%c%%2dM1.000.spectrum
	 *
	 * This will pick out only center-of-limb intensities (mu = 1).
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	int i, argmatch;
	char *relative = strrchr (filename, '/');
	int T, logg, met;
	double mu;
	char metsign;

	argmatch = sscanf (relative, "/T%5dG%2d%c%2dM%lf.spectrum", &T, &logg, &metsign, &met, &mu);
	phoebe_debug ("%2d matched; ", argmatch);
	if (argmatch == 5 && mu > 0.999) {
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

		/* Reallocate the memory in blocks of 3800 spectra (for efficiency;
		 * there are 3800 spectra in the generic repository): */
		if (PHOEBE_spectra_repository.no % 3800 == 0)
			PHOEBE_spectra_repository.prop = phoebe_realloc (PHOEBE_spectra_repository.prop, (PHOEBE_spectra_repository.no/3800+1)*3800*sizeof (*(PHOEBE_spectra_repository.prop)));
		PHOEBE_spectra_repository.no++;

		/* Add the parsed data to the repository: */
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].filename = strdup (filename);
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].type = -1;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].lambda_min = 1760;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].lambda_max = 40000;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].resolution = -1;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].temperature = T;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].gravity = logg;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].metallicity = met;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].alpha = -1;
		PHOEBE_spectra_repository.prop[PHOEBE_spectra_repository.no-1].microturbulence = -1;
		phoebe_debug ("parsed:  %s\n", relative+1);
	}
	else
		phoebe_debug ("skipped: %s\n", relative+1);

	return SUCCESS;
}

int phoebe_spectra_set_repository (char *rep_name)
{
	/**
	 * phoebe_spectra_set_repository:
	 * @rep_name: path to the repository
	 *
	 * Traverses the path @rep_name and parses all found filenames for spectra.
	 * Filenames must match the predefined format:
	 * 
	 *   T%%5dG%%2d%%c%%2dV000K%%1dS%%2cNV%%3c%%c.ASC
	 *
	 * Only 0 rotational velocity spectra are parsed because PHOEBE uses its
	 * internal function for rotational broadening. Where available, this
	 * function calls ftw() for traversing the tree; the spectra can then be
	 * separated into subdirectories (i.e. Munari et al. 2005) and ftw() will
	 * traverse up to 10 subdirectories deep. When ftw() is not available,
	 * a simplified parsing function is used that requires the spectra to be
	 * all in a single subdirectory.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

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

	if (!phoebe_filename_is_directory (rep_name))
		return ERROR_SPECTRA_REPOSITORY_NOT_DIRECTORY;

#ifdef HAVE_FTW_H
	/* Under Linux we can use glibc's ftw() function to walk the file tree: */
	ftw (rep_name, intern_spectra_alt_repository_process, 10);
#else
{
	/*
	 * Under Windows, unfortunately, we can't... Traversing the tree manually
	 * is very complicated to write from scratch. So if you want to use the
	 * spectra repository under windows, make sure all spectra are in a single
	 * subdirectory!
	 */

	int status;
	DIR *dirlist;
	struct dirent *file;
	char *filename;

	status = phoebe_open_directory (&dirlist, rep_name);
	if (status != SUCCESS) {
		phoebe_lib_error ("directory %s cannot be opened, aborting.\n", rep_name);
		return status;
	}

	while ((file = readdir (dirlist))) {
		filename = phoebe_concatenate_strings (rep_name, "/", file->d_name, NULL);
		intern_spectra_alt_repository_process (filename, NULL, 0);
		free (filename);
	}

	phoebe_close_directory (&dirlist);
}
#endif

	/*
	 * The above snippets put all the T, log g, M/H values into arrays that we
	 * will use to generate a 3D grid. However, these values are added in
	 * order of appearance, whereas we need them sorted. That is why:
	 */

	qsort (PHOEBE_spectra_repository.Teffnodes->val.iarray, PHOEBE_spectra_repository.Teffnodes->dim, sizeof (*(PHOEBE_spectra_repository.Teffnodes->val.iarray)), diff_int);
	qsort (PHOEBE_spectra_repository.loggnodes->val.iarray, PHOEBE_spectra_repository.loggnodes->dim, sizeof (*(PHOEBE_spectra_repository.loggnodes->val.iarray)), diff_int);
	qsort (PHOEBE_spectra_repository.metnodes->val.iarray,  PHOEBE_spectra_repository.metnodes->dim,  sizeof (*(PHOEBE_spectra_repository.metnodes->val.iarray)),  diff_int);

	phoebe_debug ("Temperature nodes:\n");
	for (i = 0; i < PHOEBE_spectra_repository.Teffnodes->dim; i++)
		phoebe_debug ("\t%d\n", PHOEBE_spectra_repository.Teffnodes->val.iarray[i]);
	phoebe_debug ("Gravity nodes:\n");
	for (i = 0; i < PHOEBE_spectra_repository.loggnodes->dim; i++)
		phoebe_debug ("\t%d\n", PHOEBE_spectra_repository.loggnodes->val.iarray[i]);
	phoebe_debug ("Metallicity nodes:\n");
	for (i = 0; i < PHOEBE_spectra_repository.metnodes->dim; i++)
		phoebe_debug ("\t%d\n", PHOEBE_spectra_repository.metnodes->val.iarray[i]);

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
	/**
	 * phoebe_spectra_free_repository:
	 *
	 * Frees all fields of the spectrum repository. Since PHOEBE currently
	 * allows for a single spectrum repository, the function takes no arguments.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

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

/******************************************************************************/

PHOEBE_spectrum *phoebe_spectrum_new ()
{
	/**
	 * phoebe_spectrum_new:
	 *
	 * Initializes a new spectrum.
	 *
	 * Returns: a pointer to the initialized #PHOEBE_spectrum.
	 */

	PHOEBE_spectrum *spectrum = phoebe_malloc (sizeof (*spectrum));
	spectrum->R   = 0;
	spectrum->Rs  = 0;
	spectrum->dx  = 0;
	spectrum->disp = PHOEBE_SPECTRUM_DISPERSION_NONE;
	spectrum->data = phoebe_hist_new ();
	return spectrum;
}

PHOEBE_spectrum *phoebe_spectrum_new_from_file (char *filename)
{
	/**
	 * phoebe_spectrum_new_from_file:
	 * @filename: spectrum filename
	 *
	 * Opens a two-column @filename and reads its contents to a newly allocated
	 * spectrum. The columns are assumed to contain bin centers and fluxes. The
	 * function tries to guess the dispersion type and assigns all spectrum
	 * fields accordingly.
	 * 
	 * Returns: a read-in #PHOEBE_spectrum.
	 */

	FILE *input;
	PHOEBE_spectrum *spectrum;
	PHOEBE_vector *bin_centers;
	int linecount = 1, idx = 0;
	
	char line[255];
	char *strptr;
	double wl, flux;
	
	input = fopen (filename, "r");
	if (!input) return NULL;
	
	spectrum = phoebe_spectrum_new ();
	bin_centers = phoebe_vector_new ();
	
	while (fgets (line, 254, input)) {
		/* Remove empty lines: */
		if ( (strptr = strchr (line, '\n')) ) *strptr = '\0';
		
		/* Remove comments (if any): */
		if ( (strptr = strchr (line, '#')) ) *strptr = '\0';
		
		if (sscanf (line, "%lf %lf", &wl, &flux) == 2) {
			phoebe_vector_realloc (bin_centers, bin_centers->dim + 1);
			phoebe_spectrum_realloc (spectrum, spectrum->data->bins + 1);
			bin_centers->val[bin_centers->dim-1] = wl;
			spectrum->data->val[spectrum->data->bins-1] = flux;
		}
		else if (sscanf (line, "%lf", &flux) == 1) {
			phoebe_vector_realloc (bin_centers, bin_centers->dim + 1);
			phoebe_spectrum_realloc (spectrum, spectrum->data->bins + 1);
			bin_centers->val[bin_centers->dim-1] = 2500.0+idx;
			spectrum->data->val[spectrum->data->bins-1] = flux;
			idx += 1;
		}
		
		linecount++;
	}
	
	fclose (input);
	
	if (spectrum->data->bins == 0) {
		phoebe_spectrum_free (spectrum);
		phoebe_vector_free (bin_centers);
		return NULL;
	}
	
	phoebe_hist_set_ranges (spectrum->data, bin_centers);

	phoebe_vector_free (bin_centers);
	
	/* Guess the dispersion function: */
	phoebe_spectrum_dispersion_guess (&(spectrum->disp), spectrum);
	
	/* Guess a sampling power: */
	if (spectrum->disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR) {
		spectrum->dx = spectrum->data->range[1]-spectrum->data->range[0];
		spectrum->Rs = 0;
		spectrum->R  = 0;
	}
	else {
		spectrum->dx = 0;
		spectrum->Rs = 0.5*(spectrum->data->range[0]+spectrum->data->range[1])/(spectrum->data->range[1]-spectrum->data->range[0]);
		spectrum->R  = 0.5*(spectrum->data->range[0]+spectrum->data->range[1])/(spectrum->data->range[1]-spectrum->data->range[0]);
	}

	return spectrum;
}

PHOEBE_spectrum *phoebe_spectrum_new_from_repository (double Teff, double logg, double met)
{
	/**
	 * phoebe_spectrum_new_from_repository:
	 * @Teff: effective temperature in K
	 * @logg: surface gravity in cgs units
	 * @met:  metallicity in solar abundances
	 *
	 * Queries the spectrum repository and interpolates a spectrum with the
	 * passed parameters. If the parameters are out of range for the given
	 * repository, NULL is returned. The function uses all available nodes in
	 * the repository for the interpolation; if the queried spectrum is
	 * missing, the first adjacent spectrum is looked up. The interpolation
	 * is linear.
	 *
	 * Returns: a queried #PHOEBE_spectrum, or %NULL in case of failure.
	 */

	int i, j, k, l, m;
	int imin, imax, jmin, jmax, kmin, kmax, loop;
	int status;
	double x[3], lo[3], hi[3];
	PHOEBE_spectrum *result;
	PHOEBE_spectrum *fv[8];

	if (PHOEBE_spectra_repository.no == 0) {
		phoebe_lib_error ("there are no spectra in the repository.\n");
		return NULL;
	}

	i = j = k = 0;
	while (Teff      >= PHOEBE_spectra_repository.Teffnodes->val.iarray[i] && i < PHOEBE_spectra_repository.Teffnodes->dim) i++;
	while (10.0*logg >= PHOEBE_spectra_repository.loggnodes->val.iarray[j] && j < PHOEBE_spectra_repository.loggnodes->dim) j++;
	while (10.0*met  >= PHOEBE_spectra_repository.metnodes-> val.iarray[k] && k < PHOEBE_spectra_repository.metnodes-> dim) k++;
	i--; j--; k--;

	if (i < 0 || j < 0 || k < 0)
		return NULL;

	/* A possible shortcut: is the queried spectrum a node? */
	if (Teff == PHOEBE_spectra_repository.Teffnodes->val.iarray[i] &&
	    10.0*logg-PHOEBE_spectra_repository.loggnodes->val.iarray[j] < 1e-3 &&
	    10.0* met-PHOEBE_spectra_repository.metnodes-> val.iarray[k] < 1e-3)
			return phoebe_spectrum_new_from_file (PHOEBE_spectra_repository.table[i][j][k]->filename);

	/*
	 * Because of the gaps in parameter values we need to allow adjacent
	 * cells to be used as nodes. If these are also not available, bail out.
	 */

	loop = 0;
	while (TRUE) {
		switch (loop) {
			case  0: imin = i;   jmin = j;   kmin = k;   imax = i+1; jmax = j+1; kmax = k+1; break;
			case  1: imin = i;   jmin = j;   kmin = k;   imax = i+1; jmax = j+1; kmax = k+2; break;
			case  2: imin = i;   jmin = j;   kmin = k;   imax = i+1; jmax = j+2; kmax = k+1; break;
			case  3: imin = i;   jmin = j;   kmin = k;   imax = i+1; jmax = j+2; kmax = k+2; break;
			case  4: imin = i;   jmin = j;   kmin = k-1; imax = i+1; jmax = j+1; kmax = k+1; break;
			case  5: imin = i;   jmin = j;   kmin = k-1; imax = i+1; jmax = j+2; kmax = k+1; break;
			case  6: imin = i;   jmin = j-1; kmin = k;   imax = i+1; jmax = j+1; kmax = k+1; break;
			case  7: imin = i;   jmin = j-1; kmin = k;   imax = i+1; jmax = j+1; kmax = k+2; break;
			case  8: imin = i;   jmin = j-1; kmin = k-1; imax = i+1; jmax = j+1; kmax = k+1; break;
			case  9: imin = i;   jmin = j-1; kmin = k-1; imax = i+2; jmax = j+1; kmax = k+1; break;
			case 10: imin = i;   jmin = j;   kmin = k;   imax = i+2; jmax = j+1; kmax = k+1; break;
			case 11: imin = i;   jmin = j;   kmin = k;   imax = i+2; jmax = j+1; kmax = k+2; break;
			case 12: imin = i;   jmin = j;   kmin = k;   imax = i+2; jmax = j+2; kmax = k+1; break;
			case 13: imin = i;   jmin = j;   kmin = k;   imax = i+2; jmax = j+2; kmax = k+2; break;
			case 14: imin = i;   jmin = j;   kmin = k-1; imax = i+2; jmax = j+1; kmax = k+1; break;
			case 15: imin = i;   jmin = j;   kmin = k-1; imax = i+2; jmax = j+2; kmax = k+1; break;
			case 16: imin = i;   jmin = j-1; kmin = k;   imax = i+2; jmax = j+1; kmax = k+1; break;
			case 17: imin = i;   jmin = j-1; kmin = k;   imax = i+2; jmax = j+1; kmax = k+2; break;
			case 18: imin = i-1; jmin = j;   kmin = k;   imax = i+1; jmax = j+1; kmax = k+1; break;
			case 19: imin = i-1; jmin = j;   kmin = k;   imax = i+1; jmax = j+1; kmax = k+2; break;
			case 20: imin = i-1; jmin = j;   kmin = k;   imax = i+1; jmax = j+2; kmax = k+1; break;
			case 21: imin = i-1; jmin = j;   kmin = k;   imax = i+1; jmax = j+2; kmax = k+2; break;
			case 22: imin = i-1; jmin = j;   kmin = k-1; imax = i+1; jmax = j+1; kmax = k+1; break;
			case 23: imin = i-1; jmin = j;   kmin = k-1; imax = i+1; jmax = j+2; kmax = k+1; break;
			case 24: imin = i-1; jmin = j-1; kmin = k;   imax = i+1; jmax = j+1; kmax = k+1; break;
			case 25: imin = i-1; jmin = j-1; kmin = k;   imax = i+1; jmax = j+1; kmax = k+2; break;
			case 26: imin = i-1; jmin = j-1; kmin = k-1; imax = i+1; jmax = j+1; kmax = k+1; break;
			default:
				return NULL;
		}
		phoebe_debug ("loop %2d: testing [%d][%d][%d], [%d][%d][%d]:\n", loop, imin, jmin, kmin, imax, jmax, kmax);
		if (imin >= 0 && jmin >= 0 && kmin >= 0               &&
			imax < PHOEBE_spectra_repository.Teffnodes->dim   &&
			jmax < PHOEBE_spectra_repository.loggnodes->dim   &&
			kmax < PHOEBE_spectra_repository.metnodes->dim    &&
			PHOEBE_spectra_repository.table[imin][jmin][kmin] &&
			PHOEBE_spectra_repository.table[imin][jmin][kmax] &&
			PHOEBE_spectra_repository.table[imin][jmax][kmin] &&
			PHOEBE_spectra_repository.table[imin][jmax][kmax] &&
			PHOEBE_spectra_repository.table[imax][jmin][kmin] &&
			PHOEBE_spectra_repository.table[imax][jmin][kmax] &&
			PHOEBE_spectra_repository.table[imax][jmax][kmin] &&		
			PHOEBE_spectra_repository.table[imax][jmax][kmax])
			break;
		else loop++;
	}

	phoebe_debug ("i = %d, j = %d, k = %d\n", i, j, k);
	phoebe_debug ("T[%d] = %d, T[%d] = %d, T = %lf.\n", imin, PHOEBE_spectra_repository.Teffnodes->val.iarray[imin], imax, PHOEBE_spectra_repository.Teffnodes->val.iarray[imax], Teff);
	phoebe_debug ("l[%d] = %d, l[%d] = %d, l = %lf.\n", jmin, PHOEBE_spectra_repository.loggnodes->val.iarray[jmin], jmax, PHOEBE_spectra_repository.loggnodes->val.iarray[jmax], logg);
	phoebe_debug ("m[%d] = %d, m[%d] = %d, m = %lf.\n", kmin, PHOEBE_spectra_repository.metnodes-> val.iarray[kmin], kmax, PHOEBE_spectra_repository.metnodes-> val.iarray[kmax], met);

	/* Let's build interpolation structures: */

	x[0] = Teff;
	lo[0] = (double) PHOEBE_spectra_repository.Teffnodes->val.iarray[imin];
	hi[0] = (double) PHOEBE_spectra_repository.Teffnodes->val.iarray[imax];

	 x[1] = logg;
	lo[1] = (double) PHOEBE_spectra_repository.loggnodes->val.iarray[jmin] / 10.0;
	hi[1] = (double) PHOEBE_spectra_repository.loggnodes->val.iarray[jmax] / 10.0;

     x[2] = met;
	lo[2] = (double) PHOEBE_spectra_repository.metnodes->val.iarray[kmin] / 10.0;
	hi[2] = (double) PHOEBE_spectra_repository.metnodes->val.iarray[kmax] / 10.0;

	/* Read in the node spectra; if the readout fails, free memory and abort. */
	for (l = 0; l < 8; l++) {
		fv[l] = phoebe_spectrum_new_from_file (PHOEBE_spectra_repository.table[imin+(l%2)*(imax-imin)][jmin+((l/2)%2)*(jmax-jmin)][kmin+((l/4)%2)*(kmax-kmin)]->filename);

		if (!fv[l]) {
			PHOEBE_vector *fluxes = phoebe_vector_new_from_column (PHOEBE_spectra_repository.table[imin+(l%2)*(imax-imin)][jmin+((l/2)%2)*(jmax-jmin)][kmin+((l/4)%2)*(kmax-kmin)]->filename, 1);
			
			if (!fluxes) {
				/* We're in trouble: even column 1 could not be read. Bail out. */
				phoebe_lib_error ("spectrum %s not found, aborting.\n", PHOEBE_spectra_repository.table[imin+(l%2)*(imax-imin)][jmin+((l/2)%2)*(jmax-jmin)][kmin+((l/4)%2)*(kmax-kmin)]->filename);
				for (m = 0; m < l-1; m++)
					phoebe_spectrum_free (fv[m]);
				return NULL;
			}
			
			fv[l] = phoebe_spectrum_create (2500, 10500, 1, PHOEBE_SPECTRUM_DISPERSION_LINEAR);
			phoebe_debug ("created spectrum %d with dim %d: ", l, fv[l]->data->bins);
			
			if (!PHOEBE_spectra_repository.table[imin+(l%2)*(imax-imin)][jmin+((l/2)%2)*(jmax-jmin)][kmin+((l/4)%2)*(kmax-kmin)]) {
				for (m = 0; m < l-1; m++)
					phoebe_spectrum_free (fv[m]);
				return NULL;
			}
			
			status = phoebe_hist_set_values (fv[l]->data, fluxes);
			if (status != SUCCESS)
				phoebe_lib_error ("%s", phoebe_error (status));
			phoebe_vector_free (fluxes);
		}
	}

	/* Everything seems to be ok; proceed to the interpolation.               */
	phoebe_interpolate (3, x, lo, hi, TYPE_SPECTRUM, fv);

	/* Free all except the first spectrum: */
	for (i = 1; i < 8; i++)
		phoebe_spectrum_free (fv[i]);

	/* Assign the passed argument to the first spectrum: */
	result = fv[0];

	/* All spectra in the repository are in pixel space: */
	result->disp = PHOEBE_SPECTRUM_DISPERSION_LINEAR;

	return result;
}

PHOEBE_spectrum *phoebe_spectrum_create (double ll, double ul, double Rdx, PHOEBE_spectrum_dispersion disp)
{
	/**
	 * phoebe_spectrum_create:
	 * @ll: lower wavelength limit in angstroms
	 * @ul: upper wavelength limit in angstroms
	 * @Rdx:  resolving power for log dispersion, or dispersion for linear dispersion
	 * @disp: spectrum dispersion type
	 * 
	 * Creates an empty spectrum of resolving power or dispersion @Rdx (depending
	 * on dispersion type), sampled on the wavelength interval [@ll, @ul]. If
	 * the dispersion type is %PHOEBE_SPECTRUM_DISPERSION_LOG, the resolving
	 * power is constant and the dispersion changes throughout the spectrum. If
	 * the dispersion type is %PHOEBE_SPECTRUM_DISPERSION_LINEAR, the dispersion
	 * is constant and the resolving power changes. The R and Rs fields of the
	 * output spectrum are set for log dispersion, and the dx field is set for
	 * linear dispersion.
	 * 
	 * Returns: a newly created #PHOEBE_spectrum.
	 */
	
	int status;
	PHOEBE_spectrum *spectrum;
	int N, i;
	double q;
	
	switch (disp) {
		case PHOEBE_SPECTRUM_DISPERSION_LINEAR:
			N = (int) ((ul-ll)/Rdx+1e-6);
		break;
		case PHOEBE_SPECTRUM_DISPERSION_LOG:
			q = (2.*Rdx+1)/(2.*Rdx-1);
			N = 1 + (int) (log (ul/ll) / log (q));
		break;
		default:
			phoebe_lib_error ("unsupported spectrum dispersion type, aborting.\n");
			return NULL;
		break;
	}
	
	spectrum = phoebe_spectrum_new ();
	spectrum->disp = disp;
	
	if (disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR) {
		spectrum->dx = Rdx;
		spectrum->R  = 0;
		spectrum->Rs = 0;
	}
	else {
		spectrum->dx = 0;
		spectrum->R  = Rdx;
		spectrum->Rs = Rdx;
	}
	
	status = phoebe_spectrum_alloc (spectrum, N);
	
	if (status != SUCCESS) {
		phoebe_lib_error ("phoebe_spectrum_create: %s", phoebe_error (status));
		return NULL;
	}
	
	spectrum->data->range[0] = ll; spectrum->data->val[0] = 0.0;
	
	switch (disp) {
		case PHOEBE_SPECTRUM_DISPERSION_LINEAR:
			for (i = 1; i < N; i++) {
				spectrum->data->range[i] = spectrum->data->range[i-1] + Rdx;
				spectrum->data->val[i] = 0.0;
			}
			spectrum->data->range[N] = spectrum->data->range[N-1] + Rdx;
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
	/**
	 * phoebe_spectrum_duplicate:
	 * @spectrum: spectrum to be duplicated
	 * 
	 * Makes a duplicate copy of @spectrum.
	 *
	 * Returns: a duplicated #PHOEBE_spectrum.
	 */

	/*
	 * It is somewhat tricky to avoid
	 * memory leaks in this function: the call to phoebe_spectrum_new ()
	 * initializes an empty histogram as convenience
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

	copy->dx   = spectrum->dx;
	copy->R    = spectrum->R;
	copy->Rs   = spectrum->Rs;
	copy->disp = spectrum->disp;
	
	phoebe_hist_free (copy->data);
	copy->data = phoebe_hist_duplicate (spectrum->data);
	
	return copy;
}

PHOEBE_vector *phoebe_spectrum_get_column (PHOEBE_spectrum *spectrum, int col)
{
	/**
	 * phoebe_spectrum_get_column:
	 * @spectrum: input spectrum
	 * @col: column index
	 *
	 * Allocates a vector and copies the contents of the @col-th @spectrum
	 * column to it. The value of @col may be 1 (wavelength) or 2 (flux).
	 *
	 * Returns: #PHOEBE_vector, or #NULL in case of a failure.
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
	/**
	 * phoebe_spectrum_alloc:
	 * @spectrum: initialized spectrum
	 * @dim: spectrum dimension
	 * 
	 * Allocates memory for @spectrum with dimension @dim. The @spectrum must
	 * be initialized with phoebe_spectrum_new().
	 *
	 * Returns: #PHOEBE_error_code.
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
	/**
	 * phoebe_spectrum_realloc:
	 * @spectrum: initialized or allocated spectrum
	 * @dim: new spectrum dimension
	 * 
	 * Reallocates memory for @spectrum with the new dimension @dim. If @dim
	 * is smaller than the current dimension, @spectrum will be truncated.
	 * Otherwise all original values are retained.
	 *
	 * Returns: #PHOEBE_error_code.
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
	/**
	 * phoebe_spectrum_free:
	 * @spectrum: spectrum to be freed
	 *
	 * Frees the allocated space for @spectrum.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!spectrum) return SUCCESS;

	if (spectrum->data)
		phoebe_hist_free (spectrum->data);

	free (spectrum);

	return SUCCESS;
}

char *phoebe_spectrum_dispersion_type_get_name (PHOEBE_spectrum_dispersion disp)
{
	/**
	 * phoebe_spectrum_dispersion_type_get_name:
	 * @disp: enumerated #PHOEBE_spectrum_dispersion
	 * 
	 * Converts the enumerated value to a string.
	 *
	 * Returns: string
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

int phoebe_spectrum_dispersion_guess (PHOEBE_spectrum_dispersion *disp, PHOEBE_spectrum *spectrum)
{
	/**
	 * phoebe_spectrum_dispersion_guess:
	 * @disp: placeholder for the guessed dispersion
	 * @spectrum: spectrum for which the dispersion should be guessed
	 *
	 * Tries to guess the type of the spectrum dispersion by evaluating all
	 * bin differences and quotients. If a difference is constant, @disp is
	 * set to linear dispersion; if a quotient is constant, @disp is set to
	 * log dispersion; otherwise @disp is set to no dispersion. The constancy
	 * of the differences and quotients is checked to 1e-6 numerical accuracy.
	 *
	 * Returns: #PHOEBE_error_code.
	 */
	
	int i;
	
	*disp = PHOEBE_SPECTRUM_DISPERSION_LINEAR;
	for (i = 0; i < spectrum->data->bins-1; i++) {
		if (fabs (2*spectrum->data->range[i+1]-spectrum->data->range[i]-spectrum->data->range[i+2]) > 1e-6) {
			*disp = PHOEBE_SPECTRUM_DISPERSION_NONE;
			break;
		}
	}
	if (*disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR)
		return SUCCESS;
	
	*disp = PHOEBE_SPECTRUM_DISPERSION_LOG;
	for (i = 0; i < spectrum->data->bins-1; i++)
		if (fabs (spectrum->data->range[i+1]/spectrum->data->range[i] - spectrum->data->range[i+2]/spectrum->data->range[i+1]) > 1e-6) {
			*disp = PHOEBE_SPECTRUM_DISPERSION_NONE;
			break;
		}
	
	return SUCCESS;
}

int phoebe_spectrum_crop (PHOEBE_spectrum *spectrum, double ll, double ul)
{
	/**
	 * phoebe_spectrum_crop:
	 * @spectrum: spectrum to be cropped
	 * @ll: lower wavelength limit
	 * @ul: upper wavelength limit
	 * 
	 * Crops the passed spectrum to the [ll, ul] interval. All other spectrum
	 * properties are retained.
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	return phoebe_hist_crop (spectrum->data, ll, ul);
}

int phoebe_spectrum_apply_doppler_shift (PHOEBE_spectrum **dest, PHOEBE_spectrum *src, double velocity)
{
	/**
	 * phoebe_spectrum_apply_doppler_shift:
	 * @dest: a placeholter for the shifted spectrum
	 * @src:  original (unshifted) spectrum
	 * @velocity: radial velocity in km/s
	 * 
	 * Applies a non-relativistic Doppler shift to the spectrum. It does that
	 * by first shifting the ranges according to the passed velocity and then
	 * rebinning the obtained histogram to the original bins. c=299791 km/s is
	 * used for the speed of light.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, status;
	
	*dest = phoebe_spectrum_duplicate (src);
	
	for (i = 0; i < src->data->bins+1; i++)
		src->data->range[i] *= 1.0 + velocity / 299791.0;
	
	status = phoebe_hist_rebin ((*dest)->data, src->data, PHOEBE_HIST_CONSERVE_VALUES);
	
	return status;
}

int phoebe_spectrum_rebin (PHOEBE_spectrum **src, PHOEBE_spectrum_dispersion disp, double ll, double ul, double Rdx)
{
	/**
	 * phoebe_spectrum_rebin:
	 * src:
	 * disp:
	 * ll:
	 * ul:
	 * Rdx:
	 * 
	 * Inline rebinning of the spectrum @src. It works for both degrading and
	 * enhancing spectra. In case of failure the original spectrum is not
	 * modified.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status, i;
	PHOEBE_spectrum *dest;

	if (!*src)
		return ERROR_SPECTRUM_NOT_INITIALIZED;
	if ((*src)->data->bins == 0)
		return ERROR_SPECTRUM_NOT_ALLOCATED;
	if (disp == PHOEBE_SPECTRUM_DISPERSION_NONE)
		return ERROR_SPECTRUM_UNKNOWN_DISPERSION;
	
	/* Create a new spectrum with a given dispersion type and sampling power: */
	dest = phoebe_spectrum_create (ll, ul, Rdx, disp);
	
	/* Resample the histogram in the spectrum: */
	status = phoebe_hist_resample (dest->data, (*src)->data, PHOEBE_HIST_CONSERVE_VALUES);
	if (status != SUCCESS) {
		phoebe_spectrum_free (dest);
		return status;
	}
	
	/* If the spectrum dispersion is changed from logarithmic to linear,
	 * the continuum needs to be renormalized:
	 */
	
	if ((*src)->disp == PHOEBE_SPECTRUM_DISPERSION_LOG && disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR)
		for (i = 0; i < dest->data->bins; i++)
			dest->data->val[i] /= (*src)->R/dest->R/0.5/(dest->data->range[i]+dest->data->range[i+1]);
	
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
	
	phoebe_spectrum_rebin (&s1, PHOEBE_SPECTRUM_DISPERSION_LINEAR, ll, ul, R);
	phoebe_spectrum_rebin (&s2, PHOEBE_SPECTRUM_DISPERSION_LINEAR, ll, ul, R);
	
	*dest = phoebe_spectrum_create (ll, ul, R, PHOEBE_SPECTRUM_DISPERSION_LINEAR);
	
	for (i = 0; i < (*dest)->data->bins; i++)
		(*dest)->data->val[i] = s1->data->val[i] * s2->data->val[i];
	
	phoebe_spectrum_free (s1);
	phoebe_spectrum_free (s2);
	
	return SUCCESS;
}

int phoebe_spectra_divide (PHOEBE_spectrum **dest, PHOEBE_spectrum *src1, PHOEBE_spectrum *src2)
{
	/*
	 * This function divides the flux parts of the two spectra. It is primitive
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
			(*dest)->data->val[i] = src1->data->val[i] / src2->data->val[i];
		} else {
			phoebe_spectrum_free (*dest);
			return ERROR_SPECTRA_NOT_ALIGNED;
		}
	}
	/* And, of course, don't forget the last bin: */
	(*dest)->data->range[i] = src1->data->range[i];

	return SUCCESS;
}

bool phoebe_spectra_compare (PHOEBE_spectrum *spec1, PHOEBE_spectrum *spec2)
{
	/**
	 * phoebe_spectra_compare:
	 * @spec1: spectrum 1 to be compared
	 * @spec2: spectrum 2 to be compared
	 *
	 * Compares spectra @spec1 and @spec2 by comparing the resolving power,
	 * sampling, dispersion, and by evaluating the absolute value of
	 * the differences of all elements (both wavelength and flux) and comparing
	 * it against PHOEBE_NUMERICAL_ACCURACY.
	 * 
	 * Returns: #TRUE if the spectra are the same, #FALSE otherwise.
	 */

	if (fabs (spec1->dx - spec2->dx) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;
	if (fabs (spec1->R  - spec2->R)  > PHOEBE_NUMERICAL_ACCURACY) return FALSE;
	if (fabs (spec1->Rs - spec2->Rs) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;
	if (spec1->disp != spec2->disp) return FALSE;

	return phoebe_hist_compare (spec1->data, spec2->data);
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
	/**
	 * phoebe_spectrum_integrate:
	 * @spectrum: input spectrum
	 * @ll: lower wavelength interval limit
	 * @ul: upper wavelength interval limit
	 * @result: placeholder for the integral value
	 * 
	 * Integrates the @spectrum on the wavelength interval [@ll, @ul] using a
	 * simple rectangular rule:
	 *
	 *   I = \int_ll^ul s (\lambda) d\lambda = \sum s (\lambda) \Delta \lambda
	 * 
	 * The units of @ll and @ul must match the units of the passed @spectrum.
	 *
	 * The sum goes over all covered \Delta \lambda's, taking into account
	 * the partial coverage at interval borders.
	 *
	 * Returns: #PHOEBE_error_code.
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
	
	while (ul > spectrum->data->range[l-1]) {
		sum += (min (spectrum->data->range[l], ul) - max (ll, spectrum->data->range[l-1])) * spectrum->data->val[l-1];
		if (l == spectrum->data->bins) break; l++;
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

int phoebe_spectrum_set_dispersion (PHOEBE_spectrum *spectrum, double dx)
{
	/*
	 * This function sets the spectrum's sampling resolution.
	 */
	
	spectrum->dx = dx;

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
