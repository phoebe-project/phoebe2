#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

#include "phoebe_accessories.h"
#include "phoebe_configuration.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
#include "phoebe_ld.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

PHOEBE_passband **PHOEBE_passbands;
int               PHOEBE_passbands_no;

int intern_compare_passbands_by_effwl (const void *a, const void *b)
{
	/*
	 * This function is needed by qsort function that sorts passbands by 
	 * effective wavelength.
	 */

	PHOEBE_passband * const *ia = a;
	PHOEBE_passband * const *ib = b;

	return (int) ((*ia)->effwl - (*ib)->effwl);
} 

int intern_compare_passbands_by_set (const void *a, const void *b)
{
	/*
	 * This function is needed by qsort function that sorts passbands by set.
	 */
 
	PHOEBE_passband * const *ia = a;
	PHOEBE_passband * const *ib = b;

	return strcmp ((*ia)->set, (*ib)->set);
}

PHOEBE_passband *phoebe_passband_new ()
{
	/**
	 * phoebe_passband_new:
	 *
	 * Initializes memory for a new #PHOEBE_passband and sets all fields to
	 * #NULL.
	 *
	 * Returns: #PHOEBE_passband.
	 */

	PHOEBE_passband *passband = phoebe_malloc (sizeof (*passband));

	passband->id    = 0;
	passband->set   = NULL;
	passband->name  = NULL;
	passband->effwl = 0.0;
	passband->tf    = NULL;
	passband->ld    = NULL;

	return passband;
}

PHOEBE_passband *phoebe_passband_new_from_file (char *filename)
{
	/**
	 * phoebe_passband_new_from_file:
	 * @filename: passband transmission function file
	 *
	 * Reads in passband transmission function (PTF) from the passed @filename.
	 *
	 * Passband transmission functions are installed in a directory the path
	 * to which is stored in a #PHOEBE_PTF_DIR variable. In order to be able
	 * to pass a relative filename to phoebe_hist_new_from_file(), this
	 * wrapper does the following, sequentially:
	 *
	 * 1) if an absolute path is passed, use it;
	 * 2) if there is a file with that name in the current directory, use it;
	 * 3) if there is a file with that name in PHOEBE_PTF_DIR, use it;
	 * 4) admit defeat and return NULL.
	 *
	 * Returns: #PHOEBE_passband on success, #NULL on failure.
	 */

	FILE *ptf_file;

	PHOEBE_passband *passband;
	char *full_filename = NULL;

	char *ptfdir;
	char line[255], keyword[255];
	char *ptr;

	int bins = 0;
	double *wla = NULL, *tfa = NULL;
	double wl, tf;
	double wlfactor = 1.0;                 /* Default wavelength factor value */

	/* If the absolute path is given, use it as is or bail out. */
	if (filename[0] == '/') {
		if (!(ptf_file = fopen (filename, "r")))
			return NULL;
	}

	/* If a relative path is given, try the current working directory; if it  */
	/* is still not found, try the passbands directory. If still it can't be  */
	/* found, bail out.                                                       */
	else {
		ptf_file = fopen (filename, "r");

		if (!ptf_file) {
			phoebe_config_entry_get ("PHOEBE_PTF_DIR", &ptfdir);
			full_filename = phoebe_concatenate_strings (ptfdir, "/", filename, NULL);
			ptf_file = fopen (full_filename, "r");
			
			if (!ptf_file) return NULL;
		}
	}

	/* By now either the file exists or NULL has been returned. Let's parse   */
	/* the header.                                                            */

	passband = phoebe_passband_new ();

	while (fgets (line, 255, ptf_file)) {
		line[strlen(line)-1] = '\0';
		if (strchr (line, '#')) {
			/* This can be either be a comment or a header entry. */
			if (sscanf (line, "# %s", &(keyword[0])) != 1) continue;
			if (strcmp (keyword, "PASS_SET") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("PASS_SET encountered: %s\n", ptr);
				passband->set = strdup (ptr);
				continue;
			}
			if (strcmp (keyword, "PASSBAND") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("PASSBAND encountered: %s\n", ptr);
				passband->name = strdup (ptr);
				continue;
			}
			if (strcmp (keyword, "EFFWL") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("EFFWL encountered: %lf\n", atof (ptr));
				passband->effwl = atof (ptr);
				continue;
			}
			if (strcmp (keyword, "WLFACTOR") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("WLFACTOR encountered: %lf\n", atof (ptr));
				wlfactor = atof (ptr);
				continue;
			}
			/* It's just a comment, ignore it. */
			continue;
		}
		else {
			if (sscanf (line, "%lf\t%lf", &wl, &tf) != 2) continue;
			bins++;
			wla = phoebe_realloc (wla, bins * sizeof (*wla));
			tfa = phoebe_realloc (tfa, bins * sizeof (*tfa));
			wla[bins-1] = wlfactor*wl; tfa[bins-1] = tf;
		}
	}

	fclose (ptf_file);

	if (bins == 0 || !passband->name) {
		phoebe_passband_free (passband);
		passband = NULL;
	}
	else {
		passband->effwl *= wlfactor;
		passband->tf = phoebe_hist_new_from_arrays (bins, wla, tfa);
		free (wla); free (tfa);

		/* Finally, issue warnings for missing header information: */
		if (!passband->set)
			phoebe_lib_warning ("passband set not found for passband %s.\n", passband->name);
		if (!passband->effwl)
			phoebe_lib_warning ("effective wavelength not found for passband %s.\n", passband->name);
	}

	if (full_filename)
		free (full_filename);

	return passband;
}

int phoebe_read_in_passbands (char *dir_name)
{
	/**
	 * phoebe_read_in_passbands:
	 * @dir_name: directory where passband transmission functions (PTFs) are
	 * stored
	 *
	 * Opens the @dir_name directory, scans all files in that directory and
	 * reads in all found passbands. Finally, it sorts them first by effective
	 * wavelength and then by set name.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	DIR *ptf_dir;
	struct dirent *ptf_file;
	char filename[255];

	int status;

	PHOEBE_passband *passband;

	status = phoebe_open_directory (&ptf_dir, dir_name);
	if (status != SUCCESS)
		return status;

	while ( (ptf_file = readdir (ptf_dir)) ) {
		sprintf (filename, "%s/%s", dir_name, ptf_file->d_name);

		if (phoebe_filename_is_directory (filename)) continue;

		passband = phoebe_passband_new_from_file (filename);
		if (!passband)
			phoebe_debug ("File %s skipped.\n", filename);
		else {
			PHOEBE_passbands_no++;
			PHOEBE_passbands = phoebe_realloc (PHOEBE_passbands, PHOEBE_passbands_no * sizeof (*PHOEBE_passbands));
			PHOEBE_passbands[PHOEBE_passbands_no-1] = passband;
		}
	}

	qsort(PHOEBE_passbands, PHOEBE_passbands_no, sizeof(*PHOEBE_passbands), intern_compare_passbands_by_effwl);
	qsort(PHOEBE_passbands, PHOEBE_passbands_no, sizeof(*PHOEBE_passbands), intern_compare_passbands_by_set);

	phoebe_close_directory (&ptf_dir);

	return SUCCESS;
}

PHOEBE_passband *phoebe_passband_lookup (const char *name)
{
	/**
	 * phoebe_passband_lookup:
	 * @name: passband name, of the form PASS_SET:PASSBAND
	 *
	 * Traverses the global passband table PHOEBE_passbands and returns a
	 * pointer to the passband that matches the passed name. The name is
	 * constructed by PASS_SET:PASSBAND, i.e. "Johnson:V".
	 *
	 * Returns: #PHOEBE_passband on success, or #NULL if @name is not matched.
	 */

	int i;
	char *band = strchr (name, ':');
	char *set;

	if (!band)
		return NULL;

	band++;
	set = strdup (name);
	set[strlen(name)-strlen(band)-1] = '\0';

	for (i = 0; i < PHOEBE_passbands_no; i++)
		if (strcmp (PHOEBE_passbands[i]->set, set) == 0 && strcmp (PHOEBE_passbands[i]->name, band) == 0) {
			free (set);
			return PHOEBE_passbands[i];
		}

	free (set);
	return NULL;
}

PHOEBE_passband *phoebe_passband_lookup_by_id (const char *id)
{
	/**
	 * phoebe_passband_lookup_by_id:
	 * @id: curve ID (light or RV curve)
	 *
	 * Looks up the passband that corresponds to the curve @id.
	 *
	 * Returns: #PHOEBE_passband on success, or #NULL if @id is not matched.
	 */

	int i;
	int lcno, rvno;
	char *cid, *filter;
	PHOEBE_passband *passband;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	for (i = 0; i < lcno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_id"), i, &cid);
		if (strcmp (cid, id) == 0) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), i, &filter);
			passband = phoebe_passband_lookup (filter);
			return passband;
		}
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);
	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_id"), i, &cid);
		if (strcmp (cid, id) == 0) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), i, &filter);
			passband = phoebe_passband_lookup (filter);
			return passband;
		}
	}

	return NULL;
}

int phoebe_passband_free (PHOEBE_passband *passband)
{
	/**
	 * phoebe_passband_free:
	 * @passband: #PHOEBE_passband to be freed
	 * 
	 * Frees @passband memory.
	 */

	if (!passband)
		return SUCCESS;

	free (passband->set);
	free (passband->name);
	phoebe_hist_free (passband->tf);
	phoebe_ld_free (passband->ld);
	free (passband);

	return SUCCESS;
}

int phoebe_free_passbands ()
{
	/**
	 * phoebe_free_passbands:
	 *
	 * Traverses all defined passbands and frees them.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	for (i = 0; i < PHOEBE_passbands_no; i++)
		phoebe_passband_free (PHOEBE_passbands[i]);
	free (PHOEBE_passbands);

	return SUCCESS;
}

int wd_passband_id_lookup (int *id, const char *passband)
{
	/**
	 * wd_passband_id_lookup:
	 * @id: WD passband ID
	 * @passband: PHOEBE passband name
	 * 
	 * Looks up the WD ID of the passband. If the passband is not set or if it
	 * is not supported by WD, ERROR_PASSBAND_INVALID is returned.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (strcmp (passband,    "Stromgren:u") == 0) { *id =  1; return SUCCESS; }
	if (strcmp (passband,    "Stromgren:v") == 0) { *id =  2; return SUCCESS; }
	if (strcmp (passband,    "Stromgren:b") == 0) { *id =  3; return SUCCESS; }
	if (strcmp (passband,    "Stromgren:y") == 0) { *id =  4; return SUCCESS; }
	if (strcmp (passband,      "Johnson:U") == 0) { *id =  5; return SUCCESS; }
	if (strcmp (passband,      "Johnson:B") == 0) { *id =  6; return SUCCESS; }
	if (strcmp (passband,      "Johnson:V") == 0) { *id =  7; return SUCCESS; }
	if (strcmp (passband,      "Johnson:R") == 0) { *id =  8; return SUCCESS; }
	if (strcmp (passband,      "Johnson:I") == 0) { *id =  9; return SUCCESS; }
	if (strcmp (passband,      "Johnson:J") == 0) { *id = 10; return SUCCESS; }
	if (strcmp (passband,      "Johnson:H") == 0) { *id = 28; return SUCCESS; }
	if (strcmp (passband,      "Johnson:K") == 0) { *id = 11; return SUCCESS; }
	if (strcmp (passband,      "Johnson:L") == 0) { *id = 12; return SUCCESS; }
	if (strcmp (passband,      "Johnson:M") == 0) { *id = 13; return SUCCESS; }
	if (strcmp (passband,      "Johnson:N") == 0) { *id = 14; return SUCCESS; }
	if (strcmp (passband,      "Cousins:R") == 0) { *id = 15; return SUCCESS; }
	if (strcmp (passband,      "Cousins:I") == 0) { *id = 16; return SUCCESS; }
	/*                                230           *id = 17;                 */
	/*                                250           *id = 18;                 */
	/*                                270           *id = 19;                 */
	/*                                290           *id = 20;                 */
	/*                                310           *id = 21;                 */
	/*                                330           *id = 22;                 */
	if (strcmp (passband,   "Hipparcos:BT") == 0) { *id = 23; return SUCCESS; }
	if (strcmp (passband,   "Hipparcos:VT") == 0) { *id = 24; return SUCCESS; }
	if (strcmp (passband,   "Hipparcos:Hp") == 0) { *id = 25; return SUCCESS; }
	if (strcmp (passband,      "CoRoT:exo") == 0) { *id = 26; return SUCCESS; }
	if (strcmp (passband,    "CoRoT:sismo") == 0) { *id = 27; return SUCCESS; }
	if (strcmp (passband,      "Johnson:H") == 0) { *id = 28; return SUCCESS; }
	if (strcmp (passband,       "Geneva:U") == 0) { *id = 29; return SUCCESS; }
	if (strcmp (passband,       "Geneva:B") == 0) { *id = 30; return SUCCESS; }
	if (strcmp (passband,      "Geneva:B1") == 0) { *id = 31; return SUCCESS; }
	if (strcmp (passband,      "Geneva:B2") == 0) { *id = 32; return SUCCESS; }
	if (strcmp (passband,       "Geneva:V") == 0) { *id = 33; return SUCCESS; }
	if (strcmp (passband,      "Geneva:V1") == 0) { *id = 34; return SUCCESS; }
	if (strcmp (passband,       "Geneva:G") == 0) { *id = 35; return SUCCESS; }
	if (strcmp (passband,    "Kepler:mean") == 0) { *id = 36; return SUCCESS; }
	if (strcmp (passband,       "Sloan:u'") == 0) { *id = 37; return SUCCESS; }
	if (strcmp (passband,       "Sloan:g'") == 0) { *id = 38; return SUCCESS; }
	if (strcmp (passband,       "Sloan:r'") == 0) { *id = 39; return SUCCESS; }
	if (strcmp (passband,       "Sloan:i'") == 0) { *id = 40; return SUCCESS; }
	if (strcmp (passband,       "Sloan:z'") == 0) { *id = 41; return SUCCESS; }
	if (strcmp (passband,         "LSST:u") == 0) { *id = 42; return SUCCESS; }
	if (strcmp (passband,         "LSST:g") == 0) { *id = 43; return SUCCESS; }
	if (strcmp (passband,         "LSST:r") == 0) { *id = 44; return SUCCESS; }
	if (strcmp (passband,         "LSST:i") == 0) { *id = 45; return SUCCESS; }
	if (strcmp (passband,         "LSST:z") == 0) { *id = 46; return SUCCESS; }
	if (strcmp (passband,        "LSST:y3") == 0) { *id = 47; return SUCCESS; }
	if (strcmp (passband,        "LSST:y4") == 0) { *id = 48; return SUCCESS; }

	*id = -1;
	return ERROR_PASSBAND_INVALID;
}

PHOEBE_curve *phoebe_bin_data (PHOEBE_curve *data, int bins)
{
	/**
	 * phoebe_bin_data:
	 * @data: original data to be binned
	 * @bins: number of bins
	 * 
	 * 
	 */
	
	int i, j, culled;
	PHOEBE_curve *input, *binned;
	
	if (!data)
		return ERROR_CURVE_NOT_INITIALIZED;
	
	input = phoebe_curve_duplicate (data);
	
	if (input->itype == PHOEBE_COLUMN_HJD)
		phoebe_curve_transform (input, PHOEBE_COLUMN_PHASE, input->dtype, input->wtype);

	binned = phoebe_curve_new ();
	phoebe_curve_alloc (binned, bins);
	phoebe_curve_set_properties (binned, input->type, input->filename, input->passband, PHOEBE_COLUMN_PHASE, input->dtype, PHOEBE_COLUMN_WEIGHT, input->sigma);
	
	for (i = 0; i < bins; i++) {
		binned->indep->val[i] = -0.5 + (double)i/(bins-1);
		binned->dep->val[i] = 0.0;
		binned->weight->val[i] = 0.0;
		binned->flag->val.iarray[i] = PHOEBE_DATA_REGULAR;
	}

	for (i = 0; i < input->indep->dim; i++) {
		binned->dep->val[(int)(bins*(0.5+input->indep->val[i]))] += input->dep->val[i];
		binned->weight->val[(int)(bins*(0.5+input->indep->val[i]))] += 1;
	}

	phoebe_curve_free (input);

	culled = 0;
	for (i = 0; i < bins-culled; i++) {
		if (binned->weight->val[i] > 0.5)
			binned->dep->val[i] /= binned->weight->val[i];
		else {
			culled += 1;
			for (j = i; j < bins-culled; j++) {
				binned->indep->val[j]       = binned->indep->val[j+1];
				binned->dep->val[j]         = binned->dep->val[j+1];
				binned->weight->val[j]      = binned->weight->val[j+1];
				binned->flag->val.iarray[j] = binned->flag->val.iarray[j+1];
			}
			i--;
		}
	}

	if (culled != 0)
		phoebe_curve_realloc (binned, bins-culled);

	return binned;
}
