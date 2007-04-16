#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

#include "phoebe_accessories.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
#include "phoebe_types.h"

PHOEBE_passband **PHOEBE_passbands;
int               PHOEBE_passbands_no;

PHOEBE_passband *phoebe_passband_new ()
{
	/*
	 * This function allocates memory for a passband record.
	 */

	PHOEBE_passband *passband = phoebe_malloc (sizeof (*passband));

	passband->id    = 0;
	passband->set   = NULL;
	passband->name  = NULL;
	passband->effwl = 0.0;
	passband->tf    = NULL;

	return passband;
}

PHOEBE_passband *phoebe_passband_new_from_file (char *filename)
{
	/*
	 * Passband transmission functions are installed in a directory the path
	 * to which is stored in a PHOEBE_PTF_DIR variable. In order to be able
	 * to pass a relative filename to phoebe_hist_new_from_file (), this
	 * wrapper does the following, sequentially:
	 *
	 * 1) if an absolute path is passed, use it; otherwise return NULL;
	 * 2) if there is a file with that name in the current directory, use it;
	 * 3) if there is a file with that name in PHOEBE_PTF_DIR, use it;
	 * 4) admit defeat and return NULL.
	 */

	FILE *ptf_file;

	PHOEBE_passband *passband;
	char *full_filename = NULL;

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
			full_filename = concatenate_strings (PHOEBE_PTF_DIR, "/", filename, NULL);
			ptf_file = fopen (full_filename, "r");
			
			if (!ptf_file) return NULL;
		}
	}

	/* By now either the file exists or NULL has been returned. Let's parse   */
	/* the header.                                                            */

	passband = phoebe_passband_new ();

	while (!feof (ptf_file)) {
		fgets (line, 255, ptf_file);
		if (feof (ptf_file)) break;
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

	if (bins == 0 || !passband->name) {
		phoebe_passband_free (passband);
		passband = NULL;
	}
	else {
		passband->effwl *= wlfactor;
		passband->tf = phoebe_hist_new_from_arrays (bins, wla, tfa);
		free (wla); free (tfa);

		/* Finally, issue warnings for missing header information:            */
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
	/*
	 * This function opens the 'dir_name' directory, scans all files in that
	 * directory and reads in all found passbands.
	 */

	DIR *ptf_dir;
	struct dirent *ptf_file;
	char filename[255];

	int status;

	PHOEBE_passband *passband;

	status = phoebe_open_directory (&ptf_dir, dir_name);
	if (status != SUCCESS)
		return status;

	while (ptf_file = readdir (ptf_dir)) {
		sprintf (filename, "%s/%s", dir_name, ptf_file->d_name);

		if (filename_is_directory (filename)) continue;

		passband = phoebe_passband_new_from_file (filename);
		if (!passband)
			phoebe_debug ("File %s skipped.\n", filename);
		else {
			PHOEBE_passbands_no++;
			PHOEBE_passbands = phoebe_realloc (PHOEBE_passbands, PHOEBE_passbands_no * sizeof (*PHOEBE_passbands));
			PHOEBE_passbands[PHOEBE_passbands_no-1] = passband;
		}
	}

	phoebe_close_directory (&ptf_dir);

	return SUCCESS;
}

PHOEBE_passband *phoebe_passband_lookup (const char *name)
{
	/*
	 * This function traverses the global passband table PHOEBE_passbands
	 * and returns a pointer to the passband that matches the passed name.
	 * If it is not found, NULL is returned.
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

int phoebe_passband_free (PHOEBE_passband *passband)
{
	/*
	 * This function frees memory occupied by the passband record.
	 */

	free (passband->set);
	free (passband->name);
	phoebe_hist_free (passband->tf);
	free (passband);

	return SUCCESS;
}
