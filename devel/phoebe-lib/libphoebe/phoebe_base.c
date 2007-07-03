#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "phoebe_build_config.h"
#include "phoebe_global.h"

#include "phoebe_accessories.h"
#include "phoebe_configuration.h"
#include "phoebe_constraints.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fitting.h"
#include "phoebe_ld.h"
#include "phoebe_parameters.h"

#ifdef HAVE_LIBGSL
	#ifndef PHOEBE_GSL_DISABLED
		/* This part initializes PHOEBE randomizer, but only if GSL is present.   */
		#include <gsl/gsl_rng.h>
	#endif
#endif

bool PHOEBE_INTERRUPT;

int intern_phoebe_variables_init ()
{
	/*
	 * This function initializes all global literal strings, so that they may
	 * be used fearlessly by other functions.
	 */

	int i;

	/*
	 * First things first: let's initialize PHOEBE version number and date; we
	 * get the values from configure.ac, so this is done automatically.
	 */

	phoebe_debug ("  setting version number and version date.\n");
	PHOEBE_VERSION_NUMBER = strdup (PHOEBE_RELEASE_NAME);
	PHOEBE_VERSION_DATE   = strdup (PHOEBE_RELEASE_DATE);

	/*
	 * Let's declare a global parameter filename string to "Undefined". This is
	 * needed for command line parameter filename loading.
	 */

	PHOEBE_PARAMETERS_FILENAME = strdup ("Undefined");

	/* Initialize the hashed parameter table: */

	PHOEBE_pt = phoebe_malloc (sizeof (*PHOEBE_pt));
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++)
		PHOEBE_pt->bucket[i] = NULL;

	/* Initialize the linked list of constraints: */
	PHOEBE_ct = NULL;

	/*
	 * The following are global parameter variables. Since they will be dynami-
	 * cally stored by phoebe_realloc, we need to set it to NULL.
	 */

	PHOEBE_passbands_no  = 0;
	PHOEBE_passbands     = NULL;

	return SUCCESS;
}

void intern_phoebe_sigint_handler (int signum)
{
	/*
	 * This is an alternate handler for the interrupt (CTRL+C) handler. Since
	 * the interrupt handler by default breaks the program itself rather than
	 * the process it executes, we need to catch the signal here and block it.
	 */

	phoebe_debug ("sigint blocked!\n");
	PHOEBE_INTERRUPT = TRUE;
	phoebe_lib_error ("break called; exitting PHOEBE.\n\n");
	exit (0);
	return;
}


int phoebe_init ()
{
	/* This function initializes all core parameters.                         */

	int status;

	char working_string[255];
	char *working_str = working_string;

	char keyword_string[255];
	char *keyword_str = keyword_string;

	FILE *config_file;

	/* Welcome to PHOEBE! :) Let's initialize all the variables first:        */
	intern_phoebe_variables_init ();

	/*
	 * Assign a current directory (i.e. the directory from which PHOEBE was
	 * started) to PHOEBE_STARTUP_DIR; this is used for resolving relative
	 * pathnames.
	 */

	getcwd (working_str, 255);
	PHOEBE_STARTUP_DIR = strdup (working_str);

	/*
	 * Get the current locale decimal point, store it to the global variable,
	 * change it to "C" and change it back to original upon exit.
	 */

	PHOEBE_INPUT_LOCALE = strdup (setlocale (LC_NUMERIC, NULL));
	setlocale (LC_NUMERIC, "C");

	if (!getenv ("HOME"))
		return ERROR_HOME_ENV_NOT_DEFINED;
	USER_HOME_DIR = strdup (getenv ("HOME"));

	/*
	 * Although it sounds silly, let's check full permissions of the home
	 * directory:
	 */

	if (!filename_has_full_permissions (USER_HOME_DIR))
		return ERROR_HOME_HAS_NO_PERMISSIONS;

	/* Everything OK, let's initialize PHOEBE environment file:               */
	sprintf (working_str, "%s/.phoebe2", USER_HOME_DIR);
	PHOEBE_HOME_DIR = strdup (working_str);
	sprintf (working_str, "%s/phoebe.config", PHOEBE_HOME_DIR);
	PHOEBE_CONFIG = strdup (working_str);

	/* Let's presume there's no config file. If there is, we'll set it to 1:  */
	PHOEBE_CONFIG_EXISTS = 0;

	/* Read out the configuration from the config file.                       */
	if (filename_exists (PHOEBE_CONFIG)) {
		PHOEBE_CONFIG_EXISTS = 1;
		config_file = fopen (PHOEBE_CONFIG, "r");

		while (!feof (config_file)) {
			/*
			 * The following line reads the line from the input file and checks
			 * if everything went OK; if en error occured (e.g. EoF reached),
			 * it breaks the loop.
			 */

			if (!fgets (keyword_str, 254, config_file)) break;

			if (strstr (keyword_str, "PHOEBE_BASE_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_BASE_DIR %s", working_str) != 1) PHOEBE_BASE_DIR = strdup ("");
				else PHOEBE_BASE_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_SOURCE_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_SOURCE_DIR %s", working_str) != 1) PHOEBE_SOURCE_DIR = strdup ("");
				else PHOEBE_SOURCE_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_DEFAULTS_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_DEFAULTS_DIR %s", working_str) != 1) PHOEBE_DEFAULTS_DIR = strdup ("");
				else PHOEBE_DEFAULTS_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_TEMP_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_TEMP_DIR %s", working_str) != 1) PHOEBE_TEMP_DIR = strdup ("");
				else PHOEBE_TEMP_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_DATA_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_DATA_DIR %s", working_str) != 1) PHOEBE_DATA_DIR = strdup ("");
				else PHOEBE_DATA_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_PTF_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_PTF_DIR %s", working_str) != 1) PHOEBE_PTF_DIR = strdup ("");
				else PHOEBE_PTF_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_PLOTTING_PACKAGE")) {
				if (sscanf (keyword_str, "PHOEBE_PLOTTING_PACKAGE %s", working_str) != 1) PHOEBE_PLOTTING_PACKAGE = strdup ("");
				else PHOEBE_PLOTTING_PACKAGE = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_LD_SWITCH"))
				if (sscanf (keyword_str, "PHOEBE_LD_SWITCH %d", &PHOEBE_LD_SWITCH) != 1) PHOEBE_LD_SWITCH = 0;
			if (strstr (keyword_str, "PHOEBE_LD_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_LD_DIR %s", working_str) != 1) PHOEBE_LD_DIR = strdup ("");
				else PHOEBE_LD_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_KURUCZ_SWITCH"))
				if (sscanf (keyword_str, "PHOEBE_KURUCZ_SWITCH %d", &PHOEBE_KURUCZ_SWITCH) != 1) PHOEBE_KURUCZ_SWITCH = 0;
			if (strstr (keyword_str, "PHOEBE_KURUCZ_DIR")) {
				if (sscanf (keyword_str, "PHOEBE_KURUCZ_DIR %s", working_str) != 1) PHOEBE_KURUCZ_DIR = strdup ("");
				else PHOEBE_KURUCZ_DIR = strdup (working_str);
			}
			if (strstr (keyword_str, "PHOEBE_3D_PLOT_CALLBACK_OPTION"))
				if (sscanf (keyword_str, "PHOEBE_3D_PLOT_CALLBACK_OPTION %d", &PHOEBE_3D_PLOT_CALLBACK_OPTION) != 1) PHOEBE_3D_PLOT_CALLBACK_OPTION = 0;
			if (strstr (keyword_str, "PHOEBE_CONFIRM_ON_SAVE"))
				if (sscanf (keyword_str, "PHOEBE_CONFIRM_ON_SAVE %d", &PHOEBE_CONFIRM_ON_SAVE) != 1) PHOEBE_CONFIRM_ON_SAVE = 1;
			if (strstr (keyword_str, "PHOEBE_CONFIRM_ON_QUIT"))
				if (sscanf (keyword_str, "PHOEBE_CONFIRM_ON_QUIT %d", &PHOEBE_CONFIRM_ON_QUIT) != 1) PHOEBE_CONFIRM_ON_QUIT = 1;
			if (strstr (keyword_str, "PHOEBE_WARN_ON_SYNTHETIC_SCATTER"))
				if (sscanf (keyword_str, "PHOEBE_WARN_ON_SYNTHETIC_SCATTER %d", &PHOEBE_WARN_ON_SYNTHETIC_SCATTER) != 1) PHOEBE_WARN_ON_SYNTHETIC_SCATTER = 1;
		}
		fclose (config_file);
	}

	if (!filename_exists (PHOEBE_CONFIG)) {
		phoebe_create_configuration_file ();
	}

	/*
	 * All PHOEBE parameters are referenced globally, so we have to initialize
	 * them before they could be used:
	 */

	phoebe_debug ("* declaring parameters...\n");
	phoebe_init_parameters ();

	/* Read in all supported passbands and their transmission functions:      */
	phoebe_debug ("* reading in passbands:\n");
	phoebe_read_in_passbands (PHOEBE_PTF_DIR);
	phoebe_debug ("  %d passbands read in.\n", PHOEBE_passbands_no);

	/* Add options to all KIND_MENU parameters:                               */
	phoebe_init_parameter_options ();

	/* Choose a randomizer seed:                                              */
	srand (time (0));

	/*
	 * Catch the interrupt (CTRL+C) signal and use an alternative handler
	 * that will break a running script rather than the scripter itself:
	 */

	signal (SIGINT, intern_phoebe_sigint_handler);

	/* Set the interrupt state to false:                                      */
	PHOEBE_INTERRUPT = FALSE;

	/* If LD tables are present, do the readout:                              */
	if (PHOEBE_LD_SWITCH == 1) {
		status = read_in_ld_nodes (PHOEBE_LD_DIR);
		if (status != SUCCESS) {
			phoebe_lib_error ("reading LD table coefficients failed, disabling readouts.\n");
		PHOEBE_LD_SWITCH = 0;
		}
	}

	/* Initialize PHOEBE randomizer:                                            */
	#ifdef HAVE_LIBGSL
		#ifndef PHOEBE_GSL_DISABLED
			PHOEBE_randomizer = gsl_rng_alloc (gsl_rng_mt19937);
			gsl_rng_set (PHOEBE_randomizer, time (0));
		#endif
	#endif

	return SUCCESS;
}

int phoebe_quit ()
{
	/*
	 * This function restores the state of the machine to pre-PHOEBE circum-
	 * stances and frees all memory for an elegant exit.
	 */

	/* Restore the original locale of the system:                             */
	setlocale (LC_NUMERIC, PHOEBE_INPUT_LOCALE);

	/* Free the LD table:                                                     */
	if (PHOEBE_LD_SWITCH == 1) {
		phoebe_ld_table_free ();
	}

	/* Free all global PHOEBE strings:                                        */
	free (PHOEBE_STARTUP_DIR);
	free (PHOEBE_INPUT_LOCALE);
	free (USER_HOME_DIR);
	free (PHOEBE_HOME_DIR);
	free (PHOEBE_CONFIG);
	free (PHOEBE_BASE_DIR);
	free (PHOEBE_SOURCE_DIR);
	free (PHOEBE_DEFAULTS_DIR);
	free (PHOEBE_TEMP_DIR);
	free (PHOEBE_DATA_DIR);
	free (PHOEBE_PTF_DIR);
	free (PHOEBE_PLOTTING_PACKAGE);
	free (PHOEBE_LD_DIR);
	free (PHOEBE_KURUCZ_DIR);
	free (PHOEBE_VERSION_NUMBER);
	free (PHOEBE_VERSION_DATE);
	free (PHOEBE_PARAMETERS_FILENAME);

	/* Free parameters and their options:                                     */
	phoebe_free_parameters ();

	/* Free passband list: */
	phoebe_free_passbands ();

	/* Free constraints: */
	phoebe_free_constraints ();

	#ifdef HAVE_LIBGSL
		#ifndef PHOEBE_GSL_DISABLED
			gsl_rng_free (PHOEBE_randomizer);
		#endif
	#endif

	exit (0);
}
