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
	PHOEBE_pt_list = NULL;
	PHOEBE_pt      = phoebe_parameter_table_new ();

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
	char *pathname;
	int switch_state;

	char working_string[255];
	char *working_str = working_string;

	/* Welcome to PHOEBE! :) Let's initialize all the variables first: */
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

	/* Everything OK, let's initialize PHOEBE environment file: */
	sprintf (working_str, "%s/.phoebe2", USER_HOME_DIR);
	PHOEBE_HOME_DIR = strdup (working_str);
	sprintf (working_str, "%s/phoebe.config", PHOEBE_HOME_DIR);
	PHOEBE_CONFIG = strdup (working_str);

	/* Initialize all configuration parameters: */
	phoebe_debug ("* declaring configuration options...\n");
	phoebe_init_config_entries ();

	/*
	 * Read out the configuration file; the function also sets
	 * PHOEBE_CONFIG_EXISTS to 1 or 0 if it is found or not, respectively.
	 * If the file is not found, defaults are assumed.
	 */

	phoebe_debug ("* looking for a configuration file...\n");
	phoebe_config_load (PHOEBE_CONFIG);
	if (!PHOEBE_CONFIG_EXISTS)
		phoebe_lib_warning ("  PHOEBE configuration file not found, assuming defaults.\n");

	/* Initialize all parameters: */
	phoebe_debug ("* declaring parameters...\n");
	phoebe_init_parameters ();

	/* Read in all supported passbands and their transmission functions:      */
	phoebe_debug ("* reading in passbands:\n");
	phoebe_config_entry_get ("PHOEBE_PTF_DIR", &pathname);
	phoebe_read_in_passbands (pathname);
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
	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &switch_state);
	if (switch_state == 1) {
		phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname);
		status = read_in_ld_nodes (pathname);
		if (status != SUCCESS) {
			phoebe_lib_error ("reading LD table coefficients failed, disabling readouts.\n");
			phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 0);
		}
	}

	return SUCCESS;
}

int phoebe_quit ()
{
	/**
	 * phoebe_quit:
	 *
	 * Restores the state of the machine to pre-PHOEBE circumstances and
	 * frees all memory for an elegant exit.
	 */

	int   state;

	/* Restore the original locale of the system: */
	setlocale (LC_NUMERIC, PHOEBE_INPUT_LOCALE);

	/* Free the LD table:                                                     */
	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &state);
	if (state) {
		phoebe_ld_table_free ();
	}

	/* Free all global PHOEBE strings:                                        */
	free (PHOEBE_STARTUP_DIR);
	free (PHOEBE_INPUT_LOCALE);
	free (USER_HOME_DIR);
	free (PHOEBE_HOME_DIR);
	free (PHOEBE_CONFIG);
	free (PHOEBE_PLOTTING_PACKAGE);
	free (PHOEBE_VERSION_NUMBER);
	free (PHOEBE_VERSION_DATE);
	free (PHOEBE_PARAMETERS_FILENAME);

	/* Free passband list: */
	phoebe_free_passbands ();

	/* Free constraints: */
	phoebe_free_constraints ();

	/* Free parameters and their options: */
	phoebe_free_parameters ();

	/* Free parameter table: */
	phoebe_parameter_table_free (PHOEBE_pt);

	exit (0);
}
