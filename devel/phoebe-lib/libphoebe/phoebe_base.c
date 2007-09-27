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

	/* First things first: let's initialize PHOEBE version number and date;   */
	/* we get the values from configure.ac, so this is done automatically.    */

	phoebe_debug ("  setting version number and version date.\n");
	PHOEBE_VERSION_NUMBER = strdup (PHOEBE_RELEASE_NAME);
	PHOEBE_VERSION_DATE   = strdup (PHOEBE_RELEASE_DATE);

	/* Initialize configuration table: */
	PHOEBE_config_table = NULL;
	PHOEBE_config_table_size = 0;

	/* Initialize configuration directory name: */
	PHOEBE_HOME_DIR = NULL;

	/* Let's declare a global parameter filename string to "Undefined". This  */
	/* is needed for command line parameter filename loading.                 */

	PHOEBE_PARAMETERS_FILENAME = strdup ("Undefined");

	/* Initialize the hashed parameter table: */
	PHOEBE_pt_list = NULL;
	PHOEBE_pt      = phoebe_parameter_table_new ();

	/* The following are global parameter variables. Since they will be dyna- */
	/* mically stored by phoebe_realloc, we need to set it to NULL.           */

	PHOEBE_passbands_no  = 0;
	PHOEBE_passbands     = NULL;

	PHOEBE_ld_table_size = 0;
	PHOEBE_ld_table      = NULL;

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

	phoebe_debug ("Welcome to PHOEBE-lib debugger! :)");

	phoebe_debug ("* initialize the variables...\n");
	intern_phoebe_variables_init ();

	/*
	 * Get the current locale decimal point, store it to the global variable,
	 * change it to "C" and change it back to original upon exit.
	 */

	PHOEBE_INPUT_LOCALE = strdup (setlocale (LC_NUMERIC, NULL));
	setlocale (LC_NUMERIC, "C");

	if (!getenv ("HOME"))
		return ERROR_HOME_ENV_NOT_DEFINED;
	USER_HOME_DIR = strdup (getenv ("HOME"));

	/* Choose a randomizer seed: */
	srand (time (0));

	/*
	 * Catch the interrupt (CTRL+C) signal and use an alternative handler
	 * that will break a running script rather than the scripter itself:
	 */

	signal (SIGINT, intern_phoebe_sigint_handler);
	PHOEBE_INTERRUPT = FALSE;

	return SUCCESS;
}

int phoebe_configure ()
{
	/**
	 * phoebe_configure:
	 *
	 * Probes the existence of the configuration file, loads it and reads in
	 * the configuration entries. The configuration filename is hardcoded to
	 * phoebe.config (this should be changed in future), but the configuration
	 * directory PHOEBE_HOME_DIR can be set by the drivers.
	 *
	 * The order of the checked config directories is:
	 *
	 * 1) PHOEBE_HOME_DIR if not NULL,
	 * 2) ~/.phoebe2,
	 * 3) ~/.phoebe
	 *
	 * Once the configuration file is open, the function configures all
	 * PHOEBE features (passband transmission functions, limb darkening
	 * coefficients etc).
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status;
	char homedir[255], conffile[255];
	char *pathname;
	bool switch_state;

	phoebe_debug ("* adding configuration entries...\n");
	phoebe_config_populate ();

	phoebe_debug ("* probing configuration directories...\n");
	if (PHOEBE_HOME_DIR) {
		/* This happens when the driver supplied PHOEBE_HOME_DIR variable. */
		sprintf (conffile, "%s/phoebe.config", PHOEBE_HOME_DIR);
		status = phoebe_config_peek (conffile);
		if (status == SUCCESS) {
			PHOEBE_CONFIG = strdup (conffile);
			phoebe_config_load (PHOEBE_CONFIG);
		}
		else if (status == ERROR_PHOEBE_CONFIG_LEGACY_FILE) {
			PHOEBE_CONFIG = strdup (conffile);
			phoebe_lib_warning ("importing legacy configuration file (pre-0.30).");
			phoebe_config_import (PHOEBE_CONFIG);
		}
		else {
			phoebe_lib_error ("Config file not found in %s, reverting to defaults.\n", PHOEBE_HOME_DIR);
		}
	}

	if (!PHOEBE_HOME_DIR) {
		/* Check for config in ~/phoebe2: */
		sprintf (homedir, "%s/.phoebe2", USER_HOME_DIR);
		sprintf (conffile, "%s/phoebe.config", homedir);

		status = phoebe_config_peek (conffile);

		if (status == SUCCESS) {
			PHOEBE_HOME_DIR = strdup (homedir);
			PHOEBE_CONFIG = strdup (conffile);
			phoebe_config_load (PHOEBE_CONFIG);
		}
		else if (status == ERROR_PHOEBE_CONFIG_LEGACY_FILE) {
			PHOEBE_HOME_DIR = strdup (homedir);
			PHOEBE_CONFIG = strdup (conffile);
			phoebe_lib_warning ("importing legacy configuration file (pre-0.30).");
			phoebe_config_import (PHOEBE_CONFIG);
		}
	}

	if (!PHOEBE_HOME_DIR) {
		/* Check for config in ~/.phoebe: */
		sprintf (homedir, "%s/.phoebe", USER_HOME_DIR);
		sprintf (conffile, "%s/phoebe.config", homedir);

		status = phoebe_config_peek (conffile);

		if (status == SUCCESS) {
			PHOEBE_HOME_DIR = strdup (homedir);
			PHOEBE_CONFIG = strdup (conffile);
			phoebe_config_load (PHOEBE_CONFIG);
		}
		else if (status == ERROR_PHOEBE_CONFIG_LEGACY_FILE) {
			PHOEBE_HOME_DIR = strdup (homedir);
			PHOEBE_CONFIG = strdup (conffile);
			phoebe_lib_warning ("importing legacy configuration file (pre-0.30).");
			phoebe_config_import (PHOEBE_CONFIG);
		}
	}

	if (!PHOEBE_HOME_DIR) {
		/* Admit defeat and revert to defaults. */
		phoebe_lib_warning ("configuration file not found, reverting to defaults.\n");
	}

	/* It is time now to configure all PHOEBE features. */

	getcwd (homedir, 255);
	PHOEBE_STARTUP_DIR = strdup (homedir);

	phoebe_config_entry_get ("PHOEBE_PTF_DIR", &pathname);
	phoebe_read_in_passbands (pathname);
	phoebe_debug ("* %d passbands read in.\n", PHOEBE_passbands_no);

	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &switch_state);
	if (switch_state == 1) {
		phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname);
		status = read_in_ld_nodes (pathname);
		if (status != SUCCESS) {
			phoebe_lib_error ("reading LD table coefficients failed, disabling readouts.\n");
			phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 0);
		}
	}

	phoebe_debug ("* declaring parameters...\n");
	phoebe_init_parameters ();

	phoebe_debug ("* declaring parameter options...\n");
	phoebe_init_parameter_options ();


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

	/* Restore the original locale of the system: */
	setlocale (LC_NUMERIC, PHOEBE_INPUT_LOCALE);

	/* Free the LD table:                                                     */
	phoebe_ld_table_free ();

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

	/* Free configuration entries: */
	phoebe_config_free ();

	/* Free parameter table: */
	phoebe_parameter_table_free (PHOEBE_pt);

	exit (0);
}
