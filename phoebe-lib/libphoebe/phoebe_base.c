#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#include "phoebe_spectra.h"

#ifdef HAVE_LIBGSL
	#ifndef PHOEBE_GSL_DISABLED
		/* This part initializes PHOEBE randomizer, but only if GSL is present.   */
		#include <gsl/gsl_rng.h>
	#endif
#endif

/**
 * SECTION:phoebe_base
 * @title: PHOEBE base
 * @short_description: initialization functions
 *
 * These are the functions that initialize PHOEBE and its components.
 */


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

	PHOEBE_ld_table      = NULL;

	PHOEBE_spectra_repository.no = 0;
	PHOEBE_spectra_repository.prop = NULL;

	return SUCCESS;
}

void intern_phoebe_sigint_handler (int signum)
{
	/*
	 * This is an alternate handler for the interrupt (CTRL+C) handler. Since
	 * the interrupt handler by default breaks the program itself rather than
	 * the process it executes, we need to catch the signal here and block it.
	 */

	phoebe_debug ("sigint %d blocked!\n", signum);
	PHOEBE_INTERRUPT = TRUE;
	phoebe_lib_error ("break called; exitting PHOEBE.\n\n");
	exit (0);
	return;
}


int phoebe_init ()
{
	/**
	 * phoebe_init:
	 *
	 * Initializes PHOEBE. The function initializes all PHOEBE variables,
	 * sets the "C" locale, sets environment variables, randomizes the seed
	 * and sets up the break event handler.
	 *
	 * This function must be called before any of PHOEBE's functions that rely
	 * on parameters or variables is used.
	 */

	phoebe_debug ("Welcome to PHOEBE-lib debugger! :)");

	phoebe_debug ("* initialize the variables...\n");
	intern_phoebe_variables_init ();

	/*
	 * Get the current locale decimal point, store it to the global variable,
	 * change it to "C" and change it back to original upon exit.
	 */

	PHOEBE_INPUT_LOCALE = strdup (setlocale (LC_NUMERIC, NULL));
	setlocale (LC_NUMERIC, "C");

#ifdef __MINGW32__
// If HOME is not defined on Windows, take the current directory where the program is started
	if (!getenv ("HOME"))
		USER_HOME_DIR = ".";
	else
		USER_HOME_DIR = strdup (getenv ("HOME"));
#else
	if (!getenv ("HOME"))
		return ERROR_HOME_ENV_NOT_DEFINED;
	USER_HOME_DIR = strdup (getenv ("HOME"));
#endif

	/* Choose a randomizer seed: */
	srand (phoebe_seed ());

	/*
	 * Catch the interrupt (CTRL+C) signal and use an alternative handler
	 * that will break a running script rather than the scripter itself:
	 */

	signal (SIGINT, intern_phoebe_sigint_handler);
	PHOEBE_INTERRUPT = FALSE;

	return SUCCESS;
}

int phoebe_load_ld_tables ()
{
	/**
	 * phoebe_load_ld_tables:
	 *
	 * Frees the existing ld table and loads it from the current #PHOEBE_LD_DIR
	 * directory.
	 *
	 * Returns: 
	 */

	bool intern;
	char *pathname;
	char *model_list;

	phoebe_config_entry_get ("PHOEBE_LD_INTERN", &intern);

	phoebe_ld_table_free (PHOEBE_ld_table);

	if (intern == 1) {
		phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname);
		model_list = phoebe_concatenate_strings (pathname, "/models.list", NULL);
		PHOEBE_ld_table = phoebe_ld_table_intern_load (model_list);
		free (model_list);
	}
	else {
		phoebe_config_entry_get ("PHOEBE_LD_VH_DIR", &pathname);
		PHOEBE_ld_table = phoebe_ld_table_vh1993_load (pathname);
	}

	if (!PHOEBE_ld_table) {
		phoebe_lib_error ("reading LD table coefficients failed, disabling readouts.\n");
		phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 0);
	}

	return SUCCESS;
}

int phoebe_configure ()
{
	/**
	 * phoebe_configure:
	 *
	 * Looks for the configuration file, loads it (if found) and reads in the
	 * configuration entries. The configuration filename is hardcoded to
	 * phoebe.config (this should be changed in future), but the configuration
	 * directory PHOEBE_HOME_DIR can be set by the drivers.
	 *
	 * The order of the checked config directories is:
	 *
	 * 1) PHOEBE_HOME_DIR if not NULL,
	 * 2) Current ~/.phoebe-VERSION (i.e. ~/.phoebe-0.32),
	 * 3) Previous (but compatible) ~/.phoebe-VERSIONs (i.e. ~/.phoebe-0.31,
	 *    ~/.phoebe-0.30), in the decreasing order (most recent come first),
	 * 4) Legacy ~/.phoebe
	 *
	 * If a legacy (pre-0.30) config file is found, its configuration entries
	 * will be imported. In all cases the configuration directory will be set to
	 * the current ~/.phoebe-VERSION so that saving a configuration file stores
	 * the settings to the current directory.
	 * 
	 * Once the configuration file is open, the function configures all
	 * PHOEBE features (passband transmission functions, limb darkening
	 * coefficients etc).
	 *
	 * If the configuration file is not found, defaults are assumed.
	 *
	 * Returns: #PHOEBE_error_code.
	 */
	
	int status;
	char dname[255], homedir[255], conffile[255];
	char *pathname;
	bool switch_state, return_flag = SUCCESS;
	
	/* Current working directory: */
	if (!getcwd (dname, 255)) {
		phoebe_lib_warning ("current working directory cannot be accessed or is insanely long.\n");
		PHOEBE_STARTUP_DIR = strdup ("");
	}
	else
		PHOEBE_STARTUP_DIR = strdup (dname);
	
	phoebe_debug ("* adding configuration entries...\n");
	phoebe_config_populate ();
	
	phoebe_debug ("* looking for a configuration directory...\n");
	if (PHOEBE_HOME_DIR) {
		/* This happens when the driver supplied PHOEBE_HOME_DIR variable. */
		sprintf (conffile, "%s/phoebe.config", PHOEBE_HOME_DIR);
		switch (phoebe_config_peek (conffile)) {
			case SUCCESS:
				PHOEBE_CONFIG = strdup (conffile);
				phoebe_config_load (PHOEBE_CONFIG);
			break;
			case ERROR_PHOEBE_CONFIG_LEGACY_FILE:
				return_flag = ERROR_PHOEBE_CONFIG_LEGACY_FILE;
				phoebe_lib_warning ("importing legacy configuration file (pre-0.30).\n");
				phoebe_config_import (conffile);
				PHOEBE_CONFIG = strdup (conffile);
			break;
			default:
				return_flag = ERROR_PHOEBE_CONFIG_NOT_FOUND;
				phoebe_lib_warning ("config file not found in %s, reverting to defaults.\n", PHOEBE_HOME_DIR);
		}
	}
	else {
		/* Check for config in ~/.phoebe-VERSION: */
		sprintf (homedir, "%s/.phoebe-%s", USER_HOME_DIR, PACKAGE_VERSION);
		sprintf (conffile, "%s/phoebe.config", homedir);

		switch (phoebe_config_peek (conffile)) {
			case SUCCESS:
				PHOEBE_HOME_DIR = strdup (homedir);
				PHOEBE_CONFIG = strdup (conffile);
				phoebe_config_load (PHOEBE_CONFIG);
			break;
			case ERROR_PHOEBE_CONFIG_LEGACY_FILE:
				return_flag = ERROR_PHOEBE_CONFIG_LEGACY_FILE;
				phoebe_lib_warning ("importing legacy configuration file (pre-0.30).\n");
				phoebe_config_import (conffile);
				
				/* The config file should point to the ~/.phoebe-VERSION dir: */
				sprintf (homedir,  "%s/.phoebe-%s", USER_HOME_DIR, PACKAGE_VERSION);
				sprintf (conffile, "%s/phoebe.config", homedir);
				
				PHOEBE_HOME_DIR = strdup (homedir);
				PHOEBE_CONFIG = strdup (conffile);
			break;
			default:
			{
				/* Check for config in ~/.phoebe-{ext} directories: */
				char ext[2][5] = {"0.31", "0.30"};
				int i;

				for (i = 0; i < 2; i++) {
					sprintf (homedir, "%s/.phoebe-%s", USER_HOME_DIR, ext[i]);
					sprintf (conffile, "%s/phoebe.config", homedir);

					status = phoebe_config_peek (conffile);
					if (status == SUCCESS) {
						return_flag = ERROR_PHOEBE_CONFIG_SUPPORTED_FILE;
						phoebe_lib_warning ("importing PHOEBE %s configuration file; please review your configuration settings.\n", ext[i]);
						phoebe_config_load (conffile);

						/* The config file should point to the ~/.phoebe-VERSION dir: */
						sprintf (homedir,  "%s/.phoebe-%s", USER_HOME_DIR, PACKAGE_VERSION);
						sprintf (conffile, "%s/phoebe.config", homedir);
						break;
					}
				}
			}
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
			phoebe_lib_warning ("importing legacy configuration file (pre-0.30).");
			phoebe_config_import (conffile);

			/* The config file should point to the ~/.phoebe-VERSION dir: */
			sprintf (homedir,  "%s/.phoebe-%s", USER_HOME_DIR, PACKAGE_VERSION);
			sprintf (conffile, "%s/phoebe.config", homedir);

			PHOEBE_HOME_DIR = strdup (homedir);
			PHOEBE_CONFIG = strdup (conffile);
		}
	}

	if (!PHOEBE_HOME_DIR) {
		/* Admit defeat and revert to defaults. */
		return_flag = ERROR_PHOEBE_CONFIG_NOT_FOUND;
		phoebe_lib_warning ("configuration file not found, reverting to defaults.\n");
	}

	/* It is time now to configure all PHOEBE features. */

	phoebe_config_entry_get ("PHOEBE_PTF_DIR", &pathname);
	phoebe_read_in_passbands (pathname);
	phoebe_debug ("* %d passbands read in.\n", PHOEBE_passbands_no);

	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &switch_state);
	if (switch_state == 1) {
		phoebe_config_entry_get ("PHOEBE_LD_INTERN", &switch_state);
		if (switch_state == 1) {
			phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname);
			phoebe_ld_attach_all (pathname);
			phoebe_debug ("* LD tables attached.\n");
		}
		else {
			phoebe_config_entry_get ("PHOEBE_LD_VH_DIR", &pathname);
		}

		phoebe_load_ld_tables ();
	}

	phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &switch_state);
	if (switch_state == 1) {
		phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &pathname);
		status = phoebe_spectra_set_repository (pathname);
		if (status != SUCCESS) {
			phoebe_lib_error ("Spectra repository cannot be accessed, disabling readouts.\n");
			phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", 0);
		}
	}

	phoebe_debug ("* declaring parameters...\n");
	phoebe_init_parameters ();

	phoebe_debug ("* declaring parameter options...\n");
	phoebe_init_parameter_options ();

	return return_flag;
}

int phoebe_quit ()
{
	/**
	 * phoebe_quit:
	 *
	 * Frees all PHOEBE-related memory, restores the default locale and exits
	 * PHOEBE cleanly.
	 */

	/* Restore the original locale of the system: */
	setlocale (LC_NUMERIC, PHOEBE_INPUT_LOCALE);

	/* Free the LD table: */
	phoebe_ld_table_free (PHOEBE_ld_table);

	/* Free the spectra table: */
	phoebe_spectra_free_repository ();

	/* Free all global PHOEBE strings: */
	free (PHOEBE_STARTUP_DIR);
	free (PHOEBE_INPUT_LOCALE);
#ifdef __MINGW32__
	if (getenv ("HOME"))
#endif
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
