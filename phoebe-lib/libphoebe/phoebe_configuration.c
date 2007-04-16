#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>

#include "phoebe_build_config.h"
#include "phoebe_error_handling.h"

/* GNU readline library may be used for configuration:                        */
#ifdef HAVE_LIBREADLINE
	#ifndef PHOEBE_READLINE_DISABLED
		#include <readline/readline.h>
		#include <readline/history.h>
	#endif
#endif

/* Global configuration parameters: */
char *USER_HOME_DIR;

char *PHOEBE_VERSION_NUMBER;
char *PHOEBE_VERSION_DATE;
char *PHOEBE_PARAMETERS_FILENAME;
char *PHOEBE_DEFAULTS;

int   PHOEBE_CONFIG_EXISTS;
char *PHOEBE_STARTUP_DIR;
char *PHOEBE_HOME_DIR;
char *PHOEBE_CONFIG;
char *PHOEBE_BASE_DIR;
char *PHOEBE_SOURCE_DIR;
char *PHOEBE_DEFAULTS_DIR;
char *PHOEBE_TEMP_DIR;
char *PHOEBE_DATA_DIR;
char *PHOEBE_PTF_DIR;
char *PHOEBE_PLOTTING_PACKAGE;
int   PHOEBE_LD_SWITCH;
char *PHOEBE_LD_DIR;
int   PHOEBE_KURUCZ_SWITCH;
char *PHOEBE_KURUCZ_DIR;

char *PHOEBE_INPUT_LOCALE;

int PHOEBE_3D_PLOT_CALLBACK_OPTION;
int PHOEBE_CONFIRM_ON_SAVE;
int PHOEBE_CONFIRM_ON_QUIT;
int PHOEBE_WARN_ON_SYNTHETIC_SCATTER;

int phoebe_create_configuration_file ()
{
	int status, error;
	char answer;

	printf ("\nWelcome to %s!\n\n", PHOEBE_VERSION_NUMBER);
	printf ("I am about to create a top-level configuration directory:\n\n\t%s\n\n", PHOEBE_HOME_DIR);
	printf ("and guide you through the setup process. Do you wish to continue? [Y/n] ");
	answer = fgetc (stdin);

	if (answer != 'y' && answer != 'Y' && answer != '\n') {
		printf ("\n");
		exit (0);
	}

	status = mkdir (PHOEBE_HOME_DIR, 0755);
	error = errno;

	if (status == -1) {
		if (error == EEXIST)
			printf ("\nDirectory %s already exists, proceeding to configuration.\n\n", PHOEBE_HOME_DIR);
		else if (error == EACCES) {
			printf ("\nYou don't have permissions to create %s.\n", PHOEBE_HOME_DIR);
			printf ("Please enable write access to your home directory and restart PHOEBE.\n\n");
			exit (-1);
		}
		else if (error == ENOSPC) {
			printf ("\nYour hard disk is out of space. Please make some room and try again.\n\n");
			exit (-1);
		}
		else {
			printf ("\nAn undefined error has occured while creating configuration directory.\n");
			printf ("Please try to resolve the issue and restart PHOEBE.\n\n");
			exit (-1);
		}
	}
	else
		printf ("\nPHOEBE configuration directory created.\n\n");

	/* If libreadline is present, use it:                                       */
	#ifdef HAVE_LIBREADLINE
		#ifndef PHOEBE_READLINE_DISABLED
			{
			char *basedir, *srcdir, *defdir, *tempdir, *datadir, *ptfdir, *lddir;
			char *ldswitch, *yn;
			char prompt[255], defaultdir[255];

			printf ("Please supply names of directories to be used by PHOEBE:\n");
			printf ("  note that you may use tab-completion features to access directories;\n");
			printf ("  if you are happy with defaults enclosed in [...], just press ENTER.\n\n");

			/*************************** BASE DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_BASE_DIR != NULL) {
				sprintf (prompt, "PHOEBE base directory      [%s]: ", PHOEBE_BASE_DIR);
				basedir = readline (prompt);
				if (strcmp (basedir, "") == 0) basedir = strdup (PHOEBE_BASE_DIR);
			}
			else {
				sprintf (prompt, "PHOEBE base directory      [/usr/local/share/phoebe2]: ");
				basedir = readline (prompt);
				if (strcmp (basedir, "") == 0) basedir = strdup ("/usr/local/share/phoebe2");
			}
			if (basedir[strlen(basedir)-1] == '/') basedir[strlen(basedir)-1] = '\0';

			/************************** SOURCE DIRECTORY ****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_SOURCE_DIR != NULL) {
				sprintf (prompt, "PHOEBE source directory    [%s]: ", PHOEBE_SOURCE_DIR);
				srcdir = readline (prompt);
				if (strcmp (srcdir, "") == 0) srcdir = strdup (PHOEBE_SOURCE_DIR);
			}
			else {
				sprintf (prompt, "PHOEBE source directory    [%s/src]: ", basedir);
				srcdir = readline (prompt);
				if (strcmp (srcdir, "") == 0) {
					sprintf (defaultdir, "%s/src", basedir);
					srcdir = strdup (defaultdir);
				}
			}
			if (srcdir[strlen(srcdir)-1] == '/') srcdir[strlen(srcdir)-1] = '\0';

			/************************* DEFAULTS DIRECTORY ***************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_DEFAULTS_DIR != NULL) {
				sprintf (prompt, "PHOEBE defaults directory  [%s]: ", PHOEBE_DEFAULTS_DIR);
				defdir = readline (prompt);
				if (strcmp (defdir, "") == 0) defdir = strdup (PHOEBE_DEFAULTS_DIR);
			}
			else {
				sprintf (prompt, "PHOEBE defaults directory  [%s/defaults]: ", basedir);
				defdir = readline (prompt);
				if (strcmp (defdir, "") == 0) {
					sprintf (defaultdir, "%s/defaults", basedir);
					defdir = strdup (defaultdir);
				}
			}
			if (defdir[strlen(defdir)-1] == '/') defdir[strlen(defdir)-1] = '\0';

			/*************************** TEMP DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_TEMP_DIR != NULL) {
				sprintf (prompt, "PHOEBE temporary directory [%s]: ", PHOEBE_TEMP_DIR);
				tempdir = readline (prompt);
				if (strcmp (tempdir, "") == 0) tempdir = strdup (PHOEBE_TEMP_DIR);
			}
			else {
				sprintf (prompt, "PHOEBE temporary directory [/tmp]: ");
				tempdir = readline (prompt);
				if (strcmp (tempdir, "") == 0) tempdir = strdup ("/tmp");
			}
			if (tempdir[strlen(tempdir)-1] == '/') tempdir[strlen(tempdir)-1] = '\0';

			/*************************** DATA DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_DATA_DIR != NULL) {
				sprintf (prompt, "PHOEBE data directory      [%s]: ", PHOEBE_DATA_DIR);
				datadir = readline (prompt);
				if (strcmp (datadir, "") == 0) datadir = strdup (PHOEBE_DATA_DIR);
			}
			else {
				sprintf (prompt, "PHOEBE data directory      [%s/phoebe/data]: ", USER_HOME_DIR);
				datadir = readline (prompt);
				if (strcmp (datadir, "") == 0) {
					sprintf (defaultdir, "%s/phoebe/data", USER_HOME_DIR);
					datadir = strdup (defaultdir);
				}
			}
			if (datadir[strlen(datadir)-1] == '/') datadir[strlen(datadir)-1] = '\0';

			/**************************** PTF DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_PTF_DIR != NULL) {
				sprintf (prompt, "PHOEBE PTF directory       [%s]: ", PHOEBE_PTF_DIR);
				ptfdir = readline (prompt);
				if (strcmp (ptfdir, "") == 0) ptfdir = strdup (PHOEBE_PTF_DIR);
			}
			else {
				sprintf (prompt, "PHOEBE PTF directory       [%s/ptf]: ", basedir);
				ptfdir = readline (prompt);
				if (strcmp (ptfdir, "") == 0) {
					sprintf (defaultdir, "%s/phoebe/ptf", USER_HOME_DIR);
					ptfdir = strdup (defaultdir);
				}
			}
			if (ptfdir[strlen(datadir)-1] == '/') ptfdir[strlen(datadir)-1] = '\0';

			/****************************** LD TABLES *******************************/

			ldswitch = readline ("Are Van Hamme (1993) limb darkening tables present on your system [y/N]? ");
			if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') ;
			else {
				if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_LD_DIR != NULL) {
					sprintf (prompt, "PHOEBE LD directory        [%s]: ", PHOEBE_LD_DIR);
					lddir = readline (prompt);
					if (strcmp (lddir, "") == 0) lddir = strdup (PHOEBE_LD_DIR);
				}
				else {
					sprintf (prompt, "PHOEBE LD directory        [%s/catalogs/ld]: ", USER_HOME_DIR);
					lddir = readline (prompt);
					if (strcmp (lddir, "") == 0) {
						sprintf (defaultdir, "%s/catalogs/ld", USER_HOME_DIR);
						lddir = strdup (defaultdir);
					}
				}
				if (lddir[strlen(lddir)-1] == '/') lddir[strlen(lddir)-1] = '\0';
			}

			/************************************************************************/

			printf ("\nConfiguration summary:\n----------------------\n");
			printf ("PHOEBE base directory:       %s\n", basedir);
			printf ("PHOEBE source directory:     %s\n", srcdir);
			printf ("PHOEBE defaults directory:   %s\n", defdir);
			printf ("PHOEBE temporary directory:  %s\n", tempdir);
			printf ("PHOEBE data directory:       %s\n", datadir);
			printf ("PHOEBE passbands directory:  %s\n", ptfdir);

			if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N')
				printf ("Limb darkening tables:       not present\n");
			else {
				printf ("Limb darkening tables:       present\n");
				printf ("PHOEBE LD directory:         %s\n", lddir);
			}

			yn = readline ("\nAre these settings ok [Y/n]? ");
			if (strcmp (yn, "") == 0 || yn[0] == 'y' || yn[0] == 'Y') {
				FILE *config = fopen (PHOEBE_CONFIG, "w");
				fprintf (config, "PHOEBE_BASE_DIR\t\t%s\n", basedir);
				fprintf (config, "PHOEBE_SOURCE_DIR\t%s\n", srcdir);
				fprintf (config, "PHOEBE_DEFAULTS_DIR\t%s\n", defdir);
				fprintf (config, "PHOEBE_TEMP_DIR\t\t%s\n", tempdir);
				fprintf (config, "PHOEBE_DATA_DIR\t\t%s\n", datadir);
				fprintf (config, "PHOEBE_PTF_DIR\t\t%s\n", ptfdir);


				if (PHOEBE_BASE_DIR     != NULL) free (PHOEBE_BASE_DIR);
				if (PHOEBE_SOURCE_DIR   != NULL) free (PHOEBE_SOURCE_DIR);
				if (PHOEBE_DEFAULTS_DIR != NULL) free (PHOEBE_DEFAULTS_DIR);
				if (PHOEBE_TEMP_DIR     != NULL) free (PHOEBE_TEMP_DIR);
				if (PHOEBE_DATA_DIR     != NULL) free (PHOEBE_DATA_DIR);
				if (PHOEBE_PTF_DIR      != NULL) free (PHOEBE_PTF_DIR);
				if (PHOEBE_LD_DIR       != NULL) free (PHOEBE_LD_DIR);

				PHOEBE_BASE_DIR     = strdup (basedir);
				PHOEBE_SOURCE_DIR   = strdup (srcdir);
				PHOEBE_DEFAULTS_DIR = strdup (defdir);
				PHOEBE_TEMP_DIR     = strdup (tempdir);
				PHOEBE_DATA_DIR     = strdup (datadir);
				PHOEBE_PTF_DIR      = strdup (ptfdir);

				if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') {
					fprintf (config, "PHOEBE_LD_SWITCH\t0\n");
					fprintf (config, "PHOEBE_LD_DIR\t\t\n");
					PHOEBE_LD_SWITCH = 0;
					PHOEBE_LD_DIR = strdup ("");
				}
				else {
					fprintf (config, "PHOEBE_LD_SWITCH\t1\n");
					fprintf (config, "PHOEBE_LD_DIR\t\t%s\n", lddir);
					PHOEBE_LD_SWITCH = 1;
					PHOEBE_LD_DIR = strdup (lddir);
				}
				fclose (config);

				printf ("\nConfiguration file written. If you ever want to change your settings,\n");
				printf ("simply edit %s/phoebe.config or start PHOEBE\n", PHOEBE_HOME_DIR);
				printf ("with -c (--configure) switch.\n");
				return SUCCESS;
			}
			else {
				printf ("\nAborting configuration. Please restart PHOEBE to try again.\n\n");
				exit (0);
			}
		}
		#endif
	#endif

		/* This part is executed if GNU readline isn't found:                     */
			{
			char basedir[255], srcdir[255], defdir[255], tempdir[255], datadir[255], ptfdir[255], lddir[255];
			char ldswitch[255], yn[255];

			printf ("Please supply names of directories to be used by PHOEBE:\n");
			printf ("  if you are happy with defaults enclosed in [...], just press ENTER.\n\n");

			/*************************** BASE DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_BASE_DIR != NULL) {
				printf ("PHOEBE base directory      [%s]: ", PHOEBE_BASE_DIR);
				fgets (basedir, 255, stdin); basedir[strlen(basedir)-1] = '\0';
				if (strcmp (basedir, "") == 0) sprintf (basedir, "%s", PHOEBE_BASE_DIR);
			}
			else {
				printf ("PHOEBE base directory      [/usr/local/share/phoebe2]: ");
				fgets (basedir, 255, stdin); basedir[strlen(basedir)-1] = '\0';
				if (strcmp (basedir, "") == 0) sprintf (basedir, "/usr/local/share/phoebe2");
			}
			if (basedir[strlen(basedir)-1] == '/') basedir[strlen(basedir)-1] = '\0';

			/************************** SOURCE DIRECTORY ****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_SOURCE_DIR != NULL) {
				printf ("PHOEBE source directory    [%s]: ", PHOEBE_SOURCE_DIR);
				fgets (srcdir, 255, stdin); srcdir[strlen(srcdir)-1] = '\0';
				if (strcmp (srcdir, "") == 0) sprintf (srcdir, "%s", PHOEBE_SOURCE_DIR);
			}
			else {
				printf ("PHOEBE source directory    [%s/src]: ", basedir);
				fgets (srcdir, 255, stdin); srcdir[strlen(srcdir)-1] = '\0';
				if (strcmp (srcdir, "") == 0) sprintf (srcdir, "%s/src", basedir);
			}
			if (srcdir[strlen(srcdir)-1] == '/') srcdir[strlen(srcdir)-1] = '\0';

			/************************* DEFAULTS DIRECTORY ***************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_DEFAULTS_DIR != NULL) {
				printf ("PHOEBE defaults directory  [%s]: ", PHOEBE_DEFAULTS_DIR);
				fgets (defdir, 255, stdin); defdir[strlen(defdir)-1] = '\0';
				if (strcmp (defdir, "") == 0) sprintf (defdir, "%s", PHOEBE_DEFAULTS_DIR);
			}
			else {
				printf ("PHOEBE defaults directory  [%s/defaults]: ", basedir);
				fgets (defdir, 255, stdin); defdir[strlen(defdir)-1] = '\0';
				if (strcmp (defdir, "") == 0) sprintf (defdir, "%s/defaults", basedir);
			}
			if (defdir[strlen(defdir)-1] == '/') defdir[strlen(defdir)-1] = '\0';

			/*************************** TEMP DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_TEMP_DIR != NULL) {
				printf ("PHOEBE temporary directory [%s]: ", PHOEBE_TEMP_DIR);
				fgets (tempdir, 255, stdin); tempdir[strlen(tempdir)-1] = '\0';
				if (strcmp (tempdir, "") == 0) sprintf (tempdir, "%s", PHOEBE_TEMP_DIR);
			}
			else {
				printf ("PHOEBE temporary directory [/tmp]: ");
				fgets (tempdir, 255, stdin); tempdir[strlen(tempdir)-1] = '\0';
				if (strcmp (tempdir, "") == 0) sprintf (tempdir, "/tmp");
			}
			if (tempdir[strlen(tempdir)-1] == '/') tempdir[strlen(tempdir)-1] = '\0';

			/*************************** DATA DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_DATA_DIR != NULL) {
				printf ("PHOEBE data directory      [%s]: ", PHOEBE_DATA_DIR);
				fgets (datadir, 255, stdin); datadir[strlen(datadir)-1] = '\0';
				if (strcmp (datadir, "") == 0) sprintf (datadir, "%s", PHOEBE_DATA_DIR);
			}
			else {
				printf ("PHOEBE data directory      [%s/data]: ", basedir);
				fgets (datadir, 255, stdin); datadir[strlen(datadir)-1] = '\0';
				if (strcmp (datadir, "") == 0) sprintf (datadir, "%s/data", basedir);
			}
			if (datadir[strlen(datadir)-1] == '/') datadir[strlen(datadir)-1] = '\0';

			/**************************** PTF DIRECTORY *****************************/

			if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_PTF_DIR != NULL) {
				printf ("PHOEBE PTF directory       [%s]: ", PHOEBE_PTF_DIR);
				fgets (ptfdir, 255, stdin); ptfdir[strlen(ptfdir)-1] = '\0';
				if (strcmp (ptfdir, "") == 0) sprintf (ptfdir, "%s", PHOEBE_PTF_DIR);
			}
			else {
				printf ("PHOEBE PTF directory       [%s/ptf]: ", basedir);
				fgets (ptfdir, 255, stdin); ptfdir[strlen(ptfdir)-1] = '\0';
				if (strcmp (ptfdir, "") == 0) sprintf (ptfdir, "%s/ptf", basedir);
			}
			if (ptfdir[strlen(datadir)-1] == '/') ptfdir[strlen(datadir)-1] = '\0';

			/****************************** LD TABLES *******************************/

			printf ("Are Van Hamme (1993) limb darkening tables present on your system [y/N]? ");
			fgets (ldswitch, 255, stdin); ldswitch[strlen(ldswitch)-1] = '\0';

			if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') ;
			else {
				if (PHOEBE_CONFIG_EXISTS == 1 && PHOEBE_LD_DIR != NULL) {
					printf ("PHOEBE LD directory        [%s]: ", PHOEBE_LD_DIR);
					fgets (lddir, 255, stdin); lddir[strlen(lddir)-1] = '\0';
					if (strcmp (lddir, "") == 0) sprintf (lddir, "%s", PHOEBE_LD_DIR);
				}
				else {
					printf ("PHOEBE LD directory        [%s/catalogs/ld]: ", USER_HOME_DIR);
					fgets (lddir, 255, stdin); lddir[strlen(lddir)-1] = '\0';
					if (strcmp (lddir, "") == 0) sprintf (lddir, "%s/catalogs/ld", USER_HOME_DIR);
				}
				if (lddir[strlen(lddir)-1] == '/') lddir[strlen(lddir)-1] = '\0';
			}

			/************************************************************************/

			printf ("\nConfiguration summary:\n----------------------\n");
			printf ("PHOEBE base directory:       %s\n", basedir);
			printf ("PHOEBE source directory:     %s\n", srcdir);
			printf ("PHOEBE defaults directory:   %s\n", defdir);
			printf ("PHOEBE temporary directory:  %s\n", tempdir);
			printf ("PHOEBE data directory:       %s\n", datadir);
			printf ("PHOEBE passbands directory:  %s\n", ptfdir);

			if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N')
				printf ("Limb darkening tables:       not present\n");
			else {
				printf ("Limb darkening tables:       present\n");
				printf ("PHOEBE LD directory:         %s\n", lddir);
			}

			printf ("\nAre these settings ok [Y/n]? ");
			fgets (yn, 255, stdin); yn[strlen(yn)-1] = '\0';

			if (strcmp (yn, "") == 0 || yn[0] == 'y' || yn[0] == 'Y') {
				FILE *config = fopen (PHOEBE_CONFIG, "w");
				fprintf (config, "PHOEBE_BASE_DIR\t\t%s\n",   basedir);
				fprintf (config, "PHOEBE_SOURCE_DIR\t%s\n",   srcdir);
				fprintf (config, "PHOEBE_DEFAULTS_DIR\t%s\n", defdir);
				fprintf (config, "PHOEBE_TEMP_DIR\t\t%s\n",   tempdir);
				fprintf (config, "PHOEBE_DATA_DIR\t\t%s\n",   datadir);
				fprintf (config, "PHOEBE_PTF_DIR\t\t%s\n",    ptfdir);

				if (PHOEBE_BASE_DIR     != NULL) free (PHOEBE_BASE_DIR);
				if (PHOEBE_SOURCE_DIR   != NULL) free (PHOEBE_SOURCE_DIR);
				if (PHOEBE_DEFAULTS_DIR != NULL) free (PHOEBE_DEFAULTS_DIR);
				if (PHOEBE_TEMP_DIR     != NULL) free (PHOEBE_TEMP_DIR);
				if (PHOEBE_DATA_DIR     != NULL) free (PHOEBE_DATA_DIR);
				if (PHOEBE_PTF_DIR      != NULL) free (PHOEBE_PTF_DIR);
				if (PHOEBE_LD_DIR       != NULL) free (PHOEBE_LD_DIR);

				PHOEBE_BASE_DIR     = strdup (basedir);
				PHOEBE_SOURCE_DIR   = strdup (srcdir);
				PHOEBE_DEFAULTS_DIR = strdup (defdir);
				PHOEBE_TEMP_DIR     = strdup (tempdir);
				PHOEBE_DATA_DIR     = strdup (datadir);
				PHOEBE_PTF_DIR      = strdup (ptfdir);

				if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') {
					fprintf (config, "PHOEBE_LD_SWITCH\t0\n");
					fprintf (config, "PHOEBE_LD_DIR\t\t\n");
					PHOEBE_LD_SWITCH = 0;
					PHOEBE_LD_DIR = strdup ("");
				}
				else {
					fprintf (config, "PHOEBE_LD_SWITCH\t1\n");
					fprintf (config, "PHOEBE_LD_DIR\t\t%s\n", lddir);
					PHOEBE_LD_SWITCH = 1;
					PHOEBE_LD_DIR = strdup (lddir);
				}
				fclose (config);

				printf ("\nConfiguration file written. If you ever want to change your settings,\n");
				printf ("simply edit %s/phoebe.config or start PHOEBE\n", PHOEBE_HOME_DIR);
				printf ("with -c (--configure) switch.\n");
				return SUCCESS;
			}
			else {
				printf ("\nAborting configuration. Please restart PHOEBE to try again.\n\n");
				exit (0);
			}
		}
	
	return SUCCESS;
}
