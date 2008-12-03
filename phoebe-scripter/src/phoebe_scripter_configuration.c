#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_build_config.h"

/* GNU readline library may be used for configuration: */
#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
	#include <readline/readline.h>
	#include <readline/history.h>
#endif

int scripter_create_config_file ()
{
	int status, error;
	char answer;

	char *pathname;

	if (!PHOEBE_HOME_DIR) {
		char homedir[255];
		char configfile[255];
		sprintf (homedir, "%s/.phoebe-%s", USER_HOME_DIR, PACKAGE_VERSION);
		PHOEBE_HOME_DIR = strdup (homedir);
		sprintf (configfile, "%s/phoebe.config", PHOEBE_HOME_DIR);
		PHOEBE_CONFIG = strdup (configfile);
	}

	printf ("\nWelcome to %s!\n\n", PHOEBE_VERSION_NUMBER);
	printf ("I am about to create a top-level configuration directory:\n\n\t%s\n\n", PHOEBE_HOME_DIR);
	printf ("and guide you through the setup process. Do you wish to continue? [Y/n] ");
	answer = fgetc (stdin);

	if (answer != 'y' && answer != 'Y' && answer != '\n') {
		printf ("\n");
		exit (0);
	}

#ifdef __MINGW32__
	status = mkdir (PHOEBE_HOME_DIR);
#else
	status = mkdir (PHOEBE_HOME_DIR, 0755);
#endif
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

	/* If libreadline is present, use it: */
#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
	{
		char *basedir, *defdir, *tempdir, *datadir, *ptfdir, *lddir, *kuruczdir;
		char *plugindir, *ldswitch, *ldintern, *kuruczswitch, *yn;
		char prompt[255], defaultdir[255];

		printf ("Please supply names of directories to be used by PHOEBE:\n");
		printf ("  note that you may use tab-completion features to access directories;\n");
		printf ("  if you are happy with defaults enclosed in [...], just press ENTER.\n\n");

		/*************************** BASE DIRECTORY ***************************/

		phoebe_config_entry_get ("PHOEBE_BASE_DIR", &pathname);

		sprintf (prompt, "PHOEBE base directory      [%s]: ", pathname);
		basedir = readline (prompt);
		if (strcmp (basedir, "") == 0) basedir = strdup (pathname);
		if (basedir[strlen(basedir)-1] == '/') basedir[strlen(basedir)-1] = '\0';

		/************************* DEFAULTS DIRECTORY *************************/

		phoebe_config_entry_get ("PHOEBE_DEFAULTS_DIR", &pathname);

		if (PHOEBE_CONFIG_EXISTS) {
			sprintf (prompt, "PHOEBE defaults directory  [%s]: ", pathname);
			defdir = readline (prompt);
			if (strcmp (defdir, "") == 0) defdir = strdup (pathname);
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

		/*************************** TEMP DIRECTORY ***************************/

		phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &pathname);

		sprintf (prompt, "PHOEBE temporary directory [%s]: ", pathname);
		tempdir = readline (prompt);
		if (strcmp (tempdir, "") == 0) tempdir = strdup (pathname);
		if (tempdir[strlen(tempdir)-1] == '/') tempdir[strlen(tempdir)-1] = '\0';

		/*************************** DATA DIRECTORY ***************************/

		phoebe_config_entry_get ("PHOEBE_DATA_DIR", &pathname);

		if (PHOEBE_CONFIG_EXISTS) {
			sprintf (prompt, "PHOEBE data directory      [%s]: ", pathname);
			datadir = readline (prompt);
			if (strcmp (datadir, "") == 0) datadir = strdup (pathname);
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

		/**************************** PTF DIRECTORY ***************************/

		phoebe_config_entry_get ("PHOEBE_PTF_DIR", &pathname);

			if (PHOEBE_CONFIG_EXISTS) {
			sprintf (prompt, "PHOEBE PTF directory       [%s]: ", pathname);
			ptfdir = readline (prompt);
			if (strcmp (ptfdir, "") == 0) ptfdir = strdup (pathname);
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

		/************************* PLUGINS DIRECTORY **************************/

		phoebe_config_entry_get ("PHOEBE_PLUGINS_DIR", &pathname);

		sprintf (prompt, "PHOEBE plugins directory   [%s]: ", pathname);
		plugindir = readline (prompt);
		if (strcmp (plugindir, "") == 0) plugindir = strdup (pathname);
		if (plugindir[strlen(plugindir)-1] == '/') plugindir[strlen(plugindir)-1] = '\0';

		/****************************** LD TABLES *****************************/

		ldswitch = readline ("Are limb darkening tables present on your system [y/N]? ");
		if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') ;
		else {
			ldintern = readline ("Do you want to use internal limb darkening tables [Y/n]? ");

			phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname);
			if (PHOEBE_CONFIG_EXISTS) {
				sprintf (prompt, "PHOEBE LD directory        [%s]: ", pathname);
				lddir = readline (prompt);
				if (strcmp (lddir, "") == 0) lddir = strdup (pathname);
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

		/****************************** KURUCZ DIR ****************************/

		kuruczswitch = readline ("Are Kurucz model atmospheres present on your system [y/N]? ");
		if (strcmp (kuruczswitch, "") == 0 || kuruczswitch[0] == 'n' || kuruczswitch[0] == 'N') ;
		else {
			phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &pathname);
			if (PHOEBE_CONFIG_EXISTS) {
				sprintf (prompt, "PHOEBE Kurucz directory    [%s]: ", pathname);
				kuruczdir = readline (prompt);
				if (strcmp (kuruczdir, "") == 0) kuruczdir = strdup (pathname);
			}
			else {
				sprintf (prompt, "PHOEBE Kurucz directory    [%s/catalogs/kurucz]: ", USER_HOME_DIR);
				kuruczdir = readline (prompt);
				if (strcmp (kuruczdir, "") == 0) {
					sprintf (defaultdir, "%s/catalogs/kurucz", USER_HOME_DIR);
					kuruczdir = strdup (defaultdir);
				}
			}
			if (kuruczdir[strlen(kuruczdir)-1] == '/') kuruczdir[strlen(kuruczdir)-1] = '\0';
		}

		/************************************************************************/

		printf ("\nConfiguration summary:\n----------------------\n");
		printf ("PHOEBE base directory:       %s\n", basedir);
		printf ("PHOEBE defaults directory:   %s\n", defdir);
		printf ("PHOEBE temporary directory:  %s\n", tempdir);
		printf ("PHOEBE data directory:       %s\n", datadir);
		printf ("PHOEBE passbands directory:  %s\n", ptfdir);
		printf ("PHOEBE plugins directory:    %s\n", plugindir);

		if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N')
			printf ("Limb darkening tables:       not present\n");
		else {
			if (strcmp (ldintern, "") == 0 || ldintern[0] == 'y' || ldintern[0] == 'Y')
				printf ("Limb darkening tables:       Kurucz NEWODF\n");
			else
				printf ("Limb darkening tables:       Van Hamme (1993)\n");
			printf ("PHOEBE LD directory:         %s\n", lddir);
		}

		if (strcmp (kuruczswitch, "") == 0 || kuruczswitch[0] == 'n' || kuruczswitch[0] == 'N')
			printf ("Kurucz's model atmospheres:  not present\n");
		else {
			printf ("Kurucz's model atmospheres:  present\n");
			printf ("PHOEBE Kurucz directory:     %s\n", kuruczdir);
		}

		yn = readline ("\nAre these settings ok [Y/n]? ");
		if (strcmp (yn, "") == 0 || yn[0] == 'y' || yn[0] == 'Y') {
			phoebe_config_entry_set ("PHOEBE_BASE_DIR",     basedir);
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", defdir);
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR",     tempdir);
			phoebe_config_entry_set ("PHOEBE_DATA_DIR",     datadir);
			phoebe_config_entry_set ("PHOEBE_PTF_DIR",      ptfdir);
			phoebe_config_entry_set ("PHOEBE_PLUGINS_DIR",  plugindir);

			if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') {
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 0);
			}
			else {
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 1);
				if (strcmp (ldintern, "") == 0 || ldintern[0] == 'y' || ldintern[0] == 'Y')
					phoebe_config_entry_set ("PHOEBE_LD_INTERN", 1);
				else
					phoebe_config_entry_set ("PHOEBE_LD_INTERN", 0);
				phoebe_config_entry_set ("PHOEBE_LD_DIR", lddir);
			}

			if (strcmp (kuruczswitch, "") == 0 || kuruczswitch[0] == 'n' || kuruczswitch[0] == 'N') {
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", 0);
			}
			else {
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", 1);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR", kuruczdir);
			}

			phoebe_config_save (PHOEBE_CONFIG);

			printf ("\nConfiguration file written. If you ever want to change your settings,\n");
			printf ("simply edit %s or start PHOEBE\n", PHOEBE_CONFIG);
			printf ("with -c (--configure) switch.\n");
			return SUCCESS;
		}
		else {
			printf ("\nAborting configuration. Please restart PHOEBE to try again.\n\n");
			exit (0);
		}
	}
#endif

	/* This part is executed if GNU readline isn't found: */
	{
		char basedir[255], defdir[255], tempdir[255], datadir[255], ptfdir[255], lddir[255], kuruczdir[255];
		char plugindir[255], ldswitch[255], ldintern[255], kuruczswitch[255], yn[255];
		
		printf ("Please supply names of directories to be used by PHOEBE:\n");
		printf ("  if you are happy with defaults enclosed in [...], just press ENTER.\n\n");

		/*************************** BASE DIRECTORY ***************************/
		
		phoebe_config_entry_get ("PHOEBE_BASE_DIR", &pathname);
		
		printf ("PHOEBE base directory      [%s]: ", pathname);
		fgets (basedir, 255, stdin); basedir[strlen(basedir)-1] = '\0';
		if (strcmp (basedir, "") == 0) sprintf (basedir, "%s", pathname);
		if (basedir[strlen(basedir)-1] == '/') basedir[strlen(basedir)-1] = '\0';
		
		/************************* DEFAULTS DIRECTORY *************************/
		
		phoebe_config_entry_get ("PHOEBE_DEFAULTS_DIR", &pathname);
		
		if (PHOEBE_CONFIG_EXISTS) {
			printf ("PHOEBE defaults directory  [%s]: ", pathname);
			fgets (defdir, 255, stdin); defdir[strlen(defdir)-1] = '\0';
			if (strcmp (defdir, "") == 0) sprintf (defdir, "%s", pathname);
		}
		else {
			printf ("PHOEBE defaults directory  [%s/defaults]: ", basedir);
			fgets (defdir, 255, stdin); defdir[strlen(defdir)-1] = '\0';
			if (strcmp (defdir, "") == 0) sprintf (defdir, "%s/defaults", basedir);
		}
		if (defdir[strlen(defdir)-1] == '/') defdir[strlen(defdir)-1] = '\0';
		
		/*************************** TEMP DIRECTORY ***************************/
		
		phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &pathname);
		
		printf ("PHOEBE temporary directory [%s]: ", pathname);
		fgets (tempdir, 255, stdin); tempdir[strlen(tempdir)-1] = '\0';
		if (strcmp (tempdir, "") == 0) sprintf (tempdir, "%s", pathname);
		if (tempdir[strlen(tempdir)-1] == '/') tempdir[strlen(tempdir)-1] = '\0';
		
		/*************************** DATA DIRECTORY ***************************/
		
		phoebe_config_entry_get ("PHOEBE_DATA_DIR", &pathname);
		
		if (PHOEBE_CONFIG_EXISTS) {
			printf ("PHOEBE data directory      [%s]: ", pathname);
			fgets (datadir, 255, stdin); datadir[strlen(datadir)-1] = '\0';
			if (strcmp (datadir, "") == 0) sprintf (datadir, "%s", pathname);
		}
		else {
			printf ("PHOEBE data directory      [%s/data]: ", basedir);
			fgets (datadir, 255, stdin); datadir[strlen(datadir)-1] = '\0';
			if (strcmp (datadir, "") == 0) sprintf (datadir, "%s/data", basedir);
		}
		if (datadir[strlen(datadir)-1] == '/') datadir[strlen(datadir)-1] = '\0';
		
		/**************************** PTF DIRECTORY ***************************/
		
		phoebe_config_entry_get ("PHOEBE_PTF_DIR", &pathname);
		
		if (PHOEBE_CONFIG_EXISTS) {
			printf ("PHOEBE PTF directory       [%s]: ", pathname);
			fgets (ptfdir, 255, stdin); ptfdir[strlen(ptfdir)-1] = '\0';
			if (strcmp (ptfdir, "") == 0) sprintf (ptfdir, "%s", pathname);
		}
		else {
			printf ("PHOEBE PTF directory       [%s/ptf]: ", basedir);
			fgets (ptfdir, 255, stdin); ptfdir[strlen(ptfdir)-1] = '\0';
			if (strcmp (ptfdir, "") == 0) sprintf (ptfdir, "%s/ptf", basedir);
		}
		if (ptfdir[strlen(datadir)-1] == '/') ptfdir[strlen(datadir)-1] = '\0';
		
		/************************* PLUGINS DIRECTORY **************************/
		
		phoebe_config_entry_get ("PHOEBE_PLUGINS_DIR", &pathname);
		
		printf ("PHOEBE plugins directory   [%s]: ", pathname);
		fgets (plugindir, 255, stdin); plugindir[strlen(plugindir)-1] = '\0';
		if (strcmp (plugindir, "") == 0) sprintf (plugindir, "%s", pathname);
		if (plugindir[strlen(plugindir)-1] == '/') plugindir[strlen(plugindir)-1] = '\0';

		/****************************** LD TABLES *****************************/
		
		printf ("Are limb darkening tables present on your system [y/N]? ");
		fgets (ldswitch, 255, stdin); ldswitch[strlen(ldswitch)-1] = '\0';
		
		if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') ;
		else {
			printf ("Do you want to use internal limb darkening tables [Y/n]? ");
			fgets (ldintern, 255, stdin); ldintern[strlen(ldintern)-1] = '\0';

			if (PHOEBE_CONFIG_EXISTS) {
				phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname);
				printf ("PHOEBE LD directory        [%s]: ", pathname);
				fgets (lddir, 255, stdin); lddir[strlen(lddir)-1] = '\0';
				if (strcmp (lddir, "") == 0) sprintf (lddir, "%s", pathname);
			}
			else {
				printf ("PHOEBE LD directory        [%s/catalogs/ld]: ", USER_HOME_DIR);
				fgets (lddir, 255, stdin); lddir[strlen(lddir)-1] = '\0';
				if (strcmp (lddir, "") == 0) sprintf (lddir, "%s/catalogs/ld", USER_HOME_DIR);
			}
			if (lddir[strlen(lddir)-1] == '/') lddir[strlen(lddir)-1] = '\0';
		}
		
		/****************************** KURUCZ DIR ****************************/
		
		printf ("Are Kurucz model atmospheres present on your system [y/N]? ");
		fgets (kuruczswitch, 255, stdin); kuruczswitch[strlen(kuruczswitch)-1] = '\0';
		
		if (strcmp (kuruczswitch, "") == 0 || kuruczswitch[0] == 'n' || kuruczswitch[0] == 'N') ;
		else {
			if (PHOEBE_CONFIG_EXISTS) {
				phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &pathname);
				printf ("PHOEBE Kurucz directory:   [%s]: ", pathname);
				fgets (kuruczdir, 255, stdin); kuruczdir[strlen(kuruczdir)-1] = '\0';
				if (strcmp (kuruczdir, "") == 0) sprintf (kuruczdir, "%s", pathname);
			}
			else {
				printf ("PHOEBE Kurucz directory    [%s/catalogs/kurucz]: ", USER_HOME_DIR);
				fgets (kuruczdir, 255, stdin); kuruczdir[strlen(lddir)-1] = '\0';
				if (strcmp (kuruczdir, "") == 0) sprintf (kuruczdir, "%s/catalogs/ld", USER_HOME_DIR);
			}
			if (kuruczdir[strlen(kuruczdir)-1] == '/') kuruczdir[strlen(kuruczdir)-1] = '\0';
		}
		
		/**********************************************************************/
		
		printf ("\nConfiguration summary:\n----------------------\n");
		printf ("PHOEBE base directory:       %s\n", basedir);
		printf ("PHOEBE defaults directory:   %s\n", defdir);
		printf ("PHOEBE temporary directory:  %s\n", tempdir);
		printf ("PHOEBE data directory:       %s\n", datadir);
		printf ("PHOEBE passbands directory:  %s\n", ptfdir);
		printf ("PHOEBE plugins directory:    %s\n", plugindir);
		
		if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N')
			printf ("Limb darkening tables:       not present\n");
		else {
			if (strcmp (ldintern, "") == 0 || ldintern[0] == 'y' || ldintern[0] == 'Y')
				printf ("Limb darkening tables:       Kurucz NEWODF\n");
			else
				printf ("Limb darkening tables:       Van Hamme (1993)\n");
			printf ("PHOEBE LD directory:         %s\n", lddir);
		}
		
		if (strcmp (kuruczswitch, "") == 0 || kuruczswitch[0] == 'n' || kuruczswitch[0] == 'N')
			printf ("Kurucz's model atmospheres:  not present\n");
		else {
			printf ("Kurucz's model atmospheres:  present\n");
			printf ("PHOEBE Kurucz directory:     %s\n", kuruczdir);
		}
		
		printf ("\nAre these settings ok [Y/n]? ");
		fgets (yn, 255, stdin); yn[strlen(yn)-1] = '\0';
		
		if (strcmp (yn, "") == 0 || yn[0] == 'y' || yn[0] == 'Y') {
			phoebe_config_entry_set ("PHOEBE_BASE_DIR",     basedir);
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", defdir);
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR",     tempdir);
			phoebe_config_entry_set ("PHOEBE_DATA_DIR",     datadir);
			phoebe_config_entry_set ("PHOEBE_PTF_DIR",      ptfdir);
			phoebe_config_entry_set ("PHOEBE_PLUGINS_DIR",  plugindir);
			
			if (strcmp (ldswitch, "") == 0 || ldswitch[0] == 'n' || ldswitch[0] == 'N') {
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 0);
			}
			else {
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 1);
				if (strcmp (ldintern, "") == 0 || ldintern[0] == 'y' || ldintern[0] == 'Y')
					phoebe_config_entry_set ("PHOEBE_LD_INTERN", 1);
				else
					phoebe_config_entry_set ("PHOEBE_LD_INTERN", 0);
				phoebe_config_entry_set ("PHOEBE_LD_DIR", lddir);
			}
			
			if (strcmp (kuruczswitch, "") == 0 || kuruczswitch[0] == 'n' || kuruczswitch[0] == 'N') {
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", 0);
			}
			else {
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", 1);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR", kuruczdir);
			}
			
			phoebe_config_save (PHOEBE_CONFIG);
			
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
