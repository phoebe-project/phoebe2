#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_accessories.h"
#include "phoebe_build_config.h"
#include "phoebe_configuration.h"
#include "phoebe_error_handling.h"

#ifdef __MINGW32__
#include <io.h>
#endif

PHOEBE_config_entry **PHOEBE_config_table;
int                   PHOEBE_config_table_size;

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
char *PHOEBE_PLOTTING_PACKAGE;

char *PHOEBE_INPUT_LOCALE;

int PHOEBE_3D_PLOT_CALLBACK_OPTION;
int PHOEBE_CONFIRM_ON_SAVE;
int PHOEBE_CONFIRM_ON_QUIT;
int PHOEBE_WARN_ON_SYNTHETIC_SCATTER;

int phoebe_config_populate ()
{
	/**
	 * phoebe_config_populate:
	 *
	 * Initializes all configuration keywords and their default values and
	 * adds them to the global #PHOEBE_config_table. All drivers need to
	 * implement their own *_config_populate () function analogous to this
	 * function.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	char buffer[255];

#ifdef __MINGW32__
	/* Windows: */
	char path[255];
	getcwd(path, sizeof(path));

	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_BASE_DIR",      path);
	sprintf(buffer, "%s\\defaults", path);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_DEFAULTS_DIR",  buffer);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_TEMP_DIR",      getenv("TEMP"));
	sprintf(buffer, "%s\\data", path);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_DATA_DIR",      buffer);
	sprintf(buffer, "%s\\ptf", path);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_PTF_DIR",       buffer);

	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_LD_SWITCH",     TRUE);
	sprintf(buffer, "%s\\ld", path);
	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_LD_INTERN",     TRUE);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_LD_DIR",        buffer);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_LD_VH_DIR",     buffer);

	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_KURUCZ_SWITCH", FALSE);
	sprintf(buffer, "%s\\kurucz", path);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_KURUCZ_DIR",    buffer);
	sprintf(buffer, "%s\\plugins", path);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_PLUGINS_DIR",    buffer);
#else
	/* Linux & Mac: */
	sprintf (buffer, "%s/share/phoebe", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_BASE_DIR",      buffer);
	sprintf (buffer, "%s/share/phoebe/defaults", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_DEFAULTS_DIR",  buffer);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_TEMP_DIR",      "/tmp");
	sprintf (buffer, "%s/share/phoebe/data", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_DATA_DIR",      buffer);
	sprintf (buffer, "%s/share/phoebe/ptf", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_PTF_DIR",       buffer);

	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_LD_SWITCH",     FALSE);
	sprintf (buffer, "%s/share/phoebe/ld", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_LD_INTERN",     TRUE);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_LD_DIR",        buffer);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_LD_VH_DIR",     buffer);

	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_KURUCZ_SWITCH", FALSE);
	sprintf (buffer, "%s/share/phoebe/kurucz", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_KURUCZ_DIR",    buffer);

	sprintf (buffer, "%s/lib/phoebe/plugins", PHOEBE_TOP_DIR);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_PLUGINS_DIR",   buffer);
#endif

	return SUCCESS;
}

int phoebe_config_free ()
{
	/**
	 * phoebe_config_free:
	 *
	 * Frees all configuration entries in the global #PHOEBE_config_table.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	for (i = 0; i < PHOEBE_config_table_size; i++)
		phoebe_config_entry_free (PHOEBE_config_table[i]);

	free (PHOEBE_config_table);

	return SUCCESS;
}

PHOEBE_config_entry *phoebe_config_entry_new ()
{
	/**
	 * phoebe_config_entry_new:
	 *
	 * Initializes memory for #PHOEBE_config_entry and NULLifies all structure
	 * pointers. The memory should be freed by phoebe_config_entry_free() once
	 * the entry is no longer necessary.
	 *
	 * Returns: a newly allocated #PHOEBE_config_entry.
	 */

	PHOEBE_config_entry *entry = phoebe_malloc (sizeof (*entry));

	entry->keyword = NULL;
	entry->value   = NULL;
	entry->defval  = NULL;

	return entry;
}

int phoebe_config_entry_add (PHOEBE_type type, char *keyword, ...)
{
	/**
	 * phoebe_config_entry_add:
	 * @type: a #PHOEBE_type of the configuration entry; %TYPE_INT, %TYPE_BOOL,
	 *        %TYPE_DOUBLE and %TYPE_STRING are supported.
	 * @keyword: a string identifying the configuration entry.
	 * @...: the default value of the configuration entry; the type of the
	 *       argument should match @type.
	 *
	 * Adds a new configuration entry to the #PHOEBE_config_table and bumps
	 * the number of configuration entries #PHOEBE_config_table_size (a
	 * global variable) by 1. phoebe_config_entry_add() is the only function
	 * that should be used to add entries to the configuration table.
	 *
	 * Examples:
	 *
	 * |[
	 * status = phoebe_config_entry_add (PHOEBE_STRING, "PHOEBE_BASE_DIR", "/usr/local/share/phoebe");
	 * status = phoebe_config_entry_add (PHOEBE_BOOL,   "PHOEBE_LD_SWITCH", 1); 
	 * ]|
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	va_list arg;
	PHOEBE_config_entry *entry = phoebe_config_entry_new ();

	entry->defval = phoebe_malloc (sizeof (*(entry->defval)));

	entry->type = type;
	entry->keyword = strdup (keyword);

	va_start (arg, keyword);

	switch (entry->type) {
		case TYPE_INT:
			entry->defval->i = va_arg (arg, int);
		break;
		case TYPE_BOOL:
			entry->defval->b = va_arg (arg, bool);
		break;
		case TYPE_DOUBLE:
			entry->defval->d = va_arg (arg, double);
		break;
		case TYPE_STRING: {
			char *str = va_arg (arg, char *);
			entry->defval->str = phoebe_malloc (strlen (str) + 1);
			strcpy (entry->defval->str, str);
		}
		break;
		default:
			phoebe_lib_error ("invalid type encountered in phoebe_config_entry_add ().\n");
			return ERROR_PHOEBE_CONFIG_ENTRY_INVALID_TYPE;
	}
	va_end (arg);

	PHOEBE_config_table_size++;
	PHOEBE_config_table = phoebe_realloc (PHOEBE_config_table, PHOEBE_config_table_size * sizeof (*PHOEBE_config_table));
	PHOEBE_config_table[PHOEBE_config_table_size-1] = entry;

	return SUCCESS;
}

int phoebe_config_entry_set (char *keyword, ...)
{
	/**
	 * phoebe_config_entry_set:
	 * @keyword: a string identifying the configuration entry.
	 * @...: a value of the type that matches @keyword type.
	 *
	 * Sets the configuration option @keyword to the passed value.
	 *
	 * Example:
	 *
	 * |[
	 * status = phoebe_config_entry_set ("PHOEBE_TEMP_DIR", "/tmp");
	 * status = phoebe_config_entry_set ("PHOEBE_LD_SWITCH", 0);
	 * ]|
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	va_list arg;

	for (i = 0; i <= PHOEBE_config_table_size; i++) {
		if (i == PHOEBE_config_table_size) {
			phoebe_lib_error ("configuration keyword %s is invalid, please report this!\n", keyword);
			return ERROR_PHOEBE_CONFIG_INVALID_KEYWORD;
		}
		if (strcmp (keyword, PHOEBE_config_table[i]->keyword) == 0)
			break;
	}

	va_start (arg, keyword);

	if (!PHOEBE_config_table[i]->value) {
		PHOEBE_config_table[i]->value = phoebe_malloc (sizeof (*(PHOEBE_config_table[i]->value)));
		PHOEBE_config_table[i]->value->str = NULL;
	}

	switch (PHOEBE_config_table[i]->type) {
		case TYPE_INT:
			PHOEBE_config_table[i]->value->i = va_arg (arg, int);
		break;
		case TYPE_BOOL:
			PHOEBE_config_table[i]->value->b = va_arg (arg, bool);
		break;
		case TYPE_DOUBLE:
			PHOEBE_config_table[i]->value->d = va_arg (arg, double);
		break;
		case TYPE_STRING: {
			char *str = va_arg (arg, char *);
			if (PHOEBE_config_table[i]->value->str)
				free (PHOEBE_config_table[i]->value->str);
			PHOEBE_config_table[i]->value->str = phoebe_malloc (strlen (str) + 1);
			strcpy (PHOEBE_config_table[i]->value->str, str);
		}
		break;
		default:
			phoebe_lib_error ("invalid type encountered in phoebe_config_entry_add ().\n");
			return ERROR_PHOEBE_CONFIG_ENTRY_INVALID_TYPE;
	}
	va_end (arg);

	return SUCCESS;
}

int phoebe_config_entry_get (char *keyword, ...)
{
	/**
	 * phoebe_config_entry_get:
	 * @keyword: a string identifying the configuration entry.
	 * @...: a reference to the value of the type that matches @keyword type.
	 *
	 * Assigns the configuration option @keyword to the passed value.
	 *
	 * Example:
	 *
	 * |[
	 * char *tmpdir;
	 * int ldstate;
	 *
	 * status = phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);
	 * status = phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &ldstate);
	 * ]|
	 *
	 * Do not free the contents of so-assigned strings - these merely point
	 * to the actual values in the configuration table.
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	va_list arg;

	for (i = 0; i <= PHOEBE_config_table_size; i++) {
		if (i == PHOEBE_config_table_size) {
			phoebe_lib_error ("configuration keyword %s is invalid, please report this!\n", keyword);
			return ERROR_PHOEBE_CONFIG_INVALID_KEYWORD;
		}
		if (strcmp (keyword, PHOEBE_config_table[i]->keyword) == 0)
			break;
	}

	va_start (arg, keyword);

	switch (PHOEBE_config_table[i]->type) {
		case TYPE_INT:
			if (PHOEBE_config_table[i]->value)
				*(va_arg (arg, int *)) = PHOEBE_config_table[i]->value->i;
			else
				*(va_arg (arg, int *)) = PHOEBE_config_table[i]->defval->i;
		break;
		case TYPE_BOOL:
			if (PHOEBE_config_table[i]->value)
				*(va_arg (arg, bool *)) = PHOEBE_config_table[i]->value->b;
			else
				*(va_arg (arg, bool *)) = PHOEBE_config_table[i]->defval->b;
		break;
		case TYPE_DOUBLE:
			if (PHOEBE_config_table[i]->value)
				*(va_arg (arg, double *)) = PHOEBE_config_table[i]->value->d;
			else
				*(va_arg (arg, double *)) = PHOEBE_config_table[i]->defval->d;
		break;
		case TYPE_STRING:
			if (PHOEBE_config_table[i]->value)
				*(va_arg (arg, char **)) = PHOEBE_config_table[i]->value->str;
			else
				*(va_arg (arg, char **)) = PHOEBE_config_table[i]->defval->str;
		break;
		default:
			phoebe_lib_error ("invalid type encountered in phoebe_config_entry_get ().\n");
			return ERROR_PHOEBE_CONFIG_ENTRY_INVALID_TYPE;
	}
	va_end (arg);

	return SUCCESS;
}

int phoebe_config_entry_free (PHOEBE_config_entry *entry)
{
	/**
	 * phoebe_config_entry_free:
	 * @entry: a pointer to the configuration entry that should be freed.
	 *
	 * Frees the memory occupied by the configuration entry @entry.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!entry)
		return SUCCESS;

	if (entry->keyword)
		free (entry->keyword);

	switch (entry->type) {
		case TYPE_INT:
			if (entry->value)  free (entry->value);
			if (entry->defval) free (entry->defval);
		break;
		case TYPE_BOOL:
			if (entry->value)  free (entry->value);
			if (entry->defval) free (entry->defval);
		break;
		case TYPE_DOUBLE:
			if (entry->value)  free (entry->value);
			if (entry->defval) free (entry->defval);
		break;
		case TYPE_STRING:
			if (entry->value) {
				if (entry->value->str)
					free (entry->value->str);
				free (entry->value);
			}
			if (entry->defval) {
				if (entry->defval->str)
					free (entry->defval->str);
				free (entry->defval);
			}
		break;
		default:
			phoebe_lib_error ("invalid type encountered in phoebe_config_entry_free ().\n");
			return ERROR_PHOEBE_CONFIG_ENTRY_INVALID_TYPE;
	}

	free (entry);

	return SUCCESS;
}

int intern_config_parse (char *line)
{
	/*
	 * This is the internal function that parses the line read from the
	 * configuration file. It also sets the value of corresponding keywords.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	char *entry, *delim, *keyword;

	delim = strchr (line, '=');
	if (!delim)
		return ERROR_PHOEBE_CONFIG_LEGACY_FILE;

	entry = line;
	while (entry[0] == ' ' || entry[0] == '\t') entry++;

	keyword = phoebe_malloc (strlen (entry) - strlen (delim) + 1);
	strncpy (keyword, entry, strlen (entry) - strlen (delim));
	keyword[strlen (entry) - strlen (delim)] = '\0';

	while (keyword[strlen(keyword)-1] == ' ' || keyword[strlen(keyword)-1] == '\t')
		keyword[strlen(keyword)-1] = '\0';

	for (i = 0; i <= PHOEBE_config_table_size; i++) {
		if (i == PHOEBE_config_table_size) {
			phoebe_lib_warning ("keyword %s in configuration file not recognized, skipping.\n", keyword);
			free (keyword);
			return ERROR_PHOEBE_CONFIG_INVALID_KEYWORD;
		}
		if (strcmp (keyword, PHOEBE_config_table[i]->keyword) == 0)
			break;
	}

	switch (PHOEBE_config_table[i]->type) {
		case TYPE_INT: {
			int value;
			if (sscanf (delim, "= %d", &value) != 1) {
				free (keyword);
				return ERROR_PHOEBE_CONFIG_INVALID_LINE;
			}
			phoebe_config_entry_set (keyword, value);
		}
		break;
		case TYPE_BOOL: {
			int value;
			if (sscanf (delim, "= %d", &value) != 1) {
				free (keyword);
				return ERROR_PHOEBE_CONFIG_INVALID_LINE;
			}
			phoebe_config_entry_set (keyword, value);
		}
		break;
		case TYPE_DOUBLE: {
			double value;
			if (sscanf (delim, "= %lf", &value) != 1) {
				free (keyword);
				return ERROR_PHOEBE_CONFIG_INVALID_LINE;
			}
			phoebe_config_entry_set (keyword, value);
		}
		break;
		case TYPE_STRING: {
			char value[255];
#ifdef __MINGW32__
			// Spaces in a directory name must be kept on Windows
			strcpy(value, delim + 2);
			while (strchr("\t \r\n", value[strlen(value) - 1]) != NULL)
				(value[strlen(value) - 1] = '\0');
#else
			if (sscanf (delim, "= %s", value) != 1) {
				free (keyword);
				return ERROR_PHOEBE_CONFIG_INVALID_LINE;
			}
#endif
			phoebe_debug("keyword = <%s>, value = <%s>\n", keyword, value);
			phoebe_config_entry_set (keyword, value);
		}
		break;
		default:
			phoebe_lib_error ("invalid type passed to intern_config_parse (), ignoring.\n");
	}

	free (keyword);

	return SUCCESS;
}

int phoebe_config_peek (char *filename)
{
	/**
	 * phoebe_config_peek:
	 * @filename: configuration filename
	 *
	 * Use this function to peek into the configuration file. The function
	 * returns #SUCCESS if the configuration file exists and conforms to the
	 * current version specs, #ERROR_PHOEBE_CONFIG_LEGACY_FILE if the file
	 * exists but conforms to the earlier (legacy) versions of PHOEBE, and it
	 * returns #ERROR_PHOEBE_CONFIG_NOT_FOUND or #ERROR_PHOEBE_CONFIG_OPEN_FAILED
	 * if the file is not found or cannot be opened.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	FILE *config;
	char entry[255];

	if (!phoebe_filename_exists (filename))
		return ERROR_PHOEBE_CONFIG_NOT_FOUND;

	config = fopen (filename, "r");
	if (!config)
		return ERROR_PHOEBE_CONFIG_OPEN_FAILED;

	if (!fgets (entry, 255, config))
		return ERROR_PHOEBE_CONFIG_OPEN_FAILED;

	if (!strchr (entry, '='))
		return ERROR_PHOEBE_CONFIG_LEGACY_FILE;

	fclose (config);

	return SUCCESS;
}

int phoebe_config_load (char *filename)
{
	/**
	 * phoebe_config_load:
	 * @filename: configuration filename.
	 *
	 * Opens a configuration file @filename and reads out all configuration
	 * fields. The file has to conform to the current PHOEBE version (0.30)
	 * specifications, namely KEYWORD = VALUE entries.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status;
	FILE *config;

	char keyword_str[255];

	if (!phoebe_filename_exists (filename)) {
		PHOEBE_CONFIG_EXISTS = 0;
		return ERROR_PHOEBE_CONFIG_NOT_FOUND;
	}

	config = fopen (filename, "r");
	if (!config) {
		PHOEBE_CONFIG_EXISTS = 0;
		return ERROR_PHOEBE_CONFIG_OPEN_FAILED;
	}

	PHOEBE_CONFIG_EXISTS = 1;

	while (!feof (config)) {
		if (!fgets (keyword_str, 254, config)) break;
		status = intern_config_parse (keyword_str);

		if (status != SUCCESS && status != ERROR_PHOEBE_CONFIG_INVALID_KEYWORD) {
			phoebe_lib_error ("%s", phoebe_error (status));
			continue;
		}
	}

	fclose (config);

	return SUCCESS;
}

int phoebe_config_save (char *filename)
{
	/**
	 * phoebe_config_save:
	 * @filename: configuration filename.
	 *
	 * Saves all entries in the global configuration table #PHOEBE_config_table
	 * to a configuration file @filename.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	FILE *config;

	config = fopen (filename, "w");
	if (!config) {
		PHOEBE_CONFIG_EXISTS = 0;
		return ERROR_PHOEBE_CONFIG_OPEN_FAILED;
	}

	for (i = 0; i < PHOEBE_config_table_size; i++) {
		switch (PHOEBE_config_table[i]->type) {
			case TYPE_INT:
				if (PHOEBE_config_table[i]->value)
					fprintf (config, "%-20s = %d\n",  PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->value->i);
				else
					fprintf (config, "%-20s = %d\n",  PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->defval->i);
			break;
			case TYPE_BOOL:
				if (PHOEBE_config_table[i]->value)
					fprintf (config, "%-20s = %d\n",  PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->value->b);
				else
					fprintf (config, "%-20s = %d\n",  PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->defval->b);
			break;
			case TYPE_DOUBLE:
				if (PHOEBE_config_table[i]->value)
					fprintf (config, "%-20s = %lf\n", PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->value->d);
				else
					fprintf (config, "%-20s = %lf\n", PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->defval->d);
			break;
			case TYPE_STRING:
				if (PHOEBE_config_table[i]->value)
					fprintf (config, "%-20s = %s\n",  PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->value->str);
				else
					fprintf (config, "%-20s = %s\n",  PHOEBE_config_table[i]->keyword, PHOEBE_config_table[i]->defval->str);
			break;
			default:
				phoebe_lib_error ("invalid type passed to phoebe_config_save (), please report this!\n");
		}
	}

	fclose (config);

	return SUCCESS;
}

int phoebe_config_import (char *filename)
{
	/**
	 * phoebe_config_import:
	 *
	 * Opens a pre-0.30 configuration file and reads out the entries. The
	 * configuration file had a hard-coded set of keywords that we check for
	 * one by one.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	FILE *config;
	char keyword_str[255];
	char working_str[255];
	int readint;

	if (!phoebe_filename_exists (filename)) {
		PHOEBE_CONFIG_EXISTS = 0;
		return ERROR_PHOEBE_CONFIG_NOT_FOUND;
	}

	config = fopen (filename, "r");
	if (!config) {
		PHOEBE_CONFIG_EXISTS = 0;
		return ERROR_PHOEBE_CONFIG_OPEN_FAILED;
	}

	PHOEBE_CONFIG_EXISTS = 1;

	while (!feof (config)) {
		if (!fgets (keyword_str, 254, config)) break;

		if (strstr (keyword_str, "PHOEBE_BASE_DIR"))
			if (sscanf (keyword_str, "PHOEBE_BASE_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_BASE_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_DEFAULTS_DIR"))
			if (sscanf (keyword_str, "PHOEBE_DEFAULTS_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_TEMP_DIR"))
			if (sscanf (keyword_str, "PHOEBE_TEMP_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_TEMP_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_DATA_DIR"))
			if (sscanf (keyword_str, "PHOEBE_DATA_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_DATA_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_PTF_DIR"))
			if (sscanf (keyword_str, "PHOEBE_PTF_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_PTF_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_FF_DIR"))
			if (sscanf (keyword_str, "PHOEBE_FF_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_PTF_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_LD_SWITCH"))
			if (sscanf (keyword_str, "PHOEBE_LD_SWITCH %d", &readint) == 1)
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", readint);

		if (strstr (keyword_str, "PHOEBE_LD_DIR"))
			if (sscanf (keyword_str, "PHOEBE_LD_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_LD_DIR", working_str);

		if (strstr (keyword_str, "PHOEBE_KURUCZ_SWITCH"))
			if (sscanf (keyword_str, "PHOEBE_KURUCZ_SWITCH %d", &readint) == 1)
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", readint);

		if (strstr (keyword_str, "PHOEBE_KURUCZ_DIR"))
			if (sscanf (keyword_str, "PHOEBE_KURUCZ_DIR %s", working_str) == 1)
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR", working_str);

		/*
		 * The following obsolete entries are ignored; please handle them
		 * properly within suitable drivers.
		 *
		 *   PHOEBE_LC_DIR
		 *   PHOEBE_DC_DIR
		 *   PHOEBE_PLOTTING_PACKAGE
		 *   PHOEBE_3D_PLOT_CALLBACK_OPTION
		 *   PHOEBE_CONFIRM_ON_SAVE
		 *   PHOEBE_CONFIRM_ON_QUIT
		 *   PHOEBE_WARN_ON_SYNTHETIC_SCATTER
		 */
	}
	fclose (config);

	return SUCCESS;
}
