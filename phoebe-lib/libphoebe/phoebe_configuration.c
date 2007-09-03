#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_accessories.h"
#include "phoebe_build_config.h"
#include "phoebe_configuration.h"
#include "phoebe_error_handling.h"

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

int phoebe_init_config_entries ()
{
	/**
	 * phoebe_init_config_entries:
	 *
	 * Initializes all configuration keywords and their default values.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	PHOEBE_config_table = NULL;
	PHOEBE_config_table_size = 0;

	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_BASE_DIR",      "/usr/local/share/phoebe");
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_SOURCE_DIR",    "/usr/local/src/phoebe");
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_DEFAULTS_DIR",  "/usr/local/share/phoebe/defaults");
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_TEMP_DIR",      "/tmp");
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_DATA_DIR",      "/usr/local/share/phoebe/data");
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_PTF_DIR",       "/usr/local/share/phoebe/ptf");

	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_LD_SWITCH",     FALSE);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_LD_DIR",        "/usr/local/share/phoebe/ld");

	phoebe_config_entry_add (TYPE_BOOL,   "PHOEBE_KURUCZ_SWITCH", FALSE);
	phoebe_config_entry_add (TYPE_STRING, "PHOEBE_KURUCZ_DIR",    "/usr/local/share/phoebe/kurucz");

/*
	PHOEBE_PLOTTING_PACKAGE = strdup ("");
	PHOEBE_3D_PLOT_CALLBACK_OPTION = 0;
	PHOEBE_CONFIRM_ON_SAVE = 1;
	PHOEBE_CONFIRM_ON_QUIT = 1;
	PHOEBE_WARN_ON_SYNTHETIC_SCATTER = 1;
*/

	return SUCCESS;
}

PHOEBE_config_entry *phoebe_config_entry_new ()
{
	PHOEBE_config_entry *entry = phoebe_malloc (sizeof (*entry));

	entry->keyword = NULL;
	entry->value   = NULL;
	entry->defval  = NULL;

	return entry;
}

int phoebe_config_entry_add (PHOEBE_type type, char *keyword, ...)
{
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

	if (!PHOEBE_config_table[i]->value)
		PHOEBE_config_table[i]->value = phoebe_malloc (sizeof (*(PHOEBE_config_table[i]->value)));

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
	if (!entry)
		return SUCCESS;

	if (entry->keyword)
		free (entry->keyword);

#warning SWITCH_ON_ANYTYPE_FIELDS_FOR_VALUE_AND_DEFVAL_HERE

	free (entry);

	return SUCCESS;
}

int intern_config_parse (char *line)
{
	int i;
	char *entry, *delim, *keyword;

	delim = strchr (line, '=');
	if (!delim)
		return ERROR_PHOEBE_CONFIG_INVALID_LINE;

	entry = line;
	while (entry[0] == ' ' || entry[0] == '\t') entry++;

	keyword = phoebe_malloc (strlen (entry) - strlen (delim) + 1);
	strncpy (keyword, entry, strlen (entry) - strlen (delim));
	keyword[strlen (entry) - strlen (delim)] = '\0';

	while (keyword[strlen(keyword)-1] == ' ' || keyword[strlen(keyword)-1] == '\t')
		keyword[strlen(keyword)-1] = '\0';

	for (i = 0; i <= PHOEBE_config_table_size; i++) {
		if (i == PHOEBE_config_table_size) {
			phoebe_lib_error ("keyword %s in configuration file not recognized, skipping.\n", keyword);
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
			if (sscanf (delim, "= %s", value) != 1) {
				free (keyword);
				return ERROR_PHOEBE_CONFIG_INVALID_LINE;
			}
			phoebe_config_entry_set (keyword, value);
		}
		break;
		default:
			phoebe_lib_error ("invalid type passed to intern_config_parse (), ignoring.\n");
	}

	free (keyword);

	return SUCCESS;
}

int phoebe_config_load (char *filename)
{
	/**
	 * phoebe_configuration_load:
	 *
	 * @filename: configuration filename.
	 *
	 * Opens a configuration file @filename and reads out all configuration
	 * fields.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status;
	FILE *config;

	char keyword_str[255];

	if (!filename_exists (filename)) {
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
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			continue;
		}
	}

	fclose (config);

	return SUCCESS;
}

int phoebe_config_save (char *filename)
{
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

int phoebe_config_import ()
{
	return SUCCESS;
}

int phoebe_config_check ()
{
	return SUCCESS;
}
