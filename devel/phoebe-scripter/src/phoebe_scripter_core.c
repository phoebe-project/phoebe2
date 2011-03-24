#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ltdl.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_build_config.h"
#include "phoebe_scripter_commands.h"
#include "phoebe_scripter_core.h"
#include "phoebe_scripter_directives.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter.lng.h"
#include "phoebe_scripter_io.h"
#include "phoebe_scripter_types.h"

#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
	#include <readline/readline.h>
	#include <readline/history.h>
#endif

YY_BUFFER_STATE main_thread;
PHOEBE_COMMAND_LINE_ARGS PHOEBE_args;
bool PHOEBE_INTERRUPT;

int parse_startup_line (int argc, char *argv[])
{
	/*
	 * This function parses the command line and looks for known switches; it
	 * doesn't really do anything with them, it just writes what it found to
	 * the PHOEBE_args variable that is supposed to be used for real actions.
	 */

	int i;

	for (i = 1; i < argc; i++) {
		if ( (strcmp (argv[i],  "-h"   ) == 0) ||
		     (strcmp (argv[i],  "-?"   ) == 0) ||
				 (strcmp (argv[i], "--help") == 0) ) {
			printf ("\n%s command line arguments: [-cehv] [keyword_file]\n\n", PHOEBE_VERSION_NUMBER);
			printf ("  -c, --configure     ..  configure PHOEBE\n");
			printf ("  -e, --execute       ..  execute PHOEBE script\n");
			printf ("  -h, --help, -?      ..  this help screen\n");
			printf ("  -v, --version       ..  display PHOEBE version and exit\n");
			printf ("\n");
			printf ("  --config-dir dir/   ..  use dir/ as configuration directory\n");
			printf ("\n");
			phoebe_quit ();
		}

		if ( (strcmp (argv[i],  "-v"      ) == 0) ||
			 (strcmp (argv[i], "--version") == 0) ) {
			printf ("\n%s %s\n", PHOEBE_SCRIPTER_RELEASE_NAME, PHOEBE_SCRIPTER_RELEASE_DATE);
			printf ("  Send comments and/or requests to phoebe-discuss@lists.sourceforge.net\n\n");
			phoebe_quit ();
		}

		if ( (strcmp (argv[i],  "-c"        ) == 0) ||
			 (strcmp (argv[i], "--configure") == 0) ) {
			PHOEBE_args.CONFIGURE_SWITCH = 1;
		}

		if ( (strcmp (argv[i],  "-e"      ) == 0) ||
			 (strcmp (argv[i], "--execute") == 0) ) {
			/* We cannot start the script from here because the scripter isn't ini- */
			/* tialized yet; thus we just initialize the proper switches:           */
			PHOEBE_args.SCRIPT_SWITCH = 1;
			
			/* If someone forgot to put the script name, we should warn him and     */
			/* exit:                                                                */
			if ( (i+1 == argc) || (argv[i+1][0] == '-') ) {
				printf ("\n%s command line arguments: [-hsv] [keyword_file]\n\n", PHOEBE_VERSION_NUMBER);
				printf ("  -e, --execute       ..  execute PHOEBE script\n");
				printf ("  -h, --help, -?      ..  this help screen\n");
				printf ("  -s, --scripter      ..  Run PHOEBE scripter interactively\n");
				printf ("  -v, --version       ..  display PHOEBE version and exit\n");
				printf ("\n");
				phoebe_warning ("there is no argument given to the '-e' switch.\n\n");
				phoebe_quit ();
			}

			/* Otherwise let's read in the script's name:                           */
			PHOEBE_args.SCRIPT_NAME = strdup (argv[i+1]);
			i++;                    /* This will skip the script name. */
			if (i >= argc) break;   /* If this was the last switch, break the loop. */
		}

		if (strcmp (argv[i],  "--config-dir") == 0) {
			PHOEBE_args.CONFIG_DIR = strdup (argv[i+1]);
			i++; i++;
			if (i >= argc) break;   /* If this was the last switch, break the loop. */
		}

		if ( argv[i][0] != '-' ) {
			/* This means that the command line argument doesn't contain '-'; thus  */
			/* it is a parameter file. All other arguments to - and -- switches     */
			/* (without -/--) are read in before and they should never evaluate     */
			/* here.                                                                */

			PHOEBE_args.PARAMETER_SWITCH = 1;
			PHOEBE_args.PARAMETER_FILE = strdup (argv[i]);
		}
	}

	return SUCCESS;
}

int scripter_parameters_init ()
{
	/*
	 * This function initializes all scripter-related parameters and adds them
	 * to the global parameter table.
	 */

	phoebe_parameter_add ("scripter_verbosity_level",          "The level of scripter verbosity",                   KIND_PARAMETER,  NULL, "%d",  0.0,    0.0,    0.0,  NO, TYPE_INT,          1);
	phoebe_parameter_add ("scripter_ordinate_reversed_switch", "Reverse the direction of the ordinate on LC plots", KIND_SWITCH,     NULL, "%d",  0.0,    0.0,    0.0,  NO, TYPE_BOOL,         NO);

	return SUCCESS;
}

int scripter_plugins_init ()
{
	int (*initfunc) ();
	int status;
	lt_dlhandle handle;
	char *path;

	status = phoebe_config_entry_get ("PHOEBE_PLUGINS_DIR", &path);
	if (status != SUCCESS)
		return status;

	lt_dlinit ();

	status = lt_dladdsearchdir (path);
	if (status != SUCCESS)
		return ERROR_PLUGINS_DIR_LOAD_FAILED;

	fprintf (PHOEBE_output, "Loading scripter plugins:\n");

	handle = lt_dlopen ("phoebe_polyfit.la");
	if (!handle)
		phoebe_scripter_output ("* polyfit plugin not found in %s.\n", path);
	else {
		initfunc = lt_dlsym (handle, "phoebe_plugin_start");
		initfunc ();
	}

	fprintf (PHOEBE_output, "\n");

	return SUCCESS;
}

int scripter_init ()
{
	/*
	 * This function initializes all global scripter variables and flags.
	 */

	/* Scripter output device: */
	PHOEBE_output = stdout;

	/* Create a global symbol table and set the current symbol table to point
	 * to it:
	 */
	symbol_table = symbol_table_add (NULL, "global");

	/* Define important constants: */
	scripter_symbol_commit (symbol_table, "CONST_PI",   scripter_ast_add_double (3.14159265359));
	scripter_symbol_commit (symbol_table, "CONST_E",    scripter_ast_add_double (2.71828182846));
	scripter_symbol_commit (symbol_table, "CONST_AU",   scripter_ast_add_double (149597870.691));
	scripter_symbol_commit (symbol_table, "CONST_RSUN", scripter_ast_add_double (696000.0));
	scripter_symbol_commit (symbol_table, "CONST_MSUN", scripter_ast_add_double (1.99E30));

	/* Initialize the main symbol table (used for the main script flow): */
	symbol_table = symbol_table_add (symbol_table, "phoebe_main");

	/* Initialize all scripter-related parameters: */
	scripter_parameters_init ();

	/* Initialize the table of scripter commands: */
	scripter_commands = phoebe_malloc (sizeof (*scripter_commands));
	scripter_commands->no = 0;
	scripter_commands->command = NULL;
	scripter_register_all_commands ();

	/* Initialize the table of builtin functions: */
	scripter_functions = phoebe_malloc (sizeof (*scripter_functions));
	scripter_functions->no = 0;
	scripter_functions->func = NULL;
	scripter_function_register_all ();

	fprintf (PHOEBE_output, "\nThis is %s scripter.\n\n", PHOEBE_VERSION_NUMBER);

	return SUCCESS;
}

int scripter_config_populate ()
{
	/*
	 * This function adds the configuration entries to the config table.
	 * Don't worry about freeing them, the library takes care of that
	 * automatically.
	 */

	phoebe_config_entry_add (TYPE_STRING, "SCRIPTER_BASE_DIR", "/usr/local/share/phoebe_scripter");
	phoebe_config_entry_add (TYPE_STRING, "SCRIPTER_HELP_DIR", "/usr/local/share/phoebe_scripter/help");

	return SUCCESS;
}

int scripter_main_loop ()
{
	/* This function does the line buffering for lexer. All user interaction  */
	/* goes through here. The loop is endless and only quit directive (or a   */
	/* segmentation fault ;) ) breaks it.                                     */

	char prompt[5];
	char *buffer = NULL;
	char *line;

	int i, block_level;

	phoebe_debug ("main scripter loop entered.\n");

	block_level = 0;
	sprintf (prompt, "> ");

#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
	phoebe_debug ("initializing GNU readline command line history.\n");
	using_history ();
#endif

	do {

#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
		phoebe_debug ("initializing GNU readline prompt.\n");
		line = readline (prompt);
		
		if (line == '\0') {
			printf ("\n");
			free (line);
			scripter_quit ();
			phoebe_quit ();
		}
		else if (strlen(line) > 0)
			add_history (line);
#endif

#if (defined HAVE_LIBREADLINE && defined PHOEBE_READLINE_DISABLED) || !defined HAVE_LIBREADLINE
		phoebe_debug ("initializing built-in (primitive) prompt.\n");
		line = phoebe_malloc (255);
		printf ("%s", prompt);
		fgets (line, 255, stdin);
#endif

		if (line != '\0') {
			for (i = 0; i < strlen(line); i++) {
				if (line[i] == '{') block_level++;
				if (line[i] == '}') block_level--;
			}
			buffer = phoebe_malloc (strlen (line)+2);
			strcpy (buffer, line);
			strcat (buffer, " ");
			free (line);
		}

		while (block_level > 0) {
			sprintf (prompt, "%2d> ", block_level);

#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
			line = readline (prompt);
			if (line == '\0') printf ("\n"), scripter_directive_quit (NULL);
			else if (strlen(line) > 0) add_history (line);
#endif

#if (defined HAVE_LIBREADLINE && defined PHOEBE_READLINE_DISABLED) || !defined HAVE_LIBREADLINE
			line = phoebe_malloc (255);
			printf ("%s", prompt);
			fgets (line, 255, stdin);
#endif

			for (i = 0; i < strlen(line); i++) {
				if (line[i] == '{') block_level++;
				if (line[i] == '}') block_level--;
			}
			
			buffer = phoebe_realloc (buffer, strlen (buffer) + strlen (line) + 2);
			strcat (buffer, line);
			strcat (buffer, " ");
			free (line);
		}

		if (line != 0) {
			main_thread = yy_scan_string (buffer);
			yyparse ();
			yy_delete_buffer (main_thread);
		}

		free (buffer);
		sprintf (prompt, "> ");
	} while (TRUE);

	return SUCCESS;
}

int scripter_execute_script_from_stream (FILE *stream)
{
	char *buffer = NULL;
	char line[255];
	YY_BUFFER_STATE yybuf;

	while (!feof (stream)) {
		fgets (line, 255, stream);
		if (feof (stream)) break;

		if (strlen(line) == 0) continue;
		if (buffer == NULL) {
			buffer = phoebe_malloc (strlen (line)+1);
			strcpy (buffer, line);
		}
		else {
			buffer = phoebe_realloc (buffer, strlen(buffer) + strlen(line) + 1);
			strcat (buffer, line);
		}
	}

	yybuf = yy_scan_string (buffer);
	yyparse ();
	yy_delete_buffer (yybuf);
	free (buffer);

	return SUCCESS;
}

int scripter_execute_script_from_buffer (char *buffer)
{
	/* This is a simple function that calls the lexer/parser on 'buffer'.     */

	YY_BUFFER_STATE yybuf = yy_scan_string (buffer);
	yyparse ();
	yy_delete_buffer (yybuf);
	return SUCCESS;
}

int scripter_quit ()
{
	/* This function cleans up the memory after all scripter elements. */

	/* Program arguments: */
	if (PHOEBE_args.SCRIPT_SWITCH) free (PHOEBE_args.SCRIPT_NAME);
	if (PHOEBE_args.PARAMETER_SWITCH) free (PHOEBE_args.PARAMETER_FILE);

	/* Symbol table: */
	symbol_table_free_all (symbol_table);

	/* Scripter commands: */
	scripter_commands_free_all (scripter_commands);

	/* Scripter builtin functions: */
	scripter_function_free_all (scripter_functions);

	return SUCCESS;	
}
