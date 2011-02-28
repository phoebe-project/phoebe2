#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_build_config.h"

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_core.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter.lng.h"
#include "phoebe_scripter_io.h"
#include "phoebe_scripter_types.h"

int intern_read_in_stream (char *filename, char **buffer)
{
	FILE *stream;
	char line[255];

	stream = fopen (filename, "r");
	while (!feof (stream)) {
		fgets (line, 254, stream);
		if (feof (stream)) break;
		if (!(*buffer)) {
			*buffer = phoebe_malloc (strlen (line)+1);
			strcpy (*buffer, line);
		}
		else {
			*buffer = phoebe_realloc (*buffer, strlen(*buffer) + strlen(line) + 1);
			strcat (*buffer, line);
		}
	}
	fclose (stream);

	return SUCCESS;
}

int scripter_directive_calc (scripter_ast_list *args)
{
	/*
	 * This is the calculation directive, where a single argument - an
	 * expression - is passed.
	 *
	 * Synopsis:
	 *
	 *   calc expr
	 */

	scripter_ast_value *vals;
	int status = scripter_command_args_evaluate (args, &vals, 1, 1, type_any);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return status;
	}

	fprintf (PHOEBE_output, "\t");
	scripter_ast_value_print (vals[0]);
	fprintf (PHOEBE_output, "\n");

	scripter_ast_value_array_free (vals, 1);
	return SUCCESS;
}

int scripter_directive_clear (scripter_ast_list *args)
{
	/*
	 * This directive clears the terminal screen. This should be done only
	 * if the output is in fact stdout (rather than redirected to a file).
	 *
	 * Synopsis:
	 *
	 *   clear
	 */

	int status = scripter_command_args_evaluate (args, NULL, 0, 0);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return status;
	}

	if (PHOEBE_output == stdout) printf ("\e[2J");
	return SUCCESS;
}

int scripter_directive_execute (scripter_ast_list *args)
{
	/*
	 * This directive executes an external script.
	 *
	 * Synopsis:
	 *
	 *   execute "/path/to/script"
	 */

	char *filename;

	YY_BUFFER_STATE lexbuffer;
	char *buffer = NULL;
	scripter_ast_value *vals;
	
	int status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return status;
	}

	filename = vals[0].value.str;

	if (!phoebe_filename_exists (filename)) {
		phoebe_scripter_output ("script '%s' not found.\n", filename);
		scripter_ast_value_array_free (vals, 1);
		return ERROR_FILE_NOT_FOUND;
	}
	if (!phoebe_filename_has_read_permissions (filename)) {
		phoebe_scripter_output ("script '%s' does not have valid permissions, aborting.\n", filename);
		scripter_ast_value_array_free (vals, 1);
		return ERROR_FILE_NO_PERMISSIONS;
	}
	if (phoebe_filename_is_directory (filename)) {
		phoebe_scripter_output ("script '%s' is not a file, aborting.\n", filename);
		scripter_ast_value_array_free (vals, 1);
		return ERROR_FILE_NOT_REGULAR;
	}

	status = intern_read_in_stream (filename, &buffer);
	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 1);
		return status;
	}

	lexbuffer = yy_scan_string (buffer);
	yyparse ();
	yy_delete_buffer (lexbuffer);
	yy_switch_to_buffer (main_thread);
	free (buffer);
	scripter_ast_value_array_free (vals, 1);
	return SUCCESS;
}

int scripter_directive_help (scripter_ast_list *args)
{
	/*
	 * This is the 'help' directive. It is used to get help on a passed
	 * directive, qualifier or 'help' keyword.
	 *
	 * Synopsis:
	 *
	 *   help help
	 *   help directive
	 *   help command
	 *
	 * Since this directive takes an arbitrary argument type, error handling
	 * must be done separately.
	 */

	int status;
	char *helpdir;
	char filename[255], line[255];
	FILE *helpfile;

	if (!args) {
		/* This means generic help directive with no arguments passed.        */
		fprintf (PHOEBE_output, "\n  This is %s scripter. For general help on\n", PHOEBE_VERSION_NUMBER);
		fprintf (PHOEBE_output, "  PHOEBE scripting please refer to PHOEBE API documentation. For online\n");
		fprintf (PHOEBE_output, "  help use 'help directive' or 'help command'. To quit, use 'quit'.\n\n");
		return SUCCESS;
	}

	status = phoebe_config_entry_get ("SCRIPTER_HELP_DIR", &helpdir);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return status;
	}

	if (args->elem->type == ast_int)
		/* This means we have a directive: */
		sprintf (filename, "%s/%s.help", helpdir, scripter_ast_kind_name (args->elem->value.integer));

	if (args->elem->type == ast_string)
		/* This means we have a command: */
		sprintf (filename, "%s/%s.help", helpdir, args->elem->value.string);

	helpfile = fopen (filename, "r");
	if (!helpfile) {
		phoebe_scripter_output ("help file cannot be found.\n");
		phoebe_scripter_output ("It can either be that help hasn't been written yet\n");
		phoebe_scripter_output ("or that the help directory is not properly set.\n");
		return SUCCESS;
	}

	fprintf (PHOEBE_output, "\n");
	while (!feof (helpfile)) {
		fgets (line, 255, helpfile);
		if (feof (helpfile)) break;
		fprintf (PHOEBE_output, "%s", line);
	}
	fclose (helpfile);
	fprintf (PHOEBE_output, "\n");

	return SUCCESS;
}

int scripter_directive_if (scripter_ast_list *args)
{
	/*
	 * This is the IF and IF-ELSE handler.
	 *
	 * Synopsis:
	 *
	 *   if (condition) statement
	 *   if (condition) statement else statement
	 *
	 * Return values:
	 *
	 *   ERROR_ARG_NOT_BOOL
	 *   SUCCESS
	 */

	scripter_ast_value condition  = scripter_ast_evaluate (args->elem);
	scripter_ast *action_if_true  = NULL;
	scripter_ast *action_if_false = NULL;

	if (args->next)       action_if_true  = args->next->elem;
	if (args->next->next) action_if_false = args->next->next->elem;

	/*
	 * The first argument is a condition, so it must be boolean.
	 */

	if (condition.type == type_bool) {
		if (condition.value.b == TRUE)
			scripter_ast_evaluate (action_if_true);
		else if (args->next->next)
			scripter_ast_evaluate (action_if_false);
	} else {
		phoebe_scripter_output ("non-boolean condition encountered, aborting.\n");
		return ERROR_ARG_NOT_BOOL;
	}

	return SUCCESS;
}

int intern_info_on_variables (scripter_ast *ast)
{
	scripter_ast_value val = scripter_ast_evaluate (ast);

	fprintf (PHOEBE_output, "\n");
	fprintf (PHOEBE_output, "  Variable name:       %s\n", ast->value.variable);

	switch (val.type) {
		case type_int:
			fprintf (PHOEBE_output, "  Variable type:       integer\n");
		break;
		case type_bool:
			fprintf (PHOEBE_output, "  Variable type:       boolean\n");
		break;
		case type_double:
			fprintf (PHOEBE_output, "  Variable type:       real\n");
		break;
		case type_string:
			fprintf (PHOEBE_output, "  Variable type:       string\n");
		break;
		case type_vector:
			fprintf (PHOEBE_output, "  Variable type:       vector\n");
			fprintf (PHOEBE_output, "  Dimension:           %d\n", val.value.vec->dim);
		break;
		case type_curve: {
			char *type;
			phoebe_curve_type_get_name (val.value.curve->type, &type);
			fprintf (PHOEBE_output, "  Variable type:       curve (%s)\n", type);
			fprintf (PHOEBE_output, "  Dimension:           %d\n", val.value.curve->indep->dim);
			free (type);
		}
		break;
		case type_spectrum:
			fprintf (PHOEBE_output, "  Variable type:       spectrum\n");
			fprintf (PHOEBE_output, "  Dimension:           %d\n", val.value.spectrum->data->bins);
			fprintf (PHOEBE_output, "  Dispersion type:     %s\n", phoebe_spectrum_dispersion_type_get_name (val.value.spectrum->disp));
			if (val.value.spectrum->disp == PHOEBE_SPECTRUM_DISPERSION_LINEAR)
				fprintf (PHOEBE_output, "  Dispersion:          %2.2lf\n", val.value.spectrum->dx);
			else {
				fprintf (PHOEBE_output, "  Resolving power:     %0.0lf\n", val.value.spectrum->R);
				fprintf (PHOEBE_output, "  Sampling power:      %0.0lf\n", val.value.spectrum->Rs);
			}
			fprintf (PHOEBE_output, "  Wavelength interval: [%0.0lf, %0.0lf]\n", val.value.spectrum->data->range[0], val.value.spectrum->data->range[val.value.spectrum->data->bins]);
		break;
		case type_minfeedback:
			fprintf (PHOEBE_output, "  Variable type:       minimizer feedback\n");
		break;
		default:
			phoebe_scripter_output ("exception handler invoked in intern_info_on_variables (code %d), please report this!\n", val.type);
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	fprintf (PHOEBE_output, "\n");

	scripter_ast_value_free (val);
	return SUCCESS;
}

int intern_info_on_qualifiers (scripter_ast *ast)
{
	/*
	 * This internal function prints out the information (directive 'info')
	 * for the passed AST qualifier node. Type check has already been performed
	 * in the directive function itself, so it shouldn't be done here.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int i, status;
	char format[255];
	scripter_ast_value val = scripter_ast_evaluate (ast);
	char *qualifier = val.value.str;

	/* No need to check the parameter validity, it's already been checked. */
	PHOEBE_parameter *par = phoebe_parameter_lookup (qualifier);

	/* Qualifier, keyword and description:                                    */
	fprintf (PHOEBE_output, "\n");
	fprintf (PHOEBE_output, "  Description:    %s\n", par->description);
	fprintf (PHOEBE_output, "  Qualifier:      %s\n", par->qualifier);

	/* Type and value:                                                        */
	switch (par->type) {
		case TYPE_INT: {
			int value;
			status = phoebe_parameter_get_value (par, &value);
			if (status != SUCCESS) return status;
			fprintf (PHOEBE_output, "  Type:           integer\n");
			sprintf (format, "  Value:          %s\n", par->format);
			fprintf (PHOEBE_output, format, value);
		}
		break;
		case TYPE_BOOL: {
			bool value;
			fprintf (PHOEBE_output, "  Type:           boolean\n");
			phoebe_parameter_get_value (par, &value);
			if (value)
				fprintf (PHOEBE_output, "  Value:          yes\n");
			else
				fprintf (PHOEBE_output, "  Value:          no\n");
		}
		break;
		case TYPE_DOUBLE: {
			double value;
			phoebe_parameter_get_value (par, &value);
			fprintf (PHOEBE_output, "  Type:           real\n");
			sprintf (format, "  Value:          %s\n", par->format);
			fprintf (PHOEBE_output, format, value);
		}
		break;
		case TYPE_STRING: {
			const char *value;
			phoebe_parameter_get_value (par, &value);
			fprintf (PHOEBE_output, "  Type:           string\n");
			sprintf (format, "  Value:          %s\n", par->format);
			fprintf (PHOEBE_output, format, value);
		}
		break;
		case TYPE_INT_ARRAY: {
			PHOEBE_array *array = phoebe_array_new_from_qualifier (qualifier);
			fprintf (PHOEBE_output, "  Type:           array of integers\n");
			if (array) {
				fprintf (PHOEBE_output, "  Value:          ");
				phoebe_array_print (array);
				fprintf (PHOEBE_output, "\n");
				phoebe_array_free (array);
			}
			else
				fprintf (PHOEBE_output, "  Value:          <empty array>\n");
		}
		break;
		case TYPE_BOOL_ARRAY: {
			PHOEBE_array *array = phoebe_array_new_from_qualifier (qualifier);
			fprintf (PHOEBE_output, "  Type:           array of booleans\n");
			if (array) {
				fprintf (PHOEBE_output, "  Value:          ");
				phoebe_array_print (array);
				fprintf (PHOEBE_output, "\n");
				phoebe_array_free (array);
			}
			else
				fprintf (PHOEBE_output, "  Value:          <empty array>\n");
		}
		break;
		case TYPE_DOUBLE_ARRAY: {
			PHOEBE_vector *vec = phoebe_vector_new_from_qualifier (qualifier);
			fprintf (PHOEBE_output, "  Type:           array of reals\n");
			if (vec) {
				fprintf (PHOEBE_output, "  Value:          ");
				phoebe_vector_print (vec);
				fprintf (PHOEBE_output, "\n");
				phoebe_vector_free (vec);
			}
			else
				fprintf (PHOEBE_output, "  Value:          <empty array>\n");
		}
		break;
		case TYPE_STRING_ARRAY: {
			PHOEBE_array *array = phoebe_array_new_from_qualifier (qualifier);
			fprintf (PHOEBE_output, "  Type:           array of strings\n");
			if (array) {
				fprintf (PHOEBE_output, "  Value:          ");
				phoebe_array_print (array);
				fprintf (PHOEBE_output, "\n");
				phoebe_array_free (array);
			}
			else
				fprintf (PHOEBE_output, "  Value:          <empty array>\n");
		}
		break;
		default:
			phoebe_scripter_output ("exception handler invoked in intern_info_on_qualifiers (), please report this!\n");
		break;
	}

	/* Next, is it adjustable?                                                */
	if (par->kind == KIND_ADJUSTABLE) {
		fprintf (PHOEBE_output, "  Adjustable:     yes\n");
		fprintf (PHOEBE_output, "  | Marked TBA:   ");
		if (par->tba == TRUE)
			fprintf (PHOEBE_output, "yes\n");
		else
			fprintf (PHOEBE_output, "no\n");
		fprintf (PHOEBE_output, "  | Step size:    %lf\n", par->step);
		fprintf (PHOEBE_output, "  | Lower limit:  %lf\n", par->min);
		fprintf (PHOEBE_output, "  | Upper limit:  %lf\n", par->max);
	}
	else
		fprintf (PHOEBE_output, "  Adjustable:     no\n");

	if (par->kind == KIND_MENU) {
		fprintf (PHOEBE_output, "\n  Available entries:\n");
		for (i = 0; i < par->menu->optno; i++)
			fprintf (PHOEBE_output, "  %2d. %s\n", i+1, par->menu->option[i]);
	}

	/* Finally, an empty line to finish:                                      */
	fprintf (PHOEBE_output, "\n");

	scripter_ast_value_free (val);
	return SUCCESS;
}

int scripter_directive_info (scripter_ast_list *args)
{
	/*
	 * This is the 'info' directive. It takes one argument, a qualifier or a
	 * variable. In case of qualifiers it lists relevant information for the
	 * passed parameter; in case of variables it lists the type and the
	 * dimension (if variable is an array or a spectrum). It always returns
	 * void.
	 *
	 * Synopsis:
	 *
	 *   info qualifier
	 *   info variable
	 */

	scripter_ast_value *vals;
	int status;

	/* First step: verify the number of arguments without actually evaluating */
	/* them (thus type_any):                                                  */
	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_any);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return status;
	}
	scripter_ast_value_array_free (vals, 1);

	/* Next step: switch against the AST type:                                */
	switch (args->elem->type) {
		case ast_variable:
			intern_info_on_variables (args->elem);
			return SUCCESS;
		break;
		case ast_qualifier:
			intern_info_on_qualifiers (args->elem);
			return SUCCESS;
		break;
		default:
			phoebe_scripter_output ("info on that type is currently not implemented, sorry.\n");
			return SUCCESS;
		break;
	}

	return SUCCESS;
}

int scripter_directive_list (scripter_ast_list *args)
{
	/*
	 * This directive lists objects that are identified by the passed keyword.
	 *
	 * Synopsis:
	 *
	 *   list keyword
	 *
	 * The following keywords are supported:
	 *
	 *   parameters  ..  qualifiers of all model parameters
	 *   qualifiers  ..  alias of the above
	 *   tba         ..  parameters that are marked for adjustment
	 *   spots       ..  all spots and their parameters
	 *
	 * This directive is grammar-controlled, so there is no need for error-
	 * handling here.
	 */

	int i;
	char *ident = args->elem->value.variable;

	if (strcmp (ident, "qualifiers") == 0 || strcmp (ident, "parameters") == 0) {
		PHOEBE_parameter_list *list;
		for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
			list = PHOEBE_pt->bucket[i];
			while (list) {
				fprintf (PHOEBE_output, "\t%s\n", list->par->qualifier);
				list = list->next;
			}
		}
	}
	else if (strcmp (ident, "tba") == 0) {
		PHOEBE_parameter_list *list = phoebe_parameter_list_get_marked_tba ();
		while (list) {
			fprintf (PHOEBE_output, "\t%s\n", list->par->qualifier);
			list = list->next;
		}
	}
	else {
		phoebe_scripter_output ("argument '%s' to directive 'list' unknown.\n", ident);
	}

	return SUCCESS;
}

int scripter_directive_print (scripter_ast_list *args)
{
	/*
	 * The 'print' directive prints out an arbitrary number of arguments,
	 * which may be any of PHOEBE's non-void types.
	 *
	 * Synopsis:
	 *
	 *   print exprs
	 *
	 * This directive takes an arbitrary number of arguments, so generic
	 * error handling doesn't apply.
	 */

	int i;
	int argno = scripter_ast_list_length (args);
	scripter_ast_value *vals = phoebe_malloc (argno * sizeof (*vals));

	for (i = 0; i < argno; i++) {
		vals[i] = scripter_ast_evaluate (args->elem);
		if (vals[i].type == type_void) {
			scripter_ast_value_array_free (vals, i);
			phoebe_scripter_output ("argument %d type mismatch: void passed, non-void expected.\n", i+1);
			return ERROR_SCRIPTER_ARGUMENT_TYPE_MISMATCH;
		}
		args = args->next;
	}

	if (PHOEBE_output == stdout)
		fprintf (PHOEBE_output, "\t");

	for (i = 0; i < argno; i++)
		scripter_ast_value_print (vals[i]);

	if (PHOEBE_output == stdout)
		fprintf (PHOEBE_output, "\n");

	scripter_ast_value_array_free (vals, argno);
	return SUCCESS;
}

int scripter_directive_show (scripter_ast_list *args)
{
	/* 
	 * This directive prints the contents of the passed qualifier to screen.
	 *
	 * Synopsis:
	 *
	 *   show qualifier
	 */

	int status;
	char format[255];
	scripter_ast_value *vals;
	PHOEBE_parameter *par;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_qualifier);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return status;
	}

	par = phoebe_parameter_lookup (vals[0].value.str);
	if (!par) {
		phoebe_scripter_output ("parameter %s not recognized, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 1);
		return ERROR_QUALIFIER_NOT_FOUND;
	}

	switch (par->type) {
		case TYPE_INT: {
			int value;
			phoebe_parameter_get_value (par, &value);
			sprintf (format, "\t%s\n", par->format);
			fprintf (PHOEBE_output, format, value);
		}
		break;
		case TYPE_BOOL: {
			bool value;
			phoebe_parameter_get_value (par, &value);
			if (value)
				fprintf (PHOEBE_output, "\tYES\n");
			else
				fprintf (PHOEBE_output, "\tNO\n");
		}
		break;
		case TYPE_DOUBLE: {
			double value;
			phoebe_parameter_get_value (par, &value);
			sprintf (format, "\t%s\n", par->format);
			fprintf (PHOEBE_output, format, value);
		}
		break;
		case TYPE_STRING: {
			const char *value;
			phoebe_parameter_get_value (par, &value);
			sprintf (format, "\t%s\n", par->format);
			fprintf (PHOEBE_output, format, value);
		}
		break;
		case TYPE_INT_ARRAY: {
			PHOEBE_array *array = phoebe_array_new_from_qualifier (par->qualifier);
			if (array) {
				fprintf (PHOEBE_output, "\t");
				phoebe_array_print (array);
				fprintf (PHOEBE_output, "\n");
				phoebe_array_free (array);
			}
			else
				fprintf (PHOEBE_output, "\t<empty array>\n");
		}
		break;
		case TYPE_BOOL_ARRAY: {
			PHOEBE_array *array = phoebe_array_new_from_qualifier (par->qualifier);
			if (array) {
				fprintf (PHOEBE_output, "\t");
				phoebe_array_print (array);
				fprintf (PHOEBE_output, "\n");
				phoebe_array_free (array);
			}
			else
				fprintf (PHOEBE_output, "\t<empty array>\n");
		}
		break;
		case TYPE_DOUBLE_ARRAY: {
			PHOEBE_vector *vec = phoebe_vector_new_from_qualifier (par->qualifier);
			if (vec) {
				fprintf (PHOEBE_output, "\t");
				phoebe_vector_print (vec);
				fprintf (PHOEBE_output, "\n");
				phoebe_vector_free (vec);
			}
			else
				fprintf (PHOEBE_output, "\t<empty array>\n");
		}
		break;
		case TYPE_STRING_ARRAY: {
			PHOEBE_array *array = phoebe_array_new_from_qualifier (par->qualifier);
			if (array) {
				fprintf (PHOEBE_output, "\t");
				phoebe_array_print (array);
				fprintf (PHOEBE_output, "\n");
				phoebe_array_free (array);
			}
			else
				fprintf (PHOEBE_output, "\t<empty array>\n");
		}
		break;
		default:
			phoebe_scripter_output ("exception handler invoked in scripter_directive_show (), please report this!\n");
		break;
	}

	scripter_ast_value_array_free (vals, 1);
	return SUCCESS;
}

int scripter_directive_quit (scripter_ast_list *args)
{
	int c;
	int status = scripter_command_args_evaluate (args, NULL, 0, 0);
	if (status != SUCCESS) return status;

	phoebe_scripter_output ("are you sure you want to quit [y/N]: ");
	c = getchar ();
	if ( ( c  == 'y' ) || (c == 'Y') ) {
		scripter_quit ();
		phoebe_quit ();
	}

	return SUCCESS;
}

int scripter_directive_stdump (scripter_ast_list *args)
{
	/*
	 * This directive is used for debugging: it dumps the symbol table on
	 * screen. It takes no arguments.
	 *
	 * Synopsis:
	 *
	 *   stdump
	 */

	scripter_ast_value *vals;
	int status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) return status;

	scripter_symbol_table *table = symbol_table_lookup (symbol_table, vals[0].value.str);
	if (!table) {
		phoebe_scripter_output ("symbol table '%s' does not exist, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 1);
		return SUCCESS;
	}
	symbol_table_print (table);

	scripter_ast_value_array_free (vals, 1);
	return SUCCESS;
}
