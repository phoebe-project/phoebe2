#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter_io.h"
#include "phoebe_scripter_types.h"

scripter_symbol_table          *symbol_table;
PHOEBE_scripter_command_table  *scripter_commands;
PHOEBE_scripter_function_table *scripter_functions;

scripter_symbol_table *symbol_table_add (scripter_symbol_table *parent, char *name)
{
	int i;

	scripter_symbol_table *table = phoebe_malloc (sizeof (*table));
	table->name = strdup (name);
	for (i = 0; i < NO_OF_HASH_BUCKETS; i++)
		table->symbol[i] = NULL;
	table->next = parent;

	return table;
}

scripter_symbol_table *symbol_table_lookup (scripter_symbol_table *table, char *name)
{
	scripter_symbol_table *t = table;

	while (t) {
		if (strcmp (t->name, name) == 0) return t;
		t = t->next;
	}
	return NULL;
}

scripter_symbol_table *symbol_table_remove (scripter_symbol_table *table, char *name)
{
	scripter_symbol_table *t, *prev;
	
	prev = NULL;
	for (t = table; t != NULL; t = t->next) {
		if (strcmp (t->name, name) == 0) {
			if (!prev)
				table = t->next;
			else
				prev->next = t->next;
			symbol_table_free (t);
			return table;
		}
		prev = t;
	}
	return NULL;
}

int symbol_table_free (scripter_symbol_table *table)
{
	int i;

	free (table->name);
	for (i = 0; i < NO_OF_HASH_BUCKETS; i++) {
		if (table->symbol[i])
			scripter_symbol_free_list (table->symbol[i]);
	}
	free (table);

	return SUCCESS;
}

int symbol_table_free_all (scripter_symbol_table *table)
{
	scripter_symbol_table *next;
	
	for (; table; table = next) {
		next = table->next;
		symbol_table_free (table);
	}
	return SUCCESS;
}

int symbol_table_print (scripter_symbol_table *table)
{
	int i;
	scripter_symbol *s;

	fprintf (PHOEBE_output, "------------------------------------------------------\n");
	fprintf (PHOEBE_output, "| Symbol table name: %25s       |\n", table->name);
	fprintf (PHOEBE_output, "------------------------------------------------------\n");
	fprintf (PHOEBE_output, "| Hash: | Variable:    | Variable:    | Variable:    |\n");
	fprintf (PHOEBE_output, "------------------------------------------------------\n");
	for (i = 0; i < NO_OF_HASH_BUCKETS; i++) {
		if (table->symbol[i] != NULL) {
			s = table->symbol[i];
			fprintf (PHOEBE_output, "| %3d   | ", i);
			while (s) {
				char *name = strdup (s->name);
				if (strlen(s->name) >= 12) {
					name[11] = '*'; name[12] = '\0';
				}
				fprintf (PHOEBE_output, "%-12s | ", name);
				s = s->next;
				free (name);
			}
			fprintf (PHOEBE_output, "\n");
		}
	}
	fprintf (PHOEBE_output, "------------------------------------------------------\n");

	return SUCCESS;
}

/******************************************************************************/

unsigned int scripter_symbol_hash (const char *id)
{
/*
 * This is a new hash function that behaves much better than the old one,
 * which is still kept for reference below. For formalism details please
 * refer to PHOEBE API.
 */

	unsigned int h = 0;
	unsigned char *p;

	for (p = (unsigned char *) id; *p != '\0'; p++)
		h = HASH_MULTIPLIER * h + *p;

	return h % NO_OF_HASH_BUCKETS;
}

scripter_symbol *scripter_symbol_commit (scripter_symbol_table *table, char *id, scripter_ast *value)
{
	int hash = scripter_symbol_hash (id);
	scripter_symbol *s = table->symbol[hash];
	
	while (s) {
		if (strcmp (s->name, id) == 0) break;
		s = s->next;
	}

	if (s) {
		scripter_ast_free (s->link);
		s->link = value;
	}
	else {
		s = phoebe_malloc (sizeof (*s));

		s->name = strdup (id);
		s->link = value;       /* Don't copy here, the AST node is not freed! */

		s->next = table->symbol[hash];
		table->symbol[hash] = s;
	}

	return SUCCESS;
}

scripter_symbol *scripter_symbol_lookup (scripter_symbol_table *table, char *id)
{
	unsigned int hash = scripter_symbol_hash (id);
	scripter_symbol *s = table->symbol[hash];

	while (s) {
		if (strcmp (s->name, id) == 0) break;
		s = s->next;
	}
	if (!s) {
		/* The symbol doesn't appear in this table. Does it appear in global: */
		scripter_symbol_table *global = symbol_table_lookup (symbol_table, "global");
		s = global->symbol[hash];
		
		while (s) {
			if (strcmp (s->name, id) == 0) break;
			s = s->next;
		}
		
		if (!s)
			return NULL;
	}

	return s;
}

int scripter_symbol_remove (scripter_symbol_table *table, char *id)
{
	unsigned int hash = scripter_symbol_hash (id);
	scripter_symbol *s = table->symbol[hash];
	scripter_symbol *p = NULL;

	while (s) {
		if (strcmp (s->name, id) == 0) break;
		p = s;
		s = s->next;
	}

	if (!s) {
		/* This means that the symbol is not in the current table. Is it in a */
		/* global symbol table?                                               */
		table = symbol_table_lookup (symbol_table, "global");
		s = table->symbol[hash];
		
		while (s) {
			if (strcmp (s->name, id) == 0) break;
			p = s;
			s = s->next;
		}
	}

	if (!s) return ERROR_SCRIPTER_INVALID_VARIABLE;

	if (!p) table->symbol[hash] = s->next; else p->next = s->next;
	scripter_symbol_free (s);

	return SUCCESS;
}

int scripter_symbol_free (scripter_symbol *s)
{
	free (s->name);
	scripter_ast_free (s->link);
	free (s);
	return SUCCESS;
}

int scripter_symbol_free_list (scripter_symbol *s)
{
	scripter_symbol *next;
	
	for (; s; s = next) {
		next = s->next;
		scripter_symbol_free (s);
	}
	return SUCCESS;
}

/******************************************************************************/
/*                             SCRIPTER_AST_VALUE                             */
/******************************************************************************/

void scripter_ast_value_free (scripter_ast_value val)
{
	/*
	 * This function frees all allocated fields of the scripter_ast_value
	 * structure, namely strings and arrays. Since the value variable itself
	 * is always locally defined, its validity is preserved only until the
	 * local function exits, so there's no fear of memory leak there.
	 */

	switch (val.type) {
		case type_int:
			/* Fall through */
		break;
		case type_bool:
			/* Fall through */
		break;
		case type_double:
			/* Fall through */
		break;
		case type_string:
			free (val.value.str);
		break;
		case type_vector:
			phoebe_vector_free (val.value.vec);
		break;
		case type_array:
			phoebe_array_free (val.value.array);
		break;
		case type_curve:
			phoebe_curve_free (val.value.curve);
		break;
		case type_spectrum:
			phoebe_spectrum_free (val.value.spectrum);
		break;
		case type_qualifier:
			free (val.value.str);
		break;
		case type_minfeedback:
			phoebe_minimizer_feedback_free (val.value.feedback);
		break;
		case type_any:
			/* Fall through */
		break;
		case type_void:
			/* Fall through */
		break;
		default:
			phoebe_scripter_output ("exception handler invoked in scripter_ast_value_free (), please report this.\n");
			phoebe_scripter_output ("attempted to free type code %d.\n", val.type);
		break;
	}

	return;
}

int scripter_ast_value_array_free (scripter_ast_value *vals, int dim)
{
	/*
	 * This function frees the array of scripter_ast_value's value-by-value.
	 * It does this by calling scripter_ast_value_free () function. This
	 * function should be called at the end of scripter commands where error-
	 * handling is performed by the scripter_command_args_evaluate () function.
	 */

	int i;

	for (i = 0; i < dim; i++)
		scripter_ast_value_free (vals[i]);
	free (vals);

	return SUCCESS;
}

/******************************************************************************/

int scripter_command_args_evaluate (scripter_ast_list *args, scripter_ast_value **vals, int Nmin, int Nmax, ...)
{
	/*
	 * This function takes a list of arguments, counts them and compares the
	 * number to the expected number of parameters, which may also be a range.
	 * If the check succeeds, the function proceeds to type verification, one
	 * argument at the time. Each argument is evaluated and its value is
	 * written to the newly allocated *vals[] array. Any optional arguments
	 * that are missing set the corresponding vals array element to type_void.
	 *
	 * In case of a type mismatch, type propagation is first attempted. The
	 * following type propagations are supported:
	 *
	 *   int -> double
	 *   int -> boolean (just for 0 and 1)
	 *
	 * Synopsis:
	 *
	 *   scripter_evaluate_command_args (args, vals, Nmin, Nmax, argtype1,
	 *                                                        ..., argtypeNmax)
	 *
	 * Where:
	 *
	 *   args     ..  a list of arguments passed to the command
	 *   vals     ..  a pointer to the array of scripter_ast_value-s, which
	 *                is passed uninitialized. It is allocated in this function
	 *                and freed if an error occured. If SUCCESS is returned,
	 *                the user must free this array in the calling function.
	 *                If N=0 (see below), vals is not used (simply pass NULL).
	 *   Nmin     ..  required number of arguments to the command
	 *   Nmax     ..  maximum number of arguments to the command
	 *   argtype  ..  each argument's type_* type that is expected by the
	 *                function
	 *
	 * Return value:
	 *
	 *   SUCCESS
	 *   ERROR_SCRIPTER_ARGUMENT_NUMBER_MISMATCH
	 *   ERROR_SCRIPTER_ARGUMENT_TYPE_MISMATCH
	 */

	int i;
	va_list partypes;
	int type, passed_type;

	int argno = scripter_ast_list_length (args);
	if (argno < Nmin || argno > Nmax) {
		if (Nmin == Nmax)
			phoebe_scripter_output ("argument number mismatch: %d passed, %d expected.\n", argno, Nmin);
		else
			phoebe_scripter_output ("argument number mismatch: %d passed, %d-%d expected.\n", argno, Nmin, Nmax);
		return ERROR_SCRIPTER_ARGUMENT_NUMBER_MISMATCH;
	}

	if (Nmax != 0)
		*vals = phoebe_malloc (Nmax * sizeof (**vals));
	va_start (partypes, Nmax);

	for (i = 0; i < argno; i++) {
		type = va_arg (partypes, int);
		(*vals)[i] = scripter_ast_evaluate (args->elem);
		passed_type = (*vals)[i].type;

		/* Check if the argument is of any (non-void) type; if so, the check  */
		/* always succeeds:                                                   */
		if (type == type_any && passed_type != type_void) {
			args = args->next;
			continue;
		}

		/* Do the propagation if necessary: */
		if (passed_type == type_int && type == type_bool)
			propagate_int_to_bool (&((*vals)[i]));
		if (passed_type == type_int && type == type_double)
			propagate_int_to_double (&((*vals)[i]));

		if ((*vals)[i].type != type) {
			scripter_ast_value_array_free (*vals, i);
			phoebe_scripter_output ("argument %d type mismatch: %s passed, %s expected.\n", i+1, scripter_ast_value_type_get_name (passed_type), scripter_ast_value_type_get_name (type));
			return ERROR_SCRIPTER_ARGUMENT_TYPE_MISMATCH;
		}
		args = args->next;
	}
	for (i = argno; i < Nmax; i++) {
		/* These are the missing optional arguments that we set to void.      */
		(*vals)[i].type = type_void;
	}

	va_end (partypes);

	return SUCCESS;
}

PHOEBE_scripter_command *scripter_command_new ()
{
	PHOEBE_scripter_command *command = phoebe_malloc (sizeof (*command));
	command->name = NULL;
	command->func = NULL;

	return command;
}

int scripter_command_free (PHOEBE_scripter_command *command)
{
	free (command->name);
	free (command);

	return SUCCESS;
}

int scripter_command_register (char *name, scripter_ast_value (*func) ())
{
	PHOEBE_scripter_command *command = scripter_command_new ();
	command->name = strdup (name);
	command->func = func;

	scripter_commands->no++;
	scripter_commands->command = phoebe_realloc (scripter_commands->command, scripter_commands->no * sizeof (*(scripter_commands->command)));
	scripter_commands->command[scripter_commands->no-1] = command;

	return SUCCESS;
}

int scripter_command_get_index (char *name, int *index)
{
	int i;

	for (i = 0; i < scripter_commands->no; i++) {
		if (strcmp (scripter_commands->command[i]->name, name) == 0) {
			*index = i;
			return SUCCESS;
		}
	}
	*index = -1;
	return ERROR_SCRIPTER_COMMAND_DOES_NOT_EXIST;
}

int scripter_commands_free_all (PHOEBE_scripter_command_table *table)
{
	int i;

	for (i = 0; i < table->no; i++) {
		scripter_command_free (table->command[i]);
	}
	free (table->command);
	free (table);

	return SUCCESS;
}

/******************************************************************************/

int scripter_function_register (char *func)
{
	scripter_functions->no++;
	scripter_functions->func = phoebe_realloc (scripter_functions->func, scripter_functions->no * sizeof (*(scripter_functions->func)));

	scripter_functions->func[scripter_functions->no-1] = strdup (func);

	return SUCCESS;
}

int scripter_function_register_all ()
{
	scripter_function_register ("sin");
	scripter_function_register ("cos");
	scripter_function_register ("tan");
	scripter_function_register ("asin");
	scripter_function_register ("acos");
	scripter_function_register ("atan");
	scripter_function_register ("exp");
	scripter_function_register ("ln");
	scripter_function_register ("log");
	scripter_function_register ("sqrt");
	scripter_function_register ("norm");
	scripter_function_register ("rand");
	scripter_function_register ("trunc");
	scripter_function_register ("round");
	scripter_function_register ("int");
	scripter_function_register ("frac");
	scripter_function_register ("abs");
	scripter_function_register ("dim");
	scripter_function_register ("strlen");
	scripter_function_register ("isnan");

	return SUCCESS;
}

int scripter_function_free_all (PHOEBE_scripter_function_table *table)
{
	int i;

	for (i = 0; i < table->no; i++)
		free (table->func[i]);
	free (table->func);
	free (table);

	return SUCCESS;
}

bool scripter_function_defined (char *func)
{
	int i;

	for (i = 0; i < scripter_functions->no; i++)
		if (strcmp (func, scripter_functions->func[i]) == 0)
			return TRUE;

	return FALSE;
}
