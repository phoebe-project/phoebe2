/* This is PHOEBE scripter's Abstract Syntax Tree (AST) implementation.       */

#include <limits.h>
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_build_config.h"

#include "phoebe_scripter_arithmetics.h"
#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_commands.h"
#include "phoebe_scripter_directives.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter.lng.h"
#include "phoebe_scripter_io.h"
#include "phoebe_scripter_types.h"

char *scripter_ast_kind_name (scripter_ast_kind kind)
{
	switch (kind) {
		case kind_add:                         return "add";                         break;
		case kind_and:                         return "and";                         break;
		case kind_array:                       return "array";                       break;
		case kind_block:                       return "block";                       break;
		case kind_builtin:                     return "builtin";                     break;
		case kind_calc:                        return "calc";                        break;
		case kind_clear:                       return "clear";                       break;
		case kind_command:                     return "command";                     break;
		case kind_datastat:                    return "datastat";                    break;
		case kind_dec:                         return "dec";                         break;
		case kind_decby:                       return "decby";                       break;
		case kind_define:                      return "define";                      break;
		case kind_div:                         return "div";                         break;
		case kind_divby:                       return "divby";                       break;
		case kind_element:                     return "element";                     break;
		case kind_equal:                       return "equal";                       break;
		case kind_execute:                     return "execute";                     break;
		case kind_execute_macro:               return "execute_macro";               break;
		case kind_export:                      return "export";                      break;
		case kind_field:                       return "field";                       break;
		case kind_for:                         return "for";                         break;
		case kind_func:                        return "func";                        break;
		case kind_gequal:                      return "gequal";                      break;
		case kind_greater:                     return "greater";                     break;
		case kind_help:                        return "help";                        break;
		case kind_idiv:                        return "idiv";                        break;
		case kind_if:                          return "if";                          break;
		case kind_ignore:                      return "ignore";                      break;
		case kind_inc:                         return "inc";                         break;
		case kind_incby:                       return "incby";                       break;
		case kind_info:                        return "info";                        break;
		case kind_lequal:                      return "lequal";                      break;
		case kind_less:                        return "less";                        break;
		case kind_lexerror:                    return "lexerror";                    break;
		case kind_list:                        return "list";                        break;
		case kind_macro:                       return "macro";                       break;
		case kind_matrix_element:              return "matrix_element";              break;
		case kind_mul:                         return "mul";                         break;
		case kind_multby:                      return "multby";                      break;
		case kind_nequal:                      return "nequal";                      break;
		case kind_not:                         return "not";                         break;
		case kind_or:                          return "or";                          break;
		case kind_os_cd:                       return "os_cd";                       break;
		case kind_os_ls:                       return "os_ls";                       break;
		case kind_os_pwd:                      return "os_pwd";                      break;
		case kind_raise:                       return "raise";                       break;
		case kind_print:                       return "print";                       break;
		case kind_qual_value:                  return "qual_value";                  break;
		case kind_quit:                        return "quit";                        break;
		case kind_return:                      return "return";                      break;
		case kind_set:                         return "set";                         break;
		case kind_set_element:                 return "set_element";                 break;
		case kind_set_matrix_element:          return "set_matrix_element";          break;
		case kind_show:                        return "show";                        break;
		case kind_statement:                   return "statement";                   break;
		case kind_stdump:                      return "stdump";                      break;
		case kind_sub:                         return "sub";                         break;
		case kind_system_call:                 return "system_call";                 break;
		case kind_unarym:                      return "unarym";                      break;
		case kind_unaryp:                      return "unaryp";                      break;
		case kind_unset:                       return "unset";                       break;
		case kind_while:                       return "while";                       break;
		default:                               return "invalid kind";                break;
	}
}

scripter_ast *scripter_ast_add_int (const int val)
{
	scripter_ast *out  = phoebe_malloc (sizeof (*out));
	out->type          = ast_int;
	out->value.integer = val;
	return out;
}

scripter_ast *scripter_ast_add_double (const double val)
{
	scripter_ast *out = phoebe_malloc (sizeof (*out));
	out->type         = ast_double;
	out->value.real   = val;
	return out;
}

scripter_ast *scripter_ast_add_bool (const bool val)
{
	scripter_ast *out  = phoebe_malloc (sizeof (*out));
	out->type          = ast_bool;
	out->value.boolean = val;
	return out;
}

scripter_ast *scripter_ast_add_string (const char *val)
{
	scripter_ast *out = phoebe_malloc (sizeof (*out));
	out->type         = ast_string;
	out->value.string = strdup (val);
	return out;
}

scripter_ast *scripter_ast_add_array (PHOEBE_array *array)
{
	scripter_ast *out = phoebe_malloc (sizeof (*out));
	out->type         = ast_array;
	out->value.array  = phoebe_array_duplicate (array);
	return out;
}

scripter_ast *scripter_ast_add_vector (PHOEBE_vector *vec)
{
	scripter_ast *out = phoebe_malloc (sizeof (*out));
	out->type         = ast_vector;
	out->value.vec    = phoebe_vector_duplicate (vec);
	return out;
}

scripter_ast *scripter_ast_add_curve (PHOEBE_curve *curve)
{
	scripter_ast *out = phoebe_malloc (sizeof (*out));
	out->type         = ast_curve;
	out->value.curve  = phoebe_curve_duplicate (curve);
	return out;
}

scripter_ast *scripter_ast_add_spectrum (PHOEBE_spectrum *spectrum)
{
	scripter_ast *out = phoebe_malloc (sizeof (*out));
	out->type = ast_spectrum;
	out->value.spectrum = phoebe_spectrum_duplicate (spectrum);
	return out;
}

scripter_ast *scripter_ast_add_variable (const char *val)
{
	scripter_ast *out   = phoebe_malloc (sizeof (*out));
	out->type           = ast_variable;
	out->value.variable = strdup (val);
	return out;
}

scripter_ast *scripter_ast_add_qualifier (const char *val)
{
	scripter_ast *out    = phoebe_malloc (sizeof (*out));
	out->type            = ast_qualifier;
	out->value.qualifier = strdup (val);
	return out;
}

scripter_ast *scripter_ast_add_function (const char *val)
{
	scripter_ast *out    = phoebe_malloc (sizeof (*out));
	out->type            = ast_function;
	out->value.variable  = strdup (val);
	return out;
}

scripter_ast *scripter_ast_add_minfeedback (PHOEBE_minimizer_feedback *feedback)
{
	scripter_ast *out      = phoebe_malloc (sizeof (*out));
	out->type              = ast_minfeedback;
	out->value.minfeedback = phoebe_minimizer_feedback_duplicate (feedback);
	return out;
}

scripter_ast *scripter_ast_add_node (const scripter_ast_kind kind, scripter_ast_list *args)
{
	scripter_ast *out    = phoebe_malloc (sizeof (*out));
	out->type            = ast_node;
	out->value.node.kind = kind;
	out->value.node.args = args;
	return out;
}

scripter_ast_list *scripter_ast_construct_list (scripter_ast *ast, scripter_ast_list *list)
{
	scripter_ast_list *out = phoebe_malloc (sizeof (*out));
	out->elem = ast;
	out->next = list;
	return out;
}

scripter_ast_list *scripter_ast_reverse_list (scripter_ast_list *c, scripter_ast_list *p)
{
	/*
	 * When creating an AST list, the order of arguments is last element to
	 * the first, e.g. f(x,y,z) would create z->y->x parentage. This isn't what
	 * we want and the following function reverses this order through recursion
	 * by calling itself. The result after that is x->y->z, as it should be.
	 */

	scripter_ast_list *rev;

	if (!c) return p;
	rev = scripter_ast_reverse_list (c->next, c);
	c->next = p;

	return rev;
}

int scripter_ast_list_length (scripter_ast_list *in)
{
	int i = 0;
	for (; in != 0; in = in->next) i++;
	return i;
}

int scripter_ast_print (int depth, scripter_ast *in)
{
	int i;
	scripter_ast_list *args;

	for (i = 0; i < depth; i++)
		fprintf (PHOEBE_output, "| ");

	switch (in->type) {
		case ast_int:
			fprintf (PHOEBE_output, "| %d\n", in->value.integer);
		break;
		case ast_bool:
			if (in->value.boolean) fprintf (PHOEBE_output, "  TRUE\n");
			else fprintf (PHOEBE_output, "  FALSE\n");
		break;
		case ast_double:
			fprintf (PHOEBE_output, "| %lf\n", in->value.real);
		break;
		case ast_string:
			fprintf (PHOEBE_output, "| \"%s\"\n", in->value.string);
		break;
		case ast_vector:
			fprintf (PHOEBE_output, "| "); phoebe_vector_print (in->value.vec);
		break;
		case ast_array:
			fprintf (PHOEBE_output, "| "); phoebe_array_print (in->value.array);
		break;
		case ast_curve:
			fprintf (PHOEBE_output, "| "); phoebe_curve_print (in->value.curve);
		case ast_spectrum:
			fprintf (PHOEBE_output, "| "); phoebe_spectrum_print (in->value.spectrum);
		break;
		case ast_variable:
			fprintf (PHOEBE_output, "| %s\n", in->value.variable);
		break;
		case ast_qualifier:
			fprintf (PHOEBE_output, "| %s\n", in->value.qualifier);
		break;
		case ast_function:
			fprintf (PHOEBE_output, "| %s\n", in->value.function);
		break;
		case ast_minfeedback:
			fprintf (PHOEBE_output, "| "); phoebe_minimizer_feedback_print (in->value.minfeedback);
		break;
		case ast_node:
			depth += 1;
			fprintf (PHOEBE_output, "| %s\n", scripter_ast_kind_name (in->value.node.kind));
			args = in->value.node.args;
			while (args) {
				scripter_ast_print (depth, args->elem);
				args = args->next;
			}
		break;
		default:
			fprintf (PHOEBE_output, "exception handler invoked in scripter_ast_print (), please report this.\n");
		break;
	}

	return SUCCESS;
}

scripter_ast *scripter_ast_duplicate (scripter_ast *in)
{
	scripter_ast *out;
	scripter_ast_list *args;
	scripter_ast_list *argcopies = NULL;

	switch (in->type) {
		case ast_int:
			out = scripter_ast_add_int (in->value.integer);
		break;
		case ast_bool:
			out = scripter_ast_add_bool (in->value.boolean);
		break;
		case ast_double:
			out = scripter_ast_add_double (in->value.real);
		break;
		case ast_string:
			out = scripter_ast_add_string (in->value.string);
		break;
		case ast_vector:
			out = scripter_ast_add_vector (in->value.vec);
		break;
		case ast_array:
			out = scripter_ast_add_array (in->value.array);
		break;
		case ast_curve:
			out = scripter_ast_add_curve (in->value.curve);
		break;
		case ast_spectrum:
			out = scripter_ast_add_spectrum (in->value.spectrum);
		break;
		case ast_variable:
			out = scripter_ast_add_variable (in->value.variable);
		break;
		case ast_qualifier:
			out = scripter_ast_add_qualifier (in->value.qualifier);
		break;
		case ast_function:
			out = scripter_ast_add_function (in->value.function);
		break;
		case ast_minfeedback:
			out = scripter_ast_add_minfeedback (in->value.minfeedback);
		break;
		case ast_node:
			args = in->value.node.args;
			while (args) {
				argcopies = scripter_ast_construct_list (scripter_ast_duplicate (args->elem), argcopies);
				args = args->next;
			}
			out = scripter_ast_add_node (in->value.node.kind, scripter_ast_reverse_list (argcopies, NULL));
		break;
		default:
			out = NULL;
			fprintf (PHOEBE_output, "exception handler invoked in scripter_ast_duplicate (), please report this.\n");
		break;
	}

	return out;
}

int scripter_ast_free (scripter_ast *in)
{
	switch (in->type) {
		case ast_int:
			/* Nothing to be done */
		break;
		case ast_bool:
			/* Nothing to be done */
		break;
		case ast_double:
			/* Nothing to be done */
		break;
		case ast_string:
			free (in->value.string);
		break;
		case ast_vector:
			phoebe_vector_free (in->value.vec);
		break;
		case ast_array:
			phoebe_array_free (in->value.array);
		break;
		case ast_curve:
			phoebe_curve_free (in->value.curve);
		break;
		case ast_spectrum:
			phoebe_spectrum_free (in->value.spectrum);
		break;
		case ast_variable:
			free (in->value.variable);
		break;
		case ast_qualifier:
			free (in->value.qualifier);
		break;
		case ast_function:
			free (in->value.function);
		break;
		case ast_minfeedback:
			phoebe_minimizer_feedback_free (in->value.minfeedback);
		break;
		case ast_node:
			scripter_ast_list_free (in->value.node.args);
		break;
		default:
			fprintf (PHOEBE_output, "exception handler invoked in scripter_ast_free (), please report this.\n");
		break;
	}
	free (in);

	return SUCCESS;
}

int scripter_ast_list_free (scripter_ast_list *list)
{
	scripter_ast_list *next;

	if (!list)
		return SUCCESS;

	next = list->next;

	while (next) {
		scripter_ast_free (list->elem);
		free (list);
		list = next;
		next = list->next;
	}

	scripter_ast_free (list->elem);
	free (list);

	return SUCCESS;
}

scripter_ast_value scripter_ast_evaluate (scripter_ast *in)
{
	/* 
	 * This is the most important function of the AST implementation. Based on
	 * the AST leaf type, it decides what to do and how to do it. It returns
	 * any type (the union) and the function must decide which union field to
	 * use based on the AST type.
	 */

	int status;
	scripter_ast_value out;

	if (!in) {
		out.type = type_void;
		return out;
	}

	switch (in->type) {
		case ast_int:
			out.type    = type_int;
			out.value.i = in->value.integer;
			return out;
		case ast_bool:
			out.type    = type_bool;
			out.value.b = in->value.boolean;
			return out;
		case ast_double:
			out.type    = type_double;
			out.value.d = in->value.real;
			return out;
		case ast_string:
			out.type      = type_string;
			out.value.str = strdup (in->value.string);
			return out;
		case ast_vector:
			out.type      = type_vector;
			out.value.vec = phoebe_vector_duplicate (in->value.vec);
			return out;
		case ast_array:
			out.type        = type_array;
			out.value.array = phoebe_array_duplicate (in->value.array);
			return out;
		case ast_curve:
			out.type        = type_curve;
			out.value.curve = phoebe_curve_duplicate (in->value.curve);
			return out;
		case ast_spectrum:
			out.type           = type_spectrum;
			out.value.spectrum = phoebe_spectrum_duplicate (in->value.spectrum);
			return out;
		case ast_qualifier:
			out.type = type_qualifier;
			out.value.str = strdup (in->value.string);
			return out;
		case ast_function:
			out.type = type_function;
			out.value.str = strdup (in->value.variable);
		case ast_minfeedback:
			out.type = type_minfeedback;
			out.value.feedback = phoebe_minimizer_feedback_duplicate (in->value.minfeedback);
			return out;
		case ast_variable: {
			/*
			 * Do not under any circumstance return pointers to the data (i.e.
			 * to strings, vectors, arrays, ...), we need a true copy returned.
			 * Each directive/command must free the data after it's done using
			 * it so that it doesn't leave memory leaks behind. If we passed a
			 * pointer here, it would get freed and the next evaluation of the
			 * variable would cause a segfault.
			 */
			
			scripter_symbol *s = scripter_symbol_lookup (symbol_table, in->value.variable);
			if (!s) {
				phoebe_scripter_output ("variable '%s' is not initialized.\n", in->value.variable);
				out.type = type_void; return out;
			}
			if (!s->link) {
				phoebe_scripter_output ("variable '%s' is empty.\n", in->value.variable);
				out.type = type_void; return out;
			}

			switch (s->link->type) {
				case ast_int:
					out.type = type_int;
					out.value.i = s->link->value.integer;
				break;
				case ast_bool:
					out.type = type_bool;
					out.value.b = s->link->value.boolean;
				break;
				case ast_double:
					out.type = type_double;
					out.value.d = s->link->value.real;
				break;
				case ast_string:
					out.type = type_string;
					out.value.str = strdup (s->link->value.string);
				break;
				case ast_vector:
					out.type = type_vector;
					out.value.vec = phoebe_vector_duplicate (s->link->value.vec);
				break;
				case ast_array:
					out.type = type_array;
					out.value.array = phoebe_array_duplicate (s->link->value.array);
				break;
				case ast_curve:
					out.type = type_curve;
					out.value.curve = phoebe_curve_duplicate (s->link->value.curve);
				break;
				case ast_qualifier:
					out.type = type_qualifier;
					out.value.str = strdup (s->link->value.qualifier);
				break;
				case ast_function:
					out.type = type_function;
					out.value.str = strdup (s->link->value.variable);
				break;
				case ast_spectrum:
					out.type = type_spectrum;
					out.value.spectrum = phoebe_spectrum_duplicate (s->link->value.spectrum);
				break;
				case ast_minfeedback:
					out.type = type_minfeedback;
					out.value.feedback = phoebe_minimizer_feedback_duplicate (s->link->value.minfeedback);
				break;
				default:
					phoebe_scripter_output ("variable '%s' doesn't hold a numeric or string value (s->link->type = %d).\n", in->value.variable, s->link->type);
				break;
			}
			return out;
		}
		case ast_node:
			if (in->value.node.kind == kind_lexerror) {
				/*
				 * This means that an error occured during the lexer phase.
				 * Typically this is caused by unterminated strings.
				 */

				scripter_ast_value index = scripter_ast_evaluate (in->value.node.args->elem);
				switch (index.value.i) {
					case LEXER_UNTERMINATED_LITERAL:
						phoebe_scripter_output ("unterminated string constant, aborting.\n");
					break;
					default:
						phoebe_scripter_output ("exception handler invoked in kind_lexerror, please report this!\n");
					break;
				}
				out.type = type_void;
				return out;
			}
			if (in->value.node.kind == kind_testme) {
				/*
				 * This is an internal development directive. It may be used
				 * to test anything that comes to mind without having to write
				 * dedicated directive or command.
				 */

				double L1, L2;
				phoebe_calculate_critical_potentials (0.2, 1.0, 0.0, &L1, &L2);
				return out;
			}
			if (in->value.node.kind == kind_statement) {
				/*
				 * This is the most elementary building block of PHOEBE
				 * scripts. It comes in two flavors: direct and redirected:
				 *
				 *   direct:      statement
				 *   redirected:  statement  -> "output"
				 *                statement ->> "output"
				 *
				 * The difference between the two redirection operators -> and
				 * ->> are write and append, respectively.
				 *
				 * The difference between the direct and the redirected method
				 * is in the AST argument. If it is NULL, then it is the direct
				 * method. If it is non-NULL, is is redirected, with the file-
				 * name passed as argument.
				 */

				/* Check whether the statement node has an argument: */
				if (!in->value.node.args->next) {
					/* It doesn't. Execute it: */
					scripter_ast_evaluate (in->value.node.args->elem);
				}
				else {
					/* It does. Get the filename: */
					scripter_ast_value fname = scripter_ast_evaluate (in->value.node.args->next->elem);

					/* Does the passed argument evaluate to a literal: */
					if (fname.type != type_string) {
						phoebe_warning ("attempted redirection to a non-string identifier, reverting to terminal.\n");
						scripter_ast_value_free (fname);
						scripter_ast_evaluate (in->value.node.args->elem);
						out.type = type_void;
						return out;
					}

					/*
					 * It does. Let's open it; here we make a distinction
					 * between the -> and ->> mode; as the second parameter,
					 * the parser passes either "w" or "a", depending on the
					 * redirection operator:
					 */
					PHOEBE_output = fopen (fname.value.str, in->value.node.args->next->next->elem->value.string);

					if (!PHOEBE_output) {
				    	PHOEBE_output = stdout;
						phoebe_scripter_output ("redirection to \"%s\" failed, reverting to stdout.\n", in->value.node.args->next->elem->value.string);
						scripter_ast_evaluate (in->value.node.args->elem);
					}
					else {
						scripter_ast_evaluate (in->value.node.args->elem);
				    	fclose (PHOEBE_output);
				    	PHOEBE_output = stdout;
					}
				scripter_ast_value_free (fname);
				}

				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_block) {
				/*
				 * This is the second most elementary building block of PHOEBE
				 * scripts. It is a block that consists of multiple statements.
				 */

				scripter_ast_list *s = in->value.node.args;

				if (!s) {
					phoebe_debug ("redundant block.\n");
					out.type = type_void;
					return out;
				}

				while (s) {
					scripter_ast_evaluate (s->elem);
					s = s->next;
				}

				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_return) {
				/*
				 * The 'return' directive is used to return the value from
				 * functions. The directive itself simply forwards the passed
				 * value. It is up to the function to test the AST node type
				 * against kind_return and act accordingly if it matches.
				 */

				scripter_ast_value val = scripter_ast_evaluate (in->value.node.args->elem);
				return val;
			}

			if (in->value.node.kind == kind_system_call) {
				/*
				 * This means the user used the '!' delimeter to execute a
				 * system command. The return value is always void.
				 */

				char *syscommand = in->value.node.args->elem->value.string;
				system (syscommand);
				out.type = type_void; return out;
			}

			if (in->value.node.kind == kind_os_ls) {
				/*
				 * This is the 'ls' command.
				 */

				out.type = type_void;

				if (strlen (in->value.node.args->elem->value.string) == 0)
					system ("ls --color=auto");
				else {
					char com[255];
					sprintf (com, "ls --color=auto %s", in->value.node.args->elem->value.string);
					system (com);
					return out;
				}
			}

			if (in->value.node.kind == kind_os_cd) {
				/*
				 * This is the 'cd' command for changing directories. The
				 * implementation is currently linux-specific and it should
				 * be rewritten once other architectures become actual.
				 */

				out.type = type_void;

				if (strlen (in->value.node.args->elem->value.string) == 0) {
					int status = chdir (USER_HOME_DIR);
					if (status != 0)
						phoebe_scripter_output ("Changing directory to $HOME failed, aborting.\n");
					return out;
				}
				else {
					int status = chdir (in->value.node.args->elem->value.string);
					if (status != 0)
						phoebe_scripter_output ("changing directory to %s failed, aborting.\n", in->value.node.args->elem->value.string);
					return out;
				}
			}

			if (in->value.node.kind == kind_os_pwd) {
				/*
				 * This is the 'pwd' directive. It displays the path of the
				 * current directory.
				 */

				char cwd[255];
				getcwd (cwd, 255);
				fprintf (PHOEBE_output, "%s\n", cwd);
				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_array) {
				/*
				 * The kind_array AST is the explicit form for arrays, e.g.
				 * {1,2,3} or {a,b,c}. It is important that it doesn't get
				 * evaluated during the grammar parsing, because it will
				 * generate syntax errors in macros and functions. For
				 * example, consider the following snippet:
				 *
				 *   macro test () {
				 *     set a=1
				 *     set i={a,a+1,a+2}
				 *   }
				 *
				 * Since the macro definition doesn't get evaluated until
				 * the whole macro is read, the set statement would produce
				 * a syntax error on the unknown variable a. That is why we
				 * need this kind here rather than use phoebe_vector_new_from
				 * _list () directly on parsed data.
				 */

				out.value.vec = phoebe_vector_new_from_list (in->value.node.args);
				if (out.value.vec) out.type = type_vector;
				else {
					out.value.array = phoebe_array_new_from_list (in->value.node.args);
					if (out.value.array) out.type = type_array;
					else out.type = type_void;
				}
				return out;
			}

			if (in->value.node.kind == kind_while) {
				/*
				 * This is the simplest loop in PHOEBE scripter. It is
				 * analogous to while loops in other languages. It always
				 * returns void.
				 */

				scripter_ast_value condition = scripter_ast_evaluate (in->value.node.args->elem);
				out.type = type_void;

				if (condition.type != type_bool) {
					phoebe_scripter_output ("the 'while' loop condition is not boolean.\n");
					return out;
				}

				while (condition.value.b) {
					scripter_ast_evaluate (in->value.node.args->next->elem);
					condition = scripter_ast_evaluate (in->value.node.args->elem);
				}
				return out;
			}

			if (in->value.node.kind == kind_for) {
				/*
				 * This is the 'for' loop. It always returns void.
				 */

				/*
				 * The following line will always evaluate successfully
				 * because of grammar rules:
				 */
				char *ident = in->value.node.args->elem->value.variable;
				scripter_ast_value value;
				scripter_ast_value cond;

				scripter_ast *action = in->value.node.args->next->next->next->elem;
				scripter_ast *block  = in->value.node.args->next->next->next->next->elem;

				out.type = type_void;

				/* Check whether a value is a numeric type: */
				value = scripter_ast_evaluate (in->value.node.args->next->elem);
				if (value.type != type_int && value.type != type_double) {
					phoebe_scripter_output ("the initial value is not a numeric value, aborting.\n");
					return out;
				}

				/* Create a symbol for iteration, initialize it and free the  */
				/* the original ast value:                                    */
				if (value.type == type_int)
					scripter_symbol_commit (symbol_table, ident, scripter_ast_add_int (value.value.i));
				if (value.type == type_double)
					scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double (value.value.d));

				/* Check whether a condition is boolean; be sure to create    */
				/* the iterator first, otherwise the condition will fail in   */
				/* most cases.                                                */
				cond = scripter_ast_evaluate (in->value.node.args->next->next->elem);
				if (cond.type != type_bool) {
					phoebe_scripter_output ("the condition is not boolean, so it cannot be evaluated.\n");
					scripter_symbol_remove (symbol_table, ident);
					return out;
				}

				/* Enter the loop:                                            */
				while (cond.value.b) {
					scripter_ast_evaluate (block);
					scripter_ast_evaluate (action);
					cond = scripter_ast_evaluate (in->value.node.args->next->next->elem);
				}

				/* Wrap it up and exit:                                       */
				return out;
			}

			if (in->value.node.kind == kind_unaryp) {
				/* This means that we have a unary '+'.                               */
				out = scripter_ast_evaluate (in->value.node.args->elem);

				if (out.type != type_int    &&
					out.type != type_double &&
					out.type != type_vector) {
					phoebe_scripter_output ("unary '+' operator operates only on numeric types.\n");
					out.type = type_void;
				}
				return out;
			}

			if (in->value.node.kind == kind_unarym) {
				/* This means that we have a unary '-'.                               */
				scripter_ast_value val = scripter_ast_evaluate (in->value.node.args->elem);

				switch (val.type) {
					case type_int:
						out.type = type_int;
						out.value.i = -val.value.i;
					break;
					case type_double:
						out.type = type_double;
						out.value.d = -val.value.d;
					break;
					case type_vector:
						out.type = type_vector;
						out.value.vec = phoebe_vector_duplicate (val.value.vec);
						phoebe_vector_multiply_by (out.value.vec, -1.0);
					break;
					default:
						phoebe_scripter_output ("unary '-' operator operates only on numeric types.\n");
						out.type = type_void;
					break;
				}
				return out;
			}

			if (in->value.node.kind == kind_not) {
				/* This means that we have a logical NOT.                             */

				scripter_ast_value val = scripter_ast_evaluate (in->value.node.args->elem);

				if (val.type != type_bool) {
					phoebe_scripter_output ("the unary '!' operates only on boolean expressions, aborting.\n");
					out.type = type_void;
					return out;
				}

				out.type = type_bool;
				if (val.value.b) out.value.b = FALSE;
					else out.value.b = TRUE;

				return out;
			}

			if (in->value.node.kind == kind_inc) {
				/* This is the increment operator ++. */

				char *ident = in->value.node.args->elem->value.variable;
				scripter_symbol *symb = scripter_symbol_lookup (symbol_table, ident);
				scripter_ast_value value;

				out.type = type_void;
				if (!symb) {
					phoebe_scripter_output ("identifier '%s' not initialized, aborting.\n", ident);
					return out;
				}

				value = scripter_ast_evaluate (symb->link);
				switch (value.type) {
					case type_int:
						symb->link->value.integer++;
					break;
					case type_double:
						symb->link->value.real++;
					break;
					default:
						phoebe_scripter_output ("identifier '%s' is non-numeric, aborting.\n", ident);
					break;
				}
				return out;
			}

			if (in->value.node.kind == kind_dec) {
				/* This is the decrement operator --.                         */

				char *ident = in->value.node.args->elem->value.variable;
				scripter_symbol *symb = scripter_symbol_lookup (symbol_table, ident);
				scripter_ast_value value;

				out.type = type_void;
				if (!symb) {
					phoebe_scripter_output ("identifier '%s' not initialized, aborting.\n", ident);
					return out;
				}

				value = scripter_ast_evaluate (symb->link);
				switch (value.type) {
					case type_int:
						symb->link->value.integer--;
					break;
					case type_double:
						symb->link->value.real--;
					break;
					default:
						phoebe_scripter_output ("identifier '%s' is non-numeric, aborting.\n", ident);
					break;
				}
				return out;
			}
			if (in->value.node.kind == kind_incby) {
				/* This is the increment-by operator +=.                      */

				char   *ident         = in->value.node.args->elem->value.variable;
				scripter_symbol *symb = scripter_symbol_lookup (symbol_table, ident);
				scripter_ast_value by = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value value;

				out.type = type_void;
				if (!symb) {
					phoebe_scripter_output ("identifier '%s' not initialized, aborting.\n", ident);
					return out;
				}
				if (by.type != type_int && by.type != type_double) {
					phoebe_scripter_output ("increment doesn't hold a numeric value.\n", ident);
					return out;
				}
				value = scripter_ast_evaluate (symb->link);
				if (value.type != type_int && value.type != type_double) {
					phoebe_scripter_output ("identifier '%s' doesn't hold a numeric value.\n", ident);
					return out;
				}

				if (value.type == type_int && by.type == type_int)
					symb->link->value.integer += by.value.i;

				if (value.type == type_double && by.type == type_int)
					symb->link->value.real += by.value.i;

				if (value.type == type_double && by.type == type_double)
					symb->link->value.real += by.value.d;

				if (value.type == type_int && by.type == type_double)
					scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double (value.value.i + by.value.d));

				return out;
			}
			if (in->value.node.kind == kind_decby) {
				/* This is the decrement-by operator -= on the variable.      */

				char   *ident         = in->value.node.args->elem->value.variable;
				scripter_symbol *symb = scripter_symbol_lookup (symbol_table, ident);
				scripter_ast_value by = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value value;

				out.type = type_void;
				if (!symb) {
					phoebe_scripter_output ("identifier '%s' not initialized, aborting.\n", ident);
					return out;
				}
				if (by.type != type_int && by.type != type_double) {
					phoebe_scripter_output ("increment doesn't hold a numeric value.\n", ident);
					return out;
				}
				value = scripter_ast_evaluate (symb->link);
				if (value.type != type_int && value.type != type_double) {
					phoebe_scripter_output ("identifier '%s' doesn't hold a numeric value.\n", ident);
					return out;
				}

				if (value.type == type_int && by.type == type_int)
					symb->link->value.integer -= by.value.i;

				if (value.type == type_double && by.type == type_int)
					symb->link->value.real -= by.value.i;

				if (value.type == type_double && by.type == type_double)
					symb->link->value.real -= by.value.d;

				if (value.type == type_int && by.type == type_double)
					scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double (value.value.i - by.value.d));

				return out;
			}
			if (in->value.node.kind == kind_multby) {
				/* This is the multiply-by operator *= on the variable.       */

				char   *ident         = in->value.node.args->elem->value.variable;
				scripter_symbol *symb = scripter_symbol_lookup (symbol_table, ident);
				scripter_ast_value by = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value value;

				out.type = type_void;
				if (!symb) {
					phoebe_scripter_output ("identifier '%s' not initialized, aborting.\n", ident);
					return out;
				}
				if (by.type != type_int && by.type != type_double) {
					phoebe_scripter_output ("increment doesn't hold a numeric value.\n", ident);
					return out;
				}
				value = scripter_ast_evaluate (symb->link);
				if (value.type != type_int && value.type != type_double) {
					phoebe_scripter_output ("identifier '%s' doesn't hold a numeric value.\n", ident);
					return out;
				}

				if (value.type == type_int && by.type == type_int)
					symb->link->value.integer *= by.value.i;

				if (value.type == type_double && by.type == type_int)
					symb->link->value.real *= by.value.i;

				if (value.type == type_double && by.type == type_double)
					symb->link->value.real *= by.value.d;

				if (value.type == type_int && by.type == type_double)
					scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double (by.value.d * value.value.i));

				return out;
			}
			if (in->value.node.kind == kind_divby) {
				/* This is the divide-by operator /= on the variable.         */

				char   *ident         = in->value.node.args->elem->value.variable;
				scripter_symbol *symb = scripter_symbol_lookup (symbol_table, ident);
				scripter_ast_value by = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value value;

				out.type = type_void;
				if (!symb) {
					phoebe_scripter_output ("identifier '%s' not initialized, aborting.\n", ident);
					return out;
				}
				if (by.type != type_int && by.type != type_double) {
					phoebe_scripter_output ("increment doesn't hold a numeric value.\n", ident);
					return out;
				}
				value = scripter_ast_evaluate (symb->link);
				if (value.type != type_int && value.type != type_double) {
					phoebe_scripter_output ("identifier '%s' doesn't hold a numeric value.\n", ident);
					return out;
				}

				if (value.type == type_int) {
					if (by.type == type_int)
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double ((double) value.value.i / (double) by.value.i));
					if (by.type == type_double)
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double ((double) value.value.i / by.value.d));
				}
				if (value.type == type_double) {
					if (by.type == type_int) propagate_int_to_double (&by);
					scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double (value.value.d / by.value.d));
				}

				return out;
			}

			if (in->value.node.kind == kind_field) {
				/*
				 * This operator is used to access fields in a structure. So
				 * far only built-in structures are supported. They are:
				 *
				 *   PHOEBE_curve
				 *   PHOEBE_minimizer_feedback
				 *
				 * Synopsis:
				 *
				 *   set fieldvar = var.field
				 */

				if (in->value.node.args->elem->type == ast_variable) {
					char *ident = in->value.node.args->elem->value.variable;
					char *field = in->value.node.args->next->elem->value.variable;
					scripter_symbol *var = scripter_symbol_lookup (symbol_table, ident);

					if (!var) {
						phoebe_scripter_output ("variable '%s' is not initialized, aborting.\n", ident);
						out.type = type_void; return out;
					}

					switch (var->link->type) {
						case ast_curve: {
							PHOEBE_curve *curve = var->link->value.curve;
						
							if (strcmp (field, "type") == 0) {
								char *type;
								phoebe_curve_type_get_name (curve->type, &type);
								out.type = type_string;
								out.value.str = type;
								return out;
							}
							if (strcmp (field, "indep") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (curve->indep);
								return out;
							}
							if (strcmp (field, "dep") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (curve->dep);
								return out;
							}
							phoebe_scripter_output ("field '%s' is not contained in this structure, aborting.\n", field);
						}
						break;
						case ast_minfeedback: {
							PHOEBE_minimizer_feedback *feedback = var->link->value.minfeedback;

							if (strcmp (field, "algorithm") == 0) {
								char *algname;
								phoebe_minimizer_type_get_name (feedback->algorithm, &algname);
								out.type = type_string;
								out.value.str = algname;
								return out;
							}
							if (strcmp (field, "converged") == 0) {
								out.type = type_bool;
								out.value.i = feedback->converged;
								return out;
							}
							if (strcmp (field, "iters") == 0) {
								out.type = type_int;
								out.value.i = feedback->iters;
								return out;
							}
							if (strcmp (field, "cputime") == 0) {
								out.type = type_double;
								out.value.d = feedback->cputime;
								return out;
							}
							if (strcmp (field, "pars") == 0) {
								out.type = type_array;
								out.value.array = phoebe_array_duplicate (feedback->qualifiers);
								return out;
							}
							if (strcmp (field, "initvals") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (feedback->initvals);
								return out;
							}
							if (strcmp (field, "newvals") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (feedback->newvals);
								return out;
							}
							if (strcmp (field, "ferrors") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (feedback->ferrors);
								return out;
							}
							if (strcmp (field, "chi2s") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (feedback->chi2s);
								return out;
							}
							if (strcmp (field, "wchi2s") == 0) {
								out.type = type_vector;
								out.value.vec = phoebe_vector_duplicate (feedback->wchi2s);
								return out;
							}
							if (strcmp (field, "cfval") == 0) {
								out.type = type_double;
								out.value.d = feedback->cfval;
								return out;
							}

							phoebe_scripter_output ("field '%s' is not contained in this structure, aborting.\n", field);
						}
						break;
						default:
							phoebe_scripter_output ("variable '%s' is not a structure, aborting.\n", ident);
							out.type = type_void; return out;
					}

					out.type = type_void;
					return out;
				}
			}

			if (in->value.node.kind == kind_element) {
				/*
				 * This function accesses the elements of the passed expression.
				 * It is generic in a sense that it applies to any type of
				 * expression, not just the variables.
				 */

				scripter_ast_value expr, index;

				index = scripter_ast_evaluate (in->value.node.args->next->elem);

				if (index.type != type_int) {
					phoebe_scripter_output ("a non-integer index encountered, aborting.\n");
					out.type = type_void;
					scripter_ast_value_free (index);
					return out;
				}

				expr = scripter_ast_evaluate (in->value.node.args->elem);

				switch (expr.type) {
					case type_vector:
						if (index.value.i < 1 || index.value.i > expr.value.vec->dim) {
							phoebe_scripter_output ("index %d is out of range [%d, %d], aborting.\n", index.value.i, 1, expr.value.vec->dim);
							scripter_ast_value_free (index);
							scripter_ast_value_free (expr);
							out.type = type_void; return out;
						}

						out.type = type_double;
						out.value.d = expr.value.vec->val[index.value.i-1];
						scripter_ast_value_free (expr);
						scripter_ast_value_free (index);
						return out;
					case type_array:
						if (index.value.i < 1 || index.value.i > expr.value.array->dim) {
							phoebe_scripter_output ("index %d is out of range [%d, %d], aborting.\n", index.value.i, 1, expr.value.array->dim);
							scripter_ast_value_free (index);
							scripter_ast_value_free (expr);
							out.type = type_void; return out;
						}
						switch (expr.value.array->type) {
							case TYPE_INT_ARRAY:
								out.type = type_int;
								out.value.i = expr.value.array->val.iarray[index.value.i-1];
							break;
							case TYPE_BOOL_ARRAY:
								out.type = type_bool;
								out.value.b = expr.value.array->val.barray[index.value.i-1];
							break;
							case TYPE_STRING_ARRAY:
								out.type = type_string;
								out.value.str = strdup (expr.value.array->val.strarray[index.value.i-1]);
							break;
							default:
								phoebe_scripter_output ("exception handler invoked in scripter_ast_evaluate, kind_element; please report this!\n");
							break;
						}
						scripter_ast_value_free (expr);
						scripter_ast_value_free (index);
						return out;
					case type_spectrum:
						if (index.value.i < 1 || index.value.i > 2) {
							phoebe_scripter_output ("there are only two columns in a spectrum variable, aborting.\n");
							scripter_ast_value_free (index);
							scripter_ast_value_free (expr);
							out.type = type_void; return out;
						}
						out.type = type_vector;
						out.value.vec = phoebe_spectrum_get_column (expr.value.spectrum, index.value.i);
						if (!out.value.vec)
							out.type = type_void;
						scripter_ast_value_free (index);
						scripter_ast_value_free (expr);
						return out;
					default:
						phoebe_scripter_output ("the expression is not an array, aborting.\n");
						out.type = type_void;
						return out;
				}
			}

			if (in->value.node.kind == kind_matrix_element) {
				/* 
				 * This function evaluates the 'var[i][j]' expression to the
				 * corresponding matrix element value.
				 */

				char *ident = in->value.node.args->elem->value.variable;
				scripter_ast_value ind1 = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value ind2 = scripter_ast_evaluate (in->value.node.args->next->next->elem);
				scripter_symbol *var = scripter_symbol_lookup (symbol_table, ident);

				if (!var) {
					phoebe_scripter_output ("variable '%s' is not initialized, aborting\n", ident);
					out.type = type_void; return out;
				}
				if (var->link->type != ast_spectrum) {
					phoebe_scripter_output ("variable '%s' is not a spectrum, aborting\n", ident);
					out.type = type_void; return out;
				}
				if (ind1.type != type_int) {
					phoebe_scripter_output ("column index is not an integer, aborting.\n");
					out.type = type_void;
					return out;
				}
				if (ind2.type != type_int) {
					phoebe_scripter_output ("row index is not an integer, aborting.\n");
					out.type = type_void;
					return out;
				}

				if (var->link->type == ast_spectrum) {
					if (ind1.value.i < 1 || ind1.value.i > 2) {
						phoebe_scripter_output ("there are only two columns in a spectrum variable, aborting.\n");
						out.type = type_void; return out;
					}
					if (ind2.value.i < 1 || (ind1.value.i == 1 && ind2.value.i > var->link->value.spectrum->data->bins+1) || (ind1.value.i == 2 && ind2.value.i > var->link->value.spectrum->data->bins) ) {
						phoebe_scripter_output ("row index out of range, aborting.\n");
						out.type = type_void; return out;
					}
					out.type = type_double;
					if (ind1.value.i == 1)
						out.value.d = var->link->value.spectrum->data->range[ind2.value.i-1];
					else
						out.value.d = var->link->value.spectrum->data->val[ind2.value.i-1];
					return out;
				}
			}

			if (in->value.node.kind == kind_export) {
				/* This is the 'export' directive. Its purpose is to put the  */
				/* local identifier to the global symbol table.               */

				scripter_symbol *s;
				char *ident = in->value.node.args->elem->value.variable;

				/* The result of export is always void:                       */
				out.type = type_void;

				s = scripter_symbol_lookup (symbol_table, ident);
				if (!s) {
					phoebe_scripter_output ("identifier '%s' doesn't exist, aborting.\n", ident);
					return out;
				}

				scripter_symbol_commit (symbol_table_lookup (symbol_table, "global"), ident, s->link);
				return out;
			}

			if (in->value.node.kind == kind_unset) {
				char *ident = in->value.node.args->elem->value.variable;
				int status = scripter_symbol_remove (symbol_table, ident);
				if (status != SUCCESS)
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));

				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_set) {
				/*
				 * This is a set statement which assigns a value to the
				 * variable. Synopsis:
				 * 
				 *   set ident = expr
				 *   set idexpr        (for ++, --, +=, -=, *=, and /=)
				 */

				scripter_ast_value value;

				/* Handle identifier expressions first: */
				if (in->value.node.args->elem->value.node.kind == kind_inc    ||
					in->value.node.args->elem->value.node.kind == kind_dec    ||
					in->value.node.args->elem->value.node.kind == kind_incby  ||
					in->value.node.args->elem->value.node.kind == kind_decby  ||
					in->value.node.args->elem->value.node.kind == kind_multby ||
					in->value.node.args->elem->value.node.kind == kind_divby
					) {
					scripter_ast_evaluate (in->value.node.args->elem);
					out.type = type_void;
					return out;
				}

				/* It is a regular set statement. */

				/* The following line always evaluates because of the grammar: */
				char *ident = in->value.node.args->elem->value.variable;

				/* We need to evaluate the expression AST node and to store   */
				/* the evaluated result; otherwise we might end up storing an */
				/* expression that changes its value with its variables.      */
				value = scripter_ast_evaluate (in->value.node.args->next->elem);

				switch (value.type) {
					case type_int:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_int (value.value.i));
					break;
					case type_bool:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_bool (value.value.b));
					break;
					case type_double:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_double (value.value.d));
					break;
					case type_string:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_string (value.value.str));
					break;
					case type_array:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_array (value.value.array));
					break;
					case type_vector:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_vector (value.value.vec));
					break;
					case type_curve:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_curve (value.value.curve));
					break;
					case type_spectrum:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_spectrum (value.value.spectrum));
					break;
					case type_minfeedback:
						scripter_symbol_commit (symbol_table, ident, scripter_ast_add_minfeedback (value.value.feedback));
					break;
					case type_void:
						phoebe_scripter_output ("assignment failed.\n");
					break;
					default:
						phoebe_scripter_output ("exception handler invoked in kind_set, please report this.\n");
					break;
				}

				/* The commit makes a duplicate of a node, so we must free    */
				/* the original:                                              */
				scripter_ast_value_free (value);

				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_set_element) {
				char *ident = in->value.node.args->elem->value.variable;
				scripter_ast_value index = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value value = scripter_ast_evaluate (in->value.node.args->next->next->elem);
				scripter_symbol *var = scripter_symbol_lookup (symbol_table, ident);

				if (!var) {
					phoebe_scripter_output ("variable '%s' is not initialized, aborting\n", ident);
					out.type = type_void;
					return out;
				}

				if (index.type != type_int) {
					phoebe_scripter_output ("element index is not an integer, aborting.\n");
					out.type = type_void;
					return out;
				}

				if (var->link->type == ast_spectrum) {
					if (var->link->type != ast_spectrum) {
						phoebe_scripter_output ("variable '%s' is not a spectrum, aborting\n", ident);
						out.type = type_void; return out;
					}

					if (index.value.i < 1 || index.value.i > 2) {
						phoebe_scripter_output ("index %d for '%s' is out of range [1, 2], aborting.\n", index.value.i, ident);
						out.type = type_void; return out;
					}

					if (value.type != type_vector) {
						phoebe_scripter_output ("spectrum element value must be an array, aborting.\n");
						out.type = type_void;
						return out;
					}

					if (value.value.vec->dim != var->link->value.spectrum->data->bins) {
						phoebe_scripter_output ("the dimensions of the spectrum and array don't match, aborting.\n");
						out.type = type_void;
						return out;
					}

					if (index.value.i == 1)
						phoebe_hist_set_ranges (var->link->value.spectrum->data, value.value.vec);
					else
						phoebe_hist_set_values (var->link->value.spectrum->data, value.value.vec);

					out.type = type_void; return out;
				}

				if (var->link->type == ast_vector) {
					if (var->link->type != ast_vector) {
						phoebe_scripter_output ("variable '%s' is not an array, aborting\n", ident);
						out.type = type_void; return out;
					}

					if (value.type != type_int && value.type != type_double) {
						phoebe_scripter_output ("array element value must be a real value, aborting.\n");
						out.type = type_void;
						return out;
					}

					if (value.type == type_int)
						propagate_int_to_double (&value);

					if (index.value.i < 1 || index.value.i > var->link->value.vec->dim) {
						phoebe_scripter_output ("index %d for '%s' is out of range [%d, %d], aborting.\n", index.value.i, ident, 1, var->link->value.vec->dim);
						out.type = type_void; return out;
					}
					index.value.i--;

					phoebe_debug ("%s[%d] = %f\n", ident, index.value.i, value.value.d);
					var->link->value.vec->val[index.value.i] = value.value.d;
					out.type = type_void; return out;
				}

				phoebe_scripter_output ("you can set element values only for arrays and spectra.\n");
				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_set_matrix_element) {
				char *ident = in->value.node.args->elem->value.variable;
				scripter_ast_value ind1 = scripter_ast_evaluate (in->value.node.args->next->elem);
				scripter_ast_value ind2 = scripter_ast_evaluate (in->value.node.args->next->next->elem);
				scripter_ast_value value = scripter_ast_evaluate (in->value.node.args->next->next->next->elem);
				scripter_symbol *var = scripter_symbol_lookup (symbol_table, ident);

				if (!var) {
					phoebe_scripter_output ("variable '%s' is not initialized, aborting\n", ident);
					out.type = type_void;
					return out;
				}

				if (ind1.type != type_int) {
					phoebe_scripter_output ("column index is not an integer, aborting.\n");
					out.type = type_void;
					return out;
				}

				if (ind2.type != type_int) {
					phoebe_scripter_output ("row index is not an integer, aborting.\n");
					out.type = type_void;
					return out;
				}

				if (var->link->type == ast_spectrum) {
					if (ind1.value.i < 1 || ind1.value.i > 2) {
						phoebe_scripter_output ("column index %d for '%s' is out of range [1, 2], aborting.\n", ind1.value.i, ident);
						out.type = type_void; return out;
					}
					if (ind2.value.i < 1 || (ind1.value.i == 1 && ind2.value.i > var->link->value.spectrum->data->bins+1) || (ind1.value.i == 2 && ind2.value.i > var->link->value.spectrum->data->bins)) {
						phoebe_scripter_output ("row index %d for '%s' is out of range [1, %d], aborting.\n", ind2.value.i, ident, var->link->value.spectrum->data->bins);
						out.type = type_void; return out;
					}

					if (value.type == type_int)
						propagate_int_to_double (&value);

					if (value.type != type_double) {
						phoebe_scripter_output ("spectrum element value must be real, aborting.\n");
						out.type = type_void;
						return out;
					}

					if (ind1.value.i == 1) {
						var->link->value.spectrum->data->range[ind2.value.i-1] = value.value.d;
					}
					else {
						var->link->value.spectrum->data->val[ind2.value.i-1] = value.value.d;
					}
					out.type = type_void;
					return out;
				}

				phoebe_scripter_output ("you can set multi-D element values only for spectra.\n");
				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_builtin) {
				/*
				 * These are built-in mathematical functions. The returned
				 * value is double if everything is ok and void if an error
				 * occured.
				 */

				/* The following assignment is restricted by the grammar, so  */
				/* it cannot fail:                                            */
				char *func  = in->value.node.args->elem->value.function;
				scripter_ast_value value = scripter_ast_evaluate (in->value.node.args->next->elem);

				switch (value.type) {
					case type_int:
						propagate_int_to_double (&value);
						/* Don't break it here, otherwise the propagation     */
						/* won't work! */
					case type_double:
						if (strcmp (func,   "sin") == 0) out.value.d = sin   (value.value.d);
						if (strcmp (func,   "cos") == 0) out.value.d = cos   (value.value.d);
						if (strcmp (func,   "tan") == 0) out.value.d = tan   (value.value.d);
						if (strcmp (func,  "asin") == 0) out.value.d = asin  (value.value.d);
						if (strcmp (func,  "acos") == 0) out.value.d = acos  (value.value.d);
						if (strcmp (func,  "atan") == 0) out.value.d = atan  (value.value.d);
						if (strcmp (func,   "exp") == 0) out.value.d = exp   (value.value.d);
						if (strcmp (func,    "ln") == 0) out.value.d = log   (value.value.d);
						if (strcmp (func,   "log") == 0) out.value.d = log10 (value.value.d);
						if (strcmp (func,  "sqrt") == 0) out.value.d = sqrt  (value.value.d);
						if (strcmp (func,  "norm") == 0) out.value.d =        value.value.d;
						if (strcmp (func,  "rand") == 0) out.value.d = value.value.d * rand () / RAND_MAX;
						if (strcmp (func,  "frac") == 0) out.value.d = fabs (value.value.d - (int) value.value.d);
						if (strcmp (func,   "abs") == 0) out.value.d = fabs (value.value.d);
						out.type = type_double;
						if (strcmp (func,   "int") == 0) {
							out.type = type_int;
							out.value.i = (int) value.value.d;
						}
						if (strcmp (func,   "dim") == 0) {
							out.type = type_int;
							out.value.i = 1;
						}
						if (strcmp (func, "trunc") == 0) {
							out.type = type_int;
							out.value.i = (int) value.value.d;
						}
						if (strcmp (func, "round") == 0) {
							out.type = type_int;
							if (fabs (value.value.d - (int) value.value.d) < 0.5)
								out.value.i = (int) value.value.d;
							else
								if (value.value.d > 0)
									out.value.i = (int) value.value.d + 1;
								else
									out.value.i = (int) value.value.d - 1;
						}
						if (strcmp (func,   "isnan") == 0) {
							out.type = type_bool;
							out.value.b = !(value.value.d == value.value.d);
						}
					break;
					case type_vector:
						out.value.vec = phoebe_vector_new ();
						if (strcmp (func,  "sin") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, sin);
						if (strcmp (func,  "cos") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, cos);
						if (strcmp (func,  "tan") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, tan);
						if (strcmp (func, "asin") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, asin);
						if (strcmp (func, "acos") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, acos);
						if (strcmp (func, "atan") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, atan);
						if (strcmp (func,  "exp") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, exp);
						if (strcmp (func,   "ln") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, log);
						if (strcmp (func,  "log") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, log10);
						if (strcmp (func, "sqrt") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, sqrt);
						if (strcmp (func,  "abs") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, fabs);
						if (strcmp (func, "frac") == 0) phoebe_vector_submit (out.value.vec, value.value.vec, frac);
						out.type = type_vector;
						if (strcmp (func, "norm") == 0) {
							/* This one is special, since it returns double and not vector: */
							phoebe_vector_free (out.value.vec);
							phoebe_vector_norm (&(out.value.d), value.value.vec);
							out.type = type_double;
						}
						if (strcmp (func, "dim") == 0) {
							/* This one is special, since it returns double and not vector: */
							phoebe_vector_free (out.value.vec);
							phoebe_vector_dim (&(out.value.i), value.value.vec);
							out.type = type_int;
						}
						if (strcmp (func,  "rand") == 0 || strcmp   (func, "int") == 0 ||
						    strcmp (func, "trunc") == 0 || strcmp (func, "round") == 0 ) {
							phoebe_vector_free (out.value.vec);
							phoebe_scripter_output ("function %s() cannot act on arrays, aborting.\n", func);
							out.type = type_void;
							return out;
						}
					break;
					case type_array:
						if (strcmp (func, "dim") == 0) {
							out.type = type_int;
							out.value.i = value.value.array->dim;
						}
						else {
							/* Functions on arrays not yet implemented! */
							phoebe_scripter_output ("functions on arrays not yet implemented, sorry!\n");
							out.type = type_void;
						}
					break;
					case type_spectrum:
						if (strcmp (func, "dim") == 0) {
							out.type = type_int;
							out.value.i = value.value.spectrum->data->bins;
						}
						else {
							phoebe_scripter_output ("evaluating functions on spectra not yet implemented, sorry.\n");
							out.type = type_void;
						}
					break;
					case type_string:
						if (strcmp (func, "strlen") == 0) {
							out.type = type_int;
							out.value.i = strlen (value.value.str);
						}
					break;
					case type_curve:
						if (strcmp (func, "dim") == 0) {
							out.type = type_int;
							out.value.i = value.value.curve->indep->dim;
						}
					break;
					default:
						phoebe_scripter_output ("function %s() called with non-numeric argument, aborting.\n", func);
						out.type = type_void;
					break;
				}

				scripter_ast_value_free (value);
				return out;
			}

			if (in->value.node.kind == kind_command) {
				/*
				 * This handler catches all self-standing commands and all
				 * commands passed as expressions and evaluates them. Command
				 * name recognition is done by querying the global table of
				 * all registered commands, scripter_commands.
				 */

				char *command = in->value.node.args->elem->value.string;
				int index;

				/*
				 * Error handling is not necessary here, because grammar rules
				 * already take care of it.
				 */

				scripter_command_get_index (command, &index);
				out = scripter_commands->command[index]->func (in->value.node.args->next);
				return out;
			}

			if (in->value.node.kind == kind_macro) {
				/*
				 * This directive defines the macro (procedure).
				 * 
				 * Macros are self-contained pieces of input returning void.
				 * They take arbitrarily many arguments enclosed in parenthe-
				 * ses, which are here passed as the last element of the AST
				 * list.
				 */

				scripter_ast *ident = in->value.node.args->elem;

				/* The following assignment will always succeed because of    */
				/* grammar rules:                                             */
				char *id = ident->value.variable;

				/* Add an entry to the global symbol list for the macro name: */
				scripter_symbol_commit (symbol_table_lookup (symbol_table, "global"), id, scripter_ast_duplicate (in));

				/* That's it, return void.                                    */
				out.type = type_void; return out;
			}

			if (in->value.node.kind == kind_define) {
				/*
				 * This directive defines a user-supplied function.         
				 * 
				 * Functions are self-contained pieces of input returning a
				 * numerical value. They take arbitrarily many arguments
				 * enclosed in parentheses, which are here passed as the last
				 * element of the AST list.
				 *
				 * For a discussion on AST list manipulation see the comments
				 * to kind_macro above.
				 */

				scripter_ast *ident = in->value.node.args->elem;

				/* The following assignment will always succeed because of    */
				/* grammar rules:                                             */
				char *id = ident->value.variable;

				/* Add an entry to the symbol list for the function name:     */
				scripter_symbol_commit (symbol_table_lookup (symbol_table, "global"), id, scripter_ast_duplicate (in));

				/* That's it, return void.                                    */
				out.type = type_void; return out;
			}

			if (in->value.node.kind == kind_func) {
				/*
				 * This event is called by the user-defined function (the
				 * 'define' directive). All user-defined functions return a
				 * numeric value.
				 */

				scripter_ast_list *arguments = NULL;
				scripter_ast_list *params;
				scripter_ast      *body;

				scripter_ast_list *list, *arglist;

				scripter_ast_list  *s;
				scripter_ast_value retval;

				scripter_symbol *id = scripter_symbol_lookup (symbol_table, in->value.node.args->elem->value.variable);

				if (!id) {
					phoebe_scripter_output ("function '%s' not defined.\n", in->value.node.args->elem->value.variable);
					out.type = type_void; return out;
				}
				if ( id->link->type != ast_node || id->link->value.node.kind != kind_define ) {
					phoebe_scripter_output ("identifier '%s' is not a function.\n", in->value.node.args->elem->value.variable);
					out.type = type_void; return out;
				}

				body   = id->link->value.node.args->next->elem;
				params = id->link->value.node.args->next->next;

				/* Now we have to evaluate the function arguments: */
				for (list = in->value.node.args->next; list; list = list->next) {
					scripter_ast_value val = scripter_ast_evaluate (list->elem);
					if (val.type == type_void) {
						phoebe_scripter_output ("invalid argument in function call, aborting.\n");
						scripter_ast_list_free (arguments);
						out.type = type_void; return out;
					}
					switch (val.type) {
						case type_int:
							arguments = scripter_ast_construct_list (scripter_ast_add_int (val.value.i), arguments);
						break;
						case type_bool:
							arguments = scripter_ast_construct_list (scripter_ast_add_bool (val.value.b), arguments);
						break;
						case type_double:
							arguments = scripter_ast_construct_list (scripter_ast_add_double (val.value.d), arguments);
						break;
						case type_string:
							arguments = scripter_ast_construct_list (scripter_ast_add_string (val.value.str), arguments);
						break;
						case type_vector:
							arguments = scripter_ast_construct_list (scripter_ast_add_vector (val.value.vec), arguments);
						break;
						case type_spectrum:
							arguments = scripter_ast_construct_list (scripter_ast_add_spectrum (val.value.spectrum), arguments);
						break;
						case type_minfeedback:
							arguments = scripter_ast_construct_list (scripter_ast_add_minfeedback (val.value.feedback), arguments);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in function evaluation, please report this.\n");
							scripter_ast_list_free (arguments);
							scripter_ast_value_free (val);
							out.type = type_void;
							return out;
						break;
					}
					scripter_ast_value_free (val);
				}

				arguments = scripter_ast_reverse_list (arguments, NULL);

				if (scripter_ast_list_length (params) != scripter_ast_list_length (arguments)) {
					if (scripter_ast_list_length (params) == 1) phoebe_scripter_output ("function %s expects %d parameter, but %d are passed.\n", in->value.node.args->elem->value.variable, scripter_ast_list_length (params), scripter_ast_list_length (arguments));
					else if (scripter_ast_list_length (arguments) != 1) phoebe_scripter_output ("function %s expects %d parameters, but %d are passed.\n", in->value.node.args->elem->value.variable, scripter_ast_list_length (params), scripter_ast_list_length (arguments));
					else phoebe_scripter_output ("function %s expects %d parameters, but %d are passed.\n", in->value.node.args->elem->value.variable, scripter_ast_list_length (params), scripter_ast_list_length (arguments));
					scripter_ast_list_free (arguments);
					out.type = type_void; return out;
				}

				/* Start a new environment where the function will be evaluated: */
				symbol_table = symbol_table_add (symbol_table, in->value.node.args->elem->value.variable);

				for (list = params, arglist = arguments; list; list = list->next, arglist = arglist->next)
					scripter_symbol_commit (symbol_table, list->elem->value.variable, scripter_ast_duplicate (arglist->elem));

				scripter_ast_list_free (arguments);

				/* We shall now traverse all statements in the function body  */
				/* until there are no more statements or until the return     */
				/* statement is encountered, whichever comes first:           */

				s = body->value.node.args;

				/* Since 's' are statements, their evaluation always returns  */
				/* void. What we need is the child to the statement, which in */
				/* case of the return statement returns non-void value. That  */
				/* is why we are evaluating s->elem->value.node.args->elem    */
				/* instead of just s->elem below.                             */

				out.type = type_void;
				while (s) {
					retval = scripter_ast_evaluate (s->elem->value.node.args->elem);
					if (s->elem->value.node.args->elem->value.node.kind == kind_return) {
						out = retval;
						break;
					}
					s = s->next;
				}

				/* The function was executed, free the symbol table: */
				symbol_table = symbol_table_remove (symbol_table, in->value.node.args->elem->value.variable);

				return out;
			}

			if (in->value.node.kind == kind_execute_macro) {
				scripter_ast_list *arguments = NULL;
				scripter_ast_list *params;
				scripter_ast      *body;
				scripter_ast_list *list, *arglist;

				scripter_symbol *id = scripter_symbol_lookup (symbol_table, in->value.node.args->elem->value.variable);

				if (!id) {
					phoebe_scripter_output ("macro '%s' not defined.\n", in->value.node.args->elem->value.variable);
					out.type = type_void; return out;
				}
				if (id->link->type != ast_node || id->link->value.node.kind != kind_macro) {
					if (id->link->value.node.kind == kind_define)
						phoebe_scripter_output ("'%s' is a function, not a macro; aborting.\n", in->value.node.args->elem->value.variable);
					else
						phoebe_scripter_output ("'%s' is an identifier, not a macro; aborting.\n", in->value.node.args->elem->value.variable);
					out.type = type_void; return out;
				}

				body   = id->link->value.node.args->next->elem;
				params = id->link->value.node.args->next->next;
				
				/* Now we have to evaluate the macro arguments: */
				for (list = in->value.node.args->next; list; list = list->next) {
					scripter_ast_value val = scripter_ast_evaluate (list->elem);

					if (val.type == type_void) {
						phoebe_scripter_output ("invalid argument in macro call, aborting.\n");
						scripter_ast_list_free (arguments);
						out.type = type_void; return out;
					}
					switch (val.type) {
						case type_int:
							arguments = scripter_ast_construct_list (scripter_ast_add_int (val.value.i), arguments);
						break;
						case type_bool:
							arguments = scripter_ast_construct_list (scripter_ast_add_bool (val.value.b), arguments);
						break;
						case type_double:
							arguments = scripter_ast_construct_list (scripter_ast_add_double (val.value.d), arguments);
						break;
						case type_string:
							arguments = scripter_ast_construct_list (scripter_ast_add_string (val.value.str), arguments);
						break;
						case type_vector:
							arguments = scripter_ast_construct_list (scripter_ast_add_vector (val.value.vec), arguments);
						break;
						case type_qualifier:
							arguments = scripter_ast_construct_list (scripter_ast_add_qualifier (val.value.str), arguments);
						break;
						case type_spectrum:
							arguments = scripter_ast_construct_list (scripter_ast_add_spectrum (val.value.spectrum), arguments);
						break;
						case type_minfeedback:
							arguments = scripter_ast_construct_list (scripter_ast_add_minfeedback (val.value.feedback), arguments);
						break;
						default:
							phoebe_scripter_output ("handling exception invoked in kind_execute_macro, please report this.\n");
							scripter_ast_list_free (arguments);
							scripter_ast_value_free (val);
							out.type = type_void;
							return out;
						break;
					}
					scripter_ast_value_free (val);
				}

				arguments = scripter_ast_reverse_list (arguments, NULL);

				if (scripter_ast_list_length (params) != scripter_ast_list_length (arguments)) {
					if (scripter_ast_list_length (params) == 1) phoebe_scripter_output ("macro %s expects %d parameter, but %d are passed.\n", in->value.node.args->elem->value.variable, scripter_ast_list_length (params), scripter_ast_list_length (arguments));
					else if (scripter_ast_list_length (arguments) != 1) phoebe_scripter_output ("macro %s expects %d parameters, but %d are passed.\n", in->value.node.args->elem->value.variable, scripter_ast_list_length (params), scripter_ast_list_length (arguments));
					else phoebe_scripter_output ("macro %s expects %d parameters, but %d are passed.\n", in->value.node.args->elem->value.variable, scripter_ast_list_length (params), scripter_ast_list_length (arguments));
					scripter_ast_list_free (arguments);
					out.type = type_void; return out;
				}

				/* Start a new environment where the function will be evaluated: */
				symbol_table = symbol_table_add (symbol_table, in->value.node.args->elem->value.variable);

				for (list = params, arglist = arguments; list; list = list->next, arglist = arglist->next)
					scripter_symbol_commit (symbol_table, list->elem->value.variable, scripter_ast_duplicate (arglist->elem));
				
				scripter_ast_list_free (arguments);

				scripter_ast_evaluate (body);
				
				symbol_table = symbol_table_remove (symbol_table, in->value.node.args->elem->value.variable);
				
				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_calc)
				scripter_directive_calc (in->value.node.args);

			if (in->value.node.kind == kind_clear)
				scripter_directive_clear (in->value.node.args);

			if (in->value.node.kind == kind_execute)
				scripter_directive_execute (in->value.node.args);

			if (in->value.node.kind == kind_help)
				scripter_directive_help (in->value.node.args);

			if (in->value.node.kind == kind_info)
				scripter_directive_info (in->value.node.args);

			if (in->value.node.kind == kind_list)
				scripter_directive_list (in->value.node.args);

			if (in->value.node.kind == kind_print)
				scripter_directive_print (in->value.node.args);

			if (in->value.node.kind == kind_quit)
				scripter_directive_quit (in->value.node.args);

			if (in->value.node.kind == kind_show)
				scripter_directive_show (in->value.node.args);

			if (in->value.node.kind == kind_stdump)
				scripter_directive_stdump (in->value.node.args);

			if (in->value.node.kind == kind_if) {
				scripter_directive_if (in->value.node.args);
				out.type = type_void;
				return out;
			}

			if (in->value.node.kind == kind_add) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_add (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_sub) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_subtract (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_mul) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_multiply (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_div) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_divide (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_idiv) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				if (val1.type == type_int && val2.type == type_int) {
					out.type = type_int;
					out.value.i = val1.value.i % val2.value.i;
					scripter_ast_value_free (val1);
					scripter_ast_value_free (val2);
					return out;
				}
				out.type = type_void;
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_raise) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_raise (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_and) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				if (val1.type == type_bool && val2.type == type_bool) {
					out.type = type_bool;
					out.value.b = (val1.value.b && val2.value.b);
					scripter_ast_value_free (val1);
					scripter_ast_value_free (val2);
					return out;
				}
				out.type = type_void;
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_or) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				if (val1.type == type_bool && val2.type == type_bool) {
					out.type = type_bool;
					out.value.b = (val1.value.b || val2.value.b);
					scripter_ast_value_free (val1);
					scripter_ast_value_free (val2);
					return out;
				}
				out.type = type_void;
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_equal) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_equal (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_nequal) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_nequal (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_lequal) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_lequal (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_gequal) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_gequal (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_greater) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_greater (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}

			if (in->value.node.kind == kind_less) {
				scripter_ast_value val1 = scripter_ast_evaluate (in->value.node.args->elem);
				scripter_ast_value val2 = scripter_ast_evaluate (in->value.node.args->next->elem);

				status = scripter_ast_values_less (&out, val1, val2);
				if (status != SUCCESS) {
					phoebe_scripter_output ("%s", phoebe_scripter_error (status));
					out.type = type_void;
				}
				scripter_ast_value_free (val1);
				scripter_ast_value_free (val2);
				return out;
			}
 		}

	out.type = type_void;
	return out;
}

int scripter_ast_value_print (scripter_ast_value val)
{
	switch (val.type) {
		case (type_int):
			fprintf (PHOEBE_output, "%d", val.value.i);
		break;
		case (type_bool):
			if (val.value.b == TRUE)
				fprintf (PHOEBE_output, "TRUE");
			else
				fprintf (PHOEBE_output, "FALSE");
		break;
		case (type_double):
			fprintf (PHOEBE_output, "%lf", val.value.d);
		break;
		case (type_string):
			fprintf (PHOEBE_output, "%s", val.value.str);
		break;
		case (type_array):
			phoebe_array_print (val.value.array);
		break;
		case (type_vector):
			phoebe_vector_print (val.value.vec);
		break;
		case (type_qualifier):
			phoebe_scripter_output ("Please use get_parameter_value() to access qualifier values.\n");
		break;
		case (type_curve):
			phoebe_curve_print (val.value.curve);
		break;
		case (type_spectrum):
			phoebe_spectrum_print (val.value.spectrum);
		break;
		case (type_minfeedback):
			phoebe_minimizer_feedback_print (val.value.feedback);
		break;
		default:
			phoebe_scripter_output ("exception handler invoked in scripter_ast_value_print (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_get_type (scripter_ast *in)
{
	return in->type;
}

char *scripter_ast_value_type_get_name (int type)
{
	switch (type) {
		case type_int:         return "integer";
		case type_bool:        return "boolean";
		case type_double:      return "double";
		case type_string:      return "string";
		case type_vector:      return "vector";
		case type_array:       return "array";
		case type_curve:       return "curve";
		case type_spectrum:    return "spectrum";
		case type_qualifier:   return "qualifier";
		case type_function:    return "built-in function";
		case type_minfeedback: return "minimizer feedback";
		case type_any:         return "non-void";
		case type_void:        return "void";
		default:
			phoebe_scripter_output ("exception handler invoked in scripter_ast_value_type_get_name (), please report this!\n");
			return "invalid";
	}
}

/******************************************************************************/
/*                     A U X I L I A R Y   F U N C T I O N S                  */
/******************************************************************************/

PHOEBE_vector *phoebe_vector_new_from_list (scripter_ast_list *list)
{
	/*
	 * This function returns a vector which is defined by a list, e.g.
	 * set a={1,2,3}. If the vector is not valid, NULL is returned.
	 */

	PHOEBE_vector *vec = phoebe_vector_new ();
	int dim = scripter_ast_list_length (list);
	scripter_ast_value val;
	int i;

	phoebe_vector_alloc (vec, dim);

	for (i = 0; i < dim; i++) {
		val = scripter_ast_evaluate (list->elem);
		if (val.type == type_int) {
			int value = val.value.i; val.type = type_double; val.value.d = (double) value;
		}
		if (val.type != type_double) {
			phoebe_vector_free (vec);
			return NULL;
		}
		vec->val[i] = val.value.d;
		list = list->next;
	}

	return vec;
}

PHOEBE_array *phoebe_array_new_from_list (scripter_ast_list *list)
{
	/*
	 * This function returns a string array which is defined by the list, e.g.
	 * set a={"a1", "a2", "b"}. If the array is not valid, NULL is returned.
	 */

	PHOEBE_array *array = phoebe_array_new (TYPE_STRING_ARRAY);
	int dim = scripter_ast_list_length (list);
	scripter_ast_value val;
	int i;

	phoebe_array_alloc (array, dim);

	for (i = 0; i < dim; i++) {
		val = scripter_ast_evaluate (list->elem);
		if (val.type != type_string) {
			phoebe_array_free (array);
			return NULL;
		}
		array->val.strarray[i] = strdup (val.value.str);
		list = list->next;
	}

	return array;
}
