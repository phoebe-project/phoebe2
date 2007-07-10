#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_constraints.h"
#include "phoebe_constraints.lex.h"
#include "phoebe_error_handling.h"
#include "phoebe_parameters.h"

PHOEBE_ast *phoebe_ast_add_index (int idx)
{
	PHOEBE_ast *ast = phoebe_malloc (sizeof (*ast));
	ast->type = PHOEBE_AST_INDEX;
	ast->val.idx = idx;

	return ast;
}

PHOEBE_ast *phoebe_ast_add_numval (double numval)
{
	PHOEBE_ast *ast = phoebe_malloc (sizeof (*ast));
	ast->type       = PHOEBE_AST_NUMVAL;
	ast->val.numval = numval;

	return ast;
}

PHOEBE_ast *phoebe_ast_add_builtin (char *builtin)
{
	PHOEBE_ast *ast    = phoebe_malloc (sizeof (*ast));
	ast->type          = PHOEBE_AST_STRING;
	ast->val.str       = strdup (builtin);

	return ast;
}

PHOEBE_ast *phoebe_ast_add_parameter (PHOEBE_parameter *par)
{
	PHOEBE_ast *ast    = phoebe_malloc (sizeof (*ast));
	ast->type          = PHOEBE_AST_PARAMETER;
	ast->val.par       = par;

	return ast;
}

PHOEBE_ast *phoebe_ast_add_node (PHOEBE_node_type type, PHOEBE_ast_list *args)
{
	PHOEBE_ast *ast    = phoebe_malloc (sizeof (*ast));
	ast->type          = PHOEBE_AST_NODE;
	ast->val.node.type = type;
	ast->val.node.args = args;

	return ast;
}

int phoebe_ast_list_length (PHOEBE_ast_list *list)
{
	PHOEBE_ast_list *ptr = list;
	int i = 0;

	while (ptr) {
		i++;
		ptr = ptr->next;
	}

	return i;
}

PHOEBE_ast_value phoebe_ast_evaluate (PHOEBE_ast *ast)
{
	PHOEBE_ast_value val;

	if (!ast) {
		phoebe_lib_error ("the passed AST is not initialized, aborting.\n");
		val.type = PHOEBE_AST_VALUE_VOID;
		return val;
	}

	switch (ast->type) {
		case PHOEBE_AST_INDEX:
			val.type = PHOEBE_AST_VALUE_INT;
			val.val.idx = ast->val.idx;
		break;
		case PHOEBE_AST_NUMVAL:
			val.type = PHOEBE_AST_VALUE_DOUBLE;
			val.val.numval = ast->val.numval;
		break;
		case PHOEBE_AST_STRING:
			val.type = PHOEBE_AST_VALUE_STRING;
			val.val.str = ast->val.str;
		break;
		case PHOEBE_AST_PARAMETER:
			val.type = PHOEBE_AST_VALUE_PARAMETER;
			val.val.par = ast->val.par;
		break;
		case PHOEBE_AST_NODE:
			switch (ast->val.node.type) {
				case PHOEBE_NODE_TYPE_CONSTRAINT: {
					if (phoebe_ast_list_length (ast->val.node.args) == 2) {
						PHOEBE_ast_value par = phoebe_ast_evaluate (ast->val.node.args->elem);
						PHOEBE_ast_value expr = phoebe_ast_evaluate (ast->val.node.args->next->elem);
						printf ("call to set %s to %lf\n", par.val.par->qualifier, expr.val.numval);
					}
					if (phoebe_ast_list_length (ast->val.node.args) == 3) {
						PHOEBE_ast_value par = phoebe_ast_evaluate (ast->val.node.args->elem);
						PHOEBE_ast_value idx = phoebe_ast_evaluate (ast->val.node.args->next->elem);
						PHOEBE_ast_value expr = phoebe_ast_evaluate (ast->val.node.args->next->next->elem);
						printf ("call to set %s[%d] to %lf\n", par.val.par->qualifier, idx.val.idx, expr.val.numval);
					}
				}
				break;
				case PHOEBE_NODE_TYPE_PARAMETER:
					val.type = PHOEBE_AST_VALUE_PARAMETER;
					if (phoebe_ast_list_length (ast->val.node.args) == 1)
						phoebe_parameter_get_value (ast->val.node.args->elem->val.par, &val.val.numval);
					if (phoebe_ast_list_length (ast->val.node.args) == 2)
						phoebe_parameter_get_value (ast->val.node.args->elem->val.par, ast->val.node.args->next->elem->val.idx, &val.val.numval);
				break;
				case PHOEBE_NODE_TYPE_ADD: {
					PHOEBE_ast_value op1 = phoebe_ast_evaluate (ast->val.node.args->elem);
					PHOEBE_ast_value op2 = phoebe_ast_evaluate (ast->val.node.args->next->elem);
					val.type = PHOEBE_AST_VALUE_DOUBLE;
					val.val.numval = op1.val.numval + op2.val.numval;
				}
				break;
				case PHOEBE_NODE_TYPE_SUB: {
					PHOEBE_ast_value op1 = phoebe_ast_evaluate (ast->val.node.args->elem);
					PHOEBE_ast_value op2 = phoebe_ast_evaluate (ast->val.node.args->next->elem);
					val.type = PHOEBE_AST_VALUE_DOUBLE;
					val.val.numval = op1.val.numval - op2.val.numval;
				}
				break;
				case PHOEBE_NODE_TYPE_MUL: {
					PHOEBE_ast_value op1 = phoebe_ast_evaluate (ast->val.node.args->elem);
					PHOEBE_ast_value op2 = phoebe_ast_evaluate (ast->val.node.args->next->elem);
					val.type = PHOEBE_AST_VALUE_DOUBLE;
					val.val.numval = op1.val.numval * op2.val.numval;
				}
				break;
				case PHOEBE_NODE_TYPE_DIV: {
					PHOEBE_ast_value op1 = phoebe_ast_evaluate (ast->val.node.args->elem);
					PHOEBE_ast_value op2 = phoebe_ast_evaluate (ast->val.node.args->next->elem);
					val.type = PHOEBE_AST_VALUE_DOUBLE;
					val.val.numval = op1.val.numval / op2.val.numval;
				}
				break;
				case PHOEBE_NODE_TYPE_POT: {
					PHOEBE_ast_value op1 = phoebe_ast_evaluate (ast->val.node.args->elem);
					PHOEBE_ast_value op2 = phoebe_ast_evaluate (ast->val.node.args->next->elem);
					val.type = PHOEBE_AST_VALUE_DOUBLE;
					val.val.numval = pow (op1.val.numval, op2.val.numval);
				}
				break;
				case PHOEBE_NODE_TYPE_BUILTIN: {
					PHOEBE_ast_value func = phoebe_ast_evaluate (ast->val.node.args->elem);
					PHOEBE_ast_value arg  = phoebe_ast_evaluate (ast->val.node.args->next->elem);

					val.type = PHOEBE_AST_VALUE_DOUBLE;
					if (strcmp (func.val.str, "sin") == 0)
						val.val.numval = sin (arg.val.numval);
					if (strcmp (func.val.str, "cos") == 0)
						val.val.numval = cos (arg.val.numval);
					if (strcmp (func.val.str, "tan") == 0)
						val.val.numval = tan (arg.val.numval);
					if (strcmp (func.val.str, "asin") == 0)
						val.val.numval = asin (arg.val.numval);
					if (strcmp (func.val.str, "acos") == 0)
						val.val.numval = acos (arg.val.numval);
					if (strcmp (func.val.str, "atan") == 0)
						val.val.numval = atan (arg.val.numval);
					if (strcmp (func.val.str, "exp") == 0)
						val.val.numval = exp (arg.val.numval);
					if (strcmp (func.val.str, "ln") == 0)
						val.val.numval = log (arg.val.numval);
					if (strcmp (func.val.str, "log") == 0)
						val.val.numval = log10 (arg.val.numval);
					if (strcmp (func.val.str, "sqrt") == 0)
						val.val.numval = sqrt (arg.val.numval);
				}
				break;
				default:
					phoebe_lib_error ("exception handler invoked in phoebe_ast_evaluate () node switch, please report this!\n");
					val.type = PHOEBE_AST_VALUE_VOID;
				break;
			}
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_ast_evaluate () type switch, please report this!\n");
			val.type = PHOEBE_AST_VALUE_VOID;
		break;
	}

	return val;
}

int phoebe_ast_print (int depth, PHOEBE_ast *in)
{
	int i;
	PHOEBE_ast_list *args;

	for (i = 0; i < depth; i++)
		printf ("| ");

	switch (in->type) {
		case PHOEBE_AST_INDEX:
			printf ("| %d\n", in->val.idx);
		break;
		case PHOEBE_AST_NUMVAL:
			printf ("| %lf\n", in->val.numval);
		break;
		case PHOEBE_AST_STRING:
			printf ("| \"%s\"\n", in->val.str);
		break;
		case PHOEBE_AST_PARAMETER:
			printf ("| %s\n", in->val.par->qualifier);
		break;
		case PHOEBE_AST_NODE:
			depth += 1;
			printf ("| %d\n", in->val.node.type);
			args = in->val.node.args;
			while (args) {
				phoebe_ast_print (depth, args->elem);
				args = args->next;
			}
		break;
		default:
			printf ("exception handler invoked in scripter_ast_print (), please report this.\n");
		break;
	}

	return SUCCESS;
}

PHOEBE_ast_list *phoebe_ast_construct_list (PHOEBE_ast *ast, PHOEBE_ast_list *list)
{
	PHOEBE_ast_list *out = phoebe_malloc (sizeof (*out));
	out->elem = ast;
	out->next = list;
	return out;
}

int phoebe_constraint_add_to_table (PHOEBE_ast *ast)
{
	PHOEBE_constraint *constraint = phoebe_malloc (sizeof (*constraint));

	constraint->func = ast;
	constraint->next = PHOEBE_ct;
	PHOEBE_ct = constraint;

	return SUCCESS;
}

int phoebe_constraint_new (const char *cstr)
{
	int status;

	/*
	 * The following flex call creates a copy of the string 'cstr' and
	 * works on that copy - the original will be preserved and the buffer
	 * needs to be freed after a successful parse.
	 */

	YY_BUFFER_STATE state = yy_scan_string (cstr);

	/*
	 * The following bison call initiates the parsing on that string. It
	 * calls yylex (), which is the lexing part, so we don't have to worry
	 * about it. yyparse () returns a status 0 (success), 1 (syntax error)
	 * or 2 (memory exhausted).
	 */

	status = yyparse ();

	switch (status) {
		case 0:
			printf ("Parsing was successful.\n");
		break;
		case 1:
			printf ("Syntax error encountered.\n");
		break;
		case 2:
			printf ("Memory exhausted.\n");
		break;
		default:
			printf ("exception handler invoked.\n");
	}

	/*
	 * Parsing is done, AST is created, we now must free the buffer.
	 */

	yy_delete_buffer (state);

	return SUCCESS;
}

int phoebe_ast_free (PHOEBE_ast *ast)
{
	PHOEBE_ast_list *list, *next;

	switch (ast->type) {
		case PHOEBE_AST_INDEX:
			/* Nothing to be done */
		case PHOEBE_AST_NUMVAL:
			/* Nothing to be done */
		break;
		case PHOEBE_AST_STRING:
			free (ast->val.str);
		break;
		case PHOEBE_AST_PARAMETER:
			/* Nothing to be done */
		break;
		case PHOEBE_AST_NODE:
			list = ast->val.node.args;
			if (!list)
				return SUCCESS;

			next = list->next;
			while (next) {
				phoebe_ast_free (list->elem);
				free (list);
				list = next;
				next = list->next;
			}
			phoebe_ast_free (list->elem);
			free (list);
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_ast_free (), please report this.\n");
		break;
	}
	free (ast);

	return SUCCESS;
}

int phoebe_free_constraints ()
{
	while (PHOEBE_ct) {
		PHOEBE_constraint *c = PHOEBE_ct->next;
		phoebe_ast_free (PHOEBE_ct->func);
		free (PHOEBE_ct);
		PHOEBE_ct = c;
	}

	return SUCCESS;
}
