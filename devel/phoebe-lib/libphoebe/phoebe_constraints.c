#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_constraints.h"
#include "phoebe_constraints.lng.h"
#include "phoebe_error_handling.h"
#include "phoebe_parameters.h"

char *phoebe_ast_node_type_get_name (PHOEBE_node_type type)
{
	switch (type) {
		case PHOEBE_NODE_TYPE_CONSTRAINT:
			return "constraint";
		break;
		case PHOEBE_NODE_TYPE_PARAMETER:
			return "parameter";
		break;
		case PHOEBE_NODE_TYPE_ADD:
			return "+";
		break;
		case PHOEBE_NODE_TYPE_SUB:
			return "-";
		break;
		case PHOEBE_NODE_TYPE_MUL:
			return "*";
		break;
		case PHOEBE_NODE_TYPE_DIV:
			return "/";
		break;
		case PHOEBE_NODE_TYPE_POT:
			return "^";
		break;
		case PHOEBE_NODE_TYPE_BUILTIN:
			return "func";
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_ast_node_type_get_name (), please report this!\n");
	}

	return NULL;
}

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

PHOEBE_ast *phoebe_ast_add_parameter (char *qualifier)
{
	PHOEBE_ast *ast    = phoebe_malloc (sizeof (*ast));
	ast->type          = PHOEBE_AST_PARAMETER;
	ast->val.str       = strdup (qualifier);

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

PHOEBE_ast_list *phoebe_ast_list_reverse (PHOEBE_ast_list *c, PHOEBE_ast_list *p)
{
	/*
	 * When creating an AST list, the order of arguments is last element to
	 * the first, e.g. f(x,y,z) would create z->y->x parentage. This isn't what
	 * we want and the following function reverses this order through recursion
	 * by calling itself. The result after that is x->y->z, as it should be.
	 */

	PHOEBE_ast_list *rev;

	if (!c) return p;
	rev = phoebe_ast_list_reverse (c->next, c);
	c->next = p;

	return rev;
}

PHOEBE_ast *phoebe_ast_duplicate (PHOEBE_ast *ast)
{
	PHOEBE_ast *out;
	PHOEBE_ast_list *args;
	PHOEBE_ast_list *argcopies = NULL;

	switch (ast->type) {
		case PHOEBE_AST_INDEX:
			out = phoebe_ast_add_index (ast->val.idx);
		break;
		case PHOEBE_AST_NUMVAL:
			out = phoebe_ast_add_numval (ast->val.numval);
		break;
		case PHOEBE_AST_STRING:
			out = phoebe_ast_add_builtin (ast->val.str);
		break;
		case PHOEBE_AST_PARAMETER:
			out = phoebe_ast_add_parameter (ast->val.str);
		break;
		case PHOEBE_AST_NODE:
			args = ast->val.node.args;
			while (args) {
				argcopies = phoebe_ast_construct_list (phoebe_ast_duplicate (args->elem), argcopies);
				args = args->next;
			}
			out = phoebe_ast_add_node (ast->val.node.type, phoebe_ast_list_reverse (argcopies, NULL));
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_ast_duplicate (), please report this.\n");
			out = NULL;
		break;
	}

	return out;
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
			val.val.str = ast->val.str;
		break;
		case PHOEBE_AST_NODE:
			switch (ast->val.node.type) {
				case PHOEBE_NODE_TYPE_CONSTRAINT:
					val.type = PHOEBE_AST_VALUE_VOID;
					if (phoebe_ast_list_length (ast->val.node.args) == 2) {
						PHOEBE_ast_value parv = phoebe_ast_evaluate (ast->val.node.args->elem);
						PHOEBE_parameter *par = phoebe_parameter_lookup (parv.val.str);
						PHOEBE_ast_value expr = phoebe_ast_evaluate (ast->val.node.args->next->elem);
						phoebe_debug ("    constraining %s to %lf\n", par->qualifier, expr.val.numval);
						/*
						 * Do not use phoebe_parameter_set_value () here,
						 * access the table element directly; otherwise there
						 * would be an infinite recursion.
						 */
						par->value.d = expr.val.numval;
					}
					if (phoebe_ast_list_length (ast->val.node.args) == 3) {
						PHOEBE_ast_value parv = phoebe_ast_evaluate (ast->val.node.args->elem);
						PHOEBE_parameter *par = phoebe_parameter_lookup (parv.val.str);
						PHOEBE_ast_value idx = phoebe_ast_evaluate (ast->val.node.args->next->elem);
						PHOEBE_ast_value expr = phoebe_ast_evaluate (ast->val.node.args->next->next->elem);
						phoebe_debug ("    constraining %s[%d] to %lf\n", par->qualifier, idx.val.idx, expr.val.numval);
						/*
						 * Do not use phoebe_parameter_set_value () here,
						 * access the table element directly; otherwise there
						 * would be an infinite recursion.
						 */
						par->value.vec->val[idx.val.idx-1] = expr.val.numval;
					}
				break;
				case PHOEBE_NODE_TYPE_PARAMETER:
					val.type = PHOEBE_AST_VALUE_PARAMETER;
					if (phoebe_ast_list_length (ast->val.node.args) == 1)
						phoebe_parameter_get_value (phoebe_parameter_lookup (ast->val.node.args->elem->val.str), &val.val.numval);
					if (phoebe_ast_list_length (ast->val.node.args) == 2)
						phoebe_parameter_get_value (phoebe_parameter_lookup (ast->val.node.args->elem->val.str), ast->val.node.args->next->elem->val.idx-1, &val.val.numval);
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
			printf ("| %s\n", in->val.str);
		break;
		case PHOEBE_AST_NODE:
			depth += 1;
			printf ("| %s\n", phoebe_ast_node_type_get_name (in->val.node.type));
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

int intern_constraint_add_to_table (PHOEBE_ast *ast)
{
	PHOEBE_ast_list *constraint = phoebe_malloc (sizeof (*constraint));

	constraint->elem = ast;
	constraint->next = PHOEBE_pt->lists.constraints;
	PHOEBE_pt->lists.constraints = constraint;

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

	YY_BUFFER_STATE state = pc_scan_string (cstr);

	/*
	 * The following bison call initiates the parsing on that string. It
	 * calls yylex (), which is the lexing part, so we don't have to worry
	 * about it. yyparse () returns a status 0 (success), 1 (syntax error)
	 * or 2 (memory exhausted).
	 */

	status = pcparse ();

	switch (status) {
		case 0:
			phoebe_debug ("phoebe_constraint_new (): parsing successful.\n");
		break;
		case 1:
			phoebe_lib_error ("phoebe_constraint_new (): syntax error encountered in \"%s\", aborting.\n", cstr);
		break;
		case 2:
			phoebe_lib_error ("phoebe_constraint_new (): memory exhausted while parsing \"%s\".\n", cstr);
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_constraint_new (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	/*
	 * Parsing is done, AST is created, we now must free the buffer.
	 */

	pc_delete_buffer (state);

	return SUCCESS;
}

char *phoebe_constraint_get_qualifier (PHOEBE_ast *constraint)
{
	/**
	 * phoebe_constraint_get_qualifier
	 *
	 * Returns a newly allocated qualifier that appears in the constraint.
	 */

	char *qualifier;
	PHOEBE_parameter *par = phoebe_parameter_lookup (constraint->val.node.args->elem->val.str);

	if (par->type == TYPE_DOUBLE)
		qualifier = strdup (par->qualifier);
	else {
		qualifier = phoebe_malloc ((strlen(par->qualifier)+5)*sizeof (*qualifier));
		sprintf (qualifier, "%s[%d]", par->qualifier, constraint->val.node.args->next->elem->val.idx);
	}

	return qualifier;
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

int phoebe_constraint_satisfy_all ()
{
	PHOEBE_ast_list *constraint;

	constraint = PHOEBE_pt->lists.constraints;
	while (constraint) {
		phoebe_ast_evaluate (constraint->elem);
		constraint = constraint->next;
	}

	return SUCCESS;
}

int phoebe_free_constraints ()
{
	PHOEBE_ast_list *c = PHOEBE_pt->lists.constraints;

	while (c) {
		PHOEBE_pt->lists.constraints = c->next;
		phoebe_ast_free (c->elem);
		free (c);
		c = PHOEBE_pt->lists.constraints;
	}

	return SUCCESS;
}
