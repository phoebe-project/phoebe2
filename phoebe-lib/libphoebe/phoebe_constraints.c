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
