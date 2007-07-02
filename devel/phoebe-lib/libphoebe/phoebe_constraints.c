#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_constraints.h"
#include "phoebe_error_handling.h"
#include "phoebe_parameters.h"

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
