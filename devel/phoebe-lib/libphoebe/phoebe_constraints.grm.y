/*
 * This is the parser for PHOEBE constraints. The constraints need to be user-
 * defineable, so the library needs to parse the passed string to obtain the
 * actual function.
 */

%{
#include <stdio.h>

#include "phoebe_constraints.h"
/*#include "phoebe_constraints.lng.h"*/
#include "phoebe_parameters.h"

extern int yyerror (const char *str);
extern int intern_constraint_add_to_table (PHOEBE_ast *ast);
%}

/*
 * You might be tempted to add a (PHOEBE_parameter *) field to the union
 * below instead of using qualifiers; don't. The reason is that the pointers
 * to parameters that would be set for the constraints would point to the
 * parameter in the parameter table that was active at the time of constraint
 * generation, which would be wrong: we need the pointer to the currently
 * active table.
 */

%union {
	int    idx;
	double val;
	char  *str;
	struct PHOEBE_ast *ast;
	struct PHOEBE_ast_list *args;
}

%token <idx> INDEX
%type  <ast> index

%token <val> NUMVAL
%type  <ast> numval

%token <str> BUILTIN
%type  <ast> builtin

%token <str> PARAMETER
%type  <ast> parameter

%type  <ast> constraint
%type  <ast> expr

%left  '+' '-'
%left  '*' '/'
%right '^'

%%

input:			  /* empty */
				| input constraint {
					intern_constraint_add_to_table ($2);
					printf ("________________________________________________________________________________\n");
					phoebe_ast_print (0, $2);
					printf ("--------------------------------------------------------------------------------\n");
				}
				;

constraint:		  PARAMETER '=' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_CONSTRAINT, phoebe_ast_construct_list (phoebe_ast_add_parameter ($1), phoebe_ast_construct_list ($3, NULL)));
				}
				| PARAMETER index '=' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_CONSTRAINT, phoebe_ast_construct_list (phoebe_ast_add_parameter ($1), phoebe_ast_construct_list ($2, phoebe_ast_construct_list ($4, NULL))));
				}
				;

numval:			NUMVAL {
					$$ = phoebe_ast_add_numval ($1);
				}
				;

index:			INDEX {
					$$ = phoebe_ast_add_index ($1);
				}
				;

builtin: 		BUILTIN {
					$$ = phoebe_ast_add_builtin ($1);
					free ($1);
				}
				;

parameter:		PARAMETER {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_PARAMETER, phoebe_ast_construct_list (phoebe_ast_add_parameter ($1), NULL));
				}
				| PARAMETER index {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_PARAMETER, phoebe_ast_construct_list (phoebe_ast_add_parameter ($1), phoebe_ast_construct_list ($2, NULL)));
				}
				;

expr:			  numval    { $$ = $1; }
				| parameter { $$ = $1; }
				| expr '+' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_ADD, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				| expr '-' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_SUB, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				| expr '*' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_MUL, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				| expr '/' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_DIV, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				| expr '^' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_POT, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				| builtin '(' expr ')' {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_BUILTIN, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				;
		
%%

int pcerror (const char *str)
{
	fprintf (stderr, "error: %s\n", str);
	return 1;
}

int pcwrap ()
{
	return 1;
}
