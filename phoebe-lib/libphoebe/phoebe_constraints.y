/*
 * This is the parser for PHOEBE constraints. The constraints need to be user-
 * defineable, so the library needs to parse the passed string to obtain the
 * actual function.
 */

%{
#include <stdio.h>

#include "phoebe_constraints.h"
#include "phoebe_constraints.lex.h"
#include "phoebe_parameters.h"

extern int yyerror (const char *str);
%}

%union {
	double val;
	char  *str;
	struct PHOEBE_parameter *par;
	struct PHOEBE_ast *ast;
	struct PHOEBE_ast_list *args;
}

%token <par> PARAMETER
%type  <ast> parameter

%token <str> BUILTIN
%type  <ast> builtin

%token <val> NUMVAL
%type  <ast> numval

%type  <ast> constraint
%type  <ast> expr

%left  '+' '-'
%left  '*' '/'
%right '^'

%%

input:			  /* empty */
				| input constraint {
					phoebe_constraint_add_to_table ($2);
					printf ("________________________________________________________________________________\n");
					phoebe_ast_print (0, $2);
					printf ("--------------------------------------------------------------------------------\n");
				}
				;

constraint:		parameter '=' expr {
					$$ = phoebe_ast_add_node (PHOEBE_NODE_TYPE_CONSTRAINT, phoebe_ast_construct_list ($1, phoebe_ast_construct_list ($3, NULL)));
				}
				;

numval:			NUMVAL {
					$$ = phoebe_ast_add_numval ($1);
				}
				;

builtin: 		BUILTIN {
					$$ = phoebe_ast_add_builtin ($1);
					free ($1);
				}
				;

parameter:		PARAMETER {
					$$ = phoebe_ast_add_parameter ($1);
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

int yyerror (const char *str)
{
	fprintf (stderr, "error: %s\n", str);
	return;
}

int yywrap ()
{
	return 1;
}
