/******************************************************************************/
/*                                                                            */
/*                         PHOEBE scripter Parser                             */
/*                                                                            */
/******************************************************************************/

%{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "phoebe_scripter_build_config.h"

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter_io.h"

int yydebug=1;
int yyerror (char const *str);

%}

%union {
  bool                      swc;	/* Switch                                 */
  int                       idx;	/* Index                                  */
  double                    val;	/* Double-precision value                 */
  char                     *str;	/* String (literal or a variable)         */
  PHOEBE_vector             vec;    /* Vector data struct                     */
  struct scripter_ast      *ast;	/* Abstract Syntax Tree leaf              */
  struct scripter_ast_list *lst;	/* AST arguments list                     */
}

%locations
%error-verbose

%token <idx> LEXERROR
%type  <ast> lexerror

%token <idx> INTEGER
%type  <ast> integer

%token <swc> BOOLEAN
%type  <ast> boolean

%token <val> VALUE
%type  <ast> value

%token <str> IDENT
%type  <ast> ident

%token <str> QUALIFIER
%type  <ast> qualifier

%token <str> COMMAND

%token <str> LITERAL
%type  <ast> literal

%token <str> FUNCTION
%type  <ast> function

%token <str> SYSTEM_CALL
%type  <ast> system_call

%type  <ast> os_call
%token <str> LS CD

%type  <ast> array

%type  <ast> arg  expr idexpr
%type  <lst> args exprs

%token TESTME

%token HELP CALC SET UNSET EXPORT DEFINE IF ELSE PRINT QUIT WHILE FOR SHOW
%token EXECUTE MACRO PROMPT PWD INFO STDUMP RETURN CLEAR LIST
%token START_BLOCK END_BLOCK

%token WRITETO APPENDTO

%right AND OR
%left  EQ NEQ LEQ LE GEQ GR
%right INC DEC INCBY DECBY MULTBY DIVBY
%left  '!'

%type  <ast> statement block help set_line if_line
%type  <ast> define_line command_line while_line
%type  <ast> for_line macro_line list_line
%type  <ast> export_line return_line
%type  <lst> statements

%type <idx> directive

%right '='
%left  '-' '+'
%left  '*' '/' '%'
%left  NEG
%right '^'

%%

input
	: /* empty */
	| input statement {
		#ifdef PHOEBE_DEBUG_SUPPORT
			fprintf (PHOEBE_output, "________________________________________________________________________________\n");
			scripter_ast_print (0, $2);
			fprintf (PHOEBE_output, "--------------------------------------------------------------------------------\n");
		#endif
		scripter_ast_evaluate ($2);
		scripter_ast_free ($2);
	}
	;

/*************************  BASIC GRAMMAR TYPES  ******************************/

lexerror
	: LEXERROR {
		$$ = scripter_ast_add_int ($1);
	}
	;

integer
	: INTEGER {
		$$ = scripter_ast_add_int ($1);
	}
	;
value
	: VALUE {
		$$ = scripter_ast_add_double ($1);
	}
	;
boolean
	: BOOLEAN {
		$$ = scripter_ast_add_bool ($1);
	}
	;
literal
	: LITERAL {
		$$ = scripter_ast_add_string ($1);
		free ($1);
	}
	;
ident
	: IDENT {
		$$ = scripter_ast_add_variable ($1);
		free ($1);
	}
	;
qualifier
	: QUALIFIER {
		$$ = scripter_ast_add_qualifier ($1);
		free ($1);
	}
	;
array
	: START_BLOCK exprs END_BLOCK {
		scripter_ast_list *s = scripter_ast_reverse_list ($2, NULL);
		$$ = scripter_ast_add_node (kind_array, s);
	}
	;

/******************************************************************************/

statement
	: lexerror      { $$ = scripter_ast_add_node (kind_lexerror,  scripter_ast_construct_list ($1, NULL)); }
	| block         { $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| directive exprs {
		scripter_ast_list *args = scripter_ast_reverse_list ($2, NULL);
		$$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list (scripter_ast_add_node ($1, args), NULL));
	}
	| help          { $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| set_line		{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| list_line     { $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| export_line	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| return_line	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| define_line	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| if_line		{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| command_line	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| while_line	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| for_line		{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| macro_line	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| os_call		{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| system_call	{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, NULL)); }
	| statement WRITETO expr
					{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, scripter_ast_construct_list (scripter_ast_add_string ("w"), NULL)))); }
	| statement APPENDTO expr
					{ $$ = scripter_ast_add_node (kind_statement, scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, scripter_ast_construct_list (scripter_ast_add_string ("a"), NULL)))); }
	;
statements
	: statements statement {
		$$ = scripter_ast_construct_list ($2, $1);
	}
	| statement {
		$$ = scripter_ast_construct_list ($1, NULL);
	}
	;
block
	: START_BLOCK /* empty */ END_BLOCK {
		$$ = scripter_ast_add_node (kind_block, NULL);
 	}
	| START_BLOCK statements END_BLOCK {
		scripter_ast_list *s = scripter_ast_reverse_list ($2, NULL);
		$$ = scripter_ast_add_node (kind_block, s);
	}
	;
directive
	: CALC    { $$ = kind_calc;    }
	| CLEAR   { $$ = kind_clear;   }
	| EXECUTE { $$ = kind_execute; }
	| INFO	  { $$ = kind_info;    }
	| PRINT   { $$ = kind_print;   }
	| QUIT    { $$ = kind_quit;    }
	| SHOW    { $$ = kind_show;    }
	| STDUMP  { $$ = kind_stdump;  }
	| TESTME  { $$ = kind_testme;  }
	;
help: HELP           { $$ = scripter_ast_add_node (kind_help, NULL); }
	| HELP directive { $$ = scripter_ast_add_node (kind_help, scripter_ast_construct_list (scripter_ast_add_int ($2), NULL)); }
	| HELP COMMAND   {
		$$ = scripter_ast_add_node (kind_help, scripter_ast_construct_list (scripter_ast_add_string ($2), NULL));
		free ($2);
	}
	| HELP qualifier { $$ = scripter_ast_add_node (kind_help, scripter_ast_construct_list ($2, NULL)); }
	| HELP IF        { $$ = scripter_ast_add_node (kind_help, scripter_ast_construct_list (scripter_ast_add_int (kind_if), NULL)); }
	| HELP WHILE     { $$ = scripter_ast_add_node (kind_help, scripter_ast_construct_list (scripter_ast_add_int (kind_while), NULL)); }
	;
system_call
	: SYSTEM_CALL {
		$$ = scripter_ast_add_node (kind_system_call, scripter_ast_construct_list (scripter_ast_add_string ($1), NULL));
		free ($1);
	}
	;
os_call
	: LS {
		$$ = scripter_ast_add_node (kind_os_ls, scripter_ast_construct_list (scripter_ast_add_string ($1), NULL));
		free ($1);
	}
	| CD {
		$$ = scripter_ast_add_node (kind_os_cd, scripter_ast_construct_list (scripter_ast_add_string ($1), NULL));
		free ($1);
	}
	| PWD {
		$$ = scripter_ast_add_node (kind_os_pwd, NULL);
	}
	;
while_line
	: WHILE '(' expr ')' statement {
		$$ = scripter_ast_add_node (kind_while, scripter_ast_construct_list ($3, scripter_ast_construct_list ($5, NULL)));
	}
	;
for_line
	: FOR '(' ident '=' expr ';' expr ';' idexpr ')' statement {
		$$ = scripter_ast_add_node (kind_for, scripter_ast_construct_list ($3, scripter_ast_construct_list ($5, scripter_ast_construct_list ($7, scripter_ast_construct_list ($9, scripter_ast_construct_list ($11, NULL))))));
	}
	;
if_line
	: IF '(' expr ')' statement {
		$$ = scripter_ast_add_node (kind_if, scripter_ast_construct_list ($3, scripter_ast_construct_list ($5, NULL)));
	}
	| IF '(' expr ')' statement ELSE statement {
		$$ = scripter_ast_add_node (kind_if, scripter_ast_construct_list ($3, scripter_ast_construct_list ($5, scripter_ast_construct_list ($7, NULL))));
	}
	;
command_line
	: COMMAND '(' exprs ')' {
		scripter_ast_list *params = scripter_ast_reverse_list ($3, NULL);
		$$ = scripter_ast_add_node (kind_command, scripter_ast_construct_list (scripter_ast_add_string ($1), params));
		free ($1);
	}
	| ident '(' ')' {
		$$ = scripter_ast_add_node (kind_execute_macro, scripter_ast_construct_list ($1, NULL));
	}
	| ident '(' exprs ')' {
		scripter_ast_list *s = scripter_ast_reverse_list ($3, NULL);
		$$ = scripter_ast_add_node (kind_execute_macro, scripter_ast_construct_list ($1, s));
	}
	;
set_line
	: SET ident '=' expr {
		$$ = scripter_ast_add_node (kind_set, scripter_ast_construct_list ($2, scripter_ast_construct_list ($4, NULL)));
	}
	| SET idexpr {
		$$ = scripter_ast_add_node (kind_set, scripter_ast_construct_list ($2, NULL));
	}
	| SET ident '[' expr ']' '=' expr {
		$$ = scripter_ast_add_node (kind_set_element, scripter_ast_construct_list ($2, scripter_ast_construct_list ($4, scripter_ast_construct_list ($7, NULL))));
	}
	| SET ident '[' expr ']' '[' expr ']' '=' expr {
		$$ = scripter_ast_add_node (kind_set_matrix_element, scripter_ast_construct_list ($2, scripter_ast_construct_list ($4, scripter_ast_construct_list ($7, scripter_ast_construct_list ($10, NULL)))));
	}
	| UNSET ident {
		$$ = scripter_ast_add_node (kind_unset, scripter_ast_construct_list ($2, NULL));
	}
	;
list_line
	: LIST ident {
		$$ = scripter_ast_add_node (kind_list, scripter_ast_construct_list ($2, NULL));
	}
	;
export_line
	: EXPORT ident {
		$$ = scripter_ast_add_node (kind_export, scripter_ast_construct_list ($2, NULL));
	}
	;
return_line
	: RETURN expr {
		$$ = scripter_ast_add_node (kind_return, scripter_ast_construct_list ($2, NULL));
	}
	;
define_line
	: DEFINE ident '(' args ')' block {
		scripter_ast_list *params = scripter_ast_reverse_list ($4, NULL);
		$$ = scripter_ast_add_node (kind_define, scripter_ast_construct_list ($2, scripter_ast_construct_list ($6, params)));
	}
	| DEFINE ident '(' ')' block {
		$$ = scripter_ast_add_node (kind_define, scripter_ast_construct_list ($2, scripter_ast_construct_list ($5, NULL)));
	}
	;
macro_line
	: MACRO ident '(' args ')' block {
		scripter_ast_list *params = scripter_ast_reverse_list ($4, NULL);
		$$ = scripter_ast_add_node (kind_macro, scripter_ast_construct_list ($2, scripter_ast_construct_list ($6, params)));
	}
	| MACRO ident '(' ')' block
			{ $$ = scripter_ast_add_node (kind_macro, scripter_ast_construct_list ($2, scripter_ast_construct_list ($5, NULL))); }
	;
arg
	: ident {
	}
	;
args
	: args ',' arg {
		$$ = scripter_ast_construct_list ($3, $1);
	}
	| arg {
		$$ = scripter_ast_construct_list ($1, NULL);
	}
	;
function
	: FUNCTION {
		$$ = scripter_ast_add_function ($1);
		free ($1);
	}
	;
expr
	: integer				{ $$ = $1; }
	| value					{ $$ = $1; }
	| boolean				{ $$ = $1; }
	| array					{ $$ = $1; }
	| ident					{ $$ = $1; }
	| qualifier				{ $$ = $1; }
	| literal				{ $$ = $1; }
	| '(' expr ')'			{ $$ = $2; }
	| expr '[' expr ']'     { $$ = scripter_ast_add_node (kind_element, scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '.' ident        { $$ = scripter_ast_add_node (kind_field,   scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '+' expr			{ $$ = scripter_ast_add_node (kind_add,     scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '-' expr			{ $$ = scripter_ast_add_node (kind_sub,     scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '*' expr			{ $$ = scripter_ast_add_node (kind_mul,     scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '/' expr			{ $$ = scripter_ast_add_node (kind_div,     scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '%' expr			{ $$ = scripter_ast_add_node (kind_idiv,    scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr '^' expr			{ $$ = scripter_ast_add_node (kind_raise,   scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| '+' expr %prec NEG	{ $$ = scripter_ast_add_node (kind_unaryp,  scripter_ast_construct_list ($2, NULL)); }
	| '-' expr %prec NEG	{ $$ = scripter_ast_add_node (kind_unarym,  scripter_ast_construct_list ($2, NULL)); }
	| expr EQ  expr			{ $$ = scripter_ast_add_node (kind_equal,   scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr NEQ expr			{ $$ = scripter_ast_add_node (kind_nequal,  scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr LEQ expr			{ $$ = scripter_ast_add_node (kind_lequal,  scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr GEQ expr			{ $$ = scripter_ast_add_node (kind_gequal,  scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr LE  expr			{ $$ = scripter_ast_add_node (kind_less,    scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr GR  expr			{ $$ = scripter_ast_add_node (kind_greater, scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr AND expr			{ $$ = scripter_ast_add_node (kind_and,     scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| expr OR  expr			{ $$ = scripter_ast_add_node (kind_or,      scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| '!' expr				{ $$ = scripter_ast_add_node (kind_not,     scripter_ast_construct_list ($2, NULL)); }
	| function '(' expr ')'	{
		$$ = scripter_ast_add_node (kind_builtin, scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL)));
	}
	| ident '(' ')'	{
		$$ = scripter_ast_add_node (kind_func, scripter_ast_construct_list ($1, NULL));
	}
	| ident '(' exprs ')'	{
		scripter_ast_list *s = scripter_ast_reverse_list ($3, NULL);
		$$ = scripter_ast_add_node (kind_func, scripter_ast_construct_list ($1, s));
	}
	| COMMAND '(' exprs ')' {
		scripter_ast_list *params = scripter_ast_reverse_list ($3, NULL);
		$$ = scripter_ast_add_node (kind_command, scripter_ast_construct_list (scripter_ast_add_string ($1), params));
		free ($1);
	}
	;
exprs
	: /* empty */ {
		$$ = NULL;
	}
	| exprs ',' expr {
		$$ = scripter_ast_construct_list ($3, $1);
	}
	| expr {
		$$ = scripter_ast_construct_list ($1, NULL);
	}
	;
idexpr
	: ident INC			{ $$ = scripter_ast_add_node (kind_inc,     scripter_ast_construct_list ($1, NULL)); }
	| ident DEC			{ $$ = scripter_ast_add_node (kind_dec,     scripter_ast_construct_list ($1, NULL)); }
	| ident INCBY  expr	{ $$ = scripter_ast_add_node (kind_incby,   scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| ident DECBY  expr	{ $$ = scripter_ast_add_node (kind_decby,   scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| ident MULTBY expr	{ $$ = scripter_ast_add_node (kind_multby,  scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	| ident DIVBY  expr	{ $$ = scripter_ast_add_node (kind_divby,   scripter_ast_construct_list ($1, scripter_ast_construct_list ($3, NULL))); }
	;

%%

int yyerror (const char *str)
{
    phoebe_scripter_output ("%s\n", str);
	return SUCCESS;
}

int yywrap ()
{
  return 1;
}
