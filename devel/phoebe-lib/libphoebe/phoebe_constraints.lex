/*
 * This is the lexer for PHOEBE constraints. The constraints need to be user-
 * defineable, so the library needs to parse the passed string to obtain the
 * actual function.
 */

%{
#include <stdlib.h>

#include "phoebe_constraints.tab.h"
#include "phoebe_parameters.h"
%}

%option warn
%option outfile="phoebe_constraints.lex.c"
%option header-file="phoebe_constraints.lex.h"

_DIGIT	[0-9]
_LETTER	[a-zA-Z]
_UNDER	"\_"

_INT	{_DIGIT}+
_REAL	{_DIGIT}+"."{_DIGIT}*
_EXP	({_INT}|{_REAL})[eE]([+-])?({_INT})
_NUM	({_INT}|{_REAL}|{_EXP})
_QUAL	{_LETTER}+({_LETTER}|{_DIGIT}|{_UNDER})*

_WSPACE	[ \t\n]

_BUILTIN	sin|cos|tan|asin|acos|atan|exp|ln|log|sqrt

%%

{_NUM}		{
			yylval.val = atof (yytext);
			return NUMVAL;
			}
{_BUILTIN}	{
			yylval.str = strdup (yytext);
			return BUILTIN;
			}
{_QUAL}		{
			PHOEBE_parameter *par = phoebe_parameter_lookup (yytext);
			if (par) {
				yylval.par = par;
				return PARAMETER;
			}
			else {
				printf ("Parameter %s not found, aborting.\n", yytext);
			}
			}
"["{_INT}"]"	{
			sscanf (yytext, "[%d]", &yylval.idx);
			return INDEX;
			}
{_WSPACE}+	/* Eat whitespaces */
.			return yytext[0];

%%
