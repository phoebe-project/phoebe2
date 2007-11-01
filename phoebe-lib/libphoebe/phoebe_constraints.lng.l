/*
 * This is the lexer for PHOEBE constraints. The constraints need to be user-
 * defineable, so the library needs to parse the passed string to obtain the
 * actual function.
 */

%{
#include <stdlib.h>

#include "phoebe_constraints.grm.h"
#include "phoebe_error_handling.h"
#include "phoebe_parameters.h"
%}

%option warn
%option prefix="pc"
%option outfile="lex.yy.c"
%option header-file="phoebe_constraints.lng.h"

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
			pclval.val = atof (yytext);
			return NUMVAL;
			}
{_BUILTIN}	{
			pclval.str = strdup (yytext);
			return BUILTIN;
			}
{_QUAL}		{
			/* Check if the parameter exists; if not, warn about it. */
			PHOEBE_parameter *par = phoebe_parameter_lookup (yytext);
			if (par) {
				pclval.str = strdup (yytext);
				return PARAMETER;
			}
			else {
				phoebe_lib_error ("parameter %s not found, ignoring.\n", yytext);
			}
			}
"["{_INT}"]"	{
			sscanf (yytext, "[%d]", &pclval.idx);
			return INDEX;
			}
{_WSPACE}+	/* Eat whitespaces */
.			return yytext[0];

%%
