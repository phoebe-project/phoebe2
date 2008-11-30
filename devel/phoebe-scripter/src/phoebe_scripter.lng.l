/******************************************************************************/
/*                                                                            */
/*                         PHOEBE scripter Lexer                              */
/*                                                                            */
/******************************************************************************/

%{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter.grm.h"
#include "phoebe_scripter_types.h"
%}

%option warn
%option outfile="lex.yy.c"
%option header-file="phoebe_scripter.lng.h"

%x literal

LETTER     [a-zA-Z]
DIGIT      [0-9]
UNDERSCORE "\_"
WSPACE     [ \t\n]
INT	       {DIGIT}+
BOOL	   TRUE|FALSE|YES|NO
REAL       {DIGIT}+"."{DIGIT}*
NUMBER     ({INT})|({REAL})
EXPONENT   ({NUMBER})[eE]([+-])?{INT}
NUMERIC    ({REAL}|{EXPONENT})
SYSTEMCALL ^![^\n]{2,}
WORD       {LETTER}+({LETTER}|{DIGIT}|{UNDERSCORE})*
FILENAME   ({LETTER}|{DIGIT}|{UNDERSCORE}|"."|"/"|"-"|"\"")+

%%

	/*
	 * Define a string buffer for storing literals. The length of a literal
	 * should not exceed 255 characters.
	 */

	char buffer[255];
	char *bufptr;

 /****************************  DIRECTIVES:  *********************************/

calc			return CALC;
cd              return CD;
clear			return CLEAR;
define			return DEFINE;
else			return ELSE;
execute			return EXECUTE;
export			return EXPORT;
for             return FOR;
help|"?"        return HELP;
info			return INFO;
if              return IF;
list            return LIST;
macro			return MACRO;
print			return PRINT;
pwd             return PWD;
quit			return QUIT;
return			return RETURN;
set             return SET;
show			return SHOW;
stdump			return STDUMP;
unset|undefine  return UNSET;
while			return WHILE;

testme          return TESTME;

 /************************  FILE SYSTEM INTERFACE:  **************************/

"ls"{WSPACE}*		{
			yylval.str = strdup ("");
			return LS;
			}
"ls"{WSPACE}+{FILENAME}		{
			if (strchr (yytext, '"') != NULL)
			  {
			  char *tmp = strdup (yytext);
			  char *idx = tmp; idx++; idx++;
			  while (idx[0] == ' ' || idx[0] == '\t') idx++;
			  yylval.str = strdup (idx);
			  }
			else
			  {
			  char *tmp = strdup (yytext);
			  char *idx = tmp; idx++; idx++;
			  while (idx[0] == ' ' || idx[0] == '\t') idx++;
			  yylval.str = phoebe_malloc (strlen(idx)+2);
			  sprintf (yylval.str, "\"%s\"", idx);
			  }
			return LS;
			}
"cd"{WSPACE}*		{
			yylval.str = strdup ("");
			return CD;
			}
"cd"{WSPACE}+{FILENAME}		{
			if (strchr (yytext, '"') != NULL)
			  {
			  /* Get rid of quotes: */
			  char *tmp = strdup (yytext);
			  char *idx = strchr (tmp, '"'); idx++;
			  yylval.str = strdup (idx); yylval.str[strlen(yylval.str)-1] = '\0';
			  }
			else
			  {
			  char *tmp = strdup (yytext);
			  char *idx = tmp; idx++; idx++;
			  while (idx[0] == ' ' || idx[0] == '\t') idx++;
			  yylval.str = strdup (idx);
			  }
			return CD;
			}

 /************************ COMPARISON OPERATORS: *****************************/

"=="			return EQ;
"!="			return NEQ;
"<"				return LE;
"<="			return LEQ;
">"				return GR;
">="			return GEQ;
"&&"			return AND;
"||"			return OR;

"++"			return INC;
"--"			return DEC;
"+="			return INCBY;
"-="			return DECBY;
"*="			return MULTBY;
"/="			return DIVBY;

"->"			return WRITETO;
"->>"			return APPENDTO;

 /********************* GENERAL REGULAR EXPRESSIONS: *************************/

\"			{
			bufptr = buffer;
			BEGIN(literal);
			}
<literal>\" {
			/* Closing quotes encountered. */
			*bufptr = '\0';
			yylval.str = strdup (buffer);
			BEGIN(INITIAL);
			return LITERAL;
			}
<literal><<EOF>> {
			/* Unterminated literal, handle the exception here. */
			yylval.idx = LEXER_UNTERMINATED_LITERAL;
			BEGIN(INITIAL);
			return LEXERROR;
			}

<literal>\\a       *bufptr++ = '\a';      /* alert (bell)         */
<literal>\\b       *bufptr++ = '\b';      /* backspace            */
<literal>\\f       *bufptr++ = '\f';      /* form feed            */
<literal>\\n       *bufptr++ = '\n';      /* new line             */
<literal>\\r       *bufptr++ = '\r';      /* carriage return      */
<literal>\\t       *bufptr++ = '\t';      /* horizontal tab       */
<literal>\\v       *bufptr++ = '\v';      /* vertical tab         */
<literal>\\.       *bufptr++ = yytext[1]; /* copy everything else */

<literal>[^\\\n\"]+ {
			char *yptr = yytext;
	        while (*yptr)
                *bufptr++ = *yptr++;
			}

"#"[^\n]*   {
            /* Drop comments */
			}

{INT}		{
			yylval.idx = atoi (yytext);
			return INTEGER;
			}
{NUMERIC}	{
			yylval.val = atof (yytext);
			return VALUE;
			}
{BOOL}		{
			if (strcmp (yytext, "TRUE") == 0 || strcmp (yytext, "YES") == 0) yylval.swc = YES;
			else yylval.swc = NO;
			return BOOLEAN;
			}
{WORD}		{
			int dummy;
			yylval.str = strdup (yytext);
			if (scripter_function_defined (yylval.str)) return FUNCTION;
			if (scripter_command_get_index (yylval.str, &dummy) == SUCCESS) return COMMAND;
			if (phoebe_is_qualifier (yylval.str)) return QUALIFIER;
			return IDENT;
			}
{SYSTEMCALL}	{
			yytext++;
			yylval.str = strdup (yytext);
			return SYSTEM_CALL;
			}

 /****************** OTHER USEFUL REGULAR EXPRESSIONS: ***********************/

"{"			return START_BLOCK;
"}"			return END_BLOCK;
{WSPACE}+	/* Eat whitespaces */
.			return yytext[0];

%%
