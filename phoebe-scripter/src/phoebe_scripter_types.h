#ifndef PHOEBE_SCRIPTER_TYPES_H
	#define PHOEBE_SCRIPTER_TYPES_H 1

#include <stdarg.h>

#include "phoebe_scripter_ast.h"

typedef enum PHOEBE_lexer_error {
	LEXER_UNTERMINATED_LITERAL
} PHOEBE_lexer_error;

enum {
	HASH_MULTIPLIER    = 31,
	NO_OF_HASH_BUCKETS = 103
};

typedef struct scripter_symbol {
	char                   *name;
	scripter_ast           *link;
	struct scripter_symbol *next;
} scripter_symbol;

typedef struct scripter_symbol_table {
	char *name;
	scripter_symbol *symbol[NO_OF_HASH_BUCKETS];
	struct scripter_symbol_table *next;
} scripter_symbol_table;

typedef struct PHOEBE_scripter_command {
	char                *name;        /* Command name                */
	scripter_ast_value (*func) ();    /* Command function            */
} PHOEBE_scripter_command;

typedef struct PHOEBE_scripter_command_table {
	int no;
	PHOEBE_scripter_command **command;
} PHOEBE_scripter_command_table;

extern PHOEBE_scripter_command_table *scripter_commands;

/******************************************************************************/

extern scripter_symbol_table *symbol_table;

/******************************************************************************/

scripter_symbol_table *symbol_table_add      (scripter_symbol_table *parent, char *name);
scripter_symbol_table *symbol_table_lookup   (scripter_symbol_table *table,  char *name);
scripter_symbol_table *symbol_table_remove   (scripter_symbol_table *table,  char *name);
int                    symbol_table_print    (scripter_symbol_table *table);
int                    symbol_table_free     (scripter_symbol_table *table);
int                    symbol_table_free_all (scripter_symbol_table *table);

/******************************************************************************/

unsigned int     scripter_symbol_hash      (const char *id);
scripter_symbol *scripter_symbol_commit    (scripter_symbol_table *table, char *id, scripter_ast *value);
scripter_symbol *scripter_symbol_lookup    (scripter_symbol_table *table, char *id);
int              scripter_symbol_remove    (scripter_symbol_table *table, char *id);
int              scripter_symbol_free      (scripter_symbol *s);
int              scripter_symbol_free_list (scripter_symbol *s);

/******************************************************************************/

void scripter_ast_value_free       (scripter_ast_value val);
int  scripter_ast_value_array_free (scripter_ast_value *vals, int dim);

/******************************************************************************/

int scripter_command_args_evaluate (scripter_ast_list *args, scripter_ast_value **vals, int Nmin, int Nmax, ...);

/******************************************************************************/

PHOEBE_scripter_command *scripter_command_new  ();
int                      scripter_command_free (PHOEBE_scripter_command *command);
int                      scripter_command_register (char *name, scripter_ast_value (*func) ());
int                      scripter_command_get_index (char *name, int *index);

int scripter_commands_free_all (PHOEBE_scripter_command_table *table);

/******************************************************************************/

typedef struct PHOEBE_scripter_function_table {
	int no;
	char **func;
} PHOEBE_scripter_function_table;

extern PHOEBE_scripter_function_table *scripter_functions;

int   scripter_function_register     (char *func);
int   scripter_function_register_all ();
bool  scripter_function_defined      (char *func);
int   scripter_function_free_all     (PHOEBE_scripter_function_table *table);

#endif
