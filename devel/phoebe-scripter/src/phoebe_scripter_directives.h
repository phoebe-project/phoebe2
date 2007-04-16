#ifndef PHOEBE_SCRIPTER_DIRECTIVES_H
	#define PHOEBE_SCRIPTER_DIRECTIVES_H 1

#include "phoebe_scripter_ast.h"

int scripter_directive_calc    (scripter_ast_list *args);
int scripter_directive_clear   (scripter_ast_list *args);
int scripter_directive_execute (scripter_ast_list *args);
int scripter_directive_help    (scripter_ast_list *args);
int scripter_directive_if      (scripter_ast_list *args);
int scripter_directive_info    (scripter_ast_list *args);
int scripter_directive_list    (scripter_ast_list *args);
int scripter_directive_print   (scripter_ast_list *args);
int scripter_directive_quit    (scripter_ast_list *args);
int scripter_directive_show    (scripter_ast_list *args);
int scripter_directive_stdump  (scripter_ast_list *args);

#endif
