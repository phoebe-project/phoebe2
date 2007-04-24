#ifndef PHOEBE_SCRIPTER_ARITHMETICS_H
	#define PHOEBE_SCRIPTER_ARITHMETICS_H

#include "phoebe_scripter_ast.h"

int scripter_ast_values_add      (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2);
int scripter_ast_values_subtract (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2);
int scripter_ast_values_multiply (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2);
int scripter_ast_values_divide   (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2);
int scripter_ast_values_raise    (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2);

#endif

