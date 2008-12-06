#ifndef PHOEBE_SCRIPTER_IO_H
	#define PHOEBE_SCRIPTER_IO_H 1

#include <stdio.h>
#include "phoebe_scripter_ast.h"

extern FILE *PHOEBE_output;

int phoebe_vector_print             (PHOEBE_vector *vec);
int phoebe_array_print              (PHOEBE_array *array);
int phoebe_curve_print              (PHOEBE_curve *curve);
int phoebe_spectrum_print           (PHOEBE_spectrum *spectrum);
int phoebe_minimizer_feedback_print (PHOEBE_minimizer_feedback *feedback);

int propagate_int_to_double    (scripter_ast_value *val);
int propagate_int_to_bool      (scripter_ast_value *val);
int propagate_int_to_menu_item (scripter_ast_value *val, char *qualifier);
int propagate_int_to_vector    (scripter_ast_value *val, int dim);

#endif
