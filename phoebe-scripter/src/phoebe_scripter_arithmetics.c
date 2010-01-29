#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter_io.h"

int scripter_ast_values_add (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function adds the two AST values. It checks all combinations of
	 * value types, propagates if necessary and performs the sumation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int i;
	int status;
	char str[255];

	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->type = type_int;
					out->value.i = val1.value.i + val2.value.i;
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_double;
					propagate_int_to_double (&val1);
					out->value.d = val1.value.d + val2.value.d;
				break;
				case type_string:
					out->type = type_string;
					sprintf (str, "%d", val1.value.i);
					out->value.str = phoebe_concatenate_strings (str, val2.value.str, NULL);
				break;
				case type_vector:
					out->type = type_vector;
					propagate_int_to_double (&val1);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d + val2.value.vec->val[i];
				break;
				case type_array:
					switch (val2.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_INT_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.array->val.iarray[i] = val1.value.i + val2.value.array->val.iarray[i];
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_DOUBLE_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.array->val.darray[i] = val1.value.i + val2.value.array->val.darray[i];
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							sprintf (str, "%d", val1.value.i);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (str, val2.value.array->val.strarray[i], NULL);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.i + val2.value.spectrum->data->val[i];
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_minfeedback:
					phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");
				break;
			}
		break;
		case type_bool:
			phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
			return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->type = type_double;
					propagate_int_to_double (&val2);
					out->value.d = val1.value.d + val2.value.d;
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_double;
					out->value.d = val1.value.d + val2.value.d;
				break;
				case type_string:
					out->type = type_string;
					sprintf (str, "%lf", val1.value.d);
					out->value.str = phoebe_concatenate_strings (str, val2.value.str, NULL);
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d + val2.value.vec->val[i];
				break;
				case type_array:
					switch (val2.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.vec->val[i] = val1.value.d + val2.value.array->val.iarray[i];
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.vec->val[i] = val1.value.d + val2.value.array->val.darray[i];
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							sprintf (str, "%lf", val1.value.d);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (str, val2.value.array->val.strarray[i], NULL);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.d + val2.value.spectrum->data->val[i];
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_minfeedback:
					phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");
				break;
			}
		break;
		case type_string:
			switch (val2.type) {
				case type_int:
					out->type = type_string;
					sprintf (str, "%d", val2.value.i);
					out->value.str = phoebe_concatenate_strings (val1.value.str, str, NULL);
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_string;
					sprintf (str, "%lf", val2.value.d);
					out->value.str = phoebe_concatenate_strings (val1.value.str, str, NULL);
				break;
				case type_string:
					out->type = type_string;
					out->value.str = phoebe_concatenate_strings (val1.value.str, val2.value.str, NULL);
				break;
				case type_vector:
					phoebe_scripter_output ("operator '+' cannot act between strings and arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					switch (val2.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++) {
								sprintf (str, "%d", val2.value.array->val.iarray[i]);
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.str, str, NULL);
							}
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++) {
								sprintf (str, "%lf", val2.value.array->val.darray[i]);
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.str, str, NULL);
							}
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.str, val2.value.array->val.strarray[i], NULL);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '+' cannot act between strings and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_minfeedback:
					phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int:
					out->type = type_vector;
					propagate_int_to_double (&val2);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] + val2.value.d;
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] + val2.value.d;
				break;
				case type_string:
					phoebe_scripter_output ("operator '+' cannot act between strings and arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					status = phoebe_vector_add (out->value.vec, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_array:
					if (val1.value.vec->dim != val2.value.array->dim) {
						out->type = type_void;
						return ERROR_VECTOR_DIMENSIONS_MISMATCH;
					}
					switch (val2.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.vec->val[i] = val1.value.vec->val[i] + val2.value.array->val.iarray[i];
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++)
								out->value.vec->val[i] = val1.value.vec->val[i] + val2.value.array->val.darray[i];
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val2.value.array->dim);
							for (i = 0; i < val2.value.array->dim; i++) {
								sprintf (str, "%lf", val1.value.vec->val[i]);
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (str, val2.value.array->val.strarray[i], NULL);
							}
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '+' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_minfeedback:
					phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");			
				break;
			}
		break;
		case type_array:
			switch (val2.type) {
				case type_int:
					switch (val1.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_INT_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.array->val.iarray[i] =  val1.value.array->val.iarray[i] + val2.value.i;
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_DOUBLE_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.array->val.darray[i] =  val1.value.array->val.darray[i] + val2.value.i;
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							sprintf (str, "%d", val2.value.i);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.array->val.strarray[i], str, NULL);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					switch (val1.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.vec->val[i] =  val1.value.array->val.iarray[i] + val2.value.d;
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.vec->val[i] =  val1.value.array->val.darray[i] + val2.value.d;
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							sprintf (str, "%lf", val2.value.d);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.array->val.strarray[i], str, NULL);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_string:
					switch (val1.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++) {
								sprintf (str, "%d", val1.value.array->val.iarray[i]);
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (str, val2.value.str, NULL);
							}
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++) {
								sprintf (str, "%lf", val1.value.array->val.darray[i]);
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (str, val2.value.str, NULL);
							}
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.array->val.strarray[i], val2.value.str, NULL);
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_vector:
					if (val2.value.vec->dim != val1.value.array->dim) {
						out->type = type_void;
						return ERROR_VECTOR_DIMENSIONS_MISMATCH;
					}
					switch (val1.value.array->type) {
						case TYPE_INT_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.vec->val[i] = val1.value.array->val.iarray[i] + val2.value.vec->val[i];
						break;
						case TYPE_BOOL_ARRAY:
							phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
							return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
						break;
						case TYPE_DOUBLE_ARRAY:
							out->type = type_vector;
							out->value.vec = phoebe_vector_new ();
							phoebe_vector_alloc (out->value.vec, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++)
								out->value.vec->val[i] = val1.value.array->val.darray[i] + val2.value.vec->val[i];
						break;
						case TYPE_STRING_ARRAY:
							out->type = type_array;
							out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
							phoebe_array_alloc (out->value.array, val1.value.array->dim);
							for (i = 0; i < val1.value.array->dim; i++) {
								sprintf (str, "%lf", val2.value.vec->val[i]);
								out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.array->val.strarray[i], str, NULL);
							}
						break;
						default:
							phoebe_scripter_output ("exception handler invoked in scripter_ast_values_add, please report this!\n");
							out->type = type_void;
					}
				break;
				case type_array:
					if (val1.value.array->type != val2.value.array->type) {
						phoebe_scripter_output ("not yet implemented, sorry.\n");
						out->type = type_void;
					}
					if (val1.value.array->dim != val2.value.array->dim) {
						out->type = type_void;
						return ERROR_VECTOR_DIMENSIONS_MISMATCH;
					}
					else if (val1.value.array->type == TYPE_INT_ARRAY) {
						out->type = type_array;
						out->value.array = phoebe_array_new (TYPE_INT_ARRAY);
						phoebe_array_alloc (out->value.array, val1.value.array->dim);
						for (i = 0; i < val1.value.array->dim; i++)
							out->value.array->val.iarray[i] = val1.value.array->val.iarray[i] + val2.value.array->val.iarray[i];
					}
					else if (val1.value.array->type == TYPE_BOOL_ARRAY) {
						phoebe_scripter_output ("operator '+' cannot act on boolean arrays, aborting.\n");
						return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
					}
					else if (val1.value.array->type == TYPE_DOUBLE_ARRAY) {
						out->type = type_array;
						out->value.array = phoebe_array_new (TYPE_DOUBLE_ARRAY);
						phoebe_array_alloc (out->value.array, val1.value.array->dim);
						for (i = 0; i < val1.value.array->dim; i++)
							out->value.array->val.darray[i] = val1.value.array->val.darray[i] + val2.value.array->val.darray[i];
					}
					else if (val1.value.array->type == TYPE_STRING_ARRAY) {
						out->type = type_array;
						out->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
						phoebe_array_alloc (out->value.array, val1.value.array->dim);
						for (i = 0; i < val1.value.array->dim; i++)
							out->value.array->val.strarray[i] = phoebe_concatenate_strings (val1.value.array->val.strarray[i], val2.value.array->val.strarray[i], NULL);
					}
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '+' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_minfeedback:
					phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");			
				break;
			}
		break;
		case type_spectrum:
			switch (val2.type) {
				case type_int:
					out->type = type_spectrum;
					propagate_int_to_double (&val2);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] + val2.value.d;
					}
					out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] + val2.value.d;
					}
					out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
				break;
				case type_string:
					phoebe_scripter_output ("operator '+' cannot act between strings and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_vector:
					phoebe_scripter_output ("operator '+' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					phoebe_scripter_output ("operator '+' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					status = phoebe_spectra_add (&(out->value.spectrum), val1.value.spectrum, val2.value.spectrum);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_minfeedback:
					phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");			
				break;
			}
		break;
		case type_minfeedback:
			phoebe_scripter_output ("operator '+' cannot act on minimizer feedback, aborting.\n");
			return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
		break;
		case type_void:
			return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
		break;
		default:
			printf ("exception handler invoked in scripter_ast_values_add (), please report this!\n");			
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_subtract (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function subtracts the two AST values. It checks all combinations of
	 * value types, propagates if necessary and performs the subtraction.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int i;
	int status;

	/*
	 * Since the '-' operator acts only on some data types, first we eliminate
	 * all data types that are not supported:
	 */

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '-' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '-' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '-' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->type = type_int;
					out->value.i = val1.value.i - val2.value.i;
				break;
				case type_double:
					out->type = type_double;
					propagate_int_to_double (&val1);
					out->value.d = val1.value.d - val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					propagate_int_to_double (&val1);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d - val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					propagate_int_to_double (&val1);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.d - val2.value.spectrum->data->val[i];
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_subtract (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->type = type_double;
					propagate_int_to_double (&val2);
					out->value.d = val1.value.d - val2.value.d;
				break;
				case type_double:
					out->type = type_double;
					out->value.d = val1.value.d - val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d - val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.d - val2.value.spectrum->data->val[i];
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_subtract (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int:
					out->type = type_vector;
					propagate_int_to_double (&val2);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] - val2.value.d;
				break;
				case type_double:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] - val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					status = phoebe_vector_subtract (out->value.vec, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '-' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_subtract (), please report this!\n");
				break;
			}
		break;
		case type_array:
			switch (val2.type) {
				case type_int:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_double:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_vector:
					phoebe_scripter_output ("operator '-' cannot act between numeric and non-numeric arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '-' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_subtract (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			switch (val2.type) {
				case type_int:
					out->type = type_spectrum;
					propagate_int_to_double (&val2);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] - val2.value.d;
					}
					out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] - val2.value.d;
					}
					out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
				break;
				case type_vector:
					phoebe_scripter_output ("operator '-' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					phoebe_scripter_output ("operator '-' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					status = phoebe_spectra_subtract (&(out->value.spectrum), val1.value.spectrum, val2.value.spectrum);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_subtract (), please report this!\n");
				break;
			}
		break;
		case type_void:
			return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
		break;
		default:
			printf ("exception handler invoked in scripter_ast_values_subtract (), please report this!\n");
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_multiply (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function multiplies the two AST values. It checks all combinations
	 * of value types, propagates if necessary and performs the multiplication.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int i;
	int status;

	/*
	 * Since the '*' operator acts only on some data types, first we eliminate
	 * all data types that are not supported:
	 */

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '*' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '*' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '*' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->type = type_int;
					out->value.i = val1.value.i * val2.value.i;
				break;
				case type_double:
					out->type = type_double;
					propagate_int_to_double (&val1);
					out->value.d = val1.value.d * val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					propagate_int_to_double (&val1);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d * val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					propagate_int_to_double (&val1);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_multiply_by (&out->value.spectrum, val2.value.spectrum, val1.value.d);
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_multiply (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->type = type_double;
					propagate_int_to_double (&val2);
					out->value.d = val1.value.d * val2.value.d;
				break;
				case type_double:
					out->type = type_double;
					out->value.d = val1.value.d * val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d * val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_multiply_by (&out->value.spectrum, val2.value.spectrum, val1.value.d);
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_multiply (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int:
					out->type = type_vector;
					propagate_int_to_double (&val2);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] * val2.value.d;
				break;
				case type_double:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] * val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					status = phoebe_vector_multiply (out->value.vec, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					if (val1.value.vec->dim != val2.value.spectrum->data->bins) {
						phoebe_scripter_output ("operand dimensions mismatch, aborting.\n");
						out->type = type_void;
						return SUCCESS;
					}
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_duplicate (val2.value.spectrum);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] *= val1.value.vec->val[i];
					return SUCCESS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_multiply (), please report this!\n");
				break;
			}
		break;
		case type_array:
			switch (val2.type) {
				case type_int:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_double:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_vector:
					phoebe_scripter_output ("operator '*' cannot act between numeric and non-numeric arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '*' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_multiply (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			switch (val2.type) {
				case type_int:
					out->type = type_spectrum;
					propagate_int_to_double (&val2);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_multiply_by (&out->value.spectrum, val1.value.spectrum, val2.value.d);
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_multiply_by (&out->value.spectrum, val1.value.spectrum, val2.value.d);
				break;
				case type_vector:
					if (val2.value.vec->dim != val1.value.spectrum->data->bins) {
						phoebe_scripter_output ("operand dimensions mismatch, aborting.\n");
						out->type = type_void;
						return SUCCESS;
					}
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_duplicate (val1.value.spectrum);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] *= val2.value.vec->val[i];
					return SUCCESS;
				break;
				case type_array:
					phoebe_scripter_output ("operator '*' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_multiply (), please report this!\n");
				break;
			}
		break;
		case type_void:
			return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
		break;
		default:
			printf ("exception handler invoked in scripter_ast_values_multiply (), please report this!\n");
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_divide (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function divides the two AST values. It checks all combinations
	 * of value types, propagates if necessary and performs the division.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int i;
	int status;

	/*
	 * Since the '/' operator acts only on some data types, first we eliminate
	 * all data types that are not supported:
	 */

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '/' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '/' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '/' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->type = type_double;
					propagate_int_to_double (&val1);
					propagate_int_to_double (&val2);
					out->value.d = val1.value.d / val2.value.d;
				break;
				case type_double:
					out->type = type_double;
					propagate_int_to_double (&val1);
					out->value.d = val1.value.d / val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					propagate_int_to_double (&val1);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d / val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					propagate_int_to_double (&val1);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.d / val2.value.spectrum->data->val[i];
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_divide (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->type = type_double;
					propagate_int_to_double (&val2);
					out->value.d = val1.value.d / val2.value.d;
				break;
				case type_double:
					out->type = type_double;
					out->value.d = val1.value.d / val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d / val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.d / val2.value.spectrum->data->val[i];
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_divide (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int:
					out->type = type_vector;
					propagate_int_to_double (&val2);
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] / val2.value.d;
				break;
				case type_double:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.vec->val[i] / val2.value.d;
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					status = phoebe_vector_divide (out->value.vec, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '/' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_divide (), please report this!\n");
				break;
			}
		break;
		case type_array:
			switch (val2.type) {
				case type_int:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_double:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_vector:
					phoebe_scripter_output ("operator '/' cannot act between numeric and non-numeric arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_spectrum:
					phoebe_scripter_output ("operator '/' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_divide (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			switch (val2.type) {
				case type_int:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] / val2.value.i;
					}
					out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] / val2.value.d;
					}
					out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
				break;
				case type_vector:
					phoebe_scripter_output ("operator '/' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					phoebe_scripter_output ("operator '/' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					status = phoebe_spectra_divide (&(out->value.spectrum), val1.value.spectrum, val2.value.spectrum);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						return status;
					}
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_divide (), please report this!\n");
				break;
			}
		break;
		case type_void:
			return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
		break;
		default:
			printf ("exception handler invoked in scripter_ast_values_divide (), please report this!\n");
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_raise (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function raises the first AST value to the second AST value.
	 * It checks all combinations of value types, propagates if necessary
	 * and performs the raise operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int i, status;

	/*
	 * Since the '/' operator acts only on some data types, first we eliminate
	 * all data types that are not supported:
	 */

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '^' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '^' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_array || val2.type == type_array) {
		phoebe_scripter_output ("operator '^' cannot act on non-numeric arrays, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '^' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->type = type_int;
					out->value.i = (int) pow (val1.value.i, val2.value.i);
				break;
				case type_double:
					out->type = type_double;
					out->value.d = pow (val1.value.i, val2.value.d);
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = pow (val1.value.i, val2.value.vec->val[i]);
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.i, val2.value.spectrum->data->val[i]);
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					out->type = type_void;
					return ERROR_EXCEPTION_HANDLER_INVOKED;
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->type = type_double;
					out->value.d = pow (val1.value.d, val2.value.i);
				break;
				case type_double:
					out->type = type_double;
					out->value.d = pow (val1.value.d, val2.value.d);
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = pow (val1.value.d, val2.value.vec->val[i]);
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.d, val2.value.spectrum->data->val[i]);
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					out->type = type_void;
					return ERROR_EXCEPTION_HANDLER_INVOKED;
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = pow (val1.value.vec->val[i], val2.value.i);
				break;
				case type_double:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = pow (val1.value.vec->val[i], val2.value.d);
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val1.value.vec->dim);
					status = phoebe_vector_raise (out->value.vec, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						out->type = type_void;
						return status;
					}
				break;
				case type_spectrum:
					if (val1.value.vec->dim != val2.value.spectrum->data->bins) {
						out->type = type_void;
						return ERROR_VECTOR_DIMENSIONS_MISMATCH;
					}
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.vec->val[i], val2.value.spectrum->data->val[i]);
					}
					out->value.spectrum->data->range[i] = val2.value.spectrum->data->range[i];
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					out->type = type_void;
					return ERROR_EXCEPTION_HANDLER_INVOKED;
				break;
			}
		break;
		case type_spectrum:
			out->type = type_spectrum;
			out->value.spectrum = phoebe_spectrum_new ();
			phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);

			switch (val2.type) {
				case type_int:
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.i);
					}
				break;
				case type_double:
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.d);
					}
				break;
				case type_vector:
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.vec->val[i]);
					}
				break;
				case type_spectrum:
					for (i = 0; i < out->value.spectrum->data->bins; i++) {
						out->value.spectrum->data->range[i] = val1.value.spectrum->data->range[i];
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.spectrum->data->val[i]);
					}
				break;
				case type_void:
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				default:
					out->type = type_void;
					return ERROR_EXCEPTION_HANDLER_INVOKED;
				break;
			}
		break;
		default:
			out->type = type_void;
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_equal (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function checks whether the first AST value and the second AST
	 * value are equal. It checks all combinations of value types, propagates
	 * if necessary and performs the == operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	if (val1.type != val2.type) {
		out->value.b = FALSE;
		return SUCCESS;
	}

	out->type = type_bool;
	switch (val1.type) {
		case type_int:
			out->value.b = (val1.value.i == val2.value.i);
		break;
		case type_bool:
			out->value.b = (val1.value.b == val2.value.b);
		break;
		case type_double:
			if (fabs (val1.value.d - val2.value.d) > PHOEBE_NUMERICAL_ACCURACY)
				return FALSE;
			else
				return TRUE;
		break;
		case type_string:
			if (strcmp (val1.value.str, val2.value.str) == 0)
				out->value.b = TRUE;
			else
				out->value.b = FALSE;
		break;
		case type_vector:
			out->value.b = phoebe_vector_compare (val1.value.vec, val2.value.vec);
		break;
		case type_array:
			out->value.b = phoebe_array_compare (val1.value.array, val2.value.array);
		break;
		case type_spectrum:
			out->value.b = phoebe_spectra_compare (val1.value.spectrum, val2.value.spectrum);
		break;
		case type_curve:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		case type_qualifier:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		case type_minfeedback:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		default:
			out->type = type_void;
			phoebe_scripter_output ("exception handler invoked in scripter_ast_values_equal(), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_nequal (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function checks whether the first AST value and the second AST
	 * value are not equal. It checks all combinations of value types,
	 * propagates if necessary and performs the != operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	if (val1.type != val2.type) {
		out->value.b = TRUE;
		return SUCCESS;
	}

	out->type = type_bool;
	switch (val1.type) {
		case type_int:
			out->value.b = (val1.value.i != val2.value.i);
		break;
		case type_bool:
			out->value.b = (val1.value.b != val2.value.b);
		break;
		case type_double:
			if (fabs (val1.value.d - val2.value.d) > PHOEBE_NUMERICAL_ACCURACY)
				return TRUE;
			else
				return FALSE;
		break;
		case type_string:
			if (strcmp (val1.value.str, val2.value.str) == 0)
				out->value.b = FALSE;
			else
				out->value.b = TRUE;
		break;
		case type_vector:
			out->value.b = ! phoebe_vector_compare (val1.value.vec, val2.value.vec);
		break;
		case type_array:
			out->value.b = ! phoebe_array_compare (val1.value.array, val2.value.array);
		break;
		case type_spectrum:
			out->value.b = ! phoebe_spectra_compare (val1.value.spectrum, val2.value.spectrum);
		break;
		case type_curve:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		case type_qualifier:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		case type_minfeedback:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		default:
			out->type = type_void;
			phoebe_scripter_output ("exception handler invoked in scripter_ast_values_nequal(), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_lequal (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function checks whether the first AST value is less or equal than
	 * the second AST value. It checks all combinations of value types,
	 * propagates if necessary and performs the <= operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int status;

	if (val1.type == type_void || val2.type == type_void) {
		out->type = type_void;
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '<=' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '<=' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_array || val2.type == type_array) {
		phoebe_scripter_output ("operator '<=' cannot act on non-numeric arrays, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '<=' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_curve || val2.type == type_curve) {
		phoebe_scripter_output ("operator '<=' cannot act on curves, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_qualifier || val2.type == type_qualifier) {
		phoebe_scripter_output ("operator '<=' cannot act on qualifiers, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	out->type = type_bool;
	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.i <= val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.i <= val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, (double) val1.value.i);
					status = phoebe_vector_leq_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_lequal (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.d <= val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.d <= val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, val1.value.d);
					status = phoebe_vector_leq_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_lequal (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, (double) val2.value.i);
					status = phoebe_vector_leq_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_double: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, val2.value.d);
					status = phoebe_vector_leq_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_vector:
					status = phoebe_vector_leq_than (&out->value.b, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
					}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_lequal (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		default:
			out->type = type_void;
			phoebe_scripter_output ("exception handler invoked in scripter_ast_nodes_lequal(), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_gequal (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function checks whether the first AST value is greater or equal
	 * to the second AST value. It checks all combinations of value types,
	 * propagates if necessary and performs the >= operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int status;

	if (val1.type == type_void || val2.type == type_void) {
		out->type = type_void;
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '>=' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '>=' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_array || val2.type == type_array) {
		phoebe_scripter_output ("operator '>=' cannot act on non-numeric arrays, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '>=' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_curve || val2.type == type_curve) {
		phoebe_scripter_output ("operator '>=' cannot act on curves, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_qualifier || val2.type == type_qualifier) {
		phoebe_scripter_output ("operator '>=' cannot act on qualifiers, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	out->type = type_bool;
	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.i >= val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.i >= val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, (double) val1.value.i);
					status = phoebe_vector_geq_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_gequal (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.d >= val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.d >= val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, val1.value.d);
					status = phoebe_vector_geq_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_gequal (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, (double) val2.value.i);
					status = phoebe_vector_geq_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_double: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, val2.value.d);
					status = phoebe_vector_geq_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_vector:
					status = phoebe_vector_geq_than (&out->value.b, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
					}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_gequal (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		default:
			out->type = type_void;
			phoebe_scripter_output ("exception handler invoked in scripter_ast_values_gequal(), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_greater (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function checks whether the first AST value is greater than
	 * the second AST value. It checks all combinations of value types,
	 * propagates if necessary and performs the > operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int status;

	if (val1.type == type_void || val2.type == type_void) {
		out->type = type_void;
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '>' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '>' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_array || val2.type == type_array) {
		phoebe_scripter_output ("operator '>' cannot act on non-numeric arrays, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '>' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_curve || val2.type == type_curve) {
		phoebe_scripter_output ("operator '>' cannot act on curves, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_qualifier || val2.type == type_qualifier) {
		phoebe_scripter_output ("operator '>' cannot act on qualifiers, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	out->type = type_bool;
	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.i > val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.i > val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, (double) val1.value.i);
					status = phoebe_vector_greater_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_greater (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.d > val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.d > val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, val1.value.d);
					status = phoebe_vector_greater_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_greater (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, (double) val2.value.i);
					status = phoebe_vector_greater_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_double: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, val2.value.d);
					status = phoebe_vector_greater_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_vector:
					status = phoebe_vector_greater_than (&out->value.b, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
					}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_greater (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		default:
			out->type = type_void;
			phoebe_scripter_output ("exception handler invoked in scripter_ast_values_greater(), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int scripter_ast_values_less (scripter_ast_value *out, scripter_ast_value val1, scripter_ast_value val2)
{
	/*
	 * This function checks whether the first AST value is less than
	 * the second AST value. It checks all combinations of value types,
	 * propagates if necessary and performs the < operation.
	 *
	 * Return values:
	 *
	 *   ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS
	 *   SUCCESS
	 */

	int status;

	if (val1.type == type_void || val2.type == type_void) {
		out->type = type_void;
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_bool || val2.type == type_bool) {
		phoebe_scripter_output ("operator '<' cannot act on booleans, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_string || val2.type == type_string) {
		phoebe_scripter_output ("operator '<' cannot act on strings, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_array || val2.type == type_array) {
		phoebe_scripter_output ("operator '<' cannot act on non-numeric arrays, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_minfeedback || val2.type == type_minfeedback) {
		phoebe_scripter_output ("operator '<' cannot act on minimizer feedback, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_curve || val2.type == type_curve) {
		phoebe_scripter_output ("operator '<' cannot act on curves, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	if (val1.type == type_qualifier || val2.type == type_qualifier) {
		phoebe_scripter_output ("operator '<' cannot act on qualifiers, aborting.\n");
		return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
	}

	out->type = type_bool;
	switch (val1.type) {
		case type_int:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.i < val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.i < val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, (double) val1.value.i);
					status = phoebe_vector_less_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_less (), please report this!\n");
				break;
			}
		break;
		case type_double:
			switch (val2.type) {
				case type_int:
					out->value.b = (val1.value.d < val2.value.i);
				break;
				case type_double:
					out->value.b = (val1.value.d < val2.value.d);
				break;
				case type_vector: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val2.value.vec->dim);
					phoebe_vector_pad (vec, val1.value.d);
					status = phoebe_vector_less_than (&out->value.b, vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_less (), please report this!\n");
				break;
			}
		break;
		case type_vector:
			switch (val2.type) {
				case type_int: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, (double) val2.value.i);
					status = phoebe_vector_less_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_double: {
					PHOEBE_vector *vec = phoebe_vector_new ();
					phoebe_vector_alloc (vec, val1.value.vec->dim);
					phoebe_vector_pad (vec, val2.value.d);
					status = phoebe_vector_less_than (&out->value.b, val1.value.vec, vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
						return status;
					}
					phoebe_vector_free (vec);
				}
				break;
				case type_vector:
					status = phoebe_vector_less_than (&out->value.b, val1.value.vec, val2.value.vec);
					if (status != SUCCESS) {
						phoebe_scripter_output ("%s", phoebe_scripter_error (status));
						out->type = type_void;
					}
				break;
				default:
					printf ("exception handler invoked in scripter_ast_values_less (), please report this!\n");
				break;
			}
		break;
		case type_spectrum:
			out->type = type_void;
			phoebe_scripter_output ("not yet implemented.\n");
		break;
		default:
			out->type = type_void;
			phoebe_scripter_output ("exception handler invoked in scripter_ast_values_less(), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}
