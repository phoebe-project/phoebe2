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
					out->value.str = concatenate_strings (str, val2.value.str, NULL);
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
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					propagate_int_to_double (&val1);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.d + val2.value.spectrum->data->val[i];
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
					out->value.str = concatenate_strings (str, val2.value.str, NULL);
				break;
				case type_vector:
					out->type = type_vector;
					out->value.vec = phoebe_vector_new ();
					phoebe_vector_alloc (out->value.vec, val2.value.vec->dim);
					for (i = 0; i < out->value.vec->dim; i++)
						out->value.vec->val[i] = val1.value.d + val2.value.vec->val[i];
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_spectrum:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val2.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.d + val2.value.spectrum->data->val[i];
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
					out->value.str = concatenate_strings (val1.value.str, str, NULL);
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_string;
					sprintf (str, "%lf", val2.value.d);
					out->value.str = concatenate_strings (val1.value.str, str, NULL);
				break;
				case type_string:
					out->type = type_string;
					out->value.str = concatenate_strings (val1.value.str, val2.value.str, NULL);
				break;
				case type_vector:
					phoebe_scripter_output ("operator '+' cannot act between strings and arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
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
					phoebe_scripter_output ("not yet implemented, sorry.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
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
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
				break;
				case type_string:
					phoebe_scripter_output ("operator '+' cannot act between strings and arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_vector:
					phoebe_scripter_output ("operator '+' cannot act between numeric and non-numeric arrays, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_array:
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] + val2.value.d;
				break;
				case type_bool:
					phoebe_scripter_output ("operator '+' cannot act on booleans, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] + val2.value.d;
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.d - val2.value.spectrum->data->val[i];
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.d - val2.value.spectrum->data->val[i];
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] - val2.value.d;
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] - val2.value.d;
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
					phoebe_scripter_output ("operator '*' cannot act between arrays and spectra, aborting.\n");
					return ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS;
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.d / val2.value.spectrum->data->val[i];
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.d / val2.value.spectrum->data->val[i];
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
					propagate_int_to_double (&val2);
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] / val2.value.d;
				break;
				case type_double:
					out->type = type_spectrum;
					out->value.spectrum = phoebe_spectrum_new ();
					phoebe_spectrum_alloc (out->value.spectrum, val1.value.spectrum->data->bins);
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = val1.value.spectrum->data->val[i] / val2.value.d;
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
					out->type = type_void;
					phoebe_scripter_output ("not yet implemented, sorry.\n");
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.i, val2.value.spectrum->data->val[i]);
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.d, val2.value.spectrum->data->val[i]);
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.vec->val[i], val2.value.spectrum->data->val[i]);
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
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.i);
				break;
				case type_double:
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.d);
				break;
				case type_vector:
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.vec->val[i]);
				break;
				case type_spectrum:
					for (i = 0; i < out->value.spectrum->data->bins; i++)
						out->value.spectrum->data->val[i] = pow (val1.value.spectrum->data->val[i], val2.value.spectrum->data->val[i]);
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
