#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_build_config.h"
#include "phoebe_scripter_commands.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter_grammar.h"
#include "phoebe_scripter_grammar.tab.h"
#include "phoebe_scripter_io.h"
#include "phoebe_scripter_types.h"

#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
	#include <readline/readline.h>
#endif

int scripter_register_all_commands ()
{
	/* Commands for handling parameter files:                                 */
	scripter_command_register ("open_parameter_file",          scripter_open_parameter_file);
	scripter_command_register ("save_parameter_file",          scripter_save_parameter_file);
	scripter_command_register ("create_wd_lci_file",           scripter_create_wd_lci_file);

	/* Commands for initializing variables:                                   */
	scripter_command_register ("array",                        scripter_array);
	scripter_command_register ("column",                       scripter_column);
	scripter_command_register ("spectrum",                     scripter_spectrum);

	/* Commands for handling parameters:                                      */
	scripter_command_register ("set_parameter_value",          scripter_set_parameter_value);
	scripter_command_register ("set_parameter_limits",         scripter_set_parameter_limits);
	scripter_command_register ("set_parameter_step",           scripter_set_parameter_step);
	scripter_command_register ("mark_for_adjustment",          scripter_mark_for_adjustment);
	scripter_command_register ("get_parameter_value",          scripter_get_parameter_value);

	/* Commands for handling constraints:                                     */
	scripter_command_register ("add_constraint",               scripter_add_constraint);

	/* Commands for handling data:                                            */
	scripter_command_register ("set_lc_properties",            scripter_set_lc_properties);
	scripter_command_register ("compute_lc",                   scripter_compute_lc);
	scripter_command_register ("compute_rv",                   scripter_compute_rv);
	scripter_command_register ("compute_chi2",                 scripter_compute_chi2);
	scripter_command_register ("transform_hjd_to_phase",       scripter_transform_hjd_to_phase);
	scripter_command_register ("transform_flux_to_magnitude",  scripter_transform_flux_to_magnitude);

	/* Commands for minimization methods:                                     */
	scripter_command_register ("minimize_using_nms",           scripter_minimize_using_nms);
	scripter_command_register ("minimize_using_dc",            scripter_minimize_using_dc);
	scripter_command_register ("adopt_minimizer_results",      scripter_adopt_minimizer_results);

	/* Commands for handling spectral energy distributions (SED):             */
	scripter_command_register ("set_spectra_repository",       scripter_set_spectra_repository);
	scripter_command_register ("set_spectrum_properties",      scripter_set_spectrum_properties);
	scripter_command_register ("get_spectrum_from_repository", scripter_get_spectrum_from_repository);
	scripter_command_register ("get_spectrum_from_file",       scripter_get_spectrum_from_file);
	scripter_command_register ("broaden_spectrum",             scripter_broaden_spectrum);
	scripter_command_register ("crop_spectrum",                scripter_crop_spectrum);
	scripter_command_register ("resample_spectrum",            scripter_resample_spectrum);
	scripter_command_register ("integrate_spectrum",           scripter_integrate_spectrum);
	scripter_command_register ("apply_doppler_shift",          scripter_apply_doppler_shift);
	scripter_command_register ("apply_rotational_broadening",  scripter_apply_rotational_broadening);
	scripter_command_register ("merge_spectra",                scripter_merge_spectra);
	scripter_command_register ("multiply_spectra",             scripter_multiply_spectra);

	/* Commands for handling limb darkening coefficients:                     */
	scripter_command_register ("get_ld_coefficients",          scripter_get_ld_coefficients);

	/* Plotting commands:                                                     */
	scripter_command_register ("plot_lc_using_gnuplot",        scripter_plot_lc_using_gnuplot);
	scripter_command_register ("plot_rv_using_gnuplot",        scripter_plot_rv_using_gnuplot);
	scripter_command_register ("plot_spectrum_using_gnuplot",  scripter_plot_spectrum_using_gnuplot);
	scripter_command_register ("plot_using_gnuplot",           scripter_plot_using_gnuplot);

	/* Other auxiliary commands:                                              */
	scripter_command_register ("prompt",                       scripter_prompt);

	return SUCCESS;
}

scripter_ast_value scripter_open_parameter_file (scripter_ast_list *args)
{
	/*
	 * This functions reads in parameters from the passed parameter file.
	 *
	 * Synopsis:
	 *
	 *   open_parameter_file (string)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	int status;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) return out;

	status = phoebe_open_parameter_file (vals[0].value.str);

	if (status != SUCCESS)
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 1);

	return out;
}

scripter_ast_value scripter_save_parameter_file (scripter_ast_list *args)
{
	/*
	 * This functions saves parameters to the passed parameter file.
	 *
	 * Synopsis:
	 *
	 *   save_parameter_file (string)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	int status;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) return out;

	status = phoebe_save_parameter_file (vals[0].value.str);
	if (status != SUCCESS)
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 1);
	return out;
}

scripter_ast_value scripter_set_parameter_value (scripter_ast_list *args)
{
	/*
	 * This function sets the parameter value referenced by its qualifier.
	 * The value type must match the type of the qualifier, which is verified
	 * during argument evaluation. If the parameter is an array, the third
	 * argument that is passed is the index, starting from 1.
	 *
	 * Synopsis:
	 *
	 *   set_parameter_value (qualifier, value)
	 *   set_parameter_value (qualifier, value, index)
	 *
	 * Where:
	 *
	 *   qualifier  ..  reference to the parameter
	 *   value      ..  value of any type that corresponds to the qualifier
	 *   index      ..  array index, in cases when parameter is an array
	 */

	int status;

	scripter_ast_value qualifier, val, index, out;
	PHOEBE_parameter *par;

	/* If anything goes wrong, we return void. */
	out.type = type_void;

	if (!args) {
		phoebe_scripter_output ("argument number mismatch: 0 passed, 2 or 3 expected.\n");
		return out;
	}
	if (!args->next) {
		phoebe_scripter_output ("argument number mismatch: 1 passed, 2 or 3 expected.\n");
		return out;
	}

	qualifier = scripter_ast_evaluate (args->elem);

	if (qualifier.type != type_qualifier) {
		phoebe_scripter_output ("argument 1 type mismatch: %s passed, qualifier expected.\n", scripter_ast_value_type_get_name (qualifier.type));
		scripter_ast_value_free (qualifier);
		return out;
	}

	par = phoebe_parameter_lookup (qualifier.value.str);
	if (!par) {
		phoebe_scripter_output ("parameter %s is not defined, aborting.\n", qualifier.value.str);
		scripter_ast_value_free (qualifier);
		return out;
	}

	switch (par->type) {
		case TYPE_INT:
			if (args->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 2 passed, 2 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (val.type != type_int) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, integer expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}

			status = phoebe_parameter_set_value (par, val.value.i);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			return out;
		break;
		case TYPE_BOOL:
			if (args->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 2 passed, 2 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (val.type == type_int) propagate_int_to_bool (&val);
			if (val.type != type_bool) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, boolean expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}
			status = phoebe_parameter_set_value (par, val.value.b);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			return out;
		break;
		case TYPE_DOUBLE:
			if (args->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 2 passed, 2 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);
		
			if (val.type == type_int) propagate_int_to_double (&val);
			if (val.type != type_double) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, double expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}
			status = phoebe_parameter_set_value (par, val.value.d);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			return out;
		break;
		case TYPE_STRING:
			if (args->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 2 passed, 2 expected.\n");
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (par->kind == KIND_MENU && val.type == type_int)
				propagate_int_to_menu_item (&val, qualifier.value.str);

			if (val.type != type_string) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, string expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}
			status = phoebe_parameter_set_value (par, val.value.str);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			return out;
		break;
		case TYPE_INT_ARRAY:
			if (!args->next->next) {
				phoebe_scripter_output ("argument number mismatch: 2 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}
			if (args->next->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 3 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (val.type != type_int) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, integer expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}

			index = scripter_ast_evaluate (args->next->next->elem);

			if (index.type != type_int) {
				phoebe_scripter_output ("argument 3 type mismatch: %s passed, integer expected.\n", scripter_ast_value_type_get_name (index.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				scripter_ast_value_free (index);
				return out;
			}
			status = phoebe_parameter_set_value (par, index.value.i-1, val.value.i);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			scripter_ast_value_free (index);
			return out;
		break;
		case TYPE_BOOL_ARRAY:
			if (!args->next->next) {
				phoebe_scripter_output ("argument number mismatch: 2 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}
			if (args->next->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 3 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (val.type == type_int) propagate_int_to_bool (&val);
			if (val.type != type_bool) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, boolean expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}

			index = scripter_ast_evaluate (args->next->next->elem);

			if (index.type != type_int) {
				phoebe_scripter_output ("argument 3 type mismatch: %s passed, integer expected.\n", scripter_ast_value_type_get_name (index.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				scripter_ast_value_free (index);
				return out;
			}
			status = phoebe_parameter_set_value (par, index.value.i-1, val.value.b);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			scripter_ast_value_free (index);
			return out;
		break;
		case TYPE_DOUBLE_ARRAY:
			if (!args->next->next) {
				phoebe_scripter_output ("argument number mismatch: 2 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}
			if (args->next->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 3 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (val.type == type_int) propagate_int_to_double (&val);
			if (val.type != type_double) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, double expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}

			index = scripter_ast_evaluate (args->next->next->elem);

			if (index.type != type_int) {
				phoebe_scripter_output ("argument 3 type mismatch: %s passed, integer expected.\n", scripter_ast_value_type_get_name (index.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				scripter_ast_value_free (index);
				return out;
			}
			status = phoebe_parameter_set_value (par, index.value.i-1, val.value.d);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			scripter_ast_value_free (index);
			return out;
		break;
		case TYPE_STRING_ARRAY:
			if (!args->next->next) {
				phoebe_scripter_output ("argument number mismatch: 2 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}
			if (args->next->next->next) {
				phoebe_scripter_output ("argument number mismatch: more than 3 passed, 3 expected.\n");
				scripter_ast_value_free (qualifier);
				return out;
			}

			val = scripter_ast_evaluate (args->next->elem);

			if (par->kind == KIND_MENU && val.type == type_int) propagate_int_to_menu_item (&val, qualifier.value.str);
			if (val.type != type_string) {
				phoebe_scripter_output ("argument 2 type mismatch: %s passed, string expected.\n", scripter_ast_value_type_get_name (val.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				return out;
			}

			index = scripter_ast_evaluate (args->next->next->elem);

			if (index.type != type_int) {
				phoebe_scripter_output ("argument 3 type mismatch: %s passed, integer expected.\n", scripter_ast_value_type_get_name (index.type));
				scripter_ast_value_free (qualifier);
				scripter_ast_value_free (val);
				scripter_ast_value_free (index);
				return out;
			}
			status = phoebe_parameter_set_value (par, index.value.i-1, val.value.str);
			if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_free (qualifier);
			scripter_ast_value_free (val);
			scripter_ast_value_free (index);
			return out;
		break;
		default:
			phoebe_scripter_output ("exception handler invoked in set_parameter_value (), please report this!\n");
			return out;
		break;
	}
}

scripter_ast_value scripter_mark_for_adjustment (scripter_ast_list *args)
{
	/*
	 * This functions sets the TBA (To Be Adjusted) bit of the adjustable
	 * parameter on or off.
	 *
	 * Synopsis:
	 *
	 *   mark_for_adjustment (qualifier, state)
	 *
	 * Where:
	 *
	 *   qualifier  ..  parameter name
	 *   state      ..  bit state, TRUE (1) or FALSE (0)
	 */

	int status;
	scripter_ast_value out, *vals;

	PHOEBE_parameter *par;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 2, 2, type_qualifier, type_bool);
	if (status != SUCCESS) return out;

	par = phoebe_parameter_lookup (vals[0].value.str);
	if (!par) {
		phoebe_scripter_output ("parameter %s not recognized, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 2);
		return out;
	}

	if (par->kind != KIND_ADJUSTABLE) {
		phoebe_scripter_output ("qualifier %s is not adjustable.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 2);
		return out;
	}

	status = phoebe_parameter_set_tba (par, vals[1].value.b);
	if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 2);

	return out;
}

scripter_ast_value scripter_set_parameter_limits (scripter_ast_list *args)
{
	/*
	 * This function sets lower and upper limits (min, max) of the parameter
	 * referenced by the qualifier. If the lower limit is larger than the
	 * upper limit, assignment will fail.
	 *
	 * Synopsis:
	 *
	 *   set_parameter_limits (qualifier, min, max)
	 */

	int status;
	scripter_ast_value out, *vals;
	PHOEBE_parameter *par;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 3, 3, type_qualifier, type_double, type_double);
	if (status != SUCCESS) return out;

	par = phoebe_parameter_lookup (vals[0].value.str);
	if (!par) {
		phoebe_scripter_output ("parameter %s not recognized, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 3);
		return out;
	}

	if (par->kind != KIND_ADJUSTABLE) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (ERROR_QUALIFIER_NOT_ADJUSTABLE));
		scripter_ast_value_array_free (vals, 3);
		return out;
	}

	/* Check whether max is larger than min:                                  */
	if (vals[1].value.d >= vals[2].value.d) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (ERROR_PARAMETER_INVALID_LIMITS));
		scripter_ast_value_array_free (vals, 3);
		return out;
	}

	status = phoebe_parameter_set_limits (par, vals[1].value.d, vals[2].value.d);
	if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 3);
	return out;
}

scripter_ast_value scripter_set_parameter_step (scripter_ast_list *args)
{
	/*
	 * This function sets the step size for the DC minimizer for the parameter
	 * referenced by the qualifier.
	 *
	 * Synopsis:
	 *
	 *   set_parameter_step (qualifier, step)
	 */

	int status;
	scripter_ast_value out, *vals;
	PHOEBE_parameter *par;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 2, 2, type_qualifier, type_double);
	if (status != SUCCESS) return out;

	par = phoebe_parameter_lookup (vals[0].value.str);
	if (!par) {
		phoebe_scripter_output ("parameter %s not recognized, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 2);
		return out;
	}

	if (par->kind != KIND_ADJUSTABLE) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (ERROR_QUALIFIER_NOT_ADJUSTABLE));
		scripter_ast_value_array_free (vals, 2);
		return out;
	}

	status = phoebe_parameter_set_step (par, vals[1].value.d);
	if (status != SUCCESS) phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_create_wd_lci_file (scripter_ast_list *args)
{
	/*
	 * This part allows the user to create WD2003 lci input file to use it
	 * directly with WD.
	 *
	 * Synopsis:
	 *
	 *   create_wd_lci_file (filename, mpage, curve)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	WD_LCI_parameters params;
	int status;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 3, 3, type_string, type_int, type_int);
	if (status != SUCCESS) return out;

	status = read_in_wd_lci_parameters (&params, vals[1].value.i, vals[2].value.i-1);
	if (status != SUCCESS) printf ("%s", phoebe_scripter_error (status));
	else {
		status = create_lci_file (vals[0].value.str, &params);
		if (status != SUCCESS)
			phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		else
			phoebe_scripter_output ("WD lci file '%s' created.\n", vals[0].value.str);
	}

	scripter_ast_value_array_free (vals, 3);
	return out;
}

scripter_ast_value scripter_array (scripter_ast_list *args)
{
	/*
	 * This command initializes a 0-padded array.
	 *
	 * Synopsis:
	 *
	 *   set vec = array (dim)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 1, 1, type_int);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	out.type = type_vector;
	out.value.vec =	phoebe_vector_new ();
	status = phoebe_vector_alloc (out.value.vec, vals[0].value.i);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 1);
		out.type = type_void;
		return out;
	}

	phoebe_vector_pad (out.value.vec, 0.0);

	scripter_ast_value_array_free (vals, 1);
	return out;
}

scripter_ast_value scripter_column (scripter_ast_list *args)
{
	/*
	 * This command reads in the i-th column from the passed filename.
	 *
	 * Synopsis:
	 *
	 *   set array = column (filename, i)
	 *
	 * Where:
	 *
	 *   filename  ..  absolute filename (string)
	 *   i         ..  column index (starting from 1)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_string, type_int);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	out.type = type_vector;
	out.value.vec = phoebe_vector_new_from_column (vals[0].value.str, vals[1].value.i);
	if (!out.value.vec) {
		phoebe_scripter_output ("file '%s' not found, aborting.\n", vals[0].value.str);
		out.type = type_void;
	}

	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_spectrum (scripter_ast_list *args)
{
	/*
	 * This command initializes a spectrum.
	 *
	 * Synopsis:
	 *
	 *   set vec = spectrum (dim, R, Rs)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 3, 3, type_int, type_double, type_double);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	out.type = type_spectrum;
	out.value.spectrum = phoebe_spectrum_new ();
	status = phoebe_spectrum_alloc (out.value.spectrum, vals[0].value.i);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 1);
		out.type = type_void;
		return out;
	}
	out.value.spectrum->R  = vals[1].value.d;
	out.value.spectrum->Rs = vals[2].value.d;

	scripter_ast_value_array_free (vals, 3);
	return out;
}

scripter_ast_value scripter_prompt (scripter_ast_list *args)
{
	/*
	 * This command provides input from the user.
	 *
	 * Synopsis:
	 *
	 *   set answer = prompt ("Question? " [, "expected_type"])
	 *
	 * First argument is a string that is used as the prompt to which
	 * the user has to give an answer. The answer is then parsed and
	 * returned as an expression. Typically one whould use this command as:
	 *
	 *   set answer = prompt ("Is this correct [Y/n]? ")
	 *
	 * Second argument is optional and may be used to test the answer type.
	 * The expected_type is a string that must be one of the following:
	 *
	 *   integer, boolean, double, string
	 *
	 * The command is completely fool-proof and will loop until the answer
	 * to the prompt is acceptable by the lexer. The command will interrupt
	 * on empty input or on CTRL+D. An example:
	 *
	 *   set answer = prompt ("Pass me a string: ", "string")
	 */

	char *readout;
	bool all_is_ok;

	scripter_ast_value out;
	scripter_ast_value *vals;
	YY_BUFFER_STATE yybuf;

	PHOEBE_type expected_type = -1;
	int type;

	int status = scripter_command_args_evaluate (args, &vals, 1, 2, type_string, type_string);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	if (vals[1].type != type_void) {
		/* This means that we know which type to expect:                      */
		expected_type = -1;
		if (strcmp (vals[1].value.str, "integer") == 0) expected_type = TYPE_INT;
		if (strcmp (vals[1].value.str, "boolean") == 0) expected_type = TYPE_BOOL;
		if (strcmp (vals[1].value.str, "double")  == 0) expected_type = TYPE_DOUBLE;
		if (strcmp (vals[1].value.str, "string")  == 0) expected_type = TYPE_STRING;
		if (expected_type == -1) {
			phoebe_scripter_output ("type '%s' not recognized, aborting.\n", vals[1].value.str);
			scripter_ast_value_array_free (vals, 2);
			out.type = type_void;
			return out;
		}
	}

	do {

#if defined HAVE_LIBREADLINE && !defined PHOEBE_READLINE_DISABLED
		readout = readline (vals[0].value.str);
#endif

#if (defined HAVE_LIBREADLINE && defined PHOEBE_READLINE_DISABLED) || (!defined HAVE_LIBREADLINE)
		fprintf (PHOEBE_output, "%s", vals[0].value.str);
		readout = phoebe_malloc (255);
		fgets (readout, 255, stdin);
		readout[strlen(readout)-1] = '\0';
#endif

		/* This will return us to the main prompt on CTRL+D: */
		if (!readout) {
			printf ("\n");
			out.type = type_void;
			return out;
		}

		/* This will return us to the main prompt on newline: */
		if (*readout == '\0' || *readout == '\n') {
			out.type = type_void;
			return out;
		}

		yybuf = yy_scan_string (readout);
		type = yylex ();

		switch (type) {
			case INTEGER:
				if (expected_type != -1 && expected_type != TYPE_INT) {
					phoebe_scripter_output ("please provide %s-type input.\n", phoebe_type_get_name (expected_type));
					all_is_ok = FALSE;
					continue;
				}
				out.type = type_int;
				out.value.i = atoi (readout);
			break;
			case BOOLEAN:
				if (expected_type != -1 && expected_type != TYPE_BOOL) {
					phoebe_scripter_output ("please provide %s-type input.\n", phoebe_type_get_name (expected_type));
					all_is_ok = FALSE;
					continue;
				}
				out.type = type_bool;
				out.value.b = atob (readout);
			break;
			case VALUE:
				if (expected_type != -1 && expected_type != TYPE_DOUBLE) {
					phoebe_scripter_output ("please provide %s-type input.\n", phoebe_type_get_name (expected_type));
					all_is_ok = FALSE;
					continue;
				}
				out.type = type_double;
				out.value.d = atof (readout);
			break;
			case LITERAL:
				if (expected_type != -1 && expected_type != TYPE_STRING) {
					phoebe_scripter_output ("please provide %s-type input.\n", phoebe_type_get_name (expected_type));
					all_is_ok = FALSE;
					continue;
				}
				out.type = type_string;
				/* Strip the quotation marks: */
				readout[strlen(readout)-1] = '\0';
				out.value.str = strdup (readout+1);
			break;
			case IDENT:
				if (expected_type != -1 && expected_type != TYPE_STRING) {
					phoebe_scripter_output ("please provide %s-type input.\n", phoebe_type_get_name (expected_type));
					all_is_ok = FALSE;
					continue;
				}
				out.type = type_string;
				out.value.str = strdup (readout);
			break;
			default:
				phoebe_scripter_output ("prompt can only parse integers, booleans, doubles and strings.\n");
				all_is_ok = FALSE;
				continue;
			break;
		}

		yy_delete_buffer (yybuf);
		free (readout);
		all_is_ok = TRUE;
	} while (!all_is_ok);

	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_get_parameter_value (scripter_ast_list *args)
{
	/*
	 * This function returns the value of the passed qualifier.
	 *
	 * Synopsis:
	 *
	 *   set var = get_parameter_value (qualifier)
	 *   set var = get_parameter_value (qualifier, index)
	 *
	 * The first form will read out scalar values into the variable var and
	 * array values into the vector variable. The second form is applicable
	 * only on vector parameters and will return a scalar value of the cor-
	 * responding type and of the corresponding index.
	 */

	int status;
	scripter_ast_value out, *vals;
	PHOEBE_parameter *par;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 2, type_qualifier, type_int);
	if (status != SUCCESS) return out;

	par = phoebe_parameter_lookup (vals[0].value.str);
	if (!par) {
		phoebe_scripter_output ("parameter %s not recognized, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 2);
		return out;
	}

	if (vals[1].type != type_void && (par->type == TYPE_INT || par->type == TYPE_BOOL || par->type == TYPE_DOUBLE || par->type == TYPE_STRING)) {
		phoebe_scripter_output ("qualifier %s is not an array, aborting.\n", vals[0].value.str);
		scripter_ast_value_array_free (vals, 2);
		return out;
	}

	/* Fix OB1 for the index: */
	if (vals[1].type != type_void)
		vals[1].value.i -= 1;

	switch (par->type) {
		case (TYPE_INT):
			out.type = type_int;
			status = phoebe_parameter_get_value (par, &(out.value.i));
		break;
		case (TYPE_BOOL):
			out.type = type_bool;
			status = phoebe_parameter_get_value (par, &(out.value.b));
		break;
		case (TYPE_DOUBLE):
			out.type = type_double;
			status = phoebe_parameter_get_value (par, &(out.value.d));
		break;
		case (TYPE_STRING):
			out.type = type_string;
			status = phoebe_parameter_get_value (par, &(out.value.str));
		break;
		case (TYPE_INT_ARRAY):
			if (vals[1].type == type_void) {
				out.type = type_array;
				out.value.array = phoebe_array_new_from_qualifier (vals[0].value.str);
			}
			else {
				out.type = type_int;
				status = phoebe_parameter_get_value (par, vals[1].value.i, &out.value.i);
			}
		break;
		case (TYPE_BOOL_ARRAY):
			if (vals[1].type == type_void) {
				out.type = type_array;
				out.value.array = phoebe_array_new_from_qualifier (vals[0].value.str);
			}
			else {
				out.type = type_bool;
				status = phoebe_parameter_get_value (par, vals[1].value.i, &out.value.b);
			}
		break;
		case (TYPE_DOUBLE_ARRAY):
			if (vals[1].type == type_void) {
				out.type = type_vector;
				out.value.vec = phoebe_vector_new_from_qualifier (vals[0].value.str);
			}
			else {
				out.type = type_double;
				status = phoebe_parameter_get_value (par, vals[1].value.i, &out.value.d);
			}
		break;
		case (TYPE_STRING_ARRAY):
			if (vals[1].type == type_void) {
				out.type = type_array;
				out.value.array = phoebe_array_new_from_qualifier (vals[0].value.str);
			}
			else {
				char *readout;
				out.type = type_string;
				status = phoebe_parameter_get_value (par, vals[1].value.i, &readout);
				out.value.str = strdup (readout);
			}
		break;
		default:
			phoebe_scripter_output ("Exception handler invoked in scripter_get_parameter_value (), please report this!\n");
			out.type = type_void;
		break;
	}

	if (status != SUCCESS)
		out.type = type_void;

	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_add_constraint (scripter_ast_list *args)
{
	/*
	 * This function adds a constraint to the pool of constraints.
	 *
	 * Synopsis:
	 *
	 *   add_constraint ("constraint")
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	int status;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) return out;

	status = phoebe_constraint_new (vals[0].value.str);
	if (status != SUCCESS)
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 1);
	return out;
}

scripter_ast_value scripter_minimize_using_dc (scripter_ast_list *args)
{
	/*
	 * This function calls WD's Differential Corrections (DC) minimizer.
	 *
	 * Synopsis:
	 *
	 *   set chi2 = minimize_using_dc ()
	 */

	PHOEBE_minimizer_feedback *feedback;

	scripter_ast_value out;
	int status = scripter_command_args_evaluate (args, NULL, 0, 0);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	phoebe_scripter_output ("DC: initiating the minimization.\n");

	feedback = phoebe_minimizer_feedback_new ();
	status = phoebe_minimize_using_dc (PHOEBE_output, feedback);

	/* Return the minimizer structure value if everything was ok:             */
	if (status == SUCCESS) {
		out.type = type_minfeedback;
		out.value.feedback = feedback;
	}
	else {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
	}

	/* Say goodbye:                                                           */
	phoebe_scripter_output ("DC minimization done.\n");

	return out;
}

scripter_ast_value scripter_minimize_using_nms (scripter_ast_list *args)
{
	/*
	 * This function calls the Nelder & Mead simplex minimizer. It only works
	 * if GSL is present.
	 *
	 * Synopsis:
	 *
	 *   set chi2 = minimize_using_nms (tolerance, max_iter)
	 */

	PHOEBE_minimizer_feedback *feedback;

	scripter_ast_value out;
	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_double, type_int);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	if (vals[1].value.i < 0) {
		phoebe_scripter_output ("attempt to set negative number of iterations, aborting.\n");
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}
	if (vals[1].value.i == 0) {
		phoebe_scripter_output ("NMS: tol = %2.2e, unlimited iterations:\n", vals[0].value.d);
		vals[1].value.i = 1e8;
	}
	else
		phoebe_scripter_output ("NMS: tol = %2.2e, max %d iterations:\n", vals[0].value.d, vals[1].value.i);

	/* Initialize the feedback structure:                                     */
	feedback = phoebe_minimizer_feedback_new ();

	/* Call the downhill simplex minimizer:                                   */
	status = phoebe_minimize_using_nms (vals[0].value.d, vals[1].value.i, PHOEBE_output, feedback);

	/* Return the minimizer structure value if everything was ok:             */
	if (status == SUCCESS) {
		out.type = type_minfeedback;
		out.value.feedback = feedback;
	}
	else {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
	}

	/* Say goodbye:                                                             */
	phoebe_scripter_output ("NMS minimization done.\n");
	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_adopt_minimizer_results (scripter_ast_list *args)
{
	/*
	 * This function adopts the results of the minimizer (DC, NMS, ...) that
	 * are stored in a read-only feedback structure.
	 *
	 * Synopsis:
	 *
	 *   adopt_minimizer_results (feedback)
	 */

	int status;

	scripter_ast_value out;
	scripter_ast_value *vals;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_minfeedback);
	if (status != SUCCESS) return out;

	status = phoebe_minimizer_feedback_accept (vals[0].value.feedback);
	if (status != SUCCESS)
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));

	scripter_ast_value_array_free (vals, 1);
	out.type = type_void;
	return out;
}

scripter_ast_value scripter_compute_perr0_phase (scripter_ast_list *args)
{
	/*
	 * This function computes the orbital phase of the periastron passage.
	 *
	 * Synopsis:
	 *
	 *   compute_perr0_phase (omega, e)
	 *
	 * Where:
	 *
	 *   omega  ..  argument of periastron
	 *   e      ..  orbital eccentricity
	 */

	scripter_ast_value out;
	double phase;

	scripter_ast_value *vals;
	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_double, type_double);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	status = calculate_periastron_orbital_phase (&phase, vals[0].value.d, vals[1].value.d);
	if (status != SUCCESS) {
		phoebe_scripter_error (status);
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	out.type = type_double;
	out.value.d = phase;
	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_transform_hjd_to_phase (scripter_ast_list *args)
{
	/*
	 * This function transforms independent HJD variable to phase.
	 *
	 * Synopsis:
	 *
	 *   set new = transform_hjd_to_phase (orig [, hjd0, period, dpdt, pshift])
	 *
	 * Where:
	 *
	 *   orig     ..  original data given in HJD
	 *   hjd0     ..  zero HJD
	 *   period   ..  orbital period of the binary
	 *   dpdt     ..  orbital period change
	 *   pshift   ..  phase shift
	 *
	 * Ephemeris parameters are optional. If they're missing, the value is
	 * taken from the parameter table.
	 *
	 * A final remark: after the function exits, all local variables are freed,
	 * including the vec pointer, but *not* the underlaying array. So it is ok
	 * to simply point the output value to this local pointer, although the
	 * pointer itself will be freed.
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;

	PHOEBE_vector *vec;

	double hjd0, period, dpdt, pshift;

	int status = scripter_command_args_evaluate (args, &vals, 1, 5, type_vector, type_double, type_double, type_double, type_double);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);

	/* If we have any optional arguments, use them to overwrite original values */
	if (vals[1].type != type_void)
		hjd0 = vals[1].value.d;
	if (vals[2].type != type_void)
		period = vals[2].value.d;
	if (vals[3].type != type_void)
		dpdt = vals[3].value.d;
	if (vals[4].type != type_void)
		pshift = vals[4].value.d;

	vec = phoebe_vector_duplicate (vals[0].value.vec);

	/* Now we have data stored in the 'vec' variable. We now transform it:    */
	transform_hjd_to_phase (vec, hjd0, period, dpdt, pshift);

	out.type = type_vector;
	out.value.vec = vec;
	scripter_ast_value_array_free (vals, 5);
	return out;
}

scripter_ast_value scripter_transform_flux_to_magnitude (scripter_ast_list *args)
{
	/*
	 * This function transforms dependent flux variable to magnitude.
	 *
	 * Synopsis:
	 *
	 *  set new = transform_flux_to_magnitude (orig [, mnorm])
	 *
	 * Where:
	 *
	 *  orig   ..  the input array containing fluxes
	 *  mnorm  ..  magnitude that corresponds to unity flux
	 *
	 * Norm magnitude argument is optional. If it's missing, the value is taken
	 * from the parameter table.
	 *
	 * A final remark: after the function exits, all local variables are freed,
	 * including the vec pointer, but *not* the underlaying array. So it is ok
	 * to simply point the output value to this local pointer, although the
	 * pointer itself will be freed.
	 */

	double mnorm;
	PHOEBE_vector *vec;
	scripter_ast_value out;

	scripter_ast_value *vals;
	int status = scripter_command_args_evaluate (args, &vals, 1, 2, type_vector, type_double);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	if (vals[1].type == type_void)
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mnorm"), &mnorm);
	else
		mnorm = vals[1].value.d;

	vec = phoebe_vector_duplicate (vals[0].value.vec);
	status = transform_flux_to_magnitude (vec, mnorm);
	if (status != SUCCESS) {
		phoebe_vector_free (vec);
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	out.type = type_vector;
	out.value.vec = vec;
	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_plot_lc_using_gnuplot (scripter_ast_list *args)
{
	/*
	 * This function plots the light curve that is passed as the first argument.
	 * The plot is not meant to be publication-ready, it is meant only for a
	 * quick inspection. It always plots in phase on the [-0.6, 0.6] range in
	 * magnitude units.
	 *
	 * Synopsis:
	 *
	 *   plot_lc_using_gnuplot (curve, obsdata_switch, syndata_switch)
	 */

	int i;
	int status;
	int lcno, curve;

	int index = 0;

	PHOEBE_curve **lc             = NULL;
	PHOEBE_plot_properties *props = NULL;

	PHOEBE_vector **indeps, **deps;

	scripter_ast_value out;
	scripter_ast_value *vals;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 3, 3, type_int, type_bool, type_bool);
	if (status != SUCCESS) return out;

	/* Is the curve initialized?                                              */
	curve = vals[0].value.i;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	if (curve <= 0 || curve > lcno) {
		phoebe_scripter_output ("light curve %d is not initialized, aborting.\n", vals[0].value.i);
		scripter_ast_value_array_free (vals, 3);
		return out;
	}

	/* Is observational data to be plotted?                                   */
	if (vals[1].value.b) {
		lc = phoebe_malloc (sizeof (*lc));
		lc[index] = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, curve-1);
		phoebe_curve_transform (lc[index], PHOEBE_COLUMN_PHASE, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_UNDEFINED);
		alias_phase_points (lc[index]->indep, lc[index]->dep, NULL, -0.6, 0.6);

		props = phoebe_malloc (sizeof (*props));
		props[index].lines = FALSE;
		props[index].ctype = 3;
		props[index].ptype = 13;
		props[index].ltype = 3;
	}

	/* Is synthetic data to be plotted?                                       */
	if (vals[2].value.b) {
		PHOEBE_vector *indep;

		if (vals[1].value.b) index = 1;

		lc = phoebe_realloc (lc, (index+1) * sizeof (*lc));
		lc[index] = phoebe_curve_new ();
		lc[index]->type = PHOEBE_CURVE_LC;

		/*
		 * This may not be the best way to do it, but it suffices for now: we
		 * build a synthetic curve with 300 vertices equidistantly distributed
		 * in phase range [-0.6, 0.6].
		 */

		indep = phoebe_vector_new ();
		phoebe_vector_alloc (indep, 300);
		for (i = 0; i < 300; i++) indep->val[i] = -0.6 + 1.2 * (double) i/299;

		/* Read in synthetic data: */
		status = phoebe_curve_compute (lc[index], indep, vals[0].value.i-1, PHOEBE_COLUMN_PHASE, PHOEBE_COLUMN_FLUX);
		if (status != SUCCESS) {
			phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_array_free (vals, 3);
			return out;
		}

		phoebe_vector_free (indep);

		/* Let's set synthetic curve properties: we want red lines:               */
		props = phoebe_realloc (props, (index+1) * sizeof (*props));
		props[index].lines = TRUE;
		props[index].ctype = 1;
		props[index].ptype = 13;
		props[index].ltype = 1;
	}

	/* Function plot_using_gnuplot () takes arrays of vectors, and we have    */
	/* arrays of curves. We need to construct the former from the latter:     */

	indeps = phoebe_malloc ((index+1) * sizeof (*indeps));
	  deps = phoebe_malloc ((index+1) * sizeof (  *deps));

	for (i = 0; i <= index; i++) {
		indeps[i] = lc[i]->indep;
		  deps[i] = lc[i]->dep;
	}

	/* Everything is set now, let's plot the figure using gnuplot:            */
	plot_using_gnuplot (index+1, NO, indeps, deps, props);

	/* Let's clean everything up:                                             */
	free (indeps);
	free (deps);
	free (props);

	for (i = 0; i <= index; i++) phoebe_curve_free (lc[i]);
	free (lc);

	scripter_ast_value_array_free (vals, 3);

	/* That's all! Have a nice plot! ;)                                         */
	return out;
}

scripter_ast_value scripter_plot_rv_using_gnuplot (scripter_ast_list *args)
{
	/*
	 * This function plots the RV curve that is passed as the first argument.
	 * The plot is not meant to be publication-ready, it is meant only for a
	 * quick inspection. It always plots in phase on the [-0.6, 0.6] range in
	 * km/s units.
	 *
	 * Synopsis:
	 *
	 *   plot_rv_using_gnuplot (curve, obsdata_switch, syndata_switch)
	 */

	int i;
	int status;
	int index = 0;
	int curve, rvno;

	const char *readout_str;

	PHOEBE_curve **rv             = NULL;
	PHOEBE_plot_properties *props = NULL;

	PHOEBE_vector **indeps, **deps;

	scripter_ast_value out;
	scripter_ast_value *vals;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 3, 3, type_int, type_bool, type_bool);
	if (status != SUCCESS) return out;

	/* Is the curve initialized?                                              */
	curve = vals[0].value.i;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);
	if (curve <= 0 || curve > rvno) {
		phoebe_scripter_output ("RV curve %d is not initialized, aborting.\n", vals[0].value.i);
		scripter_ast_value_array_free (vals, 3);
		return out;
	}

	/* Is observational data to be plotted?                                   */
	if (vals[1].value.b) {
		rv = phoebe_malloc (sizeof (*rv));
		rv[index] = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, curve-1);
		phoebe_curve_transform (rv[index], PHOEBE_COLUMN_PHASE, rv[index]->dtype, PHOEBE_COLUMN_UNDEFINED);
		alias_phase_points (rv[index]->indep, rv[index]->dep, NULL, -0.6, 0.6);

		props = phoebe_malloc (sizeof (*props));
		props[index].lines = FALSE;
		props[index].ctype = 3;
		props[index].ptype = 13;
		props[index].ltype = 3;
	}

	/* Is synthetic data to be plotted?                                         */
	if (vals[2].value.b) {
		PHOEBE_vector *indep;
		PHOEBE_column_type dtype;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), vals[0].value.i-1, &readout_str);
		phoebe_column_get_type (&dtype, readout_str);

		if (vals[1].value.b) index = 1;

		rv = phoebe_realloc (rv, (index+1) * sizeof (*rv));
		rv[index] = phoebe_curve_new ();
		rv[index]->type = PHOEBE_CURVE_RV;

		/*
		 * This may not be the best way to do it, but it suffices for now: we
		 * build a synthetic curve with 300 vertices equidistantly distributed
		 * in phase range [-0.6, 0.6].
		 */

		indep = phoebe_vector_new ();
		phoebe_vector_alloc (indep, 300);
		for (i = 0; i < 300; i++) indep->val[i] = -0.6 + 1.2 * (double) i/299;

		/* Read in synthetic data:                                            */
		if (dtype == PHOEBE_COLUMN_PRIMARY_RV)
			status = phoebe_curve_compute (rv[index], indep, vals[0].value.i-1, PHOEBE_COLUMN_PHASE, PHOEBE_COLUMN_PRIMARY_RV);
		else
			status = phoebe_curve_compute (rv[index], indep, vals[0].value.i-1, PHOEBE_COLUMN_PHASE, PHOEBE_COLUMN_SECONDARY_RV);

		if (status != SUCCESS) {
			phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_array_free (vals, 3);
			return out;
		}

		phoebe_vector_free (indep);

		/* Let's set synthetic curve properties: we want red lines:           */
		props = phoebe_realloc (props, (index+1) * sizeof (*props));
		props[index].lines = TRUE;
		props[index].ctype = 1;
		props[index].ptype = 13;
		props[index].ltype = 1;
		}

	/* Function plot_using_gnuplot () takes arrays of vectors, and we have    */
	/* arrays of curves. We need to construct the former from the latter:     */

	indeps = phoebe_malloc ((index+1) * sizeof (*indeps));
	  deps = phoebe_malloc ((index+1) * sizeof (  *deps));

	for (i = 0; i <= index; i++) {
		indeps[i] = rv[i]->indep;
		  deps[i] = rv[i]->dep;
	}

	/* Everything is set now, let's plot the figure using gnuplot:            */
	plot_using_gnuplot (index+1, NO, indeps, deps, props);

	/* Let's clean everything up:                                               */
	free (indeps);
	free (deps);
	free (props);

	for (i = 0; i <= index; i++) phoebe_curve_free (rv[i]);
	free (rv);

	scripter_ast_value_array_free (vals, 3);

	/* That's all! Have a nice plot! ;)                                         */
	return out;
}

scripter_ast_value scripter_plot_spectrum_using_gnuplot (scripter_ast_list *args)
{
	/*
	 * This function plots the spectrum that is passed as the first argument.
	 * The plot is not meant to be publication-ready, it is meant only for a
	 * quick inspection.
	 *
	 * Synopsis:
	 *
	 *   plot_spectrum_using_gnuplot (spectrum [, ll, ul])
	 *
	 * Where:
	 *
	 *   ll  ..  lower wavelength interval limit
	 *   ul  ..  upper wavelength interval limit
	 */

	int i;

	PHOEBE_vector *indep, *dep;
	double ll, ul;
	int dim;

	PHOEBE_plot_properties *props;

	scripter_ast_value out;
	scripter_ast_value *vals;
	int status;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 3, type_spectrum, type_double, type_double);
	if (status != SUCCESS) return out;

	if (vals[1].type != type_void) ll = vals[1].value.d; else ll = vals[0].value.spectrum->data->range[0];
	if (vals[2].type != type_void) ul = vals[2].value.d; else ul = vals[0].value.spectrum->data->range[vals[0].value.spectrum->data->bins];

	if (ll >= ul) {
		phoebe_scripter_output ("lower limit cannot be higher or equal to the upper limit, aborting.\n");
		scripter_ast_value_array_free (vals, 3);
		return out;
	}

	indep = phoebe_vector_new ();
	  dep = phoebe_vector_new ();

	dim = 0;
	for (i = 0; i < vals[0].value.spectrum->data->bins; i++) {
		if (vals[0].value.spectrum->data->range[i] >= ll && vals[0].value.spectrum->data->range[i] <= ul) {
			phoebe_vector_realloc (indep, dim+1);
			phoebe_vector_realloc (  dep, dim+1);
			indep->val[dim] = 0.5 * (vals[0].value.spectrum->data->range[i]+vals[0].value.spectrum->data->range[i+1]);
			  dep->val[dim] = vals[0].value.spectrum->data->val[i];
			dim++;
		}
	}

	props = phoebe_malloc (sizeof (*props));
	props[0].lines = TRUE;
	props[0].ctype = 3;
	props[0].ptype = 13;
	props[0].ltype = 1;

	/* Everything is set now, let's plot the figure using gnuplot:              */
	plot_using_gnuplot (1, FALSE, &indep, &dep, props);

	phoebe_vector_free (indep);
	phoebe_vector_free (dep);
	free (props);
	scripter_ast_value_array_free (vals, 3);

	/* That's all! Have a nice plot! ;)                                         */
	return out;
}

scripter_ast_value scripter_plot_using_gnuplot (scripter_ast_list *args)
{
	/*
	 * This function plots curves of vector pairs (x1,y1), ..., (xn,yn) on
	 * screen. It takes an arbitrary number of vector pairs.
	 *
	 * Synopsis:
	 *
	 *   plot_using_gnuplot (x1, y1, ..., xn, yn)
	 *
	 * Since this function takes an even number of arguments and there is no
	 * generic test for such restrictions (nor there should be), it does its
	 * error-handling by itself.
	 */

	scripter_ast_list *list = args;
	scripter_ast_value out;
	scripter_ast_value val;

	int status;

	PHOEBE_vector        **indeps = NULL;
	PHOEBE_vector          **deps = NULL;
	PHOEBE_plot_properties *props = NULL;

	int dim = 0;

	bool reverse_y;

	out.type = type_void;

	if (!list) {
		phoebe_scripter_output ("plot_using_gnuplot () takes at least 2 arguments, but none are passed.\n");
		return out;
	}

	while (list) {
		dim++;

		val = scripter_ast_evaluate (list->elem);
		if (val.type != type_vector) {
			phoebe_scripter_output ("argument %d is not a vector, aborting.\n", 2*dim-1);
			if (dim > 1) {
				free (indeps);
				free (deps);
			}
			return out;
		}
		indeps = phoebe_realloc (indeps, dim * sizeof (*indeps));
		indeps[dim-1] = val.value.vec;

		list = list->next;
		if (!list) {
			phoebe_scripter_output ("plot_using_gnuplot () takes an even number of parameters, but %d are passed.\n", 2*dim-1);
			free (indeps);
			if (dim > 1) free (deps);
			return out;
		}

		val = scripter_ast_evaluate (list->elem);
		if (val.type != type_vector) {
			phoebe_scripter_output ("argument %d is not a vector, aborting.\n", 2*dim);
			free (indeps);
			if (dim > 1) free (deps);
			return out;
		}

		deps = phoebe_realloc (deps, dim * sizeof (*deps));
		deps[dim-1] = val.value.vec;

		props = phoebe_realloc (props, dim * sizeof (*props));
		props[dim-1].lines = 0;
		props[dim-1].ctype = 0;
		props[dim-1].ltype = 3;
		props[dim-1].ptype = 13;

		list = list->next;
		}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("scripter_ordinate_reversed_switch"), &reverse_y);

	status = plot_using_gnuplot (dim, reverse_y, indeps, deps, props);
	if (status != SUCCESS) phoebe_scripter_output (phoebe_scripter_error (status));

	free (indeps);
	free (deps);
	free (props);

	return out;
}

scripter_ast_value scripter_compute_lc (scripter_ast_list *args)
{
	/*
	 * This function computes a synthetic light curve. It takes an array of
	 * phase points as the first argument and the identifying curve as the
	 * second argument (where it gets all the parameters from).
	 *
	 * Synopsis:
	 *
	 * set dep = compute_lc (indep, curve)
	 */

	int lcno;
	scripter_ast_value out;
	PHOEBE_curve *curve;
	PHOEBE_column_type itype;
	char *readout_str;

	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_vector, type_int);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);

	if (vals[1].value.i < 1 || vals[1].value.i > lcno) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (ERROR_UNINITIALIZED_CURVE));
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	/* Get the independent variable setting (HJD or phase): */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
	status = phoebe_column_get_type (&itype, readout_str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	curve = phoebe_curve_new ();
	status = phoebe_curve_compute (curve, vals[0].value.vec, vals[1].value.i-1, itype, PHOEBE_COLUMN_FLUX);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		phoebe_curve_free (curve);
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	out.type = type_curve;
	out.value.curve = curve;
	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_set_lc_properties (scripter_ast_list *args)
{
	/*
	 * Synopsis:
	 *
	 *  set_lc_properties (curve, passband, indep, dep, weight, sigma, filename)
	 */

	scripter_ast_value out;
	int status;

	scripter_ast_value *vals;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 7, 7, type_int, type_int, type_int, type_int, type_int, type_double, type_string);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	propagate_int_to_menu_item (&(vals[1]), "phoebe_lc_filter");
	status = phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_filter"), vals[0].value.i-1, vals[1].value.str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 7);
		return out;
	}

	propagate_int_to_menu_item (&(vals[2]), "phoebe_lc_indep");
	status = phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_indep"), vals[0].value.i-1, vals[2].value.str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 7);
		return out;
	}

	propagate_int_to_menu_item (&(vals[3]), "phoebe_lc_dep");
	status = phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_dep"), vals[0].value.i-1, vals[3].value.str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 7);
		return out;
	}

	propagate_int_to_menu_item (&(vals[4]), "phoebe_lc_indweight");
	status = phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_indweight"), vals[0].value.i-1, vals[4].value.str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 7);
		return out;
	}

	status = phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_sigma"), vals[0].value.i-1, vals[5].value.d);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 7);
		return out;
	}

	status = phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_filename"), vals[0].value.i-1, vals[6].value.d);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 7);
		return out;
	}

	scripter_ast_value_array_free (vals, 7);
	return out;
}

scripter_ast_value scripter_compute_rv (scripter_ast_list *args)
{
	/*
	 * This function computes a synthetic RV curve. It takes an array of
	 * phase points as the first argument and the identifying curve as the
	 * second argument (where it gets all the parameters from).
	 *
	 * Synopsis:
	 *
	 * set dep = compute_rv (indep, curve)
	 */

	int rvno;
	scripter_ast_value out;
	PHOEBE_curve *curve;

	PHOEBE_column_type itype, dtype;
	char *readout_str;

	scripter_ast_value *vals;
	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_vector, type_int);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	if (vals[1].value.i < 1 || vals[1].value.i > rvno) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (ERROR_UNINITIALIZED_CURVE));
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	/* Get the independent variable setting (HJD or phase): */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
	status = phoebe_column_get_type (&itype, readout_str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	/* Do we plot a primary or a secondary RV curve: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), vals[1].value.i-1, &readout_str);
	status = phoebe_column_get_type (&dtype, readout_str);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	curve = phoebe_curve_new ();
	status = phoebe_curve_compute (curve, vals[0].value.vec, vals[1].value.i-1, itype, dtype);

	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		phoebe_curve_free (curve);
		scripter_ast_value_array_free (vals, 2);
		out.type = type_void;
		return out;
	}

	out.type = type_curve;
	out.value.curve = curve;

	scripter_ast_value_array_free (vals, 2);
	return out;
}

scripter_ast_value scripter_compute_chi2 (scripter_ast_list *args)
{
	/*
	 * This command computes the chi2 value for the passed curve. If the curve
	 * index is omitted, a vector of chi2 values for all passbands is computed.
	 *
	 * Synopsis:
	 *
	 *   set var = compute_chi2 ([curve_idx])
	 */

	scripter_ast_value *vals;
	scripter_ast_value out;

	PHOEBE_vector *chi2s = NULL;

	double chi2;
	int index, lcno, rvno;

	PHOEBE_curve *syncurve;
	PHOEBE_curve *obs;

	int status = scripter_command_args_evaluate (args, &vals, 0, 1, type_int);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	if (vals[0].type != type_void) {
		index = vals[0].value.i;
	}
	else {
		index = 1;
		chi2s = phoebe_vector_new ();
		phoebe_vector_alloc (chi2s, lcno+rvno);
	}

	if (index < 1 || index > lcno + rvno) {
		phoebe_scripter_output ("passband index %d out of range, aborting.\n", index);
		scripter_ast_value_array_free (vals, 1);
		if (chi2s) phoebe_vector_free (chi2s);
		out.type = type_void;
		return out;
	}

	while (index <= lcno + rvno) {
		if (index <= lcno) {
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, index-1);
			phoebe_curve_transform (obs, obs->itype, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_SIGMA);

			/* Synthesize a theoretical curve: */
			syncurve = phoebe_curve_new ();
			phoebe_curve_compute (syncurve, obs->indep, index-1, obs->itype, PHOEBE_COLUMN_FLUX);
		}
		else {
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, index-lcno-1);
			phoebe_curve_transform (obs, obs->itype, obs->dtype, PHOEBE_COLUMN_SIGMA);

			syncurve = phoebe_curve_new ();
			phoebe_curve_compute (syncurve, obs->indep, index-lcno-1, obs->itype, obs->dtype);
		}

		status = phoebe_cf_compute (&chi2, PHOEBE_CF_CHI2, syncurve->dep, obs->dep, obs->weight, 1.0);
		if (status != SUCCESS) {
			phoebe_scripter_output ("%s", phoebe_scripter_error (status));
			scripter_ast_value_array_free (vals, 1);
			if (vals[0].type == type_void) phoebe_vector_free (chi2s);
			phoebe_curve_free (obs);
			phoebe_curve_free (syncurve);
			out.type = type_void;
			return out;
		}

		phoebe_curve_free (obs);
		phoebe_curve_free (syncurve);

		if (vals[0].type != type_void) {
			scripter_ast_value_array_free (vals, 1);
			out.type = type_double;
			out.value.d = chi2;
			return out;
		}
		else {
			chi2s->val[index-1] = chi2;
			index++;
		}
	}

	scripter_ast_value_array_free (vals, 1);
	out.type = type_vector;
	out.value.vec = chi2s;
	return out;
}

scripter_ast_value scripter_get_ld_coefficients (scripter_ast_list *args)
{
	/*
	 * Synopsis:
	 *
	 *   set ldcoeffs = get_ld_coefficients (curve, T, logg, M/H)
	 */

	scripter_ast_value out;

	int lcno;
	double x, y;

	const char *ld_str;
	LDLaw ldlaw;

	const char *passband_str;
	PHOEBE_passband *passband;

	scripter_ast_value *vals;
	int status = scripter_command_args_evaluate (args, &vals, 4, 4, type_int, type_double, type_double, type_double);
	if (status != SUCCESS) {
		out.type = type_void;
		return out;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);

	if (vals[0].value.i < 1 || vals[0].value.i > lcno) {
		phoebe_scripter_output ("curve %d is out of range, aborting.\n", vals[0].value.i);
		scripter_ast_value_array_free (vals, 4);
		out.type = type_void;
		return out;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), vals[0].value.i-1, &passband_str);
	passband = phoebe_passband_lookup (passband_str);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_model"), &ld_str);
	ldlaw = phoebe_ld_model_type (ld_str);

	status = phoebe_get_ld_coefficients (ldlaw, passband, vals[3].value.d, (int) vals[1].value.d, vals[2].value.d, &x, &y);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 4);
		out.type = type_void;
		return out;
	}

	if (ldlaw == LD_LAW_LINEAR) {
		out.type = type_double;
		out.value.d = x;
	}
	else {
		out.type = type_vector;
		out.value.vec = phoebe_vector_new ();
		phoebe_vector_alloc (out.value.vec, 2);
		out.value.vec->val[0] = x;
		out.value.vec->val[1] = y;
	}

	scripter_ast_value_array_free (vals, 4);
	return out;
}

scripter_ast_value scripter_set_spectra_repository (scripter_ast_list *args)
{
	/*
	 * This command sets the spectra repository to the directory name passed
	 * as argument, queries it and outputs the number of found spectra in the
	 * repository.
	 *
	 * Synopsis:
	 *
	 *   set_spectra_repository (dirname)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	PHOEBE_specrep specrep;
	int status;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) return out;

	status = query_spectra_repository (vals[0].value.str, &specrep);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		scripter_ast_value_array_free (vals, 1);
		return out;
	}

	if (PHOEBE_KURUCZ_DIR) free (PHOEBE_KURUCZ_DIR);
	PHOEBE_KURUCZ_DIR = strdup (vals[0].value.str);

	phoebe_scripter_output ("repository %s:\n", PHOEBE_KURUCZ_DIR);
	phoebe_scripter_output ("%d spectra found.\n", specrep.no);

	scripter_ast_value_array_free (vals, 1);
	return out;
}

scripter_ast_value scripter_set_spectrum_properties (scripter_ast_list *args)
{
	/*
	 * This command sets the properties of the SED variable.
	 *
	 * Synopsis:
	 *
	 *   set_spectrum_properties (s, R, Rs)
	 *
	 * where:
	 *
	 *   s   ..  spectrum
	 *   R   ..  true resolving power (R = \lambda/FWHM)
	 *   Rs  ..  sampling power (Rs = \lambda/d\lambda)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	scripter_symbol *var;
	int status;

	out.type = type_void;

	if (args->elem->type != ast_variable) {
		phoebe_scripter_output ("first argument must be a variable, aborting.\n");
		return out;
	}

	status = scripter_command_args_evaluate (args, &vals, 3, 3, type_spectrum, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		return out;
	}

	var = scripter_symbol_lookup (symbol_table, args->elem->value.variable);

	var->link->value.spectrum->R  = vals[1].value.d;
	var->link->value.spectrum->Rs = vals[2].value.d;
	scripter_ast_value_array_free (vals, 3);

	return out;
}

scripter_ast_value scripter_get_spectrum_from_repository (scripter_ast_list *args)
{
	scripter_ast_value out;
	PHOEBE_spectrum *spectrum;

	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 3, 3, type_double, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	status = phoebe_spectrum_new_from_repository (&spectrum, 50000.0, (int) vals[0].value.d, (int) (10.0*vals[1].value.d), (int) vals[2].value.d, 2500, 10500);
	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 3);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 3);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_get_spectrum_from_file (scripter_ast_list *args)
{
	scripter_ast_value out;
	PHOEBE_spectrum *spectrum;

	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 1, 1, type_string);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	spectrum = phoebe_spectrum_new_from_file (vals[0].value.str);
	scripter_ast_value_array_free (vals, 1);

	if (!spectrum) {
		phoebe_scripter_output ("an error occured while opening the file, aborting.\n");
		out.type = type_void;
		return out;
	}

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_apply_doppler_shift (scripter_ast_list *args)
{
	scripter_ast_value out;
	PHOEBE_spectrum *spectrum;

	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_spectrum, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	status = phoebe_spectrum_apply_doppler_shift (&spectrum, vals[0].value.spectrum, vals[1].value.d);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 2);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 2);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_apply_rotational_broadening (scripter_ast_list *args)
{
	scripter_ast_value out;
	PHOEBE_spectrum *spectrum;

	scripter_ast_value *vals;

	int status = scripter_command_args_evaluate (args, &vals, 3, 3, type_spectrum, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	status = phoebe_spectrum_apply_rotational_broadening (&spectrum, vals[0].value.spectrum, vals[1].value.d, vals[2].value.d);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 3);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 3);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_multiply_spectra (scripter_ast_list *args)
{
	scripter_ast_value out;
	scripter_ast_value *vals;
	PHOEBE_spectrum *spectrum;

	double ll, ul, R;

	int status = scripter_command_args_evaluate (args, &vals, 2, 5, type_spectrum, type_spectrum, type_double, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	if (vals[2].type != type_void) ll = vals[2].value.d; else ll = 2500;
	if (vals[3].type != type_void) ul = vals[3].value.d; else ul = 10500;
	if (vals[4].type != type_void)  R = vals[4].value.d; else  R = 50000;

	status = phoebe_spectra_multiply (&spectrum, vals[0].value.spectrum, vals[1].value.spectrum, ll, ul, R);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 5);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 5);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_crop_spectrum (scripter_ast_list *args)
{
	/*
	 * This command crops the spectrum to the passed limits.
	 *
	 * Synopsis:
	 *
	 *   crop_spectrum (spectrum, ll, ul)
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	PHOEBE_spectrum *spectrum;

	int status = scripter_command_args_evaluate (args, &vals, 3, 3, type_spectrum, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	spectrum = phoebe_spectrum_duplicate (vals[0].value.spectrum);
	status = phoebe_spectrum_crop (spectrum, vals[1].value.d, vals[2].value.d);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 3);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 3);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_broaden_spectrum (scripter_ast_list *args)
{
	/*
	 *
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	PHOEBE_spectrum *spectrum;

	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_spectrum, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	status = phoebe_spectrum_broaden (&spectrum, vals[0].value.spectrum, vals[1].value.d);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 2);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 2);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_resample_spectrum (scripter_ast_list *args)
{
	/*
	 * This command resamples the input spectrum.
	 *
	 * Synopsis:
	 *
	 *   set out = resample_spectrum (in, Rs)
	 *
	 * where:
	 *
	 *   in   ..  input (unchanged) spectrum
	 *   Rs   ..  sampling power of the output spectrum
	 */

	scripter_ast_value out;
	scripter_ast_value *vals;
	PHOEBE_spectrum *spectrum;

	int status = scripter_command_args_evaluate (args, &vals, 2, 2, type_spectrum, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	spectrum = phoebe_spectrum_duplicate (vals[0].value.spectrum);
	status = phoebe_spectrum_rebin (&spectrum, spectrum->disp, vals[0].value.spectrum->data->range[0], vals[0].value.spectrum->data->range[vals[0].value.spectrum->data->bins], vals[1].value.d);

	scripter_ast_value_array_free (vals, 2);

	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_merge_spectra (scripter_ast_list *args)
{
	scripter_ast_value out;
	scripter_ast_value *vals;
	PHOEBE_spectrum *spectrum;
	double w1, w2;
	double l1, l2, Rs;

	int status = scripter_command_args_evaluate (args, &vals, 2, 4, type_spectrum, type_spectrum, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	if (vals[2].type != type_void) w1 = vals[2].value.d; else w1 = 0.5;
	if (vals[3].type != type_void) w2 = vals[3].value.d; else w2 = 0.5;

	l1 = vals[0].value.spectrum->data->range[0];
	l2 = vals[0].value.spectrum->data->range[vals[0].value.spectrum->data->bins];
	Rs = vals[0].value.spectrum->Rs;

	status = phoebe_spectra_merge (&spectrum, vals[0].value.spectrum, vals[1].value.spectrum, w1, w2, l1, l2, Rs);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 4);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 4);

	out.type = type_spectrum;
	out.value.spectrum = spectrum;
	return out;
}

scripter_ast_value scripter_integrate_spectrum (scripter_ast_list *args)
{
	scripter_ast_value out;
	scripter_ast_value *vals;
	double ll, ul, result;

	int status = scripter_command_args_evaluate (args, &vals, 1, 3, type_spectrum, type_double, type_double);
	if (status != SUCCESS) {
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	if (vals[1].type != type_void) ll = vals[1].value.d; else ll =  2500.0;
	if (vals[2].type != type_void) ul = vals[2].value.d; else ul = 10500.0;

	status = phoebe_spectrum_integrate (vals[0].value.spectrum, ll, ul, &result);

	if (status != SUCCESS) {
		scripter_ast_value_array_free (vals, 3);
		phoebe_scripter_output ("%s", phoebe_scripter_error (status));
		out.type = type_void;
		return out;
	}

	scripter_ast_value_array_free (vals, 3);

	out.type = type_double;
	out.value.d = result;
	return out;
}
