#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

PHOEBE_parameter_tag *PHOEBE_parameters;
int                   PHOEBE_parameters_no;

int declare_parameter (char *qualifier, char *dependency, char *description, PHOEBE_parameter_kind kind, double min, double max, double step, bool tba, ...)
{
	/*
	 * This function adds a new entry to the PHOEBE_parameter struct
	 * (defined in phoebe_global.h) dynamically and fills it in with values
	 * given as arguments to this function. It should be called *only* by
	 * declare_all_parameters () function.
	 * 
	 * This function handles two types of parameters - native types (all ints,
	 * doubles, char *s, ...) and parameter arrays. In case of the latter, no
	 * initialization is performed, since the array sizes depend on particular
	 * dependency parameters (e.g. LC number, ...).
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	va_list args;
	int i = PHOEBE_parameters_no;

	PHOEBE_parameters_no++;
	PHOEBE_parameters = phoebe_realloc (PHOEBE_parameters, PHOEBE_parameters_no * sizeof (*PHOEBE_parameters));

	PHOEBE_parameters[i].qualifier   = strdup (qualifier);
	PHOEBE_parameters[i].dependency  = strdup (dependency);
	PHOEBE_parameters[i].description = strdup (description);
	PHOEBE_parameters[i].kind        = kind;
	PHOEBE_parameters[i].min         = min;
	PHOEBE_parameters[i].max         = max;
	PHOEBE_parameters[i].step        = step;
	PHOEBE_parameters[i].tba         = tba;

	va_start (args, tba);
	PHOEBE_parameters[i].type        = va_arg (args, PHOEBE_type);

	switch (PHOEBE_parameters[i].type) {
		case TYPE_INT:
			PHOEBE_parameters[i].value.i = va_arg (args, int);
		break;
		case TYPE_DOUBLE:
			PHOEBE_parameters[i].value.d = va_arg (args, double);
		break;
		case TYPE_BOOL:
			PHOEBE_parameters[i].value.b = va_arg (args, bool);
		break;
		case TYPE_STRING: {
			char *str = va_arg (args, char *);
			PHOEBE_parameters[i].value.str = phoebe_malloc (strlen (str) + 1);
			strcpy (PHOEBE_parameters[i].value.str, str);
		}
		break;
		case TYPE_INT_ARRAY:
			PHOEBE_parameters[i].value.iarray = NULL;
			PHOEBE_parameters[i].defaultvalue.i = va_arg (args, int);
		break;
		case TYPE_DOUBLE_ARRAY:
			PHOEBE_parameters[i].value.vec = phoebe_vector_new ();
			PHOEBE_parameters[i].defaultvalue.d = va_arg (args, double);
		break;
		case TYPE_BOOL_ARRAY:
			PHOEBE_parameters[i].value.barray = NULL;
			PHOEBE_parameters[i].defaultvalue.b = va_arg (args, bool);
		break;
		case TYPE_STRING_ARRAY: {
			char *str = va_arg (args, char *);
			PHOEBE_parameters[i].value.array = phoebe_array_new (TYPE_STRING_ARRAY);
			PHOEBE_parameters[i].defaultvalue.str = strdup (str);
		}
		break;
	}
	va_end (args);

	if (kind != KIND_MENU) PHOEBE_parameters[i].menu = NULL;
	else {
		PHOEBE_parameters[i].menu = phoebe_malloc (sizeof (*PHOEBE_parameters[i].menu));
		PHOEBE_parameters[i].menu->optno  = 0;
		PHOEBE_parameters[i].menu->option = NULL;
	}

	return SUCCESS;
}

int release_parameter_by_index (int index) 
{
	/*
	 * This functions frees memory occupied by the qualifier with index 'index'.
	 *
	 * Return values:
	 *
	 *   ERROR_RELEASE_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	if (index >= PHOEBE_parameters_no) {
		phoebe_lib_error ("index %d out of range [0, %d] in release_parameter ()\n", index, PHOEBE_parameters_no-1);
		return ERROR_RELEASE_INDEX_OUT_OF_RANGE;
	}

	free (PHOEBE_parameters[index].qualifier);
	free (PHOEBE_parameters[index].description);
	free (PHOEBE_parameters[index].dependency);

	switch (PHOEBE_parameters[index].type) {
		case TYPE_INT:
			/* Nothing to be freed */
		break;
		case TYPE_BOOL:
			/* Nothing to be freed */
		break;
		case TYPE_DOUBLE:
			/* Nothing to be freed */
		break;
		case TYPE_STRING:
			free (PHOEBE_parameters[index].value.str);
		break;
		case TYPE_INT_ARRAY:
			if (PHOEBE_parameters[index].value.iarray)
				free (PHOEBE_parameters[index].value.iarray);
		break;
		case TYPE_BOOL_ARRAY:
			if (PHOEBE_parameters[index].value.barray)
				free (PHOEBE_parameters[index].value.barray);
		break;
		case TYPE_DOUBLE_ARRAY:
			phoebe_vector_free (PHOEBE_parameters[index].value.vec);
		break;
		case TYPE_STRING_ARRAY:
			phoebe_array_free (PHOEBE_parameters[index].value.array);
			free (PHOEBE_parameters[index].defaultvalue.str);
		break;
		default:
			phoebe_lib_error ("exception handler invoked in release_parameter_by_index (), please report this!\n");
		break;
	}

	return SUCCESS;
}

int release_parameter_by_qualifier (char *qualifier)
{
	/*
	 * This function is a front-end to release_parameter_by_index (). Instead
	 * of taking the index, it takes a qualifier. It is slower because it first
	 * has to lookup the qualifier, but it is more convenient for external qua-
	 * lifiers.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_RELEASE_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;
	return release_parameter_by_index (index);
}

int update_parameter_arrays (char *dependency, int oldval)
{
	/*
	 * This function is called whenever the dimension of parameter arrays must
	 * be changed. Typically this happens when the number of observed data cur-
	 * ves is changed, the number of spots is changed etc.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status, dim;
	int i, j;
	PHOEBE_type type;

	status = phoebe_index_from_qualifier (&index, dependency);
	if (status != SUCCESS) return status;

	dim = PHOEBE_parameters[index].value.i;

	/* If the dimension is the same, there's nothing to be done. */
	if (oldval == dim) return SUCCESS;

	for (i = 0; i < PHOEBE_parameters_no; i++)
		if (strcmp (PHOEBE_parameters[i].dependency, dependency) == 0) {
			phoebe_debug ("resizing array %s to dim %d\n", PHOEBE_parameters[i].qualifier, dim);
			type = PHOEBE_parameters[i].type;
			switch (type) {
				case TYPE_INT_ARRAY:
					PHOEBE_parameters[i].value.iarray = phoebe_realloc (PHOEBE_parameters[i].value.iarray, dim * sizeof (int));
					for (j = oldval; j < dim; j++)
						PHOEBE_parameters[i].value.iarray[j] = PHOEBE_parameters[i].defaultvalue.i;
				break;
				case TYPE_BOOL_ARRAY:
					PHOEBE_parameters[i].value.barray = phoebe_realloc (PHOEBE_parameters[i].value.barray, dim * sizeof (bool));
					for (j = oldval; j < dim; j++)
						PHOEBE_parameters[i].value.barray[j] = PHOEBE_parameters[i].defaultvalue.b;
				break;
				case TYPE_DOUBLE_ARRAY:
					phoebe_vector_realloc (PHOEBE_parameters[i].value.vec, dim);
					for (j = oldval; j < dim; j++)
						PHOEBE_parameters[i].value.vec->val[j] = PHOEBE_parameters[i].defaultvalue.d;
				break;
				case TYPE_STRING_ARRAY:
					phoebe_array_realloc (PHOEBE_parameters[i].value.array, dim);
					for (j = oldval; j < dim; j++)
						PHOEBE_parameters[i].value.array->val.strarray[j] = strdup (PHOEBE_parameters[i].defaultvalue.str);
				break;
			}
		}

	return SUCCESS;
}

bool phoebe_parameter_menu_option_is_valid (char *qualifier, char *option)
{
	/*
	 * This function is a boolean test for parameter menu options. It returns
	 * TRUE if the option is valid and FALSE if it is invalid.
	 */

	int i, index;

	/* Is the qualifier valid: */
	if (phoebe_index_from_qualifier (&index, qualifier) != SUCCESS) return FALSE;

	/* Is the qualified parameter a menu: */
	if (PHOEBE_parameters[index].kind != KIND_MENU) return FALSE;

	/* Is the option one of the menu options: */
	for (i = 0; i < PHOEBE_parameters[index].menu->optno; i++)
		if (strcmp (PHOEBE_parameters[index].menu->option[i], option) == 0)
			return TRUE;

	return FALSE;
}

int add_option_to_parameter_menu (char *qualifier, char *option)
{
	/*
	 * This function adds an option 'option' to the parameter menu of the
	 * passed qualifier. The qualifier's kind must be KIND_MENU, otherwise
	 * the function will abort.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_PARAMETER_KIND_NOT_MENU
	 *   SUCCESS
	 */

	PHOEBE_parameter_kind kind;
	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	status = phoebe_kind_from_qualifier (&kind, qualifier);
	if (status != SUCCESS) return status;

	if (kind != KIND_MENU) {
		phoebe_lib_error ("qualifier %s kind is not a menu, aborting.\n", qualifier);
		return ERROR_PARAMETER_KIND_NOT_MENU;
	}

	PHOEBE_parameters[index].menu->optno++;
	PHOEBE_parameters[index].menu->option = phoebe_realloc (PHOEBE_parameters[index].menu->option, PHOEBE_parameters[index].menu->optno * sizeof (*(PHOEBE_parameters[index].menu->option)));
	PHOEBE_parameters[index].menu->option[PHOEBE_parameters[index].menu->optno-1] = strdup (option);

	return SUCCESS;
}

int declare_all_parameters ()
{
	/*
	 * This (and only this) function holds all parameters that are accessible
	 * to PHOEBE; there is no GUI connection or any other plug-in connection in
	 * this function, only native PHOEBE parameters.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	/* **********************   Model parameters   ************************** */

	declare_parameter ("phoebe_name",                        "",                 "Star name",                                  KIND_PARAMETER,    0.0,    0.0,    0.0,  NO, TYPE_STRING,       "");
	declare_parameter ("phoebe_indep",                       "",                 "Independent variable",                       KIND_MENU,         0.0,    0.0,    0.0,  NO, TYPE_STRING,       "Phase");
	declare_parameter ("phoebe_model",                       "",                 "The model (morphological constraint)",       KIND_MENU,         0.0,    0.0,    0.0,  NO, TYPE_STRING,       "Unconstrained binary system");

	declare_parameter ("phoebe_lcno",                        "dependency",       "Number of observed light curves",            KIND_MODIFIER,     0.0,    0.0,    0.0,  NO, TYPE_INT,          0);
	declare_parameter ("phoebe_rvno",                        "dependency",       "Number of observed RV curves",               KIND_MODIFIER,     0.0,    0.0,    0.0,  NO, TYPE_INT,          0);
	declare_parameter ("phoebe_spno",                        "dependency",       "Number of observed spectra",                 KIND_MODIFIER,     0.0,    0.0,    0.0,  NO, TYPE_INT,          0);

	/* **********************   Model constraints   ************************* */

	declare_parameter ("phoebe_asini_switch",                "",                 "(a sin i) is kept constant",                 KIND_SWITCH,       0.0,    0.0,    0.0,  NO, TYPE_BOOL,         NO);
	declare_parameter ("phoebe_asini",                       "",                 "(a sin i) constant",                         KIND_PARAMETER,    0.0,   1E10,    0.0,  NO, TYPE_DOUBLE,       10.0);

	declare_parameter ("phoebe_cindex_switch",               "",                 "Use the color-index constraint",             KIND_SWITCH,       0.0,    0.0,    0.0,  NO, TYPE_BOOL,         NO);
	declare_parameter ("phoebe_cindex",                      "phoebe_lcno",      "Color-index values",                         KIND_PARAMETER,   -100,    100,   1e-2,  NO, TYPE_DOUBLE_ARRAY, 1.0);

	declare_parameter ("phoebe_msc1_switch",                 "",                 "Main-sequence constraint for star 1",        KIND_PARAMETER,      0,      0,      0,  NO, TYPE_BOOL,         NO);
	declare_parameter ("phoebe_msc2_switch",                 "",                 "Main-sequence constraint for star 2",        KIND_PARAMETER,      0,      0,      0,  NO, TYPE_BOOL,         NO);

	/* ***********************   Data parameters   ************************** */

	declare_parameter ("phoebe_lc_filename",                 "phoebe_lcno",      "Observed LC data filename",                  KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	declare_parameter ("phoebe_lc_sigma",                    "phoebe_lcno",      "Observed LC data standard deviation",        KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_DOUBLE_ARRAY, 0.01);
	declare_parameter ("phoebe_lc_filter",                   "phoebe_lcno",      "Observed LC data filter",                    KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "440nm (B)");
	declare_parameter ("phoebe_lc_indep",                    "phoebe_lcno",      "Observed LC data independent variable",      KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Time (HJD)");
	declare_parameter ("phoebe_lc_dep",                      "phoebe_lcno",      "Observed LC data dependent variable",        KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Magnitude");
	declare_parameter ("phoebe_lc_indweight",                "phoebe_lcno",      "Observed LC data individual weighting",      KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Standard deviation");
	declare_parameter ("phoebe_lc_levweight",                "phoebe_lcno",      "Observed LC data level weighting",           KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Poissonian scatter");
	declare_parameter ("phoebe_lc_active",                   "phoebe_lcno",      "Observed LC data is used",                   KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL_ARRAY,    YES);

	declare_parameter ("phoebe_rv_filename",                 "phoebe_rvno",      "Observed RV data filename",                  KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	declare_parameter ("phoebe_rv_sigma",                    "phoebe_rvno",      "Observed RV data standard deviation",        KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_DOUBLE_ARRAY, 1.0);
	declare_parameter ("phoebe_rv_filter",                   "phoebe_rvno",      "Observed RV data filter",                    KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "550nm (V)");
	declare_parameter ("phoebe_rv_indep",                    "phoebe_rvno",      "Observed RV data independent variable",      KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Time (HJD)");
	declare_parameter ("phoebe_rv_dep",                      "phoebe_rvno",      "Observed RV data dependent variable",        KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Primary RV");
	declare_parameter ("phoebe_rv_indweight",                "phoebe_rvno",      "Observed RV data individual weighting",      KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Standard deviation");
	declare_parameter ("phoebe_rv_active",                   "phoebe_rvno",      "Observed RV data is used",                   KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL_ARRAY,    YES);

	declare_parameter ("phoebe_mnorm",                       "",                 "Flux-normalizing magnitude",                 KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_DOUBLE,       10.0);

	declare_parameter ("phoebe_bins_switch",                 "",                 "Data binning",                               KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	declare_parameter ("phoebe_bins",                        "",                 "Number of bins",                             KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_INT,           100);

	declare_parameter ("phoebe_ie_switch",                   "",                 "Interstellar extinction (reddening)",        KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	declare_parameter ("phoebe_ie_factor",                   "",                 "Interstellar extinction coefficient",        KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_DOUBLE,        3.1);
	declare_parameter ("phoebe_ie_excess",                   "",                 "Interstellar extinction color excess value", KIND_PARAMETER,    0.0,    0.0,    0.0, NO, TYPE_DOUBLE,        0.0);

	declare_parameter ("phoebe_proximity_rv1_switch",        "",                 "Proximity effects for primary star RV",      KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	declare_parameter ("phoebe_proximity_rv2_switch",        "",                 "Proximity effects for secondary star RV",    KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);

	/* **********************   System parameters   ************************* */

	declare_parameter ("phoebe_hjd0",                        "",                 "Origin of HJD time",                         KIND_ADJUSTABLE, -1E10,   1E10, 0.0001, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_period",                      "",                 "Orbital period in days",                     KIND_ADJUSTABLE,   0.0,   1E10, 0.0001, NO, TYPE_DOUBLE,        1.0);
	declare_parameter ("phoebe_dpdt",                        "",                 "First time derivative of period (days/day)", KIND_ADJUSTABLE,  -1.0,    1.0,   1E-6, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_pshift",                      "",                 "Phase shift",                                KIND_ADJUSTABLE,  -0.5,    0.5,   0.01, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_sma",                         "",                 "Semi-major axis in solar radii",             KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,       10.0);
	declare_parameter ("phoebe_rm",                          "",                 "Mass ratio (secondary over primary)",        KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,        1.0);
	declare_parameter ("phoebe_incl",                        "",                 "Inclination in degrees",                     KIND_ADJUSTABLE,   0.0,  180.0,   0.01, NO, TYPE_DOUBLE,       80.0);
	declare_parameter ("phoebe_vga",                         "",                 "Center-of-mass velocity in km/s",            KIND_ADJUSTABLE,  -1E3,    1E3,    1.0, NO, TYPE_DOUBLE,        0.0);

	/* ********************   Component parameters   ************************ */

	declare_parameter ("phoebe_teff1",                       "",                 "Primary star effective temperature in K",    KIND_ADJUSTABLE,  3500,  50000,     10, NO, TYPE_DOUBLE,     6000.0);
	declare_parameter ("phoebe_teff2",                       "",                 "Secondary star effective temperature in K",  KIND_ADJUSTABLE,  3500,  50000,     10, NO, TYPE_DOUBLE,     6000.0);
	declare_parameter ("phoebe_pot1",                        "",                 "Primary star surface potential",             KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,       10.0);
	declare_parameter ("phoebe_pot2",                        "",                 "Secondary star surface potential",           KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,       10.0);
	declare_parameter ("phoebe_logg1",                       "",                 "Primary star surface potential",             KIND_ADJUSTABLE,   0.0,   10.0,   0.01, NO, TYPE_DOUBLE,        4.3);
	declare_parameter ("phoebe_logg2",                       "",                 "Primary star surface potential",             KIND_ADJUSTABLE,   0.0,   10.0,   0.01, NO, TYPE_DOUBLE,        4.3);
	declare_parameter ("phoebe_met1",                        "",                 "Primary star metallicity",                   KIND_ADJUSTABLE, -10.0,   10.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_met2",                        "",                 "Secondary star metallicity",                 KIND_ADJUSTABLE, -10.0,   10.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_f1",                          "",                 "Primary star synchronicity parameter",       KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        1.0);
	declare_parameter ("phoebe_f2",                          "",                 "Secondary star synchronicity parameter",     KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        1.0);
	declare_parameter ("phoebe_alb1",                        "",                 "Primary star surface albedo",                KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.6);
	declare_parameter ("phoebe_alb2",                        "",                 "Secondary star surface albedo",              KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.6);
	declare_parameter ("phoebe_grb1",                        "",                 "Primary star gravity brightening",           KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.32);
	declare_parameter ("phoebe_grb2",                        "",                 "Primary star gravity brightening",           KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.32);

	/* **********************   Orbit parameters   ************************** */

	declare_parameter ("phoebe_ecc",                         "",                 "Orbital eccentricity",                       KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_perr0",                       "",                 "Argument of periastron",                     KIND_ADJUSTABLE,   0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE,        0.0);
	declare_parameter ("phoebe_dperdt",                      "",                 "First time derivative of periastron",        KIND_ADJUSTABLE,  -1.0,    1.0,   1E-6, NO, TYPE_DOUBLE,        0.0);

	/* *********************   Surface parameters   ************************* */

	declare_parameter ("phoebe_hla",                         "phoebe_lcno",      "LC primary star flux leveler",               KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY, 12.566371);
	declare_parameter ("phoebe_cla",                         "phoebe_lcno",      "LC secondary star flux leveler",             KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY, 12.566371);
	declare_parameter ("phoebe_opsf",                        "phoebe_lcno",      "Third light contribution",                   KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);

	declare_parameter ("phoebe_passband_treatment_mode",     "",                 "Passband treatment mode",                    KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING,        "Interpolation");
	declare_parameter ("phoebe_atm1_switch",                 "",                 "Use Kurucz's models for primary star",       KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	declare_parameter ("phoebe_atm2_switch",                 "",                 "Use Kurucz's models for secondary star",     KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	declare_parameter ("phoebe_reffect_switch",              "",                 "Detailed reflection effect",                 KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	declare_parameter ("phoebe_reffect_reflections",         "",                 "Number of detailed reflections",             KIND_PARAMETER,      2,     10,      1, NO, TYPE_INT,             2);

	declare_parameter ("phoebe_usecla_switch",               "",                 "Decouple CLAs from temperature",             KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);

	/* ********************   Extrinsic parameters   ************************ */

	declare_parameter ("phoebe_el3_units",                   "",                 "Units of third light",                       KIND_MENU,         0.0,    0.0,    0.0, NO, TYPE_STRING,        "Total light");
	declare_parameter ("phoebe_el3",                         "phoebe_lcno",      "Third light contribution",                   KIND_ADJUSTABLE,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);
	declare_parameter ("phoebe_extinction",                  "phoebe_lcno",      "Interstellar extinction coefficient",        KIND_ADJUSTABLE,   0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);

	/* *********************   Fitting parameters   ************************* */

	declare_parameter ("phoebe_grid_finesize1",              "",                 "Fine grid size on primary star",             KIND_PARAMETER,      5,     60,      1, NO, TYPE_INT,            20);
	declare_parameter ("phoebe_grid_finesize2",              "",                 "Fine grid size on secondary star",           KIND_PARAMETER,      5,     60,      1, NO, TYPE_INT,            20);
	declare_parameter ("phoebe_grid_coarsesize1",            "",                 "Coarse grid size on primary star",           KIND_PARAMETER,      5,     60,      1, NO, TYPE_INT,             5);
	declare_parameter ("phoebe_grid_coarsesize2",            "",                 "Coarse grid size on secondary star",         KIND_PARAMETER,      5,     60,      1, NO, TYPE_INT,             5);

	declare_parameter ("phoebe_compute_hla_switch",          "",                 "Compute passband (HLA) levels",              KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	declare_parameter ("phoebe_compute_vga_switch",          "",                 "Compute gamma velocity",                     KIND_SWITCH,       0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);

	/* **********************   DC fit parameters   ************************* */

	declare_parameter ("phoebe_dc_symder_switch",            "",                 "Should symmetrical DC derivatives be used",  KIND_SWITCH,         0,      0,      0, NO, TYPE_BOOL,          YES);
	declare_parameter ("phoebe_dc_lambda",                   "",                 "Levenberg-Marquardt multiplier for DC",      KIND_PARAMETER,    0.0,    1.0,   1e-3, NO, TYPE_DOUBLE,       1e-3);

	declare_parameter ("phoebe_dc_spot1src",                 "",                 "Adjusted spot 1 source (at which star is the spot)", KIND_PARAMETER, 1,   2, 1, NO, TYPE_INT, 1);
	declare_parameter ("phoebe_dc_spot2src",                 "",                 "Adjusted spot 2 source (at which star is the spot)", KIND_PARAMETER, 1,   2, 1, NO, TYPE_INT, 2);
	declare_parameter ("phoebe_dc_spot1id",                  "",                 "Adjusted spot 1 id (which spot is to be adjusted)",  KIND_PARAMETER, 1, 100, 1, NO, TYPE_INT, 1);
	declare_parameter ("phoebe_dc_spot2id",                  "",                 "Adjusted spot 2 id (which spot is to be adjusted)",  KIND_PARAMETER, 1, 100, 1, NO, TYPE_INT, 1);

	/* *******************   Perturbations parameters   ********************* */

	declare_parameter ("phoebe_ld_model",                    "",                 "Limb darkening model",                       KIND_MENU,           0,      0,      0, NO, TYPE_STRING, "Logarithmic law");
	declare_parameter ("phoebe_ld_xbol1",                    "",                 "Primary star bolometric LD coefficient x",   KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	declare_parameter ("phoebe_ld_ybol1",                    "",                 "Secondary star bolometric LD coefficient x", KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	declare_parameter ("phoebe_ld_xbol2",                    "",                 "Primary star bolometric LD coefficient y",   KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	declare_parameter ("phoebe_ld_ybol2",                    "",                 "Secondary star bolometric LD coefficient y", KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	declare_parameter ("phoebe_ld_lcx1",                     "phoebe_lcno",      "Primary star bandpass LD coefficient x",     KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_lcx2",                     "phoebe_lcno",      "Secondary star bandpass LD coefficient x",   KIND_ADJUSTABLE,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_lcy1",                     "phoebe_lcno",      "Primary star bandpass LD coefficient y",     KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_lcy2",                     "phoebe_lcno",      "Secondary star bandpass LD coefficient y",   KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_rvx1",                     "phoebe_rvno",      "Primary RV bandpass LD coefficient x",       KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_rvx2",                     "phoebe_rvno",      "Secondary RV bandpass LD coefficient x",     KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_rvy1",                     "phoebe_rvno",      "Primary RV bandpass LD coefficient y",       KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	declare_parameter ("phoebe_ld_rvy2",                     "phoebe_rvno",      "Secondary RV bandpass LD coefficient y",     KIND_PARAMETER,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);

	declare_parameter ("phoebe_spots_no1",                   "dependency",       "Number of spots on primary star",            KIND_PARAMETER,      0,    100,      1, NO, TYPE_INT,            0);
	declare_parameter ("phoebe_spots_no2",                   "dependency",       "Number of spots on secondary star",          KIND_PARAMETER,      0,    100,      1, NO, TYPE_INT,            0);
	declare_parameter ("phoebe_spots_move1",                 "",                 "Spots on primary star move in longitude",    KIND_SWITCH,         0,      0,      0, NO, TYPE_BOOL,         YES);
	declare_parameter ("phoebe_spots_move2",                 "",                 "Spots on secondary star move in longitude",  KIND_SWITCH,         0,      0,      0, NO, TYPE_BOOL,         YES);
	declare_parameter ("phoebe_spots_lat1",                  "phoebe_spots_no1", "Latitude of the spot on primary star",       KIND_PARAMETER,    0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	declare_parameter ("phoebe_spots_long1",                 "phoebe_spots_no1", "Longitude of the spot on primary star",      KIND_PARAMETER,    0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	declare_parameter ("phoebe_spots_rad1",                  "phoebe_spots_no1", "Radius of the spot on primary star",         KIND_PARAMETER,    0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.2);
	declare_parameter ("phoebe_spots_temp1",                 "phoebe_spots_no1", "Temperature of the spot on primary star",    KIND_PARAMETER,    0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.9);
	declare_parameter ("phoebe_spots_lat2",                  "phoebe_spots_no2", "Latitude of the spot on secondary star",     KIND_PARAMETER,    0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	declare_parameter ("phoebe_spots_long2",                 "phoebe_spots_no2", "Longitude of the spot on secondary star",    KIND_PARAMETER,    0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	declare_parameter ("phoebe_spots_rad2",                  "phoebe_spots_no2", "Radius of the spot on secondary star",       KIND_PARAMETER,    0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.2);
	declare_parameter ("phoebe_spots_temp2",                 "phoebe_spots_no2", "Temperature of the spot on secondary star",  KIND_PARAMETER,    0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.9);

	/* *********************   Utilities parameters   *********************** */

	declare_parameter ("phoebe_synscatter_switch",           "",                 "Synthetic scatter",                          KIND_SWITCH,         0,      0,      0, NO, TYPE_BOOL,          NO);
	declare_parameter ("phoebe_synscatter_sigma",            "",                 "Synthetic scatter standard deviation",       KIND_PARAMETER,    0.0,  100.0,   0.01, NO, TYPE_DOUBLE,      0.01);
	declare_parameter ("phoebe_synscatter_seed",             "",                 "Synthetic scatter seed",                     KIND_PARAMETER,    1E8,    1E9,      1, NO, TYPE_DOUBLE,     1.5E8);
	declare_parameter ("phoebe_synscatter_levweight",        "",                 "Synthetic scatter weighting",                KIND_MENU,           0,      0,      0, NO, TYPE_STRING, "Poissonian scatter");

	/* *********************   Scripter parameters   ************************ */

	declare_parameter ("scripter_verbosity_level",           "",                 "Scripter verbosity level",                   KIND_PARAMETER,      0,      1,      1, NO, TYPE_INT,            1);

	/* *********************   Plotting parameters   ************************ */

	declare_parameter ("plot_reversey_switch",               "",                 "Reverse ordinate (Y) axis direction",        KIND_SWITCH,         0,      0,      0, NO, TYPE_BOOL,      NO);

	/* *****************   Not yet implemented parameters   ***************** */
/*
	declare_parameter ("phoebe_synthetic_reddening",         "",                 "Synthetic interstellar reddening",           KIND_SWITCH,         0,      0,      0, NO, TYPE_BOOL,          NO);
	declare_parameter ("phoebe_synthetic_reddening_factor",  "",                 "Synthetic interstellar reddening factor",    KIND_PARAMETER,      0,     10,   0.01, NO, TYPE_DOUBLE,       3.1);
	declare_parameter ("phoebe_synthetic_reddening_excess",  "",                 "Synthetic interstellar reddening excess",    KIND_PARAMETER,      0,     10,   0.01, NO, TYPE_DOUBLE,       0.0);
*/
	return SUCCESS;
}

int add_options_to_all_parameters ()
{
	/*
	 * This function adds options to all KIND_MENU parameters. In principle
	 * all calls to add_option_to_parameter_menu () function should be checked
	 * for return value, but since the function issues a warning in case a
	 * qualifier does not contain a menu, it is not really necessary.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	/* Parameter: phoebe_indep                                                */
	add_option_to_parameter_menu ("phoebe_indep", "Time (HJD)");
	add_option_to_parameter_menu ("phoebe_indep", "Phase");

	/* Parameter: phoebe_model                                                */
	add_option_to_parameter_menu ("phoebe_model", "X-ray binary");
	add_option_to_parameter_menu ("phoebe_model", "Unconstrained binary system");
	add_option_to_parameter_menu ("phoebe_model", "Overcontact binary of the W UMa type");
	add_option_to_parameter_menu ("phoebe_model", "Detached binary");
	add_option_to_parameter_menu ("phoebe_model", "Overcontact binary not in thermal contact");
	add_option_to_parameter_menu ("phoebe_model", "Semi-detached binary, primary star fills Roche lobe");
	add_option_to_parameter_menu ("phoebe_model", "Semi-detached binary, secondary star fills Roche lobe");
	add_option_to_parameter_menu ("phoebe_model", "Double contact binary");

	/* Parameter: phoebe_lc_filter                                            */

	{
	char *passband_str;
	for (i = 0; i < PHOEBE_passbands_no; i++) {
		passband_str = concatenate_strings (PHOEBE_passbands[i]->set, ":", PHOEBE_passbands[i]->name, NULL);
		add_option_to_parameter_menu ("phoebe_lc_filter", passband_str);
		free (passband_str);
	}
	}
/*
	add_option_to_parameter_menu ("phoebe_lc_filter", "350nm (u)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "411nm (v)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "467nm (b)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "547nm (y)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "360nm (U)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "440nm (B)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "550nm (V)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "700nm (R)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "900nm (I)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "1250nm (J)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "2200nm (K)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "3400nm (L)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "5000nm (M)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "10200nm (N)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "647nm (Rc)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "786nm (Ic)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "419nm (Bt)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "523nm (Vt)");
	add_option_to_parameter_menu ("phoebe_lc_filter", "505nm (Hp)");
*/
	/* Parameter: phoebe_lc_indep                                             */
	add_option_to_parameter_menu ("phoebe_lc_indep", "Time (HJD)");
	add_option_to_parameter_menu ("phoebe_lc_indep", "Phase");

	/* Parameter: phoebe_lc_dep                                               */
	add_option_to_parameter_menu ("phoebe_lc_dep", "Magnitude");
	add_option_to_parameter_menu ("phoebe_lc_dep", "Flux");

	/* Parameter: phoebe_lc_indweight                                         */
	add_option_to_parameter_menu ("phoebe_lc_indweight", "Standard weight");
	add_option_to_parameter_menu ("phoebe_lc_indweight", "Standard deviation");
	add_option_to_parameter_menu ("phoebe_lc_indweight", "Unavailable");

	/* Parameter: phoebe_lc_levweight                                         */
	add_option_to_parameter_menu ("phoebe_lc_levweight", "No level-dependent weighting");
	add_option_to_parameter_menu ("phoebe_lc_levweight", "Poissonian scatter");
	add_option_to_parameter_menu ("phoebe_lc_levweight", "Low light scatter");

	/* Parameter: phoebe_rv_filter                                            */
	{
	char *passband_str;
	for (i = 0; i < PHOEBE_passbands_no; i++) {
		passband_str = concatenate_strings (PHOEBE_passbands[i]->set, ":", PHOEBE_passbands[i]->name, NULL);
		add_option_to_parameter_menu ("phoebe_rv_filter", passband_str);
		free (passband_str);
	}
	}
/*
	add_option_to_parameter_menu ("phoebe_rv_filter", "350nm (u)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "411nm (v)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "467nm (b)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "547nm (y)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "360nm (U)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "440nm (B)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "550nm (V)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "700nm (R)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "900nm (I)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "1250nm (J)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "2200nm (K)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "3400nm (L)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "5000nm (M)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "10200nm (N)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "647nm (Rc)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "786nm (Ic)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "419nm (Bt)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "523nm (Vt)");
	add_option_to_parameter_menu ("phoebe_rv_filter", "505nm (Hp)");
*/
	/* Parameter: phoebe_rv_indep                                             */
	add_option_to_parameter_menu ("phoebe_rv_indep", "Time (HJD)");
	add_option_to_parameter_menu ("phoebe_rv_indep", "Phase");

	/* Parameter: phoebe_rv_dep                                               */
	add_option_to_parameter_menu ("phoebe_rv_dep", "Primary RV");
	add_option_to_parameter_menu ("phoebe_rv_dep", "Secondary RV");

	/* Parameter: phoebe_rv_indweight                                         */
	add_option_to_parameter_menu ("phoebe_rv_indweight", "Standard weight");
	add_option_to_parameter_menu ("phoebe_rv_indweight", "Standard deviation");
	add_option_to_parameter_menu ("phoebe_rv_indweight", "Unavailable");

	/* Parameter: phoebe_ld_model                                             */
	add_option_to_parameter_menu ("phoebe_ld_model",     "Linear cosine law");
	add_option_to_parameter_menu ("phoebe_ld_model",     "Logarithmic law");
	add_option_to_parameter_menu ("phoebe_ld_model",     "Square root law");

	/* Parameter: phoebe_synscatter_levweight                                 */
	add_option_to_parameter_menu ("phoebe_synscatter_levweight", "No level-dependent weighting");
	add_option_to_parameter_menu ("phoebe_synscatter_levweight", "Poissonian scatter");
	add_option_to_parameter_menu ("phoebe_synscatter_levweight", "Low light scatter");

	/* Parameter: phoebe_passband_treatment_mode                              */
	add_option_to_parameter_menu ("phoebe_passband_treatment_mode", "None");
	add_option_to_parameter_menu ("phoebe_passband_treatment_mode", "Interpolation");
	add_option_to_parameter_menu ("phoebe_passband_treatment_mode", "Rigorous");

	/* Parameter: phoebe_el3_units                                            */
	add_option_to_parameter_menu ("phoebe_el3_units", "Total light");
	add_option_to_parameter_menu ("phoebe_el3_units", "Flux");

	return SUCCESS;
}

int release_all_parameter_options ()
{
	/*
	 * This function recurses through all parameters, identifies the ones that
	 * contain the menu and releases all menu items. It should be called upon
	 * exit from PHOEBE.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i, j;

	for (i = 0; i < PHOEBE_parameters_no; i++) {
		if (PHOEBE_parameters[i].kind == KIND_MENU) {
			for (j = 0; j < PHOEBE_parameters[i].menu->optno; j++)
				free (PHOEBE_parameters[i].menu->option[j]);
			free (PHOEBE_parameters[i].menu->option);
			free (PHOEBE_parameters[i].menu);
		}
	}

	return SUCCESS;
}

int release_all_parameters ()
{
	/*
	 * This function releases (frees) all parameters that have been defined by
	 * declare_all_parameters () and any individual calls to declare_parameter
	 * () function. It is meant to be called for a clean exit from PHOEBE.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	for (i = 0; i < PHOEBE_parameters_no; i++)
		release_parameter_by_index (i);

	return SUCCESS;
}

int phoebe_qualifier_from_index (const char **qualifier, int index)
{
	/*
	 * This is a public function for accessing parameter qualifiers from the
	 * global parameter table. The returned string must *not* be freed.
	 *
	 * Return values:
	 *
	 *   ERROR_PARAMETER_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	if (index < 0 || index > PHOEBE_parameters_no) {
		*qualifier = NULL;
		return ERROR_PARAMETER_INDEX_OUT_OF_RANGE;
	}

	*qualifier = PHOEBE_parameters[index].qualifier;
	return SUCCESS;
}

int phoebe_type_from_index (PHOEBE_type *type, int index)
{
	/*
	 * This is a public function for accessing parameter types from the global
	 * parameter table.
	 *
	 * Return values:
	 *
	 *   ERROR_PARAMETER_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	if (index < 0 || index > PHOEBE_parameters_no) {
		*type = -1;
		return ERROR_PARAMETER_INDEX_OUT_OF_RANGE;
	}

	*type = PHOEBE_parameters[index].type;
	return SUCCESS;
}

int phoebe_qualifier_from_description (const char **qualifier, char *description)
{
	/*
	 * This function returns the qualifier of the given description. It will
	 * probably be never used, but here it is for sake of completion.
	 *
	 * Return values:
	 *
	 *   ERROR_DESCRIPTION_NOT_FOUND
	 *   SUCCESS
	 */
	
	int i;
	
	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].description, description) == 0 ) {
			*qualifier = PHOEBE_parameters[i].qualifier;
			return SUCCESS;
		}

	return ERROR_DESCRIPTION_NOT_FOUND;
}

int phoebe_description_from_qualifier (const char **description, char *qualifier)
{
	/*
	 * This function returns the description of the given qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */
	
	int i;
	
	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].qualifier, qualifier) == 0 ) {
			*description = PHOEBE_parameters[i].description;
			return SUCCESS;
		}

	return ERROR_QUALIFIER_NOT_FOUND;
}

int phoebe_index_from_qualifier (int *index, char *qualifier)
{
	/*
	 * This function returns the parameter table index of the given qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int i;

	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].qualifier, qualifier) == 0 ) {
			*index = i;
			return SUCCESS;
		}

	return ERROR_QUALIFIER_NOT_FOUND;
}

int phoebe_index_from_description (int *index, char *description)
{
	/*
	 * This function returns the parameter table index of the given description.
	 * Again, this function probably won't be ever used, but here it is for
	 * sake of completion.
	 *
	 * Return values:
	 *
	 *   ERROR_DESCRIPTION_NOT_FOUND
	 *   SUCCESS
	 */
	
	int i;
	
	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].description, description) == 0 ) {
			*index = i;
			return SUCCESS;
		}

	return ERROR_DESCRIPTION_NOT_FOUND;
}

int phoebe_type_from_qualifier (PHOEBE_type *type, char *qualifier)
{
	/*
	 * This function returns the parameter type of the given qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int i;
	
	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].qualifier, qualifier) == 0 ) {
			*type = PHOEBE_parameters[i].type;
			return SUCCESS;
		}

	return ERROR_QUALIFIER_NOT_FOUND;
}

int phoebe_type_from_description (PHOEBE_type *type, char *description)
{
	/*
	 * This function returns the parameter type of the given description.
	 *
	 * Return values:
	 *
	 *   ERROR_DESCRIPTION_NOT_FOUND
	 *   SUCCESS
	 */
	
	int i;
	
	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].description, description) == 0 ) {
			*type = PHOEBE_parameters[i].type;
			return SUCCESS;
		}

	return ERROR_DESCRIPTION_NOT_FOUND;
}

int phoebe_kind_from_qualifier (PHOEBE_parameter_kind *kind, char *qualifier)
{
	/*
	 * This function returns the parameter kind from the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int i;
	
	for (i = 0; i < PHOEBE_parameters_no; i++)
		if ( strcmp (PHOEBE_parameters[i].qualifier, qualifier) == 0 ) {
			*kind = PHOEBE_parameters[i].kind;
			return SUCCESS;
		}

	return ERROR_QUALIFIER_NOT_FOUND;
}

/* ************************************************************************** */

int intern_get_value_int (int *value, char *qualifier)
{
	/*
	 * This function is used as short-cut to get the integer value of the
	 * parameter determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.i;
	return SUCCESS;
}

int intern_get_value_bool (bool *value, char *qualifier)
{
	/*
	 * This function is used as short-cut to get a boolean value of the
	 * parameter determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.b;
	return SUCCESS;
}

int intern_get_value_double (double *value, char *qualifier)
{
	/*
	 * This function is used as short-cut to get a real value of the
	 * parameter determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;
		
	*value = PHOEBE_parameters[index].value.d;
	return SUCCESS;
}

int intern_get_value_string (const char **value, char *qualifier)
{
	/*
	 * This function is used as short-cut to get a string value of the
	 * parameter determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.str;
	return SUCCESS;
}

int intern_get_value_list_int (int *value, char *qualifier, int row)
{
	/*
	 * This function is used as short-cut to get an integer element value of the
	 * parameter array determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.iarray[row];
	return SUCCESS;
}

int intern_get_value_list_bool (bool *value, char *qualifier, int row)
{
	/*
	 * This function is used as short-cut to get a boolean element value of the
	 * parameter array determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.barray[row];
	return SUCCESS;
}

int intern_get_value_list_double (double *value, char *qualifier, int row)
{
	/*
	 * This function is used as short-cut to get a real element value of the
	 * parameter array determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.vec->val[row];
	return SUCCESS;
}

int intern_get_value_list_string (const char **value, char *qualifier, int row)
{
	/*
	 * This function is used as short-cut to get a string element value of the
	 * parameter array determined by the passed qualifier.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*value = PHOEBE_parameters[index].value.array->val.strarray[row];
	return SUCCESS;
}

int phoebe_get_parameter_value (char *qualifier, ...)
{
	/*
	 * This is the public function for changing qualifier values. It is the
	 * only function that should be used for this purpose, all other functions
	 * should be regarded as internal and should not be used.
	 *
	 * Synopsis:
	 *
	 *   phoebe_get_parameter_value (qualifier, [index, ], &value)
	 *
	 * Return values:
	 *
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index = 0;
	va_list args;

	int status;
	PHOEBE_type type;

	status = phoebe_type_from_qualifier (&type, qualifier);
	if (status != SUCCESS) return status;

	va_start (args, qualifier);
	switch (type) {
		case TYPE_INT: {
			int *value = va_arg (args, int *);
			intern_get_value_int (value, qualifier);
		}
		break;
		case TYPE_BOOL: {
			bool *value = va_arg (args, bool *);
			intern_get_value_bool (value, qualifier);
		}
		break;
		case TYPE_DOUBLE: {
			double *value = va_arg (args, double *);
			intern_get_value_double (value, qualifier);
		}
		break;
		case TYPE_STRING: {
			const char **value = va_arg (args, const char **);
			intern_get_value_string (value, qualifier);
		}
		break;
		case TYPE_INT_ARRAY:
			index = va_arg (args, int);
			{
			int *value = va_arg (args, int *);
			int i, range;

			status = phoebe_index_from_qualifier (&i, qualifier);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value (PHOEBE_parameters[i].dependency, &range);
			if (index < 0 || index > range-1) return ERROR_INDEX_OUT_OF_RANGE;

			intern_get_value_list_int (value, qualifier, index);
			}
		break;
		case TYPE_BOOL_ARRAY:
			index = va_arg (args, int);
			{
			bool *value = va_arg (args, bool *);
			int i, range;

			status = phoebe_index_from_qualifier (&i, qualifier);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value (PHOEBE_parameters[i].dependency, &range);
			if (index < 0 || index > range-1) return ERROR_INDEX_OUT_OF_RANGE;

			intern_get_value_list_bool (value, qualifier, index);
			}
		break;
		case TYPE_DOUBLE_ARRAY:
			index = va_arg (args, int);
			{
			double *value = va_arg (args, double *);
			int i, range;

			status = phoebe_index_from_qualifier (&i, qualifier);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value (PHOEBE_parameters[i].dependency, &range);
			if (index < 0 || index > range-1) return ERROR_INDEX_OUT_OF_RANGE;

			intern_get_value_list_double (value, qualifier, index);
			}
		break;
		case TYPE_STRING_ARRAY:
			index = va_arg (args, int);
			{
			const char **value = va_arg (args, const char **);
			int i, range;

			status = phoebe_index_from_qualifier (&i, qualifier);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value (PHOEBE_parameters[i].dependency, &range);
			if (index < 0 || index > range-1) return ERROR_INDEX_OUT_OF_RANGE;

			status = intern_get_value_list_string (value, qualifier, index);
			if (status != SUCCESS) return status;
			}
		break;
		}
	va_end (args);

	return SUCCESS;
}

/* ************************************************************************** */

int intern_set_value_int (char *qualifier, int value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier to its integer value. This function should not
	 * verify qualifier validity: since it is internal, it is only to be
	 * called by phoebe_set_parameter_value () function and that function
	 * verifies qualifier validity. If the qualifier is a parent to qualifier
	 * arrays (such as phoebe_lcno, phoebe_rvno etc), function update_parameter_
	 * _arrays () is called to redimension all dependent qualifiers.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_bool (qualifier, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int status, index, oldval;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	status = phoebe_get_parameter_value (qualifier, &oldval);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].value.i = value;
	if (strcmp (PHOEBE_parameters[index].dependency, "dependency") == 0)
		update_parameter_arrays (qualifier, oldval);

	return SUCCESS;
}

int intern_set_value_bool (char *qualifier, bool value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier to its boolean value. This function should not
	 * verify qualifier validity: since it is internal, it is only to be
	 * called by phoebe_set_parameter_value () function and that function
	 * verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_bool (qualifier, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int status, index;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].value.b = value;

	return SUCCESS;
}

int intern_set_value_double (char *qualifier, double value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier to its double value. This function should not
	 * verify qualifier validity: since it is internal, it is only to be
	 * called by phoebe_set_parameter_value () function and that function
	 * verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_double (qualifier, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int status, index;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].value.d = value;

	return SUCCESS;
}

int intern_set_value_string (char *qualifier, const char *value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier to its string value. This function should not
	 * verify qualifier validity: since it is internal, it is only to be
	 * called by phoebe_set_parameter_value () function and that function
	 * verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_string (qualifier, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int status, index;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	free (PHOEBE_parameters[index].value.str);
	PHOEBE_parameters[index].value.str = phoebe_malloc (strlen (value) + 1);
	strcpy (PHOEBE_parameters[index].value.str, value);

	/*
	 * If the qualified parameter is a menu, let's check if the option is
	 * valid. If it is not, just warn about it, but set its value anyway.
	 */

	if (PHOEBE_parameters[index].kind == KIND_MENU && !phoebe_parameter_menu_option_is_valid (qualifier, (char *) value))
		printf ("PHOEBE warning: option \"%s\" is not a valid menu option.\n", value);

	return SUCCESS;
}

int intern_set_value_list_int (char *qualifier, int row, int value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier array element to its integer value. This
	 * function should not verify qualifier validity: since it is internal,
	 * it is only to be called by phoebe_set_parameter_value () function and
	 * that function verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_list_int (qualifier, row, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int status, index, range;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
	if (status != SUCCESS) return status;

	if (row < 0 || row > range-1) return ERROR_INDEX_OUT_OF_RANGE;

	PHOEBE_parameters[index].value.iarray[row] = value;
	return SUCCESS;
}

int intern_set_value_list_bool (char *qualifier, int row, bool value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier array element to its boolean value. This
	 * function should not verify qualifier validity: since it is internal,
	 * it is only to be called by phoebe_set_parameter_value () function and
	 * that function verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_list_bool (qualifier, row, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int status, index, range;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
	if (status != SUCCESS) return status;

	if (row < 0 || row > range-1) return ERROR_INDEX_OUT_OF_RANGE;

	PHOEBE_parameters[index].value.barray[row] = value;
	return SUCCESS;
}

int intern_set_value_list_double (char *qualifier, int row, double value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier array element to its double value. This
	 * function should not verify qualifier validity: since it is internal,
	 * it is only to be called by phoebe_set_parameter_value () function and
	 * that function verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_list_double (qualifier, row, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int status, index, range;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
	if (status != SUCCESS) return status;

	if (row < 0 || row > range-1) return ERROR_INDEX_OUT_OF_RANGE;

	PHOEBE_parameters[index].value.vec->val[row] = value;
	return SUCCESS;
}

int intern_set_value_list_string (char *qualifier, int row, const char *value)
{
	/*
	 * This is the internal (not exported) function that sets the value of
	 * the particular qualifier array element to its string value. This
	 * function should not verify qualifier validity: since it is internal,
	 * it is only to be called by phoebe_set_parameter_value () function and
	 * that function verifies qualifier validity.
	 *
	 * Synopsis:
	 *
	 *   intern_set_value_list_string (qualifier, row, value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int status, index, range;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
	if (status != SUCCESS) return status;

	if (row < 0 || row > range-1) return ERROR_INDEX_OUT_OF_RANGE;

	free (PHOEBE_parameters[index].value.array->val.strarray[row]);
	PHOEBE_parameters[index].value.array->val.strarray[row] = phoebe_malloc (strlen (value) + 1);
	strcpy (PHOEBE_parameters[index].value.array->val.strarray[row], value);

	/*
	 * If the qualified parameter is a menu, let's check if the option is
	 * valid. If it is not, just warn about it, but set its value anyway.
	 */

	if (PHOEBE_parameters[index].kind == KIND_MENU && !phoebe_parameter_menu_option_is_valid (qualifier, (char *) value))
		printf ("PHOEBE warning: option \"%s\" is not a valid menu option.\n", value);

	return SUCCESS;
}

int phoebe_set_parameter_value (char *qualifier, ...)
{
	/*
	 * This is the public function for changing qualifier values. It is the
	 * only function that should be used for this purpose, all other functions
	 * should be regarded as internal and should not be used.
	 *
	 * Synopsis:
	 *
	 *   phoebe_set_parameter_value (qualifier, [curve, ] value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int index = 0;
	va_list args;
	int status;

	PHOEBE_type type;

	status = phoebe_type_from_qualifier (&type, qualifier);
	if (status != SUCCESS) return status;

	va_start (args, qualifier);
	switch (type) {
		case TYPE_INT: {
			int value = va_arg (args, int);
			status = intern_set_value_int (qualifier, value);
		}
		break;
		case TYPE_BOOL: {
			bool value = va_arg (args, bool);
			status = intern_set_value_bool (qualifier, value);
		}
		break;
		case TYPE_DOUBLE: {
			double value = va_arg (args, double);
			status = intern_set_value_double (qualifier, value);
		}
		break;
		case TYPE_STRING: {
			char *value = va_arg (args, char *);
			status = intern_set_value_string (qualifier, value);
		}
		break;
		case TYPE_INT_ARRAY:
			index = va_arg (args, int);
			{
			int value = va_arg (args, int);
			status = intern_set_value_list_int (qualifier, index, value);
			}
		break;
		case TYPE_BOOL_ARRAY:
			index = va_arg (args, int);
			{
			bool value = va_arg (args, bool);
			status = intern_set_value_list_bool (qualifier, index, value);
			}
		break;
		case TYPE_DOUBLE_ARRAY:
			index = va_arg (args, int);
			{
			double value = va_arg (args, double);
			status = intern_set_value_list_double (qualifier, index, value);
			}
		break;
		case TYPE_STRING_ARRAY:
			index = va_arg (args, int);
			{
			char *value = va_arg (args, char *);
			status = intern_set_value_list_string (qualifier, index, value);
			}
		break;
	}
	va_end (args);

	return status;
}

int phoebe_set_parameter_tba (char *qualifier, bool tba)
{
	/*
	 * This is the public function for changing qualifier's TBA (To Be Adjusted
	 * bit.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;
	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].tba = tba;

	return SUCCESS;
}

int phoebe_get_parameter_tba (char *qualifier, bool *tba)
{
	/*
	 * This is a public function for reading out qualifier's TBA (To Be Adjusted
	 * bit.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*tba = PHOEBE_parameters[index].tba;
	return SUCCESS;
}

int phoebe_get_parameter_step (char *qualifier, double *step)
{
	/*
	 * This is a public function for reading out qualifier's step size used
	 * for minimization.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*step = PHOEBE_parameters[index].step;
	return SUCCESS;
}

int phoebe_get_parameter_lower_limit (char *qualifier, double *valmin)
{
	/*
	 * This is the public function for reading out the lower parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*valmin = PHOEBE_parameters[index].min;
	return SUCCESS;
}

int phoebe_set_parameter_lower_limit (char *qualifier, double valmin)
{
	/*
	 * This is the public function for changing the lower parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;
	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].min = valmin;
	return SUCCESS;
}

int phoebe_get_parameter_upper_limit (char *qualifier, double *valmax)
{
	/*
	 * This is the public function for reading out the upper parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*valmax = PHOEBE_parameters[index].max;
	return SUCCESS;
}

int phoebe_set_parameter_upper_limit (char *qualifier, double valmax)
{
	/*
	 * This is the public function for changing the upper parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;
	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].max = valmax;
	return SUCCESS;
}

int phoebe_get_parameter_limits (char *qualifier, double *valmin, double *valmax)
{
	/*
	 * This is the public function for reading out qualifier limits.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	*valmin = PHOEBE_parameters[index].min;
	*valmax = PHOEBE_parameters[index].max;

	return SUCCESS;
}

int phoebe_set_parameter_limits (char *qualifier, double valmin, double valmax)
{
	/*
	 * This is the public function for changing qualifier limits.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;
	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].min = valmin;
	PHOEBE_parameters[index].max = valmax;

	return SUCCESS;
}

int phoebe_set_parameter_step (char *qualifier, double step)
{
	/*
	 * This is the public function for changing qualifier step.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;
	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	PHOEBE_parameters[index].step = step;

	return SUCCESS;
}

int get_input_independent_variable (const char *value, PHOEBE_input_indep *indep)
{
	/*
	 * This function returns the input independent variable enumeration.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_INDEP
	 *   SUCCESS
	 */

	if (strcmp (value, "Phase")      == 0) {
		*indep = INPUT_PHASE;
		return SUCCESS;
	}
	if (strcmp (value, "Time (HJD)") == 0) {
		*indep = INPUT_HJD;
		return SUCCESS;
	}
	
	return ERROR_INVALID_INDEP;
}

int get_input_dependent_variable (const char *value, PHOEBE_input_dep *dep)
{
	/*
	 * This function returns the input independent variable enumeration.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_DEP
	 *   SUCCESS
	 */

	if (strcmp (value, "Magnitude")    == 0) {
		*dep = INPUT_MAGNITUDE;
		return SUCCESS;
	}
	if (strcmp (value, "Flux")         == 0) {
		*dep = INPUT_FLUX;
		return SUCCESS;
	}
	if (strcmp (value, "Primary RV")   == 0) {
		*dep = INPUT_PRIMARY_RV;
		return SUCCESS;
	}
	if (strcmp (value, "Secondary RV") == 0) {
		*dep = INPUT_SECONDARY_RV;
		return SUCCESS;
	}

	return ERROR_INVALID_DEP;
}

int get_input_weight (const char *value, PHOEBE_input_weight *weight)
{
	/*
	 * This function returns the input individual weight enumeration.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_WEIGHT
	 *   SUCCESS
	 */

	if (strcmp (value, "Standard weight")  == 0) {
		*weight = INPUT_STANDARD_WEIGHT;
		return SUCCESS;
	}
	if (strcmp (value, "Standard deviation") == 0) {
		*weight = INPUT_STANDARD_DEVIATION;
		return SUCCESS;
	}
	if (strcmp (value, "Unavailable")  == 0) {
		*weight = INPUT_UNAVAILABLE;
		return SUCCESS;
	}
	
	return ERROR_INVALID_WEIGHT;
}

int get_output_independent_variable (const char *value, PHOEBE_output_indep *indep)
{
	/*
	 * This function returns the output independent variable enumeration.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_INDEP
	 *   SUCCESS
	 */

	if (strcmp (value, "Phase")      == 0) {
		*indep = OUTPUT_PHASE;
		return SUCCESS;
	}
	if (strcmp (value, "Time (HJD)") == 0) {
		*indep = OUTPUT_HJD;
		return SUCCESS;
	}

	return ERROR_INVALID_INDEP;
}

int get_output_dependent_variable (const char *value, PHOEBE_output_dep *dep)
{
	/*
	 * This function returns the output dependent variable enumeration.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_DEP
	 *   SUCCESS
	 */

	if (strcmp (value, "Magnitude")               == 0) {
		*dep = OUTPUT_MAGNITUDE;
		return SUCCESS;
	}
	if (strcmp (value, "Total flux")              == 0) {
		*dep = OUTPUT_TOTAL_FLUX;
		return SUCCESS;
	}
	if (strcmp (value, "Primary star flux")       == 0) {
		*dep = OUTPUT_PRIMARY_FLUX;
		return SUCCESS;
	}
	if (strcmp (value, "Secondary star flux")     == 0) {
		*dep = OUTPUT_SECONDARY_FLUX;
		return SUCCESS;
	}
	if (strcmp (value, "Primary RV")              == 0) {
		*dep = OUTPUT_PRIMARY_RV;
		return SUCCESS;
	}
	if (strcmp (value, "Secondary RV")            == 0) {
		*dep = OUTPUT_SECONDARY_RV;
		return SUCCESS;
	}
	if (strcmp (value, "Both RVs")                == 0) {
		*dep = OUTPUT_BOTH_RVS;
		return SUCCESS;
	}
	if (strcmp (value, "Primary normalized RV")   == 0) {
		*dep = OUTPUT_PRIMARY_NORMALIZED_RV;
		return SUCCESS;
	}
	if (strcmp (value, "Secondary normalized RV") == 0) {
		*dep = OUTPUT_SECONDARY_NORMALIZED_RV;
		return SUCCESS;
	}

	return ERROR_INVALID_DEP;
}

int get_output_weight (const char *value, PHOEBE_output_weight *weight)
{
	/*
	 * This function returns the output weighting variable enumeration.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_WEIGHT
	 *   SUCCESS
	 */

	if (strcmp (value, "Standard weight")    == 0) {
		*weight = OUTPUT_STANDARD_WEIGHT;
		return SUCCESS;
	}
	if (strcmp (value, "Standard deviation") == 0) {
		*weight = OUTPUT_STANDARD_DEVIATION;
		return SUCCESS;
	}
	if (strcmp (value, "Unavailable")        == 0) {
		*weight = OUTPUT_UNAVAILABLE;
		return SUCCESS;
	}

	return ERROR_INVALID_WEIGHT;
}

int get_ld_model_id (int *ldmodel)
{
	/*
	 * This function assigns the limb darkening law id to variable ldmodel. If
	 * an error occured during readout, -1 is assigned and ERROR_INVALID_LDLAW
	 * is returned.
	 */

	const char *ld;
	phoebe_get_parameter_value ("phoebe_ld_model", &ld);

	*ldmodel = -1;
	if (strcmp (ld, "Linear cosine law") == 0) *ldmodel = 1;
	if (strcmp (ld, "Logarithmic law")   == 0) *ldmodel = 2;
	if (strcmp (ld, "Square root law")   == 0) *ldmodel = 3;
	if (*ldmodel == -1) return ERROR_INVALID_LDLAW;

	return SUCCESS;
}

int phoebe_el3_units_id (PHOEBE_el3_units *el3_units)
{
	/*
	 * This function assigns the third light units id to el3_units variable.
	 * If an error occurs, -1 is assigned and ERROR_INVALID_EL3_UNITS code is
	 * returned.
	 */

	const char *el3str;
	phoebe_get_parameter_value ("phoebe_el3_units", &el3str);

	*el3_units = PHOEBE_EL3_UNITS_INVALID_ENTRY;
	if (strcmp (el3str, "Total light") == 0) *el3_units = PHOEBE_EL3_UNITS_TOTAL_LIGHT;
	if (strcmp (el3str,        "Flux") == 0) *el3_units = PHOEBE_EL3_UNITS_FLUX;
	if (*el3_units == PHOEBE_EL3_UNITS_INVALID_ENTRY)
		return ERROR_INVALID_EL3_UNITS;

	return SUCCESS;
}

int intern_get_from_keyword_file (char *qualifier, char *value_str)
{
	/*
	 * This is an internal function that looks up a qualifier in the global
	 * parameter table and sets its value to the passed string representation
	 * of that value, value_str.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;

	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	switch (PHOEBE_parameters[index].type) {
		case TYPE_INT:
			status = phoebe_set_parameter_value (qualifier, atoi (value_str));
		break;
		case TYPE_BOOL:
			status = phoebe_set_parameter_value (qualifier, atoi (value_str));
		break;
		case TYPE_DOUBLE:
			status = phoebe_set_parameter_value (qualifier, atof (value_str));
		break;
		case TYPE_STRING:
			/* Strip the string of quotes if necessary:                             */
			while (value_str[0] == '"') value_str++;
			while (value_str[strlen(value_str)-1] == '"') value_str[strlen(value_str)-1] = '\0';
			status = phoebe_set_parameter_value (qualifier, value_str);
		break;
	}

	if (status != SUCCESS)
		phoebe_lib_error ("%s", phoebe_error (status));

	return;
}

int intern_get_wavelength_dependent_parameter_from_keyword_file (char *qualifier, int idx, char *value_str)
{
	/*
	 * Much like the function above, only that this one takes three arguments,
	 * the 2nd one being the index of the wavelength-dependent parameter.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index, status;
	
	status = phoebe_index_from_qualifier (&index, qualifier);
	if (status != SUCCESS) return status;

	switch (PHOEBE_parameters[index].type) {
		case TYPE_INT_ARRAY:
			status = phoebe_set_parameter_value (qualifier, idx, atoi (value_str));
		break;
		case TYPE_BOOL_ARRAY:
			status = phoebe_set_parameter_value (qualifier, idx, atoi (value_str));
		break;
		case TYPE_DOUBLE_ARRAY:
			status = phoebe_set_parameter_value (qualifier, idx, atof (value_str));
		break;
		case TYPE_STRING_ARRAY:
			while (value_str[0] == '"') value_str++;
			while (value_str[strlen(value_str)-1] == '"') value_str[strlen(value_str)-1] = '\0';
			status = phoebe_set_parameter_value (qualifier, idx, value_str);
		break;
	}

	if (status != SUCCESS)
		phoebe_lib_error ("%s", phoebe_error (status));

	return status;
}

int open_parameter_file (const char *filename)
{
	/*
	 * This function opens PHOEBE 0.3x parameter files. The return value is:
	 *
	 *   ERROR_FILE_IS_EMPTY
	 *   ERROR_INVALID_HEADER
	 *   ERROR_FILE_NOT_REGULAR
	 *   ERROR_FILE_NO_PERMISSIONS
	 *   ERROR_FILE_NOT_FOUND
	 *   SUCCESS
	 */

	char readout_string[256];
	char *readout_str = readout_string;

	char keyword_string[256];
	char *keyword_str = keyword_string;

	char *value_str;

	int i, status;
	int lineno = 0;

	FILE *keyword_file;

	phoebe_debug ("entering function 'open_parameter_file ()'\n");

	/* First a checkup if everything is OK with the filename:                   */
	if (!filename_exists ((char *) filename))               return ERROR_FILE_NOT_FOUND;
	if (!filename_is_regular_file ((char *) filename))      return ERROR_FILE_NOT_REGULAR;
	if (!filename_has_read_permissions ((char *) filename)) return ERROR_FILE_NO_PERMISSIONS;

	keyword_file = fopen (filename, "r");

	/* Parameter files start with a commented header line; see if it's there: */

	fgets (readout_str, 255, keyword_file); lineno++;
	if (feof (keyword_file)) return ERROR_FILE_IS_EMPTY;
	readout_str[strlen(readout_str)-1] = '\0';

	if (strstr (readout_str, "PHOEBE") != 0) {
		/* Yep, it's there! Read out version number and if it's a legacy file,    */
		/* call a suitable function for opening legacy keyword files.             */

		char *version_str = strstr (readout_str, "PHOEBE");
		double version;
			
		if (sscanf (version_str, "PHOEBE %lf", &version) != 1) {
			/* Just in case, if the header line is invalid.                       */
			phoebe_lib_error ("Invalid header line in %s, aborting.\n", filename);
			return ERROR_INVALID_HEADER;
		}
		if (version < 0.30) {
			fclose (keyword_file);
			phoebe_debug ("opening legacy parameter file.\n");
			status = open_legacy_parameter_file (filename);
			return status;
		}
		phoebe_debug ("PHOEBE parameter file version: %2.2lf\n", version);
	}

	while (!feof (keyword_file)) {
		fgets (readout_str, 255, keyword_file); lineno++;
		if (feof (keyword_file)) break;

		/* Clear the read string of leading and trailing spaces, tabs, newlines,  */
		/* comments and empty lines:                                              */
		readout_str[strlen(readout_str)-1] = '\0';
		if (strchr (readout_str, '#') != 0) (strchr (readout_str, '#'))[0] = '\0';
		while (readout_str[0] == ' ' || readout_str[0] == '\t') readout_str++;
		while (readout_str[strlen(readout_str)-1] == ' ' || readout_str[strlen(readout_str)-1] == '\t') readout_str[strlen(readout_str)-1] = '\0';
		if (strlen (readout_str) == 0) continue;

		/*
		 * Read out the qualifier only. We can't read out the value here
		 * because it may be a string containing spaces and that would result
		 * in an incomplete readout.
		 */

		if (sscanf (readout_str, "%s = %*s", keyword_str) != 1) {
			phoebe_lib_error ("line %d of the parameter file is invalid, skipping.\n", lineno);
			continue;
		}

		value_str = strchr (readout_str, '=');
		if (value_str == NULL) {
			/* If the keyword doesn't have '=', it will be skipped.                 */
			phoebe_lib_error ("qualifier %s initialization (line %d) is invalid.\n", keyword_str, lineno);
			continue;
		}

		/* value_str now points to '=', we need the next character:               */
		value_str++;

		/* Eat all empty spaces at the beginning and at the end:                  */
		while (value_str[0] == ' ' || value_str[0] == '\t') value_str++;
		while (value_str[strlen (value_str)-1] == ' ' || value_str[strlen (value_str)-1] == '\t') value_str[strlen(value_str)-1] = '\0';

		phoebe_debug ("qualifier: %s, value: %s\n", keyword_str, value_str);

		if (strchr (keyword_str, '.') != NULL) {
			/*
			 * If there is a '.' in a qualifier, it means it is a field of an
			 * adjustable parameter and we need to handle it differently.
			 */

			for (i = 0; i < PHOEBE_parameters_no; i++)
				if (strncmp (keyword_str, PHOEBE_parameters[i].qualifier, strlen (PHOEBE_parameters[i].qualifier)) == 0) break;
			if ( (i == PHOEBE_parameters_no - 1) && (strncmp (keyword_str, PHOEBE_parameters[i].qualifier, strlen (PHOEBE_parameters[i].qualifier)) != 0) )
				phoebe_lib_error ("qualifier %s is unkown, skipping.\n", keyword_str);

			if (PHOEBE_parameters[i].type == TYPE_DOUBLE) {
				if (strstr (keyword_str, ".VAL")  != NULL) PHOEBE_parameters[i].value.d = atof (value_str);
				if (strstr (keyword_str, ".MIN")  != NULL) PHOEBE_parameters[i].min     = atof (value_str);
				if (strstr (keyword_str, ".MAX")  != NULL) PHOEBE_parameters[i].max     = atof (value_str);
				if (strstr (keyword_str, ".STEP") != NULL) PHOEBE_parameters[i].step    = atof (value_str);
				if (strstr (keyword_str, ".ADJ")  != NULL) {
					if (strcmp (value_str, "YES") == 0 || strcmp (value_str, "TRUE") == 0 || strcmp (value_str, "1") == 0)
						PHOEBE_parameters[i].tba = YES;
					else
						PHOEBE_parameters[i].tba = NO;
				}
			}
			if (PHOEBE_parameters[i].type == TYPE_DOUBLE_ARRAY) {
				if (strstr (keyword_str, ".MIN")  != NULL) PHOEBE_parameters[i].min     = atof (value_str);
				if (strstr (keyword_str, ".MAX")  != NULL) PHOEBE_parameters[i].max     = atof (value_str);
				if (strstr (keyword_str, ".STEP") != NULL) PHOEBE_parameters[i].step    = atof (value_str);
				if (strstr (keyword_str, ".ADJ")  != NULL) {
					if (strcmp (value_str, "YES") == 0 || strcmp (value_str, "TRUE") == 0 || strcmp (value_str, "1") == 0)
						PHOEBE_parameters[i].tba = YES;
					else
						PHOEBE_parameters[i].tba = NO;
				}
				if (strstr (keyword_str, ".VAL")  != NULL) {
					char *index = strchr (keyword_str, '[');
					int idx;
					sscanf (index, "[%d]", &idx);
					PHOEBE_parameters[i].value.vec->val[idx-1] = atof (value_str);
				}
			}
		}
		else {
			for (i = 0; i < PHOEBE_parameters_no; i++) {
				if (strcmp (keyword_str, PHOEBE_parameters[i].qualifier) == 0) {
					intern_get_from_keyword_file (PHOEBE_parameters[i].qualifier, value_str);
					break;
				}
				if (strncmp (keyword_str, PHOEBE_parameters[i].qualifier, strlen (PHOEBE_parameters[i].qualifier)) == 0) {
					/* This is appopriate for wavelength-dependent parameters; a conse- */
					/* cutive number is attached to the end of the keyword for distinc- */
					/* tion.                                                            */

					char *index = strchr (keyword_str, '[');
					int idx;

					if (index == NULL) {
						phoebe_lib_error ("qualifier %s is invalid, ignoring.\n", keyword_str);
						break;
					}

					sscanf (index, "[%d]", &idx);
					intern_get_wavelength_dependent_parameter_from_keyword_file (PHOEBE_parameters[i].qualifier, idx-1, value_str);
					break;
				}
			}

			if ( (i == PHOEBE_parameters_no) && (strncmp (keyword_str, PHOEBE_parameters[i-1].qualifier, strlen (PHOEBE_parameters[i-1].qualifier)) != 0) )
				phoebe_lib_error ("qualifier '%s' isn't recognized by PHOEBE.\n", keyword_str);
		}

		/* As explained above, we have to repoint strings:                        */
		readout_str = readout_string;
		keyword_str = keyword_string;
	}

	fclose (keyword_file);

	phoebe_debug ("leaving function 'open_parameter_file ()'\n");

	return SUCCESS;
}

int save_to_parameter_file (char *keyword, FILE *file)
{
	int index, status, range;
	int j;

	status = phoebe_index_from_qualifier (&index, keyword);
	if (status != SUCCESS) return status;

	if (PHOEBE_parameters[index].kind == KIND_ADJUSTABLE) {
		switch (PHOEBE_parameters[index].type) {
			case TYPE_DOUBLE:
				fprintf (file, "%s.VAL  = %lf\n", keyword, PHOEBE_parameters[index].value.d);
			break;
			case TYPE_DOUBLE_ARRAY:
				status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
				if (status != SUCCESS) return status;
				for (j = 0; j < range; j++)
					fprintf (file, "%s[%d].VAL  = %lf\n", keyword, j+1, PHOEBE_parameters[index].value.vec->val[j]);
			break;
		}
		fprintf (file, "%s.ADJ  = %d\n",  keyword, PHOEBE_parameters[index].tba);
		fprintf (file, "%s.STEP = %lf\n", keyword, PHOEBE_parameters[index].step);
		fprintf (file, "%s.MIN  = %lf\n", keyword, PHOEBE_parameters[index].min);
		fprintf (file, "%s.MAX  = %lf\n", keyword, PHOEBE_parameters[index].max);
	}
	else {
		switch (PHOEBE_parameters[index].type) {
			case TYPE_INT:
				fprintf (file, "%s = %d\n", keyword, PHOEBE_parameters[index].value.i);
			break;
			case TYPE_BOOL:
				fprintf (file, "%s = %d\n", keyword, PHOEBE_parameters[index].value.b);
			break;
			case TYPE_DOUBLE:
				fprintf (file, "%s = %lf\n", keyword, PHOEBE_parameters[index].value.d);
			break;
			case TYPE_STRING:
				fprintf (file, "%s = \"%s\"\n", keyword, PHOEBE_parameters[index].value.str);
			break;
			case TYPE_INT_ARRAY:
				status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
				if (status != SUCCESS) return status;
				for (j = 0; j < range; j++)
					fprintf (file, "%s[%d] = %d\n", keyword, j+1, PHOEBE_parameters[index].value.iarray[j]);
			break;
			case TYPE_BOOL_ARRAY:
				status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
				if (status != SUCCESS) return status;
				for (j = 0; j < range; j++)
					fprintf (file, "%s[%d] = %d\n", keyword, j+1, PHOEBE_parameters[index].value.barray[j]);
			break;
			case TYPE_DOUBLE_ARRAY:
				status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
				if (status != SUCCESS) return status;
				for (j = 0; j < range; j++)
					fprintf (file, "%s[%d] = %lf\n", keyword, j+1, PHOEBE_parameters[index].value.vec->val[j]);
			break;
			case TYPE_STRING_ARRAY:
				status = phoebe_get_parameter_value (PHOEBE_parameters[index].dependency, &range);
				if (status != SUCCESS) return status;
				for (j = 0; j < range; j++)
					fprintf (file, "%s[%d] = \"%s\"\n", keyword, j+1, PHOEBE_parameters[index].value.array->val.strarray[j]);
			break;
			default:
				phoebe_lib_error ("exception handler invoked in save_to_parameter_file (), please report this!\n");
			break;
			}
		}

	return SUCCESS;
}

int save_parameter_file (const char *filename)
	{
	/* This function saves PHOEBE 0.3x keyword files. The return value is:      */
	/*                                                                          */
	/*   ERROR_FILE_NOT_REGULAR                                                 */
	/*   ERROR_FILE_NO_PERMISSIONS                                              */
	/*   SUCCESS                                                                */

	FILE *keyword_file;
	int i;

	keyword_file = fopen (filename, "w");

	/* First a checkup if everything is OK with the filename:                   */
	if (!keyword_file) return ERROR_FILE_IS_INVALID;
	if (!filename_is_regular_file ((char *) filename))       return ERROR_FILE_NOT_REGULAR;
	if (!filename_has_write_permissions ((char *) filename)) return ERROR_FILE_NO_PERMISSIONS;

	/* Write a version header:                                                  */
	fprintf (keyword_file, "# Parameter file conforming to %s\n", PHOEBE_VERSION_NUMBER);

	for (i = 0; i < PHOEBE_parameters_no; i++)
		save_to_parameter_file (PHOEBE_parameters[i].qualifier, keyword_file);

	fclose (keyword_file);
	return SUCCESS;
	}

int open_legacy_parameter_file (const char *filename)
	{
	phoebe_lib_error ("Not yet implemented!\n");
	return SUCCESS;
	}
