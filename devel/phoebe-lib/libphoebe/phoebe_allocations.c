#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_base.h"
#include "phoebe_calculations.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"
#include "phoebe_ld.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

char *parse_data_line (char *in)
{
	/*
	 * This function takes a string, parses it (cleans it of all comment
	 * delimeters, spaces, tabs and newlines), copies the contents to the newly
	 * allocated string.
	 */

	char *value = NULL;

	/*
	 * If the line contains the comment delimeter '#', discard everything that
	 * follows it (strip the input line):
	 */

	if (strchr (in, '#') != NULL)
		in[strlen(in)-strlen(strchr(in,'#'))] = '\0';

	/*
	 * If the line is empty (or it became empty after removing the comment),
	 * return NULL:
	 */

	if (strlen(in) == 0) return NULL;

	/*
	 * If we have spaces, tabs or newlines in front of the first character in
	 * line, remove them by incrementing the pointer by 1:
	 */

	while ( (*in ==  ' ') || (*in == '\t') || (*in == '\n') || (*in == 13) ) {
		if (strlen (in) > 1) in++;
		else return NULL;
	}

	/*
	 * Let's do the same for the tail of the string:
	 */

	while ( (in[strlen(in)-1] ==  ' ') || (in[strlen(in)-1] == '\t') || (in[strlen(in)-1] == '\n') || (in[strlen(in)-1] == 13) )
		in[strlen(in)-1] = '\0';

	value = strdup (in);
	return value;
}

int phoebe_compute_lc (PHOEBE_curve *curve, PHOEBE_vector *nodes, char *lciname, WD_LCI_parameters *params)
{
	/*
	 *
	 */

	PHOEBE_parameter *par;
	bool state;

	create_lci_file (lciname, params);
	call_wd_to_get_fluxes (curve, nodes);

	par = phoebe_parameter_lookup ("phoebe_ie_switch");
	phoebe_parameter_get_value (par, &state);

	if (state == TRUE) {
		int i, lcno;
		double A;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
		for (i = 0; i < lcno; i++) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_extinction"), &A);
			apply_extinction_correction (curve, A);
		}

	}

	return SUCCESS;
}

int read_in_synthetic_data (PHOEBE_curve *curve, PHOEBE_vector *indep, int curve_index, PHOEBE_column_type itype, PHOEBE_column_type dtype)
{
	/*
	 * This function creates a WD input file 'lcin.active' and calls LC to
	 * calculate the data. The points in which values should be calculated are
	 * passed in the indep vector. In general, these should correspond to
	 * observational data, but may be also some equidistant set of times or
	 * phases if only a synthetic curve is to be created without any
	 * observational counterpart.
	 */

	int i;
	int mpage;
	int jdphs;
	int status;

	char *filter;
	char *filename;
	WD_LCI_parameters params;

	double A;

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (!indep)
		return ERROR_VECTOR_NOT_INITIALIZED;

	switch (itype) {
		case PHOEBE_COLUMN_HJD:
			jdphs = 1;
		break;
		case PHOEBE_COLUMN_PHASE:
			jdphs = 2;
		break;
		default:
			phoebe_lib_error ("exception handler invoked by itype switch in read_in_synthetic_data (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	switch (dtype) {
		case PHOEBE_COLUMN_FLUX:
			mpage = 1;
			curve->type = PHOEBE_CURVE_LC;
		break;
		case PHOEBE_COLUMN_MAGNITUDE:
			mpage = 1;
			curve->type = PHOEBE_CURVE_LC;
		break;
		case PHOEBE_COLUMN_PRIMARY_RV:
			mpage = 2;
			curve->type = PHOEBE_CURVE_RV;
		break;
		case PHOEBE_COLUMN_SECONDARY_RV:
			mpage = 2;
			curve->type = PHOEBE_CURVE_RV;
		break;
		default:
			phoebe_lib_error ("exception handler invoked by dtype switch in read_in_synthetic_data (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), curve_index, &filter);
	curve->passband = phoebe_passband_lookup (filter);

	status = wd_lci_parameters_get (&params, mpage, curve_index);
	if (status != SUCCESS) return status;

	params.JDPHS = jdphs;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_extinction"), curve_index, &A);

	filename = phoebe_resolve_relative_filename ("lcin.active");
	create_lci_file (filename, &params);

	switch (dtype) {
		case PHOEBE_COLUMN_MAGNITUDE:
			call_wd_to_get_fluxes (curve, indep);
			apply_extinction_correction (curve, A);
		break;
		case PHOEBE_COLUMN_FLUX:
			call_wd_to_get_fluxes (curve, indep);
			apply_extinction_correction (curve, A);
		break;
		case PHOEBE_COLUMN_PRIMARY_RV:
			call_wd_to_get_rv1 (curve, indep);
		break;
		case PHOEBE_COLUMN_SECONDARY_RV:
			call_wd_to_get_rv2 (curve, indep);
		break;
		default:
			phoebe_lib_error ("exception handler invoked by dtype switch in read_in_synthetic_data (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	remove (filename);
	free (filename);

	if (dtype == PHOEBE_COLUMN_MAGNITUDE) {
		double mnorm;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mnorm"), &mnorm);
		transform_flux_to_magnitude (curve->dep, mnorm);
	}

	if (dtype == PHOEBE_COLUMN_PRIMARY_RV || dtype == PHOEBE_COLUMN_SECONDARY_RV) {
		for (i = 0; i < curve->dep->dim; i++)
			curve->dep->val[i] *= 100.0;
	}

	return SUCCESS;
}

int read_in_ephemeris_parameters (double *hjd0, double *period, double *dpdt, double *pshift)
{
	/*
	 * This function speeds up the ephemeris readout.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hjd0"), hjd0);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_period"), period);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dpdt"), dpdt);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pshift"), pshift);

	return SUCCESS;
}

int read_in_adjustable_parameters (int *tba, double **values)
{
	int i;
	PHOEBE_parameter_list *list = phoebe_parameter_list_get_marked_tba ();

	*tba = 0;
	while (list) {
		(*tba)++;
		list = list->next;
	}

	*values = phoebe_malloc (*tba * sizeof (**values));
	list = phoebe_parameter_list_get_marked_tba ();

	i = 0;
	while (list) {
		phoebe_parameter_get_value (list->par, &(*values[i]));
		i++; list = list->next;
	}

	return SUCCESS;
/*
	int i, j, k;
	int status, calchla_idx, calcvga_idx, sma_idx, incl_idx;
	bool calchla, calcvga, asini;
	int lcno;

	phoebe_parameter_get_value ("phoebe_compute_hla_switch", &calchla);
	phoebe_parameter_get_value ("phoebe_compute_vga_switch", &calcvga);
	phoebe_parameter_get_value ("phoebe_asini_switch",   &asini);

	phoebe_parameter_get_value ("phoebe_lcno", &lcno);

	status = phoebe_index_from_qualifier (&calchla_idx, "phoebe_hla");
	if (status != SUCCESS) return status;
	status = phoebe_index_from_qualifier (&calcvga_idx, "phoebe_vga");
	if (status != SUCCESS) return status;
	status = phoebe_index_from_qualifier (&sma_idx, "phoebe_sma");
	if (status != SUCCESS) return status;
	status = phoebe_index_from_qualifier (&incl_idx, "phoebe_incl");
	if (status != SUCCESS) return status;

	*values = NULL;
	*indices = NULL;

	int multi;

	j = 0;
	for (i = 0; i < PHOEBE_parameters_no; i++) {
		if (PHOEBE_parameters[i].kind == KIND_ADJUSTABLE && PHOEBE_parameters[i].tba == YES) {
*/
			/* First, let's check whether the user tries something silly like ad-   */
			/* justing HLAs when they should be calculated:                         */
/*
			if (i == calchla_idx && calchla == TRUE) {
				phoebe_lib_error ("light levels (HLAs) cannot be adjusted if the calculation\n");
				phoebe_lib_error ("switch 'phoebe_compute_hla_switch' is turned on. Ignoring HLAs.\n");
				continue;
			}
			if (i == calcvga_idx && calcvga == TRUE) {
				phoebe_lib_error ("gamma velocity (VGA) cannot be adjusted if the calculation\n");
				phoebe_lib_error ("switch 'phoebe_compute_vga_switch' is turned on. Ignoring VGA.\n");
				continue;
			}
			if (i == sma_idx && asini == TRUE) {
				bool smaadj = PHOEBE_parameters[sma_idx].tba;
				bool incadj = PHOEBE_parameters[incl_idx].tba;

				if (smaadj == TRUE && incadj == TRUE) {
					phoebe_lib_error ("semi-major axis and inclination cannot both be adjusted\n");
					phoebe_lib_error ("when a sin(i) = const constraint is used. Ignoring SMA.\n");
					continue;
				}
			}
*/
			/* Now we have to see whether it's a system parameter (one for all      */
			/* light curves) or is it a filter-dependent parameter (one for each    */
			/* light curve):                                                        */
/*
			switch (PHOEBE_parameters[i].type) {
				case TYPE_DOUBLE:
					j++;
					multi = 0;
				break;
				case TYPE_DOUBLE_ARRAY:
					j += lcno;
					multi = 1;
				break;
				default:
					phoebe_lib_error ("adjustable parameter is not a number??\n");
					return ERROR_ARG_NOT_DOUBLE;
				break;
			}
*/
			/* Next, we (re)allocate memory:                                  */
/*
			*values   = phoebe_realloc (*values,   j * sizeof (**values));
			*indices  = phoebe_realloc (*indices,  j * sizeof (**indices));
*/
			/* Finally, we fill in all values: if multi = 0, then only (j-1). field */
			/* is updated, otherwise all filter-dependent fields are updated:       */
/*
			for (k = j - (multi * (lcno-1)) - 1; k < j; k++) {
				(*indices)[k] = i;
				if (PHOEBE_parameters[i].type == TYPE_DOUBLE)
					(*values)[k] = PHOEBE_parameters[i].value.d;
				if (PHOEBE_parameters[i].type == TYPE_DOUBLE_ARRAY)
					(*values)[k] = PHOEBE_parameters[i].value.vec->val[k - (j - (multi * (lcno-1)) - 1)];
			}
		}
	}

	*tba = j;
*/
	return SUCCESS;
}

int get_passband_id (const char *passband)
{
	/*
	 * This function returns the passband id of the passed passband name.
	 *
	 * Return values:
	 *
	 *   id  ..  passband id if recognition was successful
	 */

	int passid = -1;

	if (strcmp (passband,  "Stromgren:u") == 0) passid =  1;
	if (strcmp (passband,  "Stromgren:v") == 0) passid =  2;
	if (strcmp (passband,  "Stromgren:b") == 0) passid =  3;
	if (strcmp (passband,  "Strongren:y") == 0) passid =  4;
	if (strcmp (passband,    "Johnson:U") == 0) passid =  5;
	if (strcmp (passband,    "Johnson:B") == 0) passid =  6;
	if (strcmp (passband,    "Johnson:V") == 0) passid =  7;
	if (strcmp (passband,    "Johnson:R") == 0) passid =  8;
	if (strcmp (passband,    "Johnson:I") == 0) passid =  9;
	if (strcmp (passband,    "Johnson:J") == 0) passid = 10;
	if (strcmp (passband,    "Johnson:H") == 0) passid = 26;
	if (strcmp (passband,    "Johnson:K") == 0) passid = 11;
	if (strcmp (passband,    "Johnson:L") == 0) passid = 12;
	if (strcmp (passband,    "Johnson:M") == 0) passid = 13;
	if (strcmp (passband,    "Johnson:N") == 0) passid = 14;
	if (strcmp (passband,    "Cousins:R") == 0) passid = 15;
	if (strcmp (passband,    "Cousins:I") == 0) passid = 16;
	if (strcmp (passband, "Hipparcos:BT") == 0) passid = 23;
	if (strcmp (passband, "Hipparcos:VT") == 0) passid = 24;
	if (strcmp (passband, "Hipparcos:Hp") == 0) passid = 25;
	if (passid == -1) {
		phoebe_lib_warning ("passband not set or invalid, reverting to Johnson V.\n");
		passid = 7;
	}

	return passid;
}

int get_level_weighting_id (const char *type)
	{
	/* This function returns the id of the passed level weighting PDF.          */

	int id = -1;

	if (strcmp (type, "None") == 0)               id = 0;
	if (strcmp (type, "Poissonian scatter") == 0) id = 1;
	if (strcmp (type, "Low light scatter") == 0)  id = 2;

	if (id == -1)
		{
		phoebe_lib_error ("level weighting type invalid, assuming Poissonian scatter.\n");
		return 1;
		}
	return id;
	}
