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

/**
 * SECTION:phoebe_allocations
 * @title: PHOEBE allocations
 * @short_description: functions that allocate memory for large structures
 *
 * These are the functions that allocate memory and read in the values for
 * large structures such as LC/DC parameters, etc. This source will have
 * become obsolete as allocations are moved to phoebe_types.
 */

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
