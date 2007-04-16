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

int read_in_synthetic_data (PHOEBE_curve *curve, PHOEBE_vector *indep, int curve_index, PHOEBE_output_dep var)
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
	int status;

	char *filename;
	WD_LCI_parameters params;

	double A;

	PHOEBE_el3_units el3units;
	double           el3value;

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (!indep)
		return ERROR_VECTOR_NOT_INITIALIZED;

	switch (var) {
		case OUTPUT_TOTAL_FLUX:
			mpage = 1;
			curve->type = PHOEBE_CURVE_LC;
		break;
		case OUTPUT_MAGNITUDE:
			mpage = 1;
			curve->type = PHOEBE_CURVE_LC;
		break;
		case OUTPUT_PRIMARY_RV:
			mpage = 2;
			curve->type = PHOEBE_CURVE_RV;
		break;
		case OUTPUT_SECONDARY_RV:
			mpage = 2;
			curve->type = PHOEBE_CURVE_RV;
		break;
		default:
			return ERROR_INVALID_DEP;
		break;
	}

	status = read_in_wd_lci_parameters (&params, mpage, curve_index);
	if (status != SUCCESS) return status;

	phoebe_get_parameter_value ("phoebe_extinction", curve_index, &A);

	status = phoebe_el3_units_id (&el3units);
	if (status != SUCCESS) return status;

	phoebe_get_parameter_value ("phoebe_el3", curve_index, &el3value);

	filename = resolve_relative_filename ("lcin.active");
	create_lci_file (filename, params);

	switch (var) {
		case OUTPUT_MAGNITUDE:
			call_wd_to_get_fluxes (curve, indep);
			apply_third_light_correction (curve, el3units, el3value);
			apply_extinction_correction (curve, A);
		break;
		case OUTPUT_PRIMARY_FLUX:
			call_wd_to_get_fluxes (curve, indep);
			apply_third_light_correction (curve, el3units, el3value);
			apply_extinction_correction (curve, A);
		break;
		case OUTPUT_SECONDARY_FLUX:
			call_wd_to_get_fluxes (curve, indep);
			apply_third_light_correction (curve, el3units, el3value);
			apply_extinction_correction (curve, A);
		break;
		case OUTPUT_TOTAL_FLUX:
			call_wd_to_get_fluxes (curve, indep);
			apply_third_light_correction (curve, el3units, el3value);
			apply_extinction_correction (curve, A);
		break;
		case OUTPUT_PRIMARY_RV:
			call_wd_to_get_rv1    (curve, indep);
		break;
		case OUTPUT_SECONDARY_RV:
			call_wd_to_get_rv2    (curve, indep);
		break;
	}

	remove (filename);
	free (filename);

	if ( var == OUTPUT_MAGNITUDE ) {
		double mnorm;
		phoebe_get_parameter_value ("phoebe_mnorm", &mnorm);
		transform_flux_to_magnitude (curve->dep, mnorm);
	}

	if ( var == OUTPUT_PRIMARY_RV || var == OUTPUT_SECONDARY_RV ) {
		for (i = 0; i < curve->dep->dim; i++)
			curve->dep->val[i] *= 100.0;
	}

	return SUCCESS;
}

int read_in_observational_data (const char *filename,
	PHOEBE_curve *obs,  int indep,  int outindep,
	                    int dep,    int outdep,
	                    int weight, int outweight,
	bool alias, double phmin, double phmax)
{
	/*
	 * This function reads out observational data. It should be fool-proof to
	 * a reasonable extent. Memory has also been checked against leaks.
	 *
	 * Return codes:
	 *
	 *   ERROR_CURVE_NOT_INITIALIZED
	 *   ERROR_CURVE_ALREADY_ALLOCATED
	 *   ERROR_NONSENSE_DATA_REQUEST
	 *   ERROR_NEGATIVE_STANDARD_DEVIATION
	 *   ERROR_FILE_NOT_FOUND
	 *   ERROR_FILE_IS_INVALID
	 *   SUCCESS
	 */

	int no_of_columns;

	int status;

	phoebe_debug ("entering read_in_observational_data ()\n");

	phoebe_debug ("  expected input:   indep: %-6s; dep: %-13s; weight: %s\n", phoebe_input_indep_name (indep), phoebe_input_dep_name (dep), phoebe_input_weight_name (weight));
	phoebe_debug ("  requested output: indep: %-6s; dep: %-13s; weight: %s\n", phoebe_output_indep_name (outindep), phoebe_output_dep_name (outdep), phoebe_output_weight_name (outweight));

	/*
	 * If the data is not initialized or is already allocated, it will cause a
	 * memory leak.
	 */

	if (!obs) {
		phoebe_lib_error ("PHOEBE_curve is not initialized, aborting.\n");
		return ERROR_CURVE_NOT_INITIALIZED;
	}
	if (obs->indep->val) {
		phoebe_lib_error ("PHOEBE_curve is already allocated, aborting.\n");
		return ERROR_CURVE_ALREADY_ALLOCATED;
	}

	/* Check how many columns are expected in the input file:                   */
	if (weight == INPUT_UNAVAILABLE || outweight == OUTPUT_UNAVAILABLE)
		no_of_columns = 2;
	else
		no_of_columns = 3;

	/*
	 * Let's verify if the user wants to get something that doesn't make sense:
	 */

	if  (
		!( (dep == INPUT_FLUX        ) && (outdep    == OUTPUT_TOTAL_FLUX             ) ) &&
		!( (dep == INPUT_FLUX        ) && (outdep    == OUTPUT_MAGNITUDE              ) ) &&
		!( (dep == INPUT_MAGNITUDE   ) && (outdep    == OUTPUT_TOTAL_FLUX             ) ) &&
		!( (dep == INPUT_MAGNITUDE   ) && (outdep    == OUTPUT_MAGNITUDE              ) ) &&
		!( (dep == INPUT_PRIMARY_RV  ) && (outdep    == OUTPUT_PRIMARY_RV             ) ) &&
		!( (dep == INPUT_PRIMARY_RV  ) && (outdep    == OUTPUT_PRIMARY_NORMALIZED_RV  ) ) &&
		!( (dep == INPUT_SECONDARY_RV) && (outdep    == OUTPUT_SECONDARY_RV           ) ) &&
		!( (dep == INPUT_SECONDARY_RV) && (outdep    == OUTPUT_SECONDARY_NORMALIZED_RV) )
		)
	return ERROR_NONSENSE_DATA_REQUEST;

	/* Check for the filename validity:                                       */
	if (!filename_exists (filename))          return ERROR_FILE_NOT_FOUND;
	if (!filename_is_regular_file (filename)) return ERROR_FILE_IS_INVALID;

	obs = phoebe_curve_new_from_file ((char *) filename);

	/* Now do all the necessary transformations:                                */
	if ( indep == INPUT_HJD && outindep == OUTPUT_PHASE ) {
		double hjd0, period, dpdt, pshift;
		read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);
		transform_hjd_to_phase (obs->indep, hjd0, period, dpdt, 0.0);
	}

	if ( indep == INPUT_PHASE && outindep == OUTPUT_HJD ) {
		double hjd0, period, dpdt, pshift;
		read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);
		transform_phase_to_hjd (obs->indep, hjd0, period, dpdt, 0.0);
	}

	if ( dep == INPUT_MAGNITUDE && outdep == OUTPUT_TOTAL_FLUX ) {
 		double mnorm;
		phoebe_get_parameter_value ("phoebe_mnorm", &mnorm);

		/*
		 * If weights need to be transformed, we need to transform them *after*
		 * we transform magnitudes to fluxes, because the transformation funcion
		 * uses fluxes and not magnitudes.
		 */

 		transform_magnitude_to_flux (obs->dep, mnorm);
		if (weight == INPUT_STANDARD_DEVIATION && outweight != OUTPUT_UNAVAILABLE)
			transform_magnitude_sigma_to_flux_sigma (obs->weight, obs->dep);
	}
	if ( dep == INPUT_FLUX && outdep == OUTPUT_MAGNITUDE ) {
		double mnorm;
		phoebe_get_parameter_value ("phoebe_mnorm", &mnorm);

		/*
		 * If weights need to be transformed, we need to transform them *before*
		 * we transform fluxes to magnitudes, because the transformation funcion
		 * uses fluxes and not magnitudes.
		 */

		if (weight == INPUT_STANDARD_DEVIATION && outweight != OUTPUT_UNAVAILABLE)
			transform_flux_sigma_to_magnitude_sigma (obs->weight, obs->dep);
		transform_flux_to_magnitude (obs->dep, mnorm);
	}
	if ( dep == INPUT_PRIMARY_RV && outdep == OUTPUT_PRIMARY_NORMALIZED_RV ) {
		double sma, period;
		phoebe_get_parameter_value ("phoebe_sma",    &sma);
		phoebe_get_parameter_value ("phoebe_period", &period);
		normalize_kms_to_orbit (obs->dep, sma, period);
		if ( (weight == INPUT_STANDARD_DEVIATION) && (outweight != OUTPUT_UNAVAILABLE) )
			normalize_kms_to_orbit (obs->weight, sma, period);
	}
	if ( dep == INPUT_SECONDARY_RV && outdep == OUTPUT_SECONDARY_NORMALIZED_RV ) {
		double sma, period;
		phoebe_get_parameter_value ("phoebe_sma", &sma);
		phoebe_get_parameter_value ("phoebe_period", &period);
		normalize_kms_to_orbit (obs->dep, sma, period);
		if ( (weight == INPUT_STANDARD_DEVIATION) && (outweight != OUTPUT_UNAVAILABLE) )
			normalize_kms_to_orbit (obs->weight, sma, period);
	}

	if ( weight == INPUT_STANDARD_DEVIATION && outweight == OUTPUT_STANDARD_WEIGHT ) {
		status = transform_sigma_to_weight (obs->weight);
		if (status != SUCCESS) {
			obs->indep->dim = 0; free (obs->indep->val); obs->indep->val = NULL;
			obs->dep->dim   = 0; free (obs->dep->val);   obs->dep->val   = NULL;
			return status;
		}
	}
	if ( weight == INPUT_STANDARD_WEIGHT && outweight == OUTPUT_STANDARD_DEVIATION ) {
		status = transform_weight_to_sigma (obs->weight);
		if (status != SUCCESS) {
			obs->indep->dim = 0; free (obs->indep->val); obs->indep->val = NULL;
			obs->dep->dim   = 0; free (obs->dep->val);   obs->dep->val   = NULL;
			return status;
		}
	}
	if ( weight == INPUT_UNAVAILABLE && outweight == OUTPUT_STANDARD_DEVIATION ) {
		if (obs->weight && obs->weight->dim == 0) {
			phoebe_vector_alloc (obs->weight, obs->dep->dim);
			phoebe_vector_pad (obs->weight, 0.01);
		}
	}
	if ( weight == INPUT_UNAVAILABLE && outweight == OUTPUT_STANDARD_WEIGHT ) {
		if (obs->weight && obs->weight->dim == 0) {
			phoebe_vector_alloc (obs->weight, obs->dep->dim);
			phoebe_vector_pad (obs->weight, 1.0);
		}
	}

	if (alias == YES) {
		if (outindep != OUTPUT_PHASE)
			phoebe_lib_error ("cannot alias HJD points, ignoring.\n");
		else {
			if (no_of_columns == 2)
				alias_phase_points (obs->indep, obs->dep, NULL, phmin, phmax);
			else
				alias_phase_points (obs->indep, obs->dep, obs->weight, phmin, phmax);
		}
	}

	/* Done! Let's get outta here!                                              */
	phoebe_debug ("leaving read_in_observational_data ()\n");
	return SUCCESS;
}

int read_in_wd_lci_parameters (WD_LCI_parameters *params, int MPAGE, int curve)
{
	/*
	 * This function reads out all variables that build up the LCI file.
	 * 
	 * Return values:
	 * 
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_UNINITIALIZED_CURVE
	 *   ERROR_UNSUPPORTED_MPAGE  
	 *   ERROR_INVALID_LDLAW
	 *   SUCCESS
	 */

	int status;
	int i;

	int lcno, rvno;

	const char *filter = NULL;
	const char *readout_str;
	bool readout_bool;

	PHOEBE_passband *passband;

	phoebe_get_parameter_value ("phoebe_lcno", &lcno);
	phoebe_get_parameter_value ("phoebe_rvno", &rvno);

	/*
	 * The MPAGE switch determines the type of a synthetic curve the program
	 * should compute:
	 *
	 *   MPAGE = 1 .. light curve
	 *   MPAGE = 2 .. radial velocity curve
	 *
	 * It is passed as argument to this function. First we check if the chosen
	 * curve is initialized and, if everything is ok, we set the params->MPAGE
	 * field to the passed value.
	 */

	if (MPAGE == 1)
		if (curve < 0 || curve > lcno-1)
			return ERROR_UNINITIALIZED_CURVE;
	if (MPAGE == 2)
		if (curve < 0 || curve > rvno-1)
			return ERROR_UNINITIALIZED_CURVE;
	if (MPAGE != 1 && MPAGE != 2)
		return ERROR_UNSUPPORTED_MPAGE;

	params->MPAGE = MPAGE;

	phoebe_get_parameter_value ("phoebe_reffect_switch", &readout_bool);
	if (readout_bool == YES) params->MREF = 2; else params->MREF = 1;
	phoebe_get_parameter_value ("phoebe_reffect_reflections", &(params->NREF));

	phoebe_get_parameter_value ("phoebe_spots_move1", &readout_bool);
	if (readout_bool) params->IFSMV1 = 1; else params->IFSMV1 = 0;
	phoebe_get_parameter_value ("phoebe_spots_move2", &readout_bool);
	if (readout_bool) params->IFSMV2 = 1; else params->IFSMV2 = 0;
	phoebe_get_parameter_value ("phoebe_proximity_rv1_switch", &readout_bool);
	if (readout_bool) params->ICOR1  = 1; else params->ICOR1  = 0;
	phoebe_get_parameter_value ("phoebe_proximity_rv2_switch", &readout_bool);
	if (readout_bool) params->ICOR2  = 1; else params->ICOR2  = 0;

	status = get_ld_model_id (&(params->LD));
	if (status != SUCCESS) return status;

	phoebe_get_parameter_value ("phoebe_indep", &readout_str);
	if (strcmp (readout_str, "Time (HJD)")  == 0) params->JDPHS = 1;
	if (strcmp (readout_str, "Phase") == 0) params->JDPHS = 2;

	phoebe_get_parameter_value ("phoebe_hjd0", &(params->HJD0));
	phoebe_get_parameter_value ("phoebe_period", &(params->PERIOD));
	phoebe_get_parameter_value ("phoebe_dpdt", &(params->DPDT));
	phoebe_get_parameter_value ("phoebe_pshift", &(params->PSHIFT));

	phoebe_get_parameter_value ("phoebe_synscatter_switch", &readout_bool);
	if (readout_bool) {
		phoebe_get_parameter_value ("phoebe_synscatter_sigma", &(params->SIGMA));
		phoebe_get_parameter_value ("phoebe_synscatter_levweight", &readout_str);
		if (strcmp (readout_str, "No level-dependent weighting") == 0) params->WEIGHTING = 0;
		if (strcmp (readout_str, "Poissonian scatter")           == 0) params->WEIGHTING = 1;
		if (strcmp (readout_str, "Low light scatter")            == 0) params->WEIGHTING = 2;
		phoebe_get_parameter_value ("phoebe_synscatter_seed", &(params->SEED));
	}
	else {
		params->SIGMA = 0.0; params->WEIGHTING = 0; params->SEED = 0.0;
	}

	/* The following parameters are dummies, since PHOEBE calls WD algorithm    */
	/* for each phase individually:                                             */
	params->HJDST  = 0.0;
	params->HJDSP  = 1.0;
	params->HJDIN  = 0.1;
	params->PHSTRT = 0.0;
	params->PHSTOP = 1.0;
	params->PHIN   = 0.1;

	/* Normalizing magnitude (PHNORM) tells WD at what phase the normalized     */
	/* flux should be 1. Usually this is 0.25, but could in general be at some  */
	/* other phase. Since there is no support for that in PHOEBE yet, it is     */
	/* hardcoded to 0.25, but it should be changed.                             */
	params->PHNORM = 0.25;

	phoebe_get_parameter_value ("phoebe_model", &readout_str);
	if (strcmp (readout_str, "X-ray binary"                                         ) == 0) params->MODE = -1;
	if (strcmp (readout_str, "Unconstrained binary system"                          ) == 0) params->MODE =  0;
	if (strcmp (readout_str, "Overcontact binary of the W UMa type"                 ) == 0) params->MODE =  1;
	if (strcmp (readout_str, "Detached binary"                                      ) == 0) params->MODE =  2;
	if (strcmp (readout_str, "Overcontact binary not in thermal contact"            ) == 0) params->MODE =  3;
	if (strcmp (readout_str, "Semi-detached binary, primary star fills Roche lobe"  ) == 0) params->MODE =  4;
	if (strcmp (readout_str, "Semi-detached binary, secondary star fills Roche lobe") == 0) params->MODE =  5;
	if (strcmp (readout_str, "Double contact binary"                                ) == 0) params->MODE =  6;

	phoebe_get_parameter_value ("phoebe_msc1_switch", &(params->MSC1));
	phoebe_get_parameter_value ("phoebe_msc2_switch", &(params->MSC2));

	phoebe_get_parameter_value ("phoebe_compute_hla_switch", &readout_bool);
	if (readout_bool && lcno > 0) params->CALCHLA = 1; else params->CALCHLA = 0;
	phoebe_get_parameter_value ("phoebe_compute_vga_switch", &readout_bool);
	if (readout_bool && rvno > 0) params->CALCVGA = 1; else params->CALCVGA = 0;
	phoebe_get_parameter_value ("phoebe_asini_switch", &(params->ASINI));
	phoebe_get_parameter_value ("phoebe_cindex_switch", &(params->CINDEX));

	phoebe_get_parameter_value ("phoebe_usecla_switch", &readout_bool);
	if (readout_bool) params->IPB = 1; else params->IPB  = 0;
	phoebe_get_parameter_value ("phoebe_atm1_switch", &readout_bool);
	if (readout_bool) params->IFAT1 = 1; else params->IFAT1 = 0;
	phoebe_get_parameter_value ("phoebe_atm2_switch", &readout_bool);
	if (readout_bool) params->IFAT2 = 1; else params->IFAT2 = 0;

	phoebe_get_parameter_value ("phoebe_grid_finesize1", &(params->N1));
	phoebe_get_parameter_value ("phoebe_grid_finesize2", &(params->N2));

	phoebe_get_parameter_value ("phoebe_perr0", &(params->PERR0));
	phoebe_get_parameter_value ("phoebe_dperdt", &(params->DPERDT));

	/* THE applies only to X-ray binaries, but it isn't supported yet.          */
	params->THE    = 0.0;

	/* VUNIT is the internal radial velocity unit. This should be calculated    */
	/* rather than determined by the user, however it isn't written yet. It is  */
	/* thus hardcoded to the most obvious number, that is 100 km/s.             */
	params->VUNIT  = 100.0;

	phoebe_get_parameter_value ("phoebe_ecc",      &(params->E));
	phoebe_get_parameter_value ("phoebe_sma",      &(params->SMA));
	phoebe_get_parameter_value ("phoebe_f1",       &(params->F1));
	phoebe_get_parameter_value ("phoebe_f2",       &(params->F2));
	phoebe_get_parameter_value ("phoebe_vga",      &(params->VGA));
	phoebe_get_parameter_value ("phoebe_incl",     &(params->INCL));
	phoebe_get_parameter_value ("phoebe_grb1",     &(params->GR1));
	phoebe_get_parameter_value ("phoebe_grb2",     &(params->GR2));
	phoebe_get_parameter_value ("phoebe_logg1",    &(params->LOGG1));
	phoebe_get_parameter_value ("phoebe_logg2",    &(params->LOGG2));
	phoebe_get_parameter_value ("phoebe_met1",     &(params->MET1));
	phoebe_get_parameter_value ("phoebe_met2",     &(params->MET2));
	phoebe_get_parameter_value ("phoebe_teff1",    &(params->TAVH));
	phoebe_get_parameter_value ("phoebe_teff2",    &(params->TAVC));   
	phoebe_get_parameter_value ("phoebe_alb1",     &(params->ALB1));
	phoebe_get_parameter_value ("phoebe_alb2",     &(params->ALB2));
	phoebe_get_parameter_value ("phoebe_pot1",     &(params->PHSV));
	phoebe_get_parameter_value ("phoebe_pot2",     &(params->PCSV));
	phoebe_get_parameter_value ("phoebe_rm",       &(params->RM));
	phoebe_get_parameter_value ("phoebe_ld_xbol1", &(params->XBOL1));
	phoebe_get_parameter_value ("phoebe_ld_xbol2", &(params->XBOL2));
	phoebe_get_parameter_value ("phoebe_ld_ybol1", &(params->YBOL1));
	phoebe_get_parameter_value ("phoebe_ld_ybol2", &(params->YBOL2));

	if (MPAGE == 2)
		phoebe_get_parameter_value ("phoebe_rv_filter", curve, &filter);
	else
		phoebe_get_parameter_value ("phoebe_lc_filter", curve, &filter);

	passband = phoebe_passband_lookup (filter);
	if (!passband) {
		phoebe_lib_warning ("passband not set or invalid, reverting to Johnson V.\n");
		params->IBAND = 7;
		params->WLA   = 550.0;
	} else {
		params->IBAND = get_passband_id (filter);
		params->WLA   = passband->effwl;
	}

	if (MPAGE == 1) {
		phoebe_get_parameter_value ("phoebe_hla", curve, &(params->HLA));
		phoebe_get_parameter_value ("phoebe_cla", curve, &(params->CLA));
	}
	else {
		/* HLAs and CLAs don't make any sense for RV curves, so we initialize     */
		/* them to some canonic dummy values.                                     */
		params->HLA = 1.0;
		params->CLA = 1.0;
	}

	if (MPAGE == 2) {
		phoebe_get_parameter_value ("phoebe_ld_rvx1", curve, &(params->X1A));
		phoebe_get_parameter_value ("phoebe_ld_rvx2", curve, &(params->X2A));
		phoebe_get_parameter_value ("phoebe_ld_rvy1", curve, &(params->Y1A));
		phoebe_get_parameter_value ("phoebe_ld_rvy2", curve, &(params->Y2A));
	}
	else {
		phoebe_get_parameter_value ("phoebe_ld_lcx1", curve, &(params->X1A));
		phoebe_get_parameter_value ("phoebe_ld_lcx2", curve, &(params->X2A));
		phoebe_get_parameter_value ("phoebe_ld_lcy1", curve, &(params->Y1A));
		phoebe_get_parameter_value ("phoebe_ld_lcy2", curve, &(params->Y2A));
	}

	/* Third light is extrinsic and will be computed later by PHOEBE:         */
	params->EL3 = 0.0;

	if (MPAGE == 1) {
/*		phoebe_get_parameter_value ("phoebe_el3", curve, &(params->EL3));*/
		phoebe_get_parameter_value ("phoebe_opsf", curve, &(params->OPSF));
	}
	else {
		/* Third light and opacity function don't make sense for RVs.             */
/*		params->EL3 = 0.0;*/
		params->OPSF = 0.0;
	}

	/* MZERO and FACTOR variables set offsets in synthetic light curves. PHOEBE */
	/* controls this by its own variables, so we hardcode these to 0 and 1.     */
	params->MZERO  = 0.0;
	params->FACTOR = 1.0;

	/* Spots:                                                                 */
	phoebe_get_parameter_value ("phoebe_spots_no1", &(params->SPRIM));
	phoebe_get_parameter_value ("phoebe_spots_no2", &(params->SSEC));

	/* Allocate memory for spot parameters and fill in the values:            */
	if (params->SPRIM != 0) {
		params->XLAT1  = phoebe_malloc (params->SPRIM * sizeof (*(params->XLAT1)));
		params->XLONG1 = phoebe_malloc (params->SPRIM * sizeof (*(params->XLONG1)));
		params->RADSP1 = phoebe_malloc (params->SPRIM * sizeof (*(params->RADSP1)));
		params->TEMSP1 = phoebe_malloc (params->SPRIM * sizeof (*(params->TEMSP1)));

		for (i = 0; i < params->SPRIM; i++) {
			phoebe_get_parameter_value ("phoebe_spots_lat1",  i, &(params->XLAT1[i]));
			phoebe_get_parameter_value ("phoebe_spots_long1", i, &(params->XLONG1[i]));
			phoebe_get_parameter_value ("phoebe_spots_rad1",  i, &(params->RADSP1[i]));
			phoebe_get_parameter_value ("phoebe_spots_temp1", i, &(params->TEMSP1[i]));
		}
	}
	if (params->SSEC != 0) {
		params->XLAT2  = phoebe_malloc (params->SSEC * sizeof (*(params->XLAT2)));
		params->XLONG2 = phoebe_malloc (params->SSEC * sizeof (*(params->XLONG2)));
		params->RADSP2 = phoebe_malloc (params->SSEC * sizeof (*(params->RADSP2)));
		params->TEMSP2 = phoebe_malloc (params->SSEC * sizeof (*(params->TEMSP2)));

		for (i = 0; i < params->SSEC; i++) {
			phoebe_get_parameter_value ("phoebe_spots_lat2",  i, &(params->XLAT2[i]));
			phoebe_get_parameter_value ("phoebe_spots_long2", i, &(params->XLONG2[i]));
			phoebe_get_parameter_value ("phoebe_spots_rad2",  i, &(params->RADSP2[i]));
			phoebe_get_parameter_value ("phoebe_spots_temp2", i, &(params->TEMSP2[i]));
		}
	}

	return SUCCESS;
}

int read_in_wd_dci_parameters (WD_DCI_parameters *params, int *marked_tba)
{
	/*
	 * This function reads in DCI parameters. Be sure to allocate (and free)
	 * the passed WD_DCI_parameters structure (and its substructures).
	 *
	 * Supported error states:
	 *
	 *  ERROR_INVALID_INDEP
	 *  ERROR_INVALID_DEP
	 *  ERROR_INVALID_WEIGHT
	 *  ERROR_MINIMIZER_NO_CURVES
	 *  ERROR_MINIMIZER_NO_PARAMS
	 *  ERROR_INVALID_DATA
	 *  SUCCESS
	 */

	char *pars[] = {
		"phoebe_spots_lat1",
		"phoebe_spots_long1",
		"phoebe_spots_rad1",
		"phoebe_spots_temp1",
		"phoebe_spots_lat2",
		"phoebe_spots_long2",
		"phoebe_spots_rad2",
		"phoebe_spots_temp2",
		"phoebe_sma",
		"phoebe_ecc",
		"phoebe_perr0",
		"phoebe_f1",
		"phoebe_f2",
		"phoebe_pshift",
		"phoebe_vga",
		"phoebe_incl",
		"phoebe_grb1",
		"phoebe_grb2",
		"phoebe_teff1",
		"phoebe_teff2",
		"phoebe_alb1",
		"phoebe_alb2",
		"phoebe_pot1",
		"phoebe_pot2",
		"phoebe_rm",
		"phoebe_hjd0",
		"phoebe_period",
		"phoebe_dpdt",
		"phoebe_dperdt",
		"",
		"phoebe_hla",
		"phoebe_cla",
		"phoebe_ld_lcx1",
		"phoebe_ld_lcx2",
		"phoebe_el3"
	};

	int i, status;
	int lcno, rvno, cno;
	bool readout_bool;
	const char *readout_str;

	int rv1index = -1;
	int rv2index = -1;

	/* DC features 35 adjustable parameters and we initialize such arrays:    */
	bool    *tba;
	double *step;

	phoebe_get_parameter_value ("phoebe_lcno", &lcno);
	phoebe_get_parameter_value ("phoebe_rvno", &rvno);
	cno = lcno + rvno;

	*marked_tba = 0;

	/* Check whether there are any experimental data present:                 */
	if (cno == 0) return ERROR_MINIMIZER_NO_CURVES;

	/* Allocate memory:                                                       */
	 tba = phoebe_malloc (35 * sizeof ( *tba));
	step = phoebe_malloc (35 * sizeof (*step));

	/* Read in TBA states and step-sizes; note the '!' in front of the tba    */
	/* readout function; that's because WD uses 1 for false and 0 for true.   */
	for (i = 0; i < 35; i++) {
		if (i == 29) { tba[i] = !FALSE; continue; } /* reserved WD channel */
		phoebe_get_parameter_tba (pars[i], &(tba[i])); tba[i] = !tba[i];
		phoebe_get_parameter_step (pars[i], &step[i]);
		if (i > 29)
			*marked_tba += lcno * (1-tba[i]);
		else
			*marked_tba += 1-tba[i];
	}

	params->tba  = tba;
	params->step = step;

	/* Check the presence of RV and LC data:                                  */
	{
	PHOEBE_input_dep dep;
	params->rv1data = FALSE; params->rv2data = FALSE;
	for (i = 0; i < rvno; i++) {
		phoebe_get_parameter_value ("phoebe_rv_dep", i, &readout_str);
		get_input_dependent_variable (readout_str, &dep);
		if (dep == INPUT_PRIMARY_RV) {
			params->rv1data = TRUE;
			rv1index = i;
		}
		if (dep == INPUT_SECONDARY_RV) {
			params->rv2data = TRUE;
			rv2index = i;
		}
		if (dep == -1) return ERROR_INVALID_DEP;
	}
	params->nlc = lcno;
	}

	/* DC-related parameters:                                                 */
	phoebe_get_parameter_value ("phoebe_dc_lambda", &(params->dclambda));
	phoebe_get_parameter_value ("phoebe_dc_symder_switch", &(params->symder));
	phoebe_get_parameter_value ("phoebe_grid_coarsesize1", &(params->n1c));
	phoebe_get_parameter_value ("phoebe_grid_coarsesize2", &(params->n2c));
	phoebe_get_parameter_value ("phoebe_grid_finesize1", &(params->n1f));
	phoebe_get_parameter_value ("phoebe_grid_finesize2", &(params->n2f));

	/* Reflection effect:                                                       */
	phoebe_get_parameter_value ("phoebe_reffect_switch", &readout_bool);
	if (readout_bool) params->refswitch = 2; else params->refswitch = 1;
	phoebe_get_parameter_value ("phoebe_reffect_reflections", &(params->refno));

	/* Eclipse/proximity effect:                                                */
	phoebe_get_parameter_value ("phoebe_proximity_rv1_switch", &(params->rv1proximity));
	phoebe_get_parameter_value ("phoebe_proximity_rv2_switch", &(params->rv2proximity));

	/* Limb darkening effect:                                                   */
	status = get_ld_model_id (&(params->ldmodel));
	if (status != SUCCESS) return status;

	/* Morphological constraint:                                                */
	{
	const char *mode;
	phoebe_get_parameter_value ("phoebe_model", &mode);
	params->morph = -2;
	if (strcmp (mode, "X-ray binary"                                         ) == 0) params->morph = -1;
	if (strcmp (mode, "Unconstrained binary system"                          ) == 0) params->morph =  0;
	if (strcmp (mode, "Overcontact binary of the W UMa type"                 ) == 0) params->morph =  1;
	if (strcmp (mode, "Detached binary"                                      ) == 0) params->morph =  2;
	if (strcmp (mode, "Overcontact binary not in thermal contact"            ) == 0) params->morph =  3;
	if (strcmp (mode, "Semi-detached binary, primary star fills Roche lobe"  ) == 0) params->morph =  4;
	if (strcmp (mode, "Semi-detached binary, secondary star fills Roche lobe") == 0) params->morph =  5;
	if (strcmp (mode, "Double contact binary"                                ) == 0) params->morph =  6;
	if (params->morph == -2) return ERROR_INVALID_MODEL;
	}

	/* Independent fit variable:                                                */
	{
	const char *indep;
	phoebe_get_parameter_value ("phoebe_indep", &indep);
	params->indep = 0;
	if (strcmp (indep, "Time (HJD)") == 0) params->indep = 1;
	if (strcmp (indep, "Phase") == 0) params->indep = 2;
	if (params->indep == 0) return ERROR_INVALID_INDEP;
	}

	/* Luminosity decoupling:                                                   */
	phoebe_get_parameter_value ("phoebe_usecla_switch", &(params->cladec));

	/* Model atmosphere switches:                                               */
	phoebe_get_parameter_value ("phoebe_atm1_switch", &(params->ifat1));
	phoebe_get_parameter_value ("phoebe_atm2_switch", &(params->ifat2));

	/* Model parameters:                                                        */
	phoebe_get_parameter_value ("phoebe_hjd0",     &(params->hjd0));
	phoebe_get_parameter_value ("phoebe_period",   &(params->period));
	phoebe_get_parameter_value ("phoebe_dpdt",     &(params->dpdt));
	phoebe_get_parameter_value ("phoebe_pshift",   &(params->pshift));
	phoebe_get_parameter_value ("phoebe_perr0",    &(params->perr0));
	phoebe_get_parameter_value ("phoebe_dperdt",   &(params->dperdt));
	phoebe_get_parameter_value ("phoebe_ecc",      &(params->ecc));
	phoebe_get_parameter_value ("phoebe_sma",      &(params->sma));
	phoebe_get_parameter_value ("phoebe_f1",       &(params->f1));
	phoebe_get_parameter_value ("phoebe_f2",       &(params->f2));
	phoebe_get_parameter_value ("phoebe_vga",      &(params->vga));
	phoebe_get_parameter_value ("phoebe_incl",     &(params->incl));
	phoebe_get_parameter_value ("phoebe_grb1",     &(params->grb1));
	phoebe_get_parameter_value ("phoebe_grb2",     &(params->grb2));
	phoebe_get_parameter_value ("phoebe_met1",     &(params->met1));
	phoebe_get_parameter_value ("phoebe_teff1",    &(params->teff1));
	phoebe_get_parameter_value ("phoebe_teff2",    &(params->teff2));
	phoebe_get_parameter_value ("phoebe_alb1",     &(params->alb1));
	phoebe_get_parameter_value ("phoebe_alb2",     &(params->alb2));
	phoebe_get_parameter_value ("phoebe_pot1",     &(params->pot1));
	phoebe_get_parameter_value ("phoebe_pot2",     &(params->pot2));
	phoebe_get_parameter_value ("phoebe_rm",       &(params->rm));
	phoebe_get_parameter_value ("phoebe_ld_xbol1", &(params->xbol1));
	phoebe_get_parameter_value ("phoebe_ld_xbol2", &(params->xbol2));
	phoebe_get_parameter_value ("phoebe_ld_ybol1", &(params->ybol1));
	phoebe_get_parameter_value ("phoebe_ld_ybol2", &(params->ybol2));

	/* Passband-dependent parameters:                                           */
	{
	int index;
	PHOEBE_passband *passband;

	params->passband   = phoebe_malloc (cno * sizeof (*(params->passband)));
	params->wavelength = phoebe_malloc (cno * sizeof (*(params->wavelength)));
	params->sigma      = phoebe_malloc (cno * sizeof (*(params->sigma)));
	params->hla        = phoebe_malloc (cno * sizeof (*(params->hla)));
	params->cla        = phoebe_malloc (cno * sizeof (*(params->cla)));
	params->x1a        = phoebe_malloc (cno * sizeof (*(params->x1a)));
	params->y1a        = phoebe_malloc (cno * sizeof (*(params->y1a)));
	params->x2a        = phoebe_malloc (cno * sizeof (*(params->x2a)));
	params->y2a        = phoebe_malloc (cno * sizeof (*(params->y2a)));
	params->el3        = phoebe_malloc (cno * sizeof (*(params->el3)));
	params->opsf       = phoebe_malloc (cno * sizeof (*(params->opsf)));
	params->levweight  = phoebe_malloc (cno * sizeof (*(params->levweight)));

	for (i = 0; i < rvno; i++) {
		PHOEBE_input_dep dep;
		phoebe_get_parameter_value ("phoebe_rv_dep", i, &readout_str);
		get_input_dependent_variable (readout_str, &dep);
		if (dep == INPUT_SECONDARY_RV && rvno == 2) index = 1; else index = 0;

		phoebe_get_parameter_value ("phoebe_rv_filter", i, &readout_str);
		passband = phoebe_passband_lookup (readout_str);
		params->passband[index]   = get_passband_id (readout_str);
		params->wavelength[index] = passband->effwl;

		phoebe_get_parameter_value ("phoebe_rv_sigma", i, &(params->sigma[index]));
		params->hla[index]        = 10.0;
		params->cla[index]        = 10.0;
		phoebe_get_parameter_value ("phoebe_ld_rvx1", i, &(params->x1a[index]));
		phoebe_get_parameter_value ("phoebe_ld_rvy1", i, &(params->y1a[index]));
		phoebe_get_parameter_value ("phoebe_ld_rvx2", i, &(params->x2a[index]));
		phoebe_get_parameter_value ("phoebe_ld_rvy2", i, &(params->y2a[index]));
		params->el3[index]        = 0.0;
		params->opsf[index]       = 0.0;
		params->levweight[index]  = 0;
	}

	for (i = rvno; i < cno; i++) {
		phoebe_get_parameter_value ("phoebe_lc_filter", i-rvno, &readout_str);
		passband = phoebe_passband_lookup (readout_str);
		params->passband[i]   = get_passband_id (readout_str);
		params->wavelength[i] = passband->effwl;
		phoebe_get_parameter_value ("phoebe_lc_sigma", i-rvno, &(params->sigma[i]));
		phoebe_get_parameter_value ("phoebe_hla", i-rvno, &(params->hla[i]));
		phoebe_get_parameter_value ("phoebe_cla", i-rvno, &(params->cla[i]));
		phoebe_get_parameter_value ("phoebe_ld_lcx1", i-rvno, &(params->x1a[i]));
		phoebe_get_parameter_value ("phoebe_ld_lcy1", i-rvno, &(params->y1a[i]));
		phoebe_get_parameter_value ("phoebe_ld_lcx2", i-rvno, &(params->x2a[i]));
		phoebe_get_parameter_value ("phoebe_ld_lcy2", i-rvno, &(params->y2a[i]));
		phoebe_get_parameter_value ("phoebe_el3", i-rvno, &(params->el3[i]));
		phoebe_get_parameter_value ("phoebe_opsf", i-rvno, &(params->opsf[i]));
		phoebe_get_parameter_value ("phoebe_lc_levweight", i-rvno, &readout_str);
		params->levweight[i]  = get_level_weighting_id (readout_str);
	}
	}

	/* Spot parameters:                                                       */
	phoebe_get_parameter_value ("phoebe_spots_no1", &(params->spot1no));
	phoebe_get_parameter_value ("phoebe_spots_no2", &(params->spot2no));
	phoebe_get_parameter_value ("phoebe_dc_spot1src", &(params->spot1src));
	phoebe_get_parameter_value ("phoebe_dc_spot2src", &(params->spot2src));
	phoebe_get_parameter_value ("phoebe_dc_spot1id", &(params->spot1id));
	phoebe_get_parameter_value ("phoebe_dc_spot2id", &(params->spot2id));
	phoebe_get_parameter_value ("phoebe_spots_move1", &(params->spots1move));
	phoebe_get_parameter_value ("phoebe_spots_move2", &(params->spots2move));

	params->spot1lat  = phoebe_malloc (params->spot1no * sizeof (*(params->spot1lat)));
	params->spot1long = phoebe_malloc (params->spot1no * sizeof (*(params->spot1long)));
	params->spot1rad  = phoebe_malloc (params->spot1no * sizeof (*(params->spot1rad)));
	params->spot1temp = phoebe_malloc (params->spot1no * sizeof (*(params->spot1temp)));
	params->spot2lat  = phoebe_malloc (params->spot2no * sizeof (*(params->spot2lat)));
	params->spot2long = phoebe_malloc (params->spot2no * sizeof (*(params->spot2long)));
	params->spot2rad  = phoebe_malloc (params->spot2no * sizeof (*(params->spot2rad)));
	params->spot2temp = phoebe_malloc (params->spot2no * sizeof (*(params->spot2temp)));

	for (i = 0; i < params->spot1no; i++) {
		phoebe_get_parameter_value ("phoebe_spots_lat1", i, &(params->spot1lat[i]));
		phoebe_get_parameter_value ("phoebe_spots_long1", i, &(params->spot1long[i]));
		phoebe_get_parameter_value ("phoebe_spots_rad1", i, &(params->spot1rad[i]));
		phoebe_get_parameter_value ("phoebe_spots_temp1", i, &(params->spot1temp[i]));
	}
	for (i = 0; i < params->spot2no; i++) {
		phoebe_get_parameter_value ("phoebe_spots_lat2", i, &(params->spot2lat[i]));
		phoebe_get_parameter_value ("phoebe_spots_long2", i, &(params->spot2long[i]));
		phoebe_get_parameter_value ("phoebe_spots_rad2", i, &(params->spot2rad[i]));
		phoebe_get_parameter_value ("phoebe_spots_temp2", i, &(params->spot2temp[i]));
	}

	/* Observational data:                                                      */
	{
	int indep;

	/* Allocate observational data arrays:                                    */
	params->obs = phoebe_malloc (cno * sizeof (*(params->obs)));

	/* Initialize individual data arrays to receive data:                     */
	for (i = 0; i < cno; i++)
		params->obs[i]  = phoebe_curve_new ();

	if (params->indep == 1) indep = OUTPUT_HJD; else indep = OUTPUT_PHASE;

	if (params->rv1data) {
		int status;
		PHOEBE_input_indep  varindep;
		PHOEBE_input_dep    vardep;
		PHOEBE_input_weight varweight;

		phoebe_get_parameter_value ("phoebe_rv_indep", rv1index, &readout_str);
		status = get_input_independent_variable (readout_str, &varindep);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_rv_dep", rv1index, &readout_str);
		status = get_input_dependent_variable (readout_str, &vardep);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_rv_indweight", rv1index, &readout_str);
		status = get_input_weight (readout_str, &varweight);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_rv_filename", rv1index, &readout_str);
		status = read_in_observational_data
			(
			readout_str,
			params->obs[0],
			varindep,
			indep,
			vardep,
			OUTPUT_PRIMARY_RV,
			varweight,
			OUTPUT_STANDARD_WEIGHT,
			NO,
			-0.5,
			+0.5
			);

		if (status != SUCCESS) return status;
	}
	if (params->rv2data) {
		int status, index;
		PHOEBE_input_indep  varindep;
		PHOEBE_input_dep    vardep;
		PHOEBE_input_weight varweight;

		phoebe_get_parameter_value ("phoebe_rv_indep", rv2index, &readout_str);
		status = get_input_independent_variable (readout_str, &varindep);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_rv_dep", rv2index, &readout_str);
		status = get_input_dependent_variable (readout_str, &vardep);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_rv_indweight", rv2index, &readout_str);
		status = get_input_weight (readout_str, &varweight);
		if (status != SUCCESS) return status;

		if (params->rv1data) index = 1; else index = 0;

		phoebe_get_parameter_value ("phoebe_rv_filename", rv2index, &readout_str);
		status = read_in_observational_data
			(
			readout_str,
			params->obs[index],
			varindep,
			indep,
			vardep,
			OUTPUT_SECONDARY_RV,
			varweight,
			OUTPUT_STANDARD_WEIGHT,
			NO,
			-0.5,
			+0.5
			);

		if (status != SUCCESS) return status;
	}
	for (i = rvno; i < cno; i++) {
		int status;
		PHOEBE_input_indep  varindep;
		PHOEBE_input_dep    vardep;
		PHOEBE_input_weight varweight;

		phoebe_get_parameter_value ("phoebe_lc_indep", i-rvno, &readout_str);
		status = get_input_independent_variable (readout_str, &varindep);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_lc_dep", i-rvno, &readout_str);
		status = get_input_dependent_variable (readout_str, &vardep);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_lc_indweight", i-rvno, &readout_str);
		status = get_input_weight (readout_str, &varweight);
		if (status != SUCCESS) return status;

		phoebe_get_parameter_value ("phoebe_lc_filename", i-rvno, &readout_str);
		status = read_in_observational_data
			(
			readout_str,
			params->obs[i],
			varindep,
			indep,
			vardep,
			OUTPUT_TOTAL_FLUX,
			varweight,
			OUTPUT_STANDARD_WEIGHT,
			NO,
			-0.5,
			+0.5
			);

		if (status != SUCCESS) return status;
	}
	}

	return SUCCESS;
	}

WD_DCI_parameters *wd_dci_parameters_new ()
{
	/*
	 * This function initializes the WD DCI structure for allocation.
	 */

	/* Allocate memory for the structure itself: */
	WD_DCI_parameters *pars = phoebe_malloc (sizeof (*pars));

	/* There are several parameters that determine the dimension of arrays.   */
	/* Set them all to 0 here:                                                */
	pars->rv1data    = 0;
	pars->rv2data    = 0;
	pars->nlc        = 0;
	pars->spot1no    = 0;
	pars->spot2no    = 0;

	/* NULLify structure arrays: */
	pars->tba        = NULL;
	pars->step       = NULL;
	pars->passband   = NULL;
	pars->wavelength = NULL;
	pars->sigma      = NULL;
	pars->hla        = NULL;
	pars->cla        = NULL;
	pars->x1a        = NULL;
	pars->y1a        = NULL;
	pars->x2a        = NULL;
	pars->y2a        = NULL;
	pars->el3        = NULL;
	pars->opsf       = NULL;
	pars->levweight  = NULL;
	pars->spot1lat   = NULL;
	pars->spot1long  = NULL;
	pars->spot1rad   = NULL;
	pars->spot1temp  = NULL;
	pars->spot2lat   = NULL;
	pars->spot2long  = NULL;
	pars->spot2rad   = NULL;
	pars->spot2temp  = NULL;

	/* Finally, these are arrays of curves. We NULLify them as well:          */
	pars->obs   = NULL;

	return pars;
}

int wd_dci_parameters_free (WD_DCI_parameters *params)
{
	/*
	 * This function frees the arrays of the structure and the structure itself.
	 */

	int i;
	int curve_no = params->rv1data + params->rv2data + params->nlc;

	if (params->tba)  free (params->tba);
	if (params->step) free (params->step);

	if (curve_no != 0) {
		if (params->passband)   free (params->passband);
		if (params->wavelength) free (params->wavelength);
		if (params->sigma)      free (params->sigma);
		if (params->hla)        free (params->hla);
		if (params->cla)        free (params->cla);
		if (params->x1a)        free (params->x1a);
		if (params->y1a)        free (params->y1a);
		if (params->x2a)        free (params->x2a);
		if (params->y2a)        free (params->y2a);
		if (params->el3)        free (params->el3);
		if (params->opsf)       free (params->opsf);
		if (params->levweight)  free (params->levweight);
	}

	if (params->spot1no != 0) {
		if (params->spot1lat)  free (params->spot1lat);
		if (params->spot1long) free (params->spot1long);
		if (params->spot1rad)  free (params->spot1rad);
		if (params->spot1temp) free (params->spot1temp);
	}

	if (params->spot2no != 0) {
		if (params->spot2lat)  free (params->spot2lat);
		if (params->spot2long) free (params->spot2long);
		if (params->spot2rad)  free (params->spot2rad);
		if (params->spot2temp) free (params->spot2temp);
	}

	if (curve_no != 0) {
		if (params->obs) {
			for (i = 0; i < curve_no; i++)
				phoebe_curve_free (params->obs[i]);
			free (params->obs);
		}
	}

	free (params);

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

	phoebe_get_parameter_value ("phoebe_hjd0", hjd0);
	phoebe_get_parameter_value ("phoebe_period", period);
	phoebe_get_parameter_value ("phoebe_dpdt", dpdt);
	phoebe_get_parameter_value ("phoebe_pshift", pshift);

	return SUCCESS;
}

int read_in_adjustable_parameters (int *tba, double **values, int **indices)
{
	/*
	 * This function cross-checks the global parameter table for all adjustable
	 * parameters, queries the adjustment switch state, step size and the
	 * initial parameter value; it then fills in the values to the list
	 * variable. The memory is allocated in this call and it is up to the user
	 * to free it after using it.
	 */

	int i, j, k;
	int status, calchla_idx, calcvga_idx, sma_idx, incl_idx;
	bool calchla, calcvga, asini;
	int lcno;

	phoebe_get_parameter_value ("phoebe_compute_hla_switch", &calchla);
	phoebe_get_parameter_value ("phoebe_compute_vga_switch", &calcvga);
	phoebe_get_parameter_value ("phoebe_asini_switch",   &asini);

	phoebe_get_parameter_value ("phoebe_lcno", &lcno);

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
			/* First, let's check whether the user tries something silly like ad-   */
			/* justing HLAs when they should be calculated:                         */

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

			/* Now we have to see whether it's a system parameter (one for all      */
			/* light curves) or is it a filter-dependent parameter (one for each    */
			/* light curve):                                                        */

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

			/* Next, we (re)allocate memory:                                        */
			*values   = phoebe_realloc (*values,   j * sizeof (**values));
			*indices  = phoebe_realloc (*indices,  j * sizeof (**indices));

			/* Finally, we fill in all values: if multi = 0, then only (j-1). field */
			/* is updated, otherwise all filter-dependent fields are updated:       */
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

	if (strcmp (type, "No level-dependent weighting") == 0) id = 0;
	if (strcmp (type, "Poissonian scatter")           == 0) id = 1;
	if (strcmp (type, "Low light scatter")            == 0) id = 2;

	if (id == -1)
		{
		phoebe_lib_error ("level weighting type invalid, assuming Poissonian scatter.\n");
		return 1;
		}
	return id;
	}
