#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_calculations.h"
#include "phoebe_configuration.h"
#include "phoebe_constraints.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fitting.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"
#include "phoebe_nms.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

#include "../libwd/wd.h"

int calls_to_cf = 0;

double phoebe_chi2_cost_function (PHOEBE_vector *adjpars, PHOEBE_nms_parameters *params)
{
	/*
	 * This function evaluates chi2 of the O-C difference and returns its value
	 * to the minimizer. The first passed argument is a simple vector of all
	 * parameters that are set for adjustment. The second parameter is the
	 * passed_parameters struct defined below. *pars points to the address of
	 * the WD_LCI_parameters struct defined in the main minimizer function,
	 * **pointers is an array of pointers to double that point to those ele-
	 * ments of *pars which are set for adjustment.
	 */

	int status, i;
	double cfval = 0.0;
	char *qualifier, *lcin;
	PHOEBE_parameter *par;
	int index;

	int                    lcno       = params->lcno;
	int                    rvno       = params->rvno;
	PHOEBE_el3_units       l3units    = params->l3units;
	PHOEBE_array          *qualifiers = params->qualifiers;
	PHOEBE_vector         *l_bounds   = params->l_bounds;
	PHOEBE_vector         *u_bounds   = params->u_bounds;
	PHOEBE_curve         **obs        = params->obs;
	PHOEBE_vector         *psigma     = params->psigma;
	PHOEBE_array          *lexp       = params->lexp;
	PHOEBE_vector         *chi2s      = params->chi2s;
	PHOEBE_vector         *levels     = params->levels;
	PHOEBE_vector         *l3         = params->l3;

	WD_LCI_parameters    **lcipars;

	calls_to_cf++;

	phoebe_debug ("  call to CF: #%02d:\n", calls_to_cf);

	/*
	 * Set the values in all parameter tables to match the current
	 * iteration values:
	 */

	printf ("  assigning new parameter values:\n");
	for (i = 0; i < qualifiers->dim; i++) {
		status = phoebe_qualifier_string_parse (qualifiers->val.strarray[i], &qualifier, &index);
		par = phoebe_parameter_lookup (qualifier);
		if (index == 0) {
			printf ("    %s: %lf -> %lf\n", par->qualifier, par->value.d, adjpars->val[i]);
			phoebe_parameter_set_value (par, adjpars->val[i]);
		}
		else {
			printf ("    %s[%d]: %lf -> %lf\n", par->qualifier, index, par->value.vec->val[index-1], adjpars->val[i]);
			phoebe_parameter_set_value (par, index-1, adjpars->val[i]);
		}
		free (qualifier);

		if (adjpars->val[i] < l_bounds->val[i]) {
			printf ("    %s out of bounds (%lf, %lf)\n", par->qualifier, l_bounds->val[i], u_bounds->val[i]);
			return 1e10*(1.0+l_bounds->val[i]-adjpars->val[i]);
		}
		if (adjpars->val[i] > u_bounds->val[i]) {
			printf ("    %s out of bounds (%lf, %lf)\n", par->qualifier, l_bounds->val[i], u_bounds->val[i]);
			return 1e10*(1.0+adjpars->val[i]-u_bounds->val[i]);
		}
	}

	/* Satisfy all the constraints: */
	phoebe_constraint_satisfy_all ();

	/* Read in model parameters for each curve/passband: */
	lcipars = phoebe_malloc ( (lcno+rvno) * sizeof (*lcipars));
	for (i = 0; i < lcno; i++) {
		lcipars[i] = phoebe_malloc (sizeof (**lcipars));
		wd_lci_parameters_get (lcipars[i], /* MPAGE = */ 1, i);
	}
	for (i = 0; i < rvno; i++) {
		lcipars[lcno+i] = phoebe_malloc (sizeof (**lcipars));
		wd_lci_parameters_get (lcipars[lcno+i], /* MPAGE = */ 2, i);
	}

	/* Compute theoretical light and RV curves: */
	for (i = 0; i < lcno + rvno; i++) {
		PHOEBE_curve *curve = phoebe_curve_new ();

		lcin = phoebe_create_temp_filename ("phoebe_lci_XXXXXX");
		create_lci_file (lcin, lcipars[i]);
		phoebe_compute_lc_using_wd (curve, obs[i]->indep, lcin);
		remove (lcin);
		free (lcin);

		if (params->autolevels && i < lcno) {
			double alpha;
			phoebe_calculate_plum_correction (&alpha, curve, obs[i], lcipars[i]->WEIGHTING, l3->val[i], l3units);
			phoebe_vector_multiply_by (curve->dep, 1./alpha);
			levels->val[i] = alpha;
			printf ("    alpha[%d] = %lf\n", i, alpha);
		}
		if (params->autogamma && i >= lcno) {
			phoebe_lib_warning ("automatic computation of gamma velocity pending on RV readouts.\n");
			/*
			double gamma;
			phoebe_calculate_gamma_correction (&gamma, curve, obs[i]);
			phoebe_vector_offset (curve->dep, gamma);
			printf ("    gamma[%d] = %lf\n", i, gamma);
			*/
		}

		phoebe_cf_compute (&(chi2s->val[i]), PHOEBE_CF_WITH_ALL_WEIGHTS, curve->dep, obs[i]->dep, obs[i]->weight, psigma->val[i], lexp->val.iarray[i], 1.0);
		cfval += chi2s->val[i];
		phoebe_curve_free (curve);
	}

	return cfval;
}

int phoebe_minimize_using_nms (FILE *nms_output, PHOEBE_minimizer_feedback *feedback)
{
	/*
	 * This is a N&M simplex function that we use for multi-D minimization. All
	 * parameters are passed through an array of structs.
	 *
	 * Error codes:
	 *
	 *  ERROR_QUALIFIER_NOT_FOUND
	 *  ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED
	 *  ERROR_MINIMIZER_INVALID_FILE
	 *  ERROR_MINIMIZER_NO_CURVES
	 *  ERROR_MINIMIZER_NO_PARAMS
	 *  ERROR_INVALID_INDEP
	 *  ERROR_INVALID_DEP
	 *  ERROR_INVALID_WEIGHT
	 *  SUCCESS
	 */

	PHOEBE_parameter_table *table;

	int status, i, j, index;
	char *readout_str;
	clock_t clock_start, clock_stop;
	PHOEBE_parameter_list *tba;
	PHOEBE_parameter *par;
	PHOEBE_column_type indep, rvdep;

	PHOEBE_nms_parameters *passed;

	double accuracy;
	int iter_max;

	int lcno, rvno;
	PHOEBE_el3_units l3units;
	PHOEBE_array *active_lcindices, *active_rvindices;
	char *qualifier;
	int dim_tba;
	PHOEBE_array *qualifiers;
	PHOEBE_vector *l_bounds, *u_bounds;
	PHOEBE_vector *psigma;
	PHOEBE_array  *lexp;
	PHOEBE_curve **obs;
	PHOEBE_vector *chi2s;
	PHOEBE_vector *levels;
	PHOEBE_vector *l3;

	/* NMS infrastructure: */
	int iter = 0;
	PHOEBE_vector *steps, *adjpars;
	PHOEBE_nms_simplex *simplex;
	double step, cfval;

	phoebe_debug ("entering downhill simplex minimizer.\n");

	/* Is the feedback structure initialized? */
	if (!feedback) return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	/* Is the feedback structure clear of any data? */
	if (feedback->qualifiers->dim != 0)
		return ERROR_MINIMIZER_FEEDBACK_ALREADY_ALLOCATED;

	feedback->algorithm = PHOEBE_MINIMIZER_NMS;
	
	/* Fire up the stop watch: */
	clock_start = clock ();

	/* Get the accuracy and maximum number of iterations */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_nms_accuracy"), &accuracy);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_nms_iters_max"), &iter_max);

	/* Get the number of active LC and RV curves: */
	active_lcindices = phoebe_active_curves_get (PHOEBE_CURVE_LC);
	active_rvindices = phoebe_active_curves_get (PHOEBE_CURVE_RV);
	if (active_lcindices) lcno = active_lcindices->dim; else lcno = 0;
	if (active_rvindices) rvno = active_rvindices->dim; else rvno = 0;

	if (lcno + rvno == 0) return ERROR_MINIMIZER_NO_CURVES;

	/* Copy PHOEBE spot parameters into WD spot structures: */
	status = wd_spots_parameters_get ();
	if (status != SUCCESS) return status;

	/* Get a list of parameters marked for adjustment: */
	tba = phoebe_parameter_list_get_marked_tba ();
	if (!tba) return ERROR_MINIMIZER_NO_PARAMS;

	/*
	 * Count the number of parameters marked for adjustment; dim_tba will hold
	 * the dimension of the subspace that will be adjusted rather than the
	 * actual number of parameters. It will also cross-check if any of the
	 * parameters marked for adjustment are constrained; if so, they will
	 * be removed from the queue.
	 */

	qualifiers = phoebe_array_new (TYPE_STRING_ARRAY);
	l_bounds   = phoebe_vector_new ();
	u_bounds   = phoebe_vector_new ();
	dim_tba = 0;

	while (tba) {
		if (tba->par->type == TYPE_DOUBLE) {
			qualifier = strdup (tba->par->qualifier);
			if (phoebe_qualifier_is_constrained (qualifier)) {
				phoebe_debug ("parameter %s is constrained, skipping.\n", qualifier);
				free (qualifier);
			}
			else {
				/* Add the qualifier to the list of qualifiers: */
				double lb, ub;
				dim_tba++;
				phoebe_array_realloc (qualifiers, dim_tba);
				phoebe_vector_realloc (l_bounds, dim_tba);
				phoebe_vector_realloc (u_bounds, dim_tba);
				qualifiers->val.strarray[dim_tba-1] = qualifier;
				phoebe_parameter_get_limits (tba->par, &lb, &ub);
				l_bounds->val[dim_tba-1] = lb; u_bounds->val[dim_tba-1] = ub;
			}
		}
		else /* tba->par->type == TYPE_DOUBLE_ARRAY */ {
			for (i = 0; i < lcno; i++) {
				qualifier = phoebe_malloc ((strlen(tba->par->qualifier)+3+(int)log(10*lcno))*sizeof(*qualifier));
				sprintf (qualifier, "%s[%d]", tba->par->qualifier, active_lcindices->val.iarray[i]+1);
				if (phoebe_qualifier_is_constrained (qualifier)) {
					printf ("parameter %s is constrained, skipping.\n", qualifier);
					free (qualifier);
				}
				else {
					/* Add the qualifier to the list of qualifiers: */
					double lb, ub;
					dim_tba++;
					phoebe_array_realloc (qualifiers, dim_tba);
					phoebe_vector_realloc (l_bounds, dim_tba);
					phoebe_vector_realloc (u_bounds, dim_tba);
					qualifiers->val.strarray[dim_tba-1] = qualifier;
					phoebe_parameter_get_limits (tba->par, &lb, &ub);
					l_bounds->val[dim_tba-1] = lb; u_bounds->val[dim_tba-1] = ub;
				}
			}
		}

		tba = tba->next;
	}

	if (dim_tba == 0) {
		phoebe_array_free (qualifiers);
		phoebe_vector_free (l_bounds);
		phoebe_vector_free (u_bounds);
		return ERROR_MINIMIZER_NO_PARAMS;
	}

	/* Create a copy of the parameter table and activate it: */
	table = phoebe_parameter_table_duplicate (PHOEBE_pt);
	phoebe_parameter_table_activate (table);

	phoebe_debug ("* parameters set for adjustment:\n");
	for (i = 0; i < dim_tba; i++)
		phoebe_debug ("  %d: %s\n", i+1, qualifiers->val.strarray[i]);

	/* Will the computation be done in HJD- or in phase-space? */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
	phoebe_column_get_type (&indep, readout_str);

	/* Read in the observed data and transform them appropriately: */
	obs = phoebe_malloc ( (lcno+rvno) * sizeof (*obs));
	for (i = 0; i < lcno; i++) {
		obs[i] = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, active_lcindices->val.iarray[i]);
		if (!obs[i]) {
			for (j = 0; j < i; j++)
				phoebe_curve_free (obs[j]);
			free (obs);
			return ERROR_FILE_NOT_FOUND;
		}
		phoebe_curve_transform (obs[i], indep, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_SIGMA);
	}

	for (i = 0; i < rvno; i++) {
		obs[lcno+i] = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, active_rvindices->val.iarray[i]);
		if (!obs[lcno+i]) {
			for (j = 0; j < lcno+i; j++)
				phoebe_curve_free (obs[j]);
			free (obs);
			return ERROR_FILE_NOT_FOUND;
		}
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), i, &readout_str);
		phoebe_column_get_type (&rvdep, readout_str);
		phoebe_curve_transform (obs[lcno+i], indep, rvdep, PHOEBE_COLUMN_SIGMA);
	}

	/* Read in passband sigma and level weights: */
	psigma = phoebe_vector_new ();            phoebe_vector_alloc (psigma, lcno+rvno);
	lexp = phoebe_array_new (TYPE_INT_ARRAY); phoebe_array_alloc (lexp, lcno+rvno);
	
	for (i = 0; i < lcno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_sigma"), active_lcindices->val.iarray[i], &(psigma->val[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_levweight"), active_lcindices->val.iarray[i], &readout_str);
		lexp->val.iarray[i] = intern_get_level_weighting_id (readout_str);
	}
	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), active_rvindices->val.iarray[i], &(psigma->val[lcno+i]));
		lexp->val.iarray[lcno+i] = 0;
	}
	
	/* Read in the vector of third light: */
	l3 = phoebe_vector_new (); phoebe_vector_alloc (l3, lcno);
	phoebe_el3_units_id (&l3units);
	for (i = 0; i < lcno; i++)
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"), active_lcindices->val.iarray[i], &(l3->val[i]));
	
	/* Initialize and allocate the vector of passband chi2 values: */
	chi2s = phoebe_vector_new ();
	phoebe_vector_alloc (chi2s, lcno+rvno);
	
	/* Populate the structure that will be passed to the cost function: */
	passed = phoebe_malloc (sizeof (*passed));

	passed->qualifiers = qualifiers;
	passed->l_bounds   = l_bounds;
	passed->u_bounds   = u_bounds;
	passed->psigma     = psigma;
	passed->lexp       = lexp;
	passed->lcno       = lcno;
	passed->rvno       = rvno;
	passed->obs        = obs;
	passed->chi2s      = chi2s;
	passed->l3         = l3;
	passed->l3units    = l3units;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_compute_hla_switch"), &(passed->autolevels));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_compute_vga_switch"), &(passed->autogamma));
	
	/* If levels are to be computed, allocate the suitable vector: */
	if (passed->autolevels) {
		levels = phoebe_vector_new ();
		phoebe_vector_alloc (levels, lcno);
	}
	else levels = NULL;
	passed->levels = levels;
	
	adjpars = phoebe_vector_new (); phoebe_vector_alloc (adjpars, dim_tba);
	steps   = phoebe_vector_new (); phoebe_vector_alloc (steps, dim_tba);
	
	/* Allocate memory for the feedback structure: */
	phoebe_minimizer_feedback_alloc (feedback, dim_tba, lcno+rvno, lcno);
	
	/* Read out initial values and step sizes: */
	for (i = 0; i < dim_tba; i++) {
		feedback->qualifiers->val.strarray[i] = strdup (qualifiers->val.strarray[i]);
		
		status = phoebe_qualifier_string_parse (qualifiers->val.strarray[i], &qualifier, &index);
		par = phoebe_parameter_lookup (qualifier);
		if (index == 0)
			phoebe_parameter_get_value (par, &(adjpars->val[i]));
		else
			phoebe_parameter_get_value (par, index-1, &(adjpars->val[i]));
		
		phoebe_parameter_get_step (par, &(steps->val[i]));
		feedback->initvals->val[i] = adjpars->val[i];
		free (qualifier);
	}

	/* Allocate the minimizer: */
	simplex = phoebe_nms_simplex_new ();
	phoebe_nms_simplex_alloc (simplex, dim_tba);

	phoebe_nms_simplex_set (simplex, phoebe_chi2_cost_function, passed, adjpars, &step, steps);

	do {
		iter++;
		status = phoebe_nms_simplex_iterate (simplex, phoebe_chi2_cost_function, passed, adjpars, &step, &cfval);
		if (status != SUCCESS) {
			printf ("no success, breaking!\n");
			break;
		}

		fprintf (nms_output, "%5d ", iter);
		for (i = 0; i < dim_tba; i++) {
			fprintf (nms_output, "%10.3e ", adjpars->val[i]);
		}
		fprintf (nms_output, "f() = %7.3f step = %.3f\n", cfval, step);
	} while (step > accuracy && iter < iter_max);

	/* Stop the clock watch: */
	clock_stop = clock ();

	/* Populate the feedback structure: */
	feedback->cputime   = (double) (clock_stop - clock_start) / CLOCKS_PER_SEC;
	feedback->iters     = iter;
	feedback->cfval     = cfval;
	for (i = 0; i < dim_tba; i++) {
		feedback->newvals->val[i] = adjpars->val[i];
		feedback->ferrors->val[i] = accuracy;
	}
	for (i = 0; i < lcno+rvno; i++) {
		feedback->chi2s->val[i] = chi2s->val[i];
		feedback->wchi2s->val[i] = chi2s->val[i];
	}
	for (i = 0; i < lcno; i++)
		feedback->__cla->val[i] = 10.0;

	/* There is no correlation matrix: */
	phoebe_matrix_free (feedback->cormat);
	feedback->cormat = NULL;

	/* Free observed data: */
	for (i = 0; i < lcno + rvno; i++)
		phoebe_curve_free (obs[i]);
	free (obs);

	/* Free the placeholder memory: */
	phoebe_vector_free (levels);
	phoebe_vector_free (chi2s);
	phoebe_vector_free (l3);
	phoebe_vector_free (l_bounds);
	phoebe_vector_free (u_bounds);
	phoebe_array_free (qualifiers);
	phoebe_vector_free (psigma);
	phoebe_array_free (lexp);
	phoebe_array_free (active_lcindices);
	phoebe_array_free (active_rvindices);

	/* Free the passed parameters structure: */
	free (passed);

	/* Free NMS-related parameters: */
	phoebe_nms_simplex_free (simplex);
	phoebe_vector_free (adjpars);
	phoebe_vector_free (steps);

	/* Free the parameter table: */
	phoebe_parameter_table_free (table);
	phoebe_parameter_table_activate (PHOEBE_pt_list->table);

	phoebe_debug ("leaving downhill simplex minimizer.\n");

	status = SUCCESS;
	return status;

/*
	double *initvals;
	int    *indices;
	int     cno, lcno, rvno;
	int     CALCHLA;
	int     CALCVGA;
	bool    ASINI;
	bool    color_constraint;
	int     CC;
	int     to_be_adjusted;
	double *weight;
	double *average;
	double *cindex;
*/
	/* An array of initial parameter values.        */
	/* An array of global table reference indices   */
	/* Curves (both LC and RV) number               */
	/* Should light levels be calculated or fitted  */
	/* Should gamma velocity be calculated          */
	/* A switch whether a \sin i should be constant */
	/* Color-constraint switch                      */
	/* Conditional constraining (CC) switch         */
	/* The number of adjustables to be adjusted     */
	/* Array of pointers to the WD_LCI_parameters   */
	/* An array of observational data               */
	/* Weights of passband curves                   */
	/* Average values of observations for HLA/VGA   */
	/* An array of color indices                    */
	/* A list of individual passband chi2 values    */
/*
	int curve, index, parindex;
	double parvalue;

	PHOEBE_column_type master_indep;

	const char *filename;
	char *passband;
	PHOEBE_passband *passband_ptr;
	PHOEBE_column_type itype, dtype, wtype;
	double sigma;

	size_t iter = 0;
	size_t i, k;
	int status;
	double size;
	double dpdt;
	const char *indep;
	const char *readout_str;

	phoebe_parameter_get_value ("phoebe_indep", &indep);
	phoebe_parameter_get_value ("phoebe_dpdt",  &dpdt);
*/
	/* Before we do anything, let's check whether the setup is sane: */
/*
	if (!hla_request_is_sane  ()) return ERROR_MINIMIZER_HLA_REQUEST_NOT_SANE;
	if (!vga_request_is_sane  ()) return ERROR_MINIMIZER_VGA_REQUEST_NOT_SANE;
	if (!dpdt_request_is_sane ()) return ERROR_MINIMIZER_DPDT_REQUEST_NOT_SANE;
*/
	/* Count the available curves: */
/*
	phoebe_parameter_get_value ("phoebe_lcno", &lcno);
	phoebe_parameter_get_value ("phoebe_rvno", &rvno);
	cno = lcno + rvno;
	phoebe_debug ("total number of curves (both LC and RV): %d\n", cno);

	if (cno == 0) return ERROR_MINIMIZER_NO_CURVES;
*/
	/*
	 * Check the filenames; we do this early because nothing is yet
	 * initialized and we don't have to worry about any memory leaks.
	 */
/*
	phoebe_debug ("checking whether everything's ok with filenames.\n");
	for (i = 0; i < lcno; i++) {
		phoebe_parameter_get_value ("phoebe_lc_filename", i, &readout_str);
		if (!phoebe_filename_exists (readout_str))
			return ERROR_MINIMIZER_INVALID_FILE;
	}
	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value ("phoebe_rv_filename", i, &readout_str);
		if (!phoebe_filename_exists (readout_str))
			return ERROR_MINIMIZER_INVALID_FILE;
	}
*/
	/* Verify the types of RV curves:                                         */
/*
	passed_pars.rv1 = FALSE; passed_pars.rv2 = FALSE;
	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value ("phoebe_rv_dep", i, &readout_str);
		status = phoebe_column_get_type (&dtype, readout_str);
		if (status != SUCCESS) return status;

		if (dtype == PHOEBE_COLUMN_PRIMARY_RV)   passed_pars.rv1 = TRUE;
		if (dtype == PHOEBE_COLUMN_SECONDARY_RV) passed_pars.rv2 = TRUE;
	}

	phoebe_debug ("RV1 curve is present: %d\n", passed_pars.rv1);
	phoebe_debug ("RV2 curve is present: %d\n", passed_pars.rv2);


	passed_pars.pars = phoebe_malloc ( cno * sizeof (*(passed_pars.pars)));
	weight           = phoebe_malloc ( cno * sizeof (*weight));
	average          = phoebe_malloc ( cno * sizeof (*average));
	cindex           = phoebe_malloc (lcno * sizeof (*cindex));


	CALCHLA          = params[0].CALCHLA;
	CALCVGA          = params[0].CALCVGA;
	CC               = params[0].MSC1 && params[0].MSC2;
	ASINI            = params[0].ASINI;
	color_constraint = params[0].CINDEX;

	if (CALCHLA == 1)
		phoebe_debug ("light levels (HLAs), if any, set to be calculated\n");
	else
		phoebe_debug ("light levels (HLAs), if any, set to be fitted\n");

	if (CALCVGA == 1)
		phoebe_debug ("gamma velocity (VGA), if applicable, set to be calculated\n");
	else
		phoebe_debug ("gamma velocity (VGA), if applicable, set to be fitted\n");

	if (ASINI == TRUE)
		phoebe_debug ("The value of a sin(i) is kept constant\n");

	phoebe_debug ("conditional constraint used: %d\n", CC);

	for (curve = 0; curve < cno; curve++) {
		k = 0;
*/
		/* Now comes the pointing part:                                           */
/*
		for (i = 0; i < to_be_adjusted; i++) {

			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_hjd0")    == 0) pointers[curve][i] = &(params[curve].HJD0);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_period")  == 0) pointers[curve][i] = &(params[curve].PERIOD);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_dpdt")    == 0) pointers[curve][i] = &(params[curve].DPDT);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_pshift")  == 0) pointers[curve][i] = &(params[curve].PSHIFT);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_sma")     == 0) pointers[curve][i] = &(params[curve].SMA);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_rm")      == 0) pointers[curve][i] = &(params[curve].RM);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_incl")    == 0) pointers[curve][i] = &(params[curve].INCL);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_vga")     == 0) pointers[curve][i] = &(params[curve].VGA);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_teff1")   == 0) pointers[curve][i] = &(params[curve].TAVH);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_teff2")   == 0) pointers[curve][i] = &(params[curve].TAVC);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_pot1")    == 0) pointers[curve][i] = &(params[curve].PHSV);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_pot2")    == 0) pointers[curve][i] = &(params[curve].PCSV);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_logg1")   == 0) pointers[curve][i] = &(params[curve].LOGG1);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_logg2")   == 0) pointers[curve][i] = &(params[curve].LOGG2);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_met1")    == 0) pointers[curve][i] = &(params[curve].MET1);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_met2")    == 0) pointers[curve][i] = &(params[curve].MET2);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_ecc")     == 0) pointers[curve][i] = &(params[curve].E);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_perr0")   == 0) pointers[curve][i] = &(params[curve].PERR0);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_dperdt")  == 0) pointers[curve][i] = &(params[curve].DPERDT);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_f1")      == 0) pointers[curve][i] = &(params[curve].F1);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_f2")      == 0) pointers[curve][i] = &(params[curve].F2);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_alb1")    == 0) pointers[curve][i] = &(params[curve].ALB1);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_alb2")    == 0) pointers[curve][i] = &(params[curve].ALB2);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_grb1")    == 0) pointers[curve][i] = &(params[curve].GR1);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_grb2")    == 0) pointers[curve][i] = &(params[curve].GR2);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_hla")     == 0) pointers[curve][i] = &(params[k++].HLA);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_cla")     == 0) pointers[curve][i] = &(params[k++].CLA);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_el3")     == 0) pointers[curve][i] = &(params[k++].EL3);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_opsf")    == 0) pointers[curve][i] = &(params[k++].OPSF);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_ld_lcx1") == 0) pointers[curve][i] = &(params[k++].X1A);
			if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_ld_lcx2") == 0) pointers[curve][i] = &(params[k++].X2A);

			if (k == lcno) k = 0;
		}

		passed_pars.pars[curve] = &params[curve];
		status = calculate_weighted_average (&(average[curve]), obs[curve]->dep, obs[curve]->weight);
	}
*/
	/* Fill in the structure the pointer to which will be passed to chi2      */
	/* function:                                                              */
/*
	passed_pars.to_be_adjusted   = to_be_adjusted;
	passed_pars.lcno             = lcno;
	passed_pars.rvno             = rvno;
	passed_pars.CALCHLA          = CALCHLA;
	passed_pars.CALCVGA          = CALCVGA;
	passed_pars.ASINI            = ASINI;
	passed_pars.color_constraint = color_constraint;
	passed_pars.CC               = CC;
	passed_pars.indices          = indices;
	passed_pars.pointers         = pointers;
	passed_pars.obs              = obs;
	passed_pars.weight           = weight;
	passed_pars.average          = average;
	passed_pars.cindex           = cindex;

*/
	/* The main iterating loop:                                               */
/*
	do {
		iter++;

		if (PHOEBE_INTERRUPT) break;

		status = gsl_multimin_fminimizer_iterate (s);
		if (status) break;

		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, accuracy);

		if (status == GSL_SUCCESS)
			fprintf (nms_output, "converged to minimum at\n");
		fprintf (nms_output, "%3d ", iter);
		for (i = 0; i < to_be_adjusted; i++)
			fprintf (nms_output, "%6.3lf ", gsl_vector_get (s->x, i));
		for (i = 0; i < lcno; i++)
			fprintf (nms_output, "L%d = %6.3f ", i+1, params[i].HLA);
		if (CC == 1)
			fprintf (nms_output, "T1 = %0.0f T2 = %0.0f q = %4.4f i = %4.4f a = %4.4f O1 = %4.4f O2 = %4.4f ", params[0].TAVH, params[0].TAVC, params[0].RM, params[0].INCL, params[0].SMA, params[0].PHSV, params[0].PCSV);
		if (ASINI == TRUE)
			fprintf (nms_output, "a = %4.4f i = %4.4f ", params[0].SMA, params[0].INCL);
		if (lcno != cno)
*/
			/* Print VGA only if there are RVs present */
/*
			fprintf (nms_output, "VGA = %3.3f ", params[cno-1].VGA);
		fprintf (nms_output, "chi2 = %10.3f step = %.3f\n", s->fval, size);
	} while (status == GSL_CONTINUE && iter < iter_no);

	if (PHOEBE_INTERRUPT) {
		PHOEBE_INTERRUPT = FALSE;

		gsl_multimin_fminimizer_free (s);

		gsl_vector_free (adjpars);
		gsl_vector_free (step_size);

		for (i = 0; i < cno; i++) {
			free (pointers[i]);
			phoebe_curve_free (obs[i]);
		}

		free (initvals);
		free (indices);

		free (weight);
		free (average);
		free (cindex);

		free (pointers);
		free (params);
		free (obs);

		return ERROR_SIGINT;
	}

	phoebe_debug ("GSL exit status: %d\n", status);
	if (status != GSL_SUCCESS && iter == iter_no) status = SUCCESS;
*/
	/* Allocate the feedback structure and fill in its contents:              */
/*
	phoebe_debug ("allocated fields for feedback: %d + %d*%d + %d = %d\n", to_be_adjusted, CALCHLA, lcno, CALCVGA, to_be_adjusted+CALCHLA*lcno+CALCVGA);
	status = phoebe_minimizer_feedback_alloc (feedback, to_be_adjusted+CALCHLA*lcno+CALCVGA, cno);
	if (status != SUCCESS) return status;

	feedback->iters     = iter;
	feedback->cfval     = s->fval;

	index = 0;
*/
	/* If HLA's were calculated (rather than fitted), update their values:    */
/*
	if (CALCHLA == 1) {
		phoebe_index_from_qualifier (&parindex, "phoebe_hla");
		for (i = 0; i < lcno; i++) {
			phoebe_parameter_get_value ("phoebe_hla", i, &parvalue);
			feedback->qualifiers->val.strarray[index] = strdup ("phoebe_hla");
			feedback->initvals->val[index] = parvalue;
			feedback->newvals->val[index] = params[i].HLA;
			feedback->ferrors->val[index] = sqrt (-1);
#warning OBSOLETE
			feedback->indices->val.iarray[index] = parindex;
			index++;
		}
	}
*/
	/* We do the same with v_\gamma:                                          */
/*
	if (CALCVGA == 1) {
		phoebe_index_from_qualifier (&parindex, "phoebe_vga");
		phoebe_parameter_get_value ("phoebe_vga", &parvalue);

		feedback->qualifiers->val.strarray[index] = strdup ("phoebe_vga");
		feedback->initvals->val[index] = parvalue;
		feedback->newvals->val[index] = params[i].VGA;
		feedback->ferrors->val[index] = sqrt (-1);
#warning OBSOLETE
		feedback->indices->val.iarray[index] = parindex;
		index++;
	}

	for (i = 0; i < to_be_adjusted; i++) {
		feedback->qualifiers->val.strarray[index] = ;
		feedback->initvals->val[index]       = initvals[i];
		feedback->newvals->val[index]        = gsl_vector_get (s->x, i);
		feedback->ferrors->val[index]        = accuracy*gsl_vector_get (s->x, i);
#warning OBSOLETE
		feedback->indices->val.iarray[index] = indices[i];
		index++;
	}
*/
	/* Supply unweighted and passband-weighted chi2 values to the feedback:   */
/*
	{
		double total_weight = 0.0;
		for (i = 0; i < cno; i++) {
			total_weight += weight[i];
		}

		for (i = 0; i < cno; i++) {
			(*feedback)-> chi2s->val[i] = chi2s->val[i];
			(*feedback)->wchi2s->val[i] = weight[i]/total_weight * chi2s->val[i];
		}
	}
*/
	/* Let's clean up:                                                        */
/*

	free (initvals);
	free (indices);

	free (weight);
	free (average);
	free (cindex);

	free (pointers);
	free (params);
	free (obs);

*/
}

int phoebe_minimize_using_dc (FILE *dc_output, PHOEBE_minimizer_feedback *feedback)
{
	/**
	 * phoebe_minimize_using_dc:
	 * @dc_output: output file descriptor (where DC output goes to)
	 * @feedback: a placeholder for the minimization result
	 *
	 * Emloys WD's DC to compute the corrections to parameters marked for
	 * adjustment. Note that this function does not compute the resulting
	 * model, it only predicts the corrections to parameter values. DC is
	 * known to diverge if the starting point in the parameter space is too
	 * far from observations.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status, i, j;
	clock_t clock_start, clock_stop;
	char *basedir;
	char *atmcof, *atmcofplanck;
	WD_DCI_parameters *params;
	PHOEBE_parameter_list *marked_tba, *tba;
	int lcno, rvno, no_tba;
	bool calchla = FALSE, calcvga = FALSE;
	double spots_conversion_factor;
	PHOEBE_array *active_lcindices, *active_rvindices;

	/* Minimizer results: */
	double *corrections;
	double *errors;
	double *chi2s;
	double *cormat;
	double *__cla;
	double  cfval;

	PHOEBE_el3_units l3units;
	integer L3perc;

	phoebe_debug ("entering differential corrections minimizer.\n");

	/* Check if the feedback structure is initialized and not allocated: */
	if (!feedback)
		return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;
	if (feedback->qualifiers->dim != 0)
		return ERROR_MINIMIZER_FEEDBACK_ALREADY_ALLOCATED;

	feedback->algorithm = PHOEBE_MINIMIZER_DC;

	/* Assign the filenames for atmcof and atmcofplanck needed by WD: */
	phoebe_config_entry_get ("PHOEBE_BASE_DIR", &basedir);
	atmcof       = phoebe_concatenate_strings (basedir, "/wd/phoebe_atmcof.dat",       NULL);
	atmcofplanck = phoebe_concatenate_strings (basedir, "/wd/phoebe_atmcofplanck.dat", NULL);

	if (!phoebe_filename_exists (atmcof)) {
		free (atmcof); free (atmcofplanck);
		return ERROR_ATMCOF_NOT_FOUND;
	}
	if (!phoebe_filename_exists (atmcofplanck)) {
		free (atmcof); free (atmcofplanck);
		return ERROR_ATMCOFPLANCK_NOT_FOUND;
	}

	/* Everything seems to be ok. Fire up the stop watch: */
	clock_start = clock ();

	/* Copy PHOEBE spot parameters into WD spot structures: */
	status = wd_spots_parameters_get ();
	if (status != SUCCESS) return status;

	/* Get a list of parameters marked for adjustment: */
	marked_tba = phoebe_parameter_list_get_marked_tba ();
	if (!marked_tba)
		return ERROR_MINIMIZER_NO_PARAMS;

	/* Get a list of active light and RV curves: */
	active_lcindices = phoebe_active_curves_get (PHOEBE_CURVE_LC);
	active_rvindices = phoebe_active_curves_get (PHOEBE_CURVE_RV);
	if (active_lcindices) lcno = active_lcindices->dim; else lcno = 0;
	if (active_rvindices) rvno = active_rvindices->dim; else rvno = 0;

	/* Read in WD DCI parameters: */
	params = wd_dci_parameters_new ();
	status = wd_dci_parameters_get (params, &no_tba);
	if (status != SUCCESS) return status;

	/* Count the true number of RV curves: */
	rvno = params->rv1data + params->rv2data;

	/* Allocate memory for the results: */
	corrections = phoebe_malloc (no_tba * sizeof (*corrections));
	errors      = phoebe_malloc (no_tba * sizeof (*errors));
	chi2s       = phoebe_malloc ((lcno + rvno) * sizeof (*chi2s));
	cormat      = phoebe_malloc (no_tba * no_tba * sizeof (*cormat));
	__cla       = phoebe_malloc (lcno * sizeof (*__cla));

	/* Create the DCI file from the params variable: */
	status = create_dci_file ("dcin.active", params);
	if (status != SUCCESS)
		return status;

	/* Get third light units: */
	status = phoebe_el3_units_id (&l3units);
	if (status != SUCCESS)
		phoebe_lib_warning ("Third light units invalid, assuming defaults (flux units).\n");

	/* Assign a third light units switch for WD: */
	L3perc = 0;
	if (l3units == PHOEBE_EL3_UNITS_TOTAL_LIGHT)
		L3perc = 1;

	/* Run one DC iteration and store the results in the allocated arrays: */
	printf ("nph = %d; delph = %lf\n", params->nph, params->delph);
	
	wd_dc (atmcof, atmcofplanck, &L3perc, params->knobs, params->indeps, params->fluxes, params->weights, &(params->nph), &(params->delph), corrections, errors, chi2s, cormat, __cla, &cfval);
	
	/*
	 * Allocate the feedback structure and fill it in. The number of parameter
	 * fields equals the number of parameters marked for adjustment plus the
	 * number of light curves if HLAs are marked for computation plus 1 if VGA
	 * VGA is marked for computation.
	 */

	phoebe_minimizer_feedback_alloc (feedback, no_tba+(calchla*lcno)+calcvga, lcno+rvno, lcno);

	feedback->iters = 1;

	for (i = 0; i < rvno; i++)
		feedback->chi2s->val[i] = 10000.0*chi2s[i]/params->sigma[i]/params->sigma[i];
	for (i = rvno; i < lcno + rvno; i++)
		feedback->chi2s->val[i] = chi2s[i]/params->sigma[i]/params->sigma[i];

	/* Weighted chi2s are not handled yet, let's free the memory: */
	phoebe_vector_free (feedback->wchi2s);
	feedback->wchi2s = NULL;

	feedback->cfval = cfval;

	spots_conversion_factor = 1./phoebe_spots_units_to_wd_conversion_factor ();

	tba = phoebe_parameter_list_get_marked_tba (); i = 0;
	while (tba) {
		switch (tba->par->type) {
			case TYPE_DOUBLE:
				if (strcmp (tba->par->qualifier, "phoebe_teff1") == 0 ||
					strcmp (tba->par->qualifier, "phoebe_teff2") == 0) {
					corrections[i] *= 10000.0;
					errors[i] *= 10000.0;
				}
				else if (strcmp (tba->par->qualifier, "phoebe_vga") == 0) {
					corrections[i] *= 100.0;
					errors[i] *= 100.0;
				}
				else if ((spots_conversion_factor != 1.0) && ((strcmp (tba->par->qualifier, "wd_spots_lat1") == 0) ||
					(strcmp (tba->par->qualifier, "wd_spots_lat2") == 0) ||
					(strcmp (tba->par->qualifier, "wd_spots_lon1") == 0) ||
					(strcmp (tba->par->qualifier, "wd_spots_lon2") == 0) ||
					(strcmp (tba->par->qualifier, "wd_spots_rad1") == 0) ||
					(strcmp (tba->par->qualifier, "wd_spots_rad2") == 0))) {
					corrections[i] *= spots_conversion_factor;
					errors[i] *= spots_conversion_factor;
				}
				feedback->qualifiers->val.strarray[i] = strdup (tba->par->qualifier);
				phoebe_parameter_get_value (tba->par, &(feedback->initvals->val[i]));
				feedback->newvals->val[i] = feedback->initvals->val[i] + corrections[i];
				feedback->ferrors->val[i] = errors[i];

				/* Handle cyclic values: */
				if (strcmp (feedback->qualifiers->val.strarray[i], "phoebe_perr0") == 0)
					feedback->newvals->val[i] = fmod (2*M_PI+feedback->newvals->val[i], 2*M_PI);
				else if (strcmp (feedback->qualifiers->val.strarray[i], "phoebe_pshift") == 0) {
					if (feedback->newvals->val[i] < -0.5)
						feedback->newvals->val[i] += 1.0;
					else if (feedback->newvals->val[i] > 0.5)
						feedback->newvals->val[i] -= 1;
				}
				i++;
			break;
			case TYPE_DOUBLE_ARRAY:
				for (j = 0; j < lcno; j++) {
					feedback->qualifiers->val.strarray[i+j] = phoebe_malloc ((strlen(tba->par->qualifier)+5) * sizeof (char));
					sprintf (feedback->qualifiers->val.strarray[i+j], "%s[%d]", tba->par->qualifier, active_lcindices->val.iarray[j]+1);
					phoebe_parameter_get_value (tba->par, active_lcindices->val.iarray[j], &(feedback->initvals->val[i+j]));

					/* Correct for HLA/CLA overshoot in case el3 is in %: */
					if (strcmp (tba->par->qualifier, "phoebe_hla") == 0 && l3units == PHOEBE_EL3_UNITS_TOTAL_LIGHT)
						corrections[i+j] *= 1.0 - params->el3[j];
					if (strcmp (tba->par->qualifier, "phoebe_cla") == 0 && l3units == PHOEBE_EL3_UNITS_TOTAL_LIGHT)
						corrections[i+j] *= 1.0 - params->el3[j];

					feedback->newvals->val[i+j] = feedback->initvals->val[i+j] + corrections[i+j];
					feedback->ferrors->val[i+j] = errors[i+j];
				}
				i += lcno;
			break;
			default:
				phoebe_lib_error ("exception handler invoked in phoebe_minimize_using_dc (), please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		tba = tba->next;
	}

	for (i = 0; i < no_tba; i++)
		for (j = 0; j < no_tba; j++)
			feedback->cormat->val[i][j] = cormat[i*no_tba+j];

	for (i = 0; i < lcno; i++)
		feedback->__cla->val[i] = __cla[i];
	
	/* Free all the allocated structures: */
	phoebe_array_free (active_lcindices);
	phoebe_array_free (active_rvindices);
	wd_dci_parameters_free (params);
	free (corrections);
	free (errors);
	free (chi2s);
	free (cormat);
	free (__cla);

	/* Stop the clock watch and compute the total CPU time on the process: */
	clock_stop = clock ();
	feedback->cputime = (double) (clock_stop - clock_start) / CLOCKS_PER_SEC;

	phoebe_debug ("leaving differential corrections minimizer.\n");

	return SUCCESS;
}
