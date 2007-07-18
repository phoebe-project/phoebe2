#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../libwd/wd.h"

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_calculations.h"
#include "phoebe_constraints.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fitting.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"
#include "phoebe_nms.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

/* At this point we should check whether GSL is available; if it isn't, then  */
/* we don't have much to compile here.                                        */

#ifdef HAVE_LIBGSL
#ifndef PHOEBE_GSL_DISABLED

#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_siman.h>

gsl_rng *PHOEBE_randomizer;

int calls_to_cf = 0;

double intern_chi2_cost_function (PHOEBE_vector *adjpars, PHOEBE_nms_parameters *params)
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

	int i, j, k;
	double cfval;

	int                    no_tba  = params->no_tba;
	int                    dim_tba = params->dim_tba;
	int                    lcno    = params->lcno;
	int                    rvno    = params->rvno;
	PHOEBE_curve         **obs     = params->obs;
	WD_LCI_parameters    **pars    = params->pars;

	PHOEBE_parameter_list *tba     = params->tba;

	PHOEBE_ast_list *constraint;

	calls_to_cf++;

	printf ("Call to CF: #%02d:\n", calls_to_cf);
	printf ("Number of marked parameters:    %d\n", no_tba);
	printf ("Dimension of adjusted subspace: %d\n", dim_tba);
	for (i = 0; i < lcno+rvno; i++)
		printf ("curve %d: %s, %d data points.\n", i+1, obs[i]->passband->name, obs[i]->indep->dim);

	/*
	 * Set the values in all parameter tables to match the current
	 * iteration values:
	 */

	j = 0;
	while (tba) {
		switch (tba->par->type) {
			case TYPE_DOUBLE:
				printf ("%s: %lf -> %lf\n", tba->par->qualifier, tba->par->value.d, adjpars->val[j]);
				phoebe_parameter_set_value (tba->par, adjpars->val[j]);
				j++;
			break;
			case TYPE_DOUBLE_ARRAY:
				for (k = 0; k < lcno; k++) {
					printf ("%s[%d]: %lf -> %lf\n", tba->par->qualifier, k, tba->par->value.vec->val[k], adjpars->val[j]);
					phoebe_parameter_set_value (tba->par, k, adjpars->val[j]);
					j++;
				}
			break;
			default:
				phoebe_lib_error ("exception handler invoked in intern_cost_function (), please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
			
		tba = tba->next;
	}

	/* Fulfill all the constraints: */
	constraint = PHOEBE_pt->lists.constraints;
	while (constraint) {
		phoebe_ast_evaluate (constraint->elem);
		constraint = constraint->next;
	}

	/* Return to the main NMS function. */
	cfval = 0.0;
	return cfval;
/*
	int status, i, curve;
	PHOEBE_vector *chi2weights;

	PHOEBE_el3_units el3units;
	double A;

	double          chi2             = 0.0;
	bool            rv1present       = (*(NMS_passed_parameters *) params).rv1;
	bool            rv2present       = (*(NMS_passed_parameters *) params).rv2;
	bool            color_constraint = (*(NMS_passed_parameters *) params).color_constraint;
	int             tba              = (*(NMS_passed_parameters *) params).to_be_adjusted;
	int             CALCHLA          = (*(NMS_passed_parameters *) params).CALCHLA;
	int             CALCVGA          = (*(NMS_passed_parameters *) params).CALCVGA;
	bool            ASINI            = (*(NMS_passed_parameters *) params).ASINI;
	int             MSC              = (*(NMS_passed_parameters *) params).CC;
	int            *indices          = (*(NMS_passed_parameters *) params).indices;
	double         *weight           = (*(NMS_passed_parameters *) params).weight;
	double         *average          = (*(NMS_passed_parameters *) params).average;
	double         *cindex           = (*(NMS_passed_parameters *) params).cindex;
	double       ***pointers         = (*(NMS_passed_parameters *) params).pointers;
	PHOEBE_vector **chi2s            = (*(NMS_passed_parameters *) params).chi2s;
*/
	/*
	 * During this function we'll store chi2 values of individual datasets to
	 * the vector chi2s and their respective weights to chi2weights. So let us
	 * initialize these two vectors and allocate the data for them:
	 */
/*
	*chi2s = phoebe_vector_new ();
	phoebe_vector_alloc (*chi2s, lcno + rvno);

	chi2weights = phoebe_vector_new ();
	phoebe_vector_alloc (chi2weights, lcno + rvno);
*/
	/*
	 * 1st step: impose value constraints for parameters marked for adjustment.
	 * The trick used here is to make a chi2 barrier over which the solution
	 * won't be able to escape - in particular, a barrier of height 10^10. :)
	 */
/*
	for (i = 0; i < tba; i++) {
		if (  (gsl_vector_get (adjpars, i) < PHOEBE_parameters[indices[i]].min)
		   || (gsl_vector_get (adjpars, i) > PHOEBE_parameters[indices[i]].max) )
			chi2 = 10E10;
	}
*/
	/*
	 * The following loop traverses through all observed light curves, computes
	 * synthetic counterparts at observed times/phases and calculates the
	 * chi2 value for each O-C pair.
	 */

		/* Take care of conditional constraining if it's used:                */
/*
		if (MSC) {
			double T1, T2, P0, L1, L2, M1, M2, q, a, R1, R2, O1, O2;
*/
			/* Scan all TBA parameters to find the temperatures:                    */
/*
			for (i = 0; i < tba; i++) {
				if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_teff1") == 0)
					T1 = gsl_vector_get (adjpars, i);
				if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_teff2") == 0)
					T2 = gsl_vector_get (adjpars, i);
			}

			P0 = pars[curve]->PERIOD;
			calculate_main_sequence_parameters (T1, T2, P0, &L1, &L2, &M1, &M2, &q, &a, &R1, &R2, &O1, &O2);
			pars[curve]->RM   = q;
			pars[curve]->SMA  = a;
			pars[curve]->PHSV = O1;
			pars[curve]->PCSV = O2;
			pars[curve]->HLA  = L1;
		}
		if (ASINI) {
*/
			/* 'indices' is a cno-dimensional array of integers holding indices to  */
			/* parameter qualifiers set for adjustment. We use this to find SMA and */
			/* INCL.                                                                */
/*
			double sma, incl, asini;
			phoebe_parameter_get_value ("phoebe_asini", &asini);

			for (i = 0; i < tba; i++) {
				if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_sma") == 0) {
					sma = gsl_vector_get (adjpars, i);
					if (asini > sma) gsl_vector_set ((gsl_vector *) adjpars, i, sma = asini);
					incl = 180.0/M_PI*asin (asini/sma);
					if (pars[curve]->INCL > 90.0) incl += 90.0;
					pars[curve]->INCL = incl;
					break;
				}
				if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_incl") == 0) {
					incl = gsl_vector_get (adjpars, i);
					sma = asini / sin (incl/180.0*M_PI);
					pars[curve]->SMA = sma;
					break;
				}
			}
		}

		phoebe_parameter_get_value ("phoebe_extinction", curve, &A);

		status = phoebe_el3_units_id (&el3units);
		if (status != SUCCESS) {
			phoebe_lib_error (phoebe_error (status));
#warning FIX INVALID RETURN VALUE
			return -1.0;
		}

		create_lci_file ("lcin.active", *pars[curve]);
		call_wd_to_get_fluxes (syncurve, obs[curve]->indep);
		apply_third_light_correction (syncurve, el3units, pars[curve]->EL3);
		apply_extinction_correction (syncurve, A);
*/
		/* If WD crashes, catch it and exit PHOEBE to see what went wrong:    */
/*
		if (isnan (syncurve->dep->val[0])) {
			phoebe_lib_error ("     *** WD CRASH CAPTURE ***\n");
			phoebe_lib_error ("lcin.active copied to broken.lcin\n");
			phoebe_lib_error ("lcout.active copied to broken.lcout\n");
			phoebe_lib_error ("data dumped to broken.data\n");
			system ("cp lcin.active broken.lcin");
			system ("cp lcout.active broken.lcout");
			FILE *tmpout = fopen ("broken.data", "w");
			for (i = 0; i < syncurve->dep->dim; i++)
				fprintf (tmpout, "%lf %lf\n", obs[curve]->indep->val[i], syncurve->dep->val[i]);
			fclose (tmpout);
			phoebe_lib_error ("       ***** FIX ME!!! *****\n");
			exit (-1);
		}

		if (CALCHLA == 1) {
*/
			/* Compute (rather than minimize) passband luminosities: */
/*
			double av1;

			if (!color_constraint || curve == 0) {
				status = calculate_average (&av1, syncurve->dep);
*/
				/* If the computation went crazy, keep the last sane value:           */
/*
				if (av1 < 1e-1 || isnan (av1)) av1 = average[curve];
				pars[curve]->HLA *= average[curve] / av1;
				for (i = 0; i < syncurve->dep->dim; i++)
					syncurve->dep->val[i] *= average[curve] / av1;
			}
		}
*/
		/* Color-index constraint:                                            */
/*
		if (color_constraint) {
			phoebe_debug ("using the color-index constraint.\n");
			pars[curve]->HLA = cindex[curve] * pars[0]->HLA;
			printf ("index = %lf, L%d = %lf\n", cindex[curve], curve + 1, pars[curve]->HLA);
		}
*/
		/* The weighted chi2 computation:                                     */
/*
		calculate_chi2 (syncurve->dep, obs[curve]->dep, obs[curve]->weight, PHOEBE_CF_CHI2, &((*chi2s)->val[curve]));
		chi2weights->val[curve] = weight[curve];
*/
		/* The cycle is complete, we may release synthetic data:              */
/*
	phoebe_curve_free (syncurve);
	}

	if (rvno != 0) {
		PHOEBE_curve *rv1curve = phoebe_curve_new ();
		PHOEBE_curve *rv2curve = phoebe_curve_new ();
*/
		/*
		 * RVs are sorted in obsdep[] array so that RV1 always preceeds RV2.
		 * Read in the data:
		 */
/*
		if (rv1present) {
			for (i = 0; i < tba; i++)
				*pointers[curve][i] = gsl_vector_get (adjpars, i);

			create_lci_file ("lcin.active", *pars[curve]);
			call_wd_to_get_rv1 (rv1curve, obs[curve]->indep);
		}

		if (rv2present && rv1present) {
			for (i = 0; i < tba; i++)
				*pointers[curve+1][i] = gsl_vector_get (adjpars, i);

			create_lci_file ("lcin.active", *pars[curve+1]);
			call_wd_to_get_rv2 (rv2curve, obs[curve+1]->indep);
		}

		if (rv2present && !rv1present) {
			for (i = 0; i < tba; i++)
				*pointers[curve][i] = gsl_vector_get (adjpars, i);

			create_lci_file ("lcin.active", *pars[curve]);
			call_wd_to_get_rv2 (rv2curve, obs[curve]->indep);
		}
*/
		/* If gamma velocity VGA is set to be calculated, do it:              */
/*
		if (CALCVGA) {
			double vga;
			if ( rv1present && !rv2present)
				vga = calculate_vga (rv1curve->dep, NULL, average[curve], 0.0, pars[curve]->VGA);
			if (!rv1present &&  rv2present)
				vga = calculate_vga (NULL, rv2curve->dep, 0.0, average[curve], pars[curve]->VGA);
			if ( rv1present &&  rv2present)
				vga = calculate_vga (rv1curve->dep, rv2curve->dep, average[curve], average[curve+1], pars[curve]->VGA);

			for (i = 0; i < lcno + rvno; i++) pars[i]->VGA = vga;
		}

		if (ASINI) {
			double sma, incl, asini;
			phoebe_parameter_get_value ("phoebe_asini", &asini);

			for (i = 0; i < tba; i++) {
				if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_sma") == 0) {
					sma = gsl_vector_get (adjpars, i);
					if (asini > sma) gsl_vector_set ((gsl_vector *) adjpars, i, sma = asini);
					incl = 180.0/M_PI*asin (asini/sma);
					if (pars[curve]->INCL > 90.0) incl += 90.0;
					pars[curve]->INCL = incl;
					break;
				}
				if (strcmp (PHOEBE_parameters[indices[i]].qualifier, "phoebe_incl") == 0) {
					incl = gsl_vector_get (adjpars, i);
					sma = asini / sin (incl/180.0*M_PI);
					pars[curve]->SMA = sma;
					break;
				}
			}
		}

		if (rv1present) {
			calculate_chi2 (rv1curve->dep, obs[curve]->dep, obs[curve]->weight, PHOEBE_CF_CHI2, &(*chi2s)->val[curve]);
			chi2weights->val[curve] = weight[curve];
		}
		if (rv2present && rv1present) {
			calculate_chi2 (rv2curve->dep, obs[curve+1]->dep, obs[curve+1]->weight, PHOEBE_CF_CHI2, &(*chi2s)->val[curve+1]);
			chi2weights->val[curve+1] = weight[curve+1];
		}
		if (rv2present && !rv1present) {
			calculate_chi2 (rv2curve->dep, obs[curve]->dep, obs[curve]->weight, PHOEBE_CF_CHI2, &(*chi2s)->val[curve]);
			chi2weights->val[curve] = weight[curve];
		}

		phoebe_curve_free (rv1curve);
		phoebe_curve_free (rv2curve);
	}
*/
	/* We're through with computing synthetic data. Now we calculate chi2:      */
/*
	phoebe_join_chi2 (&chi2, *chi2s, chi2weights);
*/
	/* Free the chi2 containers:                                                */
/*
	phoebe_vector_free (chi2weights);

	return chi2;
*/
}
#endif
#endif

int phoebe_minimize_using_nms (double accuracy, int iter_no, FILE *nms_output, PHOEBE_minimizer_feedback *feedback)
{
	/*
	 * This is a GSL simplex function that we use for multi-D minimization. All
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

#if defined (HAVE_LIBGSL) && !defined (PHOEBE_GSL_DISABLED)

	int status, i, j, c;
	char *readout_str;
	clock_t clock_start, clock_stop;
	PHOEBE_parameter_list *marked_tba, *list;
	PHOEBE_column_type indep;

	PHOEBE_nms_parameters *passed;

	int lcno, rvno;
	int no_tba, dim_tba;
	PHOEBE_curve **obs;       
	WD_LCI_parameters **params;
	double ***pointers;

	/* NMS infrastructure: */
	PHOEBE_vector *steps, *adjpars;
	PHOEBE_nms_simplex *s;
	double size;

	phoebe_debug ("entering downhill simplex minimizer.\n");

	/* Is the feedback structure initialized? */
	if (!feedback) return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	/* Is the feedback structure clear of any data? */
	if (feedback->qualifiers->dim != 0)
		return ERROR_MINIMIZER_FEEDBACK_ALREADY_ALLOCATED;

	/* Fire up the stop watch: */
	clock_start = clock ();

	/* Create a copy of the parameter table and activate it: */
phoebe_parameter_table_print (PHOEBE_pt);
	table = phoebe_parameter_table_duplicate (PHOEBE_pt);
	phoebe_parameter_table_activate (table);
phoebe_parameter_table_print (PHOEBE_pt);

	/* Get the number of LC and RV curves: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);
	if (lcno + rvno == 0) return ERROR_MINIMIZER_NO_CURVES;

	/* Get a list of parameters marked for adjustment: */
	marked_tba = phoebe_parameter_list_get_marked_tba ();
	if (!marked_tba) return ERROR_MINIMIZER_NO_PARAMS;

	/*
	 * Count the number of parameters marked for adjustment; no_tba will hold
	 * the number of parameters, and dim_tba will hold the dimension of the
	 * subspace that will be adjusted; dim_tba should thus be increased by
	 * one for each passband-independent parameter and by the number of light
	 * curves for each passband-dependent parameter.
	 */

	list = marked_tba; no_tba = 0; dim_tba = 0;
	while (list) {
		no_tba++;
		if (list->par->type == TYPE_DOUBLE) dim_tba++;
		else dim_tba += lcno;
		list = list->next;
	}

	phoebe_debug ("Parameters set for adjustment:\n");
	list = marked_tba; i = 0;
	while (list) {
		phoebe_debug ("%2d. qualifier:     %s\n", i+1, list->par->qualifier);
		i++; list = list->next;
	}

phoebe_constraint_new ("phoebe_sma = 10/sin(phoebe_incl/180*3.1415926)");
phoebe_constraint_new ("phoebe_hla[1] = 0.5*phoebe_hla[0]");

	/* Will the computation be done in HJD- or in phase-space? */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
	phoebe_column_get_type (&indep, readout_str);

	/* Read in the observed data and transform them appropriately: */
	obs = phoebe_malloc ( (lcno+rvno) * sizeof (*obs));
	for (i = 0; i < lcno; i++) {
		obs[i] = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, i);
		if (!obs[i]) {
			for (j = 0; j < i; j++)
				phoebe_curve_free (obs[j]);
			free (obs);
			return ERROR_FILE_NOT_FOUND;
		}
		phoebe_curve_transform (obs[i], indep, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_WEIGHT);
	}
	for (i = 0; i < rvno; i++) {
		obs[lcno+i] = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, i);
		if (!obs[lcno+i]) {
			for (j = 0; j < lcno+i; j++)
				phoebe_curve_free (obs[j]);
			free (obs);
			return ERROR_FILE_NOT_FOUND;
		}
		phoebe_curve_transform (obs[lcno+i], indep, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_WEIGHT);
	}

	/* Read in model parameters for each curve/passband: */
	params = phoebe_malloc ( (lcno+rvno) * sizeof (*params));
	for (i = 0; i < lcno; i++) {
		params[i] = phoebe_malloc (sizeof (**params));
		read_in_wd_lci_parameters (params[i], /* MPAGE = */ 1, i);
	}
	for (i = 0; i < rvno; i++) {
		params[lcno+i] = phoebe_malloc (sizeof (**params));
		read_in_wd_lci_parameters (params[lcno+i], /* MPAGE = */ 2, i);
	}

	/* Populate the structure that will be passed to the cost function: */
	passed = phoebe_malloc (sizeof (*passed));

	passed->tba      = table->lists.marked_tba;
	passed->no_tba   = no_tba;
	passed->dim_tba  = dim_tba;
	passed->lcno     = lcno;
	passed->rvno     = rvno;
	passed->obs      = obs;
	passed->pars     = params;

	/* Initialize the cost function: */
/*
	cf.f      = &intern_chi2_cost_function;
	cf.n      = dim_tba;
	cf.params = (void *) passed;
*/
	/* Read out initial values and step sizes: */
	adjpars = phoebe_vector_new (); phoebe_vector_alloc (adjpars, dim_tba);
	steps   = phoebe_vector_new ();	phoebe_vector_alloc (steps, dim_tba);

	list = marked_tba; i = 0;
	while (list) {
		switch (list->par->type) {
			case TYPE_DOUBLE:
				adjpars->val[i] = list->par->value.d;
				steps->val[i]   = list->par->step;
				i++;
			break;
			case TYPE_DOUBLE_ARRAY:
				for (j = 0; j < lcno; j++) {
					printf ("setting parameter %s for i=%d\n", list->par->qualifier, i+j);
					adjpars->val[i+j] = list->par->value.vec->val[j];
					steps->val[i+j]   = list->par->step;
				}
				i += lcno;
			break;
			default:
				phoebe_lib_error ("exception handler invoked in phoebe_minimize_using_nms (), please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		list = list->next;
	}

	/* Allocate the minimizer: */
	s = phoebe_nms_simplex_new ();
	phoebe_nms_simplex_alloc (s, dim_tba);

	phoebe_nms_set (s, &intern_chi2_cost_function, adjpars, passed, &size, steps);

	/* Stop the clock watch: */
	clock_stop = clock ();

	/* Populate the feedback structure: */
	feedback->algorithm = PHOEBE_MINIMIZER_NMS;
	feedback->cputime   = (double) (clock_stop - clock_start) / CLOCKS_PER_SEC;

	/* Free observed data and model parameters: */
	for (i = 0; i < lcno + rvno; i++) {
		phoebe_curve_free (obs[i]);
		free (params[i]);
	}
	free (obs);
	free (params);

	/* Free the passed parameters structure: */
	free (passed);

	/* Free NMS-related parameters: */
	phoebe_nms_simplex_free (s);
	phoebe_vector_free (adjpars);
	phoebe_vector_free (steps);

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
	PHOEBE_vector *chi2s;     
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
		if (!filename_exists (readout_str))
			return ERROR_MINIMIZER_INVALID_FILE;
	}
	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value ("phoebe_rv_filename", i, &readout_str);
		if (!filename_exists (readout_str))
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

		phoebe_parameter_get_value ("phoebe_cindex", curve, &(cindex[curve]));

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

	passed_pars.chi2s            = &chi2s;
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
#endif

	phoebe_lib_error ("GSL library not present, cannot initiate NMS minimizer.\n");
	return ERROR_GSL_NOT_INSTALLED;
}

int kick_parameters (double sigma)
{
	/*
	 * This function kicks current parameters stochastically, following the
	 * Gaussian PDF. The passed argument is the relative standard deviation,
	 * \sigma_{rel} = \sigma_{abs} / parvalue. It requires GSL to sample the
	 * kick strength by the Gaussian PDF.
	 *
	 * Synopsis:
	 *
	 *   kick_parameters (sigma)
	 * 
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_GSL_NOT_INSTALLED
	 *   SUCCESS
	 */

#warning FUNCTION TEMPORARILY DISABLED FOR REVIEW

/*
#ifdef HAVE_LIBGSL
#ifndef PHOEBE_GSL_DISABLED

	int status;
	int to_be_adjusted;
*/
/*	double *initvals; */        /* An array of initial parameter values.        */
/*	int    *indices;  */        /* An array of global table reference indices   */
/*
	int lcno, rvno;
	int i;

	phoebe_debug ("entering kick_parameters ()\n");
*/
	/* Count the available curves:                                            */
/*
	phoebe_parameter_get_value ("phoebe_lcno", &lcno);
	phoebe_parameter_get_value ("phoebe_rvno", &rvno);
*/
	/* The following block reads out parameters marked for adjustment:        */
/*
	read_in_adjustable_parameters (&to_be_adjusted, &initvals, &indices);

	for (i = 0; i < to_be_adjusted; i++)
		if (PHOEBE_parameters[indices[i]].type == TYPE_DOUBLE)
			{
			phoebe_parameter_set_value (PHOEBE_parameters[indices[i]].qualifier, initvals[i] + gsl_ran_gaussian (PHOEBE_randomizer, sigma * initvals[i]));
			phoebe_debug ("  kicked %s from %lf to %lf\n", PHOEBE_parameters[indices[i]].qualifier, initvals[i], PHOEBE_parameters[indices[i]].value.d);
			}
		else
			{
#warning IMPLEMENT CURVE-DEPENDENT KICKS!
			}

	free (initvals);
	free (indices);

	phoebe_debug ("leaving kick_parameters ()\n");
	return SUCCESS;
#endif
#endif

	phoebe_lib_error ("GSL library not present, cannot kick parameters.\n");
	return ERROR_GSL_NOT_INSTALLED;
*/
}

int phoebe_minimize_using_dc (FILE *dc_output, PHOEBE_minimizer_feedback *feedback)
{
	/*
	 * This is WD's built-in DC algorithm and as such it doesn't depend on GSL.
	 * Macro wd_dc () is provided to access the fortran subroutine.
	 *
	 * Return values:
	 *
	 *   ERROR_INVALID_INDEP
	 *   ERROR_INVALID_DEP
	 *   ERROR_INVALID_WEIGHT
	 *   ERROR_INVALID_DATA
	 *   ERROR_MINIMIZER_NO_CURVES
	 *   ERROR_MINIMIZER_NO_PARAMS
	 *   ERROR_MINIMIZER_HLA_REQUEST_NOT_SANE
	 *   ERROR_MINIMIZER_VGA_REQUEST_NOT_SANE
	 *   ERROR_MINIMIZER_DPDT_REQUEST_NOT_SANE
	 *   SUCCESS
	 */

	int status, i, j;
	clock_t clock_start, clock_stop;
	char atmcof[255], atmcofplanck[255];
	WD_DCI_parameters *params;
	PHOEBE_parameter_list *marked_tba;
	int no_tba;
	int lcno = 0, rvno = 0;
	bool calchla = FALSE, calcvga = FALSE;

	/* Minimizer results: */
	double *corrections;
	double *errors;
	double *chi2s;
	double  cfval;

	phoebe_debug ("entering differential corrections minimizer.\n");

	/* Check if the feedback structure is initialized and not allocated: */
	if (!feedback)
		return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;
	if (feedback->qualifiers->dim != 0)
		return ERROR_MINIMIZER_FEEDBACK_ALREADY_ALLOCATED;

	/* Everything seems to be ok. Fire up the stop watch: */
	clock_start = clock ();

	/* Get a list of parameters marked for adjustment: */
	marked_tba = phoebe_parameter_list_get_marked_tba ();
	if (!marked_tba)
		return ERROR_MINIMIZER_NO_PARAMS;

	/* Read in WD DCI parameters: */
	params = wd_dci_parameters_new ();
	status = read_in_wd_dci_parameters (params, &no_tba);
	if (status != SUCCESS) return status;

	/* Count the number of light and RV curves: */
	lcno = params->nlc;
	rvno = params->rv1data + params->rv2data;

	/* Allocate memory for the results: */
	corrections = phoebe_malloc (no_tba * sizeof (*corrections));
	errors      = phoebe_malloc (no_tba * sizeof (*errors));
	chi2s       = phoebe_malloc ((lcno + rvno) * sizeof (*chi2s));

	/* Create the DCI file from the params variable: */
	create_dci_file ("dcin.active", params);

	/* Assign the filenames for atmcof and atmcofplanck needed by WD: */
	sprintf (atmcof,       "%s/wd/atmcof.dat",       PHOEBE_BASE_DIR);
	sprintf (atmcofplanck, "%s/wd/atmcofplanck.dat", PHOEBE_BASE_DIR);

	/* Run one DC iteration and store the results in the allocated arrays: */
	wd_dc (atmcof, atmcofplanck, corrections, errors, chi2s, &cfval);

	/*
	 * Allocate the feedback structure and fill it in. The number of parameter
	 * fields equals the number of parameters marked for adjustment plus the
	 * number of light curves if HLAs are marked for computation plus 1 if VGA
	 * VGA is marked for computation.
	 */

	phoebe_minimizer_feedback_alloc (feedback, no_tba+(calchla*lcno)+calcvga, lcno+rvno);

	feedback->algorithm = PHOEBE_MINIMIZER_DC;
	feedback->iters = 1;

	for (i = 0; i < lcno + rvno; i++) {
		feedback->chi2s->val[i] = chi2s[i];
	}

	/* Weighted chi2s are not handled yet, let's free the memory: */
	phoebe_vector_free (feedback->wchi2s);
	feedback->wchi2s = NULL;

	feedback->cfval = cfval;
/*
	fprintf (dc_output, "%-18s %-12s %-12s %-12s %-12s\n", "Qualifier:", "Original:", "Correction:", "   New:", "  Error:");
	fprintf (dc_output, "--------------------------------------------------------------------\n");
*/
	i = 0;
	while (marked_tba) {
		if (marked_tba->par->type == TYPE_INT    ||
			marked_tba->par->type == TYPE_BOOL   ||
			marked_tba->par->type == TYPE_DOUBLE ||
			marked_tba->par->type == TYPE_STRING) {
			phoebe_parameter_get_value (marked_tba->par, &(feedback->initvals->val[i]));
			feedback->qualifiers->val.strarray[i] = strdup (marked_tba->par->qualifier);
			feedback->newvals->val[i] = feedback->initvals->val[i] + corrections[i];
			feedback->ferrors->val[i] = errors[i];
			i++;
		}
		else {
			for (j = 0; j < marked_tba->par->value.array->dim; j++) {
				phoebe_parameter_get_value (marked_tba->par, j, &(feedback->initvals->val[i+j]));
				feedback->qualifiers->val.strarray[i+j] = strdup (marked_tba->par->qualifier);
				feedback->newvals->val[i+j] = feedback->initvals->val[i+j] + corrections[i+j];
				feedback->ferrors->val[i+j] = errors[i+j];
			}
			i += j;
		}

		marked_tba = marked_tba->next;
	}

	/* Free all the allocated structures: */
	wd_dci_parameters_free (params);
	free (corrections);
	free (errors);
	free (chi2s);

	/* Stop the clock watch and compute the total CPU time on the process: */
	clock_stop = clock ();
	feedback->cputime = (double) (clock_stop - clock_start) / CLOCKS_PER_SEC;

	phoebe_debug ("leaving differential corrections minimizer.\n");

	return SUCCESS;
}
