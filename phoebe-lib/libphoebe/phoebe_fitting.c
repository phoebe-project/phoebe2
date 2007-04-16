#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "phoebe_build_config.h"
#include "cfortran.h"

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_calculations.h"
#include "phoebe_error_handling.h"
#include "phoebe_fitting.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

PROTOCCALLSFFUN6(VOID,WDDC,wddc,STRING,STRING,DOUBLEV,DOUBLEV,DOUBLEV,DOUBLEV)
#define WD_FIND_MINIMUM_WITH_DC(FATMCOF,FATMCOFPL,CORRECTION,STDDEV,CHISQS,CHISQ) CCALLSFFUN6(WDDC,wddc,STRING,STRING,DOUBLEV,DOUBLEV,DOUBLEV,DOUBLEV,FATMCOF,FATMCOFPL,CORRECTION,STDDEV,CHISQS,CHISQ)

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

double intern_chi2_cost_function (const gsl_vector *adjpars, void *params)
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

	int i, curve, status;
	PHOEBE_vector *chi2weights;

	PHOEBE_el3_units el3units;
	double A;

	double          chi2             = 0.0;
	int             lcno             = (*(NMS_passed_parameters *) params).lcno;
	int             rvno             = (*(NMS_passed_parameters *) params).rvno;
	bool            rv1present       = (*(NMS_passed_parameters *) params).rv1;
	bool            rv2present       = (*(NMS_passed_parameters *) params).rv2;
	bool            color_constraint = (*(NMS_passed_parameters *) params).color_constraint;
	int             tba              = (*(NMS_passed_parameters *) params).to_be_adjusted;
	int             CALCHLA          = (*(NMS_passed_parameters *) params).CALCHLA;
	int             CALCVGA          = (*(NMS_passed_parameters *) params).CALCVGA;
	bool            ASINI            = (*(NMS_passed_parameters *) params).ASINI;
	int             MSC              = (*(NMS_passed_parameters *) params).CC;
	int            *indices          = (*(NMS_passed_parameters *) params).indices;
	PHOEBE_curve  **obs              = (*(NMS_passed_parameters *) params).obs;
	double         *weight           = (*(NMS_passed_parameters *) params).weight;
	double         *average          = (*(NMS_passed_parameters *) params).average;
	double         *cindex           = (*(NMS_passed_parameters *) params).cindex;
	double       ***pointers         = (*(NMS_passed_parameters *) params).pointers;
	WD_LCI_parameters **pars         = (*(NMS_passed_parameters *) params).pars;
	PHOEBE_vector **chi2s            = (*(NMS_passed_parameters *) params).chi2s;

	/*
	 * During this function we'll store chi2 values of individual datasets to
	 * the vector chi2s and their respective weights to chi2weights. So let us
	 * initialize these two vectors and allocate the data for them:
	 */

	*chi2s = phoebe_vector_new ();
	phoebe_vector_alloc (*chi2s, lcno + rvno);

	chi2weights = phoebe_vector_new ();
	phoebe_vector_alloc (chi2weights, lcno + rvno);

	/*
	 * 1st step: impose value constraints for parameters marked for adjustment.
	 * The trick used here is to make a chi2 barrier over which the solution
	 * won't be able to escape - in particular, a barrier of height 10^10. :)
	 */

	for (i = 0; i < tba; i++) {
		if (  (gsl_vector_get (adjpars, i) < PHOEBE_parameters[indices[i]].min)
		   || (gsl_vector_get (adjpars, i) > PHOEBE_parameters[indices[i]].max) )
			chi2 = 10E10;
	}

	/*
	 * The following loop traverses through all observed light curves, computes
	 * synthetic counterparts at observed times/phases and calculates the
	 * chi2 value for each O-C pair.
	 */

	for (curve = 0; curve < lcno; curve++) {
		/* Initialize a synthetic curve: */
		PHOEBE_curve *syncurve = phoebe_curve_new ();

		/* Take care of conditional constraining if it's used:                */
		if (MSC) {
			double T1, T2, P0, L1, L2, M1, M2, q, a, R1, R2, O1, O2;

			/* Scan all TBA parameters to find the temperatures:                    */
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
			/* 'indices' is a cno-dimensional array of integers holding indices to  */
			/* parameter qualifiers set for adjustment. We use this to find SMA and */
			/* INCL.                                                                */

			double sma, incl, asini;
			phoebe_get_parameter_value ("phoebe_asini", &asini);

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

		/*
		 * Set the values in a parameter table to match current iteration
		 * values:
		 */

		for (i = 0; i < tba; i++)
			*pointers[curve][i] = gsl_vector_get (adjpars, i);

		phoebe_get_parameter_value ("phoebe_extinction", curve, &A);

		status = phoebe_el3_units_id (&el3units);
		if (status != SUCCESS) {
			phoebe_lib_error (phoebe_error (status));
			return -1.0;
		}

		create_lci_file ("lcin.active", *pars[curve]);
		call_wd_to_get_fluxes (syncurve, obs[curve]->indep);
		apply_third_light_correction (syncurve, el3units, pars[curve]->EL3);
		apply_extinction_correction (syncurve, A);

		/* If WD crashes, catch it and exit PHOEBE to see what went wrong:    */
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

		/* Finally, let's compute (rather than minimize) the levels HLA:   	  */
		if (CALCHLA == 1) {
			double av1;

			if (!color_constraint || curve == 0) {
				status = calculate_average (&av1, syncurve->dep);

				/* If the computation went crazy, keep the last sane value:           */
				if (av1 < 1e-1 || isnan (av1)) av1 = average[curve];
				pars[curve]->HLA *= average[curve] / av1;
				for (i = 0; i < syncurve->dep->dim; i++)
					syncurve->dep->val[i] *= average[curve] / av1;
			}
		}

		/* Color-index constraint:                                            */
		if (color_constraint) {
			phoebe_debug ("using the color-index constraint.\n");
			pars[curve]->HLA = cindex[curve] * pars[0]->HLA;
			printf ("index = %lf, L%d = %lf\n", cindex[curve], curve + 1, pars[curve]->HLA);
		}

		/* The weighted chi2 computation:                                     */
		calculate_chi2 (syncurve->dep, obs[curve]->dep, obs[curve]->weight, PHOEBE_CF_CHI2, &((*chi2s)->val[curve]));
		chi2weights->val[curve] = weight[curve];

		/* The cycle is complete, we may release synthetic data:              */
		phoebe_curve_free (syncurve);
	}

	if (rvno != 0) {
		PHOEBE_curve *rv1curve = phoebe_curve_new ();
		PHOEBE_curve *rv2curve = phoebe_curve_new ();

		/*
		 * RVs are sorted in obsdep[] array so that RV1 always preceeds RV2.
		 * Read in the data:
		 */

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

		/* If gamma velocity VGA is set to be calculated, do it:              */
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
			phoebe_get_parameter_value ("phoebe_asini", &asini);

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

	/* We're through with computing synthetic data. Now we calculate chi2:      */
	phoebe_join_chi2 (&chi2, *chi2s, chi2weights);
	
	/* Free the chi2 containers:                                                */
	phoebe_vector_free (chi2weights);

	return chi2;
}
#endif
#endif

int find_minimum_with_nms (double accuracy, int iter_no, FILE *nms_output, PHOEBE_minimizer_feedback **feedback)
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

#ifdef HAVE_LIBGSL
#ifndef PHOEBE_GSL_DISABLED
	double *initvals;         /* An array of initial parameter values.        */
	int    *indices;          /* An array of global table reference indices   */
	int     cno, lcno, rvno;  /* Curves (both LC and RV) number               */
	int     CALCHLA;          /* Should light levels be calculated or fitted  */
	int     CALCVGA;          /* Should gamma velocity be calculated          */
	bool    ASINI;            /* A switch whether a \sin i should be constant */
	bool    color_constraint; /* Color-constraint switch                      */
	int     CC;               /* Conditional constraining (CC) switch         */
	int     to_be_adjusted;   /* The number of adjustables to be adjusted     */
	double ***pointers;       /* Array of pointers to the WD_LCI_parameters   */
	PHOEBE_curve **obs;       /* An array of observational data               */
	double *weight;           /* Weights of passband curves                   */
	double *average;          /* Average values of observations for HLA/VGA   */
	double *cindex;           /* An array of color indices                    */

	PHOEBE_vector *chi2s;     /* A list of individual passband chi2 values    */

	int curve, index, parindex;
	double parvalue;

	clock_t clock_start, clock_stop;

	const gsl_multimin_fminimizer_type *T;
	gsl_multimin_fminimizer *s = NULL;
	gsl_vector *step_size, *adjpars;
	gsl_multimin_function cost_function;

	size_t iter = 0;
	size_t i, j, k;
	int status;
	double size;
	double dpdt;
	const char *indep;
	const char *readout_str;

	NMS_passed_parameters passed_pars;
	WD_LCI_parameters *params;

	PHOEBE_input_dep dep;

	phoebe_debug ("entering downhill simplex minimizer.\n");

	phoebe_get_parameter_value ("phoebe_indep", &indep);
	phoebe_get_parameter_value ("phoebe_dpdt",  &dpdt);

	/* Before we do anything, let's check whether the setup is sane:          */
	if (!hla_request_is_sane  ()) return ERROR_MINIMIZER_HLA_REQUEST_NOT_SANE;
	if (!vga_request_is_sane  ()) return ERROR_MINIMIZER_VGA_REQUEST_NOT_SANE;
	if (!dpdt_request_is_sane ()) return ERROR_MINIMIZER_DPDT_REQUEST_NOT_SANE;

	/* Is the feedback structure initialized:                                 */
	if (!feedback) return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	/* Fire up the stop watch:                                                */
	clock_start = clock ();

	/* Initializing the NMS minimizer:                                        */
	phoebe_debug ("initializing the minimizer.\n");
	T = gsl_multimin_fminimizer_nmsimplex;

	/* Count the available curves:                                            */
	phoebe_get_parameter_value ("phoebe_lcno", &lcno);
	phoebe_get_parameter_value ("phoebe_rvno", &rvno);
	cno = lcno + rvno;
	phoebe_debug ("total number of curves (both LC and RV): %d\n", cno);

	if (cno == 0) return ERROR_MINIMIZER_NO_CURVES;

	/*
	 * Check the filenames; we do this early because nothing is yet
	 * initialized and we don't have to worry about any memory leaks.
	 */

	phoebe_debug ("checking whether everything's ok with filenames.\n");
	for (i = 0; i < lcno; i++) {
		phoebe_get_parameter_value ("phoebe_lc_filename", i, &readout_str);
		if (!filename_exists (readout_str))
			return ERROR_MINIMIZER_INVALID_FILE;
	}
	for (i = 0; i < rvno; i++) {
		phoebe_get_parameter_value ("phoebe_rv_filename", i, &readout_str);
		if (!filename_exists (readout_str))
			return ERROR_MINIMIZER_INVALID_FILE;
	}

	/* Verify the types of RV curves:                                         */
	passed_pars.rv1 = FALSE; passed_pars.rv2 = FALSE;
	for (i = 0; i < rvno; i++) {
		phoebe_get_parameter_value ("phoebe_rv_dep", i, &readout_str);
		status = get_input_dependent_variable (readout_str, &dep);
		if (status != SUCCESS) return status;

		if (dep == INPUT_PRIMARY_RV)   passed_pars.rv1 = TRUE;
		if (dep == INPUT_SECONDARY_RV) passed_pars.rv2 = TRUE;
	}

	phoebe_debug ("RV1 curve is present: %d\n", passed_pars.rv1);
	phoebe_debug ("RV2 curve is present: %d\n", passed_pars.rv2);

	/* The following block reads out parameters marked for adjustment:        */
	status = read_in_adjustable_parameters (&to_be_adjusted, &initvals, &indices);
	if (status != SUCCESS) return status;
	phoebe_debug ("total number of parameters to be adjusted: %d\n", to_be_adjusted);

	if (to_be_adjusted == 0) return ERROR_MINIMIZER_NO_PARAMS;

	phoebe_debug ("Parameters set for adjustment:\n");

	for (i = 0; i < to_be_adjusted; i++) {
		phoebe_debug ("%2d. qualifier:     %s\n", i+1, PHOEBE_parameters[indices[i]].qualifier);
		phoebe_debug ("    index:         %d\n", indices[i]);
		phoebe_debug ("    initial value: %lf\n", initvals[i]);
	}

	adjpars   = gsl_vector_alloc (to_be_adjusted);
	step_size = gsl_vector_alloc (to_be_adjusted);

	pointers  = phoebe_malloc (cno * sizeof (*pointers));
	for (i = 0; i < cno; i++)
		pointers[i]  = phoebe_malloc (to_be_adjusted * sizeof (**pointers));

	passed_pars.pars = phoebe_malloc ( cno * sizeof (*(passed_pars.pars)));
	params           = phoebe_malloc ( cno * sizeof (*params));
	obs              = phoebe_malloc ( cno * sizeof (*obs));
	weight           = phoebe_malloc ( cno * sizeof (*weight));
	average          = phoebe_malloc ( cno * sizeof (*average));
	cindex           = phoebe_malloc (lcno * sizeof (*cindex));

	/* First we must read in all data, so that the following segment may se-  */
	/* quentially assign pointers to it:                                      */

	for (curve = 0; curve < lcno; curve++)
		read_in_wd_lci_parameters (&params[curve], /*MPAGE=*/1, curve);
	for (curve = lcno; curve < cno; curve++)
		read_in_wd_lci_parameters (&params[curve], /*MPAGE=*/2, curve-lcno);

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
		/* Now comes the pointing part:                                           */
		for (i = 0; i < to_be_adjusted; i++) {
			gsl_vector_set (adjpars, i, initvals[i]);
			gsl_vector_set (step_size, i, PHOEBE_parameters[indices[i]].step);

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

		/* Initialize observational data:                                         */
		obs[curve]  = phoebe_curve_new ();

		/* Read the LC data in and do all exception checking rigorously:          */
		if (curve < lcno) {
			const char *file;
			PHOEBE_input_indep  indep;
			PHOEBE_input_dep    dep;
			PHOEBE_input_weight weight;

			phoebe_get_parameter_value ("phoebe_lc_filename", curve, &file);

			phoebe_get_parameter_value ("phoebe_lc_indep", curve, &readout_str);
			status = get_input_independent_variable (readout_str, &indep);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value ("phoebe_lc_dep", curve, &readout_str);
			status = get_input_dependent_variable (readout_str, &dep);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value ("phoebe_lc_indweight", curve, &readout_str);
			status = get_input_weight (readout_str, &weight);
			if (status != SUCCESS) return status;

			status = read_in_observational_data
				(
				file,
				obs[curve],
				indep,
#warning FIX PHASE-ONLY COMPUTATION
				OUTPUT_PHASE,
				dep,
				OUTPUT_TOTAL_FLUX,
				weight,
				OUTPUT_STANDARD_WEIGHT,
				NO,
				-0.5,
				+0.5
				);
			if (status != SUCCESS) return status;
		}

		/* Read in RV data in and do all exception checking rigorously:       */
		if (curve == lcno) {
			const char *file;
			PHOEBE_input_indep  indep;
			PHOEBE_input_dep    dep;
			PHOEBE_input_weight weight;

			phoebe_get_parameter_value ("phoebe_rv_filename", curve-lcno, &file);

			phoebe_get_parameter_value ("phoebe_rv_indep", curve-lcno, &readout_str);
			status = get_input_independent_variable (readout_str, &indep);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value ("phoebe_rv_dep", curve-lcno, &readout_str);
			status = get_input_dependent_variable (readout_str, &dep);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value ("phoebe_rv_indweight", curve-lcno, &readout_str);
			status = get_input_weight (readout_str, &weight);
			if (status != SUCCESS) return status;

			if (rvno == 2) {
				if (dep == INPUT_PRIMARY_RV) {
					status = read_in_observational_data (
						file,
						obs[curve],
						indep,
#warning FIX PHASE-ONLY COMPUTATION
						OUTPUT_PHASE,
						dep,
						OUTPUT_PRIMARY_RV,
						weight,
						OUTPUT_STANDARD_WEIGHT,
						NO,
						-0.5,
						+0.5
					);
					if (status != SUCCESS) return status;
				}
				if (dep == INPUT_SECONDARY_RV) {
					status = read_in_observational_data (
						file,
						obs[curve+1],
						indep,
#warning FIX PHASE-ONLY COMPUTATION
						OUTPUT_PHASE,
						dep,
						OUTPUT_SECONDARY_RV,
						weight,
						OUTPUT_STANDARD_WEIGHT,
						NO,
						-0.5,
						+0.5
					);
					if (status != SUCCESS) return status;
				}
			} else {
				if (dep == INPUT_PRIMARY_RV) {
					status = read_in_observational_data (
						file,
						obs[curve],
						indep,
#warning FIX PHASE-ONLY COMPUTATION
						OUTPUT_PHASE,
						dep,
						OUTPUT_PRIMARY_RV,
						weight,
						OUTPUT_STANDARD_WEIGHT,
						NO,
						-0.5,
						+0.5
					);
					if (status != SUCCESS) return status;
				}
				if (dep == INPUT_SECONDARY_RV) {
					status = read_in_observational_data (
						file,
						obs[curve],
						indep,
#warning FIX PHASE-ONLY COMPUTATION
						OUTPUT_PHASE,
						dep,
						OUTPUT_SECONDARY_RV,
						weight,
						OUTPUT_STANDARD_WEIGHT,
						NO,
						-0.5,
						+0.5
					);
					if (status != SUCCESS) return status;
				}
			}

			/* Transform read experimental data to 100km/s units:                   */
			for (i = 0; i < obs[curve]->dep->dim; i++)
				obs[curve]->dep->val[i] /= 100.0;
		}

		if (curve == lcno+1) {
			const char *file;
			PHOEBE_input_indep  indep;
			PHOEBE_input_dep    dep;
			PHOEBE_input_weight weight;

			phoebe_get_parameter_value ("phoebe_rv_filename", curve-lcno, &file);

			phoebe_get_parameter_value ("phoebe_rv_indep", curve-lcno, &readout_str);
			status = get_input_independent_variable (readout_str, &indep);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value ("phoebe_rv_dep", curve-lcno, &readout_str);
			status = get_input_dependent_variable (readout_str, &dep);
			if (status != SUCCESS) return status;

			phoebe_get_parameter_value ("phoebe_rv_indweight", curve-lcno, &readout_str);
			status = get_input_weight (readout_str, &weight);
			if (status != SUCCESS) return status;

			if (dep == INPUT_PRIMARY_RV) {
				status = read_in_observational_data (
					file,
					obs[curve-1],
					indep,
#warning FIX PHASE-ONLY COMPUTATION
					OUTPUT_PHASE,
					dep,
					OUTPUT_PRIMARY_RV,
					weight,
					OUTPUT_STANDARD_WEIGHT,
					NO,
					-0.5,
					+0.5
				);
				if (status != SUCCESS) return status;
			}
			if (dep == INPUT_SECONDARY_RV) {
				status = read_in_observational_data (
					file,
					obs[curve],
					indep,
#warning FIX PHASE-ONLY COMPUTATION
					OUTPUT_PHASE,
					dep,
					OUTPUT_SECONDARY_RV,
					weight,
					OUTPUT_STANDARD_WEIGHT,
					NO,
					-0.5,
					+0.5
				);
			if (status != SUCCESS) return status;
			}

			/* Transform read experimental data to 100km/s units:                   */
			for (i = 0; i < obs[curve]->dep->dim; i++)
				obs[curve]->dep->val[i] /= 100.0;
		}

		/* Get standard deviations, average values and color indices for the  */
		/* observed data:                                                     */
		if (curve < lcno) {
			PHOEBE_input_dep dep;
			double sigma;

			phoebe_get_parameter_value ("phoebe_lc_dep", curve, &readout_str);
			get_input_dependent_variable (readout_str, &dep);

			phoebe_get_parameter_value ("phoebe_lc_sigma", curve, &sigma);
			if (dep == INPUT_MAGNITUDE)
				sigma = pow (10, 2./5.*sigma) - 1.0;

			weight[curve] = 1./sigma/sigma;

			phoebe_get_parameter_value ("phoebe_cindex", curve, &(cindex[curve]));
		}
		if (curve == lcno) {
			double sigma;
			phoebe_get_parameter_value ("phoebe_rv_sigma", 0, &sigma);
			weight[curve] = 1./(sigma/100.0)/(sigma/100.0);
		}
		if (curve == lcno+1) {
			double sigma;
			phoebe_get_parameter_value ("phoebe_rv_sigma", 1, &sigma);
			weight[curve] = 1./(sigma/100.0)/(sigma/100.0);
		}

		status = calculate_weighted_average (&(average[curve]), obs[curve]->dep, obs[curve]->weight);
	}


	/* Fill in the structure the pointer to which will be passed to chi2      */
	/* function:                                                              */
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

	/* Initialize method and iterate */
	cost_function.f      = &intern_chi2_cost_function;
	cost_function.n      = to_be_adjusted;
	cost_function.params = (void *) &passed_pars;

	s = gsl_multimin_fminimizer_alloc (T, to_be_adjusted);
	gsl_multimin_fminimizer_set (s, &cost_function, adjpars, step_size);

	/* The main iterating loop:                                               */
	do {
		iter++;

		if (PHOEBE_INTERRUPT) break;

		status = gsl_multimin_fminimizer_iterate (s);
		if (status) break;

		/* This is an optional action to be performed after each iteration. It is */
		/* a statement passed by the user through a command line.                 */
		/*		if (action) scripter_ast_evaluate (action);*/

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
		if (lcno != cno)               /* Print VGA only if there are RVs present */
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

	/* Stop the clock watch and compute the total CPU time on the process:    */
	clock_stop = clock ();

	/* Allocate the feedback structure and fill in its contents:              */
	phoebe_debug ("allocated fields for feedback: %d + %d*%d + %d = %d\n", to_be_adjusted, CALCHLA, lcno, CALCVGA, to_be_adjusted+CALCHLA*lcno+CALCVGA);
	status = phoebe_minimizer_feedback_alloc (feedback, to_be_adjusted+CALCHLA*lcno+CALCVGA, cno);
	if (status != SUCCESS) return status;

	(*feedback)->algorithm = PHOEBE_MINIMIZER_NMS;
	(*feedback)->iters     = iter;
	(*feedback)->cfval     = s->fval;
	(*feedback)->cputime   = (double) (clock_stop - clock_start) / CLOCKS_PER_SEC;

	index = 0;

	/* If HLA's were calculated (rather than fitted), update their values:    */
	if (CALCHLA == 1) {
		phoebe_index_from_qualifier (&parindex, "phoebe_hla");
		for (i = 0; i < lcno; i++) {
			phoebe_get_parameter_value ("phoebe_hla", i, &parvalue);
			(*feedback)->indices->val.iarray[index] = parindex;
			(*feedback)->initvals->val[index] = parvalue;
			(*feedback)->newvals->val[index] = params[i].HLA;
			(*feedback)->ferrors->val[index] = sqrt (-1);
			index++;
		}
	}

	/* We do the same with v_\gamma:                                          */
	if (CALCVGA == 1) {
		phoebe_index_from_qualifier (&parindex, "phoebe_vga");
		phoebe_get_parameter_value ("phoebe_vga", &parvalue);

		(*feedback)->indices->val.iarray[index] = parindex;
		(*feedback)->initvals->val[index] = parvalue;
		(*feedback)->newvals->val[index] = params[i].VGA;
		(*feedback)->ferrors->val[index] = sqrt (-1);
		index++;
	}

	for (i = 0; i < to_be_adjusted; i++) {
		(*feedback)->indices->val.iarray[index] = indices[i];
		(*feedback)->initvals->val[index]       = initvals[i];
		(*feedback)->newvals->val[index]        = gsl_vector_get (s->x, i);
		(*feedback)->ferrors->val[index]        = accuracy*gsl_vector_get (s->x, i);
		index++;
	}

	/* Supply unweighted and passband-weighted chi2 values to the feedback:   */
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

	/* Let's clean up:                                                        */
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

	phoebe_debug ("leaving downhill simplex minimizer.\n");

	return status;
#endif
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

#ifdef HAVE_LIBGSL
#ifndef PHOEBE_GSL_DISABLED

	int status;
	int to_be_adjusted;
	double *initvals;         /* An array of initial parameter values.        */
	int    *indices;          /* An array of global table reference indices   */
	int lcno, rvno;
	int i;

	phoebe_debug ("entering kick_parameters ()\n");

	/* Count the available curves:                                            */
	phoebe_get_parameter_value ("phoebe_lcno", &lcno);
	phoebe_get_parameter_value ("phoebe_rvno", &rvno);

	/* The following block reads out parameters marked for adjustment:        */
	read_in_adjustable_parameters (&to_be_adjusted, &initvals, &indices);

	for (i = 0; i < to_be_adjusted; i++)
		if (PHOEBE_parameters[indices[i]].type == TYPE_DOUBLE)
			{
			phoebe_set_parameter_value (PHOEBE_parameters[indices[i]].qualifier, initvals[i] + gsl_ran_gaussian (PHOEBE_randomizer, sigma * initvals[i]));
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
}

int find_minimum_with_dc (FILE *dc_output, PHOEBE_minimizer_feedback **feedback)
{
	/*
	 * This is WD's built-in DC algorithm and as such it doesn't depend on GSL.
	 * Macro WD_FIND_MINIMUM_WITH_DC is provided to access the fortran sub-
	 * routine.
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

	char *pars[] = {
		/*  0 */ "phoebe_spots_lat1",
		/*  1 */ "phoebe_spots_long1",
		/*  2 */ "phoebe_spots_rad1",
		/*  3 */ "phoebe_spots_temp1",
		/*  4 */ "phoebe_spots_lat2",
		/*  5 */ "phoebe_spots_long2",
		/*  6 */ "phoebe_spots_rad2",
		/*  7 */ "phoebe_spots_temp2",
		/*  8 */ "phoebe_sma",
		/*  9 */ "phoebe_ecc",
		/* 10 */ "phoebe_perr0",
		/* 11 */ "phoebe_f1",
		/* 12 */ "phoebe_f2",
		/* 13 */ "phoebe_pshift",
		/* 14 */ "phoebe_vga",
		/* 15 */ "phoebe_incl",
		/* 16 */ "phoebe_grb1",
		/* 17 */ "phoebe_grb2",
		/* 18 */ "phoebe_teff1",
		/* 19 */ "phoebe_teff2",
		/* 20 */ "phoebe_alb1",
		/* 21 */ "phoebe_alb2",
		/* 22 */ "phoebe_pot1",
		/* 23 */ "phoebe_pot2",
		/* 24 */ "phoebe_rm",
		/* 25 */ "phoebe_hjd0",
		/* 26 */ "phoebe_period",
		/* 27 */ "phoebe_dpdt",
		/* 28 */ "phoebe_dperdt",
		/* 29 */ "",
		/* 30 */ "phoebe_hla",
		/* 31 */ "phoebe_cla",
		/* 32 */ "phoebe_ld_lcx1",
		/* 33 */ "phoebe_ld_lcx2",
		/* 34 */ "phoebe_el3"
	};

	WD_DCI_parameters *params;
	int i, j, index, qindex;
	int status;

	char atmcof[255], atmcofplanck[255];

	/* Define arrays that will hold DC results; they will be allocated when a */
	/* number of parameters set for adjustment are known.                     */
	double *corrections;
	double *errors;
	double *chi2s;
	double  cfval;

	double parvalue;

	bool calcvga;
	bool calchla;
	bool cindex;
	int  rvno = 0;
	int marked_tba;

	clock_t clock_start, clock_stop;

	phoebe_debug ("entering differential corrections minimizer.\n");

	/* Before we do anything, let's check whether the setup is sane:          */

	if (!hla_request_is_sane  ()) return ERROR_MINIMIZER_HLA_REQUEST_NOT_SANE;
	if (!vga_request_is_sane  ()) return ERROR_MINIMIZER_VGA_REQUEST_NOT_SANE;
	if (!dpdt_request_is_sane ()) return ERROR_MINIMIZER_DPDT_REQUEST_NOT_SANE;

	/* Check whether third light units are Flux; DC doesn't yet support %.    */
	{
	PHOEBE_el3_units el3units;
	status = phoebe_el3_units_id (&el3units);
	if (el3units == PHOEBE_EL3_UNITS_TOTAL_LIGHT) {
		phoebe_lib_error ("handling 3rd light in total light units not yet supported by DC!\n");
		return ERROR_INVALID_EL3_UNITS;
	}
	}

	/* Is the feedback structure initialized:                                 */
	if (!feedback) return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	/* Everything seems to be ok. Fire up the stop watch:                     */
	clock_start = clock ();

	params = wd_dci_parameters_new ();
	status = read_in_wd_dci_parameters (params, &marked_tba);
	if (status != SUCCESS) return status;

	if (params->rv1data == TRUE) rvno++;
	if (params->rv2data == TRUE) rvno++;

	sprintf (atmcof,       "%s/wd/atmcof.dat",       PHOEBE_BASE_DIR);
	sprintf (atmcofplanck, "%s/wd/atmcofplanck.dat", PHOEBE_BASE_DIR);

	phoebe_get_parameter_value ("phoebe_compute_hla_switch", &calchla);
	if (params->nlc == 0) calchla = 0;

	phoebe_get_parameter_value ("phoebe_compute_vga_switch", &calcvga);
	if (rvno == 0) calcvga = 0;

	phoebe_get_parameter_value ("phoebe_cindex_switch",      &cindex);

	corrections = phoebe_malloc (marked_tba * sizeof (*corrections));
	errors      = phoebe_malloc (marked_tba * sizeof (*errors));
	chi2s       = phoebe_malloc ((params->nlc + rvno) * sizeof (*chi2s));

	if (calcvga && rvno > 0) {
		double observational_average, synthetic_average;

		if (params->rv1data && params->rv2data) {
			double obsav1, obsav2;
			status = calculate_model_vga (&synthetic_average, params->obs[0]->indep, params->obs[0]->dep, params->obs[1]->indep, params->obs[1]->dep);

			status = calculate_average (&obsav1, params->obs[0]->dep);
			status = calculate_average (&obsav2, params->obs[1]->dep);

			observational_average =
				  (double) params->obs[0]->dep->dim / (params->obs[0]->dep->dim + params->obs[1]->dep->dim) * obsav1
				+ (double) params->obs[1]->dep->dim / (params->obs[0]->dep->dim + params->obs[1]->dep->dim) * obsav2;
		} else {
			status = calculate_model_vga (&synthetic_average, params->obs[0]->indep, params->obs[0]->dep, NULL, NULL);
			status = calculate_average (&observational_average, params->obs[0]->dep);
		}
#warning FIX VGA COMPUTATION
		params->vga += observational_average - synthetic_average;
	}

	if (calchla == TRUE) {
		double observational_average, synthetic_average;

		for (i = rvno; i < rvno + params->nlc; i++) {
			calculate_average (&observational_average, params->obs[i]->dep);
			calculate_model_level (&synthetic_average, i-rvno, params->obs[i]->indep);
			params->hla[i] *= observational_average/synthetic_average;
			if (cindex) break;
		}
	}

	if (cindex) {
		for (i = rvno + 1; i < rvno + params->nlc; i++) {
			double cindex;
			phoebe_get_parameter_value ("phoebe_cindex", i-rvno, &cindex);
			params->hla[i] = params->hla[rvno] * cindex;
		}
	}

	create_dci_file ("dcin.active", params);
	WD_FIND_MINIMUM_WITH_DC (atmcof, atmcofplanck, corrections, errors, chi2s, &cfval);

	/* Allocate and fill-in the feedback structure; the number of parameter   */
	/* fields equals the number of parameters marked for adjustment plus the  */
	/* number of light curves if HLAs are marked for computation plus 1 if    */
	/* VGA is marked for computation.                                         */

	phoebe_minimizer_feedback_alloc (feedback, marked_tba+(calchla*params->nlc)+calcvga, rvno+params->nlc);

	(*feedback)->algorithm = PHOEBE_MINIMIZER_DC;
	(*feedback)->iters = 1;

	for (i = 0; i < params->nlc + rvno; i++) {
		(*feedback)->chi2s->val[i] = chi2s[i];
	}

	(*feedback)->cfval = cfval;

	/* This isn't handled yet: */
	phoebe_vector_free ((*feedback)->wchi2s);
	(*feedback)->wchi2s = NULL;

	fprintf (dc_output, "%-18s %-12s %-12s %-12s %-12s\n", "Qualifier:", "Original:", "Correction:", "   New:", "  Error:");
	fprintf (dc_output, "--------------------------------------------------------------------\n");

	index = 0;

	if (calchla == TRUE)
		for (i = rvno; i < rvno + params->nlc; i++) {
			int idx;
			phoebe_get_parameter_value ("phoebe_hla", i-rvno, &parvalue);
			fprintf (dc_output, "%s[%d]", pars[30], i-rvno+1);
			for (idx = strlen("phoebe_hla"); idx <= 12; idx++) fprintf (dc_output, " ");
			fprintf (dc_output, "%12.6lf %12.6lf %12.6lf %10s\n", parvalue, params->hla[i] - parvalue, params->hla[i], "n/a");

			phoebe_index_from_qualifier (&qindex, "phoebe_hla");
			(*feedback)-> indices->val.iarray[index] = qindex;
			(*feedback)->initvals->val[index] = parvalue;
			(*feedback)-> newvals->val[index] = params->hla[i];
			(*feedback)-> ferrors->val[index] = sqrt (-1);

			index++;
		}

	if (calcvga == TRUE) {
		phoebe_get_parameter_value ("phoebe_vga", &parvalue);
		fprintf (dc_output, "phoebe_vga      %12.6lf %12.6lf %12.6lf %10s\n", parvalue, params->vga - parvalue, params->vga, "n/a");

		phoebe_index_from_qualifier (&qindex, "phoebe_vga");
		(*feedback)-> indices->val.iarray[index] = qindex;
		(*feedback)->initvals->val[index] = parvalue;
		(*feedback)-> newvals->val[index] = params->vga;
		(*feedback)-> ferrors->val[index] = sqrt (-1);

		index++;
	}

	for (i = 0; i < 35; i++)
		if (params->tba[i] == 0) {
			if (i < 29) {
				int idx;
				status = phoebe_get_parameter_value (pars[i], &parvalue);
				if (status != SUCCESS) return status;
				fprintf (dc_output, "%s", pars[i]);
				if (strcmp (pars[i],   "phoebe_vga") == 0) { corrections[index-calchla*params->nlc-calcvga] *=   100.0; errors[index-calchla*params->nlc-calcvga] *=   100.0; }
				if (strcmp (pars[i], "phoebe_teff1") == 0) { corrections[index-calchla*params->nlc-calcvga] *= 10000.0; errors[index-calchla*params->nlc-calcvga] *= 10000.0; }
				if (strcmp (pars[i], "phoebe_teff2") == 0) { corrections[index-calchla*params->nlc-calcvga] *= 10000.0; errors[index-calchla*params->nlc-calcvga] *= 10000.0; }
				for (idx = strlen(pars[i])-3; idx <= 12; idx++) fprintf (dc_output, " ");
				fprintf (dc_output, "%12.6lf %12.6lf %12.6lf %12.6lf\n", parvalue, corrections[index-calchla*params->nlc-calcvga], parvalue + corrections[index-calchla*params->nlc-calcvga], errors[index-calchla*params->nlc-calcvga]);

				phoebe_index_from_qualifier (&qindex, pars[i]);
				(*feedback)-> indices->val.iarray[index] = qindex;
				(*feedback)->initvals->val[index] = parvalue;
				(*feedback)-> newvals->val[index] = parvalue + corrections[index-calchla*params->nlc-calcvga];
				(*feedback)-> ferrors->val[index] = errors[index-calchla*params->nlc-calcvga];

				index++;
			}
			if (i == 30) {
				double hla0 = 0.0;
				for (j = 0; j < params->nlc; j++) {
					int idx;

					phoebe_get_parameter_value (pars[i], j, &parvalue);
					if (j == 0) hla0 = parvalue + corrections[index-calchla*params->nlc-calcvga];

					fprintf (dc_output, "%s[%d]", pars[i], j+1);
					for (idx = strlen(pars[i]); idx <= 12; idx++)
						fprintf (dc_output, " ");

					if (cindex == TRUE && j != 0) {
						double cindex_val;
						phoebe_get_parameter_value ("phoebe_cindex", j, &cindex_val);
/*						phoebe_set_parameter_value (pars[i], j, hla0 * cindex_val);*/
						fprintf (dc_output, "%12.6lf %12s %12.6lf %12.6lf\n", parvalue, "n/a  ", hla0 * cindex_val, errors[index-calchla*params->nlc-calcvga]);
					}
					else {
/*						phoebe_set_parameter_value (pars[i], j, parvalue + corrections[index-calchla*params->nlc-calcvga]);*/
						fprintf (dc_output, "%12.6lf %12.6lf %12.6lf %12.6lf\n", parvalue, corrections[index-calchla*params->nlc-calcvga], parvalue + corrections[index-calchla*params->nlc-calcvga], errors[index-calchla*params->nlc-calcvga]);
					}

					phoebe_index_from_qualifier (&qindex, pars[i]);
					(*feedback)-> indices->val.iarray[index] = qindex;
					(*feedback)->initvals->val[index]        = parvalue;
					(*feedback)-> newvals->val[index]        = parvalue + corrections[index-calchla*params->nlc-calcvga];
					(*feedback)-> ferrors->val[index]        = errors[index-calchla*params->nlc-calcvga];

					index++;
				}
			}
			if (i > 30) {
				for (j = 0; j < params->nlc; j++) {
					int idx;
					phoebe_get_parameter_value (pars[i], j, &parvalue);
					fprintf (dc_output, "%s[%d]", pars[i], j+1);
					for (idx = strlen(pars[i]); idx <= 12; idx++) fprintf (dc_output, " ");
					fprintf (dc_output, "%12.6lf %12.6lf %12.6lf %12.6lf\n", parvalue, corrections[index-calchla*params->nlc-calcvga], parvalue + corrections[index-calchla*params->nlc-calcvga], errors[index-calchla*params->nlc-calcvga]);

					phoebe_index_from_qualifier (&qindex, pars[i]);
					(*feedback)-> indices->val.iarray[index] = qindex;
					(*feedback)->initvals->val[index] = parvalue;
					(*feedback)-> newvals->val[index] = parvalue + corrections[index-calchla*params->nlc-calcvga];
					(*feedback)-> ferrors->val[index] = errors[index-calchla*params->nlc-calcvga];

					index++;
				}
			}
		}

	/* Stop the clock watch and compute the total CPU time on the process:    */
	clock_stop = clock ();

	(*feedback)->cputime = (double) (clock_stop - clock_start) / CLOCKS_PER_SEC;

	wd_dci_parameters_free (params);

	free (corrections);
	free (errors);
	free (chi2s);

	phoebe_debug ("leaving differential corrections minimizer.\n");

	return SUCCESS;
}

#warning NMS_PRINT () NOT YET IMPLEMENTED
int nms_print (const char *fmt, ...)
	{
	va_list ap;
	int r;

	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
	}
