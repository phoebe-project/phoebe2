#include <phoebe/phoebe.h>
#include <phoebe/phoebe_scripter.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_rng.h>

#include "polyfit.h"

scripter_ast_value scripter_polyfit (scripter_ast_list *args)
{
	scripter_ast_value out;
	scripter_ast_value *vals;
	int status, i, nobs, iter;

	polyfit_options  *options;
	polyfit_triplet  *data;
	polyfit_solution *result, *best;

	double *knots = NULL;
	double *test, chi2, chi2test, u;

	gsl_rng *r;

	out.type = type_void;

	status = scripter_command_args_evaluate (args, &vals, 1, 1, type_curve);
	if (status != SUCCESS) return out;

	options = polyfit_options_default ();

	/* Wrap this around for now: */
	data = phoebe_malloc (vals[0].value.curve->indep->dim * sizeof(*data));
	for (i = 0; i < vals[0].value.curve->indep->dim; i++) {
		data[i].x = vals[0].value.curve->indep->val[i];
		data[i].y = vals[0].value.curve->dep->val[i];
		data[i].z = 1e-3/vals[0].value.curve->weight->val[i]/vals[0].value.curve->weight->val[i];
	}
	nobs = i;

	if (!options->ann_compat)
		printf ("# %d data points read in.\n# \n", nobs);

	/* Sort the data by phase: */
	qsort (data, nobs, sizeof (*data), polyfit_sort_by_phase);

	if (options->find_knots) {
		status = polyfit_find_knots (data, nobs, &knots, options);
		if (!options->ann_compat) {
			if (status != 0)
				printf ("# * automatic search for knots failed, reverting to defaults.\n");
			else
				printf ("# * automatic search for knots successful.\n");
			printf ("# \n");
		}
	}

	/* The initial phase intervals for knots: */
	if (!knots) {
		knots = malloc (options->knots * sizeof (*knots));
		knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
	}

	/* Sort the knots in ascending order: */
	qsort (knots, options->knots, sizeof (*knots), polyfit_sort_by_value);

	if (options->find_step) {
		double diff;
		/* Step size would be the minimum width between two knots / 5: */
		diff = fabs (knots[1]-knots[0]);
		for (i = 1; i < options->knots-1; i++)
			if (fabs (knots[i+1]-knots[i]) < diff)
				diff = fabs (knots[i+1]-knots[i]);
		if (fabs (knots[i+1]-knots[0]) < diff)
			diff = fabs (knots[i+1]-knots[0]);

		options->step_size = diff / 5.0;
	}

	if (!options->ann_compat) {
		printf ("# Fitting polynomial order: %d\n", options->polyorder);
		printf ("# Initial set of knots: {");
		for (i = 0; i < options->knots-1; i++)
			printf ("%lf, ", knots[i]);
		printf ("%lf}\n", knots[i]);
		printf ("# Number of iterations for knot search: %d\n", options->iters);
		printf ("# Step size for knot search: %lf\n# \n", options->step_size);
	}

	r = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set (r, 1);

	result = polyfit_solution_init (options);

	polyfit (result, data, nobs, knots, 0, options);
	chi2 = result->chi2;

	if (!options->ann_compat)
		printf ("# Original chi2: %lf\n", result->chi2);

	test = malloc (options->knots * sizeof (*test));

	for (iter = 0; iter < options->iters; iter++) {
		for (i = 0; i < options->knots; i++) {
			u = gsl_rng_uniform (r);
			test[i] = knots[i] + options->step_size * 2 * u - options->step_size;
			if (test[i] < -0.5) test[i] += 1.0;
			if (test[i] >  0.5) test[i] -= 1.0;
		}

		polyfit (result, data, nobs, test, 0, options);

		if (result->chi2 < chi2) {
			chi2 = result->chi2;

			for (i = 0; i < options->knots; i++)
				knots[i] = test[i];
		}
	}

	if (!options->ann_compat)
		printf ("# Final chi2:    %lf\n# \n", chi2);

	/* Final run: */
	polyfit (result, data, nobs, knots, 0, options);
	polyfit_print (result, options, 1);

	gsl_rng_free (r);
	free (test);
	free (knots);
	free (data);

	polyfit_solution_free (result, options);
	polyfit_options_free (options);

	scripter_ast_value_array_free (vals, 1);
	return out;
}

int phoebe_plugin_start ()
{
	printf ("polyfit plugin started.\n");

	scripter_command_register ("polyfit", scripter_polyfit);
	return SUCCESS;
}

int phoebe_plugin_stop ()
{
	printf ("polyfit plugin stopped.\n");
	return SUCCESS;
}
