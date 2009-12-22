#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <phoebe/phoebe.h>

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter_io.h"

FILE *PHOEBE_output;

int phoebe_vector_print (PHOEBE_vector *vec)
{
	/*
	 * This function prints the contents of vector 'vec' to screen/output.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	if (!vec) {
		fprintf (PHOEBE_output, "<empty array>");
		return SUCCESS;
	}

	fprintf (PHOEBE_output, "{");
	for (i = 0; i < vec->dim - 1; i++)
		fprintf (PHOEBE_output, "%f, ", vec->val[i]);
	fprintf (PHOEBE_output, "%f}", vec->val[i]);

	return SUCCESS;
}

int phoebe_array_print (PHOEBE_array *array)
{
	/*
	 * This function prints the contents of array 'array' to screen/output.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	if (array == NULL) {
		fprintf (PHOEBE_output, "<empty array>");
		return SUCCESS;
	}

	fprintf (PHOEBE_output, "{");

	for (i = 0; i < array->dim; i++) {
		switch (array->type) {
			case TYPE_INT_ARRAY:    fprintf (PHOEBE_output, "%d",     array->  val.iarray[i]); break;
			case TYPE_DOUBLE_ARRAY: fprintf (PHOEBE_output, "%f",     array->  val.darray[i]); break;
			case TYPE_BOOL_ARRAY:   fprintf (PHOEBE_output, "%d",     array->  val.barray[i]); break;
			case TYPE_STRING_ARRAY: fprintf (PHOEBE_output, "\"%s\"", array->val.strarray[i]); break;
			default:
				phoebe_scripter_output ("exception handler invoked in phoebe_array_print (), please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		if (i == array->dim - 1) fprintf (PHOEBE_output, "}");
		else fprintf (PHOEBE_output, ", ");
	}

	return SUCCESS;
}

int phoebe_spectrum_print (PHOEBE_spectrum *spectrum)
{
	/*
	 * This function prints the contents of spectrum 'spectrum' to screen/
	 * output.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	if (!spectrum) {
		fprintf (PHOEBE_output, "<empty spectrum>");
		return SUCCESS;
	}

	if (spectrum->disp == PHOEBE_SPECTRUM_DISPERSION_LOG)
		for (i = 0; i < spectrum->data->bins; i++)
			fprintf (PHOEBE_output, "%14.4f\t%14.4f\n", 0.5*(spectrum->data->range[i]+spectrum->data->range[i+1]), spectrum->data->val[i]);
	else
		for (i = 0; i < spectrum->data->bins; i++)
			fprintf (PHOEBE_output, "%18.8f\t%18.8e\n", 0.5*(spectrum->data->range[i]+spectrum->data->range[i+1]), spectrum->data->val[i]);

	return SUCCESS;
}

int phoebe_curve_print (PHOEBE_curve *curve)
{
	/*
	 * This function prints the contents of a curve structure to screen/
	 * output.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	char *type;

	fprintf (PHOEBE_output, "\n");
	phoebe_curve_type_get_name (curve->type, &type);
	fprintf (PHOEBE_output, "  curve type:           %s\n", type);
	if (curve->passband)
		fprintf (PHOEBE_output, "  passband:             %s %s\n", curve->passband->set, curve->passband->name);
	else
		fprintf (PHOEBE_output, "  passband:             undefined\n");
	fprintf (PHOEBE_output, "  data points:          %d\n", curve->indep->dim);

	free (type);
	return SUCCESS;
}

int phoebe_minimizer_feedback_print (PHOEBE_minimizer_feedback *feedback)
{
	/*
	 * This function prints the contents of a minimizer feedback to screen/
	 * output.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i, j;
	char *algname;

	fprintf (PHOEBE_output, "\n");
	phoebe_minimizer_type_get_name (feedback->algorithm, &algname);
	fprintf (PHOEBE_output, "  algorithm:            %s\n", algname);
	if (feedback->converged)
		fprintf (PHOEBE_output, "  converged:            yes\n");
	else
		fprintf (PHOEBE_output, "  converged:            no\n");
	fprintf (PHOEBE_output, "  no. of iterations:    %d\n", feedback->iters);
	fprintf (PHOEBE_output, "  CPU time:             %2.2lf seconds\n", feedback->cputime);

	fprintf (PHOEBE_output, "  Adjusted parameters:  ");
	if (feedback->qualifiers) {
		for (i = 0; i < feedback->qualifiers->dim; i++)
			fprintf (PHOEBE_output, "%s ", feedback->qualifiers->val.strarray[i]);
		fprintf (PHOEBE_output, "\n");
	}
	else {
		fprintf (PHOEBE_output, "n/a\n");
	}

	fprintf (PHOEBE_output, "  initial values:       ");
	if (feedback->initvals) {
		for (i = 0; i < feedback->initvals->dim; i++) {
			fprintf (PHOEBE_output, "%lf ", feedback->initvals->val[i]);
		}
		fprintf (PHOEBE_output, "\n");
	}
	else {
		fprintf (PHOEBE_output, "n/a\n");
	}

	fprintf (PHOEBE_output, "  converged values:     ");
	if (feedback->newvals) {
		for (i = 0; i < feedback->newvals->dim; i++) {
			fprintf (PHOEBE_output, "%lf ", feedback->newvals->val[i]);
		}
		fprintf (PHOEBE_output, "\n");
	}
	else {
		fprintf (PHOEBE_output, "n/a\n");
	}

	fprintf (PHOEBE_output, "  formal errors:        ");
	if (feedback->newvals) {
		for (i = 0; i < feedback->ferrors->dim; i++) {
			if (isnan (feedback->ferrors->val[i]))
				fprintf (PHOEBE_output, "n/a ");
			else
				fprintf (PHOEBE_output, "%lf ", feedback->ferrors->val[i]);
		}
		fprintf (PHOEBE_output, "\n");
	}
	else {
		fprintf (PHOEBE_output, "n/a\n");
	}

	fprintf (PHOEBE_output, "  chi2 values:          ");
	if (feedback->chi2s) {
		for (i = 0; i < feedback->chi2s->dim; i++) {
			fprintf (PHOEBE_output, "%lf ", feedback->chi2s->val[i]);
		}
		fprintf (PHOEBE_output, "\n");
	}
	else {
		fprintf (PHOEBE_output, "n/a\n");
	}

	fprintf (PHOEBE_output, "  weighted chi2 values: ");
	if (feedback->wchi2s) {
		for (i = 0; i < feedback->wchi2s->dim; i++) {
			fprintf (PHOEBE_output, "%lf ", feedback->wchi2s->val[i]);
		}
		fprintf (PHOEBE_output, "\n");
	}
	else {
		fprintf (PHOEBE_output, "n/a\n");
	}

	fprintf (PHOEBE_output, "  cost function value:  %lf\n", feedback->cfval);

	fprintf (PHOEBE_output, "  correlation matrix: ");
	if (feedback->cormat) {
		fprintf (PHOEBE_output, "\n\t");
		for (i = 0; i < feedback->cormat->cols; i++) {
			for (j = 0; j < feedback->cormat->rows; j++) {
				fprintf (PHOEBE_output, "% 3.3lf ", feedback->cormat->val[i][j]);
			}
			fprintf (PHOEBE_output, "\n\t");
		}
	}
	else {
		fprintf (PHOEBE_output, "  n/a\n");
	}

	return SUCCESS;
}

int propagate_int_to_double (scripter_ast_value *val)
{
	double new;

	if (val->type != type_int)
		return /* SCRIPTER_VALUE_TYPE_NOT_INT */ -1;
	new = (double) val->value.i;
	val->type = type_double;
	val->value.d = new;

	return SUCCESS;
}

int propagate_int_to_bool (scripter_ast_value *val)
{
	bool new;

	if (val->type != type_int)
		return /* SCRIPTER_VALUE_TYPE_NOT_INT */ -1;
	
	if      (val->value.i == 0) new = FALSE;
	else if (val->value.i == 1) new = TRUE;
	else return /* SCRIPTER_CANNOT_PROPAGATE_ERROR */ -1;

	val->type = type_bool;
	val->value.b = new;

	return SUCCESS;
}

int propagate_int_to_menu_item (scripter_ast_value *val, char *qualifier)
{
	/*
	 * This propagator is called when the passed integer is the index to the
	 * parameter menu item.
	 */

	PHOEBE_parameter *par = phoebe_parameter_lookup (qualifier);
	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	if (par->kind != KIND_MENU)
		return ERROR_PARAMETER_KIND_NOT_MENU;

	if (val->value.i < 1 || val->value.i > par->menu->optno) {
		return ERROR_PARAMETER_MENU_ITEM_OUT_OF_RANGE;
	}
	
	val->type = type_string;
	val->value.str = strdup (par->menu->option[val->value.i-1]);

	return SUCCESS;
}

int propagate_int_to_vector (scripter_ast_value *val, int dim)
{
	PHOEBE_vector *new = phoebe_vector_new ();

	if (val->type != type_int)
		return /* SCRIPTER_VALUE_TYPE_NOT_INT */ -1;

	phoebe_vector_alloc (new, dim);
	phoebe_vector_pad (new, (double) val->value.i);
	
	val->type = type_vector;
	val->value.vec = new;

	return SUCCESS;
}
