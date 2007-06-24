#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_global.h"

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_calculations.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

PHOEBE_vector *phoebe_vector_new ()
{
	/*
	 * This function initializes a vector for allocation.
	 */

	PHOEBE_vector *vec = phoebe_malloc (sizeof (*vec));

	vec->dim = 0;
	vec->val = NULL;

	return vec;
}

PHOEBE_vector *phoebe_vector_new_from_qualifier (char *qualifier)
{
	/*
	 * This function returns a newly allocated vector the values of which are
	 * taken from the array parameter represented by its qualifier.
	 *
	 * If an error occured, NULL is returned.
	 */

	int i;
	PHOEBE_vector *vec;

	PHOEBE_parameter *par = phoebe_parameter_lookup (qualifier);
	if (!par) return NULL;
	if (par->type != TYPE_DOUBLE_ARRAY) return NULL;
	if (par->value.vec->dim == 0) return NULL;

	vec = phoebe_vector_new ();
	phoebe_vector_alloc (vec, par->value.vec->dim);

	for (i = 0; i < par->value.vec->dim; i++)
		vec->val[i] = par->value.vec->val[i];

	return vec;
}

PHOEBE_vector *phoebe_vector_new_from_column (char *filename, int col)
{
	/*
	 * This function reads in the 'col'-th column from file 'filename',
	 * parses it and stores it into the returned vector. If a parse error
	 * occured, NULL is returned.
	 */

	FILE *input;
	PHOEBE_vector *vec;
	int i, linecount = 1;

	input = fopen (filename, "r");
	if (input == NULL) return NULL;

	vec = phoebe_vector_new ();
	while (!feof (input)) {
		double val;
		char line[255];
		char *delimeter = line;

		fgets (line, 254, input);
		if (feof (input)) break;

		/* Remove the trailing newline (unix or dos):                         */
		line[strlen(line)-1] = '\0';
		if (strchr (line, 13) != NULL) (strchr (line, 13))[0] = '\0';

		/* Remove comments (if any):                                          */
		if (strchr (line, '#') != NULL) (strchr (line, '#'))[0] = '\0';

		/* Remove any leading whitespaces and empty lines:                    */
		while ( (delimeter[0] == ' ' || delimeter[0] == '\t') && delimeter[0] != '\0') delimeter++;
		if (delimeter[0] == '\0') {
			linecount++;
			continue;
		}

		for (i = 1; i < col; i++) {
			while (delimeter[0] != ' ' && delimeter[0] != '\t' && delimeter[0] != '\0') delimeter++;
			while ( (delimeter[0] == ' ' || delimeter[0] == '\t') && delimeter[0] != '\0') delimeter++;
			if (delimeter[0] == '\0') {
				phoebe_lib_error ("column %d in line %d cannot be read, skipping.\n", col, linecount);
				break;
			}
		}

		if (delimeter[0] != '\0')
			if (sscanf (delimeter, "%lf", &val) == 1) {
				phoebe_vector_realloc (vec, vec->dim + 1);
				vec->val[vec->dim-1] = val;
			}

		linecount++;
	}
	fclose (input);

	return vec;
}

PHOEBE_vector *phoebe_vector_duplicate (PHOEBE_vector *vec)
{
	/*
	 * This function makes a duplicate copy of vector 'vec'.
	 */

	int i;
	PHOEBE_vector *new;

	if (!vec) return NULL;

	new = phoebe_vector_new ();
	phoebe_vector_alloc (new, vec->dim);
	for (i = 0; i < vec->dim; i++)
		new->val[i] = vec->val[i];

	return new;
}

int phoebe_vector_alloc (PHOEBE_vector *vec, int dimension)
{
	/*
	 * This function allocates storage memory for a vector of 'dimension'.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_ALREADY_ALLOCATED
	 *   ERROR_VECTOR_INVALID_DIMENSION
	 *   SUCCESS
	 */

	if (vec->dim != 0)
		return ERROR_VECTOR_ALREADY_ALLOCATED;

	if (dimension < 1)
		return ERROR_VECTOR_INVALID_DIMENSION;

	vec->dim = dimension;
	vec->val = phoebe_malloc (sizeof (*(vec->val)) * dimension);
	return SUCCESS;
}

int phoebe_vector_realloc (PHOEBE_vector *vec, int dimension)
{
	/*
	 * This function reallocates storage memory for a vector of 'dimension'.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_INVALID_DIMENSION
	 *   SUCCESS
	 */

	if (dimension < 1)
		return ERROR_VECTOR_INVALID_DIMENSION;

	vec->dim = dimension;
	vec->val = phoebe_realloc (vec->val, sizeof (*(vec->val)) * dimension);
	return SUCCESS;
}

int phoebe_vector_pad (PHOEBE_vector *vec, double value)
{
	/*
	 * This function pads all vector components with 'value'.
	 */

	int i;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = value;

	return SUCCESS;
}

int phoebe_vector_free (PHOEBE_vector *vec)
{
	/*
	 * This function frees the storage memory allocated for vector 'vec'.
	 */

	if (!vec) return SUCCESS;
	if (vec->val) free (vec->val);
	free (vec);
	return SUCCESS;
}

int phoebe_vector_add (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function adds vectors fac1 and fac2 and returns the sum vector.
	 *
	 * Return values:
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	phoebe_vector_alloc (result, fac1->dim);
	for (i = 0; i < fac1->dim; i++)
		result->val[i] = fac1->val[i] + fac2->val[i];

	return SUCCESS;
}

int phoebe_vector_subtract (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function subtracts vectors fac1 and fac2 and returns the difference
	 * vector.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	phoebe_vector_alloc (result, fac1->dim);
	for (i = 0; i < fac1->dim; i++)
		result->val[i] = fac1->val[i] - fac2->val[i];

	return SUCCESS;
}

int phoebe_vector_multiply (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function multiplies vectors fac1 and fac2 and returns the product
	 * vector.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	phoebe_vector_alloc (result, fac1->dim);
	for (i = 0; i < fac1->dim; i++)
		result->val[i] = fac1->val[i] * fac2->val[i];

	return SUCCESS;
}

int phoebe_vector_divide (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function divides vectors fac1 and fac2 and returns the quotient
	 * vector.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	phoebe_vector_alloc (result, fac1->dim);
	for (i = 0; i < fac1->dim; i++)
		result->val[i] = fac1->val[i] / fac2->val[i];

	return SUCCESS;
}

int phoebe_vector_raise (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function raises all elements of vector fac1 to the factor of element
	 * of the vector fac2, basically new[i] = fac1[i]^fac2[i].
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	phoebe_vector_alloc (result, fac1->dim);
	for (i = 0; i < fac1->dim; i++)
		result->val[i] = pow (fac1->val[i], fac2->val[i]);

	return SUCCESS;
}

int phoebe_vector_multiply_by (PHOEBE_vector *fac1, double factor)
{
	/*
	 * This function multiplies all elements of the vector fac1 with the scalar
	 * value 'factor'.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	for (i = 0; i < fac1->dim; i++)
		fac1->val[i] = fac1->val[i] * factor;

	return SUCCESS;
}

int phoebe_vector_dot_product (double *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function returns the scalar (dot) product of the two vectors.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;
	*result = 0.0;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < fac1->dim; i++)
		*result += fac1->val[i] * fac2->val[i];
	*result = sqrt (*result);

	return SUCCESS;
}

int phoebe_vector_vec_product (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/*
	 * This function returns the vector product of the two vectors.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_DIMENSION_NOT_THREE
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	if (fac1->dim != 3)
		return ERROR_VECTOR_DIMENSION_NOT_THREE;

	phoebe_vector_alloc (result, fac1->dim);
	result->val[0] = fac1->val[1] * fac2->val[2] - fac1->val[2] * fac2->val[1];
	result->val[1] = fac1->val[2] * fac2->val[0] - fac1->val[0] * fac2->val[2];
	result->val[2] = fac1->val[0] * fac2->val[1] - fac1->val[1] * fac2->val[0];

	return SUCCESS;
}

int phoebe_vector_submit (PHOEBE_vector *result, PHOEBE_vector *vec, double func ())
{
	/*
	 * This function calculates the functional value of func() for each element
	 * of the vector individually.
	 */

	int i;

	phoebe_vector_alloc (result, vec->dim);
	for (i = 0; i < vec->dim; i++)
		result->val[i] = func (vec->val[i]);

	return SUCCESS;
}

int phoebe_vector_norm (double *result, PHOEBE_vector *vec)
{
	/*
	 * This function returns the norm of the vector.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	*result = 0.0;
	for (i = 0; i < vec->dim; i++)
		*result += vec->val[i] * vec->val[i];
	*result = sqrt (*result);

	return SUCCESS;
}

int phoebe_vector_dim (int *result, PHOEBE_vector *vec)
{
	/*
	 * This function returns the dimension of the vector.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	*result = vec->dim;
	return SUCCESS;
}

int phoebe_vector_randomize (PHOEBE_vector *result, double limit)
{
	/*
	 * This function fills all vector elements with random numbers from the
	 * interval [0, limit]. 'limit' may also be negative, then it's [limit, 0].
	 * The vector must be allocated prior to calling this function.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_IS_EMPTY
	 *   SUCCESS
	 */

	int i;

	if (result->dim == 0) return ERROR_VECTOR_IS_EMPTY;

	for (i = 0; i < result->dim; i++)
		result->val[i] = limit * rand() / RAND_MAX;

	return SUCCESS;
}

int phoebe_vector_min_max (PHOEBE_vector *vec, double *min, double *max)
{
	/*
	 * This function scans through the dataset and assigns the minimal and the
	 * maximal value encountered to 'min' and 'max' variables.
	 */

	int i;
	*min = *max = vec->val[0];

	for (i = 1; i < vec->dim; i++) {
		if (*min > vec->val[i]) *min = vec->val[i];
		if (*max < vec->val[i]) *max = vec->val[i];
	}

	return SUCCESS;
}

int phoebe_vector_rescale (PHOEBE_vector *vec, double ll, double ul)
{
	/*
	 * This function rescales the values of elements in the vector vec to the
	 * [ll, ul] interval. Usually this is useful to map the weights to the
	 * [0.01, 10.0] interval that is suitable for DC.
	 *
	 * Return values:
	 *
	 *   ERROR_VECTOR_NOT_INITIALIZED
	 *   ERROR_VECTOR_IS_EMPTY
	 *   ERROR_VECTOR_INVALID_LIMITS
	 *   SUCCESS
	 */

	int i, status;
	double vmin, vmax;

	if (!vec)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec->dim == 0)
		return ERROR_VECTOR_IS_EMPTY;
	if (ll >= ul)
		return ERROR_VECTOR_INVALID_LIMITS;

	status = phoebe_vector_min_max (vec, &vmin, &vmax);
	if (status != SUCCESS)
		return status;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = ll + (vec->val[i]-vmin)/(vmax-vmin)*(ul-ll);

	return SUCCESS;
}

bool phoebe_vector_compare (PHOEBE_vector *vec1, PHOEBE_vector *vec2)
{
	/*
	 * This function compares two passed vectors. It returns TRUE if all vector
	 * elements are the same; it returns FALSE otherwise. The comparison is
	 * done by comparing the difference of both elements to EPSILON.
	 */

	int i;
	if (vec1->dim != vec2->dim) return FALSE;

	for (i = 0; i < vec1->dim; i++)
		if (fabs (vec1->val[i] - vec2->val[i]) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;

	return TRUE;
}

int phoebe_vector_less_than (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2)
{
	/*
	 * This function tests whether *all* vector elements of vec1 are less
	 * than their respective counterparts of vec2. If so, TRUE is returned,
	 * otherwise FALSE is returned.
	 * 
	 * Return values:
	 *
	 *   ERROR_VECTOR_NOT_INITIALIZED
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (!vec1 || !vec2) return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec1->dim != vec2->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < vec1->dim; i++)
		if (vec1->val[i] >= vec2->val[i]) {
			*result = FALSE;
			return SUCCESS;
		}

	*result = TRUE;
	return SUCCESS;
}

int phoebe_vector_leq_than (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2)
{
	/*
	 * This function tests whether *all* vector elements of vec1 are less or
	 * equal to their respective counterparts of vec2. If so, TRUE is returned,
	 * otherwise FALSE is returned.
	 * 
	 * Return values:
	 *
	 *   ERROR_VECTOR_NOT_INITIALIZED
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH;
	 *   SUCCESS
	 */

	int i;

	if (!vec1 || !vec2) return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec1->dim != vec2->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < vec1->dim; i++)
		if (vec1->val[i] > vec2->val[i]) {
			*result = FALSE;
			return SUCCESS;
		}

	*result = TRUE;
	return SUCCESS;
}

int phoebe_vector_greater_than (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2)
{
	/*
	 * This function tests whether *all* vector elements of vec1 are greater
	 * than their respective counterparts of vec2. If so, TRUE is returned,
	 * otherwise FALSE is returned.
	 * 
	 * Return values:
	 *
	 *   ERROR_VECTOR_NOT_INITIALIZED
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (!vec1 || !vec2) return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec1->dim != vec2->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < vec1->dim; i++)
		if (vec1->val[i] <= vec2->val[i]) {
			*result = FALSE;
			return SUCCESS;
		}

	*result = TRUE;
	return SUCCESS;
}

int phoebe_vector_geq_than (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2)
{
	/*
	 * This function tests whether *all* vector elements of vec1 are greater
	 * or equal to their respective counterparts of vec2. If so, TRUE is
	 * returned, otherwise FALSE is returned.
	 * 
	 * Return values:
	 *
	 *   ERROR_VECTOR_NOT_INITIALIZED
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH
	 *   SUCCESS
	 */

	int i;

	if (!vec1 || !vec2) return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec1->dim != vec2->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < vec1->dim; i++)
		if (vec1->val[i] < vec2->val[i]) {
			*result = FALSE;
			return SUCCESS;
		}

	*result = TRUE;
	return SUCCESS;
}

int phoebe_vector_append_element (PHOEBE_vector *vec, double val)
{
	/*
	 * This function appends an element with value 'val' to the vector 'vec'.
	 */

	phoebe_vector_realloc (vec, vec->dim+1);
	vec->val[vec->dim-1] = val;
	return SUCCESS;
}

int phoebe_vector_remove_element (PHOEBE_vector *vec, int index)
{
	/*
	 * This function removes the 'index'-th element from vector 'vec'.
	 *
	 * Return values:
	 * 
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int i;

	if (index >= vec->dim) return ERROR_INDEX_OUT_OF_RANGE;

	for (i = index; i < vec->dim; i++)
		vec->val[i] = vec->val[i+1];
	phoebe_vector_realloc (vec, vec->dim-1);
	return SUCCESS;
}

/******************************************************************************/

PHOEBE_hist *phoebe_hist_new ()
{
	/*
	 * This function initializes a new histogram.
	 */

	PHOEBE_hist *hist = phoebe_malloc (sizeof (*hist));

	hist->bins  = 0;
	hist->range = NULL;
	hist->val   = NULL;

	return hist;
}

PHOEBE_hist *phoebe_hist_new_from_arrays (int bins, double *binarray, double *valarray)
{
	/*
	 * This function stores the arrays 'bins' and 'vals' into a histogram.
	 * The dimensions of the two arrays must be the same.
	 */

	int i;
	PHOEBE_hist *hist;
	
	hist = phoebe_hist_new ();
	phoebe_hist_alloc (hist, bins);

	hist->range[0] = binarray[0] - (binarray[1]-binarray[0])/2;
	hist->val[0]   = valarray[0];
	for (i = 1; i < bins; i++) {
		hist->range[i] = (binarray[i-1] + binarray[i])/2;
		hist->val[i] = valarray[i];
	}
	hist->range[bins] = binarray[bins-1] + (binarray[bins-1]-binarray[bins-2])/2;

	return hist;
}

PHOEBE_hist *phoebe_hist_new_from_file (char *filename)
{
	/*
	 * This function reads in the two (first) columns from file 'filename',
	 * parses them and stores them into the returned histogram. If a parse
	 * error occured, NULL is returned.
	 */

	FILE *input;
	PHOEBE_hist *hist;

	int linecount = 1, dim = 0;
	double bin, val;
	double *binarray = NULL, *valarray = NULL;

	char line[255];
	char *lineptr;

	input = fopen (filename, "r");
	if (!input) return NULL;

	while (!feof (input)) {
		fgets (line, 254, input);
		if (feof (input)) break;
		lineptr = line;

		/* Remove the trailing newline (unix or dos):                         */
		line[strlen(line)-1] = '\0';
		if (strchr (line, 13)) (strchr (line, 13))[0] = '\0';

		/* Remove comments (if any):                                          */
		if (strchr (line, '#')) (strchr (line, '#'))[0] = '\0';

		/* Remove any leading whitespaces and empty lines:                    */
		while ( (*lineptr == ' ' || *lineptr == '\t') && *lineptr != '\0') lineptr++;
		if (*lineptr == '\0') {
			linecount++;
			continue;
		}

		if (sscanf (lineptr, "%lf %lf", &bin, &val) == 2) {
			dim++;
			binarray = phoebe_realloc (binarray, dim * sizeof (*binarray));
			valarray = phoebe_realloc (valarray, dim * sizeof (*valarray));
			binarray[dim-1] = bin;
			valarray[dim-1] = val;
		}

		linecount++;
	}
	fclose (input);

	if (dim == 0)
		return NULL;

	hist = phoebe_hist_new_from_arrays (dim, binarray, valarray);

	free (binarray);
	free (valarray);

	return hist;
}

PHOEBE_hist *phoebe_hist_duplicate (PHOEBE_hist *hist)
{
	/*
	 * This function duplicates the contents of the passed histogram.
	 */

	int i;
	PHOEBE_hist *duplicate;

	if (!hist) return NULL;

	duplicate = phoebe_hist_new ();
	phoebe_hist_alloc (duplicate, hist->bins);
	for (i = 0; i < hist->bins; i++) {
		duplicate->range[i] = hist->range[i];
		duplicate->val[i]   = hist->val[i];
	}
	duplicate->range[i] = hist->range[i];

	return duplicate;
}

int phoebe_hist_alloc (PHOEBE_hist *hist, int bins)
{
	/*
	 * This function allocates memory for all histogram structures.
	 */

	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (hist->bins != 0)
		return ERROR_HIST_ALREADY_ALLOCATED;
	if (bins < 1)
		return ERROR_HIST_INVALID_DIMENSION;

	hist->bins  = bins;
	hist->range = phoebe_malloc ( (bins+1) * sizeof (*hist->range));
	hist->val   = phoebe_malloc (  bins    * sizeof (*hist->val));
	return SUCCESS;
}

int phoebe_hist_realloc (PHOEBE_hist *hist, int bins)
{
	/*
	 * This function reallocates memory for all histogram structures.
	 */

	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (bins < 1)
		return ERROR_HIST_INVALID_DIMENSION;

	hist->bins  = bins;
	hist->range = phoebe_realloc (hist->range, (bins+1) * sizeof (*hist->range));
	hist->val   = phoebe_realloc (hist->val,    bins    * sizeof (*hist->val));
	return SUCCESS;
}

int phoebe_hist_free (PHOEBE_hist *hist)
{
	/*
	 * This function frees the contents of the histogram.
	 */

	if (!hist) return SUCCESS;

	if (hist->range) free (hist->range);
	if (hist->val)   free (hist->val);
	free (hist);

	return SUCCESS;
}

int phoebe_hist_set_ranges (PHOEBE_hist *hist, PHOEBE_vector *bin_centers)
{
	/*
	 * This function converts central bin positions to bin ranges.
	 */

	int i;

	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (!bin_centers)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (bin_centers->dim != hist->bins)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 1; i < bin_centers->dim; i++)
		hist->range[i] = 0.5 * (bin_centers->val[i-1] + bin_centers->val[i]);
	hist->range[0] = 2 * bin_centers->val[0] - hist->range[1];
	hist->range[bin_centers->dim] = 2 * bin_centers->val[bin_centers->dim-1] - hist->range[bin_centers->dim-1];

	return SUCCESS;
}

int phoebe_hist_set_values (PHOEBE_hist *hist, PHOEBE_vector *values)
{
	/*
	 * This function sets bin values to the passed vector elements.
	 */

	int i;

	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (!values)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (values->dim != hist->bins)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < values->dim; i++)
		hist->val[i] = values->val[i];

	return SUCCESS;
}

int phoebe_hist_get_bin_centers (PHOEBE_hist *hist, PHOEBE_vector *bin_centers)
{
	/*
	 * This function converts histogram ranges to bin centers.
	 */

	int i;

	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (!bin_centers)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (bin_centers->dim != hist->bins)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	for (i = 0; i < hist->bins; i++)
		bin_centers->val[i] = 0.5 * (hist->range[i] + hist->range[i+1]);

	return SUCCESS;
}

int phoebe_hist_get_bin (int *bin, PHOEBE_hist *hist, double r)
{
	/*
	 * This function assigns a consecutive bin number that corresponds to the
	 * range parameter r. If the range parameter is smaller than the first
	 * histogram range or if it is larger than the last histogram range,
	 * ERROR_HIST_OUT_OF_RANGE state is returned.
	 */

	if (r < hist->range[0] || r >= hist->range[hist->bins])
		return ERROR_HIST_OUT_OF_RANGE;

	*bin = 0;
	while (r >= hist->range[*bin]) (*bin)++;
	(*bin)--;

	return SUCCESS;
}

int phoebe_hist_crop (PHOEBE_hist *hist, double ll, double ul)
{
	/*
	 * This function takes the histogram 'in' and crops it to the [ll, ul]
	 * interval.
	 */

	int i = 0, j = 0, k, bins;
	double *range, *val;

	/* Error handling: */
	if (!hist) return ERROR_HIST_NOT_INITIALIZED;
	if (hist->bins == 0) return ERROR_HIST_NOT_ALLOCATED;
	if (ll >= ul) return ERROR_HIST_INVALID_RANGES;
	if (ll > hist->range[hist->bins] || ul < hist->range[0]) return ERROR_HIST_INVALID_RANGES;

	/* Fast forward to the cropped interval: */
	while (hist->range[i] <  ll) i++; j = i;
	while (hist->range[j] <= ul) j++; j--;

	/* If the inverval is narrower than the bin width, bail out: */
	if (i == j)
		return ERROR_SPECTRUM_INVALID_RANGE;

	/* Allocate new arrays: */
	 bins = j-i;
	range = phoebe_malloc ((bins+1) * sizeof (*range));
	  val = phoebe_malloc ( bins    * sizeof (*val));

	/* Copy the values to the new arrays: */
	for (k = 0; k < bins; k++) {
		range[k] = hist->range[i+k];
		  val[k] = hist->val[i+k];
	}
	range[k] = hist->range[i+k];

	/* Free the old arrays and point the structure to the new arrays: */
	free (hist->range); free (hist->val);
	hist->bins = bins; hist->range = range; hist->val = val;

	/* Done, return success! */
	return SUCCESS;
}

int phoebe_hist_evaluate (double *y, PHOEBE_hist *hist, double x)
{
	/*
	 * This is a representation of a histogram by a continuous function
	 * obtained by linear interpolation (and extrapolation at the edges)
	 * that is evaluated in value x.
	 */

	int i = 0;
	double n0, n1, val;

	/* Set the y value to 0, in case of out-of-range evaluation: */
	*y = 0.0;

	/* Error handling: */
	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (hist->bins == 0)
		return ERROR_HIST_NOT_ALLOCATED;
	if (x < hist->range[0] || x > hist->range[hist->bins])
		return ERROR_HIST_OUT_OF_RANGE;

	/*
	 * The nodes are bin centers rather than ranges, so we need to fast-
	 * forward accordingly. The nodes after this will be:
	 *
	 *   n0 = 1/2 (r_i + r_{i+1}), n1 = 1/2 (r_{i+1} + r_{i+2})
	 *
	 * If i == 0 or i == N, we must extrapolate.
	 */

	while (x > 0.5*(hist->range[i]+hist->range[i+1])) {
		i++;
		if (i == hist->bins) break;
	}
	if (i != hist->bins) i--;

	if (i == -1) {                      /* Extrapolation on the left border:  */
		n0 = 0.5 * (hist->range[0] + hist->range[1]);
		n1 = 0.5 * (hist->range[1] + hist->range[2]);
		val = hist->val[0] + (x-n0)/(n1-n0) * (hist->val[1]-hist->val[0]);
	}
	else if (i == hist->bins) {         /* Extrapolation on the right border: */
		n0 = 0.5 * (hist->range[hist->bins-2] + hist->range[hist->bins-1]);
		n1 = 0.5 * (hist->range[hist->bins-1] + hist->range[hist->bins]);
		val = hist->val[hist->bins-2] + (x-n0)/(n1-n0) * (hist->val[hist->bins-1]-hist->val[hist->bins-2]);
	}
	else {                              /* Interpolation:                     */
		n0 = 0.5 * (hist->range[i]   + hist->range[i+1]);
		n1 = 0.5 * (hist->range[i+1] + hist->range[i+2]);
		val = hist->val[i] + (x-n0)/(n1-n0) * (hist->val[i+1]-hist->val[i]);
	}

	*y = val;

	return SUCCESS;
}

int phoebe_hist_pad (PHOEBE_hist *hist, double val)
{
	/*
	 * This function pads the values of the histogram to value 'val'.
	 */

	int i;

	if (!hist)
		return ERROR_HIST_NOT_INITIALIZED;
	if (hist->bins == 0)
		return ERROR_HIST_NOT_ALLOCATED;

	for (i = 0; i < hist->bins; i++)
		hist->val[i] = val;

	return SUCCESS;
}

int phoebe_hist_rebin (PHOEBE_hist *out, PHOEBE_hist *in, PHOEBE_hist_rebin_type type)
{
	/*
	 * This function takes the input histrogram 'in' and rebins its contents
	 * to the output histogram 'out'.
	 */

	int i = 0, j = 0;
	double ll, ul, yll, yul;
	PHOEBE_vector *centers;

	/* First let's do some error handling:                                    */
	if (!in || !out) return ERROR_HIST_NOT_INITIALIZED;
	if (in->bins == 0 || out->bins == 0) return ERROR_HIST_NOT_ALLOCATED;
	if (in->range[0] > out->range[out->bins] || out->range[0] > in->range[in->bins]) return ERROR_HIST_NO_OVERLAP;

	/*
	 * There are two ways how to rebin the histogram:
	 *
	 *   1) conserve the values and
	 *   2) conserve the value densities.
	 *
	 * The first option is better if we are degrading the histogram, and the
	 * second option is better if we are oversampling the histogram. The
	 * approach is similar yet distinct to the point that it is easier to
	 * switch on the type and do the computation completely separately.
	 */

	switch (type) {
		case PHOEBE_HIST_CONSERVE_DENSITY:
			/* Get bin centers: */
			centers = phoebe_vector_new ();
			phoebe_vector_alloc (centers, in->bins);
			phoebe_hist_get_bin_centers (in, centers);

			/* Pad the output histogram with zeroes: */
			phoebe_hist_pad (out, 0.0);

			/* Fast-forward the indices to overlap: */
			while (out->range[i+1] <  in->range[0]) i++;
			while ( in->range[j+1] < out->range[0]) j++;

			/*
			 * If there are bins before centers->val[0], we have to extra-
			 * polate:
			 */

			while (out->range[i] < centers->val[0]) {
				 ll = max (in->range[0], out->range[i]);
				 ul = min (centers->val[0], out->range[i+1]);
				yll = in->val[0] + (ll-centers->val[0])/(centers->val[1]-centers->val[0]) * (in->val[1]-in->val[0]);
				yul = in->val[0] + (ul-centers->val[0])/(centers->val[1]-centers->val[0]) * (in->val[1]-in->val[0]);
				out->val[i] += 0.5*(ul-ll)/(in->range[1]-in->range[0])*(yll+yul);
				phoebe_debug ("E: %d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, ll, ul, yll, yul, out->val[i]);
				i++;
			};
			if (i > 0) i--;

			/* Do the interpolated rebinning: */
			for ( ; i < out->bins; i++) {
				do {
					 ll = max (centers->val[j],   out->range[i]);
					 ul = min (centers->val[j+1], out->range[i+1]);
					yll = in->val[j] + (ll-centers->val[j])/(centers->val[j+1]-centers->val[j]) * (in->val[j+1]-in->val[j]);
					yul = in->val[j] + (ul-centers->val[j])/(centers->val[j+1]-centers->val[j]) * (in->val[j+1]-in->val[j]);
					out->val[i] += 0.5*(ul-ll)/(in->range[j+1]-in->range[j])*(yll+yul);
					phoebe_debug ("I: %d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, ll, ul, yll, yul, out->val[i]);
					j++;
					if (j == in->bins-1) break;
				} while (centers->val[j] < out->range[i+1]);
				if (centers->val[j] < out->range[i+1] && j == in->bins-1) break;
				j--;
			}

			/*
			 * If there are bins after centers->val[N-1], we have to extra-
			 * polate:
			 */

			while (i < out->bins && out->range[i] < in->range[in->bins]) {
				 ll = max (centers->val[j], out->range[i]);
				 ul = min (in->range[j+1], out->range[i+1]);
				yll = in->val[j-1] + (ll-centers->val[j-1])/(centers->val[j]-centers->val[j-1]) * (in->val[j]-in->val[j-1]);
				yul = in->val[j-1] + (ul-centers->val[j-1])/(centers->val[j]-centers->val[j-1]) * (in->val[j]-in->val[j-1]);
				out->val[i] += 0.5*(ul-ll)/(in->range[j+1]-in->range[j])*(yll+yul);
				phoebe_debug ("E: %d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, ll, ul, yll, yul, out->val[i]);
				i++;
			};

			phoebe_vector_free (centers);
		break;
		case PHOEBE_HIST_CONSERVE_VALUES:
			/* Pad the output histogram with zeroes: */
			phoebe_hist_pad (out, 0.0);

			/* Fast-forward the indices to overlap: */
			while (out->range[i+1] < in->range[0]) i++;
			while (in->range[j+1] < out->range[0]) j++;

			/* Do the resampling:                                                     */
			for ( ; i < out->bins; i++) {
				do {
					ll = max (in->range[j], out->range[i]);
					ul = min (in->range[j+1], out->range[i+1]);
					out->val[i] += (ul - ll) / (in->range[j+1] - in->range[j]) * in->val[j];
					j++;
					if (j == in->bins) break;
				} while (in->range[j] < out->range[i+1]);
				if (in->range[j] < out->range[i+1] && j == in->bins) break;
				j--;
			}	
		break;
		default:
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	/* Hopefully everything went ok. ;) The resampling is done!               */
	return SUCCESS;
}

int phoebe_hist_integrate (double *integral, PHOEBE_hist *hist, double ll, double ul)
{
	/*
	 * This function "integrates" the passed histogram hist from ll to ul.
	 */

	int i, ll_index, ul_index;
	int status;

	status = phoebe_hist_get_bin (&ll_index, hist, ll);
	if (status != SUCCESS) return status;

	/*
	 * We need to make an explicit check here because the upper interval range
	 * in a histogram is exclusive, and usually we integrate upper-limit-
	 * inclusively.
	 */

	if (ul == hist->range[hist->bins])
		ul_index = hist->bins-1;
	else {
		status = phoebe_hist_get_bin (&ul_index, hist, ul);
		if (status != SUCCESS) return status;
	}

	*integral = 0.0;

	*integral += hist->val[ll_index] * (hist->range[ll_index+1]-ll);
	for (i = ll_index+1; i < ul_index; i++) {
		*integral += hist->val[i] * (hist->range[i+1]-hist->range[i]);
	}
	*integral += hist->val[i] * (ul-hist->range[i]);

	return SUCCESS;
}

int phoebe_hist_shift (PHOEBE_hist *hist, double shift)
{
	/*
	 * This function shifts the contents of the histogram hist in pixel-space
	 * by the passed shift. If the shift is positive, the contents are shifted
	 * to the right; if negative, they are shifted to the left.
	 */

	PHOEBE_hist *copy = phoebe_hist_duplicate (hist);
	int    int_shift = fabs ((int) shift);
	double frac_shift = fabs (frac (shift));
	int i;

	if (shift > 0) {
		for (i = 0; i < int_shift; i++)
			copy->val[i] = 0.0;
		copy->val[i] = hist->val[0] * (1.0-frac_shift); i++;
		for ( ; i < hist->bins; i++)
			copy->val[i] = hist->val[i-int_shift-1] * frac_shift
			             + hist->val[i-int_shift] * (1.0 - frac_shift);
	}

	if (shift < 0) {
		for (i = 0; i < hist->bins-int_shift-1; i++)
			copy->val[i] = hist->val[i+int_shift]   * (1.0-frac_shift)
			             + hist->val[i+int_shift+1] * frac_shift;
		copy->val[i] = hist->val[i+int_shift] * (1.0-frac_shift); i++;
		for ( ; i < hist->bins; i++)
			copy->val[i] = 0.0;
	}

	for (i = 0; i < hist->bins; i++)
		hist->val[i] = copy->val[i];

	phoebe_hist_free (copy);

	return SUCCESS;
}

int phoebe_hist_correlate (double *cfval, PHOEBE_hist *h1, PHOEBE_hist *h2, double sigma1, double sigma2, double ll, double ul, double xi)
{
	/*
	 * This function computes the correlation of the passed histograms h1 and
	 * and h2 in pixel space.
	 *
	 *   Corr (h1, h2) (xi) = \int_x0^xN h1(x) h2(x-xi) dx
	 *                      = 1/(N \sig_h1 \sig_h2) \sum_j h1(x_j) h2(x_j-xi)
	 */

	int i;
	PHOEBE_hist *h2s;
	int llidx, ulidx;

	/* Error checking: */
	if (!h1 || !h2)
		return ERROR_HIST_NOT_INITIALIZED;
	if (h1->bins == 0 || h2->bins == 0)
		return ERROR_HIST_NOT_ALLOCATED;
	if (h1->range[h1->bins] < ll || h2->range[h2->bins] < ll)
		return ERROR_HIST_INVALID_RANGES;

	/* Shift a second histogram: */
	h2s = phoebe_hist_duplicate (h2);
	phoebe_hist_shift (h2s, xi);

	/* Find the starting and ending indices: */
	llidx = 0; ulidx = 0;
	while (h1->range[llidx] < ll) llidx++;
	while (h1->range[h1->bins-ulidx-1] > ul) ulidx++;
	ulidx = h1->bins-ulidx-1;

	*cfval = 0.0;
	for (i = llidx; i <= ulidx; i++)
		*cfval += h1->val[i] * h2s->val[i];
	*cfval /= (ulidx-llidx+1)*sigma1*sigma2;

	phoebe_hist_free (h2s);

	return SUCCESS;
}

/******************************************************************************/

PHOEBE_array *phoebe_array_new (PHOEBE_type type)
{
	/*
	 * This function initializes a new array of PHOEBE_type type. It does *not*
	 * allocate any space, use phoebe_array_alloc () function for that.
	 */

	PHOEBE_array *array;

	if (type != TYPE_INT_ARRAY    && type != TYPE_BOOL_ARRAY &&
		type != TYPE_DOUBLE_ARRAY && type != TYPE_STRING_ARRAY) {
		phoebe_lib_error ("phoebe_array_new (): passed type not an array, aborting.\n");
		return NULL;
	}

	array = phoebe_malloc (sizeof (*array));

	array->dim  = 0;
	array->type = type;

	switch (type) {
		case TYPE_INT_ARRAY:    array->val.iarray   = NULL; break;
		case TYPE_DOUBLE_ARRAY: array->val.darray   = NULL; break;
		case TYPE_BOOL_ARRAY:   array->val.barray   = NULL; break;
		case TYPE_STRING_ARRAY: array->val.strarray = NULL; break;
	}

	return array;
}

int phoebe_array_alloc (PHOEBE_array *array, int dimension)
{
	/*
	 * This function allocates storage memory for a array of 'dimension'.
	 *
	 * Return values:
	 *
	 *   ERROR_ARRAY_NOT_INITIALIZED
	 *   ERROR_ARRAY_ALREADY_ALLOCATED
	 *   ERROR_ARRAY_INVALID_DIMENSION
	 *   SUCCESS
	 */

	if (!array)
		return ERROR_ARRAY_NOT_INITIALIZED;

	if (array->dim != 0)
		return ERROR_ARRAY_ALREADY_ALLOCATED;

	if (dimension < 1)
		return ERROR_ARRAY_INVALID_DIMENSION;

	array->dim = dimension;
	switch (array->type) {
		case TYPE_INT_ARRAY:    array->  val.iarray = phoebe_malloc (sizeof (*(array->  val.iarray)) * dimension); break;
		case TYPE_DOUBLE_ARRAY: array->  val.darray = phoebe_malloc (sizeof (*(array->  val.darray)) * dimension); break;
		case TYPE_BOOL_ARRAY:   array->  val.barray = phoebe_malloc (sizeof (*(array->  val.barray)) * dimension); break;
		case TYPE_STRING_ARRAY: array->val.strarray = phoebe_malloc (sizeof (*(array->val.strarray)) * dimension); break;
	}

	return SUCCESS;
}

int phoebe_array_realloc (PHOEBE_array *array, int dimension)
{
	/*
	 * This function reallocates storage memory for an array of 'dimension'.
	 *
	 * Return values:
	 *
	 *   ERROR_ARRAY_NOT_INITIALIZED
	 *   ERROR_ARRAY_INVALID_DIMENSION
	 *   SUCCESS
	 */

	int i;
	int olddim = array->dim;

	if (!array)
		return ERROR_ARRAY_NOT_INITIALIZED;

	if (dimension < 1)
		return ERROR_ARRAY_INVALID_DIMENSION;

	array->dim = dimension;
	switch (array->type) {
		case TYPE_INT_ARRAY:    array->val.iarray   = phoebe_realloc (array->val.iarray,   dimension * sizeof (*(array->val.iarray)));   break;
		case TYPE_BOOL_ARRAY:   array->val.barray   = phoebe_realloc (array->val.barray,   dimension * sizeof (*(array->val.barray)));   break;
		case TYPE_DOUBLE_ARRAY: array->val.darray   = phoebe_realloc (array->val.darray,   dimension * sizeof (*(array->val.darray)));   break;
		case TYPE_STRING_ARRAY:
			array->val.strarray = phoebe_realloc (array->val.strarray, dimension * sizeof (*(array->val.strarray)));
			for (i = olddim; i < dimension; i++)
				array->val.strarray[i] = NULL;
		break;
	}

	return SUCCESS;
}

PHOEBE_array *phoebe_array_new_from_qualifier (char *qualifier)
{
	/*
	 * This function returns a newly allocated array the values of which are
	 * taken from the array parameter represented by its qualifier.
	 *
	 * If an error occured, NULL is returned.
	 *
	 */

	int i;
	PHOEBE_array *array;

	PHOEBE_parameter *par = phoebe_parameter_lookup (qualifier);
	if (!par) return NULL;
	if (par->type != TYPE_INT_ARRAY    &&
		par->type != TYPE_BOOL_ARRAY   &&
		par->type != TYPE_DOUBLE_ARRAY &&
		par->type != TYPE_STRING_ARRAY) return NULL;
	if (par->value.array->dim == 0) return NULL;

	/* Create and allocate a new array of the given type: */
	array = phoebe_array_new (par->type);
	phoebe_array_alloc (array, par->value.array->dim);

	switch (par->type) {
		case TYPE_INT_ARRAY:
			for (i = 0; i < par->value.array->dim; i++)
				array->val.iarray[i] = par->value.array->val.iarray[i];
		break;
		case TYPE_BOOL_ARRAY:
			for (i = 0; i < par->value.array->dim; i++)
				array->val.barray[i] = par->value.array->val.barray[i];
		break;
		case TYPE_DOUBLE_ARRAY:
			for (i = 0; i < par->value.array->dim; i++)
				array->val.darray[i] = par->value.array->val.darray[i];
		break;
		case TYPE_STRING_ARRAY:
			for (i = 0; i < par->value.array->dim; i++)
				array->val.strarray[i] = strdup (par->value.array->val.strarray[i]);
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_array_new_from_qualifier ()!\n");
		break;
	}

	return array;
}

PHOEBE_array *phoebe_array_duplicate (PHOEBE_array *array)
{
	/*
	 * This function makes a duplicate copy of array 'array'.
	 */

	int i;
	PHOEBE_array *new;

	if (!array) return NULL;

	new = phoebe_array_new (array->type);
	phoebe_array_alloc (new, array->dim);

	switch (new->type) {
		case (TYPE_INT_ARRAY):
			for (i = 0; i < new->dim; i++) new->val.iarray[i] = array->val.iarray[i];
		break;
		case (TYPE_BOOL_ARRAY):
			for (i = 0; i < new->dim; i++) new->val.barray[i] = array->val.barray[i];
		break;
		case (TYPE_DOUBLE_ARRAY):
			for (i = 0; i < new->dim; i++) new->val.darray[i] = array->val.darray[i];
		break;
		case (TYPE_STRING_ARRAY):
			for (i = 0; i < new->dim; i++) new->val.strarray[i] = strdup (array->val.strarray[i]);
		break;
	}

	return new;
}

int phoebe_array_free (PHOEBE_array *array)
{
	/* This function frees the storage memory allocated for array 'array'.       */

	if (!array) return SUCCESS;

	switch (array->type) {
		case TYPE_INT_ARRAY:
			if (array->val.iarray)
				free (array->  val.iarray);
		break;
		case TYPE_DOUBLE_ARRAY:
			if (array->val.darray)
				free (array->  val.darray);
		break;
		case TYPE_BOOL_ARRAY:
			if (array->val.barray)
				free (array->val.barray);
		break;
		case TYPE_STRING_ARRAY:
			if (array->val.strarray) {
				int i;
				for (i = 0; i < array->dim; i++)
					free (array->val.strarray[i]);
				free (array->val.strarray);
			}
		break;
	}
	free (array);
	return SUCCESS;
}

/******************************************************************************/

int phoebe_curve_type_get_name (PHOEBE_curve_type ctype, char **name)
{
	/*
	 * This function assigns a curve type (LC, RV) name to the argument 'name'
	 * based on the passed enumerated type. The calling function has to free
	 * the string afterwards.
	 *
	 * Return values:
	 *
	 *   ERROR_EXCEPTION_HANDLER_INVOKED
	 *   SUCCESS
	 */

	switch (ctype) {
		case PHOEBE_CURVE_UNDEFINED: *name = phoebe_strdup ("Undefined"); break;
		case PHOEBE_CURVE_LC:        *name = phoebe_strdup ("LC");        break;
		case PHOEBE_CURVE_RV:        *name = phoebe_strdup ("RV");        break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_curve_type_get_name (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int phoebe_column_type_get_name (PHOEBE_column_type ctype, char **name)
{
	/*
	 * This function assigns a column type name to the argument 'name'
	 * based on the passed enumerated type. The calling function has to free
	 * the string afterwards.
	 *
	 * Return values:
	 *
	 *   ERROR_EXCEPTION_HANDLER_INVOKED
	 *   SUCCESS
	 */

	switch (ctype) {
		case PHOEBE_COLUMN_UNDEFINED:    *name = phoebe_strdup ("Undefined");          break;
		case PHOEBE_COLUMN_HJD:          *name = phoebe_strdup ("HJD");                break;
		case PHOEBE_COLUMN_PHASE:        *name = phoebe_strdup ("Phase");              break;
		case PHOEBE_COLUMN_MAGNITUDE:    *name = phoebe_strdup ("Magnitude");          break;
		case PHOEBE_COLUMN_FLUX:         *name = phoebe_strdup ("Flux");               break;
		case PHOEBE_COLUMN_PRIMARY_RV:   *name = phoebe_strdup ("Primary RV");         break;
		case PHOEBE_COLUMN_SECONDARY_RV: *name = phoebe_strdup ("Secondary RV");       break;
		case PHOEBE_COLUMN_SIGMA:        *name = phoebe_strdup ("Standard deviation"); break;
		case PHOEBE_COLUMN_WEIGHT:       *name = phoebe_strdup ("Standard weight");    break;
		case PHOEBE_COLUMN_INVALID:      *name = phoebe_strdup ("Invalid");            break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_column_type_get_name ():\n");
			phoebe_lib_error ("function called with numeric code %d; please report this!\n", ctype);
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

int phoebe_column_get_type (PHOEBE_column_type *type, const char *string)
{
	/*
	 * This function returns the enumerated type of the column from the
	 * string value found in model parameters.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	     if (strcmp (string, "Time (HJD)"        ) == 0) *type = PHOEBE_COLUMN_HJD;
	else if (strcmp (string, "Phase"             ) == 0) *type = PHOEBE_COLUMN_PHASE;
	else if (strcmp (string, "Magnitude"         ) == 0) *type = PHOEBE_COLUMN_MAGNITUDE;
	else if (strcmp (string, "Flux"              ) == 0) *type = PHOEBE_COLUMN_FLUX;
	else if (strcmp (string, "Primary RV"        ) == 0) *type = PHOEBE_COLUMN_PRIMARY_RV;
	else if (strcmp (string, "Secondary RV"      ) == 0) *type = PHOEBE_COLUMN_SECONDARY_RV;
	else if (strcmp (string, "Standard deviation") == 0) *type = PHOEBE_COLUMN_SIGMA;
	else if (strcmp (string, "Standard weight"   ) == 0) *type = PHOEBE_COLUMN_WEIGHT;
	else if (strcmp (string, "Unavailable"       ) == 0) *type = PHOEBE_COLUMN_UNDEFINED;
	else                                                 *type = PHOEBE_COLUMN_INVALID;

	if (*type == PHOEBE_COLUMN_INVALID)
		return ERROR_COLUMN_INVALID;

	return SUCCESS;
}

PHOEBE_curve *phoebe_curve_new ()
{
	/*
	 * This function initializes a PHOEBE_curve structure for input. It returns
	 * a pointer to this newly initialized structure.
	 */

	/* Allocate memory for the structure itself:                              */
	PHOEBE_curve *curve = phoebe_malloc (sizeof (*curve));

	curve->type   = PHOEBE_CURVE_UNDEFINED;

	curve->itype  = PHOEBE_COLUMN_UNDEFINED;
	curve->dtype  = PHOEBE_COLUMN_UNDEFINED;
	curve->wtype  = PHOEBE_COLUMN_UNDEFINED;

	/* NULLify all structure pointers so that subsequent allocation is clean: */
	curve->indep  = phoebe_vector_new ();
	curve->dep    = phoebe_vector_new ();
	curve->weight = phoebe_vector_new ();

	curve->passband = NULL;
	curve->filename = NULL;
	curve->sigma    = 0.0;

	return curve;
}

PHOEBE_curve *phoebe_curve_new_from_file (char *filename)
{
	/*
	 * This function initializes a new curve and reads the contents from the
	 * passed filename.
	 */

	PHOEBE_curve *out;

	FILE *file;
	int i = 0;

	char line[255];

	char *in;

	int no_of_columns;
	int line_number = 0;

	double x, y, z;

	/* Do some error handling here.                                           */
	if (!filename_exists (filename))          return NULL;
	if (!filename_is_regular_file (filename)) return NULL;
	if (!(file = fopen (filename, "r")))      return NULL;

	out = phoebe_curve_new ();

	out->filename = strdup (filename);

	/* Do the readout:                                                        */
	while (!feof (file)) {
		fgets (line, 255, file);
		line_number++;

		/* If the line is commented or empty, skip it:                        */
		if ( !(in = parse_data_line (line)) ) continue;

		/* Read out the values from the parsed string:                        */
		no_of_columns = sscanf (in, "%lf %lf %lf", &x, &y, &z);
		if (no_of_columns < 2) {
			phoebe_lib_error ("phoebe_curve_new_from_file (): line %d discarded\n in file \"%s\".\n", line_number, filename);
			free (in);
			continue;
		} else {
			phoebe_vector_realloc (out->indep, i+1);
			phoebe_vector_realloc (out->dep,   i+1);
			out->indep->val[i] = x; out->dep->val[i] = y;
			if (no_of_columns == 3) {
				phoebe_vector_realloc (out->weight, i+1);
				out->weight->val[i] = z;
			}
		}

		free (in);
		i++;
	}
	fclose (file);

	return out;
}

PHOEBE_curve *phoebe_curve_new_from_pars (PHOEBE_curve_type ctype, int index)
{
	/*
	 * This function creates a curve from the values stored in parameters of
	 * the index-th curve.
	 */

	char *param;
	PHOEBE_curve *curve;
	PHOEBE_column_type itype, dtype, wtype;
	PHOEBE_passband *passband;
	char *filename;
	double sigma;
	
	int status;

	if (ctype == PHOEBE_CURVE_LC) {
		/***********************/
		/* 1. phoebe_lc_indep: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_indep"), index, &param);
		status = phoebe_column_get_type (&itype, param);
		if (status != SUCCESS)
			return NULL;

		/*********************/
		/* 2. phoebe_lc_dep: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_dep"), index, &param);
		status = phoebe_column_get_type (&dtype, param);
		if (status != SUCCESS)
			return NULL;

		/***************************/
		/* 3. phoebe_lc_indweight: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_indweight"), index, &param);
		status = phoebe_column_get_type (&wtype, param);
		if (status != SUCCESS)
			return NULL;

		/**************************/
		/* 4. phoebe_lc_filename: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filename"), index, &filename);

		/************************/
		/* 5. phoebe_lc_filter: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), index, &param);
		passband = phoebe_passband_lookup (param);
		if (!passband)
			return NULL;

		/***********************/
		/* 6. phoebe_lc_sigma: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_sigma"), index, &sigma);
	}

	if (ctype == PHOEBE_CURVE_RV) {
		/***********************/
		/* 1. phoebe_rv_indep: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indep"), index, &param);
		status = phoebe_column_get_type (&itype, param);
		if (status != SUCCESS)
			return NULL;

		/*********************/
		/* 2. phoebe_rv_dep: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), index, &param);
		status = phoebe_column_get_type (&dtype, param);
		if (status != SUCCESS)
			return NULL;

		/***************************/
		/* 3. phoebe_rv_indweight: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indweight"), index, &param);
		status = phoebe_column_get_type (&wtype, param);
		if (status != SUCCESS)
			return NULL;

		/**************************/
		/* 4. phoebe_lc_filename: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filename"), index, &filename);

		/************************/
		/* 5. phoebe_lc_filter: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), index, &param);
		passband = phoebe_passband_lookup (param);
		if (!passband)
			return NULL;

		/***********************/
		/* 6. phoebe_lc_sigma: */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), index, &sigma);
	}

	curve = phoebe_curve_new_from_file (filename);
	phoebe_curve_set_properties (curve, ctype, filename, passband, itype, dtype, wtype, sigma);
	return curve;
}

PHOEBE_curve *phoebe_curve_duplicate (PHOEBE_curve *curve)
{
	/*
	 * This function copies the contents of the curve 'curve' to a newly
	 * allocated curve structure 'new', which is returned.
	 */

	int i;
	PHOEBE_curve *new;

	if (!curve) {
		phoebe_lib_error ("curve passed for duplication not initialized, aborting.\n");
		return NULL;
	}

	new = phoebe_curve_new ();

	new->type     = curve->type;
	new->passband = curve->passband;
	new->itype    = curve->itype;
	new->dtype    = curve->dtype;
	new->wtype    = curve->wtype;
	new->L1       = curve->L1;
	new->L2       = curve->L2;
	new->SBR1     = curve->SBR1;
	new->SBR2     = curve->SBR2;

	if (curve->filename)
		new->filename = strdup (curve->filename);
	else
		new->filename = NULL;

	new->passband = curve->passband; /* No need to copy this! */
	new->sigma    = curve->sigma;

	phoebe_curve_alloc (new, curve->indep->dim);
	for (i = 0; i < curve->indep->dim; i++) {
		new->indep->val[i]  = curve->indep->val[i];
		new->dep->val[i]    = curve->dep->val[i];
		if (curve->weight->dim != 0)
			new->weight->val[i] = curve->weight->val[i];
	}

	return new;
}

int phoebe_curve_alloc (PHOEBE_curve *curve, int dim)
{
	/*
	 * This function allocates the arrays of the PHOEBE_curve structure.
	 *
	 * Return values:
	 *
	 *   ERROR_CURVE_NOT_INITIALIZED
	 *   SUCCESS
	 */

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;

	phoebe_vector_alloc (curve->indep, dim);
	phoebe_vector_alloc (curve->dep, dim);
	phoebe_vector_alloc (curve->weight, dim);

	return SUCCESS;
}

int phoebe_curve_transform (PHOEBE_curve *curve, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype)
{
	/*
	 * This function transforms the data to itype, dtype, wtype column types.
	 */

	int status;
	char *readout_str;

	phoebe_debug ("entering phoebe_curve_transform ()\n");

	phoebe_column_type_get_name (curve->itype, &readout_str);
	phoebe_debug ("* transforming %s", readout_str);
	phoebe_column_type_get_name (itype, &readout_str);
	phoebe_debug (" to %s\n", readout_str);

	phoebe_column_type_get_name (curve->dtype, &readout_str);
	phoebe_debug ("* transforming %s", readout_str);
	phoebe_column_type_get_name (dtype, &readout_str);
	phoebe_debug (" to %s\n", readout_str);

	phoebe_column_type_get_name (curve->wtype, &readout_str);
	phoebe_debug ("* transforming %s", readout_str);
	phoebe_column_type_get_name (wtype, &readout_str);
	phoebe_debug (" to %s\n", readout_str);

	if (curve->itype == PHOEBE_COLUMN_HJD && itype == PHOEBE_COLUMN_PHASE) {
		double hjd0, period, dpdt, pshift;
		read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);
		status = transform_hjd_to_phase (curve->indep, hjd0, period, dpdt, 0.0);
		if (status != SUCCESS) return status;
		curve->itype = itype;
	}

	if (curve->itype == PHOEBE_COLUMN_PHASE && itype == PHOEBE_COLUMN_HJD) {
		double hjd0, period, dpdt, pshift;
		read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);
		status = transform_phase_to_hjd (curve->indep, hjd0, period, dpdt, 0.0);
		if (status != SUCCESS) return status;
		curve->itype = itype;
	}

	if (curve->dtype == PHOEBE_COLUMN_MAGNITUDE && dtype == PHOEBE_COLUMN_FLUX) {
 		double mnorm;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mnorm"), &mnorm);

		/*
		 * If weights need to be transformed, we need to transform them *after*
		 * we transform magnitudes to fluxes, because the transformation funcion
		 * uses fluxes and not magnitudes.
		 */

 		status = transform_magnitude_to_flux (curve->dep, mnorm);
		if (status != SUCCESS) return status;
		if (curve->wtype == PHOEBE_COLUMN_SIGMA && wtype != PHOEBE_COLUMN_UNDEFINED) {
			status = transform_magnitude_sigma_to_flux_sigma (curve->weight, curve->dep);
			if (status != SUCCESS) return status;
		}
		curve->dtype = dtype;
	}

	if (curve->dtype == PHOEBE_COLUMN_FLUX && dtype == PHOEBE_COLUMN_MAGNITUDE) {
		double mnorm;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mnorm"), &mnorm);

		/*
		 * If weights need to be transformed, we need to transform them *before*
		 * we transform fluxes to magnitudes, because the transformation funcion
		 * uses fluxes and not magnitudes.
		 */

		if (curve->wtype == PHOEBE_COLUMN_SIGMA && wtype != PHOEBE_COLUMN_UNDEFINED) {
			status = transform_flux_sigma_to_magnitude_sigma (curve->weight, curve->dep);
			if (status != SUCCESS) return status;
		}
		status = transform_flux_to_magnitude (curve->dep, mnorm);
		if (status != SUCCESS) return status;
		curve->dtype = dtype;
	}

	if (curve->wtype == PHOEBE_COLUMN_SIGMA && wtype == PHOEBE_COLUMN_WEIGHT) {
		status = transform_sigma_to_weight (curve->weight);
		if (status != SUCCESS) return status;
		curve->wtype = wtype;
	}

	if (curve->wtype == PHOEBE_COLUMN_WEIGHT && wtype == PHOEBE_COLUMN_SIGMA) {
		status = transform_weight_to_sigma (curve->weight);
		if (status != SUCCESS) return status;
		curve->wtype = wtype;
	}

	if (curve->wtype == PHOEBE_COLUMN_UNDEFINED && wtype == PHOEBE_COLUMN_SIGMA) {
		if (curve->weight && curve->weight->dim == 0) {
			phoebe_vector_alloc (curve->weight, curve->dep->dim);
			phoebe_vector_pad (curve->weight, 0.01);
			curve->wtype = wtype;
		}
		else {
			phoebe_lib_error ("weight column contents undefined, ignoring.\n");
		}
	}

	if (curve->wtype == PHOEBE_COLUMN_UNDEFINED && wtype == PHOEBE_COLUMN_WEIGHT) {
		if (curve->weight && curve->weight->dim == 0) {
			phoebe_vector_alloc (curve->weight, curve->dep->dim);
			phoebe_vector_pad (curve->weight, 1.00);
			curve->wtype = wtype;
		}
		else {
			phoebe_lib_error ("weight column contents undefined, ignoring.\n");
		}
	}

	phoebe_debug ("leaving phoebe_curve_transform ()\n");

	return SUCCESS;
}

int phoebe_curve_set_properties (PHOEBE_curve *curve, PHOEBE_curve_type type, char *filename, PHOEBE_passband *passband, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype, double sigma)
{
	/*
	 * This function sets all property fields of the passed curve.
	 *
	 * Return codes:
	 *
	 *   ERROR_CURVE_NOT_INITIALIZED
	 *   SUCCESS
	 */

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;

	curve->type     = type;
	curve->itype    = itype;
	curve->dtype    = dtype;
	curve->wtype    = wtype;
	curve->sigma    = sigma;
	curve->passband = passband;

	if (filename) {
		if (curve->filename)
			free (curve->filename);
		curve->filename = strdup (filename);
	}
	else
		if (curve->filename) {
			free (curve->filename);
			curve->filename = NULL;
		}

	return SUCCESS;
}

int phoebe_curve_free (PHOEBE_curve *curve)
{
	/*
	 * This function traverses through the PHOEBE_curve structure, frees its
	 * contents if they were allocated and frees the structure itself.
	 *
	 * Return values:
	 *
	 *   ERROR_CURVE_NOT_INITIALIZED
	 *   SUCCESS
	 */

	if (!curve) return ERROR_CURVE_NOT_INITIALIZED;

	phoebe_vector_free (curve->indep);
	phoebe_vector_free (curve->dep);
	phoebe_vector_free (curve->weight);

	if (curve->filename)
		free (curve->filename);

	free (curve);

	return SUCCESS;
}

/******************************************************************************/

int phoebe_minimizer_type_get_name (PHOEBE_minimizer_type minimizer, char **name)
{
	/*
	 * This function assigns a common minimizer name to the argument 'name'
	 * based on the passed enumerated type. The calling function has to free
	 * the string afterwards.
	 *
	 * Return values:
	 *
	 *   ERROR_EXCEPTION_HANDLER_INVOKED
	 *   SUCCESS
	 */

	switch (minimizer) {
		case PHOEBE_MINIMIZER_NMS: *name = phoebe_strdup ("NMS"); break;
		case PHOEBE_MINIMIZER_DC:  *name = phoebe_strdup ("DC");  break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_minimizer_type_get_name (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
		break;
	}

	return SUCCESS;
}

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_new ()
{
	/*
	 * This function initializes a PHOEBE_minimizer_feedback structure for
	 * input. It returns a pointer to this newly initialized structure.
	 */

	/* Allocate memory for the structure itself:                              */
	PHOEBE_minimizer_feedback *feedback = phoebe_malloc (sizeof (*feedback));

	/* NULLify all structure pointers so that subsequent allocation is clean: */
	feedback->qualifiers = phoebe_array_new (TYPE_STRING_ARRAY);
	feedback->initvals   = phoebe_vector_new ();
	feedback->newvals    = phoebe_vector_new ();
	feedback->ferrors    = phoebe_vector_new ();
	feedback->chi2s      = phoebe_vector_new ();
	feedback->wchi2s     = phoebe_vector_new ();

	return feedback;
}

int phoebe_minimizer_feedback_alloc (PHOEBE_minimizer_feedback *feedback, int tba, int cno)
{
	/*
	 * This function allocates the arrays of the PHOEBE_minimizer_feedback
	 * structure. There are two independent array dimensions: one is determined
	 * by the number of parameters set for adjustment (tba) and the other is
	 * determined by the number of data curves (cno). The 'tba' number takes
	 * into account passband-dependent parameters as well, i.e. if a passband-
	 * dependent parameter is set for adjustment, the 'tba' number increases
	 * by the number of passbands, not just by 1.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	if (!feedback)
		return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	phoebe_array_alloc  (feedback->qualifiers, tba);
	phoebe_vector_alloc (feedback->initvals,   tba);
	phoebe_vector_alloc (feedback->newvals,    tba);
	phoebe_vector_alloc (feedback->ferrors,    tba);
	phoebe_vector_alloc (feedback->chi2s,      cno);
	phoebe_vector_alloc (feedback->wchi2s,     cno);

	return SUCCESS;
}

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_duplicate (PHOEBE_minimizer_feedback *feedback)
{
	/*
	 * This function copies the contents of the feedback structure 'feedback'
	 * to the newly allocated feedback structure 'new', which is returned.
	 */

	PHOEBE_minimizer_feedback *dup;

	if (!feedback) {
		phoebe_lib_error ("feedback structure not initialized, aborting.\n");
		return NULL;
	}

	dup = phoebe_minimizer_feedback_new ();

	dup->algorithm = feedback->algorithm;
	dup->cputime   = feedback->cputime;
	dup->iters     = feedback->iters;
	dup->cfval     = feedback->cfval;

	/* There is no need to check for existence of feedback fields explicitly, */
	/* the phoebe_*_duplicate functions do that automatically.                */

	dup->qualifiers = phoebe_array_duplicate  (feedback->qualifiers);
	dup->initvals   = phoebe_vector_duplicate (feedback->initvals);
	dup->newvals    = phoebe_vector_duplicate (feedback->newvals);
	dup->ferrors    = phoebe_vector_duplicate (feedback->ferrors);
	dup->chi2s      = phoebe_vector_duplicate (feedback->chi2s);
	dup->wchi2s     = phoebe_vector_duplicate (feedback->wchi2s);

	return dup;
}

int phoebe_minimizer_feedback_free (PHOEBE_minimizer_feedback *feedback)
{
	/*
	 * This function traverses through the PHOEBE_minimizer_feedback structure,
	 * frees its contents if they were allocated and frees the structure itself.
	 */

	if (!feedback) return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	phoebe_array_free  (feedback->qualifiers);
	phoebe_vector_free (feedback->initvals);
	phoebe_vector_free (feedback->newvals);
	phoebe_vector_free (feedback->ferrors);
	phoebe_vector_free (feedback->chi2s);
	phoebe_vector_free (feedback->wchi2s);

	free (feedback);

	return SUCCESS;
}

/******************************************************************************/

char *phoebe_type_get_name (PHOEBE_type type)
{
	switch (type) {
		case TYPE_INT:                return "integer";
		case TYPE_BOOL:               return "boolean";
		case TYPE_DOUBLE:             return "double";
		case TYPE_STRING:             return "string";
		case TYPE_INT_ARRAY:          return "integer array";
		case TYPE_BOOL_ARRAY:         return "boolean array";
		case TYPE_DOUBLE_ARRAY:       return "double array";
		case TYPE_STRING_ARRAY:       return "string array";
		case TYPE_CURVE:              return "curve";
		case TYPE_SPECTRUM:           return "spectrum";
		case TYPE_MINIMIZER_FEEDBACK: return "minimizer feedback";
		case TYPE_ANY:                return "any";
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_type_get_name (), please report this!\n");
			return NULL;
	}
}
