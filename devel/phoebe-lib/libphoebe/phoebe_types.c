#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_global.h"

#include "phoebe_accessories.h"
#include "phoebe_calculations.h"
#include "phoebe_constraints.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

#ifndef min
	#define min(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
	#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

PHOEBE_vector *phoebe_vector_new ()
{
    /**
     * phoebe_vector_new:
     *
     * Initializes a vector for allocation.
     *
     * Returns: A #PHOEBE_vector.
     */

	PHOEBE_vector *vec = phoebe_malloc (sizeof (*vec));

	vec->dim = 0;
	vec->val = NULL;

	return vec;
}

PHOEBE_vector *phoebe_vector_new_from_qualifier (char *qualifier)
{
	/**
	 * phoebe_vector_new_from_qualifier:
	 * @qualifier: The array parameter.
	 *
	 * Allocates a new #PHOEBE_vector and assigns it elements
	 * taken from the array parameter represented by @qualifier.
	 *
	 * Returns: A #PHOEBE_vector, or NULL if an error occured.
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
	/**
	 * phoebe_vector_new_from_column:
	 * @filename: The full path to the file to be read.
	 * @col: The column to be read.
	 *
	 * Reads in the @col-th column from file @filename,
	 * parses it and stores it into the returned vector.
	 *
	 * Returns: A #PHOEBE_vector, or NULL if an error occured.
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

		if (!fgets (line, 254, input)) break;

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

PHOEBE_vector *phoebe_vector_new_from_array (PHOEBE_array *array)
{
	/**
	 * phoebe_vector_new_from_array:
	 * @array: #PHOEBE_array of type #TYPE_INT or #TYPE_DOUBLE
	 *
	 * Converts the array of doubles into #PHOEBE_vector.
	 *
	 * Returns: #PHOEBE_vector, or NULL in case of failure.
	 */

	int i;
	PHOEBE_vector *vec;

	if (!array)
		return NULL;

	if (array->type != TYPE_INT_ARRAY && array->type != TYPE_DOUBLE_ARRAY) {
		phoebe_lib_error ("cannot convert non-numeric arrays to vectors, aborting.\n");
		return NULL;
	}

	if (array->dim == 0)
		return NULL;

	vec = phoebe_vector_new ();
	phoebe_vector_alloc (vec, array->dim);
	for (i = 0; i < array->dim; i++)
		switch (array->type) {
			case TYPE_INT_ARRAY:
				vec->val[i] = (double) array->val.iarray[i];
			break;
			case TYPE_DOUBLE_ARRAY:
				vec->val[i] = (double) array->val.darray[i];
			break;
			default:
				/* Can't really get here. */
			break;
		}

	return vec;
}

PHOEBE_vector *phoebe_vector_new_from_range (int dim, double start, double end)
{
	/**
	 * phoebe_vector_new_from_range:
	 * @dim: vector dimension
	 * @start: first vector element
	 * @end: last vector element
	 *
	 * Creates a vector that starts at @start, ends at @end, and has @dim
	 * equidistant steps.
	 */

	PHOEBE_vector *vec;
	int i;

	if (dim < 1) {
		phoebe_lib_error ("phoebe_vector_new_from_range () called with dim %d, aborting.\n", dim);
		return NULL;
	}

	vec = phoebe_vector_new ();
	phoebe_vector_alloc (vec, dim);

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = start + (double) i*(end-start)/(vec->dim-1);

	return vec;
}

PHOEBE_vector *phoebe_vector_duplicate (PHOEBE_vector *vec)
{
	/**
	 * phoebe_vector_duplicate:
	 * @vec: The #PHOEBE_vector to copy.
	 *
	 * Makes a duplicate copy of #PHOEBE_vector @vec.
	 *
	 * Returns: A #PHOEBE_vector.
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
	/**
	 * phoebe_vector_alloc:
	 * @vec:       The #PHOEBE_vector to store.
	 * @dimension: The size of the new #PHOEBE_vector.
	 *
	 * Allocates storage memory for a #PHOEBE_vector of @dimension.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_ALREADY_ALLOCATED and ERROR_VECTOR_INVALID_DIMENSION.
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
	/**
	 * phoebe_vector_realloc:
	 * @vec:       #PHOEBE_vector to reallocate.
	 * @dimension: new size for @vec.
	 *
	 * Reallocates storage memory for a #PHOEBE_vector of @dimension.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_INVALID_DIMENSION.
	 */

	if (dimension < 0)
		return ERROR_VECTOR_INVALID_DIMENSION;
	if (vec->dim == dimension)
		return SUCCESS;

	vec->dim = dimension;
	vec->val = phoebe_realloc (vec->val, sizeof (*(vec->val)) * dimension);
	if (dimension == 0) vec->val = NULL;

	return SUCCESS;
}

int phoebe_vector_pad (PHOEBE_vector *vec, double value)
{
	/**
	 * phoebe_vector_pad:
	 * @vec:   The #PHOEBE_vector to pad with @value.
	 * @value: The new value for all elements of @vec.
	 *
	 * Pads all components of @vec with @value.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = value;

	return SUCCESS;
}

int phoebe_vector_free (PHOEBE_vector *vec)
{
	/**
	 * phoebe_vector_free:
	 * @vec: The #PHOEBE_vector to free.
	 *
	 * Frees the storage memory allocated for #PHOEBE_vector @vec.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	if (!vec) return SUCCESS;
	if (vec->val) free (vec->val);
	free (vec);
	return SUCCESS;
}

int phoebe_vector_add (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/**
	 * phoebe_vector_add:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Adds #PHOEBE_vector @fac1 to @fac2 and returns the sum #PHOEBE_vector @result.
	 *
	 * Returns: #PHOEBE_error_code
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
	/**
	 * phoebe_vector_subtract:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Subtracts #PHOEBE_vector @fac2 from @fac1 and returns the difference
	 * #PHOEBE_vector @result.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_multiply:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Multiplies #PHOEBE_vector @fac1 with @fac2 and returns the product
	 * #PHOEBE_vector @result.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_divide:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Divides #PHOEBE_vector @fac1 with @fac2 and returns the quotient
	 * #PHOEBE_vector @result.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_raise:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Raises all elements of #PHOEBE_vector @fac1 to the corresponding element
	 * of #PHOEBE_vector @fac2, basically result[i] = fac1[i]^fac2[i].
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_DIMENSIONS_MISMATCH.
	 */

	int i;

	if (fac1->dim != fac2->dim)
		return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	phoebe_vector_alloc (result, fac1->dim);
	for (i = 0; i < fac1->dim; i++)
		result->val[i] = pow (fac1->val[i], fac2->val[i]);

	return SUCCESS;
}

int phoebe_vector_offset (PHOEBE_vector *vec, double offset)
{
	/**
	 * phoebe_vector_offset:
	 * @vec:    #PHOEBE_vector to be offset
	 * @offset: offset value.
	 *
	 * Adds @offset to all elements of #PHOEBE_vector @vec.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	if (!vec)
		return ERROR_VECTOR_NOT_INITIALIZED;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] += offset;

	return SUCCESS;
}

int phoebe_vector_sum (PHOEBE_vector *vec, double *sum)
{
	/**
	 * phoebe_vector_sum:
	 * @vec: #PHOEBE_vector for which a sum is computed
	 * @median: placeholder for the sum value
	 *
	 * Computes a sum of all vector elements.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	if (!vec) return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec->dim < 1) return ERROR_VECTOR_INVALID_DIMENSION;
	
	*sum = 0.0;
	for (i = 0; i < vec->dim; i++)
		*sum += vec->val[i];

	return SUCCESS;
}

int phoebe_vector_mean (PHOEBE_vector *vec, double *mean)
{
	/**
	 * phoebe_vector_mean:
	 * @vec: #PHOEBE_vector for which a mean is computed
	 * @median: placeholder for the mean value
	 *
	 * Computes the mean (arithmetic average) of all vector elements.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status;
	double sum;

	/* We'll let phoebe_vector_sum do the error handling for us: */
	status = phoebe_vector_sum (vec, &sum);
	if (status != SUCCESS)
		return status;

	*mean = sum / vec->dim;
	return SUCCESS;
}

int phoebe_vector_median (PHOEBE_vector *vec, double *median)
{
	/**
	 * phoebe_vector_median:
	 * @vec: #PHOEBE_vector for which a median is computed
	 * @median: placeholder for the median value
	 *
	 * Sorts the vector elements and returns a median value. The passed
	 * vector @vec is not changed.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	PHOEBE_vector *copy;

	if (!vec) return ERROR_VECTOR_NOT_INITIALIZED;
	if (vec->dim < 1) return ERROR_VECTOR_IS_EMPTY;

	/* We have to sort the array, that is why we need a duplicate: */
	copy = phoebe_vector_duplicate (vec);
	qsort (copy->val, copy->dim, sizeof (*(copy->val)), diff);

	if (copy->dim % 2 == 0) *median = copy->val[copy->dim/2];
	else *median = 0.5*(copy->val[copy->dim/2]+copy->val[copy->dim/2+1]);

	phoebe_vector_free (copy);
	return SUCCESS;
}

int phoebe_vector_standard_deviation (PHOEBE_vector *vec, double *sigma)
{
	/**
	 * phoebe_vector_standard_deviation:
	 * @vec: #PHOEBE_vector for which standard deviation is computed
	 * @sigma: placeholder for standard deviation value
	 *
	 * Computes standard deviation (@sigma) of vector elements.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	int status;
	double mean;


	/* We'll let phoebe_vector_mean do the error handling for us: */
	status = phoebe_vector_mean (vec, &mean);
	if (status != SUCCESS) return status;

	*sigma = 0.0;
	for (i = 0; i < vec->dim; i++)
		*sigma += (vec->val[i] - mean) * (vec->val[i] - mean);
	*sigma = sqrt (*sigma / (vec->dim - 1));

	return SUCCESS;
}

int phoebe_vector_multiply_by (PHOEBE_vector *vec, double factor)
{
	/**
	 * phoebe_vector_multiply_by:
	 * @vec:    #PHOEBE_vector to be modified
	 * @factor: scaling factor
	 *
	 * Multiplies all elements of the #PHOEBE_vector @vec with the scalar
	 * value @factor.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	if (!vec)
		return ERROR_VECTOR_NOT_INITIALIZED;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] *= factor;

	return SUCCESS;
}

int phoebe_vector_dot_product (double *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2)
{
	/**
	 * phoebe_vector_dot_product:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Returns the scalar (dot) product of two #PHOEBE_vectors.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_vec_product:
	 * @result: The placeholder for a return value.
	 * @fac1:   Vector 1.
	 * @fac2:   Vector 2.
	 *
	 * Returns the vector product of the two #PHOEBE_vectors. Both @fac1 and @fac2
	 * need to have exactly 3 elements.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_DIMENSION_NOT_THREE, and ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_submit:
	 * @result: The placeholder for a return value.
	 * @vec:    The #PHOEBE_vector to submit to @func.
	 * @func:   A function.
	 *
	 * Calculates the functional value of @func for each element of @vec.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;

	phoebe_vector_alloc (result, vec->dim);
	for (i = 0; i < vec->dim; i++)
		result->val[i] = func (vec->val[i]);

	return SUCCESS;
}

int phoebe_vector_norm (double *result, PHOEBE_vector *vec)
{
	/**
	 * phoebe_vector_norm:
	 * @result: The placeholder for a return value.
	 * @vec:    The #PHOEBE_vector to norm.
	 *
	 * Returns the norm of #PHOEBE_vector @vec.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
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
	/**
	 * phoebe_vector_dim:
	 * @result: The placeholder for a return value.
	 * @vec:    The #PHOEBE_vector to examine.
	 *
	 * Returns the dimension of @vec.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	*result = vec->dim;
	return SUCCESS;
}

int phoebe_vector_randomize (PHOEBE_vector *result, double limit)
{
	/**
	 * phoebe_vector_randomize:
	 * @result: The placeholder for a return value.
	 * @limit:  The upper limit for generated numbers.
	 *
	 * Fills all vector elements with random numbers from the
	 * interval [0, @limit]. @limit may also be negative, then it's [@limit, 0].
	 * The vector must be allocated prior to calling this function.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_VECTOR_IS_EMPTY.
	 */

	int i;

	if (result->dim == 0) return ERROR_VECTOR_IS_EMPTY;

	for (i = 0; i < result->dim; i++)
		result->val[i] = limit * rand() / RAND_MAX;

	return SUCCESS;
}

int phoebe_vector_min_max (PHOEBE_vector *vec, double *min, double *max)
{
	/**
	 * phoebe_vector_min_max:
	 * @vec: The #PHOEBE_vector to scan for minimum and maximum.
	 * @min: The placeholder for the minimal value in @vec.
	 * @max: The placeholder for the maximal value in @vec.
	 *
	 * Scans through the dataset of and assigns the minimal and the
	 * maximal value encountered to @min and @max variables.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;
	*min = *max = vec->val[0];

	for (i = 1; i < vec->dim; i++) {
		if (*min > vec->val[i]) *min = vec->val[i];
		if (*max < vec->val[i]) *max = vec->val[i];
	}

	return SUCCESS;
}

int phoebe_vector_min_index (PHOEBE_vector *vec, int *index)
{
	/**
	 * phoebe_vector_min_index:
	 * @vec:   The #PHOEBE_vector to scan for minimum.
	 * @index: The placeholder for the index of the minimal value in @vec.
	 *
	 * Scans through the #PHOEBE_vector @vec and assigns the index of
	 * the lowest value in @vec to @index.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;
	double vmin = vec->val[0];

	for (i = 1; i < vec->dim; i++)
		if (vmin > vec->val[i]) {
			vmin = vec->val[i];
			*index = i;
		}

	return SUCCESS;
}

int phoebe_vector_max_index (PHOEBE_vector *vec, int *index)
{
	/**
	 * phoebe_vector_max_index:
	 * @vec:   The #PHOEBE_vector to scan for maximum.
	 * @index: The placeholder for the index of the maximal value in @vec.
	 *
	 * Scans through the #PHOEBE_vector @vec and assigns the index of
	 * the highest value in @vec to @index.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;
	double vmax = vec->val[0];

	for (i = 1; i < vec->dim; i++)
		if (vmax < vec->val[i]) {
			vmax = vec->val[i];
			*index = i;
		}

	return SUCCESS;
}

int phoebe_vector_rescale (PHOEBE_vector *vec, double ll, double ul)
{
	/**
	 * phoebe_vector_rescale:
	 * @vec: The #PHOEBE_vector to rescale.
	 * @ll:  The lower limit for rescaling.
	 * @ul:  The upper limit for rescaling.
	 *
	 * Linearly rescales the values of elements in the vector @vec to the
	 * [@ll, @ul] interval.
	 *
	 * Returns: #PHOEBE_error_code.
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
	/**
	 * phoebe_vector_compare:
	 * @vec1:   Vector 1.
	 * @vec2:   Vector 2.
	 *
	 * Compares two passed #PHOEBE_vectors. It returns TRUE if all vector
	 * elements are the same; it returns FALSE otherwise. The comparison is
	 * done by comparing the difference of both elements to #PHOEBE_NUMERICAL_ACCURACY.
	 *
	 * Returns: A boolean indicating whether the two vectors have identical elements.
	 */

	int i;
	if (vec1->dim != vec2->dim) return FALSE;

	for (i = 0; i < vec1->dim; i++)
		if (fabs (vec1->val[i] - vec2->val[i]) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;

	return TRUE;
}

int phoebe_vector_less_than (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2)
{
	/**
	 * phoebe_vector_less_than:
	 * @result: The placeholder for the result.
	 * @vec1:   Vector 1.
	 * @vec2:   Vector 2.
	 *
	 * Tests whether *all* vector elements of @vec1 are less
	 * than their respective counterparts of @vec2. If so, TRUE is returned,
	 * otherwise FALSE is returned.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors:
     *   ERROR_VECTOR_NOT_INITIALIZED and
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_leq_than:
	 * @result: The placeholder for the result.
	 * @vec1:   Vector 1.
	 * @vec2:   Vector 2.
	 *
	 * Tests whether *all* vector elements of @vec1 are less or
	 * equal to their respective counterparts of @vec2. If so, TRUE is returned,
	 * otherwise FALSE is returned.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors:
     *   ERROR_VECTOR_NOT_INITIALIZED and
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_greater_than:
	 * @result: The placeholder for the result.
	 * @vec1:   Vector 1.
	 * @vec2:   Vector 2.
	 *
	 * Tests whether *all* vector elements of @vec1 are greater
	 * than their respective counterparts of @vec2. If so, TRUE is returned,
	 * otherwise FALSE is returned.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors:
     *   ERROR_VECTOR_NOT_INITIALIZED and
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_geq_than:
	 * @result: The placeholder for the result.
	 * @vec1:   Vector 1.
	 * @vec2:   Vector 2.
	 *
	 * Tests whether *all* vector elements of @vec1 are greater
	 * or equal to their respective counterparts of @vec2. If so, TRUE is
	 * returned, otherwise FALSE is returned.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors:
     *   ERROR_VECTOR_NOT_INITIALIZED and
	 *   ERROR_VECTOR_DIMENSIONS_MISMATCH.
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
	/**
	 * phoebe_vector_append_element:
	 * @vec: The vector to extend.
	 * @val: The value to append.
	 *
	 * Appends an element with value @val to the #PHOEBE_vector @vec.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	phoebe_vector_realloc (vec, vec->dim+1);
	vec->val[vec->dim-1] = val;
	return SUCCESS;
}

int phoebe_vector_remove_element (PHOEBE_vector *vec, int index)
{
	/**
	 * phoebe_vector_remove_element:
	 * @vec:   The vector to shorten.
	 * @index: The index of the element to be removed.
	 *
	 * Removes the @index-th element from #PHOEBE_vector @vec.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_INDEX_OUT_OF_RANGE.
	 */

	int i;

	if (index >= vec->dim) return ERROR_INDEX_OUT_OF_RANGE;

	for (i = index; i < vec->dim; i++)
		vec->val[i] = vec->val[i+1];
	phoebe_vector_realloc (vec, vec->dim-1);
	return SUCCESS;
}

/* **************************************************************************** */

PHOEBE_matrix *phoebe_matrix_new ()
{
    /**
     * phoebe_matrix_new:
     *
     * Initializes an unallocated #PHOEBE_matrix.
     *
     * Returns: An empty #PHOEBE_matrix.
     */

	PHOEBE_matrix *matrix = phoebe_malloc (sizeof (*matrix));

	matrix->rows = 0;
	matrix->cols = 0;
	matrix->val = NULL;

	return matrix;
}

int phoebe_matrix_alloc (PHOEBE_matrix *matrix, int cols, int rows)
{
	/**
	 * phoebe_matrix_alloc:
	 * @matrix: A #PHOEBE_matrix to allocate.
	 * @cols:   The number of columns for @matrix.
	 * @rows:   The number of rows for @matrix.
	 *
	 * Allocates storage memory for @matrix. The
	 * subscripts are such that the elements of @matrix should be accessed
	 * by @matrix[@row][@col].
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 * Possible errors: ERROR_MATRIX_ALREADY_ALLOCATED and ERROR_MATRIX_INVALID_DIMENSION.
	 */

	int i;

	if (matrix->rows != 0 || matrix->cols != 0)
		return ERROR_MATRIX_ALREADY_ALLOCATED;

	if (rows < 1 || cols < 1)
		return ERROR_MATRIX_INVALID_DIMENSION;

	matrix->rows = rows;
	matrix->cols = cols;
	matrix->val = phoebe_malloc (rows * sizeof (*(matrix->val)));
	for (i = 0; i < rows; i++)
		matrix->val[i] = phoebe_malloc (cols * sizeof (**(matrix->val)));

	return SUCCESS;
}

PHOEBE_matrix *phoebe_matrix_duplicate (PHOEBE_matrix *matrix)
{
	/**
	 * phoebe_matrix_duplicate:
	 * @matrix: The #PHOEBE_matrix to copy.
	 *
	 * Makes a duplicate copy of #PHOEBE_matrix @matrix.
	 *
	 * Returns: A #PHOEBE_matrix.
	 */

	int i, j;
	PHOEBE_matrix *copy;

	if (!matrix) return NULL;

	copy = phoebe_matrix_new ();
	phoebe_matrix_alloc (copy, matrix->cols, matrix->rows);
	for (i = 0; i < matrix->cols; i++)
		for (j = 0; j < matrix->rows; j++)
			copy->val[i][j] = matrix->val[i][j];

	return copy;
}

int phoebe_matrix_set_row (PHOEBE_matrix *matrix, PHOEBE_vector *vec, int row)
{
    /**
     * phoebe_matrix_set_row:
     * @matrix: A #PHOEBE_matrix to modify.
     * @vec:    The #PHOEBE_vector to be placed in @matrix.
     * @row:    The row of @matrix to be replaced with @vec.
     *
     * Sets the elements of @row of @matrix to the values of elements in @vec.
     *
     * Returns: A #PHOEBE_error_code indicating the success of the operation.
     */

	int j;

	if (!matrix) printf ("matrix problems!\n");
	if (!vec) printf ("vector problems!\n");
	if (row < 0 || row >= matrix->rows) printf ("row problems: %d!\n", row);

	for (j = 0; j < vec->dim; j++)
		matrix->val[row][j] = vec->val[j];

	return SUCCESS;
}

int phoebe_matrix_get_row (PHOEBE_vector *vec, PHOEBE_matrix *matrix, int row)
{
	/**
	 * phoebe_matrix_get_row:
	 * @vec:    The #PHOEBE_vector to store the values from @row.
	 * @matrix: The #PHOEBE_matrix to read @row from.
	 * @row:    The row of @matrix to be copied to @vec.
	 *
	 * Copies the contents of @row of @matrix to @vec, assuming that @vec is already
	 * allocated.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;

	if (!matrix) {};
	if (row < 0 || row >= matrix->rows) {};

	for (i = 0; i < matrix->cols; i++)
		vec->val[i] = matrix->val[row][i];

	return SUCCESS;
}

int phoebe_matrix_free  (PHOEBE_matrix *matrix)
{
	/**
	 * phoebe_matrix_free:
	 * @matrix: The #PHOEBE_matrix to be freed.
	 *
	 * Frees the storage memory allocated for @matrix.
	 *
	 * Returns: A #PHOEBE_error_code indicating the success of the operation.
	 */

	int i;

	if (!matrix) return SUCCESS;

	for (i = 0; i < matrix->rows; i++)
		free (matrix->val[i]);
	free (matrix->val);

	free (matrix);
	return SUCCESS;
}

/* **************************************************************************** */

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
		if (!fgets (line, 254, input)) break;
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
	if (bins < 0)
		return ERROR_HIST_INVALID_DIMENSION;
	if (hist->bins == bins)
		return SUCCESS;

	hist->bins  = bins;
	hist->range = phoebe_realloc (hist->range, (bins+1) * sizeof (*hist->range));
	hist->val   = phoebe_realloc (hist->val,    bins    * sizeof (*hist->val));
	if (bins == 0) hist->val = NULL;
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
	/**
	 * phoebe_hist_set_values:
	 * @hist: histogram to be modified
	 * @values: a vector of values to be copied to the histogram
	 * 
	 * Sets bin values to the passed vector elements. The histogram @hist
	 * must be initialized and allocated, and the dimension of @values must
	 * match the number of bins.
	 *
	 * Returns: #PHOEBE_error_code.
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

bool phoebe_hist_compare (PHOEBE_hist *hist1, PHOEBE_hist *hist2)
{
	/**
	 * phoebe_hist_compare:
	 * @hist1: histogram 1 to be compared
	 * @hist2: histogram 2 to be compared
	 *
	 * Compares the contents of histograms @hist1 and @hist2. All ranges and
	 * values are compared by evaluating the absolute difference between the
	 * elements and comparing that to #PHOEBE_NUMERICAL_ACCURACY.
	 *
	 * Returns: #TRUE if the histograms are the same, #FALSE otherwise.
	 */

	int i;

	if (hist1->bins != hist2->bins)
		return FALSE;

	for (i = 0; i <= hist1->bins; i++) {
		if (fabs (hist1->range[i]-hist2->range[i]) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;
		if (i != hist1->bins && fabs (hist1->val[i]-hist2->val[i]) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;
	}

	return TRUE;
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

int phoebe_hist_resample (PHOEBE_hist *out, PHOEBE_hist *in, PHOEBE_hist_rebin_type type)
{
	int i = 0, j = 0;
	PHOEBE_vector *c_in, *c_out;

	phoebe_hist_pad (out, 0.0);
	
	switch (type) {
		case PHOEBE_HIST_CONSERVE_DENSITY:
			while (i != in->bins && j != out->bins) {
				out->val[j] += (min(in->range[i+1], out->range[j+1])-max(in->range[i], out->range[j]))/(in->range[i+1]-in->range[i])*in->val[i];
				if (out->range[j+1] > in->range[i+1]) i++; else j++;
			}
		break;
		case PHOEBE_HIST_CONSERVE_VALUES:
			c_in = phoebe_vector_new ();
			phoebe_vector_alloc (c_in, in->bins);
			phoebe_hist_get_bin_centers (in, c_in);
			
			c_out = phoebe_vector_new ();
			phoebe_vector_alloc (c_out, out->bins);
			phoebe_hist_get_bin_centers (out, c_out);

			while (j < c_out->dim && c_out->val[j] < c_in->val[0]) j++;
			
			for ( ; j < c_out->dim; j++) {
				while (i < c_in->dim && c_out->val[j] > c_in->val[i]) i++; i--;
				if (i == c_in->dim-1) break;
				out->val[j] = in->val[i] + (c_out->val[j]-c_in->val[i])/(c_in->val[i+1]-c_in->val[i])*(in->val[i+1]-in->val[i]);
			}

			phoebe_vector_free (c_in);
			phoebe_vector_free (c_out);
		break;
	}
	
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
	/**
	 * phoebe_hist_shift:
	 * @hist: histogram to be shifted
	 * @shift: pixel-space shift
	 *
	 * Shifts the contents of the histogram @hist in pixel-space by the passed
	 * @shift. If @shift is positive, the contents are shifted to the right;
	 * if it is negative, they are shifted to the left. The bin structure is
	 * retained, the bins outside the shifted range are padded with 0.
	 *
	 * Returns: #PHOEBE_error_code.
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

/* **************************************************************************** */

PHOEBE_array *phoebe_array_new (PHOEBE_type type)
{
	/*
	 * This function initializes a new array of PHOEBE_type type. It does *not*
	 * allocate any space, use phoebe_array_alloc () function for that.
	 */

	PHOEBE_array *array;

	if (type != TYPE_INT_ARRAY    && type != TYPE_BOOL_ARRAY &&
		type != TYPE_DOUBLE_ARRAY && type != TYPE_STRING_ARRAY) {
		phoebe_lib_error ("phoebe_array_new (): passed type %s not an array, aborting.\n", phoebe_type_get_name (type));
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
		default:
			/* for suppressing compiler warning only. */
		break;
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
		default:
			/* for suppressing compiler warning only */
		break;
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

	if (dimension < 0)
		return ERROR_ARRAY_INVALID_DIMENSION;

	if (olddim == dimension)
		return SUCCESS;

	array->dim = dimension;
	switch (array->type) {
		case TYPE_INT_ARRAY:
			array->val.iarray = phoebe_realloc (array->val.iarray, dimension * sizeof (*(array->val.iarray)));
			if (dimension == 0) array->val.iarray = NULL;
		break;
		case TYPE_BOOL_ARRAY:
			array->val.barray = phoebe_realloc (array->val.barray, dimension * sizeof (*(array->val.barray)));
			if (dimension == 0) array->val.barray = NULL;
		break;
		case TYPE_DOUBLE_ARRAY:
			array->val.darray = phoebe_realloc (array->val.darray, dimension * sizeof (*(array->val.darray)));
			if (dimension == 0) array->val.darray = NULL;
		break;
		case TYPE_STRING_ARRAY:
			array->val.strarray = phoebe_realloc (array->val.strarray, dimension * sizeof (*(array->val.strarray)));
			for (i = olddim; i < dimension; i++)
				array->val.strarray[i] = NULL;
			if (dimension == 0) array->val.strarray = NULL;
		break;
		default:
			/* for suppressing compiler warning only */
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

PHOEBE_array *phoebe_array_new_from_column (char *filename, int col)
{
	/**
	 * phoebe_array_new_from_column:
	 * @filename: absolute path to the file to be read.
	 * @col: column to be read.
	 *
	 * Reads in the @col-th column from file @filename, parses it and stores
	 * it into the returned array. The first element determines the type of
	 * the array.
	 *
	 * Returns: #PHOEBE_array, or NULL if an error occured.
	 */

	FILE *input;
	PHOEBE_array *array = NULL;
	int i, linecount = 1;
	char *line = NULL, *delimeter;

	input = fopen (filename, "r");
	if (!input) return NULL;

	while (!feof (input)) {
		line = phoebe_readline (input);
		if (feof (input)) break;

		delimeter = line;

		/* Remove the trailing newline (unix or dos): */
		line[strlen(line)-1] = '\0';
		if (strchr (line, 13) != NULL) (strchr (line, 13))[0] = '\0';

		/* Remove comments (if any): */
		if (strchr (line, '#') != NULL) (strchr (line, '#'))[0] = '\0';

		/* Remove any leading whitespaces and empty lines: */
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

		if (delimeter[0] != '\0') {
			if (!array) {
				char test_string[255];
				int test_int;

				/* This is the first row; parse the column type: */
				sscanf (delimeter, "%s", test_string);

				if (sscanf (test_string, "%d", &test_int) == 1)
					/* It starts with a number; does it contain '.' or 'e': */
					if (strchr (test_string, '.') || strchr (test_string, 'e') || strchr (test_string, 'E'))
						array = phoebe_array_new (TYPE_DOUBLE_ARRAY);
					else
						array = phoebe_array_new (TYPE_INT_ARRAY);
				else
					if (strcmp (test_string, "true")  == 0 || strcmp (test_string, "TRUE") == 0 ||
						strcmp (test_string, "false") == 0 || strcmp (test_string, "FALSE") == 0)
						array = phoebe_array_new (TYPE_BOOL_ARRAY);
					else
						array = phoebe_array_new (TYPE_STRING_ARRAY);
			}

			switch (array->type) {
				case TYPE_INT_ARRAY: {
					int val;
					if (sscanf (delimeter, "%d", &val) == 1) {
						phoebe_array_realloc (array, array->dim+1);
						array->val.iarray[array->dim-1] = val;
					}
					else
						phoebe_lib_warning ("line %d in file %s discarded.\n", linecount, filename);
				}
				break;
				case TYPE_BOOL_ARRAY: {
					char val[255];
					if (sscanf (delimeter, "%s", val) == 1) {
						phoebe_array_realloc (array, array->dim+1);
						if (strcmp (val, "true") == 0 || strcmp (val, "TRUE") == 0)
							array->val.barray[array->dim-1] = TRUE;
						else
							array->val.barray[array->dim-1] = FALSE;
					}
					else
						phoebe_lib_warning ("line %d in file %s discarded.\n", linecount, filename);
				}
				break;
				case TYPE_DOUBLE_ARRAY: {
					double val;
					if (sscanf (delimeter, "%lf", &val) == 1) {
						phoebe_array_realloc (array, array->dim+1);
						array->val.darray[array->dim-1] = val;
					}
					else
						phoebe_lib_warning ("line %d in file %s discarded.\n", linecount, filename);
				}
				break;
				case TYPE_STRING_ARRAY: {
					char val[255];
					if (sscanf (delimeter, "%s", val) == 1) {
						phoebe_array_realloc (array, array->dim+1);
						array->val.strarray[array->dim-1] = strdup (val);
					}
					else
						phoebe_lib_warning ("line %d in file %s discarded.\n", linecount, filename);
				}
				break;
				default:
					/* Can't really happen. */
				break;
			}
		}

		free (line);
		linecount++;
	}
	fclose (input);

	return array;
}

PHOEBE_array *phoebe_array_duplicate (PHOEBE_array *array)
{
	/*
	 * This function makes a duplicate copy of array 'array'.
	 */

	int i;
	PHOEBE_array *copy;

	if (!array) return NULL;

	copy = phoebe_array_new (array->type);
	phoebe_array_alloc (copy, array->dim);

	switch (copy->type) {
		case (TYPE_INT_ARRAY):
			for (i = 0; i < copy->dim; i++)
				copy->val.iarray[i] = array->val.iarray[i];
		break;
		case (TYPE_BOOL_ARRAY):
			for (i = 0; i < copy->dim; i++)
				copy->val.barray[i] = array->val.barray[i];
		break;
		case (TYPE_DOUBLE_ARRAY):
			for (i = 0; i < copy->dim; i++)
				copy->val.darray[i] = array->val.darray[i];
		break;
		case (TYPE_STRING_ARRAY):
			for (i = 0; i < copy->dim; i++)
				copy->val.strarray[i] = strdup (array->val.strarray[i]);
		break;
		default:
			/* for suppressing compiler warning only */
		break;
	}

	return copy;
}

bool phoebe_array_compare (PHOEBE_array *array1, PHOEBE_array *array2)
{
	/**
	 * phoebe_array_compare:
	 * @array1:   Array 1.
	 * @array2:   Array 2.
	 *
	 * Compares two passed #PHOEBE_arrays. It returns TRUE if all array
	 * elements are the same; it returns FALSE otherwise. In cases of double
	 * precision numbers the comparison is done by comparing the difference of
	 * both elements to #PHOEBE_NUMERICAL_ACCURACY.
	 *
	 * Returns: Boolean indicating whether the two arrays have identical elements.
	 */

	int i;

	if (array1->type != array2->type) return FALSE;
	if (array1->dim  != array2->dim) return FALSE;

	switch (array1->type) {
		case TYPE_INT_ARRAY:
			for (i = 0; i < array1->dim; i++)
				if (array1->val.iarray[i] != array2->val.iarray[i]) return FALSE;
		break;
		case TYPE_BOOL_ARRAY:
			for (i = 0; i < array1->dim; i++)
				if (array1->val.iarray[i] != array2->val.iarray[i]) return FALSE;
		break;
		case TYPE_DOUBLE_ARRAY:
			for (i = 0; i < array1->dim; i++)
				if (fabs (array1->val.darray[i] - array2->val.darray[i]) > PHOEBE_NUMERICAL_ACCURACY) return FALSE;
		break;
		case TYPE_STRING_ARRAY:
			for (i = 0; i < array1->dim; i++)
				if (strcmp (array1->val.strarray[i], array2->val.strarray[i]) != 0) return FALSE;
		break;
		default:
			/* can't happen, fall through */
		break;
	}

	return TRUE;
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
		default:
			/* for suppressing compiler warning only */
		break;
	}
	free (array);
	return SUCCESS;
}

/* **************************************************************************** */

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
	/**
	 * phoebe_column_get_type:
	 * @type: placeholder for the enumerated column type
	 * @string: string representation of the column type
	 * 
	 * Parses the passed @string and converts it to the enumerated type of
	 * the column (#PHOEBE_column_type).
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!string) {
		*type = PHOEBE_COLUMN_INVALID;
		return ERROR_COLUMN_INVALID;
	}

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
	curve->flag   = phoebe_array_new (TYPE_INT_ARRAY);

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

	/* Do some error handling here. */
	if (!phoebe_filename_exists (filename))          return NULL;
	if (!phoebe_filename_is_regular_file (filename)) return NULL;
	if (!(file = fopen (filename, "r")))             return NULL;

	out = phoebe_curve_new ();

	out->filename = strdup (filename);

	/* Do the readout: */
	while (!feof (file)) {
		if (!fgets (line, 255, file)) break;
		line_number++;

		/* If the line is commented or empty, skip it: */
		if ( !(in = phoebe_clean_data_line (line)) ) continue;
		
		/* Read out the values from the parsed string: */
		no_of_columns = sscanf (in, "%lf %lf %lf", &x, &y, &z);
		if (no_of_columns < 2) {
			/* Check if the data point is deleted (if it starts with '!'): */
			no_of_columns = sscanf (in, "!%lf %lf %lf", &x, &y, &z);
			if (no_of_columns < 2) {
				phoebe_lib_error ("phoebe_curve_new_from_file (): line %d discarded\n in file \"%s\".\n", line_number, filename);
				free (in);
				continue;
			}
		}
		
		phoebe_curve_realloc (out, i+1);

		out->indep->val[i] = x;
		out->dep->val[i] = y;
		if (no_of_columns == 3)
			out->weight->val[i] = z;
		else
			out->weight->val[i] = 1.0;
		if (*in == '!')
			out->flag->val.iarray[i] = PHOEBE_DATA_DELETED;
		else
			out->flag->val.iarray[i] = PHOEBE_DATA_REGULAR;
		
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
	PHOEBE_passband *passband = NULL;
	char *filename;
	double sigma;
	bool bin_data;

	int status;

	if (ctype == PHOEBE_CURVE_LC) {
		/* ********************* */
		/* 1. phoebe_lc_indep:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_indep"), index, &param);
		status = phoebe_column_get_type (&itype, param);
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ******************* */
		/* 2. phoebe_lc_dep:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_dep"), index, &param);
		status = phoebe_column_get_type (&dtype, param);
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ************************* */
		/* 3. phoebe_lc_indweight:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_indweight"), index, &param);
		status = phoebe_column_get_type (&wtype, param);
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ************************ */
		/* 4. phoebe_lc_filename:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filename"), index, &filename);

		/* ********************** */
		/* 5. phoebe_lc_filter:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), index, &param);
		passband = phoebe_passband_lookup (param);
		if (!passband) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ********************* */
		/* 6. phoebe_lc_sigma:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_sigma"), index, &sigma);
	}

	if (ctype == PHOEBE_CURVE_RV) {
		/* ********************* */
		/* 1. phoebe_rv_indep:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indep"), index, &param);
		status = phoebe_column_get_type (&itype, param);
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ******************* */
		/* 2. phoebe_rv_dep:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), index, &param);
		status = phoebe_column_get_type (&dtype, param);
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ************************* */
		/* 3. phoebe_rv_indweight:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indweight"), index, &param);
		status = phoebe_column_get_type (&wtype, param);
		if (status != SUCCESS) {
			phoebe_lib_error ("%s", phoebe_error (status));
			return NULL;
		}

		/* ************************ */
		/* 4. phoebe_rv_filename:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filename"), index, &filename);

		/* ********************** */
		/* 5. phoebe_rv_filter:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), index, &param);
		passband = phoebe_passband_lookup (param);
		if (!passband) {
			phoebe_lib_error ("passband lookup failed, please review your settings.\n");
			return NULL;
		}

		/* ********************* */
		/* 6. phoebe_rv_sigma:   */
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), index, &sigma);
	}

	curve = phoebe_curve_new_from_file (filename);
	if (!curve) {
		phoebe_lib_error ("filename %s cannot be opened, aborting.\n", filename);
		return NULL;
	}

	phoebe_curve_set_properties (curve, ctype, filename, passband, itype, dtype, wtype, sigma);

	/* Bin data if requested: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_bins_switch"), &bin_data);
	if (ctype == PHOEBE_CURVE_LC && bin_data) {
		int bins;
		PHOEBE_curve *binned;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_bins"), &bins);
		binned = phoebe_bin_data (curve, bins);
		phoebe_curve_free (curve);
		return binned;
	}
	
	return curve;
}

PHOEBE_curve *phoebe_curve_duplicate (PHOEBE_curve *curve)
{
	/*
	 * This function copies the contents of the curve 'curve' to a newly
	 * allocated curve structure 'new', which is returned.
	 */

	int i;
	PHOEBE_curve *dup;

	if (!curve) {
		phoebe_lib_error ("curve passed for duplication not initialized, aborting.\n");
		return NULL;
	}

	dup = phoebe_curve_new ();

	dup->type     = curve->type;
	dup->passband = curve->passband;
	dup->itype    = curve->itype;
	dup->dtype    = curve->dtype;
	dup->wtype    = curve->wtype;

	if (curve->filename)
		dup->filename = strdup (curve->filename);
	else
		dup->filename = NULL;

	dup->passband = curve->passband; /* No need to copy this! */
	dup->sigma    = curve->sigma;

	phoebe_curve_alloc (dup, curve->indep->dim);
	for (i = 0; i < curve->indep->dim; i++) {
		dup->indep->val[i]       = curve->indep->val[i];
		dup->dep->val[i]         = curve->dep->val[i];
		dup->flag->val.iarray[i] = curve->flag->val.iarray[i];
		if (curve->weight->dim != 0)
			dup->weight->val[i] = curve->weight->val[i];
	}

	return dup;
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
	if (dim < 1)
		return ERROR_CURVE_INVALID_DIMENSION;

	phoebe_vector_alloc (curve->indep, dim);
	phoebe_vector_alloc (curve->dep, dim);
	phoebe_vector_alloc (curve->weight, dim);
	phoebe_array_alloc  (curve->flag, dim);

	return SUCCESS;
}

int phoebe_curve_realloc (PHOEBE_curve *curve, int dim)
{
	/**
	 * phoebe_curve_realloc:
	 * @curve: #PHOEBE_curve to reallocate.
	 * @dim: the new size of @curve.
	 *
	 * Reallocates storage memory for #PHOEBE_curve @curve.
	 *
	 * Returns: #PHOEBE_error_code.
	 *
	 * Error codes:
	 *
	 *   #ERROR_CURVE_INVALID_DIMENSION
	 *   #SUCCESS
	 */

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (dim < 0)
		return ERROR_CURVE_INVALID_DIMENSION;
	if (curve->indep->dim == dim)
		return SUCCESS;

	phoebe_vector_realloc (curve->indep,  dim);
	phoebe_vector_realloc (curve->dep,    dim);
	phoebe_vector_realloc (curve->weight, dim);
	phoebe_array_realloc  (curve->flag,   dim);

	if (dim == 0) {
		curve->indep  = NULL;
		curve->dep    = NULL;
		curve->weight = NULL;
		curve->flag   = NULL;
	}

	return SUCCESS;
}

int phoebe_curve_compute (PHOEBE_curve *curve, PHOEBE_vector *nodes, int index, PHOEBE_column_type itype, PHOEBE_column_type dtype)
{
	/**
	 * phoebe_curve_compute:
	 * @curve: a pointer to the initialized #PHOEBE_curve
	 * @nodes: a vector of nodes in which the curve should be computed
	 * @index: curve index
	 * @itype: requested independent data type (see #PHOEBE_column_type)
	 * @dtype: requested dependent data type (see #PHOEBE_column_type)
	 *
	 * Computes the @index-th model light curve or RV curve in @nodes.
	 * The computation is governed by the enumerated choices of @itype
	 * and @dtype.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, j;
	int mpage;
	int jdphs;
	int status;

	bool fti;
	double cadence;
	int rate;

	char *filter;
	char *lcin;
	PHOEBE_vector *verts;
	PHOEBE_curve  *fticurve;
	WD_LCI_parameters params;

	double A;

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (!nodes)
		return ERROR_VECTOR_NOT_INITIALIZED;

	switch (itype) {
		case PHOEBE_COLUMN_HJD:
			jdphs = 1;
		break;
		case PHOEBE_COLUMN_PHASE:
			jdphs = 2;
		break;
		case PHOEBE_COLUMN_INVALID:
			return ERROR_COLUMN_INVALID;
		break;
		default:
			phoebe_lib_error ("exception handler invoked by itype switch in phoebe_curve_compute (), please report this!\n");
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
		case PHOEBE_COLUMN_INVALID:
			return ERROR_COLUMN_INVALID;
		break;
		default:
			phoebe_lib_error ("exception handler invoked by dtype switch in phoebe_curve_compute (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	switch (mpage) {
		case 1:
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), index, &filter);
			break;
		case 2:
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), index, &filter);
			break;
	}

	/* Generate a unique LCI filename: */
	lcin = phoebe_create_temp_filename ("phoebe_lci_XXXXXX");
	if (!lcin) return ERROR_FILE_OPEN_FAILED;

	curve->passband = phoebe_passband_lookup (filter);

	/* Read in all parameters and create the LCI file: */
	status = wd_lci_parameters_get (&params, mpage, index);
	if (status != SUCCESS) return status;
	params.JDPHS = jdphs;
	create_lci_file (lcin, &params);
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_extinction"), index, &A);

	/* If finite integration time should be taken into account, we need to
	 * oversample the nodes vector.
	 */

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cadence_switch"), &fti);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cadence"), &cadence);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cadence_rate"), &rate);
	cadence /= 86400.0;

	if (fti && mpage == 1) {
		verts = phoebe_vector_new ();
		phoebe_vector_alloc (verts, (rate+1)*nodes->dim);

		printf ("itype: %d\tcadence: %lf\trate: %d\n", itype, cadence, rate);
		for (i = 0; i < nodes->dim; i++)
			for (j = -rate/2; j <= rate/2; j++)
				verts->val[i*(rate+1)+j+rate/2] = (jdphs == 1 ? nodes->val[i]+(double)j/rate*cadence : nodes->val[i]+(double)j/rate*cadence/params.PERIOD);
		
		fticurve = phoebe_curve_duplicate (curve);
	}

	switch (dtype) {
		case PHOEBE_COLUMN_MAGNITUDE:
			status = phoebe_compute_lc_using_wd (fti ? fticurve : curve, fti ? verts : nodes, lcin);
			if (status != SUCCESS) return status;
			apply_extinction_correction (fti ? fticurve : curve, A);
		break;
		case PHOEBE_COLUMN_FLUX:
			status = phoebe_compute_lc_using_wd (fti ? fticurve : curve, fti ? verts : nodes, lcin);
			if (status != SUCCESS) return status;
			apply_extinction_correction (fti ? fticurve : curve, A);
		break;
		case PHOEBE_COLUMN_PRIMARY_RV:
			status = phoebe_compute_rv1_using_wd (curve, nodes, lcin);
			if (status != SUCCESS) return status;
		break;
		case PHOEBE_COLUMN_SECONDARY_RV:
			status = phoebe_compute_rv2_using_wd (curve, nodes, lcin);
			if (status != SUCCESS) return status;
		break;
		default:
			phoebe_lib_error ("exception handler invoked by dtype switch in phoebe_curve_compute (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	remove (lcin);
	free (lcin);

	if (fti && mpage == 1) {
		/* If finite integration time is selected, the fluxes need to be averaged out. */
		phoebe_curve_alloc (curve, nodes->dim);
		for (i = 0; i < nodes->dim; i++) {
			curve->indep->val[i] = nodes->val[i];
			curve->dep->val[i] = 0.0;
			for (j = -rate/2; j <= rate/2; j++) {
				curve->dep->val[i] += fticurve->dep->val[i*(rate+1)+j+rate/2];
				printf ("%3d\t% lf\t% lf", i*(rate+1)+j+rate/2, fticurve->indep->val[i*(rate+1)+j+rate/2], fticurve->dep->val[i*(rate+1)+j+rate/2]);
				if (j != rate/2)
					printf ("\n");
			}
			curve->dep->val[i] /= (rate+1);
			printf ("\t% lf\t% lf\n", curve->indep->val[i], curve->dep->val[i]);
		}
		phoebe_curve_free (fticurve);
		phoebe_vector_free (verts);
	}
	
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

int intern_read_in_ephemeris_parameters (double *hjd0, double *period, double *dpdt, double *pshift)
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

int phoebe_curve_transform (PHOEBE_curve *curve, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype)
{
	/**
	 * phoebe_curve_transform:
	 * @curve: #PHOEBE_curve to be transformed
	 * @itype: requested independent variable (time or phase)
	 * @dtype: requested dependent variable (flux, magnitude, RV)
	 * @wtype: requested weight variable (standard weight, standard deviation,
	 *         none)
	 *
	 * Transforms curve columns to requested types.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status;
	char *rs1, *rs2;

	phoebe_debug ("entering phoebe_curve_transform ()\n");

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;

	phoebe_column_type_get_name (curve->itype, &rs1);
	phoebe_column_type_get_name (       itype, &rs2);
	phoebe_debug ("* requested transformation from %s to %s\n", rs1, rs2);
	free (rs1); free (rs2);

	phoebe_column_type_get_name (curve->dtype, &rs1);
	phoebe_column_type_get_name (       dtype, &rs2);
	phoebe_debug ("* requested transformation from %s to %s\n", rs1, rs2);
	free (rs1); free (rs2);

	phoebe_column_type_get_name (curve->wtype, &rs1);
	phoebe_column_type_get_name (       wtype, &rs2);
	phoebe_debug ("* requested transformation from %s to %s\n", rs1, rs2);
	free (rs1); free (rs2);

	if (curve->itype == PHOEBE_COLUMN_HJD && itype == PHOEBE_COLUMN_PHASE) {
		double hjd0, period, dpdt, pshift;
		intern_read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);
		status = transform_hjd_to_phase (curve->indep, hjd0, period, dpdt, 0.0);
		if (status != SUCCESS) return status;
		curve->itype = itype;
	}

	if (curve->itype == PHOEBE_COLUMN_PHASE && itype == PHOEBE_COLUMN_HJD) {
		double hjd0, period, dpdt, pshift;
		intern_read_in_ephemeris_parameters (&hjd0, &period, &dpdt, &pshift);
		status = transform_phase_to_hjd (curve->indep, hjd0, period, dpdt, 0.0);
		if (status != SUCCESS) return status;
		curve->itype = itype;
	}

	if (curve->dtype == PHOEBE_COLUMN_MAGNITUDE && dtype == PHOEBE_COLUMN_FLUX) {
 		double mnorm, mean;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mnorm"), &mnorm);

 		status = transform_magnitude_to_flux (curve->dep, mnorm);
		if (status != SUCCESS) return status;

		/*
		 * Next we need to transform standard deviation of the curve; it
		 * takes fluxes, so it must be called after the magnitudes have been
		 * transformed to fluxes.
		 */

		phoebe_vector_mean (curve->dep, &mean);
		curve->sigma = 0.5 * mean * (pow (10, 2./5.*curve->sigma) - pow (10, -2./5.*curve->sigma));

		/*
		 * If weights need to be transformed, we need to transform them *after*
		 * we transform magnitudes to fluxes, because the transformation
		 * function takes fluxes and not magnitudes.
		 */

		if (curve->wtype == PHOEBE_COLUMN_SIGMA && wtype != PHOEBE_COLUMN_UNDEFINED) {
			status = transform_magnitude_sigma_to_flux_sigma (curve->weight, curve->dep);
			if (status != SUCCESS) return status;
		}

		curve->dtype = dtype;
	}

	if (curve->dtype == PHOEBE_COLUMN_FLUX && dtype == PHOEBE_COLUMN_MAGNITUDE) {
		double mnorm, mean;
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

		/*
		 * Next we need to transform the standard deviation of the curve; it
		 * takes fluxes, so it must be called before the fluxes are transformed
		 * to magnitudes.
		 */

		phoebe_vector_mean (curve->dep, &mean);
		curve->sigma = -5./2. * log10 (1.0 + curve->sigma/mean);

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
		phoebe_vector_pad (curve->weight, 1.0);
		curve->wtype = wtype;
	}

	if (curve->wtype == PHOEBE_COLUMN_UNDEFINED && wtype == PHOEBE_COLUMN_WEIGHT) {
		phoebe_vector_pad (curve->weight, 1.0);
		curve->wtype = wtype;
	}

	phoebe_debug ("leaving phoebe_curve_transform ()\n");

	return SUCCESS;
}

int phoebe_curve_alias (PHOEBE_curve *curve, double phmin, double phmax)
{
	/**
	 * phoebe_curve_alias:
	 * @curve: curve to be aliased
	 * @phmin: start phase
	 * @phmax: end phase
	 *
	 * This function redimensiones the array of data phases by aliasing points
	 * to outside the [-0.5, 0.5] range. If the new interval is narrower, the
	 * points are omitted, otherwise they are aliased.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, j, dim;

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;

	if (phmin >= phmax)
		return ERROR_INVALID_PHASE_INTERVAL;

	if (curve->itype != PHOEBE_COLUMN_PHASE)
		return ERROR_INVALID_INDEP;

	dim = curve->indep->dim;

	/* Make the aliasing loop: */
	for (i = 0; i < dim; i++) {
		/*
		 * First alias the points; this is important because the target
		 * interval may not be overlapping with the original interval and all
		 * original points would then be removed.
		 */
		j = 1;
		while (curve->indep->val[i]-j > phmin) {
			phoebe_vector_append_element (curve->indep, curve->indep->val[i]-j);
			phoebe_vector_append_element (curve->dep,   curve->dep->val[i]);
			if (curve->weight)
				phoebe_vector_append_element (curve->weight, curve->weight->val[i]);
			phoebe_array_realloc (curve->flag, curve->flag->dim+1);
			if (curve->flag->val.iarray[i] == PHOEBE_DATA_REGULAR)
				curve->flag->val.iarray[curve->flag->dim-1] = PHOEBE_DATA_ALIASED;
			else if (curve->flag->val.iarray[i] == PHOEBE_DATA_DELETED)
				curve->flag->val.iarray[curve->flag->dim-1] = PHOEBE_DATA_DELETED_ALIASED;
			else
				phoebe_lib_error ("Exception handler invoked in phoebe_curve_alias, please report this!\n");
			
			j++;
		}
		j = 1;
		while (curve->indep->val[i]+j < phmax) {
			phoebe_vector_append_element (curve->indep, curve->indep->val[i]+j);
			phoebe_vector_append_element (curve->dep,   curve->dep->val[i]);
			if (curve->weight)
				phoebe_vector_append_element (curve->weight, curve->weight->val[i]);
			phoebe_array_realloc (curve->flag, curve->flag->dim+1);
			if (curve->flag->val.iarray[i] == PHOEBE_DATA_REGULAR)
				curve->flag->val.iarray[curve->flag->dim-1] = PHOEBE_DATA_ALIASED;
			else if (curve->flag->val.iarray[i] == PHOEBE_DATA_DELETED)
				curve->flag->val.iarray[curve->flag->dim-1] = PHOEBE_DATA_DELETED_ALIASED;
			else
				phoebe_lib_error ("Exception handler invoked in phoebe_curve_alias, please report this!\n");

			j++;
		}

		/* If the original point is outside of the phase interval, tag it: */
		if (curve->indep->val[i] < phmin || curve->indep->val[i] > phmax)
			curve->flag->val.iarray[i] = PHOEBE_DATA_OMITTED;
	}

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

	if (!passband)
		return ERROR_PASSBAND_INVALID;

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
	phoebe_array_free  (curve->flag);

	if (curve->filename)
		free (curve->filename);

	free (curve);

	return SUCCESS;
}

/* **************************************************************************** */

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
	/**
	 * phoebe_minimizer_feedback_new:
	 *
	 * Initializes a PHOEBE_minimizer_feedback structure.
	 *
	 * Returns: newly initialized #PHOEBE_minimizer_feedback.
	 */
	
	/* Allocate memory for the structure itself: */
	PHOEBE_minimizer_feedback *feedback = phoebe_malloc (sizeof (*feedback));
	
	/* Initialize structure elements: */
	feedback->qualifiers = phoebe_array_new (TYPE_STRING_ARRAY);
	feedback->initvals   = phoebe_vector_new ();
	feedback->newvals    = phoebe_vector_new ();
	feedback->ferrors    = phoebe_vector_new ();
	feedback->u_res      = phoebe_vector_new ();
	feedback->i_res      = phoebe_vector_new ();
	feedback->p_res      = phoebe_vector_new ();
	feedback->f_res      = phoebe_vector_new ();
	feedback->chi2s      = phoebe_vector_new ();
	feedback->wchi2s     = phoebe_vector_new ();
	feedback->cormat     = phoebe_matrix_new ();

	/* This is a workaround element to get computed CLA values from WD: */
	feedback->__cla      = phoebe_vector_new ();
	
	feedback->converged  = TRUE;
	
	return feedback;
}

int phoebe_minimizer_feedback_alloc (PHOEBE_minimizer_feedback *feedback, int tba, int cno, int __lcno)
{
	/**
	 * phoebe_minimizer_feedback_alloc:
	 * @feedback: #PHOEBE_minimizer_feedback structure to be allocated
	 * @tba: number of parameters to be adjusted
	 * @cno: number of input curves
	 * @__lcno: number of light curves for a temporary workaround
	 *
	 * Allocates the arrays of the #PHOEBE_minimizer_feedback structure. There
	 * are two independent array dimensions: one is determined by the number of
	 * parameters set for adjustment (@tba) and the other is determined by the
	 * number of data curves (@cno). The @tba number takes into account
	 * passband-dependent parameters as well, i.e. if a passband-dependent
	 * parameter is set for adjustment, @tba increases by the number of
	 * passbands, not just by 1.
	 *
	 * Returns: #PHOEBE_error_code
	 */

	if (!feedback)
		return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	phoebe_array_alloc  (feedback->qualifiers, tba);
	phoebe_vector_alloc (feedback->initvals,   tba);
	phoebe_vector_alloc (feedback->newvals,    tba);
	phoebe_vector_alloc (feedback->ferrors,    tba);
	phoebe_vector_alloc (feedback->u_res,      cno);
	phoebe_vector_alloc (feedback->i_res,      cno);
	phoebe_vector_alloc (feedback->p_res,      cno);
	phoebe_vector_alloc (feedback->f_res,      cno);
	phoebe_vector_alloc (feedback->chi2s,      cno);
	phoebe_vector_alloc (feedback->wchi2s,     cno);
	phoebe_matrix_alloc (feedback->cormat,     tba, tba);
	phoebe_vector_alloc (feedback->__cla,      __lcno);

	return SUCCESS;
}

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_duplicate (PHOEBE_minimizer_feedback *feedback)
{
	/**
	 * phoebe_minimizer_feedback_duplicate:
	 * @feedback: #PHOEBE_minimizer_feedback to be duplicated
	 * 
	 * Duplicates the contents of the #PHOEBE_minimizer_feedback structure @feedback.
	 * 
	 * Returns: a pointer to the duplicated #PHOEBE_minimizer_feedback.
	 */

	PHOEBE_minimizer_feedback *dup;

	if (!feedback) {
		phoebe_lib_error ("feedback structure not initialized, aborting.\n");
		return NULL;
	}

	dup = phoebe_minimizer_feedback_new ();

	dup->algorithm = feedback->algorithm;
	dup->converged = feedback->converged;
	dup->cputime   = feedback->cputime;
	dup->iters     = feedback->iters;
	dup->cfval     = feedback->cfval;

	/* There is no need to check for existence of feedback fields explicitly, */
	/* the phoebe_*_duplicate functions do that automatically.                */

	dup->qualifiers = phoebe_array_duplicate  (feedback->qualifiers);
	dup->initvals   = phoebe_vector_duplicate (feedback->initvals);
	dup->newvals    = phoebe_vector_duplicate (feedback->newvals);
	dup->ferrors    = phoebe_vector_duplicate (feedback->ferrors);
	dup->u_res      = phoebe_vector_duplicate (feedback->u_res);
	dup->i_res      = phoebe_vector_duplicate (feedback->i_res);
	dup->p_res      = phoebe_vector_duplicate (feedback->p_res);
	dup->f_res      = phoebe_vector_duplicate (feedback->f_res);
	dup->chi2s      = phoebe_vector_duplicate (feedback->chi2s);
	dup->wchi2s     = phoebe_vector_duplicate (feedback->wchi2s);
	dup->cormat     = phoebe_matrix_duplicate (feedback->cormat);
	dup->__cla      = phoebe_vector_duplicate (feedback->__cla);

	return dup;
}

int phoebe_minimizer_feedback_accept (PHOEBE_minimizer_feedback *feedback)
{
	/**
	 * phoebe_minimizer_feedback_accept:
	 * @feedback: minimizer feedback with new values of parameters.
	 *
	 * Traverses through all the parameters stored in the feedback structure
	 * and copies the values to the currently active parameter table. After
	 * all the values have been updated, the function satisfies all constraints
	 * as well.
	 *
	 * Returns: #PHOEBE_error_code.
	 */
	
	int i, index;
	char *qualifier;

	if (!feedback)
		return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;
	
	/* Get the secondary luminosities for those cases where they have been calculated by WD 
	   (otherwise the next statement will just get the input values back) 
	*/
	for (i = 0; i < feedback->__cla->dim; i++)
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_cla"), i, feedback->__cla->val[i]);

	/* Get the new values for the adjustable parameters */	
	for (i = 0; i < feedback->qualifiers->dim; i++) {
		phoebe_qualifier_string_parse (feedback->qualifiers->val.strarray[i], &qualifier, &index);
		if (index == 0)
			phoebe_parameter_set_value (phoebe_parameter_lookup (qualifier), feedback->newvals->val[i]);
		else
			phoebe_parameter_set_value (phoebe_parameter_lookup (qualifier), index-1, feedback->newvals->val[i]);
		
		free (qualifier);
	}

	/* Satisfy all the constraints: */
	phoebe_constraint_satisfy_all ();
	
	return SUCCESS;
}

int phoebe_minimizer_feedback_free (PHOEBE_minimizer_feedback *feedback)
{
	/**
	 * phoebe_minimizer_feedback_free:
	 * @feedback: #PHOEBE_minimizer_feedback pointer to be freed
	 * 
	 * Traverses through the #PHOEBE_minimizer_feedback structure,
	 * frees its contents if they were allocated, and frees the structure itself.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!feedback) return ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED;

	phoebe_array_free  (feedback->qualifiers);
	phoebe_vector_free (feedback->initvals);
	phoebe_vector_free (feedback->newvals);
	phoebe_vector_free (feedback->ferrors);
	phoebe_vector_free (feedback->u_res);
	phoebe_vector_free (feedback->i_res);
	phoebe_vector_free (feedback->p_res);
	phoebe_vector_free (feedback->f_res);
	phoebe_vector_free (feedback->chi2s);
	phoebe_vector_free (feedback->wchi2s);
	phoebe_matrix_free (feedback->cormat);
	phoebe_vector_free (feedback->__cla);

	free (feedback);

	return SUCCESS;
}

/* ************************************************************************** */

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

PHOEBE_value phoebe_value_duplicate (PHOEBE_type type, PHOEBE_value val)
{
	/**
	 * phoebe_value_duplicate:
	 * @type: input value type
	 * @val:  input value
	 *
	 * Copies the value and its type to the new instance.
	 * 
	 * Returns: copy of @val.
	 */
	
	PHOEBE_value copy;

	switch (type) {
		case TYPE_INT:
			copy.i = val.i;
		break;
		case TYPE_BOOL:
			copy.b = val.b;
		break;
		case TYPE_DOUBLE:
			copy.d = val.d;
		break;
		case TYPE_STRING:
			copy.str = strdup (val.str);
		break;
		case TYPE_INT_ARRAY:
			copy.array = phoebe_array_duplicate (val.array);
		break;
		case TYPE_BOOL_ARRAY:
			copy.array = phoebe_array_duplicate (val.array);
		break;
		case TYPE_DOUBLE_ARRAY:
			copy.vec = phoebe_vector_duplicate (val.vec);
		break;
		case TYPE_STRING_ARRAY:
			copy.array = phoebe_array_duplicate (val.array);
		break;
		case TYPE_CURVE:
			copy.curve = phoebe_curve_duplicate (val.curve);
		break;
		case TYPE_SPECTRUM:
			copy.spectrum = phoebe_spectrum_duplicate (val.spectrum);
		break;
		case TYPE_MINIMIZER_FEEDBACK:
			copy.feedback = phoebe_minimizer_feedback_duplicate (val.feedback);
		break;
		case TYPE_ANY:
			phoebe_lib_error ("invalid type (TYPE_ANY) passed to phoebe_value_duplicate (), please report this!\n");
			copy.i = -1;
			return copy;
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_value_duplicate (), please report this!\n");
			copy.i = -1;
			return copy;
	}

	return copy;
}
