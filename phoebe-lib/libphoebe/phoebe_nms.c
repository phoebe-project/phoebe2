#include <stdlib.h>

#include "phoebe_error_handling.h"
#include "phoebe_nms.h"
#include "phoebe_types.h"

PHOEBE_nms_simplex *phoebe_nms_simplex_new ()
{
	PHOEBE_nms_simplex *simplex = phoebe_malloc (sizeof (*simplex));

	simplex->corners  = phoebe_matrix_new ();
	simplex->values  = phoebe_vector_new ();
	simplex->ws1 = phoebe_vector_new ();
	simplex->ws2 = phoebe_vector_new ();

	return simplex;
}

int phoebe_nms_simplex_alloc (PHOEBE_nms_simplex *simplex, int n)
{
	phoebe_matrix_alloc (simplex->corners, /* rows = */ n, /* cols = */ n+1);
	phoebe_vector_alloc (simplex->values, n+1);
	phoebe_vector_alloc (simplex->ws1, n);
	phoebe_vector_alloc (simplex->ws2, n);

	return SUCCESS;
}

int phoebe_nms_simplex_free (PHOEBE_nms_simplex *simplex)
{
	if (!simplex) return SUCCESS;

	phoebe_matrix_free (simplex->corners);
	phoebe_vector_free (simplex->values);
	phoebe_vector_free (simplex->ws1);
	phoebe_vector_free (simplex->ws2);
	free (simplex);

	return SUCCESS;
}

double phoebe_nms_move_corner (double coeff, PHOEBE_nms_simplex *simplex, int corner, PHOEBE_vector *xc, PHOEBE_nms_parameters *params, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *))
{
  /* moves a simplex corner scaled by coeff (negative value represents 
     mirroring by the middle point of the "other" corner points)
     and gives new corner in xc and function value at xc as a 
     return value 
   */

	int i, j;
	double mp;

	if (simplex->corners->rows < 2) {
		phoebe_lib_error ("simplex cannot have less than two corners!\n");
		return -1;
	}

	for (j = 0; j < simplex->corners->cols; j++) {
		mp = 0.0;
		for (i = 0; i < simplex->corners->rows; i++) {
			if (i != corner) {
				mp += (simplex->corners->val[i][j]);
			}
		}
		mp /= (double) (simplex->corners->rows - 1);
		xc->val[j] = mp - coeff * (mp - simplex->corners->val[corner][j]);
	}

	return f (xc, params);
}

int phoebe_nms_contract_by_best (PHOEBE_nms_simplex *simplex, int best, PHOEBE_vector *xc, PHOEBE_nms_parameters *params, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *))
{
  /* Function contracts the simplex in respect to 
     best valued corner. That is, all corners besides the 
     best corner are moved. */

	/* the xc vector is simply work simplex here */

	int i, j;

	for (i = 0; i < simplex->corners->rows; i++) {
		if (i != best) {
			for (j = 0; j < simplex->corners->cols; j++) {
				simplex->corners->val[i][j] = 0.5 * (simplex->corners->val[i][j] + simplex->corners->val[best][j]);
            }

			/* evaluate function in the new point */
			phoebe_matrix_get_row (xc, simplex->corners, i);
			simplex->values->val[i] = f (xc, params);
		}
	}

	return SUCCESS;
}

int phoebe_nms_calc_center (const PHOEBE_nms_simplex *simplex, PHOEBE_vector *mp)
{
	/* calculates the center of the simplex to mp */

	int i, j;
	double val;

	for (j = 0; j < simplex->corners->cols; j++) {
		val = 0.0;
		for (i = 0; i < simplex->corners->rows; i++) {
			val += simplex->corners->val[i][j];
		}
		val /= simplex->corners->rows;
		mp->val[j] = val;
	}

	return SUCCESS;
}

double phoebe_nms_size (PHOEBE_nms_simplex *simplex)
{
	int i;
	PHOEBE_vector *res;

	double norm;
	double ss = 0.0;

	/* Calculate middle point */
	phoebe_nms_calc_center (simplex, simplex->ws2);

	for (i = 0; i < simplex->corners->rows; i++) {
		res = phoebe_vector_new ();
		phoebe_matrix_get_row (simplex->ws1, simplex->corners, i);
		phoebe_vector_subtract (res, simplex->ws1, simplex->ws2);
		phoebe_vector_norm (&norm, res);
		ss += norm;
		phoebe_vector_free (res);
	}

	return ss / (double) (simplex->corners->rows);
}

int phoebe_nms_iterate (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_vector *x, PHOEBE_nms_parameters *params, double *size, double *fval)
{
	/* Simplex iteration tries to minimize function f value */
	/* xc and xc2 vectors store tried corner point coordinates */

	PHOEBE_vector *xc  = simplex->ws1;
	PHOEBE_vector *xc2 = simplex->ws2;
	PHOEBE_vector *values  = simplex->values;
	PHOEBE_matrix *corners  = simplex->corners;

	int n = values->dim;
	int i;
	int hi = 0, s_hi = 0, lo = 0;
	double dhi, ds_hi, dlo;
	int status;
	double val, val2;

	/* get index of highest, second highest and lowest point */

	dhi = ds_hi = dlo = values->val[0];

	for (i = 1; i < n; i++)	{
		val = values->val[i];
		if (val < dlo) {
			dlo = val;
			lo = i;
		}
		else if (val > dhi) {
			ds_hi = dhi;
			s_hi = hi;
			dhi = val;
			hi = i;
		}
		else if (val > ds_hi) {
			ds_hi = val;
			s_hi = i;
		}
	}

	/* reflect the highest value */

	val = phoebe_nms_move_corner (-1.0, simplex, hi, xc, params, f);

	if (val < values->val[lo]) {
		/* reflected point becomes lowest point, try expansion */

		val2 = phoebe_nms_move_corner (-2.0, simplex, hi, xc2, params, f);

		if (val2 < values->val[lo]) {
			phoebe_matrix_set_row (corners, xc2, hi);
			values->val[hi] = val2;
		}
		else {
			phoebe_matrix_set_row (corners, xc, hi);
			values->val[hi] = val;
		}
	}

	/* reflection does not improve things enough */

	else if (val > values->val[s_hi]) {
		if (val <= values->val[hi]) {
			/* if trial point is better than highest point, replace 
			   highest point */

			phoebe_matrix_set_row (corners, xc, hi);
			values->val[hi] = val;
		}

		/* try one dimensional contraction */

		val2 = phoebe_nms_move_corner (0.5, simplex, hi, xc2, params, f);

		if (val2 <= values->val[hi]) {
			phoebe_matrix_set_row (corners, xc2, hi);
			values->val[hi] = val2;
		}
		else {
			/* contract the whole simplex in respect to the best point */

			status = phoebe_nms_contract_by_best (simplex, lo, xc, params, f);
			if (status != SUCCESS) {
				return status;
			}
		}
	}
	else {
		/* trial point is better than second highest point. 
         Replace highest point by it */

		phoebe_matrix_set_row (corners, xc, hi);
		values->val[hi] = val;
	}

	/* return lowest point of simplex as x */

	phoebe_vector_min_index (values, &lo);
	phoebe_matrix_get_row (x, corners, lo);
	*fval = values->val[lo];

	/* Update simplex size */
	*size = phoebe_nms_size (simplex);

	return SUCCESS;
}

int phoebe_nms_set (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_vector *x, PHOEBE_nms_parameters *params, double *size, PHOEBE_vector *step_size)
{
	int i, j;
	double val;

	/* first point is the original x0 */

	val = f (x, params);
	phoebe_matrix_set_row (simplex->corners, x, 0);
	simplex->values->val[0] = val;

	/* following points are initialized to x0 + step_size */
	for (i = 0; i < x->dim; i++) {
		for (j = 0; j < x->dim; j++)
			simplex->ws1->val[j] = x->val[j];

		simplex->ws1->val[i] += step_size->val[i];

		phoebe_matrix_set_row (simplex->corners, simplex->ws1, i+1);
		simplex->values->val[i+1] = f (simplex->ws1, params);
	}

	/* Initialize simplex size */
	*size = phoebe_nms_size (simplex);

	return SUCCESS;
}
