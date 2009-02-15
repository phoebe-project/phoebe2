#include <math.h>
#include <stdlib.h>

#include "phoebe_error_handling.h"
#include "phoebe_nms.h"
#include "phoebe_types.h"

PHOEBE_nms_simplex *phoebe_nms_simplex_new ()
{
	PHOEBE_nms_simplex *simplex = phoebe_malloc (sizeof (*simplex));

	simplex->corners = phoebe_matrix_new ();
	simplex->values  = phoebe_vector_new ();
	simplex->ws1     = phoebe_vector_new ();
	simplex->ws2     = phoebe_vector_new ();
	simplex->center  = phoebe_vector_new ();
	simplex->delta   = phoebe_vector_new ();
	simplex->xmc     = phoebe_vector_new ();
	simplex->S2      = 0.0;

	return simplex;
}

int phoebe_nms_simplex_alloc (PHOEBE_nms_simplex *simplex, int n)
{
	phoebe_matrix_alloc (simplex->corners, /* cols = */ n, /* rows = */ n+1);
	phoebe_vector_alloc (simplex->values, n+1);
	phoebe_vector_alloc (simplex->ws1, n);
	phoebe_vector_alloc (simplex->ws2, n);
	phoebe_vector_alloc (simplex->center, n);
	phoebe_vector_alloc (simplex->delta, n);
	phoebe_vector_alloc (simplex->xmc, n);

	return SUCCESS;
}

int phoebe_nms_simplex_free (PHOEBE_nms_simplex *simplex)
{
	if (!simplex) return SUCCESS;

	phoebe_matrix_free (simplex->corners);
	phoebe_vector_free (simplex->values);
	phoebe_vector_free (simplex->ws1);
	phoebe_vector_free (simplex->ws2);
	phoebe_vector_free (simplex->center);
	phoebe_vector_free (simplex->delta);
	phoebe_vector_free (simplex->xmc);

	free (simplex);

	return SUCCESS;
}

double phoebe_nms_try_corner_move (const double coeff, PHOEBE_nms_simplex *simplex, int corner, PHOEBE_vector *xc, PHOEBE_nms_parameters *params, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *))
{
	/* moves a simplex corner scaled by coeff (negative value represents 
     mirroring by the middle point of the "other" corner points)
     and gives new corner in xc and function value at xc as a 
     return value 
   */

	const int P = simplex->corners->rows;
	int i;

	double alpha = (1 - coeff) * P / (P - 1.0);
	double beta = (P * coeff - 1.0) / (P - 1.0);

	for (i = 0; i < xc->dim; i++)
		xc->val[i] = alpha*simplex->center->val[i] + beta*simplex->corners->val[corner][i];

	return f (xc, params);
}

void phoebe_nms_update_point (PHOEBE_nms_simplex *simplex, int i, PHOEBE_vector *x, double val)
{
	const int P = simplex->corners->rows;
	double d, xmcd;
	int j;

	for (j = 0; j < simplex->delta->dim; j++) {
		/* Compute delta = x - x_orig */
		simplex->delta->val[j] = x->val[j]-simplex->corners->val[i][j];

		/* Compute xmc = x_orig - c */
		simplex->xmc->val[j] = simplex->corners->val[i][j]-simplex->center->val[j];
	}

	/* Update size: S2' = S2 + (2/P) * (x_orig - c).delta + (P-1)*(delta/P)^2 */
	phoebe_vector_norm (&d, simplex->delta);
	phoebe_vector_dot_product (&xmcd, simplex->xmc, simplex->delta);
	simplex->S2 += (2.0/P)*xmcd + (P-1.0)/P*d*d/P;

	/* Update center:  c' = c + (x - x_orig) / P */
	for (j = 0; j < simplex->center->dim; j++) {
		simplex->center->val[j] += simplex->delta->val[j]/P;
		simplex->corners->val[i][j] = x->val[j];
	}

	simplex->values->val[i] = val;
}

int phoebe_nms_contract_by_best (PHOEBE_nms_simplex *simplex, PHOEBE_nms_parameters *params, int best, PHOEBE_vector *xc, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *))
{
  /* Function contracts the simplex in respect to best valued
     corner. That is, all corners besides the best corner are moved.
     (This function is rarely called in practice, since it is the last
     choice, hence not optimised - BJG)  */

	int i, j, k;
	double newval;

	for (i = 0; i < simplex->corners->rows; i++) {
		if (i != best) {
			for (j = 0; j < simplex->corners->cols; j++) {
				newval = 0.5 * (simplex->corners->val[i][j] + simplex->corners->val[best][j]);
				simplex->corners->val[i][j] = newval;
			}

			/* evaluate function in the new point */
			for (k = 0; k < xc->dim; k++)
				xc->val[k] = simplex->corners->val[i][k];
			newval = f (xc, params);
			simplex->values->val[i] = newval;

			/* notify caller that we found at least one bad function value.
			 * we finish the contraction (and do not abort) to allow the user
			 * to handle the situation
			 */
/*
			if (!gsl_finite (newval)) {
				status = GSL_EBADFUNC;
			}
*/
		}
	}

	return SUCCESS;
}

int phoebe_nms_compute_center (const PHOEBE_nms_simplex *simplex, PHOEBE_vector *center)
{
	/* calculates the center of the simplex and stores in center */

	const int P = simplex->corners->rows;
	int i, j;

	phoebe_vector_pad (center, 0.0);

	for (i = 0; i < simplex->corners->rows; i++)
		for (j = 0; j < simplex->corners->cols; j++)
			center->val[j] += simplex->corners->val[i][j];

	for (j = 0; j < center->dim; j++)
		center->val[j] /= P;

	return SUCCESS;
}

double phoebe_nms_compute_size (PHOEBE_nms_simplex *simplex, const PHOEBE_vector *center)
{
	/* calculates simplex size as rms sum of length of vectors 
     from simplex center to corner points:     

     sqrt( sum ( || y - y_middlepoint ||^2 ) / n )
   */

	int i, j;
	double t, ss = 0.0;

	for (i = 0; i < simplex->corners->rows; i++) {

		for (j = 0; j < simplex->corners->cols; j++)
			simplex->ws1->val[j] = simplex->corners->val[i][j]-center->val[j];

		phoebe_vector_norm (&t, simplex->ws1);
		ss += t*t;
    }

	/* Store squared size in the state */
	simplex->S2 = ss / simplex->corners->rows;
	return sqrt (simplex->S2);
}

int phoebe_nms_simplex_set (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_nms_parameters *params, PHOEBE_vector *x, double *size, const PHOEBE_vector *step_size)
{
	int i, j;
	double val;

	val = f (x, params);
	for (j = 0; j < x->dim; j++)
		simplex->corners->val[0][j] = x->val[j];
	simplex->values->val[0] = val;

	/* Change ws1 to x+step, one by one: */
	for (i = 0; i < x->dim; i++) {

		for (j = 0; j < x->dim; j++)
			simplex->ws1->val[j] = x->val[j];

		simplex->ws1->val[i] = x->val[i]+step_size->val[i];
		val = f (simplex->ws1, params);

		for (j = 0; j < simplex->corners->cols; j++)
			simplex->corners->val[i+1][j] = simplex->ws1->val[j];
		simplex->values->val[i+1] = val;
	}

	phoebe_nms_compute_center (simplex, simplex->center);

	/* Initialize simplex size */
	*size = phoebe_nms_compute_size (simplex, simplex->center);

	return SUCCESS;
}

int phoebe_nms_simplex_iterate (PHOEBE_nms_simplex *simplex, double (*f) (PHOEBE_vector *, PHOEBE_nms_parameters *), PHOEBE_nms_parameters *params, PHOEBE_vector *x, double *size, double *fval)
{
	/* Simplex iteration tries to minimize function f value */
	/* Includes corrections from Ivo Alxneit <ivo.alxneit@psi.ch> */

	/* xc and xc2 vectors store tried corner point coordinates */

	const int n = simplex->values->dim;
	int i, j;
	int hi, s_hi, lo;
	double dhi, ds_hi, dlo;
	int status;
	double val, val2;

	/* get index of highest, second highest and lowest point */

	dhi = dlo = simplex->values->val[0];
	hi = 0;
	lo = 0;

	ds_hi = simplex->values->val[1];
	s_hi = 1;

	for (i = 1; i < n; i++) {
		val = simplex->values->val[i];
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

	/* try reflecting the highest value point */
	val = phoebe_nms_try_corner_move (-1.0, simplex, hi, simplex->ws1, params, f);

	if (val < simplex->values->val[lo]) {
		/* reflected point is lowest, try expansion */
		val2 = phoebe_nms_try_corner_move (-2.0, simplex, hi, simplex->ws2, params, f);

		if (val2 < simplex->values->val[lo]) {
			phoebe_nms_update_point (simplex, hi, simplex->ws2, val2);
		}
		else {
			phoebe_nms_update_point (simplex, hi, simplex->ws1, val);
		}
	}
	else if (val > simplex->values->val[s_hi]) {
		/* reflection does not improve things enough */

		if (val <= simplex->values->val[hi]) {
			/* if trial point is better than highest point, replace
			   highest point */
			phoebe_nms_update_point (simplex, hi, simplex->ws1, val);
		}

		/* try one-dimensional contraction */
		val2 = phoebe_nms_try_corner_move (0.5, simplex, hi, simplex->ws2, params, f);

		if (val2 <= simplex->values->val[hi]) {
			phoebe_nms_update_point (simplex, hi, simplex->ws2, val2);
		}
		else {
			/* contract the whole simplex about the best point */

			status = phoebe_nms_contract_by_best (simplex, params, lo, simplex->ws1, f);
		}
	}
	else {
		/* trial point is better than second highest point.  Replace
		   highest point by it */

		phoebe_nms_update_point (simplex, hi, simplex->ws1, val);
    }

	/* return lowest point of simplex as x */
	phoebe_vector_min_index (simplex->values, &lo);
	for (j = 0; j < simplex->corners->cols; j++)
		x->val[j] = simplex->corners->val[lo][j];
	*fval = simplex->values->val[lo];

	/* Update simplex size */
	{
		double S2 = simplex->S2;

		if (S2 > 0) {
			*size = sqrt (S2);
		}
		else {
			/* recompute if accumulated error has made size invalid */
			*size = phoebe_nms_compute_size (simplex, simplex->center);
		}
	}

	return SUCCESS;
}
