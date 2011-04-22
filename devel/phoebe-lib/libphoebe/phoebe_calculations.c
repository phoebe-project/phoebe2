#include "phoebe_build_config.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "phoebe_accessories.h"
#include "phoebe_calculations.h"
#include "phoebe_configuration.h"
#include "phoebe_error_handling.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"
#include "phoebe_parameters.h"
#include "phoebe_spectra.h"
#include "phoebe_types.h"

#include "../libwd/wd.h"

double frac (double x)
{
	/*
	 * Returns the fractional part of a number.
	 */

	return fabs (x - (int) x);
}

int diff (const void *a, const void *b)
{
	/*
	 * Compares two doubles for sorting functions.
	 */

	const double *da = (const double *) a;
	const double *db = (const double *) b;

	return (*da > *db) - (*da < *db);
}

int diff_int (const void *a, const void *b)
{
	/*
	 * Compares two integers for sorting functions.
	 */

	const int *da = (const int *) a;
	const int *db = (const int *) b;

	return (*da > *db) - (*da < *db);
}

int phoebe_interpolate (int N, double *x, double *lo, double *hi, PHOEBE_type type, ...)
{
	/**
	 * phoebe_interpolate:
	 * @N: dimension of the interpolation space
	 * @x: @N-dimensional vector to the point of interest
	 * @lo: @N-dimensional vector of lower node values
	 * @hi: @N-dimensional vector of upper node values
	 * @type: #PHOEBE_type od interpolation values
	 * @...: (2^N)-dimensional vector of node function values of type @type.
	 *
	 * This is a general multi-dimensional linear interpolation function.
	 * It should be reasonably optimized - I gave it a lot of thought. It
	 * should also be reasonably well tested.
	 *
	 * The order of nodes and function values is very important; because of
	 * optimization, its order is inverse-binary:
	 *
	 *  lo[0] = lo[par1], lo[1] = lo[par2], ..., lo[N-1] = lo[parN]
	 *
	 *  hi[0] = hi[par1], hi[1] = hi[par2], ..., hi[N-1] = hi[parN]
	 *
	 *  fv[0] = fv (0 0 0 ... 0)
	 *
	 *  fv[1] = fv (1 0 0 ... 0)
	 *
	 *  fv[2] = fv (0 1 0 ... 0)
	 *
	 *  fv[3] = fv (1 1 0 ... 0)
	 *
	 *  fv[4] = fv (0 0 1 ... 0)
	 *
	 *  fv[5] = fv (1 0 1 ... 0)
	 *
	 *  fv[6] = fv (0 1 1 ... 0)
	 *
	 *  fv[7] = fv (1 1 1 ... 0)
	 *
	 *   .....................
	 *
	 *  where 0 and 1 are used for lower and upper node values, respectively,
	 *  listed in a consecutive parameter order.
	 *
	 * The function *modifies* the passed array fv[], so make sure you copy
	 * its contents if it is needed later. The result of the interpolation is
	 * contained in the first element of that array, fv[0].
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	int i, j, k;
	double **n;

	union {
		double *d;
		PHOEBE_vector **vec;
		PHOEBE_spectrum **s;
	} fv;

	int *powers;

	/* Let's first sort out the type of the function values: */
	va_list args;
	va_start (args, type);
	switch (type) {
		case TYPE_DOUBLE:
			fv.d = va_arg (args, double *);
		break;
		case TYPE_DOUBLE_ARRAY:
			fv.vec = va_arg (args, PHOEBE_vector **);
		break;
		case TYPE_SPECTRUM:
			fv.s = va_arg (args, PHOEBE_spectrum **);
		break;
		default:
			return ERROR_INVALID_TYPE;
	}

	/* An array of powers of 2 that we use for parametrizing the n-matrix: */
	powers = malloc ( (N+1) * sizeof (*powers));
	powers[0] = 1;
	for (i = 1; i < N+1; i++)
		powers[i] = powers[i-1] * 2;

	/* Allocate space to hold all the nodes: */
	n = malloc ( powers[N] * sizeof (*n));
	for (j = 0; j < powers[N]; j++)
		n[j] = malloc (N * sizeof (**n));

	/* Fill in the nodes: */
	for (i = 0; i < N; i++)
		for (j = 0; j < powers[N]; j++)
			n[j][i] = lo[i] + ( (j/powers[i]) % 2 ) * (hi[i] - lo[i]);

	for (i = 0; i < N; i++)
		for (j = 0; j < powers[N-i-1]; j++)
			switch (type) {
				case TYPE_DOUBLE:
					fv.d[j] += (x[N-i-1]-n[j][N-i-1])/(n[j+powers[N-i-1]][N-i-1]-n[j][N-i-1])*(fv.d[j+powers[N-i-1]]-fv.d[j]);
				break;
				case TYPE_DOUBLE_ARRAY:
					for (k = 0; k < fv.vec[j]->dim; k++)
					fv.vec[j]->val[k] += (x[N-i-1]-n[j][N-i-1])/(n[j+powers[N-i-1]][N-i-1]-n[j][N-i-1])*(fv.vec[j+powers[N-i-1]]->val[k]-fv.vec[j]->val[k]);
				break;
				case TYPE_SPECTRUM:
					for (k = 0; k < fv.s[j]->data->bins; k++)
						fv.s[j]->data->val[k] += (x[N-i-1]-n[j][N-i-1])/(n[j+powers[N-i-1]][N-i-1]-n[j][N-i-1])*(fv.s[j+powers[N-i-1]]->data->val[k]-fv.s[j]->data->val[k]);
				break;
				default:
					return ERROR_INVALID_TYPE;
			}

	for (j = 0; j < powers[N]; j++)
		free (n[j]);
	free (n);
	free (powers);

	return SUCCESS;
}

bool phoebe_phsv_constrained (int wd_model)
{
	/**
	 * phoebe_phsv_constrained:
	 * @wd_model: Model number used by WD (-1 to 6)
	 *
	 * Indicates whether the primary potential is constrained by the choice of model.
	 *
	 * Returns: TRUE/FALSE.
	 */
	switch (wd_model) {
		case 4: /* Semi-detached, primary fills Roche lobe */
		case 6: /* Double contact */
			return TRUE;
		default: 
			return FALSE;
	}
}

bool phoebe_pcsv_constrained (int wd_model)
{
	/**
	 * phoebe_pcsv_constrained:
	 * @wd_model: Model number used by WD (-1 to 6)
	 *
	 * Indicates whether the secondary potential is constrained by the choice of model.
	 *
	 * Returns: TRUE/FALSE.
	 */
	switch (wd_model) {
		case -1: /* X-ray binary */
		case  1: /* W UMa */
		case  3: /* Overcontact, no thermal contact */
		case  5: /* Semi-detached, secondary fills Roche lobe */
		case  6: /* Double contact */
			return TRUE;
		default: 
			return FALSE;
	}
}

void intern_call_wd_lc (char *atmcof, char *atmcofplanck, char *lcin, integer *request, integer *nodes, integer *L3perc, double *indep, double *dep, double *ypos, double *zpos)
{
	int wd_model;
	double params[14];
	char *phoebe_model;
	
	wd_lc (atmcof, atmcofplanck, lcin, request, nodes, L3perc, indep, dep, ypos, zpos, params);
	
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_plum1"),   params[ 0]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_plum2"),   params[ 1]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_mass1"),   params[ 2]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_mass2"),   params[ 3]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_radius1"), params[ 4]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_radius2"), params[ 5]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_mbol1"),   params[ 6]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_mbol2"),   params[ 7]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_logg1"),   params[ 8]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_logg2"),   params[ 9]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_sbr1"),    params[10]);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_sbr2"),    params[11]);
	
	/* If morphology is constrained, update parameter values: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_model"), &phoebe_model);
	wd_model = phoebe_wd_model (phoebe_model);
	
	if (phoebe_phsv_constrained (wd_model))
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_pot1"), params[12]);
	if (phoebe_pcsv_constrained (wd_model))
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_pot2"), params[13]);
	
	return;
}

int get_atmcof_filenames (char **atmcof, char **atmcofplanck)
{
	char *basedir;

	phoebe_config_entry_get ("PHOEBE_BASE_DIR", &basedir);

	*atmcof = phoebe_concatenate_strings (basedir, "/wd/phoebe_atmcof.dat", NULL);
	if (!phoebe_filename_exists (*atmcof)) {
		free (*atmcof);
		return ERROR_ATMCOF_NOT_FOUND;
	}

	*atmcofplanck = phoebe_concatenate_strings (basedir, "/wd/phoebe_atmcofplanck.dat", NULL);
	if (!phoebe_filename_exists (*atmcofplanck)) {
		free (*atmcof); free (*atmcofplanck);
		return ERROR_ATMCOFPLANCK_NOT_FOUND;
	}

	return SUCCESS;
}

int phoebe_compute_lc_using_wd (PHOEBE_curve *curve, PHOEBE_vector *indep, char *lcin)
{
	/**
	 * phoebe_compute_lc_using_wd:
	 * @curve: #PHOEBE_curve placeholder for computed fluxes
	 * @indep: an array of independent variable values (HJDs or phases)
	 * @lcin:  WD's lci filename
	 *
	 * Uses WD's LC code through a FORTRAN wrapper to obtain the fluxes.
	 * #PHOEBE_curve @curve must be initialized.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, status;
	char *atmcof, *atmcofplanck;

	PHOEBE_el3_units l3units;
	integer request, nodes, L3perc;

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (!indep)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (indep->dim == 0)
		return ERROR_VECTOR_IS_EMPTY;

	status = get_atmcof_filenames (&atmcof, &atmcofplanck);
	if (status != SUCCESS)
		return status;

	curve->type = PHOEBE_CURVE_LC;
	phoebe_curve_alloc (curve, indep->dim);

	for (i = 0; i < indep->dim; i++)
		curve->indep->val[i] = indep->val[i];

	request = 1;
	nodes = (integer) indep->dim;

	status = phoebe_el3_units_id (&l3units);
	if (status != SUCCESS)
		phoebe_lib_warning ("Third light units invalid, assuming defaults (flux units).\n");

	L3perc = 0;
	if (l3units == PHOEBE_EL3_UNITS_TOTAL_LIGHT)
		L3perc = 1;

	intern_call_wd_lc (atmcof, atmcofplanck, lcin, &request, &nodes, &L3perc, curve->indep->val, curve->dep->val, NULL, NULL);

	free (atmcof); free (atmcofplanck);
	
	return SUCCESS;
}

int phoebe_compute_rv1_using_wd (PHOEBE_curve *rv1, PHOEBE_vector *indep, char *lcin)
{
	/**
	 * phoebe_compute_rv1_using_wd:
	 * @rv1: #PHOEBE_curve placeholder for computed radial velocities
	 * @indep: an array of independent variable values (HJDs or phases);
	 *
	 * Uses WD's LC code through a FORTRAN wrapper to obtain the primary
	 * star radial velocities. #PHOEBE_curve @rv1 must be initialized.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, status;
	char *atmcof, *atmcofplanck;

	integer request, nodes, L3perc;

	if (!rv1)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (!indep)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (indep->dim == 0)
		return ERROR_VECTOR_IS_EMPTY;

	status = get_atmcof_filenames (&atmcof, &atmcofplanck);
	if (status != SUCCESS)
		return status;

	rv1->type  = PHOEBE_CURVE_RV;
	phoebe_curve_alloc (rv1, indep->dim);

	for (i = 0; i < indep->dim; i++)
		rv1->indep->val[i] = indep->val[i];

	request = 2;
	nodes = (integer) indep->dim;

	L3perc = 0;

	intern_call_wd_lc (atmcof, atmcofplanck, lcin, &request, &nodes, &L3perc, indep->val, rv1->dep->val, NULL, NULL);

	return SUCCESS;
}

int phoebe_compute_rv2_using_wd (PHOEBE_curve *rv2, PHOEBE_vector *indep, char *lcin)
{
	/**
	 * phoebe_computer_rv2_using_wd:
	 * @rv2: #PHOEBE_curve placeholder for computed radial velocities
	 * @indep: an array of independent variable values (HJDs or phases);
	 *
	 * Uses WD's LC code through a FORTRAN wrapper to obtain the secondary
	 * star radial velocities. #PHOEBE_curve @rv2 must be initialized.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, status;
	char *atmcof, *atmcofplanck;

	integer request, nodes, L3perc;

	if (!rv2)
		return ERROR_CURVE_NOT_INITIALIZED;
	if (!indep)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (indep->dim == 0)
		return ERROR_VECTOR_IS_EMPTY;

	status = get_atmcof_filenames (&atmcof, &atmcofplanck);
	if (status != SUCCESS)
		return status;

	rv2->type  = PHOEBE_CURVE_RV;
	phoebe_curve_alloc (rv2, indep->dim);

	for (i = 0; i < indep->dim; i++)
		rv2->indep->val[i] = indep->val[i];

	request = 3;
	nodes = (integer) indep->dim;

	L3perc = 0;

	intern_call_wd_lc (atmcof, atmcofplanck, lcin, &request, &nodes, &L3perc, indep->val, rv2->dep->val, NULL, NULL);

	return SUCCESS;
}

int phoebe_compute_pos_using_wd (PHOEBE_vector *poscoy, PHOEBE_vector *poscoz, char *lcin, double phase)
{
	/**
	 * phoebe_compute_pos_using_wd:
	 * @poscoy: #PHOEBE_vector placeholder for the plane-of-sky y coordinates
	 * @poscoz: #PHOEBE_vector placeholder for the plane-of-sky z coordinates
	 * @phase: phase at which the plane-of-sky coordinates are computed
	 *
	 * Uses WD's LC code through a FORTRAN wrapper to obtain the plane-of-sky
	 * coordinates. #PHOEBE_vector#s @poscoy and @poscoz must be initialized.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, status;
	int n1, n2, nspots, dim;
	char *atmcof, *atmcofplanck;

	double theta;
	int dim1 = 0, dim2 = 0;

	integer request, nodes, L3perc;
	double phs, dummy;

	if (!poscoy || !poscoz)
		return ERROR_VECTOR_NOT_INITIALIZED;
	if (poscoy->dim != 0 || poscoz->dim != 0)
		return ERROR_VECTOR_ALREADY_ALLOCATED;

	status = get_atmcof_filenames (&atmcof, &atmcofplanck);
	if (status != SUCCESS)
		return status;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize1"), &n1);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize2"), &n2);

	/* Count the number of points: */
	for (i = 1; i <= n1; i++) {
		theta = M_PI/2.0 * ((double) i - 0.5) / ((double) n1);
		dim1 += 1 + (int) (1.3 * n1 * sin(theta));
	}
	for (i = 1; i <= n2; i++) {
		theta = M_PI/2.0 * ((double) i - 0.5) / ((double) n2);
		dim2 += 1 + (int) (1.3 * n2 * sin(theta));
	}

	//phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no"), &nspots);
	PHOEBE_array *active_spotindices;
	phoebe_active_spots_get (&nspots, &active_spotindices);
	if (nspots > 0)
		phoebe_array_free (active_spotindices);

	/*
	 * Each star is covered by dimN per 1/4-sphere, and each spot adds 4 points
	 * around the surface point. That is why the maximum dimension (in absence
	 * of eclipses and with spots covering the whole hemishpere) is:
	 */

	/**************************************************************************/
	/*                                                                        */
	/*   NOTICE: WE ARE MULTIPLYING dimN WITH 4 INSTEAD OF 2 BECAUSE OF THE   */
	/*           BUG IN WD THAT SOMETIMES PLOTS POINTS THAT ARE INVISIBLE!    */
	/*           STAR SHAPE PLOTTING SHOULD REALLY BE DONE IN PHOEBE RATHER   */
	/*           THAN IN WD.                                                  */
	/*                                                                        */
	/**************************************************************************/

	dim = (4*nspots+1) * (4*dim1 + 4*dim2);
	phoebe_vector_alloc (poscoy, dim);
	phoebe_vector_pad (poscoy, sqrt(-1));
	phoebe_vector_alloc (poscoz, dim);
	phoebe_vector_pad (poscoz, sqrt(-1));

	phs = phase;
	request = 4;
	nodes   = 1;
	L3perc  = 0;

	intern_call_wd_lc (atmcof, atmcofplanck, lcin, &request, &nodes, &L3perc, &phs, &dummy, poscoy->val, poscoz->val);

	i = 0;
	while (!isnan(poscoy->val[i]) && i < poscoy->dim) i++;
	phoebe_vector_realloc (poscoy, i-1);
	phoebe_vector_realloc (poscoz, i-1);

	return SUCCESS;
}

int call_wd_to_get_logg_values (double *logg1, double *logg2)
{
	char *atmcof, *atmcofplanck, *lcin;
	int request, nodes, L3perc, status;

	WD_LCI_parameters wd_params;

	status = get_atmcof_filenames (&atmcof, &atmcofplanck);
	if (status != SUCCESS)
		return status;

	status = wd_lci_parameters_get (&wd_params, 2, 0);
	if (status != SUCCESS) return status;

	wd_params.JDPHS = 2;

	/* Generate a unique LCI filename: */
	lcin = phoebe_create_temp_filename ("phoebe_lci_XXXXXX");
	if (!lcin) return ERROR_FILE_OPEN_FAILED;
	create_lci_file (lcin, &wd_params);

	double indep[1], dep[1];
	indep[0] = 0;

	request = 2;
	nodes = 1;
	L3perc = 0;

	intern_call_wd_lc (atmcof, atmcofplanck, lcin, &request, &nodes, &L3perc, indep, dep, NULL, NULL);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"),   logg1);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"),   logg2);

	remove (lcin);
	free (lcin);

	return SUCCESS;
}

int phoebe_calculate_plum_correction (double *alpha, PHOEBE_curve *syn, PHOEBE_curve *obs, int levweight, double l3, PHOEBE_el3_units l3units)
{
	/**
	 * phoebe_calculate_plum_correction:
	 * @alpha:     placeholder for passband luminosity correction:
	 *             L1 -> L1/@alpha, L2 -> L2/@alpha
	 * @syn:       synthetic (model) light curve
	 * @obs:       observed light curve
	 * @levweight: level-dependent weighting (0 .. none, 1 .. sqrt, 2 .. lin)
	 * @l3:        value of third light
	 * @l3units:   units of third light
	 *
	 * Computes the correction @alpha by solving:
	 *
	 *   \sum_i w_i (o_i - @alpha c_i)^2 = min
	 *
	 * where o_i are observed data points with individual weights w_i, and c_i
	 * are the computed fluxes with the original value of L1. Solving this
	 * system requires solving the following least squares equation:
	 *
	 *   @alpha = (A W A^T)^{-1} A W b = \sum_i w_i o_i c_i / \sum_i w_i o_i^2
	 *
	 * If there is third light in the system, the computation is slightly
	 * modified. In case of third light in flux units, l3 is subtracted from
	 * both o_i and c_i prior to the computation of @alpha. In case of third
	 * light given in percent of total light, the computation is done as before
	 * since the total flux is scaled with (divided by) (1-l3).
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	double wsum_xx = 0.0, wsum_xy = 0.0, lw;

	if (!obs || !syn)
		return ERROR_CURVE_NOT_INITIALIZED;
	
	if (l3 > 1e-5 && l3units == PHOEBE_EL3_UNITS_FLUX) {
		for (i = 0; i < obs->dep->dim; i++) {
			switch (levweight) {
				case 0: lw = 1.0;                    break;
				case 1: lw = sqrt(obs->dep->val[i]); break;
				case 2: lw = obs->dep->val[i];       break;
				default:
					phoebe_lib_error ("level-dependent weighting invalid.\n");
					return ERROR_INVALID_WEIGHT;
			}
			
			wsum_xy += obs->weight->val[i] * lw * (obs->dep->val[i]-l3) * (syn->dep->val[i]-l3);
			wsum_xx += obs->weight->val[i] * lw * (obs->dep->val[i]-l3) * (obs->dep->val[i]-l3);
		}
	}
	else {
		for (i = 0; i < obs->dep->dim; i++) {
			switch (levweight) {
				case 0: lw = 1.0;                    break;
				case 1: lw = sqrt(obs->dep->val[i]); break;
				case 2: lw = obs->dep->val[i];       break;
				default:
					phoebe_lib_error ("level-dependent weighting (lw=%d) invalid.\n", levweight);
					return ERROR_INVALID_WEIGHT;
			}

			wsum_xy += obs->weight->val[i] * lw * obs->dep->val[i] * syn->dep->val[i];
			wsum_xx += obs->weight->val[i] * lw * obs->dep->val[i] * obs->dep->val[i];
		}
	}
	
	*alpha = wsum_xy/wsum_xx;

	return SUCCESS;
}

int phoebe_calculate_gamma_correction (double *gamma, PHOEBE_curve *syn, PHOEBE_curve *obs)
{
	/**
	 * phoebe_calculate_gamma_correction:
	 * @gamma: placeholder for gamma (center-of-mass) velocity correction:
	 *         v -> v + gamma
	 * @syn:   synthetic (model) light curve
	 * @obs:   observed light curve
	 *
	 * Computes the correction @gamma by solving:
	 *
	 *   \sum_i w_i (o_i - c_i - @gamma)^2 = min
	 *
	 * where o_i are observed data points with individual weights w_i, and c_i
	 * are the computed fluxes with the original value of @gamma. Solving this
	 * system requires solving the following least squares equation:
	 *
	 *   @gamma = (A W A^T)^{-1} A W b = \sum_i w_i o_i / \sum_i w_i
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	double wsum_y = 0.0, wsum = 0.0;

	if (!obs || !syn)
		return ERROR_CURVE_NOT_INITIALIZED;

	for (i = 0; i < obs->dep->dim; i++) {
		wsum_y += obs->weight->val[i] * obs->dep->val[i];
		wsum   += obs->weight->val[i];
	}
	*gamma = wsum_y/wsum;

	return SUCCESS;
}

double phoebe_calculate_pot1 (double D, double q, double r, double F, double lambda, double nu)
{
	/**
	 * phoebe_calculate_pot1:
	 * @D: instantaneous separation in units of semi-major axis
	 * @q: mass ratio
	 * @r: effective radius of the star
	 * @F: asynchronicity parameter
	 * @lambda: direction cosine, @lambda = sin(theta) cos(phi)
	 * @nu: direction cosine, @nu = cos(theta)
	 *
	 * Calculates the primary star surface potential according to Eq.(3.16)
	 * in PHOEBE scientific reference.
	 *
	 * Returns: value of the primary star surface potential
	 */

	return 1./r + q*(pow(D*D+r*r-2*r*lambda*D,-0.5)-r*lambda/D/D) + 0.5*F*F*(1.+q)*r*r*(1-nu*nu);
}
	
double phoebe_calculate_pot2 (double D, double q, double r, double F, double lambda, double nu)
{
	/**
	 * phoebe_calculate_pot2:
	 * @D: instantaneous separation in units of semi-major axis
	 * @q: mass ratio
	 * @r: effective radius of the star
	 * @F: asynchronicity parameter
	 * @lambda: direction cosine, @lambda = sin(theta) cos(phi)
	 * @nu: direction cosine, @nu = cos(theta)
	 *
	 * Calculates the secondary star surface potential by transforming the
	 * coordinate system to the center of the secondary star according to
	 * Eq.(3.15) in PHOEBE scientific reference, and using Eq.(3.16) to do
	 * the calculation.
	 *
	 * Returns: value of the secondary star surface potential
	 */

	double phsv;

	/* We transform the coordinate system: q -> 1/q: */
	q = 1./q;

	phsv = phoebe_calculate_pot1 (D, q, r, F, lambda, nu);
	return phsv/q + 0.5 * (q-1)/q;
}

double intern_dOmega_dx_cartesian (double x, double y, double z, double D, double q, double F)
{
	return -x*pow(x*x+y*y+z*z,-1.5) -q*(x-D)*pow((x-D)*(x-D)+y*y+z*z,-1.5) -q/D/D + F*F*(1+q)*x;
}

double intern_d2Omega_dx2_cartesian (double x, double y, double z, double D, double q, double F)
{
	return (2.*x*x-y*y-z*z)/pow(x*x+y*y+z*z,2.5) + q*(2.*(x-D)*(x-D)-y*y-z*z)/pow((x-D)*(x-D)+y*y+z*z,2.5) + F*F*(1+q);
}

int phoebe_calculate_critical_potentials (double q, double F, double e, double *L1crit, double *L2crit)
{
	/**
	 * phoebe_calculate_critical_potentials:
	 * @q: mass ratio
	 * @F: asynchronicity parameter
	 * @e: eccentricity
	 * @L1crit: placeholder for the inner (Roche) lobe value of the potential
	 * @L2crit: placeholder for the outer lobe value of the potential
	 *
	 * Calculates the value of the primary surface potential in Lagrange points
	 * L1 and L2.
	 * 
	 * Returns: #PHOEBE_error_code
	 */

	double D = 1.0 - e;
	double  xL = D/2.0;                         /* Initial x coordinate value */
	double dxL = 1.1e-6;                        /* Initial x coordinate step  */
	double Force;                               /* Gravitational force        */
	double dxLdF;                               /* Spatial derivative         */

	double xL1;                                 /* L1 x coordinate            */
	double xL2;                                 /* L2 x coordinate            */

	double q2;
	double factor;

	/* First L1: we iterate to accuracy better than 1e-6: */

	while (fabs(dxL) > 1e-6) {
		dxL = - intern_dOmega_dx_cartesian (xL, 0.0, 0.0, D, q, F)/intern_d2Omega_dx2_cartesian (xL, 0, 0, D, q, F);
		xL = xL + dxL;
	}

	xL1 = xL;
	*L1crit = phoebe_calculate_pot1 (D, q, xL1, F, 1.0, 0.0);

	/* Next, L2: we have to make sure that L2 is properly defined, i.e. behind  */
	/* the lower mass star, and that it makes sense to calculate it only in     */
	/* synchronous rotation and circular orbit case:                            */

	if (q > 1.0) q2 = 1.0/q; else q2 = q;
	D = 1.0; F = 1.0; dxL = 1.1e-6;

	factor = pow (q2/3/(q2+1), 1./3.);
	xL = 1 + factor + 1./3. * factor*factor + 1./9. * factor*factor*factor;
	while (fabs(dxL) > 1e-6)
		{
		xL = xL + dxL;
		Force = F*F*(q2+1)*xL - 1.0/xL/xL - q2*(xL-D)/fabs(pow(D-xL,3)) - q2/D/D;
		dxLdF  = 1.0/(F*F*(q2+1) + 2.0/xL/xL/xL + 2*q2/fabs(pow(D-xL,3)));
		dxL = -Force * dxLdF;
		}
	if (q > 1.0) xL = D - xL;
	xL2 = xL;
	*L2crit = 1.0/fabs(xL2) + q*(1.0/fabs(xL2-1)-xL2) + 1./2.*(q+1)*xL2*xL2;

	return SUCCESS;
}

int phoebe_calculate_masses (double sma, double P, double q, double *M1, double *M2)
{
	/**
	 * phoebe_calculate_masses:
	 * @sma: semi-major axis in solar radii
	 * @P:   orbital period in days
	 * @q:   mass ratio, M2/M1
	 * @M1:  placeholder for the primary star mass
	 * @M2:  placeholder for the secondary star mass
	 *
	 * Computes the mass of the primary and the secondary star, in units of
	 * solar mass.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	double G    = 6.673e-11;
	double RSun = 696000000.0;
	double MSun = 1.989e30;
	double day  = 86400.0;

	double Mtot = 4.0*M_PI*M_PI*sma*sma*sma*RSun*RSun*RSun/G/P/P/day/day;

	*M1 =   Mtot/(1.0+q)/MSun;
	*M2 = q*Mtot/(1.0+q)/MSun;

	return SUCCESS;
}

double phoebe_compute_phase_from_true_anomaly (double T, double e, double omega, double pshift)
{
	/*
	 * phoebe_compute_phase_from_true_anomaly:
	 * @T: true anomaly
	 * @e: eccentricity
	 * @omega: argument of periastron
	 * @pshift: phase shift
	 *
	 * Computes the phase of the passed true anomaly by solving the following
	 * equations:
	 *
	 * Eccentric anomaly:
	 * E = 2 atan ( sqrt ( (1-@e)/(1+@e) ) tan(@T/2) )
	 *
	 * Mean anomaly:
	 * M = E - @e sin (E)
	 * 
	 * Phase:
	 * P = (M + @omega)/(2*pi) - 0.25 + @pshift
	 */

	double E = 2*atan(sqrt((1-e)/(1+e)) * tan(T/2));
	double M = E - e*sin(E);
	return (M+omega)/(2.0*M_PI) - 0.25 + pshift;
}

int phoebe_compute_critical_phases (double *pp, double *scp, double *icp, double *anp, double *dnp, double perr0, double ecc, double pshift)
{
	/**
	 * phoebe_compute_critical_phases:
	 * @pp:  placeholder for periastron passage phase
	 * @scp: placeholder for superior conjunction phase
	 * @icp: placeholder for inferior conjunction phase
	 * @anp: placeholder for ascending node phase
	 * @dnp: placeholder for descending node phase
	 * @perr0: argument of periastron, in radians
	 * @ecc: orbital eccentricity
	 *
	 * Computes critical phases by solving the Kepler problem for critical
	 * true anomalies:
	 * 
	 * Periastron passage: @pp = @perr0/(2pi) - 0.25 + @pshift
	 * Superior conjunction: T = pi/2-@perr0
	 * Inferior conjunction: T = 3pi/2-@perr0
	 * Ascending node:       T = -@perr0
	 * Descending node:      T = pi-@perr0
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	/* Periastron orbital phase: */
	*pp  = perr0/(2.0*M_PI) - 0.25 + pshift;
	if (*pp <= -0.5) *pp += 1.0;
	if (*pp >   0.5) *pp -= 1.0;

	/* Superior conjunction phase: */
	*scp = phoebe_compute_phase_from_true_anomaly (M_PI/2.0-perr0, ecc, perr0, pshift);
	if (*scp <= -0.5) *scp += 1.0;
	if (*scp >   0.5) *scp -= 1.0;

	/* Inferior conjunction phase: */
	*icp = phoebe_compute_phase_from_true_anomaly (3*M_PI/2.0-perr0, ecc, perr0, pshift);
	if (*icp <= -0.5) *icp += 1.0;
	if (*icp >   0.5) *icp -= 1.0;

	/* Ascending node phase: */
	*anp = phoebe_compute_phase_from_true_anomaly (-perr0, ecc, perr0, pshift);
	if (*anp <= -0.5) *anp += 1.0;
	if (*anp >   0.5) *anp -= 1.0;

	/* Descending node phase: */
	*dnp = phoebe_compute_phase_from_true_anomaly (M_PI-perr0, ecc, perr0, pshift);
	if (*dnp <= -0.5) *dnp += 1.0;
	if (*dnp >   0.5) *dnp -= 1.0;

	return SUCCESS;
}

int phoebe_cf_compute (double *cfval, PHOEBE_cost_function cf, PHOEBE_vector *syndep, PHOEBE_vector *obsdep, PHOEBE_vector *obssig, double psigma, int lexp, double scale)
{
	/**
	 * phoebe_cf_compute:
	 * @cfval: the computed cost function value.
	 * @cf: a #PHOEBE_cost_function to be evaluated.
	 * @syndep: a #PHOEBE_vector of the model data.
	 * @obsdep: a #PHOEBE_vector of the observed data.
	 * @obssig: a #PHOEBE_vector of standard deviations.
	 * @psigma: passband standard deviation
	 * @lexp:   light level weighting exponent (0, 1 or 2)
	 * @scale: a scaling constant for computing the residuals.
	 *
	 * Computes the cost function value @cfval of the passed cost function
	 * @cf. The residuals are computed from vectors @syndep and @obsdep. If
	 * the cost function is weighted, each residual is multiplied by the
	 * inverse square of the individual @obsweight value. Since the residuals
	 * for different curves are usually compared, a scaling constant @scale
	 * can be used to renormalize the data. The @scale is usually computed as
	 * 4\pi/(L1+L2+4\piL3).
	 *
	 * Cost function @cf is of the following:
	 *
	 *   #PHOEBE_CF_STANDARD_DEVIATION
	 *
	 *   #PHOEBE_CF_WEIGHTED_STANDARD_DEVIATION
	 *
	 *   #PHOEBE_CF_SUM_OF_SQUARES
	 *
	 *   #PHOEBE_CF_EXPECTATION_CHI2
	 *
	 *   #PHOEBE_CF_CHI2
	 *
	 * Returns: a #PHOEBE_error_code.
	 */

	int i;
	double c2 = 0.0;
	double  w = 0.0;

	/* First let's do some error checking: */
	if (!syndep || !obsdep || syndep->dim <= 1 || obsdep->dim <= 1)
		return ERROR_CHI2_INVALID_DATA;

	if ( (cf == PHOEBE_CF_WEIGHTED_STANDARD_DEVIATION || cf == PHOEBE_CF_CHI2) && !obssig )
		return ERROR_CHI2_INVALID_DATA;

	if (syndep->dim != obsdep->dim)
		return ERROR_CHI2_DIFFERENT_SIZES;

	if ( (cf == PHOEBE_CF_WEIGHTED_STANDARD_DEVIATION || cf == PHOEBE_CF_CHI2) && (syndep->dim != obsdep->dim || obsdep->dim != obssig->dim) )
		return ERROR_CHI2_DIFFERENT_SIZES;
	
	switch (cf) {
		case PHOEBE_CF_STANDARD_DEVIATION:
			/*
			 * Standard deviation without weighting applied:
			 *
			 *    cfval = \sqrt {1/(N-1) \sum_i (x_calc-x_obs)^2}
			 */

			for (i = 0; i < syndep->dim; i++)
				c2 += (syndep->val[i]-obsdep->val[i])*(syndep->val[i]-obsdep->val[i]);
			*cfval = scale * sqrt (c2 / (obsdep->dim-1));
		break;
		case PHOEBE_CF_WEIGHTED_STANDARD_DEVIATION:
			/*
			 * Standard deviation with weighting applied:
			 *
			 *    cfval = \sqrt {1/(\sum w_i - 1) \sum_i w_i (x_calc-x_obs)^2}
			 */

			for (i = 0; i < syndep->dim; i++) {
				 w += 1./obssig->val[i]/obssig->val[i];
				c2 += 1./obssig->val[i]/obssig->val[i] * (syndep->val[i]-obsdep->val[i]) * (syndep->val[i]-obsdep->val[i]);
			}
			*cfval = scale * sqrt (c2 / (w-1));
		break;
		case PHOEBE_CF_SUM_OF_SQUARES:
			/*
			 * Sum of squares:
			 *
			 *   cfval = \sum_i (x_calc - x_obs)^2
			 */

			for (i = 0; i < syndep->dim; i++)
				c2 += (syndep->val[i]-obsdep->val[i])*(syndep->val[i]-obsdep->val[i]);
			*cfval = scale*scale * c2;
		break;
		case PHOEBE_CF_UNWEIGHTED:
			/*
			 * Unweighted:
			 *
			 * cfval = \sum_i (x_calc - x_obs)^2
			 */
			for (i = 0; i < syndep->dim; i++)
				c2 += (syndep->val[i]-obsdep->val[i])*(syndep->val[i]-obsdep->val[i]);
			*cfval = scale*scale * c2;
		break;
		case PHOEBE_CF_WITH_INTRINSIC_WEIGHTS:
			/*
			 * Weighted with intrinsic weights:
			 *
			 * cfval = \sum_i (x_calc - x_obs)^2/sigma_i^2
			 */
			for (i = 0; i < syndep->dim; i++)
				c2 += 1./obssig->val[i]/obssig->val[i]*(syndep->val[i]-obsdep->val[i])*(syndep->val[i]-obsdep->val[i]);
			*cfval = scale*scale * c2;
		break;
		case PHOEBE_CF_WITH_PASSBAND_WEIGHTS:
			/*
			 * Weighted with intrinsic and passband weights:
			 *
			 * cfval = 1/sigma_p^2 \sum_i (x_calc - x_obs)^2/sigma_i^2
			 */
			for (i = 0; i < syndep->dim; i++)
				c2 += 1./obssig->val[i]/obssig->val[i]*(syndep->val[i]-obsdep->val[i])*(syndep->val[i]-obsdep->val[i]);
			*cfval = scale*scale/psigma/psigma * c2;
		break;
		case PHOEBE_CF_WITH_ALL_WEIGHTS:
			/*
			 * Weighted with all weights:
			 *
			 * cfval = 1/sigma_p^2 \sum_i (x_calc - x_obs)^2/sigma_i^2/x_obs^R,
			 *
			 * where R = 0 for no level-dependent weighting, R = 1 for Poisson
			 * regime, and R = 2 for low light level regime.
			 */
			for (i = 0; i < syndep->dim; i++)
				c2 += 1./obssig->val[i]/obssig->val[i]*(syndep->val[i]-obsdep->val[i])*(syndep->val[i]-obsdep->val[i])/pow(obsdep->val[i],lexp);
			*cfval = scale*scale/psigma/psigma * c2;
		break;
		case PHOEBE_CF_EXPECTATION_CHI2:
			/*
			 * Expectation chi2:
			 *
			 *   cfval = \sum_i (x_calc-x_obs)^2 / x_calc
			 */

			for (i = 0; i < syndep->dim; i++)
				c2 += (obsdep->val[i]-syndep->val[i])*(obsdep->val[i]-syndep->val[i])/syndep->val[i];
			*cfval = scale * c2;
		break;
		case PHOEBE_CF_CHI2:
			/*
			 * Standard (sometimes called reduced) chi2:
			 *
			 *   cfval = \sum_i (x_calc-x_obs)^2 / \sigma_i^2
			 */

			for (i = 0; i < syndep->dim; i++)
				c2 += 1./obssig->val[i]/obssig->val[i] * (obsdep->val[i]-syndep->val[i])*(obsdep->val[i]-syndep->val[i]);
			*cfval = c2;
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_cf_compute (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}

	return SUCCESS;
}

int phoebe_join_chi2 (double *chi2, PHOEBE_vector *chi2s, PHOEBE_vector *weights)
{
	/*
	 * This function joins all chi2s according to their individual weights that
	 * are passed as arguments. The scheme for joining them is:
	 *
	 *   \chi^2 = 1/(\sum_i w_i) \sum_i w_i \chi^2_i
	 */

	int i;
	double w = 0.0;

	if (!chi2s)                     return ERROR_VECTOR_NOT_INITIALIZED;
	if (!weights)                   return ERROR_VECTOR_NOT_INITIALIZED;
	if (chi2s->dim < 1)             return ERROR_VECTOR_INVALID_DIMENSION;
	if (weights->dim < 1)           return ERROR_VECTOR_INVALID_DIMENSION;
	if (chi2s->dim != weights->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	*chi2 = 0.0;
	for (i = 0; i < chi2s->dim; i++) {
		    w += weights->val[i];
		*chi2 += weights->val[i] * chi2s->val[i];
	}
	*chi2 /= w;

	return SUCCESS;
}

int calculate_weighted_sum (double *sum, PHOEBE_vector *dep, PHOEBE_vector *weight)
{
	/*
	 * This function calculates the weighted sum of all values in the array.
	 */

	int i;

	if (!dep)                    return ERROR_VECTOR_NOT_INITIALIZED;
	if (!weight)                 return ERROR_VECTOR_NOT_INITIALIZED;
	if (dep->dim < 1)            return ERROR_VECTOR_INVALID_DIMENSION;
	if (weight->dim < 1)         return ERROR_VECTOR_INVALID_DIMENSION;
	if (dep->dim != weight->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	*sum = 0.0;
	for (i = 0; i < dep->dim; i++)
		*sum += dep->val[i] * weight->val[i];
	
	return SUCCESS;
}

int calculate_weighted_average (double *average, PHOEBE_vector *dep, PHOEBE_vector *weight)
{
	/*
	 * This function calculates the weighted average of the values in the array.
	 */

	int i;
	double sum;

	if (!dep)                    return ERROR_VECTOR_NOT_INITIALIZED;
	if (!weight)                 return ERROR_VECTOR_NOT_INITIALIZED;
	if (dep->dim < 1)            return ERROR_VECTOR_INVALID_DIMENSION;
	if (weight->dim < 1)         return ERROR_VECTOR_INVALID_DIMENSION;
	if (dep->dim != weight->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	*average = 0.0;
	for (i = 0; i < weight->dim; i++)
		*average += weight->val[i];

	calculate_weighted_sum (&sum, dep, weight);
	*average = sum / *average;

	return SUCCESS;
}

int calculate_weighted_sigma (double *sigma, PHOEBE_vector *dep, PHOEBE_vector *weight)
{
	/*
	 * This function calculates weighted standard deviation of the vector dep:
	 *
	 *   sigma^2 = 1/(\sum_i w_i - 1) \sum_i=1^N w_i (x_i - \bar x)^2
	 */

	int i, status;
	double w = 0.0;
	double average;

	if (!dep)                    return ERROR_VECTOR_NOT_INITIALIZED;
	if (!weight)                 return ERROR_VECTOR_NOT_INITIALIZED;
	if (dep->dim < 1)            return ERROR_VECTOR_INVALID_DIMENSION;
	if (weight->dim < 1)         return ERROR_VECTOR_INVALID_DIMENSION;
	if (dep->dim != weight->dim) return ERROR_VECTOR_DIMENSIONS_MISMATCH;

	status = calculate_weighted_average (&average, dep, weight);
	if (status != SUCCESS) return status;

	*sigma = 0.0;
	for (i = 0; i < dep->dim; i++) {
		*sigma += weight->val[i] * (dep->val[i] - average) * (dep->val[i] - average);
		w += weight->val[i];
	}
	*sigma = sqrt (*sigma / (w - 1));

	return SUCCESS;
}


double intern_calculate_phase_from_ephemeris (double hjd, double hjd0, double period, double dpdt, double pshift)
	{
	/* This function transforms heliocentric julian date (HJD) with given ori-  */
	/* gin HJD0, period P, time derivative DPDT and phase shift PSHIFT to phase */
	/* on interval [-0.5,0.5].                                                  */

	double phase;

	if (fabs(dpdt) < 1E-15) phase = pshift + fmod ((hjd-hjd0)/period, 1.0);
	else                    phase = pshift + fmod (1.0/dpdt * log (period+dpdt*(hjd-hjd0)), 1.0);

	/* If HJD0 is larger than HJD, then the difference is negative and we       */
	/* must fix that:                                                           */
	if (phase < 0.0) phase += 1.0;

	/* Now we have the phase interval [0,1], but we want [-0.5, 0.5]:           */
	if (phase > 0.5) phase -= 1.0;

	return phase;
	}

int transform_hjd_to_phase (PHOEBE_vector *vec, double hjd0, double period, double dpdt, double pshift)
	{
	/* This function transforms heliocentric Julian dates within the passed     */
	/* vector to phases according to the ephemeris parameters.                  */

	int i;
	
	for (i = 0; i < vec->dim; i++)
		vec->val[i] = intern_calculate_phase_from_ephemeris (vec->val[i], hjd0, period, dpdt, pshift);

	return SUCCESS;
	}

int transform_phase_to_hjd (PHOEBE_vector *vec, double hjd0, double period, double dpdt, double pshift)
	{
	/* This function transforms phase to HJD following this prescription:       */
	/*                                                                          */
	/*   HJD = HJD0 + (Phase - PSHIFT) * PERIOD * (1.0 + DPDT)                  */

	int i;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = hjd0 + (vec->val[i] - pshift) * period * (1.0 + dpdt);

	return SUCCESS;
	}

int transform_magnitude_to_flux (PHOEBE_vector *vec, double mnorm)
	{
	/* This function transforms dependent data contents from magnitudes to flux */
	/* with respect to the user-defined mnorm value (from the Data tab).        */

	int i;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = pow (10, -2./5. * (vec->val[i] - mnorm));

	return SUCCESS;
	}

int normalize_kms_to_orbit (PHOEBE_vector *vec, double sma, double period)
	{
	/* This function normalizes radial velocities from km/s to orbit:           */
	/*                                                                          */
	/*   RVnorm = RV * 2\pi RSun/86400s/day * sma/period                        */

	int i;

	for (i = 0; i < vec->dim; i++) {
		vec->val[i] *= 2*M_PI*696000.0/86400.0*sma/period;
	}

	return SUCCESS;
	}

int transform_flux_sigma_to_magnitude_sigma (PHOEBE_vector *weights, PHOEBE_vector *fluxes)
{
	/*
	 * This function transforms flux sigmas to magnitude sigmas. The used
	 * formula is the following:
	 *
	 *   \sigma_m^+ = -5/2 * \log (1 + \sigma_j / j)
	 *   \sigma_m^- = -5/2 * \log (1 - \sigma_j / j)
	 *   \sigma_m   =  1/2 (\sigma_m^- - \sigma_m^+)
	 *
	 * where j is the individual point flux.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;
	double seplus, seminus;

	for (i = 0; i < weights->dim; i++) {
		seminus = -5./2. * log10 (1.0 - weights->val[i]/fluxes->val[i]);
		seplus  = -5./2. * log10 (1.0 + weights->val[i]/fluxes->val[i]);
		weights->val[i] = 1./2. * (seminus - seplus);
	}

	return SUCCESS;
}

int transform_magnitude_sigma_to_flux_sigma (PHOEBE_vector *weights, PHOEBE_vector *fluxes)
{
	/*
	 * This function transforms magnitude sigmas to flux sigmas. Be sure to
	 * call this function only *after* calling transform_magnitude_to_flux ()
	 * function, because it needs transformed fluxes for this transformation!
	 *
	 * Since we do not have any notion of the \sigma_m^+ and \sigma_m^-, we
	 * pick the safer route and adopt the larger of the two, \sigma_m^+. The
	 * formula to do the transformation is thus:
	 *
	 *   \sigma_j = j [ 10^{2/5 \sigma_m^+} - 1 ].
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;

	for (i = 0; i < weights->dim; i++)
		weights->val[i] = fluxes->val[i] * (pow (10, 2./5.*weights->val[i])-1);

	return SUCCESS;
}

int transform_flux_to_magnitude (PHOEBE_vector *vec, double mnorm)
{
	/*
	 * This function transforms dependent data contents from flux to magnitudes
	 * with respect to the user-defined mnorm value.
	 */

	int i;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = mnorm - 5./2. * log10 (vec->val[i]);

	return SUCCESS;
}

int transform_sigma_to_weight (PHOEBE_vector *vec)
{
	/*
	 * phoebe_transform_sigma_to_weight:
	 * @vec: array to be transformed
	 * 
	 * Transforms standard deviation (sigma) to weights by the following
	 * equation:
	 * 
	 *   w = 1/sigma^2.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	/* People sometimes pass differences instead of sigmas to PHOEBE, which
	 * doesn't work. We test for any negative values here and bail out if
	 * found.
	 */
	for (i = 0; i < vec->dim; i++)
		if (vec->val[i] < 0)
			return ERROR_NEGATIVE_STANDARD_DEVIATION;

	/* Transform: */
	for (i = 0; i < vec->dim; i++)
		vec->val[i] = 1.0/vec->val[i]/vec->val[i];

	return SUCCESS;
}

int transform_weight_to_sigma (PHOEBE_vector *vec)
{
	/*
	 * This function transforms standard deviations (sigmas) to weights.
	 *
	 * It doesn't really make sense to transform weights to sigmas, because
	 * we don't have any notion of the weight level dependency. What this
	 * function does do is give you an idea of sigma-to-sigma variation.
	 */

	int i;

	for (i = 0; i < vec->dim; i++)
		vec->val[i] = 1./sqrt(vec->val[i]);

	return SUCCESS;
}

int calculate_main_sequence_parameters (double T1, double T2, double P0, double *L1, double *L2, double *M1, double *M2, double *q, double *a, double *R1, double *R2, double *Omega1, double *Omega2)
	{
	/* This is the table of conversions between the surface temperature T and   */
	/* luminosity L (expressed in solar luminosity), ref. Carroll & Ostlie      */
	/* (1996):                                                                  */

	double TL[46][2] =
		{
		{44500, 790000.0}, {41000, 420000.0}, {38000, 260000.0}, {35800, 170000.0},
		{33000,  97000.0}, {30000,  52000.0}, {25400,  16000.0}, {22000,   5700.0},
		{18700,   1900.0}, {15400,    830.0}, {14000,    500.0}, {13000,    320.0},
		{11900,    180.0}, {10500,     95.0}, { 9520,     54.0}, { 9230,     35.0},
		{ 8970,     26.0}, { 8720,     21.0}, { 8200,     14.0}, { 7850,     10.5},
		{ 7580,      8.6}, { 7200,      6.5}, { 6890,      3.2}, { 6440,      2.9},
		{ 6200,      2.1}, { 6030,      1.5}, { 5860,      1.1}, { 5780,      1.0},
		{ 5670,     0.79}, { 5570,     0.66}, { 5250,     0.42}, { 5080,     0.37},
		{ 4900,     0.29}, { 4730,     0.26}, { 4590,     0.19}, { 4350,     0.15},
		{ 4060,     0.10}, { 3850,    0.077}, { 3720,    0.061}, { 3580,    0.045},
		{ 3470,    0.036}, { 3370,    0.019}, { 3240,    0.011}, { 3050,   0.0053},
		{ 2940,   0.0034}, { 2640,   0.0012}
		};
	double Tsun = 5780;
	double Rsun = 696000000;
	double Msun = 1.99E30;
	double G    = 6.67E-11;

	int i;

	/* First step: let's check whether supplied temperatures are OK:            */
	if ( (T1 > TL[0][0]) || (T1 < TL[45][0]) ) return ERROR_MS_TEFF1_OUT_OF_RANGE;
	if ( (T2 > TL[0][0]) || (T2 < TL[45][0]) ) return ERROR_MS_TEFF2_OUT_OF_RANGE;
	
	/* Second step: let's find both luminosities from the T-L table:            */
	i = 0; while (T1 < TL[i][0]) i++;
	*L1 = TL[i][1] + (T1-TL[i][0]) / (TL[i-1][0]-TL[i][0]) * (TL[i-1][1]-TL[i][1]);
	i = 0; while (T2 < TL[i][0]) i++;
	*L2 = TL[i][1] + (T2-TL[i][0]) / (TL[i-1][0]-TL[i][0]) * (TL[i-1][1]-TL[i][1]);

	/* Third step: let's calculate the masses and the mass ratio:               */
	*M1 = pow (*L1, 1./3.5); *M2 = pow (*L2, 1./3.5); *q = *M2 / *M1;

	/* Forth step: let's calculate semi-major axis:                             */
	*a = pow (G*(*M1+*M2)*Msun * pow (P0*24.*3600.,2) / 4. / M_PI / M_PI, 1./3.) / Rsun;

	/* Fifth step: let's calculate the radii:                                   */
	*R1 = pow (*L1, 1./2.) * pow (T1/Tsun, -2.);
	*R2 = pow (*L2, 1./2.) * pow (T2/Tsun, -2.);

	/* Sixth step: let's calculate the potentials:                              */
	*Omega1 = phoebe_calculate_pot1 (1, *q, *R1 / *a, 1.0, 1.0, 0.0);
	*Omega2 = phoebe_calculate_pot2 (1, *q, *R2 / *a, 1.0, 1.0, 0.0);

	/* That's all. Goodbye!                                                     */
	return SUCCESS;
	}

int calculate_synthetic_scatter_seed (double *seed)
	{
	srand (time (0));
	*seed = 100000001.0 + (double) rand () / RAND_MAX * 100000000.0;

	return SUCCESS;
	}

double intern_interstellar_extinction_coefficient_a (double lambda)
{
	double a = sqrt(-1);
	double x = 10000./lambda;
	double y = x - 1.82;

	if (lambda >= 1700.0 && lambda < 3030.0)
		a = 1.752 - 0.316*x - 0.104/(pow(x-4.67,2)+0.341);
	if (lambda >= 3030.0 && lambda < 9091.0)
		a = 1.0 + 0.17699*y - 0.50447*pow(y,2) - 0.02427*pow(y,3) + 0.72085*pow(y,4)
	                      + 0.01979*pow(y,5) - 0.77530*pow(y,6) + 0.32999*pow(y,7);
	if (lambda >= 9091.0 && lambda < 33330.0)
		a = 0.574 * pow (x, 1.61);
	
	return a;
}

double intern_interstellar_extinction_coefficient_b (double lambda)
{
	double b = sqrt(-1);
	double x = 10000./lambda;
	double y = x - 1.82;

	if (lambda >= 1700.0 && lambda < 3030.0)
		b = -3.090 + 1.825*x + 1.206/(pow(x-4.62,2)+0.263);
	if (lambda >= 3030.0 && lambda < 9091.0)
		b = 1.41338*pow(y,1) + 2.28305*pow(y,2) + 1.07233*pow(y,3) - 5.38434*pow(y,4)
                         - 0.62251*pow(y,5) + 5.30260*pow(y,6) - 2.09002*pow(y,7);
	if (lambda >= 9091.0 && lambda < 33330.0)
		b = -0.527 * pow (x, 1.61);
	
	return b;
}

int apply_extinction_correction (PHOEBE_curve *curve, double A)
{
	/*
	 * This function applies a simplified interstellar extinction correction
	 * to the passed array of fluxes. The correction is simplified because it
	 * assumes a single wavelength per passband.
	 */

	int i;

	if (A != 0.0)
		for (i = 0; i < curve->dep->dim; i++)
			curve->dep->val[i] *= pow (10.0, -0.4*A);

	return SUCCESS;
}

int apply_third_light_correction (PHOEBE_curve *curve, PHOEBE_el3_units el3units, double el3value)
{

	int i;

	switch (el3units) {
		case PHOEBE_EL3_UNITS_TOTAL_LIGHT: {
			double L1, L2, el3flux;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_plum1"), &L1);
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_plum2"), &L2);

			if (el3value > 1.0-PHOEBE_NUMERICAL_ACCURACY)
				return ERROR_INVALID_EL3_VALUE;

			el3flux = el3value*(L1+L2)/4.0/3.1415926/(1.0-el3value);

			for (i = 0; i < curve->dep->dim; i++)
				curve->dep->val[i] += el3flux;
		}
		break;
		case PHOEBE_EL3_UNITS_FLUX:
			for (i = 0; i < curve->dep->dim; i++)
				curve->dep->val[i] += el3value;
		break;
		default:
			phoebe_lib_error ("exception handler invoked in apply_third_light_correction (), please report this!\n");
		break;
	}

	return SUCCESS;
}

int apply_interstellar_extinction_correction (PHOEBE_vector *wavelength, PHOEBE_vector *spectrum, double R, double E)
{
	/*
	 * This function takes the spectrum and applies the reddening based
	 * on the extinction coefficient 'R' and the color excess 'E'.
	 */

	int i;

	for (i = 0; i < spectrum->dim; i++)
		spectrum->val[i] = spectrum->val[i] / pow (10.0, 2./5.*R*E*(intern_interstellar_extinction_coefficient_a (wavelength->val[i]) + intern_interstellar_extinction_coefficient_b (wavelength->val[i])/R));

	return SUCCESS;
}

int calculate_teff_from_bv_index (int star_type, double bv, double *teff)
	{
	/* This function calculates effective temperature for MS, subgiants and gi- */
	/* ants following the idea of Flower (1996), but the coefficients come from */
	/* private communication (Flower 2004-10-11).                               */
	/*                                                                          */
	/* The star_type variable is 0 for MS, subgiants and giants and 1 for su-   */
	/* pergiants.                                                               */

	if (star_type == 0)
		{
		*teff = 3.979145106714099 -
		        0.654992268598245 * bv +
					  1.740690042385095 * bv * bv -
					  4.608815154057166 * bv * bv * bv +
					  6.792599779944473 * bv * bv * bv * bv -
					  5.396909891322525 * bv * bv * bv * bv * bv +
					  2.192970376522490 * bv * bv * bv * bv * bv * bv -
					  0.359495739295671 * bv * bv * bv * bv * bv * bv * bv;
		return SUCCESS;
		}

	if (star_type == 1)
		{
		*teff = 4.012559732366214 -
		        1.055043117465989 * bv +
					  2.133394538571825 * bv * bv -
					  2.459769794654992 * bv * bv * bv +
					  1.349423943497744 * bv * bv * bv * bv -
					  0.283942579112032 * bv * bv * bv * bv * bv;
		return SUCCESS;
		}

	*teff = 0.0;
	return ERROR_CINDEX_INVALID_TYPE;
	}
