#ifndef PHOEBE_CALCULATIONS_H
	#define PHOEBE_CALCULATIONS_H 1

#include "phoebe_parameters.h"
#include "phoebe_types.h"

typedef enum PHOEBE_cost_function {
	PHOEBE_CF_STANDARD_DEVIATION,
	PHOEBE_CF_WEIGHTED_STANDARD_DEVIATION,
	PHOEBE_CF_SUM_OF_SQUARES,
	PHOEBE_CF_UNWEIGHTED,
	PHOEBE_CF_WITH_INTRINSIC_WEIGHTS,
	PHOEBE_CF_WITH_PASSBAND_WEIGHTS,
	PHOEBE_CF_WITH_ALL_WEIGHTS,
	PHOEBE_CF_EXPECTATION_CHI2,
	PHOEBE_CF_CHI2
} PHOEBE_cost_function;

/* Missing mathematical functions: */

double frac (double x);
int    diff (const void *a, const void *b);
int    diff_int (const void *a, const void *b);

int phoebe_interpolate (int N, double *x, double *lo, double *hi, PHOEBE_type type, ...);

int phoebe_cf_compute (double *cfval, PHOEBE_cost_function cf, PHOEBE_vector *syndep, PHOEBE_vector *obsdep, PHOEBE_vector *obssig, double psigma, int lexp, double scale);
int phoebe_join_chi2  (double *chi2, PHOEBE_vector *chi2s, PHOEBE_vector *weights);

/*
 * The following define statements have been derived from f2c prototypes.
 * If you run 'f2c -P lc.sub.f', f2c will create lc.sub.P. This file should
 * then be stripped of comments and appended to wd.h. The #define statement
 * below should then correspond to that prototype.
 */

#define wd_lc(atmtab,pltab,lcin,request,vertno,L3perc,indeps,deps,poscoy,poscoz,params) lc_(atmtab,pltab,lcin,request,vertno,L3perc,indeps,deps,poscoy,poscoz,params,strlen(atmtab),strlen(pltab),strlen(lcin))

int phoebe_compute_lc_using_wd  (PHOEBE_curve *curve, PHOEBE_vector *indep, char *lcin);
int phoebe_compute_rv1_using_wd (PHOEBE_curve *rv1,   PHOEBE_vector *indep, char *lcin);
int phoebe_compute_rv2_using_wd (PHOEBE_curve *rv2,   PHOEBE_vector *indep, char *lcin);
int phoebe_compute_pos_using_wd (PHOEBE_vector *poscoy, PHOEBE_vector *poscoz, char *lcin, double phase);

int call_wd_to_get_logg_values (double *logg1, double *logg2);

int phoebe_calculate_plum_correction (double *alpha, PHOEBE_curve *syn, PHOEBE_curve *obs, int levweight, double l3, PHOEBE_el3_units l3units);
int phoebe_calculate_gamma_correction (double *gamma, PHOEBE_curve *syn, PHOEBE_curve *obs);

double phoebe_calculate_pot1   (double D, double q, double r, double F, double lambda, double nu);
double phoebe_calculate_pot2   (double D, double q, double r, double F, double lambda, double nu);
int    phoebe_calculate_masses (double sma, double P, double q, double *M1, double *M2);

int phoebe_calculate_critical_potentials (double q, double F, double e, double *L1crit, double *L2crit);
int phoebe_compute_critical_phases       (double *pp, double *scp, double *icp, double *anp, double *dnp, double perr0, double ecc, double pshift);

int calculate_weighted_sum     (double *sum,     PHOEBE_vector *dep, PHOEBE_vector *weight);
int calculate_weighted_average (double *average, PHOEBE_vector *dep, PHOEBE_vector *weight);
int calculate_weighted_sigma   (double *sigma,   PHOEBE_vector *dep, PHOEBE_vector *weight);

double intern_calculate_phase_from_ephemeris (double hjd, double hjd0, double period, double dpdt, double pshift);

int transform_hjd_to_phase                  (PHOEBE_vector *vec, double hjd0, double period, double dpdt, double pshift);
int transform_phase_to_hjd                  (PHOEBE_vector *vec, double hjd0, double period, double dpdt, double pshift);
int transform_magnitude_to_flux             (PHOEBE_vector *vec, double mnorm);
int transform_magnitude_sigma_to_flux_sigma (PHOEBE_vector *weights, PHOEBE_vector *fluxes);
int transform_flux_to_magnitude             (PHOEBE_vector *vec, double mnorm);
int transform_flux_sigma_to_magnitude_sigma (PHOEBE_vector *weights, PHOEBE_vector *fluxes);
int normalize_kms_to_orbit                  (PHOEBE_vector *vec, double sma, double period);
int transform_sigma_to_weight               (PHOEBE_vector *vec);
int transform_weight_to_sigma               (PHOEBE_vector *vec);

int calculate_main_sequence_parameters (double T1, double T2, double P0,
			  double *L1, double *L2, double *M1, double *M2, double *q, double *a,
			  double *R1, double *R2, double *Omega1, double *Omega2);

int calculate_synthetic_scatter_seed (double *seed);

/* ***********************   Extrinsic corrections   ************************ */

int apply_extinction_correction  (PHOEBE_curve *curve, double A);
int apply_third_light_correction (PHOEBE_curve *curve, PHOEBE_el3_units el3units, double el3value);

/* ************************************************************************** */

int apply_interstellar_extinction_correction (PHOEBE_vector *wavelength, PHOEBE_vector *spectrum, double R, double E);
int calculate_teff_from_bv_index (int star_type, double bv, double *teff);

bool phoebe_phsv_constrained (int wd_model);
bool phoebe_pcsv_constrained (int wd_model);

#endif
