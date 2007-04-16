#include <stdlib.h>
#include <math.h>

#include "phoebe_global.h"

/*
 * This source file contains all Roche geometry related functions.
 */

double rp_explicit (double rp, double Omega, double D, double q)
{
	/*
 	 * This is the explicit function that connects the polar radius and the
	 * Kopal potential.
 	 */

	return rp*rp*rp*rp * Omega*Omega
	     -    rp*rp*rp * 2*Omega
	     +       rp*rp * (D*D*Omega*Omega + 1 - q*q)
	     -          rp * 2*Omega*D*D
	     +               D*D;
}

double drp_explicit (double rp, double Omega, double D, double q)
{
	/*
	 * This is the derivative of the above function.
	 */

	return rp*rp*rp * 4*Omega*Omega
	     -    rp*rp * 6*Omega
	     +       rp * 2*(D*D*Omega*Omega + 1 - q*q)
	     -            2*Omega*D*D;
}

double r_explicit (double rp, double r, double q, double D, double l, double n, double F)
{
	/*
	 * This is the explicit function that connects the radius r(l,m,n) and the
	 * Kopal potential. l, m and n are the direction cosines; m is computed
	 * from l and n, so it is not passed as an argument here.
	 */

	return 1.0/(
		1.0/rp
		+q*pow(D*D+rp*rp, -0.5)
		-q*(pow(D*D+r*r-2*r*l*D, -0.5) - r*l/D/D)
		-0.5*F*F*(1.0+q)*r*r*(1-n*n));
}

double phoebe_compute_polar_radius (double Omega, double D, double q)
{
	/*
	 * This function computes the polar radius iteratively, using the Newton-
	 * Raphson scheme.
	 */

	double r0;
	double r = 0.5;

	do {
		r0 = r;
		r = r0 - rp_explicit (r0, Omega, D, q) / drp_explicit (r0, Omega, D, q);
	} while (fabs (r-r0) > PHOEBE_NUMERICAL_ACCURACY);

	return r;
}

double phoebe_get_radius (double rp, double q, double D, double F, double lambda, double nu)
{
	/*
	 * This function computes the radius at the given direction iteratively,
	 * using the Newton-Raphson scheme.
	 */

	double r = rp;
	double r0;

	do {
		r0 = r;
		r = r_explicit (rp, r, q, D, lambda, nu, F);
		if (r > 1) break;          /* This means that the star is overcontact */
	} while (fabs (r-r0) > PHOEBE_NUMERICAL_ACCURACY);
	
	if (r > 1) return sqrt(-1);        /* We don't handle overcontacts yet :( */
	else return r;
}
