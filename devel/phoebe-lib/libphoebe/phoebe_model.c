#include <stdlib.h>
#include <math.h>

#include "phoebe_error_handling.h"
#include "phoebe_model.h"

PHOEBE_star_surface *phoebe_star_surface_new ()
{
	/**
	 * phoebe_star_surface_new:
	 * 
	 * Initializes a new #PHOEBE_star_surface structure.
	 * 
	 * Returns: pointer to the newly initialized #PHOEBE_star_surface.
	 */

	PHOEBE_star_surface *surface = phoebe_malloc (sizeof (*surface));

	surface->elemno = 0;
	surface->theta  = NULL;
	surface->phi    = NULL;
	surface->rho    = NULL;
	surface->grad   = NULL;

	surface->mmsave = NULL;
	surface->sinth  = NULL;
	surface->costh  = NULL;
	surface->sinphi = NULL;
	surface->cosphi = NULL;

	return surface;
}

int phoebe_star_surface_alloc (PHOEBE_star_surface *surface, int elemno)
{
	/**
	 * phoebe_star_surface_alloc:
	 * @surface: star surface elements to be allocated
	 * @elemno: number of elements on the surface
	 * 
	 * This function should not be used, it is meant for future expansion.
	 * It will allocate a given number of surface elements instead of gridded
	 * sampling strategy.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!surface)
		return ERROR_STAR_SURFACE_NOT_INITIALIZED;

	if (surface->elemno != 0)
		return ERROR_STAR_SURFACE_ALREADY_ALLOCATED;

	if (elemno <= 0)
		return ERROR_STAR_SURFACE_INVALID_DIMENSION;

	surface->elemno = elemno;
	surface->theta  = phoebe_malloc (elemno * sizeof (*(surface->theta)));
	surface->phi    = phoebe_malloc (elemno * sizeof (*(surface->phi)));
	surface->rho    = phoebe_malloc (elemno * sizeof (*(surface->rho)));
	surface->grad   = phoebe_malloc (elemno * sizeof (*(surface->grad)));

	return SUCCESS;
}

PHOEBE_star_surface *phoebe_star_surface_rasterize (int gridsize)
{
	/**
	 * phoebe_star_surface_rasterize:
	 * @gridsize: raster size
	 *
	 * Rasterizes surface onto a gridded mesh with @gridsize elements in co-
	 * latitude and 1.3 @gridsize sin(#theta) elements in longitude. The
	 * function allocates and assigns values to the following
	 * #PHOEBE_star_surface fields: #elemno, #theta, #phi, #sinth, #costh,
	 * #sinph, #cosph, #mmsave.
	 * 
	 * Returns: rasterized star surface.
	 */

	PHOEBE_star_surface *surface = phoebe_star_surface_new ();

	double DENSITY_FACTOR = 1.3;
	int i, j;

	double theta, sinth, costh;
	int mm;

	int index = 0;

	if (gridsize < 3) {
		phoebe_lib_error ("surface grid size %d is invalid, aborting.\n", gridsize);
		return NULL;
	}

	/* Allocate memory for PHOEBE arrays: */
	surface->theta  = phoebe_malloc (DENSITY_FACTOR*gridsize*gridsize * sizeof(*(surface->theta)));
	surface->phi    = phoebe_malloc (DENSITY_FACTOR*gridsize*gridsize * sizeof(*(surface->phi)));

	/* Allocate memory for WD arrays: */
	surface->mmsave = phoebe_malloc ((gridsize+1) * sizeof (*(surface->mmsave)));
	surface->sinth  = phoebe_malloc (DENSITY_FACTOR*gridsize*gridsize * sizeof (*(surface->sinth)));
	surface->costh  = phoebe_malloc (DENSITY_FACTOR*gridsize*gridsize * sizeof (*(surface->costh)));
	surface->sinphi = phoebe_malloc (DENSITY_FACTOR*gridsize*gridsize * sizeof(*(surface->sinphi)));
	surface->cosphi = phoebe_malloc (DENSITY_FACTOR*gridsize*gridsize * sizeof(*(surface->cosphi)));

	surface->mmsave[0] = 0;

	for (i = 1; i <= gridsize; i++) {
		theta = M_PI/2.0 * ((double) i - 0.5) / ((double) gridsize);
		sinth = sin (theta);
		costh = cos (theta);
		mm = 1 + (int) (1.3 * gridsize * sinth);
		surface->mmsave[i] = surface->mmsave[i-1] + mm;
		for (j = 0; j < mm; j++) {
			surface-> theta[index] = theta;
			surface-> sinth[index] = sinth;
			surface-> costh[index] = costh;
			surface->   phi[index] = M_PI * ((double) j + 0.5) / mm;
			surface->sinphi[index] = sin (surface->phi[index]);
			surface->cosphi[index] = cos (surface->phi[index]);
			index++;
		}
	}
	surface->elemno = index;

	return surface;
}

int phoebe_star_surface_compute_radii (PHOEBE_star_surface *surface, double Omega, double q, double D, double F)
{
	/**
	 * phoebe_star_surface_compute_radii:
	 * @surface: surface for which to compute the local radii
	 * @Omega: surface equipotential value
	 * @q: mass ratio
	 * @D: instantaneous separation
	 * @F: asynchronicity parameter
	 *
	 * Computes local radii of all surface elements of the #PHOEBE_star_surface
	 * @surface. It allocates and assigns values to the #rho field.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	double rp, lambda, nu;

	if (!surface)
		return ERROR_STAR_SURFACE_NOT_INITIALIZED;

	if (surface->elemno <= 0)
		return ERROR_STAR_SURFACE_NOT_ALLOCATED;

	if (surface->rho)
		free (surface->rho);
	surface->rho = phoebe_malloc (surface->elemno * sizeof(*(surface->rho)));

	rp = phoebe_compute_polar_radius (Omega, D, q);
	for (i = 0; i < surface->elemno; i++) {
		lambda = surface->sinth[i] * surface->cosphi[i];
		    nu = surface->costh[i];
		surface->rho[i] = phoebe_compute_radius (rp, q, D, F, lambda, nu);
	}

	return SUCCESS;
}

int phoebe_star_surface_compute_grads (PHOEBE_star_surface *surface, double q, double D, double F)
{
	/**
	 * phoebe_star_surface_compute_grads:
	 * @surface: surface for which to compute the local radii
	 * @q: mass ratio
	 * @D: instantaneous separation
	 * @F: asynchronicity parameter
	 *
	 * Computes local gradients of all surface elements of the
	 * #PHOEBE_star_surface @surface. It allocates and assigns values to the
	 * #grad field. It requires @surface field #rho to already be allocated
	 * and its values assigned. See phoebe_star_surface_compute_radii() for
	 * details.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	double lambda, nu;

	if (!surface)
		return ERROR_STAR_SURFACE_NOT_INITIALIZED;

	if (surface->elemno <= 0)
		return ERROR_STAR_SURFACE_NOT_ALLOCATED;

	if (!surface->rho)
		return ERROR_STAR_SURFACE_RADII_NOT_COMPUTED;

	if (surface->grad)
		free (surface->grad);
	surface->grad = phoebe_malloc (surface->elemno * sizeof(*(surface->grad)));

	for (i = 0; i < surface->elemno; i++) {
		lambda = surface->sinth[i] * surface->cosphi[i];
		    nu = surface->costh[i];
		surface->grad[i] = phoebe_compute_gradient (surface->rho[i], q, D, F, lambda, nu);
	}

	return SUCCESS;
}

int phoebe_star_surface_free (PHOEBE_star_surface *surface)
{
	/**
	 * phoebe_star_surface_free:
	 * @surface: star surface to be freed
	 *
	 * Frees memory allocated for the star surface @surface and its fields.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!surface)
		return SUCCESS;

	if (surface->elemno != 0) {
		free (surface->theta);
		free (surface->phi);
		free (surface->rho);
		free (surface->grad);
	}

	free (surface);

	return SUCCESS;
}

double intern_rpole_implicit (double rp, double Omega, double D, double q)
{
	/**
	 * intern_rpole_implicit:
	 * @rp: initial guess for the polar radius
	 * @Omega: surface equipotential value
	 * @D: instantaneous separation
	 * @q: mass ratio
	 *
	 * Computes the implicit function of the surface potential at polar radius.
	 * This is internal function and should never be used outside the library
	 * code.
	 *
	 * Returns: implicit function value.
	 */

	return rp*rp*rp*rp * Omega*Omega
	     -    rp*rp*rp * 2*Omega
	     +       rp*rp * (D*D*Omega*Omega + 1 - q*q)
	     -          rp * 2*Omega*D*D
	     +               D*D;
}

double intern_drpole_implicit (double rp, double Omega, double D, double q)
{
	/**
	 * intern_drpole_implicit:
	 * @rp: initial guess for the polar radius
	 * @Omega: surface equipotential value
	 * @D: instantaneous separation
	 * @q: mass ratio
	 *
	 * Computes the implicit derivative of the surface potential at polar
	 * radius. This is internal function and should never be used outside the
	 * library code.
	 *
	 * Returns: implicit function value.
	 */

	return rp*rp*rp * 4*Omega*Omega
	     -    rp*rp * 6*Omega
	     +       rp * 2*(D*D*Omega*Omega + 1 - q*q)
	     -            2*Omega*D*D;
}

double intern_radius_implicit (double rp, double r, double q, double D, double l, double n, double F)
{
	/**
	 * intern_radius_implicit:
	 * @rp: polar radius
	 * @r: first guess of the radius value
	 * @q: mass ratio
	 * @D: instantaneous separation
	 * @l: direction cosine, @l = sin(#theta) cos(#phi)
	 * @n: direction cosine, @n = cos(#theta)
	 * @F: asynchronicity parameter
	 *
	 * Computes the local radius of the surface element in the direction of
	 * (@l, @n) and surface defined by @Omega, @q, @D, and @F.
	 *
	 * Returns: radius value.
	 */

	return 1.0/(
		1.0/rp
		+q*pow(D*D+rp*rp, -0.5)
		-q*(pow(D*D+r*r-2*r*l*D, -0.5) - r*l/D/D)
		-0.5*F*F*(1.0+q)*r*r*(1-n*n));
}

double intern_dOmegadx (double x, double y, double z, double q, double D, double F)
{
	/**
	 *
	 */

	return -x/pow(x*x+y*y+z*z,3./2.)
			+q*(D-x)/pow((D-x)*(D-x)+y*y+z*z,3./2.)
			+F*F*(1.+q)*x
			-q/D/D;
}

double intern_dOmegady (double x, double y, double z, double q, double D, double F)
{
	/**
	 *
	 */

	return -y*(1./pow(x*x+y*y+z*z,3./2.)
			+q/pow((D-x)*(D-x)+y*y+z*z,3./2.)
			-F*F*(1.+q));
}

double intern_dOmegadz (double x, double y, double z, double q, double D, double F)
{
	/**
	 *
	 */

	return -z*(1./pow(x*x+y*y+z*z,3./2.)
			+q/pow((D-x)*(D-x)+y*y+z*z,3./2.));
}


double phoebe_compute_polar_radius (double Omega, double D, double q)
{
	/**
	 * phoebe_compute_polar_radius:
	 * @Omega: surface equipotential value
	 * @D: instantaneous separation
	 * @q: mass ratio
	 *
	 * Given the value of surface potential @Omega, instantaneous separation @D,
	 * and mass ratio @q, this function computes the value of polar radius in
	 * units of semi-major axis. The computation is done iteratively, using the
	 * Newton-Raphson scheme.
	 *
	 * Returns: polar radius.
	 */

	double r0;
	double r = 0.5;

	do {
		r0 = r;
		r = r0 - intern_rpole_implicit (r0, Omega, D, q) / intern_drpole_implicit (r0, Omega, D, q);
	} while (fabs (r-r0) > 1e-6);

	return r;
}

double phoebe_compute_radius (double rp, double q, double D, double F, double lambda, double nu)
{
	/**
	 * phoebe_compute_radius:
	 * @rp: polar radius
	 * @q: mass ratio
	 * @D: instantaneous separation
	 * @F: asynchronicity parameter
	 * @lambda: direction cosine, @lambda = sin(#theta) cos(#phi)
	 * @nu: direction cosine, @nu = cos(#theta)
	 *
	 * Computes the fractional radius at the given direction iteratively,
	 * using the Newton-Raphson scheme.
	 *
	 * Returns: radius in units of semi-major axis.
	 */

	double r = rp;
	double r0;

	do {
		r0 = r;
		r = intern_radius_implicit (rp, r, q, D, lambda, nu, F);
		if (r > 1) return -1;      /* This means that the star is overcontact */
	} while (fabs (r-r0) > 1e-6);
	
	return r;
}

double phoebe_compute_gradient (double r, double q, double D, double F, double lambda, double nu)
{
	double dOdx, dOdy, dOdz;

	dOdx = intern_dOmegadx (r*lambda, r*sqrt(1-lambda*lambda-nu*nu), r*nu, q, 1.0, F);
	dOdy = intern_dOmegady (r*lambda, r*sqrt(1-lambda*lambda-nu*nu), r*nu, q, 1.0, F);
	dOdz = intern_dOmegadz (r*lambda, r*sqrt(1-lambda*lambda-nu*nu), r*nu, q, 1.0, F);

	return sqrt (dOdx*dOdx + dOdy*dOdy + dOdz*dOdz);
}
