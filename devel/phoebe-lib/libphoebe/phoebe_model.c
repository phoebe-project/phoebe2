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

	surface->elemno  = 0;
	surface->theta   = NULL;
	surface->phi     = NULL;
	surface->rho     = NULL;
	surface->grad    = NULL;
	surface->cosbeta = NULL;

	surface->mmsave  = NULL;
	surface->sinth   = NULL;
	surface->costh   = NULL;
	surface->sinphi  = NULL;
	surface->cosphi  = NULL;

	return surface;
}

int phoebe_star_surface_alloc (PHOEBE_star_surface *surface, int lat_raster)
{
	/**
	 * phoebe_star_surface_alloc:
	 * @surface: star surface elements to be allocated
	 * @lat_raster: latitude raster size (number of latitude circles)
	 * 
	 * Computes the number of elements from the passed latitude raster size
	 * @lat_raster, and allocates memory for local surface quantities. The
	 * number of elements on one half of the stellar hemisphere is computed by
	 * dividing each of @theta = pi/(2@lat_raster) latitude circles into 1.3
	 * @lat_raster sin(@theta) longitude steps and adding them up. Factor 1.3 is
	 * introduced to make rasterization along longitude more dense, needed for
	 * a better edge determination. This number is then multiplied by 4 to get
	 * the whole star surface.
	 *
	 * Allocated local surface quantities are: stellar co-latitude (@theta),
	 * longitude (@phi), local radius (@rho), and local gradient (@grad).
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	double DENSITY_FACTOR = 1.3;
	int i, gridsize;
	double theta;

	if (!surface)
		return ERROR_STAR_SURFACE_NOT_INITIALIZED;

	if (surface->elemno != 0)
		return ERROR_STAR_SURFACE_ALREADY_ALLOCATED;

	if (lat_raster <= 2)
		return ERROR_STAR_SURFACE_INVALID_DIMENSION;

	gridsize = 0;
	for (i = 0; i < lat_raster; i++) {
		theta = M_PI/2.0 * ((double) i + 0.5) / ((double) lat_raster);
		gridsize += 1 + (int) (DENSITY_FACTOR * lat_raster * sin (theta));
	}

	/* PHOEBE arrays: */
	surface->elemno  = 4*gridsize;
	surface->theta   = phoebe_malloc (surface->elemno * sizeof (*(surface->theta)));
	surface->phi     = phoebe_malloc (surface->elemno * sizeof (*(surface->phi)));
	surface->rho     = phoebe_malloc (surface->elemno * sizeof (*(surface->rho)));
	surface->grad    = phoebe_malloc (surface->elemno * sizeof (*(surface->grad)));
	surface->cosbeta = phoebe_malloc (surface->elemno * sizeof (*(surface->cosbeta)));

	/* WD arrays: */
/*
	surface->mmsave = phoebe_malloc ((lat_raster+1) * sizeof (*(surface->mmsave)));
	surface->sinth  = phoebe_malloc (gridsize * sizeof (*(surface->sinth)));
	surface->costh  = phoebe_malloc (gridsize * sizeof (*(surface->costh)));
	surface->sinphi = phoebe_malloc (gridsize * sizeof (*(surface->sinphi)));
	surface->cosphi = phoebe_malloc (gridsize * sizeof (*(surface->cosphi)));
*/

	return SUCCESS;
}

int phoebe_star_surface_rasterize (PHOEBE_star_surface *surface, int lat_raster)
{
	/**
	 * phoebe_star_surface_rasterize:
	 * @surface: stellar surface to be rasterized
	 * @lat_raster: raster size
	 *
	 * Rasterizes @surface onto a gridded mesh with @lat_raster elements in co-
	 * latitude and 1.3 @lat_raster sin(@theta) elements in longitude. Memory
	 * for the raster is allocated by the function and should be freed by the
	 * user after use.
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	int i, j, status;

	double theta, phi;
	int mm;

	int index = 0;

	/* No need to do any error handling, the following function does that. */

	status = phoebe_star_surface_alloc (surface, lat_raster);
	if (status != SUCCESS) {
		phoebe_lib_error ("%s", phoebe_error (status));
		return status;
	}

	for (i = 1; i <= lat_raster; i++) {
		theta = M_PI/2.0 * ((double) i - 0.5) / ((double) lat_raster);
		mm = 1 + (int) (1.3 * lat_raster * sin(theta));
		for (j = 0; j < mm; j++) {
			surface->theta[index] = theta;
			surface->theta[surface->elemno/2-index-1] = M_PI - theta;
			surface->theta[surface->elemno/2+index]   = theta;
			surface->theta[surface->elemno-index-1]   = M_PI - theta;

			phi = M_PI*((double)j+0.5)/mm;
			surface->phi[index] = phi;
			surface->phi[surface->elemno/2-index-1] = phi;
			surface->phi[surface->elemno/2+index] = 2*M_PI - phi;
			surface->phi[surface->elemno-index-1] = 2*M_PI - phi;

			index++;
		}
	}

/*
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
*/
	return SUCCESS;
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
	 * @surface. Memory for @surface needs to be allocated prior to calling
	 * this function.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;
	double rp, lambda, nu;

	if (!surface)
		return ERROR_STAR_SURFACE_NOT_INITIALIZED;

	if (!surface->rho || surface->elemno == 0)
		return ERROR_STAR_SURFACE_NOT_ALLOCATED;

	rp = phoebe_compute_polar_radius (Omega, D, q);

	/* Local radii need to be computed for only 1/4 of the stellar surface;
	 * the rest we can get by invoking symmetry.
	 */
	for (i = 0; i < surface->elemno/4; i++) {
		lambda = sin (surface->theta[i]) * cos (surface->phi[i]);
		    nu = cos (surface->theta[i]);
		surface->rho[i] =
			surface->rho[surface->elemno/2-i-1] =
			surface->rho[surface->elemno/2+i] =
			surface->rho[surface->elemno-i-1] =
				phoebe_compute_radius (rp, q, D, F, lambda, nu);
	}

	return SUCCESS;
}

double intern_dOmegadx (double x, double y, double z, double q, double D, double F)
{
	return -x/pow(x*x+y*y+z*z,3./2.)+q*(D-x)/pow((D-x)*(D-x)+y*y+z*z,3./2.)
	       +F*F*(1.+q)*x-q/D/D;
}

double intern_dOmegady (double x, double y, double z, double q, double D, double F)
{
	return -y*(1./pow(x*x+y*y+z*z,3./2.)+q/pow((D-x)*(D-x)+y*y+z*z,3./2.)
	       -F*F*(1.+q));
}

double intern_dOmegadz (double x, double y, double z, double q, double D, double F)
{
	return -z*(1./pow(x*x+y*y+z*z,3./2.)+q/pow((D-x)*(D-x)+y*y+z*z,3./2.));
}

int phoebe_star_surface_compute_gradients (PHOEBE_star_surface *surface, double q, double D, double F)
{
	/**
	 * phoebe_star_surface_compute_gradients:
	 * @surface: surface for which to compute local gradients
	 * @q: mass ratio
	 * @D: instantaneous separation
	 * @F: asynchronicity parameter
	 *
	 * Returns: #PHOEBE_error_code
	 */

	int i;
	double r, l, n;

	if (!surface)
		return ERROR_STAR_SURFACE_NOT_INITIALIZED;

	if (!surface->grad || surface->elemno == 0)
		return ERROR_STAR_SURFACE_NOT_ALLOCATED;

	for (i = 0; i < surface->elemno/4; i++) {
		r = surface->rho[i];
		l = sin(surface->theta[i])*cos(surface->phi[i]);
		n = cos(surface->theta[i]);

		surface->grad[i].x =
			surface->grad[surface->elemno/2-i-1].x =
			surface->grad[surface->elemno/2+i].x =
			surface->grad[surface->elemno-i-1].x =
				intern_dOmegadx (r*l, r*sqrt(1-l*l-n*n), r*n, q, D, F);

		surface->grad[i].y =
			surface->grad[surface->elemno/2-i-1].y =
			surface->grad[surface->elemno/2+i].y =
			surface->grad[surface->elemno-i-1].y =
				intern_dOmegady (r*l, r*sqrt(1-l*l-n*n), r*n, q, D, F);

		surface->grad[i].z =
			surface->grad[surface->elemno/2-i-1].z =
			surface->grad[surface->elemno/2+i].z =
			surface->grad[surface->elemno-i-1].z =
				intern_dOmegadz (r*l, r*sqrt(1-l*l-n*n), r*n, q, D, F);
	}

	return SUCCESS;
}

int phoebe_star_surface_compute_cosbeta (PHOEBE_star_surface *surface)
{
	/**
	 * phoebe_star_surface_compute_cosbeta:
	 * @surface: surface for which to compute cos(beta)
	 *
	 * Computes the cosine of the angle between surface normal (normalized
	 * gradient of the surface potential) and radius vector. This angle is
	 * directly related to surface deformation and thus surface element. The
	 * function requires gradients to be computed already.
	 *
	 * Returns: #PHOEBE_error_code
	 */

	int i;

	double l, n;

	for (i = 0; i < surface->elemno/4; i++) {
		l = sin(surface->theta[i])*cos(surface->phi[i]);
		n = cos(surface->theta[i]);

		surface->cosbeta[i] =
			surface->cosbeta[surface->elemno/2-i-1] =
			surface->cosbeta[surface->elemno/2+i] =
			surface->cosbeta[surface->elemno-i-1] =
				-(l*surface->grad[i].x+n*surface->grad[i].z
				+sqrt(1-l*l-n*n)*surface->grad[i].y)
				/sqrt(surface->grad[i].x*surface->grad[i].x+surface->grad[i].y*surface->grad[i].y+surface->grad[i].z*surface->grad[i].z);
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
		free (surface->cosbeta);
/*
		free(surface->mmsave);
		free(surface->sinth);
		free(surface->costh);
		free(surface->sinphi);
		free(surface->cosphi);
*/
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
