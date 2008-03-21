#ifndef PHOEBE_MODEL_H
	#define PHOEBE_MODEL_H 1

typedef int PHOEBE_star_id;

typedef struct PHOEBE_star_surface {
	int     elemno;
	double *theta;
	double *phi;
	double *rho;
	struct {
		double x;
		double y;
		double z;
	} *grad;
	double *cosbeta;

	/* These are WD-compatible arrays and may be removed in future: */
	int    *mmsave;
	double *sinth;
	double *costh;
	double *sinphi;
	double *cosphi;
} PHOEBE_star_surface;

PHOEBE_star_surface *phoebe_star_surface_new               ();
int                  phoebe_star_surface_alloc             (PHOEBE_star_surface *surface, int lat_raster);
int                  phoebe_star_surface_rasterize         (PHOEBE_star_surface *surface, int lat_raster);
int                  phoebe_star_surface_compute_radii     (PHOEBE_star_surface *surface, double Omega, double q, double D, double F);
int                  phoebe_star_surface_compute_gradients (PHOEBE_star_surface *surface, double q, double D, double F);
int                  phoebe_star_surface_compute_cosbeta   (PHOEBE_star_surface *surface);
int                  phoebe_star_surface_free              (PHOEBE_star_surface *surface);

typedef struct PHOEBE_star {
	PHOEBE_star_id       id;
	PHOEBE_star_surface *surface;
} PHOEBE_star;

/* Roche model computation: */

double phoebe_compute_polar_radius (double Omega, double D, double q);
double phoebe_compute_radius       (double rp, double q, double D, double F, double lambda, double nu);

#endif
