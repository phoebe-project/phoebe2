#ifndef PHOEBE_ALLOCATIONS_H
	#define PHOEBE_ALLOCATIONS_H

#include "phoebe_global.h"
#include "phoebe_fortran_interface.h"

int    read_in_ephemeris_parameters  (double *hjd0, double *period, double *dpdt, double *pshift);
int    read_in_adjustable_parameters (int *tba, double **values);

int    get_level_weighting_id        (const char *type);

#endif
