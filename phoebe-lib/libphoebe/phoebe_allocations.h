#ifndef PHOEBE_ALLOCATIONS_H
	#define PHOEBE_ALLOCATIONS_H

#include "phoebe_global.h"
#include "phoebe_fortran_interface.h"

char  *phoebe_clean_data_line        (char *line);

int    read_in_ephemeris_parameters  (double *hjd0, double *period, double *dpdt, double *pshift);
int    read_in_adjustable_parameters (int *tba, double **values);

int    get_passband_id               (const char *passband);
int    get_level_weighting_id        (const char *type);

#endif
