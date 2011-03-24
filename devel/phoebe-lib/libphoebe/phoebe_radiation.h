#ifndef PHOEBE_RADIATION_H
	#define PHOEBE_RADIATION_H 1

#include "phoebe_types.h"

int phoebe_compute_passband_intensity (double *intensity, PHOEBE_hist *SED, PHOEBE_hist *PTF, int mode);

#endif

