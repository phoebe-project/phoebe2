#ifndef PHOEBE_PLOTTING_H
	#define PHOEBE_PLOTTING_H 1

#include <stdarg.h>

#include "phoebe_types.h"

typedef struct PHOEBE_plot_properties
	{
	bool lines;
	int  ctype;
	int  ltype;
	int  ptype;
	} PHOEBE_plot_properties;

int plot_using_gnuplot (int dim, bool reverse_y, PHOEBE_vector **indep, PHOEBE_vector **dep, PHOEBE_plot_properties *props);

#endif
