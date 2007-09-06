#ifndef PHOEBE_SCRIPTER_PLOTTING_H
	#define PHOEBE_SCRIPTER_PLOTTING_H 1

#include <phoebe/phoebe.h>

typedef struct scripter_plot_properties
{
	bool lines;
	int  ctype;
	int  ltype;
	int  ptype;
} scripter_plot_properties;

int plot_using_gnuplot (int dim, bool reverse_y, PHOEBE_vector **indep, PHOEBE_vector **dep, scripter_plot_properties *props);

#endif
