#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "phoebe_error_handling.h"
#include "phoebe_parameters.h"

int apply_asini_constraint (double *a, double *i, double asini, int direction)
{
switch (direction) {
	case +1:
		/* This means that i is constant and a should be computed:                */
		*a = asini / sin(*i/180.0*M_PI);
	break;
	case -1:
		/* This means that a is constant and i should be computed:                */
		*i = 180.0/M_PI*asin(asini/*a);
	break;
}

}
