#ifndef PHOEBE_ALLOCATIONS_H
	#define PHOEBE_ALLOCATIONS_H

#include "phoebe_global.h"
#include "phoebe_types.h"

char  *parse_data_line               (char *in);

int    read_in_synthetic_data        (PHOEBE_curve *curve, PHOEBE_vector *indep, int curve_index, PHOEBE_output_dep var);
int    read_in_observational_data    (const char *filename, PHOEBE_curve *obs, int indep, int outindep, int dep, int outdep, int weight, int outweight, bool alias, double phmin, double phmax);
int    read_in_ephemeris_parameters  (double *hjd0, double *period, double *dpdt, double *pshift);
int    read_in_adjustable_parameters (int *tba, double **values, int **indices);

int    get_passband_id               (const char *passband);
int    get_level_weighting_id        (const char *type);

int    read_in_wd_lci_parameters     (WD_LCI_parameters *params, int MPAGE, int curve);
int    read_in_wd_dci_parameters     (WD_DCI_parameters *params, int *marked_tba);

WD_DCI_parameters *wd_dci_parameters_new  ();
int                wd_dci_parameters_free (WD_DCI_parameters *params);

#endif
