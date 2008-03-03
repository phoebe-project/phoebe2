#ifndef PHOEBE_LD_H
	#define PHOEBE_LD_H 1

#include "phoebe_global.h"

typedef struct LD_table {
	PHOEBE_array *Mnodes;
	PHOEBE_array *Tnodes;
	PHOEBE_array *lgnodes;
	struct {
		char *fn;
		long int pos;
	} ***table;
} LD_table;

extern LD_table *PHOEBE_ld_table;

typedef enum LDLaw {
	LD_LAW_LINEAR = 1,
	LD_LAW_LOG,
	LD_LAW_SQRT,
	LD_LAW_INVALID
} LDLaw;

LD_table *phoebe_ld_table_vh1993_load (char *dir);
int       phoebe_ld_table_free        (LD_table *LD);
int       phoebe_ld_get_coefficients  (LDLaw ldlaw, PHOEBE_passband *passband, double M, double T, double lg, double *x, double *y);

LDLaw   phoebe_ld_model_type (const char *ldlaw);

char *phoebe_ld_get_vh1993_passband_name (PHOEBE_passband *passband);

#endif
