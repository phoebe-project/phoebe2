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

/**
 * LD_model:
 * @LD_LAW_LINEAR:  Linear cosine model
 * @LD_LAW_LOG:     Logarithmic model
 * @LD_LAW_SQRT:    Square root model
 * @LD_LAW_INVALID: Invalid model or not set
 */

typedef enum LD_model {
	LD_LAW_LINEAR = 1,
	LD_LAW_LOG,
	LD_LAW_SQRT,
	LD_LAW_INVALID
} LD_model;

PHOEBE_ld *phoebe_ld_new           ();
PHOEBE_ld *phoebe_ld_new_from_file (const char *filename);
int        phoebe_ld_alloc         (PHOEBE_ld *table, int dim);
int        phoebe_ld_realloc       (PHOEBE_ld *table, int dim);
int        phoebe_ld_attach        (PHOEBE_ld *table);
int        phoebe_ld_attach_all    (char *dir);
int        phoebe_ld_free          (PHOEBE_ld *table);

LD_table *phoebe_ld_table_intern_load (char *model_list);
LD_table *phoebe_ld_table_vh1993_load (char *dir);
int       phoebe_ld_table_free        (LD_table *LD);
int       phoebe_ld_get_coefficients  (LD_model ldlaw, PHOEBE_passband *passband, double M, double T, double lg, double *x, double *y);

LD_model  phoebe_ld_model_type (const char *ldlaw);

char *phoebe_ld_get_vh1993_passband_name (PHOEBE_passband *passband);

#endif
