#ifndef PHOEBE_LD_H
	#define PHOEBE_LD_H 1

#include "phoebe_global.h"

typedef struct LDrecord {
		char    *filename;
		long int pos;
		int      T;
		double   lg;
		double   M;
	} LDrecord;

typedef struct LDelem {
	int T;
	double lg;
	double M;
} LDelem;

typedef struct LDtable {
	LDelem  *elem;
	char    *filename;
	long int filepos;
	struct LDtable *Tprev;
	struct LDtable *Tnext;
	struct LDtable *lgprev;
	struct LDtable *lgnext;
	struct LDtable *Mprev;
	struct LDtable *Mnext;
} LDtable;

extern int      PHOEBE_ld_table_size;
extern LDtable *PHOEBE_ld_table;

typedef enum LDLaw {
	LD_LAW_LINEAR = 1,
	LD_LAW_LOG,
	LD_LAW_SQRT,
	LD_LAW_INVALID
} LDLaw;

LDLaw   phoebe_ld_model_type (const char *ldlaw);

char *phoebe_ld_get_vh1993_passband_name (PHOEBE_passband *passband);

LDelem *phoebe_ld_elem_new (double M, int T, double lg);
int phoebe_ld_table_free ();

int read_in_ld_nodes (char *dir);

int phoebe_get_ld_coefficients (LDLaw ldlaw, PHOEBE_passband *passband, double M, int T, double lg, double *x, double *y);

#endif
