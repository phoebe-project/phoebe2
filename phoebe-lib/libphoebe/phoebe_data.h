#ifndef PHOEBE_DATA_H
	#define PHOEBE_DATA_H 1

#include "phoebe_types.h"

/* Defined in phoebe_types.h:

typedef struct PHOEBE_passband {
	int          id;
	char        *set;
	char        *name;
	double       effwl;
	PHOEBE_hist *tf;
} PHOEBE_passband;
*/

extern PHOEBE_passband **PHOEBE_passbands;
extern int               PHOEBE_passbands_no;

PHOEBE_passband *phoebe_passband_new            ();
PHOEBE_passband *phoebe_passband_new_from_file  (char *filename);
PHOEBE_passband *phoebe_passband_lookup         (const char *name);
PHOEBE_passband *phoebe_passband_lookup_by_id   (const char *id);
int              phoebe_passband_free           (PHOEBE_passband *passband);

int              phoebe_read_in_passbands       (char *dir_name);
int              phoebe_free_passbands          ();

#endif
