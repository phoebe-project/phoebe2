#ifndef PHOEBE_CONFIGURATION_H
	#define PHOEBE_CONFIGURATION_H 1

#include "phoebe_types.h"

typedef struct PHOEBE_config_entry {
	PHOEBE_type type;
	char       *keyword;
	anytype    *value;
	anytype    *defval;
} PHOEBE_config_entry;

PHOEBE_config_entry **PHOEBE_config_table;
int                   PHOEBE_config_table_size;

PHOEBE_config_entry *phoebe_config_entry_new  ();
int                  phoebe_config_entry_add  (PHOEBE_type type, char *keyword, ...);
int                  phoebe_config_entry_get  (char *keyword, ...);
int                  phoebe_config_entry_set  (char *keyword, ...);
int                  phoebe_config_entry_free (PHOEBE_config_entry *entry);

int                  phoebe_init_config_entries ();
int                  phoebe_free_config_entries ();

int phoebe_config_load   (char *filename);
int phoebe_config_save   (char *filename);
int phoebe_config_import ();
int phoebe_config_check  ();

#endif
