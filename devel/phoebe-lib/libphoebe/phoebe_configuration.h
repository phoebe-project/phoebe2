#ifndef PHOEBE_CONFIGURATION_H
	#define PHOEBE_CONFIGURATION_H 1

#include "phoebe_types.h"

typedef struct PHOEBE_config_entry {
	PHOEBE_type type;
	char       *keyword;
	PHOEBE_value    *value;
	PHOEBE_value    *defval;
} PHOEBE_config_entry;

extern PHOEBE_config_entry **PHOEBE_config_table;
extern int                   PHOEBE_config_table_size;

PHOEBE_config_entry *phoebe_config_entry_new  ();
int                  phoebe_config_entry_add  (PHOEBE_type type, char *keyword, ...);
int                  phoebe_config_entry_get  (char *keyword, ...);
int                  phoebe_config_entry_set  (char *keyword, ...);
int                  phoebe_config_entry_free (PHOEBE_config_entry *entry);

int                  phoebe_config_populate   ();
int                  phoebe_config_free       ();

int                  phoebe_configure         ();

int                  phoebe_config_peek       (char *filename);
int                  phoebe_config_load       (char *filename);
int                  phoebe_config_save       (char *filename);
int                  phoebe_config_import     (char *filename);

#endif
