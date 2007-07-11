#ifndef PHOEBE_PARAMETERS_H
	#define PHOEBE_PARAMETERS_H 1

#include "phoebe_types.h"

typedef enum PHOEBE_parameter_kind {
	KIND_PARAMETER,
	KIND_MODIFIER,
	KIND_ADJUSTABLE,
	KIND_SWITCH,
	KIND_MENU
} PHOEBE_parameter_kind;

typedef struct PHOEBE_parameter_options {
	int                      optno;
	char                   **option;
} PHOEBE_parameter_options;

typedef struct PHOEBE_parameter_list {
	struct PHOEBE_parameter *par;
	struct PHOEBE_parameter_list *next;
} PHOEBE_parameter_list;

typedef struct PHOEBE_parameter
{
	char                        *qualifier;
	char                        *description;
	PHOEBE_parameter_kind        kind;
	PHOEBE_type                  type;
	anytype                      value;
	double                       min;
	double                       max;
	double                       step;
	bool                         tba;
	anytype                      defaultvalue;
	PHOEBE_parameter_options    *menu;
	PHOEBE_parameter_list       *deps;
} PHOEBE_parameter;

enum {
	PHOEBE_PT_HASH_MULTIPLIER = 31,
	PHOEBE_PT_HASH_BUCKETS    = 103
};

typedef struct PHOEBE_parameter_table {
	PHOEBE_parameter_list *bucket[PHOEBE_PT_HASH_BUCKETS];
	struct {
		PHOEBE_parameter_list *marked_tba;
	} lists;
} PHOEBE_parameter_table;

typedef struct PHOEBE_parameter_table_list {
	PHOEBE_parameter_table *table;
	struct PHOEBE_parameter_table_list *next;
} PHOEBE_parameter_table_list;

/* A global list of all parameter tables: */
PHOEBE_parameter_table_list *PHOEBE_pt_list;

/* A pointer to the currently active parameter table: */
PHOEBE_parameter_table *PHOEBE_pt;

/**************************   PARAMETER TABLE   *******************************/

PHOEBE_parameter_table *phoebe_parameter_table_new       ();
PHOEBE_parameter_table *phoebe_parameter_table_duplicate (PHOEBE_parameter_table *table);
int                     phoebe_parameter_table_activate  (PHOEBE_parameter_table *table);
int                     phoebe_parameter_table_print     (PHOEBE_parameter_table *table);
int                     phoebe_parameter_table_free      (PHOEBE_parameter_table *table);

/****************************   PARAMETERS   **********************************/

PHOEBE_parameter *phoebe_parameter_new          ();
int               phoebe_parameter_add          (char *qualifier, char *description, PHOEBE_parameter_kind kind, char *dependency, double min, double max, double step, bool tba, ...);
unsigned int      phoebe_parameter_hash         (char *qualifier);
PHOEBE_parameter *phoebe_parameter_lookup       (char *qualifier);
int               phoebe_parameter_commit       (PHOEBE_parameter *par);
int               phoebe_parameter_add_option   (PHOEBE_parameter *par, char *option);
int               phoebe_parameter_update_deps  (PHOEBE_parameter *par, int oldval);
int               phoebe_parameter_free         (PHOEBE_parameter *par);

bool              phoebe_parameter_menu_option_is_valid (char *qualifier, char *option);

int               phoebe_init_parameters        ();
int               phoebe_free_parameters        ();

int               phoebe_init_parameter_options ();

/******************************************************************************/

int phoebe_parameter_get_value        (PHOEBE_parameter *par, ...);
int phoebe_parameter_set_value        (PHOEBE_parameter *par, ...);

int phoebe_parameter_get_tba          (PHOEBE_parameter *par, bool *tba);
int phoebe_parameter_set_tba          (PHOEBE_parameter *par, bool  tba);

int phoebe_parameter_get_step         (PHOEBE_parameter *par, double *step);
int phoebe_parameter_set_step         (PHOEBE_parameter *par, double  step);

int phoebe_parameter_get_lower_limit  (PHOEBE_parameter *par, double *valmin);
int phoebe_parameter_set_lower_limit  (PHOEBE_parameter *par, double  valmin);

int phoebe_parameter_get_upper_limit  (PHOEBE_parameter *par, double *valmax);
int phoebe_parameter_set_upper_limit  (PHOEBE_parameter *par, double  valmax);

int phoebe_parameter_get_limits       (PHOEBE_parameter *par, double *valmin, double *valmax);
int phoebe_parameter_set_limits       (PHOEBE_parameter *par, double  valmin, double  valmax);

PHOEBE_parameter_list *phoebe_parameter_list_get_marked_tba  ();
int                    phoebe_parameter_list_sort_marked_tba ();

/* ***************************   Third light   ****************************** */

typedef enum PHOEBE_el3_units {
	PHOEBE_EL3_UNITS_TOTAL_LIGHT,
	PHOEBE_EL3_UNITS_FLUX,
	PHOEBE_EL3_UNITS_INVALID_ENTRY
} PHOEBE_el3_units;

int phoebe_el3_units_id (PHOEBE_el3_units *el3_units);

/* ************************************************************************** */

/* Opening and saving keyword files:                                          */

int phoebe_open_parameter_file        (const char *filename);
int phoebe_open_legacy_parameter_file (const char *filename);
int phoebe_save_parameter_file        (const char *filename);

#endif
