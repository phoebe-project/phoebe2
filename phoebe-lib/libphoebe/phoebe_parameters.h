#ifndef PHOEBE_PARAMETERS_H
	#define PHOEBE_PARAMETERS_H 1

#include "phoebe_types.h"

typedef enum PHOEBE_parameter_kind {
	KIND_PARAMETER,
	KIND_MODIFIER,
	KIND_ADJUSTABLE,
	KIND_SWITCH,
	KIND_MENU,
	KIND_COMPUTED
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
	PHOEBE_value                 value;
	char                        *format;
	double                       min;
	double                       max;
	double                       step;
	bool                         tba;
	PHOEBE_value                 defaultvalue;
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
		struct PHOEBE_ast_list *constraints;
	} lists;
} PHOEBE_parameter_table;

typedef struct PHOEBE_parameter_table_list {
	PHOEBE_parameter_table *table;
	struct PHOEBE_parameter_table_list *next;
} PHOEBE_parameter_table_list;

/* A global list of all parameter tables: */
extern PHOEBE_parameter_table_list *PHOEBE_pt_list;

/* A pointer to the currently active parameter table: */
extern PHOEBE_parameter_table *PHOEBE_pt;

/**************************   PARAMETER TABLE   *******************************/

PHOEBE_parameter_table *phoebe_parameter_table_new       ();
PHOEBE_parameter_table *phoebe_parameter_table_duplicate (PHOEBE_parameter_table *table);
int                     phoebe_parameter_table_activate  (PHOEBE_parameter_table *table);
int                     phoebe_parameter_table_print     (PHOEBE_parameter_table *table);
int                     phoebe_parameter_table_free      (PHOEBE_parameter_table *table);

/****************************   PARAMETERS   **********************************/

PHOEBE_parameter *phoebe_parameter_new            ();
int               phoebe_parameter_add            (char *qualifier, char *description, PHOEBE_parameter_kind kind, char *dependency, char *format, double min, double max, double step, bool tba, ...);
unsigned int      phoebe_parameter_hash           (char *qualifier);
PHOEBE_parameter *phoebe_parameter_lookup         (char *qualifier);
int               phoebe_parameter_commit         (PHOEBE_parameter *par);
int               phoebe_parameter_add_option     (PHOEBE_parameter *par, char *option);
int               phoebe_parameter_update_deps    (PHOEBE_parameter *par, int oldval);
int               phoebe_parameter_free           (PHOEBE_parameter *par);

int               phoebe_parameter_option_get_index (PHOEBE_parameter *par, char *option, int *index);
bool              phoebe_parameter_option_is_valid (char *qualifier, char *option);

int               phoebe_init_parameters        ();
int               phoebe_free_parameters        ();

int               phoebe_init_parameter_options ();

int               phoebe_parameters_check_bounds  (char **offender);

bool              phoebe_is_qualifier             (char *qualifier);
int               phoebe_qualifier_string_parse   (char *input, char **qualifier, int *index);
bool              phoebe_qualifier_is_constrained (char *qualifier);

/******************************************************************************/

int phoebe_parameter_get_value  (PHOEBE_parameter *par, ...);
int phoebe_parameter_set_value  (PHOEBE_parameter *par, ...);

int phoebe_parameter_get_tba    (PHOEBE_parameter *par, bool *tba);
int phoebe_parameter_set_tba    (PHOEBE_parameter *par, bool  tba);

int phoebe_parameter_get_step   (PHOEBE_parameter *par, double *step);
int phoebe_parameter_set_step   (PHOEBE_parameter *par, double  step);

int phoebe_parameter_get_min    (PHOEBE_parameter *par, double *valmin);
int phoebe_parameter_set_min    (PHOEBE_parameter *par, double  valmin);

int phoebe_parameter_get_max    (PHOEBE_parameter *par, double *valmax);
int phoebe_parameter_set_max    (PHOEBE_parameter *par, double  valmax);

int phoebe_parameter_get_limits (PHOEBE_parameter *par, double *valmin, double *valmax);
int phoebe_parameter_set_limits (PHOEBE_parameter *par, double  valmin, double  valmax);
bool phoebe_parameter_is_within_limits (PHOEBE_parameter *par);

PHOEBE_parameter_list *phoebe_parameter_list_reverse         (PHOEBE_parameter_list *c, PHOEBE_parameter_list *p);
PHOEBE_parameter_list *phoebe_parameter_list_get_marked_tba  ();
int                    phoebe_parameter_list_sort_marked_tba (PHOEBE_parameter_list *list);

/* ***************************   Third light   ****************************** */

typedef enum PHOEBE_el3_units {
	PHOEBE_EL3_UNITS_TOTAL_LIGHT,
	PHOEBE_EL3_UNITS_FLUX,
	PHOEBE_EL3_UNITS_INVALID_ENTRY
} PHOEBE_el3_units;

int phoebe_el3_units_id (PHOEBE_el3_units *el3_units);

/* ************************************************************************** */

PHOEBE_array *phoebe_active_curves_get (PHOEBE_curve_type type);
int phoebe_active_spots_get (int *active_spots_no, PHOEBE_array **active_spotindices);
double phoebe_spots_units_to_wd_conversion_factor ();

/* ************************************************************************** */

/* Opening and saving keyword files:                                          */

int phoebe_open_parameter_file        (const char *filename);
int phoebe_open_legacy_parameter_file (const char *filename);
int phoebe_save_parameter_file        (const char *filename);

int phoebe_restore_default_parameters ();

int phoebe_parameter_file_import_bm3 (const char *bm3file, const char *datafile);

#endif
