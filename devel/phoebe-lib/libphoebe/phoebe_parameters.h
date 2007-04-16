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

typedef struct PHOEBE_parameter_tag {
	char                     *qualifier;
	char                     *description;
	char                     *dependency;
	PHOEBE_parameter_kind     kind;
	PHOEBE_type               type;
	anytype                   value;
	double                    min;
	double                    max;
	double                    step;
	bool                      tba;
	anytype                   defaultvalue;
	PHOEBE_parameter_options *menu;
	void                     *widget;
} PHOEBE_parameter_tag;

extern PHOEBE_parameter_tag *PHOEBE_parameters;
extern int                   PHOEBE_parameters_no;

/* Functions for manipulating PHOEBE parameters' table:                       */

int declare_parameter              (char *qualifier, char *dependency, char *description, PHOEBE_parameter_kind kind, double min, double max, double step, bool tba, ...);
int declare_all_parameters         ();

int release_parameter_by_index     (int index);
int release_parameter_by_qualifier (char *qualifier);
int release_all_parameters         ();

int add_option_to_parameter_menu   (char *qualifier, char *option);
int add_options_to_all_parameters  ();
int release_all_parameter_options  ();

int update_parameter_arrays        (char *bond, int oldval);

int phoebe_qualifier_from_index       (const char **qualifier, int index);
int phoebe_qualifier_from_description (const char **qualifier, char *description);
int phoebe_description_from_qualifier (const char **description, char *qualifier);
int phoebe_index_from_qualifier       (int *index, char *qualifier);
int phoebe_index_from_description     (int *index, char *description);
int phoebe_kind_from_qualifier        (PHOEBE_parameter_kind *kind, char *qualifier);
int phoebe_type_from_index            (PHOEBE_type *type, int index);
int phoebe_type_from_qualifier        (PHOEBE_type *type, char *qualifier);
int phoebe_type_from_description      (PHOEBE_type *type, char *description);

bool phoebe_parameter_menu_option_is_valid (char *qualifier, char *option);

/*
 * The following functions are internal and are called by the phoebe_get_
 * _parameter_value and phoebe_set_parameter_value functions. Their usage is
 * deprecated and their prototypes are thus intern'ed and commented out.
 *
 * int intern_get_value_int         (int         *value, char *qualifier);
 * int intern_get_value_bool        (bool        *value, char *qualifier);
 * int intern_get_value_double      (double      *value, char *qualifier);
 * int intern_get_value_string      (const char **value, char *qualifier);
 * int intern_get_value_list_int    (int         *value, char *qualifier, int row);
 * int intern_get_value_list_bool   (bool        *value, char *qualifier, int row);
 * int intern_get_value_list_double (double      *value, char *qualifier, int row);
 * int intern_get_value_list_string (const char **value, char *qualifier, int row);
 *
 * int intern_set_value_int         (char *qualifier, int value);
 * int intern_set_value_double      (char *qualifier, double value);
 * int intern_set_value_bool        (char *qualifier, bool value);
 * int intern_set_value_string      (char *qualifier, const char *value);
 * int intern_set_value_list_string (char *qualifier, int row, const char *value);
 * int intern_set_value_list_int    (char *qualifier, int row, int value);
 * int intern_set_value_list_double (char *qualifier, int row, double value);
 * int intern_set_value_list_bool   (char *qualifier, int row, bool value);
 */

int phoebe_get_parameter_value        (char *qualifier, ...);
int phoebe_set_parameter_value        (char *qualifier, ...);

int phoebe_get_parameter_tba          (char *qualifier, bool *tba);
int phoebe_set_parameter_tba          (char *qualifier, bool  tba);

int phoebe_get_parameter_step         (char *qualifier, double *step);
int phoebe_set_parameter_step         (char *qualifier, double  step);

int phoebe_get_parameter_lower_limit  (char *qualifier, double *valmin);
int phoebe_set_parameter_lower_limit  (char *qualifier, double  valmin);

int phoebe_get_parameter_upper_limit  (char *qualifier, double *valmax);
int phoebe_set_parameter_upper_limit  (char *qualifier, double  valmax);

int phoebe_get_parameter_limits       (char *qualifier, double *valmin, double *valmax);
int phoebe_set_parameter_limits       (char *qualifier, double  valmin, double  valmax);

int get_input_independent_variable    (const char *value, PHOEBE_input_indep  *indep);
int get_input_dependent_variable      (const char *value, PHOEBE_input_dep    *dep);
int get_input_weight                  (const char *value, PHOEBE_input_weight *weight);
int get_output_independent_variable   (const char *value, PHOEBE_output_indep *indep);
int get_output_dependent_variable     (const char *value, PHOEBE_output_dep *dep);
int get_output_weight                 (const char *value, PHOEBE_output_weight *weight);

int get_ld_model_id                   (int *ldmodel);

/* ***************************   Third light   ****************************** */

typedef enum PHOEBE_el3_units {
	PHOEBE_EL3_UNITS_TOTAL_LIGHT,
	PHOEBE_EL3_UNITS_FLUX,
	PHOEBE_EL3_UNITS_INVALID_ENTRY
} PHOEBE_el3_units;

int phoebe_el3_units_id (PHOEBE_el3_units *el3_units);

/* ************************************************************************** */

/* Opening and saving keyword files:                                          */

int open_parameter_file        (const char *filename);
int open_legacy_parameter_file (const char *filename);
int save_parameter_file        (const char *filename);

#endif
