#ifndef PHOEBE_SCRIPTER_AST_H
	#define PHOEBE_SCRIPTER_AST_H 1

#include <stdio.h>
#include <phoebe/phoebe_types.h>
#include <phoebe/phoebe_spectra.h>

typedef enum {
	kind_add,
	kind_and,
	kind_array,
	kind_block,
	kind_builtin,
	kind_calc,
	kind_clear,
	kind_command,
	kind_datastat,
	kind_dec,
	kind_decby,
	kind_define,
	kind_div,
	kind_divby,
	kind_element,
	kind_equal,
	kind_execute,
	kind_execute_macro,
	kind_export,
	kind_field,
	kind_for,
	kind_func,
	kind_gequal,
	kind_greater,
	kind_help,
	kind_idiv,
	kind_if,
	kind_ignore,
	kind_inc,
	kind_incby,
	kind_info,
	kind_lequal,
	kind_less,
	kind_lexerror,
	kind_list,
	kind_macro,
	kind_matrix_element,
	kind_mul,
	kind_multby,
	kind_nequal,
	kind_not,
	kind_or,
	kind_os_cd,
	kind_os_ls,
	kind_os_pwd,
	kind_raise,
	kind_print,
	kind_qual_value,
	kind_quit,
	kind_return,
	kind_set,
	kind_set_element,
	kind_set_matrix_element,
	kind_show,
	kind_statement,
	kind_stdump,
	kind_sub,
	kind_system_call,
	kind_testme,
	kind_unarym,
	kind_unaryp,
	kind_unset,
	kind_while
} scripter_ast_kind;

typedef struct scripter_ast {
	enum {
		ast_int,
		ast_bool,
		ast_double,
		ast_string,
		ast_vector,
		ast_array,
	  	ast_curve,
        ast_spectrum,
		ast_variable,
		ast_qualifier,
		ast_function,
	  	ast_minfeedback,
		ast_node
	} type;
	union {
		int                        integer;
		double                     real;
		bool                       boolean;
		char                      *string;
		PHOEBE_vector             *vec;
		PHOEBE_array              *array;
	  	PHOEBE_curve              *curve;
        PHOEBE_spectrum           *spectrum;
		char                      *variable;
		char                      *qualifier;
		char                      *function;
	  	PHOEBE_minimizer_feedback *minfeedback;
		struct {
			scripter_ast_kind         kind;
			struct scripter_ast_list *args;
		} node;
	} value;
} scripter_ast;

typedef struct scripter_ast_list { 
	scripter_ast *elem;
	struct scripter_ast_list *next;
} scripter_ast_list;

typedef struct scripter_ast_value {
	enum {
		type_int,
		type_bool,
		type_double,
		type_string,
		type_vector,
		type_array,
		type_curve,
		type_spectrum,
		type_qualifier,
		type_function,
		type_minfeedback,
		type_any,
		type_void
	} type;
	PHOEBE_value value;
} scripter_ast_value;

char              *scripter_ast_kind_name        (scripter_ast_kind kind);

scripter_ast      *scripter_ast_add_int          (const int val);
scripter_ast      *scripter_ast_add_double       (const double val);
scripter_ast      *scripter_ast_add_bool         (const bool val);
scripter_ast      *scripter_ast_add_string       (const char *val);
scripter_ast      *scripter_ast_add_array        (PHOEBE_array *array);
scripter_ast      *scripter_ast_add_vector       (PHOEBE_vector *vec);
scripter_ast      *scripter_ast_add_curve        (PHOEBE_curve *curve);
scripter_ast      *scripter_ast_add_spectrum     (PHOEBE_spectrum *spectrum);
scripter_ast      *scripter_ast_add_literal      (const char *val);
scripter_ast      *scripter_ast_add_variable     (const char *val);
scripter_ast      *scripter_ast_add_qualifier    (const char *val);
scripter_ast      *scripter_ast_add_function     (const char *val);
scripter_ast      *scripter_ast_add_minfeedback  (PHOEBE_minimizer_feedback *feedback);
scripter_ast      *scripter_ast_add_node         (const scripter_ast_kind kind, scripter_ast_list *args);

scripter_ast      *scripter_ast_duplicate        (scripter_ast *in);

scripter_ast_list *scripter_ast_construct_list   (scripter_ast *ast, scripter_ast_list *list);
scripter_ast_list *scripter_ast_reverse_list     (scripter_ast_list *r, scripter_ast_list *s);
int                scripter_ast_list_length      (scripter_ast_list *in);

scripter_ast_value scripter_ast_evaluate         (scripter_ast *in);
int                scripter_ast_free             (scripter_ast *ast);
int                scripter_ast_list_free        (scripter_ast_list *list);

int                scripter_ast_value_print      (scripter_ast_value val);

int                scripter_ast_print            (int depth, scripter_ast *in);

int                scripter_ast_get_type         (scripter_ast *in);

char *scripter_ast_value_type_get_name (int type);

/* *********  This part of the code describes auxiliary functions:  ********* */

PHOEBE_vector *phoebe_vector_new_from_list (scripter_ast_list *list);
PHOEBE_array *phoebe_array_new_from_list (scripter_ast_list *list);

#endif
