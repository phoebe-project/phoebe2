#ifndef _PHOEBE_CONSTRAINTS_H
#define _PHOEBE_CONSTRAINTS_H 1

#include "phoebe_parameters.h"

typedef enum {
	PHOEBE_NODE_TYPE_CONSTRAINT,
	PHOEBE_NODE_TYPE_PARAMETER,
	PHOEBE_NODE_TYPE_ADD,
	PHOEBE_NODE_TYPE_SUB,
	PHOEBE_NODE_TYPE_MUL,
	PHOEBE_NODE_TYPE_DIV,
	PHOEBE_NODE_TYPE_POT,
	PHOEBE_NODE_TYPE_BUILTIN
} PHOEBE_node_type;

typedef struct PHOEBE_ast {
	enum {
		PHOEBE_AST_INDEX,
		PHOEBE_AST_NUMVAL,
		PHOEBE_AST_STRING,
		PHOEBE_AST_PARAMETER,
		PHOEBE_AST_NODE
	} type;
	union {
		int    idx;
		double numval;
		char  *str;
		struct {
			PHOEBE_node_type        type;
			struct PHOEBE_ast_list *args;
		} node;
	} val;
} PHOEBE_ast;

typedef struct PHOEBE_ast_value {
	enum {
		PHOEBE_AST_VALUE_VOID,
		PHOEBE_AST_VALUE_INT,
		PHOEBE_AST_VALUE_DOUBLE,
		PHOEBE_AST_VALUE_STRING,
		PHOEBE_AST_VALUE_PARAMETER
	} type;
	union {
		int               idx;
		double            numval;
		char             *str;
	} val;
} PHOEBE_ast_value;

typedef struct PHOEBE_ast_list {
	PHOEBE_ast *elem;
	struct PHOEBE_ast_list *next;
} PHOEBE_ast_list;

PHOEBE_ast_list *phoebe_ast_construct_list (PHOEBE_ast *ast, PHOEBE_ast_list *list);
PHOEBE_ast_list *phoebe_ast_list_reverse (PHOEBE_ast_list *c, PHOEBE_ast_list *p);
int phoebe_ast_list_length (PHOEBE_ast_list *list);

PHOEBE_ast *phoebe_ast_add_index     (int idx);
PHOEBE_ast *phoebe_ast_add_numval    (double numval);
PHOEBE_ast *phoebe_ast_add_builtin   (char *builtin);
PHOEBE_ast *phoebe_ast_add_parameter (char *qualifier);
PHOEBE_ast *phoebe_ast_add_node      (PHOEBE_node_type type, PHOEBE_ast_list *args);

PHOEBE_ast *phoebe_ast_duplicate     (PHOEBE_ast *ast);

PHOEBE_ast_value phoebe_ast_evaluate (PHOEBE_ast *ast);

int phoebe_ast_print (int depth, PHOEBE_ast *in);
int phoebe_ast_free (PHOEBE_ast *ast);

int   phoebe_constraint_new (const char *constraint);
char *phoebe_constraint_get_qualifier (PHOEBE_ast *constraint);
int   phoebe_constraint_satisfy_all ();
int   phoebe_free_constraints ();

extern int pcparse (void);

#endif /* _PHOEBE_CONSTRAINTS_H */
