#ifndef PHOEBE_GLOBAL_H
	#define PHOEBE_GLOBAL_H 1

#include "phoebe_types.h"

#include <stdarg.h>

/* ************************************************************************** */
/*                        Global PHOEBE setup strings:                        */
/* ************************************************************************** */

extern char *USER_HOME_DIR;

extern char *PHOEBE_VERSION_NUMBER;
extern char *PHOEBE_VERSION_DATE;
extern char *PHOEBE_PARAMETERS_FILENAME;
extern char *PHOEBE_DEFAULTS;

extern int   PHOEBE_CONFIG_EXISTS;
extern char *PHOEBE_STARTUP_DIR;
extern char *PHOEBE_HOME_DIR;
extern char *PHOEBE_CONFIG;
extern char *PHOEBE_BASE_DIR;
extern char *PHOEBE_SOURCE_DIR;
extern char *PHOEBE_DEFAULTS_DIR;
extern char *PHOEBE_TEMP_DIR;
extern char *PHOEBE_DATA_DIR;
extern char *PHOEBE_PTF_DIR;
extern char *PHOEBE_PLOTTING_PACKAGE;
extern int   PHOEBE_LD_SWITCH;
extern char *PHOEBE_LD_DIR;
extern int   PHOEBE_KURUCZ_SWITCH;
extern char *PHOEBE_KURUCZ_DIR;

extern char *PHOEBE_INPUT_LOCALE;

extern bool PHOEBE_INTERRUPT;

/* ************************************************************************** */
/*                    Global PHOEBE configuration options:                    */
/* ************************************************************************** */

extern int PHOEBE_3D_PLOT_CALLBACK_OPTION;
extern int PHOEBE_CONFIRM_ON_SAVE;
extern int PHOEBE_CONFIRM_ON_QUIT;
extern int PHOEBE_WARN_ON_SYNTHETIC_SCATTER;

#define PHOEBE_NUMERICAL_ACCURACY 1E-10

/*
typedef struct PHOEBE_lc_plot_parameters
	{
	bool                 alias;
	bool                 bin;
	bool                 deredden;
	bool                 synthderedden;
	bool                 residuals;
	bool                *synthetic;
	bool                *experimental;
	double              *filter;
	double              *shift;
	PHOEBE_output_indep  indep;
	PHOEBE_output_dep    dep;
	PHOEBE_output_weight weight;
	double               phstart;
	double               phend;
	double               vertices;
	} PHOEBE_lc_plot_parameters;
*/

#endif
