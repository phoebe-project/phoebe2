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
extern char *PHOEBE_PLOTTING_PACKAGE;

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

#endif
