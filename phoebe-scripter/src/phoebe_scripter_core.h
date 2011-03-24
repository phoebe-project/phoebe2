#ifndef SCRIPTER_MAIN_LOOP_H
	#define SCRIPTER_MAIN_LOOP_H 1

#include <stdio.h>
#include <phoebe/phoebe.h>
#include "phoebe_scripter.lng.h"

extern YY_BUFFER_STATE main_thread;

typedef struct PHOEBE_COMMAND_LINE_ARGS {
	int PARAMETER_SWITCH;
	char *PARAMETER_FILE;
	int SCRIPT_SWITCH;
	char *SCRIPT_NAME;
	int CONFIGURE_SWITCH;
	char *CONFIG_DIR;
} PHOEBE_COMMAND_LINE_ARGS;

extern PHOEBE_COMMAND_LINE_ARGS PHOEBE_args;

int parse_startup_line (int argc, char *argv[]);

int scripter_init ();
int scripter_config_populate ();
int scripter_parameters_init ();
int scripter_plugins_init ();
int scripter_main_loop ();
int scripter_execute_script_from_stream (FILE *stream);
int scripter_execute_script_from_buffer (char *buffer);
int scripter_quit ();

#endif
