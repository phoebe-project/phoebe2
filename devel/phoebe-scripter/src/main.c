#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_ast.h"
#include "phoebe_scripter_configuration.h"
#include "phoebe_scripter_core.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter.lng.h"
#include "phoebe_scripter.grm.h"

int main (int argc, char *argv[])
{
	int status;

	phoebe_debug ("starting up PHOEBE scripter:\n");

	phoebe_debug ("  initializing PHOEBE library.\n");

	status = phoebe_init ();
	if (status != SUCCESS)
		phoebe_fatal (phoebe_scripter_error (status));

	phoebe_debug ("  adding config entries.\n");
	scripter_config_populate ();

	phoebe_debug ("  parsing the start-up line.\n");
	parse_startup_line (argc, argv);

	if (PHOEBE_args.CONFIG_DIR != NULL) {
		phoebe_debug ("  found a --config-dir switch.\n");
		free (PHOEBE_HOME_DIR);
		PHOEBE_HOME_DIR = strdup (PHOEBE_args.CONFIG_DIR);
	}

	phoebe_debug ("  configuring PHOEBE...\n");
	phoebe_configure ();

	phoebe_debug ("  initializing the scripter.\n");

	status = scripter_init ();
	if (status != SUCCESS)
		phoebe_fatal (phoebe_scripter_error (status));

	status = scripter_plugins_init ();
	if (status != SUCCESS)
		phoebe_scripter_output (phoebe_scripter_error (status));

	if (PHOEBE_args.CONFIGURE_SWITCH == 1) {
		phoebe_debug ("  found a '-c' switch, proceeding to configuration mode.\n");
		scripter_create_config_file ();
	}

	if (PHOEBE_args.SCRIPT_SWITCH == 1) {
		FILE *in = fopen (PHOEBE_args.SCRIPT_NAME, "r");

		phoebe_debug ("  found a '-e' switch, proceeding to the execute mode.\n");

		if (!in) {
			printf ("\n");
			phoebe_warning ("cannot open script %s, aborting.\n\n", PHOEBE_args.SCRIPT_NAME);
			phoebe_quit ();
		}

		phoebe_debug ("  script to be executed: %s.\n", PHOEBE_args.SCRIPT_NAME);

		phoebe_debug ("  initializing the scripter.\n");
		status = scripter_init ();
		if (status != SUCCESS)
			phoebe_fatal (phoebe_scripter_error (status));

		phoebe_debug ("PHOEBE start-up successful.\n");
		phoebe_debug ("executing the script from the stream.\n");
		scripter_execute_script_from_stream (in);
		fclose (in);

		return SUCCESS;
	}

	if (PHOEBE_args.PARAMETER_SWITCH)
		status = phoebe_open_parameter_file (PHOEBE_args.PARAMETER_FILE);
	if (status != SUCCESS) {
		phoebe_warning (phoebe_scripter_error (status));
		scripter_quit ();
		phoebe_quit ();
	}
	
	phoebe_debug ("  entering the main scripter loop.\n");
	phoebe_debug ("PHOEBE start-up successful.\n");
	scripter_main_loop ();

	return SUCCESS;
}
