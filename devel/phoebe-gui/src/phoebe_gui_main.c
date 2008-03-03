#ifdef HAVE_CONFIG_H
#  include "phoebe_gui_build_config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_base.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_error_handling.h"
#include "phoebe_gui_main.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_types.h"

int parse_startup_line (int argc, char *argv[])
{
	/*
	 * This function parses the command line and looks for known switches.
	 */

	int i, status;

	for (i = 1; i < argc; i++) {
		if ( (strcmp (argv[i],  "-h"   ) == 0) ||
		     (strcmp (argv[i],  "-?"   ) == 0) ||
		     (strcmp (argv[i], "--help") == 0) ) {
			printf ("\n%s command line arguments: [-hv] [parameter_file]\n\n", PHOEBE_GUI_RELEASE_NAME);
			printf ("  -h, --help, -?      ..  this help screen\n");
			printf ("  -v, --version       ..  display PHOEBE version and exit\n");
			printf ("\n");
			phoebe_quit ();
		}

		if ( (strcmp (argv[i],  "-v"      ) == 0) ||
			 (strcmp (argv[i], "--version") == 0) ) {
			printf ("\n%s, %s\n", PHOEBE_GUI_RELEASE_NAME, PHOEBE_GUI_RELEASE_DATE);
			printf ("  Send comments and/or requests to phoebe-discuss@lists.sourceforge.net\n\n");
			phoebe_quit ();
		}

		if ( argv[i][0] != '-' ) {
			/*
			 * This means that the command line argument doesn't contain '-';
			 * thus it is a parameter file.
			 */

			status = phoebe_open_parameter_file (argv[i]);
			if (status != SUCCESS)
				phoebe_gui_output ("%s", phoebe_error (status));
			else {
				gui_reinit_treeviews ();
				gui_set_values_to_widgets ();
				PHOEBE_FILEFLAG = TRUE;
				PHOEBE_FILENAME = strdup (argv[i]);
			}
		}
	}

	return SUCCESS;
}

int main (int argc, char *argv[])
{
	int status;

	gtk_set_locale ();
	gtk_init (&argc, &argv);
	glade_init ();

	status = phoebe_init ();
	if (status != SUCCESS) {
		printf ("%s", phoebe_gui_error (status));
		exit (0);
	}

	/* Add all GUI-related options here: */
	phoebe_config_entry_add (TYPE_BOOL, "GUI_CONFIRM_ON_OVERWRITE", TRUE);

	phoebe_configure ();

	phoebe_gui_init ();
	parse_startup_line (argc, argv);

	gtk_main ();
	phoebe_gui_quit ();

	phoebe_quit ();

	return SUCCESS;
}
