#include <stdlib.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"
#include "phoebe_gui_base.h"
#include "phoebe_gui_global.h"

#include "phoebe_gui_treeviews.h"

gchar *PHOEBE_GLADE_XML_DIR;
gchar *PHOEBE_GLADE_PIXMAP_DIR;

int phoebe_gui_init ()
{
	/*
	 * Initialize a global PHOEBE_GLADE_XML_DIR string to point to a valid
	 * directory. Try the global installation directory first (contained in
	 * the GLADE_XML_DIR shell variable), then handle the invocation from src/,
	 * and finally the invocation from the top-level phoebe-gui/ dir. If
	 * all fails, print the error and exit.
	 */

	gchar *glade_xml_file;
	gchar *glade_pixmap_file;

	PHOEBE_GLADE_XML_DIR = g_strdup (GLADE_XML_DIR);
	glade_xml_file = g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe.glade", NULL);

	if (!g_file_test (glade_xml_file, G_FILE_TEST_EXISTS)) {
		g_free (PHOEBE_GLADE_XML_DIR); g_free (glade_xml_file);
		PHOEBE_GLADE_XML_DIR = g_build_filename ("../glade", NULL);
		glade_xml_file = g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe.glade", NULL);
		if (!g_file_test (glade_xml_file, G_FILE_TEST_EXISTS)) {
			g_free (PHOEBE_GLADE_XML_DIR); g_free (glade_xml_file);
			PHOEBE_GLADE_XML_DIR = g_build_filename ("glade", NULL);
			glade_xml_file = g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe.glade", NULL);
			if (!g_file_test (glade_xml_file, G_FILE_TEST_EXISTS)) {
#warning DO_PROPER_ERROR_HANDLING_HERE
				printf ("*** Glade files cannot be found, aborting.\n");
				exit (-1);
			}
		}
	}

	/*
	 * Initialize a global PHOEBE_GLADE_PIXMAP_DIR string to point to a valid
	 * directory. Try the global installation directory first (contained in
	 * the GLADE_PIXMAP_DIR shell variable), then handle the invocation from
	 * src/, and finally the invocation from the top-level phoebe-gui/ dir. If
	 * all fails, print the error and exit.
	 */

	PHOEBE_GLADE_PIXMAP_DIR = g_strdup (GLADE_PIXMAP_DIR);
	glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	if (!g_file_test (glade_pixmap_file, G_FILE_TEST_EXISTS)) {
		g_free (PHOEBE_GLADE_PIXMAP_DIR); g_free (glade_pixmap_file);
		PHOEBE_GLADE_PIXMAP_DIR = g_build_filename ("../pixmaps", NULL);
		glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
		if (!g_file_test (glade_pixmap_file, G_FILE_TEST_EXISTS)) {
			g_free (PHOEBE_GLADE_PIXMAP_DIR); g_free (glade_pixmap_file);
			PHOEBE_GLADE_PIXMAP_DIR = g_build_filename ("pixmaps", NULL);
			glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
			if (!g_file_test (glade_pixmap_file, G_FILE_TEST_EXISTS)) {
#warning DO_PROPER_ERROR_HANDLING_HERE
				printf ("*** Pixmaps files cannot be found, aborting.\n");
				exit (-1);
			}
		}
	}

	gui_init_widgets ();

	return SUCCESS;
}

int phoebe_gui_quit ()
{
	gui_free_widgets ();

	return SUCCESS;
}
