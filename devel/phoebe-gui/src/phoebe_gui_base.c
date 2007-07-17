#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"
#include "phoebe_gui_base.h"

#include "phoebe_gui_treeviews.h"

int phoebe_gui_init ()
{
    /* ************************   Glade XML files   *************************** */

    // g_print("--- Setting up main window ------\n");

    /* The main window */
	GladeXML *phoebe_window_xml = glade_xml_new("../glade/phoebe.glade", NULL, NULL);
	phoebe_window = glade_xml_get_widget(phoebe_window_xml, "phoebe_window");
	glade_xml_signal_autoconnect(phoebe_window_xml);
    gtk_widget_show(phoebe_window);

    // g_print("--- Setting up load-lc window ---\n");

    /* The LC load window */
    GladeXML *phoebe_load_lc_xml = glade_xml_new("../glade/phoebe_load_lc.glade", NULL, NULL);
	phoebe_load_lc_window = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_window");
	phoebe_load_lc_filechooserbutton = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
	glade_xml_signal_autoconnect(phoebe_load_lc_xml);

    // g_print("--- Setting up load-rv window ---\n");

	/* The RV load window */
    GladeXML *phoebe_load_rv_xml = glade_xml_new("../glade/phoebe_load_rv.glade", NULL, NULL);
	phoebe_load_rv_window = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_window");
	phoebe_load_rv_filechooserbutton = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filechooserbutton");
	glade_xml_signal_autoconnect(phoebe_load_rv_xml);

	/* The spots load window */
    GladeXML *phoebe_load_spots_xml = glade_xml_new("../glade/phoebe_load_spots.glade", NULL, NULL);
	phoebe_load_spots_window = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_window");
	glade_xml_signal_autoconnect(phoebe_load_spots_xml);

    gui_init_treeviews(phoebe_window_xml);
	gui_init_widgets (phoebe_window_xml);

	g_object_unref(phoebe_window_xml);
	g_object_unref(phoebe_load_lc_xml);
	g_object_unref(phoebe_load_rv_xml);
	g_object_unref(phoebe_load_spots_xml);
	return SUCCESS;
}

int phoebe_gui_quit ()
{
	gui_free_widgets ();

	return SUCCESS;
}
