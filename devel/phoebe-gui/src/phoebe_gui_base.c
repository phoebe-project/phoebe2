#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"
#include "phoebe_gui_base.h"

#include "phoebe_gui_treeviews.h"

int phoebe_gui_init ()
{
    /* ************************   Glade XML files   *************************** */

    /* The main window */
	GladeXML *phoebe_window = glade_xml_new("phoebe.glade", NULL, NULL);
	phoebe_window_widget = glade_xml_get_widget(phoebe_window, "phoebe_window");
	glade_xml_signal_autoconnect(phoebe_window);
    gtk_widget_show(phoebe_window_widget);

	/* The filechooser dialog */
	GladeXML *phoebe_filechooser_dialog = glade_xml_new("phoebe_filechooser.glade", NULL, NULL);
	phoebe_filechooser_dialog_widget = glade_xml_get_widget(phoebe_filechooser_dialog, "phoebe_filechooser_dialog");


    gui_init_treeviews(phoebe_window);
	gui_init_widgets (phoebe_window);


	g_object_unref(phoebe_window);
	g_object_unref(phoebe_filechooser_dialog);

	return SUCCESS;
}

int phoebe_gui_quit ()
{
	gui_free_widgets ();

	return SUCCESS;
}
