#ifndef PHOEBE_GUI_BASE_H
	#define PHOEBE_GUI_BASE_H 1

int phoebe_gui_init ();
int phoebe_gui_quit ();

#endif

#include <gtk/gtk.h>

GtkWidget *phoebe_window;
GtkWidget *phoebe_load_lc_window;
GtkWidget *phoebe_load_lc_filechooserbutton;
GtkWidget *phoebe_load_rv_window;
