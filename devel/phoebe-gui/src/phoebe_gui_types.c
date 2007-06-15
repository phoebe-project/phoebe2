#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"

int gui_init_widgets ()
{
	return SUCCESS;
}

GUI_widget *gui_widget_new ()
{
	GUI_widget *widget = phoebe_malloc (sizeof (*widget));

	widget->name = NULL;
	widget->type = 0;
	widget->gtk  = NULL;
	widget->par  = NULL;

	return SUCCESS;
}

int gui_widget_free (GUI_widget *widget)
{
	if (!widget)
		return SUCCESS;

	if (widget->name) free (widget->name);
	free (widget);

	return SUCCESS;
}

int gui_widget_hookup (GUI_widget *widget, GtkWidget *gtk, GUI_widget_type type, char *name, PHOEBE_parameter *par)
{
	if (!widget)
	/*
	 * A suggestion: create a phoebe_gui_errors.h and create an enum for these
	 * error codes.
	 */
		return /* ERROR_GUI_WIDGET_NOT_FOUND; */ -1;

	widget->name = strdup (name);
	widget->gtk  = gtk;
	widget->type = type;
	widget->par  = par;

	return SUCCESS;
}

/* ADD HASH TABLE WRAPPERS HERE */

//
// In gui_widget_add (...) you'd then have the calls to this function, sth like:
//
//   GUI_widget *widget = gui_widget_new ();
//   gui_widget_hookup (widget, gtk, GUI_WIDGET_VALUE, "widget_name", phoebe_parameter_lookup ("parameter_name"));
//   gui_widget_commit (widget); <-- this plugs the widget to the hashed table
//
// In phoebe_gui_init () you'd then have:
//
//   gui_widget_add (/* name = */ "gui_widget_name", /* gtk = */ glade_xml_lookup ("gui_widget_name"), /* type = */ GUI_WIDGET_VALUE, /* par = */ phoebe_parameter_lookup ("phoebe_qualifier"));
//
// You could do some error handling if you wish, i.e. if gtk or par are null etc.
//

int gui_widget_add (char *name, GtkWidget *gtk, GUI_widget_type type, PHOEBE_parameter *par)
{
	GUI_widget *widget = gui_widget_new();	
	
	if (!gtk)
		return -1;

	if (!par)
		return -1;

	gui_widget_hookup (widget, gtk, type, name, par);
	gui_widget_commit(widget);

	return SUCCESS;
}

unsigned int gui_widget_hash (char *name)
{
	/*
	 * This is the hashing function for storing widgets into the widget
	 * table.
	 */

	unsigned int h = 0;
	unsigned char *w;

	for (w = (unsigned char *) name; *w != '\0'; w++)
		h = GUI_WT_HASH_MULTIPLIER * h + *w;

	return h % GUI_WT_HASH_BUCKETS;
}

int gui_widget_commit (GUI_widget *widget)
{
	int hash = gui_widget_hash (widget->name);
	GUI_wt_bucket *elem = GUI_wt->elem[hash];

	while (elem) {
		if (strcmp (elem->widget->name, widget->name) == 0) break;
		elem = elem->next;
	}

	if (elem) {
		/* gui_error(widget already commited...) */
		return SUCCESS;
	}

	else{
		elem = phoebe_malloc (sizeof (*elem));

		elem->widget = widget;
		elem->next	 = GUI_wt->elem[hash];
		GUI_wt->elem[hash] = elem;
	}

	return SUCCESS;
}

int gui_free_widgets ()
{
	/*
	 * This function frees all widgets from the widget table.
	 */

	int i;
	GUI_wt_bucket *elem;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		while (GUI_wt->elem[i]) {
			elem = GUI_wt->elem[i];
			GUI_wt->elem[i] = elem->next;
			gui_widget_free (elem->widget);
			free (elem);
		}
	}
	free (GUI_wt);

	return SUCCESS;
}
