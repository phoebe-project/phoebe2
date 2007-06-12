#ifndef PHOEBE_GUI_TYPES
#define PHOEBE_GUI_TYPES 1

#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

typedef enum GUI_widget_type {
	GUI_WIDGET_VALUE,
	GUI_WIDGET_VALUE_MIN,
	GUI_WIDGET_VALUE_MAX,
	GUI_WIDGET_VALUE_STEP,
	GUI_WIDGET_SWITCH_TBA
} GUI_widget_type;

typedef struct GUI_widget {
	char             *name;
	GUI_widget_type   type;
	PHOEBE_parameter *par;
	GtkWidget        *gtk;
} GUI_widget;

/* 
 * Create a hashed GUI_widget_table here (see phoebe_parameters.c for
 * reference) and, instead of a complicated table structure, just something
 * like GUI_widget_table *GUI_wt.
 */

#endif
