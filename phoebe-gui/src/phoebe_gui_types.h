#ifndef PHOEBE_GUI_TYPES
	#define PHOEBE_GUI_TYPES 1

#include <gtk/gtk.h>
#include <glade/glade.h>
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

GUI_widget 	   *gui_widget_new 		();
int 			gui_widget_add 		(char *name, GtkWidget *gtk, GUI_widget_type type, PHOEBE_parameter *par);
unsigned int	gui_widget_hash 	(char *name);
int 			gui_widget_hookup 	(GUI_widget *widget, GtkWidget *gtk, GUI_widget_type type, char *name, PHOEBE_parameter *par);
int 			gui_widget_commit 	(GUI_widget *widget);
int 			gui_widget_free     (GUI_widget *widget);

int				gui_init_widgets	(GladeXML *phoebe_window);
int 			gui_free_widgets 	();

/***************************   WIDGET TABLE   ********************************/

enum {
	/* Do we need more than 103 buckets? */

	GUI_WT_HASH_MULTIPLIER = 31, /*113*/
	GUI_WT_HASH_BUCKETS    = 103 /*337*/
};

typedef struct GUI_wt_bucket {
	GUI_widget				*widget;
	struct GUI_wt_bucket *next;
} GUI_wt_bucket;

typedef struct GUI_widget_table {
	GUI_wt_bucket *elem[GUI_WT_HASH_BUCKETS];
} GUI_widget_table;

GUI_widget_table *GUI_wt;

#endif
