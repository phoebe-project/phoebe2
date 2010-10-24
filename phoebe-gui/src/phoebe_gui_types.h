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
	char             	*name;		/* Widget qualifier                         	*/
	GUI_widget_type   	 type;      /* Widget type; do we really need this?     	*/
	PHOEBE_parameter 	*par;		/* Link to the parameter table              	*/
	GtkWidget        	*gtk;		/* Pointer to the widget                    	*/
	int               	 aux;       /* Auxiliary data, such as the column index 	*/
	struct GUI_widget	*dep;		/* A GUI_widget that this instance depends on	*/
} GUI_widget;

GUI_widget		*gui_widget_new		();
int 			 gui_widget_add		(char *name, GtkWidget *gtk, int aux, GUI_widget_type type, PHOEBE_parameter *par, GUI_widget *dep);
unsigned int	 gui_widget_hash 	(char *name);
int 			 gui_widget_hookup 	(GUI_widget *widget, GtkWidget *gtk, int aux, GUI_widget_type type, char *name, PHOEBE_parameter *par, GUI_widget *dep);
GUI_widget		*gui_widget_lookup 	(char *name);
int 			 gui_widget_commit 	(GUI_widget *widget);
int 			 gui_widget_free	(GUI_widget *widget);

int gui_init_parameter_options		();
int gui_init_combo_boxes			();

int				 gui_init_widgets	();
int 			 gui_free_widgets	();

int              gui_init_angle_widgets       ();
int              gui_update_angle_values      ();
int              gui_export_values_to_radians ();

int				 gui_get_value_from_widget		(GUI_widget *widget);
int 		     gui_set_value_to_widget		(GUI_widget *widget);
int 			 gui_get_values_from_widgets	();
int				 gui_set_values_to_widgets		();

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
	GUI_wt_bucket *bucket[GUI_WT_HASH_BUCKETS];
} GUI_widget_table;

GUI_widget_table *GUI_wt;

void gui_toggle_sensitive_widgets_for_minimization (bool enable);

#endif
