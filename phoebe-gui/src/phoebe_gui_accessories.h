#include <gtk/gtk.h>
#include <glade/glade.h>

#include "phoebe_gui_types.h"

void gui_set_text_view_from_file (GtkWidget *text_view, gchar *filename);
void gui_detach_box_from_parent (GtkWidget *box, GtkWidget *parent, gboolean *flag, gchar *window_title, gint x, gint y);

int gui_open_parameter_file();
int gui_save_parameter_file();

int gui_show_configuration_dialog();
