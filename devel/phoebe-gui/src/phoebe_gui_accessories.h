#include <gtk/gtk.h>
#include <glade/glade.h>

#include "phoebe_gui_types.h"

void gui_set_text_view_from_file (GtkWidget *text_view, gchar *filename);
GtkWidget *gui_detach_box_from_parent (GtkWidget *box, GtkWidget *parent, gboolean *flag, gchar *window_title, gint x, gint y);
GtkWidget *gui_show_temp_window();
void gui_hide_temp_window(GtkWidget *temp_window, GtkWidget *new_active_window);

int gui_open_parameter_file();
int gui_save_parameter_file();
gchar *gui_get_filename_with_overwrite_confirmation(GtkWidget *dialog, char *gui_confirmation_title);

int gui_show_configuration_dialog();

int gui_question	(char* title, char* message);
int gui_warning		(char* title, char* message);
int gui_notice		(char* title, char* message);
int gui_error		(char* title, char* message);
int gui_status      (const char *format, ...);

void gui_beep();

int gui_update_el3_lum_value ();
