#include <gtk/gtk.h>
#include <glade/glade.h>

void set_text_view_from_file (GtkWidget *text_view, gchar *filename);
void detach_box_from_parent (GtkWidget *box, GtkWidget *parent, gboolean *flag, gchar *window_title, gint x, gint y);

