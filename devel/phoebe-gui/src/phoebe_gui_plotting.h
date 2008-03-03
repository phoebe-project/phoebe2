#include <gtk/gtk.h>
#include <glade/glade.h>

int gui_plot_lc_using_gnuplot (gdouble x_offset, gdouble y_offset, gdouble zoom);
int gui_plot_rv_using_gnuplot (gdouble x_offset, gdouble y_offset, gdouble zoom);
int gui_plot_eb_using_gnuplot ();
int gui_plot_lc_to_ascii 	  (gchar *filename);
int gui_plot_rv_to_ascii 	  (gchar *filename);
