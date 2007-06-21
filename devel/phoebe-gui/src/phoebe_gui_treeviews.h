#include <gtk/gtk.h>
#include <glade/glade.h>

/* The treeviews */
GtkWidget *phoebe_data_lc_treeview;
GtkWidget *phoebe_data_rv_treeview;

/* These columns will appear in the phoebe_data_lc/rv_treeview */
typedef enum GUI_curvelist_columns
{
    CURVELIST_COL_FILENAME,
    CURVELIST_COL_FILTER,
    CURVELIST_COL_ITYPE,
    CURVELIST_COL_DTYPE,
    CURVELIST_COL_WTYPE,
    CURVELIST_COL_SIGMA,
    CURVELIST_COL_COUNT,
}GUI_curvelist_columns;

/* Initializes the treeviews */
int gui_init_treeviews(GladeXML *phoebe_window);

/* This function will connect the data container (model) to the data view widget (a treeview) */
void intern_connect_curves_view_to_model(GtkWidget*, GtkTreeModel*);

/* Creates a model for storing phoebe_curves data */
GtkTreeModel *intern_create_curves_model(void);
