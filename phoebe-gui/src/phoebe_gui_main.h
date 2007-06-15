#include <phoebe/phoebe.h>

GtkWidget *phoebe_window;
GtkWidget *phoebe_filechooser_dialog;

/* At the moment, I don't see a better way to obtain this reference from, say, 
 * phoebe_gui_callbacks.c... fix if possible. */
GtkWidget *phoebe_data_lc_tree_view;

/* These columns will appear in the phoebe_data_lc/rv_treeview */
typedef enum curves_view_columns
{
    filename,
    passband,
    itype,
    dtype,
    wtype,
    sigma,
    column_count,
}curves_view_columns;

/* This function will connect the data container (model) to the data view widget (a treeview) */
void connect_curves_view_to_model(GtkWidget*, GtkTreeModel*);

/* Creates a model for storing phoebe_curves data */
GtkTreeModel *create_curves_model(void);
