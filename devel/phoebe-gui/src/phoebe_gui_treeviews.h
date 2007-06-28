#include <gtk/gtk.h>
#include <glade/glade.h>

/* The treeviews */
GtkWidget *phoebe_data_lc_treeview;
GtkWidget *phoebe_para_lc_levels_treeview;
GtkWidget *phoebe_para_lc_el3_treeview;
GtkWidget *phoebe_para_lc_levweight_treeview;
GtkWidget *phoebe_para_lc_ld_treeview;

/* These columns make up the curve model for various treeviews */
typedef enum lc_model_columns
{
    LC_COL_ACTIVE,
    LC_COL_FILENAME,
    LC_COL_FILTER,
    LC_COL_ITYPE,
    LC_COL_DTYPE,
    LC_COL_WTYPE,
    LC_COL_SIGMA,
    LC_COL_LEVWEIGHT,
    LC_COL_HLA,
    LC_COL_CLA,
    LC_COL_OPSF,
    LC_COL_EL3,
    LC_COL_EXTINCTION,
    LC_COL_LCX1,
    LC_COL_LCX2,
    LC_COL_LCY1,
    LC_COL_LCY2,
    LC_COL_COUNT,
}lc_model_columns;

/* Creates a model for storing phoebe_curves data */
GtkTreeModel *lc_model_create(void);

/* Initializes the treeviews */
int gui_init_treeviews(GladeXML *parent_window);

/* Initializes all LC related treeviews*/
int gui_init_lc_treeviews(GladeXML *parent_window);
