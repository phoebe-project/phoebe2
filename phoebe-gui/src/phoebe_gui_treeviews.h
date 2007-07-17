#include <gtk/gtk.h>
#include <glade/glade.h>

/* *** The treeviews *** */

/* LC treeviews */
GtkWidget *phoebe_data_lc_treeview;
GtkWidget *phoebe_para_lc_levels_treeview;
GtkWidget *phoebe_para_lc_el3_treeview;
GtkWidget *phoebe_para_lc_levweight_treeview;
GtkWidget *phoebe_para_lc_ld_treeview;

/* RV treeviews */
GtkWidget *phoebe_data_rv_treeview;
GtkWidget *phoebe_para_rv_ld_treeview;

/* Spots treeview */
GtkWidget *phoebe_para_surf_spots_treeview;

/* These columns make up the light curve model for various treeviews */
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
    LC_COL_X1,
    LC_COL_X2,
    LC_COL_Y1,
    LC_COL_Y2,
    LC_COL_COUNT,
}lc_model_columns;

/* These columns make up the RV curve model for various treeviews */
typedef enum rv_model_columns
{
    RV_COL_ACTIVE,
    RV_COL_FILENAME,
    RV_COL_FILTER,
    RV_COL_ITYPE,
    RV_COL_DTYPE,
    RV_COL_WTYPE,
    RV_COL_SIGMA,
    RV_COL_X1,
    RV_COL_X2,
    RV_COL_Y1,
    RV_COL_Y2,
    RV_COL_COUNT,
}rv_model_columns;

/* These columns make up the spots list model */
typedef enum spots_model_columns
{
    SPOTS_COL_ADJUST,
    SPOTS_COL_SOURCE,
    SPOTS_COL_LAT,
    SPOTS_COL_LON,
    SPOTS_COL_RAD,
    SPOTS_COL_TEMP,
    SPOTS_COL_COUNT,
}spots_model_columns;

/* Creates a model for storing light curves data */
GtkTreeModel *lc_model_create(void);

/* Creates a model for storing RV curves data */
GtkTreeModel *rv_model_create(void);

/* Creates a model for storing spots data */
GtkTreeModel *spots_model_create(void);

/* Initializes the treeviews */
int gui_init_treeviews(GladeXML *parent_window);

/* Initializes all LC related treeviews */
int gui_init_lc_treeviews(GladeXML *parent_window);

/* Initializes all RV related treeviews */
int gui_init_rv_treeviews(GladeXML *parent_window);

/* Initializes the spots treeview */
int gui_init_spots_treeview (GladeXML *parent_window);
