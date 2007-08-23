#include <gtk/gtk.h>
#include <glade/glade.h>

/* LC treeviews */
GtkWidget *phoebe_data_lc_treeview;
GtkWidget *phoebe_para_lc_levels_treeview;
GtkWidget *phoebe_para_lc_el3_treeview;
GtkWidget *phoebe_para_lc_levweight_treeview;
GtkWidget *phoebe_para_lc_ld_treeview;

/* Initializes the treeviews */
int gui_init_treeviews(GladeXML *parent_window, GladeXML *phoebe_load_lc_dialog);

/* These columns make up the light curve model for various treeviews */
typedef enum lc_model_columns
{
    LC_COL_ACTIVE,
    LC_COL_FILENAME,
    LC_COL_FILTER,
	LC_COL_FILTERNO,
    LC_COL_ITYPE,
    LC_COL_ITYPE_STR,
    LC_COL_DTYPE,
    LC_COL_DTYPE_STR,
    LC_COL_WTYPE,
    LC_COL_WTYPE_STR,
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

/* Creates a model for storing light curves data */
GtkTreeModel *lc_model_create(void);

/* Initializes all LC related treeviews */
int gui_init_lc_treeviews(GladeXML *parent_window);

/* RV treeviews */
GtkWidget *phoebe_data_rv_treeview;
GtkWidget *phoebe_para_rv_ld_treeview;

/* These columns make up the RV curve model for various treeviews */
typedef enum rv_model_columns
{
    RV_COL_ACTIVE,
    RV_COL_FILENAME,
    RV_COL_FILTER,
    RV_COL_ITYPE,
    RV_COL_ITYPE_STR,
    RV_COL_DTYPE,
    RV_COL_DTYPE_STR,
    RV_COL_WTYPE,
    RV_COL_WTYPE_STR,
    RV_COL_SIGMA,
    RV_COL_X1,
    RV_COL_X2,
    RV_COL_Y1,
    RV_COL_Y2,
    RV_COL_COUNT,
}rv_model_columns;

/* Creates a model for storing RV curves data */
GtkTreeModel *rv_model_create(void);

/* Initializes all RV related treeviews */
int gui_init_rv_treeviews(GladeXML *parent_window);

/* Spots treeview */
GtkWidget *phoebe_para_surf_spots_treeview;

/* These columns make up the spots list model */
typedef enum spots_model_columns
{
    SPOTS_COL_ADJUST,
    SPOTS_COL_SOURCE,
    SPOTS_COL_SOURCE_STR,
    SPOTS_COL_LAT,
    SPOTS_COL_LATADJUST,
    SPOTS_COL_LATSTEP,
    SPOTS_COL_LATMIN,
    SPOTS_COL_LATMAX,
    SPOTS_COL_LON,
    SPOTS_COL_LONADJUST,
    SPOTS_COL_LONSTEP,
    SPOTS_COL_LONMIN,
    SPOTS_COL_LONMAX,
    SPOTS_COL_RAD,
    SPOTS_COL_RADADJUST,
    SPOTS_COL_RADSTEP,
    SPOTS_COL_RADMIN,
    SPOTS_COL_RADMAX,
    SPOTS_COL_TEMP,
    SPOTS_COL_TEMPADJUST,
    SPOTS_COL_TEMPSTEP,
    SPOTS_COL_TEMPMIN,
    SPOTS_COL_TEMPMAX,
    SPOTS_COL_COUNT,
}spots_model_columns;

/* Creates a model for storing spots data */
GtkTreeModel *spots_model_create(void);

/* Initializes the spots treeview */
int gui_init_spots_treeview (GladeXML *parent_window);

/* Cell data function for transforming the result of combobox selection (integer)
   into human readable strings that should appear in the spots treeview. */
void spots_source_cell_data_func(GtkTreeViewColumn *column, GtkCellRenderer *renderer, GtkTreeModel *model, GtkTreeIter *iter, gpointer data);

///* Data sheet treevies */
//GtkWidget *phoebe_sidesheet_fitt_treeview;
//GtkWidget *phoebe_sidesheet_data_treeview;
//
///* These columns make up the data sheet model */
//typedef enum datasheet_model_columns
//{
//    DS_COL_PARAM_TBA,
//    DS_COL_PARAM_NAME,
//    DS_COL_PARAM_VALUE,
//    DS_COL_PARAM_ERROR,
//    DS_COL_PARAM_STEP,
//    DS_COL_PARAM_MIN,
//    DS_COL_PARAM_MAX,
//    DS_COL_COUNT,
//}datasheet_model_columns;
//
///* Creates a model for storing data sheet lists */
//GtkTreeModel *datasheets_model_create(void);
//
///* Initializes the data sheets treeviews */
//int gui_init_datasheets (GladeXML *parent_window);

/* Initialzes filter comboboxes */
int gui_init_filter_combobox (GtkWidget *combo_box);



