#include <gtk/gtk.h>
#include <glade/glade.h>

typedef enum lc_model_columns {
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
    LC_COL_COUNT
} lc_model_columns;

typedef enum rv_model_columns {
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
    RV_COL_COUNT
} rv_model_columns;

typedef enum spots_model_columns {
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
    SPOTS_COL_COUNT
} SPOTS_model_columns;

typedef enum adjustible_spots_model_columns {
	ADJ_SPOTS_COL_SOURCE1,
	ADJ_SPOTS_COL_LAT1,
    ADJ_SPOTS_COL_LAT1ADJUST,
    ADJ_SPOTS_COL_LAT1STEP,
    ADJ_SPOTS_COL_LAT1MIN,
    ADJ_SPOTS_COL_LAT1MAX,
    ADJ_SPOTS_COL_LON1,
    ADJ_SPOTS_COL_LON1ADJUST,
    ADJ_SPOTS_COL_LON1STEP,
    ADJ_SPOTS_COL_LON1MIN,
    ADJ_SPOTS_COL_LON1MAX,
    ADJ_SPOTS_COL_RAD1,
    ADJ_SPOTS_COL_RAD1ADJUST,
    ADJ_SPOTS_COL_RAD1STEP,
    ADJ_SPOTS_COL_RAD1MIN,
    ADJ_SPOTS_COL_RAD1MAX,
    ADJ_SPOTS_COL_TEMP1,
    ADJ_SPOTS_COL_TEMP1ADJUST,
    ADJ_SPOTS_COL_TEMP1STEP,
    ADJ_SPOTS_COL_TEMP1MIN,
    ADJ_SPOTS_COL_TEMP1MAX,
    ADJ_SPOTS_COL_SOURCE2,
	ADJ_SPOTS_COL_LAT2,
    ADJ_SPOTS_COL_LAT2ADJUST,
    ADJ_SPOTS_COL_LAT2STEP,
    ADJ_SPOTS_COL_LAT2MIN,
    ADJ_SPOTS_COL_LAT2MAX,
    ADJ_SPOTS_COL_LON2,
    ADJ_SPOTS_COL_LON2ADJUST,
    ADJ_SPOTS_COL_LON2STEP,
    ADJ_SPOTS_COL_LON2MIN,
    ADJ_SPOTS_COL_LON2MAX,
    ADJ_SPOTS_COL_RAD2,
    ADJ_SPOTS_COL_RAD2ADJUST,
    ADJ_SPOTS_COL_RAD2STEP,
    ADJ_SPOTS_COL_RAD2MIN,
    ADJ_SPOTS_COL_RAD2MAX,
    ADJ_SPOTS_COL_TEMP2,
    ADJ_SPOTS_COL_TEMP2ADJUST,
    ADJ_SPOTS_COL_TEMP2STEP,
    ADJ_SPOTS_COL_TEMP2MIN,
    ADJ_SPOTS_COL_TEMP2MAX,
    ADJ_SPOTS_COL_COUNT
} adjustible_spots_model_columns;

typedef enum sidesheet_results_model_columns {
    RS_COL_PARAM_NAME,
    RS_COL_PARAM_VALUE,
    RS_COL_COUNT
} sidesheet_results_model_columns;

typedef enum sidesheet_fitting_model_columns {
    FS_COL_PARAM_NAME,
    FS_COL_PARAM_VALUE,
    FS_COL_PARAM_STEP,
    FS_COL_PARAM_MIN,
    FS_COL_PARAM_MAX,
    FS_COL_COUNT
} sidesheet_fitting_model_columns;

int gui_init_treeviews				();
int gui_reinit_treeviews			();
int gui_init_lc_treeviews			();
int gui_reinit_lc_treeviews			();
int gui_init_rv_treeviews			();
int gui_reinit_rv_treeviews			();
int gui_init_spots_treeview 		();
int gui_reinit_spots_treeview		();
int gui_init_sidesheet_res_treeview	();
int gui_init_sidesheet_fit_treeview ();
int gui_fill_sidesheet_res_treeview ();
int gui_fill_sidesheet_fit_treeview ();
int gui_init_filter_combobox 		(GtkWidget *combo_box);
int gui_init_lc_obs_combobox		();
int gui_init_rv_obs_combobox		();

void on_lc_model_row_changed(GtkTreeModel *tree_model, GtkTreePath *path, GtkTreeIter *iter, gpointer user_data);
void on_rv_model_row_changed(GtkTreeModel *tree_model, GtkTreePath *path, GtkTreeIter *iter, gpointer user_data);

int gui_edit_data_lc_treeview();
int gui_edit_data_rv_treeview();
