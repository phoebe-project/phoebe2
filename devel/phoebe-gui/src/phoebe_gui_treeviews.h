#include <gtk/gtk.h>
#include <glade/glade.h>

typedef enum GUI_lc_model_columns {
	LC_COL_ACTIVE,
	LC_COL_FILENAME,
	LC_COL_ID,
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
	LC_COL_EL3_LUM,
	LC_COL_EXTINCTION,
	LC_COL_X1,
	LC_COL_X2,
	LC_COL_Y1,
	LC_COL_Y2,
	LC_COL_PLOT_OBS,
	LC_COL_PLOT_SYN,
	LC_COL_PLOT_OBS_COLOR,
	LC_COL_PLOT_SYN_COLOR,
	LC_COL_PLOT_OFFSET,
	LC_COL_COUNT
} GUI_lc_model_columns;

typedef enum GUI_rv_model_columns {
    RV_COL_ACTIVE,
    RV_COL_FILENAME,
    RV_COL_ID,
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
	RV_COL_PLOT_OBS,
	RV_COL_PLOT_SYN,
	RV_COL_PLOT_OBS_COLOR,
	RV_COL_PLOT_SYN_COLOR,
	RV_COL_PLOT_OFFSET,
    RV_COL_COUNT
} GUI_rv_model_columns;

typedef enum GUI_spots_model_columns {
    SPOTS_COL_ACTIVE,
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
    SPOTS_COL_ADJUST,
    SPOTS_COL_COUNT
} GUI_spots_model_columns;

typedef enum GUI_sidesheet_results_model_columns {
    RS_COL_PARAM_NAME,
    RS_COL_PARAM_VALUE,
	RS_COL_PARAM_ERROR,
    RS_COL_COUNT
} GUI_sidesheet_results_model_columns;

typedef enum GUI_sidesheet_parameter_order {
	SIDESHEET_LAGRANGE_1,
	SIDESHEET_LAGRANGE_2,
	SIDESHEET_MASS_1,
	SIDESHEET_MASS_2,
	SIDESHEET_RADIUS_1,
	SIDESHEET_RADIUS_2,
	SIDESHEET_MBOL_1,
	SIDESHEET_MBOL_2,
	SIDESHEET_LOGG_1,
	SIDESHEET_LOGG_2,
	SIDESHEET_SBR_1,
	SIDESHEET_SBR_2,
	SIDESHEET_NUM_PARAMS
} GUI_sidesheet_parameter_order;

typedef enum GUI_sidesheet_fitting_model_columns {
    FS_COL_PARAM_NAME,
    FS_COL_PARAM_VALUE,
    FS_COL_PARAM_STEP,
    FS_COL_PARAM_MIN,
    FS_COL_PARAM_MAX,
    FS_COL_COUNT
} GUI_sidesheet_fitting_model_columns;

typedef enum GUI_minimizer_feedback_model_columns {
	MF_COL_QUALIFIER,
	MF_COL_INITVAL,
	MF_COL_NEWVAL,
	MF_COL_ERROR,
	MF_COL_COUNT
} GUI_minimizer_feedback_model_columns;

typedef enum GUI_statistics_treeview_columns {
	CURVE_COL_NAME,
	CURVE_COL_NPOINTS,
	CURVE_COL_U_RES,
	CURVE_COL_I_RES,
	CURVE_COL_P_RES,
	CURVE_COL_F_RES,
	CURVE_COL_COUNT
} GUI_statistics_treeview_columns;

int gui_init_treeviews					();
int gui_reinit_treeviews				();

int gui_init_lc_treeviews				();
int gui_reinit_lc_treeviews				();

int gui_init_rv_treeviews				();
int gui_reinit_rv_treeviews				();

int gui_init_spots_treeview 			();
int gui_reinit_spots_treeview			();

int gui_init_sidesheet_res_treeview		();
int gui_init_sidesheet_fit_treeview 	();

int gui_fill_sidesheet_res_treeview 	();

int gui_fill_sidesheet_fit_treeview 	();

int gui_init_fitt_mf_treeview			();
int gui_fill_fitt_mf_treeview			();

int gui_fit_statistics_treeview_init    ();

int gui_init_lc_plot_treeview           ();

int gui_init_filter_combobox 			(GtkWidget *combo_box, gint activefilter);

int gui_data_lc_treeview_add 			();
int gui_data_rv_treeview_add 			();
int gui_data_lc_treeview_edit			();
int gui_data_rv_treeview_edit			();
int gui_para_lum_levels_edit			();
int gui_para_lum_levels_calc			(GtkTreeModel *model, GtkTreeIter iter);
int gui_para_lum_levels_calc_selected		();
int gui_para_lum_el3_edit			();
int gui_fitt_levelweight_edit			();
int gui_para_lc_coefficents_edit		();
int gui_para_rv_coefficents_edit 		();
int gui_data_lc_treeview_remove			();
int gui_data_rv_treeview_remove			();

int gui_set_treeview_value              (GtkTreeModel *model, int col_id, int row_id, double value);
int gui_update_cla_value                (int row);

int gui_spots_treeview_toggle_show_all	();
int gui_spots_parameters_marked_tba		();

int gui_spots_add						();
int gui_spots_edit						();

/* extern bool phoebe_para_spots_units_combobox_init; */

