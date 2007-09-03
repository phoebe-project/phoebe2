#include <glade/glade.h>
#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_treeviews.h"

gboolean PHOEBE_WINDOW_SIDESHEET_IS_DETACHED = FALSE;
gboolean PHOEBE_WINDOW_LC_PLOT_IS_DETACHED   = FALSE;
gboolean PHOEBE_WINDOW_RV_PLOT_IS_DETACHED   = FALSE;
gboolean PHOEBE_WINDOW_FITTING_IS_DETACHED   = FALSE;

int gui_init_widgets ()
{
	/*
	 * This function hooks all widgets to the parameters and adds them to the
	 * widget hash table.
	 */

	int i;

	PHOEBE_parameter *par;
	gchar *glade_xml_file;
    gchar *glade_pixmap_file;
	GladeXML  *phoebe_window;

	GtkWidget *phoebe_data_lc_treeview,       *phoebe_para_lc_levels_treeview,
              *phoebe_para_lc_el3_treeview,   *phoebe_para_lc_levweight_treeview,
		      *phoebe_para_lc_ld_treeview,    *phoebe_data_rv_treeview,
		      *phoebe_para_rv_ld_treeview,    *phoebe_para_surf_spots_treeview,
		      *phoebe_sidesheet_res_treeview, *phoebe_sidesheet_fit_treeview;

	/* Read in the main PHOEBE window: */
	glade_xml_file    = g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe.glade", NULL);
	glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
	phoebe_window = glade_xml_new (glade_xml_file, NULL, NULL);
	g_free (glade_xml_file);

	glade_xml_signal_autoconnect (phoebe_window);

	GUI_wt = phoebe_malloc (sizeof (GUI_widget_table));
	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++)
		GUI_wt->bucket[i] = NULL;

	/* *************************** GUI Parameters *************************** */

	phoebe_parameter_add ("gui_ld_model_autoupdate",	"Automatically update LD model",	KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_fitt_method",			"Fitting method",					KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Diferential Corrections");
	phoebe_parameter_add ("gui_lc_plot_synthetic",		"Plot synthetic LC",				KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_lc_plot_observed",		"Plot observed LC",					KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_verticesno",		"Number of vertices for LC",		KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_INT,		100);
	phoebe_parameter_add ("gui_lc_plot_obsmenu",		"Select observed LC",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_aliasing",		"Turn on data aliasing",			KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_lc_plot_residuals",		"Plot residuals",					KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_x",				"X-axis of LC plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Phase");
	phoebe_parameter_add ("gui_lc_plot_y",				"Y-axis of LC plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Magnitude");
	phoebe_parameter_add ("gui_lc_plot_phstart",		"Phase start",						KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_lc_plot_phend",			"Phase end",						KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_lc_plot_offset",			"Offset level",						KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_zoom",			"Zoom level",						KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_coarse",			"Coarse grid",						KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_fine",			"Coarse grid",						KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_synthetic",		"Plot synthetic RV curve",			KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_rv_plot_observed",		"Plot observed RV curve",			KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_verticesno",		"Number of vertices for RV curve",	KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_INT,		100);
	phoebe_parameter_add ("gui_rv_plot_obsmenu",		"Select observed RV curve",			KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_alias",			"Turn on data aliasing",			KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_rv_plot_residuals",		"Plot residuals",					KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_x",				"X-axis of RV plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Phase");
	phoebe_parameter_add ("gui_rv_plot_y",				"Y-axis of RV plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Both RVs");
	phoebe_parameter_add ("gui_rv_plot_phstart",		"Phase start",						KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_rv_plot_phend",			"Phase end",						KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_rv_plot_offset",			"Offset level",						KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_zoom",			"Zoom level",						KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_coarse",			"Coarse grid",						KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_fine",			"Coarse grid",						KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);

	/* *************************** Main Window    **************************** */

	gui_widget_add ("phoebe_window",									glade_xml_get_widget (phoebe_window, "phoebe_window"), 													0, 					GUI_WIDGET_VALUE, 		NULL);

	/* ************************    GUI Treeviews   ************************* */

	phoebe_data_lc_treeview           = glade_xml_get_widget (phoebe_window, "phoebe_data_lc_treeview");
	phoebe_para_lc_levels_treeview    = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_levels_treeview");
	phoebe_para_lc_el3_treeview       = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_el3_treeview");
	phoebe_para_lc_levweight_treeview = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_weighting_treeview");
	phoebe_para_lc_ld_treeview        = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_lccoefs_treeview");
	phoebe_data_rv_treeview           = glade_xml_get_widget (phoebe_window, "phoebe_data_rv_treeview");
	phoebe_para_rv_ld_treeview        = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_rvcoefs_treeview");
	phoebe_para_surf_spots_treeview   = glade_xml_get_widget (phoebe_window, "phoebe_para_surf_spots_treeview");
	phoebe_sidesheet_res_treeview     = glade_xml_get_widget (phoebe_window, "phoebe_sidesheet_res_treeview");
	phoebe_sidesheet_fit_treeview     = glade_xml_get_widget (phoebe_window, "phoebe_sidesheet_fit_treeview");

	gui_widget_add ("phoebe_data_lc_treeview",							phoebe_data_lc_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_para_lc_el3_treeview",						phoebe_para_lc_el3_treeview, 																			0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_para_lc_levels_treeview",					phoebe_para_lc_levels_treeview,																			0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_para_lc_levweight_treeview",				phoebe_para_lc_levweight_treeview,																		0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_para_lc_ld_treeview",						phoebe_para_lc_ld_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_data_rv_treeview",							phoebe_data_rv_treeview, 																				0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_para_rv_ld_treeview",						phoebe_para_rv_ld_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_para_surf_spots_treeview",					phoebe_para_surf_spots_treeview,																		0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_sidesheet_res_treeview",					phoebe_sidesheet_res_treeview,																			0, 					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_sidesheet_fit_treeview",					phoebe_sidesheet_fit_treeview,																			0, 					GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_lc_plot_options_obs_combobox",				glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_obs_combobox"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup("gui_lc_plot_obsmenu"));
	gui_widget_add ("phoebe_rv_plot_options_obs_combobox",				glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_obs_combobox"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup("gui_rv_plot_obsmenu"));

	gui_widget_add ("phoebe_fitt_first_treeview",                       glade_xml_get_widget(phoebe_window, "phoebe_fitt_first_treeview"),                                      0,                  GUI_WIDGET_VALUE,       NULL);
	gui_widget_add ("phoebe_fitt_second_treeview",                      glade_xml_get_widget(phoebe_window, "phoebe_fitt_second_treeview"),                                     0,                  GUI_WIDGET_VALUE,       NULL);
	gui_widget_add ("phoebe_fitt_third_treeview",                       glade_xml_get_widget(phoebe_window, "phoebe_fitt_third_treeview"),                                      0,                  GUI_WIDGET_VALUE,       NULL);

	gui_init_treeviews ();

	/* *************************    Data Widgets   **************************** */

	gui_widget_add ("phoebe_data_star_name_entry", 						glade_xml_get_widget (phoebe_window, "phoebe_data_star_name_entry"), 									0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_name"));
	gui_widget_add ("phoebe_data_star_model_combobox", 					glade_xml_get_widget (phoebe_window, "phoebe_data_star_model_combobox"), 								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_model"));

	gui_widget_add ("phoebe_data_lc_filename",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_FILENAME,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filename"));
	gui_widget_add ("phoebe_data_lc_sigma",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 	  					LC_COL_SIGMA,		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_sigma"));
	gui_widget_add ("phoebe_data_lc_filter",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_FILTER,		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filter"));
	gui_widget_add ("phoebe_data_lc_indep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_ITYPE_STR,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indep"));
	gui_widget_add ("phoebe_data_lc_dep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_DTYPE_STR,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_dep"));
	gui_widget_add ("phoebe_data_lc_wtype",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_WTYPE_STR,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indweight"));
	gui_widget_add ("phoebe_data_lc_active",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_ACTIVE,		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_active"));

	gui_widget_add ("phoebe_data_rv_filename",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_FILENAME,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_filename"));
	gui_widget_add ("phoebe_data_rv_sigma",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_SIGMA,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_sigma"));
	gui_widget_add ("phoebe_data_rv_filter",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_FILTER,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_filter"));
	gui_widget_add ("phoebe_data_rv_indep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_ITYPE_STR,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_indep"));
	gui_widget_add ("phoebe_data_rv_dep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_DTYPE_STR,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_dep"));
	gui_widget_add ("phoebe_data_rv_wtype",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_WTYPE_STR,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_indweight"));
	gui_widget_add ("phoebe_data_rv_active",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_ACTIVE,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_active"));

	gui_widget_add ("phoebe_data_options_indep_combobox",	 			glade_xml_get_widget(phoebe_window, "phoebe_data_options_indep_combobox"), 								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_indep"));

	gui_widget_add ("phoebe_data_options_bins_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_options_bins_checkbutton"), 							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins_switch"));
	gui_widget_add ("phoebe_data_options_binsno_spinbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_options_binsno_spinbutton"), 							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins"));

	gui_widget_add ("phoebe_data_lcoptions_mag_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_lcoptions_mag_spinbutton"), 							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_mnorm"));

	gui_widget_add ("phoebe_data_rvoptions_psepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_psepe_checkbutton"), 						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"));
	gui_widget_add ("phoebe_data_rvoptions_ssepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_ssepe_checkbutton"), 						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"));

	gui_widget_add ("phoebe_data_options_filtermode_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_data_options_filtermode_combobox"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_passband_mode"));

	/* **********************    Parameters Widgets   ************************* */

	par = phoebe_parameter_lookup ("phoebe_hjd0");
	gui_widget_add ("phoebe_para_eph_hjd0_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_hjd0adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_hjd0step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_hjd0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_hjd0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_period");
	gui_widget_add ("phoebe_para_eph_period_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_period_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_periodadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_periodstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_periodmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_periodmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_dpdt");
	gui_widget_add ("phoebe_para_eph_dpdt_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdt_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_dpdtadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_dpdtstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_dpdtmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_dpdtmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pshift");
	gui_widget_add ("phoebe_para_eph_pshift_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshift_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_pshiftadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_pshiftstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_pshiftmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_pshiftmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_sma");
	gui_widget_add ("phoebe_para_sys_sma_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_sma_spinbutton"),									0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_smaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smaadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_smastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smastep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_smamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smamax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_smamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smamin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_rm");
	gui_widget_add ("phoebe_para_sys_rm_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rm_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_rmadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_rmstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmstep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_rmmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_rmmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_vga");
	gui_widget_add ("phoebe_para_sys_vga_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vga_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_vgaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgaadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_vgastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgastep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_vgamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgamax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_vgamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgamin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_incl");
	gui_widget_add ("phoebe_para_sys_incl_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_incl_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_incladjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_incladjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_inclstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_inclmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_inclmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_perr0");
	gui_widget_add ("phoebe_para_orb_perr0_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_perr0adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_perr0step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_perr0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_perr0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_dperdt");
	gui_widget_add ("phoebe_para_orb_dperdt_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdt_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_dperdtadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_dperdtstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_dperdtmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_dperdtmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_ecc");
	gui_widget_add ("phoebe_para_orb_ecc_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_ecc_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_eccadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_eccstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccstep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_eccmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_eccmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_f1");
	gui_widget_add ("phoebe_para_orb_f1_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_f1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_f1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1step_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_f1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_f1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_f2");
	gui_widget_add ("phoebe_para_orb_f2_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_f2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_f2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2step_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_f2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_f2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_teff1");
	gui_widget_add ("phoebe_para_comp_tavh_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavh_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_tavhadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_tavhstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_tavhmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_tavhmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_teff2");
	gui_widget_add ("phoebe_para_comp_tavc_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavc_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_tavcadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_tavcstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_tavcmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_tavcmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pot1");
	gui_widget_add ("phoebe_para_comp_phsv_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsv_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_phsvadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_phsvstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_phsvmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_phsvmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pot2");
	gui_widget_add ("phoebe_para_comp_pcsv_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsv_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_pcsvadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_pcsvstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_pcsvmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_pcsvmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_met1");
	gui_widget_add ("phoebe_para_comp_met1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_met1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_met1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_met1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_met1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_met2");
	gui_widget_add ("phoebe_para_comp_met2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_met2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_met2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_met2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_met2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_alb1");
	gui_widget_add ("phoebe_para_surf_alb1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_alb1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_alb1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_alb1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_alb1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_alb2");
	gui_widget_add ("phoebe_para_surf_alb2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_alb2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_alb2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_alb2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_alb2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_grb1");
	gui_widget_add ("phoebe_para_surf_gr1_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_gr1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_gr1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_gr1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_gr1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_grb2");
	gui_widget_add ("phoebe_para_surf_gr2_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_gr2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_gr2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_gr2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_gr2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par);

    gui_widget_add ("phoebe_para_lum_levels_levweight", 				(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_levels_treeview), 				LC_COL_LEVWEIGHT,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_levweight"));

    par = phoebe_parameter_lookup ("phoebe_hla");
    gui_widget_add ("phoebe_para_lum_levels_prim",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_levels_treeview), 				LC_COL_HLA,			GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_lum_levels_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_lum_levels_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP, 	par);

	par = phoebe_parameter_lookup ("phoebe_cla");
	gui_widget_add ("phoebe_para_lum_levels_sec",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_levels_treeview), 				LC_COL_CLA,			GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_lum_levels_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_lum_levels_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP, 	par);

    par = phoebe_parameter_lookup ("phoebe_el3");
    gui_widget_add ("phoebe_para_lum_el3",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_el3_treeview),					LC_COL_EL3,			GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_lum_el3ajdust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3ajdust_checkbutton"),							0,					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_lum_el3step_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3step_spinbutton"),								0,					GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_lum_el3units_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3units_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup("phoebe_el3_units"));

	par = phoebe_parameter_lookup ("phoebe_opsf");
	gui_widget_add ("phoebe_para_lum_el3_opacity",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_el3_treeview),					LC_COL_OPSF,		GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_lum_el3_opacityadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacityadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,  par);
	gui_widget_add ("phoebe_para_lum_el3_opacitystep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacitystep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP, 	par);

	par = phoebe_parameter_lookup ("phoebe_extinction");
	gui_widget_add ("phoebe_para_lum_el3_ext",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_el3_treeview),					LC_COL_EXTINCTION,	GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_lum_el3_extadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extadjust_checkbutton"),						0,					GUI_WIDGET_SWITCH_TBA,  par);
	gui_widget_add ("phoebe_para_lum_el3_extstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extstep_spinbutton"),							0,					GUI_WIDGET_VALUE_STEP,  par);

	gui_widget_add ("phoebe_para_lum_atmospheres_prim_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_prim_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm1_switch"));
	gui_widget_add ("phoebe_para_lum_atmospheres_sec_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_sec_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm2_switch"));

	gui_widget_add ("phoebe_para_lum_options_reflections_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_reflections_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_switch"));
	gui_widget_add ("phoebe_para_lum_options_reflections_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_reflections_spinbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_reflections"));
	gui_widget_add ("phoebe_para_lum_options_decouple_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_decouple_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_usecla_switch"));

	gui_widget_add ("phoebe_para_lum_noise_lcscatter_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_lcscatter_checkbutton"),						0,					GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_synscatter_switch"));
	gui_widget_add ("phoebe_para_lum_noise_sigma_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_sigma_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_sigma"));
	gui_widget_add ("phoebe_para_lum_noise_seed_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_seed_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_seed"));
	gui_widget_add ("phoebe_para_lum_noise_lcscatter_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_lcscatter_combobox"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_levweight"));

	gui_widget_add ("phoebe_para_ld_model_combobox",					glade_xml_get_widget(phoebe_window, "phoebe_para_ld_model_combobox"),									0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_model"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_primx_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_primx_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol1"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_primy_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_primx_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol1"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_secx_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_secx_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol2"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_secy_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_secx_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol2"));

	par = phoebe_parameter_lookup ("phoebe_ld_lcx1");
	gui_widget_add ("phoebe_para_ld_lccoefs_primx",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_X1,			GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_ld_lccoefs_primy",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_Y1,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_lcy1"));
	gui_widget_add ("phoebe_para_ld_lccoefs_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,	par);
	gui_widget_add ("phoebe_para_ld_lccoefs_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP,	par);

	par = phoebe_parameter_lookup ("phoebe_ld_lcx2");
	gui_widget_add ("phoebe_para_ld_lccoefs_secx",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_X2,			GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_ld_lccoefs_secy",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_Y2,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_lcy2"));
	gui_widget_add ("phoebe_para_ld_lccoefs_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,	par);
	gui_widget_add ("phoebe_para_ld_lccoefs_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP,	par);

	/* ***********************    Fitting Widgets   ************************* */

	gui_widget_add ("phoebe_fitt_parameters_finesize1_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_finesize1_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize1"));
	gui_widget_add ("phoebe_fitt_parameters_finesize2_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_finesize2_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize2"));
	gui_widget_add ("phoebe_fitt_parameters_coarsize1_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_coarsize1_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize1"));
	gui_widget_add ("phoebe_fitt_parameters_coarsize2_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_coarsize2_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize2"));
	gui_widget_add ("phoebe_fitt_parameters_lambda_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_lambda_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_dc_lambda"));

	/* *************************    GUI Widgets   *************************** */

	gui_widget_add ("phoebe_para_ld_model_autoupdate_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_model_autoupdate_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_ld_model_autoupdate"));
	gui_widget_add ("phoebe_fitt_method_combobox",						glade_xml_get_widget(phoebe_window, "phoebe_fitt_method_combobox"),										0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_fitt_method"));
	gui_widget_add ("phoebe_lc_plot_options_syn_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_syn_checkbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_synthetic"));
	gui_widget_add ("phoebe_lc_plot_options_obs_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_obs_checkbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_observed"));
	gui_widget_add ("phoebe_lc_plot_options_vertices_no_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_vertices_no_spinbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_verticesno"));
//	gui_widget_add ("phoebe_lc_plot_options_obs_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_obs_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_obsmenu"));
	gui_widget_add ("phoebe_lc_plot_options_alias_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_alias_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_aliasing"));
	gui_widget_add ("phoebe_lc_plot_options_residuals_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_residuals_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_residuals"));
	gui_widget_add ("phoebe_lc_plot_options_x_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_x_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_x"));
	gui_widget_add ("phoebe_lc_plot_options_y_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_y_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_y"));
	gui_widget_add ("phoebe_lc_plot_options_phstart_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phstart_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_phstart"));
	gui_widget_add ("phoebe_lc_plot_options_phend_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phend_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_phend"));
	gui_widget_add ("phoebe_lc_plot_controls_offset_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_offset_combobox"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_offset"));
	gui_widget_add ("phoebe_lc_plot_controls_zoom_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_zoom_combobox"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_zoom"));
	gui_widget_add ("phoebe_lc_plot_controls_coarse_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_coarse_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_coarse"));
	gui_widget_add ("phoebe_lc_plot_controls_fine_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_fine_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_fine"));
	gui_widget_add ("phoebe_rv_plot_options_syn_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_syn_checkbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_synthetic"));
	gui_widget_add ("phoebe_rv_plot_options_obs_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_obs_checkbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_observed"));
	gui_widget_add ("phoebe_rv_plot_options_vertices_no_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_vertices_no_spinbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_verticesno"));
//	gui_widget_add ("phoebe_rv_plot_options_obs_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_obs_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_obsmenu"));
	gui_widget_add ("phoebe_rv_plot_options_alias_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_alias_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_alias"));
	gui_widget_add ("phoebe_rv_plot_options_residuals_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_residuals_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_residuals"));
	gui_widget_add ("phoebe_rv_plot_options_x_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_x_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_x"));
	gui_widget_add ("phoebe_rv_plot_options_y_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_y_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_y"));
	gui_widget_add ("phoebe_rv_plot_options_phstart_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phstart_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_phstart"));
	gui_widget_add ("phoebe_rv_plot_options_phend_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phend_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_phend"));
	gui_widget_add ("phoebe_rv_plot_controls_offset_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_offset_combobox"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_offset"));
	gui_widget_add ("phoebe_rv_plot_scrolledwindow",					glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_scrolledwindow"),									0,					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_rv_plot_controls_zoom_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_zoom_combobox"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_zoom"));
	gui_widget_add ("phoebe_rv_plot_controls_coarse_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_coarse_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_coarse"));
	gui_widget_add ("phoebe_rv_plot_controls_fine_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_fine_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_fine"));

	gui_widget_add ("phoebe_sidesheet_detach_button",                   glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_detach_button"),                                  0,                  GUI_WIDGET_VALUE,       NULL);
	gui_widget_add ("phoebe_fitt_detach_button",                        glade_xml_get_widget(phoebe_window, "phoebe_fitt_detach_button"),                                       0,                  GUI_WIDGET_VALUE,       NULL);
	gui_widget_add ("phoebe_lc_plot_detach_button",                     glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_detach_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL);
	gui_widget_add ("phoebe_rv_plot_detach_button",                     glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_detach_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL);

	/* ************************    GUI Containers   ************************* */

	gui_widget_add ("phoebe_lc_plot_image",								glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_image"),									        0,					GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_sidesheet_parent_table",					glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_parent_table"),									0,					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_sidesheet_vbox",							glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_vbox"),											0,					GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_lc_plot_table",								glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_table"),											0,					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_lc_plot_parent_table",						glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_parent_table"),										0,					GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_rv_plot_table",								glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_table"),											0,					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_rv_plot_parent_table",						glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_parent_table"),										0,					GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_fitt_fitting_frame",						glade_xml_get_widget(phoebe_window, "phoebe_fitt_fitting_frame"),										0,					GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_fitt_fitting_parent_table",					glade_xml_get_widget(phoebe_window, "phoebe_fitt_fitting_parent_table"),								0,					GUI_WIDGET_VALUE, 		NULL);

	g_object_unref (phoebe_window);

	gui_init_parameter_options ();

	gtk_widget_show (gui_widget_lookup ("phoebe_window")->gtk);
	gtk_window_set_icon (GTK_WINDOW(gui_widget_lookup ("phoebe_window")->gtk), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

	{
		int _i;
		GUI_wt_bucket *elem;

		for (_i = 0; _i < GUI_WT_HASH_BUCKETS; _i++) {
			elem = GUI_wt->bucket[_i];
			while (elem) {
				printf ("%50s", elem->widget->name);
				elem = elem->next;
			}
			printf ("\n");
		}
	}

	gui_set_values_to_widgets();

	return SUCCESS;
}

int gui_init_parameter_options()
{
	PHOEBE_parameter *par;
	int status = 0;

	par = phoebe_parameter_lookup("gui_fitt_method");
	phoebe_parameter_add_option (par, "Diferential Corrections");
	phoebe_parameter_add_option (par, "Nelder & Mead's Simplex");

	par = phoebe_parameter_lookup("gui_lc_plot_x");
	phoebe_parameter_add_option (par, "Phase");
	phoebe_parameter_add_option (par, "Time");

	par = phoebe_parameter_lookup("gui_lc_plot_y");
	phoebe_parameter_add_option (par, "Magnitude");
	phoebe_parameter_add_option (par, "Primary flux");
	phoebe_parameter_add_option (par, "Secondary flux");
	phoebe_parameter_add_option (par, "Total flux");
	phoebe_parameter_add_option (par, "Normalized flux");

	par = phoebe_parameter_lookup("gui_rv_plot_x");
	phoebe_parameter_add_option (par, "Phase");
	phoebe_parameter_add_option (par, "Time");

	par = phoebe_parameter_lookup("gui_rv_plot_y");
	phoebe_parameter_add_option (par, "Primary RV");
	phoebe_parameter_add_option (par, "Secondary RV");
	phoebe_parameter_add_option (par, "Both RVs");
	phoebe_parameter_add_option (par, "Primary normalized RV");
	phoebe_parameter_add_option (par, "Secondary normalized RV");
	phoebe_parameter_add_option (par, "Primary EC corrections");
	phoebe_parameter_add_option (par, "Secondary EC correction");

	return status;
}

GUI_widget *gui_widget_new ()
{
	GUI_widget *widget = phoebe_malloc (sizeof (*widget));

	widget->name = NULL;
	widget->type = 0;
	widget->gtk  = NULL;
	widget->par  = NULL;
	widget->aux  = 0;

	return widget;
}

int gui_widget_free (GUI_widget *widget)
{
	if (!widget)
		return SUCCESS;

	if (widget->name)
		free (widget->name);

	free (widget);

	return SUCCESS;
}

int gui_widget_hookup (GUI_widget *widget, GtkWidget *gtk, int aux, GUI_widget_type type, char *name, PHOEBE_parameter *par)
{
	if (!widget) {
		printf ("*** a pointer to widget %s passed to gui_widget_hookup () is NULL!\n", name);
	/*
	 * A suggestion: create a phoebe_gui_errors.h and create an enum for these
	 * error codes.
	 */
		return /* ERROR_GUI_WIDGET_NOT_FOUND; */ -1;
	}

	widget->name = strdup (name);
	widget->gtk  = gtk;
	widget->aux  = aux;
	widget->type = type;
	widget->par  = par;

	return SUCCESS;
}

GUI_widget *gui_widget_lookup (char *name)
{
	unsigned int hash = gui_widget_hash (name);
	GUI_wt_bucket *bucket = GUI_wt->bucket[hash];

	while (bucket) {
		if (strcmp (bucket->widget->name, name) == 0) break;
		bucket = bucket->next;
	}

	if (!bucket) {
		printf ("*** widget lookup failure: %s not found.\n", name);
		return NULL;
	}

	return bucket->widget;
}

int gui_widget_add (char *name, GtkWidget *gtk, int aux, GUI_widget_type type, PHOEBE_parameter *par)
{
	GUI_widget *widget;

	if (!gtk) {
		printf ("*** widget %s passed to gui_widget_add () is NULL!\n", name);
		return -1;
	}

	widget = gui_widget_new ();
	gui_widget_hookup (widget, gtk, aux, type, name, par);
	gui_widget_commit (widget);

	return SUCCESS;
}

unsigned int gui_widget_hash (char *name)
{
	/*
	 * This is the hashing function for storing widgets into the widget
	 * table.
	 */
	unsigned int h = 0;
	unsigned char *w;

	for (w = (unsigned char *) name; *w != '\0'; w++)
		h = GUI_WT_HASH_MULTIPLIER * h + *w;

	return h % GUI_WT_HASH_BUCKETS;
}

int gui_widget_commit (GUI_widget *widget)
{
	int hash = gui_widget_hash (widget->name);
	GUI_wt_bucket *bucket = GUI_wt->bucket[hash];

	while (bucket) {
		if (strcmp (bucket->widget->name, widget->name) == 0) break;
		bucket = bucket->next;
	}

	if (bucket) {
		/* gui_error(widget already commited...) */
		return SUCCESS;
	}

	else {
		bucket = phoebe_malloc (sizeof (*bucket));

		bucket->widget = widget;
		bucket->next	 = GUI_wt->bucket[hash];
		GUI_wt->bucket[hash] = bucket;
	}

	return SUCCESS;
}

int gui_free_widgets ()
{
	/*
	 * This function frees all widgets from the widget table.
	 */

	int i;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		while (GUI_wt->bucket[i]) {
			bucket = GUI_wt->bucket[i];
			GUI_wt->bucket[i] = bucket->next;
			gui_widget_free (bucket->widget);
			free (bucket);
		}
	}
	free (GUI_wt);

	return SUCCESS;
}

int gui_get_value_from_widget (GUI_widget *widget)
{
	int status = 0;

	if (widget->par)
		printf ("\tparameter type: %s\n", phoebe_type_get_name (widget->par->type));
	else
		printf ("\tparameter type: n/a\n");

	if (GTK_IS_TREE_MODEL (widget->gtk)) {
		GtkTreeModel *model = GTK_TREE_MODEL (widget->gtk);
		GtkTreeIter iter;
		int index;
		bool state;

		printf ("\twidget type: tree model\n");

		state = gtk_tree_model_get_iter_first (model, &iter);

		while (state) {
			index = atoi (gtk_tree_model_get_string_from_iter (model, &iter));
			switch (widget->par->type) {
				case TYPE_INT_ARRAY: {
					int value;
					gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
					printf ("\tsetting value %d to %d\n", index, value);
					status = phoebe_parameter_set_value (widget->par, index, value);
				}
				break;
				case TYPE_BOOL_ARRAY: {
					bool value;
					gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
					printf ("\tsetting value %d to %d\n", index, value);
					status = phoebe_parameter_set_value (widget->par, index, value);
				}
				break;
				case TYPE_DOUBLE_ARRAY: {
					double value;
					gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
					printf ("\tsetting value %d to %lf\n", index, value);
					status = phoebe_parameter_set_value (widget->par, index, value);
				}
				break;
				case TYPE_STRING_ARRAY: {
					char *value;
					gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
					printf ("\tsetting value %d to %s\n", index, value);
					status = phoebe_parameter_set_value (widget->par, index, value);
				}
				break;
				default:
					/* change to phoebe_gui_error! */
					printf ("\t*** I'm not supposed to be here!\n");
					printf ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_TREE_VIEW_COLUMN block, GUI_WIDGET_VALUE block; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
			state = gtk_tree_model_iter_next (model, &iter);
		}
		return SUCCESS;
	}

	if (GTK_IS_SPIN_BUTTON (widget->gtk)) {
		printf ("\twidget type: spin button\n");
		switch (widget->type) {
			case GUI_WIDGET_VALUE: {
				switch (widget->par->type) {
					case TYPE_INT:
						printf ("\tsetting value to %d\n", gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON (widget->gtk)));
						status = phoebe_parameter_set_value (widget->par, gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON (widget->gtk)));
					break;
					case TYPE_DOUBLE:
						printf ("\tsetting value to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
						status = phoebe_parameter_set_value (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
					break;
					default:
						/* change to phoebe_gui_error! */
						printf ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->par->type switch; please report this!\n");
						return ERROR_EXCEPTION_HANDLER_INVOKED;
				}
			}
			break;
			case GUI_WIDGET_VALUE_MIN: {
				printf ("\tsetting min to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
				status = phoebe_parameter_set_lower_limit (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
			}
			break;
			case GUI_WIDGET_VALUE_MAX: {
				printf ("\tsetting max to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
				status = phoebe_parameter_set_upper_limit (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
			}
			break;
			case GUI_WIDGET_VALUE_STEP: {
				printf ("\tsetting step to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
				status = phoebe_parameter_set_step (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
			}
			default:
				/* change to phoebe_gui_error! */
				printf ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_SPIN_BUTTON block, par->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_ENTRY (widget->gtk)) {
		printf ("\twidget type: entry\n");
		printf ("\tsetting value to %s\n", gtk_entry_get_text (GTK_ENTRY (widget->gtk)));
		status = phoebe_parameter_set_value (widget->par, gtk_entry_get_text (GTK_ENTRY (widget->gtk)));
		return status;
	}

	if (GTK_IS_RADIO_BUTTON (widget->gtk)) {
		printf ("\twidget type: radio button\n");
		printf ("\thandler not yet implemented.\n");
		return SUCCESS;
	}

	if (GTK_IS_CHECK_BUTTON (widget->gtk)) {
		printf ("\twidget type: check button\n");
		switch (widget->type) {
			case GUI_WIDGET_VALUE:
				printf ("\tsetting value to %d\n", gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
				status = phoebe_parameter_set_value (widget->par, gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
			break;
			case GUI_WIDGET_SWITCH_TBA:
				printf ("\tsetting tba to %d\n", gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
				status = phoebe_parameter_set_tba (widget->par, gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
			break;
			default:
				printf ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_CHECK_BUTTON block, widget->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_COMBO_BOX (widget->gtk)) {
		printf ("\twidget type: combo box\n");
		if (gtk_combo_box_get_active((GtkComboBox*) widget->gtk) >= 0){
			printf ("\tsetting option to index %d, value %s\n", gtk_combo_box_get_active((GtkComboBox*) widget->gtk), strdup (widget->par->menu->option[gtk_combo_box_get_active((GtkComboBox*) widget->gtk)]));
			status = phoebe_parameter_set_value (widget->par, strdup (widget->par->menu->option[gtk_combo_box_get_active((GtkComboBox*) widget->gtk)]));
			return status;
		}
		else{
			printf ("\t*** nothing selected in combo.\n");
			return SUCCESS;
		}
	}

	if (GTK_IS_WIDGET (widget->gtk)) {
		printf ("\tI'm a widget without a parameter.\n");
		return SUCCESS;
	}

	printf ("\t*** I got where I am not supposed to be!!\n");
	printf ("\t*** exception handler invoked in gui_get_value_from_widget (); please report this!\n");

	return SUCCESS;
}

int gui_set_value_to_widget (GUI_widget *widget)
{
	int status = 0;

	if (widget->par)
		printf ("\tparameter type: %s\n", phoebe_type_get_name (widget->par->type));
	else
		printf ("\tparameter type: n/a\n");

	if (GTK_IS_TREE_MODEL (widget->gtk)) {
		GtkTreeModel *model = GTK_TREE_MODEL (widget->gtk);
		GtkTreeIter iter;
		int index;
		bool state;

		printf ("\twidget type: tree model\n");

		state = gtk_tree_model_get_iter_first (model, &iter);

		while (state) {
			index = atoi (gtk_tree_model_get_string_from_iter (model, &iter));
			switch (widget->par->type) {
				case TYPE_INT_ARRAY: {
					int value;
					status = phoebe_parameter_get_value (widget->par, index, &value);
					printf ("\tsetting value %d to %d\n", index, value);
					gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
				}
				break;
				case TYPE_BOOL_ARRAY: {
					bool value;
					status = phoebe_parameter_get_value (widget->par, index, &value);
					printf ("\tsetting value %d to %d\n", index, value);
					gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
				}
				break;
				case TYPE_DOUBLE_ARRAY: {
					double value;
					status = phoebe_parameter_get_value (widget->par, index, &value);
					printf ("\tsetting value %d to %lf\n", index, value);
					gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
				}
				break;
				case TYPE_STRING_ARRAY: {
					char *value;
					status = phoebe_parameter_get_value (widget->par, index, &value);
					printf ("\tsetting value %d to %s\n", index, value);
					gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
				}
				break;
				default:
					/* change to phoebe_gui_error! */
					printf ("\t*** I'm not supposed to be here!\n");
					printf ("\t*** exception handler invoked in gui_set_value_from_widget (), GTK_IS_TREE_MODEL block, GUI_WIDGET_VALUE block; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
			state = gtk_tree_model_iter_next (model, &iter);
		}
		return SUCCESS;
	}

	if (GTK_IS_SPIN_BUTTON (widget->gtk)) {
		printf ("\twidget type: spin button\n");
		switch (widget->type){
			case GUI_WIDGET_VALUE: {
				switch (widget->par->type) {
					case TYPE_INT: {
						int value;
						printf ("\tpar->type: int, widget->type: value\n");
						status = phoebe_parameter_get_value (widget->par, &value);
						printf ("\tsetting value to %d\n", value);
						gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
					}
					break;
					case TYPE_DOUBLE: {
						double value;
						printf ("\tpar->type: double, widget->type: value\n");
						status = phoebe_parameter_get_value (widget->par, &value);
						printf ("\tsetting value to %lf\n", value);
						gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
					}
					break;
					default:
					/* change to phoebe_gui_error! */
					printf ("\t*** I'm not supposed to be here!\n");
					printf ("\t*** exception handler invoked in gui_set_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->par->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
				}
			}
			break;
			case GUI_WIDGET_VALUE_MIN: {
				double value;
				status = phoebe_parameter_get_lower_limit(widget->par, &value);
				printf("\tsetting min to %lf\n", value);
				gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
			}
			break;
			case GUI_WIDGET_VALUE_MAX: {
				double value;
				status = phoebe_parameter_get_upper_limit(widget->par, &value);
				printf("\tsetting max to %lf\n", value);
				gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
			}
			break;
			case GUI_WIDGET_VALUE_STEP: {
				double value;
				status = phoebe_parameter_get_step(widget->par, &value);
				printf("\tsetting step to %lf\n", value);
				gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
			}
			break;
			default:
				/* change to phoebe_gui_error! */
				printf ("\t*** I'm not supposed to be here!\n");
				printf ("\t*** exception handler invoked in gui_set_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_ENTRY (widget->gtk)){
		printf ("\twidget type: entry\n");
		char *value;
		status = phoebe_parameter_get_value(widget->par, &value);
		printf ("\tsetting value to %s\n", value);
		gtk_entry_set_text(GTK_ENTRY(widget->gtk), value);
		return status;
	}

	if (GTK_IS_RADIO_BUTTON (widget->gtk)){
		printf ("\twidget type: radio button\n");
		printf ("\t*** handler not yet implemented.\n");
		return status;
	}

	if (GTK_IS_CHECK_BUTTON (widget->gtk)){
		printf ("\twidget type: check button\n");
		switch(widget->type){
			bool value;
			case GUI_WIDGET_VALUE: {
				status = phoebe_parameter_get_value(widget->par, &value);
				printf ("\tsetting value to %d\n", value);
				gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget->gtk), value);
			}
			break;
			case GUI_WIDGET_SWITCH_TBA: {
				status = phoebe_parameter_get_tba(widget->par, &value);
				printf ("\tsetting value to %d\n", value);
				gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget->gtk), value);
			}
			break;
			default:
				/* change to phoebe_gui_error! */
				printf ("\t*** exception handler invoked in sui_set_value_to_widget (), GTK_IS_CHECK_BUTTON block, widget->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_COMBO_BOX(widget->gtk)) {
		if (widget->par){
			printf ("\twidget type: combo box\n");
			char *value;
			status = phoebe_parameter_get_value(widget->par, &value);
			// go through all the values in the combo, compare with value; when same, set it!
			if(strlen(value) > 0){
				int i;
				for (i = 0; i < widget->par->menu->optno; i++){
					gtk_combo_box_set_active(GTK_COMBO_BOX(widget->gtk), i);
					if ( strcmp(gtk_combo_box_get_active_text(GTK_COMBO_BOX(widget->gtk)), value ) == 0 ){
						printf ("\tsetting option to %s\n", value);
						break;
					}
				}
			}
			return status;
		}
	}

	if(GTK_IS_WIDGET(widget->gtk)){
		printf ("\tI'm a widget without a parameter.\n");
		return status;
	}

	printf ("\t*** I got where I am not supposed to be!!\n");
	printf ("\t*** exception handler invoked in gui_set_value_to_widget (); please report this!\n");

	return SUCCESS;
}

int gui_get_values_from_widgets ()
{
	printf("\n\n******** Entering gui_get_values_from_widgets!******* \n\n");

	int i, status;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		bucket = GUI_wt->bucket[i];
		while (bucket) {
			printf ("processing widget %s:\n", bucket->widget->name);
			status = gui_get_value_from_widget (bucket->widget);
			printf ("\tstatus: %s", phoebe_error (status));
			bucket = bucket->next;
		}
	}

	gui_fill_sidesheet_fit_treeview ();
	gui_fill_sidesheet_res_treeview ();

	return SUCCESS;
}

int gui_set_values_to_widgets ()
{
	printf("\n\n ******* Entering gui_set_values_to_widgets!******* \n\n");

	int i, status;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		bucket = GUI_wt->bucket[i];
		while (bucket) {
			printf ("processing widget %s: \n", bucket->widget->name);
			status = gui_set_value_to_widget (bucket->widget);
			printf ("%s", phoebe_error (status));
			bucket = bucket->next;
		}
	}

	gui_fill_sidesheet_fit_treeview ();
	gui_fill_sidesheet_res_treeview ();

	return SUCCESS;
}
