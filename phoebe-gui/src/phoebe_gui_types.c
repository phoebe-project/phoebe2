#include <glade/glade.h>
#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"
#include "phoebe_gui_treeviews.h"

int gui_init_widgets (GladeXML* phoebe_window)
{
	/*
	 * This function hooks all widgets to the parameters and adds them to the
	 * widget hash table.
	 */

	int i;

	PHOEBE_parameter *par;

	GUI_wt = phoebe_malloc(sizeof(GUI_widget_table));
	for(i=0; i<GUI_WT_HASH_BUCKETS; i++)GUI_wt->bucket[i]=NULL;

	/* *************************** GUI Parameters **************************** */

	phoebe_parameter_add ("gui_ld_model_autoupdate",	"Automatically update LD model",	KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_fitt_method",			"Fitting method",					KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_synthetic",		"Plot synthetic LC",				KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_lc_plot_observed",		"Plot observed LC",					KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_verticesno",		"Number of vertices for LC",		KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_INT,		100);
	phoebe_parameter_add ("gui_lc_plot_obsmenu",		"Select observed LC",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_aliasing",		"Turn on data aliasing",			KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_lc_plot_residuals",		"Plot residuals",					KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_x",				"X-axis of LC plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_y",				"Y-axis of LC plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
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
	phoebe_parameter_add ("gui_rv_plot_x",				"X-axis of RV plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_y",				"Y-axis of RV plot",				KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_phstart",		"Phase start",						KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_rv_plot_phend",			"Phase end",						KIND_PARAMETER,	NULL, 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_rv_plot_offset",			"Offset level",						KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_zoom",			"Zoom level",						KIND_MENU,		NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_coarse",			"Coarse grid",						KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_fine",			"Coarse grid",						KIND_SWITCH,	NULL, 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);

	/* *************************    Data Widgets   **************************** */

	gui_widget_add ("phoebe_data_star_name_entry", 						glade_xml_get_widget(phoebe_window, "phoebe_data_star_name_entry"), 									GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_name"));
	gui_widget_add ("phoebe_data_star_model_combobox", 					glade_xml_get_widget(phoebe_window, "phoebe_data_star_model_combobox"), 								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_model"));

	gui_widget_add ("phoebe_data_lc_filename",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_FILENAME), 			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filename"));
	gui_widget_add ("phoebe_data_lc_sigma",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_SIGMA), 	  			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_sigma"));
	gui_widget_add ("phoebe_data_lc_filter",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_FILTER), 			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filter"));
	gui_widget_add ("phoebe_data_lc_indep",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_ITYPE_STR), 	  		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indep"));
	gui_widget_add ("phoebe_data_lc_dep",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_DTYPE_STR), 			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_dep"));
	gui_widget_add ("phoebe_data_lc_wtype",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_WTYPE_STR), 			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indweight"));
	gui_widget_add ("phoebe_data_lc_active",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_ACTIVE), 			GUI_WIDGET_VALUE	, 	phoebe_parameter_lookup ("phoebe_lc_active"));

	gui_widget_add ("phoebe_data_rv_filename",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_FILENAME),			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_filename"));
	gui_widget_add ("phoebe_data_rv_sigma",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_SIGMA),				GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_sigma"));
	gui_widget_add ("phoebe_data_rv_filter",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_FILTER),				GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_filter"));
	gui_widget_add ("phoebe_data_rv_indep",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_ITYPE_STR),			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_indep"));
	gui_widget_add ("phoebe_data_rv_dep",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_DTYPE_STR),			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_dep"));
	gui_widget_add ("phoebe_data_rv_wtype",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_WTYPE_STR),			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_indweight"));
	gui_widget_add ("phoebe_data_rv_active",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_rv_treeview, RV_COL_ACTIVE),				GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_active"));

	gui_widget_add ("phoebe_data_options_time_radiobutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_options_time_radiobutton"), 							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_indep"));
	gui_widget_add ("phoebe_data_options_bins_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_options_bins_checkbutton"), 							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins_switch"));
	gui_widget_add ("phoebe_data_options_binsno_spinbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_options_binsno_spinbutton"), 							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins"));

	gui_widget_add ("phoebe_data_lcoptions_mag_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_lcoptions_mag_spinbutton"), 							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_mnorm"));
	gui_widget_add ("phoebe_data_rvoptions_psepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_psepe_checkbutton"), 						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"));
	gui_widget_add ("phoebe_data_rvoptions_ssepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_ssepe_checkbutton"), 						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"));

	/* **********************    Parameters Widgets   ************************* */

	par = phoebe_parameter_lookup ("phoebe_hjd0");
	gui_widget_add ("phoebe_para_eph_hjd0_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_hjd0adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0adjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_hjd0step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_hjd0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0max_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_hjd0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0min_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_period");
	gui_widget_add ("phoebe_para_eph_period_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_period_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_periodadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_periodstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_periodmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_periodmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_dpdt");
	gui_widget_add ("phoebe_para_eph_dpdt_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdt_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_dpdtadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtadjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_dpdtstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_dpdtmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtmax_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_dpdtmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtmin_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pshift");
	gui_widget_add ("phoebe_para_eph_pshift_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshift_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_eph_pshiftadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_eph_pshiftstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_eph_pshiftmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_eph_pshiftmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_sma");
	gui_widget_add ("phoebe_para_sys_sma_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_sma_spinbutton"),									GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_smaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smaadjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_smastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smastep_spinbutton"), 								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_smamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smamax_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_smamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smamin_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_rm");
	gui_widget_add ("phoebe_para_sys_rm_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rm_spinbutton"), 									GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_rmadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmadjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_rmstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmstep_spinbutton"), 								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_rmmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmmax_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_rmmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmmin_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_vga");
	gui_widget_add ("phoebe_para_sys_vga_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vga_spinbutton"), 									GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_vgaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgaadjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_vgastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgastep_spinbutton"), 								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_vgamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgamax_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_vgamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgamin_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_incl");
	gui_widget_add ("phoebe_para_sys_incl_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_incl_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_sys_incladjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_incladjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_sys_inclstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_sys_inclmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclmax_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_sys_inclmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclmin_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_perr0");
	gui_widget_add ("phoebe_para_orb_perr0_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_perr0adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_perr0step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_perr0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_perr0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_dperdt");
	gui_widget_add ("phoebe_para_orb_dperdt_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdt_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_dperdtadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_dperdtstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_dperdtmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_dperdtmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_ecc");
	gui_widget_add ("phoebe_para_orb_ecc_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_ecc_spinbutton"), 									GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_eccadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccadjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_eccstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccstep_spinbutton"), 								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_eccmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccmax_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_eccmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccmin_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_f1");
	gui_widget_add ("phoebe_para_orb_f1_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1_spinbutton"), 									GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_f1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1adjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_f1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1step_spinbutton"), 								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_f1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1max_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_f1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1min_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_f2");
	gui_widget_add ("phoebe_para_orb_f2_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2_spinbutton"), 									GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_orb_f2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2adjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_orb_f2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2step_spinbutton"), 								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_orb_f2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2max_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_orb_f2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2min_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_teff1");
	gui_widget_add ("phoebe_para_comp_tavh_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavh_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_tavhadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_tavhstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_tavhmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_tavhmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_teff2");
	gui_widget_add ("phoebe_para_comp_tavc_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavc_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_tavcadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_tavcstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_tavcmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_tavcmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pot1");
	gui_widget_add ("phoebe_para_comp_phsv_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsv_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_phsvadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_phsvstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_phsvmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_phsvmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pot2");
	gui_widget_add ("phoebe_para_comp_pcsv_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsv_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_pcsvadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvadjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_pcsvstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvstep_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_pcsvmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvmax_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_pcsvmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvmin_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_met1");
	gui_widget_add ("phoebe_para_comp_met1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_met1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_met1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_met1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_met1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_met2");
	gui_widget_add ("phoebe_para_comp_met2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_met2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_met2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_met2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_met2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_logg1");
	gui_widget_add ("phoebe_para_comp_logg1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg1_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_logg1adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg1adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_logg1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg1step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_logg1max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg1max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_logg1min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg1min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_logg2");
	gui_widget_add ("phoebe_para_comp_logg2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg2_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_comp_logg2adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg2adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_comp_logg2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg2step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_comp_logg2max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg2max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_comp_logg2min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg2min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_alb1");
	gui_widget_add ("phoebe_para_surf_alb1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_alb1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_alb1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_alb1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_alb1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_alb2");
	gui_widget_add ("phoebe_para_surf_alb2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_alb2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2adjust_checkbutton"), 						GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_alb2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_alb2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2max_spinbutton"), 							GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_alb2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2min_spinbutton"), 							GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_grb1");
	gui_widget_add ("phoebe_para_surf_gr1_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_gr1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1adjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_gr1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_gr1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1max_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_gr1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1min_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_grb2");
	gui_widget_add ("phoebe_para_surf_gr2_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2_spinbutton"), 								GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_surf_gr2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2adjust_checkbutton"), 							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_surf_gr2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2step_spinbutton"), 							GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_surf_gr2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2max_spinbutton"), 								GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_para_surf_gr2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2min_spinbutton"), 								GUI_WIDGET_VALUE_MIN, 	par);

    gui_widget_add ("phoebe_para_lum_levels_levweight", 				(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_levels_treeview, LC_COL_LEVWEIGHT), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_levweight"));

    par = phoebe_parameter_lookup ("phoebe_hla");
    gui_widget_add ("phoebe_para_lum_levels_prim",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_levels_treeview, LC_COL_HLA), 		GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_lum_levels_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primadjust_checkbutton"),					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_lum_levels_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primstep_spinbutton"),						GUI_WIDGET_VALUE_STEP, 	par);

	par = phoebe_parameter_lookup ("phoebe_cla");
	gui_widget_add ("phoebe_para_lum_levels_sec",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_levels_treeview, LC_COL_CLA), 		GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_para_lum_levels_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secadjust_checkbutton"),					GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_lum_levels_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secstep_spinbutton"),						GUI_WIDGET_VALUE_STEP, 	par);

    par = phoebe_parameter_lookup ("phoebe_el3");
    gui_widget_add ("phoebe_para_lum_el3",								(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_el3_treeview, LC_COL_EL3),			GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_lum_el3ajdust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3ajdust_checkbutton"),							GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_para_lum_el3step_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3step_spinbutton"),								GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_para_lum_el3_percent_radiobutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_percent_radiobutton"),							GUI_WIDGET_VALUE, 		par);

	par = phoebe_parameter_lookup ("phoebe_opsf");
	gui_widget_add ("phoebe_para_lum_el3_opacity",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_el3_treeview, LC_COL_OPSF),			GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_lum_el3_opacityadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacityadjust_checkbutton"),					GUI_WIDGET_SWITCH_TBA,  par);
	gui_widget_add ("phoebe_para_lum_el3_opacitystep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacitystep_spinbutton"),						GUI_WIDGET_VALUE_STEP, 	par);

	par = phoebe_parameter_lookup ("phoebe_extinction");
	gui_widget_add ("phoebe_para_lum_el3_ext",							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_el3_treeview, LC_COL_EXTINCTION),		GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_lum_el3_extadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extadjust_checkbutton"),						GUI_WIDGET_SWITCH_TBA,  par);
	gui_widget_add ("phoebe_para_lum_el3_extstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extstep_spinbutton"),							GUI_WIDGET_VALUE_STEP,  par);

	gui_widget_add ("phoebe_para_lum_atmospheres_prim_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_prim_checkbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm1_switch"));
	gui_widget_add ("phoebe_para_lum_atmospheres_sec_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_sec_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm2_switch"));

	gui_widget_add ("phoebe_para_lum_options_reflections_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_reflections_checkbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_switch"));
	gui_widget_add ("phoebe_para_lum_options_reflections_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_reflections_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_reflections"));
	gui_widget_add ("phoebe_para_lum_options_decouple_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_decouple_checkbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_usecla_switch"));

	gui_widget_add ("phoebe_para_lum_noise_lcscatter_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_lcscatter_checkbutton"),						GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_synscatter_switch"));
	gui_widget_add ("phoebe_para_lum_noise_sigma_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_sigma_spinbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_sigma"));
	gui_widget_add ("phoebe_para_lum_noise_seed_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_seed_spinbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_seed"));
	gui_widget_add ("phoebe_para_lum_noise_lcscatter_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_noise_lcscatter_combobox"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_levweight"));

	gui_widget_add ("phoebe_para_ld_model_combobox",					glade_xml_get_widget(phoebe_window, "phoebe_para_ld_model_combobox"),									GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_model"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_primx_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_primx_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol1"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_primy_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_primx_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol1"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_secx_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_secx_spinbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol2"));
	gui_widget_add ("phoebe_para_ld_bolcoefs_secy_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_secx_spinbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol2"));

	par = phoebe_parameter_lookup ("phoebe_ld_lcx1");
	gui_widget_add ("phoebe_para_ld_lccoefs_primx",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_ld_treeview, LC_COL_X1),				GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_ld_lccoefs_primy",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_ld_treeview, LC_COL_Y1),				GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_lcy1"));
	gui_widget_add ("phoebe_para_ld_lccoefs_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primadjust_checkbutton"),					GUI_WIDGET_SWITCH_TBA,	par);
	gui_widget_add ("phoebe_para_ld_lccoefs_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primstep_spinbutton"),						GUI_WIDGET_VALUE_STEP,	par);

	par = phoebe_parameter_lookup ("phoebe_ld_lcx2");
	gui_widget_add ("phoebe_para_ld_lccoefs_secx",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_ld_treeview, LC_COL_X2),				GUI_WIDGET_VALUE,		par);
	gui_widget_add ("phoebe_para_ld_lccoefs_secy",						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_para_lc_ld_treeview, LC_COL_Y2),				GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_lcy2"));
	gui_widget_add ("phoebe_para_ld_lccoefs_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secadjust_checkbutton"),					GUI_WIDGET_SWITCH_TBA,	par);
	gui_widget_add ("phoebe_para_ld_lccoefs_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secstep_spinbutton"),						GUI_WIDGET_VALUE_STEP,	par);

	/* ***********************    Fitting Widgets   ************************* */

	gui_widget_add ("phoebe_fitt_parameters_finesize1_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_finesize1_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize1"));
	gui_widget_add ("phoebe_fitt_parameters_finesize2_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_finesize2_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize2"));
	gui_widget_add ("phoebe_fitt_parameters_coarsize1_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_coarsize1_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize1"));
	gui_widget_add ("phoebe_fitt_parameters_coarsize2_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_coarsize2_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize2"));
	gui_widget_add ("phoebe_fitt_parameters_lambda_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_lambda_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_dc_lambda"));

	/* *************************    GUI Widgets   *************************** */

	gui_widget_add ("phoebe_para_ld_model_autoupdate_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_model_autoupdate_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_ld_model_autoupdate"));
	gui_widget_add ("phoebe_fitt_method_combobox",						glade_xml_get_widget(phoebe_window, "phoebe_fitt_method_combobox"),										GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_fitt_method"));
	gui_widget_add ("phoebe_lc_plot_options_syn_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_syn_checkbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_synthetic"));
	gui_widget_add ("phoebe_lc_plot_options_obs_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_obs_checkbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_observed"));
	gui_widget_add ("phoebe_lc_plot_options_vertices_no_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_vertices_no_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_verticesno"));
	gui_widget_add ("phoebe_lc_plot_options_obs_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_obs_combobox"),								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_obsmenu"));
	gui_widget_add ("phoebe_lc_plot_options_alias_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_alias_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_aliasing"));
	gui_widget_add ("phoebe_lc_plot_options_residuals_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_residuals_checkbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_residuals"));
	gui_widget_add ("phoebe_lc_plot_options_x_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_x_combobox"),								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_x"));
	gui_widget_add ("phoebe_lc_plot_options_y_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_y_combobox"),								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_y"));
	gui_widget_add ("phoebe_lc_plot_options_phstart_spinbutton",		glade_xml_get_widget(phoebe_window, "gui_phoebe_lc_plot_options_phstart_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_phstart"));
	gui_widget_add ("phoebe_lc_plot_options_phend_spinbutton",			glade_xml_get_widget(phoebe_window, "gui_phoebe_lc_plot_options_phend_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_phend"));
	gui_widget_add ("phoebe_lc_plot_controls_offset_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_offset_combobox"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_offset"));
	gui_widget_add ("phoebe_lc_plot_controls_zoom_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_zoom_combobox"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_zoom"));
	gui_widget_add ("phoebe_lc_plot_controls_coarse_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_coarse_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_coarse"));
	gui_widget_add ("phoebe_lc_plot_controls_fine_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_fine_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_fine"));
	gui_widget_add ("phoebe_rv_plot_options_syn_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_syn_checkbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_synthetic"));
	gui_widget_add ("phoebe_rv_plot_options_obs_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_obs_checkbutton"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_observed"));
	gui_widget_add ("phoebe_rv_plot_options_vertices_no_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_vertices_no_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_verticesno"));
	gui_widget_add ("phoebe_rv_plot_options_obs_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_obs_combobox"),								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_obsmenu"));
	gui_widget_add ("phoebe_rv_plot_options_alias_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_alias_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_alias"));
	gui_widget_add ("phoebe_rv_plot_options_residuals_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_residuals_checkbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_residuals"));
	gui_widget_add ("phoebe_rv_plot_options_x_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_x_combobox"),								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_x"));
	gui_widget_add ("phoebe_rv_plot_options_y_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_y_combobox"),								GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_y"));
	gui_widget_add ("phoebe_rv_plot_options_phstart_spinbutton",		glade_xml_get_widget(phoebe_window, "gui_phoebe_rv_plot_options_phstart_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_phstart"));
	gui_widget_add ("phoebe_rv_plot_options_phend_spinbutton",			glade_xml_get_widget(phoebe_window, "gui_phoebe_rv_plot_options_phend_spinbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_phend"));
	gui_widget_add ("phoebe_rv_plot_controls_offset_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_offset_combobox"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_offset"));
	gui_widget_add ("phoebe_rv_plot_scrolledwindow",					glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_scrolledwindow"),									GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_rv_plot_controls_zoom_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_zoom_combobox"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_zoom"));
	gui_widget_add ("phoebe_rv_plot_controls_coarse_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_coarse_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_coarse"));
	gui_widget_add ("phoebe_rv_plot_controls_fine_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_fine_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_fine"));

	/* ************************    GUI Containers   ************************* */

	gui_widget_add ("phoebe_lc_plot_scrolledwindow",					glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_scrolledwindow"),									GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_sidesheet_parent_table",					glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_parent_table"),									GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_sidesheet_vbox",							glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_vbox"),											GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_lc_plot_table",								glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_table"),											GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_lc_plot_parent_table",						glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_parent_table"),										GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_rv_plot_table",								glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_table"),											GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_rv_plot_parent_table",						glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_parent_table"),										GUI_WIDGET_VALUE, 		NULL);

	gui_widget_add ("phoebe_fitt_fitting_frame",						glade_xml_get_widget(phoebe_window, "phoebe_fitt_fitting_frame"),										GUI_WIDGET_VALUE, 		NULL);
	gui_widget_add ("phoebe_fitt_fitting_parent_table",					glade_xml_get_widget(phoebe_window, "phoebe_fitt_fitting_parent_table"),								GUI_WIDGET_VALUE, 		NULL);

	return SUCCESS;
}

GUI_widget *gui_widget_new ()
{
	GUI_widget *widget = phoebe_malloc (sizeof (*widget));

	widget->name = NULL;
	widget->type = 0;
	widget->gtk  = NULL;
	widget->par  = NULL;

	return widget;
}

int gui_widget_free (GUI_widget *widget)
{
	if (!widget)
		return SUCCESS;

	if (widget->name) free (widget->name);
	free (widget);

	return SUCCESS;
}

int gui_widget_hookup (GUI_widget *widget, GtkWidget *gtk, GUI_widget_type type, char *name, PHOEBE_parameter *par)
{
	if (!widget)
	/*
	 * A suggestion: create a phoebe_gui_errors.h and create an enum for these
	 * error codes.
	 */
		return /* ERROR_GUI_WIDGET_NOT_FOUND; */ -1;

	widget->name = strdup (name);
	widget->gtk  = gtk;
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

	if (!bucket) return NULL;
	return bucket->widget;
}

int gui_widget_add (char *name, GtkWidget *gtk, GUI_widget_type type, PHOEBE_parameter *par)
{
	GUI_widget *widget = gui_widget_new();

	if (!gtk)
		return -1;

	gui_widget_hookup (widget, gtk, type, name, par);

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

	else{
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
	//PHOEBE_parameter *par = widget->par;

	if (GTK_IS_SPIN_BUTTON (widget->gtk)) {
		switch (widget->par->type) {
			case TYPE_INT: {
				status = phoebe_parameter_set_value (widget->par, gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(widget->gtk)));
			}
			break;
			case TYPE_DOUBLE:
			case TYPE_DOUBLE_ARRAY: {
				switch(widget->type){
					case GUI_WIDGET_VALUE:{
						status = phoebe_parameter_set_value (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON(widget->gtk)));
					}
					break;
					case GUI_WIDGET_VALUE_STEP: {
						status = phoebe_parameter_set_step (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON(widget->gtk)));
					}
					break;
					case GUI_WIDGET_VALUE_MIN: {
						status = phoebe_parameter_set_lower_limit (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON(widget->gtk)));
					}
					break;
					case GUI_WIDGET_VALUE_MAX: {
						status = phoebe_parameter_set_upper_limit (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON(widget->gtk)));
					}
					break;
					default:
						/* change to phoebe_gui_error! */
						printf ("exception handler invoked in gui_get_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->type switch; please report this!\n");
						return ERROR_EXCEPTION_HANDLER_INVOKED;
				}
			}
			break;
			default:
				/* change to phoebe_gui_error! */
				printf ("exception handler invoked in gui_get_value_from_widget (), GTK_IS_SPIN_BUTTON block, par->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_ENTRY (widget->gtk)) {
		status = phoebe_parameter_set_value (widget->par, gtk_entry_get_text (GTK_ENTRY(widget->gtk)));
		return status;
	}

	if (GTK_IS_RADIO_BUTTON (widget->gtk)) {
		printf ("not supported yet!\n");
		return SUCCESS;
	}

	if (GTK_IS_CHECK_BUTTON (widget->gtk)) {
		switch (widget->type) {
			case GUI_WIDGET_VALUE: {
				status = phoebe_parameter_set_value (widget->par, gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(widget->gtk)));
			}
			break;
			case GUI_WIDGET_SWITCH_TBA: {
				status = phoebe_parameter_set_tba (widget->par, gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(widget->gtk)));
			}
			break;
			default:
				/* change to phoebe_gui_error! */
				printf ("exception handler invoked in gui_get_value_from_widget (), GTK_IS_CHECK_BUTTON block, widget->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_COMBO_BOX(widget->gtk)) {
		printf ("not supported yet!\n");
		return SUCCESS;
	}

	if (GTK_IS_TREE_VIEW_COLUMN (widget->gtk)) {
        GtkTreeViewColumn *column = (GtkTreeViewColumn*) widget->gtk;
        int column_id = GPOINTER_TO_UINT (g_object_get_data ((GObject*) column, "column_id"));
        GtkTreeModel *model = gtk_tree_view_get_model ((GtkTreeView*) g_object_get_data ((GObject*) column, "parent_tree"));

        GtkTreeIter iter;
        int index = 0;
        int valid = gtk_tree_model_get_iter_first (model, &iter);

        while (valid) {
			switch (widget->type) {
				case GUI_WIDGET_VALUE: {
					switch (widget->par->type) {
						case TYPE_INT_ARRAY: {
							int value;
							gtk_tree_model_get (model, &iter, column_id, &value, -1);
							status = phoebe_parameter_set_value (widget->par, index, value);
						}
						break;
						case TYPE_DOUBLE_ARRAY: {
							double value;
							gtk_tree_model_get (model, &iter, column_id, &value, -1);
                    		status = phoebe_parameter_set_value (widget->par, index, value);
						}
						break;
						case TYPE_STRING_ARRAY: {
							char *value;
							gtk_tree_model_get (model, &iter, column_id, &value, -1);
                    		status = phoebe_parameter_set_value (widget->par, index, value);
						}
						break;
						case TYPE_BOOL_ARRAY: {
							bool value;
							gtk_tree_model_get (model, &iter, column_id, &value, -1);
                    		status = phoebe_parameter_set_value (widget->par, index, value);
						}
						break;
						default:
							/* change to phoebe_gui_error! */
							printf ("I'm not supposed to be here!\n");
							printf ("exception handler invoked in gui_get_value_from_widget (), GTK_IS_TREE_VIEW_COLUMN block, GUI_WIDGET_VALUE block; please report this!\n");
							return ERROR_EXCEPTION_HANDLER_INVOKED;
					}
                }
                break;
                case GUI_WIDGET_SWITCH_TBA: {
                    bool tba;
                    gtk_tree_model_get(model, &iter, column_id, &tba, -1);
                    status = phoebe_parameter_set_tba (widget->par, tba);
                }
                break;
                case GUI_WIDGET_VALUE_STEP: {
                    double value;
                    gtk_tree_model_get(model, &iter, column_id, &value, -1);
                    status = phoebe_parameter_set_step (widget->par, value);
                }
                break;
                case GUI_WIDGET_VALUE_MIN: {
                    double value;
                    gtk_tree_model_get(model, &iter, column_id, &value, -1);
                    status = phoebe_parameter_set_lower_limit (widget->par, value);
                }
                break;
                case GUI_WIDGET_VALUE_MAX: {
                    double value;
                    gtk_tree_model_get (model, &iter, column_id, &value, -1);
                    status = phoebe_parameter_set_upper_limit (widget->par, value);
                }
                break;
				default:
					/* change to phoebe_gui_error! */
					printf ("I'm not supposed to be here!\n");
					printf ("exception handler invoked in gui_get_value_from_widget (), GTK_IS_TREE_VIEW_COLUMN block, widget->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
            }
			valid = gtk_tree_model_iter_next (model, &iter);
			index ++;
        }
		return status;
	}

	if(GTK_IS_WIDGET(widget->gtk)){
		printf ("I'm a widget without a parameter.\n");
		return SUCCESS;
	}

	printf ("I got where I am not supposed to be!!\n");
	return SUCCESS;
}

int gui_set_value_to_widget(GUI_widget *widget)
{
	int status = 0;

	if (GTK_IS_SPIN_BUTTON (widget->gtk)){
		switch(widget->par->type){
			case TYPE_INT: {
				int value;
				status = phoebe_parameter_get_value(widget->par, &value);
				gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
			}
			break;
			case TYPE_DOUBLE:
			case TYPE_DOUBLE_ARRAY: {
				double value;
				switch(widget->type){
					case GUI_WIDGET_VALUE: {
						status = phoebe_parameter_get_value(widget->par, &value);
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
					}
					break;
					case GUI_WIDGET_VALUE_STEP: {
						status = phoebe_parameter_get_step(widget->par, &value);
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
					}
					break;
					case GUI_WIDGET_VALUE_MIN: {
						status = phoebe_parameter_get_lower_limit(widget->par, &value);
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
					}
					break;
					case GUI_WIDGET_VALUE_MAX: {
						status = phoebe_parameter_get_upper_limit(widget->par, &value);
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(widget->gtk), value);
					}
					break;
					default:
						/* change to phoebe_gui_error! */
						printf ("exception handler invoked in gui_set_value_to_widget (), GTK_IS_SPIN_BUTTON block, widget->type switch; please report this!\n");
						return ERROR_EXCEPTION_HANDLER_INVOKED;
				}
			}
			break;
			default:
				/* change to phoebe_gui_error! */
				printf ("exception handler invoked in gui_set_value_to_widget (), GTK_IS_SPIN_BUTTON block, widget->par->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_ENTRY (widget->gtk)){
		char *value;
		status = phoebe_parameter_get_value(widget->par, &value);
		gtk_entry_set_text(GTK_ENTRY(widget->gtk), value);
		return status;
	}

	if (GTK_IS_RADIO_BUTTON (widget->gtk)){
		printf ("not supported yet!\n");
		return status;
	}

	if (GTK_IS_CHECK_BUTTON (widget->gtk)){
		switch(widget->type){
			bool value;
			case GUI_WIDGET_VALUE: {
				status = phoebe_parameter_get_value(widget->par, &value);
				gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget->gtk), value);
			}
			break;
			case GUI_WIDGET_SWITCH_TBA: {
				status = phoebe_parameter_get_tba(widget->par, &value);
				gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget->gtk), value);
			}
			break;
			default:
				/* change to phoebe_gui_error! */
				printf ("exception handler invoked in gui_set_value_to_widget (), GTK_IS_CHECK_BUTTON block, widget->type switch; please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
		return status;
	}

	if (GTK_IS_COMBO_BOX(widget->gtk)) {
		printf ("not supported yet!\n");
		return status;
	}

	if (GTK_IS_TREE_VIEW_COLUMN (widget->gtk)) {
		GtkTreeViewColumn *column = (GtkTreeViewColumn*) widget->gtk;
        int column_id = GPOINTER_TO_UINT (g_object_get_data ((GObject*) column, "column_id"));
        GtkTreeModel *model = gtk_tree_view_get_model ((GtkTreeView*) g_object_get_data ((GObject*) column, "parent_tree"));

        GtkTreeIter iter;
        int index = 0;
        int valid = gtk_tree_model_get_iter_first (model, &iter);

        while (valid) {
			switch (widget->type) {
				case GUI_WIDGET_VALUE: {
					switch (widget->par->type) {
						case TYPE_INT_ARRAY: {
							int value;
							status = phoebe_parameter_get_value (widget->par, index, &value);
							gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
						}
						break;
						case TYPE_DOUBLE_ARRAY: {
							double value;
							status = phoebe_parameter_get_value (widget->par, index, &value);
                    		gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
						}
						break;
						case TYPE_STRING_ARRAY: {
							char *value;
							status = phoebe_parameter_get_value (widget->par, index, &value);
                    		gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
						}
						break;
						case TYPE_BOOL_ARRAY: {
							bool value;
							status = phoebe_parameter_get_value (widget->par, index, &value);
                    		gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
						}
						break;
						default:
							/* change to phoebe_gui_error! */
							printf ("I'm not supposed to be here!\n");
							printf ("exception handler invoked in gui_set_value_to_widget (), GTK_IS_TREE_VIEW_COLUMN block, GUI_WIDGET_VALUE block; please report this!\n");
							return ERROR_EXCEPTION_HANDLER_INVOKED;
					}
                }
                break;
                case GUI_WIDGET_SWITCH_TBA: {
                    bool tba;
                    status = phoebe_parameter_get_tba (widget->par, &tba);
					gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, tba, -1);
                }
                break;
                case GUI_WIDGET_VALUE_STEP: {
                    double value;
                    status = phoebe_parameter_get_step (widget->par, &value);
					gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
                }
                break;
                case GUI_WIDGET_VALUE_MIN: {
                    double value;
                    status = phoebe_parameter_get_lower_limit (widget->par, &value);
					gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
                }
                break;
                case GUI_WIDGET_VALUE_MAX: {
                    double value;
                    status = phoebe_parameter_get_upper_limit (widget->par, &value);
					gtk_list_store_set (GTK_LIST_STORE(model), &iter, column_id, value, -1);
                }
                break;
				default:
					/* change to phoebe_gui_error! */
					printf ("I'm not supposed to be here!\n");
					printf ("exception handler invoked in gui_set_value_to_widget (), GTK_IS_TREE_VIEW_COLUMN block, widget->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
            }
			valid = gtk_tree_model_iter_next (model, &iter);
			index ++;
        }
		return status;
	}

	if(GTK_IS_WIDGET(widget->gtk)){
		printf ("I'm a widget without a parameter.\n");
		return status;
	}

	printf ("I got where I am not supposed to be!!\n");
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
			printf ("processing widget %s: ", bucket->widget->name);
			status = gui_get_value_from_widget (bucket->widget);
			printf ("%s", phoebe_error (status));
			bucket = bucket->next;
		}
	}

	return SUCCESS;
}

int gui_set_values_to_widgets ()
{
	printf("\n\n******** Entering gui_set_values_to_widgets!******* \n\n");

	int i, status;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		bucket = GUI_wt->bucket[i];
		while (bucket) {
			printf ("processing widget %s: ", bucket->widget->name);
			status = gui_set_value_to_widget (bucket->widget);
			printf ("%s", phoebe_error (status));
			bucket = bucket->next;
		}
	}

	return SUCCESS;
}
