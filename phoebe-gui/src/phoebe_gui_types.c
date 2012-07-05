#include <stdlib.h>

#include <glade/glade.h>
#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_error_handling.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_plotting.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_treeviews.h"

gboolean PHOEBE_WINDOW_SIDESHEET_IS_DETACHED = FALSE;
gboolean PHOEBE_WINDOW_LC_PLOT_IS_DETACHED   = FALSE;
gboolean PHOEBE_WINDOW_RV_PLOT_IS_DETACHED   = FALSE;
gboolean PHOEBE_WINDOW_FITTING_IS_DETACHED   = FALSE;
gboolean PHOEBE_SPOTS_SHOW_ALL				 = FALSE;

void gui_set_button_image (gchar *button_name, gchar *pixmap_file)
{
	/*
	 * Replaces gtk_button_set_image which does not work on cygwin
	 * Removes the old icon image and then adds a new one
	 */

	GtkWidget *image = gtk_image_new_from_file(pixmap_file);
	GtkContainer *button = GTK_CONTAINER(gui_widget_lookup(button_name)->gtk);
	if (image && button) {
		GList *glist = gtk_container_get_children(button);
		if (glist)
			gtk_container_remove(button, glist->data);
		gtk_container_add (button, image);
		gtk_widget_show (image);
	}
}

int gui_init_widgets ()
{
	/*
	 * This function hooks all widgets to the parameters and adds them to the
	 * widget hash table.
	 */

	int i;

	PHOEBE_parameter *par;

	gchar *glade_xml_file;
    gchar *glade_pixmap_file, *detach_pixmap_file;

	GladeXML  *phoebe_window;

	GtkWidget *phoebe_data_lc_treeview,       		*phoebe_para_lc_levels_treeview,
              *phoebe_para_lc_el3_treeview,   		*phoebe_para_lc_levweight_treeview,
		      *phoebe_para_lc_ld_treeview,    		*phoebe_data_rv_treeview,
		      *phoebe_para_rv_ld_treeview,    		*phoebe_para_spots_treeview,
		      *phoebe_sidesheet_res_treeview, 		*phoebe_sidesheet_fit_treeview,
              *phoebe_lc_plot_treeview,             *phoebe_rv_plot_treeview,
	          *gui_fit_lum_treeview;

	glade_xml_file    	= g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe_cairo.glade", NULL);
	glade_pixmap_file 	= g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
	detach_pixmap_file 	= g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "detach.png", NULL);

	phoebe_window 		= glade_xml_new (glade_xml_file, NULL, NULL);

	glade_xml_signal_autoconnect (phoebe_window);
	g_free (glade_xml_file);

	GUI_wt = phoebe_malloc (sizeof (GUI_widget_table));
	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++)
		GUI_wt->bucket[i] = NULL;

	/* *************************** GUI Parameters *************************** */

	phoebe_parameter_add ("gui_ld_model_autoupdate",	"Automatically update LD model",	KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_fitt_method",			"Fitting method",					KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Differential Corrections");
	phoebe_parameter_add ("gui_lc_plot_synthetic",		"Plot synthetic LC",				KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_observed",		"Plot observed LC",					KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);

#warning RENAME_THESE_PARAMETERS_INTO_SOMETHING_BETTER
	phoebe_parameter_add ("gui_lcplot_observed",        "List of observed data switches",   KIND_SWITCH,    "phoebe_lcno", "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL_ARRAY, NO);
	phoebe_parameter_add ("gui_lcplot_synthetic",       "List of synthetic data switches",  KIND_SWITCH,    "phoebe_lcno", "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL_ARRAY, NO);
	phoebe_parameter_add ("gui_lcplot_obscolor",        "List of observed data colors",     KIND_PARAMETER, "phoebe_lcno", "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING_ARRAY, "#0000FF");
	phoebe_parameter_add ("gui_lcplot_syncolor",        "List of synthetic data colors",    KIND_PARAMETER, "phoebe_lcno", "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING_ARRAY, "#FF0000");
	phoebe_parameter_add ("gui_rvplot_observed",        "List of observed data switches",   KIND_SWITCH,    "phoebe_rvno", "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL_ARRAY, NO);
	phoebe_parameter_add ("gui_rvplot_synthetic",       "List of synthetic data switches",  KIND_SWITCH,    "phoebe_rvno", "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL_ARRAY, NO);
	phoebe_parameter_add ("gui_rvplot_obscolor",        "List of observed data colors",     KIND_PARAMETER, "phoebe_rvno", "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING_ARRAY, "#0000FF");
	phoebe_parameter_add ("gui_rvplot_syncolor",        "List of synthetic data colors",    KIND_PARAMETER, "phoebe_rvno", "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING_ARRAY, "#FF0000");

	phoebe_parameter_add ("gui_lc_plot_verticesno",		"Number of vertices for LC",		KIND_PARAMETER,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_INT,		100);
	phoebe_parameter_add ("gui_lc_plot_obsmenu",		"Select observed LC",				KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_lc_plot_aliasing",		"Turn on data aliasing",			KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_lc_plot_residuals",		"Plot residuals",					KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_x",				"X-axis of LC plot",				KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Phase");
	phoebe_parameter_add ("gui_lc_plot_y",				"Y-axis of LC plot",				KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Flux");
	phoebe_parameter_add ("gui_lc_plot_phstart",		"Phase start",						KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_lc_plot_phend",			"Phase end",						KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	 0.6);
	phoebe_parameter_add ("gui_lc_plot_x_offset",		"X axis Offset",					KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	0.0);
	phoebe_parameter_add ("gui_lc_plot_y_offset",		"Y axis Offset",					KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	0.0);
	phoebe_parameter_add ("gui_lc_plot_zoom",			"Zoom amount",						KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	0.0);
	phoebe_parameter_add ("gui_lc_plot_zoom_level",		"Zoom level",						KIND_PARAMETER,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_INT,		0);
	phoebe_parameter_add ("gui_lc_plot_coarse",			"Coarse grid",						KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_lc_plot_fine",			"Coarse grid",						KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_synthetic",		"Plot synthetic RV curve",			KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_observed",		"Plot observed RV curve",			KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_rv_plot_verticesno",		"Number of vertices for RV curve",	KIND_PARAMETER,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_INT,		100);
	phoebe_parameter_add ("gui_rv_plot_obsmenu",		"Select observed RV curve",			KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"");
	phoebe_parameter_add ("gui_rv_plot_alias",			"Turn on data aliasing",			KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		YES);
	phoebe_parameter_add ("gui_rv_plot_residuals",		"Plot residuals",					KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_x",				"X-axis of RV plot",				KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"Phase");
	phoebe_parameter_add ("gui_rv_plot_y",				"Y-axis of RV plot",				KIND_MENU,		NULL, "%s", 0.0, 0.0, 0.0, NO, TYPE_STRING,	"RV in km/s");
	phoebe_parameter_add ("gui_rv_plot_phstart",		"Phase start",						KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	-0.6);
	phoebe_parameter_add ("gui_rv_plot_phend",			"Phase end",						KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	 0.6);
	phoebe_parameter_add ("gui_rv_plot_x_offset",		"X axis Offset",					KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	0.0);
	phoebe_parameter_add ("gui_rv_plot_y_offset",		"Y axis Offset",					KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	0.0);
	phoebe_parameter_add ("gui_rv_plot_zoom",			"Zoom amount",						KIND_PARAMETER,	NULL, "%lf", 0.0, 0.0, 0.0, NO, TYPE_DOUBLE,	0.0);
	phoebe_parameter_add ("gui_rv_plot_zoom_level",		"Zoom level",						KIND_PARAMETER,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_INT,		0);
	phoebe_parameter_add ("gui_rv_plot_coarse",			"Coarse grid",						KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_rv_plot_fine",			"Coarse grid",						KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL,		NO);
	phoebe_parameter_add ("gui_3d_plot_autoupdate",		"Autoupdate plot on phase change",	KIND_SWITCH,	NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_BOOL, 	NO);
	phoebe_parameter_add ("gui_verbosity_level",		"Level of GUI terminal verbosity", 	KIND_PARAMETER, NULL, "%d", 0.0, 0.0, 0.0, NO, TYPE_INT, 		1);

	/* *************************** Main Window    **************************** */

	gui_widget_add ("phoebe_window",									glade_xml_get_widget (phoebe_window, "phoebe_window"), 													0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_statusbar",									glade_xml_get_widget (phoebe_window, "phoebe_statusbar"), 												0, 					GUI_WIDGET_VALUE, 		NULL, NULL);

	/* ************************    GUI Treeviews   ************************* */

	phoebe_data_lc_treeview           = glade_xml_get_widget (phoebe_window, "phoebe_data_lc_treeview");
	phoebe_para_lc_levels_treeview    = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_levels_treeview");
	phoebe_para_lc_el3_treeview       = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_el3_treeview");
	phoebe_para_lc_levweight_treeview = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_weighting_treeview");
	phoebe_para_lc_ld_treeview        = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_lccoefs_treeview");
	phoebe_data_rv_treeview           = glade_xml_get_widget (phoebe_window, "phoebe_data_rv_treeview");
	phoebe_para_rv_ld_treeview        = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_rvcoefs_treeview");
	phoebe_para_spots_treeview   	  = glade_xml_get_widget (phoebe_window, "phoebe_para_spots_treeview");
	phoebe_sidesheet_res_treeview     = glade_xml_get_widget (phoebe_window, "phoebe_sidesheet_res_treeview");
	phoebe_sidesheet_fit_treeview     = glade_xml_get_widget (phoebe_window, "phoebe_sidesheet_fit_treeview");
	phoebe_lc_plot_treeview           = glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_treeview");
	phoebe_rv_plot_treeview           = glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_treeview");

	gui_fit_lum_treeview              = glade_xml_get_widget (phoebe_window, "phoebe_fitt_third_treeview");

	gui_widget_add ("phoebe_data_lc_treeview",							phoebe_data_lc_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_para_lc_el3_treeview",						phoebe_para_lc_el3_treeview, 																			0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_para_lc_levels_treeview",					phoebe_para_lc_levels_treeview,																			0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_para_lc_levweight_treeview",				phoebe_para_lc_levweight_treeview,																		0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_para_lc_ld_treeview",						phoebe_para_lc_ld_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_data_rv_treeview",							phoebe_data_rv_treeview, 																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_para_rv_ld_treeview",						phoebe_para_rv_ld_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_para_spots_treeview",						phoebe_para_spots_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_sidesheet_res_treeview",					phoebe_sidesheet_res_treeview,																			0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_sidesheet_fit_treeview",					phoebe_sidesheet_fit_treeview,																			0, 					GUI_WIDGET_VALUE, 		NULL, NULL);

	gui_widget_add ("phoebe_fitt_first_treeview",                       glade_xml_get_widget(phoebe_window, "phoebe_fitt_first_treeview"),                                      0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_fitt_second_treeview",                      glade_xml_get_widget(phoebe_window, "phoebe_fitt_second_treeview"),                                     0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_fitt_third_treeview",                       gui_fit_lum_treeview,                                                                                   0,                  GUI_WIDGET_VALUE,       NULL, NULL);

	gui_widget_add ("phoebe_lc_plot_treeview",					        phoebe_lc_plot_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_treeview",					        phoebe_rv_plot_treeview,																				0, 					GUI_WIDGET_VALUE, 		NULL, NULL);

	gui_init_treeviews ();

#warning RENAME_THIS_ONE_AS_WELL
	gui_widget_add ("phoebe_lc_plot_passband_info",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_lc_plot_treeview), LC_COL_PLOT_OBS,       GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_lcplot_observed"), NULL);
	gui_widget_add ("phoebe_lc_plot_passband_syn",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_lc_plot_treeview), LC_COL_PLOT_SYN,       GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_lcplot_synthetic"), NULL);
	gui_widget_add ("phoebe_lc_plot_obscolor",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_lc_plot_treeview), LC_COL_PLOT_OBS_COLOR, GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_lcplot_obscolor"), NULL);
	gui_widget_add ("phoebe_lc_plot_syncolor",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_lc_plot_treeview), LC_COL_PLOT_SYN_COLOR, GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_lcplot_syncolor"), NULL);

	/* LC plotting canvas: */
{
	GtkWidget *plot_area, *plot_button;
	GUI_plot_type ptype = GUI_PLOT_LC;

	plot_area   = glade_xml_get_widget (phoebe_window, "phoebe_plot_lc_graph_area");
	plot_button = glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_plot_button");

	gui_widget_add ("phoebe_lc_plot_area",	plot_area, 0, GUI_WIDGET_VALUE, NULL, NULL);

	/* To this plot button we will attach all property widgets: */
	g_object_set_data (G_OBJECT (plot_button), "plot_type",          &ptype);
	g_object_set_data (G_OBJECT (plot_button), "plot_vertices",      glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_vertices_no_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_alias_switch",  glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_alias_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_resid_switch",  glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_residuals_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_x_request",     glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_x_combobox"));
	g_object_set_data (G_OBJECT (plot_button), "plot_y_request",     glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_y_combobox"));
	g_object_set_data (G_OBJECT (plot_button), "phase_start",        glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_phstart_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "phase_end",          glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_phend_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "coarse_grid_switch", glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_coarse_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "fine_grid_switch",   glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_fine_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_x_coordinate",  glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_coordinates_x_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_y_coordinate",  glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_coordinates_y_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cp_index",      glade_xml_get_widget (phoebe_window, "plot_lc_plot_coordinates_closest_passband"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cx_coordinate", glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_coordinates_cx_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cy_coordinate", glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_coordinates_cy_value"));
	g_object_set_data (G_OBJECT (plot_button), "controls_left",      glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_left_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_down",      glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_down_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_right",     glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_right_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_up",        glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_up_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_reset",     glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_reset_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_zoomin",    glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_zoomin_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_zoomout",   glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_controls_zoomout_button"));
	g_object_set_data (G_OBJECT (plot_button), "save_plot",          glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_save_button"));
	g_object_set_data (G_OBJECT (plot_button), "clear_plot",         glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_clear_button"));
	g_object_set_data (G_OBJECT (plot_button), "save_data",          glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_save_data_button"));

	g_object_set_data (G_OBJECT (plot_button), "plot_passband_info", gui_widget_lookup ("phoebe_lc_plot_passband_info")->gtk);

	/* Initialize the plot area with these widgets and their associations: */
	gui_plot_area_init (plot_area, plot_button);
}

	/* RV plotting canvas: */
{
	GtkWidget *plot_area, *plot_button;
	GUI_plot_type ptype = GUI_PLOT_RV;

	gui_widget_add ("phoebe_rv_plot_passband_info",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_rv_plot_treeview), RV_COL_PLOT_OBS,       GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_rvplot_observed"), NULL);
	gui_widget_add ("phoebe_rv_plot_passband_syn",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_rv_plot_treeview), RV_COL_PLOT_SYN,       GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_rvplot_synthetic"), NULL);
	gui_widget_add ("phoebe_rv_plot_obscolor",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_rv_plot_treeview), RV_COL_PLOT_OBS_COLOR, GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_rvplot_obscolor"), NULL);
	gui_widget_add ("phoebe_rv_plot_syncolor",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_rv_plot_treeview), RV_COL_PLOT_SYN_COLOR, GUI_WIDGET_VALUE, phoebe_parameter_lookup ("gui_rvplot_syncolor"), NULL);

	plot_area   = glade_xml_get_widget (phoebe_window, "phoebe_plot_rv_graph_area");
	plot_button = glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_plot_button");

	gui_widget_add ("phoebe_rv_plot_area",	plot_area, 0, GUI_WIDGET_VALUE, NULL, NULL);
	
	/* To this plot button we will attach all property widgets: */
	g_object_set_data (G_OBJECT (plot_button), "plot_type",          &ptype);
	g_object_set_data (G_OBJECT (plot_button), "plot_vertices",      glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_vertices_no_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_alias_switch",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_alias_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_resid_switch",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_residuals_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_x_request",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_x_combobox"));
	g_object_set_data (G_OBJECT (plot_button), "plot_y_request",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_y_combobox"));
	g_object_set_data (G_OBJECT (plot_button), "phase_start",        glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_phstart_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "phase_end",          glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_phend_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "coarse_grid_switch", glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_coarse_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "fine_grid_switch",   glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_fine_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_x_coordinate",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_x_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_y_coordinate",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_y_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cp_index",      glade_xml_get_widget (phoebe_window, "plot_rv_plot_coordinates_closest_passband"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cx_coordinate", glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_cx_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cy_coordinate", glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_cy_value"));
	g_object_set_data (G_OBJECT (plot_button), "controls_left",      glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_left_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_down",      glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_down_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_right",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_right_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_up",        glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_up_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_reset",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_reset_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_zoomin",    glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_zoomin_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_zoomout",   glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_zoomout_button"));
	g_object_set_data (G_OBJECT (plot_button), "save_plot",          glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_save_button"));
	g_object_set_data (G_OBJECT (plot_button), "clear_plot",         glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_clear_button"));
	g_object_set_data (G_OBJECT (plot_button), "save_data",         glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_save_data_button"));

	g_object_set_data (G_OBJECT (plot_button), "plot_passband_info", gui_widget_lookup ("phoebe_rv_plot_passband_info")->gtk);

	/* Initialize the plot area with these widgets and their associations: */
	gui_plot_area_init (plot_area, plot_button);
}

	/* Mesh plotting canvas: */
{
	GtkWidget *plot_area, *plot_button;
	GUI_plot_type ptype = GUI_PLOT_MESH;

	plot_area   = glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_graph_area");
	plot_button = glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_plot_button");

	/* To this plot button we will attach all property widgets: */
	g_object_set_data (G_OBJECT (plot_button), "plot_type",         &ptype);
	g_object_set_data (G_OBJECT (plot_button), "mesh_phase",        glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_phase"));
	g_object_set_data (G_OBJECT (plot_button), "mesh_autoupdate",   glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_autoupdate_meshbutton"));
/*
	g_object_set_data (G_OBJECT (plot_button), "plot_vertices",      NULL);
	g_object_set_data (G_OBJECT (plot_button), "plot_alias_switch",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_alias_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_resid_switch",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_residuals_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_x_request",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_x_combobox"));
	g_object_set_data (G_OBJECT (plot_button), "plot_y_request",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_y_combobox"));
	g_object_set_data (G_OBJECT (plot_button), "phase_start",        glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_phstart_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "phase_end",          glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_phend_spinbutton"));
	g_object_set_data (G_OBJECT (plot_button), "coarse_grid_switch", glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_coarse_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "fine_grid_switch",   glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_fine_checkbutton"));
	g_object_set_data (G_OBJECT (plot_button), "plot_x_coordinate",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_x_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_y_coordinate",  glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_y_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cp_index",      glade_xml_get_widget (phoebe_window, "plot_rv_plot_coordinates_closest_passband"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cx_coordinate", glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_cx_value"));
	g_object_set_data (G_OBJECT (plot_button), "plot_cy_coordinate", glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_coordinates_cy_value"));
	g_object_set_data (G_OBJECT (plot_button), "controls_left",      glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_left_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_down",      glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_down_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_right",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_right_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_up",        glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_up_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_reset",     glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_controls_reset_button"));
*/
	g_object_set_data (G_OBJECT (plot_button), "controls_zoomin",    glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_zoom_in_button"));
	g_object_set_data (G_OBJECT (plot_button), "controls_zoomout",   glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_zoom_out_button"));
	g_object_set_data (G_OBJECT (plot_button), "save_plot",          glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_save_button"));
	g_object_set_data (G_OBJECT (plot_button), "clear_plot",         glade_xml_get_widget (phoebe_window, "phoebe_plot_mesh_clear_button"));
/*
	g_object_set_data (G_OBJECT (plot_button), "plot_passband_info", gui_widget_lookup ("phoebe_rv_plot_passband_info")->gtk);
*/

	/* Initialize the plot area with these widgets and their associations: */
	gui_plot_area_init (plot_area, plot_button);
}

	/* *************************    Data Widgets   **************************** */

	gui_widget_add ("phoebe_data_star_name_entry", 						glade_xml_get_widget (phoebe_window, "phoebe_data_star_name_entry"), 									0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_name"), NULL);
	gui_widget_add ("phoebe_data_star_model_combobox", 					glade_xml_get_widget (phoebe_window, "phoebe_data_star_model_combobox"), 								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_model"), NULL);

	gui_widget_add ("phoebe_data_lc_filename",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_FILENAME,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filename"), NULL);
	gui_widget_add ("phoebe_data_lc_sigma",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 	  					LC_COL_SIGMA,		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_sigma"), NULL);
	gui_widget_add ("phoebe_data_lc_filter",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_FILTER,		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filter"), NULL);
	gui_widget_add ("phoebe_data_lc_indep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_ITYPE_STR,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indep"), NULL);
	gui_widget_add ("phoebe_data_lc_dep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_DTYPE_STR,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_dep"), NULL);
	gui_widget_add ("phoebe_data_lc_wtype",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_WTYPE_STR,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indweight"), NULL);
	gui_widget_add ("phoebe_data_lc_active",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview), 						LC_COL_ACTIVE,		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_active"), NULL);
	gui_widget_add ("phoebe_data_lc_id",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview),						LC_COL_ID,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_lc_id"), NULL);

	gui_widget_add ("phoebe_data_cadence_switch",                       glade_xml_get_widget (phoebe_window, "phoebe_cadence_switch"),                                          0,                  GUI_WIDGET_VALUE,       phoebe_parameter_lookup ("phoebe_cadence_switch"), NULL);
	gui_widget_add ("phoebe_data_cadence",                              glade_xml_get_widget (phoebe_window, "phoebe_cadence"),                                                 0,                  GUI_WIDGET_VALUE,       phoebe_parameter_lookup ("phoebe_cadence"), NULL);
	gui_widget_add ("phoebe_data_cadence_rate",                         glade_xml_get_widget (phoebe_window, "phoebe_cadence_rate"),                                            0,                  GUI_WIDGET_VALUE,       phoebe_parameter_lookup ("phoebe_cadence_rate"), NULL);
	gui_widget_add ("phoebe_data_cadence_timestamp",                    glade_xml_get_widget (phoebe_window, "phoebe_cadence_timestamp"),                                       0,                  GUI_WIDGET_VALUE,       phoebe_parameter_lookup ("phoebe_cadence_timestamp"), NULL);

	gui_widget_add ("phoebe_data_rv_filename",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_FILENAME,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_filename"), NULL);
	gui_widget_add ("phoebe_data_rv_sigma",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_SIGMA,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_sigma"), NULL);
	gui_widget_add ("phoebe_data_rv_filter",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_FILTER,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_filter"), NULL);
	gui_widget_add ("phoebe_data_rv_indep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_ITYPE_STR,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_indep"), NULL);
	gui_widget_add ("phoebe_data_rv_dep",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_DTYPE_STR,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_dep"), NULL);
	gui_widget_add ("phoebe_data_rv_wtype",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_WTYPE_STR,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_indweight"), NULL);
	gui_widget_add ("phoebe_data_rv_active",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_ACTIVE,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_active"), NULL);
	gui_widget_add ("phoebe_data_rv_id",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_rv_treeview),						RV_COL_ID,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_rv_id"), NULL);

	gui_widget_add ("phoebe_data_options_indep_combobox",	 			glade_xml_get_widget(phoebe_window, "phoebe_data_options_indep_combobox"), 								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_indep"), NULL);

	gui_widget_add ("phoebe_data_options_bins_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_options_bins_checkbutton"), 							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins_switch"), NULL);
	gui_widget_add ("phoebe_data_options_binsno_spinbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_options_binsno_spinbutton"), 							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins"), NULL);

	gui_widget_add ("phoebe_data_lcoptions_mag_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_data_lcoptions_mag_spinbutton"), 							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_mnorm"), NULL);

	gui_widget_add ("phoebe_data_rvoptions_psepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_psepe_checkbutton"), 						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"), NULL);
	gui_widget_add ("phoebe_data_rvoptions_ssepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_ssepe_checkbutton"), 						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"), NULL);

	gui_widget_add ("phoebe_data_options_filtermode_combobox",			glade_xml_get_widget(phoebe_window, "phoebe_data_options_filtermode_combobox"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_passband_mode"), NULL);

	gui_widget_add ("phoebe_data_lc_addnoise_checkbutton",				glade_xml_get_widget(phoebe_window, "phoebe_data_lc_addnoise_checkbutton"),								0,					GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_synscatter_switch"), NULL);
	gui_widget_add ("phoebe_data_lc_sigma_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_data_lc_sigma_spinbutton"),									0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_sigma"), NULL);
	gui_widget_add ("phoebe_data_lc_seed_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_data_lc_seed_spinbutton"),									0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_seed"), NULL);
	gui_widget_add ("phoebe_data_lc_scatter_combobox",					glade_xml_get_widget(phoebe_window, "phoebe_data_lc_scatter_combobox"),									0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_levweight"), NULL);

	/* **********************    Parameters Widgets   ************************* */

	par = phoebe_parameter_lookup ("phoebe_hjd0");
	gui_widget_add ("phoebe_para_eph_hjd0_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_eph_hjd0adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_hjd0step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_hjd0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_hjd0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_hjd0min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_period");
	gui_widget_add ("phoebe_para_eph_period_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_period_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_eph_periodadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_periodstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_periodmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_periodmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_periodmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_dpdt");
	gui_widget_add ("phoebe_para_eph_dpdt_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdt_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_eph_dpdtadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_dpdtstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_dpdtmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_dpdtmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_dpdtmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_pshift");
	gui_widget_add ("phoebe_para_eph_pshift_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshift_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_eph_pshiftadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_pshiftstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_pshiftmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_eph_pshiftmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_eph_pshiftmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_sma");
	gui_widget_add ("phoebe_para_sys_sma_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_sma_spinbutton"),									0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_sys_smaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smaadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_smastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smastep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_smamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smamax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_smamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_smamin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_rm");
	gui_widget_add ("phoebe_para_sys_rm_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rm_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_sys_rmadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_rmstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmstep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_rmmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_rmmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_rmmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_vga");
	gui_widget_add ("phoebe_para_sys_vga_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vga_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_sys_vgaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgaadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_vgastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgastep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_vgamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgamax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_vgamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_vgamin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_incl");
	gui_widget_add ("phoebe_para_sys_incl_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_sys_incl_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_sys_incladjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_sys_incladjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_inclstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_inclmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_sys_inclmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_sys_inclmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_perr0");
	gui_widget_add ("phoebe_para_orb_perr0_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_orb_perr0adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_perr0step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_perr0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_perr0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_perr0min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_dperdt");
	gui_widget_add ("phoebe_para_orb_dperdt_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdt_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_orb_dperdtadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_dperdtstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_dperdtmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_dperdtmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_dperdtmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_ecc");
	gui_widget_add ("phoebe_para_orb_ecc_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_ecc_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_orb_eccadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccadjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_eccstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccstep_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_eccmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccmax_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_eccmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_eccmin_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_f1");
	gui_widget_add ("phoebe_para_orb_f1_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_orb_f1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_f1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1step_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_f1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_f1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f1min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_f2");
	gui_widget_add ("phoebe_para_orb_f2_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2_spinbutton"), 									0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_orb_f2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_f2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2step_spinbutton"), 								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_f2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_orb_f2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_orb_f2min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	gui_widget_add ("gui_perr0_phase",                                  glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_per_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_supcon_phase",                                 glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_supconj_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_infcon_phase",                                 glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_infconj_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_ascnode_phase",                                glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_ascnode_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_descnode_phase",                               glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_descnode_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_perr0_hjd",                                  glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_per_hjd"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_supcon_hjd",                                 glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_supconj_hjd"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_infcon_hjd",                                 glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_infconj_hjd"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_ascnode_hjd",                                glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_ascnode_hjd"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_descnode_hjd",                               glade_xml_get_widget (phoebe_window, "phoebe_para_orb_crph_descnode_hjd"), 0, GUI_WIDGET_VALUE, NULL, NULL);

	par = phoebe_parameter_lookup ("phoebe_teff1");
	gui_widget_add ("phoebe_para_comp_tavh_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavh_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_comp_tavhadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_tavhstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_tavhmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_tavhmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavhmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_teff2");
	gui_widget_add ("phoebe_para_comp_tavc_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavc_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_comp_tavcadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_tavcstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_tavcmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_tavcmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_tavcmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_pot1");
	gui_widget_add ("phoebe_para_comp_phsv_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsv_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_comp_phsvadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_phsvstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_phsvmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_phsvmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsvmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_pot2");
	gui_widget_add ("phoebe_para_comp_pcsv_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsv_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_comp_pcsvadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvadjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_pcsvstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvstep_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_pcsvmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvmax_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_comp_pcsvmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsvmin_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_met1");
	gui_widget_add ("phoebe_para_comp_met1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	par = phoebe_parameter_lookup ("phoebe_met2");
	gui_widget_add ("phoebe_para_comp_met2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_met2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	par = phoebe_parameter_lookup ("phoebe_logg1");
	gui_widget_add ("phoebe_para_comp_logg1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	par = phoebe_parameter_lookup ("phoebe_logg2");
	gui_widget_add ("phoebe_para_comp_logg2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_comp_logg2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);

	gui_widget_add ("phoebe_para_lum_atmospheres_grav_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_grav_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_compute_logg_switch"), NULL);

	par = phoebe_parameter_lookup ("phoebe_alb1");
	gui_widget_add ("phoebe_para_surf_alb1_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_surf_alb1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_alb1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_alb1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_alb1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb1min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_alb2");
	gui_widget_add ("phoebe_para_surf_alb2_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_surf_alb2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2adjust_checkbutton"), 						0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_alb2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_alb2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2max_spinbutton"), 							0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_alb2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_alb2min_spinbutton"), 							0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_grb1");
	gui_widget_add ("phoebe_para_surf_gr1_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_surf_gr1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_gr1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_gr1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_gr1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr1min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_grb2");
	gui_widget_add ("phoebe_para_surf_gr2_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2_spinbutton"), 								0,					GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_surf_gr2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2adjust_checkbutton"), 							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_gr2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2step_spinbutton"), 							0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_gr2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2max_spinbutton"), 								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_surf_gr2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_para_surf_gr2min_spinbutton"), 								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);

	gui_widget_add ("gui_rpole1_value",                                 glade_xml_get_widget (phoebe_window, "phoebe_rpole1_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rside1_value",                                 glade_xml_get_widget (phoebe_window, "phoebe_rside1_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rback1_value",                                 glade_xml_get_widget (phoebe_window, "phoebe_rback1_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rpoint1_value",                                glade_xml_get_widget (phoebe_window, "phoebe_rpoint1_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rpole2_value",                                 glade_xml_get_widget (phoebe_window, "phoebe_rpole2_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rside2_value",                                 glade_xml_get_widget (phoebe_window, "phoebe_rside2_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rback2_value",                                 glade_xml_get_widget (phoebe_window, "phoebe_rback2_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);
	gui_widget_add ("gui_rpoint2_value",                                glade_xml_get_widget (phoebe_window, "phoebe_rpoint2_value"), 0, GUI_WIDGET_VALUE, NULL, NULL);

    gui_widget_add ("phoebe_para_lum_levels_levweight", 				(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_levels_treeview), 				LC_COL_LEVWEIGHT,	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_levweight"), NULL);

    par = phoebe_parameter_lookup ("phoebe_hla");
    gui_widget_add ("phoebe_para_lum_levels_prim",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_levels_treeview), 				LC_COL_HLA,			GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_primmin_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primmin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_primmax_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_primmax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_cla");
	gui_widget_add ("phoebe_para_lum_levels_sec",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_levels_treeview), 				LC_COL_CLA,			GUI_WIDGET_VALUE, 		par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_secmin_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secmin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_levels_secmax_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_secmax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);

    par = phoebe_parameter_lookup ("phoebe_el3");
    gui_widget_add ("phoebe_para_lum_el3",								(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_el3_treeview),					LC_COL_EL3,			GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_lum_el3ajdust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3ajdust_checkbutton"),							0,					GUI_WIDGET_SWITCH_TBA, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3step_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3step_spinbutton"),								0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3min_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3min_spinbutton"),								0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3max_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3max_spinbutton"),								0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3units_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3units_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup("phoebe_el3_units"), NULL);

	phoebe_parameter_add ("gui_el3_lum_units",                          "Third light contribution in total light units",  KIND_COMPUTED, "phoebe_lcno", "%lf",  0.0,   1.0,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);
	par = phoebe_parameter_lookup ("gui_el3_lum_units");
	gui_widget_add ("phoebe_para_lum_el3_lum_units",					(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) gui_fit_lum_treeview), LC_COL_EL3_LUM, GUI_WIDGET_VALUE, par, NULL);

	par = phoebe_parameter_lookup ("phoebe_opsf");
	gui_widget_add ("phoebe_para_lum_el3_opacity",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_el3_treeview),					LC_COL_OPSF,		GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_opacityadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacityadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,  par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_opacitystep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacitystep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_opacitymin_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacitymin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN, 	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_opacitymax_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_opacitymax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX, 	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_extinction");
	gui_widget_add ("phoebe_para_lum_el3_ext",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_el3_treeview),					LC_COL_EXTINCTION,	GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_extadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extadjust_checkbutton"),						0,					GUI_WIDGET_SWITCH_TBA,  par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_extstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extstep_spinbutton"),							0,					GUI_WIDGET_VALUE_STEP,  par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_extmin_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extmin_spinbutton"),							0,					GUI_WIDGET_VALUE_MIN,  	par, NULL);
	gui_widget_add ("phoebe_para_lum_el3_extmax_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_lum_el3_extmax_spinbutton"),							0,					GUI_WIDGET_VALUE_MAX,  	par, NULL);

	gui_widget_add ("phoebe_para_lum_atmospheres_prim_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_prim_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm1_switch"), NULL);
	gui_widget_add ("phoebe_para_lum_atmospheres_sec_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_atmospheres_sec_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm2_switch"), NULL);

	gui_widget_add ("phoebe_para_lum_options_reflections_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_reflections_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_switch"), NULL);
	gui_widget_add ("phoebe_para_lum_options_reflections_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_reflections_spinbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_reflections"), NULL);
	gui_widget_add ("phoebe_para_lum_options_decouple_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_lum_options_decouple_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_usecla_switch"), NULL);

	gui_widget_add ("phoebe_para_ld_model_combobox",					glade_xml_get_widget(phoebe_window, "phoebe_para_ld_model_combobox"),									0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_model"), NULL);
	gui_widget_add ("phoebe_para_ld_bolcoefs_primx_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_primx_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol1"), NULL);
	gui_widget_add ("phoebe_para_ld_bolcoefs_primy_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_primy_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol1"), NULL);
	gui_widget_add ("phoebe_para_ld_bolcoefs_secx_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_secx_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol2"), NULL);
	gui_widget_add ("phoebe_para_ld_bolcoefs_secy_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_bolcoefs_secy_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol2"), NULL);

	par = phoebe_parameter_lookup ("phoebe_ld_lcx1");
	gui_widget_add ("phoebe_para_ld_lccoefs_primx",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_X1,			GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_primy",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_Y1,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_lcy1"), NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,	par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP,	par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_primmin_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primmin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN,	par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_primmax_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_primmax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX,	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_ld_lcx2");
	gui_widget_add ("phoebe_para_ld_lccoefs_secx",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_X2,			GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_secy",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_lc_ld_treeview),						LC_COL_Y2,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_lcy2"), NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,	par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP,	par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_secmin_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secmin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN,	par, NULL);
	gui_widget_add ("phoebe_para_ld_lccoefs_secmax_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_lccoefs_secmax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX,	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_ld_rvx1");
	gui_widget_add ("phoebe_para_ld_rvcoefs_primx",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_rv_ld_treeview),						RV_COL_X1,			GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_primy",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_rv_ld_treeview),						RV_COL_Y1,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_rvy1"), NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_primadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_primadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,	par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_primstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_primstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP,	par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_primmin_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_primmin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN,	par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_primmax_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_primmax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX,	par, NULL);

	par = phoebe_parameter_lookup ("phoebe_ld_rvx2");
	gui_widget_add ("phoebe_para_ld_rvcoefs_secx",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_rv_ld_treeview),						RV_COL_X2,			GUI_WIDGET_VALUE,		par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_secy",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_rv_ld_treeview),						RV_COL_Y2,			GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_ld_rvy2"), NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_secadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_secadjust_checkbutton"),					0,					GUI_WIDGET_SWITCH_TBA,	par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_secstep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_secstep_spinbutton"),						0,					GUI_WIDGET_VALUE_STEP,	par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_secmin_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_secmin_spinbutton"),						0,					GUI_WIDGET_VALUE_MIN,	par, NULL);
	gui_widget_add ("phoebe_para_ld_rvcoefs_secmax_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_ld_rvcoefs_secmax_spinbutton"),						0,					GUI_WIDGET_VALUE_MAX,	par, NULL);

	gui_widget_add ("phoebe_para_spots_primmove_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_primmove_checkbutton"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup("phoebe_spots_corotate1"),	NULL);
	gui_widget_add ("phoebe_para_spots_secmove_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_secmove_checkbutton"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup("phoebe_spots_corotate2"),	NULL);
	gui_widget_add ("phoebe_para_spots_units_combobox",					glade_xml_get_widget(phoebe_window, "phoebe_para_spots_units_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup("phoebe_spots_units"), NULL);

	gui_widget_add ("phoebe_para_spots_active_switch",					(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_ACTIVE,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_spots_active_switch"), NULL);
	gui_widget_add ("phoebe_para_spots_tba_switch",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_ADJUST,	GUI_WIDGET_SWITCH_TBA,	phoebe_parameter_lookup ("phoebe_spots_tba_switch"), NULL);
	gui_widget_add ("phoebe_para_spots_source",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_SOURCE,	GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_spots_source"), NULL);

	gui_widget_add ("phoebe_para_spots_lat",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LAT,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_spots_colatitude"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_latadjust",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LATADJUST,GUI_WIDGET_SWITCH_TBA,	phoebe_parameter_lookup ("phoebe_spots_colatitude_tba"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_latstep",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LATSTEP,	GUI_WIDGET_VALUE_STEP,	phoebe_parameter_lookup ("phoebe_spots_colatitude_step"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_latmin",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LATMIN,	GUI_WIDGET_VALUE_MIN,	phoebe_parameter_lookup ("phoebe_spots_colatitude_min"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_latmax",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LATMAX,	GUI_WIDGET_VALUE_MAX,	phoebe_parameter_lookup ("phoebe_spots_colatitude_max"), gui_widget_lookup("phoebe_para_spots_units_combobox"));

	gui_widget_add ("phoebe_para_spots_lon",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LON,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_spots_longitude"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_lonadjust",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LONADJUST,GUI_WIDGET_SWITCH_TBA,	phoebe_parameter_lookup ("phoebe_spots_longitude_tba"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_lonstep",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LONSTEP,	GUI_WIDGET_VALUE_STEP,	phoebe_parameter_lookup ("phoebe_spots_longitude_step"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_lonmin",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LONMIN,	GUI_WIDGET_VALUE_MIN,	phoebe_parameter_lookup ("phoebe_spots_longitude_min"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_lonmax",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_LONMAX,	GUI_WIDGET_VALUE_MAX,	phoebe_parameter_lookup ("phoebe_spots_longitude_max"), gui_widget_lookup("phoebe_para_spots_units_combobox"));

	gui_widget_add ("phoebe_para_spots_rad",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_RAD,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_spots_radius"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_radadjust",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_RADADJUST,GUI_WIDGET_SWITCH_TBA,	phoebe_parameter_lookup ("phoebe_spots_radius_tba"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_radstep",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_RADSTEP,	GUI_WIDGET_VALUE_STEP,	phoebe_parameter_lookup ("phoebe_spots_radius_step"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_radmin",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_RADMIN,	GUI_WIDGET_VALUE_MIN,	phoebe_parameter_lookup ("phoebe_spots_radius_min"), gui_widget_lookup("phoebe_para_spots_units_combobox"));
	gui_widget_add ("phoebe_para_spots_radmax",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_RADMAX,	GUI_WIDGET_VALUE_MAX,	phoebe_parameter_lookup ("phoebe_spots_radius_max"), gui_widget_lookup("phoebe_para_spots_units_combobox"));

	gui_widget_add ("phoebe_para_spots_temp",							(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_TEMP,		GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_spots_tempfactor"), NULL);
	gui_widget_add ("phoebe_para_spots_tempadjust",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_TEMPADJUST,GUI_WIDGET_SWITCH_TBA,	phoebe_parameter_lookup ("phoebe_spots_tempfactor_tba"), NULL);
	gui_widget_add ("phoebe_para_spots_tempstep",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_TEMPSTEP,	GUI_WIDGET_VALUE_STEP,	phoebe_parameter_lookup ("phoebe_spots_tempfactor_step"), NULL);
	gui_widget_add ("phoebe_para_spots_tempmin",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_TEMPMIN,	GUI_WIDGET_VALUE_MIN,	phoebe_parameter_lookup ("phoebe_spots_tempfactor_min"), NULL);
	gui_widget_add ("phoebe_para_spots_tempmax",						(GtkWidget *) gtk_tree_view_get_model ((GtkTreeView *) phoebe_para_spots_treeview),						SPOTS_COL_TEMPMAX,	GUI_WIDGET_VALUE_MAX,	phoebe_parameter_lookup ("phoebe_spots_tempfactor_max"), NULL);

	gui_widget_add ("phoebe_para_spots_lat_label",						glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lat_frame_label"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_lon_label",						glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lon_frame_label"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_rad_label",						glade_xml_get_widget(phoebe_window, "phoebe_para_spots_rad_frame_label"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_temp_label",						glade_xml_get_widget(phoebe_window, "phoebe_para_spots_temp_frame_label"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);

	gui_widget_add ("phoebe_para_spots_lat_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lat_spinbutton"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_latadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_latadjust_checkbutton"),							0,					GUI_WIDGET_SWITCH_TBA,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_latstep_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_latstep_spinbutton"),							0,					GUI_WIDGET_VALUE_STEP,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_latmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_latmin_spinbutton"),								0,					GUI_WIDGET_VALUE_MIN,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_latmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_latmax_spinbutton"),								0,					GUI_WIDGET_VALUE_MAX,				NULL,	NULL);

	gui_widget_add ("phoebe_para_spots_lon_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lon_spinbutton"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_lonadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lonadjust_checkbutton"),							0,					GUI_WIDGET_SWITCH_TBA,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_lonstep_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lonstep_spinbutton"),							0,					GUI_WIDGET_VALUE_STEP,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_lonmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lonmin_spinbutton"),								0,					GUI_WIDGET_VALUE_MIN,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_lonmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_lonmax_spinbutton"),								0,					GUI_WIDGET_VALUE_MAX,				NULL,	NULL);

	gui_widget_add ("phoebe_para_spots_rad_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_para_spots_rad_spinbutton"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_radadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_radadjust_checkbutton"),							0,					GUI_WIDGET_SWITCH_TBA,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_radstep_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_radstep_spinbutton"),							0,					GUI_WIDGET_VALUE_STEP,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_radmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_radmin_spinbutton"),								0,					GUI_WIDGET_VALUE_MIN,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_radmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_radmax_spinbutton"),								0,					GUI_WIDGET_VALUE_MAX,				NULL,	NULL);

	gui_widget_add ("phoebe_para_spots_temp_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_temp_spinbutton"),								0,					GUI_WIDGET_VALUE,					NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_tempadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_tempadjust_checkbutton"),						0,					GUI_WIDGET_SWITCH_TBA,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_tempstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_para_spots_tempstep_spinbutton"),							0,					GUI_WIDGET_VALUE_STEP,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_tempmin_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_tempmin_spinbutton"),							0,					GUI_WIDGET_VALUE_MIN,				NULL,	NULL);
	gui_widget_add ("phoebe_para_spots_tempmax_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_para_spots_tempmax_spinbutton"),							0,					GUI_WIDGET_VALUE_MAX,				NULL,	NULL);

	/* ***********************    Fitting Widgets   ************************* */

	gui_widget_add ("phoebe_fitt_parameters_finesize1_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_finesize1_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize1"), NULL);
	gui_widget_add ("phoebe_fitt_parameters_finesize2_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_finesize2_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize2"), NULL);
	gui_widget_add ("phoebe_fitt_parameters_coarsize1_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_coarsize1_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize1"), NULL);
	gui_widget_add ("phoebe_fitt_parameters_coarsize2_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_coarsize2_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize2"), NULL);
	gui_widget_add ("phoebe_fitt_parameters_lambda_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_fitt_parameters_lambda_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_dc_lambda"), 		NULL);
	gui_widget_add ("phoebe_fitt_dc_symder_switch_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_fitt_dc_symder_switch_checkbutton"),						0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_dc_symder_switch"), 	NULL);

	gui_widget_add ("phoebe_fitt_nms_iters_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_fitt_nms_iters_spinbutton"),								0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_nms_iters_max"),	NULL);
	gui_widget_add ("phoebe_fitt_nms_accuracy_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_fitt_nms_accuracy_spinbutton"),								0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_nms_accuracy"),	NULL);

	gui_widget_add ("phoebe_fitt_feedback_label",						glade_xml_get_widget(phoebe_window, "phoebe_fitt_feedback_label"),										0,					GUI_WIDGET_VALUE,		NULL,	NULL);
	gui_widget_add ("phoebe_fitt_nms_nolimit_checkbutton",				glade_xml_get_widget(phoebe_window, "phoebe_fitt_nms_nolimit_checkbutton"),								0,					GUI_WIDGET_VALUE,		NULL, 	NULL);

	gui_widget_add ("phoebe_fitt_dc_frame",								glade_xml_get_widget(phoebe_window, "phoebe_fitt_dc_frame"),											0,					GUI_WIDGET_VALUE,		NULL,	NULL);
	gui_widget_add ("phoebe_fitt_nms_frame",							glade_xml_get_widget(phoebe_window, "phoebe_fitt_nms_frame"),											0,					GUI_WIDGET_VALUE,		NULL,	NULL);

	gui_widget_add ("phoebe_fitt_cfval",                                glade_xml_get_widget (phoebe_window, "phoebe_fitt_cfval"),                                              0,                  GUI_WIDGET_VALUE,       NULL,   NULL);
	gui_widget_add ("phoebe_fitt_cfval_compute",                        glade_xml_get_widget (phoebe_window, "phoebe_fitt_cfval_compute_button"),                               0,                  GUI_WIDGET_VALUE,       NULL,   NULL);

	/* *************************    GUI Widgets   *************************** */

	gui_widget_add ("phoebe_para_ld_model_autoupdate_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_para_ld_model_autoupdate_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_ld_model_autoupdate"), NULL);
	gui_widget_add ("phoebe_fitt_method_combobox",						glade_xml_get_widget(phoebe_window, "phoebe_fitt_method_combobox"),										0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_fitt_method"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_vertices_no_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_vertices_no_spinbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_verticesno"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_alias_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_alias_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_aliasing"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_residuals_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_residuals_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_residuals"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_x_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_x_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_x"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_y_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_y_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_y"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_phstart_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phstart_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_phstart"), NULL);
	gui_widget_add ("phoebe_lc_plot_options_phend_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phend_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_phend"), NULL);
	gui_widget_add ("phoebe_lc_plot_controls_coarse_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_coarse_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_coarse"), NULL);
	gui_widget_add ("phoebe_lc_plot_controls_fine_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_controls_fine_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_lc_plot_fine"), NULL);
	//gui_widget_add ("phoebe_rv_plot_options_syn_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_syn_checkbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_synthetic"), NULL);
	//gui_widget_add ("phoebe_rv_plot_options_obs_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_obs_checkbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_observed"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_vertices_no_spinbutton",	glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_vertices_no_spinbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_verticesno"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_alias_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_alias_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_alias"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_residuals_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_residuals_checkbutton"),					0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_residuals"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_x_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_x_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_x"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_y_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_y_combobox"),								0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_y"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_phstart_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_phstart_spinbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_phstart"), NULL);
	gui_widget_add ("phoebe_rv_plot_options_phend_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_phend_spinbutton"),							0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_phend"), NULL);
	gui_widget_add ("phoebe_rv_plot_scrolledwindow",					glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_scrolledwindow"),									0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_controls_coarse_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_coarse_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_coarse"), NULL);
	gui_widget_add ("phoebe_rv_plot_controls_fine_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_controls_fine_checkbutton"),						0,					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_rv_plot_fine"), NULL);

	gui_widget_add ("phoebe_para_comp_phsv_calculate_button",           glade_xml_get_widget(phoebe_window, "phoebe_para_comp_phsv_calculate_button"),                          0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_para_comp_pcsv_calculate_button",           glade_xml_get_widget(phoebe_window, "phoebe_para_comp_pcsv_calculate_button"),                          0,                  GUI_WIDGET_VALUE,       NULL, NULL);

	gui_widget_add ("phoebe_sidesheet_detach_button",                   glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_detach_button"),                                  0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_fitt_detach_button",                        glade_xml_get_widget(phoebe_window, "phoebe_fitt_detach_button"),                                       0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_lc_plot_detach_button",                     glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_detach_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_detach_button",                     glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_detach_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL, NULL);

	gui_widget_add ("phoebe_rv_plot_options_obs_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_obs_combobox"),								0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup("gui_rv_plot_obsmenu"), gui_widget_lookup("phoebe_data_rv_filter"));

	gui_widget_add ("phoebe_lc_plot_options_phstart_label",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phstart_label"),							0,					GUI_WIDGET_VALUE,		NULL, NULL);
	gui_widget_add ("phoebe_lc_plot_options_phend_label",				glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_options_phend_label"),								0,					GUI_WIDGET_VALUE,		NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_options_phstart_label",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_phstart_label"),							0,					GUI_WIDGET_VALUE,		NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_options_phend_label",				glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_options_phend_label"),								0,					GUI_WIDGET_VALUE,		NULL, NULL);
	gui_widget_add ("phoebe_plot_mesh_autoupdate_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_plot_mesh_autoupdate_checkbutton"),							0,					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("gui_3d_plot_autoupdate"), NULL);

	/* The following widgets need to be disabled during minimization calculations */
	gui_widget_add ("phoebe_fitt_progressbar",                          glade_xml_get_widget(phoebe_window, "phoebe_fitt_progressbar"),                                         0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_fitt_fitting_corrmat_button",               glade_xml_get_widget(phoebe_window, "phoebe_fitt_fitting_corrmat_button"),                              0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_fitt_calculate_button",                     glade_xml_get_widget(phoebe_window, "phoebe_fitt_calculate_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_lc_plot_plot_button",                       glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_plot_button"),                                      0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_plot_button",                       glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_plot_button"),                                      0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_plot_mesh_plot_button",                     glade_xml_get_widget(phoebe_window, "phoebe_plot_mesh_plot_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_open_toolbutton",                           glade_xml_get_widget(phoebe_window, "phoebe_open_toolbutton"),                                          0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_file_open_menuitem",                        glade_xml_get_widget(phoebe_window, "phoebe_file_open_menuitem"),                                       0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_fitt_updateall_button",                     glade_xml_get_widget(phoebe_window, "phoebe_fitt_updateall_button"),                                    0,                  GUI_WIDGET_VALUE,       NULL, NULL);
	gui_widget_add ("phoebe_para_lum_levels_calc_button",               glade_xml_get_widget(phoebe_window, "phoebe_para_lum_levels_calc_button"),                              0,                  GUI_WIDGET_VALUE,       NULL, NULL);


	/* ************************    GUI Containers   ************************* */

	gui_widget_add ("phoebe_lc_plot_image",								glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_image"),									        0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_image",								glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_image"),									        0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_eb_plot_image",								glade_xml_get_widget(phoebe_window, "phoebe_eb_plot_image"),									        0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_plot_mesh_phase",							glade_xml_get_widget(phoebe_window, "phoebe_plot_mesh_phase"),										    0,					GUI_WIDGET_VALUE, 		NULL, NULL);

	gui_widget_add ("phoebe_sidesheet_parent_table",					glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_parent_table"),									0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_sidesheet_vbox",							glade_xml_get_widget(phoebe_window, "phoebe_sidesheet_vbox"),											0,					GUI_WIDGET_VALUE, 		NULL, NULL);

	gui_widget_add ("phoebe_lc_plot_table",								glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_table"),											0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_lc_plot_parent_table",						glade_xml_get_widget(phoebe_window, "phoebe_lc_plot_parent_table"),										0,					GUI_WIDGET_VALUE, 		NULL, NULL);

	gui_widget_add ("phoebe_rv_plot_table",								glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_table"),											0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_rv_plot_parent_table",						glade_xml_get_widget(phoebe_window, "phoebe_rv_plot_parent_table"),										0,					GUI_WIDGET_VALUE, 		NULL, NULL);

	gui_widget_add ("phoebe_fitt_frame",								glade_xml_get_widget(phoebe_window, "phoebe_fitt_frame"),												0,					GUI_WIDGET_VALUE, 		NULL, NULL);
	gui_widget_add ("phoebe_fitt_parent_table",							glade_xml_get_widget(phoebe_window, "phoebe_fitt_parent_table"),										0,					GUI_WIDGET_VALUE, 		NULL, NULL);

	g_object_unref (phoebe_window);

	gui_set_button_image ("phoebe_sidesheet_detach_button", detach_pixmap_file);
	gui_set_button_image ("phoebe_fitt_detach_button",      detach_pixmap_file);
	gui_set_button_image ("phoebe_lc_plot_detach_button",   detach_pixmap_file);
	gui_set_button_image ("phoebe_rv_plot_detach_button",   detach_pixmap_file);
	g_free (detach_pixmap_file);

	gui_init_parameter_options ();
	gui_init_combo_boxes();

	gtk_widget_show (gui_widget_lookup ("phoebe_window")->gtk);
	gtk_window_set_icon (GTK_WINDOW(gui_widget_lookup ("phoebe_window")->gtk), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));
	g_free (glade_pixmap_file);
	
	{
		int _i;
		GUI_wt_bucket *elem;

		for (_i = 0; _i < GUI_WT_HASH_BUCKETS; _i++) {
			elem = GUI_wt->bucket[_i];
			while (elem) {
				phoebe_debug ("%50s", elem->widget->name);
				elem = elem->next;
			}
			phoebe_debug ("\n");
		}
	}

	gui_set_values_to_widgets();
	gui_update_angle_values ();

	gui_status("PHOEBE started. All fields initialized to default values.");

	return SUCCESS;
}

int gui_init_combo_boxes()
{
	int i, optindex, optcount, status = 0;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		bucket = GUI_wt->bucket[i];
		while (bucket) {
			if (bucket->widget->par) {
				if (bucket->widget->par->kind == KIND_MENU && bucket->widget->par->type == TYPE_STRING) {
					optcount = bucket->widget->par->menu->optno;
					for (optindex = 0; optindex < optcount; optindex++)
						gtk_combo_box_append_text (GTK_COMBO_BOX (bucket->widget->gtk), bucket->widget->par->menu->option[optindex]);
				}
			}
			bucket = bucket->next;
		}
	}
	return status;
}

int gui_init_parameter_options()
{
	PHOEBE_parameter *par;
	int status = 0;

	par = phoebe_parameter_lookup("gui_fitt_method");
	phoebe_parameter_add_option (par, "Differential Corrections");
	phoebe_parameter_add_option (par, "Nelder & Mead's Simplex");

	par = phoebe_parameter_lookup("gui_lc_plot_x");
	phoebe_parameter_add_option (par, "Phase");
	phoebe_parameter_add_option (par, "Time (HJD)");

	par = phoebe_parameter_lookup("gui_lc_plot_y");
	phoebe_parameter_add_option (par, "Flux");
	phoebe_parameter_add_option (par, "Magnitude");

	par = phoebe_parameter_lookup("gui_rv_plot_x");
	phoebe_parameter_add_option (par, "Phase");
	phoebe_parameter_add_option (par, "Time (HJD)");

	par = phoebe_parameter_lookup("gui_rv_plot_y");
	phoebe_parameter_add_option (par, "RV in km/s");
/*
	phoebe_parameter_add_option (par, "Primary RV");
	phoebe_parameter_add_option (par, "Secondary RV");
	phoebe_parameter_add_option (par, "Primary+Secondary RV");
*/
	return status;
}

int gui_init_angle_widgets ()
{
	/*
	 * gui_init_angle_widgets:
	 *
	 * All angles in the parameter file are saved in radians. The GUI offers
	 * an option to display angles in degrees. This function checks the setup
	 * and sets the adjustments accordingly. It does not, however, transform
	 * the values. The gui_update_angle_values() function below does that.
	 *
	 * Returns: status.
	 */
	
	GtkWidget *widget;
	GtkAdjustment *adjust;
	char *units;
	int i = 0;

	char *widget_name[] = {
		"phoebe_para_orb_perr0_spinbutton",
		"phoebe_para_orb_perr0step_spinbutton",
		"phoebe_para_orb_perr0min_spinbutton",
		"phoebe_para_orb_perr0max_spinbutton",
		"phoebe_para_orb_dperdt_spinbutton",
		"phoebe_para_orb_dperdtstep_spinbutton",
		"phoebe_para_orb_dperdtmin_spinbutton",
		"phoebe_para_orb_dperdtmax_spinbutton",
		NULL
	};

	phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
	if (strcmp (units, "Radians") == 0)
		return SUCCESS;

	while (widget_name[i]) {
		widget = gui_widget_lookup (widget_name[i])->gtk;
		adjust = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON (widget));
		adjust->upper = 360.0;
		i++;
	}
	
	return SUCCESS;
}

int gui_update_angle_values ()
{
	/*
	 * gui_update_angle_values:
	 *
	 * All angles in the parameter file are saved in radians. The GUI offers
	 * an option to display angles in degrees. This function transforms all
	 * current values to reflect this. It should be called as soon as the
	 * file is opened.
	 *
	 * Returns: status.
	 */
	
	GtkWidget *widget;
	double conv = 180./3.1415926535897931;
	char *units;
	int i;

	char *widget_name[] = {
		"phoebe_para_orb_perr0_spinbutton",
		"phoebe_para_orb_perr0step_spinbutton",
		"phoebe_para_orb_perr0min_spinbutton",
		"phoebe_para_orb_perr0max_spinbutton",
		"phoebe_para_orb_dperdt_spinbutton",
		"phoebe_para_orb_dperdtstep_spinbutton",
		"phoebe_para_orb_dperdtmin_spinbutton",
		"phoebe_para_orb_dperdtmax_spinbutton",
		NULL
	};

	phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
	if (strcmp (units, "Radians") == 0)
		return SUCCESS;

	i = 0;
	while (widget_name[i] != NULL) {
		widget = gui_widget_lookup (widget_name[i])->gtk;
		gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget), conv*gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget)));
		i++;
	}

	return SUCCESS;
}

int gui_export_angles_to_radians ()
{
	/*
	 * gui_export_angles_to_radians:
	 *
	 * All angles in the parameter file are saved in radians. The GUI offers
	 * an option to display angles in degrees. If degrees are displayed, the
	 * angles need to be transformed back to radians before any computation
	 * is done or a file is saved.
	 *
	 * Returns: status.
	 */
	
	PHOEBE_parameter *par;
	double conv = 3.1415926535897931/180.0;
	char *units;
	double val;
	int i;

	char *qualifier[] = {
		"phoebe_perr0",
		"phoebe_dperdt",
		NULL
	};

	phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
	if (strcmp (units, "Radians") == 0)
		return SUCCESS;

	i = 0;
	while (qualifier[i] != NULL) {
		par = phoebe_parameter_lookup (qualifier[i]);
		phoebe_parameter_get_value (par, &val);
		phoebe_parameter_set_value (par, conv*val);
		phoebe_parameter_get_min   (par, &val);
		phoebe_parameter_set_min   (par, conv*val);
		phoebe_parameter_get_max   (par, &val);
		phoebe_parameter_set_max   (par, conv*val);
		phoebe_parameter_get_step  (par, &val);
		phoebe_parameter_set_step  (par, conv*val);
		i++;
	}

	return SUCCESS;
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

int gui_widget_hookup (GUI_widget *widget, GtkWidget *gtk, int aux, GUI_widget_type type, char *name, PHOEBE_parameter *par, GUI_widget *dep)
{
	if (!widget) {
		phoebe_debug ("*** a pointer to widget %s passed to gui_widget_hookup () is NULL!\n", name);
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
	widget->dep  = dep;

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
		phoebe_debug ("*** widget lookup failure: %s not found.\n", name);
		return NULL;
	}

	return bucket->widget;
}

int gui_widget_add (char *name, GtkWidget *gtk, int aux, GUI_widget_type type, PHOEBE_parameter *par, GUI_widget *dep)
{
	GUI_widget *widget;

	if (!gtk) {
		phoebe_debug ("*** widget %s passed to gui_widget_add () is NULL!\n", name);
		return -1;
	}

	widget = gui_widget_new ();
	gui_widget_hookup (widget, gtk, aux, type, name, par, dep);
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

	if (!widget->par){
		phoebe_debug ("\tparameter type: n/a\n");
		return status;
	}
	else{
		phoebe_debug ("\tparameter type: %s\n", phoebe_type_get_name (widget->par->type));

		if (GTK_IS_TREE_MODEL (widget->gtk)) {
			GtkTreeModel *model = GTK_TREE_MODEL (widget->gtk);
			GtkTreeIter iter;
			int index;
			bool state;
			gchar *iterstr;

			phoebe_debug ("\twidget type: tree model\n");

			state = gtk_tree_model_get_iter_first (model, &iter);

			while (state) {
				iterstr = gtk_tree_model_get_string_from_iter (model, &iter);
				index = atoi (iterstr);
				g_free (iterstr);
				
				switch (widget->par->type) {
					case TYPE_INT_ARRAY: {
						int value;
						gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
						phoebe_debug ("\tsetting value %d to %d\n", index, value);
						status = phoebe_parameter_set_value (widget->par, index, value);
					}
					break;
					case TYPE_BOOL_ARRAY: {
						bool value;
						gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
						phoebe_debug ("\tsetting value %d to %d\n", index, value);
						status = phoebe_parameter_set_value (widget->par, index, value);
					}
					break;
					case TYPE_DOUBLE_ARRAY: {
						double value;
						gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
						phoebe_debug ("\tsetting value %d to %lf\n", index, value);
						status = phoebe_parameter_set_value (widget->par, index, value);
					}
					break;
					case TYPE_STRING_ARRAY: {
						char *value;
						gtk_tree_model_get (model, &iter, widget->aux, &value, -1);
						phoebe_debug ("\tsetting value %d to %s\n", index, value);
						status = phoebe_parameter_set_value (widget->par, index, value);
						free (value);
					}
					break;
					default:
						/* change to phoebe_gui_error! */
						phoebe_gui_output ("\t*** I'm not supposed to be here!\n");
						phoebe_gui_output ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_TREE_VIEW_COLUMN block, GUI_WIDGET_VALUE block; please report this!\n");
						return ERROR_EXCEPTION_HANDLER_INVOKED;
				}
				state = gtk_tree_model_iter_next (model, &iter);
			}
			return SUCCESS;
		}

		if (GTK_IS_SPIN_BUTTON (widget->gtk)) {
			phoebe_debug ("\twidget type: spin button\n");
			switch (widget->type) {
				case GUI_WIDGET_VALUE: {
					switch (widget->par->type) {
						case TYPE_INT:
							phoebe_debug ("\tsetting value to %d\n", gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON (widget->gtk)));
							status = phoebe_parameter_set_value (widget->par, gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON (widget->gtk)));
						break;
						case TYPE_DOUBLE:
							phoebe_debug ("\tsetting value to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
							status = phoebe_parameter_set_value (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
						break;
						default:
							/* change to phoebe_gui_error! */
							phoebe_gui_output ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->par->type switch; please report this!\n");
							return ERROR_EXCEPTION_HANDLER_INVOKED;
					}
				}
				break;
				case GUI_WIDGET_VALUE_MIN: {
					phoebe_debug ("\tsetting min to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
					status = phoebe_parameter_set_min (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
				}
				break;
				case GUI_WIDGET_VALUE_MAX: {
					phoebe_debug ("\tsetting max to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
					status = phoebe_parameter_set_max (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
				}
				break;
				case GUI_WIDGET_VALUE_STEP: {
					phoebe_debug ("\tsetting step to %lf\n", gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
					status = phoebe_parameter_set_step (widget->par, gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget->gtk)));
				}
				break;
				default:
					/* change to phoebe_gui_error! */
					phoebe_gui_output ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_SPIN_BUTTON block, par->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
			return status;
		}

		if (GTK_IS_ENTRY (widget->gtk)) {
			phoebe_debug ("\twidget type: entry\n");
			phoebe_debug ("\tsetting value to %s\n", gtk_entry_get_text (GTK_ENTRY (widget->gtk)));
			status = phoebe_parameter_set_value (widget->par, gtk_entry_get_text (GTK_ENTRY (widget->gtk)));
			return status;
		}

		if (GTK_IS_RADIO_BUTTON (widget->gtk)) {
			phoebe_debug ("\twidget type: radio button\n");
			phoebe_debug ("\thandler not yet implemented.\n");
			return SUCCESS;
		}

		if (GTK_IS_CHECK_BUTTON (widget->gtk)) {
			phoebe_debug ("\twidget type: check button\n");
			switch (widget->type) {
				case GUI_WIDGET_VALUE:
					phoebe_debug ("\tsetting value to %d\n", gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
					status = phoebe_parameter_set_value (widget->par, gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
				break;
				case GUI_WIDGET_SWITCH_TBA:
					phoebe_debug ("\tsetting tba to %d\n", gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
					status = phoebe_parameter_set_tba (widget->par, gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget->gtk)));
				break;
				default:
					phoebe_gui_output ("\t*** exception handler invoked in gui_get_value_from_widget (), GTK_IS_CHECK_BUTTON block, widget->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
			return status;
		}

		if (GTK_IS_COMBO_BOX (widget->gtk)) {
			phoebe_debug ("\twidget type: combo box\n");
			if (gtk_combo_box_get_active((GtkComboBox*) widget->gtk) >= 0){
				phoebe_debug ("\tsetting option to index %d, value %s\n", gtk_combo_box_get_active((GtkComboBox*) widget->gtk), widget->par->menu->option[gtk_combo_box_get_active((GtkComboBox*) widget->gtk)]);
				status = phoebe_parameter_set_value (widget->par, widget->par->menu->option[gtk_combo_box_get_active((GtkComboBox*) widget->gtk)]);
				return status;
			}
			else{
				phoebe_gui_output ("\t*** nothing selected in the combo for %s.\n", widget->par->qualifier);
				return SUCCESS;
			}
		}
	}

	phoebe_gui_output ("\t*** I got where I am not supposed to be!!\n");
	phoebe_gui_output ("\t*** exception handler invoked in gui_get_value_from_widget (); please report this!\n");

	return SUCCESS;
}

int gui_set_value_to_widget (GUI_widget *widget)
{
	int status = 0;

	if (widget->dep){
		status = gui_set_value_to_widget(widget->dep);
		phoebe_debug("\t *** going to process the dependency on %s first! ***\n", widget->dep->name);
	}

	if (!widget->par) {
		phoebe_debug ("\tparameter type: n/a\n");
		return status;
	}
	else{
		phoebe_debug ("\tparameter type: %s\n", phoebe_type_get_name (widget->par->type));

		if (GTK_IS_TREE_MODEL (widget->gtk)) {
			GtkTreeModel *model = GTK_TREE_MODEL (widget->gtk);
			GtkTreeIter iter;
			int index;
			bool state;
			gchar *iterstr;

			phoebe_debug ("\twidget type: tree model\n");

			state = gtk_tree_model_get_iter_first (model, &iter);

			while (state) {
				iterstr = gtk_tree_model_get_string_from_iter (model, &iter);
				index = atoi (iterstr);
				g_free (iterstr);
				
				switch (widget->par->type) {
					case TYPE_INT_ARRAY: {
						int value;
						status = phoebe_parameter_get_value (widget->par, index, &value);
						phoebe_debug ("\tsetting value %d to %d\n", index, value);
						gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
					}
					break;
					case TYPE_BOOL_ARRAY: {
						bool value;
						status = phoebe_parameter_get_value (widget->par, index, &value);
						phoebe_debug ("\tsetting value %d to %d\n", index, value);
						gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
					}
					break;
					case TYPE_DOUBLE_ARRAY: {
						double value;
						status = phoebe_parameter_get_value (widget->par, index, &value);
						phoebe_debug ("\tsetting value %d to %lf\n", index, value);
						gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
					}
					break;
					case TYPE_STRING_ARRAY: {
						char *value;
						status = phoebe_parameter_get_value (widget->par, index, &value);
						phoebe_debug ("\tsetting value %d to %s\n", index, value);
						gtk_list_store_set(GTK_LIST_STORE(model), &iter, widget->aux, value, -1);
					}
					break;
					default:
						/* change to phoebe_gui_error! */
						phoebe_debug ("\t*** I'm not supposed to be here!\n");
						phoebe_debug ("\t*** exception handler invoked in gui_set_value_from_widget (), GTK_IS_TREE_MODEL block, GUI_WIDGET_VALUE block; please report this!\n");
						return ERROR_EXCEPTION_HANDLER_INVOKED;
				}
				state = gtk_tree_model_iter_next (model, &iter);
			}
			return SUCCESS;
		}

		if (GTK_IS_SPIN_BUTTON (widget->gtk)) {
			phoebe_debug ("\twidget type: spin button\n");
			switch (widget->type){
				case GUI_WIDGET_VALUE: {
					switch (widget->par->type) {
						case TYPE_INT: {
							int value;
							phoebe_debug ("\tpar->type: int, widget->type: value\n");
							status = phoebe_parameter_get_value (widget->par, &value);
							phoebe_debug ("\tsetting value to %d\n", value);
							gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
						}
						break;
						case TYPE_DOUBLE: {
							double value;
							phoebe_debug ("\tpar->type: double, widget->type: value\n");
							status = phoebe_parameter_get_value (widget->par, &value);
							phoebe_debug ("\tsetting value to %lf\n", value);
							gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
						}
						break;
						default:
						/* change to phoebe_gui_error! */
						phoebe_debug ("\t*** I'm not supposed to be here!\n");
						phoebe_debug ("\t*** exception handler invoked in gui_set_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->par->type switch; please report this!\n");
						return ERROR_EXCEPTION_HANDLER_INVOKED;
					}
				}
				break;
				case GUI_WIDGET_VALUE_MIN: {
					double value;
					status = phoebe_parameter_get_min (widget->par, &value);
					phoebe_debug("\tsetting min to %lf\n", value);
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
				}
				break;
				case GUI_WIDGET_VALUE_MAX: {
					double value;
					status = phoebe_parameter_get_max (widget->par, &value);
					phoebe_debug ("\tsetting max to %lf\n", value);
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
				}
				break;
				case GUI_WIDGET_VALUE_STEP: {
					double value;
					status = phoebe_parameter_get_step (widget->par, &value);
					phoebe_debug ("\tsetting step to %lf\n", value);
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (widget->gtk), value);
				}
				break;
				default:
					/* change to phoebe_gui_error! */
					phoebe_debug ("\t*** I'm not supposed to be here!\n");
					phoebe_debug ("\t*** exception handler invoked in gui_set_value_from_widget (), GTK_IS_SPIN_BUTTON block, widget->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
			return status;
		}

		if (GTK_IS_ENTRY (widget->gtk)){
			phoebe_debug ("\twidget type: entry\n");
			char *value;
			status = phoebe_parameter_get_value(widget->par, &value);
			phoebe_debug ("\tsetting value to %s\n", value);
			gtk_entry_set_text(GTK_ENTRY(widget->gtk), value);
			return status;
		}

		if (GTK_IS_RADIO_BUTTON (widget->gtk)){
			phoebe_debug ("\twidget type: radio button\n");
			phoebe_debug ("\t*** handler not yet implemented.\n");
			return status;
		}

		if (GTK_IS_CHECK_BUTTON (widget->gtk)){
			phoebe_debug ("\twidget type: check button\n");
			switch(widget->type){
				bool value;
				case GUI_WIDGET_VALUE: {
					status = phoebe_parameter_get_value(widget->par, &value);
					phoebe_debug ("\tsetting value to %d\n", value);
					gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget->gtk), value);
				}
				break;
				case GUI_WIDGET_SWITCH_TBA: {
					status = phoebe_parameter_get_tba(widget->par, &value);
					phoebe_debug ("\tsetting value to %d\n", value);
					gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget->gtk), value);
				}
				break;
				default:
					/* change to phoebe_gui_error! */
					phoebe_debug ("\t*** exception handler invoked in sui_set_value_to_widget (), GTK_IS_CHECK_BUTTON block, widget->type switch; please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
			return status;
		}

		if (GTK_IS_COMBO_BOX(widget->gtk)) {
			if (widget->par){
				phoebe_debug ("\twidget type: combo box\n");
				char *value;
				int index;
				status = phoebe_parameter_get_value(widget->par, &value);
				status = phoebe_parameter_option_get_index(widget->par, value, &index);
				gtk_combo_box_set_active(GTK_COMBO_BOX(widget->gtk), index);
				return status;
			}
		}
	}

	phoebe_debug ("\t*** I got where I am not supposed to be!!\n");
	phoebe_debug ("\t*** exception handler invoked in gui_set_value_to_widget (); please report this!\n");

	return SUCCESS;
}

int gui_get_values_from_widgets ()
{
 	phoebe_debug("\n\n******** Entering gui_get_values_from_widgets!******* \n\n");

	int i, status;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		bucket = GUI_wt->bucket[i];
		while (bucket) {
			phoebe_debug ("processing widget %s:\n", bucket->widget->name);
			status = gui_get_value_from_widget (bucket->widget);
			phoebe_debug ("\tstatus: %s", phoebe_gui_error (status));
			bucket = bucket->next;
		}
	}

	gui_export_angles_to_radians();
	
	gui_status("Readout completed.");

	return SUCCESS;
}

int gui_set_values_to_widgets ()
{
    gui_status ("Adopting new parameter values...");
	
	phoebe_debug ("\n\n ******* Entering gui_set_values_to_widgets!******* \n\n");
	
	int i, status;
	GUI_wt_bucket *bucket;
	
	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		bucket = GUI_wt->bucket[i];
		while (bucket) {
			phoebe_debug ("processing widget %s: \n", bucket->widget->name);
			status = gui_set_value_to_widget (bucket->widget);
			phoebe_debug ("%s", phoebe_gui_error (status));
			bucket = bucket->next;
		}
	}
	
	gui_fill_sidesheet_fit_treeview ();
	gui_fill_fitt_mf_treeview ();
	gui_update_angle_values ();
	on_orbital_elements_changed ((GtkSpinButton *)NULL, (gpointer)NULL);
	
	gui_status ("Parameters updated.");
	
	return SUCCESS;
}

void gui_toggle_sensitive_widgets_for_minimization (bool enable)
{
	/* Disable widgets during minimization, otherwise they may give problems
	 * (especially those reading/writing files that DC also uses)
	 */
	
	GUI_widget *widget;

	/* Widgets that must be disabled while fitting is in progress: */

	char *widgets[] = {
				"phoebe_fitt_calculate_button",
				"phoebe_fitt_updateall_button",
				"phoebe_fitt_fitting_corrmat_button",
				"phoebe_fitt_method_combobox",
				"phoebe_open_toolbutton",
				"phoebe_lc_plot_plot_button",
				"phoebe_rv_plot_plot_button",
				"phoebe_plot_mesh_plot_button",
				"phoebe_file_open_menuitem",
				"phoebe_para_lum_levels_calc_button",
				"phoebe_open_toolbutton",
				NULL
    };

	int i = 0;
	while (widgets[i]) {
		if ((widget = gui_widget_lookup (widgets[i])))
/*			gtk_widget_set_sensitive (widget->gtk, enable);*/
		i++;
	}

	return;
}


