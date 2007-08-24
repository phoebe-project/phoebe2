#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_callbacks.h"

int gui_init_treeviews(GladeXML *phoebe_window, GladeXML *phoebe_load_lc_dialog)
{
    gui_init_lc_treeviews         (phoebe_window);
    gui_init_rv_treeviews         (phoebe_window);
    gui_init_spots_treeview       (phoebe_window);
//    gui_init_datasheets           (phoebe_window);

    return SUCCESS;
}

int gui_init_lc_treeviews(GladeXML *phoebe_window)
{
    // g_print("---------- Initializing lc treeviews ---------------\n");

    phoebe_data_lc_treeview             = glade_xml_get_widget (phoebe_window, "phoebe_data_lc_treeview");
    phoebe_para_lc_el3_treeview         = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_el3_treeview");
    phoebe_para_lc_levels_treeview      = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_levels_treeview");
    phoebe_para_lc_levweight_treeview   = glade_xml_get_widget (phoebe_window, "phoebe_para_lum_weighting_treeview");
    phoebe_para_lc_ld_treeview          = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_lccoefs_treeview");
	phoebe_plot_lc_observed_combobox	= glade_xml_get_widget (phoebe_window, "phoebe_lc_plot_options_obs_combobox");

    GtkTreeModel *lc_model = lc_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", LC_COL_ACTIVE, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_ACTIVE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_ACTIVE);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_lc_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_FILENAME));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_FILENAME);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", LC_COL_FILTER, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_FILTER));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_levels_treeview, column, LC_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_levweight_treeview, column, LC_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_FILTER);

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT(phoebe_plot_lc_observed_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT(phoebe_plot_lc_observed_combobox), renderer, "text", LC_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", LC_COL_ITYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_ITYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_ITYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", LC_COL_DTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_DTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_DTYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", LC_COL_WTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_WTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_WTYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", LC_COL_SIGMA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_SIGMA));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_SIGMA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Level weighting", renderer, "text", LC_COL_LEVWEIGHT, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_levweight_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_LEVWEIGHT));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levweight_treeview, column, LC_COL_LEVWEIGHT);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_levels_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_HLA));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, LC_COL_HLA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_levels_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_CLA));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, LC_COL_CLA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Opacity function", renderer, "text", LC_COL_OPSF, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_el3_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_OPSF));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_OPSF);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Third light", renderer, "text", LC_COL_EL3, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_el3_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_EL3));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_EL3);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Extinction", renderer, "text", LC_COL_EXTINCTION, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_el3_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_EXTINCTION));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_EXTINCTION);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", LC_COL_X1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_X1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_X1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", LC_COL_X2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_X2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_X2);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", LC_COL_Y1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_Y1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_Y1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", LC_COL_Y2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_Y2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_Y2);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_data_lc_treeview,            lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_el3_treeview,        lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_levels_treeview,     lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_levweight_treeview,  lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_ld_treeview,         lc_model);
	gtk_combo_box_set_model ((GtkComboBox*)phoebe_plot_lc_observed_combobox,   lc_model);

    return SUCCESS;
}

int gui_init_rv_treeviews(GladeXML *phoebe_window)
{
    // g_print("---------- Initializing rv treeviews ---------------\n");

    phoebe_data_rv_treeview    = glade_xml_get_widget (phoebe_window, "phoebe_data_rv_treeview");
    phoebe_para_rv_ld_treeview = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_rvcoefs_treeview");
	phoebe_plot_rv_observed_combobox	= glade_xml_get_widget (phoebe_window, "phoebe_rv_plot_options_obs_combobox");

    GtkTreeModel *rv_model = rv_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", RV_COL_ACTIVE, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_ACTIVE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_ACTIVE);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_rv_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", RV_COL_FILENAME, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_FILENAME));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_FILENAME);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_FILTER));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_FILTER);

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_plot_rv_observed_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT(phoebe_plot_rv_observed_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT(phoebe_plot_rv_observed_combobox), renderer, "text", RV_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", RV_COL_ITYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_ITYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_ITYPE_STR);

    printf("RV indep column number upon creation: %d\n", GPOINTER_TO_UINT(g_object_get_data ((GObject*) column, "column_id")));

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", RV_COL_DTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_DTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_DTYPE_STR);

    printf("RV dep column number upon creation: %d\n", GPOINTER_TO_UINT(g_object_get_data ((GObject*) column, "column_id")));

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", RV_COL_WTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_WTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_WTYPE_STR);

    printf("RV inweight column number upon creation: %d\n", GPOINTER_TO_UINT(g_object_get_data ((GObject*) column, "column_id")));

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", RV_COL_SIGMA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_SIGMA));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_SIGMA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", RV_COL_X1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_X1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_X1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", RV_COL_X2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_X2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_X2);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", RV_COL_Y1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_Y1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_Y1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", RV_COL_Y2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_Y2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_Y2);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_data_rv_treeview,            rv_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_rv_ld_treeview,         rv_model);
	gtk_combo_box_set_model ((GtkComboBox*)phoebe_plot_rv_observed_combobox,   rv_model);

    return SUCCESS;
}


int gui_init_spots_treeview  (GladeXML *phoebe_window)
{
    // g_print("---------- Initializing spots treeview -------------\n");

    phoebe_para_surf_spots_treeview = glade_xml_get_widget (phoebe_window, "phoebe_para_surf_spots_treeview");

    GtkTreeModel *spots_model = spots_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Adjust", renderer, "active", SPOTS_COL_ADJUST, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_ADJUST));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_ADJUST);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_para_surf_spots_adjust_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Source", renderer, "text", SPOTS_COL_SOURCE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_SOURCE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_SOURCE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Latitude", renderer, "text", SPOTS_COL_LAT, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LAT));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_LAT);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Longitude", renderer, "text", SPOTS_COL_LON, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LON));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_LON);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Radius", renderer, "text", SPOTS_COL_RAD, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_RAD));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_RAD);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temperature", renderer, "text", SPOTS_COL_TEMP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_TEMP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_TEMP);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_surf_spots_treeview, spots_model);

    return SUCCESS;
}

//int gui_init_datasheets(GladeXML *phoebe_window)
//{
//    phoebe_sidesheet_data_treeview = glade_xml_get_widget (phoebe_window, "phoebe_sidesheet_data_treeview");
//    phoebe_sidesheet_fitt_treeview = glade_xml_get_widget (phoebe_window, "phoebe_sidesheet_fitt_treeview");
//
//    GtkTreeModel *data_model = datasheets_model_create();
//    GtkTreeModel *fitt_model = datasheets_model_create();
//
//    GtkCellRenderer     *renderer;
//    GtkTreeViewColumn   *column;
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Parameter", renderer, "text", DS_COL_PARAM_NAME, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, column, DS_COL_PARAM_NAME);
//
//    renderer    = gtk_cell_renderer_toggle_new ();
//    column      = gtk_tree_view_column_new_with_attributes("TBA", renderer, "active", DS_COL_PARAM_TBA, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_data_treeview, column, DS_COL_PARAM_TBA);
//
//    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_sidesheet_data_tba_checkbutton_toggled), NULL);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Parameter", renderer, "text", DS_COL_PARAM_NAME, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_data_treeview, column, DS_COL_PARAM_NAME);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", DS_COL_PARAM_VALUE, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, column, DS_COL_PARAM_VALUE);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", DS_COL_PARAM_VALUE, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_data_treeview, column, DS_COL_PARAM_VALUE);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Error", renderer, "text", DS_COL_PARAM_ERROR, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, column, DS_COL_PARAM_ERROR);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Error", renderer, "text", DS_COL_PARAM_ERROR, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_data_treeview, column, DS_COL_PARAM_ERROR);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Step", renderer, "text", DS_COL_PARAM_STEP, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, column, DS_COL_PARAM_STEP);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Min", renderer, "text", DS_COL_PARAM_MIN, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, column, DS_COL_PARAM_MIN);
//
//    renderer    = gtk_cell_renderer_text_new ();
//    column      = gtk_tree_view_column_new_with_attributes("Max", renderer, "text", DS_COL_PARAM_MAX, NULL);
//    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, column, DS_COL_PARAM_MAX);
//
//    gtk_tree_view_set_model ((GtkTreeView*)phoebe_sidesheet_data_treeview, data_model);
//    gtk_tree_view_set_model ((GtkTreeView*)phoebe_sidesheet_fitt_treeview, fitt_model);
//
//    fill_datasheets();
//
//    return SUCCESS;
//}
//
//int fill_datasheets()
//{
//    GtkTreeModel *model;
//    GtkTreeIter iter;
//
//    int i;
//	PHOEBE_parameter_list *list;
//
//	model = gtk_tree_view_get_model((GtkTreeView*)phoebe_sidesheet_data_treeview);
//	gtk_list_store_clear((GtkListStore*)model);
//
//	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++)
//	{
//		list = PHOEBE_pt->bucket[i];
//		while (list)
//		{
//            gtk_list_store_append((GtkListStore*)model, &iter);
//            gtk_list_store_set((GtkListStore*)model, &iter, DS_COL_PARAM_TBA,   list->par->tba,
//                                                            DS_COL_PARAM_NAME,  list->par->qualifier,
//                                                            DS_COL_PARAM_VALUE, list->par->value,
//                                                            DS_COL_PARAM_ERROR, 0.0, -1);
//
//			list = list->next;
//		}
//	}
//
//	model = gtk_tree_view_get_model((GtkTreeView*)phoebe_sidesheet_fitt_treeview);
//	gtk_list_store_clear((GtkListStore*)model);
//
//	list = PHOEBE_pt->lists.marked_tba;
//	while (list)
//		{
//            gtk_list_store_append((GtkListStore*)model, &iter);
//            gtk_list_store_set((GtkListStore*)model, &iter, DS_COL_PARAM_NAME,  list->par->qualifier,
//                                                            DS_COL_PARAM_VALUE, list->par->value,
//                                                            DS_COL_PARAM_ERROR, 0.0,
//                                                            DS_COL_PARAM_STEP,  list->par->step,
//                                                            DS_COL_PARAM_MIN,   list->par->min,
//                                                            DS_COL_PARAM_MAX,   list->par->max, -1);
//
//			list = list->next;
//		}
//
//	return SUCCESS;
//}

static void cell_data_func (GtkCellLayout *cell_layout,
                            GtkCellRenderer *renderer,
                            GtkTreeModel *model,
                            GtkTreeIter *iter,
                            gpointer data)
{
	if(gtk_tree_model_iter_has_child(model, iter)) g_object_set(renderer, "sensitive", FALSE, NULL);
	else g_object_set(renderer, "sensitive", TRUE, NULL);
}

int gui_init_filter_combobox (GtkWidget *combo_box)
{
	GtkTreeStore 		*store;
	GtkTreeIter 		 toplevel, child;
	GtkCellRenderer 	*renderer;

	int i;

	char par[255] = "";
	char set[255];
	char name[255];

	store = gtk_tree_store_new (2, G_TYPE_STRING, G_TYPE_INT);

	gtk_combo_box_set_model (GTK_COMBO_BOX(combo_box), GTK_TREE_MODEL (store));
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (combo_box));

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT(combo_box), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT(combo_box), renderer, "text", 0);

	gtk_cell_layout_set_cell_data_func(GTK_CELL_LAYOUT(combo_box), renderer, cell_data_func, NULL, NULL);

	for(i=0;i<PHOEBE_passbands_no;i++){
		sprintf(set, "%s",(PHOEBE_passbands[i])->set);
		sprintf(name, "%s  (%.0lfnm)",(PHOEBE_passbands[i])->name, (PHOEBE_passbands[i])->effwl/10.);

		if(strcmp(par, set)){
			strcpy(par, set);
			gtk_tree_store_append (store, &toplevel, NULL);
			gtk_tree_store_set (store, &toplevel, 0, par, 1, i, -1);
			gtk_tree_store_append (store, &child, &toplevel);
			gtk_tree_store_set (store, &child, 0, name, 1, i, -1);
		}
		else
		{
			gtk_tree_store_append (store, &child, &toplevel);
			gtk_tree_store_set (store, &child, 0, name, 1, i, -1);
		}
	}

	g_object_unref (store);

	return SUCCESS;
}

GtkTreeModel *lc_model_create()
{
    GtkListStore *model = gtk_list_store_new(LC_COL_COUNT,          /* number of columns    */
                                             G_TYPE_BOOLEAN,        /* active               */
                                             G_TYPE_STRING,         /* filename             */
                                             G_TYPE_STRING,         /* passband             */
                                             G_TYPE_INT,	        /* passband number      */
                                             G_TYPE_INT,            /* itype                */
                                             G_TYPE_STRING,         /* itype as string      */
                                             G_TYPE_INT,            /* dtype                */
                                             G_TYPE_STRING,         /* dtype as string      */
                                             G_TYPE_INT,            /* wtype                */
                                             G_TYPE_STRING,         /* wtype as string      */
                                             G_TYPE_DOUBLE,         /* sigma                */
                                             G_TYPE_STRING,         /* level weighting      */
                                             G_TYPE_DOUBLE,         /* hla                  */
                                             G_TYPE_DOUBLE,         /* cla                  */
                                             G_TYPE_DOUBLE,         /* opsf                 */
                                             G_TYPE_DOUBLE,         /* el3                  */
                                             G_TYPE_DOUBLE,         /* extinction           */
                                             G_TYPE_DOUBLE,         /* lcx1                 */
                                             G_TYPE_DOUBLE,         /* lcx2                 */
                                             G_TYPE_DOUBLE,         /* lcy1                 */
                                             G_TYPE_DOUBLE);        /* lcy2                 */
    return (GtkTreeModel*)model;
}

GtkTreeModel *rv_model_create()
{
    GtkListStore *model = gtk_list_store_new(RV_COL_COUNT,          /* number of columns    */
                                             G_TYPE_BOOLEAN,        /* active               */
                                             G_TYPE_STRING,         /* filename             */
                                             G_TYPE_STRING,         /* passband             */
                                             G_TYPE_INT,            /* itype                */
                                             G_TYPE_STRING,         /* itype as string      */
                                             G_TYPE_INT,            /* dtype                */
                                             G_TYPE_STRING,         /* dtype as string      */
                                             G_TYPE_INT,            /* wtype                */
                                             G_TYPE_STRING,         /* wtype as string      */
                                             G_TYPE_DOUBLE,         /* sigma                */
                                             G_TYPE_DOUBLE,         /* rvx1                 */
                                             G_TYPE_DOUBLE,         /* rvx2                 */
                                             G_TYPE_DOUBLE,         /* rvy1                 */
                                             G_TYPE_DOUBLE);        /* rvy2                 */
    return (GtkTreeModel*)model;
}

GtkTreeModel *spots_model_create()
{
    GtkListStore *model = gtk_list_store_new(SPOTS_COL_COUNT,       /* number of columns    */
                                             G_TYPE_BOOLEAN,        /* adjustable           */
                                             G_TYPE_INT,            /* source               */
                                             G_TYPE_STRING,			/* source as string		*/
                                             G_TYPE_DOUBLE,         /* latitude             */
                                             G_TYPE_BOOLEAN,        /* latitude    adjust   */
                                             G_TYPE_DOUBLE,         /* latitude    step     */
                                             G_TYPE_DOUBLE,         /* latitude    min      */
                                             G_TYPE_DOUBLE,         /* latitude    max      */
                                             G_TYPE_DOUBLE,         /* longitude            */
                                             G_TYPE_BOOLEAN,        /* longitude   adjust   */
                                             G_TYPE_DOUBLE,         /* longitude   step     */
                                             G_TYPE_DOUBLE,         /* longitude   min      */
                                             G_TYPE_DOUBLE,         /* longitude   max      */
                                             G_TYPE_DOUBLE,         /* radius               */
                                             G_TYPE_BOOLEAN,        /* radius      adjust   */
                                             G_TYPE_DOUBLE,         /* radius      step     */
                                             G_TYPE_DOUBLE,         /* radius      min      */
                                             G_TYPE_DOUBLE,         /* radius      max      */
                                             G_TYPE_DOUBLE,         /* temperature          */
                                             G_TYPE_BOOLEAN,        /* temperature adjust   */
                                             G_TYPE_DOUBLE,         /* temperature step     */
                                             G_TYPE_DOUBLE,         /* temperature min      */
                                             G_TYPE_DOUBLE);        /* temperature max      */

    return (GtkTreeModel*)model;
}

//GtkTreeModel *datasheets_model_create()
//{
//    GtkListStore *model = gtk_list_store_new(DS_COL_COUNT,          /* number of columns    */
//                                             G_TYPE_BOOLEAN,        /* parameter tba        */
//                                             G_TYPE_STRING ,        /* parameter name       */
//                                             G_TYPE_DOUBLE,         /* parameter value      */
//                                             G_TYPE_DOUBLE,         /* parameter error      */
//                                             G_TYPE_DOUBLE,         /* parameter step       */
//                                             G_TYPE_DOUBLE,         /* parameter min        */
//                                             G_TYPE_DOUBLE);        /* parameter max        */
//    return (GtkTreeModel*)model;
//}
