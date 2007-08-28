#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_types.h"

int gui_init_treeviews ()
{
    gui_init_lc_treeviews			();
    gui_init_rv_treeviews   		();
    gui_init_spots_treeview 		();
    gui_init_sidesheet_res_treeview ();
    gui_fill_sidesheet_res_treeview ();
    gui_init_sidesheet_fit_treeview ();
    gui_fill_sidesheet_fit_treeview ();

    return SUCCESS;
}

int gui_init_lc_treeviews ()
{
	GtkWidget *phoebe_data_lc_treeview 				= gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
	GtkWidget *phoebe_para_lc_levels_treeview 		= gui_widget_lookup ("phoebe_para_lc_levels_treeview")->gtk;
	GtkWidget *phoebe_para_lc_el3_treeview 			= gui_widget_lookup ("phoebe_para_lc_el3_treeview")->gtk;
	GtkWidget *phoebe_para_lc_levweight_treeview 	= gui_widget_lookup ("phoebe_para_lc_levweight_treeview")->gtk;
	GtkWidget *phoebe_para_lc_ld_treeview 			= gui_widget_lookup ("phoebe_para_lc_ld_treeview")->gtk;
	GtkWidget *phoebe_plot_lc_observed_combobox 	= gui_widget_lookup ("phoebe_plot_lc_observed_combobox")->gtk;

    GtkTreeModel *lc_model = (GtkTreeModel *) gtk_list_store_new (
		LC_COL_COUNT,          /* number of columns    */
		G_TYPE_BOOLEAN,        /* active               */
		G_TYPE_STRING,         /* filename             */
		G_TYPE_STRING,         /* passband             */
		G_TYPE_INT,	           /* passband number      */
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

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Active", renderer, "active", LC_COL_ACTIVE, NULL);
    g_object_set_data ((GObject *) column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data ((GObject *) column, "column_id",   GUINT_TO_POINTER (LC_COL_ACTIVE));
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
    g_signal_connect (renderer, "toggled", GTK_SIGNAL_FUNC (on_phoebe_data_lc_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    g_object_set_data ((GObject *) column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data ((GObject *) column, "column_id", GUINT_TO_POINTER (LC_COL_FILENAME));
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    g_object_set_data ((GObject *) column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data ((GObject *) column, "column_id", GUINT_TO_POINTER (LC_COL_FILTER));
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
/*
    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_levels_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_levweight_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_ld_treeview, column, -1);
*/
	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox), renderer, "text", LC_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Col. 1", renderer, "text", LC_COL_ITYPE_STR, NULL);
    g_object_set_data ((GObject*) column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data ((GObject*) column, "column_id", GUINT_TO_POINTER (LC_COL_ITYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*) phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Col. 2", renderer, "text", LC_COL_DTYPE_STR, NULL);
    g_object_set_data((GObject*) column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*) column, "column_id", GUINT_TO_POINTER (LC_COL_DTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*) phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", LC_COL_WTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_WTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", LC_COL_SIGMA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_lc_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_SIGMA));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Level weighting", renderer, "text", LC_COL_LEVWEIGHT, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_levweight_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_LEVWEIGHT));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levweight_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_levels_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_HLA));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_levels_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_CLA));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Opacity function", renderer, "text", LC_COL_OPSF, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_el3_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_OPSF));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Third light", renderer, "text", LC_COL_EL3, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_el3_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_EL3));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Extinction", renderer, "text", LC_COL_EXTINCTION, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_el3_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_EXTINCTION));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", LC_COL_X1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_X1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", LC_COL_X2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_X2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", LC_COL_Y1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_Y1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", LC_COL_Y2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_lc_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(LC_COL_Y2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    gtk_tree_view_set_model ((GtkTreeView *) phoebe_data_lc_treeview,            lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_el3_treeview,        lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_levels_treeview,     lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_levweight_treeview,  lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_ld_treeview,         lc_model);
	gtk_combo_box_set_model ((GtkComboBox *) phoebe_plot_lc_observed_combobox,   lc_model);

    return SUCCESS;
}

int gui_init_rv_treeviews ()
{
	GtkWidget *phoebe_data_rv_treeview 			= gui_widget_lookup ("phoebe_data_rv_treeview")->gtk;
	GtkWidget *phoebe_para_rv_ld_treeview 		= gui_widget_lookup ("phoebe_para_rv_ld_treeview")->gtk;
	GtkWidget *phoebe_plot_rv_observed_combobox = gui_widget_lookup ("phoebe_plot_rv_observed_combobox")->gtk;

    GtkTreeModel *rv_model = (GtkTreeModel*)gtk_list_store_new(
		RV_COL_COUNT,          /* number of columns    */
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

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", RV_COL_ACTIVE, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_ACTIVE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_rv_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", RV_COL_FILENAME, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_FILENAME));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_FILTER));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_plot_rv_observed_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT(phoebe_plot_rv_observed_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT(phoebe_plot_rv_observed_combobox), renderer, "text", RV_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", RV_COL_ITYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_ITYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    printf("RV indep column number upon creation: %d\n", GPOINTER_TO_UINT(g_object_get_data ((GObject*) column, "column_id")));

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", RV_COL_DTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_DTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    printf("RV dep column number upon creation: %d\n", GPOINTER_TO_UINT(g_object_get_data ((GObject*) column, "column_id")));

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", RV_COL_WTYPE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_WTYPE_STR));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    printf("RV inweight column number upon creation: %d\n", GPOINTER_TO_UINT(g_object_get_data ((GObject*) column, "column_id")));

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", RV_COL_SIGMA, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_data_rv_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_SIGMA));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", RV_COL_X1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_X1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", RV_COL_X2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_X2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", RV_COL_Y1, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_Y1));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", RV_COL_Y2, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_rv_ld_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RV_COL_Y2));
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_data_rv_treeview,            rv_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_rv_ld_treeview,         rv_model);
	gtk_combo_box_set_model ((GtkComboBox*)phoebe_plot_rv_observed_combobox,   rv_model);

    return SUCCESS;
}


int gui_init_spots_treeview  ()
{
	GtkWidget *phoebe_para_surf_spots_treeview = gui_widget_lookup("phoebe_para_surf_spots_treeview")->gtk;

    GtkTreeModel *spots_model = (GtkTreeModel*)gtk_list_store_new(
		SPOTS_COL_COUNT,       /* number of columns    */
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

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Adjust", renderer, "active", SPOTS_COL_ADJUST, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_ADJUST));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_para_surf_spots_adjust_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Source", renderer, "text", SPOTS_COL_SOURCE_STR, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_SOURCE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Latitude", renderer, "text", SPOTS_COL_LAT, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LAT));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. step", renderer, "text", SPOTS_COL_LATSTEP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LATSTEP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. min", renderer, "text", SPOTS_COL_LATMIN, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LATMIN));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. max", renderer, "text", SPOTS_COL_LATMAX, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LATMAX));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Longitude", renderer, "text", SPOTS_COL_LON, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LON));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. step", renderer, "text", SPOTS_COL_LONSTEP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LONSTEP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. min", renderer, "text", SPOTS_COL_LONMIN, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LONMIN));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. max", renderer, "text", SPOTS_COL_LONMAX, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_LONMAX));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Radius", renderer, "text", SPOTS_COL_RAD, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_RAD));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. step", renderer, "text", SPOTS_COL_RADSTEP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_RADSTEP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. min", renderer, "text", SPOTS_COL_RADMIN, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_RADMIN));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. max", renderer, "text", SPOTS_COL_RADMAX, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_RADMAX));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temperature", renderer, "text", SPOTS_COL_TEMP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_TEMP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. step", renderer, "text", SPOTS_COL_TEMPSTEP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_TEMPSTEP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. min", renderer, "text", SPOTS_COL_TEMPMIN, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_TEMPMIN));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. max", renderer, "text", SPOTS_COL_TEMPMAX, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_para_surf_spots_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(SPOTS_COL_TEMPMAX));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, -1);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_surf_spots_treeview, spots_model);

    return SUCCESS;
}

int gui_init_sidesheet_res_treeview()
{
	GtkWidget *phoebe_sidesheet_res_treeview = gui_widget_lookup("phoebe_sidesheet_res_treeview")->gtk;

	GtkTreeModel *model = (GtkTreeModel*)gtk_list_store_new(
		RS_COL_COUNT,		/* Number of columns 	*/
		G_TYPE_STRING,		/* Parameter name		*/
		G_TYPE_DOUBLE);		/* Parameter value		*/

	GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Parameter", renderer, "text", RS_COL_PARAM_NAME, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_res_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RS_COL_PARAM_NAME));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_res_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", RS_COL_PARAM_VALUE, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_res_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(RS_COL_PARAM_VALUE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_res_treeview, column, -1);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_sidesheet_res_treeview, model);

    return SUCCESS;
}

int gui_init_sidesheet_fit_treeview()
{
	GtkWidget *phoebe_sidesheet_fit_treeview = gui_widget_lookup("phoebe_sidesheet_fit_treeview")->gtk;

	GtkTreeModel *model = (GtkTreeModel*)gtk_list_store_new(
		FS_COL_COUNT,		/* Number of columns	*/
		G_TYPE_STRING,		/* Parameter name		*/
		G_TYPE_DOUBLE,		/* Parameter value		*/
		G_TYPE_DOUBLE,		/* Parameter step		*/
		G_TYPE_DOUBLE,		/* Parameter min		*/
		G_TYPE_DOUBLE);		/* Parameter max		*/

	GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Parameter", renderer, "text", FS_COL_PARAM_NAME, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_fit_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(FS_COL_PARAM_NAME));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", FS_COL_PARAM_VALUE, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_fit_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(FS_COL_PARAM_VALUE));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Step", renderer, "text", FS_COL_PARAM_STEP, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_fit_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(FS_COL_PARAM_STEP));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Min", renderer, "text", FS_COL_PARAM_MIN, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_fit_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(FS_COL_PARAM_MIN));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Max", renderer, "text", FS_COL_PARAM_MAX, NULL);
    g_object_set_data((GObject*)column, "parent_tree", phoebe_sidesheet_fit_treeview);
    g_object_set_data((GObject*)column, "column_id", GUINT_TO_POINTER(FS_COL_PARAM_MAX));
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    gtk_tree_view_set_model((GtkTreeView*)phoebe_sidesheet_fit_treeview, model);

    return SUCCESS;
}

static void filter_cell_data_func  (GtkCellLayout *cell_layout,
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

	gtk_cell_layout_set_cell_data_func(GTK_CELL_LAYOUT(combo_box), renderer, filter_cell_data_func, NULL, NULL);

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

int gui_fill_sidesheet_res_treeview()
{
	int status = 0;

	GtkTreeView *phoebe_sidesheet_res_treeview = (GtkTreeView*)gui_widget_lookup("phoebe_sidesheet_res_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_sidesheet_res_treeview);
	GtkTreeIter iter;

	gtk_list_store_clear((GtkListStore*)model);

	PHOEBE_parameter *par;
	double value;

	par = phoebe_parameter_lookup("phoebe_plum1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_plum2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mass1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mass2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_radius1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_radius2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mbol1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mbol2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_logg1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_logg2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_sbr1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_sbr2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, par->qualifier, RS_COL_PARAM_VALUE, value, -1);

	return status;
}

int gui_fill_sidesheet_fit_treeview()
{
	int status = 0;

	GtkTreeView *phoebe_sidesheet_fit_treeview = (GtkTreeView*)gui_widget_lookup("phoebe_sidesheet_fit_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_sidesheet_fit_treeview);
	GtkTreeIter iter;

	gtk_list_store_clear((GtkListStore*)model);

	PHOEBE_parameter_list *pars_tba = phoebe_parameter_list_get_marked_tba();
	PHOEBE_parameter *par;
	double value, step, min, max;

	while(pars_tba){
		par = pars_tba->par;

		status = phoebe_parameter_get_value(par, &value);
		status = phoebe_parameter_get_step(par, &step);
		status = phoebe_parameter_get_lower_limit(par, &min);
		status = phoebe_parameter_get_upper_limit(par, &max);

		gtk_list_store_append((GtkListStore*)model, &iter);
		gtk_list_store_set((GtkListStore*)model, &iter, FS_COL_PARAM_NAME, par->qualifier,
														FS_COL_PARAM_VALUE, value,
														FS_COL_PARAM_STEP, step,
														FS_COL_PARAM_MIN, min,
														FS_COL_PARAM_MAX, max, -1);
		pars_tba = pars_tba->next;
	}

	return status;
}
