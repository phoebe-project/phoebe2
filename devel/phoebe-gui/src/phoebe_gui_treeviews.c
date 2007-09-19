#include <stdlib.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_global.h"
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
    gui_init_fitt_mf_treeview		();
    gui_init_fitt_curve_treeview	();

    return SUCCESS;
}

int gui_reinit_treeviews()
{
	gui_reinit_lc_treeviews		();
	gui_reinit_rv_treeviews		();
	gui_reinit_spots_treeview	();

	return SUCCESS;
}

int gui_init_lc_treeviews ()
{
	GtkWidget *phoebe_data_lc_treeview 				= gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
	GtkWidget *phoebe_para_lc_levels_treeview 		= gui_widget_lookup ("phoebe_para_lc_levels_treeview")->gtk;
	GtkWidget *phoebe_para_lc_el3_treeview 			= gui_widget_lookup ("phoebe_para_lc_el3_treeview")->gtk;
	GtkWidget *phoebe_para_lc_levweight_treeview 	= gui_widget_lookup ("phoebe_para_lc_levweight_treeview")->gtk;
	GtkWidget *phoebe_para_lc_ld_treeview 			= gui_widget_lookup ("phoebe_para_lc_ld_treeview")->gtk;
	GtkWidget *phoebe_fitt_third_treeview           = gui_widget_lookup ("phoebe_fitt_third_treeview")->gtk;

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
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
    g_signal_connect (renderer, "toggled", GTK_SIGNAL_FUNC (on_phoebe_data_lc_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_sizing (column, GTK_TREE_VIEW_COLUMN_AUTOSIZE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);

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

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_fitt_third_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Col. 1", renderer, "text", LC_COL_ITYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*) phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Col. 2", renderer, "text", LC_COL_DTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*) phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", LC_COL_WTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", LC_COL_SIGMA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Level weighting", renderer, "text", LC_COL_LEVWEIGHT, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levweight_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_fitt_third_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_fitt_third_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Opacity function", renderer, "text", LC_COL_OPSF, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Third light", renderer, "text", LC_COL_EL3, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Third light", renderer, "text", LC_COL_EL3, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_fitt_third_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Extinction", renderer, "text", LC_COL_EXTINCTION, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", LC_COL_X1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", LC_COL_X2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", LC_COL_Y1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", LC_COL_Y2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, -1);

    g_signal_connect (lc_model, "row_changed", GTK_SIGNAL_FUNC (gui_on_lc_model_row_changed), NULL);

    gtk_tree_view_set_model ((GtkTreeView *) phoebe_data_lc_treeview,            lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_el3_treeview,        lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_levels_treeview,     lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_levweight_treeview,  lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_ld_treeview,         lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_fitt_third_treeview,         lc_model);

    return SUCCESS;
}

int gui_reinit_lc_treeviews ()
{
	int i;
	int lcno;
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
	GtkListStore *store = GTK_LIST_STORE(gtk_tree_view_get_model(GTK_TREE_VIEW(gui_widget_lookup("phoebe_data_lc_treeview")->gtk)));
	GtkTreeIter iter;

	gtk_list_store_clear(store);

	status = phoebe_parameter_get_value(par, &lcno);
	for(i = 0; i < lcno; i++){
		gtk_list_store_append(store, &iter);
	}

	return status;
}

int gui_init_lc_obs_combobox()
{
	GtkWidget 		*phoebe_plot_lc_observed_combobox 	= gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox")->gtk;
	GtkTreeModel 	*lc_model			 				= GTK_TREE_MODEL(gui_widget_lookup ("phoebe_data_lc_filter")->gtk);

	GtkCellRenderer *renderer;

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT (phoebe_plot_lc_observed_combobox), renderer, "text", LC_COL_FILTER);

	gtk_combo_box_set_model ((GtkComboBox *) phoebe_plot_lc_observed_combobox,   lc_model);

    return SUCCESS;
}

int gui_init_rv_treeviews ()
{
	GtkWidget *phoebe_data_rv_treeview 			= gui_widget_lookup ("phoebe_data_rv_treeview")->gtk;
	GtkWidget *phoebe_para_rv_ld_treeview 		= gui_widget_lookup ("phoebe_para_rv_ld_treeview")->gtk;

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
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_rv_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", RV_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", RV_COL_ITYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", RV_COL_DTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", RV_COL_WTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", RV_COL_SIGMA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", RV_COL_X1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", RV_COL_X2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", RV_COL_Y1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", RV_COL_Y2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);

    g_signal_connect (rv_model, "row_changed", GTK_SIGNAL_FUNC (gui_on_rv_model_row_changed), NULL);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_data_rv_treeview,            rv_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_rv_ld_treeview,         rv_model);

    return SUCCESS;
}

int gui_reinit_rv_treeviews ()
{
	int i;
	int rvno;
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
	GtkListStore *store = GTK_LIST_STORE(gtk_tree_view_get_model(GTK_TREE_VIEW(gui_widget_lookup("phoebe_data_rv_treeview")->gtk)));
	GtkTreeIter iter;

	gtk_list_store_clear(store);

	status = phoebe_parameter_get_value(par, &rvno);
	for(i = 0; i < rvno; i++){
		gtk_list_store_append(store, &iter);
	}

	return status;
}

int gui_init_rv_obs_combobox()
{
	GtkWidget 		*phoebe_plot_rv_observed_combobox 	= gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox")->gtk;
	GtkTreeModel 	*rv_model							= GTK_TREE_MODEL(gui_widget_lookup ("phoebe_data_rv_filter")->gtk);

	GtkCellRenderer *renderer;

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_plot_rv_observed_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT (phoebe_plot_rv_observed_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT (phoebe_plot_rv_observed_combobox), renderer, "text", RV_COL_FILTER);

	gtk_combo_box_set_model ((GtkComboBox *) phoebe_plot_rv_observed_combobox,   rv_model);

    return SUCCESS;
}

int gui_init_spots_treeview  ()
{
	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;

    GtkTreeModel *spots_model = (GtkTreeModel*)gtk_list_store_new(
		SPOTS_COL_COUNT,       	/* number of columns    */
		G_TYPE_BOOLEAN,        	/* active	            */
		G_TYPE_INT,            	/* source               */
		G_TYPE_STRING,			/* source as string		*/
		G_TYPE_DOUBLE,         	/* latitude             */
		G_TYPE_BOOLEAN,        	/* latitude    adjust   */
		G_TYPE_DOUBLE,         	/* latitude    step     */
		G_TYPE_DOUBLE,         	/* latitude    min      */
		G_TYPE_DOUBLE,         	/* latitude    max      */
		G_TYPE_DOUBLE,         	/* longitude            */
		G_TYPE_BOOLEAN,        	/* longitude   adjust   */
		G_TYPE_DOUBLE,         	/* longitude   step 	*/
		G_TYPE_DOUBLE,         	/* longitude   min 	    */
		G_TYPE_DOUBLE,         	/* longitude   max	    */
		G_TYPE_DOUBLE,         	/* radius               */
		G_TYPE_BOOLEAN,        	/* radius 	   adjust   */
		G_TYPE_DOUBLE,         	/* radius      step     */
		G_TYPE_DOUBLE,         	/* radius      min      */
		G_TYPE_DOUBLE,         	/* radius      max      */
		G_TYPE_DOUBLE,         	/* temperature          */
		G_TYPE_BOOLEAN,       	/* temperature adjust   */
		G_TYPE_DOUBLE,         	/* temperature step     */
		G_TYPE_DOUBLE,         	/* temperature min      */
		G_TYPE_DOUBLE,			/* temperature max      */
		G_TYPE_BOOLEAN);        /* adjustable			*/

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", SPOTS_COL_ACTIVE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_para_spots_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Source", renderer, "text", SPOTS_COL_SOURCE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Latitude", renderer, "text", SPOTS_COL_LAT, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);

	renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. adjust", renderer, "active", SPOTS_COL_LATADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. Step", renderer, "text", SPOTS_COL_LATADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. Min", renderer, "text", SPOTS_COL_LATMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. Max", renderer, "text", SPOTS_COL_LATMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Longitude", renderer, "text", SPOTS_COL_LON, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. adjust", renderer, "active", SPOTS_COL_LONADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. Step", renderer, "text", SPOTS_COL_LONSTEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. Min", renderer, "text", SPOTS_COL_LONMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. Max", renderer, "text", SPOTS_COL_LONMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Radius", renderer, "text", SPOTS_COL_RAD, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. adjust", renderer, "active", SPOTS_COL_RADADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. Step", renderer, "text", SPOTS_COL_RADSTEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. Min", renderer, "text", SPOTS_COL_RADMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. Max", renderer, "text", SPOTS_COL_RADMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temperature", renderer, "text", SPOTS_COL_TEMP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. adjust", renderer, "active", SPOTS_COL_TEMPADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. Step", renderer, "text", SPOTS_COL_TEMPSTEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. Min", renderer, "text", SPOTS_COL_TEMPMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. Max", renderer, "text", SPOTS_COL_TEMPMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Adjust", renderer, "active", SPOTS_COL_ADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_para_spots_adjust_checkbutton_toggled), NULL);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_spots_treeview, spots_model);

    return SUCCESS;
}

int gui_reinit_spots_treeview ()
{
	int status;
	int i;
	int spots_no;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_spots_no");
	GtkListStore *store = GTK_LIST_STORE(gtk_tree_view_get_model(GTK_TREE_VIEW(gui_widget_lookup("phoebe_para_spots_treeview")->gtk)));
	GtkTreeIter iter;

	status = phoebe_parameter_get_value(par, &spots_no);
	gtk_list_store_clear(store);

	for(i = 0; i < spots_no; i++)
		gtk_list_store_append(store, &iter);

	return status;
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
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_res_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", RS_COL_PARAM_VALUE, NULL);
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
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", FS_COL_PARAM_VALUE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Step", renderer, "text", FS_COL_PARAM_STEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Min", renderer, "text", FS_COL_PARAM_MIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Max", renderer, "text", FS_COL_PARAM_MAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);

    gtk_tree_view_set_model((GtkTreeView*)phoebe_sidesheet_fit_treeview, model);

    return SUCCESS;
}

int gui_init_fitt_curve_treeview()
{
	int status = 0;

	GtkWidget *phoebe_fitt_curve_treeview = gui_widget_lookup("phoebe_fitt_second_treeview")->gtk;

	GtkTreeModel *model = (GtkTreeModel*)gtk_list_store_new(
		CURVE_COL_COUNT,	/* Number of columns	*/
		G_TYPE_STRING,		/* Curve name			*/
		G_TYPE_INT,			/* Number of points		*/
		G_TYPE_DOUBLE,		/* Old chi2				*/
		G_TYPE_DOUBLE);		/* New chi2				*/

	GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Curve", renderer, "text", CURVE_COL_NAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_curve_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Number of points", renderer, "text", CURVE_COL_NPOINTS, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_curve_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Original Chi2", renderer, "text", CURVE_COL_INITCHI2, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_curve_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("New Chi2", renderer, "text", CURVE_COL_NEWCHI2, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_curve_treeview, column, -1);

    gtk_tree_view_set_model((GtkTreeView*)phoebe_fitt_curve_treeview, model);

	return status;
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
		//sprintf(name, "%s  (%.0lfnm)",(PHOEBE_passbands[i])->name, (PHOEBE_passbands[i])->effwl/10.);
		sprintf(name, "%s:%s", (PHOEBE_passbands[i])->set, (PHOEBE_passbands[i])->name);

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
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "P. Lum. 1", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_plum2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "P. Lum. 2", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mass1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Mass 1", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mass2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Mass 2", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_radius1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Radius 1", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_radius2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Radius 2", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mbol1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Mbol 2", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_mbol2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Mbol 1", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_logg1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Log(g) 1", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_logg2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Log(g) 2", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_sbr1");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Surf. Bright. 1", RS_COL_PARAM_VALUE, value, -1);

	par = phoebe_parameter_lookup("phoebe_sbr2");
	status = phoebe_parameter_get_value(par, &value);
	gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Surf. Bright. 2", RS_COL_PARAM_VALUE, value, -1);

	return status;
}

int gui_fill_sidesheet_fit_treeview()
{
	int status = 0;

	GtkTreeView *phoebe_sidesheet_fit_treeview = (GtkTreeView*)gui_widget_lookup("phoebe_sidesheet_fit_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_sidesheet_fit_treeview);
	GtkTreeIter iter;

	PHOEBE_parameter_list *pars_tba = phoebe_parameter_list_get_marked_tba();
	PHOEBE_parameter *par;
	double value, step, min, max;

	gtk_list_store_clear((GtkListStore*)model);

	while(pars_tba){
		par = pars_tba->par;

		status = phoebe_parameter_get_value(par, &value);
		status = phoebe_parameter_get_step(par, &step);
		status = phoebe_parameter_get_min(par, &min);
		status = phoebe_parameter_get_max(par, &max);

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

void gui_on_lc_model_row_changed(GtkTreeModel *tree_model,
                             GtkTreePath  *path,
                             GtkTreeIter  *iter,
                             gpointer      user_data)
{
	PHOEBE_parameter *par = phoebe_parameter_lookup("gui_lc_plot_obsmenu");
	GtkTreeIter lc_iter;
	char *option;

	int state = gtk_tree_model_get_iter_first(tree_model, &lc_iter);

	par->menu->option = NULL;
	par->menu->optno = 0;

	while (state){
		gtk_tree_model_get(tree_model, &lc_iter, LC_COL_FILTER, &option, -1);
		if (option) phoebe_parameter_add_option(par, option);
		else break;
		state = gtk_tree_model_iter_next(tree_model, &lc_iter);
	}
}

void gui_on_rv_model_row_changed(GtkTreeModel *tree_model,
                             GtkTreePath  *path,
                             GtkTreeIter  *iter,
                             gpointer      user_data)
{
	PHOEBE_parameter *par = phoebe_parameter_lookup("gui_rv_plot_obsmenu");
	GtkTreeIter rv_iter;
	char *option;

	int state = gtk_tree_model_get_iter_first(tree_model, &rv_iter);

	par->menu->option = NULL;
	par->menu->optno = 0;

	while (state){
		gtk_tree_model_get(tree_model, &rv_iter, RV_COL_FILTER, &option, -1);
		if (option) phoebe_parameter_add_option(par, option);
		else break;
		state = gtk_tree_model_iter_next(tree_model, &rv_iter);
	}
}

int gui_data_lc_treeview_add()
{
	int status = 0;

	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_load_lc.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_load_lc_xml                   = glade_xml_new        (glade_xml_file, NULL, NULL);

    GtkWidget *phoebe_load_lc_dialog                = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_dialog");
	GtkWidget *phoebe_load_lc_filechooserbutton     = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
    GtkWidget *phoebe_load_lc_column1_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column1_combobox");
    GtkWidget *phoebe_load_lc_column2_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column2_combobox");
    GtkWidget *phoebe_load_lc_column3_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column3_combobox");
    GtkWidget *phoebe_load_lc_sigma_spinbutton      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_sigma_spinbutton");
    GtkWidget *phoebe_load_lc_preview_textview      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_preview_textview");
    GtkWidget *phoebe_load_lc_filter_combobox       = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filter_combobox");

	g_object_unref (phoebe_load_lc_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_load_lc_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_load_lc_dialog), "PHOEBE - Add LC Data");

	gui_init_filter_combobox (phoebe_load_lc_filter_combobox);

	g_signal_connect (G_OBJECT (phoebe_load_lc_filechooserbutton),
					  "selection_changed",
					  G_CALLBACK (on_phoebe_load_lc_filechooserbutton_selection_changed),
					  (gpointer) phoebe_load_lc_preview_textview);

	/* Default values for column combo boxes: */
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column1_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column2_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column3_combobox,  0);

    gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_lc_dialog);
    switch (result){
        case GTK_RESPONSE_OK:{
            GtkTreeModel *model;
            GtkTreeIter iter;

			GtkTreeIter filter_iter;
			gint 		filter_number;
			gchar 		filter_selected[255] = "Johnson:V";

			GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

			if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_lc_filter_combobox), &filter_iter)) {
				gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_lc_filter_combobox)), &filter_iter, 1, &filter_number, -1);
				sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
			}

            PHOEBE_parameter *indep     = phoebe_parameter_lookup ("phoebe_lc_indep");
            PHOEBE_parameter *dep       = phoebe_parameter_lookup ("phoebe_lc_dep");
            PHOEBE_parameter *indweight = phoebe_parameter_lookup ("phoebe_lc_indweight");

            gtk_list_store_append ((GtkListStore*) model, &iter);
            gtk_list_store_set ((GtkListStore*) model, &iter,
								LC_COL_ACTIVE,      TRUE,
								LC_COL_FILENAME,    gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_lc_filechooserbutton),
								LC_COL_FILTER,      filter_selected,
								LC_COL_FILTERNO,	filter_number,
                                LC_COL_ITYPE,       gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_lc_column1_combobox),
                                LC_COL_ITYPE_STR,   strdup (indep->menu->option[gtk_combo_box_get_active((GtkComboBox*) phoebe_load_lc_column1_combobox)]),
                                LC_COL_DTYPE,       gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_lc_column2_combobox),
                                LC_COL_DTYPE_STR,   strdup (dep->menu->option[gtk_combo_box_get_active((GtkComboBox*) phoebe_load_lc_column2_combobox)]),
                                LC_COL_WTYPE,       gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_lc_column3_combobox),
                                LC_COL_WTYPE_STR,   strdup (indweight->menu->option[gtk_combo_box_get_active((GtkComboBox*) phoebe_load_lc_column3_combobox)]),
                                LC_COL_SIGMA,       gtk_spin_button_get_value ((GtkSpinButton*) phoebe_load_lc_sigma_spinbutton),
                                LC_COL_LEVWEIGHT,   "Poissonian scatter",
                                LC_COL_HLA,         12.566371,
                                LC_COL_CLA,         12.566371,
                                LC_COL_OPSF,        0.0,
                                LC_COL_EL3,         0.0,
                                LC_COL_EXTINCTION,  0.0,
                                LC_COL_X1,          0.5,
                                LC_COL_X2,          0.5,
                                LC_COL_Y1,          0.5,
                                LC_COL_Y2,          0.5,
								-1);

			/* Update the number of light curves parameter: */
            PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
            int lcno;

            phoebe_parameter_get_value(par, &lcno);
            phoebe_parameter_set_value(par, lcno + 1);

            printf("Number of light curves: %d\n", lcno + 1);

            /* Select the new row in the list: */
			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_lc_treeview), &iter);
        }
        break;

        case GTK_RESPONSE_CANCEL:
        break;
    }

    gtk_widget_destroy (phoebe_load_lc_dialog);

    return status;
}

int gui_data_lc_treeview_edit()
{
	int status = 0;

	GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

    if(gtk_tree_model_get_iter_first(model, &iter)){
        gchar     *glade_xml_file                       = g_build_filename    (PHOEBE_GLADE_XML_DIR, "phoebe_load_lc.glade", NULL);
		gchar     *glade_pixmap_file                    = g_build_filename    (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

        GladeXML  *phoebe_load_lc_xml                   = glade_xml_new       (glade_xml_file, NULL, NULL);

        GtkWidget *phoebe_load_lc_dialog                = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_dialog");
        GtkWidget *phoebe_load_lc_filechooserbutton     = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
        GtkWidget *phoebe_load_lc_column1_combobox      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_column1_combobox");
        GtkWidget *phoebe_load_lc_column2_combobox      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_column2_combobox");
        GtkWidget *phoebe_load_lc_column3_combobox      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_column3_combobox");
        GtkWidget *phoebe_load_lc_sigma_spinbutton      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_sigma_spinbutton");
        GtkWidget *phoebe_load_lc_preview_textview      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_preview_textview");
        GtkWidget *phoebe_load_lc_filter_combobox       = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_filter_combobox");

        gchar *filename;
        gint itype;
        gint dtype;
        gint wtype;
		gchar *filter;
        gdouble sigma;

        gchar filter_selected[255] = "Johnson:V";
		gint filter_number;
		GtkTreeIter filter_iter;

        g_object_unref(phoebe_load_lc_xml);

		gtk_window_set_icon (GTK_WINDOW (phoebe_load_lc_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
		gtk_window_set_title (GTK_WINDOW(phoebe_load_lc_dialog), "PHOEBE - Edit LC Data");

        gui_init_filter_combobox(phoebe_load_lc_filter_combobox);

		g_signal_connect (G_OBJECT (phoebe_load_lc_filechooserbutton),
						  "selection_changed",
						  G_CALLBACK (on_phoebe_load_lc_filechooserbutton_selection_changed),
						  (gpointer) phoebe_load_lc_preview_textview);

        GtkTreeSelection *selection;
        selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_lc_treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
            gtk_tree_model_get(model, &iter,    LC_COL_FILENAME, &filename,
                                                LC_COL_FILTER,   &filter,
												LC_COL_FILTERNO, &filter_number,
                                                LC_COL_ITYPE,    &itype,
                                                LC_COL_DTYPE,    &dtype,
                                                LC_COL_WTYPE,    &wtype,
                                                LC_COL_SIGMA,    &sigma, -1);

            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column1_combobox,  itype);
            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column2_combobox,  dtype);
            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column3_combobox,  wtype);
            gtk_spin_button_set_value    ((GtkSpinButton*) phoebe_load_lc_sigma_spinbutton,  sigma);

			sprintf(filter_selected, "%s", filter);

			if(filename){
	            gtk_file_chooser_set_filename((GtkFileChooser*)phoebe_load_lc_filechooserbutton, filename);
			}
        }

        gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_lc_dialog);
        switch (result){
            case GTK_RESPONSE_OK:{

				if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_lc_filter_combobox), &filter_iter)) {
					gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_lc_filter_combobox)), &filter_iter, 1, &filter_number, -1);
					sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
				}

                PHOEBE_parameter *indep     = phoebe_parameter_lookup("phoebe_lc_indep");
                PHOEBE_parameter *dep       = phoebe_parameter_lookup("phoebe_lc_dep");
                PHOEBE_parameter *indweight = phoebe_parameter_lookup("phoebe_lc_indweight");

                gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_ACTIVE,      TRUE,
                                                                LC_COL_FILENAME,    gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_lc_filechooserbutton),
                                                                LC_COL_FILTER,      filter_selected,
																LC_COL_FILTERNO,	filter_number,
                                                                LC_COL_ITYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column1_combobox),
                                                                LC_COL_ITYPE_STR,   strdup(indep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column1_combobox)]),
                                                                LC_COL_DTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column2_combobox),
                                                                LC_COL_DTYPE_STR,   strdup(dep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column2_combobox)]),
                                                                LC_COL_WTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column3_combobox),
                                                                LC_COL_WTYPE_STR,   strdup(indweight->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column3_combobox)]),
                                                                LC_COL_SIGMA,       gtk_spin_button_get_value((GtkSpinButton*)phoebe_load_lc_sigma_spinbutton),
                                                                LC_COL_LEVWEIGHT,   "Poissonian scatter",
                                                                LC_COL_HLA,         12.566371,
                                                                LC_COL_CLA,         12.566371,
                                                                LC_COL_OPSF,        0.0,
                                                                LC_COL_EL3,         0.0,
                                                                LC_COL_EXTINCTION,  0.0,
                                                                LC_COL_X1,          0.5,
                                                                LC_COL_X2,          0.5,
                                                                LC_COL_Y1,          0.5,
                                                                LC_COL_Y2,          0.5, -1);
            }
            break;

            case GTK_RESPONSE_CANCEL:
            break;
        }
        gtk_widget_destroy (phoebe_load_lc_dialog);
    }

    return status;
}

int gui_data_lc_treeview_remove()
{
	int status = 0;

	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_lc_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        gtk_list_store_remove((GtkListStore*)model, &iter);

        PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
        int lcno;

        phoebe_parameter_get_value(par, &lcno);
        phoebe_parameter_set_value(par, lcno - 1);

        printf("Number of light curves: %d\n", lcno - 1);
    }

    return status;
}

int gui_data_rv_treeview_add()
{
	int status = 0;

	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_load_rv.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
	gchar	  *dir;

	GladeXML  *phoebe_load_rv_xml                   = glade_xml_new        (glade_xml_file, NULL, NULL);

    GtkWidget *phoebe_load_rv_dialog                = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_dialog");
	GtkWidget *phoebe_load_rv_filechooserbutton     = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_filechooserbutton");
    GtkWidget *phoebe_load_rv_column1_combobox      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_column1_combobox");
    GtkWidget *phoebe_load_rv_column2_combobox      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_column2_combobox");
    GtkWidget *phoebe_load_rv_column3_combobox      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_column3_combobox");
    GtkWidget *phoebe_load_rv_sigma_spinbutton      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_sigma_spinbutton");
    GtkWidget *phoebe_load_rv_preview_textview      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_preview_textview");

    GtkWidget *phoebe_load_rv_filter_combobox       = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filter_combobox");

	gtk_window_set_icon (GTK_WINDOW (phoebe_load_rv_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_load_rv_dialog), "PHOEBE - Add RV Data");

    gui_init_filter_combobox(phoebe_load_rv_filter_combobox);

	g_signal_connect (G_OBJECT (phoebe_load_rv_filechooserbutton),
					  "selection_changed",
					  G_CALLBACK (on_phoebe_load_rv_filechooserbutton_selection_changed),
					  (gpointer) phoebe_load_rv_preview_textview);

    g_object_unref(phoebe_load_rv_xml);

    GtkTreeModel *model;
    GtkTreeIter iter;

	/* Default values for column combo boxes: */
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column1_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column2_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column3_combobox,  0);

	phoebe_config_entry_get ("PHOEBE_DATA_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)phoebe_load_rv_filechooserbutton, dir);

    gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_rv_dialog);
    switch (result){
        case GTK_RESPONSE_OK:{

			GtkTreeIter filter_iter;
			gint 		filter_number;
			gchar 		filter_selected[255] = "Johnson:V";

			GtkWidget *phoebe_data_rv_treeview = gui_widget_lookup("phoebe_data_rv_treeview")->gtk;
            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

			if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_rv_filter_combobox), &filter_iter)) {
				gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_rv_filter_combobox)), &filter_iter, 1, &filter_number, -1);
				sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
			}

            PHOEBE_parameter *indep     = phoebe_parameter_lookup("phoebe_rv_indep");
            PHOEBE_parameter *dep       = phoebe_parameter_lookup("phoebe_rv_dep");
            PHOEBE_parameter *indweight = phoebe_parameter_lookup("phoebe_rv_indweight");

            gtk_list_store_append((GtkListStore*)model, &iter);
            gtk_list_store_set((GtkListStore*)model, &iter, RV_COL_ACTIVE,      TRUE,
                                                            RV_COL_FILENAME,    gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_rv_filechooserbutton),
                                                            RV_COL_FILTER,      filter_selected,
                                                            RV_COL_ITYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox),
                                                            RV_COL_ITYPE_STR,   strdup(indep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox)]),
                                                            RV_COL_DTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox),
                                                            RV_COL_DTYPE_STR,   strdup(dep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox)]),
                                                            RV_COL_WTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column3_combobox),
                                                            RV_COL_WTYPE_STR,   strdup(indweight->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column3_combobox)]),
                                                            RV_COL_SIGMA,       gtk_spin_button_get_value((GtkSpinButton*)phoebe_load_rv_sigma_spinbutton),
                                                            RV_COL_X1,          0.5,
                                                            RV_COL_X2,          0.5,
                                                            RV_COL_Y1,          0.5,
                                                            RV_COL_Y2,          0.5, -1);

            PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
            int rvno;

            phoebe_parameter_get_value(par, &rvno);
            phoebe_parameter_set_value(par, rvno + 1);

            printf("Number of RV curves: %d\n", rvno + 1);

			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview), &iter);


        }
        break;

        case GTK_RESPONSE_CANCEL:
        break;
    }
    gtk_widget_destroy (phoebe_load_rv_dialog);

    return status;
}

int gui_data_rv_treeview_edit()
{
	int status = 0;

	GtkTreeModel *model;
    GtkTreeIter iter;

	GtkWidget *phoebe_data_rv_treeview = gui_widget_lookup("phoebe_data_rv_treeview")->gtk;
    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

    if(gtk_tree_model_get_iter_first(model, &iter)){
        gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_load_rv.glade", NULL);
		gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

        GladeXML  *phoebe_load_rv_xml                   = glade_xml_new        (glade_xml_file, NULL, NULL);
        GtkWidget *phoebe_load_rv_dialog                = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_dialog");
        GtkWidget *phoebe_load_rv_filechooserbutton     = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_filechooserbutton");
        GtkWidget *phoebe_load_rv_column1_combobox      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_column1_combobox");
        GtkWidget *phoebe_load_rv_column2_combobox      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_column2_combobox");
        GtkWidget *phoebe_load_rv_column3_combobox      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_column3_combobox");
        GtkWidget *phoebe_load_rv_sigma_spinbutton      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_sigma_spinbutton");
        GtkWidget *phoebe_load_rv_preview_textview      = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_preview_textview");
        GtkWidget *phoebe_load_rv_filter_combobox       = glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_filter_combobox");

		gchar *filename;
        gint itype;
        gint dtype;
        gint wtype;
        gdouble sigma;
        gchar *filter;

        gchar filter_selected[255] = "Johnson:V";
		gint filter_number;
		GtkTreeIter filter_iter;

		gtk_window_set_icon (GTK_WINDOW (phoebe_load_rv_dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));
		gtk_window_set_title (GTK_WINDOW(phoebe_load_rv_dialog), "PHOEBE - Edit RV Data");

        gui_init_filter_combobox(phoebe_load_rv_filter_combobox);

		g_signal_connect (G_OBJECT (phoebe_load_rv_filechooserbutton),
						  "selection_changed",
						  G_CALLBACK (on_phoebe_load_rv_filechooserbutton_selection_changed),
					 	  (gpointer) phoebe_load_rv_preview_textview);

        g_object_unref(phoebe_load_rv_xml);

        GtkTreeSelection *selection;
        selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
            gtk_tree_model_get(model, &iter,    RV_COL_FILENAME, &filename,
                                                RV_COL_FILTER,   &filter,
                                                RV_COL_ITYPE,    &itype,
                                                RV_COL_DTYPE,    &dtype,
                                                RV_COL_WTYPE,    &wtype,
                                                RV_COL_SIGMA,    &sigma, -1);

            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column1_combobox,  itype);
            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column2_combobox,  dtype);
            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column3_combobox,  wtype);
            gtk_spin_button_set_value    ((GtkSpinButton*) phoebe_load_rv_sigma_spinbutton,  sigma);

			sprintf(filter_selected, "%s", filter);

			if(filename){
	            gtk_file_chooser_set_filename((GtkFileChooser*)phoebe_load_rv_filechooserbutton, filename);
			}
        }

        gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_rv_dialog);
        switch (result){
            case GTK_RESPONSE_OK:{

				if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_rv_filter_combobox), &filter_iter)) {
					gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_rv_filter_combobox)), &filter_iter, 1, &filter_number, -1);
					sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
				}

                result++;

                PHOEBE_parameter *indep     = phoebe_parameter_lookup("phoebe_rv_indep");
                PHOEBE_parameter *dep       = phoebe_parameter_lookup("phoebe_rv_dep");
                PHOEBE_parameter *indweight = phoebe_parameter_lookup("phoebe_rv_indweight");

                gtk_list_store_set((GtkListStore*)model, &iter, RV_COL_ACTIVE,      TRUE,
                                                                RV_COL_FILENAME,    gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_rv_filechooserbutton),
                                                                RV_COL_FILTER,      filter_selected,
                                                                RV_COL_ITYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox),
                                                                RV_COL_ITYPE_STR,   strdup(indep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox)]),
                                                                RV_COL_DTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox),
                                                                RV_COL_DTYPE_STR,   strdup(dep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox)]),
                                                                RV_COL_WTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column3_combobox),
                                                                RV_COL_WTYPE_STR,   strdup(indweight->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column3_combobox)]),
                                                                RV_COL_SIGMA,       gtk_spin_button_get_value((GtkSpinButton*)phoebe_load_rv_sigma_spinbutton),
                                                                RV_COL_X1,          0.5,
                                                                RV_COL_X2,          0.5,
                                                                RV_COL_Y1,          0.5,
                                                                RV_COL_Y2,          0.5, -1);
            }
            break;

            case GTK_RESPONSE_CANCEL:
            break;
        }
        gtk_widget_destroy (phoebe_load_rv_dialog);
    }
    return status;
}

int gui_data_rv_treeview_remove()
{
	int status = 0;

	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_data_rv_treeview = gui_widget_lookup("phoebe_data_rv_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        gtk_list_store_remove((GtkListStore*)model, &iter);

        PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
        int rvno;

        phoebe_parameter_get_value(par, &rvno);
        phoebe_parameter_set_value(par, rvno - 1);

        printf("Number of RV curves: %d\n", rvno - 1);
    }

    return status;
}

int gui_init_fitt_mf_treeview()
{
	int status = 0;

	GtkWidget *phoebe_fitt_mf_treeview = gui_widget_lookup("phoebe_fitt_first_treeview")->gtk;

	GtkTreeModel *model = (GtkTreeModel*)gtk_list_store_new(
		MF_COL_COUNT,		/* Number of columns	*/
		G_TYPE_STRING,		/* Qualifier			*/
		G_TYPE_DOUBLE,		/* Initial value		*/
		G_TYPE_DOUBLE,		/* New value			*/
		G_TYPE_DOUBLE);		/* Error				*/

	GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Parameter", renderer, "text", MF_COL_QUALIFIER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_mf_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Initial value", renderer, "text", MF_COL_INITVAL, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_mf_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("New value", renderer, "text", MF_COL_NEWVAL, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_mf_treeview, column, -1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Error", renderer, "text", MF_COL_ERROR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_fitt_mf_treeview, column, -1);

    gtk_tree_view_set_model((GtkTreeView*)phoebe_fitt_mf_treeview, model);

	return status;
}


int gui_fill_fitt_mf_treeview()
{
	GtkTreeView *phoebe_fitt_mf_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_first_treeview")->gtk);
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_fitt_mf_treeview);
	GtkTreeIter iter;
	double value;

	int status = 0;

	//status = gui_get_values_from_widgets();

	gtk_list_store_clear(GTK_LIST_STORE(model));

	PHOEBE_parameter_list *pars_tba = phoebe_parameter_list_get_marked_tba();
	PHOEBE_parameter *par;

	while(pars_tba){
		par = pars_tba->par;

		status = phoebe_parameter_get_value(par, &value);

		gtk_list_store_append(GTK_LIST_STORE(model), &iter);
		gtk_list_store_set(GTK_LIST_STORE(model), &iter,
			MF_COL_QUALIFIER, par->qualifier,
			MF_COL_INITVAL, value, -1);
		pars_tba = pars_tba->next;
	}
	return status;
}


int gui_spots_parameters_marked_tba()
{
	int result = 0;

	GtkWidget *latadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_latadjust_checkbutton")->gtk;
	GtkWidget *lonadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_lonadjust_checkbutton")->gtk;
	GtkWidget *radadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_radadjust_checkbutton")->gtk;
	GtkWidget *tempadjust_checkbutton = gui_widget_lookup ("phoebe_para_spots_tempadjust_checkbutton")->gtk;

	result += 	gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(latadjust_checkbutton)) +
				gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(lonadjust_checkbutton)) +
				gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(radadjust_checkbutton)) +
				gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(tempadjust_checkbutton));

	return result;
}


int gui_para_lum_levels_edit()
{
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
 	int lcno;
	phoebe_parameter_get_value(par, &lcno);

	if(lcno>0){

		GtkTreeModel     *model;
		GtkTreeIter       iter;
		GtkTreeSelection *selection;

		gchar *passband;
		gdouble hla;
		gdouble cla;

		GtkWidget *treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
		model = gtk_tree_view_get_model((GtkTreeView*)treeview);

		treeview = gui_widget_lookup("phoebe_para_lc_levels_treeview")->gtk;

        selection = gtk_tree_view_get_selection((GtkTreeView*)treeview);
       	if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        	gtk_tree_model_get(model, &iter,    LC_COL_FILTER,	&passband,
                                                LC_COL_HLA,  	&hla,
												LC_COL_CLA, 	&cla, -1);


    		gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_levels.glade", NULL);
			gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

			GladeXML  *phoebe_levels_xml	              	= glade_xml_new        (glade_xml_file, NULL, NULL);

   			GtkWidget *phoebe_levels_dialog                	= glade_xml_get_widget (phoebe_levels_xml, "phoebe_levels_dialog");
			GtkWidget *phoebe_levels_passband_label		    = glade_xml_get_widget (phoebe_levels_xml, "phoebe_levels_passband_label");
    		GtkWidget *phoebe_levels_primary_spinbutton     = glade_xml_get_widget (phoebe_levels_xml, "phoebe_levels_primary_spinbutton");
    		GtkWidget *phoebe_levels_secondary_spinbutton   = glade_xml_get_widget (phoebe_levels_xml, "phoebe_levels_secondary_spinbutton");

			g_object_unref (phoebe_levels_xml);

			gtk_window_set_icon (GTK_WINDOW (phoebe_levels_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
			gtk_window_set_title (GTK_WINDOW(phoebe_levels_dialog), "PHOEBE - Edit Levels");

			gtk_label_set_text (GTK_LABEL (phoebe_levels_passband_label), passband);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_levels_primary_spinbutton), hla);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_levels_secondary_spinbutton), cla);

    		gint result = gtk_dialog_run ((GtkDialog*)phoebe_levels_dialog);
   			switch (result){
        		case GTK_RESPONSE_OK:{
			             		gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_HLA, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_levels_primary_spinbutton)),
                    															LC_COL_CLA, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_levels_secondary_spinbutton)), -1);
            		}

        		break;

       			case GTK_RESPONSE_CANCEL:
       			break;
   			}

    		gtk_widget_destroy (phoebe_levels_dialog);
		}
	}

	return status;
}

int gui_para_lum_el3_edit()
{
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
 	int lcno;
	phoebe_parameter_get_value(par, &lcno);

	if(lcno>0){

		GtkTreeModel     *model;
		GtkTreeIter       iter;
		GtkTreeSelection *selection;

		gchar *passband;
		gdouble el3;
		gdouble opsf;
		gdouble extinction;

		GtkWidget *treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
		model = gtk_tree_view_get_model((GtkTreeView*)treeview);

		treeview = gui_widget_lookup("phoebe_para_lc_el3_treeview")->gtk;

        selection = gtk_tree_view_get_selection((GtkTreeView*)treeview);
       	if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        	gtk_tree_model_get(model, &iter,    LC_COL_FILTER,		&passband,
                                                LC_COL_EL3,  		&el3,
                                                LC_COL_OPSF,  		&opsf,
												LC_COL_EXTINCTION, 	&extinction, -1);

    		gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_third_light.glade", NULL);
			gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

			GladeXML  *phoebe_third_light_xml	              	= glade_xml_new        (glade_xml_file, NULL, NULL);

   			GtkWidget *phoebe_third_light_dialog                = glade_xml_get_widget (phoebe_third_light_xml, "phoebe_third_light_dialog");
			GtkWidget *phoebe_third_light_passband_label		= glade_xml_get_widget (phoebe_third_light_xml, "phoebe_third_light_passband_label");
    		GtkWidget *phoebe_third_light_opacity_spinbutton    = glade_xml_get_widget (phoebe_third_light_xml, "phoebe_third_light_opacity_spinbutton");
    		GtkWidget *phoebe_third_light_el3_spinbutton   		= glade_xml_get_widget (phoebe_third_light_xml, "phoebe_third_light_el3_spinbutton");
    		GtkWidget *phoebe_third_light_extinction_spinbutton = glade_xml_get_widget (phoebe_third_light_xml, "phoebe_third_light_extinction_spinbutton");

			g_object_unref (phoebe_third_light_xml);

			gtk_window_set_icon (GTK_WINDOW (phoebe_third_light_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
			gtk_window_set_title (GTK_WINDOW(phoebe_third_light_dialog), "PHOEBE - Edit Third Light");

			gtk_label_set_text (GTK_LABEL (phoebe_third_light_passband_label), passband);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_third_light_opacity_spinbutton), opsf);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_third_light_el3_spinbutton), el3);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_third_light_extinction_spinbutton), extinction);

    		gint result = gtk_dialog_run ((GtkDialog*)phoebe_third_light_dialog);
   			switch (result){
        		case GTK_RESPONSE_OK:{
			             		gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_EL3, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_third_light_el3_spinbutton)),
																				LC_COL_OPSF, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_third_light_opacity_spinbutton)),
                    															LC_COL_EXTINCTION, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_third_light_extinction_spinbutton)), -1);
            		}
        		break;

       			case GTK_RESPONSE_CANCEL:
       			break;
   			}

    		gtk_widget_destroy (phoebe_third_light_dialog);
		}
	}

	return status;
}


int gui_fitt_levelweight_edit()
{
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
 	int lcno;
	phoebe_parameter_get_value(par, &lcno);

	if(lcno>0){

		GtkTreeModel     *model;
		GtkTreeIter       iter;
		GtkTreeSelection *selection;

		gchar *passband;
		gchar *levweight;

		GtkWidget *treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
		model = gtk_tree_view_get_model((GtkTreeView*)treeview);

		treeview = gui_widget_lookup("phoebe_para_lc_levweight_treeview")->gtk;

        selection = gtk_tree_view_get_selection((GtkTreeView*)treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
			gtk_tree_model_get(model, &iter,    LC_COL_FILTER,		&passband,
												LC_COL_LEVWEIGHT,	&levweight, -1);

    		gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_weighting.glade", NULL);
			gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

			GladeXML  *phoebe_weighting_xml	             	= glade_xml_new        (glade_xml_file, NULL, NULL);

   			GtkWidget *phoebe_weighting_dialog              = glade_xml_get_widget (phoebe_weighting_xml, "phoebe_weighting_dialog");
			GtkWidget *phoebe_weighting_passband_label		= glade_xml_get_widget (phoebe_weighting_xml, "phoebe_weighting_passband_label");
			GtkWidget *phoebe_weighting_combobox			= glade_xml_get_widget (phoebe_weighting_xml, "phoebe_weighting_combobox");

			g_object_unref (phoebe_weighting_xml);

			gtk_window_set_icon (GTK_WINDOW (phoebe_weighting_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
			gtk_window_set_title (GTK_WINDOW(phoebe_weighting_dialog), "PHOEBE - Edit Third Light");

			gtk_label_set_text (GTK_LABEL (phoebe_weighting_passband_label), passband);

			if(strcmp(levweight, "No level-dependent weighting")==0) gtk_combo_box_set_active (GTK_COMBO_BOX (phoebe_weighting_combobox), 0);
			if(strcmp(levweight, "Poissonian scatter")==0) gtk_combo_box_set_active (GTK_COMBO_BOX (phoebe_weighting_combobox), 1);
			if(strcmp(levweight, "Low light scatter")==0) gtk_combo_box_set_active (GTK_COMBO_BOX (phoebe_weighting_combobox), 2);

    		gint result = gtk_dialog_run ((GtkDialog*)phoebe_weighting_dialog);
   			switch (result){
        		case GTK_RESPONSE_OK:
					gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_LEVWEIGHT, gtk_combo_box_get_active_text (GTK_COMBO_BOX (phoebe_weighting_combobox)), -1);
        		break;

       			case GTK_RESPONSE_CANCEL:
       			break;
   			}

    		gtk_widget_destroy (phoebe_weighting_dialog);
		}
	}

	return status;
}
