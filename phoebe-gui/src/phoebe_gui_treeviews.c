#include <stdlib.h>
#include <math.h>

#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_accessories.h"

/* bool phoebe_para_spots_units_combobox_init = FALSE;*/

int gui_init_treeviews ()
{
    gui_init_lc_treeviews			();
    gui_init_rv_treeviews   		();
    gui_init_spots_treeview 		();
    gui_init_sidesheet_res_treeview ();
    gui_init_sidesheet_fit_treeview ();
    gui_fill_sidesheet_fit_treeview ();
    gui_init_fitt_mf_treeview		();
    gui_fit_statistics_treeview_init ();

    return SUCCESS;
}

int gui_reinit_treeviews ()
{
	gui_reinit_lc_treeviews		();
	gui_reinit_rv_treeviews		();
	gui_reinit_spots_treeview	();

	return SUCCESS;
}

void gui_numeric_cell_edited (GtkCellRendererText *renderer, gchar *path, gchar *new_text, gpointer user_data)
{
	/*
	 * gui_numeric_cell_edited:
	 *
	 * This is a generic callback for all text-based editable cells. It is
	 * triggered when a cell has been edited and its job is to update the
	 * value in the list store to the newly typed value.
	 *
	 * The callback should be used by *all* editable text renderers. To
	 * identify themselves to this callback properly, all renderers must
	 * attach a "column" property, i.e. the following line should be
	 * included when defining a renderer:
	 *
	 *   g_object_set_data (
	 *       G_OBJECT (renderer),
	 *       "column",
	 *       GINT_TO_POINTER (ENUMERATED_COLUMN_NAME)
	 *   );
	 *
	 * Furthermore, a signal must be passed with the user_data pointer that
	 * points to the tree model that should be modified, i.e.:
	 *
	 *   g_signal_connect (
	 *       renderer,
	 *       "edited",
	 *       GTK_SIGNAL_FUNC (gui_numeric_cell_edited),
	 *       model
	 *   );
	 *
	 * The string -> value conversion is done by atof().
	 */

	GtkTreeModel *model = (GtkTreeModel *) user_data;
	int column = GPOINTER_TO_INT (g_object_get_data (G_OBJECT (renderer), "column"));
	GtkTreeIter iter;

	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, column, atof (new_text), -1);

	return;
}

void gui_text_cell_edited (GtkCellRendererText *renderer, gchar *path, gchar *new_text, gpointer user_data)
{
	/*
	 * gui_text_cell_edited:
	 *
	 * This is a generic callback for all text-based editable cells. It is
	 * triggered when a cell has been edited and its job is to update the
	 * value in the list store to the newly typed value.
	 *
	 * The callback should be used by *all* editable text renderers. To
	 * identify themselves to this callback properly, all renderers must
	 * attach a "column" property, i.e. the following line should be
	 * included when defining a renderer:
	 *
	 *   g_object_set_data (
	 *       G_OBJECT (renderer),
	 *       "column",
	 *       GINT_TO_POINTER (ENUMERATED_COLUMN_NAME)
	 *   );
	 *
	 * Furthermore, a signal must be passed with the user_data pointer that
	 * points to the tree model that should be modified, i.e.:
	 *
	 *   g_signal_connect (
	 *       renderer,
	 *       "edited",
	 *       GTK_SIGNAL_FUNC (gui_numeric_cell_edited),
	 *       model
	 *   );
	 */

	GtkTreeModel *model = (GtkTreeModel *) user_data;
	int column = GPOINTER_TO_INT (g_object_get_data (G_OBJECT (renderer), "column"));
	GtkTreeIter iter;

	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, column, new_text, -1);

	return;
}

void gui_toggle_cell_edited (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
{
	/*
	 * gui_toggle_cell_edited:
	 *
	 * This is a generic callback for all toggle-based editable cells. It is
	 * triggered when a cell has been toggled and its job is to update the
	 * state in the list store to the newly toggled state.
	 *
	 * The callback should be used by *all* editable toggle renderers. To
	 * identify themselves to this callback properly, all renderers must
	 * attach a "column" property, i.e. the following line should be
	 * included when defining a renderer:
	 *
	 *   g_object_set_data (
	 *       G_OBJECT (renderer),
	 *       "column",
	 *       GINT_TO_POINTER (ENUMERATED_COLUMN_NAME)
	 *   );
	 *
	 * Furthermore, a signal must be passed with the user_data pointer that
	 * points to the tree model that should be modified, i.e.:
	 *
	 *   g_signal_connect (
	 *       renderer,
	 *       "toggled",
	 *       GTK_SIGNAL_FUNC (gui_numeric_cell_edited),
	 *       model
	 *   );
	 *
	 * The string -> value conversion is done by atof().
	 */

	GtkTreeModel *model = (GtkTreeModel *) user_data;
	int column = GPOINTER_TO_INT (g_object_get_data (G_OBJECT (renderer), "column"));
	GtkTreeIter iter;
	bool state = gtk_cell_renderer_toggle_get_active (renderer);

	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, column, !state, -1);

	return;
}

int gui_ld_treeview_update ()
{
#warning FINISH_LD_UPDATE_FUNCTION
	GtkWidget *treeview = gui_widget_lookup ("phoebe_para_lc_ld_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model (GTK_TREE_VIEW (treeview));
}

void gui_gdk_color_to_string (GdkColor color, 	gchar *colorstring)
{
	sprintf (colorstring, "#%04x%04x%04x", color.red, color.green, color.blue);
}

void gui_select_color(GtkCellRenderer *renderer, GtkCellEditable *editable, const gchar *path, gpointer user_data)
{
	GtkTreeModel *model = (GtkTreeModel *) user_data;
	GdkColor color;
	gchar *colorname;
	GtkColorSelection *colorsel;

	int column = GPOINTER_TO_INT (g_object_get_data (G_OBJECT (renderer), "column"));
	GtkTreeIter iter;

	GtkWidget *dialog = gtk_color_selection_dialog_new("Select Color");
	colorsel = GTK_COLOR_SELECTION(GTK_COLOR_SELECTION_DIALOG(dialog)->colorsel);

	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_tree_model_get (model, &iter, column, &colorname, -1);

	if (gdk_color_parse (colorname, &color) == TRUE) {
		gtk_color_selection_set_current_color(colorsel, &color);
	}

	if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_OK) {
		gchar colorstring[20];
		gtk_color_selection_get_current_color(colorsel, &color);
		gui_gdk_color_to_string(color, colorstring);
	
		gtk_list_store_set (GTK_LIST_STORE (model), &iter, column, colorstring, -1);
	} 

	gtk_widget_destroy(dialog);
}


int gui_init_lc_treeviews ()
{
	GtkWidget *phoebe_data_lc_treeview           = gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
	GtkWidget *phoebe_para_lc_levels_treeview    = gui_widget_lookup ("phoebe_para_lc_levels_treeview")->gtk;
	GtkWidget *phoebe_para_lc_el3_treeview       = gui_widget_lookup ("phoebe_para_lc_el3_treeview")->gtk;
	GtkWidget *phoebe_para_lc_levweight_treeview = gui_widget_lookup ("phoebe_para_lc_levweight_treeview")->gtk;
	GtkWidget *phoebe_para_lc_ld_treeview        = gui_widget_lookup ("phoebe_para_lc_ld_treeview")->gtk;
	GtkWidget *phoebe_fitt_third_treeview        = gui_widget_lookup ("phoebe_fitt_third_treeview")->gtk;
	GtkWidget *phoebe_lc_plot_treeview           = gui_widget_lookup ("phoebe_lc_plot_treeview")->gtk;

	GtkTreeModel *lc_model = (GtkTreeModel *) gtk_list_store_new (
		LC_COL_COUNT,          /* number of columns     */
		G_TYPE_BOOLEAN,        /* active                */
		G_TYPE_STRING,         /* filename              */
		G_TYPE_STRING,		   /* id				    */
		G_TYPE_STRING,         /* passband              */
		G_TYPE_INT,	           /* passband number       */
		G_TYPE_INT,            /* itype                 */
		G_TYPE_STRING,         /* itype as string       */
		G_TYPE_INT,            /* dtype                 */
		G_TYPE_STRING,         /* dtype as string       */
		G_TYPE_INT,            /* wtype                 */
		G_TYPE_STRING,         /* wtype as string       */
		G_TYPE_DOUBLE,         /* sigma                 */
		G_TYPE_STRING,         /* level weighting       */
		G_TYPE_DOUBLE,         /* hla                   */
		G_TYPE_DOUBLE,         /* cla                   */
		G_TYPE_DOUBLE,         /* opsf                  */
		G_TYPE_DOUBLE,         /* el3                   */
		G_TYPE_DOUBLE,         /* el3 in lum units      */
		G_TYPE_DOUBLE,         /* extinction            */
		G_TYPE_DOUBLE,         /* lcx1                  */
		G_TYPE_DOUBLE,         /* lcx2                  */
		G_TYPE_DOUBLE,         /* lcy1                  */
		G_TYPE_DOUBLE,         /* lcy2                  */
		G_TYPE_BOOLEAN,        /* plot observed switch  */
		G_TYPE_BOOLEAN,        /* plot synthetic switch */
		G_TYPE_STRING,         /* observed data color   */
		G_TYPE_STRING,         /* synthetic data color  */
		G_TYPE_DOUBLE          /* plot offset           */
	);

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
	gtk_tree_view_column_set_resizable (column, TRUE);
	
	renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("ID", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Filter", renderer, "text", LC_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("ID", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_el3_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("ID", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_levels_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("ID", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_levweight_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("ID", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("ID", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_fitt_third_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
	column      = gtk_tree_view_column_new_with_attributes ("Passband ID:", renderer, "text", LC_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_lc_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Indep", renderer, "text", LC_COL_ITYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Dep", renderer, "text", LC_COL_DTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Weighting", renderer, "text", LC_COL_WTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Sigma", renderer, "text", LC_COL_SIGMA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_data_lc_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Level weighting", renderer, "text", LC_COL_LEVWEIGHT, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_levweight_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView *) phoebe_para_lc_levels_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_fitt_third_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_levels_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_fitt_third_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Opacity function", renderer, "text", LC_COL_OPSF, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_el3_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Third light", renderer, "text", LC_COL_EL3, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_el3_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Third light", renderer, "text", LC_COL_EL3_LUM, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_fitt_third_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Extinction", renderer, "text", LC_COL_EXTINCTION, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_el3_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("X1", renderer, "text", LC_COL_X1, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("X2", renderer, "text", LC_COL_X2, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) phoebe_para_lc_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Y1", renderer, "text", LC_COL_Y1, NULL);
    gtk_tree_view_insert_column((GtkTreeView *) phoebe_para_lc_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Y2", renderer, "text", LC_COL_Y2, NULL);
    gtk_tree_view_insert_column((GtkTreeView *) phoebe_para_lc_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	/**************************************************************************/
	/*                                                                        */
	/*                         LC PLOTTING TREEVIEW                           */
	/*                                                                        */
	/**************************************************************************/

	renderer    = gtk_cell_renderer_toggle_new ();
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (LC_COL_PLOT_OBS));
	g_signal_connect (renderer, "toggled", GTK_SIGNAL_FUNC (gui_toggle_cell_edited), lc_model);
	column      = gtk_tree_view_column_new_with_attributes ("Observed:", renderer, "active", LC_COL_PLOT_OBS, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_lc_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_toggle_new ();
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (LC_COL_PLOT_SYN));
	g_signal_connect (renderer, "toggled", GTK_SIGNAL_FUNC (gui_toggle_cell_edited), lc_model);
	column      = gtk_tree_view_column_new_with_attributes ("Synthetic:", renderer, "active", LC_COL_PLOT_SYN, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_lc_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_combo_new ();
	g_object_set (renderer, "editable", TRUE, "background-set", TRUE, NULL);
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (LC_COL_PLOT_OBS_COLOR));
	g_signal_connect (renderer, "edited", GTK_SIGNAL_FUNC (gui_text_cell_edited), lc_model);
	g_signal_connect (renderer, "editing-started", GTK_SIGNAL_FUNC (gui_select_color), lc_model);
	column      = gtk_tree_view_column_new_with_attributes ("Obs color:", renderer, "text", LC_COL_PLOT_OBS_COLOR, "background", LC_COL_PLOT_OBS_COLOR, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_lc_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_combo_new ();
	g_object_set (renderer, "editable", TRUE, "background-set", TRUE, NULL);
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (LC_COL_PLOT_SYN_COLOR));
	g_signal_connect (renderer, "edited", GTK_SIGNAL_FUNC (gui_text_cell_edited), lc_model);
	g_signal_connect (renderer, "editing-started", GTK_SIGNAL_FUNC (gui_select_color), lc_model);
	column      = gtk_tree_view_column_new_with_attributes ("Syn color:", renderer, "text", LC_COL_PLOT_SYN_COLOR, "background", LC_COL_PLOT_SYN_COLOR, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_lc_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_text_new ();
	g_object_set (renderer, "editable", TRUE, NULL);
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (LC_COL_PLOT_OFFSET));
	g_signal_connect (renderer, "edited", GTK_SIGNAL_FUNC (gui_numeric_cell_edited), lc_model);
	column      = gtk_tree_view_column_new_with_attributes ("Y Offset:", renderer, "text", LC_COL_PLOT_OFFSET, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_lc_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	/**************************************************************************/

/* OBSOLETE:
	g_signal_connect (lc_model, "row_changed", GTK_SIGNAL_FUNC (on_phoebe_data_lc_model_row_changed), NULL);
*/

    gtk_tree_view_set_model ((GtkTreeView *) phoebe_data_lc_treeview,            lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_el3_treeview,        lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_levels_treeview,     lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_levweight_treeview,  lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_lc_ld_treeview,         lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_fitt_third_treeview,         lc_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_lc_plot_treeview,            lc_model);

    return SUCCESS;
}

int gui_reinit_lc_treeviews ()
{
	int i;
	int lcno;
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup ("phoebe_lcno");
	GtkListStore *store = GTK_LIST_STORE (gtk_tree_view_get_model (GTK_TREE_VIEW (gui_widget_lookup ("phoebe_data_lc_treeview")->gtk)));
	GtkTreeIter iter;

	gtk_list_store_clear (store);

	status = phoebe_parameter_get_value (par, &lcno);
	for(i = 0; i < lcno; i++)
		gtk_list_store_append (store, &iter);

	return status;
}

int gui_init_rv_treeviews ()
{
	GtkWidget *phoebe_data_rv_treeview 			= gui_widget_lookup ("phoebe_data_rv_treeview")->gtk;
	GtkWidget *phoebe_para_rv_ld_treeview 		= gui_widget_lookup ("phoebe_para_rv_ld_treeview")->gtk;
	GtkWidget *phoebe_rv_plot_treeview          = gui_widget_lookup ("phoebe_rv_plot_treeview")->gtk;

    GtkTreeModel *rv_model = (GtkTreeModel*) gtk_list_store_new (
		RV_COL_COUNT,          /* number of columns     */
		G_TYPE_BOOLEAN,        /* active                */
		G_TYPE_STRING,         /* filename              */
		G_TYPE_STRING,		   /* ID				    */
		G_TYPE_STRING,         /* passband              */
		G_TYPE_INT,            /* itype                 */
		G_TYPE_STRING,         /* itype as string       */
		G_TYPE_INT,            /* dtype                 */
		G_TYPE_STRING,         /* dtype as string       */
		G_TYPE_INT,            /* wtype                 */
		G_TYPE_STRING,         /* wtype as string       */
		G_TYPE_DOUBLE,         /* sigma                 */
		G_TYPE_DOUBLE,         /* rvx1                  */
		G_TYPE_DOUBLE,         /* rvx2                  */
		G_TYPE_DOUBLE,         /* rvy1                  */
		G_TYPE_DOUBLE,         /* rvy2                  */
		G_TYPE_BOOLEAN,        /* plot observed switch  */
		G_TYPE_BOOLEAN,        /* plot synthetic switch */
		G_TYPE_STRING,         /* observed data color   */
		G_TYPE_STRING,         /* synthetic data color  */
		G_TYPE_DOUBLE          /* plot offset           */
	);

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", RV_COL_ACTIVE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_rv_active_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", RV_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("ID", renderer, "text", RV_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("ID", renderer, "text", RV_COL_ID, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", RV_COL_ITYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", RV_COL_DTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", RV_COL_WTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", RV_COL_SIGMA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", RV_COL_X1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", RV_COL_X2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", RV_COL_Y1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", RV_COL_Y2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	/**************************************************************************/
	/*                                                                        */
	/*                         RV PLOTTING TREEVIEW                           */
	/*                                                                        */
	/**************************************************************************/

	renderer    = gtk_cell_renderer_text_new ();
	column      = gtk_tree_view_column_new_with_attributes ("Passband ID:", renderer, "text", RV_COL_ID, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_rv_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_toggle_new ();
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (RV_COL_PLOT_OBS));
	g_signal_connect (renderer, "toggled", GTK_SIGNAL_FUNC (gui_toggle_cell_edited), rv_model);
	column      = gtk_tree_view_column_new_with_attributes ("Observed:", renderer, "active", RV_COL_PLOT_OBS, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_rv_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_toggle_new ();
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (RV_COL_PLOT_SYN));
	g_signal_connect (renderer, "toggled", GTK_SIGNAL_FUNC (gui_toggle_cell_edited), rv_model);
	column      = gtk_tree_view_column_new_with_attributes ("Synthetic:", renderer, "active", RV_COL_PLOT_SYN, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_rv_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_combo_new ();
	g_object_set (renderer, "editable", TRUE, "background-set", TRUE, NULL);
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (RV_COL_PLOT_OBS_COLOR));
	g_signal_connect (renderer, "edited", GTK_SIGNAL_FUNC (gui_text_cell_edited), rv_model);
	g_signal_connect (renderer, "editing-started", GTK_SIGNAL_FUNC (gui_select_color), rv_model);
	column      = gtk_tree_view_column_new_with_attributes ("Obs color:", renderer, "text", RV_COL_PLOT_OBS_COLOR, "background", RV_COL_PLOT_OBS_COLOR, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_rv_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_combo_new ();
	g_object_set (renderer, "editable", TRUE, "background-set", TRUE, NULL);
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (RV_COL_PLOT_SYN_COLOR));
	g_signal_connect (renderer, "edited", GTK_SIGNAL_FUNC (gui_text_cell_edited), rv_model);
	g_signal_connect (renderer, "editing-started", GTK_SIGNAL_FUNC (gui_select_color), rv_model);
	column      = gtk_tree_view_column_new_with_attributes ("Syn color:", renderer, "text", RV_COL_PLOT_SYN_COLOR, "background", RV_COL_PLOT_SYN_COLOR, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_rv_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_text_new ();
	g_object_set (renderer, "editable", TRUE, NULL);
	g_object_set_data (G_OBJECT (renderer), "column", GINT_TO_POINTER (RV_COL_PLOT_OFFSET));
	g_signal_connect (renderer, "edited", GTK_SIGNAL_FUNC (gui_numeric_cell_edited), rv_model);
	column      = gtk_tree_view_column_new_with_attributes ("Y Offset:", renderer, "text", RV_COL_PLOT_OFFSET, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_rv_plot_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	/**************************************************************************/

    g_signal_connect (rv_model, "row_changed", GTK_SIGNAL_FUNC (on_phoebe_data_rv_model_row_changed), NULL);

    gtk_tree_view_set_model ((GtkTreeView *) phoebe_data_rv_treeview,            rv_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_para_rv_ld_treeview,         rv_model);
    gtk_tree_view_set_model ((GtkTreeView *) phoebe_rv_plot_treeview,            rv_model);

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

static void gui_spots_cell_data_function (GtkCellLayout *cell_layout, GtkCellRenderer *renderer, GtkTreeModel *model, GtkTreeIter *iter, gpointer data)
{
	int source;
	gtk_tree_model_get(model, iter, SPOTS_COL_SOURCE, &source, -1);

	if(source == 1)
		g_object_set(renderer, "text", "Primary", NULL);
	else
		g_object_set(renderer, "text", "Secondary", NULL);
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
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Source", renderer, "text", SPOTS_COL_SOURCE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_cell_data_func(column, renderer, (GtkTreeCellDataFunc)gui_spots_cell_data_function, NULL, FALSE);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Colatitude", renderer, "text", SPOTS_COL_LAT, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. adjust", renderer, "active", SPOTS_COL_LATADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. Step", renderer, "text", SPOTS_COL_LATADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. Min", renderer, "text", SPOTS_COL_LATMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lat. Max", renderer, "text", SPOTS_COL_LATMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Longitude", renderer, "text", SPOTS_COL_LON, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. adjust", renderer, "active", SPOTS_COL_LONADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. Step", renderer, "text", SPOTS_COL_LONSTEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. Min", renderer, "text", SPOTS_COL_LONMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Lon. Max", renderer, "text", SPOTS_COL_LONMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Radius", renderer, "text", SPOTS_COL_RAD, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. adjust", renderer, "active", SPOTS_COL_RADADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. Step", renderer, "text", SPOTS_COL_RADSTEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. Min", renderer, "text", SPOTS_COL_RADMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Rad. Max", renderer, "text", SPOTS_COL_RADMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temperature", renderer, "text", SPOTS_COL_TEMP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. adjust", renderer, "active", SPOTS_COL_TEMPADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. Step", renderer, "text", SPOTS_COL_TEMPSTEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. Min", renderer, "text", SPOTS_COL_TEMPMIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temp. Max", renderer, "text", SPOTS_COL_TEMPMAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    gtk_tree_view_column_set_visible(column, PHOEBE_SPOTS_SHOW_ALL);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Adjust", renderer, "active", SPOTS_COL_ADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_spots_treeview, column, -1);
    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_para_spots_adjust_checkbutton_toggled), NULL);
	gtk_tree_view_column_set_resizable (column, TRUE);

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

	/* phoebe_para_spots_units_combobox_init = FALSE; */
	status = phoebe_parameter_get_value(par, &spots_no);
	gtk_list_store_clear(store);

	for(i = 0; i < spots_no; i++)
		gtk_list_store_append(store, &iter);

	return status;
}


static void gui_crit_cell_data_function (GtkCellLayout *cell_layout, GtkCellRenderer *renderer, GtkTreeModel *model, GtkTreeIter *iter, gpointer data)
{
	gdouble 	val;
	gchar	   	buf[20];
	gchar	   *name;

	gdouble pot1, pot2;

	pot1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_phsv_spinbutton")->gtk));
	pot2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_pcsv_spinbutton")->gtk));
/*
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot1"), &pot1);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot2"), &pot2);
*/
	gtk_tree_model_get (model, iter, RS_COL_PARAM_NAME, &name, RS_COL_PARAM_VALUE, &val, -1);

	if (pot1 < val && !strcmp(name, "Ω(L<sub>1</sub>)")) {
		g_snprintf(buf, sizeof(buf), "<b>%f</b>", val);
		g_object_set(renderer, "foreground", "Red", "foreground-set", TRUE, "markup", buf, NULL);
	}
	else if (pot2 < val && !strcmp(name, "Ω(L<sub>2</sub>)")) {
		g_snprintf(buf, sizeof(buf), "<b>%f</b>", val);
		g_object_set(renderer, "foreground", "Red", "foreground-set", TRUE, "markup", buf, NULL);
	}
	else
		g_object_set(renderer, "foreground-set", FALSE, NULL);

	g_free (name);
}


int gui_init_sidesheet_res_treeview_old ()
{
	/*
	 * Creates an empty treeview with two columns, "Parameter" and "Value".
	 * It also connects 
	 */

	GtkWidget *phoebe_sidesheet_res_treeview = gui_widget_lookup ("phoebe_sidesheet_res_treeview")->gtk;

	GtkTreeModel *model = (GtkTreeModel *) gtk_list_store_new (
		RS_COL_COUNT,		/* Number of columns 	*/
		G_TYPE_STRING,		/* Parameter name		*/
		G_TYPE_DOUBLE);		/* Parameter value		*/

	GtkCellRenderer     *renderer;
	GtkTreeViewColumn   *column;

	renderer    = gtk_cell_renderer_text_new ();
	column      = gtk_tree_view_column_new_with_attributes ("Parameter", renderer, "markup", RS_COL_PARAM_NAME, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_sidesheet_res_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

	renderer    = gtk_cell_renderer_text_new ();
	column      = gtk_tree_view_column_new_with_attributes ("Value", renderer, "markup", RS_COL_PARAM_VALUE, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_sidesheet_res_treeview, column, -1);
	gtk_tree_view_column_set_cell_data_func (column, renderer, (GtkTreeCellDataFunc) gui_crit_cell_data_function, NULL, NULL);
	gtk_tree_view_column_set_resizable (column, TRUE);

	gtk_tree_view_set_model ((GtkTreeView *) phoebe_sidesheet_res_treeview, model);

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
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Value", renderer, "text", FS_COL_PARAM_VALUE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Step", renderer, "text", FS_COL_PARAM_STEP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Min", renderer, "text", FS_COL_PARAM_MIN, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Max", renderer, "text", FS_COL_PARAM_MAX, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_sidesheet_fit_treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    gtk_tree_view_set_model((GtkTreeView*)phoebe_sidesheet_fit_treeview, model);

    return SUCCESS;
}

int gui_fit_statistics_treeview_init ()
{
	int status = 0;
	
	GtkWidget *treeview = gui_widget_lookup ("phoebe_fitt_second_treeview")->gtk;
	
	GtkTreeModel *model = (GtkTreeModel *) gtk_list_store_new (
		CURVE_COL_COUNT,	/* Number of columns                */
		G_TYPE_STRING,		/* Curve name                       */
		G_TYPE_INT,			/* Number of points                 */
		G_TYPE_DOUBLE,      /* Unweighted residuals             */
		G_TYPE_DOUBLE,      /* Residuals with intrinsic weights */
		G_TYPE_DOUBLE,      /* Residuals with passband weights  */
		G_TYPE_DOUBLE		/* Residuals with all weights       */
	);
	
	GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;
	
    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Curve", renderer, "text", CURVE_COL_NAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);
	
    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Number of points", renderer, "text", CURVE_COL_NPOINTS, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);
	
    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Unweighted", renderer, "text", CURVE_COL_U_RES, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Intrinsic weights", renderer, "text", CURVE_COL_I_RES, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Intrinsic + passband weights", renderer, "text", CURVE_COL_P_RES, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes ("Fully weighted", renderer, "text", CURVE_COL_F_RES, NULL);
    gtk_tree_view_insert_column ((GtkTreeView *) treeview, column, -1);
	gtk_tree_view_column_set_resizable (column, TRUE);

    gtk_tree_view_set_model ((GtkTreeView *) treeview, model);
	
	return status;
}

static void gui_filter_cell_data_func (GtkCellLayout *cell_layout, GtkCellRenderer *renderer, GtkTreeModel *model, GtkTreeIter *iter, gpointer data)
{
	if(gtk_tree_model_iter_has_child(model, iter)) g_object_set(renderer, "sensitive", FALSE, NULL);
	else g_object_set(renderer, "sensitive", TRUE, NULL);
}

int gui_init_filter_combobox (GtkWidget *combo_box, gint activefilter)
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

	gtk_cell_layout_set_cell_data_func(GTK_CELL_LAYOUT(combo_box), renderer, gui_filter_cell_data_func, NULL, NULL);

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
		if (i == activefilter)
			gtk_combo_box_set_active_iter (GTK_COMBO_BOX (combo_box), &child);
	}

	g_object_unref (store);

	return SUCCESS;
}

int gui_fill_sidesheet_res_treeview ()
{
	GtkTreeView *sidesheet = (GtkTreeView *) gui_widget_lookup ("phoebe_sidesheet_res_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model (sidesheet);
	double val;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mass1"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_MASS_1, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mass2"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_MASS_2, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_radius1"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_RADIUS_1, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_radius2"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_RADIUS_2, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_LOGG_1, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_LOGG_2, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mbol1"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_MBOL_1, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_mbol2"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_MBOL_2, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_sbr1"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_SBR_1, val);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_sbr2"), &val);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_SBR_2, val);

	return SUCCESS;
}

int gui_update_cla_value (int row)
{
	GtkTreeView *master = (GtkTreeView *) gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model (master);
	GtkTreeIter iter;
	double cla;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_plum2"), &cla);
	return gui_set_treeview_value (model, LC_COL_CLA, row, cla);
}

int gui_set_treeview_value (GtkTreeModel *model, int col_id, int row_id, double value)
{
	/*
	 * Sets a numeric cell value of the passed treeview model.
	 */

	GtkTreeIter iter;
	char path[3];

	sprintf (path, "%d", row_id);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, col_id, value, -1);

	return SUCCESS;
}

int gui_init_sidesheet_res_treeview ()
{
	int status = 0, i;

	GtkTreeIter iter;
	GtkWidget *phoebe_sidesheet_res_treeview = gui_widget_lookup ("phoebe_sidesheet_res_treeview")->gtk;

	GtkTreeModel *model = (GtkTreeModel *) gtk_list_store_new (
		RS_COL_COUNT,		/* Number of columns 	  */
		G_TYPE_STRING,		/* Parameter name		  */
		G_TYPE_DOUBLE,		/* Parameter value		  */
		G_TYPE_DOUBLE);		/* Parameter formal error */

	GtkCellRenderer     *renderer;
	GtkTreeViewColumn   *column;
	char path[3];

	renderer    = gtk_cell_renderer_text_new ();
	column      = gtk_tree_view_column_new_with_attributes ("Parameter", renderer, "markup", RS_COL_PARAM_NAME, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_sidesheet_res_treeview, column, -1);

	renderer    = gtk_cell_renderer_text_new ();
	column      = gtk_tree_view_column_new_with_attributes ("Value", renderer, "markup", RS_COL_PARAM_VALUE, NULL);
	gtk_tree_view_insert_column ((GtkTreeView *) phoebe_sidesheet_res_treeview, column, -1);
/*
	gtk_tree_view_column_set_cell_data_func (column, renderer, (GtkTreeCellDataFunc) gui_crit_cell_data_function, NULL, NULL);
*/
	gtk_tree_view_set_model ((GtkTreeView *) phoebe_sidesheet_res_treeview, model);

	printf ("*** in sidesheet's init function.\n");

	for (i = 0; i < SIDESHEET_NUM_PARAMS; i++)
		gtk_list_store_append (GTK_LIST_STORE (model), &iter);

	/* Potential in L1: */
	sprintf (path, "%d", SIDESHEET_LAGRANGE_1);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "Ω(L<sub>1</sub>)", -1);

	/* Potential in L2: */
	sprintf (path, "%d", SIDESHEET_LAGRANGE_2);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "Ω(L<sub>2</sub>)", -1);

	/* Primary star mass: */
	sprintf (path, "%d", SIDESHEET_MASS_1);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "M<sub>1</sub>", -1);

	/* Secondary star mass: */
	sprintf (path, "%d", SIDESHEET_MASS_2);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "M<sub>2</sub>", -1);

	/* Primary star radius: */
	sprintf (path, "%d", SIDESHEET_RADIUS_1);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "R<sub>1</sub>", -1);

	/* Secondary star radius: */
	sprintf (path, "%d", SIDESHEET_RADIUS_2);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "R<sub>2</sub>", -1);

	/* Primary star bolometric magnitude: */
	sprintf (path, "%d", SIDESHEET_MBOL_1);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "M<sub>bol,1</sub>", -1);

	/* Secondary star bolometric magnitude: */
	sprintf (path, "%d", SIDESHEET_MBOL_2);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "M<sub>bol,2</sub>", -1);

	/* Primary star log(g): */
	sprintf (path, "%d", SIDESHEET_LOGG_1);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "log(g<sub>1</sub>)", -1);

	/* Secondary star log(g): */
	sprintf (path, "%d", SIDESHEET_LOGG_2);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "log(g<sub>2</sub>)", -1);

	/* Primary star polar surface brightness: */
	sprintf (path, "%d", SIDESHEET_SBR_1);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "Polar SBR<sub>1</sub>", -1);

	/* Secondary star polar surface brightness: */
	sprintf (path, "%d", SIDESHEET_SBR_2);
	gtk_tree_model_get_iter_from_string (model, &iter, path);
	gtk_list_store_set (GTK_LIST_STORE (model), &iter, RS_COL_PARAM_NAME, "Polar SBR<sub>2</sub>", -1);

	/* Also update constrained potentials */
	{
	char *phoebe_model;
	int wd_model;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_model"), &phoebe_model);
	wd_model = phoebe_wd_model (phoebe_model);

	if (phoebe_phsv_constrained (wd_model))
		gui_set_value_to_widget (gui_widget_lookup("phoebe_para_comp_phsv_spinbutton"));
	if (phoebe_pcsv_constrained (wd_model))
		gui_set_value_to_widget (gui_widget_lookup("phoebe_para_comp_pcsv_spinbutton"));
	}

	return status;
}

int gui_fill_treeview_with_spot_parameter(GtkTreeModel *model, bool sidesheet, char *spot_par_name)
{
	PHOEBE_parameter *par = phoebe_parameter_lookup(spot_par_name);
	PHOEBE_parameter *par_tba = NULL, *par_step = NULL, *par_min = NULL, *par_max = NULL;
	char par_name[255];
	char full_qualifier[255];
	int i, spno;
	double value, step, min, max;
	bool tba, active;
	GtkTreeIter iter;
	int status = 0;

	sprintf(par_name, "%s_tba", spot_par_name);
	par_tba = phoebe_parameter_lookup(par_name);
	sprintf(par_name, "%s_step", spot_par_name);
	par_step = phoebe_parameter_lookup(par_name);
	sprintf(par_name, "%s_min", spot_par_name);
	par_min = phoebe_parameter_lookup(par_name);
	sprintf(par_name, "%s_max", spot_par_name);
	par_max = phoebe_parameter_lookup(par_name);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no"), &spno);

	for (i = 0; i < spno; i++){
		status = phoebe_parameter_get_value(par_tba, i, &tba);
		if (!tba) continue;
		status = phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_active_switch"), i, &active);
		if (!active) continue;
		status = phoebe_parameter_get_value(par, i, &value);
		sprintf(full_qualifier, "%s[%d]", spot_par_name, i+1);
		gtk_list_store_append((GtkListStore*)model, &iter);
		if (sidesheet) {
			status = phoebe_parameter_get_value(par_step, i, &step);
			status = phoebe_parameter_get_value(par_min, i, &min);
			status = phoebe_parameter_get_value(par_max, i, &max);
			gtk_list_store_set(GTK_LIST_STORE(model), &iter,
									FS_COL_PARAM_NAME, full_qualifier,
									FS_COL_PARAM_VALUE, value,
									FS_COL_PARAM_STEP, step,
									FS_COL_PARAM_MIN, min,
									FS_COL_PARAM_MAX, max, -1);
		}
		else
			gtk_list_store_set(GTK_LIST_STORE(model), &iter,
									MF_COL_QUALIFIER, full_qualifier,
									MF_COL_INITVAL, value, -1);
	}

	return status;
}

int gui_fill_treeview_with_spot_parameters(GtkTreeModel *model, bool sidesheet)
{
	int status = 0;
	status = gui_fill_treeview_with_spot_parameter(model, sidesheet, "phoebe_spots_colatitude");
	status = gui_fill_treeview_with_spot_parameter(model, sidesheet, "phoebe_spots_longitude");
	status = gui_fill_treeview_with_spot_parameter(model, sidesheet, "phoebe_spots_radius");
	status = gui_fill_treeview_with_spot_parameter(model, sidesheet, "phoebe_spots_tempfactor");

	return status;
}

int gui_fill_sidesheet_fit_treeview ()
{
	int status = 0;

	GtkTreeView *phoebe_sidesheet_fit_treeview = (GtkTreeView *) gui_widget_lookup ("phoebe_sidesheet_fit_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model (phoebe_sidesheet_fit_treeview);
	GtkTreeIter iter;

	PHOEBE_parameter_list *pars_tba = phoebe_parameter_list_get_marked_tba ();
	PHOEBE_parameter *par;
	double value, step, min, max;

	gtk_list_store_clear ((GtkListStore *) model);

	while (pars_tba) {
		par = pars_tba->par;

		switch (par->type) {
			case TYPE_DOUBLE:
				status = phoebe_parameter_get_value(par, &value);
				status = phoebe_parameter_get_step(par, &step);
				status = phoebe_parameter_get_min(par, &min);
				status = phoebe_parameter_get_max(par, &max);

				gtk_list_store_append(GTK_LIST_STORE(model), &iter);

				/* We need a little hack here if angles are in degrees: */
				if (strcmp (par->qualifier, "phoebe_perr0") == 0 ||
					strcmp (par->qualifier, "phoebe_dperdt") == 0) {
					double cv;
					char *units;
					phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
					if (strcmp (units, "Radians") == 0)
						cv = 1.0;
					else
						cv = M_PI/180.0;

					gtk_list_store_set (
					    (GtkListStore *) model, &iter,
						FS_COL_PARAM_NAME, par->qualifier,
						FS_COL_PARAM_VALUE, value/cv,
						FS_COL_PARAM_STEP, step/cv,
						FS_COL_PARAM_MIN, min/cv,
						FS_COL_PARAM_MAX, max/cv, -1);
				}
				else
					gtk_list_store_set((GtkListStore *) model, &iter,
						FS_COL_PARAM_NAME, par->qualifier,
						FS_COL_PARAM_VALUE, value,
						FS_COL_PARAM_STEP, step,
						FS_COL_PARAM_MIN, min,
						FS_COL_PARAM_MAX, max, -1);
			break;
			case TYPE_DOUBLE_ARRAY: {
				int i, n = par->value.vec->dim;
				char full_qualifier[255];
				for (i = 0; i < n; i++){
					status = phoebe_parameter_get_value(par, i, &value);
					status = phoebe_parameter_get_step(par, &step);
					status = phoebe_parameter_get_min(par, &min);
					status = phoebe_parameter_get_max(par, &max);

					sprintf(full_qualifier, "%s[%d]", par->qualifier, i+1);
					gtk_list_store_append((GtkListStore*)model, &iter);
					gtk_list_store_set((GtkListStore*)model, &iter,
														FS_COL_PARAM_NAME, full_qualifier,
														FS_COL_PARAM_VALUE, value,
														FS_COL_PARAM_STEP, step,
														FS_COL_PARAM_MIN, min,
														FS_COL_PARAM_MAX, max, -1);
			}
			break;
			default:
				status = -1;
			}
		}
		pars_tba = pars_tba->next;
	}

	status = gui_fill_treeview_with_spot_parameters(model, TRUE);

	return status;
}

int gui_data_lc_treeview_add ()
{
	int status = 0;
	int optindex, optcount;

	PHOEBE_parameter *indep     = phoebe_parameter_lookup ("phoebe_lc_indep");
	PHOEBE_parameter *dep       = phoebe_parameter_lookup ("phoebe_lc_dep");
	PHOEBE_parameter *indweight = phoebe_parameter_lookup ("phoebe_lc_indweight");

	gchar     *glade_xml_file    = g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe_load_lc.glade", NULL);
	gchar     *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_load_lc_xml = glade_xml_new (glade_xml_file, NULL, NULL);

    GtkWidget *phoebe_load_lc_dialog                = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_dialog");
	GtkWidget *phoebe_load_lc_filechooserbutton     = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
    GtkWidget *phoebe_load_lc_column1_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column1_combobox");
    GtkWidget *phoebe_load_lc_column2_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column2_combobox");
    GtkWidget *phoebe_load_lc_column3_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column3_combobox");
    GtkWidget *phoebe_load_lc_sigma_spinbutton      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_sigma_spinbutton");
    GtkWidget *phoebe_load_lc_preview_textview      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_preview_textview");
    GtkWidget *phoebe_load_lc_filter_combobox       = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filter_combobox");
    GtkWidget *phoebe_load_lc_id_entry				= glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_id_entry");

	gui_status ("Adding a light curve.");
	g_object_unref (phoebe_load_lc_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_load_lc_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_load_lc_dialog), "Add Observed Light Curve Data");
	
	gui_init_filter_combobox (phoebe_load_lc_filter_combobox, -1);
	
	g_signal_connect (G_OBJECT (phoebe_load_lc_filechooserbutton),
					  "selection_changed",
					  G_CALLBACK (on_phoebe_load_lc_filechooserbutton_selection_changed),
					  (gpointer) phoebe_load_lc_preview_textview);
	
	/* Populate the combo boxes: */
	optcount = indep->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_lc_column1_combobox), strdup(indep->menu->option[optindex]));
	
	optcount = dep->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_lc_column2_combobox), strdup(dep->menu->option[optindex]));
	
	optcount = indweight->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_lc_column3_combobox), strdup(indweight->menu->option[optindex]));
	
	/* Default values for column combo boxes: */
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column1_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column2_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_lc_column3_combobox,  0);
	
	gchar *dir;
	phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);
	
	if(PHOEBE_DIRFLAG)
		dir = PHOEBE_DIRNAME;
	gtk_file_chooser_set_current_folder((GtkFileChooser*)phoebe_load_lc_filechooserbutton, dir);
	
    gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_lc_dialog);
    switch (result){
        case GTK_RESPONSE_OK:{
            GtkTreeModel *model;
            GtkTreeIter iter;
			gchar *id;
			
			GtkTreeIter filter_iter;
			gint 		filter_number;
			gchar 		filter_selected[255] = "Johnson:V";

			gchar* filename = gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_lc_filechooserbutton);
			if(!phoebe_filename_exists(filename))gui_notice("Invalid filename", "You haven't supplied a filename for your data.");
			else{
				if(!PHOEBE_DIRFLAG) PHOEBE_DIRFLAG = TRUE;
				PHOEBE_DIRNAME = gtk_file_chooser_get_current_folder ((GtkFileChooser*)phoebe_load_lc_filechooserbutton);
			}
			
			GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);
			
			if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_lc_filter_combobox), &filter_iter)) {
				gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_lc_filter_combobox)), &filter_iter, 1, &filter_number, -1);
				sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
			}
			
			id = gtk_entry_get_text (GTK_ENTRY (phoebe_load_lc_id_entry));
			if (strlen (id) < 1) id = filter_selected;

            gtk_list_store_append ((GtkListStore*) model, &iter);
            gtk_list_store_set ((GtkListStore*) model, &iter,
								LC_COL_ACTIVE,      TRUE,
								LC_COL_FILENAME,    filename,
								LC_COL_ID,			id,
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
                                LC_COL_EL3_LUM,     0.0,
                                LC_COL_EXTINCTION,  0.0,
                                LC_COL_X1,          0.5,
                                LC_COL_X2,          0.5,
                                LC_COL_Y1,          0.5,
                                LC_COL_Y2,          0.5,
								LC_COL_PLOT_OBS,    FALSE,
								LC_COL_PLOT_SYN,    FALSE,
								LC_COL_PLOT_OBS_COLOR, "#0000FF",
								LC_COL_PLOT_SYN_COLOR, "#FF0000",
								LC_COL_PLOT_OFFSET, 0.0,
								-1);

			/* Update the number of light curves parameter: */
            PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
            int lcno;

            phoebe_parameter_get_value(par, &lcno);
            phoebe_parameter_set_value(par, lcno + 1);
            gui_ld_coeffs_need_updating();
            gui_fill_sidesheet_fit_treeview ();
            gui_fill_fitt_mf_treeview();

            phoebe_debug("Number of light curves: %d\n", lcno + 1);

            /* Select the new row in the list: */
			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_lc_treeview), &iter);

			gui_status ("Light curve added.");
        }
        break;

        case GTK_RESPONSE_CANCEL:
            gui_status ("Light curve adding cancelled.");
        break;
    }

    gtk_widget_destroy (phoebe_load_lc_dialog);

    return status;
}

int gui_data_lc_treeview_edit ()
{
	int status = 0;

	PHOEBE_parameter *indep     = phoebe_parameter_lookup ("phoebe_lc_indep");
	PHOEBE_parameter *dep       = phoebe_parameter_lookup ("phoebe_lc_dep");
	PHOEBE_parameter *indweight = phoebe_parameter_lookup ("phoebe_lc_indweight");
	PHOEBE_parameter *lcfilter	= phoebe_parameter_lookup ("phoebe_lc_filter");
	int optindex, optcount;

	GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
    model = gtk_tree_view_get_model ((GtkTreeView *) phoebe_data_lc_treeview);

    if (gtk_tree_model_get_iter_first (model, &iter)) {
        gchar     *glade_xml_file                       = g_build_filename    (PHOEBE_GLADE_XML_DIR, "phoebe_load_lc.glade", NULL);
		gchar     *glade_pixmap_file                    = g_build_filename    (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

        GladeXML  *phoebe_load_lc_xml                   = glade_xml_new       (glade_xml_file, NULL, NULL);

        GtkWidget *phoebe_load_lc_dialog                = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_dialog");
        GtkWidget *phoebe_load_lc_filechooserbutton     = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
        GtkWidget *phoebe_load_lc_column1_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column1_combobox");
        GtkWidget *phoebe_load_lc_column2_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column2_combobox");
        GtkWidget *phoebe_load_lc_column3_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column3_combobox");
        GtkWidget *phoebe_load_lc_sigma_spinbutton      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_sigma_spinbutton");
        GtkWidget *phoebe_load_lc_preview_textview      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_preview_textview");
        GtkWidget *phoebe_load_lc_filter_combobox       = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filter_combobox");
        GtkWidget *phoebe_load_lc_id_entry				= glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_id_entry");

        gchar *id;
        gint itype, dtype, wtype;
        gchar *itype_str, *dtype_str, *wtype_str;
		gchar *filter;
        gdouble sigma;

        gchar* filename;

        gchar filter_selected[255] = "Johnson:V";
		gint filter_number;
		GtkTreeIter filter_iter;

        g_object_unref (phoebe_load_lc_xml);

		gtk_window_set_icon  (GTK_WINDOW (phoebe_load_lc_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
		gtk_window_set_title (GTK_WINDOW (phoebe_load_lc_dialog), "PHOEBE - Edit LC Data");

		g_signal_connect (G_OBJECT (phoebe_load_lc_filechooserbutton),
						  "selection_changed",
						  G_CALLBACK (on_phoebe_load_lc_filechooserbutton_selection_changed),
						  (gpointer) phoebe_load_lc_preview_textview);

        GtkTreeSelection *selection;
        selection = gtk_tree_view_get_selection ((GtkTreeView *) phoebe_data_lc_treeview);
        if (gtk_tree_selection_get_selected (selection, &model, &iter)) {
            gtk_tree_model_get(model, &iter,    LC_COL_FILENAME, 	&filename,
												LC_COL_ID,          &id,
                                                LC_COL_FILTER,   	&filter,
                                                LC_COL_ITYPE_STR,	&itype_str,
                                                LC_COL_DTYPE_STR, 	&dtype_str,
                                                LC_COL_WTYPE_STR, 	&wtype_str,
                                                LC_COL_SIGMA,    	&sigma, -1);

			phoebe_parameter_option_get_index (indep, itype_str, &itype);
			phoebe_parameter_option_get_index (dep, dtype_str, &dtype);
			phoebe_parameter_option_get_index (indweight, wtype_str, &wtype);
			phoebe_parameter_option_get_index (lcfilter, filter, &filter_number);

			/* Populate the combo boxes: */
			optcount = indep->menu->optno;
			for(optindex = 0; optindex < optcount; optindex++)
				gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_lc_column1_combobox), strdup(indep->menu->option[optindex]));

			optcount = dep->menu->optno;
			for(optindex = 0; optindex < optcount; optindex++)
				gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_lc_column2_combobox), strdup(dep->menu->option[optindex]));

			optcount = indweight->menu->optno;
			for(optindex = 0; optindex < optcount; optindex++)
				gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_lc_column3_combobox), strdup(indweight->menu->option[optindex]));

			gtk_combo_box_set_active ((GtkComboBox*)   phoebe_load_lc_column1_combobox,  itype);
			gtk_combo_box_set_active ((GtkComboBox*)   phoebe_load_lc_column2_combobox,  dtype);
			gtk_combo_box_set_active ((GtkComboBox*)   phoebe_load_lc_column3_combobox,  wtype);
		        gui_init_filter_combobox (phoebe_load_lc_filter_combobox, filter_number);

			gtk_spin_button_set_value ((GtkSpinButton*) phoebe_load_lc_sigma_spinbutton, sigma);
			gtk_entry_set_text ((GtkEntry*)	phoebe_load_lc_id_entry, id);

			sprintf(filter_selected, "%s", filter);

			if (filename)
				gtk_file_chooser_set_filename ((GtkFileChooser *) phoebe_load_lc_filechooserbutton, filename);
			else {
				gchar *dir;
				phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

				gtk_file_chooser_set_current_folder ((GtkFileChooser *) phoebe_load_lc_filechooserbutton, dir);
			}
        }

        gint result = gtk_dialog_run ((GtkDialog *) phoebe_load_lc_dialog);
        switch (result) {
            case GTK_RESPONSE_OK: {
				gchar *new_id;

            	filename = gtk_file_chooser_get_filename ((GtkFileChooser *) phoebe_load_lc_filechooserbutton);
				if (!phoebe_filename_exists (filename))
					gui_notice ("Invalid filename", "You haven't supplied a filename for your data.");
				else {
					if (!PHOEBE_DIRFLAG) PHOEBE_DIRFLAG = TRUE;
					PHOEBE_DIRNAME = gtk_file_chooser_get_current_folder ((GtkFileChooser*)phoebe_load_lc_filechooserbutton);
				}

				if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_lc_filter_combobox), &filter_iter)) {
					gint new_filter_number;
					gtk_tree_model_get (gtk_combo_box_get_model (GTK_COMBO_BOX (phoebe_load_lc_filter_combobox)), &filter_iter, 1, &new_filter_number, -1);
					sprintf (filter_selected, "%s:%s", PHOEBE_passbands[new_filter_number]->set, PHOEBE_passbands[new_filter_number]->name);
					if (new_filter_number != filter_number)
						gui_ld_coeffs_need_updating ();
				}

				new_id = (gchar *) strdup (gtk_entry_get_text ((GtkEntry *) phoebe_load_lc_id_entry));
				if(strlen (new_id) < 1) new_id = id;

                gtk_list_store_set ((GtkListStore *) model, &iter, LC_COL_ACTIVE,      TRUE,
                                                                   LC_COL_FILENAME,    filename,
                                                                   LC_COL_ID,          new_id,
                                                                   LC_COL_FILTER,      filter_selected,
																   LC_COL_FILTERNO,    filter_number,
                                                                   LC_COL_ITYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column1_combobox),
                                                                   LC_COL_ITYPE_STR,   strdup(indep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column1_combobox)]),
                                                                   LC_COL_DTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column2_combobox),
                                                                   LC_COL_DTYPE_STR,   strdup(dep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column2_combobox)]),
                                                                   LC_COL_WTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column3_combobox),
                                                                   LC_COL_WTYPE_STR,   strdup(indweight->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_lc_column3_combobox)]),
                                                                   LC_COL_SIGMA,       gtk_spin_button_get_value((GtkSpinButton*)phoebe_load_lc_sigma_spinbutton), -1);
            }
            break;
            case GTK_RESPONSE_CANCEL:
				/* Fall through. */
            break;
        }
        gtk_widget_destroy (phoebe_load_lc_dialog);
    }

    return status;
}

int gui_data_lc_treeview_remove ()
{
	int lcno, status = 0;
	PHOEBE_parameter *par;

	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
    selection = gtk_tree_view_get_selection ((GtkTreeView *) phoebe_data_lc_treeview);
    if (gtk_tree_selection_get_selected (selection, &model, &iter)) {
        gtk_list_store_remove ((GtkListStore*) model, &iter);

        par = phoebe_parameter_lookup ("phoebe_lcno");
        phoebe_parameter_get_value (par, &lcno);
        phoebe_parameter_set_value (par, lcno-1);
        gui_fill_sidesheet_fit_treeview ();
        gui_fill_fitt_mf_treeview ();

        phoebe_debug("Number of light curves: %d\n", lcno - 1);
        gui_status("A light curve removed.");
    }

    return status;
}

int gui_data_rv_treeview_add()
{
    gui_status("Adding a radial velocity curve...");
	int status = 0;

	PHOEBE_parameter *indep     = phoebe_parameter_lookup("phoebe_rv_indep");
	PHOEBE_parameter *dep       = phoebe_parameter_lookup("phoebe_rv_dep");
	PHOEBE_parameter *indweight = phoebe_parameter_lookup("phoebe_rv_indweight");
	int optindex, optcount;

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
    GtkWidget *phoebe_load_rv_id_entry				= glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_id_entry");

    GtkWidget *phoebe_load_rv_filter_combobox       = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filter_combobox");

	gtk_window_set_icon (GTK_WINDOW (phoebe_load_rv_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_load_rv_dialog), "PHOEBE - Add RV Data");

	gui_init_filter_combobox(phoebe_load_rv_filter_combobox, -1);

	g_signal_connect (G_OBJECT (phoebe_load_rv_filechooserbutton),
					  "selection_changed",
					  G_CALLBACK (on_phoebe_load_rv_filechooserbutton_selection_changed),
					  (gpointer) phoebe_load_rv_preview_textview);

    g_object_unref(phoebe_load_rv_xml);

    GtkTreeModel *model;
    GtkTreeIter iter;

    /* Populate the combo boxes: */
	optcount = indep->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_rv_column1_combobox), strdup(indep->menu->option[optindex]));

	optcount = dep->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_rv_column2_combobox), strdup(dep->menu->option[optindex]));

	optcount = indweight->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_rv_column3_combobox), strdup(indweight->menu->option[optindex]));

	/* Default values for column combo boxes: */
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column1_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column2_combobox,  0);
	gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column3_combobox,  0);

	gchar *dir;
	phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

	if(PHOEBE_DIRFLAG)
		dir = PHOEBE_DIRNAME;
	gtk_file_chooser_set_current_folder((GtkFileChooser*)phoebe_load_rv_filechooserbutton, dir);

    gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_rv_dialog);
    switch (result){
        case GTK_RESPONSE_OK:{

			GtkTreeIter filter_iter;
			gint 		filter_number;
			gchar 		filter_selected[255] = "Johnson:V";

			gchar* filename = gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_rv_filechooserbutton);
			if(!phoebe_filename_exists(filename))gui_notice("Invalid filename", "You haven't supplied a filename for your data.");
			else{
				if(!PHOEBE_DIRFLAG) PHOEBE_DIRFLAG = TRUE;
				PHOEBE_DIRNAME = gtk_file_chooser_get_current_folder ((GtkFileChooser*)phoebe_load_rv_filechooserbutton);
			}

			GtkWidget *phoebe_data_rv_treeview = gui_widget_lookup("phoebe_data_rv_treeview")->gtk;
            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

			if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_rv_filter_combobox), &filter_iter)) {
				gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_rv_filter_combobox)), &filter_iter, 1, &filter_number, -1);
				sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
			}

			gchar *id;
			id = (gchar *)gtk_entry_get_text((GtkEntry*)phoebe_load_rv_id_entry);
			if(strlen(id) < 1)id = filter_selected;

            gtk_list_store_append((GtkListStore*)model, &iter);
            gtk_list_store_set((GtkListStore*)model, &iter,
				RV_COL_ACTIVE,      TRUE,
                RV_COL_FILENAME,    filename,
                RV_COL_ID,			id,
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
                RV_COL_Y2,          0.5,
				RV_COL_PLOT_OBS,    FALSE,
				RV_COL_PLOT_SYN,    FALSE,
				RV_COL_PLOT_OBS_COLOR, "#0000FF",
				RV_COL_PLOT_SYN_COLOR, "#FF0000",
				RV_COL_PLOT_OFFSET, 0.0,
				-1);

            PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
            int rvno;

            phoebe_parameter_get_value(par, &rvno);
            phoebe_parameter_set_value(par, rvno + 1);

            phoebe_debug("Number of RV curves: %d\n", rvno + 1);

			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview), &iter);
			gui_status("A radial velocity curve added.");
        }
        break;

        case GTK_RESPONSE_CANCEL:
            gui_status("Adding radial velocity curve cancelled.");
        break;
    }
    gtk_widget_destroy (phoebe_load_rv_dialog);

    return status;
}

int gui_data_rv_treeview_edit()
{
	int status = 0;

	PHOEBE_parameter *indep     = phoebe_parameter_lookup("phoebe_rv_indep");
	PHOEBE_parameter *dep       = phoebe_parameter_lookup("phoebe_rv_dep");
	PHOEBE_parameter *indweight = phoebe_parameter_lookup("phoebe_rv_indweight");
	PHOEBE_parameter *rvfilter  = phoebe_parameter_lookup("phoebe_rv_filter");
	int optindex, optcount;

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
        GtkWidget *phoebe_load_rv_id_entry		= glade_xml_get_widget (phoebe_load_rv_xml, "phoebe_load_rv_id_entry");

	gchar *filename, *id;
        int itype, dtype, wtype;
        gchar *itype_str, *dtype_str, *wtype_str;
        gdouble sigma;
        gchar *filter;

        gchar filter_selected[255] = "Johnson:V";
	gint filter_number;
	GtkTreeIter filter_iter;

	gtk_window_set_icon (GTK_WINDOW (phoebe_load_rv_dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_load_rv_dialog), "PHOEBE - Edit RV Data");

	g_signal_connect (G_OBJECT (phoebe_load_rv_filechooserbutton),
					  "selection_changed",
					  G_CALLBACK (on_phoebe_load_rv_filechooserbutton_selection_changed),
				 	  (gpointer) phoebe_load_rv_preview_textview);

        g_object_unref(phoebe_load_rv_xml);

        GtkTreeSelection *selection;
        selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
            gtk_tree_model_get(model, &iter,    RV_COL_FILENAME, 	&filename,
												RV_COL_ID, 			&id,
                                                RV_COL_FILTER,   	&filter,
                                                RV_COL_ITYPE_STR,   &itype_str,
                                                RV_COL_DTYPE_STR,   &dtype_str,
                                                RV_COL_WTYPE_STR,   &wtype_str,
                                                RV_COL_SIGMA,    	&sigma, -1);

			phoebe_parameter_option_get_index(indep, itype_str, &itype);
			phoebe_parameter_option_get_index(dep, dtype_str, &dtype);
			phoebe_parameter_option_get_index(indweight, wtype_str, &wtype);
			phoebe_parameter_option_get_index(rvfilter, filter, &filter_number);

			/* Populate the combo boxes: */
			optcount = indep->menu->optno;
			for(optindex = 0; optindex < optcount; optindex++)
				gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_rv_column1_combobox), strdup(indep->menu->option[optindex]));

			optcount = dep->menu->optno;
			for(optindex = 0; optindex < optcount; optindex++)
				gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_rv_column2_combobox), strdup(dep->menu->option[optindex]));

			optcount = indweight->menu->optno;
			for(optindex = 0; optindex < optcount; optindex++)
				gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_load_rv_column3_combobox), strdup(indweight->menu->option[optindex]));

            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column1_combobox,  itype);
            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column2_combobox,  dtype);
            gtk_combo_box_set_active     ((GtkComboBox*)   phoebe_load_rv_column3_combobox,  wtype);
            gui_init_filter_combobox(phoebe_load_rv_filter_combobox, filter_number);

            gtk_spin_button_set_value    ((GtkSpinButton*) phoebe_load_rv_sigma_spinbutton,  sigma);

            gtk_entry_set_text			 ((GtkEntry*)	   phoebe_load_rv_id_entry,			 id);

			sprintf(filter_selected, "%s", filter);

			if(filename)
	            gtk_file_chooser_set_filename((GtkFileChooser*)phoebe_load_rv_filechooserbutton, filename);
			else{
				gchar *dir;
				phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

				gtk_file_chooser_set_current_folder((GtkFileChooser*)phoebe_load_rv_filechooserbutton, dir);
			}
        }

        gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_rv_dialog);
        switch (result){
            case GTK_RESPONSE_OK:{

            	filename = gtk_file_chooser_get_filename ((GtkFileChooser*)phoebe_load_rv_filechooserbutton);
				if(!phoebe_filename_exists(filename))gui_notice("Invalid filename", "You haven't supplied a filename for your data.");

				else{
					if(!PHOEBE_DIRFLAG) PHOEBE_DIRFLAG = TRUE;
					PHOEBE_DIRNAME = gtk_file_chooser_get_current_folder ((GtkFileChooser*)phoebe_load_rv_filechooserbutton);
				}

				if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_rv_filter_combobox), &filter_iter)) {
					gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_rv_filter_combobox)), &filter_iter, 1, &filter_number, -1);
					sprintf (filter_selected, "%s:%s", PHOEBE_passbands[filter_number]->set, PHOEBE_passbands[filter_number]->name);
				}

				gchar *new_id;
				new_id = (gchar *)gtk_entry_get_text((GtkEntry*)phoebe_load_rv_id_entry);
				if(strlen(new_id) < 1)new_id = id;

                gtk_list_store_set((GtkListStore*)model, &iter, RV_COL_ACTIVE,      TRUE,
                                                                RV_COL_FILENAME,    filename,
                                                                RV_COL_ID,			new_id,
                                                                RV_COL_FILTER,      filter_selected,
                                                                RV_COL_ITYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox),
                                                                RV_COL_ITYPE_STR,   strdup(indep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox)]),
                                                                RV_COL_DTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox),
                                                                RV_COL_DTYPE_STR,   strdup(dep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox)]),
                                                                RV_COL_WTYPE,       gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column3_combobox),
                                                                RV_COL_WTYPE_STR,   strdup(indweight->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column3_combobox)]),
                                                                RV_COL_SIGMA,       gtk_spin_button_get_value((GtkSpinButton*)phoebe_load_rv_sigma_spinbutton), -1);
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

        phoebe_debug("Number of RV curves: %d\n", rvno - 1);
        gui_status("A radial velocity curve removed.");
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


int gui_fill_fitt_mf_treeview ()
{
	GtkTreeView *phoebe_fitt_mf_treeview = GTK_TREE_VIEW (gui_widget_lookup ("phoebe_fitt_first_treeview")->gtk);
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_fitt_mf_treeview);
	GtkTreeIter iter;
	double value;

	int status = 0;

	gtk_list_store_clear(GTK_LIST_STORE(model));

	PHOEBE_parameter_list *pars_tba = phoebe_parameter_list_get_marked_tba();
	PHOEBE_parameter *par;

	while(pars_tba){
		par = pars_tba->par;

		switch (par->type){
			case TYPE_DOUBLE: {
				status = phoebe_parameter_get_value(par, &value);

				gtk_list_store_append(GTK_LIST_STORE(model), &iter);

				/* We need a little hack here if angles are in degrees: */
				if (strcmp (par->qualifier, "phoebe_perr0") == 0 ||
					strcmp (par->qualifier, "phoebe_dperdt") == 0) {
					double cv;
					char *units;
					phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
					if (strcmp (units, "Radians") == 0)
						cv = 1.0;
					else
						cv = M_PI/180.0;

					gtk_list_store_set (GTK_LIST_STORE (model), &iter,
						MF_COL_QUALIFIER, par->qualifier,
						MF_COL_INITVAL,   value/cv, -1);
				}
				else
					gtk_list_store_set(GTK_LIST_STORE(model), &iter,
						MF_COL_QUALIFIER, par->qualifier,
						MF_COL_INITVAL, value, -1);
			}
			break;
			case TYPE_DOUBLE_ARRAY: {
				int i, n = par->value.vec->dim;
				char full_qualifier[255];
				for (i = 0; i < n; i++){
					status = phoebe_parameter_get_value(par, i, &value);

					sprintf(full_qualifier, "%s[%d]", par->qualifier, i+1);
					gtk_list_store_append(GTK_LIST_STORE(model), &iter);
					gtk_list_store_set(GTK_LIST_STORE(model), &iter,
						MF_COL_QUALIFIER, full_qualifier,
						MF_COL_INITVAL, value, -1);
					}
			}
			break;
			default:
				status = -1;
		}
		pars_tba = pars_tba->next;
	}

	status = gui_fill_treeview_with_spot_parameters(model, FALSE);

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
    		GtkWidget *phoebe_cla_adjust_checkbutton = gui_widget_lookup("phoebe_para_lum_levels_secadjust_checkbutton")->gtk;

			g_object_unref (phoebe_levels_xml);

			gtk_window_set_icon (GTK_WINDOW (phoebe_levels_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
			gtk_window_set_title (GTK_WINDOW(phoebe_levels_dialog), "PHOEBE - Edit Levels");

			gtk_label_set_text (GTK_LABEL (phoebe_levels_passband_label), passband);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_levels_primary_spinbutton), hla);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_levels_secondary_spinbutton), cla);

			if (!GTK_WIDGET_IS_SENSITIVE(phoebe_cla_adjust_checkbutton))
				gtk_widget_set_sensitive(phoebe_levels_secondary_spinbutton, FALSE);

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

int gui_para_lum_levels_calc(GtkTreeModel *model, GtkTreeIter iter)
{
	int status = 0;
	int index;
	PHOEBE_el3_units l3units;
	PHOEBE_curve *syncurve;
	PHOEBE_curve *obs;
	gdouble hla, cla, l3;
	gdouble alpha, lw;
	char *lw_str;

	index = atoi (gtk_tree_model_get_string_from_iter (model, &iter));
	
	obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, index);
	if (!obs)
		return ERROR_CURVE_NOT_INITIALIZED;
	
	phoebe_curve_transform (obs, obs->itype, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_SIGMA);
			
	/* Synthesize a theoretical curve: */
	syncurve = phoebe_curve_new ();
	phoebe_curve_compute (syncurve, obs->indep, index, obs->itype, PHOEBE_COLUMN_FLUX);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_levweight"), index, &lw_str);
	lw = -1;
	if (strcmp (lw_str, "None") == 0)               lw = 0;
	if (strcmp (lw_str, "Poissonian scatter") == 0) lw = 1;
	if (strcmp (lw_str, "Low light scatter") == 0)  lw = 2;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"), index, &l3);
	phoebe_el3_units_id (&l3units);

	status = phoebe_calculate_plum_correction (&alpha, syncurve, obs, lw, l3, l3units);
	phoebe_curve_free (obs);
	phoebe_curve_free (syncurve);
	if (status != SUCCESS)
		return status;
		
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hla"), index, &hla);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_plum2"), &cla);
	hla /= alpha;
	cla /= alpha;
		
	gtk_list_store_set ((GtkListStore *) model, &iter, LC_COL_HLA, hla, LC_COL_CLA, cla, -1);
	
	return status;
}

int gui_para_lum_levels_calc_selected()
{
	int status = 0, lcno;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);

	if (lcno > 0) {
		GtkTreeModel     *model;
		GtkTreeIter       iter;
		GtkTreeSelection *selection;

		GtkWidget *treeview = gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;
		model = gtk_tree_view_get_model ((GtkTreeView *) treeview);

		treeview = gui_widget_lookup ("phoebe_para_lc_levels_treeview")->gtk;
		selection = gtk_tree_view_get_selection ((GtkTreeView *) treeview);

		if (gtk_tree_selection_get_selected (selection, &model, &iter)) {
			gui_get_values_from_widgets ();
			gui_para_lum_levels_calc (model, iter);
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


int gui_para_lc_coefficents_edit ()
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
		gdouble x1,x2,y1,y2;

		GtkWidget *treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
		model = gtk_tree_view_get_model((GtkTreeView*)treeview);

		treeview = gui_widget_lookup("phoebe_para_lc_ld_treeview")->gtk;

        selection = gtk_tree_view_get_selection((GtkTreeView*)treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
			gtk_tree_model_get(model, &iter,    LC_COL_FILTER,	&passband,
												LC_COL_X1,		&x1,
												LC_COL_X2,		&x2,
												LC_COL_Y1,		&y1,
												LC_COL_Y2,		&y2, -1);

    		gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_lc_coefficients.glade", NULL);
			gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

			GladeXML  *phoebe_lc_coefficents_xml	        = glade_xml_new        (glade_xml_file, NULL, NULL);

   			GtkWidget *phoebe_lc_coefficents_dialog         = glade_xml_get_widget (phoebe_lc_coefficents_xml, "phoebe_lc_coefficents_dialog");
			GtkWidget *phoebe_lc_coefficents_passband_label	= glade_xml_get_widget (phoebe_lc_coefficents_xml, "phoebe_lc_coefficents_passband_label");
			GtkWidget *phoebe_lc_coefficents_x1_spinbutton	= glade_xml_get_widget (phoebe_lc_coefficents_xml, "phoebe_lc_coefficents_x1_spinbutton");
			GtkWidget *phoebe_lc_coefficents_x2_spinbutton	= glade_xml_get_widget (phoebe_lc_coefficents_xml, "phoebe_lc_coefficents_x2_spinbutton");
			GtkWidget *phoebe_lc_coefficents_y1_spinbutton	= glade_xml_get_widget (phoebe_lc_coefficents_xml, "phoebe_lc_coefficents_y1_spinbutton");
			GtkWidget *phoebe_lc_coefficents_y2_spinbutton	= glade_xml_get_widget (phoebe_lc_coefficents_xml, "phoebe_lc_coefficents_y2_spinbutton");
			GtkWidget *phoebe_ld_secondary_adjust_checkbutton = gui_widget_lookup("phoebe_para_ld_lccoefs_secadjust_checkbutton")->gtk;

			g_object_unref (phoebe_lc_coefficents_xml);

			gtk_window_set_icon (GTK_WINDOW (phoebe_lc_coefficents_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
			gtk_window_set_title (GTK_WINDOW(phoebe_lc_coefficents_dialog), "PHOEBE - LC Coefficents");

			gtk_label_set_text (GTK_LABEL (phoebe_lc_coefficents_passband_label), passband);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_x1_spinbutton), x1);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_x2_spinbutton), x2);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_y1_spinbutton), y1);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_y2_spinbutton), y2);

			if (!GTK_WIDGET_IS_SENSITIVE(phoebe_ld_secondary_adjust_checkbutton)) {
				gtk_widget_set_sensitive(phoebe_lc_coefficents_x2_spinbutton, FALSE);
				gtk_widget_set_sensitive(phoebe_lc_coefficents_y2_spinbutton, FALSE);
			}

    		gint result = gtk_dialog_run ((GtkDialog*)phoebe_lc_coefficents_dialog);
   			switch (result){
        		case GTK_RESPONSE_OK:{
			             		gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_X1, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_x1_spinbutton)),
																				LC_COL_X2, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_x2_spinbutton)),
																				LC_COL_Y1, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_y1_spinbutton)),
																				LC_COL_Y2, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_lc_coefficents_y2_spinbutton)), -1);
            		}
        		break;

       			case GTK_RESPONSE_CANCEL:
       			break;
   			}

    		gtk_widget_destroy (phoebe_lc_coefficents_dialog);
		}
	}

	return status;
}


int gui_para_rv_coefficents_edit ()
{
	int status = 0;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
 	int rvno;
	phoebe_parameter_get_value(par, &rvno);

	if(rvno>0){

		GtkTreeModel     *model;
		GtkTreeIter       iter;
		GtkTreeSelection *selection;

		gchar *passband;
		gdouble x1,x2,y1,y2;

		GtkWidget *treeview = gui_widget_lookup("phoebe_data_rv_treeview")->gtk;
		model = gtk_tree_view_get_model((GtkTreeView*)treeview);

		treeview = gui_widget_lookup("phoebe_para_rv_ld_treeview")->gtk;

        selection = gtk_tree_view_get_selection((GtkTreeView*)treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
			gtk_tree_model_get(model, &iter,    RV_COL_FILTER,	&passband,
												RV_COL_X1,		&x1,
												RV_COL_X2,		&x2,
												RV_COL_Y1,		&y1,
												RV_COL_Y2,		&y2, -1);

			/* We use the same dialog as in the LC case */
    		gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_lc_coefficients.glade", NULL);
			gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

			GladeXML  *phoebe_rv_coefficents_xml	        = glade_xml_new        (glade_xml_file, NULL, NULL);

   			GtkWidget *phoebe_rv_coefficents_dialog         = glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_dialog");
			GtkWidget *phoebe_rv_coefficents_passband_label	= glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_passband_label");
			GtkWidget *phoebe_rv_coefficents_x1_spinbutton	= glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_x1_spinbutton");
			GtkWidget *phoebe_rv_coefficents_x2_spinbutton	= glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_x2_spinbutton");
			GtkWidget *phoebe_rv_coefficents_y1_spinbutton	= glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_y1_spinbutton");
			GtkWidget *phoebe_rv_coefficents_y2_spinbutton	= glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_y2_spinbutton");
			GtkWidget *phoebe_rv_coefficents_frame_label	= glade_xml_get_widget (phoebe_rv_coefficents_xml, "phoebe_lc_coefficents_frame_label");
			g_object_unref (phoebe_rv_coefficents_xml);

			gtk_window_set_icon (GTK_WINDOW (phoebe_rv_coefficents_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
			gtk_window_set_title (GTK_WINDOW(phoebe_rv_coefficents_dialog), "PHOEBE - RV Coefficents");
			gtk_label_set_markup(GTK_LABEL(phoebe_rv_coefficents_frame_label), "<b>RV Coefficents</b>");

			gtk_label_set_text (GTK_LABEL (phoebe_rv_coefficents_passband_label), passband);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_x1_spinbutton), x1);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_x2_spinbutton), x2);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_y1_spinbutton), y1);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_y2_spinbutton), y2);

    		gint result = gtk_dialog_run ((GtkDialog*)phoebe_rv_coefficents_dialog);
   			switch (result){
        		case GTK_RESPONSE_OK:{
			             		gtk_list_store_set((GtkListStore*)model, &iter, RV_COL_X1, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_x1_spinbutton)),
																				RV_COL_X2, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_x2_spinbutton)),
																				RV_COL_Y1, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_y1_spinbutton)),
																				RV_COL_Y2, gtk_spin_button_get_value (GTK_SPIN_BUTTON (phoebe_rv_coefficents_y2_spinbutton)), -1);
            		}
        		break;

       			case GTK_RESPONSE_CANCEL:
       			break;
   			}

    		gtk_widget_destroy (phoebe_rv_coefficents_dialog);
		}
	}

	return status;
}


void gui_spots_dialog_set_spinbutton_adjustments(bool add_spot,
						 GtkWidget *phoebe_load_spots_lat_spinbutton, GtkWidget *phoebe_load_spots_latstep_spinbutton, GtkWidget *phoebe_load_spots_latmin_spinbutton, GtkWidget *phoebe_load_spots_latmax_spinbutton,
						 GtkWidget *phoebe_load_spots_lon_spinbutton, GtkWidget *phoebe_load_spots_lonstep_spinbutton, GtkWidget *phoebe_load_spots_lonmin_spinbutton, GtkWidget *phoebe_load_spots_lonmax_spinbutton,
						 GtkWidget *phoebe_load_spots_rad_spinbutton, GtkWidget *phoebe_load_spots_radstep_spinbutton, GtkWidget *phoebe_load_spots_radmin_spinbutton, GtkWidget *phoebe_load_spots_radmax_spinbutton)
{
	GtkAdjustment *load_spots_lat_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_lat_spinbutton));
	GtkAdjustment *load_spots_latstep_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_latstep_spinbutton));
	GtkAdjustment *load_spots_latmin_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_latmin_spinbutton));
	GtkAdjustment *load_spots_latmax_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_latmax_spinbutton));

	GtkAdjustment *load_spots_lon_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_lon_spinbutton));
	GtkAdjustment *load_spots_lonstep_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_lonstep_spinbutton));
	GtkAdjustment *load_spots_lonmin_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_lonmin_spinbutton));
	GtkAdjustment *load_spots_lonmax_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_lonmax_spinbutton));

	GtkAdjustment *load_spots_rad_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_rad_spinbutton));
	GtkAdjustment *load_spots_radstep_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_radstep_spinbutton));
	GtkAdjustment *load_spots_radmin_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_radmin_spinbutton));
	GtkAdjustment *load_spots_radmax_adjustment	= gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(phoebe_load_spots_radmax_spinbutton));

	GtkWidget *phoebe_para_spots_units_combobox = gui_widget_lookup ("phoebe_para_spots_units_combobox")->gtk;

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(phoebe_para_spots_units_combobox)) == 0) {
		/* Radians */
		load_spots_lat_adjustment->upper = M_PI;
		load_spots_latstep_adjustment->upper = M_PI;
		load_spots_latmin_adjustment->upper = M_PI;
		load_spots_latmax_adjustment->upper = M_PI;
		load_spots_lat_adjustment->step_increment = 0.02;
		load_spots_latstep_adjustment->step_increment = 0.02;
		load_spots_latmin_adjustment->step_increment = 0.02;
		load_spots_latmax_adjustment->step_increment = 0.02;

		load_spots_lon_adjustment->upper = 2*M_PI;
		load_spots_lonstep_adjustment->upper = 2*M_PI;
		load_spots_lonmin_adjustment->upper = 2*M_PI;
		load_spots_lonmax_adjustment->upper = 2*M_PI;
		load_spots_lon_adjustment->step_increment = 0.02;
		load_spots_lonstep_adjustment->step_increment = 0.02;
		load_spots_lonmin_adjustment->step_increment = 0.02;
		load_spots_lonmax_adjustment->step_increment = 0.02;

		load_spots_rad_adjustment->upper = M_PI;
		load_spots_radstep_adjustment->upper = M_PI;
		load_spots_radmin_adjustment->upper = M_PI;
		load_spots_radmax_adjustment->upper = M_PI;
		load_spots_rad_adjustment->step_increment = 0.02;
		load_spots_radstep_adjustment->step_increment = 0.02;
		load_spots_radmin_adjustment->step_increment = 0.02;
		load_spots_radmax_adjustment->step_increment = 0.02;

		if (add_spot) {
			gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_load_spots_latmax_spinbutton), M_PI);
			gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_load_spots_lonmax_spinbutton), 2*M_PI);
			gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_load_spots_radmax_spinbutton), M_PI);
		}
	}
	else {
		/* Degrees */
		load_spots_lat_adjustment->upper = 180;
		load_spots_latstep_adjustment->upper = 180;
		load_spots_latmin_adjustment->upper = 180;
		load_spots_latmax_adjustment->upper = 180;
		load_spots_lat_adjustment->step_increment = 1;
		load_spots_latstep_adjustment->step_increment = 1;
		load_spots_latmin_adjustment->step_increment = 1;
		load_spots_latmax_adjustment->step_increment = 1;

		load_spots_lon_adjustment->upper = 360;
		load_spots_lonstep_adjustment->upper = 360;
		load_spots_lonmin_adjustment->upper = 360;
		load_spots_lonmax_adjustment->upper = 360;
		load_spots_lon_adjustment->step_increment = 1;
		load_spots_lonstep_adjustment->step_increment = 1;
		load_spots_lonmin_adjustment->step_increment = 1;
		load_spots_lonmax_adjustment->step_increment = 1;

		load_spots_rad_adjustment->upper = 180;
		load_spots_radstep_adjustment->upper = 180;
		load_spots_radmin_adjustment->upper = 180;
		load_spots_radmax_adjustment->upper = 180;
		load_spots_rad_adjustment->step_increment = 1;
		load_spots_radstep_adjustment->step_increment = 1;
		load_spots_radmin_adjustment->step_increment = 1;
		load_spots_radmax_adjustment->step_increment = 1;

		if (add_spot) {
			gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_load_spots_latmax_spinbutton), 180);
			gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_load_spots_lonmax_spinbutton), 360);
			gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_load_spots_radmax_spinbutton), 180);
		}
	}
}

int gui_spots_add()
{
    gui_status("Adding a spot...");
	int status = 0;

	GtkTreeModel *model;
    GtkTreeIter iter;

	gchar     *glade_xml_file                           = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_load_spots.glade", NULL);
    GladeXML  *phoebe_load_spots_xml                    = glade_xml_new        (glade_xml_file, NULL, NULL);

   	GtkWidget *phoebe_load_spots_dialog                 = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_dialog");
   	GtkWidget *phoebe_load_spots_lat_spinbutton         = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lat_spinbutton");
   	GtkWidget *phoebe_load_spots_latadjust_checkbutton  = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latadjust_checkbutton");
   	GtkWidget *phoebe_load_spots_latstep_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latstep_spinbutton");
   	GtkWidget *phoebe_load_spots_latmax_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latmax_spinbutton");
   	GtkWidget *phoebe_load_spots_latmin_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latmin_spinbutton");
   	GtkWidget *phoebe_load_spots_lon_spinbutton         = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lon_spinbutton");
   	GtkWidget *phoebe_load_spots_lonadjust_checkbutton  = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonadjust_checkbutton");
   	GtkWidget *phoebe_load_spots_lonstep_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonstep_spinbutton");
   	GtkWidget *phoebe_load_spots_lonmax_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonmax_spinbutton");
   	GtkWidget *phoebe_load_spots_lonmin_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonmin_spinbutton");
   	GtkWidget *phoebe_load_spots_rad_spinbutton         = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_rad_spinbutton");
   	GtkWidget *phoebe_load_spots_radadjust_checkbutton  = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radadjust_checkbutton");
   	GtkWidget *phoebe_load_spots_radstep_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radstep_spinbutton");
   	GtkWidget *phoebe_load_spots_radmax_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radmax_spinbutton");
   	GtkWidget *phoebe_load_spots_radmin_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radmin_spinbutton");
   	GtkWidget *phoebe_load_spots_temp_spinbutton        = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_temp_spinbutton");
   	GtkWidget *phoebe_load_spots_tempadjust_checkbutton = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempadjust_checkbutton");
   	GtkWidget *phoebe_load_spots_tempstep_spinbutton    = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempstep_spinbutton");
   	GtkWidget *phoebe_load_spots_tempmax_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempmax_spinbutton");
  	GtkWidget *phoebe_load_spots_tempmin_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempmin_spinbutton");
   	GtkWidget *phoebe_load_spots_source_combobox        = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_source_combobox");

   	g_object_unref(phoebe_load_spots_xml);

	gui_spots_dialog_set_spinbutton_adjustments(TRUE,
							phoebe_load_spots_lat_spinbutton, phoebe_load_spots_latstep_spinbutton, phoebe_load_spots_latmin_spinbutton, phoebe_load_spots_latmax_spinbutton,
							phoebe_load_spots_lon_spinbutton, phoebe_load_spots_lonstep_spinbutton, phoebe_load_spots_lonmin_spinbutton, phoebe_load_spots_lonmax_spinbutton,
							phoebe_load_spots_rad_spinbutton, phoebe_load_spots_radstep_spinbutton, phoebe_load_spots_radmin_spinbutton, phoebe_load_spots_radmax_spinbutton);

	gtk_window_set_title (GTK_WINDOW(phoebe_load_spots_dialog), "PHOEBE - Add Spot");
	int result = gtk_dialog_run ((GtkDialog*)phoebe_load_spots_dialog);
   	switch (result)	{
   	    case GTK_RESPONSE_OK:{

			GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
			model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);

			int source = gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_spots_source_combobox) + 1;
			char *source_str;

			if(source == 1)source_str = "Primary";
			else source_str = "Secondary";

			gtk_list_store_append((GtkListStore*)model, &iter);
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ACTIVE,		TRUE,
															SPOTS_COL_SOURCE,       source,
															SPOTS_COL_SOURCE_STR,   source_str,
															SPOTS_COL_LAT,          gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lat_spinbutton),
															SPOTS_COL_LATADJUST,    gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_latadjust_checkbutton),
															SPOTS_COL_LATSTEP,		gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_latstep_spinbutton),
															SPOTS_COL_LATMIN,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_latmin_spinbutton),
                                                            SPOTS_COL_LATMAX,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_latmax_spinbutton),
															SPOTS_COL_LON,          gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lon_spinbutton),
                                                            SPOTS_COL_LONADJUST,    gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_lonadjust_checkbutton),
                                                            SPOTS_COL_LONSTEP,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lonstep_spinbutton),
                                                            SPOTS_COL_LONMIN,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lonmin_spinbutton),
                                                            SPOTS_COL_LONMAX,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lonmax_spinbutton),
                                                            SPOTS_COL_RAD,          gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_rad_spinbutton),
															SPOTS_COL_RADADJUST,    gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_radadjust_checkbutton),
                                                            SPOTS_COL_RADSTEP,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_radstep_spinbutton),
                                                            SPOTS_COL_RADMIN,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_radmin_spinbutton),
                                                            SPOTS_COL_RADMAX,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_radmax_spinbutton),
                                                            SPOTS_COL_TEMP,         gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_temp_spinbutton),
															SPOTS_COL_TEMPADJUST,   gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_tempadjust_checkbutton),
                                                            SPOTS_COL_TEMPSTEP,     gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_tempstep_spinbutton),
                                                            SPOTS_COL_TEMPMIN,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_tempmin_spinbutton),
                                                            SPOTS_COL_TEMPMAX,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_tempmax_spinbutton),
															SPOTS_COL_ADJUST,       FALSE, -1);
			PHOEBE_parameter *par;
			int spots_no;

			par = phoebe_parameter_lookup("phoebe_spots_no");
			phoebe_parameter_get_value(par, &spots_no);
			phoebe_parameter_set_value(par, spots_no + 1);

			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview), &iter);
			on_phoebe_para_spots_treeview_cursor_changed ((GtkTreeView *)phoebe_para_spots_treeview, (gpointer)NULL);  // Show the new spot as the current one

			gui_status("A spot added.");
   	    }
   	    break;

   	    case GTK_RESPONSE_CANCEL:
            gui_status("Adding spot cancelled.");
   	    break;
	}
	gtk_widget_destroy (phoebe_load_spots_dialog);

	return status;
}


int gui_spots_edit()
{
	int status = 0;

	GtkTreeModel *model;
    GtkTreeIter iter;

    GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
	model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);

	if(gtk_tree_model_get_iter_first(model, &iter)){
       gchar     *glade_xml_file                           = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_load_spots.glade", NULL);
       GladeXML  *phoebe_load_spots_xml                    = glade_xml_new        (glade_xml_file, NULL, NULL);

		GtkWidget *phoebe_load_spots_dialog                 = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_dialog");
		GtkWidget *phoebe_load_spots_lat_spinbutton         = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lat_spinbutton");
		GtkWidget *phoebe_load_spots_latadjust_checkbutton  = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latadjust_checkbutton");
		GtkWidget *phoebe_load_spots_latstep_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latstep_spinbutton");
		GtkWidget *phoebe_load_spots_latmax_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latmax_spinbutton");
		GtkWidget *phoebe_load_spots_latmin_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_latmin_spinbutton");
		GtkWidget *phoebe_load_spots_lon_spinbutton         = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lon_spinbutton");
		GtkWidget *phoebe_load_spots_lonadjust_checkbutton  = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonadjust_checkbutton");
		GtkWidget *phoebe_load_spots_lonstep_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonstep_spinbutton");
		GtkWidget *phoebe_load_spots_lonmax_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonmax_spinbutton");
		GtkWidget *phoebe_load_spots_lonmin_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_lonmin_spinbutton");
		GtkWidget *phoebe_load_spots_rad_spinbutton         = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_rad_spinbutton");
		GtkWidget *phoebe_load_spots_radadjust_checkbutton  = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radadjust_checkbutton");
		GtkWidget *phoebe_load_spots_radstep_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radstep_spinbutton");
		GtkWidget *phoebe_load_spots_radmax_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radmax_spinbutton");
		GtkWidget *phoebe_load_spots_radmin_spinbutton      = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_radmin_spinbutton");
		GtkWidget *phoebe_load_spots_temp_spinbutton        = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_temp_spinbutton");
		GtkWidget *phoebe_load_spots_tempadjust_checkbutton = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempadjust_checkbutton");
		GtkWidget *phoebe_load_spots_tempstep_spinbutton    = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempstep_spinbutton");
		GtkWidget *phoebe_load_spots_tempmax_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempmax_spinbutton");
		GtkWidget *phoebe_load_spots_tempmin_spinbutton     = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_tempmin_spinbutton");
		GtkWidget *phoebe_load_spots_source_combobox        = glade_xml_get_widget (phoebe_load_spots_xml, "phoebe_load_spots_source_combobox");

		g_object_unref(phoebe_load_spots_xml);

		gui_spots_dialog_set_spinbutton_adjustments(FALSE,
								phoebe_load_spots_lat_spinbutton, phoebe_load_spots_latstep_spinbutton, phoebe_load_spots_latmin_spinbutton, phoebe_load_spots_latmax_spinbutton,
								phoebe_load_spots_lon_spinbutton, phoebe_load_spots_lonstep_spinbutton, phoebe_load_spots_lonmin_spinbutton, phoebe_load_spots_lonmax_spinbutton,
								phoebe_load_spots_rad_spinbutton, phoebe_load_spots_radstep_spinbutton, phoebe_load_spots_radmin_spinbutton, phoebe_load_spots_radmax_spinbutton);

		double lat, latstep, latmin, latmax;
		double lon, lonstep, lonmin, lonmax;
		double rad, radstep, radmin, radmax;
		double temp, tempstep, tempmin, tempmax;
		bool latadjust, lonadjust, radadjust, tempadjust;
		int source;

		GtkTreeSelection *selection;
		selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
		if (gtk_tree_selection_get_selected(selection, &model, &iter)){
			gtk_tree_model_get(model, &iter,
				SPOTS_COL_SOURCE,       &source,
				SPOTS_COL_LAT,          &lat,
				SPOTS_COL_LATADJUST,    &latadjust,
				SPOTS_COL_LATSTEP,      &latstep,
				SPOTS_COL_LATMIN,       &latmin,
				SPOTS_COL_LATMAX,       &latmax,
				SPOTS_COL_LON,          &lon,
				SPOTS_COL_LONADJUST,    &lonadjust,
				SPOTS_COL_LONSTEP,      &lonstep,
				SPOTS_COL_LONMIN,       &lonmin,
				SPOTS_COL_LONMAX,       &lonmax,
				SPOTS_COL_RAD,          &rad,
				SPOTS_COL_RADADJUST,    &radadjust,
				SPOTS_COL_RADSTEP,      &radstep,
				SPOTS_COL_RADMIN,       &radmin,
				SPOTS_COL_RADMAX,       &radmax,
				SPOTS_COL_TEMP,         &temp,
				SPOTS_COL_TEMPADJUST,   &tempadjust,
				SPOTS_COL_TEMPSTEP,     &tempstep,
				SPOTS_COL_TEMPMIN,      &tempmin,
				SPOTS_COL_TEMPMAX,      &tempmax, -1);

			gtk_combo_box_set_active    ((GtkComboBox*)     phoebe_load_spots_source_combobox,          source - 1);
			gtk_toggle_button_set_active((GtkToggleButton*) phoebe_load_spots_latadjust_checkbutton,    latadjust);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_lat_spinbutton,           lat);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_latstep_spinbutton,       latstep);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_latmin_spinbutton,        latmin);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_latmax_spinbutton,        latmax);
			gtk_toggle_button_set_active((GtkToggleButton*) phoebe_load_spots_lonadjust_checkbutton,    lonadjust);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_lon_spinbutton,           lon);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_lonstep_spinbutton,       lonstep);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_lonmin_spinbutton,        lonmin);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_lonmax_spinbutton,        lonmax);
			gtk_toggle_button_set_active((GtkToggleButton*) phoebe_load_spots_radadjust_checkbutton,    radadjust);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_rad_spinbutton,           rad);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_radstep_spinbutton,       radstep);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_radmin_spinbutton,        radmin);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_radmax_spinbutton,        radmax);
			gtk_toggle_button_set_active((GtkToggleButton*) phoebe_load_spots_tempadjust_checkbutton,   tempadjust);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_temp_spinbutton,          temp);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_tempstep_spinbutton,      tempstep);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_tempmin_spinbutton,       tempmin);
			gtk_spin_button_set_value   ((GtkSpinButton*)   phoebe_load_spots_tempmax_spinbutton,       tempmax);
		}

		gtk_window_set_title (GTK_WINDOW(phoebe_load_spots_dialog), "PHOEBE - Edit Spot Parameters");
		int result = gtk_dialog_run ((GtkDialog*)phoebe_load_spots_dialog);
        switch (result){
            case GTK_RESPONSE_OK:{

                source = gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_spots_source_combobox) + 1;
                char *source_str;

				if(source == 1)source_str = "Primary";
				else source_str = "Secondary";

                gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST,       FALSE,
                                                                SPOTS_COL_SOURCE,       source,
                                                                SPOTS_COL_SOURCE_STR,   source_str,
                                                                SPOTS_COL_LAT,          gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lat_spinbutton),
                                                                SPOTS_COL_LATADJUST,    gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_latadjust_checkbutton),
                                                                SPOTS_COL_LATSTEP,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_latstep_spinbutton),
                                                                SPOTS_COL_LATMIN,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_latmin_spinbutton),
                                                                SPOTS_COL_LATMAX,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_latmax_spinbutton),
                                                                SPOTS_COL_LON,          gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lon_spinbutton),
                                                                SPOTS_COL_LONADJUST,    gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_lonadjust_checkbutton),
                                                                SPOTS_COL_LONSTEP,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lonstep_spinbutton),
                                                                SPOTS_COL_LONMIN,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lonmin_spinbutton),
                                                                SPOTS_COL_LONMAX,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_lonmax_spinbutton),
                                                                SPOTS_COL_RAD,          gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_rad_spinbutton),
                                                                SPOTS_COL_RADADJUST,    gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_radadjust_checkbutton),
                                                                SPOTS_COL_RADSTEP,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_radstep_spinbutton),
                                                                SPOTS_COL_RADMIN,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_radmin_spinbutton),
                                                                SPOTS_COL_RADMAX,       gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_radmax_spinbutton),
                                                                SPOTS_COL_TEMP,         gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_temp_spinbutton),
                                                                SPOTS_COL_TEMPADJUST,   gtk_toggle_button_get_active((GtkToggleButton*)phoebe_load_spots_tempadjust_checkbutton),
                                                                SPOTS_COL_TEMPSTEP,     gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_tempstep_spinbutton),
                                                                SPOTS_COL_TEMPMIN,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_tempmin_spinbutton),
                                                                SPOTS_COL_TEMPMAX,      gtk_spin_button_get_value   ((GtkSpinButton*)  phoebe_load_spots_tempmax_spinbutton), -1);

				gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview), &iter);
				on_phoebe_para_spots_treeview_cursor_changed ((GtkTreeView *)phoebe_para_spots_treeview, (gpointer)NULL);  // Show the new spot parameters as the current ones
            }
        }
        gtk_widget_destroy (phoebe_load_spots_dialog);
	}
	return status;
}
