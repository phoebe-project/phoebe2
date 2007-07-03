#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"

int gui_init_treeviews(GladeXML *phoebe_window)
{
    gui_init_lc_treeviews(phoebe_window);
    gui_init_rv_treeviews(phoebe_window);

    return SUCCESS;
}

int gui_init_lc_treeviews(GladeXML *phoebe_window)
{
    phoebe_data_lc_treeview             = glade_xml_get_widget (phoebe_window, "phoebe_data_lc_treeview");
    phoebe_para_lc_el3_treeview         = glade_xml_get_widget (phoebe_window, "phoebe_params_lumins_3rdlight_treeview");
    phoebe_para_lc_levels_treeview      = glade_xml_get_widget (phoebe_window, "phoebe_params_lumins_levels_treeview");
    phoebe_para_lc_levweight_treeview   = glade_xml_get_widget (phoebe_window, "phoebe_params_lumins_weighting_treeview");
    phoebe_para_lc_ld_treeview          = glade_xml_get_widget (phoebe_window, "phoebe_params_ld_lccoefs_treeview");

    GtkTreeModel *lc_model = lc_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", LC_COL_ACTIVE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_ACTIVE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_FILENAME);

    /* we don't need the filename in every treeview

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_FILENAME);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_levels_treeview, column, LC_COL_FILENAME);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_levweight_treeview, column, LC_COL_FILENAME);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", LC_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_FILENAME);

    */

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", LC_COL_FILTER, NULL);
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

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", LC_COL_ITYPE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_ITYPE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", LC_COL_DTYPE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_DTYPE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", LC_COL_WTYPE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_WTYPE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", LC_COL_SIGMA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_SIGMA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Level weighting", renderer, "text", LC_COL_LEVWEIGHT, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levweight_treeview, column, LC_COL_LEVWEIGHT);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Primary levels", renderer, "text", LC_COL_HLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, LC_COL_HLA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Secondary levels", renderer, "text", LC_COL_CLA, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_levels_treeview, column, LC_COL_CLA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Opacity function", renderer, "text", LC_COL_OPSF, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_OPSF);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Third light", renderer, "text", LC_COL_EL3, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_EL3);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Extinction", renderer, "text", LC_COL_EXTINCTION, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_el3_treeview, column, LC_COL_EXTINCTION);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", LC_COL_X1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_X1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", LC_COL_X2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_X2);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", LC_COL_Y1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_Y1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", LC_COL_Y2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_lc_ld_treeview, column, LC_COL_Y2);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_data_lc_treeview,            lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_el3_treeview,        lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_levels_treeview,     lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_levweight_treeview,  lc_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_lc_ld_treeview,         lc_model);

    return SUCCESS;
}

int gui_init_rv_treeviews(GladeXML *phoebe_window)
{
    phoebe_data_rv_treeview             = glade_xml_get_widget (phoebe_window, "phoebe_data_rv_treeview");
    phoebe_para_rv_ld_treeview          = glade_xml_get_widget (phoebe_window, "phoebe_params_ld_rvcoefs_treeview");

    GtkTreeModel *rv_model = rv_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", RV_COL_ACTIVE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_ACTIVE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filename", renderer, "text", RV_COL_FILENAME, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_FILENAME);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Filter", renderer, "text", RV_COL_FILTER, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_FILTER);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", RV_COL_ITYPE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_ITYPE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", RV_COL_DTYPE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_DTYPE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", RV_COL_WTYPE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_WTYPE);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Sigma", renderer, "text", RV_COL_SIGMA, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_SIGMA);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X1", renderer, "text", RV_COL_X1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_X1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("X2", renderer, "text", RV_COL_X2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_X2);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y1", renderer, "text", RV_COL_Y1, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_Y1);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Y2", renderer, "text", RV_COL_Y2, NULL);
    gtk_tree_view_insert_column((GtkTreeView*)phoebe_para_rv_ld_treeview, column, RV_COL_Y2);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_data_rv_treeview,            rv_model);
    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_rv_ld_treeview,         rv_model);

    return SUCCESS;
}

GtkTreeModel *lc_model_create()
{
    /* Creating the model:                                                                  */
    GtkListStore *model = gtk_list_store_new(LC_COL_COUNT,          /* number of columns    */
                                             G_TYPE_BOOLEAN,        /* active               */
                                             G_TYPE_STRING,         /* filename             */
                                             G_TYPE_STRING,         /* passband             */
                                             G_TYPE_STRING,         /* itype                */
                                             G_TYPE_STRING,         /* dtype                */
                                             G_TYPE_STRING,         /* wtype                */
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
    /* Creating the model:                                                                  */
    GtkListStore *model = gtk_list_store_new(RV_COL_COUNT,          /* number of columns    */
                                             G_TYPE_BOOLEAN,        /* active               */
                                             G_TYPE_STRING,         /* filename             */
                                             G_TYPE_STRING,         /* passband             */
                                             G_TYPE_STRING,         /* itype                */
                                             G_TYPE_STRING,         /* dtype                */
                                             G_TYPE_STRING,         /* wtype                */
                                             G_TYPE_DOUBLE,         /* sigma                */
                                             G_TYPE_DOUBLE,         /* rvx1                 */
                                             G_TYPE_DOUBLE,         /* rvx2                 */
                                             G_TYPE_DOUBLE,         /* rvy1                 */
                                             G_TYPE_DOUBLE);        /* rvy2                 */
    return (GtkTreeModel*)model;
}
