#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_callbacks.h"

int gui_init_treeviews(GladeXML *phoebe_window)
{
    gui_init_lc_treeviews    (phoebe_window);
    gui_init_rv_treeviews    (phoebe_window);
    gui_init_spots_treeview  (phoebe_window);

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

    GtkTreeModel *lc_model = lc_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", LC_COL_ACTIVE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_ACTIVE);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_lc_active_checkbutton_toggled), NULL);

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
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", LC_COL_ITYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_ITYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", LC_COL_DTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_DTYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", LC_COL_WTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_lc_treeview, column, LC_COL_WTYPE_STR);

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
    // g_print("---------- Initializing rv treeviews ---------------\n");

    phoebe_data_rv_treeview    = glade_xml_get_widget (phoebe_window, "phoebe_data_rv_treeview");
    phoebe_para_rv_ld_treeview = glade_xml_get_widget (phoebe_window, "phoebe_para_ld_rvcoefs_treeview");

    GtkTreeModel *rv_model = rv_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Active", renderer, "active", RV_COL_ACTIVE, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_ACTIVE);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_data_rv_active_checkbutton_toggled), NULL);

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
    column      = gtk_tree_view_column_new_with_attributes("Col. 1", renderer, "text", RV_COL_ITYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_ITYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 2", renderer, "text", RV_COL_DTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_DTYPE_STR);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Col. 3", renderer, "text", RV_COL_WTYPE_STR, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_data_rv_treeview, column, RV_COL_WTYPE_STR);

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


int gui_init_spots_treeview  (GladeXML *phoebe_window)
{

    // g_print("---------- Initializing spots treeview -------------\n");

    phoebe_para_surf_spots_treeview = glade_xml_get_widget (phoebe_window, "phoebe_para_surf_spots_treeview");

    GtkTreeModel *spots_model = spots_model_create();

    GtkCellRenderer     *renderer;
    GtkTreeViewColumn   *column;

    renderer    = gtk_cell_renderer_toggle_new ();
    column      = gtk_tree_view_column_new_with_attributes("Adjust", renderer, "active", SPOTS_COL_ADJUST, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_ADJUST);

    g_signal_connect(renderer, "toggled", GTK_SIGNAL_FUNC(on_phoebe_para_surf_spots_adjust_checkbutton_toggled), NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(column, "Source");
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_SOURCE);
    gtk_tree_view_column_pack_start(column, renderer, TRUE);
    gtk_tree_view_column_set_cell_data_func(column, renderer, spots_source_cell_data_func, NULL, NULL);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Latitude", renderer, "text", SPOTS_COL_LAT, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_LAT);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Longitude", renderer, "text", SPOTS_COL_LON, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_LON);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Radius", renderer, "text", SPOTS_COL_RAD, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_RAD);

    renderer    = gtk_cell_renderer_text_new ();
    column      = gtk_tree_view_column_new_with_attributes("Temperature", renderer, "text", SPOTS_COL_TEMP, NULL);
    gtk_tree_view_insert_column ((GtkTreeView*)phoebe_para_surf_spots_treeview, column, SPOTS_COL_TEMP);

    gtk_tree_view_set_model ((GtkTreeView*)phoebe_para_surf_spots_treeview, spots_model);

    return SUCCESS;
}

GtkTreeModel *lc_model_create()
{
    GtkListStore *model = gtk_list_store_new(LC_COL_COUNT,          /* number of columns    */
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

void spots_source_cell_data_func(GtkTreeViewColumn   *column,
                                 GtkCellRenderer     *renderer,
                                 GtkTreeModel        *model,
                                 GtkTreeIter         *iter,
                                 gpointer             data)
{
    int   source;
    char *source_str;

    gtk_tree_model_get(model, iter, SPOTS_COL_SOURCE, &source, -1);

    switch(source)
    {
        case 0:
            source_str = "Primary star";
            break;
        case 1:
            source_str = "Secondary star";
            break;
        default:
            source_str = "";
            break;
    }
    g_object_set(renderer, "text", source_str, NULL);
}
