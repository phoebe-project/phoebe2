#include <phoebe/phoebe.h>

#include "phoebe_gui_treeviews.h"

int gui_init_treeviews(GladeXML *phoebe_window)
{
    phoebe_data_lc_treeview = glade_xml_get_widget(phoebe_window, "phoebe_data_lc_treeview");
    intern_connect_curves_view_to_model(phoebe_data_lc_treeview, intern_create_curves_model());

    phoebe_data_rv_treeview = glade_xml_get_widget(phoebe_window, "phoebe_data_rv_treeview");
    intern_connect_curves_view_to_model(phoebe_data_rv_treeview, intern_create_curves_model());

    return SUCCESS;
}

GtkTreeModel *intern_create_curves_model()
{
    /* Creating the model:                                                               */
    GtkListStore *model = gtk_list_store_new(CURVELIST_COL_COUNT,  /* number of columns  */
                                             G_TYPE_STRING,        /* filename           */
                                             G_TYPE_STRING,        /* passband           */
                                             G_TYPE_STRING,        /* itype              */
                                             G_TYPE_STRING,        /* dtype              */
                                             G_TYPE_STRING,        /* wtype              */
                                             G_TYPE_DOUBLE);       /* sigma              */
    return (GtkTreeModel*)model;
}

void intern_connect_curves_view_to_model(GtkWidget *view, GtkTreeModel *model)
{
    /* Renderer tells us the type of the cell: is it text, progress-bar, toggle... */
    GtkCellRenderer     *renderer;

    /* Filling the columns: */
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes ((GtkTreeView*)view,        /* the treeview to insert the column in                     */
                                                 -1,                        /* where the new column will be inserted; -1 is for "end"   */
                                                 "Filename",                /* the column header                                        */
                                                 renderer,                  /* the cell renderer                                        */
                                                                            /* the optional list of column attributes (to be explored): */
                                                 "text",                    /* content type (I guess)                                   */
                                                 CURVELIST_COL_FILENAME,    /* column number                                            */
                                                 NULL);                     /* end of attribute list                                    */

    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes ((GtkTreeView*)view, -1, "Passband",         renderer, "text", CURVELIST_COL_FILTER, NULL);

    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes ((GtkTreeView*)view, -1, "Independant var.", renderer, "text", CURVELIST_COL_ITYPE,    NULL);

    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes ((GtkTreeView*)view, -1, "Dependant var.",   renderer, "text", CURVELIST_COL_DTYPE,    NULL);

    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes ((GtkTreeView*)view, -1, "Error type",       renderer, "text", CURVELIST_COL_DTYPE,    NULL);

    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes ((GtkTreeView*)view, -1, "Sigma",            renderer, "text", CURVELIST_COL_SIGMA,   NULL);

    gtk_tree_view_set_model((GtkTreeView*)view, model);
}
