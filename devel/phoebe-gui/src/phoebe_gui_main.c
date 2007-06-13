#ifdef HAVE_CONFIG_H
#  include <phoebe_gui_build_config.h>
#endif

#include <phoebe/phoebe.h>

#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_main.h"

/* These columns will appear in the phoebe_data_lc/rv_treeview */
typedef enum curves_view_columns
{
    filename,
    passband,
    itype,
    dtype,
    wtype,
    sigma,
    column_count,
}curves_view_columns;

/* This function will connect the data container (model) to the data view widget (a treeview) */
void connect_curves_view_to_model(GtkTreeView*, GtkTreeModel*);

/* Creates a model for storing phoebe_curves data */
GtkListStore *create_curves_model(void);

int main (int argc, char *argv[])
{
    GladeXML *phoebe_gui;

    gtk_set_locale();
    gtk_init(&argc, &argv);
    glade_init();

    phoebe_gui = glade_xml_new("phoebe.glade", NULL, NULL);
    glade_xml_signal_autoconnect (phoebe_gui);

    phoebe_window = glade_xml_get_widget(phoebe_gui, "phoebe_window");
    
    GtkTreeView *phoebe_data_lc_treeview = (GtkTreeView*)glade_xml_get_widget(phoebe_gui, "phoebe_data_lc_treeview");
    lc_curves_model = (GtkListStore*)create_curves_model();
    connect_curves_view_to_model(phoebe_data_lc_treeview, lc_curves_model);
    
    GtkTreeView *phoebe_data_rv_treeview = (GtkTreeView *) glade_xml_get_widget(phoebe_gui, "phoebe_data_rv_treeview");
    rv_curves_model = (GtkListStore*)create_curves_model();
    connect_curves_view_to_model(phoebe_data_rv_treeview, rv_curves_model);

    gtk_widget_show(phoebe_window);

    gtk_main();

    return SUCCESS;
}

GtkListStore *create_curves_model()
{
    /* --- Creating the model:                                                   */
    GtkListStore *model = gtk_list_store_new(column_count,  /* number of columns */
                                             G_TYPE_STRING, /* filename          */
                                             G_TYPE_STRING, /* passband          */
                                             G_TYPE_STRING, /* itype             */
                                             G_TYPE_STRING, /* dtype             */
                                             G_TYPE_STRING, /* wtype             */
                                             G_TYPE_DOUBLE);/* sigma             */
    return model;
}

void connect_curves_view_to_model(GtkTreeView *view, GtkTreeModel *model)
{
    /* Renderer tells us the type of the cell: is it text, progress-bar, toggle... */
    GtkCellRenderer     *renderer;

    /* --- Filling the columns: --- */
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes (view,       /* the treeview to insert the column in                     */
                                                 -1,         /* where the new column will be inserted; -1 is for "end"   */
                                                 "Filename", /* the column header                                        */
                                                 renderer,   /* the cell renderer                                        */
                                                             /* the optional list of column attributes (to be explored): */
                                                 "text",     /* content type (I guess)                                   */
                                                 filename,   /* column number                                            */
                                                 NULL);      /* end of attribute list                                    */
                                                 
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes (view, -1, "Passband",         renderer, "text", passband, NULL);
    
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes (view, -1, "Independant var.", renderer, "text", itype,    NULL);
    
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes (view, -1, "Dependant var.",   renderer, "text", dtype,    NULL);
    
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes (view, -1, "Error type",       renderer, "text", wtype,    NULL);
    
    renderer = gtk_cell_renderer_text_new ();
    gtk_tree_view_insert_column_with_attributes (view, -1, "Sigma",            renderer, "text", sigma,   NULL);
    
    gtk_tree_view_set_model(view, model);
}











