#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_base.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_types.h"

void
on_phoebe_test_toolbutton_0_clicked      (GtkToolButton   *toolbutton,
                                        	gpointer         user_data)
{
    gui_get_values_from_widgets();
}

void
on_phoebe_test_toolbutton_1_clicked      (GtkToolButton   *toolbutton,
                                        	gpointer         user_data)
{
    gui_set_values_to_widgets();
}

/* ******************************************************************** *
 *
 *                    phoebe_sidesheet_data_treeview events
 *
 * ******************************************************************** */


void on_phoebe_sidesheet_data_tba_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_data_lc_treeview events
 *
 * ******************************************************************** */


void
on_phoebe_data_lc_treeview_row_activated
                                        (GtkTreeView      *treeview,
                                        GtkTreePath       *path,
                                        GtkTreeViewColumn *column,
                                        gpointer           user_data)
{

}


void
on_phoebe_data_lc_treeview_cursor_changed
                                        (GtkTreeView *tree_view,
                                         gpointer     user_data)
{

}


void
on_phoebe_data_lc_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data)
{
    GladeXML  *phoebe_load_lc_xml                   = glade_xml_new        ("../glade/phoebe_load_lc.glade", NULL, NULL);

    GtkWidget *phoebe_load_lc_dialog                = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_dialog");
	GtkWidget *phoebe_load_lc_filechooserbutton     = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
    GtkWidget *phoebe_load_lc_column1_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column1_combobox");
    GtkWidget *phoebe_load_lc_column2_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column2_combobox");
    GtkWidget *phoebe_load_lc_column3_combobox      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_column3_combobox");
    GtkWidget *phoebe_load_lc_sigma_spinbutton      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_sigma_spinbutton");
    GtkWidget *phoebe_load_lc_reddening_checkbutton = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_reddening_checkbutton");
    GtkWidget *phoebe_load_lc_r_spinbutton			= glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_r_spinbutton");
    GtkWidget *phoebe_load_lc_e_spinbutton          = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_e_spinbutton");
    GtkWidget *phoebe_load_lc_preview_textview      = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_preview_textview");
    GtkWidget *phoebe_load_lc_filter_combobox       = glade_xml_get_widget (phoebe_load_lc_xml, "phoebe_load_lc_filter_combobox");

	g_object_unref (phoebe_load_lc_xml);

	gtk_window_set_icon (GTK_WINDOW(phoebe_load_lc_dialog), gdk_pixbuf_new_from_file("ico.png", NULL));
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
			gchar 		filter_selected[255] = "Undefined";

            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

			if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_lc_filter_combobox), &filter_iter)) {
				gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_lc_filter_combobox)), &filter_iter, 1, &filter_number, -1);
				sprintf (filter_selected, "%s  (%.0lfnm)", PHOEBE_passbands[filter_number]->name, PHOEBE_passbands[filter_number]->effwl/10.);
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

            PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
            int lcno;

            phoebe_parameter_get_value(par, &lcno);
            phoebe_parameter_set_value(par, lcno + 1);

            printf("Number of light curves: %d\n", lcno + 1);
			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_lc_treeview), &iter);
        }
        break;

        case GTK_RESPONSE_CANCEL:
        break;
    }

    gtk_widget_destroy (phoebe_load_lc_dialog);
}


void
on_phoebe_data_lc_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreeModel     *model;
    GtkTreeIter       iter;

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

    if(gtk_tree_model_get_iter_first(model, &iter)){
        GladeXML  *phoebe_load_lc_xml                   = glade_xml_new       ("../glade/phoebe_load_lc.glade", NULL, NULL);
        GtkWidget *phoebe_load_lc_dialog                = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_dialog");
        GtkWidget *phoebe_load_lc_filechooserbutton     = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_filechooserbutton");
        GtkWidget *phoebe_load_lc_column1_combobox      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_column1_combobox");
        GtkWidget *phoebe_load_lc_column2_combobox      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_column2_combobox");
        GtkWidget *phoebe_load_lc_column3_combobox      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_column3_combobox");
        GtkWidget *phoebe_load_lc_sigma_spinbutton      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_sigma_spinbutton");
        GtkWidget *phoebe_load_lc_reddening_checkbutton = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_reddening_checkbutton");
        GtkWidget *phoebe_load_lc_r_spinbutton			= glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_r_spinbutton");
        GtkWidget *phoebe_load_lc_e_spinbutton          = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_e_spinbutton");
        GtkWidget *phoebe_load_lc_preview_textview      = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_preview_textview");
        GtkWidget *phoebe_load_lc_filter_combobox       = glade_xml_get_widget(phoebe_load_lc_xml, "phoebe_load_lc_filter_combobox");

        gchar *filename;
        gint itype;
        gint dtype;
        gint wtype;
		gchar *filter;
        gdouble sigma;

        gchar filter_selected[255] = "Undefined";
		gint filter_number;
		GtkTreeIter filter_iter;

        g_object_unref(phoebe_load_lc_xml);

		gtk_window_set_icon (GTK_WINDOW(phoebe_load_lc_dialog), gdk_pixbuf_new_from_file("ico.png", NULL));
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
					sprintf (filter_selected, "%s  (%.0lfnm)", PHOEBE_passbands[filter_number]->name, PHOEBE_passbands[filter_number]->effwl/10.);
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
}


void
on_phoebe_data_lc_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_lc_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        gtk_list_store_remove((GtkListStore*)model, &iter);

        PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
        int lcno;

        phoebe_parameter_get_value(par, &lcno);
        phoebe_parameter_set_value(par, lcno - 1);

        printf("Number of light curves: %d\n", lcno - 1);
    }
}


void on_phoebe_data_lc_active_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data)
{
    GtkTreeModel *model;
    GtkTreeIter iter;
    int active;

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

    if(gtk_tree_model_get_iter_from_string(model, &iter, path)){
        g_object_get(renderer, "active", &active, NULL);

        if(active)
            gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_ACTIVE, FALSE, -1);
        else
            gtk_list_store_set((GtkListStore*)model, &iter, LC_COL_ACTIVE, TRUE, -1);
    }
}

void on_phoebe_load_lc_filechooserbutton_selection_changed (GtkFileChooserButton *filechooserbutton, gpointer user_data)
{
	set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
}


/* ******************************************************************** *
 *
 *                    phoebe_data_rv_treeview events
 *
 * ******************************************************************** */


void
on_phoebe_data_rv_treeview_row_activated
                                        (GtkTreeView        *treeview,
                                         GtkTreePath        *path,
                                         GtkTreeViewColumn  *column,
                                         gpointer            user_data)
{

}


void
on_phoebe_data_rv_treeview_cursor_changed
                                        (GtkTreeView *tree_view,
                                         gpointer     user_data)
{

}


void
on_phoebe_data_rv_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data)
{
    GladeXML  *phoebe_load_rv_xml                   = glade_xml_new       ("../glade/phoebe_load_rv.glade", NULL, NULL);

    GtkWidget *phoebe_load_rv_dialog                = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_dialog");
	GtkWidget *phoebe_load_rv_filechooserbutton     = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filechooserbutton");
    GtkWidget *phoebe_load_rv_column1_combobox      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_column1_combobox");
    GtkWidget *phoebe_load_rv_column2_combobox      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_column2_combobox");
    GtkWidget *phoebe_load_rv_column3_combobox      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_column3_combobox");
    GtkWidget *phoebe_load_rv_sigma_spinbutton      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_sigma_spinbutton");
    GtkWidget *phoebe_load_rv_preview_textview      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_preview_textview");

    GtkWidget *phoebe_load_rv_filter_combobox       = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filter_combobox");

	gtk_window_set_icon (GTK_WINDOW(phoebe_load_rv_dialog), gdk_pixbuf_new_from_file("ico.png", NULL));
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

    gint result = gtk_dialog_run ((GtkDialog*)phoebe_load_rv_dialog);
    switch (result){
        case GTK_RESPONSE_OK:{

			GtkTreeIter filter_iter;
			gint 		filter_number;
			gchar 		filter_selected[255] = "Undefined";

            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

			if (gtk_combo_box_get_active_iter (GTK_COMBO_BOX (phoebe_load_rv_filter_combobox), &filter_iter)) {
				gtk_tree_model_get (gtk_combo_box_get_model(GTK_COMBO_BOX(phoebe_load_rv_filter_combobox)), &filter_iter, 1, &filter_number, -1);
				sprintf (filter_selected, "%s  (%.0lfnm)", PHOEBE_passbands[filter_number]->name, PHOEBE_passbands[filter_number]->effwl/10.);
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

            char *indep_str = strdup(indep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column1_combobox)]);
            printf("RV indep given to model when adding: %s\n", indep_str);
			gtk_tree_model_get(model, &iter, RV_COL_ITYPE_STR, &indep_str, -1);
			int status = phoebe_parameter_set_value(indep, 0, indep_str);
			printf ("%s", phoebe_error (status));

			char *dep_str = strdup(dep->menu->option[gtk_combo_box_get_active((GtkComboBox*)phoebe_load_rv_column2_combobox)]);
            printf("RV dep given to model when adding: %s\n", dep_str);
			gtk_tree_model_get(model, &iter, RV_COL_DTYPE_STR, &dep_str, -1);
			status = phoebe_parameter_set_value(dep, 0, dep_str);
			printf ("%s", phoebe_error (status));

			gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview), &iter);
        }
        break;

        case GTK_RESPONSE_CANCEL:
        break;
    }
    gtk_widget_destroy (phoebe_load_rv_dialog);
}

void on_phoebe_load_rv_filechooserbutton_selection_changed (GtkFileChooserButton *filechooserbutton, gpointer user_data)
{
	set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
}

void
on_phoebe_data_rv_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreeModel *model;
    GtkTreeIter iter;

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

    if(gtk_tree_model_get_iter_first(model, &iter)){
        GladeXML  *phoebe_load_rv_xml                   = glade_xml_new       ("../glade/phoebe_load_rv.glade", NULL, NULL);
        GtkWidget *phoebe_load_rv_dialog                = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_dialog");
        GtkWidget *phoebe_load_rv_filechooserbutton     = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filechooserbutton");
        GtkWidget *phoebe_load_rv_column1_combobox      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_column1_combobox");
        GtkWidget *phoebe_load_rv_column2_combobox      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_column2_combobox");
        GtkWidget *phoebe_load_rv_column3_combobox      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_column3_combobox");
        GtkWidget *phoebe_load_rv_sigma_spinbutton      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_sigma_spinbutton");
        GtkWidget *phoebe_load_rv_preview_textview      = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_preview_textview");
        GtkWidget *phoebe_load_rv_filter_combobox       = glade_xml_get_widget(phoebe_load_rv_xml, "phoebe_load_rv_filter_combobox");

		gchar *filename;
        gint itype;
        gint dtype;
        gint wtype;
        gdouble sigma;
        gchar *filter;

        gchar filter_selected[255] = "Undefined";
		gint filter_number;
		GtkTreeIter filter_iter;

		gtk_window_set_icon (GTK_WINDOW(phoebe_load_rv_dialog), gdk_pixbuf_new_from_file("ico.png", NULL));
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
					sprintf (filter_selected, "%s  (%.0lfnm)", PHOEBE_passbands[filter_number]->name, PHOEBE_passbands[filter_number]->effwl/10.);
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
}


void
on_phoebe_data_rv_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_data_rv_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        gtk_list_store_remove((GtkListStore*)model, &iter);

        PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
        int rvno;

        phoebe_parameter_get_value(par, &rvno);
        phoebe_parameter_set_value(par, rvno - 1);

        printf("Number of RV curves: %d\n", rvno - 1);
    }
}


void on_phoebe_data_rv_active_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data)
{
    GtkTreeModel *model;
    GtkTreeIter iter;
    int active;

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

    if(gtk_tree_model_get_iter_from_string(model, &iter, path)){
        g_object_get(renderer, "active", &active, NULL);

        if(active)
            gtk_list_store_set((GtkListStore*)model, &iter, RV_COL_ACTIVE, FALSE, -1);
        else
            gtk_list_store_set((GtkListStore*)model, &iter, RV_COL_ACTIVE, TRUE, -1);
    }
}


/* ******************************************************************** *
 *
 *               phoebe_para_surf_spots_treeview events
 *
 * ******************************************************************** */


void
on_phoebe_para_surf_spots_treeview_row_activated
                                        (GtkTreeView      *treeview,
                                        GtkTreePath       *path,
                                        GtkTreeViewColumn *column,
                                        gpointer           user_data)
{

}


void
on_phoebe_para_surf_spots_treeview_cursor_changed
                                        (GtkTreeView *tree_view,
                                         gpointer     user_data)
{

}

void
on_phoebe_para_surf_spots_add_button_clicked   (GtkButton       *button,
                                                gpointer         user_data)
{
    GladeXML  *phoebe_load_spots_xml                    = glade_xml_new       ("../glade/phoebe_load_spots.glade", NULL, NULL);

	GtkWidget *phoebe_load_spots_dialog                 = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_dialog");
	GtkWidget *phoebe_load_spots_lat_spinbutton         = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lat_spinbutton");
	GtkWidget *phoebe_load_spots_latadjust_checkbutton  = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latadjust_checkbutton");
	GtkWidget *phoebe_load_spots_latstep_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latstep_spinbutton");
	GtkWidget *phoebe_load_spots_latmax_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latmax_spinbutton");
	GtkWidget *phoebe_load_spots_latmin_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latmin_spinbutton");
	GtkWidget *phoebe_load_spots_lon_spinbutton         = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lon_spinbutton");
	GtkWidget *phoebe_load_spots_lonadjust_checkbutton  = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonadjust_checkbutton");
	GtkWidget *phoebe_load_spots_lonstep_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonstep_spinbutton");
	GtkWidget *phoebe_load_spots_lonmax_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonmax_spinbutton");
	GtkWidget *phoebe_load_spots_lonmin_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonmin_spinbutton");
	GtkWidget *phoebe_load_spots_rad_spinbutton         = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_rad_spinbutton");
	GtkWidget *phoebe_load_spots_radadjust_checkbutton  = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radadjust_checkbutton");
	GtkWidget *phoebe_load_spots_radstep_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radstep_spinbutton");
	GtkWidget *phoebe_load_spots_radmax_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radmax_spinbutton");
	GtkWidget *phoebe_load_spots_radmin_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radmin_spinbutton");
	GtkWidget *phoebe_load_spots_temp_spinbutton        = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_temp_spinbutton");
	GtkWidget *phoebe_load_spots_tempadjust_checkbutton = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempadjust_checkbutton");
	GtkWidget *phoebe_load_spots_tempstep_spinbutton    = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempstep_spinbutton");
	GtkWidget *phoebe_load_spots_tempmax_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempmax_spinbutton");
	GtkWidget *phoebe_load_spots_tempmin_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempmin_spinbutton");
	GtkWidget *phoebe_load_spots_source_combobox        = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_source_combobox");

	g_object_unref(phoebe_load_spots_xml);

    GtkTreeModel *model;
    GtkTreeIter iter;

	int result = gtk_dialog_run ((GtkDialog*)phoebe_load_spots_dialog);
	switch (result)	{
	    case GTK_RESPONSE_OK:{

            model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_surf_spots_treeview);

            /* source gets set separately because 1 is for primary, and 2 for secondary, while the
               combo returns 0 for primary and 1 for secondary; and, we need it later */
            int source = gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_spots_source_combobox) + 1;
            char *source_str;

            if(source == 1)source_str = "Primary";
            else source_str = "Secondary";

            gtk_list_store_append((GtkListStore*)model, &iter);
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

            PHOEBE_parameter *par;
            int spots_no;

            if (source == 1){
                /* the primary */
                par = phoebe_parameter_lookup("phoebe_spots_no1");
                phoebe_parameter_get_value(par, &spots_no);
                phoebe_parameter_set_value(par, spots_no + 1);
                printf("Number of spots on the primary: %d\n", spots_no + 1);
            }
            else{
                /* the secondary */
                par = phoebe_parameter_lookup("phoebe_spots_no2");
                phoebe_parameter_get_value(par, &spots_no);
                phoebe_parameter_set_value(par, spots_no + 1);
                printf("Number of spots on the secondary: %d\n", spots_no + 1);
            }
	    }
        break;

        case GTK_RESPONSE_CANCEL:
        break;
	}

    gtk_widget_destroy (phoebe_load_spots_dialog);
}


void
on_phoebe_para_surf_spots_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreeModel *model;
    GtkTreeIter iter;

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_surf_spots_treeview);

    if(gtk_tree_model_get_iter_first(model, &iter)){
        GladeXML  *phoebe_load_spots_xml                    = glade_xml_new       ("../glade/phoebe_load_spots.glade", NULL, NULL);

        GtkWidget *phoebe_load_spots_dialog                 = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_dialog");
        GtkWidget *phoebe_load_spots_lat_spinbutton         = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lat_spinbutton");
        GtkWidget *phoebe_load_spots_latadjust_checkbutton  = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latadjust_checkbutton");
        GtkWidget *phoebe_load_spots_latstep_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latstep_spinbutton");
        GtkWidget *phoebe_load_spots_latmax_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latmax_spinbutton");
        GtkWidget *phoebe_load_spots_latmin_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_latmin_spinbutton");
        GtkWidget *phoebe_load_spots_lon_spinbutton         = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lon_spinbutton");
        GtkWidget *phoebe_load_spots_lonadjust_checkbutton  = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonadjust_checkbutton");
        GtkWidget *phoebe_load_spots_lonstep_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonstep_spinbutton");
        GtkWidget *phoebe_load_spots_lonmax_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonmax_spinbutton");
        GtkWidget *phoebe_load_spots_lonmin_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_lonmin_spinbutton");
        GtkWidget *phoebe_load_spots_rad_spinbutton         = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_rad_spinbutton");
        GtkWidget *phoebe_load_spots_radadjust_checkbutton  = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radadjust_checkbutton");
        GtkWidget *phoebe_load_spots_radstep_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radstep_spinbutton");
        GtkWidget *phoebe_load_spots_radmax_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radmax_spinbutton");
        GtkWidget *phoebe_load_spots_radmin_spinbutton      = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_radmin_spinbutton");
        GtkWidget *phoebe_load_spots_temp_spinbutton        = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_temp_spinbutton");
        GtkWidget *phoebe_load_spots_tempadjust_checkbutton = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempadjust_checkbutton");
        GtkWidget *phoebe_load_spots_tempstep_spinbutton    = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempstep_spinbutton");
        GtkWidget *phoebe_load_spots_tempmax_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempmax_spinbutton");
        GtkWidget *phoebe_load_spots_tempmin_spinbutton     = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_tempmin_spinbutton");
        GtkWidget *phoebe_load_spots_source_combobox        = glade_xml_get_widget(phoebe_load_spots_xml, "phoebe_load_spots_source_combobox");

        g_object_unref(phoebe_load_spots_xml);

        double lat, latstep, latmin, latmax;
        double lon, lonstep, lonmin, lonmax;
        double rad, radstep, radmin, radmax;
        double temp, tempstep, tempmin, tempmax;
        bool latadjust, lonadjust, radadjust, tempadjust;
        int source_old, source_new;

        GtkTreeSelection *selection;
        selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_surf_spots_treeview);
        if (gtk_tree_selection_get_selected(selection, &model, &iter)){
            gtk_tree_model_get(model, &iter,    SPOTS_COL_SOURCE,       &source_old,
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

            gtk_combo_box_set_active    ((GtkComboBox*)     phoebe_load_spots_source_combobox,          source_old - 1);
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

        int result = gtk_dialog_run ((GtkDialog*)phoebe_load_spots_dialog);
        switch (result){
            case GTK_RESPONSE_OK:{

                /* source gets set separately because 1 is for primary, and 2 for secondary, while the
                   combo returns 0 for primary and 1 for secondary; and, we need it later */
                source_new = gtk_combo_box_get_active ((GtkComboBox*) phoebe_load_spots_source_combobox) + 1;
                char *source_str;

				if(source_new == 1)source_str = "Primary";
				else source_str = "Secondary";

                gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST,       FALSE,
                                                                SPOTS_COL_SOURCE,       source_new,
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

                /* source of the spot has been changed, so the spots_nos have to be changed as well */
                if(source_old != source_new){
                    PHOEBE_parameter *par;
                    int spots_no;

                    if (source_old == 1){
                        /* the spot USED to be on the primary, and now it's on the secondary */
                        par = phoebe_parameter_lookup("phoebe_spots_no1");
                        phoebe_parameter_get_value(par, &spots_no);
                        phoebe_parameter_set_value(par, spots_no - 1);
                        printf("Number of spots on the primary: %d\n", spots_no - 1);

                        par = phoebe_parameter_lookup("phoebe_spots_no2");
                        phoebe_parameter_get_value(par, &spots_no);
                        phoebe_parameter_set_value(par, spots_no + 1);
                        printf("Number of spots on the secondary: %d\n", spots_no + 1);
                    }
                    else{
                        /* the spot USED to be on the secondary, and now it's on the primary */
                        par = phoebe_parameter_lookup("phoebe_spots_no2");
                        phoebe_parameter_get_value(par, &spots_no);
                        phoebe_parameter_set_value(par, spots_no - 1);
                        printf("Number of spots on the secondary: %d\n", spots_no - 1);

                        par = phoebe_parameter_lookup("phoebe_spots_no1");
                        phoebe_parameter_get_value(par, &spots_no);
                        phoebe_parameter_set_value(par, spots_no + 1);
                        printf("Number of spots on the primary: %d\n", spots_no + 1);
                    }
                }
            }
            break;

            case GTK_RESPONSE_CANCEL:
            break;
        }
        gtk_widget_destroy (phoebe_load_spots_dialog);
    }
}


void
on_phoebe_para_surf_spots_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_surf_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)){
        int source;
        gtk_tree_model_get(model, &iter, SPOTS_COL_SOURCE, &source, -1);

        gtk_list_store_remove((GtkListStore*)model, &iter);

        PHOEBE_parameter *par;
        int spots_no;

        if (source == 1){
            /* the primary */
            par = phoebe_parameter_lookup("phoebe_spots_no1");
            phoebe_parameter_get_value(par, &spots_no);
            phoebe_parameter_set_value(par, spots_no - 1);
            printf("Number of spots on the primary: %d\n", spots_no - 1);
        }
        else{
            /* the secondary */
            par = phoebe_parameter_lookup("phoebe_spots_no2");
            phoebe_parameter_get_value(par, &spots_no);
            phoebe_parameter_set_value(par, spots_no - 1);
            printf("Number of spots on the secondary: %d\n", spots_no - 1);
        }
    }
}


void on_phoebe_para_surf_spots_adjust_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data)
{
    GtkTreeModel *model;
    GtkTreeIter   iter;
    int           active;

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_surf_spots_treeview);

    if(gtk_tree_model_get_iter_from_string(model, &iter, path)){
        g_object_get(renderer, "active", &active, NULL);

        if(active)
            gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
        else
            gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
    }
}


/* ******************************************************************** *
 *
 *                    phoebe_window menubar events
 *
 * ******************************************************************** */


gboolean
on_phoebe_window_delete_event          (GtkWidget *widget,
                                        GdkEvent  *event,
                                        gpointer   user_data)
{
    gtk_main_quit();
    return FALSE;
}


void
on_phoebe_file_new_menuitem_activate   (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_phoebe_file_open_menuitem_activate  (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_phoebe_file_save_menuitem_activate  (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_phoebe_file_saveas_menuitem_activate
                                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_phoebe_file_quit_menuitem_activate  (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    gtk_main_quit();
}


void
on_phoebe_settings_configuration_menuitem_activate
                                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_phoebe_help_about_menuitem_activate (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_window toolbar events
 *
 * ******************************************************************** */


void
on_phoebe_lc_plot_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 700, 550);
}


void
on_phoebe_rv_plot_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 700, 550);
}


void
on_phoebe_fiitting_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_fitting_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_fitting_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}


void
on_phoebe_scripter_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_settings_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_quit_toolbutton_clicked      (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
    gtk_main_quit();
}


void
on_phoebe_open_toolbutton_clicked      (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_save_toolbutton_clicked      (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_window data tab events
 *
 * ******************************************************************** */


void
on_phoebe_data_star_name_entry_changed
                                        (GtkEditable *editable,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_star_model_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_lcoptions_mag_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_lcoptions_mag_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_rvoptions_psepe_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_rvoptions_ssepe_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_options_bins_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_options_binsno_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_options_binsno_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_options_the_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_options_the_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_eph_dpdt events
 *
 * ******************************************************************** */


void
on_phoebe_para_eph_dpdt_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdt_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dpdtmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_eph_period events
 *
 * ******************************************************************** */


void
on_phoebe_para_eph_periodmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_periodmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_periodmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_periodmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_periodstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_periodstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_periodadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_period_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_period_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_eph_hjd0 events
 *
 * ******************************************************************** */


void
on_phoebe_para_eph_hjd0_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_hjd0min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_eph_dperdt events
 *
 * ******************************************************************** */


void
on_phoebe_para_eph_dperdt_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdt_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtmax_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_dperdtmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_eph_perr0 events
 *
 * ******************************************************************** */


void
on_phoebe_para_eph_perr0min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0max_spinbutton_change_value
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_perr0_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_eph_pshift events
 *
 * ******************************************************************** */


void
on_phoebe_para_eph_pshiftmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_pshiftmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_pshiftmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_pshiftmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_pshiftstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_pshiftstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_eph_pshiftadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_incl events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_incl_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_incl_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_incladjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_inclstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_inclstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_inclmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_inclmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_inclmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_inclmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_ecc events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_ecc_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_ecc_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_eccmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_vga events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_vgamin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vgamin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vgamax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vgamax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vgastep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vgastep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vgaadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vga_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_vga_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_rm events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_rm_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rm_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_rmmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_sma events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_smamin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_smamin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_smamax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_smamax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_smastep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_smastep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_smaadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_sma_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_sma_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_f1 events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_f1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_sys_f2 events
 *
 * ******************************************************************** */


void
on_phoebe_para_sys_f2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_sys_f2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_met2 events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_met2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2min_spinbutton_wrapped
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_met1 events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_met1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_met1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_pcsv events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_pcsv_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsv_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_pcsvmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_phsv events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_phsvmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsvmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsvmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsvmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsvstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsvstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsvadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsv_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_phsv_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_tavc events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_tavc_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavc_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavcmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_tavh events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_tavhmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavhmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavhmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavhmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavhstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavhstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavh_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavh_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_tavh_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_logg1 events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_logg1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_comp_logg2 events
 *
 * ******************************************************************** */


void
on_phoebe_para_comp_logg2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2min_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_comp_logg2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_surf_alb1 events
 *
 * ******************************************************************** */


void
on_phoebe_para_surf_alb1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_surf_alb2 events
 *
 * ******************************************************************** */


void
on_phoebe_para_surf_alb2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_alb2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_surf_gr1 events
 *
 * ******************************************************************** */


void
on_phoebe_para_surf_gr1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_surf_gr2 events
 *
 * ******************************************************************** */


void
on_phoebe_para_surf_gr2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_surf_gr2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_levels events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_levels_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_edit_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_primadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_secadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_primstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_primstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_secstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_levels_secstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_el3 events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_el3_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_el3_opacityadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_el3_opacitystep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_el3_opacitystep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_el3ajdust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_el3step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_el3step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_weighting events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_weighting_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_weighting_edit_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_atmospheres events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_atmospheres_prim_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_atmospheres_sec_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_atmospheres_grav_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_noise events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_noise_seed_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_noise_seed_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_noise_seedgen_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_noise_sigma_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_noise_sigma_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_noise_lcscatter_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_noise_lcscatter_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_options_reflections_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_options events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_options_decouple_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_options_reflections_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_lum_options_reflections_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_ld_bolcoefs events
 *
 * ******************************************************************** */


void
on_phoebe_para_ld_bolcoefs_secy_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_secy_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_primy_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_primy_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_secx_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_secx_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_primx_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_bolcoefs_primx_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_model_autoupdate_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_ld_model events
 *
 * ******************************************************************** */


void
on_phoebe_para_ld_model_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_model_tables_claret_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_model_tables_vanhamme_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_para_ld_lccoefs events
 *
 * ******************************************************************** */


void
on_phoebe_para_ld_lccoefs_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_lccoefs_secstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_lccoefs_secstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_lccoefs_primstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_lccoefs_primstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_lccoefs_secadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_para_ld_lccoefs_primadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}

/* ******************************************************************** *
 *
 *                    phoebe_window detach events
 *
 * ******************************************************************** */

void
on_phoebe_sidesheet_detach_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_sidesheet_vbox");
	GUI_widget *parent = gui_widget_lookup ("phoebe_sidesheet_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_SIDESHEET_IS_DETACHED, "PHOEBE - Data sheets", 300, 600);
}

void
on_phoebe_lc_plot_detach_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 700, 550);
}

void
on_phoebe_rv_plot_detach_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 700, 550);
}

void
on_phoebe_fitt_fitting_detach_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_fitting_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_fitting_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}

/* ******************************************************************** *
 *
 *                    phoebe_window plot events
 *
 * ******************************************************************** */

void
on_phoebe_lc_plot_options_obs_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
	GUI_widget *combobox = gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox");
	gtk_widget_set_sensitive (combobox->gtk, gtk_toggle_button_get_active(togglebutton));
	if(gtk_combo_box_get_active(GTK_COMBO_BOX(combobox->gtk))==-1) gtk_combo_box_set_active(GTK_COMBO_BOX(combobox->gtk),0);
}

void
on_phoebe_rv_plot_options_obs_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
	GUI_widget *combobox = gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox");
	gtk_widget_set_sensitive (combobox->gtk, gtk_toggle_button_get_active(togglebutton));
	if(gtk_combo_box_get_active(GTK_COMBO_BOX(combobox->gtk))==-1) gtk_combo_box_set_active(GTK_COMBO_BOX(combobox->gtk),0);
}
