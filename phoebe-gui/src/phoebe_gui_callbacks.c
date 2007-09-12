#include <stdlib.h>
#include <stdio.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_plotting.h"

void
on_phoebe_test_toolbutton_0_clicked (GtkToolButton   *toolbutton, gpointer user_data)
{
    gui_get_values_from_widgets();
}

void
on_phoebe_test_toolbutton_1_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
    gui_set_values_to_widgets();
}

/* ******************************************************************** *
 *
 *                    phoebe fitting tab events
 *
 * ******************************************************************** */

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback;
int accept_flag = 0;


void on_phoebe_fitt_calculate_button_clicked (GtkToolButton   *toolbutton, gpointer user_data)
{
	phoebe_minimizer_feedback = phoebe_minimizer_feedback_new();
	FILE *output = fopen("phoebe_out", "w");

	GtkTreeView *phoebe_fitt_mf_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_first_treeview")->gtk);
	GtkComboBox *phoebe_fitt_method_combobox = GTK_COMBO_BOX(gui_widget_lookup("phoebe_fitt_method_combobox")->gtk);
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_fitt_mf_treeview);
	GtkTreeIter iter;
	int index, count;

	int status = 0;

	if (gtk_combo_box_get_active(phoebe_fitt_method_combobox) == 0){
		status = phoebe_minimize_using_dc(output, phoebe_minimizer_feedback);
		printf("DC minimizer says: %s", phoebe_error(status));
	}

	if (gtk_combo_box_get_active(phoebe_fitt_method_combobox) == 1){
		status = phoebe_minimize_using_nms(0.1, 1, output, phoebe_minimizer_feedback);
		printf("NMS minimizer says: %s", phoebe_error(status));
	}

	gtk_list_store_clear(GTK_LIST_STORE(model));

	count = phoebe_minimizer_feedback->qualifiers->dim;
	for(index = 0; index < count; index++){
		gtk_list_store_append(GTK_LIST_STORE(model), &iter);
		gtk_list_store_set(GTK_LIST_STORE(model), &iter,
		MF_COL_QUALIFIER, phoebe_minimizer_feedback->qualifiers->val.strarray[index],
		MF_COL_INITVAL, phoebe_minimizer_feedback->initvals->val[index],
		MF_COL_NEWVAL, phoebe_minimizer_feedback->newvals->val[index],
		MF_COL_ERROR, phoebe_minimizer_feedback->ferrors->val[index], -1);
	}

	accept_flag = 1;

}


void on_phoebe_fitt_fitting_updateall_button_clicked (GtkToolButton   *toolbutton, gpointer user_data)
{
	int status;
	if (accept_flag){
		status = phoebe_minimizer_feedback_accept(phoebe_minimizer_feedback);
		accept_flag = 0;
	}
}


/* ******************************************************************** *
 *
 *                    phoebe_data_lc_treeview events
 *
 * ******************************************************************** */


void
on_phoebe_data_lc_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_data_lc_treeview_edit();
}


void
on_phoebe_data_lc_add_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_lc_treeview_add();
}


void
on_phoebe_data_lc_edit_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_lc_treeview_edit();
}


void
on_phoebe_data_lc_remove_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_lc_treeview_remove();
}


void on_phoebe_data_lc_active_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
{
    GtkTreeModel *model;
    GtkTreeIter iter;
    int active;

	GtkWidget *phoebe_data_lc_treeview = gui_widget_lookup("phoebe_data_lc_treeview")->gtk;
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
on_phoebe_data_rv_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_data_rv_treeview_edit();
}


void
on_phoebe_data_rv_add_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_data_rv_treeview_add();
}


void on_phoebe_load_rv_filechooserbutton_selection_changed (GtkFileChooserButton *filechooserbutton, gpointer user_data)
{
	set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
}


void
on_phoebe_data_rv_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_data_rv_treeview_edit();
}


void
on_phoebe_data_rv_remove_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_rv_treeview_remove();
}


void on_phoebe_data_rv_active_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
{
    GtkTreeModel *model;
    GtkTreeIter iter;
    int active;

	GtkWidget *phoebe_data_rv_treeview = gui_widget_lookup("phoebe_data_rv_treeview")->gtk;
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
on_phoebe_para_spots_treeview_cursor_changed (GtkTreeView *tree_view, gpointer user_data)
{
	GtkWidget *lat_label			  = gui_widget_lookup ("phoebe_para_spots_lat_label")->gtk;
	GtkWidget *lat_spinbutton         = gui_widget_lookup ("phoebe_para_spots_lat_spinbutton")->gtk;
	GtkWidget *latadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_latadjust_checkbutton")->gtk;
	GtkWidget *latstep_spinbutton     = gui_widget_lookup ("phoebe_para_spots_latstep_spinbutton")->gtk;
	GtkWidget *latmax_spinbutton      = gui_widget_lookup ("phoebe_para_spots_latmax_spinbutton")->gtk;
	GtkWidget *latmin_spinbutton      = gui_widget_lookup ("phoebe_para_spots_latmin_spinbutton")->gtk;
	GtkWidget *lon_label			  = gui_widget_lookup ("phoebe_para_spots_lon_label")->gtk;
	GtkWidget *lon_spinbutton         = gui_widget_lookup ("phoebe_para_spots_lon_spinbutton")->gtk;
	GtkWidget *lonadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_lonadjust_checkbutton")->gtk;
	GtkWidget *lonstep_spinbutton     = gui_widget_lookup ("phoebe_para_spots_lonstep_spinbutton")->gtk;
	GtkWidget *lonmax_spinbutton      = gui_widget_lookup ("phoebe_para_spots_lonmax_spinbutton")->gtk;
	GtkWidget *lonmin_spinbutton      = gui_widget_lookup ("phoebe_para_spots_lonmin_spinbutton")->gtk;
	GtkWidget *rad_label			  = gui_widget_lookup ("phoebe_para_spots_rad_label")->gtk;
	GtkWidget *rad_spinbutton         = gui_widget_lookup ("phoebe_para_spots_rad_spinbutton")->gtk;
	GtkWidget *radadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_radadjust_checkbutton")->gtk;
	GtkWidget *radstep_spinbutton     = gui_widget_lookup ("phoebe_para_spots_radstep_spinbutton")->gtk;
	GtkWidget *radmax_spinbutton      = gui_widget_lookup ("phoebe_para_spots_radmax_spinbutton")->gtk;
	GtkWidget *radmin_spinbutton      = gui_widget_lookup ("phoebe_para_spots_radmin_spinbutton")->gtk;
	GtkWidget *temp_label			  = gui_widget_lookup ("phoebe_para_spots_temp_label")->gtk;
	GtkWidget *temp_spinbutton        = gui_widget_lookup ("phoebe_para_spots_temp_spinbutton")->gtk;
	GtkWidget *tempadjust_checkbutton = gui_widget_lookup ("phoebe_para_spots_tempadjust_checkbutton")->gtk;
	GtkWidget *tempstep_spinbutton    = gui_widget_lookup ("phoebe_para_spots_tempstep_spinbutton")->gtk;
	GtkWidget *tempmax_spinbutton     = gui_widget_lookup ("phoebe_para_spots_tempmax_spinbutton")->gtk;
	GtkWidget *tempmin_spinbutton     = gui_widget_lookup ("phoebe_para_spots_tempmin_spinbutton")->gtk;

	double lat,  latstep,  latmin,  latmax;  bool latadjust;
	double lon,  lonstep,  lonmin,  lonmax;  bool lonadjust;
	double rad,  radstep,  radmin,  radmax;  bool radadjust;
	double temp, tempstep, tempmin, tempmax; bool tempadjust;

	char *source, *markup;

	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)){
		gtk_tree_model_get(model, &iter,SPOTS_COL_SOURCE_STR, &source,
										SPOTS_COL_LAT, &lat,
										SPOTS_COL_LATADJUST, &latadjust,
										SPOTS_COL_LATSTEP, &latstep,
										SPOTS_COL_LATMIN, &latmin,
										SPOTS_COL_LATMAX, &latmax,
										SPOTS_COL_LON, &lon,
										SPOTS_COL_LONADJUST, &lonadjust,
										SPOTS_COL_LONSTEP, &lonstep,
										SPOTS_COL_LONMIN, &lonmin,
										SPOTS_COL_LONMAX, &lonmax,
										SPOTS_COL_RAD, &rad,
										SPOTS_COL_RADADJUST, &radadjust,
										SPOTS_COL_RADSTEP, &radstep,
										SPOTS_COL_RADMIN, &radmin,
										SPOTS_COL_RADMAX, &radmax,
										SPOTS_COL_TEMP, &temp,
										SPOTS_COL_TEMPADJUST, &tempadjust,
										SPOTS_COL_TEMPSTEP, &tempstep,
										SPOTS_COL_TEMPMIN, &tempmin,
										SPOTS_COL_TEMPMAX, &tempmax, -1);

		markup = g_markup_printf_escaped ("<b>Latitude of spot %d on %s</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1, source);
		gtk_label_set_markup (GTK_LABEL (lat_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(latadjust_checkbutton), latadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lat_spinbutton), lat);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latstep_spinbutton), latstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latmin_spinbutton), latmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latmax_spinbutton), latmax);

		markup = g_markup_printf_escaped ("<b>Longitude of spot %d on %s</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1, source);
		gtk_label_set_markup (GTK_LABEL (lon_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lonadjust_checkbutton), lonadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lon_spinbutton), lon);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonstep_spinbutton), lonstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonmin_spinbutton), lonmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonmax_spinbutton), lonmax);

		markup = g_markup_printf_escaped ("<b>Radius of spot %d on %s</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1, source);
		gtk_label_set_markup (GTK_LABEL (rad_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radadjust_checkbutton), radadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rad_spinbutton), rad);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radstep_spinbutton), radstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radmin_spinbutton), radmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radmax_spinbutton), radmax);

		markup = g_markup_printf_escaped ("<b>Temperature of spot %d on %s</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1, source);
		gtk_label_set_markup (GTK_LABEL (temp_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tempadjust_checkbutton), tempadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(temp_spinbutton), temp);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(tempstep_spinbutton), tempstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(tempmin_spinbutton), tempmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(tempmax_spinbutton), tempmax);
    }
}


void
on_phoebe_para_spots_add_button_clicked (GtkButton *button, gpointer user_data)
{
/* The original coding:
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

    GtkTreeModel *model;
    GtkTreeIter iter;

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
                par = phoebe_parameter_lookup("phoebe_spots_no1");
                phoebe_parameter_get_value(par, &spots_no);
                phoebe_parameter_set_value(par, spots_no + 1);
                printf("Number of spots on the primary: %d\n", spots_no + 1);
            }
            else{
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
*/

	GtkTreeModel *model;
    GtkTreeIter iter;

    GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
	model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);

    gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST,       NO,
													SPOTS_COL_SOURCE,       1,
													SPOTS_COL_SOURCE_STR,   "primary star",
													SPOTS_COL_LAT,          0,
													SPOTS_COL_LATADJUST,    FALSE,
													SPOTS_COL_LON,          0,
													SPOTS_COL_LONADJUST,    FALSE,
													SPOTS_COL_RAD,          0,
													SPOTS_COL_RADADJUST,    FALSE,
													SPOTS_COL_TEMP,         0,
													SPOTS_COL_TEMPADJUST,   FALSE,-1);
	PHOEBE_parameter *par;
	int spots_no;

	par = phoebe_parameter_lookup("phoebe_spots_no1");
	phoebe_parameter_get_value(par, &spots_no);
	phoebe_parameter_set_value(par, spots_no + 1);
	printf("Number of spots on the primary: %d\n", spots_no + 1);
}


void
on_phoebe_para_spots_edit_button_clicked (GtkButton *button, gpointer user_data)
{
/* The original coding:
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

        double lat, latstep, latmin, latmax;
        double lon, lonstep, lonmin, lonmax;
        double rad, radstep, radmin, radmax;
        double temp, tempstep, tempmin, tempmax;
        bool latadjust, lonadjust, radadjust, tempadjust;
        int source_old, source_new;

        GtkTreeSelection *selection;
        selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
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

                if(source_old != source_new){
                    PHOEBE_parameter *par;
                    int spots_no;

                    if (source_old == 1){
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
*/

	GtkTreeModel *model;
    GtkTreeIter iter;

    GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
	model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);

    gtk_list_store_append((GtkListStore*)model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST,       FALSE,
													SPOTS_COL_SOURCE,       2,
													SPOTS_COL_SOURCE_STR,   "secondary star",
													SPOTS_COL_LAT,          0,
													SPOTS_COL_LATADJUST,    FALSE,
													SPOTS_COL_LON,          0,
													SPOTS_COL_LONADJUST,    FALSE,
													SPOTS_COL_RAD,          0,
													SPOTS_COL_RADADJUST,    FALSE,
													SPOTS_COL_TEMP,         0,
													SPOTS_COL_TEMPADJUST,   FALSE,-1);
	PHOEBE_parameter *par;
	int spots_no;

	par = phoebe_parameter_lookup("phoebe_spots_no2");
	phoebe_parameter_get_value(par, &spots_no);
	phoebe_parameter_set_value(par, spots_no + 1);
	printf("Number of spots on the secondary: %d\n", spots_no + 1);

}


void
on_phoebe_para_spots_remove_button_clicked (GtkButton *button, gpointer user_data)
{
    GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
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


void on_phoebe_para_spots_adjust_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
{
	GtkWidget *latadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_latadjust_checkbutton")->gtk;
	GtkWidget *lonadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_lonadjust_checkbutton")->gtk;
	GtkWidget *radadjust_checkbutton  = gui_widget_lookup ("phoebe_para_spots_radadjust_checkbutton")->gtk;
	GtkWidget *tempadjust_checkbutton = gui_widget_lookup ("phoebe_para_spots_tempadjust_checkbutton")->gtk;

    GtkTreeModel *model;
    GtkTreeIter   iter;
    int           active;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);

    if(gtk_tree_model_get_iter_from_string(model, &iter, path)){
        g_object_get(renderer, "active", &active, NULL);

        if(active){
            gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(latadjust_checkbutton), FALSE);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lonadjust_checkbutton), FALSE);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radadjust_checkbutton), FALSE);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tempadjust_checkbutton), FALSE);
        }
        else{
            gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(latadjust_checkbutton), TRUE);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lonadjust_checkbutton), TRUE);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radadjust_checkbutton), TRUE);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tempadjust_checkbutton), TRUE);
        }
    }
}


void on_phoebe_para_spots_latadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATADJUST, gtk_toggle_button_get_active(togglebutton), -1);
}


void on_phoebe_para_spots_lat_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LAT, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_latstep_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATSTEP, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_latmin_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATMIN, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_latmax_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATMAX, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_lonadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONADJUST, gtk_toggle_button_get_active(togglebutton), -1);
}


void on_phoebe_para_spots_lon_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LON, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_lonstep_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONSTEP, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_lonmin_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONMIN, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_lonmax_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONMAX, gtk_spin_button_get_value(spinbutton), -1);
}


void on_phoebe_para_spots_radadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADADJUST, gtk_toggle_button_get_active(togglebutton), -1);
}


void on_phoebe_para_spots_rad_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RAD, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_radstep_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADSTEP, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_radmin_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADMIN, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_radmax_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADMAX, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_tempadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPADJUST, gtk_toggle_button_get_active(togglebutton), -1);
}

void on_phoebe_para_spots_temp_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMP, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_tempstep_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPSTEP, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_tempmin_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPMIN, gtk_spin_button_get_value(spinbutton), -1);
}

void on_phoebe_para_spots_tempmax_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPMAX, gtk_spin_button_get_value(spinbutton), -1);
}


/* ******************************************************************** *
 *
 *                    phoebe_window menubar events
 *
 * ******************************************************************** */


gboolean on_phoebe_window_delete_event (GtkWidget *widget, GdkEvent *event, gpointer user_data)
{
    gtk_main_quit();
    return FALSE;
}


void on_phoebe_file_new_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{

}


void on_phoebe_file_open_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = gui_open_parameter_file ();

	if( status == SUCCESS ){
		printf("Parameter file successfuly open.\n");
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_error (status));
}


void on_phoebe_file_save_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = gui_save_parameter_file ();

	if( status == SUCCESS )
		printf("Parameter file successfuly saved.\n");
	else
		printf ("%s", phoebe_error (status));
}


void on_phoebe_file_saveas_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = gui_save_parameter_file ();

	if( status == SUCCESS )
		printf("Parameter file successfuly saved.\n");
	else
		printf ("%s", phoebe_error (status));
}


void on_phoebe_file_quit_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
    gtk_main_quit();
}


void on_phoebe_settings_configuration_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gchar     *glade_xml_file					= g_build_filename     	(PHOEBE_GLADE_XML_DIR, "phoebe_settings.glade", NULL);
	gchar     *glade_pixmap_file				= g_build_filename     	(PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_settings_xml				= glade_xml_new			(glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_settings_dialog			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_dialog");
	GtkWidget *basedir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_basedir_filechooserbutton");
	GtkWidget *srcdir_filechooserbutton			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_srcdir_filechooserbutton");
	GtkWidget *defaultsdir_filechooserbutton	= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_defaultsdir_filechooserbutton");
	GtkWidget *workingdir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_workingdir_filechooserbutton");
	GtkWidget *datadir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_datadir_filechooserbutton");

	GtkWidget *vh_checkbutton					= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_vh_checkbutton");
	GtkWidget *vh_lddir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_vh_lddir_filechooserbutton");

	GtkWidget *kurucz_checkbutton				= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_kurucz_checkbutton");
	GtkWidget *kurucz_filechooserbutton			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_kurucz_filechooserbutton");

	gchar 		*dir;
	gboolean	toggle;
	gint 		result;

	g_object_unref (phoebe_settings_xml);

	phoebe_config_entry_get ("PHOEBE_BASE_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)basedir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_SOURCE_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)srcdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_DEFAULTS_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)defaultsdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)workingdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_DATA_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)datadir_filechooserbutton, dir);

	g_signal_connect(G_OBJECT(vh_checkbutton), "toggled", G_CALLBACK(on_phoebe_settings_checkbutton_toggled), (gpointer)vh_lddir_filechooserbutton);
	g_signal_connect(G_OBJECT(kurucz_checkbutton), "toggled", G_CALLBACK(on_phoebe_settings_checkbutton_toggled), (gpointer)kurucz_filechooserbutton);

	gtk_window_set_icon (GTK_WINDOW (phoebe_settings_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_settings_dialog), "PHOEBE - Settings");

	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
	if (toggle){
			phoebe_config_entry_get ("PHOEBE_LD_DIR", &dir);
			gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(vh_checkbutton), TRUE);
			gtk_widget_set_sensitive (vh_lddir_filechooserbutton, TRUE);
			gtk_file_chooser_set_filename((GtkFileChooser*)vh_lddir_filechooserbutton, dir);
	}

	phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
	if (toggle){
			phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &dir);
			gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton), TRUE);
			gtk_widget_set_sensitive (kurucz_filechooserbutton, TRUE);
			gtk_file_chooser_set_filename((GtkFileChooser*)kurucz_filechooserbutton, dir);
	}

	result = gtk_dialog_run ((GtkDialog*)phoebe_settings_dialog);
	switch (result){
		case GTK_RESPONSE_OK:{
			phoebe_config_entry_set ("PHOEBE_BASE_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)basedir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_SOURCE_DIR",	gtk_file_chooser_get_filename ((GtkFileChooser*)srcdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", gtk_file_chooser_get_filename ((GtkFileChooser*)defaultsdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)workingdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DATA_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)datadir_filechooserbutton));

			phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(vh_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_LD_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)vh_lddir_filechooserbutton));
			}
			else if (toggle)
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	FALSE);

			phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)kurucz_filechooserbutton));
			}
			else if (toggle)
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	FALSE);
		}
        break;

		case GTK_RESPONSE_YES:{
			phoebe_config_entry_set ("PHOEBE_BASE_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)basedir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_SOURCE_DIR",	gtk_file_chooser_get_filename ((GtkFileChooser*)srcdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", gtk_file_chooser_get_filename ((GtkFileChooser*)defaultsdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)workingdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DATA_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)datadir_filechooserbutton));

			phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(vh_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_LD_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)vh_lddir_filechooserbutton));
			}
			else if (toggle)
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	FALSE);

			phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)kurucz_filechooserbutton));
			}
			else if (toggle)
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	FALSE);

			phoebe_config_save (PHOEBE_CONFIG);
		}
		break;

		case GTK_RESPONSE_CANCEL:
		break;
	}

	gtk_widget_destroy (phoebe_settings_dialog);
}


void on_phoebe_help_about_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{

}


/* ******************************************************************** *
 *
 *                    phoebe_window toolbar events
 *
 * ******************************************************************** */


void on_phoebe_lc_plot_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 750, 550);
}


void on_phoebe_rv_plot_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 750, 550);
}


void on_phoebe_fiitting_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_fitting_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_fitting_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}


void on_phoebe_scripter_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{

}


void on_phoebe_settings_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	gchar     *glade_xml_file					= g_build_filename     	(PHOEBE_GLADE_XML_DIR, "phoebe_settings.glade", NULL);
	gchar     *glade_pixmap_file				= g_build_filename     	(PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_settings_xml				= glade_xml_new			(glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_settings_dialog			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_dialog");
	GtkWidget *basedir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_basedir_filechooserbutton");
	GtkWidget *srcdir_filechooserbutton			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_srcdir_filechooserbutton");
	GtkWidget *defaultsdir_filechooserbutton	= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_defaultsdir_filechooserbutton");
	GtkWidget *workingdir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_workingdir_filechooserbutton");
	GtkWidget *datadir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_datadir_filechooserbutton");

	GtkWidget *vh_checkbutton					= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_vh_checkbutton");
	GtkWidget *vh_lddir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_vh_lddir_filechooserbutton");

	GtkWidget *kurucz_checkbutton				= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_kurucz_checkbutton");
	GtkWidget *kurucz_filechooserbutton			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_kurucz_filechooserbutton");

	gchar 		*dir;
	gboolean	toggle;
	gint 		result;

	g_object_unref (phoebe_settings_xml);

	phoebe_config_entry_get ("PHOEBE_BASE_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)basedir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_SOURCE_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)srcdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_DEFAULTS_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)defaultsdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)workingdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_DATA_DIR", &dir);
	gtk_file_chooser_set_filename((GtkFileChooser*)datadir_filechooserbutton, dir);

	g_signal_connect(G_OBJECT(vh_checkbutton), "toggled", G_CALLBACK(on_phoebe_settings_checkbutton_toggled), (gpointer)vh_lddir_filechooserbutton);
	g_signal_connect(G_OBJECT(kurucz_checkbutton), "toggled", G_CALLBACK(on_phoebe_settings_checkbutton_toggled), (gpointer)kurucz_filechooserbutton);

	gtk_window_set_icon (GTK_WINDOW (phoebe_settings_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_settings_dialog), "PHOEBE - Settings");

	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
	if (toggle){
			phoebe_config_entry_get ("PHOEBE_LD_DIR", &dir);
			gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(vh_checkbutton), TRUE);
			gtk_widget_set_sensitive (vh_lddir_filechooserbutton, TRUE);
			gtk_file_chooser_set_filename((GtkFileChooser*)vh_lddir_filechooserbutton, dir);
	}

	phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
	if (toggle){
			phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &dir);
			gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton), TRUE);
			gtk_widget_set_sensitive (kurucz_filechooserbutton, TRUE);
			gtk_file_chooser_set_filename((GtkFileChooser*)kurucz_filechooserbutton, dir);
	}

	result = gtk_dialog_run ((GtkDialog*)phoebe_settings_dialog);
	switch (result){
		case GTK_RESPONSE_OK:{
			phoebe_config_entry_set ("PHOEBE_BASE_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)basedir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_SOURCE_DIR",	gtk_file_chooser_get_filename ((GtkFileChooser*)srcdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", gtk_file_chooser_get_filename ((GtkFileChooser*)defaultsdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)workingdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DATA_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)datadir_filechooserbutton));

			phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(vh_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_LD_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)vh_lddir_filechooserbutton));
			}
			else if (!toggle)
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	FALSE);

			phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)kurucz_filechooserbutton));
			}
			else if (!toggle)
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	FALSE);
		}
        break;

		case GTK_RESPONSE_YES:{
			phoebe_config_entry_set ("PHOEBE_BASE_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)basedir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_SOURCE_DIR",	gtk_file_chooser_get_filename ((GtkFileChooser*)srcdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", gtk_file_chooser_get_filename ((GtkFileChooser*)defaultsdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)workingdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DATA_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser*)datadir_filechooserbutton));

			phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(vh_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_LD_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)vh_lddir_filechooserbutton));
			}
			else if (toggle)
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH",	FALSE);

			phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton))){
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	TRUE);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR",		gtk_file_chooser_get_filename ((GtkFileChooser*)kurucz_filechooserbutton));
			}
			else if (toggle)
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH",	FALSE);

			phoebe_config_save (PHOEBE_CONFIG);
		}
		break;

		case GTK_RESPONSE_CANCEL:
		break;
	}

	gtk_widget_destroy (phoebe_settings_dialog);
}


void on_phoebe_settings_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkWidget *filechooserbutton = GTK_WIDGET(user_data);
	gtk_widget_set_sensitive (filechooserbutton, gtk_toggle_button_get_active(togglebutton));
}


void on_phoebe_quit_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
    gtk_main_quit();
}


void on_phoebe_open_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status = gui_open_parameter_file ();

	if( status == SUCCESS ){
		printf("Parameter file successfuly open.\n");
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_error (status));
}

void on_phoebe_save_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status = gui_save_parameter_file ();

	if( status == SUCCESS )
		printf("Parameter file successfuly saved.\n");
	else
		printf ("%s", phoebe_error (status));
}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_levels events
 *
 * ******************************************************************** */


void
on_phoebe_para_lum_levels_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{

}


void
on_phoebe_para_lum_levels_edit_button_clicked (GtkButton *button, gpointer user_data)
{
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
}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_el3 events
 *
 * ******************************************************************** */


void on_phoebe_para_lum_el3_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{

}


void on_phoebe_para_lum_el3_edit_button_clicked (GtkButton *button, gpointer user_data)
{
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
}

/* ******************************************************************** *
 *
 *                    phoebe_para_lum_weighting events
 *
 * ******************************************************************** */


void on_phoebe_para_lum_weighting_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{

}


void on_phoebe_para_lum_weighting_edit_button_clicked (GtkButton *button, gpointer user_data)
{
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
}


/* ******************************************************************** *
 *
 *                    phoebe_window detach events
 *
 * ******************************************************************** */


void on_phoebe_sidesheet_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_sidesheet_vbox");
	GUI_widget *parent = gui_widget_lookup ("phoebe_sidesheet_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_SIDESHEET_IS_DETACHED, "PHOEBE - Data sheets", 300, 600);
}


void on_phoebe_lc_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 750, 550);
}


void on_phoebe_rv_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 750, 550);
}


void on_phoebe_fitt_fitting_detach_button_clicked (GtkButton *button, gpointer user_data)
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


void on_phoebe_lc_plot_options_obs_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GUI_widget *combobox = gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox");
	gtk_widget_set_sensitive (combobox->gtk, gtk_toggle_button_get_active(togglebutton));
	if(gtk_combo_box_get_active(GTK_COMBO_BOX(combobox->gtk))==-1) gtk_combo_box_set_active(GTK_COMBO_BOX(combobox->gtk),0);
}


void on_phoebe_rv_plot_options_obs_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GUI_widget *combobox = gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox");
	gtk_widget_set_sensitive (combobox->gtk, gtk_toggle_button_get_active(togglebutton));
	if(gtk_combo_box_get_active(GTK_COMBO_BOX(combobox->gtk))==-1) gtk_combo_box_set_active(GTK_COMBO_BOX(combobox->gtk),0);
}


void on_phoebe_lc_plot_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	int lcno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
	phoebe_parameter_get_value(par, &lcno);

	if(lcno > 0){
		gui_get_values_from_widgets();
		gui_plot_lc_using_gnuplot();
	}

	else printf("Nothing to plot...\n");
}
