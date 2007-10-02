#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_plotting.h"

void on_phoebe_para_tba_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	char *widget_name = (char*)gtk_widget_get_name(GTK_WIDGET(togglebutton));
	gui_get_value_from_widget(gui_widget_lookup(widget_name));

	gui_fill_sidesheet_fit_treeview ();
	gui_fill_fitt_mf_treeview();
}

void on_phoebe_data_star_name_entry_changed (GtkEditable *editable, gpointer user_data)
{
	GtkWidget *phoebe_window = gui_widget_lookup("phoebe_window")->gtk;
	GtkWidget *star_name_entry = gui_widget_lookup("phoebe_data_star_name_entry")->gtk;
	char *star_name = (char*)gtk_entry_get_text(GTK_ENTRY(star_name_entry));

	gtk_window_set_title (GTK_WINDOW(phoebe_window), star_name);
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

	GtkTreeView 	*phoebe_fitt_mf_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_first_treeview")->gtk);
	GtkComboBox 	*phoebe_fitt_method_combobox = GTK_COMBO_BOX(gui_widget_lookup("phoebe_fitt_method_combobox")->gtk);
	GtkLabel		*phoebe_fitt_feedback_label = GTK_LABEL(gui_widget_lookup("phoebe_fitt_feedback_label")->gtk);
	GtkSpinButton 	*phoebe_fitt_nms_iters_spinbutton = GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_fitt_nms_iters_spinbutton")->gtk);
	GtkSpinButton 	*phoebe_fitt_nms_accuracy_spinbutton = GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_fitt_nms_accuracy_spinbutton")->gtk);
	GtkTreeModel 	*model = gtk_tree_view_get_model(phoebe_fitt_mf_treeview);
	GtkTreeIter iter;
	int index, count;
	char status_message[255] = "Minimizer feedback";

	int status = 0;

	status = gui_get_values_from_widgets();

	if (gtk_combo_box_get_active(phoebe_fitt_method_combobox) == 0){
		status = phoebe_minimize_using_dc(stdout, phoebe_minimizer_feedback);
		printf("DC minimizer says: %s", phoebe_error(status));
	}

	if (gtk_combo_box_get_active(phoebe_fitt_method_combobox) == 1){
		status = phoebe_minimize_using_nms(gtk_spin_button_get_value_as_int(phoebe_fitt_nms_iters_spinbutton), gtk_spin_button_get_value(phoebe_fitt_nms_accuracy_spinbutton), stdout, phoebe_minimizer_feedback);
		printf("NMS minimizer says: %s", phoebe_error(status));
	}

	if (status == SUCCESS){
		sprintf(status_message, "%s: done %d iterations in %f seconds; cost function value: %f", (gtk_combo_box_get_active(phoebe_fitt_method_combobox)? "Nelder-Mead Simplex":"Differential corrections"), phoebe_minimizer_feedback->iters, phoebe_minimizer_feedback->cputime, phoebe_minimizer_feedback->cfval);
		gtk_label_set_text(phoebe_fitt_feedback_label, status_message);

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
	else{
		sprintf(status_message, "%s: %s", (gtk_combo_box_get_active(phoebe_fitt_method_combobox)? "Nelder-Mead Simplex":"Differential corrections"), phoebe_error(status));
		gtk_label_set_text(phoebe_fitt_feedback_label, status_message);
	}
}


void on_phoebe_fitt_updateall_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status;
	if (accept_flag){
		status = phoebe_minimizer_feedback_accept(phoebe_minimizer_feedback);
		status = gui_set_values_to_widgets();
		gui_fill_sidesheet_fit_treeview ();
		gui_fill_fitt_mf_treeview();
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
	gui_set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
}


void on_phoebe_data_lc_model_row_changed (GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data)
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
	gui_set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
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


void on_phoebe_data_rv_model_row_changed (GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data)
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


/* ******************************************************************** *
 *
 *               phoebe_para_surf_spots_treeview events
 *
 * ******************************************************************** */


void on_phoebe_para_spots_treeview_cursor_changed (GtkTreeView *tree_view, gpointer user_data)
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

		markup = g_markup_printf_escaped ("<b>Latitude of spot %d</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1);
		gtk_label_set_markup (GTK_LABEL (lat_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(latadjust_checkbutton), latadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lat_spinbutton), lat);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latstep_spinbutton), latstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latmin_spinbutton), latmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latmax_spinbutton), latmax);

		markup = g_markup_printf_escaped ("<b>Longitude of spot %d</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1);
		gtk_label_set_markup (GTK_LABEL (lon_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lonadjust_checkbutton), lonadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lon_spinbutton), lon);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonstep_spinbutton), lonstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonmin_spinbutton), lonmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonmax_spinbutton), lonmax);

		markup = g_markup_printf_escaped ("<b>Radius of spot %d</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1);
		gtk_label_set_markup (GTK_LABEL (rad_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radadjust_checkbutton), radadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rad_spinbutton), rad);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radstep_spinbutton), radstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radmin_spinbutton), radmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radmax_spinbutton), radmax);

		markup = g_markup_printf_escaped ("<b>Temperature of spot %d</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1);
		gtk_label_set_markup (GTK_LABEL (temp_label), markup);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tempadjust_checkbutton), tempadjust);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(temp_spinbutton), temp);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(tempstep_spinbutton), tempstep);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(tempmin_spinbutton), tempmin);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(tempmax_spinbutton), tempmax);
    }
}


void on_phoebe_para_spots_add_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_spots_add();
}


void on_phoebe_para_spots_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_spots_edit();
}


void on_phoebe_para_spots_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_spots_edit();
}


void on_phoebe_para_spots_remove_button_clicked (GtkButton *button, gpointer user_data)
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

		par = phoebe_parameter_lookup("phoebe_spots_no");
		phoebe_parameter_get_value(par, &spots_no);
		phoebe_parameter_set_value(par, spots_no - 1);
		printf("Number of spots: %d\n", spots_no - 1);
    }
}


void on_phoebe_para_spots_active_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
{
	GtkTreeModel *model;
    GtkTreeIter   iter;
    int           active;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);

    if(gtk_tree_model_get_iter_from_string(model, &iter, path)){
        g_object_get(renderer, "active", &active, NULL);

        if(active)
            gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ACTIVE, FALSE, -1);
        else
            gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ACTIVE, TRUE, -1);
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
    	gtk_tree_selection_select_iter (gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview), &iter);
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
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
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
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
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
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
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
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
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
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_error (status));
}


void on_phoebe_file_save_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = 0;

	status = gui_get_values_from_widgets();
	status = gui_save_parameter_file ();

	if( status != SUCCESS )
		printf ("%s", phoebe_error (status));
}


void on_phoebe_file_saveas_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = 0;

	status = gui_get_values_from_widgets();
	status = gui_save_parameter_file ();

	if( status != SUCCESS )
		printf ("%s", phoebe_error (status));
}


void on_phoebe_file_quit_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
    gtk_main_quit();
}


void on_phoebe_settings_configuration_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gui_show_configuration_dialog();
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

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 726, 522);
}


void on_phoebe_rv_plot_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 726, 522);
}


void on_phoebe_fitting_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}


void on_phoebe_scripter_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{

}


void on_phoebe_settings_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	gui_show_configuration_dialog();
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
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_error (status));
}

void on_phoebe_save_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status = 0;

	status = gui_get_values_from_widgets();
	status = gui_save_parameter_file ();

	if( status != SUCCESS )
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
	gui_para_lum_levels_edit();
}


void
on_phoebe_para_lum_levels_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lum_levels_edit();
}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_el3 events
 *
 * ******************************************************************** */


void on_phoebe_para_lum_el3_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_para_lum_el3_edit();
}


void on_phoebe_para_lum_el3_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lum_el3_edit();
}

/* ******************************************************************** *
 *
 *                    phoebe_para_lum_weighting events
 *
 * ******************************************************************** */


void on_phoebe_para_lum_weighting_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_fitt_levelweight_edit();
}

void on_phoebe_para_lum_weighting_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_fitt_levelweight_edit();
}


/* ******************************************************************** *
 *
 *              	phoebe_para_lc_coefficents_treeview events
 *
 * ******************************************************************** */


void on_phoebe_para_ld_lccoefs_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_para_lc_coefficents_edit();
}

void on_phoebe_para_ld_lccoefs_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lc_coefficents_edit();
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

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_SIDESHEET_IS_DETACHED, "PHOEBE - Data sheets", 300, 600);
}


void on_phoebe_lc_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 726, 522);
}


void on_phoebe_rv_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 726, 522);
}


void on_phoebe_fitt_fitting_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}

/* ******************************************************************** *
 *
 *                    phoebe_window plot events
 *
 * ******************************************************************** */


void on_phoebe_rv_plot_options_x_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GUI_widget *rv_xstart_label = gui_widget_lookup("phoebe_rv_plot_options_phstart_label");
	GUI_widget *rv_xend_label = gui_widget_lookup("phoebe_rv_plot_options_phend_label");

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0){
		/* Phase */
		gtk_label_set_text(GTK_LABEL(rv_xstart_label->gtk), "Phase start:");
		gtk_label_set_text(GTK_LABEL(rv_xend_label->gtk), "Phase end:");
	}
	else{
		/* Time */
		gtk_label_set_text(GTK_LABEL(rv_xstart_label->gtk), "Time start: ");
		gtk_label_set_text(GTK_LABEL(rv_xend_label->gtk), "Time end: ");
	}
}


void on_phoebe_lc_plot_options_x_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GUI_widget *lc_xstart_label = gui_widget_lookup("phoebe_lc_plot_options_phstart_label");
	GUI_widget *lc_xend_label = gui_widget_lookup("phoebe_lc_plot_options_phend_label");

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0){
		/* Phase */
		gtk_label_set_text(GTK_LABEL(lc_xstart_label->gtk), "Phase start:");
		gtk_label_set_text(GTK_LABEL(lc_xend_label->gtk), "Phase end:");
	}
	else{
		/* Time */
		gtk_label_set_text(GTK_LABEL(lc_xstart_label->gtk), "Time start: ");
		gtk_label_set_text(GTK_LABEL(lc_xend_label->gtk), "Time end: ");
	}
}


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


void on_phoebe_rv_plot_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	int rvno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
	phoebe_parameter_get_value(par, &rvno);

	if(rvno > 0){
		gui_get_values_from_widgets();
		gui_plot_rv_using_gnuplot();
	}

	else printf("Nothing to plot...\n");
}

void on_phoebe_star_shape_plot_button_clicked (GtkButton *button, gpointer user_data)
{
		//gui_get_values_from_widgets();
		gui_plot_eb_using_gnuplot();
}
