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
#include "phoebe_gui_error_handling.h"

gchar   *PHOEBE_FILENAME = NULL;
gboolean PHOEBE_FILEFLAG = FALSE;

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
	char title[255];

	if (strlen(star_name) > 0)
		sprintf(title, "PHOEBE - %s", star_name);
	else
		sprintf(title, "PHOEBE");

	gtk_window_set_title (GTK_WINDOW(phoebe_window), title);
}


void on_phoebe_data_lc_seedgen_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int seed;
	GtkWidget *seed_spin_button = gui_widget_lookup("phoebe_data_lc_seed_spinbutton")->gtk;

	srand (time (0));
	seed = (int) (100000001.0 + (double) rand () / RAND_MAX * 100000000.0);
	gtk_spin_button_set_value (GTK_SPIN_BUTTON (seed_spin_button), seed);
}


/* ******************************************************************** *
 *
 *                    phoebe fitting tab events
 *
 * ******************************************************************** */

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback;
int accept_flag = 0;

void on_phoebe_fitt_calculate_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	phoebe_minimizer_feedback = phoebe_minimizer_feedback_new();

	GtkTreeView 	*phoebe_fitt_mf_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_first_treeview")->gtk);
	GtkTreeView		*phoebe_fitt_second_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_second_treeview")->gtk);
	GtkComboBox 	*phoebe_fitt_method_combobox = GTK_COMBO_BOX(gui_widget_lookup("phoebe_fitt_method_combobox")->gtk);
	GtkLabel		*phoebe_fitt_feedback_label = GTK_LABEL(gui_widget_lookup("phoebe_fitt_feedback_label")->gtk);
	GtkSpinButton 	*phoebe_fitt_nms_iters_spinbutton = GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_fitt_nms_iters_spinbutton")->gtk);
	GtkSpinButton 	*phoebe_fitt_nms_accuracy_spinbutton = GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_fitt_nms_accuracy_spinbutton")->gtk);
	GtkTreeModel 	*model;
	GtkTreeIter iter;
	int index, count, state;
	char *id;
	char status_message[255] = "Minimizer feedback";
	PHOEBE_curve *curve;

	int status = 0;

	status = gui_get_values_from_widgets();

	if (gtk_combo_box_get_active(phoebe_fitt_method_combobox) == 0){
		status = phoebe_minimize_using_dc(stdout, phoebe_minimizer_feedback);
		phoebe_gui_debug("DC minimizer says: %s", phoebe_error(status));
	}

	if (gtk_combo_box_get_active(phoebe_fitt_method_combobox) == 1){
		status = phoebe_minimize_using_nms(gtk_spin_button_get_value_as_int(phoebe_fitt_nms_iters_spinbutton), gtk_spin_button_get_value(phoebe_fitt_nms_accuracy_spinbutton), stdout, phoebe_minimizer_feedback);
		phoebe_gui_debug("NMS minimizer says: %s", phoebe_error(status));
	}

	if (status == SUCCESS){
		sprintf(status_message, "%s: done %d iterations in %f seconds; cost function value: %f", (gtk_combo_box_get_active(phoebe_fitt_method_combobox)? "Nelder-Mead Simplex":"Differential corrections"), phoebe_minimizer_feedback->iters, phoebe_minimizer_feedback->cputime, phoebe_minimizer_feedback->cfval);
		gtk_label_set_text(phoebe_fitt_feedback_label, status_message);

		model = gtk_tree_view_get_model(phoebe_fitt_mf_treeview);
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

		model = gtk_tree_view_get_model(phoebe_fitt_second_treeview);
		gtk_list_store_clear(GTK_LIST_STORE(model));

		phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lcno"), &count);
		for(index = 0; index < count; index++){
			curve = phoebe_curve_new_from_pars(PHOEBE_CURVE_LC, index);
			phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lc_id"), index, &id);
			gtk_list_store_append(GTK_LIST_STORE(model), &iter);
			gtk_list_store_set(GTK_LIST_STORE(model), &iter,
			CURVE_COL_NAME, id,
			CURVE_COL_NPOINTS, curve->indep->dim,
			CURVE_COL_NEWCHI2, phoebe_minimizer_feedback->chi2s->val[index], -1);
		}

		phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_rvno"), &count);
		for(index = 0; index < count; index++){
			curve = phoebe_curve_new_from_pars(PHOEBE_CURVE_RV, index);
			phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_rv_id"), index, &id);
			gtk_list_store_append(GTK_LIST_STORE(model), &iter);
			gtk_list_store_set(GTK_LIST_STORE(model), &iter,
			CURVE_COL_NAME, id,
			CURVE_COL_NPOINTS, curve->indep->dim,
			CURVE_COL_NEWCHI2, phoebe_minimizer_feedback->chi2s->val[index], -1);
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


void on_phoebe_fitt_method_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GUI_widget *phoebe_fitt_dc_frame = gui_widget_lookup("phoebe_fitt_dc_frame");
	GUI_widget *phoebe_fitt_nms_frame = gui_widget_lookup("phoebe_fitt_nms_frame");

	switch(gtk_combo_box_get_active(widget)){
		case 0:
			/* DC */
			gtk_widget_hide_all(phoebe_fitt_nms_frame->gtk);
			gtk_widget_show_all(phoebe_fitt_dc_frame->gtk);
		break;
		case 1:
			/* NMS */
			gtk_widget_hide_all(phoebe_fitt_dc_frame->gtk);
			gtk_widget_show_all(phoebe_fitt_nms_frame->gtk);
		break;
	}
}


void on_phoebe_fitt_fitting_corrmat_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_cormat.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_cormat_dialog_xml      		= glade_xml_new        (glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_cormat_dialog        			= glade_xml_get_widget (phoebe_cormat_dialog_xml, "phoebe_cormat_dialog");

	GtkWidget *phoebe_cormat_dialog_textview		= glade_xml_get_widget (phoebe_cormat_dialog_xml, "phoebe_cormat_dialog_textview");

	GtkTextBuffer *cormat_buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (phoebe_cormat_dialog_textview));
	GtkTextIter iter;
	int rows, cols;
	char cormat_string[255];

	g_object_unref (phoebe_cormat_dialog_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_cormat_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_cormat_dialog), "PHOEBE - Correlation matrix");

	gtk_dialog_run(GTK_DIALOG(phoebe_cormat_dialog));

	if(phoebe_minimizer_feedback){
		gtk_text_buffer_get_iter_at_line (cormat_buffer, &iter, 0);
		for(rows = 0; rows < phoebe_minimizer_feedback->cormat->rows; rows++){
			for(cols = 0; cols < phoebe_minimizer_feedback->cormat->cols; cols++){
				sprintf(cormat_string, "%lf\t", phoebe_minimizer_feedback->cormat->val[rows][cols]);
				gtk_text_buffer_insert (cormat_buffer, &iter, cormat_string, -1);
			}
			sprintf(cormat_string, "\n");
			gtk_text_buffer_insert (cormat_buffer, &iter, cormat_string, -1);
		}
	}
		
	gtk_widget_destroy(GTK_WIDGET(phoebe_cormat_dialog));
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
    if(gui_warning("Quit PHOEBE?", "By quitting Phoebe all unsaved data will be lost. Are you sure you want to quit?") == 1)
    	gtk_main_quit();
    return TRUE;
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

	//status = gui_get_values_from_widgets();

	//if(PHOEBE_FILEFLAG)
		//status = phoebe_save_parameter_file(PHOEBE_FILENAME);
	//else
		//status = gui_save_parameter_file ();

	printf("In on_phoebe_file_save_menuitem_activate\n");

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
    if(gui_warning("Quit PHOEBE?", "By quitting Phoebe all unsaved data will be lost. Are you sure you want to quit?") == 1)
    	gtk_main_quit();
}


void on_phoebe_settings_configuration_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gui_show_configuration_dialog();
}


void on_phoebe_help_about_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_about.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_about_xml                   = glade_xml_new        (glade_xml_file, NULL, NULL);
	GtkWidget *phoebe_about_dialog                = glade_xml_get_widget (phoebe_about_xml, "phoebe_about_dialog");

	gtk_window_set_icon (GTK_WINDOW (phoebe_about_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));

	gint result = gtk_dialog_run ((GtkDialog*)phoebe_about_dialog);
	switch (result){
		case GTK_RESPONSE_CLOSE:
		break;
	}

	gtk_widget_destroy (phoebe_about_dialog);
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
	if(gui_warning("Quit PHOEBE?", "By quitting Phoebe all unsaved data will be lost. Are you sure you want to quit?") == 1)
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

	if(PHOEBE_FILEFLAG)
		status = phoebe_save_parameter_file(PHOEBE_FILENAME);
	else
		status = gui_save_parameter_file ();

	printf("In on_phoebe_save_toolbutton_clicked\n");
	printf("\tPHOEBE_FILEFLAG = %d\n", PHOEBE_FILEFLAG);
	printf("\tPHOEBE_FILENAME = %s\n", PHOEBE_FILENAME);

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
	GUI_widget *rv_xstart_label			= gui_widget_lookup("phoebe_rv_plot_options_phstart_label");
	GUI_widget *rv_xend_label			= gui_widget_lookup("phoebe_rv_plot_options_phend_label");

	GUI_widget *rv_xstart_spinbutton	= gui_widget_lookup("phoebe_rv_plot_options_phstart_spinbutton");
	GUI_widget *rv_xend_spinbutton		= gui_widget_lookup("phoebe_rv_plot_options_phend_spinbutton");


	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0){
		/* Phase */
		gtk_label_set_text(GTK_LABEL(rv_xstart_label->gtk), "Phase start:");
		gtk_label_set_text(GTK_LABEL(rv_xend_label->gtk), "Phase end:");
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xstart_spinbutton->gtk), -0.6);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xend_spinbutton->gtk), 0.6);
	}
	else{
		/* Time */
		gtk_label_set_text(GTK_LABEL(rv_xstart_label->gtk), "Time start: ");
		gtk_label_set_text(GTK_LABEL(rv_xend_label->gtk), "Time end: ");
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xstart_spinbutton->gtk), -0.1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xend_spinbutton->gtk), 1.1);
	}
}


void on_phoebe_lc_plot_options_x_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GUI_widget *lc_xstart_label 		= gui_widget_lookup("phoebe_lc_plot_options_phstart_label");
	GUI_widget *lc_xend_label 			= gui_widget_lookup("phoebe_lc_plot_options_phend_label");

	GUI_widget *lc_xstart_spinbutton	= gui_widget_lookup("phoebe_lc_plot_options_phstart_spinbutton");
	GUI_widget *lc_xend_spinbutton		= gui_widget_lookup("phoebe_lc_plot_options_phend_spinbutton");

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0){
		/* Phase */
		gtk_label_set_text(GTK_LABEL(lc_xstart_label->gtk), "Phase start:");
		gtk_label_set_text(GTK_LABEL(lc_xend_label->gtk), "Phase end:");
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xstart_spinbutton->gtk), -0.6);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xend_spinbutton->gtk), 0.6);
	}
	else{
		/* Time */
		gtk_label_set_text(GTK_LABEL(lc_xstart_label->gtk), "Time start: ");
		gtk_label_set_text(GTK_LABEL(lc_xend_label->gtk), "Time end: ");
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xstart_spinbutton->gtk), -0.1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xend_spinbutton->gtk), 1.1);
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

extern gdouble lc_zoom;
extern gint lc_zoom_level;
extern gdouble lc_x_offset;
extern gdouble lc_y_offset;

gdouble lc_zoom=0.0;
gint 	lc_zoom_level=0;
gdouble lc_x_offset=0.0;
gdouble lc_y_offset=0.0;

int phoebe_gui_lc_plot (gdouble x_offset, gdouble y_offset, gdouble zoom)
{
	int lcno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
	phoebe_parameter_get_value(par, &lcno);

	if(lcno > 0){
		gui_get_values_from_widgets();
		gui_plot_lc_using_gnuplot(x_offset, y_offset, zoom);

		gui_fill_sidesheet_res_treeview();
	}

	return SUCCESS;
}

void on_phoebe_lc_plot_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

void on_phoebe_lc_plot_save_button_clicked (GtkButton *button, gpointer user_data)
{
	int lcno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
	phoebe_parameter_get_value(par, &lcno);

	if(lcno > 0){
		GtkWidget *dialog;
		gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
		int status = 0;

		dialog = gtk_file_chooser_dialog_new ("Save LC Curves to ASCII File",
										  	GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
										 	GTK_FILE_CHOOSER_ACTION_OPEN,
										 	GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  	GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
										  	NULL);

		gtk_file_chooser_set_do_overwrite_confirmation (GTK_FILE_CHOOSER (dialog), TRUE);

		gchar *dir;
		phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

		gtk_file_chooser_set_current_folder((GtkFileChooser*)dialog, dir);

    	gtk_window_set_icon (GTK_WINDOW(dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

		if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT){
			gchar *filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
			gui_get_values_from_widgets();
			status =gui_plot_lc_to_ascii (filename);
			
			//else
			//	gui_notice ("Missing write permissions", "Selected directory has no write permissions.");

			g_free (filename);
		}

		gtk_widget_destroy (dialog);

		gui_fill_sidesheet_res_treeview();
	}
}

void on_phoebe_lc_plot_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_image_set_from_pixbuf(GTK_IMAGE(gui_widget_lookup ("phoebe_lc_plot_image")->gtk), NULL);
}

void on_phoebe_lc_plot_controls_reset_button_clicked (GtkButton *button, gpointer user_data)
{
	lc_zoom=0.0;
	lc_zoom_level=0;
	lc_x_offset=0.0;
	lc_y_offset=0.0;
	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

void on_phoebe_lc_plot_controls_right_button_clicked (GtkButton *button, gpointer user_data)
{
	lc_x_offset+=0.1;
	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

void on_phoebe_lc_plot_controls_up_button_clicked (GtkButton *button, gpointer user_data)
{
	lc_y_offset+=0.1;
	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

void on_phoebe_lc_plot_controls_left_button_clicked (GtkButton *button, gpointer user_data)
{
	lc_x_offset-=0.1;
	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

void on_phoebe_lc_plot_controls_down_button_clicked (GtkButton *button, gpointer user_data)
{
	lc_y_offset-=0.1;
	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

void on_phoebe_lc_plot_controls_zoomin_button_clicked (GtkButton *button, gpointer user_data)
{
	if (lc_zoom_level<5){
		lc_zoom-=0.1;
		lc_zoom_level+=1;
		phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
	}
}

void on_phoebe_lc_plot_controls_zoomout_button_clicked (GtkButton *button, gpointer user_data)
{
	if (lc_zoom_level>-5){
		lc_zoom+=0.1;
		lc_zoom_level-=1;
		phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
	}
}



extern gdouble rv_zoom;
extern gint rv_zoom_level;
extern gdouble rv_x_offset;
extern gdouble rv_y_offset;

gdouble rv_zoom=0.0;
gint 	rv_zoom_level=0;
gdouble rv_x_offset=0.0;
gdouble rv_y_offset=0.0;

int phoebe_gui_rv_plot (gdouble x_offset, gdouble y_offset, gdouble zoom)
{
	int rvno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
	phoebe_parameter_get_value(par, &rvno);

	if(rvno > 0){
		gui_get_values_from_widgets();
		gui_plot_rv_using_gnuplot (x_offset, y_offset, zoom);

		gui_fill_sidesheet_res_treeview();
	}

	return SUCCESS;
}

void on_phoebe_rv_plot_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

void on_phoebe_rv_plot_save_button_clicked (GtkButton *button, gpointer user_data)
{
	int rvno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
	phoebe_parameter_get_value(par, &rvno);

	if(rvno > 0){
		GtkWidget *dialog;
		gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
		int status = 0;

		dialog = gtk_file_chooser_dialog_new ("Save RV Curves to ASCII File",
										  	GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
										 	GTK_FILE_CHOOSER_ACTION_OPEN,
										 	GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  	GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
										  	NULL);

		gtk_file_chooser_set_do_overwrite_confirmation (GTK_FILE_CHOOSER (dialog), TRUE);

		gchar *dir;
		phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

		gtk_file_chooser_set_current_folder((GtkFileChooser*)dialog, dir);

    	gtk_window_set_icon (GTK_WINDOW(dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

		if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT){
			gchar *filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
			gui_get_values_from_widgets();
			status =gui_plot_rv_to_ascii (filename);
			
			//else
			//	gui_notice ("Missing write permissions", "Selected directory has no write permissions.");

			g_free (filename);
		}

		gtk_widget_destroy (dialog);

		gui_fill_sidesheet_res_treeview();
	}
}

void on_phoebe_rv_plot_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_image_set_from_pixbuf(GTK_IMAGE(gui_widget_lookup ("phoebe_rv_plot_image")->gtk), NULL);
}

void on_phoebe_rv_plot_controls_reset_button_clicked (GtkButton *button, gpointer user_data)
{
	rv_zoom=0.0;
	rv_zoom_level=0;
	rv_x_offset=0.0;
	rv_y_offset=0.0;
	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

void on_phoebe_rv_plot_controls_right_button_clicked (GtkButton *button, gpointer user_data)
{
	rv_x_offset+=0.1;
	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

void on_phoebe_rv_plot_controls_up_button_clicked (GtkButton *button, gpointer user_data)
{
	rv_y_offset+=0.1;
	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

void on_phoebe_rv_plot_controls_left_button_clicked (GtkButton *button, gpointer user_data)
{
	rv_x_offset-=0.1;
	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

void on_phoebe_rv_plot_controls_down_button_clicked (GtkButton *button, gpointer user_data)
{
	rv_y_offset-=0.1;
	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

void on_phoebe_rv_plot_controls_zoomin_button_clicked (GtkButton *button, gpointer user_data)
{
	if (rv_zoom_level<5){
		rv_zoom-=0.1;
		rv_zoom_level+=1;
		phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
	}
}

void on_phoebe_rv_plot_controls_zoomout_button_clicked (GtkButton *button, gpointer user_data)
{
	if (rv_zoom_level>-5){
		rv_zoom+=0.1;
		rv_zoom_level-=1;
		phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
	}
}


void on_phoebe_star_shape_plot_button_clicked (GtkButton *button, gpointer user_data)
{
		gui_get_values_from_widgets();
		gui_plot_eb_using_gnuplot();
}

void on_phoebe_star_shape_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_image_set_from_pixbuf(GTK_IMAGE(gui_widget_lookup ("phoebe_eb_plot_image")->gtk), NULL);
}


void on_phoebe_pot_calc_close_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_widget_destroy (GTK_WIDGET(user_data));
}

void on_phoebe_pot_calc_update_button_clicked (GtkButton *button, gpointer user_data)
{
	gint sel = (int) user_data;
	GtkWidget *pot_spinbutton=NULL;

	if (sel==1){pot_spinbutton = gui_widget_lookup("phoebe_para_comp_phsv_spinbutton")->gtk;}
	if (sel==2){pot_spinbutton = gui_widget_lookup("phoebe_para_comp_pcsv_spinbutton")->gtk;}

	gdouble pot = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_pot_spinbutton")));
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(pot_spinbutton), pot);
}

void on_phoebe_pot_calc_calculate_button_clicked (GtkButton *button, gpointer user_data)
{
	gboolean circ 	= gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(g_object_get_data (G_OBJECT (button), "data_circ_checkbutton")));
	gdouble d		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_d_spinbutton")));
	gdouble rm		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_rm_spinbutton")));
	gdouble r		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_r_spinbutton")));
	gdouble f		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_f_spinbutton")));
	gdouble lambda	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_lambda_spinbutton")));
	gdouble nu		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_nu_spinbutton")));

	gdouble pot 	= phoebe_calculate_pot1((int)(!circ), d, rm, r, f, lambda, nu);

	gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_pot_spinbutton")), pot);
}

void on_phoebe_para_comp_phsv_calculate_button_clicked (GtkButton *button, gpointer user_data)
{
    gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_potential_calculator.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_pot_calc_xml      			= glade_xml_new        (glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_pot_calc_dialog        		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potential_calculator_dialog");

	GtkWidget *phoebe_pot_calc_circ_checkbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_circularorbit_checkbutton");
	GtkWidget *phoebe_pot_calc_d_spinbutton			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_d_spinbutton");
	GtkWidget *phoebe_pot_calc_rm_spinbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_rm_spinbutton");
	GtkWidget *phoebe_pot_calc_r_spinbutton			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_r_spinbutton");
	GtkWidget *phoebe_pot_calc_f_spinbutton			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_f_spinbutton");
	GtkWidget *phoebe_pot_calc_lambda_spinbutton	= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_lambda_spinbutton");
	GtkWidget *phoebe_pot_calc_nu_spinbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_nu_spinbutton");
	GtkWidget *phoebe_pot_calc_pot_spinbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_pot_spinbutton");

	GtkWidget *phoebe_pot_calc_calculate_button		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_calculate_button");
	GtkWidget *phoebe_pot_calc_update_button		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_update_button");
	GtkWidget *phoebe_pot_calc_close_button			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_close_button");

	GtkWidget *phoebe_main_rm = gui_widget_lookup ("phoebe_para_sys_rm_spinbutton")->gtk;
	GtkWidget *phoebe_main_f1 = gui_widget_lookup ("phoebe_para_orb_f1_spinbutton")->gtk;

	gtk_spin_button_set_value (GTK_SPIN_BUTTON(phoebe_pot_calc_rm_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_main_rm)));
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(phoebe_pot_calc_f_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_main_f1)));

	g_object_unref (phoebe_pot_calc_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_pot_calc_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_pot_calc_dialog), "PHOEBE - Potential Calculator");

	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_circ_checkbutton", (gpointer) phoebe_pot_calc_circ_checkbutton);
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_d_spinbutton", (gpointer) phoebe_pot_calc_d_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_rm_spinbutton", (gpointer) phoebe_pot_calc_rm_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_r_spinbutton", (gpointer) phoebe_pot_calc_r_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_f_spinbutton", (gpointer) phoebe_pot_calc_f_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_lambda_spinbutton", (gpointer) phoebe_pot_calc_lambda_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_nu_spinbutton", (gpointer) phoebe_pot_calc_nu_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_pot_spinbutton", (gpointer) phoebe_pot_calc_pot_spinbutton );

	g_object_set_data (G_OBJECT (phoebe_pot_calc_update_button), "data_pot_spinbutton", (gpointer) phoebe_pot_calc_pot_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_update_button), "data_pot_spinbutton", (gpointer) phoebe_pot_calc_pot_spinbutton );

	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_close_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_close_button_clicked), (gpointer) phoebe_pot_calc_dialog);
	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_update_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_update_button_clicked), (gpointer) 1);
	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_calculate_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_calculate_button_clicked), NULL);

	gtk_widget_show (phoebe_pot_calc_dialog);
}



void on_phoebe_para_comp_pcsv_calculate_button_clicked (GtkButton *button, gpointer user_data)
{
    gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_potential_calculator.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_pot_calc_xml      			= glade_xml_new        (glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_pot_calc_dialog        		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potential_calculator_dialog");

	GtkWidget *phoebe_pot_calc_circ_checkbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_circularorbit_checkbutton");
	GtkWidget *phoebe_pot_calc_d_spinbutton			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_d_spinbutton");
	GtkWidget *phoebe_pot_calc_rm_spinbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_rm_spinbutton");
	GtkWidget *phoebe_pot_calc_r_spinbutton			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_r_spinbutton");
	GtkWidget *phoebe_pot_calc_f_spinbutton			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_f_spinbutton");
	GtkWidget *phoebe_pot_calc_lambda_spinbutton	= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_lambda_spinbutton");
	GtkWidget *phoebe_pot_calc_nu_spinbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_nu_spinbutton");
	GtkWidget *phoebe_pot_calc_pot_spinbutton		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_pot_spinbutton");

	GtkWidget *phoebe_pot_calc_calculate_button		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_calculate_button");
	GtkWidget *phoebe_pot_calc_update_button		= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_update_button");
	GtkWidget *phoebe_pot_calc_close_button			= glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_close_button");

	GtkWidget *phoebe_main_rm = gui_widget_lookup ("phoebe_para_sys_rm_spinbutton")->gtk;
	GtkWidget *phoebe_main_f2 = gui_widget_lookup ("phoebe_para_orb_f2_spinbutton")->gtk;

	gtk_spin_button_set_value (GTK_SPIN_BUTTON(phoebe_pot_calc_rm_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_main_rm)));
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(phoebe_pot_calc_f_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_main_f2)));

	gtk_label_set_markup (GTK_LABEL(glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_radius_label")),"<b>Secondary Star Radius</b>");
	gtk_label_set_text (GTK_LABEL(glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_r_label")),"R2:");
	gtk_label_set_markup (GTK_LABEL(glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_sync_label")),"<b>Secondary Star Synchronicity Parameter</b>");
	gtk_label_set_text (GTK_LABEL(glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_f_label")),"F2:");
	gtk_label_set_markup (GTK_LABEL(glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_potential_label")),"<b>Secondary Star Surface Potential</b>");
	gtk_label_set_text (GTK_LABEL(glade_xml_get_widget (phoebe_pot_calc_xml, "phoebe_potentialcalc_pot_label")),"PCSV:");

	g_object_unref (phoebe_pot_calc_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_pot_calc_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_pot_calc_dialog), "PHOEBE - Potential Calculator");

	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_circ_checkbutton", (gpointer) phoebe_pot_calc_circ_checkbutton);
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_d_spinbutton", (gpointer) phoebe_pot_calc_d_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_rm_spinbutton", (gpointer) phoebe_pot_calc_rm_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_r_spinbutton", (gpointer) phoebe_pot_calc_r_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_f_spinbutton", (gpointer) phoebe_pot_calc_f_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_lambda_spinbutton", (gpointer) phoebe_pot_calc_lambda_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_nu_spinbutton", (gpointer) phoebe_pot_calc_nu_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "data_pot_spinbutton", (gpointer) phoebe_pot_calc_pot_spinbutton );

	g_object_set_data (G_OBJECT (phoebe_pot_calc_update_button), "data_pot_spinbutton", (gpointer) phoebe_pot_calc_pot_spinbutton );

	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_close_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_close_button_clicked), (gpointer) phoebe_pot_calc_dialog);
	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_update_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_update_button_clicked), (gpointer) 2);
	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_calculate_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_calculate_button_clicked), NULL);

	gtk_widget_show (phoebe_pot_calc_dialog);
}


void on_phoebe_ld_dialog_close_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_widget_destroy (GTK_WIDGET(user_data));
}


void on_phoebe_ld_dialog_interpolate_button_clicked (GtkButton *button, gpointer user_data)
{
	double tavh 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_tavh_spinbutton")));
	double tavc 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_tavc_spinbutton")));
	double logg1 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_logg1_spinbutton")));
	double logg2 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_logg2_spinbutton")));
	double met1 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_met1_spinbutton")));
	double met2 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_met2_spinbutton")));

	char* ldlaw 	= strdup(phoebe_parameter_lookup ("phoebe_ld_model")->menu->option[gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_id_combobox")))]);

	int index = gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_id_combobox")));
	char *id;
	
	phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lc_id"), index, &id);

	double x1, x2, y1, y2;

	phoebe_get_ld_coefficients (phoebe_ld_model_type (ldlaw), phoebe_passband_lookup_by_id(id), met1, tavh, logg1, &x1, &y1);
	phoebe_get_ld_coefficients (phoebe_ld_model_type (ldlaw), phoebe_passband_lookup_by_id(id), met2, tavc, logg2, &x2, &y2);

	gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_x1_spinbutton")), x1);
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_x2_spinbutton")), x2);
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_y1_spinbutton")), y1);
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_y2_spinbutton")), y2);
}


void on_phoebe_ld_dialog_update_button_clicked (GtkButton *button, gpointer user_data)
{
	double x1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_x1_spinbutton")));
	double x2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_x2_spinbutton")));
	double y1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_y1_spinbutton")));
	double y2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_y2_spinbutton")));

	int index = gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_id_combobox")));
	char *id, *id_in_model;
	
	phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lc_id"), index, &id);
	
	GtkTreeModel *model = GTK_TREE_MODEL(gui_widget_lookup("phoebe_para_ld_lccoefs_primx")->gtk);
	GtkTreeIter iter;

	int state = gtk_tree_model_get_iter_first (model, &iter);

	while (state) {
		index = atoi (gtk_tree_model_get_string_from_iter (model, &iter));

		gtk_tree_model_get (model, &iter, LC_COL_ID, &id_in_model, -1);
		if(strcmp(id, id_in_model) == 0){
			gtk_list_store_set (GTK_LIST_STORE(model), &iter, 	LC_COL_X1, x1,
																LC_COL_X2, x2,
																LC_COL_Y1, y1,
																LC_COL_Y2, y2, -1);
			break;
		}
		state = gtk_tree_model_iter_next (model, &iter);
	}
}

void on_phoebe_para_ld_model_tables_vanhamme_button_clicked (GtkButton *button, gpointer user_data)
{

	GtkTreeModel 	*lc_model = GTK_TREE_MODEL(gui_widget_lookup ("phoebe_data_lc_filter")->gtk);
	GtkCellRenderer *renderer;

	PHOEBE_parameter *ldlaw = phoebe_parameter_lookup ("phoebe_ld_model");
	int optindex, optcount;
	
	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_ld_interpolator.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_ld_dialog_xml      			= glade_xml_new        (glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_ld_dialog        				= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog");

	GtkWidget *phoebe_ld_dialog_law_combobox		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_law_combobox");
	GtkWidget *phoebe_ld_dialog_id_combobox			= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_id_combobox");
	GtkWidget *phoebe_ld_dialog_tavh_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_tavh_spinbutton");
	GtkWidget *phoebe_ld_dialog_logg1_spinbutton	= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_logg1_spinbutton");
	GtkWidget *phoebe_ld_dialog_met1_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_met1_spinbutton");
	GtkWidget *phoebe_ld_dialog_tavc_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_tavc_spinbutton");
	GtkWidget *phoebe_ld_dialog_logg2_spinbutton	= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_logg2_spinbutton");
	GtkWidget *phoebe_ld_dialog_met2_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_met2_spinbutton");
	GtkWidget *phoebe_ld_dialog_x1_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_x1_spinbutton");
	GtkWidget *phoebe_ld_dialog_y1_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_y1_spinbutton");
	GtkWidget *phoebe_ld_dialog_x2_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_x2_spinbutton");
	GtkWidget *phoebe_ld_dialog_y2_spinbutton		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_y2_spinbutton");

	GtkWidget *phoebe_ld_dialog_interpolate_button 	= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_interpolate_button");
	GtkWidget *phoebe_ld_dialog_update_button 		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_update_button");
	GtkWidget *phoebe_ld_dialog_close_button 		= glade_xml_get_widget (phoebe_ld_dialog_xml, "phoebe_ld_dialog_close_button");

	g_object_unref (phoebe_ld_dialog_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_ld_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_ld_dialog), "PHOEBE - LD Coefficients Inerpolation");

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_ld_dialog_id_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT (phoebe_ld_dialog_id_combobox), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT (phoebe_ld_dialog_id_combobox), renderer, "text", LC_COL_ID);
	gtk_combo_box_set_model (GTK_COMBO_BOX(phoebe_ld_dialog_id_combobox), lc_model);

	gtk_combo_box_set_active(GTK_COMBO_BOX(phoebe_ld_dialog_id_combobox), 0);

	optcount = ldlaw->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_ld_dialog_law_combobox), strdup(ldlaw->menu->option[optindex]));

	gtk_combo_box_set_active(GTK_COMBO_BOX(phoebe_ld_dialog_law_combobox), gtk_combo_box_get_active(GTK_COMBO_BOX(gui_widget_lookup("phoebe_para_ld_model_combobox")->gtk)));

	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_law_combobox", (gpointer) phoebe_ld_dialog_law_combobox);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_id_combobox", (gpointer) phoebe_ld_dialog_id_combobox);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_tavh_spinbutton", (gpointer) phoebe_ld_dialog_tavh_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_logg1_spinbutton", (gpointer) phoebe_ld_dialog_logg1_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_met1_spinbutton", (gpointer) phoebe_ld_dialog_met1_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_tavc_spinbutton", (gpointer) phoebe_ld_dialog_tavc_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_logg2_spinbutton", (gpointer) phoebe_ld_dialog_logg2_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_met2_spinbutton", (gpointer) phoebe_ld_dialog_met2_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_x1_spinbutton", (gpointer) phoebe_ld_dialog_x1_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_y1_spinbutton", (gpointer) phoebe_ld_dialog_y1_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_x2_spinbutton", (gpointer) phoebe_ld_dialog_x2_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_y2_spinbutton", (gpointer) phoebe_ld_dialog_y2_spinbutton );

	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button), "data_x1_spinbutton", (gpointer) phoebe_ld_dialog_x1_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button), "data_y1_spinbutton", (gpointer) phoebe_ld_dialog_y1_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button), "data_x2_spinbutton", (gpointer) phoebe_ld_dialog_x2_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button), "data_y2_spinbutton", (gpointer) phoebe_ld_dialog_y2_spinbutton );
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button), "data_id_combobox", (gpointer) phoebe_ld_dialog_id_combobox);

	g_signal_connect (GTK_WIDGET(phoebe_ld_dialog_close_button), "clicked", G_CALLBACK (on_phoebe_ld_dialog_close_button_clicked), (gpointer) phoebe_ld_dialog);
	g_signal_connect (GTK_WIDGET(phoebe_ld_dialog_update_button), "clicked", G_CALLBACK (on_phoebe_ld_dialog_update_button_clicked), NULL);
	g_signal_connect (GTK_WIDGET(phoebe_ld_dialog_interpolate_button), "clicked", G_CALLBACK (on_phoebe_ld_dialog_interpolate_button_clicked), NULL);

	gtk_widget_show (phoebe_ld_dialog);
}
