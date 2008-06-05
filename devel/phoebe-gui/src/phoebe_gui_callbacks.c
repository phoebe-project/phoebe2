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

bool LD_COEFFS_NEED_UPDATING = TRUE;
bool LOGG_VALUES_NEED_RECALCULATING = TRUE;

G_MODULE_EXPORT void on_phoebe_para_tba_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	char *widget_name = (char*)gtk_widget_get_name(GTK_WIDGET(togglebutton));
	gui_get_value_from_widget(gui_widget_lookup(widget_name));

	gui_fill_sidesheet_fit_treeview ();
	gui_fill_fitt_mf_treeview();
}

G_MODULE_EXPORT void on_phoebe_data_star_name_entry_changed (GtkEditable *editable, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_data_lc_seedgen_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int seed;
	GtkWidget *seed_spin_button = gui_widget_lookup("phoebe_data_lc_seed_spinbutton")->gtk;

	srand (time (0));
	seed = (int) (100000001.0 + (double) rand () / RAND_MAX * 100000000.0);
	gtk_spin_button_set_value (GTK_SPIN_BUTTON (seed_spin_button), seed);
}

void on_phoebe_potential_parameter_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	int status = 0;

	GtkWidget *phoebe_para_sys_rm_spinbutton = gui_widget_lookup("phoebe_para_sys_rm_spinbutton")->gtk;
	GtkWidget *phoebe_para_orb_ecc_spinbutton = gui_widget_lookup("phoebe_para_orb_ecc_spinbutton")->gtk;
	GtkWidget *phoebe_para_orb_f1_spinbutton = gui_widget_lookup("phoebe_para_orb_f1_spinbutton")->gtk;

	GtkTreeView *phoebe_sidesheet_res_treeview = (GtkTreeView*)gui_widget_lookup("phoebe_sidesheet_res_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model(phoebe_sidesheet_res_treeview);
	GtkTreeIter iter;

	double q, e, F, L1, L2;

	gtk_tree_model_get_iter_first (model, &iter);

	q = gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_para_sys_rm_spinbutton));
	e = gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_para_orb_ecc_spinbutton));
	F = gtk_spin_button_get_value(GTK_SPIN_BUTTON(phoebe_para_orb_f1_spinbutton));

	printf("\nq = %f, e = %f, f1 = %f\n", q, e, F);

	status = phoebe_calculate_critical_potentials(q, F, e, &L1, &L2);

	printf("L1 = %f, L2 = %f\n", L1, L2);

	/* Conviniently the potentials are in the first two rows */

	gtk_tree_model_get_iter_first (model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Ω(L<sub>1</sub>)", RS_COL_PARAM_VALUE, L1, -1);

	gtk_tree_model_iter_next (model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Ω(L<sub>2</sub>)", RS_COL_PARAM_VALUE, L2, -1);
}

static PHOEBE_passband *phoebe_bolometric_passband()
{
	/*
	 * Creates the bolometric "passband", used for calculating bolometric limb darkening coefficients.
	 */

	PHOEBE_passband *passband = phoebe_passband_new();
	passband->id = 0;
	passband->set = "Bolometric";
	return passband;
}

int gui_interpolate_all_ld_coefficients (char* ldlaw, double tavh, double tavc, double logg1, double logg2, double met1, double met2)
{
	/* Interpolate all LD coefficients */
	PHOEBE_passband *passband;
	int lcno, index;
	char *notice_title = "LD coefficients interpolation";
	int calc_ld1 = 1, calc_ld2 = 1;
	GtkTreeModel *model = GTK_TREE_MODEL(gui_widget_lookup("phoebe_para_ld_lccoefs_primx")->gtk);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);

	//printf("    Updating LD coefficients\n    LD law: %s\n    met1 = %f, tavh = %f, logg1 = %f\n    met2 = %f, tavc = %f, logg2 = %f\n", ldlaw, met1, tavh, logg1, met2, tavc, logg2);

	for (index = -1; (index < lcno) && (calc_ld1 || calc_ld2); index++) {
		GtkTreeIter iter;
		double x1, x2, y1, y2;

		if (index == -1) {
			/* Bolometric LD coefficients */
			passband = phoebe_bolometric_passband();
		}
		else {
			char *id;
			int i;

			phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lc_id"), index, &id);
			passband = phoebe_passband_lookup_by_id(id);
			gtk_tree_model_get_iter_first (model, &iter);
			for (i = 0; i < index; i++)
				gtk_tree_model_iter_next (model, &iter);
		}

		//printf("    Now calculating LD coeffs for %s:%s light curve.\n", passband->set, passband->name);

		if (calc_ld1) {
			switch (phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met1, tavh, logg1, &x1, &y1)) {
				case SUCCESS:
					//printf("    x1 = %f, y1 = %f\n", x1, y1);
					if (index == -1) {
						/* Bolometric LD coefficients */
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_primx_spinbutton")->gtk), x1);
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_primy_spinbutton")->gtk), y1);
					}
					else {
						gtk_list_store_set (GTK_LIST_STORE(model), &iter,	LC_COL_X1, x1,
													LC_COL_Y1, y1, -1);
					}
					break;
				case ERROR_LD_TABLES_MISSING:
					gui_notice(notice_title, "Van Hamme tables are missing");
					calc_ld1 = 0; calc_ld2 = 0;
					break;
				case ERROR_LD_PARAMS_OUT_OF_RANGE:
					gui_notice(notice_title, "Parameters for the primary component are out of range");
				default:
					calc_ld1 = 0;
					break;
			}
		}

		if (calc_ld2) {
			switch (phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met2, tavh, logg2, &x2, &y2)) {
				case SUCCESS:
					//printf("    x2 = %f, y2 = %f\n", x2, y2);
					if (index == -1) {
						/* Bolometric LD coefficients */
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_secx_spinbutton")->gtk), x2);
						gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_secy_spinbutton")->gtk), y2);
					}
					else {
						gtk_list_store_set (GTK_LIST_STORE(model), &iter,	LC_COL_X2, x2,
													LC_COL_Y2, y2, -1);
					}
					break;
				case ERROR_LD_PARAMS_OUT_OF_RANGE:
					gui_notice(notice_title, "Parameters for the secondary component are out of range");
				default:
					calc_ld2 = 0;
					break;
			}
		}
	}

	return (calc_ld1 && calc_ld2);
}

void gui_update_ld_coefficients()
{
	/* Update calculated values (log g, mass, radius, ...) and interpolate LD coefficients */
	double tavh, tavc, logg1, logg2, met1, met2;
	char* ldlaw;

	gui_get_values_from_widgets();
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff1"), &tavh);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff2"), &tavc);
	if (LOGG_VALUES_NEED_RECALCULATING) {
		//printf("    Recalculating log g values\n");
		call_wd_to_get_logg_values (&logg1, &logg2);
		LOGG_VALUES_NEED_RECALCULATING = FALSE;
	}
	else {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"), &logg1);
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"), &logg2);
	}
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met1"), &met1);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met2"), &met2);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_model"), &ldlaw);

	if (gui_interpolate_all_ld_coefficients(ldlaw, tavh, tavc, logg1, logg2, met1, met2))
		LD_COEFFS_NEED_UPDATING = FALSE;
	gui_fill_sidesheet_res_treeview();
}

void gui_update_ld_coefficients_on_autoupdate()
{
	/* Update LD coefficients when parameter gui_ld_model_autoupdate is set */
	GtkWidget *phoebe_para_ld_model_autoupdate_checkbutton = gui_widget_lookup("phoebe_para_ld_model_autoupdate_checkbutton")->gtk;

	if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(phoebe_para_ld_model_autoupdate_checkbutton)))
		gui_update_ld_coefficients();
}

void gui_update_ld_coefficients_when_needed()
{
	/* Update LD coefficients when changes have been made to M/T/log g/LD law */
	if (LD_COEFFS_NEED_UPDATING)
		gui_update_ld_coefficients_on_autoupdate();
}


/* ******************************************************************** *
 *
 *                    phoebe parameter changes that require LD update
 *
 * ******************************************************************** */

void gui_ld_coeffs_need_updating()
{
	LD_COEFFS_NEED_UPDATING = TRUE;
}

void gui_logg_values_need_recalculating()
{
	LOGG_VALUES_NEED_RECALCULATING = TRUE;
	gui_ld_coeffs_need_updating();
}

G_MODULE_EXPORT void on_phoebe_para_ld_model_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_ld_coeffs_need_updating();
}

G_MODULE_EXPORT void on_phoebe_para_comp_tavh_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_ld_coeffs_need_updating();
}

G_MODULE_EXPORT void on_phoebe_para_comp_tavc_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_ld_coeffs_need_updating();
}

G_MODULE_EXPORT void on_phoebe_para_comp_phsv_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_comp_pcsv_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_comp_met1_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_ld_coeffs_need_updating();
}

G_MODULE_EXPORT void on_phoebe_para_comp_met2_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_ld_coeffs_need_updating();
}

G_MODULE_EXPORT void on_phoebe_para_sys_sma_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_sys_rm_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_eph_period_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_orb_ecc_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_orb_f1_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}

G_MODULE_EXPORT void on_phoebe_para_orb_f2_spinbutton_changed (GtkComboBox *widget, gpointer user_data)
{
	gui_logg_values_need_recalculating();
}


/* ******************************************************************** *
 *
 *                    phoebe fitting tab events
 *
 * ******************************************************************** */

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback;
int accept_flag = 0;

G_MODULE_EXPORT
void on_phoebe_fitt_calculate_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GtkTreeView 	*phoebe_fitt_mf_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_first_treeview")->gtk);
	GtkTreeView		*phoebe_fitt_second_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_second_treeview")->gtk);
	GtkComboBox 	*phoebe_fitt_method_combobox = GTK_COMBO_BOX(gui_widget_lookup("phoebe_fitt_method_combobox")->gtk);
	GtkLabel		*phoebe_fitt_feedback_label = GTK_LABEL(gui_widget_lookup("phoebe_fitt_feedback_label")->gtk);
	GtkTreeModel 	*model;
	GtkTreeIter iter;
	int index, count;
	char *id;
	char status_message[255] = "Minimizer feedback";
	PHOEBE_curve *curve;

	int status = 0;

	phoebe_minimizer_feedback = phoebe_minimizer_feedback_new ();

	gui_update_ld_coefficients_when_needed();
	status = gui_get_values_from_widgets();

	switch (gtk_combo_box_get_active (phoebe_fitt_method_combobox)) {
		case 0:
			status = phoebe_minimize_using_dc(stdout, phoebe_minimizer_feedback);
			phoebe_gui_debug("DC minimizer says: %s", phoebe_error(status));
		break;
		case 1:
			status = phoebe_minimize_using_nms (stdout, phoebe_minimizer_feedback);
			phoebe_gui_debug ("NMS minimizer says: %s", phoebe_error(status));
		break;
		default:
			phoebe_minimizer_feedback_free (phoebe_minimizer_feedback);
			gui_error ("Invalid minimization algorithm", "The minimization algorithm is not selected or is invalid. Please select a valid entry in the fitting tab and try again.");
			return;
		break;
	}

	if (status == SUCCESS) {
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
	gdk_beep ();
}

int gui_spot_index(int spotsrc, int spotid)
{
	/* Returns the index of the spot given by its source (primary/secondary) and its id */
	int i, spno, current_spotsrc, current_spotid = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no"), &spno);
	for (i = 0; i < spno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_source"), i, &current_spotsrc);
		if (current_spotsrc == spotsrc) {
			if (spotid == ++current_spotid)
				return i;
		}
	}

	return -1;
}

void gui_set_spot_parameter(char *wd_qualifier, char *phoebe_qualifier, int spotsrc, int spotid)
{
	/* Sets the corresponding PHOEBE spot parameter based on the value of the WD parameter */
	PHOEBE_parameter *wd_par = phoebe_parameter_lookup (wd_qualifier);
	if (wd_par->tba) {
		double value;
		phoebe_parameter_get_value (wd_par, &value);
		PHOEBE_parameter *phoebe_par = phoebe_parameter_lookup(phoebe_qualifier);
		int spotindex = gui_spot_index(spotsrc, spotid);
		if (spotindex >= 0)
			phoebe_parameter_set_value (phoebe_par, spotindex, value);
		phoebe_parameter_set_tba(wd_par, FALSE);  // so that it is no longer displayed in the side sheet, the phoebe parameter will be displayed instead
	}
}

void gui_set_spot_parameters()
{
	/* Sets the PHOEBE spot parameters based on the WD parameters after fitting */
	int spotid, spotsrc;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot1src"), &spotsrc);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot1id"), &spotid);

	gui_set_spot_parameter ("wd_spots_lat1",  "phoebe_spots_colatitude", spotsrc, spotid);
	gui_set_spot_parameter ("wd_spots_long1", "phoebe_spots_longitude",  spotsrc, spotid);
	gui_set_spot_parameter ("wd_spots_rad1",  "phoebe_spots_radius",     spotsrc, spotid);
	gui_set_spot_parameter ("wd_spots_temp1", "phoebe_spots_tempfactor", spotsrc, spotid);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot2src"), &spotsrc);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot2id"), &spotid);

	gui_set_spot_parameter ("wd_spots_lat2",  "phoebe_spots_colatitude", spotsrc, spotid);
	gui_set_spot_parameter ("wd_spots_long2", "phoebe_spots_longitude",  spotsrc, spotid);
	gui_set_spot_parameter ("wd_spots_rad2",  "phoebe_spots_radius",     spotsrc, spotid);
	gui_set_spot_parameter ("wd_spots_temp2", "phoebe_spots_tempfactor", spotsrc, spotid);
}

G_MODULE_EXPORT
void on_phoebe_fitt_updateall_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status;

	if (accept_flag) {
		status = phoebe_minimizer_feedback_accept (phoebe_minimizer_feedback);
		gui_set_spot_parameters();
		status = gui_set_values_to_widgets();
		on_phoebe_para_spots_treeview_cursor_changed((GtkTreeView *)NULL, (gpointer)NULL);  // Change the values of the current spot
		gui_update_ld_coefficients_on_autoupdate();
		gui_fill_sidesheet_fit_treeview ();
		gui_fill_fitt_mf_treeview();
		accept_flag = 0;
	}
}

G_MODULE_EXPORT void on_phoebe_fitt_method_combobox_changed (GtkComboBox *widget, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_fitt_fitting_corrmat_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_cormat.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_cormat_dialog_xml      		= glade_xml_new        (glade_xml_file, NULL, NULL);

	GtkWidget *phoebe_cormat_dialog        			= glade_xml_get_widget (phoebe_cormat_dialog_xml, "phoebe_cormat_dialog");
	GtkWidget *phoebe_cormat_treeview				= glade_xml_get_widget (phoebe_cormat_dialog_xml, "phoebe_cormat_dialog_treeview");

	int rows, cols, cormat_cols, cormat_rows;
	char cormat_string[255];

	g_object_unref (phoebe_cormat_dialog_xml);

	gtk_window_set_icon (GTK_WINDOW (phoebe_cormat_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW(phoebe_cormat_dialog), "PHOEBE - Correlation matrix");

	if(phoebe_minimizer_feedback){
		cormat_cols = phoebe_minimizer_feedback->cormat->cols;
		cormat_rows = phoebe_minimizer_feedback->cormat->rows;
		GType cormat_col_types[cormat_cols+1];

		GtkListStore *cormat_model;
		GtkCellRenderer *renderer;
		GtkTreeViewColumn *column;
		GtkTreeIter iter;

		for(cols = 0; cols < cormat_cols+1; cols++){
			cormat_col_types[cols] = G_TYPE_STRING;
		}
		cormat_model = gtk_list_store_newv (cormat_cols+1, cormat_col_types);

		renderer = gtk_cell_renderer_text_new ();
		column	  = gtk_tree_view_column_new_with_attributes ("Parameter", renderer, "text", 0, NULL);
		gtk_tree_view_insert_column (GTK_TREE_VIEW(phoebe_cormat_treeview), column, -1);

		for(cols = 0; cols < cormat_cols; cols++){
			renderer    = gtk_cell_renderer_text_new ();
			column      = gtk_tree_view_column_new_with_attributes (phoebe_minimizer_feedback->qualifiers->val.strarray[cols], renderer, "text", cols+1, NULL);
			gtk_tree_view_insert_column (GTK_TREE_VIEW(phoebe_cormat_treeview), column, -1);
		}

		gtk_list_store_append(cormat_model, &iter);
		gtk_list_store_set(cormat_model, &iter, 0, "", -1);
		for(cols = 0; cols < cormat_cols; cols++){
			gtk_list_store_set(cormat_model, &iter, cols + 1, phoebe_minimizer_feedback->qualifiers->val.strarray[cols], -1);
		}

		for(rows = 0; rows < cormat_rows; rows++){
			gtk_list_store_append(cormat_model, &iter);
			gtk_list_store_set(cormat_model, &iter, 0, phoebe_minimizer_feedback->qualifiers->val.strarray[rows], -1);
			for(cols = 0; cols < cormat_cols; cols++){
				sprintf(cormat_string, "% 20.3lf", phoebe_minimizer_feedback->cormat->val[rows][cols]);
				gtk_list_store_set(cormat_model, &iter, cols + 1, cormat_string, -1);
			}
		}
		gtk_tree_view_set_model(GTK_TREE_VIEW(phoebe_cormat_treeview), GTK_TREE_MODEL(cormat_model));
	}

	gtk_dialog_run(GTK_DIALOG(phoebe_cormat_dialog));
	gtk_widget_destroy(GTK_WIDGET(phoebe_cormat_dialog));
}

G_MODULE_EXPORT void on_phoebe_fitt_nms_nolimit_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkWidget *phoebe_fitt_nms_iters_spinbutton	= gui_widget_lookup("phoebe_fitt_nms_iters_spinbutton")->gtk;
	gint iters;

	if(gtk_toggle_button_get_active(togglebutton)){
		g_object_set_data (G_OBJECT (phoebe_fitt_nms_iters_spinbutton), "old_value", GINT_TO_POINTER (gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(phoebe_fitt_nms_iters_spinbutton))));
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_fitt_nms_iters_spinbutton), 0);
		gtk_widget_set_sensitive(phoebe_fitt_nms_iters_spinbutton, FALSE);
	}
	else{
		iters = GPOINTER_TO_INT (g_object_get_data (G_OBJECT (phoebe_fitt_nms_iters_spinbutton), "old_value"));
		if (iters == 0) iters = 1;
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_fitt_nms_iters_spinbutton), iters);
		gtk_widget_set_sensitive(phoebe_fitt_nms_iters_spinbutton, TRUE);
	}
}

G_MODULE_EXPORT void on_phoebe_fitt_nms_iters_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	if(gtk_spin_button_get_value(spinbutton) == 0){
		GtkWidget *phoebe_fitt_nms_nolimit_checkbutton = gui_widget_lookup("phoebe_fitt_nms_nolimit_checkbutton")->gtk;
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(phoebe_fitt_nms_nolimit_checkbutton), TRUE);
		gtk_widget_set_sensitive(GTK_WIDGET(spinbutton), FALSE);
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

G_MODULE_EXPORT void on_phoebe_data_lc_active_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_load_lc_filechooserbutton_selection_changed (GtkFileChooserButton *filechooserbutton, gpointer user_data)
{
	gui_set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
}

G_MODULE_EXPORT void on_phoebe_data_lc_model_row_changed (GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_load_rv_filechooserbutton_selection_changed (GtkFileChooserButton *filechooserbutton, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_data_rv_active_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_data_rv_model_row_changed (GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data)
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


G_MODULE_EXPORT void on_phoebe_para_spots_treeview_cursor_changed (GtkTreeView *tree_view, gpointer user_data)
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

		markup = g_markup_printf_escaped ("<b>Colatitude of spot %d</b>", atoi(gtk_tree_model_get_string_from_iter(model, &iter))+1);
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

G_MODULE_EXPORT void on_phoebe_para_spots_add_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_spots_add();
}

G_MODULE_EXPORT void on_phoebe_para_spots_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_spots_edit();
}

G_MODULE_EXPORT void on_phoebe_para_spots_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_spots_edit();
}

G_MODULE_EXPORT void on_phoebe_para_spots_remove_button_clicked (GtkButton *button, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_para_spots_active_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_para_spots_adjust_checkbutton_toggled (GtkCellRendererToggle *renderer, gchar *path, gpointer user_data)
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

G_MODULE_EXPORT void on_phoebe_para_spots_units_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GtkWidget *lat_spinbutton = gui_widget_lookup ("phoebe_para_spots_lat_spinbutton")->gtk;
	GtkWidget *latstep_spinbutton = gui_widget_lookup ("phoebe_para_spots_latstep_spinbutton")->gtk;
	GtkWidget *latmin_spinbutton = gui_widget_lookup ("phoebe_para_spots_latmin_spinbutton")->gtk;
	GtkWidget *latmax_spinbutton = gui_widget_lookup ("phoebe_para_spots_latmax_spinbutton")->gtk;
	GtkAdjustment *lat_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(lat_spinbutton));
	GtkAdjustment *latstep_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(latstep_spinbutton));
	GtkAdjustment *latmin_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(latmin_spinbutton));
	GtkAdjustment *latmax_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(latmax_spinbutton));

	GtkWidget *lon_spinbutton = gui_widget_lookup ("phoebe_para_spots_lon_spinbutton")->gtk;
	GtkWidget *lonstep_spinbutton = gui_widget_lookup ("phoebe_para_spots_lonstep_spinbutton")->gtk;
	GtkWidget *lonmin_spinbutton = gui_widget_lookup ("phoebe_para_spots_lonmin_spinbutton")->gtk;
	GtkWidget *lonmax_spinbutton = gui_widget_lookup ("phoebe_para_spots_lonmax_spinbutton")->gtk;
	GtkAdjustment *lon_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(lon_spinbutton));
	GtkAdjustment *lonstep_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(lonstep_spinbutton));
	GtkAdjustment *lonmin_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(lonmin_spinbutton));
	GtkAdjustment *lonmax_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(lonmax_spinbutton));

	GtkWidget *rad_spinbutton = gui_widget_lookup ("phoebe_para_spots_rad_spinbutton")->gtk;
	GtkWidget *radstep_spinbutton = gui_widget_lookup ("phoebe_para_spots_radstep_spinbutton")->gtk;
	GtkWidget *radmin_spinbutton = gui_widget_lookup ("phoebe_para_spots_radmin_spinbutton")->gtk;
	GtkWidget *radmax_spinbutton = gui_widget_lookup ("phoebe_para_spots_radmax_spinbutton")->gtk;
	GtkAdjustment *rad_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(rad_spinbutton));
	GtkAdjustment *radstep_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(radstep_spinbutton));
	GtkAdjustment *radmin_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(radmin_spinbutton));
	GtkAdjustment *radmax_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(radmax_spinbutton));

	double change_factor;

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0) {
		/* Radians */
		change_factor = M_PI/180.0;

		lat_adjustment->upper = M_PI;
		latstep_adjustment->upper = M_PI;
		latmin_adjustment->upper = M_PI;
		latmax_adjustment->upper = M_PI;
		lat_adjustment->step_increment = 0.02;
		latstep_adjustment->step_increment = 0.02;
		latmin_adjustment->step_increment = 0.02;
		latmax_adjustment->step_increment = 0.02;

		lon_adjustment->upper = 2*M_PI;
		lonstep_adjustment->upper = 2*M_PI;
		lonmin_adjustment->upper = 2*M_PI;
		lonmax_adjustment->upper = 2*M_PI;
		lon_adjustment->step_increment = 0.02;
		lonstep_adjustment->step_increment = 0.02;
		lonmin_adjustment->step_increment = 0.02;
		lonmax_adjustment->step_increment = 0.02;

		rad_adjustment->upper = M_PI;
		radstep_adjustment->upper = M_PI;
		radmin_adjustment->upper = M_PI;
		radmax_adjustment->upper = M_PI;
		rad_adjustment->step_increment = 0.02;
		radstep_adjustment->step_increment = 0.02;
		radmin_adjustment->step_increment = 0.02;
		radmax_adjustment->step_increment = 0.02;
	}
	else {
		/* Degrees */
		change_factor = 180.0/M_PI;

		lat_adjustment->upper = 180;
		latstep_adjustment->upper = 180;
		latmin_adjustment->upper = 180;
		latmax_adjustment->upper = 180;
		lat_adjustment->step_increment = 1;
		latstep_adjustment->step_increment = 1;
		latmin_adjustment->step_increment = 1;
		latmax_adjustment->step_increment = 1;

		lon_adjustment->upper = 360;
		lonstep_adjustment->upper = 360;
		lonmin_adjustment->upper = 360;
		lonmax_adjustment->upper = 360;
		lon_adjustment->step_increment = 1;
		lonstep_adjustment->step_increment = 1;
		lonmin_adjustment->step_increment = 1;
		lonmax_adjustment->step_increment = 1;

		rad_adjustment->upper = 180;
		radstep_adjustment->upper = 180;
		radmin_adjustment->upper = 180;
		radmax_adjustment->upper = 180;
		rad_adjustment->step_increment = 1;
		radstep_adjustment->step_increment = 1;
		radmin_adjustment->step_increment = 1;
		radmax_adjustment->step_increment = 1;
	}

	/* if (phoebe_para_spots_units_combobox_init) { */
		GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
		GtkTreeModel *model = gtk_tree_view_get_model((GtkTreeView*)phoebe_para_spots_treeview);
		double lat, latstep, latmin, latmax;
		double lon, lonstep, lonmin, lonmax;
		double rad, radstep, radmin, radmax;
		GtkTreeIter iter;
		int state = gtk_tree_model_get_iter_first(model, &iter);

		while (state) {
			gtk_tree_model_get(model, &iter,
				SPOTS_COL_LAT,          &lat,
				SPOTS_COL_LATSTEP,      &latstep,
				SPOTS_COL_LATMIN,       &latmin,
				SPOTS_COL_LATMAX,       &latmax,
				SPOTS_COL_LON,          &lon,
				SPOTS_COL_LONSTEP,      &lonstep,
				SPOTS_COL_LONMIN,       &lonmin,
				SPOTS_COL_LONMAX,       &lonmax,
				SPOTS_COL_RAD,          &rad,
				SPOTS_COL_RADSTEP,      &radstep,
				SPOTS_COL_RADMIN,       &radmin,
				SPOTS_COL_RADMAX,       &radmax, -1);

			gtk_list_store_set((GtkListStore*)model, &iter,
				SPOTS_COL_LAT,          lat * change_factor,
				SPOTS_COL_LATSTEP,      latstep * change_factor,
				SPOTS_COL_LATMIN,       latmin * change_factor,
				SPOTS_COL_LATMAX,       latmax * change_factor,
				SPOTS_COL_LON,          lon * change_factor,
				SPOTS_COL_LONSTEP,      lonstep * change_factor,
				SPOTS_COL_LONMIN,       lonmin * change_factor,
				SPOTS_COL_LONMAX,       lonmax * change_factor,
				SPOTS_COL_RAD,          rad * change_factor,
				SPOTS_COL_RADSTEP,      radstep * change_factor,
				SPOTS_COL_RADMIN,       radmin * change_factor,
				SPOTS_COL_RADMAX,       radmax * change_factor, -1);
			state = gtk_tree_model_iter_next(model, &iter);
		}

		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lat_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(lat_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latstep_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(latstep_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latmin_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(latmin_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(latmax_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(latmax_spinbutton)) * change_factor);

		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lon_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(lon_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonstep_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(lonstep_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonmin_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(lonmin_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lonmax_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(lonmax_spinbutton)) * change_factor);

		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rad_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(rad_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radstep_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(radstep_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radmin_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(radmin_spinbutton)) * change_factor);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(radmax_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(radmax_spinbutton)) * change_factor);
	/* }
	 else
		phoebe_para_spots_units_combobox_init = TRUE; */
}

void gui_adjust_spot_parameter(char *par_name, int index, bool tba)
{
	PHOEBE_parameter *tba_par;
	char tba_par_name[80];

	sprintf(tba_par_name, "%s_tba", par_name);
	tba_par = phoebe_parameter_lookup(tba_par_name);
	phoebe_parameter_set_value (tba_par, index, tba);

	gui_get_value_from_widget(gui_widget_lookup("phoebe_para_spots_treeview"));
	gui_fill_sidesheet_fit_treeview ();
	gui_fill_fitt_mf_treeview();
}

G_MODULE_EXPORT void on_phoebe_para_spots_latadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATADJUST, gtk_toggle_button_get_active(togglebutton), -1);
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
		gui_adjust_spot_parameter("phoebe_spots_colatitude", atoi (gtk_tree_model_get_string_from_iter (model, &iter)), gtk_toggle_button_get_active(togglebutton));
    }
}

G_MODULE_EXPORT void on_phoebe_para_spots_lat_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LAT, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_latstep_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATSTEP, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_latmin_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATMIN, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_latmax_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LATMAX, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_lonadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONADJUST, gtk_toggle_button_get_active(togglebutton), -1);
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
		gui_adjust_spot_parameter("phoebe_spots_longitude", atoi (gtk_tree_model_get_string_from_iter (model, &iter)), gtk_toggle_button_get_active(togglebutton));
    }
}

G_MODULE_EXPORT void on_phoebe_para_spots_lon_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LON, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_lonstep_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONSTEP, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_lonmin_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONMIN, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_lonmax_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_LONMAX, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_radadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADADJUST, gtk_toggle_button_get_active(togglebutton), -1);
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
		gui_adjust_spot_parameter("phoebe_spots_radius", atoi (gtk_tree_model_get_string_from_iter (model, &iter)), gtk_toggle_button_get_active(togglebutton));
    }
}

G_MODULE_EXPORT void on_phoebe_para_spots_rad_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RAD, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_radstep_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADSTEP, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_radmin_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADMIN, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_radmax_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_RADMAX, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_tempadjust_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPADJUST, gtk_toggle_button_get_active(togglebutton), -1);
		if(gui_spots_parameters_marked_tba() > 0 )
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, TRUE, -1);
		else
			gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_ADJUST, FALSE, -1);
		gui_adjust_spot_parameter("phoebe_spots_tempfactor", atoi (gtk_tree_model_get_string_from_iter (model, &iter)), gtk_toggle_button_get_active(togglebutton));
    }
}

G_MODULE_EXPORT void on_phoebe_para_spots_temp_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMP, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_tempstep_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPSTEP, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_tempmin_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkTreeSelection *selection;
    GtkTreeModel     *model;
    GtkTreeIter       iter;

	GtkWidget *phoebe_para_spots_treeview = gui_widget_lookup("phoebe_para_spots_treeview")->gtk;
    selection = gtk_tree_view_get_selection((GtkTreeView*)phoebe_para_spots_treeview);
    if (gtk_tree_selection_get_selected(selection, &model, &iter))
		gtk_list_store_set((GtkListStore*)model, &iter, SPOTS_COL_TEMPMIN, gtk_spin_button_get_value(spinbutton), -1);
}

G_MODULE_EXPORT void on_phoebe_para_spots_tempmax_spinbutton_value_changed(GtkSpinButton *spinbutton, gpointer user_data)
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


G_MODULE_EXPORT gboolean on_phoebe_window_delete_event (GtkWidget *widget, GdkEvent *event, gpointer user_data)
{
    if(gui_warning("Quit PHOEBE?", "By quitting Phoebe all unsaved data will be lost. Are you sure you want to quit?") == 1)
    	gtk_main_quit();
    return TRUE;
}

G_MODULE_EXPORT void on_phoebe_file_new_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{

}

G_MODULE_EXPORT void on_phoebe_file_open_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = gui_open_parameter_file ();

	if( status == SUCCESS ){
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_error (status));
}

G_MODULE_EXPORT void on_phoebe_file_save_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = 0;

	status = gui_get_values_from_widgets();

	if(PHOEBE_FILEFLAG)
		status = phoebe_save_parameter_file(PHOEBE_FILENAME);
	else
		status = gui_save_parameter_file ();

	if( status != SUCCESS )
		printf ("%s\n", phoebe_error (status));
}

G_MODULE_EXPORT void on_phoebe_file_saveas_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = 0;

	status = gui_get_values_from_widgets();
	status = gui_save_parameter_file ();

	if( status != SUCCESS )
		printf ("%s", phoebe_error (status));
}

G_MODULE_EXPORT void on_phoebe_file_import_bm3_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gint status, import = -1;

	gchar *glade_xml_file    = g_build_filename (PHOEBE_GLADE_XML_DIR, "phoebe_import_bm3.glade", NULL);
	gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_import_bm3 = glade_xml_new (glade_xml_file, NULL, NULL);
	GtkWidget *phoebe_import_bm3_dialog = glade_xml_get_widget (phoebe_import_bm3, "phoebe_import_bm3_dialog");
	GtkWidget *bm3entry = glade_xml_get_widget (phoebe_import_bm3, "phoebe_import_bm3_input_file_chooser");
	GtkWidget *dataentry = glade_xml_get_widget (phoebe_import_bm3, "phoebe_import_bm3_data_file_chooser");

	g_object_unref (phoebe_import_bm3);

	gtk_window_set_icon (GTK_WINDOW (phoebe_import_bm3_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));

	while (import != SUCCESS) {
		status = gtk_dialog_run ((GtkDialog *) phoebe_import_bm3_dialog);
		if (status == GTK_RESPONSE_OK) {
			gchar *bm3file = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (bm3entry));
			gchar *datafile = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dataentry));
			import = phoebe_parameter_file_import_bm3 (bm3file, datafile);
			if (import == SUCCESS) {
				gui_reinit_treeviews ();
				gui_set_values_to_widgets ();
				gtk_widget_destroy (phoebe_import_bm3_dialog);
			}
			else {
				gui_notice ("BM3 file import failed", "The passed Binary Maker 3 parameter file failed to open.\n");
			}
		}
		else /* if clicked on Cancel: */ {
			import = SUCCESS;
			gtk_widget_destroy (phoebe_import_bm3_dialog);
		}
	}
}

G_MODULE_EXPORT void on_phoebe_file_quit_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
    if(gui_warning("Quit PHOEBE?", "By quitting Phoebe all unsaved data will be lost. Are you sure you want to quit?") == 1)
    	gtk_main_quit();
}

G_MODULE_EXPORT void on_phoebe_settings_configuration_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gui_show_configuration_dialog();
}

G_MODULE_EXPORT void on_phoebe_help_about_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
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


G_MODULE_EXPORT void on_phoebe_lc_plot_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_fitting_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}

G_MODULE_EXPORT void on_phoebe_scripter_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{

}

G_MODULE_EXPORT void on_phoebe_settings_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	gui_show_configuration_dialog();
}

G_MODULE_EXPORT void on_phoebe_settings_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GtkWidget *filechooserbutton = GTK_WIDGET(user_data);
	gtk_widget_set_sensitive (filechooserbutton, gtk_toggle_button_get_active(togglebutton));
}

G_MODULE_EXPORT void on_phoebe_quit_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	if(gui_warning("Quit PHOEBE?", "By quitting Phoebe all unsaved data will be lost. Are you sure you want to quit?") == 1)
    	gtk_main_quit();
}

G_MODULE_EXPORT void on_phoebe_open_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status = gui_open_parameter_file ();

	if( status == SUCCESS ){
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_error (status));
}

G_MODULE_EXPORT void on_phoebe_save_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	int status;
	bool confirm;

	status = gui_get_values_from_widgets ();

	if (PHOEBE_FILEFLAG) {
		phoebe_config_entry_get ("GUI_CONFIRM_ON_OVERWRITE", &confirm);
		if (!confirm)
			status = phoebe_save_parameter_file (PHOEBE_FILENAME);
		else {
			char *message = phoebe_concatenate_strings ("Do you want to overwrite file ", PHOEBE_FILENAME, "?", NULL);
			int answer = gui_question ("Overwrite?", message);
			free (message);

			if (answer == 1)
				status = phoebe_save_parameter_file (PHOEBE_FILENAME);
			else
				status = gui_save_parameter_file ();
		}
	}
	else
		status = gui_save_parameter_file ();

	phoebe_gui_debug ("In on_phoebe_save_toolbutton_clicked\n");
	phoebe_gui_debug ("\tPHOEBE_FILEFLAG = %d\n", PHOEBE_FILEFLAG);
	phoebe_gui_debug ("\tPHOEBE_FILENAME = %s\n", PHOEBE_FILENAME);

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


G_MODULE_EXPORT void on_phoebe_para_lum_el3_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_para_lum_el3_edit();
}

G_MODULE_EXPORT void on_phoebe_para_lum_el3_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lum_el3_edit();
}

/* ******************************************************************** *
 *
 *                    phoebe_para_lum_weighting events
 *
 * ******************************************************************** */


G_MODULE_EXPORT void on_phoebe_para_lum_weighting_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_fitt_levelweight_edit();
}

G_MODULE_EXPORT void on_phoebe_para_lum_weighting_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_fitt_levelweight_edit();
}


/* ******************************************************************** *
 *
 *              	phoebe_para_lc_coefficents_treeview events
 *
 * ******************************************************************** */


G_MODULE_EXPORT void on_phoebe_para_ld_lccoefs_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_para_lc_coefficents_edit();
}

G_MODULE_EXPORT void on_phoebe_para_ld_lccoefs_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lc_coefficents_edit();
}


/* ******************************************************************** *
 *
 *              	phoebe_para_lc_coefficents_treeview events
 *
 * ******************************************************************** */


G_MODULE_EXPORT void on_phoebe_para_ld_rvcoefs_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_para_rv_coefficents_edit();
}

G_MODULE_EXPORT void on_phoebe_para_ld_rvcoefs_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_rv_coefficents_edit();
}


/* ******************************************************************** *
 *
 *                    phoebe_window detach events
 *
 * ******************************************************************** */


G_MODULE_EXPORT void on_phoebe_sidesheet_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_sidesheet_vbox");
	GUI_widget *parent = gui_widget_lookup ("phoebe_sidesheet_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_SIDESHEET_IS_DETACHED, "PHOEBE - Data sheets", 300, 600);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_fitt_fitting_detach_button_clicked (GtkButton *button, gpointer user_data)
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


G_MODULE_EXPORT void on_phoebe_rv_plot_options_x_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GUI_widget *rv_xstart_label			= gui_widget_lookup("phoebe_rv_plot_options_phstart_label");
	GUI_widget *rv_xend_label			= gui_widget_lookup("phoebe_rv_plot_options_phend_label");

	GUI_widget *rv_xstart_spinbutton	= gui_widget_lookup("phoebe_rv_plot_options_phstart_spinbutton");
	GUI_widget *rv_xend_spinbutton		= gui_widget_lookup("phoebe_rv_plot_options_phend_spinbutton");


	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0){
		/* Phase */
		gtk_label_set_text(GTK_LABEL(rv_xstart_label->gtk), "Phase start:");
		gtk_label_set_text(GTK_LABEL(rv_xend_label->gtk), "Phase end:");
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(rv_xstart_spinbutton->gtk), -10.0, 10.0);
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(rv_xend_spinbutton->gtk), -10.0, 10.0);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xstart_spinbutton->gtk), -0.6);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xend_spinbutton->gtk), 0.6);
	}
	else{
		/* Time */
		gtk_label_set_text(GTK_LABEL(rv_xstart_label->gtk), "Time start: ");
		gtk_label_set_text(GTK_LABEL(rv_xend_label->gtk), "Time end: ");
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(rv_xstart_spinbutton->gtk), -1e10, 1e10);
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(rv_xend_spinbutton->gtk), -1e10, 1e10);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xstart_spinbutton->gtk), -0.1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(rv_xend_spinbutton->gtk), 1.1);
	}
}

G_MODULE_EXPORT void on_phoebe_lc_plot_options_x_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	GUI_widget *lc_xstart_label 		= gui_widget_lookup("phoebe_lc_plot_options_phstart_label");
	GUI_widget *lc_xend_label 			= gui_widget_lookup("phoebe_lc_plot_options_phend_label");

	GUI_widget *lc_xstart_spinbutton	= gui_widget_lookup("phoebe_lc_plot_options_phstart_spinbutton");
	GUI_widget *lc_xend_spinbutton		= gui_widget_lookup("phoebe_lc_plot_options_phend_spinbutton");

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0){
		/* Phase */
		gtk_label_set_text(GTK_LABEL(lc_xstart_label->gtk), "Phase start:");
		gtk_label_set_text(GTK_LABEL(lc_xend_label->gtk), "Phase end:");
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(lc_xstart_spinbutton->gtk), -10.0, 10.0);
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(lc_xend_spinbutton->gtk), -10.0, 10.0);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xstart_spinbutton->gtk), -0.6);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xend_spinbutton->gtk), 0.6);
	}
	else{
		/* Time */
		gtk_label_set_text(GTK_LABEL(lc_xstart_label->gtk), "Time start: ");
		gtk_label_set_text(GTK_LABEL(lc_xend_label->gtk), "Time end: ");
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(lc_xstart_spinbutton->gtk), -1e10, 1e10);
		gtk_spin_button_set_range(GTK_SPIN_BUTTON(lc_xend_spinbutton->gtk), -1e10, 1e10);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xstart_spinbutton->gtk), -0.1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(lc_xend_spinbutton->gtk), 1.1);
	}
}

G_MODULE_EXPORT void on_phoebe_lc_plot_options_obs_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GUI_widget *combobox = gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox");
	gtk_widget_set_sensitive (combobox->gtk, gtk_toggle_button_get_active(togglebutton));
	if(gtk_combo_box_get_active(GTK_COMBO_BOX(combobox->gtk))==-1) gtk_combo_box_set_active(GTK_COMBO_BOX(combobox->gtk),0);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_options_obs_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	GUI_widget *combobox = gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox");
	gtk_widget_set_sensitive (combobox->gtk, gtk_toggle_button_get_active(togglebutton));
	if(gtk_combo_box_get_active(GTK_COMBO_BOX(combobox->gtk))==-1) gtk_combo_box_set_active(GTK_COMBO_BOX(combobox->gtk),0);
}

//extern gdouble lc_zoom;
//extern gint lc_zoom_level;
//extern gdouble lc_x_offset;
//extern gdouble lc_y_offset;
//
//gdouble lc_zoom=0.0;
//gint 	lc_zoom_level=0;
//gdouble lc_x_offset=0.0;
//gdouble lc_y_offset=0.0;

int phoebe_gui_lc_plot (gdouble x_offset, gdouble y_offset, gdouble zoom)
{
	int lcno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_lcno");
	phoebe_parameter_get_value(par, &lcno);

	if(lcno > 0){
		gui_update_ld_coefficients_when_needed();
		gui_get_values_from_widgets();
		gui_plot_lc_using_gnuplot(x_offset, y_offset, zoom);

		LOGG_VALUES_NEED_RECALCULATING = FALSE;
		gui_fill_sidesheet_res_treeview();
	}
	else
		gui_notice ("No light curves have been defined", "To plot a light curve, you need to define it first in the Data tab.");

	return SUCCESS;
}

G_MODULE_EXPORT void on_phoebe_lc_plot_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset;
	double lc_y_offset;
	double lc_zoom;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);

	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_save_button_clicked (GtkButton *button, gpointer user_data)
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
										 	GTK_FILE_CHOOSER_ACTION_SAVE,
										 	GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  	GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
										  	NULL);

		/* gtk_file_chooser_set_do_overwrite_confirmation (GTK_FILE_CHOOSER (dialog), TRUE); */

		gchar *dir;
		phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

		gtk_file_chooser_set_current_folder((GtkFileChooser*)dialog, dir);

    	gtk_window_set_icon (GTK_WINDOW(dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

		gchar *filename = gui_get_filename_with_overwrite_confirmation(dialog, "Save LC Curves to ASCII File");
		if (filename){
			gui_update_ld_coefficients_when_needed();
			gui_get_values_from_widgets();
			status =gui_plot_lc_to_ascii (filename);
			LOGG_VALUES_NEED_RECALCULATING = FALSE;

			g_free (filename);
		}

		gtk_widget_destroy (dialog);

		gui_fill_sidesheet_res_treeview();
	}
	else
		gui_notice ("No light curves have been defined", "To save a light curve, you need to define it first in the Data tab.");

}

G_MODULE_EXPORT void on_phoebe_lc_plot_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_image_set_from_pixbuf(GTK_IMAGE(gui_widget_lookup ("phoebe_lc_plot_image")->gtk), NULL);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_reset_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;
	int lc_zoom_level = 0;

	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), lc_x_offset);
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), lc_y_offset);
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), lc_zoom);
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_zoom_level"), lc_zoom_level);

	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_right_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);

	lc_x_offset+=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), lc_x_offset);

	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_up_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);

	lc_y_offset+=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), lc_y_offset);

	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_left_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);

	lc_x_offset-=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), lc_x_offset);

	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_down_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);

	lc_y_offset-=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), lc_y_offset);

	phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_zoomin_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;
	int lc_zoom_level = 0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom_level"), &lc_zoom_level);

	if (lc_zoom_level<5){
		lc_zoom-=0.1;
		lc_zoom_level+=1;

		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), lc_zoom);
		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_zoom_level"), lc_zoom_level);
		phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
	}
}

G_MODULE_EXPORT void on_phoebe_lc_plot_controls_zoomout_button_clicked (GtkButton *button, gpointer user_data)
{
	double lc_x_offset = 0.0;
	double lc_y_offset = 0.0;
	double lc_zoom = 0.0;
	int lc_zoom_level = 0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_x_offset"), &lc_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_y_offset"), &lc_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), &lc_zoom);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_lc_plot_zoom_level"), &lc_zoom_level);

	if (lc_zoom_level>-5){
		lc_zoom+=0.1;
		lc_zoom_level-=1;

		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_zoom"), lc_zoom);
		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_lc_plot_zoom_level"), lc_zoom_level);
		phoebe_gui_lc_plot (lc_x_offset, lc_y_offset, lc_zoom);
	}
}

//extern gdouble rv_zoom;
//extern gint rv_zoom_level;
//extern gdouble rv_x_offset;
//extern gdouble rv_y_offset;
//
//gdouble rv_zoom=0.0;
//gint 	rv_zoom_level=0;
//gdouble rv_x_offset=0.0;
//gdouble rv_y_offset=0.0;

int phoebe_gui_rv_plot (gdouble x_offset, gdouble y_offset, gdouble zoom)
{
	int rvno;

	PHOEBE_parameter *par = phoebe_parameter_lookup("phoebe_rvno");
	phoebe_parameter_get_value(par, &rvno);

	if(rvno > 0){
		gui_get_values_from_widgets();
		gui_plot_rv_using_gnuplot (x_offset, y_offset, zoom);
		LOGG_VALUES_NEED_RECALCULATING = FALSE;

		gui_fill_sidesheet_res_treeview();
	}
	else
		gui_notice ("No RV curves have been defined", "To plot an RV curve, you need to define it first in the Data tab.");

	return SUCCESS;
}

G_MODULE_EXPORT void on_phoebe_rv_plot_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset;
	double rv_y_offset;
	double rv_zoom;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);

	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_save_button_clicked (GtkButton *button, gpointer user_data)
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
										 	GTK_FILE_CHOOSER_ACTION_SAVE,
										 	GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  	GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
										  	NULL);

		/* gtk_file_chooser_set_do_overwrite_confirmation (GTK_FILE_CHOOSER (dialog), TRUE); */

		gchar *dir;
		phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);

		gtk_file_chooser_set_current_folder((GtkFileChooser*)dialog, dir);

    	gtk_window_set_icon (GTK_WINDOW(dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

		gchar *filename = gui_get_filename_with_overwrite_confirmation(dialog, "Save RV Curves to ASCII File");
		if (filename){
			gui_get_values_from_widgets();
			status =gui_plot_rv_to_ascii (filename);
			LOGG_VALUES_NEED_RECALCULATING = FALSE;

			g_free (filename);
		}

		gtk_widget_destroy (dialog);

		gui_fill_sidesheet_res_treeview();
	}
	else
		gui_notice ("No RV curves have been defined", "To save an RV curve, you need to define it first in the Data tab.");
}

G_MODULE_EXPORT void on_phoebe_rv_plot_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_image_set_from_pixbuf(GTK_IMAGE(gui_widget_lookup ("phoebe_rv_plot_image")->gtk), NULL);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_reset_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;
	int rv_zoom_level = 0;

	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), rv_x_offset);
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), rv_y_offset);
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), rv_zoom);
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_zoom_level"), rv_zoom_level);

	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_right_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);

	rv_x_offset+=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), rv_x_offset);

	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_up_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);

	rv_y_offset+=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), rv_y_offset);

	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_left_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);

	rv_x_offset-=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), rv_x_offset);

	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_down_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);

	rv_y_offset-=0.1;
	phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), rv_y_offset);

	phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_zoomin_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;
	int rv_zoom_level = 0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom_level"), &rv_zoom_level);

	if (rv_zoom_level<5){
		rv_zoom-=0.1;
		rv_zoom_level+=1;

		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), rv_zoom);
		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_zoom_level"), rv_zoom_level);
		phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
	}
}

G_MODULE_EXPORT void on_phoebe_rv_plot_controls_zoomout_button_clicked (GtkButton *button, gpointer user_data)
{
	double rv_x_offset = 0.0;
	double rv_y_offset = 0.0;
	double rv_zoom = 0.0;
	int rv_zoom_level = 0;

	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_x_offset"), &rv_x_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_y_offset"), &rv_y_offset);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), &rv_zoom);
	phoebe_parameter_get_value(phoebe_parameter_lookup("gui_rv_plot_zoom_level"), &rv_zoom_level);

	if (rv_zoom_level>-5){
		rv_zoom+=0.1;
		rv_zoom_level-=1;

		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_zoom"), rv_zoom);
		phoebe_parameter_set_value(phoebe_parameter_lookup("gui_rv_plot_zoom_level"), rv_zoom_level);
		phoebe_gui_rv_plot (rv_x_offset, rv_y_offset, rv_zoom);
	}
}

G_MODULE_EXPORT void on_phoebe_star_shape_plot_button_clicked (GtkButton *button, gpointer user_data)
{
		gui_update_ld_coefficients_when_needed();
		gui_get_values_from_widgets();
		gui_plot_eb_using_gnuplot();
}

G_MODULE_EXPORT void on_phoebe_star_shape_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_image_set_from_pixbuf(GTK_IMAGE(gui_widget_lookup ("phoebe_eb_plot_image")->gtk), NULL);
}

G_MODULE_EXPORT void on_phoebe_star_shape_phase_spinbutton_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	GtkWidget *phoebe_star_shape_autoupdate_checkbutton = gui_widget_lookup("phoebe_star_shape_autoupdate_checkbutton")->gtk;

	if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(phoebe_star_shape_autoupdate_checkbutton))){
		gui_update_ld_coefficients_when_needed();
		gui_get_values_from_widgets();
		gui_plot_eb_using_gnuplot();
	}
}



/* ******************************************************************** *
 *
 *                    potential calculator dialog
 *
 * ******************************************************************** */


G_MODULE_EXPORT void on_phoebe_pot_calc_close_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_widget_destroy (GTK_WIDGET(user_data));
}

G_MODULE_EXPORT void on_phoebe_pot_calc_update_button_clicked (GtkButton *button, gpointer user_data)
{
	gint sel = GPOINTER_TO_INT (user_data);
	GtkWidget *pot_spinbutton=NULL;

	if (sel==1){pot_spinbutton = gui_widget_lookup("phoebe_para_comp_phsv_spinbutton")->gtk;}
	if (sel==2){pot_spinbutton = gui_widget_lookup("phoebe_para_comp_pcsv_spinbutton")->gtk;}

	gdouble pot = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_pot_spinbutton")));
	gtk_spin_button_set_value (GTK_SPIN_BUTTON(pot_spinbutton), pot);
}

G_MODULE_EXPORT
void on_phoebe_pot_calc_calculate_button_clicked (GtkButton *button, gpointer user_data)
{
	gboolean circ 	= gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(g_object_get_data (G_OBJECT (button), "data_circ_checkbutton")));
	gdouble d		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_d_spinbutton")));
	gdouble rm		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_rm_spinbutton")));
	gdouble r		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_r_spinbutton")));
	gdouble f		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_f_spinbutton")));
	gdouble lambda	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_lambda_spinbutton")));
	gdouble nu		= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_nu_spinbutton")));

	gdouble pot;

	gchar *source = phoebe_strdup ((gchar *) g_object_get_data (G_OBJECT (button), "star"));

	if (strcmp (source, "primary") == 0)
		pot = phoebe_calculate_pot1 ((int)(!circ), d, rm, r, f, lambda, nu);
	else
		pot = phoebe_calculate_pot2 ((int)(!circ), d, rm, r, f, lambda, nu);

	gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_pot_spinbutton")), pot);

	free (source);
}

G_MODULE_EXPORT
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

	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "star", "primary");

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

G_MODULE_EXPORT
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

	g_object_set_data (G_OBJECT (phoebe_pot_calc_calculate_button), "star", "secondary");

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

	g_object_set_data (G_OBJECT (phoebe_pot_calc_update_button), "data_pot_spinbutton", (gpointer) phoebe_pot_calc_pot_spinbutton);

	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_close_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_close_button_clicked), (gpointer) phoebe_pot_calc_dialog);
	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_update_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_update_button_clicked), (gpointer) 2);
	g_signal_connect (GTK_WIDGET(phoebe_pot_calc_calculate_button), "clicked", G_CALLBACK (on_phoebe_pot_calc_calculate_button_clicked), NULL);

	gtk_widget_show (phoebe_pot_calc_dialog);
}


/* ******************************************************************** *
 *
 *                    ld interpolation dialog
 *
 * ******************************************************************** */


G_MODULE_EXPORT void on_phoebe_ld_dialog_close_button_clicked (GtkButton *button, gpointer user_data)
{
	gtk_widget_destroy (GTK_WIDGET(user_data));
}

G_MODULE_EXPORT void on_phoebe_ld_dialog_interpolate_button_clicked (GtkButton *button, gpointer user_data)
{
	double tavh 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_tavh_spinbutton")));
	double tavc 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_tavc_spinbutton")));
	double logg1 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_logg1_spinbutton")));
	double logg2 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_logg2_spinbutton")));
	double met1 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_met1_spinbutton")));
	double met2 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_met2_spinbutton")));

	int ldlawindex	= gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_law_combobox")));
	char* ldlaw 	= strdup(phoebe_parameter_lookup ("phoebe_ld_model")->menu->option[ldlawindex]);

	int index = gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_id_combobox")));
	PHOEBE_passband *passband;

	if (index == 0) {
		/* Bolometric LD coefficients */
		passband = phoebe_bolometric_passband();
	}
	else {
		char *id;
		phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lc_id"), index - 1, &id);
		passband = phoebe_passband_lookup_by_id(id);
	}

	double x1, x2, y1, y2;
	char *title = "LD coefficients interpolation";

	switch (phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met1, tavh, logg1, &x1, &y1)) {
		case SUCCESS:
			gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_x1_spinbutton")), x1);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_y1_spinbutton")), y1);
			break;
		case ERROR_LD_TABLES_MISSING:
			gui_notice(title, "Van Hamme tables are missing");
			return;
		case ERROR_LD_PARAMS_OUT_OF_RANGE:
			gui_notice(title, "Parameters for the primary component are out of range");
			break;
	}

	switch (phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met2, tavc, logg2, &x2, &y2)) {
		case SUCCESS:
			gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_x2_spinbutton")), x2);
			gtk_spin_button_set_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_y2_spinbutton")), y2);
			break;
		case ERROR_LD_PARAMS_OUT_OF_RANGE:
			gui_notice(title, "Parameters for the secondary component are out of range");
			break;
	}
}

G_MODULE_EXPORT void on_phoebe_ld_dialog_update_button_clicked (GtkButton *button, gpointer user_data)
{
	double x1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_x1_spinbutton")));
	double x2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_x2_spinbutton")));
	double y1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_y1_spinbutton")));
	double y2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_y2_spinbutton")));

	int index = gtk_combo_box_get_active (GTK_COMBO_BOX (g_object_get_data (G_OBJECT (button), "data_id_combobox")));
	char *id, *id_in_model;

	if (index == 0) {
		/* Bolometric LD coefficients */
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_primx_spinbutton")->gtk), x1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_secx_spinbutton")->gtk), x2);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_primy_spinbutton")->gtk), y1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_secy_spinbutton")->gtk), y2);
	}
	else {
		phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_lc_id"), index - 1, &id);

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
}

static void gui_ld_filter_cell_data_func (GtkCellLayout *cell_layout, GtkCellRenderer *renderer, GtkTreeModel *model, GtkTreeIter *iter, gpointer data)
{
	if(gtk_tree_model_iter_has_child(model, iter)) g_object_set(renderer, "sensitive", FALSE, NULL);
	else g_object_set(renderer, "sensitive", TRUE, NULL);
}

int gui_init_ld_filter_combobox (GtkWidget *combo_box)
{
	GtkTreeStore 		*store;
	GtkTreeIter 		 toplevel;
	GtkCellRenderer 	*renderer;

	int i, lcno;
	char *cid;

	store = gtk_tree_store_new (2, G_TYPE_STRING, G_TYPE_INT);

	gtk_combo_box_set_model (GTK_COMBO_BOX(combo_box), GTK_TREE_MODEL (store));
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (combo_box));

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT(combo_box), renderer, TRUE);
	gtk_cell_layout_add_attribute (GTK_CELL_LAYOUT(combo_box), renderer, "text", 0);

	gtk_cell_layout_set_cell_data_func(GTK_CELL_LAYOUT(combo_box), renderer, gui_ld_filter_cell_data_func, NULL, NULL);

	gtk_tree_store_append (store, &toplevel, NULL);
	gtk_tree_store_set (store, &toplevel, 0, "Bolometric", 1, 0, -1);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	for (i = 0; i < lcno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_id"), i, &cid);
		gtk_tree_store_append (store, &toplevel, NULL);
		gtk_tree_store_set (store, &toplevel, 0, cid, 1, i + 1, -1);
	}
	g_object_unref (store);

	gtk_combo_box_set_active(GTK_COMBO_BOX(combo_box), 0);
	return SUCCESS;
}

G_MODULE_EXPORT void on_phoebe_para_ld_model_tables_vanhamme_button_clicked (GtkButton *button, gpointer user_data)
{
	GtkCellRenderer *renderer;

	PHOEBE_parameter *ldlaw = phoebe_parameter_lookup ("phoebe_ld_model");
	int optindex, optcount;

	double logg1, logg2;

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
	gtk_window_set_title (GTK_WINDOW(phoebe_ld_dialog), "PHOEBE - LD Coefficients Interpolation");

	renderer = gtk_cell_renderer_text_new ();
	gtk_cell_layout_clear (GTK_CELL_LAYOUT (phoebe_ld_dialog_id_combobox));
	gtk_cell_layout_pack_start (GTK_CELL_LAYOUT (phoebe_ld_dialog_id_combobox), renderer, TRUE);
	gui_init_ld_filter_combobox(phoebe_ld_dialog_id_combobox);

	optcount = ldlaw->menu->optno;
	for(optindex = 0; optindex < optcount; optindex++)
		gtk_combo_box_append_text(GTK_COMBO_BOX(phoebe_ld_dialog_law_combobox), strdup(ldlaw->menu->option[optindex]));

	gtk_combo_box_set_active(GTK_COMBO_BOX(phoebe_ld_dialog_law_combobox), gtk_combo_box_get_active(GTK_COMBO_BOX(gui_widget_lookup("phoebe_para_ld_model_combobox")->gtk)));

	gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_ld_dialog_tavh_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_comp_tavh_spinbutton")->gtk)));
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_ld_dialog_met1_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_comp_met1_spinbutton")->gtk)));
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_ld_dialog_tavc_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_comp_tavc_spinbutton")->gtk)));
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_ld_dialog_met2_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_comp_met2_spinbutton")->gtk)));

	phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_logg1"), &logg1);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_ld_dialog_logg1_spinbutton), logg1);

	phoebe_parameter_get_value(phoebe_parameter_lookup("phoebe_logg2"), &logg2);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(phoebe_ld_dialog_logg2_spinbutton), logg2);

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

void on_phoebe_settings_confirmation_save_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	/*
	 * This handler is invoked when Settings->Options->ConfirmOnOverwrite has
	 * been toggled. It changes the configuration parameter GUI_CONFIRM_ON_
	 * OVERWRITE.
	 */

	printf ("entered.\n");
	if (togglebutton->active == TRUE)
		phoebe_config_entry_set ("GUI_CONFIRM_ON_OVERWRITE", TRUE);
	else
		phoebe_config_entry_set ("GUI_CONFIRM_ON_OVERWRITE", FALSE);
}
