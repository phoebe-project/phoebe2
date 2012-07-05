#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <libgen.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_global.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_plotting.h"
#include "phoebe_gui_error_handling.h"
#include "phoebe_gui_build_config.h"

#ifdef PHOEBE_GUI_THREADS
#include <pthread.h>
#endif

bool LD_COEFFS_NEED_UPDATING = TRUE;
bool LOGG_VALUES_NEED_RECALCULATING = TRUE;

GtkWidget *GUI_DETACHED_LC_PLOT_WINDOW;
GtkWidget *GUI_DETACHED_RV_PLOT_WINDOW;
GtkWidget *GUI_DETACHED_FITTING_WINDOW;
GtkWidget *GUI_DETACHED_SIDESHEET_WINDOW;

gchar *GUI_SAVED_DATA_DIR;

double intern_angle_factor ()
{
	char *units;
	phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
	if (strcmp (units, "Radians") == 0)
		return 1.0;
	else
		return M_PI/180.0;
}

G_MODULE_EXPORT
void on_combo_box_selection_changed_get_index (GtkComboBox *combo, gpointer user_data)
{
	*((int *) user_data) = gtk_combo_box_get_active (combo);
	return;
}

G_MODULE_EXPORT
void on_combo_box_selection_changed_get_string (GtkComboBox *combo, gpointer user_data)
{
	*((const char **) user_data) = gtk_combo_box_get_active_text (combo);
	return;
}

G_MODULE_EXPORT
void on_spin_button_value_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	*((double *) user_data) = gtk_spin_button_get_value (spinbutton);
	return;
}

G_MODULE_EXPORT
void on_spin_button_intvalue_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	*((int *) user_data) = gtk_spin_button_get_value (spinbutton);
	return;
}

G_MODULE_EXPORT
void on_toggle_button_value_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	/* This is a generic callback that stores a state of the toggle button
	 * that emitted the signal to the boolean pointed by user_data.
	 */
	
	*((bool *) user_data) = gtk_toggle_button_get_active (togglebutton);
	return;
}

G_MODULE_EXPORT
void on_toggle_make_sensitive (GtkToggleButton *togglebutton, gpointer user_data)
{
	/* Glade has the definitions of togglebutton and user_data swapped; that
	 * is why the code below is actually right although it looks wrong!
	 */
	
	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (user_data)))
		gtk_widget_set_sensitive (GTK_WIDGET (togglebutton), TRUE);
	else
		gtk_widget_set_sensitive (GTK_WIDGET (togglebutton), FALSE);

	return;
}

G_MODULE_EXPORT
void on_toggle_make_unsensitive (GtkToggleButton *togglebutton, gpointer user_data)
{
	/* Glade has the definitions of togglebutton and user_data swapped; that
	 * is why the code below is actually right although it looks wrong!
	 */
	
	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (user_data)))
		gtk_widget_set_sensitive (GTK_WIDGET (togglebutton), FALSE);
	else
		gtk_widget_set_sensitive (GTK_WIDGET (togglebutton), TRUE);

	return;
}

G_MODULE_EXPORT void on_plot_controls_reset_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	data->x_left = data->x_ll;
	data->x_right = data->x_ul;
	data->y_top = data->y_max;
	data->y_bottom = data->y_min;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT void on_plot_controls_right_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	double x_shift = 0.1 * (data->x_right - data->x_left);
	data->x_left += x_shift;
	data->x_right += x_shift;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT void on_plot_controls_up_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	double y_shift = 0.1 * (data->y_top - data->y_bottom);
	data->y_top += y_shift;
	data->y_bottom += y_shift;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT void on_plot_controls_left_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	double x_shift = -0.1 * (data->x_right - data->x_left);
	data->x_left += x_shift;
	data->x_right += x_shift;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT void on_plot_controls_down_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	double y_shift = -0.1 * (data->y_top - data->y_bottom);
	data->y_top += y_shift;
	data->y_bottom += y_shift;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT void on_plot_controls_zoomin_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	double x_offset = (data->x_right - data->x_left)/4;
	double y_offset = (data->y_top - data->y_bottom)/4;

	data->x_left += x_offset;
	data->x_right -= x_offset;
	data->y_bottom += y_offset;
	data->y_top -= y_offset;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT void on_plot_controls_zoomout_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	double x_offset = (data->x_right - data->x_left)/2;
	double y_offset = (data->y_top - data->y_bottom)/2;

	data->x_left -= x_offset;
	data->x_right += x_offset;
	data->y_bottom -= y_offset;
	data->y_top += y_offset;
	gui_plot_area_refresh (data);
	return;
}

G_MODULE_EXPORT
void on_plot_save_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	FILE *output;

	GtkWidget *dialog;
	gchar *filename;
	gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	dialog = gtk_file_chooser_dialog_new ("Export figure to a text file",
		  	GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
		 	GTK_FILE_CHOOSER_ACTION_SAVE,
		 	GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
		  	GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
		  	NULL);

	gtk_window_set_default_size (GTK_WINDOW (dialog), 600, 450);

	if (!GUI_SAVED_DATA_DIR)
		phoebe_config_entry_get ("PHOEBE_DATA_DIR", &GUI_SAVED_DATA_DIR);

	gtk_file_chooser_set_current_folder ((GtkFileChooser*)dialog, GUI_SAVED_DATA_DIR);
   	gtk_window_set_icon (GTK_WINDOW(dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

	filename = gui_get_filename_with_overwrite_confirmation (dialog, "Save LC Curves to ASCII File");

	if (filename) {
#ifdef __MINGW32__
		GtkWidget *temp_window;
		if (PHOEBE_WINDOW_LC_PLOT_IS_DETACHED)
			temp_window = gui_show_temp_window ();
#endif
/*
		gui_update_ld_coefficients_when_needed ();
		gui_get_values_from_widgets ();
		status = gui_plot_lc_to_ascii (filename);
		LOGG_VALUES_NEED_RECALCULATING = FALSE;
		gui_fill_sidesheet_res_treeview ();
*/
		output = fopen (filename, "w");
		gui_plot_area_draw (data, output);
		fclose (output);

		gui_status ("Saved LC to file %s", filename);
#ifdef __MINGW32__
		if (PHOEBE_WINDOW_LC_PLOT_IS_DETACHED)
			gui_hide_temp_window (temp_window, GUI_DETACHED_LC_PLOT_WINDOW);
#endif

		GUI_SAVED_DATA_DIR = strdup (dirname (filename));
		g_free (filename);
	}

	gtk_widget_destroy (dialog);

	return;
}

G_MODULE_EXPORT void on_plot_clear_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	gui_plot_clear_canvas (data);
	return;
}

G_MODULE_EXPORT void on_plot_save_data_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	int i;
	FILE *fin, *fout;
	int lines = 0;
	double dummy;
	char line[255];
	char *temp, ch;
	

	for (i = 0; i < data->objno; i++) {
		if (data->request[i].data_changed) {
			fin = fopen (data->request[i].filename, "r");
			temp = phoebe_create_temp_filename ("phoebe_data_XXXXXX");
			fout = fopen (temp, "w");

			while (fgets (line, 255, fin)) {
				if (!phoebe_clean_data_line (line)) {
					/* Non-data line */
					fputs (line, fout);
					continue;
				}
				if ( sscanf (line,  "%lf %lf %lf", &dummy, &dummy, &dummy) < 2 &&
				     sscanf (line, "!%lf %lf %lf", &dummy, &dummy, &dummy) < 2) {
					/* Invalid data line */
					fputs (line, fout);
					continue;
				}
				fputs (line, fout);
				lines += 1;
				if (data->request[i].query->flag->val.iarray[lines] == PHOEBE_DATA_DELETED) fputc ('!', fout);
			}
			fclose (fin);
			fclose (fout);

			fin = fopen (temp, "r");
			fout = fopen (data->request[i].filename, "w");
			while ((ch = fgetc (fin)) != EOF) fputc (ch, fout);

			fclose (fout);
			fclose (fin);
	
			free (temp);

			data->request[i].data_changed = FALSE;
		}
	}
	
}


/******************************************************************************/

G_MODULE_EXPORT void on_phoebe_para_tba_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	char *widget_name = (char*)gtk_widget_get_name(GTK_WIDGET(togglebutton));
	gui_get_value_from_widget(gui_widget_lookup(widget_name));

	gui_fill_sidesheet_fit_treeview ();
	gui_fill_fitt_mf_treeview();
}

void phoebe_gui_constrain_cla_adjust ()
{
	GtkWidget *cla_adjust_checkbutton = gui_widget_lookup("phoebe_para_lum_levels_secadjust_checkbutton")->gtk;
	GtkWidget *lum_decouple_checkbutton = gui_widget_lookup("phoebe_para_lum_options_decouple_checkbutton")->gtk;
	GtkWidget *star_model_combobox = gui_widget_lookup("phoebe_data_star_model_combobox")->gtk;

	if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(lum_decouple_checkbutton)) || (gtk_combo_box_get_active(GTK_COMBO_BOX(star_model_combobox)) == 1)) /* Unconstrained binary system */
		gtk_widget_set_sensitive(cla_adjust_checkbutton, TRUE);
	else {
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(cla_adjust_checkbutton), FALSE);
		gtk_widget_set_sensitive(cla_adjust_checkbutton, FALSE);
	}
}

void phoebe_gui_constrain_phsv (bool constrain)
{
	GtkWidget *phsv_adjust_checkbutton = gui_widget_lookup("phoebe_para_comp_phsvadjust_checkbutton")->gtk;
	GtkWidget *phsv_spinbutton = gui_widget_lookup("phoebe_para_comp_phsv_spinbutton")->gtk;
	GtkWidget *phsv_calculate_button = gui_widget_lookup("phoebe_para_comp_phsv_calculate_button")->gtk;

	gtk_widget_set_sensitive(phsv_adjust_checkbutton, !constrain);
	gtk_widget_set_sensitive(phsv_spinbutton, !constrain);
	gtk_widget_set_sensitive(phsv_calculate_button, !constrain);
	if (constrain)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(phsv_adjust_checkbutton), FALSE);
}

void phoebe_gui_constrain_pcsv (bool constrain)
{
	GtkWidget *pcsv_adjust_checkbutton = gui_widget_lookup("phoebe_para_comp_pcsvadjust_checkbutton")->gtk;
	GtkWidget *pcsv_spinbutton = gui_widget_lookup("phoebe_para_comp_pcsv_spinbutton")->gtk;
	GtkWidget *pcsv_calculate_button = gui_widget_lookup("phoebe_para_comp_pcsv_calculate_button")->gtk;

	gtk_widget_set_sensitive(pcsv_adjust_checkbutton, !constrain);
	gtk_widget_set_sensitive(pcsv_spinbutton, !constrain);
	gtk_widget_set_sensitive(pcsv_calculate_button, !constrain);
	if (constrain)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(pcsv_adjust_checkbutton), FALSE);
}

void phoebe_gui_constrain_secondary_params (bool constrain)
{
	GtkWidget *tavc_spinbutton = gui_widget_lookup("phoebe_para_comp_tavc_spinbutton")->gtk;
	GtkWidget *tavc_adjust_checkbutton = gui_widget_lookup("phoebe_para_comp_tavcadjust_checkbutton")->gtk;
	GtkWidget *alb2_spinbutton = gui_widget_lookup("phoebe_para_surf_alb2_spinbutton")->gtk;
	GtkWidget *alb2_adjust_checkbutton = gui_widget_lookup("phoebe_para_surf_alb2adjust_checkbutton")->gtk;
	GtkWidget *gr2_spinbutton = gui_widget_lookup("phoebe_para_surf_gr2_spinbutton")->gtk;
	GtkWidget *gr2_adjust_checkbutton = gui_widget_lookup("phoebe_para_surf_gr2adjust_checkbutton")->gtk;
	GtkWidget *ld_secondary_adjust_checkbutton = gui_widget_lookup("phoebe_para_ld_lccoefs_secadjust_checkbutton")->gtk;

	gtk_widget_set_sensitive(tavc_spinbutton, !constrain);
	gtk_widget_set_sensitive(tavc_adjust_checkbutton, !constrain);
	if (constrain)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(tavc_adjust_checkbutton), FALSE);

	gtk_widget_set_sensitive(alb2_spinbutton, !constrain);
	gtk_widget_set_sensitive(alb2_adjust_checkbutton, !constrain);
	if (constrain)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(alb2_adjust_checkbutton), FALSE);

	gtk_widget_set_sensitive(gr2_spinbutton, !constrain);
	gtk_widget_set_sensitive(gr2_adjust_checkbutton, !constrain);
	if (constrain)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(gr2_adjust_checkbutton), FALSE);

	gtk_widget_set_sensitive(ld_secondary_adjust_checkbutton, !constrain);
	if (constrain)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(ld_secondary_adjust_checkbutton), FALSE);
}

G_MODULE_EXPORT
void on_angle_units_changed (GtkComboBox *widget, gpointer user_data)
{
	GtkWidget *perr0_spinbutton = gui_widget_lookup ("phoebe_para_orb_perr0_spinbutton")->gtk;
	GtkWidget *perr0step_spinbutton = gui_widget_lookup ("phoebe_para_orb_perr0step_spinbutton")->gtk;
	GtkWidget *perr0min_spinbutton = gui_widget_lookup ("phoebe_para_orb_perr0min_spinbutton")->gtk;
	GtkWidget *perr0max_spinbutton = gui_widget_lookup ("phoebe_para_orb_perr0max_spinbutton")->gtk;
	GtkAdjustment *perr0_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(perr0_spinbutton));
	GtkAdjustment *perr0step_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(perr0step_spinbutton));
	GtkAdjustment *perr0min_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(perr0min_spinbutton));
	GtkAdjustment *perr0max_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(perr0max_spinbutton));

	GtkWidget *dperdt_spinbutton = gui_widget_lookup ("phoebe_para_orb_dperdt_spinbutton")->gtk;
	GtkWidget *dperdtstep_spinbutton = gui_widget_lookup ("phoebe_para_orb_dperdtstep_spinbutton")->gtk;
	GtkWidget *dperdtmin_spinbutton = gui_widget_lookup ("phoebe_para_orb_dperdtmin_spinbutton")->gtk;
	GtkWidget *dperdtmax_spinbutton = gui_widget_lookup ("phoebe_para_orb_dperdtmax_spinbutton")->gtk;
	GtkAdjustment *dperdt_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(dperdt_spinbutton));
	GtkAdjustment *dperdtstep_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(dperdtstep_spinbutton));
	GtkAdjustment *dperdtmin_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(dperdtmin_spinbutton));
	GtkAdjustment *dperdtmax_adjustment = gtk_spin_button_get_adjustment (GTK_SPIN_BUTTON(dperdtmax_spinbutton));

	double change_factor;

	if (gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 0) {
		/* Radians */
		change_factor = M_PI/180.0;

		perr0_adjustment->upper = 2*M_PI;
		perr0step_adjustment->upper = 2*M_PI;
		perr0min_adjustment->upper = 2*M_PI;
		perr0max_adjustment->upper = 2*M_PI;
		perr0_adjustment->step_increment = 0.01;
		perr0step_adjustment->step_increment = 0.01;
		perr0min_adjustment->step_increment = 0.01;
		perr0max_adjustment->step_increment = 0.01;

		dperdt_adjustment->lower = -M_PI/2;
		dperdtmin_adjustment->lower = -M_PI/2;
		dperdtmax_adjustment->lower = -M_PI/2;
		dperdt_adjustment->upper = M_PI/2;
		dperdtstep_adjustment->upper = M_PI/2;
		dperdtmin_adjustment->upper = M_PI/2;
		dperdtmax_adjustment->upper = M_PI/2;
		dperdt_adjustment->step_increment = 0.02;
		dperdtstep_adjustment->step_increment = 0.02;
		dperdtmin_adjustment->step_increment = 0.02;
		dperdtmax_adjustment->step_increment = 0.02;
	}
	else {
		/* Degrees */
		change_factor = 180.0/M_PI;

		perr0_adjustment->upper = 360;
		perr0step_adjustment->upper = 360;
		perr0min_adjustment->upper = 360;
		perr0max_adjustment->upper = 360;
		perr0_adjustment->step_increment = 1;
		perr0step_adjustment->step_increment = 1;
		perr0min_adjustment->step_increment = 1;
		perr0max_adjustment->step_increment = 1;

		dperdt_adjustment->lower = -90;
		dperdtmin_adjustment->lower = -90;
		dperdtmax_adjustment->lower = -90;
		dperdt_adjustment->upper = 90;
		dperdtstep_adjustment->upper = 90;
		dperdtmin_adjustment->upper = 90;
		dperdtmax_adjustment->upper = 90;
		dperdt_adjustment->step_increment = 1;
		dperdtstep_adjustment->step_increment = 1;
		dperdtmin_adjustment->step_increment = 1;
		dperdtmax_adjustment->step_increment = 1;
	}

	gtk_spin_button_set_value(GTK_SPIN_BUTTON(perr0_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(perr0_spinbutton)) * change_factor);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(perr0step_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(perr0step_spinbutton)) * change_factor);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(perr0min_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(perr0min_spinbutton)) * change_factor);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(perr0max_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(perr0max_spinbutton)) * change_factor);

	gtk_spin_button_set_value(GTK_SPIN_BUTTON(dperdt_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(dperdt_spinbutton)) * change_factor);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(dperdtstep_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(dperdtstep_spinbutton)) * change_factor);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(dperdtmin_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(dperdtmin_spinbutton)) * change_factor);
	gtk_spin_button_set_value(GTK_SPIN_BUTTON(dperdtmax_spinbutton), gtk_spin_button_get_value(GTK_SPIN_BUTTON(dperdtmax_spinbutton)) * change_factor);
}

G_MODULE_EXPORT
void on_auto_logg_switch_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	/* Toggle log(g) spin button sensitivity based on the toggle state: */
	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(togglebutton)) == TRUE) {
		gtk_widget_set_sensitive (GTK_WIDGET(gui_widget_lookup("phoebe_para_comp_logg1_spinbutton")->gtk), FALSE);
		gtk_widget_set_sensitive (GTK_WIDGET(gui_widget_lookup("phoebe_para_comp_logg2_spinbutton")->gtk), FALSE);
	}
	else {
		gtk_widget_set_sensitive (GTK_WIDGET(gui_widget_lookup("phoebe_para_comp_logg1_spinbutton")->gtk), TRUE);
		gtk_widget_set_sensitive (GTK_WIDGET(gui_widget_lookup("phoebe_para_comp_logg2_spinbutton")->gtk), TRUE);
	}
}

G_MODULE_EXPORT void on_phoebe_data_star_model_combobox_changed (GtkComboBox *widget, gpointer user_data)
{
	int star_model = gtk_combo_box_get_active(widget) - 1;
	phoebe_gui_constrain_secondary_params( (star_model == 1) ); /* 1: W UMa */
	phoebe_gui_constrain_pcsv(phoebe_pcsv_constrained(star_model));
	phoebe_gui_constrain_phsv(phoebe_phsv_constrained(star_model));
	phoebe_gui_constrain_cla_adjust();
}

G_MODULE_EXPORT void on_phoebe_para_lum_options_decouple_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	phoebe_gui_constrain_cla_adjust();
}

G_MODULE_EXPORT
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

//	printf("\nq = %f, e = %f, f1 = %f\n", q, e, F);

	status = phoebe_calculate_critical_potentials(q, F, e, &L1, &L2);

//	printf("L1 = %f, L2 = %f\n", L1, L2);

	/* Conviniently the potentials are in the first two rows */

	gtk_tree_model_get_iter_first (model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Ω(L<sub>1</sub>)", RS_COL_PARAM_VALUE, L1, -1);

	gtk_tree_model_iter_next (model, &iter);
	gtk_list_store_set((GtkListStore*)model, &iter, RS_COL_PARAM_NAME, "Ω(L<sub>2</sub>)", RS_COL_PARAM_VALUE, L2, -1);
}

int gui_interpolate_all_ld_coefficients (char *ldlaw, double tavh, double tavc, double logg1, double logg2, double met1, double met2)
{
	/* Interpolate all LD coefficients */
	PHOEBE_passband *passband;
	int status, lcno, rvno, index;
	char *notice_title = "LD coefficients interpolation";
	GtkTreeModel *model = GTK_TREE_MODEL (gui_widget_lookup ("phoebe_para_ld_lccoefs_primx")->gtk);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	for (index = -1; index < lcno; index++) {
		GtkTreeIter iter;
		double x1, x2, y1, y2;

		if (index == -1)
			/* Bolometric LD coefficients */
			passband = phoebe_passband_lookup ("Bolometric:3000A-10000A");
		else {
			char *id;
			int i;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_id"), index, &id);
			passband = phoebe_passband_lookup_by_id (id);
			if (!passband) {
				printf ("Passband %s is invalid or unsupported.\n", id);
			}
			
			gtk_tree_model_get_iter_first (model, &iter);
			for (i = 0; i < index; i++)
				gtk_tree_model_iter_next (model, &iter);
		}
		
		/* PRIMARY STAR: */
		status = phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met1, tavh, logg1, &x1, &y1);
//		printf("star 1: teff=%1.1f, logg=%3.3f, [M/H]=%2.2f, x=%f, y=%f\n", tavh, logg1, met1, x1, y1);

		switch (status) {
			case SUCCESS:
				if (index == -1) {
					/* Bolometric LD coefficients */
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_ld_bolcoefs_primx_spinbutton")->gtk), x1);
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_ld_bolcoefs_primy_spinbutton")->gtk), y1);
				}
				else
					gtk_list_store_set (GTK_LIST_STORE (model), &iter, LC_COL_X1, x1, LC_COL_Y1, y1, -1);
			break;
			case ERROR_LD_TABLES_MISSING:
			{
				char message[255];
				sprintf (message, "Limb darkening coefficients missing for passband %s:%s. To enable LD readouts, you must install the corresponding LD table.", passband->set, passband->name);
				gui_notice (notice_title, message);
				return status;
			}
			break;
			default:
				gui_notice (notice_title, phoebe_gui_error (status));
				printf ("tavh = %lf, logg1 = %lf, met1 = %lf\n", tavh, logg1, met1);
				return status;
		}

		/* SECONDARY STAR: */
		status = phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met2, tavc, logg2, &x2, &y2); 
//		printf("star 2: teff=%1.1f, logg=%3.3f, [M/H]=%2.2f: x=%f, y=%f\n", tavc, logg2, met2, x2, y2);

		switch (status) {
			case SUCCESS:
				if (index == -1) {
					/* Bolometric LD coefficients */
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_ld_bolcoefs_secx_spinbutton")->gtk), x2);
					gtk_spin_button_set_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_ld_bolcoefs_secy_spinbutton")->gtk), y2);
				}
				else
					gtk_list_store_set (GTK_LIST_STORE(model), &iter, LC_COL_X2, x2, LC_COL_Y2, y2, -1);
			break;
			case ERROR_LD_TABLES_MISSING:
			{
				char message[255];
				sprintf (message, "Limb darkening coefficients missing for passband %s:%s. To enable LD readouts, you must install the corresponding LD table.", passband->set, passband->name);
				gui_notice (notice_title, message);
				return status;
			}
			break;
			default:
				gui_notice (notice_title, phoebe_gui_error (status));
				printf ("tavc = %lf, logg2 = %lf, met2 = %lf\n", tavc, logg2, met2);
				return status;
		}
	}

	return SUCCESS;
}

int gui_update_ld_coefficients ()
{
	/*
	 * Update calculated values (log g, mass, radius, ...) and interpolate LD
	 * coefficients
	 */

	int status;
	double tavh, tavc, logg1, logg2, met1, met2;
	char* ldlaw;

	gui_get_values_from_widgets ();
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff1"), &tavh);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff2"), &tavc);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met1"), &met1);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met2"), &met2);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_model"), &ldlaw);

	/* For log(g) we need to see whether the automatic updating is in place: */
	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (gui_widget_lookup ("phoebe_para_lum_atmospheres_grav_checkbutton")->gtk)) == TRUE) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"), &logg1);
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"), &logg2);
	}
	else {
		logg1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_logg1_spinbutton")->gtk));
		logg2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_logg2_spinbutton")->gtk));
	}

	status = gui_interpolate_all_ld_coefficients (ldlaw, tavh, tavc, logg1, logg2, met1, met2);

	if (status == SUCCESS)
		LD_COEFFS_NEED_UPDATING = FALSE;

	gui_fill_sidesheet_res_treeview ();

	return status;
}

int gui_update_ld_coefficients_on_autoupdate ()
{
	int status = SUCCESS;
	
	/* Update LD coefficients when parameter gui_ld_model_autoupdate is set */
	GtkWidget *phoebe_para_ld_model_autoupdate_checkbutton = gui_widget_lookup("phoebe_para_ld_model_autoupdate_checkbutton")->gtk;

//	printf("LD autoupdate status: %d\nLD need updating: %d\n", gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(phoebe_para_ld_model_autoupdate_checkbutton)), LD_COEFFS_NEED_UPDATING);

	if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(phoebe_para_ld_model_autoupdate_checkbutton)))
		status = gui_update_ld_coefficients ();
	
	return status;
}

int gui_update_ld_coefficients_when_needed ()
{
	int status = SUCCESS;
	
	/* Update LD coefficients when changes have been made to M/T/log g/LD law */
	if (LD_COEFFS_NEED_UPDATING)
		status = gui_update_ld_coefficients_on_autoupdate();

	return status;
}


/* ******************************************************************** *
 *
 *                    phoebe parameter changes that require LD update
 *
 * ******************************************************************** */

void gui_ld_coeffs_need_updating ()
{
	LD_COEFFS_NEED_UPDATING = TRUE;
}

void gui_logg_values_need_recalculating ()
{
	LOGG_VALUES_NEED_RECALCULATING = TRUE;
	gui_ld_coeffs_need_updating ();
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
 *                    Running minimization in a different thread
 *
 * ******************************************************************** */

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback;
int accept_flag = 0;

#ifdef PHOEBE_GUI_THREADS
bool fitting_in_progress = FALSE;

int gui_progress_pulse (gpointer data)
{
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(data));
    return fitting_in_progress;
}

void *gui_dc_fitting_thread (void *args)
{
	int status;

	fitting_in_progress = TRUE;
	gui_toggle_sensitive_widgets_for_minimization(FALSE);
	status = phoebe_minimize_using_dc (stdout, phoebe_minimizer_feedback);
	pthread_testcancel();
	gui_on_fitting_finished (status);

	pthread_exit(NULL);
	return NULL;
}

void *gui_nms_fitting_thread (void *args)
{
	int status;

	fitting_in_progress = TRUE;
	gui_toggle_sensitive_widgets_for_minimization(FALSE);
	status = phoebe_minimize_using_nms (stdout, phoebe_minimizer_feedback);
	pthread_testcancel();
	gui_on_fitting_finished (status);

	pthread_exit(NULL);
	return NULL;
}
#endif

void gui_on_fitting_finished (int status)
{
	/* Called after the minimization procedure finished */
	GtkTreeView 	*phoebe_fitt_mf_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_first_treeview")->gtk);
	GtkTreeView	*phoebe_fitt_second_treeview = GTK_TREE_VIEW(gui_widget_lookup("phoebe_fitt_second_treeview")->gtk);
	GtkLabel	*phoebe_fitt_feedback_label = GTK_LABEL(gui_widget_lookup("phoebe_fitt_feedback_label")->gtk);
	GtkTreeModel 	*model;
	GtkTreeIter iter;
	PHOEBE_curve *curve;
	int index;
	PHOEBE_minimizer_feedback *feedback = phoebe_minimizer_feedback;
	char *id;
	char status_message[255] = "Minimizer feedback";
	char method[255];

#ifdef PHOEBE_GUI_THREADS
	fitting_in_progress = FALSE;
	gui_toggle_sensitive_widgets_for_minimization(TRUE);
	gtk_widget_hide(gui_widget_lookup("phoebe_fitt_progressbar")->gtk);
#endif

	switch (feedback->algorithm) {
		case PHOEBE_MINIMIZER_DC:
			sprintf (method, "DC minimizer");
			break;
		case PHOEBE_MINIMIZER_NMS:
			sprintf (method, "NMS minimizer");
			break;
		default:
			phoebe_gui_debug ("gui_on_fitting_finished: invalid algorithm passed (code %d).\n", feedback->algorithm);
			return;
	}

	sprintf (status_message, "%s: %s", method, phoebe_gui_error (status));
	status_message[strlen(status_message)-1] = '\0';
	gtk_label_set_text (phoebe_fitt_feedback_label, status_message);
	gui_status (status_message);
	
	if (status == SUCCESS) {
		PHOEBE_array *lc, *rv;
		int lcno, rvno;

		sprintf (status_message, "%s: done %d iterations in %f seconds; cost function value: %f", method, feedback->iters, feedback->cputime, feedback->cfval);
		gtk_label_set_text (phoebe_fitt_feedback_label, status_message);

		/* Results treeview: */
		model = gtk_tree_view_get_model (phoebe_fitt_mf_treeview);
		gtk_list_store_clear (GTK_LIST_STORE (model));

		for (index = 0; index < feedback->qualifiers->dim; index++) {
			gtk_list_store_append (GTK_LIST_STORE (model), &iter);

			/* We need a little hack here if angles are in degrees: */
			if (strcmp (feedback->qualifiers->val.strarray[index], "phoebe_perr0") == 0 ||
			    strcmp (feedback->qualifiers->val.strarray[index], "phoebe_dperdt") == 0) {
				double cv = intern_angle_factor ();
				gtk_list_store_set (GTK_LIST_STORE (model), &iter,
					MF_COL_QUALIFIER, feedback->qualifiers->val.strarray[index],
					MF_COL_INITVAL,   feedback->initvals->val[index]/cv,
					MF_COL_NEWVAL,    feedback->newvals->val[index]/cv,
					MF_COL_ERROR,     feedback->ferrors->val[index]/cv,
					-1);
			}
			else
				gtk_list_store_set (GTK_LIST_STORE (model), &iter,
					MF_COL_QUALIFIER, feedback->qualifiers->val.strarray[index],
					MF_COL_INITVAL,   feedback->initvals->val[index],
					MF_COL_NEWVAL,    feedback->newvals->val[index],
					MF_COL_ERROR,     feedback->ferrors->val[index],
					-1);
		}

		/* Statistics treeview: */
		model = gtk_tree_view_get_model (phoebe_fitt_second_treeview);
		gtk_list_store_clear (GTK_LIST_STORE (model));

		lc = phoebe_active_curves_get (PHOEBE_CURVE_LC); if (lc) lcno = lc->dim; else lcno = 0;
		rv = phoebe_active_curves_get (PHOEBE_CURVE_RV); if (rv) rvno = rv->dim; else rvno = 0;
		
		for (index = 0; index < lcno; index++) {
			curve = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, lc->val.iarray[index]);
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_id"), lc->val.iarray[index], &id);
			gtk_list_store_append (GTK_LIST_STORE (model), &iter);
			gtk_list_store_set (GTK_LIST_STORE (model), &iter,
				CURVE_COL_NAME, id,
				CURVE_COL_NPOINTS, curve->indep->dim,
				(feedback->algorithm == PHOEBE_MINIMIZER_DC ? CURVE_COL_I_RES : CURVE_COL_F_RES), feedback->chi2s->val[rvno+index],
				-1);
			phoebe_curve_free (curve);
		}

		for (index = 0; index < rvno; index++) {
			curve = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, rv->val.iarray[index]);
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_id"), rv->val.iarray[index], &id);
			gtk_list_store_append (GTK_LIST_STORE (model), &iter);
			gtk_list_store_set (GTK_LIST_STORE (model), &iter,
				CURVE_COL_NAME, id,
				CURVE_COL_NPOINTS, curve->indep->dim,
				(feedback->algorithm == PHOEBE_MINIMIZER_DC ? CURVE_COL_I_RES : CURVE_COL_F_RES), feedback->chi2s->val[index],
				-1);
			phoebe_curve_free (curve);
		}

		phoebe_array_free (lc);
		phoebe_array_free (rv);
		accept_flag = 1;
	}

	gui_beep ();
}

/* ******************************************************************** *
 *
 *                    phoebe fitting tab events
 *
 * ******************************************************************** */

G_MODULE_EXPORT
void on_phoebe_fitt_calculate_button_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
#ifdef PHOEBE_GUI_THREADS
	pthread_t thread;
	int thread_return_code;
	GtkWidget *phoebe_fitt_progressbar = gui_widget_lookup("phoebe_fitt_progressbar")->gtk;
#endif
	int status, fit_method;
	GtkComboBox *phoebe_fitt_method_combobox = GTK_COMBO_BOX(gui_widget_lookup("phoebe_fitt_method_combobox")->gtk);

	phoebe_minimizer_feedback = phoebe_minimizer_feedback_new ();

	/* Make sure LD coefficients are current: */
	status = gui_update_ld_coefficients_when_needed ();
	if (status != SUCCESS)
		return;
	
	status = gui_get_values_from_widgets();
	fit_method = gtk_combo_box_get_active (phoebe_fitt_method_combobox);

	switch (fit_method) {
		case 0:
			gui_status("Running DC minimization, please be patient...");
#ifdef PHOEBE_GUI_THREADS
			thread_return_code = pthread_create(&thread, NULL, gui_dc_fitting_thread, NULL);
			if (thread_return_code) printf("Error while attempting to run DC on a separate thread: return code from pthread_create() is %d\n", thread_return_code);
#else 
			status = phoebe_minimize_using_dc (stdout, phoebe_minimizer_feedback);
#endif
			break;
		case 1:
			gui_status("Running NMS minimization, please be patient...");
#ifdef PHOEBE_GUI_THREADS
			thread_return_code = pthread_create(&thread, NULL, gui_nms_fitting_thread, NULL);
			if (thread_return_code) printf("Error while attempting to run NMS on a separate thread: return code from pthread_create() is %d\n", thread_return_code);
#else 
			status = phoebe_minimize_using_nms (stdout, phoebe_minimizer_feedback);
#endif
			break;
		default:
			phoebe_minimizer_feedback_free (phoebe_minimizer_feedback);
			gui_error ("Invalid minimization algorithm", "The minimization algorithm is not selected or is invalid. Please select a valid entry in the fitting tab and try again.");
			return;
		break;
	}

#ifdef PHOEBE_GUI_THREADS
	if (thread_return_code) {
		gui_error ("Error on minimization", "Could not create a separate thread to run the minimization calculations.");
		return;
	}
	gtk_widget_show(phoebe_fitt_progressbar);
	g_timeout_add_full(G_PRIORITY_DEFAULT, 100, gui_progress_pulse, (gpointer)phoebe_fitt_progressbar, NULL);
#else
	gui_on_fitting_finished (status);
#endif
}


int gui_spot_index(int spotsrc, int spotid)
{
	/* Returns the index of the active spot given by its source (primary/secondary) and its id */
	int i, spno, current_spotsrc, current_spotid = 0;
	bool active;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no"), &spno);
	for (i = 0; i < spno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_source"), i, &current_spotsrc);
		if (current_spotsrc == spotsrc) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_active_switch"), i, &active);
			if (active) {
				if (spotid == ++current_spotid)
					return i;
			}
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
		if (status != SUCCESS) {
			gui_error ("Invalid results returned by the minimizer", "The result by the minimizer cannot be read; please report this to the developers!");
			return;
		}
		gui_update_el3_lum_value ();
		gui_set_spot_parameters();
		status = gui_set_values_to_widgets ();
		on_phoebe_para_spots_treeview_cursor_changed ((GtkTreeView *) NULL, (gpointer) NULL);  // Change the values of the current spot
		gui_update_ld_coefficients_on_autoupdate ();
		gui_fill_sidesheet_fit_treeview ();
		gui_fill_fitt_mf_treeview ();
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

	if(phoebe_minimizer_feedback && (phoebe_minimizer_feedback->algorithm == PHOEBE_MINIMIZER_DC)) {
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


G_MODULE_EXPORT
void on_phoebe_data_lc_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_data_lc_treeview_edit ();
}

G_MODULE_EXPORT
void on_phoebe_data_lc_add_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_lc_treeview_add ();
}

G_MODULE_EXPORT
void on_phoebe_data_lc_edit_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_lc_treeview_edit ();
}

G_MODULE_EXPORT
void on_phoebe_data_lc_remove_button_clicked (GtkButton *button, gpointer user_data)
{
    gui_data_lc_treeview_remove ();
}

G_MODULE_EXPORT
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

G_MODULE_EXPORT
void on_phoebe_load_lc_filechooserbutton_selection_changed (GtkFileChooserButton *filechooserbutton, gpointer user_data)
{
	gui_set_text_view_from_file ((GtkWidget *) user_data, gtk_file_chooser_get_filename ((GtkFileChooser*)filechooserbutton));
}

/*
G_MODULE_EXPORT
void on_phoebe_data_lc_model_row_changed (GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data)
{
	PHOEBE_parameter *par = phoebe_parameter_lookup ("gui_lc_plot_obsmenu");
	GtkTreeIter lc_iter;
	char *option;
	
	int state = gtk_tree_model_get_iter_first (tree_model, &lc_iter);
	
	par->menu->option = NULL;
	par->menu->optno = 0;
	
	while (state) {
		gtk_tree_model_get (tree_model, &lc_iter, LC_COL_FILTER, &option, -1);
		if (option) phoebe_parameter_add_option (par, option);
		else break;
		state = gtk_tree_model_iter_next (tree_model, &lc_iter);
	}
}
*/

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
	PHOEBE_parameter *par = phoebe_parameter_lookup ("gui_rv_plot_obsmenu");
	GtkTreeIter rv_iter;
	char *option;

	int state = gtk_tree_model_get_iter_first (tree_model, &rv_iter);

	par->menu->option = NULL;
	par->menu->optno = 0;

	while (state){
		gtk_tree_model_get (tree_model, &rv_iter, RV_COL_FILTER, &option, -1);
		if (option) phoebe_parameter_add_option (par, option);
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
		phoebe_debug("Number of spots: %d\n", spots_no - 1);
		gui_status("A spot removed.");
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
	//gui_get_value_from_widget(gui_widget_lookup("phoebe_para_spots_treeview"));
	phoebe_parameter_set_value(phoebe_parameter_lookup ("phoebe_spots_active_switch"), atoi(gtk_tree_model_get_string_from_iter (model, &iter)), !active);
	gui_fill_sidesheet_fit_treeview ();
	gui_fill_fitt_mf_treeview();
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

G_MODULE_EXPORT
void on_phoebe_file_open_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = gui_open_parameter_file ();

	if( status == SUCCESS ){
		gui_reinit_treeviews();
		gui_set_values_to_widgets();
	}
	else
		printf ("%s", phoebe_gui_error (status));
}

void gui_save_parameter_file_with_confirmation ()
{
	int status;
	bool confirm;

	phoebe_gui_debug ("\tPHOEBE_FILEFLAG = %d\n", PHOEBE_FILEFLAG);
	if (PHOEBE_FILEFLAG)
		phoebe_gui_debug ("\tPHOEBE_FILENAME = %s\n", PHOEBE_FILENAME);

	status = gui_get_values_from_widgets ();

	if (PHOEBE_FILEFLAG) {
		phoebe_config_entry_get ("GUI_CONFIRM_ON_OVERWRITE", &confirm);
		if (!confirm)
			status = phoebe_save_parameter_file (PHOEBE_FILENAME);
		else {
			char *message = phoebe_concatenate_strings ("Do you want to overwrite ", PHOEBE_FILENAME, "?", NULL);
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

	if( status != SUCCESS )
		gui_error ("Error on Save", phoebe_gui_error (status));
}

G_MODULE_EXPORT void on_phoebe_file_save_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gui_save_parameter_file_with_confirmation();
}

G_MODULE_EXPORT void on_phoebe_file_saveas_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	int status = 0;

	status = gui_get_values_from_widgets();
	status = gui_save_parameter_file ();

	if( status != SUCCESS )
		printf ("%s", phoebe_gui_error (status));
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

G_MODULE_EXPORT
void on_phoebe_settings_configuration_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gui_show_configuration_dialog ();
}

G_MODULE_EXPORT
void on_phoebe_help_about_menuitem_activate (GtkMenuItem *menuitem, gpointer user_data)
{
	gchar     *glade_xml_file                       = g_build_filename     (PHOEBE_GLADE_XML_DIR, "phoebe_about.glade", NULL);
	gchar     *glade_pixmap_file                    = g_build_filename     (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	GladeXML  *phoebe_about_xml                   = glade_xml_new        (glade_xml_file, NULL, NULL);
	GtkWidget *phoebe_about_dialog                = glade_xml_get_widget (phoebe_about_xml, "phoebe_about_dialog");

	gtk_window_set_icon (GTK_WINDOW (phoebe_about_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_about_dialog_set_version(GTK_ABOUT_DIALOG(phoebe_about_dialog), VERSION);

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

	GUI_DETACHED_LC_PLOT_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	GUI_DETACHED_RV_PLOT_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_fitting_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_parent_table");

	GUI_DETACHED_FITTING_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
}

G_MODULE_EXPORT void on_phoebe_scripter_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{

}

G_MODULE_EXPORT void on_phoebe_settings_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	gui_show_configuration_dialog();
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
		printf ("%s", phoebe_gui_error (status));
}

G_MODULE_EXPORT void on_phoebe_save_toolbutton_clicked (GtkToolButton *toolbutton, gpointer user_data)
{
	phoebe_gui_debug ("In on_phoebe_save_toolbutton_clicked\n");
	gui_save_parameter_file_with_confirmation ();
}


/* ******************************************************************** *
 *
 *                    phoebe_para_lum_levels events
 *
 * ******************************************************************** */


G_MODULE_EXPORT void
on_phoebe_para_lum_levels_treeview_row_activated (GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data)
{
	gui_para_lum_levels_edit();
}

G_MODULE_EXPORT void
on_phoebe_para_lum_levels_edit_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lum_levels_edit();
}

G_MODULE_EXPORT void on_phoebe_para_lum_levels_calc_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_para_lum_levels_calc_selected();
}

G_MODULE_EXPORT void on_phoebe_para_lum_levels_calc_all_button_clicked (GtkButton *button, gpointer user_data)
{
	int lcno;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);

	if (lcno > 0) {
		GtkTreeModel     *model;
		GtkTreeIter       iter;
		GtkWidget        *treeview = gui_widget_lookup ("phoebe_data_lc_treeview")->gtk;

		model = gtk_tree_view_get_model ((GtkTreeView *) treeview);
		gui_get_values_from_widgets ();

		int state = gtk_tree_model_get_iter_first (model, &iter);

		while (state) {
			gui_para_lum_levels_calc (model, iter);
			state = gtk_tree_model_iter_next (model, &iter);
		}
	}
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

	GUI_DETACHED_SIDESHEET_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_SIDESHEET_IS_DETACHED, "PHOEBE - Data sheets", 300, 600);
}

G_MODULE_EXPORT void on_phoebe_lc_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_lc_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_lc_plot_parent_table");

	GUI_DETACHED_LC_PLOT_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_LC_PLOT_IS_DETACHED, "PHOEBE - LC Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_rv_plot_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_rv_plot_table");
	GUI_widget *parent = gui_widget_lookup ("phoebe_rv_plot_parent_table");

	GUI_DETACHED_RV_PLOT_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_RV_PLOT_IS_DETACHED, "PHOEBE - RV Plot", 726, 522);
}

G_MODULE_EXPORT void on_phoebe_fitt_fitting_detach_button_clicked (GtkButton *button, gpointer user_data)
{
	GUI_widget *box = gui_widget_lookup ("phoebe_fitt_frame");
	GUI_widget *parent = gui_widget_lookup ("phoebe_fitt_parent_table");

	GUI_DETACHED_FITTING_WINDOW = gui_detach_box_from_parent (box->gtk, parent->gtk, &PHOEBE_WINDOW_FITTING_IS_DETACHED, "PHOEBE - Fitting", 600, 400);
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
	gdouble d      = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_d_spinbutton")));
	gdouble rm     = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_rm_spinbutton")));
	gdouble r      = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_r_spinbutton")));
	gdouble f      = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_f_spinbutton")));
	gdouble lambda = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_lambda_spinbutton")));
	gdouble nu     = gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_nu_spinbutton")));

	gdouble pot;

	gchar *source = phoebe_strdup ((gchar *) g_object_get_data (G_OBJECT (button), "star"));

	if (strcmp (source, "primary") == 0)
		pot = phoebe_calculate_pot1 (d, rm, r, f, lambda, nu);
	else
		pot = phoebe_calculate_pot2 (d, rm, r, f, lambda, nu);

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

G_MODULE_EXPORT
void on_phoebe_ld_dialog_interpolate_button_clicked (GtkButton *button, gpointer user_data)
{
	double tavh 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_tavh_spinbutton")));
	double tavc 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_tavc_spinbutton")));
	double logg1 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_logg1_spinbutton")));
	double logg2 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_logg2_spinbutton")));
	double met1 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_met1_spinbutton")));
	double met2 	= gtk_spin_button_get_value (GTK_SPIN_BUTTON(g_object_get_data (G_OBJECT (button), "data_met2_spinbutton")));

	int ldlawindex	= gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_law_combobox")));
	char *ldlaw 	= strdup(phoebe_parameter_lookup ("phoebe_ld_model")->menu->option[ldlawindex]);

	int index = gtk_combo_box_get_active(GTK_COMBO_BOX(g_object_get_data (G_OBJECT (button), "data_id_combobox")));
	PHOEBE_passband *passband;

	int lcno, rvno, status;
	char *id;
	double x1, x2, y1, y2;

	/* Make sure that ID and filter parameter arrays are updated: */
	gui_get_value_from_widget (gui_widget_lookup ("phoebe_data_lc_id"));
	gui_get_value_from_widget (gui_widget_lookup ("phoebe_data_rv_id"));
	gui_get_value_from_widget (gui_widget_lookup ("phoebe_data_lc_filter"));
	gui_get_value_from_widget (gui_widget_lookup ("phoebe_data_rv_filter"));

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	if (index == 0) {
		/* Bolometric LD coefficients */
		passband = phoebe_passband_lookup ("Bolometric:3000A-10000A");
	}
	else {
		if (index-1 < lcno)
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_id"), index-1, &id);
		else
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_id"), index-lcno-1, &id);
		passband = phoebe_passband_lookup_by_id (id);
	}

	if (!passband) {
		gui_notice ("LD coefficient interpolation", "The selected passband is either unsupported or is invalid.");
		return;
	}

	status = phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met1, tavh, logg1, &x1, &y1);
	if (status != SUCCESS)
		gui_notice ("LD coefficient interpolation", phoebe_gui_error (status));
	else {
		gtk_spin_button_set_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_x1_spinbutton")), x1);
		gtk_spin_button_set_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_y1_spinbutton")), y1);
	}

	status = phoebe_ld_get_coefficients (phoebe_ld_model_type (ldlaw), passband, met2, tavc, logg2, &x2, &y2);
	if (status != SUCCESS)
		gui_notice ("LD coefficient interpolation", phoebe_gui_error (status));
	else {
		gtk_spin_button_set_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_x2_spinbutton")), x2);
		gtk_spin_button_set_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_y2_spinbutton")), y2);
	}
}

G_MODULE_EXPORT
void on_phoebe_ld_dialog_update_button_clicked (GtkButton *button, gpointer user_data)
{
	GtkTreeModel *model;
	GtkTreeIter iter;
	char path[10];
	int lcno, rvno;
	
	double x1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_x1_spinbutton")));
	double x2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_x2_spinbutton")));
	double y1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_y1_spinbutton")));
	double y2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (button), "data_y2_spinbutton")));

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	int index = gtk_combo_box_get_active (GTK_COMBO_BOX (g_object_get_data (G_OBJECT (button), "data_id_combobox")));
	
	if (index == 0) {
		/* Bolometric LD coefficients */
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_primx_spinbutton")->gtk), x1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_secx_spinbutton")->gtk), x2);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_primy_spinbutton")->gtk), y1);
		gtk_spin_button_set_value(GTK_SPIN_BUTTON(gui_widget_lookup("phoebe_para_ld_bolcoefs_secy_spinbutton")->gtk), y2);
	}
	else {
		if (index-1 < lcno) {
			/* Light curves: */
			model = GTK_TREE_MODEL (gui_widget_lookup ("phoebe_para_ld_lccoefs_primx")->gtk);
			sprintf (path, "%d", index-1);
			gtk_tree_model_get_iter_from_string (model, &iter, path);
			gtk_list_store_set (GTK_LIST_STORE (model), &iter,
			    LC_COL_X1, x1, LC_COL_X2, x2, LC_COL_Y1, y1, LC_COL_Y2, y2, -1);
		}
		else {
			/* RV curves: */
			model = GTK_TREE_MODEL (gui_widget_lookup ("phoebe_para_ld_rvcoefs_primx")->gtk);
			sprintf (path, "%d", index-lcno-1);
			gtk_tree_model_get_iter_from_string (model, &iter, path);
			gtk_list_store_set (GTK_LIST_STORE (model), &iter,
			    RV_COL_X1, x1, RV_COL_X2, x2, RV_COL_Y1, y1, RV_COL_Y2, y2, -1);
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
	GtkTreeStore 	*store;
	GtkTreeModel    *lcs, *rvs;
	GtkTreeIter 	 toplevel, iter;
	GtkCellRenderer *renderer;

	int i, lcno, rvno;
	char *cid, path[10];

	lcs = gtk_tree_view_get_model ((GtkTreeView *) gui_widget_lookup ("phoebe_data_lc_treeview")->gtk);
	rvs = gtk_tree_view_get_model ((GtkTreeView *) gui_widget_lookup ("phoebe_data_rv_treeview")->gtk);

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
		sprintf (path, "%d", i);
		gtk_tree_model_get_iter_from_string (lcs, &iter, path);
		gtk_tree_model_get (lcs, &iter, LC_COL_ID, &cid, -1);
		gtk_tree_store_append (store, &toplevel, NULL);
		gtk_tree_store_set (store, &toplevel, 0, cid, 1, i+1, -1);
	}
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);
	for (i = 0; i < rvno; i++) {
		sprintf (path, "%d", i);
		gtk_tree_model_get_iter_from_string (rvs, &iter, path);
		gtk_tree_model_get (rvs, &iter, RV_COL_ID, &cid, -1);
		gtk_tree_store_append (store, &toplevel, NULL);
		gtk_tree_store_set (store, &toplevel, 0, cid, 1, i+1, -1);
	}
	
	g_object_unref (store);
	
	gtk_combo_box_set_active (GTK_COMBO_BOX (combo_box), 0);
	return SUCCESS;
}

G_MODULE_EXPORT
void on_phoebe_para_ld_model_tables_vanhamme_button_clicked (GtkButton *button, gpointer user_data)
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

	/* For log(g) we need to see whether the automatic updating is in place: */
	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (gui_widget_lookup ("phoebe_para_lum_atmospheres_grav_checkbutton")->gtk)) == TRUE) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"), &logg1);
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"), &logg2);
	}
	else {
		logg1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_logg1_spinbutton")->gtk));
		logg2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_logg2_spinbutton")->gtk));
	}

	gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_ld_dialog_logg1_spinbutton), logg1);
	gtk_spin_button_set_value (GTK_SPIN_BUTTON (phoebe_ld_dialog_logg2_spinbutton), logg2);

	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_law_combobox",     (gpointer) phoebe_ld_dialog_law_combobox);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_id_combobox",      (gpointer) phoebe_ld_dialog_id_combobox);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_tavh_spinbutton",  (gpointer) phoebe_ld_dialog_tavh_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_logg1_spinbutton", (gpointer) phoebe_ld_dialog_logg1_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_met1_spinbutton",  (gpointer) phoebe_ld_dialog_met1_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_tavc_spinbutton",  (gpointer) phoebe_ld_dialog_tavc_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_logg2_spinbutton", (gpointer) phoebe_ld_dialog_logg2_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_met2_spinbutton",  (gpointer) phoebe_ld_dialog_met2_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_x1_spinbutton",    (gpointer) phoebe_ld_dialog_x1_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_y1_spinbutton",    (gpointer) phoebe_ld_dialog_y1_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_x2_spinbutton",    (gpointer) phoebe_ld_dialog_x2_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_interpolate_button), "data_y2_spinbutton",    (gpointer) phoebe_ld_dialog_y2_spinbutton);

	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button),      "data_x1_spinbutton",    (gpointer) phoebe_ld_dialog_x1_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button),      "data_y1_spinbutton",    (gpointer) phoebe_ld_dialog_y1_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button),      "data_x2_spinbutton",    (gpointer) phoebe_ld_dialog_x2_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button),      "data_y2_spinbutton",    (gpointer) phoebe_ld_dialog_y2_spinbutton);
	g_object_set_data (G_OBJECT (phoebe_ld_dialog_update_button),      "data_id_combobox",      (gpointer) phoebe_ld_dialog_id_combobox);

	g_signal_connect (GTK_WIDGET(phoebe_ld_dialog_close_button),       "clicked", G_CALLBACK (on_phoebe_ld_dialog_close_button_clicked), (gpointer) phoebe_ld_dialog);
	g_signal_connect (GTK_WIDGET(phoebe_ld_dialog_update_button),      "clicked", G_CALLBACK (on_phoebe_ld_dialog_update_button_clicked), NULL);
	g_signal_connect (GTK_WIDGET(phoebe_ld_dialog_interpolate_button), "clicked", G_CALLBACK (on_phoebe_ld_dialog_interpolate_button_clicked), NULL);

	gtk_widget_show (phoebe_ld_dialog);
}

G_MODULE_EXPORT void on_phoebe_settings_confirmation_save_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	/*
	 * This handler is invoked when Settings->Options->ConfirmOnOverwrite has
	 * been toggled. It changes the configuration parameter
	 * GUI_CONFIRM_ON_OVERWRITE.
	 */

	if (togglebutton->active == TRUE)
		phoebe_config_entry_set ("GUI_CONFIRM_ON_OVERWRITE", TRUE);
	else
		phoebe_config_entry_set ("GUI_CONFIRM_ON_OVERWRITE", FALSE);
}

G_MODULE_EXPORT void on_phoebe_beep_after_plot_and_fit_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
	/*
	 * This handler is invoked when Settings->Options->BeepAfterPlotAndFit has
	 * been toggled. It changes the configuration parameter
	 * GUI_BEEP_AFTER_PLOT_AND_FIT.
	 */

	if (togglebutton->active == TRUE)
		phoebe_config_entry_set ("GUI_BEEP_AFTER_PLOT_AND_FIT", TRUE);
	else
		phoebe_config_entry_set ("GUI_BEEP_AFTER_PLOT_AND_FIT", FALSE);
}

G_MODULE_EXPORT
void on_critical_potentials_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	/*
	 * Called when parameters that determine critical potentials are changed.
	 */

	GtkTreeView *sidesheet = (GtkTreeView *) gui_widget_lookup ("phoebe_sidesheet_res_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model (sidesheet);

	double q = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_sys_rm_spinbutton")->gtk));
	double F = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_f1_spinbutton")->gtk));
	double e = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_ecc_spinbutton")->gtk));

	double L1, L2;

	phoebe_calculate_critical_potentials (q, F, e, &L1, &L2);

	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_LAGRANGE_1, L1);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_LAGRANGE_2, L2);
}

G_MODULE_EXPORT
void on_stellar_masses_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	/*
	 * Called when parameters that determine stellar masses are changed.
	 */

	GtkTreeView *sidesheet = (GtkTreeView *) gui_widget_lookup ("phoebe_sidesheet_res_treeview")->gtk;
	GtkTreeModel *model = gtk_tree_view_get_model (sidesheet);

	double q = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_sys_rm_spinbutton")->gtk));
	double a = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_sys_sma_spinbutton")->gtk));
	double P = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_eph_period_spinbutton")->gtk));

	double M1, M2;

	phoebe_calculate_masses (a, P, q, &M1, &M2);

	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_MASS_1, M1);
	gui_set_treeview_value (model, RS_COL_PARAM_VALUE, SIDESHEET_MASS_2, M2);
}

G_MODULE_EXPORT
void on_orbital_elements_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	/* This callback is called when eccentricity or argument of periastron
	 * are changed and we need to recompute critical phases.
	 */

	double pp, scp, icp, anp, dnp;
	double w_unit = intern_angle_factor();
	double e    = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_ecc_spinbutton")->gtk));
	double w    = w_unit * gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_perr0_spinbutton")->gtk));
	double dp   = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_eph_pshift_spinbutton")->gtk));
	double p    = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_eph_period_spinbutton")->gtk));
	double hjd0 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_eph_hjd0_spinbutton")->gtk));
	char value[255];

	phoebe_compute_critical_phases (&pp, &scp, &icp, &anp, &dnp, w, e, dp);

	sprintf (value, "%lf", pp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_perr0_phase")->gtk), value);
	sprintf (value, "%lf", scp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_supcon_phase")->gtk), value);
	sprintf (value, "%lf", icp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_infcon_phase")->gtk), value);
	sprintf (value, "%lf", anp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_ascnode_phase")->gtk), value);
	sprintf (value, "%lf", dnp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_descnode_phase")->gtk), value);

	sprintf (value, "%lf", hjd0 + p * pp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_perr0_hjd")->gtk), value);
	sprintf (value, "%lf", hjd0 + p * scp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_supcon_hjd")->gtk), value);
	sprintf (value, "%lf", hjd0 + p * icp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_infcon_hjd")->gtk), value);
	sprintf (value, "%lf", hjd0 + p * anp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_ascnode_hjd")->gtk), value);
	sprintf (value, "%lf", hjd0 + p * dnp);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_descnode_hjd")->gtk), value);

	return;
}

G_MODULE_EXPORT
void on_phoebe_fitt_cfval_compute_button_clicked (GtkButton *button, gpointer user_data)
{
	gui_notice("Not implemented", "The calculation of the cost function value has not been implemented yet!");
/*
#warning WILL_FAIL_ON_NO_DATA_AND_GIVE_WRONG_RESULT_FOR_INACTIVE_DATA
	int lcno, rvno, index, status;
	double chi2;
	char cfval[255];

	PHOEBE_vector *chi2s;
	PHOEBE_curve *obs, *syncurve;

	GtkWidget *label = gui_widget_lookup ("phoebe_fitt_cfval")->gtk;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	index = 1;
	chi2s = phoebe_vector_new ();
	phoebe_vector_alloc (chi2s, lcno+rvno);

	while (index <= lcno + rvno) {
		if (index <= lcno) {
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, index-1);

			if (!obs) {
				printf ("handle me.\n");
			}

			phoebe_curve_transform (obs, obs->itype, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_SIGMA);
*/
			/* Synthesize a theoretical curve: */
/*
			syncurve = phoebe_curve_new ();
			phoebe_curve_compute (syncurve, obs->indep, index-1, obs->itype, PHOEBE_COLUMN_FLUX);
		}
		else {
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, index-lcno-1);
			if (!obs) {
				printf ("handle me.\n");
			}

			phoebe_curve_transform (obs, obs->itype, obs->dtype, PHOEBE_COLUMN_SIGMA);

			syncurve = phoebe_curve_new ();
			phoebe_curve_compute (syncurve, obs->indep, index-lcno-1, obs->itype, obs->dtype);
		}

		status = phoebe_cf_compute (&chi2, PHOEBE_CF_CHI2, syncurve->dep, obs->dep, obs->weight, 1.0);
		if (status != SUCCESS) {
			printf ("handle me.\n");
		}

		phoebe_curve_free (obs);
		phoebe_curve_free (syncurve);

		chi2s->val[index-1] = chi2;
		index++;
	}

	sprintf (cfval, "%lf", chi2s->val[0]);
	gtk_label_set_text (GTK_LABEL (label), cfval);
*/
}

G_MODULE_EXPORT
void on_star_shape_changed (GtkSpinButton *spinbutton, gpointer user_data)
{
	/* This callback is called when any of the parameters that determine the
	 * shape of the star are changed; it recomputes the radii.
	 */

	double rpole, rside, rpoint, rback;

	char value[255];

	double pot1 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_phsv_spinbutton")->gtk));
	double pot2 = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_comp_pcsv_spinbutton")->gtk));
	double e    = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_ecc_spinbutton")->gtk));
	double q    = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_sys_rm_spinbutton")->gtk));
	double F1   = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_f1_spinbutton")->gtk));
	double F2   = gtk_spin_button_get_value (GTK_SPIN_BUTTON (gui_widget_lookup ("phoebe_para_orb_f2_spinbutton")->gtk));

	double L1crit, L2crit;

	phoebe_calculate_critical_potentials (q, F1, e, &L1crit, &L2crit);

	if (pot1 < L2crit) {
		rpole  = -1.0;
		rpoint = -1.0;
		rside  = -1.0;
		rback  = -1.0;
	}
	else {
		rpole  = phoebe_compute_polar_radius (pot1, (1-e), q);
		rpoint = phoebe_compute_radius (rpole, q, (1-e), F1,  1, 0);
		rside  = phoebe_compute_radius (rpole, q, (1-e), F1,  0, 0);
		rback  = phoebe_compute_radius (rpole, q, (1-e), F1, -1, 0);
	}

	if (rpole  < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rpole);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rpole1_value")->gtk), value);
	if (rpoint < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rpoint);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rpoint1_value")->gtk), value);
	if (rside  < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rside);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rside1_value")->gtk), value);
	if (rback  < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rback);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rback1_value")->gtk), value);

	pot2 = pot2/q + 0.5*(q-1)/q;
	q = 1.0/q;

	phoebe_calculate_critical_potentials (q, F2, e, &L1crit, &L2crit);

	if (pot2 < L2crit) {
		rpole  = -1.0;
		rpoint = -1.0;
		rside  = -1.0;
		rback  = -1.0;
	}
	else {
		rpole  = phoebe_compute_polar_radius (pot2, (1-e), q);
		rpoint = phoebe_compute_radius (rpole, q, (1-e), F2,  1, 0);
		rside  = phoebe_compute_radius (rpole, q, (1-e), F2,  0, 0);
		rback  = phoebe_compute_radius (rpole, q, (1-e), F2, -1, 0);
	}

	if (rpole  < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rpole);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rpole2_value")->gtk), value);
	if (rpoint < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rpoint);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rpoint2_value")->gtk), value);
	if (rside  < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rside);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rside2_value")->gtk), value);
	if (rback  < 0) sprintf (value, "overflow"); else sprintf (value, "%4.4lf", rback);
	gtk_label_set_text (GTK_LABEL (gui_widget_lookup ("gui_rback2_value")->gtk), value);

	return;
}
