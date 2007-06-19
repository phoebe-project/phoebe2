#include <glade/glade.h>
#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"

int gui_init_widgets ()
{
	/*
	 * This function hooks all widgets to the parameters and adds them to the
	 * widget hash table.
	 */

	/*************************   Glade XML files   ****************************/

	GladeXML *phoebe_window = glade_xml_new("phoebe.glade", NULL, NULL);

	/**************************    Data Widgets   *****************************/

	gui_widget_add ("phoebe_data_star_name_entry", 					glade_xml_get_widget(phoebe_window, "phoebe_data_star_name_entry"), 					GUI_WIDGET_VALUE,		phoebe_parameter_lookup ("phoebe_name"));
	/*gui_widget_add ("phoebe_data_star_model_combobox", 			glade_xml_get_widget(phoebe_window, "phoebe_data_star_model_combobox"), 			GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_model"));
	gui_widget_add ("phoebe_data_lcoptions_mag_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_data_lcoptions_mag_spinbutton"), 		GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_mnorm"));
	gui_widget_add ("phoebe_data_rvoptions_psepe_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_psepe_checkbutton"), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"));
	gui_widget_add ("phoebe_data_rvoptions_ssepe_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_ssepe_checkbutton"), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"));
	
	//???gui_widget_add ("phoebe_data_options_time_radiobutton", 	glade_xml_get_widget(phoebe_window, "phoebe_data_options_time_radiobutton"), 		GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_indep"));
	//???gui_widget_add ("phoebe_data_options_phase_radiobutton", 	glade_xml_get_widget(phoebe_window, "phoebe_data_options_phase_radiobutton"), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_indep"));

	gui_widget_add ("phoebe_data_options_bins_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_data_options_bins_checkbutton"), 		GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_bins_switch"));
	gui_widget_add ("phoebe_data_options_binsno_spinbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_data_options_binsno_spinbutton"), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_bins"));
	
	//???gui_widget_add ("phoebe_data_options_the_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_data_options_the_spinbutton"), 		GUI_WIDGET_VALUE, 	phoebe_parameter_lookup (""));

	gui_widget_add ("phoebe_data_options_binsno_spinbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_data_options_binsno_spinbutton"), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_bins"));

	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_filename"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_sigma"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_filter"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_indep"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_dep"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_indweight"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_levweight"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_active"));

	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_filename"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_sigma"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_filter"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_indep"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_dep"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_indweight"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_active"));


	/***********************    Parameters Widgets   **************************/

	/*gui_widget_add ("phoebe_params_ephemeris_hjd0_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0_spinbutton"), 				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_hjd0"));
	gui_widget_add ("phoebe_params_ephemeris_hjd0adjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0adjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_hjd0"));
	gui_widget_add ("phoebe_params_ephemeris_hjd0step_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0step_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_hjd0"));
	gui_widget_add ("phoebe_params_ephemeris_hjd0max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0max_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_hjd0"));
	gui_widget_add ("phoebe_params_ephemeris_hjd0min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0min_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_hjd0"));

	gui_widget_add ("phoebe_params_ephemeris_period_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_period_spinbutton"), 			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_period"));
	gui_widget_add ("phoebe_params_ephemeris_periodadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_period"));
	gui_widget_add ("phoebe_params_ephemeris_periodstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_period"));
	gui_widget_add ("phoebe_params_ephemeris_periodmax_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodmax_spinbutton"), 		GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_period"));
	gui_widget_add ("phoebe_params_ephemeris_periodmin_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodmin_spinbutton"), 		GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_period"));

	gui_widget_add ("phoebe_params_ephemeris_dpdt_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdt_spinbutton"), 				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_dpdt"));
	gui_widget_add ("phoebe_params_ephemeris_dpdtadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_dpdt"));
	gui_widget_add ("phoebe_params_ephemeris_dpdtstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_dpdt"));
	gui_widget_add ("phoebe_params_ephemeris_dpdtmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_dpdt"));
	gui_widget_add ("phoebe_params_ephemeris_dpdtmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_dpdt"));

	gui_widget_add ("phoebe_params_ephemeris_pshift_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshift_spinbutton"), 			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_pshift"));
	gui_widget_add ("phoebe_params_ephemeris_pshiftadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_pshift"));
	gui_widget_add ("phoebe_params_ephemeris_pshiftstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_pshift"));
	gui_widget_add ("phoebe_params_ephemeris_pshiftmax_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftmax_spinbutton"), 		GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_pshift"));
	gui_widget_add ("phoebe_params_ephemeris_pshiftmin_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftmin_spinbutton"), 		GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_pshift"));

	gui_widget_add ("phoebe_params_system_sma_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_sma_spinbutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_sma"));
	gui_widget_add ("phoebe_params_system_smaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_smaadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_sma"));
	gui_widget_add ("phoebe_params_system_smastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_smastep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_sma"));
	gui_widget_add ("phoebe_params_system_smamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_smamax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_sma"));
	gui_widget_add ("phoebe_params_system_smamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_smamin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_sma"));

	gui_widget_add ("phoebe_params_system_rm_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_rm_spinbutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_rm"));
	gui_widget_add ("phoebe_params_system_rmadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_rm"));
	gui_widget_add ("phoebe_params_system_rmstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmstep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_rm"));
	gui_widget_add ("phoebe_params_system_rmmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmmax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_rm"));
	gui_widget_add ("phoebe_params_system_rmmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmmin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_rm"));

	gui_widget_add ("phoebe_params_system_vga_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_vga_spinbutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_vga"));
	gui_widget_add ("phoebe_params_system_vgaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgaadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_vga"));
	gui_widget_add ("phoebe_params_system_vgastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgastep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_vga"));
	gui_widget_add ("phoebe_params_system_vgamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgamax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_vga"));
	gui_widget_add ("phoebe_params_system_vgamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgamin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_vga"));

	gui_widget_add ("phoebe_params_system_incl_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_incl_spinbutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_incl"));
	gui_widget_add ("phoebe_params_system_incladjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_system_incladjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_incl"));
	gui_widget_add ("phoebe_params_system_inclstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_inclstep_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_incl"));
	gui_widget_add ("phoebe_params_system_inclmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_inclmax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	phoebe_parameter_lookup ("phoebe_incl"));
	gui_widget_add ("phoebe_params_system_inclmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_inclmin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	phoebe_parameter_lookup ("phoebe_incl"));


	/* ** */
	return SUCCESS;
}

GUI_widget *gui_widget_new ()
{
	GUI_widget *widget = phoebe_malloc (sizeof (*widget));

	widget->name = NULL;
	widget->type = 0;
	widget->gtk  = NULL;
	widget->par  = NULL;

	return SUCCESS;
}

int gui_widget_free (GUI_widget *widget)
{
	if (!widget)
		return SUCCESS;

	if (widget->name) free (widget->name);
	free (widget);

	return SUCCESS;
}

int gui_widget_hookup (GUI_widget *widget, GtkWidget *gtk, GUI_widget_type type, char *name, PHOEBE_parameter *par)
{
	if (!widget)
	/*
	 * A suggestion: create a phoebe_gui_errors.h and create an enum for these
	 * error codes.
	 */
		return /* ERROR_GUI_WIDGET_NOT_FOUND; */ -1;

	widget->name = strdup (name);
	widget->gtk  = gtk;
	widget->type = type;
	widget->par  = par;

	return SUCCESS;
}

/* ADD HASH TABLE WRAPPERS HERE */

//
// In gui_widget_add (...) you'd then have the calls to this function, sth like:
//
//   GUI_widget *widget = gui_widget_new ();
//   gui_widget_hookup (widget, gtk, GUI_WIDGET_VALUE, "widget_name", phoebe_parameter_lookup ("parameter_name"));
//   gui_widget_commit (widget); <-- this plugs the widget to the hashed table
//
// In phoebe_gui_init () you'd then have:
//
//   gui_widget_add (/* name = */ "gui_widget_name", /* gtk = */ glade_xml_lookup ("gui_widget_name"), /* type = */ GUI_WIDGET_VALUE, /* par = */ phoebe_parameter_lookup ("phoebe_qualifier"));
//
// You could do some error handling if you wish, i.e. if gtk or par are null etc.
//

int gui_widget_add (char *name, GtkWidget *gtk, GUI_widget_type type, PHOEBE_parameter *par)
{
	GUI_widget *widget = gui_widget_new();
	
	if (!gtk)
		return -1;

	if (!par)
		return -1;

	gui_widget_hookup (widget, gtk, type, name, par);

	//gui_widget_commit (widget);

	return SUCCESS;
}

unsigned int gui_widget_hash (char *name)
{
	/*
	 * This is the hashing function for storing widgets into the widget
	 * table.
	 */
	unsigned int h = 0;
	unsigned char *w;

	for (w = (unsigned char *) name; *w != '\0'; w++)
		h = GUI_WT_HASH_MULTIPLIER * h + *w;

	return h % GUI_WT_HASH_BUCKETS;
}

int gui_widget_commit (GUI_widget *widget)
{
	int hash = gui_widget_hash (widget->name);
	GUI_wt_bucket *elem = GUI_wt->elem[hash];

	while (elem) {
		if (strcmp (elem->widget->name, widget->name) == 0) break;
		elem = elem->next;
	}

	if (elem) {
		/* gui_error(widget already commited...) */
		return SUCCESS;
	}

	else{
		elem = phoebe_malloc (sizeof (*elem));

		elem->widget = widget;
		elem->next	 = GUI_wt->elem[hash];
		GUI_wt->elem[hash] = elem;
	}

	return SUCCESS;
}

int gui_free_widgets ()
{
	/*
	 * This function frees all widgets from the widget table.
	 */

	int i;
	GUI_wt_bucket *elem;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		while (GUI_wt->elem[i]) {
			elem = GUI_wt->elem[i];
			GUI_wt->elem[i] = elem->next;
			gui_widget_free (elem->widget);
			free (elem);
		}
	}
	free (GUI_wt);

	return SUCCESS;
}
