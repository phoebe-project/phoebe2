#include <glade/glade.h>
#include <gtk/gtk.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"
#include "phoebe_gui_treeviews.h"

int gui_init_widgets (GladeXML* phoebe_window)
{
	/*
	 * This function hooks all widgets to the parameters and adds them to the
	 * widget hash table.
	 */

	int i;

	PHOEBE_parameter *par;

	GUI_wt = phoebe_malloc(sizeof(GUI_widget_table));
	for(i=0; i<GUI_WT_HASH_BUCKETS; i++)GUI_wt->bucket[i]=NULL;

	/**************************    Data Widgets   *****************************/

	gui_widget_add ("phoebe_data_star_name_entry", 								glade_xml_get_widget(phoebe_window, "phoebe_data_star_name_entry"), 								GUI_WIDGET_VALUE,			phoebe_parameter_lookup ("phoebe_name"));
	gui_widget_add ("phoebe_data_star_model_combobox", 						glade_xml_get_widget(phoebe_window, "phoebe_data_star_model_combobox"), 						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_model"));
	gui_widget_add ("phoebe_data_lcoptions_mag_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_data_lcoptions_mag_spinbutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_mnorm"));
	gui_widget_add ("phoebe_data_rvoptions_psepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_psepe_checkbutton"), 				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"));
	gui_widget_add ("phoebe_data_rvoptions_ssepe_checkbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_rvoptions_ssepe_checkbutton"), 				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"));

	gui_widget_add ("phoebe_data_options_time_radiobutton", 					glade_xml_get_widget(phoebe_window, "phoebe_data_options_time_radiobutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_indep"));
	gui_widget_add ("phoebe_data_options_bins_checkbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_data_options_bins_checkbutton"), 					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins_switch"));
	gui_widget_add ("phoebe_data_options_binsno_spinbutton", 	   		glade_xml_get_widget(phoebe_window, "phoebe_data_options_binsno_spinbutton"), 				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_bins"));

	gui_widget_add ("phoebe_data_lc_filename",  					(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_FILENAME), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filename"));
	gui_widget_add ("phoebe_data_lc_sigma", 						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_SIGMA), 	   GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_sigma"));
	gui_widget_add ("phoebe_data_lc_filter", 						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_FILTER), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_filter"));
	gui_widget_add ("phoebe_data_lc_indep", 						(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_ITYPE), 	   GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_indep"));
	gui_widget_add ("phoebe_data_lc_dep", 							(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_DTYPE), 	   GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_lc_dep"));
	gui_widget_add ("phoebe_data_lc_wtype",     					(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_WTYPE), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_indweight"));
	gui_widget_add ("phoebe_para_lc_levweight", 					(GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_LEVWEIGHT), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_levweight"));
	gui_widget_add ("phoebe_data_lc_active", 					    (GtkWidget*)gtk_tree_view_get_column((GtkTreeView*)phoebe_data_lc_treeview, LC_COL_ACTIVE), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_lc_active"));

	gui_widget_add ("",  					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_rv_filename"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_rv_sigma"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_rv_filter"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_rv_indep"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_rv_dep"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_indweight"));
	gui_widget_add ("", 					glade_xml_get_widget(phoebe_window, ""), 	GUI_WIDGET_VALUE, 	phoebe_parameter_lookup ("phoebe_rv_active"));


	/* **********************    Parameters Widgets   ************************* */

	par = phoebe_parameter_lookup ("phoebe_hjd0");
	gui_widget_add ("phoebe_params_ephemeris_hjd0_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_ephemeris_hjd0adjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0adjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_ephemeris_hjd0step_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0step_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_ephemeris_hjd0max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0max_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_ephemeris_hjd0min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_hjd0min_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_period");
	gui_widget_add ("phoebe_params_ephemeris_period_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_period_spinbutton"), 			GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_ephemeris_periodadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_ephemeris_periodstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_ephemeris_periodmax_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodmax_spinbutton"), 		GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_ephemeris_periodmin_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_periodmin_spinbutton"), 		GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_dpdt");
	gui_widget_add ("phoebe_params_ephemeris_dpdt_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdt_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_ephemeris_dpdtadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_ephemeris_dpdtstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_ephemeris_dpdtmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_ephemeris_dpdtmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_dpdtmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pshift");
	gui_widget_add ("phoebe_params_ephemeris_pshift_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshift_spinbutton"), 			GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_ephemeris_pshiftadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_ephemeris_pshiftstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_ephemeris_pshiftmax_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftmax_spinbutton"), 		GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_ephemeris_pshiftmin_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_ephemeris_pshiftmin_spinbutton"), 		GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_sma");
	gui_widget_add ("phoebe_params_system_sma_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_sma_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_system_smaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_smaadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_system_smastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_smastep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_system_smamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_smamax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_system_smamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_smamin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_rm");
	gui_widget_add ("phoebe_params_system_rm_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_rm_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_system_rmadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_system_rmstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmstep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_system_rmmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmmax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_system_rmmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_rmmin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_vga");
	gui_widget_add ("phoebe_params_system_vga_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_system_vga_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_system_vgaadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgaadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_system_vgastep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgastep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_system_vgamax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgamax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_system_vgamin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_vgamin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_incl");
	gui_widget_add ("phoebe_params_system_incl_spinbutton", 		         glade_xml_get_widget(phoebe_window, "phoebe_params_system_incl_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_system_incladjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_system_incladjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_system_inclstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_system_inclstep_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_system_inclmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_inclmax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_system_inclmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_system_inclmin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_perr0");
	gui_widget_add ("phoebe_params_orbit_perr0_spinbutton", 		         glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_perr0_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_orbit_perr0adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_perr0adjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_orbit_perr0step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_perr0step_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_orbit_perr0max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_perr0max_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_orbit_perr0min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_perr0min_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_dperdt");
	gui_widget_add ("phoebe_params_orbit_dperdt_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_dperdt_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_orbit_dperdtadjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_dperdtadjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_orbit_dperdtstep_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_dperdtstep_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_orbit_dperdtmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_dperdtmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_orbit_dperdtmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_dperdtmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_ecc");
	gui_widget_add ("phoebe_params_orbit_ecc_spinbutton", 		      	glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_ecc_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_orbit_eccadjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_eccadjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_orbit_eccstep_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_eccstep_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_orbit_eccmax_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_eccmax_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_orbit_eccmin_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_eccmin_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_f1");
	gui_widget_add ("phoebe_params_orbit_f1_spinbutton", 		      		glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f1_spinbutton"), 						GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_orbit_f1adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f1adjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_orbit_f1step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f1step_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_orbit_f1max_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f1max_spinbutton"), 					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_orbit_f1min_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f1min_spinbutton"), 					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_f2");
	gui_widget_add ("phoebe_params_orbit_f2_spinbutton", 		      		glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f2_spinbutton"), 						GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_orbit_f2adjust_checkbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f2adjust_checkbutton"), 			GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_orbit_f2step_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f2step_spinbutton"), 				GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_orbit_f2max_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f2max_spinbutton"), 					GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_orbit_f2min_spinbutton", 					glade_xml_get_widget(phoebe_window, "phoebe_params_orbit_f2min_spinbutton"), 					GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_teff1");
	gui_widget_add ("phoebe_params_component_tavh_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavh_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_tavhadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavhadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_tavhstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavhstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_tavhmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavhmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_tavhmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavhmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_teff2");
	gui_widget_add ("phoebe_params_component_tavc_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavc_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_tavcadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavcadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_tavcstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavcstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_tavcmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavcmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_tavcmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_tavcmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pot1");
	gui_widget_add ("phoebe_params_component_phsv_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_component_phsv_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_phsvadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_phsvadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_phsvstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_phsvstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_phsvmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_phsvmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_phsvmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_phsvmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_pot2");
	gui_widget_add ("phoebe_params_component_pcsv_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_component_pcsv_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_pcsvadjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_pcsvadjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_pcsvstep_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_pcsvstep_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_pcsvmax_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_pcsvmax_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_pcsvmin_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_pcsvmin_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_met1");
	gui_widget_add ("phoebe_params_component_met1_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_component_met1_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_met1adjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_met1adjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_met1step_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_met1step_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_met1max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_met1max_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_met1min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_met1min_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_met2");
	gui_widget_add ("phoebe_params_component_met2_spinbutton", 		      glade_xml_get_widget(phoebe_window, "phoebe_params_component_met2_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_met2adjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_met2adjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_met2step_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_met2step_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_met2max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_met2max_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_met2min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_component_met2min_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_logg1");
	gui_widget_add ("phoebe_params_component_logg1_spinbutton", 		   glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg1_spinbutton"), 			GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_logg1adjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg1adjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_logg1step_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg1step_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_logg1max_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg1max_spinbutton"), 		GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_logg1min_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg1min_spinbutton"), 		GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_logg2");
	gui_widget_add ("phoebe_params_component_logg2_spinbutton", 		   glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg2_spinbutton"), 			GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_component_logg2adjust_checkbutton", 	glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg2adjust_checkbutton"), 	GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_component_logg2step_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg2step_spinbutton"), 		GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_component_logg2max_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg2max_spinbutton"), 		GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_component_logg2min_spinbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_component_logg2min_spinbutton"), 		GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_alb1");
	gui_widget_add ("phoebe_params_surface_alb1_spinbutton", 		  	 	glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb1_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_surface_alb1adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb1adjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_surface_alb1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb1step_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_surface_alb1max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb1max_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_surface_alb1min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb1min_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_alb2");
	gui_widget_add ("phoebe_params_surface_alb2_spinbutton", 		  	 	glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb2_spinbutton"), 				GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_surface_alb2adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb2adjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_surface_alb2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb2step_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_surface_alb2max_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb2max_spinbutton"), 			GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_surface_alb2min_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_alb2min_spinbutton"), 			GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_grb1");
	gui_widget_add ("phoebe_params_surface_gr1_spinbutton", 		  	 		glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr1_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_surface_gr1adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr1adjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_surface_gr1step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr1step_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_surface_gr1max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr1max_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_surface_gr1min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr1min_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);

	par = phoebe_parameter_lookup ("phoebe_grb2");
	gui_widget_add ("phoebe_params_surface_gr2_spinbutton", 		  	 		glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr2_spinbutton"), 					GUI_WIDGET_VALUE, 		par);
	gui_widget_add ("phoebe_params_surface_gr2adjust_checkbutton", 		glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr2adjust_checkbutton"), 		GUI_WIDGET_SWITCH_TBA, 	par);
	gui_widget_add ("phoebe_params_surface_gr2step_spinbutton", 			glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr2step_spinbutton"), 			GUI_WIDGET_VALUE_STEP, 	par);
	gui_widget_add ("phoebe_params_surface_gr2max_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr2max_spinbutton"), 				GUI_WIDGET_VALUE_MAX, 	par);
	gui_widget_add ("phoebe_params_surface_gr2min_spinbutton", 				glade_xml_get_widget(phoebe_window, "phoebe_params_surface_gr2min_spinbutton"), 				GUI_WIDGET_VALUE_MIN, 	par);


	gui_widget_add ("phoebe_params_lumins_levels_primadjust_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_levels_primadjust_checkbutton"),		GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_hla"));
	gui_widget_add ("phoebe_params_lumins_levels_primstep_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_levels_primstep_spinbutton"),			GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_hla"));
	gui_widget_add ("phoebe_params_lumins_levels_secadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_levels_secadjust_checkbutton"),			GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_cla"));
	gui_widget_add ("phoebe_params_lumins_levels_secstep_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_levels_secstep_spinbutton"),				GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_cla"));

	gui_widget_add ("phoebe_params_lumins_3lightajdust_checkbutton",				glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_3lightajdust_checkbutton"),				GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_el3"));
	gui_widget_add ("phoebe_params_lumins_3lightstep_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_3lightstep_spinbutton"),					GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_el3"));
	gui_widget_add ("phoebe_params_lumins_3rdlight_percent_radiobutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_3rdlight_percent_radiobutton"),			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_el3_units"));
	gui_widget_add ("phoebe_params_lumins_3light_opacityadjust_checkbutton",	glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_3light_opacityadjust_checkbutton"),	GUI_WIDGET_SWITCH_TBA, 		phoebe_parameter_lookup ("phoebe_opsf"));
	gui_widget_add ("phoebe_params_lumins_3light_opacitystep_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_3light_opacitystep_spinbutton"),		GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_opsf"));

	gui_widget_add ("phoebe_params_lumins_atmospheres_prim_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_atmospheres_prim_checkbutton"),			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm1_switch"));
	gui_widget_add ("phoebe_params_lumins_atmospheres_sec_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_atmospheres_sec_checkbutton"),			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_atm2_switch"));

	gui_widget_add ("phoebe_params_lumins_options_reflections_checkbutton",		glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_options_reflections_checkbutton"),		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_switch"));
	gui_widget_add ("phoebe_params_lumins_options_reflections_spinbutton",		glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_options_reflections_spinbutton"),		GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_reffect_reflections"));
	gui_widget_add ("phoebe_params_lumins_options_decouple_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_options_decouple_checkbutton"),			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_usecla_switch"));

	gui_widget_add ("phoebe_params_lumins_noise_lcscatter_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_noise_lcscatter_checkbutton"),			GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_synscatter_switch"));
	gui_widget_add ("phoebe_params_lumins_noise_sigma_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_noise_sigma_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_sigma"));
	gui_widget_add ("phoebe_params_lumins_noise_seed_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_noise_seed_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_seed"));
	gui_widget_add ("phoebe_params_lumins_noise_lcscatter_combobox",				glade_xml_get_widget(phoebe_window, "phoebe_params_lumins_noise_lcscatter_combobox"),				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_synscatter_levweight"));


	gui_widget_add ("phoebe_params_ld_model_combobox",									glade_xml_get_widget(phoebe_window, "phoebe_params_ld_model_combobox"),									GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_model"));
	gui_widget_add ("phoebe_params_ld_bolomcoefs_primx_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_params_ld_bolomcoefs_primx_spinbutton"),				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol1"));
	gui_widget_add ("phoebe_params_ld_bolomcoefs_primy_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_params_ld_bolomcoefs_primx_spinbutton"),				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol1"));
	gui_widget_add ("phoebe_params_ld_bolomcoefs_secx_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_params_ld_bolomcoefs_secx_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_xbol2"));
	gui_widget_add ("phoebe_params_ld_bolomcoefs_secy_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_params_ld_bolomcoefs_secx_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_ld_ybol2"));
	gui_widget_add ("phoebe_params_ld_lccoefs_primadjust_checkbutton",			glade_xml_get_widget(phoebe_window, "phoebe_params_ld_lccoefs_primadjust_checkbutton"),			GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_ld_lcx1"));
	gui_widget_add ("phoebe_params_ld_lccoefs_primstep_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_params_ld_lccoefs_primstep_spinbutton"),				GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_ld_lcx1"));
	gui_widget_add ("phoebe_params_ld_lccoefs_secadjust_checkbutton",				glade_xml_get_widget(phoebe_window, "phoebe_params_ld_lccoefs_secadjust_checkbutton"),				GUI_WIDGET_SWITCH_TBA, 	phoebe_parameter_lookup ("phoebe_ld_lcx2"));
	gui_widget_add ("phoebe_params_ld_lccoefs_secstep_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_params_ld_lccoefs_secstep_spinbutton"),					GUI_WIDGET_VALUE_STEP, 	phoebe_parameter_lookup ("phoebe_ld_lcx2"));


	gui_widget_add ("phoebe_params_spots_primmove_checkbutton",						glade_xml_get_widget(phoebe_window, "phoebe_params_spots_primmove_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_spots_no1"));
	gui_widget_add ("phoebe_params_spots_secmove_checkbutton",						glade_xml_get_widget(phoebe_window, "phoebe_params_spots_secmove_checkbutton"),						GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_spots_no2"));


	gui_widget_add ("phoebe_fitting_parameters_finesize1_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_fitting_parameters_finesize1_spinbutton"),				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize1"));
	gui_widget_add ("phoebe_fitting_parameters_finesize2_spinbutton",				glade_xml_get_widget(phoebe_window, "phoebe_fitting_parameters_finesize2_spinbutton"),				GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_finesize2"));
	gui_widget_add ("phoebe_fitting_parameters_coarsesize1_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_fitting_parameters_coarsesize1_spinbutton"),			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize1"));
	gui_widget_add ("phoebe_fitting_parameters_coarsesize2_spinbutton",			glade_xml_get_widget(phoebe_window, "phoebe_fitting_parameters_coarsesize2_spinbutton"),			GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_grid_coarsesize2"));
	gui_widget_add ("phoebe_fitting_parameters_lambda_spinbutton",					glade_xml_get_widget(phoebe_window, "phoebe_fitting_parameters_lambda_spinbutton"),					GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("phoebe_dc_lambda"));



	/* *************************    GUI Widgets   *************************** */

	phoebe_parameter_add ("gui_load_lc_column1",                 "Column 1 of data file",                  KIND_MENU,  NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,       "Time");
	gui_widget_add ("phoebe_load_lc_column1_combobox",								glade_xml_get_widget(phoebe_window, "phoebe_load_lc_column1_combobox"),							GUI_WIDGET_VALUE, 		phoebe_parameter_lookup ("gui_load_lc_column1"));

	return SUCCESS;
}

GUI_widget *gui_widget_new ()
{
	GUI_widget *widget = phoebe_malloc (sizeof (*widget));

	widget->name = NULL;
	widget->type = 0;
	widget->gtk  = NULL;
	widget->par  = NULL;

	return widget;
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

	gui_widget_commit (widget);

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
	GUI_wt_bucket *bucket = GUI_wt->bucket[hash];

	while (bucket) {
		if (strcmp (bucket->widget->name, widget->name) == 0) break;
		bucket = bucket->next;
	}

	if (bucket) {
		/* gui_error(widget already commited...) */
		return SUCCESS;
	}

	else{
		bucket = phoebe_malloc (sizeof (*bucket));

		bucket->widget = widget;
		bucket->next	 = GUI_wt->bucket[hash];
		GUI_wt->bucket[hash] = bucket;
	}

	return SUCCESS;
}

int gui_free_widgets ()
{
	/*
	 * This function frees all widgets from the widget table.
	 */

	int i;
	GUI_wt_bucket *bucket;

	for (i = 0; i < GUI_WT_HASH_BUCKETS; i++) {
		while (GUI_wt->bucket[i]) {
			bucket = GUI_wt->bucket[i];
			GUI_wt->bucket[i] = bucket->next;
			gui_widget_free (bucket->widget);
			free (bucket);
		}
	}
	free (GUI_wt);

	return SUCCESS;
}
