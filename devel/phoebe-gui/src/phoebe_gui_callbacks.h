#include <gtk/gtk.h>
#include <glade/glade.h>

void
on_phoebe_test_toolbutton_0_clicked      (GtkToolButton   *toolbutton,
                                        	gpointer         user_data);

void
on_phoebe_test_toolbutton_1_clicked      (GtkToolButton   *toolbutton,
                                        	gpointer         user_data);

void on_phoebe_sidesheet_data_tba_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data);

void
on_phoebe_open_toolbutton_clicked      (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_phoebe_save_toolbutton_clicked      (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_phoebe_file_new_menuitem_activate   (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_file_open_menuitem_activate  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_file_save_menuitem_activate  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_file_saveas_menuitem_activate
                                        (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_file_quit_menuitem_activate  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_settings_configuration_menuitem_activate
                                        (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_help_about_menuitem_activate (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_phoebe_fiitting_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_phoebe_scripter_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_phoebe_settings_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_phoebe_quit_toolbutton_clicked      (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_phoebe_data_star_name_entry_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_data_star_model_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_phoebe_data_lcoptions_mag_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_data_lcoptions_mag_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_data_rvoptions_psepe_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_data_rvoptions_ssepe_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_data_lc_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_data_lc_active_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data);

void
on_phoebe_data_lc_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_data_lc_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_data_lc_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_data_rv_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_data_rv_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_data_rv_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_data_rv_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_data_rv_active_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data);

void
on_phoebe_data_options_bins_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_data_options_binsno_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_data_options_binsno_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_data_options_the_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_data_options_the_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdt_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdt_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dpdtmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_periodadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_period_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_period_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_hjd0min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdt_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdt_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtmax_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_dperdtmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0max_spinbutton_change_value
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_perr0_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_eph_pshiftadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_incl_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_incl_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_incladjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_inclstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_inclstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_inclmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_inclmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_inclmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_inclmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_ecc_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_ecc_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_eccmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgamin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgamin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgamax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgamax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgastep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgastep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vgaadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vga_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_vga_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rm_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rm_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_rmmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smamin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smamin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smamax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smamax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smastep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smastep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_smaadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_sma_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_sma_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_sys_f2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2min_spinbutton_wrapped
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_met1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsv_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsv_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_pcsvmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsvadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsv_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_phsv_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavc_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavc_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavcmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavhmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavhmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavhmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavhmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavhstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavhstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavh_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavh_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_tavh_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2min_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_comp_logg2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_alb2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_surf_gr2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_surf_spots_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_para_surf_spots_adjust_checkbutton_toggled
                                        (GtkCellRendererToggle *renderer,
                                         gchar                 *path,
                                         gpointer               user_data);

void
on_phoebe_para_surf_spots_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_surf_spots_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_surf_spots_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_edit_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_primadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_secadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_primstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_primstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_secstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_levels_secstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3_opacityadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3_opacitystep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3_opacitystep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3ajdust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_el3step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_weighting_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_para_lum_weighting_edit_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_lum_atmospheres_prim_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_atmospheres_sec_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_atmospheres_grav_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_seed_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_seed_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_seedgen_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_sigma_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_sigma_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_lcscatter_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_noise_lcscatter_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_phoebe_para_lum_options_reflections_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_options_decouple_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_lum_options_reflections_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_lum_options_reflections_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_secy_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_secy_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_primy_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_primy_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_secx_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_secx_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_primx_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_ld_bolcoefs_primx_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_model_autoupdate_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_model_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_phoebe_para_ld_model_tables_claret_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_ld_model_tables_vanhamme_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_secstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_secstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_primstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_primstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_secadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_para_ld_lccoefs_primadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_phoebe_load_lc_filechooserbutton_selection_changed
										(GtkFileChooserButton *filechooserbutton,
										gpointer user_data);

void
on_phoebe_load_rv_filechooserbutton_selection_changed
										(GtkFileChooserButton *filechooserbutton,
										gpointer user_data);

void
on_phoebe_lc_plot_detach_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);
