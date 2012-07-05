#include <gtk/gtk.h>
#include <glade/glade.h>

extern gboolean PHOEBE_FILEFLAG;
extern gchar *PHOEBE_FILENAME;

/* Generic callbacks that should be used by all respective widget types: */

G_MODULE_EXPORT void on_combo_box_selection_changed_get_index  (GtkComboBox     *combo,        gpointer user_data);
G_MODULE_EXPORT void on_combo_box_selection_changed_get_string (GtkComboBox     *combo,        gpointer user_data);
G_MODULE_EXPORT void on_spin_button_value_changed              (GtkSpinButton   *spinbutton,   gpointer user_data);
G_MODULE_EXPORT void on_spin_button_intvalue_changed           (GtkSpinButton   *spinbutton,   gpointer user_data);
G_MODULE_EXPORT void on_toggle_button_value_toggled            (GtkToggleButton *togglebutton, gpointer user_data);
G_MODULE_EXPORT void on_toggle_make_sensitive                  (GtkToggleButton *togglebutton, gpointer user_data);
G_MODULE_EXPORT void on_toggle_make_unsensitive                (GtkToggleButton *togglebutton, gpointer user_data);

/* Plotting callbacks: */

G_MODULE_EXPORT void on_plot_controls_reset_button_clicked     (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_controls_right_button_clicked     (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_controls_up_button_clicked        (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_controls_left_button_clicked      (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_controls_down_button_clicked      (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_controls_zoomin_button_clicked    (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_controls_zoomout_button_clicked   (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_save_button_clicked               (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_clear_button_clicked              (GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_plot_save_data_button_clicked          (GtkButton *button, gpointer user_data);

/* Specific callbacks that pertain to the particular action: */

G_MODULE_EXPORT void on_phoebe_test_toolbutton_0_clicked 					(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_test_toolbutton_1_clicked 					(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_open_toolbutton_clicked 						(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_save_toolbutton_clicked 						(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_fitting_toolbutton_clicked 					(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_scripter_toolbutton_clicked 					(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_settings_toolbutton_clicked 					(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_quit_toolbutton_clicked 						(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_file_new_menuitem_activate 					(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_file_open_menuitem_activate 					(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_file_save_menuitem_activate 					(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_file_saveas_menuitem_activate 				(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_file_import_bm3_menuitem_activate            (GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_file_quit_menuitem_activate 					(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_settings_configuration_menuitem_activate 	(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_help_about_menuitem_activate 				(GtkMenuItem *menuitem, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_lc_treeview_row_activated 				(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_lc_active_checkbutton_toggled 			(GtkCellRendererToggle *renderer, gchar *path, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_lc_add_button_clicked 					(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_lc_edit_button_clicked 					(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_lc_remove_button_clicked 				(GtkButton *button, gpointer user_data);
/* G_MODULE_EXPORT void on_phoebe_data_lc_model_row_changed 					(GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data); */
G_MODULE_EXPORT void on_phoebe_data_lc_seedgen_button_clicked 				(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_rv_treeview_row_activated 				(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_rv_add_button_clicked 					(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_rv_edit_button_clicked 					(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_rv_remove_button_clicked 				(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_rv_active_checkbutton_toggled 			(GtkCellRendererToggle *renderer, gchar *path, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_data_rv_model_row_changed 					(GtkTreeModel *tree_model, GtkTreePath  *path, GtkTreeIter *iter, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_adjust_checkbutton_toggled 		(GtkCellRendererToggle *renderer, gchar *path, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_active_checkbutton_toggled 		(GtkCellRendererToggle *renderer, gchar *path, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_add_button_clicked 				(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_edit_button_clicked 				(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_remove_button_clicked 			(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_treeview_row_activated 			(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_lum_levels_treeview_row_activated 		(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_lum_levels_edit_button_clicked 			(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_lum_levels_calc_button_clicked 			(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_lum_el3_treeview_row_activated 			(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_lum_weighting_treeview_row_activated 	(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_lum_weighting_edit_button_clicked 		(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_ld_lccoefs_edit_button_clicked 			(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_ld_lccoefs_treeview_row_activated 		(GtkTreeView *treeview, GtkTreePath *path, GtkTreeViewColumn *column, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_load_lc_filechooserbutton_selection_changed 	(GtkFileChooserButton *filechooserbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_load_rv_filechooserbutton_selection_changed 	(GtkFileChooserButton *filechooserbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_lc_plot_detach_button_clicked 				(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_lc_plot_options_x_combobox_changed 			(GtkComboBox *widget, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_rv_plot_detach_button_clicked 				(GtkButton *button, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_rv_plot_options_x_combobox_changed 			(GtkComboBox *widget, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_settings_toolbutton_clicked 					(GtkToolButton *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_fitt_updateall_button_clicked 				(GtkToolButton   *toolbutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_fitt_method_combobox_changed 				(GtkComboBox *widget, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_settings_confirmation_save_checkbutton_toggled (GtkToggleButton *togglebutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_beep_after_plot_and_fit_checkbutton_toggled  (GtkToggleButton *togglebutton, gpointer user_data);
G_MODULE_EXPORT void on_phoebe_para_spots_treeview_cursor_changed           (GtkTreeView *tree_view, gpointer user_data);

G_MODULE_EXPORT void on_critical_potentials_changed                         (GtkSpinButton *spinbutton, gpointer user_data);
G_MODULE_EXPORT void on_orbital_elements_changed                            (GtkSpinButton *spinbutton, gpointer user_data);
G_MODULE_EXPORT void on_star_shape_changed                                  (GtkSpinButton *spinbutton, gpointer user_data);
G_MODULE_EXPORT void on_stellar_masses_changed                              (GtkSpinButton *spinbutton, gpointer user_data);
G_MODULE_EXPORT void on_orbital_elements_changed                            (GtkSpinButton *spinbutton, gpointer user_data);
G_MODULE_EXPORT void on_angle_units_changed                                 (GtkComboBox *widget, gpointer user_data);
G_MODULE_EXPORT void on_auto_logg_switch_toggled                            (GtkToggleButton *togglebutton, gpointer user_data);

void gui_ld_coeffs_need_updating();
void gui_on_fitting_finished (int status);
