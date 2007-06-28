#include <phoebe/phoebe.h>

#include "phoebe_gui_base.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_callbacks.h"

gboolean
on_phoebe_window_delete_event          (GtkWidget *widget,
                                        GdkEvent  *event,
                                        gpointer   user_data)
{
    gtk_main_quit();
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


void
on_phoebe_fiitting_toolbutton_clicked  (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

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


void
on_phoebe_data_star_name_entry_editing_done
                                        (GtkCellEditable *celleditable,
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
    GtkTreePath       *path;
    GtkTreeIter        iter;
    GtkTreeModel      *model;

    /* get the clicked row */
    gtk_tree_view_get_cursor (tree_view, &path, NULL);

    /* get the model */
    model = gtk_tree_view_get_model(tree_view);

    /* get the clicked row from the model */
    if (gtk_tree_model_get_iter(model, &iter, path))
    {
        char *row_num;
        gtk_tree_model_get(model, &iter, CURVELIST_COL_FILTER, &row_num, -1);
        g_print ("The row number %s, containing a light curve in passband %s, has been clicked.\n", gtk_tree_path_to_string(path), row_num);
        g_free(row_num);
    }
}

void
on_phoebe_data_lc_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data)
{
    gtk_widget_show (phoebe_load_lc_window);
}


void
on_phoebe_data_lc_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_lc_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreePath       *path;
    GtkTreeIter        iter;
    GtkTreeModel      *model;

    /* get the selected row */
    gtk_tree_view_get_cursor ((GtkTreeView*)phoebe_data_lc_treeview, &path, NULL);

    /* get the model */
    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

    /* get the row from the model */
    if (gtk_tree_model_get_iter(model, &iter, path))
    {
        g_print ("The row number %s will be removed.\n", gtk_tree_path_to_string(path));
        gtk_list_store_remove((GtkListStore*)model, &iter);
    }
}


void
on_phoebe_data_rv_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_rv_treeview_cursor_changed
                                        (GtkTreeView *tree_view,
                                         gpointer     user_data)
{
    GtkTreePath       *path;
    GtkTreeIter        iter;
    GtkTreeModel      *model;

    /* get the clicked row */
    gtk_tree_view_get_cursor (tree_view, &path, NULL);

    /* get the model */
    model = gtk_tree_view_get_model(tree_view);

    /* get the clicked row from the model */
    if (gtk_tree_model_get_iter(model, &iter, path))
    {
        char *filename;
        gtk_tree_model_get(model, &iter, CURVELIST_COL_FILENAME, &filename, -1);
        g_print ("The row number %s, containing a RV curve with filename %s, has been clicked.\n", gtk_tree_path_to_string(path), filename);
        g_free(filename);
    }
}


void
on_phoebe_data_rv_add_button_clicked   (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_rv_edit_button_clicked  (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_data_rv_remove_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkTreePath       *path;
    GtkTreeIter        iter;
    GtkTreeModel      *model;

    /* get the selected row */
    gtk_tree_view_get_cursor ((GtkTreeView*)phoebe_data_rv_treeview, &path, NULL);

    /* get the model */
    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_rv_treeview);

    /* get the row from the model */
    if (gtk_tree_model_get_iter(model, &iter, path))
    {
        g_print ("The row number %s will be removed.\n", gtk_tree_path_to_string(path));
        gtk_list_store_remove((GtkListStore*)model, &iter);
    }
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


void
on_phoebe_params_ephemeris_dpdt_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdt_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dpdtmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_periodadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_period_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_period_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_hjd0min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdt_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdt_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtmax_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_dperdtmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0max_spinbutton_change_value
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_perr0_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_pshiftadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_incl_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_incl_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_incladjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_inclstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_inclstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_inclmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_inclmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_inclmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ephemeris_inclmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_ecc_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_ecc_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_eccmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgamin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgamin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgamax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgamax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgastep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgastep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vgaadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vga_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_vga_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rm_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rm_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_rmmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smamin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smamin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smamax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smamax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smastep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smastep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_smaadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_sma_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_sma_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_system_f2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2min_spinbutton_wrapped
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_met1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsv_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsv_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_pcsvmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsvadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsv_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_phsv_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavc_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavc_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavcmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavhmin_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavhmin_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavhmax_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavhmax_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavhstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavhstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavh_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavh_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_tavh_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2min_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_component_logg2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_alb2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr1min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2min_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2min_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2max_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2max_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2step_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2step_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2adjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_surface_gr2_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_edit_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_primadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_secadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_primstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_primstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_secstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_levels_secstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3rdlight_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3light_opacityadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3light_opacitystep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3light_opacitystep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3lightajdust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3lightstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_3lightstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_weighting_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_weighting_edit_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_atmospheres_prim_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_atmospheres_sec_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_atmospheres_grav_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_seed_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_seed_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_seedgen_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_sigma_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_sigma_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_lcscatter_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_noise_lcscatter_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_options_reflections_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_options_decouple_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_options_reflections_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_lumins_options_reflections_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_secy_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_secy_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_primy_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_primy_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_secx_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_secx_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_primx_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_bolomcoefs_primx_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_model_autoupdate_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_model_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_model_tables_claret_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_model_tables_vanhamme_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_treeview_row_activated
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_secstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_secstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_primstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_primstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_secadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_lccoefs_primadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_ld_rvcoefs_treeview_row_collapsed
                                        (GtkTreeView     *treeview,
                                        GtkTreePath     *path,
                                        GtkTreeViewColumn *column,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_primno_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_primno_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_primmove_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_secno_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_secno_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_lonadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_latadjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_radjust_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_tadjsut_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_componentno_comboboxentry_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_spotno_comboboxentry_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_latstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_latstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_lonstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_lonstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_rstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_rstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_tstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust1_tstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_tstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_tstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_tadjsut_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_radjsut_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_latadjsut_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_lonadjsut_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_rstep_spinbutton_remove_widget
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_rstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_lonstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_lonstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_latstep_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_params_spots_adjust2_latstep_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_syn_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_obs_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_obs_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_alias_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_residuals_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_x_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_y_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_phstart_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_phstart_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_phend_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_lc_options_phend_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_obs_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_syn_checkbutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_obs_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_checkbutton49_toggled               (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_checkbutton50_toggled               (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_phend_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_phend_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_phstart_spinbutton_editing_done
                                        (GtkCellEditable *celleditable,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_phstart_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_y_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}


void
on_phoebe_plots_rv_options_x_combobox_changed
                                        (GtkComboBox     *combobox,
                                        gpointer         user_data)
{

}

/**********************************************************************
 *
 *                    phoebe_settings_window events
 *
 **********************************************************************/

gboolean
on_phoebe_settings_window_delete_event (GtkWidget *widget,
                                        GdkEvent  *event,
                                        gpointer   user_data)
{

}

void
on_phoebe_settings_ok_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}

void
on_phoebe_settings_save_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}

void
on_phoebe_settings_cancel_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{

}


void on_phoebe_load_lc_ok_button_clicked
                                        (GtkButton       *button,
                                         gpointer         user_data)
{
    GtkTreeModel *model;
    char *filename;
    PHOEBE_curve *new_lc;

    filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (phoebe_load_lc_filechooserbutton));
    new_lc = phoebe_curve_new_from_file(filename);

    char *itype, *dtype, *wtype;
    phoebe_column_type_get_name(new_lc->itype, &itype);
    phoebe_column_type_get_name(new_lc->dtype, &dtype);
    phoebe_column_type_get_name(new_lc->wtype, &wtype);

    model = gtk_tree_view_get_model((GtkTreeView*)phoebe_data_lc_treeview);

    GtkTreeIter iter;
    gtk_list_store_append((GtkListStore*)model, &iter);
    gtk_list_store_set((GtkListStore*)model, &iter, 0, new_lc->filename, 1, "Undefined", 2, itype, 3, dtype, 4, wtype, 5, new_lc->sigma, -1);

    g_free(itype);
    g_free(dtype);
    g_free(wtype);

    g_free (filename);
    gtk_widget_hide (phoebe_load_lc_window);
}
