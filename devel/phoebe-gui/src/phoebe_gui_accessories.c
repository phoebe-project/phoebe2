#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_build_config.h"

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_error_handling.h"
#include "phoebe_gui_types.h"
#include "phoebe_gui_global.h"

gchar   *PHOEBE_FILENAME = NULL;
gboolean PHOEBE_FILEFLAG = FALSE;

gchar   *PHOEBE_DIRNAME = NULL;
gboolean PHOEBE_DIRFLAG = FALSE;

int      PHOEBE_STATUS_MESSAGES_COUNT = 0;
int      PHOEBE_STATUS_MESSAGES_MAX_COUNT = 10;

void gui_set_text_view_from_file (GtkWidget *text_view, gchar *filename)
{
	/*
	 * This function fills the text view text_view with the first maxlines lines
	 * of a file filename.
	 */

	GtkTextBuffer *text_buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (text_view));
	GtkTextIter iter;

	FILE *file = fopen(filename, "r");

	if (file) {
		char line[255];
		int i=1;
		int maxlines=50;

		if (!fgets (line, 255, file)) {
			fclose (file);
			return;
		}
		
		gtk_text_buffer_set_text (text_buffer, line, -1);
		
		gtk_text_buffer_get_iter_at_line (text_buffer, &iter, i);
		while (fgets (line, 255, file) && i < maxlines) {
			gtk_text_buffer_insert (text_buffer, &iter, line, -1);
			i++;
		}
		
		if (!feof (file))
			gtk_text_buffer_insert (text_buffer, &iter, ". . . . . . .", -1);
		
		fclose(file);
	}
}

void tmp_circumvent_delete_event (GtkWidget *widget, gpointer user_data)
{
	GtkWidget *box 		= g_object_get_data (G_OBJECT (widget), "data_box");
	GtkWidget *parent 	= g_object_get_data (G_OBJECT (widget), "data_parent");
	gboolean *flag 		= g_object_get_data (G_OBJECT (widget), "data_flag");

	GtkWidget *window = gtk_widget_get_parent(box);

	gtk_widget_reparent(box, parent);
	gtk_widget_destroy(window);

	*flag = !(*flag);
}

GtkWidget *gui_show_temp_window()
{
	GtkWidget *temp_window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(temp_window), "Working...");
	gtk_window_set_default_size(GTK_WINDOW(temp_window), 300, 0);
	gtk_window_set_position(GTK_WINDOW(temp_window), GTK_WIN_POS_CENTER_ALWAYS);
	gtk_window_set_decorated(GTK_WINDOW(temp_window), FALSE);
	gtk_widget_show_all(temp_window);
	return temp_window;
}

void gui_hide_temp_window(GtkWidget *temp_window, GtkWidget *new_active_window)
{
	gtk_widget_destroy(temp_window);
	gtk_window_present(GTK_WINDOW(new_active_window));
}

GtkWidget *gui_detach_box_from_parent (GtkWidget *box, GtkWidget *parent, gboolean *flag, gchar *window_title, gint x, gint y)
{
	/*
	 * This function detaches the box from its parent. If the flag=FALSE, than
	 * it creates a new window and packs the box inside the window, otherwise
	 * it packs the box in its original place inside the main window.
	 */

	GtkWidget *window;
	gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);

	if(*flag){
		window = gtk_widget_get_parent (box);

		gtk_widget_reparent(box, parent);
		gtk_widget_destroy(window);

        gui_status("%s reatached.", window_title);
	}
	else{
		window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
		g_object_set_data (G_OBJECT (window), "data_box", 		(gpointer) box);
		g_object_set_data (G_OBJECT (window), "data_parent", 	(gpointer) parent);
		g_object_set_data (G_OBJECT (window), "data_flag",		(gpointer) flag);

		gtk_window_set_icon (GTK_WINDOW(window), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));
		gtk_window_set_title (GTK_WINDOW (window), window_title);
		gtk_widget_reparent(box, window);
		gtk_widget_set_size_request (window, x, y);
		gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
		g_signal_connect (GTK_WIDGET(window), "delete-event", G_CALLBACK (tmp_circumvent_delete_event), NULL);
		gtk_widget_show_all (window);

		gui_status("%s detached.", window_title);
	}
	*flag = !(*flag);
	return window;
}

void gui_beep()
{
	bool beep;

	phoebe_config_entry_get ("GUI_BEEP_AFTER_PLOT_AND_FIT", &beep);
	if (beep) {
		gdk_beep();
	}
}

int gui_open_parameter_file ()
{
	GtkWidget *dialog;
	gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
	int status = -1;
	char *filename;

	dialog = gtk_file_chooser_dialog_new (
		"Open PHOEBE parameter file",
		GTK_WINDOW (gui_widget_lookup ("phoebe_window")->gtk),
		GTK_FILE_CHOOSER_ACTION_OPEN,
		GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
		GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
		NULL);

	if (PHOEBE_DIRFLAG)
		gtk_file_chooser_set_current_folder ((GtkFileChooser*) dialog, PHOEBE_DIRNAME);
	else {
		gchar *dir;
		phoebe_config_entry_get ("PHOEBE_DATA_DIR", &dir);
		gtk_file_chooser_set_current_folder ((GtkFileChooser*) dialog, dir);
	}

	gtk_window_set_default_size (GTK_WINDOW (dialog), 600, 450);
    gtk_window_set_icon (GTK_WINDOW (dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));

	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT) {
		filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
		status = phoebe_open_parameter_file (filename);

		PHOEBE_FILEFLAG = TRUE;
		PHOEBE_FILENAME = strdup (filename);

		PHOEBE_DIRFLAG = TRUE;
		PHOEBE_DIRNAME = gtk_file_chooser_get_current_folder (GTK_FILE_CHOOSER (dialog));

		if (status == SUCCESS) {
			gui_update_angle_values ();
			gui_status("%s successfully opened.", filename);
		}
        else
			gui_status("Opening %s failed with status %d.", filename, status);

        g_free (filename);
	}
	else
		gui_status ("Open PHOEBE parameter file canceled.");

	gtk_widget_destroy (dialog);

	return status;
}

gchar *gui_get_filename_with_overwrite_confirmation(GtkWidget *dialog, char *gui_confirmation_title)
{
	/* Returns a valid filename or NULL if the user canceled
	   Replaces functionality of gtk_file_chooser_set_do_overwrite_confirmation
	   avaliable in Gtk 2.8 (PHOEBE only requires 2.6)
	*/
	gchar *filename;

	while (TRUE) {
		if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT){
			filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
			if (!phoebe_filename_exists(filename))
				/* file does not exist */
				return filename;
			if(gui_warning(gui_confirmation_title, "This file already exists. Do you want to replace it?")){
				/* file may be overwritten */
				return filename;
			}
			/* user doesn't want to overwrite, display the dialog again */
			g_free (filename);
		}
		else{
		    gui_status("%s cancelled.", gui_confirmation_title);
		    return NULL;
		}
	}
}

int gui_save_parameter_file()
{
	GtkWidget *dialog;
	gchar *glade_pixmap_file = g_build_filename (PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
	int status = 0;

	dialog = gtk_file_chooser_dialog_new ("Save PHOEBE parameter file",
										  GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
										  GTK_FILE_CHOOSER_ACTION_SAVE,
										  GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
										  NULL);

	gtk_window_set_default_size (GTK_WINDOW (dialog), 600, 450);

	if(PHOEBE_DIRFLAG)
		gtk_file_chooser_set_current_folder((GtkFileChooser*)dialog, PHOEBE_DIRNAME);
	else{
		gchar *dir;
		phoebe_config_entry_get("PHOEBE_DATA_DIR", &dir);
		gtk_file_chooser_set_current_folder((GtkFileChooser*)dialog, dir);
	}

	/* gtk_file_chooser_set_do_overwrite_confirmation (GTK_FILE_CHOOSER (dialog), TRUE); */
    gtk_window_set_icon (GTK_WINDOW(dialog), gdk_pixbuf_new_from_file(glade_pixmap_file, NULL));

	gchar *filename = gui_get_filename_with_overwrite_confirmation (dialog, "Save PHOEBE parameter file");
	if (filename) {
		gui_export_angles_to_radians ();
		status = phoebe_save_parameter_file (filename);

		PHOEBE_FILEFLAG = TRUE;
		PHOEBE_FILENAME = strdup(filename);

		PHOEBE_DIRFLAG = TRUE;
		PHOEBE_DIRNAME = gtk_file_chooser_get_current_folder (GTK_FILE_CHOOSER (dialog));

		if (status == SUCCESS)
			gui_status("%s successfully saved.", filename);
        else
			gui_status("Saving %s failed with status %d.", filename, status);

        g_free (filename);
	}

	gtk_widget_destroy (dialog);

	return status;
}

int gui_show_configuration_dialog ()
{
	int status = 0;
	
	gchar     *glade_xml_file					= g_build_filename     	(PHOEBE_GLADE_XML_DIR, "phoebe_settings.glade", NULL);
	gchar     *glade_pixmap_file				= g_build_filename     	(PHOEBE_GLADE_PIXMAP_DIR, "ico.png", NULL);
	
	GladeXML  *phoebe_settings_xml				= glade_xml_new			(glade_xml_file, NULL, NULL);
	
	GtkWidget *phoebe_settings_dialog			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_dialog");
	GtkWidget *basedir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_basedir_filechooserbutton");
	GtkWidget *defaultsdir_filechooserbutton	= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_defaultsdir_filechooserbutton");
	GtkWidget *workingdir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_workingdir_filechooserbutton");
	GtkWidget *datadir_filechooserbutton		= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_datadir_filechooserbutton");
	GtkWidget *ptfdir_filechooserbutton			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_configuration_ptfdir_filechooserbutton");

	GtkWidget *ld_none                          = glade_xml_get_widget  (phoebe_settings_xml, "phoebe_settings_ld_none");
	GtkWidget *ld_internal                      = glade_xml_get_widget  (phoebe_settings_xml, "phoebe_settings_ld_internal_tables");
	GtkWidget *ld_internal_dir		            = glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_ld_internal_dir");
	GtkWidget *ld_vanhamme						= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_ld_vanhamme_tables");
	GtkWidget *ld_vanhamme_dir					= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_ld_vanhamme_dir");
	
	GtkWidget *kurucz_checkbutton				= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_kurucz_checkbutton");
	GtkWidget *kurucz_filechooserbutton			= glade_xml_get_widget	(phoebe_settings_xml, "phoebe_settings_kurucz_filechooserbutton");
	
	GtkWidget *confirm_on_overwrite_checkbutton = glade_xml_get_widget (phoebe_settings_xml, "phoebe_settings_confirmation_save_checkbutton");
	GtkWidget *beep_after_plot_and_fit_checkbutton = glade_xml_get_widget (phoebe_settings_xml, "phoebe_settings_beep_after_plot_and_fit_checkbutton");
	GtkWidget *units_widget = glade_xml_get_widget (phoebe_settings_xml, "phoebe_settings_angle_units");
	gchar     *dir;
	gboolean   toggle;
	gint 	   result;
	
	/* Connect all signals defined in the Glade file: */
	glade_xml_signal_autoconnect (phoebe_settings_xml);
	
	g_object_unref (phoebe_settings_xml);
	
	phoebe_config_entry_get ("PHOEBE_BASE_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) basedir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_DEFAULTS_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) defaultsdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) workingdir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_DATA_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) datadir_filechooserbutton, dir);
	phoebe_config_entry_get ("PHOEBE_PTF_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) ptfdir_filechooserbutton, dir);
	
	g_signal_connect (G_OBJECT (confirm_on_overwrite_checkbutton),    "toggled", G_CALLBACK (on_phoebe_settings_confirmation_save_checkbutton_toggled), NULL);
	g_signal_connect (G_OBJECT (beep_after_plot_and_fit_checkbutton), "toggled", G_CALLBACK (on_phoebe_beep_after_plot_and_fit_checkbutton_toggled),    NULL);
	
	gtk_window_set_icon  (GTK_WINDOW (phoebe_settings_dialog), gdk_pixbuf_new_from_file (glade_pixmap_file, NULL));
	gtk_window_set_title (GTK_WINDOW (phoebe_settings_dialog), "PHOEBE -- Settings");
	
	phoebe_config_entry_get ("PHOEBE_LD_SWITCH", &toggle);
	
	if (!toggle)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (ld_none), TRUE);
	else {
		phoebe_config_entry_get ("PHOEBE_LD_INTERN", &toggle);
		if (toggle)
			gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (ld_internal), TRUE);
		else
			gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (ld_vanhamme), TRUE);
	}
	
	phoebe_config_entry_get ("PHOEBE_LD_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) ld_internal_dir, dir);
	
	phoebe_config_entry_get ("PHOEBE_LD_VH_DIR", &dir);
	gtk_file_chooser_set_filename ((GtkFileChooser *) ld_vanhamme_dir, dir);
	
	phoebe_config_entry_get ("PHOEBE_KURUCZ_SWITCH", &toggle);
	if (toggle) {
		phoebe_config_entry_get ("PHOEBE_KURUCZ_DIR", &dir);
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON(kurucz_checkbutton), TRUE);
		gtk_file_chooser_set_filename((GtkFileChooser*)kurucz_filechooserbutton, dir);
	}
	
	phoebe_config_entry_get ("GUI_CONFIRM_ON_OVERWRITE", &toggle);
	if (toggle)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (confirm_on_overwrite_checkbutton), 1);
	else
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (confirm_on_overwrite_checkbutton), 0);
	
	phoebe_config_entry_get ("GUI_BEEP_AFTER_PLOT_AND_FIT", &toggle);
	if (toggle)
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (beep_after_plot_and_fit_checkbutton), 1);
	else
		gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (beep_after_plot_and_fit_checkbutton), 0);
	
	/* ANGLE UNITS: */
	{
		char *units;
		phoebe_config_entry_get ("GUI_ANGLE_UNITS", &units);
		if (strcmp(units, "Radians") == 0)
			gtk_combo_box_set_active (GTK_COMBO_BOX(units_widget), 0);
		else
			gtk_combo_box_set_active (GTK_COMBO_BOX(units_widget), 1);
	}

	/* Now that everything is set according to the config, add a signal to
	 * change angle units:
	 */
	g_signal_connect (units_widget, "changed", G_CALLBACK(on_angle_units_changed), NULL);

	result = gtk_dialog_run (GTK_DIALOG (phoebe_settings_dialog));
	
	switch (result) {
		case GTK_RESPONSE_OK:  /* =  OK button  */
		case GTK_RESPONSE_YES: /* = Save button */
			phoebe_config_entry_set ("PHOEBE_BASE_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser *) basedir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DEFAULTS_DIR", gtk_file_chooser_get_filename ((GtkFileChooser *) defaultsdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_TEMP_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser *) workingdir_filechooserbutton));
			phoebe_config_entry_set ("PHOEBE_DATA_DIR", 	gtk_file_chooser_get_filename ((GtkFileChooser *) datadir_filechooserbutton));

			dir = gtk_file_chooser_get_filename ((GtkFileChooser *) ptfdir_filechooserbutton);
			phoebe_config_entry_set ("PHOEBE_PTF_DIR", dir);
			free(PHOEBE_passbands);
			PHOEBE_passbands = NULL;
			PHOEBE_passbands_no = 0;
			phoebe_read_in_passbands (dir);
			g_free (dir);
						
			phoebe_config_entry_set ("PHOEBE_LD_DIR",    gtk_file_chooser_get_filename ((GtkFileChooser*) ld_internal_dir));
			phoebe_config_entry_set ("PHOEBE_LD_VH_DIR", gtk_file_chooser_get_filename ((GtkFileChooser*) ld_vanhamme_dir));
			
			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (ld_none))) {
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", FALSE);
			}
			else if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (ld_internal))) {
				char *pathname;
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", TRUE);
				phoebe_config_entry_set ("PHOEBE_LD_INTERN", TRUE);
				phoebe_config_entry_get ("PHOEBE_LD_DIR", &pathname); 
				phoebe_ld_attach_all (pathname);
			}
			else {
				phoebe_config_entry_set ("PHOEBE_LD_SWITCH", TRUE);
				phoebe_config_entry_set ("PHOEBE_LD_INTERN", FALSE);
			}
			
			phoebe_load_ld_tables ();

			if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (kurucz_checkbutton))) {
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", TRUE);
				phoebe_config_entry_set ("PHOEBE_KURUCZ_DIR",    gtk_file_chooser_get_filename ((GtkFileChooser *) kurucz_filechooserbutton));
			}
			else
				phoebe_config_entry_set ("PHOEBE_KURUCZ_SWITCH", FALSE);

			if (gtk_combo_box_get_active (GTK_COMBO_BOX (units_widget)) == 1)
				phoebe_config_entry_set ("GUI_ANGLE_UNITS", "Degrees");
			else
				phoebe_config_entry_set ("GUI_ANGLE_UNITS", "Radians");

			if (result == GTK_RESPONSE_YES) {
				if (!PHOEBE_HOME_DIR || !phoebe_filename_is_directory (PHOEBE_HOME_DIR)) {
					char homedir[255], confdir[255];

					sprintf (homedir, "%s/.phoebe-%s", USER_HOME_DIR, PACKAGE_VERSION);
					sprintf (confdir, "%s/phoebe.config", homedir);

					PHOEBE_HOME_DIR = strdup (homedir);
					PHOEBE_CONFIG   = strdup (confdir);

#ifdef __MINGW32__
					mkdir (PHOEBE_HOME_DIR);
#else
					mkdir (PHOEBE_HOME_DIR, 0755);
#endif
				}

				phoebe_config_save (PHOEBE_CONFIG);

				if (status == SUCCESS)
					gui_status ("Configuration successfully saved.");
                		else
					gui_status ("Configuration failed: %s", phoebe_gui_error (status));
			}
        break;
		case GTK_RESPONSE_CANCEL:
            gui_status ("Configuration aborted.");
		break;
	}
	
	gtk_widget_destroy (phoebe_settings_dialog);
	
	return status;
}

int gui_warning (char *title, char *message)
{
	GtkWidget *dialog;
	int answer = 0;
	
	dialog = gtk_message_dialog_new ( GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
									  GTK_DIALOG_DESTROY_WITH_PARENT,
									  GTK_MESSAGE_WARNING,
									  GTK_BUTTONS_NONE,
									  message);
	
	gtk_dialog_add_buttons(GTK_DIALOG (dialog), GTK_STOCK_NO, GTK_RESPONSE_REJECT,
										  		GTK_STOCK_YES, GTK_RESPONSE_ACCEPT,
										  		NULL);
	
	gtk_window_set_title(GTK_WINDOW (dialog), title);
	
	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
		answer = 1;
	
	gtk_widget_destroy (dialog);
	
	return answer;
}

int gui_question (char *title, char *message)
{
	GtkWidget *dialog;
	int answer = 0, response;
	
	dialog = gtk_message_dialog_new (GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
									 GTK_DIALOG_DESTROY_WITH_PARENT,
									 GTK_MESSAGE_QUESTION,
	    							 GTK_BUTTONS_YES_NO,
									 message);
	
	gtk_window_set_title (GTK_WINDOW (dialog), title);

	response = gtk_dialog_run (GTK_DIALOG (dialog));
	if (response == GTK_RESPONSE_ACCEPT || response == GTK_RESPONSE_YES || response == GTK_RESPONSE_OK)
		answer = 1;
	
	gtk_widget_destroy (dialog);
	
	return answer;
}

int gui_notice(char* title, char* message)
{
	GtkWidget *dialog;
	int answer = 0;

	dialog = gtk_message_dialog_new ( GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
									  GTK_DIALOG_DESTROY_WITH_PARENT,
									  GTK_MESSAGE_INFO,
									  GTK_BUTTONS_NONE,
									  message);

	gtk_dialog_add_buttons(GTK_DIALOG (dialog), GTK_STOCK_OK, GTK_RESPONSE_ACCEPT,
										  		NULL);

	gtk_window_set_title(GTK_WINDOW (dialog), title);

	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
		answer = 1;

	gtk_widget_destroy (dialog);

	return answer;
}

int gui_error(char* title, char* message)
{
	GtkWidget *dialog;
	int answer = 0;

	dialog = gtk_message_dialog_new ( GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
									  GTK_DIALOG_DESTROY_WITH_PARENT,
									  GTK_MESSAGE_ERROR,
									  GTK_BUTTONS_NONE,
									  message);

	gtk_dialog_add_buttons(GTK_DIALOG (dialog), GTK_STOCK_OK, GTK_RESPONSE_ACCEPT,
										  		NULL);

	gtk_window_set_title(GTK_WINDOW (dialog), title);

	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
		answer = 1;

	gtk_widget_destroy (dialog);

	return answer;
}

int gui_status (const char *format, ...)
{
    GtkStatusbar *status = GTK_STATUSBAR(gui_widget_lookup("phoebe_statusbar")->gtk);
    int i;
    int result = -1;
    char message[256];

    if(PHOEBE_STATUS_MESSAGES_COUNT >= PHOEBE_STATUS_MESSAGES_MAX_COUNT){
        for(i=0; i<PHOEBE_STATUS_MESSAGES_MAX_COUNT; i++)gtk_statusbar_pop(status, i);
        PHOEBE_STATUS_MESSAGES_COUNT = 0;
    }

    va_list args;
    va_start (args, format);
    result = vsprintf (message, format, args);
    va_end (args);

    gtk_statusbar_push(status, PHOEBE_STATUS_MESSAGES_COUNT, (const gchar*)message);
    PHOEBE_STATUS_MESSAGES_COUNT += 1;

    return result;
}

int gui_update_el3_lum_value ()
{
	int i, lcno, status;
	double L1, L2, l3;
	PHOEBE_el3_units l3units;

	status = phoebe_el3_units_id (&l3units);
	if (status != SUCCESS) {
		gui_warning ("Third light computation issue", "Third light units are not set or are invalid. Please review the settings in the Luminosities tab.");
		return SUCCESS;
	}

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);

	for (i = 0; i < lcno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"), i, &l3);

		/* If third light units are flux, then simply update the value and exit: */
		if (l3units == PHOEBE_EL3_UNITS_FLUX)
			phoebe_parameter_set_value (phoebe_parameter_lookup ("gui_el3_lum_units"), i, 4*M_PI*l3);
		else {
			/* Otherwise we need to convert. */
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hla"), i, &L1);
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cla"), i, &L2);
			
			phoebe_parameter_set_value (phoebe_parameter_lookup ("gui_el3_lum_units"), i, (L1+L2)*l3/(1.-l3));
		}
	}

	return SUCCESS;
}
