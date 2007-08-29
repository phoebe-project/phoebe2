#include <phoebe/phoebe.h>

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_types.h"

void set_text_view_from_file (GtkWidget *text_view, gchar *filename)
{
	/*
	 * This function fills the text view text_view with the first maxlines lines
	 * of a file filename.
	 */

	GtkTextBuffer *text_buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (text_view));
	GtkTextIter iter;

	FILE *file = fopen(filename, "r");

	if(file){
		char line[255];
		int i=1;
		int maxlines=50;

		fgets (line, 255, file);
		gtk_text_buffer_set_text (text_buffer, line, -1);

		gtk_text_buffer_get_iter_at_line (text_buffer, &iter, i);
			while(!feof (file) && i<maxlines){
				fgets (line, 255, file);
				gtk_text_buffer_insert (text_buffer, &iter, line, -1);
				i++;
			}

		if(!feof (file))gtk_text_buffer_insert (text_buffer, &iter, "...", -1);

		fclose(file);
	}
}

void detach_box_from_parent (GtkWidget *box, GtkWidget *parent, gboolean *flag, gchar *window_title, gint x, gint y)
{
	/*
	 * This function detaches the box from its parent. If the flag=FALSE, than
	 * it creates a new window and packs the box inside the window, otherwise
	 * it packs the box in its original place inside the main window.
	 */

	GtkWidget *window;

	if(*flag){
		window = gtk_widget_get_parent(box);

		gtk_widget_reparent(box, parent);
		gtk_widget_destroy(window);
		*flag = !(*flag);
	}
	else{
		window = gtk_window_new (GTK_WINDOW_TOPLEVEL);

		gtk_window_set_icon (GTK_WINDOW(window), gdk_pixbuf_new_from_file("ico.png", NULL));
		gtk_window_set_title (GTK_WINDOW (window), window_title);
		gtk_widget_reparent(box, window);
		gtk_widget_set_size_request (window, x, y);
		gtk_window_set_deletable(GTK_WINDOW(window), FALSE);
		gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
		gtk_widget_show_all (window);
		*flag = !(*flag);
	}
}

int gui_open_parameter_file()
{
	GtkWidget *dialog;
	int status = 0;

	dialog = gtk_file_chooser_dialog_new ("Open PHOEBE parameter file",
										  GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
										  GTK_FILE_CHOOSER_ACTION_OPEN,
										  GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
										  NULL);


	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT){
		char *filename;

		filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
		status = phoebe_open_parameter_file(filename);
		g_free (filename);
	}

	gtk_widget_destroy (dialog);

	return status;
}

int gui_save_parameter_file()
{
	GtkWidget *dialog;
	int status = 0;

	dialog = gtk_file_chooser_dialog_new ("Save PHOEBE parameter file",
										  GTK_WINDOW(gui_widget_lookup("phoebe_window")->gtk),
										  GTK_FILE_CHOOSER_ACTION_SAVE,
										  GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
										  GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
										  NULL);

	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT){
		char *filename;

		filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
		status = phoebe_save_parameter_file(filename);
		g_free (filename);
	}

	gtk_widget_destroy (dialog);

	return status;
}
