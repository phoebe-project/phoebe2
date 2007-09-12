#include <stdlib.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_plotting.h"
#include "phoebe_gui_types.h"

int gui_plot_lc_using_gnuplot ()
{
	PHOEBE_curve *obs;
	PHOEBE_curve *syn;

	PHOEBE_vector *indep;

	gchar *tmpdir;
	gchar oname[255];	/* observed curve filename  */
	gchar sname[255]; 	/* synthetic curve filename */
	gchar cname[255];	/* gnuplot command filename */
	gchar pname[255]; 	/* plot filename			*/
	gchar  line[255];  	/* buffer line				*/

	gint ofd, sfd, cfd, pfd;	/* file descriptors */

	gint i;
	gint status;

	gboolean plot_obs;
	gboolean plot_syn;
	gint vertices = 100;
	gint index = 0;

	GUI_widget *plot_image				= gui_widget_lookup ("phoebe_lc_plot_image");
	GUI_widget *syn_checkbutton 		= gui_widget_lookup ("phoebe_lc_plot_options_syn_checkbutton");
	GUI_widget *obs_checkbutton 		= gui_widget_lookup ("phoebe_lc_plot_options_obs_checkbutton");
	GUI_widget *vertices_no_spinbutton 	= gui_widget_lookup ("phoebe_lc_plot_options_vertices_no_spinbutton");
	GUI_widget *obs_combobox 			= gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox");
	GUI_widget *x_combobox 				= gui_widget_lookup ("phoebe_lc_plot_options_x_combobox");
	GUI_widget *y_combobox				= gui_widget_lookup ("phoebe_lc_plot_options_y_combobox");

	plot_obs = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(obs_checkbutton->gtk));
	plot_syn = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(syn_checkbutton->gtk));

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	if(1){
		index = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox->gtk));

		obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, index);
		syn = phoebe_curve_new ();
		syn->type = PHOEBE_CURVE_LC;

		phoebe_curve_transform (obs, PHOEBE_COLUMN_PHASE, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_UNDEFINED);


		sprintf(oname, "%s/phoebe-lc-XXXXXX", tmpdir);
		ofd = mkstemp (oname);

		for (i=0;i<obs->indep->dim;i++) {
			sprintf(line, "%lf\t%lf\t%lf\n", obs->indep->val[i], obs->dep->val[i], obs->weight->val[i]) ;
			write(ofd, line, strlen(line));
		}
		close(ofd);


		indep = phoebe_vector_new ();
		phoebe_vector_alloc (indep, vertices);
		for (i = 0; i < vertices; i++) indep->val[i] = -0.6 + 1.2 * (double) i/(vertices-1);

		status = phoebe_curve_compute (syn, indep, index, PHOEBE_COLUMN_PHASE, PHOEBE_COLUMN_FLUX);
		phoebe_vector_free (indep);

		sprintf(sname, "%s/phoebe-lc-XXXXXX", tmpdir);
		sfd = mkstemp (sname);

		for (i=0;i<syn->indep->dim;i++) {
			sprintf(line, "%lf\t%lf\n", syn->indep->val[i], syn->dep->val[i]) ;
			write(sfd, line, strlen(line));
		}
		close(sfd);


	//----------------

	sprintf(pname, "%s/phoebe-lc-plot-XXXXXX", tmpdir);
	pfd = mkstemp (pname);

	sprintf(cname, "%s/phoebe-lc-XXXXXX", tmpdir);
	cfd = mkstemp (cname);

	sprintf(line, "set terminal png truecolor nocrop enhanced small size 614,336\n");
	write(cfd, line, strlen(line));
	sprintf(line, "set output '%s'\n", pname);
	write(cfd, line, strlen(line));
	sprintf(line, "set offsets 0.05, 0.05, 0, 0\n");
	write(cfd, line, strlen(line));

	sprintf(line, "set mxtics\n");
	write(cfd, line, strlen(line));
	sprintf(line, "set mytics\n");
	write(cfd, line, strlen(line));

	sprintf(line, "set grid xtics ytics\n");
	write(cfd, line, strlen(line));
	sprintf(line, "set xrange [-0.7:0.7]\n");
	write(cfd, line, strlen(line));
	sprintf(line, "set xlabel 'Phase'\n");
	write(cfd, line, strlen(line));
	sprintf(line, "set ylabel 'Flux'\n");
	write(cfd, line, strlen(line));
	sprintf(line, "set lmargin 5\n");
	write(cfd, line, strlen(line));

	sprintf(line, "plot  '%s' w l notitle,'%s' w p notitle\n", sname, oname);
	write(cfd, line, strlen(line));

	close(cfd);

	sprintf(line,"gnuplot \"%s\"",cname);

	system(line);

	gtk_image_set_from_pixbuf(GTK_IMAGE(plot_image->gtk), gdk_pixbuf_new_from_file(pname, NULL));

	close(pfd);

	remove(oname);
	remove(sname);
	remove(cname);
	remove(pname);


	//----------------

	phoebe_curve_free(obs);
	phoebe_curve_free(syn);
	}

	return SUCCESS;
}

