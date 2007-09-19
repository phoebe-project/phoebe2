#include <stdlib.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_plotting.h"
#include "phoebe_gui_types.h"

int gui_plot_get_curve_limits (PHOEBE_curve *curve, double *xmin, double *ymin, double *xmax, double *ymax)
{
	int i;

	*xmin = 0; *xmax = 0;
	*ymin = 0; *ymax = 0;

	*xmin = curve->indep->val[0]; *xmax = curve->indep->val[0];
	*ymin = curve->dep->val[0];   *ymax = curve->dep->val[0];
	
	for (i = 1; i < curve->indep->dim; i++) {
		if (*xmin > curve->indep->val[i]) *xmin = curve->indep->val[i];
		if (*xmax < curve->indep->val[i]) *xmax = curve->indep->val[i];
		if (*ymin > curve->dep->val[i]  ) *ymin = curve->dep->val[i];
		if (*ymax < curve->dep->val[i]  ) *ymax = curve->dep->val[i];
	}

	return SUCCESS;
}

int gui_plot_get_plot_limits (PHOEBE_curve *syn, PHOEBE_curve *obs, double *xmin, double *ymin, double *xmax, double *ymax, gboolean plot_syn, gboolean plot_obs, double x_offset, double y_offset, double zoom)
{
	double xmin1, xmax1, ymin1, ymax1;              /* Synthetic data limits    */
	double xmin2, xmax2, ymin2, ymax2;              /* Experimental data limits */

	if (plot_syn)	gui_plot_get_curve_limits (syn, &xmin1, &ymin1, &xmax1, &ymax1);
	if (plot_obs)	gui_plot_get_curve_limits (obs, &xmin2, &ymin2, &xmax2, &ymax2);

	if (plot_syn)
	{
		*xmin = xmin1 + x_offset * (xmax1 - xmin1) - (0.1 + zoom) * (xmax1 - xmin1);
		*xmax = xmax1 + x_offset * (xmax1 - xmin1) + (0.1 + zoom) * (xmax1 - xmin1);
		*ymin = ymin1 + y_offset * (ymax1 - ymin1) - (0.1 + zoom) * (ymax1 - ymin1);
		*ymax = ymax1 + y_offset * (ymax1 - ymin1) + (0.1 + zoom) * (ymax1 - ymin1);
		xmin1 = *xmin; xmax1 = *xmax; ymin1 = *ymin; ymax1 = *ymax;
	}

	if (plot_obs)
	{
		*xmin = xmin2 + x_offset * (xmax2 - xmin2) - (0.1 + zoom) * (xmax2 - xmin2);
		*xmax = xmax2 + x_offset * (xmax2 - xmin2) + (0.1 + zoom) * (xmax2 - xmin2);
		*ymin = ymin2 + y_offset * (ymax2 - ymin2) - (0.1 + zoom) * (ymax2 - ymin2);
		*ymax = ymax2 + y_offset * (ymax2 - ymin2) + (0.1 + zoom) * (ymax2 - ymin2);
		xmin2 = *xmin; xmax2 = *xmax; ymin2 = *ymin; ymax2 = *ymax;
	}

	if (plot_syn && plot_obs)
	{
		if (xmin1 < xmin2) *xmin = xmin1; else *xmin = xmin2;
		if (xmax1 > xmax2) *xmax = xmax1; else *xmax = xmax2;
		if (ymin1 < ymin2) *ymin = ymin1; else *ymin = ymin2;
		if (ymax1 > ymax2) *ymax = ymax1; else *ymax = ymax2;
	}

	return SUCCESS;
}

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

	GtkWidget *plot_image				= gui_widget_lookup ("phoebe_lc_plot_image")->gtk;
	GtkWidget *syn_checkbutton 			= gui_widget_lookup ("phoebe_lc_plot_options_syn_checkbutton")->gtk;
	GtkWidget *obs_checkbutton 			= gui_widget_lookup ("phoebe_lc_plot_options_obs_checkbutton")->gtk;
	GtkWidget *vertices_no_spinbutton 	= gui_widget_lookup ("phoebe_lc_plot_options_vertices_no_spinbutton")->gtk;
	GtkWidget *obs_combobox 			= gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox")->gtk;
	GtkWidget *x_combobox 				= gui_widget_lookup ("phoebe_lc_plot_options_x_combobox")->gtk;
	GtkWidget *y_combobox				= gui_widget_lookup ("phoebe_lc_plot_options_y_combobox")->gtk;

	GtkWidget *coarse_grid				= gui_widget_lookup ("phoebe_lc_plot_controls_coarse_checkbutton")->gtk;
	GtkWidget *fine_grid				= gui_widget_lookup ("phoebe_lc_plot_controls_fine_checkbutton")->gtk;

	gint VERITCES 	= gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(vertices_no_spinbutton));
	gint INDEX		= -1;
	gint INDEP; 
	gint DEP;

	gdouble XMIN = 0.0, XMAX = 0.0, YMIN = 0.0, YMAX = 0.0;

	gboolean plot_obs = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(obs_checkbutton));
	gboolean plot_syn = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(syn_checkbutton));

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	//----------------

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 0)	INDEP 	= PHOEBE_COLUMN_HJD;
	if (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 1)	INDEP 	= PHOEBE_COLUMN_PHASE;

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 0)	DEP 	= PHOEBE_COLUMN_MAGNITUDE;
	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 1)	DEP 	= PHOEBE_COLUMN_FLUX;

	INDEX = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox));

	if (INDEX > -1){

	obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, INDEX);
	phoebe_curve_transform (obs, INDEP, DEP, PHOEBE_COLUMN_UNDEFINED);
	alias_phase_points (obs->indep, obs->dep, NULL, -0.6, 0.6);

	syn = phoebe_curve_new ();
	syn->type = PHOEBE_CURVE_LC;

	indep = phoebe_vector_new ();
	phoebe_vector_alloc (indep, VERITCES);
	for (i = 0; i < VERITCES; i++) indep->val[i] = -0.6 + 1.2 * (double) i/(VERITCES-1);

	status = phoebe_curve_compute (syn, indep, INDEX, INDEP, DEP);
	phoebe_vector_free (indep);

	/* write both curves to files */
	sprintf(oname, "%s/phoebe-lc-XXXXXX", tmpdir);
	ofd = mkstemp (oname);
	for (i=0;i<obs->indep->dim;i++) {
		sprintf(line, "%lf\t%lf\t%lf\n", obs->indep->val[i], obs->dep->val[i], obs->weight->val[i]) ;
		write(ofd, line, strlen(line));
	}
	close(ofd);

	sprintf(sname, "%s/phoebe-lc-XXXXXX", tmpdir);
	sfd = mkstemp (sname);
	for (i=0;i<syn->indep->dim;i++) {
		sprintf(line, "%lf\t%lf\n", syn->indep->val[i], syn->dep->val[i]) ;
		write(sfd, line, strlen(line));
	}
	close(sfd);

	//----------------

	/* open command file */
	sprintf(cname, "%s/phoebe-lc-XXXXXX", tmpdir);
	cfd = mkstemp (cname);

	sprintf(line, "set terminal png small size 615,341\n"); 			write(cfd, line, strlen(line));
	sprintf(line, "set mxtics 2\n"); 									write(cfd, line, strlen(line));
	sprintf(line, "set mytics 2\n"); 									write(cfd, line, strlen(line));
	sprintf(line, "set lmargin 5\n");									write(cfd, line, strlen(line));
	sprintf(line, "set tmargin 2\n");									write(cfd, line, strlen(line));
	sprintf(line, "set rmargin 2\n");									write(cfd, line, strlen(line));
	sprintf(line, "set bmargin 4\n");									write(cfd, line, strlen(line));

	sprintf(line, "set xlabel '%s'\n", gtk_combo_box_get_active_text (GTK_COMBO_BOX (x_combobox)));
		write(cfd, line, strlen(line));
	sprintf(line, "set ylabel '%s'\n", gtk_combo_box_get_active_text (GTK_COMBO_BOX (y_combobox)));
		write(cfd, line, strlen(line));

	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (coarse_grid)))
		sprintf(line, "set grid xtics ytics\n");						write(cfd, line, strlen(line));
	if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (fine_grid)))
		sprintf(line, "set grid mxtics mytics\n");						write(cfd, line, strlen(line));


	gui_plot_get_plot_limits (syn, obs, &XMIN, &YMIN, &XMAX, &YMAX, plot_syn, plot_obs, 0.0, 0.0, 0.0);


	sprintf(line, "set xrange [%lf:%lf]\n", XMIN, XMAX); 				write(cfd, line, strlen(line));
	if (DEP == PHOEBE_COLUMN_MAGNITUDE)
		{sprintf(line, "set yrange [%lf:%lf]\n", YMAX, YMIN); 			write(cfd, line, strlen(line));}
	if (DEP == PHOEBE_COLUMN_FLUX)
		{sprintf(line, "set yrange [%lf:%lf]\n", YMIN, YMAX); 			write(cfd, line, strlen(line));}

	if (INDEP == PHOEBE_COLUMN_HJD)
		{sprintf(line, "set format x '%%7.0f'\n");			 			write(cfd, line, strlen(line));}

	sprintf(pname, "%s/phoebe-lc-plot-XXXXXX", tmpdir);
	pfd = mkstemp (pname);

	sprintf(line, "set output '%s'\n", pname);							write(cfd, line, strlen(line));

	if (plot_syn && plot_obs)
		{sprintf(line, "plot  '%s' w l notitle,'%s' w p lt 3 lw 1 pt 6 notitle\n", sname, oname);	write(cfd, line, strlen(line));}
	else if (plot_syn)
		{sprintf(line, "plot  '%s' w l notitle\n", sname);											write(cfd, line, strlen(line));}
	else if (plot_obs)
		{sprintf(line, "plot  '%s' w p lt 3 lw 1 pt 6 notitle\n", oname);							write(cfd, line, strlen(line));}

	close(cfd);

	sprintf(line,"gnuplot \"%s\"", cname);

	system(line);

	if (plot_syn || plot_obs)
		gtk_image_set_from_pixbuf(GTK_IMAGE(plot_image), gdk_pixbuf_new_from_file(pname, NULL));

	close(pfd);

	//----------------

	remove(oname);
	remove(sname);
	remove(cname);
	remove(pname);

	phoebe_curve_free(syn);
	phoebe_curve_free(obs);
	}
	
	return SUCCESS;
}

int gui_plot_rv_using_gnuplot ()
{
	PHOEBE_curve *obs;
	PHOEBE_curve *syn; 

	PHOEBE_vector *indep;

	gchar *tmpdir;
	gchar oname[255];	
	gchar sname[255];
	gchar cname[255];
	gchar pname[255]; 	
	gchar  line[255]; 

	gint ofd, sfd; 
	gint cfd, pfd;	

	gint i;
	gint status;

	GtkWidget *plot_image				= gui_widget_lookup ("phoebe_rv_plot_image")->gtk;
	GtkWidget *syn_checkbutton 			= gui_widget_lookup ("phoebe_rv_plot_options_syn_checkbutton")->gtk;
	GtkWidget *obs_checkbutton 			= gui_widget_lookup ("phoebe_rv_plot_options_obs_checkbutton")->gtk;
	GtkWidget *vertices_no_spinbutton 	= gui_widget_lookup ("phoebe_rv_plot_options_vertices_no_spinbutton")->gtk;
	GtkWidget *obs_combobox 			= gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox")->gtk;
	GtkWidget *x_combobox 				= gui_widget_lookup ("phoebe_rv_plot_options_x_combobox")->gtk;
	GtkWidget *y_combobox				= gui_widget_lookup ("phoebe_rv_plot_options_y_combobox")->gtk;

	GtkWidget *coarse_grid				= gui_widget_lookup ("phoebe_rv_plot_controls_coarse_checkbutton")->gtk;
	GtkWidget *fine_grid				= gui_widget_lookup ("phoebe_rv_plot_controls_fine_checkbutton")->gtk;

	gint VERITCES 	= gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(vertices_no_spinbutton));
	gint INDEX		= -1;
	gint INDEP;
	gint DEP;

	gdouble XMIN = 0.0, XMAX = 0.0, YMIN = 0.0, YMAX = 0.0;

	gboolean plot_obs = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(obs_checkbutton));
	gboolean plot_syn = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(syn_checkbutton));
	gboolean both = FALSE;

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 0)	INDEP 	= PHOEBE_COLUMN_HJD;
	if (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 1)	INDEP 	= PHOEBE_COLUMN_PHASE;

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 0)	DEP 	= PHOEBE_COLUMN_PRIMARY_RV;
	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 1)	DEP 	= PHOEBE_COLUMN_SECONDARY_RV;

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 2)	DEP 	= PHOEBE_COLUMN_PRIMARY_RV;
/*
	PHOEBE_COLUMN_UNDEFINED,
	PHOEBE_COLUMN_HJD,
	PHOEBE_COLUMN_PHASE,
	PHOEBE_COLUMN_MAGNITUDE,
	PHOEBE_COLUMN_FLUX,
	PHOEBE_COLUMN_PRIMARY_RV,
	PHOEBE_COLUMN_SECONDARY_RV,
	PHOEBE_COLUMN_SIGMA,
	PHOEBE_COLUMN_WEIGHT,
	PHOEBE_COLUMN_INVALID
*/	
	INDEX = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox));

	if (INDEX > -1){

		obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, INDEX);
		phoebe_curve_transform (obs, INDEP, DEP, PHOEBE_COLUMN_UNDEFINED);

		syn = phoebe_curve_new ();
		syn->type = PHOEBE_CURVE_RV;

		indep = phoebe_vector_new ();
		phoebe_vector_alloc (indep, VERITCES);
		for (i = 0; i < VERITCES; i++) indep->val[i] = -0.6 + 1.2 * (double) i/(VERITCES-1);

		status = phoebe_curve_compute (syn, indep, INDEX, INDEP, DEP);
		phoebe_vector_free (indep);

		/* write both curves to files */
		sprintf(oname, "%s/phoebe-rv-XXXXXX", tmpdir);
		ofd = mkstemp (oname);
		for (i=0;i<obs->indep->dim;i++) {
			sprintf(line, "%lf\t%lf\t%lf\n", obs->indep->val[i], obs->dep->val[i], obs->weight->val[i]) ;
			write(ofd, line, strlen(line));
		}
		close(ofd);

		sprintf(sname, "%s/phoebe-rv-XXXXXX", tmpdir);
		sfd = mkstemp (sname);
		for (i=0;i<syn->indep->dim;i++) {
			sprintf(line, "%lf\t%lf\n", syn->indep->val[i], syn->dep->val[i]) ;
			write(sfd, line, strlen(line));
		}
		close(sfd);

		sprintf(cname, "%s/phoebe-lc-XXXXXX", tmpdir);
		cfd = mkstemp (cname);

		sprintf(line, "set terminal png small size 615,341\n"); 			write(cfd, line, strlen(line));
		sprintf(line, "set mxtics 2\n"); 									write(cfd, line, strlen(line));
		sprintf(line, "set mytics 2\n"); 									write(cfd, line, strlen(line));
		sprintf(line, "set lmargin 5\n");									write(cfd, line, strlen(line));
		sprintf(line, "set tmargin 2\n");									write(cfd, line, strlen(line));
		sprintf(line, "set rmargin 2\n");									write(cfd, line, strlen(line));
		sprintf(line, "set bmargin 4\n");									write(cfd, line, strlen(line));

		sprintf(line, "set xlabel '%s'\n", gtk_combo_box_get_active_text (GTK_COMBO_BOX (x_combobox)));
			write(cfd, line, strlen(line));
		sprintf(line, "set ylabel '%s'\n", gtk_combo_box_get_active_text (GTK_COMBO_BOX (y_combobox)));
			write(cfd, line, strlen(line));

		if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (coarse_grid)))
			sprintf(line, "set grid xtics ytics\n");						write(cfd, line, strlen(line));
		if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (fine_grid)))
			sprintf(line, "set grid mxtics mytics\n");						write(cfd, line, strlen(line));

		gui_plot_get_plot_limits (syn, obs, &XMIN, &YMIN, &XMAX, &YMAX, plot_syn, plot_obs, 0.0, 0.0, 0.0);


		sprintf(line, "set xrange [%lf:%lf]\n", XMIN, XMAX); 				write(cfd, line, strlen(line));
		if (DEP == PHOEBE_COLUMN_MAGNITUDE)
			{sprintf(line, "set yrange [%lf:%lf]\n", YMAX, YMIN); 			write(cfd, line, strlen(line));}
		if (DEP == PHOEBE_COLUMN_FLUX)
			{sprintf(line, "set yrange [%lf:%lf]\n", YMIN, YMAX); 			write(cfd, line, strlen(line));}

		if (INDEP == PHOEBE_COLUMN_HJD)
			{sprintf(line, "set format x '%%7.0f'\n");			 			write(cfd, line, strlen(line));}

		sprintf(pname, "%s/phoebe-lc-plot-XXXXXX", tmpdir);
		pfd = mkstemp (pname);
	
		sprintf(line, "set output '%s'\n", pname);							write(cfd, line, strlen(line));
	
		if (plot_syn && plot_obs)
			{sprintf(line, "plot  '%s' w l notitle,'%s' w p lt 3 lw 1 pt 6 notitle\n", sname, oname);	write(cfd, line, strlen(line));}
		else if (plot_syn)
			{sprintf(line, "plot  '%s' w l notitle\n", sname);											write(cfd, line, strlen(line));}
		else if (plot_obs)
			{sprintf(line, "plot  '%s' w p lt 3 lw 1 pt 6 notitle\n", oname);							write(cfd, line, strlen(line));}

		close(cfd);

		sprintf(line,"gnuplot \"%s\"", cname);

		system(line);

		if (plot_syn || plot_obs)
			gtk_image_set_from_pixbuf(GTK_IMAGE(plot_image), gdk_pixbuf_new_from_file(pname, NULL));
	
		close(pfd);

		//----------------

		remove(oname);
		remove(sname);
		remove(cname);
		remove(pname);


		phoebe_curve_free(syn);
		phoebe_curve_free(obs);
	}
	
	return SUCCESS;
}

