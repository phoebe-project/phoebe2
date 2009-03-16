#include <stdlib.h>

#include <phoebe/phoebe.h>
#include <gtk/gtk.h>
#include <cairo.h>
#include <math.h>

#include "phoebe_gui_build_config.h"

#include "phoebe_gui_accessories.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_error_handling.h"
#include "phoebe_gui_plotting.h"
#include "phoebe_gui_treeviews.h"
#include "phoebe_gui_types.h"

#ifdef __MINGW32__
#include <glib/gfileutils.h>
#include <windows.h>
#include <winuser.h>
#else
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#define gui_plot_width(x) (x->width - 2 * x->layout->xmargin - x->leftmargin - x->layout->rmargin)
#define gui_plot_height(x) (x->height - 2 * x->layout->ymargin - x->layout->tmargin - x->layout->bmargin)

static const double dash_coarse_grid[] = { 4.0, 1.0 };
static const double dash_fine_grid[] = { 1.0 };

GUI_plot_layout *gui_plot_layout_new ()
{
	GUI_plot_layout *layout = phoebe_malloc (sizeof (*layout));

	layout->lmargin = 10;
	layout->rmargin = 10;
	layout->tmargin = 10;
	layout->bmargin = 30;

	layout->xmargin = 10;
	layout->ymargin = 10;

	layout->label_lmargin = 2;
	layout->label_rmargin = 3;

	layout->x_tick_length = layout->xmargin - 2;
	layout->y_tick_length = layout->ymargin - 2;

	return layout;
}

int gui_plot_layout_free (GUI_plot_layout *layout)
{
	free (layout);

	return SUCCESS;
}

void gui_plot_offset_zoom_limits (double min_value, double max_value, double offset, double zoom, double *newmin, double *newmax)
{
	*newmin = min_value + offset * (max_value - min_value) - zoom * (max_value - min_value);
	*newmax = max_value + offset * (max_value - min_value) + zoom * (max_value - min_value);
	//printf ("min = %lf, max = %lf, offset = %lf, zoom = %lf, newmin = %lf, newmax = %lf\n", min_value, max_value, offset, zoom, *newmin, *newmax);
}

bool gui_plot_xvalue (GUI_plot_data *data, double value, double *x)
{
	double xmin, xmax;
	gui_plot_offset_zoom_limits (data->x_ll, data->x_ul, data->x_offset, data->zoom, &xmin, &xmax);
	*x = data->leftmargin + data->layout->xmargin + (value - xmin) * gui_plot_width(data)/(xmax - xmin);
	if (*x < data->leftmargin) return FALSE;
	if (*x > data->width - data->layout->rmargin) return FALSE;
	return TRUE;
}

bool gui_plot_yvalue (GUI_plot_data *data, double value, double *y)
{
	double ymin, ymax;
	gui_plot_offset_zoom_limits (data->y_min, data->y_max, data->y_offset, data->zoom, &ymin, &ymax);
	*y = data->height - (data->layout->bmargin + data->layout->ymargin) - (value - ymin) * gui_plot_height(data)/(ymax - ymin);
	if (*y < data->layout->tmargin) return FALSE;
	if (*y > data->height - data->layout->bmargin) return FALSE;
	return TRUE;
}

void gui_plot_coordinates_from_pixels (GUI_plot_data *data, double xpix, double ypix, double *xval, double *yval)
{
	double xmin, xmax, ymin, ymax;

	gui_plot_offset_zoom_limits (data->x_ll, data->x_ul, data->x_offset, data->zoom, &xmin, &xmax);
	gui_plot_offset_zoom_limits (data->y_min, data->y_max, data->y_offset, data->zoom, &ymin, &ymax);

	*xval = xmin + (xmax - xmin) * (xpix - (data->leftmargin + data->layout->xmargin))/gui_plot_width(data);
	*yval = ymax - (ymax - ymin) * (ypix - (data->layout->tmargin+data->layout->ymargin))/gui_plot_height(data);
}

bool gui_plot_tick_values (double low_value, double high_value, double *first_tick, double *tick_spacing, int *ticks, int *minorticks, char format[])
{
	int logspacing, factor;
	double spacing;
	
	if (low_value >= high_value)
		return FALSE;

	logspacing = floor(log10(high_value - low_value)) - 1;
	factor = ceil((high_value - low_value)/pow(10, logspacing + 1));
	//printf ("low = %lf, hi = %lf, factor = %d, logspacing = %d\n", low_value, high_value, factor, logspacing);
	if (factor > 5) {
		logspacing++;
		factor = 1;
		*minorticks = 2;
	}
	else {
		if ((factor == 3) || (factor == 4)) {
			factor = 5;
		}
		*minorticks = factor;
	}
	spacing = factor * pow(10, logspacing);

	*tick_spacing = spacing;
	*first_tick = floor(low_value/spacing) * spacing;
	*ticks = ceil((high_value - low_value)/spacing) + 2;
	sprintf(format, "%%.%df", (logspacing > 0) ? 0 : -logspacing);
	//printf("ticks = %d, format = %s, first_tick = %lf, spacing = %lf, factor = %d, logspacing = %d\n", *ticks, format, *first_tick, spacing, factor, logspacing);

	return TRUE;
}

void gui_plot_clear_canvas (GUI_plot_data *data)
{
	GtkWidget *widget = data->container;

	if (data->canvas)
		cairo_destroy (data->canvas);

	data->canvas = gdk_cairo_create (widget->window);
	data->width  = widget->allocation.width;
	data->height = widget->allocation.height;

	cairo_set_source_rgb (data->canvas, 0, 0, 0);
	cairo_set_line_width (data->canvas, 1);

	// Calculate left margin
	PHOEBE_column_type dtype;
	int ticks, minorticks;
	double first_tick, tick_spacing;
	double x, ymin, ymax;
	char format[20], label[20];
	cairo_text_extents_t te;
	
	// Determine the lowest and highest y value that will be plotted
	data->leftmargin = data->layout->lmargin;
	gui_plot_coordinates_from_pixels (data, 0, data->layout->tmargin, &x, &ymax);
	gui_plot_coordinates_from_pixels (data, 0, data->height - data->layout->bmargin, &x, &ymin);

	phoebe_column_get_type (&dtype, data->y_request);
	gui_plot_tick_values ( ((dtype == PHOEBE_COLUMN_MAGNITUDE) ? ymax : ymin), ((dtype == PHOEBE_COLUMN_MAGNITUDE) ? ymin : ymax), &first_tick, &tick_spacing, &ticks, &minorticks, format);

	// Calculate how large the y labels will be
	cairo_select_font_face (data->canvas, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
	cairo_set_font_size (data->canvas, 12);
	sprintf(label, format, ymin);
	cairo_text_extents (data->canvas, label, &te);
	if (te.width + te.x_bearing + data->layout->label_lmargin + data->layout->label_rmargin > data->leftmargin) data->leftmargin = te.width + te.x_bearing + data->layout->label_lmargin + data->layout->label_rmargin;
	sprintf(label, format, ymax);
	cairo_text_extents (data->canvas, label, &te);
	if (te.width + te.x_bearing + data->layout->label_lmargin + data->layout->label_rmargin > data->leftmargin) data->leftmargin = te.width + te.x_bearing + data->layout->label_lmargin + data->layout->label_rmargin;

	cairo_rectangle (data->canvas, data->leftmargin, data->layout->tmargin, data->width - data->leftmargin - data->layout->rmargin, data->height - data->layout->tmargin - data->layout->bmargin);
	cairo_stroke (data->canvas);
}

gboolean on_plot_area_expose_event (GtkWidget *widget, GdkEventExpose *event, gpointer user_data)
{
	/*
	 * on_plot_area_expose_event:
	 * @widget: plot container
	 * @event: expose event
	 * @user_data: #GUI_plot_data structure
	 *
	 * This callback is invoked every time the plot needs to be redrawn. For
	 * example, it is called on resizing, detaching, obscuring, etc. It is
	 * thus the only function that is allowed to call the actual drawing
	 * function. It is also responsible for plotting the graph box.
	 *
	 * Returns: #FALSE to keep the propagation of the signal.
	 */

	GUI_plot_data *data = (GUI_plot_data *) user_data;

	gui_plot_clear_canvas (data);
	gui_plot_area_draw (data, NULL);

	return FALSE;
}

gboolean on_plot_area_enter (GtkWidget *widget, GdkEventCrossing *event, gpointer user_data)
{
	gtk_widget_grab_focus (widget);
	return TRUE;
}

int gui_plot_get_closest (GUI_plot_data *data, double x, double y, int *cp, int *ci)
{
	/*
	 * gui_plot_get_closest:
	 * @x:  pointer x coordinate
	 * @y:  pointer y coordinate
	 * @cp: closest passband index placeholder
	 * @ci: closest data index placeholder
	 *
	 * Returns the passband index @cp and the data index @ci of the closest
	 * point to the passed (@x,@y) coordinates. This function is *HIGHLY*
	 * unoptimized and can (and should) perform several times better.
	 */

	int i, p;
	double cx, cy, cf, cf0;

	*cp = 0;
	*ci = 0;

	for (p = 0; p < data->cno; p++)
		if (data->request[p].query)
			break;
	if (p == data->cno)
		return GUI_ERROR_NO_CURVE_MARKED_FOR_PLOTTING;

	*cp = p;

	/* Starting with the first data point in the first passband: */
	cx = data->request[p].query->indep->val[0];
	cy = data->request[p].query->dep->val[0] + data->request[p].offset;
	cf0 = (x-cx)*(x-cx)+(y-cy)*(y-cy);

	for ( ; p < data->cno; p++) {
		if (!data->request[p].query) continue;
		for (i = 0; i < data->request[p].query->indep->dim; i++) {
			cx = data->request[p].query->indep->val[i];
			cy = data->request[p].query->dep->val[i] + data->request[p].offset;

			cf = (x-cx)*(x-cx)+(y-cy)*(y-cy);
			if (cf < cf0) {
				*cp = p;
				*ci = i;
				cf0 = cf;
			}
		}
	}

	return SUCCESS;
}

gboolean on_plot_area_motion (GtkWidget *widget, GdkEventMotion *event, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;

	double x, y;
	char x_str[20], y_str[20];
	char cp_str[255], *cp_ptr, cx_str[20], cy_str[20];

	int cp, ci, p;

	gui_plot_coordinates_from_pixels (data, event->x, event->y, &x, &y);
	if (gui_plot_get_closest (data, x, y, &cp, &ci) != SUCCESS)
		return FALSE;

	sprintf (x_str, "%lf", x);
	sprintf (y_str, "%lf", y);

	gtk_label_set_text (GTK_LABEL (data->x_widget), x_str);
	gtk_label_set_text (GTK_LABEL (data->y_widget), y_str);

	for (p = 0; p < data->cno; p++)
		if (data->request[p].query)
			break;

	if (p == data->cno)
		return FALSE;

#warning CURVE_RECOGNITION_FAILS_FOR_RV_CURVES
	if (data->ctype == PHOEBE_CURVE_LC)
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_id"), cp, &cp_ptr);
	else if (data->ctype == PHOEBE_CURVE_RV)
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_id"), cp, &cp_ptr);
	else {
		printf ("*** UNHANDLED EXCEPTION ENCOUNTERED IN on_plot_area_motion()!\n");
		return FALSE;
	}

	sprintf (cp_str, "in %s:", cp_ptr);

	sprintf (cx_str, "%lf", data->request[cp].query->indep->val[ci]);
	sprintf (cy_str, "%lf", data->request[cp].query->dep->val[ci]);

	gtk_label_set_text (GTK_LABEL (data->cp_widget), cp_str);
	gtk_label_set_text (GTK_LABEL (data->cx_widget), cx_str);
	gtk_label_set_text (GTK_LABEL (data->cy_widget), cy_str);

	return FALSE;
}

void on_plot_button_clicked (GtkButton *button, gpointer user_data)
{
	/*
	 * on_plot_button_clicked:
	 * @button: Plot button widget 
	 * @user_data: #GUI_plot_data structure
	 *
	 * This is the main allocation plotting function. It queries all widgets
	 * for their content and stores them to the #GUI_plot_data structure for
	 * follow-up plotting. The plotting itself is *not* done by this function;
	 * rather, the call to a graph refresh is issued that, in turn, invokes
	 * the expose event handler that governs the plotting. The purpose of
	 * this function is to get everything ready for the expose event handler.
	 */

	GUI_plot_data *data = (GUI_plot_data *) user_data;
	int i, j, status;
	PHOEBE_column_type itype, dtype;
	double x_min, x_max, y_min, y_max;

	bool plot_obs, plot_syn, first_time = YES;

	phoebe_gui_debug ("* entering on_plot_button_clicked().\n");

	/* Read in parameter values from their respective widgets: */
	gui_get_values_from_widgets ();

	/* See what is requested: */
	phoebe_column_get_type (&itype, data->x_request);
	phoebe_column_get_type (&dtype, data->y_request);

	for (i = 0; i < data->cno; i++) {
		plot_obs = data->request[i].plot_obs;
		plot_syn = data->request[i].plot_syn;

		/* Free any pre-existing data: */
		if (data->request[i].raw)   phoebe_curve_free (data->request[i].raw);   data->request[i].raw   = NULL;
		if (data->request[i].query) phoebe_curve_free (data->request[i].query); data->request[i].query = NULL;
		if (data->request[i].model) phoebe_curve_free (data->request[i].model); data->request[i].model = NULL;

		/* Prepare observed data (if toggled): */
		if (plot_obs) {
			data->request[i].raw = phoebe_curve_new_from_pars (data->ctype, i);
			if (!data->request[i].raw) {
				char notice[255];
				plot_obs = NO;
				sprintf (notice, "Observations for curve %d failed to open and cannot be plotted. Please review the information given in the Data tab.", i+1);
				gui_notice ("Observed data not found", notice);
			}

			/* Transform the data to requested types: */
			if (plot_obs) {
				data->request[i].query = phoebe_curve_duplicate (data->request[i].raw);
				if (data->ctype == PHOEBE_CURVE_LC)
					status = phoebe_curve_transform (data->request[i].query, itype, dtype, PHOEBE_COLUMN_UNDEFINED);
				else if (data->ctype == PHOEBE_CURVE_RV)
					status = phoebe_curve_transform (data->request[i].query, itype, data->request[i].raw->dtype, PHOEBE_COLUMN_UNDEFINED);
				else {
					printf ("*** EXCEPTION ENCOUNTERED IN ON_PLOT_BUTTON_CLICKED\n");
					return;
				}
				if (status != SUCCESS) {
					char notice[255];
					plot_obs = NO;
					sprintf (notice, "Observations for curve %d cannot be transformed to the requested plotting axes. Plotting will be suppressed.", i+1);
					gui_notice ("Observed data transformation failure", notice);
				}
			}

			/* Alias data if requested: */
			if (data->alias && plot_obs) {
				status = phoebe_curve_alias (data->request[i].query, data->x_ll, data->x_ul);
				if (status != SUCCESS) {
					char notice[255];
					plot_obs = NO;
					sprintf (notice, "Observations for curve %d cannot be aliased. Plotting will be suppressed.", i+1);
					gui_notice ("Observed data aliasing failure", notice);
				}
			}

			/* Calculate residuals when requested */
			if (data->residuals && plot_obs) {
				PHOEBE_vector *indep = phoebe_vector_duplicate (data->request[i].query->indep);
				data->request[i].model = phoebe_curve_new ();
				data->request[i].model->type = data->ctype;
				data->request[i].model->itype = itype;

				if (data->ctype == PHOEBE_CURVE_LC)
					status = phoebe_curve_compute (data->request[i].model, indep, i, itype, dtype);
				else if (data->ctype == PHOEBE_CURVE_RV) {
					char *param; PHOEBE_column_type rvtype;
					phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), i, &param);
					phoebe_column_get_type (&rvtype, param);
					status = phoebe_curve_compute (data->request[i].model, indep, i, itype, rvtype);
				}
				else {
					printf ("*** EXCEPTION ENCOUNTERED IN ON_PLOT_BUTTON_CLICKED\n");
					return;
				}

				if (status != SUCCESS) {
					char notice[255];
					plot_obs = NO;
					sprintf (notice, "Model computation for curve %d failed with the following message: %s", i+1, phoebe_gui_error (status));
					gui_notice ("Model curve computation failed", notice);
				}

				for (j = 0; j < data->request[i].model->indep->dim; j++) {
					data->request[i].query->dep->val[j] -= data->request[i].model->dep->val[j];
					data->request[i].model->dep->val[j] = 0.0;
				}
				if (!plot_syn) {
					phoebe_curve_free (data->request[i].model); 
					data->request[i].model = NULL;
				}
			}

			/* Determine plot limits: */
			if (plot_obs) {
				phoebe_vector_min_max (data->request[i].query->indep, &x_min, &x_max);
				phoebe_vector_min_max (data->request[i].query->dep,   &y_min, &y_max);

				if (first_time) {
					data->x_min = x_min;
					data->x_max = x_max;
					data->y_min = y_min + data->request[i].offset;
					data->y_max = y_max + data->request[i].offset;
					first_time = NO;
				}
				else {
					data->x_min = min (data->x_min, x_min);
					data->x_max = max (data->x_max, x_max);
					data->y_min = min (data->y_min, y_min + data->request[i].offset);
					data->y_max = max (data->y_max, y_max + data->request[i].offset);
				}
			}
		}

		/* Prepare synthetic (model) data (if toggled): */
		if (plot_syn && !(data->residuals)) {
			PHOEBE_vector *indep = phoebe_vector_new_from_range (data->vertices, data->x_ll, data->x_ul);

			data->request[i].model = phoebe_curve_new ();
			data->request[i].model->type = data->ctype;

			if (data->ctype == PHOEBE_CURVE_LC)
				status = phoebe_curve_compute (data->request[i].model, indep, i, itype, dtype);
			else if (data->ctype == PHOEBE_CURVE_RV) {
				char *param; PHOEBE_column_type rvtype;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), i, &param);
				phoebe_column_get_type (&rvtype, param);
				status = phoebe_curve_compute (data->request[i].model, indep, i, itype, rvtype);
			}
			else {
				printf ("*** EXCEPTION ENCOUNTERED IN ON_PLOT_BUTTON_CLICKED\n");
				return;
			}
			if (status != SUCCESS) {
				char notice[255];
				plot_syn = NO;
				phoebe_curve_free (data->request[i].model);
				data->request[i].model = NULL;
				sprintf (notice, "Model computation for curve %d failed with the following message: %s", i+1, phoebe_gui_error (status));
				gui_notice ("Model curve computation failed", notice);
			}

			gui_fill_sidesheet_res_treeview ();

			/* Plot aliasing for synthetic curves is not implemented by the
			 * library -- phoebe_curve_compute () computes the curve on the
			 * passed range. Perhaps it would be better to have that function
			 * compute *up to* 1.0 in phase and alias the rest. But this is
			 * not so urgent.
			 */

			/* Determine plot limits */
			if (plot_syn) {
				phoebe_vector_min_max (data->request[i].model->indep, &x_min, &x_max);
				phoebe_vector_min_max (data->request[i].model->dep,   &y_min, &y_max);

				if (first_time) {
					data->x_min = x_min;
					data->x_max = x_max;
					data->y_min = y_min + data->request[i].offset;
					data->y_max = y_max + data->request[i].offset;
					first_time = NO;
				}
				else {
					data->x_min = min (data->x_min, x_min);
					data->x_max = max (data->x_max, x_max);
					data->y_min = min (data->y_min, y_min + data->request[i].offset);
					data->y_max = max (data->y_max, y_max + data->request[i].offset);
				}
			}

			phoebe_vector_free (indep);
		}
	}

	/* If we are plotting magnitudes, reverse the y-axis: */
	if (dtype == PHOEBE_COLUMN_MAGNITUDE) {
		double store = data->y_min;
		data->y_min = data->y_max;
		data->y_max = store;
	}

	gui_plot_area_refresh (data);

	phoebe_gui_debug ("* leaving on_plot_button_clicked.\n");

	return;
}

int gui_plot_area_refresh (GUI_plot_data *data)
{
	/*
	 * gui_plot_area_refresh:
	 * @data: #GUI_plot_data structure
	 *
	 * This is a small hack to invoke the expose event handler to refresh the
	 * plot. It is not perfectly elegant, but it is simple and is serves the
	 * purpose.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	gtk_widget_hide (data->container);
	gtk_widget_show (data->container);

	return SUCCESS;
}

void gui_plot_xticks (GUI_plot_data *data)
{
	int i, j, ticks, minorticks;
	double x, first_tick, tick_spacing, value;
	double xmin, xmax, y;
	char format[20];
	char label[20];
	cairo_text_extents_t te;
	
	// Determine the lowest and highest x value that will be plotted
	gui_plot_coordinates_from_pixels (data, data->leftmargin, 0, &xmin, &y);
	gui_plot_coordinates_from_pixels (data, data->width - data->layout->rmargin, 0, &xmax, &y);

	if (!gui_plot_tick_values (xmin, xmax, &first_tick, &tick_spacing, &ticks, &minorticks, format))
		return;

	cairo_set_source_rgb (data->canvas, 0.0, 0.0, 0.0);
	cairo_select_font_face (data->canvas, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
	cairo_set_font_size (data->canvas, 12);

	for (i = 0; i < ticks; i++) {
		value = first_tick + i * tick_spacing;
		if (!gui_plot_xvalue (data, value, &x)) continue;

		// Top tick
		cairo_move_to (data->canvas, x, data->layout->tmargin);
		cairo_rel_line_to (data->canvas, 0, data->layout->x_tick_length);
		// Bottom tick
		cairo_move_to (data->canvas, x, data->height - data->layout->bmargin);
		cairo_rel_line_to (data->canvas, 0, - data->layout->x_tick_length);
		cairo_stroke (data->canvas);

		if (data->coarse_grid) {
			cairo_set_line_width (data->canvas, 0.5);
			cairo_set_dash(data->canvas, dash_coarse_grid, sizeof(dash_coarse_grid) / sizeof(dash_coarse_grid[0]), 0);
			cairo_move_to (data->canvas, x, data->layout->tmargin);
			cairo_line_to (data->canvas, x, data->height - data->layout->bmargin);
			cairo_stroke (data->canvas);
		}

		// Print the label
		sprintf(label, format, value);
		cairo_text_extents (data->canvas, label, &te);
		cairo_move_to (data->canvas, x - te.width/2 - te.x_bearing, data->height - te.height + te.y_bearing);
		cairo_show_text (data->canvas, label);
		cairo_stroke (data->canvas);
	}

	// Minor ticks
	for (i = 0; i < ticks - 1; i++) {
		for (j = 1; j < minorticks; j++) {
			if (!gui_plot_xvalue (data, first_tick + (minorticks * i + j) * tick_spacing/minorticks, &x)) continue;

			cairo_move_to (data->canvas, x, data->layout->tmargin);
			cairo_rel_line_to (data->canvas, 0, data->layout->x_tick_length/2);
			cairo_move_to (data->canvas, x, data->height - data->layout->bmargin);
			cairo_rel_line_to (data->canvas, 0, -data->layout->x_tick_length/2);
			cairo_stroke (data->canvas);

			if (data->fine_grid) {
				cairo_set_line_width (data->canvas, 0.5);
				cairo_set_dash(data->canvas, dash_fine_grid, sizeof(dash_fine_grid) / sizeof(dash_fine_grid[0]), 0);
				cairo_move_to (data->canvas, x, data->layout->tmargin);
				cairo_line_to (data->canvas, x, data->height - data->layout->bmargin);
				cairo_stroke (data->canvas);
			}
		}
	}
}

void gui_plot_yticks (GUI_plot_data *data)
{
	int i, j, ticks, minorticks;
	double y, first_tick, tick_spacing, value;
	double ymin, ymax, x;
	char format[20];
	char label[20];
	cairo_text_extents_t te;
	PHOEBE_column_type dtype;
	
	// Determine the lowest and highest y value that will be plotted
	gui_plot_coordinates_from_pixels (data, 0, data->layout->tmargin, &x, &ymax);
	gui_plot_coordinates_from_pixels (data, 0, data->height - data->layout->bmargin, &x, &ymin);

	phoebe_column_get_type (&dtype, data->y_request);
	if (!gui_plot_tick_values ( ((dtype == PHOEBE_COLUMN_MAGNITUDE) ? ymax : ymin), ((dtype == PHOEBE_COLUMN_MAGNITUDE) ? ymin : ymax), &first_tick, &tick_spacing, &ticks, &minorticks, format))
		return;

	cairo_set_source_rgb (data->canvas, 0.0, 0.0, 0.0);
	cairo_select_font_face (data->canvas, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
	cairo_set_font_size (data->canvas, 12);

	for (i = 0; i < ticks; i++) {
		value = first_tick + i * tick_spacing;
		if (!gui_plot_yvalue (data, value, &y)) continue;

		// Left tick
		cairo_move_to (data->canvas, data->leftmargin, y);
		cairo_rel_line_to (data->canvas, data->layout->y_tick_length, 0);
		// Right tick
		cairo_move_to (data->canvas, data->width - data->layout->rmargin, y);
		cairo_rel_line_to (data->canvas, - data->layout->y_tick_length, 0);
		cairo_stroke (data->canvas);

		if (data->coarse_grid) {
			cairo_set_line_width (data->canvas, 0.5);
			cairo_set_dash(data->canvas, dash_coarse_grid, sizeof(dash_coarse_grid) / sizeof(dash_coarse_grid[0]), 0);
			cairo_move_to (data->canvas, data->leftmargin, y);
			cairo_line_to (data->canvas, data->width - data->layout->rmargin, y);
			cairo_stroke (data->canvas);
		}

		// Print the label
		sprintf(label, format, value);
		cairo_text_extents (data->canvas, label, &te);
		cairo_move_to (data->canvas, data->leftmargin - te.width - te.x_bearing - data->layout->label_rmargin, y - te.height/2 - te.y_bearing);
		cairo_show_text (data->canvas, label);
	}
	// Minor ticks
	for (i = 0; i < ticks - 1; i++) {
		for (j = 1; j < minorticks; j++) {
			if (!gui_plot_yvalue (data, first_tick + (minorticks * i + j) * tick_spacing/minorticks, &y)) continue;
			cairo_move_to (data->canvas, data->leftmargin, y);
			cairo_rel_line_to (data->canvas, data->layout->y_tick_length/2, 0);
			cairo_move_to (data->canvas, data->width - data->layout->rmargin, y);
			cairo_rel_line_to (data->canvas, -data->layout->y_tick_length/2, 0);
			cairo_stroke (data->canvas);

			if (data->fine_grid) {
				cairo_set_line_width (data->canvas, 0.5);
				cairo_set_dash(data->canvas, dash_fine_grid, sizeof(dash_fine_grid) / sizeof(dash_fine_grid[0]), 0);
				cairo_move_to (data->canvas, data->leftmargin, y);
				cairo_line_to (data->canvas, data->width - data->layout->rmargin, y);
				cairo_stroke (data->canvas);
			}
		}
	}
	cairo_stroke (data->canvas);
}

void gui_plot_interpolate_to_border (GUI_plot_data *data, double xin, double yin, double xout, double yout, double *xborder, double *yborder)
{
	bool x_outside_border = FALSE, y_outside_border = FALSE;
	double xmargin ,ymargin;

	if (xout < data->leftmargin) {
		xmargin = data->leftmargin;
		x_outside_border = TRUE;
	}
	else if (xout > data->width - data->layout->rmargin) {
		xmargin = data->width - data->layout->rmargin;
		x_outside_border = TRUE;
	}

	if (yout < data->layout->tmargin) {
		ymargin = data->layout->tmargin;
		y_outside_border = TRUE;
	}
	else if (yout > data->height - data->layout->bmargin) {
		ymargin = data->height - data->layout->bmargin;
		y_outside_border = TRUE;
	}

	if (x_outside_border && y_outside_border) {
		/* Both xout and yout lie outside the border */
		*yborder = yout + (yin - yout) * (xmargin - xout)/(xin - xout);
		if ((*yborder < data->layout->tmargin) && (*yborder > data->height - data->layout->bmargin)) {
			*yborder = ymargin;
			*xborder = xout + (xin - xout) * (ymargin - yout)/(yin - yout);
		}
		else {
			*xborder = xmargin;
		}
	}
	else if (x_outside_border) {
		/* Only xout lies outside the border */
		*xborder = xmargin;
		*yborder = yout + (yin - yout) * (xmargin - xout)/(xin - xout);
	}
	else {
		/* Only yout lies outside the border */
		*yborder = ymargin;
		*xborder = xout + (xin - xout) * (ymargin - yout)/(yin - yout);
	}
}

int gui_plot_area_draw (GUI_plot_data *data, FILE *redirect)
{
	int i, j;
	double x, y;
	bool needs_ticks = FALSE;

	printf ("* entering gui_plot_area_draw ()\n");

	/* Is there anything to be done? */
	if (data->cno == 0)
		return SUCCESS;

	for (i = 0; i < data->cno; i++) {
		if (data->request[i].query) {
			if (!redirect)
				cairo_set_source_rgb (data->canvas, 0, 0, 1);
			else
				fprintf (redirect, "# Observed data-set %d:\n", i);

			for (j = 0; j < data->request[i].query->indep->dim; j++) {
				if (!gui_plot_xvalue (data, data->request[i].query->indep->val[j], &x)) continue;
				if (!gui_plot_yvalue (data, data->request[i].query->dep->val[j] + data->request[i].offset, &y)) continue;
				if (data->request[i].query->flag->val.iarray[j] == PHOEBE_DATA_OMITTED) continue;

				if (!redirect) {
					cairo_arc (data->canvas, x, y, 2.0, 0, 2 * M_PI);
					cairo_stroke (data->canvas);
				}
				else
					fprintf (redirect, "%lf\t%lf\n", data->request[i].query->indep->val[j], data->request[i].query->dep->val[j] + data->request[i].offset);
			}
			needs_ticks = TRUE;
		}

		if (data->request[i].model) {
			bool last_point_plotted = FALSE;
			bool x_in_plot, y_in_plot;
			double lastx, lasty;

			if (redirect)
				fprintf (redirect, "# Synthetic data set %d:\n", i);
			else
				cairo_set_source_rgb (data->canvas, 1, 0, 0);

			for (j = 0; j < data->request[i].model->indep->dim; j++) {
				x_in_plot = gui_plot_xvalue (data, data->request[i].model->indep->val[j], &x);
				y_in_plot = gui_plot_yvalue (data, data->request[i].model->dep->val[j] + data->request[i].offset, &y);
				if (!x_in_plot || !y_in_plot) {
					if (last_point_plotted) {
						double xborder, yborder;
						gui_plot_interpolate_to_border (data, lastx, lasty, x, y, &xborder, &yborder);
						if (!redirect)
							cairo_line_to (data->canvas, xborder, yborder);
						else
							fprintf (redirect, "%lf\t%lf\n", data->request[i].model->indep->val[j], data->request[i].model->dep->val[j] + data->request[i].offset);

						last_point_plotted = FALSE;
					}
				}
				else {
					if (!last_point_plotted && (j > 0)) {
						double xborder, yborder;
						gui_plot_interpolate_to_border (data, x, y, lastx, lasty, &xborder, &yborder);
						if (!redirect)
							cairo_move_to (data->canvas, xborder, yborder);
						last_point_plotted = TRUE;
					}
					if (last_point_plotted) {
						if (!redirect)
							cairo_line_to (data->canvas, x, y);
						else
							fprintf (redirect, "%lf\t%lf\n", data->request[i].model->indep->val[j], data->request[i].model->dep->val[j] + data->request[i].offset);
					}
					else {
						if (!redirect)
							cairo_move_to (data->canvas, x, y);
						last_point_plotted = TRUE;
					}
				}
				lastx = x;
				lasty = y;
			}

			if (!redirect)
				cairo_stroke (data->canvas);

			needs_ticks = TRUE;
		}
	}

	if (needs_ticks && !redirect) {
		gui_plot_xticks (data);
		gui_plot_yticks (data);
	}

	printf ("* leaving gui_plot_area_draw ()\n");

	return SUCCESS;
}

void on_lc_plot_treeview_row_changed (GtkTreeModel *tree_model, GtkTreePath *path, GtkTreeIter *iter, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	int i, rows;
	GtkTreeIter traverser;

	bool obs, syn;
	char *obscolor, *syncolor;
	double offset;

	/* Count rows in the model: */
	rows = gtk_tree_model_iter_n_children (tree_model, NULL);
	printf ("no. of rows: %d\n", rows);

	/* Reallocate memory for the plot properties: */
	data->request = phoebe_realloc (data->request, rows * sizeof (*(data->request)));
	if (rows == 0) data->request = NULL;

	/* Traverse all rows and update the values in the plot structure: */
	for (i = 0; i < rows; i++) {
		gtk_tree_model_iter_nth_child (tree_model, &traverser, NULL, i);
		gtk_tree_model_get (tree_model, &traverser, LC_COL_PLOT_OBS, &obs, LC_COL_PLOT_SYN, &syn, LC_COL_PLOT_OBS_COLOR, &obscolor, LC_COL_PLOT_SYN_COLOR, &syncolor, LC_COL_PLOT_OFFSET, &offset, -1);
		data->request[i].plot_obs = obs;
		data->request[i].plot_syn = syn;
		data->request[i].obscolor = obscolor;
		data->request[i].syncolor = syncolor;
		data->request[i].offset   = offset;
		data->request[i].raw      = NULL;
		data->request[i].query    = NULL;
		data->request[i].model    = NULL;
printf ("row %d/%d: (%d, %d, %s, %s, %lf)\n", i, rows, data->request[i].plot_obs, data->request[i].plot_syn, data->request[i].obscolor, data->request[i].syncolor, data->request[i].offset);
	}
	data->cno = rows;

	return;
}

void on_rv_plot_treeview_row_changed (GtkTreeModel *tree_model, GtkTreePath *path, GtkTreeIter *iter, gpointer user_data)
{
	GUI_plot_data *data = (GUI_plot_data *) user_data;
	int i, rows;
	GtkTreeIter traverser;

	bool obs, syn;
	char *obscolor, *syncolor;
	double offset;

	printf ("* entered on_rv_plot_treeview_row_changed() function.\n");

	/* Count rows in the model: */
	rows = gtk_tree_model_iter_n_children (tree_model, NULL);
	printf ("no. of rows: %d\n", rows);

	/* Reallocate memory for the plot properties: */
	data->request = phoebe_realloc (data->request, rows * sizeof (*(data->request)));
	if (rows == 0) data->request = NULL;

	/* Traverse all rows and update the values in the plot structure: */
	for (i = 0; i < rows; i++) {
		gtk_tree_model_iter_nth_child (tree_model, &traverser, NULL, i);
		gtk_tree_model_get (tree_model, &traverser, RV_COL_PLOT_OBS, &obs, RV_COL_PLOT_SYN, &syn, RV_COL_PLOT_OBS_COLOR, &obscolor, RV_COL_PLOT_SYN_COLOR, &syncolor, RV_COL_PLOT_OFFSET, &offset, -1);
		data->request[i].plot_obs = obs;
		data->request[i].plot_syn = syn;
		data->request[i].obscolor = obscolor;
		data->request[i].syncolor = syncolor;
		data->request[i].offset   = offset;
		data->request[i].raw      = NULL;
		data->request[i].query    = NULL;
		data->request[i].model    = NULL;
printf ("row %d/%d: (%d, %d, %s, %s, %lf)\n", i, rows, data->request[i].plot_obs, data->request[i].plot_syn, data->request[i].obscolor, data->request[i].syncolor, data->request[i].offset);
	}
	data->cno = rows;

	return;
}

void on_plot_treeview_row_deleted (GtkTreeModel *model, GtkTreePath *path, gpointer user_data)
{
	GtkTreeIter iter;

	gtk_tree_model_get_iter (model, &iter, path);
	gtk_tree_model_row_changed (model, path, &iter);

	return;
}

int gui_plot_area_init (GtkWidget *area, GtkWidget *button)
{
	/*
	 * gui_plot_area_init:
	 * @area: plot container widget
	 * @button: "Plot" button
	 * 
	 * This is a generic function to initialize the passed plot area for
	 * plotting with Cairo and connects the data pointer through the plot
	 * button.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	GtkWidget *widget;
	GUI_plot_data *data;

	if (!area) {
		printf ("  *** NULL pointer passed to the function.\n");
		return -1 /*** FIX THIS ***/;
	}

	/* Move this to the alloc function: */
	data = phoebe_malloc (sizeof (*data));
	data->layout     = gui_plot_layout_new ();
	data->canvas     = NULL;
	data->request    = NULL;
	data->cno        = 0;
	data->x_offset   = 0.0;
	data->y_offset   = 0.0;
	data->zoom_level = 0.0;
	data->zoom       = 0;
	data->leftmargin = data->layout->lmargin;
	/***********************************/

	gtk_widget_add_events (area, GDK_POINTER_MOTION_MASK | GDK_KEY_PRESS_MASK | GDK_ENTER_NOTIFY_MASK);

	g_signal_connect (area, "expose-event", G_CALLBACK (on_plot_area_expose_event), data);
	g_signal_connect (area, "motion-notify-event", G_CALLBACK (on_plot_area_motion), data);
	g_signal_connect (area, "enter-notify-event", G_CALLBACK (on_plot_area_enter), NULL);
/*
	g_signal_connect (area, "key-press-event", G_CALLBACK (on_key_press_event), data);
*/

	/* Get associations from the button and attach change-sensitive callbacks: */
	data->ctype = *((PHOEBE_curve_type *) (g_object_get_data (G_OBJECT (button), "curve_type")));

	data->container = area;

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_x_request");
	data->x_request = gtk_combo_box_get_active_text (GTK_COMBO_BOX (widget));
	g_signal_connect (widget, "changed", G_CALLBACK (on_combo_box_selection_changed_get_string), &(data->x_request));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_y_request");
	data->y_request = gtk_combo_box_get_active_text (GTK_COMBO_BOX (widget));
	g_signal_connect (widget, "changed", G_CALLBACK (on_combo_box_selection_changed_get_string), &(data->y_request));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "phase_start");
	data->x_ll = gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget));
	g_signal_connect (widget, "value-changed", G_CALLBACK (on_spin_button_value_changed), &(data->x_ll));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "phase_end");
	data->x_ul = gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget));
	g_signal_connect (widget, "value-changed", G_CALLBACK (on_spin_button_value_changed), &(data->x_ul));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_alias_switch");
	data->alias = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget));
	g_signal_connect (widget, "toggled", G_CALLBACK (on_toggle_button_value_toggled), &(data->alias));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_resid_switch");
	data->residuals = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget));
	g_signal_connect (widget, "toggled", G_CALLBACK (on_toggle_button_value_toggled), &(data->residuals));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_vertices");
	data->vertices = gtk_spin_button_get_value (GTK_SPIN_BUTTON (widget));
	g_signal_connect (widget, "value-changed", G_CALLBACK (on_spin_button_intvalue_changed), &(data->vertices));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "coarse_grid_switch");
	data->coarse_grid = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget));
	g_signal_connect (widget, "toggled", G_CALLBACK (on_toggle_button_value_toggled), &(data->coarse_grid));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "fine_grid_switch");
	data->fine_grid = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (widget));
	g_signal_connect (widget, "toggled", G_CALLBACK (on_toggle_button_value_toggled), &(data->fine_grid));

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_left");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_left_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_down");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_down_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_right");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_right_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_up");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_up_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_reset");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_reset_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_zoomin");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_zoomin_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "controls_zoomout");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_controls_zoomout_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "save_plot");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_save_button_clicked), data);

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "clear_plot");
	g_signal_connect (widget, "clicked", G_CALLBACK (on_plot_clear_button_clicked), data);

	/* Sadly, columns don't have any "changed" signal, so we need to use the
	 * model. That implies different actions for LCs and RVs, so the
	 * implementation is not as elegant as for the rest. Furthermore, removing
	 * rows does not emit the "row-changed" signal, so we have to catch the
	 * "row-deleted" signal as well and re-route it to "row-changed".
	 */

	widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_passband_info");
	if (data->ctype == PHOEBE_CURVE_LC)
		g_signal_connect (widget, "row-changed", G_CALLBACK (on_lc_plot_treeview_row_changed), data);
	else if (data->ctype == PHOEBE_CURVE_RV)
		g_signal_connect (widget, "row-changed", G_CALLBACK (on_rv_plot_treeview_row_changed), data);

	g_signal_connect (widget, "row-deleted", G_CALLBACK (on_plot_treeview_row_deleted), data);

	/* Assign the widgets (must be GtkLabels) that will keep track of mouse
	 * coordinates:
	 */

	data->x_widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_x_coordinate");
	data->y_widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_y_coordinate");

	data->cp_widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_cp_index");
	data->cx_widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_cx_coordinate");
	data->cy_widget = (GtkWidget *) g_object_get_data (G_OBJECT (button), "plot_cy_coordinate");

	/* Finally, attach a callback that will plot the data: */
	g_signal_connect (button, "clicked", G_CALLBACK (on_plot_button_clicked), data);

	return SUCCESS;
}

int gui_tempfile(char *filename) 
{
#ifdef __MINGW32__
	return g_mkstemp(filename);
#else
	return mkstemp(filename);
#endif
}

void gui_plot(char *filename) 
{
	char command[255];

#ifdef __MINGW32__
	sprintf(command,"wgnuplot \"%s\"", filename);
	WinExec(command, SW_SHOWMINIMIZED);
#else
	sprintf(command,"gnuplot \"%s\"", filename);
	system(command);
#endif
}

int gui_plot_get_curve_limits (PHOEBE_curve *curve, double *xmin, double *ymin, double *xmax, double *ymax)
{
	int i;

	*xmin = 0; *xmax = 0;
	*ymin = 0; *ymax = 0;

	if (!curve)
		return ERROR_CURVE_NOT_INITIALIZED;

	*xmin = curve->indep->val[0]; *xmax = curve->indep->val[0];
	*ymin = curve->dep->val[0];   *ymax = curve->dep->val[0];

	for (i = 1; i < curve->indep->dim; i++) {
		if (curve->flag->val.iarray[i] == PHOEBE_DATA_OMITTED) continue;
		if (*xmin > curve->indep->val[i]) *xmin = curve->indep->val[i];
		if (*xmax < curve->indep->val[i]) *xmax = curve->indep->val[i];
		if (*ymin > curve->dep->val[i]  ) *ymin = curve->dep->val[i];
		if (*ymax < curve->dep->val[i]  ) *ymax = curve->dep->val[i];
	}

	return SUCCESS;
}

int gui_plot_get_offset_zoom_limits (double min, double max, double offset, double zoom, double *newmin, double *newmax)
{
	*newmin = min + offset * (max - min) - (0.1 + zoom) * (max - min);
	*newmax = max + offset * (max - min) + (0.1 + zoom) * (max - min);

	return SUCCESS;
}

int gui_plot_get_plot_limits (PHOEBE_curve *syn, PHOEBE_curve *obs, double *xmin, double *ymin, double *xmax, double *ymax, gboolean plot_syn, gboolean plot_obs, double x_offset, double y_offset, double zoom)
{
	int status;
	double xmin1, xmax1, ymin1, ymax1;              /* Synthetic data limits    */
	double xmin2, xmax2, ymin2, ymax2;              /* Experimental data limits */

	if (plot_syn) {
		status = gui_plot_get_curve_limits (syn, &xmin1, &ymin1, &xmax1, &ymax1);
		if (status != SUCCESS) return status;
	}
	if (plot_obs) {
		status = gui_plot_get_curve_limits (obs, &xmin2, &ymin2, &xmax2, &ymax2);
		if (status != SUCCESS) return status;
	}

	if (plot_syn)
	{
		gui_plot_get_offset_zoom_limits (xmin1, xmax1, x_offset, zoom, xmin, xmax);
		gui_plot_get_offset_zoom_limits (ymin1, ymax1, y_offset, zoom, ymin, ymax);
		xmin1 = *xmin; xmax1 = *xmax; ymin1 = *ymin; ymax1 = *ymax;
	}

	if (plot_obs)
	{
		gui_plot_get_offset_zoom_limits (xmin2, xmax2, x_offset, zoom, xmin, xmax);
		gui_plot_get_offset_zoom_limits (ymin2, ymax2, y_offset, zoom, ymin, ymax);
		xmin2 = *xmin; xmax2 = *xmax; ymin2 = *ymin; ymax2 = *ymax;
	}

	if (plot_syn && plot_obs) {
		if (xmin1 < xmin2) *xmin = xmin1; else *xmin = xmin2;
		if (xmax1 > xmax2) *xmax = xmax1; else *xmax = xmax2;
		if (ymin1 < ymin2) *ymin = ymin1; else *ymin = ymin2;
		if (ymax1 > ymax2) *ymax = ymax1; else *ymax = ymax2;
	}

	return SUCCESS;
}

int gui_plot_lc_using_gnuplot (gdouble x_offset, gdouble y_offset, gdouble zoom)
{
	PHOEBE_curve *obs = NULL;
	PHOEBE_curve *syn = NULL;

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
	GtkWidget *alias_checkbutton	 	= gui_widget_lookup ("phoebe_lc_plot_options_alias_checkbutton")->gtk;
	GtkWidget *residual_checkbutton	 	= gui_widget_lookup ("phoebe_lc_plot_options_residuals_checkbutton")->gtk;
	GtkWidget *vertices_no_spinbutton	= gui_widget_lookup ("phoebe_lc_plot_options_vertices_no_spinbutton")->gtk;
	GtkWidget *obs_combobox 			= gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox")->gtk;
	GtkWidget *x_combobox 				= gui_widget_lookup ("phoebe_lc_plot_options_x_combobox")->gtk;
	GtkWidget *y_combobox				= gui_widget_lookup ("phoebe_lc_plot_options_y_combobox")->gtk;
	GtkWidget *phstart_spinbutton 		= gui_widget_lookup ("phoebe_lc_plot_options_phstart_spinbutton")->gtk;
	GtkWidget *phend_spinbutton			= gui_widget_lookup ("phoebe_lc_plot_options_phend_spinbutton")->gtk;

	GtkWidget *coarse_grid				= gui_widget_lookup ("phoebe_lc_plot_controls_coarse_checkbutton")->gtk;
	GtkWidget *fine_grid				= gui_widget_lookup ("phoebe_lc_plot_controls_fine_checkbutton")->gtk;

	gint VERTICES 	= gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(vertices_no_spinbutton));
	gint INDEX		= -1;
	gint INDEP = (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 0) ? PHOEBE_COLUMN_PHASE : PHOEBE_COLUMN_HJD;
	gint DEP   = (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 0) ? PHOEBE_COLUMN_FLUX  : PHOEBE_COLUMN_MAGNITUDE;

	gdouble XMIN = 0.0, XMAX = 0.0, YMIN = 0.0, YMAX = 0.0;

	gboolean plot_obs = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(obs_checkbutton));
	gboolean plot_syn = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(syn_checkbutton));

	gboolean ALIAS = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(alias_checkbutton));
	gboolean residuals = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(residual_checkbutton));

	gdouble phstart = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phstart_spinbutton));
	gdouble phend = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phend_spinbutton));

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	//----------------

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == -1){
		gtk_combo_box_set_active (GTK_COMBO_BOX(y_combobox), 0);
		DEP 	= PHOEBE_COLUMN_FLUX;
	}

	INDEX = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox));

	if (INDEX < 0){
		INDEX = 0;
		gtk_combo_box_set_active (GTK_COMBO_BOX(obs_combobox), 0);
	}

	if (plot_obs) {
		obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, INDEX);
		if (!obs) {
			plot_obs = FALSE;
			gui_notice ("Observed curve not available", "The filename of the observed curve is not given or is invalid.");
		}
		else {
			phoebe_curve_transform (obs, INDEP, DEP, PHOEBE_COLUMN_UNDEFINED);
			if (ALIAS)
				phoebe_curve_alias (obs, phstart, phend);
		}
	}

	if (plot_syn) {
		syn = phoebe_curve_new ();
		syn->type = PHOEBE_CURVE_LC;

		if (residuals && plot_obs) {
			indep = phoebe_vector_duplicate (obs->indep);
		}
		else {
			indep = phoebe_vector_new ();
			phoebe_vector_alloc (indep, VERTICES);
			if (INDEP == PHOEBE_COLUMN_HJD && plot_obs){
				double hjd_min,hjd_max;
				phoebe_vector_min_max (obs->indep, &hjd_min, &hjd_max);
				for (i = 0; i < VERTICES; i++)
					indep->val[i] = hjd_min + (hjd_max-hjd_min) * (double) i/(VERTICES-1);
			}
			else {
				for (i = 0; i < VERTICES; i++)
					indep->val[i] = phstart + (phend-phstart) * (double) i/(VERTICES-1);
			}
		}

		status = phoebe_curve_compute (syn, indep, INDEX, INDEP, DEP);
		if (status != SUCCESS) {
			char *message = phoebe_concatenate_strings ("Configuration problem: ", phoebe_error (status), NULL);
			gui_notice ("LC plot", message);
			free (message);
			return status;
		}

		if (ALIAS)
			phoebe_curve_alias (syn, phstart, phend);
		if (residuals && plot_obs) {
			for (i = 0; i < syn->indep->dim; i++) {
				obs->dep->val[i] -= syn->dep->val[i];
				syn->dep->val[i] = 0.0;
			}
		}

		phoebe_vector_free (indep);
	}

	/* Write the data to a file: */
	if (plot_obs) {
		sprintf(oname, "%s/phoebe-lc-XXXXXX", tmpdir);
		ofd = gui_tempfile (oname);
		for (i=0;i<obs->indep->dim;i++) {
			sprintf(line, "%lf\t%lf\t%lf\n", obs->indep->val[i], obs->dep->val[i], obs->weight->val[i]) ;
			write(ofd, line, strlen(line));
		}
		close(ofd);
	}
	if (plot_syn) {
		sprintf(sname, "%s/phoebe-lc-XXXXXX", tmpdir);
		sfd = gui_tempfile (sname);
		for (i = 0; i < syn->indep->dim; i++) {
			sprintf(line, "%lf\t%lf\n", syn->indep->val[i], syn->dep->val[i]) ;
			write(sfd, line, strlen(line));
		}
		close(sfd);
	}

	/* open command file */
	sprintf(cname, "%s/phoebe-lc-XXXXXX", tmpdir);
	cfd = gui_tempfile (cname);

	/* gnuplot 4.0 has a bug in the docs that says keyword "size" is recognized
	 * whereas it isn't. That is why we check for the gnuplot version in the
	 * configure script and use it here.
	 */

#ifdef PHOEBE_GUI_GNUPLOT_LIBGD
	sprintf(line, "set terminal png small size 590,310\n"); 			write(cfd, line, strlen(line));
#else
	sprintf(line, "set terminal png small picsize 590 310\n"); 			write(cfd, line, strlen(line));
#endif
	sprintf(line, "set mxtics 2\n"); 									write(cfd, line, strlen(line));
	sprintf(line, "set mytics 2\n"); 									write(cfd, line, strlen(line));
	sprintf(line, "set lmargin 6\n");									write(cfd, line, strlen(line));
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

	gui_plot_get_plot_limits (syn, obs, &XMIN, &YMIN, &XMAX, &YMAX, plot_syn, plot_obs, x_offset, y_offset, zoom);

	sprintf(line, "set xrange [%lf:%lf]\n", XMIN, XMAX); 				write(cfd, line, strlen(line));
	if (DEP == PHOEBE_COLUMN_MAGNITUDE)
		{sprintf(line, "set yrange [%lf:%lf]\n", YMAX, YMIN); 			write(cfd, line, strlen(line));}
	if (DEP == PHOEBE_COLUMN_FLUX)
		{sprintf(line, "set yrange [%lf:%lf]\n", YMIN, YMAX); 			write(cfd, line, strlen(line));}

	if (INDEP == PHOEBE_COLUMN_HJD)
		{sprintf(line, "set format x '%%7.0f'\n");			 			write(cfd, line, strlen(line));}

	sprintf(pname, "%s/phoebe-lc-plot-XXXXXX", tmpdir);
	pfd = gui_tempfile (pname);

	sprintf(line, "set output '%s'\n", pname);							write(cfd, line, strlen(line));

	if (plot_syn && plot_obs)
		{sprintf(line, "plot '%s' w p lt 3 lw 1 pt 6 notitle, '%s' w l lt 1 notitle\n", oname, sname);	write(cfd, line, strlen(line));}
	else if (plot_syn)
		{sprintf(line, "plot '%s' w l lt 1 notitle\n", sname);											write(cfd, line, strlen(line));}
	else if (plot_obs)
		{sprintf(line, "plot '%s' w p lt 3 lw 1 pt 6 notitle\n", oname);								write(cfd, line, strlen(line));}

	close(cfd);

	gui_plot(cname);

	if (plot_syn || plot_obs) {
		GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(pname, NULL);
		gtk_image_set_from_pixbuf(GTK_IMAGE(plot_image), pixbuf);
		gdk_pixbuf_unref(pixbuf);
	}

	close(pfd);

	//----------------

	remove(oname);
	remove(sname);
	remove(cname);
	remove(pname);

	if (plot_syn) phoebe_curve_free(syn);
	if (plot_obs) phoebe_curve_free(obs);

	gui_beep();

	return SUCCESS;
}

int gui_plot_lc_to_ascii (gchar *filename)
{
	PHOEBE_curve *obs = NULL;
	PHOEBE_curve *syn = NULL;

	PHOEBE_vector *indep;

	FILE *file;

	gint i;
	gint status;

	GtkWidget *syn_checkbutton 			= gui_widget_lookup ("phoebe_lc_plot_options_syn_checkbutton")->gtk;
	GtkWidget *obs_checkbutton 			= gui_widget_lookup ("phoebe_lc_plot_options_obs_checkbutton")->gtk;
	GtkWidget *alias_checkbutton	 	= gui_widget_lookup ("phoebe_lc_plot_options_alias_checkbutton")->gtk;
	GtkWidget *residual_checkbutton	 	= gui_widget_lookup ("phoebe_lc_plot_options_residuals_checkbutton")->gtk;
	GtkWidget *vertices_no_spinbutton	= gui_widget_lookup ("phoebe_lc_plot_options_vertices_no_spinbutton")->gtk;
	GtkWidget *obs_combobox 			= gui_widget_lookup ("phoebe_lc_plot_options_obs_combobox")->gtk;
	GtkWidget *x_combobox 				= gui_widget_lookup ("phoebe_lc_plot_options_x_combobox")->gtk;
	GtkWidget *y_combobox				= gui_widget_lookup ("phoebe_lc_plot_options_y_combobox")->gtk;
	GtkWidget *phstart_spinbutton 		= gui_widget_lookup ("phoebe_lc_plot_options_phstart_spinbutton")->gtk;
	GtkWidget *phend_spinbutton			= gui_widget_lookup ("phoebe_lc_plot_options_phend_spinbutton")->gtk;

	gint VERITCES 	= gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(vertices_no_spinbutton));
	gint INDEX		= -1;
	gint INDEP = (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 0) ? PHOEBE_COLUMN_PHASE : PHOEBE_COLUMN_HJD;
	gint DEP   = (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == 0) ? PHOEBE_COLUMN_FLUX  : PHOEBE_COLUMN_MAGNITUDE;

	gboolean plot_obs = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(obs_checkbutton));
	gboolean plot_syn = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(syn_checkbutton));

	gboolean ALIAS = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(alias_checkbutton));
	gboolean residuals = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(residual_checkbutton));

	gdouble phstart = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phstart_spinbutton));
	gdouble phend = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phend_spinbutton));

	if (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox)) == -1){
		gtk_combo_box_set_active (GTK_COMBO_BOX(y_combobox), 0);
		DEP 	= PHOEBE_COLUMN_FLUX;
	}

	INDEX = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox));

	if (INDEX < 0){
		INDEX = 0;
		gtk_combo_box_set_active (GTK_COMBO_BOX(obs_combobox), 0);
	}


	if (plot_obs) {
		obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, INDEX);
		if (!obs) {
			plot_obs = FALSE;
			gui_notice ("Observed curve not available", "The filename of the observed curve is not given or is invalid.");
		}
		else {
			phoebe_curve_transform (obs, INDEP, DEP, PHOEBE_COLUMN_UNDEFINED);
			if (ALIAS) phoebe_curve_alias (obs, phstart, phend);
		}
	}

	if (plot_syn) {
		syn = phoebe_curve_new ();
		syn->type = PHOEBE_CURVE_LC;

		if (residuals && plot_obs) {
			indep = phoebe_vector_duplicate (obs->indep);
		}
		else {
			indep = phoebe_vector_new ();
			phoebe_vector_alloc (indep, VERITCES);
			if (INDEP == PHOEBE_COLUMN_HJD && plot_obs){
				double hjd_min,hjd_max;
				phoebe_vector_min_max (obs->indep, &hjd_min, &hjd_max);
				for (i = 0; i < VERITCES; i++)
					indep->val[i] = hjd_min + (hjd_max-hjd_min) * (double) i/(VERITCES-1);
			}
			else {
				for (i = 0; i < VERITCES; i++)
					indep->val[i] = phstart + (phend-phstart) * (double) i/(VERITCES-1);
			}
		}
		status = phoebe_curve_compute (syn, indep, INDEX, INDEP, DEP);
		if (status != SUCCESS) {
			gui_notice("LC plot", phoebe_error(status));
			return status;
		}

		if (ALIAS)
			phoebe_curve_alias (syn, phstart, phend);
		if (residuals && plot_obs) {
			for (i = 0; i < syn->indep->dim; i++) {
				obs->dep->val[i] -= syn->dep->val[i];
				syn->dep->val[i] = 0.0;
			}
		}

		phoebe_vector_free (indep);
	}

	file = fopen (filename, "w");
	if (!file) {
		gui_notice ("File cannot be saved", "The file cannot be opened for output, aborting.");
	}
	else {
		if (plot_obs) {
			fprintf(file, "#OBSERVED DATA\n");
			for (i=0;i<obs->indep->dim;i++) fprintf(file, "%lf\t%lf\n",obs->indep->val[i], obs->dep->val[i]);
		}
		if (plot_syn) {
			fprintf(file, "#SYNTHETIC DATA\n");
			for (i=0;i<syn->indep->dim;i++) fprintf(file, "%lf\t%lf\n",syn->indep->val[i], syn->dep->val[i]);
		}
		fclose(file);
	}

	if (plot_syn) phoebe_curve_free (syn);
	if (plot_obs) phoebe_curve_free (obs);

	return SUCCESS;
}

gint gui_rv_component_index (gint DEP)
{
	/* Returns the index number of the RV file that corresponds to the given component. */

	char *param;
	PHOEBE_column_type dtype;
	int status, index;

	for (index = 0; index <= 1; index++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), index, &param);
		status = phoebe_column_get_type (&dtype, param);
		if ((status == SUCCESS) && (dtype == DEP))
			return index;
	}

	return -1;
}

int gui_rv_hjd_minmax (double *hjd_min, double *hjd_max)
{
	gint index[2];
	int i, present = 0;
	
	index[0] = gui_rv_component_index(PHOEBE_COLUMN_PRIMARY_RV);
	index[1] = gui_rv_component_index(PHOEBE_COLUMN_SECONDARY_RV);

	for (i = 0; i <= 1; i++) {
		if (index[i] >= 0) {
			PHOEBE_curve *obs = NULL;
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, index[i]);
			if (obs) {
				double min, max;
				phoebe_vector_min_max (obs->indep, &min, &max);
				phoebe_curve_free(obs);
				if (present) {
					if (*hjd_min > min) *hjd_min = min;
					if (*hjd_max < max) *hjd_max = max;
				}
				else {
					present = 1;
					*hjd_min = min;
					*hjd_max = max;
				}
			}
		}
	}

	return present;
}

int gui_plot_rv_using_gnuplot_setup (gint INDEX, gint DEP, gint INDEP, gboolean plot_obs, gboolean plot_syn, gboolean plot_residuals, gchar *oname, gchar *sname, 
					gdouble *XMIN, gdouble *XMAX, gdouble *YMIN, gdouble *YMAX,
					gdouble x_offset, gdouble y_offset, gdouble zoom, double hjd_min, double hjd_max, gboolean *plot_observations)
{
	/* Sets up the data and synthetic files for one of the radial velocity curves. */
	PHOEBE_curve *obs = NULL;
	PHOEBE_curve *syn = NULL;

	PHOEBE_vector *indep;

	gchar *tmpdir;

	gint ofd, sfd;

	gint i;
	gint status;

	gchar  line[255];

	GtkWidget *vertices_no_spinbutton 	= gui_widget_lookup ("phoebe_rv_plot_options_vertices_no_spinbutton")->gtk;
	GtkWidget *alias_checkbutton	 	= gui_widget_lookup ("phoebe_rv_plot_options_alias_checkbutton")->gtk;
	GtkWidget *phstart_spinbutton 		= gui_widget_lookup ("phoebe_rv_plot_options_phstart_spinbutton")->gtk;
	GtkWidget *phend_spinbutton		= gui_widget_lookup ("phoebe_rv_plot_options_phend_spinbutton")->gtk;

	gint VERTICES 	= gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(vertices_no_spinbutton));

	gboolean ALIAS = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(alias_checkbutton));

	gdouble phstart = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phstart_spinbutton));
	gdouble phend = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phend_spinbutton));

	gint RV_INDEX = gui_rv_component_index(DEP);
	if (plot_residuals)
		plot_obs = TRUE;

	if (plot_obs) {
		if (RV_INDEX < 0) {
			plot_obs = FALSE;
		}
		else {
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, RV_INDEX);
			if (!obs) {
				plot_obs = FALSE;
				gui_notice ("Observed curve not available", "The filename of the observed curve is not given or is invalid.");
			}
			else {
				phoebe_curve_transform (obs, INDEP, DEP, PHOEBE_COLUMN_UNDEFINED);
				if (ALIAS)
					phoebe_curve_alias (obs, phstart, phend);
			}
		}
	}

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	if (plot_syn || plot_residuals) {
		syn = phoebe_curve_new ();
		syn->type = PHOEBE_CURVE_RV;

		if (plot_residuals && plot_obs) {
			indep = phoebe_vector_duplicate (obs->indep);
		}
		else {
			double xmin, xmax;
			indep = phoebe_vector_new ();
			phoebe_vector_alloc (indep, VERTICES);
			// First determine more accurate start and end points depending on the offset and zoom to get a more detailed synthetic curve
			if (INDEP == PHOEBE_COLUMN_HJD) {
				gui_plot_get_offset_zoom_limits (hjd_min, hjd_max, x_offset, zoom - 0.1, &xmin, &xmax);
			}
			else {
				gui_plot_get_offset_zoom_limits (phstart, phend, x_offset, zoom - 0.1, &xmin, &xmax);
			}
			for (i = 0; i < VERTICES; i++)
				indep->val[i] = xmin + (xmax-xmin) * (double) i/(VERTICES-1);
		}

		status = phoebe_curve_compute (syn, indep, INDEX, INDEP, DEP);
		if (status != SUCCESS) {
			gui_notice("RV plot", phoebe_error(status));
			return status;
		}

		if (ALIAS)
			phoebe_curve_alias (syn, phstart, phend);
		if (plot_residuals) {
			for (i = 0; i < syn->indep->dim; i++) {
				if (plot_obs)
					obs->dep->val[i] -= syn->dep->val[i];
				syn->dep->val[i] = 0.0;
			}
		}

		phoebe_vector_free (indep);

		if (plot_syn) {
			/* Write the data to a file: */
			sprintf(sname, "%s/phoebe-rv-XXXXXX", tmpdir);
			sfd = gui_tempfile (sname);
			for (i = 0; i < syn->indep->dim; i++) {
				if (obs->flag->val.iarray[i] == PHOEBE_DATA_OMITTED) continue;
				sprintf(line, "%lf\t%lf\n", syn->indep->val[i], syn->dep->val[i]) ;
				write(sfd, line, strlen(line));
			}
			close(sfd);
		}
	}

	if (plot_obs) {
		/* Write the data to a file: */
		sprintf(oname, "%s/phoebe-rv-XXXXXX", tmpdir);
		ofd = gui_tempfile (oname);

		for (i=0;i<obs->indep->dim;i++) {
			if (obs->flag->val.iarray[i] == PHOEBE_DATA_OMITTED) continue;
			sprintf(line, "%lf\t%lf\t%lf\n", obs->indep->val[i], obs->dep->val[i], obs->weight->val[i]) ;
			write(ofd, line, strlen(line));
		}
		close(ofd);
	}

	gui_plot_get_plot_limits (syn, obs, XMIN, YMIN, XMAX, YMAX, plot_syn, plot_obs, x_offset, y_offset, zoom);

	if (plot_obs) phoebe_curve_free(obs);
	if (plot_syn) phoebe_curve_free(syn);

	*plot_observations = plot_obs;
	return SUCCESS;
}

int gui_plot_rv_using_gnuplot (gdouble x_offset, gdouble y_offset, gdouble zoom)
{
	gchar *tmpdir;
	gchar o1name[255];
	gchar s1name[255];
	gchar o2name[255];
	gchar s2name[255];
	gchar cname[255];
	gchar pname[255];
	gchar  line[255];

	gint cfd, pfd, status;

	gboolean plot_obs1 = 0, plot_obs2 = 0, plot_component;
	gchar *plot = "plot";

	GtkWidget *plot_image				= gui_widget_lookup ("phoebe_rv_plot_image")->gtk;
	GtkWidget *syn_checkbutton 			= gui_widget_lookup ("phoebe_rv_plot_options_syn_checkbutton")->gtk;
	GtkWidget *obs_checkbutton 			= gui_widget_lookup ("phoebe_rv_plot_options_obs_checkbutton")->gtk;
	GtkWidget *residual_checkbutton	 	= gui_widget_lookup ("phoebe_rv_plot_options_residuals_checkbutton")->gtk;
	GtkWidget *obs_combobox 			= gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox")->gtk;
	GtkWidget *x_combobox 				= gui_widget_lookup ("phoebe_rv_plot_options_x_combobox")->gtk;
	GtkWidget *y_combobox				= gui_widget_lookup ("phoebe_rv_plot_options_y_combobox")->gtk;

	GtkWidget *coarse_grid				= gui_widget_lookup ("phoebe_rv_plot_controls_coarse_checkbutton")->gtk;
	GtkWidget *fine_grid				= gui_widget_lookup ("phoebe_rv_plot_controls_fine_checkbutton")->gtk;

	gint INDEX		= -1;
	gint INDEP = (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 0) ? PHOEBE_COLUMN_PHASE : PHOEBE_COLUMN_HJD;

	gdouble XMIN = 0.0, XMAX = 0.0, YMIN = 0.0, YMAX = 0.0;
	gdouble XMIN2 = 0.0, XMAX2 = 0.0, YMIN2 = 0.0, YMAX2 = 0.0;
	double hjd_min, hjd_max;

	gboolean plot_obs = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(obs_checkbutton));
	gboolean plot_syn = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(syn_checkbutton));

	gboolean residuals = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(residual_checkbutton));

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	if (INDEP == PHOEBE_COLUMN_HJD) {
		if (!gui_rv_hjd_minmax(&hjd_min, &hjd_max)) {
			// No dates available
			INDEP = PHOEBE_COLUMN_PHASE;
		}
	}

	INDEX = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox));

	if (INDEX < 0){
		INDEX = 0;
		gtk_combo_box_set_active (GTK_COMBO_BOX(obs_combobox), 0);
	}

	plot_component = gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox));
	switch (plot_component) {
		case 0:	status = gui_plot_rv_using_gnuplot_setup (INDEX, PHOEBE_COLUMN_PRIMARY_RV, INDEP, plot_obs, plot_syn, residuals, o1name, s1name, 
					&XMIN, &XMAX, &YMIN, &YMAX, x_offset, y_offset, zoom, hjd_min, hjd_max, &plot_obs1);
			if (status != SUCCESS) return status;
			break;
		case 1:	status = gui_plot_rv_using_gnuplot_setup (INDEX, PHOEBE_COLUMN_SECONDARY_RV, INDEP, plot_obs, plot_syn, residuals, o2name, s2name, 
					&XMIN, &XMAX, &YMIN, &YMAX, x_offset, y_offset, zoom, hjd_min, hjd_max, &plot_obs2);
			if (status != SUCCESS) return status;
			break;
		case 2:	status = gui_plot_rv_using_gnuplot_setup (INDEX, PHOEBE_COLUMN_PRIMARY_RV, INDEP, plot_obs, plot_syn, residuals, o1name, s1name, 
					&XMIN, &XMAX, &YMIN, &YMAX, x_offset, y_offset, zoom, hjd_min, hjd_max, &plot_obs1);
			if (status != SUCCESS) return status;
			status = gui_plot_rv_using_gnuplot_setup (INDEX, PHOEBE_COLUMN_SECONDARY_RV, INDEP, plot_obs, plot_syn, residuals, o2name, s2name, 
					&XMIN2, &XMAX2, &YMIN2, &YMAX2, x_offset, y_offset, zoom, hjd_min, hjd_max, &plot_obs2);
			if (status != SUCCESS) return status;
			if ((INDEP == PHOEBE_COLUMN_HJD) && (!plot_obs1)) {
				XMIN = XMIN2;
				XMAX = XMAX2;
				YMIN = YMIN2;
				YMAX = YMAX2;
			}
			else if ((INDEP != PHOEBE_COLUMN_HJD) || (plot_obs2)) {
				if (XMIN2 < XMIN)
					XMIN = XMIN2;
				if (XMAX2 > XMAX)
					XMAX = XMAX2;
				if (YMIN2 < YMIN)
					YMIN = YMIN2;
				if (YMAX2 > YMAX)
					YMAX = YMAX2;
			}
			break;
	}

	sprintf(cname, "%s/phoebe-rv-XXXXXX", tmpdir);
	cfd = gui_tempfile (cname);

#ifdef PHOEBE_GUI_GNUPLOT_LIBGD
	sprintf(line, "set terminal png small size 590,310\n"); 			write(cfd, line, strlen(line));
#else
	sprintf(line, "set terminal png small picsize 590 310\n"); 			write(cfd, line, strlen(line));
#endif
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

	sprintf(line, "set xrange [%lf:%lf]\n", XMIN, XMAX); 				write(cfd, line, strlen(line));
	sprintf(line, "set yrange [%lf:%lf]\n", YMIN, YMAX); 			write(cfd, line, strlen(line));

	if (INDEP == PHOEBE_COLUMN_HJD)
		{sprintf(line, "set format x '%%7.0f'\n");			 			write(cfd, line, strlen(line));}

	sprintf(pname, "%s/phoebe-lc-plot-XXXXXX", tmpdir);
	pfd = gui_tempfile (pname);

	sprintf(line, "set output '%s'\n", pname);							write(cfd, line, strlen(line));

	if (plot_obs1) {
		sprintf(line, "%s '%s' w p lt 3 lw 1 pt 6 notitle", plot, o1name);		write(cfd, line, strlen(line));
		plot = ",";
	}
	if (plot_obs2) {
		sprintf(line, "%s '%s' w p lt 2 lw 1 pt 6 notitle", plot, o2name);		write(cfd, line, strlen(line));
		plot = ",";
	}

	if (plot_syn) {
		switch (plot_component) {
			case 0:	sprintf(line, "%s '%s' w l lt 1 notitle", plot, s1name);	write(cfd, line, strlen(line));
				break;
			case 1:	sprintf(line, "%s '%s' w l lt 4 notitle", plot, s2name);	write(cfd, line, strlen(line));
				break;
			case 2:	sprintf(line, "%s '%s' w l lt 1 notitle", plot, s1name);	write(cfd, line, strlen(line));
				plot = ",";
				sprintf(line, "%s '%s' w l lt 4 notitle", plot, s2name);	write(cfd, line, strlen(line));
				break;
		}
	}

	sprintf(line, "\n");									write(cfd, line, strlen(line));

	close(cfd);

	gui_plot(cname);

	if (plot_syn || plot_obs1 || plot_obs2) {
		GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(pname, NULL);
		gtk_image_set_from_pixbuf(GTK_IMAGE(plot_image), pixbuf);
		gdk_pixbuf_unref(pixbuf);
	}
	else if (!plot_syn && !plot_obs1 && !plot_obs2)
		gui_notice("RV plot", "Nothing to plot.");

	close(pfd);

	//----------------

	remove(o1name);
	remove(s1name);
	remove(o2name);
	remove(s2name);
	remove(cname);
	remove(pname);

	gui_beep();

	return SUCCESS;
}

int gui_plot_rv_to_ascii_one_component (FILE *file, gint INDEX, gint INDEP, gint DEP, gboolean plot_obs, gboolean plot_syn)
{
	PHOEBE_curve *obs = NULL;
	PHOEBE_curve *syn = NULL;

	PHOEBE_vector *indep;

	gint i;
	gint status;

	GtkWidget *vertices_no_spinbutton 	= gui_widget_lookup ("phoebe_rv_plot_options_vertices_no_spinbutton")->gtk;
	GtkWidget *residual_checkbutton	 	= gui_widget_lookup ("phoebe_rv_plot_options_residuals_checkbutton")->gtk;
	GtkWidget *alias_checkbutton	 	= gui_widget_lookup ("phoebe_rv_plot_options_alias_checkbutton")->gtk;
	GtkWidget *phstart_spinbutton 		= gui_widget_lookup ("phoebe_rv_plot_options_phstart_spinbutton")->gtk;
	GtkWidget *phend_spinbutton		= gui_widget_lookup ("phoebe_rv_plot_options_phend_spinbutton")->gtk;

	gint VERTICES 	= gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON(vertices_no_spinbutton));

	gboolean ALIAS = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(alias_checkbutton));
	gboolean residuals = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON(residual_checkbutton));
	if (residuals) {
		plot_obs = TRUE;
		plot_syn = FALSE;
	}

	gdouble phstart = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phstart_spinbutton));
	gdouble phend = gtk_spin_button_get_value (GTK_SPIN_BUTTON(phend_spinbutton));

	gint RV_INDEX = gui_rv_component_index(DEP);

	if (plot_obs) {
		if (RV_INDEX < 0) {
			plot_obs = FALSE;
		}
		else {
			obs = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, RV_INDEX);
			if (!obs) {
				plot_obs = FALSE;
				gui_notice ("Observed curve not available", "The filename of the observed curve is not given or is invalid.");
			}
			else {
				phoebe_curve_transform (obs, INDEP, DEP, PHOEBE_COLUMN_UNDEFINED);
				if (ALIAS)
					phoebe_curve_alias (obs, phstart, phend);
			}
		}
	}

	if (plot_syn || residuals) {
		syn = phoebe_curve_new ();
		syn->type = PHOEBE_CURVE_RV;

		if (residuals && plot_obs) {
			indep = phoebe_vector_duplicate (obs->indep);
		}
		else {
			indep = phoebe_vector_new ();
			phoebe_vector_alloc (indep, VERTICES);
			if (INDEP == PHOEBE_COLUMN_HJD) {
				double hjd_min,hjd_max;
				if (obs)
					phoebe_vector_min_max (obs->indep, &hjd_min, &hjd_max);
				else
					gui_rv_hjd_minmax(&hjd_min, &hjd_max);
				for (i = 0; i < VERTICES; i++) indep->val[i] = hjd_min + (hjd_max-hjd_min) * (double) i/(VERTICES-1);
			}
			else {
				for (i = 0; i < VERTICES; i++) indep->val[i] = phstart + (phend-phstart) * (double) i/(VERTICES-1);
			}
		}

		status = phoebe_curve_compute (syn, indep, INDEX, INDEP, DEP);
		if (status != SUCCESS) {
			gui_notice("RV plot", phoebe_error(status));
			return status;
		}

		if (ALIAS)
			phoebe_curve_alias (syn, phstart, phend);
		if (residuals) {
			for (i = 0; i < syn->indep->dim; i++) {
				if (plot_obs)
					obs->dep->val[i] -= syn->dep->val[i];
				syn->dep->val[i] = 0.0;
			}
		}

		phoebe_vector_free (indep);
	}

	if (plot_obs) {
		fprintf(file, "#%sOBSERVED DATA %s COMPONENT\n", (residuals) ? "RESIDUALS " : "", (DEP == PHOEBE_COLUMN_PRIMARY_RV) ? "PRIMARY" : "SECONDARY");
		for (i=0;i<obs->indep->dim;i++) fprintf(file, "%lf\t%lf\n",obs->indep->val[i], obs->dep->val[i]);
		phoebe_curve_free(obs);
	}

	if (plot_syn) {
		fprintf(file, "#SYNTHETIC DATA %s COMPONENT\n", (DEP == PHOEBE_COLUMN_PRIMARY_RV) ? "PRIMARY" : "SECONDARY");
		for (i=0;i<syn->indep->dim;i++) fprintf(file, "%lf\t%lf\n",syn->indep->val[i], syn->dep->val[i]);
		phoebe_curve_free(syn);
	}

	return SUCCESS;
}

int gui_plot_rv_to_ascii (gchar *filename)
{
	FILE *file;

	GtkWidget *syn_checkbutton 			= gui_widget_lookup ("phoebe_rv_plot_options_syn_checkbutton")->gtk;
	GtkWidget *obs_checkbutton 			= gui_widget_lookup ("phoebe_rv_plot_options_obs_checkbutton")->gtk;
	GtkWidget *obs_combobox 			= gui_widget_lookup ("phoebe_rv_plot_options_obs_combobox")->gtk;
	GtkWidget *x_combobox 				= gui_widget_lookup ("phoebe_rv_plot_options_x_combobox")->gtk;
	GtkWidget *y_combobox				= gui_widget_lookup ("phoebe_rv_plot_options_y_combobox")->gtk;

	gint INDEX		= -1;
	gint INDEP = (gtk_combo_box_get_active (GTK_COMBO_BOX(x_combobox)) == 0) ? PHOEBE_COLUMN_PHASE : PHOEBE_COLUMN_HJD;

	gboolean plot_obs = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(obs_checkbutton));
	gboolean plot_syn = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(syn_checkbutton));

	INDEX = gtk_combo_box_get_active(GTK_COMBO_BOX(obs_combobox));

	if (INDEX < 0){
		INDEX = 0;
		gtk_combo_box_set_active (GTK_COMBO_BOX(obs_combobox), 0);
	}

	file = fopen(filename,"w");
	if (!file) {
		gui_notice ("File cannot be saved", "The file cannot be opened for output, aborting.");
	}
	else {
		switch (gtk_combo_box_get_active (GTK_COMBO_BOX(y_combobox))) {
			case 0:	gui_plot_rv_to_ascii_one_component (file, INDEX, INDEP, PHOEBE_COLUMN_PRIMARY_RV, plot_obs, plot_syn);
				break;
			case 1:	gui_plot_rv_to_ascii_one_component (file, INDEX, INDEP, PHOEBE_COLUMN_SECONDARY_RV, plot_obs, plot_syn);
				break;
			case 2:	gui_plot_rv_to_ascii_one_component (file, INDEX, INDEP, PHOEBE_COLUMN_PRIMARY_RV, plot_obs, plot_syn);
				gui_plot_rv_to_ascii_one_component (file, INDEX, INDEP, PHOEBE_COLUMN_SECONDARY_RV, plot_obs, plot_syn);
				break;
		}

		fclose(file);
	}

	return SUCCESS;
}

int gui_plot_eb_using_gnuplot ()
{
	gint status, i;

	gchar *filename;

	PHOEBE_vector *poscoy, *poscoz;

	WD_LCI_parameters *params;

	gchar *tmpdir;
	gchar ebname[255];
	gchar cname[255];
	gchar pname[255];
	gchar  line[255];

	gint ebfd, cfd, pfd;

	GtkWidget *plot_image 		= gui_widget_lookup ("phoebe_eb_plot_image")->gtk;
	GtkWidget *phase_spinbutton = gui_widget_lookup ("phoebe_star_shape_phase_spinbutton")->gtk;
	GError *err = NULL;

	double phase = gtk_spin_button_get_value (GTK_SPIN_BUTTON (phase_spinbutton));

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);

	params = phoebe_malloc (sizeof (*params));
	status = wd_lci_parameters_get (params, 5, 0);
	if (status != SUCCESS) {
		gui_notice ("Star shape plot", phoebe_error(status));
		return status;
	}

	/* 3D plotting is always done in phase space, regardless of the settings: */
	params->JDPHS = 2;

	filename = phoebe_resolve_relative_filename ("lcin.active");
	create_lci_file (filename, params);
	free (params);

	poscoy = phoebe_vector_new ();
	poscoz = phoebe_vector_new ();
	status = call_wd_to_get_pos_coordinates (poscoy, poscoz, phase);
	if (status != SUCCESS) {
		gui_notice ("Star shape plot", phoebe_error (status));
		return status;
	}

	sprintf(ebname, "%s/phoebe-eb-XXXXXX", tmpdir);
	ebfd = gui_tempfile (ebname);
	for (i=0;i<poscoy->dim;i++) {
		sprintf(line, "%lf\t%lf\n", poscoy->val[i], poscoz->val[i]) ;
		write(ebfd, line, strlen(line));
	}

	phoebe_vector_free (poscoy);
	phoebe_vector_free (poscoz);

	close(ebfd);

	sprintf(cname, "%s/phoebe-eb-XXXXXX", tmpdir);
	cfd = gui_tempfile (cname);

#ifdef PHOEBE_GUI_GNUPLOT_LIBGD
	sprintf(line, "set terminal png small size 694,458\n"); 			write(cfd, line, strlen(line));
#else
	sprintf(line, "set terminal png small picsize 694 458\n"); 			write(cfd, line, strlen(line));
#endif
	sprintf(line, "set mxtics 2\n"); 									write(cfd, line, strlen(line));
	sprintf(line, "set mytics 2\n"); 									write(cfd, line, strlen(line));
	sprintf(line, "set lmargin 5\n");									write(cfd, line, strlen(line));
	sprintf(line, "set tmargin 2\n");									write(cfd, line, strlen(line));
	sprintf(line, "set rmargin 2\n");									write(cfd, line, strlen(line));
	sprintf(line, "set bmargin 4\n");									write(cfd, line, strlen(line));

	sprintf(line, "set yrange [-0.8:0.8]\n"); 							write(cfd, line, strlen(line));
	sprintf(line, "set xrange [-1.3:1.3]\n"); 							write(cfd, line, strlen(line));

	sprintf(pname, "%s/phoebe-eb-plot-XXXXXX", tmpdir);
	pfd = gui_tempfile (pname);
	close(pfd);

	sprintf(line, "set output '%s'\n", pname);							write(cfd, line, strlen(line));

	sprintf(line, "plot  '%s' w d notitle\n", ebname);					write(cfd, line, strlen(line));

	close(cfd);


	gui_plot(cname);

	GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(pname, &err);
	if (err != NULL)
		phoebe_debug("Error in gdk_pixbuf_new_from_file(%s): (%d) %s", pname, err->code, err->message);
	gtk_image_set_from_pixbuf(GTK_IMAGE(plot_image), pixbuf);
	gdk_pixbuf_unref(pixbuf);

	//----------------

	remove(ebname);
	remove(cname);
	remove(pname);

	return SUCCESS;
}
