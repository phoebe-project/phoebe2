#include <stdio.h>

#include <gtk/gtk.h>
#include <glade/glade.h>

#include <phoebe/phoebe.h>

typedef struct GUI_plot_layout {
	int lmargin;           /* left plot margin in pixels                      */
	int rmargin;           /* right plot margin in pixels                     */
	int tmargin;           /* top plot margin in pixels                       */
	int bmargin;           /* bottom plot margin in pixels                    */
	int xmargin;           /* x-spacing between the plot border and the graph */
	int ymargin;           /* y-spacing between the plot border and the graph */
	int label_lmargin;     /* spacing between the frame and the y-axis label  */
	int label_rmargin;     /* spacing between the y-axis label and the graph  */
	int x_tick_length;     /* major x-tick length in pixels                   */
	int y_tick_length;     /* major y-tick length in pixels                   */
} GUI_plot_layout;

GUI_plot_layout *gui_plot_layout_new  ();
int              gui_plot_layout_free (GUI_plot_layout *layout);

typedef enum GUI_plot_type {
	GUI_PLOT_LC,
	GUI_PLOT_RV,
	GUI_PLOT_MESH,
	GUI_PLOT_UNDEFINED
} GUI_plot_type;

typedef struct GUI_plot_request {
	bool          plot_obs;
	bool          plot_syn;
	bool          data_changed;  /* Indicates whether data points have been deleted or undeleted */
	char         *obscolor;      /* Hexadecimal color code (with a leading #) */
	char         *syncolor;      /* Hexadecimal color code (with a leading #) */
	double        offset;
	double        phase;         /* For mesh plots */
	char         *filename;
	PHOEBE_curve *raw;
	PHOEBE_curve *query;
	PHOEBE_curve *model;         /* For LC and RV plots                       */
	PHOEBE_star_surface *mesh;   /* For mesh (plane-of-sky projection) plots  */
} GUI_plot_request;

typedef struct GUI_plot_data {
	GUI_plot_layout   *layout;       /* Plot layout (margins, ticks, ...)      */
	GUI_plot_request  *request;      /* Structure with all data and properties */
	GUI_plot_type      ptype;        /* Plot type (LC, RV or mesh)             */
	GtkWidget         *container;    /* Widget container                       */
	cairo_t           *canvas;       /* Cairo canvas                           */
	double             width;        /* Graph width in pixels                  */
	double             height;       /* Graph height in pixels                 */
	int                objno;        /* Number of objects for plotting         */
	bool               alias;        /* Should data be aliased?                */
	bool               residuals;    /* Should residuals be plotted?           */
	const char        *x_request;    /* Requested x-coordinate                 */
	const char        *y_request;    /* Requested y-coordinate                 */
	double             x_ll;         /* Lower plotting limit for the x-axis    */
	double             x_ul;         /* Upper plotting limit for the x-axis    */
	double             y_ll;         /* Lower plotting limit for the y-axis    */
	double             y_ul;         /* Upper plotting limit for the y-axis    */
	double             x_min;        /* Minimum x value in the query dataset   */
	double             x_max;        /* Maximum x value in the query dataset   */
	double             y_min;        /* Minimum y value in the query dataset   */
	double             y_max;        /* Maximum y value in the query dataset   */
	int                vertices;     /* Number of vertices for synthetic plots */
	bool               coarse_grid;  /* Should a coarse grid be plotted?       */
	bool               fine_grid;    /* Should a fine grid be plotted?         */
	GtkWidget         *x_widget;     /* Widget to be connected to x-coordinate */
	GtkWidget         *y_widget;     /* Widget to be connected to y-coordinate */
	GtkWidget         *cp_widget;    /* Widget to be connected to closest psb. */
	GtkWidget         *cx_widget;    /* Widget to be connected to closest x pt */
	GtkWidget         *cy_widget;    /* Widget to be connected to closest y pt */
	double             leftmargin;   /* The value of left margin               */
	bool               select_zoom;  /* Indicates whether a rectangle to zoom in is being drawn */
	double             select_x;     /* Window x value at which zoom started   */
	double             select_y;     /* Window y value at which zoom started   */
	double             x_left;       /* Current left x value                   */
	double             x_right;      /* Current right x value                  */
	double             y_top;        /* Current top y value                    */
	double             y_bottom;     /* Current bottom y value                 */
	bool               block_signal; /* Whether the row-changed signal should be blocked */
} GUI_plot_data;

GUI_plot_data *gui_plot_data_new ();
int            gui_plot_data_free ();

/* Signal callbacks pertinent to plotting: */

gboolean on_plot_area_expose_event          (GtkWidget *widget, GdkEventExpose *event, gpointer user_data);
gboolean on_plot_area_toggle_delete_button_clicked (GtkMenuItem *item, gpointer user_data);
void     on_plot_button_clicked             (GtkButton *button, gpointer user_data);
void     on_lc_plot_treeview_row_changed    (GtkTreeModel *tree_model, GtkTreePath *path, GtkTreeIter *iter, gpointer user_data);
void     on_rv_plot_treeview_row_changed    (GtkTreeModel *tree_model, GtkTreePath *path, GtkTreeIter *iter, gpointer user_data);
void     on_plot_treeview_row_deleted       (GtkTreeModel *model, GtkTreePath *path, gpointer user_data);

/* Plot functions */

int  gui_plot_area_init    (GtkWidget *area, GtkWidget *button);
int  gui_plot_area_draw    (GUI_plot_data *data, FILE *redirect);
int  gui_plot_area_refresh (GUI_plot_data *data);
void gui_plot_xticks       (GUI_plot_data *data);
void gui_plot_yticks       (GUI_plot_data *data);
void gui_plot_clear_canvas (GUI_plot_data *data);
