GtkWidget *phoebe_window;

/* At the moment, I don't see a better way to obtain this reference from, say, 
 * phoebe_gui_callbacks.c... fix if possible. */
GtkListStore *lc_curves_model;
GtkListStore *rv_curves_model;
