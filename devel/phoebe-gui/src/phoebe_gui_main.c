#ifdef HAVE_CONFIG_H
#  include <phoebe_gui_build_config.h>
#endif

#include <phoebe/phoebe.h>

#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_main.h"

int main (int argc, char *argv[])
{
    gtk_set_locale();
    gtk_init(&argc, &argv);
    glade_init();

    phoebe_gui = glade_xml_new("phoebe.glade", NULL, NULL);
    glade_xml_signal_autoconnect (phoebe_gui);

    phoebe_window = glade_xml_get_widget(phoebe_gui, "phoebe_window");
    phoebe_settings_window = glade_xml_get_widget(phoebe_gui, "phoebe_settings_window");

    gtk_widget_show(phoebe_window);

    gtk_main();

	return SUCCESS;
}
