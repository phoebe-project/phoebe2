#ifdef HAVE_CONFIG_H
#  include "phoebe_gui_build_config.h"
#endif

#include <stdlib.h>
#include <phoebe/phoebe.h>

#include "phoebe_gui_base.h"
#include "phoebe_gui_callbacks.h"
#include "phoebe_gui_main.h"

int main (int argc, char *argv[])
{
	gtk_set_locale ();
	gtk_init (&argc, &argv);
	glade_init ();

	phoebe_init ();

	phoebe_gui_init ();
	gtk_main ();
	phoebe_gui_quit ();

	phoebe_quit ();

	return SUCCESS;
}
