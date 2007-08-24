#include <phoebe/phoebe.h>

#include "phoebe_gui_types.h"
#include "phoebe_gui_base.h"

#include "phoebe_gui_treeviews.h"

int phoebe_gui_init ()
{
	gui_init_widgets ();

	return SUCCESS;
}

int phoebe_gui_quit ()
{
	gui_free_widgets ();

	return SUCCESS;
}
