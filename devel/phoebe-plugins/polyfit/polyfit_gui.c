#include <phoebe/phoebe.h>

int phoebe_plugin_start ()
{
	printf ("polyfit plugin started.\n");
	return SUCCESS;
}

int phoebe_plugin_stop ()
{
	printf ("polyfit plugin stopped.\n");
	return SUCCESS;
}

