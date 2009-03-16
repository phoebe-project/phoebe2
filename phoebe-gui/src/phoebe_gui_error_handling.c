#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>

#include <phoebe/phoebe.h>

#include "phoebe_gui_error_handling.h"

char *phoebe_gui_error (int code)
{
	/*
	 * This function takes the error code and translates it to a human-readable
	 * string.
	 */

	switch (code) {
		case GUI_ERROR_GENERIC_ERROR:
			return "unknown error occured in the GUI.\n";
		case GUI_ERROR_NO_CURVE_MARKED_FOR_PLOTTING:
			return "no curve is selected for plotting.\n";
		default:
			return phoebe_error (code);
	}
}

int phoebe_gui_fatal (const char *fmt, ...)
{
	va_list ap;

	printf ("PHOEBE GUI fatal: ");
	va_start (ap, fmt);
	vprintf (fmt, ap);
	va_end (ap);

	exit (EXIT_FAILURE);
}

int phoebe_gui_debug (const char *fmt, ...)
{
	va_list ap;
	int r;
	int verbosity;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("gui_verbosity_level"), &verbosity);
	if (verbosity == 0) return 0;

	printf ("GUI debug: ");
	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
}

int phoebe_gui_output (const char *fmt, ...)
{
	va_list ap;

	printf ("GUI: ");
	va_start (ap, fmt);
	vprintf (fmt, ap);
	va_end (ap);

	return SUCCESS;
}
