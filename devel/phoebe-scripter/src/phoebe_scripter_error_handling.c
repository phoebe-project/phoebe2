#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_error_handling.h"

char *phoebe_scripter_error (int code)
{
	/*
	 * This function takes the error code and translates it to a human-readable
	 * string.
	 */

	switch (code) {
		case ERROR_SCRIPTER_INVALID_VARIABLE:
			return "there is no such variable in the current symbol table.\n";
		case ERROR_SCRIPTER_ARGUMENT_NUMBER_MISMATCH:
			return "the passed number of arguments is invalid, aborting.\n";
		case ERROR_SCRIPTER_ARGUMENT_TYPE_MISMATCH:
			return "argument type is invalid, aborting.\n";
		case ERROR_SCRIPTER_COMMAND_DOES_NOT_EXIST:
			return "command not found, aborting.\n";
		case ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS:
			return "the operands to the binary operation are incompatible, aborting.\n";
		case ERROR_SCRIPTER_GNUPLOT_NOT_INSTALLED:
			return "the scripter was not compiled with gnuplot support.\n";
		default:
			return phoebe_error (code);
	}
}

int phoebe_notice (const char *fmt, ...)
	{
	va_list ap;
	int r;

	printf ("PHOEBE notice: ");
	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
	}

int phoebe_warning (const char *fmt, ...)
	{
	va_list ap;
	int r;

	printf ("PHOEBE warning: ");
	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
	}

int phoebe_fatal (const char *fmt, ...)
	{
	va_list ap;

	printf ("PHOEBE fatal: ");
	va_start (ap, fmt);
	vprintf (fmt, ap);
	va_end (ap);

	exit (EXIT_FAILURE);
	}

int phoebe_scripter_output (const char *fmt, ...)
{
	va_list ap;
	int r;
	int verbosity;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("scripter_verbosity_level"), &verbosity);
	if (verbosity == 0) return 0;

	printf ("PHOEBE scripter: ");
	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
}
