#ifndef PHOEBE_SCRIPTER_ERROR_HANDLING_H
	#define PHOEBE_SCRIPTER_ERROR_HANDLING_H 1

#include <unistd.h>
#include <phoebe/phoebe.h>

typedef enum PHOEBE_scripter_error_code {
	ERROR_SCRIPTER_ARGUMENT_NUMBER_MISMATCH = 1001,
	ERROR_SCRIPTER_ARGUMENT_TYPE_MISMATCH,
	ERROR_SCRIPTER_COMMAND_DOES_NOT_EXIST,
	ERROR_SCRIPTER_INVALID_VARIABLE,
	ERROR_SCRIPTER_INCOMPATIBLE_OPERANDS,
	ERROR_SCRIPTER_GNUPLOT_NOT_INSTALLED
} PHOEBE_scripter_error_code;

char *phoebe_scripter_error  (int code);

int   phoebe_notice          (const char *fmt, ...);
int   phoebe_warning         (const char *fmt, ...);
int   phoebe_fatal           (const char *fmt, ...);
int   phoebe_scripter_output (const char *fmt, ...);

#endif
