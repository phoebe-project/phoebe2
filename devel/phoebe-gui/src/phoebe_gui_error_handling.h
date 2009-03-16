#ifndef PHOEBE_GUI_ERROR_HANDLING_H
	#define PHOEBE_GUI_ERROR_HANDLING_H

#include <unistd.h>
#include <phoebe/phoebe.h>

typedef enum PHOEBE_gui_error_code {
	GUI_ERROR_GENERIC_ERROR = 10001,
	GUI_ERROR_NO_CURVE_MARKED_FOR_PLOTTING
} PHOEBE_gui_error_code;

char *phoebe_gui_error  (int code);

int   phoebe_gui_fatal   (const char *fmt, ...);
int   phoebe_gui_debug   (const char *fmt, ...);
int   phoebe_gui_output  (const char *fmt, ...);

#endif
