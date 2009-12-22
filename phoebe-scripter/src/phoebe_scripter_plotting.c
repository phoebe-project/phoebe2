#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <phoebe/phoebe.h>

#include "phoebe_scripter_build_config.h"
#include "phoebe_scripter_error_handling.h"
#include "phoebe_scripter_plotting.h"

int plot_using_gnuplot (int dim, bool reverse_y, PHOEBE_vector **indep, PHOEBE_vector **dep, scripter_plot_properties *props)
{
	/**
	 * plot_using_gnuplot:
	 * @dim:       the number of (indep, dep) pairs to plot
	 * @reverse_y: should the y axis of the plot be reversed
	 * @indep:     a vector of independent variable arrays
	 * @dep:       a vector of dependent variable arrays
	 * @props:     plot properties - color and line type of the plot
	 *
	 * Plots the data through a system call to gnuplot.
	 *
	 * Returns: #PHOEBE_scripter_error_code.
	 */

#ifdef PHOEBE_GNUPLOT_SUPPORT
	FILE *command;
	int i, j, status, error;

	char buffer[1024];
	char command_line[255];
	char *tmpdir;

	for (i = 0; i < dim; i++)
		if (indep[i]->dim != dep[i]->dim)
			return ERROR_PLOT_DIMENSION_MISMATCH;

#ifndef __MINGW32__
	char **temp_files = phoebe_malloc (dim * sizeof (*temp_files));
	FILE **data_files = phoebe_malloc (dim * sizeof (*data_files));

	for (i = 0; i < dim; i++) {
		temp_files[i] = phoebe_malloc (255 * sizeof (**temp_files));
		phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);
		sprintf (temp_files[i], "%s/phoebe_tmp_XXXXXX", tmpdir);
		status = mkstemp (temp_files[i]);
		error = errno;

		if (status == -1)
			switch (error) {
				case EEXIST: return ERROR_PLOT_TEMP_FILE_EXISTS;
				case EINVAL: return ERROR_PLOT_TEMP_MALFORMED_FILENAME;
				default:     return ERROR_PLOT_TEMP_FAILURE;
			}
		unlink (temp_files[i]);
		status = mkfifo (temp_files[i], S_IRUSR | S_IWUSR);
		error = errno;

		if (status == -1)
			switch (error) {
				case EACCES:
					for (j = 0; j < i; j++) unlink (temp_files[j]);
					return ERROR_PLOT_FIFO_PERMISSION_DENIED;
				case EEXIST:
					for (j = 0; j < i; j++) unlink (temp_files[j]);
					return ERROR_PLOT_FIFO_FILE_EXISTS;
				default:
					for (j = 0; j < i; j++) unlink (temp_files[j]);
					return ERROR_PLOT_FIFO_FAILURE;
			}
	}
#endif

	sprintf (buffer, "plot ");
	for (i = 0; i < dim; i++) {
		if (props[i].lines)
#ifdef __MINGW32__
			sprintf (command_line, "\"%s\" with steps lt %d", "-", props[i].ltype);
#else
			sprintf (command_line, "\"%s\" with steps lt %d", temp_files[i], props[i].ltype);
#endif
		else
#ifdef __MINGW32__
			sprintf (command_line, "\"%s\" with points lt %d pt %d", "-", props[i].ltype, props[i].ptype);
#else
			sprintf (command_line, "\"%s\" with points lt %d pt %d", temp_files[i], props[i].ltype, props[i].ptype);
#endif

		strcat  (buffer, command_line);
		if (i < dim-1) strcat (buffer, ", "); else strcat (buffer, "\n");
	}

#ifdef __MINGW32__
	command = popen ("pgnuplot -persist", "w");
#else
	command = popen ("gnuplot -persist", "w");
#endif
	fprintf (command, "unset key\n");
	fprintf (command, "set offsets 0.05, 0.05, 0, 0\n");
	if (reverse_y) fprintf (command, "set yrange [] reverse\n");
	fprintf (command, "%s", buffer);
	fflush (command);

	for (i = 0; i < dim; i++) {
#ifdef __MINGW32__
		for (j = 0; j < indep[i]->dim; j++) {
			fprintf (command, "%12.12lf\t%12.12lf\n", indep[i]->val[j], dep[i]->val[j]);
		}
		fprintf(command, "e\n");
#else
		data_files[i] = fopen (temp_files[i], "w");
		for (j = 0; j < indep[i]->dim; j++)
			fprintf (data_files[i], "%12.12lf\t%12.12lf\n", indep[i]->val[j], dep[i]->val[j]);
		fclose (data_files[i]);
#endif
	}

	pclose (command);

#ifndef __MINGW32__
	for (i = 0; i < dim; i++) {
		unlink (temp_files[i]);
		free (temp_files[i]);
	}
	free (temp_files);
	free (data_files);
#endif

	return SUCCESS;
#endif

	return ERROR_SCRIPTER_GNUPLOT_NOT_INSTALLED;
}
