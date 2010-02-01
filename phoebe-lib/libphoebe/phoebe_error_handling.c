#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>

#include "phoebe_build_config.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
#include "phoebe_parameters.h"

char *phoebe_error (PHOEBE_error_code code)
{
	/**
	 * phoebe_error:
	 * @code:
	 *
	 * Takes the error code and translates it to a human-readable string.
	 *
	 * Returns:
	 *
	 */

	switch (code) {
		case SUCCESS:
			return "success.\n";
		case ERROR_SIGINT:
			return "interrupt received, aborting.\n";
		case ERROR_EXCEPTION_HANDLER_INVOKED:
			return "exception handler invoked, please report this!\n";
		case ERROR_HOME_ENV_NOT_DEFINED:
			return "environment variable $HOME is not defined. PHOEBE needs this variable\n              to install its configuration files. Please define it with e.g.:\n\n                export HOME=/home/user\n\n              and restart PHOEBE.\n";
		case ERROR_HOME_HAS_NO_PERMISSIONS:
			return "your home directory (defined by $HOME variable) doesn't exist or it doesn't\n              have proper permissions set (e.g. 755). Please correct this and restart\n              PHOEBE.\n";
		case ERROR_PHOEBE_CONFIG_ENTRY_INVALID_TYPE:
			return "invalid type encountered in phoebe_config_entry_add ().\n";
		case ERROR_PHOEBE_CONFIG_NOT_FOUND:
			return "PHOEBE configuration file not found.\n";
		case ERROR_PHOEBE_CONFIG_OPEN_FAILED:
			return "PHOEBE configuration file failed to open.\n";
		case ERROR_PHOEBE_CONFIG_LEGACY_FILE:
			return "configuration file pertains to a pre-0.30 version of PHOEBE.\n";
		case ERROR_PHOEBE_CONFIG_SUPPORTED_FILE:
			return "configuration file pertains to a compatible but older version of PHOEBE.\n";
		case ERROR_PHOEBE_CONFIG_INVALID_LINE:
			return "invalid line encountered in the configuration file, skipping.\n";
		case ERROR_PHOEBE_CONFIG_INVALID_KEYWORD:
			return "invalid keyword encountered in the configuration file, skipping.\n";
		case ERROR_PLUGINS_DIR_LOAD_FAILED:
			return "plugins directory failed to open, disabling plugins.\n";
		case ERROR_ATMCOF_NOT_FOUND:
			return "auxiliary file phoebe_atmcof.dat not found; please review your configuration.\n";
		case ERROR_ATMCOFPLANCK_NOT_FOUND:
			return "auxiliary file phoebe_atmcofplanck.dat not found; please review your configuration.\n";
		case ERROR_FILE_NOT_FOUND:
			return "the file doesn't exist.\n";
		case ERROR_FILE_NOT_REGULAR:
			return "the file is not a regular file.\n";
		case ERROR_FILE_NO_PERMISSIONS:
			return "you don't have permissions to access the file.\n";
		case ERROR_FILE_IS_INVALID:
			return "the file is invalid.\n";
		case ERROR_FILE_OPEN_FAILED:
			return "the file failed to open.\n";
		case ERROR_FILE_IS_EMPTY:
			return "the file is empty.\n";
		case ERROR_FILE_HAS_NO_DATA:
			return "the file contains no data, aborting.\n";
		case ERROR_DIRECTORY_PERMISSION_DENIED:
			return "cannot access directory: permission denied.\n";
		case ERROR_DIRECTORY_TOO_MANY_FILE_DESCRIPTORS:
			return "cannot access directory: too many file descriptors are in use.\n";
		case ERROR_DIRECTORY_TOO_MANY_OPEN_FILES:
			return "cannot access directory: too many open files.\n";
		case ERROR_DIRECTORY_NOT_FOUND:
			return "cannot access directory: directory not found.\n";
		case ERROR_DIRECTORY_INSUFFICIENT_MEMORY:
			return "cannot access directory: insufficient memory.\n";
		case ERROR_DIRECTORY_NOT_A_DIRECTORY:
			return "cannot access directory: argument is not a directory.\n";
		case ERROR_DIRECTORY_BAD_FILE_DESCRIPTOR:
			return "cannot access directory: bad file descriptor.\n";
		case ERROR_DIRECTORY_UNKNOWN_ERROR:
			return "cannot access directory: an unkown error occured. Please report this!\n";
		case ERROR_PARAMETER_NOT_INITIALIZED:
			return "parameter is not initialized, aborting.\n";
		case ERROR_PARAMETER_ALREADY_DECLARED:
			return "parameter has already been declared, aborting.\n";
		case ERROR_PARAMETER_OPTION_DOES_NOT_EXIST:
			return "parameter option does not exist, aborting.\n";
		case ERROR_PARAMETER_TABLE_NOT_INITIALIZED:
			return "parameter table is not initialized, aborting.\n";
		case ERROR_PARAMETER_OUT_OF_BOUNDS:
			return "parameter is out of bounds, aborting.\n";
		case ERROR_DATA_NOT_INITIALIZED:
			return "the variable to hold the data is not initialized, aborting.\n";
		case ERROR_DATA_INVALID_SIZE:
			return "passed dimension of the data variable is invalid, aborting.\n";
		case ERROR_VECTOR_NOT_INITIALIZED:
			return "array is not initialized, aborting.\n";
		case ERROR_VECTOR_ALREADY_ALLOCATED:
			return "array is already allocated, aborting.\n";
		case ERROR_VECTOR_IS_EMPTY:
			return "array is empty, aborting.\n";
		case ERROR_VECTOR_INVALID_DIMENSION:
			return "array dimension is invalid, aborting.\n";
		case ERROR_VECTOR_DIMENSIONS_MISMATCH:
			return "dimensions of both vectors do not match, aborting.\n";
		case ERROR_VECTOR_DIMENSION_NOT_THREE:
			return "the cross product can be evaluated only in 3D, aborting.\n";
		case ERROR_VECTOR_INVALID_LIMITS:
			return "the upper limit is smaller than the lower limit, aborting.\n";
		case ERROR_MATRIX_NOT_INITIALIZED:
			return "matrix is not initialized, aborting.\n";
		case ERROR_MATRIX_ALREADY_ALLOCATED:
			return "matrix is already allocated, aborting.\n";
		case ERROR_MATRIX_INVALID_DIMENSION:
			return "matrix dimensions are invalid, aborting.\n";
		case ERROR_HIST_NOT_INITIALIZED:
			return "histogram is not initialized, aborting.\n";
		case ERROR_HIST_NOT_ALLOCATED:
			return "histogram is not allocated, aborting.\n";
		case ERROR_HIST_ALREADY_ALLOCATED:
			return "histogram is already allocated, aborting.\n";
		case ERROR_HIST_INVALID_DIMENSION:
			return "histogram dimension is invalid, aborting.\n";
		case ERROR_HIST_INVALID_RANGES:
			return "histogram ranges are invalid, aborting.\n";
		case ERROR_HIST_OUT_OF_RANGE:
			return "the passed value is out of histogram range, aborting.\n";
		case ERROR_HIST_NO_OVERLAP:
			return "input and output histogram ranges do not overlap, aborting.\n";
		case ERROR_ARRAY_NOT_INITIALIZED:
			return "array is not initialized, aborting.\n";
		case ERROR_ARRAY_ALREADY_ALLOCATED:
			return "array is already allocated, aborting.\n";
		case ERROR_ARRAY_INVALID_DIMENSION:
			return "array dimension is invalid, aborting.\n";
		case ERROR_COLUMN_INVALID:
			return "column entry is invalid, aborting.\n";
		case ERROR_CURVE_NOT_INITIALIZED:
			return "curve is not initialized, aborting.\n";
		case ERROR_CURVE_ALREADY_ALLOCATED:
			return "curve is already allocated, aborting.\n";
		case ERROR_CURVE_INVALID_DIMENSION:
			return "curve dimension is invalid, aborting.\n";
		case ERROR_COMPUTED_PARAMS_NOT_INITIALIZED:
			return "the structure for computed parameters not initialized, aborting.\n";
		case ERROR_SPECTRUM_NOT_INITIALIZED:
			return "spectrum is not initialized, aborting.\n";
		case ERROR_SPECTRUM_NOT_ALLOCATED:
			return "spectrum is not allocated, aborting.\n";
		case ERROR_SPECTRUM_ALREADY_ALLOCATED:
			return "spectrum is already allocated, aborting.\n";
		case ERROR_SPECTRUM_INVALID_DIMENSION:
			return "spectrum dimension is invalid, aborting.\n";
		case ERROR_SPECTRUM_NOT_IN_REPOSITORY:
			return "spectrum not found in the repository, aborting.\n";
		case ERROR_SPECTRUM_INVALID_SAMPLING:
			return "spectrum sampling resolution is invalid, aborting.\n";
		case ERROR_SPECTRUM_INVALID_RESOLUTION:
			return "spectrum resolution is invalid, aborting.\n";
		case ERROR_SPECTRUM_INVALID_RANGE:
			return "spectrum range is invalid, aborting.\n";
		case ERROR_SPECTRUM_UNKNOWN_DISPERSION:
			return "cannot operate on a spectrum of unknown dispersion, aborting.\n";
		case ERROR_INDEX_OUT_OF_RANGE:
			return "array index out of range, aborting.\n";
		case ERROR_INVALID_TYPE:
			return "invalid type encountered, aborting.\n";
		case ERROR_INVALID_INDEP:
			return "independent variable not set or invalid, aborting.\n";
		case ERROR_INVALID_DEP:
			return "dependent variable not set or invalid, aborting.\n";
		case ERROR_INVALID_WEIGHT:
			return "individual weighting scheme not set or invalid, aborting.\n";
		case ERROR_INVALID_DATA:
			return "data file cannot be read or is invalid, aborting.\n";
		case ERROR_INVALID_MODEL:
			return "morphological constraints not set or invalid, aborting.\n";
		case ERROR_INVALID_LDLAW:
			return "limb darkening law not set or invalid, aborting.\n";
		case ERROR_INVALID_EL3_UNITS:
			return "third light units not set or invalid, aborting.\n";
		case ERROR_INVALID_EL3_VALUE:
			return "third light contribution exceeds the total light, aborting.\n";
		case ERROR_INVALID_MAIN_INDEP:
			return "main independent quantity not set or invalid, aborting.\n";
		case ERROR_INVALID_HEADER:
			return "the header of parameter file is invalid, aborting.\n";
		case ERROR_INVALID_WAVELENGTH_INTERVAL:
			return "the passed wavelength interval is invalid, aborting.\n";
		case ERROR_INVALID_SAMPLING_POWER:
			return "the spectrum's sampling power is invalid, aborting.\n";
		case ERROR_INVALID_PHASE_INTERVAL:
			return "phase interval is invalid (phmax <= phmin), aborting.\n";
		case ERROR_INVALID_NORMAL_MAG: 
			return "overflow error for luminosity, please lower the value of the normal magnitude.\n"; 
		case ERROR_UNINITIALIZED_CURVE:
			return "the curve you are trying to compute is not initialized.\n";
		case ERROR_STAR_SURFACE_NOT_INITIALIZED:
			return "star surface is not initialized, aborting.\n";
		case ERROR_STAR_SURFACE_NOT_ALLOCATED:
			return "star surface is not allocated, aborting.\n";
		case ERROR_STAR_SURFACE_ALREADY_ALLOCATED:
			return "star surface is already allocated, aborting.\n";
		case ERROR_STAR_SURFACE_INVALID_DIMENSION:
			return "star surface dimension is invalid, aborting.\n";
		case ERROR_UNSUPPORTED_MPAGE:
			return "unsupported calculation type, aborting.\n";
		case ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED:
			return "minimizer feedback structure not initialized, aborting.\n";
		case ERROR_MINIMIZER_FEEDBACK_ALREADY_ALLOCATED:
			return "minimizer feedback structure already allocated, aborting.\n";
		case ERROR_MINIMIZER_NO_CURVES:
			return "data curves not defined, nothing to be done.\n";
		case ERROR_MINIMIZER_NO_PARAMS:
			return "no parameters marked for adjustment, nothing to be done.\n";
		case ERROR_MINIMIZER_INVALID_FILE:
			return "observational data files you supplied don't seem to exist, aborting.\n";
		case ERROR_MINIMIZER_HLA_REQUEST_NOT_SANE:
			return "passband levels marked for computation and adjustment at the same time, aborting.\n";
		case ERROR_MINIMIZER_VGA_REQUEST_NOT_SANE:
			return "center-of-mass velocity marked for computation and adjustment at the same time, aborting.\n";
		case ERROR_MINIMIZER_DPDT_REQUEST_NOT_SANE:
			return "a non-null dP/dt value requires computation to be done in time-space, but phase-space is selected; aborting.\n";
		case ERROR_NONSENSE_DATA_REQUEST:
			return "invalid data conversion request, aborting.\n";
		case ERROR_NEGATIVE_STANDARD_DEVIATION:
			return "negative standard deviation value encountered, aborting.\n";
		case ERROR_CHI2_INVALID_TYPE:
			return "type passed to the chi2 function is invalid, aborting.\n";
		case ERROR_CHI2_INVALID_DATA:
			return "data passed to the chi2 function are invalid, aborting.\n";
		case ERROR_CHI2_DIFFERENT_SIZES:
			return "sizes of observational and synthetic data differ, aborting.\n";
		case ERROR_MS_TEFF1_OUT_OF_RANGE:
			return "primary temperature out-of-range for the main-sequence calculator, aborting.\n";
		case ERROR_MS_TEFF2_OUT_OF_RANGE:
			return "secondary temperature out-of-range for the main-sequence calculator, aborting.\n";
		case ERROR_PLOT_DIMENSION_MISMATCH:
			return "data arrays you are trying to plot have different dimensions, aborting.\n";
		case ERROR_PLOT_FIFO_PERMISSION_DENIED:
			return "you don't have permissions to access the selected temporary directory, aborting.\n";
		case ERROR_PLOT_FIFO_FILE_EXISTS:
			return "file assigned to the gnuplot FIFO already exists, aborting.\n";
		case ERROR_PLOT_FIFO_FAILURE:
			return "undocumented mkfifo () failure, please report this!\n";
		case ERROR_PLOT_TEMP_FILE_EXISTS:
			return "temporary file assigned to the gnuplot already exists, aborting.\n";
		case ERROR_PLOT_TEMP_FAILURE:
			return "undocumented mkstemp () failure, please report this!\n";
		case ERROR_GSL_NOT_INSTALLED:
			return "GSL library that is required for that operation is missing, aborting.\n";
		case ERROR_PLOT_TEMP_MALFORMED_FILENAME:
			return "temporary filename cannot be created, aborting.\n";
		case ERROR_LD_LAW_INVALID:
			return "LD law is not set or is invalid, aborting.\n";
		case ERROR_LD_TABLES_MISSING:
			return "LD coefficient tables cannot be found, aborting.\n";
		case ERROR_LD_PARAMS_OUT_OF_RANGE:
			return "parameter values for LD table lookup out of range, aborting.\n";
		case ERROR_LD_TABLE_NOT_INITIALIZED:
			return "the passed LD table is not initialized, aborting.\n";
		case ERROR_LD_TABLE_ALREADY_ALLOCATED:
			return "the passed LD table is already allocated, aborting.\n";
		case ERROR_LD_TABLE_INVALID_DIMENSION:
			return "the requested dimension for the LD table is invalid, aborting.\n";
		case ERROR_LD_TABLE_PASSBAND_NOT_SPECIFIED:
			return "the passed LD table does not specify the passband correctly, aborting.\n";
		case ERROR_LD_TABLE_PASSBAND_NOT_FOUND:
			return "the passband of the passed LD table is not found. aborting.\n";
		case ERROR_CINDEX_INVALID_TYPE:
			return "the type passed to the Teff (B-V) function is invalid, aborting.\n";
		case ERROR_PARAMETER_INVALID_LIMITS:
			return "lower limit is larger than the upper limit, aborting.\n";
		case ERROR_PARAMETER_INDEX_OUT_OF_RANGE:
			return "parameter index out of range, aborting.\n";
		case ERROR_PARAMETER_KIND_NOT_MENU:
			return "qualifier is not a menu, aborting.\n";
		case ERROR_PARAMETER_MENU_ITEM_OUT_OF_RANGE:
			return "parameter value is invalid (item out of range), aborting.\n";
		case ERROR_PASSBAND_TF_FILE_NOT_FOUND:
			return "passband transmission function not found, skipping.\n";
		case ERROR_PASSBAND_INVALID:
			return "the passband is not set or is invalid, aborting.\n";
		case ERROR_QUALIFIER_STRING_IS_NULL:
			return "the string passed for the qualifier is null, aborting.\n";
		case ERROR_QUALIFIER_STRING_MALFORMED:
			return "the string passed as a qualifier element is malformed, aborting.\n";
		case ERROR_QUALIFIER_NOT_FOUND:
			return "the passed qualifier doesn't exist, aborting.\n";
		case ERROR_QUALIFIER_NOT_ADJUSTABLE:
			return "the passed qualifier is not adjustable, aborting.\n";
		case ERROR_QUALIFIER_NOT_ARRAY:
			return "the passed qualifier is not an array, aborting.\n";
		case ERROR_DESCRIPTION_NOT_FOUND:
			return "the passed description doesn't exist, aborting.\n";
		case ERROR_ARG_NOT_INT:
			return "argument of type integer expected, aborting.\n";
		case ERROR_ARG_NOT_BOOL:
			return "argument of type boolean expected, aborting.\n";
		case ERROR_ARG_NOT_DOUBLE:
			return "argument of type double expected, aborting.\n";
		case ERROR_ARG_NOT_STRING:
			return "argument of type string expected, aborting.\n";
		case ERROR_ARG_NOT_INT_ARRAY:
			return "argument of type int array expected, aborting.\n";
		case ERROR_ARG_NOT_BOOL_ARRAY:
			return "argument of type boolean array expected, aborting.\n";
		case ERROR_ARG_NOT_DOUBLE_ARRAY:
			return "argument of type double array expected, aborting.\n";
		case ERROR_ARG_NOT_STRING_ARRAY:
			return "argument of type string array expected, aborting.\n";
		case ERROR_ARG_NUMBER_MISMATCH:
			return "invalid number of arguments passed, aborting.\n";
		case ERROR_RELEASE_INDEX_OUT_OF_RANGE:
			return "parameter index you are trying to release is out of range, aborting.\n";
		case ERROR_SPECTRA_REPOSITORY_NOT_FOUND:
			return "spectra repository not found, aborting.\n";
		case ERROR_SPECTRA_REPOSITORY_INVALID_NAME:
			return "the name of the spectra repository is invalid, aborting.\n";
		case ERROR_SPECTRA_REPOSITORY_EMPTY:
			return "the spectra repository is empty, aborting.\n";
		case ERROR_SPECTRA_REPOSITORY_NOT_DIRECTORY:
			return "spectra repository must be a directory, aborting.\n";
		case ERROR_SPECTRA_DIMENSION_MISMATCH:
			return "the two spectra don't have the same dimension, aborting.\n";
		case ERROR_SPECTRA_NO_OVERLAP:
			return "attempting to resample the spectrum to the non-overlapping region, aborting.\n";
		case ERROR_SPECTRA_NOT_ALIGNED:
			return "spectra are not aligned in wavelength, aborting.\n";
		case ERROR_BROADENING_INADEQUATE_ACCURACY:
			return "rotational broadening cannot be accurately applied for |vsini| < 5km/s, aborting.\n";
		case ERROR_WD_LCI_PARAMETERS_NOT_INITIALIZED:
			return "LCI parameters not initialized, aborting.\n";
		case ERROR_DC_TOO_MANY_SPOTS_TBA:
			return "too many spots are marked for adjustment (up to two are allowed), aborting.\n";
		case ERROR_DC_TOO_MANY_RVS:
			return "too many RV curves for WD's DC fit, aborting.\n";
		case ERROR_SPOT_NOT_INITIALIZED:
			return "the spot is not initialized, aborting.\n";
		case ERROR_SPOT_INVALID_SOURCE:
			return "spot source star invalid (should be 1 or 2), aborting.\n";
		case ERROR_SPOT_INVALID_COLATITUDE:
			return "spot latitude is invalid (0 < colat < pi), aborting.\n";
		case ERROR_SPOT_INVALID_LONGITUDE:
			return "spot latitude is invalid (0 < lon < 2pi), aborting.\n";
		case ERROR_SPOT_INVALID_RADIUS:
			return "spot angular radius is invalid (0 < rad < pi), aborting.\n";
		case ERROR_SPOT_INVALID_TEMPERATURE:
			return "spot temperature factor is invalid (0 < temp < 100), aborting.\n";
		case ERROR_SPOT_INVALID_WD_NUMBER:
			return "spot DC designation is invalid (should be 1 or 2), aborting.\n";
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_error () by code %d, please report this!\n", code);
			return "exception handler invoked.\n";
	}
}

int phoebe_lib_error (const char *fmt, ...)
{
	va_list ap;
	int r;

	printf ("PHOEBE-lib error: ");
	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
}

int phoebe_lib_warning (const char *fmt, ...)
{
	va_list ap;
	int r;

	printf ("PHOEBE-lib warning: ");
	va_start (ap, fmt);
	r = vprintf (fmt, ap);
	va_end (ap);

	return r;
}

int phoebe_debug (const char *fmt, ...)
{
	/**
	 * phoebe_debug:
	 * @fmt: printf-compatible format
	 * @...: arguments to @fmt
	 *
	 * Writes the message to stdout in case PHOEBE was compiled
	 * with --enable-debug switch, otherwise it just returns control to the
	 * main program.
	 *
	 * Returns: number of characters output, or -1 on failure.
	 */
	
	int r = -1;
	
#ifdef PHOEBE_DEBUG_SUPPORT
	va_list ap;
		printf ("PHOEBE debug: ");
		va_start (ap, fmt);
		r = vprintf (fmt, ap);
		va_end (ap);
#endif
	
	return r;
}

void *phoebe_malloc (size_t size)
{
    /**
     * phoebe_malloc:
     *
     * Allocates the space with the check whether the memory was exhausted.
     */
	register void *value = malloc (size);
	if (value == 0)
    {
		printf ("Virtual memory exhauseted.\n");
		exit (-1);
    }
	return value;
}

void *phoebe_realloc (void *ptr, size_t size)
{
	/**
	 * phoebe_realloc:
	 *
	 * Reallocates the space with the check whether the memory
	 * was exhausted. If the size equals 0, realloc calls free() on ptr.
	 */

	register void *value = realloc (ptr, size);
	if ( (value == 0) && (size != 0) )
    {
		printf ("Virtual memory exhauseted.\n");
		exit (-1);
    }
	return value;
}

bool hla_request_is_sane ()
{
	/**
	 * hla_request_is_sane:
	 *
	 * Tests the HLA request. If the passband levels are marked
	 * for adjustment and for computation at the same time, FALSE is returned.
	 * Otherwise, TRUE is returned.
	 */

	bool hla_adj_state, hla_comp_state;

	phoebe_parameter_get_tba   (phoebe_parameter_lookup ("phoebe_hla"), &hla_adj_state);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_compute_hla_switch"), &hla_comp_state);

	if (hla_adj_state && hla_comp_state)
		return FALSE;
	else
		return TRUE;
}

bool vga_request_is_sane ()
{
	/**
	 * vga_request_is_sane:
	 *
	 * Tests the VGA request. If the center-of-mass velocity is
	 * marked for adjustment and for computation at the same time, FALSE is
	 * returned. Otherwise, TRUE is returned.
	 */

	bool vga_adj_state, vga_comp_state;

	phoebe_parameter_get_tba   (phoebe_parameter_lookup ("phoebe_vga"), &vga_adj_state);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_compute_vga_switch"), &vga_comp_state);

	if (vga_adj_state && vga_comp_state)
		return FALSE;
	else
		return TRUE;
}

bool dpdt_request_is_sane ()
{
	/**
	 * dpdt_request_is_sane:
	 *
	 * Tests whether time-dependent parameter, dP/dt, is being
	 * used in time (HJD) space. If so, TRUE is returned, and if not, FALSE
	 * is returned.
	 */

	double dpdt;
	const char *indep;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dpdt"),  &dpdt);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &indep);

	if (strcmp (indep, "Phase") && fabs (dpdt) > PHOEBE_NUMERICAL_ACCURACY)
		return FALSE;
	else
		return TRUE;
}
