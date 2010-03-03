#ifndef PHOEBE_TYPES_H
	#define PHOEBE_TYPES_H

#ifndef FALSE
	#define FALSE 0
#endif
#ifndef TRUE
	#define TRUE  1
#endif

typedef enum bool {
	NO,
	YES
} bool;

/**
 * PHOEBE_type:
 * @TYPE_INT:                The integer type.
 * @TYPE_BOOL:               The boolean type.
 * @TYPE_DOUBLE:             The double precision floating-point type.
 * @TYPE_STRING:             The string type.
 * @TYPE_INT_ARRAY:          Array of integers.
 * @TYPE_BOOL_ARRAY:         Array of booleans.
 * @TYPE_DOUBLE_ARRAY:       Array of doubles.
 * @TYPE_STRING_ARRAY:       Array of strings.
 * @TYPE_CURVE:              The #PHOEBE_curve type.
 * @TYPE_SPECTRUM:           The #PHOEBE_spectrum type.
 * @TYPE_MINIMIZER_FEEDBACK: The #PHOEBE_minimizer_feedback type.
 * @TYPE_ANY:                Can be any #PHOEBE_type.
 *
 * Various data types used in Phoebe.
 */

typedef enum PHOEBE_type {
	TYPE_INT,
	TYPE_BOOL,
	TYPE_DOUBLE,
	TYPE_STRING,
	TYPE_INT_ARRAY,
	TYPE_BOOL_ARRAY,
	TYPE_DOUBLE_ARRAY,
	TYPE_STRING_ARRAY,
	TYPE_CURVE,
	TYPE_SPECTRUM,
	TYPE_MINIMIZER_FEEDBACK,
	TYPE_ANY
} PHOEBE_type;

char *phoebe_type_get_name (PHOEBE_type type);

/* **************************************************************************** */

/**
 * PHOEBE_vector:
 * @dim: Vector dimension.
 * @val: An array of vector values.
 */
typedef struct PHOEBE_vector {
	int dim;
	double *val;
} PHOEBE_vector;

PHOEBE_vector *phoebe_vector_new                ();
PHOEBE_vector *phoebe_vector_new_from_qualifier (char *qualifier);
PHOEBE_vector *phoebe_vector_new_from_column    (char *filename, int col);
PHOEBE_vector *phoebe_vector_new_from_range     (int dim, double start, double end);
PHOEBE_vector *phoebe_vector_duplicate          (PHOEBE_vector *vec);
int            phoebe_vector_alloc              (PHOEBE_vector *vec, int dimension);
int            phoebe_vector_realloc            (PHOEBE_vector *vec, int dimension);
int            phoebe_vector_pad                (PHOEBE_vector *vec, double value);
int            phoebe_vector_free               (PHOEBE_vector *vec);

int            phoebe_vector_add                (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_subtract           (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_multiply           (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_divide             (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_raise              (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_offset             (PHOEBE_vector *vec, double offset);
int            phoebe_vector_sum                (PHOEBE_vector *vec, double *sum);
int            phoebe_vector_mean               (PHOEBE_vector *vec, double *mean);
int            phoebe_vector_median             (PHOEBE_vector *vec, double *median);
int            phoebe_vector_standard_deviation (PHOEBE_vector *vec, double *sigma);
int            phoebe_vector_multiply_by        (PHOEBE_vector *vec, double factor);
int            phoebe_vector_dot_product        (double *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_vec_product        (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_submit             (PHOEBE_vector *result, PHOEBE_vector *vec, double func ());
int            phoebe_vector_norm               (double *result, PHOEBE_vector *vec);
int            phoebe_vector_dim                (int *result, PHOEBE_vector *vec);
int            phoebe_vector_randomize          (PHOEBE_vector *result, double limit);
int            phoebe_vector_min_max            (PHOEBE_vector *vec, double *min, double *max);
int            phoebe_vector_min_index          (PHOEBE_vector *vec, int *index);
int            phoebe_vector_max_index          (PHOEBE_vector *vec, int *index);
int            phoebe_vector_rescale            (PHOEBE_vector *vec, double ll, double ul);
bool           phoebe_vector_compare            (PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_less_than          (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_leq_than           (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_greater_than       (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_geq_than           (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);

int            phoebe_vector_append_element     (PHOEBE_vector *vec, double val);
int            phoebe_vector_remove_element     (PHOEBE_vector *vec, int index);

/* **************************************************************************** */

/**
 * PHOEBE_matrix:
 * @rows: The horizontal dimension of the matrix.
 * @cols: The vertical dimension of the matrix.
 * @val:  The elements of the matrix.
 */
typedef struct PHOEBE_matrix {
	int rows;
	int cols;
	double **val;
} PHOEBE_matrix;

PHOEBE_matrix *phoebe_matrix_new     ();
int            phoebe_matrix_alloc   (PHOEBE_matrix *matrix, int cols, int rows);
int            phoebe_matrix_free    (PHOEBE_matrix *matrix);
int            phoebe_matrix_get_row (PHOEBE_vector *vec, PHOEBE_matrix *matrix, int row);
int            phoebe_matrix_set_row (PHOEBE_matrix *matrix, PHOEBE_vector *vec, int row);

/* **************************************************************************** */

/**
 * PHOEBE_array:
 * @dim:  The size of the array.
 * @type: The type of the array.
 * @val:  The elements of the array.
 */
typedef struct PHOEBE_array {
	int              dim;
	PHOEBE_type     type;
	union {
		int      *iarray;
		double   *darray;
		bool     *barray;
		char  **strarray;
	} val;
} PHOEBE_array;

PHOEBE_array  *phoebe_array_new                 (PHOEBE_type type);
PHOEBE_array  *phoebe_array_new_from_qualifier  (char *qualifier);
PHOEBE_array  *phoebe_array_new_from_column     (char *filename, int col);
int            phoebe_array_alloc               (PHOEBE_array *array, int dimension);
int            phoebe_array_realloc             (PHOEBE_array *array, int dimension);
PHOEBE_array  *phoebe_array_duplicate           (PHOEBE_array *array);
bool           phoebe_array_compare             (PHOEBE_array *array1, PHOEBE_array *array2);
int            phoebe_array_free                (PHOEBE_array *array);

PHOEBE_vector *phoebe_vector_new_from_array     (PHOEBE_array *array);

/* **************************************************************************** */

/*
 * PHOEBE histograms:
 *
 *   In PHOEBE, histograms are used for all sorts of discrete PDF data like
 *   spectra. This is a general structure with the following layout:
 *
 *         [ bin[0] )[ bin[1] )[ bin[2] )[ bin[3] )[ bin[4] )[
 *      ---|---------|---------|---------|---------|---------|---  x
 *       r[0]      r[1]      r[2]      r[3]      r[4]      r[5]
 *
 *   The range for bin[i] is given by range[i] to range[i+1]. For n bins
 *   there are n+1 entries in the array range. Each bin is inclusive at the
 *   lower end and exclusive at the upper end. Ranges are necessary for
 *   non-uniformely spaced bins, as in spectra, where bins are spaced
 *   logarithmically.
 */

/**
 * PHOEBE_hist:
 * @bins:  Number of histogram bins.
 * @range: A vector of histogram ranges.
 * @val:   A vector of histogram values.
 */
typedef struct PHOEBE_hist {
	int bins;
	double *range;
	double *val;
} PHOEBE_hist;

/**
 * PHOEBE_hist_rebin_type:
 * @PHOEBE_HIST_CONSERVE_VALUES:  When rebinning, conserve the values.
 * @PHOEBE_HIST_CONSERVE_DENSITY: When rebinning, conserve the densities.
 *
 * There are two ways to rebin a histogram:
 *
 *   1) conserve the values and
 *   2) conserve the value densities.
 *
 * The first option is better if we are degrading the histogram, and the
 * second option is better if we are oversampling the histogram.
 */

typedef enum PHOEBE_hist_rebin_type {
	PHOEBE_HIST_CONSERVE_VALUES = 20,
	PHOEBE_HIST_CONSERVE_DENSITY
} PHOEBE_hist_rebin_type;

PHOEBE_hist *phoebe_hist_new             ();
PHOEBE_hist *phoebe_hist_new_from_arrays (int bins, double *binarray, double *valarray);
PHOEBE_hist *phoebe_hist_new_from_file   (char *filename);
PHOEBE_hist *phoebe_hist_duplicate       (PHOEBE_hist *hist);
int          phoebe_hist_alloc           (PHOEBE_hist *hist, int bins);
int          phoebe_hist_realloc         (PHOEBE_hist *hist, int bins);
int          phoebe_hist_free            (PHOEBE_hist *hist);

int          phoebe_hist_set_ranges      (PHOEBE_hist *hist, PHOEBE_vector *bin_centers);
int          phoebe_hist_set_values      (PHOEBE_hist *hist, PHOEBE_vector *values);
int          phoebe_hist_get_bin_centers (PHOEBE_hist *hist, PHOEBE_vector *bin_centers);
int          phoebe_hist_get_bin         (int *bin, PHOEBE_hist *hist, double r);
int          phoebe_hist_evaluate        (double *y, PHOEBE_hist *hist, double x);
int          phoebe_hist_integrate       (double *integral, PHOEBE_hist *hist, double ll, double ul);
int          phoebe_hist_shift           (PHOEBE_hist *hist, double shift);
int          phoebe_hist_correlate       (double *cfval, PHOEBE_hist *h1, PHOEBE_hist *h2, double sigma1, double sigma2, double ll, double ul, double xi);
int          phoebe_hist_pad             (PHOEBE_hist *hist, double val);
int          phoebe_hist_crop            (PHOEBE_hist *hist, double ll, double ul);
bool         phoebe_hist_compare         (PHOEBE_hist *hist1, PHOEBE_hist *hist2);
int          phoebe_hist_resample        (PHOEBE_hist *out, PHOEBE_hist *in, PHOEBE_hist_rebin_type type);
int          phoebe_hist_rebin           (PHOEBE_hist *out, PHOEBE_hist *in, PHOEBE_hist_rebin_type type);

/* **************************************************************************** */

typedef struct PHOEBE_ld {
	char *set;
	char *name;
	char *reftable;
	PHOEBE_vector *lin_x;
	PHOEBE_vector *log_x;
	PHOEBE_vector *log_y;
	PHOEBE_vector *sqrt_x;
	PHOEBE_vector *sqrt_y;
} PHOEBE_ld;

/**
 * PHOEBE_passband:
 * @id:    ID number of the passband
 * @set:   Filter-set of the passband, i.e. "Cousins"
 * @name:  Passband identifier, i.e. "Rc"
 * @effwl: Effective wavelength of the passband.
 * @tf:    Passband transmission function
 * @ld:    Limb darkening table (attached optionally)
 */

typedef struct PHOEBE_passband {
	int          id;
	char        *set;
	char        *name;
	double       effwl;
	PHOEBE_hist *tf;
	PHOEBE_ld   *ld;
} PHOEBE_passband;

/* **************************************************************************** */

/**
 * PHOEBE_curve_type:
 * @PHOEBE_CURVE_UNDEFINED: The type of the curve is unknown.
 * @PHOEBE_CURVE_LC:        A light curve.
 * @PHOEBE_CURVE_RV:        A radial velocity curve.
 */
typedef enum PHOEBE_curve_type {
	PHOEBE_CURVE_UNDEFINED,
	PHOEBE_CURVE_LC,
	PHOEBE_CURVE_RV
} PHOEBE_curve_type;

int phoebe_curve_type_get_name (PHOEBE_curve_type ctype, char **name);

/**
 * PHOEBE_column_type:
 *
 * Various sorts of data that can be found in files usable by Phoebe.
 */
typedef enum PHOEBE_column_type {
	PHOEBE_COLUMN_UNDEFINED,
	PHOEBE_COLUMN_HJD,
	PHOEBE_COLUMN_PHASE,
	PHOEBE_COLUMN_MAGNITUDE,
	PHOEBE_COLUMN_FLUX,
	PHOEBE_COLUMN_PRIMARY_RV,
	PHOEBE_COLUMN_SECONDARY_RV,
	PHOEBE_COLUMN_SIGMA,
	PHOEBE_COLUMN_WEIGHT,
	PHOEBE_COLUMN_INVALID
} PHOEBE_column_type;

int phoebe_column_type_get_name (PHOEBE_column_type ctype, char **name);
int phoebe_column_get_type (PHOEBE_column_type *type, const char *string);

typedef enum PHOEBE_data_flag {
	PHOEBE_DATA_REGULAR,
	PHOEBE_DATA_ALIASED,
	PHOEBE_DATA_DELETED,
	PHOEBE_DATA_OMITTED,
	PHOEBE_DATA_DELETED_ALIASED
} PHOEBE_data_flag;

/**
 * PHOEBE_curve:
 * @type:     Type of the curve.
 * @passband: Passband of the curve.
 * @indep:    Elements of the independant variable vector.
 * @dep:      Elements of the dependant variable vector.
 * @weight:   Elements of the weight vector.
 * @flag:     data flag of the enumerated #PHOEBE_data_flag type
 * @itype:    Column type of the independant variable.
 * @dtype:    Column type of the dependant variable.
 * @wtype:    Column type of the weights.
 * @filename: Absolute path to the file containing the curve.
 * @sigma:    Sigma value of the curve.
 */

typedef struct PHOEBE_curve {
	PHOEBE_curve_type  type;
	PHOEBE_passband   *passband;
	PHOEBE_vector     *indep;
	PHOEBE_vector     *dep;
	PHOEBE_vector     *weight;
	PHOEBE_array      *flag;
	PHOEBE_column_type itype;
	PHOEBE_column_type dtype;
	PHOEBE_column_type wtype;
	char              *filename;
	double             sigma;
} PHOEBE_curve;

PHOEBE_curve *phoebe_curve_new            ();
PHOEBE_curve *phoebe_curve_new_from_file  (char *filename);
PHOEBE_curve *phoebe_curve_new_from_pars  (PHOEBE_curve_type ctype, int index);
PHOEBE_curve *phoebe_curve_duplicate      (PHOEBE_curve *curve);
int           phoebe_curve_alloc          (PHOEBE_curve *curve, int dim);
int           phoebe_curve_realloc        (PHOEBE_curve *curve, int dim);
int           phoebe_curve_compute        (PHOEBE_curve *curve, PHOEBE_vector *nodes, int index, PHOEBE_column_type itype, PHOEBE_column_type dtype);
int           phoebe_curve_transform      (PHOEBE_curve *curve, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype);
int           phoebe_curve_alias          (PHOEBE_curve *curve, double phmin, double phmax);
int           phoebe_curve_set_properties (PHOEBE_curve *curve, PHOEBE_curve_type type, char *filename, PHOEBE_passband *passband, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype, double sigma);
int           phoebe_curve_free           (PHOEBE_curve *curve);

/* **************************************************************************** */

/**
 * PHOEBE_spectrum_dispersion:
 *
 * Spectrum dispersion tells us how dx is connected to dlambda. If the
 * dispersion is linear, then dx is proportional to dlambda. If it is log,
 * dx is proportional to dlambda/lambda. If there is no dispersion function,
 * that means that there is no simple transformation from wavelength space
 * to pixel space. In that case everything must be done according to histogram
 * ranges.
 */
typedef enum PHOEBE_spectrum_dispersion {
	PHOEBE_SPECTRUM_DISPERSION_LINEAR,
	PHOEBE_SPECTRUM_DISPERSION_LOG,
	PHOEBE_SPECTRUM_DISPERSION_NONE
} PHOEBE_spectrum_dispersion;

/**
 * PHOEBE_spectrum:
 *
 * The spectrum structure is defined here, but the manipulation functions
 * reside in phoebe_spectra files.
 */
typedef struct PHOEBE_spectrum {
	double R;
	double Rs;
	double dx;
	PHOEBE_spectrum_dispersion disp;
	PHOEBE_hist *data;
} PHOEBE_spectrum;

/* **************************************************************************** */

/**
 * PHOEBE_minimizer_type:
 * @PHOEBE_MINIMIZER_NMS: The Nead-Medler minimizer.
 * @PHOEBE_MINIMIZER_DC:  The differential-corrections minimizer.
 *
 * The available minimizers.
 */
typedef enum PHOEBE_minimizer_type {
	PHOEBE_MINIMIZER_NMS,
	PHOEBE_MINIMIZER_DC
} PHOEBE_minimizer_type;

int phoebe_minimizer_type_get_name (PHOEBE_minimizer_type minimizer, char **name);

/**
 * PHOEBE_minimizer_feedback:
 * @algorithm:  Minimizer (algorithm) type
 * @cputime:    CPU time required for algorithm execution
 * @iters:      Number of performed iterations
 * @cfval:      Cost function value (combined chi2)
 * @qualifiers: A list of TBA qualifiers
 * @initvals:   A list of initial parameter values
 * @newvals:    A list of new parameter values
 * @ferrors:    A list of formal error estimates
 * @u_res:      Unweighted residuals
 * @i_res:      Residuals weighted with intrinsic weights
 * @p_res:      Residuals weighted with intrinsic and passband weights
 * @f_res:      Fully weighted residuals (intrinsic + passband + level)
 * @chi2s:      A list of passband chi2 values
 * @wchi2s:     A list of weighted passband chi2 values
 * @cormat:     Correlation matrix
 */
typedef struct PHOEBE_minimizer_feedback {
	PHOEBE_minimizer_type algorithm;
	bool             converged;
	double           cputime;
	int              iters;
	double           cfval;
	PHOEBE_array    *qualifiers;
	PHOEBE_vector   *initvals;
	PHOEBE_vector   *newvals;
	PHOEBE_vector   *ferrors;
	PHOEBE_vector   *u_res;
	PHOEBE_vector   *i_res;
	PHOEBE_vector   *p_res;
	PHOEBE_vector   *f_res;
	PHOEBE_vector   *chi2s;
	PHOEBE_vector   *wchi2s;
	PHOEBE_matrix   *cormat;
	PHOEBE_vector   *__cla;
} PHOEBE_minimizer_feedback;

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_new       ();
PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_duplicate (PHOEBE_minimizer_feedback *feedback);
int                        phoebe_minimizer_feedback_alloc     (PHOEBE_minimizer_feedback *feedback, int tba, int cno, int __lcno);
int                        phoebe_minimizer_feedback_accept    (PHOEBE_minimizer_feedback *feedback);
int                        phoebe_minimizer_feedback_free      (PHOEBE_minimizer_feedback *feedback);

/* **************************************************************************** */

typedef union PHOEBE_value {
	int                        i;
	double                     d;
	bool                       b;
	char                      *str;
	PHOEBE_vector             *vec;
	PHOEBE_array              *array;
	PHOEBE_curve              *curve;
	PHOEBE_spectrum           *spectrum;
	PHOEBE_minimizer_feedback *feedback;
} PHOEBE_value;

extern PHOEBE_spectrum *phoebe_spectrum_duplicate (PHOEBE_spectrum *spectrum);
PHOEBE_value phoebe_value_duplicate (PHOEBE_type type, PHOEBE_value val);

/* **************************************************************************** */

#endif
