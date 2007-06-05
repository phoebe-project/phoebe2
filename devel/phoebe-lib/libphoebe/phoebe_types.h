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

/******************************************************************************/

typedef struct PHOEBE_vector {
	int dim;          /* Vector dimension                                     */
	double *val;      /* An array of vector values                            */
} PHOEBE_vector;

PHOEBE_vector *phoebe_vector_new                ();
PHOEBE_vector *phoebe_vector_new_from_qualifier (char *qualifier);
PHOEBE_vector *phoebe_vector_new_from_column    (char *filename, int col);
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
int            phoebe_vector_multiply_by        (PHOEBE_vector *fac1, double factor);
int            phoebe_vector_dot_product        (double *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_vec_product        (PHOEBE_vector *result, PHOEBE_vector *fac1, PHOEBE_vector *fac2);
int            phoebe_vector_submit             (PHOEBE_vector *result, PHOEBE_vector *vec, double func ());
int            phoebe_vector_norm               (double *result, PHOEBE_vector *vec);
int            phoebe_vector_dim                (int *result, PHOEBE_vector *vec);
int            phoebe_vector_randomize          (PHOEBE_vector *result, double limit);
int            phoebe_vector_min_max            (PHOEBE_vector *vec, double *min, double *max);
bool           phoebe_vector_compare            (PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_less_than          (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_leq_than           (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_greater_than       (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);
int            phoebe_vector_geq_than           (bool *result, PHOEBE_vector *vec1, PHOEBE_vector *vec2);

int            phoebe_vector_append_element     (PHOEBE_vector *vec, double val);
int            phoebe_vector_remove_element     (PHOEBE_vector *vec, int index);

/******************************************************************************/

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
int            phoebe_array_alloc               (PHOEBE_array *array, int dimension);
int            phoebe_array_realloc             (PHOEBE_array *array, int dimension);
PHOEBE_array  *phoebe_array_new_from_qualifier  (char *qualifier);
PHOEBE_array  *phoebe_array_duplicate           (PHOEBE_array *array);
int            phoebe_array_free                (PHOEBE_array *array);

/******************************************************************************/

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

typedef struct PHOEBE_hist {
	int bins;               /* Number of histogram bins     */
	double *range;          /* A vector of histogram ranges */
	double *val;            /* A vector of histogram values */
} PHOEBE_hist;

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
int          phoebe_hist_rebin           (PHOEBE_hist *out, PHOEBE_hist *in, PHOEBE_hist_rebin_type type);

/******************************************************************************/

typedef struct PHOEBE_passband {
	int          id;
	char        *set;
	char        *name;
	double       effwl;
	PHOEBE_hist *tf;
} PHOEBE_passband;

/******************************************************************************/

typedef enum PHOEBE_curve_type {
	PHOEBE_CURVE_UNDEFINED,
	PHOEBE_CURVE_LC,
	PHOEBE_CURVE_RV
} PHOEBE_curve_type;

int phoebe_curve_type_get_name (PHOEBE_curve_type ctype, char **name);

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

int phoebe_column_type_get_name    (PHOEBE_column_type ctype, char **name);
int phoebe_column_type_from_string (const char *string, PHOEBE_column_type *type);

typedef struct PHOEBE_curve {
	PHOEBE_curve_type  type;
	PHOEBE_passband   *passband;
	PHOEBE_vector     *indep;
	PHOEBE_vector     *dep;
	PHOEBE_vector     *weight;
	PHOEBE_column_type itype;
	PHOEBE_column_type dtype;
	PHOEBE_column_type wtype;
	char              *filename;
	double             sigma;
	/* Temporary fields: */
	double             L1;
	double             L2;
	double             SBR1;
	double             SBR2;
	/* ***************** */
} PHOEBE_curve;

PHOEBE_curve *phoebe_curve_new            ();
PHOEBE_curve *phoebe_curve_new_from_file  (char *filename);
PHOEBE_curve *phoebe_curve_new_from_pars  (PHOEBE_curve_type ctype, int index);
PHOEBE_curve *phoebe_curve_duplicate      (PHOEBE_curve *curve);
int           phoebe_curve_alloc          (PHOEBE_curve *curve, int dim);
int           phoebe_curve_transform      (PHOEBE_curve *curve, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype);
int           phoebe_curve_set_properties (PHOEBE_curve *curve, PHOEBE_curve_type type, char *filename, PHOEBE_passband *passband, PHOEBE_column_type itype, PHOEBE_column_type dtype, PHOEBE_column_type wtype, double sigma);
int           phoebe_curve_free           (PHOEBE_curve *curve);

/******************************************************************************/

/*
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

/*
 * The spectrum structure is defined here, but the manipulation functions
 * reside in phoebe_spectra.c/h files.
*/

typedef struct PHOEBE_spectrum {
	double  R;
	double  Rs;
	PHOEBE_spectrum_dispersion disp;
	PHOEBE_hist *data;
} PHOEBE_spectrum;

/******************************************************************************/

typedef enum PHOEBE_minimizer_type {
	PHOEBE_MINIMIZER_NMS,
	PHOEBE_MINIMIZER_DC
} PHOEBE_minimizer_type;

int phoebe_minimizer_type_get_name (PHOEBE_minimizer_type minimizer, char **name);

typedef struct PHOEBE_minimizer_feedback {
	PHOEBE_minimizer_type algorithm; /* Minimizer (algorithm) type            */
	double           cputime;    /* CPU time required for algorithm execution */
	int              iters;      /* Number of performed iterations            */
	double           cfval;      /* Cost function value (combined chi2)       */
	PHOEBE_array    *qualifiers; /* A list of TBA qualifiers                  */
	PHOEBE_vector   *initvals;   /* A list of initial parameter values        */
	PHOEBE_vector   *newvals;    /* A list of new parameter values            */
	PHOEBE_vector   *ferrors;    /* A list of formal error estimates          */
	PHOEBE_vector   *chi2s;      /* A list of passband chi2 values            */
	PHOEBE_vector   *wchi2s;     /* A list of weighted passband chi2 values   */
	PHOEBE_array    *indices;    /* A list of indices of TBA parameters       */
	struct PHOEBE_parameter_list *pars; /* A list of parameters marked for adjustment*/
} PHOEBE_minimizer_feedback;

PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_new       ();
PHOEBE_minimizer_feedback *phoebe_minimizer_feedback_duplicate (PHOEBE_minimizer_feedback *feedback);
int                        phoebe_minimizer_feedback_alloc     (PHOEBE_minimizer_feedback *feedback, int tba, int cno);
int                        phoebe_minimizer_feedback_free      (PHOEBE_minimizer_feedback *feedback);

/******************************************************************************/

typedef union anytype {
	int                        i;
	double                     d;
	bool                       b;
	char                      *str;
	PHOEBE_vector             *vec;
	PHOEBE_array              *array;
	PHOEBE_curve              *curve;
	PHOEBE_spectrum           *spectrum;
	PHOEBE_minimizer_feedback *feedback;
	int                       *iarray;
	double                    *darray;
	bool                      *barray;
	char                     **strarray;
} anytype;

/******************************************************************************/

typedef struct WD_LCI_parameters {
	int    MPAGE;
	int    NREF;
	int    MREF;
	int    IFSMV1;
	int    IFSMV2;
	int    ICOR1;
	int    ICOR2;
	int    LD;
	int    JDPHS;
	double HJD0;
	double PERIOD;
	double DPDT;
	double PSHIFT;
	double SIGMA;
	int    WEIGHTING;
	double SEED;
	double HJDST;
	double HJDSP;
	double HJDIN;
	double PHSTRT;
	double PHSTOP;
	double PHIN;
	double PHNORM;
	int    MODE;
	int    IPB;
	int    CALCHLA;
	int    CALCVGA;
	bool   MSC1;
	bool   MSC2;
	bool   ASINI;
	bool   CINDEX;
	int    IFAT1;
	int    IFAT2;
	int    N1;
	int    N2;
	double PERR0;
	double DPERDT;
	double THE;
	double VUNIT;
	double E;
	double SMA;
	double F1;
	double F2;
	double VGA;
	double INCL;
	double GR1;
	double GR2;
	double LOGG1;
	double LOGG2;
	double MET1;
	double MET2;
	double TAVH;
	double TAVC;
	double ALB1;
	double ALB2;
	double PHSV;
	double PCSV;
	double RM;
	double XBOL1;
	double XBOL2;
	double YBOL1;
	double YBOL2;
	int    IBAND;
	double HLA;
	double CLA;
	double X1A;
	double X2A;
	double Y1A;
	double Y2A;
	double EL3;
	double OPSF;
	double MZERO;
	double FACTOR;
	double WLA;
	int    SPRIM;
	double *XLAT1;
	double *XLONG1;
	double *RADSP1;
	double *TEMSP1;
	int    SSEC;
	double *XLAT2;
	double *XLONG2;
	double *RADSP2;
	double *TEMSP2;
} WD_LCI_parameters;

typedef struct WD_DCI_parameters {
	bool   *tba;
	double *step;
	double dclambda;
	int    nlc;
	bool   rv1data;
	bool   rv2data;
	bool   symder;
	int    refswitch;
	int    refno;
	bool   rv1proximity;
	bool   rv2proximity;
	int    ldmodel;
	int    indep;
	int    morph;
	bool   cladec;
	bool   ifat1;
	bool   ifat2;
	int    n1c;
	int    n2c;
	int    n1f;
	int    n2f;
	double hjd0;
	double period;
	double dpdt;
	double pshift;
	double perr0;
	double dperdt;
	double ecc;
	double sma;
	double f1;
	double f2;
	double vga;
	double incl;
	double grb1;
	double grb2;
	double met1;
	double teff1;
	double teff2;
	double alb1;
	double alb2;
	double pot1;
	double pot2;
	double rm;
	double xbol1;
	double xbol2;
	double ybol1;
	double ybol2;
	int    *passband;
	double *wavelength;
	double *sigma;
	double *hla;
	double *cla;
	double *x1a;
	double *y1a;
	double *x2a;
	double *y2a;
	double *el3;
	double *opsf;
	int    *levweight;
	int    spot1no;
	int    spot2no;
	int    spot1src;
	int    spot2src;
	int    spot1id;
	int    spot2id;
	bool   spots1move;
	bool   spots2move;
	double *spot1lat;
	double *spot1long;
	double *spot1rad;
	double *spot1temp;
	double *spot2lat;
	double *spot2long;
	double *spot2rad;
	double *spot2temp;
	PHOEBE_curve **obs;
} WD_DCI_parameters;

typedef enum PHOEBE_input_indep {
	INPUT_HJD,
	INPUT_PHASE
} PHOEBE_input_indep;

typedef enum PHOEBE_input_dep {
	INPUT_FLUX,
	INPUT_MAGNITUDE,
	INPUT_PRIMARY_RV,
	INPUT_SECONDARY_RV
} PHOEBE_input_dep;

typedef enum PHOEBE_input_weight {
	INPUT_STANDARD_WEIGHT,
	INPUT_STANDARD_DEVIATION,
	INPUT_UNAVAILABLE
} PHOEBE_input_weight;

typedef enum PHOEBE_output_indep {
	OUTPUT_HJD,
	OUTPUT_PHASE
} PHOEBE_output_indep;

typedef enum PHOEBE_output_dep {
	OUTPUT_MAGNITUDE,
	OUTPUT_PRIMARY_FLUX,
	OUTPUT_SECONDARY_FLUX,
	OUTPUT_TOTAL_FLUX,
	OUTPUT_PRIMARY_RV,
	OUTPUT_SECONDARY_RV,
	OUTPUT_BOTH_RVS,
	OUTPUT_PRIMARY_NORMALIZED_RV,
	OUTPUT_SECONDARY_NORMALIZED_RV
} PHOEBE_output_dep;

typedef enum PHOEBE_output_weight {
	OUTPUT_STANDARD_WEIGHT,
	OUTPUT_STANDARD_DEVIATION,
	OUTPUT_UNAVAILABLE
} PHOEBE_output_weight;

char          *phoebe_input_indep_name          (PHOEBE_input_indep indep);
char          *phoebe_input_dep_name            (PHOEBE_input_dep   dep);
char          *phoebe_input_weight_name         (PHOEBE_input_weight weight);
char          *phoebe_output_indep_name         (PHOEBE_output_indep indep);
char          *phoebe_output_dep_name           (PHOEBE_output_dep dep);
char          *phoebe_output_weight_name        (PHOEBE_output_weight);

int            phoebe_lc_params_update          (WD_LCI_parameters params, int curve);
int            phoebe_rv_params_update          (WD_LCI_parameters params, int curve);

#endif
