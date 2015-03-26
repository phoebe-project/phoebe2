
/****************************************************************************/
/*                                                                          */
/*                            DiffCorr.h                                    */
/*                            ------------                                  */
/*                                                                          */
/* Contains the prototypes. typedef's, const, etc. for the Differential     */
/* correction algoritum                                                     */
/* June 4, 2014       Joe                                                   */
/****************************************************************************/





#ifndef DIFFCORR_H
#define DIFFCORR_H




#define	TOL	0.00001
/*#define	TOL_DC	0.0001*/
#define	TOL_DC	0.00001
/*#define	maxIterations	80*/




#define CON 1.4
#define CON2 (CON*CON)
#define BIG 1.0e30
#define NTAB 10
#define SAFE 2.0

#define FMAX(a,b) ((a) > (b) ? (a) : (b))

enum derType { NUMERICAL, ANALYTICAL, NONE };
enum derCalc { NOT_CALC_ED, CALC_ED };
enum stoppingCriteria { MIN_DX, MIN_DELTA_DX, MIN_CHI2, MIN_DELTA_CHI2};



#ifndef FALSE
#define FALSE (0)
#define TRUE (!(FALSE))
#endif
 
 
 
 
/*
 * prototypes
 */



/* if we ever extend to calling c functions ???? */
/*
typedef double (*lFunction)(double *x, double t);
typedef double (*derFunctions)(double *x, double t, int eleIndex);

derFunctions	dFunc;
lFunction	lFunc;
*/

/* wrapper functions to the python function calls */
void lFunc(double *x, double *model);
void dFunc(double *x, int eleIndex, double *dF);


typedef struct {

double	*SolutionVecM,		/* This is dx = J^-1 . df */
	*model,
	*data,
	*params,
	*paramsLast,
	*paramsDifferance,
	*dL,			/* dL = y_observed - y_calc = y_observed - F_c(x) */
	*initialGuess,
	derError,
	stopValue;	/* the value of the stoppingCriteria to end the calculation */

/* wrapper functions to the python function calls */
/* we don't really need them here now, but leave as hooks for future use */
/*
derFunctions	*dFunc;		/ the derivative functions /
lFunction	*lFunc;		/ the "light" calculation function /
*/

int	nParams,
	nDataPoints,
	*derivativeType,
	maxIterations,
	stoppingCriteriaType;

} DifferentialCorrection;





void DCFront(void);




/* ************************************************************************
 * ** Function: DiffCorr                                                 **
 * ** Functionality: Uses the DC algorithum to find the values of a      **
 * **                models (function's) pareters that allow the model   **
 * **                to best reproduc the data.                          **
 * **                      **
 * **                        **
 * **                                                                    **
 * **                                                                    **
 * ** Diagnostics: Returns through the call variable perror a (0) if no  **
 * **              error.  Othere error codes will need to be defined.   **
 * ************************************************************************
 */
void DiffCorr(DifferentialCorrection *diffCorr, int *perror);


void MatrixVectorProduct(double *MatVec, double **Mat, double *Vec, int numRow, int numCol);
void Transpose(double **TransposeMat, double **Mat,	int nRow, int nCol);
void EquateVectors(double *VectorEmpty, double *VectorFilled, int nEntries);
double Delta(int I, int J);
void
	MatInverseGaussJordan(double **A, double **AInv, double **AP, int N);

void VectorSumWithScalarMult(double *solution, double *vector1, double scalar,
double *vector2, int nEntries);

double dotProduct(double *vector1, double *vector2, int nEntries);

void CGLS(double *SolutionVecM, double **MatG, double *RHSVecD,
			int nRow, int nCol);

void PrintVector(double *Vector, int nEntries);

void free_dmatrix(double **m, int nr, int nc);
double **dmatrix(int nrow, int ncol);
void free_dvector(double *v, int nrow);
double *dvector(int nrow);
int *ivector(int nrow);
void free_ivector(int *v, int nrow);


/*  elelmentToVary stars at 0 
    the retuned integer just tells if there was a failure, i.e. h == 0
*/
int dfridr(void (*func)(double *, double *), double *x, double h,
		int numElements, int elementToVary, int nDataPoints,
		double *dF, double *err);



/* ************************************************************************
 * ** Function: calcChi2                                                 **
 * ** Functionality: Given a set of Obsered data and the corresponding   **
 * **                Calculated (model) data and the number of data      **
 * **                points, calculates a chi^2 value.  This does not    **
 * **                account for error in the observed data.             **
 * **                                                                    **
 * **                                                                    **
 * ** Diagnostics: None at this time.                                    **
 * ************************************************************************
 */

double calcChi2(double *Obs, double *Calc, int nDataPoints);





#endif
