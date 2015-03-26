/****************************************************************************/
/*                                                                          */
/*                            DiffCorr.c                                    */
/*                            ------------                                  */
/*                                                                          */
/* Contains the functions used in finding model parameters using the        */
/* Differential Correction Method                                           */
/*                                                                          */
/* File Created:  June, 2014     JMG                                        */
/* Last Modified: Aug 13, 2014                                              */
/****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "DiffCorr.h"











void DiffCorr(DifferentialCorrection *diffCorr, int *perror)
{
double	*SolutionVecM,
	*model,
	*data,
	*params,
	*paramsLast,
	*paramsDifferance,
	*dL,
	*initialGuess,
	derError,
	stopValue,

	/* local variables */
	**MatG,
	*MatGCol,
	h,
	dxMag,
	dxMagLast,
	dxDelta,
	chi2Mag,
	chi2MagLast,
	chi2Delta,
	magnitude;

int	*derivativeType,
	nParams,
	maxIterations,
	stoppingCriteriaType,


	/* local variables */
	nRow,
	nCol,
	I,
	i,
	k,
	numFixedParams=0,
	iCounter,
	stopIterating = FALSE,
	error = 0;


	/* Initialize the local pointers */
	nRow = diffCorr->nDataPoints;
	nCol = diffCorr->nParams;
	nParams = diffCorr->nParams;
	maxIterations = diffCorr->maxIterations;
	stoppingCriteriaType = diffCorr->stoppingCriteriaType;
	stopValue = diffCorr->stopValue;


	SolutionVecM = diffCorr->SolutionVecM;
	model = diffCorr->model;
	data = diffCorr->data;
	params = diffCorr->params;
	paramsLast = diffCorr->paramsLast;
	paramsDifferance = diffCorr->paramsDifferance;
	dL = diffCorr->dL;
	initialGuess = diffCorr->initialGuess;


	derivativeType = diffCorr->derivativeType;




	for (i = 0; i < nParams; i++)
	{
		if(derivativeType[i] == NONE) numFixedParams++;
	}
	if(numFixedParams == nParams)
	{
		printf("All Parameters Fixed???...problem in file %s line %d\n",__FILE__, __LINE__);
		exit (EXIT_FAILURE);
	}


	/*
	 * allocate memory for the matrice and its associated column vector
	 */
	MatG = dmatrix(nRow, nCol);
	MatGCol = dvector(nRow);



	/* set the params values to the initial guess to start */
	for (I = 0; I < nCol; I++)
	{
		params[I] = initialGuess[I];
	}
	



	magnitude = 100;
	iCounter = 1;
	while( (iCounter < maxIterations) && !stopIterating )
	{

		/* fill in model vector */
		lFunc(params, model);


		/* fill in the dL vector...the RHS */
		VectorSumWithScalarMult(dL, data, -1, model, nRow);
		/* i is the index of the element to vary */
		for (i = 0; i < nCol; i++)
		{

			switch (derivativeType[i])
			{
				case ANALYTICAL:
					dFunc(params,i, MatGCol);
					break;

				case NUMERICAL: /* NEED BETTER WAY TO GET h */
					if (abs(params[i]) < 2) h = 0.02;
					else h = 1;
					(void ) dfridr(lFunc, params, h,
						nParams, i, nRow, MatGCol, &derError);

					break;

				case NONE:
					for (k = 0; k < nRow; k++)
					{
						MatGCol[k] = 0.00;
					}
					break;

				default:
					printf("Unknown derrivative type for element %d\n",i);
					break;
			}

			for (k = 0; k < nRow; k++)
			{
				MatG[k][i] = MatGCol[k];
			}



		}


		CGLS(SolutionVecM, MatG, dL, nRow, nCol);


		EquateVectors(paramsLast, params, nParams);
		for (i = 0; i < nCol; i++)
		{
			params[i] = params[i] + SolutionVecM[i];
		}


	

		/* let's calculate these all here, in case we need them
		 * for other reason's later
		 */
		dxMag = 
			sqrt(dotProduct(SolutionVecM, SolutionVecM, nCol));
		dxDelta = fabs(dxMag - dxMagLast);
		dxMagLast = dxMag;

		chi2Mag = calcChi2(data, model, nRow);
		chi2Delta = fabs(chi2Mag - chi2MagLast);
		chi2MagLast = chi2Mag;

		/* see if we should continue */
		switch (stoppingCriteriaType)
		{
			case MIN_DX:
				if (dxMag < stopValue)
				{
					magnitude = dxMag;
					stopIterating = TRUE;
				}
				break;

			case MIN_DELTA_DX:
				if (dxDelta < stopValue)
				{
					magnitude = dxDelta;
					stopIterating = TRUE;
				}
				break;

			case MIN_CHI2:
				if (chi2Mag < stopValue)
				{
					magnitude = chi2Mag;
					stopIterating = TRUE;
				}
				break;

			case MIN_DELTA_CHI2:
				if (chi2Delta < stopValue)
				{
					magnitude = chi2Delta;
					stopIterating = TRUE;
				}
				break;

			default:
				printf("Unknown stopping criteria\n");
				break;
		}


		iCounter++;

	}



	printf("\n\n This took %d iterations --- magnitude = %g\n\n\n",
			iCounter-1, magnitude);



	/* free the matrix memory */
	free_dmatrix(MatG, nRow, nCol);
	free_dvector(MatGCol, nRow);

	diffCorr->derError = derError;
	*perror = error;


}







/* ========================================================================
 * ==                                                                    ==
 * ==                     function MatrixVectorProduct                   ==
 * ==                                                                    ==
 * == Functionality: MatrixVectorProduct calculates the produce of       ==
 * == a matrix Mat and a vector Vec and returns the result in MatVec     ==
 * == the dimensions of the matrix are required                          ==
 * ==                                                                    ==
 * == Diagnostics: None  at this time                                    ==
 * ========================================================================
 */
void MatrixVectorProduct(double *MatVec, double **Mat, double *Vec,
		int numRow, int numCol)
{
int	matRow,
	matColumn;

	for (matRow = 0; matRow < numRow; matRow++)
	{
		MatVec[matRow] = 0;
		for (matColumn = 0; matColumn < numCol; matColumn++)
		{
			MatVec[matRow] +=
				Mat[matRow][matColumn]*Vec[matColumn];
		}
	}

}


/* ========================================================================
 * ==                                                                    ==
 * ==                     function DELTA                                 ==
 * ==                                                                    ==
 * == Functionality: DELTA is the Kroniker delta function.  It is equal  ==
 * == to 1 for I == J,  and equal to 0 otherwise.                        ==
 * == It returns a number of type double.                                ==
 * ==                                                                    ==
 * == Diagnostics: None                                                  ==
 * ==                                                                    ==
 * == NOTE: This could be replaced by a macro for speed!                 ==
 * ========================================================================
 */
double Delta(int I, int J)
{

	if ( I == J) return (1.0);
	else return (0.0);

}


/*=====================================================================
 * ==                                                                ==
 * ==                    funtion MatInverseGaussJordan               ==
 * ==                                                                ==
 * == This code finds the inverse of a (square) nXn matrix by the    ==
 * == Gauss-Jordan method...See Scarborough pg. 285.                 ==
 * == See Scarborough "Numerical Mathematical Analysis", pg. 285.    ==
 * == (an older Numerical Methods book,  but a good one).            ==
 * == This method does not require that A be symetric, But it's      ==
 * == probably not efficient enough to be used for a very large      ==
 * == matrix (i.e. 100x100
C======================================================================
*/
void MatInverseGaussJordan(double **A, double **AInv, double **AP, int N)
{
int		ColCount,
		I,
		J;

double	UnitFac,
		RowFac;



	/* Set-Up AP, Copy A into AP */
	for (I = 1; I <= N; I++)
		for (J = 1; J <= N; J++)
			AP[I][J] = A[I][J];


	/* add extra column to AP, this is the first so AP(1,N+1) = 1 */
        AP[1][N+1] = 1.0;
		for (I = 2; I <= N; I++)
			AP[I][N+1] = 0.0;




	/*
	 * loop to make the element AP(ColCount,J) = 1,
	 * all other elements in that column = 0
	 */
	for (ColCount = 1; ColCount <= N; ColCount++)
	{
		UnitFac = AP[ColCount][1];
		
		for (J = 1; J <= N+1; J++)
			AP[ColCount][J] = AP[ColCount][J]/UnitFac;


		for (I = 1; I <= N; I++)
		{
            RowFac = AP[I][1];
            J = 1;

			if ( I != ColCount)
				for (J = 1; J <= N+1; J++)
					AP[I][J] = AP[I][J] - AP[ColCount][J]*RowFac;

		}



		/* shift rows to the left, make the last row a column
		 * of the identity matrix so that element
		 * AP(ColCount+1,N+1) = 1 (i.e. the next column)
		 */
		for (I = 1; I <= N; I++)
		{
			for (J = 1; J <= N; J++)
				AP[I][J] = AP[I][J+1];

			AP[I][N+1] = Delta(I, ColCount+1);
		}

	}



	/* the N rows & N columns of AP are the inverse of A, store in AInv */
	for (I = 1; I <= N; I++)
		for (J = 1; J <= N; J++)
			AInv[I][J] = AP[I][J];


}


/*=====================================================================
 * ==                                                                ==
 * ==                            funtion CGLS                        ==
 * ==                                                                ==
 * ==     ==
 * ==            ==
 * ==       ==
 * ==       ==
 * == 
C======================================================================
*/
void CGLS(double *SolutionVecM, double **MatG, double *RHSVecD,
			int nRow, int nCol)
{
int	k;

double	alpha,
		beta,
		magnitude,
		*p,
		*pLast,
		*s,
		*sLast,
		*rLast,
		*r,
		*Gp,
		*SolutionVecMLast,
		*SolutionVecMDifferance,
		**TransposeMatG;

	/*
	 * allocate memory for the matrices
	 */
	Gp = dvector(nRow);
	p = dvector(nCol);
	SolutionVecMLast = dvector(nCol);
	SolutionVecMDifferance = dvector(nCol);
	pLast =  dvector(nCol);
	rLast = dvector(nCol);
	r = dvector(nCol);
	s = dvector(nRow);
	sLast = dvector(nRow);
	TransposeMatG = dmatrix(nCol, nRow);


	Transpose(TransposeMatG, MatG, nRow, nCol);

	/* Initialize solution vector to zero */
	for(k = 0; k < nCol; k++)
	{
		SolutionVecM[k] = 0;
		SolutionVecMDifferance[k] = 1.0;
	}

	EquateVectors(SolutionVecMLast, SolutionVecM, nCol);
	beta = 0;
	EquateVectors(s, RHSVecD, nRow);
	EquateVectors(sLast, s, nRow);
	MatrixVectorProduct(r, TransposeMatG, s, nCol, nRow);
	EquateVectors(rLast, r, nCol);
	/* Initialize pLast vector to zero */
	for(k = 0; k < nCol; k++)
	{
		pLast[k] = 0;
	}

	/* Iterate to get the solution */
	k = 1;
	magnitude = 100;
	while((k < 40) && (magnitude>TOL) )
	{
		EquateVectors(pLast, p, nCol);
		VectorSumWithScalarMult(p, r, beta, p, nCol);
		MatrixVectorProduct(Gp, MatG, p, nRow, nCol);
		alpha = dotProduct(r, r, nCol)/dotProduct(Gp, Gp, nRow);
		EquateVectors(SolutionVecMLast, SolutionVecM, nCol);
		VectorSumWithScalarMult(SolutionVecM, SolutionVecM, alpha, p, nCol);
		EquateVectors(sLast, s, nRow);
		VectorSumWithScalarMult(s, s, -alpha, Gp, nRow);
		beta = dotProduct(r, r, nCol)/dotProduct(rLast, rLast, nCol);
		EquateVectors(rLast, r, nCol);
		MatrixVectorProduct(r, TransposeMatG, s, nCol, nRow);

		/* Diagnostic### */
		/* PrintVector(SolutionVecM, nCol); */
		
		/*	setup difference vector between the solution and solution last
			to see if we have convergance
		*/
		VectorSumWithScalarMult(SolutionVecMDifferance,
			SolutionVecM, -1, SolutionVecMLast, nCol);
		k++;
		magnitude = sqrt(dotProduct(SolutionVecMDifferance,
							SolutionVecMDifferance, nCol));

	}


	/* free the matrix memory */
	free_dmatrix(TransposeMatG, nCol, nRow);
	free_dvector(Gp, nRow);
	free_dvector(p, nCol);
	free_dvector(pLast, nCol);
	free_dvector(s, nRow);
	free_dvector(sLast, nRow);
	free_dvector(rLast, nCol);
	free_dvector(r, nCol);
	free_dvector(SolutionVecMLast, nCol);
	free_dvector(SolutionVecMDifferance, nCol);


}


/*=====================================================================
 * ==                                                                ==
 * ==                           funtion Transpose                    ==
 * ==                                                                ==
 * == This code finds calculates the transpose of a matrix     ==
 * ======================================================================
*/
void Transpose(double **TransposeMat, double **Mat,	int nRow, int nCol)
{
int	matRow,
	matColumn;

	for (matRow = 0; matRow < nRow; matRow++)
	{
		for (matColumn = 0; matColumn < nCol; matColumn++)
		{
			TransposeMat[matColumn][matRow] = Mat[matRow][matColumn];
		}
	}


}


/*=====================================================================
 * ==                                                                ==
 * ==                        funtion EquateVectors                   ==
 * ==                                                                ==
 * == This code equates two vectors     ==
 * ======================================================================
*/
void EquateVectors(double *VectorEmpty, double *VectorFilled, int nEntries)
{
int	k;

	for (k = 0; k < nEntries; k++)
	{
		VectorEmpty[k] = VectorFilled[k];
	}


}


/*=====================================================================
 * ==                                                                ==
 * ==                        funtion PrintVector                     ==
 * ==                                                                ==
 * == This code equates two vectors     ==
 * ======================================================================
*/
void PrintVector(double *Vector, int nEntries)
{
int	k;

	for (k = 0; k < nEntries; k++)
	{
		printf(" %lf   ",Vector[k]);
	}
	printf("\n");

}


/*=====================================================================
 * ==                                                                ==
 * ==                        funtion dotProduct                      ==
 * ==                                                                ==
 * == This code equates two vectors     ==
 * ======================================================================
*/
double dotProduct(double *vector1, double *vector2, int nEntries)
{
int	k;

double	dotproduct=0;

	for (k = 0; k < nEntries; k++)
	{
		dotproduct = dotproduct + vector1[k]*vector2[k];
	}

return dotproduct;
}


/*=====================================================================
 * ==                                                                ==
 * ==                        funtion VectorSumWithScalarMult         ==
 * ==                                                                ==
 * == This code adds a vector to a scalar x vector     ==
 * ======================================================================
*/
void VectorSumWithScalarMult(double *solution, double *vector1, double scalar,
double *vector2, int nEntries)
{
int	k;

	for (k = 0; k < nEntries; k++)
	{
		solution[k] = vector1[k] + scalar*vector2[k];
	}


}



/* ========================================================================
 * ==                                                                    ==
 * ==                functions dmatrix &  free_dmatrix                   ==
 * ==                                                                    ==
 * == Functionality: dmatrix() & free_dmatrix() are memory management    ==
 * == functions used to allocate and free a double precision array.      ==
 * == They are based on the Num. Recipes routines with slight            ==
 * == modifications.                                                     ==
 * ==                                                                    ==
 * == Diagnostics: None                                                  ==
 * ==                                                                    ==
 * ========================================================================
 */
void free_dmatrix(double **m, int nr, int nc)
{
int	i;

	for(i=0; i< nr; i++) free(m[i]);

	free(m);
}


double **dmatrix(int nrow, int ncol)
{
int	i;

double	**m;

	m=(double **) calloc((size_t) nrow, sizeof(double*));
	if (!m) printf("allocation failure 1 in dmatrix()\n");

	for(i=0;i<nrow;i++) {
		m[i]=(double *) calloc((size_t) ncol, sizeof(double));
		if (!m[i]) printf("allocation failure 2 in dmatrix()\n");
	}
	return m;
}

void free_dvector(double *v, int nrow)
{

	free(v);
}

void free_ivector(int *v, int nrow)
{

	free(v);
}


double *dvector(int nrow)
{
double	*v;

	v=(double *) calloc((size_t) nrow, sizeof(double));
	if (!v) printf("allocation failure in dvector()\n");
	return v;
}

int *ivector(int nrow)
{
int	*v;

	v=(int *) calloc((size_t) nrow, sizeof(int));
	if (!v) printf("allocation failure in ivector()\n");
	return v;
}







/*  elelmentToVary stars at 0 
    the retuned integer just tells if there was a failure, i.e. h == 0
*/
int dfridr(void (*func)(double *, double *), double *x, double h,
		int numElements, int elementToVary, int nDataPoints, double *dF, double *err)

{
int	i,
	j,
	k,
	*derCalced;

double	errt,
	fac,
	hh,
	**a,
	*xphh,
	*xmhh,
	**dFphh,
	**dFmhh,
	ans;


	if (h == 0.0)
	{
	 	printf("h must be nonzero in dfridr.  See line %d in file %s\n",__LINE__, __FILE__);
		return (EXIT_FAILURE);
	}

	xphh = dvector(numElements);
	xmhh = dvector(numElements);
	derCalced = ivector(NTAB+1);
	for (i=1;i<=NTAB;i++)
	{
		derCalced[i] = NOT_CALC_ED;
	}

	/* dFphh and dFmhh are arrays that hold the function values where
	   the rows hold the number of data ponts (i.e. maybe thing function of time)
	   and columns hold the evaluations in the
	   neighborhood of the particular x value
	 */
	dFphh = dmatrix(NTAB+1, nDataPoints);
	dFmhh = dmatrix(NTAB+1, nDataPoints);
	a = dmatrix(NTAB+1, NTAB+1);



	/* first we need to fill the arrays that will hold the function calls ...
	   We do first and second ... hh and hh/CON because we will definetly
	   use these.  We only do others if we need.
	 */
	hh=h;
	EquateVectors(xphh, x, numElements);
	EquateVectors(xmhh, x, numElements);
	xphh[elementToVary] = x[elementToVary] + hh;
	xmhh[elementToVary] = x[elementToVary] - hh;

	/* onld NR start with 1 */
	(*func)(xphh,dFphh[1]);
	(*func)(xmhh, dFmhh[1]);
	derCalced[1] = CALC_ED;

	/* Now get dFphh[2] and dFmhh[2]
	 */
	hh /= CON;
	xphh[elementToVary] = x[elementToVary] + hh;
	xmhh[elementToVary] = x[elementToVary] - hh;
	(*func)(xphh,dFphh[2]);
	(*func)(xmhh, dFmhh[2]);
	derCalced[2] = CALC_ED;




	/* loop over number of data points in dF */
	for(k=0; k < nDataPoints; k++)
	{


	hh=h;
	EquateVectors(xphh, x, numElements);
	EquateVectors(xmhh, x, numElements);
	xphh[elementToVary] = x[elementToVary] + hh;
	xmhh[elementToVary] = x[elementToVary] - hh;

	/* carry over from old NR, start with element 1! */
	a[1][1]=(dFphh[1][k]-dFmhh[1][k])/(2.0*hh);
	*err=BIG;

	for (i=2;i<=NTAB;i++)
	{
		hh /= CON;
		xphh[elementToVary] = x[elementToVary] + hh;
		xmhh[elementToVary] = x[elementToVary] - hh;
		/* if we need these derivatives, calculate them now */
		if (derCalced[i] == NOT_CALC_ED)
		{
			(*func)(xphh,dFphh[i]);
			(*func)(xmhh, dFmhh[i]);
			derCalced[i] = CALC_ED;
		}
		a[1][i]=(dFphh[i][k]-dFmhh[i][k])/(2.0*hh);
		fac=CON2;
		for (j=2;j<=i;j++)
		{
			a[j][i]=(a[j-1][i]*fac-a[j-1][i-1])/(fac-1.0);
			fac=CON2*fac;
			errt=FMAX(fabs(a[j][i]-a[j-1][i]),fabs(a[j][i]-a[j-1][i-1]));
			if (errt <= *err)
			{
				*err=errt;
				ans=a[j][i];
			}
		}
		if (fabs(a[i][i]-a[i-1][i-1]) >= SAFE*(*err))
		{

			/* Break out of loop by setting i = NTAB+1 ...
			   last ans was best ans
			 */
			 i = NTAB + 1;
		}
	}
	dF[k] = ans;


	} /* end k loop */


	free_dmatrix(a, NTAB+1, NTAB+1);
	free_dmatrix(dFphh, NTAB+1, nDataPoints);
	free_dmatrix(dFmhh, NTAB+1, nDataPoints);

	free_dvector(xphh, numElements);
	free_dvector(xmhh, numElements);
	free_ivector(derCalced, NTAB+1);

/*for(k=0; k < nDataPoints; k++)
{
printf("dF[%d] = %f\n",k,dF[k]);
}*/

return (EXIT_SUCCESS);
}




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
double calcChi2(double *Obs, double *Calc, int nDataPoints)
{
double	chi2=0.0;

int	i;

	for(i = 0; i < nDataPoints; i++)
	{
		chi2 = chi2 + (Obs[i] - Calc[i])*(Obs[i] - Calc[i]);
	}

return chi2;
}
