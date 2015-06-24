/* ** ***********************************************************************
 * **                           _DiffCorr.c                                **
 * **                                                                      **
 * ** This module contains the code that allows the differential correction
 * ** (see DiffCorr.c) algoritum to be called from  python.  The calling 
 * ** sequence from python is...
        parameterSolution = _DiffCorr.DiffCorr(nParams,  \
                                nDataPoints, \
                                maxIterations, \
                                stoppingCriteriaType, \
                                stopValue, \
                                data, \
                                initialGuess, \
                                derivativeType, \
                                dFunc_dx, \
				corMat,
				sigmaMat
                                lFunc, \
                                system)

 * ** parameterSolution - a numpy array returned by the c-code that holds
 * **                     the values of the model parameters that give the
 * **                     best agreement to the data.
 * **
 * ** nParams - the number of parameters to be found
 * **
 * ** nDataPoints - the number of data points passed to be fit
 * ** 
 * ** maxIterations - the maximum nuber of iterations to allow DC to perform
 * **
 * ** stoppingCriteriaType - the criteria to use to stop DC.  Currelty 
 * **                        implemented are:
 * **   MIN_DX ... when the magnitude of the step size, the change
                   in length of the parameter array, dx, x_new = x_last + dx,
		   becomes less than stopValue iterations stop
        MIN_DELTA_DX ... when the magnitude of the *CHANGE IN* step size,
                   dxNew - dxLast, becomes less than stopValue iterations stop
        MIN_CHI2 ...  when the value of chi^2, defined as
                   Sum[(model_i - data_i)^2], becomes less than stopValue
                   iterations stop
	MIN_DELTA_CHI2 ... when the magnitude of the *CHANGE IN* chi^2,
                    Abs[chi_New^2 - chi_Last^2] becomes less than stopValue
                   iterations stop
 * **   NOTE WELL: in the python calling procedure the above are integers
 	# stopping criteria type
	MIN_DX = 0
	MIN_DELTA_DX = 1
	MIN_CHI2 = 2
	MIN_DELTA_CHI2 = 3
 * ** in the c-code they are enums.
 * **
 * **                        implemented are:
 * ** stopValue - given the stopping criteria above, the value to stop iterations
 * **
 * ** data - a numpy array containing the data the model is to fit
 * **
 * ** initialGuess - the values of the model parameters to use to start
 * **                the DC algorithum.  For DC to work these values MUST
 * **                must be reasonably close to the solution
 * **
 * ** derivativeType - DC uses derivative information.  This variable
 * **                  is a numpy array, each element tells DC what type of 
                       derivative to use for each parameter.  Possible values are
	NUMERICAL - the c-code will use an algorithum, similar to dfridr
	            in Numerical Recipes, that will calculate a numerical
		    derivative  via function calls.  It can be expensive.
	ANALYTICAL - the python code must supply a function, in dFunc_dx,
	             to call to evaluate the derivative.
	NONE - a value of NONE means this parameter's value is to be held fixed
	       The value supplied in the initialGuess array will be used
 * ** this array will look something like: 
      derivativeType = np.array([ANALYTICAL, ANALYTICAL, ANALYTICAL], dtype=np.int32)
 * ** the values must be np.int32 to fit through the calling procedure correctly
 * ** the values can be varied as in
      derivativeType = np.array([ANALYTICAL, NUMERICAL, NONE], dtype=np.int32)
 * **   NOTE WELL: in the python calling procedure the above are integers
	#derivative type
	NUMERICAL = 0
	ANALYTICAL = 1
	NONE = 2
 * ** in the c-code they are enums.
 * **   
 * **
 * ** dFunc_dx - a python **LIST** (not array) that contains the functions
 * **           names to be used to calculate analytical derivatives.
 * **    looks something like: 
	# set the derivative functions as a list
	dFunc_dx = [d_function_dx, d_function_dy, d_function_dz]
 * **   NOTE WELL: the calling sequence must be of the form
 * **      def d_function_dx(x, system):
 * **         x = array of parameter values
 * **         system = python class containing anything the function needs to
 * **                  calculate the model's derivative.  It must return 
 * **                  all the values, for example, the values for all times.
 * **                  the c-code expects a numpy array...
 * **                  value = np.arange(length,dtype=np.float64)
 * **
 * **
 * ** lFunc - the model (function) to calculate the light curve.  Same calling
 * **         calling sequence/return rules as the derivative functions
 * **
 * ** system - a python class (or any python object) containing variables, etc
 * **          that the function(s) above would need to calculate the light
 * **           curve or derivatives.
 * **
 * **  Created      : June. 2014                                           **
 * **  Last Modified: Aug. 14, 2014                                        **
 * ** ***********************************************************************
 */


#include <Python.h>
/*#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION*/

#include <numpy/arrayobject.h>

#include <math.h>
#include "DiffCorr.h"


/* Not a fan of global variables, but seemed cleanest to make
   the python functions to call from C global PyObject variables
*/
PyObject	*py_dFunc,
		*py_lFunc,
		*py_system;

int		py_nParams,
		py_NDataPoints;




/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for a Differential Correction routine written  C.";
static char DiffCorr_docstring[] =
    "Applies the DC algorithm for find the parameters in a model.";

/* Available functions */
static PyObject *DiffCorr_DiffCorr(PyObject *self, PyObject *args);



/* Module specification */
static PyMethodDef module_methods[] = {
    {"DiffCorr", DiffCorr_DiffCorr, METH_VARARGS, DiffCorr_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_DiffCorr(void)
{
    PyObject *m = Py_InitModule3("_DiffCorr", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}





static PyObject *DiffCorr_DiffCorr(PyObject *self, PyObject *args)
{

double	*params,
	typenum = NPY_DOUBLE;

int	nRow,
	nCol,
	I,
	i,
	numFixedParams=0,
	error;

npy_intp 	dims[1],
		matDims[3];



PyObject	*py_data,
		*py_initialGuess,
		*py_derivativeType,
		*py_corMat,
		*py_sigmaMat;

PyArrayObject	*py_paramSolution; /* This will be used to pass the parameter
					solution back to python */


PyArray_Descr *descr;

DifferentialCorrection *diffCorr;



	diffCorr =(DifferentialCorrection *)
		calloc((size_t) 1, sizeof(DifferentialCorrection));





	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "iiiidOOOOOOOO", 
    				&(diffCorr->nParams),
				&(diffCorr->nDataPoints),
    				&(diffCorr->maxIterations),
    				&(diffCorr->stoppingCriteriaType),
				&(diffCorr->stopValue),
				&py_data,
				&py_initialGuess,
				&py_derivativeType,
				&py_dFunc,
				&py_corMat,
				&py_sigmaMat,
				&py_lFunc,
				&py_system))
        return NULL;


	/* This is global so we wuill know how many parameters
	   there are when we call the python functions from c
	 */
	py_nParams = diffCorr->nParams;
	py_NDataPoints = diffCorr->nDataPoints;




	// make sure lFunc is callable
	if (!PyCallable_Check(py_lFunc))
	{
		PyErr_SetString(PyExc_TypeError, "Need a callable object for the light curve calculation function!");
		return NULL;
	}

	/* get the derivative types */
	diffCorr->derivativeType = (int *)PyArray_DATA(
			PyArray_FROM_OTF(py_derivativeType, NPY_INT32, NPY_IN_ARRAY ));

	for(i=0;i<diffCorr->nParams; i++)
	{
		if(diffCorr->derivativeType[i] != NONE)
		{
			/* Check that derivative function is callable */
			if (!PyCallable_Check( PyList_GetItem(py_dFunc, i) ))
			{
				PyErr_SetString(PyExc_TypeError,
					"Need a callable object derivative function!");
				return NULL;
			}
	
		}
		else numFixedParams++;
			
	}

	/* check that we haven't been given all parameters fixed */
	if(numFixedParams == diffCorr->nParams)
	{
		printf("All Parameters Fixed???...problem in file %s line %d\n",__FILE__, __LINE__);
		exit (EXIT_FAILURE);
	}


	/* #### think about this, is it redundent ##### */
	nRow = diffCorr->nDataPoints;
	nCol = diffCorr->nParams;


	/* initial guess vector set in calling function */
	diffCorr->initialGuess = (double*)PyArray_DATA(
		PyArray_FROM_OTF(py_initialGuess, NPY_DOUBLE, NPY_IN_ARRAY));

	diffCorr->data = (double*)PyArray_DATA(
		PyArray_FROM_OTF(py_data, NPY_DOUBLE, NPY_IN_ARRAY));

	/* get the sigmaMat, really an array whose elements are 1/sigma_i^2 */
	diffCorr->sigmaMat = (double*)PyArray_DATA(
		PyArray_FROM_OTF(py_sigmaMat, NPY_DOUBLE, NPY_IN_ARRAY));

	/* get the covariance matrix that will be calculated by C and returned*/
	descr = PyArray_DescrFromType(typenum);
	if (PyArray_AsCArray(&py_corMat, (void **)&(diffCorr->corMat), matDims, 2, descr) < 0 )
	{
		PyErr_SetString(PyExc_TypeError, "error converting to c array");
		return NULL;
	}
	

	/*
	 * allocate memory for the matrices the c-routine will manipulate and use
	 */
	diffCorr->model = dvector(nRow);
	diffCorr->dL = dvector(nRow);
	/* pythpn controls ... diffCorr->params = dvector(diffCorr->nParams);*/
	diffCorr->paramsLast = dvector(diffCorr->nParams);
	diffCorr->paramsDifferance = dvector(diffCorr->nParams);
	diffCorr->SolutionVecM = dvector(nCol);


	dims[0] = diffCorr->nParams;
	py_paramSolution = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);


	/* set the params array to point to the data of the pyObject.
	   this is how we will get the answer back to the python calling routine
	*/
	params = (double*)PyArray_DATA(py_paramSolution);
	diffCorr->params = params;



	/* get from python ...
	diffCorr->dFunc_dx =
		(derFunctions *) calloc((size_t) diffCorr->nParams, sizeof(derFunctions));
	diffCorr->dFunc_dx[0] = d_function_dx;
	diffCorr->dFunc_dx[1] = d_function_dy;
	diffCorr->dFunc_dx[2] = d_function_dz;
*/




	/* Call DiffCorr to obtain the solution */
	/* ###############***************************################## */
	/* ###############***************************################## */
	error = 0; /* this is a hook for future use */
	DiffCorr(diffCorr , &error);





	/* for checking for now !! ####### */
	/*printf("  Solution = \n");
	for (I = 0; I < diffCorr->nParams; I++)
	{
		printf("%f\n", ((double *)(py_paramSolution->data))[I]);
	}*/


	/* free the matrix memory */
	free_dvector(diffCorr->SolutionVecM, nCol);
	


	free_dvector(diffCorr->model, nRow);
	free_dvector(diffCorr->dL, nRow);
	free_dvector(diffCorr->paramsDifferance, diffCorr->nParams);
	free_dvector(diffCorr->paramsLast, diffCorr->nParams);
	free_dvector(diffCorr->initialGuess, diffCorr->nParams);
	free_ivector(diffCorr->derivativeType, diffCorr->nParams);

	/* we do not free these vectors, python takes care of that 
	free_dvector(diffCorr->data, nRow);
	free(diffCorr->dFunc_dx);
	diffCorr->data = dvector(nRow);
	free_dvector(diffCorr->params, diffCorr->nParams);
	*/


	free(diffCorr);


	return PyArray_Return(py_paramSolution);

}




/* model is the calculated/theoretical "data" our mathematical model gives us
 * when we supply the values of the parameters stored in x
 */
void lFunc(double *x, double *model)
{
double		*temp_x,
		*temp_model;

PyObject	*arglist;

PyArrayObject	*py_xx,
		*py_model;

int		i;

npy_intp	dims[1];



/* #### Old diagnstic...get ride of soon 
	if (!PyCallable_Check(py_lFunc))
	{
		PyErr_SetString(PyExc_TypeError, "Need a callable object in function!");
		printf("can't call lc function\n");
	}
*/


	/* Build up the argument list... */

	dims[0] = py_NDataPoints;
	py_xx = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	temp_x = (double*)PyArray_DATA(py_xx);
	for(i = 0; i < py_NDataPoints; i++)
	{
		temp_x[i] = x[i];
	}


	arglist = Py_BuildValue("(OO)", py_xx, py_system);

	/* ...for calling the Python  function.*/
	py_model = (PyArrayObject *)PyEval_CallObject(py_lFunc,arglist);

	if (py_model)
	{
		temp_model = (double*)PyArray_DATA(py_model);
	}
	else printf("result  failed in lc calc. wrapper\n");

	/* Now copy the returned array into the model array.  temp_model goes away*/
	for(i = 0; i < py_NDataPoints; i++)
	{
		model[i] = temp_model[i];
	}

	/* Do we need to DEREF py_model and arglist py_xx ???? */
}




void dFunc(double *x, int eleIndex, double *dF)
{
double		*temp_x,
		*temp_dF;

PyObject	*arglist;

PyArrayObject	*py_xx,
		*py_dF;

int		i;

npy_intp	dims[1];


	/* Build up the argument list... */
	dims[0] = py_NDataPoints;
	py_xx = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	temp_x = (double*)PyArray_DATA(py_xx);
	for(i = 0; i < py_NDataPoints; i++)
	{
		temp_x[i] = x[i];
	}
	arglist = Py_BuildValue("(OO)", py_xx, py_system);

	/* ...for calling the Python  function.*/
	py_dF = (PyArrayObject *)PyEval_CallObject(PyList_GetItem(py_dFunc, eleIndex),arglist);

	if (py_dF)
	{
		temp_dF = (double*)PyArray_DATA(py_dF);
	}
	else printf("Failure in derivative calc wrapper, py_dF not defined\n");


	/* Now copy the returned array into the model array.  temp_dF goes away*/
	for(i = 0; i < py_NDataPoints; i++)
	{
		dF[i] = temp_dF[i];
	}

	/* Do we need to DEREF py_dF and arglist py_xx ???? */

}
