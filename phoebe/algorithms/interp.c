#include <Python.h>
#include <numpy/arrayobject.h>

static char *interp_docstring =
    "This module wraps the interpolation function.";

int first_larger_than (double target, double *arr, int numElems)
{
	/**
	 * first_larger_than:
	 * @target: comparison value
	 * @arr:    array to be searched
	 * @numElems: array length
	 * 
	 * Finds the first element in the sorted array @arr that is larger
	 * than the comparison value @target. The function uses a binary
	 * search algorithm.
	 */
	
    int low = 0, high = numElems, mid;
    
    while (low != high) {
        mid = (low + high) / 2;
        if (arr[mid] <= target)
            low = mid + 1;
        else
            high = mid;
    }

    return low;
}

static PyObject *interp(PyObject *self, PyObject *args)
{
	/**
	 * interp:
	 * @self: function reference
	 * @args: passed arguments to the function
	 * 
	 * Multi-dimensional linear interpolation of arrays.
	 */

    PyObject *req_obj, *grid_obj;
    PyObject *req_arr, *grid_arr;
    PyObject *axes, *axptr;
    PyObject *ret_arr;
    double **ax, **n, **fvv;
    int *axlen, *axidx, *powers;
    double *req, *grid, *lo, *hi, *retval;
    int i, j, k, l, idx, numAxes, numPts, numVals, numFVs;
	double *prod;
	npy_intp retdim[2];
    
    if (!PyArg_ParseTuple(args, "OOO", &req_obj, &axes, &grid_obj) || !PyTuple_Check(axes)) {
        printf("argument type mismatch: req and grid need to be numpy arrays and axes a tuple of numpy arrays.\n");
        return NULL;
    }
	
    req_arr = PyArray_FROM_OTF(req_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    grid_arr = PyArray_FROM_OTF(grid_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (!req_arr || !grid_arr) {
        if (!req_arr)
			printf("req_arr failed.\n");
		if (!grid_arr)
			printf("grid_arr failed.\n");

        Py_DECREF(req_arr);
        Py_DECREF(grid_arr);
        return NULL;
    }
	
	numAxes = PyTuple_Size(axes);
	ax = malloc(numAxes*sizeof(*ax));
	axlen = malloc(numAxes*sizeof(*axlen));
	axidx = malloc(numAxes*sizeof(*axidx));
	lo = malloc(numAxes*sizeof(*lo));
	hi = malloc(numAxes*sizeof(*hi));
	prod = malloc(numAxes*sizeof(*prod));

	powers = malloc ((numAxes+1)*sizeof (*powers));
	powers[0] = 1;
	for (i = 1; i < numAxes+1; i++)
		powers[i] = powers[i-1]*2;

	/* Allocate space to hold all the nodes: */
	n = malloc (powers[numAxes]*sizeof(*n));
	for (i = 0; i < powers[numAxes]; i++)
		n[i] = malloc (numAxes*sizeof(**n));

	/* Unpack the axes: */
	for (i = 0; i < numAxes; i++) {
		axptr = PyTuple_GET_ITEM(axes, i);
		axlen[i] = (int) PyArray_DIM(axptr, 0);
		ax[i] = (double *) PyArray_DATA(axptr);
		//~ ax[i] = (double *) PyArray_GetPtr(axptr);
		//~ printf("axis %d: ndim=%d, length=%d, itemsize=%d, flags=%d, dtype=%d.\n", i, (int) PyArray_NDIM(axptr), (int) PyArray_DIM(axptr, 0), (int) PyArray_ITEMSIZE(axptr), (int) PyArray_FLAGS(axptr), PyArray_DTYPE((PyArrayObject *) axptr)->type);
	}
	//~ printf("Axes values (first 3):\n");
	//~ for (i = 0; i < 3; i++)
		//~ printf("ax[%d][%d] = %f\n", 0, i, ax[0][i]);

	req = (double *) PyArray_DATA(req_arr);
	numPts = PyArray_DIM(req_arr, 0);

	//~ printf("Requested pts (first 5 elements):\n");
	//~ for (i = 0; i < 5; i++)
		//~ printf("req[%d] = %lf\n", i, req[i]);

	grid = (double *) PyArray_DATA(grid_arr);
	numVals = PyArray_DIM(grid_arr, PyArray_NDIM(grid_arr)-1);

	/* Allocate function value arrays: */
	numFVs = (int) pow(2, numAxes);
	fvv = malloc(numFVs*sizeof(*fvv));
	for (i = 0; i < numFVs; i++)
		fvv[i] = malloc(numVals*sizeof(**fvv));
	retval = malloc(numVals*numPts*sizeof(*retval));

	//~ printf("Interpolation geometry:\n");
	//~ printf("  numAxes = %d     # number of axes that span interpolation space.\n", numAxes);
	//~ printf("  numPts  = %d     # number of individual points to be interpolated.\n", numPts);
	//~ printf("  numVals = %d     # number of values per point to be interpolated.\n", numVals);
	//~ printf("  numFVs  = %d     # number of required function values per interpolation.\n", numFVs);

	/* The main loop: go through all requested vertices and find the
	 * corresponding values:
	 */
	for (i = 0; i < numPts; i++) {
		for (j = numAxes-1; j >= 0; j--) {
			axidx[j] = first_larger_than(req[i*numAxes+j], ax[j], axlen[j]);
			lo[j] = ax[j][axidx[j]-1]; hi[j] = ax[j][axidx[j]];
			prod[j] = (j==numAxes-1) ? 1.0 : prod[j+1]*axlen[j+1];
			//~ printf("idx=%02d, val=%f, arr[%d]=%f, arr[%d]=%f, prod[%d]=%f\n", axidx[j], req[i*numAxes+j], axidx[j]-1, lo[j], axidx[j], hi[j], j, prod[j]);
		}

		for (k = 0; k < numFVs; k++) {
			//~ printf("%2d ", k);
			idx = 0;
			for (j = 0; j < numAxes; j++) {
				idx += (axidx[j]-1+(k/(int)pow(2, j))%2)*prod[j];
				//~ printf("% 1.1f ", ax[j][axidx[j]-1+(k/(int)pow(2, j))%2]);
			}
			for (l = 0; l < numVals; l++)
				fvv[k][l] = grid[idx*numVals+l];
		}

		//~ for (j = 0; j < numAxes; j++)
			//~ printf("%lf\t%lf\t%lf\n", lo[j], req[i*numAxes+j], hi[j]);

		/* Populate the nodes: */
		for (k = 0; k < numAxes; k++)
			for (j = 0; j < powers[numAxes]; j++)
				n[j][k] = lo[k] + ((j/powers[k])%2)*(hi[k]-lo[k]);

		for (k = 0; k < numAxes; k++)
			for (j = 0; j < powers[numAxes-k-1]; j++)
				for (l = 0; l < numVals; l++)
					fvv[j][l] += ((&req[i*numAxes])[numAxes-k-1]-n[j][numAxes-k-1])/(n[j+powers[numAxes-k-1]][numAxes-k-1]-n[j][numAxes-k-1])*(fvv[j+powers[numAxes-k-1]][l]-fvv[j][l]);

		//~ printf("Interpolated value: %f\n", fv[0]);
		for (l = 0; l < numVals; l++)
			retval[i*numVals+l] = fvv[0][l];
	}

	retdim[0] = numPts; retdim[1] = numVals;
	ret_arr = PyArray_SimpleNewFromData(2, retdim, NPY_DOUBLE, (void *) retval);
	PyArray_UpdateFlags((PyArrayObject *) ret_arr, NPY_ARRAY_OWNDATA);

	/* Free all the arrays we don't need anymore. */
    free(lo);
    free(hi);
	free(prod);

	/* Do not free any of the ax[i]; they're all borrowed pointers! */
    free(ax);

    free(axlen);
    free(axidx);

	free (powers);
	for (i = 0; i < numFVs; i++) {
		free (n[i]);
		free(fvv[i]);
	}
	free (n);
	free(fvv);
    
    return ret_arr;
}

static PyObject *reference_function_for_decrefs_otherwise_useless(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *ll_obj, *ul_obj, *fv_obj;
    PyObject *x_arr, *ll_arr, *ul_arr, *fv_arr;
    double *x, *ll, *ul, *fv;
    int i, j, N, Npts;

    if (!PyArg_ParseTuple(args, "OOOO", &x_obj, &ll_obj, &ul_obj, &fv_obj))
        return NULL;
    
    x_arr  = PyArray_FROM_OTF(x_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
    ll_arr = PyArray_FROM_OTF(ll_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    ul_arr = PyArray_FROM_OTF(ul_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    fv_arr = PyArray_FROM_OTF(fv_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    if (!x_arr || !ll_arr || !ul_arr | !fv_arr) {
        Py_DECREF(x_arr);
        Py_DECREF(ll_arr);
        Py_DECREF(ul_arr);
        Py_DECREF(fv_arr);
        return NULL;
    }

    /* Now we need to check the dimensionality of arrays: */
    switch (PyArray_NDIM(x_arr)) {
        case 1:
            /* This means that we have a single point to be interpolated. */
            N  = PyArray_DIM(x_arr, 0);
            x  = (double *) PyArray_DATA(x_arr);
            ll = (double *) PyArray_DATA(ll_arr);
            ul = (double *) PyArray_DATA(ul_arr);
            fv = (double *) PyArray_DATA(fv_arr);

            // interpolate(N, x, ll, ul, TYPE_DOUBLE, fv);

            Py_DECREF(x_arr);
            Py_DECREF(ll_arr);
            Py_DECREF(ul_arr);
            Py_DECREF(fv_arr);

            return Py_BuildValue("f", fv[0]);
        break;
        case 2:
            /* This means that we have an array of points to be interpolated. */
            Npts = PyArray_DIM(x_arr, 0);
            N = PyArray_DIM(x_arr, 1);
            x  = (double *) PyArray_DATA(x_arr);
            ll = (double *) PyArray_DATA(ll_arr);
            ul = (double *) PyArray_DATA(ul_arr);
            fv = (double *) PyArray_DATA(fv_arr);

            for (j = 0; j < Npts; j++) {
                for (i = 0; i < N; i++) {
                    printf("%f ", (&x[j*N])[i]);
                }
                for (i = 0; i < N; i++) {
                    printf("%f ", (&ll[j*N])[i]);
                }
                for (i = 0; i < N; i++) {
                    printf("%f ", (&ul[j*N])[i]);
                }
                for (i = 0; i < 4; i++)
                    printf("%f ", (&fv[j*4])[i]);

                // interpolate(N, &x[j*N], &ll[j*N], &ul[j*N], TYPE_DOUBLE, &fv[j*(int)pow(2,N)]);

                printf("%f\n", fv[j*(int)pow(2,N)]);
            }

            Py_DECREF(x_arr);
            Py_DECREF(ll_arr);
            Py_DECREF(ul_arr);
            Py_DECREF(fv_arr);

            return Py_BuildValue("");
        break;
        default:
            printf("How'd I get here?\n");

            Py_DECREF(x_arr);
            Py_DECREF(ll_arr);
            Py_DECREF(ul_arr);
            Py_DECREF(fv_arr);

            return Py_BuildValue("");
    }
    //~ printf("dims: %d\n", PyArray_NDIM(x_arr));
    //~ printf("length 0: %d\n", PyArray_DIM(x_arr, 0));
    //~ printf("length 1: %d\n", PyArray_DIM(x_arr, 1));

    
}

static PyMethodDef interp_methods[] = {
    {"interp",                   interp, METH_VARARGS, "Interpolate function"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initinterp(void)
{
    PyObject *m = Py_InitModule3("interp", interp_methods, interp_docstring);
    if (!m)
        return;
    
    import_array();
}
