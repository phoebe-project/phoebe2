#include <Python.h>
#include <numpy/arrayobject.h>

static char const *interp_docstring =
    "This module wraps the interpolation function.";

int flt(double target, double *arr, int numElems)
{
	/**
	 * flt:
	 * @target: comparison value
	 * @arr:    array to be searched
	 * @numElems: array length
	 * 
	 * Finds the first element in the sorted array @arr that is larger
	 * than the comparison value @target. The function uses a binary
	 * search algorithm. The name 'flt' stands for first larger than.
     * 
     * Returns: index of the first element larger than @target, or -1
     * if @target is out of bounds.
	 */
	
    int low = 0, high = numElems, mid;

    /* We only need to test the upper boundary; the lower boundary is
     * breached if 0 is returned. The calling functions thus must test
     * against flt index < 1. */
    
    if (target > arr[numElems-1])
        return -1;
    
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
	 * @args: passed arguments to the function.
     * 
	 * Multi-dimensional linear interpolation of arrays.
     * 
     * The @args list expects three arguments: @req_obj, @axes and
     * @grid_obj. Given N axes and M points of interest, @req_obj is an
     * NxM numpy array (N columns, M rows) where each column stores the
     * value along the respective axis and each row corresponds to a
     * single point to be interpolated; @axes is a tuple of numpy
     * arrays, with each array holding all unique vertices along its
     * respective axis, and @grid_obj is a N1xN2x...xNNx1 numpy array
     * where Ni are lengths of individual axes, and the last element
     * is the vertex value.
     * 
     * Example: we have the following vertices with corresponding values:
     * 
     *   v0 = (0, 2), f(v0) = 5
     *   v1 = (0, 3), f(v0) = 6
     *   v2 = (1, 3), f(v0) = 7
     *   v3 = (1, 2), f(v0) = 8
     * 
     * We are interested in f(0.5, 2.5) and f(0.75, 2.25). Note that
     * all values need to be floats, thus:
     * 
     *   req_obj = np.array(((0.5, 2.5), (0.75, 2.25)))
     *   axes = (np.array((0.0, 1.0)), np.array((2.0, 3.0)))
     *   grid_obj = np.array(( ((5.0,), (6.0,)), ((7.0,), (8.0,))))
     * 
     * Returns: Mx1 numpy array of interpolated values.
	 */

    PyObject *req_obj, *grid_obj;
    PyObject *req_arr, *grid_arr;
    PyObject *axes, *axptr;
    PyObject *ret_arr;
    double **ax, **n, **fvv;
    int *axlen, *axidx, *powers;
    double *req, *grid, *lo, *hi, *retval;
    int i, j, k, l, idx, numAxes, numPts, numVals, numFVs, out_of_bounds;
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
	ax = (double**) malloc(numAxes*sizeof(*ax));
	axlen = (int*) malloc(numAxes*sizeof(*axlen));
	axidx = (int*)malloc(numAxes*sizeof(*axidx));
	lo = (double*)malloc(numAxes*sizeof(*lo));
	hi = (double*)malloc(numAxes*sizeof(*hi));
	prod = (double*)malloc(numAxes*sizeof(*prod));

	powers = (int*)malloc((numAxes+1)*sizeof(*powers));
	
  powers[0] = 1;
	for (i = 1; i < numAxes+1; i++)
		powers[i] = powers[i-1]*2;

	/* Allocate space to hold all the nodes: */
	n = (double**)malloc (powers[numAxes]*sizeof(*n));
	for (i = 0; i < powers[numAxes]; i++)
		n[i] = (double*)malloc (numAxes*sizeof(**n));

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
	fvv = (double**)malloc(numFVs*sizeof(*fvv));
	for (i = 0; i < numFVs; i++)
		fvv[i] = (double*)malloc(numVals*sizeof(**fvv));
	retval = (double*) malloc(numVals*numPts*sizeof(*retval));

	//~ printf("Interpolation geometry:\n");
	//~ printf("  numAxes = %d     # number of axes that span interpolation space.\n", numAxes);
	//~ printf("  numPts  = %d     # number of individual points to be interpolated.\n", numPts);
	//~ printf("  numVals = %d     # number of values per point to be interpolated.\n", numVals);
	//~ printf("  numFVs  = %d     # number of required function values per interpolation.\n", numFVs);

	/* The main loop: go through all requested vertices and find the
	 * corresponding values:
	 */
	for (i = 0; i < numPts; i++) {
        out_of_bounds = 0;
        
        /* Run the axes first to make sure interpolation is possible. */
        for (j = numAxes-1; j >= 0; j--) {
			axidx[j] = flt(req[i*numAxes+j], ax[j], axlen[j]);
            if (axidx[j] < 1)
                /* AN OUT-OF-BOUNDS SITUATION -- both sides handled. */
                out_of_bounds = 1;
        }

        /* Must do this here to be able to continue the main loop. */
        if (out_of_bounds) {
            for (l = 0; l < numVals; l++)
                retval[i*numVals+l] = sqrt(-1);
            continue;
        }
        
		for (j = numAxes-1; j >= 0; j--) {
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

	free(powers);
	for (i = 0; i < numFVs; i++) {
		free(n[i]);
		free(fvv[i]);
	}
	free(n);
	free(fvv);
    
    return ret_arr;
}

#if 0
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

#endif

static PyMethodDef interp_methods[] = {
    {"interp", interp, METH_VARARGS, "Interpolate function"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initinterp(void)
{
    PyObject *m = Py_InitModule3("interp", interp_methods, interp_docstring);
    if (!m)
        return;
    
    import_array();
}
