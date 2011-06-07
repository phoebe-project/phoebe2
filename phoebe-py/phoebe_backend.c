#include <Python.h>
#include <phoebe/phoebe.h>


static PyObject *phoebeInit (PyObject *self, PyObject *args)
{
	int status = phoebe_init();
	if (status != SUCCESS) {
		printf ("%s", phoebe_error (status));
		return NULL;
	}

	return Py_BuildValue ("i", status);
}

static PyObject *phoebeConfigure (PyObject *self, PyObject *args)
{
	int status = phoebe_configure();
	if (status != SUCCESS) {
		printf ("%s", phoebe_error (status));
		return NULL;
	}

	return Py_BuildValue ("i", status);
}

static PyObject *phoebeParameter (PyObject *self, PyObject *args)
{
	/**
	 * phoebeParameter:
	 * 
	 * Packs all PHOEBE_parameter properties into a list to be parsed in
	 * python into the parameter class. The following fields are parsed:
	 *
	 *   parameter->qualifier
	 *   parameter->description
	 *   parameter->kind
	 *   parameter->format
	 */
	
	PyObject *list;
	PHOEBE_parameter *par;
	char *qualifier;
	int i;
	
	PyArg_ParseTuple (args, "s", &qualifier);
	par = phoebe_parameter_lookup (qualifier);
	if (!par) return Py_BuildValue ("");
	
	list = PyList_New (8);

	PyList_SetItem (list, 0, Py_BuildValue ("s", par->qualifier));
	PyList_SetItem (list, 1, Py_BuildValue ("s", par->description));
	PyList_SetItem (list, 2, Py_BuildValue ("i", par->kind));
	PyList_SetItem (list, 3, Py_BuildValue ("s", par->format));
	PyList_SetItem (list, 4, Py_BuildValue ("d", par->min));
	PyList_SetItem (list, 5, Py_BuildValue ("d", par->max));
	PyList_SetItem (list, 6, Py_BuildValue ("d", par->step));

	switch (par->type) {
		case TYPE_INT:
			PyList_SetItem (list, 7, Py_BuildValue ("i", par->value.i));
		break;
		case TYPE_BOOL:
			PyList_SetItem (list, 7, Py_BuildValue ("b", par->value.b));
		break;
		case TYPE_DOUBLE:
			PyList_SetItem (list, 7, Py_BuildValue ("d", par->value.d));
		break;
		case TYPE_STRING:
			PyList_SetItem (list, 7, Py_BuildValue ("s", par->value.str));
		break;
		case TYPE_INT_ARRAY: {
			int i;
			PyObject *array = PyList_New (par->value.array->dim);
			for (i = 0; i < par->value.array->dim; i++)
				PyList_SetItem (array, i, Py_BuildValue ("i", par->value.array->val.iarray));
			PyList_SetItem (list, 7, array);
		}
		break;
		case TYPE_BOOL_ARRAY: {
			PyObject *array = PyList_New (par->value.array->dim);
			for (i = 0; i < par->value.array->dim; i++)
				PyList_SetItem (array, i, Py_BuildValue ("b", par->value.array->val.barray));
			PyList_SetItem (list, 7, array);
		}
		break;
		case TYPE_DOUBLE_ARRAY: {
			PyObject *array = PyList_New (par->value.array->dim);
			for (i = 0; i < par->value.array->dim; i++)
				PyList_SetItem (array, i, Py_BuildValue ("d", par->value.array->val.darray));
			PyList_SetItem (list, 7, array);
		}
		break;
		case TYPE_STRING_ARRAY: {
			PyObject *array = PyList_New (par->value.array->dim);
			for (i = 0; i < par->value.array->dim; i++)
				PyList_SetItem (array, i, Py_BuildValue ("s", par->value.array->val.strarray));
			PyList_SetItem (list, 7, array);
		}
		break;
		default:
			/* If we end up here, yell and scream! */
			printf ("exception encountered in phoebe_backend.c, phoebeParameter().\n");
		break;
	}

	return list;
}

static PyMethodDef PhoebeMethods[] = {
	{"init",             phoebeInit,       METH_VARARGS, "Initialize PHOEBE backend"},
	{"configure",        phoebeConfigure,  METH_VARARGS, "Configure all internal PHOEBE structures"},
	{"parameter",        phoebeParameter,  METH_VARARGS, "Return a list of parameter properties"},
	{NULL,               NULL,             0,            NULL}
};

PyMODINIT_FUNC initphoebeBackend (void)
{
	PyObject *backend = Py_InitModule ("phoebeBackend", PhoebeMethods);
	if (!backend)
		return;
}
