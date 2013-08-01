#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "Python.h"
#include "numpy/arrayobject.h"
    
double min(double a, double b, double c)
{
    double m = fmin(a,b);
    return fmin(m,c);
}

double max(double a, double b, double c)
{
    double m = fmax(a,b);
    return fmax(m,c);
}

int sgn(double p11, double p12, double p21, double p22, double p31, double p32)
{
    double x = (p11 - p31) * (p22 - p32) - (p21 - p31) * (p12 - p32);
    
    if (x > 0.0) return 0;
    return 1;
}

int point_in_triangle(double p1, double p2, double v11, double v12, double v21, double v22, double v31, double v32, double bb[4])
{
    int b1, b2, b3;

    // First check if the point is outside the binding box of the triangle.
    // This is a lot cheaper and if it is outside, we now it can't be inside
    // the triangle.
    if (p1<bb[0] || p1>bb[1] || p2<bb[2] || p2>bb[3])
        return 0;

    // If it's not, then check if it's inside the triangle.
    b1 = sgn(p1, p2, v11, v12, v21, v22);
    b2 = sgn(p1, p2, v21, v22, v31, v32);
    b3 = sgn(p1, p2, v31, v32, v11, v12);

    if ((b1 == b2) && (b2 == b3))
        return 1;
    return 0;
}

int point_of_triangle(double p1, double p2, double v11, double v12, double v21, double v22, double v31, double v32, double tol)
{
    if (!((fabs(p1-v11) < tol && fabs(p2-v12) < tol) ||
          (fabs(p1-v21) < tol && fabs(p2-v22) < tol) ||
          (fabs(p1-v31) < tol && fabs(p2-v32) < tol)))
          return 0;
          
    return 1;
}

void vph(double **tab, int *st, double **sp, int N, double tol, int *visible, int *partial, int *hidden, int lengths[3])
{
    // Triangle bounding boxes
    double bbox[N][4];
    int ref[N][6];
    int hid[N][3];
    
    int vp[N]; // vis + par
    int vpi;
    int vtl = 1; // vis triangle length
    int ptl = 0; // par triangle length
    int htl = 0; // hid triangle length

    int i,j;
    
    double acp[5];
    int hp[3];
    int pirng[3];
    
    int rhp;
    int pit,pot;    
    double sm;
    
    for (i = 0; i < N; i++){
        bbox[i][0] = min(tab[i][0],tab[i][3],tab[i][6]);
        bbox[i][1] = max(tab[i][0],tab[i][3],tab[i][6]);
        bbox[i][2] = min(tab[i][1],tab[i][4],tab[i][7]);
        bbox[i][3] = max(tab[i][1],tab[i][4],tab[i][7]);
        
        for (j = 0; j < 6; j++)
            ref[i][j] = -1;
            
        for (j = 0; j < 3; j++)
            hid[i][j] = -1;
    }
    
    vp[0] = st[0];
    visible[0] = st[0];
    ref[0][0] = 0;
    ref[0][1] = 0;
    ref[0][2] = 0;
    ref[0][3] = 1;
    ref[0][4] = 0;
    ref[0][5] = 2;
    
    for (j = 0; j < 5; j++)
        acp[j] = sp[0][j];
    
    for (i = 1; i < 3*N; i++){
        if (!(fabs(acp[0] - sp[i][0]) < tol && fabs(acp[1] - sp[i][1]) < tol && fabs(acp[2] - sp[i][2]) < tol)){
            for (j = 0; j < 5; j++)
                acp[j] = sp[i][j];
        }
        ref[(int)sp[i][3]][2*(int)sp[i][4]] = (int)acp[3];
        ref[(int)sp[i][3]][2*(int)sp[i][4]+1] = (int)acp[4];
    }
    
    for (i = 1; i < N; i++){
        
        for (j = 0; j < 3; j++){
            hp[j] = 0;
            rhp = hid[ref[i][2*j]][ref[i][2*j+1]];
            
            if (rhp != -1){
                hp[j] = rhp;
                pirng[j] = 0;
            }
            else {
                hid[ref[i][2*j]][ref[i][2*j+1]] = 0;
                pirng[j] = 1;
            }
        }
        
        for (vpi = 0; vpi < vtl+ptl; vpi++){
            
            for (j = 0; j < 3; j++){
                if (pirng[j] > 0){
                    pit = point_in_triangle(tab[st[i]][3*j],tab[st[i]][3*j+1],
                                            tab[vp[vpi]][0],tab[vp[vpi]][1],
                                            tab[vp[vpi]][3],tab[vp[vpi]][4],
                                            tab[vp[vpi]][6],tab[vp[vpi]][7],
                                            bbox[vp[vpi]]);
                    if(pit){
                        pot=point_of_triangle(tab[st[i]][3*j],tab[st[i]][3*j+1],
                                              tab[vp[vpi]][0],tab[vp[vpi]][1],
                                              tab[vp[vpi]][3],tab[vp[vpi]][4],
                                              tab[vp[vpi]][6],tab[vp[vpi]][7],
                                              tol);
                        if (!pot){
                            hp[j] = 1;
                            hid[ref[i][2*j]][ref[i][2*j+1]] = 1;
                        }
                    }
                }
            }
            
            if (hp[0] == 1 && hp[1] == 1 && hp[2] == 1){
                hidden[htl] = st[i];
                htl += 1;
                break;
            }
        }
        
        sm = hp[0] + hp[1] + hp[2];
        if (sm > 0 && sm < 3){
            partial[ptl] = st[i];
            vp[vtl+ptl] = st[i];
            ptl += 1;
        }
        else if (sm == 0){
            visible[vtl] = st[i];
            vp[vtl+ptl] = st[i];
            vtl += 1;
        }
    }
    
    lengths[0] = vtl;
    lengths[1] = ptl;
    lengths[2] = htl;
}

static PyObject *decl(PyObject *self, PyObject *args)
{
    
    PyArrayObject *tab;
    PyArrayObject *st;
    PyArrayObject *sp;
    double tol;
    int i,j,N;
    double **dtab;
    int *ist;
    double **isp;
    
    int *visible;
    int *partial;
    int *hidden;
    int lengths[3];
    
    PyArrayObject *rv, *rp, *rh;
    int dims[1];
    
    if (!PyArg_ParseTuple(args, "O!O!O!d", &PyArray_Type, &tab, &PyArray_Type, &st, &PyArray_Type, &sp, &tol))
        return NULL;
        
    if (tab->nd != 2 || st->nd != 1 || sp->nd != 2) {
		PyErr_SetString(PyExc_ValueError, "Wrong dimensions of some arguments.");
		return NULL;
	} 

    N = tab->dimensions[0];
    
    dtab = (double **)malloc(N * sizeof(double *));
    ist = (int *)malloc(N * sizeof(int));
    isp = (double **)malloc(3*N * sizeof(double *));
    
    for (i = 0; i < N; i++){
        dtab[i] = (double *)malloc(9 * sizeof(double));
        for (j = 0; j < 9; j++)
            dtab[i][j] = *(double *)(tab->data + i*tab->strides[0] + j*tab->strides[1]);
            
        ist[i] = *(int *)(st->data + i*st->strides[0]);
    }
    for (i = 0; i < 3*N; i++){
        isp[i] = (double *)malloc(5 * sizeof(double));
        for (j = 0; j < 5; j++)
            isp[i][j] = *(double *)(sp->data + i*sp->strides[0] + j*sp->strides[1]);
    }
    
    visible = (int *)malloc(N * sizeof(int));
    partial = (int *)malloc(N * sizeof(int));
    hidden = (int *)malloc(N * sizeof(int));
    
    vph(dtab, ist, isp, N, tol, visible, partial, hidden, lengths);
 
    dims[0] = lengths[0];
    rv = (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_INT);
    for (i = 0; i < lengths[0]; i++)
        ((int *)rv->data)[i] = visible[i];
    dims[0] = lengths[1];
    rp = (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_INT);
    for (i = 0; i < lengths[1]; i++)
        ((int *)rp->data)[i] = partial[i];
    dims[0] = lengths[2];
    rh = (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_INT);
    for (i = 0; i < lengths[2]; i++)
        ((int *)rh->data)[i] = hidden[i];
 
    for (i = 0; i < N; i++)
        free(dtab[i]);
    for (i = 0; i < 3*N; i++)
        free(isp[i]);
    free(dtab);
    free(ist);
    free(isp);
    
    free(visible);
    free(partial);
    free(hidden);
 
    PyObject *rtup = PyTuple_New(3);
    PyTuple_SetItem(rtup, 0, PyArray_Return(rv));
    PyTuple_SetItem(rtup, 1, PyArray_Return(rp));
    PyTuple_SetItem(rtup, 2, PyArray_Return(rh));
    return rtup;
}

static PyMethodDef eclipseMethods[] = {
  {"decl",  decl,  METH_VARARGS, "Detect visible, partially visible and hidden triangles."},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
initceclipse(void)
{
  (void) Py_InitModule("ceclipse", eclipseMethods);
  import_array();
}

