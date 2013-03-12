/*
 * compile with
 *		cc -g -fullwarn -o marching2F marching2F.c ./Meschach/meschach.a -lm 
 *  on linux/ubuntu 
 *		cc -g -shared -o marching2FLib.so marching2FLib.c -lm -lmeschach -lpython2.7
 *		cc -g -ansi -shared -o marching2FLib.so marching2FLib.c -lm -lmeschach -lpython2.7
 */

#include <stdlib.h>
#include <stdio.h>

#include <python2.7/Python.h>
#include <numpy/arrayobject.h>


/* needed on linux/ubuntu to get M_PI???? */
#define	__USE_XOPEN
#include <math.h>
#include <values.h>
#include <signal.h>

/* not on SGI typedef unsigned int u_int;*/
typedef unsigned int u_int;
#include <meschach/matrix.h>
/*#include "./Meschach/matrix.h"*/
#include "utlist.h"

#define	numDim	3
#define	MAXITER	100
#define	epsilon10	0.0000000001


enum error_numbers {NO_ERROR=0, LIST_EMPTY=10};

enum Potential_Type {SPHERE, BINARY_ROCHE, MISALIGNED_BINARY_ROCHE,
			ROTATE_ROCHE, TORUS};


/* *****************************************************************
 * Macro PhMFillVECtor
 * Fills the VECtor VEC with components v1, v2, v3
 * *****************************************************************/
#define PhMFillVECtor(VEC, v0, v1, v2) \
        (VEC)->ve[0] = (v0);\
        (VEC)->ve[1] = (v1);\
        (VEC)->ve[2] = (v2)  /* No ending ";" ... put into code when called */



 /* meshVertex MUST be pointer!!! */
#define dumpMeshVertexNoVECtors(mV) \
	printf("nx = %f\n",(mV)->nx);\
	printf("ny = %f\n",(mV)->ny);\
	printf("nz = %f\n",(mV)->nz);\
	printf("nn = %f\n",(mV)->nn);\
	printf("t1x = %f\n",(mV)->t1x);\
	printf("t1y = %f\n",(mV)->t1y);\
	printf("t1z = %f\n",(mV)->t1z);\
	printf("t2x = %f\n",(mV)->t2x);\
	printf("t2y = %f\n",(mV)->t2y);\
	printf("t2z = %f\n",(mV)->t2z);\
	printf("\n\n=====END DUMP ======\n\n\n") /* No ";" at end...put that into actual code */


 /* meshVertex MUST be pointer!!! */
#define dumpMeshVertex(mV) \
	printf("nx = %f\n",(mV)->nx);\
	printf("ny = %f\n",(mV)->ny);\
	printf("nz = %f\n",(mV)->nz);\
	printf("nn = %f\n",(mV)->nn);\
	printf("t1x = %f\n",(mV)->t1x);\
	printf("t1y = %f\n",(mV)->t1y);\
	printf("t1z = %f\n",(mV)->t1z);\
	printf("t2x = %f\n",(mV)->t2x);\
	printf("t2y = %f\n",(mV)->t2y);\
	printf("t2z = %f\n",(mV)->t2z);\
	printf("r = "); v_output((mV)->r);\
	printf("n = "); v_output((mV)->n);\
	printf("t1 = "); v_output((mV)->t1);\
	printf("t2 = "); v_output((mV)->t2);\
	printf("\n\n=====END DUMP ======\n\n\n") /* No ";" at end...put that into actual code */

 /* meshVertex MUST be pointer!!! */
#define dumpMeshVertexPoint(mV) \
	v_output((mV)->r); printf("\n"); /* No ";" at end...put that into actual code */

#define PhMCopyMeshVertex(listMeshVertex, meshVertex)\
	(listMeshVertex)->nx = (meshVertex)->nx;\
	(listMeshVertex)->ny = (meshVertex)->ny;\
	(listMeshVertex)->nz = (meshVertex)->nz;\
	(listMeshVertex)->nn = (meshVertex)->nn;\
	(listMeshVertex)->t1x = (meshVertex)->t1x;\
	(listMeshVertex)->t1y = (meshVertex)->t1y;\
	(listMeshVertex)->t1z = (meshVertex)->t1z;\
	(listMeshVertex)->t2x = (meshVertex)->t2x;\
	(listMeshVertex)->t2y = (meshVertex)->t2y;\
	(listMeshVertex)->t2z = (meshVertex)->t2z;\
	(listMeshVertex)->r = v_copy((meshVertex)->r, VNULL);\
	(listMeshVertex)->n = v_copy((meshVertex)->n, VNULL);\
	(listMeshVertex)->t1 = v_copy((meshVertex)->t1, VNULL);\
	(listMeshVertex)->t2 = v_copy((meshVertex)->t2, VNULL) /* No ";" at end...put that into actual code */







typedef struct PhTMeshVertex {
	double	nx,		/* x-component of the normal VECtor */
		ny,		/* y-component of the normal VECtor */
		nz,		/* z-component of the normal VECtor */
		nn,		/* norm of the normal VECtor */
		t1x,		/* x-component of the first tangent VECtor */
		t1y,		/* y-component of the first tangent VECtor */
		t1z,		/* z-component of the first tangent VECtor */
		t2x,		/* x-component of the second tangent VECtor */
		t2y,		/* y-component of the second tangent VECtor */
		t2z;		/* z-component of the second tangent VECtor */

	VEC	*r,	/* Relative radius VECtor */
		*n,	/* normal VECtor */
		*t1,	/* the first tangent VECtor */
		*t2;	/* the second tangent VECtor */

} PhTMeshVertex;



typedef struct PhTListMeshVertex {
    PhTMeshVertex	*meshVertex;
	int				index;
	int				value; /* diagnostics only ****/
    struct PhTListMeshVertex *next; /* next meshVertex in List */
} PhTListMeshVertex;


typedef struct PhTListTriangles {
	int				index;
    PhTMeshVertex	meshVertex0;
    PhTMeshVertex	meshVertex1;
    PhTMeshVertex	meshVertex2;
    struct PhTListTriangles *next; /* next Triangle in List */
} PhTListTriangles;


#define	MAX_NUM_POT_PARAMS	10 /* That should handel it???? */
typedef struct PhTPotential {

	int	potentialType;

	/*	BinaryRoche parameters
		@param r:      relative radius vector (3 components)
		@param D:      instantaneous separation
		@param q:      mass ratio
		@param F:      synchronicity parameter
		@param Omega:  value of the potential

		Additional MisalignedBinaryRoche parameters
		@param theta:  misalignment coordinate
		@param phi:    misalignment coordinate

		Additional RotateRoche parameters
		RPole

		Additional Torus parameters
		rMinor

	*/
	double	R,		/* Radius of the sphereical potential...its value */

		D,		/* instantaneous separation */
		q,		/* mass ratio */
		F,		/* synchronicity parameter */
		Omega,		/* value of the potential */

		theta,  	/* misalignment coordinate */
		phi,    	/* misalignment coordinate */

		RPole,

		rMinor,

		delta,		/* discritization scale, delta in the code */
		/* potential functions */
		(*pot)(VEC *,  void *),
		(*dpdx)(VEC *, void *),
		(*dpdy)(VEC *, void *),
		(*dpdz)(VEC *, void *);

} PhTPotential;

static PyObject* py_getMesh(PyObject* self, PyObject* py_args);

void matInvert(MAT *M, MAT *MInverse);

VEC *cart2local(PhTMeshVertex *meshVertex, VEC *r);
VEC *local2cart(PhTMeshVertex *meshVertex, VEC *r);


void MeshVertexInit(PhTMeshVertex **meshVertex, VEC *r,
			double (*dpdx)(VEC *, void *),
			double (*dpdy)(VEC *, void *),
			double (*dpdz)(VEC *, void *),
			void *args, int error);


PhTMeshVertex *projectOntoPotential(VEC *r, void *args);

double Sphere(VEC *r, void *args);
double dSpheredx(VEC *r, void *args);
double dSpheredx(VEC *r, void *args);
double dSpheredy(VEC *r, void *args);
double dSpheredz(VEC *r, void *args);


double BinaryRoche(VEC *r, void *args);
double dBinaryRochedx(VEC *r, void *args);
double dBinaryRochedy(VEC *r, void *args);
double dBinaryRochedz(VEC *r, void *args);



double RotateRoche(VEC *r, void *args);
double dRotateRochedx(VEC *r, void *args);
double dRotateRochedy(VEC *r, void *args);
double dRotateRochedz(VEC *r, void *args);


double **discretize(double delta,  int maxNumTriangles, int *totalNumTriangles, void *args);


PhTListMeshVertex *initListMeshVertexNode(PhTMeshVertex *meshVertex, int index);

void deleteNodeListMeshVertex(PhTListMeshVertex *head,
			PhTListMeshVertex *end, int index);

void deleteListMeshVertex(PhTListMeshVertex **head,
		PhTListMeshVertex *ptr, PhTListMeshVertex *end);


PhTListTriangles *initListTriangeNode(int index, PhTMeshVertex *meshVertex0,
 PhTMeshVertex *meshVertex1, PhTMeshVertex *meshVertex2);


PhTMeshVertex *returnNodeMeshVertexByIndex(PhTListMeshVertex *head,
							int index);

/* deletes the node of specific index    */
void deleteNodeMeshVertex(PhTListMeshVertex **head, PhTListMeshVertex *end,
							int index);

void PhLLInsertNode(PhTListMeshVertex **head, PhTListMeshVertex *new, int index);


void PhTriangleTableFill(double **table, PhTListTriangles *triangles,
		int numListTriangles, void *args, int error);


void sigHandler(int sig);

int	totalNumListMeshVertexV; /* diagnostics only */

static PyObject* py_getMesh(PyObject* self, PyObject* py_args)
{
void *args;

double **table;

int	i,
	j,
	N,
	M,
	totalNumTriangles=0,
	potentialType,
	refCounter,
	numberParametersPassed;

double	x,
	y,
	z,
	**buffer,
	*a,
	tempDouble[MAX_NUM_POT_PARAMS];

PyArrayObject *matout;

PyObject	*temp_p,
		*temp_p2;

int	dims[2];

PhTPotential	*potential;

	/*
	 * Set-up exception handlers
	 */
	signal(SIGFPE,  sigHandler);
	signal(SIGILL,  sigHandler);
	signal(SIGSEGV, sigHandler);
	signal(SIGBUS,  sigHandler);
	signal(SIGSYS,  sigHandler);
	signal(SIGABRT,  sigHandler);


	potential = (PhTPotential *)calloc(1, sizeof(PhTPotential));


	/* ************Get Potential Type************** */
	refCounter = 0;
	temp_p = PyTuple_GetItem(py_args,refCounter); 
	if(temp_p == NULL) {return NULL;} 
 
        /* Check if temp_p is numeric */ 
	if (PyNumber_Check(temp_p) != 1)
	{ 
		PyErr_SetString(PyExc_TypeError,"Non-numeric argument."); 
		return NULL; 
	} 
	temp_p2 = PyNumber_Long(temp_p); 
	potentialType = PyLong_AsUnsignedLong(temp_p2); 
	printf("potentialType = %d\n", potentialType);




	/* ******Get number of arguments being passed as a check Type******** */
	refCounter = 1;
	temp_p = PyTuple_GetItem(py_args,refCounter); 
	if(temp_p == NULL) {return NULL;} 
 
        /* Check if temp_p is numeric */ 
        if (PyNumber_Check(temp_p) != 1)
	{ 
		PyErr_SetString(PyExc_TypeError,"Non-numeric argument."); 
		return NULL; 
	} 
	temp_p2 = PyNumber_Long(temp_p); 
	numberParametersPassed = PyLong_AsUnsignedLong(temp_p2); 
	printf("numberParametersPassed = %d\n", numberParametersPassed);


	/* ** Get the parameters being passed ****** */
	for (i=(refCounter+1); i<=numberParametersPassed+1; i++)
	{ /* <=numberParametersPassed+1 because there were 2 in front of the params*/
		temp_p = PyTuple_GetItem(py_args,i); 
		if(temp_p == NULL) {return NULL;} 
 
		/* Check if temp_p is numeric */ 
		if (PyFloat_Check(temp_p) != 1)
		{ 
            	PyErr_SetString(PyExc_TypeError,"Non-numeric argument."); 
            	return NULL; 
		} 
 
		/* Convert number to python float and than C double */ 
		temp_p2 = PyNumber_Float(temp_p); 
		tempDouble[i-(refCounter+1)] = PyFloat_AsDouble(temp_p2); 
		printf("i = %d  tempDouble[%d] = %f\n", i, i-(refCounter+1), tempDouble[i-(refCounter+1)]);
		Py_DECREF(temp_p2);
	}



	switch ( potentialType )
	{
		case SPHERE:
			potential->potentialType = potentialType;
			potential->R = tempDouble[0];
			potential->delta = tempDouble[1];
			potential->pot = &Sphere;
			potential->dpdx = &dSpheredx;
			potential->dpdy = &dSpheredy;
			potential->dpdz = &dSpheredz;
			break;
		case BINARY_ROCHE:
			potential->potentialType = potentialType;
			potential->D = tempDouble[0];
			potential->q = tempDouble[1];
			potential->F = tempDouble[2];
			potential->Omega = tempDouble[3];
			potential->delta = tempDouble[4];
			potential->pot = &BinaryRoche;
			potential->dpdx = &dBinaryRochedx;
			potential->dpdy = &dBinaryRochedy;
			potential->dpdz = &dBinaryRochedz;
			break;
		case MISALIGNED_BINARY_ROCHE:
			potential->potentialType = potentialType;
			potential->R = tempDouble[0];
			break;
		case ROTATE_ROCHE:
			potential->potentialType = potentialType;
			potential->Omega = tempDouble[0];
			potential->RPole = tempDouble[1];
			potential->delta = tempDouble[2];
			potential->pot = &RotateRoche;
			potential->dpdx = &dRotateRochedx;
			potential->dpdy = &dRotateRochedy;
			potential->dpdz = &dRotateRochedz;
			break;
		case TORUS:
			potential->potentialType = potentialType;
			potential->R = tempDouble[0];
			break;
		default:
			printf("E  R  R  O  R:  Unknow option for potential type\n");
			return (0);
			break;
	}




	args = (void *)(potential);

printf("potential->delta = %f\n",potential->delta);
	/*discretize(0.1,  15000, &totalNumTriangles, args);*/
	table = discretize(potential->delta,  100000, &totalNumTriangles, args); /*This seems to work */
	/*discretize(0.3,  8000, &totalNumTriangles, args); This seems to work */
	/* discretize(0.5,  8000, &totalNumTriangles, args);  This seems to work...*/

	N = totalNumTriangles; M = 16;
	dims[0] = N; dims[1]=M;
printf("totalNumTriangles = %d\n", totalNumTriangles);

 	matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	if (matout==(PyArrayObject *)NULL) printf("??? MEM ISSUE???\n");
	if (matout->data==NULL) printf("??? MEM ISSUE???\n");
printf("I'm at 4\n");

	/*matout->data = &(item[0]);*/
printf("I'm at 5\n");

	a=(double *)matout->data;
printf("I'm at 6\n");
	buffer = (double **)calloc(N, sizeof(double*));
printf("I'm at 6B\n");
	for ( i=0; i<N; i++)  {
          buffer[i]=a+i*M;  }
	for(i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			buffer[i][j] = table[i][j];
		}
	}

	for(i = 0; i < N; i++)
	{
		free(table[i]);
	}	
 	free(table);
	free(buffer);
	free(potential);
printf("I'm at 7\n");

	/*return PyArray_Return(matout);*/
return Py_BuildValue("Oi", matout, N*M);  
}


/*  memory is allocated by calling routine */
void matInvert(MAT *M, MAT *MInverse)
{
double	detM;

	detM = M->me[0][0]*M->me[1][1]*M->me[2][2] -
	M->me[0][2]*M->me[1][1]*M->me[2][0] +
	M->me[0][1]*M->me[1][2]*M->me[2][0] -
	M->me[0][0]*M->me[1][2]*M->me[2][1] +
	M->me[0][2]*M->me[1][0]*M->me[2][1] -
	M->me[0][1]*M->me[1][0]*M->me[2][2];

	MInverse->me[0][0] = (M->me[1][1]*M->me[2][2]-M->me[1][2]*M->me[2][1])/detM;
	MInverse->me[0][1] = (M->me[0][2]*M->me[2][1]-M->me[0][1]*M->me[2][2])/detM;
	MInverse->me[0][2] = (M->me[0][1]*M->me[1][2]-M->me[0][2]*M->me[1][1])/detM;
	MInverse->me[1][0] = (M->me[1][2]*M->me[2][0]-M->me[1][0]*M->me[2][2])/detM;
	MInverse->me[1][1] = (M->me[0][0]*M->me[2][2]-M->me[0][2]*M->me[2][0])/detM;
	MInverse->me[1][2] = (M->me[0][2]*M->me[1][0]-M->me[0][0]*M->me[1][2])/detM;
	MInverse->me[2][0] = (M->me[1][0]*M->me[2][1]-M->me[2][0]*M->me[1][1])/detM;
	MInverse->me[2][1] = (M->me[0][1]*M->me[2][0]-M->me[0][0]*M->me[2][1])/detM;
	MInverse->me[2][2] = (M->me[0][0]*M->me[1][1]-M->me[0][1]*M->me[1][0])/detM;


}




/*  memory for the returned VECtor must be allocated here */
VEC *cart2local(PhTMeshVertex *meshVertex, VEC *r)
{
MAT	*M,
	*MInverse;

VEC	*localVEC;



	/*
	 * This function converts VECtor @r from the cartesian
	 * coordinate system (defined by i, j, k) to the
	 * local coordinate system (defined by n, t1, t2).
    
	 * @param v: MeshVertex that defines the local coordinate system
	 * @param r: VECtor in the cartesian coordinate system
	 */
	localVEC = v_get(numDim); /* Allocate the memory for the local system */
	M = m_get(numDim,numDim);
	MInverse = m_get(numDim,numDim);
	M = set_col(M, 0, meshVertex->n);
	M = set_col(M, 1, meshVertex->t1);
	M = set_col(M, 2, meshVertex->t2);

	matInvert(M, MInverse);
	mv_mlt(MInverse, r, localVEC);

	m_free(M);
	m_free(MInverse);

	return (localVEC);

}




/*  memory for the returned VECtor must be allocated here */
VEC *local2cart(PhTMeshVertex *meshVertex, VEC *r)
{
MAT	*M;

VEC	*cartVEC;



	/*
	 * This function converts VECtor @r from the local
	 * coordinate system (defined by n, t1, t2) to the
	 * cartesian coordinate system (defined by i, j, k).
    
	 * @param v: MeshVertex that defines the local coordinate system
	 * @param r: VECtor in the local coordinate system
	 */
	cartVEC = v_get(numDim); /* Allocate the memory for the cartesian system */
	M = m_get(numDim,numDim);
	M = set_col(M, 0, meshVertex->n);
	M = set_col(M, 1, meshVertex->t1);
	M = set_col(M, 2, meshVertex->t2);

	mv_mlt(M, r, cartVEC);

	m_free(M);

	return (cartVEC);

}








void MeshVertexInit(PhTMeshVertex **meshVertex, VEC *r,
			double (*dpdx)(VEC *, void *),
			double (*dpdy)(VEC *, void *),
			double (*dpdz)(VEC *, void *),
			void *args, int error)
{
double	norm,	/* for local convienience define shorter names */
	nx,
	ny,
	nz,
	t1x,
	t1y,
	t1z,
	t2x,
	t2y,
	t2z;

	/* We allocate the memory here, must free it elsewhere */ 
	(*meshVertex) =
		(PhTMeshVertex *)calloc(1, sizeof(PhTMeshVertex));
		
		
		

	(*meshVertex)->nx = nx = dpdx(r, args);
	(*meshVertex)->ny = ny = dpdy(r, args);
	(*meshVertex)->nz = nz = dpdz(r, args);
	(*meshVertex)->nn = norm = sqrt(nx*nx+ny*ny+nz*nz);
	
	nx = (*meshVertex)->nx /= norm;
	ny = (*meshVertex)->ny /= norm;
	nz = (*meshVertex)->nz /= norm;

        if( (nx > 0.5) || (ny > 0.5) )
	{
            norm = sqrt(ny*ny+nx*nx);
            (*meshVertex)->t1x = t1x = ny/norm;
            (*meshVertex)->t1y = t1y = -nx/norm;
            (*meshVertex)->t1z = t1z = 0.0;
	}
        else
	{
            norm = sqrt(nx*nx+nz*nz);
            (*meshVertex)->t1x = t1x = -nz/norm;
            (*meshVertex)->t1y = t1y = 0.0;
            (*meshVertex)->t1z = t1z = nx/norm;
	}
        (*meshVertex)->t2x = t2x = ny*t1z - nz*t1y;
        (*meshVertex)->t2y = t2y = nz*t1x - nx*t1z;
        (*meshVertex)->t2z = t2z = nx*t1y - ny*t1x;


	(*meshVertex)->r = v_get(numDim);
	(*meshVertex)->n = v_get(numDim);
	(*meshVertex)->t1 = v_get(numDim);
	(*meshVertex)->t2 = v_get(numDim);

	PhMFillVECtor((*meshVertex)->r, r->ve[0], r->ve[1], r->ve[2]);
	PhMFillVECtor((*meshVertex)->n, nx, ny, nz);
	PhMFillVECtor((*meshVertex)->t1, t1x, t1y, t1z);
	PhMFillVECtor((*meshVertex)->t2, t2x, t2y, t2z);

	error = 0; /* Will need some diagnostics at some point */
}






PhTMeshVertex *projectOntoPotential(VEC *r, void *args)
{
	/*
	 * Last term must be constant factor (i.e. the value of the potential)
	 */
VEC	*ri,
	*g,
	*VECDifference;

	/* For now assume all fuctions of this type take 4/5 args */
double	(*pot)(VEC *, void *),
	(*dpdx)(VEC *, void *),
	(*dpdy)(VEC *, void *),
	(*dpdz)(VEC *, void *);

PhTMeshVertex *meshVertex; /* memory  allocated in call to MeshVertexInit*/

PhTPotential *potential;

double	R,
	grsq,
	scaleFactor,
	D,
	q,
	F,
	Omega;

int	n_iter=0,
	error=0;

	ri = v_get(numDim);
	g = v_get(numDim);
	VECDifference = v_get(numDim);
	v_zero(ri);  /* set ri to the initial value of zero */


	potential = (PhTPotential *)args;
 	pot = potential->pot;
	dpdx = potential->dpdx;
	dpdy = potential->dpdy;
	dpdz = potential->dpdz;

	/* setup parameters for Roche*/
	/* D = 0.85 looks good */
	/* q = 0.8 looks good */
	/*dArgs = (double *)calloc(4, sizeof(double));
	dArgs[0] = D = 0.75; 
	dArgs[1] = q = 0.7; 
	dArgs[2] = F = 1.2;
	dArgs[3] = Omega = 3.8;*/
  
/*	pot = &Sphere;
	dpdx = &dSpheredx;
	dpdy = &dSpheredy;
	dpdz = &dSpheredz;
	
 	pot = &BinaryRoche;
	dpdx = &dBinaryRochedx;
	dpdy = &dBinaryRochedy;
	dpdz = &dBinaryRochedz;*/

 	v_sub(r, ri, VECDifference); 
	while( (in_prod(VECDifference, VECDifference) > 1e-12) &&
			(n_iter < MAXITER ) )
	{
		ri = v_copy(r, ri); 
		g->ve[0] = dpdx(r, args);
		g->ve[1] = dpdy(r, args);
		g->ve[2] = dpdz(r, args);
		grsq = in_prod(g, g);

		/* r = ri - (pot((r, D, q, F, Omega)/grsq)*g; */
		scaleFactor = (-pot(r, args)/grsq);
		v_mltadd(ri, g, scaleFactor, r);
 		v_sub(r, ri, VECDifference); 
		n_iter+=1;
	}


	if (n_iter>=MAXITER)
	{
		printf("warning: projection did not converge\n");
	}
	/*printf("n_iter = %d\n", n_iter);*/
	MeshVertexInit( (&meshVertex), r, dpdx, dpdy, dpdz, args, error);

/*	printf("After convergence .... meshVertex = \n");
	dumpMeshVertex(meshVertex);
	*/

	return meshVertex;
}





double Sphere(VEC *r, void *args)
{
double	R;

PhTPotential *potential;

	potential = (PhTPotential *)args;
	R = potential->R;

    return (in_prod(r, r) - R*R);
}

double dSpheredx(VEC *r, void *args)
{
    return (2*r->ve[0]);
}


double dSpheredy(VEC *r, void *args)
{
    return (2*r->ve[1]);
}


double dSpheredz(VEC *r, void *args)
{
    return (2*r->ve[2]);
}





double BinaryRoche(VEC *r, void *args)
{
/*BinaryRoche (r, D, q, F, Omega=0.0):
    Computes a value of the potential. If @Omega is passed, it computes
    the difference.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param Omega:  value of the potential
    */
double	x,
	y,
	z,
	D,
	q,
	F,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	D = potential->D;
	q = potential->q;
	F = potential->F;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

    return 1.0/sqrt(in_prod(r, r)) +
    	q*(1.0/sqrt((x-D)*(x-D)+y*y+z*z)-x/D/D) + 
	0.5*F*F*(1+q)*(x*x+y*y) - Omega;

}


double dBinaryRochedx(VEC *r, void *args)
{
/*dBinaryRochedx (r, D, q, F):
    Computes a derivative of the potential with respect to x.
    the difference.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param Omega:  value of the potential
    */
double	x,
	y,
	z,
	D,
	q,
	F,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	D = potential->D;
	q = potential->q;
	F = potential->F;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];


	return -x*pow(in_prod(r, r),(-1.5)) -
		q*(x-D)*pow( (x-D)*(x-D)+y*y+z*z,(-1.5) ) - 
		q/D/D + F*F*(1+q)*x;

}




double dBinaryRochedy(VEC *r, void *args)
{
/*dBinaryRochedx (r, D, q, F):
    Computes a derivative of the potential with respect to y.
    the difference.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param Omega:  value of the potential
    */
double	x,
	y,
	z,
	D,
	q,
	F,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	D = potential->D;
	q = potential->q;
	F = potential->F;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

	return -y*pow(in_prod(r, r),-1.5) -
		q*y*pow( (x-D)*(x-D)+y*y+z*z, -1.5) +
		F*F*(1+q)*y;
}





double dBinaryRochedz(VEC *r, void *args)
{
/*dBinaryRochedx (r, D, q, F):
    Computes a derivative of the potential with respect to z.
    the difference.
    
    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    @param Omega:  value of the potential
    */
double	x,
	y,
	z,
	D,
	q,
	F,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	D = potential->D;
	q = potential->q;
	F = potential->F;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

	return -z*pow(in_prod(r, r),-1.5) -
		q*z*pow((x-D)*(x-D)+y*y+z*z, -1.5);

}










double RotateRoche(VEC *r, void *args)
{
/*RotateRoche(r, Omega, Rpole):
    Computes a value of the potential.

def RotateRoche(r, Omega, Rpole):
    """
    Roche shape of rotating star.
    
    In our units, 0.544... is the critical angular velocity.
    """
    Omega = Omega*0.54433105395181736
    r_ = (r[0]**2+r[1]**2+r[2]**2)**0.5
    return 1./Rpole - 1/r_ -0.5*Omega**2*(r[0]**2+r[1]**2)

    */
double	x,
	y,
	z,
	rMag,
	RPole,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	Omega = potential->Omega;
	RPole = potential->RPole;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

	Omega = Omega*0.54433105395181736;
	rMag = sqrt(x*x+y*y+z*z);
/*printf("in rotate... x = %f %f  %f... %f  %f  %f\n",x,y,z,Omega, rMag,1.0/RPole - 1.0/rMag -0.5*pow(Omega,2)*(x*x+y*y)); */
	return 1.0/RPole - 1.0/rMag -0.5*pow(Omega,2)*(x*x+y*y);

}




double dRotateRochedx(VEC *r, void *args)
{
/*RotateRoche(r, Omega, Rpole):
    Computes a value of the potential.

def RotateRoche(r, Omega, Rpole):
    """
    Roche shape of rotating star.
    
    In our units, 0.544... is the critical angular velocity.
    """
    Omega = Omega*0.54433105395181736
    r_ = (r[0]**2+r[1]**2+r[2]**2)**0.5
    return 1./Rpole - 1/r_ -0.5*Omega**2*(r[0]**2+r[1]**2)

def dRotateRochedx(r, Omega):
    Omega = Omega*0.54433105395181736
    return r[0]*(r[0]**2+r[1]**2+r[2]**2)**-1.5 - Omega**2*r[0]

    */
double	x,
	y,
	z,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

	Omega = Omega*0.54433105395181736;
/*printf("in drotate/dx  x = %f  %f  %f... %f  %f \n",x,y,z,Omega, x/pow(x*x+y*y+z*z,1.5) - Omega*Omega*x); */
	return x/pow(x*x+y*y+z*z,1.5) - Omega*Omega*x;

}




double dRotateRochedy(VEC *r, void *args)
{
/*RotateRoche(r, Omega, Rpole):
    Computes a value of the potential.

def RotateRoche(r, Omega, Rpole):
    """
    Roche shape of rotating star.
    
    In our units, 0.544... is the critical angular velocity.
    """
    Omega = Omega*0.54433105395181736
    r_ = (r[0]**2+r[1]**2+r[2]**2)**0.5
    return 1./Rpole - 1/r_ -0.5*Omega**2*(r[0]**2+r[1]**2)

def dRotateRochedy(r, Omega):
    Omega = Omega*0.54433105395181736
    return r[1]*(r[0]**2+r[1]**2+r[2]**2)**-1.5 - Omega**2*r[1]

    */
double	x,
	y,
	z,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

	Omega = Omega*0.54433105395181736;
/*printf("in drotate/dy .. x= %f  %f  %f... %f  %f \n",x,y,z,Omega, y/pow(x*x+y*y+z*z,1.5) - Omega*Omega*y); */
	return y/pow(x*x+y*y+z*z,1.5) - Omega*Omega*y;

}




double dRotateRochedz(VEC *r, void *args)
{
/*RotateRoche(r, Omega, Rpole):
    Computes a value of the potential.

def RotateRoche(r, Omega, Rpole):
    """
    Roche shape of rotating star.
    
    In our units, 0.544... is the critical angular velocity.
    """
    Omega = Omega*0.54433105395181736
    r_ = (r[0]**2+r[1]**2+r[2]**2)**0.5
    return 1./Rpole - 1/r_ -0.5*Omega**2*(r[0]**2+r[1]**2)

def dRotateRochedz(r, Omega):
    Omega = Omega*0.54433105395181736
    return r[2]*(r[0]**2+r[1]**2+r[2]**2)**-1.5

    */
double	x,
	y,
	z,
	Omega;


PhTPotential *potential;

	potential = (PhTPotential *)args;
	Omega = potential->Omega;

	x = r->ve[0];
	y = r->ve[1];
	z = r->ve[2];

	Omega = Omega*0.54433105395181736;
/*printf("in drotate/dz...   x = %f  %f  %f... %f  %f \n",x,y,z,Omega, z/pow(x*x+y*y+z*z,1.5)); */
	return z/pow(x*x+y*y+z*z,1.5);

}




/* ************************************************************************
 * ** Function: discretize                                         **
 * ** Functionality: Initiates and does the triangulization   **
 * **           **
 * **       if maxNumTriangles is set to <0  (-1) that means no max      **
 * **                      **
 * **                **
 * **          **
 * **       p0 = an meshVertex point      **
 * **       pk = an meshVertex point      **
 * **       V =  list of all vertices    **
 * **       P =  Actual front polygon **
 * **       Ts = list of triangles         **
 * **                                                                    **
 * **                                                                    **
 * ** Diagnostics: none at this time                                     **
 * **                                                                    **
 * ************************************************************************
 */

double **discretize(double delta,  int maxNumTriangles, int *totalNumTriangles, void *args)
{
PhTMeshVertex	p0, /* memory allocated in call to MeshVertexInit   */
				pk,	/* via the call to projectOntoPotential         */
				*PMeshVertex,
				*P_iMinus1,
				*P_iPlus1,
				*p0m,
				*v1,
				*v2;

PhTListMeshVertex	*V,
					*P,
					*headV=(PhTListMeshVertex *)NULL, /* needed for the linked list routines */
					*headP=(PhTListMeshVertex *)NULL; /* needed for the linked list routines */

PhTListMeshVertex	*listMeshVertex,
					*headListMeshVertex=(PhTListMeshVertex *)NULL;
					/* needed for the linked list routines */

PhTListTriangles	*Ts,
					*headTs=(PhTListTriangles *)NULL,
					*listTriangles;
					/* needed for the linked list routines */

VEC		*r0,
		*qk,
		*qkPart2,
		*qkPart3,
		*VECDifference,
		*V1,
		*V2,
		*V3,
		*V3Cart;


int		i,
		iMinus1,
		iPlus1,
		iPlus2,
		numListMeshVertexV=0,
		numListMeshVertexP=0,
		numListTriangles=0,
		minOmegaIndex,
		numTriangles2Create,
		error=0;

double	*omega,
		minOmega=10000.,
		dOmega,
		xi1,
		eta1,
		zeta1,
		xi2,
		eta2,
		zeta2,
		xi3,
		eta3,
		zeta3,
		omega1,
		omega2,
		**table;

	/*
	 * """
	 * Computes and returns a table of triangulated surface elements.
	 * Table columns are:
	 * 
	 * center-x, center-y, center-z, area, v1-x, v1-y, v1-z, v2-x, v2-y,
	 * v2-z, v3-x, v3-y, v3-z, normal-x, normal-y, normal-z
	 *
	 * Arguments:
	 * 	- delta:  triangle side in units of SMA
	 * 	- D:      instantaneous separation in units of SMA
	 * 	- q:      mass ratio
	 * 	- F:      synchronicity parameter
	 * 	- Omega:  value of the surface potential
	 * """
	 * Check...# q=0.76631, Omega=6.1518, F=5 is failing.
	 */

	VECDifference = v_get(numDim);
	V1 = v_get(numDim);
	V2 = v_get(numDim);
	V3 = v_get(numDim);


	r0 = v_get(numDim);
	/* check this to see if x<0 gives correct normal PhMFillVECtor(r0, -0.02, 0.0, 0.0); */
	/*PhMFillVECtor(r0, 0.0, 0.0, 8.0);*/
	/*PhMFillVECtor(r0, 1.0, 1.0, 1.0);*/
	PhMFillVECtor(r0, 0.2, 0.2, 0.2);
	p0 = *(projectOntoPotential(r0, args));
	V = initListMeshVertexNode(&p0, numListMeshVertexV); numListMeshVertexV++; totalNumListMeshVertexV++;
	LL_APPEND(headV, V);
/*	V = [p0] # A list of all vertices
	P = []   # Actual front polygon
	Ts = []  # Triangles
	*/
	/*printf("V->meshvertex = \n");
	dumpMeshVertex((V->meshVertex));
	printf("head = \n");
	dumpMeshVertex((headV->meshVertex));*/

	printf("\n\nFirst vertex .... meshVertex = \n");
	dumpMeshVertex(&p0);

	qk = v_get(numDim);
	qkPart2 = v_get(numDim);
	qkPart3 = v_get(numDim);
	/* # Create the first hexagon:*/
    for(i = 0; i < 6; i++)
	{
		/*
		 * calculate: qi+2 := p_1 + delta*cos(i*Pi/3)*t1 +
		 *                        delta*sin(i*Pi/3)*t2;
		 */
		sv_mlt( (delta*cos(i*M_PI/3.0)) , p0.t1, qkPart2);
		sv_mlt( (delta*sin(i*M_PI/3.0)) , p0.t2, qkPart3);
		v_add(qkPart2, qkPart3, qkPart2); /* put result in part 2 */
		v_add(p0.r, qkPart2, qk);
		/*printf("qk(%d) = \n\n\n",i); v_output(qk); printf("\n");*/

		pk = *(projectOntoPotential(qk, args));
	printf("vertex #%d .... meshVertex = \n",i);
	printf("r^2 = %f\n",v_norm2(pk.r));
	dumpMeshVertex(&pk);

		V = initListMeshVertexNode(&pk, numListMeshVertexV); numListMeshVertexV++; totalNumListMeshVertexV++;
		LL_APPEND(headV, V);
		/*LL_FOREACH(headV,listMeshVertex)
			printf("%d ", listMeshVertex->index); 
		printf("\n");*/


		P = initListMeshVertexNode(&pk, numListMeshVertexP); numListMeshVertexP++;
		LL_APPEND(headP, P);

/*		P.append (pk)
		V.append (pk)
*/
		/*print P */


	}




	listMeshVertex = headV;
	/*Ts = initListTriangeNode(numListTriangles, listMeshVertex->meshVertex,
			(listMeshVertex->next)->meshVertex,
			(listMeshVertex->next->next)->meshVertex);*/


	/* Add in the first 6 triangels */
    for(i = 0; i < 6; i++)
	{
		iPlus2 = i+2;
		if(iPlus2 > 6) iPlus2 = 1;
		Ts = initListTriangeNode(numListTriangles,
			returnNodeMeshVertexByIndex(listMeshVertex, 0),
			returnNodeMeshVertexByIndex(listMeshVertex, i+1),
			returnNodeMeshVertexByIndex(listMeshVertex, iPlus2));
		numListTriangles++;

		LL_APPEND(headTs, Ts);
	}


	/*printf("List Triangels =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%d ", listTriangles->index); 
	printf("\n");*/

/*	printf("List Triangel nx's =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%f   %f  %f||", listTriangles->meshVertex0.nx, listTriangles->meshVertex1.nx, listTriangles->meshVertex2.nx); 
	printf("\n");
	printf("List Triangel ny's =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%f   %f  %f||", listTriangles->meshVertex0.ny, listTriangles->meshVertex1.ny, listTriangles->meshVertex2.ny); 
	printf("\n");
	printf("List Triangel nz's =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%f   %f  %f||", listTriangles->meshVertex0.nz, listTriangles->meshVertex1.nz, listTriangles->meshVertex2.nz); 
	printf("\n");
*/


printf("List Triangel r's =     ");
LL_FOREACH(headTs,listTriangles)
	printf("(%f   %f  %f)  (%f   %f  %f)  (%f   %f  %f)\n||",
	 listTriangles->meshVertex0.r->ve[0], listTriangles->meshVertex0.r->ve[1], listTriangles->meshVertex0.r->ve[2],
	 listTriangles->meshVertex1.r->ve[0], listTriangles->meshVertex1.r->ve[1], listTriangles->meshVertex1.r->ve[2],
	 listTriangles->meshVertex2.r->ve[0], listTriangles->meshVertex2.r->ve[1], listTriangles->meshVertex2.r->ve[2]); 
printf("\n");



	/*printf("\n\n\nBefore Loop...P = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");


	printf("\n\n\nV = \n");
	LL_FOREACH(headV,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");*/


	while((numListMeshVertexP > 0) &&
		((numListTriangles < maxNumTriangles) || 
			(maxNumTriangles < 0) ? 1 : 0 ) )
	{
		listMeshVertex = headP;
		omega = (double *)calloc(numListMeshVertexP, sizeof(double));


		minOmega=10000.;
		for(i=0; i< numListMeshVertexP; i++)
		{
			iMinus1 = ((i-1) >= 0) ? i-1 : (numListMeshVertexP-1);

			iPlus1 = ((i+1) < numListMeshVertexP) ? i+1 : 0;

		/*printf("i = %d   iMinus1 = %d   iPlus1 = %d\n",i, iMinus1, iPlus1);*/
			PMeshVertex = returnNodeMeshVertexByIndex(headP, i);
			P_iMinus1 = returnNodeMeshVertexByIndex(headP, iMinus1);
			P_iPlus1 = returnNodeMeshVertexByIndex(headP, iPlus1);
			v_sub(P_iMinus1->r, PMeshVertex->r, VECDifference);
			V1 = cart2local(PMeshVertex, VECDifference);
			xi1=V1->ve[0]; eta1=V1->ve[1]; zeta1=V1->ve[2];

			v_sub(P_iPlus1->r, PMeshVertex->r, VECDifference);
			V2 = cart2local(PMeshVertex, VECDifference);
			xi2=V2->ve[0]; eta2=V2->ve[1]; zeta2=V2->ve[2];

			/* Check this logic to see if it is what we want
			 * not sure it reproduces the same as the original python */
/*			omega2 = fmod(atan2(zeta2, eta2),(2*M_PI));
			omega1 = fmod(atan2(zeta1, eta1), (2*M_PI));*/
			omega2 = atan2(zeta2, eta2);
			omega1 = atan2(zeta1, eta1);
			/*omega[i] = fmod(omega2 - omega1,(2*M_PI));;*/
			omega[i] = omega2 - omega1;
			if(omega[i] < 0) omega[i] += 2*M_PI;
			/*printf("omega[%d] = %f  xi1 = (%f , %f  , %f)  xi2 = (%f , %f  , %f)\n",i,omega[i]*180/M_PI, xi1, eta1, zeta1, xi2, eta2, zeta2);*/

			/* putting epsilon 10 here makes minOmega the lower
			 * index when two or more angles are equal.  This
			 * better mimics the python behavior
			 */
			if(omega[i] < minOmega-epsilon10)
			{
				minOmega = omega[i];
				minOmegaIndex = i;
			}
		}

/* ++++++++++++++++++++++++++++++++++++++ */
	/*if(numListTriangles<8)
	{
	minOmega = omega[0];
	minOmegaIndex = 0;
	}*/




		free(omega);
		omega = (double *) NULL;

		/*printf("minOmega = omega[%d] = %f\n",minOmegaIndex, minOmega);*/



		/*# The number of triangles to be generated: */
/*printf("minOmega*3/M_PI = %f  trunc(minOmega*3/M_PI) = %f  trunc(minOmega*3/M_PI)+1 = %f\n",
minOmega*3/M_PI, trunc(minOmega*3/M_PI), trunc(minOmega*3/M_PI)+1);*/
		numTriangles2Create = floor(minOmega*3/M_PI)+1;
		dOmega = minOmega/(numTriangles2Create*1.0);
		/*printf("1. Number of triangles to be generated: %d; domega = %f = %f\n",
					numTriangles2Create, dOmega, dOmega*180.0/M_PI);*/
		if( (dOmega < 0.8) && (numTriangles2Create > 1) )
		{
			numTriangles2Create -= 1;
			dOmega = minOmega/(numTriangles2Create*1.0);
		}
		/* WHAT DOES THIS MEAN??? ==> ### INSERT THE REMAINING HOOKS HERE!*/

		/*printf("Number of triangles to be generated: %d; domega = %f\n",
					numTriangles2Create, dOmega);*/

		/*# Generate the triangles: */

		/* minidx-1 ==> minOmegaIndex-1 */
		iMinus1 =
			((minOmegaIndex-1) >= 0) ? minOmegaIndex-1 : (numListMeshVertexP-1);

		/* minidx+1  ==> minOmegaIndex+1 */
		iPlus1 = ((minOmegaIndex+1) < numListMeshVertexP) ? minOmegaIndex+1 : 0;

		p0m = returnNodeMeshVertexByIndex(headP, minOmegaIndex);
		v1 = returnNodeMeshVertexByIndex(headP, iMinus1);
		v2 = returnNodeMeshVertexByIndex(headP, iPlus1);


		/*for(i=0; i< numTriangles2Create; i++)*/
		/*for(i=1; i< numTriangles2Create; i++)*/
		i = 1;
		do /* want to make sure this loop is executed at least
		    * once, when numTriangles2Create = 1 */
		{
			double norm3; /* we only need it inside of this loop: restrict scope */

			v_sub(v1->r, p0m->r, VECDifference);
			V1 = cart2local(p0m, VECDifference);
			xi1=V1->ve[0]; eta1=V1->ve[1]; zeta1=V1->ve[2];

			xi3 = 0.0;
			eta3 = (eta1*cos(i*dOmega)-zeta1*sin(i*dOmega));
			zeta3 = (eta1*sin(i*dOmega)+zeta1*cos(i*dOmega));
            norm3 = sqrt(eta3*eta3+zeta3*zeta3);
			eta3 /= norm3/delta;
			zeta3 /= norm3/delta;
			V3->ve[0]= xi3; V3->ve[1]= eta3; V3->ve[2]= zeta3;
			V3Cart = local2cart (p0m, V3);
			v_add(p0m->r, V3Cart, qk);
/*printf("xi1 = (%f  %f  %f)  xi2 = (%f  %f  %f)\n",xi1,eta1,zeta1,xi2,eta2,zeta2);
printf("qk(%d) = \n\n\n",i); v_output(qk); printf("\n");
*/
            pk = *(projectOntoPotential(qk, args));
/*printf("Inner v1.qk = %f ... theta = %f\n",in_prod(v1->r,qk),
acos(in_prod(v1->r,qk)/(sqrt(in_prod(v1->r,v1->r))*sqrt(in_prod(qk,qk)))));*/

			/* put this point into the V list IFF numTriangles2Create != 1 */
			if(numTriangles2Create != 1)
			{
				V = initListMeshVertexNode( (&pk), numListMeshVertexV);
						numListMeshVertexV++; totalNumListMeshVertexV++;
				LL_APPEND(headV, V);
			}
/*printf("pk(%d) = \n\n\n",i); v_output(pk.r); printf("\n");*/

/*	printf("==============\nBefore ADD...P = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->value); 
	printf("   value\n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("   index\n");
*/
			/* now we must put new points into the P list, but the oder of
			 * the P list is very important for the algorithum's success.
			 * the new points must be inserted "in order" at the location of
			 * the point p0m, where we are centering the new triangles.  That
			 * way the "border" points of the front polygon will be in order.
			 * the index of p0m is minOmegaIndex.  We want to insert just
			 * before there.  
			 * After insertion the index of p0m inceases by 1
			 * we must renumber in insertion and deletion
			 * where i starts at 1
			 *  ***A  L  S  O***: If only creating 1 triangle no new points
			 * are inserted ==== CHECK THIS LOGIC
			 */
			if(numTriangles2Create != 1)
			{
				P = initListMeshVertexNode( (&pk), numListMeshVertexP);
					numListMeshVertexP++;
				PhLLInsertNode(&headP, P, (minOmegaIndex+(i-1)) );
			}

/*	printf("\n\n\nfter ADD...P = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->value); 
	printf("   value\n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("   index\n");
	printf("==================\n");
*/


			if(i == 1)
			{
				/* get the first triangle*/
				if(numTriangles2Create == 1) /* only 1 triangle to create */
				{
					Ts = initListTriangeNode(numListTriangles, v1, v2, p0m);
					numListTriangles++;
					LL_APPEND(headTs, Ts);
				}
				else /* still first triangle */
				{
					Ts = initListTriangeNode(numListTriangles, v1, (&pk), p0m);
					numListTriangles++;
					LL_APPEND(headTs, Ts);
				}
			}
			else
			{
			PhTMeshVertex	*vTemp; /* just need here so limit scope */
				/* get the rest of the triangles for this polygon */
				vTemp = returnNodeMeshVertexByIndex(headV, 
						numListMeshVertexV-2 /* next to last one */);
				Ts = initListTriangeNode(numListTriangles, vTemp, (&pk), p0m);
				numListTriangles++;
				LL_APPEND(headTs, Ts);
			}

			i++;
		}while (i < numTriangles2Create);



		/* if numTriangles2Create > 1 get the last triangle */
		if(numTriangles2Create > 1)
		{	/*  N O T   W E L L:   v1's value is now changed from above*/
		
			/***********************************************************/
			v1 = returnNodeMeshVertexByIndex(headV, 
					numListMeshVertexV-1 /* last one */);
			Ts = initListTriangeNode(numListTriangles, v1, v2, p0m);
			numListTriangles++;
			LL_APPEND(headTs, Ts);
		}


	/*printf("=====================\n\nBefore Delete...P = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");*/
		/* can now safely delete p0m from the P list without deleteing
		 * the memory space for p0m itself.  only do this for the P list
		 * leave in the V list
		 * NOTE: after the insertions the index of p0m increases by 1
		 * i starts at 1.
		 * Also note:  i increses by 1 even when numTriangles2Create==1
		 * and no points were inserted.  This means the index of the point
		 * to delete has NOT changed so we must account for that
		 *  M U ST   L O O K for a more efficient way to do this...
		 */
		/* remove this meshVertex from the P list, leave in the V list */
		if(numTriangles2Create == 1) i--; /* should make i = 1 */
		deleteNodeMeshVertex((&headP), (PhTListMeshVertex *)NULL,
					(minOmegaIndex+(i-1)) );
		numListMeshVertexP--;
/*	printf("\n\nAfter Delete...P = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->value); 
	printf("   value\n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("   index\n");
	printf("+++++++++++++++++++++++++++++++++++++++++++++\n");
*/


	/*printf("\n\n\n P = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");

	printf("\n\n\n V = \n");
	LL_FOREACH(headV,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");

	printf("List Triangels =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%d ", listTriangles->index); 
	printf("\n");*/


/*printf(" the number of triangles is %d\n",numListTriangles);*/
/*	dumpMeshVertex(v1);
	dumpMeshVertex(v2);*/

		/*  T  E  M  P */
		/*break;*/
	}

/*
printf("List Triangel r's =     ");
LL_FOREACH(headTs,listTriangles)
	printf("%f   %f  %f||", listTriangles->meshVertex0.r->ve[0], listTriangles->meshVertex1.r->ve[1], listTriangles->meshVertex2.r->ve[2]); 
printf("\n");
*/

	/* We allocate the memory for the table here, must free it elsewhere */
	table = (double **)calloc(numListTriangles, sizeof(double *));
	for(i = 0; i < numListTriangles; i++)
		table[i] = (double *)calloc(16, sizeof(double));

	PhTriangleTableFill(table, headTs, numListTriangles, args, error);

	*totalNumTriangles = numListTriangles;

	/*printf("\n\n\nV = \n");
	LL_FOREACH(headV,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");

	printf("\n\n\nP = \n");
	LL_FOREACH(headP,listMeshVertex)
		printf("%d ", listMeshVertex->index); 
	printf("\n");*/



	printf("List Triangels =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%d ", listTriangles->index); 
	printf("\n");

	/*printf("List Triangel nx's =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%f   %f  %f||", listTriangles->meshVertex0.nx, listTriangles->meshVertex1.nx, listTriangles->meshVertex2.nx); 
	printf("\n");
	printf("List Triangel ny's =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%f   %f  %f||", listTriangles->meshVertex0.ny, listTriangles->meshVertex1.ny, listTriangles->meshVertex2.ny); 
	printf("\n");
	printf("List Triangel nz's =     ");
	LL_FOREACH(headTs,listTriangles)
		printf("%f   %f  %f||", listTriangles->meshVertex0.nz, listTriangles->meshVertex1.nz, listTriangles->meshVertex2.nz); 
	printf("\n");*/



/*	LL_FOREACH(headV,listMeshVertex)
		dumpMeshVertexPoint(listMeshVertex->meshVertex);
	printf("\n");
*/

	v_free(r0);
	v_free(qk);
	v_free(qkPart2);
	v_free(qkPart3);

return(table);
}








/* this initialises a node, allocates memory for the node, and returns   */
/* a pointer to the new node. Must pass it the node details, index, etc  */
PhTListMeshVertex *initListMeshVertexNode(PhTMeshVertex *meshVertex, int index)
{
PhTListMeshVertex *ptr;
	ptr = (PhTListMeshVertex *) calloc( 1, sizeof(PhTListMeshVertex ) );
	if( ptr == NULL )
	{                                  /* error allocating node?      */
		printf("Memory Allocation failed\n");
		return (PhTListMeshVertex *) NULL;   /* then return NULL, else      */
	}
	else                               /* allocated node successfully */
	{
		ptr->index = index;          /* fill in name details        */
		ptr->value = totalNumListMeshVertexV; /* diagnostics        */
		ptr->meshVertex = (PhTMeshVertex *)calloc(1, sizeof(PhTMeshVertex ) );
		PhMCopyMeshVertex(ptr->meshVertex, meshVertex); 
		ptr->next = (PhTListMeshVertex *)NULL;
		return ptr;                    /* return pointer to new node  */
	}
}



/* deletes the node of specific index    */
void deleteNodeMeshVertex(PhTListMeshVertex **head, PhTListMeshVertex *end,
							int index)
{
int		i;

PhTListMeshVertex	*temp, /* node to be deleted */
					*prev,
					*list;
	temp = *head;    
	prev = *head;   /* start of the list, will cycle to node before temp    */

		/* find node to be deleted */
		while( temp->index != index )        /* move temp to the node before*/
		{
			temp = temp->next;              /* the one to be deleted       */
		}
		/*printf("index = %d\n",temp->index);*/

	if( temp == prev )                    /* are we deleting first node  */
	{
		(*head) = (*head)->next;                  /* moves head to next node     */
		if( end == temp )                   /* is it end, only one node?   */
			end = end->next;                 /* adjust end as well          */
		free(temp->meshVertex);
		free( temp );                       /* free space occupied by node */
		temp = (PhTListMeshVertex *)NULL;
	}
	else                                   /* if not the first node, then */
	{
		while( prev->next != temp )        /* move prev to the node before*/
		{
			prev = prev->next;              /* the one to be deleted       */
		}
		prev->next = temp->next;            /* link previous node to next  */
		if( end == temp )                   /* if this was the end node,   */
			end = prev;                     /* then reset the end pointer  */
		v_free(temp->meshVertex->r);
		v_free(temp->meshVertex->n);
		v_free(temp->meshVertex->t1);
		v_free(temp->meshVertex->t2);
		free(temp->meshVertex);
		free( temp );                       /* free space occupied by node */
	}


	/* Now renumber the list indexes */
	i = 0;
	LL_FOREACH((*head),list)
			list->index = i++;


}





/* this deletes all nodes from the place specified by ptr                 */
/* if you pass it head in both elements, it will free up entire list                       */
void deleteListMeshVertex(PhTListMeshVertex **head, PhTListMeshVertex *ptr,
							PhTListMeshVertex *end)
{
PhTListMeshVertex	*temp,
					**phead=head;

	if( *phead == NULL ) return;   /* don't try to delete an empty list */

	if( ptr == *phead )       /* if we are deleting the entire list */
	{
		*phead = (PhTListMeshVertex *)NULL;         /* then reset head and end to signify empty   */
		end = (PhTListMeshVertex *)NULL;          /* list                                       */
	}
	else 
	{
		temp = *phead;          /* if its not the entire list, readjust end  */
		while( temp->next != ptr )         /* locate previous node to ptr  */
			temp = temp->next;
		end = temp;                        /* set end to node before ptr   */
	}

	while( ptr != NULL )    /* whilst there are still nodes to delete     */
	{
		temp = ptr->next;     /* record address of next node                */
		v_free(ptr->meshVertex->r);
		v_free(ptr->meshVertex->n);
		v_free(ptr->meshVertex->t1);
		v_free(ptr->meshVertex->t2);
		free(ptr->meshVertex);
		free( ptr );          /* free this node                             */
		ptr = temp;           /* point to next node to be deleted           */
	}
}




/* this initialises a triangle, allocates memory for the triangle,
 *  and returns a pointer to the new triangle. Must pass it the
 * triangle details, index, the three meshVertexes, etc.  The 3 mesh
 * here are independent of headV... see how that works for now
 * so the meshVertex info MUST be copied, its not just a pointer  */
PhTListTriangles *initListTriangeNode(int index, PhTMeshVertex *meshVertex0,
 PhTMeshVertex *meshVertex1, PhTMeshVertex *meshVertex2)
{
PhTListTriangles *ptr;
	ptr = (PhTListTriangles *) calloc( 1, sizeof(PhTListTriangles ) );
	if( ptr == NULL )
	{                                  /* error allocating node?      */
		printf("Memory Allocation failed\n");
		return (PhTListTriangles *) NULL;   /* then return NULL, else      */
	}
	else                               /* allocated node successfully */
	{
		ptr->index = index;          /* fill in name details        */
		/* Rememeber....for now at leat memory already allocated and this 
		 * is NOT a pointer */
		PhMCopyMeshVertex(&(ptr->meshVertex0), meshVertex0); 
		PhMCopyMeshVertex(&(ptr->meshVertex1), meshVertex1); 
		PhMCopyMeshVertex(&(ptr->meshVertex2), meshVertex2); 
		ptr->next = (PhTListTriangles *)NULL;
		return ptr;                    /* return pointer to new node  */
	}
}





/* returns the pointer to a meshvertex node of a specific index
 * in the linked list head*/
PhTMeshVertex *returnNodeMeshVertexByIndex(PhTListMeshVertex *head,
							int index)
{
PhTListMeshVertex	*temp; /* node to be found */
	temp = head;   /* start of the list, will cycle to node  */

		/* find node to be deleted */
		while( temp->index != index )  /* loop over to get the correct index*/
		{
			temp = temp->next;         /* not the index, go to next one     */
			/* If we get to the end of the list and we didn't find the index*/
			/* we have a problem....return NULL and print out error message */
			if (temp == (PhTListMeshVertex *)NULL)
			{
				printf(" COULD NOT FIND meshVertex of index %d\n", index);
				return((PhTMeshVertex *)NULL);
			}
		}
		/* If we get here, we found it */
		return(temp->meshVertex);

}





/* inserts a new node, uses uses index to align node, then reindexes so in */
/* order.  Pass it the address of the new node to be inserted,             */
/* with details all filled in                                              */
/* N  O  T  E   W  E  L  L:  Memory for the new node must be allocated
 * already and the new node can not appear anywhere else in the list
 * that would make "next" very confussed (can only point to one place)
 *       ****INSERTS BEFORE index****
 */
void PhLLInsertNode(PhTListMeshVertex **head, PhTListMeshVertex *new, int index)
{
PhTListMeshVertex	*temp,
					*prev,
					*list;

int	i;

	if( *head == NULL )
	{/* if an empty list,          */
		*head = new;                          /* set 'head' to it           */
		(*head)->next = NULL;                   /* set end of list to NULL    */
		return;                              /* and finish                 */
	}

	temp = *head;                             /* start at beginning of list */
                      /* whilst current index != index to be inserted then */
	while( temp->index != index )
	{
		prev = temp;			/* gets us the previous node */
		temp = temp->next;		/* goto the next node in list */
		if( temp == NULL )		/* dont go past end of list   */
			break;
	}


   /* we are at the point to insert,                             */
   /* First check to see if its inserting before the first node! */
	if( temp == *head )
	{
		new->next = *head;		/* link next field to original list   */
		*head = new;				/* head adjusted to new node          */
	}
	else
	{
		/*printf("\n\nIn else1\n");
		LL_FOREACH(*head,e)
			printf("%d ", e->index); 
		printf("\n");*/
		/* put the new node between prev and next */
		/* N  O  T  E   W  E  L  L:  Memory for the new node must be allocated
		 * already and the new node can not appear anywhere else in the list
		 * that would make "next" very confussed (can only point to one place)
		 */
		prev->next = new;
		new->next = temp;
		/*printf("\n\nIn else2\n");
		LL_FOREACH(*head,list)
      	  printf("%d ", list->index); 
		printf("\n");*/
	}

	/* Now renumber the list indexes */
	i = 0;
	LL_FOREACH((*head),list)
		list->index = i++;


}





/* inserts a new node, uses uses index to align node, then reindexes so in */
/* order.  Pass it the address of the new node to be inserted,             */
/* with details all filled in                                              */
/* N  O  T  E   W  E  L  L:  Memory for the new node must be allocated
 * already and the new node can not appear anywhere else in the list
 * that would make "next" very confussed (can only point to one place)
 *       ****INSERTS BEFORE index****
 */
void PhTriangleTableFill(double **table, PhTListTriangles *triangles,
		int numListTriangles, void *args, int error)
{
PhTListTriangles	*temp;

VEC		*centroid,
		*vertex0,
		*vertex1,
		*vertex2;

int		index;

double	cx,
		cy,
		cz,
		side1,
		side2,
		side3,
		s; /* semiperimeter: http://mathworld.wolfram.com/Semiperimeter.html */

PhTMeshVertex	c;


	if( triangles == (PhTListTriangles *)NULL )
	{/* if an empty list,          */
		error = LIST_EMPTY;
		return;
	}

	temp = triangles;/* start at beginning of list */

	centroid = v_get(numDim);
	vertex0 = v_get(numDim);
	vertex1 = v_get(numDim);
	vertex2 = v_get(numDim);


	index = 0;
	while( temp != (PhTListTriangles *)NULL )
	{

		vertex0 = v_copy(temp->meshVertex0.r, vertex0);
		vertex1 = v_copy(temp->meshVertex1.r, vertex1);
		vertex2 = v_copy(temp->meshVertex2.r, vertex2);

		cx = (vertex0->ve[0]+vertex1->ve[0]+vertex2->ve[0])/3.0;
		cy = (vertex0->ve[1]+vertex1->ve[1]+vertex2->ve[1])/3.0;
		cz = (vertex0->ve[2]+vertex1->ve[2]+vertex2->ve[2])/3.0;

		PhMFillVECtor(centroid, cx, cy, cz);

		c = *(projectOntoPotential(centroid, args));
		side1 = v_norm2(v_sub(vertex0, vertex1, VNULL));
		side2 = v_norm2(v_sub(vertex0, vertex2, VNULL));
		side3 = v_norm2(v_sub(vertex2, vertex1, VNULL));
		s = 0.5*(side1 + side2 + side3);

		table[index][ 0] = c.r->ve[0];
		table[index][ 1] = c.r->ve[1];
		table[index][ 2] = c.r->ve[2];
		table[index][ 3] = sqrt(s*(s-side1)*(s-side2)*(s-side3));
		table[index][ 4] = vertex0->ve[0];
		table[index][ 5] = vertex0->ve[1];
		table[index][ 6] = vertex0->ve[2];
		table[index][ 7] = vertex1->ve[0];
		table[index][ 8] = vertex1->ve[1];
		table[index][ 9] = vertex1->ve[2];
		table[index][10] = vertex2->ve[0];
		table[index][11] = vertex2->ve[1];
		table[index][12] = vertex2->ve[2];
		table[index][13] = c.n->ve[0];
		table[index][14] = c.n->ve[1];
		table[index][15] = c.n->ve[2];
		index++;


/*printf("Polygon[{{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}}],\n",
vertex0->ve[0],
vertex0->ve[1],
vertex0->ve[2],
vertex1->ve[0],
vertex1->ve[1],
vertex1->ve[2],
vertex2->ve[0],
vertex2->ve[1],
vertex2->ve[2]);*/

		temp = temp->next;		/* goto the next node in list */
	}

	v_free(centroid);
	v_free(vertex0);
	v_free(vertex1);
	v_free(vertex2);



}


/*
 *	
 *	Handles the exception interrupt
 *
 */
void sigHandler(int sig)
{
	
	int type;
	
	static char *types[7] = {"Illegal Instruction", 
							 "Floating Point Exception", 
							 "Segmentation Violation",
							 "Bus Error", 
							 "Bad System Call",
							 "Abort Signal Caught - Abnormal Termination Signal",
							 "Error Type Unknown"};
	
	if (sig == 4) type = 0;
	else if (sig == 8) type = 1;
	else if (sig == 11) type = 2;
	else if (sig == 10) type = 3;
	else if (sig == 12) type = 4;
	else if (sig == 6) type = 5;
	else type = 5;


	printf("\n\nERROR received!  Signal = %d ==> Type = %s\n\n", 
			sig, types[type]);
	
	printf("\n\nProgram Terminated\n");
	
	exit(1);

}






/*
 * Bind Python function names to our C functions
 */
static PyMethodDef myModule_methods[] = {
	{"getMesh", py_getMesh, METH_VARARGS},
	{NULL, NULL},
};


/*
 * Python calls this to let us initialize our module
 */
void initmarching2FLib()
{
	(void) Py_InitModule("marching2FLib", myModule_methods);
	import_array();
	
}

