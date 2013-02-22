#include "Python.h"
#include "numpy/arrayobject.h"


static PyObject *simple_subdivide(PyObject *self, PyObject *args)
{
  // Python calling signature:
  
  
  double spot_long,spot_colat,spot_angrad; // will hold spot parameters
  double x1,x2,x3,y1,y2,y3,z1,z2,z3; // Cartesian coordinates
  double phi1,theta1,phi2,theta2,phi3,theta3; // Spherical coordinates
  double d1,d2,d3; //distances
  double s1,s2,s3,p1,p2,p3; //shortcuts
  int inside,outside; // booleans
  int i,N;
  PyArrayObject *tri_array,*tri_in_on_spot; // triangle vertices coordinates
  double *tri; // keeps track of triangles on/in spots
  int dims[1];
  
  // see http://docs.python.org/2/c-api/arg.html for details on formatting strings
  // Python Scripting for Computational Science by Hans Petter Langtangen is
  // also very helpful
  if (!PyArg_ParseTuple(args, "O!(ddd)", &PyArray_Type, &tri_array, &spot_long, &spot_colat, &spot_angrad))
      return NULL;
 
  // error checks: first array should be two-dimensional, second one two-dimensional and 
  // thr type of both should be double
  if (tri_array->nd != 2 || tri_array->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "dim(obj)!=2 or dim(rot)!=2 or type not equal to double");
      return NULL;
  }
  
  if ((int)tri_array->dimensions[0]!=9){
      PyErr_SetString(PyExc_ValueError, "object array has incorrect size");
      return NULL;
  }
  
  // get dimensions of the array. it is also possible to access the, *tri_in_spot
  // elements of the array as if it was two-dimensional, but in this
  // case it is just easier to treat the arrays as one dimensional
  // with Nx3 elements
  N= (int)tri_array->dimensions[1];
  
  // we need to 'source' the data from the arrays and create a new 
  // return array
  tri = (double *)(tri_array->data);
  
  dims[0]=(int)tri_array->dimensions[1]*2;
  tri_in_on_spot = (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_INT);
  
  //////////// the original part
  for (i=0; i<N;i+=1)
  {
      // Cartesian coordinates
      x1 = tri[i];
      y1 = tri[i+N];
      z1 = tri[i+2*N];

      x2 = tri[i+3*N];
      y2 = tri[i+4*N];
      z2 = tri[i+5*N];
      
      x3 = tri[i+6*N];
      y3 = tri[i+7*N];
      z3 = tri[i+8*N];
      
      // Spherical coordinates
      //r1 = sqrt(x1*x1 + y1*y1 + z1*z1)
      //r2 = sqrt(x2*x2 + y2*y2 + z2*z2)
      //r3 = sqrt(x3*x3 + y3*y3 + z3*z3)
      
      phi1 = atan2(y1,x1);
      phi2 = atan2(y2,x2);
      phi3 = atan2(y3,x3);
      
      theta1 = atan2(sqrt(x1*x1+y1*y1),z1);
      theta2 = atan2(sqrt(x2*x2+y2*y2),z2);
      theta3 = atan2(sqrt(x3*x3+y3*y3),z3);
      
      // distances of all vertex coordinates to spot
      s1 = sin(0.5*(spot_colat-theta1));
      s2 = sin(0.5*(spot_colat-theta2));
      s3 = sin(0.5*(spot_colat-theta3));
      p1 = sin(0.5*(phi1-spot_long));
      p2 = sin(0.5*(phi2-spot_long));
      p3 = sin(0.5*(phi3-spot_long));
      
      d1 = 2.0*asin(sqrt(s1*s1+sin(spot_colat)*sin(theta1)*p1*p1));
      d2 = 2.0*asin(sqrt(s2*s2+sin(spot_colat)*sin(theta2)*p2*p2));
      d3 = 2.0*asin(sqrt(s3*s3+sin(spot_colat)*sin(theta3)*p3*p3));
      
      inside = (d1<=spot_angrad) && (d2<=spot_angrad) && (d3<=spot_angrad);
      outside = (d1>spot_angrad) && (d2>spot_angrad) && (d3>spot_angrad);
      
      ((int *)tri_in_on_spot->data)[i] = (1-inside) && (1-outside);
      ((int *)tri_in_on_spot->data)[i+N] = inside;
      
  }
  //////////// 
  
  // create a new numpy array and return it back to python.
  return PyArray_Return(tri_in_on_spot);
}









// register all functions
static PyMethodDef subdivMethods[] = {
  {"simple_subdivide",  simple_subdivide}, //python name, C name
  {NULL, NULL} //required ending
};

// module init function
PyMODINIT_FUNC
initcsubdivision(void)
{
  (void) Py_InitModule("csubdivision", subdivMethods);
  import_array(); //needed if numpy is used
}


//       subroutine simple_subdivide(N,M,tri,s,nor,mu,thresh,
//      +  newtri,news,newnor,newcen,newmu)
// Cf2py intent(in) N
// Cf2py intent(in) M
// Cf2py intent(in) tri
// Cf2py intent(in) s
// Cf2py intent(in) nor
// Cf2py intent(in) mu
// Cf2py intent(in) thresh
// Cf2py intent(in,out) newtri
// Cf2py intent(in,out) news
// Cf2py intent(in,out) newnor
// Cf2py intent(in,out) newcen
// Cf2py intent(in,out) newmu
// C     thresh is meant to put a limiting size on the triangles
// C     to subdivide (i.e. when already smaller than this threshold
// C     skip it
// 
//       integer i,j,k,N,M
//       real*8 tri(N,9),s(N),nor(N,3),cmp(N),mu(N)
//       real*8 thresh
//       real*8 newcen(M,3),newtri(M,9),newmu(M)
//       real*8 news(M),newnor(M,3),newcmp(M)
// C     -- define verticies
//       real*8 C1(3),C2(3),C3(3)
//       real*8 C4(3),C5(3),C6(3)
//       real*8 dummyN(3),dummyC(3)
//       real*8 a,b,c,h
//       
//       do 20, i=1,N
//         if ((thresh.gt.0d0).AND.(s(i)*mu(i).LT.thresh)) then
//            newtri(i,1) = tri(i,1)
//            newtri(i,2) = tri(i,2)
//            newtri(i,3) = tri(i,3)
//            newtri(i,4) = tri(i,4)
//            newtri(i,5) = tri(i,5)
//            newtri(i,6) = tri(i,6)
//            newtri(i,7) = tri(i,7)
//            newtri(i,8) = tri(i,8)
//            newtri(i,9) = tri(i,9)
//            news(i) = s(i)
//            newmu(i) = mu(i)
//            newnor(i,1) = nor(i,1)
//            newnor(i,2) = nor(i,2)
//            newnor(i,3) = nor(i,3)
//            newcen(i,1) = (newtri(i,1)+newtri(i,4)+newtri(i,7))/3.0
//            newcen(i,2) = (newtri(i,2)+newtri(i,5)+newtri(i,8))/3.0
//            newcen(i,3) = (newtri(i,3)+newtri(i,6)+newtri(i,9))/3.0
//            CYCLE
//         endif
// C       #-- don't subidivide small triangles
// C       #-- 3 original vertices
//         C1(1) = tri(i,1)
//         C1(2) = tri(i,2)
//         C1(3) = tri(i,3)
//         C2(1) = tri(i,4)
//         C2(2) = tri(i,5)
//         C2(3) = tri(i,6)
//         C3(1) = tri(i,7)
//         C3(2) = tri(i,8)
//         C3(3) = tri(i,9)
// C       #-- 3 new vertices
//         C4(1) = (C1(1)+C2(1))/2.
//         C4(2) = (C1(2)+C2(2))/2.
//         C4(3) = (C1(3)+C2(3))/2.
//         C5(1) = (C1(1)+C3(1))/2.
//         C5(2) = (C1(2)+C3(2))/2.
//         C5(3) = (C1(3)+C3(3))/2.
//         C6(1) = (C2(1)+C3(1))/2.
//         C6(2) = (C2(2)+C3(2))/2.
//         C6(3) = (C2(3)+C3(3))/2.
// C       #-- 4 new triangles
// C       #   TRIANGLE 1
//         newtri(i,1)    = C1(1)
//         newtri(i,2)    = C1(2)
//         newtri(i,3)    = C1(3)
//         newtri(i,4)    = C4(1)
//         newtri(i,5)    = C4(2)
//         newtri(i,6)    = C4(3)
//         newtri(i,7)    = C5(1)
//         newtri(i,8)    = C5(2)
//         newtri(i,9)    = C5(3)
// C       #   TRIANGLE 2
//         newtri(N+i,1)  = C6(1)
//         newtri(N+i,2)  = C6(2)
//         newtri(N+i,3)  = C6(3)
//         newtri(N+i,4)  = C4(1)
//         newtri(N+i,5)  = C4(2)
//         newtri(N+i,6)  = C4(3)
//         newtri(N+i,7)  = C5(1)
//         newtri(N+i,8)  = C5(2)
//         newtri(N+i,9)  = C5(3)
// C       #   TRIANGLE 3
//         newtri(2*N+i,1)= C6(1)
//         newtri(2*N+i,2)= C6(2)
//         newtri(2*N+i,3)= C6(3)
//         newtri(2*N+i,4)= C4(1)
//         newtri(2*N+i,5)= C4(2)
//         newtri(2*N+i,6)= C4(3)
//         newtri(2*N+i,7)= C2(1)
//         newtri(2*N+i,8)= C2(2)
//         newtri(2*N+i,9)= C2(3)
// C       #   TRIANGLE 4
//         newtri(3*N+i,1)= C6(1)
//         newtri(3*N+i,2)= C6(2)
//         newtri(3*N+i,3)= C6(3)
//         newtri(3*N+i,4)= C3(1)
//         newtri(3*N+i,5)= C3(2)
//         newtri(3*N+i,6)= C3(3)
//         newtri(3*N+i,7)= C5(1)
//         newtri(3*N+i,8)= C5(2)
//         newtri(3*N+i,9)= C5(3)
// C       #-- compute new sizes
//         a = sqrt((C1(1)-C4(1))**2+(C1(2)-C4(2))**2+(C1(3)-C4(3))**2)
//         b = sqrt((C5(1)-C4(1))**2+(C5(2)-C4(2))**2+(C5(3)-C4(3))**2)
//         c = sqrt((C1(1)-C5(1))**2+(C1(2)-C5(2))**2+(C1(3)-C5(3))**2)
//         h = (a+b+c)/2d0
//         news(i) = sqrt(h*(h-a)*(h-b)*(h-c))
//         a = sqrt((C6(1)-C4(1))**2+(C6(2)-C4(2))**2+(C6(3)-C4(3))**2)
//         b = sqrt((C5(1)-C4(1))**2+(C5(2)-C4(2))**2+(C5(3)-C4(3))**2)
//         c = sqrt((C6(1)-C5(1))**2+(C6(2)-C5(2))**2+(C6(3)-C5(3))**2)
//         h = (a+b+c)/2d0
//         news(N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
//         a = sqrt((C6(1)-C4(1))**2+(C6(2)-C4(2))**2+(C6(3)-C4(3))**2)
//         b = sqrt((C2(1)-C4(1))**2+(C2(2)-C4(2))**2+(C2(3)-C4(3))**2)
//         c = sqrt((C6(1)-C2(1))**2+(C6(2)-C2(2))**2+(C6(3)-C2(3))**2)
//         h = (a+b+c)/2d0
//         news(2*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
//         a = sqrt((C6(1)-C3(1))**2+(C6(2)-C3(2))**2+(C6(3)-C3(3))**2)
//         b = sqrt((C5(1)-C3(1))**2+(C5(2)-C3(2))**2+(C5(3)-C3(3))**2)
//         c = sqrt((C6(1)-C5(1))**2+(C6(2)-C5(2))**2+(C6(3)-C5(3))**2)
//         h = (a+b+c)/2d0
//         news(3*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
// C       #-- new centers
//         newcen(i,1) = (C1(1)+C4(1)+C5(1))/3d0
//         newcen(i,2) = (C1(2)+C4(2)+C5(2))/3d0
//         newcen(i,3) = (C1(3)+C4(3)+C5(3))/3d0
//         newcen(N+i,1) = (C6(1)+C4(1)+C5(1))/3d0
//         newcen(N+i,2) = (C6(2)+C4(2)+C5(2))/3d0
//         newcen(N+i,3) = (C6(3)+C4(3)+C5(3))/3d0
//         newcen(2*N+i,1) = (C6(1)+C4(1)+C2(1))/3d0
//         newcen(2*N+i,2) = (C6(2)+C4(2)+C2(2))/3d0
//         newcen(2*N+i,3) = (C6(3)+C4(3)+C2(3))/3d0
//         newcen(3*N+i,1) = (C6(1)+C3(1)+C5(1))/3d0
//         newcen(3*N+i,2) = (C6(2)+C3(2)+C5(2))/3d0
//         newcen(3*N+i,3) = (C6(3)+C3(3)+C5(3))/3d0
// C       #-- new normals
//         newnor(i,1) = nor(i,1)
//         newnor(i,2) = nor(i,2)
//         newnor(i,3) = nor(i,3)
//         newnor(N+i,1) = nor(i,1)
//         newnor(N+i,2) = nor(i,2)
//         newnor(N+i,3) = nor(i,3)
//         newnor(2*N+i,1) = nor(i,1)
//         newnor(2*N+i,2) = nor(i,2)
//         newnor(2*N+i,3) = nor(i,3)
//         newnor(3*N+i,1) = nor(i,1)
//         newnor(3*N+i,2) = nor(i,2)
//         newnor(3*N+i,3) = nor(i,3)
// C       #-- new mus
//         newmu(i) = mu(i)
//         newmu(N+i) = mu(i)
//         newmu(2*N+i) = mu(i)
//         newmu(3*N+i) = mu(i)
//    20 continue
//       RETURN
//       END
// 
// 
// 
