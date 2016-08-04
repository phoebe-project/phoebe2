#if !defined(__reflection_h)
#define __reflection_h

/*
  Library for solving the stellar radiosity problem. Wilson called this reflection effects. Basically is problem of evaluating radiosity by taking into account intrinsic radiant exitance and reflection of it from surface of the stars.

  Author: Martin Horvat, July 2016  

  Ref:
  
  Stars:
    * Wilson, R. E.  Accuracy and efficiency in the binary star reflection effect, Astrophysical Journal,  356, 613-622, 1990 June
    * Budaj J., The reflection effect in interacting binaries or in planet–star systems, The Astronomical Journal, 141:59 (12pp), 2011 February
    * Josef Kallrath, Eugene F. Milone (auth.)-Eclipsing Binary Stars - Modeling and Analysis (Springer-Verlag New York, 2009)

  Radiometry:
    * http://light-measurement.com/calculation-of-radiometric-quantities/
    * https://en.wikipedia.org/wiki/Radiometry
    * https://en.wikipedia.org/wiki/Radiosity_(radiometry)
*/


#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
	
#include "../utils/utils.h"
#include "ld_models.h"
#include "../triang/triang_marching.h"

/*
  Check if the line segment
    
    r = c1 + (c2 - c1)t      t in [0,1]
  
  cuts the triangles with vertices
    
    (v[0], v[1], v[2])
*/

template <class T>
bool triangle_cuts_line(T *v[3], T c1[3], T c2[3]){
  
  // define transpose matrix A^T = [r2-r1, r3-r1, c1 -c2]
  T A[3][3], b[3], x[3];
  
  for (int i = 0; i < 3; ++i){
    A[0][i] = v[1][i] - v[0][i];
    A[1][i] = v[2][i] - v[0][i];
    A[2][i] = c1[i] - c2[i];
    b[i] = c1[i] - v[0][i];
  }
  
  // solve system A^t x = b or x^t A = b^t
  if (!utils::solve3D(b, A, x)) {
    std::cerr << "triangle_cuts_line::This should not happen\n";
    return false;
  }
  
  // if 0<=x[i]<=1 then there is intersection
  for (int i = 0; i < 3; ++i) 
    if (x[i] < 0 || x[i] > 1) return false;
  
  return (x[0] + x[1]<=1);
}

template <class T>
bool triangle_cuts_line(T n[3], T *v[3], T c1[3], T c2[3]){
  
  //
  // Is line cuts the plane in which the triangle lies
  //
  
  T f[3], g[3];
  
  for (int i = 0; i < 3; ++i) {
    f[i] = c1[i] - v[0][i];
    g[i] = c1[i] - c2[i];
  }
  
  T t3 = utils::dot3D(g,n);
  
  if (t3 == 0) return false;
  
  t3 = utils::dot3D(f,n)/t3;
  
  if (t3 < 0 || t3 > 1) return false; 
  
  //
  // Check if the point of intesection is in the triangle
  //
  
  T a[3], b[3], c[3];
  
  for (int i = 0; i < 3; ++i) {
    a[i] = v[1][i] - v[0][i];
    b[i] = v[2][i] - v[0][i];
    c[i] = f[i] - t3*g[i];
  }
  
  // find largest/most pronounced components of the normal
  int l = 0;
  {
    T t, m = std::abs(n[0]);
    for (int i = 1; i < 3; ++i) if ((t = std::abs(n[i])) > m) {
      m = t;
      l = i;
    } 
  }
  
  // define components that need to be taken into account
  int j, k;
  
  if (l == 0) {
    j = 1; 
    k = 2;
  } else if (l == 1){
    j = 0;
    k = 2;
  } else { // l == 2
    j = 0;
    k = 1;
  }
  
  // solve equation:
  //  [ a[j] b[j] ] [t1] = [c[j]]
  //  [ a[k] b[k] ] [t2] = [c[k]]
  
  T det = 1/(a[j]*b[k] - a[k]*b[j]);
  
  T t1 = det*(b[k]*c[j] - b[j]*c[k]);
  if (t1 < 0 || t1 > 1) return false;
   
  T t2 = det*(a[j]*c[k] - a[k]*c[j]);
  if (t2 < 0 || t2 > 1) return false;
  
  return (t1 + t2 <= 1);
}

/*
  Solving the Wilson's limb-darkened radiosity problem.
    
  Input:

    V - vector of vertices
    Tr - vector of triangles
    NatT - vector of normals at triangles
    A - vector of areas of triangles
    LDmodels - vector of limb darkening models in use 
    LDidx - vector of indices of models used on each of triangles
    
    epsC - threshold for permitted cos(theta)
              cos(theta_i) > epsC to be considered in line-of-sight
           ideally epsC = 0, epsC=0.00872654 corresponds to 89.5deg
  Output:
    Fmat - matrix of LD view factors
*/ 

template <class T>
struct Tmat_elem {
  int i, j;
  T F;
  Tmat_elem() {}
  Tmat_elem(int i, int j, const T &F) : i(i), j(j), F(F) {}
};
  

template <class T>
void triangle_mesh_radiosity_wilson(
  std::vector <T3Dpoint<T>> & V,                  // inputs 
  std::vector <T3Dpoint<int>> & Tr,
  std::vector <T3Dpoint<T>> & NatT,
  std::vector <T> & A,
  std::vector <TLDmodel<T>*> & LDmodels,           
  std::vector <int> & LDidx,

  std::vector <Tmat_elem<T>> & Fmat,              // output
  const T & epsC = 0.00872654) {

  //
  // Calculate the centroids of triangles
  //  
  
  int Nt = Tr.size();

  T *CatT = new T [3*Nt];
  
  {
    T *c = CatT, *v[3];
        
    for (auto && t : Tr){
    
      // pointers to vertices
      for (int k = 0; k < 3; ++k) v[k] = V[t[k]].data;   
      
      // centroid
      for (int k = 0; k < 3; ++k) 
        c[k] = (v[0][k] + v[1][k] + v[2][k])/3;
      
      c += 3;
    }
  }
  
  // 
  // Calculate depth and view-factor matrix DF thereby
  // using over-simplified check visibility, where only line-of-sight 
  // between centroids of triangles is checked
  //

  struct Tp {
    
    int i;
    
    T h, F;
    
    bool operator < (const Tp & rhs) const { return h < rhs.h; } 
  };
 
  // depth and view-factor matrix
  std::vector<std::vector<Tp>> DF(Nt);
  
  {

    T tmp, s, s2, *n, *n1, *c = CatT, *c1, a[3];
    
    Tp p, p1;
    
    for (int i = 0; i < Nt; ++i, c += 3 ) {   // loop over the triangles Ti
      
      n = NatT[i].data;             // normal of Ti

      c1 = CatT;
      for (int j = 0; j < i; ++j, c1 += 3) {   // loop over the triangles Tj
        
        //  
        // Check if it is possible to see the centroid of Tj from 
        // the centroid of Ti and vice versa
        //
        
        // vector connected centroids c -> c1: a = c1 - c
        utils::sub3D(c1, c, a);
        
        //s = utils::hypot3(a); <-- to slow
        s = std::sqrt(s2 = utils::norm2(a));
        
        // looking at Tj from Ti
        if ((p.h = utils::dot3D(n, a)) <= (tmp = epsC*s)) continue;
       
        n1 = NatT[j].data;            // normal of Tj
               
        // looking at Ti from Tj
        if ((p1.h = -utils::dot3D(n1, a)) <= tmp) continue;
      
        // conclusion: probably Tj illuminates Ti and vice versa
         
        //     
        // calculate Lambert view factor
        //
        p.F = p1.F = p.h*p1.h/(s2*s2);
        
        //
        // calculate LD view factors
        //
        
        // looking at Tj from Ti
        p.i = i;
        p.F *= A[j]*LDmodels[LDidx[i]]->F(p.h/s);
        
        // looking at Ti from Tj
        p1.i = j;
        p1.F *= A[i]*LDmodels[LDidx[j]]->F(p1.h/s);
        
        //
        // storing the results in depth and view-factor matrix
        //
           
        DF[p.i].push_back(p1);  // registering pair (p.i, p1)
        DF[p1.i].push_back(p);  // registering pair (p1.i, p)

      }
    }
  }
  
  //
  // Check if the line of sign from centroids of triangles is obstructed
  // and generate reduced depth-view factor matrix DF
  // 
  
  {
    int *t, i = 0;
    
    bool ok_visible;
    
    T *c = CatT, *c1, *n, *v[3];
    
    for (auto && q : DF) {
                  
        // if there is one element visible there is no obstruction possible
        if (q.size() > 1) {
        
        // sorting w.r.t. depth from triangle with index p.first 
        std::sort(q.begin(), q.end());
          
        auto it_b = q.begin(), it = it_b + 1;
              
        // look over triangles and see is line-of-sight is obstructed
        while (it != q.end()) {
          
          // centroid of the triangle view from c
          c1 = CatT + 3*it->i;
          
          ok_visible = true;
          
          // check if line c1 <-> c is cut by triangle at less depth
          // from triangle with index p.first 
          for (auto it1 = it_b; ok_visible && it1 != it; ++it1) {
            
            // pointers to vertices
            t = Tr[it1->i].data;
            for (int k = 0; k < 3; ++k) v[k] = V[t[k]].data;
            
            // normal of the triangle
            n = NatT[it1->i].data;
            
            // check if triangle cuts the line
            //ok_visible = !triangle_cuts_line(v, c, c1);
            ok_visible = !triangle_cuts_line(n, v, c, c1);
          }
          
          // line-of-sight of triangles with indices (p.first, it->i) 
          // is obstructed, erasing *it element
          if (!ok_visible) {
            
            // erase conjugate pair (it->i, p.first) from DF[it->i]
            auto & z = DF[it->i];
            for (auto it1 = z.begin(), it1_e = z.end(); it1 != it1_e; ++it1)
              if (it1->i == i) {
                z.erase(it1);
                break;
              }

            // erase (u.first, it->i) from DF[p.first]
            it = q.erase(it);                      
          } else ++it;
        }
      }
      
      ++i;
      c += 3;
    }  
  }
  
  delete [] CatT;
  
  //
  // Generate LD view factor matrix F by collecting data 
  // from depth-view factor matrix
  // 
  
  Fmat.clear();
  
  int i = 0;
  for (auto && p : DF){
    for (auto && q : p) Fmat.emplace_back(i, q.i, q.F);
    ++i;
  }
}

/*
  Solving the radiosity equation 
    
    (1 - diag(R) F) M = M0
  
  where 
    R is vector of reflection coefficients, 
    M0 is vector of intrinsic radiant exitances from triangles
    M is vector of radiosity (intrinsic and reflection) of triangles
  
  Method: 
    Simple iteration
      
      M_{k+1} = M0  + diag(R) F M_{k}
    
    with initial condition
      M_0 = M0
  
  Input:
    Fmat - matrix of view factor 
    R - vector of albedo/reflection of triangles
    M0 - vector of intrisic radiant exitance of triangles
    epsM - relative precision of radiosity    
    max_iter - maximal number of iterations
 
  Output:
    M - vector of radiosity (intrinsic and reflection) of triangles
     
  Returns:
    true if we reached wanted relative precision, false otherwise
*/
template <class T>
bool solve_radiosity_equation(
  std::vector<Tmat_elem<T>> &Fmat,      // input
  std::vector<T> &R,  
  std::vector<T> &M0, 
  std::vector<T> &M,                    // output
  const T & epsM = 1e-12,
  const T & max_iter = 100) {
  
  int 
    Nt = R.size(),        // number of triangles
    it = 0,               // number of iterations
    size = Nt*sizeof(T);  // size of vectors in bytes
    
  T *buf = new T [2*Nt], *S0 = buf, *S1 = buf + Nt,   // prepare buffer
    *pM = M0.data(), *tS, t, dS, Smax;
  
  // initial condition
  memcpy(S0, pM, size);     
    
  do {
    
    // iteration step
    memcpy(S1, pM, size);
    
    for (auto && f: Fmat) S1[f.i] += R[f.i]*f.F*S0[f.j];
  
    // check convergence
    dS = Smax = 0;
    for (int j = 0; j < Nt; ++j) {
      if (S1[j] > Smax) Smax = S1[j];
      t = std::abs(S1[j] - S0[j]);
      if (t > dS)  dS = t;
    }
  
    //std::cerr << dS << '\t' << Smax << '\t' << dS/Smax << '\n';
    
    // swap pointers as to save previous step
    tS = S0, S0 = S1, S1 = tS;
  
  } while (dS > epsM*Smax && ++it < max_iter);
  
  
  // copy the results to the output
  M.assign(S0, S0 + Nt);

  delete [] buf;  

  return it < max_iter;  
}


#if 0
/* 
  Matrix in Compressed Row Storage (CRS) format 
  Ref: http://netlib.org/linalg/html_templates/node91.html 

*/
template <class T>
struct Tmatrix_CRS {
  int n;
  
  T *val;
  
  int *col_ind, *row_ptr;
};


/*
  Solving the radiosity equation 
    
    (1 - diag(rho) F) M = M0
  
  where 
    rho is vector of reflection coefficients, 
    M0 is vector of intrinsic radiant exitances from triangles
    M is vector of radiosity (intrinsic and reflection) of triangles
  
  
  The matrix F is given in Compressed Row Storage (CRS) format

  Returns:
    true if we reached wanted relative precision, false otherwise

  Ref: 
  * http://www.netlib.org/linalg/html_templates/node98.html
*/
template <class T>
bool solve_radiosity_equation(
  Tmatrix_CRS<T> &Fmat, 
  T *rho,  
  T *M0, 
  T *M,
  const T & epsM = 1e-12,
  const T & max_iter = 100) {
  
  int it = 0;
  
  // TODO ??????
  
  return it < max_iter;  
}

#endif

#endif //#if !defined(__reflection_h) 
