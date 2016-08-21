#if !defined(__triang_marching_h)
#define __triang_marching_h

/*
  Library for triangulation using maching algorithm specialized for 
  closed surfaces and connected surfaces. Supports only genus 0 smooth 
  surfaces.
  
  Ref:
    * E. Hartmann, A marching method for the triangulation of surfaces, 
    The Visual Computer (1998) 14: 95-108
     
    * https://en.wikipedia.org/wiki/Surface_triangulation
    
  Author: April, August 2016
*/ 

#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <limits>

#include "../utils/utils.h"
#include "triang_mesh.h"

/*
  Triangulization of closed surfaces using maching algorithm.
*/
template <class T, class Tbody>
struct Tmarching: public Tbody {
  
  /* 
    Interval structure for vertex includes the point on the surface 
    and vector base in tangent plane
  */
  struct Tvertex {
    
    int index;   // index
    
    bool omega_changed; // true if frontal angle changed
    
    T norm,      // norm of the gradient
      omega,     // frontal angle
      r[3],      // point on the surface
      b[3][3];   // b[0] = t1, b[1] = t2, b[2] = n
  };
  
 
  /*
   Create internal vertex (copy point and generate base) 
   
   Input: 
    p[3] - point on the surface
    g[3] - gradient 
    
   Output:
     v - internal vertex
  */
  void create_internal_vertex(T r[3], T g[3], Tvertex & v){
        
    for (int i = 0; i < 3; ++i) v.r[i] = r[i];
    
    //
    // Define screen coordinate system: b[3]= {t1,t2,view}
    //

    T *t1 = v.b[0], 
      *t2 = v.b[1], 
      *n  = v.b[2],
      fac;
      
    fac = 1/(v.norm = utils::hypot3(g[0], g[1], g[2]));
    
    for (int i = 0; i < 3; ++i) n[i] = fac*g[i];
    
    //
    // creating base in the tangent plane
    //
    
    // defining vector t1
    if (std::abs(n[0]) >= 0.5 || std::abs(n[1]) >= 0.5){
      //fac = 1/std::sqrt(n[1]*n[1] + n[0]*n[0]);
      fac = 1/std::hypot(n[0], n[1]);
      t1[0] = fac*n[1];
      t1[1] = -fac*n[0];
      t1[2] = 0.0;
    } else {
      //fac = 1/std::sqrt(n[0]*n[0] + n[2]*n[2]);
      fac = 1/std::hypot(n[0], n[2]);
      t1[0] = -fac*n[2];
      t1[1] = 0.0;
      t1[2] = fac*n[0];
    }
    
    // t2 = n x t1
    t2[0] = n[1]*t1[2] - n[2]*t1[1];
    t2[1] = n[2]*t1[0] - n[0]*t1[2];
    t2[2] = n[0]*t1[1] - n[1]*t1[0];
  }
  
  /*
    Projecting a point r positioned near the surface onto surface anc 
    calculate vertex. The surface is defined as
     
      F = 0 == constrain
    
    Input:
      r - point near the surface
      max_iter - maximal number of iterations
    
    Output:
      v - vertex containing the point r and local base of tangent space
    
    Return:
      true: nr. of steps < max_iter
      false: otherwise
  */
  bool project_onto_potential(T ri[3], Tvertex & v, const int & max_iter){
    
    //
    // Newton-Raphson iteration to solve F(u_k - t grad(F))=0
    //
    
    int n = 0;
   
    T g[4], r[3] = {ri[0], ri[1], ri[2]}, t, dr1, r1, fac;
    
    // decreasing precision is dangerous as it can miss the surface
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();
        
    do {
    
      // g = (grad F, F) 
      this->grad(r, g);
      
      // fac = F/|grad(F)|^2
      fac = g[3]/utils::norm2(g);
      
      // dr = F/|grad(F)|^2 grad(F) 
      // r' = r - dr 
      dr1 = r1 = 0;        
      for (int i = 0; i < 3; ++i) {
        
        r[i] -= (t = fac*g[i]);
        
        // calc. L_infty norm of vec{dr}
        if ((t = std::abs(t)) > dr1) dr1 = t;
        
        // calc L_infty of of vec{r'}
        if ((t = std::abs(r[i])) > r1) r1 = t;
      }
      
    } while (dr1 > eps*r1 + min && ++n < max_iter);
    
    // creating vertex
    create_internal_vertex(r, g, v);
    
    return (n < max_iter);
  }
  
  /*
    Projecting a point r positioned near the surface onto surface anc 
    calculate vertex. The surface is defined as
     
      F = 0 == constrain
    
    Input:
      ri - point near the surface
      max_iter - maximal number of iterations
    
    Output:
      r[3] - vertex points
      n[3] - normal at vertex
    
    Return:
      true: nr. of steps < max_iter
      false: otherwise
  */
  
  bool project_onto_potential(T ri[3], T r[3], T n[3], const int & max_iter, T *gnorm = 0){
    
    //
    // Newton-Raphson iteration to solve F(u_k - t grad(F))=0
    //
    
    int nr_iter = 0;
   
    T g[4], t, dr1, r1, fac;
    
    // decreasing precision is dangerous as it can miss the surface
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();
    
    if (r != ri) for (int i = 0; i < 3; ++i) r[i] = ri[i];
         
    do {
    
      // g = (grad F, F) 
      this->grad(r, g);
      
      // fac = F/|grad(F)|^2
      fac = g[3]/utils::norm2(g);
      
      // dr = F/|grad(F)|^2 grad(F) 
      // r' = r - dr 
      dr1 = r1 = 0;        
      for (int i = 0; i < 3; ++i) {
        
        r[i] -= (t = fac*g[i]);
        
        // calc. L_infty norm of vec{dr}
        if ((t = std::abs(t)) > dr1) dr1 = t;
        
        // calc L_infty of of vec{r'}
        if ((t = std::abs(r[i])) > r1) r1 = t;
      }
      
    } while (dr1 > eps*r1 + min && ++nr_iter < max_iter);
    
    // creating simplified vertex, 
    // note: std::hypot(,,) is comming in C++17
    
    if (gnorm)
      fac = 1/(*gnorm = utils::hypot3(g[0], g[1], g[2])); 
    else 
      fac = 1/utils::hypot3(g[0], g[1], g[2]);
    
    for (int i = 0; i < 3; ++i) n[i] = fac*g[i];
    
    return (nr_iter < max_iter);
  }
  
  
  /*
    Distance between the two 3D vectors.
    
    Input:
      a,b - vectors
    
    Return:
      |a-b|_2 -- L2 norm of th difference of vectors
  */
  
  T dist(T *a, T *b){
    //T s = 0;
    //for (int i = 0; i < 3; ++i) s += sqr(a[i] - b[i]);
    //return s;
    
    // std::hypot(,,) is comming in C++17
    return utils::hypot3(a[0] - b[0], a[1] - b[1], a[2] - b[2]); 
  } 
  

  Tmarching(void *params) : Tbody(params) { }
  
  /*
    Triangulization using marching method of genus 0 closed and surfaces
  
    Input: 
      init_r[3] - initial position 
      init_g[3] - initial gradient
      delta - size of triangles edges projected to tangent space
      max_triangles - maximal number of triangles used
    Output:
      V - vector of vertices
      NatV - vector of normals at vertices (read N at V)
      Tr - vector of triangles
      GatV - norm of the gradient at vertices
  */ 
  bool triangulize(
    T init_r[3],
    T init_g[3],
    const T & delta, 
    const unsigned & max_triangles, 
    std::vector <T3Dpoint<T>> & V,
    std::vector <T3Dpoint<T>> & NatV,
    std::vector <T3Dpoint<int>> & Tr,
    std::vector<T> * GatV = 0
    ) 
  {
    
    V.clear();
    Tr.clear();
    
    const int max_iter = 100;
     
    T qk[3];
    
    Tvertex v, vk, *vp, Pi[6];
    
    // construct the vector base
    create_internal_vertex(init_r, init_g, v);
    //v.index = 0; v.omega_changed = true;  // NOT NEEDED!
   
    // add vertex to the set, index 0
    
    //V.emplace_back(v.r, v.b[2]);        // saving only (r, normal)
    V.emplace_back(v.r);                  // saving only r
    if (GatV) GatV->emplace_back(v.norm); // saving g
    NatV.emplace_back(v.b[2]);            // saving only normal
    
    //
    // Create initial frontal polygon
    // Step 0:
    //
    typedef std::vector<Tvertex> Tfront_polygon;
    
    Tfront_polygon P; // front polygon, working here as circular list
    
    T s, c, st, ct, sa[6], ca[6];
    
    utils::sincos_array(5, utils::M_PI3, sa, ca, delta);
      
    for (int k = 0; k < 6; ++k){
      
      for (int i = 0; i < 3; ++i) 
        qk[i] = v.r[i] + ca[k]*v.b[0][i] + sa[k]*v.b[1][i];
        
      if (!project_onto_potential(qk, vk, max_iter)){
        std::cerr << "Warning: Projection did not converge\n";
      }  
      
      vk.index = k + 1;  // = V.size();
      vk.omega_changed = true;
      P.push_back(vk);
                  
      //V.emplace_back(vk.r, vk.b[2]);
      
      V.emplace_back(vk.r);                     // saving only r
      if (GatV) GatV->emplace_back(vk.norm);    // saving norm
      NatV.emplace_back(vk.b[2]);               // saving only normal
    }
    
    //
    // Creating initial hexagon
    //
    for (int k = 0; k < 5; ++k) Tr.emplace_back(0, k + 1, k + 2);
    Tr.emplace_back(0, 6, 1);
    
    //
    //  Triangulization of genus 0 surfaces
    //
    int n, nt;
    
    T domega, omega, omega_min, t, tt;
    
    typename Tfront_polygon::iterator 
            it_min, it_begin, it_last, it, it_prev, it_next; 
    
    do {
      
      //
      // Calculate the front angles and choose the point with the smallest
      // Step 1
      //

      n = P.size();
      
      std::cerr << Tr.size() << '\t' <<  n << std::endl;
      
      // set it_prev, it, it_next: as circular list
      it = it_next = it_begin = P.begin();
      if (n > 1) ++it_next;
      it_last = P.end(); it_prev = --it_last; 
      
      omega_min = utils::M_2PI; 
      
      while (--n >= 0) { 
        
        if (it -> omega_changed) { // calculate frontal angle if need
           
          c = s = ct = st = 0;
          for (int i = 0; i < 3; ++i) {
            t  = it_prev->r[i] - it->r[i];  // = dr1[i], dr1 = p_prev - p_cur
            c += t*it->b[0][i];             // = dr1[i]*t1[i]
            s += t*it->b[1][i];             // = dr1[i]*t2[i]
          
            tt  = it_next->r[i] - it->r[i];  // = dr2[i], dr2 = p_next - p_cur
            ct += tt*it->b[0][i];            // = dr2[i]*t1[i]
            st += tt*it->b[1][i];            // = dr2[i]*t2[i]
          }
          
          // = arg[ dr1.dr2 + I k.(dr1 x dr2) ]  
          // omega = atan2(st,ct) - atan2(s,c); 
          omega = std::atan2(c*st - s*ct, c*ct + s*st);
          
          // omega = omega mod 2 Pi (offset 0)        
          if (omega < 0) omega += utils::M_2PI;

          it -> omega = omega; 
          it -> omega_changed = false;
          
        } else  omega = it -> omega;
        
        
        // saving the minimal value of omega        
        if (omega < omega_min) {
          it_min = it;
          omega_min = omega;
        }
           
        // cyclic permutation of pointers
        it_prev = it;
        it = it_next;
        if (it_next == it_last) it_next = it_begin; else ++it_next;
      }

      //
      // Discuss the point with the minimal angle 
      // Step 3
      //
      
      //std::cerr << "omega_min=" << omega_min << std::endl;
      
      // number of triangles to be generated
      nt = int(omega_min/utils::M_PI3) + 1;   
      domega = omega_min/nt;
      
      if (domega < 0.8 && nt > 1) domega = omega_min/(--nt);
      
     // This does not help !!
     // else
     // if (nt == 1 && domega > 0.8 && dist(it_prev->r, it_next->r) > 1.2*delta) {
     //   domega = omega_min/(++nt);
     // }
      
      // This does not help !!
      //if (omega_min < 3 && 
      //   (dist(it_prev->r, it_min->r) < 0.5*delta || 
      //    dist(it_next->r, it_min->r) < 0.5*delta)) { 
      //  nt = 1;
      //} 
      
      // !!! there are some additional recommendations
      
      //std::cerr << "nt=" << nt << std::endl;
      
      // prepare pointers to vertices in P
      it_prev = it_next = it_min;
      if (it_min != it_begin) --it_prev; else it_prev = it_last;
      if (it_min != it_last) ++it_next; else it_next = it_begin; 
      
      it_prev->omega_changed = true;
      it_next->omega_changed = true;
        
      if (nt > 1) {
        
        // projection of dr = p_next - p_min to tangent space
        //  c = dr.t1
        //  s = dr.t2 
        c = s = 0;    
        for (int i = 0; i < 3; ++i){
          t = it_prev->r[i] - it_min->r[i];   // = dr[i]
          c += t*it_min->b[0][i];             // = dr[i]*t1[i]
          s += t*it_min->b[1][i];             // = dr[i]*t2[i]
        }
        
        // returning fac*(sin(k domega), cos(k domega)) 
        // where fac = delta/|(c, s)|
        //calc_sincos(nt - 1, domega, sa, ca, delta/std::sqrt(c*c + s*s));
        utils::sincos_array(nt - 1, domega, sa, ca, delta/std::hypot(c, s));
        
        vp = Pi;        // new front from it_min 
        n = V.size();   // size of the set of vertices
                
        for (int k = 1; k < nt; ++k, ++n, ++vp){
          
          // rotate in tangent plane
          ct = c*ca[k] - s*sa[k];
          st = c*sa[k] + s*ca[k];

          // forming point on tangent plane
          for (int i = 0; i < 3; ++i)
            qk[i] = it_min->r[i] + it_min->b[0][i]*ct + it_min->b[1][i]*st;

          if (!project_onto_potential(qk, *vp, max_iter)){
            
            T g[4];
            
            std::cerr << "Warning: Projection did not converge\n";
            
            this->grad(qk, g);
            
            std::cerr.precision(16);
            
            std::cerr 
              << "Start\n"
              << qk[0] << ' ' << qk[1] << ' ' << qk[2] << '\n'
              << g[0]  << ' ' << g[1]  << ' ' << g[2]  << '\n'
              << g[3]  << '\n';
              
          
            this->grad(vp->r, g);
            
            std::cerr 
              << "End\n"
              << vp->r[0] << ' ' << vp->r[1] << ' ' << vp->r[2] << '\n'
              << g[0] << ' ' << g[1] << ' ' << g[2] << '\n'
              << g[3] << '\n';
          }
          
          vp->index = n; // = V.size();
          vp->omega_changed = true;
          
          // V.emplace_back(vp->r, vp->b[2]);  
          V.emplace_back(vp->r);                    // saving only r
          if (GatV) GatV->emplace_back(vp->norm);   // saving g
          NatV.emplace_back(vp->b[2]);              // saving only normal
          
          Tr.emplace_back((k == 1 ? it_prev->index : n-1), n, it_min->index);
        }
        
        // n = V.size();
        Tr.emplace_back(n - 1, it_next->index, it_min->index);
        
        *(it_min++) = *Pi;
        
        P.insert(it_min, Pi + 1, Pi + nt - 1);
        
      } else {
        
        Tr.emplace_back(it_prev->index, it_next->index, it_min->index);
        
        P.erase(it_min);
      }
        
      //std::cerr << "P.size()=" << P.size() << std::endl;
      
    } while (P.size() > 3 && Tr.size() < max_triangles);
    
    //
    // Processing the last three vertices 
    //
    if (Tr.size() < max_triangles - 1) {
      #if 0 // generic
      it = P.begin();
      
      int ind[3];
      
      ind[0] = (it++)->index;
      ind[1] = (it++)->index;
      ind[2] = it->index;
      
      Tr.emplace_back(ind);      
      #else // P being vector
      Tr.emplace_back(P[0].index, P[1].index, P[2].index);
      #endif
    }
    
    return (Tr.size() < max_triangles);
  }
  
  /*
    Calculate the central_points of triangles i.e. barycenters 
    projected down to surface and normals at that points
    
      properties:
        areas of triangles and a normal
      
    Input:
      V - vector of vertices
      Tr - vector of triangles - connectivity matrix
      
    Output:
      C - central points 
      NatC - normals at central points
      GatC - norm of gradient at cetral points
  */
  void central_points(
    std::vector <T3Dpoint<T>> & V,
    std::vector <T3Dpoint<int>> & Tr,
    
    std::vector <T3Dpoint<T>> * C = 0,
    std::vector <T3Dpoint<T>> * NatC = 0,
    std::vector <T> * GatC = 0
  ) 
  {
    if (C == 0 && NatC == 0 && GatC == 0) return;
        
    if (C) {
      C->clear();
      C->reserve(Tr.size());
    }
    
    if (NatC) {
      NatC->clear();
      NatC->reserve(Tr.size());
    }
    
    T g, *p_g  = 0;
    if (GatC) {
      GatC->clear();
      GatC->reserve(Tr.size());
      p_g = &g;
    }

    const int max_iter = 100;
    
    T *tp, v[3], n[3], q[3], r[3][3];
  
    int i, j;

    for (auto && t: Tr) {
      
      //
      // store points of the vector
      //
      for (i = 0; i < 3; ++i) {
        tp = V[t[i]].data;
        for (j = 0; j < 3; ++j) r[i][j] = tp[j];
      }
      
      //
      // central point    
      //
      for (i = 0; i < 3; ++i) 
        q[i] = (r[0][i] + r[1][i] + r[2][i])/3;
      
      if (project_onto_potential(q, v, n, max_iter, p_g)){ 
        if (C) C->emplace_back(v);
        if (NatC) NatC->emplace_back(n); 
        if (GatC) GatC->emplace_back(g);
      } else std::cerr << "Warning: Projection did not converge\n";
    }    
  }
      
}; // class marching


#endif // #if !defined(__triang_marching_h)
