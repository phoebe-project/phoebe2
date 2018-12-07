#pragma once

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

#include "utils.h"
#include "triang_mesh.h"

/*
  Triangulation of closed surfaces using maching algorithm.
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

  typedef std::vector<Tvertex> Tfront_polygon;
  typedef std::pair<int,int> Tbad_pair;

  bool precision;

  /*
   Create internal vertex (copy point and generate base)

   Input:
    p[3] - point on the surface
    g[3] - gradient

   Output:
     v - internal vertex
  */

  //#define DEBUG
  void create_internal_vertex(T r[3], T g[3], Tvertex & v, const T & phi = 0){

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

    //
    // Rotate in tangent plane
    //
    if (phi != 0) {

      T q1[3], q2[3];

      for (int i = 0; i < 3; ++i) {
        q1[i] = t1[i];
        q2[i] = t2[i];
      }

      T s, c;

      utils::sincos(phi, &s, &c);

      for (int i = 0; i < 3; ++i) {
        t1[i] = c*q1[i] + s*q2[i];
        t2[i] = -s*q1[i] + c*q2[i];
      }
    }

    #if defined(DEBUG)
    std::cerr<< "create_internal_vertex:\n";
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) std::cerr << v.b[i][j] << ' ';
      std::cerr << '\n';
    }
    #endif
  }

  #if defined(DEBUG)
  #undef DEBUG
  #endif

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

  //#define DEBUG
  bool project_onto_potential(T ri[3], Tvertex & v, const int & max_iter, T *ni = 0, const T & eps = 20*std::numeric_limits<T>::epsilon()){

    //
    // Newton-Raphson iteration to solve F(u_k - t grad(F))=0
    //

    int n;

    T g[4], r[3], t, dr1, r1, fac;

    // decreasing precision is dangerous as it can miss the surface
    const T min = 10*std::numeric_limits<T>::min();

    do {
      n = 0;
      
      for (int i = 0; i < 3; ++i) r[i] = ri[i];
      
      do {

        // g = (grad F, F)
        this->grad(r, g, precision);

        if (ni) {
          T sum = 0;
          for (int i = 0; i < 3; ++i) sum += ni[i]*g[i];
          if (sum < 0) return false;
        }

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

      #if defined(DEBUG)
      std::cerr
        << "PROJ: g=(" << g[0] << "," << g[1]<< "," << g[2] <<  "," << g[3] << ")"
        << " r=(" << r[0] << "," << r[1]<< "," << r[2] << ")"
        << " " << dr1 <<" "<< precision << '\n';
      #endif

      if (!precision && n >= max_iter)
        precision = true;
      else break;

    } while (1);

    // creating vertex
    this->grad_only(r, g, precision);
    create_internal_vertex(r, g, v);

    return (n < max_iter);
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif

  /*
    Slide along the iso-surface from point ri with normal gi on the surface in direction ui
    for distance a. The surface is defined as

      F = 0 == constrain

    Input:
      ri - point near the surface
      gi - gradient or normal at point r
      ui - direction of sliding
      a - distance to slide
      max_iter - maximal number of iterations

    Output:
      v - vertex containing the last point and local base of tangent space

    Return:
      true: nr. of steps < max_iter
      false: otherwise
  */

  //#define DEBUG
  bool slide_over_potential(T ri[3], T gi[3], T ui[3], T a, Tvertex & v, const int & max_iter){

    #if defined(DEBUG)
    std::cerr << "Sliding...";
    #endif

    //
    // Plane of sliding
    //
    T n[3];

    utils::cross3D(gi, ui, n);

    #if defined(DEBUG)
    std::cerr << "slide_over_potential::init\n"
      << " gi=" << gi[0] << ' ' << gi[1] << ' ' << gi[2]
      << " ui=" << ui[0] << ' ' << ui[1] << ' ' << ui[2]
      << " n=" << n[0] << ' ' << n[1] << ' ' << n[2] << '\n';
    #endif

    T r[3] = {ri[0], ri[1], ri[2]};

    {
      int N = 10;

      T fac, da = a/N, g[3], r1[3], t[3], k[4][3];

      //
      // N steps of RK iterations
      //
      for (int i = 0; i < N; ++i) {

        // 1. step
        if (i == 0)
          for (int j = 0; j < 3; ++j) g[j] = gi[j]; // if g is manually set
        else
          this->grad_only(r, g, precision);

        utils::cross3D(n, g, t);
        fac = da/utils::hypot3(t);
        for (int j = 0; j < 3; ++j) k[0][j] = fac*t[j];

        // 2. step
        for (int j = 0; j < 3; ++j) r1[j] = r[j] + 0.5*k[0][j];
        this->grad_only(r1, g, precision);
        utils::cross3D(n, g, t);
        fac = da/utils::hypot3(t);
        for (int j = 0; j < 3; ++j) k[1][j] = fac*t[j];

        // 3. step
        for (int j = 0; j < 3; ++j) r1[j] = r[j] + 0.5*k[1][j];
        this->grad_only(r1, g, precision);
        utils::cross3D(n, g, t);
        fac = da/utils::hypot3(t);
        for (int j = 0; j < 3; ++j) k[2][j] = fac*t[j];

        // 4. step
        for (int j = 0; j < 3; ++j) r1[j] = r[j] + k[2][j];
        this->grad_only(r1, g, precision);
        utils::cross3D(n, g, t);
        fac = da/utils::hypot3(t);
        for (int j = 0; j < 3; ++j) k[3][j] = fac*t[j];

        // joining steps together
        for (int j = 0; j < 3; ++j)
          r[j] += (k[0][j] + 2*(k[1][j] + k[2][j]) + k[3][j])/6;

        #if defined(DEBUG)
        T g1[4];
        this->grad(r, g1, precision);
        std::cerr << "RK: g=" << g1[3] << '\n';
        #endif
      }
    }


    //
    // Newton-Raphson iteration to solve F(u_k - t grad(F))=0
    //
    int it = 0;

    T g[4], t, dr1, r1, fac;

    // decreasing precision is dangerous as it can miss the surface
    const T eps = 10*std::numeric_limits<T>::epsilon();
    const T min = 10*std::numeric_limits<T>::min();

    do {

      do {

        // g = (grad F, F)
        this->grad(r, g, precision);

        if (gi) {
          T sum = 0;
          for (int i = 0; i < 3; ++i) sum += gi[i]*g[i];
          if (sum < 0) return false;
        }

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

      } while (dr1 > eps*r1 + min && ++it < max_iter);

      #if defined(DEBUG)
      std::cerr.precision(16);
      std::cerr
        << "PROJ: g=(" << g[0] << "," << g[1]<< "," << g[2] <<  "," << g[3] << ")"
        << " r=(" << r[0] << "," << r[1]<< "," << r[2] << ")"
        << " " << dr1 <<" "<< precision << " " << it << '\n';
      #endif

      if (!precision && it >= max_iter) {
        precision = true;
        it = 0;
      } else break;

    } while (1);

    // creating vertex
    this->grad_only(r, g, precision);
    create_internal_vertex(r, g, v);

    return (it < max_iter);
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif



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

  // #define DEBUG
  bool project_onto_potential(T ri[3], T r[3], T n[3], const int & max_iter, T *gnorm = 0, const T & eps = 20*std::numeric_limits<T>::epsilon()){

    //
    // Newton-Raphson iteration to solve F(u_k - t grad(F))=0
    //

    int nr_iter;

    T g[4], t, dr1, r1, fac;

    // decreasing precision is dangerous as it can miss the surface
    const T min = 10*std::numeric_limits<T>::min();

    do {
      
      nr_iter  = 0;
      
      for (int i = 0; i < 3; ++i) r[i] = ri[i];
      
      do {

        // g = (grad F, F)
        this->grad(r, g, precision);

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

      #if defined(DEBUG)
      std::cerr
        << "PROJ: g=(" << g[0] << "," << g[1]<< "," << g[2] <<  "," << g[3] << ")"
        << " r=(" << r[0] << "," << r[1]<< "," << r[2] << ")"
        << " " << dr1 <<" "<< precision << '\n';
      #endif

      if (!precision && nr_iter >= max_iter)
        precision = true;
      else break;

    } while(1);

    this->grad_only(r, g, precision);

    // creating simplified vertex,
    // note: std::hypot(,,) is comming in C++17

    if (gnorm)
      fac = 1/(*gnorm = utils::hypot3(g[0], g[1], g[2]));
    else
      fac = 1/utils::hypot3(g[0], g[1], g[2]);

    for (int i = 0; i < 3; ++i) n[i] = fac*g[i];

    return (nr_iter < max_iter);
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif

  /*
    Distance between the two 3D vectors.

    Input:
      a,b - vectors

    Return:
      |a-b|_2 -- L2 norm of th difference of vectors
  */

  T dist(T *a, T *b){
    // std::hypot(,,) is comming in C++17
    return utils::hypot3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }

  T dist2(T *a, T *b){
    T s = 0;
    for (int i = 0; i < 3; ++i) s += utils::sqr(a[i] - b[i]);
    return s;
  }


  int split_angle(Tvertex & v_prev, Tvertex & v, Tvertex & v_next, T *a) {

    T q[3][2] = {{0.,0.}, {0.,0.}, {0.,0.}};

    // projecting vectors onto tangent plane of vertex v
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        q[0][i] += (v_prev.r[j] - v.r[j])*v.b[i][j];
        q[1][i] += (v_next.r[j] - v.r[j])*v.b[i][j];
        q[2][i] += a[j]*v.b[i][j];
      }

    T s[2] = {
      utils::cross2D(q[2], q[0]),
      utils::cross2D(q[1], q[2])
    };

    if (s[0] > 0 && s[1] > 0) return 1;

    if (s[0] < 0 && s[1] < 0) return -1;

    return 0;
  }


  Tmarching(T *params) : Tbody(params) { }

  /*
    Triangulation using marching method of genus 0 closed and surfaces

    Input:
      init_r[3] - initial position
      init_g[3] - initial gradient
      delta - size of triangles edges projected to tangent space
      max_triangles - maximal number of triangles used
      init_phi - rotation of the initial hexagon
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
    std::vector<T> * GatV = 0,
    const T &init_phi = 0
    )
  {

    // start with normal precision defined by T
    precision = false;

    V.clear();
    Tr.clear();

    const int max_iter = 100;

    T qk[3];

    Tvertex v, vk, *vp, Pi[6];

    // construct the vector base
    create_internal_vertex(init_r, init_g, v, init_phi);

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

    Tfront_polygon P; // front polygon, working here as circular list

    T s, c, st, ct, sa[6], ca[6], u[3];

    utils::sincos_array(5, utils::m_pi3, sa, ca, delta);

    for (int k = 0; k < 6; ++k){

      for (int i = 0; i < 3; ++i)
        qk[i] = v.r[i] + (u[i] = ca[k]*v.b[0][i] + sa[k]*v.b[1][i]);

      if (
          !slide_over_potential(v.r, v.b[2], u, delta, vk, max_iter) &&
          !project_onto_potential(qk, vk, max_iter, v.b[2])
         ) {
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

      //std::cerr << Tr.size() << '\t' <<  n << std::endl;

      // set it_prev, it, it_next: as circular list
      it = it_next = it_begin = P.begin();
      if (n > 1) ++it_next;
      it_last = P.end(); it_prev = --it_last;

      omega_min = utils::m_2pi;

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
          if (omega < 0) omega += utils::m_2pi;

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
      nt = int(omega_min/utils::m_pi3) + 1;
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
            qk[i] = it_min->r[i] + (u[i] = it_min->b[0][i]*ct + it_min->b[1][i]*st);

          if (!project_onto_potential(qk, *vp, max_iter, it_min->b[2]) &&
              !slide_over_potential(it_min->r, it_min->b[2], u, delta, *vp, max_iter)) {

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
    Triangulization using marching method of genus 0 closed and surfaces
    Has
    -- additionals checks
    -- support multifronts

    Input:
      init_r[3] - initial position
      init_g[3] - initial gradient
      delta - size of triangles edges projected to tangent space
      max_triangles - maximal number of triangles used
      init_phi - rotation of the initial hexagon
    Output:
      V - vector of vertices
      NatV - vector of normals at vertices (read N at V)
      Tr - vector of triangles
      GatV - norm of the gradient at vertices
    Return:
     0 - no error
     1 - too triangles
     2 - problem with converges
  */ 
  int triangulize_full(
    T init_r[3],
    T init_g[3],
    const T & delta,
    const unsigned & max_triangles,
    std::vector <T3Dpoint<T>> & V,
    std::vector <T3Dpoint<T>> & NatV,
    std::vector <T3Dpoint<int>> & Tr,
    std::vector<T> * GatV = 0,
    const T &init_phi = 0
    )
  {

    // start with normal precision defined by T
    precision = false;
 
    // error 
    int error = 0;
  
    V.clear();
    Tr.clear();

    const int max_iter = 100;

    //
    // Create initial frontal polygon
    // Step 0:
    //
    typedef std::vector<Tvertex> Tfront_polygon;
    
    // list of frontal polygon, working here as circular list
    std::vector<Tfront_polygon> lP(1); 
    
    {
      Tvertex v, vk;

      // construct the vector base
      create_internal_vertex(init_r, init_g, v, init_phi);

      // add vertex to the set, index 0
      V.emplace_back(v.r);                  // saving only r
      if (GatV) GatV->emplace_back(v.norm); // saving g
      NatV.emplace_back(v.b[2]);            // saving only normal

      T sa[6], ca[6], qk[3], u[3];

      utils::sincos_array(5, utils::m_pi3, sa, ca, delta);
       
      for (int k = 0; k < 6 && error == 0; ++k){
        
        for (int i = 0; i < 3; ++i) 
          qk[i] = v.r[i] + (u[i] = ca[k]*v.b[0][i] + sa[k]*v.b[1][i]);

        if (!project_onto_potential(qk, vk, max_iter, v.b[2]) &&
            !slide_over_potential(v.r, v.b[2], u, delta, vk, max_iter)) {
          std::cerr << "Warning: Projection did not converge for initial frontal polygon.\n";
          error = 2;
        }  
        
        // store points into initial front
        vk.index = k + 1;  // = V.size();
        vk.omega_changed = true;
        lP[0].push_back(vk);

        V.emplace_back(vk.r);                     // saving only r
        if (GatV) GatV->emplace_back(vk.norm);    // saving norm
        NatV.emplace_back(vk.b[2]);               // saving only normal
      }

      //
      // Creating initial hexagon -- triangle faces in Tr
      //
      for (int k = 0; k < 5; ++k) Tr.emplace_back(0, k + 1, k + 2);
      Tr.emplace_back(0, 6, 1);
    }

    //
    //  Triangulization of genus 0 surfaces
    //

    T delta2 = 0.5*delta*delta;    // TODO: should be more dynamical
    
    do {

      // current front
      Tfront_polygon & P  = lP.back();

      do {

        //
        // Processing the last three vertices
        //
        if (P.size() == 3) {
          Tr.emplace_back(P[0].index, P[1].index, P[2].index);
          // erasing discussed front
          lP.pop_back();
          break;
        }

        // pointers associated to the front

        auto it_begin = P.begin(), it_end = P.end(), it_last = it_end - 1;

        //
        // If a non-neighboring vertices are to close form new fronts
        // Step 2
        //
        {
          int s;

          bool ok = true;

          T a[3];

          auto
            it = it_begin,
            it_next = it + 1,
            it_prev = it_last;

          while (1) {

            auto
              it1 = it + 2,
              it1_next = (it1 != it_last ? it1 + 1 : it_begin),
              it1_prev = it + 1,
              it1_last = (it == it_begin ? it_last - 1 : it_last);

            while (1) {

              // are on the side the object
              if (utils::dot3D(it->b[2], it1->b[2]) > 0) {

                utils::sub3D(it1->r, it->r, a);

                // if near enough and looking inside from it and from it1
                if (utils::norm2(a) < delta2) {

                  // check if same side of both edges and determine the side
                  // depending of it_prev -> it -> it_next circle
                  s = split_angle(*it_prev, *it, *it_next, a);

                  if (s != 0 && s*split_angle(*it1_prev, *it1, *it1_next, a) < 0) {

                    // create new last front
                    #if defined(DEBUG)
                    std::cerr
                      << "P.size=" << P.size()
                      << " lP.size=" << lP.size()
                      << " i=" << int(it - it_begin)
                      << " j=" << int(it1 - it_begin)
                      << " len=" << int(it1 + 1 - it)
                      << std::endl;
                    #endif

                    it->omega_changed = true;
                    it1->omega_changed = true;

                    Tfront_polygon P1(it, it1 + 1);

                    P.erase(it + 1, it1);

                    lP.push_back(P1);

                    ok = false;
                    break;
                  }
                }
              }

              if (it1 == it1_last) break;

              it1_prev = it1;
              it1 = it1_next;

              if (it1_next == it_last)
                it1_next = it_begin;
              else
                ++it1_next;
            }

            if (!ok || it + 2 == it_last) break;

            it_prev = it;
            it = it_next++;
          }

          // if new fronts are created then interrupt workflow
          if (!ok) break;
        }

        //
        // Calculate the front angles and choose the point with the smallest
        // Step 1
        //

        T omega_min = utils::m_2pi;

        typename Tfront_polygon::iterator it_min;

        {

          T omega, t, tt, c, s, st, ct;

          // set it_prev, it, it_next: as circular list
          auto it = it_begin, it_next = it + 1, it_prev = it_last;

          while (1) {

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
              if (omega < 0) omega += utils::m_2pi;

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

            if (it_next == it_begin) break;

            if (it_next == it_last)
              it_next = it_begin;
            else
              ++it_next;
          }
        }


        //
        // Discuss the point with the minimal angle
        // Step 3
        //

        {
          // prepare pointers to vertices in P
          auto
            it_prev = it_min,
            it_next = it_min;

          if (it_min != it_begin) --it_prev; else it_prev = it_last;
          if (it_min != it_last) ++it_next; else it_next = it_begin;

          // number of triangles to be generated
          int nt = int(omega_min/utils::m_pi3) + 1;

          T domega = omega_min/nt;

          // correct domega for extreme cases
          if (domega < 0.8 && nt > 1)
            domega = omega_min/(--nt);
          else if (nt == 1 && domega > 0.8 && dist2(it_prev->r, it_next->r) > 1.4*delta2)
            domega = omega_min/(++nt);
          else if (omega_min < 3 && (dist2(it_prev->r, it_min->r) < 0.25*delta2 || dist2(it_next->r, it_min->r) < 0.25*delta2))
            nt = 1;


          it_prev->omega_changed = true;
          it_next->omega_changed = true;

          if (nt > 1) {

            // projection of dr = p_next - p_min to tangent space
            //  c = dr.t1
            //  s = dr.t2

            T c = 0, s = 0, t;

            for (int i = 0; i < 3; ++i){
              t = it_prev->r[i] - it_min->r[i];   // = dr[i]
              c += t*it_min->b[0][i];             // = dr[i]*t1[i]
              s += t*it_min->b[1][i];             // = dr[i]*t2[i]
            }

            // returning fac*(sin(k domega), cos(k domega))
            // where fac = delta/|(c, s)|

            T sa[6], ca[6], u[3];

            utils::sincos_array(nt - 1, domega, sa, ca, delta/std::hypot(c, s));

            int n = V.size();             // size of the set of vertices

            T st, ct, qk[3];

            Tvertex Pi[6], *vp = Pi;      // new front from it_min
            
            for (int k = 1; k < nt && error == 0; ++k, ++n, ++vp){
              
              // rotate in tangent plane
              ct = c*ca[k] - s*sa[k];
              st = c*sa[k] + s*ca[k];

              // forming point on tangent plane
              for (int i = 0; i < 3; ++i)
                qk[i] = it_min->r[i] + (u[i] = it_min->b[0][i]*ct + it_min->b[1][i]*st);

              if (!project_onto_potential(qk, *vp, max_iter, it_min->b[2]) &&
                  !slide_over_potential(it_min->r, it_min->b[2], u, delta, *vp, max_iter)) {

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
            
                error = 2;
              }

              vp->index = n; // = V.size();
              vp->omega_changed = true;

              // V.emplace_back(vp->r, vp->b[2]);
              V.emplace_back(vp->r);                    // saving only r
              if (GatV) GatV->emplace_back(vp->norm);   // saving g
              NatV.emplace_back(vp->b[2]);              // saving only normal

              Tr.emplace_back((k == 1 ? it_prev->index : n - 1), n, it_min->index);
            }

            // n = V.size();
            Tr.emplace_back(n - 1, it_next->index, it_min->index);

            *(it_min++) = *Pi;

            P.insert(it_min, Pi + 1, Pi + nt - 1);

          } else {

            Tr.emplace_back(it_prev->index, it_next->index, it_min->index);

            P.erase(it_min);
          }
        }

        if (Tr.size() >= max_triangles) error = 1;
        
      } while (error == 0);
    
      
    } while (lP.size() > 0 && error == 0);
   
    return error;
  }

  /*
    Collect all bad points of the polygon front P

    P.size() > 3

  */

  Tbad_pair check_bad_pairs(Tfront_polygon &P, const T &delta2){

    int s;

    T a[3];

    auto
      it_begin = P.begin(),
      it_last = P.end() - 1,

      it0  = it_begin,
      it0_next = it0 + 1,
      it0_prev = it_last,
      it0_last = it_last - 2;

    while (1) {

      // checking only those it1 - it0 > 1
      auto
        it1 = (it0_next == it_last ? it0_next + 1 : it_begin),
        it1_next = (it1 == it_last ? it_begin : it1 + 1),
        it1_prev = (it1 == it_begin ? it_last : it1 - 1),

        // avoiding that the last element of it1 is the neighbour of it0
        it1_last = (it0 == it_begin ? it_last - 1 : it_last);

      while (1) {

        // are on the side the object
        if (utils::dot3D(it0->b[2], it1->b[2]) > 0) {

          utils::sub3D(it1->r, it0->r, a);

          // if near enough and looking inside from it and from it1
          if (utils::norm2(a) < delta2) {

            // check if same side of both edges and determine the side
            // depending of it_prev -> it -> it_next circle
            s = split_angle(*it0_prev, *it0, *it0_next, a);

            if (s != 0 &&
                s*split_angle(*it1_prev, *it1, *it1_next, a) < 0) {

              // create new last front
              #if defined(DEBUG)
              std::cerr
                << "P.size=" << P.size()
                << " i=" << int(it0 - it_begin)
                << " j=" << int(it1 - it_begin)
                << " len=" << int(it1 + 1 - it)
                << std::endl;
              #endif

              return Tbad_pair(it0 - it_begin, it1 - it_begin);
            }
          }
        }

        if (it1 == it1_last) break;
        it1_prev = it1;
        it1 = it1_next;

        if (it1_next == it_last)
          it1_next = it_begin;
        else
          ++it1_next;
      }

      if (it0 == it0_last) break;
      it0_prev = it0;
      it0 = it0_next++;
    }

    return Tbad_pair(0, 0);
  }


  /*
    Collect all bad points between polygon front P and part
    of the polygon
  */
  //#define DEBUG
  Tbad_pair check_bad_pairs(
    Tfront_polygon &P,
    typename Tfront_polygon::iterator & start,
    typename Tfront_polygon::iterator & end,
    const T &delta2 ){

    int s;

    T a[3];

    // checking all combinations of (it0, it1)
    auto
      it_begin = P.begin(),
      it_last = P.end() - 1,

      it0 = start,
      it0_next = (it0 == it_last  ? it_begin : it0 + 1),
      it0_prev = (it0 == it_begin ? it_last  : it0 - 1),
      it0_prev_start = it0_prev,
      it0_last = end - 1;

    while (1) {

      auto
        it1 = (it0_next == it_last ? it_begin : it0_next + 1),
        it1_next = (it1 == it_last  ? it_begin : it1 + 1),
        it1_prev = (it1 == it_begin ? it_last  : it1 - 1),

        // avoid running twice over the [start,end)
        it1_last = (
          it0 == start ?
          (it0_prev_start == it_begin ? it_last : it0_prev_start - 1):
          it0_prev_start
        );

      while (1) {

        // are on the side the object
        if (utils::dot3D(it0->b[2], it1->b[2]) > 0) {

          utils::sub3D(it1->r, it0->r, a);

          // if near enough and looking inside from it and from it1
          if (utils::norm2(a) < delta2) {

            // check if same side of both edges and determine the side
            // depending of it_prev -> it -> it_next circle
            s = split_angle(*it0_prev, *it0, *it0_next, a);

            if (s != 0 && s*split_angle(*it1_prev, *it1, *it1_next, a) < 0) {

              int ind[2] = {
                static_cast<int>(it0 - it_begin),
                static_cast<int>(it1 - it_begin)
              };

              // create new last front
              #if defined(DEBUG)
              std::cerr
                << "P.size=" << P.size()
                << " i=" << ind[0]
                << " j=" << ind[1]
                << " len=" << int(it1 + 1 - it0)
                << std::endl;
              #endif

              if (ind[0] < ind[1]) return Tbad_pair(ind[0], ind[1]);

              return Tbad_pair(ind[1], ind[0]);
            }
          }
        }

        if (it1 == it1_last) break;

        it1_prev = it1;
        it1 = it1_next;

        if (it1_next == it_last)
          it1_next = it_begin;
        else
          ++it1_next;
      }

      if (it0 == it0_last) break;

      it0_prev = it0;
      it0 = it0_next;

      if (it0_next == it_last)
        it0_next = it_begin;
      else
        ++it0_next;
    }

    return Tbad_pair(0, 0);
  }
  #if defined(DEBUG)
  #undef DEBUG
  #endif

  /*
    Triangulization using marching method of genus 0 closed and surfaces.

    Has:
      -- additionals checks
      -- supports multifronts
      -- clever detection of bad pairs/points

    Input:
      init_r[3] - initial position
      init_g[3] - initial gradient
      delta - size of triangles edges projected to tangent space
      max_triangles - maximal number of triangles used
      init_phi - rotation of the initial hexagon

    Output:
      V - vector of vertices
      NatV - vector of normals at vertices (read N at V)
      Tr - vector of triangles
      GatV - norm of the gradient at vertices
  
    Return:
     0 - no error
     1 - too triangles
     2 - problem with converges
  */
  
  int triangulize_full_clever(
    T init_r[3],
    T init_g[3],
    const T & delta,
    const unsigned & max_triangles,
    std::vector <T3Dpoint<T>> & V,
    std::vector <T3Dpoint<T>> & NatV,
    std::vector <T3Dpoint<int>> & Tr,
    std::vector<T> * GatV = 0,
    const T & init_phi = 0) 
  {

    // start with normal precision defined by T
    precision = false;
   
    // error 
    int error = 0;

    V.clear();
    Tr.clear();

    const int max_iter = 100;

    // list of front polygons: front is threated as circular list
    std::vector<Tfront_polygon> lP(1);

    // list of bad pairs
    //   pair.first = pair.second means there is no bad pair
    std::vector<Tbad_pair> lB;

    //
    // Create initial frontal polygon lP[0] and initial bad point lB[0]
    // Step 0:
    //

    {
      Tvertex v, vk;

      Tfront_polygon & P  = lP.back();

      lB.emplace_back(0,0);   // no bad pair detected

      // construct the vector base
      create_internal_vertex(init_r, init_g, v, init_phi);

      // add vertex to the set, index 0
      V.emplace_back(v.r);                  // saving only r
      if (GatV) GatV->emplace_back(v.norm); // saving g
      NatV.emplace_back(v.b[2]);            // saving only normal

      T sa[6], ca[6], qk[3], u[3];

      utils::sincos_array(5, utils::m_pi3, sa, ca, delta);
       
      for (int k = 0; k < 6 && error == 0; ++k){
        
        for (int i = 0; i < 3; ++i) 
          qk[i] = v.r[i] + (u[i] = ca[k]*v.b[0][i] + sa[k]*v.b[1][i]);

        if (
            !slide_over_potential(v.r, v.b[2], u, delta, vk, max_iter) &&
            !project_onto_potential(qk, vk, max_iter, v.b[2])
           ) {
          std::cerr << "Warning: Projection did not converge for initial frontal polygon!\n";
          error = 2;
        }

        // store points into initial front
        vk.index = k + 1;  // = V.size();
        vk.omega_changed = true;
        P.push_back(vk);

        V.emplace_back(vk.r);                     // saving only r
        if (GatV) GatV->emplace_back(vk.norm);    // saving norm
        NatV.emplace_back(vk.b[2]);               // saving only normal
      }

      //
      // Creating initial hexagon -- triangle faces in Tr
      //
      for (int k = 0; k < 5; ++k) Tr.emplace_back(0, k + 1, k + 2);
      Tr.emplace_back(0, 6, 1);
    }

    //
    //  Triangulization of genus 0 surfaces
    //
    
    T delta2 = 0.5*delta*delta;    // TODO: should be more dynamical

    do {

      // current front polygon
      Tfront_polygon & P  = lP.back();
      Tbad_pair & B = lB.back();

      do {

        //
        // Processing the last three vertices
        //
        if (P.size() == 3) {
          Tr.emplace_back(P[0].index, P[1].index, P[2].index);

          // erasing discussed front
          lP.pop_back();

          // erase discussed possible bad pair
          lB.pop_back();

          break;
        }

        // pointers associated to the front
        auto it_begin = P.begin(), it_end = P.end(), it_last = it_end - 1;

        //
        // If a non-neighboring vertices are to close form new fronts
        // Step 2
        //
        {
          // if bad pair is set do the cut of the front
          if (B.first != B.second) {

            // separate fronts P -> P, P1
            auto
              it0 = it_begin + B.first,
              it1 = it_begin + B.second;

            it0->omega_changed = true;
            it1->omega_changed = true;

            Tfront_polygon P1(it0, it1 + 1);
            P.erase(it0 + 1, it1);
            B = check_bad_pairs(P, delta2);

            lP.push_back(P1);
            lB.push_back(check_bad_pairs(P1, delta2));

            break;
          }
        }

        //
        // Calculate the front angles and choose the point with the smallest
        // Step 1
        //

        T omega_min = utils::m_2pi;

        typename Tfront_polygon::iterator it_min;

        {

          T omega, t, tt, c, s, st, ct;

          // set it_prev, it, it_next: as circular list
          auto it = it_begin, it_next = it + 1, it_prev = it_last;

          while (1) {

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
              if (omega < 0) omega += utils::m_2pi;

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

            if (it_next == it_begin) break;

            if (it_next == it_last)
              it_next = it_begin;
            else
              ++it_next;
          }
        }


        //
        // Discuss the point with the minimal angle
        // Step 3
        //

        {
          // prepare pointers to vertices in P
          auto
            it_prev = it_min,
            it_next = it_min;

          if (it_min != it_begin) --it_prev; else it_prev = it_last;
          if (it_min != it_last) ++it_next; else it_next = it_begin;

          // number of triangles to be generated
          int nt = int(omega_min/utils::m_pi3) + 1;

          T domega = omega_min/nt;

          // correct domega for extreme cases
          if (domega < 0.8 && nt > 1) {
            domega = omega_min/(--nt);
          } else if (nt == 1 && domega > 0.8 &&
                   dist2(it_prev->r, it_next->r) > 1.4*delta2) {
            domega = omega_min/(++nt);
          } else if (omega_min < 3 &&
                    ( dist2(it_prev->r, it_min->r) < 0.25*delta2 ||
                      dist2(it_next->r, it_min->r) < 0.25*delta2)
                  )  {
            nt = 1;
          }
          it_prev->omega_changed = true;
          it_next->omega_changed = true;

          if (nt > 1) {

            // projection of dr = p_next - p_min to tangent space
            //  c = dr.t1
            //  s = dr.t2

            T c = 0, s = 0, t;

            for (int i = 0; i < 3; ++i){
              t = it_prev->r[i] - it_min->r[i];   // = dr[i]
              c += t*it_min->b[0][i];             // = dr[i]*t1[i]
              s += t*it_min->b[1][i];             // = dr[i]*t2[i]
            }

            // returning fac*(sin(k domega), cos(k domega))
            // where fac = delta/|(c, s)|

            T sa[6], ca[6], u[3];

            utils::sincos_array(nt - 1, domega, sa, ca, delta/std::hypot(c, s));

            int n = V.size();             // size of the set of vertices

            T st, ct, qk[3];

            Tvertex Pi[6], *vp = Pi;      // new front from it_min
            
            for (int k = 1; k < nt && error == 0; ++k, ++n, ++vp){
              
              // rotate in tangent plane
              ct = c*ca[k] - s*sa[k];
              st = c*sa[k] + s*ca[k];

              // forming point on tangent plane
              for (int i = 0; i < 3; ++i)
                qk[i] = it_min->r[i] + (u[i] = it_min->b[0][i]*ct + it_min->b[1][i]*st);

              if (!project_onto_potential(qk, *vp, max_iter, it_min->b[2]) &&
                  !slide_over_potential(it_min->r, it_min->b[2], u, delta, *vp, max_iter)) {

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
                
                error = 2;
              }

              vp->index = n; // = V.size();
              vp->omega_changed = true;

              // V.emplace_back(vp->r, vp->b[2]);
              V.emplace_back(vp->r);                    // saving only r
              if (GatV) GatV->emplace_back(vp->norm);   // saving g
              NatV.emplace_back(vp->b[2]);              // saving only normal

              // add triangle
              Tr.emplace_back((k == 1 ? it_prev->index : n - 1), n, it_min->index);
            }

            // Note: n = V.size();

            // add triangle
            Tr.emplace_back(n - 1, it_next->index, it_min->index);

            // add vertices to front and replace minimal
            *(it_min++) = *Pi;

            auto
              it0 = P.insert(it_min, Pi + 1, Pi + nt - 1),

            // check if there are any bad pairs
              it1 = (--it0) + nt - 1;

            B = check_bad_pairs(P, it0, it1, delta2);

          } else {
            // add triangle
            Tr.emplace_back(it_prev->index, it_next->index, it_min->index);

            // erase vertex from the front
            P.erase(it_min);
          }
        }

        if (Tr.size() >= max_triangles) error = 1;
                
      } while (error == 0);

    } while (lP.size() > 0 && error == 0);

    return error;
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
    
    Return:
     true if ok, false if not ok
  */
  bool central_points(
    std::vector <T3Dpoint<T>> & V,
    std::vector <T3Dpoint<int>> & Tr,

    std::vector <T3Dpoint<T>> * C = 0,
    std::vector <T3Dpoint<T>> * NatC = 0,
    std::vector <T> * GatC = 0
  )
  {
    if (C == 0 && NatC == 0 && GatC == 0) return true;

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
    const T eps = 100*std::numeric_limits<T>::epsilon();
    
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

      if (project_onto_potential(q, v, n, max_iter, p_g, eps)){
        if (C) C->emplace_back(v);
        if (NatC) NatC->emplace_back(n);
        if (GatC) GatC->emplace_back(g);
      } else return false; //std::cerr << "central_points::Warning: Projection did not converge\n";
    }
    
    return true;
  }

}; // class marching

