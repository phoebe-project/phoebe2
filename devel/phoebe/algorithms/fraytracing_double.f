      subroutine raytracing(N,Ps,triangles,hidden)
Cf2py intent(in) N
Cf2py intent(in) Ps
Cf2py intent(in) triangles
Cf2py intent(in,out) hidden
C     Hidden(i) has value -1 if not hidden
C     Hidden(i) has value j if hidden by triangle j
C     ==========================================================================
C     This function only takes care of centers of triangles, not of the
C     triangles themselves. It is therefore most likely not very useful...
C     ==========================================================================
      implicit none
      
      integer i,j,N

      real Ps(N,2), triangles(N,9)
      real v0(2),v1(2),v2(2)
      real hidden(N)
      real dot00,dot01,dot02,dot11,dot12
      real u,v,w,invDenom

   99 do 10, i=2,N
        hidden(i) = -1.0
        do 11, j=1,i-1
C         -- compute vectors
          v0(1) = triangles(j,7)-triangles(j,1)
          v0(2) = triangles(j,8)-triangles(j,2)
          v1(1) = triangles(j,4)-triangles(j,1)
          v1(2) = triangles(j,5)-triangles(j,2)
          v2(1) = Ps(i,1)-triangles(j,1)
          v2(2) = Ps(i,2)-triangles(j,2)
C         -- compute dot products          
          dot00 = v0(1)*v0(1) + v0(2)*v0(2)
          dot01 = v0(1)*v1(1) + v0(2)*v1(2)
          dot02 = v0(1)*v2(1) + v0(2)*v2(2)
          dot11 = v1(1)*v1(1) + v1(2)*v1(2)
          dot12 = v1(1)*v2(1) + v1(2)*v2(2)
C         -- compute barycentric coordinates          
          invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
          u = (dot11 * dot02 - dot01 * dot12) * invDenom
          v = (dot00 * dot12 - dot01 * dot02) * invDenom
          w = u+v
C         -- check if point is in triangle          
          if ((u.gt.0).AND.(v.gt.0).AND.(w.LT.1)) then
            hidden(i) = j
            EXIT
          endif
   11   continue
   10 continue
      RETURN
      END











      subroutine fine(N,triangles,nors,thresh,tol,hidden,
     + hidden_partial,hidden_vertices,hiding_partial)
Cf2py intent(in) N
Cf2py intent(in) triangles
Cf2py intent(in) nors
Cf2py intent(in) thresh
Cf2py intent(in) tol
Cf2py intent(in,out) hidden
Cf2py intent(in,out) hidden_partial
Cf2py intent(in,out) hidden_vertices
Cf2py intent(in,out) hiding_partial

C     ==========================================================================
C     This subroutine runs over all triangles from fron to back (it is assumed
C     that they are pre-sorted). At each step, we check if any of the vertices
C     of the triangle under consideration lies inside a triangle which is in
C     front of it. If so, we consider the triangle partially hidden. If all
C     vertices are 'partially hidden', we consider the triangle to be totally
C     hidden. Else, we consider the triangle as totally visible.
C     ==========================================================================

      implicit none
      
      integer i,j,k,N
      logical v0h,v1h,v2h

      real*8 triangles(N,9)
      real*8 nors(N,3)
      real*8 v0(2),v1(2),v2(2)
      integer hidden(N)
      integer hidden_partial(N),hidden_vertices(N,3)
      integer hiding_partial(N)
      integer hiding_partial_vertices(N,3)
      real*8 dot00,dot01,dot02,dot11,dot12
      real*8 u,v,w,invDenom,alpha,thresh,one,zero,tol
      
c      one = 0.999999
      one = 1d0-tol
      zero = tol
      hidden(1) = -1
      hidden_vertices(1,1) = -1
      hidden_vertices(1,2) = -1
      hidden_vertices(1,3) = -1
      hiding_partial(1) = -1
      hiding_partial_vertices(1,1) = -1
      hiding_partial_vertices(1,2) = -1
      hiding_partial_vertices(1,3) = -1
      hidden_partial(1) = -1
      do 20, i=2,N
        hidden(i) = -1
        hidden_vertices(i,1) = -1
        hidden_vertices(i,2) = -1
        hidden_vertices(i,3) = -1
        hidden_partial(i) = -1
        hiding_partial(i) = -1
        hiding_partial_vertices(i,1) = -1
        hiding_partial_vertices(i,2) = -1
        hiding_partial_vertices(i,3) = -1
C       forget about angles really facing backwards (used to be thresh.LT.)
C        if (thresh.GT.3.1416d0) then
        if (thresh.GT.3.1416d0) then
        alpha = acos(-nors(i,3)/sqrt(nors(i,1)**2+nors(i,2)**2+
     +         nors(i,3)**2))
        if (alpha.GT.thresh) then
            hidden(i) = 1
            CYCLE
        endif
        endif
        do 21, j=1,i-1
C         If this triangle is completely hidden, don't bother checking
          if (hidden(j).gt.(-1)) CYCLE
          do 22, k=1,3
C           -- is any of the vertices of the last (considered) triangle
C           obscured by a triangle in front of it?
C           -- compute vectors: 
            v0(1) = triangles(j,7)-triangles(j,1)
            v0(2) = triangles(j,8)-triangles(j,2)
            v1(1) = triangles(j,4)-triangles(j,1)
            v1(2) = triangles(j,5)-triangles(j,2)
            v2(1) = triangles(i,3*k-2)-triangles(j,1)
            v2(2) = triangles(i,3*k-1)-triangles(j,2)
C           -- compute dot products          
            dot00 = v0(1)*v0(1) + v0(2)*v0(2)
            dot01 = v0(1)*v1(1) + v0(2)*v1(2)
            dot02 = v0(1)*v2(1) + v0(2)*v2(2)
            dot11 = v1(1)*v1(1) + v1(2)*v1(2)
            dot12 = v1(1)*v2(1) + v1(2)*v2(2)
C             -- compute barycentric coordinates          
            invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            w = u+v
C           -- check if point is in triangle. If it is, the index refers to
C           to triangle which is in front of it
            if ((u.gt.zero).AND.(v.gt.zero).AND.(w.LT.one)) then
              hidden_vertices(i,k) = j
            endif
C           -- NEXT
C           -- is any of the triangles in front of it obscuring this
C              triangle?
C           -- compute vectors: 
            v0(1) = triangles(i,7)-triangles(i,1)
            v0(2) = triangles(i,8)-triangles(i,2)
            v1(1) = triangles(i,4)-triangles(i,1)
            v1(2) = triangles(i,5)-triangles(i,2)
            v2(1) = triangles(j,3*k-2)-triangles(i,1)
            v2(2) = triangles(j,3*k-1)-triangles(i,2)
C           -- compute dot products          
            dot00 = v0(1)*v0(1) + v0(2)*v0(2)
            dot01 = v0(1)*v1(1) + v0(2)*v1(2)
            dot02 = v0(1)*v2(1) + v0(2)*v2(2)
            dot11 = v1(1)*v1(1) + v1(2)*v1(2)
            dot12 = v1(1)*v2(1) + v1(2)*v2(2)
C             -- compute barycentric coordinates          
            invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            w = u+v
C           -- check if point is in triangle. If it is, the index refers to
C           to triangle which is in front of it
            if ((u.gt.zero).AND.(v.gt.zero).AND.(w.LT.one)) then
              hidden_partial(i) = j
            endif
C         -- if this triangle i is completely hidden, remember it. We will not
C         treat this one again. No need to check if its partially obscured 
          if ((hidden_vertices(i,1).gt.-1)
     +       .AND.(hidden_vertices(i,2).gt.-1)
     +       .AND.(hidden_vertices(i,3).gt.-1)) then
            hidden(i) = hidden_vertices(i,1)
            EXIT
C         -- otherwise, it's possible that this triangle is partially hidden
          elseif (hidden_vertices(i,1).gt.-1) then
            hidden_partial(i) = hidden_vertices(i,1)
          elseif (hidden_vertices(i,2).gt.-1) then
            hidden_partial(i) = hidden_vertices(i,2)
          elseif (hidden_vertices(i,3).gt.-1) then
            hidden_partial(i) = hidden_vertices(i,3)
          endif
   22     continue
   21   continue
   20 continue
      RETURN
      END    
      PROGRAM dummy
      END