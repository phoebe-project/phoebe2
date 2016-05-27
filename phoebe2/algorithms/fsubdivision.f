      subroutine simple_subdivide(N,M,tri,s,nor,mu,thresh,
     +  newtri,news,newnor,newcen,newmu)
Cf2py intent(in) N
Cf2py intent(in) M
Cf2py intent(in) tri
Cf2py intent(in) s
Cf2py intent(in) nor
Cf2py intent(in) mu
Cf2py intent(in) thresh
Cf2py intent(in,out) newtri
Cf2py intent(in,out) news
Cf2py intent(in,out) newnor
Cf2py intent(in,out) newcen
Cf2py intent(in,out) newmu
C     thresh is meant to put a limiting size on the triangles
C     to subdivide (i.e. when already smaller than this threshold
C     skip it

      integer i,j,k,N,M
      real*8 tri(N,9),s(N),nor(N,3),cmp(N),mu(N)
      real*8 thresh
      real*8 newcen(M,3),newtri(M,9),newmu(M)
      real*8 news(M),newnor(M,3),newcmp(M)
C     -- define verticies
      real*8 C1(3),C2(3),C3(3)
      real*8 C4(3),C5(3),C6(3)
      real*8 dummyN(3),dummyC(3)
      real*8 a,b,c,h
      
      do 20, i=1,N
        if ((thresh.gt.0d0).AND.(s(i)*mu(i).LT.thresh)) then
           newtri(i,1) = tri(i,1)
           newtri(i,2) = tri(i,2)
           newtri(i,3) = tri(i,3)
           newtri(i,4) = tri(i,4)
           newtri(i,5) = tri(i,5)
           newtri(i,6) = tri(i,6)
           newtri(i,7) = tri(i,7)
           newtri(i,8) = tri(i,8)
           newtri(i,9) = tri(i,9)
           news(i) = s(i)
           newmu(i) = mu(i)
           newnor(i,1) = nor(i,1)
           newnor(i,2) = nor(i,2)
           newnor(i,3) = nor(i,3)
           newcen(i,1) = (newtri(i,1)+newtri(i,4)+newtri(i,7))/3.0
           newcen(i,2) = (newtri(i,2)+newtri(i,5)+newtri(i,8))/3.0
           newcen(i,3) = (newtri(i,3)+newtri(i,6)+newtri(i,9))/3.0
           CYCLE
        endif
C       #-- don't subidivide small triangles
C       #-- 3 original vertices
        C1(1) = tri(i,1)
        C1(2) = tri(i,2)
        C1(3) = tri(i,3)
        C2(1) = tri(i,4)
        C2(2) = tri(i,5)
        C2(3) = tri(i,6)
        C3(1) = tri(i,7)
        C3(2) = tri(i,8)
        C3(3) = tri(i,9)
C       #-- 3 new vertices
        C4(1) = (C1(1)+C2(1))/2.
        C4(2) = (C1(2)+C2(2))/2.
        C4(3) = (C1(3)+C2(3))/2.
        C5(1) = (C1(1)+C3(1))/2.
        C5(2) = (C1(2)+C3(2))/2.
        C5(3) = (C1(3)+C3(3))/2.
        C6(1) = (C2(1)+C3(1))/2.
        C6(2) = (C2(2)+C3(2))/2.
        C6(3) = (C2(3)+C3(3))/2.
C       #-- 4 new triangles
C       #   TRIANGLE 1
        newtri(i,1)    = C1(1)
        newtri(i,2)    = C1(2)
        newtri(i,3)    = C1(3)
        newtri(i,4)    = C4(1)
        newtri(i,5)    = C4(2)
        newtri(i,6)    = C4(3)
        newtri(i,7)    = C5(1)
        newtri(i,8)    = C5(2)
        newtri(i,9)    = C5(3)
C       #   TRIANGLE 2
        newtri(N+i,1)  = C4(1)
        newtri(N+i,2)  = C4(2)
        newtri(N+i,3)  = C4(3)
        newtri(N+i,4)  = C6(1)
        newtri(N+i,5)  = C6(2)
        newtri(N+i,6)  = C6(3)
        newtri(N+i,7)  = C5(1)
        newtri(N+i,8)  = C5(2)
        newtri(N+i,9)  = C5(3)
C       #   TRIANGLE 3
        newtri(2*N+i,1)= C6(1)
        newtri(2*N+i,2)= C6(2)
        newtri(2*N+i,3)= C6(3)
        newtri(2*N+i,4)= C4(1)
        newtri(2*N+i,5)= C4(2)
        newtri(2*N+i,6)= C4(3)
        newtri(2*N+i,7)= C2(1)
        newtri(2*N+i,8)= C2(2)
        newtri(2*N+i,9)= C2(3)
C       #   TRIANGLE 4
        newtri(3*N+i,1)= C6(1)
        newtri(3*N+i,2)= C6(2)
        newtri(3*N+i,3)= C6(3)
        newtri(3*N+i,4)= C3(1)
        newtri(3*N+i,5)= C3(2)
        newtri(3*N+i,6)= C3(3)
        newtri(3*N+i,7)= C5(1)
        newtri(3*N+i,8)= C5(2)
        newtri(3*N+i,9)= C5(3)
C       #-- compute new sizes
        a = sqrt((C1(1)-C4(1))**2+(C1(2)-C4(2))**2+(C1(3)-C4(3))**2)
        b = sqrt((C5(1)-C4(1))**2+(C5(2)-C4(2))**2+(C5(3)-C4(3))**2)
        c = sqrt((C1(1)-C5(1))**2+(C1(2)-C5(2))**2+(C1(3)-C5(3))**2)
        h = (a+b+c)/2d0
        news(i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C6(1)-C4(1))**2+(C6(2)-C4(2))**2+(C6(3)-C4(3))**2)
        b = sqrt((C5(1)-C4(1))**2+(C5(2)-C4(2))**2+(C5(3)-C4(3))**2)
        c = sqrt((C6(1)-C5(1))**2+(C6(2)-C5(2))**2+(C6(3)-C5(3))**2)
        h = (a+b+c)/2d0
        news(N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C6(1)-C4(1))**2+(C6(2)-C4(2))**2+(C6(3)-C4(3))**2)
        b = sqrt((C2(1)-C4(1))**2+(C2(2)-C4(2))**2+(C2(3)-C4(3))**2)
        c = sqrt((C6(1)-C2(1))**2+(C6(2)-C2(2))**2+(C6(3)-C2(3))**2)
        h = (a+b+c)/2d0
        news(2*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C6(1)-C3(1))**2+(C6(2)-C3(2))**2+(C6(3)-C3(3))**2)
        b = sqrt((C5(1)-C3(1))**2+(C5(2)-C3(2))**2+(C5(3)-C3(3))**2)
        c = sqrt((C6(1)-C5(1))**2+(C6(2)-C5(2))**2+(C6(3)-C5(3))**2)
        h = (a+b+c)/2d0
        news(3*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
C       #-- new centers
        newcen(i,1) = (C1(1)+C4(1)+C5(1))/3d0
        newcen(i,2) = (C1(2)+C4(2)+C5(2))/3d0
        newcen(i,3) = (C1(3)+C4(3)+C5(3))/3d0
        newcen(N+i,1) = (C6(1)+C4(1)+C5(1))/3d0
        newcen(N+i,2) = (C6(2)+C4(2)+C5(2))/3d0
        newcen(N+i,3) = (C6(3)+C4(3)+C5(3))/3d0
        newcen(2*N+i,1) = (C6(1)+C4(1)+C2(1))/3d0
        newcen(2*N+i,2) = (C6(2)+C4(2)+C2(2))/3d0
        newcen(2*N+i,3) = (C6(3)+C4(3)+C2(3))/3d0
        newcen(3*N+i,1) = (C6(1)+C3(1)+C5(1))/3d0
        newcen(3*N+i,2) = (C6(2)+C3(2)+C5(2))/3d0
        newcen(3*N+i,3) = (C6(3)+C3(3)+C5(3))/3d0
C       #-- new normals
        newnor(i,1) = nor(i,1)
        newnor(i,2) = nor(i,2)
        newnor(i,3) = nor(i,3)
        newnor(N+i,1) = nor(i,1)
        newnor(N+i,2) = nor(i,2)
        newnor(N+i,3) = nor(i,3)
        newnor(2*N+i,1) = nor(i,1)
        newnor(2*N+i,2) = nor(i,2)
        newnor(2*N+i,3) = nor(i,3)
        newnor(3*N+i,1) = nor(i,1)
        newnor(3*N+i,2) = nor(i,2)
        newnor(3*N+i,3) = nor(i,3)
C       #-- new mus
        newmu(i) = mu(i)
        newmu(N+i) = mu(i)
        newmu(2*N+i) = mu(i)
        newmu(3*N+i) = mu(i)
   20 continue
      RETURN
      END
































      subroutine simple_subdivide_middle(N,M,tri,s,nor,mu,thresh,
     +  newtri,news,newnor,newcen,newmu)
Cf2py intent(in) N
Cf2py intent(in) M
Cf2py intent(in) tri
Cf2py intent(in) s
Cf2py intent(in) nor
Cf2py intent(in) mu
Cf2py intent(in) thresh
Cf2py intent(in,out) newtri
Cf2py intent(in,out) news
Cf2py intent(in,out) newnor
Cf2py intent(in,out) newcen
Cf2py intent(in,out) newmu
C     thresh is meant to put a limiting size on the triangles
C     to subdivide (i.e. when already smaller than this threshold
C     skip it

      integer i,j,k,N,M
      real*8 tri(N,9),s(N),nor(N,3),cmp(N),mu(N)
      real*8 thresh
      real*8 newcen(M,3),newtri(M,9),newmu(M)
      real*8 news(M),newnor(M,3),newcmp(M)
C     -- define verticies
      real*8 C1(3),C2(3),C3(3),C4(3)
      real*8 dummyN(3),dummyC(3)
      real*8 a,b,c,h
      
      do 20, i=1,N
        if ((thresh.gt.0d0).AND.(s(i)*mu(i).LT.thresh)) then
           newtri(i,1) = tri(i,1)
           newtri(i,2) = tri(i,2)
           newtri(i,3) = tri(i,3)
           newtri(i,4) = tri(i,4)
           newtri(i,5) = tri(i,5)
           newtri(i,6) = tri(i,6)
           newtri(i,7) = tri(i,7)
           newtri(i,8) = tri(i,8)
           newtri(i,9) = tri(i,9)
           news(i) = s(i)
           newmu(i) = mu(i)
           newnor(i,1) = nor(i,1)
           newnor(i,2) = nor(i,2)
           newnor(i,3) = nor(i,3)
           newcen(i,1) = (newtri(i,1)+newtri(i,4)+newtri(i,7))/3.0
           newcen(i,2) = (newtri(i,2)+newtri(i,5)+newtri(i,8))/3.0
           newcen(i,3) = (newtri(i,3)+newtri(i,6)+newtri(i,9))/3.0
           CYCLE
        endif
C       #-- don't subidivide small triangles
C       #-- 3 original vertices
        C1(1) = tri(i,1)
        C1(2) = tri(i,2)
        C1(3) = tri(i,3)
        C2(1) = tri(i,4)
        C2(2) = tri(i,5)
        C2(3) = tri(i,6)
        C3(1) = tri(i,7)
        C3(2) = tri(i,8)
        C3(3) = tri(i,9)
C       #-- 1 new vertex
        C4(1) = (C1(1)+C2(1)+C3(1))/3d0
        C4(2) = (C1(2)+C2(2)+C3(2))/3d0
        C4(3) = (C1(3)+C2(3)+C3(3))/3d0
C       #-- 3 new triangles
C       #   TRIANGLE 1
        newtri(i,1)    = C1(1)
        newtri(i,2)    = C1(2)
        newtri(i,3)    = C1(3)
        newtri(i,4)    = C2(1)
        newtri(i,5)    = C2(2)
        newtri(i,6)    = C2(3)
        newtri(i,7)    = C4(1)
        newtri(i,8)    = C4(2)
        newtri(i,9)    = C4(3)
C       #   TRIANGLE 2
        newtri(N+i,1)  = C2(1)
        newtri(N+i,2)  = C2(2)
        newtri(N+i,3)  = C2(3)
        newtri(N+i,4)  = C3(1)
        newtri(N+i,5)  = C3(2)
        newtri(N+i,6)  = C3(3)
        newtri(N+i,7)  = C4(1)
        newtri(N+i,8)  = C4(2)
        newtri(N+i,9)  = C4(3)
C       #   TRIANGLE 3
        newtri(2*N+i,1)= C3(1)
        newtri(2*N+i,2)= C3(2)
        newtri(2*N+i,3)= C3(3)
        newtri(2*N+i,4)= C1(1)
        newtri(2*N+i,5)= C1(2)
        newtri(2*N+i,6)= C1(3)
        newtri(2*N+i,7)= C4(1)
        newtri(2*N+i,8)= C4(2)
        newtri(2*N+i,9)= C4(3)
C       #-- compute new sizes
        a = sqrt((C1(1)-C4(1))**2+(C1(2)-C4(2))**2+(C1(3)-C4(3))**2)
        b = sqrt((C2(1)-C4(1))**2+(C2(2)-C4(2))**2+(C2(3)-C4(3))**2)
        c = sqrt((C1(1)-C2(1))**2+(C1(2)-C2(2))**2+(C1(3)-C2(3))**2)
        h = (a+b+c)/2d0
        news(i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C3(1)-C4(1))**2+(C3(2)-C4(2))**2+(C3(3)-C4(3))**2)
        b = sqrt((C2(1)-C4(1))**2+(C2(2)-C4(2))**2+(C2(3)-C4(3))**2)
        c = sqrt((C3(1)-C2(1))**2+(C3(2)-C2(2))**2+(C3(3)-C2(3))**2)
        h = (a+b+c)/2d0
        news(N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C3(1)-C4(1))**2+(C3(2)-C4(2))**2+(C3(3)-C4(3))**2)
        b = sqrt((C1(1)-C4(1))**2+(C1(2)-C4(2))**2+(C1(3)-C4(3))**2)
        c = sqrt((C3(1)-C1(1))**2+(C3(2)-C1(2))**2+(C3(3)-C1(3))**2)
        h = (a+b+c)/2d0
        news(2*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
C       #-- new centers
        newcen(i,1) = (C1(1)+C4(1)+C2(1))/3d0
        newcen(i,2) = (C1(2)+C4(2)+C2(2))/3d0
        newcen(i,3) = (C1(3)+C4(3)+C2(3))/3d0
        newcen(N+i,1) = (C2(1)+C4(1)+C3(1))/3d0
        newcen(N+i,2) = (C2(2)+C4(2)+C3(2))/3d0
        newcen(N+i,3) = (C2(3)+C4(3)+C3(3))/3d0
        newcen(2*N+i,1) = (C3(1)+C4(1)+C1(1))/3d0
        newcen(2*N+i,2) = (C3(2)+C4(2)+C1(2))/3d0
        newcen(2*N+i,3) = (C3(3)+C4(3)+C1(3))/3d0
C       #-- new normals
        newnor(i,1) = nor(i,1)
        newnor(i,2) = nor(i,2)
        newnor(i,3) = nor(i,3)
        newnor(N+i,1) = nor(i,1)
        newnor(N+i,2) = nor(i,2)
        newnor(N+i,3) = nor(i,3)
        newnor(2*N+i,1) = nor(i,1)
        newnor(2*N+i,2) = nor(i,2)
        newnor(2*N+i,3) = nor(i,3)
C       #-- new mus
        newmu(i) = mu(i)
        newmu(N+i) = mu(i)
        newmu(2*N+i) = mu(i)
   20 continue
      RETURN
      END

      subroutine subdivide(N,M,tri,s,nor,Phi,q,d,F,maxit,tol,
     +  thresh,
     +  newtri,news,newnor,newcen)
Cf2py intent(in) N
Cf2py intent(in) M
Cf2py intent(in) tri
Cf2py intent(in) s
Cf2py intent(in) nor
Cf2py intent(in) q
Cf2py intent(in) d
Cf2py intent(in) F
Cf2py intent(in) thresh
Cf2py intent(in,out) newtri
Cf2py intent(in,out) news
Cf2py intent(in,out) newnor
Cf2py intent(in,out) newcen
C     thresh is meant to put a limiting size on the triangles
C     to subdivide (i.e. when already smaller than this threshold
C     skip it

      integer i,j,k,N,M
      real tri(N,9),s(N),nor(N,3),cmp(N)
      double precision thresh
      real newcen(M,3),newtri(M,9)
      real news(M),newnor(M,3),newcmp(M)
      integer maxiter
      real tol
C     -- define verticies
      real C1(3),C2(3),C3(3)
      real C4(3),C5(3),C6(3)
      real dummyN(3),dummyC(3)
      real a,b,c,h
      real Phi,q,d,F
      
      do 20, i=1,N
C        if (s(i).LT.thresh) CYCLE
C       #-- don't subidivide small triangles
C       #-- 3 original vertices
        C1(1) = tri(i,1)
        C1(2) = tri(i,2)
        C1(3) = tri(i,3)
        C2(1) = tri(i,4)
        C2(2) = tri(i,5)
        C2(3) = tri(i,6)
        C3(1) = tri(i,7)
        C3(2) = tri(i,8)
        C3(3) = tri(i,9)
C       #-- 3 new vertices
        C4(1) = (C1(1)+C2(1))/2.
        C4(2) = (C1(2)+C2(2))/2.
        C4(3) = (C1(3)+C2(3))/2.
        C5(1) = (C1(1)+C3(1))/2.
        C5(2) = (C1(2)+C3(2))/2.
        C5(3) = (C1(3)+C3(3))/2.
        C6(1) = (C2(1)+C3(1))/2.
        C6(2) = (C2(2)+C3(2))/2.
        C6(3) = (C2(3)+C3(3))/2.
C       #   update values of new vertices
C        call find_surface(C4,maxit,tol,Phi,q,d,F,dummyN)
C        call find_surface(C5,maxit,tol,Phi,q,d,F,dummyN)
C        call find_surface(C6,maxit,tol,Phi,q,d,F,dummyN)
C       #-- 4 new triangles
C       #   TRIANGLE 1
        newtri(i,1)    = C1(1)
        newtri(i,2)    = C1(2)
        newtri(i,3)    = C1(3)
        newtri(i,4)    = C4(1)
        newtri(i,5)    = C4(2)
        newtri(i,6)    = C4(3)
        newtri(i,7)    = C5(1)
        newtri(i,8)    = C5(2)
        newtri(i,9)    = C5(3)
C       #   TRIANGLE 2
        newtri(N+i,1)  = C6(1)
        newtri(N+i,2)  = C6(2)
        newtri(N+i,3)  = C6(3)
        newtri(N+i,4)  = C4(1)
        newtri(N+i,5)  = C4(2)
        newtri(N+i,6)  = C4(3)
        newtri(N+i,7)  = C5(1)
        newtri(N+i,8)  = C5(2)
        newtri(N+i,9)  = C5(3)
C       #   TRIANGLE 3
        newtri(2*N+i,1)= C6(1)
        newtri(2*N+i,2)= C6(2)
        newtri(2*N+i,3)= C6(3)
        newtri(2*N+i,4)= C4(1)
        newtri(2*N+i,5)= C4(2)
        newtri(2*N+i,6)= C4(3)
        newtri(2*N+i,7)= C2(1)
        newtri(2*N+i,8)= C2(2)
        newtri(2*N+i,9)= C2(3)
C       #   TRIANGLE 4
        newtri(3*N+i,1)= C6(1)
        newtri(3*N+i,2)= C6(2)
        newtri(3*N+i,3)= C6(3)
        newtri(3*N+i,4)= C3(1)
        newtri(3*N+i,5)= C3(2)
        newtri(3*N+i,6)= C3(3)
        newtri(3*N+i,7)= C5(1)
        newtri(3*N+i,8)= C5(2)
        newtri(3*N+i,9)= C5(3)
C       #-- compute new sizes
        a = sqrt((C1(1)-C4(1))**2+(C1(2)-C4(2))**2+(C1(3)-C4(3))**2)
        b = sqrt((C5(1)-C4(1))**2+(C5(2)-C4(2))**2+(C5(3)-C4(3))**2)
        c = sqrt((C1(1)-C5(1))**2+(C1(2)-C5(2))**2+(C1(3)-C5(3))**2)
        h = (a+b+c)/2.0
        news(i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C6(1)-C4(1))**2+(C6(2)-C4(2))**2+(C6(3)-C4(3))**2)
        b = sqrt((C5(1)-C4(1))**2+(C5(2)-C4(2))**2+(C5(3)-C4(3))**2)
        c = sqrt((C6(1)-C5(1))**2+(C6(2)-C5(2))**2+(C6(3)-C5(3))**2)
        h = (a+b+c)/2.0
        news(N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C6(1)-C4(1))**2+(C6(2)-C4(2))**2+(C6(3)-C4(3))**2)
        b = sqrt((C2(1)-C4(1))**2+(C2(2)-C4(2))**2+(C2(3)-C4(3))**2)
        c = sqrt((C6(1)-C2(1))**2+(C6(2)-C2(2))**2+(C6(3)-C2(3))**2)
        h = (a+b+c)/2.0
        news(2*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
        a = sqrt((C6(1)-C3(1))**2+(C6(2)-C3(2))**2+(C6(3)-C3(3))**2)
        b = sqrt((C5(1)-C3(1))**2+(C5(2)-C3(2))**2+(C5(3)-C3(3))**2)
        c = sqrt((C6(1)-C5(1))**2+(C6(2)-C5(2))**2+(C6(3)-C5(3))**2)
        h = (a+b+c)/2.0
        news(3*N+i) = sqrt(h*(h-a)*(h-b)*(h-c))
C       #-- new centers
        newcen(i,1) = (C1(1)+C4(1)+C5(1))/3.0
        newcen(i,2) = (C1(2)+C4(2)+C5(2))/3.0
        newcen(i,3) = (C1(3)+C4(3)+C5(3))/3.0
        newcen(N+i,1) = (C6(1)+C4(1)+C5(1))/3.0
        newcen(N+i,2) = (C6(2)+C4(2)+C5(2))/3.0
        newcen(N+i,3) = (C6(3)+C4(3)+C5(3))/3.0
        newcen(2*N+i,1) = (C6(1)+C4(1)+C2(1))/3.0
        newcen(2*N+i,2) = (C6(2)+C4(2)+C2(2))/3.0
        newcen(2*N+i,3) = (C6(3)+C4(3)+C2(3))/3.0
        newcen(3*N+i,1) = (C6(1)+C3(1)+C5(1))/3.0
        newcen(3*N+i,2) = (C6(2)+C3(2)+C5(2))/3.0
        newcen(3*N+i,3) = (C6(3)+C3(3)+C5(3))/3.0
C       #-- update centers and normals
c        do 21, k=1,4
c          do 22, j=1,3
c            dummyC(j) = newcen((k-1)*N+i,j)
c            dummyN(j) = nor(i,j)
c   22     continue
c          call find_surface(dummyC,maxit,tol,Phi,q,d,F,dummyN)
c          do 23, j=1,3
c            newcen((k-1)*N+i,j) = dummyC(j)
c            newnor((k-1)*N+i,j) = nor(i,j)
c   23     continue
c   21   continue     
   20 continue
      RETURN
      END


      subroutine find_surface(coordin,maxit,tol,Phi,q,d,F,
     +  rgrad)
Cf2py intent(in,out) coordin
Cf2py intent(in) maxit
Cf2py intent(in) tol
Cf2py intent(in) Phi
Cf2py intent(in) q
Cf2py intent(in) d
Cf2py intent(in) F
Cf2py intent(in,out) rgrad
C     Given some coordinate 'coordin', project it down on to
C     the Roche surface, and also return the gradient
      real coordin(3),tol
      integer maxit,i,j
      real rold(3),rgrad(3),deltaf(3)
      real f_,sdeltaf,Phi,q,d,F,check
      
C     -- start iteration loop
      do 20, i=1,maxit
        do 21, j=1,3
          rold(j) = coordin(j)
   21   continue
C       -- Calculate Roche potential and roche potential gradient
        call roche(rold(1),rold(2),rold(3),Phi,q,d,F,f_)
        call roche_grad(rold(1),rold(2),rold(3),q,d,F,deltaf)
C       -- calculate sum(deltaf**2)
        sdeltaf = 0.
        do 22, j=1,3
          sdeltaf = sdeltaf + deltaf(j)**2
   22   continue
C       -- calculate new coordinates
        check = 0.
        do 23, j=1,3
          coordin(j) = rold(j) - f_/sdeltaf*deltaf(j)
          check = check + (rold(j)-coordin(j))**2
   23   continue
        if (check.LT.tol) EXIT
        if (tol.GE.maxit) write(*,*) 'maximum iterations succeeded'
   20 continue
      call roche_grad(coordin(1),coordin(2),coordin(3),q,d,F,rgrad)
      RETURN
      END
    
      subroutine roche(x,y,z,Phi,q,d,F,value)
Cf2py intent(in) x
Cf2py intent(in) y
Cf2py intent(in) z
Cf2py intent(in) Phi
Cf2py intent(in) q
Cf2py intent(in) d
Cf2py intent(in) F
Cf2py intent(in,out) value
      real x,y,z,Phi,q,d,F,value
      real r,lam,nu,term1,term2,term3
      r = sqrt(x**2+y**2+z**2)
      lam = x/r
      nu = z/r
      term1 = 1.0/r
      term2 = q * (1.0/sqrt(d**2 - 2*lam*d*r + r**2) - lam*r/d**2)
      term3 = 0.5 * F**2 * (q+1.0) * r**2 * (1.0-nu**2)
      value = Phi - (term1+term2+term3)
      RETURN
      END

      subroutine roche_grad(x,y,z,q,d,F,Omega)
Cf2py intent(in) x
Cf2py intent(in) y
Cf2py intent(in) z
Cf2py intent(in) q
Cf2py intent(in) d
Cf2py intent(in) F
Cf2py intent(in,out) Omega
      real x,y,z,q,d,F,Omega(3)
      real r,r_
      r = sqrt(x**2+y**2+z**2)
      r_= sqrt((d-x)**2+y**2+z**2)
      Omega(1) = +x/r**3 - q*(d-x)/r_**3 - F**2*(1.0+q)*x + q/d**2
      Omega(2) = +y/r**3 + q*y    /r_**3 - F**2*(1.0+q)*y
      Omega(3) = +z/r**3 + q*z    /r_**3
      RETURN
      END
      PROGRAM dummy
      END