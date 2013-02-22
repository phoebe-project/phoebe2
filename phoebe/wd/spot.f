      SUBROUTINE SPOT(KOMP,N,SINTH,COSTH,SINFI,COSFI,TEMF)
C
c   If a surface point is in more than one spot, this subroutine
c      adopts the product of the spot temperature factors.
C
c   "Latitudes" here actually run from 0 at one pole to 180 deg.
c      at the other.
C
c   Version of February 11, 1998
C
      implicit real*8 (a-h,o-z)
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      COMMON /SPOTS/ SINLAT(2,100),COSLAT(2,100),SINLNG(2,100),COSLNG
     $(2,100),RAD(2,100),TEMSP(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      TEMF=1.d0
      nl=(2-komp)*nl1+(komp-1)*nl2
      DO 15 I=1,N
      do 42 j=1,nl
   42 if(kks(komp,j).eq.-i) Lspot(komp,j)=Lspot(komp,j)+1
      COSDFI=COSFI*COSLNG(KOMP,I)+SINFI*SINLNG(KOMP,I)
      S=dacos(COSTH*COSLAT(KOMP,I)+SINTH*SINLAT(KOMP,I)*COSDFI)
      IF(S.GT.RAD(KOMP,I)) GOTO 15
      TEMF=TEMF*TEMSP(KOMP,I)
      if(mpage.ne.3) goto 15
      do 24 j=1,nl
      kk=kks(komp,j)
      if(kk.eq.-i) Lspot(komp,j)=0
      if(kk.eq.i) Lspot(komp,j)=Lspot(komp,j)+1
   24 continue
   15 continue
      RETURN
      END
