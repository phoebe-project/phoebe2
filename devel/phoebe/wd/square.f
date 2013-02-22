      SUBROUTINE square (OBS,NOBS,ML,OUT,sd,xlamda,D,CN,CNN,cnc,
     $clc,ss,cl,ll,mm)
c  Version of January 16, 2002
      implicit real*8 (a-h,o-z)
      dimension obs(*),out(*),sd(*),cn(*),cnn(*),cnc(*),clc(*),
     $cl(*),ll(*),mm(*)
c
c  cnc ("cn copy") is the original normal equation matrix
c  cn is the re-scaled version of cnc
c  cnn comes in as the original n.e. matrix, then is copied
c    from cn to become the re-scaled n.e.'s, and finally is
c    inverted by DMINV to become the inverse of the re-scaled
c    n.e. matrix.
c
      S=0.D0
      CLL=0.D0
      CAY=NOBS-ML
      JMAX=ML*ML
      DO 20 J=1,JMAX
   20 CN(J)=0.D0
      DO 21 J=1,ML
   21 CL(J)=0.D0
      DO 25 NOB=1,NOBS
      III=NOB+NOBS*ML
      OBSQQ=OBS(III)
      DO 23 K=1,ML
      DO 23 I=1,ML
      II=NOB+NOBS*(I-1)
      KK=NOB+NOBS*(K-1)
      J=I+(K-1)*ML
      OBSII=OBS(II)
      OBSKK=OBS(KK)
      CN(J)=CN(J)+OBSII*OBSKK
   23 cnc(j)=cn(j)
      DO 24 I=1,ML
      II=NOB+NOBS*(I-1)
      OBSII=OBS(II)
   24 CL(I)=CL(I)+OBSQQ*OBSII
   25 CLL=CLL+OBSQQ*OBSQQ
      do 123 k=1,ml
      do 123 i=1,ml
      xlf=0.d0
      if(i.eq.k) xlf=xlamda
      j=i+(k-1)*ml
      ji=i+(i-1)*ml
      ki=k+(k-1)*ml
  123 cn(j)=cn(j)/dsqrt(cnc(ji)*cnc(ki))+xlf
      do 124 i=1,ml
      ji=i+(i-1)*ml
      clc(i)=cl(i)
  124 cl(i)=cl(i)/dsqrt(cnc(ji))
      DO 50 J=1,JMAX
   50 CNN(J)=CN(J)
      CALL DMINV(CNN,ML,D,LL,MM)
      CALL DGMPRD(CNN,CL,OUT,ML,ML,1)
      do 125 i=1,ml
      ji=i+(i-1)*ml
  125 out(i)=out(i)/dsqrt(cnc(ji))
      DO 26 I=1,ML
   26 S=S+clc(I)*OUT(I)
      S=CLL-S
      SS=S
      SIGSQ=S/CAY
      DO 27 J=1,ML
      JJ=J*ML+J-ML
      CNJJ=CNN(JJ)
      ARG=SIGSQ*CNJJ
   27 sd(J)=dsqrt(arg/cnc(jj))
      RETURN
      END
