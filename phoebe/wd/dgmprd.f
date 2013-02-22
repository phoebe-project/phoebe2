      SUBROUTINE DGMPRD(A,B,R,N,M,L)
c  Version of April 9, 1992
      DIMENSION A(*),B(*),R(*)
      DOUBLE PRECISION A,B,R
      IR=0
      IK=-M
      DO 10 K=1,L
      IK=IK+M
      DO 10 J=1,N
      IR=IR+1
      JI=J-N
      IB=IK
      R(IR)=0.D0
      DO 10 I=1,M
      JI=JI+N
      IB=IB+1
   10 R(IR)=R(IR)+A(JI)*B(IB)
      RETURN
      END
