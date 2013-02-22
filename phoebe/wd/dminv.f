      SUBROUTINE DMINV(A,N,D,L,M)
c  Version of January 9, 2002
      DIMENSION A(*),L(*),M(*)
      DOUBLE PRECISION A,D,BIGA,HOLD
      D=1.D0
      NK=-N
      DO 80 K=1,N
      NK=NK+N
      L(K)=K
      M(K)=K
      KK=NK+K
      BIGA=A(KK)
      DO 20 J=K,N
      IZ=N*(J-1)
      DO 20 I=K,N
      IJ=IZ+I
      IF(DABS(BIGA).GE.DABS(A(IJ))) GOTO 20
      BIGA=A(IJ)
      L(K)=I
      M(K)=J
   20 CONTINUE
      J=L(K)
      IF(J.LE.K) GOTO 35
      KI=K-N
      DO 30 I=1,N
      KI=KI+N
      HOLD=-A(KI)
      JI=KI-K+J
      A(KI)=A(JI)
   30 A(JI) =HOLD
   35 I=M(K)
      IF(I.LE.K) GOTO 45
      JP=N*(I-1)
      DO 40 J=1,N
      JK=NK+J
      JI=JP+J
      HOLD=-A(JK)
      A(JK)=A(JI)
   40 A(JI) =HOLD
   45 IF(BIGA.NE.0.D0) GOTO 48
      D=0.D0
      RETURN
   48 DO 55 I=1,N
      IF(I.EQ.K) GOTO 55
      IK=NK+I
      A(IK)=A(IK)/(-BIGA)
   55 CONTINUE
      DO 65 I=1,N
      IK=NK+I
      HOLD=A(IK)
      IJ=I-N
      DO 65 J=1,N
      IJ=IJ+N
      IF(I.EQ.K) GOTO 65
      IF(J.EQ.K) GOTO 65
      KJ=IJ-I+K
      A(IJ)=HOLD*A(KJ)+A(IJ)
   65 CONTINUE
      KJ=K-N
      DO 75 J=1,N
      KJ=KJ+N
      IF(J.EQ.K) GOTO 75
      A(KJ)=A(KJ)/BIGA
   75 CONTINUE
      D=D*BIGA
      A(KK)=1.D0/BIGA
   80 CONTINUE
      K=N
  100 K=(K-1)
      IF(K.LE.0) RETURN
      I=L(K)
      IF(I.LE.K) GOTO 120
      JQ=N*(K-1)
      JR=N*(I-1)
      DO 110 J=1,N
      JK=JQ+J
      HOLD=A(JK)
      JI=JR+J
      A(JK)=-A(JI)
  110 A(JI) =HOLD
  120 J=M(K)
      IF(J.LE.K) GOTO 100
      KI=K-N
      DO 130 I=1,N
      KI=KI+N
      HOLD=A(KI)
      JI=KI-K+J
      A(KI)=-A(JI)
  130 A(JI) =HOLD
      GO TO 100
      END
