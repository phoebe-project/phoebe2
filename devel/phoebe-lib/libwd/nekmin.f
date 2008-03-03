      SUBROUTINE NEKMIN(RM,OMEG,X,Z)
c  Version of October 9, 1995
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION DN(4),EN(2),OUT(2),LL(2),MM(2)
      Z=.05d0
   15 P1=X*X+Z*Z
      RP1=DSQRT(P1)
      P115=P1*RP1
      P2=(1.d0-X)**2+Z*Z
      RP2=DSQRT(P2)
      P215=P2*RP2
      DODZ=-Z/P115-RM*Z/P215
      OM=1.d0/RP1+RM/RP2+(1.d0+RM)*.5d0*X*X-RM*X
      DELOM=OMEG-OM
      DELZ=DELOM/DODZ
      Z=DABS(Z+DELZ)
      ABDELZ=DABS(DELZ)
      IF(ABDELZ.GT..00001d0) GOTO 15
   16 P1=X*X+Z*Z
      RP1=DSQRT(P1)
      P115=P1*RP1
      P125=P1*P115
      P2=(1.d0-X)**2+Z*Z
      RP2=DSQRT(P2)
      P215=P2*RP2
      P225=P2*P215
      DN(1)=-X/P115+RM*(1.d0-X)/P215+(1.d0+RM)*X-RM
      DN(2)=(3.d0*X*X-P1)/P125+(3.d0*RM*(1.d0-X)**2-RM*((1.d0-X)**2
     $+z*z))/p225+(RM+1.d0)
      DN(3)=-Z/P115-RM*Z/P215
      DN(4)=3.d0*X*Z/P125-3.d0*RM*Z*(1.d0-X)/P225
      OME=1.d0/RP1+RM/RP2+(1.d0+RM)*.5d0*X*X-RM*X
      EN(1)=OMEG-OME
      EN(2)=-DN(1)
      CALL DMINV(DN,2,D,LL,MM)
      CALL DGMPRD(DN,EN,OUT,2,2,1)
      DT1=OUT(1)
      DT2=OUT(2)
      ABDX=DABS(DT1)
      X=X+DT1
      ABDZ=DABS(DT2)
      Z=Z+DT2
      IF(ABDX.GT.1.d-8) GOTO 16
      IF(ABDZ.GT.1.d-8) GOTO 16
      RETURN
      END
