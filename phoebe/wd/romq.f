      SUBROUTINE ROMQ(omein,Q,F,D,EC,TH,FI,R,DRDO,DRDQ,DODQ,KOMP,MODE)
c  Version of December 5, 2003
      implicit real*8 (a-h,o-z)
      theq=1.570796326794897d0
      MOD46=(MODE-5)**2
      MOD56=(2*MODE-11)**2
      modkom=mode*(komp+komp-3)
      ome=omein
      DQ=1.d-4*Q
      QP=Q+DQ
      TOL=5.d-8
C     TH, FI SHOULD BE IN RADIANS.
      sinth=dsin(th)
      XNUSQ=sinth*sinth
      XLAM=sinth*dcos(FI)
      RMA=Q
      QF=1.d0
      DP=1.d0-EC
      QFM=1.d0
      IF(KOMP.NE.2) GOTO 23
      RMA=1.d0/Q
      QF=1.d0/Q
      QFM=-1.d0/Q**2
   23 CONTINUE
      CALL ELLONE(F,DP,RMA,X,OMEG,XLD,OMD)
      OM2SAV=OMEG
      RMAP=QP
      IF(KOMP.NE.2) GOTO 92
      OMEG=OMEG*Q+(1.d0-Q)*.5d0
      IF(MOD56.EQ.1) OME=OMEG
      RMAP=1.d0/QP
      GOTO 93
   92 CONTINUE
      IF(MOD46.EQ.1) OME=OMEG
   93 CONTINUE
      POT=OME
      IF(KOMP.EQ.2) POT=OME/Q+.5d0*(Q-1.d0)/Q
      CALL ELLONE(F,DP,RMAP,XP,OMP,XLD,OMD)
      DODQ=(OMP-OM2SAV)/DQ
      RM1=RMA+1.d0
      DS=D*D
      RF=F*F
      R=1.d0/POT
      KOUNT=0
      DELR=0.d0
      IF(FI.NE.0.d0) GOTO 85
      IF(TH.NE.THEQ) GOTO 85
      IF(MODE.EQ.6) GOTO 114
      IF(MODE.NE.4) GOTO 80
      IF(KOMP.EQ.1) GOTO 114
      GOTO 85
   80 IF(MODE.NE.5) GOTO 85
      IF(KOMP.EQ.2) GOTO 114
   85 CONTINUE
   14 R=R+DELR
      KOUNT=KOUNT+1
      IF(KOUNT.LT.20) GOTO 70
  217 if(mode.eq.6) goto 114
      if(modkom.eq.-4) goto 114
      if(modkom.eq.5) goto 114
      DOMR=-1.d15
      R=-1.d0
      GOTO 116
   70 RSQ=R*R
      PAR=DS-2.d0*XLAM*R*D+RSQ
      RPAR=dsqrt(PAR)
      OM=1.d0/R+RMA*(1.d0/RPAR-XLAM*R/DS)+RM1*.5d0*RSQ*XNUSQ*RF
      DOMR=1.d0/(RF*RM1*XNUSQ*R-1.d0/RSQ-(RMA*(R-XLAM*D))/(PAR*RPAR)-
     $RMA*XLAM/DS)
      DELR=(POT-OM)*DOMR
      ABDELR=dabs(DELR)
      IF(ABDELR.GT.TOL) GOTO 14
      DOMRSV=DOMR
      IF(R.GE.1.d0) GOTO 217
      IF(FI.NE.0.d0) GO TO 116
      IF(TH.NE.THEQ)GO TO 116
      IF(OME-OMEG) 217,114,116
  114 DOMR=1.d15
      R=X
      goto 118
  116 DRDQ=(1.d0/RPAR-R*XLAM/DS+.5d0*RF*RSQ*XNUSQ)/(1.d0/RSQ+RMA*
     $((1.d0/(PAR*RPAR))*(R-XLAM*D)+XLAM/DS)-RF*XNUSQ*RM1*R)
      DRDQ=DRDQ*QFM
  118 drdo=domr*qf
      IF(MODE.EQ.6) GOTO 215
      IF(MODE.NE.4) GOTO 180
      IF(KOMP.EQ.1) GOTO 215
      RETURN
  180 IF(MODE.NE.5) RETURN
      IF(KOMP.EQ.2) GOTO 215
      RETURN
  215 IF(FI.NE.0.d0) GOTO 230
      IF(TH.NE.THEQ) GOTO 230
      DRDQ=(XP-X)/DQ
      RETURN
  230 DRDQ=DRDQ+DOMRSV*DODQ
      RETURN
      END
