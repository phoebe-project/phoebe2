      SUBROUTINE ELLONE(FF,dd,rm,xl1,OM1,XL2,OM2)
c  Version of December 4, 2003
C     XL2 AND OM2 VALUES ASSUME SYNCHRONOUS ROTATION AND CIRCULAR ORBIT.
C     THEY ARE NOT NEEDED FOR NON-SYNCHRONOUS OR NON-CIRCULAR CASES.
c
c  Starting on August 13, 2003, ELLONE includes a 2nd derivative term
c    in the N-R solution for the null point of effective gravity (XL1)
c
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON /ECCEN/ecc,smaxis,period,vgadum,sindum,vfdum,vfadum,vgmdum,
     $v1dum,v2dum,ifcdum
      ot=1.d0/3.d0
      icase=2
      if(ff.ne.1.d0.or.ecc.gt.0.d0) icase=1
      rmass=rm
      d=dd
      xl=d/(1.d0+dsqrt(rm))
      oldxl=xl
      DO 5 I=1,icase
      RFAC=ff*ff
      IF(I.EQ.2) RFAC=1.D0
      IF(I.EQ.2) D=1.D0
      DSQ=D*D
      DELXL=0.D0
      RM1=RMASS+1.D0
      kount=0
   88 XL=XL+DELXL
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  The next block of lines halves the delxl step in case
c  xl were to jump beyond the value d or below value
c  0.0 during the iteration.
      if(i.eq.2) goto 170
      if(xl.lt.dd.and.xl.gt.0.d0) goto 170
      delxl=0.5d0*delxl
      xl=oldxl
      goto 88
  170 continue
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      kount=kount+1
      XSQ=XL*XL
      P=(D-XL)**2
      RP=DABS(D-XL)
      PRP=P*RP
      F=RFAC*RM1*XL-1.D0/XSQ-RMASS*(XL-D)/PRP-RMASS/DSQ
      DXLDF=1.D0/(RFAC*RM1+2.D0/(XSQ*XL)+2.D0*RMASS/PRP)
      d2xldf2=6.d0*dxldf**3*(1.d0/xl**4-rmass/rp**4)
      DELXL=-F*DXLDF+.5d0*f*f*d2xldf2
      ABDEL=DABS(DELXL)
      oldxl=xl
      IF(ABDEL.GT.1.d-10) GOTO 88
      IF(I.EQ.2) GOTO 8
      xl1=xl
      OM1=1.D0/XL+RMASS*((1.D0/RP)-XL/DSQ)+RM1*.5D0*XSQ*RFAC
      IF(rm.GT.1.d0) RMASS=1.D0/RMASS
      XMU3=RMASS/(3.D0*(RMASS+1.D0))
      XMU3CR=XMU3**ot
    5 XL=1.D0+XMU3CR+XMU3CR*XMU3CR/3.D0+XMU3/9.D0
    8 IF(rm.GT.1.d0) XL=D-XL
      rm1=rm+1.d0
      XL2=XL
      OM2=1.D0/DABS(XL)+rm*((1.D0/DSQRT(1.D0-XL-XL+XL*XL))-XL)+RM1*
     $.5D0*XL*XL
      RETURN
      END
