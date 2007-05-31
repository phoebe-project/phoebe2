      SUBROUTINE OPENSTREAM(FN)
      implicit none
      character FN*(*)
      open(unit=1, file=FN, status='UNKNOWN')
      END

      SUBROUTINE CLOSESTREAM()
      close(unit=1)
      END

      SUBROUTINE CREATELCILINE1(MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,
     +                          ICOR2,LD)
      implicit none
      integer MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD
      write(1,1) MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD
    1 format(8(I1,1X))
      END

      SUBROUTINE CREATELCILINE2(JDPHS,HJD0,PERIOD,DPDT,PSHIFT,STDDEV,
     +                          NOISE,SEED)
      implicit none
      integer JDPHS,NOISE
      double precision HJD0,PERIOD,DPDT,PSHIFT,STDDEV,SEED
      write(1,2) JDPHS,HJD0,PERIOD,DPDT,PSHIFT,STDDEV,NOISE,SEED
    2 format(I1,F15.6,D15.10,D13.6,F10.4,D10.4,I2,F11.0)
      END

      SUBROUTINE CREATELCILINE3(JDSTRT,JDEND,JDINC,PHSTRT,PHEND,PHINC,
     +                          PHNORM)
      implicit none
      double precision JDSTRT,JDEND,JDINC,PHSTRT,PHEND,PHINC,PHNORM
      write(1,3) JDSTRT,JDEND,JDINC,PHSTRT,PHEND,PHINC,PHNORM
    3 format(F14.6,F15.6,F13.6,4F12.6)
      END

      SUBROUTINE CREATELCILINE4(MODE,IPB,IFAT1,IFAT2,N1,N2,PERR0,
     +                          DPERDT,THE,VUNIT)
      implicit none
      integer MODE,IPB,IFAT1,IFAT2,N1,N2
      double precision PERR0,DPERDT,THE,VUNIT
      write(1,4) MODE,IPB,IFAT1,IFAT2,N1,N2,PERR0,DPERDT,THE,VUNIT
    4 format(4I2,2I4,F13.6,D12.5,F7.5,F8.2)
      END

      SUBROUTINE CREATELCILINE5(E,SMA,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN)
      implicit none
      double precision E,SMA,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN
      write(1,5) E,SMA,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN
    5 format(F6.5,D13.6,2F10.4,F10.4,F9.3,2F7.3,F7.2)
      END

      SUBROUTINE CREATELCILINE6(TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,
     +                          XBOL1,XBOL2,YBOL1,YBOL2)
      implicit none
      double precision TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,
     +                 YBOL1,YBOL2
      write(1,6) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,
     +           YBOL2
    6 format(2(F7.4,1X),2F7.3,3D13.6,4F7.3)
      END

      SUBROUTINE CREATELCILINE7(IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,
     +                          OPSF,MZERO,FACTOR,WLA)
      implicit none
      integer IBAND
      double precision WLA,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,MZERO,
     +                 FACTOR
      write(1,7) IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,MZERO,FACTOR,
     +           WLA
    7 format(I3,2F10.5,4F7.3,F8.4,D10.4,F8.3,F8.4,F9.6)
      END

      SUBROUTINE CREATELCILINE8(XLAT1,XLONG1,RADSP1,TEMSP1)
      implicit none
      double precision XLAT1,XLONG1,RADSP1,TEMSP1
      write(1,8) XLAT1,XLONG1,RADSP1,TEMSP1
    8 format(4F9.5)
      END

      SUBROUTINE CREATELCILINE10(XLAT2,XLONG2,RADSP2,TEMSP2)
      implicit none
      double precision XLAT2,XLONG2,RADSP2,TEMSP2
      write(1,10) XLAT2,XLONG2,RADSP2,TEMSP2
   10 format(4F9.5)
      END

      SUBROUTINE CREATESPOTSSTOPLINE()
      implicit none
      double precision SEP
      SEP = 300.0
      write(1,1) SEP
    1 format(1X,F4.0)
      END

      SUBROUTINE CREATECLOUDSSTOPLINE()
      implicit none
      double precision SEP
      SEP = 150.0
      write(1,2) SEP
    2 format(F4.0)
      END

      SUBROUTINE CREATELCIENDLINE ()
      implicit none
      integer SEP
      SEP = 9
      write(1,1) SEP
    1 format(I1)
      END

      SUBROUTINE CREATEDCILINE1(DEL1,DEL2,DEL3,DEL4,DEL5,DEL6,DEL7,
     +                          DEL8)
      implicit none
      double precision DEL1,DEL2,DEL3,DEL4,DEL5,DEL6,DEL7,DEL8
      write(1,1) DEL1,DEL2,DEL3,DEL4,DEL5,DEL6,DEL7,DEL8
    1 format(10(1X,D7.1))
      END

      SUBROUTINE CREATEDCILINE2(DEL10,DEL11,DEL12,DEL13,DEL14,DEL16,
     +                          DEL17,DEL18,DEL19,DEL20)
      implicit none
      double precision DEL10,DEL11,DEL12,DEL13,DEL14,DEL16,DEL17,DEL18,
     +                 DEL19,DEL20
      write(1,2) DEL10,DEL11,DEL12,DEL13,DEL14,DEL16,DEL17,DEL18,DEL19,
     +           DEL20
    2 format(10(1X,D7.1))
      END

      SUBROUTINE CREATEDCILINE3(DEL21,DEL22,DEL23,DEL24,DEL25,DEL31,
     +                          DEL32,DEL33,DEL34)
      implicit none
      double precision DEL21,DEL22,DEL23,DEL24,DEL25,DEL31,DEL32,DEL33,
     +                 DEL34
      write(1,3) DEL21,DEL22,DEL23,DEL24,DEL25,DEL31,DEL32,DEL33,DEL34
    3 format(10(1X,D7.1))
      END

      SUBROUTINE CREATEDCILINE4(KEP,IFDER,IFM,IFR,XLAMDA)
      implicit none
      integer I,KEP(35),IFDER,IFM,IFR
      double precision XLAMDA
      write(1,4) (KEP(I),I=1,35),IFDER,IFM,IFR,XLAMDA
    4 format(1X,2(4I1,1X),7I1,1X,4(5I1,1X),I1,1X,I1,1X,I1,D10.3)
      END

      SUBROUTINE CREATEDCILINE5(KSPA,NSPA,KSPB,NSPB)
      implicit none
      integer KSPA,NSPA,KSPB,NSPB
      write(1,5) KSPA,NSPA,KSPB,NSPB
    5 format(4I3)
      END

      SUBROUTINE CREATEDCILINE6(IFVC1,IFVC2,NLC,K0,KDISK,ISYM,NPPL)
      implicit none
      integer IFVC1,IFVC2,NLC,K0,KDISK,ISYM,NPPL
      write(1,6) IFVC1,IFVC2,NLC,K0,KDISK,ISYM,NPPL
    6 format(I1,1X,I1,1X,5I2)
      END

      SUBROUTINE CREATEDCILINE7(NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD)
      implicit none
      integer NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD
      write(1,7) NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD
    7 format(7(I1,1X))
      END

      SUBROUTINE CREATEDCILINE8(JDPHS,HJD0,PERIOD,DPDT,PSHIFT)
      implicit none
      integer JDPHS
      double precision HJD0,PERIOD,DPDT,PSHIFT
      write(1,8) JDPHS,HJD0,PERIOD,DPDT,PSHIFT
    8 format(I1,F15.6,D17.10,D14.6,F10.4)
      END

      SUBROUTINE CREATEDCILINE9(MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,
     +                          PERR0,DPERDT,THE,VUNIT)
      implicit none
      integer MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L
      double precision PERR0,DPERDT,THE,VUNIT
      write(1,9) MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,PERR0,DPERDT,THE,
     +           VUNIT
    9 format(4I2,4I3,F13.6,D12.5,F8.5,F9.3)
      END

      SUBROUTINE CREATEDCILINE10(E,A,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN)
      implicit none
      double precision E,A,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN
      write(1,10) E,A,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN
   10 format(F6.5,D13.6,2F10.4,F10.4,F9.3,2F7.3,F7.2)
      END

      SUBROUTINE CREATEDCILINE11(TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,
     +                           XBOL1,XBOL2,YBOL1,YBOL2)
      implicit none
      double precision TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,
     +                 YBOL1,YBOL2
      write(1,11) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,
     +            YBOL2
   11 format(F7.4,F8.4,2F7.3,3D13.6,4F7.3)
      END

      SUBROUTINE CREATEDCILINERV(IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,
     +                           OPSF,SIGMA,WLA)
      implicit none
      integer IBAND
      double precision WLA,HLA,CLA,X1A,X2A,Y1A,Y2A,OPSF,SIGMA
      write(1,12) IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,OPSF,SIGMA,WLA
   12 format(I3,2F10.5,4F7.3,D10.3,D12.5,F10.6)
      END

      SUBROUTINE CREATEDCILINELC(IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,
     +                            OPSF,NOISE,SIGMA,WLA)
      implicit none
      integer NOISE,IBAND
      double precision WLA,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,SIGMA
      write(1,13) IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,NOISE,
     +            SIGMA,WLA
   13 format(I3,2F10.5,4F7.3,F8.4,D10.3,I2,D12.5,F10.6)
      END

      SUBROUTINE CREATEDATALINE(INDEP,DEP,WEIGHT)
      implicit none
      double precision INDEP,DEP,WEIGHT
      write(1,1) INDEP,DEP,WEIGHT
    1 format(5(F14.5,F8.4,F6.2))
      END

      SUBROUTINE CREATEDATASTOPLINE ()
      implicit none
      double precision SEP
      SEP = -10001.0
      write (1,3) SEP
    3 format(F9.0)
      END

      SUBROUTINE CREATEDCIENDLINE ()
      implicit none
      integer SEP
      SEP = 2
      write(1,2) SEP
    2 format(I2)
      END

























      SUBROUTINE WDGETCURVE (FATMCOF,FATMCOFPL,REQMODE,PTSNO,PHASES,
     $VALUES,L1,L2,PSBR1,PSBR2)

C     This is LC from 2004-10-18 WD, changed to the subroutine so it
C     may be called without problems from PHOEBE. All stop statements have
C     been replaced with goto 9999 statements, 9999 being the end of the sub-
C     routine. When called from PHOEBE, this function returns fluxes & RVs.
C
C     This subroutine should be called with the following arguments:
C
C     INPUTS:
C
C       1) FATMCOF      ..  location of the atmcof.dat file
C       2) FATMCOFPL    ..  location of the atmcofplanck.dat file
C       3) REQMODE      ..  the requested mode:
C                             1 .. total flux
C                             2 .. RV1 in km/s
C                             3 .. RV2 in km/s
C       4) PTSNO        ..  number of the requested data points
C       5) PHASES       ..  an array of doubles with independent data points
C
C     OUTPUTS:
C
C       6) VALUES       ..  an empty array of doubles to which calculated
C                           values are written
C       7) L1           ..  star 1 luminosity (HLA)
C       8) L2           ..  star 2 luminosity (CLA)
C       9) PSBR1        ..  star 1 polar surface brightness
C      10) PSBR2        ..  star 2 polar surface brightness

      implicit real*8(a-h,o-z)
      character fatmcof*(*),fatmcofpl*(*)
      integer reqmode,ptsno
      double precision phases(*),values(*),L1,L2,PSBR1,PSBR2
      dimension rad(4),drdo(4),xtha(4),xfia(4),po(2)
      dimension rv(3011),grx(3011),gry(3011),grz(3011),rvq(3011),
     $grxq(3011),gryq(3011),grzq(3011),slump1(3011),slump2(3011),
     $fr1(3011),fr2(3011),glump1(3011),glump2(3011),xx1(3011),
     $xx2(3011),yy1(3011),yy2(3011),zz1(3011),zz2(3011),grv1(3011),
     $grv2(3011),rftemp(3011),rf1(3011),rf2(3011),csbt1(3011),
     $csbt2(3011),gmag1(3011),gmag2(3011),glog1(3011),glog2(3011),
     $hld(3200),snfi(6400),csfi(6400),tld(6400),snth(260),csth(260),
     $theta(520),rho(520),aa(20),bb(20),mmsave(124)
      dimension fbin1(100000),fbin2(100000),delv1(100000),delv2(100000),
     $count1(100000),count2(100000),delwl1(100000),delwl2(100000),
     $resf1(100000),resf2(100000),wl1(100000),wl2(100000),dvks1(100),
     $dvks2(100),wll1(100),wll2(100),tau1(100),tau2(100),emm1(100),
     $emm2(100),ewid1(100),ewid2(100),depth1(100),depth2(100),
     $hbarw1(100),hbarw2(100),taug(100000),emmg(100000)
      DIMENSION XLAT(2,100),xlong(2,100)
      dimension xcl(100),ycl(100),zcl(100),rcl(100),op1(100),fcl(100),
     $dens(100),encl(100),edens(100),xmue(100),yskp(14000),zskp(14000)
      dimension message(2,4)
      dimension abun(19),glog(11),grand(250800),plcof(1250)
      common /abung/ abun,glog
      common /arrayleg/ grand,istart
      common /planckleg/ plcof
      common /atmmessages/ message,komp
      common /ramprange/ tlowtol,thightol,glowtol,ghightol
      COMMON /FLVAR/ PSHIFT,DP,EF,EFC,ECOS,perr0,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      COMMON /DPDX/ DPDX1,DPDX2,PHSV,PCSV
      COMMON /ECCEN/ E,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      COMMON /KFAC/ KFF1,KFF2,kfo1,kfo2
      COMMON /INVAR/ KH,IPB,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,ifsmv2,
     $icor1,icor2,ld,ncl,jdphs,ipc
      COMMON /SPOTS/SINLAT(2,100),COSLAT(2,100),SINLNG(2,100),COSLNG
     $(2,100),RADSP(2,100),temsp(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      common /cld/ acm,opsf
      common /ardot/ dperdt,hjd,hjd0,perr
      common /prof2/ du1,du2,du3,du4,binw1,binw2,sc1,sc2,sl1,sl2,
     $clight
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      COMMON /NSPT/ NSP1,NSP2
      data xtha(1),xtha(2),xtha(3),xtha(4),xfia(1),xfia(2),xfia(3),
     $xfia(4)/0.d0,1.570796d0,1.570796d0,1.570796d0,
     $0.d0,0.d0,1.5707963d0,3.14159365d0/
c
c           Bandpass Label Assignments for Stellar Atmospheres
c
c    Label   Bandpass   Reference for Response Function
c    -----   --------   -------------------------------
c       1        u      Crawford, D.L. and Barnes, J.V. 1974, AJ, 75, 978
c       2        v          "                "           "
c       3        b          "                "           "
c       4        y          "                "           "
c       5        U      Buser, R. 1978, Ang, 62, 411
c       6        B      Azusienis and Straizys 1969, Sov. Astron., 13, 316
c       7        V          "             "                "
c       8        R      Johnson, H.L. 1965, ApJ, 141, 923
c       9        I         "            "    "
c      10        J         "            "    "
c      11        K         "            "    "
c      12        L         "            "    "
c      13        M         "            "    "
c      14        N         "            "    "
c      15        R_c    Bessell, M.S. 1983, PASP, 95, 480
c      16        I_c       "            "    "
c      17      230      Kallrath, J., Milone, E.F., Terrell, D., Young, A.T.
c                          1998, ApJ, 508, 308
c      18      250         "             "             "           "
c      19      270         "             "             "           "
c      20      290         "             "             "           "
c      21      310         "             "             "           "
c      22      330         "             "             "           "
c
  205 format('**********************************************************
     $************')
  204 format('*************  Next block of output   ********************
     $************')
   79 format(6x,'JD',17x,'Phase     light 1     light 2     (1+2+3)    n
     $orm lite   dist      mag+K')
   96 FORMAT(6x,'JD',13x,'Phase',5x,'r1pol',6x,'r1pt',5x,'r1sid',5x,'r1b
     $ak',5x,'r2pol',5x,'r2pt',6x,'r2sid',5x,'r2bak')
  296 format(f14.6,f13.5,8f10.5)
   45 FORMAT(6x,'JD',14x,'Phase     V Rad 1     V Rad 2      del V1
     $ del V2   V1 km/s      V2 km/s')
   93 format(f14.6,f13.5,4f12.6,2d13.4)
   47 FORMAT(2x,'band',7x,'L1',9x,'L2',7x,'x1',6x,'x2',6x,'y1',6x,
     $'y2',6x,'el3     opsf      m zero   factor',2x,'wv lth')
   48 FORMAT('  ecc',5x,'s-m axis',7x,'F1',9x,'F2',7x,'Vgam',7x,
     $'Incl',6x,'g1',6x,'g2  Nspot1 Nspot 2',4x,'[M/H]')
   54 FORMAT(2x,'T1',6x,'T2',5x,'Alb 1  Alb 2',4x,'Pot 1',8x,'Pot 2',
     $11x,'M2/M1',2x,'x1(bolo) x2(bolo) y1(bolo) y2(bolo)')
   33 FORMAT(I4,I5,I6,I6,I6,I4,f13.6,d14.5,f9.5,f10.2,d16.4)
   74 FORMAT(' DIMENSIONLESS RADIAL VELOCITIES CONTAIN FACTOR P/(2PI*A)'
     $)
   43 format(91x,'superior',5x,'inferior')
   44 format(76x,'periastron',2x,'conjunction',2x,'conjunction')
   46 FORMAT('grid1/4    grid2/4',2X,'polar sbr 1',3X,'polar sbr 2'
     $,3X,'surf. area 1',2X,'surf. area 2',7X,'phase',8X,
     $'phase',8x,'phase')
   50 FORMAT(40H PRIMARY COMPONENT EXCEEDS CRITICAL LOBE)
   51 FORMAT(42H SECONDARY COMPONENT EXCEEDS CRITICAL LOBE)
   41 FORMAT('star',4X,'r pole',5X,'deriv',5X,'r point',5X,'deriv',
     $6X,'r side',6X,'deriv',5X,'r back',6X,'deriv')
    2 FORMAT(F6.5,d13.6,2F10.4,F10.4,f9.3,2f7.3,f7.2)
    5 FORMAT(F6.5,d13.6,2F11.4,F11.4,F10.3,2f8.3,i5,i7,f10.2)
    6 FORMAT(2(F7.4,1X),2f7.3,3d13.6,4F7.3)
    8 FORMAT(f7.4,f8.4,2F7.3,3d13.6,f8.3,f9.3,f9.3,f9.3)
    3 FORMAT(f15.6,F15.5,4F12.8,F10.5,f10.4)
    1 FORMAT(4I2,2I3,f13.6,d12.5,f7.5,F8.2)
    4 FORMAT(i3,2F10.5,4F7.3,F8.4,d10.4,F8.3,F8.4,f9.6)
   34 FORMAT(i5,1X,2F11.5,4f8.3,F9.4,d11.4,F9.3,F9.4,f9.6)
   49 FORMAT(' PROGRAM SHOULD NOT BE USED IN MODE 1 OR 3 WITH NON-ZERO E
     $CCENTRICITY')
   10 FORMAT('MODE   IPB  IFAT1 IFAT2  N1  N2',4x,'Arg. Per',7x,'dPerdt
     $',4x,'Th e',4x,'V UNIT(km/s)    V FAC')
  148 format('   mpage  nref   mref   ifsmv1   ifsmv2   icor1   icor2
     $ld')
  171 format('JDPHS',5x,'J.D. zero',7x,'Period',11x,'dPdt',
     $6x,'Ph. shift',3x,'fract. sd.',2x,'noise',5x,'seed')
  244 format('Note: The light curve output contains simulated observa',
     $'tional scatter, as requested,')
  245 format('with standard deviation',f9.5,' of light at the reference'
     $,' phase.')
  149 format(i6,2i7,i8,i9,i9,i8,i6)
  170 format(i3,f17.6,d18.10,d14.6,f10.4,d13.4,i5,f13.0)
   40 FORMAT(I3,8F11.5)
   94 FORMAT(i6,i11,4F14.6,F13.6,f13.6,f13.6)
   84 FORMAT(1X,I4,4F12.5)
   85 FORMAT(4f9.5)
   83 FORMAT(1X,'STAR  CO-LATITUDE  LONGITUDE  SPOT RADIUS  TEMP. FACTOR
     $')
  150 format(' Star',9x,'M/Msun   (Mean Radius)/Rsun',5x,'M Bol',4x,'Log
     $ g (cgs)')
  250 format(4x,I1,4x,f12.3,11x,f7.2,6x,f6.2,8x,f5.2)
  350 format(' Primary star exceeds outer contact surface')
  351 format(' Secondary star exceeds outer contact surface')
   22 format(8(i1,1x))
  649 format(i1,f15.6,d15.10,d13.6,f10.4,d10.4,i2,f11.0)
   63 format(3f9.4,f7.4,d11.4,f9.4,d11.3,f9.4,f7.3)
   64 format(3f10.4,f9.4,d12.4,f10.4,d12.4,f9.4,f9.3,d12.4)
   69 format('      xcl       ycl       zcl      rcl       op1         f
     $cl        ne       mu e      encl     dens')
 2048 format(d11.5,f9.4,f9.2,i3)
 2049 format(i3,d14.5,f18.2,f20.2,i14)
  907 format(6x,'del v',6x,'del wl (mic.)',7x,'wl',9x,'profile',6x,'res
     $flux')
  903 format(6f14.7)
   92 format('Phase =',f14.6)
  142 format('star',4x,'bin width (microns)',3x,'continuum scale',4x,'co
     $ntinuum slope',2x,'nfine')
  167 format(30x,'star',i2)
  138 format(f9.6,d12.5,f10.5,i5)
  152 format(f20.6,d23.5,17x,f13.5,i6)
  157 format('star ',i1,'   line wavelength',4x,'equivalent width (micro
     $ns)',5x,'rect. line depth',2x,'kks')
  217 format(f14.6,f15.6,f13.6,4f12.6)
  218 format(f14.6,f16.6,f14.6,4f12.6)
  219 format(5x,'JD start',9x,'JD stop',6x,'JD incr',6x,
     $'Ph start',4x,'Ph. stop',5x,'Ph incr',5x,'Ph norm')
  283 format('log g below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  284 format('log g above ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  285 format('T above ramp range for at least one',
     $' point on star',i2,', black body applied locally.')
  286 format('T below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  287 format('Input [M/H] = ',f6.3,' is not a value recognized by ',
     $'the program. Replaced by ',f5.2)
      ot=1.d0/3.d0
      KH=17
      pi=dacos(-1.d0)
      clight=2.99792458d5
      en0=6.0254d23
      rsuncm=6.960d10
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  Ramp ranges are set below. The following values seem to work.
c  They may be changed.
      tlowtol=1500.d0
      thightol=50000.d0
      glowtol=4.0d0
      ghightol=4.0d0
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      abun(1)=1.d0
      abun(2)=0.5d0
      abun(3)=0.3d0
      abun(4)=0.2d0
      abun(5)=0.1d0
      abun(6)=0.0d0
      abun(7)=-0.1d0
      abun(8)=-0.2d0
      abun(9)=-0.3d0
      abun(10)=-0.5d0
      abun(11)=-1.0d0
      abun(12)=-1.5d0
      abun(13)=-2.0d0
      abun(14)=-2.5d0
      abun(15)=-3.0d0
      abun(16)=-3.5d0
      abun(17)=-4.0d0
      abun(18)=-4.5d0
      abun(19)=-5.0d0
      glog(1)=0.0d0
      glog(2)=0.5d0
      glog(3)=1.0d0
      glog(4)=1.5d0
      glog(5)=2.0d0
      glog(6)=2.5d0
      glog(7)=3.0d0
      glog(8)=3.5d0
      glog(9)=4.0d0
      glog(10)=4.5d0
      glog(11)=5.0d0
      nn=100
      gau=0.d0
      open(unit=22,file=FATMCOF,status='old')
      read(22,*) grand
      open(unit=23,file=FATMCOFPL,status='old')
      read(23,*) plcof
      close(22)
      close(23)
      open(unit=5,file='lcin.active',status='old')
      open(unit=6,file='lcout.active')
      ibef=0
      DO 1000 IT=1,1000
      read(5,22) mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      if(mpage.ne.9) goto 414
c     close(5)
c     close(6)
c     stop
      goto 9999
  414 continue
      if(ibef.eq.0) goto 335
      write(6,*)
      write(6,*)
      write(6,*)
      write(6,*)
      write(6,*)
      write(6,204)
      write(6,*)
      write(6,*)
      write(6,*)
      write(6,*)
  335 ibef=1
      message(1,1)=0
      message(1,2)=0
      message(1,3)=0
      message(1,4)=0
      message(2,1)=0
      message(2,2)=0
      message(2,3)=0
      message(2,4)=0
      read(5,649) jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed
      read(5,217) hjdst,hjdsp,hjdin,phstrt,phstop,phin,phn
      READ(5,1) MODE,IPB,IFAT1,IFAT2,N1,N2,perr0,dperdt,the,VUNIT
      READ(5,2) E,A,F1,F2,VGA,XINCL,GR1,GR2,abunin
      read(5,6) tavh,tavc,alb1,alb2,poth,potc,rm,xbol1,xbol2,ybol1,
     $ybol2
      READ(5,4)iband,HLUM,CLUM,XH,xc,yh,yc,EL3,opsf,ZERO,FACTOR,wl
      acm=rsuncm*a
c*******************************************************************
c  The following lines take care of abundances that may not be among
c  the 19 Kurucz values (see abun array). abunin is reset at the
c  allowed value nearest the input value.

      call binnum(abun,19,abunin,iab)
      dif1=abunin-abun(iab)
      if(iab.eq.19) goto 702
      dif2=abun(iab+1)-abun(iab)
      dif=dif1/dif2
      if((dif.ge.0.d0).and.(dif.le.0.5d0)) goto 702
      iab=iab+1
  702 continue
      if(dif1.ne.0.d0) write(6,287) abunin,abun(iab)
      abunin=abun(iab)
      istart=1+(iab-1)*13200
c***************************************************************
      nf1=1
      nf2=1
      if(mpage.ne.3) goto 897
      colam=clight/wl
      read(5,2048) binwm1,sc1,sl1,nf1
      binw1=colam*binwm1
      do 86 iln=1,100
      read(5,138) wll1(iln),ewid1(iln),depth1(iln),kks(1,iln)
      if(wll1(iln).lt.0.d0) goto 89
      emm1(iln)=0.d0
      if(depth1(iln).lt.0.d0) emm1(iln)=depth1(iln)
      tau1(iln)=0.d0
      if(depth1(iln).gt.0.d0) tau1(iln)=-dlog(1.d0-depth1(iln))
      hbarw1(iln)=0.d0
      if(depth1(iln).ne.0.d0) hbarw1(iln)=.5d0*clight*ewid1(iln)/
     $(wll1(iln)*dabs(depth1(iln)))
      nl1=iln
   86 continue
   89 continue
      read(5,2048) binwm2,sc2,sl2,nf2
      binw2=colam*binwm2
      do 99 iln=1,100
      read(5,138) wll2(iln),ewid2(iln),depth2(iln),kks(2,iln)
      if(wll2(iln).lt.0.d0) goto 91
      emm2(iln)=0.d0
      if(depth2(iln).lt.0.d0) emm2(iln)=depth2(iln)
      tau2(iln)=0.d0
      if(depth2(iln).gt.0.d0) tau2(iln)=-dlog(1.d0-depth2(iln))
      hbarw2(iln)=0.d0
      if(depth2(iln).ne.0.d0) hbarw2(iln)=.5d0*clight*ewid2(iln)/
     $(wll2(iln)*dabs(depth2(iln)))
      nl2=iln
   99 continue
   91 continue
      do 622 iln=1,nl1
      flam=(wll1(iln)/wl)**2
  622 dvks1(iln)=clight*(flam-1.d0)/(flam+1.d0)
      do 623 iln=1,nl2
      flam=(wll2(iln)/wl)**2
  623 dvks2(iln)=clight*(flam-1.d0)/(flam+1.d0)
  897 continue
      NSP1=0
      NSP2=0
      DO 88 KP=1,2
      DO 87 I=1,100
      READ(5,85)XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
      xlng(kp,i)=xlong(kp,i)
      IF(XLAT(KP,I).GE.200.d0) GOTO 88
      SINLAT(KP,I)=dsin(XLAT(KP,I))
      COSLAT(KP,I)=dcos(XLAT(KP,I))
      SINLNG(KP,I)=dsin(XLONG(KP,I))
      COSLNG(KP,I)=dcos(XLONG(KP,I))
      IF(KP.EQ.1)NSP1=NSP1+1
   87 IF(KP.EQ.2)NSP2=NSP2+1
   88 CONTINUE
      ncl=0
      do 62 i=1,100
      read(5,63) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i)
      if(xcl(i).gt.100.d0) goto 66
      ncl=ncl+1
      dens(i)=edens(i)*xmue(i)/en0
   62 continue
   66 dint1=pi*(1.d0-xbol1/3.d0)
      dint2=pi*(1.d0-xbol2/3.d0)
      if(ld.eq.2) DINT1=dint1+PI*2.d0*ybol1/9.d0
      if(ld.eq.2) DINT2=dint2+PI*2.d0*ybol2/9.d0
      if(ld.eq.3) dint1=dint1-.2d0*pi*ybol1
      if(ld.eq.3) dint2=dint2-.2d0*pi*ybol2
      NSTOT=NSP1+NSP2
      NP1=N1+1
      NP2=N1+N2+2
      IRTE=0
      IRVOL1=0
      IRVOL2=0
      do 421 imm=1,124
  421 mmsave(imm)=0
c
c  Note: If mmsave array is re-dimensioned, change upper limit in DO 421 loop.
c    imm of 124 is OK up to N=60.
c
      nn1=n1
      CALL SINCOS(1,nn1,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)
      CALL SINCOS(2,N2,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)
      hjd=hjd0
      CALL modlog(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $rm,poth,potc,gr1,gr2,alb1,alb2,n1,n2,f1,f2,mod,xincl,the,mode,
     $snth,csth,snfi,csfi,grv1,grv2,xx1,yy1,zz1,xx2,yy2,zz2,glump1,
     $glump2,csbt1,csbt2,gmag1,gmag2,glog1,glog2)
      CALL VOLUME(VOL1,RM,POTH,DP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR1,1)
      CALL VOLUME(VOL2,RM,POTC,DP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR2,1)
      if(e.eq.0.d0) goto 117
      DAP=1.d0+E
      P1AP=POTH-2.d0*E*RM/(1.d0-E*E)
      VL1=VOL1
      CALL VOLUME(VL1,RM,P1AP,DAP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR1,2)
      DPDX1=(POTH-P1AP)*(1.d0-E*E)*.5d0/E
      P2AP=POTC-2.d0*E/(1.d0-E*E)
      VL2=VOL2
      CALL VOLUME(VL2,RM,P2AP,DAP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR2,2)
      DPDX2=(POTC-P2AP)*(1.d0-E*E)*.5d0/E
  117 CONTINUE
      PHSV=POTH
      PCSV=POTC
      IF(E.EQ.0.d0) GOTO 61
      IF(MOD.EQ.1) WRITE(6,49)
   61 CONTINUE
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POTH,POTC,N1,N2,F1,F2,D,HLUM
     $,clum,xh,xc,yh,yc,gr1,gr2,wl,sm1,sm2,tpolh,tpolc,sbrh,sbrc,
     $tavh,tavc,alb1,alb2,xbol1,xbol2,ybol1,ybol2,phn,rm,xincl,
     $hot,cool,snth,csth,snfi,csfi,tld,glump1,glump2,xx1,xx2,yy1,yy2,
     $zz1,zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,
     $gmag2,glog1,glog2,fbin1,fbin2,delv1,delv2,count1,count2,delwl1,
     $delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,
     $hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,
     $zskp,mode,iband,ifat1,ifat2,1)
	  L1 = HLUM
	  L2 = CLUM
      PSBR1 = SBRH
      PSBR2 = SBRC
      KH=0
      if(kfo1.eq.0) goto 380
      write(6,350)
      goto 381
  380 IF(KFF1.EQ.1) WRITE(6,50)
  381 if(kfo2.eq.0) goto 382
      write(6,351)
      goto 383
  382 IF(KFF2.EQ.1) WRITE(6,51)
  383 IF((KFF1+KFF2+kfo1+kfo2).GT.0) WRITE(6,*)
      write(6,148)
      write(6,149) mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      write(6,*)
      write(6,171)
      write(6,170) jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed
      write(6,*)
      write(6,219)
      write(6,218) hjdst,hjdsp,hjdin,phstrt,phstop,phin,phn
      write(6,*)
      WRITE(6,10)
      WRITE(6,33)MODE,IPB,IFAT1,IFAT2,N1,N2,perr0,dperdt,the,VUNIT,vfac
      WRITE(6,*)
      WRITE(6,48)
      WRITE(6,5)E,A,F1,F2,VGA,XINCL,GR1,GR2,NSP1,NSP2,abunin
      WRITE(6,*)
      WRITE(6,54)
      WRITE(6,8)TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,rm,XBOL1,xbol2,ybol1,
     $ybol2
      WRITE(6,*)
      WRITE(6,47)
      WRITE(6,34)iband,HLUM,CLUM,XH,XC,yh,yc,el3,opsf,ZERO,FACTOR,wl
      ns1=1
      ns2=2
      if(mpage.ne.3) goto 174
      write(6,*)
      write(6,142)
      write(6,2049) ns1,binwm1,sc1,sl1,nf1
      write(6,2049) ns2,binwm2,sc2,sl2,nf2
      write(6,*)
      write(6,157) ns1
      do 155 iln=1,nl1
  155 write(6,152) wll1(iln),ewid1(iln),depth1(iln),kks(1,iln)
      write(6,*)
      write(6,157) ns2
      do 151 iln=1,nl2
  151 write(6,152) wll2(iln),ewid2(iln),depth2(iln),kks(2,iln)
  174 continue
      write(6,*)
      WRITE(6,*)
      IF(NSTOT.GT.0) WRITE(6,83)
      DO 188 KP=1,2
      IF((NSP1+KP-1).EQ.0) GOTO 188
      IF((NSP2+(KP-2)**2).EQ.0) GOTO 188
      NSPOT=NSP1
      IF(KP.EQ.2) NSPOT=NSP2
      DO 187 I=1,NSPOT
  187 WRITE(6,84)KP,XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
  188 WRITE(6,*)
      if(ncl.eq.0) goto 67
      write(6,69)
      do 68 i=1,ncl
   68 write(6,64) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i),dens(i)
      write(6,*)
   67 continue
      write(6,150)
      rr1=.6203505d0*vol1**ot
      rr2=.6203505d0*vol2**ot
      tav1=10000.d0*tavh
      tav2=10000.d0*tavc
      call mlrg(a,period,rm,rr1,rr2,tav1,tav2,sms1,sms2,sr1,sr2,
     $bolm1,bolm2,xlg1,xlg2)
      write(6,250) ns1,sms1,sr1,bolm1,xlg1
      write(6,250) ns2,sms2,sr2,bolm2,xlg2
      write(6,*)
      write(6,43)
      write(6,44)
      WRITE(6,46)
      WRITE(6,94) MMSAVE(NP1),MMSAVE(NP2),SBRH,SBRC,SM1,SM2,PHPERI,
     $pconsc,pconic
      WRITE(6,*)
      if(stdev.eq.0.d0.or.mpage.ne.1) goto 246
      write(6,244)
      write(6,245) stdev
  246 continue
      WRITE(6,*)
      ALL=HOT+COOL+EL3
      IF(MODE.EQ.-1) ALL=COOL+EL3
      if(mpage.eq.1) write(6,79)
      if(mpage.eq.2) write(6,45)
      if(mpage.eq.4) write(6,96)
      LL1=MMSAVE(N1)+1
      NPP2=NP2-1
      LL2=MMSAVE(NPP2)+1
      LLL1=MMSAVE(NP1)
      LLL2=MMSAVE(NP2)
      LLLL1=(LL1+LLL1)/2
      LLLL2=(LL2+LLL2)/2
      POTH=PHSV
      POTC=PCSV
      PO(1)=POTH
      PO(2)=POTC
      IF(E.EQ.0.d0) IRVOL1=1
      IF(E.EQ.0.d0) IRVOL2=1
      IF(E.EQ.0.d0) IRTE=1
      start=hjdst
      stopp=hjdsp
      step=hjdin
      if(jdphs.ne.2) goto 887
      start=phstrt
      stopp=phstop
      step=phin
  887 continue
c     do 20 phjd=start,stopp,step
      do 20 index=1,ptsno,1
c     hjdi=phjd
c     phasi=phjd
      hjdi=phases(index)
      phasi=phases(index)
      call jdph(hjdi,phasi,hjd0,period,dpdt,hjdo,phaso)
      hjd=hjdi
      phas=phasi
      if(jdphs.ne.1) hjd=hjdo
      if(jdphs.ne.2) phas=phaso
      CALL modlog(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $rm,poth,potc,gr1,gr2,alb1,alb2,n1,n2,f1,f2,mod,xincl,the,mode,
     $snth,csth,snfi,csfi,grv1,grv2,xx1,yy1,zz1,xx2,yy2,zz2,glump1,
     $glump2,csbt1,csbt2,gmag1,gmag2,glog1,glog2)
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POTH,POTC,N1,N2,F1,F2,D,hlum,
     $clum,xh,xc,yh,yc,gr1,gr2,wl,sm1,sm2,tpolh,tpolc,sbrh,sbrc,
     $tavh,tavc,alb1,alb2,xbol1,xbol2,ybol1,ybol2,phas,rm,xincl,
     $hot,cool,snth,csth,snfi,csfi,tld,glump1,glump2,xx1,xx2,yy1,yy2,
     $zz1,zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,
     $gmag2,glog1,glog2,fbin1,fbin2,delv1,delv2,count1,count2,delwl1,
     $delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,
     $hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,
     $zskp,mode,iband,ifat1,ifat2,0)
  128 format('HJD = ',f14.5,'    Phase = ',f14.5)
  131 format(3x,'Y Sky Coordinate',4x,'Z Sky Coordinate')
  130 format(f16.6,f20.6)
      if(mpage.ne.5) goto 127
      write(6,*)
      write(6,*)
      write(6,128) hjd,phas
      write(6,*)
      write(6,131)
      do 129 imp=1,ipc
      write(6,130) yskp(imp),zskp(imp)
  129 continue
      goto 20
  127 continue
      HTT=HOT
      IF(MODE.EQ.-1) HTT=0.d0
      TOTAL=HTT+COOL+EL3
      TOTALL=TOTAL/ALL
c     The following line added for PHOEBE:
      QPFLUX=ALL
c     up to here.
      TOT=TOTALL*FACTOR
      if(stdev.le.0.d0) goto 348
      call rangau(seed,nn,stdev,gau)
      ranf=1.d0+gau*dsqrt(totall**noise)
      total=total*ranf
      tot=tot*ranf
      totall=totall*ranf
  348 continue
      SMAGG=-1.085736d0*dlog(TOTALL)+ZERO
      if(mpage.eq.1) write(6,3) hjd,phas,htt,cool,total,tot,d,smagg
      if(mpage.eq.2) write(6,93) hjd,phas,vsum1,vsum2,vra1,vra2,vkm1,
     $vkm2
c     The following block added for PHOEBE:
      if(reqmode.eq.1) values(index)=total
      if(reqmode.eq.2) values(index)=vkm1
      if(reqmode.eq.3) values(index)=vkm2
c     up to here.
      if(mpage.ne.3) goto 81
      write(6,92) phas
      write(6,*)
      write(6,167) ns1
      write(6,907)
      do 906 i=in1min,in1max
  906 write(6,903) delv1(i),delwl1(i),wl1(i),fbin1(i),resf1(i)
      write(6,*)
      write(6,167) ns2
      write(6,907)
      do 908 i=in2min,in2max
  908 write(6,903) delv2(i),delwl2(i),wl2(i),fbin2(i),resf2(i)
      write(6,*)
      write(6,205)
      write(6,*)
      write(6,*)
   81 continue
      if(mpage.eq.4) write(6,296) hjd,phas,rv(1),rv(ll1),rv(llll1),
     $rv(lll1),rvq(1),rvq(ll2),rvq(llll2),rvq(lll2)
   20 CONTINUE
      do 909 komp=1,2
      write(6,*)
      if(message(komp,1).eq.1) write(6,283) komp
      if(message(komp,2).eq.1) write(6,284) komp
      if(message(komp,3).eq.1) write(6,285) komp
      if(message(komp,4).eq.1) write(6,286) komp
  909 continue
c     if(mpage.eq.5) stop
      if(mpage.eq.5) goto 9999
      WRITE(6,*)
      WRITE(6,41)
      WRITE(6,*)
      do 119 ii=1,2
      gt1=dfloat(2-ii)
      gt2=dfloat(ii-1)
      f=f1*gt1+f2*gt2
      do 118 i=1,4
      call romq(po(ii),rm,f,dp,e,xtha(i),xfia(i),rad(i),drdo(i),
     $drdq,dodq,ii,mode)
  118 continue
      write(6,40) ii,rad(1),drdo(1),rad(2),drdo(2),rad(3),drdo(3),
     $rad(4),drdo(4)
  119 continue
      WRITE(6,*)
      if(mpage.eq.2) write(6,74)
 1000 CONTINUE
c     The following part added for PHOEBE's multiple LC invocation:
 9999 continue
      close(unit=5)
      close(unit=6)
c     STOP
      END

      SUBROUTINE LIGHT(phs,xincl,xh,xc,yh,yc,n1,n2,sumhot,sumkul,rv,grx,
     $gry,grz,rvq,grxq,gryq,grzq,mmsave,theta,rho,aa,bb,slump1,slump2,
     $somhot,somkul,d,wl,snth,csth,snfi,csfi,tld,gmag1,gmag2,glog1,
     $glog2,fbin1,fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,
     $resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,
     $ycl,zcl,rcl,op1,fcl,edens,encl,dens,taug,emmg,yskp,zskp,iband,
     $ifat1,ifat2,ifphn)
c   Version of October 18, 2004
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),SLUMP1(*),SLUMP2(*),MMSAVE(*),THETA(*),RHO(*),AA(*),BB(*)
      DIMENSION SNTH(*),CSTH(*),SNFI(*),CSFI(*),tld(*),gmag1(*),
     $gmag2(*),glog1(*),glog2(*)
      dimension xcl(*),ycl(*),zcl(*),rcl(*),op1(*),fcl(*),dens(*),
     $encl(*),edens(*),yskp(*),zskp(*)
      dimension fbin1(*),fbin2(*),delv1(*),delv2(*),count1(*),count2(*),
     $delwl1(*),delwl2(*),resf1(*),resf2(*),wl1(*),wl2(*),dvks1(*),
     $dvks2(*),tau1(*),tau2(*),hbarw1(*),hbarw2(*),taug(*),emm1(*),
     $emm2(*),emmg(*)
      dimension message(2,4)
      common /atmmessages/ message,komp
      common /coflimbdark/ x,y
      COMMON /misc/ X1
      COMMON /KFAC/ KFF1,KFF2,kfo1,kfo2
      COMMON /NSPT/ NSP1,NSP2
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm
     $,ifs1dm,ifs2dm,icr1dm,icr2dm,ld,ncl,jdphs,ipc
      common /flvar/ du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,
     $du12,du13,du14,du15,du16,du17,vunit,vfvu,du20,qfacd
      common /prof2/ vo1,vo2,ff1,ff2,binw1,binw2,sc1,sc2,sl1,sl2,
     $clight
      common /cld/ acm,opsf
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      common /setest/ sefac
      common /flpro/ vksf,binc,binw,difp,deldel,renfsq
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      COMMON /SPOTS/ SINLAT(2,100),COSLAT(2,100),SINLNG(2,100),COSLNG
     $(2,100),RAD(2,100),TEMSP(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      pi=dacos(-1.d0)
      twopi=pi+pi
      pih=.5d0*pi
      dtr=pi/180.d0
      kstp=4
      cirf=.002d0
      if(ifphn.eq.1) goto 16
      if(mpage.ne.3) goto 16
      nbins=90000
      binc1=.5d0*dfloat(nbins)
      binc2=binc1
      in1max=0
      in2max=0
      in1min=300000
      in2min=300000
      marm1=10
      marp1=10
      marm2=10
      marp2=10
      do 916 i=1,nbins
      fbin1(i)=0.d0
      fbin2(i)=0.d0
      count1(i)=0.d0
      count2(i)=0.d0
      delv1(i)=0.d0
      delv2(i)=0.d0
  916 continue
   16 continue
      PHA=PHS*twopi
      K=6
      KK=K+1
      XINC=XINCL*dtr
      L=1
      TEST=(PHS-.5d0)**2
      TESTS=(TEST-.071525d0)**2
      SINI=dsin(XINC)
      COSPH=dcos(PHA)
      SINPH=dsin(PHA)
      SINSQ=SINPH**2
      COSI=dcos(XINC)
      NP1=N1+1
      NP2=N1+N2+2
      LLL1=MMSAVE(NP1)
      LLL2=MMSAVE(NP2)
      NPP2=NP2-1
      LL1=MMSAVE(N1)+1
      LL2=MMSAVE(NPP2)+1
      LLLL1=(LL1+LLL1)/2
      LLLL2=(LL2+LLL2)/2
      SINSQE=0.d0
      IF(SINI.GT.0.d0) SINSQE=((1.10d0*(RV(LLL1)+RVQ(LLL2))/D)**2
     $-cosi**2)/SINI**2
      CICP=COSI*COSPH
      CISP=COSI*SINPH
      XLOS=COSPH*SINI
      YLOS=-SINPH*SINI
      ZLOS=COSI
      SUM=0.d0
      SOM=0.d0
      IF(TEST.LE..0625d0) GOTO 18
      COMP=-1.d0
      CMP=1.d0
      COMPP=1.d0
      KOMP=2
      nl=nl2
      ffc=ff2
      voc=vo2*vfvu
      NSPOT=NSP2
      IFAT=IFAT2
      CMPP=0.d0
      X=XC
      y=yc
      EN=N2
      NPH=N2
      NP=2*N2
      nf=nf2
      GOTO 28
   18 X=XH
      y=yh
      COMP=1.d0
      KOMP=1
      nl=nl1
      ffc=ff1
      voc=vo1*vfvu
      NSPOT=NSP1
      IFAT=IFAT1
      CMP=0.d0
      COMPP=-1.d0
      CMPP=1.d0
      EN=N1
      NPH=N1
      NP=2*N1
      nf=nf1
   28 DELTH=pih/EN
      enf=dfloat(nf)
      renfsq=1.d0/(enf*enf)
      nfm1=nf-1
      r2nfdt=0.5d0*delth/enf
      vfvuff=vfvu*ffc
      AR=CMPP*RV(LLLL1)+CMP*RVQ(LLLL2)
      BR=CMPP*RV(1)+CMP*RVQ(1)
      ASQ=AR*AR
      BSQ=BR*BR
      AB=AR*BR
      absq=ab*ab
      ASBS=ASQ-BSQ
      KF=(2-KOMP)*KFF1+(KOMP-1)*KFF2
      CMPPD=CMPP*D
      CMPD=CMP*D
      NPP=NP+1
      TEMF=1.d0
      ipc=0
      DO 36 I=1,NP
      IF(I.GT.NPH)GOTO 54
      UPDOWN=1.d0
      IK=I
      GOTO 55
   54 UPDOWN=-1.d0
      IK=NPP-I
   55 CONTINUE
      IPN1=IK+(KOMP-1)*N1
      SINTH=SNTH(IPN1)
      COSTH=CSTH(IPN1)*UPDOWN
      tanth=sinth/costh
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      XM=MM
      MH=MM
      MM=2*MM
      DELFI=pi/XM
      r2nfdf=.5d0/enf
      deldel=delth*delfi
      IP=(KOMP-1)*NP1+IK
      IY=MMSAVE(IP)+1
      IF(TEST.LE..0625d0)GOTO 19
      GX=GRXQ(IY)
      GY=-GRYQ(IY)
      GZ=UPDOWN*GRZQ(IY)
      grmag=gmag2(iy)
      GOTO 29
   19 GX=GRX(IY)
      GY=-GRY(IY)
      GZ=UPDOWN*GRZ(IY)
      grmag=gmag1(iy)
   29 COSSAV=(XLOS*GX+YLOS*GY+ZLOS*GZ)/GRMAG
      SUMJ=0.d0
      SOMJ=0.d0
      MPP=MM+1
      IY=IY-1
      DO 26 J=1,MM
      IF(J.GT.MH) GOTO 58
      RTLEFT=1.d0
      JK=J
      GOTO 59
   58 RTLEFT=-1.d0
      JK=MPP-J
   59 CONTINUE
      IX=IY+JK
      IS=IX+(KOMP-1)*LLL1
      SINFI=SNFI(IS)*RTLEFT
      COSFI=CSFI(IS)
      STSF=SINTH*SINFI
      STCF=SINTH*COSFI
      IF(TEST.LE..0625d0)GOTO 39
      IF(RVQ(IX).EQ.-1.d0) GOTO 26
      GX=GRXQ(IX)
      GY=RTLEFT*GRYQ(IX)
      GZ=UPDOWN*GRZQ(IX)
      R=RVQ(IX)
      grmag=gmag2(ix)
      GOTO 49
   39 IF(RV(IX).EQ.-1.d0) GOTO 26
      GX=GRX(IX)
      GY=RTLEFT*GRY(IX)
      GZ=UPDOWN*GRZ(IX)
      R=RV(IX)
      grmag=gmag1(ix)
   49 COSGAM=(XLOS*GX+YLOS*GY+ZLOS*GZ)/GRMAG
      ZZ=R*COSTH
      YY=R*COMP*STSF
      XX=CMPD+COMP*STCF*R
      if(mpage.ne.5) goto 174
      if(cosgam.gt.0.d0) goto 174
      ipc=ipc+1
      yskp(ipc)=(xx-qfacd)*sinph+yy*cosph
      zskp(ipc)=(-xx+qfacd)*cicp+yy*cisp+zz*sini
      if(nspot.eq.0) goto 174
      call spot(komp,nspot,sinth,costh,sinfi,cosfi,temf)
      if(temf.eq.1.d0) goto 174
      yskr=yskp(ipc)
      zskr=zskp(ipc)
      kstp=4
      cirf=.002d0
      stp=twopi/dfloat(kstp)
      do 179 ang=stp,twopi,stp
      ipc=ipc+1
      yskp(ipc)=yskr+dsin(ang)*cirf
      zskp(ipc)=zskr+dcos(ang)*cirf
  179 continue
  174 continue
      if(sinsq.gt.sinsqe) goto 27
      IF(TESTS.LT.2.2562d-3) GOTO 170
      IF((STCF*R).GT.(sefac*(CMP+COMP*X1))) GOTO 129
  170 PROD=COSSAV*COSGAM
      IF(PROD.GT.0.d0) GOTO 22
      COSSAV=-COSSAV
      YSKY=XX*SINPH+YY*COSPH-cmpd*SINPH
      ZSKY=-XX*CICP+yy*CISP+ZZ*SINI+CMPD*CICP
      RHO(L)=dsqrt(YSKY**2+ZSKY**2)
      THETA(L)=dasin(ZSKY/RHO(L))
      IF(YSKY.LT.0.d0) GOTO 92
      THETA(L)=twopi+THETA(L)
      GOTO 93
   92 THETA(L)=pi-THETA(L)
   93 IF (THETA(L).GE.twopi) THETA(L)=THETA(L)
     $-twopi
      L=L+1
      GOTO 27
   22 COSSAV=COSGAM
      GOTO 27
  129 COSSAV=COSGAM
      IF(KF.LE.0) GOTO 27
      ZZ=R*COSTH
      YY=R*COMP*STSF
      XX=CMPD+COMP*STCF*R
      YSKY=XX*SINPH+YY*COSPH-cmpd*SINPH
      ZSKY=-XX*CICP+YY*CISP+ZZ*SINI+CMPD*CICP
      rptsq=YSKY**2+ZSKY**2
      rtstsq=absq/(BSQ+ASBS*(ZSKY**2/rptsq))
      IF(rptsq.LE.rtstsq) GOTO 26
   27 IF(COSGAM.GE.0.d0) GOTO 26
      COSGAM=-COSGAM
      DARKEN=1.d0-X+X*COSGAM
      if(ld.ne.2) goto 141
      if(cosgam.eq.0.d0) goto 141
      darken=darken-y*cosgam*dlog(cosgam)
      goto 147
  141 continue
      if(ld.eq.3) darken=darken-y*(1.d0-dsqrt(cosgam))
  147 if(darken.lt.0.d0) darken=0.d0
      CORFAC=1.d0
      do 923 jn=1,nl
      Lspot(komp,jn)=0
  923 if(kks(komp,jn).eq.0) Lspot(komp,jn)=1
      IF(NSPOT.EQ.0) GOTO 640
      CALL SPOT(KOMP,NSPOT,SINTH,COSTH,SINFI,COSFI,TEMF)
      IF(TEMF.EQ.1.d0) GOTO 640
      TSP=TLD(IS)*TEMF
      if(ifat.eq.0) call planckint(tld(is),iband,xintlog,xintbase)
      if(ifat.eq.0) call planckint(tsp,iband,xintlog,xintspot)
      IF(IFAT.EQ.0) GOTO 941
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      CALL atmx(TLD(IS),glogg,iband,xintlog,xintbase)
      CALL atmx(TSP,glogg,iband,xintlog,xintspot)
  941 CORFAC=xintspot/xintbase
  640 CONTINUE
      rit=1.d0
      if(ncl.eq.0) goto 818
      do 815 icl=1,ncl
      opsfcl=opsf*fcl(icl)
      call cloud(xlos,ylos,zlos,xx,yy,zz,xcl(icl),ycl(icl),zcl(icl),
     $rcl(icl),wl,op1(icl),opsfcl,edens(icl),acm,encl(icl),cmpd,
     $ri,dx,dens(icl),tau)
      rit=rit*ri
  815 continue
  818 continue
      DIF=rit*COSGAM*DARKEN*CORFAC*(CMP*SLUMP2(IX)+CMPP*SLUMP1(IX))
      v=-r*(STCF*YLOS-stsf*XLOS)*COMP
      if(ifphn.eq.1) goto 423
      if(mpage.ne.3) goto 423
      vflump=vfvuff*r*comp*costh
      vcks=v*vfvuff
      vks=vcks+voc
      vksf=vks
      dvdr=vcks/r
      dvdth=vcks/tanth
      dvdfib=vfvuff*r*comp*(sinfi*ylos+cosfi*xlos)
c     dvdfic=dvdfib*sinth
      difp=dif*deldel*renfsq
c  dvdth and dvdfi (below) each need another term involving dr/d(theta)
c    or dr/d(fi), that I will put in later. There will be a small loss
c    of accuracy for distorted stars without those terms. See notes.
      if(komp.eq.2) goto 422
      binc=binc1
      binw=binw1
      do 1045 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1046 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1047
      dfif=dfloat(jfn)*r2nfdf*delfi
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      dlr=0.d0
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
 1047 call linpro(komp,dvks1,hbarw1,tau1,emm1,count1,taug,emmg,fbin1,
     $delv1)
      if(inmin.lt.in1min) in1min=inmin
      if(inmax.gt.in1max) in1max=inmax
 1046 continue
 1045 continue
      goto 423
  422 continue
      binc=binc2
      binw=binw2
      do 1145 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1146 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1147
      dfif=dfloat(jfn)*r2nfdf*delfi
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      dlr=0.d0
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
      ffi=dacos(cosfi)
      if(sinfi.lt.0.d0) ffi=twopi-ffi
 1147 call linpro(komp,dvks2,hbarw2,tau2,emm2,count2,taug,emmg,fbin2,
     $delv2)
      if(inmin.lt.in2min) in2min=inmin
      if(inmax.gt.in2max) in2max=inmax
 1146 continue
 1145 continue
  423 continue
      DIFF=DIF*V
      SOMJ=SOMJ+DIFF
      SUMJ=SUMJ+DIF
   26 CONTINUE
      SOMJ=SOMJ*DELFI
      SUMJ=SUMJ*DELFI
      SOM=SOM+SOMJ
   36 SUM=SUM+SUMJ
      IF(SINSQ.GE.SINSQE) GOTO 75
      L=L-1
      LK=k
      if(L.lt.14) LK=L/2-1
      CALL fourls(theta,rho,L,LK,aa,bb)
   75 IF(TEST.LE..0625d0) GOTO 118
      SUMKUL=SUM*DELTH
      SOMKUL=SOM*DELTH
      X=XH
      y=yh
      KOMP=1
      nl=nl1
      ffc=ff1
      voc=vo1*vfvu
      NSPOT=NSP1
      IFAT=IFAT1
      EN=N1
      SAFTY=2.6d0*RV(LLL1)/EN
      RMAX=RVQ(LLL2)+SAFTY
      RMIN=RVQ(1)-SAFTY
      NPH=N1
      NP=2*N1
      nf=nf1
      GOTO 128
  118 X=XC
      y=yc
      KOMP=2
      nl=nl2
      ffc=ff2
      voc=vo2*vfvu
      NSPOT=NSP2
      IFAT=IFAT2
      SUMHOT=SUM*DELTH
      SOMHOT=SOM*DELTH
      if(inmax.gt.in1max) in1max=inmax
      if(inmin.lt.in1min) in1min=inmin
      EN=N2
      SAFTY=2.6d0*RVQ(LLL2)/EN
      RMAX=RV(LLL1)+SAFTY
      RMIN=RV(1)-SAFTY
      NPH=N2
      NP=2*N2
      nf=nf2
  128 DELTH=pih/EN
      enf=dfloat(nf)
      nfm1=nf-1
      renfsq=1.d0/(enf*enf)
      r2nfdt=.5d0*delth/enf
      vfvuff=vfvu*ffc
      SOM=0.d0
      SUM=0.d0
      NPP=NP+1
      TEMF=1.d0
      inmin=300000
      inmax=0
      DO 136 I=1,NP
      IF(I.GT.NPH) GOTO 154
      UPDOWN=1.d0
      IK=I
      GOTO 155
  154 UPDOWN=-1.d0
      IK=NPP-I
  155 CONTINUE
      IPN1=IK+(KOMP-1)*N1
      SINTH=SNTH(IPN1)
      COSTH=CSTH(IPN1)*UPDOWN
      tanth=sinth/costh
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      XM=MM
      MH=MM
      MM=2*MM
      DELFI=pi/XM
      deldel=delth*delfi
      SOMJ=0.d0
      SUMJ=0.d0
      SIGN=0.d0
      DRHO=1.d0
      MPP=MM+1
      DO 126 J=1,MM
      IF(J.GT.MH) GOTO 158
      RTLEFT=1.d0
      JK=J
      GOTO 159
  158 RTLEFT=-1.d0
      JK=MPP-J
  159 CONTINUE
      IP=(KOMP-1)*NP1+IK
      IX=MMSAVE(IP)+JK
      IS=IX+LLL1*(KOMP-1)
      SINFI=SNFI(IS)*RTLEFT
      COSFI=CSFI(IS)
      STSF=SINTH*SINFI
      STCF=SINTH*COSFI
      IF(TEST.LE..0625d0)GOTO 139
      IF(RV(IX).EQ.-1.d0) GOTO 126
      GX=GRX(IX)
      GY=RTLEFT*GRY(IX)
      GZ=UPDOWN*GRZ(IX)
      R=RV(IX)
      grmag=gmag1(ix)
      GOTO 149
  139 IF(RVQ(IX).EQ.-1.d0) GOTO 126
      GX=GRXQ(IX)
      GY=RTLEFT*GRYQ(IX)
      GZ=UPDOWN*GRZQ(IX)
      R=RVQ(IX)
      grmag=gmag2(ix)
  149 COSGAM=(XLOS*GX+YLOS*GY+ZLOS*GZ)/GRMAG
      IF(COSGAM.LT.0.d0) GOTO 104
      SIGN=0.d0
      OLSIGN=0.d0
      GOTO 126
  104 COSGAM=-COSGAM
      ZZ=R*COSTH
      YY=R*COMPP*STSF
      XX=CMPPD+COMPP*STCF*R
      DARKEN=1.d0-X+X*COSGAM
      if(ld.ne.2) goto 142
      if(cosgam.eq.0.d0) goto 142
      darken=darken-y*cosgam*dlog(cosgam)
      goto 148
  142 continue
      if(ld.eq.3) darken=darken-y*(1.d0-dsqrt(cosgam))
  148 if(darken.lt.0.d0) darken=0.d0
      OLDIF=DIF
      CORFAC=1.d0
      do 823 jn=1,nl
      Lspot(komp,jn)=0
  823 if(kks(komp,jn).eq.0) Lspot(komp,jn)=1
      IF(NSPOT.EQ.0) GOTO 660
      CALL SPOT(KOMP,NSPOT,SINTH,COSTH,SINFI,COSFI,TEMF)
      IF(TEMF.EQ.1.d0) GOTO 660
      TSP=TLD(IS)*TEMF
      if(ifat.eq.0) call planckint(tld(is),iband,xintlog,xintbase)
      if(ifat.eq.0) call planckint(tsp,iband,xintlog,xintspot)
      IF(IFAT.EQ.0) GOTO 661
      glogg=cmp*glog1(ix)+cmpp*glog2(ix)
      CALL atmx(TLD(IS),glogg,iband,xintlog,xintbase)
      CALL atmx(TSP,glogg,iband,xintlog,xintspot)
  661 CORFAC=xintspot/xintbase
  660 CONTINUE
      rit=1.d0
      if(ncl.eq.0) goto 718
      do 715 icl=1,ncl
      opsfcl=opsf*fcl(icl)
      call cloud(xlos,ylos,zlos,xx,yy,zz,xcl(icl),ycl(icl),zcl(icl),
     $rcl(icl),wl,op1(icl),opsfcl,edens(icl),acm,encl(icl),cmppd,
     $ri,dx,dens(icl),tau)
      rit=rit*ri
  715 continue
  718 continue
      DIF=rit*COSGAM*DARKEN*CORFAC*(CMPP*SLUMP2(IX)+CMP*SLUMP1(IX))
      v=R*(STCF*YLOS-STSF*XLOS)*COMP
      DIFF=DIF*V
      IF(SINSQ.GT.SINSQE) GOTO 63
      OLSIGN=SIGN
      OLDRHO=DRHO
      YSKY=XX*SINPH+YY*COSPH-cmpd*SINPH
      ZSKY=-XX*CICP+yy*CISP+ZZ*SINI+CMPD*CICP
      RRHO=dsqrt(ysky*ysky+zsky*zsky)
      IF(RRHO.GT.RMAX)GOTO 63
      IF(RRHO.LT.RMIN)GOTO 126
      THET=dasin(ZSKY/RRHO)
      IF(YSKY.LT.0.d0) GOTO 192
      THET=twopi+THET
      GOTO 193
  192 THET=pi-THET
  193 IF(THET.GE.twopi) THET=THET-twopi
      RHHO=0.d0
      DO 52 N=1,KK
      ENNN=N-1
      ENTHET=ENNN*THET
   52 RHHO=RHHO+AA(N)*dcos(ENTHET)+BB(N)*dsin(ENTHET)
      SIGN=1.d0
      IF(RRHO.LE.RHHO) sign=-1.d0
      if(mpage.eq.3) goto 861
      DRHO=dabs(RRHO-RHHO)
      IF((SIGN*OLSIGN).GE.0.d0) GOTO 60
      SUMDR=DRHO+OLDRHO
      FACT=-(.5d0-DRHO/SUMDR)
      IF(FACT.LT.0.d0) GOTO 198
      RDIF=OLDIF
      GOTO 199
  198 RDIF=DIF
  199 CORR=FACT*RDIF*SIGN
      CORRR=CORR*V
      SUMJ=SUMJ+CORR
      SOMJ=SOMJ+CORRR
   60 IF(SIGN.LT.0.d0) GOTO 126
   63 SUMJ=SUMJ+DIF
      SOMJ=SOMJ+DIFF
      if(mpage.ne.5) goto 127
      ipc=ipc+1
      yskp(ipc)=(xx-qfacd)*sinph+yy*cosph
      zskp(ipc)=(-xx+qfacd)*cicp+yy*cisp+zz*sini
      if(nspot.eq.0) goto 126
      call spot(komp,nspot,sinth,costh,sinfi,cosfi,temf)
      if(temf.eq.1.d0) goto 126
      yskr=yskp(ipc)
      zskr=zskp(ipc)
      stp=twopi/dfloat(kstp)
      do 189 ang=stp,twopi,stp
      ipc=ipc+1
      yskp(ipc)=yskr+dsin(ang)*cirf
      zskp(ipc)=zskr+dcos(ang)*cirf
  189 continue
      goto 126
  127 continue
      if(mpage.ne.3) goto 126
      if(ifphn.eq.1) goto 126
  861 vflump=vfvuff*r*comp*costh
      vcks=v*vfvuff
      vks=vcks+voc
      vksf=vks
      dvdr=vcks/r
      dvdth=vcks/tanth
      dvdfib=vfvuff*r*comp*(sinfi*ylos+cosfi*xlos)
      difp=dif*deldel*renfsq
      if(komp.eq.2) goto 452
      binc=binc1
      binw=binw1
      do 1245 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      snthl=costh*dthf
      zz=r*(costh-sinth*dthf)
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1246 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1247
      dfif=dfloat(jfn)*r2nfdf*delfi
      dlr=0.d0
      xx=cmppd+compp*r*snthl*(cosfi-sinfi*dfif)
      yy=r*compp*snthl*(sinfi+cosfi*dfif)
      ysky=(xx-cmpd)*sinph+yy*cosph
      zsky=(cmpd-xx)*cicp+yy*cisp+zz*sini
      rrho=dsqrt(ysky*ysky+zsky*zsky)
      if(rrho.lt.rhho) goto 1246
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
 1247 call linpro(komp,dvks1,hbarw1,tau1,emm1,count1,taug,emmg,fbin1,
     $delv1)
      if(inmax.gt.in1max) in1max=inmax
      if(inmin.lt.in1min) in1min=inmin
 1246 continue
 1245 continue
      goto 126
  452 continue
      binc=binc2
      binw=binw2
      do 1445 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      snthl=costh*dthf
      zz=r*(costh-sinth*dthf)
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1446 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1447
      dfif=dfloat(jfn)*r2nfdf*delfi
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      dlr=0.d0
      xx=cmppd+compp*r*snthl*(cosfi-sinfi*dfif)
      yy=r*compp*snthl*(sinfi+cosfi*dfif)
      ysky=(xx-cmpd)*sinph+yy*cosph
      zsky=(cmpd-xx)*cicp+yy*cisp+zz*sini
      rrho=dsqrt(ysky*ysky+zsky*zsky)
      if(rrho.lt.rhho) goto 1446
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
 1447 call linpro(komp,dvks2,hbarw2,tau2,emm2,count2,taug,emmg,fbin2,
     $delv2)
      if(inmax.gt.in2max) in2max=inmax
      if(inmin.lt.in2min) in2min=inmin
 1446 continue
 1445 continue
  126 CONTINUE
      SOMJ=SOMJ*DELFI
      SUMJ=SUMJ*DELFI
      SOM=SOM+SOMJ
  136 SUM=SUM+SUMJ
      if(mpage.eq.5) return
      IF(TEST.LE..0625d0) GOTO 120
      SOMHOT=SOM*DELTH
      SUMHOT=SUM*DELTH
      GOTO 121
  120 SUMKUL=SUM*DELTH
      SOMKUL=SOM*DELTH
  121 continue
      if(ifphn.eq.1) return
      if(mpage.ne.3) return
      in1min=in1min-marm1
      in1max=in1max+marp1
      in2min=in2min-marm2
      in2max=in2max+marp2
      if(nl1.eq.0) goto 3115
      do 2912 i=in1min,in1max
      fbin1(i)=1.d0-fbin1(i)/sumhot
      if(count1(i).eq.0.d0) goto 2918
      delv1(i)=delv1(i)/count1(i)
      goto 2919
 2918 delv1(i)=binw1*(dfloat(i)-binc1)
 2919 vdc=delv1(i)/clight
      vfc=dsqrt((1.d0+vdc)/(1.d0-vdc))
      delwl1(i)=wl*(vfc-1.d0)
      wl1(i)=wl*vfc
      resf1(i)=(sl1*delwl1(i)+sc1)*fbin1(i)
 2912 continue
 3115 if(nl2.eq.0) return
      do 2914 i=in2min,in2max
      fbin2(i)=1.d0-fbin2(i)/sumkul
      if(count2(i).eq.0.d0) goto 2917
      delv2(i)=delv2(i)/count2(i)
      goto 2920
 2917 delv2(i)=binw2*(dfloat(i)-binc2)
 2920 vdc=delv2(i)/clight
      vfc=dsqrt((1.d0+vdc)/(1.d0-vdc))
      delwl2(i)=wl*(vfc-1.d0)
      wl2(i)=wl*vfc
      resf2(i)=(sl2*delwl2(i)+sc2)*fbin2(i)
 2914 continue
      return
      END

      SUBROUTINE SINCOS (KOMP,N,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)
c  Version of November 9, 1995
      implicit real*8 (a-h,o-z)
      DIMENSION SNTH(*),CSTH(*),SNFI(*),CSFI(*),MMSAVE(*)
      IP=(KOMP-1)*(N1+1)+1
      IQ=IP-1
      IS=0
      IF(KOMP.EQ.2) IS=MMSAVE(IQ)
      MMSAVE(IP)=0
      EN=N
      DO 8 I=1,N
      EYE=I
      EYE=EYE-.5d0
      TH=1.570796326794897d0*EYE/EN
      IPN1=I+N1*(KOMP-1)
      SNTH(IPN1)=dsin(TH)
      CSTH(IPN1)=dcos(TH)
      EM=SNTH(IPN1)*EN*1.3d0
      MM=EM+1.d0
      XM=MM
      IP=(KOMP-1)*(N1+1)+I+1
      IQ=IP-1
      MMSAVE(IP)=MMSAVE(IQ)+MM
      DO 8 J=1,MM
      IS=IS+1
      XJ=J
      FI=3.141592653589793d0*(XJ-.5d0)/XM
      CSFI(IS)=dcos(FI)
      SNFI(IS)=dsin(FI)
    8 CONTINUE
      RETURN
      END

      SUBROUTINE SURFAS(RMASS,POTENT,N,N1,KOMP,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,FF,D,SNTH,CSTH,SNFI,CSFI,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,
     $GMAG2,glog1,glog2,GREXP)
c  Version of June 9, 2004
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),MMSAVE(*),FR1(*),FR2(*),HLD(*),SNTH(*),CSTH(*),SNFI(*),CSFI(*)
     $,GRV1(*),GRV2(*),XX1(*),YY1(*),ZZ1(*),XX2(*),YY2(*),ZZ2(*),GLUMP1
     $(*),GLUMP2(*),CSBT1(*),CSBT2(*),GMAG1(*),GMAG2(*),glog1(*),
     $glog2(*)
      common /gpoles/ gplog1,gplog2
      common /radi/ R1H,RLH,R1C,RLC
      COMMON /misc/ X1
      COMMON /ECCEN/e,smaxis,period,vgadum,sindum,vfdum,vfadum,vgmdum,
     $v1dum,v2dum,ifcdum
      DSQ=D*D
      RMAS=RMASS
c     KOMP=1: star 1; KOMP=2: star 2.
      IF(KOMP.EQ.2) RMAS=1.d0/RMASS
      RF=FF**2
      RTEST=0.d0
      IP=(KOMP-1)*(N1+1)+1
      IQ=IP-1
      IS=0
      ISX=(KOMP-1)*MMSAVE(IQ)
      MMSAVE(IP)=0
      KFLAG=0
      CALL ELLONE (FF,D,RMAS,X1,OMEGA,XL2,OM2)
      IF(KOMP.EQ.2) OMEGA=RMASS*OMEGA+.5d0*(1.d0-RMASS)
      X2=X1
      IF(KOMP.EQ.2) X1=1.d0-X1
      IF(E.NE.0.d0) GOTO 43
      IF(POTENT.LT.OMEGA) CALL NEKMIN(RMASS,POTENT,X1,ZZ)
      IF(POTENT.LT.OMEGA) X2=1.d0-X1
   43 COMP=dfloat(3-2*KOMP)
      CMP=dfloat(KOMP-1)
      CMPD=CMP*D
      TESTER=CMPD+COMP*X1
      RM1=RMASS+1.d0
      RMS=RMASS
      RM1S=RM1
      IF(KOMP.NE.2) GOTO 15
      POT=POTENT/RMASS+.5d0*(RMASS-1.d0)/RMASS
      RM=1.d0/RMASS
      RM1=RM+1.d0
      GOTO 20
   15 POT=POTENT
      RM=RMASS
   20 EN=N
c ********************************************
c  Find the relative polar radius, R/a
      DELR=0.d0
      R=1.d0/pot
      knt=0
  714 R=R+DELR
      knt=knt+1
      tolr=1.d-6*dabs(r)
      RSQ=R*R
      PAR=DSQ+RSQ
      RPAR=dsqrt(PAR)
      OM=1.d0/R+RM/RPAR
      DOMR=1.d0/(-1.d0/RSQ-RM*R/(PAR*RPAR))
      DELR=(POT-OM)*DOMR
      ABDELR=dabs(DELR)
      IF(ABDELR.GT.tolr) GOTO 714
      rpole=r
      rsave=r
c ********************************************
c  Now compute GRPOLE (exactly at the pole)
      x=cmpd
      zsq=rpole*rpole
      PAR1=x*x+zsq
      RPAR1=dsqrt(PAR1)
      XNUM1=1.d0/(PAR1*RPAR1)
      XL=D-X
      PAR2=XL**2+zsq
      RPAR2=dsqrt(PAR2)
      XNUM2=1.d0/(PAR2*RPAR2)
      OMZ=-rpole*(XNUM1+RMS*XNUM2)
      OMX=RMS*XL*XNUM2-X*XNUM1+RM1S*X*RF-RMS/DSQ
      IF(KOMP.EQ.2) OMX=RMS*XL*XNUM2-X*XNUM1-RM1S*XL*RF+1.d0/DSQ
      grpole=dsqrt(OMX*OMX+OMZ*OMZ)
c ********************************************
      call gabs(komp,smaxis,rmass,e,period,d,rpole,xmas,xmaso,absgr,
     $glogg)
      if(komp.eq.1) gplog1=glogg
      if(komp.eq.2) gplog2=glogg
      DO 8 I=1,N
      IF(I.NE.2) GOTO 82
      IF(KOMP.EQ.1) RTEST=.3d0*RV(1)
      IF(KOMP.EQ.2) RTEST=.3d0*RVQ(1)
   82 CONTINUE
      IPN1=I+N1*(KOMP-1)
      SINTH=SNTH(IPN1)
      XNU=CSTH(IPN1)
      XNUSQ=XNU**2
      EM=SINTH*EN*1.3d0
      XLUMP=1.d0-XNUSQ
      MM=EM+1.d0
      afac=rf*rm1*xlump
      DO 8 J=1,MM
      KOUNT=0
      IS=IS+1
      ISX=ISX+1
      DELR=0.d0
      COSFI=CSFI(ISX)
      XMU=SNFI(ISX)*SINTH
      XLAM=SINTH*COSFI
      bfac=xlam*d
      efac=rm*xlam/dsq
      R=RSAVE
      oldr=r
      knth=0
   14 R=R+DELR
      tolr=1.d-6*dabs(r)
      if(kount.lt.1) goto 170
      if(knth.gt.20) goto 170
      if(r.gt.0.d0.and.r.lt.tester) goto 170
      knth=knth+1
      delr=0.5d0*delr
      r=oldr
      goto 14
  170 continue
      KOUNT=KOUNT+1
      IF(KOUNT.LT.80) GOTO 70
      KFLAG=1
      R=-1.d0
      GOTO 86
   70 continue
      RSQ=R*R
      rcube=r*rsq
      PAR=DSQ-2.d0*XLAM*R*D+RSQ
      RPAR=dsqrt(PAR)
      par32=par*rpar
      par52=par*par32
      OM=1.d0/R+RM*((1.d0/RPAR)-XLAM*R/DSQ)+RM1*.5d0*RSQ*XLUMP*RF
      denom=RF*RM1*XLUMP*R-1.d0/RSQ-(RM*(R-XLAM*D))/par32-efac
      domr=1.d0/denom
      d2rdo2=-domr*(afac+2.d0/rcube-rm*(1.d0/par32-3.d0*(r-bfac)**2/
     $par52))/denom**2
      DELR=(POT-OM)*DOMR+.5d0*(pot-om)**2*d2rdo2
      oldr=r
      ABDELR=dabs(DELR)
      IF(ABDELR.GT.tolr) GOTO 14
      ABR=dabs(R)
      IF(R.GT.RTEST) GOTO 74
      KFLAG=1
      R=-1.d0
      IF(KOMP.EQ.2) GOTO 98
      GOTO 97
   74 IF(ABR.LT.TESTER) RSAVE=R
      Z=R*XNU
      Y=COMP*R*XMU
      X2T=ABR*XLAM
      X=CMPD+COMP*X2T
      IF(KOMP.EQ.2) GOTO 62
      IF(X.LT.X1) GOTO 65
      KFLAG=1
      R=-1.d0
      GOTO 97
   62 IF(X2T.LT.X2) GOTO 65
      KFLAG=1
      R=-1.d0
      GOTO 98
   65 SUMSQ=Y**2+Z**2
      PAR1=X**2+SUMSQ
      RPAR1=dsqrt(PAR1)
      XNUM1=1.d0/(PAR1*RPAR1)
      XL=D-X
      PAR2=XL**2+SUMSQ
      RPAR2=dsqrt(PAR2)
      XNUM2=1.d0/(PAR2*RPAR2)
      OMZ=-Z*(XNUM1+RMS*XNUM2)
      OMY=Y*(RM1S*RF-XNUM1-RMS*XNUM2)
      OMX=RMS*XL*XNUM2-X*XNUM1+RM1S*X*RF-RMS/DSQ
      IF(KOMP.EQ.2) OMX=RMS*XL*XNUM2-X*XNUM1-RM1S*XL*RF+1.d0/DSQ
      GRMAG=dsqrt(OMX*OMX+OMY*OMY+OMZ*OMZ)
      grvrat=grmag/grpole
      GRAV=grvrat**GREXP
      A=COMP*XLAM*OMX
      B=COMP*XMU*OMY
      C=XNU*OMZ
      COSBET=-(A+B+C)/GRMAG
      IF(COSBET.LT..7d0) COSBET=.7d0
   86 IF(KOMP.EQ.2) GOTO 98
   97 RV(IS)=R
      GRX(IS)=OMX
      GRY(IS)=OMY
      GRZ(IS)=OMZ
      GMAG1(IS)=dsqrt(OMX*OMX+OMY*OMY+OMZ*OMZ)
      glog1(is)=dlog10(grvrat*absgr)
      FR1(IS)=1.d0
      GLUMP1(IS)=R*R*SINTH/COSBET
      GRV1(IS)=GRAV
      XX1(IS)=X
      YY1(IS)=Y
      ZZ1(IS)=Z
      CSBT1(IS)=COSBET
      GOTO 8
   98 RVQ(IS)=R
      GRXQ(IS)=OMX
      GRYQ(IS)=OMY
      GRZQ(IS)=OMZ
      GMAG2(IS)=dsqrt(OMX*OMX+OMY*OMY+OMZ*OMZ)
      glog2(is)=dlog10(grvrat*absgr)
      FR2(IS)=1.d0
      GLUMP2(IS)=R*R*SINTH/COSBET
      GRV2(IS)=GRAV
      XX2(IS)=X
      YY2(IS)=Y
      ZZ2(IS)=Z
      CSBT2(IS)=COSBET
    8 CONTINUE
      if(e.ne.0.d0.or.ff.ne.1.d0) goto 53
      IF(KFLAG.EQ.0) GOTO 53
      ISS=IS-1
      IF(KOMP.NE.1) GOTO 50
      CALL RING(RMASS,POTENT,1,N,FR1,HLD,R1H,RLH)
      DO 55 I=1,ISS
      IPL=I+1
      IF(RV(I).GE.0.d0)GOTO 55
      FR1(IPL)=FR1(IPL)+FR1(I)
      FR1(I)=0.d0
   55 CONTINUE
   53 IF(KOMP.EQ.2) GOTO 54
      IS=0
      DO 208 I=1,N
      IPN1=I+N1*(KOMP-1)
      EM=SNTH(IPN1)*EN*1.3d0
      MM=EM+1.d0
      DO 208 J=1,MM
      IS=IS+1
      GLUMP1(IS)=FR1(IS)*GLUMP1(IS)
  208 CONTINUE
      RETURN
   50 if(e.ne.0.d0.or.ff.ne.1.d0) goto 54
      CALL RING(RMASS,POTENT,2,N,FR2,HLD,R1C,RLC)
      DO 56 I=1,IS
      IPL=I+1
      IF(RVQ(I).GE.0.d0) GOTO 56
      FR2(IPL)=FR2(IPL)+FR2(I)
      FR2(I)=0.d0
   56 CONTINUE
   54 CONTINUE
      IS=0
      DO 108 I=1,N
      IPN1=I+N1*(KOMP-1)
      EM=SNTH(IPN1)*EN*1.3d0
      MM=EM+1.d0
      DO 108 J=1,MM
      IS=IS+1
      GLUMP2(IS)=FR2(IS)*GLUMP2(IS)
  108 CONTINUE
      RETURN
      END

      SUBROUTINE BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,
     $HLD,SLUMP1,SLUMP2,THETA,RHO,AA,BB,PHSV,PCSV,N1,N2,F1,F2,d,hlum,
     $clum,xh,xc,yh,yc,gr1,gr2,wl,sm1,sm2,tpolh,tpolc,sbrh,sbrc,
     $tavh,tavc,alb1,alb2,xbol1,xbol2,ybol1,ybol2,phas,rm,
     $xincl,hot,cool,snth,csth,snfi,csfi,tld,glump1,glump2,xx1,xx2,
     $yy1,yy2,zz1,zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,
     $csbt2,gmag1,gmag2,glog1,glog2,fbin1,fbin2,delv1,delv2,count1,
     $count2,delwl1,delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,
     $emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,encl,edens,
     $taug,emmg,yskp,zskp,mode,iband,ifat1,ifat2,ifphn)
c  Version of December 18, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),
     $GRZQ(*),MMSAVE(*),FR1(*),FR2(*),HLD(*),SLUMP1(*),SLUMP2(*),
     $THETA(*),RHO(*),AA(*),BB(*),SNTH(*),CSTH(*),SNFI(*),CSFI(*),TLD(*)
     $,GLUMP1(*),GLUMP2(*),XX1(*),XX2(*),YY1(*),YY2(*),ZZ1(*),ZZ2(*),
     $GRV1(*),GRV2(*),RFTEMP(*),RF1(*),RF2(*),CSBT1(*),CSBT2(*)
     $,GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      dimension fbin1(*),fbin2(*),delv1(*),delv2(*),count1(*),count2(*),
     $delwl1(*),delwl2(*),resf1(*),resf2(*),wl1(*),wl2(*),dvks1(*),
     $dvks2(*),tau1(*),tau2(*),hbarw1(*),hbarw2(*),taug(*),emm1(*),
     $emm2(*),emmg(*)
      dimension xcl(*),ycl(*),zcl(*),rcl(*),op1(*),fcl(*),dens(*),
     $edens(*),encl(*),yskp(*),zskp(*)
      COMMON /INVAR/ KH,IPBDUM,IRTE,NREF,IRVOL1,irvol2,mref,ifsmv1,
     $ifsmv2,icor1,icor2,ld,ncl,jdphs,ipc
      COMMON /FLVAR/ PSHIFT,DP,EF,EFC,ECOS,perr0,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      common /nspt/ nsp1,nsp2
      common /ardot/ dperdt,hjd,hjd0,perr
      common /spots/ snlat(2,100),cslat(2,100),snlng(2,100),
     $cslng(2,100),rdsp(2,100),tmsp(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      COMMON /ECCEN/ E,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      common /prof2/ vo1,vo2,ff1,ff2,du1,du2,du3,du4,du5,du6,du7
      pi=3.141592653589793d0
      twopi=pi+pi
      ff1=f1
      ff2=f2
      qfac1=1.d0/(1.d0+rm)
      qfac=rm*qfac1
      IF(MODE.EQ.1) XC=XH
      if(mode.eq.1) yc=yh
      PSFT=PHAS-PHPERI
   29 if(PSFT.GT.1.d0) PSFT=PSFT-1.d0
      if(psft.gt.1.d0) goto 29
   30 if(PSFT.LT.0.d0) PSFT=PSFT+1.d0
      if(psft.lt.0.d0) goto 30
      XMEAN=PSFT*twopi
      tr=xmean
      do 60 kp=1,2
      nsp=nsp1*(2-kp)+nsp2*(kp-1)
      ff=f1*dfloat(2-kp)+f2*dfloat(kp-1)
      ifsmv=ifsmv1*(2-kp)+ifsmv2*(kp-1)
      if(ifsmv.eq.0) goto 60
      do 61 i=1,nsp
      xlg=xlng(kp,i)+twopi*ff*(phas-pconsc)-(tr-trc)
      snlng(kp,i)=dsin(xlg)
      cslng(kp,i)=dcos(xlg)
   61 continue
   60 continue
      if(e.ne.0.d0) call KEPLER(XMEAN,E,DUM,TR)
      U=TR+PERR
      COSU=dcos(U)
      GPHA=U*.1591549d0-.25d0
   40 if(GPHA.lt.0.d0) GPHA=GPHA+1.d0
      if(gpha.lt.0.d0) goto 40
   50 if(GPHA.GE.1.d0) GPHA=GPHA-1.d0
      if(gpha.ge.1.d0) goto 50
      D=EF/(1.d0+E*dcos(TR))
      qfacd=qfac*d
      IF(IRTE.EQ.1) GOTO 19
      CALL LCR(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $slump1,SLUMP2,RM,PHSV,PCSV,N1,N2,F1,F2,D,HLUM,CLUM,xh,xc,yh,yc,
     $gr1,gr2,SM1,SM2,TPOLH,TPOLC,SBRH,SBRC,IFAT1,IFAT2,TAVH,TAVC,
     $alb1,alb2,xbol1,xbol2,ybol1,ybol2,vol1,vol2,snth,csth,snfi,csfi,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,dint1,dint2,grv1,grv2,
     $csbt1,csbt2,rftemp,rf1,rf2,gmag1,gmag2,glog1,glog2,mode,iband)
   19 CONTINUE
      VO1=qfac*SINI*(ECOS+COSU)/EFC+VGAM
      VO2=-qfac1*SINI*(ECOS+COSU)/EFC+VGAM
      call light(gpha,xincl,xh,xc,yh,yc,n1,n2,hot,cool,rv,grx,gry,grz,
     $rvq,grxq,gryq,grzq,mmsave,theta,rho,aa,bb,slump1,slump2,somhot,
     $somkul,d,wl,snth,csth,snfi,csfi,tld,gmag1,gmag2,glog1,glog2,fbin1,
     $fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,resf2,wl1,wl2,
     $dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,rcl,op1,
     $fcl,edens,encl,dens,taug,emmg,yskp,zskp,iband,ifat1,ifat2,ifphn)
      VRA1=0.d0
      VRA2=0.d0
      IF(HOT.GT.0.d0) VRA1=F1*SOMHOT/HOT
      IF(COOL.GT.0.d0) VRA2=F2*SOMKUL/COOL
      vsum1=vo1
      vsum2=vo2
      if(icor1.eq.1) vsum1=vo1+vra1
      if(icor2.eq.1) vsum2=vo2+vra2
      vfcc=vfac/vunit
      VKM1=VSUM1*vfcc
      VKM2=VSUM2*vfcc
      RETURN
      END
      SUBROUTINE DURA(F,XINCL,RM,D,THE,OMEG,R)
c  Version of May 19, 1996
C
C     PARAMETER 'THE' IS THE SEMI-DURATION OF X-RAY ECLIPSE, AND SHOULD
C     BE IN CIRCULAR MEASURE.
      IMPLICIT REAL*8(A-H,O-Z)
      DELX=0.D0
      FSQ=F*F
      RMD=1.d0/RM
      RMD1=RMD+1.D0
      XINC=.017453293d0*XINCL
      TH=6.2831853071795865d0*THE
      CI=DCOS(XINC)
      SI=DSIN(XINC)
      DSQ=D*D
      ST=DSIN(TH)
      CT=DCOS(TH)
      COTI=CI/SI
      TT=ST/CT
      C1=CT*SI
      C2=TT*ST*SI
      C3=C1+C2
      C4=COTI*CI/CT
      C5=C3+C4
      C6=C2+C4
      C7=(ST*ST+COTI*COTI)/CT**2
      X=D*(SI*SI*ST*ST+CI*CI)+.00001D0
   15 X=X+DELX
      PAR=X*X+C7*(D-X)**2
      RPAR=DSQRT(PAR)
      PAR32=PAR*RPAR
      PAR52=PAR*PAR32
      FC=(C6*D-C5*X)/PAR32+C1**3*C5*RMD/(D-X)**2+C3*FSQ*RMD1*X-C2*FSQ*D*
     $RMD1-C1*RMD/DSQ
      DFCDX=(-C5*PAR-3.D0*(C6*D-C5*X)*((1.D0+C7)*X-C7*D))/PAR52+2.D0*C1
     $**3*C5*RMD/(D-X)**3+C3*FSQ*RMD1
      DELX=-FC/DFCDX
      ABDELX=DABS(DELX)
      IF(ABDELX.GT.1.d-7) GOTO 15
      Y=-(D-X)*TT
      Z=-(D-X)*COTI/CT
      YZ2=Y*Y+Z*Z
      OMEG=1.D0/DSQRT(X*X+YZ2)+RMD/DSQRT((D-X)**2+YZ2)+.5D0*RMD1*FSQ*
     $(X*X+Y*Y)-RMD*X/DSQ
      OMEG=RM*OMEG+.5d0*(1.d0-RM)
      R=DSQRT(X*X+YZ2)
      RETURN
      END

      SUBROUTINE LCR(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,
     $hld,SLUMP1,SLUMP2,RM,POTH,POTC,N1,N2,F1,F2,D,HLUM,CLUM,xh,xc,yh,
     $yc,GR1,GR2,SM1,SM2,TPOLH,TPOLC,SBRH,SBRC,IFAT1,IFAT2,TAVH,TAVC,
     $alb1,alb2,xbol1,xbol2,ybol1,ybol2,vol1,vol2,snth,csth,snfi,csfi,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,dint1,dint2,grv1,grv2,
     $csbt1,csbt2,rftemp,rf1,rf2,gmag1,gmag2,glog1,glog2,mode,iband)
c  Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),SLUMP1(*),SLUMP2(*),MMSAVE(*),FR1(*),FR2(*),HLD(*),SNTH(*),
     $CSTH(*),SNFI(*),CSFI(*),TLD(*),GLUMP1(*),GLUMP2(*),XX1(*),XX2(*)
     $,YY1(*),YY2(*),ZZ1(*),ZZ2(*),GRV1(*),GRV2(*),RFTEMP(*),RF1(*),
     $RF2(*),CSBT1(*),CSBT2(*),GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      dimension message(2,4)
      common /atmmessages/ message,komp
      common /coflimbdark/ xld,yld
      COMMON /DPDX/ DPDX1,DPDX2,PHSV,PCSV
      COMMON /ECCEN/ E,dum1,dum2,dum3,dum4,dum5,dum6,dum7,dum8,dum9,ifc
      COMMON /SUMM/ SUMM1,SUMM2
      COMMON /INVAR/ KHDUM,IPB,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,
     $ifsmv2,icor1,icor2,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      nn1=n1
      VL1=VOL1
      VL2=VOL2
      DP=1.d0-E
      IF(IRVOL1.EQ.1) GOTO 88
      CALL VOLUME(VL1,RM,POTH,DP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM1,SM1,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR1,1)
      IF(E.EQ.0.d0) GOTO 88
      POTHD=PHSV
      IF(IFC.EQ.2) POTHD=PHSV+DPDX1*(1.d0/D-1.d0/(1.d0-E))
      CALL VOLUME(VL1,RM,POTHD,D,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM1,SM1,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR1,IFC)
   88 CONTINUE
      IF(IRVOL2.EQ.1) GOTO 86
      CALL VOLUME(VL2,RM,POTC,DP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM2,SM2,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR2,1)
      IF(E.EQ.0.d0) GOTO 86
      POTCD=PCSV
      IF(IFC.EQ.2) POTCD=PCSV+DPDX2*(1.d0/D-1.d0/(1.d0-E))
      CALL VOLUME(VL2,RM,POTCD,D,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM2,SM2,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR2,IFC)
   86 CONTINUE
      TPOLH=TAVH*dsqrt(dsqrt(SM1/SUMM1))
      TPOLC=TAVC*dsqrt(dsqrt(SM2/SUMM2))
      g1=gmag1(1)
      g2=gmag2(1)
      IF(MODE.EQ.1)TPOLC=TPOLH*dsqrt(dsqrt((G2/G1)**GR1))
      IF(MODE.EQ.1)TAVC=TPOLC/dsqrt(dsqrt(SM2/SUMM2))
      tph=10000.d0*tpolh
      tpc=10000.d0*tpolc
      komp=1
      xld=xh
      yld=yh
      if(ifat1.eq.0) call planckint(tph,iband,xintlog1,xint1)
      IF(IFAT1.NE.0) CALL atmx(tph,gplog1,iband,xintlog1,xint1)
      call lum(hlum,xh,yh,tpolh,n1,n1,1,sbrh,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ1d,fr1,sm1d,ifat1,vold,rm,
     $poth,f1,d,snth,iband)
      komp=2
      xld=xc
      yld=yc
      if(ifat2.eq.0) call planckint(tpc,iband,xintlog2,xint2)
      IF(IFAT2.NE.0) CALL atmx(tpc,gplog2,iband,xintlog2,xint2)
      sbrc=sbrh*xint2/xint1
      call lum(clum,xc,yc,tpolc,n2,n1,2,sbrt,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ2d,fr2,sm2d,ifat2,vold,rm,
     $potc,f2,d,snth,iband)
      IF(IPB.EQ.1) SBRC=SBRT
      IF(MODE.GT.0)CLUM=CLUM*SBRC/SBRT
      IF(MODE.LE.0)SBRC=SBRT
      if(mref.eq.2) goto 30
      radrat=(vol1/vol2)**(1.d0/3.d0)
      ratbol=radrat**2*(tavh/tavc)**4
      rb=1.d0/ratbol
      xld=xh
      yld=yh
      call olump(rv,grx,gry,grz,rvq,grxq,gryq,grzq,slump1,slump2,mmsave
     $,gr1,alb1,rb,tpolh,sbrh,summ1,n1,n2,1,ifat1,xc,yc,d,snth
     $,csth,snfi,csfi,tld,glump1,glump2,glog1,glog2,grv1,grv2,iband)
      rb=ratbol
      xld=xc
      yld=yc
      call olump(rv,grx,gry,grz,rvq,grxq,gryq,grzq,slump1,slump2,mmsave
     $,gr2,alb2,rb,tpolc,sbrc,summ2,n1,n2,2,ifat2,xh,yh,d,snth
     $,csth,snfi,csfi,tld,glump1,glump2,glog1,glog2,grv1,grv2,iband)
      return
   30 continue
      sbr1b=tpolh**4/dint1
      sbr2b=tpolc**4/dint2
      LT=N1+1
      IMAX1=MMSAVE(LT)
      DO 80 I=1,IMAX1
      RFTEMP(I)=1.d0
   80 RF1(I)=1.d0
      LT=N1+N2+2
      IMAX2=MMSAVE(LT)
      DO 81 I=1,IMAX2
   81 RF2(I)=1.d0
      DO 93 NR=1,NREF
      xld=xh
      yld=yh
      CALL LUMP(GRX,GRY,GRZ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,MMSAVE,
     $alb1,tpolh,sbrh,n1,n2,1,ifat1,fr1,snth,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,xbol2,ybol2,grv1,
     $grv2,sbr1b,sbr2b,rftemp,rf2,gmag1,gmag2,glog1,glog2,dint1,iband)
      xld=xc
      yld=yc
      CALL LUMP(GRX,GRY,GRZ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,MMSAVE,
     $ALB2,TPOLC,SBRC,N1,N2,2,IFAT2,fr2,snth,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,xbol1,ybol1,
     $grv1,grv2,sbr1b,sbr2b,rf2,rf1,gmag1,gmag2,glog1,glog2,dint2,iband)
      DO 70 I=1,IMAX1
   70 RF1(I)=RFTEMP(I)
   93 CONTINUE
      RETURN
      END

      SUBROUTINE VOLUME(V,Q,P,D,FF,N,N1,KOMP,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM,SM,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1
     $,GMAG2,glog1,glog2,GREXP,IFC)
c  Version of December 5, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),MMSAVE(*),FR1(*),FR2(*),HLD(*),SNTH(*),CSTH(*),SNFI(*),CSFI(*)
     $,GRV1(*),GRV2(*),GLUMP1(*),GLUMP2(*),XX1(*),YY1(*),ZZ1(*),XX2(*),
     $YY2(*),ZZ2(*),CSBT1(*),CSBT2(*),GMAG1(*),GMAG2(*),glog1(*),
     $glog2(*)
      if(ifc.eq.1) v=0.d0
      DP=1.d-5*P
      ot=1.d0/3.d0
      IF (IFC.EQ.1) DP=0.d0
      tolr=1.d-8
      DELP=0.d0
      KNTR=0
   16 P=P+DELP
      KNTR=KNTR+1
      IF(KNTR.GE.20) tolr=tolr+tolr
      PS=P
      DO 17 I=1,IFC
      P=PS
      IF(I.EQ.1) P=P+DP
      CALL SURFAS(Q,P,N,N1,KOMP,RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,
     $MMSAVE,FR1,FR2,HLD,FF,D,SNTH,CSTH,SNFI,CSFI,GRV1,GRV2,XX1,YY1,ZZ1,
     $XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2,glog1,glog2,
     $grexp)
      IF(KOMP.EQ.2) GOTO 14
      call lum(1.d0,1.d0,0.d0,1.d0,n,n1,1,sbrd,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ,fr1,sm,0,vol,q,p,ff,d,
     $snth,7)
      GOTO 15
   14 call lum(1.d0,1.d0,0.d0,1.d0,n,n1,2,sbrd,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ,fr2,sm,0,vol,q,p,ff,d,
     $snth,7)
   15 CONTINUE
      IF(I.EQ.1) VOLS=VOL
      VOL2=VOLS
   17 VOL1=VOL
      rmean=(.238732414d0*vol)**ot
      rmsq=rmean**2
c
c  Here use a polar estimate for d(potential)/dr (absolute value).
c
      domdrabs=1.d0/rmsq+q*rmean/(d*d+rmsq)
      tolp=domdrabs*tolr
      IF(IFC.EQ.1) V=VOL
      IF(IFC.EQ.1) RETURN
      DPDV=DP/(VOL2-VOL1)
      DELP=(V-VOL1)*DPDV
      ABDELP=dabs(DELP)
      IF(ABDELP.GT.tolp) GOTO 16
      P=PS
      RETURN
      END

      SUBROUTINE ATMX(t,g,ifil,xintlog,xint)
      implicit real*8 (a-h,o-z)
c Version of January 23, 2004
      dimension abun(19),glog(11),grand(250800)
      dimension pl(10),yy(4),pha(4),tte(2),effwvl(25)
      dimension message(2,4)
      common/abung/abun,glog
      common/arrayleg/ grand,istart
      common /ramprange/ tlowtol,thightol,glowtol,ghightol
      common /atmmessages/ message,komp
      data effwvl/350.d0,412.d0,430.d0,546.d0,365.d0,440.d0,
     $550.d0,680.d0,870.d0,1220.d0,2145.d0,3380.d0,4900.d0,
     $9210.d0,650.d0,790.d0,230.d0,250.d0,270.d0,290.d0,
     $310.d0,330.d0,430.d0,520.d0,500.d0/
      tlog=dlog10(t)
      trec=1.d0/t
      tlow=3500.d0-tlowtol
      if(t.le.tlow) goto 66
      thigh=50000.d0+thightol
      fractol=thightol/50000.d0
      glow=0.d0-glowtol
      if(g.le.glow) goto 77
      ghigh=5.d0+ghightol
      if(g.ge.ghigh) goto 78
      tt=t
      gg=g
      if(g.ge.0.d0) goto 11
      gg=0.d0
      goto 12
  11  if(g.le.5.d0) goto 12
      gg=5.d0
  12  continue
ccccccccccccccccccccccccccccccccccccccccccccccccccccc
c The following is for 4-point interpolation in log g.
ccccccccccccccccccccccccccccccccccccccccccccccccccccc
      m=4
      ifreturn=0
      icase=istart+(ifil-1)*528
      call binnum(glog,11,g,j)
      k=min(max(j-(m-1)/2,1),12-m)
      if(g.le.0.d0) j=1
  10  continue
      ib=icase+(k-1)*48
      ib=ib-48
ccccccccccccccccccccccccccccccccccccccccccccccccccccc
      do 4 ii=1,m
      ib=ib+48
      do 719 ibin=1,4
      it=ib+(ibin-1)*12
      it1=it+1
      if(tt.le.grand(it1)) goto 720
  719 continue
      ibin=ibin-1
  720 continue
      tb=grand(it)
      if(tb.ne.0.d0) goto 55
      if(ibin.eq.4) ibin=ibin-1
      if(ibin.eq.1) ibin=ibin+1
      it=ib+(ibin-1)*12
      it1=it+1
      tb=grand(it)
  55  continue
      te=grand(it1)
      ibinsav=ibin
      thigh=te+fractol*te
      ibb=ib+1+(ibin-1)*12
      do 1 jj=1,10
      if(grand(ibb+jj).ne.0.d0) goto 1
      goto 2
   1  continue
   2  ma=jj-1
      pha(ii)=(tt-tb)/(te-tb)
      yy(ii)=0.d0
      call legendre(pha(ii),pl,ma)
      if(pha(ii).lt.0.d0) call legendre(0.d0,pl,ma)
      do 3 kk=1,ma
      kj=ibb+kk
   3  yy(ii)=yy(ii)+pl(kk)*grand(kj)
      if(pha(ii).ge.0.d0) goto 4
      tlow=tb-tlowtol
      call planckint(tlow,ifil,yylow,dum)
      if(t.ge.tlow) goto 424
      call planckint(t,ifil,yy(ii),dum)
      goto 4
  424 continue
      tlowmidlog=0.5d0*dlog10(tb*tlow)
      wvlmax=10.d0**(6.4624d0-tlowmidlog)
      if(effwvl(ifil).lt.wvlmax) goto 425
      tblog=dlog10(tb)
      tlowlog=dlog10(tlow)
      slope=(yy(ii)-yylow)/(tblog-tlowlog)
      yy(ii)=yylow+slope*(tlog-tlowlog)
      goto 4
  425 continue
      tbrec=1.d0/tb
      tlowrec=1.d0/tlow
      slope=(yy(ii)-yylow)/(tbrec-tlowrec)
      yy(ii)=yylow+slope*(trec-tlowrec)
   4  ibin=ibinsav
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Next, do a m-point Lagrange interpolation.
      xintlog=0.d0
      do 501 ii=1,m
      xnum=1.d0
      denom=1.d0
      nj=k+ii-1
      do 500 iij=1,m
      njj=k+iij-1
      if(ii.eq.iij) goto 500
      xnum=xnum*(gg-glog(njj))
      denom=denom*(glog(nj)-glog(njj))
 500  continue
      xintlog=xintlog+yy(ii)*xnum/denom
 501  continue
cccccccccccccccccccccccccccccccccccccccccccccccc
c  Check if a ramp function will be needed, or if we are
c  close to the border and need to interpolate between less
c  than 4 points.
ccccccccccccccccccccccccccccccccccccccccccccccccc
      if(g.lt.0.d0) goto 7
      if(g.gt.5.d0) goto 9
      if(t.lt.3500.d0) goto 99
      if(pha(1).le.1.d0) goto 99
      if(ifreturn.eq.1) goto 99
      if(j.eq.1) goto 5
      if(pha(3).gt.1.d0) goto 5
      k=k+1
      if(pha(2).gt.1.d0) goto 41
  42  continue
      if(k.gt.8) m=12-k
      ifreturn=1
      goto 10
  41  continue
      if(j.lt.10) goto 5
      k=k+1
      goto 42
ccccccccccccccccccccccccccccccccccccccccccccccccc
   5  continue
      ib=icase+(j-1)*48
      ib=ib-48
      do 61 kik=1,2
      ib=ib+48
      do 619 ibin=1,4
      it=ib+(ibin-1)*12
      it1=it+1
      if(tt.le.grand(it1)) goto 620
  619 continue
      ibin=ibin-1
  620 continue
      tb=grand(it)
      if(tb.ne.0.d0) goto 67
      if(ibin.eq.1) ibin=ibin+1
      if(ibin.eq.4) ibin=ibin-1
      it=ib+(ibin-1)*12
      it1=it+1
      tb=grand(it)
  67  continue
      te=grand(it1)
      tte(kik)=t
      if(t.gt.te) tte(kik)=te
      ibb=ib+1+(ibin-1)*12
      do 111 jj=1,10
      if(grand(ibb+jj).ne.0.d0) goto 111
      goto 22
 111  continue
  22  ma=jj-1
      pha(kik)=(tte(kik)-tb)/(te-tb)
      call legendre(pha(kik),pl,ma)
      yy(kik)=0.d0
      do 33 kk=1,ma
      kj=ibb+kk
  33  yy(kik)=yy(kik)+pl(kk)*grand(kj)
      ibin=ibinsav
  61  continue
      if(g.gt.5.d0) goto 43
      if(g.lt.0.d0) goto 47
      slope=(yy(2)-yy(1))*2.d0
      yy(1)=yy(2)+slope*(g-glog(j+1))
      slope=(tte(2)-tte(1))*2.d0
      te=tte(1)+slope*(g-glog(j))
      thigh=te*(1.d0+fractol)
      if(t.gt.thigh) goto 79
      call planckint(thigh,ifil,yyhigh,dum)
      thighmidlog=0.5d0*dlog10(te*thigh)
      wvlmax=10.d0**(6.4624d0-thighmidlog)
      if(effwvl(ifil).lt.wvlmax) goto 426
      thighlog=dlog10(thigh)
      telog=dlog10(te)
      slope=(yyhigh-yy(1))/(thighlog-telog)
      xintlog=yyhigh+slope*(tlog-thighlog)
      goto 99
  426 continue
      thighrec=1.d0/thigh
      terec=1.d0/te
      slope=(yyhigh-yy(1))/(thighrec-terec)
      xintlog=yyhigh+slope*(trec-thighrec)
      goto 99
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  43  yy(1)=yy(2)
      te=tte(2)
      call planckint(thigh,ifil,yyhigh,dum)
      thighmidlog=0.5d0*dlog10(te*thigh)
      wvlmax=10.d0**(6.4624d0-thighmidlog)
      if(effwvl(ifil).lt.wvlmax) goto 427
      thighlog=dlog10(thigh)
      telog=dlog10(te)
      slope=(yyhigh-yy(1))/(thighlog-telog)
      xintlog=yyhigh+slope*(tlog-thighlog)
      goto 44
  427 continue
      thighrec=1.d0/thigh
      terec=1.d0/te
      slope=(yyhigh-yy(1))/(thighrec-terec)
      xintlog=yyhigh+slope*(trec-thighrec)
      goto 44
  47  continue
      te=tte(1)
      call planckint(thigh,ifil,yyhigh,dum)
      thighmidlog=0.5d0*dlog10(te*thigh)
      wvlmax=10.d0**(6.4624d0-thighmidlog)
      if(effwvl(ifil).lt.wvlmax) goto 428
      thighlog=dlog10(thigh)
      telog=dlog10(te)
      slope=(yyhigh-yy(1))/(thighlog-telog)
      xintlog=yyhigh+slope*(tlog-thighlog)
      goto 63
  428 continue
      thighrec=1.d0/thigh
      terec=1.d0/te
      slope=(yyhigh-yy(1))/(thighrec-terec)
      xintlog=yyhigh+slope*(trec-thighrec)
      goto 63
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
   7  continue
      thigh=6000.d0*(1.d0+fractol)
      if(t.gt.thigh) goto 79
      if(pha(1).le.1.d0) goto 63
      goto 5
  63  continue
      call planckint(t,ifil,yylow,dum)
      slope=(yylow-xintlog)/glow
      xintlog=yylow+slope*(g-glow)
      goto 99
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
   9  continue
      thigh=50000.d0*(1.d0+fractol)
      if(t.gt.thigh) goto 79
      if(t.gt.50000.d0) goto 52
  44  continue
      call planckint(t,ifil,yyhigh,dum)
      slope=(yyhigh-xintlog)/(ghigh-5.d0)
      xintlog=yyhigh+slope*(g-ghigh)
      goto 99
  52  continue
      j=10
      goto 5
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  66  continue
      message(komp,4)=1
      call planckint(t,ifil,xintlog,xint)
      return
  77  continue
      message(komp,1)=1
      call planckint(t,ifil,xintlog,xint)
      return
  78  continue
      message(komp,2)=1
      call planckint(t,ifil,xintlog,xint)
      return
  79  continue
      message(komp,3)=1
      call planckint(t,ifil,xintlog,xint)
      return
  99  continue
      xint=10.d0**xintlog
      return
      end
      SUBROUTINE FOURLS(th,ro,nobs,nth,aa,bb)
      implicit real*8(a-h,o-z)
c   version of September 14, 1998
c
c    Input integer nth is the largest Fourier term fitted (e.g.
c       for nth=6, terms up to sine & cosine of 6 theta are
c       evaluated).
c    This subroutine can handle nth only up to 6. Additional
c      programming is needed for larger values.
c
      dimension aa(*),bb(*),th(*),ro(*),obs(5000),ll(14),mm(14),
     $cn(196),cl(14),out(14)
      mpl=nth+1
      ml=mpl+nth
      jjmax=ml*ml
      nobsml=nobs*ml
      nobmpl=nobs*mpl
      do 90 i=1,nobs
      obs(i)=1.d0
      iz=nobsml+i
      obs(iz)=ro(i)
      if(nth.eq.0) goto 90
      ic=i+nobs
      is=i+nobmpl
      sint=dsin(th(i))
      cost=dcos(th(i))
      obs(ic)=cost
      obs(is)=sint
      if(nth.eq.1) goto 90
      ic=ic+nobs
      is=is+nobs
      sncs=sint*cost
      cs2=cost*cost
      obs(ic)=cs2+cs2-1.d0
      obs(is)=sncs+sncs
      if(nth.eq.2) goto 90
      ic=ic+nobs
      is=is+nobs
      sn3=sint*sint*sint
      cs3=cs2*cost
      obs(ic)=4.d0*cs3-3.d0*cost
      obs(is)=3.d0*sint-4.d0*sn3
      if(nth.eq.3) goto 90
      ic=ic+nobs
      is=is+nobs
      cs4=cs2*cs2
      obs(ic)=8.d0*(cs4-cs2)+1.d0
      obs(is)=4.d0*(2.d0*cs3*sint-sncs)
      if(nth.eq.4) goto 90
      ic=ic+nobs
      is=is+nobs
      cs5=cs3*cs2
      obs(ic)=16.d0*cs5-20.d0*cs3+5.d0*cost
      obs(is)=16.d0*sn3*sint*sint-20.d0*sn3+5.d0*sint
      if(nth.eq.5) goto 90
      ic=ic+nobs
      is=is+nobs
      obs(ic)=32.d0*cs3*cs3-48.d0*cs4+18.d0*cs2-1.d0
      obs(is)=32.d0*sint*(cs5-cs3)+6.d0*sncs
   90 continue
      do 20 jj=1,jjmax
   20 cn(jj)=0.d0
      do 21 j=1,ml
   21 cl(j)=0.d0
      do 24 nob=1,nobs
      iii=nob+nobsml
      do 23 k=1,ml
      do 23 i=1,ml
      ii=nob+nobs*(i-1)
      kk=nob+nobs*(k-1)
      j=i+(k-1)*ml
   23 cn(j)=cn(j)+obs(ii)*obs(kk)
      do 24 i=1,ml
      ii=nob+nobs*(i-1)
   24 cl(i)=cl(i)+obs(iii)*obs(ii)
      call dminv(cn,ml,d,ll,mm)
      call dgmprd(cn,cl,out,ml,ml,1)
      do 51 i=2,mpl
      aa(i)=out(i)
      ipl=i+nth
   51 bb(i)=out(ipl)
      aa(1)=out(1)
      bb(1)=0.d0
      return
      end
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
      SUBROUTINE RING(Q,OM,KOMP,L,FR,HLD,R1,RL)
c   Version of September 14, 1998
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION RAD(100),THET(100),AA(3),BB(3),FI(150),THA(150),FR(*),
     $HLD(*)
      IX=0
      LR=L+1
      DO 92 I=1,LR
      THA(I)=0.D0
   92 FI(I)=-.1D0
      OMEGA=OM
      K=3
      EL=dfloat(L)
      DEL=2.D0/EL
      CALL ELLONE(1.d0,1.d0,Q,xlsv,OM1,XL2,OM2)
      CALL NEKMIN(Q,OM,xlsv,Z)
      XL=xlsv
      QQ=Q
      XLSQ=XL*XL
      IF(Q.GT.1.D0) QQ=1.D0/Q
      RMAX=DEXP(.345D0*DLOG(QQ)-1.125D0)
      R=RMAX*(OM1-OMEGA)/(OM1-OM2)
      DO 22 IT=1,L
      EYT=dfloat(IT)
      TH=EYT*1.570796326794897d0/EL
      COSQ=DCOS(TH)**2
      DELR=0.D0
   14 R=DABS(R+DELR)
      RSQ=R*R
      X2R2=XLSQ+RSQ
      RX2R2=DSQRT(X2R2)
      XM2R2=(XL-1.D0)**2+RSQ
      RXM2R2=DSQRT(XM2R2)
      OM=1.D0/RX2R2+Q*(1.D0/RXM2R2-XL)+.5D0*(Q+1.D0)*(XLSQ+RSQ*COSQ)
      DOMDR=-R/(X2R2*RX2R2)-Q*R/(XM2R2*RXM2R2)+(Q+1.D0)*COSQ*R
      DELR=(OMEGA-OM)/DOMDR
      ABDELR=DABS(DELR)
      IF(ABDELR.GT..00001D0) GOTO 14
      RAD(IT)=R
   22 THET(IT)=TH*4.D0
      R1=RAD(1)
      RL=RAD(L)
      R90SQ=RL*RL
      DO 18 IJ=1,L
      EYJ=IJ
      RAD(IJ)=RAD(IJ)-(RL-R1)*(EYJ-1.D0)/EL
   18 CONTINUE
      DO 65 N=1,K
      AA(N)=0.D0
   65 BB(N)=0.D0
      DO 29 J=1,L
      DO 29 N=1,K
      EN=N-1
      ENTHET=EN*THET(J)
      AA(N)=AA(N)+RAD(J)*DCOS(ENTHET)*DEL
   29 BB(N)=BB(N)+RAD(J)*DSIN(ENTHET)*DEL
      AA(1)=AA(1)*.5D0
      IF(KOMP.EQ.2) XL=1.d0-xlsv
      XLSQ=XL*XL
      DIS=RL/XL-.0005D0
      DO 42 IR=1,L
      LL=IR-1
      EY=dfloat(L+1-IR)
      THA(IR)=1.570796326794897D0*EY/EL
      IF(THA(IR).LT.1.570796326794897D0) GOTO 82
      COT=0.D0
      GOTO 83
   82 COT=1.D0/DTAN(THA(IR))
   83 IF(COT.GE.DIS) GOTO 50
      COSSQ=DCOS(THA(IR))**2
      A0=AA(1)
      A1=AA(2)
      A2=AA(3)
      B1=BB(2)
      B2=BB(3)
      DELSIN=0.D0
      KNTR=0
      SINTH=DSQRT(COSSQ*(XLSQ+R90SQ)/R90SQ)
   88 SINTH=SINTH+DELSIN
      KNTR=KNTR+1
      IF(SINTH.GT.1.D0) SINTH=1.D0/SINTH
      CSQ=1.D0-SINTH*SINTH
      COSTH=DSQRT(CSQ)
      SINSQ=SINTH*SINTH
      SIN4=8.D0*COSTH**3*SINTH-4.D0*COSTH*SINTH
      COS4=8.D0*CSQ*(CSQ-1.D0)+1.D0
      C4SQ=COS4*COS4
      SINCOS=SIN4*COS4
      RRR=A0+A1*COS4+A2*(C4SQ+C4SQ-1.D0)+B1*SIN4+(B2+B2)*SINCOS
      ARC=dasin(SINTH)
      RR=RRR+(RL-R1)*(2.D0*ARC/3.141592653589793D0-1.D0/EL)
      IF(KNTR.GT.30) GOTO 42
      P=RR*SINTH
      DRDSIN=-A1*SINTH/COSTH-4.D0*A2*SINTH+B1-(B2+B2)*SINSQ/COSTH+(B2+B2
     $)*COSTH+(RL+RL-R1-R1)/(3.141592653589793D0*COSTH)
      DPDSIN=RR+SINTH*DRDSIN
      F=P*P/COSSQ-RR*RR-XLSQ
      DFDSIN=(P+P)*DPDSIN/COSSQ-(RR+RR)*DRDSIN
      DELSIN=-F/DFDSIN
      ABDEL=DABS(DELSIN)
      IF(ABDEL.GT..00001D0)  GOTO 88
   42 FI(IR)=DATAN(RR*COSTH/XL)
   50 LL1=LL+1
      DELTH=1.570796326794897D0/EL
      DO 75 I=1,L
      EY=dfloat(L+1-I)-.5d0
      THE=1.570796326794897D0*EY/EL
      SNTH=DSIN(THE)
      EM=dsin(THE)*EL*1.3d0
      MM=EM+1.d0
      XM=dfloat(MM)
      DELFI=3.141592653589793D0/XM
      HDELFI=1.570796326794897D0/XM
      DO 75 J=1,MM
      IX=IX+1
      IF(I.LE.LL1) GOTO 43
      HLD(IX)=1.d0
      GOTO 75
   43 XJ=MM+1-J
      FE=3.141592653589793D0*(XJ-.5D0)/XM
      PH2=FE+HDELFI
      PHB=PH2
      IF(FI(I).GT.(FE-HDELFI)) GOTO 51
      HLD(IX)=1.d0
      GOTO 75
   51 IPL=I+1
      IF(FI(IPL).GT.0.D0) GOTO 66
      RR=A0+A1-A2+(RL-R1)*(1.D0-1.D0/EL)
      PH1=DELFI*(XJ-1.D0)
      TH1=DATAN(XL/RR)
      GOTO 56
   66 IF(FI(IPL).LT.(FE+HDELFI)) GOTO 52
      HLD(IX)=0.d0
      GOTO 75
   52 IF(FI(IPL).LT.(FE-HDELFI)) GOTO 53
      PH1=FI(IPL)
      TH1=THA(IPL)
      GOTO 56
   53 DELSIN=0.D0
      SINTH=DSQRT(COSSQ*(XLSQ+R90SQ)/R90SQ)
      TANFE=DTAN(FE-HDELFI)
   77 SINTH=SINTH+DELSIN
      IF(SINTH.GT.1.D0) SINTH=1.D0/SINTH
      SINSQ=SINTH*SINTH
      CSQ=1.D0-SINSQ
      COSTH=DSQRT(CSQ)
      SIN4=8.D0*COSTH**3*SINTH-4.D0*COSTH*SINTH
      COS4=8.D0*CSQ*(CSQ-1.D0)+1.D0
      C4SQ=COS4*COS4
      SINCOS=SIN4*COS4
      RRR=A0+A1*COS4+A2*(C4SQ+C4SQ-1.D0)+B1*SIN4+(B2+B2)*SINCOS
      ARC=dasin(SINTH)
      RR=RRR+(RL-R1)*(2.D0*ARC/3.141592653589793D0-1.D0/EL)
      DRDSIN=-A1*SINTH/COSTH-4.D0*A2*SINTH+B1-(B2+B2)*SINSQ/COSTH+(B2+B2
     $)*COSTH+(RL+RL-R1-R1)/(3.141592653589793D0*COSTH)
      F=RR*COSTH-XL*TANFE
      DFDSIN=COSTH*DRDSIN-RR*SINTH/COSTH
      DELSIN=-F/DFDSIN
      ABDEL=DABS(DELSIN)
      IF(ABDEL.GT..00001D0)  GOTO 77
      PH1=FE-HDELFI
      TH1=DATAN(XL/(RR*SINTH*DCOS(PH1)))
   56 IF(FI(I).GT.(FE+HDELFI)) GOTO 57
      PHB=FI(I)
      TH2=THA(I)
      GOTO 60
   57 DELSIN=0.D0
      SINTH=DSQRT(COSSQ*(XLSQ+R90SQ)/R90SQ)
      TANFE=DTAN(FE+HDELFI)
   78 SINTH=SINTH+DELSIN
      IF(SINTH.GT.1.D0) SINTH=1.D0/SINTH
      SINSQ=SINTH*SINTH
      CSQ=1.D0-SINSQ
      COSTH=DSQRT(CSQ)
      SIN4=8.D0*COSTH**3*SINTH-4.D0*COSTH*SINTH
      COS4=8.D0*CSQ*(CSQ-1.D0)+1.D0
      C4SQ=COS4*COS4
      SINCOS=SIN4*COS4
      RRR=A0+A1*COS4+A2*(C4SQ+C4SQ-1.D0)+B1*SIN4+(B2+B2)*SINCOS
      ARC=dasin(SINTH)
      RR=RRR+(RL-R1)*(2.D0*ARC/3.141592653589793D0-1.D0/EL)
      DRDSIN=-A1*SINTH/COSTH-4.D0*A2*SINTH+B1-(B2+B2)*SINSQ/COSTH+(B2+B2
     $)*COSTH+(RL+RL-R1-R1)/(3.141592653589793D0*COSTH)
      F=RR*COSTH-XL*TANFE
      DFDSIN=COSTH*DRDSIN-RR*SINTH/COSTH
      DELSIN=-F/DFDSIN
      ABDEL=DABS(DELSIN)
      IF(ABDEL.GT..00001D0)  GOTO 78
      TH2=DATAN(XL/(RR*SINTH*DCOS(PH2)))
   60 CTHT=DCOS(THA(IPL))
      CTH1=DCOS(TH1)
      CTH2=DCOS(TH2)
      STH1=DSIN(TH1)
      STH2=DSIN(TH2)
      DTH=TH2-TH1
      DCTH=CTH1-CTH2
      OMDP=PH2*DCTH-.5D0*(PH1*STH1+PHB*STH2)*DTH
      OMP=DELFI*(CTHT-CTH1)
      OMN=OMP+OMDP
      HLD(IX)=OMN/(DELTH*DELFI*SNTH)
   75 CONTINUE
      DO 94 JB=1,IX
      JA=IX+1-JB
   94 FR(JB)=HLD(JA)
      RETURN
      END

      SUBROUTINE MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2
     $,HLD,RM,POTH,POTC,GR1,GR2,ALB1,ALB2,N1,N2,F1,F2,MOD,XINCL,THE,
     $MODE,SNTH,CSTH,SNFI,CSFI,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1
     $,GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
c    Version of December 18, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ
     $(*),MMSAVE(*),FR1(*),FR2(*),HLD(*),GRV1(*),GRV2(*),XX1(*),YY1(*),
     $ZZ1(*),XX2(*),YY2(*),ZZ2(*),GLUMP1(*),GLUMP2(*),CSBT1(*),CSBT2(*)
     $,GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      DIMENSION DRR(4),RES(2),ANS(2),LX(2),MX(2)
      DIMENSION SNTH(*),CSTH(*),SNFI(*),CSFI(*)
      common /kfac/ kff1,kff2,kfo1,kfo2
      common /setest/ sefac
      common /ardot/ dperdt,hjd,hjd0,perr
      COMMON /FLVAR/ PSHIFT,DP,EF,EFC,ECOS,perr0,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      COMMON /ECCEN/ E,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      COMMON /INVAR/ KH,IPBDUM,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,
     $ifsmv2,icor1,icor2,ld,ncl,jdphs,ipc
   95 FORMAT(' WARNING: ALTHOUGH COMPONENT 2 DOES NOT EXCEED ITS LIMITIN
     $G LOBE AT THE END OF ECLIPSE, IT DOES EXCEED THE LOBE AT PERIASTRO
     $N')
   99 FORMAT(' SPECIFIED ECLIPSE DURATION INCONSISTENT WITH OTHER PARAME
     $TERS')
      perr=perr0+dperdt*(hjd-hjd0)
      DP=1.d0-E
      MOD=(MODE-2)**2
      IF(MODE.EQ.1) GR2=GR1
      IF(MODE.EQ.1) ALB2=ALB1
      IF(MOD.EQ.1) POTC=POTH
      MD4=(MODE-5)**2
      MD5=(2*MODE-11)**2
      call ellone(f1,dp,rm,xl1,po1cr,xl2,omo1)
      sefac=.8712d0
      doc=(po1cr-poth)/(po1cr-omo1)
      if(doc.gt.0.d0) sefac=.201d0*doc*doc-.386d0*doc+.8712d0
      RMR=1.d0/RM
      CALL ELLONE(F2,DP,RMR,XL1,po2c,XL2,omo2)
      po2cr=rm*po2c+(1.d0-rm)*.5d0
      if(md4.eq.1) poth=po1cr
      if(md5.eq.1) potc=po2cr
      kff1=0
      kff2=0
      if(poth.lt.po1cr) kff1=1
      if(potc.lt.po2cr) kff2=1
      kfo1=0
      kfo2=0
      if(e.ne.0.d0) goto 100
      if(f1.ne.1.d0) goto 105
      if(poth.lt.omo1) kfo1=1
  105 if(f2.ne.1.d0) goto 100
      if(potc.lt.omo1) kfo2=1
  100 continue
c     The constant below is \pi/180:
      SINI=dsin(.017453292519943d0*XINCL)
c     The constant below is 2\pi RSun/86400:
      VF=50.61455d0/PERIOD
      VFAC=VF*A
      VGAM=VGA*VUNIT/VFAC
      VFVU=VFAC
      IFC=2
      IF(E.NE.0.d0) GOTO 60
      perr=1.570796326794897d0
      IFC=1
   60 CONTINUE
      TRC=1.570796326794897d0-perr
   39 if(TRC.LT.0.d0) TRC=TRC+6.283185307179586d0
      if(trc.lt.0.d0) goto 39
   40 if(trc.ge.6.283185307179586d0) trc=trc-6.283185307179586d0
      if(trc.ge.6.283185307179586d0) goto 40
      HTRC=.5d0*TRC
      IF(dabs(1.570796326794897d0-HTRC).LT.7.d-6) GOTO 101
      IF(dabs(4.712388980384690d0-HTRC).LT.7.d-6) GOTO 101
      ECAN=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(HTRC))
      GOTO 103
  101 ECAN=3.141592653589793d0
  103 XMC=ECAN-E*dsin(ECAN)
      IF(XMC.LT.0.d0) XMC=XMC+6.283185307179586d0
      PHPER=1.d0-XMC/6.283185307179586d0
      call conjph(e,perr,pshift,trsc,tric,econsc,econic,xmsc,xmic,
     $pconsc,pconic)
   38 if(pconsc.ge.1.d0) pconsc=pconsc-1.d0
      if(pconsc.ge.1.d0) goto 38
   41 if(pconsc.lt.0.d0) pconsc=pconsc+1.d0
      if(pconsc.lt.0.d0) goto 41
   68 if(pconic.ge.1.d0) pconic=pconic-1.d0
      if(pconic.ge.1.d0) goto 68
   71 if(pconic.lt.0.d0) pconic=pconic+1.d0
      if(pconic.lt.0.d0) goto 71
      PHPERI=PHPER+pconsc
      EF=1.d0-E*E
      EFC=dsqrt(EF)
      ECOS=E*dcos(perr)
c     Continue only in case of X-ray binaries:
      IF(MODE.NE.-1) RETURN
      if(kh.eq.17) goto 241
      if((kh-12)**2.eq.1) goto 241
      if((kh-12)**2.eq.4) goto 241
      IF((KH-11)**2.LE.1) GOTO 241
      IF((2*KH-41)**2.EQ.81) GOTO 241
      RETURN
  241 CONTINUE
      EFCC=dsqrt((1.d0-E)/(1.d0+E))
      THER=THE*6.283185307179586d0
      DELTR=.001d0
      DTR1=0.d0
      DTR2=0.d0
      VOLTOL=5.d-6
      DXMTOL=5.d-6
      TR0=1.570796326794897d0-perr
      HTR0=.5d0*TR0
      IF((1.570796326794897d0-dabs(HTR0)).LT.7.d-6) GOTO 201
      IF((4.712388980384690d0-dabs(HTR0)).LT.7.d-6) GOTO 201
      ECAN0=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(HTR0))
      GOTO 203
  201 ECAN0=3.141592653589793d0
  203 XM0=ECAN0-E*dsin(ECAN0)
      XM1=XM0-THER*(1.d0-.2d0*E)
      XM2=XM0+THER*(1.d0-.2d0*E)
      CALL KEPLER(XM1,E,DUM,TRR1)
      CALL KEPLER(XM2,E,DUM,TRR2)
  160 TRR1=TRR1+DTR1
      TRR2=TRR2+DTR2
      DO 161 IB=1,3
      TR1=TRR1
      TR2=TRR2
      IF(IB.EQ.2) TR1=TRR1+DELTR
      IF(IB.EQ.3) TR2=TRR2+DELTR
      IF(TR1.GT.TR0) TR0=TR0+6.283185307179586d0
      IF(TR0.GT.TR2) TR2=TR2+6.283185307179586d0
      DS1=EF/(1.d0+E*dcos(TR1))
      DS2=EF/(1.d0+E*dcos(TR2))
      TRE1=(TR0-TR1)/6.283185307179586d0
      TRE2=(TR2-TR0)/6.283185307179586d0
      CALL DURA(F2,XINCL,RM,DS1,TRE1,POTR,RA)
      CALL VOLUME(VS1,RM,POTR,DS1,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ
     $,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,
     $GMAG2,glog1,glog2,GR1,1)
      CALL DURA(F2,XINCL,RM,DS2,TRE2,POTR,RA)
      CALL VOLUME(VS2,RM,POTR,DS2,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ
     $,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,
     $GMAG2,glog1,glog2,GR2,1)
      IF(IB.NE.1) GOTO 185
      ECAN1=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(.5d0*TR1))
      ECAN2=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(.5d0*TR2))
      POTC=POTR
      DTHE=DS2
      DVOL=VS2-VS1
      XM1=ECAN1-E*dsin(ECAN1)
      XM2=ECAN2-E*dsin(ECAN2)
      IF(XM1.LT.0.d0) XM1=XM1+6.283185307179586d0
      IF(XM2.LT.0.d0) XM2=XM2+6.283185307179586d0
      DXM=XM2-XM1-2.d0*THER
      DDMDN1=-EFCC*(1.d0-E*dcos(ECAN1))*dcos(.5d0*ECAN1)**2/
     $dcos(.5d0*tr1)**2
      DDMDN2=EFCC*(1.d0-E*dcos(ECAN2))*dcos(.5d0*ECAN2)**2/
     $dcos(.5d0*tr2)**2
  185 CONTINUE
      IF(IB.NE.2) GOTO 162
      DRR(1)=(VS2-VS1-DVOL)/DELTR
      DRR(2)=DDMDN1
  162 CONTINUE
      IF(IB.NE.3) GOTO 161
      DRR(3)=(VS2-VS1-DVOL)/DELTR
      DRR(4)=DDMDN2
  161 CONTINUE
      RES(1)=-DVOL
      RES(2)=-DXM
      CALL DMINV(DRR,2,DUMM,LX,MX)
      CALL DGMPRD(DRR,RES,ANS,2,2,1)
      DTR1=ANS(1)
      DTR2=ANS(2)
      IF(dabs(DTR1).GT.VOLTOL) GOTO 160
      IF(dabs(DTR2).GT.DXMTOL) GOTO 160
      POTH=9999.99d0
      RMR=1.d0/RM
      CALL ELLONE(F2,DTHE,RMR,XLA,OM1,XL2,OM2)
      OM1=RM*OM1+(1.d0-RM)*.5d0
      IF(POTC.LT.OM1) GOTO 22
      IF(RA.LE.XLA) GOTO 28
   22 WRITE(6,99)
      RETURN
   28 CONTINUE
      IF(E.NE.0.d0) CALL VOLUME(VTHE,RM,POTC,DTHE,F2,N2,N1,2,RV,GRX,
     $GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,
     $SUMMD,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,
     $GLUMP2,GMAG1,GMAG2,glog1,glog2,GR2,1)
      IF(E.NE.0.d0) CALL VOLUME(VTHE,RM,POTC,DP,F2,N2,N1,2,RV,GRX,
     $GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,
     $SUMMD,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,
     $GLUMP2,GMAG1,GMAG2,glog1,glog2,GR2,2)
      CALL ELLONE(F2,DP,RMR,XLD,OMP,XL2,OM2)
      OMP=RM*OMP+(1.d0-RM)*.5d0
      IF(POTC.LT.OMP) WRITE(6,95)
      RETURN
      END

      SUBROUTINE KEPLER(XM,EC,ECAN,TR)
c  Version of October 6, 1995
      IMPLICIT REAL*8(A-H,O-Z)
      TOL=1.d-8
      DLECAN=0.D0
      ECAN=XM
   18 ECAN=ECAN+DLECAN
      XMC=ECAN-EC*DSIN(ECAN)
      DEDM=1.D0/(1.D0-EC*DCOS(ECAN))
      DLECAN=(XM-XMC)*DEDM
      ABDLEC=DABS(DLECAN)
      IF(ABDLEC.GT.TOL) GOTO 18
      TR=2.D0*DATAN(DSQRT((1.D0+EC)/(1.D0-EC))*DTAN(.5D0*ECAN))
      IF(TR.LT.0.) TR=TR+6.2831853071795865d0
      RETURN
      END

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

      SUBROUTINE LUM (xlum,x,y,tpoll,n,n1,komp,sbr,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ,fr,sm,ifat,vol,rm,om,
     $f,d,snth,iband)
c   Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      dimension rv(*),rvq(*),mmsave(*),fr(*),snth(*),glump1(*),
     $glump2(*),glog1(*),glog2(*),grv1(*),grv2(*)
      dimension message(2,4)
      common /atmmessages/ message,kompcom
      common /radi/ R1H,RLH,R1C,RLC
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm,
     $is1dm,is2dm,ic1dm,ic2dm,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      kompcom=komp
      TPOLE=10000.d0*TPOLL
      KR=0
      cmp=dfloat(komp-1)
      cmpp=dfloat(2-komp)
      gplog=cmpp*gplog1+cmp*gplog2
      IF(ifat.eq.0) call planckint(tpole,iband,pollog,polin)
      IF(IFAT.NE.0) call atmx(TPOLE,gplog,iband,pollog,polin)
      EN=dfloat(N)
      DELTH=1.570796326794897d0/EN
      SUM=0.d0
      SUMM=0.d0
      SM=0.d0
      VOL=0.d0
      DO 36 I=1,N
      IPN1=I+N1*(komp-1)
      SINTH=SNTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      XM=dfloat(MM)
      DELFI=3.141592653589793d0/XM
      DFST=DELFI*SINTH
      SUMJ=0.d0
      SUMMJ=0.d0
      SMJ=0.d0
      VOLJ=0.d0
      DO 26 J=1,MM
      IP=(komp-1)*(N1+1)+I
      IX=MMSAVE(IP)+J
      IF(komp.EQ.1) GOTO 39
      IF(RVQ(IX).EQ.-1.d0) GOTO 25
      R=RVQ(IX)
      GOTO 49
   39 IF(RV(IX).EQ.-1.d0) GOTO 25
      R=RV(IX)
   49 grav=cmpp*grv1(ix)+cmp*grv2(ix)
      TLOCAL=TPOLE*dsqrt(dsqrt(GRAV))
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      if(ifat.eq.0) call planckint(tlocal,iband,xinlog,xint)
      IF(IFAT.NE.0) CALL atmx(TLOCAL,glogg,iband,xinlog,xint)
      GRAVM=xint/polin
      di=cmpp*glump1(ix)+cmp*glump2(ix)
      DIF=DI*GRAVM
      DIFF=DI*GRAV
      SMJ=SMJ+DI
      SUMJ=SUMJ+DIF
      SUMMJ=SUMMJ+DIFF
      VOLJ=VOLJ+R*R*R*FR(IX)
      GOTO 26
   25 KR=1
   26 CONTINUE
      SMJ=SMJ*DELFI
      SUMJ=SUMJ*DELFI
      SUMMJ=SUMMJ*DELFI
      SM=SM+SMJ
      SUMM=SUMM+SUMMJ
      VOL=VOL+VOLJ*DFST
   36 SUM=SUM+SUMJ
      darkin=3.141592653589793d0*(1.d0-x/3.d0)
      if(ld.eq.2) darkin=darkin+.6981317d0*y
      if(ld.eq.3) darkin=darkin-.6283185d0*y
      SBR=.25d0*XLUM/(SUM*DELTH*DARKIN)
      SM=SM*DELTH*4.d0
      SUMM=SUMM*DELTH*4.d0
      VOL=VOL*1.3333333333333d0*DELTH
      IF(KR.EQ.0) RETURN
      CALL ELLONE(F,D,RM,XL1,OMD,XLD,omdum)
      CALL NEKMIN(RM,OM,XL1,ZD)
      IF(komp.EQ.2) XL1=D-XL1
      R1=cmpp*R1H+cmp*R1C
      RL=cmpp*RLH+cmp*RLC
      VOL=VOL+1.047198d0*XL1*R1*RL
      RETURN
      END

      SUBROUTINE LUMP(GRX,GRY,GRZ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,
     $MMSAVE,ALB,TPOLL,SBR,N1,N2,KOMP,IFAT,fr,snth,
     $TLD,GLUMP1,GLUMP2,XX1,XX2,YY1,YY2,ZZ1,ZZ2,xbol,ybol
     $,GRV1,GRV2,SBR1B,SBR2B,RF,RFO,GMAG1,GMAG2,glog1,glog2,DINT,iband)
c   Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION GRX(*),GRY(*),GRZ(*),GRXQ(*),GRYQ(*),grzq(*),
     $SLUMP1(*),SLUMP2(*),MMSAVE(*),FR(*),SNTH(*),
     $TLD(*),GLUMP1(*),GLUMP2(*),XX1(*),XX2(*),YY1(*)
     $,YY2(*),ZZ1(*),ZZ2(*),GRV1(*),GRV2(*),RF(*),RFO(*),
     $GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      dimension message(2,4)
      common /atmmessages/ message,kompcom
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm
     $,ifs1dm,ifs2dm,icr1dm,icr2dm,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      kompcom=komp
      IQ=(KOMP-1)*(N1+1)
      IS=(KOMP-1)*MMSAVE(IQ)
      PI=3.141592653589793d0
      PIH=.5d0*PI
      TPOLE=10000.d0*TPOLL
      cmp=dfloat(komp-1)
      cmpp=dfloat(2-komp)
      gplog=cmpp*gplog1+cmp*gplog2
      if(ifat.eq.0) call planckint(tpole,iband,pollog,pint)
      IF(IFAT.NE.0) CALL atmx(tpole,gplog,iband,pollog,pint)
      COMPP=dfloat(2*KOMP-3)
      COMP=-COMPP
      N=(2-KOMP)*N1+(KOMP-1)*N2
      NO=(2-KOMP)*N2+(KOMP-1)*N1
      NOD=2*NO
      EN=dfloat(N)
      ENO=dfloat(NO)
      DELTHO=PIH/ENO
      CNST=ALB*DELTHO*SBR2B/(DINT*SBR1B)
      IF(KOMP.EQ.2) CNST=ALB*DELTHO*SBR1B/(DINT*SBR2B)
      DO 191 I=1,N
      IPN1=I+N1*(KOMP-1)
      SINTH=SNTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      IP=(KOMP-1)*(N1+1)+I
      IY=MMSAVE(IP)
      DO 193 J=1,MM
      IX=IY+J
      SUM=0.d0
      IF(FR(IX).EQ.0.d0) GOTO 193
      DO 190 IOTH=1,NOD
      IOTHS=IOTH
      IF(IOTH.GT.NO) IOTHS=NOD-IOTH+1
      IPNO=IOTHS+N1*(2-KOMP)
      SINTHO=SNTH(IPNO)
      EMO=SINTHO*ENO*1.3d0
      MMO=EMO+1.d0
      MMOD=2*MMO
      IPO=(2-KOMP)*(N1+1)+IOTHS
      IYO=MMSAVE(IPO)
      XMO=MMO
      DELFIO=PI/XMO
      DO 190 JOFI=1,MMOD
      JOFU=JOFI
      IF(JOFI.GT.MMO) JOFU=MMOD-JOFI+1
      IXO=IYO+JOFU
      IX1=IX
      IX2=IXO
      IF(KOMP.EQ.1) GOTO 200
      IF(GLUMP1(IXO).EQ.0.d0) GOTO 184
      IX1=IXO
      IX2=IX
      GOTO 201
  200 CONTINUE
      IF(GLUMP2(IXO).EQ.0.d0) GOTO 179
  201 RTL1=1.d0
      RTL2=1.d0
      UPD1=1.d0
      UPD2=1.d0
      IF(KOMP.EQ.2) GOTO 22
      IF(JOFI.GT.MMO) RTL2=-1.d0
      IF(IOTH.GT.NO) UPD2=-1.d0
      GOTO 23
   22 IF(JOFI.GT.MMO) RTL1=-1.d0
      IF(IOTH.GT.NO) UPD1=-1.d0
   23 CONTINUE
      GX2=GRXQ(IX2)
      GY2=GRYQ(IX2)*RTL2
      GZ2=GRZQ(IX2)*UPD2
      X1C=XX1(IX1)
      X2C=XX2(IX2)
      Y1C=YY1(IX1)*RTL1
      Y2C=YY2(IX2)*RTL2
      Z1C=ZZ1(IX1)*UPD1
      Z2C=ZZ2(IX2)*UPD2
      DX=(X2C-X1C)*COMP
      DY=(Y2C-Y1C)*COMP
      DZ=(Z2C-Z1C)*COMP
      DLRSQ=DX*DX+DY*DY+DZ*DZ
      CSNUM2=(DX*GX2+DY*GY2+DZ*GZ2)*COMPP
      IF(CSNUM2.GE.0.d0) GOTO 190
      GX1=GRX(IX1)
      GY1=GRY(IX1)*RTL1
      GZ1=GRZ(IX1)*UPD1
      CSNUM1=(DX*GX1+DY*GY1+DZ*GZ1)*COMP
      IF(CSNUM1.GE.0.d0) GOTO 190
      DMAG=dsqrt(DLRSQ)
      CSGM1=-CSNUM1/(DMAG*GMAG1(IX1))
      CSGM2=-CSNUM2/(DMAG*GMAG2(IX2))
      IF(KOMP.EQ.2) GOTO 181
      DGAM2=1.d0-XBOL+XBOL*CSGM2
      if(ld.ne.2) goto 179
      if(csgm2.eq.0.d0) goto 179
      dgam2=dgam2-ybol*csgm2*dlog(csgm2)
      goto 147
  179 continue
      if(ld.eq.3) dgam2=dgam2-ybol*(1.d0-dsqrt(csgm2))
  147 if(dgam2.lt.0.d0) dgam2=0.d0
      DSUM=GRV2(IXO)*GLUMP2(IXO)*RFO(IXO)*CSGM1*CSGM2*DGAM2/DLRSQ
      GOTO 182
  181 DGAM1=1.d0-XBOL+XBOL*CSGM1
      if(ld.ne.2) goto 184
      if(csgm1.eq.0.d0) goto 184
      dgam1=dgam1-ybol*csgm1*dlog(csgm1)
      goto 148
  184 continue
      if(ld.eq.3) dgam1=dgam1-ybol*(1.d0-dsqrt(csgm1))
  148 if(dgam1.lt.0.d0) dgam1=0.d0
      DSUM=GRV1(IXO)*GLUMP1(IXO)*RFO(IXO)*CSGM2*CSGM1*DGAM1/DLRSQ
  182 CONTINUE
      SUM=SUM+DSUM*DELFIO
  190 CONTINUE
      RF(IX)=(CNST*SUM/(CMPP*GRV1(IX)+CMP*GRV2(IX)))+1.d0
  193 CONTINUE
  191 CONTINUE
      DO 8 I=1,N
      IPN1=I+N1*(KOMP-1)
      SINTH=SNTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      IP=(KOMP-1)*(N1+1)+I
      IY=MMSAVE(IP)
      DO 8 J=1,MM
      IS=IS+1
      IX=IY+J
      IF(FR(IX).EQ.0.d0) GOTO 8
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      grv=cmpp*grv1(ix)+cmp*grv2(ix)
      TNEW=TPOLE*dsqrt(dsqrt(GRV*RF(IX)))
      TLD(IS)=TNEW
      if(ifat.eq.0) call planckint(tnew,iband,xintlog,xint)
      if(ifat.ne.0) call atmx(tnew,glogg,iband,xintlog,xint)
      GRREFL=xint/pint
      IF(KOMP.EQ.1) GOTO 77
      slump2(ix)=glump2(ix)*grrefl*sbr
      GOTO 8
   77 slump1(ix)=glump1(ix)*grrefl*sbr
    8 CONTINUE
      RETURN
      END

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

      SUBROUTINE OLUMP(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,
     $MMSAVE,GREXP,ALB,RB,TPOLL,SBR,SUMM,N1,N2,KOMP,IFAT,x,y,D,
     $SNTH,CSTH,SNFI,CSFI,tld,glump1,glump2,glog1,glog2,grv1,grv2,iband)
c   Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),SLUMP1(*),SLUMP2(*),MMSAVE(*),F(3),W(3),SNTH(*),CSTH(*),
     $SNFI(*),CSFI(*),tld(*),glump1(*),glump2(*),glog1(*),glog2(*),
     $grv1(*),grv2(*)
      dimension message(2,4)
      common /atmmessages/ message,kompcom
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm,
     $ifs1dm,ifs2dm,icr1dm,icr2dm,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      kompcom=komp
      IQ=(KOMP-1)*(N1+1)
      IS=(KOMP-1)*MMSAVE(IQ)
      FP=7.957747d-2
      PI=3.141592653589793d0
      PIH=1.570796326794897d0
      PI32=4.712388980384690d0
      F(1)=.1127017d0
      F(2)=.5d0
      F(3)=.8872983d0
      W(1)=.277777777777777d0
      W(2)=.444444444444444d0
      W(3)=.277777777777777d0
      TPOLE=10000.d0*TPOLL
      cmp=dfloat(komp-1)
      cmpp=dfloat(2-komp)
      gplog=cmpp*gplog1+cmp*gplog2
      if(ifat.eq.0) call planckint(tpole,iband,pollog,pint)
      IF(IFAT.NE.0) CALL atmx(tpole,gplog,iband,pollog,pint)
      COMPP=dfloat(2*KOMP-3)
      COMP=-COMPP
      CMPD=CMP*D
      CMPPD=CMPP*D
      N=(2-KOMP)*N1+(KOMP-1)*N2
      ENN=(15.d0+X)*(1.d0+GREXP)/(15.d0-5.d0*X)
      NP=N1+1+(2-KOMP)*(N2+1)
      NPP=N1*(KOMP-1)+(NP-1)*(2-KOMP)
      LL=MMSAVE(NPP)+1
      LLL=MMSAVE(NP)
      LLLL=(LL+LLL)/2
      AR=RV(LLL)*CMP+RVQ(LLL)*CMPP
      BR=RV(LLLL)*CMP+RVQ(LLLL)*CMPP
      CR=RV(1)*CMP+RVQ(1)*CMPP
      BOA=BR/AR
      BOAL=1.d0-BOA*BOA
      BOC2=(BR/CR)**2
      CC=1.d0/(1.d0-.25d0*ENN*(1.d0-BOA**2)*(.9675d0-.3008d0*BOA))
      HCN=.5d0*CC*ENN
      DF=1.d0-X/3.d0
      if(ld.eq.2) df=df+2.d0*y/9.d0
      if(ld.eq.3) df=df-.2d0*y
      EN=dfloat(N)
      DO 8 I=1,N
      IPN1=I+N1*(KOMP-1)
      SINTH=SNTH(IPN1)
      COSTH=CSTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      IP=(KOMP-1)*(N1+1)+I
      IY=MMSAVE(IP)
      DO 8 J=1,MM
      IS=IS+1
      STCF=SINTH*CSFI(IS)
      STSF=SINTH*SNFI(IS)
      IX=IY+J
      IF(KOMP.EQ.1) GOTO 39
      IF(RVQ(IX).EQ.-1.d0) GOTO 8
      GX=GRXQ(IX)
      GY=GRYQ(IX)
      GZ=GRZQ(IX)
      R=RVQ(IX)
      GOTO 49
   39 IF(RV(IX).EQ.-1.d0)GOTO 8
      GX=GRX(IX)
      GY=GRY(IX)
      GZ=GRZ(IX)
      R=RV(IX)
   49 GRMAG=dsqrt(GX*GX+GY*GY+GZ*GZ)
      ZZ=R*COSTH
      YY=R*COMP*STSF
      XX=CMPD+COMP*STCF*R
      XXREF=(CMPPD+COMPP*XX)*COMPP
      GRAV=cmpp*grv1(ix)+cmp*grv2(ix)
      TLOCAL=TPOLE*dsqrt(dsqrt(GRAV))
      DIST=dsqrt(XXREF*XXREF+YY*YY+ZZ*ZZ)
      RMX=dasin(.5d0*(BR+CR)/DIST)
      XCOS=XXREF/DIST
      YCOS=YY/DIST
      ZCOS=ZZ/DIST
      COSINE=(XCOS*GX+YCOS*GY+ZCOS*GZ)/GRMAG
      RC=PIH-dacos(COSINE)
      AH=RC/RMX
      RP=dabs(AH)
      IF(AH.LE..99999d0) GOTO 22
      P=1.d0
      GOTO 16
   22 IF(AH.GE.-.99999d0) GOTO 24
      ALBEP=0.d0
      GOTO 19
   24 SUM=0.d0
      FIST=dasin(RP)
      FII=PIH-FIST
      DO 15 IT=1,3
      FE=FII*F(IT)+FIST
      PAR=1.d0-(RP/dsin(FE))**2
      RPAR=dsqrt(PAR)
      SUM=PAR*RPAR*W(IT)+SUM
   15 CONTINUE
      FTRI=(1.d0-X)*RP*dsqrt(1.d0-RP**2)+.666666666666666d0*X*FII
     $-.666666666666667d0*x*sum*fii
      FSEC=(PIH+FIST)*DF
      P=(FTRI+FSEC)/(PI*DF)
      IF(COSINE.LT.0.d0) P=1.d0-P
      RTF=dsqrt(1.d0-AH**2)
      DENO=PI32-3.d0*(AH*RTF+dasin(AH))
      IF(DENO.NE.0.d0) GOTO 117
      ABAR=1.d0
      GOTO 116
  117 ABAR=2.d0*RTF**3/DENO
  116 COSINE=dcos(PIH-RMX*ABAR)
   16 COSQ=1.d0/(1.d0+(YY/XXREF)**2)
      COT2=(ZZ/XXREF)**2
      Z=BOAL/(1.d0+BOC2*COT2)
      E=CC-HCN*COSQ*Z
      ALBEP=ALB*E*P
   19 IF(COSINE.LE.0.d0) ALBEP=0.d0
      TNEW=TLOCAL*dsqrt(dsqrt(1.d0+(FP*SUMM/(DIST*DIST*GRAV))*
     $cosine*rb*ALBEP))
      TLD(IS)=TNEW
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      if(ifat.eq.0) call planckint(tnew,iband,xintlog,xint)
      if(ifat.ne.0) CALL atmx(TNEW,glogg,iband,xintlog,xint)
      grrefl=xint/pint
      IF(KOMP.EQ.1) GOTO 77
      slump2(ix)=glump2(ix)*grrefl*sbr
      GOTO 8
   77 slump1(ix)=glump1(ix)*grrefl*sbr
    8 CONTINUE
      RETURN
      END

      SUBROUTINE MLRG(a,p,q,r1,r2,t1,t2,sm1,sm2,sr1,sr2,bolm1,
     $bolm2,xlg1,xlg2)
c  Version of January 16, 2002
c
c  This subroutine computes absolute dimensions and other quantities
c  for the stars of a binary star system.
c  a = orbital semi-major axis, the sum of the two a's for the two
c  stars. The unit is a solar radius.
c  r1,r2 = relative mean (equivalent sphere) radii for stars 1 and 2. Th
c  unit is the orbital semimajor axis.
c  p = orbit period in days.
c  q = mass ratio, m2/m1.
c  t1,t2= flux-weighted mean surface temperatures for stars 1 and 2,in K
c  sm1,sm2= masses of stars 1 and 2 in solar units.
c  sr1,sr2= mean radii of stars 1 and 2 in solar units.
c  bolm1, bolm2= absolute bolometric magnitudes of stars 1, 2.
c  xlg1, xlg2= log (base 10) of mean surface acceleration (effective gra
c  for stars 1 and 2.
c
      implicit real*8 (a-h,o-z)
      G=6.668d-8
      tsun=5800.d0
      rsunau=214.8d0
      sunmas=1.991d33
      sunrad=6.960d10
      gmr=G*sunmas/sunrad**2
      sunmb=4.77d0
      sr1=r1*a
      sr2=r2*a
      yrsid=365.2564d0
      tmass=(a/rsunau)**3/(p/yrsid)**2
      sm1=tmass/(1.d0+q)
      sm2=tmass*q/(1.d0+q)
      bol1=(t1/tsun)**4*sr1**2
      bol2=(t2/tsun)**4*sr2**2
      bolm1=sunmb-2.5d0*dlog10(bol1)
      bolm2=sunmb-2.5d0*dlog10(bol2)
      xlg1=dlog10(gmr*sm1/sr1**2)
      xlg2=dlog10(gmr*sm2/sr2**2)
      return
      end
      SUBROUTINE CLOUD(cosa,cosb,cosg,x1,y1,z1,xc,yc,zc,rr,wl,op1,
     $opsf,edens,acm,en,cmpd,ri,dx,dens,tau)
c  Version of January 9, 2002
      implicit real*8 (a-h,o-z)
      dx=0.d0
      tau=0.d0
      ri=1.d0
      sige=.6653d-24
      dtdxes=sige*edens
c  cosa can be zero, so an alternate path to the solution is needed
      dabcoa=dabs(cosa)
      dabcob=dabs(cosb)
      if(dabcoa.lt.dabcob) goto 32
      w=cosb/cosa
      v=cosg/cosa
      u=y1-yc-w*x1
      t=z1-zc-v*x1
      aa=1.d0+w*w+v*v
      bb=2.d0*(w*u+v*t-xc)
      cc=xc*xc+u*u+t*t-rr*rr
      dubaa=aa+aa
      dis=bb*bb-4.d0*aa*cc
      if(dis.le.0.d0) return
      sqd=dsqrt(dis)
      xx1=(-bb+sqd)/dubaa
      xx2=(-bb-sqd)/dubaa
      yy1=w*(xx1-x1)+y1
      yy2=w*(xx2-x1)+y1
      zz1=v*(xx1-x1)+z1
      zz2=v*(xx2-x1)+z1
      goto 39
   32 w=cosa/cosb
      v=cosg/cosb
      u=x1-xc-w*y1
      t=z1-zc-v*y1
      aa=1.d0+w*w+v*v
      bb=2.d0*(w*u+v*t-yc)
      cc=yc*yc+u*u+t*t-rr*rr
      dubaa=aa+aa
      dis=bb*bb-4.d0*aa*cc
      if(dis.le.0.d0) return
      sqd=dsqrt(dis)
      yy1=(-bb+sqd)/dubaa
      yy2=(-bb-sqd)/dubaa
      xx1=w*(yy1-y1)+x1
      xx2=w*(yy2-y1)+x1
      zz1=v*(yy1-y1)+z1
      zz2=v*(yy2-y1)+z1
   39 dis=bb*bb-4.d0*aa*cc
      if(dis.le.0.d0) return
      sqd=dsqrt(dis)
      xs1=(xx1-cmpd)*cosa+yy1*cosb+zz1*cosg
      xs2=(xx2-cmpd)*cosa+yy2*cosb+zz2*cosg
      xxnear=xx1
      yynear=yy1
      zznear=zz1
      xxfar=xx2
      yyfar=yy2
      zzfar=zz2
      xsnear=xs1
      xsfar=xs2
      if(xs1.gt.xs2) goto 38
      xxnear=xx2
      yynear=yy2
      zznear=zz2
      xxfar=xx1
      yyfar=yy1
      zzfar=zz1
      xsnear=xs2
      xsfar=xs1
   38 continue
      xss=(x1-cmpd)*cosa+y1*cosb+z1*cosg
      if(xss.ge.xsnear) return
      if(xss.le.xsfar) goto 20
      xxfar=x1
      yyfar=y1
      zzfar=z1
   20 continue
      dtaudx=dtdxes+(op1*wl**en+opsf)*dens
      dx=dsqrt((xxnear-xxfar)**2+(yynear-yyfar)**2+(zznear-zzfar)**2)
      tau=dx*dtaudx*acm
      ri=dexp(-tau)
      return
      end

      SUBROUTINE LINPRO(komp,dvks,hbarw,tau,emm,count,taug,emmg,fbin,
     $delv)
c  Version of November 3, 2000
      implicit real*8(a-h,o-z)
      dimension dvks(*),hbarw(*),tau(*),emm(*),count(*),fbin(*),delv(*),
     $taug(*),emmg(*)
      common /flpro/ vks,binc,binw,difp,dum1,dum2
      common /ipro/ nbins,nl,inmax,inmin,idum1,idum2
      COMMON /SPOTS/ SINLAT(2,100),COSLAT(2,100),SINLNG(2,100),COSLNG
     $(2,100),RAD(2,100),TEMSP(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      inmin=300000
      inmax=0
c
c  The 83 loop pre-computes the limiting bin numbers, encompassing all lines
c
      do 83 iln=1,nl
      if(Lspot(komp,iln).eq.0) goto 83
      vksg=vks+dvks(iln)
      vksp=vksg+hbarw(iln)
      vksm=vksg-hbarw(iln)
      indp=vksp/binw+binc
      indm=vksm/binw+binc
      if(indm.lt.inmin) inmin=indm
      if(indp.gt.inmax) inmax=indp
   83 continue
      do 82 i=inmin,inmax
      emmg(i)=0.d0
   82 taug(i)=0.d0
c
c  The 84 loop puts fractional contributions into the two end bins
c    (first part, up to 28 continue) and puts full contributions
c    into the middle bins (the 26 loop).
c
      do 84 iln=1,nl
      if(Lspot(komp,iln).eq.0) goto 84
      vksg=vks+dvks(iln)
      vksp=vksg+hbarw(iln)
      vksm=vksg-hbarw(iln)
      indp=vksp/binw+binc
      indm=vksm/binw+binc
      vks1=(dfloat(indm+1)-binc)*binw
      vks2=(dfloat(indp)-binc)*binw
      fr1=(vks1-vksm)/binw
      fr2=(vksp-vks2)/binw
      taug(indm)=taug(indm)+fr1*tau(iln)
      emmg(indm)=emmg(indm)+fr1*emm(iln)
      delv(indm)=delv(indm)+fr1*vksm
      count(indm)=count(indm)+fr1
      taug(indp)=taug(indp)+fr2*tau(iln)
      emmg(indp)=emmg(indp)+fr2*emm(iln)
      delv(indp)=delv(indp)+fr2*vksp
      count(indp)=count(indp)+fr2
      if(indp.ne.indm) goto 28
      taug(indp)=taug(indp)-tau(iln)
      emmg(indp)=emmg(indp)-emm(iln)
      delv(indp)=delv(indp)-.5d0*(vksm+vksp)
      count(indp)=count(indp)-1.d0
   28 continue
      ind=indm
      idmax=indp-indm-1
      if(idmax.le.0) goto 84
      do 26 id=1,idmax
      ind=ind+1
      vksb=(dfloat(ind)-binc)*binw
      taug(ind)=taug(ind)+tau(iln)
      emmg(ind)=emmg(ind)+emm(iln)
      delv(ind)=delv(ind)+vksb
      count(ind)=count(ind)+1.d0
   26 continue
   84 continue
c
c  The 85 loop collects the absorption and emission contributions to all
c     active bins, with the absorption lines summed via an optical thickness
c     treatment and the emission lines summed directly according to contributed
c     flux. The sign on emmg is negative because emmg is negative.
c
      do 85 i=inmin,inmax
   85 fbin(i)=fbin(i)+(1.d0-dexp(-taug(i))+emmg(i))*difp
      return
      end

      SUBROUTINE JDPH(xjdin,phin,t0,p0,dpdt,xjdout,phout)
c  Version of February 2, 1999
c
c  Subroutine jdph computes a phase (phout) based on an input
c   JD (xjdin), reference epoch (t0), period (p0), and dP/dt (dpdt).
c   It also computes a JD (xjdout) from an input phase (phin) and the
c   same ephemeris. So jdph can be used either to get phase from
c   JD or JD from phase.
c
      implicit real*8(a-h,o-z)
      tol=1.d-6
      abdpdt=dabs(dpdt)
      deltop=(xjdin-t0)/p0
      fcsq=0.d0
      fccb=0.d0
      fc4th=0.d0
      fc5th=0.d0
      fc=deltop*dpdt
      if(dabs(fc).lt.1.d-18) goto 25
      fcsq=fc*fc
      if(dabs(fcsq).lt.1.d-24) goto 25
      fccb=fc*fcsq
      if(dabs(fccb).lt.1.d-27) goto 25
      fc4th=fc*fccb
      if(dabs(fc4th).lt.1.d-28) goto 25
      fc5th=fc*fc4th
   25 phout=deltop*(1.d0-.5d0*fc+fcsq/3.d0-.25d0*fccb+.2d0*fc4th-
     $fc5th/6.d0)
      pddph=dpdt*phin
      xjdout=p0*phin*(1.d0+.5d0*pddph+pddph**2/6.d0+pddph**3/24.d0
     $+pddph**4/120.d0+pddph**5/720.d0)+t0
      if(abdpdt.lt.tol) return
      phout=dlog(1.d0+deltop*dpdt)/dpdt
      xjdout=(dexp(dpdt*phin)-1.d0)*p0/dpdt+t0
      return
      end
      SUBROUTINE RANUNI(sn,smod,sm1p1)
c  Version of January 17, 2003
c
c   On each call, subroutine ranuni generates a pseudo-random number,
c     sm1p1, distributed with uniform probability over the range
c     -1. to +1.
c
c   The input number sn, from which both output numbers are generated,
c     should be larger than the modulus 1.00000001d8 and smaller
c     than twice the modulus. The returned number smod will be in
c     that range and can be used as the input sn on the next call
c
      implicit real*8(a-h,o-z)
      st=23.d0
      xmod=1.00000001d8
      smod=st*sn
      goto 2
    1 smod=smod-xmod
    2 if(smod.gt.xmod) goto 1
      sm1p1=(2.d0*smod/xmod-1.d0)
      return
      end
      SUBROUTINE RANGAU(smod,nn,sd,gau)
      implicit real*8(a-h,o-z)
c  Version of February 6, 1997
      ffac=0.961d0
      sfac=ffac*3.d0*sd/(dsqrt(3.d0*dfloat(nn)))
      g1=0.d0
      do 22 i=1,nn
      sn=smod
      call ranuni(sn,smod,sm1p1)
      g1=g1+sm1p1
   22 continue
      gau=sfac*g1
      return
      end

      SUBROUTINE legendre(x,pleg,n)
c  Version of January 7, 2002
      implicit real*8 (a-h,o-z)
      dimension pleg(n)
      pleg(1)=1.d0
      pleg(2)=x
      if(n.le.2) return
      denom=1.d0
      do 1 i=3,n
      fac1=x*(2.d0*denom+1.d0)
      fac2=denom
      denom=denom+1.d0
      pleg(i)=(fac1*pleg(i-1)-fac2*pleg(i-2))/denom
   1  continue
      return
      end

      SUBROUTINE binnum(x,n,y,j)
c  Version of May 31, 2005
      implicit real*8(a-h,o-z)
      dimension x(n)
      mon=1
      if(x(1).gt.x(2)) mon=-1
      do 1 i=1,n
      if(mon.eq.-1) goto 3
      if(y.le.x(i)) goto 2
      goto 1
   3  if(y.gt.x(i)) goto 2
   1  continue
   2  continue
      if (i.lt.1) goto 5
      if (i.eq.1) goto 4
   4  j=i-1
      goto 6
   5  j=i
   6  continue
      return
      end

      SUBROUTINE gabs(komp,smaxis,qq,ecc,period,dd,rad,xm,xmo,absgr,
     $glog)
      implicit real*8(a-h,o-z)
c  Version of September 17, 2004
c
c  Input definitions:
c   smaxis is the length of the orbital semi-major axis in solar radii.
c   qq is the mass ratio in the sense m2/m1. Stars 1 and 2 are as defined
c     in the external program (star 1 is near superior conjunction at
c     phase zero).
c   ecc is orbital eccentricity
c   period is orbit period in days
c   dd is the instantaneous separation of the star centers in unit of th
c     orbital semi-major axis
c   rad is the polar radius of the star at issue in unit of the orbital
c     semi-major axis
c  Output definitions:
c   absgr is the polar acceleration due to effective gravity in cm/sec^2
c   glog is log_10 of absgr
c
      twopi=6.2831853072d0
      gbig=6.670d-8
      sunmas=1.989d33
      sunrad=6.9599d10
      psec=8.64d4*period
      acm=sunrad*smaxis
      pyears=period/365.2422d0
      aau=smaxis/214.9426d0
      tmass=aau**3/pyears**2
      qf=1.d0/(1.d0+qq)
      qfm=qq*qf
      sign=-1.d0
      if(komp.eq.2) goto 10
      qfm=qf
      qf=qq*qf
      sign=1.d0
   10 continue
      xm=tmass*qfm
      xmo=tmass*qf
      gbigm=gbig*xm*sunmas
      gbigmo=gbig*xmo*sunmas
      rcm=rad*acm
      dcm=dd*acm
      dcmsq=dcm*dcm
      efac=dsqrt((1.d0+ecc)*(1.d0-ecc))
      av=twopi*efac/(psec*dd*dd)
      avsq=av*av
      rcmsq=rcm*rcm
      hypsq=rcmsq+dcmsq
      hyp=dsqrt(hypsq)
      snalf=rcm/hyp
      csalf=dcm/hyp
      gz=-gbigm/rcmsq
      gzo=-snalf*gbigmo/hypsq
      gxo=sign*csalf*gbigmo/hypsq
      gxcf=-sign*avsq*dcm*qf
      gxs=gxo+gxcf
      gzs=gz+gzo
      absgr=dsqrt(gxs*gxs+gzs*gzs)
      glog=dlog10(absgr)
      return
      end
			
      subroutine planckint(t,ifil,ylog,y)
      implicit real*8 (a-h,o-z)
c  Version of January 9, 2002
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  IMPORTANT README
c  This subroutine returns the log10 (ylog) of a Planck central
c  intensity (y), as well as the Planck central intensity (y) itself.
c  The subroutine ONLY WORKS FOR TEMPERATURES GREATER THAN OR EQUAL
c  500 K OR LOWER THAN 500,300 K. For teperatures outside this range,
c  the program stops and prints a message.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dimension plcof(1250)
      dimension pl(10)
      common /invar/ id1,id2,id3,id4,id5,id6,id7,id8,id9,
     $id10,id11,ld,id13,id14,id15
      common /planckleg/ plcof
      common /coflimbdark/ xld,yld
      if(t.lt.500.d0) goto 11
      if(t.ge.1900.d0) goto 1
      tb=500.d0
      te=2000.d0
      ibin=1
      goto 5
   1  if(t.ge.5500.d0) goto 2
      tb=1800.d0
      te=5600.d0
      ibin=2
      goto 5
   2  if(t.ge.20000.d0) goto 3
      tb=5400.d0
      te=20100.d0
      ibin=3
      goto 5
   3  if(t.ge.100000.d0) goto 4
      tb=19900.d0
      te=100100.d0
      ibin=4
      goto 5
   4  if(t.gt.500300.d0) goto 11
      tb=99900.d0
      te=500300.d0
      ibin=5
   5  continue
      ib=(ifil-1)*50+(ibin-1)*10
      phas=(t-tb)/(te-tb)
      call legendre(phas,pl,10)
      y=0.d0
      do 6 j=1,10
      jj=j+ib
   6  y=y+pl(j)*plcof(jj)
      dark=1.d0-xld/3.d0
      if(ld.eq.2) dark=dark+yld/4.5d0
      if(ld.eq.3) dark=dark-0.2d0*yld
c     0.49714987269413 = log10 (pi)
      ylog=y-dlog10(dark)-0.49714987269413d0
      y=10.d0**ylog
      return
  11  continue
      write(6,*) "Program stopped in PLANCKINT, T=", T
      stop
  80  format('Program stopped in PLANCKINT,
     $T outside 500 - 500,300 K range.')
      end
			
      SUBROUTINE conjph(ecc,argper,phzero,trsc,tric,econsc,econic,
     $xmsc,xmic,pconsc,pconic)
      implicit real*8(a-h,o-z)
c  Version of December 15, 2003
c
c  Subroutine conjph computes the phases of superior and inferior conjunction
c    (pconsc and pconic) of star 1
c
      pi=dacos(-1.d0)
      pih=.5d0*pi
      pi32=1.5d0*pi
      twopi=pi+pi
      ecfac=dsqrt((1.d0-ecc)/(1.d0+ecc))
c
c  sc in variable names (like trsc) means superior conjunction, and
c  ic means inferior conjunction (always for star 1).
c
      trsc=pih-argper
      tric=pi32-argper
      econsc=2.d0*datan(ecfac*dtan(.5d0*trsc))
      econic=2.d0*datan(ecfac*dtan(.5d0*tric))
      xmsc=econsc-ecc*dsin(econsc)
      xmic=econic-ecc*dsin(econic)
      pconsc=(xmsc+argper)/twopi-.25d0+phzero
      pconic=(xmic+argper)/twopi-.25d0+phzero
      return
      end









































      subroutine WDDC (fatmcof,fatmcofpl,correction,stddev,chi2s,chi2)
      implicit real*8 (a-h,o-z)
      character fatmcof*(*),fatmcofpl*(*)
      double precision correction(*),stddev(*),chi2s(*)
      double precision chi2
      dimension phas(6000),flux(6000),wt(6000),br(6000),bl(6000),
     $obs(120000),hold(120000),cn(2500),cnn(2500),out(50),sd(50),
     $ccl(50),iband(25),
     $ll(50),mm(50),hla(25),cla(25),x1a(25),x2a(25),y1a(25),y2a(25),
     $el3a(25),wla(25),noise(25),sigma(25),knobs(27),mmsavh(124),
     $mmsavl(124),theta(260),rho(260),aa(20),bb(20),hld(3200),v(60),
     $cnout(2025),snthh(130),csthh(130),snfih(6400),csfih(6400),
     $snthl(130),csthl(130),snfil(6400),csfil(6400),tldh(6400),
     $tldl(6400),opsfa(25),phjd(6000),dfdph(6000),dfdap(6000),para(500)
      dimension rv(3011),grx(3011),gry(3011),grz(3011),rvq(3011),
     $grxq(3011),gryq(3011),grzq(3011),slump1(3011),slump2(3011),
     $srv(3011),sgrx(3011),sgry(3011),sgrz(3011),srvq(3011),sgrxq(3011)
     $,sgryq(3011),sgrzq(3011),srvl(3011),sgrxl(3011),sgryl(3011),
     $sgrzl(3011),srvql(3011),sgrxql(3011),sgryql(3011),sgrzql(3011),
     $slmp1(3011),slmp2(3011),slmp1l(3011),slmp2l(3011),fr1(3011),
     $fr2(3011),glump1(3011),glump2(3011),grv1(3011),grv2(3011),
     $xx1(3011),xx2(3011),yy1(3011),yy2(3011),zz1(3011),zz2(3011),
     $gmag1(3011),gmag2(3011),csbt1(3011),csbt2(3011),rf1(3011),
     $rf2(3011),rftemp(3011),sxx1(3011),sxx2(3011),syy1(3011),
     $syy2(3011),szz1(3011),szz2(3011),sgmg1(3011),sgmg2(3011),
     $sgrv1(3011),sgrv2(3011),sglm1(3011),sglm2(3011),scsb1(3011),
     $scsb2(3011),srf1(3011),srf2(3011),sglm1l(3011),sglm2l(3011),
     $sgrv1l(3011),sgrv2l(3011),sxx1l(3011),sxx2l(3011),syy1l(3011),
     $syy2l(3011),szz1l(3011),szz2l(3011),sgmg1l(3011),sgmg2l(3011),
     $scsb1l(3011),scsb2l(3011),srf1l(3011),srf2l(3011),yskp(1),
     $zskp(1),glog1(3011),glog2(3011)
      dimension erv(3011),egrx(3011),egry(3011),egrz(3011),
     $elmp1(3011),eglm1(3011),egrv1(3011),exx1(3011),eyy1(3011),
     $ezz1(3011),egmg1(3011),ecsb1(3011),erf1(3011),ervq(3011),
     $egrxq(3011),egryq(3011),egrzq(3011),elmp2(3011),eglm2(3011),
     $egrv2(3011),exx2(3011),eyy2(3011),ezz2(3011),egmg2(3011),
     $ecsb2(3011),erf2(3011),
     $ervl(3011),egrxl(3011),egryl(3011),egrzl(3011),elmp1l(3011),
     $eglm1l(3011),egrv1l(3011),exx1l(3011),eyy1l(3011),ezz1l(3011),
     $egmg1l(3011),ecsb1l(3011),erf1l(3011),ervql(3011),egrxql(3011),
     $egryql(3011),egrzql(3011),elmp2l(3011),eglm2l(3011),
     $egrv2l(3011),exx2l(3011),eyy2l(3011),ezz2l(3011),egmg2l(3011),
     $ecsb2l(3011),erf2l(3011),sfr1(3011),sfr1l(3011),efr1(3011),
     $efr1l(3011),sfr2(3011),sfr2l(3011),efr2(3011),efr2l(3011),
     $stldh(6400),stldl(6400),etldh(6400),etldl(6400)
      dimension cnc(2500),clc(50)
      dimension del(35),keep(36),kep(35),nshift(36),low(35),dlif(35),
     $xtha(4),xfia(4),arad(4),xlat(2,100),xlong(2,100),radsp(2,100)
     $,temsp(2,100),po(2),omcr(2)
      dimension xcl(100),ycl(100),zcl(100),rcl(100),op1(100),fcl(100),
     $dens(100),encl(100),edens(100),xmue(100)
      dimension message(2,4)
      dimension abun(19),glog(11),grand(250800),plcof(1250)
c
c The following dimensioned variables are not used by DC. They are
c    dimensioned only for compatibility with usage of subroutine
c    LIGHT by program LC.
c
      dimension fbin1(1),fbin2(1),delv1(1),delv2(1),count1(1),
     $count2(1),delwl1(1),delwl2(1),resf1(1),resf2(1),wl1(1),wl2(1),
     $dvks1(1),dvks2(1),tau1(1),tau2(1),taug(1),hbarw1(1),hbarw2(1),
     $emm1(1),emm2(1),emmg(1)
      common /abung/ abun,glog
      common /arrayleg/ grand,istart
      common /planckleg/ plcof
      common /atmmessages/ message,komp
      common /ramprange/ tlowtol,thightol,glowtol,ghightol
      COMMON /ECCEN/ EC,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      COMMON /DPDX/ DPDX1,DPDX2,POT1,POT2
      COMMON /SUMM/ SUMM1,SUMM2
      common /ardot/ dperdt,hjd,hjd0,perdum
      COMMON /FLVAR/ PSH,DP,EF,EFC,ECOS,pert,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      COMMON /INVAR/ KH,IPB,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,ifsmv2,
     $icor1,icor2,ld,ncl,jdphs,ipc
      COMMON /SPOTS/ SNLAT(2,100),CSLAT(2,100),SNLNG(2,100),CSLNG(2,100)
     $,rdsp(2,100),tmsp(2,100),xlng(2,100),kks(2,100),Lspot(2,100)
      COMMON /NSPT/ NSP1,NSP2
      common /cld/ acm,opsf
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      common /prof2/ duma,dumb,dumc,dumd,du1,du2,du3,du4,du5,du6,du7
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      DATA ARAD(1),ARAD(2),ARAD(3),ARAD(4)/4HPOLE,5HPOINT,4HSIDE,4HBACK/
c
c           Bandpass Label Assignments for Stellar Atmospheres
c
c    Label   Bandpass   Reference for Response Function
c    -----   --------   -------------------------------
c       1        u      Crawford, D.L. and Barnes, J.V. 1974, AJ, 75, 978
c       2        v          "                "           "
c       3        b          "                "           "
c       4        y          "                "           "
c       5        U      Buser, R. 1978, Ang, 62, 411
c       6        B      Azusienis and Straizys 1969, Sov. Astron., 13, 316
c       7        V          "             "                "
c       8        R      Johnson, H.L. 1965, ApJ, 141, 923
c       9        I         "            "    "
c      10        J         "            "    "
c      11        K         "            "    "
c      12        L         "            "    "
c      13        M         "            "    "
c      14        N         "            "    "
c      15        R_c    Bessell, M.S. 1983, PASP, 95, 480
c      16        I_c       "            "    "
c      17      230      Kallrath, J., Milone, E.F., Terrell, D., Young, A.T.
c                          1998, ApJ, 508, 308
c      18      250         "             "             "           "
c      19      270         "             "             "           "
c      20      290         "             "             "           "
c      21      310         "             "             "           "
c      22      330         "             "             "           "
c
   15 FORMAT(1X,16(F11.5))
   16 FORMAT(1X,18(F7.4))
   67 FORMAT(20A4)
   17 FORMAT(1X,22(F6.3))
   19 FORMAT(1X,26(F5.2))
   21 FORMAT(f27.16,f36.16,d22.6)
   55 FORMAT(10(3X,d8.1))
   56 FORMAT(10(1X,d7.1))
   20 format(1x,2(4i1,1x),7i1,1x,4(5i1,1x),i1,1x,i1,1x,i1,d10.3)
  101 FORMAT(' ')
    1 FORMAT(I3,I6,I6,I7,I6,I4,I4,I4,f15.6,d13.5,f10.5,f16.3,f14.4)
  701 FORMAT(4I2,4I3,f13.6,d12.5,F8.5,F9.3)
    2 FORMAT(5(F14.5,F8.4,F6.2))
   85 FORMAT(i3,2F10.5,4(1X,F6.3),f8.4,d10.3,i6,d14.5,f10.6)
   18 format(i3,2f10.5,4f7.3,f8.4,d10.3,i2,d12.5,f10.6)
  218 FORMAT(i3,2F10.5,4F7.3,d10.3,d12.5,f10.6)
   37 FORMAT(1X,11F12.7)
  137 FORMAT(1X,F11.7)
  715 format(22x,'Input-Output in F Format')
  716 format(22x,'Input-Output in D Format')
   43 format('No.',2x,'Curve',4x,'Input Param.',8x,'Correction',5x,
     $'Output Param.',4x,'Standard Deviation')
  615 format(i2,i7,4f18.10)
  616 format(i2,i7,4d18.10)
  138 FORMAT(1X,'SUM OF ABSOLUTE VALUES OF CHECKS IS',1X,D12.6)
  181 FORMAT(7X,'NORMAL EQUATIONS')
  183 FORMAT (7X,'CORRELATION COEFFICIENTS')
  184 FORMAT(7X,'NORMAL EQUATIONS TIMES INVERSE')
  185 FORMAT(1X,'CHECK OF COMPUTED DEPENDENT VARIABLES FROM NORMAL EQUAT
     $IONS')
   82 FORMAT(7X,'UNWEIGHTED OBSERVATIONAL EQUATIONS')
   83 FORMAT(7X,'WEIGHTED OBSERVATIONAL EQUATIONS')
    9 FORMAT(33X,'OBSERVATIONS')
  955 FORMAT(3(9x,'phase   V rad   wt'))
   10 FORMAT(3(9x,'phase   light   wt'))
  755 FORMAT(3(9x,'JD      V rad   wt'))
  756 FORMAT(3(9x,'JD      light   wt'))
   40 FORMAT('   Sum(W*Res**2) for input values       Sum(W*Res**2) pred
     $icted          determinant')
   11 format(1x,'band',5x,'L1',8x,'L2     x1     x2     y1     y2   3rd
     $lt',4x,'opsf     NOISE    Sigma',6x,'Wave L')
  111 FORMAT(1x,'band',5x,'L1',8x,'L2     x1    x2    y1    y2     ops
     $f',7x,'Sigma',6x,'Wave L')
   12 FORMAT('MODE   IPB  IFAT1  IFAT2  N1  N2 N1L N2L',7x,'Arg Per',5x
     $,'dperdt',7x,'TH e',8x,'V unit(km/s)     V FAC')
  206 FORMAT(F6.5,d13.6,2F11.4,F11.4,f10.3,2f8.3,i6,i9,f9.2,i3)
  205 FORMAT('  ecc',4x,'S-M axis',7x,'F1',9x,'F2',8x,'V Gam',
     $7x,'INCL      G1      G2',2x,'Nspot 1',2x,'Nspot 2','  [M/H] iab')
  402 FORMAT('    DEL EC     DEL PER    DEL F1     DEL F2     DEL PHS
     $ DEL INCL    DEL G1     DEL G2     DEL T1     DEL T2')
  403 FORMAT('    DEL ALB1   DEL ALB2   DEL POT1   DEL POT2   DEL Q
     $ DEL L1     DEL L2     DEL X1     DEL X2')
  406 FORMAT(' ADJUSTMENT CONTROL INTEGERS; 1 SUPPRESSES ADJUSTMENT, 0 A
     $LLOWS ADJUSTMENT.')
  702 FORMAT(F6.5,d13.6,2F10.4,F10.4,f9.3,2f7.3,f7.2)
  706 FORMAT(F7.4,f8.4,2F7.3,3d13.6,4f7.3)
  408 FORMAT(2F8.4,2F9.3,2d15.6,d13.6,4f9.3)
  705 FORMAT(I1,1X,I1,1X,5I2)
   54 FORMAT('     T1      T2',5x,'Alb 1',4x,'Alb 2',9x,'Pot 1',10x,
     $'Pot 2',8x,'M2/M1  x1(bolo) x2(bolo) y1(bolo) y2(bolo)')
  707 FORMAT('    IFVC1   IFVC2   NLC   KO   KDISK   ISYM   nppl')
  917 format('nref',3x,'mref',3x,'ifsmv1',3x,'ifsmv2',3x,'icor1',3x,
     $'icor2',3x,'ld')
  708 FORMAT(8(4X,I3))
  912 format(i3,i7,i8,i9,i8,i8,i7)
  650 FORMAT(20X,'RADII AND RELATED QUANTITIES (FROM INPUT)')
  651 FORMAT(5X,'DOM1/DQ',5X,'DOM2/DQ',5X,'OM1-Q CORR.',5X,'OM2-Q CORR.'
     $,5X,'OM1 S.D.',4X,'OM2 S.D.',4X,'Q  S.D.')
  652 FORMAT(1X,3F12.6,4X,F12.6,4X,3F12.6)
  653 FORMAT(' COMPONENT',11X,'R',9X,'DR/DOM',8X,'DR/DQ',11X,'S.D.')
  654 FORMAT(2X,I1,1X,A6,4F14.6)
  684 FORMAT(i2,4F13.5)
  985 FORMAT(4f9.5)
  983 FORMAT(1X,'STAR  CO-LATITUDE  LONGITUDE  SPOT RADIUS  TEMP.FACTOR
     $')
  399 FORMAT('    DEL LAT    DEL LONG   DEL RAD    DEL TEMPF  DEL LAT
     $ del LONG   del RAD    del TEMPF')
   60 FORMAT(4I3)
   61 FORMAT(1X,4I6)
   66 FORMAT('   STAR  SPOT   STAR  SPOT')
  166 FORMAT(' SPOTS TO BE ADJUSTED')
  440 FORMAT(' AS1=FIRST ADJUSTED SPOT')
  441 FORMAT(' AS2=SECOND ADJUSTED SPOT')
  405 FORMAT(' ORDER OF PARAMETERS IS AS FOLLOWS:')
 1440 FORMAT('  (1) - AS1 LATITUDE')
 1441 FORMAT('  (2) - AS1 LONGITUDE')
 1442 FORMAT('  (3) - AS1 ANGULAR RADIUS')
 1443 FORMAT('  (4) - AS1 TEMPERATURE FACTOR')
 1444 FORMAT('  (5) - AS2 LATITUDE')
 1445 FORMAT('  (6) - AS2 LONGITUDE')
 1446 FORMAT('  (7) - AS2 ANGULAR RADIUS')
 1447 FORMAT('  (8) - AS2 TEMPERATURE FACTOR')
 1448 FORMAT('  (9) - A=ORBITAL SEMI-MAJOR AXIS')
 1449 FORMAT(' (10) - E=ORBITAL ECCENTRICITY')
 1450 FORMAT(' (11) - PERR0=ARGUMENT of PERIASTRON at time HJD0')
 1451 FORMAT(' (12) - F1=STAR 1 ROTATION PARAMETER')
 1452 FORMAT(' (13) - F2=STAR 2 ROTATION PARAMETER')
 1453 FORMAT(' (14) - PHASE SHIFT= PHASE OF PRIMARY CONJUNCTION')
 1454 FORMAT(' (15) - VGAM=SYSTEMIC RADIAL VELOCITY')
 1455 FORMAT(' (16) - INCL=ORBITAL INCLINATION')
 1456 FORMAT(' (17) - g1=STAR 1 GRAVITY DARKENING EXPONENT')
 1457 FORMAT(' (18) - g2=STAR 2 GRAVITY DARKENING EXPONENT')
 1458 FORMAT(' (19) - T1=STAR 1 AVERAGE SURFACE TEMPERATURE')
 1459 FORMAT(' (20) - T2=STAR 2 AVERAGE SURFACE TEMPERATURE')
 1460 FORMAT(' (21) - ALB1=STAR 1 BOLOMETRIC ALBEDO')
 1461 FORMAT(' (22) - ALB2=STAR 2 BOLOMETRIC ALBEDO')
 1462 FORMAT(' (23) - POT1=STAR 1 SURFACE POTENTIAL')
 1463 FORMAT(' (24) - POT2=STAR 2 SURFACE POTENTIAL')
 1464 FORMAT(' (25) - Q=MASS RATIO (STAR 2/STAR 1)')
 1470 FORMAT(' (26) - HJD0= Hel. JD reference time')
 1471 FORMAT(' (27) - PERIOD= orbital period')
 1472 FORMAT(' (28) - DPDT= time derivative of orbital period')
 1473 FORMAT(' (29) - DPERDT= time derivative of argument of periastron'
     $)
 1474 FORMAT(' (30) - unused channel reserved for future expansion')
 1465 FORMAT(' (31) - L1=STAR 1 RELATIVE MONOCHROMATIC LUMINOSITY')
 1466 FORMAT(' (32) - L2=STAR 2 RELATIVE MONOCHROMATIC LUMINOSITY')
 1467 FORMAT(' (33) - X1=STAR 1 LIMB DARKENING COEFFICIENT')
 1468 FORMAT(' (34) - X2=STAR 2 LIMB DARKENING COEFFICIENT')
 1469 FORMAT(' (35) - el3=third light')
  119 format(1x,i6,i13,f18.8)
  159 format(' Sums of squares of residuals for separate curves, includi
     $ng only individual weights')
  169 format('    Curve     No. of obs.   Sum of squares')
 1063 format(3f9.4,f7.4,d11.4,f9.4,d11.3,f9.4,f7.3)
   64 format(3f10.4,f9.4,d12.4,f10.4,d12.4,f9.4,f9.3,d12.4)
   69 format('      xcl       ycl       zcl      rcl       op1         f
     $cl        ne       mu e      encl     dens')
  170 format(i3,f17.6,d18.10,d14.6,f10.4)
  649 format(i1,f15.6,d17.10,d14.6,f10.4)
  171 format('JDPHS',5x,'J.D. zero',7x,'Period',11x,'dPdt',
     $6x,'Ph. shift')
  911 format(7(i1,1x))
  840 format('Do not try to adjust the ephemeris or any time derivative
     $parameters','when JDPHS = 2')
  839 format('Ordinarily one should not try to adjust both PSHIFT and',
     $' HJD0. They are perfectly correlated if the period is constant',
     $' and extremely highly correlated if the period is not constant')
  283 format('log g below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  284 format('log g above ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  285 format('T above ramp range for at least one',
     $' point on star',i2,', black body applied locally.')
  286 format('T below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  287 format('Input [M/H] = ',f6.3,' is not a value recognized by ',
     $'the program. Replaced by ',f5.2)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  Ramp ranges are set here. The values below seem to work well, however,
c  they can be changed.
      tlowtol=1500.d0
      thightol=50000.d0
      glowtol=4.0d0
      ghightol=4.0d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      abun(1)=1.d0
      abun(2)=0.5d0
      abun(3)=0.3d0
      abun(4)=0.2d0
      abun(5)=0.1d0
      abun(6)=0.0d0
      abun(7)=-0.1d0
      abun(8)=-0.2d0
      abun(9)=-0.3d0
      abun(10)=-0.5d0
      abun(11)=-1.0d0
      abun(12)=-1.5d0
      abun(13)=-2.0d0
      abun(14)=-2.5d0
      abun(15)=-3.0d0
      abun(16)=-3.5d0
      abun(17)=-4.0d0
      abun(18)=-4.5d0
      abun(19)=-5.0d0
      glog(1)=0.0d0
      glog(2)=0.5d0
      glog(3)=1.0d0
      glog(4)=1.5d0
      glog(5)=2.0d0
      glog(6)=2.5d0
      glog(7)=3.0d0
      glog(8)=3.5d0
      glog(9)=4.0d0
      glog(10)=4.5d0
      glog(11)=5.0d0
      message(1,1)=0
      message(1,2)=0
      message(2,1)=0
      message(2,2)=0
      message(1,3)=0
      message(1,4)=0
      message(2,3)=0
      message(2,4)=0
      open(unit=22,file=fatmcof,status='old')
      read(22,*) grand
      close (22)
      open(unit=23,file=fatmcofpl,status='old')
      read(23,*) plcof
      close (23)
      open(unit=5,file='dcin.active',status='old')
      open(unit=6,file='dcout.active')
      tt=2.d0/3.d0
      nbins=1
      nl=1
      inmax=1
      inmin=1
      nf1=1
      nf2=1
      toldis=1.d-5
      pi=dacos(-1.d0)
      en0=6.0254d23
      XTHA(1)=0.d0
      XTHA(2)=.5d0*PI
      XTHA(3)=.5d0*PI
      XTHA(4)=.5d0*PI
      XFIA(1)=0.d0
      XFIA(2)=0.d0
      XFIA(3)=.5d0*PI
      XFIA(4)=PI
c
c  The initializations in th 886 and 887 loops are just to avoid
c    triggering error messages from some compilers. The quantities do
c    not otherwise need initialization. Same for du1, ...., du7.
c    Same for mpage, nl1, nl2.
c
      do 886 ikks=1,2
      do 886 jkks=1,100
  886 kks(ikks,jkks)=0
      do 887 immsav=1,124
      mmsavh(immsav)=0
  887 mmsavl(immsav)=0
      du1=0.d0
      du2=0.d0
      du3=0.d0
      du4=0.d0
      du5=0.d0
      du6=0.d0
      du7=0.d0
      mpage=0
      nl1=0
      nl2=0
      IBEF=0
      KH=25
      NS=1
      NI=0
      NY=0
      KNOBS(1)=0
      WRITE(6,405)
      WRITE(6,101)
      WRITE(6,440)
      WRITE(6,441)
      WRITE(6,101)
      WRITE(6,1440)
      WRITE(6,1441)
      WRITE(6,1442)
      WRITE(6,1443)
      WRITE(6,1444)
      WRITE(6,1445)
      WRITE(6,1446)
      WRITE(6,1447)
      WRITE(6,1448)
      WRITE(6,1449)
      WRITE(6,1450)
      WRITE(6,1451)
      WRITE(6,1452)
      WRITE(6,1453)
      WRITE(6,1454)
      WRITE(6,1455)
      WRITE(6,1456)
      WRITE(6,1457)
      WRITE(6,1458)
      WRITE(6,1459)
      WRITE(6,1460)
      WRITE(6,1461)
      WRITE(6,1462)
      WRITE(6,1463)
      WRITE(6,1464)
      WRITE(6,1470)
      WRITE(6,1471)
      WRITE(6,1472)
      WRITE(6,1473)
      WRITE(6,1474)
      WRITE(6,1465)
      WRITE(6,1466)
      WRITE(6,1467)
      WRITE(6,1468)
      WRITE(6,1469)
      READ(5,56)(DEL(I),I=1,8)
      READ(5,56)(DEL(I),I=10,14),(DEL(I),I=16,20)
      READ(5,56)(DEL(I),I=21,25),(del(i),i=31,34)
      READ(5,20)(KEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      READ(5,60) KSPA,NSPA,KSPB,NSPB
      READ(5,705)IFVC1,IFVC2,NLC,KO,KDISK,ISYM,nppl
      read(5,911)nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      read(5,649) jdphs,hjd0,period,dpdt,pshift
      if(jdphs.eq.2.and.kep(26).eq.0) write(6,840)
      if(jdphs.eq.2.and.kep(27).eq.0) write(6,840)
      if(jdphs.eq.2.and.kep(28).eq.0) write(6,840)
      if(kep(14).eq.0.and.kep(26).eq.0) write(6,839)
      READ(5,701)MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,perr0,dperdt,THE,
     $vunit
      READ(5,702)E,A,F1,F2,VGA,XINCL,GR1,GR2,abunin
      READ(5,706) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,xbol1,xbol2,ybol1,
     $ybol2
      acm=6.960d10*a
      nn1=n1
      CALL SINCOS(1,N1,N1,SNTHH,CSTHH,SNFIH,CSFIH,MMSAVH)
      CALL SINCOS(2,N2,N1,SNTHH,CSTHH,SNFIH,CSFIH,MMSAVH)
      CALL SINCOS(1,N1L,N1L,SNTHL,CSTHL,SNFIL,CSFIL,MMSAVL)
      CALL SINCOS(2,N2L,N1L,SNTHL,CSTHL,SNFIL,CSFIL,MMSAVL)
      dint1=pi*(1.d0-xbol1/3.d0)
      if(ld.eq.2) dint1=dint1+pi*ybol1*2.d0/9.d0
      if(ld.eq.3) dint1=dint1-pi*ybol1*.2d0
      dint2=pi*(1.d0-xbol2/3.d0)
      if(ld.eq.2) dint2=dint2+pi*ybol2*2.d0/9.d0
      if(ld.eq.3) dint2=dint2-pi*ybol2*.2d0
      IS=ISYM+1
      KEEP(36)=0
      MM1=N1+1
      MM2=N1+N2+2
      MM3=N1L+1
      MM4=N1L+N2L+2
      M1H=MMSAVH(MM1)
      M2H=MMSAVH(MM2)
      M1L=MMSAVL(MM3)
      M2L=MMSAVL(MM4)
      MTLH=M1H+M2H
      MTLL=M1L+M2L
      NVC=IFVC1+IFVC2
      NLVC=NLC+NVC
      NVCP=NVC+1
      IF(NVC.NE.0) GOTO 288
      KEP(9)=1
      KEP(15)=1
  288 CONTINUE
      DO 84 I=1,35
      KEEP(I)=KEP(I)
   84 LOW(I)=1
      LOW(1)=0
      LOW(2)=0
      LOW(3)=0
      LOW(5)=0
      LOW(6)=0
      LOW(7)=0
      LOW(10)=0
      LOW(11)=0
      LOW(12)=0
      LOW(13)=0
      LOW(14)=0
      LOW(16)=0
      LOW(23)=0
      LOW(24)=0
      LOW(25)=0
      ifap=1-keep(29)
      ifphi=1-keep(26)*keep(27)*keep(28)
      KOSQ=(KO-2)*(KO-2)
      IF(NVC.EQ.0) GOTO 195
      DO 90 I=1,NVC
   90 READ(5,218) iband(i),HLA(I),CLA(I),X1A(I),X2A(I),y1a(i),y2a(i),
     $opsfa(i),sigma(i),wla(i)
  195 CONTINUE
      IF(NLVC.EQ.NVC) GOTO 194
      DO 190 I=NVCP,NLVC
  190 read(5,18)iband(i),hla(i),cla(i),x1a(i),x2a(i),y1a(i),y2a(i),
     $el3a(i),opsfa(i),noise(i),sigma(i),wla(i)
  194 CONTINUE
      NSP1=0
      NSP2=0
      DO 988 KP=1,2
      DO 987 I=1,100
      READ(5,985)XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
      xlng(kp,i)=xlong(kp,i)
      IF(XLAT(KP,I).GE.200.d0) GOTO 988
      SNLAT(KP,I)=dsin(XLAT(KP,I))
      CSLAT(KP,I)=dcos(XLAT(KP,I))
      SNLNG(KP,I)=dsin(XLONG(KP,I))
      CSLNG(KP,I)=dcos(XLONG(KP,I))
      RDSP(KP,I)=RADSP(KP,I)
      TMSP(KP,I)=TEMSP(KP,I)
      IF(KP.EQ.1) NSP1=NSP1+1
  987 IF(KP.EQ.2) NSP2=NSP2+1
  988 CONTINUE
      NSTOT=NSP1+NSP2
      ncl=0
      do 1062 i=1,100
      read(5,1063) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i)
      if(xcl(i).gt.100.d0) goto 1066
      ncl=ncl+1
      dens(i)=edens(i)*xmue(i)/en0
 1062 continue
 1066 continue
      para(1)=xlat(kspa,nspa)
      para(2)=xlong(kspa,nspa)
      para(3)=radsp(kspa,nspa)
      para(4)=temsp(kspa,nspa)
      para(5)=xlat(kspb,nspb)
      para(6)=xlong(kspb,nspb)
      para(7)=radsp(kspb,nspb)
      para(8)=temsp(kspb,nspb)
      para(9)=a
      para(10)=e
      para(11)=perr0
      para(12)=f1
      para(13)=f2
      para(14)=pshift
      para(15)=vga
      para(16)=xincl
      para(17)=gr1
      para(18)=gr2
      para(19)=tavh
      para(20)=tavc
      para(21)=alb1
      para(22)=alb2
      para(23)=phsv
      para(24)=pcsv
      para(25)=rm
      para(26)=hjd0
      para(27)=period
      para(28)=dpdt
      para(29)=dperdt
      para(30)=0.d0
      ib=nvc
      do 191 irx=31,30+nlc
      ib=ib+1
  191 para(irx)=hla(ib)
      ib=nvc
      do 186 irx=31+nlc,30+2*nlc
      ib=ib+1
  186 para(irx)=cla(ib)
      ib=nvc
      do 187 irx=31+2*nlc,30+3*nlc
      ib=ib+1
  187 para(irx)=x1a(ib)
      ib=nvc
      do 188 irx=31+3*nlc,30+4*nlc
      ib=ib+1
  188 para(irx)=x2a(ib)
      ib=nvc
      do 189 irx=31+4*nlc,30+5*nlc
      ib=ib+1
  189 para(irx)=el3a(ib)
      PERT=perr0
      EC=E
      hjd=hjd0
      PSH=PSHIFT
      IRTE=0
      IRVOL1=0
      IRVOL2=0
c***************************************************************
c  The following lines take care of abundances that may not be among
c  the 19 Kurucz values (see abun array). abunin is reset at the
c  allowed value nearest the input value.
      call binnum(abun,19,abunin,iab)
      dif1=abunin-abun(iab)
      if(iab.eq.19) goto 7702
      dif2=abun(iab+1)-abun(iab)
      dif=dif1/dif2
      if((dif.ge.0.d0).and.(dif.le.0.5d0)) goto 7702
      iab=iab+1
 7702 continue
      if(dif1.ne.0.d0) write(6,287) abunin,abun(iab)
      abunin=abun(iab)
      istart=1+(iab-1)*13200
c***************************************************************
      CALL MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $RM,PHSV,PCSV,GR1,GR2,ALB1,ALB2,N1,N2,F1,F2,MOD,XINCL,THE,MODE,
     $SNTHH,CSTHH,SNFIH,CSFIH,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1,
     $GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
      call ellone(f1,dp,rm,xld,omcr(1),xldd,omd)
      rmr=1.d0/rm
      call ellone(f2,dp,rmr,xld,omcr(2),xldd,omd)
      omcr(2)=rm*omcr(2)+(1.d0-rm)*.5d0
      po(1)=phsv
      po(2)=pcsv
      CALL VOLUME(VOL1,RM,PHSV,DP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR1,1)
      CALL VOLUME(VOL2,RM,PCSV,DP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR2,1)
      IF(E.EQ.0.d0) GOTO 134
      DAP=1.d0+E
      P1AP=PHSV-2.d0*E*RM/(1.d0-E*E)
      VL1=VOL1
      CALL VOLUME(VL1,RM,P1AP,DAP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR1,2)
      DPDX1=(PHSV-P1AP)*(1.d0-E*E)*.5d0/E
      P2AP=PCSV-2.d0*E/(1.d0-E*E)
      VL2=VOL2
      CALL VOLUME(VL2,RM,P2AP,DAP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR2,2)
      DPDX2=(PCSV-P2AP)*(1.d0-E*E)*.5d0/E
  134 CONTINUE
      PHP=PHPER
      POTH=PHSV
      POTC=PCSV
      POT1=PHSV
      POT2=PCSV
      DO 24 I=1,NLVC
      opsf=opsfa(i)
   24 CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POTH,POTC,N1,N2,F1,F2,d,hla(i),
     $cla(i),x1a(i),x2a(i),y1a(i),y2a(i),gr1,gr2,wla(i),sm1,sm2,tpolh,
     $tpolc,sbrh,sbrc,tavh,tavc,alb1,alb2,xbol1,xbol2,
     $ybol1,ybol2,php,rm,xincl,hot,cool,snthh,csthh,snfih,csfih,tldh,
     $glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,dint1,dint2,grv1,grv2,
     $rftemp,rf1,rf2,csbt1,csbt2,gmag1,gmag2,glog1,glog2,fbin1,fbin2,
     $delv1,delv2,
     $count1,count2,delwl1,delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,
     $tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,
     $encl,edens,taug,emmg,yskp,zskp,mode,iband(i),ifat1,ifat2,1)
      DEL(9)=0.d0
      DEL(15)=0.d0
      del(26)=0.d0
      del(27)=0.d0
      del(28)=0.d0
      del(29)=0.d0
      DEL(35)=0.d0
      WRITE(6,101)
      WRITE(6,399)
      WRITE(6,55)(DEL(I),I=1,8)
      WRITE(6,101)
      WRITE(6,402)
      WRITE(6,55)(DEL(I),I=10,14),(DEL(I),I=16,20)
      WRITE(6,101)
      WRITE(6,403)
      WRITE(6,55)(DEL(I),I=21,25),(del(i),i=31,34)
      WRITE(6,101)
      WRITE(6,406)
      WRITE(6,20)(KEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      WRITE(6,101)
      WRITE(6,166)
      WRITE(6,66)
      WRITE(6,61) KSPA,NSPA,KSPB,NSPB
      WRITE(6,101)
      WRITE(6,707)
      WRITE(6,708)IFVC1,IFVC2,NLC,KO,KDISK,ISYM,nppl
      WRITE(6,101)
      write(6,917)
      write(6,912)nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      WRITE(6,101)
      write(6,171)
      write(6,170) jdphs,hjd0,period,dpdt,pshift
      WRITE(6,101)
      WRITE(6,12)
      WRITE(6,1)MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,perr0,dperdt,THE,
     $vunit,vfac
      WRITE(6,101)
      WRITE(6,205)
      WRITE(6,206) E,A,F1,F2,VGA,XINCL,GR1,GR2,nsp1,nsp2,abunin,iab
      WRITE(6,101)
      WRITE(6,54)
      WRITE(6,408) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,xbol1,xbol2,ybol1,
     $ybol2
      IF(NVC.EQ.0) GOTO 196
      WRITE(6,101)
      WRITE(6,111)
      DO 91 I=1,NVC
   91 WRITE(6,218)iband(I),HLA(I),CLA(I),X1A(I),X2A(I),y1a(i),y2a(i),
     $opsfa(i),sigma(i),wla(i)
  196 CONTINUE
      IF(NLVC.EQ.NVC) GOTO 197
      WRITE(6,101)
      WRITE(6,11)
      DO 92 I=NVCP,NLVC
   92 write(6,85)iband(i),hla(i),cla(i),x1a(i),x2a(i),y1a(i),y2a(i),
     $el3a(i),opsfa(i),noise(i),sigma(i),wla(i)
  197 CONTINUE
      WRITE(6,101)
      IF(NSTOT.GT.0) WRITE(6,983)
      DO 688 KP=1,2
      IF((NSP1+KP-1).EQ.0) GOTO 688
      IF((NSP2+(KP-2)**2).EQ.0) GOTO 688
      NSPOT=NSP1
      IF(KP.EQ.2) NSPOT=NSP2
      DO 687 I=1,NSPOT
  687 WRITE(6,684)KP,XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
  688 WRITE(6,101)
      if(ncl.eq.0) goto 1067
      write(6,69)
      do 68 i=1,ncl
   68 write(6,64) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i),dens(i)
      write(6,101)
 1067 continue
      WRITE(6,101)
      WRITE(6,9)
      DO 75 LCV=1,NLVC
      WRITE(6,101)
      IF(LCV.LE.NVC.and.jdphs.eq.2) WRITE(6,955)
      IF(LCV.GT.NVC.and.jdphs.eq.2) WRITE(6,10)
      IF(LCV.LE.NVC.and.jdphs.eq.1) WRITE(6,755)
      IF(LCV.GT.NVC.and.jdphs.eq.1) WRITE(6,756)
      DO 74 I=NS,7000
      ifirst=nppl*(i-1)+NY+1
      last=ifirst+nppl-1
      READ(5,2) (phjd(in),flux(in),wt(in),in=ifirst,last)
      WRITE(6,2) (phjd(in),flux(in),wt(in),in=ifirst,last)
      IF(phjd(ifirst).gt.-10000.d0) GOTO 74
      NI=-(phjd(ifirst)+10000.d0)
      NY=NY+NI
      NOBS=nppl*(I-NS-1)+NI
      GOTO 150
   74 CONTINUE
  150 NS=I-1
      LC1=LCV+1
   75 KNOBS(LC1)=NOBS+KNOBS(LCV)
      do 275 ijp=1,knobs(lc1)
      phas(ijp)=phjd(ijp)
      if(jdphs.eq.1) call jdph(phjd(ijp),0.d0,hjd0,period,dpdt,xjddum,
     $phas(ijp))
  275 continue
      matrix=31
      do 427 ima=1,30
  427 matrix=matrix-keep(ima)
      matrix=matrix+nlc*(5-keep(31)-keep(32)-keep(33)-keep(34)-keep(35))
      MAT=MATRIX-1
      EM=dfloat(MATRIX-15)
      KTR=.24d0*EM+2.2d0
      IF(EM.LE.1.5d0) KTR=1
      IF(EM.GT.12.d0) KTR=5
      NCOEFF=MATRIX*KNOBS(LC1)
      NMAT=MAT*KNOBS(LC1)
      NCOF=NCOEFF
      DO 63 J=1,NCOF
   63 HOLD(J)=0.d0
      IF(KOSQ.EQ.1) GOTO 71
      DO 416 IOBS=1,NCOEFF
  416 OBS(IOBS)=0.d0
      KSMAX=37
      KSSMAX=37
      IF(E.EQ.0.d0) KSMAX=1
      IF(E.NE.0.d0) KSSMAX=1
      DO 419 KSS=1,KSSMAX
      DO 417 IB=1,NLVC
      VC1=0.d0
      VC2=0.d0
      ELIT=0.d0
      IF(IB.GT.NVC) ELIT=1.d0
      IF(IB.EQ.IFVC1) VC1=1.d0
      IF(IB.EQ.(IFVC2*(1+IFVC1))) VC2=1.d0
      IST=KNOBS(IB)+1
      IB1=IB+1
      ISP=KNOBS(IB1)
      DO 418 IX=IST,ISP
      DO 420 KS=1,KSMAX
      hjd=phjd(ix)
      dtime=hjd-hjd0
      IRTE=0
      IRVOL1=0
      IRVOL2=0
      IF(E.NE.0.d0) GOTO 297
      IF(IX.NE.IST) IRTE=1
      IF(IX.EQ.ist) GOTO 297
      IRVOL1=1
      IRVOL2=1
  297 CONTINUE
      KSR=KS
      IF(E.EQ.0.d0) KSR=KSS
      if(e.ne.0.d0) goto 1110
      IF(ISYM.NE.1) GOTO 1110
      IF(KSR.EQ.1) GOTO 420
 1110 CONTINUE
      KH=KSR-2
      IF(KH.GT.0) GOTO 740
      KH=1
      GOTO 941
  740 CONTINUE
      if(ifap*kh.eq.11) goto 842
      if(ifphi*kh.eq.14) goto 842
      IF(KEEP(KH).EQ.1) GOTO 420
  842 IF(E.EQ.0.d0) GOTO 889
      IF(KSR.LE.2) GOTO 889
      IF(KH.LE.9) IRTE=1
      IF(KH.LE.9) IRVOL1=1
      IF(KH.LE.9) IRVOL2=1
      IF(KH.EQ.12) IRVOL2=1
      IF(KH.EQ.13) IRVOL1=1
      IF(KH.EQ.15) IRTE=1
      IF(KH.EQ.15) IRVOL1=1
      IF(KH.EQ.15) IRVOL2=1
      IF(KH.EQ.16) IRTE=1
      IF(KH.EQ.16) IRVOL1=1
      IF(KH.EQ.16) IRVOL2=1
      IF(KH.EQ.17) IRVOL2=1
      IF(KH.EQ.18) IRVOL1=1
      IF(KH.EQ.19) IRVOL1=1
      IF(KH.EQ.19) IRVOL2=1
      IF(KH.EQ.20) IRVOL1=1
      IF(KH.EQ.20) IRVOL2=1
      IF(KH.EQ.21) IRVOL1=1
      IF(KH.EQ.21) IRVOL2=1
      IF(KH.EQ.22) IRVOL1=1
      IF(KH.EQ.22) IRVOL2=1
      IF(KH.EQ.23) IRVOL2=1
      IF(KH.EQ.24) IRVOL1=1
      IF(KH.GE.31) IRVOL1=1
      IF(KH.GE.31) IRVOL2=1
  889 CONTINUE
      LCF=0
      IF(KH.GT.30) LCF=IB-NVC
      KPCT1=0
      KPCT2=0
      KSP=KH
      IF(KH.GT.30) KSP=30
      IF(KH.LT.2) GOTO 808
      DO 804 ICT=1,KSP
  804 KPCT1=KPCT1+1-KEEP(ICT)
      GOTO 809
  808 KPCT1=1
  809 CONTINUE
      IF(KH.LT.31) GOTO 806
      DO 805 ICT=31,KH
  805 KPCT2=KPCT2+1-KEEP(ICT)
      GOTO 807
  806 KPCT2=1
  807 CONTINUE
      II=(KPCT1+NLC*(KPCT2-1)+LCF-1)*KNOBS(LC1)+IX
      IF(KH.EQ.9) GOTO 300
      IF(KH.EQ.15) GOTO 308
      if(kh.eq.26) goto 844
      if(kh.eq.27) goto 845
      if(kh.eq.28) goto 846
      if(kh.eq.29) goto 847
      IF(KH.EQ.35) GOTO 301
      IF(KH.NE.31) GOTO 941
      IF(MODE.LE.0) GOTO 941
      IF(IPB.EQ.1) GOTO 941
      IF(IB.GT.NVC) OBS(II)=(BR(IX)-EL3A(IB))/HLA(IB)
      GOTO 420
  941 CONTINUE
      DL=DEL(KH)
      IF(ISYM.EQ.1) DL=.5d0*DEL(KH)
      SIGN=1.d0
      ISS=1
      DO 421 IH=1,35
  421 DLIF(IH)=0.d0
      IF(KSR.LE.2) GOTO 777
      ISS=IS
      DLIF(KH)=1.d0
  777 CONTINUE
      DO 319 IL=1,ISS
      IF(E.NE.0.d0) GOTO 4011
      IF(ISYM.EQ.1.and.ix.ne.ist) GOTO 4012
      GOTO 940
 4011 IF(KSR.LE.2) GOTO 940
      goto 4014
 4012 IF(IL.EQ.2) GOTO 4016
 4014 IF(LOW(KH).EQ.1) GOTO 314
      VOL1=SVOL1
      VOL2=SVOL2
      SUMM1=SSUM1
      SUMM2=SSUM2
      SM1=SSM1
      SM2=SSM2
      DO 851 IRE=1,MTLH
  851 TLDH(IRE)=STLDH(IRE)
      DO 508 IRE=1,M1H
      RV(IRE)=SRV(IRE)
      GRX(IRE)=SGRX(IRE)
      GRY(IRE)=SGRY(IRE)
      GRZ(IRE)=SGRZ(IRE)
      GLUMP1(IRE)=SGLM1(IRE)
      GRV1(IRE)=SGRV1(IRE)
      XX1(IRE)=SXX1(IRE)
      YY1(IRE)=SYY1(IRE)
      ZZ1(IRE)=SZZ1(IRE)
      GMAG1(IRE)=SGMG1(IRE)
      CSBT1(IRE)=SCSB1(IRE)
      RF1(IRE)=SRF1(IRE)
      FR1(IRE)=SFR1(IRE)
  508 SLUMP1(IRE)=SLMP1(IRE)
      DO 309 IRE=1,M2H
      RVQ(IRE)=SRVQ(IRE)
      GRXQ(IRE)=SGRXQ(IRE)
      GRYQ(IRE)=SGRYQ(IRE)
      GRZQ(IRE)=SGRZQ(IRE)
      GLUMP2(IRE)=SGLM2(IRE)
      GRV2(IRE)=SGRV2(IRE)
      XX2(IRE)=SXX2(IRE)
      YY2(IRE)=SYY2(IRE)
      ZZ2(IRE)=SZZ2(IRE)
      GMAG2(IRE)=SGMG2(IRE)
      CSBT2(IRE)=SCSB2(IRE)
      RF2(IRE)=SRF2(IRE)
      FR2(IRE)=SFR2(IRE)
  309 SLUMP2(IRE)=SLMP2(IRE)
      GOTO 940
 4016 IF(LOW(KH).EQ.1) GOTO 4018
      VOL1=EVOL1
      VOL2=EVOL2
      SUMM1=ESUM1
      SUMM2=ESUM2
      SM1=ESM1
      SM2=ESM2
      DO 852 IRE=1,MTLH
  852 TLDH(IRE)=ETLDH(IRE)
      DO 1508 IRE=1,M1H
      RV(IRE)=ERV(IRE)
      GRX(IRE)=EGRX(IRE)
      GRY(IRE)=EGRY(IRE)
      GRZ(IRE)=EGRZ(IRE)
      GLUMP1(IRE)=EGLM1(IRE)
      GRV1(IRE)=EGRV1(IRE)
      XX1(IRE)=EXX1(IRE)
      YY1(IRE)=EYY1(IRE)
      ZZ1(IRE)=EZZ1(IRE)
      GMAG1(IRE)=EGMG1(IRE)
      CSBT1(IRE)=ECSB1(IRE)
      RF1(IRE)=ERF1(IRE)
      FR1(IRE)=EFR1(IRE)
 1508 SLUMP1(IRE)=ELMP1(IRE)
      DO 1309 IRE=1,M2H
      RVQ(IRE)=ERVQ(IRE)
      GRXQ(IRE)=EGRXQ(IRE)
      GRYQ(IRE)=EGRYQ(IRE)
      GRZQ(IRE)=EGRZQ(IRE)
      GLUMP2(IRE)=EGLM2(IRE)
      GRV2(IRE)=EGRV2(IRE)
      XX2(IRE)=EXX2(IRE)
      YY2(IRE)=EYY2(IRE)
      ZZ2(IRE)=EZZ2(IRE)
      GMAG2(IRE)=EGMG2(IRE)
      CSBT2(IRE)=ECSB2(IRE)
      RF2(IRE)=ERF2(IRE)
      FR2(IRE)=EFR2(IRE)
 1309 SLUMP2(IRE)=ELMP2(IRE)
      GOTO 940
 4018 CONTINUE
      VOL1=EVOL1L
      VOL2=EVOL2L
      SUMM1=ESUM1L
      SUMM2=ESUM2L
      SM1=ESM1L
      SM2=ESM2L
      DO 853 IRE=1,MTLL
  853 TLDL(IRE)=ETLDL(IRE)
      DO 310 IRE=1,M1L
      RV(IRE)=ERVL(IRE)
      GRX(IRE)=EGRXL(IRE)
      GRY(IRE)=EGRYL(IRE)
      GRZ(IRE)=EGRZL(IRE)
      GLUMP1(IRE)=EGLM1L(IRE)
      GRV1(IRE)=EGRV1L(IRE)
      XX1(IRE)=EXX1L(IRE)
      YY1(IRE)=EYY1L(IRE)
      ZZ1(IRE)=EZZ1L(IRE)
      GMAG1(IRE)=EGMG1L(IRE)
      CSBT1(IRE)=ECSB1L(IRE)
      RF1(IRE)=ERF1L(IRE)
      FR1(IRE)=EFR1L(IRE)
  310 SLUMP1(IRE)=ELMP1L(IRE)
      DO 311 IRE=1,M2L
      RVQ(IRE)=ERVQL(IRE)
      GRXQ(IRE)=EGRXQL(IRE)
      GRYQ(IRE)=EGRYQL(IRE)
      GRZQ(IRE)=EGRZQL(IRE)
      GLUMP2(IRE)=EGLM2L(IRE)
      GRV2(IRE)=EGRV2L(IRE)
      XX2(IRE)=EXX2L(IRE)
      YY2(IRE)=EYY2L(IRE)
      ZZ2(IRE)=EZZ2L(IRE)
      GMAG2(IRE)=EGMG2L(IRE)
      CSBT2(IRE)=ECSB2L(IRE)
      RF2(IRE)=ERF2L(IRE)
      FR2(IRE)=EFR2L(IRE)
  311 SLUMP2(IRE)=ELMP2L(IRE)
      GOTO 940
  314 CONTINUE
      VOL1=SVOL1L
      VOL2=SVOL2L
      SUMM1=SSUM1L
      SUMM2=SSUM2L
      SM1=SSM1L
      SM2=SSM2L
      DO 854 IRE=1,MTLL
  854 TLDL(IRE)=STLDL(IRE)
      DO 1310 IRE=1,M1L
      RV(IRE)=SRVL(IRE)
      GRX(IRE)=SGRXL(IRE)
      GRY(IRE)=SGRYL(IRE)
      GRZ(IRE)=SGRZL(IRE)
      GLUMP1(IRE)=SGLM1L(IRE)
      GRV1(IRE)=SGRV1L(IRE)
      XX1(IRE)=SXX1L(IRE)
      YY1(IRE)=SYY1L(IRE)
      ZZ1(IRE)=SZZ1L(IRE)
      GMAG1(IRE)=SGMG1L(IRE)
      CSBT1(IRE)=SCSB1L(IRE)
      RF1(IRE)=SRF1L(IRE)
      FR1(IRE)=SFR1L(IRE)
 1310 SLUMP1(IRE)=SLMP1L(IRE)
      DO 1311 IRE=1,M2L
      RVQ(IRE)=SRVQL(IRE)
      GRXQ(IRE)=SGRXQL(IRE)
      GRYQ(IRE)=SGRYQL(IRE)
      GRZQ(IRE)=SGRZQL(IRE)
      GLUMP2(IRE)=SGLM2L(IRE)
      GRV2(IRE)=SGRV2L(IRE)
      XX2(IRE)=SXX2L(IRE)
      YY2(IRE)=SYY2L(IRE)
      ZZ2(IRE)=SZZ2L(IRE)
      GMAG2(IRE)=SGMG2L(IRE)
      CSBT2(IRE)=SCSB2L(IRE)
      RF2(IRE)=SRF2L(IRE)
      FR2(IRE)=SFR2L(IRE)
 1311 SLUMP2(IRE)=SLMP2L(IRE)
  940 CONTINUE
      DELS=DL*SIGN
      SIGN=-1.d0
      IF(NSPA.EQ.0) GOTO 470
      xlt=xlat(kspa,nspa)+dels*dlif(1)
      xlng(kspa,nspa)=xlong(kspa,nspa)+dels*dlif(2)
      snlat(kspa,nspa)=dsin(xlt)
      cslat(kspa,nspa)=dcos(xlt)
      snlng(kspa,nspa)=dsin(xlng(kspa,nspa))
      cslng(kspa,nspa)=dcos(xlng(kspa,nspa))
      RDSP(KSPA,NSPA)=RADSP(KSPA,NSPA)+DELS*DLIF(3)
      TMSP(KSPA,NSPA)=TEMSP(KSPA,NSPA)+DELS*DLIF(4)
  470 CONTINUE
      IF(NSPB.EQ.0) GOTO 471
      xlt=xlat(kspb,nspb)+dels*dlif(5)
      xlng(kspb,nspb)=xlong(kspb,nspb)+dels*dlif(6)
      snlat(kspb,nspb)=dsin(xlt)
      cslat(kspb,nspb)=dcos(xlt)
      snlng(kspb,nspb)=dsin(xlng(kspb,nspb))
      cslng(kspb,nspb)=dcos(xlng(kspb,nspb))
      RDSP(KSPB,NSPB)=RADSP(KSPB,NSPB)+DELS*DLIF(7)
      TMSP(KSPB,NSPB)=TEMSP(KSPB,NSPB)+DELS*DLIF(8)
  471 CONTINUE
      EC=E+DELS*DLIF(10)
      PERT=perr0+DELS*DLIF(11)
      FF1=F1+DELS*DLIF(12)
      FF2=F2+DELS*DLIF(13)
      PSH=PSHIFT+DELS*DLIF(14)
      XINC=XINCL+DELS*DLIF(16)
      G1=GR1+DELS*DLIF(17)
      G2=GR2+DELS*DLIF(18)
      T1=TAVH+DELS*DLIF(19)
      T2=TAVC+DELS*DLIF(20)
      A1=ALB1+DELS*DLIF(21)
      A2=ALB2+DELS*DLIF(22)
      POT1=PHSV+DELS*DLIF(23)
      POT2=PCSV+DELS*DLIF(24)
      RMASS=RM+DELS*DLIF(25)
      HL=HLA(IB)+DELS*DLIF(31)
      CL=CLA(IB)+DELS*DLIF(32)
      X1=X1A(IB)+DELS*DLIF(33)
      X2=X2A(IB)+DELS*DLIF(34)
      y1=y1a(ib)
      y2=y2a(ib)
      opsf=opsfa(ib)
      IF(KSR.EQ.1) GOTO 802
      IF(KSR.EQ.2) GOTO 872
      IF(LOW(KH).EQ.1) GOTO 802
  872 CALL MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $RMASS,POT1,POT2,G1,G2,A1,A2,N1,N2,FF1,FF2,MOD,XINC,THE,MODE,
     $SNTHH,CSTHH,SNFIH,CSFIH,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1,
     $GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POT1,POT2,N1,N2,FF1,FF2,d,hl,cl,x1,
     $x2,y1,y2,g1,g2,wla(ib),sm1,sm2,tph,tpc,sbrh,sbrc,t1,
     $t2,a1,a2,xbol1,xbol2,ybol1,ybol2,phas(ix),rmass,xinc,hot,cool,
     $snthh,csthh,snfih,csfih,tldh,glump1,glump2,xx1,xx2,yy1,yy2,zz1,
     $zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,gmag2,
     $glog1,glog2,
     $fbin1,fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,resf2,
     $wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,
     $rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,zskp,mode,iband(ib),
     $ifat1,ifat2,1)
      GOTO 801
  802 CONTINUE
      CALL MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVL,FR1,FR2,HLD,
     $RMASS,POT1,POT2,G1,G2,A1,A2,N1L,N2L,FF1,FF2,MOD,XINC,THE,MODE,
     $SNTHL,CSTHL,SNFIL,CSFIL,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1,
     $GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVL,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POT1,POT2,N1L,N2L,FF1,FF2,d,hl,cl,
     $x1,x2,y1,y2,g1,g2,wla(ib),sm1,sm2,tph,tpc,sbrh,sbrc,
     $t1,t2,a1,a2,xbol1,xbol2,ybol1,ybol2,phas(ix),rmass,xinc,hot,cool,
     $snthl,csthl,snfil,csfil,tldl,glump1,glump2,xx1,xx2,yy1,yy2,zz1,
     $zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,gmag2,
     $glog1,glog2,
     $fbin1,fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,resf2,
     $wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,
     $rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,zskp,mode,iband(ib),
     $ifat1,ifat2,1)
  801 CONTINUE
      IF(E.NE.0.d0) GOTO 4111
      IF(ISYM.EQ.0) GOTO 602
      IF(IX.NE.IST) GOTO 602
      IF(KSR.EQ.1) GOTO 602
      IF(KSR.EQ.2) GOTO 4119
      IF(LOW(KH).EQ.1) GOTO 4112
      GOTO 4119
 4111 IF(IRTE.EQ.1) GOTO 602
      IF(KSR.GE.3) GOTO 602
      IF(KSR.EQ.2) GOTO 4119
 4112 IF(IL.EQ.2) GOTO 4116
      SVOL1L=VOL1
      SVOL2L=VOL2
      SSUM1L=SUMM1
      SSUM2L=SUMM2
      SSM1L=SM1
      SSM2L=SM2
      DO 855 IHL=1,MTLL
  855 STLDL(IHL)=TLDL(IHL)
      DO 603 IHL=1,M1L
      SRVL(IHL)=RV(IHL)
      SGRXL(IHL)=GRX(IHL)
      SGRYL(IHL)=GRY(IHL)
      SGRZL(IHL)=GRZ(IHL)
      SLMP1L(IHL)=SLUMP1(IHL)
      SGLM1L(IHL)=GLUMP1(IHL)
      SGRV1L(IHL)=GRV1(IHL)
      SXX1L(IHL)=XX1(IHL)
      SYY1L(IHL)=YY1(IHL)
      SZZ1L(IHL)=ZZ1(IHL)
      SGMG1L(IHL)=GMAG1(IHL)
      SCSB1L(IHL)=CSBT1(IHL)
      SRF1L(IHL)=RF1(IHL)
      SFR1L(IHL)=FR1(IHL)
  603 CONTINUE
      DO 606 IHL=1,M2L
      SRVQL(IHL)=RVQ(IHL)
      SGRXQL(IHL)=GRXQ(IHL)
      SGRYQL(IHL)=GRYQ(IHL)
      SGRZQL(IHL)=GRZQ(IHL)
      SLMP2L(IHL)=SLUMP2(IHL)
      SGLM2L(IHL)=GLUMP2(IHL)
      SGRV2L(IHL)=GRV2(IHL)
      SXX2L(IHL)=XX2(IHL)
      SYY2L(IHL)=YY2(IHL)
      SZZ2L(IHL)=ZZ2(IHL)
      SGMG2L(IHL)=GMAG2(IHL)
      SCSB2L(IHL)=CSBT2(IHL)
      SRF2L(IHL)=RF2(IHL)
      SFR2L(IHL)=FR2(IHL)
  606 CONTINUE
      GOTO 602
 4116 CONTINUE
      EVOL1L=VOL1
      EVOL2L=VOL2
      ESUM1L=SUMM1
      ESUM2L=SUMM2
      ESM1L=SM1
      ESM2L=SM2
      DO 856 IHL=1,MTLL
  856 ETLDL(IHL)=TLDL(IHL)
      DO 1603 IHL=1,M1L
      ERVL(IHL)=RV(IHL)
      EGRXL(IHL)=GRX(IHL)
      EGRYL(IHL)=GRY(IHL)
      EGRZL(IHL)=GRZ(IHL)
      ELMP1L(IHL)=SLUMP1(IHL)
      EGLM1L(IHL)=GLUMP1(IHL)
      EGRV1L(IHL)=GRV1(IHL)
      EXX1L(IHL)=XX1(IHL)
      EYY1L(IHL)=YY1(IHL)
      EZZ1L(IHL)=ZZ1(IHL)
      EGMG1L(IHL)=GMAG1(IHL)
      ECSB1L(IHL)=CSBT1(IHL)
      ERF1L(IHL)=RF1(IHL)
      EFR1L(IHL)=FR1(IHL)
 1603 CONTINUE
      DO 1606 IHL=1,M2L
      ERVQL(IHL)=RVQ(IHL)
      EGRXQL(IHL)=GRXQ(IHL)
      EGRYQL(IHL)=GRYQ(IHL)
      EGRZQL(IHL)=GRZQ(IHL)
      ELMP2L(IHL)=SLUMP2(IHL)
      EGLM2L(IHL)=GLUMP2(IHL)
      EGRV2L(IHL)=GRV2(IHL)
      EXX2L(IHL)=XX2(IHL)
      EYY2L(IHL)=YY2(IHL)
      EZZ2L(IHL)=ZZ2(IHL)
      EGMG2L(IHL)=GMAG2(IHL)
      ECSB2L(IHL)=CSBT2(IHL)
      ERF2L(IHL)=RF2(IHL)
      EFR2L(IHL)=FR2(IHL)
 1606 CONTINUE
      GOTO 602
 4119 IF(IL.EQ.2) GOTO 4120
      SVOL1=VOL1
      SVOL2=VOL2
      SSUM1=SUMM1
      SSUM2=SUMM2
      SSM1=SM1
      SSM2=SM2
      DO 857 IHH=1,MTLH
  857 STLDH(IHH)=TLDH(IHH)
      DO 601 IHH=1,M1H
      SRV(IHH)=RV(IHH)
      SGRX(IHH)=GRX(IHH)
      SGRY(IHH)=GRY(IHH)
      SGRZ(IHH)=GRZ(IHH)
      SLMP1(IHH)=SLUMP1(IHH)
      SGLM1(IHH)=GLUMP1(IHH)
      SGRV1(IHH)=GRV1(IHH)
      SXX1(IHH)=XX1(IHH)
      SYY1(IHH)=YY1(IHH)
      SZZ1(IHH)=ZZ1(IHH)
      SGMG1(IHH)=GMAG1(IHH)
      SCSB1(IHH)=CSBT1(IHH)
      SRF1(IHH)=RF1(IHH)
      SFR1(IHH)=FR1(IHH)
  601 CONTINUE
      DO 605 IHH=1,M2H
      SRVQ(IHH)=RVQ(IHH)
      SGRXQ(IHH)=GRXQ(IHH)
      SGRYQ(IHH)=GRYQ(IHH)
      SGRZQ(IHH)=GRZQ(IHH)
      SLMP2(IHH)=SLUMP2(IHH)
      SGLM2(IHH)=GLUMP2(IHH)
      SGRV2(IHH)=GRV2(IHH)
      SXX2(IHH)=XX2(IHH)
      SYY2(IHH)=YY2(IHH)
      SZZ2(IHH)=ZZ2(IHH)
      SGMG2(IHH)=GMAG2(IHH)
      SCSB2(IHH)=CSBT2(IHH)
      SRF2(IHH)=RF2(IHH)
      SFR2(IHH)=FR2(IHH)
  605 CONTINUE
      GOTO 602
 4120 CONTINUE
      EVOL1=VOL1
      EVOL2=VOL2
      ESUM1=SUMM1
      ESUM2=SUMM2
      ESM1=SM1
      ESM2=SM2
      DO 858 IHH=1,MTLH
  858 ETLDH(IHH)=TLDH(IHH)
      DO 1601 IHH=1,M1H
      ERV(IHH)=RV(IHH)
      EGRX(IHH)=GRX(IHH)
      EGRY(IHH)=GRY(IHH)
      EGRZ(IHH)=GRZ(IHH)
      ELMP1(IHH)=SLUMP1(IHH)
      EGLM1(IHH)=GLUMP1(IHH)
      EGRV1(IHH)=GRV1(IHH)
      EXX1(IHH)=XX1(IHH)
      EYY1(IHH)=YY1(IHH)
      EZZ1(IHH)=ZZ1(IHH)
      EGMG1(IHH)=GMAG1(IHH)
      ECSB1(IHH)=CSBT1(IHH)
      ERF1(IHH)=RF1(IHH)
      EFR1(IHH)=FR1(IHH)
 1601 CONTINUE
      DO 1605 IHH=1,M2H
      ERVQ(IHH)=RVQ(IHH)
      EGRXQ(IHH)=GRXQ(IHH)
      EGRYQ(IHH)=GRYQ(IHH)
      EGRZQ(IHH)=GRZQ(IHH)
      ELMP2(IHH)=SLUMP2(IHH)
      EGLM2(IHH)=GLUMP2(IHH)
      EGRV2(IHH)=GRV2(IHH)
      EXX2(IHH)=XX2(IHH)
      EYY2(IHH)=YY2(IHH)
      EZZ2(IHH)=ZZ2(IHH)
      EGMG2(IHH)=GMAG2(IHH)
      ECSB2(IHH)=CSBT2(IHH)
      ERF2(IHH)=RF2(IHH)
      EFR2(IHH)=FR2(IHH)
 1605 CONTINUE
  602 CONTINUE
      HTT=HOT
      IF(MODE.EQ.-1) HTT=0.d0
      XR=(HTT+COOL+EL3A(IB))*ELIT+VKM1*VC1+VKM2*VC2
      IF(KSR.NE.1) GOTO 710
      BL(IX)=XR
      GOTO 420
  710 CONTINUE
      IF(KSR.NE.2) GOTO 711
      BR(IX)=XR
      II=NMAT+IX
      OBS(II)=FLUX(IX)-XR
      if(iss.eq.2) goto 319
      GOTO 420
  711 CONTINUE
      XLR=BR(IX)
      IF(LOW(KH).EQ.1) XLR=BL(IX)
      IF(IL.NE.2) GOTO 388
      XLR=XR
      GOTO 87
  388 XNR=XR
   87 khkeep=kh*keep(kh)
      if(khkeep.ne.11.and.khkeep.ne.14) OBS(II)=(XNR-XLR)/DEL(KH)
      if(kh.eq.11) dfdap(ix)=(xnr-xlr)/del(kh)
      if(kh.eq.14) dfdph(ix)=(xnr-xlr)/del(kh)
  319 CONTINUE
      GOTO 420
  300 IF(IB.LE.NVC) OBS(II)=(BR(IX)-VGA)/A
      GOTO 420
  308 IF(IB.LE.NVC) OBS(II)=1.d0
      GOTO 420
  844 obs(ii)=dfdph(ix)/(period+dtime*dpdt)
      goto 420
  845 brac=period+dtime*dpdt
      obs(ii)=dfdph(ix)*dtime/(brac*period)
      goto 420
  846 dis=dabs(dtime*dpdt/period)
      if(dis.gt.toldis) goto 848
      brac2=2.d0*period+dtime*dpdt
      dphpd=-2.d0*(dtime/brac2)**2
     $+tt*dtime**3*(2.d0*brac2*dpdt-3.d0*dpdt**2*dtime)/brac2**4
      goto 849
  848 brac=period+dtime*dpdt
      dphpd=dtime/(brac*dpdt)-(dlog(brac)-dlog(period))/dpdt**2
  849 obs(ii)=-dfdph(ix)*dphpd
      goto 420
  847 obs(ii)=dtime*dfdap(ix)
      goto 420
  301 IF(IB.GT.NVC) OBS(II)=1.d0
  420 CONTINUE
  418 CONTINUE
  417 CONTINUE
  419 CONTINUE
      write(6,101)
      write(6,101)
      write(6,101)
      write(6,159)
      write(6,101)
      write(6,169)
      do 298 icv=1,nlvc
      icvp=icv+1
      nbs=knobs(icvp)-knobs(icv)
      iw=knobs(icv)+1
      jstart=nmat+iw
      jstop=jstart+nbs-1
      resq=0.d0
      iww=iw-1
      do 299 jres=jstart,jstop
      iww=iww+1
  299 resq=resq+wt(iww)*obs(jres)**2
      write(6,119) icv,nbs,resq
c     Added for PHOEBE:
      chi2s(icv)=resq
c     --- up to here ---
  298 continue
      write(6,101)
      do 909 komp=1,2
      if(message(komp,1).eq.1) write(6,283) komp
      if(message(komp,2).eq.1) write(6,284) komp
      if(message(komp,3).eq.1) write(6,285) komp
      if(message(komp,4).eq.1) write(6,286) komp
  909  continue
      write(6,101)
      GOTO 65
   71 JF=0
c     IF(KO.EQ.0) stop
      IF(K0.EQ.0) goto 8999
      DO 261 J=1,NCOF
  261 OBS(J)=HOLD(J)
      IF(KDISK.EQ.0) GOTO 72
      REWIND 9
      READ(9,67)(OBS(J),J=1,NCOF)
   72 READ(5,20)(KEEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      if(keep(1).ne.2) goto 866
c     close (5)
c     close (6)
c     stop
      goto 8999
  866 continue
      DO 232 I=1,35
  232 IF(KEP(I).EQ.1) KEEP(I)=1
      NOBS=KNOBS(LC1)
      matrix=31
      do 428 ima=1,30
  428 matrix=matrix-keep(ima)
      matrix=matrix+nlc*(5-keep(31)-keep(32)-keep(33)-keep(34)-keep(35))
      MAT=MATRIX-1
      EM=dfloat(MATRIX-15)
      KTR=.24d0*EM+2.2d0
      IF(EM.LE.1.5d0) KTR=1
      IF(EM.GT.12.d0) KTR=5
      NCOEFF=MATRIX*NOBS
      KC=1
      NSHIFT(1)=0
      DO 59 I=2,36
      IF(I.GT.31) KC=NLC
      KE=0
      J=I-1
      IF(KEEP(J).GT.KEP(J)) KE=1
   59 NSHIFT(I)=NOBS*KE*KC+NSHIFT(J)
      NOBBS=NOBS
      DO 30 I=1,36
      IF(KEEP(I).EQ.1) GOTO 30
      IF(I.GT.30) NOBBS=NOBS*NLC
      IF(I.EQ.36) NOBBS=NOBS
      DO 32 J=1,NOBBS
      JF=JF+1
      KX=JF+NSHIFT(I)
   32 OBS(JF)=OBS(KX)
   30 CONTINUE
   65 continue
      WRITE(6,20)(KEEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      NOBS=KNOBS(LC1)
      WRITE(6,101)
      IF(IFDER.EQ.0) KTR=5
      WRITE(6,82)
      WRITE(6,101)
      DO 96 IB=1,NLVC
      IST=KNOBS(IB)+1
      IB1=IB+1
      ISP=KNOBS(IB1)
      DO 96 I=IST,ISP
      GOTO(5,6,7,8,96),KTR
    5 WRITE(6,15)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 96
    6 WRITE(6,16)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 96
    7 WRITE(6,17)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 96
    8 WRITE(6,19)(OBS(J),J=I,NCOEFF,NOBS)
   96 CONTINUE
      IF(KO.LE.1) GOTO 70
      IF(IBEF.EQ.1) GOTO 70
      DO 62 J=1,NCOEFF
   62 HOLD(J)=OBS(J)
      IF(KDISK.EQ.0) GOTO 73
      REWIND 9
      WRITE(9,67)(OBS(J),J=1,NCOEFF)
   73 CONTINUE
   70 WRITE(6,101)
      DO 97 IB=1,NLVC
      IST=KNOBS(IB)+1
      IB1=IB+1
      ISP=KNOBS(IB1)
      NOIS=NOISE(IB)
      DO 97 I=IST,ISP
      IF(IB.GT.NVC) GOTO 444
      ROOTWT=dsqrt(WT(I))/(100.d0*SIGMA(IB))
      GOTO 445
  444 ROOTWT=dsqrt(WT(I))/(100.d0*SIGMA(IB)*dsqrt(FLUX(I))**NOIS)
  445 CONTINUE
      DO 97 LOB=I,NCOEFF,NOBS
   97 OBS(LOB)=OBS(LOB)*ROOTWT
      IF(IFDER.NE.0) WRITE(6,83)
      IF(IFDER.NE.0) WRITE(6,101)
      DO 98 I=1,NOBS
      GOTO(45,46,47,48,98),KTR
   45 WRITE(6,15)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 98
   46 WRITE(6,16)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 98
   47 WRITE(6,17)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 98
   48 WRITE(6,19)(OBS(J),J=I,NCOEFF,NOBS)
   98 CONTINUE
      CALL square (OBS,NOBS,MAT,OUT,sd,xlamda,deter,CN,CNN,cnc,clc,
     $s,ccl,ll,mm)
      MSQ=MAT*MAT
      IF(IFM.EQ.0) GOTO 436
      WRITE(6,101)
      WRITE(6,181)
      WRITE(6,101)
      DO 38 JR=1,MAT
   38 WRITE(6,37) (CN(JX),JX=JR,MSQ,MAT),CCL(JR)
      WRITE(6,101)
      WRITE(6,183)
      WRITE(6,101)
  436 CONTINUE
      NO1=23
      NO2=24
      NRM=25
      DO 334 IRM=1,24
      IF(IRM.LE.23) NO1=NO1-KEEP(IRM)
      NO2=NO2-KEEP(IRM)
  334 NRM=NRM-KEEP(IRM)
      CORO1=1-KEEP(23)
      CORO2=1-KEEP(24)
      CORQ=1-KEEP(25)
      DO 34 JM=1,MAT
      DO 33 JQ=1,MAT
      JT=JM+MAT*(JQ-1)
      IJM=(MAT+1)*(JM-1)+1
      IJQ=(MAT+1)*(JQ-1)+1
   33 V(JQ)=CNN(JT)/DSQRT(CNN(IJM)*CNN(IJQ))
      co1q=0.d0
      co2q=0.d0
      if(jm.eq.nrm.and.no1.gt.0) co1q=v(no1)*corq*coro1
      if(jm.eq.nrm.and.no2.gt.0) co2q=v(no2)*corq*coro2
   34 WRITE(6,37)(V(IM),IM=1,MAT)
      IF(IFM.EQ.0) GOTO 36
      WRITE(6,101)
      WRITE(6,184)
      WRITE(6,101)
      CALL DGMPRD(CN,CNN,CNOUT,MAT,MAT,MAT)
      DO 116 J8=1,MAT
  116 WRITE(6,37)(CNOUT(J7),J7=J8,MSQ,MAT)
      WRITE(6,101)
      WRITE(6,185)
      WRITE(6,101)
      ANSCH=0.D0
      DO 118 J5=1,MAT
      V(J5)=0.D0
      DO 117 J6=1,MAT
      int=mat*(j6-1)
      idi=j6+int
      I9=J5+int
  117 V(J5)=OUT(J6)*CN(I9)*dsqrt(cnc(idi))+V(J5)
      ERR=V(J5)-CCL(J5)
  118 ANSCH=ANSCH+DABS(ERR)
      WRITE(6,137)(V(J4),J4=1,MAT)
      WRITE(6,101)
      WRITE(6,138) ANSCH
   36 CONTINUE
      WRITE(6,101)
      WRITE(6,101)
      write(6,715)
      WRITE(6,101)
      write(6,43)
      iout=0
      do 93 kpar=1,35
      imax=1
      if(kpar.gt.30) imax=nlc
      if(keep(kpar).eq.1) goto 93
      do 94 kurv=1,imax
      kcurv=kurv
      if(kpar.le.30) kcurv=0
      iout=iout+1
      ipar=kpar
      if(kpar.gt.30) ipar=30+kurv+(kpar-31)*nlc
      parout=para(ipar)+out(iout)
      write(6,615) kpar,kcurv,para(ipar),out(iout),parout,sd(iout)
   94 continue
   93 continue
      WRITE(6,101)
      WRITE(6,101)
      write(6,716)
      WRITE(6,101)
      write(6,43)
      iout=0
      do 53 kpar=1,35
      imax=1
      if(kpar.gt.30) imax=nlc
      if(keep(kpar).eq.1) goto 53
      do 52 kurv=1,imax
      kcurv=kurv
      if(kpar.le.30) kcurv=0
      iout=iout+1
      ipar=kpar
      if(kpar.gt.30) ipar=30+kurv+(kpar-31)*nlc
      parout=para(ipar)+out(iout)
      write(6,616) kpar,kcurv,para(ipar),out(iout),parout,sd(iout)
c     Added for PHOEBE:
      correction(iout)=out(iout)
      stddev(iout)=sd(iout)
c     --- up to here ---
   52 continue
   53 continue
      WRITE(6,101)
      RESSQ=0.d0
      JST=MAT*NOBS+1
      DO 199 JRES=JST,NCOEFF
  199 RESSQ=RESSQ+OBS(JRES)**2
c     Added for PHOEBE:
      chi2=RESSQ
c     --- up to here ---
      WRITE(6,101)
      WRITE(6,40)
      WRITE(6,21) RESSQ,S,deter
      IBEF=1
      IF(IFR.EQ.0) GOTO 71
      WRITE(6,101)
      WRITE(6,101)
      WRITE(6,650)
      WRITE(6,101)
      WRITE(6,101)
      WRITE(6,653)
      WRITE(6,101)
      DO1=sd(NO1)*CORO1
      DO2=sd(NO2)*CORO2
      if(mod.eq.1) do2=do1
      if(mod.eq.1) co2q=co1q
      DQ=sd(NRM)*CORQ
      COQ=CO1Q
      F=F1
      DP=1.d0-E
      OME=PHSV
      DOM=DO1
      KOMP=0
  925 CONTINUE
      KOMP=KOMP+1
      DO 926 KD=1,4
      if(kd.ne.2) goto 928
      if(po(komp).ge.omcr(komp)) goto 928
      goto 926
  928 continue
      TH=XTHA(KD)
      FI=XFIA(KD)
      CALL ROMQ(OME,RM,F,DP,E,TH,FI,R,DRDO,DRDQ,DODQ,KOMP,MODE)
      DR=dsqrt(DRDQ**2*DQ**2+DRDO**2*DOM**2+2.d0*COQ*DRDQ*DRDO*DQ*DOM)
      WRITE(6,654)KOMP,ARAD(KD),R,DRDO,DRDQ,DR
  926 CONTINUE
      DO2DQ=DODQ
      IF(KOMP.EQ.1)DO1DQ=DODQ
      COQ=CO2Q
      F=F2
      OME=PCSV
      DOM=DO2
      WRITE(6,101)
      IF(KOMP.EQ.1) GOTO 925
      WRITE(6,101)
      WRITE(6,651)
      IF(KOMP.EQ.2) WRITE(6,652)DO1DQ,DO2DQ,CO1Q,CO2Q,DO1,DO2,DQ
      GOTO 71
 8999 continue
      close (5)
      close (6)
      END

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
      END
