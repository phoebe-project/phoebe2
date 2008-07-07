      SUBROUTINE wrdci(FN,DEL,KEP,IFDER,IFM,IFR,XLAMDA,KSPA,NSPA,KSPB,
     +NSPB,IFVC1,IFVC2,NLC,K0,KDISK,ISYM,NPPL,NREF,MREF,IFSMV1,IFSMV2,
     +ICOR1,ICOR2,LD,JDPHS,HJD0,PERIOD,DPDT,PSHIFT,MODE,IPB,IFAT1,IFAT2,
     +N1,N2,N1L,N2L,PERR0,DPERDT,THE,VUNIT,E,A,F1,F2,VGA,XINCL,GR1,GR2,
     +ABUNIN,TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,YBOL2,
     +IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,NOISE,SIGMA,WLA,NSP1,XLAT1,
     +XLONG1,RADSP1,TEMSP1,NSP2,XLAT2,XLONG2,RADSP2,TEMSP2,KNOBS,INDEP,
     +DEP,WEIGHT)

      implicit none

      double precision spend,clend,wt
      integer dciend

      parameter ( spend=   300.0)
      parameter ( clend=   150.0)
      parameter (dciend=       2)

      integer i,j

      character FN*(*)
      integer IFDER,IFM,IFR,KSPA,NSPA,KSPB,NSPB,IFVC1,IFVC2,NLC,K0,
     +        KDISK,ISYM,NPPL,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD,
     +        JDPHS,MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,NSP1,NSP2
      integer KEP(*),IBAND(*),NOISE(*),KNOBS(*)
      double precision XLAMDA,HJD0,PERIOD,DPDT,PSHIFT,PERR0,DPERDT,THE,
     +        VUNIT,E,A,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN,TAVH,TAVC,ALB1,
     +        ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,YBOL2
      double precision DEL(*),WLA(*),HLA(*),CLA(*),X1A(*),X2A(*),Y1A(*),
     +        Y2A(*),OPSF(*),SIGMA(*),EL3(*),XLAT1(*),XLONG1(*),
     +        RADSP1(*),TEMSP1(*),XLAT2(*),XLONG2(*),RADSP2(*),
     +        TEMSP2(*),INDEP(*),DEP(*),WEIGHT(*)

    1 format(10(1X,D7.1))
    2 format(1X,2(4I1,1X),7I1,1X,4(5I1,1X),I1,1X,I1,1X,I1,D10.3)
    3 format(4I3)
    4 format(I1,1X,I1,1X,5I2)
    5 format(7(I1,1X))
    6 format(I1,F15.6,D17.10,D14.6,F10.4)
    7 format(4I2,4I4,F13.6,D12.5,F8.5,F9.3)
    8 format(F6.5,D13.6,2F10.4,F10.4,F9.3,2F7.3,F7.2)
    9 format(F7.4,F8.4,2F7.3,3D13.6,4F7.3)
   10 format(I3,2F10.5,4F7.3,D10.3,D12.5,F10.6)
   11 format(I3,2F10.5,4F7.3,F8.4,D10.3,I2,D12.5,F10.6)
   12 format(4F9.5)
   13 format(1X,F4.0)
   14 format(F4.0)
   15 format(5(F14.5,F8.4,F6.2))
   16 format(I2)

      open(unit=1, file=FN, status='UNKNOWN')

      write(1,1) DEL( 1),DEL( 2),DEL( 3),DEL( 4),DEL( 5),DEL( 6),
     +           DEL( 7),DEL( 8)
      write(1,1) DEL(10),DEL(11),DEL(12),DEL(13),DEL(14),DEL(16),
     +           DEL(17),DEL(18),DEL(19),DEL(20)
      write(1,1) DEL(21),DEL(22),DEL(23),DEL(24),DEL(25),DEL(31),
     +           DEL(32),DEL(33),DEL(34)
      write(1,2) (KEP(i),i=1,35),IFDER,IFM,IFR,XLAMDA
      write(1,3) KSPA,NSPA,KSPB,NSPB
      write(1,4) IFVC1,IFVC2,NLC,K0,KDISK,ISYM,NPPL
      write(1,5) NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD
      write(1,6) JDPHS,HJD0,PERIOD,DPDT,PSHIFT
      write(1,7) MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,PERR0,DPERDT,THE,
     +           VUNIT
      write(1,8) E,A,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN
      write(1,9) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,
     +           YBOL2

      do 90, i=1,ifvc1+ifvc2
        write(1,10) IBAND(i),HLA(i),CLA(i),X1A(i),X2A(i),Y1A(i),Y2A(i),
     +              OPSF(i),SIGMA(i),WLA(i)
   90 continue

      do 91, i=ifvc1+ifvc2+1,ifvc1+ifvc2+nlc
        write(1,11) IBAND(i),HLA(i),CLA(i),X1A(i),X2A(i),Y1A(i),Y2A(i),
     +              EL3(i),OPSF(i),NOISE(i),SIGMA(i),WLA(i)
   91 continue

      do 92, i=1,NSP1
        write(1,12) XLAT1(i),XLONG1(i),RADSP1(i),TEMSP1(i)
   92 continue
      write(1,13) spend
      do 93, i=1,NSP2
        write(1,12) XLAT2(i),XLONG2(i),RADSP2(i),TEMSP2(i)
   93 continue
      write(1,13) spend

      write(1,14) clend

      do 95, j=1,ifvc1+ifvc2+nlc
      do 94, i=knobs(j)+1,knobs(j+1)
          wt=weight(i)
          if(weight(i).gt.99.9) wt=-1.d0
          write(1,15) INDEP(i),DEP(i),wt
   94 continue
      write(1,15) -10001.d0, 0.d0, 0.d0
   95 continue

      write(1,16) dciend

      close(unit=1)

      END
