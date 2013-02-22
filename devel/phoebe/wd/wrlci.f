      SUBROUTINE wrlci(FN,MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD,
     +JDPHS,HJD0,PERIOD,DPDT,PSHIFT,STDDEV,NOISE,SEED,JDSTRT,JDEND,
     +JDINC,PHSTRT,PHEND,PHINC,PHNORM,MODE,IPB,IFAT1,IFAT2,N1,N2,PERR0,
     +DPERDT,THE,VUNIT,E,SMA,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN,TAVH,TAVC,
     +ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,YBOL2,IBAND,HLA,CLA,X1A,
     +X2A,Y1A,Y2A,EL3,OPSF,MZERO,FACTOR,WLA,NSP1,XLAT1,XLONG1,RADSP1,
     +TEMSP1,NSP2,XLAT2,XLONG2,RADSP2,TEMSP2)
Cf2py intent(in) FN
Cf2py intent(in) MPAGE
Cf2py intent(in) NREF
Cf2py intent(in) MREF
Cf2py intent(in) IFSMV1
Cf2py intent(in) IFSMV2
Cf2py intent(in) ICOR1
Cf2py intent(in) ICOR2
Cf2py intent(in) LD
Cf2py intent(in) JDPHS
Cf2py intent(in) HJD0
Cf2py intent(in) PERIOD
Cf2py intent(in) DPDT
Cf2py intent(in) PSHIFT
Cf2py intent(in) STDDEV
Cf2py intent(in) NOISE
Cf2py intent(in) SEED
Cf2py intent(in) JDSTRT
Cf2py intent(in) JDEND
Cf2py intent(in) JDINC
Cf2py intent(in) PHSTRT
Cf2py intent(in) PHEND
Cf2py intent(in) PHINC
Cf2py intent(in) PHNORM
Cf2py intent(in) MODE
Cf2py intent(in) IPB
Cf2py intent(in) IFAT1
Cf2py intent(in) IFAT2
Cf2py intent(in) N1
Cf2py intent(in) N2
Cf2py intent(in) PERR0
Cf2py intent(in) DPERDT
Cf2py intent(in) THE
Cf2py intent(in) VUNIT
Cf2py intent(in) E
Cf2py intent(in) SMA
Cf2py intent(in) F1
Cf2py intent(in) F2
Cf2py intent(in) VGA
Cf2py intent(in) XINCL
Cf2py intent(in) GR1
Cf2py intent(in) GR2
Cf2py intent(in) ABUNIN
Cf2py intent(in) TAVH
Cf2py intent(in) TAVC
Cf2py intent(in) ALB1
Cf2py intent(in) ALB2
Cf2py intent(in) PHSV
Cf2py intent(in) PCSV
Cf2py intent(in) RM
Cf2py intent(in) XBOL1
Cf2py intent(in) XBOL2
Cf2py intent(in) YBOL1
Cf2py intent(in) YBOL2
Cf2py intent(in) IBAND
Cf2py intent(in) HLA
Cf2py intent(in) CLA
Cf2py intent(in) X1A
Cf2py intent(in) X2A
Cf2py intent(in) Y1A
Cf2py intent(in) Y2A
Cf2py intent(in) EL3
Cf2py intent(in) OPSF
Cf2py intent(in) MZERO
Cf2py intent(in) FACTOR
Cf2py intent(in) WLA
Cf2py intent(in) NSP1
Cf2py intent(in) XLAT1
Cf2py intent(in) XLONG1
Cf2py intent(in) RADSP1
Cf2py intent(in) TEMSP1
Cf2py intent(in) NSP2
Cf2py intent(in) XLAT2
Cf2py intent(in) XLONG2
Cf2py intent(in) RADSP2
Cf2py intent(in) TEMSP2

      implicit none

      double precision spend,clend
      integer lciend

      parameter ( spend=300.0)
      parameter ( clend=150.0)
      parameter (lciend=    9)

      integer i

      character FN*(*)
      integer MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD,JDPHS,NOISE,
     +MODE,IPB,IFAT1,IFAT2,N1,N2,IBAND,NSP1,NSP2
      double precision HJD0,PERIOD,DPDT,PSHIFT,STDDEV,SEED,JDSTRT,JDEND,
     +JDINC,PHSTRT,PHEND,PHINC,PHNORM,PERR0,DPERDT,THE,VUNIT,E,SMA,F1,
     +F2,VGA,XINCL,GR1,GR2,ABUNIN,TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,
     +XBOL1,XBOL2,YBOL1,YBOL2,WLA,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,
     +MZERO,FACTOR
      double precision XLAT1(*),XLONG1(*),RADSP1(*),TEMSP1(*),XLAT2(*),
     +XLONG2(*),RADSP2(*),TEMSP2(*)

    1 format(8(I1,1X))
    2 format(I1,F15.6,D15.10,D13.6,F10.4,D10.4,I2,F11.0)
    3 format(F14.6,F15.6,F13.6,4F12.6)
    4 format(4I2,2I4,F13.6,D12.5,F7.5,F8.2)
    5 format(F6.5,D13.6,2F10.4,F10.4,F9.3,2F7.3,F7.2)
    6 format(2(F7.4,1X),2F7.3,3D13.6,4F7.3)
    7 format(I3,2F10.5,4F7.3,F8.4,D10.4,F8.3,F8.4,F9.6)
    8 format(4F9.5)
    9 format(1X,F4.0)
   10 format(F4.0)
   11 format(I1)

      open(unit=1,file=FN,status='UNKNOWN')

      write(1,1) MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD
      write(1,2) JDPHS,HJD0,PERIOD,DPDT,PSHIFT,STDDEV,NOISE,SEED
      write(1,3) JDSTRT,JDEND,JDINC,PHSTRT,PHEND,PHINC,PHNORM
      write(1,4) MODE,IPB,IFAT1,IFAT2,N1,N2,PERR0,DPERDT,THE,VUNIT
      write(1,5) E,SMA,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN
      write(1,6) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,
     +           YBOL2
      write(1,7) IBAND,HLA,CLA,X1A,X2A,Y1A,Y2A,EL3,OPSF,MZERO,FACTOR,
     +           WLA

      do 95, i=1,NSP1
        write(1,8) XLAT1(i),XLONG1(i),RADSP1(i),TEMSP1(i)
   95 continue
      write(1,9) spend
      do 96, i=1,NSP2
        write(1,8) XLAT2(i),XLONG2(i),RADSP2(i),TEMSP2(i)
   96 continue
      write(1,9) spend

      write(1,10) clend

      write(1,11) lciend

      close(unit=1)

      END
