      SUBROUTINE wrlci(FN,MPAGE,NREF,MREF,IFSMV1,IFSMV2,ICOR1,ICOR2,LD,
     +JDPHS,HJD0,PERIOD,DPDT,PSHIFT,STDDEV,NOISE,SEED,JDSTRT,JDEND,
     +JDINC,PHSTRT,PHEND,PHINC,PHNORM,MODE,IPB,IFAT1,IFAT2,N1,N2,PERR0,
     +DPERDT,THE,VUNIT,E,SMA,F1,F2,VGA,XINCL,GR1,GR2,ABUNIN,TAVH,TAVC,
     +ALB1,ALB2,PHSV,PCSV,RM,XBOL1,XBOL2,YBOL1,YBOL2,IBAND,HLA,CLA,X1A,
     +X2A,Y1A,Y2A,EL3,OPSF,MZERO,FACTOR,WLA,NSP1,XLAT1,XLONG1,RADSP1,
     +TEMSP1,NSP2,XLAT2,XLONG2,RADSP2,TEMSP2)

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
