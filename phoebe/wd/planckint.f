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
      dimension plcof(1750)
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
      ylog=y-dlog10(dark)-0.49714987269413d0
      y=10.d0**ylog
      return
  11  continue
      write(16,*) "planckint subroutine problem: T=", t, " is illegal."
      stop
c 80  format('Program stopped in PLANCKINT,
c    $T outside 500 - 500,300 K range.')
      end
