      subroutine conjph(ecc,argper,phzero,trsc,tric,econsc,econic,
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
