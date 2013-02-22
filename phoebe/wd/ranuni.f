      subroutine ranuni(sn,smod,sm1p1)
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
