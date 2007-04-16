      subroutine rangau(smod,nn,sd,gau)
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
