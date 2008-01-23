      subroutine wrfoot(message,f1,f2,po,rm,f,dp,e,drdq,dodq,ii,mode,
     $mpage)

      implicit real*8(a-h,o-z)

      dimension message(2,*),po(*)

      dimension xtha(4),xfia(4),rad(4),drdo(4)
      data xtha(1),xtha(2),xtha(3),xtha(4),xfia(1),xfia(2),xfia(3),
     $xfia(4)/0.d0,1.570796d0,1.570796d0,1.570796d0,
     $0.d0,0.d0,1.5707963d0,3.14159365d0/

  283 format('log g below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  284 format('log g above ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  285 format('T above ramp range for at least one',
     $' point on star',i2,', black body applied locally.')
  286 format('T below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
   41 format('star',4X,'r pole',5X,'deriv',5X,'r point',5X,'deriv',
     $6X,'r side',6X,'deriv',5X,'r back',6X,'deriv')
   40 FORMAT(I3,8F11.5)
   74 FORMAT(' DIMENSIONLESS RADIAL VELOCITIES CONTAIN FACTOR P/(2PI*A)'
     $)

      do 909 komp=1,2
      write(16,*)
      if(message(komp,1).eq.1) write(16,283) komp
      if(message(komp,2).eq.1) write(16,284) komp
      if(message(komp,3).eq.1) write(16,285) komp
      if(message(komp,4).eq.1) write(16,286) komp
  909 continue

      if(mpage.eq.5) return

      write(16,*)
      write(16,41)
      write(16,*)

      do 119 ii=1,2
      gt1=dfloat(2-ii)
      gt2=dfloat(ii-1)
      f=f1*gt1+f2*gt2
      do 118 i=1,4
      call romq(po(ii),rm,f,dp,e,xtha(i),xfia(i),rad(i),drdo(i),
     $drdq,dodq,ii,mode)
  118 continue
      write(16,40) ii,rad(1),drdo(1),rad(2),drdo(2),rad(3),drdo(3),
     $rad(4),drdo(4)
  119 continue

      write(16,*)
      if(mpage.eq.2) write(16,74)

      end
