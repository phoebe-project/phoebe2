      subroutine rddata(mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,
     $jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed,hjdst,hjdsp,hjdin,
     $phstrt,phstop,phin,phn,mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,
     $the,vunit,e,a,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,
     $poth,potc,rm,xbol1,xbol2,ybol1,ybol2,iband,hlum,clum,xh,xc,yh,yc,
     $el3,opsf,zero,factor,wl,binwm1,sc1,sl1,wll1,ewid1,depth1,
     $kks,binwm2,sc2,sl2,wll2,ewid2,depth2,xlat,xlong,radsp,temsp,
     $xcl,ycl,zcl,rcl,op1,fcl,edens,xmue,encl,lpimax,ispmax,iclmax)

      implicit real*8(a-h,o-z)

      dimension wll1(*),ewid1(*),depth1(*),wll2(*),ewid2(*),depth2(*)
      dimension xlat(2,*),xlong(2,*),radsp(2,*),temsp(2,*),kks(2,*)
      dimension xcl(*),ycl(*),zcl(*),rcl(*),op1(*),fcl(*),edens(*),
     $          xmue(*),encl(*)

      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2

   22 format(8(i1,1x))
  649 format(i1,f15.6,d15.10,d13.6,f10.4,d10.4,i2,f11.0)
  217 format(f14.6,f15.6,f13.6,4f12.6)
    1 format(4I2,2I4,f13.6,d12.5,f7.5,F8.2)
    2 format(F6.5,d13.6,2F10.4,F10.4,f9.3,2f7.3,f7.2)
    6 format(2(F7.4,1X),2f7.3,3d13.6,4F7.3)
    4 format(i3,2F10.5,4F7.3,F8.4,d10.4,F8.3,F8.4,f9.6)
 2048 format(d11.5,f9.4,f9.2,i3)
  138 format(f9.6,d12.5,f10.5,i5)
   85 format(4f9.5)
   63 format(3f9.4,f7.4,d11.4,f9.4,d11.3,f9.4,f7.3)

      read(15,22) mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld

      if(mpage.eq.9) return

  414 continue

      read(15,649) jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed
      read(15,217) hjdst,hjdsp,hjdin,phstrt,phstop,phin,phn
      read(15,  1) mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,the,vunit
      read(15,  2) e,a,f1,f2,vga,xincl,gr1,gr2,abunin
      read(15,  6) tavh,tavc,alb1,alb2,poth,potc,rm,xbol1,xbol2,ybol1,
     $ybol2
      read(15,  4) iband,hlum,clum,xh,xc,yh,yc,el3,opsf,zero,factor,wl

      if(mpage.ne.3) goto 897

      read(15,2048) binwm1,sc1,sl1,nf1

      do 86 iln=1,lpimax
      read(15,138) wll1(iln),ewid1(iln),depth1(iln),kks(1,iln)
      if(wll1(iln).lt.0.d0) goto 89
   86 continue
   89 continue

      read(15,2048) binwm2,sc2,sl2,nf2

      do 99 iln=1,lpimax
      read(15,138) wll2(iln),ewid2(iln),depth2(iln),kks(2,iln)
      if(wll2(iln).lt.0.d0) goto 91
   99 continue
   91 continue

  897 continue

      DO 88 KP=1,2
      DO 87 I=1,ispmax
      READ(15,85) xlat(KP,I),xlong(KP,I),radsp(KP,I),temsp(KP,I)
      IF(XLAT(KP,I).GE.200.d0) GOTO 88
   87 continue
   88 continue

      do 62 i=1,iclmax
      read(15,63) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i)
      if(xcl(i).gt.100.d0) goto 66
   62 continue
   66 continue

      end
