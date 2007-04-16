      subroutine wrdata(hjd,phas,yskp,zskp,htt,cool,total,tot,d,smagg,
     $vsum1,vsum2,vra1,vra2,vkm1,vkm2,delv1,delwl1,wl1,fbin1,resf1,
     $delv2,delwl2,wl2,fbin2,resf2,rv,rvq,mmsave,ll1,lll1,llll1,ll2,
     $lll2,llll2)

      implicit real*8(a-h,o-z)

      common /invar/ kh,ipb,irte,nref,irvol1,irvol2,mref,ifsmv1,ifsmv2,
     $icor1,icor2,ld,ncl,jdphs,ipc
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2

      dimension yskp(*),zskp(*)
      dimension delv1(*),delwl1(*),wl1(*),fbin1(*),resf1(*)
      dimension delv2(*),delwl2(*),wl2(*),fbin2(*),resf2(*)
      dimension rv(*),rvq(*),mmsave(*)

  128 format('HJD = ',f14.5,'    Phase = ',f14.5)
  131 format(3x,'Y Sky Coordinate',4x,'Z Sky Coordinate')
  130 format(f16.6,f20.6)
    3 format(f15.6,F15.5,4F12.8,F10.5,f10.4)
   93 format(f14.6,f13.5,4f12.6,2d13.4)
   92 format('Phase =',f14.6)
  167 format(30x,'star',i2)
  907 format(6x,'del v',6x,'del wl (mic.)',7x,'wl',9x,'profile',6x,'res
     $flux')
  903 format(6f14.7)
  205 format('**********************************************************
     $************')
  296 format(f14.6,f13.5,8f10.5)

      if(mpage.ne.5) goto 127
      write(6,*)
      write(6,*)
      write(6,128) hjd,phas
      write(6,*)
      write(6,131)
      do 129 imp=1,ipc
      write(6,130) yskp(imp),zskp(imp)
  129 continue
      return
  127 continue

      if(mpage.eq.1) write(6,3) hjd,phas,htt,cool,total,tot,d,smagg
      if(mpage.eq.2) write(6,93) hjd,phas,vsum1,vsum2,vra1,vra2,vkm1,
     $vkm2

      ns1=1
      ns2=2
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
 
      end
