      subroutine wrhead(ibef,nref,mref,ifsmv1,ifsmv2,icor1,icor2,
     $ld,jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed,hjdst,hjdsp,
     $hjdin,phstrt,phstop,phin,phn,mode,ipb,ifat1,ifat2,n1,n2,perr0,
     $dperdt,the,vunit,vfac,e,a,f1,f2,vga,xincl,gr1,gr2,nsp1,nsp2,
     $abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,
     $iband,hlum,clum,xh,xc,yh,yc,el3,opsf,zero,factor,wl,binwm1,sc1,
     $sl1,binwm2,sc2,sl2,wll1,ewid1,depth1,wll2,ewid2,depth2,
     $kks,xlat,xlong,radsp,temsp,ncl,xcl,ycl,zcl,rcl,op1,fcl,edens,xmue,
     $encl,dens,ns1,sms1,sr1,bolm1,xlg1,ns2,sms2,sr2,bolm2,xlg2,mmsave,
     $sbrh,sbrc,sm1,sm2,phperi,pconsc,pconic,dif1,abunir,abun,mod)

      implicit real*8(a-h,o-z)

      common /kfac/ kff1,kff2,kfo1,kfo2
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2

      dimension wll1(*),ewid1(*),depth1(*),wll2(*),ewid2(*),depth2(*)
      dimension xlat(2,*),xlong(2,*),radsp(2,*),temsp(2,*),kks(2,*)
      dimension xcl(*),ycl(*),zcl(*),rcl(*),op1(*),fcl(*),edens(*),
     $          xmue(*),encl(*),dens(*)
      dimension mmsave(*),abun(*)

  204 format('*************  Next block of output   ********************
     $************')
  287 format('Input [M/H] = ',f6.3,' is not a value recognized by ',
     $'the program. Replaced by ',f5.2)
   49 format(' PROGRAM SHOULD NOT BE USED IN MODE 1 OR 3 WITH NON-ZERO E
     $CCENTRICITY')
  350 format(' Primary star exceeds outer contact surface')
   50 format(40H PRIMARY COMPONENT EXCEEDS CRITICAL LOBE)
  351 format(' Secondary star exceeds outer contact surface')
   51 format(42H SECONDARY COMPONENT EXCEEDS CRITICAL LOBE)
  148 format('   mpage  nref   mref   ifsmv1   ifsmv2   icor1   icor2
     $ld')
  149 format(i6,2i7,i8,i9,i9,i8,i6)
  171 format('JDPHS',5x,'J.D. zero',7x,'Period',11x,'dPdt',
     $6x,'Ph. shift',3x,'fract. sd.',2x,'noise',5x,'seed')
  170 format(i3,f17.6,d18.10,d14.6,f10.4,d13.4,i5,f13.0)
  219 format(5x,'JD start',9x,'JD stop',6x,'JD incr',6x,
     $'Ph start',4x,'Ph. stop',5x,'Ph incr',5x,'Ph norm')
  218 format(f14.6,f16.6,f14.6,4f12.6)
   10 format('MODE   IPB  IFAT1 IFAT2   N1   N2',4x,'Arg. Per',7x,'dPerd
     $t',4x,'Th e',4x,'V UNIT(km/s)    V FAC')
   33 format(I4,I5,I6,I6,I7,I5,f13.6,d14.5,f9.5,f10.2,d16.4)
   48 format('  ecc',5x,'s-m axis',7x,'F1',9x,'F2',7x,'Vgam',7x,
     $'Incl',6x,'g1',6x,'g2  Nspot1 Nspot 2',4x,'[M/H]')
    5 format(F6.5,d13.6,2F11.4,F11.4,F10.3,2f8.3,i5,i7,f10.2)
   54 format(2x,'T1',6x,'T2',5x,'Alb 1  Alb 2',4x,'Pot 1',8x,'Pot 2',
     $11x,'M2/M1',2x,'x1(bolo) x2(bolo) y1(bolo) y2(bolo)')
    8 format(f7.4,f8.4,2F7.3,3d13.6,f8.3,f9.3,f9.3,f9.3)
   47 FORMAT(2x,'band',7x,'L1',9x,'L2',7x,'x1',6x,'x2',6x,'y1',6x,
     $'y2',6x,'el3     opsf      m zero   factor',2x,'wv lth')
   34 format(i5,1X,2F11.5,4f8.3,F9.4,d11.4,F9.3,F9.4,f9.6)
  142 format('star',4x,'bin width (microns)',3x,'continuum scale',4x,'co
     $ntinuum slope',2x,'nfine')
 2049 format(i3,d14.5,f18.2,f20.2,i14)
  157 format('star ',i1,'   line wavelength',4x,'equivalent width (micro
     $ns)',5x,'rect. line depth',2x,'kks')
  152 format(f20.6,d23.5,17x,f13.5,i6)
   83 format(1X,'STAR  CO-LATITUDE  LONGITUDE  SPOT RADIUS  TEMP. FACTOR
     $')
   84 format(1X,I4,4F12.5)
   69 format('      xcl       ycl       zcl      rcl       op1         f
     $cl        ne       mu e      encl     dens')
   64 format(3f10.4,f9.4,d12.4,f10.4,d12.4,f9.4,f9.3,d12.4)
  150 format(' Star',9x,'M/Msun   (Mean Radius)/Rsun',5x,'M Bol',4x,'Log
     $ g (cgs)')
  250 format(4x,I1,4x,f12.3,11x,f7.2,6x,f6.2,8x,f5.2)
   43 format(91x,'superior',5x,'inferior')
   44 format(76x,'periastron',2x,'conjunction',2x,'conjunction')
   46 format('grid1/4    grid2/4',2X,'polar sbr 1',3X,'polar sbr 2'
     $,3X,'surf. area 1',2X,'surf. area 2',7X,'phase',8X,
     $'phase',8x,'phase')
   94 format(i6,i11,4F14.6,F13.6,f13.6,f13.6)
  244 format('Note: The light curve output contains simulated observa',
     $'tional scatter, as requested,')
  245 format('with standard deviation',f9.5,' of light at the reference'
     $,' phase.')
   79 format(6x,'JD',17x,'Phase     light 1     light 2     (1+2+3)    n
     $orm lite   dist      mag+K')
   45 format(6x,'JD',14x,'Phase     V Rad 1     V Rad 2      del V1
     $ del V2   V1 km/s      V2 km/s')
   96 format(6x,'JD',13x,'Phase',5x,'r1pol',6x,'r1pt',5x,'r1sid',5x,'r1b
     $ak',5x,'r2pol',5x,'r2pt',6x,'r2sid',5x,'r2bak')

      if(ibef.eq.0) goto 335
      write(16,*)
      write(16,*)
      write(16,*)
      write(16,*)
      write(16,*)
      write(16,204)
      write(16,*)
      write(16,*)
      write(16,*)
      write(16,*)
  335 continue

      if(dif1.ne.0.d0) write(16,287) abunir,abunin
      IF(mod.eq.1) write(16,49)

      if(kfo1.eq.0) goto 380
      write(16,350)
      goto 381
  380 IF(KFF1.EQ.1) WRITE(16,50)
  381 if(kfo2.eq.0) goto 382
      write(16,351)
      goto 383
  382 IF(KFF2.EQ.1) WRITE(16,51)
  383 IF((KFF1+KFF2+kfo1+kfo2).GT.0) WRITE(16,*)

      write(16,148)
      write(16,149) mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      write(16,*)

      write(16,171)
      write(16,170) jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed
      write(16,*)

      write(16,219)
      write(16,218) hjdst,hjdsp,hjdin,phstrt,phstop,phin,phn
      write(16,*)

      write(16,10)
      write(16,33)mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,the,vunit,vfac
      write(16,*)

      write(16,48)
      write(16,5) e,a,f1,f2,vga,xincl,gr1,gr2,nsp1,nsp2,abunin
      write(16,*)

      write(16,54)
      write(16,8) tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,
     $ybol2
      write(16,*)

      write(16,47)
      write(16,34) iband,hlum,clum,xh,xc,yh,yc,el3,opsf,zero,factor,wl

      ns1=1
      ns2=2
      if(mpage.ne.3) goto 174
      write(16,*)
      write(16,142)
      write(16,2049) ns1,binwm1,sc1,sl1,nf1
      write(16,2049) ns2,binwm2,sc2,sl2,nf2
      write(16,*)
      write(16,157) ns1
      do 155 iln=1,nl1
  155 write(16,152) wll1(iln),ewid1(iln),depth1(iln),kks(1,iln)
      write(16,*)
      write(16,157) ns2
      do 151 iln=1,nl2
  151 write(16,152) wll2(iln),ewid2(iln),depth2(iln),kks(2,iln)
  174 continue
      write(16,*)
      write(16,*)

      nstot=nsp1+nsp2
      if(nstot.gt.0) write(16,83)
      do 188 KP=1,2
      if((NSP1+KP-1).EQ.0) goto 188
      if((NSP2+(KP-2)**2).EQ.0) goto 188
      NSPOT=NSP1
      if(KP.EQ.2) NSPOT=NSP2
      do 187 I=1,NSPOT
  187 write(16,84)KP,XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
  188 write(16,*)

      if(ncl.eq.0) goto 67
      write(16,69)
      do 68 i=1,ncl
   68 write(16,64) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i),dens(i)
      write(16,*)
   67 continue

      write(16,150)
      write(16,250) ns1,sms1,sr1,bolm1,xlg1
      write(16,250) ns2,sms2,sr2,bolm2,xlg2

      np1=n1+1
      np2=n1+n2+2
      write(16,*)
      write(16,43)
      write(16,44)
      write(16,46)
      write(16,94) mmsave(np1),mmsave(np2),sbrh,sbrc,sm1,sm2,phperi,
     $pconsc,pconic
      write(16,*)
      if(stdev.eq.0.d0.or.mpage.ne.1) goto 246
      write(16,244)
      write(16,245) stdev
  246 continue
      write(16,*)

      if (mpage.eq.1) write(16,79)
      if (mpage.eq.2) write(16,45)
      if (mpage.eq.4) write(16,96)

      end
