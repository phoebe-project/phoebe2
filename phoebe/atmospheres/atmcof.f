C FILE: atmcof.f
      subroutine init(plpath,atmpath)
      implicit real*8 (a-h,o-z)
C
C     VERSION OF DEC 1, 2015
C
      dimension plcof(1250),grand(250800)
      character plpath*(*),atmpath*(*)
      common /planckleg/ plcof
      common /arrayleg/ grand
      common /meta/ initialized
Cf2py intent(in) plpath
Cf2py intent(in) atmpath
      open(unit=23,file=plpath,status='old')  
      read(23,*) plcof  
      close(23)
      open(unit=22,file=atmpath,status='old')  
      read(22,*) grand
      close(23)
      initialized=1
      end

      subroutine binnum(x,n,y,j) 
C
C     Version of January 7, 2002 
C
      implicit real*8(a-h,o-z) 
      dimension x(n)
Cf2py intent(in) x
Cf2py intent(in) n
Cf2py intent(in) y
Cf2py intent(out) j
      mon=1 
      if(x(1).gt.x(2)) mon=-1 
      do 1 i=1,n 
      if(mon.eq.-1) goto 3 
      if(y.le.x(i)) goto 2 
      goto 1 
   3  if(y.gt.x(i)) goto 2 
   1  continue 
   2  continue 
      j=i-1 
      return 
      end 

      subroutine legendre(x,pleg,n)
C
C     Version of January 7, 2002
C
      implicit real*8 (a-h,o-z)
      dimension pleg(n)
Cf2py intent(in) x
Cf2py intent(out) pleg
Cf2py intent(in) n
      pleg(1)=1.d0
      pleg(2)=x
      if(n.le.2) return
      denom=1.d0
      do 1 i=3,n
      fac1=x*(2.d0*denom+1.d0)
      fac2=denom
      denom=denom+1.d0
      pleg(i)=(fac1*pleg(i-1)-fac2*pleg(i-2))/denom
   1  continue
      return
      end

      subroutine planckint(t,ifil,ylog,y)
      implicit real*8 (a-h,o-z)
C     Version of January 9, 2002
C     cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     IMPORTANT README
C     This subroutine returns the log10 (ylog) of a Planck central
C     intensity (y), as well as the Planck central intensity (y) itself.
C     The subroutine ONLY WORKS FOR TEMPERATURES GREATER THAN OR EQUAL
C     500 K OR LOWER THAN 500,300 K. For teperatures outside this range,
C     the program stops and prints a message.
C     ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dimension plcof(1250)
      dimension pl(10)
      common /planckleg/ plcof
Cf2py intent(in) t
Cf2py intent(in) ifil
Cf2py intent(out) ylog
Cf2py intent(out) y
      xld=0
      yld=0
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

      subroutine atmx(t,g,abunin,ifil,xintlog,xint)  
      implicit real*8 (a-h,o-z)  
c Version of Aug. 16, 2002  
c New system of messages 
      dimension abun(19),glog(11),grand(250800)  
      dimension pl(10),yy(4),pha(4),tte(2),effwvl(25)  
      dimension message(2,4)  
      common/arrayleg/ grand
c     common /ramprange/ tlowtol,thightol,glowtol,ghightol  
c     common /atmmessages/ message,komp  
      data effwvl/350.d0,412.d0,430.d0,546.d0,365.d0,440.d0,  
     $550.d0,680.d0,870.d0,1220.d0,2145.d0,3380.d0,4900.d0,  
     $9210.d0,650.d0,790.d0,230.d0,250.d0,270.d0,290.d0,  
     $310.d0,330.d0,430.d0,520.d0,500.d0/
      data abun/1.d0,0.5d0,0.3d0,0.2d0,0.1d0,0.0d0,-0.1d0,  
     $-0.2d0,-0.3d0,-0.5d0,-1.0d0,-1.5d0,-2.0d0,-2.5d0,  
     $-3.0d0,-3.5d0,-4.0d0,-4.5d0,-5.0d0/  
      data glog/0.0d0,0.5d0,1.0d0,1.5d0,2.0d0,2.5d0,3.0d0,  
     $3.5d0,4.0d0,4.5d0,5.0d0/
Cf2py intent(in) t
Cf2py intent(in) g
Cf2py intent(in) abunin
Cf2py intent(in) ifil
Cf2py intent(out) xintlog
Cf2py intent(out) xint
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c  Ramp ranges are set below. The following values seem to work.   
c  They may be changed.  
      tlowtol=1500.d0  
      thightol=50000.d0  
      glowtol=0.5d0  
      ghightol=0.5d0  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c  The following lines take care of abundances that may not be among  
c  the 19 Kurucz values (see abun array). abunin is reset at the allowed value nearest  
c  the input value.  
      call binnum(abun,19,abunin,iab)  
      dif1=abunin-abun(iab)  
      if(iab.eq.19) goto 702  
      dif2=abun(iab+1)-abun(iab)  
      dif=dif1/dif2  
      if((dif.ge.0.d0).and.(dif.le.0.5d0)) goto 702  
      iab=iab+1  
  702 continue  
c~       if(dif1.ne.0.d0) write(6,287) abunin,abun(iab)  
      abunin=abun(iab)
      istart=1+(iab-1)*13200  
c***************************************************************  
      komp=1
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
      tlog=dlog10(t)  
      trec=1.d0/t  
      tlow=3500.d0-tlowtol  
      if(t.lt.tlow) goto 66  
      thigh=50000.d0+thightol  
      fractol=thightol/50000.d0  
      glow=0.d0-glowtol  
      if(g.lt.glow) goto 77  
      ghigh=5.d0+ghightol  
      if(g.gt.ghigh) goto 78  
      tt=t  
      gg=g  
      if(g.ge.0.d0) goto 11  
      gg=0.d0  
      goto 12  
  11  if(g.le.5.d0) goto 12  
      gg=5.d0  
  12  continue  
ccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c The following is for 4-point interpolation in log g.  
ccccccccccccccccccccccccccccccccccccccccccccccccccccc  
      m=4  
      ifreturn=0  
      icase=istart+(ifil-1)*528  
      call binnum(glog,11,g,j)  
      k=min(max(j-(m-1)/2,1),12-m)  
      if(g.le.0.d0) j=1  
  10  continue  
      ib=icase+(k-1)*48  
      ib=ib-48  
ccccccccccccccccccccccccccccccccccccccccccccccccccccc  
      do 4 ii=1,m  
      ib=ib+48  
      do 719 ibin=1,4  
      it=ib+(ibin-1)*12  
      it1=it+1  
      if(tt.le.grand(it1)) goto 720  
  719 continue  
      ibin=ibin-1  
  720 continue  
      tb=grand(it)  
      if(tb.ne.0.d0) goto 55  
      if(ibin.eq.4) ibin=ibin-1  
      if(ibin.eq.1) ibin=ibin+1  
      it=ib+(ibin-1)*12  
      it1=it+1  
      tb=grand(it)  
  55  continue  
      te=grand(it1)  
      ibinsav=ibin  
      thigh=te+fractol*te  
      ibb=ib+1+(ibin-1)*12  
      do 1 jj=1,10  
      if(grand(ibb+jj).ne.0.d0) goto 1  
      goto 2  
   1  continue  
   2  ma=jj-1  
      pha(ii)=(tt-tb)/(te-tb)  
      yy(ii)=0.d0  
      call legendre(pha(ii),pl,ma)  
      if(pha(ii).lt.0.d0) call legendre(0.d0,pl,ma)  
      do 3 kk=1,ma  
      kj=ibb+kk  
   3  yy(ii)=yy(ii)+pl(kk)*grand(kj)  
      if(pha(ii).ge.0.d0) goto 4  
      tlow=tb-tlowtol  
      call planckint(tlow,ifil,yylow,dum)  
      if(t.ge.tlow) goto 424  
      call planckint(t,ifil,yy(ii),dum)  
      goto 4  
  424 continue  
      tlowmidlog=0.5d0*dlog10(tb*tlow)  
      wvlmax=10.d0**(6.4624d0-tlowmidlog)  
      if(effwvl(ifil).lt.wvlmax) goto 425  
      tblog=dlog10(tb)  
      tlowlog=dlog10(tlow)  
      slope=(yy(ii)-yylow)/(tblog-tlowlog)  
      yy(ii)=yylow+slope*(tlog-tlowlog)  
      goto 4  
  425 continue  
      tbrec=1.d0/tb  
      tlowrec=1.d0/tlow  
      slope=(yy(ii)-yylow)/(tbrec-tlowrec)  
      yy(ii)=yylow+slope*(trec-tlowrec)  
   4  ibin=ibinsav  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c Next, do a m-point Lagrange interpolation.  
      xintlog=0.d0  
      do 501 ii=1,m  
      xnum=1.d0  
      denom=1.d0  
      nj=k+ii-1  
      do 500 iij=1,m  
      njj=k+iij-1  
      if(ii.eq.iij) goto 500  
      xnum=xnum*(gg-glog(njj))  
      denom=denom*(glog(nj)-glog(njj))  
 500  continue  
      xintlog=xintlog+yy(ii)*xnum/denom  
 501  continue  
cccccccccccccccccccccccccccccccccccccccccccccccc  
c  Check if a ramp function will be needed, or if we are  
c  close to the border and need to interpolate between less  
c  than 4 points.  
ccccccccccccccccccccccccccccccccccccccccccccccccc  
      if(g.lt.0.d0) goto 7  
      if(g.gt.5.d0) goto 9  
      if(t.lt.3500.d0) goto 99  
      if(pha(1).le.1.d0) goto 99  
      if(ifreturn.eq.1) goto 99 
      if(j.eq.1) goto 5  
      if(pha(3).gt.1.d0) goto 5  
      k=k+1  
      if(pha(2).gt.1.d0) goto 41   
  42  continue  
      if(k.gt.8) m=12-k  
      ifreturn=1  
      goto 10  
  41  continue  
      if(j.lt.10) goto 5  
      k=k+1  
      goto 42  
ccccccccccccccccccccccccccccccccccccccccccccccccc  
   5  continue  
      ib=icase+(j-1)*48  
      ib=ib-48  
      do 61 kik=1,2  
      ib=ib+48  
      do 619 ibin=1,4  
      it=ib+(ibin-1)*12  
      it1=it+1  
      if(tt.le.grand(it1)) goto 620  
  619 continue  
      ibin=ibin-1  
  620 continue  
      tb=grand(it)  
      if(tb.ne.0.d0) goto 67  
      if(ibin.eq.1) ibin=ibin+1  
      if(ibin.eq.4) ibin=ibin-1  
      it=ib+(ibin-1)*12  
      it1=it+1  
      tb=grand(it)  
  67  continue  
      te=grand(it1)  
      tte(kik)=t  
      if(t.gt.te) tte(kik)=te  
      ibb=ib+1+(ibin-1)*12  
      do 111 jj=1,10  
      if(grand(ibb+jj).ne.0.d0) goto 111  
      goto 22  
 111  continue  
  22  ma=jj-1  
      pha(kik)=(tte(kik)-tb)/(te-tb)  
      call legendre(pha(kik),pl,ma)  
      yy(kik)=0.d0  
      do 33 kk=1,ma  
      kj=ibb+kk  
  33  yy(kik)=yy(kik)+pl(kk)*grand(kj)  
      ibin=ibinsav  
  61  continue  
      if(g.gt.5.d0) goto 43  
      if(g.lt.0.d0) goto 47  
      slope=(yy(2)-yy(1))*2.d0  
      yy(1)=yy(2)+slope*(g-glog(j+1))  
      slope=(tte(2)-tte(1))*2.d0  
      te=tte(1)+slope*(g-glog(j))  
      thigh=te*(1.d0+fractol)  
      if(t.gt.thigh) goto 79  
      call planckint(thigh,ifil,yyhigh,dum)  
      thighmidlog=0.5d0*dlog10(te*thigh)  
      wvlmax=10.d0**(6.4624d0-thighmidlog)  
      if(effwvl(ifil).lt.wvlmax) goto 426  
      thighlog=dlog10(thigh)  
      telog=dlog10(te)  
      slope=(yyhigh-yy(1))/(thighlog-telog)  
      xintlog=yyhigh+slope*(tlog-thighlog)  
      goto 99  
  426 continue  
      thighrec=1.d0/thigh  
      terec=1.d0/te  
      slope=(yyhigh-yy(1))/(thighrec-terec)  
      xintlog=yyhigh+slope*(trec-thighrec)  
      goto 99  
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
  43  yy(1)=yy(2)  
      te=tte(2)  
      call planckint(thigh,ifil,yyhigh,dum)  
      thighmidlog=0.5d0*dlog10(te*thigh)  
      wvlmax=10.d0**(6.4624d0-thighmidlog)  
      if(effwvl(ifil).lt.wvlmax) goto 427  
      thighlog=dlog10(thigh)  
      telog=dlog10(te)  
      slope=(yyhigh-yy(1))/(thighlog-telog)  
      xintlog=yyhigh+slope*(tlog-thighlog)  
      goto 44  
  427 continue  
      thighrec=1.d0/thigh  
      terec=1.d0/te  
      slope=(yyhigh-yy(1))/(thighrec-terec)  
      xintlog=yyhigh+slope*(trec-thighrec)  
      goto 44  
  47  continue  
      te=tte(1)  
      call planckint(thigh,ifil,yyhigh,dum)  
      thighmidlog=0.5d0*dlog10(te*thigh)  
      wvlmax=10.d0**(6.4624d0-thighmidlog)  
      if(effwvl(ifil).lt.wvlmax) goto 428  
      thighlog=dlog10(thigh)  
      telog=dlog10(te)  
      slope=(yyhigh-yy(1))/(thighlog-telog)  
      xintlog=yyhigh+slope*(tlog-thighlog)  
      goto 63  
  428 continue  
      thighrec=1.d0/thigh  
      terec=1.d0/te  
      slope=(yyhigh-yy(1))/(thighrec-terec)  
      xintlog=yyhigh+slope*(trec-thighrec)  
      goto 63  
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
   7  continue  
      thigh=6000.d0*(1.d0+fractol)  
      if(t.gt.thigh) goto 79  
      if(pha(1).le.1.d0) goto 68  
      goto 5  
  68  continue  
      if(t.ge.3500.d0) goto 63  
      call planckint(tlow,ifil,yylow,dum)  
      tlowmidlog=0.5d0*dlog10(3500.d0*tlow)  
      wvlmax=10.d0**(6.4624d0-tlowmidlog)  
      if(effwvl(ifil).lt.wvlmax) goto 430  
      tlowlog=dlog10(tlow)  
      slope=(xintlog-yylow)/(dlog10(3500.d0)-tlowlog)  
      xintlog=yylow+slope*(tlog-tlowlog)  
      goto 63  
  430 continue  
      tlowrec=1.d0/tlow  
      slope=(xintlog-yylow)/((1.d0/3500.d0)-tlowrec)  
      xintlog=yylow+slope*(trec-tlowrec)  
  63  continue  
      call planckint(t,ifil,yylow,dum)  
      slope=(yylow-xintlog)/glow  
      xintlog=yylow+slope*(g-glow)  
      goto 99  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
   9  continue  
      thigh=50000.d0*(1.d0+fractol)  
      if(t.gt.thigh) goto 79  
      if(t.gt.50000.d0) goto 52  
  44  continue  
      call planckint(t,ifil,yyhigh,dum)  
      slope=(yyhigh-xintlog)/(ghigh-5.d0)  
      xintlog=yyhigh+slope*(g-ghigh)  
      goto 99  
  52  continue  
      j=10  
      goto 5  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
  66  continue  
      message(komp,4)=1  
      call planckint(t,ifil,xintlog,xint)  
      return  
  77  continue  
      message(komp,1)=1  
      call planckint(t,ifil,xintlog,xint)  
      return  
  78  continue  
      message(komp,2)=1  
      call planckint(t,ifil,xintlog,xint)  
      return  
  79  continue  
      message(komp,3)=1  
      call planckint(t,ifil,xintlog,xint)  
      return  
  99  continue 
      xint=10.d0**xintlog  
      return 
      end  
C END FILE atmcof.f
