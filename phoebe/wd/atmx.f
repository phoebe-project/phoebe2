      subroutine atmx(t,g,ifil,xintlog,xint)
      implicit real*8 (a-h,o-z)
c Version of January 23, 2004
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Model atmosphere grid properties:
c
c       itemppts  ..   number of temperature coefficients per spectrum
c                        default: itemppts=48 (4x12)
c       iloggpts  ..   number of log(g) nodes
c                        default: iloggpts=11 (atmosphere grid)
c       imetpts   ..   number of metallicity nodes
c                        default: imetpts=19  (atmosphere grid)
c       iatmpts   ..   size of the atmosphere grid per passband per
c                      metallicity
c                        default: iatmpts = 11*48 = 528
c                        11 log(g) values and
c                        48=4x12 temperature coefficients
c       iatmchunk ..   size of the atmosphere grid per metallicity
c                        default: iatmchunk = 528*25 = 13200
c       iatmsize  ..   size of the atmosphere grid
c                        default: iatmsize = 13200*19 = 250800
c
      parameter (iplmax  =48)
      parameter (itemppts=48)
      parameter (iloggpts=11)
      parameter (imetpts =19)
      parameter (iatmpts=iloggpts*itemppts)
      parameter (iatmchunk=iatmpts*iplmax)
      parameter (iatmsize=iatmchunk*imetpts)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dimension abun(imetpts),glog(iloggpts),grand(iatmsize)
      dimension pl(10),yy(4),pha(4),tte(2),effwvl(iplmax)
      dimension message(2,4)
      common/abung/abun,glog
      common/arrayleg/ grand,istart
      common /ramprange/ tlowtol,thightol,glowtol,ghightol
      common /atmmessages/ message,komp
      data effwvl/350.d0,412.d0,430.d0,546.d0,365.d0,440.d0,
     $550.d0,680.d0,870.d0,1220.d0,2145.d0,3380.d0,4900.d0,
     $9210.d0,650.d0,790.d0,230.d0,250.d0,270.d0,290.d0,
     $310.d0,330.d0,430.d0,520.d0,500.d0,640.d0,640.d0,
     $1620.d0,345.6d0,424.5d0,402.4d0,448.d0,550.d0,540.5d0,
     $580.5d0,592.0d0,355.7d0,482.5d0,626.1d0,767.2d0,909.7d0,
     $367.d0,485.d0,624.d0,752.d0,867.d0,963.d0,963.d0/
      tlog=dlog10(t)
      trec=1.d0/t
      tlow=3500.d0-tlowtol
      if(t.le.tlow) goto 66
      thigh=50000.d0+thightol
      fractol=thightol/50000.d0
      glow=0.d0-glowtol
      if(g.le.glow) goto 77
      ghigh=5.d0+ghightol
      if(g.ge.ghigh) goto 78
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
      icase=istart+(ifil-1)*iatmpts
      call binnum(glog,iloggpts,g,j)
      k=min(max(j-(m-1)/2,1),12-m)
      if(g.le.0.d0) j=1
  10  continue
      ib=icase+(k-1)*itemppts
      ib=ib-itemppts
ccccccccccccccccccccccccccccccccccccccccccccccccccccc
      do 4 ii=1,m
      ib=ib+itemppts
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
      if(pha(1).le.1.d0) goto 63
      goto 5
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
