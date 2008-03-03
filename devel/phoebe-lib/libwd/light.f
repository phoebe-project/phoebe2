      Subroutine light(phs,xincl,xh,xc,yh,yc,n1,n2,sumhot,sumkul,rv,grx,
     $gry,grz,rvq,grxq,gryq,grzq,mmsave,theta,rho,aa,bb,slump1,slump2,
     $somhot,somkul,d,wl,snth,csth,snfi,csfi,tld,gmag1,gmag2,glog1,
     $glog2,fbin1,fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,
     $resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,
     $ycl,zcl,rcl,op1,fcl,edens,encl,dens,taug,emmg,yskp,zskp,iband,
     $ifat1,ifat2,ifphn)
c   Version of October 18, 2004
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),SLUMP1(*),SLUMP2(*),MMSAVE(*),THETA(*),RHO(*),AA(*),BB(*)
      DIMENSION SNTH(*),CSTH(*),SNFI(*),CSFI(*),tld(*),gmag1(*),
     $gmag2(*),glog1(*),glog2(*)
      dimension xcl(*),ycl(*),zcl(*),rcl(*),op1(*),fcl(*),dens(*),
     $encl(*),edens(*),yskp(*),zskp(*)
      dimension fbin1(*),fbin2(*),delv1(*),delv2(*),count1(*),count2(*),
     $delwl1(*),delwl2(*),resf1(*),resf2(*),wl1(*),wl2(*),dvks1(*),
     $dvks2(*),tau1(*),tau2(*),hbarw1(*),hbarw2(*),taug(*),emm1(*),
     $emm2(*),emmg(*)
      dimension message(2,4)
      common /atmmessages/ message,komp
      common /coflimbdark/ x,y
      COMMON /misc/ X1
      COMMON /KFAC/ KFF1,KFF2,kfo1,kfo2
      COMMON /NSPT/ NSP1,NSP2
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm
     $,ifs1dm,ifs2dm,icr1dm,icr2dm,ld,ncl,jdphs,ipc
      common /flvar/ du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,
     $du12,du13,du14,du15,du16,du17,vunit,vfvu,du20,qfacd
      common /prof2/ vo1,vo2,ff1,ff2,binw1,binw2,sc1,sc2,sl1,sl2,
     $clight
      common /cld/ acm,opsf
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      common /setest/ sefac
      common /flpro/ vksf,binc,binw,difp,deldel,renfsq
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      COMMON /SPOTS/ SINLAT(2,100),COSLAT(2,100),SINLNG(2,100),COSLNG
     $(2,100),RAD(2,100),TEMSP(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      pi=dacos(-1.d0)
      twopi=pi+pi
      pih=.5d0*pi
      dtr=pi/180.d0
      kstp=4
      cirf=.002d0
      if(ifphn.eq.1) goto 16
      if(mpage.ne.3) goto 16
      nbins=90000
      binc1=.5d0*dfloat(nbins)
      binc2=binc1
      in1max=0
      in2max=0
      in1min=300000
      in2min=300000
      marm1=10
      marp1=10
      marm2=10
      marp2=10
      do 916 i=1,nbins
      fbin1(i)=0.d0
      fbin2(i)=0.d0
      count1(i)=0.d0
      count2(i)=0.d0
      delv1(i)=0.d0
      delv2(i)=0.d0
  916 continue
   16 continue
      PHA=PHS*twopi
      K=6
      KK=K+1
      XINC=XINCL*dtr
      L=1
      TEST=(PHS-.5d0)**2
      TESTS=(TEST-.071525d0)**2
      SINI=dsin(XINC)
      COSPH=dcos(PHA)
      SINPH=dsin(PHA)
      SINSQ=SINPH**2
      COSI=dcos(XINC)
      NP1=N1+1
      NP2=N1+N2+2
      LLL1=MMSAVE(NP1)
      LLL2=MMSAVE(NP2)
      NPP2=NP2-1
      LL1=MMSAVE(N1)+1
      LL2=MMSAVE(NPP2)+1
      LLLL1=(LL1+LLL1)/2
      LLLL2=(LL2+LLL2)/2
      SINSQE=0.d0
      IF(SINI.GT.0.d0) SINSQE=((1.10d0*(RV(LLL1)+RVQ(LLL2))/D)**2
     $-cosi**2)/SINI**2
      CICP=COSI*COSPH
      CISP=COSI*SINPH
      XLOS=COSPH*SINI
      YLOS=-SINPH*SINI
      ZLOS=COSI
      SUM=0.d0
      SOM=0.d0
      IF(TEST.LE..0625d0) GOTO 18
      COMP=-1.d0
      CMP=1.d0
      COMPP=1.d0
      KOMP=2
      nl=nl2
      ffc=ff2
      voc=vo2*vfvu
      NSPOT=NSP2
      IFAT=IFAT2
      CMPP=0.d0
      X=XC
      y=yc
      EN=N2
      NPH=N2
      NP=2*N2
      nf=nf2
      GOTO 28
   18 X=XH
      y=yh
      COMP=1.d0
      KOMP=1
      nl=nl1
      ffc=ff1
      voc=vo1*vfvu
      NSPOT=NSP1
      IFAT=IFAT1
      CMP=0.d0
      COMPP=-1.d0
      CMPP=1.d0
      EN=N1
      NPH=N1
      NP=2*N1
      nf=nf1
   28 DELTH=pih/EN
      enf=dfloat(nf)
      renfsq=1.d0/(enf*enf)
      nfm1=nf-1
      r2nfdt=0.5d0*delth/enf
      vfvuff=vfvu*ffc
      AR=CMPP*RV(LLLL1)+CMP*RVQ(LLLL2)
      BR=CMPP*RV(1)+CMP*RVQ(1)
      ASQ=AR*AR
      BSQ=BR*BR
      AB=AR*BR
      absq=ab*ab
      ASBS=ASQ-BSQ
      KF=(2-KOMP)*KFF1+(KOMP-1)*KFF2
      CMPPD=CMPP*D
      CMPD=CMP*D
      NPP=NP+1
      TEMF=1.d0
      ipc=0
      DO 36 I=1,NP
      IF(I.GT.NPH)GOTO 54
      UPDOWN=1.d0
      IK=I
      GOTO 55
   54 UPDOWN=-1.d0
      IK=NPP-I
   55 CONTINUE
      IPN1=IK+(KOMP-1)*N1
      SINTH=SNTH(IPN1)
      COSTH=CSTH(IPN1)*UPDOWN
      tanth=sinth/costh
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      XM=MM
      MH=MM
      MM=2*MM
      DELFI=pi/XM
      r2nfdf=.5d0/enf
      deldel=delth*delfi
      IP=(KOMP-1)*NP1+IK
      IY=MMSAVE(IP)+1
      IF(TEST.LE..0625d0)GOTO 19
      GX=GRXQ(IY)
      GY=-GRYQ(IY)
      GZ=UPDOWN*GRZQ(IY)
      grmag=gmag2(iy)
      GOTO 29
   19 GX=GRX(IY)
      GY=-GRY(IY)
      GZ=UPDOWN*GRZ(IY)
      grmag=gmag1(iy)
   29 COSSAV=(XLOS*GX+YLOS*GY+ZLOS*GZ)/GRMAG
      SUMJ=0.d0
      SOMJ=0.d0
      MPP=MM+1
      IY=IY-1
      DO 26 J=1,MM
      IF(J.GT.MH) GOTO 58
      RTLEFT=1.d0
      JK=J
      GOTO 59
   58 RTLEFT=-1.d0
      JK=MPP-J
   59 CONTINUE
      IX=IY+JK
      IS=IX+(KOMP-1)*LLL1
      SINFI=SNFI(IS)*RTLEFT
      COSFI=CSFI(IS)
      STSF=SINTH*SINFI
      STCF=SINTH*COSFI
      IF(TEST.LE..0625d0)GOTO 39
      IF(RVQ(IX).EQ.-1.d0) GOTO 26
      GX=GRXQ(IX)
      GY=RTLEFT*GRYQ(IX)
      GZ=UPDOWN*GRZQ(IX)
      R=RVQ(IX)
      grmag=gmag2(ix)
      GOTO 49
   39 IF(RV(IX).EQ.-1.d0) GOTO 26
      GX=GRX(IX)
      GY=RTLEFT*GRY(IX)
      GZ=UPDOWN*GRZ(IX)
      R=RV(IX)
      grmag=gmag1(ix)
   49 COSGAM=(XLOS*GX+YLOS*GY+ZLOS*GZ)/GRMAG
      ZZ=R*COSTH
      YY=R*COMP*STSF
      XX=CMPD+COMP*STCF*R
      if(mpage.ne.5) goto 174
      if(cosgam.gt.0.d0) goto 174
      ipc=ipc+1
      yskp(ipc)=(xx-qfacd)*sinph+yy*cosph
      zskp(ipc)=(-xx+qfacd)*cicp+yy*cisp+zz*sini
      if(nspot.eq.0) goto 174
      call spot(komp,nspot,sinth,costh,sinfi,cosfi,temf)
      if(temf.eq.1.d0) goto 174
      yskr=yskp(ipc)
      zskr=zskp(ipc)
      kstp=4
      cirf=.002d0
      stp=twopi/dfloat(kstp)
      do 179 ang=stp,twopi,stp
      ipc=ipc+1
      yskp(ipc)=yskr+dsin(ang)*cirf
      zskp(ipc)=zskr+dcos(ang)*cirf
  179 continue
  174 continue
      if(sinsq.gt.sinsqe) goto 27
      IF(TESTS.LT.2.2562d-3) GOTO 170
      IF((STCF*R).GT.(sefac*(CMP+COMP*X1))) GOTO 129
  170 PROD=COSSAV*COSGAM
      IF(PROD.GT.0.d0) GOTO 22
      COSSAV=-COSSAV
      YSKY=XX*SINPH+YY*COSPH-cmpd*SINPH
      ZSKY=-XX*CICP+yy*CISP+ZZ*SINI+CMPD*CICP
      RHO(L)=dsqrt(YSKY**2+ZSKY**2)
      THETA(L)=dasin(ZSKY/RHO(L))
      IF(YSKY.LT.0.d0) GOTO 92
      THETA(L)=twopi+THETA(L)
      GOTO 93
   92 THETA(L)=pi-THETA(L)
   93 IF (THETA(L).GE.twopi) THETA(L)=THETA(L)
     $-twopi
      L=L+1
      GOTO 27
   22 COSSAV=COSGAM
      GOTO 27
  129 COSSAV=COSGAM
      IF(KF.LE.0) GOTO 27
      ZZ=R*COSTH
      YY=R*COMP*STSF
      XX=CMPD+COMP*STCF*R
      YSKY=XX*SINPH+YY*COSPH-cmpd*SINPH
      ZSKY=-XX*CICP+YY*CISP+ZZ*SINI+CMPD*CICP
      rptsq=YSKY**2+ZSKY**2
      rtstsq=absq/(BSQ+ASBS*(ZSKY**2/rptsq))
      IF(rptsq.LE.rtstsq) GOTO 26
   27 IF(COSGAM.GE.0.d0) GOTO 26
      COSGAM=-COSGAM
      DARKEN=1.d0-X+X*COSGAM
      if(ld.ne.2) goto 141
      if(cosgam.eq.0.d0) goto 141
      darken=darken-y*cosgam*dlog(cosgam)
      goto 147
  141 continue
      if(ld.eq.3) darken=darken-y*(1.d0-dsqrt(cosgam))
  147 if(darken.lt.0.d0) darken=0.d0
      CORFAC=1.d0
      do 923 jn=1,nl
      Lspot(komp,jn)=0
  923 if(kks(komp,jn).eq.0) Lspot(komp,jn)=1
      IF(NSPOT.EQ.0) GOTO 640
      CALL SPOT(KOMP,NSPOT,SINTH,COSTH,SINFI,COSFI,TEMF)
      IF(TEMF.EQ.1.d0) GOTO 640
      TSP=TLD(IS)*TEMF
      if(ifat.eq.0) call planckint(tld(is),iband,xintlog,xintbase)
      if(ifat.eq.0) call planckint(tsp,iband,xintlog,xintspot)
      IF(IFAT.EQ.0) GOTO 941
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      CALL atmx(TLD(IS),glogg,iband,xintlog,xintbase)
      CALL atmx(TSP,glogg,iband,xintlog,xintspot)
  941 CORFAC=xintspot/xintbase
  640 CONTINUE
      rit=1.d0
      if(ncl.eq.0) goto 818
      do 815 icl=1,ncl
      opsfcl=opsf*fcl(icl)
      call cloud(xlos,ylos,zlos,xx,yy,zz,xcl(icl),ycl(icl),zcl(icl),
     $rcl(icl),wl,op1(icl),opsfcl,edens(icl),acm,encl(icl),cmpd,
     $ri,dx,dens(icl),tau)
      rit=rit*ri
  815 continue
  818 continue
      DIF=rit*COSGAM*DARKEN*CORFAC*(CMP*SLUMP2(IX)+CMPP*SLUMP1(IX))
      v=-r*(STCF*YLOS-stsf*XLOS)*COMP
      if(ifphn.eq.1) goto 423
      if(mpage.ne.3) goto 423
      vflump=vfvuff*r*comp*costh
      vcks=v*vfvuff
      vks=vcks+voc
      vksf=vks
      dvdr=vcks/r
      dvdth=vcks/tanth
      dvdfib=vfvuff*r*comp*(sinfi*ylos+cosfi*xlos)
c     dvdfic=dvdfib*sinth
      difp=dif*deldel*renfsq
c  dvdth and dvdfi (below) each need another term involving dr/d(theta)
c    or dr/d(fi), that I will put in later. There will be a small loss
c    of accuracy for distorted stars without those terms. See notes.
      if(komp.eq.2) goto 422
      binc=binc1
      binw=binw1
      do 1045 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1046 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1047
      dfif=dfloat(jfn)*r2nfdf*delfi
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      dlr=0.d0
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
 1047 call linpro(komp,dvks1,hbarw1,tau1,emm1,count1,taug,emmg,fbin1,
     $delv1)
      if(inmin.lt.in1min) in1min=inmin
      if(inmax.gt.in1max) in1max=inmax
 1046 continue
 1045 continue
      goto 423
  422 continue
      binc=binc2
      binw=binw2
      do 1145 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1146 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1147
      dfif=dfloat(jfn)*r2nfdf*delfi
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      dlr=0.d0
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
      ffi=dacos(cosfi)
      if(sinfi.lt.0.d0) ffi=twopi-ffi
 1147 call linpro(komp,dvks2,hbarw2,tau2,emm2,count2,taug,emmg,fbin2,
     $delv2)
      if(inmin.lt.in2min) in2min=inmin
      if(inmax.gt.in2max) in2max=inmax
 1146 continue
 1145 continue
  423 continue
      DIFF=DIF*V
      SOMJ=SOMJ+DIFF
      SUMJ=SUMJ+DIF
   26 CONTINUE
      SOMJ=SOMJ*DELFI
      SUMJ=SUMJ*DELFI
      SOM=SOM+SOMJ
   36 SUM=SUM+SUMJ
      IF(SINSQ.GE.SINSQE) GOTO 75
      L=L-1
      LK=k
      if(L.lt.14) LK=L/2-1
      CALL fourls(theta,rho,L,LK,aa,bb)
   75 IF(TEST.LE..0625d0) GOTO 118
      SUMKUL=SUM*DELTH
      SOMKUL=SOM*DELTH
      X=XH
      y=yh
      KOMP=1
      nl=nl1
      ffc=ff1
      voc=vo1*vfvu
      NSPOT=NSP1
      IFAT=IFAT1
      EN=N1
      SAFTY=2.6d0*RV(LLL1)/EN
      RMAX=RVQ(LLL2)+SAFTY
      RMIN=RVQ(1)-SAFTY
      NPH=N1
      NP=2*N1
      nf=nf1
      GOTO 128
  118 X=XC
      y=yc
      KOMP=2
      nl=nl2
      ffc=ff2
      voc=vo2*vfvu
      NSPOT=NSP2
      IFAT=IFAT2
      SUMHOT=SUM*DELTH
      SOMHOT=SOM*DELTH
      if(inmax.gt.in1max) in1max=inmax
      if(inmin.lt.in1min) in1min=inmin
      EN=N2
      SAFTY=2.6d0*RVQ(LLL2)/EN
      RMAX=RV(LLL1)+SAFTY
      RMIN=RV(1)-SAFTY
      NPH=N2
      NP=2*N2
      nf=nf2
  128 DELTH=pih/EN
      enf=dfloat(nf)
      nfm1=nf-1
      renfsq=1.d0/(enf*enf)
      r2nfdt=.5d0*delth/enf
      vfvuff=vfvu*ffc
      SOM=0.d0
      SUM=0.d0
      NPP=NP+1
      TEMF=1.d0
      inmin=300000
      inmax=0
      DO 136 I=1,NP
      IF(I.GT.NPH) GOTO 154
      UPDOWN=1.d0
      IK=I
      GOTO 155
  154 UPDOWN=-1.d0
      IK=NPP-I
  155 CONTINUE
      IPN1=IK+(KOMP-1)*N1
      SINTH=SNTH(IPN1)
      COSTH=CSTH(IPN1)*UPDOWN
      tanth=sinth/costh
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      XM=MM
      MH=MM
      MM=2*MM
      DELFI=pi/XM
      deldel=delth*delfi
      SOMJ=0.d0
      SUMJ=0.d0
      SIGN=0.d0
      DRHO=1.d0
      MPP=MM+1
      DO 126 J=1,MM
      IF(J.GT.MH) GOTO 158
      RTLEFT=1.d0
      JK=J
      GOTO 159
  158 RTLEFT=-1.d0
      JK=MPP-J
  159 CONTINUE
      IP=(KOMP-1)*NP1+IK
      IX=MMSAVE(IP)+JK
      IS=IX+LLL1*(KOMP-1)
      SINFI=SNFI(IS)*RTLEFT
      COSFI=CSFI(IS)
      STSF=SINTH*SINFI
      STCF=SINTH*COSFI
      IF(TEST.LE..0625d0)GOTO 139
      IF(RV(IX).EQ.-1.d0) GOTO 126
      GX=GRX(IX)
      GY=RTLEFT*GRY(IX)
      GZ=UPDOWN*GRZ(IX)
      R=RV(IX)
      grmag=gmag1(ix)
      GOTO 149
  139 IF(RVQ(IX).EQ.-1.d0) GOTO 126
      GX=GRXQ(IX)
      GY=RTLEFT*GRYQ(IX)
      GZ=UPDOWN*GRZQ(IX)
      R=RVQ(IX)
      grmag=gmag2(ix)
  149 COSGAM=(XLOS*GX+YLOS*GY+ZLOS*GZ)/GRMAG
      IF(COSGAM.LT.0.d0) GOTO 104
      SIGN=0.d0
      OLSIGN=0.d0
      GOTO 126
  104 COSGAM=-COSGAM
      ZZ=R*COSTH
      YY=R*COMPP*STSF
      XX=CMPPD+COMPP*STCF*R
      DARKEN=1.d0-X+X*COSGAM
      if(ld.ne.2) goto 142
      if(cosgam.eq.0.d0) goto 142
      darken=darken-y*cosgam*dlog(cosgam)
      goto 148
  142 continue
      if(ld.eq.3) darken=darken-y*(1.d0-dsqrt(cosgam))
  148 if(darken.lt.0.d0) darken=0.d0
      OLDIF=DIF
      CORFAC=1.d0
      do 823 jn=1,nl
      Lspot(komp,jn)=0
  823 if(kks(komp,jn).eq.0) Lspot(komp,jn)=1
      IF(NSPOT.EQ.0) GOTO 660
      CALL SPOT(KOMP,NSPOT,SINTH,COSTH,SINFI,COSFI,TEMF)
      IF(TEMF.EQ.1.d0) GOTO 660
      TSP=TLD(IS)*TEMF
      if(ifat.eq.0) call planckint(tld(is),iband,xintlog,xintbase)
      if(ifat.eq.0) call planckint(tsp,iband,xintlog,xintspot)
      IF(IFAT.EQ.0) GOTO 661
      glogg=cmp*glog1(ix)+cmpp*glog2(ix)
      CALL atmx(TLD(IS),glogg,iband,xintlog,xintbase)
      CALL atmx(TSP,glogg,iband,xintlog,xintspot)
  661 CORFAC=xintspot/xintbase
  660 CONTINUE
      rit=1.d0
      if(ncl.eq.0) goto 718
      do 715 icl=1,ncl
      opsfcl=opsf*fcl(icl)
      call cloud(xlos,ylos,zlos,xx,yy,zz,xcl(icl),ycl(icl),zcl(icl),
     $rcl(icl),wl,op1(icl),opsfcl,edens(icl),acm,encl(icl),cmppd,
     $ri,dx,dens(icl),tau)
      rit=rit*ri
  715 continue
  718 continue
      DIF=rit*COSGAM*DARKEN*CORFAC*(CMPP*SLUMP2(IX)+CMP*SLUMP1(IX))
      v=R*(STCF*YLOS-STSF*XLOS)*COMP
      DIFF=DIF*V
      IF(SINSQ.GT.SINSQE) GOTO 63
      OLSIGN=SIGN
      OLDRHO=DRHO
      YSKY=XX*SINPH+YY*COSPH-cmpd*SINPH
      ZSKY=-XX*CICP+yy*CISP+ZZ*SINI+CMPD*CICP
      RRHO=dsqrt(ysky*ysky+zsky*zsky)
      IF(RRHO.GT.RMAX)GOTO 63
      IF(RRHO.LT.RMIN)GOTO 126
      THET=dasin(ZSKY/RRHO)
      IF(YSKY.LT.0.d0) GOTO 192
      THET=twopi+THET
      GOTO 193
  192 THET=pi-THET
  193 IF(THET.GE.twopi) THET=THET-twopi
      RHHO=0.d0
      DO 52 N=1,KK
      ENNN=N-1
      ENTHET=ENNN*THET
   52 RHHO=RHHO+AA(N)*dcos(ENTHET)+BB(N)*dsin(ENTHET)
      SIGN=1.d0
      IF(RRHO.LE.RHHO) sign=-1.d0
      if(mpage.eq.3) goto 861
      DRHO=dabs(RRHO-RHHO)
      IF((SIGN*OLSIGN).GE.0.d0) GOTO 60
      SUMDR=DRHO+OLDRHO
      FACT=-(.5d0-DRHO/SUMDR)
      IF(FACT.LT.0.d0) GOTO 198
      RDIF=OLDIF
      GOTO 199
  198 RDIF=DIF
  199 CORR=FACT*RDIF*SIGN
      CORRR=CORR*V
      SUMJ=SUMJ+CORR
      SOMJ=SOMJ+CORRR
   60 IF(SIGN.LT.0.d0) GOTO 126
   63 SUMJ=SUMJ+DIF
      SOMJ=SOMJ+DIFF
      if(mpage.ne.5) goto 127
      ipc=ipc+1
      yskp(ipc)=(xx-qfacd)*sinph+yy*cosph
      zskp(ipc)=(-xx+qfacd)*cicp+yy*cisp+zz*sini
      if(nspot.eq.0) goto 126
      call spot(komp,nspot,sinth,costh,sinfi,cosfi,temf)
      if(temf.eq.1.d0) goto 126
      yskr=yskp(ipc)
      zskr=zskp(ipc)
      stp=twopi/dfloat(kstp)
      do 189 ang=stp,twopi,stp
      ipc=ipc+1
      yskp(ipc)=yskr+dsin(ang)*cirf
      zskp(ipc)=zskr+dcos(ang)*cirf
  189 continue
      goto 126
  127 continue
      if(mpage.ne.3) goto 126
      if(ifphn.eq.1) goto 126
  861 vflump=vfvuff*r*comp*costh
      vcks=v*vfvuff
      vks=vcks+voc
      vksf=vks
      dvdr=vcks/r
      dvdth=vcks/tanth
      dvdfib=vfvuff*r*comp*(sinfi*ylos+cosfi*xlos)
      difp=dif*deldel*renfsq
      if(komp.eq.2) goto 452
      binc=binc1
      binw=binw1
      do 1245 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      snthl=costh*dthf
      zz=r*(costh-sinth*dthf)
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1246 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1247
      dfif=dfloat(jfn)*r2nfdf*delfi
      dlr=0.d0
      xx=cmppd+compp*r*snthl*(cosfi-sinfi*dfif)
      yy=r*compp*snthl*(sinfi+cosfi*dfif)
      ysky=(xx-cmpd)*sinph+yy*cosph
      zsky=(cmpd-xx)*cicp+yy*cisp+zz*sini
      rrho=dsqrt(ysky*ysky+zsky*zsky)
      if(rrho.lt.rhho) goto 1246
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
 1247 call linpro(komp,dvks1,hbarw1,tau1,emm1,count1,taug,emmg,fbin1,
     $delv1)
      if(inmax.gt.in1max) in1max=inmax
      if(inmin.lt.in1min) in1min=inmin
 1246 continue
 1245 continue
      goto 126
  452 continue
      binc=binc2
      binw=binw2
      do 1445 ifn=-nfm1,nfm1,2
      dthf=dfloat(ifn)*r2nfdt
      snthl=costh*dthf
      zz=r*(costh-sinth*dthf)
      dvdfi=dvdfib*(sinth+costh*dthf)
      do 1446 jfn=-nfm1,nfm1,2
      if(nf.eq.1) goto 1447
      dfif=dfloat(jfn)*r2nfdf*delfi
      dvdth=-vflump*((cosfi-sinfi*dfif)*ylos-(sinfi+cosfi*dfif)*xlos)
      dlr=0.d0
      xx=cmppd+compp*r*snthl*(cosfi-sinfi*dfif)
      yy=r*compp*snthl*(sinfi+cosfi*dfif)
      ysky=(xx-cmpd)*sinph+yy*cosph
      zsky=(cmpd-xx)*cicp+yy*cisp+zz*sini
      rrho=dsqrt(ysky*ysky+zsky*zsky)
      if(rrho.lt.rhho) goto 1446
      vksf=vks+dvdr*dlr+dvdth*dthf+dvdfi*dfif
 1447 call linpro(komp,dvks2,hbarw2,tau2,emm2,count2,taug,emmg,fbin2,
     $delv2)
      if(inmax.gt.in2max) in2max=inmax
      if(inmin.lt.in2min) in2min=inmin
 1446 continue
 1445 continue
  126 CONTINUE
      SOMJ=SOMJ*DELFI
      SUMJ=SUMJ*DELFI
      SOM=SOM+SOMJ
  136 SUM=SUM+SUMJ
      if(mpage.eq.5) return
      IF(TEST.LE..0625d0) GOTO 120
      SOMHOT=SOM*DELTH
      SUMHOT=SUM*DELTH
      GOTO 121
  120 SUMKUL=SUM*DELTH
      SOMKUL=SOM*DELTH
  121 continue
      if(ifphn.eq.1) return
      if(mpage.ne.3) return
      in1min=in1min-marm1
      in1max=in1max+marp1
      in2min=in2min-marm2
      in2max=in2max+marp2
      if(nl1.eq.0) goto 3115
      do 2912 i=in1min,in1max
      fbin1(i)=1.d0-fbin1(i)/sumhot
      if(count1(i).eq.0.d0) goto 2918
      delv1(i)=delv1(i)/count1(i)
      goto 2919
 2918 delv1(i)=binw1*(dfloat(i)-binc1)
 2919 vdc=delv1(i)/clight
      vfc=dsqrt((1.d0+vdc)/(1.d0-vdc))
      delwl1(i)=wl*(vfc-1.d0)
      wl1(i)=wl*vfc
      resf1(i)=(sl1*delwl1(i)+sc1)*fbin1(i)
 2912 continue
 3115 if(nl2.eq.0) return
      do 2914 i=in2min,in2max
      fbin2(i)=1.d0-fbin2(i)/sumkul
      if(count2(i).eq.0.d0) goto 2917
      delv2(i)=delv2(i)/count2(i)
      goto 2920
 2917 delv2(i)=binw2*(dfloat(i)-binc2)
 2920 vdc=delv2(i)/clight
      vfc=dsqrt((1.d0+vdc)/(1.d0-vdc))
      delwl2(i)=wl*(vfc-1.d0)
      wl2(i)=wl*vfc
      resf2(i)=(sl2*delwl2(i)+sc2)*fbin2(i)
 2914 continue
      return
      END
