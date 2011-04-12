      subroutine lc(atmtab,pltab,lcin,request,vertno,L3perc,indeps,deps,
     +              skycoy,skycoz,params)
c
c  Main program for computing light and radial velocity curves,
c      line profiles, and images
c
c  Version of March 8, 2007
C
C     TO PRINT VELOCITIES IN KM/SEC, SET VUNIT=1.
C     TO PRINT NORMALIZED VELOCITIES IN SAME COLUMNS, SET VUNIT EQUAL TO
C     DESIRED VELOCITY UNIT IN KM/SEC.
C
C     PARAMETER PSHIFT IS DEFINED AS THE PHASE AT WHICH PRIMARY
C     CONJUNCTION (STAR 1 AWAY FROM OBSERVER) WOULD OCCUR IF THE
C     ARGUMENT OF PERIASTRON WERE 2 pi radians. SINCE THE NOMINAL VALUE
C     OF THIS QUANTITY IS ZERO, PSHIFT MAY BE USED TO INTRODUCE AN
C     ARBITRARY PHASE SHIFT.

      implicit real*8(a-h,o-z)

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c                      ARRAY DIMENSIONING WRAPPER
c                             May 8, 2008
c
c     The following parameters determine array sizing in the program.
c     There is no need to change any numbers in the code except these
c     in order to accomodate finer grids.
c
c        Nmax    ..    maximum grid fineness (parameters N1, N2)
c                        default:   Nmax =     60
c      igsmax    ..    maximum grid size depending on the grid fineness,
c                        i.e. igsmax=762 for N=30, 3011 for N=60 etc.
c                        default: igsmax =   3011
c      lpimax    ..    maximum dimension of line profile input arrays
c                        default: lpimax =    100
c      lpomax    ..    maximum dimension of line profile output arrays
c                        default: lpomax = 100000
c      ispmax    ..    maximum number of spots
c                        default: ispmax =    100
c      iclmax    ..    maximum number of clouds
c                        default: iclmax =    100
c      iplmax    ..    number of defined passbands
c                        default: iplmax =     25
c
      parameter (Nmax=     200)
      parameter (igsmax= 33202)
      parameter (lpimax=   100)
      parameter (lpomax=100000)
      parameter (ispmax=   100)
      parameter (iclmax=   100)
      parameter (iplmax=    48)
c
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
      parameter (itemppts=48)
      parameter (iloggpts=11)
      parameter (imetpts =19)
      parameter (iatmpts=iloggpts*itemppts)
      parameter (iatmchunk=iatmpts*iplmax)
      parameter (iatmsize=iatmchunk*imetpts)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Locations of the auxiliary files atmcof.dat and atmcofplanck.dat:
c
      character atmtab*(*),pltab*(*)
c
c     parameter (atmtab='atmcof.dat')
c     parameter ( pltab='atmcofplanck.dat')
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Other array dimensions that are set automatically are listed
c     below and should not be changed, as the above parameter statements
c     determine their values.
c
c        MMmax    ..    dimension of the array MMSAVE
c        immax    ..    maximum number of surface grid points in sky
c                       images
c       ifrmax    ..    dimension of the Fourier arrays
c       iplcof    ..    dimension of the atmcofplanck matrix, 50 per
c                       passband
c
      parameter (MMmax=2*Nmax+4)
      parameter (immax=4*igsmax+100)
      parameter (ifrmax=4*Nmax)
      parameter (iplcof=50*iplmax)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Finally, the following dimensions are considered static and
c     their size does not depend on parameters.
c
      dimension po(2)
      dimension message(2,4)
      dimension aa(20),bb(20)
c
c     Nothing needs to be changed beyond this point to accomodate
c     finer grids.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extensions:
c
c      request   ..   what do we want to compute:
c                       1  ..  light curve
c                       2  ..  primary RV curve
c                       3  ..  secondary RV curve
c                       4  ..  star shape
c       vertno   ..   number of vertices in a light/RV curve
c       L3perc   ..   3rd light switch:
c                       0  ..  3rd light passed in flux units (default)
c                       1  ..  3rd light computed from the passed
c                              percentage of L3, x=L3/(L1+L2+L3)
c       indeps   ..   an array of vertices (HJDs or phases)
c         deps   ..   an array of computed values (fluxes or RVs)
c       skycoy   ..   an array of y-coordinates of the plane of sky
c       skycoz   ..   an array of z-coordinates of the plane of sky
c       params   ..   an array of computed parameters:
c         lcin   ..   input lci filename
c
c                     params( 1) = L1     star 1 passband luminosity
c                     params( 2) = L2     star 2 passband luminosity
c                     params( 3) = M1     star 1 mass in solar masses
c                     params( 4) = M2     star 2 mass in solar masses
c                     params( 5) = R1     star 1 radius in solar radii
c                     params( 6) = R2     star 2 radius in solar radii
c                     params( 7) = Mbol1  star 1 absolute magnitude
c                     params( 8) = Mbol2  star 2 absolute magnitude
c                     params( 9) = logg1  star 1 log gravity
c                     params(10) = logg2  star 2 log gravity
c                     params(11) = SBR1   star 1 polar surface brightness
c                     params(12) = SBR2   star 2 polar surface brightness
c                     params(13) = phsv   star 1 potential
c                     params(14) = pcsv   star 2 potential
c
      integer request,vertno
      double precision indeps(*),deps(*),skycoy(*),skycoz(*),params(*)
      character lcin*(*)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      dimension rv(igsmax),grx(igsmax),gry(igsmax),grz(igsmax),
     $rvq(igsmax),grxq(igsmax),gryq(igsmax),grzq(igsmax),slump1(igsmax),
     $slump2(igsmax),fr1(igsmax),fr2(igsmax),glump1(igsmax),
     $glump2(igsmax),xx1(igsmax),xx2(igsmax),yy1(igsmax),yy2(igsmax),
     $zz1(igsmax),zz2(igsmax),grv1(igsmax),grv2(igsmax),rftemp(igsmax),
     $rf1(igsmax),rf2(igsmax),csbt1(igsmax),csbt2(igsmax),gmag1(igsmax),
     $gmag2(igsmax),glog1(igsmax),glog2(igsmax)
      dimension dvks1(lpimax),dvks2(lpimax),wll1(lpimax),wll2(lpimax),
     $tau1(lpimax),tau2(lpimax),emm1(lpimax),emm2(lpimax),ewid1(lpimax),
     $ewid2(lpimax),depth1(lpimax),depth2(lpimax),hbarw1(lpimax),
     $hbarw2(lpimax)
      dimension fbin1(lpomax),fbin2(lpomax),delv1(lpomax),delv2(lpomax),
     $count1(lpomax),count2(lpomax),delwl1(lpomax),delwl2(lpomax),
     $resf1(lpomax),resf2(lpomax),wl1(lpomax),wl2(lpomax),taug(lpomax),
     $emmg(lpomax)
      dimension XLAT(2,ispmax),xlong(2,ispmax)
      dimension xcl(iclmax),ycl(iclmax),zcl(iclmax),rcl(iclmax),
     $op1(iclmax),fcl(iclmax),dens(iclmax),encl(iclmax),edens(iclmax),
     $xmue(iclmax)
      dimension mmsave(MMmax),snth(2*Nmax),csth(2*Nmax),
     $snfi(2*igsmax+100),csfi(2*igsmax+100)
      dimension yskp(immax),zskp(immax)
      dimension theta(ifrmax),rho(ifrmax)
      dimension hld(igsmax),tld(2*igsmax)
      dimension plcof(iplcof)
      dimension abun(imetpts),glog(iloggpts),grand(iatmsize)

      common /abung/ abun,glog
      common /arrayleg/ grand,istart
      common /planckleg/ plcof
      common /atmmessages/ message,komp
      common /ramprange/ tlowtol,thightol,glowtol,ghightol
      COMMON /FLVAR/ PSHIFT,DP,EF,EFC,ECOS,perr0,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      COMMON /DPDX/ DPDX1,DPDX2,PHSV,PCSV
      COMMON /ECCEN/ E,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      COMMON /KFAC/ KFF1,KFF2,kfo1,kfo2
      COMMON /INVAR/ KH,IPB,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,ifsmv2,
     $icor1,icor2,ld,ncl,jdphs,ipc
      COMMON /SPOTS/ SINLAT(2,ispmax),COSLAT(2,ispmax),SINLNG(2,ispmax),
     $COSLNG(2,ispmax),RADSP(2,ispmax),temsp(2,ispmax),xlng(2,ispmax),
     $kks(2,ispmax),Lspot(2,ispmax)
      common /cld/ acm,opsf
      common /ardot/ dperdt,hjd,hjd0,perr
      common /prof2/ du1,du2,du3,du4,binw1,binw2,sc1,sc2,sl1,sl2,
     $clight
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      common /NSPT/ NSP1,NSP2

c
c           Bandpass Label Assignments for Stellar Atmospheres
c
c    Label   Bandpass   Reference for Response Function
c    -----   --------   -------------------------------
c       1        u      Crawford, D.L. and Barnes, J.V. 1974, AJ, 75, 978
c       2        v          "                "           "
c       3        b          "                "           "
c       4        y          "                "           "
c       5        U      Buser, R. 1978, Ang, 62, 411
c       6        B      Azusienis and Straizys 1969, Sov. Astron., 13, 316
c       7        V          "             "                "
c       8        R      Johnson, H.L. 1965, ApJ, 141, 923
c       9        I         "            "    "
c      10        J         "            "    "
c      11        K         "            "    "
c      12        L         "            "    "
c      13        M         "            "    "
c      14        N         "            "    "
c      15        R_c    Bessell, M.S. 1983, PASP, 95, 480
c      16        I_c       "            "    "
c      17      230      Kallrath, J., Milone, E.F., Terrell, D., Young, A.T.
c                          1998, ApJ, 508, 308
c      18      250         "             "             "           "
c      19      270         "             "             "           "
c      20      290         "             "             "           "
c      21      310         "             "             "           "
c      22      330         "             "             "           "
c      23     'TyB'    Tycho catalog B
c      24     'TyV'    Tycho catalog V
c      25     'HIP'    Hipparcos catalog
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      PHOEBE extensions:
c
c      26   CoRoT-exo  Carla Maceroni, private communication
c      27   CoRoT-sis  Carla Maceroni, private communication
c      28       H      Johnson, H.L. 1965, ApJ, 141, 923
c      29   Geneva U   Golay, M. 1962, Pub. Obs. Geneve No. 15 (serie A), 29
c      30   Geneva B       "             "             "           "
c      31   Geneva B1      "             "             "           "
c      32   Geneva B2      "             "             "           "
c      33   Geneva V       "             "             "           "
c      34   Geneva V1      "             "             "           "
c      35   Geneva G       "             "             "           "
c      36   Kepler     Kepler Science Book
c      37   SDSS u     Sloan DSS instrument book, Fukugita et al. (1996)
c      38   SDSS g     Sloan DSS instrument book, Fukugita et al. (1996)
c      39   SDSS r     Sloan DSS instrument book, Fukugita et al. (1996)
c      40   SDSS i     Sloan DSS instrument book, Fukugita et al. (1996)
c      41   SDSS z     Sloan DSS instrument book, Fukugita et al. (1996)
c      42   LSST u     LSST science book
c      43   LSST g     LSST science book
c      44   LSST r     LSST science book
c      45   LSST i     LSST science book
c      46   LSST z     LSST science book
c      47   LSST y3    LSST science book
c      48   LSST y4    LSST science book

      ot=1.d0/3.d0
      pi=dacos(-1.d0)
      clight=2.99792458d5
      en0=6.0254d23
      rsuncm=6.960d10
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  Ramp ranges are set below. The following values seem to work.
c  They may be changed.
      tlowtol=1500.d0
      thightol=50000.d0
      glowtol=4.0d0
      ghightol=4.0d0
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      abun(1)=1.d0
      abun(2)=0.5d0
      abun(3)=0.3d0
      abun(4)=0.2d0
      abun(5)=0.1d0
      abun(6)=0.0d0
      abun(7)=-0.1d0
      abun(8)=-0.2d0
      abun(9)=-0.3d0
      abun(10)=-0.5d0
      abun(11)=-1.0d0
      abun(12)=-1.5d0
      abun(13)=-2.0d0
      abun(14)=-2.5d0
      abun(15)=-3.0d0
      abun(16)=-3.5d0
      abun(17)=-4.0d0
      abun(18)=-4.5d0
      abun(19)=-5.0d0
      glog(1)=0.0d0
      glog(2)=0.5d0
      glog(3)=1.0d0
      glog(4)=1.5d0
      glog(5)=2.0d0
      glog(6)=2.5d0
      glog(7)=3.0d0
      glog(8)=3.5d0
      glog(9)=4.0d0
      glog(10)=4.5d0
      glog(11)=5.0d0
      nn=100
      gau=0.d0
      open(unit=22,file=atmtab,status='old')
      read(22,*) grand
      open(unit=23,file=pltab,status='old')
      read(23,*) plcof
      close(22)
      close(23)
      open(unit=15,file=lcin,status='old')
      open(unit=16,file='lcout.active')
      ibef=0
      nf1=1
      nf2=1
      DO 1000 IT=1,1000
      KH=17
      call rddata(mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,
     $jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed,hjdst,hjdsp,hjdin,
     $phstrt,phstop,phin,phn,mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,
     $the,vunit,e,a,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,
     $poth,potc,rm,xbol1,xbol2,ybol1,ybol2,iband,hlum,clum,xh,xc,yh,yc,
     $el3,opsf,zero,factor,wl,binwm1,sc1,sl1,wll1,ewid1,depth1,
     $kks,binwm2,sc2,sl2,wll2,ewid2,depth2,xlat,xlong,radsp,temsp,
     $xcl,ycl,zcl,rcl,op1,fcl,edens,xmue,encl,lpimax,ispmax,iclmax)

      if(mpage.ne.9) goto 414
      close(15)
      close(16)
      return
  414 continue
      message(1,1)=0
      message(1,2)=0
      message(2,1)=0
      message(2,2)=0
      message(1,3)=0
      message(1,4)=0
      message(2,3)=0
      message(2,4)=0
c***************************************************************
c  The following lines take care of abundances that may not be among
c  the 19 Kurucz values (see abun array). abunin is reset at the allowed value nearest
c  the input value.
      call binnum(abun,imetpts,abunin,iab)
      dif1=abunin-abun(iab)
      if(iab.eq.imetpts) goto 702
      dif2=abun(iab+1)-abun(iab)
      dif=dif1/dif2
      if((dif.ge.0.d0).and.(dif.le.0.5d0)) goto 702
      iab=iab+1
  702 continue
      abunir=abunin
      abunin=abun(iab)
      istart=1+(iab-1)*iatmchunk
c***************************************************************
      if(mpage.ne.3) goto 897
      colam=clight/wl
      binw1=colam*binwm1
      do 86 iln=1,lpimax
      if(wll1(iln).lt.0.d0) goto 89
      emm1(iln)=0.d0
      if(depth1(iln).lt.0.d0) emm1(iln)=depth1(iln)
      tau1(iln)=0.d0
      if(depth1(iln).gt.0.d0) tau1(iln)=-dlog(1.d0-depth1(iln))
      hbarw1(iln)=0.d0
      if(depth1(iln).ne.0.d0) hbarw1(iln)=.5d0*clight*ewid1(iln)/
     $(wll1(iln)*dabs(depth1(iln)))
      nl1=iln
   86 continue
   89 continue
      binw2=colam*binwm2
      do 99 iln=1,lpimax
      if(wll2(iln).lt.0.d0) goto 91
      emm2(iln)=0.d0
      if(depth2(iln).lt.0.d0) emm2(iln)=depth2(iln)
      tau2(iln)=0.d0
      if(depth2(iln).gt.0.d0) tau2(iln)=-dlog(1.d0-depth2(iln))
      hbarw2(iln)=0.d0
      if(depth2(iln).ne.0.d0) hbarw2(iln)=.5d0*clight*ewid2(iln)/
     $(wll2(iln)*dabs(depth2(iln)))
      nl2=iln
   99 continue
   91 continue
      do 622 iln=1,nl1
      flam=(wll1(iln)/wl)**2
  622 dvks1(iln)=clight*(flam-1.d0)/(flam+1.d0)
      do 623 iln=1,nl2
      flam=(wll2(iln)/wl)**2
  623 dvks2(iln)=clight*(flam-1.d0)/(flam+1.d0)
  897 continue
      NSP1=0
      NSP2=0
      DO 88 KP=1,2
      DO 87 I=1,ispmax
      xlng(kp,i)=xlong(kp,i)
      IF(XLAT(KP,I).GE.200.d0) GOTO 88
      SINLAT(KP,I)=dsin(XLAT(KP,I))
      COSLAT(KP,I)=dcos(XLAT(KP,I))
      SINLNG(KP,I)=dsin(XLONG(KP,I))
      COSLNG(KP,I)=dcos(XLONG(KP,I))
      IF(KP.EQ.1)NSP1=NSP1+1
   87 IF(KP.EQ.2)NSP2=NSP2+1
   88 CONTINUE
      ncl=0
      do 62 i=1,iclmax
      if(xcl(i).gt.100.d0) goto 66
      ncl=ncl+1
      dens(i)=edens(i)*xmue(i)/en0
   62 continue
   66 dint1=pi*(1.d0-xbol1/3.d0)
      dint2=pi*(1.d0-xbol2/3.d0)
      if(ld.eq.2) DINT1=dint1+PI*2.d0*ybol1/9.d0
      if(ld.eq.2) DINT2=dint2+PI*2.d0*ybol2/9.d0
      if(ld.eq.3) dint1=dint1-.2d0*pi*ybol1
      if(ld.eq.3) dint2=dint2-.2d0*pi*ybol2
      NSTOT=NSP1+NSP2
      NP1=N1+1
      NP2=N1+N2+2
      IRTE=0
      IRVOL1=0
      IRVOL2=0
      do 421 imm=1,MMmax
  421 mmsave(imm)=0
      nn1=n1
      CALL SINCOS(1,nn1,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)
      CALL SINCOS(2,N2,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)
      hjd=hjd0
      CALL modlog(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $rm,poth,potc,gr1,gr2,alb1,alb2,n1,n2,f1,f2,mod,xincl,the,mode,
     $snth,csth,snfi,csfi,grv1,grv2,xx1,yy1,zz1,xx2,yy2,zz2,glump1,
     $glump2,csbt1,csbt2,gmag1,gmag2,glog1,glog2)
      CALL VOLUME(VOL1,RM,POTH,DP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR1,1)
      CALL VOLUME(VOL2,RM,POTC,DP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR2,1)
      if(e.eq.0.d0) goto 117
      DAP=1.d0+E
      P1AP=POTH-2.d0*E*RM/(1.d0-E*E)
      VL1=VOL1
      CALL VOLUME(VL1,RM,P1AP,DAP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR1,2)
      DPDX1=(POTH-P1AP)*(1.d0-E*E)*.5d0/E
      P2AP=POTC-2.d0*E/(1.d0-E*E)
      VL2=VOL2
      CALL VOLUME(VL2,RM,P2AP,DAP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,
     $GMAG1,GMAG2,glog1,glog2,GR2,2)
      DPDX2=(POTC-P2AP)*(1.d0-E*E)*.5d0/E
  117 CONTINUE
      PHSV=POTH
      PCSV=POTC
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POTH,POTC,N1,N2,F1,F2,D,HLUM
     $,clum,xh,xc,yh,yc,gr1,gr2,wl,sm1,sm2,tpolh,tpolc,sbrh,sbrc,
     $tavh,tavc,alb1,alb2,xbol1,xbol2,ybol1,ybol2,phn,rm,xincl,
     $hot,cool,snth,csth,snfi,csfi,tld,glump1,glump2,xx1,xx2,yy1,yy2,
     $zz1,zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,
     $gmag2,glog1,glog2,fbin1,fbin2,delv1,delv2,count1,count2,delwl1,
     $delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,
     $hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,
     $zskp,mode,iband,ifat1,ifat2,1)
      KH=0
      rr1=.6203505d0*vol1**ot
      rr2=.6203505d0*vol2**ot
      tav1=10000.d0*tavh
      tav2=10000.d0*tavc
      call mlrg(a,period,rm,rr1,rr2,tav1,tav2,sms1,sms2,sr1,sr2,
     $bolm1,bolm2,xlg1,xlg2)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
      params( 1) = hlum
      params( 2) = clum
      params( 3) = sms1
      params( 4) = sms2
      params( 5) = sr1
      params( 6) = sr2
      params( 7) = bolm1
      params( 8) = bolm2
      params( 9) = xlg1
      params(10) = xlg2
      params(11) = sbrh
      params(12) = sbrc
      params(13) = phsv
      params(14) = pcsv
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      call wrhead(ibef,nref,mref,ifsmv1,ifsmv2,icor1,icor2,
     $ld,jdphs,hjd0,period,dpdt,pshift,stdev,noise,seed,hjdst,hjdsp,
     $hjdin,phstrt,phstop,phin,phn,mode,ipb,ifat1,ifat2,n1,n2,perr0,
     $dperdt,the,vunit,vfac,e,a,f1,f2,vga,xincl,gr1,gr2,nsp1,nsp2,
     $abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,
     $iband,hlum,clum,xh,xc,yh,yc,el3,opsf,zero,factor,wl,binwm1,sc1,
     $sl1,binwm2,sc2,sl2,wll1,ewid1,depth1,wll2,ewid2,depth2,
     $kks,xlat,xlong,radsp,temsp,ncl,xcl,ycl,zcl,rcl,op1,fcl,edens,xmue,
     $encl,dens,ns1,sms1,sr1,bolm1,xlg1,ns2,sms2,sr2,bolm2,xlg2,mmsave,
     $sbrh,sbrc,sm1,sm2,phperi,pconsc,pconic,dif1,abunir,abun,mod)

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
c     The following block supports third light to be computed from the
c     passed percentage of third luminosity.
c
      if(L3perc.eq.1) el3=(hlum+clum)*el3/(4.d0*3.141593d0*(1.d0-el3))
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ALL=HOT+COOL+EL3
      IF(MODE.EQ.-1) ALL=COOL+EL3
      LL1=MMSAVE(N1)+1
      NPP2=NP2-1
      LL2=MMSAVE(NPP2)+1
      LLL1=MMSAVE(NP1)
      LLL2=MMSAVE(NP2)
      LLLL1=(LL1+LLL1)/2
      LLLL2=(LL2+LLL2)/2
      POTH=PHSV
      POTC=PCSV
      PO(1)=POTH
      PO(2)=POTC
      IF(E.EQ.0.d0) IRVOL1=1
      IF(E.EQ.0.d0) IRVOL2=1
      IF(E.EQ.0.d0) IRTE=1
      start=hjdst
      stopp=hjdsp
      step=hjdin
      if(jdphs.ne.2) goto 887
      start=phstrt
      stopp=phstop
      step=phin
  887 continue
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
c     do 20 phjd=start,stopp,step
c     hjdi=phjd
c     phasi=phjd
c
      do 20 idx=1,vertno
      hjdi=indeps(idx)
      phasi=indeps(idx)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      call jdph(hjdi,phasi,hjd0,period,dpdt,hjdo,phaso)
      hjd=hjdi
      phas=phasi
      if(jdphs.ne.1) hjd=hjdo
      if(jdphs.ne.2) phas=phaso
      CALL modlog(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $rm,poth,potc,gr1,gr2,alb1,alb2,n1,n2,f1,f2,mod,xincl,the,mode,
     $snth,csth,snfi,csfi,grv1,grv2,xx1,yy1,zz1,xx2,yy2,zz2,glump1,
     $glump2,csbt1,csbt2,gmag1,gmag2,glog1,glog2)
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POTH,POTC,N1,N2,F1,F2,D,hlum,
     $clum,xh,xc,yh,yc,gr1,gr2,wl,sm1,sm2,tpolh,tpolc,sbrh,sbrc,
     $tavh,tavc,alb1,alb2,xbol1,xbol2,ybol1,ybol2,phas,rm,xincl,
     $hot,cool,snth,csth,snfi,csfi,tld,glump1,glump2,xx1,xx2,yy1,yy2,
     $zz1,zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,
     $gmag2,glog1,glog2,fbin1,fbin2,delv1,delv2,count1,count2,delwl1,
     $delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,
     $hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,
     $zskp,mode,iband,ifat1,ifat2,0)

      HTT=HOT
      IF(MODE.EQ.-1) HTT=0.d0
      TOTAL=HTT+COOL+EL3
      TOTALL=TOTAL/ALL
      TOT=TOTALL*FACTOR
      if(stdev.le.0.d0) goto 348
      call rangau(seed,nn,stdev,gau)
      ranf=1.d0+gau*dsqrt(totall**noise)
      total=total*ranf
      tot=tot*ranf
      totall=totall*ranf
  348 continue
      SMAGG=-1.085736d0*dlog(TOTALL)+ZERO
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
      if (request.eq.1) deps(idx)=total
      if (request.eq.2) deps(idx)=vkm1
      if (request.eq.3) deps(idx)=vkm2
      if (request.eq.4) then
        do 129 imp=1,ipc
          skycoy(imp) = yskp(imp)
          skycoz(imp) = zskp(imp)
  129   continue
      endif
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      call wrdata(hjd,phas,yskp,zskp,htt,cool,total,tot,d,smagg,
     $vsum1,vsum2,vra1,vra2,vkm1,vkm2,delv1,delwl1,wl1,fbin1,resf1,
     $delv2,delwl2,wl2,fbin2,resf2,rv,rvq,mmsave,ll1,lll1,llll1,
     $ll2,lll2,llll2)

   20 CONTINUE

      call wrfoot(message,f1,f2,po,rm,f,dp,e,drdq,dodq,ii,mode,
     $mpage)

      ibef=1

 1000 CONTINUE
      STOP
      END
