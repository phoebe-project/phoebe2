      subroutine dc(atmtab,pltab,L3perc,knobs,indeps,fluxes,weights,
     $              nph,delph,corrs,stdevs,chi2s,cormat,ccla,cfval)

c  This is the Differential Corrections Main Program.
c
c  Version of May 24, 2007
C
C     PARAMETER NUMBER 9 IS A, THE RELATIVE ORBITAL SEMI-MAJOR AXIS, IF
C     SIMULTANEOUS LIGHT AND VELOCITY SOLUTIONS ARE BEING DONE. HOWEVER,
C     IF ONLY VELOCITY CURVES ARE BEING SOLVED, PARAMETER 9 WILL
C     EFFECTIVELY BE A*SIN(I), PROVIDED AN INCLINATION OF 90 DEG. IS
C     ENTERED. IN SOME RARE SITUATIONS IT MAY BE POSSIBLE TO
C     FIND A AND I SEPARATELY from velocities only. THIS COULD BE
C     THE CASE IF THE VELOCITY PROXIMITY EFFECTS ARE IMPORTANT.
C
C     OTHER PROGRAM UNITS: ORBITAL S-M AXIS IN SOLAR RADII (6.96d5 KM),
C     PERIOD IN DAYS, PHASE IN 2 PI RADIANS, SYSTEMIC VELOCITY AND
C     THIRD LIGHT IN SAME UNITS AS VELOCITY AND LIGHT OBSERVATIONS,
C     INCLINATION IN DEGREES, TEMPERATURES IN 10000K., SPOT LATITUDES
C     IN RADIANS (0=NORTH POLE, Pi=SOUTH POLE), SPOT LONGITUDES
C     IN RADIANS (0=LINE OF CENTERS MERIDIAN, INCREASING COUNTER-
C     CLOCKWISE AS SEEN FROM NORTH POLE TO 2 Pi),
C     SPOT ANGULAR RADII IN RADIANS. SPOT TEMPERATURE FACTOR IS
C     DIMENSIONLESS.
C
      implicit real*8 (a-h,o-z)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c                      ARRAY DIMENSIONING WRAPPER
c                            March 6, 2007
c
c     The following parameters are maximum array sizes.
c     There is no need to change any dimension numbers except these
c     in order to accomodate finer grids.
c
c        Nmax    ..    maximum grid fineness parameters N1 and N2
c                        default:   Nmax =    100
c      igsmax    ..    maximum number of surface elements (depends on N1 and N2)
c                        e.g. igsmax=762 for N=30, 3011 for N=60, etc.
c                        default: igsmax =   8331
c      ispmax    ..    maximum number of spots
c                        default: ispmax =    100
c      iclmax    ..    maximum number of clouds
c                        default: iclmax =    100
c      iptmax    ..    maximum number of observed data points, including 
c                        blank points on last lines of the velocity and light curve
c                        data sets and on stop lines
c                        default: iptmax =  10000
c       ncmax    ..    maximum number of input data curves (velocity +light)
c                        default: 50
c      iplmax    ..    maximum number of passbands
c                        default: iplmax =     26
c       ipmax    ..    maximum number of parameters that are actually
c            adjusted, with band-independent parameters counted once each and
c            band-dependent parameters counted N_band times each.
c                        default: ipmax= 50
c
      parameter (Nmax=     200)
      parameter (igsmax= 33202)
      parameter (ispmax=   100)
      parameter (iclmax=   100)
      parameter (iptmax= 50000)
      parameter (ncmax=     50)
      parameter (iplmax=    48)
      parameter (ipmax=     50)
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
c        MMmax    ..    maximum dimension of the MMSAVE array
c       ifrmax    ..    maximum dimension of the horizon polar
c                       coordinate arrays
c       istmax    ..    maximum dimension of storage arrays OBS and HOLD 
c                         (iptmax * (no. of adjusted parameters + 1).
c       iplcof    ..    dimension of the atmcofplanck matrix, 50 per
c                       passband
c
      parameter (MMmax=2*Nmax+4)
      parameter (ifrmax=4*Nmax)
      parameter (istmax=iptmax*ipmax)
      parameter (iplcof=50*iplmax)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Finally, the following dimensions are considered static and
c     their size does not depend on parameters.
c
c       ichno    ..    number of parameter channels (currently 35)
c
      parameter (ichno=35)
      dimension xtha(4),xfia(4),po(2),omcr(2)
      dimension message(2,4)
      character arad(4)*10
      dimension aa(20),bb(20)
c
c     Nothing needs to be changed beyond this point to accomodate finer grids.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extensions:
c
c       L3perc   ..   switch whether third light is given in percent
c        knobs   ..   cummulative number of observations
c       indeps   ..   array of data HJDs or phases
c       fluxes   ..   array of data fluxes
c      weights   ..   array of data weights
c        corrs   ..   an array of computed corrections
c       stdevs   ..   standard deviations of fitted parameters
c        chi2s   ..   chi2 values of individual curves after the fit
c       cormat   ..   correlation matrix (a wrapped 1D array)
c         ccla   ..   computed CLA values
c        cfval   ..   cost function value (global goodness-of-fit value)
c          nph   ..   finite time integration oversampling rate
c        delph   ..   finite time integration cadence
c
      integer L3perc,knobs(*),nph
      double precision indeps(*),fluxes(*),weights(*),cfval,delph
      double precision corrs(*),stdevs(*),chi2s(*),cormat(*),ccla(*)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      dimension rv(igsmax),grx(igsmax),gry(igsmax),grz(igsmax),
     $rvq(igsmax),grxq(igsmax),gryq(igsmax),grzq(igsmax),slump1(igsmax),
     $slump2(igsmax),srv(igsmax),sgrx(igsmax),sgry(igsmax),sgrz(igsmax),
     $srvq(igsmax),sgrxq(igsmax),sgryq(igsmax),sgrzq(igsmax),
     $srvl(igsmax),sgrxl(igsmax),sgryl(igsmax),sgrzl(igsmax),
     $srvql(igsmax),sgrxql(igsmax),sgryql(igsmax),sgrzql(igsmax),
     $slmp1(igsmax),slmp2(igsmax),slmp1l(igsmax),slmp2l(igsmax),
     $fr1(igsmax),fr2(igsmax),glump1(igsmax),glump2(igsmax),
     $grv1(igsmax),grv2(igsmax),xx1(igsmax),xx2(igsmax),yy1(igsmax),
     $yy2(igsmax),zz1(igsmax),zz2(igsmax),gmag1(igsmax),gmag2(igsmax),
     $csbt1(igsmax),csbt2(igsmax),rf1(igsmax),rf2(igsmax),
     $rftemp(igsmax),sxx1(igsmax),sxx2(igsmax),syy1(igsmax),
     $syy2(igsmax),szz1(igsmax),szz2(igsmax),sgmg1(igsmax),
     $sgmg2(igsmax),sgrv1(igsmax),sgrv2(igsmax),sglm1(igsmax),
     $sglm2(igsmax),scsb1(igsmax),scsb2(igsmax),srf1(igsmax),
     $srf2(igsmax),sglm1l(igsmax),sglm2l(igsmax),sgrv1l(igsmax),
     $sgrv2l(igsmax),sxx1l(igsmax),sxx2l(igsmax),syy1l(igsmax),
     $syy2l(igsmax),szz1l(igsmax),szz2l(igsmax),sgmg1l(igsmax),
     $sgmg2l(igsmax),scsb1l(igsmax),scsb2l(igsmax),srf1l(igsmax),
     $srf2l(igsmax),glog1(igsmax),glog2(igsmax),erv(igsmax),
     $egrx(igsmax),egry(igsmax),egrz(igsmax),elmp1(igsmax),
     $eglm1(igsmax),egrv1(igsmax),exx1(igsmax),eyy1(igsmax),
     $ezz1(igsmax),egmg1(igsmax),ecsb1(igsmax),erf1(igsmax),
     $ervq(igsmax),egrxq(igsmax),egryq(igsmax),egrzq(igsmax),
     $elmp2(igsmax),eglm2(igsmax),egrv2(igsmax),exx2(igsmax),
     $eyy2(igsmax),ezz2(igsmax),egmg2(igsmax),ecsb2(igsmax),
     $erf2(igsmax),ervl(igsmax),egrxl(igsmax),egryl(igsmax),
     $egrzl(igsmax),elmp1l(igsmax),eglm1l(igsmax),egrv1l(igsmax),
     $exx1l(igsmax),eyy1l(igsmax),ezz1l(igsmax),egmg1l(igsmax),
     $ecsb1l(igsmax),erf1l(igsmax),ervql(igsmax),egrxql(igsmax),
     $egryql(igsmax),egrzql(igsmax),elmp2l(igsmax),eglm2l(igsmax),
     $egrv2l(igsmax),exx2l(igsmax),eyy2l(igsmax),ezz2l(igsmax),
     $egmg2l(igsmax),ecsb2l(igsmax),erf2l(igsmax),sfr1(igsmax),
     $sfr1l(igsmax),efr1(igsmax),efr1l(igsmax),sfr2(igsmax),
     $sfr2l(igsmax),efr2(igsmax),efr2l(igsmax)
      dimension stldh(2*igsmax),stldl(2*igsmax),etldh(2*igsmax),
     $etldl(2*igsmax)
      dimension obs(istmax),hold(istmax)
      dimension xcl(iclmax),ycl(iclmax),zcl(iclmax),rcl(iclmax),
     $op1(iclmax),fcl(iclmax),dens(iclmax),encl(iclmax),
     $edens(iclmax),xmue(iclmax)
      dimension xlat(2,ispmax),xlong(2,ispmax),radsp(2,ispmax),
     $temsp(2,ispmax)
      dimension mmsavh(MMmax),mmsavl(MMmax)
      dimension phas(iptmax),flux(iptmax),wt(iptmax),br(iptmax),
     $bl(iptmax),phjd(iptmax),dfdph(iptmax),dfdap(iptmax)
      dimension hla(ncmax),cla(ncmax),x1a(ncmax),x2a(ncmax),y1a(ncmax),
     $y2a(ncmax),el3a(ncmax),wla(ncmax),noise(ncmax),sigma(ncmax),
     $opsfa(ncmax),iband(ncmax)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c     Removed: knobs(ncmax+2)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dimension snthh(2*Nmax),csthh(2*Nmax),snthl(2*Nmax),csthl(2*Nmax),
     $snfih(2*igsmax),csfih(2*igsmax),snfil(2*igsmax),csfil(2*igsmax)
      dimension hld(igsmax),tldh(2*igsmax),tldl(2*igsmax)
      dimension theta(ifrmax),rho(ifrmax)
      dimension del(ichno),keep(ichno+1),kep(ichno),nshift(ichno+1),
     $low(ichno),dlif(ichno)
      dimension clc(ipmax),out(ipmax),sd(ipmax),ccl(ipmax),ll(ipmax),
     $mm(ipmax),cnc(ipmax**2),cn(ipmax**2),cnn(ipmax**2)
      dimension para(30+5*ncmax),v(ipmax),cnout(ipmax**2)
      dimension plcof(iplcof)
      dimension abun(imetpts),glog(iloggpts),grand(iatmsize)
c
c The following dimensioned variables are not used by DC. They are
c    dimensioned only for compatibility with usage of subroutine
c    LIGHT by program LC.
c
      dimension fbin1(1),fbin2(1),delv1(1),delv2(1),count1(1),
     $count2(1),delwl1(1),delwl2(1),resf1(1),resf2(1),wl1(1),wl2(1),
     $dvks1(1),dvks2(1),tau1(1),tau2(1),taug(1),hbarw1(1),hbarw2(1),
     $emm1(1),emm2(1),emmg(1),yskp(1),zskp(1)
      common /abung/ abun,glog
      common /arrayleg/ grand,istart
      common /planckleg/ plcof
      common /atmmessages/ message,komp
      common /ramprange/ tlowtol,thightol,glowtol,ghightol
      COMMON /ECCEN/ EC,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      COMMON /DPDX/ DPDX1,DPDX2,POT1,POT2
      COMMON /SUMM/ SUMM1,SUMM2
      common /ardot/ dperdt,hjd,hjd0,perdum
      COMMON /FLVAR/ PSH,DP,EF,EFC,ECOS,pert,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      COMMON /INVAR/ KH,IPB,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,ifsmv2,
     $icor1,icor2,ld,ncl,jdphs,ipc
      COMMON /SPOTS/ SNLAT(2,ispmax),CSLAT(2,ispmax),SNLNG(2,ispmax),
     $CSLNG(2,ispmax),rdsp(2,ispmax),tmsp(2,ispmax),xlng(2,ispmax),
     $kks(2,ispmax),Lspot(2,ispmax)
      COMMON /NSPT/ NSP1,NSP2
      common /cld/ acm,opsf
      common /ipro/ nbins,nl,inmax,inmin,nf1,nf2
      common /prof2/ duma,dumb,dumc,dumd,du1,du2,du3,du4,du5,du6,du7
      common /inprof/ in1min,in1max,in2min,in2max,mpage,nl1,nl2
      DATA ARAD(1),ARAD(2),ARAD(3),ARAD(4)/'POLE','POINT','SIDE','BACK'/
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
c
   15 FORMAT(1X,16(F11.5))
   16 FORMAT(1X,18(F7.4))
   67 FORMAT(20A4)
   17 FORMAT(1X,22(F6.3))
   19 FORMAT(1X,26(F5.2))
   21 FORMAT(f27.16,f36.16,d22.6)
   55 FORMAT(10(3X,d8.1))
   56 FORMAT(10(1X,d7.1))
   20 format(1x,2(4i1,1x),7i1,1x,4(5i1,1x),i1,1x,i1,1x,i1,d10.3)
  101 FORMAT(' ')
    1 FORMAT(I3,I6,I6,I7,I7,I5,I5,I5,f15.6,d13.5,f10.5,f16.3,f14.4)
  701 FORMAT(4I2,4I4,f13.6,d12.5,F8.5,F9.3)
c   2 FORMAT(5(F14.5,F8.4,F6.2))
    2 FORMAT(5(F14.5,F8.4,F12.2))
   85 FORMAT(i3,2F10.5,4(1X,F6.3),f8.4,d10.3,i6,d14.5,f10.6)
   18 format(i3,2f10.5,4f7.3,f8.4,d10.3,i2,d12.5,f10.6)
  218 FORMAT(i3,2F10.5,4F7.3,d10.3,d12.5,f10.6)
   37 FORMAT(1X,11F12.7)
  137 FORMAT(1X,F11.7)
  715 format(22x,'Input-Output in F Format')
  716 format(22x,'Input-Output in D Format')
   43 format('No.',2x,'Curve',4x,'Input Param.',8x,'Correction',5x,
     $'Output Param.',4x,'Standard Deviation')
  615 format(i2,i7,4f18.10)
  616 format(i2,i7,4d18.10)
  138 FORMAT(1X,'SUM OF ABSOLUTE VALUES OF CHECKS IS',1X,D12.6)
  181 FORMAT(7X,'NORMAL EQUATIONS')
  183 FORMAT (7X,'CORRELATION COEFFICIENTS')
  184 FORMAT(7X,'NORMAL EQUATIONS TIMES INVERSE')
  185 FORMAT(1X,'CHECK OF COMPUTED DEPENDENT VARIABLES FROM NORMAL EQUAT
     $IONS')
   82 FORMAT(7X,'UNWEIGHTED OBSERVATIONAL EQUATIONS')
   83 FORMAT(7X,'WEIGHTED OBSERVATIONAL EQUATIONS')
    9 FORMAT(33X,'OBSERVATIONS')
  955 FORMAT(3(9x,'phase   V rad   wt'))
   10 FORMAT(3(9x,'phase   light   wt'))
  755 FORMAT(3(9x,'JD      V rad   wt'))
  756 FORMAT(3(9x,'JD      light   wt'))
   40 FORMAT('   Sum(W*Res**2) for input values       Sum(W*Res**2) pred
     $icted          determinant')
   11 format(1x,'band',5x,'L1',8x,'L2     x1     x2     y1     y2   3rd
     $lt',4x,'opsf     NOISE    Sigma',6x,'Wave L')
  111 FORMAT(1x,'band',5x,'L1',8x,'L2     x1    x2    y1    y2     ops
     $f',7x,'Sigma',6x,'Wave L')
   12 FORMAT('MODE   IPB  IFAT1  IFAT2   N1   N2  N1L  N2L',7x,'Arg Per'
     $,5x,'dperdt',7x,'TH e',8x,'V unit(km/s)     V FAC')
  206 FORMAT(F6.5,d13.6,2F11.4,F11.4,f10.3,2f8.3,i6,i9,f9.2,i3)
  205 FORMAT('  ecc',4x,'S-M axis',7x,'F1',9x,'F2',8x,'V Gam',
     $7x,'INCL      G1      G2',2x,'Nspot 1',2x,'Nspot 2','  [M/H] iab')
  402 FORMAT('    DEL EC     DEL PER    DEL F1     DEL F2     DEL PHS
     $ DEL INCL    DEL G1     DEL G2     DEL T1     DEL T2')
  403 FORMAT('    DEL ALB1   DEL ALB2   DEL POT1   DEL POT2   DEL Q
     $ DEL L1     DEL L2     DEL X1     DEL X2')
  406 FORMAT(' ADJUSTMENT CONTROL INTEGERS; 1 SUPPRESSES ADJUSTMENT, 0 A
     $LLOWS ADJUSTMENT.')
  702 FORMAT(F6.5,d13.6,2F10.4,F10.4,f9.3,2f7.3,f7.2)
  706 FORMAT(F7.4,f8.4,2F7.3,3d13.6,4f7.3)
  408 FORMAT(2F8.4,2F9.3,2d15.6,d13.6,4f9.3)
  705 FORMAT(I1,1X,I1,1X,5I2)
   54 FORMAT('     T1      T2',5x,'Alb 1',4x,'Alb 2',9x,'Pot 1',10x,
     $'Pot 2',8x,'M2/M1  x1(bolo) x2(bolo) y1(bolo) y2(bolo)')
  707 FORMAT('    IFVC1   IFVC2   NLC   KO   KDISK   ISYM   nppl')
  917 format('nref',3x,'mref',3x,'ifsmv1',3x,'ifsmv2',3x,'icor1',3x,
     $'icor2',3x,'ld')
  708 FORMAT(8(4X,I3))
  912 format(i3,i7,i8,i9,i8,i8,i7)
  650 FORMAT(20X,'RADII AND RELATED QUANTITIES (FROM INPUT)')
  651 FORMAT(5X,'DOM1/DQ',5X,'DOM2/DQ',5X,'OM1-Q CORR.',5X,'OM2-Q CORR.'
     $,5X,'OM1 S.D.',4X,'OM2 S.D.',4X,'Q  S.D.')
  652 FORMAT(1X,3F12.6,4X,F12.6,4X,3F12.6)
  653 FORMAT(' COMPONENT',11X,'R',9X,'DR/DOM',8X,'DR/DQ',11X,'S.D.')
  654 FORMAT(2X,I1,1X,A6,4F14.6)
  684 FORMAT(i2,4F13.5)
  985 FORMAT(4f9.5)
  983 FORMAT(1X,'STAR  CO-LATITUDE  LONGITUDE  SPOT RADIUS  TEMP.FACTOR
     $')
  399 FORMAT('    DEL LAT    DEL LONG   DEL RAD    DEL TEMPF  DEL LAT
     $ del LONG   del RAD    del TEMPF')
   60 FORMAT(4I3)
   61 FORMAT(1X,4I6)
   66 FORMAT('   STAR  SPOT   STAR  SPOT')
  166 FORMAT(' SPOTS TO BE ADJUSTED')
  440 FORMAT(' AS1=FIRST ADJUSTED SPOT')
  441 FORMAT(' AS2=SECOND ADJUSTED SPOT')
  405 FORMAT(' ORDER OF PARAMETERS IS AS FOLLOWS:')
 1440 FORMAT('  (1) - AS1 LATITUDE')
 1441 FORMAT('  (2) - AS1 LONGITUDE')
 1442 FORMAT('  (3) - AS1 ANGULAR RADIUS')
 1443 FORMAT('  (4) - AS1 TEMPERATURE FACTOR')
 1444 FORMAT('  (5) - AS2 LATITUDE')
 1445 FORMAT('  (6) - AS2 LONGITUDE')
 1446 FORMAT('  (7) - AS2 ANGULAR RADIUS')
 1447 FORMAT('  (8) - AS2 TEMPERATURE FACTOR')
 1448 FORMAT('  (9) - A=ORBITAL SEMI-MAJOR AXIS')
 1449 FORMAT(' (10) - E=ORBITAL ECCENTRICITY')
 1450 FORMAT(' (11) - PERR0=ARGUMENT of PERIASTRON at time HJD0')
 1451 FORMAT(' (12) - F1=STAR 1 ROTATION PARAMETER')
 1452 FORMAT(' (13) - F2=STAR 2 ROTATION PARAMETER')
 1453 FORMAT(' (14) - PHASE SHIFT= PHASE OF PRIMARY CONJUNCTION')
 1454 FORMAT(' (15) - VGAM=SYSTEMIC RADIAL VELOCITY')
 1455 FORMAT(' (16) - INCL=ORBITAL INCLINATION')
 1456 FORMAT(' (17) - g1=STAR 1 GRAVITY DARKENING EXPONENT')
 1457 FORMAT(' (18) - g2=STAR 2 GRAVITY DARKENING EXPONENT')
 1458 FORMAT(' (19) - T1=STAR 1 AVERAGE SURFACE TEMPERATURE')
 1459 FORMAT(' (20) - T2=STAR 2 AVERAGE SURFACE TEMPERATURE')
 1460 FORMAT(' (21) - ALB1=STAR 1 BOLOMETRIC ALBEDO')
 1461 FORMAT(' (22) - ALB2=STAR 2 BOLOMETRIC ALBEDO')
 1462 FORMAT(' (23) - POT1=STAR 1 SURFACE POTENTIAL')
 1463 FORMAT(' (24) - POT2=STAR 2 SURFACE POTENTIAL')
 1464 FORMAT(' (25) - Q=MASS RATIO (STAR 2/STAR 1)')
 1470 FORMAT(' (26) - HJD0= Hel. JD reference time')
 1471 FORMAT(' (27) - PERIOD= orbital period')
 1472 FORMAT(' (28) - DPDT= time derivative of orbital period')
 1473 FORMAT(' (29) - DPERDT= time derivative of argument of periastron'
     $)
 1474 FORMAT(' (30) - unused channel reserved for future expansion')
 1465 FORMAT(' (31) - L1=STAR 1 RELATIVE MONOCHROMATIC LUMINOSITY')
 1466 FORMAT(' (32) - L2=STAR 2 RELATIVE MONOCHROMATIC LUMINOSITY')
 1467 FORMAT(' (33) - X1=STAR 1 LIMB DARKENING COEFFICIENT')
 1468 FORMAT(' (34) - X2=STAR 2 LIMB DARKENING COEFFICIENT')
 1469 FORMAT(' (35) - el3=third light')
  119 format(1x,i6,i13,f18.8)
  159 format(' Sums of squares of residuals for separate curves, includi
     $ng only individual weights')
  169 format('    Curve     No. of obs.   Sum of squares')
 1063 format(3f9.4,f7.4,d11.4,f9.4,d11.3,f9.4,f7.3)
   64 format(3f10.4,f9.4,d12.4,f10.4,d12.4,f9.4,f9.3,d12.4)
   69 format('      xcl       ycl       zcl      rcl       op1         f
     $cl        ne       mu e      encl     dens')
  170 format(i3,f17.6,d18.10,d14.6,f10.4,f11.5,i5)
  649 format(i1,f15.6,d17.10,d14.6,f10.4)
  171 format('JDPHS',5x,'J.D. zero',7x,'Period',11x,'dPdt',
     $6x,'Ph. shift')
  911 format(7(i1,1x))
  840 format('Do not try to adjust the ephemeris or any time derivative
     $parameters','when JDPHS = 2')
  839 format('Ordinarily one should not try to adjust both PSHIFT and',
     $' HJD0. They are perfectly correlated if the period is constant',
     $' and extremely highly correlated if the period is not constant')
  283 format('log g below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  284 format('log g above ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  285 format('T above ramp range for at least one',
     $' point on star',i2,', black body applied locally.')
  286 format('T below ramp range for at least one point',
     $' on star',i2,', black body applied locally.')
  287 format('Input [M/H] = ',f6.3,' is not a value recognized by ',
     $'the program. Replaced by ',f5.2)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  Ramp ranges are set here. The values below seem to work well, however,
c  they can be changed.
      tlowtol=1500.d0
      thightol=50000.d0
      glowtol=4.0d0
      ghightol=4.0d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
      message(1,1)=0
      message(1,2)=0
      message(2,1)=0
      message(2,2)=0
      message(1,3)=0
      message(1,4)=0
      message(2,3)=0
      message(2,4)=0
      open(unit=22,file=atmtab,status='old')
      read(22,*) grand
      close (22)
      open(unit=23,file=pltab,status='old')
      read(23,*) plcof
      close (23)
      open(unit=15,file='dcin.active',status='old')
      open(unit=16,file='dcout.active')
      tt=2.d0/3.d0
      nbins=1
      nl=1
      inmax=1
      inmin=1
      nf1=1
      nf2=1
      toldis=1.d-5
      pi=dacos(-1.d0)
      en0=6.0254d23
      XTHA(1)=0.d0
      XTHA(2)=.5d0*PI
      XTHA(3)=.5d0*PI
      XTHA(4)=.5d0*PI
      XFIA(1)=0.d0
      XFIA(2)=0.d0
      XFIA(3)=.5d0*PI
      XFIA(4)=PI
c
c  The initializations in th 886 and 887 loops are just to avoid
c    triggering error messages from some compilers. The quantities do
c    not otherwise need initialization. Same for du1, ...., du7.
c    Same for mpage, nl1, nl2.
c
      do 886 ikks=1,2
      do 886 jkks=1,ispmax
  886 kks(ikks,jkks)=0
      do 887 immsav=1,MMmax
      mmsavh(immsav)=0
  887 mmsavl(immsav)=0
      du1=0.d0
      du2=0.d0
      du3=0.d0
      du4=0.d0
      du5=0.d0
      du6=0.d0
      du7=0.d0
      mpage=0
      nl1=0
      nl2=0
      IBEF=0
      KH=25
      NS=1
      NI=0
      NY=0
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c     Commented out: KNOBS(1)=0
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      WRITE(16,405)
      WRITE(16,101)
      WRITE(16,440)
      WRITE(16,441)
      WRITE(16,101)
      WRITE(16,1440)
      WRITE(16,1441)
      WRITE(16,1442)
      WRITE(16,1443)
      WRITE(16,1444)
      WRITE(16,1445)
      WRITE(16,1446)
      WRITE(16,1447)
      WRITE(16,1448)
      WRITE(16,1449)
      WRITE(16,1450)
      WRITE(16,1451)
      WRITE(16,1452)
      WRITE(16,1453)
      WRITE(16,1454)
      WRITE(16,1455)
      WRITE(16,1456)
      WRITE(16,1457)
      WRITE(16,1458)
      WRITE(16,1459)
      WRITE(16,1460)
      WRITE(16,1461)
      WRITE(16,1462)
      WRITE(16,1463)
      WRITE(16,1464)
      WRITE(16,1470)
      WRITE(16,1471)
      WRITE(16,1472)
      WRITE(16,1473)
      WRITE(16,1474)
      WRITE(16,1465)
      WRITE(16,1466)
      WRITE(16,1467)
      WRITE(16,1468)
      WRITE(16,1469)
      READ(15,56)(DEL(I),I=1,8)
      READ(15,56)(DEL(I),I=10,14),(DEL(I),I=16,20)
      READ(15,56)(DEL(I),I=21,25),(del(i),i=31,34)
      READ(15,20)(KEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      READ(15,60) KSPA,NSPA,KSPB,NSPB
      READ(15,705)IFVC1,IFVC2,NLC,KO,KDISK,ISYM,nppl
      read(15,911)nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      read(15,649) jdphs,hjd0,period,dpdt,pshift
      if(jdphs.eq.2.and.kep(26).eq.0) write(16,840)
      if(jdphs.eq.2.and.kep(27).eq.0) write(16,840)
      if(jdphs.eq.2.and.kep(28).eq.0) write(16,840)
      if(kep(14).eq.0.and.kep(26).eq.0) write(16,839)
      READ(15,701)MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,perr0,dperdt,THE,
     $vunit
      READ(15,702)E,A,F1,F2,VGA,XINCL,GR1,GR2,abunin
      READ(15,706) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,xbol1,xbol2,ybol1,
     $ybol2
      acm=6.960d10*a
      nn1=n1
      CALL SINCOS(1,N1,N1,SNTHH,CSTHH,SNFIH,CSFIH,MMSAVH)
      CALL SINCOS(2,N2,N1,SNTHH,CSTHH,SNFIH,CSFIH,MMSAVH)
      CALL SINCOS(1,N1L,N1L,SNTHL,CSTHL,SNFIL,CSFIL,MMSAVL)
      CALL SINCOS(2,N2L,N1L,SNTHL,CSTHL,SNFIL,CSFIL,MMSAVL)
      dint1=pi*(1.d0-xbol1/3.d0)
      if(ld.eq.2) dint1=dint1+pi*ybol1*2.d0/9.d0
      if(ld.eq.3) dint1=dint1-pi*ybol1*.2d0
      dint2=pi*(1.d0-xbol2/3.d0)
      if(ld.eq.2) dint2=dint2+pi*ybol2*2.d0/9.d0
      if(ld.eq.3) dint2=dint2-pi*ybol2*.2d0
      IS=ISYM+1
      KEEP(36)=0
      MM1=N1+1
      MM2=N1+N2+2
      MM3=N1L+1
      MM4=N1L+N2L+2
      M1H=MMSAVH(MM1)
      M2H=MMSAVH(MM2)
      M1L=MMSAVL(MM3)
      M2L=MMSAVL(MM4)
      MTLH=M1H+M2H
      MTLL=M1L+M2L
      NVC=IFVC1+IFVC2
      NLVC=NLC+NVC
      NVCP=NVC+1
      IF(NVC.NE.0) GOTO 288
      KEP(9)=1
      KEP(15)=1
  288 CONTINUE
      DO 84 I=1,ichno
      KEEP(I)=KEP(I)
   84 LOW(I)=1
      LOW(1)=0
      LOW(2)=0
      LOW(3)=0
      LOW(5)=0
      LOW(6)=0
      LOW(7)=0
      LOW(10)=0
      LOW(11)=0
      LOW(12)=0
      LOW(13)=0
      LOW(14)=0
      LOW(16)=0
      LOW(23)=0
      LOW(24)=0
      LOW(25)=0
      ifap=1-keep(29)
      ifphi=1-keep(26)*keep(27)*keep(28)
      KOSQ=(KO-2)*(KO-2)
      IF(NVC.EQ.0) GOTO 195
      DO 90 I=1,NVC
      EL3A(I)=0.d0
   90 READ(15,218) iband(i),HLA(I),CLA(I),X1A(I),X2A(I),y1a(i),y2a(i),
     $opsfa(i),sigma(i),wla(i)
  195 CONTINUE
      IF(NLVC.EQ.NVC) GOTO 194
      DO 190 I=NVCP,NLVC
  190 read(15,18)iband(i),hla(i),cla(i),x1a(i),x2a(i),y1a(i),y2a(i),
     $el3a(i),opsfa(i),noise(i),sigma(i),wla(i)
  194 CONTINUE
      NSP1=0
      NSP2=0
      DO 988 KP=1,2
      DO 987 I=1,ispmax
      READ(15,985)XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
      xlng(kp,i)=xlong(kp,i)
      IF(XLAT(KP,I).GE.200.d0) GOTO 988
      SNLAT(KP,I)=dsin(XLAT(KP,I))
      CSLAT(KP,I)=dcos(XLAT(KP,I))
      SNLNG(KP,I)=dsin(XLONG(KP,I))
      CSLNG(KP,I)=dcos(XLONG(KP,I))
      RDSP(KP,I)=RADSP(KP,I)
      TMSP(KP,I)=TEMSP(KP,I)
      IF(KP.EQ.1) NSP1=NSP1+1
  987 IF(KP.EQ.2) NSP2=NSP2+1
  988 CONTINUE
      NSTOT=NSP1+NSP2
      ncl=0
      do 1062 i=1,iclmax
      read(15,1063) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i)
      if(xcl(i).gt.100.d0) goto 1066
      ncl=ncl+1
      dens(i)=edens(i)*xmue(i)/en0
 1062 continue
 1066 continue
      do 153 ipara=1,8
  153 para(ipara)=0.d0
      if(nspa.eq.0) goto 154
      para(1)=xlat(kspa,nspa)
      para(2)=xlong(kspa,nspa)
      para(3)=radsp(kspa,nspa)
      para(4)=temsp(kspa,nspa)
  154 continue
      if(nspb.eq.0) goto 155
      para(5)=xlat(kspb,nspb)
      para(6)=xlong(kspb,nspb)
      para(7)=radsp(kspb,nspb)
      para(8)=temsp(kspb,nspb)
  155 continue
      para(9)=a
      para(10)=e
      para(11)=perr0
      para(12)=f1
      para(13)=f2
      para(14)=pshift
      para(15)=vga
      para(16)=xincl
      para(17)=gr1
      para(18)=gr2
      para(19)=tavh
      para(20)=tavc
      para(21)=alb1
      para(22)=alb2
      para(23)=phsv
      para(24)=pcsv
      para(25)=rm
      para(26)=hjd0
      para(27)=period
      para(28)=dpdt
      para(29)=dperdt
      para(30)=0.d0
      ib=nvc
      do 191 irx=31,30+nlc
      ib=ib+1
  191 para(irx)=hla(ib)
      ib=nvc
      do 186 irx=31+nlc,30+2*nlc
      ib=ib+1
  186 para(irx)=cla(ib)
      ib=nvc
      do 187 irx=31+2*nlc,30+3*nlc
      ib=ib+1
  187 para(irx)=x1a(ib)
      ib=nvc
      do 188 irx=31+3*nlc,30+4*nlc
      ib=ib+1
  188 para(irx)=x2a(ib)
      ib=nvc
      do 189 irx=31+4*nlc,30+5*nlc
      ib=ib+1
  189 para(irx)=el3a(ib)
      PERT=perr0
      EC=E
      hjd=hjd0
      PSH=PSHIFT
      IRTE=0
      IRVOL1=0
      IRVOL2=0
c***************************************************************
c  The following lines take care of abundances that may not be among
c  the 19 Kurucz values (see abun array). abunin is reset at the
c  allowed value nearest the input value.
      call binnum(abun,imetpts,abunin,iab)
      dif1=abunin-abun(iab)
      if(iab.eq.imetpts) goto 7702
      dif2=abun(iab+1)-abun(iab)
      dif=dif1/dif2
      if((dif.ge.0.d0).and.(dif.le.0.5d0)) goto 7702
      iab=iab+1
 7702 continue
      if(dif1.ne.0.d0) write(16,287) abunin,abun(iab)
      abunin=abun(iab)
      istart=1+(iab-1)*iatmchunk
c***************************************************************
      CALL MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $RM,PHSV,PCSV,GR1,GR2,ALB1,ALB2,N1,N2,F1,F2,MOD,XINCL,THE,MODE,
     $SNTHH,CSTHH,SNFIH,CSFIH,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1,
     $GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
      call ellone(f1,dp,rm,xld,omcr(1),xldd,omd)
      rmr=1.d0/rm
      call ellone(f2,dp,rmr,xld,omcr(2),xldd,omd)
      omcr(2)=rm*omcr(2)+(1.d0-rm)*.5d0
      po(1)=phsv
      po(2)=pcsv
      CALL VOLUME(VOL1,RM,PHSV,DP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR1,1)
      CALL VOLUME(VOL2,RM,PCSV,DP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR2,1)
      IF(E.EQ.0.d0) GOTO 134
      DAP=1.d0+E
      P1AP=PHSV-2.d0*E*RM/(1.d0-E*E)
      VL1=VOL1
      CALL VOLUME(VL1,RM,P1AP,DAP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR1,2)
      DPDX1=(PHSV-P1AP)*(1.d0-E*E)*.5d0/E
      P2AP=PCSV-2.d0*E/(1.d0-E*E)
      VL2=VOL2
      CALL VOLUME(VL2,RM,P2AP,DAP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,SNTHH,CSTHH,SNFIH,CSFIH,SUMMD
     $,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2
     $,GMAG1,GMAG2,glog1,glog2,GR2,2)
      DPDX2=(PCSV-P2AP)*(1.d0-E*E)*.5d0/E
  134 CONTINUE
      PHP=PHPER
      POTH=PHSV
      POTC=PCSV
      POT1=PHSV
      POT2=PCSV
      DO 24 I=1,NLVC
      opsf=opsfa(i)
   24 CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POTH,POTC,N1,N2,F1,F2,d,hla(i),
     $cla(i),x1a(i),x2a(i),y1a(i),y2a(i),gr1,gr2,wla(i),sm1,sm2,tpolh,
     $tpolc,sbrh,sbrc,tavh,tavc,alb1,alb2,xbol1,xbol2,
     $ybol1,ybol2,php,rm,xincl,hot,cool,snthh,csthh,snfih,csfih,tldh,
     $glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,dint1,dint2,grv1,grv2,
     $rftemp,rf1,rf2,csbt1,csbt2,gmag1,gmag2,glog1,glog2,fbin1,fbin2,
     $delv1,delv2,
     $count1,count2,delwl1,delwl2,resf1,resf2,wl1,wl2,dvks1,dvks2,
     $tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,rcl,op1,fcl,dens,
     $encl,edens,taug,emmg,yskp,zskp,mode,iband(i),ifat1,ifat2,1)
      DEL(9)=0.d0
      DEL(15)=0.d0
      del(26)=0.d0
      del(27)=0.d0
      del(28)=0.d0
      del(29)=0.d0
      DEL(35)=0.d0
      WRITE(16,101)
      WRITE(16,399)
      WRITE(16,55)(DEL(I),I=1,8)
      WRITE(16,101)
      WRITE(16,402)
      WRITE(16,55)(DEL(I),I=10,14),(DEL(I),I=16,20)
      WRITE(16,101)
      WRITE(16,403)
      WRITE(16,55)(DEL(I),I=21,25),(del(i),i=31,34)
      WRITE(16,101)
      WRITE(16,406)
      WRITE(16,20)(KEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      WRITE(16,101)
      WRITE(16,166)
      WRITE(16,66)
      WRITE(16,61) KSPA,NSPA,KSPB,NSPB
      WRITE(16,101)
      WRITE(16,707)
      WRITE(16,708)IFVC1,IFVC2,NLC,KO,KDISK,ISYM,nppl
      WRITE(16,101)
      write(16,917)
      write(16,912)nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld
      WRITE(16,101)
      write(16,171)
      write(16,170) jdphs,hjd0,period,dpdt,pshift,delph,nph
      WRITE(16,101)
      WRITE(16,12)
      WRITE(16,1)MODE,IPB,IFAT1,IFAT2,N1,N2,N1L,N2L,perr0,dperdt,THE,
     $vunit,vfac
      WRITE(16,101)
      WRITE(16,205)
      WRITE(16,206) E,A,F1,F2,VGA,XINCL,GR1,GR2,nsp1,nsp2,abunin,iab
      WRITE(16,101)
      WRITE(16,54)
      WRITE(16,408) TAVH,TAVC,ALB1,ALB2,PHSV,PCSV,RM,xbol1,xbol2,ybol1,
     $ybol2
      IF(NVC.EQ.0) GOTO 196
      WRITE(16,101)
      WRITE(16,111)
      DO 91 I=1,NVC
   91 WRITE(16,218)iband(I),HLA(I),CLA(I),X1A(I),X2A(I),y1a(i),y2a(i),
     $opsfa(i),sigma(i),wla(i)
  196 CONTINUE
      IF(NLVC.EQ.NVC) GOTO 197
      WRITE(16,101)
      WRITE(16,11)
      DO 92 I=NVCP,NLVC
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
c     The following block supports third light to be printed from the
c     passed percentage of third luminosity.
c
      if (L3perc.eq.1) then
        el3=(hla(i)+cla(i))*el3a(i)/(4.d0*3.141593d0*
     $      (1.d0-el3a(i)))
      else
        el3=el3a(i)
      end if
   92 write(16,85)iband(i),hla(i),cla(i),x1a(i),x2a(i),y1a(i),y2a(i),
     $el3,opsfa(i),noise(i),sigma(i),wla(i)
c  92 write(16,85)iband(i),hla(i),cla(i),x1a(i),x2a(i),y1a(i),y2a(i),
c    $el3a(i),opsfa(i),noise(i),sigma(i),wla(i)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  197 CONTINUE
      WRITE(16,101)
      IF(NSTOT.GT.0) WRITE(16,983)
      DO 688 KP=1,2
      IF((NSP1+KP-1).EQ.0) GOTO 688
      IF((NSP2+(KP-2)**2).EQ.0) GOTO 688
      NSPOT=NSP1
      IF(KP.EQ.2) NSPOT=NSP2
      DO 687 I=1,NSPOT
  687 WRITE(16,684)KP,XLAT(KP,I),XLONG(KP,I),RADSP(KP,I),TEMSP(KP,I)
  688 WRITE(16,101)
      if(ncl.eq.0) goto 1067
      write(16,69)
      do 68 i=1,ncl
   68 write(16,64) xcl(i),ycl(i),zcl(i),rcl(i),op1(i),fcl(i),edens(i),
     $xmue(i),encl(i),dens(i)
      write(16,101)
 1067 continue
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE addition:
c
      WRITE(16,*) "            *** IMPORTANT NOTICE ***"
      WRITE(16,*) ""
      WRITE(16,*) "THE FOLLOWING BLOCK OF OBSERVATIONS WAS *NOT* USED"
      WRITE(16,*) "IN DC, IT HAS BEEN READ FROM THE DCI FILE AND COPIED"
      WRITE(16,*) "HERE. PHOEBE PASSES DATA ARRAYS TO DC DIRECTLY, NOT"
      WRITE(16,*) "THROUGH A DCI FILE. IF YOUR WEIGHTS SHOW -1.0, THAT"
      WRITE(16,*) "MEANS THAT FORMATTING RESTRICTIONS OF WD WOULD HAVE"
      WRITE(16,*) "PREVENTED THEIR PROPER OUTPUT. HOWEVER, SINCE THE"
      WRITE(16,*) "DATA ARE PASSED TO DC DIRECTLY, THESE VALUES DO NOT"
      WRITE(16,*) "PLAY ANY ROLE."
      WRITE(16,*) ""
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      WRITE(16,101)
      WRITE(16,9)
      DO 75 LCV=1,NLVC
      WRITE(16,101)
      IF(LCV.LE.NVC.and.jdphs.eq.2) WRITE(16,955)
      IF(LCV.GT.NVC.and.jdphs.eq.2) WRITE(16,10)
      IF(LCV.LE.NVC.and.jdphs.eq.1) WRITE(16,755)
      IF(LCV.GT.NVC.and.jdphs.eq.1) WRITE(16,756)
      DO 74 I=NS,iptmax
      ifirst=nppl*(i-1)+NY+1
      last=ifirst+nppl-1
      READ(15,2) (phjd(in),flux(in),wt(in),in=ifirst,last)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     WRITE(16,2) (phjd(in),flux(in),wt(in),in=ifirst,last)
      IF(phjd(ifirst).gt.-10000.d0) then
        WRITE(16,2) (indeps(in),fluxes(in),weights(in),in=ifirst,last)
        GOTO 74
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      NI=-(phjd(ifirst)+10000.d0)
      NY=NY+NI
      NOBS=nppl*(I-NS-1)+NI
      GOTO 150
   74 CONTINUE
  150 NS=I-1
      LC1=LCV+1
c  75 KNOBS(LC1)=NOBS+KNOBS(LCV)
   75 continue
      do 275 ijp=1,knobs(lc1)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c     phas(ijp)=phjd(ijp)
c     if(jdphs.eq.1) call jdph(phjd(ijp),0.d0,hjd0,period,dpdt,xjddum,
c    $phas(ijp))
c
      phas(ijp)=indeps(ijp)
      if(jdphs.eq.1) call jdph(indeps(ijp),0.d0,hjd0,period,dpdt,xjddum,
     $phas(ijp))
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  275 continue
      matrix=31
      do 427 ima=1,30
  427 matrix=matrix-keep(ima)
      matrix=matrix+nlc*(5-keep(31)-keep(32)-keep(33)-keep(34)-keep(35))
      MAT=MATRIX-1
      EM=dfloat(MATRIX-15)
      KTR=.24d0*EM+2.2d0
      IF(EM.LE.1.5d0) KTR=1
      IF(EM.GT.12.d0) KTR=5
      NCOEFF=MATRIX*KNOBS(LC1)
      NMAT=MAT*KNOBS(LC1)
      NCOF=NCOEFF
      DO 63 J=1,NCOF
   63 HOLD(J)=0.d0
      IF(KOSQ.EQ.1) GOTO 71
      DO 416 IOBS=1,NCOEFF
  416 OBS(IOBS)=0.d0
      KSMAX=37
      KSSMAX=37
      IF(E.EQ.0.d0) KSMAX=1
      IF(E.NE.0.d0) KSSMAX=1
      DO 419 KSS=1,KSSMAX
      DO 417 IB=1,NLVC
      VC1=0.d0
      VC2=0.d0
      ELIT=0.d0
      IF(IB.GT.NVC) ELIT=1.d0
      IF(IB.EQ.IFVC1) VC1=1.d0
      IF(IB.EQ.(IFVC2*(1+IFVC1))) VC2=1.d0
      IST=KNOBS(IB)+1
      IB1=IB+1
      ISP=KNOBS(IB1)
      DO 418 IX=IST,ISP
      DO 420 KS=1,KSMAX
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c     hjd=phjd(ix)
c
      hjd=indeps(ix)
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dtime=hjd-hjd0
      IRTE=0
      IRVOL1=0
      IRVOL2=0
      IF(E.NE.0.d0) GOTO 297
      IF(IX.NE.IST) IRTE=1
      IF(IX.EQ.ist) GOTO 297
      IRVOL1=1
      IRVOL2=1
  297 CONTINUE
      KSR=KS
      IF(E.EQ.0.d0) KSR=KSS
      if(e.ne.0.d0) goto 1110
      IF(ISYM.NE.1) GOTO 1110
      IF(KSR.EQ.1) GOTO 420
 1110 CONTINUE
      KH=KSR-2
      IF(KH.GT.0) GOTO 740
      KH=1
      GOTO 941
  740 CONTINUE
      if(ifap*kh.eq.11) goto 842
      if(ifphi*kh.eq.14) goto 842
      IF(KEEP(KH).EQ.1) GOTO 420
  842 continue
c 842 IF(E.EQ.0.d0) GOTO 889
c     IF(KSR.LE.2) GOTO 889
c     IF(KH.LE.9) IRTE=1
c     IF(KH.LE.9) IRVOL1=1
c     IF(KH.LE.9) IRVOL2=1
c     IF(KH.EQ.12) IRVOL2=1
c     IF(KH.EQ.13) IRVOL1=1
c     IF(KH.EQ.15) IRTE=1
c     IF(KH.EQ.15) IRVOL1=1
c     IF(KH.EQ.15) IRVOL2=1
c     IF(KH.EQ.16) IRTE=1
c     IF(KH.EQ.16) IRVOL1=1
c     IF(KH.EQ.16) IRVOL2=1
c     IF(KH.EQ.17) IRVOL2=1
c     IF(KH.EQ.18) IRVOL1=1
c     IF(KH.EQ.19) IRVOL1=1
c     IF(KH.EQ.19) IRVOL2=1
c     IF(KH.EQ.20) IRVOL1=1
c     IF(KH.EQ.20) IRVOL2=1
c     IF(KH.EQ.21) IRVOL1=1
c     IF(KH.EQ.21) IRVOL2=1
c     IF(KH.EQ.22) IRVOL1=1
c     IF(KH.EQ.22) IRVOL2=1
c     IF(KH.EQ.23) IRVOL2=1
c     IF(KH.EQ.24) IRVOL1=1
c     IF(KH.GE.31) IRVOL1=1
c     IF(KH.GE.31) IRVOL2=1
c 889 CONTINUE
      LCF=0
      IF(KH.GT.30) LCF=IB-NVC
      KPCT1=0
      KPCT2=0
      KSP=KH
      IF(KH.GT.30) KSP=30
      IF(KH.LT.2) GOTO 808
      DO 804 ICT=1,KSP
  804 KPCT1=KPCT1+1-KEEP(ICT)
      GOTO 809
  808 KPCT1=1
  809 CONTINUE
      IF(KH.LT.31) GOTO 806
      DO 805 ICT=31,KH
  805 KPCT2=KPCT2+1-KEEP(ICT)
      GOTO 807
  806 KPCT2=1
  807 CONTINUE
      II=(KPCT1+NLC*(KPCT2-1)+LCF-1)*KNOBS(LC1)+IX
      IF(KH.EQ.9) GOTO 300
      IF(KH.EQ.15) GOTO 308
      if(kh.eq.26) goto 844
      if(kh.eq.27) goto 845
      if(kh.eq.28) goto 846
      if(kh.eq.29) goto 847
      IF(KH.EQ.35) GOTO 301
      IF(KH.NE.31) GOTO 941
      IF(MODE.LE.0) GOTO 941
      IF(IPB.EQ.1) GOTO 941
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
c     The following block supports third light to be computed from the
c     passed percentage of third luminosity.
c
c     IF(IB.GT.NVC) OBS(II)=(BR(IX)-EL3A(IB))/HLA(IB)
c
      if (IB.GT.NVC) then
        if (L3perc.eq.1) then
          el3=(hla(ib)+cla(ib))*el3a(ib)/(4.d0*3.141593d0*
     $        (1.d0-el3a(ib)))
        else
          el3=el3a(ib)
        end if
        OBS(II)=(BR(IX)-EL3)/HLA(IB)
      end if
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      GOTO 420
  941 CONTINUE
      DL=DEL(KH)
      IF(ISYM.EQ.1) DL=.5d0*DEL(KH)
      SIGN=1.d0
      ISS=1
      DO 421 IH=1,ichno
  421 DLIF(IH)=0.d0
      IF(KSR.LE.2) GOTO 777
      ISS=IS
      DLIF(KH)=1.d0
  777 CONTINUE
      DO 319 IL=1,ISS
      IF(E.NE.0.d0) GOTO 4011
      IF(ISYM.EQ.1.and.ix.ne.ist) GOTO 4012
      GOTO 940
 4011 IF(KSR.LE.2) GOTO 940
      goto 4014
 4012 IF(IL.EQ.2) GOTO 4016
 4014 IF(LOW(KH).EQ.1) GOTO 314
      VOL1=SVOL1
      VOL2=SVOL2
      SUMM1=SSUM1
      SUMM2=SSUM2
      SM1=SSM1
      SM2=SSM2
      DO 851 IRE=1,MTLH
  851 TLDH(IRE)=STLDH(IRE)
      DO 508 IRE=1,M1H
      RV(IRE)=SRV(IRE)
      GRX(IRE)=SGRX(IRE)
      GRY(IRE)=SGRY(IRE)
      GRZ(IRE)=SGRZ(IRE)
      GLUMP1(IRE)=SGLM1(IRE)
      GRV1(IRE)=SGRV1(IRE)
      XX1(IRE)=SXX1(IRE)
      YY1(IRE)=SYY1(IRE)
      ZZ1(IRE)=SZZ1(IRE)
      GMAG1(IRE)=SGMG1(IRE)
      CSBT1(IRE)=SCSB1(IRE)
      RF1(IRE)=SRF1(IRE)
      FR1(IRE)=SFR1(IRE)
  508 SLUMP1(IRE)=SLMP1(IRE)
      DO 309 IRE=1,M2H
      RVQ(IRE)=SRVQ(IRE)
      GRXQ(IRE)=SGRXQ(IRE)
      GRYQ(IRE)=SGRYQ(IRE)
      GRZQ(IRE)=SGRZQ(IRE)
      GLUMP2(IRE)=SGLM2(IRE)
      GRV2(IRE)=SGRV2(IRE)
      XX2(IRE)=SXX2(IRE)
      YY2(IRE)=SYY2(IRE)
      ZZ2(IRE)=SZZ2(IRE)
      GMAG2(IRE)=SGMG2(IRE)
      CSBT2(IRE)=SCSB2(IRE)
      RF2(IRE)=SRF2(IRE)
      FR2(IRE)=SFR2(IRE)
  309 SLUMP2(IRE)=SLMP2(IRE)
      GOTO 940
 4016 IF(LOW(KH).EQ.1) GOTO 4018
      VOL1=EVOL1
      VOL2=EVOL2
      SUMM1=ESUM1
      SUMM2=ESUM2
      SM1=ESM1
      SM2=ESM2
      DO 852 IRE=1,MTLH
  852 TLDH(IRE)=ETLDH(IRE)
      DO 1508 IRE=1,M1H
      RV(IRE)=ERV(IRE)
      GRX(IRE)=EGRX(IRE)
      GRY(IRE)=EGRY(IRE)
      GRZ(IRE)=EGRZ(IRE)
      GLUMP1(IRE)=EGLM1(IRE)
      GRV1(IRE)=EGRV1(IRE)
      XX1(IRE)=EXX1(IRE)
      YY1(IRE)=EYY1(IRE)
      ZZ1(IRE)=EZZ1(IRE)
      GMAG1(IRE)=EGMG1(IRE)
      CSBT1(IRE)=ECSB1(IRE)
      RF1(IRE)=ERF1(IRE)
      FR1(IRE)=EFR1(IRE)
 1508 SLUMP1(IRE)=ELMP1(IRE)
      DO 1309 IRE=1,M2H
      RVQ(IRE)=ERVQ(IRE)
      GRXQ(IRE)=EGRXQ(IRE)
      GRYQ(IRE)=EGRYQ(IRE)
      GRZQ(IRE)=EGRZQ(IRE)
      GLUMP2(IRE)=EGLM2(IRE)
      GRV2(IRE)=EGRV2(IRE)
      XX2(IRE)=EXX2(IRE)
      YY2(IRE)=EYY2(IRE)
      ZZ2(IRE)=EZZ2(IRE)
      GMAG2(IRE)=EGMG2(IRE)
      CSBT2(IRE)=ECSB2(IRE)
      RF2(IRE)=ERF2(IRE)
      FR2(IRE)=EFR2(IRE)
 1309 SLUMP2(IRE)=ELMP2(IRE)
      GOTO 940
 4018 CONTINUE
      VOL1=EVOL1L
      VOL2=EVOL2L
      SUMM1=ESUM1L
      SUMM2=ESUM2L
      SM1=ESM1L
      SM2=ESM2L
      DO 853 IRE=1,MTLL
  853 TLDL(IRE)=ETLDL(IRE)
      DO 310 IRE=1,M1L
      RV(IRE)=ERVL(IRE)
      GRX(IRE)=EGRXL(IRE)
      GRY(IRE)=EGRYL(IRE)
      GRZ(IRE)=EGRZL(IRE)
      GLUMP1(IRE)=EGLM1L(IRE)
      GRV1(IRE)=EGRV1L(IRE)
      XX1(IRE)=EXX1L(IRE)
      YY1(IRE)=EYY1L(IRE)
      ZZ1(IRE)=EZZ1L(IRE)
      GMAG1(IRE)=EGMG1L(IRE)
      CSBT1(IRE)=ECSB1L(IRE)
      RF1(IRE)=ERF1L(IRE)
      FR1(IRE)=EFR1L(IRE)
  310 SLUMP1(IRE)=ELMP1L(IRE)
      DO 311 IRE=1,M2L
      RVQ(IRE)=ERVQL(IRE)
      GRXQ(IRE)=EGRXQL(IRE)
      GRYQ(IRE)=EGRYQL(IRE)
      GRZQ(IRE)=EGRZQL(IRE)
      GLUMP2(IRE)=EGLM2L(IRE)
      GRV2(IRE)=EGRV2L(IRE)
      XX2(IRE)=EXX2L(IRE)
      YY2(IRE)=EYY2L(IRE)
      ZZ2(IRE)=EZZ2L(IRE)
      GMAG2(IRE)=EGMG2L(IRE)
      CSBT2(IRE)=ECSB2L(IRE)
      RF2(IRE)=ERF2L(IRE)
      FR2(IRE)=EFR2L(IRE)
  311 SLUMP2(IRE)=ELMP2L(IRE)
      GOTO 940
  314 CONTINUE
      VOL1=SVOL1L
      VOL2=SVOL2L
      SUMM1=SSUM1L
      SUMM2=SSUM2L
      SM1=SSM1L
      SM2=SSM2L
      DO 854 IRE=1,MTLL
  854 TLDL(IRE)=STLDL(IRE)
      DO 1310 IRE=1,M1L
      RV(IRE)=SRVL(IRE)
      GRX(IRE)=SGRXL(IRE)
      GRY(IRE)=SGRYL(IRE)
      GRZ(IRE)=SGRZL(IRE)
      GLUMP1(IRE)=SGLM1L(IRE)
      GRV1(IRE)=SGRV1L(IRE)
      XX1(IRE)=SXX1L(IRE)
      YY1(IRE)=SYY1L(IRE)
      ZZ1(IRE)=SZZ1L(IRE)
      GMAG1(IRE)=SGMG1L(IRE)
      CSBT1(IRE)=SCSB1L(IRE)
      RF1(IRE)=SRF1L(IRE)
      FR1(IRE)=SFR1L(IRE)
 1310 SLUMP1(IRE)=SLMP1L(IRE)
      DO 1311 IRE=1,M2L
      RVQ(IRE)=SRVQL(IRE)
      GRXQ(IRE)=SGRXQL(IRE)
      GRYQ(IRE)=SGRYQL(IRE)
      GRZQ(IRE)=SGRZQL(IRE)
      GLUMP2(IRE)=SGLM2L(IRE)
      GRV2(IRE)=SGRV2L(IRE)
      XX2(IRE)=SXX2L(IRE)
      YY2(IRE)=SYY2L(IRE)
      ZZ2(IRE)=SZZ2L(IRE)
      GMAG2(IRE)=SGMG2L(IRE)
      CSBT2(IRE)=SCSB2L(IRE)
      RF2(IRE)=SRF2L(IRE)
      FR2(IRE)=SFR2L(IRE)
 1311 SLUMP2(IRE)=SLMP2L(IRE)
  940 CONTINUE
      DELS=DL*SIGN
      SIGN=-1.d0
      IF(NSPA.EQ.0) GOTO 470
      xlt=xlat(kspa,nspa)+dels*dlif(1)
      xlng(kspa,nspa)=xlong(kspa,nspa)+dels*dlif(2)
      snlat(kspa,nspa)=dsin(xlt)
      cslat(kspa,nspa)=dcos(xlt)
      snlng(kspa,nspa)=dsin(xlng(kspa,nspa))
      cslng(kspa,nspa)=dcos(xlng(kspa,nspa))
      RDSP(KSPA,NSPA)=RADSP(KSPA,NSPA)+DELS*DLIF(3)
      TMSP(KSPA,NSPA)=TEMSP(KSPA,NSPA)+DELS*DLIF(4)
  470 CONTINUE
      IF(NSPB.EQ.0) GOTO 471
      xlt=xlat(kspb,nspb)+dels*dlif(5)
      xlng(kspb,nspb)=xlong(kspb,nspb)+dels*dlif(6)
      snlat(kspb,nspb)=dsin(xlt)
      cslat(kspb,nspb)=dcos(xlt)
      snlng(kspb,nspb)=dsin(xlng(kspb,nspb))
      cslng(kspb,nspb)=dcos(xlng(kspb,nspb))
      RDSP(KSPB,NSPB)=RADSP(KSPB,NSPB)+DELS*DLIF(7)
      TMSP(KSPB,NSPB)=TEMSP(KSPB,NSPB)+DELS*DLIF(8)
  471 CONTINUE
      EC=E+DELS*DLIF(10)
      PERT=perr0+DELS*DLIF(11)
      FF1=F1+DELS*DLIF(12)
      FF2=F2+DELS*DLIF(13)
      PSH=PSHIFT+DELS*DLIF(14)
      XINC=XINCL+DELS*DLIF(16)
      G1=GR1+DELS*DLIF(17)
      G2=GR2+DELS*DLIF(18)
      T1=TAVH+DELS*DLIF(19)
      T2=TAVC+DELS*DLIF(20)
      A1=ALB1+DELS*DLIF(21)
      A2=ALB2+DELS*DLIF(22)
      POT1=PHSV+DELS*DLIF(23)
      POT2=PCSV+DELS*DLIF(24)
      RMASS=RM+DELS*DLIF(25)
      HL=HLA(IB)+DELS*DLIF(31)
      CL=CLA(IB)+DELS*DLIF(32)
      X1=X1A(IB)+DELS*DLIF(33)
      X2=X2A(IB)+DELS*DLIF(34)
      y1=y1a(ib)
      y2=y2a(ib)
      opsf=opsfa(ib)
      IF(KSR.EQ.1) GOTO 802
      IF(KSR.EQ.2) GOTO 872
      IF(LOW(KH).EQ.1) GOTO 802
  872 CALL MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $RMASS,POT1,POT2,G1,G2,A1,A2,N1,N2,FF1,FF2,MOD,XINC,THE,MODE,
     $SNTHH,CSTHH,SNFIH,CSFIH,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1,
     $GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
      hot=0.d0
      cool=0.d0
      do 551 iph=1,nph
      phasin=phas(ix)
      if(nph.gt.1.and.ib.gt.nvc) phasin=phas(ix)+delph*(dfloat(iph-1)/
     $dfloat(nph-1)-.5d0)
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVH,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POT1,POT2,N1,N2,FF1,FF2,d,hl,cl,x1,
     $x2,y1,y2,g1,g2,wla(ib),sm1,sm2,tph,tpc,sbrh,sbrc,t1,
     $t2,a1,a2,xbol1,xbol2,ybol1,ybol2,phasin,rmass,xinc,hotr,coolr,
     $snthh,csthh,snfih,csfih,tldh,glump1,glump2,xx1,xx2,yy1,yy2,zz1,
     $zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,gmag2,
     $glog1,glog2,
     $fbin1,fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,resf2,
     $wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,
     $rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,zskp,mode,iband(ib),
     $ifat1,ifat2,1)
      hot=hot+hotr/dfloat(nph)
      cool=cool+coolr/dfloat(nph)
  551 continue
      GOTO 801
  802 CONTINUE
      CALL MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVL,FR1,FR2,HLD,
     $RMASS,POT1,POT2,G1,G2,A1,A2,N1L,N2L,FF1,FF2,MOD,XINC,THE,MODE,
     $SNTHL,CSTHL,SNFIL,CSFIL,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1,
     $GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
      hot=0.d0
      cool=0.d0
      do 550 iph=1,nph
      phasin=phas(ix)
      if(nph.gt.1.and.ib.gt.nvc) phasin=phas(ix)+delph*(dfloat(iph-1)/
     $dfloat(nph-1)-.5d0)
      CALL BBL(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVL,FR1,FR2,HLD,
     $SLUMP1,SLUMP2,THETA,RHO,AA,BB,POT1,POT2,N1L,N2L,FF1,FF2,d,hl,cl,
     $x1,x2,y1,y2,g1,g2,wla(ib),sm1,sm2,tph,tpc,sbrh,sbrc,
     $t1,t2,a1,a2,xbol1,xbol2,ybol1,ybol2,phasin,rmass,xinc,hotr,coolr,
     $snthl,csthl,snfil,csfil,tldl,glump1,glump2,xx1,xx2,yy1,yy2,zz1,
     $zz2,dint1,dint2,grv1,grv2,rftemp,rf1,rf2,csbt1,csbt2,gmag1,gmag2,
     $glog1,glog2,
     $fbin1,fbin2,delv1,delv2,count1,count2,delwl1,delwl2,resf1,resf2,
     $wl1,wl2,dvks1,dvks2,tau1,tau2,emm1,emm2,hbarw1,hbarw2,xcl,ycl,zcl,
     $rcl,op1,fcl,dens,encl,edens,taug,emmg,yskp,zskp,mode,iband(ib),
     $ifat1,ifat2,1)
      hot=hot+hotr/dfloat(nph)
      cool=cool+coolr/dfloat(nph)
  550 continue
  801 CONTINUE
      IF(E.NE.0.d0) GOTO 4111
      IF(ISYM.EQ.0) GOTO 602
      IF(IX.NE.IST) GOTO 602
      IF(KSR.EQ.1) GOTO 602
      IF(KSR.EQ.2) GOTO 4119
      IF(LOW(KH).EQ.1) GOTO 4112
      GOTO 4119
 4111 IF(IRTE.EQ.1) GOTO 602
      IF(KSR.GE.3) GOTO 602
      IF(KSR.EQ.2) GOTO 4119
 4112 IF(IL.EQ.2) GOTO 4116
      SVOL1L=VOL1
      SVOL2L=VOL2
      SSUM1L=SUMM1
      SSUM2L=SUMM2
      SSM1L=SM1
      SSM2L=SM2
      DO 855 IHL=1,MTLL
  855 STLDL(IHL)=TLDL(IHL)
      DO 603 IHL=1,M1L
      SRVL(IHL)=RV(IHL)
      SGRXL(IHL)=GRX(IHL)
      SGRYL(IHL)=GRY(IHL)
      SGRZL(IHL)=GRZ(IHL)
      SLMP1L(IHL)=SLUMP1(IHL)
      SGLM1L(IHL)=GLUMP1(IHL)
      SGRV1L(IHL)=GRV1(IHL)
      SXX1L(IHL)=XX1(IHL)
      SYY1L(IHL)=YY1(IHL)
      SZZ1L(IHL)=ZZ1(IHL)
      SGMG1L(IHL)=GMAG1(IHL)
      SCSB1L(IHL)=CSBT1(IHL)
      SRF1L(IHL)=RF1(IHL)
      SFR1L(IHL)=FR1(IHL)
  603 CONTINUE
      DO 606 IHL=1,M2L
      SRVQL(IHL)=RVQ(IHL)
      SGRXQL(IHL)=GRXQ(IHL)
      SGRYQL(IHL)=GRYQ(IHL)
      SGRZQL(IHL)=GRZQ(IHL)
      SLMP2L(IHL)=SLUMP2(IHL)
      SGLM2L(IHL)=GLUMP2(IHL)
      SGRV2L(IHL)=GRV2(IHL)
      SXX2L(IHL)=XX2(IHL)
      SYY2L(IHL)=YY2(IHL)
      SZZ2L(IHL)=ZZ2(IHL)
      SGMG2L(IHL)=GMAG2(IHL)
      SCSB2L(IHL)=CSBT2(IHL)
      SRF2L(IHL)=RF2(IHL)
      SFR2L(IHL)=FR2(IHL)
  606 CONTINUE
      GOTO 602
 4116 CONTINUE
      EVOL1L=VOL1
      EVOL2L=VOL2
      ESUM1L=SUMM1
      ESUM2L=SUMM2
      ESM1L=SM1
      ESM2L=SM2
      DO 856 IHL=1,MTLL
  856 ETLDL(IHL)=TLDL(IHL)
      DO 1603 IHL=1,M1L
      ERVL(IHL)=RV(IHL)
      EGRXL(IHL)=GRX(IHL)
      EGRYL(IHL)=GRY(IHL)
      EGRZL(IHL)=GRZ(IHL)
      ELMP1L(IHL)=SLUMP1(IHL)
      EGLM1L(IHL)=GLUMP1(IHL)
      EGRV1L(IHL)=GRV1(IHL)
      EXX1L(IHL)=XX1(IHL)
      EYY1L(IHL)=YY1(IHL)
      EZZ1L(IHL)=ZZ1(IHL)
      EGMG1L(IHL)=GMAG1(IHL)
      ECSB1L(IHL)=CSBT1(IHL)
      ERF1L(IHL)=RF1(IHL)
      EFR1L(IHL)=FR1(IHL)
 1603 CONTINUE
      DO 1606 IHL=1,M2L
      ERVQL(IHL)=RVQ(IHL)
      EGRXQL(IHL)=GRXQ(IHL)
      EGRYQL(IHL)=GRYQ(IHL)
      EGRZQL(IHL)=GRZQ(IHL)
      ELMP2L(IHL)=SLUMP2(IHL)
      EGLM2L(IHL)=GLUMP2(IHL)
      EGRV2L(IHL)=GRV2(IHL)
      EXX2L(IHL)=XX2(IHL)
      EYY2L(IHL)=YY2(IHL)
      EZZ2L(IHL)=ZZ2(IHL)
      EGMG2L(IHL)=GMAG2(IHL)
      ECSB2L(IHL)=CSBT2(IHL)
      ERF2L(IHL)=RF2(IHL)
      EFR2L(IHL)=FR2(IHL)
 1606 CONTINUE
      GOTO 602
 4119 IF(IL.EQ.2) GOTO 4120
      SVOL1=VOL1
      SVOL2=VOL2
      SSUM1=SUMM1
      SSUM2=SUMM2
      SSM1=SM1
      SSM2=SM2
      DO 857 IHH=1,MTLH
  857 STLDH(IHH)=TLDH(IHH)
      DO 601 IHH=1,M1H
      SRV(IHH)=RV(IHH)
      SGRX(IHH)=GRX(IHH)
      SGRY(IHH)=GRY(IHH)
      SGRZ(IHH)=GRZ(IHH)
      SLMP1(IHH)=SLUMP1(IHH)
      SGLM1(IHH)=GLUMP1(IHH)
      SGRV1(IHH)=GRV1(IHH)
      SXX1(IHH)=XX1(IHH)
      SYY1(IHH)=YY1(IHH)
      SZZ1(IHH)=ZZ1(IHH)
      SGMG1(IHH)=GMAG1(IHH)
      SCSB1(IHH)=CSBT1(IHH)
      SRF1(IHH)=RF1(IHH)
      SFR1(IHH)=FR1(IHH)
  601 CONTINUE
      DO 605 IHH=1,M2H
      SRVQ(IHH)=RVQ(IHH)
      SGRXQ(IHH)=GRXQ(IHH)
      SGRYQ(IHH)=GRYQ(IHH)
      SGRZQ(IHH)=GRZQ(IHH)
      SLMP2(IHH)=SLUMP2(IHH)
      SGLM2(IHH)=GLUMP2(IHH)
      SGRV2(IHH)=GRV2(IHH)
      SXX2(IHH)=XX2(IHH)
      SYY2(IHH)=YY2(IHH)
      SZZ2(IHH)=ZZ2(IHH)
      SGMG2(IHH)=GMAG2(IHH)
      SCSB2(IHH)=CSBT2(IHH)
      SRF2(IHH)=RF2(IHH)
      SFR2(IHH)=FR2(IHH)
  605 CONTINUE
      GOTO 602
 4120 CONTINUE
      EVOL1=VOL1
      EVOL2=VOL2
      ESUM1=SUMM1
      ESUM2=SUMM2
      ESM1=SM1
      ESM2=SM2
      DO 858 IHH=1,MTLH
  858 ETLDH(IHH)=TLDH(IHH)
      DO 1601 IHH=1,M1H
      ERV(IHH)=RV(IHH)
      EGRX(IHH)=GRX(IHH)
      EGRY(IHH)=GRY(IHH)
      EGRZ(IHH)=GRZ(IHH)
      ELMP1(IHH)=SLUMP1(IHH)
      EGLM1(IHH)=GLUMP1(IHH)
      EGRV1(IHH)=GRV1(IHH)
      EXX1(IHH)=XX1(IHH)
      EYY1(IHH)=YY1(IHH)
      EZZ1(IHH)=ZZ1(IHH)
      EGMG1(IHH)=GMAG1(IHH)
      ECSB1(IHH)=CSBT1(IHH)
      ERF1(IHH)=RF1(IHH)
      EFR1(IHH)=FR1(IHH)
 1601 CONTINUE
      DO 1605 IHH=1,M2H
      ERVQ(IHH)=RVQ(IHH)
      EGRXQ(IHH)=GRXQ(IHH)
      EGRYQ(IHH)=GRYQ(IHH)
      EGRZQ(IHH)=GRZQ(IHH)
      ELMP2(IHH)=SLUMP2(IHH)
      EGLM2(IHH)=GLUMP2(IHH)
      EGRV2(IHH)=GRV2(IHH)
      EXX2(IHH)=XX2(IHH)
      EYY2(IHH)=YY2(IHH)
      EZZ2(IHH)=ZZ2(IHH)
      EGMG2(IHH)=GMAG2(IHH)
      ECSB2(IHH)=CSBT2(IHH)
      ERF2(IHH)=RF2(IHH)
      EFR2(IHH)=FR2(IHH)
 1605 CONTINUE
  602 CONTINUE
      HTT=HOT
      IF(MODE.EQ.-1) HTT=0.d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
c     The following block supports third light to be computed from the
c     passed percentage of third luminosity.
c
      if (L3perc.eq.1) then
        el3=(hla(ib)+cla(ib))*el3a(ib)/(4.d0*3.141593d0*(1.d0-el3a(ib)))
      else
        el3=el3a(ib)
      end if
c
c     XR=(HTT+COOL+EL3A(IB))*ELIT+VKM1*VC1+VKM2*VC2
c
      XR=(HTT+COOL+EL3)*ELIT+VKM1*VC1+VKM2*VC2
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      IF(KSR.NE.1) GOTO 710
      BL(IX)=XR
      GOTO 420
  710 CONTINUE
      IF(KSR.NE.2) GOTO 711
      BR(IX)=XR
      II=NMAT+IX
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c     OBS(II)=FLUX(IX)-XR
c
      OBS(II)=fluxes(ix)-XR
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if(iss.eq.2) goto 319
      GOTO 420
  711 CONTINUE
      XLR=BR(IX)
      IF(LOW(KH).EQ.1) XLR=BL(IX)
      IF(IL.NE.2) GOTO 388
      XLR=XR
      GOTO 87
  388 XNR=XR
   87 khkeep=kh*keep(kh)
      if(khkeep.ne.11.and.khkeep.ne.14) OBS(II)=(XNR-XLR)/DEL(KH)
      if(kh.eq.11) dfdap(ix)=(xnr-xlr)/del(kh)
      if(kh.eq.14) dfdph(ix)=(xnr-xlr)/del(kh)
  319 CONTINUE
      GOTO 420
  300 IF(IB.LE.NVC) OBS(II)=(BR(IX)-VGA)/A
      GOTO 420
  308 IF(IB.LE.NVC) OBS(II)=1.d0
      GOTO 420
  844 obs(ii)=dfdph(ix)/(period+dtime*dpdt)
      goto 420
  845 brac=period+dtime*dpdt
      obs(ii)=dfdph(ix)*dtime/(brac*period)
      goto 420
  846 dis=dabs(dtime*dpdt/period)
      if(dis.gt.toldis) goto 848
      brac2=2.d0*period+dtime*dpdt
      dphpd=-2.d0*(dtime/brac2)**2
     $+tt*dtime**3*(2.d0*brac2*dpdt-3.d0*dpdt**2*dtime)/brac2**4
      goto 849
  848 brac=period+dtime*dpdt
      dphpd=dtime/(brac*dpdt)-(dlog(brac)-dlog(period))/dpdt**2
  849 obs(ii)=-dfdph(ix)*dphpd
      goto 420
  847 obs(ii)=dtime*dfdap(ix)
      goto 420
  301 IF(IB.GT.NVC) OBS(II)=1.d0
  420 CONTINUE
  418 CONTINUE
  417 CONTINUE
  419 CONTINUE
      write(16,101)
      write(16,101)
      write(16,101)
      write(16,159)
      write(16,101)
      write(16,169)
      do 298 icv=1,nlvc
      icvp=icv+1
      nbs=knobs(icvp)-knobs(icv)
      iw=knobs(icv)+1
      jstart=nmat+iw
      jstop=jstart+nbs-1
      resq=0.d0
      iww=iw-1
      do 299 jres=jstart,jstop
      iww=iww+1
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c 299 resq=resq+wt(iww)*obs(jres)**2
c
  299 resq=resq+weights(iww)*obs(jres)**2
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      write(16,119) icv,nbs,resq
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
      chi2s(icv)=resq
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  298 continue
      write(16,101)
      do 909 komp=1,2
      if(message(komp,1).eq.1) write(16,283) komp
      if(message(komp,2).eq.1) write(16,284) komp
      if(message(komp,3).eq.1) write(16,285) komp
      if(message(komp,4).eq.1) write(16,286) komp
  909 continue
      write(16,101)
      GOTO 65
   71 JF=0
      IF(KO.EQ.0) stop
      DO 261 J=1,NCOF
  261 OBS(J)=HOLD(J)
      IF(KDISK.EQ.0) GOTO 72
      REWIND 9
      READ(9,67)(OBS(J),J=1,NCOF)
   72 READ(15,20)(KEEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      if(keep(1).ne.2) goto 866
      close (15)
      close (16)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension: stop changed to return
c
      return
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  866 continue
      DO 232 I=1,ichno
  232 IF(KEP(I).EQ.1) KEEP(I)=1
      NOBS=KNOBS(LC1)
      matrix=31
      do 428 ima=1,30
  428 matrix=matrix-keep(ima)
      matrix=matrix+nlc*(5-keep(31)-keep(32)-keep(33)-keep(34)-keep(35))
      MAT=MATRIX-1
      EM=dfloat(MATRIX-15)
      KTR=.24d0*EM+2.2d0
      IF(EM.LE.1.5d0) KTR=1
      IF(EM.GT.12.d0) KTR=5
      NCOEFF=MATRIX*NOBS
      KC=1
      NSHIFT(1)=0
      DO 59 I=2,36
      IF(I.GT.31) KC=NLC
      KE=0
      J=I-1
      IF(KEEP(J).GT.KEP(J)) KE=1
   59 NSHIFT(I)=NOBS*KE*KC+NSHIFT(J)
      NOBBS=NOBS
      DO 30 I=1,36
      IF(KEEP(I).EQ.1) GOTO 30
      IF(I.GT.30) NOBBS=NOBS*NLC
      IF(I.EQ.36) NOBBS=NOBS
      DO 32 J=1,NOBBS
      JF=JF+1
      KX=JF+NSHIFT(I)
   32 OBS(JF)=OBS(KX)
   30 CONTINUE
   65 continue
      WRITE(16,20)(KEEP(I),I=1,35),IFDER,IFM,IFR,xlamda
      NOBS=KNOBS(LC1)
      WRITE(16,101)
      IF(IFDER.EQ.0) KTR=5
      WRITE(16,82)
      WRITE(16,101)
      DO 96 IB=1,NLVC
      IST=KNOBS(IB)+1
      IB1=IB+1
      ISP=KNOBS(IB1)
      DO 96 I=IST,ISP
      GOTO(5,6,7,8,96),KTR
    5 WRITE(16,15)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 96
    6 WRITE(16,16)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 96
    7 WRITE(16,17)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 96
    8 WRITE(16,19)(OBS(J),J=I,NCOEFF,NOBS)
   96 CONTINUE
      IF(KO.LE.1) GOTO 70
      IF(IBEF.EQ.1) GOTO 70
      DO 62 J=1,NCOEFF
   62 HOLD(J)=OBS(J)
      IF(KDISK.EQ.0) GOTO 73
      REWIND 9
      WRITE(9,67)(OBS(J),J=1,NCOEFF)
   73 CONTINUE
   70 WRITE(16,101)
      DO 97 IB=1,NLVC
      IST=KNOBS(IB)+1
      IB1=IB+1
      ISP=KNOBS(IB1)
      NOIS=NOISE(IB)
      DO 97 I=IST,ISP
      IF(IB.GT.NVC) GOTO 444
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c     ROOTWT=dsqrt(WT(I))/(100.d0*SIGMA(IB))
c
c     This is level-dependent weighting for RV curves:
      ROOTWT=dsqrt(weights(I))/(SIGMA(IB))
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      GOTO 445
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE modification:
c
c 444 ROOTWT=dsqrt(WT(I))/(100.d0*SIGMA(IB)*dsqrt(FLUX(I))**NOIS)
c
c     This is level-dependent weighting for light curves:
  444 ROOTWT=dsqrt(weights(I))/(SIGMA(IB)*dsqrt(fluxes(I))**NOIS)
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  445 CONTINUE
      DO 97 LOB=I,NCOEFF,NOBS
   97 OBS(LOB)=OBS(LOB)*ROOTWT
      IF(IFDER.NE.0) WRITE(16,83)
      IF(IFDER.NE.0) WRITE(16,101)
      DO 98 I=1,NOBS
      GOTO(45,46,47,48,98),KTR
   45 WRITE(16,15)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 98
   46 WRITE(16,16)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 98
   47 WRITE(16,17)(OBS(J),J=I,NCOEFF,NOBS)
      GOTO 98
   48 WRITE(16,19)(OBS(J),J=I,NCOEFF,NOBS)
   98 CONTINUE
      CALL square (OBS,NOBS,MAT,OUT,sd,xlamda,deter,CN,CNN,cnc,clc,
     $s,ccl,ll,mm)
      MSQ=MAT*MAT
      IF(IFM.EQ.0) GOTO 436
      WRITE(16,101)
      WRITE(16,181)
      WRITE(16,101)
      DO 38 JR=1,MAT
   38 WRITE(16,37) (CN(JX),JX=JR,MSQ,MAT),CCL(JR)
      WRITE(16,101)
      WRITE(16,183)
      WRITE(16,101)
  436 CONTINUE
      NO1=23
      NO2=24
      NRM=25
      DO 334 IRM=1,24
      IF(IRM.LE.23) NO1=NO1-KEEP(IRM)
      NO2=NO2-KEEP(IRM)
  334 NRM=NRM-KEEP(IRM)
      CORO1=1-KEEP(23)
      CORO2=1-KEEP(24)
      CORQ=1-KEEP(25)
      DO 34 JM=1,MAT
      DO 33 JQ=1,MAT
      JT=JM+MAT*(JQ-1)
      IJM=(MAT+1)*(JM-1)+1
      IJQ=(MAT+1)*(JQ-1)+1
      V(JQ)=CNN(JT)/DSQRT(CNN(IJM)*CNN(IJQ))
      cormat((JM-1)*MAT+JQ)=V(JQ)
   33 continue
      co1q=0.d0
      co2q=0.d0
      if(jm.eq.nrm.and.no1.gt.0) co1q=v(no1)*corq*coro1
      if(jm.eq.nrm.and.no2.gt.0) co2q=v(no2)*corq*coro2
   34 WRITE(16,37)(V(IM),IM=1,MAT)
      IF(IFM.EQ.0) GOTO 36
      WRITE(16,101)
      WRITE(16,184)
      WRITE(16,101)
      CALL DGMPRD(CN,CNN,CNOUT,MAT,MAT,MAT)
      DO 116 J8=1,MAT
  116 WRITE(16,37)(CNOUT(J7),J7=J8,MSQ,MAT)
      WRITE(16,101)
      WRITE(16,185)
      WRITE(16,101)
      ANSCH=0.D0
      DO 118 J5=1,MAT
      V(J5)=0.D0
      DO 117 J6=1,MAT
      int=mat*(j6-1)
      idi=j6+int
      I9=J5+int
  117 V(J5)=OUT(J6)*CN(I9)*dsqrt(cnc(idi))+V(J5)
      ERR=V(J5)-CCL(J5)
  118 ANSCH=ANSCH+DABS(ERR)
      WRITE(16,137)(V(J4),J4=1,MAT)
      WRITE(16,101)
      WRITE(16,138) ANSCH
   36 CONTINUE
      WRITE(16,101)
      WRITE(16,101)
      write(16,715)
      WRITE(16,101)
      write(16,43)
      iout=0
      do 93 kpar=1,35
      imax=1
      if(kpar.gt.30) imax=nlc
      if(keep(kpar).eq.1) goto 93
      do 94 kurv=1,imax
      kcurv=kurv
      if(kpar.le.30) kcurv=0
      iout=iout+1
      ipar=kpar
      if(kpar.gt.30) ipar=30+kurv+(kpar-31)*nlc
      parout=para(ipar)+out(iout)
      write(16,615) kpar,kcurv,para(ipar),out(iout),parout,sd(iout)
   94 continue
   93 continue
      WRITE(16,101)
      WRITE(16,101)
      write(16,716)
      WRITE(16,101)
      write(16,43)
      iout=0
      do 53 kpar=1,35
      imax=1
      if(kpar.gt.30) imax=nlc
      if(keep(kpar).eq.1) goto 53
      do 52 kurv=1,imax
      kcurv=kurv
      if(kpar.le.30) kcurv=0
      iout=iout+1
      ipar=kpar
      if(kpar.gt.30) ipar=30+kurv+(kpar-31)*nlc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
c     The following block supports third light to be computed from the
c     passed percentage of third luminosity.
c
      hlum=hla(kcurv)
      if (kpar.eq.31) then
        hlum=para(ipar)+out(iout)
      endif
      clum=cla(kcurv)
      if (kpar.eq.32) then
        clum=para(ipar)+out(iout)
      endif
      if (kpar.eq.35 .and. L3perc.eq.1) then
        out(iout) = 4.d0*3.1415926d0*out(iout)/
     $              (hlum+clum+4.d0*3.1415926d0*out(iout))
        sd(iout)  = 4.d0*3.1415926d0*sd(iout)/
     $              (hlum+clum+4.d0*3.1415926d0*sd(iout))
      endif
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      parout=para(ipar)+out(iout)
      write(16,616) kpar,kcurv,para(ipar),out(iout),parout,sd(iout)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
      corrs(iout)=out(iout)
      stdevs(iout)=sd(iout)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
   52 continue
   53 continue
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
      do 99999 icla=1,nlc
        ccla(icla)=cla(nvc+icla)
99999 continue
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      WRITE(16,101)
      RESSQ=0.d0
      JST=MAT*NOBS+1
      DO 199 JRES=JST,NCOEFF
  199 RESSQ=RESSQ+OBS(JRES)**2
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     PHOEBE extension:
c
      cfval=ressq
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      WRITE(16,101)
      WRITE(16,40)
      WRITE(16,21) RESSQ,S,deter
      IBEF=1
      IF(IFR.EQ.0) GOTO 71
      WRITE(16,101)
      WRITE(16,101)
      WRITE(16,650)
      WRITE(16,101)
      WRITE(16,101)
      WRITE(16,653)
      WRITE(16,101)
      DO1=0.0
      if (NO1.ne.0) DO1=sd(NO1)*CORO1
      DO2=0.0
      if (NO2.ne.0) DO2=sd(NO2)*CORO2
      if(mod.eq.1) do2=do1
      if(mod.eq.1) co2q=co1q
      DQ=sd(NRM)*CORQ
      COQ=CO1Q
      F=F1
      DP=1.d0-E
      OME=PHSV
      DOM=DO1
      KOMP=0
  925 CONTINUE
      KOMP=KOMP+1
      DO 926 KD=1,4
      if(kd.ne.2) goto 928
      if(po(komp).ge.omcr(komp)) goto 928
      goto 926
  928 continue
      TH=XTHA(KD)
      FI=XFIA(KD)
      CALL ROMQ(OME,RM,F,DP,E,TH,FI,R,DRDO,DRDQ,DODQ,KOMP,MODE)
      DR=dsqrt(DRDQ**2*DQ**2+DRDO**2*DOM**2+2.d0*COQ*DRDQ*DRDO*DQ*DOM)
      WRITE(16,654)KOMP,ARAD(KD),R,DRDO,DRDQ,DR
  926 CONTINUE
      DO2DQ=DODQ
      IF(KOMP.EQ.1)DO1DQ=DODQ
      COQ=CO2Q
      F=F2
      OME=PCSV
      DOM=DO2
      WRITE(16,101)
      IF(KOMP.EQ.1) GOTO 925
      WRITE(16,101)
      WRITE(16,651)
      IF(KOMP.EQ.2) WRITE(16,652)DO1DQ,DO2DQ,CO1Q,CO2Q,DO1,DO2,DQ
      GOTO 71
      END
