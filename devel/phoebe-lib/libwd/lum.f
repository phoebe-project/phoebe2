      Subroutine lum (xlum,x,y,tpoll,n,n1,komp,sbr,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ,fr,sm,ifat,vol,rm,om,
     $f,d,snth,iband)
c   Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      dimension rv(*),rvq(*),mmsave(*),fr(*),snth(*),glump1(*),
     $glump2(*),glog1(*),glog2(*),grv1(*),grv2(*)
      dimension message(2,4)
      common /atmmessages/ message,kompcom
      common /radi/ R1H,RLH,R1C,RLC
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm,
     $is1dm,is2dm,ic1dm,ic2dm,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      kompcom=komp
      TPOLE=10000.d0*TPOLL
      KR=0
      cmp=dfloat(komp-1)
      cmpp=dfloat(2-komp)
      gplog=cmpp*gplog1+cmp*gplog2
      IF(ifat.eq.0) call planckint(tpole,iband,pollog,polin)
      IF(IFAT.NE.0) call atmx(TPOLE,gplog,iband,pollog,polin)
      EN=dfloat(N)
      DELTH=1.570796326794897d0/EN
      SUM=0.d0
      SUMM=0.d0
      SM=0.d0
      VOL=0.d0
      DO 36 I=1,N
      IPN1=I+N1*(komp-1)
      SINTH=SNTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      XM=dfloat(MM)
      DELFI=3.141592653589793d0/XM
      DFST=DELFI*SINTH
      SUMJ=0.d0
      SUMMJ=0.d0
      SMJ=0.d0
      VOLJ=0.d0
      DO 26 J=1,MM
      IP=(komp-1)*(N1+1)+I
      IX=MMSAVE(IP)+J
      IF(komp.EQ.1) GOTO 39
      IF(RVQ(IX).EQ.-1.d0) GOTO 25
      R=RVQ(IX)
      GOTO 49
   39 IF(RV(IX).EQ.-1.d0) GOTO 25
      R=RV(IX)
   49 grav=cmpp*grv1(ix)+cmp*grv2(ix)
      TLOCAL=TPOLE*dsqrt(dsqrt(GRAV))
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      if(ifat.eq.0) call planckint(tlocal,iband,xinlog,xint)
      IF(IFAT.NE.0) CALL atmx(TLOCAL,glogg,iband,xinlog,xint)
      GRAVM=xint/polin
      di=cmpp*glump1(ix)+cmp*glump2(ix)
      DIF=DI*GRAVM
      DIFF=DI*GRAV
      SMJ=SMJ+DI
      SUMJ=SUMJ+DIF
      SUMMJ=SUMMJ+DIFF
      VOLJ=VOLJ+R*R*R*FR(IX)
      GOTO 26
   25 KR=1
   26 CONTINUE
      SMJ=SMJ*DELFI
      SUMJ=SUMJ*DELFI
      SUMMJ=SUMMJ*DELFI
      SM=SM+SMJ
      SUMM=SUMM+SUMMJ
      VOL=VOL+VOLJ*DFST
   36 SUM=SUM+SUMJ
      darkin=3.141592653589793d0*(1.d0-x/3.d0)
      if(ld.eq.2) darkin=darkin+.6981317d0*y
      if(ld.eq.3) darkin=darkin-.6283185d0*y
      SBR=.25d0*XLUM/(SUM*DELTH*DARKIN)
      SM=SM*DELTH*4.d0
      SUMM=SUMM*DELTH*4.d0
      VOL=VOL*1.3333333333333d0*DELTH
      IF(KR.EQ.0) RETURN
      CALL ELLONE(F,D,RM,XL1,OMD,XLD,omdum)
      CALL NEKMIN(RM,OM,XL1,ZD)
      IF(komp.EQ.2) XL1=D-XL1
      R1=cmpp*R1H+cmp*R1C
      RL=cmpp*RLH+cmp*RLC
      VOL=VOL+1.047198d0*XL1*R1*RL
      RETURN
      END
