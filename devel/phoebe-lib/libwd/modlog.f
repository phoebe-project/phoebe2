      SUBROUTINE MODLOG(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2
     $,HLD,RM,POTH,POTC,GR1,GR2,ALB1,ALB2,N1,N2,F1,F2,MOD,XINCL,THE,
     $MODE,SNTH,CSTH,SNFI,CSFI,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,GLUMP1
     $,GLUMP2,CSBT1,CSBT2,GMAG1,GMAG2,glog1,glog2)
c    Version of December 18, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ
     $(*),MMSAVE(*),FR1(*),FR2(*),HLD(*),GRV1(*),GRV2(*),XX1(*),YY1(*),
     $ZZ1(*),XX2(*),YY2(*),ZZ2(*),GLUMP1(*),GLUMP2(*),CSBT1(*),CSBT2(*)
     $,GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      DIMENSION DRR(4),RES(2),ANS(2),LX(2),MX(2)
      DIMENSION SNTH(*),CSTH(*),SNFI(*),CSFI(*)
      common /kfac/ kff1,kff2,kfo1,kfo2
      common /setest/ sefac
      common /ardot/ dperdt,hjd,hjd0,perr
      COMMON /FLVAR/ PSHIFT,DP,EF,EFC,ECOS,perr0,PHPER,pconsc,pconic,
     $PHPERI,VSUM1,VSUM2,VRA1,VRA2,VKM1,VKM2,VUNIT,vfvu,trc,qfacd
      COMMON /ECCEN/ E,A,PERIOD,VGA,SINI,VF,VFAC,VGAM,VOL1,VOL2,IFC
      COMMON /INVAR/ KH,IPBDUM,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,
     $ifsmv2,icor1,icor2,ld,ncl,jdphs,ipc
   95 FORMAT(' WARNING: ALTHOUGH COMPONENT 2 DOES NOT EXCEED ITS LIMITIN
     $G LOBE AT THE END OF ECLIPSE, IT DOES EXCEED THE LOBE AT PERIASTRO
     $N')
   99 FORMAT(' SPECIFIED ECLIPSE DURATION INCONSISTENT WITH OTHER PARAME
     $TERS')
      perr=perr0+dperdt*(hjd-hjd0)
      DP=1.d0-E
      MOD=(MODE-2)**2
      IF(MODE.EQ.1) GR2=GR1
      IF(MODE.EQ.1) ALB2=ALB1
      IF(MOD.EQ.1) POTC=POTH
      MD4=(MODE-5)**2
      MD5=(2*MODE-11)**2
      call ellone(f1,dp,rm,xl1,po1cr,xl2,omo1)
      sefac=.8712d0
      doc=(po1cr-poth)/(po1cr-omo1)
      if(doc.gt.0.d0) sefac=.201d0*doc*doc-.386d0*doc+.8712d0
      RMR=1.d0/RM
      CALL ELLONE(F2,DP,RMR,XL1,po2c,XL2,omo2)
      po2cr=rm*po2c+(1.d0-rm)*.5d0
      if(md4.eq.1) poth=po1cr
      if(md5.eq.1) potc=po2cr
      kff1=0
      kff2=0
      if(poth.lt.po1cr) kff1=1
      if(potc.lt.po2cr) kff2=1
      kfo1=0
      kfo2=0
      if(e.ne.0.d0) goto 100
      if(f1.ne.1.d0) goto 105
      if(poth.lt.omo1) kfo1=1
  105 if(f2.ne.1.d0) goto 100
      if(potc.lt.omo1) kfo2=1
  100 continue
      SINI=dsin(.017453292519943d0*XINCL)
      VF=50.61455d0/PERIOD
      VFAC=VF*A
      VGAM=VGA*VUNIT/VFAC
      VFVU=VFAC
      IFC=2
      IF(E.NE.0.d0) GOTO 60
      perr=1.570796326794897d0
      IFC=1
   60 CONTINUE
      TRC=1.570796326794897d0-perr
   39 if(TRC.LT.0.d0) TRC=TRC+6.283185307179586d0
      if(trc.lt.0.d0) goto 39
   40 if(trc.ge.6.283185307179586d0) trc=trc-6.283185307179586d0
      if(trc.ge.6.283185307179586d0) goto 40
      HTRC=.5d0*TRC
      IF(dabs(1.570796326794897d0-HTRC).LT.7.d-6) GOTO 101
      IF(dabs(4.712388980384690d0-HTRC).LT.7.d-6) GOTO 101
      ECAN=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(HTRC))
      GOTO 103
  101 ECAN=3.141592653589793d0
  103 XMC=ECAN-E*dsin(ECAN)
      IF(XMC.LT.0.d0) XMC=XMC+6.283185307179586d0
      PHPER=1.d0-XMC/6.283185307179586d0
      call conjph(e,perr,pshift,trsc,tric,econsc,econic,xmsc,xmic,
     $pconsc,pconic)
   38 if(pconsc.ge.1.d0) pconsc=pconsc-1.d0
      if(pconsc.ge.1.d0) goto 38
   41 if(pconsc.lt.0.d0) pconsc=pconsc+1.d0
      if(pconsc.lt.0.d0) goto 41
   68 if(pconic.ge.1.d0) pconic=pconic-1.d0
      if(pconic.ge.1.d0) goto 68
   71 if(pconic.lt.0.d0) pconic=pconic+1.d0
      if(pconic.lt.0.d0) goto 71
      PHPERI=PHPER+pconsc
      EF=1.d0-E*E
      EFC=dsqrt(EF)
      ECOS=E*dcos(perr)
      IF(MODE.NE.-1) RETURN
      if(kh.eq.17) goto 241
      if((kh-12)**2.eq.1) goto 241
      if((kh-12)**2.eq.4) goto 241
      IF((KH-11)**2.LE.1) GOTO 241
      IF((2*KH-41)**2.EQ.81) GOTO 241
      RETURN
  241 CONTINUE
      EFCC=dsqrt((1.d0-E)/(1.d0+E))
      THER=THE*6.283185307179586d0
      DELTR=.001d0
      DTR1=0.d0
      DTR2=0.d0
      VOLTOL=5.d-6
      DXMTOL=5.d-6
      TR0=1.570796326794897d0-perr
      HTR0=.5d0*TR0
      IF((1.570796326794897d0-dabs(HTR0)).LT.7.d-6) GOTO 201
      IF((4.712388980384690d0-dabs(HTR0)).LT.7.d-6) GOTO 201
      ECAN0=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(HTR0))
      GOTO 203
  201 ECAN0=3.141592653589793d0
  203 XM0=ECAN0-E*dsin(ECAN0)
      XM1=XM0-THER*(1.d0-.2d0*E)
      XM2=XM0+THER*(1.d0-.2d0*E)
      CALL KEPLER(XM1,E,DUM,TRR1)
      CALL KEPLER(XM2,E,DUM,TRR2)
  160 TRR1=TRR1+DTR1
      TRR2=TRR2+DTR2
      DO 161 IB=1,3
      TR1=TRR1
      TR2=TRR2
      IF(IB.EQ.2) TR1=TRR1+DELTR
      IF(IB.EQ.3) TR2=TRR2+DELTR
      IF(TR1.GT.TR0) TR0=TR0+6.283185307179586d0
      IF(TR0.GT.TR2) TR2=TR2+6.283185307179586d0
      DS1=EF/(1.d0+E*dcos(TR1))
      DS2=EF/(1.d0+E*dcos(TR2))
      TRE1=(TR0-TR1)/6.283185307179586d0
      TRE2=(TR2-TR0)/6.283185307179586d0
      CALL DURA(F2,XINCL,RM,DS1,TRE1,POTR,RA)
      CALL VOLUME(VS1,RM,POTR,DS1,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ
     $,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,
     $GMAG2,glog1,glog2,GR1,1)
      CALL DURA(F2,XINCL,RM,DS2,TRE2,POTR,RA)
      CALL VOLUME(VS2,RM,POTR,DS2,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ
     $,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMMD,SMD,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,
     $GMAG2,glog1,glog2,GR2,1)
      IF(IB.NE.1) GOTO 185
      ECAN1=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(.5d0*TR1))
      ECAN2=2.d0*datan(dsqrt((1.d0-E)/(1.d0+E))*dtan(.5d0*TR2))
      POTC=POTR
      DTHE=DS2
      DVOL=VS2-VS1
      XM1=ECAN1-E*dsin(ECAN1)
      XM2=ECAN2-E*dsin(ECAN2)
      IF(XM1.LT.0.d0) XM1=XM1+6.283185307179586d0
      IF(XM2.LT.0.d0) XM2=XM2+6.283185307179586d0
      DXM=XM2-XM1-2.d0*THER
      DDMDN1=-EFCC*(1.d0-E*dcos(ECAN1))*dcos(.5d0*ECAN1)**2/
     $dcos(.5d0*tr1)**2
      DDMDN2=EFCC*(1.d0-E*dcos(ECAN2))*dcos(.5d0*ECAN2)**2/
     $dcos(.5d0*tr2)**2
  185 CONTINUE
      IF(IB.NE.2) GOTO 162
      DRR(1)=(VS2-VS1-DVOL)/DELTR
      DRR(2)=DDMDN1
  162 CONTINUE
      IF(IB.NE.3) GOTO 161
      DRR(3)=(VS2-VS1-DVOL)/DELTR
      DRR(4)=DDMDN2
  161 CONTINUE
      RES(1)=-DVOL
      RES(2)=-DXM
      CALL DMINV(DRR,2,DUMM,LX,MX)
      CALL DGMPRD(DRR,RES,ANS,2,2,1)
      DTR1=ANS(1)
      DTR2=ANS(2)
      IF(dabs(DTR1).GT.VOLTOL) GOTO 160
      IF(dabs(DTR2).GT.DXMTOL) GOTO 160
      POTH=9999.99d0
      RMR=1.d0/RM
      CALL ELLONE(F2,DTHE,RMR,XLA,OM1,XL2,OM2)
      OM1=RM*OM1+(1.d0-RM)*.5d0
      IF(POTC.LT.OM1) GOTO 22
      IF(RA.LE.XLA) GOTO 28
   22 WRITE(6,99)
      RETURN
   28 CONTINUE
      IF(E.NE.0.d0) CALL VOLUME(VTHE,RM,POTC,DTHE,F2,N2,N1,2,RV,GRX,
     $GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,
     $SUMMD,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,
     $GLUMP2,GMAG1,GMAG2,glog1,glog2,GR2,1)
      IF(E.NE.0.d0) CALL VOLUME(VTHE,RM,POTC,DP,F2,N2,N1,2,RV,GRX,
     $GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,
     $SUMMD,SMD,GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,
     $GLUMP2,GMAG1,GMAG2,glog1,glog2,GR2,2)
      CALL ELLONE(F2,DP,RMR,XLD,OMP,XL2,OM2)
      OMP=RM*OMP+(1.d0-RM)*.5d0
      IF(POTC.LT.OMP) WRITE(6,95)
      RETURN
      END
