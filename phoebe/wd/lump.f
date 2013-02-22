      SUBROUTINE LUMP(GRX,GRY,GRZ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,
     $MMSAVE,ALB,TPOLL,SBR,N1,N2,KOMP,IFAT,fr,snth,
     $TLD,GLUMP1,GLUMP2,XX1,XX2,YY1,YY2,ZZ1,ZZ2,xbol,ybol
     $,GRV1,GRV2,SBR1B,SBR2B,RF,RFO,GMAG1,GMAG2,glog1,glog2,DINT,iband)
c   Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION GRX(*),GRY(*),GRZ(*),GRXQ(*),GRYQ(*),grzq(*),
     $SLUMP1(*),SLUMP2(*),MMSAVE(*),FR(*),SNTH(*),
     $TLD(*),GLUMP1(*),GLUMP2(*),XX1(*),XX2(*),YY1(*)
     $,YY2(*),ZZ1(*),ZZ2(*),GRV1(*),GRV2(*),RF(*),RFO(*),
     $GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      dimension message(2,4)
      common /atmmessages/ message,kompcom
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm
     $,ifs1dm,ifs2dm,icr1dm,icr2dm,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      kompcom=komp
      IQ=(KOMP-1)*(N1+1)
      IS=0
      IF(KOMP.EQ.2) IS=MMSAVE(IQ)
      PI=3.141592653589793d0
      PIH=.5d0*PI
      TPOLE=10000.d0*TPOLL
      cmp=dfloat(komp-1)
      cmpp=dfloat(2-komp)
      gplog=cmpp*gplog1+cmp*gplog2
      if(ifat.eq.0) call planckint(tpole,iband,pollog,pint)
      IF(IFAT.NE.0) CALL atmx(tpole,gplog,iband,pollog,pint)
      COMPP=dfloat(2*KOMP-3)
      COMP=-COMPP
      N=(2-KOMP)*N1+(KOMP-1)*N2
      NO=(2-KOMP)*N2+(KOMP-1)*N1
      NOD=2*NO
      EN=dfloat(N)
      ENO=dfloat(NO)
      DELTHO=PIH/ENO
      CNST=ALB*DELTHO*SBR2B/(DINT*SBR1B)
      IF(KOMP.EQ.2) CNST=ALB*DELTHO*SBR1B/(DINT*SBR2B)
      DO 191 I=1,N
      IPN1=I+N1*(KOMP-1)
      SINTH=SNTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      IP=(KOMP-1)*(N1+1)+I
      IY=MMSAVE(IP)
      DO 193 J=1,MM
      IX=IY+J
      SUM=0.d0
      IF(FR(IX).EQ.0.d0) GOTO 193
      DO 190 IOTH=1,NOD
      IOTHS=IOTH
      IF(IOTH.GT.NO) IOTHS=NOD-IOTH+1
      IPNO=IOTHS+N1*(2-KOMP)
      SINTHO=SNTH(IPNO)
      EMO=SINTHO*ENO*1.3d0
      MMO=EMO+1.d0
      MMOD=2*MMO
      IPO=(2-KOMP)*(N1+1)+IOTHS
      IYO=MMSAVE(IPO)
      XMO=MMO
      DELFIO=PI/XMO
      DO 190 JOFI=1,MMOD
      JOFU=JOFI
      IF(JOFI.GT.MMO) JOFU=MMOD-JOFI+1
      IXO=IYO+JOFU
      IX1=IX
      IX2=IXO
      IF(KOMP.EQ.1) GOTO 200
      IF(GLUMP1(IXO).EQ.0.d0) GOTO 184
      IX1=IXO
      IX2=IX
      GOTO 201
  200 CONTINUE
      IF(GLUMP2(IXO).EQ.0.d0) GOTO 179
  201 RTL1=1.d0
      RTL2=1.d0
      UPD1=1.d0
      UPD2=1.d0
      IF(KOMP.EQ.2) GOTO 22
      IF(JOFI.GT.MMO) RTL2=-1.d0
      IF(IOTH.GT.NO) UPD2=-1.d0
      GOTO 23
   22 IF(JOFI.GT.MMO) RTL1=-1.d0
      IF(IOTH.GT.NO) UPD1=-1.d0
   23 CONTINUE
      GX2=GRXQ(IX2)
      GY2=GRYQ(IX2)*RTL2
      GZ2=GRZQ(IX2)*UPD2
      X1C=XX1(IX1)
      X2C=XX2(IX2)
      Y1C=YY1(IX1)*RTL1
      Y2C=YY2(IX2)*RTL2
      Z1C=ZZ1(IX1)*UPD1
      Z2C=ZZ2(IX2)*UPD2
      DX=(X2C-X1C)*COMP
      DY=(Y2C-Y1C)*COMP
      DZ=(Z2C-Z1C)*COMP
      DLRSQ=DX*DX+DY*DY+DZ*DZ
      CSNUM2=(DX*GX2+DY*GY2+DZ*GZ2)*COMPP
      IF(CSNUM2.GE.0.d0) GOTO 190
      GX1=GRX(IX1)
      GY1=GRY(IX1)*RTL1
      GZ1=GRZ(IX1)*UPD1
      CSNUM1=(DX*GX1+DY*GY1+DZ*GZ1)*COMP
      IF(CSNUM1.GE.0.d0) GOTO 190
      DMAG=dsqrt(DLRSQ)
      CSGM1=-CSNUM1/(DMAG*GMAG1(IX1))
      CSGM2=-CSNUM2/(DMAG*GMAG2(IX2))
      IF(KOMP.EQ.2) GOTO 181
      DGAM2=1.d0-XBOL+XBOL*CSGM2
      if(ld.ne.2) goto 179
      if(csgm2.eq.0.d0) goto 179
      dgam2=dgam2-ybol*csgm2*dlog(csgm2)
      goto 147
  179 continue
      if(ld.eq.3) dgam2=dgam2-ybol*(1.d0-dsqrt(csgm2))
  147 if(dgam2.lt.0.d0) dgam2=0.d0
      DSUM=GRV2(IXO)*GLUMP2(IXO)*RFO(IXO)*CSGM1*CSGM2*DGAM2/DLRSQ
      GOTO 182
  181 DGAM1=1.d0-XBOL+XBOL*CSGM1
      if(ld.ne.2) goto 184
      if(csgm1.eq.0.d0) goto 184
      dgam1=dgam1-ybol*csgm1*dlog(csgm1)
      goto 148
  184 continue
      if(ld.eq.3) dgam1=dgam1-ybol*(1.d0-dsqrt(csgm1))
  148 if(dgam1.lt.0.d0) dgam1=0.d0
      DSUM=GRV1(IXO)*GLUMP1(IXO)*RFO(IXO)*CSGM2*CSGM1*DGAM1/DLRSQ
  182 CONTINUE
      SUM=SUM+DSUM*DELFIO
  190 CONTINUE
      RF(IX)=(CNST*SUM/(CMPP*GRV1(IX)+CMP*GRV2(IX)))+1.d0
  193 CONTINUE
  191 CONTINUE
      DO 8 I=1,N
      IPN1=I+N1*(KOMP-1)
      SINTH=SNTH(IPN1)
      EM=SINTH*EN*1.3d0
      MM=EM+1.d0
      IP=(KOMP-1)*(N1+1)+I
      IY=MMSAVE(IP)
      DO 8 J=1,MM
      IS=IS+1
      IX=IY+J
      IF(FR(IX).EQ.0.d0) GOTO 8
      glogg=cmpp*glog1(ix)+cmp*glog2(ix)
      grv=cmpp*grv1(ix)+cmp*grv2(ix)
      TNEW=TPOLE*dsqrt(dsqrt(GRV*RF(IX)))
      TLD(IS)=TNEW
      if(ifat.eq.0) call planckint(tnew,iband,xintlog,xint)
      if(ifat.ne.0) call atmx(tnew,glogg,iband,xintlog,xint)
      GRREFL=xint/pint
      IF(KOMP.EQ.1) GOTO 77
      slump2(ix)=glump2(ix)*grrefl*sbr
      GOTO 8
   77 slump1(ix)=glump1(ix)*grrefl*sbr
    8 CONTINUE
      RETURN
      END
