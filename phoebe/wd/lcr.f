      SUBROUTINE LCR(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,
     $hld,SLUMP1,SLUMP2,RM,POTH,POTC,N1,N2,F1,F2,D,HLUM,CLUM,xh,xc,yh,
     $yc,GR1,GR2,SM1,SM2,TPOLH,TPOLC,SBRH,SBRC,IFAT1,IFAT2,TAVH,TAVC,
     $alb1,alb2,xbol1,xbol2,ybol1,ybol2,vol1,vol2,snth,csth,snfi,csfi,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,dint1,dint2,grv1,grv2,
     $csbt1,csbt2,rftemp,rf1,rf2,gmag1,gmag2,glog1,glog2,mode,iband)
c  Version of January 8, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),SLUMP1(*),SLUMP2(*),MMSAVE(*),FR1(*),FR2(*),HLD(*),SNTH(*),
     $CSTH(*),SNFI(*),CSFI(*),TLD(*),GLUMP1(*),GLUMP2(*),XX1(*),XX2(*)
     $,YY1(*),YY2(*),ZZ1(*),ZZ2(*),GRV1(*),GRV2(*),RFTEMP(*),RF1(*),
     $RF2(*),CSBT1(*),CSBT2(*),GMAG1(*),GMAG2(*),glog1(*),glog2(*)
      dimension message(2,4)
      common /atmmessages/ message,komp
      common /coflimbdark/ xld,yld
      COMMON /DPDX/ DPDX1,DPDX2,PHSV,PCSV
      COMMON /ECCEN/ E,dum1,dum2,dum3,dum4,dum5,dum6,dum7,dum8,dum9,ifc
      COMMON /SUMM/ SUMM1,SUMM2
      COMMON /INVAR/ KHDUM,IPB,IRTE,NREF,IRVOL1,IRVOL2,mref,ifsmv1,
     $ifsmv2,icor1,icor2,ld,ncl,jdphs,ipc
      common /gpoles/ gplog1,gplog2
      nn1=n1
      VL1=VOL1
      VL2=VOL2
      DP=1.d0-E
      IF(IRVOL1.EQ.1) GOTO 88
      CALL VOLUME(VL1,RM,POTH,DP,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM1,SM1,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR1,1)
      IF(E.EQ.0.d0) GOTO 88
      POTHD=PHSV
      IF(IFC.EQ.2) POTHD=PHSV+DPDX1*(1.d0/D-1.d0/(1.d0-E))
      CALL VOLUME(VL1,RM,POTHD,D,F1,nn1,N1,1,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM1,SM1,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR1,IFC)
   88 CONTINUE
      IF(IRVOL2.EQ.1) GOTO 86
      CALL VOLUME(VL2,RM,POTC,DP,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM2,SM2,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR2,1)
      IF(E.EQ.0.d0) GOTO 86
      POTCD=PCSV
      IF(IFC.EQ.2) POTCD=PCSV+DPDX2*(1.d0/D-1.d0/(1.d0-E))
      CALL VOLUME(VL2,RM,POTCD,D,F2,N2,N1,2,RV,GRX,GRY,GRZ,RVQ,GRXQ,
     $GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM2,SM2,GRV1,
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2
     $,glog1,glog2,GR2,IFC)
   86 CONTINUE
      TPOLH=TAVH*dsqrt(dsqrt(SM1/SUMM1))
      TPOLC=TAVC*dsqrt(dsqrt(SM2/SUMM2))
      g1=gmag1(1)
      g2=gmag2(1)
      IF(MODE.EQ.1)TPOLC=TPOLH*dsqrt(dsqrt((G2/G1)**GR1))
      IF(MODE.EQ.1)TAVC=TPOLC/dsqrt(dsqrt(SM2/SUMM2))
      tph=10000.d0*tpolh
      tpc=10000.d0*tpolc
      komp=1
      xld=xh
      yld=yh
      if(ifat1.eq.0) call planckint(tph,iband,xintlog1,xint1)
      IF(IFAT1.NE.0) CALL atmx(tph,gplog1,iband,xintlog1,xint1)
      call lum(hlum,xh,yh,tpolh,n1,n1,1,sbrh,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ1d,fr1,sm1d,ifat1,vold,rm,
     $poth,f1,d,snth,iband)
      komp=2
      xld=xc
      yld=yc
      if(ifat2.eq.0) call planckint(tpc,iband,xintlog2,xint2)
      IF(IFAT2.NE.0) CALL atmx(tpc,gplog2,iband,xintlog2,xint2)
      sbrc=sbrh*xint2/xint1
      call lum(clum,xc,yc,tpolc,n2,n1,2,sbrt,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ2d,fr2,sm2d,ifat2,vold,rm,
     $potc,f2,d,snth,iband)
      IF(IPB.EQ.1) SBRC=SBRT
      IF(MODE.GT.0)CLUM=CLUM*SBRC/SBRT
      IF(MODE.LE.0)SBRC=SBRT
      if(mref.eq.2) goto 30
      radrat=(vol1/vol2)**(1.d0/3.d0)
      ratbol=radrat**2*(tavh/tavc)**4
      rb=1.d0/ratbol
      xld=xh
      yld=yh
      call olump(rv,grx,gry,grz,rvq,grxq,gryq,grzq,slump1,slump2,mmsave
     $,gr1,alb1,rb,tpolh,sbrh,summ1,n1,n2,1,ifat1,xc,yc,d,snth
     $,csth,snfi,csfi,tld,glump1,glump2,glog1,glog2,grv1,grv2,iband)
      rb=ratbol
      xld=xc
      yld=yc
      call olump(rv,grx,gry,grz,rvq,grxq,gryq,grzq,slump1,slump2,mmsave
     $,gr2,alb2,rb,tpolc,sbrc,summ2,n1,n2,2,ifat2,xh,yh,d,snth
     $,csth,snfi,csfi,tld,glump1,glump2,glog1,glog2,grv1,grv2,iband)
      return
   30 continue
      sbr1b=tpolh**4/dint1
      sbr2b=tpolc**4/dint2
      LT=N1+1
      IMAX1=MMSAVE(LT)
      DO 80 I=1,IMAX1
      RFTEMP(I)=1.d0
   80 RF1(I)=1.d0
      LT=N1+N2+2
      IMAX2=MMSAVE(LT)
      DO 81 I=1,IMAX2
   81 RF2(I)=1.d0
      DO 93 NR=1,NREF
      xld=xh
      yld=yh
      CALL LUMP(GRX,GRY,GRZ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,MMSAVE,
     $alb1,tpolh,sbrh,n1,n2,1,ifat1,fr1,snth,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,xbol2,ybol2,grv1,
     $grv2,sbr1b,sbr2b,rftemp,rf2,gmag1,gmag2,glog1,glog2,dint1,iband)
      xld=xc
      yld=yc
      CALL LUMP(GRX,GRY,GRZ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,MMSAVE,
     $ALB2,TPOLC,SBRC,N1,N2,2,IFAT2,fr2,snth,
     $tld,glump1,glump2,xx1,xx2,yy1,yy2,zz1,zz2,xbol1,ybol1,
     $grv1,grv2,sbr1b,sbr2b,rf2,rf1,gmag1,gmag2,glog1,glog2,dint2,iband)
      DO 70 I=1,IMAX1
   70 RF1(I)=RFTEMP(I)
   93 CONTINUE
      RETURN
      END
