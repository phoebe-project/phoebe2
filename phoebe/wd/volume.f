      SUBROUTINE VOLUME(V,Q,P,D,FF,N,N1,KOMP,RV,GRX,GRY,GRZ,RVQ,
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,SNTH,CSTH,SNFI,CSFI,SUMM,SM,
     $GRV1,GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1
     $,GMAG2,glog1,glog2,GREXP,IFC)
c  Version of December 5, 2003
      implicit real*8 (a-h,o-z)
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(*
     $),MMSAVE(*),FR1(*),FR2(*),HLD(*),SNTH(*),CSTH(*),SNFI(*),CSFI(*)
     $,GRV1(*),GRV2(*),GLUMP1(*),GLUMP2(*),XX1(*),YY1(*),ZZ1(*),XX2(*),
     $YY2(*),ZZ2(*),CSBT1(*),CSBT2(*),GMAG1(*),GMAG2(*),glog1(*),
     $glog2(*)
      if(ifc.eq.1) v=0.d0
      DP=1.d-5*P
      ot=1.d0/3.d0
      IF (IFC.EQ.1) DP=0.d0
      tolr=1.d-8
      DELP=0.d0
      KNTR=0
   16 P=P+DELP
      KNTR=KNTR+1
      IF(KNTR.GE.20) tolr=tolr+tolr
      PS=P
      DO 17 I=1,IFC
      P=PS
      IF(I.EQ.1) P=P+DP
      CALL SURFAS(Q,P,N,N1,KOMP,RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,
     $MMSAVE,FR1,FR2,HLD,FF,D,SNTH,CSTH,SNFI,CSFI,GRV1,GRV2,XX1,YY1,ZZ1,
     $XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,GMAG2,glog1,glog2,
     $grexp)
      IF(KOMP.EQ.2) GOTO 14
      call lum(1.d0,1.d0,0.d0,1.d0,n,n1,1,sbrd,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ,fr1,sm,0,vol,q,p,ff,d,
     $snth,7)
      GOTO 15
   14 call lum(1.d0,1.d0,0.d0,1.d0,n,n1,2,sbrd,rv,rvq,glump1,
     $glump2,glog1,glog2,grv1,grv2,mmsave,summ,fr2,sm,0,vol,q,p,ff,d,
     $snth,7)
   15 CONTINUE
      IF(I.EQ.1) VOLS=VOL
      VOL2=VOLS
   17 VOL1=VOL
      rmean=(.238732414d0*vol)**ot
      rmsq=rmean**2
c
c  Here use a polar estimate for d(potential)/dr (absolute value).
c
      domdrabs=1.d0/rmsq+q*rmean/(d*d+rmsq)
      tolp=domdrabs*tolr
      IF(IFC.EQ.1) V=VOL
      IF(IFC.EQ.1) RETURN
      DPDV=DP/(VOL2-VOL1)
      DELP=(V-VOL1)*DPDV
      ABDELP=dabs(DELP)
      IF(ABDELP.GT.tolp) GOTO 16
      P=PS
      RETURN
      END
