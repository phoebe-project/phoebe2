      SUBROUTINE SINCOS (KOMP,N,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)
c  Version of November 9, 1995
      implicit real*8 (a-h,o-z)
      DIMENSION SNTH(*),CSTH(*),SNFI(*),CSFI(*),MMSAVE(*)
      IP=(KOMP-1)*(N1+1)+1
      IQ=IP-1
      IS=0
      IF(KOMP.EQ.2) IS=MMSAVE(IQ)
      MMSAVE(IP)=0
      EN=N
      DO 8 I=1,N
      EYE=I
      EYE=EYE-.5d0
      TH=1.570796326794897d0*EYE/EN
      IPN1=I+N1*(KOMP-1)
      SNTH(IPN1)=dsin(TH)
      CSTH(IPN1)=dcos(TH)
      EM=SNTH(IPN1)*EN*1.3d0
      MM=EM+1.d0
      XM=MM
      IP=(KOMP-1)*(N1+1)+I+1
      IQ=IP-1
      MMSAVE(IP)=MMSAVE(IQ)+MM
      DO 8 J=1,MM
      IS=IS+1
      XJ=J
      FI=3.141592653589793d0*(XJ-.5d0)/XM
      CSFI(IS)=dcos(FI)
      SNFI(IS)=dsin(FI)
    8 CONTINUE
      RETURN
      END
