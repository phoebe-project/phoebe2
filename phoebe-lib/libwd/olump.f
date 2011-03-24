      SUBROUTINE OLUMP(RV,GRX,GRY,GRZ,RVQ,GRXQ,GRYQ,GRZQ,SLUMP1,SLUMP2,  
     $MMSAVE,GREXP,ALB,RB,TPOLL,SBR,SUMM,N1,N2,KOMP,IFAT,x,y,D,       
     $SNTH,CSTH,SNFI,CSFI,tld,glump1,glump2,glog1,glog2,grv1,grv2,iband)                             
c   Version of January 8, 2003                                        
      implicit real*8 (a-h,o-z)                                          
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(* 
     $),SLUMP1(*),SLUMP2(*),MMSAVE(*),F(3),W(3),SNTH(*),CSTH(*),         
     $SNFI(*),CSFI(*),tld(*),glump1(*),glump2(*),glog1(*),glog2(*),                         
     $grv1(*),grv2(*) 
      dimension message(2,4) 
      common /atmmessages/ message,kompcom 
      common /invar/ khdum,ipbdum,irtedm,nrefdm,irv1dm,irv2dm,mrefdm,    
     $ifs1dm,ifs2dm,icr1dm,icr2dm,ld,ncl,jdphs,ipc                       
      common /gpoles/ gplog1,gplog2 
      kompcom=komp 
      IQ=(KOMP-1)*(N1+1)                                                 
      IS=(KOMP-1)*MMSAVE(IQ)                                             
      FP=7.957747d-2                                                     
      PI=3.141592653589793d0                                             
      PIH=1.570796326794897d0                                            
      PI32=4.712388980384690d0                                           
      F(1)=.1127017d0                                                    
      F(2)=.5d0                                                          
      F(3)=.8872983d0                                                    
      W(1)=.277777777777777d0                                            
      W(2)=.444444444444444d0                                            
      W(3)=.277777777777777d0                                            
      TPOLE=10000.d0*TPOLL                                               
      cmp=dfloat(komp-1) 
      cmpp=dfloat(2-komp) 
      gplog=cmpp*gplog1+cmp*gplog2 
      if(ifat.eq.0) call planckint(tpole,iband,pollog,pint) 
      IF(IFAT.NE.0) CALL atmx(tpole,gplog,iband,pollog,pint)                            
      COMPP=dfloat(2*KOMP-3)                                             
      COMP=-COMPP                                                        
      CMPD=CMP*D                                                         
      CMPPD=CMPP*D                                                       
      N=(2-KOMP)*N1+(KOMP-1)*N2                                          
      ENN=(15.d0+X)*(1.d0+GREXP)/(15.d0-5.d0*X)                          
      NP=N1+1+(2-KOMP)*(N2+1)                                            
      NPP=N1*(KOMP-1)+(NP-1)*(2-KOMP)                                    
      LL=MMSAVE(NPP)+1                                                   
      LLL=MMSAVE(NP)                                                     
      LLLL=(LL+LLL)/2                                                    
      AR=RV(LLL)*CMP+RVQ(LLL)*CMPP                                       
      BR=RV(LLLL)*CMP+RVQ(LLLL)*CMPP                                     
      CR=RV(1)*CMP+RVQ(1)*CMPP                                           
      BOA=BR/AR                                                          
      BOAL=1.d0-BOA*BOA                                                  
      BOC2=(BR/CR)**2                                                    
      CC=1.d0/(1.d0-.25d0*ENN*(1.d0-BOA**2)*(.9675d0-.3008d0*BOA))       
      HCN=.5d0*CC*ENN                                                    
      DF=1.d0-X/3.d0                                                     
      if(ld.eq.2) df=df+2.d0*y/9.d0                                      
      if(ld.eq.3) df=df-.2d0*y                                           
      EN=dfloat(N)                                                       
      DO 8 I=1,N                                                         
      IPN1=I+N1*(KOMP-1)                                                 
      SINTH=SNTH(IPN1)                                                   
      COSTH=CSTH(IPN1)                                                   
      EM=SINTH*EN*1.3d0                                                  
      MM=EM+1.d0                                                         
      IP=(KOMP-1)*(N1+1)+I                                               
      IY=MMSAVE(IP)                                                      
      DO 8 J=1,MM                                                        
      IS=IS+1                                                            
      STCF=SINTH*CSFI(IS)                                                
      STSF=SINTH*SNFI(IS)                                                
      IX=IY+J                                                            
      IF(KOMP.EQ.1) GOTO 39                                              
      IF(RVQ(IX).EQ.-1.d0) GOTO 8                                        
      GX=GRXQ(IX)                                                        
      GY=GRYQ(IX)                                                        
      GZ=GRZQ(IX)                                                        
      R=RVQ(IX)                                                          
      GOTO 49                                                            
   39 IF(RV(IX).EQ.-1.d0)GOTO 8                                          
      GX=GRX(IX)                                                         
      GY=GRY(IX)                                                         
      GZ=GRZ(IX)                                                         
      R=RV(IX)                                                           
   49 GRMAG=dsqrt(GX*GX+GY*GY+GZ*GZ)                                     
      ZZ=R*COSTH                                                         
      YY=R*COMP*STSF                                                     
      XX=CMPD+COMP*STCF*R                                                
      XXREF=(CMPPD+COMPP*XX)*COMPP                                       
      GRAV=cmpp*grv1(ix)+cmp*grv2(ix)                                         
      TLOCAL=TPOLE*dsqrt(dsqrt(GRAV))                                    
      DIST=dsqrt(XXREF*XXREF+YY*YY+ZZ*ZZ)                                
      RMX=dasin(.5d0*(BR+CR)/DIST)                                       
      XCOS=XXREF/DIST                                                    
      YCOS=YY/DIST                                                       
      ZCOS=ZZ/DIST                                                       
      COSINE=(XCOS*GX+YCOS*GY+ZCOS*GZ)/GRMAG                             
      RC=PIH-dacos(COSINE)                                               
      AH=RC/RMX                                                          
      RP=dabs(AH)                                                        
      IF(AH.LE..99999d0) GOTO 22                                         
      P=1.d0                                                             
      GOTO 16                                                            
   22 IF(AH.GE.-.99999d0) GOTO 24                                        
      ALBEP=0.d0                                                         
      GOTO 19                                                            
   24 SUM=0.d0                                                           
      FIST=dasin(RP)                                                     
      FII=PIH-FIST                                                       
      DO 15 IT=1,3                                                       
      FE=FII*F(IT)+FIST                                                  
      PAR=1.d0-(RP/dsin(FE))**2                                          
      RPAR=dsqrt(PAR)                                                    
      SUM=PAR*RPAR*W(IT)+SUM                                             
   15 CONTINUE                                                           
      FTRI=(1.d0-X)*RP*dsqrt(1.d0-RP**2)+.666666666666666d0*X*FII        
     $-.666666666666667d0*x*sum*fii                                      
      FSEC=(PIH+FIST)*DF                                                 
      P=(FTRI+FSEC)/(PI*DF)                                              
      IF(COSINE.LT.0.d0) P=1.d0-P                                        
      RTF=dsqrt(1.d0-AH**2)                                              
      DENO=PI32-3.d0*(AH*RTF+dasin(AH))                                  
      IF(DENO.NE.0.d0) GOTO 117                                          
      ABAR=1.d0                                                          
      GOTO 116                                                           
  117 ABAR=2.d0*RTF**3/DENO                                              
  116 COSINE=dcos(PIH-RMX*ABAR)                                          
   16 COSQ=1.d0/(1.d0+(YY/XXREF)**2)                                     
      COT2=(ZZ/XXREF)**2                                                 
      Z=BOAL/(1.d0+BOC2*COT2)                                            
      E=CC-HCN*COSQ*Z                                                    
      ALBEP=ALB*E*P                                                      
   19 IF(COSINE.LE.0.d0) ALBEP=0.d0                                      
      TNEW=TLOCAL*dsqrt(dsqrt(1.d0+(FP*SUMM/(DIST*DIST*GRAV))*           
     $cosine*rb*ALBEP))                                                  
      TLD(IS)=TNEW                                                       
      glogg=cmpp*glog1(ix)+cmp*glog2(ix) 
      if(ifat.eq.0) call planckint(tnew,iband,xintlog,xint) 
      if(ifat.ne.0) CALL atmx(TNEW,glogg,iband,xintlog,xint) 
      grrefl=xint/pint                                                
      IF(KOMP.EQ.1) GOTO 77                                              
      slump2(ix)=glump2(ix)*grrefl*sbr                                   
      GOTO 8                                                             
   77 slump1(ix)=glump1(ix)*grrefl*sbr                                   
    8 CONTINUE                                                           
      RETURN                                                             
      END                                                                
