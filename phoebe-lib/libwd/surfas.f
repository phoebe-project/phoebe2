      SUBROUTINE SURFAS(RMASS,POTENT,N,N1,KOMP,RV,GRX,GRY,GRZ,RVQ,       
     $GRXQ,GRYQ,GRZQ,MMSAVE,FR1,FR2,HLD,FF,D,SNTH,CSTH,SNFI,CSFI,GRV1,   
     $GRV2,XX1,YY1,ZZ1,XX2,YY2,ZZ2,CSBT1,CSBT2,GLUMP1,GLUMP2,GMAG1,      
     $GMAG2,glog1,glog2,GREXP)                                                       
c  Version of June 9, 2004                                            
      implicit real*8 (a-h,o-z)                                          
      DIMENSION RV(*),GRX(*),GRY(*),GRZ(*),RVQ(*),GRXQ(*),GRYQ(*),GRZQ(* 
     $),MMSAVE(*),FR1(*),FR2(*),HLD(*),SNTH(*),CSTH(*),SNFI(*),CSFI(*)   
     $,GRV1(*),GRV2(*),XX1(*),YY1(*),ZZ1(*),XX2(*),YY2(*),ZZ2(*),GLUMP1  
     $(*),GLUMP2(*),CSBT1(*),CSBT2(*),GMAG1(*),GMAG2(*),glog1(*),                 
     $glog2(*) 
      common /gpoles/ gplog1,gplog2 
      common /radi/ R1H,RLH,R1C,RLC                                      
      COMMON /misc/ X1                                                        
      COMMON /ECCEN/e,smaxis,period,vgadum,sindum,vfdum,vfadum,vgmdum,     
     $v1dum,v2dum,ifcdum                                                 
      DSQ=D*D                                                            
      RMAS=RMASS                                                         
      IF(KOMP.EQ.2) RMAS=1.d0/RMASS                                      
      RF=FF**2                                                           
      RTEST=0.d0                                                         
      IP=(KOMP-1)*(N1+1)+1                                               
      IQ=IP-1                                                            
      IS=0                                                               
      ISX=(KOMP-1)*MMSAVE(IQ)                                            
      MMSAVE(IP)=0                                                       
      KFLAG=0                                                            
      CALL ELLONE (FF,D,RMAS,X1,OMEGA,XL2,OM2)                           
      IF(KOMP.EQ.2) OMEGA=RMASS*OMEGA+.5d0*(1.d0-RMASS)                  
      X2=X1                                                              
      IF(KOMP.EQ.2) X1=1.d0-X1                                           
      IF(E.NE.0.d0) GOTO 43                                              
      IF(POTENT.LT.OMEGA) CALL NEKMIN(RMASS,POTENT,X1,ZZ)                
      IF(POTENT.LT.OMEGA) X2=1.d0-X1                                     
   43 COMP=dfloat(3-2*KOMP)                                              
      CMP=dfloat(KOMP-1)                                                 
      CMPD=CMP*D                                                         
      TESTER=CMPD+COMP*X1
      RM1=RMASS+1.d0                                                     
      RMS=RMASS                                                          
      RM1S=RM1                                                           
      IF(KOMP.NE.2) GOTO 15                                              
      POT=POTENT/RMASS+.5d0*(RMASS-1.d0)/RMASS                           
      RM=1.d0/RMASS                                                      
      RM1=RM+1.d0                                                        
      GOTO 20                                                            
   15 POT=POTENT                                                         
      RM=RMASS                                                           
   20 EN=N                                                               
c ******************************************** 
c  Find the relative polar radius, R/a 
      DELR=0.d0                                                          
      R=1.d0/pot                                                            
      knt=0
  714 R=R+DELR                                                           
      knt=knt+1
      tolr=1.d-6*dabs(r)
      RSQ=R*R                                                            
      PAR=DSQ+RSQ                                          
      RPAR=dsqrt(PAR)                                                    
      OM=1.d0/R+RM/RPAR       
      DOMR=1.d0/(-1.d0/RSQ-RM*R/(PAR*RPAR))     
      DELR=(POT-OM)*DOMR                                                 
      ABDELR=dabs(DELR)                                                  
      IF(ABDELR.GT.tolr) GOTO 714                                     
      rpole=r 
      rsave=r
c ******************************************** 
c  Now compute GRPOLE (exactly at the pole) 
      x=cmpd 
      zsq=rpole*rpole 
      PAR1=x*x+zsq                                                    
      RPAR1=dsqrt(PAR1)                                                  
      XNUM1=1.d0/(PAR1*RPAR1)                                            
      XL=D-X                                                             
      PAR2=XL**2+zsq                                                   
      RPAR2=dsqrt(PAR2)                                                  
      XNUM2=1.d0/(PAR2*RPAR2)                                            
      OMZ=-rpole*(XNUM1+RMS*XNUM2)                                           
      OMX=RMS*XL*XNUM2-X*XNUM1+RM1S*X*RF-RMS/DSQ                         
      IF(KOMP.EQ.2) OMX=RMS*XL*XNUM2-X*XNUM1-RM1S*XL*RF+1.d0/DSQ         
      grpole=dsqrt(OMX*OMX+OMZ*OMZ)                               
c ******************************************** 
      call gabs(komp,smaxis,rmass,e,period,d,rpole,xmas,xmaso,absgr, 
     $glogg) 
      if(komp.eq.1) gplog1=glogg 
      if(komp.eq.2) gplog2=glogg 
      DO 8 I=1,N                                                         
      IF(I.NE.2) GOTO 82                                                 
      IF(KOMP.EQ.1) RTEST=.3d0*RV(1)                                     
      IF(KOMP.EQ.2) RTEST=.3d0*RVQ(1)                                    
   82 CONTINUE                                                           
      IPN1=I+N1*(KOMP-1)                                                 
      SINTH=SNTH(IPN1)                                                   
      XNU=CSTH(IPN1)                                                     
      XNUSQ=XNU**2                                                       
      EM=SINTH*EN*1.3d0                                                  
      XLUMP=1.d0-XNUSQ                                                   
      MM=EM+1.d0                                                         
      afac=rf*rm1*xlump 
      DO 8 J=1,MM                                                        
      KOUNT=0                                                            
      IS=IS+1                                                            
      ISX=ISX+1                                                          
      DELR=0.d0                                                          
      COSFI=CSFI(ISX)                                                    
      XMU=SNFI(ISX)*SINTH                                                
      XLAM=SINTH*COSFI                                                   
      bfac=xlam*d 
      efac=rm*xlam/dsq 
      R=RSAVE                                                            
      oldr=r
      knth=0
   14 R=R+DELR                                                           
      tolr=1.d-6*dabs(r)
      if(kount.lt.1) goto 170
      if(knth.gt.20) goto 170
      if(r.gt.0.d0.and.r.lt.tester) goto 170
      knth=knth+1
      delr=0.5d0*delr
      r=oldr
      goto 14
  170 continue
      KOUNT=KOUNT+1                                                      
      IF(KOUNT.LT.80) GOTO 70                                            
      KFLAG=1                                                            
      R=-1.d0                                                            
      GOTO 86                                                            
   70 continue
      RSQ=R*R                                                            
      rcube=r*rsq 
      PAR=DSQ-2.d0*XLAM*R*D+RSQ                                          
      RPAR=dsqrt(PAR)                                                    
      par32=par*rpar 
      par52=par*par32 
      OM=1.d0/R+RM*((1.d0/RPAR)-XLAM*R/DSQ)+RM1*.5d0*RSQ*XLUMP*RF        
      denom=RF*RM1*XLUMP*R-1.d0/RSQ-(RM*(R-XLAM*D))/par32-efac      
      domr=1.d0/denom 
      d2rdo2=-domr*(afac+2.d0/rcube-rm*(1.d0/par32-3.d0*(r-bfac)**2/
     $par52))/denom**2 
      DELR=(POT-OM)*DOMR+.5d0*(pot-om)**2*d2rdo2
      oldr=r
      ABDELR=dabs(DELR)                                                  
      IF(ABDELR.GT.tolr) GOTO 14                                     
      ABR=dabs(R)                                                        
      IF(R.GT.RTEST) GOTO 74                                             
      KFLAG=1                                                            
      R=-1.d0                                                            
      IF(KOMP.EQ.2) GOTO 98                                              
      GOTO 97                                                            
   74 IF(ABR.LT.TESTER) RSAVE=R                                          
      Z=R*XNU                                                            
      Y=COMP*R*XMU                                                       
      X2T=ABR*XLAM                                                       
      X=CMPD+COMP*X2T                                                    
      IF(KOMP.EQ.2) GOTO 62                                              
      IF(X.LT.X1) GOTO 65                                                
      KFLAG=1                                                            
      R=-1.d0                                                            
      GOTO 97                                                            
   62 IF(X2T.LT.X2) GOTO 65                                              
      KFLAG=1                                                            
      R=-1.d0                                                            
      GOTO 98                                                            
   65 SUMSQ=Y**2+Z**2                                                    
      PAR1=X**2+SUMSQ                                                    
      RPAR1=dsqrt(PAR1)                                                  
      XNUM1=1.d0/(PAR1*RPAR1)                                            
      XL=D-X                                                             
      PAR2=XL**2+SUMSQ                                                   
      RPAR2=dsqrt(PAR2)                                                  
      XNUM2=1.d0/(PAR2*RPAR2)                                            
      OMZ=-Z*(XNUM1+RMS*XNUM2)                                           
      OMY=Y*(RM1S*RF-XNUM1-RMS*XNUM2)                                    
      OMX=RMS*XL*XNUM2-X*XNUM1+RM1S*X*RF-RMS/DSQ                         
      IF(KOMP.EQ.2) OMX=RMS*XL*XNUM2-X*XNUM1-RM1S*XL*RF+1.d0/DSQ         
      GRMAG=dsqrt(OMX*OMX+OMY*OMY+OMZ*OMZ)                               
      grvrat=grmag/grpole 
      GRAV=grvrat**GREXP                                         
      A=COMP*XLAM*OMX                                                    
      B=COMP*XMU*OMY                                                     
      C=XNU*OMZ                                                          
      COSBET=-(A+B+C)/GRMAG                                              
      IF(COSBET.LT..7d0) COSBET=.7d0                                     
   86 IF(KOMP.EQ.2) GOTO 98                                              
   97 RV(IS)=R                                                           
      GRX(IS)=OMX                                                        
      GRY(IS)=OMY                                                        
      GRZ(IS)=OMZ                                                        
      GMAG1(IS)=dsqrt(OMX*OMX+OMY*OMY+OMZ*OMZ)                           
      glog1(is)=dlog10(grvrat*absgr) 
      FR1(IS)=1.d0                                                       
      GLUMP1(IS)=R*R*SINTH/COSBET                                        
      GRV1(IS)=GRAV                                                      
      XX1(IS)=X                                                          
      YY1(IS)=Y                                                          
      ZZ1(IS)=Z                                                          
      CSBT1(IS)=COSBET                                                   
      GOTO 8                                                             
   98 RVQ(IS)=R                                                          
      GRXQ(IS)=OMX                                                       
      GRYQ(IS)=OMY                                                       
      GRZQ(IS)=OMZ                                                       
      GMAG2(IS)=dsqrt(OMX*OMX+OMY*OMY+OMZ*OMZ)                           
      glog2(is)=dlog10(grvrat*absgr) 
      FR2(IS)=1.d0                                                       
      GLUMP2(IS)=R*R*SINTH/COSBET                                        
      GRV2(IS)=GRAV                                                      
      XX2(IS)=X                                                          
      YY2(IS)=Y                                                          
      ZZ2(IS)=Z                                                          
      CSBT2(IS)=COSBET                                                   
    8 CONTINUE                                                           
      if(e.ne.0.d0.or.ff.ne.1.d0) goto 53
      IF(KFLAG.EQ.0) GOTO 53                                             
      ISS=IS-1                                                           
      IF(KOMP.NE.1) GOTO 50                                              
      CALL RING(RMASS,POTENT,1,N,FR1,HLD,R1H,RLH)                        
      DO 55 I=1,ISS                                                      
      IPL=I+1                                                            
      IF(RV(I).GE.0.d0)GOTO 55                                           
      FR1(IPL)=FR1(IPL)+FR1(I)                                           
      FR1(I)=0.d0                                                        
   55 CONTINUE                                                           
   53 IF(KOMP.EQ.2) GOTO 54                                              
      IS=0                                                               
      DO 208 I=1,N                                                       
      IPN1=I+N1*(KOMP-1)                                                 
      EM=SNTH(IPN1)*EN*1.3d0                                             
      MM=EM+1.d0                                                         
      DO 208 J=1,MM                                                      
      IS=IS+1                                                            
      GLUMP1(IS)=FR1(IS)*GLUMP1(IS)                                      
  208 CONTINUE                                                           
      RETURN                                                             
   50 if(e.ne.0.d0.or.ff.ne.1.d0) goto 54
      CALL RING(RMASS,POTENT,2,N,FR2,HLD,R1C,RLC)                        
      DO 56 I=1,IS                                                       
      IPL=I+1                                                            
      IF(RVQ(I).GE.0.d0) GOTO 56                                         
      FR2(IPL)=FR2(IPL)+FR2(I)                                           
      FR2(I)=0.d0                                                        
   56 CONTINUE                                                           
   54 CONTINUE                                                           
      IS=0                                                               
      DO 108 I=1,N                                                       
      IPN1=I+N1*(KOMP-1)                                                 
      EM=SNTH(IPN1)*EN*1.3d0                                             
      MM=EM+1.d0                                                         
      DO 108 J=1,MM                                                      
      IS=IS+1                                                            
      GLUMP2(IS)=FR2(IS)*GLUMP2(IS)                                      
  108 CONTINUE                                                           
      RETURN                                                             
      END                                                                
