      subroutine fourls(th,ro,nobs,nth,aa,bb)                            
      implicit real*8(a-h,o-z)                                           
c   version of September 14, 1998                                        
c                                                                        
c    Input integer nth is the largest Fourier term fitted (e.g.          
c       for nth=6, terms up to sine & cosine of 6 theta are              
c       evaluated).                                                      
c    This subroutine can handle nth only up to 6. Additional             
c      programming is needed for larger values.                          
c                                                                        
      dimension aa(*),bb(*),th(*),ro(*),obs(50000),ll(14),mm(14),
     $cn(196),cl(14),out(14)                                             
      mpl=nth+1                                                          
      ml=mpl+nth                                                         
      jjmax=ml*ml                                                        
      nobsml=nobs*ml                                                     
      nobmpl=nobs*mpl                                                    
      do 90 i=1,nobs                                                     
      obs(i)=1.d0                                                        
      iz=nobsml+i                                                        
      obs(iz)=ro(i)                                                      
      if(nth.eq.0) goto 90                                               
      ic=i+nobs                                                          
      is=i+nobmpl                                                        
      sint=dsin(th(i))                                                   
      cost=dcos(th(i))                                                   
      obs(ic)=cost                                                       
      obs(is)=sint                                                       
      if(nth.eq.1) goto 90                                               
      ic=ic+nobs                                                         
      is=is+nobs                                                         
      sncs=sint*cost                                                     
      cs2=cost*cost                                                      
      obs(ic)=cs2+cs2-1.d0                                               
      obs(is)=sncs+sncs                                                  
      if(nth.eq.2) goto 90                                               
      ic=ic+nobs                                                         
      is=is+nobs                                                         
      sn3=sint*sint*sint                                                 
      cs3=cs2*cost                                                       
      obs(ic)=4.d0*cs3-3.d0*cost                                         
      obs(is)=3.d0*sint-4.d0*sn3                                         
      if(nth.eq.3) goto 90                                               
      ic=ic+nobs                                                         
      is=is+nobs                                                         
      cs4=cs2*cs2                                                        
      obs(ic)=8.d0*(cs4-cs2)+1.d0                                        
      obs(is)=4.d0*(2.d0*cs3*sint-sncs)                                  
      if(nth.eq.4) goto 90                                               
      ic=ic+nobs                                                         
      is=is+nobs                                                         
      cs5=cs3*cs2                                                        
      obs(ic)=16.d0*cs5-20.d0*cs3+5.d0*cost                              
      obs(is)=16.d0*sn3*sint*sint-20.d0*sn3+5.d0*sint                    
      if(nth.eq.5) goto 90                                               
      ic=ic+nobs                                                         
      is=is+nobs                                                         
      obs(ic)=32.d0*cs3*cs3-48.d0*cs4+18.d0*cs2-1.d0                     
      obs(is)=32.d0*sint*(cs5-cs3)+6.d0*sncs                             
   90 continue                                                           
      do 20 jj=1,jjmax                                                   
   20 cn(jj)=0.d0                                                        
      do 21 j=1,ml                                                       
   21 cl(j)=0.d0                                                         
      do 24 nob=1,nobs                                                   
      iii=nob+nobsml                                                     
      do 23 k=1,ml                                                       
      do 23 i=1,ml                                                       
      ii=nob+nobs*(i-1)                                                  
      kk=nob+nobs*(k-1)                                                  
      j=i+(k-1)*ml                                                       
   23 cn(j)=cn(j)+obs(ii)*obs(kk)                                        
      do 24 i=1,ml                                                       
      ii=nob+nobs*(i-1)                                                  
   24 cl(i)=cl(i)+obs(iii)*obs(ii)                                       
      call dminv(cn,ml,d,ll,mm)                                          
      call dgmprd(cn,cl,out,ml,ml,1)                                     
      do 51 i=2,mpl                                                      
      aa(i)=out(i)                                                       
      ipl=i+nth                                                          
   51 bb(i)=out(ipl)                                                     
      aa(1)=out(1)                                                       
      bb(1)=0.d0                                                         
      return                                                             
      end                                                                
