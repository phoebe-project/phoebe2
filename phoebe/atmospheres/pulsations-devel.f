      Subroutine observables(radius,theta,phi,teff,
     & logg,t,l,m,freq,phases,spin,k,asl,delta_T,delta_g,
     & incls,phaseincls,mesh_phase,t0,dfdt,N,NF,
     & newradius,newtheta,newphi,
     & rvelo_r,rvelo_theta,rvelo_phi,newteff,newlogg)
     
Cf2py intent(in) radius,theta,phi,teff,logg,t,l,m,freq,phases,spin,k,asl,delta_T,delta_g,incls,phaseincls,mesh_phase,t0,dfdt,N,NF
Cf2py intent(out) newradius,newtheta,newphi,rvelo_r,rvelo_theta,rvelo_phi,newteff,newlogg
! NF=number of pulsation modes
      include 'constants.f'
      integer N,NF,l(NF),m(NF)
      real*8 radius(N),theta(N),thetacp(N),phi(N),phicp(N)
      real*8 newradius(N),newtheta(N),newphi(N)
      real*8 teff(N),logg(N),t,freq(NF),phases(NF)
      real*8 spin(NF),k(NF),t0,dfdt,tfreq
      real*8 asl(NF),incls(NF),phaseincls(NF)
      complex*16 delta_T(NF),delta_g(NF)
      real*8 mesh_phase
      real*8 newteff(N),newlogg(N)
      real*8 gravity(N),norm,inorm
      complex*16 ksi_grav(N),ksi_teff(N),temp
      complex*16 ksi_r(N),ksi_theta(N),ksi_phi(N)
      complex*16 velo_r(N),velo_theta(N),velo_phi(N)
      real*8 rvelo_r(N),rvelo_theta(N),rvelo_phi(N)
      complex*16 rad_part(N)
      complex*16 ksi_rcp(N),ksi_thetacp(N),ksi_phicp(N)

      norm=dsqrt(4.d0*pi)
      gravity=10.d0**(logg-2.d0)
      ksi_r=0.d0
      ksi_theta = 0.d0
      ksi_phi = 0.d0
      velo_r = 0.d0
      velo_theta = 0.d0
      velo_phi = 0.d0
      ksi_grav = 0.d0
      ksi_teff = 0.d0
      t = t-t0       

      phicp=phi-mesh_phase  

      Do i=1,NF
        tfreq=freq(i)+dfdt*t
        temp=j*TPI*tfreq
        thetacp=theta
        call radial(thetacp,phicp,l(i),m(i),tfreq,phases(i),
     &  t,phaseincls(i),incls(i),0.d0,N,rad_part)
        inorm=asl(i)*norm
        ksi_rcp=inorm*rad_part
        ksi_r=ksi_r+ksi_rcp*radius
        velo_r=velo_r+temp*ksi_rcp*radius
        if(l(i).gt.0)then
         call colatitudinal(thetacp,phicp,l(i),m(i),
     &   tfreq,phases(i),t,spin(i),k(i),phaseincls(i),
     &    incls(i),0.d0,N,ksi_thetacp)
         call longitudinal(thetacp,phicp,l(i),m(i),
     &    tfreq,phases(i),t,spin(i),k(i),phaseincls(i),
     &    incls(i),0.d0,N,ksi_phicp)  
     
         ksi_thetacp=inorm*ksi_thetacp
         ksi_phicp=inorm*ksi_phicp
         ksi_phi=ksi_phi+ksi_phicp
         velo_theta=velo_theta+temp*ksi_thetacp
         velo_phi=velo_phi+temp*ksi_phicp
        else
         ksi_theta = ksi_theta+0.d0
         ksi_phi =ksi_phi+0.d0
         velo_theta = velo_theta+ 0.d0
         velo_phi = velo_phi+0.d0           
        end if
        ksi_grav=ksi_grav+delta_g(i)*rad_part*gravity
        ksi_teff=ksi_teff+delta_T(i)*rad_part*teff

      End do
      newradius=radius+dreal(ksi_r)
      newtheta=theta+dreal(ksi_theta)
      newphi=phi+dreal(ksi_phi)
      rvelo_r=dreal(velo_r)
      rvelo_theta=dreal(velo_theta)
      rvelo_phi=dreal(velo_phi)       
      newteff=teff+dreal(ksi_teff)
      newlogg=dlog10(gravity+dreal(ksi_grav))+2.d0

      return
      end subroutine
!-----------------------------------------------------------------------------
      Subroutine surface(radius,theta,phi,t,l,m,freq,
     & phases,spin,k,asl,incls,phaseincls,mesh_phase,
     &  t0,dfdt,N,NF,newradius,newtheta,newphi,
     &   rvelo_r,rvelo_theta,rvelo_phi)
Cf2py intent(in) radius,theta,phi,t,l,m,freq,phases,spin,k,asl,incls,phaseincls,mesh_phase,t0,dfdt,N,NF
Cf2py intent(out) newradius,newtheta,newphi,rvelo_r,rvelo_theta,rvelo_phi
! NF=number of pulsation modes
      include 'constants.f'
      integer N,NF,l(NF),m(NF)
      real*8 radius(N),theta(N),phi(N),phicp(N)
      real*8 newradius(N),newtheta(N),newphi(N)
      real*8 mesh_phase,thetacp(N)
      real*8 spin(NF),k(NF),freq(NF),phases(NF),tfreq
      real*8 asl(NF),incls(NF),phaseincls(NF)
      real*8 t,t0,dfdt
      complex*16 velo_r(N),velo_theta(N),velo_phi(N)
      complex*16 ksi_r(N),ksi_rcp(N),ksi_theta(N)
      complex*16 ksi_phi(N),ksi_thetacp(N),ksi_phicp(N)
      real*8 norm
      complex*16 temp
      real*8 rvelo_r(N),rvelo_theta(N),rvelo_phi(N) 
      
      norm=dsqrt(4.d0*pi)
      ksi_r = 0.d0
      ksi_theta = 0.d0
      ksi_phi = 0.
      velo_r = 0.d0
      velo_theta = 0.d0
      velo_phi = 0.d0
      t = t-t0
      phicp=phi-mesh_phase 
      
      Do i=1,NF
       tfreq=freq(i)+dfdt*t
       thetacp=theta
       call radial(thetacp,phicp,l(i),m(i),tfreq,phases(i),
     & t,phaseincls(i),incls(i),0.d0,N,ksi_rcp)
       ksi_rcp=asl(i)*radius*norm*ksi_rcp
       ksi_r=ksi_r+ksi_rcp
       velo_r=velo_r+j*TPI*tfreq*ksi_rcp
       if(l(i).gt.0)then
        call colatitudinal(thetacp,phicp,l(i),m(i),
     &   tfreq,phases(i),t,spin(i),k(i),phaseincls(i),
     &    incls(i),0.d0,N,ksi_thetacp)
        call longitudinal(thetacp,phicp,l(i),m(i),
     &    tfreq,phases(i),t,spin(i),k(i),phaseincls(i),
     &    incls(i),0.d0,N,ksi_phicp)

        ksi_thetacp=asl(i)*norm*ksi_thetacp
        ksi_phicp=asl(i)*norm*ksi_phicp
        ksi_theta=ksi_theta+ksi_thetacp
        ksi_phi=ksi_phi+ksi_phicp
        temp=j*TPI*tfreq
        velo_theta= velo_theta + temp*ksi_thetacp
        velo_phi  = velo_phi   + temp*ksi_phicp
       else
        ksi_theta = ksi_theta+0.d0
        ksi_phi =ksi_phi+0.d0
        velo_theta = velo_theta+ 0.d0
        velo_phi = velo_phi+0.d0    
       end if
      
      End do
      newradius=radius+dreal(ksi_r)
      newtheta=theta+dreal(ksi_theta)
      newphi=phi+dreal(ksi_phi)
      rvelo_r=dreal(velo_r)
      rvelo_theta=dreal(velo_theta)
      rvelo_phi=dreal(velo_phi)
      return
      end subroutine
!------------------------------------------------------------------------------
 ! function to compute associated legendre polynomials    
      subroutine newplgndr(l,m,x,N,plgndr)
      include 'constants.f' 
      INTEGER l,m,N
      REAL*8 plgndr(N),x(N)
      INTEGER i,ll
      REAL*8 fact,pll(N),pmm(N)
      real*8 pmmp1(N),somx2(N)

      if(m.lt.0.or.m.gt.l.or.(maxval(dabs(x)).gt.1.d0))pause
     *'bad arguments in plgndr'
      pmm=1.d0
      if(m.gt.0) then
        somx2=dsqrt((1.d0-x)*(1.d0+x))
        fact=1.d0
        do 11 i=1,m
          pmm=-pmm*fact*somx2
          fact=fact+2.d0
11      continue
      endif
      if(l.eq.m) then
        plgndr=pmm
      else
        pmmp1=dfloat((2*m+1))*x*pmm
        if(l.eq.m+1) then
          plgndr=pmmp1
        else
          do 12 ll=m+2,l
            pll=(x*(2*ll-1)*pmmp1-dfloat((ll+m-1))*pmm)/(ll-m)
            pmm=pmmp1
            pmmp1=pll
12        continue
          plgndr=pll
        endif
      endif

      return
      END subroutine
      

!----------------------------------------------      
!    # function to compute spherical harmonics
!----------------------------------------------  
      Subroutine ylm(l,m,phi,theta,N,Y)
Cf2py intent(in) N,l,m,phi,theta
Cf2py intent(out)Y
      include 'constants.f' 
      integer l,m,N,absm
      real*8 phi(N),theta(N),lfloat,mfloat
      real*8 plgndr(N)
      complex*16 Y(N)
      real*8 dabsm,norm,factrl,arg(N)


      lfloat=dfloat(l)
      mfloat=dfloat(m)
      absm=abs(m)
      dabsm=float(absm)
!      j=(0.d0,1.d0)

      if (abs(m).gt.l)then
       write(6,*)'ERROR: abs(m)>l'
       pause
      end if

      norm=0.5d0*dsqrt(((2.d0*lfloat+1.d0)/pi)*
     &  (factrl(l-absm)/factrl(l+absm)))
      if(m.lt.0)norm=(-1.d0)**(mfloat)*norm
      arg=mfloat*phi

      call newplgndr(l,absm,dcos(theta),N,plgndr)
     
      Y=norm*plgndr*(dcos(arg)+j*dsin(arg))

      return
      end subroutine
!------------------------------------------------------------
      Subroutine sph_harm(theta,phi,l,m,alpha,beta,gamma,
     & N,sphar)
Cf2py intent(in) theta,phi,l,m,alpha,beta,gamma,N
Cf2py intent(out)sphar
      include 'constants.f' 
      integer l,m,N
      real*8 theta(N),phi(N),alpha,beta,gamma
      complex*16 Y(N),sphar(N),sphar_rot(N)

      
      if (abs(m).gt.l)then 
       sphar=0.d0
      else if((alpha.eq.0.d0).and.(beta.eq.0.d0).and.
     &  (gamma.eq.0.d0))then
       Call ylm(l,m,phi,theta,N,Y) 
       sphar=-Y

      else

       Call rotate_sph_harm(theta,phi,l,m,alpha,beta,gamma,
     &   N,sphar_rot)
       sphar=sphar_rot
            
      End if

      return
      end subroutine
!------------------------------------------------------------
      Function wignerd(ell, mu, m, 
     &    alpha, beta, gamma)
Cf2py intent(in) ell,mu,m,alpha,beta,gamma
Cf2py intent(out) wignerd
      include 'constants.f' 
      integer ell, mu, m, s_limit,s
      real*8 alpha,beta,gamma
      real*8 factor,factrl, small_d
      real*8 denom,num
      complex*16 wignerd
      integer pow1, pow2, pow3
      integer mum,ellmu,ellm
      real*8 arg
!      j=(0.d0,1.d0)

      factor = dsqrt(factrl(ell+mu) * factrl(ell-mu)
     &        * factrl(ell+m) * factrl(ell-m))
      
      mum=mu-m
      ellmu=ell-mu
      ellm=ell+m
      s_limit = max(mum, ellmu, ellm)
      small_d = 0.d0
      do s=0,s_limit
       if (((mum+s).ge.0).and.((ellmu-s).ge.0).and.
     &   ((ellm-s).ge.0))then
        denom = factrl(ellm-s) * factrl(s)
     &     *factrl(mum+s) * factrl(ellmu-s)
        pow1 = mum+s
        pow2 = 2*ell-mum-2*s
        pow3 = mum+2*s
        num = (-1.d0)**dfloat(pow1)*dcos(beta*0.5d0)**dfloat(pow2)
     &        * dsin(beta*0.5d0)**dfloat(pow3)
        small_d =small_d + num/denom
       end if
      
      End do
      small_d = factor*small_d 
      arg=dfloat(mu)*alpha+dfloat(m)*gamma
      wignerd=(dcos(arg)+j*dsin(arg))*small_d

      return
      end function
!------------------------------------------------------------------
      Subroutine rotate_sph_harm(theta,phi,l,m,alpha,beta,gamma,
     &   N,sphar_rot)
Cf2py intent(in) theta,phi,l,m,alpha,beta,gamma,N
Cf2py intent(out) sphar_rot
      include 'constants.f' 
      integer l,m,mu,N
      real*8 theta(N),phi(N),alpha,beta,gamma
      complex*16 sphar_rot(N),sphar(N),wignerd
      
      sphar_rot=0.d0
      Do mu=-l,l,1
       call sph_harm(theta,phi,l,mu,0.d0,0.d0,0.d0,N,sphar)
       sphar_rot=sphar_rot+sphar*wignerd(l,mu,m,alpha,beta,gamma)
      End do
                            
      Return 
      End subroutine
!------------------------------------------------------------------
      Function norm_n(l,m)
Cf2py intent(in) l,m
Cf2py intent(out) norm_n
      include 'constants.f' 
      integer l,m
      real*8 norm_n,factrl
      
      norm_n=dsqrt(((2.d0*l+1.d0)/(4.d0*pi))*
     & factrl(l-m)/factrl(l+m)) 
      
      return
      end function
!------------------------------------------------------------------
      Function norm_j(l,m)
Cf2py intent(in) l,m
Cf2py intent(out) norm_j      
      integer l,m
      real*8 norm_j
      norm_j=0.d0
      if(abs(m).lt.l)then
        norm_j = dsqrt((l**2.d0-m**2.d0)/
     &     (4.d0*l**2.d0-1.d0))
      end if
      return
      end function
!------------------------------------------------------      
      Function norm_atlm1(l,m,spin,k)
Cf2py intent(in) l,m,spin,k
Cf2py intent(out) norm_atlm1      
      integer l,m
      real*8 norm_atlm1,k,spin

      norm_atlm1 =spin*(l+abs(m))/l*2.d0/(2.d0*l+1.d0)*
     & (1.d0+(l+1.d0)*k)

      return
      end function   
!------------------------------------------------------      
      Function norm_atlp1(l,m,spin,k)
Cf2py intent(in) l,m,spin,k
Cf2py intent(out) norm_atlp1      
      integer l,m
      real*8 norm_atlp1,k,spin

      norm_atlp1 =spin*(l-abs(m)+1.d0)/(l+1.d0)*
     & 2.d0/(2.d0*l+1.d0)*(1.d0-l*k)

      return
      end function   
!------------------------------------------------------------------
      Subroutine radial(theta,phi,l,m,freq,phase,t,
     & alpha,beta,gamma,N,sph_radial)
Cf2py intent(in) theta,phi,l,m,freq,phase,t,alpha,beta,gamma
Cf2py intent(out) sph_radial 
      include 'constants.f' 
      integer l,m,N
      real*8 theta(N),phi(N),freq,phase,t
      real*8 alpha,beta,gamma
      complex*16 sph_radial(N)
      real*8 arg

      call sph_harm(theta,phi,l,m,alpha,beta,gamma,
     & N,sph_radial)
      arg=2.d0*pi*(freq*t+phase)
      sph_radial=sph_radial*
     & (dcos(arg)+j*dsin(arg))
      return
      end subroutine

!------------------------------------------------------------------
      subroutine rotate_theta(theta,phi,alpha,beta,N)
Cf2py intent(in) phi,alpha,beta,N
Cf2py intent(inout) theta
      include 'constants.f' 
      integer N
      real*8 theta(N),phi(N)
      real*8 alpha,beta

      if((alpha.ne.0.d0).or.(beta.ne.0.d0))then
       theta = dacos(dsin(theta)*dcos(phi+alpha)*
     &  dsin(-beta)+dcos(theta)*dcos(-beta))
      end if

      return
      end subroutine     
!------------------------------------------------------------------
      subroutine dsph_harm_dtheta(theta,phi,l,m,
     & alpha,beta,gamma,N,dspharm_dtheta)
Cf2py intent(in) theta,phi,l,m,alpha,beta,gamma,N 
Cf2py intent(out) dspharm_dtheta  
      include 'constants.f' 
      integer l,m,N
      real*8 theta(N),thetacp(N),phi(N)
      real*8 alpha,beta,gamma,factor(N),norm_j
      complex*16 dspharm_dtheta(N),sphar(N)

      if(abs(m).gt.l)then
       dspharm_dtheta=0.d0
      else
       thetacp=theta
       call rotate_theta(thetacp,phi,0.d0,0.d0,N) 
       factor=1.d0/dsin(thetacp)
       call sph_harm(theta,phi,l+1,m,
     &  alpha,beta,gamma,N,sphar)
       dspharm_dtheta=dfloat(l)*norm_j(l+1,m)*sphar
       call sph_harm(theta,phi,l-1,m,
     &  alpha,beta,gamma,N,sphar)
       dspharm_dtheta=factor*
     & (dspharm_dtheta-dfloat(l+1)*norm_j(l,m)*sphar)
      end if
      
      return
      end subroutine
!------------------------------------------------------------------ 
      subroutine dsph_harm_dphi(theta,phi,l,m,
     & alpha,beta,gamma,N,dspharm_dphi)
Cf2py intent(in) theta,phi,l,m,alpha,beta,gamma,N 
Cf2py intent(out) dspharm_dphi 
      include 'constants.f' 
      integer l,m,N
      real*8 theta(N),phi(N)
      real*8 alpha,beta,gamma
      complex*16 dspharm_dphi(N)
      call sph_harm(theta,phi,l,m,alpha,beta,gamma,
     & N,dspharm_dphi)
      dspharm_dphi=j*dfloat(m)*dspharm_dphi 
      return
      end subroutine
!------------------------------------------------------------------
      subroutine colatitudinal(theta,phi,l,m,freq,phase,
     & t,spin,k,alpha,beta,gamma,N,colat)
Cf2py intent(in) theta,phi,l,m,freq,phase,t,spin,k,alpha,beta,gamma,N
Cf2py intent(out) colat
      include 'constants.f' 
      integer l,m,N
      real*8 theta(N),thetacp(N),phi(N)
      real*8 alpha,beta,gamma,freq,phase,t,spin,k
      real*8 tempsin(N)
      real*8 norm_atlp1,norm_atlm1  
      complex*16 colat(N)
      complex*16 term1(N),term2(N),term3(N)
      complex*16 dspharm_dtheta(N),dspharm_dphi(N)
      real*8 arg
      real*8 cosarg,sinarg 
      

      thetacp=theta
      call rotate_theta(thetacp,phi,0.d0,0.d0,N)
      tempsin=dsin(thetacp)

      call dsph_harm_dtheta(theta,phi,l,m,
     & alpha,beta,gamma,N,dspharm_dtheta) 
      arg=TPI*(freq*t+phase)
! Optimization using Euler's formula: cdexp(j*2.d0*pi*(freq*t+phase))=(cosarg+j*sinarg) (with cosarg=dcos(arg) and sinarg=dsin(arg))
! Optimization using sine and cosine of sum formula: cdexp(j*2.d0*pi*(freq*t+phase+0.25d0))=(-sinarg+j*cosarg))  
! Optimization using sine and cosine of sum formula: cdexp(j*2.d0*pi*(freq*t+phase-0.25d0))=(sinarg-j*cosarg))  
      cosarg=dcos(arg)
      sinarg=dsin(arg)
      term1=k*dspharm_dtheta*
     &  (cosarg+j*sinarg)!cdexp(j*2.d0*pi*(freq*t+phase))
     
      call dsph_harm_dphi(theta,phi,l+1,m,
     & alpha,beta,gamma,N,dspharm_dphi)

      term2=norm_atlp1(l,m,spin,k) / tempsin*
     & dspharm_dphi*(-sinarg+j*cosarg)!cdexp(j*2.d0*pi*(freq*t+phase+0.25d0))
     
      call dsph_harm_dphi(theta,phi,l-1,m,
     & alpha,beta,gamma,N,dspharm_dphi)

      term3=norm_atlm1(l,m,spin,k) / tempsin*
     & dspharm_dphi*(sinarg-j*cosarg)!cdexp(j*2.d0*pi*(freq*t+phase-0.25d0))
      
      colat=term1+term2+term3   
!     
      return
      end subroutine           
!------------------------------------------------------------------
      subroutine longitudinal(theta,phi,l,m,freq,phase,
     & t,spin,k,alpha,beta,gamma,N,longit)

Cf2py intent(in) theta,phi,l,m,freq,phase,t,spin,k,alpha,beta,gamma,N
Cf2py intent(out) longit
      include 'constants.f' 
      integer l,m,N
      real*8 theta(N),thetacp(N),phi(N)
      real*8 alpha,beta,gamma,freq,phase,t,spin,k
      real*8 norm_atlp1,norm_atlm1  
      complex*16 longit(N)
      complex*16 term1(N),term2(N),term3(N)
      complex*16 dspharm_dtheta(N),dspharm_dphi(N)
      real*8 arg
      real*8 cosarg,sinarg 
      
    
      thetacp=theta
      call rotate_theta(thetacp,phi,0.d0,0.d0,N)


      call dsph_harm_dphi(theta,phi,l,m,
     & alpha,beta,gamma,N,dspharm_dphi) 
      arg=TPI*(freq*t+phase)
      cosarg=dcos(arg)
      sinarg=dsin(arg)
      term1=k/dsin(thetacp)*dspharm_dphi*
     &  (cosarg+j*sinarg)!cdexp(j*2.d0*pi*(freq*t+phase))

      call dsph_harm_dtheta(theta,phi,l+1,m,
     & alpha,beta,gamma,N,dspharm_dtheta)
      arg=TPI*(freq*t+phase+0.25d0)
      term2=-norm_atlp1(l,m,spin,k)*
     & dspharm_dtheta*(-sinarg+j*cosarg)!cdexp(j*2.d0*pi*(freq*t+phase+0.25d0))

      call dsph_harm_dtheta(theta,phi,l-1,m,
     & alpha,beta,gamma,N,dspharm_dtheta)
      arg=TPI*(freq*t+phase-0.25d0)
      term3=-norm_atlm1(l,m,spin,k)*
     & dspharm_dtheta*(sinarg-j*cosarg)!cdexp(j*2.d0*pi*(freq*t+phase-0.25d0))
      
      longit=term1+term2+term3         
      return
      end subroutine

!------------------------------------------------------------------
      FUNCTION factrl(n)
      INTEGER n
      REAL*8 factrl
CU    USES gammln
      INTEGER j,ntop
      REAL*8 a(33),gammln
      SAVE ntop,a
      DATA ntop,a(1)/0,1.d0/
      if (n.lt.0) then

        pause 'negative factorial in factrl'
      else if (n.le.ntop) then
        factrl=a(n+1)
      else if (n.le.32) then
        do 11 j=ntop+1,n
          a(j+1)=dfloat(j)*a(j)
11      continue
        ntop=n
        factrl=a(n+1)
      else
        factrl=dexp(gammln(dfloat(n)+1.d0))
      endif
      return
      END function
!-----------------------------------------------------------------
      FUNCTION gammln(xx)
      REAL*8 gammln,xx
      INTEGER j
      REAL*8 ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     *24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     *-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*dlog(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      gammln=tmp+dlog(stp*ser/x)
      return
      END function
!-----------------------------------------
