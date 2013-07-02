      subroutine temperature_zeipel(logg,tsize,teff,teff_type,
     +    beta,gp,N,tpole,teff_out)
Cf2py intent(in) logg
Cf2py intent(in) tsize
Cf2py intent(in) teff
Cf2py intent(in) teff_type
Cf2py intent(in) beta
Cf2py intent(in) gp
Cf2py intent(out) tpole
Cf2py intent(out) teff_out
C     Evaluate local temperature according to von Zeipel's law.
C     
C     If teff_type==0, teff will be interpreted as mean temperature
C     If teff_type==1, teff will be interpreted as polar temperature
C     
      implicit real*8 (a-h,o-z)
      dimension logg(N), tsize(N), teff_out(N), grav(N)
      integer teff_type
      double precision teff, logg, tsize, teff_out, beta, gp
      double precision tpole, tot_size, weight_size
      
C     Initialize mean or polar temperature        
      if (teff_type.EQ.0) then
          tpole = 0.0
      elseif (teff_type.EQ.1) then
          tpole = teff
      endif

C     Compute local gravity and compute mean temperature if needed    
      tot_size = 0.0d0
      weight_size = 0.0d0
      
      do 11 i=1,N
        grav(i) = (10.0d0**(logg(i)-2.0d0)/gp)**beta        

        if (teff_type.EQ.0) then
          tot_size = tot_size + tsize(i)
          weight_size = weight_size + grav(i)*tsize(i)
        elseif (teff_type.EQ.1) then
          teff_out(i) = grav(i)**0.25d0 * tpole
        endif

11    end do

C     Possibly polar temperature needs to be derived from mean teff        
      if (teff_type.EQ.0) then 
        tpole = teff * (tot_size / weight_size)**0.25d0
        
        do 12 i=1,N
          teff_out(i) = grav(i)**0.25d0 * tpole
12      end do          
      endif

C     Otherwise we're done
      
      RETURN
      END
            




      subroutine funczero(var_theta,cr,theta,omega,sol)
Cf2py intent(in) var_theta
Cf2py intent(in) cr
Cf2py intent(in) theta
Cf2py intent(in) omega
Cf2py intent(out) sol
C     Evaluate the function to solve for zeros in Espinosa's law
      implicit real*8 (a-h,o-z)
      
      term1 = cos(var_theta) + log(tan(0.5d0*var_theta))
      term2 = - 1.0d0/3.0d0 * omega**2 * cr**3 * cos(theta)**3
      term3 = - cos(theta) - log(tan(0.5*theta))
      
      sol = term1 + term2 + term3
      RETURN
      END
      
      
      subroutine dfunczero(var_theta,cr,omega,sol)
Cf2py intent(in) var_theta
Cf2py intent(in) cr
Cf2py intent(in) theta
Cf2py intent(in) omega
Cf2py intent(out) sol
C     Evaluate the derivative function to solve for zeros in
C     Espinosa's law
      implicit real*8 (a-h,o-z)
      
      finv = tan(var_theta*0.5d0) * cos(var_theta*0.5d0)**2
      sol = -sin(var_theta) + 0.5d0 / finv
      
      RETURN
      END      
      
      
      subroutine espinosa(cr,theta,omega,N,var_theta)
Cf2py intent(in) cr
Cf2py intent(in) theta
Cf2py intent(in) omega
Cf2py intent(out) var_theta
C     Evaluate local temperature according to von Espinosa's law.
      implicit real*8 (a-h,o-z)
      dimension cr(N), theta(N), var_theta(N)
      double precision fx0, dfx
      
      do 13 i=1,N
        x0 = theta(i)
        call funczero(x0,cr(i),theta(i),omega,fx0)
        call dfunczero(x0,cr(i),omega,dfx)
        xn = x0 - fx0 / dfx
        
    
        
        j = 0
        
        
14      if ((abs(fx0).GT.1d-10).AND.(j.LT.100)) then
            call funczero(xn,cr(i),theta(i),omega,fx0)
            call dfunczero(xn,cr(i),omega,dfx)
            xn = xn - fx0 / dfx  
            j = j+1
            goto 14
        endif
        
        if (xn.NE.xn) then
            xn = 1.5707963267948966d0
        endif
        
        var_theta(i) = xn
        
        
        
13    end do      
      RETURN
      END