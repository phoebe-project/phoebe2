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
      
      
        

      return
      end
      
