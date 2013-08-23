        real*8 function cos_angle(a,b)
C       Compute the cosine of the angle between two vectors
        real*8 a, b
        dimension a(3),b(3)
        double precision num, norm1, norm2
        
        num = (a(1)*b(1)+a(2)*b(2)+a(3)*b(3))
        norm1 = SQRT(a(1)**2 + a(2)**2 + a(3)**2)
        norm2 = SQRT(b(1)**2 + b(2)**2 + b(3)**2)
        cos_angle = num / (norm1 * norm2)
        return
        end function cos_angle
        
        
        real*8 function ld_claret(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_claret = 1d0 - coeffs(1)*(1d0-mu**0.5d0)
     +                  - coeffs(2)*(1d0-mu)
     +                  - coeffs(3)*(1d0-mu**1.5d0)
     +                  - coeffs(4)*(1d0-mu**2.0d0)
        return
        end
        
        real*8 function ld_linear(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_linear = 1d0 - coeffs(1)*(1d0-mu)
        return
        end

        real*8 function ld_nonlinear(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_nonlinear = 1d0 - coeffs(1)*(1d0-mu) - coeffs(2)*mu*LOG(mu)
        return
        end
        
        real*8 function ld_logarithmic(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_logarithmic = 1d0 - coeffs(1)*(1d0-mu) - coeffs(2)*mu*LOG(mu)
        return
        end
        
        real*8 function ld_quadratic(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_quadratic = 1d0 - coeffs(1)*(1d0-mu) - coeffs(2)*(1-mu)**2d0
        return
        end

        real*8 function ld_square_root(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_square_root = 1d0 - coeffs(1)*(1d0-mu) - coeffs(2)
     +                                              *(1-SQRT(mu))
        return
        end
        
        real*8 function ld_uniform(mu, coeffs)
        real*8 mu
        real*8 coeffs
        dimension coeffs(5)
        ld_uniform = 1d0
        return
        end
        
        
!         real*8 function disk_linear(mu, coeffs)
!         real*8 mu
!         real*8 coeffs
!         dimension coeffs(5)
!         real*8, parameter :: PI = 3.1415927
!         disk_linear = PI * (1d0 - coeffs(1)/3d0)
!         return
!         end
! 
!         real*8 function disk_nonlinear(mu, coeffs)
!         real*8 mu
!         real*8 coeffs
!         dimension coeffs(5)
!         real*8, parameter :: PI = 3.1415927
!         disk_nonlinear = PI*(1d0 - coeffs(1)/3 + 2d0/9d0*coeffs(2))
!         return
!         end
!         
!         real*8 function disk_logarithmic(mu, coeffs)
!         real*8 mu
!         real*8 coeffs
!         dimension coeffs(5)
!         real*8, parameter :: PI = 3.1415927
!         disk_logarithmic = PI*(1d0 - coeffs(1)/3 + 2d0/9d0*coeffs(2))
!         return
!         end
!         
!         real*8 function disk_quadratic(mu, coeffs)
!         real*8 mu
!         real*8 coeffs
!         dimension coeffs(5)
!         real*8, parameter :: PI = 3.1415927
!         disk_quadratic = PI*(1d0 - coeffs(1)/3d0
!      +                 - coeffs(2)/(0.5d0*2d0**2 + 1.5d0*2d0 + 1d0))
!         return
!         end
! 
!         real*8 function disk_square_root(mu, coeffs)
!         real*8 mu
!         real*8 coeffs
!         dimension coeffs(5)
!         real*8, parameter :: PI = 3.1415927
!         disk_square_root = PI*(1d0 - coeffs(1)/3d0 - coeffs(2)/5d0)
!         return
!         end
!         
!         real*8 function disk_uniform(mu, coeffs)
!         real*8 mu
!         real*8 coeffs
!         dimension coeffs(5)
!         real*8, parameter :: PI = 3.1415927
!         disk_uniform = PI
!         return
!         end

        
        subroutine reflection(irrorc, irrors, irrorn, irrorld,
     +    irredc, irreds, irredn, irrede,
     +    alb, redist, ldlaw, Nirror, Nirred, Nref, R1, R2, inco)
Cf2py intent(in) irrorc
Cf2py intent(in) irrors
Cf2py intent(in) irrorn
Cf2py intent(in) irrorld
Cf2py intent(in) irredc
Cf2py intent(in) irreds
Cf2py intent(in) irredn
Cf2py intent(in) irrede
Cf2py intent(in) alb
Cf2py intent(in) redist
Cf2py intent(in) ldlaw
Cf2py intent(out) R1
Cf2py intent(out) R2
Cf2py intent(out) inco
C       irrorc = irradiator centers
C       irrors = irradiator sizes
C       irrorn = irradiator normals
C       Nirror = length irradiator mesh
C       irredc = irradiated centers

C       irreds = irradiated sizes
C       irredn = irradiated normals
C       irrede = irradiated emergent fluxes
C       Nirred = length irradiated mesh
C       alb = bolometric albedo
C       redist = bolometric redistribution parameter
C       emer
C       Careful: R2 doesn't include the 1.0 term and is not divided by the total
C                surface (for generality). To get the "true" R2, you need to do:
C                R2 = 1.0 + R2 / total_surface

        
        integer i,j,k, ldlaw
        integer Nirror, Nirred, Nref
        real*8 distance2 ! Lines of sight squared distances
        real*8 alb, redist, R2, Imu0, Ibolmu
        real*8 los
        real*8 cos_psi1 ! Angles between normal and LOS on irred
        real*8 cos_psi2 ! Angles between normal and LOS on irror
        real*8 irrorc, irrors, irrorn, irredc, irreds, irredn
        real*8 irrede, R1, irrorld, inco, proj_Ibolmu
        real*8 cos_angle, ld_linear, ld_claret, ld_quadratic
        real*8 ld_square_root, ld_nonlinear, ld_uniform
        real*8 ld_logarithmic
        real*8 temp3
        real*8 temp5
        dimension los(3) ! Lines of sight vectors
        dimension irrorc(Nirror,3), irrors(Nirror), irrorn(Nirror,3)
        dimension irredc(Nirred,3), irreds(Nirred), irredn(Nirred,3)
        dimension irrede(Nirred), R1(Nirred)
        dimension irrorld(Nref,Nirror,5)
        dimension inco(Nirred,Nref)
        dimension proj_Ibolmu(Nref)
        dimension ldlaw(Nref)
        dimension temp3(3)
        dimension temp5(5)
        
        
        R2 = 0d0
        do 14 i=1,Nirred
          do 21 k=1,Nref
            proj_Ibolmu(k) = 0d0
21        end do
          
C         Run over all triangles from the irradiator
          do 15 j=1,Nirror
C           What are the lines of sight?
            los(1) = irrorc(j,1) - irredc(i,1)
            los(2) = irrorc(j,2) - irredc(i,2)
            los(3) = irrorc(j,3) - irredc(i,3)
C           What are the angles between the normal and the lines-of-sight
C           on the irradiated object?
            temp3(1) = irredn(i,1)
            temp3(2) = irredn(i,2)
            temp3(3) = irredn(i,3)
            cos_psi1 = cos_angle( temp3, los)
C           What are the angles between the normals and the lines-of-sight
C           on the irradiator?
            temp3(1) = irrorn(j,1)
            temp3(2) = irrorn(j,2)
            temp3(3) = irrorn(j,3)
            cos_psi2 = -cos_angle(los, temp3)
C           Figure out if this triangle receives any flux at all from the
C           the other triangle. If not, skip it
            if ((0d0 .GE. cos_psi1) .OR. (0d0 .GE. cos_psi2)) then
              cycle
            else
              distance2 = los(1)**2 + los(2)**2 + los(3)**2
              
C             Cycle over all passband dependables (lcs etc) and compute
C             the radiation budget
              do 16 k=1,Nref
                Imu0 = irrorld(k,j,5)                
                temp5(1) = irrorld(k,j,1)
                temp5(2) = irrorld(k,j,2)
                temp5(3) = irrorld(k,j,3)
                temp5(4) = irrorld(k,j,4)
                temp5(5) = irrorld(k,j,5)
C               0=claret, 1=linear, 2=nonlinear, 3=logarithm, 4=quadratic
C               5=squareroot, 6=uniform
                if (ldlaw(k).EQ.0) then
                    Ibolmu = Imu0*ld_claret(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.1) then
                    Ibolmu = Imu0*ld_linear(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.2) then
                    Ibolmu = Imu0*ld_nonlinear(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.3) then
                    Ibolmu = Imu0*ld_logarithmic(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.4) then
                    Ibolmu = Imu0*ld_quadratic(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.5) then
                    Ibolmu = Imu0*ld_square_root(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.6) then
                    Ibolmu = Imu0*ld_uniform(cos_psi2, temp5)
                endif
                
                Ibolmu = Ibolmu*irrors(j)*cos_psi2
                Ibolmu = Ibolmu*cos_psi1/distance2
                proj_Ibolmu(k) = proj_Ibolmu(k) + Ibolmu

16            end do
            endif
15        end do
          
C         Remember how much radiation enters in this triangle for this passband
          do 22 k=1,Nref
            inco(i,k) = proj_Ibolmu(k)
22        end do
                

C         Some more work for bolometric stuff                
          R1(i) = 1d0 + (1-redist)*alb*inco(i,1)/irrede(i)
          R2 = R2 + redist *alb*inco(i,1)/irrede(i)*irreds(i)
                
            
14      end do
        return
        end
        
        
        
        
        
        
        
        
        
        
        subroutine reflectionarray(irrorc, irrors, irrorn, irrorld,
     +    irredc, irreds, irredn, irrede,
     +    alb, redist, ldlaw, Nirror, Nirred, Nref, R1, R2, inco)
Cf2py intent(in) irrorc
Cf2py intent(in) irrors
Cf2py intent(in) irrorn
Cf2py intent(in) irrorld
Cf2py intent(in) irredc
Cf2py intent(in) irreds
Cf2py intent(in) irredn
Cf2py intent(in) irrede
Cf2py intent(in) alb
Cf2py intent(in) redist
Cf2py intent(in) ldlaw
Cf2py intent(out) R1
Cf2py intent(out) R2
Cf2py intent(out) inco
C       irrorc = irradiator centers
C       irrors = irradiator sizes
C       irrorn = irradiator normals
C       Nirror = length irradiator mesh
C       irredc = irradiated centers

C       irreds = irradiated sizes
C       irredn = irradiated normals
C       irrede = irradiated emergent fluxes
C       Nirred = length irradiated mesh
C       alb = bolometric albedo
C       redist = bolometric redistribution parameter
C       emer
C       Careful: R2 doesn't include the 1.0 term and is not divided by the total
C                surface (for generality). To get the "true" R2, you need to do:
C                R2 = 1.0 + R2 / total_surface

        
        integer i,j,k, ldlaw
        integer Nirror, Nirred, Nref
        real*8 distance2 ! Lines of sight squared distances
        real*8 alb, redist, R2, Imu0, Ibolmu
        real*8 los
        real*8 cos_psi1 ! Angles between normal and LOS on irred
        real*8 cos_psi2 ! Angles between normal and LOS on irror
        real*8 irrorc, irrors, irrorn, irredc, irreds, irredn
        real*8 irrede, R1, irrorld, inco, proj_Ibolmu
        real*8 cos_angle, ld_linear, ld_claret, ld_quadratic
        real*8 ld_square_root, ld_nonlinear, ld_uniform
        real*8 ld_logarithmic
        real*8 temp3
        real*8 temp5
        dimension los(3) ! Lines of sight vectors
        dimension irrorc(Nirror,3), irrors(Nirror), irrorn(Nirror,3)
        dimension irredc(Nirred,3), irreds(Nirred), irredn(Nirred,3)
        dimension irrede(Nirred), R1(Nirred)
        dimension irrorld(Nref,Nirror,5)
        dimension inco(Nirred,Nref)
        dimension proj_Ibolmu(Nref)
        dimension alb(Nirred), redist(Nirred)
        dimension ldlaw(Nref)
        dimension temp3(3)
        dimension temp5(5)
        
        
        R2 = 0d0
        do 34 i=1,Nirred
          do 41 k=1,Nref
            proj_Ibolmu(k) = 0d0
41        end do
          
C         Run over all triangles from the irradiator
          do 35 j=1,Nirror
C           What are the lines of sight?
            los(1) = irrorc(j,1) - irredc(i,1)
            los(2) = irrorc(j,2) - irredc(i,2)
            los(3) = irrorc(j,3) - irredc(i,3)
C           What are the angles between the normal and the lines-of-sight
C           on the irradiated object?
            temp3(1) = irredn(i,1)
            temp3(2) = irredn(i,2)
            temp3(3) = irredn(i,3)
            cos_psi1 = cos_angle( temp3, los)
C           What are the angles between the normals and the lines-of-sight
C           on the irradiator?
            temp3(1) = irrorn(j,1)
            temp3(2) = irrorn(j,2)
            temp3(3) = irrorn(j,3)
            cos_psi2 = -cos_angle(los, temp3)
C           Figure out if this triangle receives any flux at all from the
C           the other triangle. If not, skip it
            if ((0d0 .GE. cos_psi1) .OR. (0d0 .GE. cos_psi2)) then
              cycle
            else
              distance2 = los(1)**2 + los(2)**2 + los(3)**2
              
C             Cycle over all passband dependables (lcs etc) and compute
C             the radiation budget
              do 36 k=1,Nref
                Imu0 = irrorld(k,j,5)                
                temp5(1) = irrorld(k,j,1)
                temp5(2) = irrorld(k,j,2)
                temp5(3) = irrorld(k,j,3)
                temp5(4) = irrorld(k,j,4)
                temp5(5) = irrorld(k,j,5)
C               0=claret, 1=linear, 2=nonlinear, 3=logarithm, 4=quadratic
C               5=squareroot, 6=uniform
                if (ldlaw(k).EQ.0) then
                    Ibolmu = Imu0*ld_claret(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.1) then
                    Ibolmu = Imu0*ld_linear(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.2) then
                    Ibolmu = Imu0*ld_nonlinear(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.3) then
                    Ibolmu = Imu0*ld_logarithmic(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.4) then
                    Ibolmu = Imu0*ld_quadratic(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.5) then
                    Ibolmu = Imu0*ld_square_root(cos_psi2, temp5)
                elseif (ldlaw(k).EQ.6) then
                    Ibolmu = Imu0*ld_uniform(cos_psi2, temp5)
                endif
                
                Ibolmu = Ibolmu*irrors(j)*cos_psi2
                Ibolmu = Ibolmu*cos_psi1/distance2
                proj_Ibolmu(k) = proj_Ibolmu(k) + Ibolmu

36            end do
            endif
35        end do
          
C         Remember how much radiation enters in this triangle for this passband
          do 42 k=1,Nref
            inco(i,k) = proj_Ibolmu(k)
42        end do
                

C         Some more work for bolometric stuff                
          R1(i) = 1d0 + (1-redist(i))*alb(i)*inco(i,1)/irrede(i)
          R2 = R2 + redist(i) *alb(i)*inco(i,1)/irrede(i)*irreds(i)
                
            
34      end do
        return
        end        