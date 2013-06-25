        subroutine cos_angle_3_nx3(a,b,N,out)
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(in) N
Cf2py intent(out) out
C       Compute the cosine of the angle between two arrays of shape 3 and NX3
        implicit real*8 (a-h,o-z)
        dimension a(3),b(N,3),out(N)
        double precision num, norm1, norm2
        
        norm1 = SQRT(a(1)**2 + a(2)**2 + a(3)**2)
        do 11 i=1,N
          num = (a(1)*b(i,1)+a(2)*b(i,2)+a(3)*b(i,3))
          norm2 = SQRT(b(i,1)**2 + b(i,2)**2 + b(i,3)**2)
          out(i) = num / (norm1 * norm2)
11      end do
        return
        end
        
        subroutine cos_angle_nx3_nx3(a,b,N,out)
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(in) N
Cf2py intent(out) out
        implicit real*8 (a-h,o-z)
        dimension a(N,3),b(N,3),out(N)
        double precision num, norm1, norm2
        
        do 12 i=1,N
          num = (a(i,1)*b(i,1)+a(i,2)*b(i,2)+a(i,3)*b(i,3))
          norm1 = SQRT(a(i,1)**2 + a(i,2)**2 + a(i,3)**2)
          norm2 = SQRT(b(i,1)**2 + b(i,2)**2 + b(i,3)**2)
          out(i) = num / (norm1 * norm2)
12      end do
        return
        end