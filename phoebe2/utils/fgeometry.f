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
        
        
        subroutine cross_nx3_3(a,b,N,out)
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(in) N
Cf2py intent(out) out
        implicit real*8 (a-h,o-z)
        dimension a(N,3), b(3), out(N,3)
        
        do 13 i=1,N
          out(i,1) =  a(i,2)*b(3) - a(i,3)*b(2)
          out(i,2) = -a(i,1)*b(3) + a(i,3)*b(1)
          out(i,3) =  a(i,1)*b(2) - a(i,2)*b(1)
13      end do
        return
        end
        
        
        subroutine compute_sizes(t, N, sizes)
Cf2py intent(in) t
Cf2py intent(in) N
Cf2py intent(out) sizes
        implicit real*8 (a-h,o-z)
        dimension t(N,9), sizes(N)
        double precision a, b, c, k
        
        do 14 i=1,N
        a=sqrt((t(i,1)-t(i,4))**2+(t(i,2)-t(i,5))**2+(t(i,3)-t(i,6))**2)
        b=sqrt((t(i,1)-t(i,7))**2+(t(i,2)-t(i,8))**2+(t(i,3)-t(i,9))**2)
        c=sqrt((t(i,4)-t(i,7))**2+(t(i,5)-t(i,8))**2+(t(i,6)-t(i,9))**2)
        k = 0.5 * (a+b+c)
        sizes(i) = sqrt(k*(k-a)*(k-b)*(k-c))
          
14      end do
        return
        end
        
