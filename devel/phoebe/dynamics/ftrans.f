! FORTRAN subroutine for coordinate transformation using Euler angles

      subroutine trans(coord,euler,loc,N)
Cf2py intent(in,out) coord   
Cf2py intent(in) euler  
Cf2py intent(in) loc
Cf2py intent(in) N

! Declarations
! Number of coordinate vectors
      integer N
! Array with coordinates (used for the original and the transformed coordinates)      
      real*8 coord(N,3)
! Euler angles and translation vector      
      real*8 euler(3),loc(3)
! Temporary variables for Euler angles      
      real*8 incl,longan,theta
! Temporary variables      
      real*8 c1,c2,c3,s1,s2,s3,c1s3,c1c3
      real*8 x,y,z,x0,y0,z0
! Store Euler angles into temporary variables      
      theta=euler(1)
      longan=euler(2)
      incl=euler(3)
! Store results of goniometric functions and repeated operations, so they need to be computed only once.   
      s1 = dsin(incl)
      c1 = dcos(incl)
      s2 = dsin(longan)
      c2 = dcos(longan)
      s3 = dsin(theta)
      c3 = dcos(theta)
      c1s3=c1*s3
      c1c3=c1*c3
! Components of translation vector      
      x0=loc(1)
      y0=loc(2)
      z0=loc(3)
! Transformation loop      
      do i=1,N
! Temporary copy old coordinates, so the same array can be used for the transformed coordinates       
       x=coord(i,1)
       y=coord(i,2)
       z=coord(i,3)
       coord(i,1)=(-c2*c3+s2*c1s3)*x +(c2*s3+s2*c1c3)*y-s2*s1*z + x0
       coord(i,2)=(-s2*c3-c2*c1s3)*x +(s2*s3-c2*c1c3)*y+c2*s1*z + y0
       coord(i,3)=         (s1*s3)*x +        (s1*c3)*y +  c1*z + z0
      end do  
      end subroutine
