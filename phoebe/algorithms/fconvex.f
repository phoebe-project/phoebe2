        subroutine convex_hull(points, myhull, lindex, iindex, N)
Cf2py intent(in) points
Cf2py intent(out) myhull
Cf2py intent(out) lindex
Cf2py intent(out) iindex
        real*8 points, ipoints
        real*8 myhull, imyhull
        integer N,i,j,lindex,iindex,jindex,kindex,inv_index
        integer turn
        dimension points(N,2)
        dimension myhull(N,2)
        dimension ipoints(N,2)
        dimension imyhull(N,2)
        
        iindex = 1
        do 40, i=1,N
C         Invert points for next run
          ipoints(N-i+1,1) = points(i,1)
          ipoints(N-i+1,2) = points(i,2)
C         This run
          if (iindex.GT.2) then
            jindex = iindex
            do 41, j=1, jindex
              inv_index = jindex-j+1
              if (turn(myhull(inv_index-2,1), myhull(inv_index-2,2),
     +                myhull(inv_index-1,1), myhull(inv_index-1,2),
     +                points(i,1), points(i,2)).LT.1) then
                iindex = iindex - 1
                if (iindex.LE.2) then
                    EXIT
                end if
              else
                EXIT
              end if
41          end do
          end if
          if ((iindex.EQ.1).OR.((myhull(iindex-1,1).NE.points(i,1)).OR.
     +                      (myhull(iindex-1,2).NE.points(i,2)))) then
            myhull(iindex,1) = points(i,1)
            myhull(iindex,2) = points(i,2)
            iindex = iindex + 1
          end if              
40      end do
        
C       Second run        
        kindex = 1
        do 42, i=1,N
C         This run
          if (kindex.GT.2) then
            jindex = kindex
            do 43, j=1, jindex
              inv_index = jindex-j+1
              if (turn(imyhull(inv_index-2,1), imyhull(inv_index-2,2),
     +                imyhull(inv_index-1,1), imyhull(inv_index-1,2),
     +                ipoints(i,1), ipoints(i,2)).LT.1) then
                kindex = kindex - 1
                if (kindex.LE.2) then
                    EXIT
                end if
              else
                EXIT
              end if
43          end do
          end if
         if ((kindex.EQ.1).OR.((imyhull(kindex-1,1).NE.ipoints(i,1)).OR.
     +                      (imyhull(kindex-1,2).NE.ipoints(i,2)))) then
            imyhull(kindex,1) = ipoints(i,1)
            imyhull(kindex,2) = ipoints(i,2)
            kindex = kindex + 1
          end if              
42      end do

        
C       attach the second run results to the first run results
        do 44, i=1,kindex
          myhull(iindex+i-2,1) = imyhull(i,1)
          myhull(iindex+i-2,2) = imyhull(i,2)
44      end do
        lindex = iindex + kindex -3
        iindex = iindex - 1
        return
        end
        
        
        
        integer function turn(p1, p2, q1, q2, r1, r2)
        real*8 p1, p2, q1, q2, r1, r2
        real*8 myturn
        
        myturn = (q1-p1)*(r2-p2) - (r1-p1)*(q2-p2)
        
        if (myturn.LT.0d0) then
            turn = -1
        elseif (myturn.GT.0d0) then
            turn = +1
        else
            turn = 0
        endif
        return
        end function turn



