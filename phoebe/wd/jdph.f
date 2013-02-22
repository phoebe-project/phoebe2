      subroutine jdph(xjdin,phin,t0,p0,dpdt,xjdout,phout)
c  Version of February 2, 1999
c
c  Subroutine jdph computes a phase (phout) based on an input
c   JD (xjdin), reference epoch (t0), period (p0), and dP/dt (dpdt).
c   It also computes a JD (xjdout) from an input phase (phin) and the
c   same ephemeris. So jdph can be used either to get phase from
c   JD or JD from phase.
c
      implicit real*8(a-h,o-z)
      tol=1.d-6
      abdpdt=dabs(dpdt)
      deltop=(xjdin-t0)/p0
      fcsq=0.d0
      fccb=0.d0
      fc4th=0.d0
      fc5th=0.d0
      fc=deltop*dpdt
      if(dabs(fc).lt.1.d-18) goto 25
      fcsq=fc*fc
      if(dabs(fcsq).lt.1.d-24) goto 25
      fccb=fc*fcsq
      if(dabs(fccb).lt.1.d-27) goto 25
      fc4th=fc*fccb
      if(dabs(fc4th).lt.1.d-28) goto 25
      fc5th=fc*fc4th
   25 phout=deltop*(1.d0-.5d0*fc+fcsq/3.d0-.25d0*fccb+.2d0*fc4th-
     $fc5th/6.d0)
      pddph=dpdt*phin
      xjdout=p0*phin*(1.d0+.5d0*pddph+pddph**2/6.d0+pddph**3/24.d0
     $+pddph**4/120.d0+pddph**5/720.d0)+t0
      if(abdpdt.lt.tol) return
      phout=dlog(1.d0+deltop*dpdt)/dpdt
      xjdout=(dexp(dpdt*phin)-1.d0)*p0/dpdt+t0
      return
      end
