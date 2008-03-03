      subroutine gabs(komp,smaxis,qq,ecc,period,dd,rad,xm,xmo,absgr,
     $glog)
      implicit real*8(a-h,o-z)
c  Version of September 17, 2004
c
c  Input definitions:
c   smaxis is the length of the orbital semi-major axis in solar radii.
c   qq is the mass ratio in the sense m2/m1. Stars 1 and 2 are as defined
c     in the external program (star 1 is near superior conjunction at
c     phase zero).
c   ecc is orbital eccentricity
c   period is orbit period in days
c   dd is the instantaneous separation of the star centers in unit of th
c     orbital semi-major axis
c   rad is the polar radius if the star at issue in unit of the orbital
c     semi-major axis
c  Output definitions:
c   absgr is the polar acceleration due to effective gravity in cm/sec^2
c   glog is log_10 of absgr
c
      twopi=6.2831853072d0
      gbig=6.670d-8
      sunmas=1.989d33
      sunrad=6.9599d10
      psec=8.64d4*period
      acm=sunrad*smaxis
      pyears=period/365.2422d0
      aau=smaxis/214.9426d0
      tmass=aau**3/pyears**2
      qf=1.d0/(1.d0+qq)
      qfm=qq*qf
      sign=-1.d0
      if(komp.eq.2) goto 10
      qfm=qf
      qf=qq*qf
      sign=1.d0
   10 continue
      xm=tmass*qfm
      xmo=tmass*qf
      gbigm=gbig*xm*sunmas
      gbigmo=gbig*xmo*sunmas
      rcm=rad*acm
      dcm=dd*acm
      dcmsq=dcm*dcm
      efac=dsqrt((1.d0+ecc)*(1.d0-ecc))
      av=twopi*efac/(psec*dd*dd)
      avsq=av*av
      rcmsq=rcm*rcm
      hypsq=rcmsq+dcmsq
      hyp=dsqrt(hypsq)
      snalf=rcm/hyp
      csalf=dcm/hyp
      gz=-gbigm/rcmsq
      gzo=-snalf*gbigmo/hypsq
      gxo=sign*csalf*gbigmo/hypsq
      gxcf=-sign*avsq*dcm*qf
      gxs=gxo+gxcf
      gzs=gz+gzo
      absgr=dsqrt(gxs*gxs+gzs*gzs)
      glog=dlog10(absgr)
      return
      end
