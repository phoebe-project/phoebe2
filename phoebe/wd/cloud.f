      subroutine cloud(cosa,cosb,cosg,x1,y1,z1,xc,yc,zc,rr,wl,op1,
     $opsf,edens,acm,en,cmpd,ri,dx,dens,tau)
c  Version of January 9, 2002
      implicit real*8 (a-h,o-z)
      dx=0.d0
      tau=0.d0
      ri=1.d0
      sige=.6653d-24
      dtdxes=sige*edens
c  cosa can be zero, so an alternate path to the solution is needed
      dabcoa=dabs(cosa)
      dabcob=dabs(cosb)
      if(dabcoa.lt.dabcob) goto 32
      w=cosb/cosa
      v=cosg/cosa
      u=y1-yc-w*x1
      t=z1-zc-v*x1
      aa=1.d0+w*w+v*v
      bb=2.d0*(w*u+v*t-xc)
      cc=xc*xc+u*u+t*t-rr*rr
      dubaa=aa+aa
      dis=bb*bb-4.d0*aa*cc
      if(dis.le.0.d0) return
      sqd=dsqrt(dis)
      xx1=(-bb+sqd)/dubaa
      xx2=(-bb-sqd)/dubaa
      yy1=w*(xx1-x1)+y1
      yy2=w*(xx2-x1)+y1
      zz1=v*(xx1-x1)+z1
      zz2=v*(xx2-x1)+z1
      goto 39
   32 w=cosa/cosb
      v=cosg/cosb
      u=x1-xc-w*y1
      t=z1-zc-v*y1
      aa=1.d0+w*w+v*v
      bb=2.d0*(w*u+v*t-yc)
      cc=yc*yc+u*u+t*t-rr*rr
      dubaa=aa+aa
      dis=bb*bb-4.d0*aa*cc
      if(dis.le.0.d0) return
      sqd=dsqrt(dis)
      yy1=(-bb+sqd)/dubaa
      yy2=(-bb-sqd)/dubaa
      xx1=w*(yy1-y1)+x1
      xx2=w*(yy2-y1)+x1
      zz1=v*(yy1-y1)+z1
      zz2=v*(yy2-y1)+z1
   39 dis=bb*bb-4.d0*aa*cc
      if(dis.le.0.d0) return
      sqd=dsqrt(dis)
      xs1=(xx1-cmpd)*cosa+yy1*cosb+zz1*cosg
      xs2=(xx2-cmpd)*cosa+yy2*cosb+zz2*cosg
      xxnear=xx1
      yynear=yy1
      zznear=zz1
      xxfar=xx2
      yyfar=yy2
      zzfar=zz2
      xsnear=xs1
      xsfar=xs2
      if(xs1.gt.xs2) goto 38
      xxnear=xx2
      yynear=yy2
      zznear=zz2
      xxfar=xx1
      yyfar=yy1
      zzfar=zz1
      xsnear=xs2
      xsfar=xs1
   38 continue
      xss=(x1-cmpd)*cosa+y1*cosb+z1*cosg
      if(xss.ge.xsnear) return
      if(xss.le.xsfar) goto 20
      xxfar=x1
      yyfar=y1
      zzfar=z1
   20 continue
      dtaudx=dtdxes+(op1*wl**en+opsf)*dens
      dx=dsqrt((xxnear-xxfar)**2+(yynear-yyfar)**2+(zznear-zzfar)**2)
      tau=dx*dtaudx*acm
      ri=dexp(-tau)
      return
      end
