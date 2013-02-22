      subroutine linpro(komp,dvks,hbarw,tau,emm,count,taug,emmg,fbin,
     $delv)
c  Version of November 3, 2000
      implicit real*8(a-h,o-z)
      dimension dvks(*),hbarw(*),tau(*),emm(*),count(*),fbin(*),delv(*),
     $taug(*),emmg(*)
      common /flpro/ vks,binc,binw,difp,dum1,dum2
      common /ipro/ nbins,nl,inmax,inmin,idum1,idum2
      COMMON /SPOTS/ SINLAT(2,100),COSLAT(2,100),SINLNG(2,100),COSLNG
     $(2,100),RAD(2,100),TEMSP(2,100),xlng(2,100),kks(2,100),
     $Lspot(2,100)
      inmin=300000
      inmax=0
c
c  The 83 loop pre-computes the limiting bin numbers, encompassing all lines
c
      do 83 iln=1,nl
      if(Lspot(komp,iln).eq.0) goto 83
      vksg=vks+dvks(iln)
      vksp=vksg+hbarw(iln)
      vksm=vksg-hbarw(iln)
      indp=vksp/binw+binc
      indm=vksm/binw+binc
      if(indm.lt.inmin) inmin=indm
      if(indp.gt.inmax) inmax=indp
   83 continue
      do 82 i=inmin,inmax
      emmg(i)=0.d0
   82 taug(i)=0.d0
c
c  The 84 loop puts fractional contributions into the two end bins
c    (first part, up to 28 continue) and puts full contributions
c    into the middle bins (the 26 loop).
c
      do 84 iln=1,nl
      if(Lspot(komp,iln).eq.0) goto 84
      vksg=vks+dvks(iln)
      vksp=vksg+hbarw(iln)
      vksm=vksg-hbarw(iln)
      indp=vksp/binw+binc
      indm=vksm/binw+binc
      vks1=(dfloat(indm+1)-binc)*binw
      vks2=(dfloat(indp)-binc)*binw
      fr1=(vks1-vksm)/binw
      fr2=(vksp-vks2)/binw
      taug(indm)=taug(indm)+fr1*tau(iln)
      emmg(indm)=emmg(indm)+fr1*emm(iln)
      delv(indm)=delv(indm)+fr1*vksm
      count(indm)=count(indm)+fr1
      taug(indp)=taug(indp)+fr2*tau(iln)
      emmg(indp)=emmg(indp)+fr2*emm(iln)
      delv(indp)=delv(indp)+fr2*vksp
      count(indp)=count(indp)+fr2
      if(indp.ne.indm) goto 28
      taug(indp)=taug(indp)-tau(iln)
      emmg(indp)=emmg(indp)-emm(iln)
      delv(indp)=delv(indp)-.5d0*(vksm+vksp)
      count(indp)=count(indp)-1.d0
   28 continue
      ind=indm
      idmax=indp-indm-1
      if(idmax.le.0) goto 84
      do 26 id=1,idmax
      ind=ind+1
      vksb=(dfloat(ind)-binc)*binw
      taug(ind)=taug(ind)+tau(iln)
      emmg(ind)=emmg(ind)+emm(iln)
      delv(ind)=delv(ind)+vksb
      count(ind)=count(ind)+1.d0
   26 continue
   84 continue
c
c  The 85 loop collects the absorption and emission contributions to all
c     active bins, with the absorption lines summed via an optical thickness
c     treatment and the emission lines summed directly according to contributed
c     flux. The sign on emmg is negative because emmg is negative.
c
      do 85 i=inmin,inmax
   85 fbin(i)=fbin(i)+(1.d0-dexp(-taug(i))+emmg(i))*difp
      return
      end
