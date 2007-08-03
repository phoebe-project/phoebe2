#ifndef PHOEBE_FORTRAN_INTERFACE_H
	#define PHOEBE_FORTRAN_INTERFACE_H 1

#include <string.h>

#include "phoebe_global.h"

#define wd_wrlci(fn,mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,jdphs,hjd0,period,dpdt,pshift,stddev,noise,seed,jdstrt,jdend,jdinc,phstrt,phend,phinc,phnorm,mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,the,vunit,e,sma,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,mzero,factor,wla,nsp1,xlat1,xlong1,radsp1,temsp1,nsp2,xlat2,xlong2,radsp2,temsp2) \
           wrlci_(fn,&mpage,&nref,&mref,&ifsmv1,&ifsmv2,&icor1,&icor2,&ld,&jdphs,&hjd0,&period,&dpdt,&pshift,&stddev,&noise,&seed,&jdstrt,&jdend,&jdinc,&phstrt,&phend,&phinc,&phnorm,&mode,&ipb,&ifat1,&ifat2,&n1,&n2,&perr0,&dperdt,&the,&vunit,&e,&sma,&f1,&f2,&vga,&xincl,&gr1,&gr2,&abunin,&tavh,&tavc,&alb1,&alb2,&phsv,&pcsv,&rm,&xbol1,&xbol2,&ybol1,&ybol2,&iband,&hla,&cla,&x1a,&x2a,&y1a,&y2a,&el3,&opsf,&mzero,&factor,&wla,&nsp1,xlat1,xlong1,radsp1,temsp1,&nsp2,xlat2,xlong2,radsp2,temsp2,strlen(fn))

#define wd_wrdci(fn,del,kep,ifder,ifm,ifr,xlamda,kspa,nspa,kspb,nspb,ifvc1,ifvc2,nlc,k0,kdisk,isym,nppl,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,jdphs,hjd0,period,dpdt,pshift,mode,ipb,ifat1,ifat2,n1,n2,n1l,n2l,perr0,dperdt,the,vunit,e,a,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,noise,sigma,wla,nsp1,xlat1,xlong1,radsp1,temsp1,nsp2,xlat2,xlong2,radsp2,temsp2,vertno,indep,dep,weight) \
           wrdci_(fn,del,kep,&ifder,&ifm,&ifr,&xlamda,&kspa,&nspa,&kspb,&nspb,&ifvc1,&ifvc2,&nlc,&k0,&kdisk,&isym,&nppl,&nref,&mref,&ifsmv1,&ifsmv2,&icor1,&icor2,&ld,&jdphs,&hjd0,&period,&dpdt,&pshift,&mode,&ipb,&ifat1,&ifat2,&n1,&n2,&n1l,&n2l,&perr0,&dperdt,&the,&vunit,&e,&a,&f1,&f2,&vga,&xincl,&gr1,&gr2,&abunin,&tavh,&tavc,&alb1,&alb2,&phsv,&pcsv,&rm,&xbol1,&xbol2,&ybol1,&ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,noise,sigma,wla,&nsp1,xlat1,xlong1,radsp1,temsp1,&nsp2,xlat2,xlong2,radsp2,temsp2,&vertno,indep,dep,weight,strlen(fn))

int create_lci_file (char *filename, WD_LCI_parameters *param);
int create_dci_file (char *filename, void *pars);

#endif
