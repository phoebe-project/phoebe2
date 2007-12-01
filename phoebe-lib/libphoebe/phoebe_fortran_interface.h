#ifndef PHOEBE_FORTRAN_INTERFACE_H
	#define PHOEBE_FORTRAN_INTERFACE_H 1

#include "../libwd/f2c.h"
#include <string.h>

#include "phoebe_global.h"

#define wd_wrlci(fn,mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,jdphs,hjd0,period,dpdt,pshift,stddev,noise,seed,jdstrt,jdend,jdinc,phstrt,phend,phinc,phnorm,mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,the,vunit,e,sma,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,mzero,factor,wla,nsp1,xlat1,xlong1,radsp1,temsp1,nsp2,xlat2,xlong2,radsp2,temsp2) \
           wrlci_(fn,&mpage,&nref,&mref,&ifsmv1,&ifsmv2,&icor1,&icor2,&ld,&jdphs,&hjd0,&period,&dpdt,&pshift,&stddev,&noise,&seed,&jdstrt,&jdend,&jdinc,&phstrt,&phend,&phinc,&phnorm,&mode,&ipb,&ifat1,&ifat2,&n1,&n2,&perr0,&dperdt,&the,&vunit,&e,&sma,&f1,&f2,&vga,&xincl,&gr1,&gr2,&abunin,&tavh,&tavc,&alb1,&alb2,&phsv,&pcsv,&rm,&xbol1,&xbol2,&ybol1,&ybol2,&iband,&hla,&cla,&x1a,&x2a,&y1a,&y2a,&el3,&opsf,&mzero,&factor,&wla,&nsp1,xlat1,xlong1,radsp1,temsp1,&nsp2,xlat2,xlong2,radsp2,temsp2,strlen(fn))

#define wd_wrdci(fn,del,kep,ifder,ifm,ifr,xlamda,kspa,nspa,kspb,nspb,ifvc1,ifvc2,nlc,k0,kdisk,isym,nppl,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,jdphs,hjd0,period,dpdt,pshift,mode,ipb,ifat1,ifat2,n1,n2,n1l,n2l,perr0,dperdt,the,vunit,e,a,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,noise,sigma,wla,nsp1,xlat1,xlong1,radsp1,temsp1,nsp2,xlat2,xlong2,radsp2,temsp2,vertno,indep,dep,weight) \
           wrdci_(fn,del,kep,&ifder,&ifm,&ifr,&xlamda,&kspa,&nspa,&kspb,&nspb,&ifvc1,&ifvc2,&nlc,&k0,&kdisk,&isym,&nppl,&nref,&mref,&ifsmv1,&ifsmv2,&icor1,&icor2,&ld,&jdphs,&hjd0,&period,&dpdt,&pshift,&mode,&ipb,&ifat1,&ifat2,&n1,&n2,&n1l,&n2l,&perr0,&dperdt,&the,&vunit,&e,&a,&f1,&f2,&vga,&xincl,&gr1,&gr2,&abunin,&tavh,&tavc,&alb1,&alb2,&phsv,&pcsv,&rm,&xbol1,&xbol2,&ybol1,&ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,noise,sigma,wla,&nsp1,xlat1,xlong1,radsp1,temsp1,&nsp2,xlat2,xlong2,radsp2,temsp2,&vertno,indep,dep,weight,strlen(fn))

/**
 * WD_LCI_parameters:
 *
 * Input parameters for running the LC part of WD code. Here we depend on
 * f2c's types "integer" (which is typedeffed to long int) and "doublereal"
 * (which is typedeffed to double). The reason why we use these here are
 * the compatibility with any future changes in the f2c translator.
 */

typedef struct WD_LCI_parameters {
	integer    MPAGE;
	integer    NREF;
	integer    MREF;
	integer    IFSMV1;
	integer    IFSMV2;
	integer    ICOR1;
	integer    ICOR2;
	integer    LD;
	integer    JDPHS;
	doublereal HJD0;
	doublereal PERIOD;
	doublereal DPDT;
	doublereal PSHIFT;
	doublereal SIGMA;
	integer    WEIGHTING;
	doublereal SEED;
	doublereal HJDST;
	doublereal HJDSP;
	doublereal HJDIN;
	doublereal PHSTRT;
	doublereal PHSTOP;
	doublereal PHIN;
	doublereal PHNORM;
	integer    MODE;
	integer    IPB;
	integer    CALCHLA;
	integer    CALCVGA;
	integer    IFAT1;
	integer    IFAT2;
	integer    N1;
	integer    N2;
	doublereal PERR0;
	doublereal DPERDT;
	doublereal THE;
	doublereal VUNIT;
	doublereal E;
	doublereal SMA;
	doublereal F1;
	doublereal F2;
	doublereal VGA;
	doublereal INCL;
	doublereal GR1;
	doublereal GR2;
	doublereal LOGG1;
	doublereal LOGG2;
	doublereal MET1;
	doublereal MET2;
	doublereal TAVH;
	doublereal TAVC;
	doublereal ALB1;
	doublereal ALB2;
	doublereal PHSV;
	doublereal PCSV;
	doublereal RM;
	doublereal XBOL1;
	doublereal XBOL2;
	doublereal YBOL1;
	doublereal YBOL2;
	integer    IBAND;
	doublereal HLA;
	doublereal CLA;
	doublereal X1A;
	doublereal X2A;
	doublereal Y1A;
	doublereal Y2A;
	doublereal EL3;
	doublereal OPSF;
	doublereal MZERO;
	doublereal FACTOR;
	doublereal WLA;
	integer    SPRIM;
	doublereal *XLAT1;
	doublereal *XLONG1;
	doublereal *RADSP1;
	doublereal *TEMSP1;
	integer    SSEC;
	doublereal *XLAT2;
	doublereal *XLONG2;
	doublereal *RADSP2;
	doublereal *TEMSP2;
} WD_LCI_parameters;

/**
 * WD_DCI_parameters:
 *
 * Input parameters for running the DC part of WD code.
 */

typedef struct WD_DCI_parameters {
	integer    *tba;
	doublereal *step;
	doublereal dclambda;
	integer    nlc;
	integer    rv1data;
	integer    rv2data;
	integer    symder;
	integer    refswitch;
	integer    refno;
	integer    rv1proximity;
	integer    rv2proximity;
	integer    ldmodel;
	integer    indep;
	integer    morph;
	integer    cladec;
	integer    ifat1;
	integer    ifat2;
	integer    n1c;
	integer    n2c;
	integer    n1f;
	integer    n2f;
	doublereal hjd0;
	doublereal period;
	doublereal dpdt;
	doublereal pshift;
	doublereal perr0;
	doublereal dperdt;
	doublereal ecc;
	doublereal sma;
	doublereal f1;
	doublereal f2;
	doublereal vga;
	doublereal incl;
	doublereal grb1;
	doublereal grb2;
	doublereal met1;
	doublereal teff1;
	doublereal teff2;
	doublereal alb1;
	doublereal alb2;
	doublereal pot1;
	doublereal pot2;
	doublereal rm;
	doublereal xbol1;
	doublereal xbol2;
	doublereal ybol1;
	doublereal ybol2;
	integer    *passband;
	doublereal *wavelength;
	doublereal *sigma;
	doublereal *hla;
	doublereal *cla;
	doublereal *x1a;
	doublereal *y1a;
	doublereal *x2a;
	doublereal *y2a;
	doublereal *el3;
	doublereal *opsf;
	integer    *levweight;
	integer    spot1no;
	integer    spot2no;
	integer    spot1src;
	integer    spot2src;
	integer    spot1id;
	integer    spot2id;
	integer    spots1corotate;
	integer    spots2corotate;
	doublereal *spot1lat;
	doublereal *spot1long;
	doublereal *spot1rad;
	doublereal *spot1temp;
	doublereal *spot2lat;
	doublereal *spot2long;
	doublereal *spot2rad;
	doublereal *spot2temp;
	PHOEBE_curve **obs;
} WD_DCI_parameters;

int create_lci_file (char *filename, WD_LCI_parameters *param);
int create_dci_file (char *filename, void *pars);

int wd_lci_parameters_get (WD_LCI_parameters *params, int MPAGE, int curve);
int wd_spots_parameters_get ();

WD_DCI_parameters *wd_dci_parameters_new     ();
int                read_in_wd_dci_parameters (WD_DCI_parameters *params, int *marked_tba);
int                wd_dci_parameters_free    (WD_DCI_parameters *params);

#endif
