#ifndef PHOEBE_FORTRAN_INTERFACE_H
	#define PHOEBE_FORTRAN_INTERFACE_H 1

#include <string.h>

#include "phoebe_global.h"

#define wd_wrlci(fn,mpage,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,jdphs,hjd0,period,dpdt,pshift,stddev,noise,seed,jdstrt,jdend,jdinc,phstrt,phend,phinc,phnorm,mode,ipb,ifat1,ifat2,n1,n2,perr0,dperdt,the,vunit,e,sma,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,mzero,factor,wla,nsp1,xlat1,xlong1,radsp1,temsp1,nsp2,xlat2,xlong2,radsp2,temsp2) \
           wrlci_(fn,&mpage,&nref,&mref,&ifsmv1,&ifsmv2,&icor1,&icor2,&ld,&jdphs,&hjd0,&period,&dpdt,&pshift,&stddev,&noise,&seed,&jdstrt,&jdend,&jdinc,&phstrt,&phend,&phinc,&phnorm,&mode,&ipb,&ifat1,&ifat2,&n1,&n2,&perr0,&dperdt,&the,&vunit,&e,&sma,&f1,&f2,&vga,&xincl,&gr1,&gr2,&abunin,&tavh,&tavc,&alb1,&alb2,&phsv,&pcsv,&rm,&xbol1,&xbol2,&ybol1,&ybol2,&iband,&hla,&cla,&x1a,&x2a,&y1a,&y2a,&el3,&opsf,&mzero,&factor,&wla,&nsp1,xlat1,xlong1,radsp1,temsp1,&nsp2,xlat2,xlong2,radsp2,temsp2,strlen(fn))

#define wd_wrdci(fn,del,kep,ifder,ifm,ifr,xlamda,kspa,nspa,kspb,nspb,ifvc1,ifvc2,nlc,k0,kdisk,isym,nppl,nref,mref,ifsmv1,ifsmv2,icor1,icor2,ld,jdphs,hjd0,period,dpdt,pshift,mode,ipb,ifat1,ifat2,n1,n2,n1l,n2l,perr0,dperdt,the,vunit,e,a,f1,f2,vga,xincl,gr1,gr2,abunin,tavh,tavc,alb1,alb2,phsv,pcsv,rm,xbol1,xbol2,ybol1,ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,noise,sigma,wla,nsp1,xlat1,xlong1,radsp1,temsp1,nsp2,xlat2,xlong2,radsp2,temsp2,knobs,indep,dep,weight) \
           wrdci_(fn,del,kep,&ifder,&ifm,&ifr,&xlamda,&kspa,&nspa,&kspb,&nspb,&ifvc1,&ifvc2,&nlc,&k0,&kdisk,&isym,&nppl,&nref,&mref,&ifsmv1,&ifsmv2,&icor1,&icor2,&ld,&jdphs,&hjd0,&period,&dpdt,&pshift,&mode,&ipb,&ifat1,&ifat2,&n1,&n2,&n1l,&n2l,&perr0,&dperdt,&the,&vunit,&e,&a,&f1,&f2,&vga,&xincl,&gr1,&gr2,&abunin,&tavh,&tavc,&alb1,&alb2,&phsv,&pcsv,&rm,&xbol1,&xbol2,&ybol1,&ybol2,iband,hla,cla,x1a,x2a,y1a,y2a,el3,opsf,noise,sigma,wla,&nsp1,xlat1,xlong1,radsp1,temsp1,&nsp2,xlat2,xlong2,radsp2,temsp2,knobs,indep,dep,weight,strlen(fn))

typedef struct WD_LCI_parameters {
	int    MPAGE;
	int    NREF;
	int    MREF;
	int    IFSMV1;
	int    IFSMV2;
	int    ICOR1;
	int    ICOR2;
	int    LD;
	int    JDPHS;
	double HJD0;
	double PERIOD;
	double DPDT;
	double PSHIFT;
	double SIGMA;
	int    WEIGHTING;
	double SEED;
	double HJDST;
	double HJDSP;
	double HJDIN;
	double PHSTRT;
	double PHSTOP;
	double PHIN;
	double PHNORM;
	int    MODE;
	int    IPB;
	int    CALCHLA;
	int    CALCVGA;
	int    IFAT1;
	int    IFAT2;
	int    N1;
	int    N2;
	double PERR0;
	double DPERDT;
	double THE;
	double VUNIT;
	double E;
	double SMA;
	double F1;
	double F2;
	double VGA;
	double INCL;
	double GR1;
	double GR2;
	double LOGG1;
	double LOGG2;
	double MET1;
	double MET2;
	double TAVH;
	double TAVC;
	double ALB1;
	double ALB2;
	double PHSV;
	double PCSV;
	double RM;
	double XBOL1;
	double XBOL2;
	double YBOL1;
	double YBOL2;
	int    IBAND;
	double HLA;
	double CLA;
	double X1A;
	double X2A;
	double Y1A;
	double Y2A;
	double EL3;
	double OPSF;
	double MZERO;
	double FACTOR;
	double WLA;
	int    SPRIM;
	double *XLAT1;
	double *XLONG1;
	double *RADSP1;
	double *TEMSP1;
	int    SSEC;
	double *XLAT2;
	double *XLONG2;
	double *RADSP2;
	double *TEMSP2;
} WD_LCI_parameters;

/**
 * WD_DCI_parameters:
 *
 * Input parameters for running the DC part of WD code.
 */

typedef struct WD_DCI_parameters {
	int    *tba;
	double *step;
	double dclambda;
	int    nlc;
	int    rv1data;
	int    rv2data;
	int    symder;
	int    refswitch;
	int    refno;
	int    rv1proximity;
	int    rv2proximity;
	int    ldmodel;
	int    indep;
	int    morph;
	int    cladec;
	int    ifat1;
	int    ifat2;
	int    n1c;
	int    n2c;
	int    n1f;
	int    n2f;
	int    nph;
	double delph;
	double hjd0;
	double period;
	double dpdt;
	double pshift;
	double perr0;
	double dperdt;
	double ecc;
	double sma;
	double f1;
	double f2;
	double vga;
	double incl;
	double grb1;
	double grb2;
	double met1;
	double teff1;
	double teff2;
	double alb1;
	double alb2;
	double pot1;
	double pot2;
	double rm;
	double xbol1;
	double xbol2;
	double ybol1;
	double ybol2;
	int    *passband;
	double *wavelength;
	double *sigma;
	double *hla;
	double *cla;
	double *x1a;
	double *y1a;
	double *x2a;
	double *y2a;
	double *el3;
	double *opsf;
	double *extinction;
	int    *levweight;
	int    spot1no;
	int    spot2no;
	int    spot1src;
	int    spot2src;
	int    spot1id;
	int    spot2id;
	int    spots1corotate;
	int    spots2corotate;
	double *spot1lat;
	double *spot1long;
	double *spot1rad;
	double *spot1temp;
	double *spot2lat;
	double *spot2long;
	double *spot2rad;
	double *spot2temp;
	int    *knobs;
	double *indeps;
	double *fluxes;
	double *weights;
} WD_DCI_parameters;

int create_lci_file (char *filename, WD_LCI_parameters *param);
int create_dci_file (char *filename, void *pars);

int wd_lci_parameters_get (WD_LCI_parameters *params, int MPAGE, int curve);
int wd_spots_parameters_get ();

WD_DCI_parameters *wd_dci_parameters_new  ();
int                wd_dci_parameters_get  (WD_DCI_parameters *params, int *marked_tba);
int                wd_dci_parameters_free (WD_DCI_parameters *params);

int phoebe_wd_model (char *phoebe_model);

int intern_get_level_weighting_id (const char *type);

#endif
