#include <stdlib.h>
#include <stdio.h>

#include "phoebe_error_handling.h"
#include "phoebe_global.h"

#include "phoebe_build_config.h"
#include "cfortran.h"

#define OPENSTREAM(FN)                                          CCALLSFSUB1(OPENSTREAM,openstream,STRING,FN)
#define CLOSESTREAM()                                           CCALLSFSUB0(CLOSESTREAM,closestream)

#define CREATELCILINE1(a1,a2,a3,a4,a5,a6,a7,a8)                 CCALLSFSUB8(CREATELCILINE1,createlciline1,INT,INT,INT,INT,INT,INT,INT,INT,a1,a2,a3,a4,a5,a6,a7,a8)
#define CREATELCILINE2(b1,b2,b3,b4,b5,b6,b7,b8)                 CCALLSFSUB8(CREATELCILINE2,createlciline2,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,INT,DOUBLE,b1,b2,b3,b4,b5,b6,b7,b8)
#define CREATELCILINE3(c1,c2,c3,c4,c5,c6,c7)                    CCALLSFSUB7(CREATELCILINE3,createlciline3,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,c1,c2,c3,c4,c5,c6,c7)
#define CREATELCILINE4(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)          CCALLSFSUB10(CREATELCILINE4,createlciline4,INT,INT,INT,INT,INT,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)
#define CREATELCILINE5(e1,e2,e3,e4,e5,e6,e7,e8,e9)              CCALLSFSUB9(CREATELCILINE5,createlciline5,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,e1,e2,e3,e4,e5,e6,e7,e8,e9)
#define CREATELCILINE6(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11)      CCALLSFSUB11(CREATELCILINE6,createlciline6,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11)
#define CREATELCILINE7(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12)  CCALLSFSUB12(CREATELCILINE7,createlciline7,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12)
#define CREATELCILINE8(h1,h2,h3,h4)                             CCALLSFSUB4(CREATELCILINE8,createlciline8,DOUBLE,DOUBLE,DOUBLE,DOUBLE,h1,h2,h3,h4)
#define CREATELCILINE10(j1,j2,j3,j4)                            CCALLSFSUB4(CREATELCILINE10,createlciline10,DOUBLE,DOUBLE,DOUBLE,DOUBLE,j1,j2,j3,j4)
#define CREATELCIENDLINE()                                      CCALLSFSUB0(CREATELCIENDLINE,createlciendline)

#define CREATEDCILINE1(a1,a2,a3,a4,a5,a6,a7,a8)                 CCALLSFSUB8(CREATEDCILINE1,createdciline1,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8)
#define CREATEDCILINE2(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)          CCALLSFSUB10(CREATEDCILINE2,createdciline2,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
#define CREATEDCILINE3(a1,a2,a3,a4,a5,a6,a7,a8,a9)              CCALLSFSUB9(CREATEDCILINE3,createdciline3,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9)
#define CREATEDCILINE4(a1,a2,a3,a4,a5)                          CCALLSFSUB5(CREATEDCILINE4,createdciline4,INTV,INT,INT,INT,DOUBLE,a1,a2,a3,a4,a5)
#define CREATEDCILINE5(a1,a2,a3,a4)                             CCALLSFSUB4(CREATEDCILINE5,createdciline5,INT,INT,INT,INT,a1,a2,a3,a4)
#define CREATEDCILINE6(a1,a2,a3,a4,a5,a6,a7)                    CCALLSFSUB7(CREATEDCILINE6,createdciline6,INT,INT,INT,INT,INT,INT,INT,a1,a2,a3,a4,a5,a6,a7)
#define CREATEDCILINE7(a1,a2,a3,a4,a5,a6,a7)                    CCALLSFSUB7(CREATEDCILINE7,createdciline7,INT,INT,INT,INT,INT,INT,INT,a1,a2,a3,a4,a5,a6,a7)
#define CREATEDCILINE8(a1,a2,a3,a4,a5)                          CCALLSFSUB5(CREATEDCILINE8,createdciline8,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5)
#define CREATEDCILINE9(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)  CCALLSFSUB12(CREATEDCILINE9,createdciline9,INT,INT,INT,INT,INT,INT,INT,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)
#define CREATEDCILINE10(a1,a2,a3,a4,a5,a6,a7,a8,a9)             CCALLSFSUB9(CREATEDCILINE10,createdciline10,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9)
#define CREATEDCILINE11(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)     CCALLSFSUB11(CREATEDCILINE11,createdciline11,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)
#define CREATEDCILINERV(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)         CCALLSFSUB10(CREATEDCILINERV,createdcilinerv,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
#define CREATEDCILINELC(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12) CCALLSFSUB12(CREATEDCILINELC,createdcilinelc,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,INT,DOUBLE,DOUBLE,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)
#define CREATEDATALINE(a1,a2,a3)                                CCALLSFSUB3(CREATEDATALINE,createdataline,DOUBLE,DOUBLE,DOUBLE,a1,a2,a3)
#define CREATEDCIENDLINE()                                      CCALLSFSUB0(CREATEDCIENDLINE,createdciendline)

#define CREATESPOTSSTOPLINE()                                   CCALLSFSUB0(CREATESPOTSSTOPLINE,createspotsstopline)
#define CREATECLOUDSSTOPLINE()                                  CCALLSFSUB0(CREATECLOUDSSTOPLINE,createcloudsstopline)
#define CREATEDATASTOPLINE()                                    CCALLSFSUB0(CREATEDATASTOPLINE,createdatastopline)

int create_lci_file (char filename[], WD_LCI_parameters param)
{
	/*
	 * This function calls a bunch of fortran subroutines to create an LC
	 * input file for WD. Note that third light is computed externally to WD
	 * and EL3=0 is always passed here.
	 */

	int i;

	OPENSTREAM (filename);
	CREATELCILINE1  (param.MPAGE, param.NREF, param.MREF, param.IFSMV1, param.IFSMV2, param.ICOR1, param.ICOR2, param.LD);
	CREATELCILINE2  (param.JDPHS, param.HJD0, param.PERIOD, param.DPDT, param.PSHIFT, param.SIGMA, param.WEIGHTING, param.SEED);
	CREATELCILINE3  (param.HJDST, param.HJDSP, param.HJDIN, param.PHSTRT, param.PHSTOP, param.PHIN, param.PHNORM);
	CREATELCILINE4  (param.MODE, param.IPB, param.IFAT1, param.IFAT2, param.N1, param.N2, param.PERR0, param.DPERDT, param.THE, 100.0);
	CREATELCILINE5  (param.E, param.SMA, param.F1, param.F2, param.VGA/100., param.INCL, param.GR1, param.GR2, param.MET1);
	CREATELCILINE6  (param.TAVH/10000., param.TAVC/10000., param.ALB1, param.ALB2, param.PHSV, param.PCSV, param.RM, param.XBOL1, param.XBOL2, param.YBOL1, param.YBOL2);
	CREATELCILINE7  (param.IBAND, param.HLA, param.CLA, param.X1A, param.X2A, param.Y1A, param.Y2A, /* param.EL3 = */ 0.0, param.OPSF, 0.0, param.FACTOR, param.WLA/1000.);
	if (param.SPRIM != 0)
		for (i = 0; i < param.SPRIM; i++)
			CREATELCILINE8  (param.XLAT1[i], param.XLONG1[i], param.RADSP1[i], param.TEMSP1[i]);
	CREATESPOTSSTOPLINE ();
	if (param.SSEC != 0)
		for (i = 0; i < param.SSEC; i++)
			CREATELCILINE10 (param.XLAT2[i], param.XLONG2[i], param.RADSP2[i], param.TEMSP2[i]);
	CREATESPOTSSTOPLINE ();
	CREATECLOUDSSTOPLINE ();
	CREATELCIENDLINE ();
	CLOSESTREAM ();

	return SUCCESS;
}

int create_dci_file (char *filename, void *pars)
	{
	int i, j;

	WD_DCI_parameters *params = (WD_DCI_parameters *) pars;

	/* For easier writing of arguments to functions below ;) :                  */
	double *step = params->step;

	OPENSTREAM (filename);
	CREATEDCILINE1  (step[0], step[1], step[2], step[3], step[4], step[5], step[6], step[7]);
	CREATEDCILINE2  (step[9], step[10], step[11], step[12], step[13], step[15], step[16], step[17], step[18]/10000., step[19]/10000.);
	CREATEDCILINE3  (step[20], step[21], step[22], step[23], step[24], step[30], step[31], step[32], step[33]);
	CREATEDCILINE4  ((int *) params->tba, 0, 1, 1, params->dclambda);
	CREATEDCILINE5  (params->spot1src, params->spot1id, params->spot2src, params->spot2id);
	CREATEDCILINE6  ((int) params->rv1data, (int) params->rv2data, params->nlc, 2, 0, (int) params->symder, 1);
	CREATEDCILINE7  (params->refno, params->refswitch, (int) params->spots1move, (int) params->spots2move, (int) params->rv1proximity, (int) params->rv2proximity, params->ldmodel);
	CREATEDCILINE8  (params->indep, params->hjd0, params->period, params->dpdt, params->pshift);
	CREATEDCILINE9  (params->morph, (int) params->cladec, (int) params->ifat1, (int) params->ifat2, params->n1f, params->n2f, params->n1c, params->n2c, params->perr0, params->dperdt, 0.0, 100.0);
	CREATEDCILINE10 (params->ecc, params->sma, params->f1, params->f2, params->vga/100., params->incl, params->grb1, params->grb2, params->met1);
	CREATEDCILINE11 (params->teff1/10000., params->teff2/10000., params->alb1, params->alb2, params->pot1, params->pot2, params->rm, params->xbol1, params->xbol2, params->ybol1, params->ybol2);

	for (i = 0; i < params->rv1data + params->rv2data; i++)
		CREATEDCILINERV (params->passband[i], params->hla[i], params->cla[i], params->x1a[i], params->x2a[i], params->y1a[i], params->y2a[i], params->opsf[i], params->sigma[i]/100.0, params->wavelength[i]/1000.);
	for (i = params->rv1data + params->rv2data; i < params->rv1data + params->rv2data + params->nlc; i++)
		CREATEDCILINELC (params->passband[i], params->hla[i], params->cla[i], params->x1a[i], params->x2a[i], params->y1a[i], params->y2a[i], params->el3[i], params->opsf[i], params->levweight[i], params->sigma[i], params->wavelength[i]/1000.);

	for (i = 0; i < params->spot1no; i++)
		CREATELCILINE8 (params->spot1lat[i], params->spot1long[i], params->spot1rad[i], params->spot1temp[i]);
	CREATESPOTSSTOPLINE ();

	for (i = 0; i < params->spot2no; i++)
		CREATELCILINE8 (params->spot2lat[i], params->spot2long[i], params->spot2rad[i], params->spot2temp[i]);
	CREATESPOTSSTOPLINE ();

	CREATECLOUDSSTOPLINE ();

	for (i = 0; i < params->rv1data + params->rv2data; i++) {
		for (j = 0; j < params->obs[i]->indep->dim; j++)
			CREATEDATALINE (params->obs[i]->indep->val[j], params->obs[i]->dep->val[j]/100.0, params->obs[i]->weight->val[j]);
		CREATEDATASTOPLINE ();
	}
	for (i = params->rv1data + params->rv2data; i < params->rv1data + params->rv2data + params->nlc; i++) {
		for (j = 0; j < params->obs[i]->indep->dim; j++)
			CREATEDATALINE (params->obs[i]->indep->val[j], params->obs[i]->dep->val[j], params->obs[i]->weight->val[j]);
		CREATEDATASTOPLINE ();
	}

#warning MULTIPLE SUBSETS MISSING IN DC
/*
	if (mms.on == 1)
		{
		if (mms.no >= 1) CREATEDCILINE4  (mms.s1, 1, 1, 1, switches.XLAMDA);
		if (mms.no >= 2) CREATEDCILINE4  (mms.s2, 1, 1, 1, switches.XLAMDA);
		if (mms.no >= 3) CREATEDCILINE4  (mms.s3, 1, 1, 1, switches.XLAMDA);
		if (mms.no >= 4) CREATEDCILINE4  (mms.s4, 1, 1, 1, switches.XLAMDA);
		if (mms.no == 5) CREATEDCILINE4  (mms.s5, 1, 1, 1, switches.XLAMDA);
		}
*/
	CREATEDCIENDLINE ();

	CLOSESTREAM ();

	return 0;
	}
