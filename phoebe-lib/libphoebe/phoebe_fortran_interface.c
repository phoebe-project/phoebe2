#include <stdlib.h>
#include <stdio.h>

#include "../libwd/wd.h"
#include "phoebe_error_handling.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"

#include "phoebe_build_config.h"

int create_lci_file (char *filename, WD_LCI_parameters *param)
{
	/*
	 * This function calls the auxiliary fortran subroutine wrlci from the
	 * WD library to create the LC input file.
	 */

	double vunit = 100.0;
	double mzero = 0.0;
	double  tavh = param->TAVH/10000.0;
	double  tavc = param->TAVC/10000.0;
	double   vga = param->VGA/100.0;
	double   wla = param->WLA/10000.0;

	wd_wrlci (filename, param->MPAGE, param->NREF, param->MREF, param->IFSMV1, param->IFSMV2, param->ICOR1, param->ICOR2, param->LD,
			  param->JDPHS, param->HJD0, param->PERIOD, param->DPDT, param->PSHIFT, param->SIGMA, param->WEIGHTING, param->SEED,
			  param->HJDST, param->HJDSP, param->HJDIN, param->PHSTRT, param->PHSTOP, param->PHIN, param->PHNORM,
			  param->MODE, param->IPB, param->IFAT1, param->IFAT2, param->N1, param->N2, param->PERR0, param->DPERDT, param->THE, vunit,
			  param->E, param->SMA, param->F1, param->F2, vga, param->INCL, param->GR1, param->GR2, param->MET1,
			  tavh, tavc, param->ALB1, param->ALB2, param->PHSV, param->PCSV, param->RM, param->XBOL1, param->XBOL2, param->YBOL1, param->YBOL2,
			  param->IBAND, param->HLA, param->CLA, param->X1A, param->X2A, param->Y1A, param->Y2A, param->EL3, param->OPSF, mzero, param->FACTOR, wla,
			  param->SPRIM, param->XLAT1, param->XLONG1, param->RADSP1, param->TEMSP1, param->SSEC, param->XLAT2, param->XLONG2, param->RADSP2, param->TEMSP2);

	return SUCCESS;
}

int create_dci_file (char *filename, void *pars)
{
	/*
	 * This function calls the auxiliary fortran subroutine wrdci from the
	 * WD library to create the DC input file.
	 */

	int i, j;

	WD_DCI_parameters *params = (WD_DCI_parameters *) pars;

	int ifder = 0;
	int ifm = 1;
	int ifr = 1;
	int k0 = 2;
	int kdisk = 0;
	int nppl = 1;
	double the = 0.0;
	double vunit = 100.0;
	double vga = params->vga / 100.0;
	double tavh = params->teff1/10000.0;
	double tavc = params->teff2/10000.0;

	double *step, *wla, *sigma;
	double *indep, *dep, *weight;

	int  rvno = params->rv1data + params->rv2data;
	int   cno = rvno + params->nlc;
	int ptsno = 0;
	int index = 0;

	for (i = 0; i < cno; i++)
		ptsno += params->obs[i]->indep->dim + 1;

	if (cno != 0) {
		wla = phoebe_malloc (cno * sizeof (*wla));
		sigma = phoebe_malloc (cno * sizeof (*sigma));
		indep = phoebe_malloc (ptsno * sizeof (*indep));
		dep = phoebe_malloc (ptsno * sizeof (*dep));
		weight = phoebe_malloc (ptsno * sizeof (*weight));

		for (i = 0; i < rvno; i++) {
			wla[i] = params->wavelength[i] / 10000.0;
			sigma[i] = params->sigma[i] / 100.0;
			for (j = 0; j < params->obs[i]->indep->dim; j++) {
				 indep[index] = params->obs[i]->indep->val[j];
				   dep[index] = params->obs[i]->dep->val[j]/100.0;
				weight[index] = params->obs[i]->weight->val[j];
				index++;
			}
			indep[index] = -10001.0; dep[index] = 0.0; weight[index] = 0.0;
			index++;
		}

		for (i = rvno; i < cno; i++) {
			wla[i] = params->wavelength[i] / 10000.0;
			sigma[i] = params->sigma[i];
			for (j = 0; j < params->obs[i]->indep->dim; j++) {
				 indep[index] = params->obs[i]->indep->val[j];
				   dep[index] = params->obs[i]->dep->val[j];
				weight[index] = params->obs[i]->weight->val[j];
				index++;
			}
			indep[index] = -10001.0; dep[index] = 0.0; weight[index] = 0.0;
			index++;
		}
	}
	else {
		wla = NULL;
		sigma = NULL;
		indep = NULL;
		dep = NULL;
		weight = NULL;
	}

	step = phoebe_malloc (35 * sizeof (*step));
	for (i = 0; i < 35; i++)
		step[i] = params->step[i];

	step[14] /= 100.0;   /* vga */
	step[18] /= 10000.0; /* T1 */
	step[19] /= 10000.0; /* T2 */

	wd_wrdci (filename, step, params->tba, ifder, ifm, ifr, params->dclambda,
			  params->spot1src, params->spot1id, params->spot2src, params->spot2id,
			  params->rv1data, params->rv2data, params->nlc, k0, kdisk, params->symder, nppl,
			  params->refno, params->refswitch, params->spots1move, params->spots2move, params->rv1proximity, params->rv2proximity, params->ldmodel,
			  params->indep, params->hjd0, params->period, params->dpdt, params->pshift,
			  params->morph, params->cladec, params->ifat1, params->ifat2, params->n1f, params->n2f, params->n1c, params->n2c, params->perr0, params->dperdt, the, vunit,
			  params->ecc, params->sma, params->f1, params->f2, vga, params->incl, params->grb1, params->grb2, params->met1,
			  tavh, tavc, params->alb1, params->alb2, params->pot1, params->pot2, params->rm, params->xbol1, params->xbol2, params->ybol1, params->ybol2,
			  params->passband, params->hla, params->cla, params->x1a, params->x2a, params->y1a, params->y2a, params->el3, params->opsf, params->levweight, sigma, wla,
			  params->spot1no, params->spot1lat, params->spot1long, params->spot1rad, params->spot1temp, params->spot2no, params->spot2lat, params->spot2long, params->spot2rad, params->spot2temp,
			  ptsno, indep, dep, weight);
			  
	if (wla) free (wla);
	if (sigma) free (sigma);
	if (indep) free (indep);
	if (dep) free (dep);
	if (weight) free (weight);

	free (step);

	return SUCCESS;
}
