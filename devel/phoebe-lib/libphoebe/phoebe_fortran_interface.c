#include <stdlib.h>
#include <stdio.h>

#include "phoebe_allocations.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_ld.h"
#include "phoebe_parameters.h"

#include "../libwd/wd.h"

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

	integer ifder = 0;
	integer ifm = 1;
	integer ifr = 1;
	integer k0 = 2;
	integer kdisk = 0;
	integer nppl = 1;

	doublereal the = 0.0;
	doublereal vunit = 100.0;
	doublereal vga = params->vga / 100.0;
	doublereal tavh = params->teff1/10000.0;
	doublereal tavc = params->teff2/10000.0;

	doublereal *step, *wla, *sigma;
	doublereal *indep, *dep, *weight;

	integer  rvno = params->rv1data + params->rv2data;
	integer   cno = rvno + params->nlc;
	integer ptsno = 0;
	integer index = 0;

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

int wd_lci_parameters_get (WD_LCI_parameters *params, int MPAGE, int curve)
{
	/*
	 * This function reads out all variables that build up the LCI file.
	 * 
	 * Return values:
	 * 
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_UNINITIALIZED_CURVE
	 *   ERROR_UNSUPPORTED_MPAGE  
	 *   ERROR_INVALID_LDLAW
	 *   SUCCESS
	 */

	int i;

	int lcno, rvno;

	int    readout_int;
	bool   readout_bool;
	double readout_dbl;
	char  *readout_str;
	
	const char *filter = NULL;

	PHOEBE_passband *passband;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	/*
	 * The MPAGE switch determines the type of a synthetic curve the program
	 * should compute:
	 *
	 *   MPAGE = 1 .. light curve
	 *   MPAGE = 2 .. radial velocity curve
	 *
	 * It is passed as argument to this function. First we check if the chosen
	 * curve is initialized and, if everything is ok, we set the params->MPAGE
	 * field to the passed value.
	 */

	if (MPAGE == 1)
		if (curve < 0 || curve > lcno-1)
			return ERROR_UNINITIALIZED_CURVE;
	if (MPAGE == 2)
		if (curve < 0 || curve > rvno-1)
			return ERROR_UNINITIALIZED_CURVE;
	if (MPAGE != 1 && MPAGE != 2)
		return ERROR_UNSUPPORTED_MPAGE;

	params->MPAGE = (integer) MPAGE;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_switch"), &readout_bool);
	if (readout_bool == YES) params->MREF = 2; else params->MREF = 1;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_reflections"), &readout_int);
	params->NREF = (integer) readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_move1"), &readout_bool);
	if (readout_bool) params->IFSMV1 = 1; else params->IFSMV1 = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_move2"), &readout_bool);
	if (readout_bool) params->IFSMV2 = 1; else params->IFSMV2 = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"), &readout_bool);
	if (readout_bool) params->ICOR1  = 1; else params->ICOR1  = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"), &readout_bool);
	if (readout_bool) params->ICOR2  = 1; else params->ICOR2  = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_model"), &readout_str);
	params->LD = phoebe_ld_model_type (readout_str);
	if (params->LD == LD_LAW_INVALID)
		return ERROR_INVALID_LDLAW;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
	if (strcmp (readout_str, "Time (HJD)")  == 0) params->JDPHS = 1;
	if (strcmp (readout_str, "Phase") == 0) params->JDPHS = 2;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hjd0"), &readout_dbl);
	params->HJD0 = (doublereal) readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_period"), &readout_dbl);
	params->PERIOD = (doublereal) readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dpdt"), &readout_dbl);
	params->DPDT = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pshift"), &readout_dbl);
	params->PSHIFT = (doublereal) readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_switch"), &readout_bool);
	if (readout_bool) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_sigma"), &readout_dbl);
		params->SIGMA = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_levweight"), &readout_str);
		if (strcmp (readout_str, "No level-dependent weighting") == 0) params->WEIGHTING = 0;
		if (strcmp (readout_str, "Poissonian scatter")           == 0) params->WEIGHTING = 1;
		if (strcmp (readout_str, "Low light scatter")            == 0) params->WEIGHTING = 2;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_seed"), &readout_dbl);
		params->SEED = (doublereal) readout_dbl;
	}
	else {
		params->SIGMA = 0.0;
		params->WEIGHTING = 0;
		params->SEED = 0.0;
	}

	/* The following parameters are dummies, since PHOEBE calls WD algorithm  */
	/* for each phase individually:                                           */
	params->HJDST  = 0.0;
	params->HJDSP  = 1.0;
	params->HJDIN  = 0.1;
	params->PHSTRT = 0.0;
	params->PHSTOP = 1.0;
	params->PHIN   = 0.1;

	/* Normalizing magnitude (PHNORM) tells WD at what phase the normalized   */
	/* flux should be 1. Usually this is 0.25, but could in general be at     */
	/* some other phase. Since there is no support for that in PHOEBE yet, it */
	/* is hardcoded to 0.25, but it should be changed.                        */
	params->PHNORM = 0.25;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_model"), &readout_str);
	if (strcmp (readout_str, "X-ray binary"                                         ) == 0) params->MODE = -1;
	if (strcmp (readout_str, "Unconstrained binary system"                          ) == 0) params->MODE =  0;
	if (strcmp (readout_str, "Overcontact binary of the W UMa type"                 ) == 0) params->MODE =  1;
	if (strcmp (readout_str, "Detached binary"                                      ) == 0) params->MODE =  2;
	if (strcmp (readout_str, "Overcontact binary not in thermal contact"            ) == 0) params->MODE =  3;
	if (strcmp (readout_str, "Semi-detached binary, primary star fills Roche lobe"  ) == 0) params->MODE =  4;
	if (strcmp (readout_str, "Semi-detached binary, secondary star fills Roche lobe") == 0) params->MODE =  5;
	if (strcmp (readout_str, "Double contact binary"                                ) == 0) params->MODE =  6;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_compute_hla_switch"), &readout_bool);
	if (readout_bool && lcno > 0) params->CALCHLA = 1; else params->CALCHLA = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_compute_vga_switch"), &readout_bool);
	if (readout_bool && rvno > 0) params->CALCVGA = 1; else params->CALCVGA = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_usecla_switch"), &readout_bool);
	if (readout_bool) params->IPB = 1; else params->IPB  = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_atm1_switch"), &readout_bool);
	if (readout_bool) params->IFAT1 = 1; else params->IFAT1 = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_atm2_switch"), &readout_bool);
	if (readout_bool) params->IFAT2 = 1; else params->IFAT2 = 0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize1"), &readout_int);
	params->N1 = (integer) readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize2"), &readout_int);
	params->N2 = (integer) readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_perr0"), &readout_dbl);
	params->PERR0 = (doublereal) readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dperdt"), &readout_dbl);
	params->DPERDT = (doublereal) readout_dbl;

	/* THE applies only to X-ray binaries, but it isn't supported yet.        */
	params->THE    = 0.0;

	/* VUNIT is the internal radial velocity unit. This should be calculated  */
	/* rather than determined by the user, however it isn't written yet. It   */
	/* is thus hardcoded to the most obvious number, that is 100 km/s.        */
	params->VUNIT  = 100.0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ecc"),      &readout_dbl);
	params->E = (doublereal) readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_sma"),      &readout_dbl);
	params->SMA = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_f1"),       &readout_dbl);
	params->F1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_f2"),       &readout_dbl);
	params->F2 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_vga"),      &readout_dbl);
	params->VGA = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_incl"),     &readout_dbl);
	params->INCL = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grb1"),     &readout_dbl);
	params->GR1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grb2"),     &readout_dbl);
	params->GR2 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"),    &readout_dbl);
	params->LOGG1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"),    &readout_dbl);
	params->LOGG2 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met1"),     &readout_dbl);
	params->MET1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met2"),     &readout_dbl);
	params->MET2 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff1"),    &readout_dbl);
	params->TAVH = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff2"),    &readout_dbl);   
	params->TAVC = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_alb1"),     &readout_dbl);
	params->ALB1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_alb2"),     &readout_dbl);
	params->ALB2 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot1"),     &readout_dbl);
	params->PHSV = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot2"),     &readout_dbl);
	params->PCSV = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rm"),       &readout_dbl);
	params->RM = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_xbol1"), &readout_dbl);
	params->XBOL1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_xbol2"), &readout_dbl);
	params->XBOL2 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_ybol1"), &readout_dbl);
	params->YBOL1 = (doublereal) readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_ybol2"), &readout_dbl);
	params->YBOL2 = (doublereal) readout_dbl;

	if (MPAGE == 2)
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), curve, &filter);
	else
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), curve, &filter);

	passband = phoebe_passband_lookup (filter);
	if (!passband) {
		phoebe_lib_warning ("passband not set or invalid, reverting to Johnson V.\n");
		params->IBAND = 7;
		params->WLA   = 550.0;
	} else {
		params->IBAND = (integer) get_passband_id (filter);
		params->WLA   = (doublereal) passband->effwl;
	}

	if (MPAGE == 1) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hla"), curve, &readout_dbl);
		params->HLA = (doublereal) readout_dbl;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cla"), curve, &readout_dbl);
		params->CLA = (doublereal) readout_dbl;
	}
	else {
		/* HLAs and CLAs don't make any sense for RV curves, so we initialize     */
		/* them to some canonic dummy values.                                     */
		params->HLA = 1.0;
		params->CLA = 1.0;
	}

	if (MPAGE == 2) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx1"), curve, &readout_dbl);
		params->X1A = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx2"), curve, &readout_dbl);
		params->X2A = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy1"), curve, &readout_dbl);
		params->Y1A = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy2"), curve, &readout_dbl);
		params->Y2A = (doublereal) readout_dbl;
	}
	else {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx1"), curve, &readout_dbl);
		params->X1A = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx2"), curve, &readout_dbl);
		params->X2A = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy1"), curve, &readout_dbl);
		params->Y1A = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy2"), curve, &readout_dbl);
		params->Y2A = (doublereal) readout_dbl;
	}

	if (MPAGE == 1) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"), curve, &readout_dbl);
		params->EL3 = (doublereal) readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_opsf"), curve, &readout_dbl);
		params->OPSF = (doublereal) readout_dbl;
	}
	else {
		/* Third light and opacity function don't make sense for RVs.             */
		params->EL3 = 0.0;
		params->OPSF = 0.0;
	}

	/*
	 * MZERO and FACTOR variables set offsets in synthetic light curves. PHOEBE
	 * controls this by its own variables, so we hardcode these to 0 and 1.
	 */

	params->MZERO  = 0.0;
	params->FACTOR = 1.0;

	/* Spots: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no1"), &readout_int);
	params->SPRIM = (integer) readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no2"), &readout_int);
	params->SSEC = (integer) readout_int;

	/* Allocate memory for spot parameters and fill in the values: */
	if (params->SPRIM != 0) {
		params->XLAT1  = phoebe_malloc (params->SPRIM * sizeof (*(params->XLAT1)));
		params->XLONG1 = phoebe_malloc (params->SPRIM * sizeof (*(params->XLONG1)));
		params->RADSP1 = phoebe_malloc (params->SPRIM * sizeof (*(params->RADSP1)));
		params->TEMSP1 = phoebe_malloc (params->SPRIM * sizeof (*(params->TEMSP1)));

		for (i = 0; i < params->SPRIM; i++) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_lat1"),  i, &readout_dbl);
			params->XLAT1[i] = (doublereal) readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_long1"), i, &readout_dbl);
			params->XLONG1[i] = (doublereal) readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_rad1"),  i, &readout_dbl);
			params->RADSP1[i] = (doublereal) readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_temp1"), i, &readout_dbl);
			params->TEMSP1[i] = (doublereal) readout_dbl;
		}
	}
	if (params->SSEC != 0) {
		params->XLAT2  = phoebe_malloc (params->SSEC * sizeof (*(params->XLAT2)));
		params->XLONG2 = phoebe_malloc (params->SSEC * sizeof (*(params->XLONG2)));
		params->RADSP2 = phoebe_malloc (params->SSEC * sizeof (*(params->RADSP2)));
		params->TEMSP2 = phoebe_malloc (params->SSEC * sizeof (*(params->TEMSP2)));

		for (i = 0; i < params->SSEC; i++) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_lat2"),  i, &readout_dbl);
			params->XLAT2[i] = (doublereal) readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_long2"), i, &readout_dbl);
			params->XLONG2[i] = (doublereal) readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_rad2"),  i, &readout_dbl);
			params->RADSP2[i] = (doublereal) readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_temp2"), i, &readout_dbl);
			params->TEMSP2[i] = (doublereal) readout_dbl;
		}
	}

	return SUCCESS;
}

WD_DCI_parameters *wd_dci_parameters_new ()
{
	/*
	 * This function initializes the WD DCI structure for allocation.
	 */

	/* Allocate memory for the structure itself: */
	WD_DCI_parameters *pars = phoebe_malloc (sizeof (*pars));

	/* There are several parameters that determine the dimension of arrays.   */
	/* Set them all to 0 here:                                                */
	pars->rv1data    = 0;
	pars->rv2data    = 0;
	pars->nlc        = 0;
	pars->spot1no    = 0;
	pars->spot2no    = 0;

	/* NULLify structure arrays: */
	pars->tba        = NULL;
	pars->step       = NULL;
	pars->passband   = NULL;
	pars->wavelength = NULL;
	pars->sigma      = NULL;
	pars->hla        = NULL;
	pars->cla        = NULL;
	pars->x1a        = NULL;
	pars->y1a        = NULL;
	pars->x2a        = NULL;
	pars->y2a        = NULL;
	pars->el3        = NULL;
	pars->opsf       = NULL;
	pars->levweight  = NULL;
	pars->spot1lat   = NULL;
	pars->spot1long  = NULL;
	pars->spot1rad   = NULL;
	pars->spot1temp  = NULL;
	pars->spot2lat   = NULL;
	pars->spot2long  = NULL;
	pars->spot2rad   = NULL;
	pars->spot2temp  = NULL;

	/* Finally, these are arrays of curves. We NULLify them as well:          */
	pars->obs   = NULL;

	return pars;
}

int read_in_wd_dci_parameters (WD_DCI_parameters *params, int *marked_tba)
{
	/*
	 * This function reads in DCI parameters. Be sure to allocate (and free)
	 * the passed WD_DCI_parameters structure (and its substructures).
	 *
	 * Supported error states:
	 *
	 *  ERROR_INVALID_INDEP
	 *  ERROR_INVALID_DEP
	 *  ERROR_INVALID_WEIGHT
	 *  ERROR_MINIMIZER_NO_CURVES
	 *  ERROR_MINIMIZER_NO_PARAMS
	 *  ERROR_INVALID_DATA
	 *  SUCCESS
	 */

	char *pars[] = {
		"phoebe_spots_lat1",
		"phoebe_spots_long1",
		"phoebe_spots_rad1",
		"phoebe_spots_temp1",
		"phoebe_spots_lat2",
		"phoebe_spots_long2",
		"phoebe_spots_rad2",
		"phoebe_spots_temp2",
		"phoebe_sma",
		"phoebe_ecc",
		"phoebe_perr0",
		"phoebe_f1",
		"phoebe_f2",
		"phoebe_pshift",
		"phoebe_vga",
		"phoebe_incl",
		"phoebe_grb1",
		"phoebe_grb2",
		"phoebe_teff1",
		"phoebe_teff2",
		"phoebe_alb1",
		"phoebe_alb2",
		"phoebe_pot1",
		"phoebe_pot2",
		"phoebe_rm",
		"phoebe_hjd0",
		"phoebe_period",
		"phoebe_dpdt",
		"phoebe_dperdt",
		"",
		"phoebe_hla",
		"phoebe_cla",
		"phoebe_ld_lcx1",
		"phoebe_ld_lcx2",
		"phoebe_el3"
	};

	int i, status;
	int lcno, rvno, cno;
	bool readout_bool;
	const char *readout_str;

	int rv1index = -1;
	int rv2index = -1;

	PHOEBE_column_type master_indep, itype, dtype, wtype;

	/* DC features 35 adjustable parameters and we initialize such arrays: */
	integer *tba;
	doublereal *step;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);
	cno = lcno + rvno;

	*marked_tba = 0;

	/* If there are no observations defined, bail out: */
	if (cno == 0) return ERROR_MINIMIZER_NO_CURVES;

	/* Allocate memory: */
	 tba = phoebe_malloc (35 * sizeof ( *tba));
	step = phoebe_malloc (35 * sizeof (*step));

	/*
	 * Read in TBA states and step-sizes; note the '!' in front of the tba
	 * readout function; that's because WD sets 1 for /kept/ parameters and
	 * 0 for parameters that are marked for adjustment.
	 */

	for (i = 0; i < 35; i++) {
		if (i == 29) { tba[i] = !FALSE; continue; } /* reserved WD channel */
		phoebe_parameter_get_tba (phoebe_parameter_lookup (pars[i]), &readout_bool);
		tba[i] = (integer) (!readout_bool);
		phoebe_parameter_get_step (phoebe_parameter_lookup (pars[i]), &step[i]);
		if (i > 29)
			*marked_tba += lcno * (1-tba[i]);
		else
			*marked_tba += 1-tba[i];
	}

	params->tba  = tba;
	params->step = step;

	/* Check for presence of RV and LC data: */
	{
	params->rv1data = FALSE; params->rv2data = FALSE;
	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), i, &readout_str);
		status = phoebe_column_get_type (&dtype, readout_str);
		if (status != SUCCESS) return status;

		switch (dtype) {
			case PHOEBE_COLUMN_PRIMARY_RV:
				params->rv1data = TRUE;
				rv1index = i;
			break;
			case PHOEBE_COLUMN_SECONDARY_RV:
				params->rv2data = TRUE;
				rv2index = i;
			break;
			default:
				phoebe_lib_error ("exception handler invoked in read_in_wd_dci_parameters (), please report this!\n");
		}
	}
	params->nlc = lcno;
	}

	/* DC-related parameters:                                                 */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_lambda"), &(params->dclambda));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_symder_switch"), &(params->symder));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_coarsesize1"), &(params->n1c));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_coarsesize2"), &(params->n2c));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize1"), &(params->n1f));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize2"), &(params->n2f));

	/* Reflection effect:                                                       */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_switch"), &readout_bool);
	if (readout_bool) params->refswitch = 2; else params->refswitch = 1;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_reflections"), &(params->refno));

	/* Eclipse/proximity effect:                                                */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"), &(params->rv1proximity));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"), &(params->rv2proximity));

	/* Limb darkening effect:                                                   */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_model"), &readout_str);
	params->ldmodel = phoebe_ld_model_type (readout_str);
	if (params->ldmodel == LD_LAW_INVALID)
		return ERROR_INVALID_LDLAW;

	/* Morphological constraint:                                                */
	{
	const char *mode;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_model"), &mode);
	params->morph = -2;
	if (strcmp (mode, "X-ray binary"                                         ) == 0) params->morph = -1;
	if (strcmp (mode, "Unconstrained binary system"                          ) == 0) params->morph =  0;
	if (strcmp (mode, "Overcontact binary of the W UMa type"                 ) == 0) params->morph =  1;
	if (strcmp (mode, "Detached binary"                                      ) == 0) params->morph =  2;
	if (strcmp (mode, "Overcontact binary not in thermal contact"            ) == 0) params->morph =  3;
	if (strcmp (mode, "Semi-detached binary, primary star fills Roche lobe"  ) == 0) params->morph =  4;
	if (strcmp (mode, "Semi-detached binary, secondary star fills Roche lobe") == 0) params->morph =  5;
	if (strcmp (mode, "Double contact binary"                                ) == 0) params->morph =  6;
	if (params->morph == -2) return ERROR_INVALID_MODEL;
	}

	/* Do we work in HJD-space or in phase-space? */
	{
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
		status = phoebe_column_get_type (&master_indep, readout_str);
		if (status != SUCCESS) return status;

		if (master_indep == PHOEBE_COLUMN_HJD)
			params->indep = 1;
		else if (master_indep == PHOEBE_COLUMN_PHASE)
			params->indep = 2;
		else
			return ERROR_INVALID_INDEP;
	}

	/* Luminosity decoupling:                                                   */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_usecla_switch"), &(params->cladec));

	/* Model atmosphere switches:                                               */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_atm1_switch"), &(params->ifat1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_atm2_switch"), &(params->ifat2));

	/* Model parameters:                                                        */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hjd0"),     &(params->hjd0));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_period"),   &(params->period));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dpdt"),     &(params->dpdt));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pshift"),   &(params->pshift));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_perr0"),    &(params->perr0));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dperdt"),   &(params->dperdt));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ecc"),      &(params->ecc));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_sma"),      &(params->sma));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_f1"),       &(params->f1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_f2"),       &(params->f2));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_vga"),      &(params->vga));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_incl"),     &(params->incl));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grb1"),     &(params->grb1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grb2"),     &(params->grb2));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met1"),     &(params->met1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff1"),    &(params->teff1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff2"),    &(params->teff2));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_alb1"),     &(params->alb1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_alb2"),     &(params->alb2));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot1"),     &(params->pot1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot2"),     &(params->pot2));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rm"),       &(params->rm));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_xbol1"), &(params->xbol1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_xbol2"), &(params->xbol2));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_ybol1"), &(params->ybol1));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_ybol2"), &(params->ybol2));

	/* Passband-dependent parameters:                                           */
	{
	int index;
	PHOEBE_passband *passband;

	params->passband   = phoebe_malloc (cno * sizeof (*(params->passband)));
	params->wavelength = phoebe_malloc (cno * sizeof (*(params->wavelength)));
	params->sigma      = phoebe_malloc (cno * sizeof (*(params->sigma)));
	params->hla        = phoebe_malloc (cno * sizeof (*(params->hla)));
	params->cla        = phoebe_malloc (cno * sizeof (*(params->cla)));
	params->x1a        = phoebe_malloc (cno * sizeof (*(params->x1a)));
	params->y1a        = phoebe_malloc (cno * sizeof (*(params->y1a)));
	params->x2a        = phoebe_malloc (cno * sizeof (*(params->x2a)));
	params->y2a        = phoebe_malloc (cno * sizeof (*(params->y2a)));
	params->el3        = phoebe_malloc (cno * sizeof (*(params->el3)));
	params->opsf       = phoebe_malloc (cno * sizeof (*(params->opsf)));
	params->levweight  = phoebe_malloc (cno * sizeof (*(params->levweight)));

	for (i = 0; i < rvno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), i, &readout_str);
		status = phoebe_column_get_type (&dtype, readout_str);
		if (status != SUCCESS) {
			free (params->passband); free (params->wavelength); free (params->sigma);
			free (params->hla);      free (params->cla);        free (params->x1a);
			free (params->y1a);      free (params->x2a);        free (params->y2a);
			free (params->el3);      free (params->opsf);       free (params->levweight);
			return status;
		}

		if (dtype == PHOEBE_COLUMN_SECONDARY_RV && rvno == 2) index = 1; else index = 0;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), i, &readout_str);
		passband = phoebe_passband_lookup (readout_str);
		if (!passband) {
			free (params->passband); free (params->wavelength); free (params->sigma);
			free (params->hla);      free (params->cla);        free (params->x1a);
			free (params->y1a);      free (params->x2a);        free (params->y2a);
			free (params->el3);      free (params->opsf);       free (params->levweight);
			return ERROR_PASSBAND_INVALID;
		}

		params->passband[index]   = get_passband_id (readout_str);
		params->wavelength[index] = passband->effwl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), i, &(params->sigma[index]));
		params->hla[index]        = 10.0;
		params->cla[index]        = 10.0;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx1"), i, &(params->x1a[index]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy1"), i, &(params->y1a[index]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx2"), i, &(params->x2a[index]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy2"), i, &(params->y2a[index]));
		params->el3[index]        = 0.0;
		params->opsf[index]       = 0.0;
		params->levweight[index]  = 0;
	}

	for (i = rvno; i < cno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), i-rvno, &readout_str);

		passband = phoebe_passband_lookup (readout_str);
		if (!passband) {
			free (params->passband); free (params->wavelength); free (params->sigma);
			free (params->hla);      free (params->cla);        free (params->x1a);
			free (params->y1a);      free (params->x2a);        free (params->y2a);
			free (params->el3);      free (params->opsf);       free (params->levweight);
			return ERROR_PASSBAND_INVALID;
		}

		params->passband[i]   = get_passband_id (readout_str);
		params->wavelength[i] = passband->effwl;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_sigma"), i-rvno, &(params->sigma[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hla"), i-rvno, &(params->hla[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cla"), i-rvno, &(params->cla[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx1"), i-rvno, &(params->x1a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy1"), i-rvno, &(params->y1a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx2"), i-rvno, &(params->x2a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy2"), i-rvno, &(params->y2a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"), i-rvno, &(params->el3[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_opsf"), i-rvno, &(params->opsf[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_levweight"), i-rvno, &readout_str);
		params->levweight[i]  = get_level_weighting_id (readout_str);
	}
	}

	/* Spot parameters:                                                       */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no1"), &(params->spot1no));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no2"), &(params->spot2no));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot1src"), &(params->spot1src));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot2src"), &(params->spot2src));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot1id"), &(params->spot1id));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot2id"), &(params->spot2id));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_move1"), &(params->spots1move));
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_move2"), &(params->spots2move));

	params->spot1lat  = phoebe_malloc (params->spot1no * sizeof (*(params->spot1lat)));
	params->spot1long = phoebe_malloc (params->spot1no * sizeof (*(params->spot1long)));
	params->spot1rad  = phoebe_malloc (params->spot1no * sizeof (*(params->spot1rad)));
	params->spot1temp = phoebe_malloc (params->spot1no * sizeof (*(params->spot1temp)));
	params->spot2lat  = phoebe_malloc (params->spot2no * sizeof (*(params->spot2lat)));
	params->spot2long = phoebe_malloc (params->spot2no * sizeof (*(params->spot2long)));
	params->spot2rad  = phoebe_malloc (params->spot2no * sizeof (*(params->spot2rad)));
	params->spot2temp = phoebe_malloc (params->spot2no * sizeof (*(params->spot2temp)));

	for (i = 0; i < params->spot1no; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_lat1"), i, &(params->spot1lat[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_long1"), i, &(params->spot1long[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_rad1"), i, &(params->spot1rad[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_temp1"), i, &(params->spot1temp[i]));
	}
	for (i = 0; i < params->spot2no; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_lat2"), i, &(params->spot2lat[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_long2"), i, &(params->spot2long[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_rad2"), i, &(params->spot2rad[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_temp2"), i, &(params->spot2temp[i]));
	}

	/* Observational data: */
	{
		/* Allocate observational data arrays:                                */
		params->obs = phoebe_malloc (cno * sizeof (*(params->obs)));

		if (params->rv1data) {
			const char *filename;
			char *passband;
			PHOEBE_passband *passband_ptr;
			double sigma;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filename"), rv1index, &filename);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), rv1index, &passband);
			passband_ptr = phoebe_passband_lookup (passband);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indep"), rv1index, &readout_str);
			phoebe_column_get_type (&itype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), rv1index, &readout_str);
			phoebe_column_get_type (&dtype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indweight"), rv1index, &readout_str);
			phoebe_column_get_type (&wtype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), rv1index, &sigma);

			params->obs[0] = phoebe_curve_new_from_file ((char *) filename);
			if (!params->obs[0])
				return ERROR_FILE_NOT_FOUND;

			phoebe_curve_set_properties (params->obs[0], PHOEBE_CURVE_RV, (char *) filename, passband_ptr, itype, dtype, wtype, sigma);
			phoebe_curve_transform (params->obs[0], master_indep, dtype, PHOEBE_COLUMN_WEIGHT);
		}
		if (params->rv2data) {
			int index;
			const char *filename;
			char *passband;
			PHOEBE_passband *passband_ptr;
			double sigma;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filename"), rv2index, &filename);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), rv2index, &passband);
			passband_ptr = phoebe_passband_lookup (passband);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indep"), rv2index, &readout_str);
			phoebe_column_get_type (&itype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), rv2index, &readout_str);
			phoebe_column_get_type (&dtype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_indweight"), rv2index, &readout_str);
			phoebe_column_get_type (&wtype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), rv2index, &sigma);

			if (params->rv1data) index = 1; else index = 0;
			params->obs[index] = phoebe_curve_new_from_file ((char *) filename);
			if (!params->obs[index])
				return ERROR_FILE_NOT_FOUND;

			phoebe_curve_set_properties (params->obs[index], PHOEBE_CURVE_RV, (char *) filename, passband_ptr, itype, dtype, wtype, sigma);
			phoebe_curve_transform (params->obs[index], master_indep, dtype, PHOEBE_COLUMN_WEIGHT);
		}
		for (i = rvno; i < cno; i++) {
			const char *filename;
			char *passband;
			PHOEBE_passband *passband_ptr;
			double sigma;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filename"), i-rvno, &filename);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), i-rvno, &passband);
			passband_ptr = phoebe_passband_lookup (passband);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_indep"), i-rvno, &readout_str);
			phoebe_column_get_type (&itype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_dep"), i-rvno, &readout_str);
			phoebe_column_get_type (&dtype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_indweight"), i-rvno, &readout_str);
			phoebe_column_get_type (&wtype, readout_str);

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_sigma"), i-rvno, &sigma);

			params->obs[i] = phoebe_curve_new_from_file ((char *) filename);
			if (!params->obs[i])
				return ERROR_FILE_NOT_FOUND;

			phoebe_curve_set_properties (params->obs[i], PHOEBE_CURVE_RV, (char *) filename, passband_ptr, itype, dtype, wtype, sigma);
			phoebe_curve_transform (params->obs[i], master_indep, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_WEIGHT);
		}
	}

	return SUCCESS;
}

int wd_dci_parameters_free (WD_DCI_parameters *params)
{
	/*
	 * This function frees the arrays of the structure and the structure itself.
	 */

	int i;
	int curve_no = params->rv1data + params->rv2data + params->nlc;

	if (params->tba)  free (params->tba);
	if (params->step) free (params->step);

	if (curve_no != 0) {
		if (params->passband)   free (params->passband);
		if (params->wavelength) free (params->wavelength);
		if (params->sigma)      free (params->sigma);
		if (params->hla)        free (params->hla);
		if (params->cla)        free (params->cla);
		if (params->x1a)        free (params->x1a);
		if (params->y1a)        free (params->y1a);
		if (params->x2a)        free (params->x2a);
		if (params->y2a)        free (params->y2a);
		if (params->el3)        free (params->el3);
		if (params->opsf)       free (params->opsf);
		if (params->levweight)  free (params->levweight);
	}

	if (params->spot1no != 0) {
		if (params->spot1lat)  free (params->spot1lat);
		if (params->spot1long) free (params->spot1long);
		if (params->spot1rad)  free (params->spot1rad);
		if (params->spot1temp) free (params->spot1temp);
	}

	if (params->spot2no != 0) {
		if (params->spot2lat)  free (params->spot2lat);
		if (params->spot2long) free (params->spot2long);
		if (params->spot2rad)  free (params->spot2rad);
		if (params->spot2temp) free (params->spot2temp);
	}

	if (curve_no != 0) {
		if (params->obs) {
			for (i = 0; i < curve_no; i++)
				phoebe_curve_free (params->obs[i]);
			free (params->obs);
		}
	}

	free (params);

	return SUCCESS;
}
