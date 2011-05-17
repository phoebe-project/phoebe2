#include <stdlib.h>
#include <stdio.h>

#include "phoebe_calculations.h"
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
	double   cla = param->CLA > 1e-6 ? param->CLA : 10.0;

	wd_wrlci (filename, param->MPAGE, param->NREF, param->MREF, param->IFSMV1, param->IFSMV2, param->ICOR1, param->ICOR2, param->LD,
			  param->JDPHS, param->HJD0, param->PERIOD, param->DPDT, param->PSHIFT, param->SIGMA, param->WEIGHTING, param->SEED,
			  param->HJDST, param->HJDSP, param->HJDIN, param->PHSTRT, param->PHSTOP, param->PHIN, param->PHNORM,
			  param->MODE, param->IPB, param->IFAT1, param->IFAT2, param->N1, param->N2, param->PERR0, param->DPERDT, param->THE, vunit,
			  param->E, param->SMA, param->F1, param->F2, vga, param->INCL, param->GR1, param->GR2, param->MET1,
			  tavh, tavc, param->ALB1, param->ALB2, param->PHSV, param->PCSV, param->RM, param->XBOL1, param->XBOL2, param->YBOL1, param->YBOL2,
			  param->IBAND, param->HLA, cla, param->X1A, param->X2A, param->Y1A, param->Y2A, param->EL3, param->OPSF, mzero, param->FACTOR, wla,
			  param->SPRIM, param->XLAT1, param->XLONG1, param->RADSP1, param->TEMSP1, param->SSEC, param->XLAT2, param->XLONG2, param->RADSP2, param->TEMSP2);

	return SUCCESS;
}

int create_dci_file (char *filename, void *pars)
{
	/*
	 * This function calls the auxiliary fortran subroutine wrdci from the
	 * WD library to create the DC input file. Please note that the data
	 * are passed to WD directly, they are not read out from the DCI file.
	 */

	int i;

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

	double *step, *wla, *sigma, *cla;

	int  rvno = params->rv1data + params->rv2data;
	int   cno = rvno + params->nlc;

	if (cno != 0) {
		wla = phoebe_malloc (cno * sizeof (*wla));
		sigma = phoebe_malloc (cno * sizeof (*sigma));
		cla = phoebe_malloc (cno * sizeof (*cla));

		for (i = 0; i < rvno; i++) {
			wla[i] = params->wavelength[i] / 10000.0;
			sigma[i] = params->sigma[i] / 100.0;
			cla[i] = params->cla[i] > 1e-6 ? params->cla[i] : 10.0;
		}

		for (i = rvno; i < cno; i++) {
			wla[i] = params->wavelength[i] / 10000.0;
			sigma[i] = params->sigma[i];
			cla[i] = params->cla[i] > 1e-6 ? params->cla[i] : 10.0;
		}
	}
	else {
		wla = NULL;
		sigma = NULL;
		cla = NULL;
	}

	step = phoebe_malloc (35 * sizeof (*step));
	for (i = 0; i < 35; i++)
		step[i] = params->step[i];

	step[14] /= 100.0;   /* vga */
	step[18] /= 10000.0; /* T1 */
	step[19] /= 10000.0; /* T2 */

	/* Spots */
	double spots_units_conversion_factor = phoebe_spots_units_to_wd_conversion_factor ();
	step[0] *= spots_units_conversion_factor;
	step[1] *= spots_units_conversion_factor;
	step[2] *= spots_units_conversion_factor;
	step[4] *= spots_units_conversion_factor;
	step[5] *= spots_units_conversion_factor;
	step[6] *= spots_units_conversion_factor;

	wd_wrdci (filename, step, params->tba, ifder, ifm, ifr, params->dclambda,
			  params->spot1src, params->spot1id, params->spot2src, params->spot2id,
			  params->rv1data, params->rv2data, params->nlc, k0, kdisk, params->symder, nppl,
			  params->refno, params->refswitch, params->spots1corotate, params->spots2corotate, params->rv1proximity, params->rv2proximity, params->ldmodel,
			  params->indep, params->hjd0, params->period, params->dpdt, params->pshift,
			  params->morph, params->cladec, params->ifat1, params->ifat2, params->n1f, params->n2f, params->n1c, params->n2c, params->perr0, params->dperdt, the, vunit,
			  params->ecc, params->sma, params->f1, params->f2, vga, params->incl, params->grb1, params->grb2, params->met1,
			  tavh, tavc, params->alb1, params->alb2, params->pot1, params->pot2, params->rm, params->xbol1, params->xbol2, params->ybol1, params->ybol2,
			  params->passband, params->hla, cla, params->x1a, params->x2a, params->y1a, params->y2a, params->el3, params->opsf, params->levweight, sigma, wla,
			  params->spot1no, params->spot1lat, params->spot1long, params->spot1rad, params->spot1temp, params->spot2no, params->spot2lat, params->spot2long, params->spot2rad, params->spot2temp,
			  params->knobs, params->indeps, params->fluxes, params->weights);

	if (wla) free (wla);
	if (sigma) free (sigma);
	if (cla) free (cla);

	free (step);

	return SUCCESS;
}

int intern_get_level_weighting_id (const char *type)
{
	/**
	 * intern_get_level_weighting_id:
	 * @type: level-weighting scheme
	 *
	 * Returns: WD code for the level-weighting scheme.
	 */

	int id = -1;

	if (strcmp (type, "None") == 0)               id = 0;
	if (strcmp (type, "Poissonian scatter") == 0) id = 1;
	if (strcmp (type, "Low light scatter") == 0)  id = 2;

	if (id == -1) {
		phoebe_lib_error ("level weighting type invalid, assuming Poissonian scatter.\n");
		return 1;
	}
	
	return id;
}

int phoebe_wd_model (char *phoebe_model)
{
	/*
	 * This function translates the Phoebe model string 
	 * into the WD model number.
	 * 
	 * Return values:
	 * 
	 *   -1 to 6, 99 for undefined
	 */
	if (strcmp (phoebe_model, "X-ray binary"                                         ) == 0) return -1;
	if (strcmp (phoebe_model, "Unconstrained binary system"                          ) == 0) return  0;
	if (strcmp (phoebe_model, "Overcontact binary of the W UMa type"                 ) == 0) return  1;
	if (strcmp (phoebe_model, "Detached binary"                                      ) == 0) return  2;
	if (strcmp (phoebe_model, "Overcontact binary not in thermal contact"            ) == 0) return  3;
	if (strcmp (phoebe_model, "Semi-detached binary, primary star fills Roche lobe"  ) == 0) return  4;
	if (strcmp (phoebe_model, "Semi-detached binary, secondary star fills Roche lobe") == 0) return  5;
	if (strcmp (phoebe_model, "Double contact binary"                                ) == 0) return  6;
	return 99;
}

int wd_lci_parameters_get (WD_LCI_parameters *params, int MPAGE, int curve)
{
	/*
	 * This function reads out all variables that build up the LCI file. It
	 * is written in such a way that any failed step can return the error
	 * code without memory leaks.
	 * 
	 * Return values:
	 * 
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_UNINITIALIZED_CURVE
	 *   ERROR_UNSUPPORTED_MPAGE  
	 *   ERROR_INVALID_LDLAW
	 *   SUCCESS
	 */

	int i, status;

	int lcno, rvno, spotno;

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
	 *   MPAGE = 5 .. plane-of-sky coordinates
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
	if (MPAGE == 5) {}/* plane-of-sky coordinates are not curve-dependent */;
	if (MPAGE != 1 && MPAGE != 2 && MPAGE != 5)
		return ERROR_UNSUPPORTED_MPAGE;

	params->MPAGE = MPAGE;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_switch"), &readout_bool);
	if (readout_bool == YES) params->MREF = 2; else params->MREF = 1;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_reflections"), &readout_int);
	params->NREF = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_corotate1"), &readout_bool);
	if (readout_bool) params->IFSMV1 = 0; else params->IFSMV1 = 1;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_corotate2"), &readout_bool);
	if (readout_bool) params->IFSMV2 = 0; else params->IFSMV2 = 1;

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
	params->HJD0 = readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_period"), &readout_dbl);
	params->PERIOD = readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dpdt"), &readout_dbl);
	params->DPDT = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pshift"), &readout_dbl);
	params->PSHIFT = readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_switch"), &readout_bool);
	if (readout_bool) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_sigma"), &readout_dbl);
		params->SIGMA = readout_dbl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_levweight"), &readout_str);
		if (strcmp (readout_str, "None") == 0)               params->WEIGHTING = 0;
		if (strcmp (readout_str, "Poissonian scatter") == 0) params->WEIGHTING = 1;
		if (strcmp (readout_str, "Low light scatter")  == 0) params->WEIGHTING = 2;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_synscatter_seed"), &readout_dbl);
		params->SEED = readout_dbl;
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
	params->MODE = phoebe_wd_model(readout_str);

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
	params->N1 = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize2"), &readout_int);
	params->N2 = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_perr0"), &readout_dbl);
	params->PERR0 = readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dperdt"), &readout_dbl);
	params->DPERDT = readout_dbl;

	/* THE applies only to X-ray binaries, but it isn't supported yet.        */
	params->THE    = 0.0;

	/* VUNIT is the internal radial velocity unit. This should be calculated  */
	/* rather than determined by the user, however it isn't written yet. It   */
	/* is thus hardcoded to the most obvious number, that is 100 km/s.        */
	params->VUNIT  = 100.0;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ecc"),      &readout_dbl);
	params->E = readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_sma"),      &readout_dbl);
	params->SMA = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_f1"),       &readout_dbl);
	params->F1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_f2"),       &readout_dbl);
	params->F2 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_vga"),      &readout_dbl);
	params->VGA = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_incl"),     &readout_dbl);
	params->INCL = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grb1"),     &readout_dbl);
	params->GR1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grb2"),     &readout_dbl);
	params->GR2 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg1"),    &readout_dbl);
	params->LOGG1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_logg2"),    &readout_dbl);
	params->LOGG2 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met1"),     &readout_dbl);
	params->MET1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_met2"),     &readout_dbl);
	params->MET2 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff1"),    &readout_dbl);
	params->TAVH = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_teff2"),    &readout_dbl);   
	params->TAVC = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_alb1"),     &readout_dbl);
	params->ALB1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_alb2"),     &readout_dbl);
	params->ALB2 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot1"),     &readout_dbl);
	params->PHSV = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_pot2"),     &readout_dbl);
	params->PCSV = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rm"),       &readout_dbl);
	params->RM = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_xbol1"), &readout_dbl);
	params->XBOL1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_xbol2"), &readout_dbl);
	params->XBOL2 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_ybol1"), &readout_dbl);
	params->YBOL1 = readout_dbl;
	
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_ybol2"), &readout_dbl);
	params->YBOL2 = readout_dbl;

	/* Wavelength-dependent parameters: */
	switch (MPAGE) {
		case 1:
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), curve, &filter);	
			passband = phoebe_passband_lookup (filter);
			if (!passband) {
				phoebe_lib_warning ("passband not set or invalid, reverting to Johnson V.\n");
				params->IBAND = 7;
				params->WLA   = 550.0;
			} else {
				status = wd_passband_id_lookup (&(params->IBAND), filter);
				if (status != SUCCESS) {
					phoebe_lib_error ("passband %s not supported by WD, aborting.\n", filter);
					return status;
				}
				params->WLA = passband->effwl;
			}

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hla"), curve, &readout_dbl);
			params->HLA = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cla"), curve, &readout_dbl);
			params->CLA = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx1"), curve, &readout_dbl);

			params->X1A = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx2"), curve, &readout_dbl);
			params->X2A = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy1"), curve, &readout_dbl);
			params->Y1A = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy2"), curve, &readout_dbl);
			params->Y2A = readout_dbl;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"), curve, &readout_dbl);
			params->EL3 = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_opsf"), curve, &readout_dbl);
			params->OPSF = readout_dbl;
		break;
		case 2:
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), curve, &filter);
			passband = phoebe_passband_lookup (filter);
			if (!passband) {
				phoebe_lib_warning ("passband not set or invalid, reverting to Johnson V.\n");
				params->IBAND = 7;
				params->WLA   = 550.0;
			} else {
				status = wd_passband_id_lookup (&(params->IBAND), filter);
				if (status != SUCCESS) {
					phoebe_lib_error ("passband %s not supported by WD, aborting.\n", filter);
					return status;
				}
				params->WLA   = passband->effwl;
			}

			params->HLA = 1.0;
			params->CLA = 1.0;

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx1"), curve, &readout_dbl);
			params->X1A = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx2"), curve, &readout_dbl);
			params->X2A = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy1"), curve, &readout_dbl);
			params->Y1A = readout_dbl;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy2"), curve, &readout_dbl);
			params->Y2A = readout_dbl;

			params->EL3 = 0.0;
			params->OPSF = 0.0;
		break;
		case 5:
			params->IBAND = 7;
			params->WLA   = 550.0;
			params->HLA   = 1.0;
			params->CLA   = 1.0;
			params->X1A   = 0.5;
			params->X2A   = 0.5;
			params->Y1A   = 0.5;
			params->Y2A   = 0.5;
			params->EL3   = 0.0;
			params->OPSF  = 0.0;
		break;
		default:
			/* We can't ever get here. */
		break;
	}

	/*
	 * MZERO and FACTOR variables set offsets in synthetic light curves. PHOEBE
	 * controls this by its own variables, so we hardcode these to 0 and 1.
	 */

	params->MZERO  = 0.0;
	params->FACTOR = 1.0;

	/* Spots: */
	{
		PHOEBE_array *active_spotindices;
		phoebe_active_spots_get (&spotno, &active_spotindices);
		double spots_units_conversion_factor = phoebe_spots_units_to_wd_conversion_factor ();

		/* Nullify arrays so that we can use phoebe_realloc immediately: */
		params->XLAT1 = NULL; params->XLONG1 = NULL; params->RADSP1 = NULL; params->TEMSP1 = NULL;
		params->XLAT2 = NULL; params->XLONG2 = NULL; params->RADSP2 = NULL; params->TEMSP2 = NULL;

		params->SPRIM = 0; params->SSEC = 0;
		for (i = 0; i < spotno; i++) {
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_source"), active_spotindices->val.iarray[i], &readout_int);

			if (readout_int == 1) {
				params->SPRIM++;

				params->XLAT1  = phoebe_realloc (params->XLAT1,  params->SPRIM * sizeof (*(params->XLAT1)));
				params->XLONG1 = phoebe_realloc (params->XLONG1, params->SPRIM * sizeof (*(params->XLONG1)));
				params->RADSP1 = phoebe_realloc (params->RADSP1, params->SPRIM * sizeof (*(params->RADSP1)));
				params->TEMSP1 = phoebe_realloc (params->TEMSP1, params->SPRIM * sizeof (*(params->TEMSP1)));

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude"), active_spotindices->val.iarray[i], &readout_dbl);
				params->XLAT1[params->SPRIM-1] = readout_dbl * spots_units_conversion_factor;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude"), active_spotindices->val.iarray[i], &readout_dbl);
				params->XLONG1[params->SPRIM-1] = readout_dbl * spots_units_conversion_factor;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius"),  active_spotindices->val.iarray[i], &readout_dbl);
				params->RADSP1[params->SPRIM-1] = readout_dbl * spots_units_conversion_factor;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor"), active_spotindices->val.iarray[i], &readout_dbl);
				params->TEMSP1[params->SPRIM-1] = readout_dbl;
			}
			else {
				params->SSEC++;

				params->XLAT2  = phoebe_realloc (params->XLAT2,  params->SSEC * sizeof (*(params->XLAT2)));
				params->XLONG2 = phoebe_realloc (params->XLONG2, params->SSEC * sizeof (*(params->XLONG2)));
				params->RADSP2 = phoebe_realloc (params->RADSP2, params->SSEC * sizeof (*(params->RADSP2)));
				params->TEMSP2 = phoebe_realloc (params->TEMSP2, params->SSEC * sizeof (*(params->TEMSP2)));

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude"), active_spotindices->val.iarray[i], &readout_dbl);
				params->XLAT2[params->SSEC-1] = readout_dbl * spots_units_conversion_factor;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude"), active_spotindices->val.iarray[i], &readout_dbl);
				params->XLONG2[params->SSEC-1] = readout_dbl * spots_units_conversion_factor;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius"),  active_spotindices->val.iarray[i], &readout_dbl);
				params->RADSP2[params->SSEC-1] = readout_dbl * spots_units_conversion_factor;
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor"), active_spotindices->val.iarray[i], &readout_dbl);
				params->TEMSP2[params->SSEC-1] = readout_dbl;
			}
		}
		if (spotno > 0)
			phoebe_array_free (active_spotindices);
	}
/*
	for (i = 0; i < params->SPRIM; i++)
		printf ("sprim: %lf %lf %lf %lf\n", params->XLAT1[i], params->XLONG1[i], params->RADSP1[i], params->TEMSP1[i]);
	for (i = 0; i < params->SSEC; i++)
		printf ("sprim: %lf %lf %lf %lf\n", params->XLAT2[i], params->XLONG2[i], params->RADSP2[i], params->TEMSP2[i]);
*/	
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
	pars->extinction = NULL;
	pars->levweight  = NULL;
	pars->spot1lat   = NULL;
	pars->spot1long  = NULL;
	pars->spot1rad   = NULL;
	pars->spot1temp  = NULL;
	pars->spot2lat   = NULL;
	pars->spot2long  = NULL;
	pars->spot2rad   = NULL;
	pars->spot2temp  = NULL;

	/* Finally, these are data arrays. We NULLify them as well: */
	pars->knobs      = NULL;
	pars->indeps     = NULL;
	pars->fluxes     = NULL;
	pars->weights    = NULL;

	return pars;
}

int wd_spots_parameters_get ()
{
	int i, spno, sp1no = 0, sp2no = 0, src, tbano = 0, tbacur = 1;
	bool *colat_tba, *long_tba, *rad_tba, *temp_tba;
	double colat, lon, rad, temp;
	double colat_min, lon_min, rad_min, temp_min;
	double colat_max, lon_max, rad_max, temp_max;
	double colat_step, lon_step, rad_step, temp_step;

	PHOEBE_parameter *par;
	PHOEBE_array *active_spots;

	/* Set all adjustment switches to 0: */
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_lat1"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_long1"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_rad1"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_temp1"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_lat2"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_long2"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_rad2"), 0);
	phoebe_parameter_set_tba (phoebe_parameter_lookup ("wd_spots_temp2"), 0);

	/* First test: are there any spots defined in the model: */
	phoebe_active_spots_get (&spno, &active_spots);
	if (spno == 0)
		return SUCCESS;

	/* Second test: let's check whether there are more than 2 spots marked    */
	/* for adjustment. Fill in arrays at the same time for subsequent usage.  */

	colat_tba = phoebe_malloc (spno * sizeof (*colat_tba));
	 long_tba = phoebe_malloc (spno * sizeof (*long_tba));
	  rad_tba = phoebe_malloc (spno * sizeof (*rad_tba));
	 temp_tba = phoebe_malloc (spno * sizeof (*temp_tba));

	for (i = 0; i < spno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_tba"), active_spots->val.iarray[i], &(colat_tba[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_tba"),  active_spots->val.iarray[i], &(long_tba[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_tba"),     active_spots->val.iarray[i], &(rad_tba[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_tba"), active_spots->val.iarray[i], &(temp_tba[i]));

		if (colat_tba[i] || long_tba[i] || rad_tba[i] || temp_tba[i])
			tbano++;
	}

	if (tbano == 0) {
		free (colat_tba); free (long_tba); free (rad_tba); free (temp_tba);
		return SUCCESS;
	}
	if (tbano > 2) {
		free (colat_tba); free (long_tba); free (rad_tba); free (temp_tba);
		return ERROR_DC_TOO_MANY_SPOTS_TBA;
	}

	/* Now we know there are spots in the model and we know that at least one */
	/* and not more than 2 of them are marked for adjustment. We make a 2nd   */
	/* pass through the list and copy the values to WD spot parameters.       */

	for (i = 0; i < spno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_source"), active_spots->val.iarray[i], &src);

		if (src == 1) {
			sp1no++;

			if (colat_tba[i] || long_tba[i] || rad_tba[i] || temp_tba[i]) {
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude"), active_spots->val.iarray[i], &colat);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_min"), active_spots->val.iarray[i], &colat_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_max"), active_spots->val.iarray[i], &colat_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_step"), active_spots->val.iarray[i], &colat_step);

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude"), active_spots->val.iarray[i], &lon);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_min"), active_spots->val.iarray[i], &lon_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_max"), active_spots->val.iarray[i], &lon_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_step"), active_spots->val.iarray[i], &lon_step);

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius"), active_spots->val.iarray[i], &rad);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_min"), active_spots->val.iarray[i], &rad_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_max"), active_spots->val.iarray[i], &rad_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_step"), active_spots->val.iarray[i], &rad_step);

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor"), active_spots->val.iarray[i], &temp);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_min"), active_spots->val.iarray[i], &temp_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_max"), active_spots->val.iarray[i], &temp_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_step"), active_spots->val.iarray[i], &temp_step);

				if (tbacur == 1) {
					if (colat_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_lat1");
						phoebe_parameter_set_value (par, colat);
						phoebe_parameter_set_min   (par, colat_min);
						phoebe_parameter_set_max   (par, colat_max);
						phoebe_parameter_set_step  (par, colat_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (long_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_long1");
						phoebe_parameter_set_value (par, lon);
						phoebe_parameter_set_min   (par, lon_min);
						phoebe_parameter_set_max   (par, lon_max);
						phoebe_parameter_set_step  (par, lon_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (rad_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_rad1");
						phoebe_parameter_set_value (par, rad);
						phoebe_parameter_set_min   (par, rad_min);
						phoebe_parameter_set_max   (par, rad_max);
						phoebe_parameter_set_step  (par, rad_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (temp_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_temp1");
						phoebe_parameter_set_value (par, temp);
						phoebe_parameter_set_min   (par, temp_min);
						phoebe_parameter_set_max   (par, temp_max);
						phoebe_parameter_set_step  (par, temp_step);
						phoebe_parameter_set_tba   (par, 1);
					}

					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot1src"), 1);
					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot1id"), sp1no);
				} else {
					if (colat_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_lat2");
						phoebe_parameter_set_value (par, colat);
						phoebe_parameter_set_min   (par, colat_min);
						phoebe_parameter_set_max   (par, colat_max);
						phoebe_parameter_set_step  (par, colat_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (long_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_long2");
						phoebe_parameter_set_value (par, lon);
						phoebe_parameter_set_min   (par, lon_min);
						phoebe_parameter_set_max   (par, lon_max);
						phoebe_parameter_set_step  (par, lon_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (rad_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_rad2");
						phoebe_parameter_set_value (par, rad);
						phoebe_parameter_set_min   (par, rad_min);
						phoebe_parameter_set_max   (par, rad_max);
						phoebe_parameter_set_step  (par, rad_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (temp_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_temp2");
						phoebe_parameter_set_value (par, temp);
						phoebe_parameter_set_min   (par, temp_min);
						phoebe_parameter_set_max   (par, temp_max);
						phoebe_parameter_set_step  (par, temp_step);
						phoebe_parameter_set_tba   (par, 1);
					}

					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot2src"), 1);
					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot2id"), sp1no);
				}
				tbacur++;
			}
		}
		else /* if (src == 2) */ {
			sp2no++;

			if (colat_tba[i] || long_tba[i] || rad_tba[i] || temp_tba[i]) {
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude"), active_spots->val.iarray[i], &colat);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_min"), active_spots->val.iarray[i], &colat_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_max"), active_spots->val.iarray[i], &colat_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude_step"), active_spots->val.iarray[i], &colat_step);

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude"), active_spots->val.iarray[i], &lon);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_min"), active_spots->val.iarray[i], &lon_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_max"), active_spots->val.iarray[i], &lon_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude_step"), active_spots->val.iarray[i], &lon_step);

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius"), active_spots->val.iarray[i], &rad);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_min"), active_spots->val.iarray[i], &rad_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_max"), active_spots->val.iarray[i], &rad_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius_step"), active_spots->val.iarray[i], &rad_step);

				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor"), active_spots->val.iarray[i], &temp);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_min"), active_spots->val.iarray[i], &temp_min);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_max"), active_spots->val.iarray[i], &temp_max);
				phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor_step"), active_spots->val.iarray[i], &temp_step);

				if (tbacur == 1) {
					if (colat_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_lat1");
						phoebe_parameter_set_value (par, colat);
						phoebe_parameter_set_min   (par, colat_min);
						phoebe_parameter_set_max   (par, colat_max);
						phoebe_parameter_set_step  (par, colat_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (long_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_long1");
						phoebe_parameter_set_value (par, lon);
						phoebe_parameter_set_min   (par, lon_min);
						phoebe_parameter_set_max   (par, lon_max);
						phoebe_parameter_set_step  (par, lon_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (rad_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_rad1");
						phoebe_parameter_set_value (par, rad);
						phoebe_parameter_set_min   (par, rad_min);
						phoebe_parameter_set_max   (par, rad_max);
						phoebe_parameter_set_step  (par, rad_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (temp_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_temp1");
						phoebe_parameter_set_value (par, temp);
						phoebe_parameter_set_min   (par, temp_min);
						phoebe_parameter_set_max   (par, temp_max);
						phoebe_parameter_set_step  (par, temp_step);
						phoebe_parameter_set_tba   (par, 1);
					}

					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot1src"), 2);
					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot1id"), sp2no);
				} else {
					if (colat_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_lat2");
						phoebe_parameter_set_value (par, colat);
						phoebe_parameter_set_min   (par, colat_min);
						phoebe_parameter_set_max   (par, colat_max);
						phoebe_parameter_set_step  (par, colat_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (long_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_long2");
						phoebe_parameter_set_value (par, lon);
						phoebe_parameter_set_min   (par, lon_min);
						phoebe_parameter_set_max   (par, lon_max);
						phoebe_parameter_set_step  (par, lon_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (rad_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_rad2");
						phoebe_parameter_set_value (par, rad);
						phoebe_parameter_set_min   (par, rad_min);
						phoebe_parameter_set_max   (par, rad_max);
						phoebe_parameter_set_step  (par, rad_step);
						phoebe_parameter_set_tba   (par, 1);
					}
					if (temp_tba[i]) {
						par = phoebe_parameter_lookup ("wd_spots_temp2");
						phoebe_parameter_set_value (par, temp);
						phoebe_parameter_set_min   (par, temp_min);
						phoebe_parameter_set_max   (par, temp_max);
						phoebe_parameter_set_step  (par, temp_step);
						phoebe_parameter_set_tba   (par, 1);
					}

					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot2src"), 2);
					phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_dc_spot2id"), sp2no);
				}
				tbacur++;
			}
		}
	}

	free (colat_tba); free (long_tba); free (rad_tba); free (temp_tba);
	phoebe_array_free (active_spots);

	return SUCCESS;
}

int wd_dci_parameters_get (WD_DCI_parameters *params, int *marked_tba)
{
	/**
	 * wd_dci_parameters_get:
	 * @params: pointer to the #WD_DCI_parameters placeholder
	 * @marked_tba: placeholder for the number of parameters marked for adjustment
	 * 
	 * Reads in DCI parameters. Be sure to allocate (and free)
	 * the passed WD_DCI_parameters structure (and its substructures) before
	 * (and after) you call this function.
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

	char *pars[] = {
		"wd_spots_lat1",
		"wd_spots_long1",
		"wd_spots_rad1",
		"wd_spots_temp1",
		"wd_spots_lat2",
		"wd_spots_long2",
		"wd_spots_rad2",
		"wd_spots_temp2",
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

	int i, j, status, idx = 0, bandidx = 0;
	int lcno, rvno, spotno;
	int active_lcno, active_rvno, active_cno;
	PHOEBE_array *active_lcindices, *active_rvindices;
	double rvfactor;

	int readout_int;
	bool readout_bool;
	double readout_dbl;
	const char *readout_str;

	int rv1index = -1;
	int rv2index = -1;

	PHOEBE_column_type master_indep, dtype;

	/* DC features 35 adjustable parameters and we initialize such arrays: */
	int *tba;
	double *step;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lcno"), &lcno);
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rvno"), &rvno);

	active_lcindices = phoebe_active_curves_get (PHOEBE_CURVE_LC);
	active_rvindices = phoebe_active_curves_get (PHOEBE_CURVE_RV);
	if (active_lcindices) active_lcno = active_lcindices->dim; else active_lcno = 0;
	if (active_rvindices) active_rvno = active_rvindices->dim; else active_rvno = 0;

	if (active_rvno > 2) {
		phoebe_lib_warning ("More than 2 RV curves are currently not supported.\n");
		phoebe_array_free (active_lcindices);
		phoebe_array_free (active_rvindices);
		return ERROR_DC_TOO_MANY_RVS;
	}

	active_cno = active_lcno + active_rvno;

	*marked_tba = 0;

	/* If there are no observations defined, bail out: */
	if (active_cno == 0) return ERROR_MINIMIZER_NO_CURVES;

/***************************************************************/
	phoebe_debug ("LC#: %d\n", lcno);
	phoebe_debug ("     %d active:\t", active_lcno);
	for (i = 0; i < active_lcno-1; i++)
		phoebe_debug ("%d, ", active_lcindices->val.iarray[i]);
	if (active_lcno > 0)
		phoebe_debug ("%d\n", active_lcindices->val.iarray[i]);
	phoebe_debug ("RV#: %d\n", rvno);
	phoebe_debug ("     %d active:\t", active_rvno);
	for (i = 0; i < active_rvno-1; i++)
		phoebe_debug ("%d, ", active_rvindices->val.iarray[i]);
	if (active_rvno > 0)
		phoebe_debug ("%d\n", active_rvindices->val.iarray[i]);
/***************************************************************/

	/* Allocate memory: */
	 tba = phoebe_malloc (35 * sizeof ( *tba));
	step = phoebe_malloc (35 * sizeof (*step));

	/*
	 * Read in TBA states and step-sizes; note the '!' in front of the tba
	 * readout function; that's because WD sets 1 for /kept/ parameters and
	 * 0 for parameters that are marked for adjustment.
	 */

	wd_spots_parameters_get ();

	for (i = 0; i < 35; i++) {
		if (i == 29) { tba[i] = !FALSE; continue; } /* reserved WD channel */
		phoebe_parameter_get_tba (phoebe_parameter_lookup (pars[i]), &readout_bool);
		tba[i] = (!readout_bool);
		phoebe_parameter_get_step (phoebe_parameter_lookup (pars[i]), &step[i]);
		if (i > 29)
			*marked_tba += active_lcno * (1-tba[i]);
		else
			*marked_tba += 1-tba[i];
	}

	params->tba  = tba;
	params->step = step;

	/* Check for presence of RV and LC data: */
	params->rv1data = FALSE; params->rv2data = FALSE;
	for (i = 0; i < active_rvno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), active_rvindices->val.iarray[i], &readout_str);
		status = phoebe_column_get_type (&dtype, readout_str);
		if (status != SUCCESS) return status;

		switch (dtype) {
			case PHOEBE_COLUMN_PRIMARY_RV:
				if (params->rv1data) break;
				params->rv1data = TRUE;
				rv1index = active_rvindices->val.iarray[i];
			break;
			case PHOEBE_COLUMN_SECONDARY_RV:
				if (params->rv2data) break;
				params->rv2data = TRUE;
				rv2index = active_rvindices->val.iarray[i];
			break;
			default:
				phoebe_lib_error ("exception handler invoked in wd_dci_parameters_get(), please report this!\n");
				return ERROR_EXCEPTION_HANDLER_INVOKED;
		}
	}
	params->nlc = active_lcno;

/***************************************************************/
	phoebe_debug ("Primary RV:   %d\n", params->rv1data);
	phoebe_debug ("     index:   %d\n", rv1index);
	phoebe_debug ("Secondary RV: %d\n", params->rv2data);
	phoebe_debug ("       index: %d\n", rv2index);
/***************************************************************/

	/* DC-related parameters: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_lambda"), &readout_dbl);
	params->dclambda = readout_dbl;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_symder_switch"), &readout_bool);
	params->symder = readout_bool;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_coarsesize1"), &readout_int);
	params->n1c = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_coarsesize2"), &readout_int);
	params->n2c = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize1"), &readout_int);
	params->n1f = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_grid_finesize2"), &readout_int);
	params->n2f = readout_int;

	/* Reflection effect: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_switch"), &readout_bool);
	if (readout_bool) params->refswitch = 2; else params->refswitch = 1;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_reffect_reflections"), &readout_int);
	params->refno = readout_int;

	/* Eclipse/proximity effect: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_proximity_rv1_switch"), &readout_bool);
	params->rv1proximity = readout_bool;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_proximity_rv2_switch"), &readout_bool);
	params->rv2proximity = readout_bool;
	
	/* Limb darkening effect: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_model"), &readout_str);
	params->ldmodel = phoebe_ld_model_type (readout_str);
	if (params->ldmodel == LD_LAW_INVALID)
		return ERROR_INVALID_LDLAW;

	/* Morphology: */
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
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_indep"), &readout_str);
	status = phoebe_column_get_type (&master_indep, readout_str);
	if (status != SUCCESS) return status;

	if (master_indep == PHOEBE_COLUMN_HJD)
		params->indep = 1;
	else if (master_indep == PHOEBE_COLUMN_PHASE)
		params->indep = 2;
	else
		return ERROR_INVALID_INDEP;

	/* Luminosity decoupling: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_usecla_switch"), &readout_bool);
	params->cladec = readout_bool;

	/* Model atmosphere switches: */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_atm1_switch"), &readout_bool);
	params->ifat1 = readout_bool;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_atm2_switch"), &readout_bool);
	params->ifat2 = readout_bool;

	/* Model parameters: */
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

	/* Finite integration time; must do after system parameters. */
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cadence_switch"), &readout_bool);
	if (readout_bool) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cadence_rate"), &(params->nph));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cadence"), &readout_dbl);
		params->delph = readout_dbl/86400/params->period;
	}
	else {
		params->nph = 1;
		params->delph = 1.0;
	}

	/* Passband-dependent parameters: */
	{
	int index;
	PHOEBE_passband *passband;

	params->passband   = phoebe_malloc (active_cno * sizeof (*(params->passband)));
	params->wavelength = phoebe_malloc (active_cno * sizeof (*(params->wavelength)));
	params->sigma      = phoebe_malloc (active_cno * sizeof (*(params->sigma)));
	params->hla        = phoebe_malloc (active_cno * sizeof (*(params->hla)));
	params->cla        = phoebe_malloc (active_cno * sizeof (*(params->cla)));
	params->x1a        = phoebe_malloc (active_cno * sizeof (*(params->x1a)));
	params->y1a        = phoebe_malloc (active_cno * sizeof (*(params->y1a)));
	params->x2a        = phoebe_malloc (active_cno * sizeof (*(params->x2a)));
	params->y2a        = phoebe_malloc (active_cno * sizeof (*(params->y2a)));
	params->el3        = phoebe_malloc (active_cno * sizeof (*(params->el3)));
	params->opsf       = phoebe_malloc (active_cno * sizeof (*(params->opsf)));
	params->extinction = phoebe_malloc (active_cno * sizeof (*(params->extinction)));
	params->levweight  = phoebe_malloc (active_cno * sizeof (*(params->levweight)));

	for (i = 0; i < active_rvno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_dep"), active_rvindices->val.iarray[i], &readout_str);
		status = phoebe_column_get_type (&dtype, readout_str);
		if (status != SUCCESS) {
			wd_dci_parameters_free (params);
			return status;
		}

		if (dtype == PHOEBE_COLUMN_SECONDARY_RV && active_rvno == 2) index = 1; else index = 0;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_filter"), active_rvindices->val.iarray[i], &readout_str);
		passband = phoebe_passband_lookup (readout_str);

		if (!passband) {
			wd_dci_parameters_free (params);
			return ERROR_PASSBAND_INVALID;
		}

		status = wd_passband_id_lookup (&(params->passband[index]), readout_str);
		if (status != SUCCESS) {
			wd_dci_parameters_free (params);
			phoebe_lib_error ("passband %s not supported by WD, aborting.\n", readout_str);
			return status;
		}
		params->wavelength[index] = passband->effwl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_rv_sigma"), active_rvindices->val.iarray[i], &(params->sigma[index]));
		params->hla[index]        = 10.0;
		params->cla[index]        = 10.0;
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx1"), active_rvindices->val.iarray[i], &(params->x1a[index]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy1"), active_rvindices->val.iarray[i], &(params->y1a[index]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvx2"), active_rvindices->val.iarray[i], &(params->x2a[index]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_rvy2"), active_rvindices->val.iarray[i], &(params->y2a[index]));
		params->el3[index]        = 0.0;
		params->opsf[index]       = 0.0;
		params->extinction[index] = 0.0;
		params->levweight[index]  = 0;
	}

	for (i = active_rvno; i < active_cno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_filter"), active_lcindices->val.iarray[i-active_rvno], &readout_str);

		passband = phoebe_passband_lookup (readout_str);
		if (!passband) {
			free (params->passband); free (params->wavelength); free (params->sigma);
			free (params->hla);      free (params->cla);        free (params->x1a);
			free (params->y1a);      free (params->x2a);        free (params->y2a);
			free (params->el3);      free (params->opsf);       free (params->extinction);
			free (params->levweight);
			return ERROR_PASSBAND_INVALID;
		}

		status = wd_passband_id_lookup (&(params->passband[i]), readout_str);
		if (status != SUCCESS) {
			phoebe_lib_error ("passband %s not supported by WD, aborting.\n", readout_str);
			return status;
		}
		params->wavelength[i] = passband->effwl;

		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_sigma"),     active_lcindices->val.iarray[i-active_rvno], &(params->sigma[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_hla"),          active_lcindices->val.iarray[i-active_rvno], &(params->hla[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_cla"),          active_lcindices->val.iarray[i-active_rvno], &(params->cla[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx1"),      active_lcindices->val.iarray[i-active_rvno], &(params->x1a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy1"),      active_lcindices->val.iarray[i-active_rvno], &(params->y1a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcx2"),      active_lcindices->val.iarray[i-active_rvno], &(params->x2a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_ld_lcy2"),      active_lcindices->val.iarray[i-active_rvno], &(params->y2a[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3"),          active_lcindices->val.iarray[i-active_rvno], &(params->el3[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_opsf"),         active_lcindices->val.iarray[i-active_rvno], &(params->opsf[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_extinction"),   active_lcindices->val.iarray[i-active_rvno], &(params->extinction[i]));
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_lc_levweight"), active_lcindices->val.iarray[i-active_rvno], &readout_str);

		params->levweight[i]  = intern_get_level_weighting_id (readout_str);
	}
	}

	/* Spot parameters: */

	//phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_no"), &spotno);
	PHOEBE_array *active_spotindices;
	phoebe_active_spots_get (&spotno, &active_spotindices);
	double spots_units_conversion_factor = phoebe_spots_units_to_wd_conversion_factor();

	/* Nullify arrays so that we can use phoebe_realloc immediately: */
	params->spot1lat = NULL; params->spot1long = NULL; params->spot1rad = NULL; params->spot1temp = NULL;
	params->spot2lat = NULL; params->spot2long = NULL; params->spot2rad = NULL; params->spot2temp = NULL;

	params->spot1no = 0; params->spot2no = 0;
	for (i = 0; i < spotno; i++) {
		phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_source"), active_spotindices->val.iarray[i], &readout_int);

		if (readout_int == 1) {
			params->spot1no++;

			params->spot1lat  = phoebe_realloc (params->spot1lat,  params->spot1no * sizeof (*(params->spot1lat)));
			params->spot1long = phoebe_realloc (params->spot1long, params->spot1no * sizeof (*(params->spot1long)));
			params->spot1rad  = phoebe_realloc (params->spot1rad,  params->spot1no * sizeof (*(params->spot1rad)));
			params->spot1temp = phoebe_realloc (params->spot1temp, params->spot1no * sizeof (*(params->spot1temp)));

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude"), active_spotindices->val.iarray[i], &readout_dbl);
			params->spot1lat[params->spot1no-1] = readout_dbl * spots_units_conversion_factor;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude"), active_spotindices->val.iarray[i], &readout_dbl);
			params->spot1long[params->spot1no-1] = readout_dbl * spots_units_conversion_factor;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius"),  active_spotindices->val.iarray[i], &readout_dbl);
			params->spot1rad[params->spot1no-1] = readout_dbl * spots_units_conversion_factor;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor"), active_spotindices->val.iarray[i], &readout_dbl);
			params->spot1temp[params->spot1no-1] = readout_dbl;
		}
		else {
			params->spot2no++;

			params->spot2lat  = phoebe_realloc (params->spot2lat,  params->spot2no * sizeof (*(params->spot2lat)));
			params->spot2long = phoebe_realloc (params->spot2long, params->spot2no * sizeof (*(params->spot2long)));
			params->spot2rad  = phoebe_realloc (params->spot2rad,  params->spot2no * sizeof (*(params->spot2rad)));
			params->spot2temp = phoebe_realloc (params->spot2temp, params->spot2no * sizeof (*(params->spot2temp)));

			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_colatitude"), active_spotindices->val.iarray[i], &readout_dbl);
			params->spot2lat[params->spot2no-1] = readout_dbl * spots_units_conversion_factor;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_longitude"), active_spotindices->val.iarray[i], &readout_dbl);
			params->spot2long[params->spot2no-1] = readout_dbl * spots_units_conversion_factor;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_radius"),  active_spotindices->val.iarray[i], &readout_dbl);
			params->spot2rad[params->spot2no-1] = readout_dbl * spots_units_conversion_factor;
			phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_tempfactor"), active_spotindices->val.iarray[i], &readout_dbl);
			params->spot2temp[params->spot2no-1] = readout_dbl;
		}
	}

	if (spotno > 0)
		phoebe_array_free (active_spotindices);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot1src"), &readout_int);
	params->spot1src = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot2src"), &readout_int);
	params->spot2src = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot1id"), &readout_int);
	params->spot1id = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_dc_spot2id"), &readout_int);
	params->spot2id = readout_int;

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_corotate1"), &readout_bool);
	params->spots1corotate = (!readout_bool);

	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_corotate2"), &readout_bool);
	params->spots2corotate = (!readout_bool);

	/* Observed data: */
	{
		params->knobs = phoebe_malloc ( (active_cno+1) * sizeof (*(params->knobs)));
		params->knobs[0] = 0;
		bandidx++;

		if (params->rv1data) {
			PHOEBE_curve *rv = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, rv1index);
			if (!rv) {
				wd_dci_parameters_free (params);
				return ERROR_FILE_NOT_FOUND;
			}

			if (rv->wtype == PHOEBE_COLUMN_SIGMA) rvfactor = 100.0; else rvfactor = 1.0;
			phoebe_curve_transform (rv, master_indep, PHOEBE_COLUMN_PRIMARY_RV, PHOEBE_COLUMN_WEIGHT);

			params->indeps  = phoebe_realloc (params->indeps,  (idx+rv->indep->dim+1) * sizeof (*(params->indeps)));
			params->fluxes  = phoebe_realloc (params->fluxes,  (idx+rv->indep->dim+1) * sizeof (*(params->fluxes)));
			params->weights = phoebe_realloc (params->weights, (idx+rv->indep->dim+1) * sizeof (*(params->weights)));

			for (i = 0; i < rv->indep->dim; i++) {
				params->indeps[idx+i]  = rv->indep->val[i];
				params->fluxes[idx+i]  = rv->dep->val[i] / 100.0;
				params->weights[idx+i] = rvfactor*rvfactor*rv->weight->val[i];
			}

			params->knobs[bandidx] = (idx += rv->indep->dim);
			bandidx++;

			phoebe_curve_free (rv);
		}
		if (params->rv2data) {
			PHOEBE_curve *rv = phoebe_curve_new_from_pars (PHOEBE_CURVE_RV, rv2index);;
			if (!rv) {
				wd_dci_parameters_free (params);
				return ERROR_FILE_NOT_FOUND;
			}

			if (rv->wtype == PHOEBE_COLUMN_SIGMA) rvfactor = 100.0; else rvfactor = 1.0;
			phoebe_curve_transform (rv, master_indep, PHOEBE_COLUMN_SECONDARY_RV, PHOEBE_COLUMN_WEIGHT);

			params->indeps  = phoebe_realloc (params->indeps,  (idx+rv->indep->dim+1) * sizeof (*(params->indeps)));
			params->fluxes  = phoebe_realloc (params->fluxes,  (idx+rv->indep->dim+1) * sizeof (*(params->fluxes)));
			params->weights = phoebe_realloc (params->weights, (idx+rv->indep->dim+1) * sizeof (*(params->weights)));

			for (i = 0; i < rv->indep->dim; i++) {
				params->indeps[idx+i]  = rv->indep->val[i];
				params->fluxes[idx+i]  = rv->dep->val[i] / 100.0;
				params->weights[idx+i] = rvfactor*rvfactor*rv->weight->val[i];
			}

			params->knobs[bandidx] = (idx += rv->indep->dim);
			bandidx++;

			phoebe_curve_free (rv);
		}
		for (i = active_rvno; i < active_cno; i++) {
			PHOEBE_curve *lc = phoebe_curve_new_from_pars (PHOEBE_CURVE_LC, active_lcindices->val.iarray[i-active_rvno]);
			if (!lc) {
				wd_dci_parameters_free (params);
				return ERROR_FILE_NOT_FOUND;
			}

			phoebe_curve_transform (lc, master_indep, PHOEBE_COLUMN_FLUX, PHOEBE_COLUMN_WEIGHT);

			if (params->extinction[i] > 0)
				apply_extinction_correction (lc, -params->extinction[i]);

			params->indeps  = phoebe_realloc (params->indeps,  (idx+lc->indep->dim+1) * sizeof (*(params->indeps)));
			params->fluxes  = phoebe_realloc (params->fluxes,  (idx+lc->indep->dim+1) * sizeof (*(params->fluxes)));
			params->weights = phoebe_realloc (params->weights, (idx+lc->indep->dim+1) * sizeof (*(params->weights)));

			for (j = 0; j < lc->indep->dim; j++) {
				params->indeps[idx+j]  = lc->indep->val[j];
				params->fluxes[idx+j]  = lc->dep->val[j];
				params->weights[idx+j] = lc->weight->val[j];
			}

			params->knobs[bandidx] = (idx += lc->indep->dim);
			bandidx++;

			phoebe_curve_free (lc);
		}
	}

	phoebe_array_free (active_lcindices);
	phoebe_array_free (active_rvindices);

	return SUCCESS;
}

int wd_dci_parameters_free (WD_DCI_parameters *params)
{
	/*
	 * This function frees the arrays of the structure and the structure itself.
	 */

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
		if (params->extinction) free (params->extinction);
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
		if (params->knobs)   free (params->knobs);
		if (params->indeps)  free (params->indeps);
		if (params->fluxes)  free (params->fluxes);
		if (params->weights) free (params->weights);
	}

	free (params);

	return SUCCESS;
}
