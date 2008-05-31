%{
#include <stdio.h>
#include <string.h>

int _spotsno = 0;
int _spotidx = 0;
%}

%option prefix="phoebe_legacy"
%option noyywrap
%option nostdinit

DIGIT [0-9]

%%

\"{DIGIT}+.{DIGIT}+\"	{ double val; sscanf (yytext, "\"%lf\"", &val); fprintf (phoebe_legacyout, "%lf", val); }

NAME			fprintf (phoebe_legacyout, "phoebe_name");
MODEL			fprintf (phoebe_legacyout, "phoebe_model");
LCNO			fprintf (phoebe_legacyout, "phoebe_lcno");
RVNO			fprintf (phoebe_legacyout, "phoebe_rvno");
SPECNO			fprintf (phoebe_legacyout, "phoebe_spno");
MNORM			fprintf (phoebe_legacyout, "phoebe_mnorm");
BINNING			fprintf (phoebe_legacyout, "phoebe_bins_switch");
BINVAL			fprintf (phoebe_legacyout, "phoebe_bins");
REDDENING		fprintf (phoebe_legacyout, "phoebe_ie_switch");
REDDENING_R		fprintf (phoebe_legacyout, "phoebe_ie_factor");
REDDENING_E		fprintf (phoebe_legacyout, "phoebe_ie_excess");
LCCOL1{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_indep[%d]", idx); }
LCCOL2{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_dep[%d]", idx); }
LCCOL3{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_indweight[%d]", idx); }
LCFN{DIGIT}+	{ int idx; char *ptr = yytext+ 4; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_filename[%d]", idx); }
LCSIG{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_sigma[%d]", idx); }
LCFLT{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_filter[%d]", idx); }
WEIGHT{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_lc_levweight[%d]", idx); }
RVCOL1{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_rv_indep[%d]", idx); }
RVCOL2{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_rv_dep[%d]", idx); }
RVCOL3{DIGIT}+	{ int idx; char *ptr = yytext+ 6; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_rv_indweight[%d]", idx); }
RVFN{DIGIT}+	{ int idx; char *ptr = yytext+ 4; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_rv_filename[%d]", idx); }
RVSIG{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_rv_sigma[%d]", idx); }
RVFLT{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_rv_filter[%d]", idx); }
HJD0[ \t]+		fprintf (phoebe_legacyout, "phoebe_hjd0.VAL");
HJD0_ADJ		fprintf (phoebe_legacyout, "phoebe_hjd0.ADJ");
HJD0_DEL		fprintf (phoebe_legacyout, "phoebe_hjd0.STEP");
HJD0_MIN		fprintf (phoebe_legacyout, "phoebe_hjd0.MIN");
HJD0_MAX		fprintf (phoebe_legacyout, "phoebe_hjd0.MAX");
PERIOD[ \t]+	fprintf (phoebe_legacyout, "phoebe_period.VAL");
PERIOD_ADJ		fprintf (phoebe_legacyout, "phoebe_period.ADJ");
PERIOD_DEL		fprintf (phoebe_legacyout, "phoebe_period.STEP");
PERIOD_MIN		fprintf (phoebe_legacyout, "phoebe_period.MIN");
PERIOD_MAX		fprintf (phoebe_legacyout, "phoebe_period.MAX");
DPDT[ \t]+		fprintf (phoebe_legacyout, "phoebe_dpdt.VAL");
DPDT_ADJ		fprintf (phoebe_legacyout, "phoebe_dpdt.ADJ");
DPDT_DEL		fprintf (phoebe_legacyout, "phoebe_dpdt.STEP");
DPDT_MIN		fprintf (phoebe_legacyout, "phoebe_dpdt.MIN");
DPDT_MAX		fprintf (phoebe_legacyout, "phoebe_dpdt.MAX");
PSHIFT[ \t]+	fprintf (phoebe_legacyout, "phoebe_pshift.VAL");
PSHIFT_ADJ		fprintf (phoebe_legacyout, "phoebe_pshift.ADJ");
PSHIFT_DEL		fprintf (phoebe_legacyout, "phoebe_pshift.STEP");
PSHIFT_MIN		fprintf (phoebe_legacyout, "phoebe_pshift.MIN");
PSHIFT_MAX		fprintf (phoebe_legacyout, "phoebe_pshift.MAX");
SMA[ \t]+		fprintf (phoebe_legacyout, "phoebe_sma.VAL");
SMA_ADJ			fprintf (phoebe_legacyout, "phoebe_sma.ADJ");
SMA_DEL			fprintf (phoebe_legacyout, "phoebe_sma.STEP");
SMA_MIN			fprintf (phoebe_legacyout, "phoebe_sma.MIN");
SMA_MAX			fprintf (phoebe_legacyout, "phoebe_sma.MAX");
RM[ \t]+		fprintf (phoebe_legacyout, "phoebe_rm.VAL");
RM_ADJ			fprintf (phoebe_legacyout, "phoebe_rm.ADJ");
RM_DEL			fprintf (phoebe_legacyout, "phoebe_rm.STEP");
RM_MIN			fprintf (phoebe_legacyout, "phoebe_rm.MIN");
RM_MAX			fprintf (phoebe_legacyout, "phoebe_rm.MAX");
INCL[ \t]+		fprintf (phoebe_legacyout, "phoebe_incl.VAL");
INCL_ADJ		fprintf (phoebe_legacyout, "phoebe_incl.ADJ");
INCL_DEL		fprintf (phoebe_legacyout, "phoebe_incl.STEP");
INCL_MIN		fprintf (phoebe_legacyout, "phoebe_incl.MIN");
INCL_MAX		fprintf (phoebe_legacyout, "phoebe_incl.MAX");
VGA[ \t]+		fprintf (phoebe_legacyout, "phoebe_vga.VAL");
VGA_ADJ			fprintf (phoebe_legacyout, "phoebe_vga.ADJ");
VGA_DEL			fprintf (phoebe_legacyout, "phoebe_vga.STEP");
VGA_MIN			fprintf (phoebe_legacyout, "phoebe_vga.MIN");
VGA_MAX			fprintf (phoebe_legacyout, "phoebe_vga.MAX");
TAVH[ \t]+		fprintf (phoebe_legacyout, "phoebe_teff1.VAL");
TAVH_ADJ		fprintf (phoebe_legacyout, "phoebe_teff1.ADJ");
TAVH_DEL		fprintf (phoebe_legacyout, "phoebe_teff1.STEP");
TAVH_MIN		fprintf (phoebe_legacyout, "phoebe_teff1.MIN");
TAVH_MAX		fprintf (phoebe_legacyout, "phoebe_teff1.MAX");
TAVC[ \t]+		fprintf (phoebe_legacyout, "phoebe_teff2.VAL");
TAVC_ADJ		fprintf (phoebe_legacyout, "phoebe_teff2.ADJ");
TAVC_DEL		fprintf (phoebe_legacyout, "phoebe_teff2.STEP");
TAVC_MIN		fprintf (phoebe_legacyout, "phoebe_teff2.MIN");
TAVC_MAX		fprintf (phoebe_legacyout, "phoebe_teff2.MAX");
PHSV[ \t]+		fprintf (phoebe_legacyout, "phoebe_pot1.VAL");
PHSV_ADJ		fprintf (phoebe_legacyout, "phoebe_pot1.ADJ");
PHSV_DEL		fprintf (phoebe_legacyout, "phoebe_pot1.STEP");
PHSV_MIN		fprintf (phoebe_legacyout, "phoebe_pot1.MIN");
PHSV_MAX		fprintf (phoebe_legacyout, "phoebe_pot1.MAX");
PCSV[ \t]+		fprintf (phoebe_legacyout, "phoebe_pot2.VAL");
PCSV_ADJ		fprintf (phoebe_legacyout, "phoebe_pot2.ADJ");
PCSV_DEL		fprintf (phoebe_legacyout, "phoebe_pot2.STEP");
PCSV_MIN		fprintf (phoebe_legacyout, "phoebe_pot2.MIN");
PCSV_MAX		fprintf (phoebe_legacyout, "phoebe_pot2.MAX");
LOGG1[ \t]+		fprintf (phoebe_legacyout, "phoebe_logg1.VAL");
LOGG1_ADJ		fprintf (phoebe_legacyout, "phoebe_logg1.ADJ");
LOGG1_DEL		fprintf (phoebe_legacyout, "phoebe_logg1.STEP");
LOGG1_MIN		fprintf (phoebe_legacyout, "phoebe_logg1.MIN");
LOGG1_MAX		fprintf (phoebe_legacyout, "phoebe_logg1.MAX");
LOGG2[ \t]+		fprintf (phoebe_legacyout, "phoebe_logg2.VAL");
LOGG2_ADJ		fprintf (phoebe_legacyout, "phoebe_logg2.ADJ");
LOGG2_DEL		fprintf (phoebe_legacyout, "phoebe_logg2.STEP");
LOGG2_MIN		fprintf (phoebe_legacyout, "phoebe_logg2.MIN");
LOGG2_MAX		fprintf (phoebe_legacyout, "phoebe_logg2.MAX");
MET1[ \t]+		fprintf (phoebe_legacyout, "phoebe_met1.VAL");
MET1_ADJ		fprintf (phoebe_legacyout, "phoebe_met1.ADJ");
MET1_DEL		fprintf (phoebe_legacyout, "phoebe_met1.STEP");
MET1_MIN		fprintf (phoebe_legacyout, "phoebe_met1.MIN");
MET1_MAX		fprintf (phoebe_legacyout, "phoebe_met1.MAX");
MET2[ \t]+		fprintf (phoebe_legacyout, "phoebe_met2.VAL");
MET2_ADJ		fprintf (phoebe_legacyout, "phoebe_met2.ADJ");
MET2_DEL		fprintf (phoebe_legacyout, "phoebe_met2.STEP");
MET2_MIN		fprintf (phoebe_legacyout, "phoebe_met2.MIN");
MET2_MAX		fprintf (phoebe_legacyout, "phoebe_met2.MAX");
E[ \t]+			fprintf (phoebe_legacyout, "phoebe_ecc.VAL");
E_ADJ			fprintf (phoebe_legacyout, "phoebe_ecc.ADJ");
E_DEL			fprintf (phoebe_legacyout, "phoebe_ecc.STEP");
E_MIN			fprintf (phoebe_legacyout, "phoebe_ecc.MIN");
E_MAX			fprintf (phoebe_legacyout, "phoebe_ecc.MAX");
PERR0[ \t]+		fprintf (phoebe_legacyout, "phoebe_perr0.VAL");
PERR0_ADJ		fprintf (phoebe_legacyout, "phoebe_perr0.ADJ");
PERR0_DEL		fprintf (phoebe_legacyout, "phoebe_perr0.STEP");
PERR0_MIN		fprintf (phoebe_legacyout, "phoebe_perr0.MIN");
PERR0_MAX		fprintf (phoebe_legacyout, "phoebe_perr0.MAX");
DPERDT[ \t]+	fprintf (phoebe_legacyout, "phoebe_dperdt.VAL");
DPERDT_ADJ		fprintf (phoebe_legacyout, "phoebe_dperdt.ADJ");
DPERDT_DEL		fprintf (phoebe_legacyout, "phoebe_dperdt.STEP");
DPERDT_MIN		fprintf (phoebe_legacyout, "phoebe_dperdt.MIN");
DPERDT_MAX		fprintf (phoebe_legacyout, "phoebe_dperdt.MAX");
F1[ \t]+		fprintf (phoebe_legacyout, "phoebe_f1.VAL");
F1_ADJ			fprintf (phoebe_legacyout, "phoebe_f1.ADJ");
F1_DEL			fprintf (phoebe_legacyout, "phoebe_f1.STEP");
F1_MIN			fprintf (phoebe_legacyout, "phoebe_f1.MIN");
F1_MAX			fprintf (phoebe_legacyout, "phoebe_f1.MAX");
F2[ \t]+		fprintf (phoebe_legacyout, "phoebe_f2.VAL");
F2_ADJ			fprintf (phoebe_legacyout, "phoebe_f2.ADJ");
F2_DEL			fprintf (phoebe_legacyout, "phoebe_f2.STEP");
F2_MIN			fprintf (phoebe_legacyout, "phoebe_f2.MIN");
F2_MAX			fprintf (phoebe_legacyout, "phoebe_f2.MAX");
HLALC{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_hla[%d].VAL", idx); }
HLA_ADJ			fprintf (phoebe_legacyout, "phoebe_hla.ADJ");
HLA_DEL			fprintf (phoebe_legacyout, "phoebe_hla.STEP");
CLALC{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_cla[%d].VAL", idx); }
CLA_ADJ			fprintf (phoebe_legacyout, "phoebe_cla.ADJ");
CLA_DEL			fprintf (phoebe_legacyout, "phoebe_cla.STEP");
EL3{DIGIT}+		{ int idx; char *ptr = yytext+ 3; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_el3[%d].VAL", idx); }
EL3_ADJ			fprintf (phoebe_legacyout, "phoebe_el3.ADJ");
EL3_DEL			fprintf (phoebe_legacyout, "phoebe_el3.STEP");
OPSF{DIGIT}+	{ int idx; char *ptr = yytext+ 4; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_opsf[%d].VAL", idx); }
OPSF_ADJ		fprintf (phoebe_legacyout, "phoebe_opsf.ADJ");
OPSF_DEL		fprintf (phoebe_legacyout, "phoebe_opsf.STEP");
IPB_ON			fprintf (phoebe_legacyout, "phoebe_usecla_switch");
NREF_ON			fprintf (phoebe_legacyout, "phoebe_reffect_switch");
NREF_VAL		fprintf (phoebe_legacyout, "phoebe_reffect_reflections");
IFAT1_ON		fprintf (phoebe_legacyout, "phoebe_atm1_switch");
IFAT2_ON		fprintf (phoebe_legacyout, "phoebe_atm2_switch");
NOISE_ON		fprintf (phoebe_legacyout, "phoebe_synscatter_switch");
NOISE_TYPE		fprintf (phoebe_legacyout, "phoebe_synscatter_levweight");
NOISE_VAL		fprintf (phoebe_legacyout, "phoebe_synscatter_sigma");
SEED_VAL		fprintf (phoebe_legacyout, "phoebe_synscatter_seed");
ICOR1_ON		fprintf (phoebe_legacyout, "phoebe_proximity_rv1_switch");
ICOR2_ON		fprintf (phoebe_legacyout, "phoebe_proximity_rv2_switch");
LD				fprintf (phoebe_legacyout, "phoebe_ld_model");
XBOL1			fprintf (phoebe_legacyout, "phoebe_ld_xbol1");
XBOL2			fprintf (phoebe_legacyout, "phoebe_ld_xbol2");
YBOL1			fprintf (phoebe_legacyout, "phoebe_ld_ybol1");
YBOL2			fprintf (phoebe_legacyout, "phoebe_ld_ybol2");
X1ALC{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_lcx1[%d].VAL", idx); }
X2ALC{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_lcx2[%d].VAL", idx); }
Y1ALC{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_lcy1[%d].VAL", idx); }
Y2ALC{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_lcy2[%d].VAL", idx); }
X1ARV{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_rvx1[%d].VAL", idx); }
X2ARV{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_rvx2[%d].VAL", idx); }
Y1ARV{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_rvy1[%d].VAL", idx); }
Y2ARV{DIGIT}+	{ int idx; char *ptr = yytext+ 5; sscanf (ptr, "%d", &idx); fprintf (phoebe_legacyout, "phoebe_ld_rvy2[%d].VAL", idx); }
X1A_DEL			fprintf (phoebe_legacyout, "phoebe_ld_lcx1.STEP");
X2A_DEL			fprintf (phoebe_legacyout, "phoebe_ld_lcx2.STEP");
X1A_ADJ			fprintf (phoebe_legacyout, "phoebe_ld_lcx1.ADJ");
X2A_ADJ			fprintf (phoebe_legacyout, "phoebe_ld_lcx2.ADJ");
ALB1[ \t]+		fprintf (phoebe_legacyout, "phoebe_alb1.VAL");
ALB1_ADJ		fprintf (phoebe_legacyout, "phoebe_alb1.ADJ");
ALB1_DEL		fprintf (phoebe_legacyout, "phoebe_alb1.STEP");
ALB1_MIN		fprintf (phoebe_legacyout, "phoebe_alb1.MIN");
ALB1_MAX		fprintf (phoebe_legacyout, "phoebe_alb1.MAX");
ALB2[ \t]+		fprintf (phoebe_legacyout, "phoebe_alb2.VAL");
ALB2_ADJ		fprintf (phoebe_legacyout, "phoebe_alb2.ADJ");
ALB2_DEL		fprintf (phoebe_legacyout, "phoebe_alb2.STEP");
ALB2_MIN		fprintf (phoebe_legacyout, "phoebe_alb2.MIN");
ALB2_MAX		fprintf (phoebe_legacyout, "phoebe_alb2.MAX");
GR1[ \t]+		fprintf (phoebe_legacyout, "phoebe_grb1.VAL");
GR1_ADJ			fprintf (phoebe_legacyout, "phoebe_grb1.ADJ");
GR1_DEL			fprintf (phoebe_legacyout, "phoebe_grb1.STEP");
GR1_MIN			fprintf (phoebe_legacyout, "phoebe_grb1.MIN");
GR1_MAX			fprintf (phoebe_legacyout, "phoebe_grb1.MAX");
GR2[ \t]+		fprintf (phoebe_legacyout, "phoebe_grb2.VAL");
GR2_ADJ			fprintf (phoebe_legacyout, "phoebe_grb2.ADJ");
GR2_DEL			fprintf (phoebe_legacyout, "phoebe_grb2.STEP");
GR2_MIN			fprintf (phoebe_legacyout, "phoebe_grb2.MIN");
GR2_MAX			fprintf (phoebe_legacyout, "phoebe_grb2.MAX");
NSPOTSPRIM[^\n]*"\n" {
				int idx;
				sscanf (yytext, "NSPOTSPRIM = %d", &idx);
				_spotsno += idx;
				_spotidx = idx;
				}
NSPOTSSEC[^\n]*"\n" {
				int idx;
				sscanf (yytext, "NSPOTSSEC = %d", &idx);
				_spotsno += idx;
				fprintf (phoebe_legacyout, "phoebe_spots_no = %d\n", _spotsno);
				}
XLAT1{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 5;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_colatitude[%d].VAL", idx);
				}
XLONG1{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 6;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_source[%d] = 1\n", idx);
				fprintf (phoebe_legacyout, "phoebe_spots_longitude[%d].VAL", idx);
				}
RADSP1{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 6;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_radius[%d].VAL", idx);
				}
TEMSP1{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 6;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_tempfactor[%d].VAL", idx);
				}
XLAT2{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 5;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_colatitude[%d].VAL", _spotidx+idx);
				}
XLONG2{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 6;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_source[%d] = 2\n", _spotidx+idx);
				fprintf (phoebe_legacyout, "phoebe_spots_longitude[%d].VAL", _spotidx+idx);
				}
RADSP2{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 6;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_radius[%d].VAL", _spotidx+idx);
				}
TEMSP2{DIGIT}+	{
				int idx;
				char *ptr = yytext+ 6;
				sscanf (ptr, "%d", &idx);
				fprintf (phoebe_legacyout, "phoebe_spots_tempfactor[%d].VAL", _spotidx+idx);
				}
N1_VAL			fprintf (phoebe_legacyout, "phoebe_grid_finesize1");
N2_VAL			fprintf (phoebe_legacyout, "phoebe_grid_finesize2");
N1L_VAL			fprintf (phoebe_legacyout, "phoebe_grid_coarsesize1");
N2L_VAL			fprintf (phoebe_legacyout, "phoebe_grid_coarsesize2");

JDPHS_TIME" "*"="" "*{DIGIT}+	{ int idx; sscanf (yytext, "JDPHS_TIME = %d", &idx); if (idx == 1) fprintf (phoebe_legacyout, "phoebe_indep = \"Time (HJD)\""); else fprintf (phoebe_legacyout, "phoebe_indep = \"Phase\""); }
EL3_FLUX" "*"="" "*{DIGIT}+	{ int idx; sscanf (yytext, "EL3_FLUX = %d", &idx); if (idx == 0) fprintf (phoebe_legacyout, "phoebe_el3_units = \"Total light\""); else fprintf (phoebe_legacyout, "phoebe_el3_units = \"Flux\""); }

"Time"				fprintf (phoebe_legacyout, "Time (HJD)");

"Weight (int)"		fprintf (phoebe_legacyout, "Standard weight");
"Weight (real)"		fprintf (phoebe_legacyout, "Standard weight");
"Absolute error"	fprintf (phoebe_legacyout, "Standard deviation");

"RV in km/s"		fprintf (phoebe_legacyout, "[CHANGED IN 0.30]");
"RV in 100 km/s"	fprintf (phoebe_legacyout, "[CHANGED IN 0.30]");

"360nm (U)"			fprintf (phoebe_legacyout, "Johnson:U");
"440nm (B)"			fprintf (phoebe_legacyout, "Johnson:B");
"550nm (V)"			fprintf (phoebe_legacyout, "Johnson:V");
"700nm (R)"			fprintf (phoebe_legacyout, "Johnson:R");
"900nm (I)"			fprintf (phoebe_legacyout, "Johnson:I");
"1250nm (J)"		fprintf (phoebe_legacyout, "Johnson:J");
"1620nm (H)"		fprintf (phoebe_legacyout, "Johnson:H");
"2200nm (K)"		fprintf (phoebe_legacyout, "Johnson:K");
"3400nm (L)"		fprintf (phoebe_legacyout, "Johnson:L");
"5000nm (M)"		fprintf (phoebe_legacyout, "Johnson:M");
"10200nm (N)"		fprintf (phoebe_legacyout, "Johnson:N");

"647nm (Rc)"		fprintf (phoebe_legacyout, "Cousins:R");
"786nm (Ic)"		fprintf (phoebe_legacyout, "Cousins:I");

"419nm (Bt)"		fprintf (phoebe_legacyout, "Hipparcos:BT");
"505nm (Hp)"		fprintf (phoebe_legacyout, "Hipparcos:Hp");
"523nm (Vt)"		fprintf (phoebe_legacyout, "Hipparcos:VT");

"350nm (u)"			fprintf (phoebe_legacyout, "Stromgren:u");
"411nm (v)"			fprintf (phoebe_legacyout, "Stromgren:v");
"467nm (b)"			fprintf (phoebe_legacyout, "Stromgren:b");
"547nm (y)"			fprintf (phoebe_legacyout, "Stromgren:y");

"861nm (RVIJ)"		fprintf (phoebe_legacyout, "[CHANGED IN 0.30]");

"Linear Cosine Law"	fprintf (phoebe_legacyout, "Linear cosine law");
"Logarithmic Law"	fprintf (phoebe_legacyout, "Logarithmic law");
"Square Root Law"	fprintf (phoebe_legacyout, "Square root law");

"General binary system (no constraints)"	fprintf (phoebe_legacyout, "Unconstrained binary system");

"No Level-Dependent Weighting"		fprintf (phoebe_legacyout, "None");
"No level-dependent weighting"		fprintf (phoebe_legacyout, "None");
"Scatter Scales With Square Root"	fprintf (phoebe_legacyout, "Poissonian scatter");
"Scatter Scales With Light Level"	fprintf (phoebe_legacyout, "Low light scatter");
"Scatter scales with square root"	fprintf (phoebe_legacyout, "Poissonian scatter");
"Scatter scales with light level"	fprintf (phoebe_legacyout, "Low light scatter");

XLAMDA_VAL[^\n]*"\n"
ISYM_ON[^\n]*"\n"
LC_PHSTRT[^\n]*"\n"
LC_PHEND[^\n]*"\n"
LC_VERTICES[^\n]*"\n"
LC_INDEP[^\n]*"\n"
LC_DEP[^\n]*"\n"
PHNORM_VAL[^\n]*"\n"
FACTOR_VAL[^\n]*"\n"
RV_PHSTRT[^\n]*"\n"
RV_PHEND[^\n]*"\n"
RV_VERTICES[^\n]*"\n"
RV_INDEP[^\n]*"\n"
RV_DEP[^\n]*"\n"
MODELLGG_ON[^\n]*"\n"
THE_VAL[^\n]*"\n"

%%

extern FILE *phoebe_legacyout;
