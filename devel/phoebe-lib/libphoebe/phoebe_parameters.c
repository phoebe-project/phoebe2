#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

int phoebe_init_parameters ()
{
	/*
	 * This (and only this) function declares parameters that are used in
	 * PHOEBE; there is no GUI connection or any other plug-in connection in
	 * this function, only native PHOEBE parameters. PHOEBE drivers, such
	 * as the scripter or the GUI, have to initialize the parameters by
	 * analogous functions.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	/* **********************   Model parameters   ************************** */

	phoebe_parameter_add ("phoebe_name",                 "Common name of the binary",                  KIND_PARAMETER,  NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,       "");
	phoebe_parameter_add ("phoebe_indep",                "Independent modeling variable",              KIND_MENU,       NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,       "Phase");
	phoebe_parameter_add ("phoebe_model",                "Morphological constraints",                  KIND_MENU,       NULL, 0.0, 0.0, 0.0, NO, TYPE_STRING,       "Unconstrained binary system");

	phoebe_parameter_add ("phoebe_lcno",                 "Number of observed light curves",            KIND_MODIFIER,   NULL, 0.0, 0.0, 0.0, NO, TYPE_INT,          0);
	phoebe_parameter_add ("phoebe_rvno",                 "Number of observed RV curves",               KIND_MODIFIER,   NULL, 0.0, 0.0, 0.0, NO, TYPE_INT,          0);
	phoebe_parameter_add ("phoebe_spno",                 "Number of observed spectra",                 KIND_MODIFIER,   NULL, 0.0, 0.0, 0.0, NO, TYPE_INT,          0);

	/* **********************   Model constraints   ************************* */

	phoebe_parameter_add ("phoebe_asini_switch",         "(a sin i) is kept constant",                 KIND_SWITCH,     NULL,   0.0,    0.0,    0.0,  NO, TYPE_BOOL,         NO);
	phoebe_parameter_add ("phoebe_asini",                "(a sin i) constant",                         KIND_PARAMETER,  NULL,   0.0,   1E10,    0.0,  NO, TYPE_DOUBLE,       10.0);

	phoebe_parameter_add ("phoebe_cindex_switch",        "Use the color-index constraint",             KIND_SWITCH,     NULL,   0.0,    0.0,    0.0,  NO, TYPE_BOOL,         NO);
	phoebe_parameter_add ("phoebe_cindex",               "Color-index values",                         KIND_PARAMETER,  "phoebe_lcno",  -100,    100,   1e-2,  NO, TYPE_DOUBLE_ARRAY, 1.0);

	phoebe_parameter_add ("phoebe_msc1_switch",          "Main-sequence constraint for star 1",        KIND_PARAMETER,  NULL,     0,      0,      0,  NO, TYPE_BOOL,         NO);
	phoebe_parameter_add ("phoebe_msc2_switch",          "Main-sequence constraint for star 2",        KIND_PARAMETER,  NULL,     0,      0,      0,  NO, TYPE_BOOL,         NO);

	/* ***********************   Data parameters   ************************** */

	phoebe_parameter_add ("phoebe_lc_filename",          "Observed LC data filename",                  KIND_PARAMETER,  "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	phoebe_parameter_add ("phoebe_lc_sigma",             "Observed LC data standard deviation",        KIND_PARAMETER,  "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_DOUBLE_ARRAY, 0.01);
	phoebe_parameter_add ("phoebe_lc_filter",            "Observed LC data filter",                    KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "440nm (B)");
	phoebe_parameter_add ("phoebe_lc_indep",             "Observed LC data independent variable",      KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Time (HJD)");
	phoebe_parameter_add ("phoebe_lc_dep",               "Observed LC data dependent variable",        KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Magnitude");
	phoebe_parameter_add ("phoebe_lc_indweight",         "Observed LC data individual weighting",      KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Standard deviation");
	phoebe_parameter_add ("phoebe_lc_levweight",         "Observed LC data level weighting",           KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Poissonian scatter");
	phoebe_parameter_add ("phoebe_lc_active",            "Observed LC data is used",                   KIND_SWITCH,     "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_BOOL_ARRAY,    YES);

	phoebe_parameter_add ("phoebe_rv_filename",          "Observed RV data filename",                  KIND_PARAMETER,  "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	phoebe_parameter_add ("phoebe_rv_sigma",             "Observed RV data standard deviation",        KIND_PARAMETER,  "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_DOUBLE_ARRAY, 1.0);
	phoebe_parameter_add ("phoebe_rv_filter",            "Observed RV data filter",                    KIND_MENU,       "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "550nm (V)");
	phoebe_parameter_add ("phoebe_rv_indep",             "Observed RV data independent variable",      KIND_MENU,       "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Time (HJD)");
	phoebe_parameter_add ("phoebe_rv_dep",               "Observed RV data dependent variable",        KIND_MENU,       "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Primary RV");
	phoebe_parameter_add ("phoebe_rv_indweight",         "Observed RV data individual weighting",      KIND_MENU,       "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Standard deviation");
	phoebe_parameter_add ("phoebe_rv_active",            "Observed RV data is used",                   KIND_SWITCH,     "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_BOOL_ARRAY,    YES);

	phoebe_parameter_add ("phoebe_mnorm",                "Flux-normalizing magnitude",                 KIND_PARAMETER,  NULL,   0.0,    0.0,    0.0, NO, TYPE_DOUBLE,       10.0);

	phoebe_parameter_add ("phoebe_bins_switch",          "Data binning",                               KIND_SWITCH,     NULL,   0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	phoebe_parameter_add ("phoebe_bins",                 "Number of bins",                             KIND_PARAMETER,  NULL,   0.0,    0.0,    0.0, NO, TYPE_INT,           100);

	phoebe_parameter_add ("phoebe_ie_switch",            "Interstellar extinction (reddening)",        KIND_SWITCH,     NULL,   0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	phoebe_parameter_add ("phoebe_ie_factor",            "Interstellar extinction coefficient",        KIND_PARAMETER,  NULL,   0.0,    0.0,    0.0, NO, TYPE_DOUBLE,        3.1);
	phoebe_parameter_add ("phoebe_ie_excess",            "Interstellar extinction color excess value", KIND_PARAMETER,  NULL,   0.0,    0.0,    0.0, NO, TYPE_DOUBLE,        0.0);

	phoebe_parameter_add ("phoebe_proximity_rv1_switch", "Proximity effects for primary star RV",      KIND_SWITCH,     NULL,   0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_proximity_rv2_switch", "Proximity effects for secondary star RV",    KIND_SWITCH,     NULL,   0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);

	/* **********************   System parameters   ************************* */

	phoebe_parameter_add ("phoebe_hjd0",                 "Origin of HJD time",                         KIND_ADJUSTABLE, NULL, -1E10,   1E10, 0.0001, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_period",               "Orbital period in days",                     KIND_ADJUSTABLE, NULL,   0.0,   1E10, 0.0001, NO, TYPE_DOUBLE,        1.0);
	phoebe_parameter_add ("phoebe_dpdt",                 "First time derivative of period (days/day)", KIND_ADJUSTABLE, NULL,  -1.0,    1.0,   1E-6, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_pshift",               "Phase shift",                                KIND_ADJUSTABLE, NULL,  -0.5,    0.5,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_sma",                  "Semi-major axis in solar radii",             KIND_ADJUSTABLE, NULL,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,       10.0);
	phoebe_parameter_add ("phoebe_rm",                   "Mass ratio (secondary over primary)",        KIND_ADJUSTABLE, NULL,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,        1.0);
	phoebe_parameter_add ("phoebe_incl",                 "Inclination in degrees",                     KIND_ADJUSTABLE, NULL,   0.0,  180.0,   0.01, NO, TYPE_DOUBLE,       80.0);
	phoebe_parameter_add ("phoebe_vga",                  "Center-of-mass velocity in km/s",            KIND_ADJUSTABLE, NULL,  -1E3,    1E3,    1.0, NO, TYPE_DOUBLE,        0.0);

	/* ********************   Component parameters   ************************ */

	phoebe_parameter_add ("phoebe_teff1",                "Primary star effective temperature in K",    KIND_ADJUSTABLE, NULL,  3500,  50000,     10, NO, TYPE_DOUBLE,     6000.0);
	phoebe_parameter_add ("phoebe_teff2",                "Secondary star effective temperature in K",  KIND_ADJUSTABLE, NULL,  3500,  50000,     10, NO, TYPE_DOUBLE,     6000.0);
	phoebe_parameter_add ("phoebe_pot1",                 "Primary star surface potential",             KIND_ADJUSTABLE, NULL,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,       10.0);
	phoebe_parameter_add ("phoebe_pot2",                 "Secondary star surface potential",           KIND_ADJUSTABLE, NULL,   0.0,   1E10,   0.01, NO, TYPE_DOUBLE,       10.0);
	phoebe_parameter_add ("phoebe_logg1",                "Primary star surface potential",             KIND_ADJUSTABLE, NULL,   0.0,   10.0,   0.01, NO, TYPE_DOUBLE,        4.3);
	phoebe_parameter_add ("phoebe_logg2",                "Primary star surface potential",             KIND_ADJUSTABLE, NULL,   0.0,   10.0,   0.01, NO, TYPE_DOUBLE,        4.3);
	phoebe_parameter_add ("phoebe_met1",                 "Primary star metallicity",                   KIND_ADJUSTABLE, NULL, -10.0,   10.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_met2",                 "Secondary star metallicity",                 KIND_ADJUSTABLE, NULL, -10.0,   10.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_f1",                   "Primary star synchronicity parameter",       KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        1.0);
	phoebe_parameter_add ("phoebe_f2",                   "Secondary star synchronicity parameter",     KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        1.0);
	phoebe_parameter_add ("phoebe_alb1",                 "Primary star surface albedo",                KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.6);
	phoebe_parameter_add ("phoebe_alb2",                 "Secondary star surface albedo",              KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.6);
	phoebe_parameter_add ("phoebe_grb1",                 "Primary star gravity brightening",           KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.32);
	phoebe_parameter_add ("phoebe_grb2",                 "Primary star gravity brightening",           KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.32);

	/* **********************   Orbit parameters   ************************** */

	phoebe_parameter_add ("phoebe_ecc",                  "Orbital eccentricity",                       KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_perr0",                "Argument of periastron",                     KIND_ADJUSTABLE, NULL,   0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_dperdt",               "First time derivative of periastron",        KIND_ADJUSTABLE, NULL,  -1.0,    1.0,   1E-6, NO, TYPE_DOUBLE,        0.0);

	/* *********************   Surface parameters   ************************* */

	phoebe_parameter_add ("phoebe_hla",                  "LC primary star flux leveler",               KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY, 12.566371);
	phoebe_parameter_add ("phoebe_cla",                  "LC secondary star flux leveler",             KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY, 12.566371);
	phoebe_parameter_add ("phoebe_opsf",                 "Third light contribution",                   KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);

	phoebe_parameter_add ("phoebe_passband_mode",        "Passband treatment mode",                    KIND_MENU,       NULL,  0.0,    0.0,    0.0, NO, TYPE_STRING,        "Interpolation");
	phoebe_parameter_add ("phoebe_atm1_switch",          "Use Kurucz's models for primary star",       KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_atm2_switch",          "Use Kurucz's models for secondary star",     KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_reffect_switch",       "Detailed reflection effect",                 KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	phoebe_parameter_add ("phoebe_reffect_reflections",  "Number of detailed reflections",             KIND_PARAMETER,  NULL,    2,     10,      1, NO, TYPE_INT,             2);

	phoebe_parameter_add ("phoebe_usecla_switch",        "Decouple CLAs from temperature",             KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);

	/* ********************   Extrinsic parameters   ************************ */

	phoebe_parameter_add ("phoebe_el3_units",            "Units of third light",                       KIND_MENU,       NULL,  0.0,    0.0,    0.0, NO, TYPE_STRING,        "Total light");
	phoebe_parameter_add ("phoebe_el3",                  "Third light contribution",                   KIND_ADJUSTABLE, NULL,  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);
	phoebe_parameter_add ("phoebe_extinction",           "Interstellar extinction coefficient",        KIND_ADJUSTABLE, NULL,  0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);

	/* *********************   Fitting parameters   ************************* */

	phoebe_parameter_add ("phoebe_grid_finesize1",       "Fine grid size on primary star",             KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,            20);
	phoebe_parameter_add ("phoebe_grid_finesize2",       "Fine grid size on secondary star",           KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,            20);
	phoebe_parameter_add ("phoebe_grid_coarsesize1",     "Coarse grid size on primary star",           KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,             5);
	phoebe_parameter_add ("phoebe_grid_coarsesize2",     "Coarse grid size on secondary star",         KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,             5);

	phoebe_parameter_add ("phoebe_compute_hla_switch",   "Compute passband (HLA) levels",              KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_compute_vga_switch",   "Compute gamma velocity",                     KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);

	/* **********************   DC fit parameters   ************************* */

	phoebe_parameter_add ("phoebe_dc_symder_switch",     "Should symmetrical DC derivatives be used",  KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_dc_lambda",            "Levenberg-Marquardt multiplier for DC",      KIND_PARAMETER,  NULL,  0.0,    1.0,   1e-3, NO, TYPE_DOUBLE,       1e-3);

	phoebe_parameter_add ("phoebe_dc_spot1src",          "Adjusted spot 1 source (at which star is the spot)", KIND_PARAMETER, NULL, 1,   2, 1, NO, TYPE_INT, 1);
	phoebe_parameter_add ("phoebe_dc_spot2src",          "Adjusted spot 2 source (at which star is the spot)", KIND_PARAMETER, NULL, 1,   2, 1, NO, TYPE_INT, 2);
	phoebe_parameter_add ("phoebe_dc_spot1id",           "Adjusted spot 1 id (which spot is to be adjusted)",  KIND_PARAMETER, NULL, 1, 100, 1, NO, TYPE_INT, 1);
	phoebe_parameter_add ("phoebe_dc_spot2id",           "Adjusted spot 2 id (which spot is to be adjusted)",  KIND_PARAMETER, NULL, 1, 100, 1, NO, TYPE_INT, 1);

	/* *******************   Perturbations parameters   ********************* */

	phoebe_parameter_add ("phoebe_ld_model",             "Limb darkening model",                               KIND_MENU,      NULL,      0,      0,      0, NO, TYPE_STRING, "Logarithmic law");
	phoebe_parameter_add ("phoebe_ld_xbol1",             "Primary star bolometric LD coefficient x",           KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_ybol1",             "Secondary star bolometric LD coefficient x",         KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_xbol2",             "Primary star bolometric LD coefficient y",           KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_ybol2",             "Secondary star bolometric LD coefficient y",         KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_lcx1",              "Primary star bandpass LD coefficient x",             KIND_ADJUSTABLE, "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_lcx2",              "Secondary star bandpass LD coefficient x",           KIND_ADJUSTABLE, "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_lcy1",              "Primary star bandpass LD coefficient y",             KIND_PARAMETER,  "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_lcy2",              "Secondary star bandpass LD coefficient y",           KIND_PARAMETER,  "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvx1",              "Primary RV bandpass LD coefficient x",               KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvx2",              "Secondary RV bandpass LD coefficient x",             KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvy1",              "Primary RV bandpass LD coefficient y",               KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvy2",              "Secondary RV bandpass LD coefficient y",             KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);

	phoebe_parameter_add ("phoebe_spots_no1",            "Number of spots on primary star",                    KIND_PARAMETER,  NULL,    0,    100,      1, NO, TYPE_INT,            0);
	phoebe_parameter_add ("phoebe_spots_no2",            "Number of spots on secondary star",                  KIND_PARAMETER,  NULL,    0,    100,      1, NO, TYPE_INT,            0);
	phoebe_parameter_add ("phoebe_spots_move1",          "Spots on primary star move in longitude",            KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,         YES);
	phoebe_parameter_add ("phoebe_spots_move2",          "Spots on secondary star move in longitude",          KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,         YES);
	phoebe_parameter_add ("phoebe_spots_lat1",           "Latitude of the spot on primary star",               KIND_PARAMETER,  "phoebe_spots_no1",  0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	phoebe_parameter_add ("phoebe_spots_long1",          "Longitude of the spot on primary star",              KIND_PARAMETER,  "phoebe_spots_no1",  0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	phoebe_parameter_add ("phoebe_spots_rad1",           "Radius of the spot on primary star",                 KIND_PARAMETER,  "phoebe_spots_no1",  0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.2);
	phoebe_parameter_add ("phoebe_spots_temp1",          "Temperature of the spot on primary star",            KIND_PARAMETER,  "phoebe_spots_no1",  0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.9);
	phoebe_parameter_add ("phoebe_spots_lat2",           "Latitude of the spot on secondary star",             KIND_PARAMETER,  "phoebe_spots_no2",  0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	phoebe_parameter_add ("phoebe_spots_long2",          "Longitude of the spot on secondary star",            KIND_PARAMETER,  "phoebe_spots_no2",   0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.0);
	phoebe_parameter_add ("phoebe_spots_rad2",           "Radius of the spot on secondary star",               KIND_PARAMETER,  "phoebe_spots_no2",  0.0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.2);
	phoebe_parameter_add ("phoebe_spots_temp2",          "Temperature of the spot on secondary star",          KIND_PARAMETER,  "phoebe_spots_no2",  0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.9);

	/* *********************   Utilities parameters   *********************** */

	phoebe_parameter_add ("phoebe_synscatter_switch",    "Synthetic scatter",                                  KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,          NO);
	phoebe_parameter_add ("phoebe_synscatter_sigma",     "Synthetic scatter standard deviation",               KIND_PARAMETER,  NULL,  0.0,  100.0,   0.01, NO, TYPE_DOUBLE,      0.01);
	phoebe_parameter_add ("phoebe_synscatter_seed",      "Synthetic scatter seed",                             KIND_PARAMETER,  NULL,  1E8,    1E9,      1, NO, TYPE_DOUBLE,     1.5E8);
	phoebe_parameter_add ("phoebe_synscatter_levweight", "Synthetic scatter weighting",                        KIND_MENU,       NULL,    0,      0,      0, NO, TYPE_STRING, "Poissonian scatter");

	return SUCCESS;
}

int phoebe_init_parameter_options ()
{
	/*
	 * This function adds options to all KIND_MENU parameters. In principle
	 * all calls to phoebe_parameter_add_option () function should be checked
	 * for return value, but since the function issues a warning in case a
	 * qualifier does not contain a menu, it is not really necessary.
	 *
	 * Return values:
	 *
	 *   SUCCESS
	 */

	int i;
	PHOEBE_parameter *par;
	char *passband_str;

	par = phoebe_parameter_lookup ("phoebe_indep");
	phoebe_parameter_add_option (par, "Time (HJD)");
	phoebe_parameter_add_option (par, "Phase");

	par = phoebe_parameter_lookup ("phoebe_model");
	phoebe_parameter_add_option (par, "X-ray binary");
	phoebe_parameter_add_option (par, "Unconstrained binary system");
	phoebe_parameter_add_option (par, "Overcontact binary of the W UMa type");
	phoebe_parameter_add_option (par, "Detached binary");
	phoebe_parameter_add_option (par, "Overcontact binary not in thermal contact");
	phoebe_parameter_add_option (par, "Semi-detached binary, primary star fills Roche lobe");
	phoebe_parameter_add_option (par, "Semi-detached binary, secondary star fills Roche lobe");
	phoebe_parameter_add_option (par, "Double contact binary");

	par = phoebe_parameter_lookup ("phoebe_lc_filter");
	for (i = 0; i < PHOEBE_passbands_no; i++) {
		passband_str = concatenate_strings (PHOEBE_passbands[i]->set, ":", PHOEBE_passbands[i]->name, NULL);
		phoebe_parameter_add_option (par, passband_str);
		free (passband_str);
	}

	par = phoebe_parameter_lookup ("phoebe_lc_indep");
	phoebe_parameter_add_option (par, "Time (HJD)");
	phoebe_parameter_add_option (par, "Phase");

	par = phoebe_parameter_lookup ("phoebe_lc_dep");
	phoebe_parameter_add_option (par, "Magnitude");
	phoebe_parameter_add_option (par, "Flux");

	par = phoebe_parameter_lookup ("phoebe_lc_indweight");
	phoebe_parameter_add_option (par, "Standard weight");
	phoebe_parameter_add_option (par, "Standard deviation");
	phoebe_parameter_add_option (par, "Unavailable");

	par = phoebe_parameter_lookup ("phoebe_lc_levweight");
	phoebe_parameter_add_option (par, "No level-dependent weighting");
	phoebe_parameter_add_option (par, "Poissonian scatter");
	phoebe_parameter_add_option (par, "Low light scatter");

	par = phoebe_parameter_lookup ("phoebe_rv_filter");
	for (i = 0; i < PHOEBE_passbands_no; i++) {
		passband_str = concatenate_strings (PHOEBE_passbands[i]->set, ":", PHOEBE_passbands[i]->name, NULL);
		phoebe_parameter_add_option (par, passband_str);
		free (passband_str);
	}

	par = phoebe_parameter_lookup ("phoebe_rv_indep");
	phoebe_parameter_add_option (par, "Time (HJD)");
	phoebe_parameter_add_option (par, "Phase");

	par = phoebe_parameter_lookup ("phoebe_rv_dep");
	phoebe_parameter_add_option (par, "Primary RV");
	phoebe_parameter_add_option (par, "Secondary RV");

	par = phoebe_parameter_lookup ("phoebe_rv_indweight");
	phoebe_parameter_add_option (par, "Standard weight");
	phoebe_parameter_add_option (par, "Standard deviation");
	phoebe_parameter_add_option (par, "Unavailable");

	par = phoebe_parameter_lookup ("phoebe_ld_model");
	phoebe_parameter_add_option (par, "Linear cosine law");
	phoebe_parameter_add_option (par, "Logarithmic law");
	phoebe_parameter_add_option (par, "Square root law");

	par = phoebe_parameter_lookup ("phoebe_synscatter_levweight");
	phoebe_parameter_add_option (par, "No level-dependent weighting");
	phoebe_parameter_add_option (par, "Poissonian scatter");
	phoebe_parameter_add_option (par, "Low light scatter");

	par = phoebe_parameter_lookup ("phoebe_passband_mode");
	phoebe_parameter_add_option (par, "None");
	phoebe_parameter_add_option (par, "Interpolation");
	phoebe_parameter_add_option (par, "Rigorous");

	par = phoebe_parameter_lookup ("phoebe_el3_units");
	phoebe_parameter_add_option (par, "Total light");
	phoebe_parameter_add_option (par, "Flux");

	return SUCCESS;
}

PHOEBE_parameter *phoebe_parameter_new ()
{
	/*
	 * This function allocates memory for the new parameter and NULLifies
	 * all field pointers for subsequent allocation.
	 */

	PHOEBE_parameter *par = phoebe_malloc (sizeof (*par));

	par->qualifier   = NULL;
	par->description = NULL;
	par->menu        = NULL;
	par->deps        = NULL;
	par->widget      = NULL;

	return par;
}

int phoebe_parameter_add (char *qualifier, char *description, PHOEBE_parameter_kind kind, char *dependency, double min, double max, double step, bool tba, ...)
{
	va_list args;
	PHOEBE_parameter *par = phoebe_parameter_new ();

	par->qualifier   = strdup (qualifier);
	par->description = strdup (description);
	par->kind        = kind;
	par->min         = min;
	par->max         = max;
	par->step        = step;
	par->tba         = tba;

	/* If this parameter has a dependency, add it to the list of children: */
	if (dependency) {
		PHOEBE_parameter *dep = phoebe_parameter_lookup (dependency);
		if (!dep)
			phoebe_lib_error ("dependency %s for %s not found, ignoring.\n", dependency, qualifier);
		else {
			PHOEBE_parameter_list *list = phoebe_malloc (sizeof (*list));
			list->par  = par;
			list->next = dep->deps;
			dep->deps = list;
		}
	}

	va_start (args, tba);
	par->type = va_arg (args, PHOEBE_type);

	switch (par->type) {
		case TYPE_INT:
			par->value.i = va_arg (args, int);
		break;
		case TYPE_DOUBLE:
			par->value.d = va_arg (args, double);
		break;
		case TYPE_BOOL:
			par->value.b = va_arg (args, bool);
		break;
		case TYPE_STRING: {
			char *str = va_arg (args, char *);
			par->value.str = phoebe_malloc (strlen (str) + 1);
			strcpy (par->value.str, str);
		}
		break;
		case TYPE_INT_ARRAY:
			par->value.array = phoebe_array_new (TYPE_INT_ARRAY);
			par->defaultvalue.i = va_arg (args, int);
		break;
		case TYPE_DOUBLE_ARRAY:
			par->value.vec = phoebe_vector_new ();
			par->defaultvalue.d = va_arg (args, double);
		break;
		case TYPE_BOOL_ARRAY:
			par->value.array = phoebe_array_new (TYPE_BOOL_ARRAY);
			par->defaultvalue.b = va_arg (args, bool);
		break;
		case TYPE_STRING_ARRAY: {
			char *str = va_arg (args, char *);
			par->value.array = phoebe_array_new (TYPE_STRING_ARRAY);
			par->defaultvalue.str = strdup (str);
		}
		break;
	}
	va_end (args);

	if (kind != KIND_MENU) par->menu = NULL;
	else {
		par->menu = phoebe_malloc (sizeof (*(par->menu)));
		par->menu->optno  = 0;
		par->menu->option = NULL;
	}

	phoebe_parameter_commit (par);

	return SUCCESS;
}

int phoebe_parameter_add_option (PHOEBE_parameter *par, char *option)
{
	/*
	 * This function adds an option 'option' to the menu of the passed
	 * parameter. The kind field of the parameter must be KIND_MENU, otherwise
	 * the function will abort.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_PARAMETER_KIND_NOT_MENU
	 *   SUCCESS
	 */

	if (!par) {
		phoebe_lib_error ("invalid parameter passed to phoebe_parameter_add_option ()!\n");
		return ERROR_QUALIFIER_NOT_FOUND;
	}

	if (par->kind != KIND_MENU) {
		phoebe_lib_error ("a non-KIND_MENU parameter passed to phoebe_parameter_add_option ()!\n");
		return ERROR_PARAMETER_KIND_NOT_MENU;
	}

	par->menu->optno++;
	par->menu->option = phoebe_realloc (par->menu->option, par->menu->optno * sizeof (*(par->menu->option)));
	par->menu->option[par->menu->optno-1] = strdup (option);

	phoebe_debug ("option \"%s\" added to parameter %s.\n", option, par->qualifier);
	return SUCCESS;
}

unsigned int phoebe_parameter_hash (char *qualifier)
{
	/*
	 * This is the hashing function for storing parameters into the parameter
	 * table.
	 */

	unsigned int h = 0;
	unsigned char *p;

	for (p = (unsigned char *) qualifier; *p != '\0'; p++)
		h = PHOEBE_PT_HASH_MULTIPLIER * h + *p;

	return h % PHOEBE_PT_HASH_BUCKETS;
}

int phoebe_parameter_commit (PHOEBE_parameter *par)
{
	int hash = phoebe_parameter_hash (par->qualifier);
	PHOEBE_parameter_list *elem = PHOEBE_pt->bucket[hash];

	while (elem) {
		if (strcmp (elem->par->qualifier, par->qualifier) == 0) break;
		elem = elem->next;
	}

	if (elem) {
		phoebe_lib_error ("parameter %s already declared, ignoring.\n", par->qualifier);
		return SUCCESS;
	}
	else {
		elem = phoebe_malloc (sizeof (*elem));

		elem->par  = par;
		elem->next = PHOEBE_pt->bucket[hash];
		PHOEBE_pt->bucket[hash] = elem;
	}

	phoebe_debug ("parameter %s added to bucket %d.\n", par->qualifier, hash);
	return SUCCESS;
}

PHOEBE_parameter *phoebe_parameter_lookup (char *qualifier)
{
	unsigned int hash = phoebe_parameter_hash (qualifier);
	PHOEBE_parameter_list *elem = PHOEBE_pt->bucket[hash];

	while (elem) {
		if (strcmp (elem->par->qualifier, qualifier) == 0) break;
		elem = elem->next;
	}

	if (!elem) return NULL;
	return elem->par;
}

int phoebe_parameter_free (PHOEBE_parameter *par)
{
	/*
	 * This function frees all memory allocated for the passed parameter 'par'.
	 */

	int i;

	if (par->qualifier)   free (par->qualifier);
	if (par->description) free (par->description);

	/* Free parameter options: */
	if (par->menu) {
		for (i = 0; i < par->menu->optno; i++)
			free (par->menu->option[i]);
		free (par->menu->option);
		free (par->menu);
	}

	/* If parameters are strings or arrays, we need to free them as well: */
	if (par->type == TYPE_STRING)
		free (par->value.str);

	if (par->type == TYPE_STRING_ARRAY) {
		phoebe_array_free (par->value.array);
		free (par->defaultvalue.str);
	}

	if (par->type == TYPE_INT_ARRAY    ||
		par->type == TYPE_BOOL_ARRAY   ||
		par->type == TYPE_DOUBLE_ARRAY) {
		phoebe_array_free (par->value.array);
	}

	/* Free linked list elements, but not stored parameters: */
	while (par->deps) {
		PHOEBE_parameter_list *list = par->deps->next;
		free (par->deps);
		par->deps = list;
	}

	free (par);

	return SUCCESS;
}

int phoebe_free_parameters ()
{
	/*
	 * This function frees all parameters from the parameter table.
	 */

	int i;
	PHOEBE_parameter_list *elem;

	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		while (PHOEBE_pt->bucket[i]) {
			elem = PHOEBE_pt->bucket[i];
			PHOEBE_pt->bucket[i] = elem->next;
			phoebe_parameter_free (elem->par);
			free (elem);
		}
	}
	free (PHOEBE_pt);

	return SUCCESS;
}

int phoebe_parameter_update_deps (PHOEBE_parameter *par, int oldval)
{
	/*
	 * This function is called whenever the dimension of parameter arrays must
	 * be changed. Typically this happens when the number of observed data cur-
	 * ves is changed, the number of spots is changed etc.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	PHOEBE_parameter_list *list = par->deps;

	int dim = par->value.i;
	int status, j;

	/* If the dimension is the same, there's nothing to be done. */
	if (oldval == dim) return SUCCESS;

	while (list) {
		phoebe_debug ("resizing array %s to dim %d\n", list->par->qualifier, dim);

		switch (list->par->type) {
				case TYPE_INT_ARRAY:
					phoebe_array_realloc (list->par->value.array, dim);
					for (j = oldval; j < dim; j++)
						list->par->value.array->val.iarray[j] = list->par->defaultvalue.i;
				break;
				case TYPE_BOOL_ARRAY:
					status = phoebe_array_realloc (list->par->value.array, dim);
					if (status != SUCCESS) phoebe_lib_error ("%s", phoebe_error (status));
					for (j = oldval; j < dim; j++)
						list->par->value.array->val.barray[j] = list->par->defaultvalue.b;
				break;
				case TYPE_DOUBLE_ARRAY:
					phoebe_vector_realloc (list->par->value.vec, dim);
					for (j = oldval; j < dim; j++)
						list->par->value.vec->val[j] = list->par->defaultvalue.d;
				break;
				case TYPE_STRING_ARRAY:
					phoebe_array_realloc (list->par->value.array, dim);
					for (j = oldval; j < dim; j++)
						list->par->value.array->val.strarray[j] = strdup (list->par->defaultvalue.str);
				break;
		}

		list = list->next;
	}

	return SUCCESS;
}

bool phoebe_parameter_menu_option_is_valid (char *qualifier, char *option)
{
	/*
	 * This function is a boolean test for parameter menu options. It returns
	 * TRUE if the option is valid and FALSE if it is invalid.
	 */

	int i;

	/* Is the qualifier valid: */
	PHOEBE_parameter *par = phoebe_parameter_lookup (qualifier);
	if (!par) return FALSE;

	/* Is the qualified parameter a menu: */
	if (par->kind != KIND_MENU) return FALSE;

	/* Is the option one of the menu options: */
	for (i = 0; i < par->menu->optno; i++)
		if (strcmp (par->menu->option[i], option) == 0)
			return TRUE;

	return FALSE;
}

int phoebe_parameter_get_value (PHOEBE_parameter *par, ...)
{
	/*
	 * This is the public function for providing the values of parameters.
	 * It is the only function that should be used for this purpose, all
	 * other functions are internal and should not be used.
	 *
	 * Synopsis:
	 *
	 *   phoebe_parameter_get_value (par, [index, ], &value)
	 *
	 * Return values:
	 *
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	int index = 0;
	va_list args;

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	va_start (args, par);

	switch (par->type) {
		case TYPE_INT: {
			int *value = va_arg (args, int *);
			*value = par->value.i;
		}
		break;
		case TYPE_BOOL: {
			bool *value = va_arg (args, bool *);
			*value = par->value.b;
		}
		break;
		case TYPE_DOUBLE: {
			double *value = va_arg (args, double *);
			*value = par->value.d;
		}
		break;
		case TYPE_STRING: {
			const char **value = va_arg (args, const char **);
			*value = par->value.str;
		}
		break;
		case TYPE_INT_ARRAY:
			index = va_arg (args, int);
			{
			int *value = va_arg (args, int *);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.array->val.iarray[index];
			}
		break;
		case TYPE_BOOL_ARRAY:
			index = va_arg (args, int);
			{
			bool *value = va_arg (args, bool *);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.array->val.barray[index];
			}
		break;
		case TYPE_DOUBLE_ARRAY:
			index = va_arg (args, int);
			{
			double *value = va_arg (args, double *);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.vec->val[index];
			}
		break;
		case TYPE_STRING_ARRAY:
			index = va_arg (args, int);
			{
			const char **value = va_arg (args, const char **);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.array->val.strarray[index];
			}
		break;
		default:
			phoebe_lib_error ("exception handler invoked in phoebe_parameter_get_value (), please report this!\n");
			return ERROR_EXCEPTION_HANDLER_INVOKED;
	}
	va_end (args);

	return SUCCESS;
}

int phoebe_parameter_set_value (PHOEBE_parameter *par, ...)
{
	/*
	 * This is the public function for changing the passed parameter's value.
	 * It is the only function that should be used for this purpose, all other
	 * functions should be regarded internal and should not be used.
	 *
	 * Synopsis:
	 *
	 *   phoebe_parameter_set_value (qualifier, [curve, ] value)
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   ERROR_INDEX_OUT_OF_RANGE
	 *   SUCCESS
	 */

	int index = 0;
	va_list args;

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	va_start (args, par);

	switch (par->type) {
		case TYPE_INT: {
			int value = va_arg (args, int);
			if (par->kind == KIND_MODIFIER) {
				int oldval = par->value.i;
				par->value.i = value;
				phoebe_parameter_update_deps (par, oldval);
			}
			else
				par->value.i = value;
		}
		break;
		case TYPE_BOOL: {
			bool value = va_arg (args, bool);
			par->value.b = value;
		}
		break;
		case TYPE_DOUBLE: {
			double value = va_arg (args, double);
			par->value.d = value;
		}
		break;
		case TYPE_STRING: {
			char *value = va_arg (args, char *);
			free (par->value.str);
			par->value.str = phoebe_malloc (strlen (value) + 1);
			strcpy (par->value.str, value);

			/*
			 * If the passed parameter is a menu, let's check if the option
			 * is valid. If it is not, just warn about it, but set its value
			 * anyway.
			 */

			if (par->kind == KIND_MENU && !phoebe_parameter_menu_option_is_valid (par->qualifier, (char *) value))
				phoebe_lib_warning ("option \"%s\" is not a valid menu option.\n", value);
		}
		break;
		case TYPE_INT_ARRAY:
			index = va_arg (args, int);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			{
			int value = va_arg (args, int);
			par->value.array->val.iarray[index] = value;
			}
		break;
		case TYPE_BOOL_ARRAY:
			index = va_arg (args, int);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			{
			bool value = va_arg (args, bool);
			par->value.array->val.barray[index] = value;
			}
		break;
		case TYPE_DOUBLE_ARRAY:
			index = va_arg (args, int);
			if (index < 0 || index > par->value.vec->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			{
			double value = va_arg (args, double);
			par->value.vec->val[index] = value;
			}
		break;
		case TYPE_STRING_ARRAY:
			index = va_arg (args, int);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			{
			char *value = va_arg (args, char *);
			free (par->value.array->val.strarray[index]);
			par->value.array->val.strarray[index] = phoebe_malloc (strlen (value) + 1);
			strcpy (par->value.array->val.strarray[index], value);

			/*
			 * If the passed parameter is a menu, let's check if the option is
			 * valid. If it is not, just warn about it, but set its value
			 * anyway.
			 */

			if (par->kind == KIND_MENU && !phoebe_parameter_menu_option_is_valid (par->qualifier, (char *) value))
				phoebe_lib_warning ("option \"%s\" is not a valid menu option.\n", value);
			}
		break;
	}
	va_end (args);

	return SUCCESS;
}

int phoebe_parameter_get_tba (PHOEBE_parameter *par, bool *tba)
{
	/*
	 * This is a public function for reading out the passed parameter's TBA
	 * (To Be Adjusted) bit.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	*tba = par->tba;
	return SUCCESS;
}

int phoebe_parameter_set_tba (PHOEBE_parameter *par, bool tba)
{
	/*
	 * This is the public function for changing the passed parameter's TBA
	 * (To Be Adjusted) bit. At the same time the function adds or removes
	 * that parameter from the list of parameters marked for adjustment.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	PHOEBE_parameter_list *list, *prev = NULL;
	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	par->tba = tba;

	/*
	 * Now we need to add or remove the parameter from the TBA list in the
	 * parameter table:
	 */

	list = PHOEBE_pt->lists.marked_tba;
	if (tba) {
		while (list) {
			if (list->par == par) break;
			list = list->next;
		}
		if (!list) {
			list = phoebe_malloc (sizeof (*list));
			list->par = par;
			list->next = PHOEBE_pt->lists.marked_tba;
			PHOEBE_pt->lists.marked_tba = list;
			phoebe_debug ("Parameter %s added to the tba list.\n", list->par->qualifier);
		}
		else {
			/* The parameter is already in the list, nothing to be done. */
		}
	}
	else /* if (!tba) */ {
		while (list) {
			if (list->par == par) break;
			prev = list;
			list = list->next;
		}
		if (list) {
			if (prev)
				prev->next = list->next;
			else {
				PHOEBE_pt->lists.marked_tba = list->next;
				phoebe_debug ("Parameter %s removed from the tba list.\n", list->par->qualifier);
			}
			free (list);
		}
		else {
			/* The parameter is not in the list, nothing to be done. */
		}
	}

	return SUCCESS;
}

int phoebe_parameter_get_step (PHOEBE_parameter *par, double *step)
{
	/*
	 * This is a public function for reading out the passed parameter's step
	 * size that is used for minimization.
	 *
	 * Return values:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	*step = par->step;
	return SUCCESS;
}

int phoebe_parameter_set_step (PHOEBE_parameter *par, double step)
{
	/*
	 * This is the public function for changing the passed parameter's step.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	par->step = step;

	return SUCCESS;
}

int phoebe_parameter_get_lower_limit (PHOEBE_parameter *par, double *valmin)
{
	/*
	 * This is the public function for reading out the lower parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	*valmin = par->min;
	return SUCCESS;
}

int phoebe_parameter_set_lower_limit (PHOEBE_parameter *par, double valmin)
{
	/*
	 * This is the public function for changing the lower parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	par->min = valmin;
	return SUCCESS;
}

int phoebe_parameter_get_upper_limit (PHOEBE_parameter *par, double *valmax)
{
	/*
	 * This is the public function for reading out the upper parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	*valmax = par->max;
	return SUCCESS;
}

int phoebe_parameter_set_upper_limit (PHOEBE_parameter *par, double valmax)
{
	/*
	 * This is the public function for changing the upper parameter limit.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	par->max = valmax;
	return SUCCESS;
}

int phoebe_parameter_get_limits (PHOEBE_parameter *par, double *valmin, double *valmax)
{
	/*
	 * This is the public function for reading out the passed parameter's
	 * limits.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	*valmin = par->min;
	*valmax = par->max;

	return SUCCESS;
}

int phoebe_parameter_set_limits (PHOEBE_parameter *par, double valmin, double valmax)
{
	/*
	 * This is the public function for changing the passed parameter's limits.
	 *
	 * Error codes:
	 *
	 *   ERROR_QUALIFIER_NOT_FOUND
	 *   SUCCESS
	 */

	if (!par) return ERROR_QUALIFIER_NOT_FOUND;

	par->min = valmin;
	par->max = valmax;

	return SUCCESS;
}

PHOEBE_parameter_list *phoebe_parameter_list_get_marked_tba ()
{
	/*
	 * This is a simple wrapper function that returns a list of parameters
	 * that are marked for adjustment. This is the only function that should
	 * be used for this purpose; accessing table elements directly is not
	 * allowed.
	 *
	 * The function returns a pointer to a list of parameters marked for
	 * adjustment. If no parameters are marked, it returns NULL;
	 */

	return PHOEBE_pt->lists.marked_tba;
}

int phoebe_el3_units_id (PHOEBE_el3_units *el3_units)
{
	/*
	 * This function assigns the third light units id to el3_units variable.
	 * If an error occurs, -1 is assigned and ERROR_INVALID_EL3_UNITS code is
	 * returned.
	 */

	const char *el3str;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_el3_units"), &el3str);

	*el3_units = PHOEBE_EL3_UNITS_INVALID_ENTRY;
	if (strcmp (el3str, "Total light") == 0) *el3_units = PHOEBE_EL3_UNITS_TOTAL_LIGHT;
	if (strcmp (el3str,        "Flux") == 0) *el3_units = PHOEBE_EL3_UNITS_FLUX;
	if (*el3_units == PHOEBE_EL3_UNITS_INVALID_ENTRY)
		return ERROR_INVALID_EL3_UNITS;

	return SUCCESS;
}

int phoebe_open_parameter_file (const char *filename)
{
	/*
	 * This function opens PHOEBE 0.3x parameter files. The return value is:
	 *
	 *   ERROR_FILE_IS_EMPTY
	 *   ERROR_INVALID_HEADER
	 *   ERROR_FILE_NOT_REGULAR
	 *   ERROR_FILE_NO_PERMISSIONS
	 *   ERROR_FILE_NOT_FOUND
	 *   SUCCESS
	 */

	char readout_string[255];
	char *readout_str = readout_string;

	char keyword_string[255];
	char *keyword_str = keyword_string;

	char *value_str;

	int i, status;
	int lineno = 0;

	FILE *keyword_file;

	phoebe_debug ("entering function 'open_parameter_file ()'\n");

	/* First a checkup if everything is OK with the filename:                   */
	if (!filename_exists ((char *) filename))               return ERROR_FILE_NOT_FOUND;
	if (!filename_is_regular_file ((char *) filename))      return ERROR_FILE_NOT_REGULAR;
	if (!filename_has_read_permissions ((char *) filename)) return ERROR_FILE_NO_PERMISSIONS;

	keyword_file = fopen (filename, "r");

	/* Parameter files start with a commented header line; see if it's there: */

	fgets (readout_str, 255, keyword_file); lineno++;
	if (feof (keyword_file)) return ERROR_FILE_IS_EMPTY;
	readout_str[strlen(readout_str)-1] = '\0';

	if (strstr (readout_str, "PHOEBE") != 0) {
		/* Yep, it's there! Read out version number and if it's a legacy file,    */
		/* call a suitable function for opening legacy keyword files.             */

		char *version_str = strstr (readout_str, "PHOEBE");
		double version;
			
		if (sscanf (version_str, "PHOEBE %lf", &version) != 1) {
			/* Just in case, if the header line is invalid.                       */
			phoebe_lib_error ("Invalid header line in %s, aborting.\n", filename);
			return ERROR_INVALID_HEADER;
		}
		if (version < 0.30) {
			fclose (keyword_file);
			phoebe_debug ("opening legacy parameter file.\n");
			status = phoebe_open_legacy_parameter_file (filename);
			return status;
		}
		phoebe_debug ("PHOEBE parameter file version: %2.2lf\n", version);
	}

	while (!feof (keyword_file)) {
		fgets (readout_str, 255, keyword_file); lineno++;
		if (feof (keyword_file)) break;

		/*
		 * Clear the read string of leading and trailing spaces, tabs, newlines,
		 * comments and empty lines:
		 */

		readout_str[strlen(readout_str)-1] = '\0';
		if (strchr (readout_str, '#') != 0) (strchr (readout_str, '#'))[0] = '\0';
		while (readout_str[0] == ' ' || readout_str[0] == '\t') readout_str++;
		while (readout_str[strlen(readout_str)-1] == ' ' || readout_str[strlen(readout_str)-1] == '\t') readout_str[strlen(readout_str)-1] = '\0';
		if (strlen (readout_str) == 0) continue;

		/*
		 * Read out the qualifier only. We can't read out the value here
		 * because it may be a string containing spaces and that would result
		 * in an incomplete readout.
		 */

		if (sscanf (readout_str, "%s = %*s", keyword_str) != 1) {
			phoebe_lib_error ("line %d of the parameter file is invalid, skipping.\n", lineno);
			continue;
		}

		value_str = strchr (readout_str, '=');
		if (value_str == NULL) {
			/* If the keyword doesn't have '=', it will be skipped.                 */
			phoebe_lib_error ("qualifier %s initialization (line %d) is invalid.\n", keyword_str, lineno);
			continue;
		}

		/* value_str now points to '=', we need the next character:               */
		value_str++;

		/* Eat all empty spaces and quotes at the beginning and at the end: */
		while (value_str[0] == ' ' || value_str[0] == '\t' || value_str[0] == '"') value_str++;
		while (value_str[strlen (value_str)-1] == ' ' || value_str[strlen (value_str)-1] == '\t' || value_str[strlen (value_str)-1] == '"') value_str[strlen(value_str)-1] = '\0';

		/*
		 * What remains is a qualifier, the element index [i] in case of
		 * wavelength-dependent parameters, and fields in case of an adjustable
		 * parameter. Let's parse it. The following looks really really scary,
		 * but it's actually quite straight-forward and, more importantly,
		 * tested thoroughly. :)
		 */

		{
		PHOEBE_parameter *par;
		char *qualifier = NULL;
		char *field     = NULL;
		char *field_sep = NULL;
		char *elem_sep  = NULL;
		int   elem;
		int   olddim;

		phoebe_debug ("keyword_str: %30s %2d ", keyword_str, strlen(keyword_str));

		if ( (elem_sep = strchr (keyword_str, '[')) && (field_sep = strchr (keyword_str, '.')) ) {
			phoebe_debug ("%2d %2d ", strlen (elem_sep), strlen (field_sep));
			qualifier = phoebe_malloc ( (strlen(keyword_str)-strlen(elem_sep)+1)*sizeof(*qualifier) );
			strncpy (qualifier, keyword_str, strlen(keyword_str)-strlen(elem_sep));
			qualifier[strlen(keyword_str)-strlen(elem_sep)] = '\0';
			sscanf (elem_sep, "[%d]", &elem);
			field = phoebe_malloc (strlen(field_sep)*sizeof(*field));
			strcpy (field, field_sep+1);
			field[strlen(field_sep)-1] = '\0';
			phoebe_debug ("%30s %2d %6s", qualifier, elem, field);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_error ("qualifier %s not recognized, ignoring.\n", qualifier);
				free (qualifier);
				free (field);
				continue;
			}

			/*
			 * Here's a trick: qualifiers in the parameter file may be shuffled
			 * and array elements may appear before their dependencies are
			 * declared. Since we want fool-proof capability, we test the
			 * dependencies here and allocate space if necessary.
			 */

			switch (par->type) {
				case TYPE_INT_ARRAY:
					if (par->value.array->dim < elem) {
						olddim = par->value.array->dim;
						phoebe_array_realloc (par->value.array, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.array->val.iarray[i] = par->defaultvalue.i;
					}
					if (strcmp (field,  "VAL") == 0)
						phoebe_parameter_set_value (par, elem-1, atoi (value_str));
					if (strcmp (field,  "MIN") == 0)
						phoebe_parameter_set_lower_limit (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_upper_limit (par, atof (value_str));
					if (strcmp (field, "STEP") == 0)
						phoebe_parameter_set_step (par, atof (value_str));
					if (strcmp (field,  "ADJ") == 0)
						phoebe_parameter_set_tba (par, atob (value_str));
				break;
				case TYPE_BOOL_ARRAY:
					if (par->value.array->dim < elem) {
						olddim = par->value.array->dim;
						phoebe_array_realloc (par->value.array, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.array->val.barray[i] = par->defaultvalue.b;
					}
					if (strcmp (field,  "VAL") == 0)
						phoebe_parameter_set_value (par, elem-1, atob (value_str));
					if (strcmp (field,  "MIN") == 0)
						phoebe_parameter_set_lower_limit (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_upper_limit (par, atof (value_str));
					if (strcmp (field, "STEP") == 0)
						phoebe_parameter_set_step (par, atof (value_str));
					if (strcmp (field,  "ADJ") == 0)
						phoebe_parameter_set_tba (par, atob (value_str));
				break;
				case TYPE_DOUBLE_ARRAY:
					if (par->value.vec->dim < elem) {
						olddim = par->value.vec->dim;
						phoebe_vector_realloc (par->value.vec, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.vec->val[i] = par->defaultvalue.d;
					}
					if (strcmp (field,  "VAL") == 0)
						phoebe_parameter_set_value (par, elem-1, atof (value_str));
					if (strcmp (field,  "MIN") == 0)
						phoebe_parameter_set_lower_limit (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_upper_limit (par, atof (value_str));
					if (strcmp (field, "STEP") == 0)
						phoebe_parameter_set_step (par, atof (value_str));
					if (strcmp (field,  "ADJ") == 0)
						phoebe_parameter_set_tba (par, atob (value_str));
				break;
				case TYPE_STRING_ARRAY:
					if (par->value.array->dim < elem) {
						olddim = par->value.array->dim;
						phoebe_array_realloc (par->value.array, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.array->val.strarray[i] = strdup (par->defaultvalue.str);
					}
					if (strcmp (field,  "VAL") == 0)
						phoebe_parameter_set_value (par, elem-1, value_str);
					if (strcmp (field,  "MIN") == 0)
						phoebe_parameter_set_lower_limit (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_upper_limit (par, atof (value_str));
					if (strcmp (field, "STEP") == 0)
						phoebe_parameter_set_step (par, atof (value_str));
					if (strcmp (field,  "ADJ") == 0)
						phoebe_parameter_set_tba (par, atob (value_str));
				break;
			}
		}
		else if (elem_sep = strchr (keyword_str, '[')) {
			phoebe_debug ("%2d    ", strlen (elem_sep));
			qualifier = phoebe_malloc ( (strlen(keyword_str)-strlen(elem_sep)+1)*sizeof(*qualifier) );
			strncpy (qualifier, keyword_str, strlen(keyword_str)-strlen(elem_sep));
			qualifier[strlen(keyword_str)-strlen(elem_sep)] = '\0';
			sscanf (elem_sep, "[%d]", &elem);
			phoebe_debug ("%30s %2d", qualifier, elem);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_error ("qualifier %s not recognized, ignoring.\n", qualifier);
				free (qualifier);
				continue;
			}

			/*
			 * Here's a trick: qualifiers in the parameter file may be shuffled
			 * and array elements may appear before their dependencies are
			 * declared. Since we want fool-proof capability, we test the
			 * dependencies here and allocate space if necessary.
			 */

			switch (par->type) {
				case TYPE_INT_ARRAY:
					if (par->value.array->dim < elem) {
						olddim = par->value.array->dim;
						phoebe_array_realloc (par->value.array, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.array->val.iarray[i] = par->defaultvalue.i;
					}
					phoebe_parameter_set_value (par, elem-1, atoi (value_str));
				break;
				case TYPE_BOOL_ARRAY:
					if (par->value.array->dim < elem) {
						olddim = par->value.array->dim;
						phoebe_array_realloc (par->value.array, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.array->val.barray[i] = par->defaultvalue.b;
					}
					phoebe_parameter_set_value (par, elem-1, atob (value_str));
				break;
				case TYPE_DOUBLE_ARRAY:
					if (par->value.vec->dim < elem) {
						olddim = par->value.vec->dim;
						phoebe_vector_realloc (par->value.vec, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.vec->val[i] = par->defaultvalue.d;
					}
					phoebe_parameter_set_value (par, elem-1, atof (value_str));
				break;
				case TYPE_STRING_ARRAY:
					if (par->value.array->dim < elem) {
						olddim = par->value.array->dim;
						phoebe_array_realloc (par->value.array, elem);
						for (i = olddim; i < elem-1; i++)
							par->value.array->val.strarray[i] = strdup (par->defaultvalue.str);
					}
					phoebe_parameter_set_value (par, elem-1, value_str);
				break;
			}
		}
		else if (field_sep = strchr (keyword_str, '.')) {
			phoebe_debug ("   %2d ", strlen (field_sep));
			qualifier = phoebe_malloc ( (strlen(keyword_str)-strlen(field_sep)+1)*sizeof(*qualifier) );
			strncpy (qualifier, keyword_str, strlen(keyword_str)-strlen(field_sep));
			qualifier[strlen(keyword_str)-strlen(field_sep)] = '\0';
			field = phoebe_malloc (strlen(field_sep)*sizeof(*field));
			strcpy (field, field_sep+1);
			field[strlen(field_sep)-1] = '\0';
			phoebe_debug ("%30s    %6s", qualifier, field);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_error ("qualifier %s not recognized, ignoring.\n", qualifier);
				free (qualifier);
				free (field);
				continue;
			}

			if (strcmp (field,  "VAL") == 0)
				phoebe_parameter_set_value (par, atof (value_str));
					if (strcmp (field,  "MIN") == 0)
						phoebe_parameter_set_lower_limit (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_upper_limit (par, atof (value_str));
					if (strcmp (field, "STEP") == 0)
						phoebe_parameter_set_step (par, atof (value_str));
					if (strcmp (field,  "ADJ") == 0)
						phoebe_parameter_set_tba (par, atob (value_str));
		}
		else {
			phoebe_debug ("      ");
			qualifier = phoebe_malloc ((strlen(keyword_str)+1)*sizeof(*qualifier));
			strcpy (qualifier, keyword_str);
			phoebe_debug ("%30s   ", qualifier);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_error ("qualifier %s not recognized, ignoring.\n", qualifier);
				free (qualifier);
				continue;
			}

			switch (par->type) {
				case TYPE_INT:
					phoebe_parameter_set_value (par, atoi (value_str));
				break;
				case TYPE_BOOL:
					phoebe_parameter_set_value (par, atob (value_str));
				break;
				case TYPE_DOUBLE:
					phoebe_parameter_set_value (par, atof (value_str));
				break;
				case TYPE_STRING:
					/* Strip the string of quotes if necessary:                       */
					while (value_str[0] == '"') value_str++;
					while (value_str[strlen(value_str)-1] == '"') value_str[strlen(value_str)-1] = '\0';
					phoebe_parameter_set_value (par, value_str);
				break;
				default:
					phoebe_lib_error ("exception handler invoked in phoebe_open_parameter_file (), please report this!\n");
					return ERROR_EXCEPTION_HANDLER_INVOKED;
			}
		}
		phoebe_debug ("\n");

		if (qualifier) { free (qualifier); qualifier = NULL; }
		if (field)     { free (field);     field     = NULL; }
		}
	}

	fclose (keyword_file);

	phoebe_debug ("leaving function 'open_parameter_file ()'\n");

	return SUCCESS;
}

int intern_save_to_parameter_file (PHOEBE_parameter *par, FILE *file)
{
	/*
	 * This function saves the contents of parameter 'par' to file 'file'.
	 */

	int j;

	if (par->kind == KIND_ADJUSTABLE) {
		switch (par->type) {
			case TYPE_DOUBLE:
				fprintf (file, "%s.VAL  = %lf\n", par->qualifier, par->value.d);
			break;
			case TYPE_DOUBLE_ARRAY:
				if (par->value.vec)
					for (j = 0; j < par->value.vec->dim; j++)
						fprintf (file, "%s[%d].VAL  = %lf\n", par->qualifier, j+1, par->value.vec->val[j]);
			break;
			default:
				phoebe_lib_error ("exception handler invoked in intern_save_to_parameter_file (), please report this!\n");
		}

		fprintf (file, "%s.ADJ  = %d\n",  par->qualifier, par->tba);
		fprintf (file, "%s.STEP = %lf\n", par->qualifier, par->step);
		fprintf (file, "%s.MIN  = %lf\n", par->qualifier, par->min);
		fprintf (file, "%s.MAX  = %lf\n", par->qualifier, par->max);
	}
	else {
		switch (par->type) {
			case TYPE_INT:
				fprintf (file, "%s = %d\n", par->qualifier, par->value.i);
			break;
			case TYPE_BOOL:
				fprintf (file, "%s = %d\n", par->qualifier, par->value.b);
			break;
			case TYPE_DOUBLE:
				fprintf (file, "%s = %lf\n", par->qualifier, par->value.d);
			break;
			case TYPE_STRING:
				fprintf (file, "%s = \"%s\"\n", par->qualifier, par->value.str);
			break;
			case TYPE_INT_ARRAY:
				if (par->value.array)
					for (j = 0; j < par->value.array->dim; j++)
						fprintf (file, "%s[%d] = %d\n", par->qualifier, j+1, par->value.array->val.iarray[j]);
			break;
			case TYPE_BOOL_ARRAY:
				if (par->value.array)
					for (j = 0; j < par->value.array->dim; j++)
						fprintf (file, "%s[%d] = %d\n", par->qualifier, j+1, par->value.array->val.barray[j]);
			break;
			case TYPE_DOUBLE_ARRAY:
				if (par->value.vec)
					for (j = 0; j < par->value.vec->dim; j++)
						fprintf (file, "%s[%d] = %lf\n", par->qualifier, j+1, par->value.vec->val[j]);
			break;
			case TYPE_STRING_ARRAY:
				if (par->value.array)
					for (j = 0; j < par->value.array->dim; j++)
						fprintf (file, "%s[%d] = \"%s\"\n", par->qualifier, j+1, par->value.array->val.strarray[j]);
			break;
			default:
				phoebe_lib_error ("exception handler invoked in intern_save_to_parameter_file (), please report this!\n");
			break;
			}
		}

	return SUCCESS;
}

int phoebe_save_parameter_file (const char *filename)
{
	/*
	 * This function saves PHOEBE 0.3x keyword files.
	 *
	 * Return values:
	 *
	 *   ERROR_FILE_IS_INVALID
	 *   SUCCESS
	 */

	int i;
	FILE *parameter_file;
	PHOEBE_parameter_list *elem;

	/* First a checkup if everything is OK with the filename: */
	parameter_file = fopen (filename, "w");
	if (!parameter_file) return ERROR_FILE_IS_INVALID;

	/* Write a version header: */
	fprintf (parameter_file, "# Parameter file conforming to %s\n", PHOEBE_VERSION_NUMBER);

	/* Traverse the parameter table and save parameters one by one: */
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		elem = PHOEBE_pt->bucket[i];
		while (elem) {
			intern_save_to_parameter_file (elem->par, parameter_file);
			elem = elem->next;
		}
	}

	fclose (parameter_file);
	return SUCCESS;
}

int phoebe_open_legacy_parameter_file (const char *filename)
{
	phoebe_lib_error ("Not yet implemented!\n");
	return SUCCESS;
}
