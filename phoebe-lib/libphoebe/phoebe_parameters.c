#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "phoebe_build_config.h"

#include "phoebe_accessories.h"
#include "phoebe_constraints.h"
#include "phoebe_data.h"
#include "phoebe_error_handling.h"
#include "phoebe_fortran_interface.h"
#include "phoebe_global.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

PHOEBE_parameter_table_list *PHOEBE_pt_list;
PHOEBE_parameter_table      *PHOEBE_pt;

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

	/* ***********************   Data parameters   ************************** */

	phoebe_parameter_add ("phoebe_lc_id",                "Observed LC identification name",            KIND_PARAMETER,  "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	phoebe_parameter_add ("phoebe_lc_filename",          "Observed LC data filename",                  KIND_PARAMETER,  "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	phoebe_parameter_add ("phoebe_lc_sigma",             "Observed LC data standard deviation",        KIND_PARAMETER,  "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_DOUBLE_ARRAY, 0.01);
	phoebe_parameter_add ("phoebe_lc_filter",            "Observed LC data filter",                    KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Johnson:V");
	phoebe_parameter_add ("phoebe_lc_indep",             "Observed LC data independent variable",      KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Time (HJD)");
	phoebe_parameter_add ("phoebe_lc_dep",               "Observed LC data dependent variable",        KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Magnitude");
	phoebe_parameter_add ("phoebe_lc_indweight",         "Observed LC data individual weighting",      KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Standard deviation");
	phoebe_parameter_add ("phoebe_lc_levweight",         "Observed LC data level weighting",           KIND_MENU,       "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Poissonian scatter");
	phoebe_parameter_add ("phoebe_lc_active",            "Observed LC data is used",                   KIND_SWITCH,     "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_BOOL_ARRAY,    YES);

	phoebe_parameter_add ("phoebe_rv_id",                "Observed RV identification name",            KIND_PARAMETER,  "phoebe_lcno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	phoebe_parameter_add ("phoebe_rv_filename",          "Observed RV data filename",                  KIND_PARAMETER,  "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Undefined");
	phoebe_parameter_add ("phoebe_rv_sigma",             "Observed RV data standard deviation",        KIND_PARAMETER,  "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_DOUBLE_ARRAY, 1.0);
	phoebe_parameter_add ("phoebe_rv_filter",            "Observed RV data filter",                    KIND_MENU,       "phoebe_rvno",   0.0,    0.0,    0.0, NO, TYPE_STRING_ARRAY, "Johnson:V");
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
	phoebe_parameter_add ("phoebe_met1",                 "Primary star metallicity",                   KIND_ADJUSTABLE, NULL, -10.0,   10.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_met2",                 "Secondary star metallicity",                 KIND_ADJUSTABLE, NULL, -10.0,   10.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_f1",                   "Primary star synchronicity parameter",       KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        1.0);
	phoebe_parameter_add ("phoebe_f2",                   "Secondary star synchronicity parameter",     KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        1.0);
	phoebe_parameter_add ("phoebe_alb1",                 "Primary star surface albedo",                KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.6);
	phoebe_parameter_add ("phoebe_alb2",                 "Secondary star surface albedo",              KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.6);
	phoebe_parameter_add ("phoebe_grb1",                 "Primary star gravity brightening",           KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.32);
	phoebe_parameter_add ("phoebe_grb2",                 "Secondary star gravity brightening",         KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.32);

	/* **********************   Orbit parameters   ************************** */

	phoebe_parameter_add ("phoebe_ecc",                  "Orbital eccentricity",                       KIND_ADJUSTABLE, NULL,   0.0,    1.0,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_perr0",                "Argument of periastron",                     KIND_ADJUSTABLE, NULL,   0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE,        0.0);
	phoebe_parameter_add ("phoebe_dperdt",               "First time derivative of periastron",        KIND_ADJUSTABLE, NULL,  -1.0,    1.0,   1E-6, NO, TYPE_DOUBLE,        0.0);

	/* *********************   Surface parameters   ************************* */

	phoebe_parameter_add ("phoebe_hla",                  "LC primary star flux leveler",               KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY, 12.566371);
	phoebe_parameter_add ("phoebe_cla",                  "LC secondary star flux leveler",             KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY, 12.566371);
	phoebe_parameter_add ("phoebe_opsf",                 "Opacity frequency function",                 KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);

	phoebe_parameter_add ("phoebe_passband_mode",        "Passband treatment mode",                    KIND_MENU,       NULL,  0.0,    0.0,    0.0, NO, TYPE_STRING,        "Interpolation");
	phoebe_parameter_add ("phoebe_atm1_switch",          "Use Kurucz's models for primary star",       KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_atm2_switch",          "Use Kurucz's models for secondary star",     KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_reffect_switch",       "Detailed reflection effect",                 KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	phoebe_parameter_add ("phoebe_reffect_reflections",  "Number of detailed reflections",             KIND_PARAMETER,  NULL,    2,     10,      1, NO, TYPE_INT,             2);

	phoebe_parameter_add ("phoebe_usecla_switch",        "Decouple CLAs from temperature",             KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);

	/* ********************   Extrinsic parameters   ************************ */

	phoebe_parameter_add ("phoebe_el3_units",            "Units of third light",                       KIND_MENU,       NULL,           0.0,    0.0,    0.0, NO, TYPE_STRING,        "Total light");
	phoebe_parameter_add ("phoebe_el3",                  "Third light contribution",                   KIND_ADJUSTABLE, "phoebe_lcno",  0.0,   1E10,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);
	phoebe_parameter_add ("phoebe_extinction",           "Interstellar extinction coefficient",        KIND_ADJUSTABLE, "phoebe_lcno",  0.0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY,  0.0);

	/* *********************   Fitting parameters   ************************* */

	phoebe_parameter_add ("phoebe_grid_finesize1",       "Fine grid size on primary star",             KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,            20);
	phoebe_parameter_add ("phoebe_grid_finesize2",       "Fine grid size on secondary star",           KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,            20);
	phoebe_parameter_add ("phoebe_grid_coarsesize1",     "Coarse grid size on primary star",           KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,             5);
	phoebe_parameter_add ("phoebe_grid_coarsesize2",     "Coarse grid size on secondary star",         KIND_PARAMETER,  NULL,    5,     60,      1, NO, TYPE_INT,             5);

	phoebe_parameter_add ("phoebe_compute_hla_switch",   "Compute passband (HLA) levels",              KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);
	phoebe_parameter_add ("phoebe_compute_vga_switch",   "Compute gamma velocity",                     KIND_SWITCH,     NULL,  0.0,    0.0,    0.0, NO, TYPE_BOOL,           NO);

	/* **********************   DC fit parameters   ************************* */

	phoebe_parameter_add ("phoebe_dc_symder_switch",     "Should symmetrical DC derivatives be used",  KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,          YES);
	phoebe_parameter_add ("phoebe_dc_lambda",            "Levenberg-Marquardt multiplier for DC",      KIND_PARAMETER,  NULL,  0.0,    1.0,   1e-3, NO, TYPE_DOUBLE,       1e-3);

	phoebe_parameter_add ("phoebe_dc_spot1src",          "Adjusted spot 1 source (at which star is the spot)", KIND_PARAMETER, NULL,      1,      2,      1, NO, TYPE_INT, 1);
	phoebe_parameter_add ("phoebe_dc_spot2src",          "Adjusted spot 2 source (at which star is the spot)", KIND_PARAMETER, NULL,      1,      2,      1, NO, TYPE_INT, 2);
	phoebe_parameter_add ("phoebe_dc_spot1id",           "Adjusted spot 1 ID (which spot is to be adjusted)",  KIND_PARAMETER, NULL,      1,    100,      1, NO, TYPE_INT, 1);
	phoebe_parameter_add ("phoebe_dc_spot2id",           "Adjusted spot 2 ID (which spot is to be adjusted)",  KIND_PARAMETER, NULL,      1,    100,      1, NO, TYPE_INT, 1);

	/* **********************   NMS fit parameters   ************************* */

	phoebe_parameter_add ("phoebe_nms_iter_max",     	"Maximal number of iterations to do",				   KIND_PARAMETER, NULL,      0,1000000,      1, NO, TYPE_INT, 200);
	phoebe_parameter_add ("phoebe_nms_accuracy",     	"Desired accuracy",				   					   KIND_PARAMETER, NULL,      0,      1,      1, NO, TYPE_DOUBLE, 0.01);

	/* *******************   Perturbations parameters   ********************* */

	phoebe_parameter_add ("phoebe_ld_model",             "Limb darkening model",                               KIND_MENU,      NULL,      0,      0,      0, NO, TYPE_STRING, "Logarithmic law");
	phoebe_parameter_add ("phoebe_ld_xbol1",             "Primary star bolometric LD coefficient x",           KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_ybol1",             "Primary star bolometric LD coefficient y",         KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_xbol2",             "Secondary star bolometric LD coefficient x",           KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_ybol2",             "Secondary star bolometric LD coefficient y",         KIND_PARAMETER, NULL,    0.0,    1.0,   0.01, NO, TYPE_DOUBLE,       0.5);
	phoebe_parameter_add ("phoebe_ld_lcx1",              "Primary star bandpass LD coefficient x",             KIND_ADJUSTABLE, "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_lcx2",              "Secondary star bandpass LD coefficient x",           KIND_ADJUSTABLE, "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_lcy1",              "Primary star bandpass LD coefficient y",             KIND_PARAMETER,  "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_lcy2",              "Secondary star bandpass LD coefficient y",           KIND_PARAMETER,  "phoebe_lcno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvx1",              "Primary RV bandpass LD coefficient x",               KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvx2",              "Secondary RV bandpass LD coefficient x",             KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvy1",              "Primary RV bandpass LD coefficient y",               KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);
	phoebe_parameter_add ("phoebe_ld_rvy2",              "Secondary RV bandpass LD coefficient y",             KIND_PARAMETER,  "phoebe_rvno",  0.0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY, 0.5);

	phoebe_parameter_add ("phoebe_spots_no",             "Number of spots in the model",                       KIND_MODIFIER,   NULL,                  0,      0,      0, NO, TYPE_INT,             0);
	phoebe_parameter_add ("phoebe_spots_active_switch",  "Should the spot be included in the model",           KIND_SWITCH,     "phoebe_spots_no",     0,      0,      0, NO, TYPE_BOOL_ARRAY,   TRUE);
	phoebe_parameter_add ("phoebe_spots_tba_switch",     "Spot adjustment switch (informational only)",        KIND_SWITCH,     "phoebe_spots_no",     0,      0,      0, NO, TYPE_BOOL_ARRAY,  FALSE);

	phoebe_parameter_add ("phoebe_spots_source",          "Star on which the spot is located (1 or 2)",         KIND_PARAMETER,  "phoebe_spots_no",     1,      2,      1, NO, TYPE_INT_ARRAY,         1);
	phoebe_parameter_add ("phoebe_spots_colatitude",      "Spot co-latitude (0 at +z pole, pi at -z pole)",     KIND_PARAMETER,  "phoebe_spots_no",     0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   1.57);
	phoebe_parameter_add ("phoebe_spots_colatitude_tba",  "Spot co-latitude TBA switch",                        KIND_PARAMETER,  "phoebe_spots_no",     0,      0,      0, NO, TYPE_BOOL_ARRAY,    FALSE);
	phoebe_parameter_add ("phoebe_spots_colatitude_min",  "Spot co-latitude minimum value",                     KIND_PARAMETER,  "phoebe_spots_no",     0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.00);
	phoebe_parameter_add ("phoebe_spots_colatitude_max",  "Spot co-latitude maximum value",                     KIND_PARAMETER,  "phoebe_spots_no",     0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   M_PI);
	phoebe_parameter_add ("phoebe_spots_colatitude_step", "Spot co-latitude adjustment step",                   KIND_PARAMETER,  "phoebe_spots_no",     0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.01);
	phoebe_parameter_add ("phoebe_spots_longitude",       "Spot longitude (0 at +x, to 2pi in CCW direction)",  KIND_PARAMETER,  "phoebe_spots_no",     0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.00);
	phoebe_parameter_add ("phoebe_spots_longitude_tba",   "Spot longitude TBA switch",                          KIND_PARAMETER,  "phoebe_spots_no",     0,      0,      0, NO, TYPE_BOOL_ARRAY,    FALSE);
	phoebe_parameter_add ("phoebe_spots_longitude_min",   "Spot longitude minimum value",                       KIND_PARAMETER,  "phoebe_spots_no",     0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.00);
	phoebe_parameter_add ("phoebe_spots_longitude_max",   "Spot longitude maximum value",                       KIND_PARAMETER,  "phoebe_spots_no",     0, 2*M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY, 2*M_PI);
	phoebe_parameter_add ("phoebe_spots_longitude_step",  "Spot longitude adjustment step",                     KIND_PARAMETER,  "phoebe_spots_no",     0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.01);
	phoebe_parameter_add ("phoebe_spots_radius",          "Spot angular radius (in radians)",                   KIND_PARAMETER,  "phoebe_spots_no",     0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.20);
	phoebe_parameter_add ("phoebe_spots_radius_tba",      "Spot angular radius TBA switch",                     KIND_PARAMETER,  "phoebe_spots_no",     0,      0,      0, NO, TYPE_BOOL_ARRAY,    FALSE);
	phoebe_parameter_add ("phoebe_spots_radius_min",      "Spot angular radius minimum value",                  KIND_PARAMETER,  "phoebe_spots_no",     0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.00);
	phoebe_parameter_add ("phoebe_spots_radius_max",      "Spot angular radius maximum value",                  KIND_PARAMETER,  "phoebe_spots_no",     0,   M_PI,   0.01, NO, TYPE_DOUBLE_ARRAY,   M_PI);
	phoebe_parameter_add ("phoebe_spots_radius_step",     "Spot angular radius adjustment step",                KIND_PARAMETER,  "phoebe_spots_no",     0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.01);
	phoebe_parameter_add ("phoebe_spots_tempfactor",      "Spot temperature factor (Tspot/Tsurface)",           KIND_PARAMETER,  "phoebe_spots_no",     0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.90);
	phoebe_parameter_add ("phoebe_spots_tempfactor_tba",  "Spot temperature factor TBA switch",                 KIND_PARAMETER,  "phoebe_spots_no",     0,      0,      0, NO, TYPE_BOOL_ARRAY,    FALSE);
	phoebe_parameter_add ("phoebe_spots_tempfactor_min",  "Spot temperature factor minimum value",              KIND_PARAMETER,  "phoebe_spots_no",     0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.00);
	phoebe_parameter_add ("phoebe_spots_tempfactor_max",  "Spot temperature factor maximum value",              KIND_PARAMETER,  "phoebe_spots_no",     0,    100,   0.01, NO, TYPE_DOUBLE_ARRAY,    100);
	phoebe_parameter_add ("phoebe_spots_tempfactor_step", "Spot temperature factor adjustment step",            KIND_PARAMETER,  "phoebe_spots_no",     0,    1.0,   0.01, NO, TYPE_DOUBLE_ARRAY,   0.01);

	phoebe_parameter_add ("phoebe_spots_units",          "Spot coordinate and radius units",                   KIND_MENU,       NULL,           0.0,    0.0,    0.0, NO, TYPE_STRING,        "Radians");
	phoebe_parameter_add ("phoebe_spots_corotate1",      "Spots on star 1 co-rotate with the star",            KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,         YES);
	phoebe_parameter_add ("phoebe_spots_corotate2",      "Spots on star 2 co-rotate with the star",            KIND_SWITCH,     NULL,    0,      0,      0, NO, TYPE_BOOL,         YES);

	/* These pertain to WD's DC that can fit up to two spots simultaneously. */
	phoebe_parameter_add ("wd_spots_lat1",               "Latitude of the 1st adjusted spot",                  KIND_ADJUSTABLE, NULL,                0.0,   M_PI,   0.01, NO, TYPE_DOUBLE, 0.0);
	phoebe_parameter_add ("wd_spots_long1",              "Longitude of the 1st adjusted spot",                 KIND_ADJUSTABLE, NULL,                0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE, 0.0);
	phoebe_parameter_add ("wd_spots_rad1",               "Radius of the 1st adjusted spot",                    KIND_ADJUSTABLE, NULL,                0.0,   M_PI,   0.01, NO, TYPE_DOUBLE, 0.2);
	phoebe_parameter_add ("wd_spots_temp1",              "Temperature of 1st adjusted spot",                   KIND_ADJUSTABLE, NULL,                0.0,    100,   0.01, NO, TYPE_DOUBLE, 0.9);
	phoebe_parameter_add ("wd_spots_lat2",               "Latitude of the 2nd adjusted spot",                  KIND_ADJUSTABLE, NULL,                0.0,   M_PI,   0.01, NO, TYPE_DOUBLE, 0.0);
	phoebe_parameter_add ("wd_spots_long2",              "Longitude of the 2nd adjusted spot",                 KIND_ADJUSTABLE, NULL,                0.0, 2*M_PI,   0.01, NO, TYPE_DOUBLE, 0.0);
	phoebe_parameter_add ("wd_spots_rad2",               "Radius of the 2nd adjusted spot",                    KIND_ADJUSTABLE, NULL,                0.0,   M_PI,   0.01, NO, TYPE_DOUBLE, 0.2);
	phoebe_parameter_add ("wd_spots_temp2",              "Temperature of 2nd adjusted spot",                   KIND_ADJUSTABLE, NULL,                0.0,    100,   0.01, NO, TYPE_DOUBLE, 0.9);

	/* *********************   Utilities parameters   *********************** */

	phoebe_parameter_add ("phoebe_synscatter_switch",    "Synthetic scatter",                                  KIND_SWITCH,     NULL,                  0,      0,      0, NO, TYPE_BOOL,         NO);
	phoebe_parameter_add ("phoebe_synscatter_sigma",     "Synthetic scatter standard deviation",               KIND_PARAMETER,  NULL,                0.0,  100.0,   0.01, NO, TYPE_DOUBLE,       0.01);
	phoebe_parameter_add ("phoebe_synscatter_seed",      "Synthetic scatter seed",                             KIND_PARAMETER,  NULL,                1E8,    1E9,      1, NO, TYPE_DOUBLE,       1.5E8);
	phoebe_parameter_add ("phoebe_synscatter_levweight", "Synthetic scatter weighting",                        KIND_MENU,       NULL,                  0,      0,      0, NO, TYPE_STRING,       "Poissonian scatter");

	/* **********************   Computed parameters   *********************** */

	phoebe_parameter_add ("phoebe_plum1",                "Primary star active passband luminosity",            KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_plum2",                "Secondary star active passband luminosity",          KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_mass1",                "Primary star mass",                                  KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_mass2",                "Secondary star mass",                                KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_radius1",              "Primary star radius",                                KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_radius2",              "Secondary star radius",                              KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_mbol1",                "Primary star absolute bolometric magnitude",         KIND_COMPUTED,   NULL,               -100,    100,    0.0, NO, TYPE_DOUBLE,       0.0);
	phoebe_parameter_add ("phoebe_mbol2",                "Secondary star absolute bolometric magnitude",       KIND_COMPUTED,   NULL,               -100,    100,    0.0, NO, TYPE_DOUBLE,       0.0);
	phoebe_parameter_add ("phoebe_logg1",                "Primary star surface gravity",                       KIND_COMPUTED,   NULL,                0.0,    100,    0.0, NO, TYPE_DOUBLE,       4.3);
	phoebe_parameter_add ("phoebe_logg2",                "Secondary star surface gravity",                     KIND_COMPUTED,   NULL,                0.0,    100,    0.0, NO, TYPE_DOUBLE,       4.3);
	phoebe_parameter_add ("phoebe_sbr1",                 "Primary star polar surface brightness",              KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);
	phoebe_parameter_add ("phoebe_sbr2",                 "Secondary star polar surface brightness",            KIND_COMPUTED,   NULL,                0.0,   1E10,    0.0, NO, TYPE_DOUBLE,       1.0);

	return SUCCESS;
}

int phoebe_free_parameters ()
{
	/**
	 * phoebe_free_parameters:
	 *
	 * Frees all parameters from the currently active parameter table pointed
	 * to by a global variable @PHOEBE_pt.
	 *
	 * Returns: #PHOEBE_error_code.
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
		passband_str = phoebe_concatenate_strings (PHOEBE_passbands[i]->set, ":", PHOEBE_passbands[i]->name, NULL);
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
	phoebe_parameter_add_option (par, "None");
	phoebe_parameter_add_option (par, "Poissonian scatter");
	phoebe_parameter_add_option (par, "Low light scatter");

	par = phoebe_parameter_lookup ("phoebe_rv_filter");
	for (i = 0; i < PHOEBE_passbands_no; i++) {
		passband_str = phoebe_concatenate_strings (PHOEBE_passbands[i]->set, ":", PHOEBE_passbands[i]->name, NULL);
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
	phoebe_parameter_add_option (par, "None");
	phoebe_parameter_add_option (par, "Poissonian scatter");
	phoebe_parameter_add_option (par, "Low light scatter");

	par = phoebe_parameter_lookup ("phoebe_passband_mode");
	phoebe_parameter_add_option (par, "None");
	phoebe_parameter_add_option (par, "Interpolation");
	phoebe_parameter_add_option (par, "Rigorous");

	par = phoebe_parameter_lookup ("phoebe_el3_units");
	phoebe_parameter_add_option (par, "Total light");
	phoebe_parameter_add_option (par, "Flux");

	par = phoebe_parameter_lookup ("phoebe_spots_units");
	phoebe_parameter_add_option (par, "Radians");
	phoebe_parameter_add_option (par, "Degrees");

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
			par->defaultvalue.i = va_arg (args, int);
			par->value.i = par->defaultvalue.i;
		break;
		case TYPE_DOUBLE:
			par->defaultvalue.d = va_arg (args, double);
			par->value.d = par->defaultvalue.d;
		break;
		case TYPE_BOOL:
			par->defaultvalue.b = va_arg (args, bool);
			par->value.b = par->defaultvalue.b;
		break;
		case TYPE_STRING: {
			char *str = va_arg (args, char *);
			par->defaultvalue.str = phoebe_malloc (strlen (str) + 1);
			strcpy (par->defaultvalue.str, str);
			par->value.str = strdup (par->defaultvalue.str);
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
		default:
			phoebe_lib_error ("type %s not supported for parameters, aborting.\n", phoebe_type_get_name (par->type));
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

	phoebe_debug ("  option \"%s\" added to parameter %s.\n", option, par->qualifier);
	return SUCCESS;
}

int phoebe_qualifier_string_parse (char *input, char **qualifier, int *index)
{
	/**
	 * phoebe_qualifier_string_parse:
	 * @input: the string of the form "qualifier[element]" to be parsed.
	 * @qualifier: the newly allocated string that holds the qualifier.
	 * @index: the qualifier index.
	 *
	 * Parses the input string of the form qualifier[index]. The tokens are
	 * assigned to the passed arguments @qualifier (which is allocated) and
	 * @index. The allocated string needs to be freed by the calling function.
	 * Also, this function merely parses a passed string, it does not check
	 * for the qualifier and element validity. If the qualifier is a scalar,
	 * @index is set to 0.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	char *delim;

	int length;

	if (!input)
		return ERROR_QUALIFIER_STRING_IS_NULL;

	if ( !(delim = strchr (input, '[')) ) {
		/* The qualifier is a scalar. */
		*qualifier = strdup (input);
		*index = 0;
		return SUCCESS;
	}

	length = strlen(input)-strlen(delim);

	*qualifier = phoebe_malloc ( (length+1) * sizeof (**qualifier));
	strncpy (*qualifier, input, length);
	(*qualifier)[length] = '\0';

	if ( (sscanf (delim, "[%d]", index)) != 1)
		return ERROR_QUALIFIER_STRING_MALFORMED;

	return SUCCESS;
}

bool phoebe_qualifier_is_constrained (char *qualifier)
{
	/**
	 * phoebe_qualifier_is_constrained
	 *
	 * This function checks whether a passed qualifier also appears in the
	 * list of constraints. If so, it returns true; else it returns false.
	 */

	PHOEBE_ast_list *constraint = PHOEBE_pt->lists.constraints;

	while (constraint) {
		phoebe_debug ("comparing %s and %s\n", phoebe_constraint_get_qualifier (constraint->elem), qualifier);
		if (strcmp (phoebe_constraint_get_qualifier (constraint->elem), qualifier) == 0)
			return TRUE;
		constraint = constraint->next;
	}

	return FALSE;
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

	phoebe_debug ("  parameter %s added to bucket %d.\n", par->qualifier, hash);
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
		par->type == TYPE_BOOL_ARRAY)  {
		phoebe_array_free (par->value.array);
	}

	if (par->type == TYPE_DOUBLE_ARRAY)
		phoebe_vector_free (par->value.vec);

	/* Free linked list elements, but not stored parameters: */
	while (par->deps) {
		PHOEBE_parameter_list *list = par->deps->next;
		free (par->deps);
		par->deps = list;
	}

	free (par);

	return SUCCESS;
}

int phoebe_parameter_option_get_index (PHOEBE_parameter *par, char *option, int *index)
{
	/**
	 * phoebe_parameter_option_get_index:
	 * @par: #PHOEBE_parameter being queried
	 * @option: the option to be looked up
	 * @index: a pointer to the index to be assigned
	 *
	 * Scans through all the options of the #PHOEBE_parameter @par and returns
	 * the index of the @option. The @par's kind must be #KIND_MENU. If the
	 * option is not found, the @index is set to -1 and
	 * #ERROR_PARAMETER_OPTION_DOES_NOT_EXIST is returned.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i;

	if (!par)
		return ERROR_PARAMETER_NOT_INITIALIZED;

	if (par->kind != KIND_MENU)
		return ERROR_PARAMETER_KIND_NOT_MENU;

	for (i = 0; i <= par->menu->optno; i++) {
		if (i == par->menu->optno) {
			*index = -1;
			return ERROR_PARAMETER_OPTION_DOES_NOT_EXIST;
		}
		if (strcmp (par->menu->option[i], option) == 0) {
			*index = i;
			return SUCCESS;
		}
	}

	return SUCCESS;
}

int phoebe_parameter_update_deps (PHOEBE_parameter *par, int oldval)
{
	/**
	 * phoebe_parameter_update_deps:
	 *
	 * @par: #PHOEBE_parameter that has been changed
	 * @oldval: original value of the parameter @par
	 *
	 * Called whenever the dimension of parameter arrays must be changed.
	 * Typically this happens when the number of observed data curves
	 * changed, the number of spots is changed etc.
	 *
	 * Returns: #PHOEBE_error_code.
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
					status = phoebe_array_realloc (list->par->value.array, dim);
					if (status != SUCCESS) phoebe_lib_error ("%s", phoebe_error (status));
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
					status = phoebe_vector_realloc (list->par->value.vec, dim);
					if (status != SUCCESS) phoebe_lib_error ("%s", phoebe_error (status));
					for (j = oldval; j < dim; j++)
						list->par->value.vec->val[j] = list->par->defaultvalue.d;
				break;
				case TYPE_STRING_ARRAY:
					status = phoebe_array_realloc (list->par->value.array, dim);
					if (status != SUCCESS) phoebe_lib_error ("%s", phoebe_error (status));
					for (j = oldval; j < dim; j++)
						list->par->value.array->val.strarray[j] = strdup (list->par->defaultvalue.str);
				break;
				default:
					phoebe_lib_error ("dependent parameter is not an array, aborting.\n");
					return ERROR_INVALID_TYPE;
		}

		list = list->next;
	}

	return SUCCESS;
}

bool phoebe_parameter_option_is_valid (char *qualifier, char *option)
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
	/**
	 * phoebe_parameter_get_value:
	 * @par: #PHOEBE_parameter to be queried
	 * @...: index and a variable of the corresponding type in case of
	 *       passband-dependent parameters, or a variable of the corresponding
	 *       type in case of passband-independent parameters.
	 *
	 * Synopsis:
	 *
	 *   phoebe_parameter_get_value (par, [index, ], &value)
	 *
	 * Assigns the value of the passed parameter @par to the passed variable
	 * @value. In case of strings pointers are returned, so you should never
	 * free the variable that has been assigned.
	 *
	 * Returns: #PHOEBE_error_code
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
		case TYPE_INT_ARRAY: {
			int *value;
			index = va_arg (args, int);
			value = va_arg (args, int *);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.array->val.iarray[index];
		}
		break;
		case TYPE_BOOL_ARRAY: {
			bool *value;
			index = va_arg (args, int);
			value = va_arg (args, bool *);
			if (index < 0 || index > par->value.array->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.array->val.barray[index];
		}
		break;
		case TYPE_DOUBLE_ARRAY: {
			double *value;
			index = va_arg (args, int);
			value = va_arg (args, double *);
			if (index < 0 || index > par->value.vec->dim-1)
				return ERROR_INDEX_OUT_OF_RANGE;
			*value = par->value.vec->val[index];
		}
		break;
		case TYPE_STRING_ARRAY: {
			const char **value;
			index = va_arg (args, int);
			value = va_arg (args, const char **);
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
	/**
	 * phoebe_parameter_set_value:
	 * @par: #PHOEBE_parameter to be set.
	 * @...: an optional curve index and an #PHOEBE_value value.
	 *
	 * This is a public function for changing the value of the passed
	 * parameter @par. The function also satisfies all constraints.
	 *
	 * Synopsis:
	 *
	 *   phoebe_parameter_set_value (qualifier, [index, ] value)
	 *
	 * Returns: #PHOEBE_error_code.
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
			if (!value) value = strdup ("Undefined");
			free (par->value.str);
			par->value.str = phoebe_malloc (strlen (value) + 1);
			strcpy (par->value.str, value);

			/*
			 * If the passed parameter is a menu, let's check if the option
			 * is valid. If it is not, just warn about it, but set its value
			 * anyway.
			 */

			if (par->kind == KIND_MENU && !phoebe_parameter_option_is_valid (par->qualifier, (char *) value))
				phoebe_lib_warning ("option \"%s\" is not a valid menu option for %s.\n", value, par->qualifier);
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
			if (!value) value = strdup ("Undefined");
			free (par->value.array->val.strarray[index]);
			par->value.array->val.strarray[index] = phoebe_malloc (strlen (value) + 1);
			strcpy (par->value.array->val.strarray[index], value);

			/*
			 * If the passed parameter is a menu, let's check if the option is
			 * valid. If it is not, just warn about it, but set its value
			 * anyway.
			 */

			if (par->kind == KIND_MENU && !phoebe_parameter_option_is_valid (par->qualifier, (char *) value))
				phoebe_lib_warning ("option \"%s\" is not a valid menu option.\n", value);
			}
		break;
		default:
			phoebe_lib_error ("parameter type %s is not supported, aborting.\n", phoebe_type_get_name (par->type));
			return ERROR_INVALID_TYPE;
	}
	va_end (args);

	/* Satisfy all constraints: */
	phoebe_constraint_satisfy_all ();

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

int phoebe_parameter_list_sort_marked_tba (PHOEBE_parameter_list *list)
{
	PHOEBE_parameter_list *prev = NULL, *next;
	int i, id1 = 0, id2 = 0;

	struct { int index; PHOEBE_parameter *par; } wdorder[] = {
		{ 0, phoebe_parameter_lookup ("wd_spots_lat1") },
		{ 1, phoebe_parameter_lookup ("wd_spots_long1")},
		{ 2, phoebe_parameter_lookup ("wd_spots_rad1") },
		{ 3, phoebe_parameter_lookup ("wd_spots_temp1")},
		{ 4, phoebe_parameter_lookup ("wd_spots_lat2") },
		{ 5, phoebe_parameter_lookup ("wd_spots_long2")},
		{ 6, phoebe_parameter_lookup ("wd_spots_rad2") },
		{ 7, phoebe_parameter_lookup ("wd_spots_temp2")},
		{ 8, phoebe_parameter_lookup ("phoebe_sma")        },
		{ 9, phoebe_parameter_lookup ("phoebe_ecc")        },
		{10, phoebe_parameter_lookup ("phoebe_perr0")      },
		{11, phoebe_parameter_lookup ("phoebe_f1")         },
		{12, phoebe_parameter_lookup ("phoebe_f2")         },
		{13, phoebe_parameter_lookup ("phoebe_pshift")     },
		{14, phoebe_parameter_lookup ("phoebe_vga")        },
		{15, phoebe_parameter_lookup ("phoebe_incl")       },
		{16, phoebe_parameter_lookup ("phoebe_grb1")       },
		{17, phoebe_parameter_lookup ("phoebe_grb2")       },
		{18, phoebe_parameter_lookup ("phoebe_teff1")      },
		{19, phoebe_parameter_lookup ("phoebe_teff2")      },
		{20, phoebe_parameter_lookup ("phoebe_alb1")       },
		{21, phoebe_parameter_lookup ("phoebe_alb2")       },
		{22, phoebe_parameter_lookup ("phoebe_pot1")       },
		{23, phoebe_parameter_lookup ("phoebe_pot2")       },
		{24, phoebe_parameter_lookup ("phoebe_rm")         },
		{25, phoebe_parameter_lookup ("phoebe_hjd0")       },
		{26, phoebe_parameter_lookup ("phoebe_period")     },
		{27, phoebe_parameter_lookup ("phoebe_dpdt")       },
		{28, phoebe_parameter_lookup ("phoebe_dperdt")     },
		{29, NULL                                          },
		{30, phoebe_parameter_lookup ("phoebe_hla")        },
		{31, phoebe_parameter_lookup ("phoebe_cla")        },
		{32, phoebe_parameter_lookup ("phoebe_ld_lcx1")    },
		{33, phoebe_parameter_lookup ("phoebe_ld_lcx2")    },
		{34, phoebe_parameter_lookup ("phoebe_el3")        }
	};

	if (!list) return SUCCESS;

	next = list->next;

	while (next) {
		for (i = 0; i <= 34; i++)
			if (list->par == wdorder[i].par) {
				id1 = i;
				break;
			}
		for (i = 0; i <= 34; i++)
			if (next->par == wdorder[i].par) {
				id2 = i;
				break;
			}
		phoebe_debug ("comparing %s (%d) and %s (%d).\n", wdorder[id1].par->qualifier, id1, wdorder[id2].par->qualifier, id2);
		if (id1 > id2) {
			phoebe_debug ("sorting %s (%d) and %s (%d).\n", wdorder[id1].par->qualifier, id1, wdorder[id2].par->qualifier, id2);
			list->next = next->next;
			next->next = list;
			if (prev)
				prev->next = next;
			else
				PHOEBE_pt->lists.marked_tba = next;

			prev = next;
			next = list->next;
		}
		else {
			prev = list;
			list = list->next;
			next = list->next;
		}
	}

	return SUCCESS;
}

int phoebe_parameter_set_tba (PHOEBE_parameter *par, bool tba)
{
	/*
	 * This is the public function for changing the passed parameter's TBA
	 * (To Be Adjusted) bit. At the same time the function adds or removes
	 * that parameter from the list of parameters marked for adjustment and
	 * sorts that list in order of parameter index in WD.
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

	list = phoebe_parameter_list_get_marked_tba ();
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
			phoebe_debug ("  parameter %s added to the tba list.\n", list->par->qualifier);
			phoebe_parameter_list_sort_marked_tba (list);
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

int phoebe_parameter_get_min (PHOEBE_parameter *par, double *valmin)
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

int phoebe_parameter_set_min (PHOEBE_parameter *par, double valmin)
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

int phoebe_parameter_get_max (PHOEBE_parameter *par, double *valmax)
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

int phoebe_parameter_set_max (PHOEBE_parameter *par, double valmax)
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

PHOEBE_parameter_list *phoebe_parameter_list_reverse (PHOEBE_parameter_list *c, PHOEBE_parameter_list *p)
{
	/**
	 *
	 */

	PHOEBE_parameter_list *rev;

	if (!c) return p;
	rev = phoebe_parameter_list_reverse (c->next, c);
	c->next = p;

	return rev;
}

PHOEBE_parameter_table *phoebe_parameter_table_new ()
{
	/**
	 * phoebe_parameter_table_new:
	 *
	 * Initializes a new parameter table and adds it to the list of all
	 * parameter tables.
	 *
	 * Returns: #PHOEBE_parameter_table.
	 */

	int i;
	PHOEBE_parameter_table      *table = phoebe_malloc (sizeof (*table));
	PHOEBE_parameter_table_list *list  = phoebe_malloc (sizeof (*list));

	phoebe_debug ("* creating a new parameter table at address %p.\n", table);

	/* Initialize all buckets: */
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++)
		table->bucket[i] = NULL;

	/* NULLify the attached lists: */
	table->lists.marked_tba  = NULL;
	table->lists.constraints = NULL;

	/* Add the table to the list of tables: */
	list->table = table;
	list->next = PHOEBE_pt_list;
	PHOEBE_pt_list = list;

	return table;
}

PHOEBE_parameter_table *phoebe_parameter_table_duplicate (PHOEBE_parameter_table *table)
{
	/**
	 * phoebe_parameter_table_duplicate:
	 * @table: a #PHOEBE_parameter_table to duplicate.
	 *
	 * Makes a duplicate copy of the table @table.
	 *
	 * Returns: #PHOEBE_parameter_table.
	 */

	int i, j;

	PHOEBE_parameter_table *copy = phoebe_parameter_table_new ();
	PHOEBE_parameter_list *list, *elem, *tba, *tbacopy, *deps, *depscopy;
	PHOEBE_ast_list *constraints_src, *constraints_dest;

	/* First pass: copy all parameters and their values. */
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		elem = table->bucket[i];
		while (elem) {
			list = phoebe_malloc (sizeof (*list));
			list->par = phoebe_parameter_new ();
			list->par->qualifier    = strdup (elem->par->qualifier);
			list->par->description  = strdup (elem->par->description);
			list->par->kind         = elem->par->kind;
			list->par->type         = elem->par->type;
			list->par->value        = phoebe_value_duplicate (elem->par->type, elem->par->value);
			list->par->min          = elem->par->min;
			list->par->max          = elem->par->max;
			list->par->step         = elem->par->step;
			list->par->tba          = elem->par->tba;
			list->par->defaultvalue = elem->par->defaultvalue;

			if (elem->par->menu) {
				list->par->menu = phoebe_malloc (sizeof (*(list->par->menu)));
				list->par->menu->optno = elem->par->menu->optno;
				list->par->menu->option = phoebe_malloc (list->par->menu->optno * sizeof (*(list->par->menu->option)));
				for (j = 0; j < elem->par->menu->optno; j++)
					list->par->menu->option[j] = strdup (elem->par->menu->option[j]);
			}

			list->next = copy->bucket[i];
			copy->bucket[i] = list;
			elem = elem->next;
		}
		/* The list is copied in reverse; let's fix that: */
		copy->bucket[i] = phoebe_parameter_list_reverse (copy->bucket[i], NULL);
	}

	/* Second pass: copy all dependencies. */
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		elem = table->bucket[i];
		while (elem) {
			deps = elem->par->deps;
			while (deps) {
				depscopy = phoebe_malloc (sizeof (*depscopy));

				/* Parameters need to be looked up in the duplicated table: */
				list = copy->bucket[phoebe_parameter_hash (deps->par->qualifier)];
				while (strcmp (deps->par->qualifier, list->par->qualifier) != 0)
					list = list->next;

				depscopy->par = list->par;
				depscopy->next = elem->par->deps;
				elem->par->deps = depscopy;

				deps = deps->next;
			}
		elem = elem->next;
		}
	}

	/* Final step: copy all lists. */
	tba = table->lists.marked_tba;
	while (tba) {
		tbacopy = phoebe_malloc (sizeof (*tbacopy));

		/* The parameters in the list need to be in the duplicated table: */
		list = copy->bucket[phoebe_parameter_hash (tba->par->qualifier)];
		while (strcmp (tba->par->qualifier, list->par->qualifier) != 0)
			list = list->next;

		tbacopy->par  = list->par;
		tbacopy->next = copy->lists.marked_tba;
		copy->lists.marked_tba = tbacopy;
		tba = tba->next;
	}
	copy->lists.marked_tba = phoebe_parameter_list_reverse (copy->lists.marked_tba, NULL);

	constraints_src = table->lists.constraints;
	while (constraints_src) {
		constraints_dest = phoebe_malloc (sizeof (*constraints_dest));
		constraints_dest->elem = phoebe_ast_duplicate (constraints_src->elem);
		constraints_dest->next = copy->lists.constraints;
		copy->lists.constraints = constraints_dest;
		constraints_src = constraints_src->next;
	}
	copy->lists.constraints = phoebe_ast_list_reverse (copy->lists.constraints, NULL);

	return copy;
}

int phoebe_parameter_table_activate (PHOEBE_parameter_table *table)
{
	/**
	 * phoebe_parameter_table_activate:
	 * @table: parameter table that should be activated.
	 *
	 * Sets a pointer to the currently active parameter table @PHOEBE_pt to
	 * @table.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!table)
		return ERROR_PARAMETER_TABLE_NOT_INITIALIZED;

	PHOEBE_pt = table;
	phoebe_debug ("parameter table at address %p activated.\n", table);

	return SUCCESS;
}

int phoebe_parameter_table_print (PHOEBE_parameter_table *table)
{
	int i;
	PHOEBE_parameter_list *list;
	PHOEBE_ast_list *ast_list;

	printf ("Parameter table:\n");
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		printf ("%3d: ", i);
		list = table->bucket[i];
		while (list) {
			printf ("%s ", list->par->qualifier);
			list = list->next;
		}
		printf ("\n");
	}
	printf ("Parameters that are marked for adjustment:\n");
	list = table->lists.marked_tba;

	while (list) {
		printf ("  %s\n", list->par->qualifier);
		list = list->next;
	}

	printf ("Constraints:\n");
	ast_list = table->lists.constraints;

	while (ast_list) {
		phoebe_ast_print (0, ast_list->elem);
		ast_list = ast_list->next;
	}

	return SUCCESS;
}

int phoebe_parameter_table_free (PHOEBE_parameter_table *table)
{
	/**
	 * phoebe_parameter_table_free:
	 * @table: a #PHOEBE_parameter_table to be freed.
	 *
	 * Frees the parameter table @table and removes it from the list of all
	 * parameter tables. Note that this does not free the parameters contained
	 * in the table; use phoebe_parameters_free () for that.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	PHOEBE_parameter_table_list *list = PHOEBE_pt_list, *prev = NULL;

	while (list->table != table) {
		prev = list;
		list = list->next;
	}

	if (!prev)
		PHOEBE_pt_list = list->next;
	else
		prev->next = list->next;

	free (list);

	free (table);

	return SUCCESS;
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

double phoebe_spots_units_to_wd_conversion_factor ()
{
	char *phoebe_spots_units;
	phoebe_parameter_get_value (phoebe_parameter_lookup ("phoebe_spots_units"), &phoebe_spots_units);
	return (strcmp(phoebe_spots_units, "Radians")) ? M_PI/180.0 : 1.0;

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

	phoebe_debug ("entering function phoebe_open_parameter_file ()\n");

	/* First a checkup if everything is OK with the filename:                   */
	if (!phoebe_filename_exists ((char *) filename))               return ERROR_FILE_NOT_FOUND;
	if (!phoebe_filename_is_regular_file ((char *) filename))      return ERROR_FILE_NOT_REGULAR;
	if (!phoebe_filename_has_read_permissions ((char *) filename)) return ERROR_FILE_NO_PERMISSIONS;

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
			phoebe_debug ("  opening legacy parameter file.\n");
			status = phoebe_open_legacy_parameter_file (filename);
			return status;
		}
		phoebe_debug ("  PHOEBE parameter file version: %2.2lf\n", version);
	}

	/* Set default parameters, not provided in earlier versions */
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_spots_units"), "Radians");

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

		value_str = strchr (readout_str, '=');
		if (value_str == NULL) {
			/* If the keyword doesn't have '=', it will be skipped.                 */
			phoebe_lib_error ("qualifier %s initialization (line %d) is invalid.\n", keyword_str, lineno);
			continue;
		}
		strncpy (keyword_str, readout_str, strlen(readout_str)-strlen(value_str)+1);
		keyword_str[strlen(readout_str)-strlen(value_str)] = '\0';
		while (keyword_str[strlen(keyword_str)-1] == ' ' || keyword_str[strlen(keyword_str)-1] == '\t') keyword_str[strlen(keyword_str)-1] = '\0';

		/* value_str now points to '=', we need the next character: */
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

		if ( (elem_sep = strchr (keyword_str, '[')) && (field_sep = strchr (keyword_str, '.')) ) {
			qualifier = phoebe_malloc ( (strlen(keyword_str)-strlen(elem_sep)+1)*sizeof(*qualifier) );
			strncpy (qualifier, keyword_str, strlen(keyword_str)-strlen(elem_sep));
			qualifier[strlen(keyword_str)-strlen(elem_sep)] = '\0';
			sscanf (elem_sep, "[%d]", &elem);
			field = phoebe_malloc (strlen(field_sep)*sizeof(*field));
			strcpy (field, field_sep+1);
			field[strlen(field_sep)-1] = '\0';

			phoebe_debug ("qualifier: %s; curve: %d; field: %s\n", qualifier, elem, field);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_warning ("qualifier %s not recognized, ignoring.\n", qualifier);
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
						phoebe_parameter_set_min (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_max (par, atof (value_str));
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
						phoebe_parameter_set_min (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_max (par, atof (value_str));
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
						phoebe_parameter_set_min (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_max (par, atof (value_str));
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
						phoebe_parameter_set_min (par, atof (value_str));
					if (strcmp (field,  "MAX") == 0)
						phoebe_parameter_set_max (par, atof (value_str));
					if (strcmp (field, "STEP") == 0)
						phoebe_parameter_set_step (par, atof (value_str));
					if (strcmp (field,  "ADJ") == 0)
						phoebe_parameter_set_tba (par, atob (value_str));
				break;
				default:
					phoebe_lib_error ("dependent parameter is not an array, aborting.\n");
					return ERROR_INVALID_TYPE;
			}
		}
		else if ( (elem_sep = strchr (keyword_str, '[')) ) {
			qualifier = phoebe_malloc ( (strlen(keyword_str)-strlen(elem_sep)+1)*sizeof(*qualifier) );
			strncpy (qualifier, keyword_str, strlen(keyword_str)-strlen(elem_sep));
			qualifier[strlen(keyword_str)-strlen(elem_sep)] = '\0';
			sscanf (elem_sep, "[%d]", &elem);

			phoebe_debug ("qualifier: %s; curve: %d\n", qualifier, elem);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_warning ("qualifier %s not recognized, ignoring.\n", qualifier);
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
				default:
					phoebe_lib_error ("dependent parameter is not an array, aborting.\n");
					return ERROR_INVALID_TYPE;
			}
		}
		else if ( (field_sep = strchr (keyword_str, '.')) ) {
			qualifier = phoebe_malloc ( (strlen(keyword_str)-strlen(field_sep)+1)*sizeof(*qualifier) );
			strncpy (qualifier, keyword_str, strlen(keyword_str)-strlen(field_sep));
			qualifier[strlen(keyword_str)-strlen(field_sep)] = '\0';

			field = phoebe_malloc (strlen(field_sep)*sizeof(*field));
			strcpy (field, field_sep+1);
			field[strlen(field_sep)-1] = '\0';

			phoebe_debug ("qualifier: %s; field: %s\n", qualifier, field);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_warning ("qualifier %s not recognized, ignoring.\n", qualifier);
				free (qualifier);
				free (field);
				continue;
			}

			if (strcmp (field,  "VAL") == 0)
				phoebe_parameter_set_value (par, atof (value_str));
			if (strcmp (field,  "MIN") == 0)
				phoebe_parameter_set_min (par, atof (value_str));
			if (strcmp (field,  "MAX") == 0)
				phoebe_parameter_set_max (par, atof (value_str));
			if (strcmp (field, "STEP") == 0)
				phoebe_parameter_set_step (par, atof (value_str));
			if (strcmp (field,  "ADJ") == 0)
				phoebe_parameter_set_tba (par, atob (value_str));
		}
		else {
			qualifier = phoebe_malloc ((strlen(keyword_str)+1)*sizeof(*qualifier));
			strcpy (qualifier, keyword_str);

			par = phoebe_parameter_lookup (qualifier);
			if (!par) {
				phoebe_lib_warning ("qualifier %s not recognized, ignoring.\n", qualifier);
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

		if (qualifier) { free (qualifier); qualifier = NULL; }
		if (field)     { free (field);     field     = NULL; }
		}
	}

	fclose (keyword_file);

	phoebe_debug ("leaving function phoebe_open_parameter_file ()\n");

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
				if (strcmp(par->qualifier, "phoebe_period"))
					fprintf (file, "%s.VAL  = %lf\n", par->qualifier, par->value.d);
				else
					fprintf (file, "%s.VAL  = %.10lf\n", par->qualifier, par->value.d);
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

	/* Traverse the parameter table and save all modifiers first: */
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		elem = PHOEBE_pt->bucket[i];
		while (elem) {
			if (elem->par->kind == KIND_MODIFIER)
				intern_save_to_parameter_file (elem->par, parameter_file);
			elem = elem->next;
		}
	}

	/* Traverse the parameter table again and save the remaining parameters: */
	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		elem = PHOEBE_pt->bucket[i];
		while (elem) {
			if (elem->par->kind != KIND_MODIFIER)
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

int phoebe_restore_default_parameters ()
{
	int i;
	PHOEBE_parameter_list *elem;

	for (i = 0; i < PHOEBE_PT_HASH_BUCKETS; i++) {
		elem = PHOEBE_pt->bucket[i];
		while (elem) {
			switch (elem->par->type) {
				case TYPE_INT:
					phoebe_parameter_set_value (elem->par, elem->par->defaultvalue.i);
				break;
				case TYPE_BOOL:
					phoebe_parameter_set_value (elem->par, elem->par->defaultvalue.b);
				break;
				case TYPE_DOUBLE:
					phoebe_parameter_set_value (elem->par, elem->par->defaultvalue.d);
				break;
				case TYPE_STRING:
					phoebe_parameter_set_value (elem->par, elem->par->defaultvalue.str);
				break;
				default:
					/* there will be no arrays after this. */
				break;
			}
			elem = elem->next;
		}
	}

	return SUCCESS;
}

int phoebe_parameter_file_import_bm3 (const char *bm3file, const char *datafile)
{
	/**
	 * phoebe_parameter_file_import_bm3:
	 * @bm3file: Binary Maker 3 (bm3) filename
	 * @datafile: Observed curve filename
	 *
	 * Imports Dave Bradstreet's Binary Maker 3 parameter file.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	FILE *bm3input;
	char line[255];
	int rint;
	double rdouble;
	double ff1 = -1.0, ff2 = -1.0; /* Fillout factors, to get the morphology */

	bm3input = fopen (bm3file, "r");
	if (!bm3input)
		return ERROR_FILE_OPEN_FAILED;

	phoebe_restore_default_parameters ();

	/* These are the parameters BM3 assumes by default: */
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lcno"), 1);
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_ld_model"), "Linear cosine law");
	phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_el3_units"), "Flux");

	/* If datafile is passed, initialize it: */
	if (datafile)
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_filename"), 0, datafile);

	do {
		fgets (line, 255, bm3input);
		if (feof (bm3input)) break;
		if (strstr (line, "GEOMETRY")) {
			phoebe_debug ("parameter GEOMETRY has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "LATITUDE_GRID")) {
			sscanf (line, "LATITUDE_GRID=%d", &rint);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_n1"), rint);
			continue;
		}
		if (strstr (line, "LONGITUDE_GRID")) {
			phoebe_debug ("parameter LONGITUDE_GRID has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "MASS_RATIO")) {
			sscanf (line, "MASS_RATIO=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_rm"), rdouble);
			continue;
		}
		if (strstr (line, "INPUT_MODE")) {
			phoebe_debug ("parameter INPUT_MODE has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "OMEGA_1")) {
			sscanf (line, "OMEGA_1=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_pot1"), rdouble);
			continue;
		}
		if (strstr (line, "OMEGA_2")) {
			sscanf (line, "OMEGA_2=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_pot2"), rdouble);
			continue;
		}
		if (strstr (line, "C_1")) {
			phoebe_debug ("parameter C_1 has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "C_2")) {
			phoebe_debug ("parameter C_2 has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "R1_BACK")) {
			phoebe_debug ("parameter R1_BACK has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "R2_BACK")) {
			phoebe_debug ("parameter R2_BACK has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "FILLOUT_G")) {
			phoebe_debug ("parameter FILLOUT_G has no PHOEBE counterpart, skipping.\n");
			sscanf (line, "FILLOUT_G=%lf", &ff1);
			continue;
		}
		if (strstr (line, "FILLOUT_S")) {
			phoebe_debug ("parameter FILLOUT_S has no PHOEBE counterpart, skipping.\n");
			sscanf (line, "FILLOUT_S=%lf", &ff2);
			continue;
		}
		if (strstr (line, "WAVELENGTH")) {
			sscanf (line, "WAVELENGTH=%lf", &rdouble);

			if ((int) rdouble == 8800)
				phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_lc_filter"), 0, "Johnson:I");
				
			continue;
		}
		if (strstr (line, "TEMPERATURE_1")) {
			sscanf (line, "TEMPERATURE_1=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_teff1"), rdouble);
			continue;
		}
		if (strstr (line, "TEMPERATURE_2")) {
			sscanf (line, "TEMPERATURE_2=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_teff2"), rdouble);
			continue;
		}
		if (strstr (line, "GRAVITY_1")) {
			sscanf (line, "GRAVITY_1=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_grb1"), rdouble);
			continue;
		}
		if (strstr (line, "GRAVITY_2")) {
			sscanf (line, "GRAVITY_2=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_grb2"), rdouble);
			continue;
		}
		if (strstr (line, "LIMB_1")) {
			sscanf (line, "LIMB_1=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_ld_lcx1"), 0, rdouble);
			continue;
		}
		if (strstr (line, "LIMB_2")) {
			sscanf (line, "LIMB_2=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_ld_lcx2"), 0, rdouble);
			continue;
		}
		if (strstr (line, "REFLECTION_1")) {
			sscanf (line, "REFLECTION_1=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_alb1"), rdouble);
			continue;
		}
		if (strstr (line, "REFLECTION_2")) {
			sscanf (line, "REFLECTION_2=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_alb2"), rdouble);
			continue;
		}
		if (strstr (line, "THIRD_LIGHT")) {
			sscanf (line, "THIRD_LIGHT=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_el3"), 0, rdouble);
			continue;
		}
		if (strstr (line, "INCLINATION")) {
			sscanf (line, "INCLINATION=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_incl"), rdouble);
			continue;
		}
		if (strstr (line, "NORM_PHASE")) {
			phoebe_debug ("parameter NORM_PHASE has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "PHASE_INCREMENT")) {
			phoebe_debug ("parameter PHASE_INCREMENT has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "USE_ADVANCED_PHASE")) {
			phoebe_debug ("parameter USE_ADVANCED_PHASE has no PHOEBE counterpart, skipping.\n");
			continue;
		}
#warning IMPLEMENT_SPOT_IMPORT_FROM_BM3
		if (strstr (line, "ROTATION_F1")) {
			sscanf (line, "ROTATION_F1=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_f1"), rdouble);
			continue;
		}
		if (strstr (line, "ROTATION_F2")) {
			sscanf (line, "ROTATION_F2=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_f2"), rdouble);
			continue;
		}
		if (strstr (line, "PSEUDOSYNC")) {
			phoebe_debug ("parameter PSEUDOSYNC has no PHOEBE counterpart, skipping.\n");
			continue;
		}
		if (strstr (line, "LONG_OF_PERIASTRON")) {
			sscanf (line, "LONG_OF_PERIASTRON=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_perr0"), rdouble);
			continue;
		}
		if (strstr (line, "ECCENTRICITY")) {
			sscanf (line, "ECCENTRICITY=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_ecc"), rdouble);
			continue;
		}
		if (strstr (line, "ZERO_POINT_OF_PHASE")) {
			sscanf (line, "ZERO_POINT_OF_PHASE=%lf", &rdouble);
			phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_pshift"), rdouble);
			continue;
		}
		if (strstr (line, "USER_NORM_FACTOR")) {
			phoebe_debug ("parameter USER_NORM_FACTOR has no PHOEBE counterpart, skipping.\n");
			continue;
		}
	} while (1);

	fclose (bm3input);

	/* Determine morphology from the fillout factors: */
	if (ff1 < 0 && ff2 < 0)
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_model"), "Detached binary");
	else if (ff1 >= 0 && ff2 < 0)
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_model"), "Semi-detached binary, primary star fills Roche lobe");
	else if (ff1 < 0 && ff2 >= 0)
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_model"), "Semi-detached binary, secondary star fills Roche lobe");
	else
		phoebe_parameter_set_value (phoebe_parameter_lookup ("phoebe_model"), "Overcontact binary not in thermal contact");

	return SUCCESS;
}
