# -*- coding: utf-8 -*-
"""
List of physical constants (SI, CGS and solar)

Many constants are taken from NIST (http://physics.nist.gov/cuu/Constants/index.html)

Others have their explicit reference listed.

See also L{ivs.units.conversions} for advanced options, such as change the base
units are changing the values of the fundamental constants to conventions used
by other programs.

Their are two ways one can use the modules in this module:

    1. Just use the raw SI values in your script (e.g. C{constants.Msol}. For
    convenience, also the cgs values of most constants are given, and
    accessbile via adding the postfix C{'_cgs'}, e.g. C{constants.Msol_cgs}.
    2. If you B{always} want to use cgs values and thus want the default values
    to be cgs without bothering with the postfixes, you can change the values of
    the constants to meet a particular convention via
    L{conversions.set_convention}. After calling
    C{conversions.set_convention('cgs')}, the value of C{constants.Msol} will be
    given in grams instead of kilograms.

The safest way of getting the units of the constants exactly as you want them to
be in a script, is via explicitly converting them to the right convention, e.g.

>>> Msol = conversions.convert(constants.Msol_units,'g',constants.Msol)

This is independent of the way the conventions are set.
"""
_current_convention = 'SI'
_current_values = 'standard'
_current_frequency = 'rad'
# SI-units
# value                    name                          unit          reference
#===============================================================================
cc     = 299792458.        # speed of light              m/s
cc_air = cc/1.0002773      # speed of light in air       m/s           T=15C, standard atmospheric pressure
Msol   = 1.988547e30       # solar mass                  kg            Harmanec & Prsa 2011
Rsol   = 6.95508e8         # solar radius                m             Harmanec & Prsa 2011
Lsol   = 3.846e26          # solar luminosity            W             Harmanec & Prsa 2011
Tsol   = 5779.5747         # solar effective temperature K             Harmanec & Prsa 2011
agesol = 1.44221e+17       # solar age                   s             Bahcall 2005
Mearth = 5.9742e24         # earth mass                  kg            google
Rearth = 6371.e3            # earth radius                m             wikipedia
Mjup   = 1.8986e27         # Jupiter mass                kg            wikipedia
Rjup   = 6.9911e7          # Jupiter radius              m             mesa
Mlun   = 7.346e22          # Lunar mass                  kg            wikipedia
au     = 149597870700.     # astronomical unit           m             Harmanec & Prsa 2011
pc     = 3.085677581503e+16# parsec                      m             Harmanec & Prsa 2011
ly     = 9.460730472580800+15     # light year                  m
hh     = 6.6260689633e-34  # Planck constant             J/s
hhbar  = 1.05457162853e-34 # reduced Planck constant     J/s
kB     = 1.380650424e-23   # Boltzmann constant          J/K
NA     = 6.0221417930e23   # Avogadro constant           1/mol
sigma  = 5.67040040e-8     # Stefan-Boltzmann constant   W/m2/K4       Harmanec & Prsa 2011
GG     = 6.67384e-11       # gravitational constant      m3/kg/s2      Harmanec & Prsa 2011
GGMsol = 1.32712442099e20  # grav. constant x solar mass m3/s2         Harmanec & Prsa 2011
RR     = 8.31447215        # (ideal) gas constant        J/K/mol
aa     = 7.5657e-16        # radiation constant          J/m3/K4
a0     = 52.9177e-12       # Bohr radius of hydrogen     m
ke     = 8.9875517873681764e9 # Coulomb constant         Nm2/C2        Wikipedia
eps0   = 8.854187817620e-12   # Electric constant           F/m           Wikipedia
mu0    = 1.2566370614e-6   # Magnetic constant           N/A2          Wikipedia
alpha  = 137.036           # Fine structure constant     none           http://pntpm3.ulb.ac.be/private/divers.htm#constants
me     = 9.1093829140e-31  # Electron mass               kg            Wikipedia
qe     =-1.60217656535e-19 # Electron charge            C             Wikipedia
mp     = 1.67262177774e-27 # Proton mass                 kg            Wikipedia
qp     = 1.60217656535e-19 # Proton charge               C             Wikipedia

cc_units     = 'm s-1'
cc_air_units = 'm s-1'
Msol_units   = 'kg'
Rsol_units   = 'm'
Lsol_units   = 'kg m2 s-3'
Tsol_units   = 'K'
agesol_units = 's'
Mearth_units = 'kg'
Rearth_units = 'm'
Mjup_units   = 'kg'
Rjup_units   = 'm'
Mlun_units   = 'kg'
au_units     = 'm'
pc_units     = 'm'
ly_units     = 'm'
hh_units     = 'kg m2 s-3'
hhbar_units  = 'kg m2 s-3'
kB_units     = 'kg m2 s-2 K-1'
NA_units     = 'mol-1'
sigma_units  = 'kg s-3 K-4'
GG_units     = 'm3 kg-1 s-2'
GGMsol_units = 'm3 s-2'
RR_units     = 'kg m2 s-2 K-1 mol-1'
aa_units     = 'kg m-1 s-2 K-4'
a0_units     = 'm'
ke_units     = 'N m2 C-2'
eps0_units   = 'F m-1'
mu0_units    = 'T m A-1'
alpha_units  = ''
me_units     = 'kg'
qe_units     = 'C'
mp_units     = 'kg'
qp_units     = 'C'




# CGS-units
# value                        name                          unit           reference
#====================================================================================
cc_cgs     = 29979245800.      # speed of light              cm/s
Msol_cgs   = 1.988547e33       # solar mass                  g              Harmanec & Prsa 2011
Rsol_cgs   = 6.95508e10        # solar radius                cm             Harmanec & Prsa 2011
Lsol_cgs   = 3.846e33          # solar luminosity            erg/s          Harmanec & Prsa 2011
Mearth_cgs = 5.9742e27         # earth mass                  g              google
Rearth_cgs = 6371e5            # earth radius                cm             wikipedia
Mjup_cgs   = 1.8986e30         # Jupiter mass                g              wikipedia
Rjup_cgs   = 6.9911e9          # Jupiter radius              cm             mesa
Mlun_cgs   = 7.346e25          # Lunar mass                  g              wikipedia
au_cgs     = 149597.870700e8   # astronomical unit           cm             Harmanec & Prsa 2011
pc_cgs     = 3.085677581503e+18# parsec                      cm             Harmanec & Prsa 2011
ly_cgs     = 9.4605284e+17     # light year                  cm
hh_cgs     = 6.6260689633e-27  # Planck constant             erg/s
hhbar_cgs  = 1.05457162853e-27 # reduced Planck constant     erg/s
kB_cgs     = 1.380650424e-16   # Boltzmann constant          erg/K
sigma_cgs  = 5.67040040e-5     # Stefan-Boltzmann constant   erg/cm2/s/K4   Harmanec & Prsa 2011
GG_cgs     = 6.67384e-8        # gravitational constant      cm3/g/s2       Harmanec & Prsa 2011
GGMsol_cgs = 1.32712442099e26  # grav. constant x solar mass cm3/s2         Harmanec & Prsa 2011
RR_cgs     = 8.31447215e7      # (ideal) gas constant        erg/K/mol
aa_cgs     = 7.5657e-15        # radiation constant          erg/cm2/K4
a0_cgs     = 52.9177e-10       # Bohr radius of hydrogen     cm

# solar units
# value                        name                          unit           reference
#====================================================================================
Msol_sol  = 1.                 # solar mass                  Msol           Harmanec & Prsa 2011
GG_sol    = 3.944620808727e-07 # gravitional constant        Rsol3/Msol/s2  Harmanec & Prsa 2011
GGMsol_sol= 3.944620719386e-27 # grav. constant x solar mass Rsol3/s2       Harmanec & Prsa 2011

# other stuff
Mabs_sol = 4.75                # solar bolometric abs mag    mag            Harmanec & Prsa 2011
Mapp_sol = -26.74              # solar visual apparent mag   mag
numax_sol = 3120.,5.,'muHz'       # solar frequency of maximum amplitude
Deltanu0_sol = 134.88,0.04,'muHz' # solar large separation (Kallinger, 2010)
dnu01_sol = 6.14,0.10,'muHz'      # solar small separation (Kallinger, 2010)
dnu02_sol = 9.00,0.06,'muHz'      # solar small separation (Kallinger, 2010)


# program specific
#-- CLES evolutionary code
GG_cles      = 6.6742e-11      # SI
Msol_cles    = 1.9884e30       # SI
#-- EGGLETON evolutionary code
GG_eggleton  = 6.6672e-11      # SI
Msol_eggleton= 1.9891e30       # SI
Rsol_eggleton= 6.9598e8        # SI
#-- MESA evolutionary code
GG_mesa      = 6.67428e-11     # SI
Msol_mesa    = 1.9892e30       # SI
Rsol_mesa    = 6.9598e8        # SI
Lsol_mesa    = 3.8418e26       # SI
Mearth_mesa  = 5.9764e24       # SI
Rearth_mesa  = 6.37e6          # SI
au_mesa      = 1.495978921e11  # SI
#-- ESTA project
Rsol_esta = 6.9599e8           # SI
Lsol_esta = 3.846e26           # SI
GGMsol_esta = 1.32712438e20    # SI
GG_esta   = 6.6716823e-11      # SI
Msol_esta = 1.98919e8          # SI
Tsol_esta = 5777.54            # SI

# Atomic masses (http://pntpm3.ulb.ac.be/Nacre/introduction.html)
# Element   Atomic mass (amu)
#================================================================
n   =  1.0086649242  
H1  =  1.0078250321  
H2  =  2.0141017781  
H3  =  3.0160492681  
He3 =  3.0160293091  
He4 =  4.0026032501  
Li6 =  6.015122306509
Li7 =  7.016004073506
Be7 =  7.016929269506
Be8 =  8.00530509538 
B8  =  8.0246067271188 
Be9 =  9.012182248405  
B9  =  9.01332889191047
B10 = 10.012937097349 
B11 = 11.009305514405 
C12 = 12.000000000 
C13 = 13.0033548385
N13 = 13.005738584289
C14 = 14.0032419914 
N14 = 14.0030740072 
O14 = 14.00859528780
N15 = 15.00010897312
O15 = 15.003065460540
O16 = 15.9949146223 
O17 = 16.999131501223   
F17 = 17.002095238266   
O18 = 17.999160413851   
F18 = 18.000937665636   
F19 = 18.99840320575    
Ne19= 19.001879726612
Ne20= 19.9924401763 
Ne21= 20.99384674443   
Na21= 20.997655100751  
Ne22= 21.991385500252  
Na22= 21.994436633482  
Na23= 22.989769657262  
Mg23= 22.9941248281353 
Mg24= 23.985041874258  
Mg25= 24.985837000261  
Al25= 24.990428531760
Mg26= 25.982592999264  
Al26= 25.986891675268  
Al27= 26.981538407238  
Si27= 26.986704124264  
Si28= 27.976926494216  
Si29= 28.976494680219  
P29 = 28.981801337807   
Si30= 29.973770179221  
P30 = 29.978313768482   
P31 = 30.973761487269   
