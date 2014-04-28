r"""
Limb darkening and intensity fitting and querying

.. contents::
    :local:
    :depth: 2

.. _wonder_how_atmospheres:

.. _limbdark-atmospheres:

Section 0. Atmospheres in Phoebe 2
===================================

With the term 'atmosphere', we mean the layer of the star that is observed. In
the framework of our model, it defines how much flux is observed, given a certain
aspect angle and passband. This is set via three parameters:

    - ``atm``: source of normal emergent specific intensities. That is, if we
      look right at the center of the object (it's tangent surface being in the
      plane of the sky), how many erg/s/cm2/AA/sr do we measure?
    - ``ld_func``: determines the aspect angle dependency. There can be a linear
      drop-off in :math:`\mu=\cos\theta` with :math:`\theta` the angle between the
      line of sight and the surface normal, or it can be more complex.
    - ``ld_coeffs``: It quanitifies the ``ld_func``. It tells you how much
      erg/s/cm2/AA/sr is lost when you look at the atmosphere from a different angle.
      Typically, you choose an ``ld_func`` but let the parameters vary according
      to the properties of the star (effective temperature, surface gravity etc).

These three parameters are strongly linked: the combination of those should
always reproduce the observed flux/luminosity of the Sun, given the solar
effective temperature, radius, mass and composition. Ideally, ``atm`` and
``ld_coeffs`` are fitted to a grid of computed specific intensities, assuming
an ``ld_func`` (:py:func:`compute_grid_ld_coeffs`).

Example combination of parameter values:

    - :envvar:`atm=kurucz`
    - :envvar:`ld_func=claret`
    - :envvar:`ld_coeffs=kurucz`
    
It is possible to sever the link between the parameters, and take a different
``ld_func`` or fixed ``ld_coeffs`` for any ``atm``. Still, we want to
reproduce the solar luminosity, which means that integrated over all solid angles,
the flux should stay the same. Therefore:

    1. We compute the passband luminosity :math:`L_\mathrm{pb}` given
       ``atm`` and the original ``ld_func`` and ``ld_coeffs`` used to determine
       the ``atm`` values:
       
       .. math::
    
            L_\mathrm{pb} = 2\pi\int_S \int_{0}^{\frac{\pi}{2}}I_\mathrm{pb}(0)I_\mathrm{pb}^N(\cos\theta)\cos\theta\sin\theta d\theta dA \quad\mathrm{[erg/s]}
       
       where, for example for a linear limb darkening law:
       
       .. math::
       
            I_\mathrm{pb}^N(\mu) = \frac{I_\mathrm{pb}(\mu)}{I_\mathrm{pb}(0)}  = 1 - \epsilon + \epsilon\mu
       
       with:
       
       .. math::
       
            \left\{\begin{array}{ll}
            I_\mathrm{pb}(0) & \quad \mathrm{from }\ \mathtt{atm} \\
            I_\mathrm{pb}^N(\cos\theta) & \quad \mathrm{from }\ \mathtt{ld\_func}\\
            \epsilon & \quad \mathrm{from }\ \mathtt{ld\_coeffs}
            \end{array}\right.
       
       Note that, because of the dependence of the local intensity to local surface properties, actually
       
       .. math::
       
            I_\mathrm{pb} = I_\mathrm{pb}\left(T_\mathrm{eff}(\theta,\phi), g(\theta,\phi), z(\theta,\phi), v_\mathrm{rad}(\theta,\phi), \ldots\right).
       
       
    2. We compute the luminosity given the ``atm`` values and the newly given
       ``ld_func`` and/or ``ld_coeffs``.
       
       .. math::
    
            L_\mathrm{pb}\prime = 2\pi\int_S \int_{0}^{\frac{\pi}{2}}I_\mathrm{pb}(0)I_\mathrm{pb}^N\prime(\cos\theta)\cos\theta\sin\theta d\theta dA \quad\mathrm{[erg/s]}
       
       
    3. We rescale the ``atm`` values to reproduce the luminosity computed in (1).
    
        .. math::
        
            I_\mathrm{pb}(0)^\mathrm{new} = \frac{L_\mathrm{pb}}{L_\mathrm{pb}\prime}I_\mathrm{pb}(0)

.. _limbdark-atmospheres-fileformats:

Section 1. File formats
=======================

There are four types of grids defined in the FITS format:

    1. grids with **specific intensities**, holding in each extension a FITS
       table with a full SED at discrete limb angles, for a given set of
       parameter values. Grids supplied with Phoebe should live in the directory
       ``atmospheres/tables/spec_intens/``
    2. grids with **spectral energy distributions**, holding in each extension
       a FITS table with a full disk-integrated SED, for a given set of
       parameter values. Grids supplied with Phoebe should live in the directory
       ``atmospheres/tables/sed/``. These grids can be used as backup for when
       the specific intensities are not available. Then the disk-integrated
       intensities can still be used, but the limb-darkening coefficients need
       to be supplied manually or using another grid.
    3. grids with **LD coefficients**, holding in each extension a FITS table a
       grid of coefficients as a function of a given set of parameter values.
       These grids should live in the directory
       ``atmospheres/tables/ld_coeffs/``
    
    4. grids with **high resolution spectra**, holding ... what?

Note that SED files are actually a less-informative version of the spectral grids.
Consequently, any spectral grid file can act as an SED file.
   
Section 1.1 FITS of Specific intensities
----------------------------------------

Each FITS table has the following header keys (KEY (type)):

    - C{TEFF} (float): effective temperature (K) of the specific intensities in
      the extension
    - C{LOGG} (float): log of surface gravity (cgs)
    - C{ABUN} (float): log of metallicity wrt to solar (optional)
    - C{VMIC} (float): microturbulent velocity (km/s) (optional)
    - C{...} (float):  any other float that is relevant to this atmosphere model
      and which can be used to interpolate in
      
In principle, each of these are optional, it depends on what you eventually
want to interpolate in (or what Phoebe can handle).

Each FITS table has a data array with the following columns:

    - C{WAVELENGTH}: wavelength array
    - C{'1.000000'}: string representation of the first mu angle (mu=1)
    - C{'G.GGGGGG'}: string representation of the second mu angle (mu<1)
    - ... : string representation of other mu angles
    - C{'0.000000'}: string representation of the last mu angle (mu=0)
    
The units are ``erg/s/cm2/AA/sr``. When disk-integrated and multiplied with
the dilution factor :math:`(R_*/d)^2` (with :math:`R_*` the stellar radius
and :math:`d` the distance to the target), they transform to observed fluxes.

**Example**::

    XTENSION= 'BINTABLE'           / binary table extension                         
    BITPIX  =                    8 / array data type                                                                                                                                 
    NAXIS   =                    2 / number of array dimensions                                                                                                                      
    NAXIS1  =                   76 / length of dimension 1                                                                                                                           
    NAXIS2  =                 1221 / length of dimension 2                                                                                                                           
    PCOUNT  =                    0 / number of group parameters                                                                                                                      
    GCOUNT  =                    1 / number of groups                                                                                                                                
    TFIELDS =                   19 / number of table fields                                                                                                                          
    TTYPE1  = 'wavelength'                                                          
    TFORM1  = 'E       '                                                            
    TTYPE2  = '1.000000'                                                            
    TFORM2  = 'E       '                                                            
    TTYPE3  = '0.900000'                                                            
    TFORM3  = 'E       '                                                            
    TTYPE4  = '0.800000'                                                            
    TFORM4  = 'E       '                                                            
    TTYPE5  = '0.700000'                                                            
    TFORM5  = 'E       '                                                            
    TTYPE6  = '0.600000'                                                            
    TFORM6  = 'E       '                                                            
    TTYPE7  = '0.500000'                                                            
    TFORM7  = 'E       '                                                            
    TTYPE8  = '0.400000'                                                            
    TFORM8  = 'E       '                                                            
    TTYPE9  = '0.300000'                                                            
    TFORM9  = 'E       '                                                            
    TTYPE10 = '0.250000'                                                            
    TFORM10 = 'E       '                                                            
    TTYPE11 = '0.200000'                                                            
    TFORM11 = 'E       '                                                            
    TTYPE12 = '0.150000'                                                            
    TFORM12 = 'E       '                                                            
    TTYPE13 = '0.125000'                                                            
    TFORM13 = 'E       '                                                            
    TTYPE14 = '0.100000'                                                            
    TFORM14 = 'E       '                                                            
    TTYPE15 = '0.075000'                                                            
    TFORM15 = 'E       '                                                            
    TTYPE16 = '0.050000'                                                            
    TFORM16 = 'E       '                                                            
    TTYPE17 = '0.025000'                                                            
    TFORM17 = 'E       '                                                            
    TTYPE18 = '0.010000'                                                            
    TFORM18 = 'E       '                                                            
    TTYPE19 = '0.000000'                                                            
    TFORM19 = 'E       '                                                            
    EXTNAME = 'T03500_logg0.00'    / name of the extension                          
    TEFF    =               3500.0 / Effective temperature (K)                      
    LOGG    =                  0.0 / Log(g)                                         
    ABUN    =                    0 / log Metallicity wrt solar                      
    VMIC    =                    2 / Mircoturbulent velocity (km/s)
    
Section 1.2 FITS of SEDs
----------------------------------------

Each FITS table has the following header keys (KEY (type)):

    - C{TEFF} (float): effective temperature (K) of the specific intensities in
      the extension
    - C{LOGG} (float): log of surface gravity (cgs)
    - C{ABUN} (float): log of metallicity wrt to solar (optional)
    - C{VMIC} (float): microturbulent velocity (km/s) (optional)
    - C{...} (float):  any other float that is relevant to this atmosphere model
      and which can be used to interpolate in

In principle, each of these are optional, it depends on what you eventually
want to interpolate in (or what Phoebe can handle).

Each FITS table has a data array with the following columns:

    - C{WAVELENGTH}: wavelength array
    - C{FLUX}: disk-integrated intensities
    
The units are ``erg/s/cm2/AA/sr``. When multiplied with the dilution factor
:math:``(R_*/d)^2`` (with :math:`R_*` the stellar radius and :math:`d` the
distance to the target), they transform to observed fluxes.

Section 1.3 FITS of LD coefficients
-----------------------------------

These grids can be created from the grids of specific intensities (see
L{compute_grid_ld_coeffs}). Each FITS table has the following header keys
(KEY (type)):

    - C{EXTNAME} (str): photometric passband

Each FITS table has a data array with the following columns:

    - C{teff}: array of effective temperature for each grid point
    - C{logg}: array of log(surface gravities) for each grid point
    - C{...}: array of whatever parameter for each grid point
    - C{res}: residuals from LD function fit
    - C{dflux}: relative difference between actual intensities and intensities
      from the fit
    - C{a1}: first limb darkening coefficient
    - ...
    - C{an}: n-th limb darkening coefficient
    - C{Imu1}: flux of mu=1 SED.
    - C{alpha_b}: boosting factor
    - C{Idisk}: disk integrated intensity
    
If you want the grid-computer to be able to automatically add extra passbands
with the same setup as was used for the existing passbands, all the required
information needs to be added in the primary header. The keywords are the same
as for the grid-computer function, but preceded with the prefix ``C__``.

**Example of primary header**::

    SIMPLE  =                    T / conforms to FITS standard                      
    BITPIX  =                   64 / array data type                                
    NAXIS   =                    2 / number of array dimensions                     
    NAXIS1  =                    1                                                  
    NAXIS2  =                    1                                                  
    EXTEND  =                    T                                                  
    FILETYPE= 'LDCOEFFS'           / type of file                                   
    HIERARCH C__FITMETHOD = 'equidist_r_leastsq' / method used to fit the LD coeffic
    HIERARCH C__LD_FUNC = 'claret  ' / fitted limb darkening function               
    HIERARCH C__LIMBZERO = T / point at the disk limb included or not               
    HIERARCH C__AI_teff = 'teff    ' / iterated atmosphere parameter                
    HIERARCH C__AI_logg = 'logg    ' / iterated atmosphere parameter                
    HIERARCH C__AI_abun = 'abun    ' / iterated atmosphere parameter                
    HIERARCH C__ATMFILE000 = 'kurucz_mu_im01k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE001 = 'kurucz_mu_im02k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE002 = 'kurucz_mu_im03k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE003 = 'kurucz_mu_im05k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE004 = 'kurucz_mu_im10k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE005 = 'kurucz_mu_im15k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE006 = 'kurucz_mu_im20k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE007 = 'kurucz_mu_im25k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE008 = 'kurucz_mu_im30k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE009 = 'kurucz_mu_im35k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE010 = 'kurucz_mu_im45k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE011 = 'kurucz_mu_im50k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE012 = 'kurucz_mu_ip00k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE013 = 'kurucz_mu_ip02k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE014 = 'kurucz_mu_ip03k2.fits' / included specific intensity f
    HIERARCH C__ATMFILE015 = 'kurucz_mu_ip05k2.fits' / included specific intensity f
    VMIC    =                    2 / key from first atm_file                        
    ABUN    =                 -0.1 / key from first atm_file                        
    REF     = 'KURUCZ  '           / key from first atm_file

**Example of extension header**::

    XTENSION= 'BINTABLE'           / binary table extension                         
    BITPIX  =                    8 / array data type                                
    NAXIS   =                    2 / number of array dimensions                     
    NAXIS1  =                   88 / length of dimension 1                          
    NAXIS2  =                 5888 / length of dimension 2                          
    PCOUNT  =                    0 / number of group parameters                     
    GCOUNT  =                    1 / number of groups                               
    TFIELDS =                   12 / number of table fields                         
    TTYPE1  = 'teff    '                                                            
    TFORM1  = 'D       '                                                            
    TUNIT1  = 'NA      '                                                            
    TTYPE2  = 'logg    '                                                            
    TFORM2  = 'D       '                                                            
    TUNIT2  = 'NA      '                                                            
    TTYPE3  = 'abun    '                                                            
    TFORM3  = 'D       '                                                            
    TUNIT3  = 'NA      '                                                            
    TTYPE4  = 'alpha_b '                                                            
    TFORM4  = 'D       '                                                            
    TUNIT4  = 'NA      '                                                            
    TTYPE5  = 'res     '                                                            
    TFORM5  = 'D       '                                                            
    TUNIT5  = 'NA      '                                                            
    TTYPE6  = 'dflux   '                                                            
    TFORM6  = 'D       '                                                            
    TUNIT6  = 'NA      '                                                            
    TTYPE7  = 'a1      '                                                            
    TFORM7  = 'D       '                                                            
    TUNIT7  = 'NA      '                                                            
    TTYPE8  = 'a2      '                                                            
    TFORM8  = 'D       '                                                            
    TUNIT8  = 'NA      '                                                            
    TTYPE9  = 'a3      '                                                            
    TFORM9  = 'D       '                                                            
    TUNIT9  = 'NA      '                                                            
    TTYPE10 = 'a4      '                                                            
    TFORM10 = 'D       '                                                            
    TUNIT10 = 'NA      '                                                            
    TTYPE11 = 'Imu1    '                                                            
    TFORM11 = 'D       '                                                            
    TUNIT11 = 'NA      '
    TTYPE12 = 'Idisk   '                                                            
    TFORM12 = 'D       '                                                            
    TUNIT12 = 'NA      '                                                            
    EXTNAME = 'JOHNSON.V'


Section 2. Creating LD grids
============================

There is a lot of flexibility in creating grids with limb darkening grids,
to make sure that the values you are interested in can be interpolated if
necessary, or otherwise be fixed. Therefore, creating these kind of grids
can be quite complicated. A detailed explanation can be found in the description
of :py:func:`compute_grid_ld_coeffs`.

Section 3. Querying specific intensities
========================================

To query the tables of specific intensities from a file C{atm} for a certain
effective temperature and surface gravity, use :py:func:`get_specific_intensities`:

>>> atm = 'tables/spec_intens/kurucz_mu_ip00k0.fits'
>>> mu, wavelengths, table = get_specific_intensities(atm,
...                                                   dict(teff=4000, logg=4.0))

In the above it is also possible to specify reddening parameters or a systemic
velocity. You can immediately integrate the table in a list of passbands, (e.g.
in the C{JOHNSON.V} and C{2MASS.J} band) via :py:func:`get_limbdarkening`:

>>> mu, intensities = get_limbdarkening(atm, dict(teff=4000, logg=4.0),
...                                    passbands=('JOHNSON.V', '2MASS.J'))

So :py:func:`get_limbdarkening` is the passband-integrated version of 
:py:func:`get_specific_intensities`.

**Example usage**

::

>>> atm = 'tables/spec_intens/kurucz_mu_ip00k0.fits'
>>> atm_pars = dict(teff=4000, logg=4.0)
>>> passbands = ('JOHNSON.V', '2MASS.J')
>>> mu, intensities = get_limbdarkening(atm, atm_pars, passbands=passbands)

>>> p = plt.plot(mu, intensities[:,0], 'ko-', label=passbands[0])
>>> p = plt.plot(mu, intensities[:,1], 'ro-', label=passbands[1])

.. image:: images/atmospheres_limbdark_query_spec_intens.png
   :height: 266px
   :align: center

Section 4. Fitting LD coefficients
==================================

The next step after querying the specific intensities integrated over certain
passbands, is to fit a limbdarkening law through them. Let us continue with
the last example from the previous section. The function C{fit_law} only
allows to fit one particular passband, and we better normalise the intensity
profile with the value in the disk-center:

>>> coeffs, res, dflux = fit_law(mu, intensities[:,0]/intensities[0,0],
...                            law='claret', fitmethod='equidist_r_leastsq')

The latter outputs the LD coeffs (C{coeffs}) and some fit statistics, such
as the residuals between the true and fitted fluxes, and the relative
differences.


Section 5. Querying precomputed LD coefficients
===============================================

Once an LD grid is precomputed, you can interpolate the limb darkening
coefficients for any set of values within the grid:

>>> atm = 'tables/ld_coeffs/kurucz_p00_claret_equidist_r_leastsq_teff_logg.fits'
>>> coeffs = interp_ld_coeffs(atm, 'JOHNSON.V',
...                           atm_kwargs=dict(teff=4000, logg=4.0))

The output C{coeffs} contains the limb darkening coefficients (first N-1 values)
and the center flux enabling an absolute scaling.

If you don't know or care where a certain atmosphere file is located, you could
try specifying some the properties you want the table to have, and try to let
the library choose the atmosphere table for you:

>>> atm = choose_ld_coeffs_table('kurucz', ld_func='claret',
...                   atm_kwargs=dict(teff=4000, logg=4.0, abun=0))
>>> coeffs = interp_ld_coeffs(atm, 'JOHNSON.V',
...                           atm_kwargs=dict(teff=5000, logg=4.0, abun=0))


Section 6. Handling files and directories for atmosphere files
==============================================================

Did you ever want to know where the specific intensity files and limb darkening
coefficient files are stored on your system? Then call::

    paths = limbdark.get_paths()
    


Section 7. Phoebe parameters atm, ld_coeffs and ld_func
=========================================================

.. image:: images/atmospheres_limbdark_atm_pars.png
   :height: 500px
   :align: center

The parameters governing the local intensities and their aspect-angle dependency
(aka limb darkening) are the following three (see figure):

    1. ``atm``: determines what the local **normal** intensity is, given
       local parameters such as :math:`T_\mathrm{eff}`, :math:`\log g`, etc.
       This is equivalent to the intensity in the center of the disk, or at
       :math:`\mu=1`.
    2. ``ld_func``: determines the functional form of the aspect angle dependency.
       Common functional forms are the linear approximation, or Claret's nonlinear
       approximation.
    3. ``ld_coeffs``: fixes the parameters of the functional form given by ``ld_func``.
       If this parameter is a list of coefficients, then the coefficients will
       not be dependent on local parameters :math:`T_\mathrm{eff}`, :math:`\log g`, etc.,
       but the intensity will still be set by ``atm``. If this parameter is a
       grid name, then the coefficients will be taken according to the local
       parameters.
    
In the ideal case, ``atm`` and ``ld_coeffs`` take the same value (and is a grid name).   
Then, everything is consistent. However, sometimes you want to deviate from the
ideal case, e.g. to compare with other codes. Then **you need to be careful!**.
First note that the local normal emergent intensity (units: W/m3/sr) is not the
same as the flux (units: W/m3). The flux emerging from a local surface element
is computed by integrating over all solid angles facing outwards. The passband
luminosity is then finally obtained by integrating over all surface elements (W/m).
If the flux was computed bolometrically, i.e. integrated over the full wavelength
range, then we get fluxes in W/m2 luminosities in W.

**If you choose to set custom ld_coeffs**, then the following is done:

    1. the normal emergent intensity is taken from ``atm``, and the flux is
       computed using the ``ld_func`` with which the ``atm`` table was constructed.
    2. The normal emergent intensity is scaled to conserve flux with respect to
       the custom LD coeffs. (e.g. green solid line versus red solid line in
       above figure)

Sometimes, you cannot use any precomputed grid simply because the grid does not
cover your set of parameters (e.g. your star is too hot or too cold, or too evolved).
A last resort is always to set ``atm=blackbody`` (precomputed but a wide range of
temperatures, and not dependent on surface gravity) or ``atm=true_blackbody`` (covers
everything, but is computed on-the-fly and is thus very slow). Then you are
responsible for setting the ``ld_coeffs`` also to some specific set of coefficients.
In principle, you can also set ``atm=blackbody`` and ``ld_coeffs=kurucz``, but
that would be weird: if you have a reason not to trust your normal emergent intensity,
you shouldn't trust your limb darkening coefficients either.

"""
# Required system packages
import os
import itertools
import logging
import shutil
import subprocess
import urllib
import tarfile
import sys

# Third party modules
import numpy as np
from scipy.optimize import leastsq,fmin
from scipy.interpolate import splrep,splev

# Optional packages
try:
    import pyfits
except ImportError:
    print(("Soft warning: pyfits could not be found on your system, you can "
           "only use black body atmospheres and Gaussian synthetic spectral "
           "profiles"))
try:
    import pylab as pl
except ImportError:
    print(("Soft warning: matplotlib could not be found on your system, 2D "
           "plotting is disabled, as well as IFM functionality"))

# Phoebe modules
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.utils import decorators
from phoebe.utils import coordinates
from phoebe.utils import config
from phoebe.utils import utils
from phoebe.io import fits
from phoebe.io import ascii
from phoebe.algorithms import interp_nDgrid
from phoebe.atmospheres import sed
from phoebe.atmospheres import reddening
from phoebe.atmospheres import tools
from phoebe.atmospheres import passbands as pbmod

logger = logging.getLogger('ATMO.LD')
logger.addHandler(logging.NullHandler())

basedir = os.path.dirname(os.path.abspath(__file__))
basedir_spec_intens = os.path.join(basedir, 'tables', 'spec_intens')
basedir_ld_coeffs = os.path.join(basedir, 'tables', 'ld_coeffs')

Rsol_d_to_kms = constants.Rsol/(24*3600.)/1000.

_prsa_ld = np.linspace(0, 1, 32)
_prsa_res = _prsa_ld[0] - _prsa_ld[1]

#{ LD laws

def ld_claret(mu, coeffs):
    r"""
    Claret's limb darkening law.
    
    .. math::
    
        \frac{I(\mu)}{I(0)} = 1 - a_0(1-\mu^{0.5})-a_1(1-\mu)-a_2(1-\mu^{1.5})-a_3(1-\mu^2)
    
    .. image:: images/atmospheres_limbdark_law_claret.png 
       :height: 266px
       :align: center
    
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    Imu = 1 - coeffs[0]*(1-mu**0.5) - coeffs[1]*(1-mu) -\
              coeffs[2]*(1-mu**1.5) - coeffs[3]*(1-mu**2.)    
    
    return Imu


def ld_hillen(mu, coeffs):
    r"""
    Empirical limb darkening law devised by M. Hillen.
    
    .. math::
    
        \frac{I(\mu)}{I(0)} = \left( C + (u+v|\mu| + w\mu^2) \arctan{\left(f(\mu-d)\right)}\right) \left(\frac{s}{\mu^4+s}+1\right)
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    # skew parameter cannot be smaller than zero
    skew = coeffs[5]
    min_skew, max_skew = 0.1, 1e9
    # skew bounded
    skew_bounded = np.sin( (skew - min_skew)/(max_skew-min_skew))
    skew = skew_bounded*(max_skew-min_skew) + min_skew
    Imu = (coeffs[0] + (coeffs[1] + coeffs[2] * np.abs(mu) + coeffs[6] * mu**2) * np.arctan(coeffs[4]*(mu-coeffs[3])))*(-skew*1e-8/(mu**4+skew*1e-8)+1.)
    
    return Imu



def ld_prsa(mu, coeffs):
    r"""
    Empirical limb darkening law proposed by A. Prsa.
    """
    # 32 samples uniformly in cos(theta) = mu
    mu = 1-mu
    indices1 = np.searchsorted(_prsa_ld, mu)
    indices2 = indices1 - 1
    dy_dx = (coeffs[:,indices2] - coeffs[:,indices1]) / (_prsa_ld[indices2] - _prsa_ld[indices1])
    Imu = dy_dx * (mu - _prsa_ld[indices1]) + coeffs[:,indices1]
    
    return Imu[0]
    


def ld_linear(mu, coeffs):
    r"""
    Linear or linear cosine law
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1 - \epsilon + \epsilon\mu
        
    +------------------------------------------------------+-------------------------------------------------------+
    | .. image:: images/atmospheres_limbdark_linearld.png  | .. image:: images/atmospheres_limbdark_law_linear.png |
    |    :height: 266px                                    |    :height: 266px                                     |
    |    :align: center                                    |    :align: center                                     |
    +------------------------------------------------------+-------------------------------------------------------+
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return 1 - coeffs[0]*(1-mu)
    
    
def ld_nonlinear(mu, coeffs):
    r"""
    Nonlinear or logarithmic law
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1 - c_1 (1-\mu) - c_2\mu\ln\mu
    
    .. image:: images/atmospheres_limbdark_law_nonlinear.png 
       :height: 266px
       :align: center
    
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    if not np.isscalar(mu):
        mu[mu == 0] = 1e-16
    elif mu == 0:
        mu = 1e-16
    return 1 - coeffs[0]*(1-mu) - coeffs[1]*mu*np.log(mu)


def ld_logarithmic(mu, coeffs):
    r"""
    Nonlinear or logarithmic law
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1 - c_1 (1-\mu) - c_2\mu\ln\mu
    
    .. image:: images/atmospheres_limbdark_law_nonlinear.png 
       :height: 266px
       :align: center
    
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return ld_nonlinear(mu, coeffs)


def ld_quadratic(mu, coeffs):
    r"""
    Quadratic law
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1 - c_1 (1-\mu) - c_2(1-\mu)^2

    .. image:: images/atmospheres_limbdark_law_quadratic.png 
       :height: 266px
       :align: center
    
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return 1 - coeffs[0]*(1-mu) - coeffs[1]*(1-mu)**2.0


def ld_square_root(mu, coeffs):
    r"""
    Square root law.
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1 - c_1 (1-\mu) - c_2(1-\sqrt{\mu})
        
    From [Diaz-Cordoves1992]_.
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return 1 - coeffs[0]*(1-mu) - coeffs[1]*(1-np.sqrt(mu))



def ld_uniform(mu, coeffs):
    r"""
    Uniform law.
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1

    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return 1.0


def ld_power(mu, coeffs):
    r"""
    Power law

    .. math::
    
        \frac{I(\mu)}{I(0)}  = \mu^\alpha

    .. image:: images/atmospheres_limbdark_law_power.png 
       :height: 266px
       :align: center
    
    
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return mu**coeffs[0]


def ld_exponential(mu, coeffs):
    r"""
    Exponential law
    
    .. math::
    
        \frac{I(\mu)}{I(0)}  = 1 - c_1(1-\mu) - c_2 \frac{1}{1-e^\mu}
        
    @param mu: limb angles mu=cos(theta)
    @type mu: numpy array
    @param coeffs: limb darkening coefficients
    @type coeffs: list
    @return: normalised intensity at limb angles
    @rtype: array
    """
    return 1 - coeffs[0]*(1-mu) - h / (1-np.exp(mu))


def _r(mu):
    r"""
    Convert mu to r coordinates
    
    .. math::
    
        r = \sqrt(1-\mu^2)
    
    @param mu: angle coordinates mu=cos(theta) with theta the limb angle
    @type mu: array
    @return: disk coordinates r
    @rtype: array
    """
    return np.sqrt(1.-mu**2.)

    
def _mu(r_):
    r"""
    Convert r to mu coordinates
    
    .. math::
    
        \mu = \sqrt(1-r^2)

    
    @param r_: disk coordinates r
    @type r_: array
    @return: angle coordinates mu=cos(theta) with theta the limb angle
    @rtype: array
    """
    return np.sqrt(1.-r_**2.)



def disk_linear(coeffs):
    r"""
    Disk integration.
    
    From Kallrath & Milone.
    
    .. math::
        
        \int_{2\pi} (1 - \epsilon + \epsilon\mu d\Omega = \pi (1-\frac{\epsilon}{3})
        
    """
    return np.pi*(1 - coeffs[0]/3.)

def disk_uniform(coeffs):
    """
    Disk integration with no limbdarkening (uniform).
    
    .. math::
    
        \int_{2\pi} d\Omega = \pi
        
    """
    return np.pi

def disk_nonlinear(coeffs):
    return np.pi * (1 - coeffs[0]/3. + 2./9.*coeffs[1])

def disk_logarithmic(coeffs):
    return disk_nonlinear(coeffs)

def disk_quadratic(coeffs):
    p = 2.0
    return np.pi * (1 - coeffs[0]/3. - coeffs[1] / (0.5*p**2 + 1.5*p + 1))

def disk_square_root(coeffs):
    return np.pi * (1 - coeffs[0]/3.0 - coeffs[1]/5.)

def disk_claret(coeffs):
    a1x_,a2x_,a3x_,a4x_ = coeffs[:4]
    a0x_ = 1 - a1x_ - a2x_ - a3x_ - a4x_
    limb_coeffs = np.array([a0x_,a1x_,a2x_,a3x_,a4x_]).reshape((5,-1))
    int_moms = np.array([I_ls(0,1 + r/2.) for r in range(0,5,1)]).reshape((5,-1))
    I_lx = 2*np.pi*(limb_coeffs * int_moms).sum(axis=0)
    return I_lx

#}
#{ Specific intensities


def get_grid_dimensions(atm, atm_pars=('teff', 'logg', 'abun'),
                        beaming=False):
    """
    Retrieve all gridpoints from a specific intensity grid or LD coeffs file.
    
    @param atm: atmosphere grid filename
    @type atm: str
    @param atm_pars: grid axis
    @type atm_pars: tuple of strings
    @rtype: n x array
    @return: arrays with the axis values for each grid point.
    """
    
    # Derive the filename if the user gave a ParameterSet
    if not isinstance(atm, str):
        atm = choose_ld_coeffs_table(atm['atm'], ld_func=atm['ld_func'],
                               atm_kwargs={key:[] for key in atm_pars},
                               vgamma=0.0 if not beaming else 1.0)
    
    # Now open the file and go
    with pyfits.open(atm) as ff:
        # ld_coeffs file
        if atm_pars[0] in ff[1].data.dtype.names:
            dims = [ff[1].data.field(key) for key in atm_pars]
        
        # specific_intens files
        else:
            dims = [np.zeros(len(ff)-1) for par in atm_pars]
        
            # Just loop over all extensions
            for i, mod in enumerate(ff[1:]):
            
                # And retrieve the values from the atm_pars
                for j, key in enumerate(atm_pars):
                    dims[j][i] = mod.header[key]            
    
    return dims



def get_passbands(atm, atm_pars=('teff', 'logg', 'abun'),
                        beaming=False):
    """
    Retrieve all passbands from a LD coeffs file.
    
    @param atm: atmosphere grid filename
    @type atm: str
    @param atm_pars: grid axis
    @type atm_pars: tuple of strings
    @rtype: n x array
    @return: arrays with the axis values for each grid point.
    """
    
    # Derive the filename if the user gave a ParameterSet
    if not isinstance(atm, str):
        atm = choose_ld_coeffs_table(atm['atm'], ld_func=atm['ld_func'],
                               atm_kwargs={key:[] for key in atm_pars},
                               vgamma=0.0 if not beaming else 1.0)
    # Now open the file and go
    with pyfits.open(atm) as ff:
        passbands = [ext.header['EXTNAME'] for ext in ff[1:] if not ext.header['EXTNAME'][:4]=='_REF']
    print passbands
    return passbands




def iter_grid_dimensions(atm, atm_pars, other_pars, blackbody_teff_range=None):
    """
    Generate grid values.
    
    @param atm: atmosphere grid filename
    @type atm: str
    @param atm_pars: grid axis from specific intensity grid
    @type atm_pars: tuple of strings
    @param other_pars: grid axis independent from specific intensity grid
    @type other_pars: dict of arrays
    @return: values of grid points 
    @rtype: tuple
    """
    # Special treatment for blackbodies
    if atm == 'blackbody':
        
        if blackbody_teff_range is None:
            teffs = np.logspace(1, np.log10(100000.), 200)
        else:
            teffs = blackbody_teff_range
            
        for pars in itertools.product(*[teffs] + list(other_pars)):
            yield pars
    
    # Otherwise, for "normal" grids:
    else:
        with pyfits.open(atm) as ff:
            for pars in itertools.product(*[ff[1:]] + list(other_pars)):
                yield [pars[0].header[key] for key in atm_pars] + list(pars[1:])
    

def get_specific_intensities(atm, atm_kwargs={}, red_kwargs={}, vgamma=0):
    r"""
    Retrieve specific intensities of a model atmosphere at different angles.
    
    Retrieval is done from the precomputed table C{atm}.
    
    C{atm_kwargs} should be a dictionary with keys and values adequate for
    the grid, e.g. for a single-metallicity Kurucz grid::
        
        atm_kwargs = dict(teff=10000, logg=4.0)
    
    Extra keywords are determined by the header keys in the C{atm} file.
    
    C{red_kwargs} should be a dictionary with keys and values adequate for the
    limb darkening law, e.g.::
    
        red_kwargs = dict(law='fitzpatrick2004', Rv=3.1, ebv=0.5)
    
    The keywords are determined by the L{reddening.redden} function.
    
    C{vgamma} is radial velocity: positive is redshift, negative is blueshift
    (m/s!). This shifts the spectrum and does a first order correction for
    beaming effects:
    
        1. First the spectrum :math:`F(\lambda)` is
           :py:func:`doppler shifted <phoebe.atmospheres.tools.doppler_shift>`
           to :math:`F^D(\lambda)`.
        2. Then the spectrum :math:`F^D(\lambda)` is beamed to get
           :math:`F^B(\lambda)`:
        
            .. math::
            
                F^B(\lambda) = F^D(\lambda) + \frac{5 v_\gamma}{c} F^D(\lambda)
        
    
    You get limb angles, wavelength and a table. The shape of the limb angles
    array is N_mu, the wavelength array shape is N_wave, and the table has
    shape (N_wave,N_mu).    
    
    @param atm: filename of the grid
    @type atm: str
    @param atm_kwargs: dict with keys specifying the atmospheric parameters
    @type atm_kwargs: dict
    @param red_kwargs: dict with keys specifying the reddening parameters
    @type red_kwargs: dict
    @param vgamma: radial velocity (for doppler shifting) (m/s)
    @type vgamma: float
    @return: mu angles, wavelengths, table (Nwave x Nmu)
    @rtype: array, array, array
    """
    # Read the atmosphere file if it is not an open file already:
    if isinstance(atm, str):
        ff = pyfits.open(atm)
    else:
        ff = atm
    
    if not atm_kwargs:
        raise ValueError("No arguments given")
    
    # Try to run over each extension, and check if the values in the header
    # correspond to the values given by the keys in C{atm_kwargs}.
    try:
        for mod in ff[1:]:
            for key in atm_kwargs:
                if mod.header[key] != atm_kwargs[key]:
                    # For sure, this is not the right extension, don't check the
                    # other keys (break search for keys)
                    break
            else:
                # This is the right extension! Don't check the other extensions
                # (break search for extensions)
                break
        else:
            # We haven't found the right extension
            raise ValueError(("Did not found pars {} in "
                              "{}").format(atm_kwargs, atm))
        
        # Treat disk-integrated intensities differently. We can recognise them
        # because they have 'wavelength' and 'flux' in the data fields.
        available_fields = [name.lower() for name in mod.data.dtype.names]
        if 'wavelength' in available_fields and 'flux' in available_fields:
            wave = mod.data.field('wavelength')
            table = np.array([mod.data.field('flux')]).T
            mu = np.array([np.nan])
        
        else:
            # Retrieve the mu angles, table and wavelength array
            whole_table = np.array(mod.data.tolist(), float)
            wave = whole_table[:, 0] # first column, but skip first row
            table = whole_table[:, 1:] 
            mu = np.array(mod.columns.names[1:], float)
        
    except KeyError:
        raise KeyError("Specific intensity not available in atmosphere file")
    
    # Only close the file if we really have to.
    if isinstance(atm, str):
        ff.close()
    
    # Velocity shift for Doppler beaming if necessary
    if vgamma is not None and vgamma != 0:
        cc = constants.cc / 1000. #speed of light in km/s
        for i in range(len(mu)):
            flux_shift = tools.doppler_shift(wave, -vgamma, flux=table[:, i])
            table[:, i] = flux_shift + 5.*vgamma / cc * flux_shift
    
    # Redden if necessary
    if red_kwargs:
        for i in range(len(mu)):
            table[:, i] = reddening.redden(table[:, i], wave=wave,
                                           rtype='flux', **red_kwargs)
    
    # That's it!
    return mu, wave, table
    


def get_limbdarkening(atm, atm_kwargs={}, red_kwargs={}, vgamma=0,\
              passbands=('JOHNSON.V',), normalised=False):
    """
    Retrieve a limb darkening law for a specific star and specific bandpass.
    
    Possibility to include reddening via EB-V parameter. If not given, 
    reddening will not be performed...
    
    You choose your own reddening function.
    
    See e.g. [Heyrovsky2007]_
    
    If you specify one angle (:math:`\mu` in radians), it will take the closest
    match from the grid.
    
    :math:`\mu = \cos(\\theta)` where :math:`\\theta` is the angle between the
    surface and the line of sight. :math:`\mu=1` means :math:`\\theta=0` means
    center of the star.
    
    Example usage:
    
    >>> teff, logg = 5000, 4.5
    >>> mu,intensities = get_limbdarkening(teff=teff, logg=logg,
    ...          photbands=['JOHNSON.V'])

    @return: :math:`\mu` angles, intensities
    @rtype: array,array
    """
    # Retrieve model atmosphere for a given teff and logg
    mus, wave, table = get_specific_intensities(atm, atm_kwargs=atm_kwargs,
                                                red_kwargs=red_kwargs,
                                                vgamma=vgamma)
    
    # Put mus (limb angles) in right order
    sa = np.argsort(mus)[::-1]
    mus = mus[sa]
    table = table[:, sa]
    
    # Compute intensity over the stellar disk, and normalise
    intensities = np.zeros((len(mus), len(passbands)))
    for i in range(len(mus)-1):
        intensities[i] = sed.synthetic_flux(wave, table[:, i], passbands)
    if normalised:
        intensities/= intensities.max(axis=0)
    return mus, intensities


def fit_law(mu, Imu, law='claret', fitmethod='equidist_r_leastsq',
            limb_zero=True, oversampling=5000, debug_plot=False):
    """
    Fit an LD law to a sampled set of limb angles/intensities.
    
    Most likely options for ``law`` are:
    
    - ``claret``
    - ``linear``
    - ``quadratic``
    
    Possible options for ``fitmethod`` are:
    
    - ``leastsq``: Levenberg-Marquardt fit on the coordinates as available in the atmosphere grids
    - ``fmin``: Nelder-Mead fit on the coordinates as available in the atmosphere grids.
    - ``equidist_r_leastsq``: LM-fit in r-coordinate resampled equidistantly
    - ``equidist_mu_leastsq``: LM-fit in :math:`\mu`-coordinate resampled equidistantly
    - ``equidist_r_fmin``: NM-fit in r-coordinate resampled equidistantly
    - ``equidist_mu_fmin``: NM-fit in :math:`\mu`-coordinate resampled equidistantly
    
    If ``limb_zero=False``, the last point will not be used in the fit. This
    is most likely the :math:`\mu=0` point, or the point really at the edge.
    
    The :envvar:`oversampling` factor determines the oversampling rate of the
    splines, which will be used to make the r or mu coordinate equidistant. For
    the Prsa law, this should better be :envvar:`oversampling=32`.
    
    In my (Pieter's) experience, C{fitmethod='equidist_r_leastsq'} seems
    appropriate for the Kurucz models, though ``limb_zero=False`` seems to
    reproduce it better.
    
    Make sure the intensities are normalised!
    
    **Example usage**
    
    >>> mu,intensities = get_limbdarkening(teff=10000, logg=4.0,
    ...                 photbands=['JOHNSON.V'], normalised=True)
    >>> p = pl.figure()
    >>> p = pl.plot(mu,intensities[:,0],'ko-')
    >>> cl,ssr,rdiff = fit_law(mu,intensities[:,0],law='claret')
    
    >>> mus = np.linspace(0,1,1000)
    >>> Imus = ld_claret(mus,cl)
    >>> p = pl.plot(mus,Imus,'r-',lw=2)
    
    @return: coefficients, sum of squared residuals,relative flux difference
     between prediction and model integrated intensity
    @rtype: array, float, float
    """
    if not limb_zero:
        mu = mu[:-1]
        Imu = Imu[:-1]
    
    def ldres_fmin(coeffs, mu, Imu, law):
        return sum((Imu - globals()['ld_{}'.format(law)](mu, coeffs))**2)
    
    def ldres_leastsq(coeffs, mu, Imu, law):
        return Imu - globals()['ld_{}'.format(law)](mu, coeffs)
    
    # Prepare array for coefficients and set the initial guess
    Ncoeffs = dict(claret=4, linear=1, nonlinear=2, logarithmic=2, quadratic=2,
                   power=1, hillen=7, prsa=oversampling)
    c0 = np.zeros(Ncoeffs[law])
    c0[0] = 0.6
    
    # Better initial guess for Hillen law
    if law == 'hillen':
        c0[0] = 0.10
        c0[1] = 0.025
        c0[2] = 0.53
        c0[3] = 0.12
        c0[4] = 70.0
        c0[5] = 0.1
        c0[6] = 0.0
        
    # Do the fitting; there's several possibilities
    
    # On the original resolution with the Levenberg-Marquardt (LM) algorithm
    if fitmethod == 'leastsq':
        (csol, ierr)  = leastsq(ldres_leastsq, c0, args=(mu, Imu, law))
        
    # On the original resolution with the Nelder-Mead simplex (NMS) method    
    elif fitmethod == 'fmin':
        csol = fmin(ldres_fmin, c0, maxiter=1000, maxfun=2000,
                    args=(mu, Imu, law), disp=0)
    
    # Equidistantly sampled in limb angle + LM
    elif fitmethod == 'equidist_mu_leastsq':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order], Imu[mu_order],s=0.0, k=2)
        mu_spl = np.linspace(mu[mu_order][0], 1, oversampling)
        Imu_spl = splev(mu_spl, tck, der=0)
        if law == 'prsa':
            return Imu_spl, mu_spl, 0.
        (csol, ierr)  = leastsq(ldres_leastsq, c0, args=(mu_spl, Imu_spl, law))
    
    # Equidistantly sampled in radius coordinate + LM
    elif fitmethod == 'equidist_r_leastsq':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order], Imu[mu_order],s=0., k=2)
        r_spl = np.linspace(mu[mu_order][0], 1, oversampling)
        mu_spl = np.sqrt(1-r_spl**2)
        Imu_spl = splev(mu_spl, tck, der=0)    
        if law == 'prsa':
            return Imu_spl, mu_spl, 0.
        (csol,ierr)  = leastsq(ldres_leastsq, c0, args=(mu_spl,Imu_spl,law))
    
    # Equidistantly sampled in limb angle coordinate + NMS
    elif fitmethod == 'equidist_mu_fmin':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order], Imu[mu_order],k=2, s=0.0)
        mu_spl = np.linspace(mu[mu_order][0], 1, oversampling)
        Imu_spl = splev(mu_spl, tck, der=0)
        if law == 'prsa':
            return Imu_spl, mu_spl, 0.
        csol = fmin(ldres_fmin, c0, maxiter=1000, maxfun=2000,
                     args=(mu_spl, Imu_spl, law), disp=0)
    
    # Equidistantly sampled in radius coordinate + NMS
    elif fitmethod == 'equidist_r_fmin':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order],Imu[mu_order],k=2, s=0.0)
        r_spl = np.linspace(mu[mu_order][0],1,oversampling)
        mu_spl = np.sqrt(1-r_spl**2)
        Imu_spl = splev(mu_spl, tck, der=0)
        if law == 'prsa':
            return Imu_spl, mu_spl, 0.
        csol  = fmin(ldres_fmin, c0, maxiter=1000, maxfun=2000,args=(mu_spl,\
                                                         Imu_spl, law), disp=0)
    
    else:
        raise ValueError("Fitmethod {} not recognised".format(fitmethod))
    
    myfit = globals()['ld_{}'.format(law)](mu, csol)
    res =  np.sum(Imu - myfit)**2
    int1, int2 = np.trapz(Imu, x=mu), np.trapz(myfit, x=mu)
    dflux = (int1 - int2) / int1
    
    # Make a debug plot if needed
    if debug_plot:
        pl.figure(int(debug_plot), figsize=(12,6))
        pl.subplot(121)
        pl.plot(mu, Imu, 'ko', label='Grid points')
        pl.plot(mu_spl, Imu_spl, 'k-', mew=2, ms=10, label='Resampled')
        pl.plot(mu, myfit, 'r-', lw=2, label='Fit')
        pl.legend(loc='best', numpoints=1, prop=dict(size='small')).get_frame().set_alpha(0.5)
        pl.xlabel(r'$\mu=\cos\theta$')
        pl.subplot(122)
        pl.plot(np.sqrt(1-mu**2), Imu, 'ko', label='Grid points')
        pl.plot(np.sqrt(1-mu_spl**2), Imu_spl, 'kx-', mew=2, ms=10, label='Resampled')
        pl.plot(np.sqrt(1-mu**2), myfit, 'r-', lw=2, label='Fit')
        pl.xlabel(r'r-coordinate')
    
    return csol, res, dflux


#}
#{ LD Passband coefficients





def choose_ld_coeffs_table(atm, atm_kwargs={}, red_kwargs={}, vgamma=0.,
                           ld_func='claret',
                           fitmethod='equidist_r_leastsq'):
    """
    Derive the filename of a precalculated LD grid from the input parameters.
    
    Possibilities:
        
        - if ``atm`` is an existing absolute path, nothing is done but to return
          the filename.
        - if ``atm`` is a relative path name and it exists in the
          ``atmospheres/tables/ld_coeffs`` directory, the absolute path name
          will be returned.
        - if ``atm='blackbody'``, then ``blackbody_uniform_none_teff.fits`` is
          returned.
        
    If none of these conditions is fulfilled a little more work is done. In this
    case, we collect the information from the limb darkening function, limb
    darkening coefficients, atmospheric parameters (abundances...) and other
    parameters (reddening...), and derive the the filename. The structure of
    such a filename is::
    
        {atm}_{prefix}{abun:02.0f}_{ld_func}_{fitmethod}_{var1}_..._{varn}.fits
    
    For example :file:`kurucz_p00_claret_equidist_r_leastsq_teff_logg.fits`.
    
    We need to do a lot more work here to derive it!
    
    @param atm: atmosphere table filename or designation
    @type atm: str
    """
    # Perhaps the user gave an absolute filename: then return it
    if os.path.isfile(atm):
        return atm
    
    # Perhaps the user gave the name of table that is pre-installed (relative
    # filename)
    elif os.path.isfile(os.path.join(basedir_ld_coeffs, atm)):
        return os.path.join(basedir_ld_coeffs, atm)
    
    # If we don't have a string, return None. Otherwise make it lower case
    try:
        atm = atm.lower()
    except AttributeError:
        return None
    
    # If the user wants tabulated blackbodies, we have a file for that.
    if atm == 'blackbody':
        basename = 'blackbody_uniform_none_teff.fits'
        return os.path.join(basedir_ld_coeffs, basename)
    
    elif atm == 'true_blackbody':
        return atm
    
    # Else we need to be a little bit more clever and derive the file with the
    # tabulated values based on abundance, LD func etc...
    else:
        
        # Build the postfix
        postfix = []
        
        if 'teff' in atm_kwargs:
            postfix.append('teff')
        if 'logg' in atm_kwargs:
            postfix.append('logg')
        if 'abun' in atm_kwargs:
            postfix.append('abun')
            
        # <-- insert reddening parameters here -->    
        # Beaming: only if the velocity is not zero   
        try:
            if vgamma != 0:
                postfix.append('vgamma')
        except ValueError:
            if not np.allclose(vgamma, 0):
                postfix.append('vgamma')
        if postfix:
            postfix = "_"+"_".join(postfix)
        else:
            postfix = ''
        
        if atm == 'jorissen':
            atm = 'jorissen_m1.0_t02_st_z+0.00_a+0.00'
            
        basename = "{}_{}_{}{}.fits".format(atm, ld_func, fitmethod, postfix)
        
        ret_val = os.path.join(basedir_ld_coeffs, basename)
        
        # Although we now figured out which atmosphere file to use, it doesn't
        # seem to be present. Check that!
        if os.path.isfile(ret_val):
            return ret_val
        else:
            raise ValueError(("Cannot interpret atm parameter {}: I think "
                              "the file that I need is {}, but it doesn't "
                              "exist. If in doubt, consult the installation "
                              "section of the documentation on how to add "
                              "atmosphere tables.".format(atm, ret_val)))
            answer = raw_input(("Cannot interpret atm parameter {}: I think "
                              "the file that I need is {}, but it doesn't "
                              "exist. If in doubt, consult the installation "
                              "section of the documentation on how to add "
                              "atmosphere tables. Should I try to\n"
                              "- download this atmosphere table [Y]\n"
                              "- download all atmospheres tables [y]\n"
                              "- abort and quit [n]?\n"
                              "[Y/y/n]: ").format(atm, ret_val))
            executable = os.sep.join(__file__.split(os.sep)[:-3])
            executable = os.path.join(executable, 'download.py')
            if answer == 'Y' or not answer:
                output = subprocess.check_output(" ".join(['python', executable,'atm']), shell=True)
                return ret_val
            elif answer == 'y':
                output = subprocess.check_output(" ".join([sys.executable, executable,'atm', os.path.basename(ret_val),os.path.basename(ret_val)]), shell=True)
                return ret_val
            else:
                raise ValueError(("Cannot interpret atm parameter {}, exiting "
                                  "upon user request").format(atm))
                    
    # Finally we're done



    
    



def interp_ld_coeffs(atm, passband, atm_kwargs={}, red_kwargs={}, vgamma=0,
                     order=1, return_header=False, cuts=None,
                     mode='constant', cval=0.0, safe=True):
    """
    Interpolate an atmosphere table.
    
    @param atm: atmosphere table absolute filename
    @type atm: string
    @param atm_kwargs: dict with keys specifying the atmospheric parameters
    @type atm_kwargs: dict
    @param red_kwargs: dict with keys specifying the reddening parameters
    @type red_kwargs: dict
    @param vgamma: radial velocity
    @type vgamma: float/array
    @param passband: photometric passband
    @type passband: str
    @param order: interpolation order
    @type order: integer
    """
    # Retrieve structured information on the grid (memoized)
    axis_values, pixelgrid, labels, header = _prepare_grid(passband, atm, cuts=cuts)

    # Prepare input: the variable "labels" contains the name of all the
    # variables (teff, logg, ebv, etc... which can be interpolated. If all z
    # values for example are equal, we do not need to interpolate in
    # metallicity, and _prepare_grid will have removed the 'z' label from that
    # list. In the following 3 lines of code, we collect only those parameters
    # which need to be interpolated in the grid. Beware that this means that the
    # z variable will simply be ignored! If you ask for z=0.1 but all values in
    # the grid are z=0.0, this function will not complain and just give you the
    # z=0.0 interpolation results!
    n_dim = 1
    
    for i, label in enumerate(labels):
        
        # if the label is an atmosphere keyword and has a length, determine its
        # length
        if label in atm_kwargs and hasattr(atm_kwargs[label], '__len__'):
            n_dim = max(n_dim, len(atm_kwargs[label]))
        
        # if the label is a reddening keyword and has a length, determine its
        # length
        elif label in red_kwargs and hasattr(red_kwargs[label], '__len__'):
            n_dim = max(n_dim, len(red_kwargs[label]))
        
        # if the label is the 'vgamma' keyword and has a length, determine its
        # length
        elif label == 'vgamma' and hasattr(vgamma, '__len__'):
            n_dim = max(n_dim, len(vgamma))
        
        #else:
        #    raise ValueError("Somethin' wrong with the atmo table: ")
    
    # Initiate the array to hold the grid dimensions and interpolated values
    values = np.zeros((len(labels), n_dim))
    
    for i, label in enumerate(labels):
        
        # Get the value from the atm_kwargs or red_kwargs
        if label in atm_kwargs:
            values[i] = atm_kwargs[label]
        elif label in red_kwargs:
            values[i] = red_kwargs[label]
        elif label == 'vgamma':
            values[i] = vgamma
        elif label == 'ebv':
            values[i] = red_kwargs['extinction'] / red_kwargs['Rv']
        else:
            raise ValueError(("Somethin' wrong with the atmo table: cannot "
                              "interpret label {} (perhaps not given in "
                              "function, perhaps not available in table").format(label))
    # Try to interpolate
    try:
        pars = interp_nDgrid.interpolate(values, axis_values, pixelgrid, order=order, mode=mode, cval=cval)
        #pars = interp_nDgrid.cinterpolate(values.T, axis_values, pixelgrid).T
        if safe and np.any(np.isnan(pars[-1])) or np.any(np.isinf(pars[-1])):
            raise IndexError
    
    # Things can go outside the grid
    except IndexError:
        msg = ", ".join(['{:.3f}<{}<{:.3f}'.format(values[i].min(), labels[i],\
                                 values[i].max()) for i in range(len(labels))])
        msg = ("Parameters outside of grid {}: {}. Consider using a different "
               "atmosphere/limbdarkening grid, or use the black body "
               "approximation.").format(atm, msg)
        #import matplotlib.pyplot as plt
        #wrong = np.isnan(np.array(pars)[0])
        #plt.plot(values[0][wrong], values[1][wrong], 'ko')
        #plt.show()
        #raise SystemExit
        raise ValueError(msg)
        #logger.error(msg)
        
        pars = np.zeros((pixelgrid.shape[-1], values.shape[1]))
    
    # The intensities were interpolated in log, but we want them in linear scale
    pars[-1] = 10**pars[-1]
    # That's it!
    if return_header:
        return pars, header
    else:
        return pars




def interp_boosting(atm, passband, atm_kwargs={}, red_kwargs={}, vgamma=0,
                     order=1, interp_all=True, return_header=False):
    """
    Interpolate an atmosphere table for boosting coefficients.
    
    @param atm: atmosphere table filename or alias
    @type atm: string
    @param atm_kwargs: dict with keys specifying the atmospheric parameters
    @type atm_kwargs: dict
    @param red_kwargs: dict with keys specifying the reddening parameters
    @type red_kwargs: dict
    @param vgamma: radial velocity
    @type vgamma: float/array
    @param passband: photometric passband
    @type passband: str
    @param order: interpolation order
    @type order: integer
    """
    # Retrieve structured information on the grid (memoized)
    nointerp_columns = ['a{:d}'.format(i) for i in range(1,10)]+['res','dflux','alpha_b','idisk'] + ['imu1']
    
    axis_values, pixelgrid, labels, header = _prepare_grid(passband, atm,
                   nointerp_columns=nointerp_columns, data_columns=['alpha_b'],
                   log_columns=[])

    # Prepare input: the variable "labels" contains the name of all the
    # variables (teff, logg, ebv, etc... which can be interpolated. If all z
    # values for example are equal, we do not need to interpolate in
    # metallicity, and _prepare_grid will have removed the 'z' label from that
    # list. In the following 3 lines of code, we collect only those parameters
    # which need to be interpolated in the grid. Beware that this means that the
    # z variable will simply be ignored! If you ask for z=0.1 but all values in
    # the grid are z=0.0, this function will not complain and just give you the
    # z=0.0 interpolation results!
    n_dim = 1
    
    for i, label in enumerate(labels):
        
        # if the label is an atmosphere keyword and has a length, determine its
        # length
        if label in atm_kwargs and hasattr(atm_kwargs[label], '__len__'):
            n_dim = max(n_dim, len(atm_kwargs[label]))
        
        # if the label is a reddening keyword and has a length, determine its
        # length
        elif label in red_kwargs and hasattr(red_kwargs[label], '__len__'):
            n_dim = max(n_dim, len(red_kwargs[label]))
        
        # if the label is the 'vgamma' keyword and has a length, determine its
        # length
        elif label == 'vgamma' and hasattr(vgamma, '__len__'):
            n_dim = max(n_dim, len(vgamma))
        
        #else:
        #    raise ValueError("Somethin' wrong with the atmo table: ")
    
    # Initiate the array to hold the grid dimensions and interpolated values
    values = np.zeros((len(labels), n_dim))
    
    for i, label in enumerate(labels):
        
        # Get the value from the atm_kwargs or red_kwargs
        if label in atm_kwargs:
            values[i] = atm_kwargs[label]
        elif label in red_kwargs:
            values[i] = red_kwargs[label]
        elif label == 'vgamma':
            values[i] = vgamma
        elif label == 'ebv':
            values[i] = red_kwargs['extinction'] / red_kwargs['Rv']
        else:
            raise ValueError(("Somethin' wrong with the atmo table: cannot "
                              "interpret label {}").format(label))
    
    # Interpolate everything or just one value
    if not interp_all:
        values = values[:, :1]
    
    # Try to interpolate
    try:
        pars = interp_nDgrid.interpolate(values, axis_values, pixelgrid, order=order)
        if np.any(np.isnan(pars[-1])) or np.any(np.isinf(pars[-1])):
            raise IndexError
    
    # Things can go outside the grid
    except IndexError:
        msg = ", ".join(['{:.3f}<{}<{:.3f}'.format(values[i].min(), labels[i],\
                                 values[i].max()) for i in range(len(labels))])
        msg = ("Parameters outside of grid {}: {}. Consider using a different "
               "atmosphere/limbdarkening grid, or use the black body "
               "approximation.").format(atm, msg)
        raise ValueError(msg)
        #logger.error(msg)
        
        pars = np.zeros((pixelgrid.shape[-1], len(values[0])))
    
    # That's it!
    if return_header:
        return pars[0], header
    else:
        return pars[0]




def local_boosting(system, parset_pbdep, parset_isr={}, beaming_alg='simple'):
    """
    Calculate local boosting coefficients.
    
    """
    # Get the arguments we need concerning normal emergent intensities (atm),
    # LD coefficients and the LD function.
    atm = parset_pbdep['atm']
    
    # Passband: if it's not available (e.g. in the "Star" parset), we need to
    # take the bolometric passband
    passband = parset_pbdep.get('passband', 'OPEN.BOL')
    
    # The reference we need to take to compute stuff (if not available, it's
    # bolometric)
    ref = parset_pbdep.get('ref', '__bol')
    
    # Now, in which parameters do we need to interpolate? This is probably teff
    # and logg, but perhaps abundances, reddening,.. hell, perhaps even micro-
    # turbulent velocity or mixing length parameter.
    # 
    # We need parameters on the passband and reddening in the form of a
    # dictionary. We may add too much information, since the function
    # "interp_ld_coeffs" only extracts those that are relevant.
    atm_kwargs = dict(parset_pbdep)
    
    # No reddening for bolometric fluxes!
    if ref == '__bol':
        red_kwargs = {}
        logger.info(("Not propagating interstellar reddening info for boosting flux "
                    "(taking default from grid)"))
    else:
        red_kwargs = dict(parset_isr)
    
    # Some info
    atm = parset_pbdep['atm']
    ld_func = parset_pbdep['ld_func']
    
    # Find the possible interpolation parameters
    if beaming_alg == 'simple':
        atm_kwargs = {key:system.mesh[key][0] for key in config.atm_props[atm]}
    else:
        atm_kwargs = {key:system.mesh[key] for key in config.atm_props[atm]}
        
    atm_file = choose_ld_coeffs_table(atm, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma,
                                      ld_func=ld_func,
                                      fitmethod='equidist_r_leastsq')
    
    # We'll need to prepare the grid all over again, since we want different
    # columns:
    alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                    red_kwargs=red_kwargs, vgamma=vgamma)
    
    return alpha_b





def legendre(x):
    pl = [np.ones_like(x), x]
    denom = 1.0
    for i in range(2, 10):
        fac1 = x*(2*denom+1)
        fac2 = denom
        denom += 1
        pl.append((fac1*pl[-1]-fac2*pl[-2])/denom)
    return pl


def interp_ld_coeffs_wd(atm,passband,atm_kwargs={},red_kwargs={},vgamma=0):
    """
    Interpolate an atmosphere table using the WD method.
    
    Example usage:
    
        >>> atm_kwargs = dict(teff=6000., logg=4.0, abun=0.)
        >>> interp_ld_coeffs_wd('atmcof.dat','JOHNSON.V',atm_kwargs=atm_kwargs)
        
    Remarks:
    
        - reddening (C{red_kwargs}) is not implemented
        - doppler beaming (C{vgamma}) is not implemented
    
    Compare with native `Phoebe` LD support:
    
    @param atm: atmosphere table filename or alias
    @type atm: string
    @param atm_kwargs: dict with keys specifying the atmospheric parameters
    @type atm_kwargs: dict
    @param red_kwargs: dict with keys specifying the reddening parameters
    @type red_kwargs: dict
    @param vgamma: radial velocity
    @type vgamma: float/array
    @param passband: photometric passband
    @type passband: str
    """
    # Get atmospheric properties
    m = atm_kwargs.get('abun', 0)
    l = atm_kwargs.get('logg', 4.5)
    t = atm_kwargs.get('teff', 10000)
    p = passband
    
    # Check the bounds; Pieter will probably modify these to array tests.
    if np.any(l < 0.0) or np.any(l > 5.0):
        logger.error("log(g) outside of grid: Consider using a different atmosphere model")
    if np.any(m < -5.0) or np.any(m > 1.0):
        logger.error("[M/H] outside of grid: Consider using a different atmosphere model")
    if np.any(t < 3500) or np.any(t > 50000):
        logger.error("Teff outside of grid: Consider using a different atmosphere model")
    
    # Make sure all the interpolatable quantities are arrays of the same length
    # not that some (or all) of them can be scalars
    length = max([len(i)  if hasattr(i, '__len__') else 1 for i in [m, l, t]])
    
    # So if they are not arrays, make them in arrays with constant values
    if not hasattr(m, '__len__'):
        nm = [m]*length
    else:
        nm = m
    if not hasattr(l, '__len__'):
        nl = [l]*length
    else:
        nl = l
    if not hasattr(t, '__len__'):
        nt = [t]*length
    else:
        nt = t
    
    # Prepare lists for interpolation
    M = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
         -0.5, -0.3, -0.2, -0.1,  0.0,  0.1,  0.2,  0.3,  0.5, 1.0]
    L = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    P = ['STROMGREN.U', 'STROMGREN.V', 'STROMGREN.B', 'STROMGREN.Y',
         'JOHNSON.U', 'JOHNSON.B', 'JOHNSON.V', 'JOHNSON.R', 'JOHNSON.I',
         'JOHNSON.J', 'JOHNSON.K', 'JOHNSON.L', 'JOHNSON.M', 'JOHNSON.N',
         'COUSINS.R', 'COUSINS.I', 'OPEN.BOL', None, None, None, None, None,
         'TYCHO2.BT', 'TYCHO2.VT', 'HIPPARCOS.HP']
    
    # Get the atmosphere table's location and read it in.
    if not os.path.isfile(atm):
        atm = os.path.join(basedir_ld_coeffs, atm)
    table = _prepare_wd_grid(atm)
    
    P_index = P.index(p)
    
    ints = np.zeros(length)
    for i, (m, l, t) in enumerate(zip(nm, nl, nt)):
        # Bracket the values by finding the index of the first element in
        # the respective array that is larger than passed value (or None if
        # there is no such element):
        idx = next((i for i,v in enumerate(M) if m < v), None)
        Mb = M[idx-1:idx+1]
        idx = next((i for i,v in enumerate(L) if l < v), None)
        Lb = L[idx-1:idx+1]

        # These brackets serve as our interpolation axes; we need to prepare
        # them for the linear multi-D interpolator:
        grid_pars = np.array([(mm,ll) for mm in Mb for ll in Lb]).T

        # Next we need to prepare the data. There will be 4 interpolation
        # knots: (Mlo, Llo), (Mlo, Lhi), (Mhi, Llo) and (Mhi, Lhi). For each
        # knot we need to take the corresponding line from the 'atmcof.dat'
        # table, run a Legendre-powered interpolator and compute the
        # intensity. Variable grid_data will hold all 4 intensities.
        grid_data = []
        
        for (mm, ll) in grid_pars.T:
            # Get the pointer to the (P,M,L) block in the 'atmcof.dat' table:
            idx = (18-np.searchsorted(M, mm))*len(P)*len(L)*4 + P_index*len(L)*4 + np.searchsorted(L, ll)*4
            
            # Bracket the temperature (much like we did above for Mb and Lb):
            print [table[idx+j,1] for j in range(4)]
            j = next((i for i,v in enumerate([table[idx+j,1] for j in range(4)]) if v > t), None)
            print m,l,t, mm, ll, j
            # Read out Legendre coefficients for the bracketed temperature and
            # compute the intensity from the series expansion:
            Cl = table[idx+j,2:]
            teff = (t-table[idx+j,0])/(table[idx+j,1]-table[idx+j,0])
            Pl = np.array(legendre(teff))
            grid_data.append(np.sum(Cl*Pl, axis=0))
            #print ("DEBUG: (%1.1f %2.2f %1.1f): mm=%2.2f ll=%2.2f j=%d Teff in [%2.2f,%2.2f] I=%12.9f" % (m, l, t, mm, ll, j, table[idx+j,0], table[idx+j,1], grid_data[-1]))

        # Prepare the data for the interpolator:
        grid_data = np.array([grid_data])

        # Perform the interpolation:
        av, pg = interp_nDgrid.create_pixeltypegrid(grid_pars, grid_data)
        p = np.array([[m], [l]])
        val = interp_nDgrid.interpolate(p, av, pg)[0]
        
        # Store the result; it is in log10, per angstrom.
        ints[i] = val
        #print ("DEBUG: Iinterp = %12.9f" % (val))
        
    return np.array([1e-8*10**ints])

@decorators.memoized
def _prepare_wd_grid(atm):
    logger.info("Prepared WD grid {}: interpolate in teff, logg, abun".format(os.path.basename(atm)))
    return ascii.loadtxt(atm)


@decorators.memoized
def _prepare_grid(passband,atm, data_columns=None, log_columns=None,
                  nointerp_columns=None, cuts=None):
    """
    Read in a grid of limb darkening coefficients for one passband.
    
    The FITS file is read in, the interpolation table is prepared and the
    results are memoized.
    
    @param passband: name of the passband. This name should be an extension
    of the grid FITS file
    @type passband: str
    @param atm: filename or recognised grid name
    @type atm: str
    @return: output that can be used by L{interp_nDgrid.interpolate} (axis
    values, pixelgrid and labels)
    @rtype: array,array,list
    """
    if data_columns is None:
        #-- data columns are C{ai} columns with C{i} an integer, plus C{Imu1} that
        #   holds the centre-of-disk intensity
        data_columns = ['a{:d}'.format(i) for i in range(1,10)]+['imu1']
    if log_columns is None:
        #-- some columns are transformed to log for interpolation, because the
        #   data behaves more or less linearly in log scale.
        log_columns = ['imu1']#,'Teff']
    if nointerp_columns is None:
        #-- not all columns hold data that needs to be interpolated
        nointerp_columns = data_columns + ['alpha_b', 'res','dflux','idisk']
        #nointerp_columns = data_columns + ['alpha_b', 'dflux']
    with pyfits.open(atm) as ff:
        header = ff[0].header
        try:
            available = [col.lower() for col in ff[passband].data.names]
        except Exception as msg:
            # Give useful infor for the user
            # derive which files where used:
            extra_msg = "If the atmosphere files "
            with pyfits.open(atm) as ff:
                for key in ff[0].header:
                    if 'C__ATMFILE' in key:
                        extra_msg += ff[0].header[key]+', '
            extra_msg+= 'exist in directory {}, '.format(get_paths()[1])
            extra_msg+= 'you can add it via the command\n'
            passband_name = ".".join(str(msg).split("\'")[-2].split('.')[:-1]) + '*'
            extra_msg+= '>>> phoebe.atmospheres.limbdark.compute_grid_ld_coeffs("{}", passbands=("{}",))'.format(atm, passband_name)
            raise ValueError("Atmosphere file {} does not contain required information ({:s})\n{}".format(atm,str(msg),extra_msg))
        #-- remove columns that are not available and derive which parameters
        #   need to be interpolated
        data_columns = [col for col in data_columns if col in available]
        pars_columns = [col for col in available if not col in nointerp_columns]
        grid_pars = np.vstack([(np.log10(ff[passband].data.field(col)) if col in log_columns else ff[passband].data.field(col)) for col in pars_columns])
        grid_data = np.vstack([(np.log10(ff[passband].data.field(col)) if col in log_columns else ff[passband].data.field(col)) for col in data_columns])            
    #-- remove obsolete axis: these are the axis for which all values are the
    #   same. This can happen if the grid is not computed for different
    #   metallicities, radial velocities or reddening values, for example.
    keep_columns = np.ones(grid_pars.shape[0],bool)
    labels = []
    for i,par_column in enumerate(pars_columns):
        if np.allclose(grid_pars[i],grid_pars[i][0]):
            keep_columns[i] = False
            logger.info('Ignoring column {} during interpolation'.format(par_column))
        else:
            labels.append(par_column)
    grid_pars = grid_pars[keep_columns]
    
    # convert the imu from erg/s/cm2/AA to W/m3
    if 'imu1' in data_columns:
        index = data_columns.index('imu1')
        grid_data[index] = np.log10(10**grid_data[index]*1e-3)
    
    # Prepare to cut
    if cuts is not None:
        keep = np.ones(grid_pars.shape[1], bool)
        for lbl in cuts:
            col = labels.index(lbl)
            keep = keep & (cuts[lbl][0]<=grid_pars[col]) & (grid_pars[col]<=cuts[lbl][1])
            logger.info("Cut in {}: {}".format(lbl, cuts[lbl]))
            
        grid_pars = grid_pars[:,keep]
        grid_data = grid_data[:,keep]

    
    logger.info("Prepared grid {}: interpolate in {}".format(os.path.basename(atm),", ".join(labels)))
            
    #-- create the pixeltype grid
    axis_values, pixelgrid = interp_nDgrid.create_pixeltypegrid(grid_pars,grid_data)
    return axis_values, pixelgrid, labels, header


#}

#{ Grid creators

def compute_grid_ld_coeffs(atm_files,atm_pars=('teff', 'logg'),\
                           red_pars_iter={},red_pars_fixed={},vgamma=None,\
                           passbands=('JOHNSON.V',),\
                           law='claret',fitmethod='equidist_r_leastsq',\
                           limb_zero=True, add_boosting_factor=True,\
                           filetag='kurucz', debug_plot=False,
                           check_passband_coverage=True):
    r"""
    Create an interpolatable grid of limb darkening coefficients.
    
    A FITS file will be created, where each extension has the name of the
    passband. For reference, there is also an extension ``_REF_PASSBAND`` for
    each passband with the used response curves, for later reference.
    
    Each extension contains a  data table with some information on the fit
    and the fit statistics, as well as the limb darkening coefficients and the
    local normal emergent intensities (i.e. the intensities at the center of the
    disk which scale the fluxes in an absolute way).
    
    Boosting can be included in two ways, which can be easily combined if
    necessary. The first method is the physically most precise: beaming is
    then included by shifting the SED according the radial velocity ``vgamma``,
    and then computing intensities and limb darkening coefficients (see
    :py:func:`get_limbdarkening`). The other way is to compute the (linear)
    beaming factor :math:`\alpha_b` of an SED (:math:`(\lambda, F_\lambda)`) via
    (set :envvar:`add_boosting_factor` to ``True``):
    
    .. math::
            
        \alpha_b = \frac{\int_P (5+\frac{d\ln F_\lambda}{d\ln\lambda}\lambda F_\lambda d\lambda}{\int_P \lambda F_\lambda d\lambda}
    
    To get :math:`F_\lambda` from specific intensities, first a disk integration
    is performed (equivalent to a traditional spectral energy distribution):
    
    .. math::
        
        r = \sqrt{ 1- \mu^2}\\
        F_\lambda = \sum\left( \pi (\Delta r)^2 I_\lambda(\mu)\right)
        
    When boosting factors are computed, also disk-integrated intensities are
    computed. That might be beneficial for some calculations.
        
    There is a lot of flexibility in creating grids with limb darkening grids,
    to make sure that the values you are interested in can be interpolated if
    necessary, or otherwise be fixed. Therefore, creating these kind of grids
    can be quite complicated. Hopefully, the following examples help in
    clarifying the methods.
    
    If you have no specific intensity files available, the only thing you can
    do is use the :py:func:`black body <phoebe.atmospheres.sed.blackbody>`
    approximation. Then you have no choice in interpolation parameters (i.e.
    the only free parameter is effective temperature), nor limb darkening law
    (uniform), but you can add reddening or beaming:
    
    **Black body examples**
    
    Note: due to a strange twist of faith, there is no parameter to set the
    effective temperature range for blackbodies. If you want to expand the
    parameter space from the default 10 - 100kK in 200 steps, log-space sampled,
    you need to change the code in :py:func:`iter_grid_dimensions`. Sorry!
    
    Case 0: plain black body
    
        >>> compute_grid_ld_coeffs('blackbody')
        
    Case 1: black body with beaming
    
        >>> vgamma = np.linspace(-500, 500, 21) # km/s
        >>> compute_grid_ld_coeffs('blackbody', vgamma=vgamma)
    
    Case 2: black body with reddening
    
        >>> ebv = np.linspace(0, 0.5, 5))
        >>> compute_grid_ld_coeffs('blackbody',
        ...       red_pars_iter=dict(ebv=ebv),
        ...       red_pars_fixed=dict(law='fitzpatrick2004', Rv=3.1))
    
    When using real atmosphere files, you have a few more options:
    
    Prerequisite: you have a list of atmosphere files with at least one
    element. These tables contain specific intensities spanning a certain grid
    of parameters.
    
    **Example usage:**
    
    **Case 0**: You want to run it with minimal settings (i.e. using all the
    default settings):
    
        >>> atm_files = ['spec_intens/kurucz_mu_ip00k2.fits']
        >>> compute_grid_ld_coeffs(atm_files)
    
    **Case 1**:  You want to compute limb darkening coefficients for a Kurucz
    grid in ``teff`` and ``logg`` for solar metallicity. You want to use the
    Claret limb darkening law and have the 2MASS.J and JOHNSON.V filter. To fit
    them, you want to use the Levenberg-Marquardt method, with the stellar disk
    sampled equidistantly in the disk radius coordinate (:math:`r`, as opposed
    to limb angle :math:`\mu`). Finally you want to have a file written with the
    prefix ``kurucz_p00`` to denote solar metallicity and grid type:
    
        >>> atm_files = ['spec_intens/kurucz_mu_ip00k2.fits']
        >>> compute_grid_ld_coeffs(atm_files, atm_pars=('teff','logg'),
        ...      law='claret', passbands=['2MASS.J','JOHNSON.V'],
        ...      fitmethod='equidist_r_leastsq', filetag='kurucz_p00')
        
    This will create a FITS file, and this filename can be used in the
    C{atm} and C{ld_coeffs} parameters in Phoebe.
    
    **Case 2**: you want to add a passband to an existing file, say the one
    previously generated. Since all settings are saved inside the FITS header,
    you simply need to do:
        
        >>> atm_file = 'kurucz_p00_claret_equidist_r_leastsq_teff_logg.fits'
        >>> compute_grid_ld_coeffs(atm_file, passbands=['2MASS.J'])
    
    **Case 3**: Like Case 1, but with Doppler beaming included:
    
        >>> atm_files = ['spec_intens/kurucz_mu_ip00k2.fits']
        >>> compute_grid_ld_coeffs(atm_files, atm_pars=('teff','logg'),
        ...      vgamma=[-500, -250, 0, 250, 500],
        ...      law='claret', passbands=['2MASS.J', 'JOHNSON.V'],
        ...      fitmethod='equidist_r_leastsq', filetag='kurucz_p00')
    
    Note that not all parameters you wish to interpolate need to be part of the
    precomputed tables. This holds for the radial velocity, but also for the
    extinction parameters.
    
    **Case 4**: Suppose you want to make a grid in E(B-V):
    
        >>> atm_files = ['spec_intens/kurucz_mu_ip00k2.fits']
        >>> compute_grid_ld_coeffs(atm_files, atm_pars=('teff', 'logg'),
        ...       red_pars_iter=dict(ebv=np.linspace(0, 0.5, 5)),
        ...       red_pars_fixed=dict(law='fitzpatrick2004', Rv=3.1),
        ...       passbands=('JOHNSON.V',), fitmethod='equidist_r_leastsq',
        ...       filetag='kurucz_red')
    
    You need to split up the reddening parameters in those you which to `grid'
    (C{red_pars_iter}), and those you want to keep fixed (C{red_pars_fixed}).
    The former needs to be a dictionary, with the key the parameter name you
    wish to vary, and the value an array of the parameter values you wish to put
    in the grid. 
    
    **Case 5**: Like Case 1, but with different abundances. Then you need a
    list of different atmopshere files. They need to span now ``teff``, ``logg``
    and ``abund``. We compute the LD coefficients in all Johnson bands:
    
        >>> atm_files = ['spec_intens/kurucz_mu_ip00k2.fits',
        ...              'spec_intens/kurucz_mu_ip05k2.fits',
        ...              'spec_intens/kurucz_mu_ip10k2.fits']
        >>> compute_grid_ld_coeffs(atm_files,atm_pars=atm_pars,passbands=('JOHNSON*',),
            fitmethod='equidist_r_leastsq', filetag='kurucz')
    
    **Case 6**: Putting it all together:
    
        >>> compute_grid_ld_coeffs(atm_files, atm_pars=('teff', 'logg', 'abun'),
        ...       red_pars_iter=dict(ebv=np.linspace(0, 0.5, 5),
        ...                          Rv=[2.1, 3.1, 5.1]),
        ...       red_pars_fixed=dict(law='fitzpatrick2004'),
        ...       passbands=('JOHNSON.V',), fitmethod='equidist_r_leastsq',
        ...       filetag='kurucz_all')
    
    @param atm_files: a list of files of grids specific intensities that will be
     used to integrate the passbands over
    @type atm_files: list of str
    @param atm_pars: names of the variables that will be interpolated in by
     :py:func:`interp_ld_coeffs` (e.g. `teff`, `logg`...) except radial velocity
     (that needs to be given by ``vgamma`` (see below)) and reddening parameters
     (those need to be given by ``red_pars_iter``). Thus, all variables listed
     here need to be inherent to the atmospheres themselves.
    @type atm_pars: tuple of str
    @param red_pars_iter: dictionary with reddening parameters that need to be
     interpolated over by :py:func:`interp_ld_coeffs`. The keys are the names
     of the parameters, the values are arrays for the grid.
    @type red_pars_iter: dictionary
    @param red_pars_fixed: dictionary with reddening parameters that need to
     be fixed. The keys are the names of the parameters, the values the
     corresponding fixed values.
    @type red_pars_fixed: dictionary
    @param vgamma: list of values to use for Doppler beaming (i.e. velocity in
     km/s)
    @type vgamma: list of floats
    @param passbands: list of passbands to include. You are allowed to use
     ``'2MASS*'`` to denote all 2MASS filters
    @type passbands: list of str
    @param law: limb darkening law to use
    @type law: str, any recognised limb darkening law
    @param fitmethod: method used to fit the limb darkening law
    @type fitmethod: any recognised fit method
    @param filetag: tag used as prefix for the generated filename
    @type filetag: str
    @param add_boosting_factor: compute linear boosting factors and
     disk-integrated intensities
    @type add_boosting_factor: bool
    @param check_passband_coverage: if True, remove passbands that are not fully
     covered from the computations
    @param check_passband_coverage: bool
    """
    overwrite = None
    
    # Some convenience defaults for black body calculations
    if atm_files == 'blackbody':
        atm_files = ['blackbody']
        law = 'uniform'
        atm_pars = ('teff',)
        filetag = 'blackbody'
        fitmethod = 'none'
    
    # Let's prepare the list of passbands:
    #  - make sure 'OPEN.BOL' is always in there, this is used for bolometric
    #    intensities
    #  - Make sure the list of passbands has no duplicates
    #  - expand '*' to all passbands matching the pattern (this also removes
    #    passbands that are not defined)
    passbands_ = sorted(list(set(list(passbands))))
    passbands = []
    for passband in passbands_:
        these_responses = pbmod.list_response(name=passband.upper())
        passbands += these_responses
        if not these_responses:
            logger.warning('No passbands found matching pattern {}'.format(passband))
    passbands = sorted(set(passbands + ['OPEN.BOL']))
    
    # Sort out which passbands are not calibrated
    filter_info = pbmod.get_info()
    available = set(list(filter_info['passband']))
    overlap = set(passbands) & available
    logger.info("Skipping passbands {} (no calibration available)".format(sorted(list(set(passbands)-available))))
    passbands = sorted(list(overlap))
    
    # If the user gave a list of files, it needs to a list of specific
    # intensities.
    if not isinstance(atm_files, str):
        
        # The filename contains the filetag, the ld law, the fit method, ...
        filename = '{}_{}_{}_'.format(filetag, law, fitmethod)
        filename+= '_'.join(atm_pars)
        
        # ... reddening parameters (free), ...
        if red_pars_iter:
            filename += '_' + '_'.join(sorted(list(red_pars_iter.keys())))
            
        # ... reddening parameters (fixed), ...
        for key in red_pars_fixed:
            filename += '_{}{}'.format(key,red_pars_fixed[key])
            
        # ... radial velocity (if given)
        if vgamma is not None:
            filename += '_vgamma'
            
        filename += '.fits'
        
        logger.info("Creating new grid from {}".format(", ".join(atm_files),))
        logger.info("Using passbands {}".format(", ".join(passbands)))
    
    # If the user gave one file, it needs to a previously computed grid of
    # limbdarkening coefficients. It that case we can append new passbands to it
    else:
        filename = atm_files
        
        # Safely backup existing file if needed
        if os.path.isfile(filename+'.backup'):
            if overwrite is None:
                answer = raw_input('Backup file for {} already exists. Overwrite? (Y/n): '.format(filename))
                if answer == 'n':
                    overwrite = False
                elif answer == 'y':
                    overwrite = None
                elif answer == 'Y' or answer == '':
                    overwrite = True
                else:
                    raise ValueError("option '{}' not recognised".format(answer))
            
            if overwrite is True:
                shutil.copy(filename,filename+'.backup')
        
        # If we need to expand an existing file, we need to deduce what
        # parameters were used to compute it.
        
        # Sort out which passbands are already computed
        with pyfits.open(filename) as ff:
            existing_passbands = [ext.header['extname'] for ext in ff[1:]]
        
        overlap = set(passbands) & set(existing_passbands)
        if len(overlap):
            logger.info(("Skipping passbands {} (they already exist"
                         ")").format(overlap))
        
        passbands = sorted(list(set(passbands) - set(existing_passbands)))
        if not passbands:
            logger.info("Nothing to compute, all passbands are present")
            return None
    
        
        # Set all the parameters from the contents of this FITS file
        with pyfits.open(filename) as ff:
            
            # Fitmethod and law are easy
            fitmethod = ff[0].header['C__FITMETHOD']
            law = ff[0].header['C__LD_FUNC']
            
            # Vgamma: fitted and range?
            if 'vgamma' in ff[1].columns.names:
                vgamma = np.sort(np.unique(ff[1].data.field('vgamma')))
            else:
                vgamma = None
            
            # Fixed reddening parameters and specific intensity files require
            # a bit more work:
            red_pars_fixed = dict()
            red_pars_iter = dict()
            atm_files = []
            atm_pars = []        
            
            for key in ff[0].header.keys():
                
                # Atmosphere parameter names:
                if key[:6] == 'C__AI_':
                    atm_pars.append(key[6:])
                
                # Iterated reddening parameters:
                if key[:6] == 'C__RI_':
                    name = key[6:]
                    red_pars_iter[name] = np.sort(np.unique(ff[1].data.field(name)))
                
                # Fixed reddening parameters:
                if key[:6] == 'C__RF_':
                    red_pars_fixed[key[6:]] = ff[0].header[key]

                # Specific intensity files (should be located in correct
                # directory (phoebe/atmospheres/tables/spec_intens)
                elif key[:10] == 'C__ATMFILE':
                    direc = os.path.dirname(os.path.abspath(__file__))
                    atm_file = os.path.join(direc, 'tables', 'spec_intens',
                                            ff[0].header[key])
                    atm_files.append(atm_file)
            
    # Build a comprehensive logging message
    red_pars_fix_str = ['{}={}'.format(key, red_pars_fixed[key]) for \
          key in red_pars_fixed]
    red_pars_fix_str = ", ".join(red_pars_fix_str)
    logger.info(("Using specific intensity files "
                "{}").format(", ".join(atm_files)))
    logger.info(("Settings: fitmethod={}, ld_func={}, "
                 "reddening={}").format(fitmethod, law, red_pars_fix_str))
    logger.info(("Grid in atmosphere parameters: "
                 "{}").format(", ".join(atm_pars)))
    logger.info(("Grid in reddening parameters: "
                 "{}").format(", ".join(list(red_pars_iter.keys()))))
    logger.info(("Grid in vgamma: {}".format(vgamma)))
            
    # Collect parameter names and values
    atm_par_names = list(atm_pars)
    red_par_names = sorted(list(red_pars_iter.keys()))
    other_pars = [red_pars_iter[name] for name in red_par_names]
    if vgamma is not None:
        other_pars.append(vgamma)
        
    for atm_file in atm_files:
        do_close = False
        # Special case if blackbody: we need to build our atmosphere model
        if atm_file == 'blackbody':
            wave_ = np.logspace(1, 6, 10000)
        
        # Check if the file exists
        elif not os.path.isfile(atm_file):
            raise ValueError(("Cannot fit LD law, specific intensity file {} "
                             "does not exist").format(atm_file))
        
        # Else we pre-open the atmosphere file to try to gain some time
        else:
            do_close = True
            open_atm_file = pyfits.open(atm_file)
            
            # Now, it's possible that we're not dealing with specific intensities
            # but rather disk-integrated fluxes. In that case, we can only
            # assume uniform limb darkening (i.e. rescale the integrated
            # intensities such that we have a normal intensities, that, if
            # disk-integrated assuming a uniform law, reproduces the original
            # intensity). If the given law is not uniform, we raise an error.
            available_fields = [name.lower() for name in open_atm_file[1].data.dtype.names]
            if 'wavelength' in available_fields and 'flux' in available_fields:
                
                if not law in ['uniform', 'sed']:
                    raise ValueError("Cannot fit LD law: disk integrated intensities require uniform LD law")
                
                # we change the law in "sed" to distinguish between uniform
                # LD and these things
                law = 'sed'
                
        
        # Check for passband/specific intensity wavelength coverage, and remove
        # any filter that is not completely covered by the specific intensities
        if check_passband_coverage and atm_file != 'blackbody':
            min_wave = -np.inf
            max_wave = +np.inf
            # Run over all extenstions, and check for minimum/maximum wavelength
            # We want to have the "maximum minimum", i.e. the minimum wavelength
            # range that is covered by all extensions
            for ext in open_atm_file[1:]:
                if 'wavelength' in ext.data.dtype.names:
                    this_min = ext.data.field('wavelength').min()
                    this_max = ext.data.field('wavelength').max()
                    if this_min > min_wave:
                        min_wave = this_min
                    if this_max < max_wave:
                        max_wave = this_max
            logger.info("Wavelength range from specific intensities: {} --> {} AA".format(min_wave, max_wave))
            
            # Check passbands:
            for pb in passbands+[]:
                if pb == 'OPEN.BOL':
                    continue
                wave_, trans_ = pbmod.get_response(pb)
                has_trans = trans_ > 0.0
                this_min = wave_[has_trans].min()
                this_max = wave_[has_trans].max()
                
                if this_min < min_wave or this_max > max_wave:
                    # Compute coverage percentage:
                    new_wave_pb = np.linspace(this_min, this_max, 1000)
                    new_wave_cv = np.linspace(min_wave, max_wave, 1000)
                    new_pb = np.interp(new_wave_pb, wave_, trans_, left=0, right=0)
                    new_cv = np.interp(new_wave_cv, wave_, trans_, left=0, right=0)
                    covpct = (1.0 - np.trapz(new_cv, x=new_wave_cv) / np.trapz(new_pb, x=new_wave_pb))*100.
                    logger.info('Removing passband {}: not completely covered by specific intensities ({:.3f}% missing)'.format(pb, covpct))
                    passbands.remove(pb)
             
            logger.info("Using passbands {}".format(", ".join(passbands))) 
        
        if do_close:
            open_atm_file.close()
                
    # Prepare to store the results: for each passband, we need an array. The
    # rows of that array will be the grid points + coefficients + fit statistics
    output = {passband:[] for passband in passbands}
    
    for atm_file in atm_files:                
        
        # Special case if blackbody: we need to build our atmosphere model
        if atm_file == 'blackbody':
            wave_ = np.logspace(1, 7, 10000)
        
        # Check if the file exists
        elif not os.path.isfile(atm_file):
            raise ValueError(("Cannot fit LD law, specific intensity file {} "
                             "does not exist").format(atm_file))
        
        # Else we pre-open the atmosphere file to try to gain some time
        else:
            open_atm_file = pyfits.open(atm_file)
        
        iterator = iter_grid_dimensions(atm_file, atm_par_names, other_pars)
        for nval, val in enumerate(iterator):
            
            logger.info("{}: {}".format(atm_file, val))
            
            # The first values are from the atmosphere grid
            atm_kwargs = {name:val[i] for i, name in enumerate(atm_par_names)}
            val_ = val[len(atm_par_names):]
            
            # The next values are from the reddening
            red_kwargs = {name:val_[i] for i, name in enumerate(red_par_names)}
            
            #   but we need to also include the fixed ones
            for key in red_pars_fixed:
                red_kwargs[key] = red_pars_fixed[key]
            val_ = val_[len(red_par_names):]
            
            # Perhaps there was vrad information too.
            if val_:
                vgamma = val_[0]
            
            # Retrieve the limb darkening law
            if atm_file != 'blackbody':
                mu, Imu = get_limbdarkening(open_atm_file,
                                       atm_kwargs=atm_kwargs,
                                       red_kwargs=red_kwargs, vgamma=vgamma,
                                       passbands=passbands)

            # Blackbodies must have a uniform LD law
            elif law == 'uniform':
                Imu_blackbody = sed.blackbody(wave_, val[0], vrad=vgamma)
                Imu = sed.synthetic_flux(wave_*10, Imu_blackbody, passbands)
            
            else:
                raise ValueError("Blackbodies must have a uniform LD law (you can adjust them later)")
            
            # Compute boosting factor if necessary, and in that case also
            # compute the disk-integrated passband intensity
            if add_boosting_factor:
                extra = []
                
                if atm_file != 'blackbody':
                    mus, wave, table = get_specific_intensities(open_atm_file,
                                     atm_kwargs=atm_kwargs, red_kwargs=red_kwargs,
                                     vgamma=vgamma)
                    
                    if not np.isnan(mus[0]):
                        # Disk-integrate the specific intensities
                        rs_ = np.sqrt(1 - mus**2)
                        flux = (np.pi*np.diff(rs_[None,:]**2) * table[:,:-1]).sum(axis=1)
                    else:
                        # otherwise it's already disk-integrated
                        flux = table[:,0]
                
                else:
                    wave, flux = wave_*10, Imu_blackbody                                
                
                # Compute disk-integrated fluxes
                disk_integrateds = sed.synthetic_flux(wave, flux, passbands)
                
                # transform fluxes and wavelength
                lnF = np.log(flux)
                lnl = np.log(wave)
                # Numerical derivatives
                dlnF_dlnl = utils.deriv(lnl, lnF)
                # Fix nans and infs
                dlnF_dlnl[np.isnan(dlnF_dlnl)] = 0.0
                dlnF_dlnl[np.isinf(dlnF_dlnl)] = 0.0
                
                # Other definitions of numerical derivatives
                #-- Simple differentiation
                #dlnF_dlnl2 = np.hstack([0,np.diff(lnF)/ np.diff(lnl)])
                #dlnF_dlnl2[np.isnan(dlnF_dlnl2)] = 0.0
                #dlnF_dlnl2[np.isinf(dlnF_dlnl2)] = 0.0
                #-- Spline differentiation
                #splfit = splrep(lnl[-np.isinf(lnF)], lnF[-np.isinf(lnF)], k=2)
                #dlnF_dlnl3 = splev(lnl, splfit, der=1)
                
                # compute boosting factor
                w_fl = wave*flux
                for i, pb in enumerate(passbands):
                    pwave, ptrans = pbmod.get_response(pb)
                    Tk = np.interp(wave, pwave, ptrans)
                    num = np.trapz(Tk * (5 + dlnF_dlnl) * w_fl, x=wave)
                    
                    if num == 0:
                        extra.append(0.0)
                        continue
                    
                    den = np.trapz(Tk * w_fl, x=wave)        
                    extra.append(num/den)
                    
                    #num2 = np.trapz(Tk * (5 + dlnF_dlnl2) * w_fl, x=wave)
                    #num3 = np.trapz(Tk * (5 + dlnF_dlnl3) * w_fl, x=wave)
                    #print num/den, num2/den, num3/den
                    #raw_input('continue?')
                    
                    #if np.isnan(num/den):
                        #import matplotlib.pyplot as plt
                        #ax = plt.subplot(211)
                        #plt.title(pb)
                        #plt.plot(lnl, lnF, 'ko-')
                        #plt.twinx(plt.gca())
                        #plt.plot(np.log(pwave), ptrans, 'bo-')
                        #plt.plot(np.log(wave), Tk, 'ro-')
                        #plt.subplot(212)
                        #plt.plot(lnl, dlnF_dlnl, 'k-')
                        ##plt.plot(lnl, dlnF_dlnl2, 'r-')
                        #print num, den, num/den
                        ##num = np.trapz(Tk * (5 + dlnF_dlnl2) * w_fl, x=wave)
                        #den = np.trapz(Tk * w_fl, x=wave)
                        #print num/den

                        #plt.show()
            
            # Only if the law is not uniform, we really need to fit something
            if law != 'uniform' and not law != 'prsa':
                for i, pb in enumerate(passbands):
                    # Fit a limbdarkening law:
                    csol, res, dflux = fit_law(mu, Imu[:, i]/Imu[0, i],
                                               law=law, fitmethod=fitmethod,
                                               limb_zero=limb_zero,
                                               debug_plot=(i+1 if debug_plot else False))
                    if add_boosting_factor:
                        to_append = list(val) + [extra[i], disk_integrateds[i]] + \
                                    [res, dflux] + list(csol) + [Imu[0, i]]
                    else:
                        to_append = list(val) + [res, dflux] + list(csol) + \
                                      [Imu[0, i]]
                    output[pb].append(to_append)
                    
                    logger.info("{}: {}".format(pb, output[pb][-1]))
            
            # For the Prsa law, we need to append values for each mu
            elif law == 'prsa':
                for i, pb in enumerate(passbands):
                    # Fit a limbdarkening law:
                    allcsol, mu_grid, dflux = fit_law(mu, Imu[:, i]/Imu[0, i],
                                               law=law, fitmethod=fitmethod,
                                               limb_zero=limb_zero, oversampling=32,
                                               debug_plot=(i+1 if debug_plot else False))
                    to_append = []
                    for i, imu in enumerate(mu_grid):
                        csol = allcsol[i:i+1]
                        to_append += list(val) + [imu] + [extra[i], disk_integrateds[i]] + \
                                    [res, dflux] + csol + [Imu[0, i]]
                    output[pb].append(to_append)
                    
                    logger.info("{}: {}".format(pb, output[pb][-1]))
            
            # For a uniform limb darkening law, we can just take the 
            # disk-integrated intensity and divide by pi.
            else:
                csol = []
                for i, pb in enumerate(passbands):
                    
                    # So don't fit a law, we know what it is
                    if add_boosting_factor:
                        to_append = list(val) + [extra[i], disk_integrateds[i]] + \
                                    [0.0, 0.0] + [disk_integrateds[i]/np.pi]
                    else:
                        to_append = list(val) + [0.0, 0.0] + [disk_integrateds[i]/np.pi]
                    output[pb].append(to_append)
            
                    logger.info("{}: {}".format(pb, output[pb][-1]))
            
            if debug_plot:
                for i, pb in enumerate(passbands):
                    pl.figure(i+1)
                    pl.subplot(121)
                    pl.title(pb)
                    debug_filename = pb + '__' + '__'.join([str(j) for j in val])
                    debug_filename = debug_filename.replace('.','_')
                    pl.savefig(debug_filename)
                    pl.close()
            
            
        # Close the atm file again if necessary
        if atm_file != 'blackbody':
            open_atm_file.close()
        
    # Write the results to a FITS file
    
    # Names of the columns
    col_names = atm_par_names + red_par_names
    if law == 'prsa':
        col_names += ['mu']
    if vgamma is not None and 'vgamma' not in col_names:
            col_names.append('vgamma')
    if add_boosting_factor:
        col_names.append('alpha_b')
        col_names.append('Idisk')
    col_names = col_names + ['res', 'dflux'] + \
                   ['a{:d}'.format(i+1) for i in range(len(csol))] + ['Imu1']
    
    # If the file already exists, append to it
    if os.path.isfile(filename):
        filename = pyfits.open(filename,mode='update')
    
    # Append the tables
    for pb in sorted(list(output.keys())):
        grid = np.array(output[pb]).T
        header = dict(extname=pb)
        filename = fits.write_array(grid, filename, col_names,
                                    header_dict=header, ext='new', close=False)
        
        # Append the passband for reference
        header = dict(extname='_REF_'+pb)
        grid = np.array(pbmod.get_response(pb))
        filename = fits.write_array(grid, filename, ['WAVELENGTH', 'RESPONSE'],
                                    header_dict=header, ext='new', close=False)
    
    # Store as much information in the FITS header as possible, we could need
    # this latter on if we want to expand a certain table.
    
    # Some information on the type of file, the fitmethod and ld law
    filename[0].header.update('FILETYPE', 'LDCOEFFS', 'type of file')
    filename[0].header.update('HIERARCH C__FITMETHOD', fitmethod,
                              'method used to fit the LD coefficients')
    filename[0].header.update('HIERARCH C__LD_FUNC', law,
                              'fitted limb darkening function')
    filename[0].header.update('HIERARCH C__LIMBZERO', limb_zero,
                              'point at the disk limb included or not')
    
    # Info on the fixed reddening parameters
    for key in red_pars_fixed:
        filename[0].header.update('HIERARCH C__RF_'+key, red_pars_fixed[key],
                                  'fixed reddening parameter')
    
    # Info on the interpolatable reddening parameters
    for key in red_pars_iter:
        filename[0].header.update('HIERARCH C__RI_'+key,
                                  key, 'iterated reddening parameter')
    
    # Info on atmospheric parameters
    for key in atm_pars:
        filename[0].header.update('HIERARCH C__AI_'+key,
                                  key, 'iterated atmosphere parameter')
    
    # Info on whether Doppler boosting is included
    if vgamma is not None:
        filename[0].header.update('HIERARCH C__VI_vgamma',
                                  'vgamma', 'includes Doppler boosting')
    
    # Info on used atmosphere files.
    for iatm, atm_file in enumerate(atm_files):
        filename[0].header.update('HIERARCH C__ATMFILE{:03d}'.format(iatm),
                 os.path.basename(atm_file), 'included specific intensity file')
        
    # Copy keys from first atm_file
    if atm_files[0] != 'blackbody':
        with pyfits.open(atm_files[0]) as gg:
            for key in gg[0].header.keys():
                keys = ['SIMPLE', 'BITPIX', 'NAXIS','NAXIS1','NAXIS2','EXTEND']
                if key in keys:
                    continue
                filename[0].header.update(key, gg[0].header[key],
                                          'key from first atm_file')
        
    filename.close()
        




def match_teff_logg(atm_files):
    """
    Extract the maximal set of teff/logg coordinates that are common to all
    atm_files.
    """
    use = []
    remove = []
    ref_file = atm_files[0]
    for ival in iter_grid_dimensions(ref_file,['teff','logg'],[]):
        ival_is_in_file = 1
        for j,jatm_file in enumerate(atm_files[1:]):
            for jval in iter_grid_dimensions(jatm_file,['teff','logg'],[]):
                if jval[0]==ival[0] and jval[1]==ival[1]:
                    ival_is_in_file += 1
                    break
            else:
                print("Did not find {} in file {}".format(ival,jatm_file))
                break
        if ival_is_in_file==len(atm_files):
            use.append(ival)
            print(ival)
        else:
            remove.append(ival)
    return use,remove
                    
    
#}

#{ Analytical expressions



def intensity_moment(coeffs,ll=0,full_output=False):
    r"""
    Calculate the intensity moment (see Townsend 2003, eq. 39).
    
    
    Test analytical versus numerical implementation:
    
    >>> photband = 'JOHNSON.V'
    >>> l,logg = 2.,4.0
    >>> gridfile = 'tables/ikurucz93_z0.0_k2_ld.fits'
    >>> mu = np.linspace(0,1,100000)
    >>> P_l = legendre(l)(mu)
    >>> check = []
    >>> for teff in np.linspace(9000,12000,19):
    ...    a1x,a2x,a3x,a4x,I_x1 = get_itable(gridfile,teff=teff,logg=logg,mu=1,photband=photband,evaluate=False)
    ...    coeffs = [a1x,a2x,a3x,a4x,I_x1]
    ...    Ix_mu = ld_claret(mu,coeffs)
    ...    Ilx1 = I_x1*np.trapz(P_l*mu*Ix_mu,x=mu)
    ...    a0x = 1 - a1x - a2x - a3x - a4x
    ...    limb_coeffs = [a0x,a1x,a2x,a3x,a4x]
    ...    Ilx2 = 0
    ...    for r,ar in enumerate(limb_coeffs):
    ...        s = 1 + r/2.
    ...        myIls = _I_ls(l,s)
    ...        Ilx2 += ar*myIls
    ...    Ilx2 = I_x1*Ilx2
    ...    Ilx3 = intensity_moment(teff,logg,photband,coeffs)
    ...    check.append(abs(Ilx1-Ilx2)<0.1)
    ...    check.append(abs(Ilx1-Ilx3)<0.1)
    >>> np.all(np.array(check))
    True
    
    @parameter coeffs: LD coefficients
    @type coeffs: array
    @parameter ll: degree of the mode
    @type ll: float
    @param full_output: if True, returns intensity, coefficients and integrals separately
    @type full_output: boolean
    @return: intensity moment
    @rtype: float
    """
    #-- notation of Townsend 2002: coeffs in code are hat coeffs in the paper
    #   (for i=1..4 they are the same)
    #-- get the LD coefficients at the given temperature and logg
    a1x_,a2x_,a3x_,a4x_, I_x1 = coeffs
    a0x_ = 1 - a1x_ - a2x_ - a3x_ - a4x_
    limb_coeffs = np.array([a0x_,a1x_,a2x_,a3x_,a4x_])
    
    #-- compute intensity moment via helper coefficients
    int_moms = np.array([I_ls(ll,1 + r/2.) for r in range(0,5,1)])
    #int_moms = np.outer(int_moms,np.ones(1))
    I_lx = I_x1 * (limb_coeffs * int_moms).sum(axis=0)
    #-- return value depends on keyword
    if full_output:
        return I_x1,limb_coeffs,int_moms
    else:    
        return I_lx
    
def I_ls(ll,ss):
    """
    Limb darkening moments (Dziembowski 1977, Townsend 2002)
    
    Recursive implementation.
    
    >>> I_ls(0,0.5)
    0.6666666666666666
    >>> I_ls(1,0.5)
    0.4
    >>> I_ls(2,0.5)
    0.09523809523809523
    """
    if ll==0:
        return 1./(1.+ss)
    elif ll==1:
        return 1./(2.+ss)
    else:
        return (ss-ll+2)/(ss+ll-2+3.)*I_ls(ll-2,ss)        
    

def sphere_intensity(body,pbdep,red_kwargs={}):
    """
    Calculate total and projected intensity of an object with a a certain teff
    and logg and radius at a certain distance.
    
    This assumes spherical symmetry!
    
    Example: compute the total amount of power the sun deposits per unit area.
    This should equal the solar constant of around 1368 W/m2. First, we set a
    ParameterSet to mimick the solar values. We also need an C{lcdep}, because
    we need to have a limb darkening model.
    
    >>> ps = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True)
    >>> ps.add_constraint('{distance} = constants.au')
    >>> lc = parameters.ParameterSet(frame='phoebe',context='lcdep')
    >>> lc['ld_func'] = 'claret'
    >>> lc['ld_coeffs'] = 'tables/ikurucz93_z0.0_k2_ld.fits'
    >>> lc['passband'] = 'OPEN.BOL'
    
    Then we can compute the intensities:
    
    >>> total_intens,proj_intens = intensity_sphere(ps,lc)
    >>> conversions.convert('erg/s','Lsol',total_intens)
    0.99966261813786272
    >>> conversions.convert('erg/s/cm2','W/m2',proj_intens)
    1367.1055424026363
    
    Via this, we can also compute the reference magnitude of the absolute
    magnitude scale. The reference magnitude is the one from the sun, which
    equals 4.83:
    
    >>> ps.add_constraint('{distance} = 10*constants.pc')
    >>> total_intens,proj_intens = intensity_sphere(ps,lc)
    >>> c0 = -2.5*np.log10(proj_intens)-4.83
    
    @return: total luminosity (erg/s) and projected intensity (erg/s/cm2)
    @rtype: float,float
    """
    #-- retrieve parameters from the body. We set the distance to the star to
    #   a nominal 10 pc if it is not available or not derivable from the angular
    #   diameter
    teff = body.request_value('teff','K')
    logg = body.request_value('surfgrav','[cm/s2]')
    radius = body.request_value('radius','m')
    abun = body['abun']
    #vrad += gravitational_redshift(body_parameter_set)
    #-- retrieve the parameters of the data
    ld_func = pbdep['ld_func']
    atm = pbdep['atm']
    ld_coeffs = pbdep['ld_coeffs']
    passband = pbdep['passband']
    if not ld_func=='claret':
        logger.warning('analytical computation of sphere intensity with LD model %s not implemented yet'%(ld_func))
        return 1.,1.
    #-- retrieve the limbdarkening coefficients when they need to be looked up
    #   in a table: if the given coefficients are a string, assume it is a
    #   reference to a table. Otherwise, just use the coefficients.
    if isinstance(ld_coeffs,str):
        atm_kwargs = dict(atm=atm,ld_func=ld_func,teff=teff,logg=logg,abun=abun,ld_coeffs=ld_coeffs)
        atm = choose_ld_coeffs_table(atm, atm_kwargs=atm_kwargs,\
                                 red_kwargs=red_kwargs)
        ld_coeffs = np.array(interp_ld_coeffs(atm,passband,atm_kwargs=atm_kwargs,red_kwargs=red_kwargs))[:,0]
    #-- we compute projected and total intensity. We have to correct for solid
    #-- angle, radius of the star and distance to the star.
    theta1 = 2*np.pi*radius**2*4*np.pi
    theta2 = 2*np.pi*radius**2
    int_moment = intensity_moment(ld_coeffs,ll=0)
    #4*pi*radius**2*constants.sigma_cgs*teff**4,\
    return int_moment*theta1,\
           int_moment*theta2

#}

#{ Phoebe interface

def local_intensity(system, parset_pbdep, parset_isr={}, beaming_alg='full'):
    """
    Calculate local intensity.
    
    Attempt to generalize first version.
    
    This is the work horse for Phoebe concerning atmospheres.
    
    Small but perhaps important note: we do not take reddening into account
    for OPEN.BOL calculations, if the reddening is interstellar.
    
    Beaming options:
    
        - :envvar:`beaming_al='full'`: intensity and limb darkening coefficients
          are corrected for local velocity
        - :envvar:`beaming_al='local'`: local linear beaming coefficient is added
          to the mesh
        - :envvar:`beaming_al='simple': global linear beaming coefficients is added
          to the mesh
        - :envvar:`beaming_al='none'` or beaming parameter evaluates to False:
          beaming is not taken into account
    """
    # Get the arguments we need concerning normal emergent intensities (atm),
    # LD coefficients and the LD function.
    atm = parset_pbdep['atm']
    ld_coeffs = parset_pbdep['ld_coeffs']
    ld_func = parset_pbdep['ld_func']
    
    # Passband: if it's not available (e.g. in the "Star" parset), we need to
    # take the bolometric passband
    passband = parset_pbdep.get('passband', 'OPEN.BOL')
    
    # Doppler beaming: include it if there is such a keyword and it is turned on
    # and the algorithm is "full"
    include_vgamma = parset_pbdep.get('beaming', False)
    if not beaming_alg == 'full':
        include_vgamma = False
    
    # The reference we need to take to compute stuff (if not available, it's
    # bolometric)
    ref = parset_pbdep.get('ref', '__bol')
    
    # Radial velocity needs to be in km/s, while the natural units in the 
    # Universe are Rsol/d. If vrad needs to be computed, we'll also include
    # gravitational redshift
    #vrad = conversions.convert('Rsol/d','km/s',system.mesh['velo___bol_'][:,2])
    vgamma = Rsol_d_to_kms * system.mesh['velo___bol_'][:,2]
    if not include_vgamma or ref=='__bol':
        vgamma = 0.0
    
    # In the following we need to take care of a lot of possible inputs by the
    # user, and some possible simplifications for the calculations.
    #
    # First of all, it is possible that the whole star has the same local
    # parameters. In that case, we could compute the limb darkening coefficients
    # only for one surface element and then set all surface elements to be the
    # same.
    #
    # Basically, we need to compute intensities and limb darkening coefficients.
    # In principle, they could be derived from different sets of atmosphere
    # models, or, the LD coeffs could be set manually to be constant over the
    # whole star while the intensities are varied over the surface.
    
    # Let's see which dataset we need to take care of
    tag = 'ld_' + ref
    log_msg = "{:s}({:s}):".format(passband, tag[3:])
    
    # We need parameters on the passband and reddening in the form of a
    # dictionary. We may add too much information, since the function
    # "interp_ld_coeffs" only extracts those that are relevant.
    atm_kwargs = dict(parset_pbdep)
    
    # No reddening for bolometric fluxes!
    if ref == '__bol':
        red_kwargs = {}
        logger.info(("Not propagating interstellar reddening info for bolometric flux "
                    "(taking default from grid)"))
    else:
        red_kwargs = dict(parset_isr)
    
    # Extract parameters deciding the atmosphere, limb darkening coefficients
    # and limbdarkening law
    atm = parset_pbdep['atm']
    ldc = parset_pbdep['ld_coeffs']
    ld_func = parset_pbdep['ld_func']
    
    if not ldc == 'prsa':
        # 1. Easiest case: atm_file and ld_coeffs file are consistent
        if atm == ldc and atm in config.atm_props:
            
            # Find the possible interpolation parameters
            atm_kwargs = {key:system.mesh[key] for key in config.atm_props[atm]}
            fitmethod = config.fit_props.get(atm, 'equidist_r_leastsq')
            
            # Get the atmosphere file
            atm_file = choose_ld_coeffs_table(atm, atm_kwargs=atm_kwargs,
                                        red_kwargs=red_kwargs, vgamma=vgamma,
                                        ld_func=ld_func,
                                        fitmethod=fitmethod)
            
            coeffs = interp_ld_coeffs(atm_file, passband, atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs, vgamma=vgamma)
            
            # Fill in the LD coefficients, but put the intensities in the last
            # column
            for collnr, coefcoll in enumerate(coeffs[:-1]):
                system.mesh[tag][:, collnr] = coefcoll
            system.mesh[tag][:, -1] = coeffs[-1]
            
            log_msg += ' atm and ld_coeffs from table {}'.format(os.path.basename(atm_file))
        
        # 2. atm_file and ld_coeffs file are not the same
        elif atm in config.atm_props:
            
            # First the local normal emergent intensities.
            # Find the possible interpolation parameters
            atm_kwargs = {key:system.mesh[key] for key in config.atm_props[atm]}
            fitmethod = config.fit_props.get(atm, 'equidist_r_leastsq')
            
            # Find the interpolation file. (todo: Force a uniform ld in the future.)
            atm_file = choose_ld_coeffs_table(atm, atm_kwargs=atm_kwargs,
                                        red_kwargs=red_kwargs, vgamma=vgamma,
                                        ld_func=ld_func,
                                        fitmethod=fitmethod)
            
            # And interpolate the table
            coeffs_atm, header = interp_ld_coeffs(atm_file, passband, atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs, vgamma=vgamma, return_header=True)
            
            log_msg += ' atm from table {}'.format(os.path.basename(atm_file))
            
            # ld_coeffs can either be a file or a user-specified list of
            # coefficients. We'll first assume that we have a file, if that doesn't
            # work, we'll assume its a set of coefficients
            try:
                fitmethod_ldc = config.fit_props.get(ldc, 'equidist_r_leastsq')
                ldc_file = choose_ld_coeffs_table(ldc, atm_kwargs=atm_kwargs,
                                        red_kwargs=red_kwargs, vgamma=vgamma,
                                        ld_func=ld_func,
                                        fitmethod=fitmethod_ldc) 
            
                coeffs_ldc = interp_ld_coeffs(ldc_file, passband, atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs, vgamma=vgamma)[:-1]
                
                log_msg += ', ld_coeffs from table {}'.format(os.path.basename(ldc_file))
                
            # Otherwise it's a list or so, and we don't need to interpolate
            except TypeError:
                coeffs_ldc = ldc
                
                log_msg += ', manual ld_coeffs ({})'.format(ldc)
            
            # Find out what the luminosity is for every triangle with the original
            # limb-darkening function. Compute a correction factor to keep the
            # luminosity from the atm grid, but rescale it to match the ld_func
            # given. This is what happens when you give inconsistent atmosphere
            # and limbdarkening coefficients!
            input_ld = header['C__LD_FUNC']
            atm_disk_integral = globals()['disk_{}'.format(input_ld)](coeffs_atm[:-1])
            ldc_disk_integral = globals()['disk_{}'.format(ld_func)](coeffs_ldc)
            correction_factor = atm_disk_integral / ldc_disk_integral
            
            # Fill in the LD coefficients, but put the intensities in the last
            # column. Correct the intensities for the new limb darkening law
            for collnr, coefcoll in enumerate(coeffs_ldc):
                system.mesh[tag][:, collnr] = coefcoll
            system.mesh[tag][:, -1] = coeffs_atm[-1] * correction_factor
            
        # 3. True Blackbody we'll treat separately. There are no limits on the
        # effective temperature used. Note that black bodies are not sensitive to
        # logg (or anything else for that matter).
        elif atm == 'true_blackbody':
            
            if beaming_alg and beaming_alg != 'full':
                raise ValueError("With atm=true_blackbody you can only use beaming_alg='full' or beaming_alg='none'.")
            
            wave_ = np.logspace(1, 5, 10000)
            log_msg += (', intens via atm=true_blackbody (this is going to take '
                    'forever...)')
            
            # Seriously, don't try to do this for every triangle! Let's hope
            # uniform_pars is true...
            uniform_pars = False
            if uniform_pars:
                Imu_blackbody = sed.blackbody(wave_,
                                            atm_kwargs['teff'][0],
                                            vrad=vgamma)
                system.mesh[tag][:, -1] = sed.synthetic_flux(wave_*10,
                                            Imu_blackbody, [passband])[0]
            
            # What the frack? You must be kidding... OK, here we go, creating
            # black bodies and integrating them over the passband as we go...
            # Imagine doing this for several light curves... Djeez.
            else:
                for i,T in enumerate(atm_kwargs['teff']):
                    Imu_blackbody = sed.blackbody(wave_, T,
                                        vrad=vgamma[i])
                    system.mesh[tag][i,-1] = sed.synthetic_flux(wave_*10,
                                                    Imu_blackbody, [passband])[0]
        
        # 4. Wilson Devinney compatibility layer
        elif os.path.splitext(atm)[1] == '.dat':
            atm_kwargs = {key:system.mesh[key] for key in config.atm_props['wd']}
            print atm_kwargs
            system.mesh[tag][:, -1] = interp_ld_coeffs_wd(atm, passband,
                                         atm_kwargs=atm_kwargs,
                                         red_kwargs=red_kwargs, vgamma=vgamma)
        
        
        # 5. Escape route
        else:
            raise ValueError("atm and ld_coeffs not understood; did you register them?")
    
    # We're using the Prsa law here:
    else:
        log_msg += " (Prsa law and intensities)"
    
    # Overwrite intensities if passband gravity darkening is used (testing
    # feature)
    if 'pbgravb' in parset_pbdep:
        pbgravb = parset_pbdep['pbgravb']
        # we need a reference point for scaling the intensities, we choose the
        # point with maximum surface gravity (likely the pole)
        g = 10**system.mesh['logg']
        index = np.argmax(g)
        Imax = system.mesh[tag][index,-1]
        system.mesh[tag][index,-1] = (g/g[index])**pbgravb * Imax
    
    # 5. Take care of boosting option
    if beaming_alg == 'simple':
        alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma,
                                      interp_all=False)
        system.mesh['alpha_b_' + ref] = alpha_b
        log_msg += ' (simple boosting)'
    
    elif beaming_alg == 'local':
        alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma)
        system.mesh['alpha_b_' + ref] = alpha_b
        log_msg += ' (local boosting)'
        
    elif beaming_alg == 'global':
        alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma)
        system.mesh['alpha_b_' + ref] = alpha_b
        log_msg += ' (global boosting (currently still implemented as local))'
    
    elif beaming_alg == 'none' or not beaming_alg:
        log_msg += " (no boosting)"
    
    logger.info(log_msg)
    



def ld_intensity_prsa(system, parset_pbdep, parset_isr={}, beaming_alg='full'):
    """
    Calculate projected LD intensity with the Prsa law.
    
    Small but perhaps important note: we do not take reddening into account
    for OPEN.BOL calculations, if the reddening is interstellar.
    
    Beaming options:
    
        - :envvar:`beaming_al='full'`: intensity and limb darkening coefficients
          are corrected for local velocity
        - :envvar:`beaming_al='local'`: local linear beaming coefficient is added
          to the mesh
        - :envvar:`beaming_al='simple': global linear beaming coefficients is added
          to the mesh
        - :envvar:`beaming_al='none'` or beaming parameter evaluates to False:
          beaming is not taken into account
    """
    # Get the arguments we need concerning normal emergent intensities (atm),
    # LD coefficients and the LD function.
    atm = parset_pbdep['atm']
    ld_coeffs = parset_pbdep['ld_coeffs']
    ld_func = parset_pbdep['ld_func']
    
    # Passband: if it's not available (e.g. in the "Star" parset), we need to
    # take the bolometric passband
    passband = parset_pbdep.get('passband', 'OPEN.BOL')
    
    # Doppler beaming: include it if there is such a keyword and it is turned on
    # and the algorithm is "full"
    include_vgamma = parset_pbdep.get('beaming', False)
    if not beaming_alg == 'full':
        include_vgamma = False
    
    # The reference we need to take to compute stuff (if not available, it's
    # bolometric)
    ref = parset_pbdep.get('ref', '__bol')
    
    # Radial velocity needs to be in km/s, while the natural units in the 
    # Universe are Rsol/d. If vrad needs to be computed, we'll also include
    # gravitational redshift
    #vrad = conversions.convert('Rsol/d','km/s',system.mesh['velo___bol_'][:,2])
    vgamma = Rsol_d_to_kms * system.mesh['velo___bol_'][:,2]
    if not include_vgamma or ref=='__bol':
        vgamma = 0.0
    
    # Let's see which dataset we need to take care of
    tag = 'ld_' + ref
    log_msg = "{:s}({:s}):".format(passband, tag[3:])
    
    # We need parameters on the passband and reddening in the form of a
    # dictionary. We may add too much information, since the function
    # "interp_ld_coeffs" only extracts those that are relevant.
    atm_kwargs = dict(parset_pbdep)
    
    # No reddening for bolometric fluxes!
    if ref == '__bol':
        red_kwargs = {}
        logger.info(("Not propagating interstellar reddening info for bolometric flux "
                    "(taking default from grid)"))
    else:
        red_kwargs = dict(parset_isr)
    
    # Extract parameters deciding the atmosphere, limb darkening coefficients
    # and limbdarkening law
    atm = parset_pbdep['atm']
    ldc = parset_pbdep['ld_coeffs']
    ld_func = parset_pbdep['ld_func']
    
    # Find the possible interpolation parameters and add 'mu'
    atm_kwargs = {key:system.mesh[key] for key in config.atm_props[atm]}
    atm_kwargs['mu'] = system.mesh['mu']
    fitmethod = config.fit_props.get(atm, 'equidist_r_leastsq')
    
    # Get the atmosphere file
    atm_file = choose_ld_coeffs_table(atm, atm_kwargs=atm_kwargs,
                                red_kwargs=red_kwargs, vgamma=vgamma,
                                ld_func=ld_func,
                                fitmethod=fitmethod)
    
    coeffs = interp_ld_coeffs(atm_file, passband, atm_kwargs=atm_kwargs,
                                    red_kwargs=red_kwargs, vgamma=vgamma)
    
    # Fill in the LD coefficients, but put the intensities in the last
    # column
    for collnr, coefcoll in enumerate(coeffs[:-1]):
        system.mesh[tag][:, collnr] = coefcoll
    system.mesh[tag][:, -1] = coeffs[-1]
    
    log_msg += ' atm and ld_coeffs from table {}'.format(os.path.basename(atm_file))
    
    
    logger.info(log_msg)
    







    

def projected_velocity(system,los=[0,0,+1],method='numerical',ld_func='claret',ref=0):
    """
    Calculate mean local projected velocity.
    
    The sign of the Z-component is reversed to get negative rv for blueshifted
    objects!
    
    @param system: object to compute temperature of
    @type system: Body or derivative class
    @param los: line-of-sight vector. Best leave it at the default
    @type los: list of three floats
    @param method: flag denoting type of calculation: numerical or analytical approximation
    @type method: str ('numerical' or 'analytical')
    @param ld_func: limb-darkening model
    @type ld_func: str
    @param ref: ref of self's observation set to compute local intensities of
    @type ref: str
    """
    #-- get limb angles
    mus = system.mesh['mu']
    #-- To calculate the total projected intensity, we keep track of the
    #   partially visible triangles, and the totally visible triangles:
    keep = (mus>0) & (system.mesh['partial'] | system.mesh['visible'])
    
    mus = mus[keep]
    #-- negating the next array gives the partially visible things
    visible = system.mesh['visible'][keep]
            
    #-- compute intensity using the already calculated limb darkening coefficents
    logger.info('using limbdarkening law %s'%(ld_func))
    Imu = globals()['ld_'+ld_func](mus,system.mesh['ld_'+ref][keep].T)*system.mesh['ld_'+ref][keep,-1]
    
    proj_Imu = mus*Imu
    proj_Imu[-visible] /= 2.0
    proj_intens = system.mesh['size'][keep]*proj_Imu
    if not np.any(keep):
        proj_velo = np.nan
    elif np.all(proj_intens==0):
        proj_velo = 0.
    else:
        #proj_velo = np.average(-system.mesh['velo_'+ref+'_'][keep,2],weights=proj_intens)
        proj_velo = np.average(-system.mesh['velo___bol_'][keep,2],weights=proj_intens)
        logger.info("projected velocity with ld_func %s = %.3g Rsol/d"%(ld_func,proj_velo))
    
    return proj_velo

    
#}

#{ System administration

def register_atm_table(atm, force=False):
    """
    Register an atmosphere table.
    """
    # If atm is not a string, we can't register it, but we probably don't
    # need to either (must come from an automated tool)
    if not isinstance(atm, str):
        return None
    
    # If it's already registered, don't bother except if we want to
    # override existing results
    if not force and atm in config.atm_props:
        return None
    
    # if it's not a file, something went wrong
    if not os.path.isfile(atm):
        raise IOError("Atmosphere file {} does not exist so it cannot be used. Create the table or use a built-in value for any 'atm' and 'ld_coeffs' (one of {})".format(atm, ", ".join(config.atm_props.keys())))
    
    # Otherwise see which parameters need to be interpolated, and remember
    # those values
    pars = []
    # Read in which parameters need to be interpolated
    with pyfits.open(atm) as open_file:
        for key in open_file[0].header.keys():
            if key[:6] == 'C__AI_':
                pars.append(open_file[0].header[key])
    config.atm_props[atm] = tuple(pars)
    logger.info("Registered atm table {} to interpolate in {}".format(atm, pars))

def get_paths():
    """
    Return the paths where the LD files are located.
    
    @return: path to grid with LD coefficients, path to grid with specific
     intensities
    @rtype: str, str
    """
    return os.path.join(basedir, 'tables','ld_coeffs'),\
           os.path.join(basedir, 'tables','spec_intens')

def get_info(path):
    with pyfits.open(path) as ff:
        passbands = [ext.header['extname'] for ext in ff[1:]]
        fields = []
        for key in ff[0].header:
            if key[:5] == 'C__AI':
                fields.append(ff[0].header[key].split('C__AI_')[-1])
        
        dimensions = []
        for field in fields:
            dimensions.append(np.unique(ff[1].data.field(field)))
    
    print("Atmosphere file: {}".format(path))
    print("Passbands: {:d} ({})".format(len(passbands), ", ".join(passbands)))
    print("Interpolatable quantities: {}".format(", ".join(fields)))
    print("Dimensions:")
    for field, dim in zip(fields, dimensions):
        print "{}: {}".format(field, dim)
        
            

def add_file(filename):
    """
    Add an atmosphere file to the right installation directory.
    """
    # What kind of file is it?
    with pyfits.open(filename) as ff:
        has_key = 'FILETYPE' in ff[0].header.keys()
        if has_key and ff[0].header['FILETYPE'] == 'LDCOEFFS':
            index = 0
        else:
            index = 1
    
    # Where do we need to store it?
    path = get_paths()[index]
    
    # Do we have access to that directory? If so, copy it. Else, produce the
    # copy statement
    destination = os.path.join(path, os.path.basename(filename))
    if os.access(path):
        shutil.copy(filename, destination)
    else:
        logger.warning("No permission to write {}".format(destination))
        print("sudo cp {} {}".format(filename, destination))


def download_atm(atm=None):
    
    destin_folder = get_paths()[0]
        
    # Perhaps we need to be sudo?
    print("Copying to destination folder {}".format(destin_folder))
    if not os.access(destin_folder, os.W_OK):
        raise IOError(("User has no write priviliges in destination folder, run sudo python"
                        "before calling this function"))
        #output = subprocess.check_call(['sudo']+sys.argv)
    
    if atm is None:
        source = 'http://www.phoebe-project.org/2.0/docs/_downloads/ldcoeffs.tar.gz'
        destin = os.path.join(destin_folder, 'ldcoeffs.tar.gz')
        try:
            urllib.urlretrieve(source, destin)
            print("Downloaded tar archive from phoebe-project.org")
        except IOError:
            raise IOError(("Failed to download atmosphere file {} to {} (you probably "
                           "need to create one yourself starting from the specific "
                           "intensities)").format(source, destin))
            
    
        tar = tarfile.open(destin)
        members = tar.getmembers()
        for member in members:
            print("Extracting {}...".format(member))
        tar.extractall(path=destin_folder)
        tar.close()
    
    else:
        source = 'http://www.phoebe-project.org/2.0/docs/_downloads/{}'.format(atm)
        destin = os.path.join(destin_folder, atm)

        try:
            urllib.urlretrieve(source, destin)
            print("Downloaded tar archive from phoebe-project.org")
        except IOError:
            raise IOError(("Failed to download atmosphere file {} (you probably "
                           "need to create on yourself starting from the specific "
                           "intensities)").format(atm))
    


#}


    
if __name__=="__main__":
    import doctest
    import matplotlib.pyplot as plt
    fails,tests = doctest.testmod()
    if not fails:
        print(("All {0} tests succeeded".format(tests)))
    else:
        print(("{0}/{1} tests failed".format(fails,tests)))
    
