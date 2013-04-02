"""
Compute synthetic fluxes.
"""

import logging
import numpy as np
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.atmospheres import passbands as pbs
from phoebe.atmospheres import tools

logger = logging.getLogger('ATM.SED')

def blackbody(wl,T,vrad=0):
    r"""
    Definition of black body curve.
    
    The black body is generally defined as
    
    .. math::
    
        I(\lambda)_\mathrm{SI} = \frac{2 h c^2 }{\lambda^5\left(\exp{\frac{hc}{k_BT\lambda}}-1\right)}
        
    With :math:`\lambda` the wavelength in meter, :math:`h` Planck's constant,
    :math:`c` the velocity of light, :math:`k_B` Boltzmann's constant, and
    :math:`I` the flux density per sterradian :math:`\mathrm{W} \mathrm{m}^{-3}\mathrm{sr}^{-1}`.
    
    The output, however, is given in :math:`\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\mathrm{\AA}^{-1} \mathrm{sr}^{-1}`,
    the unit commonly used in optical photometry. The conversion to SI is:
    
    .. math::
    
        I(\lambda)_\mathrm{SI} = I(\lambda)_\mathrm{output} \times 10^7
    
    and to CGS (:math:`\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-3}\mathrm{sr}^{-1}`):
    
    .. math::
    
        I(\lambda)_\mathrm{CGS} = I(\lambda)_\mathrm{output} \times 10^8    
    
    
    To get them into the same units as the Kurucz disc-integrated SEDs, multiply
    by :math:`\sqrt{2\pi}`:
    
    .. math::
    
        I(\lambda)_\mathrm{Kurucz} =  I(\lambda)_\mathrm{output} \times\sqrt{2\pi}
        
    Doppler beaming can be taken into account by giving a radial velocity of 
    star star. In that case:
    
    .. math::
    
        I(\lambda)_\mathrm{output,DB} = I(\lambda)_\mathrm{output} + 5 \frac{v_\mathrm{rad}}{c}I(\lambda)'_\mathrm{output}
    
    with :math:`I(\lambda)'_\mathrm{output}` the doppler shifted intensity.
    
    @param wl: wavelength (nm)
    @type wl: array
    @param T: temperature (K)
    @type T: float
    @param vrad: radial velocity (km/s), :math:`v_\mathrm{rad}<0` means the spectrum will be shifted towards the blue (object coming towards us)
    @type vrad: float
    @return: intensity per sterradian :math:`\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\mathrm{\AA}^{-1} \mathrm{sr}^{-1}`
    @rtype: array
    """
    wl = wl*1e-9
    #-- now make the appropriate black body
    factor = 2.0 * constants.hh * constants.cc**2
    expont = constants.hh*constants.cc / (constants.kB*T)
    I = factor / wl**5. * 1. / (np.exp(expont/wl) - 1.)
    #from SI to erg/s/cm2/AA:
    if vrad!=0:
        I_ = tools.doppler_shift(wl,-vrad,flux=I,vrad_units='km/s')
        I = I_ + 5.*vrad/constants.cc*1000*I
    return I*1e-7

def synthetic_flux(wave,flux,passbands,units=None):
    """
    Extract flux measurements from a synthetic SED (Fnu or Flambda).
    
    The fluxes below 4micron are calculated assuming PHOTON-counting detectors
    (e.g. CCDs).
    
    Flam = int(P_lam * f_lam * lam, dlam) / int(P_lam * lam, dlam)
    
    When otherwise specified, we assume ENERGY-counting detectors (e.g. bolometers)
    
    Flam = int(P_lam * f_lam, dlam) / int(P_lam, dlam)
    
    Where P_lam is the total system dimensionless sensitivity function, which
    is normalised so that the maximum equals 1. Also, f_lam is the SED of the
    object, in units of energy per time per unit area per wavelength.
    
    The PHOTON-counting part of this routine has been thoroughly checked with
    respect to johnson UBV, geneva and stromgren filters, and only gives offsets
    with respect to the Kurucz integrated files (.geneva and stuff on his websites). These could be
    due to different normalisation.
    
    You can also readily integrate in Fnu instead of Flambda by suppling a list
    of strings to 'units'. This should have equal length of passbands, and
    should contain the strings 'flambda' and 'fnu' corresponding to each filter.
    In that case, the above formulas reduce to
    
    Fnu = int(P_nu * f_nu / nu, dnu) / int(P_nu / nu, dnu)
    
    and 
    
    Fnu = int(P_nu * f_nu, dnu) / int(P_nu, dnu)
    
    Small note of caution: P_nu is not equal to P_lam according to
    Maiz-Apellaniz, he states that P_lam = P_nu/lambda. But in the definition
    we use above here, it *is* the same!
    
    The model fluxes should B{always} be given in Flambda (erg/s/cm2/AA). The
    program will convert them to Fnu where needed.
    
    The output is a list of numbers, equal in length to the 'passband' inputs.
    The units of the output are erg/s/cm2/AA where Flambda was given, and
    erg/s/cm2/Hz where Fnu was given.
    
    The difference is only marginal for 'blue' bands. For example, integrating
    2MASS in Flambda or Fnu is only different below the 1.1% level:
    
    >>> wave,flux = get_table(teff=10000,logg=4.0)
    >>> energys = synthetic_flux(wave,flux,['2MASS.J','2MASS.J'],units=['flambda','fnu'])
    >>> e0_conv = conversions.convert('erg/s/cm2/AA','erg/s/cm2/Hz',energys[0],passband='2MASS.J')
    >>> np.abs(energys[1]-e0_conv)/energys[1]<0.012
    True
    
    But this is not the case for IRAS.F12:
    
    >>> energys = synthetic_flux(wave,flux,['IRAS.F12','IRAS.F12'],units=['flambda','fnu'])
    >>> e0_conv = conversions.convert('erg/s/cm2/AA','erg/s/cm2/Hz',energys[0],passband='IRAS.F12')
    >>> np.abs(energys[1]-e0_conv)/energys[1]>0.1
    True
    
    If you have a spectrum in micron vs Jy and want to calculate the synthetic
    fluxes in Jy, a little bit more work is needed to get everything in the
    right units. In the following example, we first generate a constant flux
    spectrum in micron and Jy. Then, we convert flux to erg/s/cm2/AA using the
    wavelengths (this is no approximation) and convert wavelength to angstrom.
    Next, we compute the synthetic fluxes in the IRAS band in Fnu, and finally
    convert the outcome (in erg/s/cm2/Hz) to Jansky.
    
    >>> wave,flux = np.linspace(0.1,200,10000),np.ones(10000)
    >>> flam = conversions.convert('Jy','erg/s/cm2/AA',flux,wave=(wave,'micron'))
    >>> lam = conversions.convert('micron','AA',wave)
    >>> energys = synthetic_flux(lam,flam,['IRAS.F12','IRAS.F25','IRAS.F60','IRAS.F100'],units=['Fnu','Fnu','Fnu','Fnu'])
    >>> energys = conversions.convert('erg/s/cm2/Hz','Jy',energys)
    
    You are responsible yourself for having a response curve covering the
    model fluxes!
    
    Now. let's put this all in practice in a more elaborate example: we want
    to check if the effective wavelength is well defined. To do that we will:
        
        1. construct a model (black body)
        2. make our own weird, double-shaped filter (CCD-type and BOL-type detector)
        3. compute fluxes in Flambda, and convert to Fnu via the effective wavelength
        4. compute fluxes in Fnu, and compare with step 3.
        
    In an ideal world, the outcome of step (3) and (4) must be equal:
    
    Step (1): We construct a black body model.
    
    
    WARNING: OPEN.BOL only works in Flambda for now.
    
    See e.g. Maiz-Apellaniz, 2006.
    
    @param wave: model wavelengths (angstrom)
    @type wave: ndarray
    @param flux: model fluxes (erg/s/cm2/AA)
    @type flux: ndarray
    @param passbands: list of photometric passbands
    @type passbands: list of str
    @param units: list containing Flambda or Fnu flag (defaults to all Flambda)
    @type units: list of strings or str
    @return: model fluxes (erg/s/cm2/AA or erg/s/cm2/Hz)
    @rtype: ndarray
    """    
    if isinstance(units,str):
        units = [units]*len(passbands)
    energys = np.zeros(len(passbands))
    
    #-- only keep relevant information on filters:
    filter_info = pbs.get_info()
    keep = np.searchsorted(filter_info['passband'],passbands)
    filter_info = filter_info[keep]
    
    for i,passband in enumerate(passbands):
        #if filters.is_color
        waver,transr = pbs.get_response(passband)
        #-- make wavelength range a bit bigger, otherwise F25 from IRAS has only
        #   one Kurucz model point in its wavelength range... this is a bit
        #   'ad hoc' but seems to work.
        region = ((waver[0]-0.4*waver[0])<=wave) & (wave<=(2*waver[-1]))
        #-- if we're working in infrared (>4e4A) and the model is not of high
        #   enough resolution (100000 points over wavelength range), interpolate
        #   the model in logscale on to a denser grid (in logscale!)
        if filter_info['eff_wave'][i]>=4e4 and sum(region)<1e5 and sum(region)>1:
            logger.debug('%10s: Interpolating model to integrate over response curve'%(passband))
            wave_ = np.logspace(np.log10(wave[region][0]),np.log10(wave[region][-1]),1e5)
            flux_ = 10**np.interp(np.log10(wave_),np.log10(wave[region]),np.log10(flux[region]),)
        else:
            wave_ = wave[region]
            flux_ = flux[region]
        if not len(wave_):
            energys[i] = np.nan
            continue
        #-- perhaps the entire response curve falls in between model points (happends with
        #   narrowband UV filters), or there's very few model points covering it
        if (np.searchsorted(wave_,waver[-1])-np.searchsorted(wave_,waver[0]))<5:
            wave__ = np.sort(np.hstack([wave_,waver]))
            flux_ = np.interp(wave__,wave_,flux_)
            wave_ = wave__
        #-- interpolate response curve onto model grid
        transr = np.interp(wave_,waver,transr,left=0,right=0)
        
        #-- integrated flux: different for bolometers and CCDs
        #-- WE WORK IN FLAMBDA
        if units is None or ((units is not None) and (units[i].upper()=='FLAMBDA')):
            if passband=='OPEN.BOL':
                energys[i] = np.trapz(flux_,x=wave_)
            elif filter_info['type'][i]=='BOL':
                energys[i] = np.trapz(flux_*transr,x=wave_)/np.trapz(transr,x=wave_)
            elif filter_info['type'][i]=='CCD':
                energys[i] = np.trapz(flux_*transr*wave_,x=wave_)/np.trapz(transr*wave_,x=wave_)
        
        #-- we work in FNU
        elif units[i].upper()=='FNU':
            #-- convert wavelengths to frequency, Flambda to Fnu
            freq_ = conversions.convert('AA','Hz',wave_)
            flux_f = conversions.convert('erg/s/cm2/AA','erg/s/cm2/Hz',flux_,wave=(wave_,'AA'))
            #-- sort again!
            sa = np.argsort(freq_)
            transr = transr[sa]
            freq_ = freq_[sa]
            flux_f = flux_f[sa]
            if filter_info['type'][i]=='BOL':
                energys[i] = np.trapz(flux_f*transr,x=freq_)/np.trapz(transr,x=freq_)
            elif filter_info['type'][i]=='CCD':
                energys[i] = np.trapz(flux_f*transr/freq_,x=wave_)/np.trapz(transr/freq_,x=wave_)
        else:
            raise ValueError,'units %s not understood'%(units)
    
    #-- that's it!
    return energys

