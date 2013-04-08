"""
Various utilities relevant for atmospheres.
"""
import logging
import numpy as np
from numpy import pi,sin,cos,sqrt
from scipy.signal import fftconvolve
from scipy.integrate import quad
from phoebe.units import constants
from phoebe.units import conversions

logger = logging.getLogger('ATMO.TOOLS')

def gravitational_redshift(the_system):
    """
    Compute the gravitional redshift of an object
    
    @param system: object to compute temperature of
    @type system: Body
    """
    #-- for systems that we cannot compute the gravitional redshift, there will
    #   effectively be none: therefor, we assume mass and radius such that the
    #   gravitional redshift is zero
    M = 0
    R = 1.
    #-- try to get information of the mass and radius of the system
    if hasattr(the_system,'params') and 'body' in the_system.params.keys() \
          and 'mass' in the_system.params['body']:
        M = the_system.params['body'].request_value('mass','kg')
        R = the_system.params['body'].request_value('radius','m')
    elif hasattr(the_system,'keys'):
        M = the_system.request_value('mass','kg')
        R = the_system.request_value('radius','m')
    if hasattr(M,'__iter__') or M==0.:
        M = 0.
        R = 1.
        logger.warning('unable to calculate gravitational redshift')
    #-- compute gravitaional redshift
    rv_grav = constants.GG*M/R/constants.cc/1000.
    logger.info('gravitational redshift = %.3f km/s'%(rv_grav))
    #-- yes, it's that easy
    return rv_grav


def doppler_shift(wave,vrad,vrad_units='km/s',flux=None):
    """
    Shift a spectrum with towards the red or blue side with some radial velocity.
    
    You can give units with the extra keywords C{vrad_units} (units of
    wavelengths are not important). The shifted wavelengths will be in the same
    units as the input wave array.
    
    If units are not supplied, the radial velocity is assumed to be in km/s.
    
    If you want to apply a barycentric correction, you'd probably want to
    reverse the sign!
    
    Example usage: shift a spectrum to the blue ('left') with 20 km/s.
    
    >>> wave = np.linspace(3000,8000,1000)
    >>> wave_shift1 = doppler_shift(wave,20.)
    >>> wave_shift2 = doppler_shift(wave,20000.,vrad_units='m/s')
    >>> print(wave_shift1[0],wave_shift1[-1])
    (3000.200138457119, 8000.5337025523177)
    >>> print(wave_shift2[0],wave_shift2[-1])
    (3000.200138457119, 8000.5337025523177)
    
    @param wave: wavelength array
    @type wave: ndarray
    @param vrad: radial velocity (negative shifts towards blue, positive towards red)
    @type vrad: float (units: km/s) or tuple (float,'units')
    @param vrad_units: units of radial velocity (default: km/s)
    @type vrad_units: str (interpretable for C{units.conversions.convert})
    @return: shifted wavelength array
    @rtype: ndarray
    """ 
    cc = constants.cc
    cc = conversions.convert('m/s',vrad_units,cc)
    wave_out = wave * (1+vrad/cc)
    if flux is not None:
        flux_out = np.interp(wave,wave_out,flux)
        return flux_out
    else:
        return wave_out


def beaming():
    pass


def vmacro_kernel(dlam,Ar,At,Zr,Zt):
    r"""
    Macroturbulent velocity kernel.
    
    It is defined as in _[Gray2005]:
    
    .. math::
    
        M(\Delta\lambda) = \frac{2A_R\Delta\lambda}{\sqrt{\pi}\zeta_R^2}\int_0^{\zeta_R/\Delta\lambda}e^{-1/u^2}du 
        
         & + \frac{2A_T\Delta\lambda}{\sqrt{\pi}\zeta_T^2}\int_0^{\zeta_T/\Delta\lambda}e^{-1/u^2}du 
        
        
    But only applies to spherical stars with linear limbdarkening.
    
    """
    dlam[dlam==0] = 1e-8
    if Zr!=Zt:
        return np.array([(2*Ar*idlam/(np.sqrt(np.pi)*Zr**2) * quad(lambda u: np.exp(-1/u**2),0,Zr/idlam)[0] + \
                          2*At*idlam/(np.sqrt(np.pi)*Zt**2) * quad(lambda u: np.exp(-1/u**2),0,Zt/idlam)[0]) 
                             for idlam in dlam])
    else:
        return np.array([(2*Ar*idlam/(np.sqrt(np.pi)*Zr**2) + 2*At*idlam/(np.sqrt(np.pi)*Zt**2))\
                           * quad(lambda u: np.exp(-1/u**2),0,Zr/idlam)[0]\
                             for idlam in dlam])


def rotational_broadening(wave_spec,flux_spec,vrot,vmac=0.,fwhm=0.25,epsilon=0.6,
                         chard=None,stepr=0,stepi=0,alam0=None,alam1=None,
                         irel=0,cont=None,method='fortran'):
    """
    Apply rotational broadening to a spectrum assuming a linear limb darkening
    law.
    
    Limb darkening law is linear, default value is epsilon=0.6
    
    Possibility to normalize as well by giving continuum in 'cont' parameter.
        
    **Parameters for rotational convolution**

    C{VROT}: v sin i (in km/s):
    
        -  if ``VROT=0`` - rotational convolution is 
        
                 - either not calculated,
                 - or, if simultaneously FWHM is rather large
                   (:math:`v_\mathrm{rot}\lambda/c < \mathrm{FWHM}/20.`),
                   :math:`v_\mathrm{rot}` is set to  :math:`\mathrm{FWHM}/20\cdot c/\lambda`;
        
        -  if ``VROT >0`` but the previous condition b) applies, the
           value of VROT is changed as  in the previous case
        
        -  if ``VROT<0`` - the value of abs(VROT) is used regardless of
           how small compared to FWHM it is
     
    C{CHARD}: characteristic scale of the variations of unconvolved stellar
    spectrum (basically, characteristic distance between two neighbouring
    wavelength points) - in A:
     
        - if =0 - program sets up default (0.01 A)
        
    C{STEPR}: wavelength step for evaluation rotational convolution;
     
        - if =0, the program sets up default (the wavelength
          interval corresponding to the rotational velocity
          devided by 3.)                           
    
        - if <0, convolved spectrum calculated on the original
          (detailed) SYNSPEC wavelength mesh


    **Parameters for instrumental convolution**

    C{FWHM}: WARNING: this is not the full width at half maximum for Gaussian
    instrumental profile, but the sigma (FWHM = 2.3548 sigma).
    
    C{STEPI}: wavelength step for evaluating instrumental convolution
          
          - if STEPI=0, the program sets up default (FWHM/10.)
          
          - if STEPI<0, convolved spectrum calculated with the previous
            wavelength mesh:
            either the original (SYNSPEC) one if vrot=0,
            or the one used in rotational convolution (vrot > 0)

    **Parameters for macroturbulent convolution**
    
    C{vmac}: macroturbulent velocity.
    
    **Wavelength interval and normalization of spectra**
    
    C{ALAM0}: initial wavelength
    C{ALAM1}: final wavelength
    C{IREL}: for =1 relative spectrum, =0 absolute spectrum
    
    @return: wavelength,flux
    @rtype: array, array
    """
    logger.info("Rot.broad with vrot=%.3f (epsilon=%.2f)"%(vrot,epsilon))
    #-- first a wavelength Gaussian convolution:
    if fwhm>0:
        fwhm /= 2.3548
        #-- make sure it's equidistant
        wave_ = np.linspace(wave_spec[0],wave_spec[-1],len(wave_spec))
        flux_ = np.interp(wave_,wave_spec,flux_spec)
        dwave = wave_[1]-wave_[0]
        n = int(2*4*fwhm/dwave)
        wave_k = np.arange(n)*dwave
        wave_k-= wave_k[-1]/2.
        kernel = np.exp(- (wave_k)**2/(2*fwhm**2))
        kernel /= sum(kernel)
        flux_conv = fftconvolve(1-flux_,kernel,mode='same')
        #-- this little tweak is necessary to keep the profiles at the right
        #   location
        if n%2==1:
            flux_spec = np.interp(wave_spec,wave_,1-flux_conv)
        else:
            print("warning test offset of profile")
            flux_spec = np.interp(wave_spec+dwave/2,wave_,1-flux_conv)
    #-- macroturbulent profile
    if vmac>0:
        vmac = vmac/(constants.cc*1e-3)*(wave_spec[0]+wave_spec[-1])/2.0
        #-- make sure it's equidistant
        wave_ = np.linspace(wave_spec[0],wave_spec[-1],len(wave_spec))
        flux_ = np.interp(wave_,wave_spec,flux_spec)
        dwave = wave_[1]-wave_[0]
        n = int(6*vmac/dwave/5)
        wave_k = np.arange(n)*dwave
        wave_k-= wave_k[-1]/2.
        kernel = vmacro_kernel(wave_k,1.,1.,vmac,vmac)
        kernel /= sum(kernel)
        flux_conv = fftconvolve(1-flux_,kernel,mode='same')
        if n%2==1:
            flux_spec = np.interp(wave_spec,wave_,1-flux_conv)
        else:
            print("warning test offset of profile")
            flux_spec = np.interp(wave_spec+dwave/2,wave_,1-flux_conv)
    if vrot>0:    
        #-- convert wavelength array into velocity space, this is easier
        #   we also need to make it equidistant!
        wave_ = np.log(wave_spec)
        velo_ = np.linspace(wave_[0],wave_[-1],len(wave_))
        flux_ = np.interp(velo_,wave_,flux_spec)
        dvelo = velo_[1]-velo_[0]
        vrot = vrot/(constants.cc*1e-3)
        #-- compute the convolution kernel and normalise it
        n = int(2*vrot/dvelo)
        velo_k = np.arange(n)*dvelo
        velo_k -= velo_k[-1]/2.
        y = 1 - (velo_k/vrot)**2 # transformation of velocity
        G = (2*(1-epsilon)*sqrt(y)+pi*epsilon/2.*y)/(pi*vrot*(1-epsilon/3.0))  # the kernel
        G /= G.sum()
        #-- convolve the flux with the kernel
        flux_conv = fftconvolve(1-flux_,G,mode='same')
        if n%2==1:
            velo_ = np.arange(len(flux_conv))*dvelo+velo_[0]
        else:
            velo_ = np.arange(len(flux_conv))*dvelo+velo_[0]-dvelo/2.
        wave_conv = np.exp(velo_)
        return wave_conv,1-flux_conv
        
    
    return wave_spec,flux_spec