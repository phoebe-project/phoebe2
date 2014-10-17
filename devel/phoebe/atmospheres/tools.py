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
from phoebe.utils import pergrams

logger = logging.getLogger('ATMO.TOOLS')

def gravitational_redshift(the_system):
    r"""
    Compute the gravitional redshift of an object
    
    .. math::
    
        RV_\mathrm{grav} = \frac{GM}{Rc}
    
    @param the_system: object
    @type the_system: Body
    """
    # For systems for which we cannot compute the gravitional redshift, there
    # will effectively be none: therefore, we assume mass and radius such that
    # the gravitional redshift is zero
    M = 0
    R = 1.
    if hasattr(the_system, 'bodies'):
        bodies = the_system.get_bodies()
    else:
        bodies = [the_system]
    
    rv_gravs = []
    for body in bodies:
        M = body.get_mass() * constants.Msol
        R = np.sqrt(body.mesh['_o_center'][:,0]**2 + \
                body.mesh['_o_center'][:,1]**2 + body.mesh['_o_center'][:,2]**2)
        R = R*constants.Rsol
        rv_grav = constants.GG*M/R/constants.cc
        rv_grav = conversions.convert('m/s', 'km/s', rv_grav)
        rv_gravs.append(rv_grav*np.ones(len(body.mesh)))
    rv_gravs = np.hstack(rv_gravs)
        
    #-- yes, it's that easy
    return rv_gravs


def doppler_shift(wave,vrad,vrad_units='km/s',flux=None):
    """
    Shift a spectrum with towards the red or blue side with some radial velocity.
    
    You can give units with the extra keywords C{vrad_units} (units of
    wavelengths are not important). The shifted wavelengths will be in the same
    units as the input wave array.
    
    If units are not supplied, the radial velocity is assumed to be in km/s.
    
    If you want to apply a barycentric correction, you'd probably want to
    reverse the sign!
    
    Example usage: shift a spectrum to the red ('right') with 20 km/s.
    
    >>> wave = np.linspace(3000,8000,1000)
    >>> wave_shift1 = doppler_shift(wave,20.)
    >>> wave_shift2 = doppler_shift(wave,20000.,vrad_units='m/s')
    >>> print(wave_shift1[0],wave_shift1[-1])
    (3000.200138457119, 8000.5337025523177)
    >>> print(wave_shift2[0],wave_shift2[-1])
    (3000.200138457119, 8000.5337025523177)
    
    Or with a complete line profile:
    
    >>> wave = np.linspace(3000,8000,20)
    >>> flux = 1.0 - 0.5*np.exp( - (wave-5500)**2/(2*500**2))
    >>> wave0 = doppler_shift(wave,5000.,)
    >>> flux1 = doppler_shift(wave,5000.,flux=flux)
    
    Notice how, with the second ('red') version, the original wavelength
    array can be re-used, while with the first ('blue') version, the original flux
    array can be re-used. Keep in mind that when the fluxes are shifted, they
    are linearly interpolated onto the shifted wavelength array, so the shift
    is not exact.
    
    >>> p = plt.figure()
    >>> p = plt.plot(wave,flux,'ko-')
    >>> p = plt.plot(wave0,flux,'bx-',lw=2,mew=2,ms=10)
    >>> p = plt.plot(wave,flux1,'r+--',lw=2,mew=2,ms=10)
    
    .. image:: images/atmospheres_tools_doppler_shift.png
       :scale: 75 %
    
    @param wave: wavelength array
    @type wave: ndarray
    @param vrad: radial velocity (negative shifts towards blue, positive towards red)
    @type vrad: float (units: km/s) or tuple (float,'units')
    @param vrad_units: units of radial velocity (default: km/s)
    @type vrad_units: str (interpretable for C{units.conversions.convert})
    @return: shifted wavelength array
    @rtype: ndarray
    """ 
    if vrad_units == 'km/s':
        cc = constants.cc/1000.
    else:
        cc = conversions.convert('m/s',vrad_units,constants.cc)
    
    
    wave_out = wave * (1+vrad/cc)
    
    if flux is not None:
        flux_out = np.interp(wave,wave_out,flux)
        return flux_out
    else:
        return wave_out


def boosting():
    pass



def broadening_instrumental(wave, flux, width=0.25, width_type='fwhm',
                            return_kernel=False):
    r"""
    Apply instrumental broadening to a spectrum.
    
    The instrumental broadening kernel is a simple Gaussian kernel:
    
    .. math::
    
        K_\mathrm{instr} = \exp\left(-\frac{(\lambda-\lambda_0)^2}{2\sigma^2}\right)
    
    where :math:`\lambda_0` is the center of the wavelength array
    :envvar:`wave`.
    
    **Example**: Construct a simple Gaussian line profile and convolve with an instrumental profile sigma=0.5AA
        
    >>> sigma = 0.5
    >>> wave = np.linspace(3995, 4005, 1001)
    >>> flux = 1.0 - 0.5*np.exp(-(wave-4000)**2/(2*sigma**2))
    
    Convolve it with an instrumental :math:`\sigma=0.5\AA`:

    >>> sigma_in = 0.5
    >>> flux_broad = tools.broadening_instrumental(wave, flux, sigma_in, width_type='sigma')
    >>> flux_broad, (wave_kernel, kernel) = tools.broadening_instrumental(wave, flux, sigma_in, width_type='sigma', return_kernel=True)
    
    ::
    
        plt.figure()
        plt.plot(wave, flux, 'k-')
        plt.plot(wave, flux_broad, 'r-', lw=2)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Normalized flux")

        plt.figure()
        plt.plot(wave_kernel, kernel, 'r-', lw=2)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Normalized flux")
    
    .. +----------------------------------------------------------------------+----------------------------------------------------------------------+
    .. | .. image:: ../../images/api/spectra/tools/broaden_instrumental01.png | .. image:: ../../images/api/spectra/tools/broaden_instrumental02.png |
    .. |   :scale: 50 %                                                       |   :scale: 50 %                                                       |
    .. +----------------------------------------------------------------------+----------------------------------------------------------------------+
        
    :parameter wave: Wavelength of the spectrum
    :type wave: array
    :parameter flux: Flux of the spectrum
    :type flux: array
    :parameter width: width of instrumental profile, in units of :envvar:`wave`
    :type width: float
    :parameter width_type: type of width
    :type width_type: str, one of ``fwhm`` or ``sigma``
    :parameter return_kernel: return kernel
    :type return_kernel: bool
    :return: broadened flux [, (wavelength, kernel)]
    :rtype: array [,(array, array)]
    """
    
    # If there is no broadening to apply, don't bother
    if width == 0:
        return flux
    
    # Convert user input width type to sigma (standard devation)
    width_type = width_type.lower()
    if width_type == 'fwhm':
        sigma = width / 2.3548
    elif width_type == 'sigma':
        sigma = width
    else:
        raise ValueError(("Unrecognised width_type='{}' (must be one of 'fwhm'"
                          "or 'sigma')").format(width_type))
    
    # Make sure the wavelength range is equidistant before applying the
    # convolution
    delta_wave = np.diff(wave).min()
    range_wave = wave.ptp()
    n_wave = int(range_wave/delta_wave)+1
    wave_ = np.linspace(wave[0], wave[-1], n_wave)
    flux_ = np.interp(wave_, wave, flux)
    dwave = wave_[1]-wave_[0]
    n_kernel = int(2*4*sigma/dwave)
    
    # The kernel might be of too low resolution, or the the wavelength range
    # might be too narrow. In both cases, raise an appropriate error
    if n_kernel == 0:
        raise ValueError(("Spectrum resolution too low for "
                          "instrumental broadening (delta_wave={}, "
                          "width={}").format(delta_wave, width))
    elif n_kernel > n_wave:
        raise ValueError(("Spectrum range too narrow for "
                          "instrumental broadening"))
    
    # Construct the broadening kernel
    wave_k = np.arange(n_kernel)*dwave
    wave_k -= wave_k[-1]/2.
    kernel = np.exp(- (wave_k)**2/(2*sigma**2))
    kernel /= sum(kernel)
    
    # Convolve the flux with the kernel
    flux_conv = fftconvolve(1-flux_, kernel, mode='same')
    
    # And interpolate the results back on to the original wavelength array, 
    # taking care of even vs. odd-length kernels
    if n_kernel % 2 == 1:
        offset = 0.0
    else:
        offset = dwave / 2.0
    flux = np.interp(wave+offset, wave_, 1-flux_conv, left=1, right=1)
    
    # Return the results.
    if return_kernel:
        return flux, (wave_k, kernel)
    else:
        return flux


def broadening_rotational(wave, flux, vrot, epsilon=0.6, return_kernel=False):
    r"""
    Apply rotational broadening to a spectrum assuming a linear limb darkening
    law.
    
    The adopted limb darkening law is the linear one, parameterize by the linear
    limb darkening parameter :envvar:`epsilon`. The default value is
    :math:`\varepsilon = 0.6`.
    
    The rotational kernel is defined in velocity space :math:`v` and is given by
    
    .. math::
        
        y = 1 - \left(\frac{v}{v_\mathrm{rot}}\right)^2 \\
        
        K_\mathrm{rot} = 2 (1-\varepsilon)\sqrt{y} + \frac{\pi}{2} \left(\frac{\varepsilon y}{\pi v_\mathrm{rot}(1-\varepsilon/3)}\right)
    
    **Construct a simple Gaussian line profile and convolve with vsini=65 km/s**
    
    
    >>> sigma = 0.5
    >>> wave = np.linspace(3995, 4005, 1001)
    >>> flux = 1.0 - 0.5*np.exp( - (wave-4000)**2/(2*sigma**2))
    
    Convolve it with a rotational velocity of :math:`v_\mathrm{rot}=65 \mathrm{km}\,\mathrm{s}^{-1}`:

    >>> vrot = 65.
    >>> flux_broad = tools.broadening_rotational(wave, flux, vrot)
    >>> flux_broad, (wave_kernel, kernel) = tools.broadening_rotational(wave, flux, vrot, return_kernel=True)

    ::
    
        plt.figure()
        plt.plot(wave, flux, 'k-')
        plt.plot(wave, flux_broad, 'r-', lw=2)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Normalised flux")

        plt.figure()
        plt.plot(wave_kernel, kernel, 'r-', lw=2)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Normalised flux")
    
    
    .. +----------------------------------------------------------------------+----------------------------------------------------------------------+
    .. | .. image:: ../../images/api/spectra/tools/broaden_rotational01.png   | .. image:: ../../images/api/spectra/tools/broaden_rotational02.png   |
    .. |   :scale: 50 %                                                       |   :scale: 50 %                                                       |
    .. +----------------------------------------------------------------------+----------------------------------------------------------------------+
    
        
    :parameter wave: Wavelength of the spectrum
    :type wave: array
    :parameter flux: Flux of the spectrum
    :type flux: array
    :parameter vrot: Rotational broadening
    :type vrot: float
    :parameter epsilon: linear limbdarkening parameter
    :type epsilon: float
    :parameter return_kernel: return kernel
    :type return_kernel: bool
    :return: broadened flux [, (wavelength, kernel)]
    :rtype: array [,(array, array)]
    """
    
    # if there is no rotational velocity, don't bother
    if vrot == 0:    
        return flux
    
    # convert wavelength array into velocity space, this is easier. We also
    # need to make it equidistant
    velo = np.log(wave)
    delta_velo = np.diff(velo).min()
    range_velo = velo.ptp()
    n_velo = int(range_velo/delta_velo) + 1
    velo_ = np.linspace(velo[0], velo[-1], n_velo)
    flux_ = np.interp(velo_, velo, flux)
    dvelo = velo_[1]-velo_[0]
    vrot = vrot / (constants.cc*1e-3)
    n_kernel = int(2*vrot/dvelo) + 1
    
    # The kernel might be of too low resolution, or the the wavelength range
    # might be too narrow. In both cases, raise an appropriate error
    if n_kernel == 0:
        raise ValueError(("Spectrum resolution too low for "
                          "rotational broadening"))
    elif n_kernel > n_velo:
        raise ValueError(("Spectrum range too narrow for "
                          "rotational broadening"))
    
    # Construct the domain of the kernel
    velo_k = np.arange(n_kernel)*dvelo
    velo_k -= velo_k[-1]/2.
    
    # transform the velocity array, construct and normalise the broadening
    # kernel
    y = 1 - (velo_k/vrot)**2
    kernel = (2*(1-epsilon)*np.sqrt(y) + \
                       np.pi*epsilon/2.*y) / (np.pi*vrot*(1-epsilon/3.0))
    kernel /= kernel.sum()
    
    # Convolve the flux with the kernel
    flux_conv = fftconvolve(1-flux_, kernel, mode='same')
    
    # And interpolate the results back on to the original wavelength array, 
    # taking care of even vs. odd-length kernels
    if n_kernel % 2 == 1:
        offset = 0.0
    else:
        offset = dvelo / 2.0
    
    flux = np.interp(velo+offset, velo_, 1-flux_conv, left=1, right=1)
    
    # Return the results
    if return_kernel:
        lambda0 = (wave[-1]+wave[0]) / 2.0
        return flux, (velo_k*lambda0, kernel)
    else:
        return flux



def vmacro_kernel(dlam, Ar, At, Zr, Zt):
    r"""
    Macroturbulent velocity kernel.
    
    See :py:func:`broadening_macroturbulent` for more information.
    """
    dlam[dlam == 0] = 1e-8
    if Zr != Zt:
        return np.array([(2*Ar*idlam/(np.sqrt(np.pi)*Zr**2) * quad(lambda u: np.exp(-1/u**2),0,Zr/idlam)[0] + \
                          2*At*idlam/(np.sqrt(np.pi)*Zt**2) * quad(lambda u: np.exp(-1/u**2),0,Zt/idlam)[0]) 
                             for idlam in dlam])
    else:
        return np.array([(2*Ar*idlam/(np.sqrt(np.pi)*Zr**2) + 2*At*idlam/(np.sqrt(np.pi)*Zt**2))\
                           * quad(lambda u: np.exp(-1/u**2),0,Zr/idlam)[0]\
                             for idlam in dlam])


def broadening_macroturbulent(wave, flux, vmacro_rad, vmacro_tan=None,
                              return_kernel=False):
    r"""
    Apply macroturbulent broadening.
    
    The macroturbulent kernel is defined as in [Gray2005]_:
    
    .. math::
    
        K_\mathrm{macro}(\Delta\lambda) = \frac{2A_R\Delta\lambda}{\sqrt{\pi}\zeta_R^2}\int_0^{\zeta_R/\Delta\lambda}e^{-1/u^2}du 
        
         & + \frac{2A_T\Delta\lambda}{\sqrt{\pi}\zeta_T^2}\int_0^{\zeta_T/\Delta\lambda}e^{-1/u^2}du 
    
    If :envvar:`vmacro_tan` is :envvar:`None`, then the value will be put equal
    to the radial component :envvar:`vmacro_rad`.
    
    **Example usage**: Construct a simple Gaussian line profile and convolve with vmacro=65km/s
        
    Construct a simple Gaussian line profile:
    
    >>> sigma = 0.5
    >>> wave = np.linspace(3995, 4005, 1001)
    >>> flux = 1.0 - 0.5*np.exp( - (wave-4000)**2/(2*sigma**2))
    
    Convolve it with a macroturbulent velocity of :math:`v_\mathrm{macro}=65 \mathrm{km}\,\mathrm{s}^{-1}`:

    >>> vmac = 65.
    >>> flux_broad = tools.broadening_macroturbulent(wave, flux, vmac)
    >>> flux_broad, (wave_kernel, kernel) = tools.broadening_macroturbulent(wave, flux, vmac, return_kernel=True)

    ::
    
        plt.figure()
        plt.plot(wave, flux, 'k-')
        plt.plot(wave, flux_broad, 'r-', lw=2)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Normalised flux")

        plt.figure()
        plt.plot(wave_kernel, kernel, 'r-', lw=2)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Normalised flux")
    
    
    .. +--------------------------------------------------------------------------+-------------------------------------------------------------------------+
    .. | .. image:: ../../images/api/spectra/tools/broaden_macroturbulent01.png   | .. image:: ../../images/api/spectra/tools/broaden_macroturbulent02.png  |
    .. |   :scale: 50 %                                                           |   :scale: 50 %                                                          |
    .. +--------------------------------------------------------------------------+-------------------------------------------------------------------------+
 
    :parameter wave: Wavelength of the spectrum
    :type wave: array
    :parameter flux: Flux of the spectrum
    :type flux: array
    :parameter vmacro_rad: macroturbulent broadening, radial component
    :type vmacro_rad: float
    :parameter vmacro_tan: macroturbulent broadening, tangential component
    :type vmacro_tan: float    
    :parameter return_kernel: return kernel
    :type return_kernel: bool
    :return: broadened flux [, (wavelength, kernel)]
    :rtype: array [,(array, array)]
    """
    if vmacro_tan is None:
        vmacro_tan = vmacro_rad
    
    if vmacro_rad == vmacro_tan == 0:
        return flux
    
    # Define central wavelength
    lambda0 = (wave[0] + wave[-1]) / 2.0
    
    vmac_rad = vmacro_rad/(constants.cc*1e-3)*lambda0
    vmac_tan = vmacro_tan/(constants.cc*1e-3)*lambda0
    
    # Make sure the wavelength range is equidistant before applying the
    # convolution
    delta_wave = np.diff(wave).min()
    range_wave = wave.ptp()
    n_wave = int(range_wave/delta_wave)+1
    wave_ = np.linspace(wave[0], wave[-1], n_wave)
    flux_ = np.interp(wave_, wave, flux)
    dwave = wave_[1]-wave_[0]
    n_kernel = int(5*max(vmac_rad, vmac_tan)/dwave)
    if n_kernel % 2 == 0:
        n_kernel += 1
    
    # The kernel might be of too low resolution, or the the wavelength range
    # might be too narrow. In both cases, raise an appropriate error
    if n_kernel == 0:
        raise ValueError(("Spectrum resolution too low for "
                          "macroturbulent broadening"))
    elif n_kernel > n_wave:
        raise ValueError(("Spectrum range too narrow for "
                          "macroturbulent broadening"))
    
    # Construct the broadening kernel
    wave_k = np.arange(n_kernel)*dwave
    wave_k -= wave_k[-1]/2.
    kernel = vmacro_kernel(wave_k, 1.0, 1.0, vmac_rad, vmac_tan)
    kernel /= sum(kernel)
    
    flux_conv = fftconvolve(1-flux_, kernel, mode='same')
    
    # And interpolate the results back on to the original wavelength array, 
    # taking care of even vs. odd-length kernels
    if n_kernel % 2 == 1:
        offset = 0.0
    else:
        offset = dwave / 2.0
    flux = np.interp(wave+offset, wave_, 1-flux_conv)
    
    # Return the results.
    if return_kernel:
        return flux, (wave_k, kernel)
    else:
        return flux





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
    logger.info("Rot.broad with vrot={:.3f}km/s (epsilon={:.2f}), sigma={:.2f}AA, vmacro={:.3f}km/s".format(vrot,epsilon,fwhm,vmac))
    #-- first a wavelength Gaussian convolution:
    if fwhm>0:
        fwhm /= 2.3548
        #-- make sure it's equidistant
        wave_ = np.linspace(wave_spec[0],wave_spec[-1],len(wave_spec))
        flux_ = np.interp(wave_,wave_spec,flux_spec)
        dwave = wave_[1]-wave_[0]
        n = int(2*4*fwhm/dwave)
        if n==0:
            logger.info("Resolution too large, cannot broaden with instrumental profile")
        else:
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
                flux_spec = np.interp(wave_spec+dwave/2,wave_,1-flux_conv)
    #-- macroturbulent profile
    if vmac>0:
        vmac = vmac/(constants.cc*1e-3)*(wave_spec[0]+wave_spec[-1])/2.0
        #-- make sure it's equidistant
        wave_ = np.linspace(wave_spec[0],wave_spec[-1],len(wave_spec))
        flux_ = np.interp(wave_,wave_spec,flux_spec)
        dwave = wave_[1]-wave_[0]
        n = int(6*vmac/dwave/5)
        if n==0:
            logger.error("Resolution too large, cannot broaden with instrumental profile")
        else:
            wave_k = np.arange(n)*dwave
            wave_k-= wave_k[-1]/2.
            kernel = vmacro_kernel(wave_k,1.,1.,vmac,vmac)
            kernel /= sum(kernel)
            flux_conv = fftconvolve(1-flux_,kernel,mode='same')
            if n%2==1:
                flux_spec = np.interp(wave_spec,wave_,1-flux_conv)
            else:
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


if __name__=="__main__":
    import doctest
    from matplotlib import pyplot as plt
    from matplotlib import pyplot as pl
    doctest.testmod()
    plt.show()
