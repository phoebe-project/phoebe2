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
            logger.error("Resolution too large, cannot broaden with instrumental profile")
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




def vsini(wave,flux,epsilon=0.6,clam=None,window=None,**kwargs):
    """
    Deterimine vsini of an observed spectrum via the Fourier transform method.
    
    According to [SimonDiaz2007]_ and [Carroll1933_:
    
    vsini = 0.660 * c/ (lambda * f1)
    
    But more general (see [Reiners2002]_, [Dravins1990]_)
    
    vsini = q1 * c/ (lambda*f1)
    
    where f1 is the first minimum of the Fourier transform.
    
    The error is estimated as the Rayleigh limit of the Fourier Transform
    
    Example usage and tests: Generate some data. We need a central wavelength (A),
    the speed of light in angstrom/s, limb darkening coefficients and test
    vsinis:
    
    >>> clam = 4480.
    >>> c = conversions.convert('m/s','A/s',constants.cc)
    >>> epsilons = np.linspace(0.,1.0,10)
    >>> vsinis = np.linspace(50,300,10)
    
    We analytically compute the shape of the Fourier transform in the following
    domain (and need the C{scipy.special.j1} for this)
    
    >>> x = np.linspace(0,30,1000)[1:]
    >>> from scipy.special import j1
    
    Keep track of the calculated and predicted q1 values:
    
    >>> q1s = np.zeros((len(epsilons),len(vsinis)))
    >>> q1s_pred = np.zeros((len(epsilons),len(vsinis)))
    
    Start a figure and set the color cycle
    
    >>> p= pl.figure()
    >>> p=pl.subplot(131)
    >>> color_cycle = [pl.cm.spectral(j) for j in np.linspace(0, 1.0, len(epsilons))]
    >>> p = pl.gca().set_color_cycle(color_cycle)
    >>> p=pl.subplot(133);p=pl.title('Broadening kernel')
    >>> p = pl.gca().set_color_cycle(color_cycle)
    
    Now run over all epsilons and vsinis and determine the q1 constant:
    
    >>> for j,epsilon in enumerate(epsilons):
    ...    for i,vsini in enumerate(vsinis):
    ...       vsini = conversions.convert('km/s','A/s',vsini)
    ...       delta = clam*vsini/c
    ...       lambdas = np.linspace(-5.,+5.,20000)
    ...       #-- analytical rotational broadening kernel
    ...       y = 1-(lambdas/delta)**2
    ...       G = (2*(1-epsilon)*np.sqrt(y)+pi*epsilon/2.*y)/(pi*delta*(1-epsilon/3))
    ...       lambdas = lambdas[-np.isnan(G)]
    ...       G = G[-np.isnan(G)]
    ...       G /= max(G)
    ...       #-- analytical FT of rotational broadening kernel
    ...       g = 2. / (x*(1-epsilon/3.)) * ( (1-epsilon)*j1(x) +  epsilon* (sin(x)/x**2 - cos(x)/x))
    ...       #-- numerical FT of rotational broadening kernel
    ...       sigma,g_ = pergrams.deeming(lambdas-lambdas[0],G,fn=2e0,df=1e-3,norm='power')
    ...       myx = 2*pi*delta*sigma
    ...       #-- get the minima
    ...       rise = np.diff(g_[1:])>=0
    ...       fall = np.diff(g_[:-1])<=0
    ...       minima = sigma[1:-1][rise & fall]
    ...       minvals = g_[1:-1][rise & fall]
    ...       q1s[j,i] =  vsini / (c/clam/minima[0])
    ...       q1s_pred[j,i] = 0.610 + 0.062*epsilon + 0.027*epsilon**2 + 0.012*epsilon**3 + 0.004*epsilon**4
    ...    p= pl.subplot(131)
    ...    p= pl.plot(vsinis,q1s[j],'o',label='$\epsilon$=%.2f'%(epsilon));pl.gca()._get_lines.count -= 1
    ...    p= pl.plot(vsinis,q1s_pred[j],'-')
    ...    p= pl.subplot(133)
    ...    p= pl.plot(lambdas,G,'-')
    
    And plot the results:
    
    >>> p= pl.subplot(131)
    >>> p= pl.xlabel('v sini [km/s]');p = pl.ylabel('q1')
    >>> p= pl.legend(prop=dict(size='small'))
    
    
    >>> p= pl.subplot(132);p=pl.title('Fourier transform')
    >>> p= pl.plot(x,g**2,'k-',label='Analytical FT')
    >>> p= pl.plot(myx,g_/max(g_),'r-',label='Computed FT')
    >>> p= pl.plot(minima*2*pi*delta,minvals/max(g_),'bo',label='Minima')
    >>> p= pl.legend(prop=dict(size='small'))
    >>> p= pl.gca().set_yscale('log')
    
    ]include figure]]images/atmospheres_tools_vsini_kernel.png]
    
    Extra keyword arguments are passed to L{pergrams.deeming}
    
    @param wave: wavelength array in Angstrom
    @type wave: ndarray
    @param flux: normalised flux of the profile
    @type flux: ndarray
    @rtype: (array,array),(array,array),array,float
    @return: periodogram (freqs in s/km), extrema (weird units), vsini values (km/s), error (km/s)
    """
    cc = conversions.convert('m/s','AA/s',constants.cc)
    #-- clip the wavelength and flux region if needed:
    if window is not None:
        keep = (window[0]<=wave) & (wave<=window[1])
        wave,flux = wave[keep],flux[keep]
    #-- what is the central wavelength? If not given, it's the middle of the
    #   wavelength range
    if clam is None: clam = ((wave[0]+wave[-1])/2)
    #-- take care of the limb darkening
    q1 = 0.610 + 0.062*epsilon + 0.027*epsilon**2 + 0.012*epsilon**3 + 0.004*epsilon**4
    #-- do the Fourier transform and normalise
    #flux = flux / (np.median(np.sort(flux)[-5:]))
    v0 = kwargs.pop('v0',1.)
    vn = kwargs.pop('vn',500.)
    dv = kwargs.pop('dv',1.)
    fn = 1./conversions.convert('km/s','AA/s',v0)*q1*cc/clam
    f0 = 1./conversions.convert('km/s','AA/s',vn)*q1*cc/clam
    df = kwargs.setdefault('df',(fn-f0) / ((vn-v0)/dv))
    kwargs.setdefault('fn',fn)
    kwargs.setdefault('f0',f0)
    kwargs.setdefault('df',df)
    freqs,ampls = pergrams.DFTpower(wave,(1-flux),**kwargs)
    
    ampls = ampls/max(ampls)
    error = 1./wave.ptp()
    #-- get all the peaks
    rise = np.diff(ampls[1:])>=0
    fall = np.diff(ampls[:-1])<=0
    minima = freqs[1:-1][rise & fall]
    minvals = ampls[1:-1][rise & fall]
    #-- compute the vsini and convert to km/s
    freqs = freqs*clam/q1/cc
    freqs = conversions.convert('s/AA','s/km',freqs,wave=(clam,'AA'))
    vsini_values = cc/clam*q1/minima
    vsini_values = conversions.convert('AA/s','km/s',vsini_values)#,wave=(clam,'AA'))
    #-- determine the error as the rayleigh limit
    error = error*clam/q1/cc
    error = conversions.convert('s/AA','s/km',error,wave=(clam,'AA'))
    return (freqs,ampls),(minima,minvals),vsini_values,error


if __name__=="__main__":
    import doctest
    from matplotlib import pyplot as plt
    from matplotlib import pyplot as pl
    doctest.testmod()
    plt.show()