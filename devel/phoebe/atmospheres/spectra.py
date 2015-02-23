"""
Interpolate and manipulate spectra.
"""
import os
import logging
import numpy as np
import scipy.integrate
try:
    import pyfits
except:
    try: # Pyfits now integrated in astropy
        import astropy.io.fits as pyfits
    except ImportError:
        print(("Soft warning: pyfits could not be found on your system, you can "
           "only use black body atmospheres and Gaussian synthetic spectral "
           "profiles"))
from phoebe.algorithms import interp_nDgrid
from phoebe.utils import decorators

logger = logging.getLogger('ATMO.SP')

_aliases = {'atlas':'ATLASp_z0.0t0.0_a0.00_spectra.fits',\
            'bstar':'BSTAR2006_z0.000v2_vis_spectra.fits',
            'ostar':'OSTAR2002_z0.000v10_vis_spectra.fits',
            'fw_HHe':'FWHHe_z0.000v5_vis_spectra.fits',
            'fw_HHeSiCNOMg':'FWHHeSiCNOMg_z0.000v5_vis_spectra.fits'}

@decorators.memoized
def _prepare_grid_spectrum(wrange, gridfile):
    """
    Example: 4570 - 4572 (N=200)
    """
    with pyfits.open(gridfile) as ff:
        grid_pars = np.zeros((2,len(ff)-1))
        grid_data = np.zeros((len(wrange)*2,len(ff)-1))
        for i,ext in enumerate(ff[1:]):
            grid_pars[0,i] = np.log10(ext.header['teff'])
            grid_pars[1,i] = ext.header['logg']
            wave = ext.data.field('wavelength')
            flux = ext.data.field('flux')
            cont = np.log10(ext.data.field('cont'))
            index1,index2 = np.searchsorted(wave,[wrange[0],wrange[-1]])
            #print index1,index2,wave.min(),wave.max(),len(wave),len(flux),len(cont)wrange[0],wrange[-1]
            index1 -= 1
            index2 += 1
            if index1 < 0:
                index1 = 0
            flux = np.interp(wrange,wave[index1:index2],flux[index1:index2])
            cont = np.interp(wrange,wave[index1:index2],cont[index1:index2])
            grid_data[:,i] = np.hstack([flux,cont])
            
    #-- create the pixeltype grid
    axis_values, pixelgrid = interp_nDgrid.create_pixeltypegrid(grid_pars,grid_data)
    
    return axis_values,pixelgrid        
            
def choose_spec_table(profile,atm_kwargs={},red_kwargs={}):
    """
    Derive the filename of grid of template spectra from the input parameters.
   
    """
    #-- perhaps the user gave a filename: then return it
    if os.path.isfile(profile):
        return profile
    #-- else we need to be a little bit more clever and derive the filename
    else:
        #-- get some basic info
        abun = atm_kwargs['abun']
        prefix = 'm' if abun<0 else 'p'
        abun = abs(abun*10)
        basename = '{}_{}{:02.0f}.fits'.format(profile,prefix,abun)        
        ret_val = os.path.join(basedir_ld_coeffs,basename)
        if os.path.isfile(ret_val):
            return ret_val
        else:
            raise ValueError("Cannot interpret profile parameter {}: I think the file that I need is {}, but it doesn't exist. If in doubt, consult the installation section of the documentation on how to add spectrum tables.".format(profile,ret_val))            

def interp_spectable(gridfile,teff,logg,wrange):
    """
    Interpolate a spectrum table.
    
    Return line shape and continuum arrays.
    
    @param teff: array of effective temperatures
    @type teff: array
    @param logg: array of surface gravities
    @type logg: array
    @param wrange: array of wavelengths (angstrom)
    @type wrange: array
    """
    
    # Get the FITS-file containing the tables
    # Retrieve structured information on the grid (memoized)
    if not os.path.isfile(gridfile):
        if gridfile in _aliases:
            gridfile = _aliases[gridfile]
        gridfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'tables','spectra',gridfile)
    
    axis_values, pixelgrid = _prepare_grid_spectrum(wrange, gridfile)
    
    # Prepare input:
    values = np.zeros((2,len(teff)))
    values[0] = np.log10(teff)
    values[1] = logg
    N = len(wrange)
    #print len(axis_values), len(values)
    #print 'axis values',axis_values[0]
    #print 'axis values',axis_values[1]
    #print 'values', values[0]
    #print 'values', values[1]
    #print values[0].min()>axis_values[0].min()
    #print values[1].min()>axis_values[1].min()
    #print values[0].max()<axis_values[0].max()
    #print values[1].max()<axis_values[1].max()
    #raise SystemExit
    try:
        pars = interp_nDgrid.interpolate(values,axis_values,pixelgrid)
    except IndexError:
        #pl.figure()
        #pl.title('Error in interpolation')
        #pl.plot(10**values[0],values[1],'ko')
        #pl.axhline(y=axis_values[1].min(),color='r',lw=2)
        #pl.axhline(y=axis_values[1].max(),color='r',lw=2)
        #pl.axvline(x=10**axis_values[0].min(),color='r',lw=2)
        #pl.axvline(x=10**axis_values[0].max(),color='r',lw=2)
        #pl.xlabel('Axis values teff')
        #pl.ylabel('Axis values logg')
        #print("DEBUG INFO: {}".format(axis_values))
        #pl.show()
        raise IndexError('Outside of grid: {:.1f}<teff<{:.1f}, {:.2f}<logg<{:.2f}'.format(10**values[0].min(),10**values[0].max(),values[1].min(),values[1].max()))
    pars = np.array(pars)
    pars[N:] = 10**pars[N:]
    pars = pars.reshape((2,len(wrange),-1))
    return pars




def moments(velo, flux, snr=200., max_moment=3):
    r"""
    Compute the moments from a line profile.
    
    The :math:`k`-th moment is defined as (see, e.g., [Aerts2010]_, Eq. 6.62-6.65)
    
    .. math::
    
        m_k = \sum_{i=1}^N (1-F_i) x_i^k \Delta x_i
        
    where :math:`F_i` is the flux and :math:`x_i` is the velocity of a given
    wavelength pixel. Then, :math:`\Delta x_i=x_i-x_{i-1}`.
    
    In this routine, numerical integration is done using Simpson's rule.
    
    We estimate the uncertainty in the flux in each pixel as
    
    .. math::
    
        \sigma_i = \mathrm{SNR}^{-1} F^{-1/2}
    
    :param velo: velocity array
    :type velo: array
    :param flux: normalised flux array
    :type flux: array
    :param snr: signal-to-noise ratio used for error estimation
    :type snr: float
    :param max_moment: highest moment to compute (including)
    :type max_moment: int
    :return: moments and error on moments
    :rtype: two-tupel arrays of shape (N x max_moment)
    """
    # Prepare output arrays
    mymoms, e_mymoms = np.zeros(max_moment+1), np.zeros(max_moment+1)
    
    # Compute zeroth moment (equivalent width)
    m0 = scipy.integrate.simps( (1-flux), x=velo)
    
    # To calculate the uncertainties, we need the error in each velocity
    # bin, and the error on the zeroth moment
    sigma_i = 1./snr / np.sqrt(flux)
    Delta_v0 = np.sqrt(scipy.integrate.simps( sigma_i,x=velo)**2)
    e_m0 = np.sqrt(2)*(Delta_v0/m0)
    
    mymoms[0] = m0
    e_mymoms[0] = e_m0
        
    # Calculate other moments
    for n in range(1, max_moment+1):
        mymoms[n] = scipy.integrate.simps((1-flux)*velo**n, x=velo)
        # And the uncertainty on it
        Delta_vn = np.sqrt(scipy.integrate.simps(sigma_i*velo**n)**2)
        e_mymoms[n] = np.sqrt(  (Delta_vn/m0)**2 + (Delta_v0/m0 * mymoms[n])**2 )
        
    return mymoms, e_mymoms