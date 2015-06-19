"""
Create interpolation tables and check them.
"""

from phoebe.atmospheres import limbdark
from phoebe.io import fits
from phoebe.atmospheres import passbands as pbmod
from phoebe.atmospheres import sed
from phoebe.utils import utils
from phoebe.units import constants
from phoebe.units import conversions
from phoebe.parameters import parameters
from phoebe.backend import universe
import matplotlib.pyplot as pl
import numpy as np
import logging
import shutil
import os
try: # Pyfits now integrated in astropy
    import pyfits
except:
    import astropy.io.fits as pyfits
import argparse
import scipy.ndimage.filters

logger = logging.getLogger("ATM.GRID")


def compute_grid_ld_coeffs(atm_files,atm_pars=('teff', 'logg'),\
                           red_pars_iter={},red_pars_fixed={},vgamma=None,\
                           passbands=('JOHNSON.V',),\
                           law='claret',fitmethod='equidist_r_leastsq',\
                           limb_zero=False, \
                           filetag=None, debug_plot=False,
                           check_passband_coverage=True,
                           blackbody_wavelength_range=None,
                           blackbody_teff_range=None):
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
    necessary. The first method is the physically most precise: boosting is
    then included by shifting the SED according the radial velocity ``vgamma``,
    and then computing intensities and limb darkening coefficients (see
    :py:func:`get_specific_intensities() <phoebe.atmospheres.limbdark.get_specific_intensities>`).
    The other way is to compute the (linear) boosting factor :math:`\alpha_b` of
    an SED (:math:`(\lambda, F_\lambda)`) via:
    
    .. math::
            
        \alpha_b = \frac{\int_P (5+\frac{d\ln F_\lambda}{d\ln\lambda})\lambda F_\lambda d\lambda}{\int_P \lambda F_\lambda d\lambda}
    
    Then the boosting amplitude in the passband :math:`P` can be computed as:
    
    .. math::
    
        A_b = \alpha_b \frac{v}{c}
    
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
    (uniform), but you can add reddening or boosting:
    
    **Black body examples**
    
    The wavelength range for blackbody can be set via :envvar:`blackbody_wavelength_range`. It needs to be an array of wavelength
    points. It defaults to :envvar:`np.logspace(1,7,10000)`.
    
    The temperature range for blackbody can be set via :envvar:`blackbody_teff_range`.
    It defaults to :envvar:`np.logspace(1, 5, 200)` (10-100kK in 200 steps).
        
    Case 0: plain black body
    
        >>> compute_grid_ld_coeffs('blackbody')
        
    Case 1: black body with boosting
    
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
    
    **Case 3**: Like Case 1, but with Doppler boosting included:
    
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
    @param vgamma: list of values to use for Doppler boosting (i.e. velocity in
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
        if filetag is None:
            filetag = 'blackbody'
        fitmethod = 'none'
    elif filetag is None:
        filetag = 'kurucz'
    
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
                    if ff[0].header[key] == 'blackbody':
                        atm_file = 'blackbody'
                    else:
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
        if atm_file == 'blackbody' and blackbody_wavelength_range is None:
            wave_ = np.logspace(1, 7, 10000)
        elif atm_file == 'blackbody':
            wave_ = blackbody_wavelength_range
        
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
                
                if not law == 'uniform':
                    raise ValueError("Cannot fit LD law: disk integrated intensities require uniform LD law")
                
        
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
        if atm_file == 'blackbody' and blackbody_wavelength_range is None:
            wave_ = np.logspace(1, 7, 10000)
        elif atm_file == 'blackbody':
            wave_ = blackbody_wavelength_range
        
        # Check if the file exists
        elif not os.path.isfile(atm_file):
            raise ValueError(("Cannot fit LD law, specific intensity file {} "
                             "does not exist").format(atm_file))
        
        # Else we pre-open the atmosphere file to try to gain some time
        else:
            open_atm_file = pyfits.open(atm_file)
        
        iterator = limbdark.iter_grid_dimensions(atm_file, atm_par_names, other_pars, blackbody_teff_range=blackbody_teff_range)
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
                mu, Imu = limbdark.get_limbdarkening(open_atm_file,
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
            
            extra = []
            
            if atm_file != 'blackbody':
                mus, wave, table = limbdark.get_specific_intensities(open_atm_file,
                                 atm_kwargs=atm_kwargs, red_kwargs=red_kwargs,
                                 vgamma=vgamma)
                
                if not np.isnan(mus[0]):
                    # Disk-integrate the specific intensities
                    rs_ = np.sqrt(1 - mus**2)
                    if rs_[-1]==1:
                        itable = table[:,:-1]
                    else:
                        itable = table
                        rs_ = np.hstack([rs_, 1.0])
                    flux = (np.pi*np.diff(rs_[None,:]**2) * itable).sum(axis=1)
                else:
                    # otherwise it's already disk-integrated
                    flux = table[:,0]
            
            else:
                wave, flux = wave_*10, Imu_blackbody*np.pi                                
            
            # Compute disk-integrated fluxes
            disk_integrateds = sed.synthetic_flux(wave, flux, passbands)
            
            # transform fluxes and wavelength
            # we first smooth somewhat to avoid numerical issues with metal lines. We go for gaussian smoothing of 10 angstrom in case we have a regular grid here.
            logger.debug("Smoothing over %f points before computing beaming factor"%(10/np.median(np.diff(wave))))
            flux_smooth = scipy.ndimage.filters.gaussian_filter1d(flux, 10/np.median(np.diff(wave)))
            wavl_smooth = scipy.ndimage.filters.gaussian_filter1d(wave, 10/np.median(np.diff(wave)))

            lnF = np.log(flux_smooth)
            lnl = np.log(wavl_smooth)
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
            if law != 'uniform' and law != 'prsa':
                for i, pb in enumerate(passbands):
                    # Fit a limbdarkening law:
                    csol, res, dflux = limbdark.fit_law(mu, Imu[:, i]/Imu[0, i],
                                               law=law, fitmethod=fitmethod,
                                               limb_zero=limb_zero,
                                               debug_plot=(i+1 if debug_plot else False))
                    # Compute disk integrated value: this seems better than the
                    # thing attempted above during boosting calculations
                    if 'disk_{}'.format(law) in globals():
                        disk_integrateds[i] = globals()['disk_{}'.format(law)](csol)*Imu[0,i]
                    to_append = list(val) + [extra[i], disk_integrateds[i]] + \
                                    [res, dflux] + list(csol) + [Imu[0, i]]
                    output[pb].append(to_append)
                    
                    logger.info("{}: {}".format(pb, output[pb][-1]))
            
            # For the Prsa law, we need to append values for each mu
            elif law == 'prsa':
                for i, pb in enumerate(passbands):
                    # Fit a limbdarkening law:
                    allcsol, mu_grid, dflux = limbdark.fit_law(mu, Imu[:, i]/Imu[0, i],
                                               law=law, fitmethod=fitmethod,
                                               limb_zero=limb_zero, oversampling=32,
                                               debug_plot=(i+1 if debug_plot else False))
                    res = 0.0
                    sa = np.argsort(mu_grid)
                    allcsol = allcsol[sa]
                    mu_grid = mu_grid[sa]
                    for j, imu in enumerate(mu_grid):
                        csol = list(allcsol[j:j+1])
                        to_append = list(val) + [imu] + [extra[i], disk_integrateds[i]] + \
                                    [res, dflux] + csol + [Imu[0, i]]
                        output[pb].append(to_append)
                    
                    logger.info("{}: {}".format(pb, output[pb][-1]))
            
            # For a uniform limb darkening law, we can just take the 
            # disk-integrated intensity and divide by pi.
            else:
                csol = []
                for i, pb in enumerate(passbands):
                    
                    # So don't fit a law, we know what it is
                    to_append = list(val) + [extra[i], disk_integrateds[i]] + \
                                    [0.0, 0.0] + csol + [disk_integrateds[i]/np.pi]
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
        


def test_solar_calibration(atm):
    """
    Print info on a grid, and check the luminosity and visual magnitude of the Sun.
    """
    limbdark.register_atm_table(atm)
    output = limbdark._prepare_grid('OPEN.BOL', atm)
    header = output[-1]
    ld_func = header['c__ld_func']
    
    
    with pyfits.open(atm) as ff:
        npassbands = (len(ff)-1)/2
        grid_passbands = [ext.header['extname'] for ext in ff[1:] if not ext.header['EXTNAME'][:4]=='_REF']
        passbands = ", ".join(grid_passbands)
    
    grid_variables = [header[key] for key in header.keys() if key[:6]=='C__AI_']
    print("Atmosphere table: {}".format(atm))
    print("==========================================================")
    print("Grid variables: {}".format(", ".join(grid_variables)))
    
    with pyfits.open(atm) as ff:
        for grid_variable in grid_variables:
            grid_values = ff[grid_passbands[0]].data.field(grid_variable)
            print("         -- {}: {} -> {}".format(grid_variable, grid_values.min(), grid_values.max()))
    print("Limb darkening function: ".format(ld_func))
    print("Available passbands ({}): {}".format(npassbands, passbands))
    
    print("\nComputations")
    print("===========================")
    
    output = limbdark._prepare_grid('JOHNSON.V', atm)
    
    sun = parameters.ParameterSet('star', label='TheSun')
    sun['shape'] = 'sphere'
    sun['atm'] = atm
    sun['ld_coeffs'] = atm
    sun['ld_func'] = ld_func

    globals = parameters.ParameterSet('position')
    globals['distance'] = 1., 'au'

    sun_mesh = parameters.ParameterSet('mesh:marching')
    sun_mesh['delta'] = 0.05

    lcdep1 = parameters.ParameterSet('lcdep')
    lcdep1['ld_func'] = ld_func
    lcdep1['ld_coeffs'] = atm
    lcdep1['atm'] = atm
    lcdep1['passband'] = 'OPEN.BOL'
    lcdep1['ref'] = 'Bolometric (numerical)'

    lcdep2 = lcdep1.copy()
    lcdep2['method'] = 'analytical'
    lcdep2['ref'] = 'Bolometric (analytical)'
    
    lcdep3 = lcdep1.copy()
    lcdep3['passband'] = 'JOHNSON.V'
    lcdep3['ref'] = 'Visual'

    if ld_func == 'claret':
        lcdeps = [lcdep1, lcdep2, lcdep3]
    else:
        lcdeps = [lcdep1, lcdep3]
        
    the_sun = universe.Star(sun, sun_mesh, pbdep=lcdeps,
                          position=globals)

    the_sun.set_time(0)

    the_sun.lc()
    
    params = the_sun.get_parameters()
    nflux = the_sun.params['syn']['lcsyn']['Bolometric (numerical)']['flux'][0]
    if ld_func == 'claret':
        aflux = the_sun.params['syn']['lcsyn']['Bolometric (analytical)']['flux'][0]
    else:
        aflux = nflux
    pflux = the_sun.params['syn']['lcsyn']['Visual']['flux'][0]
    
    mupos = the_sun.mesh['mu']>0
    
    vmag = conversions.convert('W/m3','mag', pflux, passband='JOHNSON.V')
    
    num_error_area = np.abs(np.pi-((the_sun.mesh['size']*the_sun.mesh['mu'])[mupos]).sum())/np.pi*100
    num_error_flux = np.abs(nflux-aflux)/aflux*100
    
    print("\nSolar luminosity and fluxes")
    print("===========================")
    print("Computed analytical flux of the model: {} W/m2".format(aflux))
    print("Computed numerical flux of the model:  {} (={}?) W/m2".format(nflux, the_sun.projected_intensity()))
    print("Visual magnitude: {}".format(vmag))
    
    real_error_flux = np.abs(1368.000-aflux)/aflux*100
    
    assert(num_error_area<=0.048)
    assert(num_error_flux<=0.049)
    assert(real_error_flux<=5.0)
    assert(np.abs(vmag+26.75)<=0.1)
    
    lumi2 = params.get_value('luminosity','W')
    if ld_func == 'claret':
        lumi1 = limbdark.sphere_intensity(the_sun.params['star'],the_sun.params['pbdep']['lcdep'].values()[0])[0]
    else:
        lumi1 = lumi2
    lumi3 = the_sun.luminosity(ref='__bol')
    
    lumsn = constants.Lsol#_cgs
    num_error_area = np.abs(4*np.pi-the_sun.area())/4*np.pi*100
    num_error_flux = np.abs(lumi1-lumi2)/lumi1*100
    real_error_flux = np.abs(lumi1-lumsn)/lumsn*100
    
    if ld_func == 'claret':
        print("Computed analytical luminosity: {} W".format(lumi1))
    print("Computed numerical luminosity:  {} (~{}) W".format(lumi2, lumi3))
    print("True luminosity:                {} W".format(lumsn))
    
    assert(num_error_area<=0.48)
    if ld_func == 'claret':
        assert(num_error_flux<=0.040)
    assert(real_error_flux<=0.22)
    
    print(r"Intensities are accurate to {:.3g}%".format(real_error_flux))
    print("\n ... all checks passed")



if __name__ == "__main__":
    logger = utils.get_basic_logger(clevel='debug')
    
    # add box filters:
    passbands = []
    for clam in np.arange(3000,8001,500):
        wave = np.linspace(clam-20,clam+20,1000)
        trans = np.where(np.abs(clam-wave)<10.0, 1.0, 0.0)
        passband ='BOX_10.{:.0f}'.format(clam)
        passbands.append(passband)
        limbdark.pbmod.add_response(wave, trans, passband=passband)
    
    #-- initialize the parser and subparsers
    parser = argparse.ArgumentParser(description='Compute limb darkening tables',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #-- add positional arguments
    parser.add_argument('atm_files', metavar='atm_file', type=str, nargs='*',
                   default='blackbody',
                   help='Input atmosphere file(s)')
    
    parser.add_argument('--passbands', default='', help='Comma separated list of passbands')
    parser.add_argument('--ld_func', default='claret', help='Limb darkening function')
    parser.add_argument('--filetag', default='myfiletag', help='Tag to identify file')
    
    args = vars(parser.parse_args())
    
    atm_files = args.pop('atm_files')
    passbands = [pb.strip() for pb in args.pop('passbands').split(',')]
    law = args.pop('ld_func')
    filetag = args.pop('filetag')
    
    # override default filetag
    if atm_files == 'blackbody' and filetag == 'myfiletag':
        filetag = 'blackbody'
        
    # if the table that is given is a fits file but it does not contain an
    # column 'WAVELENGTH' in its first extension, it's a ld_coeffs file and we
    # need to check it (instead of compute the LD coeffs)
    is_spec_intens_table = True
    if len(atm_files)==1 and os.path.isfile(atm_files[0]):
        with pyfits.open(atm_files[0]) as open_file:
            names = [name.upper() for name in open_file[1].data.dtype.names]
            if not 'WAVELENGTH' in names:
                is_spec_intens_table = False
                
    if is_spec_intens_table:           
        compute_grid_ld_coeffs(atm_files,atm_pars=('teff', 'logg'),\
                           red_pars_iter={},red_pars_fixed={},vgamma=None,\
                           passbands=passbands,\
                           law=law,fitmethod='equidist_r_leastsq',\
                           limb_zero=False, \
                           filetag=filetag,
                           debug_plot=False,
                           check_passband_coverage=True)
    else:
        test_solar_calibration(atm_files[0])
