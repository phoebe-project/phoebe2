"""
Limb darkening and intensity fitting and querying

Section 1. File formats
=======================

There are two types of grids defined in the FITS format:

    1. grids with B{specific intensities}, holding in each extension a FITS table
    with a full at discrete limb angles, for a given set of parameter values.
    2. grids with B{LD coefficients}, holding in each extension a FITS table a grid
    of coefficients as a function of a given set of parameter values.
   
Section 1.1 FITS of Specific intensities
----------------------------------------

Each FITS table has the following header keys (KEY (type)):

    - C{TEFF} (float): effective temperature (K) of the specific intensities in the extension
    - C{LOGG} (float): log of surface gravity (cgs)
    - C{ABUN} (float): log of metallicity wrt to solar (optional)
    - C{VMIC} (float): microturbulent velocity (km/s) (optional)
    - C{...} (float):  any other float that is relevant to this atmosphere model and which can be used to interpolate in

Each FITS table has a data array with the following columns:

    - C{WAVELENGTH}: wavelength array
    - C{'1.000000'}: string representation of the first mu angle (mu=1)
    - C{'G.GGGGGG'}: string representation of the second mu angle (mu<1)
    - ... : string representation of other mu angles
    - C{'0.000000'}: string representation of the last mu angle (mu=0)

Section 1.2 FITS of LD coefficients
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
    - C{a1}: first limb darkening coefficients
    - C{ai}: ith limb darkening coefficients
    - C{Imu1}: flux of mu=1 SED.

Section 2. Creating LD grids
============================

There is a lot of flexibility in creating grids with limb darkening grids,
to make sure that the values you are interested in can be interpolated if
necessary, or otherwise be fixed. Therefore, creating these kind of grids
can be quite complicated. Hopefully, the following examples help in clarifying
the methods.

Suppose you have a list of atmosphere tables C{atm_files}, and you want to
create a grid that can be interpolated in effective temperature, surface
gravity and abundance,

>>> atm_pars = ('teff','logg','abun')

Suppose you want to do that in the C{JOHNSON.V} passband, using the C{claret}
limb darkening law, fitted equidistantly in radius coordinates with the
Levenberg-Marquardt routine. Then the basic interface is:

>>> compute_grid_ld_coeffs(atm_files,atm_pars=atm_pars,passbands=('JOHNSON.V',),
...       fitmethod='equidist_r_leastsq',filename='abun.fits')


Not all parameters you wish to interpolate are part of the precomputed tables.
This can be for example the system's radial velocity C{vgamma}, or the
extinction parameters. Suppose you want to make a grid in E(B-V):

>>> compute_grid_ld_coeffs(atm_files,atm_pars=('teff','logg'),
...       red_pars_iter=dict(ebv=np.linspace(0,0.5,5)),
...       red_pars_fixed=dict(law='fitzpatrick2004',Rv=3.1),
...       passbands=('JOHNSON.V',),fitmethod='equidist_r_leastsq',
...       filename='red.fits')

You need to split up the reddening parameters in those you which to `grid'
(C{red_pars_iter}), and those you want to keep fixed (C{red_pars_fixed}). The
former needs to be a dictionary, with the key the parameter name you wish to
vary, and the value an array of the parameter values you wish to put in the
grid. Similar examples, assuming that C{atm_files} is a list of specific
intensity files with different metallicities, are:

>>> compute_grid_ld_coeffs(atm_files,atm_pars=('teff','logg','abun'),
...       red_pars_iter=dict(ebv=np.linspace(0,0.5,5),Rv=[2.1,3.1,5.1]),
...       red_pars_fixed=dict(law='fitzpatrick2004'),
...       passbands=('JOHNSON.V',),fitmethod='equidist_r_leastsq',
...       filename='red.fits')

This last example also generate a small grid in C{Rv}.

>>> compute_grid_ld_coeffs(atm_files,atm_pars=('teff','logg','abun'),
...       red_pars_fixed=dict(law='fitzpatrick2004',Rv=3.1,ebv=0.123),
...       passbands=('JOHNSON.V',),fitmethod='equidist_r_leastsq',
...       filename='red.fits')

While this example fixes all reddening parameters.

Section 3. Querying specific intensities
========================================

To query the tables of specific intensities from a file C{atm} for a certain
effective temperature and surface gravity, do:

>>> mu,wavelengths,table = get_specific_intensities(atm,dict(teff=4000,logg=4.0))

As before, you can also specify reddening parameters or a systemic velocity.
You can immediately integrate the table in a list of passbands, (e.g. in the
C{JOHNSON.V} and C{2MASS.J} band):

>>> mu,intensities = get_limbdarkening(atm,dict(teff=4000,logg=4.0),
...                                    passbands=('JOHNSON.V','2MASS.J'))

Section 4. Fitting LD coefficients
==================================

The next step after querying the specific intensities integrated over certain
passbands, is to fit a limbdarkening law through them. Let us continue with
the last example from the previous section. The function C{fit_law} only
allows to fit one particular passband, and we better normalise the intensity
profile with the value in the disk-center:

>>> coeffs,res,dflux = fit_law(mu,intensities[:,0]/intensities[0,0],law='claret',
...                            fitmethod='equidist_r_leastsq')

The latter outputs the LD coeffs (C{coeffs}) and some fit statistics, such
as the residuals between the true and fitted fluxes, and the relative
differences.


Section 5. Querying precomputed LD coefficients
===============================================

Once an LD grid is precomputed, you can interpolate the limb darkening
coefficients for any set of values within the grid:

>>> coeffs = interp_ld_coeffs(atm,'JOHNSON.V',atm_kwargs=dict(teff=4000,logg=4.0))

The output C{coeffs} contains the limb darkening coefficients (first N-1 values)
and the center flux enabling an absolute scaling.

"""
#-- required packages
import os
import itertools
import logging
import numpy as np
from scipy.optimize import leastsq,fmin
from scipy.interpolate import splrep,splev
#-- optional packages
try:
    import pyfits
except ImportError:
    print("Soft warning: pyfits could not be found on your system, you can only use black body atmospheres and Gaussian synthetic spectral profiles")
try:
    import pylab as pl
except ImportError:
    print("Soft warning: matplotlib could not be found on your system, 2D plotting is disabled, as well as IFM functionality")
#-- Phoebe modules
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.utils import decorators
from phoebe.io import fits
from phoebe.algorithms import interp_nDgrid
from phoebe.atmospheres import sed
from phoebe.atmospheres import reddening
from phoebe.atmospheres import tools

logger = logging.getLogger('ATMO.LD')
logger.addHandler(logging.NullHandler())

basedir = os.path.dirname(os.path.abspath(__file__))
basedir_spec_intens = os.path.join(basedir,'tables','spec_intens')
basedir_ld_coeffs = os.path.join(basedir,'tables','ld_coeffs')

#{ LD laws

def ld_claret(mu,coeffs):
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
    Imu = 1-coeffs[0]*(1-mu**0.5)-coeffs[1]*(1-mu)-coeffs[2]*(1-mu**1.5)-coeffs[3]*(1-mu**2.)    
    return Imu

def ld_linear(mu,coeffs):
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
    return 1-coeffs[0]*(1-mu)
    
def ld_nonlinear(mu,coeffs):
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
    if hasattr(mu,'__iter__'):
        mu[mu==0] = 1e-16
    elif mu==0:
        mu = 1e-16
    return 1-coeffs[0]*(1-mu)-coeffs[1]*mu*np.log(mu)

def ld_logarithmic(mu,coeffs):
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
    return ld_nonlinear(mu,coeffs)

def ld_quadratic(mu,coeffs):
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
    return 1-coeffs[0]*(1-mu)-coeffs[1]*(1-mu)**2.0

def ld_uniform(mu,coeffs):
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
    return 1.

def ld_power(mu,coeffs):
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

#}
#{ Specific intensities


def get_grid_dimensions(atm,atm_pars=('teff','logg','abun')):
    """
    Retrieve all gridpoints and their parameter values from an atmosphere grid.
    
    @param atm: atmosphere grid filename
    @type atm: str
    @param atm_pars: grid axis
    @type atm_pars: tuple of strings
    @rtype: n x array
    @return: arrays with the axis values for each grid point.
    """
    ff = pyfits.open(atm)
    dims = [np.zeros(len(ff)-1) for par in pars]
    for i,mod in enumerate(ff[1:]):
        for j,key in enumerate(pars):
            dims[i][j] = mod.header[key]            
    ff.close()
    return dims

def iter_grid_dimensions(atm,atm_pars,other_pars):
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
    if atm=='blackbody':
        teffs = np.logspace(3,np.log10(50000),100)
        for pars in itertools.product(*[teffs]+list(other_pars)):
            yield pars
    else:
        with pyfits.open(atm) as ff:
            for pars in itertools.product(*[ff[1:]]+list(other_pars)):
                yield [pars[0].header[key] for key in atm_pars] + list(pars[1:])
    

def get_specific_intensities(atm,atm_kwargs={},red_kwargs={},vgamma=0):
    """
    Retrieve specific intensities of a model atmosphere at different angles.
    
    Retrieval is done from the precomputed table C{atm}.
    
    C{atm_kwargs} should be a dictionary with keys and values adequate for
    the grid, e.g. for a single-metallicity Kurucz grid::
        
        atm_kwargs = dict(teff=10000,logg=4.0)
    
    Extra keywords are determined by the header keys in the C{atm} file.
    
    C{red_kwargs} should be a dictionary with keys and values adequate for the
    limb darkening law, e.g.::
    
        red_kwargs = dict(law='fitzpatrick2004',Rv=3.1,ebv=0.5)
    
    The keywords are determined by the L{reddening.redden} function.
    
    C{vgamma} is radial velocity: positive is redshift, negative is blueshift
    (m/s!). This shifts the spectrum and does a first order correction for
    beaming effects.
    
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
    #-- read the atmosphere file:
    ff = pyfits.open(atm)
    if not atm_kwargs:
        raise ValueError("No arguments given")
    
    #-- try to run over each extension, and check if the values in the header
    #   correspond to the values given by the keys in C{atm_kwargs}.
    try:
        for mod in ff[1:]:
            for key in atm_kwargs:
                if mod.header[key]!=atm_kwargs[key]:
                    #-- for sure, this is not the right extension, don't check
                    #   the other keys
                    break
            else:
                #-- this is the right extension! Don't check the other
                #   extensions
                break
        else:
            #-- we haven't found the right extension
            raise ValueError("Did not found pars {} in {}".format(atm_kwargs,atm))
        #-- retrieve the mu angles, table and wavelength array
        whole_table = np.array(mod.data.tolist(),float)
        wave = whole_table[:,0] # first column, but skip first row
        table = whole_table[:,1:] 
        mu = np.array(mod.columns.names[1:], float)
        logger.debug('Model LD taken directly from file (%s)'%(os.path.basename(atm)))
    except KeyError:
        raise KeyError("Specific intensity not available in atmosphere file")
    
    ff.close()
    
    #-- velocity shift for doppler beaming if necessary
    if vgamma is not None and vgamma!=0:
        cc = constants.cc/1000. #speed of light in km/s
        for i in range(len(mu)):
            flux_shift = tools.doppler_shift(wave,-vgamma,flux=table[:,i])
            table[:,i] = flux_shift + 5.*vgamma/cc*flux_shift
    
    #-- redden if necessary
    if red_kwargs:
        for i in range(len(mu)):
            table[:,i] = reddening.redden(table[:,i],wave=wave,rtype='flux',**red_kwargs)
    
    #-- that's it!
    return mu,wave,table
    

def get_limbdarkening(atm,atm_kwargs={},red_kwargs={},vgamma=0,\
              passbands=('JOHNSON.V',),normalised=False):
    """
    Retrieve a limb darkening law for a specific star and specific bandpass.
    
    Possibility to include reddening via EB-V parameter. If not given, 
    reddening will not be performed...
    
    You choose your own reddening function.
    
    See e.g. [Heyrovsky2007]_
    
    If you specify one angle (:math:`\mu` in radians), it will take the closest match
    from the grid.
    
    :math:`\mu = \cos(\\theta)` where :math:`\\theta` is the angle between the
    surface and the line of sight. :math:`\mu=1` means :math:`\\theta=0` means
    center of the star.
    
    Example usage:
    
    >>> teff,logg = 5000,4.5
    >>> mu,intensities = get_limbdarkening(teff=teff,logg=logg,photbands=['JOHNSON.V'])

    @return: :math:`\mu` angles, intensities
    @rtype: array,array
    """
    #-- retrieve model atmosphere for a given teff and logg
    mus,wave,table = get_specific_intensities(atm,atm_kwargs=atm_kwargs,
                                              red_kwargs=red_kwargs,
                                              vgamma=vgamma)
    #-- compute intensity over the stellar disk, and normalise
    intensities = np.zeros((len(mus),len(passbands)))
    for i in range(len(mus)-1):
        intensities[i] = sed.synthetic_flux(wave,table[:,i],passbands)
    if normalised:
        intensities/= intensities.max(axis=0)
    return mus,intensities

def fit_law(mu,Imu,law='claret',fitmethod='equidist_r_leastsq'):
    """
    Fit an LD law to a sampled set of limb angles/intensities.
    
    Most likely options for ``law`` are:
    
    - ``claret``
    - ``linear``
    - ``quadratic``
    
    Possible options for ``fitmethod`` are:
    
    - ``leastsq``: Levenberg-Marquardt fit on the coordinates as available in
      the atmosphere grids
    - ``fmin``: Nelder-Mead fit on the coordinates as available in the
      atmosphere grids.
    - ``equidist_r_leastsq``: LM-fit in r-coordinate resampled equidistantly
    - ``equidist_mu_leastsq``: LM-fit in :math:`\mu`-coordinate resampled equidistantly
    - ``equidist_r_fmin``: NM-fit in r-coordinate resampled equidistantly
    - ``equidist_mu_fmin``: NM-fit in :math:`\mu`-coordinate resampled equidistantly
    
    In my (Pieter) experience, C{fitmethod='equidist_r_leastsq'} seems
    appropriate for the Kurucz models.
    
    Make sure the intensities are normalised!
    
    **Example usage**
    
    >>> mu,intensities = get_limbdarkening(teff=10000,logg=4.0,photbands=['JOHNSON.V'],normalised=True)
    >>> p = pl.figure()
    >>> p = pl.plot(mu,intensities[:,0],'ko-')
    >>> cl,ssr,rdiff = fit_law(mu,intensities[:,0],law='claret')
    
    >>> mus = np.linspace(0,1,1000)
    >>> Imus = ld_claret(mus,cl)
    >>> p = pl.plot(mus,Imus,'r-',lw=2)
    
    
    
    @return: coefficients, sum of squared residuals,relative flux difference between prediction and model integrated intensity
    @rtype: array, float, float
    """
    def ldres_fmin(coeffs, mu, Imu, law):
        return sum((Imu - globals()['ld_%s'%(law)](mu,coeffs))**2)
    
    def ldres_leastsq(coeffs, mu, Imu, law):
        return Imu - globals()['ld_%s'%(law)](mu,coeffs)
    
    #-- prepare array for coefficients and set the initial guess
    Ncoeffs = dict(claret=4,linear=1,nonlinear=2,logarithmic=2,quadratic=2,
                   power=1)
    c0 = np.zeros(Ncoeffs[law])
    c0[0] = 0.6
    #-- do the fitting
    if fitmethod=='leastsq':
        (csol, ierr)  = leastsq(ldres_leastsq, c0, args=(mu,Imu,law))
    elif fitmethod=='fmin':
        csol  = fmin(ldres_fmin, c0, maxiter=1000, maxfun=2000,args=(mu,Imu,law),disp=0)
    elif fitmethod=='equidist_mu_leastsq':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order],Imu[mu_order],s=0.0, k=2)
        mu_spl = np.linspace(mu[mu_order][0],1,5000)
        Imu_spl = splev(mu_spl,tck,der=0)    
        (csol, ierr)  = leastsq(ldres_leastsq, c0, args=(mu_spl,Imu_spl,law))
    elif fitmethod=='equidist_r_leastsq':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order],Imu[mu_order],s=0., k=2)
        r_spl = np.linspace(mu[mu_order][0],1,5000)
        mu_spl = np.sqrt(1-r_spl**2)
        Imu_spl = splev(mu_spl,tck,der=0)    
        (csol,ierr)  = leastsq(ldres_leastsq, c0, args=(mu_spl,Imu_spl,law))
    elif fitmethod=='equidist_mu_fmin':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order],Imu[mu_order],k=2, s=0.0)
        mu_spl = np.linspace(mu[mu_order][0],1,5000)
        Imu_spl = splev(mu_spl,tck,der=0)
        csol  = fmin(ldres_fmin, c0, maxiter=1000, maxfun=2000,args=(mu_spl,Imu_spl,law),disp=0)
    elif fitmethod=='equidist_r_fmin':
        mu_order = np.argsort(mu)
        tck = splrep(mu[mu_order],Imu[mu_order],k=2, s=0.0)
        r_spl = np.linspace(mu[mu_order][0],1,5000)
        mu_spl = np.sqrt(1-r_spl**2)
        Imu_spl = splev(mu_spl,tck,der=0)
        csol  = fmin(ldres_fmin, c0, maxiter=1000, maxfun=2000,args=(mu_spl,Imu_spl,law),disp=0)
    else:
        raise ValueError("Fitmethod {} not recognised".format(fitmethod))
    myfit = globals()['ld_%s'%(law)](mu,csol)
    res =  np.sum(Imu - myfit)**2
    int1,int2 = np.trapz(Imu,x=mu),np.trapz(myfit,x=mu)
    dflux = (int1-int2)/int1
    return csol,res,dflux
#}
#{ LD Passband coefficients
def choose_ld_coeffs_table(atm,atm_kwargs={},red_kwargs={}):
    """
    Derive the filename of a precalculated LD grid from the input parameters.
   
    """
    #-- perhaps the user gave a filename: then return it
    if os.path.isfile(atm):
        return atm
    elif os.path.isfile(os.path.join(basedir_ld_coeffs,atm)):
        return os.path.join(basedir_ld_coeffs,atm)
    #-- if the user wants tabulated blackbodies, we have a file for that.
    elif atm=='blackbody':
        basename = 'blackbody_uniform_none_teff.fits'
        return os.path.join(basedir_ld_coeffs,basename)
    #-- else we need to be a little bit more clever and derive the file with
    #   the tabulated values based on abundance, LD func etc...
    else:
        #-- get some basic info
        abun = atm_kwargs['abun']
        ld_func = atm_kwargs['ld_func']
        ld_coeffs = atm_kwargs['ld_coeffs']
        #-- if the LD is uniform or given by the user itself, we're only
        #   interested in the center intensities, so we can use the default
        #   grid
        if ld_func=='uniform' or not isinstance(ld_coeffs,str):
            ld_func = 'claret'
        #-- do we need to interpolate in abundance?
        if hasattr(abun,'__iter__'):
            if np.all(abun==abun[0]):
                prefix = 'm' if abun[0]<0 else 'p'
                abun = abs(abun[0])*10
                basename = '{}_{}{:02.0f}_{}_equidist_r_leastsq_teff_logg.fits'.format(atm,prefix,abun,ld_func)
            else:
                prefix = ''
                raise ValueError("Cannot automatically detect atmosphere file")
        else:
            prefix = 'm' if abun<0 else 'p'
            abun = abs(abun*10)
            basename = '{}_{}{:02.0f}_{}_equidist_r_leastsq_teff_logg.fits'.format(atm,prefix,abun,ld_func)        
        ret_val = os.path.join(basedir_ld_coeffs,basename)
        if os.path.isfile(ret_val):
            return ret_val
        else:
            raise ValueError("Cannot interpret atm parameter {}: I think the file that I need is {}, but it doesn't exist. If in doubt, consult the installation section of the documentation on how to add atmosphere tables.".format(atm,ret_val))
    return atm
    
def interp_ld_coeffs(atm,passband,atm_kwargs={},red_kwargs={},vgamma=0):
    """
    Interpolate an atmosphere table.
    
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
    #-- get the FITS-file containing the tables
    #   retrieve structured information on the grid (memoized)
    atm = choose_ld_coeffs_table(atm,atm_kwargs=atm_kwargs,red_kwargs=red_kwargs)
    axis_values,pixelgrid,labels = _prepare_grid(passband,atm)
    #-- prepare input: the variable "labels" contains the name of all the
    #   variables (teff, logg, ebv, etc... which can be interpolated. If all
    #   z values for example are equal, we do not need to interpolate in
    #   metallicity, and _prepare_grid will have removed the 'z' label from
    #   that list. In the following 3 lines of code, we collect only those
    #   parameters which need to be interpolated in the grid. Beware that this
    #   means that the z variable will simply be ignored! If you ask for z=0.1
    #   but all values in the grid are z=0.0, this function will not complain
    #   and just give you the z=0.0 interpolation results!
    N = 1
    for i,label in enumerate(labels):
        if label in atm_kwargs and hasattr(atm_kwargs[label],'__len__'):
            N = max(N,len(atm_kwargs[label]))
        elif label in red_kwargs and hasattr(red_kwargs[label],'__len__'):
            N = max(N,len(red_kwargs[label]))
        elif label=='vgamma' and hasattr(vgamma,'__len__'):
            N = max(N,len(vgamma))
        #else:
        #    raise ValueError("Somethin' wrong with the atmo table: ")
    values = np.zeros((len(labels),N))
    for i,label in enumerate(labels):
        #-- get the value from the atm_kwargs or red_kwargs
        if label in atm_kwargs:
            values[i] = atm_kwargs[label]
        elif label in red_kwargs:
            values[i] = red_kwargs[label]
        elif label=='vgamma':
            values[i] = vgamma
        else:
            raise ValueError("Somethin' wrong with the atmo table: cannot interpret label {}".format(label))
    #-- try to interpolate
    try:
        pars = interp_nDgrid.interpolate(values,axis_values,pixelgrid)
        if np.any(np.isnan(pars)):
            raise IndexError
    #-- things can go outside the grid
    except IndexError:
        msg = ", ".join(['{:.3f}<{}<{:.3f}'.format(values[i].min(),labels[i],values[i].max()) for i in range(len(labels))])
        msg = "Parameters outside of grid {}: {}. Consider using a different atmosphere/limbdarkening grid, or use the black body approximation.".format(atm,msg)
        #raise IndexError(msg)
        logger.error(msg)
        pars = np.zeros((pixelgrid.shape[-1],len(values[0])))
    pars[-1] = 10**pars[-1]
    return pars

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
    
        >>> atm_kwargs = dict(teff=6000.,logg=4.0,z=0.)
        >>> interp_ld_coeffs_wd('atmcof.dat','V',atm_kwargs=atm_kwargs)
        
    Remarks:
    
        - reddening (C{red_kwargs}) is not implemented
        - doppler beaming (C{vgamma}) is not implemented
    
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
    #-- get atmospheric properties
    m = atm_kwargs.get('abun',0)
    l = atm_kwargs.get('logg',4.5)
    t = atm_kwargs.get('teff',10000)
    p = passband
    
    #-- prepare lists for interpolation
    M = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
         -0.5,-0.3,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    L = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    P = ['STROMGREN.U','STROMGREN.V','STROMGREN.B','STROMGREN.Y',
         'JOHNSON.U','JOHNSON.B','JOHNSON.V','JOHNSON.R','JOHNSON.I',
         'JOHNSON.J','JOHNSON.K','JOHNSON.L','JOHNSON.M','JOHNSON.N',
         'COUSINS.R','COUSINS.I','OPEN.BOL',None,None,None,None,None,
         'TYCHO2.BT','TYCHO2.VT','HIPPARCOS.HP']
    
    #-- get the atmosphere table's location and read it in.
    if not os.path.isfile(atm):
        atm = os.path.join(basedir_ld_coeffs,atm)
    table = _prepare_wd_grid(atm)
    
    #-- find out where we need to be in the atmosphere table and extract that
    #   row
    index1 = 18-np.searchsorted(M,m)
    index2 = np.searchsorted(L,l)
    idx = index1*len(P)*len(L)*4 + P.index(p)*len(L)*4 + index2*4
    Cl2 = table[idx+1,2:]
    
    #-- calculate the Legendre temperature
    teff = (t-table[idx+1,0])/(table[idx+1,1]-table[idx+1,0])
    Pl = np.array(legendre(teff))
    
    #-- and compute the flux
    s = np.sum(Cl2.reshape((-1,1))*Pl,axis=0)
    
    #-- that's it!
    return 10**s * 1e-8

@decorators.memoized
def _prepare_wd_grid(atm):
    logger.info("Prepared WD grid {}: interpolate in teff, logg, abun".format(os.path.basename(atm)))
    return np.loadtxt(atm)

@decorators.memoized
def _prepare_grid(passband,atm):
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
    #-- data columns are C{ai} columns with C{i} an integer, plus C{Imu1} that
    #   holds the centre-of-disk intensity
    data_columns = ['a{:d}'.format(i) for i in range(1,10)] + ['imu1']
    #-- some columns are transformed to log for interpolation, because the
    #   data behaves more or less linearly in log scale.
    log_columns = ['imu1']#,'Teff']
    #-- not all columns hold data that needs to be interpolated
    nointerp_columns = data_columns + ['res','dflux']
    with pyfits.open(atm) as ff:
        try:
            available = [col.lower() for col in ff[passband].data.names]
        except Exception as msg:
            raise KeyError("Atmosphere file {} does not contain required information ({})".format(atm,str(msg)))
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
    logger.info("Prepared grid {}: interpolate in {}".format(os.path.basename(atm),", ".join(labels)))
            
    #-- create the pixeltype grid
    axis_values, pixelgrid = interp_nDgrid.create_pixeltypegrid(grid_pars,grid_data)
    return axis_values,pixelgrid,labels

#}

#{ Grid creators

def compute_grid_ld_coeffs(atm_files,atm_pars,\
                           red_pars_iter={},red_pars_fixed={},vgamma=None,\
                           passbands=('JOHNSON.V',),\
                           law='claret',fitmethod='equidist_r_leastsq',\
                           filename=None,filetag='kurucz'):
    """
    Create an interpolatable grid of limb darkening coefficients.
    
    Example usage::
        
        >>> atm_files = ['spec_intens/kurucz_mu_ip00k2.fits']
        >>> limbdark.compute_grid_ld_coeffs(atm_files,atm_pars=('teff','logg'),
              law='claret',passbands=['2MASS.J','JOHSON.V'],fitmethod='equidist_r_leastsq',filetag='kurucz_p00')
    
    This will create a FITS file, and this filename can be used in the
    C{atm} and C{ld_coeffs} parameters in Phoebe.
    """
    overwrite = None
    #-- OPEN.BOL must always be in there:
    passbands = sorted(list(set(list(passbands)+['OPEN.BOL'])))
    print("Creating grid from {} in passbands {}".format(", ".join(atm_files),", ".join(passbands)))
    if filename is None:
        filename = '{}_{}_{}_'.format(filetag,law,fitmethod)
        filename+= '_'.join(atm_pars)
        if red_pars_iter:
            filename += '_' + '_'.join(sorted(list(red_pars_iter.keys())))
        for key in red_pars_fixed:
            filename += '_{}{}'.format(key,red_pars_fixed[key])
        if vgamma is not None:
            filename += '_vgamma'
        filename += '.fits'
    #-- if the file already exists, we'll append to it; so don't do any work
    #   twice (but first backup it!)
    if os.path.isfile(filename):
        #-- safely backup existing file if needed
        if os.path.isfile(filename+'.backup'):
            if overwrite is None:
                    answer = raw_input('Backup file for {} already exists. Overwrite? (Y/n): '.format(filename))
                    if answer=='n': overwrite = False
                    elif answer=='y': overwrite = None
                    elif answer=='Y' or answer=='': overwrite = True
                    else:
                        raise ValueError("option '{}' not recognised".format(answer))
            if overwrite is True:
                shutil.copy(filename,filename+'.backup')
        #-- sort out which passbands are already computed
        with pyfits.open(filename) as ff:
            existing_passbands = [ext.header['extname'] for ext in ff[1:]]
        logger.info("Skipping passbands {} (they already exist)".format(set(passbands) & set(existing_passbands)))
        passbands = sorted(list(set(passbands)-set(existing_passbands)))
    else:
        logger.info("Gridfile '{}' does not exist yet".format(filename))
    #-- collect parameter names and values
    atm_par_names = list(atm_pars)
    red_par_names = sorted(list(red_pars_iter.keys()))
    other_pars = [red_pars_iter[name] for name in red_par_names]
    if vgamma is not None:
        other_pars.append(vgamma)
    #-- prepare to store the results: for each passband, we need an array.
    #   The rows of that array will be the grid points + coefficients +
    #   fit statistics
    output = {passband:[] for passband in passbands}
    for atm_file in atm_files:
        #-- special case if blackbody: we need to build our atmosphere model
        if atm_file=='blackbody':
            wave_ = np.logspace(2,5,10000)
        for nval,val in enumerate(iter_grid_dimensions(atm_file,atm_par_names,other_pars)):
            print(atm_file,val)
            
            #-- the first values are from the atmosphere grid
            atm_kwargs = {name:val[i] for i,name in enumerate(atm_par_names)}
            val_ = val[len(atm_par_names):]
            #-- the next values are from the reddening
            red_kwargs = {name:val_[i] for i,name in enumerate(red_par_names)}
            #   but we need to also include the fixed ones
            for key in red_pars_fixed:
                red_kwargs[key] = red_pars_fixed[key]
            val_ = val_[len(red_par_names):]
            #-- perhaps there was vrad information too.
            if val_:
                vgamma = val_[0]
            #-- retrieve the limb darkening law
            if atm_file!='blackbody':
                mu,Imu = get_limbdarkening(atm_file,atm_kwargs=atm_kwargs,
                                       red_kwargs=red_kwargs,vgamma=vgamma,
                                       passbands=passbands)
                print(atm_file,atm_kwargs)
            elif law=='uniform':
                Imu_blackbody = sed.blackbody(wave_,val[0],vrad=vgamma)
                Imu = sed.synthetic_flux(wave_*10,Imu_blackbody,passbands)
            else:
                raise ValueError("Blackbodies must have a uniform LD law")
                
            if law!='uniform':
                for i,pb in enumerate(passbands):
                    #-- fit a limbdarkening law:
                    csol,res,dflux = fit_law(mu,Imu[:,i]/Imu[0,i],law=law,fitmethod=fitmethod)
                    output[pb].append(list(val) + [res,dflux] + list(csol) + [Imu[0,i]])
                    print(pb, output[pb][-1])
            else:
                csol = []
                for i,pb in enumerate(passbands):
                    #-- don't fit a law, we know what it is
                    output[pb].append(list(val) + [0.0,0.0] + [Imu[i]])
                    print(pb, output[pb][-1])
    #-- write to a FITS file
    col_names = atm_par_names + red_par_names
    if vgamma is not None and 'vgamma' not in col_names:
            col_names.append('vgamma')
    col_names = col_names + ['res','dflux'] + ['a{:d}'.format(i+1) for i in range(len(csol))] + ['Imu1']
    #-- if the file already exists, append to it
    if os.path.isfile(filename):
        filename = pyfits.open(filename,mode='update')
    #-- append all the tables
    for pb in sorted(list(output.keys())):
        grid = np.array(output[pb]).T
        header = dict(extname=pb)
        filename = fits.write_array(grid,filename,col_names,header_dict=header,ext='new',close=False)
    filename[0].header.update('HIERARCH FITMETHOD',fitmethod,'method used to fit the LD coefficients')
    filename[0].header.update('LD_FUNC',law,'fitted limb darkening function')
    for key in red_pars_fixed:
        filename[0].header.update('HIERARCH '+key,red_pars_fixed[key],'fixed reddening parameter')
        
    #-- copy keys from first atm_file
    if atm_files[0]!='blackbody':
        with pyfits.open(atm_files[0]) as gg:
            for key in gg[0].header.keys():
                if key in ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND']:
                    continue
                filename[0].header.update(key,gg[0].header[key],'key from first atm_file')
        
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
    
    >>> _I_ls(0,0.5)
    0.6666666666666666
    >>> _I_ls(1,0.5)
    0.4
    >>> _I_ls(2,0.5)
    0.09523809523809523
    """
    if ll==0:
        return 1./(1.+ss)
    elif ll==1:
        return 1./(2.+ss)
    else:
        return (ss-ll+2)/(ss+ll-2+3.)*_I_ls(ll-2,ss)        
    

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
    radius = body.request_value('radius','cm')
    abun = body['abun']
    #vrad += gravitational_redshift(body_parameter_set)
    if body.has_qualifier('distance'):
        distance = body.request_value('distance','cm')
    elif body.has_qualifier('angdiam'):
        angdiam = body.request_value('angdiam','sr')
        distance = radius/np.sqrt(angdiam)
    else:
        #distance = conversions.convert('pc','cm',10.)
        distance = 3.0856775815030002e+19
    #-- retrieve the parameters of the data
    ld_func = pbdep['ld_func']
    atm = pbdep['atm']
    ld_coeffs = pbdep['ld_coeffs']
    passband = pbdep['passband']
    if not ld_func=='claret':
        logger.warning('analytical computation of sphere intensity with LD model %s not implemented yet'%(ld_func))
        return 0.,0.
    #-- retrieve the limbdarkening coefficients when they need to be looked up
    #   in a table: if the given coefficients are a string, assume it is a
    #   reference to a table. Otherwise, just use the coefficients.
    if isinstance(ld_coeffs,str):
        atm_kwargs = dict(atm=atm,ld_func=ld_func,teff=teff,logg=logg,abun=abun,ld_coeffs=ld_coeffs)
        ld_coeffs = interp_ld_coeffs(ld_coeffs,passband,atm_kwargs=atm_kwargs,red_kwargs=red_kwargs)[:,0]
    #-- we compute projected and total intensity. We have to correct for solid
    #-- angle, radius of the star and distance to the star.
    theta1 = 2*np.pi*radius**2*4*np.pi
    theta2 = 2*np.pi*radius**2/distance**2
    int_moment = intensity_moment(ld_coeffs,ll=0)
    #4*pi*radius**2*constants.sigma_cgs*teff**4,\
    return int_moment*theta1,\
           int_moment*theta2

#}

#{ Phoebe interface

def local_intensity(system,parset_pbdep,parset_isr={}):
    """
    Calculate local intensity.
    
    Small but perhaps important note: we do not take reddening into account
    for OPEN.BOL calculations, if the reddening is interstellar.
    """
    #-- get the arguments we need
    atm = parset_pbdep['atm']
    ld_coeffs = parset_pbdep['ld_coeffs']
    ld_func = parset_pbdep['ld_func']
    passband = ('passband' in parset_pbdep) and parset_pbdep['passband'] or 'OPEN.BOL'
    include_vgamma = ('beaming' in parset_pbdep) and parset_pbdep['beaming'] or False
    ref = ('ref' in parset_pbdep) and parset_pbdep['ref'] or '__bol'
    #-- compute local intensity
    #-- radial velocity needs to be in km/s, while the natural units in the 
    #   Universe are Rsol/d. If vrad needs to be computed, we'll also include
    #   gravitational redshift
    #vrad = conversions.convert('Rsol/d','km/s',system.mesh['velo___bol_'][:,2])
    vrad = 8.04986111111111*system.mesh['velo___bol_'][:,2]
    if include_vgamma:
        vrad += 0. # tools.gravitational_redshift
    else:
        vrad *= 0.
    #-- in the following we need to take care of a lot of possible inputs by
    #   the user, and some possible simplifications for the calculations
    #   First of all, it is possible that the whole star has the same local
    #   parameters. In that case, we compute the limb darkening coefficients
    #   only for one surface element and then set all surface elements to be
    #   the same.
    #   Basically, we need to compute intensities and limb darkening
    #   coefficients. In principle, they could be derived from different sets
    #   of atmosphere models, or, the LD coeffs could be set manually to be
    #   constant over the whole star while the intensities are varied over the
    #   surface.
    tag = 'ld_'+ref
    log_msg = "{:s}({:s}): ".format(passband,tag[3:])
    #-- we need parameters on the passband and reddening in the form of a
    #   dictionary. We may add too much information, since the function
    #   "interp_ld_coeffs" only extract those that are relevant.
    atm_kwargs = dict(parset_pbdep)
    #-- No reddening for bolometric fluxes!
    if ref=='__bol':# and parset_isr.get_context()=='reddening:interstellar':
        red_kwargs = {}
        logger.info("Not propagating interstellar reddening info for {} (taking default from grid)".format(ref))
    else:
        red_kwargs = dict(parset_isr)
    #-- This is what I was talking about: maybe everything has the same teff and
    #   logg? We'll use the 'allclose' function, which takes care of small
    #   numerical machine-precision variations
    uniform_pars = np.allclose(system.mesh['teff'],system.mesh['teff'][0]) and \
                   np.allclose(system.mesh['logg'],system.mesh['logg'][0]) and \
                   np.allclose(system.mesh['abun'],system.mesh['abun'][0]) and \
                   np.allclose(vrad,vrad[0])
    ld_coeffs_from_grid = isinstance(ld_coeffs,str)
    if not ld_coeffs_from_grid:
        system.mesh[tag][:,:len(ld_coeffs)] = np.array(ld_coeffs)
    if uniform_pars:
        #-- take the first surface element as the representative one
        atm_kwargs['teff'] = system.mesh['teff'][:1]
        atm_kwargs['logg'] = system.mesh['logg'][:1]
        atm_kwargs['abun'] = system.mesh['abun'][:1]
        vgamma = vrad[:1]
        log_msg += '(single faces) (vgamma={})'.format(vgamma)
    else:
        atm_kwargs['teff'] = system.mesh['teff']
        atm_kwargs['logg'] = system.mesh['logg']
        atm_kwargs['abun'] = system.mesh['abun']
        vgamma = vrad
        log_msg += '(multiple faces) (vgamma_mean={})'.format(vgamma.mean())
    #-- what kind of atmosphere is used to compute the limb darkening
    #   coefficients? They come from a model atmosphere (ld_coeffs is a filename),
    #   or they are a list of LD coeffs, assumed to be constant over the
    #   surface. If a filename is given, we immediately also set the
    #   intensities of each surface element. If the user wants different
    #   stuff, we'll take care of that in the next few lines.
    if ld_coeffs_from_grid:
        log_msg += ', LD via table ld_coeffs={:s}'.format(os.path.basename(ld_coeffs) )
        coeffs = interp_ld_coeffs(ld_coeffs,passband,atm_kwargs=atm_kwargs,
                                       red_kwargs=red_kwargs,vgamma=vgamma).T
        system.mesh[tag][:,:coeffs.shape[1]-1] = coeffs[:,:-1]
        system.mesh[tag][:,-1] = coeffs[:,-1]
    else:
        log_msg += ', LD via coeffs {}'.format(str(ld_coeffs))
        system.mesh[tag][:,:len(ld_coeffs)] = np.array(ld_coeffs)
    #-- what kind of atmosphere is used to compute the intensities? Black
    #   bodies we do on the fly, which means there are no limits on the
    #   effective temperature used. Note that black bodies are not 
    #   sensitive to logg.
    if atm=='true_blackbody':
        wave_ = np.logspace(2,5,10000)
        log_msg += ', intens via atm=true_blackbody'
        if uniform_pars:
            Imu_blackbody = sed.blackbody(wave_,atm_kwargs['teff'][0],vrad=vgamma)
            system.mesh[tag][:,-1] = sed.synthetic_flux(wave_*10,Imu_blackbody,[passband])[0]
        else:
            for i,T in enumerate(atm_kwargs['teff']):
                Imu_blackbody = sed.blackbody(wave_,T,vrad=vgamma[i])
                system.mesh[tag][i,-1] = sed.synthetic_flux(wave_*10,Imu_blackbody,[passband])[0]
    #-- remember that if the ld_coeffs was a string, we already set the intensities
    elif not ld_coeffs_from_grid or atm!=ld_coeffs:
        log_msg += ', intens via table atm={:s}, '.format(os.path.basename(atm))
        #-- WD compatibility layer:
        if os.path.splitext(atm)[1]=='.dat':
            system.mesh[tag][:,-1] = interp_ld_coeffs_wd(atm,passband,atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs,vgamma=vgamma)
        #-- Phoebe layer
        else:
            system.mesh[tag][:,-1] = interp_ld_coeffs(atm,passband,atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs,vgamma=vgamma)[-1]
    #-- else, we did everything already in the "if ld_coeffs_from_grid" part
    logger.info(log_msg)
    # Optional consistency check:
    if True:
        if np.any(np.isnan(system.mesh[tag])):
            raise ValueError("Value outside of grid: check if the surface gravity, temperature etc fall inside the grids. Perhaps if you have an eccentric orbit, the star is outside of the grid only for some phases.")



def projected_intensity(system,los=[0.,0.,+1],method='numerical',ld_func='claret',
                        ref=0,with_partial_as_half=True):
    """
    Calculate local projected intensity.
    
    We can speed this up if we compute the local intensity first, keep track of
    the limb darkening coefficients and evaluate for different angles. Then we
    only have to do a table lookup once.
    
    Analytical calculation can also be an approximation!
    
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
    body = system.params.values()[0]
    if method=='numerical':
        #-- get limb angles
        mus = system.mesh['mu']
        #-- To calculate the total projected intensity, we keep track of the
        #   partially visible triangles, and the totally visible triangles
        #   (the correction for nans in the size of the faces is probably due
        #   to a bug in the marching method, though I'm not sure):
        if np.any(np.isnan(system.mesh['size'])):
            print('encountered nan')
            raise SystemExit
        keep = (mus>0) & (system.mesh['partial'] | system.mesh['visible'])# & -np.isnan(self.mesh['size'])
        mus = mus[keep]
        #-- negating the next array gives the partially visible things, that is
        #   the only reason for defining it.
        visible = system.mesh['visible'][keep]
        #-- compute intensity using the already calculated limb darkening coefficents
        logger.info('using limbdarkening law %s'%(ld_func))
        Imu = globals()['ld_{}'.format(ld_func)](mus,system.mesh['ld_'+ref][keep].T)*system.mesh['ld_'+ref][keep,4]
        proj_Imu = mus*Imu
        if with_partial_as_half:
            proj_Imu[-visible] /= 2.0
        system.mesh['proj_'+ref] = 0.
        system.mesh['proj_'+ref][keep] = proj_Imu
        #-- take care of reflected light
        if 'refl_'+ref in system.mesh.dtype.names:
            proj_Imu += system.mesh['refl_'+ref][keep]
            logger.info("Projected intensity contains reflected light")
        proj_intens = system.mesh['size'][keep]*proj_Imu        
        #-- we compute projected and total intensity. We have to correct for solid
        #-- angle, radius of the star and distance to the star.
        distance = body.request_value('distance','Rsol')
        proj_intens = proj_intens.sum()/distance**2
    #-- analytical computation
    elif method=='analytical':
        lcdep,ref = system.get_parset(ref)
        proj_intens = sphere_intensity(body,lcdep)[1]
    return proj_intens    
    

def projected_velocity(system,los=[0,0,+1],method='numerical',ld_func='claret',ref=0):
    """
    Calculate mean local projected velocity.
    
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
    if np.all(proj_intens==0):
        proj_velo = 0.
    else:
        #proj_velo = np.average(-system.mesh['velo_'+ref+'_'][keep,2],weights=proj_intens)
        proj_velo = np.average(-system.mesh['velo___bol_'][keep,2],weights=proj_intens)
        logger.info("projected velocity with ld_func %s = %.3g Rsol/d"%(ld_func,proj_velo))
    return proj_velo

    
#}
    
if __name__=="__main__":
    import doctest
    fails,tests = doctest.testmod()
    if not fails:
        print(("All {0} tests succeeded".format(tests)))
    else:
        print(("{0}/{1} tests failed".format(fails,tests)))
    