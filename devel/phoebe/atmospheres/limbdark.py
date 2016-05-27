

from phoebe.utils import config, decorators
from phoebe.algorithms import interp_nDgrid
from phoebe import u


import os
import tarfile
import urllib
import numpy as np

try:
    import pyfits
except:
    try: # Pyfits now integrated in astropy
        import astropy.io.fits as pyfits
    except ImportError:
        print(("Soft warning: pyfits could not be found on your system, you can "
           "only use black body atmospheres and Gaussian synthetic spectral "
           "profiles"))


import logging
logger = logging.getLogger("LIMBDARK")
logger.addHandler(logging.NullHandler())


basedir = os.path.dirname(os.path.abspath(__file__))
basedir_spec_intens = os.path.join(basedir, 'tables', 'spec_intens')
basedir_ld_coeffs = os.path.join(basedir, 'tables', 'ld_coeffs')

_prsa_ld = np.linspace(0, 1, 32)
_prsa_res = _prsa_ld[0] - _prsa_ld[1]

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
    """
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

def disk_power(coeffs):
    logger.warning('disk_power not well implemented, luminosities will be off')
    return np.pi



def local_intensity(body, dataset, ld_coeffs, ld_func, atm, passband='OPEN.BOL', boosting_alg='none', **kwargs):
    """
    Calculate local intensity.
    
    Attempt to generalize first version.
    
    This is the work horse for Phoebe concerning atmospheres.
    
    Small but perhaps important note: we do not take reddening into account
    for OPEN.BOL calculations, if the reddening is interstellar.
    
    boosting options:
    
        - :envvar:`boosting_alg='full'`: intensity and limb darkening coefficients
          are corrected for local velocity
        - :envvar:`boosting_alg='local'`: local linear boosting coefficient is added
          to the mesh
        - :envvar:`boosting_alg='simple': global linear boosting coefficients is added
          to the mesh
        - :envvar:`boosting_alg='none'` or boosting parameter evaluates to False:
          boosting is not taken into account
    """
    
    # Doppler boosting: include it if algorithm is "full"
    include_vgamma = boosting_alg in ['full']

    # Radial velocity needs to be in km/s, while the natural units in the 
    # Universe are Rsol/d. If vrad needs to be computed, we'll also include
    # gravitational redshift
    #vrad = conversions.convert('Rsol/d','km/s',body.mesh['velo___bol_'][:,2])
    vgamma = (body.mesh['velo___bol_'][:,2]*u.Unit('solRad/d')).to('km/s').value
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
    tag = 'ld_' + dataset
    log_msg = "{:s}({:s}):".format(passband, dataset)
    
    # We need parameters on the passband and reddening in the form of a
    # dictionary. We may add too much information, since the function
    # "interp_ld_coeffs" only extracts those that are relevant.
    atm_kwargs = kwargs # TODO: this may send to much?? (changed from alpha->beta)
    
    # No reddening for bolometric fluxes!
    # TODO: implement interstellar reddening
    logger.warning('interstellar reddening not yet ported to beta')
    red_kwargs = {}
    # if dataset == '__bol':
    #     red_kwargs = {}
    #     logger.info(("Not propagating interstellar reddening info for bolometric flux "
    #                 "(taking default from grid)"))
    # else:
    #     red_kwargs = dict(parset_isr)
    
    if not ld_coeffs == 'prsa':
        # 1. Easiest case: atm_file and ld_coeffs file are consistent
        if atm == ld_coeffs and atm in config.atm_props:
            
            # Find the possible interpolation parameters
            atm_kwargs = {key:body.mesh[key] for key in config.atm_props[atm]}
            fitmethod = config.fit_props.get(atm, 'equidist_r_leastsq')
            
            # Get the atmosphere file
            atm_file = choose_ld_coeffs_table(atm, atm_kwargs=atm_kwargs,
                                        red_kwargs=red_kwargs, vgamma=vgamma,
                                        ld_func=ld_func,
                                        fitmethod=fitmethod)
            
            # Interpolate the limb darkening coefficients
            coeffs = interp_ld_coeffs(atm_file, passband, atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs, vgamma=vgamma)
            
            # Fill in the LD coefficients, but put the intensities in the last
            # column
            for collnr, coefcoll in enumerate(coeffs[:-1]):
                body.mesh[tag][:, collnr] = coefcoll
            body.mesh[tag][:, -1] = coeffs[-1]
            
            log_msg += ' atm and ld_coeffs from table {}'.format(os.path.basename(atm_file))
        
        # 2. atm_file and ld_coeffs file are not the same
        elif atm in config.atm_props:
            
            # First the local normal emergent intensities.
            # Find the possible interpolation parameters
            atm_kwargs = {key:body.mesh[key] for key in config.atm_props[atm]}
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
                fitmethod_ldc = config.fit_props.get(ld_coeffs, 'equidist_r_leastsq')
                ldc_file = choose_ld_coeffs_table(ld_coeffs, atm_kwargs=atm_kwargs,
                                        red_kwargs=red_kwargs, vgamma=vgamma,
                                        ld_func=ld_func,
                                        fitmethod=fitmethod_ldc) 
            
                coeffs_ldc = interp_ld_coeffs(ldc_file, passband, atm_kwargs=atm_kwargs,
                                            red_kwargs=red_kwargs, vgamma=vgamma)[:-1]
                
                log_msg += ', ld_coeffs from table {}'.format(os.path.basename(ldc_file))
                
            # Otherwise it's a list or so, and we don't need to interpolate
            except TypeError:
                coeffs_ldc = ld_coeffs
                
                log_msg += ', manual ld_coeffs ({})'.format(ld_coeffs)
            
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
                body.mesh[tag][:, collnr] = coefcoll
            body.mesh[tag][:, -1] = coeffs_atm[-1] * correction_factor
            
        # 3. True Blackbody we'll treat separately. There are no limits on the
        # effective temperature used. Note that black bodies are not sensitive to
        # logg (or anything else for that matter).
        elif atm == 'true_blackbody':
            
            if boosting_alg and boosting_alg != 'full':
                raise ValueError("With atm=true_blackbody you can only use boosting_alg='full' or boosting_alg='none'.")
            
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
                body.mesh[tag][:, -1] = sed.synthetic_flux(wave_*10,
                                            Imu_blackbody, [passband])[0]
            
            # What the frack? You must be kidding... OK, here we go, creating
            # black bodies and integrating them over the passband as we go...
            # Imagine doing this for several light curves... Djeez.
            else:
                for i,T in enumerate(atm_kwargs['teff']):
                    Imu_blackbody = sed.blackbody(wave_, T,
                                        vrad=vgamma[i])
                    body.mesh[tag][i,-1] = sed.synthetic_flux(wave_*10,
                                                    Imu_blackbody, [passband])[0]
        
        # 4. Wilson Devinney compatibility layer
        elif os.path.splitext(atm)[1] == '.dat':
            atm_kwargs = {key:body.mesh[key] for key in config.atm_props['wd']}
            print(atm_kwargs)
            body.mesh[tag][:, -1] = interp_ld_coeffs_wd(atm, passband,
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
    if 'pbgravb' in kwargs:
        pbgravb = kwargs['pbgravb']
        # we need a reference point for scaling the intensities, we choose the
        # point with maximum surface gravity (likely the pole)
        g = 10**body.mesh['logg']
        index = np.argmax(g)
        Imax = body.mesh[tag][index,-1]
        body.mesh[tag][index,-1] = (g/g[index])**pbgravb * Imax
    
    # 5. Take care of boosting option
    if boosting_alg == 'simple':
        alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma,
                                      interp_all=False)
        body.mesh['alpha_b_' + ref] = alpha_b
        log_msg += ' (simple boosting)'
    
    elif boosting_alg == 'local':
        alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma)
        body.mesh['alpha_b_' + ref] = alpha_b
        log_msg += ' (local boosting)'
        
    elif boosting_alg == 'global':
        alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                      red_kwargs=red_kwargs, vgamma=vgamma)
        body.mesh['alpha_b_' + ref] = alpha_b
        log_msg += ' (global boosting (currently still implemented as local))'
    
    elif boosting_alg == 'none' or not boosting_alg:
        log_msg += " (no boosting)"
    
    logger.info(log_msg)


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
        # boosting: only if the velocity is not zero   
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
                              "atmosphere tables. If you don't have the standard "
                              "atmosphere tables yet, you might first want to try:\n"
                              "$:> python -c 'import phoebe; phoebe.download_atm()'".format(atm, ret_val)))
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
    logger.debug("Range of values requested:"+','.join(['{:.3f}<{}<{:.3f}'.format(values[i].min(), labels[i],\
                                 values[i].max()) for i in range(len(labels))]))
    # Try to interpolate
    try:
        #if use_new_interp_mod:
        #pars = interp_nDgrid.cinterpolate(values.T, axis_values, pixelgrid).T
        #else:
        pars = interp_nDgrid.interpolate(values, axis_values, pixelgrid, order=order, mode=mode, cval=cval)
        pars[-1][np.isinf(pars[-1])] = 0.0
        if safe and np.any(np.isnan(pars[-1])):# or np.any(np.isinf(pars[-1])):
            raise IndexError
    
    # Things can go outside the grid
    except IndexError:
        msg = ", ".join(['{:.3f}<{}<{:.3f}'.format(values[i].min(), labels[i],\
                                 values[i].max()) for i in range(len(labels))])
        msg = ("Parameters outside of grid {} (passband {}): {}. Consider "
               "using a different atmosphere/limbdarkening grid, or use the "
               "black body approximation.").format(atm, passband, msg)
        
        if False:
            print(msg)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title(passband)
            with pyfits.open(atm) as ff:
                extensions = [ext.header['extname'] for ext in ff[1:]]
                if not passband in extensions:
                    passband = passband + '_v1.0'
                #plt.plot(ff[passband].data.field('teff'),
                #         ff[passband].data.field('logg'), 'ko')
                print ff[passband].data.dtype.names
                plt.plot(ff[passband].data.field('teff'), ff[passband].data.field('Imu1'), 'ko')
            plt.show()
                
            wrong = np.isnan(np.array(pars)[0])
            #plt.plot(values[0][wrong], values[1][wrong], 'rx', mew=2, ms=10)
            plt.plot(values[0][wrong], 'rx', mew=2, ms=10)
            
            wrong = np.isinf(np.array(pars)[0])
            #plt.plot(values[0][wrong], values[1][wrong], 'ro')
            plt.plot(values[0][wrong], 'ro')
            plt.show()
            plt.xlim(plt.xlim()[::-1])
            plt.ylim(plt.ylim()[::-1])
            plt.show()
        #raise SystemExit
        raise ValueError(msg)
        #logger.error(msg)
        #pars = np.zeros((pixelgrid.shape[-1], values.shape[1]))
    
    # The intensities were interpolated in log, but we want them in linear scale
    pars[-1] = 10**pars[-1]
    # That's it!
    if return_header:
        return pars, header
    else:
        return pars











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
        
        # if this passband is not available, try to add _v1.0 to it
        try:
            this_ext = ff[passband]
        except KeyError:
            passband = passband + '_v1.0'
        
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
            passband_name = ".".join(str(msg).split("\'")[-2].split('.')[:-1])
            extra_msg+= '>>> from phoebe.atmospheres import create_atmospherefits\n>>> create_atmospherefits.compute_grid_ld_coeffs("{}", passbands=("{}",))'.format(atm, passband_name)
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
           os.path.join(basedir, 'tables','spec_intens'),\
               os.path.join(basedir, 'tables','spectra')

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
        print("{}: {}".format(field, dim))
        
            

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





def download_atm(atm=None, force=False):
    
    destin_folder = get_paths()[0]
        
    # Perhaps we need to be sudo?
    print("Downloading/copying atmosphere tables to destination folder {}".format(destin_folder))
    if not os.access(destin_folder, os.W_OK):
        raise IOError(("User has no write priviliges in destination folder, run sudo python "
                        "before calling this function"))
        #output = subprocess.check_call(['sudo']+sys.argv)
    
    if atm is None:
        #/srv/www/phoebe/2.0/docs/_downloads
        source = 'http://www.phoebe-project.org/docs/auxil/ldcoeffs.tar.gz'
        destin = os.path.join(destin_folder, 'ldcoeffs.tar.gz')
        
        if not force and os.path.isfile(destin):
            logger.info("File '{}' already exists, not forcing redownload (force=False)".format(destin))
        
        else:
            try:
                urllib.urlretrieve(source, destin)
                print("Downloaded tar archive from phoebe-project.org")
            except IOError:
                raise IOError(("Failed to download atmosphere file {} to {}. Are you "
                           "connected to the internet? Otherwise, you probably "
                           "need to create atmosphere tables yourself starting "
                           "from the specific intensities)").format(source, destin))            
    
            tar = tarfile.open(destin)
            members = tar.getmembers()
            for member in members:
                print("Extracting {}...".format(member))
            tar.extractall(path=destin_folder)
            tar.close()
    
    else:
        source = 'http://www.phoebe-project.org/docs/auxil/{}'.format(atm)
        destin = os.path.join(destin_folder, atm)

        try:
            urllib.urlretrieve(source, destin)
            print("Downloaded tar archive from phoebe-project.org")
        except IOError:
            raise IOError(("Failed to download atmosphere file {} (you probably "
                           "need to create on yourself starting from the specific "
                           "intensities)").format(atm))


def download_spec_intens(force=False):
    destin_folder = get_paths()[1]
    download(destin_folder,
             'http://www.phoebe-project.org/docs/auxil/spec_intens.tar.gz',
             force=force, estimated_size='580 MB (can take a while!)')
    
def download_spectra(force=False):
    destin_folder = get_paths()[2]
    download(destin_folder,
             'http://www.phoebe-project.org/docs/auxil/spectra.tar.gz',
             force=force, estimated_size='1.3 GB (can take a while!)')

def download(destin_folder, source, force=False, estimated_size=None):
    
    # Perhaps we need to be sudo?
    print("Downloading/copying {} to destination folder {}".format(source, destin_folder))
    if estimated_size:
        print("Estimated size: {}".format(estimated_size))
    if not os.access(destin_folder, os.W_OK):
        raise IOError(("User has no write priviliges in destination folder, run sudo python"
                        "before calling this function"))
        #output = subprocess.check_call(['sudo']+sys.argv)
    
    #/srv/www/phoebe/2.0/docs/_downloads
    destin = os.path.join(destin_folder, os.path.basename(source))
    if not force and os.path.isfile(destin):
        logger.info("File '{}' already exists, not forcing redownload (force=False)".format(destin))
    else:
        try:
            urllib.urlretrieve(source, destin)
            print("Downloaded tar archive from phoebe-project.org")
        except IOError:
            raise IOError(("Failed to download file {} to {}. Are you "
                       " connected to the internet? Otherwise, you probably "
                       "need to create atmosphere tables yourself starting "
                       "from the specific intensities)").format(source, destin))            
    
        tar = tarfile.open(destin)
        members = tar.getmembers()
        for member in members:
            print("Extracting {}...".format(member))
        tar.extractall(path=destin_folder)
        tar.close()
