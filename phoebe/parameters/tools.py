"""
Tools to handle parameters and ParameterSets, and add nonstandard derivative parameters.

**Constraints for the star/component context**

.. autosummary::
    
    add_vsini
    add_rotfreqcrit
    add_vrotcrit
    add_surfgrav
    add_teffpolar
    add_angdiam
    add_radius_eq

**Constraints for orbit context**

.. autosummary::
    
    add_asini
    add_ecosw
    add_esinw
    add_esinw_ecosw
    add_conserve
    make_misaligned
    make_synchronous
    add_theta_eff
    to_supconj
    to_perpass
    
    critical_times
    critical_phases

**Constraints for oscillations**

.. autosummary::

    add_amplvelo
    add_nonadiabatic_coefficients
    add_solarosc
    add_solarosc_Deltanu0
    add_solarosc_numax
    
**Constraints for reddening**

.. autosummary::
    
    add_ebv

**Constraints for global parameters**

.. autosummary::

    add_parallax
    
**Scattering**

.. autosummary::

    add_scattering
    
**Constraints for spectral synthesis**



**Helper functions**

.. autosummary::
    
    list_available_units
    group
    add_unbounded_from_bounded
    
"""
import logging
import os
import numpy as np
import textwrap
from phoebe.units import constants
from phoebe.units import conversions
from phoebe.utils import coordinates
from phoebe.parameters import parameters
from phoebe.dynamics import keplerorbit

logger = logging.getLogger("PARS.TOOLS")

#{ Common constraints for the star class
def add_angdiam(star,angdiam=None,derive='distance',unit='mas',**kwargs):
    r"""
    Add angular diameter to a Star parameterSet.
    
    The photometry is scaled such that
    
    .. math::
    
        F_\mathrm{obs} = F_\mathrm{syn} \dot R_*^2 / d^2
        
    with :math:`R_*` the stellar radius and :math:`d` the distance to the star.
    The angular diameter :math:`\theta` is defined as
    
    .. math::
    
        \theta = 2 R_* / d
        
    And so one of the angular diameter, radius or distance needs to be
    derived from the others.
    
    """
    if kwargs and 'angdiam' in star:
        raise ValueError("You cannot give extra kwargs to add_angdiam if angdiam already exist")
    
    kwargs.setdefault('description','Angular diameter')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on angdiam and add the parameter
    star.pop_constraint('angdiam',None)
    if not 'angdiam' in star:
        star.add(parameters.Parameter(qualifier='angdiam',
                                value=angdiam if angdiam is not None else 0.,
                                **kwargs))
    else:
        star['angdiam'] = angdiam
        
    #-- specify the dependent parameter
    if angdiam is None:
        star.pop_constraint('angdiam',None)
        star.add_constraint('{angdiam} = 2.0*{radius}/{distance}')
        logger.info("star '{}': 'angdiam' constrained by 'radius' and 'distance'".format(star['label'],derive))
    elif derive=='distance':
        star.pop_constraint('radius',None)
        star.add_constraint('{distance} = 2.0*{radius}/{angdiam}')
        logger.info("star '{}': '{}' constrained by 'angdiam' and 'radius'".format(star['label'],derive))
    elif derive=='radius':
        star.pop_constraint('distance',None)
        star.add_constraint('{radius} = 0.5*{angdiam}*{distance}')
        logger.info("star '{}': '{}' constrained by 'angdiam' and 'distance'".format(star['label'],derive))
    else:
        raise ValueError("Cannot derive {} from angdiam".format(derive))
    

def add_surfgrav(star,surfgrav=None,derive='mass',unit='[cm/s2]',**kwargs):
    r"""
    Add surface gravity to a Star parameterSet.
    
    Only two parameters out of surface gravity, mass and radius are
    independent. If you add surface gravity, you have to choose to derive
    either mass or radius from the other two:
    
    .. math::
    
        g = \frac{GM}{R^2}
    
    This is a list of stuff that happens:
    
    - A I{parameter} C{surfgrav} will be added if it does not exist yet
    - A I{constraint} to derive the parameter C{derive} will be added.
    - If C{surfgrav} already exists as a constraint, it will be removed
    - If there are any other constraints on the parameter C{derive}, they
      will be removed
    
    Extra C{kwargs} will be passed to the creation of C{surfgrav} if it does
    not exist yet.
   
    @param star: star parameterset
    @type star: ParameterSet of context star
    @param surfgrav: surface gravity value
    @type surfgrav: float
    @param derive: qualifier of the dependent parameter
    @type derive: str, one of C{mass}, C{radius}
    @param unit: units of surface gravity
    @type unit: str
    """
    if kwargs and 'surfgrav' in star:
        raise ValueError("You cannot give extra kwargs to add_surfgrav if it already exist")
    
    kwargs.setdefault('description','Surface gravity')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- default value
    if surfgrav is None:
        surfgrav = np.log10(constants.GG*star.get_value('mass','kg')/star.get_value('radius','m')**2) + 2.0
    
    #-- remove any constraints on surfgrav and add the parameter
    star.pop_constraint('surfgrav',None)
    if not 'surfgrav' in star:
        star.add(parameters.Parameter(qualifier='surfgrav',value=surfgrav,
                                      **kwargs))
    else:
        star['surfgrav'] = surfgrav
        
    #-- specify the dependent parameter
    if derive is None:
        star.pop_constraint('radius',None)
        star.add_constraint('{surfgrav} = constants.GG*{mass}/{radius}**2')
        logger.info("star '{}': 'surfgrav' constrained by 'mass' and 'radius'".format(star['label']))
    elif derive=='mass':
        star.pop_constraint('mass',None)
        star.add_constraint('{mass} = {surfgrav}/constants.GG*{radius}**2')
        logger.info("star '{}': '{}' constrained by 'surfgrav' and 'radius'".format(star['label'],derive))
    elif derive=='radius':
        star.pop_constraint('radius',None)
        star.add_constraint('{radius} = np.sqrt((constants.GG*{mass})/{surfgrav})')
        logger.info("star '{}': '{}' constrained by 'surfgrav' and 'mass'".format(star['label'],derive))
    else:
        raise ValueError("Cannot derive {} from surface gravity".format(derive))
    
    return star.get_parameter('surfgrav')


def add_luminosity(star, luminosity=None, derive=None,unit='[Lsol]',**kwargs):
    r"""
    Add luminosity to a Star parameterSet.
   
    @param star: star parameterset
    @type star: ParameterSet of context star
    @param surfgrav: luminosity
    @type surfgrav: float
    @param derive: qualifier of the dependent parameter
    @type derive: str, one of C{teff}, C{radius}
    @param unit: units of luminosity
    @type unit: str
    """
    if kwargs and 'luminosity' in star:
        raise ValueError("You cannot give extra kwargs to add_surfgrav if it already exist")
    
    kwargs.setdefault('description','Luminosity')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    
    #-- remove any constraints on luminosity and add the parameter
    star.pop_constraint('luminosity',None)
    if not 'luminosity' in star:
        star.add(parameters.Parameter(qualifier='luminosity',value=0.0,
                                      **kwargs))
    else:
        star['luminosity'] = luminosity
        
    #-- specify the dependent parameter
    if derive is None:
        star.add_constraint('{luminosity} = 4*np.pi*{radius}**2*constants.sigma*{teff}**4')
        logger.info("star '{}': 'luminosity' constrained by 'teff' and 'radius'".format(star['label']))
    else:
        raise ValueError("Cannot derive {} from luminosity".format(derive))

    return star.get_parameter('luminosity')
   
def add_vsini(star,vsini,derive='rotperiod',unit='km/s',**kwargs):
    r"""
    Add the vsini to a Star.
    
    If ``derive=None``, the vsini will be derived from the other parameters of
    the stars, i.e.
    
    .. math::
    
        v_\mathrm{eq}\sin i = \frac{2\pi R}{P}
    
    If ``derive='rotperiod'``, the rotation period can then be constrained by
    the vsini of the star via.
    
    .. math::
    
        \mathrm{rotperiod} = \frac{2\pi R}{v\sin i} \sin i
    
    or if ``derive='incl'``, the inclination angle
    
    .. math::
    
        \mathrm{incl} = \arcsin(\frac{P v\sin i}{2\pi R})
    
    or the stellar radius
    
    .. math::
    
        \mathrm{radius} = \frac{P v\sin i}{2\pi \sin i}
    
    
    Extra C{kwargs} will be passed to the creation of the parameter if it does
    not exist yet.
   
    Other contraints on rotation period will be removed.
   
    @param star: star parameterset
    @type star: ParameterSet of context star
    @param vsini: value for the projected equatorial rotation velocity
    @type vsini: float
    """
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Projected equatorial rotation velocity')
    kwargs.setdefault('llim',0)
    kwargs.setdefault('ulim',1000.)
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    star.pop_constraint('vsini',None)
    star.pop_constraint('rotperiodcrit',None)
    star.pop_constraint('rotperiod',None)
    star.pop('rotperiodcrit',None)
    if not 'vsini' in star:
        star.add(parameters.Parameter(qualifier='vsini',
                                      value=vsini,**kwargs))
    else:
        star['vsini'] = vsini
        
    if derive == 'rotperiod':
        star.add_constraint('{rotperiod} = 2*np.pi*{radius}/{vsini}*np.sin({incl})')
        logger.info("star '{}': 'rotperiod' constrained by 'vsini' and 'radius' and 'incl'".format(star['label']))
    elif derive == 'incl':
        star.add_constraint('{incl} = np.arcsin({rotperiod}*{vsini}/(2*np.pi*{radius}))')
        logger.info("star '{}': 'incl' constrained by 'vsini' and 'radius' and 'rotperiod'".format(star['label']))
    elif derive == 'radius':
        star.add_constraint('{radius} = {rotperiod}/(2*np.pi)*{vsini}/np.sin({incl})')
    else:
        star.add_constraint('{vsini} = (2*np.pi*{radius})/{rotperiod}*np.sin({incl})')
        logger.info("star '{}': 'vsini' constrained by 'radius', 'rotperiod' and 'incl'".format(star['label']))
    
    return star.get_parameter('vsini')
    
    
def add_rotfreqcrit(star,rotfreqcrit=None,derive='rotperiod',**kwargs):
    r"""
    Add the critical rotation frequency to a Star.
    
    The rotation period will then be constrained by the critical rotation
    frequency:
    
    .. math::
    
        \mathrm{rotperiod} = 2\pi \sqrt{27 R^3 / (8GM)} / \mathrm{rotfreqcrit}
    
    .. math::
    
        M = \frac{27}{2}\frac{\pi^2 R^3}{G P^2 f^2}
    
    Extra C{kwargs} will be passed to the creation of the parameter if it does
    not exist yet.
    
    .. warning::
    
        Be careful when mixing this constraint with other constraints that
        involve mass and radius. E.g. if you set the mass to be constrained
        by the surface gravity, you better add the constraint on the rotation
        frequency after the one on the surface gravity, so that they are
        evaluated in the right order.
   
    @param star: star parameterset
    @type star: ParameterSet of context star
    @param rotfreqcrit: value for the critical rotation frequency
    @type rotfreqcrit: float
    """
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Polar rotation freq/ critical freq')
    kwargs.setdefault('llim',1e-8)
    kwargs.setdefault('ulim',0.99)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    star.pop_constraint('rotfreqcrit',None)
    star.pop_constraint('rotperiodcrit',None)
    star.pop('rotperiodcrit',None)
    
    if rotfreqcrit is None:
        radius = star.get_value('radius','m')
        mass = star.get_value('mass','kg')
        rotperiod = star.get_value('rotperiod','s')
        rotfreqcrit = 2*np.pi/rotperiod*np.sqrt(27*radius**3/(8*constants.GG*mass))
    
    if not 'rotfreqcrit' in star:
        star.add(parameters.Parameter(qualifier='rotfreqcrit',
                                      value=rotfreqcrit,**kwargs))
    else:
        star['rotfreqcrit'] = rotfreqcrit
    
    #if derive=='mass' and star.is_constrained('radius'):
    #    star.pop_constraint('mass',None)
    #    star.add_constraint('{mass} = 27./2.*{radius}**3*np.pi**2/(constants.GG*{rotperiod}**2*{rotfreqcrit}**2)')
    #    logger.info("star '{}': 'mass' constrained by 'rotfreqcrit' and 'rotperiod' and 'radius'".format(star['label']))
    if derive=='mass':
        star.pop_constraint('mass',None)
        star.add_constraint('{mass} = 27./2.*{radius}**3*np.pi**2/(constants.GG*{rotperiod}**2*{rotfreqcrit}**2)')
        logger.info("star '{}': 'mass' constrained by 'rotfreqcrit' and 'rotperiod' and 'radius'".format(star['label']))
    elif derive=='radius':
        star.pop_constraint('radius',None)
        star.add_constraint('{radius} = 2*{surfgrav}*{rotperiod}**2*{rotfreqcrit}**2/(27.*np.pi**2)')
        logger.info("star '{}': 'radius' constrained by 'rotfreqcrit' and 'rotperiod' and 'surfgrav'".format(star['label']))
    elif derive=='rotperiod':
        star.pop_constraint('rotperiod',None)
        star.add_constraint('{rotperiod} = 2*np.pi*np.sqrt(27*{radius}**3/(8*constants.GG*{mass}))/{rotfreqcrit}')
        logger.info("star '{}': 'rotperiod' constrained by 'rotfreqcrit'".format(star['label']))
    else:
        logger.info("star '{}': 'rotperiod' not necessarily consistent with 'rotfreqcrit'".format(star['label']))
    #-- we didn't explicitly set a value here!
    star.run_constraints()
    return star.get_parameter('rotfreqcrit')

def add_vrotcrit(star, vrotcrit=None, derive='rotperiod',**kwargs):
    r"""
    Add the fractional critical rotation velocity to a star.
    
    .. math::
    
        \frac{v}{v_c} = \frac{R(\omega)}{R_\mathrm{eq}}\frac{\Omega}{\Omega_c}\\
           &          = 2\cos\left(\frac{\pi+\arccos(\omega)}{3}\right)
           
    and thus
    
    .. math::
        
        \omega = \cos\left(3\arccos(\frac{v}{2v_c})-\pi\right)
           
    Small :math:`\omega` is fractional rotation frequency (dimensionless), big
    :math:\Omega` is true rotation frequency (Hz).
    """
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Equatorial rotational velocity/critical rotational velocity')
    kwargs.setdefault('llim',0.0)
    kwargs.setdefault('ulim',1.0)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    star.pop_constraint('vrotcrit', None)
    
    if vrotcrit is None:
        radius = star.get_value('radius','m')
        mass = star.get_value('mass','kg')
        rotperiod = star.get_value('rotperiod','s')
        
    
    if not 'vrotcrit' in star:
        star.add(parameters.Parameter(qualifier='vrotcrit',
                                      value=vrotcrit, **kwargs))
    else:
        star['vrotcrit'] = vrotcrit
    
    if derive == 'rotperiod':
        star.pop_constraint('rotperiod', None)
        # Rotperiod = omega/omega_crit * omega_crit, derived from v/v_crit
        star.add_constraint('{rotperiod} = 2*np.pi / (np.cos(3*np.arccos(0.5*{vrotcrit}) - np.pi) * np.sqrt( (8*constants.GG*{mass})/(27*{radius}**3)))')
        logger.info("star '{}': 'rotperiod' constrained by 'vrotcrit'".format(star['label']))

    star.run_constraints()
    return star.get_parameter('vrotcrit')
    

def add_radius_eq(star,radius_eq=None,derive=None,unit='Rsol',**kwargs):
    """
    Equatorial radius for a fast, uniform rotating star.
    """
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Equatorial radius')
    kwargs.setdefault('llim',0)
    kwargs.setdefault('ulim',1000.)
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    if not 'radius_eq' in star:
        star.add(parameters.Parameter(qualifier='radius_eq',value=radius_eq,**kwargs))
    else:
        star['radius_eq'] = radius_eq
    
    if derive=='radius':
        rotfreqcrit = star['rotfreqcrit']
        star.add_constraint('{radius} = {rotfreqcrit}*{radius_eq}/(3.*np.cos((np.pi+np.arccos({rotfreqcrit}))/3.))')
    else:
        star.add_constraint('{radius_eq} = 3*{radius}/{rotfreqcrit}*np.cos((np.pi+np.arccos({rotfreqcrit}))/3.))')
    star.run_constraints()
    return star.get_parameter('radius_eq')

    
def add_teffpolar(star,teffpolar=None,**kwargs):
    """
    Add the polar effective temperature to a Star.
    
    The C{teff} parameter will be set to be equal to C{teffpolar}, but Phoebe
    knows how to interpret C{teffpolar}, and, if it exists, to ignore the
    value for C{teff}. If you want to know the mean passband effective
    temperature, you will need to call C{as_point_source} on the Body.
    
    Extra C{kwargs} will be passed to the creation of the parameter if it does
    not exist yet.
   
    @param star: star parameterset
    @type star: ParameterSet of context star
    @param teffpolar: value for the polar effective temperature
    @type teffpolar: float
    """
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Polar effective temperature')
    kwargs.setdefault('llim',0)
    kwargs.setdefault('ulim',1e20)
    kwargs.setdefault('unit','K')
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    star.pop_constraint('teffpolar',None)
    
    #-- set default value
    if teffpolar is None:
        teffpolar = star['teff']
    
    if not 'teffpolar' in star:
        star.add(parameters.Parameter(qualifier='teffpolar',
                                      value=teffpolar,**kwargs))
    else:
        star['teffpolar'] = teffpolar
    star.add_constraint('{teff} = {teffpolar}')
    logger.info("star '{}': 'teff' redefined to be equal to 'teffpolar'".format(star['label']))
    return star.get_parameter('teffpolar')
    

def add_solarosc(star,numax,Deltanu0=None,unit='muHz'):
    """
    Add :math:`\\nu_\mathrm{max}` and :math:`\Delta\\nu_0` to a star.
    
    See Kjeldsen 1995 for the relations:
    
    Given everything in solar units (also frequencies):
    
    .. math::
        
        \mathrm{surfgrav} =G \sqrt{T_\mathrm{eff}}  f_\mathrm{max}\quad [\mathrm{solar units}]
        
        \mathrm{radius} = \sqrt{T_\mathrm{eff}} f_\mathrm{max}/(\Delta f_0)^2 \quad [\mathrm{solar units}]
        
    """
    add_solarosc_numax(star,numax,unit=unit)
    add_solarosc_Deltanu0(star,Deltanu0,unit=unit)

def add_solarosc_numax(star,numax,unit='muHz',**kwargs):
    """
    Add :math:`\\nu_\mathrm{max}` to a star.
    """
    if kwargs and 'numax' in star:
        raise ValueError("You cannot give extra kwargs to add_solarosc_numax if it already exist")
    
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Frequency of maximum power')
    kwargs.setdefault('llim',0)
    kwargs.setdefault('ulim',1e20)
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    add_surfgrav(star,0,derive='radius')
    
    #-- remove any constraints on surfgrav and add the parameter
    star.pop_constraint('numax',None)
    if not 'numax' in star:
        star.add(parameters.Parameter(qualifier='numax',value=numax,**kwargs))
    else:
        star['numax'] = numax
    
    # we need to divide by 2pi because SI units of frequency are rad/s instead of Hz
    scale = 'constants.Msol/constants.Rsol**2/np.sqrt(constants.Tsol)*constants.GG/(constants.numax_sol[0]*1e-6)'
    star.add_constraint('{surfgrav} = '+scale+'*np.sqrt({teff})*{numax}/(2*np.pi)')
    #-- append the constraint on the radius to be after the surface gravity.
    star.add_constraint('{radius} = '+star.pop_constraint('radius'))
    logger.info("star '{}': 'surfgrav' constrained by 'numax' and 'teff'".format(star['label']))
    return star.get_parameter('numax')
    

def add_solarosc_Deltanu0(star,Deltanu0,unit='muHz',**kwargs):
    """
    Add :math:`\Delta\\nu_0` to a star.
    
    You can only add this one if numax was added first!
    """
    if kwargs and 'Deltanu0' in star:
        raise ValueError("You cannot give extra kwargs to add_solarosc if it already exist")
    
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Large separation between radial modes')
    kwargs.setdefault('llim',0)
    kwargs.setdefault('ulim',1e20)
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on surfgrav and add the parameter
    star.pop_constraint('Deltanu0',None)
    if not 'Deltanu0' in star:
        star.add(parameters.Parameter(qualifier='Deltanu0',value=Deltanu0,**kwargs))
    else:
        star['Deltanu0'] = Deltanu0
        
    star.pop_constraint('mass',None)
    star.pop_constraint('radius',None)
    scale = '*(constants.Deltanu0_sol[0]*1e-6)**2/(constants.numax_sol[0]*1e-6)/np.sqrt(constants.Tsol)'
    star.add_constraint('{radius} = np.sqrt({teff})*{numax}/{Deltanu0}**2*(2*np.pi)*constants.Rsol'+scale)
    star.add_constraint('{mass} = {surfgrav}/constants.GG*{radius}**2')
    logger.info("star '{}': 'radius' constrained by 'numax', 'teff' and 'Deltanu0'".format(star['label']))
    logger.info("star '{}': 'mass' constrained by 'surfgrav' and 'radius'".format(star['label']))
    
    return star.get_parameter('Deltanu0')
#}


#{ Common constraints for the BinaryRocheStar or Orbit

def add_asini(orbit, asini=None, derive='sma', unit='Rsol', **kwargs):
    """
    Add asini to an orbit parameterSet.
    
    The :math:`a\sin i` is parameter that you can calculate from radial velocity
    data, e.g. via :py:func:`calculate_asini <phoebe.dynamics.keplerorbit.calculate_asini>`.
    
    Only two parameters out of C{asini}, C{sma} and C{incl} are
    independent. If you add C{asini}, you have to choose to derive
    either C{sma} or C{incl} from the other two:
    
    .. math::
    
        \mathrm{asini} = \mathrm{sma} \sin(\mathrm{incl})
    
    This is a list of stuff that happens:
    
    - A I{parameter} C{asini} will be added if it does not exist yet
    - A I{constraint} to derive the parameter C{derive} will be added.
    - If C{asini} already exists as a constraint, it will be removed
    - If there are any other constraints on the parameter C{derive}, they
      will be removed
    
    Extra C{kwargs} will be passed to the creation of the Parameter C{asini} if
    it does not exist yet.
    
    See also :py:func:`phoebe.dynamics.keplerorbit.calculate_asini`.
    
    @param orbit: orbit parameterset
    @type orbit: ParameterSet of context star
    @param asini: system projected semi-major axis
    @type asini: float
    @param derive: qualifier of the dependent parameter
    @type derive: str, one of C{sma}, C{incl}
    @param unit: units of semi-major axis
    @type unit: str
    """
    if kwargs and 'asini' in orbit:
        raise ValueError("You cannot give extra kwargs to add_asini if it already exist")
    
    kwargs.setdefault('description','Projected system semi-major axis')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',orbit.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    if asini is None:
        asini = orbit.request_value('sma','m')*np.sin(orbit.request_value('incl','rad'))
        asini = conversions.convert('m',unit,asini)
    
    #-- remove any constraints on surfgrav and add the parameter
    orbit.pop_constraint('asini',None)
    if not 'asini' in orbit:
        orbit.add(parameters.Parameter(qualifier='asini',value=asini,
                                      **kwargs))
    else:
        orbit['asini'] = asini
        
    #-- specify the dependent parameter
    if derive=='sma':
        orbit.pop_constraint('sma',None)
        orbit.add_constraint('{sma} = {asini}/np.sin({incl})')
    elif derive=='incl':
        orbit.pop_constraint('incl',None)
        orbit.add_constraint('{incl} = np.arcsin({asini}/{sma})')
    elif derive is None:
        orbit.pop_constraint('asini',None)
        orbit.add_constraint('{asini} = {sma}*np.sin({incl})')
    else:
        raise ValueError("Cannot derive {} from asini".format(derive))
    logger.info("orbit '{}': '{}' constrained by 'asini'".format(orbit['label'],derive))
    return orbit.get_parameter('asini'),

def add_K1_K2(orbit, K1=None, K2=None, unit='km/s',**kwargs):
    """
    Add primary and secondary semi-amplitudes to the fit.
    
    
    """
    kwargs.setdefault('description', 'Component semi-amplitude')
    kwargs.setdefault('unit', unit)
    kwargs.setdefault('context', orbit.context)
    kwargs.setdefault('adjust', False)
    kwargs.setdefault('frame', 'phoebe')
    kwargs.setdefault('cast_type', float)
    kwargs.setdefault('repr', '%f')
    orbit.pop_constraint('q',None)
    orbit.pop_constraint('sma', None)
    
    if K1 is None:
        period = orbit.request_value('period','SI')
        ecc = orbit['ecc']
        q = orbit['q']
        sma = orbit.request_value('sma','SI')
        K1 = q*sma/(1+q)*2*np.pi/P*np.sqrt(1-e**2)
    if K2 is None:
        period = orbit.request_value('period','SI')
        ecc = orbit['ecc']
        q = orbit['q']
        sma = orbit.request_value('sma','SI')
        K2 = sma/(1+q)*2*np.pi/P*np.sqrt(1-e**2)
        
        
    if not 'K1' in orbit:
        orbit.add(parameters.Parameter(qualifier='K1',value=K1,
                                      **kwargs))
    else:
        orbit['K1'] = K1
    if not 'K2' in orbit:
        orbit.add(parameters.Parameter(qualifier='K2',value=K2,
                                      **kwargs))
    else:
        orbit['K2'] = K2
    
    orbit.add_constraint('{q} = {K1}/{K2}')
    orbit.add_constraint('{sma} = {K2}*{period}/(2*np.pi)*np.sqrt(1-{ecc}**2)*(1+{q})')
    
    logger.info("orbit '{}': q and sma constrained by K1 and K2".format(orbit['label']))
    
    return orbit.get_parameter('K1'), orbit.get_parameter('K2')


def add_ecosw(orbit,ecosw=None,derive='per0',**kwargs):
    """
    Add ecosw to an orbit parameterSet.
    
    Only two parameters out of C{ecosw}, C{ecc} and C{per0} are
    independent. If you add C{ecosw}, you have to choose to derive
    either C{per0} or C{ecc} from the other two:
    
    .. math::
    
        \mathrm{ecosw} = \mathrm{ecc} \cos(\mathrm{per0})
    
    This is a list of stuff that happens:
    
    - A I{parameter} C{ecosw} will be added if it does not exist yet
    - A I{constraint} to derive the parameter C{derive} will be added.
    - If C{asini} already exists as a constraint, it will be removed
    - If there are any other constraints on the parameter C{derive}, they
      will be removed
    
    Extra C{kwargs} will be passed to the creation of C{asini} if it does
    not exist yet.
    
    .. warning::
    
        You probably want to use :py:func:`add_esinw_ecosw`.
   
    @param orbit: orbit parameterset
    @type orbit: ParameterSet of context star
    @param ecosw: the parameter
    @type ecosw: float
    @param derive: qualifier of the dependent parameter
    @type derive: str, one of C{ecc}, C{per0}
    @param unit: units of semi-major axis
    @type unit: str
    """
    if orbit.frame=='phoebe':
        peri = 'per0'    
    else:
        peri = 'omega'
    if kwargs and 'ecosw' in orbit:
        raise ValueError("You cannot give extra kwargs to add_ecosw if it already exist")
    
    kwargs.setdefault('description','Projected system semi-major axis')
    kwargs.setdefault('context',orbit.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on surfgrav and add the parameter
    orbit.pop_constraint('ecosw',None)
    if ecosw is None:
        ecosw = orbit['ecc']*np.cos(orbit.request_value(peri,'rad'))
    if not 'ecosw' in orbit:
        orbit.add(parameters.Parameter(qualifier='ecosw',value=ecosw,
                                      **kwargs))
    else:
        orbit['ecosw'] = ecosw
        
    #-- specify the dependent parameter
    if derive=='ecc':
        orbit.pop_constraint('ecc',None)
        orbit.add_constraint('{{ecc}} = {{ecosw}}/np.cos({{{peri}}})'.format(peri=peri))
    elif derive=='per0':
        orbit.pop_constraint(peri,None)
        orbit.add_constraint('{{{peri}}} = np.arccos({{ecosw}}/{{ecc}})'.format(peri=peri))
    else:
        raise ValueError("Cannot derive {} from ecosw".format(derive))
    logger.info("orbit '{}': '{}' constrained by 'ecosw'".format(orbit['label'],derive))



def add_esinw(orbit,esinw=None,derive='ecc',**kwargs):
    """
    Add esinw to an orbit parameterSet.
    
    Only two parameters out of C{esinw}, C{ecc} and C{per0} are
    independent. If you add C{esinw}, you have to choose to derive
    either C{per0} or C{ecc} from the other two:
    
    .. math::
    
        \mathrm{esinw} = \mathrm{ecc} \sin(\mathrm{per0})
    
    This is a list of stuff that happens:
    
    - A I{parameter} C{esinw} will be added if it does not exist yet
    - A I{constraint} to derive the parameter C{derive} will be added.
    - If there are any other constraints on the parameter C{derive}, they
      will be removed
    
    Extra C{kwargs} will be passed to the creation of C{esinw} if it does
    not exist yet.
    
    .. warning::
    
        You probably want to use :py:func:`add_esinw_ecosw`.
   
    @param orbit: orbit parameterset
    @type orbit: ParameterSet of context star
    @param esinw: the parameter
    @type esinw: float
    @param derive: qualifier of the dependent parameter
    @type derive: str, one of C{ecc}, C{per0}
    @param unit: units of semi-major axis
    @type unit: str
    """
    if orbit.frame=='phoebe':
        peri = 'per0'
    else:
        peri = 'omega'
    if kwargs and 'esinw' in orbit:
        raise ValueError("You cannot give extra kwargs to add_esinw if it already exist")
    
    kwargs.setdefault('description','Eccentricy times sine of argument of periastron')
    kwargs.setdefault('context',orbit.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on surfgrav and add the parameter
    orbit.pop_constraint('esinw',None)
    if esinw is None:
        esinw = orbit['ecc']*np.sin(orbit.request_value(peri,'rad'))
    if not 'esinw' in orbit:
        orbit.add(parameters.Parameter(qualifier='esinw',value=esinw,
                                      **kwargs))
    else:
        orbit['esinw'] = esinw
        
    #-- specify the dependent parameter
    if derive=='ecc':
        orbit.pop_constraint('ecc',None)
        orbit.add_constraint('{{ecc}} = {{esinw}}/np.sin({{{peri}}})'.format(peri=peri))
    elif derive==peri:
        orbit.pop_constraint(peri,None)
        orbit.add_constraint('{{{peri}}} = np.arcsin({{esinw}}/{{ecc}})'.format(peri=peri))
    else:
        raise ValueError("Cannot derive {} from esinw".format(derive))
    logger.info("orbit '{}': '{}' constrained by 'esinw'".format(orbit['label'],derive))


def add_esinw_ecosw(orbit, esinw=None, ecosw=None):
    r"""
    Add esinw and ecosw such that they invert to a unique argument of periastron.
    
    If you add esinw and ecosw separately (via :py:func:`add_esinw` and
    :py:func:`add_ecosw`), the constraints are ambiguous against
    the argument of periastron.
    
    .. math::
    
        \omega & = \arctan\left(\frac{e\sin\omega}{e\cos\omega}\right)\\
        e      & = \sqrt{(e\cos\omega)^2 + (e\sin\omega)^2)}
    
    @param orbit: orbit parameterset
    @type orbit: ParameterSet of context star
    """
    if orbit.frame == 'phoebe':
        peri = 'per0'
    else:
        peri = 'omega'
    add_esinw(orbit)
    add_ecosw(orbit)
    orbit.pop_constraint('ecc', None)
    orbit.pop_constraint(peri,None)
    orbit.add_constraint('{{{peri}}} = np.arctan2({{esinw}},{{ecosw}})'.format(peri=peri))
    orbit.add_constraint('{ecc} = np.sqrt(({ecosw})**2+({esinw})**2)')
    if esinw is not None:
        orbit['esinw'] = esinw
    if ecosw is not None:
        orbit['ecosw'] = ecosw
    


def add_conserve(orbit, conserve='volume', **kwargs):
    """
    Add a parameter to conserve volume or equipotential value along an eccentric orbit.
    
    @param orbit: orbital parameterSet
    @type orbit: parameterSet of context orbit
    @param conserve: what to conserve, volume or equipotential
    @type conserve: str, one of ``periastron`` or ``equipot``
    """
    if kwargs and 'conserve' in orbit:   
        raise ValueError("You cannot give extra kwargs to add_conserve if it already exist")
    
    kwargs.setdefault('description','Sets the quantity to be conserved along an eccentric orbit')
    kwargs.setdefault('choices',['periastron','sup_conj','inf_conj','asc_node','desc_node','equipot'])
    kwargs.setdefault('context',orbit.context)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',str)
    kwargs.setdefault('repr','%s')
    
    if not 'conserve' in orbit:
        orbit.add(parameters.Parameter(qualifier='conserve',value=conserve,
                                      **kwargs))
    else:
        orbit['conserve'] = conserve
        
    logger.info("orbit '{}': added and set 'conserve' to '{}'".format(orbit['label'],conserve))

def make_misaligned(orbit,theta=0.,phi0=0.,precperiod=np.inf):
    r"""
    Extend the parameters of an orbit to make it misaligned.
    
    The definition of the parameters is as follows:
    
    - ``theta``: misalignment inclination. If  the orbital plane is edge-on,
      then :math:`\theta=90` means the star is viewed pole-on. :math:`\theta=180` means
      no misalignment, :math:`\theta=0` means no misalignment but retrograde spinning.
    - ``phi0``: phase angle, orients the misalignment at some reference time.
      the reference time is given by ``t0`` in the ``orbit`` parameterSet. ``phi0``
      is such that if :math:`\phi_0=90`, the star is viewed edge on at ``t0``.
    - ``precperiod``: period of precession. If ``precperiod=np.inf``, then
      there is no precession and the body will have the same orientation in
      space at all times.
    
    """
    orbit.add(parameters.Parameter(qualifier='theta',value=theta,unit='deg',cast_type=float,description='Misalignment inclination',adjust=False))
    orbit.add(parameters.Parameter(qualifier='phi0',value=phi0,unit='deg',cast_type=float,description='Misalignment phase',adjust=False))
    orbit.add(parameters.Parameter(qualifier='precperiod',value=precperiod,unit='d',cast_type=float,description='Period of precession',adjust=False))

def add_theta_eff(orbit,theta_eff=None):
    r"""
    Add an effective misalignment parameter [Avni1982]_.
    
    .. math::
    
        \theta_\mathrm{eff} = \arccos\left(\cos\theta\sqrt{1+\sin^2\phi\tan^2\phi}\right)
    
    We assume then that :math:`\theta` is fixed and :math:`\phi` will be derived.
    Note that :math:`\theta\geq\theta_\mathrm{eff}` must hold.
    
    In other words, for each set value of :math:`\theta`, :math:`\phi` will be
    set such that the angle between the pole and line-of-sight is constant. This
    keeps the vsini constant.
    """
    theta = orbit.request_value('theta','rad')
    phi = orbit.request_value('phi0','rad')
    if theta_eff is None:
        theta_eff = np.arccos(np.cos(theta)*np.sqrt(1+np.sin(phi)**2*np.tan(theta)**2))
        theta_eff = theta_eff/np.pi*180.
    orbit.add(parameters.Parameter(qualifier='theta_eff',value=theta_eff,unit='deg',cast_type=float,description='Effective misalignment parameter',adjust=False))
    orbit.add_constraint('{phi0} = np.arcsin(1.0/np.tan({theta}) * np.sqrt(np.cos({theta_eff})**2/np.cos({theta})**2 -1.0)) if {theta}!=np.pi/2 else np.pi/2.0-{phi0}')
    orbit.run_constraints()

def to_perpass(orbit):
    """
    Convert an orbital set where t0 is superior conjunction to periastron passage.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have T0 as superior conjunction.
    
    Inverse function is L{to_supconj}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    @param orbit: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit: parameterset of frame C{phoebe} and context C{orbit}
    """
    t0type = orbit.get('t0type','periastron passage')
    if t0type == 'superior conjunction':
        t_supconj = orbit['t0']
        phshift = orbit['phshift']
        P = orbit['period']
        per0 = orbit.get_value('per0','rad')
        t0 = t_supconj + (phshift - 0.25 + per0/(2*np.pi))*P
        orbit['t0'] = t0
        orbit['t0type'] = 'periastron passage'
        logger.info('Set t0type to time of periastron passage')
    elif t0type == 'periastron passage':
        logger.info('t0type was already time of periastron passage')
    else:
        raise ValueError('Do not recognize t0type "{}"'.format(t0type))
        

def to_supconj(orbit):
    """
    Convert an orbital set where t0 is periastron passage to superior conjunction.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have T0 as superior conjunction.
    
    Inverse function is L{to_perpass}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    @param orbit: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit: parameterset of frame C{phoebe} and context C{orbit}
    """
    if 't0type' in orbit:
        t0type = orbit['t0type']
    else:
        t0type = 'periastron passage'
    
    if t0type == 'periastron passage':
        t_perpass = orbit['t0']
        phshift = orbit['phshift']
        P = orbit['period']
        per0 = orbit.get_value('per0','rad')
        t0 = t_perpass - (phshift - 0.25 + per0/(2*np.pi))*P
        orbit['t0'] = t0
        if 't0type' in orbit:
            orbit['t0type'] = 'superior conjunction'
            
    logger.info('Set t0type to time of superior conjunction')


def critical_times(orbit_in, time_range=None):
    """
    Compute critical times in an orbit.
    
    They are the times of:
    
        1. Periastron passage: when the stars are closest
        2. Superior conjunction: when the primary is eclipsed (if there are eclipses)
        3. Inferior conjunction: when the secondary is eclipsed (if there are eclipses)
        4. Ascending node
        5. Descending node
    
    @param orbit_in: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit_in: parameterset of frame C{phoebe} and context C{orbit}
    @return: times of critical points in the orbit
    @rtype: array
    """
    if not orbit_in.get('t0type','periastron passage') == 'periastron passage':
        orbit = orbit_in.copy()
        to_perpass(orbit)
    else:
        orbit = orbit_in
    
    t0 = orbit['t0']
    P = orbit['period']
    per0 = orbit.request_value('per0','rad')
    ecc = orbit['ecc']
    
    crit_times = keplerorbit.calculate_critical_phases(per0, ecc) * P + t0
    crit_times[crit_times<t0] += P
    crit_times[crit_times>(t0+P)] -= P
    
    # Perhaps the user wants all the critical time points in a certain time
    # interval
    if time_range is not None:
        n_period_min = int((time_range[0]-t0) / P) - 2 # to be sure
        n_period_max = int((time_range[1]-t0) / P) + 2 # to be sure
        n_periods = np.arange(-n_period_min, n_period_max+1)
        crit_times = np.array([ct + n_periods*P for ct in crit_times])
        keep = (crit_times<=time_range[1]) & (time_range[0]<=crit_times)
        crit_times = crit_times[keep]
    
    return crit_times

def critical_phases(orbit_in, phase_range=None):
    """
    Compute critical phases in an orbit.
    
    The are the phases of:
    
        1. Periastron passage: when the stars are closest
        2. Superior conjunction: when the primary is eclipsed (if there are eclipses)
        3. Inferior conjunction: when the secondary is eclipsed (if there are eclipses)
        4. Ascending node
        5. Descending node
    
    @param orbit_in: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit_in: parameterset of frame C{phoebe} and context C{orbit}
    @return: times of critical points in the orbit
    @rtype: array
    """
    crit_times = critical_times(orbit_in)
    period = orbit_in['period']
    return (crit_times % period) / period
    
        
        
def make_synchronous(orbit, comp1=None, comp2=None):
    r"""
    Set the synchronicity parameter to match synchronous or pseudosynchronous rotation.
    
    Synchronous rotation in the circular (:math:`e=0`) case means:
    
    .. math::
    
        P_\mathrm{rot} = P_\mathrm{orbit}
    
    and Pseudosynchronous rotation means that the star has an agnular rotation
    rate equal to the orbit angular velocity at periastron. In general, the
    orbital angular velocity is given by [Hut1981]_:
    
    .. math::
        
        \omega(t) = \frac{1}{\delta^2}\left(\frac{2\pi}{P_\mathrm{orbit}}\right)\sqrt{1-e^2}
        
    At periastron, :math:`\delta=1-e`, so this gives for the synchronicity
    parameter:
    
    .. math::
    
        F = \sqrt{\frac{1+e}{(1-e)^3}}
        
    """
    ecc = orbit['ecc']
    F = np.sqrt( (1+ecc) / (1-ecc)**3)
    if comp1 is not None:
        comp1['syncpar'] = F
    if comp2 is not None:
        comp2['syncpar'] = F
    
#}

#{ Pulsation constraints

def add_amplvelo(puls,amplvelo=None,derive=None,unit='s-1',**kwargs):
    """
    Add velocity amplitude to a Puls parameterSet.
    
    The relation between the radial amplitude :math:`A(x)` and the velocity
    amplitude :math:`A(v)` is defined via the frequency :math:`f` as
    
    .. math::
    
        A(v) = 2\pi f A(x)
    
    The units of the velocity amplitude are in fractional radius per second.
    In other words, if you have it in km/s, you need to divide it by the
    radius of the star. Else, you need to implement a global constraint (i.e.
    a preprocessor).
    
    If ``derive=None``, then the velocity amplitude will be constrained by
    the radial amplitude and frequency.
    
    @param puls: pulsational parameterSet
    @type puls: ParameterSet of context ``puls``
    @param amplvelo: velocity amplitude
    @type amplvelo: float
    @param derive: type of parameter to derive
    @type derive: None or 'ampl'.
    @param unit: unit of the velocity amplitude
    @type unit: str
    """
    if kwargs and 'amplvelo' in puls:
        raise ValueError("You cannot give extra kwargs to add_amplvelo if amplvelo already exist")
    
    kwargs.setdefault('description','Velocity amplitude of the pulsation')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',puls.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on amplvelo and add the parameter
    puls.pop_constraint('amplvelo',None)
    if not 'amplvelo' in puls:
        puls.add(parameters.Parameter(qualifier='amplvelo',
                                value=amplvelo if amplvelo is not None else 0.,
                                **kwargs))
    else:
        star['amplvelo'] = amplvelo
        
    #-- specify the dependent parameter
    if amplvelo is None:
        puls.add_constraint('{amplvelo} = 2.0*np.pi*{freq}*{ampl}')
        logger.info("puls '{}': 'amplvelo' constrained by 'freq' and 'ampl'".format(puls['label']))
    elif derive=='ampl':
        puls.pop_constraint('ampl',None)
        puls.add_constraint('{ampl} = {amplvelo}/(2*np.pi*{freq})')
        logger.info("puls '{}': '{}' constrained by 'amplvelo' and 'freq'".format(puls['label'],derive))
    else:
        raise ValueError("Cannot derive {} from amplvelo".format(derive))
    




def add_nonadiabatic_coefficients(puls, fT=0, psiT=180., fg=0, psig=180.):
    r"""
    Add nonadiabatic coefficients to the pulsation parameters.
    
    Then the only independent variable that you can set is the amplitude and
    phase of the radial displacement. The amplitude and phase of
    :math:`T_\mathrm{eff}` and :math:`\log g` will be derived from that via
    the given coefficients:
    
    
        
        
    """
    pardict = dict(frame='phoebe', cast_type=float, repr='%f', adjust=False,
                 context=puls.context)
    
    puls.pop_constraint('fT',None)
    puls.pop_constraint('psiT',None)
    puls.pop_constraint('fg',None)
    puls.pop_constraint('psig',None)
    puls.pop('fT',None)
    puls.pop('psiT',None)
    puls.pop('fg',None)
    puls.pop('psig',None)
    
    pardict['description'] = 'Amplitude of temperature perturbation relative to radius perturbation'
    pardict.pop('unit',None)
    puls.add(parameters.Parameter(qualifier='fT', value=fT, **pardict))
    pardict['description'] = 'Amplitude of gravity perturbation relative to radius perturbation'
    pardict.pop('unit',None)
    puls.add(parameters.Parameter(qualifier='fg', value=fg, **pardict))
    pardict['description'] = 'Phase of temperature perturbation relative to radius perturbation'
    pardict.pop('unit','deg')
    puls.add(parameters.Parameter(qualifier='psiT', value=psiT, **pardict))
    pardict['description'] = 'Phase of gravity perturbation relative to radius perturbation'
    pardict.pop('unit','deg')
    puls.add(parameters.Parameter(qualifier='psig', value=psig, **pardict))
             
    puls.add_constraint('{amplteff} = {fT}*{ampl}')
    puls.add_constraint('{phaseteff} = {psiT}/180.*pi - 2*pi*{phase}')
    puls.add_constraint('{amplgrav} = {fg}*{ampl}')
    puls.add_constraint('{phasegrav} = {psig}/180.*pi - 2*pi*{phase}')
    
    
def add_alphaT(spdep, alphaT=0.0, **kwargs):
    kwargs.setdefault('description','Temperature dependence of depth of profile')
    kwargs.setdefault('context',spdep.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    # remove any constraints on the parameter
    spdep.pop_constraint('alphaT', None)
    if not 'alphaT' in spdep:
        spdep.add(parameters.Parameter(qualifier='alphaT', value=alphaT, **kwargs))
    else:
        spdep['alphaT'] = alphaT
    

#}

#{ Constraints on reddening

def add_ebv(reddening, ebv=None, unit='mag', derive='extinction', **kwargs):
    r"""
    Add extinction parameter E(B-V) to a reddening ParameterSet.
    
    .. math::
    
        A(V) = R_V \cdot E(B-V)
    
    """
    kwargs.setdefault('description','Interstellar reddening E(B-V)')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',reddening.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    # remove any constraints on the parameter
    reddening.pop_constraint('ebv', None)
    if not 'ebv' in reddening:
        reddening.add(parameters.Parameter(qualifier='ebv',
                           value=ebv if ebv is not None else 0., **kwargs))
    else:
        reedening['ebv'] = ebv
        
    
    # Specify dependent parameter
    if ebv is None:
        reddening.add_constraint("{ebv} = {extinction}/{Rv}")
#}

#{ Other constraints

def add_albgeom(ps, albgeom=0.0, **kwargs):
    r"""
    Add geometric albedo to a body or pbdep.
    
    The relation between Bond's albedo :math:`a_B` (which is Phoebe's passband
    albedo) and the geometric albedo :math:`a_g` is the following (assuming Lambert's law):
    
    .. math::
    
        a_g = \frac{2}{3} a_B
        
    or inversely
    
    .. math::
    
        a_B = \frac{3}{2} a_g
    
    See, for example, [Esteves2013]_ or [Lopez-Morales2007]_ for a more detailed
    explanation.
    """
    if kwargs and 'albgeom' in ps:
        raise ValueError("You cannot give extra kwargs to add_albgeom if albgeom already exist")
    
    kwargs.setdefault('description',"Geometric albedo")
    kwargs.setdefault('context',ps.context)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on albgeom and add the parameter
    ps.pop_constraint('albgeom',None)
    if not 'albgeom' in ps:
        ps.add(parameters.Parameter(qualifier='albgeom', value=albgeom,**kwargs))
    else:
        ps['albgeom'] = albgeom
        
    ps.add_constraint('{alb} = 1.5*{albgeom}')
    

def add_albmap(star, image, scale=None, invert=False, **kwargs):
    """
    Add an albedo surface map.
    """
    if kwargs and 'albmap' in star:
        raise ValueError("You cannot give extra kwargs to add_albmap if albmap already exist")
    
    kwargs.setdefault('description','Albedo surface map')
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',str)
    kwargs.setdefault('repr','%s')
    
    #-- remove any constraints on albmap and add the parameter
    star.pop_constraint('albmap',None)
    if not 'albmap' in star:
        star.add(parameters.Parameter(qualifier='albmap', value=os.path.abspath(image),**kwargs))
    else:
        star['albmap'] = albmap
        
    star.add_constraint('{{albmap_min}} = {:.16e}'.format(scale[0]))
    star.add_constraint('{{albmap_max}} = {:.16e}'.format(scale[1]))
    star.add_constraint('{{albmap_inv}} = {}'.format(invert))

def add_redistmap(star, image, scale=None, invert=False, **kwargs):
    """
    Add a global heat redistribution surface map.
    """
    if kwargs and 'redistmap' in star:
        raise ValueError("You cannot give extra kwargs to add_redistmap if redistmap already exist")
    
    kwargs.setdefault('description','Albedo surface map')
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',str)
    kwargs.setdefault('repr','%s')
    
    #-- remove any constraints on albmap and add the parameter
    star.pop_constraint('redistmap',None)
    if not 'redistmap' in star:
        star.add(parameters.Parameter(qualifier='redistmap', value=os.path.abspath(image),**kwargs))
    else:
        star['redistmap'] = redistmap
        
    star.add_constraint('{{redistmap_min}} = {:.16e}'.format(scale[0]))
    star.add_constraint('{{redistmap_max}} = {:.16e}'.format(scale[1]))
    star.add_constraint('{{redistmap_inv}} = {}'.format(invert))


def add_parallax(star,parallax=None,unit='mas',**kwargs):
    """
    Add parallax to a parameterSet.
    """
    if kwargs and 'parallax' in star:
        raise ValueError("You cannot give extra kwargs to add_parallax if parallax already exist")
    
    kwargs.setdefault('description','Parallax')
    kwargs.setdefault('unit',unit)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('frame','phoebe')
    kwargs.setdefault('cast_type',float)
    kwargs.setdefault('repr','%f')
    
    #-- remove any constraints on angdiam and add the parameter
    star.pop_constraint('parallax',None)
    if not 'parallax' in star:
        star.add(parameters.Parameter(qualifier='parallax',
                                value=parallax if parallax is not None else 0.,
                                **kwargs))
    else:
        star['parallax'] = parallax
        
    #-- specify the dependent parameter
    if parallax is None:
        star.pop_constraint('parallax',None)
        star.add_constraint('{parallax} = constants.au/{distance}')
        logger.info("star '{}': 'parallax' constrained by 'distance'".format(star['label']))
    else:
        star.add_constraint('{distance} = constants.au/{parallax}')
        logger.info("star '{}': 'distance' constrained by 'parallax'".format(star['label']))


def add_scattering(pbdep, scattering_type):
    """
    Add scattering properties to a passband dependable.
    
    The :envvar:`scattering_type` can be any of:
    
        - ``isotropic``: all info is already present, nothing is done
        - ``henyey``: Henyey-Greenstein scattering phase function
        - ``hapke``: Hapke model
    
    The necessary extra parameters are added to the :envvar:`pbdep`.
    
    @param pbdep: passband dependent parameterSet (lcdep,...)
    @type pbdep: ParameterSet
    @param scattering_type: type of scattering (see above)
    @type scattering_type: str
    """
    
    if scattering_type == 'isotropic':
        pbdep['scattering'] = 'isotropic'
        return None
    
    else:
        extra_pars = parameters.ParameterSet('scattering:{}'.format(scattering_type))
        
        for extra_par in extra_pars:
            extra_par = extra_pars.get_parameter(extra_par)
            pbdep.add(extra_par)
        
        pbdep['scattering'] = scattering_type
        logger.info("Added {} scattering parameters to pbdep {}".format(scattering_type, pbdep['ref']))


def add_unbounded_from_bounded(parset,qualifier,from_='limits'):
    r"""
    Add an unbounded version of a bounded parameter to a parameterset.
    
    This can be helpful for inclusion in fitting algorithms that cannot handle
    bounds. The original parameter will be kept, but a transformed one will be
    added. The two versions of the parameters are linked through a constraint.
    
    The transformation of a parameter :math:`P` with upper limit :math:`U`
    and :math:`L` to an unbounded parameter :math:`P'` is given by:
    
    .. math::
    
        P' = \left(\frac{\arctan(P)}{\pi} + \frac{1}{2}\right) (U-L) + L
    
        P = \tan\left(\pi\left(\frac{P'-L}{U-L}-\frac{1}{2}\right)\right)
    
    Basically, the parameter that you wish to unbound is first converted to
    SI, and then unbounded as described above. The bounds can be taken from
    the prior or from the limits of the parameter.
    
    Let's take a complicated example of the surface gravity parameter, since
    this parameter is not originally available in the parameters, it gets
    nonlinearly transformed when converted to SI, *and* nonlinearly
    transformed through the unbounding transformation.
    
    First create a parameterset (set inclination to 60 degrees for fun), add
    the surface gravity as a parameter and set the prior.
    
    >>> star = create.star_from_spectral_type('B9V',incl=60.)
    >>> tools.add_surfgrav(star,4.0,derive='mass')
    >>> star.get_parameter('surfgrav').set_prior(distribution='uniform',lower=3.5,upper=5.0)
    >>> print(star)
          teff 10715.193052                                   K - phoebe Effective temperature
        radius 3.579247                                    Rsol - phoebe Radius
          mass 4.66955987072                               Msol - phoebe Stellar mass
           atm kurucz                                        --   phoebe Bolometric Atmosphere model
     rotperiod 0.90517                                        d - phoebe Polar rotation period
       diffrot 0.0                                            d - phoebe (Eq - Polar) rotation period (<0 is solar-like)
         gravb 1.0                                           -- - phoebe Bolometric gravity brightening
      gravblaw zeipel                                        --   phoebe Gravity brightening law
          incl 60.0                                         deg - phoebe Inclination angle
          long 0.0                                          deg - phoebe Orientation on the sky (East of North)
      distance 10.0                                          pc - phoebe Distance to the star
         shape equipot                                       --   phoebe Shape of surface
        vgamma 0.0                                         km/s - phoebe Systemic velocity
           alb 1.0                                           -- - phoebe Bolometric albedo (alb heating, 1-alb reflected)
        redist 0.0                                           -- - phoebe Global redist par (1-redist) local heating, redist global heating
    irradiator False                                         --   phoebe Treat body as irradiator of other objects
          abun 0.0                                           --   phoebe Metallicity
         label B9V_2f7ccdfd-d8c9-43ac-ad6d-5b2d2f0a4d72      --   phoebe Name of the body
       ld_func claret                                        --   phoebe Bolometric limb darkening model
     ld_coeffs kurucz                                        --   phoebe Bolometric limb darkening coefficients
      surfgrav 4.0                                      [cm/s2] - phoebe Surface gravity
          mass 9.28563927225e+30                            n/a   constr {surfgrav}/constants.GG*{radius}**2
    
    Then, unbound the surface gravity from the information from the prior. You
    can see in the print-out that there is an extra parameter ``surfgrav__``
    and an extra constraint on ``surfgrav``, that takes care of the inverse
    transformation from the unbounded to the bounded parameter. From now on,
    it's probably the ``surfgrav__`` parameter that the fitting program wants
    to work with. Setting a value to ``surfgrav`` will be ignored, since it
    is fully constrained by its unbounded version.
    
    >>> tools.add_unbounded_from_bounded(star,'surfgrav',from_='prior')
    >>> print(star)
          teff 10715.193052                                   K - phoebe Effective temperature
        radius 3.579247                                    Rsol - phoebe Radius
          mass 4.66955987072                               Msol - phoebe Stellar mass
           atm kurucz                                        --   phoebe Bolometric Atmosphere model
     rotperiod 0.90517                                        d - phoebe Polar rotation period
       diffrot 0.0                                            d - phoebe (Eq - Polar) rotation period (<0 is solar-like)
         gravb 1.0                                           -- - phoebe Bolometric gravity brightening
      gravblaw zeipel                                        --   phoebe Gravity brightening law
          incl 60.0                                         deg - phoebe Inclination angle
          long 0.0                                          deg - phoebe Orientation on the sky (East of North)
      distance 10.0                                          pc - phoebe Distance to the star
         shape equipot                                       --   phoebe Shape of surface
        vgamma 0.0                                         km/s - phoebe Systemic velocity
           alb 1.0                                           -- - phoebe Bolometric albedo (alb heating, 1-alb reflected)
        redist 0.0                                           -- - phoebe Global redist par (1-redist) local heating, redist global heating
    irradiator False                                         --   phoebe Treat body as irradiator of other objects
          abun 0.0                                           --   phoebe Metallicity
         label B9V_2f7ccdfd-d8c9-43ac-ad6d-5b2d2f0a4d72      --   phoebe Name of the body
       ld_func claret                                        --   phoebe Bolometric limb darkening model
     ld_coeffs kurucz                                        --   phoebe Bolometric limb darkening coefficients
      surfgrav 3.99999999999                            [cm/s2] - phoebe Surface gravity
    surfgrav__ -4.4338065418                             m1 s-2 - phoebe Surface gravity
          mass 9.2856392721e+30                             n/a   constr {surfgrav}/constants.GG*{radius}**2
      surfgrav 99.9999999984                                n/a   constr (np.arctan({surfgrav__})/np.pi + 0.5)*(1.00000000e+03-3.16227766e+01) + 3.16227766e+01
    
    Now, we'll set the value for the newly introduced unbounded surface gravity 
    parameter ``surfgrav__`` from it's prior collect that value, and then
    check whether the surface gravity itself is the uniform distribution.

    >>> values = np.zeros((2,1000))
    >>> for i in range(1000):
    ...     star.set_value_from_prior('surfgrav__')
    ...     values[:,i] = star['surfgrav__'],star['surfgrav']

    Make a plot of the histograms. You can see that, although we were taking
    random values from the prior of the unbounded parameter (left panel, it has
    a pretty weird distribution), the original surface gravity has exactly the
    distribution that we wanted from the start! So the fitting programs can go
    nuts on the values without being restricted, the parameter that matters
    will have the required behaviour.
    
    >>> plt.figure()
    >>> plt.subplot(121)
    >>> plt.xlabel('Unbounded surface gravity [weird units]')
    >>> plt.hist(values[0],normed=True,bins=100)
    >>> plt.subplot(122)
    >>> plt.xlabel('log(Surface gravity [cm/s2]) [dex]')
    >>> plt.hist(values[1],normed=True,bins=20)

    .. image:: images/surface_gravity_bounds.png

    Finally, we check whether the unbounded surface gravity parameter is truly
    unbounded:
    
    >>> star['surfgrav__'] = -1e10
    >>> print(star['surfgrav'])
    3.5000000004
    >>> star['surfgrav__'] = +1e10
    >>> print(star['surfgrav'])
    4.99999999999
        
    @param parset: parameterSet containing the qualifier
    @type parset: ParameterSet
    @param qualifier: name of the parameter to transform
    @type qualifier: str
    """
    #-- create a new parameter with the similar properties as the original one,
    #   but without limits.
    new_qualifier = qualifier+'__'
    P = parset.get_parameter(qualifier)
    P_ = P.copy()
    P_.set_qualifier(new_qualifier)
    
    try:
        low,high = P_.transform_to_unbounded(from_=from_)
    except AttributeError:
        raise AttributeError("You cannot unbound a parameter ({}) that has no prior".format(qualifier))
    #-- throw out a previous one if there is one before adding it again.
    if new_qualifier in parset:
        thrash = parset.pop(new_qualifier)
    parset.add(P_)
    #parset.add_constraint('{{{qualifier:s}}} = np.tan(np.pi*(({{{new_qualifier:s}}}-{low:.8e})/({high:.8e}-{low:.8e})-0.5))'.format(**locals()))
    parset.add_constraint('{{{qualifier:s}}} = (np.arctan({{{new_qualifier:s}}})/np.pi + 0.5)*({high:.8e}-{low:.8e}) + {low:.8e}'.format(**locals()))
    #return (np.arctan(par)/np.pi+0.5)*(high-low)+low

#}





#{ Other stuff

def group(observations, name, scale=True, offset=True):
    """
    Group a list of observations for simultaneous fitting of pblum and l3.
    
    This is can be used to make multicolour photometry has the same scaling,
    e.g. to reflect an unknown distance in SED fitting.
    
    Observations are grouped by name, so you need to give one. If you call this
    function twice on a different group of observations but you give the same
    name, they will all belong to the same big happy group.
    
    @param observations: a list of observations
    @type observations: list of Datasets
    @param name: name of the group
    @type name: str
    @param scale: fit passband luminosity (scaling factor)
    @type scale: bool
    @param offset: fit third light contribution or systemic velocity (constant term)
    @type offset: bool
    """
    
    for obs in observations:
        if not 'group' in obs:
            obs.add(parameters.Parameter(qualifier='group', value=name,
                                         context=obs.context,
                                         adjust=False, frame='phoebe',
                                         cast_type=str, repr='%s',
                                         description='Group name for simultaneous scale and offset fitting'))
            if 'scale' in obs:
                obs.set_adjust('scale', scale)
            elif scale:
                raise ValueError('Observations of context {} have no scaling factor, set it to False when grouping'.format(obs.context))
            obs.set_adjust('offset', offset)


def scale_binary(system, factor):
    """
    Make a binary system uniformly bigger or smaller.
    
    This should have minimal to no effect on the light curve, and is effectively
    like just bringing the system closer or further. The only difference with
    this is that the scale of the system changes while the dynamical properties
    (period, mass) change appropriately (e.g. to still satisfy Kepler's third
    law). So surface gravities and such will change. (and since surface
    gravities have a minimal effect on the light curve, it will ultimately also
    change the light curve).
    
    Will probably not really scale stuff if there are constraines defined for
    certain parameters.
    """
    
    ids = []
    for parset in system.walk():
        
        # For each parameterSet, walk through all the parameters
        for qual in parset:
            
            
            # Ask a unique ID and check if this parameter has already been
            # treated. If so, continue to the next one.
            parameter = parset.get_parameter(qual)
            myid = parameter.get_unique_label()
            if myid in ids:
                continue
                            
            # Else remember the id
            ids.append(myid)
            
            # Scale if it is a scalable parameter.
            if qual == 'radius':
                parset[qual] = parset[qual] * factor
            
            elif qual == 'sma':
                parset[qual] = parset[qual] * factor
                
                # Still satisfy Kepler's third law if we're in a non
                # roche system:
            raise NotImplementedError
    


def role_reverse(system):
    """
    Switch roles of primary and secondary.
    
    For simplicity and to avoid confusion, we assume the
    primary to be the first body in the system, and the
    secondary to be the second.
    
    Reverses the roles of primary and secondary in place.
    
    :param system: BodyBag containing two BinaryRocheStars (primary, secondary)
    :type system: BodyBag
    """
    primary = system[0]
    secondary = system[1]
    
    if not (primary.get_component() == 0 or secondary.get_component==1):
        raise IndexError("Primary is not the first star in the system, I refuse reversing the roles before you swap them, because otherwise confusion will be total.")
    
    comp1 = primary.params['component']
    comp2 = secondary.params['component']
    orbit = primary.params['orbit']
    
    pot1_orig = comp1['pot']
    pot2_orig = comp2['pot']
    q_orig = orbit['q']
    phshift_orig = orbit['phshift']
    per0_orig = orbit['per0']
    
    # Compute the transformed potentials
    comp2['pot'] = pot2_orig/q_orig + (q_orig-1)/(2*q_orig)
    comp1['pot'] = pot1_orig/q_orig + (q_orig-1)/(2*q_orig)
    
    # Reverse the mass ratio
    orbit['q'] = 1./q_orig
    
    # Shift phase by 0.5
    orbit['phshift'] = phshift_orig + 0.5
    
    # Shift argument of periastron by pi
    orbit['per0'] = per0_orig - np.pi
    
    # Interchange the labels in the orbit
    c1label = orbit['c1label']
    c2label = orbit['c2label']
    orbit['c1label'] = c2label
    orbit['c2label'] = c1label
    
    #pblum1 = primary.params['pbdep']['lcdep'].values()[0]['pblum']
    #pblum2 = secondary.params['pbdep']['lcdep'].values()[0]['pblum']
    #primary.params['pbdep']['lcdep'].values()[0]['pblum'] = pblum2
    #secondary.params['pbdep']['lcdep'].values()[0]['pblum'] = pblum1
    
    system.bodies = [secondary, primary]
    




def list_available_units(qualifier, context=None):
    """
    List the available units of a parameter.
    
    If you don't give a context (e.g. ``lcdep``), a report from all context where
    the parameters is found will be given. Otherwise only that particular
    context will be used. Alternatively you can give a full parameterSet to
    the context parameter. In the latter case, the context will be derived from
    the ParameterSet.
    
    @param qualifier: name of the unit to list the available units
    @type qualifier: str
    @param context: name of the context
    @type context: str or ParameterSet
    """
    print("List of available units for parameter {}:".format(qualifier))
    if context is not None and not isinstance(context,str):
        context = context.get_context()
    
    previous = None
    for idef in parameters.defs.defs:
        if qualifier == idef['qualifier']:
            
            if context is not None and not (context in idef['context']):
                continue
            
            if context is None:
                this_context = idef['context']
            else:
                this_context = context
            
            par = parameters.Parameter(**idef)
            unit_type, units = par.list_available_units()
            name = '* context {} ({}): '.format(this_context, unit_type)
            indent = ' '*len(name)
            
            # Don't do unnecessary repeats
            if unit_type == previous:
                print("{}same as above".format(name))
                continue
            else:
                previous = unit_type
            
            lines = []
            for i, unit in enumerate(units):
                
                start = name if i == 0 else indent
                
                expl = conversions._factors[unit]
                conv = '{:.6e} {}'.format(expl[0], expl[1])
                #contents = "\n".join(textwrap.wrap("- {} ({} - {:>16s}) ".format(unit, expl[3], conv),
                                                   #initial_indent=start, subsequent_indent=indent))
                contents = "{}- {:10s} {:22s}".format(start, unit, "("+expl[3]+")")
                contents = "{} ({:>14s}) ".format(contents, conv)
                lines.append(contents)
            
            print("\n".join(lines))
                
       
#}
