"""
Tools to handle parameters and ParameterSets, and add nonstandard derivative parameters.

**Constraints for the star/component context**

.. autosummary::
    
    add_vsini
    add_rotfreqcrit
    add_solarosc
    add_solarosc_Deltanu0
    add_solarosc_numax
    add_surfgrav
    add_teffpolar
    add_angdiam

**Constraints for orbit context**

.. autosummary::
    
    add_asini
    add_conserve
    from_perpass_to_supconj
    from_supconj_to_perpass

**Constraints for oscillations**

.. autosummary::

    add_amplvelo

**Helper functions**

.. autosummary::

    add_unbounded_from_bounded
    
"""
import logging
import numpy as np
from phoebe.units import constants
from phoebe.parameters import parameters

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
   
def add_vsini(star,vsini,derive='rotperiod',unit='km/s',**kwargs):
    r"""
    Add the vsini to a Star.
    
    The rotation period will then be constrained by the vsini of the star
    
    .. math::
    
        \mathrm{rotperiod} = \frac{2\pi R}{v\sin i} \sin i
    
    or the inclination angle
    
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
    
    star.pop_constraint('vsini',None)
    star.pop_constraint('rotperiodcrit',None)
    star.pop_constraint('rotperiod',None)
    star.pop('rotperiodcrit',None)
    if not 'vsini' in star:
        star.add(parameters.Parameter(qualifier='vsini',
                                      value=vsini,**kwargs))
    else:
        star['vsini'] = vsini
        
    if derive=='rotperiod':
        star.add_constraint('{rotperiod} = 2*np.pi*{radius}/{vsini}*np.sin({incl})')
        logger.info("star '{}': 'rotperiod' constrained by 'vsini' and 'radius' and 'incl'".format(star['label']))
    elif derive=='incl':
        star.add_constraint('{incl} = np.arcsin({rotperiod}*{vsini}/(2*np.pi*{radius}))')
        logger.info("star '{}': 'incl' constrained by 'vsini' and 'radius' and 'rotperiod'".format(star['label']))
    else:
        star.add_constraint('{vsini} = (2*np.pi*{radius})/{rotperiod}*np.sin({incl})')
        logger.info("star '{}': 'vsini' constrained by 'radius', 'rotperiod' and 'incl'".format(star['label']))
    
    
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
    
    
#}


#{ Common constraints for the BinaryRocheStar or Orbit

def add_asini(orbit,asini,derive='sma',unit='Rsol',**kwargs):
    """
    Add asini to an orbit parameterSet.
    
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
    
    Extra C{kwargs} will be passed to the creation of C{asini} if it does
    not exist yet.
   
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
    else:
        raise ValueError("Cannot derive {} from asini".format(derive))
    logger.info("orbit '{}': '{}' constrained by 'asini'".format(orbit['label'],derive))


def add_conserve(orbit,conserve='volume',**kwargs):
    """
    Add a parameter to conserve volume or equipotential value along an eccentric orbit.
    
    @param orbit: orbital parameterSet
    @type orbit: parameterSet of context orbit
    @param conserve: what to conserve, volume or equipotential
    @type conserve: str, one of `periastron' or `equipot'
    """
    if kwargs and 'conserve' in orbit:   
        raise ValueError("You cannot give extra kwargs to add_conserve if it already exist")
    
    kwargs.setdefault('description','Sets the quantity to be conserved along an eccentric orbit')
    kwargs.setdefault('choices',['periastron','sup_conj','inf_conj','asc_node','desc_node','equipot'])
    kwargs.setdefault('context',orbit.context)
    kwargs.setdefault('frame','phoebe')
    
    if not 'conserve' in orbit:
        orbit.add(parameters.Parameter(qualifier='conserve',value=conserve,
                                      **kwargs))
    else:
        orbit['conserve'] = conserve
        
    logger.info("orbit '{}': added and set 'conserve' to '{}'".format(orbit['label'],conserve))


def from_supconj_to_perpass(orbit):
    """
    Convert an orbital set where t0 is superior conjunction to periastron passage.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have T0 as superior conjunction.
    
    Inverse function is L{from_perpass_to_supconj}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    @param orbit: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit: parameterset of frame C{phoebe} and context C{orbit}
    """
    t_supconj = orbit['t0']
    phshift = orbit['phshift']
    P = orbit['period']
    per0 = orbit.get_value('per0','rad')
    t0 = t_supconj + (phshift - 0.25 + per0/(2*np.pi))*P
    orbit['t0'] = t0

def from_perpass_to_supconj(orbit):
    """
    Convert an orbital set where t0 is periastron passage to superior conjunction.
    
    Typically, parameterSets coming from Wilson-Devinney or Phoebe Legacy
    have T0 as superior conjunction.
    
    Inverse function is L{from_supconj_to_perpass}.
    
    See Phoebe Scientific reference Eqs. (3.30) and (3.35).
    
    @param orbit: parameterset of frame C{phoebe} and context C{orbit}
    @type orbit: parameterset of frame C{phoebe} and context C{orbit}
    """
    t_perpass = orbit['t0']
    phshift = orbit['phshift']
    P = orbit['period']
    per0 = orbit.get_value('per0','rad')
    t0 = t_perpass - (phshift - 0.25 + per0/(2*np.pi))*P
    orbit['t0'] = t0

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
    
    #-- remove any constraints on amplvelo and add the parameter
    star.pop_constraint('amplvelo',None)
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
        star.pop_constraint('ampl',None)
        star.add_constraint('{ampl} = {amplvelo}/(2*np.pi*{freq})')
        logger.info("puls '{}': '{}' constrained by 'amplvelo' and 'freq'".format(puls['label'],derive))
    else:
        raise ValueError("Cannot derive {} from amplvelo".format(derive))
    



#}

#{ Other constraints

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
    