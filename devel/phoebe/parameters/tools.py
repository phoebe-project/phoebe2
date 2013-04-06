"""
Tools to handle parameters and ParameterSets.

.. autosummary::
    
    add_asini
    add_vsini
    add_rotfreqcrit
    add_solarosc
    add_solarosc_Deltanu0
    add_solarosc_numax
    add_surfgrav
    add_teffpolar
    add_angdiam

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
    """
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
        star.pop_constraint('radius',None)
        star.add_constraint('{mass} = {surfgrav}/constants.GG*{radius}**2')
        logger.info("star '{}': '{}' constrained by 'surfgrav' and 'radius'".format(star['label'],derive))
    elif derive=='radius':
        star.pop_constraint('mass',None)
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
    
        \mathrm{incl} = np.arcsin(\frac{P v\sin i}{2\pi R})
    
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
        logger.info("star '{}': 'rotperiod' constrained by 'vsini'".format(star['label']))
    elif derive=='incl':
        star.add_constraint('{incl} = np.arcsin({rotperiod}*{vsini}/(2*np.pi*{radius}))')
        logger.info("star '{}': 'incl' constrained by 'vsini'".format(star['label']))
    
    
def add_rotfreqcrit(star,rotfreqcrit=None,**kwargs):
    """
    Add the critical rotation frequency to a Star.
    
    The rotation period will then be constrained by the critical rotation
    frequency:
    
    .. math::
    
        \mathrm{rotperiod} = 2\pi \sqrt{27 R^3 / (8GM)} / \mathrm{rotfreqcrit}
    
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
        rotfreqcrit = 2*np.pi*np.sqrt(27*radius**3/(8*constants.GG*mass))/rotperiod
    
    if not 'rotfreqcrit' in star:
        star.add(parameters.Parameter(qualifier='rotfreqcrit',
                                      value=rotfreqcrit,**kwargs))
    else:
        star['rotfreqcrit'] = rotfreqcrit
    star.add_constraint('{rotperiod} = 2*np.pi*np.sqrt(27*{radius}**3/(8*constants.GG*{mass}))/{rotfreqcrit}')
    logger.info("star '{}': 'rotperiod' constrained by 'rotfreqcrit'".format(star['label']))
    
    
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
   

def transform_bounded_to_unbounded(parset,qualifier,from_='limits'):
    r"""
    Transform a bounded parameter to an unbounded version.
    
    This can be helpful for inclusion in fitting algorithms that cannot handle
    bounds.
    
    The original parameter will be kept, but a transformed one will be added.
    The two versions of the parameters are linked through a constraint.
    
    The transformation of a parameter :math:`P` with upper limit :math:`U`
    and :math:`L` to an unbounded parameter :math:`P'` is given by:
    
    .. math::
    
        P' = \left(\frac{\atan(P)}{\pi} + \frac{1}{2}\right) (U-L) + L
        
        P = \tan\left(\pi\left(\frac{P'-L}{U-L}-\frac{1}{2}\right)\right)
    
    We also need to transform the prior accordingly.
    
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
    if P_.has_unit():
        del(P_.unit)
    #-- throw out a previous one if there is one before adding it again.
    if new_qualifier in parset:
        thrash = parset.pop(new_qualifier)
    parset.add(P_)
    #parset.add_constraint('{{{qualifier:s}}} = np.tan(np.pi*(({{{new_qualifier:s}}}-{low:.8e})/({high:.8e}-{low:.8e})-0.5))'.format(**locals()))
    parset.add_constraint('{{{qualifier:s}}} = (np.arctan({{{new_qualifier:s}}})/np.pi + 0.5)*({high:.8e}-{low:.8e}) + {low:.8e}'.format(**locals()))
    #return (np.arctan(par)/np.pi+0.5)*(high-low)+low
    