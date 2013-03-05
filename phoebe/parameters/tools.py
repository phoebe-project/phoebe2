"""
Tools to handle parameters and ParameterSets.
"""
import logging
from phoebe.parameters import parameters

logger = logging.getLogger("PARS.TOOLS")

#{ Common constraints for the star class


def add_surfgrav(star,surfgrav,derive='mass',unit='[cm/s2]',**kwargs):
    """
    Add surface gravity to a Star parameterSet.
    
    Only two parameters out of surface gravity, mass and radius are
    independent. If you add surface gravity, you have to choose to derive
    either mass or radius from the other two:
    
    .. math::
    
        g = G  M / R^2
    
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
    
    #-- remove any constraints on surfgrav and add the parameter
    star.pop_constraint('surfgrav',None)
    if not 'surfgrav' in star:
        star.add(parameters.Parameter(qualifier='surfgrav',value=surfgrav,
                                      **kwargs))
    else:
        star['surfgrav'] = surfgrav
        
    #-- specify the dependent parameter
    if derive=='mass':
        star.pop_constraint('radius',None)
        star.add_constraint('{mass} = {surfgrav}/constants.GG*{radius}**2')
    elif derive=='radius':
        star.pop_constraint('mass',None)
        star.add_constraint('{radius} = np.sqrt((constants.GG*{mass})/{surfgrav})')
    else:
        raise ValueError("Cannot derive {} from surface gravity".format(derive))
    logger.info("star '{}': '{}' constrained by 'surfgrav'".format(star['label'],derive))
   
    
def add_rotfreqcrit(star,rotfreqcrit,**kwargs):
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
    if not 'rotfreqcrit' in star:
        star.add(parameters.Parameter(qualifier='rotfreqcrit',
                                      value=rotfreqcrit,**kwargs))
    else:
        star['rotfreqcrit'] = rotfreqcrit
    star.add_constraint('{rotperiod} = 2*np.pi*np.sqrt(27*{radius}**3/(8*constants.GG*{mass}))/{rotfreqcrit}')
    logger.info("star '{}': 'rotperiod' constrained by 'rotfreqcrit'".format(star['label']))
    
    
def add_teffpolar(star,teffpolar,**kwargs):
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
    if not 'teffpolar' in star:
        star.add(parameters.Parameter(qualifier='teffpolar',
                                      value=teffpolar,**kwargs))
    else:
        star['teffpolar'] = teffpolar
    star.add_constraint('{teff} = {teffpolar}')
    logger.info("star '{}': 'teff' redefined to be equal to 'teffpolar'".format(star['label']))
    

def add_solarosc(star,numax,Deltanu0=None,unit='muHz'):
    """
    Add numax and Deltanu0 to a Star.
    
    See Kjeldsen 1995 for the relations:
    
    Given everything in solar units (also frequencies):
    
    .. math::
        
        \mathrm{surfgrav} =G \sqrt{T_\mathrm{eff}}  f_\mathrm{max}\quad [\mathrm{solar units}]
        
        \mathrm{radius} = \sqrt{T_\mathrm{eff}} f_\mathrm{max}/(\Delta f_0)^2 \quad [\mathrm{solar units}]
        
    """
    add_solarosc_numax(star,numax,unit=unit)
    add_solarosc_Deltanu0(star,Deltanu0,unit=unit)

def add_solarosc_numax(star,numax,unit='muHz',**kwargs):
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
   
    