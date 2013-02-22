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
    either mass or radius from the other two.
    
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
    
def add_rotperiodcrit(star,rotperiodcrit,**kwargs):
    kwargs.setdefault('adjust',False)
    kwargs.setdefault('context',star.context)
    kwargs.setdefault('description','Polar rotation period/ critical period')
    kwargs.setdefault('frame','phoebe')
    
    star.pop_constraint('rotperiodcrit',None)
    star.pop_constraint('rotfreqcrit',None)
    star.pop('rotfreqcrit',None)
    if not 'rotperiodcrit' in star:
        star.add(parameters.Parameter(qualifier='rotperiodcrit',
                                      value=rotperiodcrit,**kwargs))
    else:
        star['rotperiodcrit'] = rotperiodcrit
    star.add_constraint('{rotperiod} = {rotperiodcrit}*2*np.pi*np.sqrt(27*{radius}**3/(8*constants.GG*{mass}))')
    logger.info("star '{}': 'rotperiod' constrained by 'rotperiodcrit'".format(star['label']))
    
def add_rotfreqcrit(star,rotfreqcrit,**kwargs):
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
    

#}