
import logging
from phoebe.backend import universe

logger = logging.getLogger("FRONTEND.BACKENDS")
logger.addHandler(logging.NullHandler())

def set_param_legacy(phb1, param, value):
    params = {'incl': 'phoebe_incl', 'syncpar': 'phoebe_f#'}
    
    if param not in params.keys():
        logger.warning('{} parameter ignored in phoebe legacy'.format(param))
    else:    
        phb1.set_par(params[param].replace('#', str(i+1)), value)
    
def compute_legacy(system, *args, **kwargs):
    

  
    # import phoebe legacy    
    
    # check to make sure only binary
    if not hasattr(system, 'bodies') or len(system.bodies) != 2:
        raise TypeError("object must be a binary to run phoebe legacy")
        return
        
    # check to make sure BRS
    if not all([isinstance(body, univese.BinaryRocheStar) for body in system.bodies]):
        raise TypeError("both stars must be BinaryRocheStars to run phoebe legacy")
        return
        
    # check for mesh:wd PS
    # if mesh:wd:
    #     use
    # elif mesh:marching:
    #     convert using options in **kwargs
    
    # check for any non LC/RV and disable with warning
    
    # create phoebe legacy system
  
    for i,obj in enumerate(system):
        ps = obj.params['component']
        for param in ps:
            set_param_legacy(phb1, param, ps.get_value(param))
            
    ps = obj.params['orbit']
    for params in ps:
        set_param_legacy(phb1, param, ps.get_value(param))
            
    # run phoebeBackend
    
    # fill system mesh
    
    # fill synthetics
    
