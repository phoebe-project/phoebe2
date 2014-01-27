"""
List of user-defined contraints.
"""
from phoebe.dynamics import keplerorbit
from phoebe.atmospheres import roche
import logging

logger = logging.getLogger('PROC')

def test_constraint(self):
    self.params['star']['incl'] = 2*self.params['mesh']['delta']
    
    
def binary_light_curve_constraints(self,time,eclipse_separation=0.5):
    ecc = self[0].params['orbit']['ecc']
    omega = keplerobit.per0_from_eclipse_separation(eclipse_separation,ecc)
    self[0].params['orbit']['per0'] = omega,'rad'
    
def wd_eclipse_separation(self,time,eclipse_separation=0.5):
    ecc = self.params['root']['ecc']
    omega = keplerorbit.per0_from_eclipse_separation(eclipse_separation,ecc)
    self.params['root']['omega'] = omega,'rad'
    logger.info('Derived omega={} from eclipse separation={} and ecc={}'.format(omega,eclipse_separation,ecc))


def binary_teffratio(self, time, teff_ratio=0.5, fix=0):
    """
    Teff ratio means:
    
        teff_non_fixed = teff_ratio * teff_fixed
    """
    self[1-fix].params['component']['teff'] = teff_ratio * self[fix].params['component']['teff']**( (-1)**fix)



def binary_morphology(self, time):
    """
    Take care of constrained morphology in binary systems.
    
    The :ref:`morphology <label-morphology-component-phoebe>` type is set by
    equally-named parameter in the :ref:`component <parlabel-phoebe-component>`
    parameterSet. It can handle any of the following values:
        
        - unconstrained: doesn't add anything
        - detached: sets the minimum value of the potential to be the critical one
        - semi-detached: the component fills its critical Roche lobe
        - overcontact: set the maximum value of the potential to be the critical one
    
    """
    orbit = self.params['orbit']
    component = self.params['component']
    morphology = component.get('morphology', 'detached')
    
    if morphology == 'unconstrained':
        return None
    
    elif morphology in ['detached', 'semi-detached', 'overcontact']:
        
        # Retrieve the necessary information on the component and orbit
        q = orbit['q']
        ecc = orbit['ecc']
        F = component['syncpar']
        comp = self.get_component()+1
        d = 1-ecc
        
        # Compute critical potential
        critpot = roche.calculate_critical_potentials(q, F=F, d=d, component=comp)[0]
        
        # Everything has to happen with the potential parameter:
        potpar = component.get_parameter('pot')
        
        # Semi-detached: fix the potential to its critical value
        if morphology == 'semi-detached':
            potpar.set_value(critpot)
            logger.info('{} potential set to critical (semi-detached): pot={}'.format('Primary' if comp==1 else 'Secondary',critpot))
            
        # Detached: lower limit on potential
        elif morphology == 'detached':
            potpar.set_limits(llim=critpot,ulim=1e10)
            logger.info('{} lower limit on potential set to critical (detached): pot>={}'.format('Primary' if comp==1 else 'Secondary',critpot))
        
        # Overcontact: upper limit on potential
        elif morphology == 'overcontact':
            potpar.set_limits(llim=0, ulim=critpot)
            logger.info('{} upper limit on potential set to critical (overcontact): pot<={}'.format('Primary' if comp==1 else 'Secondary',critpot))
  
  
  
def sed_scale_to_distance(self, time, group):
    """
    Transform the SED scaling factor to a distance.
    
    We read the "pblum" value from the photometry with name ``group`` and
    use it to correct the distance to the object. The corrected distance in
    stored in the ``globals`` ParameterSet as ``derived_distance``.
    
    This only works if ``time==None``, such that the rescaling is only done
    after all computations are done.
    
    Usage:
    
    1. First group your photometry. The easiest way to do this is by parsing
       a group name to :py:func:`phoebe.parameters.datasets.parse_phot`.
    2. Add a parameter :envvar:`derived_distance` to the global parameterSet,
       via soemting like::
       
            globals = system.get_globals()
            globals.add(phoebe.parameters.parameters.Parameter(qualifier='derived_distance', cast_type=float, value=1.0))
    
    @param group: name of the photometry group to read the scaling from.
    @type group: str
    """
    # Don't rescale in the middle of the computations, only do it after everyting
    # was computed (see observatory.compute)
    if time is not None:
        return None
    
    pblum = None
    globals = self.get_globals()
    # Walk over everything
    for loc, thing in self.walk_all(path_as_string=False):
        # Look for lcobs
        if isinstance(thing, str) and thing == 'lcobs':
            # Look for group members
            for ref in loc[-2]['lcobs']:
                obs = loc[-2]['lcobs'][ref]
                # We only need one dataset, since all pblums are the same within the
                # group. If we found a good lcdep, we can break
                if 'group' in obs and obs['group'] == group:
                    pblum = obs['pblum']
                    break
            # Next few lines make sure we break out of the inner loop as well
            else:
                continue
            break
    else:
        raise ValueError("No pblum found for scaling of SED")
    
    # If we needed to multiply the model with 2, we need to put the thing
    # two times closer. To ensure consistency we (should have) added a new
    # parameter to the globals. This also ensure that if we call postprocess
    # two times, we're not twice dividing by two.
    globals['derived_distance'] = globals['distance'] / pblum
                
            
                