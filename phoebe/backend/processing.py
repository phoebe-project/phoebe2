"""
List of user-defined preprocessors.
"""
from phoebe.dynamics import keplerorbit
from phoebe.atmospheres import roche
from phoebe.units import constants
from numpy import pi, sin, cos, sqrt
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


def gray_scattering(self, time):
    """
    Gray scattering means the same albedo in all passbands.
    
    This function runs over all albedo's, and founds the one which is adjustable.
    It takes the value of that one, and forces it onto the other ones. This is
    done on a body-to-body basis.
    """
    
    for body in self.get_bodies():
        
        # We collect parameterSets because that allows us to keep evaluating
        # constraints automatically. Keep track of the reference albedo (i.e.
        # the adjustable one)
        ps_with_alb = []
        reference_albedo = None
        
        for path, val in body.walk_all():
            if isinstance(val, parameter.ParameterSet):
                if 'alb' in val:
                    
                    # Keep track of reference albedo
                    if val.get_adjust('alb'):
                        if reference_albdo is not None:
                            raise ValueError("More than one albedo is set to be fitted in Body {}, can't figure out what gray scattering means in this context".format(body.get_label()))
                        reference_albedo = val.get_value('alb')
                    else:
                        # Keep track of PSet
                        ps_with_alb.append(val)
        
        # If there are albedos but none are adjustable, what do we need to set then?
        if reference_albedo is None and len(ps_with_albedo):
            raise ValueError("Albedos found in Body {}, but no reference albedo. Can't figure out what gray scattering means...".format(body.get_label()))

        # Now set the albedo's to be equal
        for ps in ps_with_alb:
            ps['alb'] = reference_albedo
            
                    
                
        
        


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
    # Don't rescale in the middle of the computations, only do it after everyting
    # was computed (see observatory.compute)
    if time is not None:
        return None
    
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
                
            
def binary_custom_variables(self, time):
    """
    Runs over all parameters and solves for the values of custom parameters and their relations.
    
    Given the complexity and customizibility of this job, the relations are
    hardcoded.
    """
    
    comp1 = self[0].params['component']
    comp2 = self[1].params['component']
    orbit = self[0].params['orbit']
    
    for loc, param in self.walk_all(path_as_string=False):
        
        # get the current component integer. This will only work for BinaryRoche
        # Stars, we'll let everything else slip through.
        try:
            component = param.get_component()
        except:
            pass
        
        # if this thing has a dependable, it must be a Parameter for which
        # relations hold. These relations *must* be predefined here, otherwise
        # we raise an Error
        try:
            replaces = param.get_replaces()
        except:
            continue
        
        if replaces:
            
            qualifier = param.get_qualifier()
            this_component = 0
            
            # Projected semi-major axis: asini = sma * sin(incl)
            if qualifier == 'asini':
                
                sma = orbit.request_value('sma','SI')
                incl = orbit.request_value('incl','SI')
                asini = orbit.request_value('asini','SI')
                
                # Derive sma <--- incl & asini
                if replaces == 'sma':
                    sma = asini / np.sin(incl)
                    orbit['sma'] = sma, 'SI'
                # Derive incl <--- asini & sma
                elif replaces == 'incl':
                    incl = np.arcsin(asini / sma)
                    orbit['incl'] = incl, 'SI'
                # Derive asini <--- sma & incl
                else:
                    asini = sma * np.sin(incl)
                    orbit['asini'] = asini, 'SI'
            
            # Add mass as a parameter
            elif qualifier == 'mass':
                this_mass = self[component].params['component'].get_parameter('mass')
                sma = orbit.request_value('sma','SI')
                period = orbit.request_value('period', 'SI')
                q = orbit['q']
                mass = this_mass.get_value('SI')
                
                # Derive primary mass <--- sma, period & q
                if replaces == 'mass' and component == 0:
                    mass = 4*pi**2 * sma**3 / period**2 / constants.GG / (1.0 + q)
                    this_mass.set_value(mass, 'SI')
                # Derive secondary mass <--- sma, period & q
                elif replaces == 'mass' and component == 1:
                    mass = 4*pi**2 * sma**3 / period**2 / constants.GG / (1.0 + 1.0/q)
                    this_mass.set_value(mass, 'SI')
                else:
                    raise NotImplementedError("Don't know how to derive '{}' from 'mass'".format(dependable))
            
            # Add polar radius as a parameter
            elif qualifier == 'radius':
                this_radius = self[component].params['component'].get_parameter('radius')
                this_pot = self[component].params['component'].get_parameter('pot')
                pot = this_pot.get_value()
                sync = self[component].params['component']['syncpar']
                d = 1-orbit['ecc']
                q = orbit['q']
                sma = orbit['sma']
                
                
                if component == 1: # means secondary
                    this_q, this_pot = roche.change_component(q, pot)
                
                # Derive radius <--- potential, q, d, F
                if replaces == 'radius':
                    radius = roche.potential2radius(pot, q, d=d,F=sync, sma=sma)
                    this_radius.set_value(radius,'Rsol')
                    
                    # We can't have both of pot and radius to be adjustable
                    if this_pot.get_adjust():
                        this_radius.set_adjust(True)
                        this_pot.set_adjust(False)
                        
                # Derive potential <--- radius, q, d, F
                elif replaces == 'pot':
                    radius = this_radius.get_value()
                    pot = roche.radius2potential(radius, q, d=d,F=sync, sma=sma)
                    this_pot.set_value(pot)
                    # We can't have both of pot and radius to be adjustable
                    if this_radius.get_adjust():
                        this_pot.set_adjust(True)
                        this_radius.set_adjust(False)                    
                    
                else:
                    raise NotImplementedError("Don't know how to derive '{}' from 'radius'".format(dependable))

            
            # Add vsini as a parameter
            elif qualifier == 'vsini':
                raise NotImplementedError
            
            elif qualifier == 'logg':
                raise NotImplementedError
            
            elif qualifier == 'ecosw':
                raise NotImplementedError
            
            elif qualifier == 'esinw':
                raise NotImplementedError
            
            elif qualifier == 'teffratio':
                raise NotImplementedError
            
            else:
                raise NotImplementedError("Don't know how to connection {} with {}".format(param.get_qualifier(), param.connections))
                    