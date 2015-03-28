import os
import functools
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
from collections import OrderedDict
from fnmatch import fnmatch
import copy
import json
import uuid
import logging
import glob
try: # Pyfits now integrated in astropy
    import pyfits
except:
    import astropy.io.fits as pyfits
import textwrap
import numpy as np
import readline
from phoebe import __version__
from phoebe.parameters import parameters, datasets
from phoebe.parameters import datasets
from phoebe.parameters import feedback
from phoebe.backend import universe
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.dynamics import keplerorbit
from phoebe.atmospheres import roche
from phoebe.atmospheres import limbdark
from phoebe.utils import config

logger = logging.getLogger("FRONT.COMMON")

def to_time(phase, ephem=None, **kwargs):
    """
    [FUTURE]
    
    converts phases to times based on an ephemeris.  The ephemeris can be
    sent as a dictionary to ephem (ie the output from bundle.get_ephem), 
    or as kwargs.  Either will recognize the following (optional) keys: 
    period, t0, phshift, dpdt.
    
    @param phase: phases to convert to times
    @type phase: float or list of floats
    @param period: period (in days)
    @type period: float
    @param t0: t0
    @type t0: float
    @param phshift: phase shift
    @type phshift: float
    @param dpdt: rate of change of period
    @type dpdt: float
    @return: time
    @rtype: float or list of floats
    """
    if ephem is not None:
        for k,v in ephem.items():
            kwargs.setdefault(k,v)
    
    period  = kwargs.get('period', 0.0)
    t0      = kwargs.get('t0', 0.0)
    phshift = kwargs.get('phshift', 0.0)
    dpdt    = kwargs.get('dpdt', 0.0)
    
    if isinstance(phase, list):
        phase = np.array(phase)
    
    # t = t0 + (phase - phshift) * (period + dpdt(t - t0)) 
    return t0 + ((phase - phshift) * period) / (1 - (phase - phshift) * dpdt)


def rebuild_trunk(fctn):
    """
    Rebuild the cached trunk *after* running the original function
    
    [FUTURE]
    """
    @functools.wraps(fctn)
    def rebuild(container, *args, **kwargs):
        return_ = fctn(container, *args, **kwargs)
        container._build_trunk()
        return return_
    return rebuild


class Container(object):
    """
    Control accessing sections and ParameterSets
    
    This class in inherited by both the Bundle and Usersettings
    """
    
    def __init__(self):
        self.trunk = []
        self.sections = OrderedDict()
        
    ## act like a dictionary
    def keys(self):
        """
        Return a list of the Bundle's keys (aka twigs)
        
        :return: a list of twigs
        :rtype: list of str
        """
        return self.twigs()
        
    def values(self):
        """
        Return the values of all items in the Bundle as a list
        
        :return: list of all items
        :rtype: list
        """
        return [ti['item'] for ti in self.trunk if not ti['hidden']]
        
    def items(self):
        """
        Return a list of the Bundle's (key, value) pairs, as 2-tuples.
        
        :return: list of pairs (key, value)
        :rtype: list of 2-tuples
        """
        return [(ti['twig_full'], ti['item']) \
                        for ti in self.trunk if not ti['hidden']]
    
    
    def get(self, twig=None, default=None, **kwargs):
        """
        Search and retrieve any item without specifying its type.
        
        :param twig: the search twig
        :type twig: str
        :param default: return value if twig is not in Bundle
        :type default: any Python object (defaults to None)
        :return: matching item
        :rtype: undefined
        :raises KeyError: when twig is not present
        """
        # Set some overridable defaults
        kwargs.setdefault('return_trunk_item', True)
        kwargs.setdefault('all', False)
        kwargs.setdefault('ignore_errors', True)
        
        # Look for the twig
        try:
            ti = self._get_by_search(twig, **kwargs)
        except KeyError:
            # If we can't ignore this error, raise it anyway
            if not kwargs['ignore_errors']:
                raise
        
        # make sure to return the default value if the key does not exist
        if ti is None:
            return default
        
        # else, retrieve the item and return it. We need to treat Parameters as
        # values, and have special treatments for their adjustables and priors
        item = ti['item']
        
        if ti['kind'] == 'Parameter:adjust':
            return item.get_adjust()
        elif ti['kind'] == 'Parameter:prior':
            return item.get_prior()
        elif ti['kind'] == 'Parameter:posterior':
            return item.get_posterior()
        else:
            # either 'value' or item itself
            if isinstance(item, parameters.Parameter):
                return item.get_value()
            return item

    @rebuild_trunk
    def set(self, twig, value, **kwargs):
        """
        Set a value in the Bundle.
        
        Value can be anything the twig's target can handle.
        
        :param twig: the search twig
        :type twig: str
        :param value: the new value
        :type value: varies
        :raises KeyError: when twig is not present
        :raises TypeError: when value cannot be cast into the right type
        """
        kwargs['return_trunk_item'] = True
        kwargs['all'] = False
        kwargs['ignore_errors'] = False
        ti = self._get_by_search(twig, **kwargs)
        item = ti['item']
        
        if ti['kind'] == 'Parameter:adjust':
            # we call self.set_adjust instead of item.set_adjust
            # because self.set_adjust handles auto prior creation
            self.set_adjust(twig, value)
        
        elif ti['kind'] == 'Parameter:prior':
            self.set_prior(twig, value)
        
        # replace this Body with the given one
        elif ti['kind'] == 'Body':
            
            # Retrieve the current Body so that we can call it's Parent, and
            # override this child's place in the Bag
            current = self.get(twig)
            parent = current.get_parent()
            
            # If the current Body has not Parent, then it is probably a single
            # Star (or other Body). Then replace the whole system:
            if parent is None:
                self.set_system(value)
            
            # Else, we need to take care of *a lot* of stuff:
            else:
                # 1. Make sure the new Body has the same orbit properties as
                #    the original one:
                current_orbit = current.get_orbit()
                if current_orbit is not None:
                    value.set_params(current_orbit)
                # 2. Make sure that the new Body has the same label as the 
                #    original Body
                value.set_label(current.get_label())
                # 3. Make sure all the labels of the Bodies are still unique
                #    If not give those a unique identifier
                labels = [body.get_label() for body in self.get_system().get_bodies()]
                for new_body in value.get_bodies():
                    if new_body.get_label() in labels:
                        new_body.set_label(uuid.uuid4())
                # 4. Make sure the mass ratio is correct and the semi-major axis
                #    satisfies Kepler's third law
                if current_orbit is not None:
                    print("Old mass = {}, old q = {}, old sma = {}".format(current.get_mass(), current_orbit['q'], current_orbit['sma']))
                    mold_over_mnew = current.get_mass()/value.get_mass()
                    if current.get_component() == 0:
                        current_orbit['q'] = current_orbit['q'] * mold_over_mnew
                    else:
                        current_orbit['q'] = current_orbit['q'] / mold_over_mnew
                    totalmass = parent.get_mass() - current.get_mass() + value.get_mass()
                    period = current_orbit['period']
                    current_orbit['sma'] = universe.keplerorbit.third_law(totalmass=totalmass,
                                                                          period=period), 'au'
                    print("New mass = {}, new q = {}, new sma = {}".format(value.get_mass(), current_orbit['q'], current_orbit['sma']))
                # 5. Set the Parent of the new value to be the same as the old
                value.set_parent(current.get_parent())
                # Finally, we can replace the Body
                parent.bodies[parent.bodies.index(current)] = value
            
        else:
            # either 'value' or None
            if isinstance(value, parameters.ParameterSet):
                # special case for orbits, we need to keep c1label and c2label
                if value.get_context() == 'orbit':
                    oldorbit = self.get_ps('orbit@' + twig)
                    value['c1label'] = oldorbit['c1label']
                    value['c2label'] = oldorbit['c2label']                
                self.set_ps(twig, value)        
            
            elif isinstance(value, tuple) and len(value) == 2 and \
                                                    isinstance(value[1], str):
                # ability to set units
                self.set_value(twig, *value)

            else:
                self.set_value(twig, value)
    
    def __getitem__(self, twig):
        """
        Define dictionary style access.
        
        Returns Bodies, ParameterSets or Parameter values (never Parameters).
        
        :param twig: twig name
        :type twig: str
        :return: the value corresponding to the twig
        :rtype: whatever value the twig corresponds to
        :raises KeyError: when twig is not available
        """
        # dictionary-style access cannot return None ad a default value if the
        # key does not exist
        return self.get(twig, ignore_errors=False)
        
    
    def __setitem__(self, twig, value):
        """
        Set an item in the Bundle.
        
        :param twig: twig name
        :type twig: str
        :param value: new value for the twig's value
        :type value: whatever value the twig accepts
        :raises KeyError: when twig is not available
        :raises TypeError: when value cannot be cast to the right type
        """
        self.set(twig, value)
        
    def __contains__(self, twig):
        """
        Check if a twig is in the Container.
        """
        try:
            ret_value = self._get_by_search(twig, all=True)
            return True
        except KeyError:
            return False
    
    def __iter__(self):
        """
        Make the class iterable.
        
        Iterating a Bundle means iterating over its keys (like a dictionary).
        """
        return iter(self.keys())
    
    ## General interface
    
    def search(self, twig, **kwargs):
        """
        Return a list of twigs matching a substring search
        
        :param twig: the search twig
        :type twig: str
        :return: list of twigs
        :rtype: list of strings
        """
        return self._search_twigs(twig, **kwargs)
        
    def twigs(self, twig=None, **kwargs):
        """
        Return a list of matching twigs
        
        (with same method as used in get_value, get_parameter, get_prior, etc)
        
        :param twig: the search twig
        :type twig: str
        :return: list of twigs
        :rtype: list of strings
        """    
        if twig is None:
            trunk = self._filter_twigs_by_kwargs(**kwargs)
            return [t['twig_full'] for t in trunk] 
        return self._match_twigs(twig, **kwargs)
    
    
    def get_ps(self, twig):
        """
        Retrieve the ParameterSet corresponding to the twig.
        
        :param twig: the search twig
        :type twig: str
        :return: ParameterSet
        :rtype: ParameterSet
        :raises KeyError: when the twig does not represent a ParameterSet
        """
        return self._get_by_search(twig, kind='ParameterSet')
        
    def get_ps_dict(self, twig):
        """
        Retrieve a dictionary of ParameterSets in a list
        
        :param twig: the search twig
        :type twig: str
        :return: dictionary of ParameterSets
        :rtype: dict
        """
        return self._get_by_search(twig, kind="OrderedDict")

    def get_parameter(self, twig):
        """
        Retrieve a Parameter
        
        :param twig: the search twig
        :type twig: str
        :return: Parameter
        :rtype: Parameter
        :raises KeyError: if twig not available or not representing a Parameter
        """
        out = self._get_by_search(twig, kind='Parameter') # was 'Parameter*'
        return out
        
    def info(self, twig):
        """
        Retrieve info on a Parameter.
        
        This is just a shortcut to str(get_parameter(twig)), except when info
        on the passband is queried. In that case, a list of possible passbands
        in the default atmosphere files is included (and ordered per atmosphere
        file).
        
        :param twig: the search twig
        :type twig: str
        :return: info
        :rtype: str
        """
        info_str = str(self.get_parameter(twig))
        
        # Special addition for default support for passbands:
        if twig.split('@')[0] == 'passband':            
            # Get the location of the predefined ld_coeffs files
            path = limbdark.get_paths()[0]
            
            # Cycle over all FITS files and see what passbands are defined
            for filename in sorted(glob.glob(os.path.join(path, '*.fits'))):
                with pyfits.open(filename) as ff:
                    grid_passbands = [ext.header['extname'].rstrip('_v1.0') for ext in ff[1:]\
                                      if not ext.header['EXTNAME'][:4]=='_REF']
                text = os.path.basename(filename)+': '+", ".join(grid_passbands)
                info_passband = textwrap.fill(text, width=80, initial_indent='',
                                              subsequent_indent=' '*5)
                info_str += '\n' + info_passband
        
        return info_str
           
           
    def get_value(self, twig, *args):
        """
        Retrieve the value of a Parameter.
        
        If the Parameter has units, it is possible to do on-the-fly unit
        conversions:
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.get_value('sma')
        10.0
        >>> mybundle.get_value('sma', 'm')
        6955080000.0
        
        :param twig: the search twig
        :type twig: str
        :raises KeyError: when twig is not available or is not a Parameter
        """
        return self.get_parameter(twig).get_value(*args)
        
    def set_value(self, twig, value, unit=None):
        """
        Set the value of a Parameter
        
        :param twig: the search twig
        :type twig: str
        :param value: the value
        :type value: depends on Parameter
        :param unit: unit of value (if not default)
        :type unit: str or None
        :raises KeyError: when twig is not available or is not a Parameter
        """
        
        # Retrieve the parameter
        param = self.get_parameter(twig)
        
        # Check if it can be changed: that is not the case if it's replaced by
        # another parameter!
        if param.get_replaced_by():
            raise ValueError("Cannot change value of parameter '{}', it is derived from other parameters".format(twig))
        
        # Perhaps the parameter is write-protected in the frontend
        if param.is_write_protected():
            if hasattr(param, 'why_protected'):
                raise ValueError("Variable {} is write-protected ({})".format(param.qualifier, param.why_protected))
            else:
                raise ValueError("Variable {} is write-protected".format(param.qualifier))
                
        # special care needs to be taken when setting labels and refs
        qualifier = param.get_qualifier()
        old_value = param.get_value()
        
        if qualifier in ['label','ref']:
            if set([c for c in readline.get_completer_delims()]).intersection(value):
            # throw a warning if a special character in the label might break tab-completion
                logger.warning('special character in refs or labels may break tab completion')

        # Setting a label means not only changing that particular Parameter,
        # but also the property of the Body
        if qualifier == 'label':
            this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
            component = self._get_by_search(this_trunk['label'])
            component.set_label(value)
            
            # also change all objref (eg in plotting)
            params = self._get_by_search('objref@', all=True,ignore_errors=True)
            for param in params:
                if param.get_value()==old_value:
                    param.cast_type = str # will change back to choose in _build_trunk
                    param.set_value(value)
            
            self._build_trunk()
        
        # Changing a ref needs to change all occurrences
        elif qualifier == 'ref':
            # get the system
            from_ = param.get_value()
            system = self.get_system()
            system.change_ref(from_, value)
            
            # also change all dataref (eg in plotting)
            params = self._get_by_search('dataref@', all=True,ignore_errors=True)
            for param in params:
                if param.get_value()==old_value:
                    param.cast_type = str # will change back to choose in _build_trunk
                    param.set_value(value)
            
            self._build_trunk()
            return None
        
        # Make sure non-standard atmosphere tables are registered
        elif qualifier in ['ld_coeffs', 'atm']:
            limbdark.register_atm_table(value)
        
        if unit is None:
            param.set_value(value)
        else:
            param.set_value(value, unit)
        
            
    def set_value_all(self, twig, value, unit=None):
        """
        Set the value of all matching Parameters
        
        :param twig: the search twig
        :type twig: str
        :param value: the value
        :type value: depends on Parameter
        :param unit: unit of value (if not default)
        :type unit: str or None
        :raises KeyError: when twig is not available or is not a Parameter
        """
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
            if unit is None:
                param.set_value(value)
            else:
                param.set_value(value, unit)
    
    def get_value_all(self, twig):
        """
        Return the values of all matching Parameters
        
        :param twig: the search twig
        :type twig: str
        :return: dict of all values matching the Parameter twig
        :rtype: dict of twig/values
        :raises KeyError: when twig is not available or is not a Parameter
        """
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        # if we decide upon a list
        #return [param.get_value() for param in params]
    
        # if we decide upon a dictionary (I want ordered because that sometimes
        # makes sense)
        matched_twigs = self._match_twigs(twig, hidden=False)
        od = OrderedDict()
        for twig, param in zip(matched_twigs, params):
            od[twig] = param.get_value()
        return od
        #return {twig:param.get_value() for twig, param in zip(matched_twigs, params)}
        
    def set_main_period(self, twig):
        """
        Set the main period of the system to the value of the twig.
        
        [FUTURE]
        """
        pass
    
    @rebuild_trunk
    def set_ps(self, twig, value):
        """
        Replace an existing ParameterSet.
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :param value: the value
        :type value: ParameterSet
        """
        # get all the info we can get
        this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
        
        # Make sure it is a ParameterSet
        if this_trunk['kind'] != 'ParameterSet':
           raise ValueError(("Twig '{}' does not refer to a ParameterSet "
                             "(it is a {})").format(twig, this_trunk['kind']))
        
        # actually, we're mainly interested in the path. And in that path, we
        # are only interested in the last Body
        bodies = [self.get_system()] + [thing for thing in this_trunk['path'] \
                                            if isinstance(thing, universe.Body)]
        
        if not bodies:
            raise ValueError(('Cannot assign ParameterSet to {} (did not find '
                              'any Body)').format(twig))
        
        # last check: we need to make sure that whatever we're setting already
        # exists. We cannot assign a new nonexistent PS to the Body (well, we
        # could, but we don't want to -- that's attach_ps responsibility)
        current_context = this_trunk['item'].get_context()
        given_context = value.get_context()
        if current_context == given_context:
            bodies[-1].set_params(value)
        else:
            raise ValueError(("Twig '{}' refers to a ParameterSet of context "
                              "'{}', but '{}' is given").format(twig,\
                                                current_context, given_context))
        
    @rebuild_trunk
    def attach_ps(self, value=None, twig=None, ):
        """
        Attach a new ParameterSet.
        
        **Example usage**::
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.attach_ps(phoebe.PS('reddening:interstellar'), 'new_system')
        >>> mybundle.attach_ps(phoebe.PS('reddening:interstellar'))
        
        [FUTURE]
        
        :param value: new ParameterSet to be added (or context)
        :type value: ParameterSet (or string)
        :param twig: location in the system to add it to (defaults to the root system)
        :type twig: str or None
        """
        # If the value is a string, create a default parameterSet here
        if isinstance(value, str):
            value = parameters.ParameterSet(value)
        
        # By default, add the ParameterSet to the entire system
        if twig is None:
            twig = self.get_system().get_label()
            
        # Get all the info we can get
        this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
        
        if isinstance(this_trunk['item'], universe.Body):
            # last check: we need to make sure that whatever we're setting
            # already exists. We cannot assign a new nonexistent PS to the Body
            # (well, we could, but we don't want to -- that's attach_ps
            # responsibility)
            try:
                given_context = value.get_context()
                this_trunk['item'].set_params(value, force=False)
            except ValueError:
                raise ValueError(("ParameterSet '{}' at Body already exists. "
                                 "Please use set_ps to override "
                                 "it.").format(given_context))
            except AttributeError:
                raise ValueError("{} is not a ParameterSet".format(value))
        elif isinstance(this_trunk['item'], dict): #then this should be a sect.
            section = twig.split('@')[0]
            key = 'label' if 'label' in value.keys() else 'ref'
            if value.get_value(key) not in [c.get_value(key) \
                                               for c in self.sections[section]]:
                self._add_to_section(section, value)
                
        else:
            raise ValueError(("You can only attach ParameterSets to a Body or "
                              "Section ('{}' refers to a "
                              "{})").format(twig, this_trunk['kind']))
        
        logger.info("ParameterSet {} added to {}".format(value.get_context(), twig))
        
    def get_adjust(self, twig):
        """
        Retrieve whether a Parameter is marked to be adjusted
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :return: adjust
        :rtype: bool
        """
        return self.get_parameter(twig).get_adjust()
        
    def set_adjust(self, twig, value=True):
        """
        Set whether a Parameter is marked to be adjusted
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :param value: adjust
        :type value: bool
        """
        par = self.get_parameter(twig)
        
        # Check if it can be adjusted: that is not the case if it's replaced by
        # another parameter!
        if par.get_replaced_by():
            raise ValueError("Cannot adjust value of parameter '{}', it is derived from other parameters".format(twig))        
        
        # the following gives issues because some parameters do not have limits
        # - If get_limits would implement default -np.inf and +np.inf, then
        # par.pdf() would return zero probability for whatever value
        # - If get_limits would implement default rediciliously large numbers,
        #   then this will also impact the total probability, which could
        #   potentially screw up fitting methods
        # Verdict: <pieterdegroote> votes for not including it
        #if not par.has_prior() and par.get_qualifier() not in ['scale', 'offset']:
        #    lims = list(par.get_limits())
        #    par.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
        par.set_adjust(value)
        
    def set_adjust_all(self, twig='', value=True):
        """
        Set whether all matching Parameters are marked to be adjusted
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :param value: adjust
        :type value: bool
        """
        pars = self._get_by_search(twig, kind='Parameter', all=True)
        
        
        for par in pars:
            # <pieterdegroote> doesn't want the three following lines. See
            # set_adjust for a rationale.
            #if not par.has_prior() and par.get_qualifier() not in ['scale','offset']:
            #    lims = par.get_limits()
            #    par.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
            par.set_adjust(value)
    
    def get_prior(self, twig):
        """
        Retrieve the prior on a Parameter
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :return: prior
        :rtype: ParameterSet
        """
        return self.get_parameter(twig).get_prior()
        
    def set_prior(self, twig, **dist_kwargs):
        """
        Set the prior on a Parameter
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :param **kwargs: necessary parameters for distribution
        :type **kwargs: varies
        """
        param = self.get_parameter(twig)
        param.set_prior(**dist_kwargs)
        
    def set_prior_all(self, twig, **dist_kwags):
        """
        Set the prior on all matching Parameters
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :param **kwargs: necessary parameters for distribution
        :type **kwargs: varies
        """
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
            param.set_prior(**dist_kwargs)
            
    def get_adjustable_parameters(self, twig=None):
        """
        [FUTURE]
        
        Return the twigs of all adjustable parameters (those that are set
        to adjust AND have a prior)
        
        :param twig: the search twig/twiglet (or None to show all)
        :type twig: str
        :return: dictionary of twig/parameter pairs
        :rtype: dict
        """      
        params = {}
        for ti in self._get_by_search(twig=twig, kind='Parameter', all=True, return_trunk_item=True):
            if hasattr(ti['item'], 'adjust') and ti['item'].adjust and ti['item'].has_prior():
                params[ti['twig']] = ti['item']
        return params
                
    def set_posterior(self, twig, **dist_kwargs):
        """
        Set the posterior of a Parameter
        
        [FUTURE]
        
        To set the posterior from a trace, issue:
        
        >>> mybundle.set_posterior('incl@orbit', sample=mysample)
        
        """
        param = self.get_parameter(twig)
        param.set_posterior(**dist_kwargs)
    
    
    def get_value_from_posterior(self, twig, size=1):
        """
        Get values from the posterior distribution.
        
        
        [FUTURE]
        """
        return None
    
    def draw_value_from_posterior_all(self, twig='', size=1, only_adjust=True):
        """
        Get values from the posterior distribution.
        
        [FUTURE]
        
        Return type is a record array! Behaves like a dict, but more powerful
        (and a different string representation so you might get confused at
        first). Plus you get keys not via '.keys()' but via dtype.names
        """
        params = self._get_by_search(twig, kind='Parameter', all=True, return_trunk_item=True)
        
        # Filter out all adjustables
        if only_adjust:
            params = [param for param in params if param['item'].get_adjust()]
        
        # Sample from posterior and collect everything in a dictionary
        output = []
        keys = []
        
        indices = None
        for param in params:
            posterior = param['item'].get_posterior()
            if posterior is not None:
                values, indices = posterior.draw(size=size, indices=indices)
                output.append(values)
                keys.append(param['twig_full'])
        
        return np.rec.fromarrays(output, names=keys)
    
    
    def set_value_from_posterior_all(self, twig='', only_adjust=True):
        """
        Set value from the posterior.
        [FUTURE]
        """
        # draw values (1 value per twig, thus size=1 (default))
        values = self.draw_value_from_posterior_all(twig=twig, only_adjust=only_adjust)
        for twig in values.dtype.names:
            self.set_value(twig, values[twig][0])
        
    
    @rebuild_trunk
    def add_compute(self, ps=None, context='compute', **kwargs):
        """
        Add a new compute ParameterSet
        
        **Example usage:**
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.add_compute(label='very_detailed', subdiv_num=10)
        
        @param ps: compute ParameterSet (or None)
        @type ps:  None or ParameterSet
        @param label: label of the compute options (will override label in ps)
        @type label: str
        """
        if ps is None:
            ps = parameters.ParameterSet(context=context)
        for k,v in kwargs.items():
            ps.set_value(k,v)
            
        self._add_to_section('compute',ps)

        # had to remove this to make the deepcopy in run_compute work 
        # correctly.  Also - this shouldn't ever be called from within
        # the Container class but only within Bundle
        #~ self._attach_set_value_signals(ps)
    
    def get_compute(self, label=None, create_default=False):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @param create_default: whether to create and attach defaults if label is None
        @type create_default: bool
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        if label is None and create_default:
            # then see if the compute options 'default' is available
            if 'default' not in self._get_dict_of_section('compute').keys():
                # then create a new compute options from the backend
                # and attach it to the bundle with label 'default'
                self.add_compute(label='default')
                self._build_trunk()
            label = 'default'

        return self._get_by_section(label,"compute")    
        
    @rebuild_trunk
    def remove_compute(self, label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        compute = self.get_compute(label)
        self.sections['compute'].remove(compute)
        self._build_trunk() # needs to be called again to reset
        
    @rebuild_trunk
    def add_fitting(self, ps=None, **kwargs):
        """
        Add a new fitting ParameterSet
        
        **Example usage:**
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.add_fitting(context='fitting:emcee', iters=1000, walkers=100)
        
        [FUTURE]
        
        @param ps: fitting ParameterSet
        @type ps:  None, or ParameterSet
        @param label: name of the fitting options (will override label in ps)
        @type label: str
        """
        if ps is None:
            context = kwargs.pop('context') if 'context' in kwargs.keys() else 'fitting:lmfit'
            ps = parameters.ParameterSet(context=context)

        for k,v in kwargs.items():
            ps.set_value(k,v)
            
        self._add_to_section('fitting',ps)
            
    def get_fitting(self, label=None):
        """
        Get a fitting ParameterSet by name
        
        [FUTURE]
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section(label,"fitting")

    @rebuild_trunk
    def remove_fitting(self, label):
        """
        Remove a given fitting ParameterSet
        
        [FUTURE]
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        fitting = self.get_fitting(label)
        self.sections['fitting'].remove(fitting)
        self._build_trunk() # needs to be called again to reset
        
    def get_feedback(self, label=None):
        """
        Get a feedback ParameterSet by name
        
        [FUTURE]
        
        @param label: name of ParameterSet
        @type label: str
        @return: feedback ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section(label,"feedback")

    @rebuild_trunk
    def remove_feedback(self, label):
        """
        Remove a given feedback ParameterSet
        
        [FUTURE]
        
        @param label: name of feedback ParameterSet
        @type label: str
        """
        feedback = self.get_feedback(label)
        self.sections['feedback'].remove(feedback)
        self._build_trunk() # needs to be called again to reset
    
    @rebuild_trunk
    def add_mpi(self, ps=None, **kwargs):
        """
        Add a new MPI ParameterSet
        
        **Example usage:**
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.add_mpi(context='mpi', np=6, label='mympi')
        
        [FUTURE]
        
        @param ps: fitting ParameterSet
        @type ps:  None, or ParameterSet
        @param label: name of the MPI options (will override label in ps)
        @type label: str
        """
        if ps is None:
            context = kwargs.pop('context', 'mpi')
            ps = parameters.ParameterSet(context=context)

        for k,v in kwargs.items():
            ps.set_value(k,v)
            
        self._add_to_section('mpi',ps)
            
    def get_mpi(self, label=None):
        """
        Get an MPI ParameterSet by name
        
        [FUTURE]
        
        @param label: name of ParameterSet
        @type label: str
        @return: mpi ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section(label, "mpi")

    @rebuild_trunk
    def remove_mpi(self, label):
        """
        Remove a given MPI ParameterSet
        
        [FUTURE]
        
        @param label: name of mpi ParameterSet
        @type label: str
        """
        mpi = self.get_fitting(label)
        self.sections['mpi'].remove(mpi)
        self._build_trunk() # needs to be called again to reset
    
    ## internal methods
    
    def _loop_through_container(self, container=None, label=None, ref=None,
                                do_sectionlevel=True, do_pslevel=True):
        """
        Loop through containere.
        
        Loop through the current container to compile all items for the trunk
        this will then be called recursively if it hits another
        Container/PS/BodyBag
        
        [FUTURE]
        """
        return_items = []

        # add the container's sections
        if do_sectionlevel:
            ris= self._get_info_from_item(self.sections,section=None,container=container,label=-1)
            return_items += ris
            
        for section_name,section in self.sections.items():
            
            if do_pslevel:
                for item in section:
                    
                    if item is None:
                        continue
                    
                    # If the system is not a BodyBag, we can't get away with
                    # calling get_value (BodyBag implements any method!)
                    # OLD IMPLEMENTATION: remove hasattr(item, 'get_value')
                    if label is None and hasattr(item, 'get_value'):
                        if 'label' in item.keys():
                            this_label = item.get_value('label')
                        else:
                            this_label = item.get_value('ref')
                    else:
                        this_label = label
                        
                    ris = self._get_info_from_item(item,section=section_name,container=container,label=this_label)
                    for ri in ris:
                        
                        return_items += [ri]
                        
                        if ri['kind']=='Container':
                            return_items += ri['item']._loop_through_container(container=self.__class__.__name__, label=ri['label'], ref=ref)

                        # OLD IMPLEMENTATION: elif ri['class_name'] == 'BodyBag':
                        elif ri['kind'] == 'Body':
                            return_items += self._loop_through_system(item, section_name=section_name)
                       
                        elif ri['kind']=='ParameterSet': # these should be coming from the sections
                            return_items += self._loop_through_ps(item, section_name=section_name, container=container, label=ri['label'], ref=ri['ref'])

            if do_sectionlevel and section_name not in ['system', 'dataset']: # we'll need to fake these later since they're not actually stored as a list of PSs
                ris = self._get_info_from_item({item.get_value('ref') if (hasattr(item, 'keys') and 'ref' in item.keys()) else item.get_value('label') if hasattr(item, 'get_value') else item.get_label():item for item in section},section=section_name,container=container,label=-1)
                return_items += ris
                
        return return_items
        
    def _loop_through_ps(self, ps, section_name, container, label, ref):
        """
        called when _loop_through_container hits a PS
        
        [FUTURE]
        """
        return_items = []
        
        for qualifier in ps:
            item = ps.get_parameter(qualifier)
            if ref is None:
                ref=ps.get_value('ref') if 'ref' in ps.keys() else None
            ris = self._get_info_from_item(item, section=section_name, context=ps.get_context(), container=container, label=label, ref=ref)
            for ri in ris:
                if ri['qualifier'] not in ['ref','label']:
                    return_items += [ri]
            
        return return_items
        
    def _loop_through_system(self, system, section_name):
        """
        called when _loop_through_container hits the system
        
        [FUTURE]
        """
        return_items = []
        ri = None
        
        # we need to hide synthetics that have no corresponding obs for the user:
        legal_syn_twigs = []
        
        for path, item in system.walk_all(path_as_string=False):
            
            # Make sure all constraints are correctly set
            if isinstance(item, parameters.ParameterSet):
                item.run_constraints()
                
            
            # make sure to catch the obs and pbdep
            did_catch = isinstance(item, str)
            did_catch = did_catch and item[-3:] in ['obs', 'dep','syn']
            did_catch = did_catch and (not isinstance(path[-2], str) if len(path)>2 else True)
            if did_catch:
                # we completely ignore existing syns, we build them on the same
                # level as obs are available
                #~ if item[-3:] == 'syn':
                    #~ continue
                
                for itype in path[-2]:
                    for isubtype in path[-2][itype].values():
                        if item==itype:
                            ris = self._get_info_from_item(isubtype, path=path, section=section_name)
                            for ri in ris:
                                #~ if item in ['lcsyn']:
                                if item[-3:]=='syn':
                                    # then this is the true syn, which we need to keep available
                                    # for saving and loading, but will hide from the user in twig access
                                    ri['hidden'] = True
                                
                                return_items += [ri]
                                
                        # we want to see if there any syns associated with the
                        # obs. If there are, we will add the synthetics to the
                        # trunk, but make sure they are arrayified and scaled
                        # properly to the obs. Note that the their might be a
                        # breach of consistency between the underlying system
                        # and the Bundle: the BodyBag keeps separate syns in
                        # each body and only combines them when necessary. The
                        # syns for the Bundle, on the other hand, are at the
                        # same level as the obs. This should be more intuitive
                        # to users but is less flexible in the end.
                        category = item[:-3]
                        if itype[-3:] == 'obs':
                            bodies = [self.get_system()] + [thing for thing in path if isinstance(thing, universe.Body)]
                            syn = bodies[-1].get_synthetic(category=category,
                                                           ref=isubtype['ref'])
                            if syn is not None: 
                                
                                # 1. Arrayify
                                syn = syn.asarray()
                                ris = self._get_info_from_item(syn, path=path, section=section_name)
                                return_items += ris
                                
                                
                                subpath = list(path[:-1]) + [syn['ref']]
                                subpath[-3] = syn.get_context()
                                
                                # 2. Convert columns to user supplied units
                                obs = bodies[-1].get_obs(category=category,
                                                         ref=isubtype['ref'])
                                if obs is not None:
                                    user_columns = obs['user_columns']
                                    user_units = obs['user_units']
                                    
                                    # Scale syns to obs
                                    if category == 'lc':
                                        syn['flux'] = obs['scale']*syn['flux'] + obs['offset']
                                    elif category == 'rv':
                                        syn['rv'] = syn['rv'] + obs['vgamma_offset']
                                    elif category == 'sp':
                                        syn['flux'] = obs['scale']*syn['flux']/syn['continuum'] + obs['offset']
                                    elif category == 'etv':
                                        syn['etv'] = syn['etv'] + obs['offset']
                                    else:
                                        logger.critical('Auto-scaling in Bundle of {} synthetics is not implemented yet'.format(category))
                                    
                                    # Remove times and insert phases if phases
                                    # were given before
                                    if user_columns is not None and 'phase' in user_columns:
                                        out = system.get_period()
                                        syn['phase'] = (syn['time'] - out[1]) / out[0]
                                        syn['time'] = []
                                        syn['columns'][syn['columns'].index('time')] = 'phase'
                                
                                # add phase to synthetic
                                #if 'time' in syn and len(syn['time']) and not 'phase' in syn:
                                    #period = self.get_value('period@orbit')
                                    #par = parameters.Parameter('phase', value=np.mod(syn['time'], period), unit='cy', context=syn.get_context())
                                    #syn.add(par)
                                
                                # Fix path
                                subpath = [ii for ii in path]
                                subpath[-1] = syn.get_context()
                                subpath.append(syn['ref'])
                                subpath[-3] = OrderedDict()
                                subpath[-3][syn.get_context()] = OrderedDict()
                                subpath[-3][syn.get_context()][syn['ref']] = syn
                                for par in syn:                                                                                                                                       
                                    mypar = syn.get_parameter(par)
                                    ris = self._get_info_from_item(mypar, path=subpath+[mypar], section=section_name)
                                    return_items += ris
                                    #for iki in ris:
                                    #    if 'lcobs' in iki['twig_full']:
                                    #        print mypar.get_context()
                                    #        print 'bla'
                                
                                    
                                                                                    
                    # but also add the collection of dep/obs
                    if item==itype:
                        ris = self._get_info_from_item(path[-2][itype], path=list(path)+[item], section=section_name)
                        #~ if ri['twig_full'] not in [r['twig_full'] for r in return_items]:
                        return_items += ris
            
            elif isinstance(item, OrderedDict):
                continue
                
            else:
                try:
                    if item['label'] == 'f2':
                        print(self._get_info_from_item(item, path=path, section=section_name))
                except:
                    pass
            
                #if 'lcobs' in [p for p in path if isinstance(p,str) and 'lcobs' in p] and isinstance(item, parameters.Parameter):
                    #print path
                    #raise SystemExit
                
                ris = self._get_info_from_item(item, path=path, section=section_name)
                #~ print path, ri['twig_full'] if ri is not None else None #, ri['context']!='syn' if ri is not None else None, (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) if ri is not None else None
                #~ print path[-1:], ri['twig_full'] if ri is not None else None, ri['context']!='syn' if ri is not None else None, (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) if ri is not None else None
                if len(ris)==0:
                    continue
                
                ri = ris[0] # for these checks, all items will be identical and may only have different kinds
                # do not add any syn parameter here
                if ri['context'] is not None and 'syn' in ri['context']:
                    continue
                # ignore parameters that are synthetics and make sure this is not a duplicate
                if (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) \
                        and ri['twig_full'] not in [r['twig_full'] for r in return_items]:
                    return_items += ris
        
        system.preprocess()
        
        return return_items
        
    def _get_info_from_item(self, item, path=None, section=None, container=None, context=None, label=None, ref=None):
        """
        general function to pull all information we need from any item we may come across
    
        providing other information (section, container, context, label) may be necessary if we don't know where we are
        
        [FUTURE]
        """
        info_items = [] # we might decide to return more than one item
        
        container = self.__class__.__name__ if container is None else container
        class_name = item.__class__.__name__
        body = False
        hidden = False
        if isinstance(item, parameters.ParameterSet):
            kind = 'ParameterSet'
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            if len(labels):
                label = labels[-1]
            elif hasattr(self, 'get_system') and label is None:
                label = self.get_system().get_label()
            else:
                label = label
            # unless we're in one of the following sections
            # in which case we identify by ref and have no label
            context = item.get_context()
            if context[-3:] in ['obs','dep','syn'] and context.split(':')[0] != 'plotting':
                section = 'dataset'
            if section in ['axes','plot','figure','fitting','mpi'] or context in ['compute:legacy']:
                label = None

            # For some contexts, we need to add labels for disambiguation
            if context in ['puls', 'compute:legacy'] or section in ['fitting','mpi']:
                ref = item.get_value('label') if 'label' in item else ref
            ref = item.get_value('ref') if 'ref' in item else ref
            unique_label = None
            qualifier = None
            
        elif isinstance(item, parameters.Parameter):
            kind = 'Parameter'
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            ref = ref
            if len(labels):
                label = labels[-1]
            elif hasattr(self, 'get_system') and label is None:
                label = self.get_system().get_label()
            else:
                label = label
            if path:
                #then coming from the system and we need to build the context from the path
                #~ context = path[-2] if isinstance(path[-2],str) else path[-2].get_context()
                context = item.get_context()
                if isinstance(context, list): #because sometimes its returning a list and sometimes a string
                    context = context[0]
            else:
                #then we're coming from a section and already know the context
                context = context
                
            if context[-3:] in ['obs','dep','syn'] and context.split(':')[0] != 'plotting':
                section = 'dataset'

            # unless we're in one of the following sections
            # in which case we identify by ref and have no label
            if section in ['axes','plot','figure','fitting','mpi']:
                label = None
                
            if path:
                if context[-3:] in ['obs','dep','syn']:
                    # then we need to get the ref of the obs or dep, which is placed differently in the path
                    # due to bad design by Pieter, this needs a hack to make sure we get the right label
                    ref = path[-2]
                # sometimes we need extra info
                elif context in ['puls']:
                    ref = path[-2]['label']
            unique_label = item.get_unique_label()
            qualifier = item.get_qualifier()
            hidden = item.get_hidden()
            
        elif isinstance(item, universe.Body):
            kind = 'Body'
            label = item.get_label()
            context = None
            ref = None
            unique_label = None
            qualifier = None
            body = True
        elif isinstance(item, Container):
            kind = 'Container'
            label = None
            context = None
            ref = item.get_label() if hasattr(item, 'get_label') else None
            unique_label = None
            qualifier = None
        elif isinstance(item, OrderedDict) or isinstance(item, dict):
            kind = 'OrderedDict'
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            if len(labels):
                label = labels[-1]
            elif hasattr(self, 'get_system') and label is None:
                label = self.get_system().get_label()
            elif label == -1:
                label = None
            else:
                label = label
            context = path[-1] if path else None
            ref = None
            unique_label = None
            qualifier = None
            
            # override context for datasets
            if context is not None and context[-3:] in ['obs','dep','syn'] and context.split(':')[0] != 'plotting':
                section = 'dataset'
            
        elif isinstance(item, feedback.Feedback):
            kind = 'Feedback'
            label = item.get_label()
            context = None
            ref = None
            unique_label = None
            qualifier = None
            body = False
        else:
            return []
            #~ raise ValueError("building trunk failed when trying to parse {}".format(kind))
            
        # now let's do specific overrides
        if context == 'orbit' and self._get_object(label).get_parent():
            # we want to hide the fact to the user that these exist at the component level
            # and instead fake its label to be that of its parent BodyBag (if it
            # has any, an overcontact might not!)
            #~ label = self.get_parent(label).get_label()
            label = self._get_object(label).get_parent().get_label()
            ref = None
        
        if context == section: 
            context_twig = None
        else:
            context_twig = context
            
            
        hidden = hidden or qualifier in ['c1label', 'c2label']
        #~ hidden = qualifier in ['ref','label', 'c1label', 'c2label']
        #hidden = False
        
        if kind=='Parameter' and False:  ### DEFER this functionality 
            # defer until we support fitting and discuss desired behavior
            # in theory, removing the and False should create these additional twigs
            # and their set/get behavior is defined in Container.get and set
            if hasattr(item, 'get_value'):
                twig = self._make_twig(['value',qualifier,ref,context_twig,label,section])
                twig_full = self._make_twig(['value',qualifier,ref,context_twig,label,section,container])
                info_items.append(dict(qualifier=qualifier, label=label,
                    container=container, section=section, kind='Parameter:value', class_name=class_name,
                    context=context, ref=ref, unique_label=unique_label, 
                    twig=twig, twig_full=twig_full, path=path, 
                    #~ twig_reverse=twig_reverse, twig_full_reverse=twig_full_reverse,
                    item=item, body=body,
                    hidden=hidden))
                
            if hasattr(item, 'adjust'):
                twig = self._make_twig(['adjust',qualifier,ref,context_twig,label,section])
                twig_full = self._make_twig(['adjust',qualifier,ref,context_twig,label,section,container])
                info_items.append(dict(qualifier=qualifier, label=label,
                    container=container, section=section, kind='Parameter:adjust', class_name=class_name,
                    context=context, ref=ref, unique_label=unique_label, 
                    twig=twig, twig_full=twig_full, path=path, 
                    #~ twig_reverse=twig_reverse, twig_full_reverse=twig_full_reverse,
                    item=item, body=body,
                    hidden=hidden))
                            
            if hasattr(item, 'get_prior') and item.has_prior():
                twig = self._make_twig(['prior',qualifier,ref,context_twig,label,section])
                twig_full = self._make_twig(['prior',qualifier,ref,context_twig,label,section,container])
                info_items.append(dict(qualifier=qualifier, label=label,
                    container=container, section=section, kind='Parameter:prior', class_name=class_name,
                    context=context, ref=ref, unique_label=unique_label, 
                    twig=twig, twig_full=twig_full, path=path, 
                    #twig_reverse=twig_reverse, twig_full_reverse=twig_full_reverse,
                    item=item, body=body,
                    hidden=hidden))
                            
            if hasattr(item, 'get_posterior') and item.has_posterior():
                twig = self._make_twig(['posterior',qualifier,ref,context_twig,label,section])
                twig_full = self._make_twig(['posterior',qualifier,ref,context_twig,label,section,container])
                info_items.append(dict(qualifier=qualifier, label=label,
                    container=container, section=section, kind='Parameter:posterior', class_name=class_name,
                    context=context, ref=ref, unique_label=unique_label, 
                    twig=twig, twig_full=twig_full, path=path, 
                    #twig_reverse=twig_reverse, twig_full_reverse=twig_full_reverse,
                    item=item, body=body,
                    hidden=hidden))
                            
            
        # twig = <qualifier>@<ref>@<context>@<label>@<section>@<container>
        twig = self._make_twig([qualifier,ref,context_twig,label,section])
        #~ twig_reverse = self._make_twig([qualifier,ref,context,label,section,container], invert=True)
        twig_full = self._make_twig([qualifier,ref,context_twig,label,section,container])
        #~ twig_full_reverse = self._make_twig([qualifier,ref,context,label,section,container], invert=True)
        
        info_items.append(dict(qualifier=qualifier, label=label,
            container=container, section=section, kind=kind, class_name=class_name,
            context=context, ref=ref, unique_label=unique_label, 
            twig=twig, twig_full=twig_full, path=path, 
            #~ twig_reverse=twig_reverse, twig_full_reverse=twig_full_reverse,
            item=item, body=body,
            hidden=hidden))
        
        return info_items
            
    def _build_trunk(self):
        """
        this function should be called (probably through the @rebuild_trunk decorator)
        whenever anything is changed (besides changing values, etc)
        
        this sets self.trunk which is then used whenever the container is searched for an item
        
        [FUTURE]
        """
        self.trunk = self._loop_through_container()
        
        # manually fake the dataset section dictionary
        tis = self._get_by_search(kind='ParameterSet', section='dataset', all=True, ignore_errors=True, return_trunk_item=True)
        if len(tis):
            ri = self._get_info_from_item({'@'.join(ti['twig'].split('@')[:-1]): ti['item'] for ti in tis})[0]
            ri['twig'] = 'dataset'
            ri['twig_full'] = 'dataset@Bundle'
            ri['section'] = 'dataset'
            self.trunk.append(ri)
        
        # manually fake the system section dictionary
        tis = self._get_by_search(kind='ParameterSet', section='system', all=True, ignore_errors=True, return_trunk_item=True)
        if len(tis):
            ri = self._get_info_from_item({'@'.join(ti['twig'].split('@')[:-1]): ti['item'] for ti in tis})[0]
            ri['twig'] = 'system'
            ri['twig_full'] = 'system@Bundle'
            ri['section'] = 'system'
            self.trunk.append(ri)
            
        # do intelligent choices for dataref, objref
        objrefs = self._get_by_search(section='system', body=True, return_key='label', all=True, ignore_errors=True)
        datarefs = ['{}@{}'.format(ti['ref'], ti['label']) for ti in self._get_by_search(section='dataset', context='*obs', kind='ParameterSet', return_trunk_item=True, all=True, ignore_errors=True)]
        
        for param in self._get_by_search('objref@', kind='Parameter', all=True, ignore_errors=True):
            param.cast_type = 'choose'
            param.choices = objrefs
            
        for param in self._get_by_search('dataref@', kind='Parameter', all=True, ignore_errors=True):
            param.cast_type = 'choose'
            param.choices = datarefs
        
        
    def _purge_trunk(self):
        """
        simply clears trunk - this will probably only be called before pickling to 
        minimize duplication and reduce filesize
        
        [FUTURE]
        """
        self.trunk = []
        
    def _make_twig(self, path, invert=False):
        """
        compile a twig for an ordered list of strings that makeup the path
        this list should be [qualifier,label,context,component,section,container]        
        
        [FUTURE]
        """
        if invert:
            path.reverse()
        while None in path:
            path.remove(None)
        
        return '{}'.format('/' if invert else '@').join(path)
        
    def _search_twigs(self, twiglet, trunk=None, **kwargs):
        """
        return a list of twigs where twiglet is a substring
        
        [FUTURE]
        """
        trunk = self._filter_twigs_by_kwargs(trunk, **kwargs)
            
        twigs = [t['twig_full'] for t in trunk]
        return [twig for twig in twigs if twiglet in twig]
        
    def _filter_twigs_by_kwargs(self, trunk=None, **kwargs):
        """
        returns a list of twigs after filtering the trunk by any other
        information stored in the dictionaries
        
        [FUTURE]
        """
        trunk = self.trunk if trunk is None else trunk
        
        kwargs.setdefault('hidden', False)
        for key in kwargs.keys():
            if len(trunk) and key in trunk[0].keys() and kwargs[key] is not None:
                trunk = [ti for ti in trunk if ti[key] is not None and 
                    (ti[key]==kwargs[key] or 
                    (isinstance(kwargs[key],list) and ti[key] in kwargs[key]) or
                    (isinstance(kwargs[key],str) and fnmatch(ti[key],kwargs[key])))]
                
        return trunk
        
        
    def _match_twigs(self, twiglet, trunk=None, ignore_exact=False, **kwargs):
        """
        return a list of twigs that match the input
        the first item is required to be first, but any other item
        must simply occur left to right
        
        [FUTURE]
        """
        trunk = self._filter_twigs_by_kwargs(trunk, **kwargs)
        
        twig_split = twiglet.split('@')
        
        matching_twigs_orig = [t['twig_full'] for t in trunk]
        matching_twigs = [t['twig_full'].split('@') for t in trunk]
        matching_indices = list(range(len(matching_twigs)))
        matching_comp_indices = [] # twiglet furthest to right is complete match
        
        # We attempt to match a twig twice; once with colons describing
        # subcontext, once without
        
        for attempt in (0,1):
            
            # second attempt is without colons denoting subcontexts
            if attempt == 1:
                matching_twigs = [[st.split(':')[0] for st in t['twig_full'].split('@')] for t in trunk]
                matching_indices = list(range(len(matching_twigs)))
                matching_comp_indices = [] # twiglet furthest to right is complete match
        
            for tsp_i,tsp in enumerate(twig_split):
                remove = []
                for mtwig_i, mtwig in enumerate(matching_twigs):
                    # mtwig is a split list of a remaining full twig
                    ind = None
                    if tsp_i==len(twig_split)-1:
                        # let the item furthest to the right be incomplete (e.g te matches teff)
                        mtwiglets = [mtwiglet[:len(tsp)] for mtwiglet in mtwig] if tsp_i!=0 else [mtwig[0][:len(tsp)]]
                        if tsp in mtwiglets:
                            ind = mtwiglets.index(tsp)
                            if tsp==mtwig[ind] and not ignore_exact: 
                                # then twiglet furthest to right is perfect match
                                # we'll remember this index and prefer it if its the only match
                                matching_comp_indices.append(mtwig_i)

                            mtwig = mtwig[ind+1:]
                    
                    else:
                        if (tsp_i != 0 and tsp in mtwig) or mtwig[0]==tsp:
                        #~ if (tsp_i != 0 and [fnmatch(mtwig_spi,tsp) for mtwig_spi in mtwig]) or fnmatch(mtwig[0],tsp):
                            # then find where we are, and then only keep stuff to the right
                            ind = mtwig.index(tsp)
                            mtwig = mtwig[ind+1:]
                    
                    if ind is None:
                        # then remove from list of matching twigs
                        remove.append(mtwig_i)
                
                for remove_i in sorted(remove,reverse=True):
                    matching_twigs.pop(remove_i)
                    matching_indices.pop(remove_i)
            
            indices = matching_comp_indices if len(matching_comp_indices)==1 and matching_comp_indices[0] in matching_indices else matching_indices
            ret_value = [matching_twigs_orig[i] for i in indices] 
            
            # If we found a match, no need to search through subcontexts
            if len(ret_value):
                break
                    
        return ret_value
        
    def _get_by_search(self, twig=None, all=False, ignore_errors=False,
                       return_trunk_item=False, return_key='item',
                       use_search=False, **kwargs):
        """
        this function searches the cached trunk
        kwargs will filter any of the keys stored in trunk (section, kind, container, etc)

        [FUTURE]
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @param all: whether to return a list or a single item
        @type all: bool
        @param ignore_errors: whether to ignore length of results errors and always return a list
        @type ignore_errors: bool
        @param return_trunk_item: whether to return the full trunk listing instead of just the param
        @type return_trunk_item: bool
        @param use_search: whether to use substring search instead of match
        @type use_search: bool
        """
        # can take kwargs for searching by any other key stored in the trunk dictionary       
        kwargs.setdefault('trunk', None)
        kwargs.setdefault('hidden', False)
        trunk = kwargs.pop('trunk')
        trunk = self._filter_twigs_by_kwargs(trunk, **kwargs)
        
        if twig is not None:
            if use_search:
                matched_twigs = self._search_twigs(twig, **kwargs)
            else:
                matched_twigs = self._match_twigs(twig, **kwargs)
        else:
            matched_twigs = [ti['twig_full'] for ti in trunk]
        
        if len(matched_twigs) == 0:
            if ignore_errors:
                if all:
                    return []
                else:
                    return None
            raise KeyError("no match found matching the criteria")
        elif all == False and ignore_errors == False and len(matched_twigs) > 1:
            
            # perhaps check here if the parameter name matches exactly:
            #par_name = twig.split('@')[0]
            #exact_matches = [tw for tw in matched_twigs if tw.split('@')[0] == par_name]
            #if len(exact_matches) == 1:
            #    items = [ti if return_trunk_item else ti[return_key] \
            #                 for ti in trunk if ti['twig_full'] in exact_matches]
            #    return items[0]
            
            results = ', '.join(matched_twigs)
            raise KeyError(("more than one result was found matching the "
                            "criteria for twig: {}.  Matching twigs: {}."
                            " If you want to set all twigs simultaneously, use "
                            "Bundle.set_value_all('{}', value)").format(twig, results, twig))
        else:
            items = [ti if return_trunk_item else ti[return_key] \
                            for ti in trunk if ti['twig_full'] in matched_twigs]
            if all:
                return items
            else:
                return items[0]
                
    def _get_by_section(self, label=None, section=None, kind="ParameterSet", all=False, ignore_errors=False, return_trunk_item=False):
        """
        shortcut to _get_by_search which automatically builds a searching twig from the label
        and has defaults for kind (ParameterSet)        
        
        [FUTURE]
        """
        twig = "{}@{}".format(label,section) if label is not None and section is not None else None
        #~ twig = section if label is None else label
        return self._get_by_search(twig,section=section,kind=kind,
                        all=all, ignore_errors=ignore_errors, 
                        return_trunk_item=return_trunk_item)
                        
    def _get_dict_of_section(self, section, kind='ParameterSet'):
        """
        shortcut to _get_by_search that takes the name of a section and will always
        return a dictionary with label as the keys and item as the values
        
        [FUTURE]
        """
        all_ti = self._get_by_search(twig=None, section=section, kind=kind,
                        all=True, ignore_errors=True,
                        return_trunk_item=True)
                 
        return {ti['label'] if ti['label'] is not None else ti['ref']:ti['item'] for ti in all_ti} if all_ti is not None else {}
    
    def _to_dict(self, return_str=False, debug=False):
        
        # We're changing stuff here, so we need to make a copy first
        this_bundle = self.copy()
        #~ this_bundle.clear_syn()
        
        this_container = this_bundle.__class__.__name__

        dump_dict = OrderedDict()
        
        dump_dict['PHOEBE Version'] = __version__
        
        if hasattr(self, 'get_system'):
            dump_dict['Hierarchy'] = _dumps_system_structure(this_bundle.get_system())

        dump_dict['ParameterSets'] = {}

        for ti in this_bundle._get_by_search(kind='ParameterSet',container=this_container,all=True,hidden=None,return_trunk_item=True):
            item = ti['item']
            
            if ti['context'] not in ['orbit','component','mesh:marching']:
                info = {}
                
                if ti['context'].split(':')[0]!='plotting' and (ti['context'][-3:]=='syn' or (ti['context'][-3:]=='obs' and item['filename'])):
                    # then unload the data first
                    item.unload()
                    
                if not (ti['context'].split(':')[0]!='plotting' and ti['context'][-3:]=='syn' and not ti['hidden']):
                    info['context'] = ti['context']
                    #~ info['class_name'] = ti['class_name']
                    #~ info['parent'] = self._make_twig([qualifier,ref,context,label,section,container])
                    dump_dict['ParameterSets'][ti['twig']] = info
                else:
                    if debug:
                        print ti['twig']
        
        dump_dict['Parameters'] = {}
        for ti in this_bundle._get_by_search(kind='Parameter',container=this_container,all=True,hidden=None,return_trunk_item=True):
            item = ti['item']
            info = {}
            
            # determine if this parameter is a default item of its PS
            # or if it will need to be created on the fly during load
            ps = parameters.ParameterSet(context=ti['context'])
            if ti['qualifier'] not in ps:
                # then we need to keep more information
                info['cast_type'] = str(item.cast_type)
                if '<type' in info['cast_type']:
                    info['cast_type'] = info['cast_type'].split("'")[1]
                
                # TODO: add more info here if needed (eg if cast_type=='choose')
            
            info['value'] = item.get_value()#to_str()
            
            if isinstance(info['value'],np.ndarray):
                info['value'] = list(info['value'])
            
            #~ if hasattr(item, 'get_unit') and item.has_unit():
                #~ info['_unit (read-only)'] = item.get_unit()
            
            if hasattr(item, 'adjust') and item.adjust:
                info['adjust'] = item.adjust
                
            #~ if item.has_prior():
                #~ info['prior'] = 'has prior, but export not yet supported'
            
            if not (ti['context'].split(':')[0]!='plotting' and ti['context'][-3:]=='syn' and not ti['hidden']):
                # the real syns are marked as hidden, whereas the fake
                # ones forced to match the obs are not hidden
                dump_dict['Parameters'][ti['twig']] = info
        
        if return_str:
            return json.dumps(dump_dict, sort_keys=True, separators=(',', ': '))
        else:
            return dump_dict

    def _save_json(self, filename, debug=False):
        """
        TESTING
        [FUTURE]
        """

        dump_dict = self._to_dict(debug=debug)
        f = open(filename, 'w')
        f.write(json.dumps(dump_dict, sort_keys=True, indent=4, separators=(',', ': ')))
        f.close()



    def _load_json(self, filename, debug=False):
        """
        TESTING
        
        [FUTURE]
        """
        f = open(filename, 'r')
        load_dict = json.load(f, object_pairs_hook=OrderedDict)
        # we use OrderedDict because order does matter for reattaching datasets
        # luckily alphabetical works perfectly dep, obs, then syn.
        f.close()
        
        self._from_dict(load_dict, debug=debug)

    @rebuild_trunk
    def _from_dict(self, load_dict, debug=False):
        if isinstance(load_dict, str):
            load_dict = json.loads(load_dict, object_pairs_hook=OrderedDict)
        if hasattr(self, 'get_system'):
            self.set_system(_loads_system_structure(load_dict['Hierarchy']))
            #print self.get_system().list(summary='full')
            
        for twig,info in load_dict['ParameterSets'].items():
            #~ print "*** _load_json", twig, info['context']#, info['section']
            label = str(twig.split('@')[0])
            where = twig.split('@').index(info['context'])
            if info['context'] not in self.sections.keys(): where+=1
            parent_twig = '@'.join(twig.split('@')[where:])
            parent_twig = parent_twig.replace('dataset','system')
            
            # lcdeps etc are ParameterSets, lcobs/lcsyn are DataSets (which are
            # a sublcass of ParameterSets but with extra functionality)
            context = str(info['context'])
            if context.split(':')[0]!='plotting' and context[-3:] in ['obs', 'syn']:
                ps = getattr(datasets, config.dataset_class[context[:-3]])(context=context)
            else:
                ps = parameters.ParameterSet(context)
                
            if 'label' in ps.keys():
                ps.set_value('label',label)
            elif 'ref' in ps.keys():
                ps.set_value('ref', label)
            if debug:
                print("self.attach_ps('{}', {}(context='{}', {}='{}'))".format(parent_twig, 'PS' if info['context'][-3:] not in ['obs','syn'] else 'DataSet', info['context'], 'ref' if info['context'][-3:] in ['obs','dep','syn'] or info['context'].split(':')[0]=='plotting' else 'label', label))
            
            self.attach_ps(ps, parent_twig)
            
        self._build_trunk()
        
        for twig,info in load_dict['Parameters'].items():
            # handle if the parameter did not exist in the PS by default
            if debug:
                print("parameter: {}".format(twig))
            
            if 'cast_type' in info:
                # then this is a new parameter that needs to be added to the PS
                if debug:
                    print("\tcreating parameter {}".format(twig))
                qualifier = twig.split('@')[0]
                ps = self._get_by_search('@'.join(twig.split('@')[1:]), hidden=None)
                try:
                    cast_type = getattr(__builtin__,info['cast_type'])
                except AttributeError:
                    cast_type = info['cast_type']
                param = parameters.Parameter(qualifier, context=ps.context, cast_type=cast_type)
                ps.add(param)
                
                # this parameter isn't in the trunk
                self._build_trunk()

            item = self._get_by_search(twig, hidden=None)
            if 'value' in info:
                if debug:
                    print("\tself.set_value('{}', '{}')".format(twig, str(info['value'])))
                item.set_value(info['value'])
            if 'adjust' in info:
                if debug:
                    print("\tself.set_adjust('{}', '{}')".format(twig, str(info['adjust'])))
                item.set_adjust(info['adjust'])
            #~ if 'prior' in info:
                #~ item.set_prior(info['prior'])
            
        
    ## generic functions to get non-system parametersets
    def _return_from_dict(self, dictionary, all=False, ignore_errors=False):
        """
        this functin takes a dictionary of results from a searching function
        ie. _get_from_section and returns either a single item or the dictionary
        
        [FUTURE]
        
        @param dictionary: the dictionary of results
        @type dictionary: dict or OrderedDict
        @param all: whether to return a single item or all in a dictionary
        @type all: bool
        @param ignore_errors: if all==False, ignore errors if 0 or > 1 result
        @type ignore_errors: bool
        """
        if all:
            return dictionary
        else:
            if len(dictionary) == 0:
                if ignore_errors:
                    return None
                raise ValueError("no parameter found matching the criteria")
                
            elif len(dictionary) > 1:
                if ignore_errors:
                    return dictionary.values()[0]
                # build a string representation of the results
                #~ results = ", ".join(["{} ({})".format(twig, dictionary[twig].get_value()) for twig in dictionary])
                # but this function is called for more than just get_value, so for now we won't show this info
                results = ", ".join(["{}".format(twig) for twig in dictionary])
                raise ValueError("more than one parameter was found matching the criteria: {}".format(results))
            
            else:
                return dictionary.values()[0] 
    
    @rebuild_trunk
    def _add_to_section(self, section, ps):
        """
        add a new parameterset to section - the label of the parameterset
        will be used as the key to retrieve it using _get_from_section
        
        [FUTURE]
        
        @param section: name of the section (key of self.sections)
        @type section: str
        @param ps: the new parameterset with label set
        @type ps: ParameterSet
        """
        if section not in self.sections.keys():
            self.sections[section] = []
        self.sections[section].append(ps)
    
def _dumps_system_structure(current_item):
    """
    dumps the system hierarchy into a dictionary that is json compatible
    
    [FUTURE]
    """
    struc_this_level = {}

    struc_this_level['label'] = current_item.get_label()
    struc_this_level['kind'] = current_item.__class__.__name__

    # now let's find if this current item has any children and if so,
    # create an entry for children and make a recursive call to this func
    children = current_item.bodies if hasattr(current_item, 'bodies') else []
    # if there are children, then we need to create the entry and 
    # intialize an empty list
    if len(children):
        struc_this_level['children'] = []
    # now let's iterate through each child, recursively create its own
    # dictionary and then attach
    for i,child in enumerate(children):
        struc_child = _dumps_system_structure(child)
        struc_this_level['children'].append(struc_child)

    return struc_this_level

def _loads_system_structure(struc, c1label=None, c2label=None):
    """
    loads the system hierarchy from a json compatible dictionary
    
    [FUTURE]
    """

    # if we're dealing with a Bag of some sort, then we need to create
    # the item while also building and attaching its children
    if 'children' in struc.keys(): # BodyBag, etc
        list_of_bodies = [_loads_system_structure(child_struc, c1label=struc['children'][0]['label'], c2label=struc['children'][1]['label']) for child_struc in struc['children']]
        this_item = getattr(universe, struc['kind'])(list_of_bodies, label=struc['label'])
    
    # if we're a component, then we can create the item and attach orbits from
    # the passed labels for c1label and c2label
    else: # BinaryRocheStar, etc
        this_item = getattr(universe, struc['kind'])(parameters.ParameterSet(context='component', label=struc['label']), orbit=parameters.ParameterSet(context='orbit', label=struc['label'], c1label=c1label, c2label=c2label))

    # and finally return the item, which should be the system for any
    # non-recursive calls
    return this_item


def take_orbit_from(donor, receiver, do_copy=False, only_lowest_level=False):
    """
    Put the receiver in the same orbit as the donor.
    """
    the_orbits, the_components = donor.get_orbits()
    
    if only_lowest_level:
        the_orbits = the_orbits[-1:]
        the_components = the_components[-1:]
    
    this_parent = donor.get_parent()
    
    for i, (orbit, comp) in enumerate(zip(the_orbits[::-1], the_components[::-1])):
        # Copy the orbit to have a new ParameterSet, but let the Parameters
        # (except that one label thing) point to the original set
        this_orbit = orbit.copy()
        this_label = 'c{}label'.format(comp+1)
        if not do_copy:
            for par in this_orbit:
                if par in [this_label, 'label']:
                    continue
                this_orbit.point_to(par, orbit.get_parameter(par))
        
        mylabel = receiver.get_label()+i*'_up'
        this_orbit[this_label] = mylabel
        this_orbit['label'] = this_parent.get_label()
        receiver = universe.BodyBag([receiver], orbit=this_orbit, label=mylabel)
        
        receiver.set_parent(this_parent)
        this_parent = this_parent.get_parent()        
    
    return receiver


def compute_pot_from(mybundle, radius, component=0, relative=True,
                     adopt=True):
    """
    Convert (relative) polar radius to potential value.
    
    The default settings require you to give the radius in relative units
    (relative to the semi-major axis). If you want to give the radius in absolute
    units (i.e. :math:`R_\odot`), then set :envvar:`relative=False`::
    
        >>> mybundle = phoebe.Bundle()
        >>> print(mybundle['pot@primary'], mybundle['pot@secondary'])
        (8.0, 8.0)
        >>> new_pot_primary = compute_pot_from(mybundle, 0.1)
        >>> new_pot_secondary = compute_pot_from(mybundle, 0.2, component=1)
        >>> print(new_pot_primary, mybundle['pot@primary'])
        (10.99503719020999, 10.99503719020999)
        >>> print(new_pot_secondary, mybundle['pot@secondary'])
        (5.9805806756909199, 5.98058067569092)

    The value for the potential is automatically set in the system, unless you
    specificy :envvar:`adopt=False`.
    
    :param mybundle: Bundle or BodyBag containing a Binary system
    :param radius: polar radius to compute the potential for (relative to the
     system semi-major axis if :envvar:`relative=True`, otherwise in solar units)
    :param component: primary (0) or secondary (1) component, or component's label
    :type component: int/str
    :param relative: flag to interpret radius as relative (default) or absolute
    :type relative: bool
    :param adopt: flag to automatically set potential value in the system (default)
     or not
    :type adopt: bool
    :return: new value of the potential
    :rtype: float
    """
    # For generality, we allow the binary to be given as a BodyBag or Bundle
    if not isinstance(mybundle, universe.BodyBag):
        mysystem = mybundle.get_system()
    
    # Translate the component label to an integer if necessary
    if isinstance(component, str):
        component = 0 if component == mysystem[0].params['orbit']['c1label'] else 1
    
    # Retrieve the orbit (the orbit is given in the primary and secondary, and
    # because it's the same orbit it doesn't matter which one you take)
    orbit = mysystem[component].params['orbit']
    
    # We need some parameters
    q = orbit['q']
    d = 1 - mybundle['ecc']
    syncpar = mysystem[component].params['component']['syncpar']
    
    # Make the radius units relative if necessary
    if not relative:
        radius = radius / orbit['sma']
    
    # Switch mass ratio to the secondary's frame of reference if necessary
    if component == 1:
        q = 1.0 / q
    
    # Compute the new potential value. component=1 means primary in the
    # function, in contrast to here. Since we switched the frame of reference
    # and potential values are always given in the primary frame of reference,
    # we need to take component=1 (primary)
    new_pot = roche.radius2potential(radius, q=q, d=d, F=syncpar, component=1)
    
    if component == 1:
        q, new_pot = roche.change_component(q, new_pot)
    
    # Adopt if necessary
    if adopt:
        mysystem[component].params['component']['pot'] = new_pot
        
    return new_pot


def compute_mass_from(mybundle, primary_mass=None, secondary_mass=None, q=None, sma=None,
             period=None, adopt=True):
    """
    Compute masses or dynamical parameters in the system and return derived values.
    
    You need to give at least three different parameters (i.e. not
    :envvar:`primary_mass`, :envvar:`secondary_mass` and :envvar:`q`).
    
    **Examples**
    
    If you only want to get the individual masses:
    
        >>> x = phoebe.Bundle()
        >>> phoebe.compute_mass_from(x)
        {'q': 1.0, 'primary_mass': 6.703428704219417, 'period': 1.0, 'sma': 10.0, 'secondary_mass': 6.703428704219417}
        
    If you want to change the primary mass, but want to keep the semi-major axis
    and period (in the example, mass ratio is automatically adopted in :envvar:`x`)::
        
        >>> phoebe.compute_mass_from(x, primary_mass=1.0, sma=x['sma'], period=x['period'])
        {'q': 12.406857408438833, 'primary_mass': 1.0, 'period': 1.0, 'sma': 10.0, 'secondary_mass': 12.406857408438833}
    
    If you want to set individual masses, keep the period but adapt the
    semi-major axis:
    
        >>> phoebe.compute_mass_from(x, primary_mass=1.0, secondary_mass=2.0, period=x['period'])
        {'q': 2.0, 'primary_mass': 1.0, 'period': 1.0, 'sma': 6.071063212082658, 'secondary_mass': 2.0}

    :param mybundle: Bundle or BodyBag containing a Binary system
    :param primary_mass: primary mass (:math:`M_\odot` or tuple with units)
    :type primary_mass: float or tuple
    :param secondary_mass: secondary mass (:math:`M_\odot` or tuple with units)
    :type secondary_mass: float or tuple
    :param q: mass ratio
    :type q: float
    :param sma: system semi-major axis
    :type sma: float
    :param period: orbital period
    :type period: float
    :param adopt: flag to automatically set potential value in the system (default)
     or not
    :type adopt: bool
    :return: new value of the potential
    :rtype: float    
    """
    
    # For generality, we allow the binary to be given as a BodyBag or Bundle
    if not isinstance(mybundle, universe.BodyBag):
        mysystem = mybundle.get_system()
        
    # If values are given as floats, they are default Phoebe units: solar mass,
    # solar radii, days. Else, they should be tuples with the units
    if primary_mass is not None and not np.isscalar(primary_mass):
        primary_mass = conversions.convert(primary_mass[1], 'Msol', primary_mass[0])
    if secondary_mass is not None and not np.isscalar(secondary_mass):
        secondary_mass = conversions.convert(secondary_mass[1], 'Msol', secondary_mass[0])
    if sma is not None and not np.isscalar(sma):
        sma = conversions.convert(sma[1], 'Rsol', sma[0])
    if period is not None and not np.isscalar(period):
        period = conversions.convert(period[1], 'd', period[0])
    
    # Retrieve the orbit (the orbit is given in the primary and secondary, and
    # because it's the same orbit it doesn't matter which one you take)
    orbit = mysystem[0].params['orbit']
    totalmass = None
    
    # Shortcut: if everything is zero, just compute the individual masses from
    # the existing parameters
    if sum([0 if x is None else 1 for x in [primary_mass, secondary_mass, q,
                                            sma, period]]) == 0:
        q = orbit['q']
        sma = orbit['sma']
        period = orbit['period']
        
    if sum([0 if x is None else 1 for x in [primary_mass, secondary_mass, q,
                                            sma, period]]) <=2:
        raise ValueError("You need to give at least 3 properties to set_mass")
    
    
    
    # Fill in the component masses if possible
    if primary_mass is not None and secondary_mass is None and q is not None:
        secondary_mass = primary_mass * q
    elif secondary_mass is not None and primary_mass is None and q is not None:
        primary_mass = secondary_mass / q
    elif primary_mass is not None and secondary_mass is None and q is None:
        # Then sma and period need to be given
        if sma is None or period is None:
            raise ValueError("If you give primary_mass without secondary_mass or q, you need to give sma and period")
    elif secondary_mass is not None and primary_mass is None and q is None:
        # Then sma and period need to be given
        if sma is None or period is None:
            raise ValueError("If you give secondary_mass without primary_mass or q, you need to give sma and period")
        
    # We have primary and secondary masses
    if primary_mass is not None and secondary_mass is not None:
        if q is not None and not np.allclose(q, secondary_mass / primary_mass):
            raise ValueError("You cannot give primary_mass, secondary_mass and an inconsistent q value")
        
        q = secondary_mass / primary_mass
        totalmass = primary_mass + secondary_mass    
    
    if primary_mass is not None and secondary_mass is not None and sma is not None and period is not None:
        raise ValueError("You cannot give primary_mass, secondary_mass (and/or q), sma and period")
    elif sma is not None:
        sma = sma * constants.Rsol / constants.au
                
    out = keplerorbit.third_law(totalmass=totalmass, sma=sma, period=period)
        
    # Which one did we compute?
    if sma is None:
        sma = out
    elif period is None:
        period = out
    elif totalmass is None:
        totalmass = out
    
    # All info should now be available, so compute primary and secondary masses
    # if possible
    if primary_mass is not None:
        secondary_mass = totalmass - primary_mass
    elif secondary_mass is not None:
        primary_mass = totalmass - secondary_mass
    elif q is not None:
        primary_mass = totalmass / (1+q)
        secondary_mass = totalmass - primary_mass
    q = secondary_mass / primary_mass
    
    # And convert sma back to solar radii
    sma = sma * constants.au / constants.Rsol
    
    # Adopt the values if necessary
    if adopt:
        # If something has changed, warn the user that the potential values
        # might be inconsistent
        if not np.allclose([q, sma, period], [orbit['q'], orbit['sma'], orbit['period']]):
            logger.warning("Change in q, sma or period: adapting potential values to maintain relative radii (approximately)")
            old_pot1 = mysystem[0].params['component']['pot']
            old_pot2 = mysystem[1].params['component']['pot']
            syncpar1 = mysystem[0].params['component']['syncpar']
            syncpar2 = mysystem[1].params['component']['syncpar']
            d = 1 - orbit['ecc']
            old_radius1 = roche.potential2radius(old_pot1, q=q, d=d, F=syncpar1, component=1, sma=orbit['sma'])
            old_radius2 = roche.potential2radius(old_pot2, q=1./q, d=d, F=syncpar2, component=2, sma=orbit['sma'])
        
            orbit['q'] = q
            orbit['sma'] = sma
            orbit['period'] = period
        
            new_pot1 = roche.radius2potential(old_radius1, q=q, d=d, F=syncpar1, component=1, sma=sma)
            new_pot2 = roche.radius2potential(old_radius2, q=1./q, d=d, F=syncpar2, component=2, sma=sma)
            mysystem[0].params['component']['pot'] = new_pot1
            mysystem[1].params['component']['pot'] = new_pot2
            
        else:
            orbit['q'] = q
            orbit['sma'] = sma
            orbit['period'] = period
    
    return dict(primary_mass=primary_mass, secondary_mass=secondary_mass, q=q,
                sma=sma, period=period)


def compute_from_spectroscopy(mybundle, period, ecc, K1=None, K2=None,
                              logg1=None, logg2=None, vsini1=None, vsini2=None):
    """
    Set value binary parameters from spectroscopic measurements.
    
    Set mass ratio q, asini, syncpar...
    """
    raise NotImplementedError


def evaluate(par_values, twigs, mybundle, computelabel='preview'):
    """
    Generic evaluate function example.
    
    This function can be used as a prototype or implementation example for
    custom fitting functions. You could use this in your own optimizers. This
    function returns logp, so optimizers need to *maximize* the return value of
    this function. To use it on minimizers, just do -2*logp. Then you are
    *minimizing* the chi2.
    """
    for twig, value in zip(twigs, par_values):
        mybundle.set_value(twig, value)
    
    if not mybundle.check():
        logp = -np.inf
    else:
        try:
            mybundle.run_compute(computelabel)
            logp = mybundle.get_logp()
        except:
            logp = -np.inf
    
    return logp
    
def _xy_from_category(category):
    """
    returns the x and y arrays given a dataset and its category
    """
    
    # TODO: handle phase here
    if category=='lc':
        xk = 'time'
        yk = 'flux'
        xl = 'Time'
        yl = 'Flux'
    elif category=='rv':
        xk = 'time'
        yk = 'rv'
        xl = 'Time'
        yl = 'RV'
    elif category=='sp':
        # TODO: check these
        xk = 'wavelength'
        yk = 'flux'
        xl = 'Wavelength'
        yl = 'Flux'
    elif category=='etv':
        xk = 'time'
        yk = 'etv'
        xl = 'Time'
        yl = 'ETV'
    else:
        logger.warning("{} category not currently supported in frontend plotting".format(category))
        xk = None
        yk = None
        xl = None
        yl = None
        
    return xk, yk, xl, yl 
        
