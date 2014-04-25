import os
import functools
from collections import OrderedDict
from fnmatch import fnmatch
import copy
import json
import uuid
import numpy as np
from phoebe.parameters import parameters, datasets
from phoebe.parameters import datasets
from phoebe.backend import universe
from phoebe.utils import config

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
            
            # Else, we need take care of *a lot* of stuff:
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
        return a list of matching twigs (with same method as used in
        get_value, get_parameter, get_prior, etc)
        
        @param twig: the search twig
        @type twig: str
        @return: list of twigs
        @rtype: list of strings
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
        return self._get_by_search(twig, kind='Parameter') # was 'Parameter*'
        
    def info(self, twig):
        """
        Retrieve info on a Parameter.
        
        This is just a shortcut to str(get_parameter(twig))
        
        :param twig: the search twig
        :type twig: str
        :return: info
        :rtype: str
        """
        return str(self.get_parameter(twig))
           
           
    def get_value(self, twig):
        """
        Retrieve the value of a Parameter
        
        :param twig: the search twig
        :type twig: str
        :raises KeyError: when twig is not available or is not a Parameter
        """
        return self.get_parameter(twig).get_value()
        
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
        param = self.get_parameter(twig)
        
        # special care needs to be taken when setting labels and refs
        qualifier = param.get_qualifier()
        
        # Setting a label means not only changing that particular Parameter,
        # but also the property of the Body
        if qualifier == 'label':
            this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
            component = self._get_by_search(this_trunk['label'])
            component.set_label(value)
            self._build_trunk()
        
        # Changing a ref needs to change all occurrences
        elif qualifier == 'ref':
            # get the system
            from_ = param.get_value()
            system = self.get_system()
            system.change_ref(from_, value)
            self._build_trunk()
            return None
        
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
    
        # if we decide upon a dictionary
        matched_twigs = self._match_twigs(twig, hidden=False)
        return {twig:param.get_value() for twig, param in zip(matched_twigs, params)}
        
        
    
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
    def attach_ps(self, twig, value):
        """
        Add a new ParameterSet.
        
        [FUTURE]
        """
        # Get all the info we can get
        this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
        
        if isinstance(this_trunk['item'], universe.Body):
            # last check: we need to make sure that whatever we're setting
            # already exists. We cannot assign a new nonexistent PS to the Body
            # (well, we could, but we don't want to -- that's attach_ps
            # responsibility)
            given_context = value.get_context()
            try:
                this_trunk['item'].set_params(value, force=False)
            except ValueError:
                raise ValueError(("ParameterSet '{}' at Body already exists. "
                                 "Please use set_ps to override "
                                 "it.").format(given_context))
        
        elif isinstance(this_trunk['item'], dict): #then this should be a sect.
            section = twig.split('@')[0]
            if value.get_value('label') not in [c.get_value('label') \
                                               for c in self.sections[section]]:
                self._add_to_section(section, value)
                
        else:
            raise ValueError(("You can only attach ParameterSets to a Body or "
                              "Section ('{}' refers to a "
                              "{})").format(twig, this_trunk['kind']))
        
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
        
        if not par.has_prior() and par.get_qualifier() not in ['l3', 'pblum']:
            lims = par.get_limits()
            par.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
        par.set_adjust(value)
        
    def set_adjust_all(self, twig, value):
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
            if not par.has_prior() and par.get_qualifier() not in ['l3','pblum']:
                lims = par.get_limits()
                par.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
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
    
    def get_compute(self,label=None,create_default=False):
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
    def add_compute(self,ps=None,**kwargs):
        """
        Add a new compute ParameterSet
        
        @param ps: compute ParameterSet (or None)
        @type ps:  None or ParameterSet
        @param label: label of the compute options (will override label in ps)
        @type label: str
        """
        if ps is None:
            ps = parameters.ParameterSet(context='compute')
        for k,v in kwargs.items():
            ps.set_value(k,v)
            
        self._add_to_section('compute',ps)

        # had to remove this to make the deepcopy in run_compute work 
        # correctly.  Also - this shouldn't ever be called from within
        # the Container class but only within Bundle
        #~ self._attach_set_value_signals(ps)
    
    @rebuild_trunk
    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        compute = self.get_compute(label)
        self.sections['compute'].remove(compute)
        
    @rebuild_trunk
    def add_fitting(self,ps=None,**kwargs):
        """
        Add a new fitting ParameterSet
        
        [FUTURE]
        
        @param ps: fitting ParameterSet
        @type ps:  None, or ParameterSet
        @param label: name of the fitting options (will override label in ps)
        @type label: str
        """
        context = kwargs.pop('context') if 'context' in kwargs.keys() else 'fitting:pymc'
        if fitting is None:
            fitting = parameters.ParameterSet(context=context)
        for k,v in kwargs.items():
            fitting.set_value(k,v)
            
        self._add_to_section('fitting',fitting)
        self._attach_set_value_signals(fitting)
            
    def get_fitting(self,label=None):
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
    def remove_fitting(self,label):
        """
        Remove a given fitting ParameterSet
        
        [FUTURE]
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        fitting = self.get_fitting(label)
        self.sections['fitting'].remove(fitting)
    
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
            if do_sectionlevel and section_name not in ['system']:
                ris = self._get_info_from_item({item.get_value('label'):item for item in section},section=section_name,container=container,label=-1)
                return_items += ris
                
            if do_pslevel:
                for item in section:
                    ris = self._get_info_from_item(item,section=section_name,container=container,label=item.get_value('label') if label is None else label)
                    for ri in ris:
                        return_items += [ri]
                        
                        if ri['kind']=='Container':
                            return_items += ri['item']._loop_through_container(container=self.__class__.__name__, label=ri['label'], ref=ref)

                        elif ri['class_name']=='BodyBag':
                            return_items += self._loop_through_system(item, section_name=section_name)
                       
                        elif ri['kind']=='ParameterSet': # these should be coming from the sections
                            return_items += self._loop_through_ps(item, section_name=section_name, container=container, label=ri['label'])

        return return_items
        
    def _loop_through_ps(self, ps, section_name, container, label):
        """
        called when _loop_through_container hits a PS
        
        [FUTURE]
        """
        return_items = []
        
        for qualifier in ps:
            item = ps.get_parameter(qualifier)
            ris = self._get_info_from_item(item, section=section_name, context=ps.get_context(), container=container, label=label)
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
            
            # make sure to catch the obs and pbdep
            if isinstance(item, str) and item[-3:] in ['obs', 'dep','syn']:
                
                # we completely ignore existing syns, we build them on the same
                # level as obs are available
                #~ if item[-3:] == 'syn':
                    #~ continue
                
                for itype in path[-2]:
                    for isubtype in path[-2][itype].values():
                        if item==itype:
                            ris = self._get_info_from_item(isubtype, path=path, section=section_name)
                            for ri in ris:
                                if item[-3:]=='syn':
                                    # then this is the true syn, which we need to keep available
                                    # for saving and loading, but will hide from the user in twig access
                                    ri['hidden']=True
                                
                                return_items += [ri]
                                
                        # we want to see if there any syns associated with the obs
                        if itype[-3:] == 'obs':
                            bodies = [self.get_system()] + [thing for thing in path if isinstance(thing, universe.Body)]
                            syn = bodies[-1].get_synthetic(category=item[:-3], ref=isubtype['ref'])
                            if syn is not None: 
                                syn = syn.asarray()
                                ris = self._get_info_from_item(syn, path=path, section=section_name)
                                return_items += ris
                                
                                
                                subpath = list(path[:-1]) + [syn['ref']]
                                subpath[-3] = syn.get_context()
                                
                                # add phase to synthetic
                                #if 'time' in syn and len(syn['time']) and not 'phase' in syn:
                                    #period = self.get_value('period@orbit')
                                    #par = parameters.Parameter('phase', value=np.mod(syn['time'], period), unit='cy', context=syn.get_context())
                                    #syn.add(par)
                                
                                for par in syn:
                                    mypar = syn.get_parameter(par)
                                    ris = self._get_info_from_item(mypar, path=subpath+[mypar], section=section_name)
                                    return_items += ris
                                    
                                                                                    
                    # but also add the collection of dep/obs
                    if item==itype:
                        ris = self._get_info_from_item(path[-2][itype], path=list(path)+[item], section=section_name)
                        #~ if ri['twig_full'] not in [r['twig_full'] for r in return_items]:
                        return_items += ris
            
            elif isinstance(item, OrderedDict):
                continue
                
            else:
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
    
        if isinstance(item, parameters.ParameterSet):
            kind = 'ParameterSet'
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            if len(labels):
                label = labels[-1]
            elif hasattr(self, 'get_system') and label is None:
                label = self.get_system().get_label()
            else:
                label = label
            context = item.get_context()
            ref = item.get_value('ref') if 'ref' in item else ref
            unique_label = None
            qualifier = None
            
        elif isinstance(item, parameters.Parameter):
            kind = 'Parameter'
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
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
            ref = ref
            if path:
                if context[-3:] in ['obs','dep','syn']:
                    # then we need to get the ref of the obs or dep, which is placed differently in the path
                    # due to bad design by Pieter, this needs a hack to make sure we get the right label
                    ref = path[-2]
                
            unique_label = item.get_unique_label()
            qualifier = item.get_qualifier()
            
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
        else:
            return []
            #~ raise ValueError("building trunk failed when trying to parse {}".format(kind))
            
        # now let's do specific overrides
        if context == 'orbit':
            # we want to hide the fact to the user that these exist at the component level
            # and instead fake its label to be that of its parent BodyBag
            #~ label = self.get_parent(label).get_label()
            label = self._get_object(label).get_parent().get_label()
            ref = None
        
        if context == section:
            context_twig = None
        else:
            context_twig = context
         
        hidden = qualifier in ['c1label', 'c2label']
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
        
        if 'hidden' not in kwargs:
            kwargs['hidden'] = False
        for key in kwargs.keys():
            #~ print "*** Container._filter_twigs_by_kwargs", key, kwargs[key]
            if len(trunk) and key in trunk[0].keys() and kwargs[key] is not None:
                trunk = [ti for ti in trunk if ti[key] is not None and 
                    (ti[key]==kwargs[key] or 
                    (isinstance(kwargs[key],list) and ti[key] in kwargs[key]) or
                    (isinstance(kwargs[key],str) and fnmatch(ti[key],kwargs[key])))]
                
        return trunk
        
        
    def _match_twigs(self, twiglet, trunk=None, **kwargs):
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
        matching_indices = range(len(matching_twigs))
        
        # We attempt to match a twig twice; once with colons describing
        # subcontext, once without
        
        for attempt in (0,1):
            
            # second attempt is without colons denoting subcontexts
            if attempt == 1:
                matching_twigs = [[st.split(':')[0] for st in t['twig_full'].split('@')] for t in trunk]
                matching_indices = range(len(matching_twigs))
                
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
            
            ret_value = [matching_twigs_orig[i] for i in matching_indices] 
            
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
            results = ', '.join(matched_twigs)
            raise KeyError(("more than one result was found matching the "
                            "criteria: {}").format(results))
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
                 
        return {ti['label']:ti['item'] for ti in all_ti} if all_ti is not None else {}
        
    def _save_json(self, filename):
        """
        TESTING
        [FUTURE]
        """
        # We're changing stuff here, so we need to make a copy first
        this_bundle = self.copy()
        
        this_container = this_bundle.__class__.__name__

        dump_dict = {}
        
        dump_dict['PHOEBE Version'] = '2.0alpha'
        
        if hasattr(self, 'get_system'):
            dump_dict['Hierarchy'] = _dumps_system_structure(this_bundle.get_system())

        dump_dict['ParameterSets'] = {}

        for ti in this_bundle._get_by_search(kind='ParameterSet',container=this_container,all=True,hidden=None,return_trunk_item=True):
            item = ti['item']
            
            if ti['context'] not in ['orbit','component','mesh:marching']:
                info = {}
                
                if ti['context'][-3:] in ['obs','syn']:
                    # then unload the data first
                    item.unload()
                    
                if not (ti['context'][-3:]=='syn' and not ti['hidden']):
                    info['context'] = ti['context']
                    #~ info['class_name'] = ti['class_name']
                    #~ info['parent'] = self._make_twig([qualifier,ref,context,label,section,container])
                    dump_dict['ParameterSets'][ti['twig']] = info
        
        dump_dict['Parameters'] = {}
        for ti in this_bundle._get_by_search(kind='Parameter',container=this_container,all=True,hidden=None,return_trunk_item=True):
            item = ti['item']
            info = {}
            
            info['value'] = item.get_value()#to_str()
            
            if isinstance(info['value'],np.ndarray):
                info['value'] = list(info['value'])
            
            #~ if hasattr(item, 'get_unit') and item.has_unit():
                #~ info['_unit (read-only)'] = item.get_unit()
            
            if hasattr(item, 'adjust') and item.adjust:
                info['adjust'] = item.adjust
                
            #~ if item.has_prior():
                #~ info['prior'] = 'has prior, but export not yet supported'
            
            if not (ti['context'][-3:]=='syn' and not ti['hidden']):
                # the real syns are marked as hidden, whereas the fake
                # ones forced to match the obs are not hidden
                dump_dict['Parameters'][ti['twig']] = info
            
        f = open(filename, 'w')
        f.write(json.dumps(dump_dict, sort_keys=True, indent=4, separators=(',', ': ')))
        f.close()


    @rebuild_trunk
    def _load_json(self, filename):
        """
        TESTING
        
        [FUTURE]
        """
        f = open(filename, 'r')
        load_dict = json.load(f)
        f.close()
        
        if hasattr(self, 'get_system'):
            self.set_system(_loads_system_structure(load_dict['Hierarchy']))
            #print self.get_system().list(summary='full')
            
        for twig,info in load_dict['ParameterSets'].items():
            label = str(twig.split('@')[0])
            where = twig.split('@').index(info['context'])
            if info['context'] not in self.sections.keys(): where+=1
            parent_twig = '@'.join(twig.split('@')[where:])
            
            # lcdeps etc are ParameterSets, lcobs/lcsyn are DataSets (which are
            # a sublcass of ParameterSets but with extra functionality)
            context = str(info['context'])
            if context[-3:] in ['obs', 'syn']:
                ps = getattr(datasets, config.dataset_class[context[:-3]])(context=context)
            else:
                ps = parameters.ParameterSet(context)
                
            if 'label' in ps.keys():
                ps.set_value('label',label)
            elif 'ref' in ps.keys():
                ps.set_value('ref', label)
            #~ print "self.attach_ps('{}', {}(context='{}', {}='{}'))".format(parent_twig, 'PS' if info['context'][-3:] not in ['obs','syn'] else 'DataSet', info['context'], 'ref' if info['context'][-3:] in ['obs','dep','syn'] else 'label', label)
            self.attach_ps(parent_twig, ps)
            
        self._build_trunk()
        
        for twig,info in load_dict['Parameters'].items():
            #~ print "self.set_value('{}', '{}')".format(twig, str(info['value']))
            item = self._get_by_search(twig, hidden=None)
            if 'value' in info:
                item.set_value(info['value'])
            if 'adjust' in info:
                #~ print "HERE", twig, info['adjust']
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
