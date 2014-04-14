import os
import functools
from collections import OrderedDict
from fnmatch import fnmatch
import copy
from phoebe.parameters import parameters
from phoebe.backend import universe

def rebuild_trunk(fctn):
    """
    rebuild the cached trunk *after* running the original function
    """
    @functools.wraps(fctn)
    def rebuild(container,*args,**kwargs):
        return_ = fctn(container, *args, **kwargs)
        container._build_trunk()
        return return_
    return rebuild

class Container(object):
    """
    Class that controlls accessing sections and parametersets
    
    This class in inherited by both the bundle and usersettings
    """
    
    def __init__(self):
        self.trunk = []
        self.sections = OrderedDict()
        
    ## act like a dictionary
    def keys(self):
        return self.list_twigs()
        
    def values(self):
        return [self.get_value(twig) for twig in self.list_twigs()]
        
    def items(self):
        return self.sections.items()
        
    def __getitem__(self, twig):
        return self.get_value(twig)
    
    def __setitem__(self, twig, value):
        self.set_value(twig, value)
    
    #~ def __iter__(self):
        #~ for _yield in self._loop_through_container(return_type='item'):
            #~ yield _yield
    
    def _loop_through_container(self,container=None,label=None):
        """
        loop through the current container to compile all items for the trunk
        this will then be called recursively if it hits another Container/PS/BodyBag
        """
        return_items = []
        for section_name,section in self.sections.items():
            for item in section:
                ri = self._get_info_from_item(item,section=section_name,container=container,label=label)
                if ri is not None:
                    return_items.append(ri)
                    
                    if ri['kind']=='Container':
                        return_items += ri['item']._loop_through_container(container=self.__class__.__name__, label=ri['label'])
                    elif ri['kind']=='BodyBag':
                        return_items += self._loop_through_system(item, section_name=section_name)
                    elif ri['kind']=='ParameterSet': # these should be coming from the sections
                        return_items += self._loop_through_ps(item, section_name=section_name, container=container, label=ri['label'])
        
        return return_items
        
    def _loop_through_ps(self, ps, section_name, container, label):
        """
        called when _loop_through_container hits a PS
        """
        return_items = []
        
        for qualifier in ps:
            item = ps.get_parameter(qualifier)
            ri = self._get_info_from_item(item, section=section_name, context=ps.get_context(), container=container, label=label)
            if ri['qualifier'] not in ['ref','label']:
                return_items.append(ri)
            
        return return_items
        
    def _loop_through_system(self, system, section_name):
        """
        called when _loop_through_container hits the system
        """
        return_items = []
        
        for path, item in system.walk_all(path_as_string=False):
            ri = self._get_info_from_item(item, path=path, section=section_name)
            
            #~ print path, ri['twig_full'] if ri is not None else None #, ri['context']!='syn' if ri is not None else None, (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) if ri is not None else None
            #~ print path[-1:], ri['twig_full'] if ri is not None else None, ri['context']!='syn' if ri is not None else None, (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) if ri is not None else None
            
            # ignore parameters that are synthetics and make sure this is not a duplicate
            if ri is not None and ri['context']!='syn' and ri['qualifier'] not in ['ref','label'] \
                    and (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) \
                    and ri['twig_full'] not in [r['twig_full'] for r in return_items]:
                return_items.append(ri)
            
        return return_items
        
    def _get_info_from_item(self, item, path=None, section=None, container=None, context=None, label=None):
        """
        general function to pull all information we need from any item we may come across
    
        providing other information (section, container, context, label) may be necessary if we don't know where we are
        """
        container = self.__class__.__name__ if container is None else container
        kind = item.__class__.__name__

        if isinstance(item, parameters.ParameterSet):
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            component = labels[-1] if len(labels) else None
            context = item.get_context()
            label = item.get_value('label') if 'label' in item else label
            unique_label = None
            qualifier = None
        elif isinstance(item, parameters.Parameter):
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            component = labels[-1] if len(labels) else None
            if path:
                #then coming from the system and we need to build the context from the path
                #~ context = path[-2] if isinstance(path[-2],str) else path[-2].get_context()
                context = item.get_context()
                if isinstance(context, list): #because sometimes its returning a list and sometimes a string
                    context = context[0]
            else:
                #then we're coming from a section and already know the context
                context = context
            if path:
                if context[-3:] in ['obs','dep']:
                    # then we need to get the ref of the obs or dep, which is placed differently in the path
                    # do to bad design by Pieter, this needs a hack to make sure we get the right label
                    label = path[-2]
            else:
                label = label
            unique_label = item.get_unique_label()
            qualifier = item.get_qualifier()
        elif isinstance(item, universe.Body):
            component = item.get_label()
            context = None
            label = None
            unique_label = None
            qualifier = None
        elif isinstance(item, Container):
            kind = 'Container'
            component = None
            context = None
            label = item.get_label() if hasattr(item, 'get_label') else label
            unique_label = None
            qualifier = None
        else:
            return None
            #~ raise ValueError("building trunk failed when trying to parse {}".format(kind))
            
        # now let's do specific overrides
        if context == 'orbit':
            component = None
        
        if context == section:
            section_twig = None
        elif section == 'system' and component != self.get_system().get_label():
            section_twig = self.get_system().get_label()
        else:
            section_twig = section
            
        # twig = <qualifier>@<label>@<context>@<component>@<section>@<container>
        #~ print [qualifier,label,context,component,section_twig,container]
        twig = self._make_twig([qualifier,label,context,component,section_twig])
        twig_full = self._make_twig([qualifier,label,context,component,section_twig,container])
        
        return dict(qualifier=qualifier, component=component,
            container=container, section=section, kind=kind, 
            context=context, label=label, unique_label=unique_label, 
            twig=twig, twig_full=twig_full, item=item)
            
    def _build_trunk(self):
        """
        this function should be called (probably through the @rebuild_trunk decorator)
        whenever anything is changed (besides changing values, etc)
        
        this sets self.trunk which is then used whenever the container is searched for an item
        """
        self.trunk = self._loop_through_container()
        
    def _make_twig(self, path):
        """
        compile a twig for an ordered list of strings that makeup the path
        this list should be [qualifier,label,context,component,section,container]        
        """
        while None in path:
            path.remove(None)
        return '@'.join(path)
        
    def _search_twigs(self, twigglet, trunk=None):
        """
        return a list of twigs where twigglet is a substring
        """
        if trunk is None:
            trunk = self.trunk
        twigs = [t['twig_full'] for t in trunk]
        return [twig for twig in twigs if twigglet in twig]
        
    def _match_twigs(self, twigglet, trunk=None):
        """
        return a list of twigs that match the input
        the first item is required to be first, but any other item
        must simply occur left to right
        """
        if trunk is None:
            trunk = self.trunk
        twig_split = twigglet.split('@')
        
        matching_twigs_orig = [t['twig_full'] for t in trunk]
        matching_twigs = [t['twig_full'].split('@') for t in trunk]
        matching_indices = range(len(matching_twigs))
        
        for tsp_i,tsp in enumerate(twig_split):
            remove = []
            for mtwig_i, mtwig in enumerate(matching_twigs):
                if (tsp_i != 0 and tsp in mtwig) or tsp==mtwig[0]:
                    # then find where we are, and then only keep stuff to the right
                    ind = mtwig.index(tsp)
                    mtwig = mtwig[ind+1:]
                else:
                    # then remove from list of matching twigs
                    remove.append(mtwig_i)
            
            for remove_i in sorted(remove,reverse=True):
                matching_twigs.pop(remove_i)
                matching_indices.pop(remove_i)
                
        return [matching_twigs_orig[i] for i in matching_indices] 
        
    def _get_by_search(self, twig=None, all=False, ignore_errors=False, return_trunk_item=False, use_search=False, **kwargs):
        """
        this function searches the cached trunk
        kwargs will filter any of the keys stored in trunk (section, kind, container, etc)
        
        @param twig: the twig/twigglet to use when searching
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
        
        # first let's search through the trunk by section
        trunk = self.trunk
        for key in kwargs.keys():
            if len(trunk) and key in trunk[0].keys():
                trunk = [ti for ti in trunk if ti[key]==kwargs[key]]
        
        if twig is not None:
            if use_search:
                matched_twigs = self._search_twigs(twig, trunk)
            else:
                matched_twigs = self._match_twigs(twig, trunk)
        else:
            matched_twigs = [ti['twig_full'] for ti in trunk]
        
        if len(matched_twigs) == 0:
            if ignore_errors:
                return None
            raise ValueError("no match found matching the criteria")
        elif all == False and ignore_errors == False and len(matched_twigs) > 1:
            results = ', '.join(matched_twigs)
            raise ValueError("more than one result was found matching the criteria: {}".format(results))
        else:
            items = [ti if return_trunk_item else ti['item'] for ti in trunk if ti['twig_full'] in matched_twigs]
            if all:
                return items
            else:
                return items[0]
                
    def _get_by_section(self, label=None, section=None, kind="ParameterSet", all=False, ignore_errors=False, return_trunk_item=False):
        """
        shortcut to _get_by_search which automatically builds a searching twig from the label
        and has defaults for kind (ParameterSet)        
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
        """
        all_ti = self._get_by_search(twig=None, section=section, kind=kind,
                        all=True, ignore_errors=True,
                        return_trunk_item=True)
                        
        return {ti['label']:ti['item'] for ti in all_ti} if all_ti is not None else {}
                
    ## generic functions to get non-system parametersets
    def _return_from_dict(self, dictionary, all=False, ignore_errors=False):
        """
        this functin takes a dictionary of results from a searching function
        ie. _get_from_section and returns either a single item or the dictionary
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
    def _add_to_section(self,section,ps):
        """
        add a new parameterset to section - the label of the parameterset
        will be used as the key to retrieve it using _get_from_section
        
        @param section: name of the section (key of self.sections)
        @type section: str
        @param ps: the new parameterset with label set
        @type ps: ParameterSet
        """
        if section not in self.sections.keys():
            self.sections[section] = []
        self.sections[section].append(ps)
        
    def list_twigs(self):
        """
        return a list of all available twigs
        """
        return [t['twig_full'] for t in self.trunk]

    def search(self, twig):
        """
        return a list of twigs matching a substring search
        
        @param twig: the search twig
        @type twig: str
        @return: list of twigs
        @rtype: list of strings
        """
        return self._search_twigs(twig)
        
    def match(self, twig):
        """
        return a list of matching twigs (with same method as used in
        get_value, get_parameter, get_prior, etc)
        
        @param twig: the search twig
        @type twig: str
        @return: list of twigs
        @rtype: list of strings
        """
        return self._match_twigs(twig)
    
    def get(self, twig):
        """
        search and retrieve any item without specifying its type
        
        @param twig: the search twig
        @type twig: str
        @return: matching item
        @rtype: varies
        """
        return self._get_by_search(twig)
        
    def get_all(self, twig):
        """
        same as get except will return a list of items instead of throwing
        and error if more than one are found
        
        @param twig: the search twig
        @type twig: str
        @return: all matching items
        @rtype: list
        """
        return self._get_by_search(twig, all=True)
        
    def get_ps(self, twig):
        """
        retrieve a ParameterSet
        
        @param twig: the search twig
        @type twig: str
        @return: ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_search(twig, kind='ParameterSet')

    def get_parameter(self, twig):
        """
        retrieve a Parameter
        
        @param twig: the search twig
        @type twig: str
        @return: Parameter
        @rtype: Parameter
        """
        return self._get_by_search(twig, kind='Parameter')
        
    def info(self, twig):
        """
        retrieve info on a Parameter.
        this is just a shortcut to str(get_parameter(twig))
        
        @param twig: the search twig
        @type twig: str
        @return: info
        @rtype: str
        """
        return str(self.get_parameter(twig))
                
    def get_value(self, twig):
        """
        retrieve the value of a Parameter
        
        @param twig: the search twig
        @type twig: str
        """
        return self.get_parameter(twig).get_value()
        
    def set_value(self, twig, value, unit=None):
        """
        set the value of a Parameter
        
        @param twig: the search twig
        @type twig: str
        @param value: the value
        @type value: depends on Parameter
        @param unit: unit of value (if not default)
        @type unit: str or None
        """
        param = self.get_parameter(twig)
        
        if unit is None:
            param.set_value(value)
        else:
            param.set_value(value, unit)
            
    def set_value_all(self, twig, value, unit=None):
        """
        set the value of all matching Parameters
        
        @param twig: the search twig
        @type twig: str
        @param value: the value
        @type value: depends on Parameter
        @param unit: unit of value (if not default)
        @type unit: str or None
        """
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
            if unit is None:
                param.set_value(value)
            else:
                param.set_value(value, unit)
                
    def get_adjust(self, twig):
        """
        retrieve whether a Parameter is marked to be adjusted
        
        @param twig: the search twig
        @type twig: str
        @return: adjust
        @rtype: bool
        """
        return self.get_parameter(twig).get_adjust()
        
    def set_adjust(self, twig, value):
        """
        set whether a Parameter is marked to be adjusted
        
        @param twig: the search twig
        @type twig: str
        @param value: adjust
        @type value: bool
        """
        param = self.get_parameter(twig)
        
        if not param.has_prior() and param.get_qualifier() not in ['l3','pblum']:
            lims = param.get_limits()
            param.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
        param.set_adjust(value)
        
    def set_adjust_all(self, twig, value):
        """
        set whether all matching Parameters are marked to be adjusted
        
        @param twig: the search twig
        @type twig: str
        @param value: adjust
        @type value: bool
        """
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
            if not param.has_prior() and param.get_qualifier() not in ['l3','pblum']:
                lims = param.get_limits()
                param.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
            param.set_adjust(value)
    
    def get_prior(self, twig):
        """
        retrieve the prior on a Parameter
        
        @param twig: the search twig
        @type twig: str
        @return: prior
        @rtype: ParameterSet
        """
        return self.get_parameter(twig).get_prior()
        
    def set_prior(self, twig, **dist_kwargs):
        """
        set the prior on a Parameter
        
        @param twig: the search twig
        @type twig: str
        @param **kwargs: necessary parameters for distribution
        @type **kwargs: varies
        """
        param = self.get_parameter(twig)
        param.set_prior(**dist_kwargs)
        
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

        self._attach_set_value_signals(ps)
            
    def get_compute(self,label=None):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section(label,"compute")
    
    @rebuild_trunk
    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        return self._remove_from_section('compute',label)
        
    @rebuild_trunk
    def add_fitting(self,ps=None,**kwargs):
        """
        Add a new fitting ParameterSet
        
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
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        self._remove_from_section('fitting',label)
        
