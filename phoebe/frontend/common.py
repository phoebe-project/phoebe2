import os
from collections import OrderedDict
from fnmatch import fnmatch
import copy
from phoebe.parameters import parameters
from phoebe.backend import universe


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
        return self.sections.keys()
        
    def values(self):
        return self.sections.values()
        
    def items(self):
        return self.sections.items()
        
    def __getitem__(self, twig):
        return self.get_value(twig)
    
    def __setitem__(self, twig, value):
        self.set_value(twig, value)
    
    #~ def __iter__(self):
        #~ for _yield in self._loop_through_container(return_type='item'):
            #~ yield _yield
    
    def _loop_through_container(self,container=None):
        return_items = []
        for section_name,section in self.sections.items():
            for item in section:
                ri = self._get_info_from_item(item,section=section_name,container=container)
                if ri is not None:
                    return_items.append(ri)
                    
                    if ri['kind']=='Container':
                        return_items += ri['item']._loop_through_container(container=self.__class__.__name__)
                    elif ri['kind']=='BodyBag':
                        return_items += self._loop_through_system(item, section_name=section_name)
                    elif ri['kind']=='ParameterSet': # these should be coming from the sections
                        return_items += self._loop_through_ps(item, section_name=section_name, container=container, label=ri['label'])
        
        return return_items
        
    def _loop_through_ps(self, ps, section_name, container, label):
        return_items = []
        
        for qualifier in ps:
            item = ps.get_parameter(qualifier)
            ri = self._get_info_from_item(item, section=section_name, context=ps.get_context(), container=container, label=label)
            if ri['qualifier'] not in ['ref','label']:
                return_items.append(ri)
            
        return return_items
        
    def _loop_through_system(self, system, section_name):
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
        container = self.__class__.__name__ if container is None else container
        kind = item.__class__.__name__

        if isinstance(item, parameters.ParameterSet):
            labels = [ipath.get_label() for ipath in path if hasattr(ipath, 'get_label')] if path else []
            component = labels[-1] if len(labels) else None
            context = item.get_context()
            label = item.get_value('label') if 'label' in item else None
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
                #~ if len(labels)>3 and labels[-3][-3:] in ['obs','dep']:
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
            label = None
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
        self.trunk = self._loop_through_container()
        
    def _make_twig(self, path):
        """
        compile a twig for an ordered list of strings that makeup the path
        this list should be [qualifier,label,context,component,section,container]        
        """
        # path should be an ordered list (bottom up) of strings
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
        return a list of twigs that match the input (from left to right)
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
        
    def _get_by_search(self, twig=None, all=False, ignore_errors=False, return_trunk_item=False, **kwargs):
        # can take kwargs for searching by any other key stored in the trunk dictionary
        
        # first let's search through the trunk by section
        trunk = self.trunk
        for key in kwargs.keys():
            if len(trunk) and key in trunk[0].keys():
                trunk = [ti for ti in trunk if ti[key]==kwargs[key]]
        
        if twig is not None:
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
                
    def _get_by_section(self, label=None, section=None, all=False, ignore_errors=False, return_trunk_item=False):
        twig = "{}@{}".format(label,section) if label is not None and section is not None else None
        #~ twig = section if label is None else label
        return self._get_by_search(twig,section=section,kind='ParameterSet',
                        all=all, ignore_errors=ignore_errors, 
                        return_trunk_item=return_trunk_item)
                        
    def _get_dict_of_section(self, section):
        all_ti = self._get_by_search(twig=None, section=section, kind='ParameterSet',
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
        return [t['twig_full'] for t in self.trunk]

    def search(self, twig):
        return self._search_twigs(twig)
        
    def match(self, twig):
        return self._match_twigs(twig)
    
    def get(self, twig):
        return self._get_by_search(twig)
        
    def get_all(self, twig):
        return self._get_by_search(twig, all=True)
        
    def get_ps(self, twig):
        return self._get_by_search(twig, kind='ParameterSet')

    def get_parameter(self, twig):
        return self._get_by_search(twig, kind='Parameter')
        
    def info(self, twig):
        return str(self.get_parameter(twig))
                
    def get_value(self, twig):
        return self.get_parameter(twig).get_value()
        
    def set_value(self, twig, value, unit=None):
        param = self.get_parameter(twig)
        
        if unit is None:
            param.set_value(value)
        else:
            param.set_value(value, unit)
            
    def set_value_all(self, twig, value, unit=None):
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
            if unit is None:
                param.set_value(value)
            else:
                param.set_value(value, unit)
                
    def get_adjust(self, twig):
        return self.get_parameter(twig).get_adjust()
        
    def set_adjust(self, twig, value):
        param = self.get_parameter(twig)
        
        if not param.has_prior() and param.get_qualifier() not in ['l3','pblum']:
            lims = param.get_limits()
            param.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
        param.set_adjust(value)
        
    def set_adjust_all(self, twig, value):
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
            if not param.has_prior() and param.get_qualifier() not in ['l3','pblum']:
                lims = param.get_limits()
                param.set_prior(distribution='uniform', lower=lims[0], upper=lims[1])
            param.set_adjust(value)
    
    def get_prior(self, twig):
        return self.get_parameter(twig).get_prior()
        
    def set_prior(self, twig, **dist_kwargs):
        param = self.get_parameter(twig)
        param.set_prior(**dist_kwargs)
        
