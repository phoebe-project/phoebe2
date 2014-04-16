import os
import functools
from collections import OrderedDict
from fnmatch import fnmatch
import copy
import json
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
        """
        Define dictionary style access.
        
        Returns Bodies, ParameterSets or Parameter values (never Parameters).
        """
        ret_value = self._get_by_search(twig)
        if isinstance(ret_value, parameters.Parameter):
            ret_value = ret_value.get_value()
        return ret_value
    
    def __setitem__(self, twig, value):
        #if isinstance(value, parameter.ParameterSet):
        #    ret_value = self._get_by_search(twig)
        #elif isinstance(value, parameter.ParameterSet):
        self.set_value(twig, value)
        
    
    #~ def __iter__(self):
        #~ for _yield in self._loop_through_container(return_type='item'):
            #~ yield _yield
    
    def _loop_through_container(self,container=None,label=None,ref=None,do_sectionlevel=True,do_pslevel=True):
        """
        loop through the current container to compile all items for the trunk
        this will then be called recursively if it hits another Container/PS/BodyBag
        """
        return_items = []
        
        # add the container's sections
        if do_sectionlevel:
            ri = self._get_info_from_item(self.sections,section=None,container=container,label=None)
            return_items.append(ri)
        
        for section_name,section in self.sections.items():
            if do_sectionlevel and section_name not in ['system']:
                #~ print "***", section_name, OrderedDict((item.get_value('label'),item) for item in section)
                ri = self._get_info_from_item({item.get_value('label'):item for item in section},section=section_name,container=container,label=label)
                return_items.append(ri)
            if do_pslevel:
                for item in section:
                    ri = self._get_info_from_item(item,section=section_name,container=container,label=item.get_value('label') if label is None else label)
                    if ri is not None:
                        return_items.append(ri)
                        
                        if ri['kind']=='Container':
                            return_items += ri['item']._loop_through_container(container=self.__class__.__name__, label=ri['label'], ref=ref)
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
            # make sure to catch the obs and pbdep
            if isinstance(item, str) and item[-3:] in ['obs', 'dep', 'syn']:
                for itype in path[-2]:
                    for isubtype in path[-2][itype].values():
                        ri = self._get_info_from_item(isubtype, path=path, section=section_name)
                        return_items.append(ri)
                    
                    # but also add the dep/obs
                    ri = self._get_info_from_item(path[-2][itype], path=list(path)+[item], section=section_name)
                    return_items.append(ri)
            
            elif isinstance(item, OrderedDict):
                continue
                
            else:    
                
                ri = self._get_info_from_item(item, path=path, section=section_name)
            
                #~ print path, ri['twig_full'] if ri is not None else None #, ri['context']!='syn' if ri is not None else None, (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) if ri is not None else None
                #~ print path[-1:], ri['twig_full'] if ri is not None else None, ri['context']!='syn' if ri is not None else None, (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) if ri is not None else None
                
                # ignore parameters that are synthetics and make sure this is not a duplicate
                if ri is not None and ri['context']!='syn' \
                        and (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]) \
                        and ri['twig_full'] not in [r['twig_full'] for r in return_items]:
                    return_items.append(ri)
                
        return return_items
        
    def _get_info_from_item(self, item, path=None, section=None, container=None, context=None, label=None, ref=None):
        """
        general function to pull all information we need from any item we may come across
    
        providing other information (section, container, context, label) may be necessary if we don't know where we are
        """
        container = self.__class__.__name__ if container is None else container
        kind = item.__class__.__name__

        if isinstance(item, parameters.ParameterSet):
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
                if context[-3:] in ['obs','dep']:
                    # then we need to get the ref of the obs or dep, which is placed differently in the path
                    # do to bad design by Pieter, this needs a hack to make sure we get the right label
                    ref = path[-2]
                
            unique_label = item.get_unique_label()
            qualifier = item.get_qualifier()
        elif isinstance(item, universe.Body):
            label = item.get_label()
            context = None
            ref = None
            unique_label = None
            qualifier = None
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
            else:
                label = label
            context = path[-1] if path else None
            ref = None
            unique_label = None
            qualifier = None
        else:
            return None
            #~ raise ValueError("building trunk failed when trying to parse {}".format(kind))
            
        # now let's do specific overrides
        if context == 'orbit':
            label = None
            ref = None
        
        if context == section:
            context_twig = None
        else:
            context_twig = context
        if section == 'system' and label != self.get_system().get_label():
            section_twig = self.get_system().get_label()
        else:
            section_twig = section
            
        section_twig = section
         
        hidden = qualifier in ['c1label', 'c2label']
        #~ hidden = qualifier in ['ref','label', 'c1label', 'c2label']
        #hidden = False
            
        # twig = <qualifier>@<ref>@<context>@<label>@<section>@<container>
        twig = self._make_twig([qualifier,ref,context_twig,label,section_twig])
        #~ twig_reverse = self._make_twig([qualifier,ref,context,label,section_twig,container], invert=True)
        twig_full = self._make_twig([qualifier,ref,context_twig,label,section_twig,container])
        #~ twig_full_reverse = self._make_twig([qualifier,ref,context,label,section_twig,container], invert=True)
        
        
        return dict(qualifier=qualifier, label=label,
            container=container, section=section, kind=kind, 
            context=context, ref=ref, unique_label=unique_label, 
            twig=twig, twig_full=twig_full, path=path, 
            #~ twig_reverse=twig_reverse, twig_full_reverse=twig_full_reverse,
            item=item,
            hidden=hidden)
            
    def _build_trunk(self):
        """
        this function should be called (probably through the @rebuild_trunk decorator)
        whenever anything is changed (besides changing values, etc)
        
        this sets self.trunk which is then used whenever the container is searched for an item
        """
        self.trunk = self._loop_through_container()
        
    def _purge_trunk(self):
        """
        simply clears trunk - this will probably only be called before pickling to 
        minimize duplication and reduce filesize
        """
        self.trunk = []
        
    def _make_twig(self, path, invert=False):
        """
        compile a twig for an ordered list of strings that makeup the path
        this list should be [qualifier,label,context,component,section,container]        
        """
        if invert:
            path.reverse()
        while None in path:
            path.remove(None)
        return '{}'.format('/' if invert else '@').join(path)
        
    def _search_twigs(self, twigglet, trunk=None, **kwargs):
        """
        return a list of twigs where twigglet is a substring
        """
        trunk = self._filter_twigs_by_kwargs(trunk, **kwargs)
            
        twigs = [t['twig_full'] for t in trunk]
        return [twig for twig in twigs if twigglet in twig]
        
    def _filter_twigs_by_kwargs(self, trunk=None, **kwargs):
        """
        returns a list of twigs after filtering the trunk by any other
        information stored in the dictionaries
        """
        trunk = self.trunk if trunk is None else trunk
        
        if 'hidden' not in kwargs:
            kwargs['hidden'] = False
        for key in kwargs.keys():
            #~ print "*** Container._filter_twigs_by_kwargs", key, kwargs[key]
            if len(trunk) and key in trunk[0].keys() and kwargs[key] is not None:
                trunk = [ti for ti in trunk if ti[key] is not None and 
                    (ti[key]==kwargs[key] or (isinstance(kwargs[key],str) and fnmatch(ti[key],kwargs[key])))]
                
        return trunk
        
        
    def _match_twigs(self, twigglet, trunk=None, **kwargs):
        """
        return a list of twigs that match the input
        the first item is required to be first, but any other item
        must simply occur left to right
        """
        trunk = self._filter_twigs_by_kwargs(trunk, **kwargs)
        
        twig_split = twigglet.split('@')
        
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
                    if (tsp_i != 0 and tsp in mtwig) or mtwig[0]==tsp:
                    #~ if (tsp_i != 0 and [fnmatch(mtwig_spi,tsp) for mtwig_spi in mtwig]) or fnmatch(mtwig[0],tsp):
                        # then find where we are, and then only keep stuff to the right
                        ind = mtwig.index(tsp)
                        mtwig = mtwig[ind+1:]
                    else:
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
        
    def _get_by_search(self, twig=None, all=False, ignore_errors=False, return_trunk_item=False, return_key='item', use_search=False, **kwargs):
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
        
        if twig is not None:
            trunk = self.trunk
            if use_search:
                matched_twigs = self._search_twigs(twig, **kwargs)
            else:
                matched_twigs = self._match_twigs(twig, **kwargs)
        else:
            trunk = self._filter_twigs_by_kwargs(**kwargs)
            matched_twigs = [ti['twig_full'] for ti in trunk]
        
        if len(matched_twigs) == 0:
            if ignore_errors:
                return None
            raise ValueError("no match found matching the criteria")
        elif all == False and ignore_errors == False and len(matched_twigs) > 1:
            results = ', '.join(matched_twigs)
            raise ValueError("more than one result was found matching the criteria: {}".format(results))
        else:
            items = [ti if return_trunk_item else ti[return_key] for ti in trunk if ti['twig_full'] in matched_twigs]
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
        
    def _save_json(self, filename):
        """
        TESTING
        """
        
        this_container = self.__class__.__name__
        #~ dump_dict = {ti['twig']: 'value': ti['item'].get_value() for ti in self.trunk if ti['container']==this_container and ti['kind']=='Parameter'}

        dump_dict = {}
        
        #~ dump_dict['_system_hierarchy'] = 'only binaries currently supported'
        #~ dump_dict['_system_hierarchy'] = {'kind': 'BodyBag', 'children': [{'kind': 'BinaryRocheStar', 'label': 'primary'}, {'kind': 'BinaryRocheStar', 'label': 'secondary'}], 'label': 'new system'}

        if hasattr(self, 'get_system'):
            dump_dict['_system_structure'] = _dumps_system_structure(self.get_system())

        # TODO will need to include 'hidden' information like refs and labels here as well
        
        for ti in self.trunk:
            if not ti['hidden'] and ti['container']==this_container and ti['kind']=='Parameter':
                item = ti['item']
                info = {}
                
                info['value'] = item.to_str()
                
                #~ if hasattr(item, 'get_unit') and item.has_unit():
                    #~ info['_unit (read-only)'] = item.get_unit()
                
                if hasattr(item, 'adjust') and item.adjust:
                    info['adjust'] = item.adjust
                    
                if item.has_prior():
                    info['prior'] = 'has prior, but export not yet supported'
                    
                dump_dict[ti['twig']] = info

        f = open(filename, 'w')
        f.write(json.dumps(dump_dict, sort_keys=True, indent=4, separators=(',', ': ')))
        f.close()

    @rebuild_trunk
    def _load_json(self, filename):
        """
        TESTING
        """
        f = open(filename, 'r')
        load_dict = json.load(f)
        f.close()
        
        if hasattr(self, 'get_system'):
            self.set_system(_loads_system_structure(load_dict['_system_structure']))
            #print self.get_system().list(summary='full')
        
        for twig,info in load_dict.items():
            if twig[0]!='_':
                #~ print "self.set_value('{}', '{}')".format(twig, str(info['value']))
                item = self.get(twig, hidden=None)
                
                if 'value' in info:
                    item.set_value(str(info['value']))
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
        
        @param section: name of the section (key of self.sections)
        @type section: str
        @param ps: the new parameterset with label set
        @type ps: ParameterSet
        """
        if section not in self.sections.keys():
            self.sections[section] = []
        self.sections[section].append(ps)
    
    def list_twigs(self, full=True, **kwargs):
        """
        return a list of all available twigs
        """
        trunk = self._filter_twigs_by_kwargs(**kwargs)
        return [t['twig_full' if full else 'twig'] for t in trunk]

    def search(self, twig, **kwargs):
        """
        return a list of twigs matching a substring search
        
        @param twig: the search twig
        @type twig: str
        @return: list of twigs
        @rtype: list of strings
        """
        return self._search_twigs(twig, **kwargs)
        
    def match(self, twig, **kwargs):
        """
        return a list of matching twigs (with same method as used in
        get_value, get_parameter, get_prior, etc)
        
        @param twig: the search twig
        @type twig: str
        @return: list of twigs
        @rtype: list of strings
        """
        return self._match_twigs(twig, **kwargs)
    
    def get(self, twig=None, **kwargs):
        """
        search and retrieve any item without specifying its type
        
        @param twig: the search twig
        @type twig: str
        @return: matching item
        @rtype: varies
        """
        return self._get_by_search(twig, **kwargs)
        
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
        
    def get_ps_dict(self, twig):
        """
        retrieve a dictionary of ParameterSets in a list
        
        @param twig: the search twig
        @type twig: str
        @return: dictionary of ParameterSets
        @rtype: dict
        """
        return self._get_by_search(twig, kind="OrderedDict")

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
        Set the value of a Parameter
        
        @param twig: the search twig
        @type twig: str
        @param value: the value
        @type value: depends on Parameter
        @param unit: unit of value (if not default)
        @type unit: str or None
        """
        param = self.get_parameter(twig)
        
        # special care needs to be taken when setting labels and refs
        qualifier = param.get_qualifier()
        if qualifier == 'label':
            this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
            component = self._get_by_search(this_trunk['label'])
            component.set_label(value)
            self._build_trunk()
        
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
    
    def set_ps(self, twig, value):
        """
        Replace an existing ParameterSet.
        """
        # get all the info we can get
        this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
        
        # Make sure it is a ParameterSet
        if this_trunk['kind'] != 'ParameterSet':
           raise ValueError("Twig '{}' does not refer to a ParameterSet (it is a {})".format(twig, this_trunk['kind']))
        
        # actually, we're mainly interested in the path. And in that path, we
        # are only interested in the last Body
        bodies = [self.get_system()] + [thing for thing in this_trunk['path'] if isinstance(thing, universe.Body)]
        
        if not bodies:
            raise ValueError('Cannot assign ParameterSet to {} (did not find any Body)'.format(twig))
        
        # last check: we need to make sure that whatever we're setting already
        # exists. We cannot assign a new nonexistent PS to the Body (well, we
        # could, but we don't want to -- that's attach_ps responsibility)
        current_context = this_trunk['item'].get_context()
        given_context = value.get_context()
        if current_context == given_context:
            bodies[-1].set_params(value)
        else:
            raise ValueError("Twig '{}' refers to a ParameterSet of context '{}', but '{}' is given".format(twig, current_context, given_context))
        
        # And rebuild the trunk
        self._build_trunk()
        
    def attach_ps(self, twig, value):
        """
        Add a new ParameterSet.
        """
        # get all the info we can get
        this_trunk = self._get_by_search(twig=twig, return_trunk_item=True)
        
        # Make sure it is a Body
        if not isinstance(this_trunk['item'], universe.Body):
           raise ValueError("You can only attach ParameterSets to a Body ('{}' refers to a {})".format(twig, this_trunk['kind']))
        
        # last check: we need to make sure that whatever we're setting already
        # exists. We cannot assign a new nonexistent PS to the Body (well, we
        # could, but we don't want to -- that's attach_ps responsibility)
        given_context = value.get_context()
        try:
            this_trunk['item'].set_params(value, force=False)
        except ValueError:
            raise ValueError("ParameterSet '{}' at Body already exists. Please use set_ps to override it.".format(given_context))
        
        
        # And rebuild the trunk
        self._build_trunk()
    
    def get_adjust(self, twig):
        """
        retrieve whether a Parameter is marked to be adjusted
        
        @param twig: the search twig
        @type twig: str
        @return: adjust
        @rtype: bool
        """
        return self.get_parameter(twig).get_adjust()
        
    def set_adjust(self, twig, value=True):
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
        
    def set_prior_all(self, twig, **dist_kwags):
        """
        set the prior on all matching Parameters
        
        @param twig: the search twig
        @type twig: str
        @param **kwargs: necessary parameters for distribution
        @type **kwargs: varies
        """
        params = self._get_by_search(twig, kind='Parameter', all=True)
        
        for param in params:
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
        compute = self.get_compute(label)
        self.sections['compute'].remove(compute)
        
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
        fitting = self.get_fitting(label)
        self.sections['fitting'].remove(fitting)
        
def _dumps_system_structure(current_item):
    """
    dumps the system hierarchy into a dictionary that is json compatible
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

def _loads_system_structure(struc, c1label=None, c2label=None, top_level=True):
    """
    loads the system hierarchy from a json compatible dictionary
    """

    # if we're dealing with a Bag of some sort, then we need to create
    # the item while also building and attaching its children
    if 'children' in struc.keys(): # BodyBag, etc
        list_of_bodies = [_loads_system_structure(child_struc, c1label=struc['children'][0]['label'], c2label=struc['children'][1]['label'], top_level=False) for child_struc in struc['children']]
        this_item = getattr(universe, struc['kind'])(list_of_bodies, label=struc['label'])
    
    # if we're a component, then we can create the item and attach orbits from
    # the passed labels for c1label and c2label
    else: # BinaryRocheStar, etc
        this_item = getattr(universe, struc['kind'])(parameters.ParameterSet(context='component', label=struc['label']), orbit=parameters.ParameterSet(context='orbit', label=struc['label'], c1label=c1label, c2label=c2label))

    # top_level should only be True when this function is initially called
    # (any recursive calls will set it to False).  For the top-level item,
    # we need to create and attach a 'position' PS
    if top_level:
        this_item.params['position'] = parameters.ParameterSet(context='position')

    # and finally return the item, which should be the system for any
    # non-recursive calls
    return this_item
