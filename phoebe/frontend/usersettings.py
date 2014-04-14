import os
import pickle
import ConfigParser
from collections import OrderedDict
from fnmatch import fnmatch
import copy
from server import Server
from phoebe.utils import utils
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
    
    def _loop_through_container(self):
        return_items = []
        for section_name,section in self.sections.items():
            for item in section:
                ri = self._get_info_from_item(item,section=section_name)
                return_items.append(ri)
                    
                if ri['kind']=='BodyBag':
                    return_items += self._loop_through_system(item, section_name=section_name)
                elif ri['kind']=='ParameterSet': # these should be coming from the sections
                    return_items += self._loop_through_ps(item, section_name=section_name, label=ri['label'])
        
        return return_items
        
    def _loop_through_ps(self, ps, section_name, label):
        return_items = []
        
        for qualifier in ps:
            item = ps.get_parameter(qualifier)
            ri = self._get_info_from_item(item, section=section_name, context=ps.get_context(), label=label)
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
            if ri is not None and ri['context']!='syn' and ri['qualifier'] not in ['ref','label'] and (ri['unique_label'] is None or ri['unique_label'] not in [r['unique_label'] for r in return_items]):
                return_items.append(ri)
            
        return return_items
        
    def _get_info_from_item(self, item, path=None, section=None, container=None, context=None, label=None):
        container = self.__class__.__name__
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
        print [qualifier,label,context,component,section_twig,container]
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
        
        for tsp in twig_split:
            remove = []
            for mtwig_i, mtwig in enumerate(matching_twigs):
                if tsp in mtwig:
                    # then find where are only keep stuff to the right
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
            raise ValueError("no parameter found matching the criteria")
        elif all == False and ignore_errors == False and len(matched_twigs) > 1:
            results = ', '.join(matched_twigs)
            raise ValueError("more than one parameter was found matching the criteria: {}".format(results))
        else:
            items = [ti if return_trunk_item else ti['item'] for ti in trunk if ti['twig_full'] in matched_twigs]
            if all:
                return items
            else:
                return items[0]
                
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
                
    def _get_by_section(self, label=None, section=None, all=False, ignore_errors=False, return_trunk_item=False):
        twig = section if label is None else "{}@{}".format(label,section) 
        #~ twig = section if label is None else label
        return self._get_by_search(twig,section=section,kind='ParameterSet',
                        all=all, ignore_errors=ignore_errors, 
                        return_trunk_item=return_trunk_item)
        
        
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
        return self._get_by_search(twig, kind=None)
        
    def get_all(self, twig):
        return self._get_by_search(twig, kind=None, all=True)
        
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
        


class Settings(Container):
    """
    Class representing settings for phoebe on a particular machine.
    
    These settings are loaded each time a bundle is loaded or initialized.
    Anything stored here is machine/user dependent so that you can load
    a bundle sent to you by someone else and still have access to your
    preferences on your machine.
    
    Included in these settings are:
    
        1. servers::
            parameters that define options for running computations, either
            locally or on a remote server, and with or without mpi
            
        2. compute::
            default options for different compute options.  These compute
            options will be available from any bundle you load and can be
            overridden at the bundle level.
            
        3. fitting::
            default options for different fitting options.  These fitting
            options will be available from any bundle you load and can be
            overridden at the bundle level.
            
        4. logger::
            options for the error logger.  This logger will be automatically
            setup whenever the settings are loaded.  You can override this
            by initializing your own logger afterwards, or by changing the
            settings and calling settings.apply_logger().
            
        4. gui::
            options for the default layout of the gui.
            
    Each of these 'sections' of the usersettings will be stored in its
    own .cfg ascii file, which can either be edited manually or through
    the interface in this class.  To write all the settings in the class
    to these files, call settings.save()
    """
    def __init__(self, basedir='~/.phoebe'):
        """
        Initialize settings and attempt to load from config files
        
        @param basedir: base directory where config files are located
        @type basedir: str
        """
        super(Settings, self).__init__()
        
        self.basedir = os.path.expanduser(basedir)
        
        # now let's load the settings from the .cfg files
        self.load()

        
    def __str__(self):
        """
        return a string representation
        """
        return self.to_string()

    def to_string(self):
        """
        """
        txt = ""
        for section in self.sections.keys():
            if isinstance(self.sections[section],list):
                for ps in self._get_by_section(section=section, all=True):
                    label = ps.get_value('label')
                    if ps is not None:
                        txt += "\n============ {}:{} ============\n".format(section,label)
                        txt += ps.to_string()
            else:
                ps = self.sections[section]
                if ps is not None:
                    txt += "\n============ {} ============\n".format(section)
                    txt += self.sections[section].to_string()
                
        return txt
        
                
    #{ General Parameters
    
    def get_logger(self):
        """
        retrieve the logger options ParameterSet
        
        @return: logger ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_search('logger', section='logger', ignore_errors=True)
        
    def apply_logger(self):
        """
        initialize the logger defined by the logger settings.
        
        If logger settings were changed since the usersettings were loaded,
        this must be called to apply them.
        """
        lps = self.get_logger()
        if lps is not None:
            logger = utils.get_basic_logger(style=lps.get_value('style'),
                                    clevel=lps.get_value('clevel'),
                                    flevel=lps.get_value('flevel'),
                                    filename=None if lps.get_value('filename')=='None' else lps.get_value('filename'),
                                    filemode=lps.get_value('filemode'))
        else:
            logger = utils.get_basic_logger()
        
    def get_gui(self):
        """
        retrive the gui options ParameterSet
        
        @return: gui ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section('default_gui',"gui")
        
    #}
        
       
    #{ Servers
    def add_server(self,label,mpi=None,**kwargs):
        """
        add a new server
        
        @param label: name to refer to the server inside settings/bundle
        @type label: str
        @param mpi: the mpi options to use for this server
        @type mpi: ParameterSet with context 'mpi'
        """
        self.sections['server'].append(Server(label,mpi,**kwargs))
        
    def get_server(self,label=None):
        """
        get a server by name
        
        @param label: name of the server
        @type label: str
        @return: server
        @rtype: Server
        """
        return self._get_by_section(label,"server")
        
    def remove_server(self,label):
        """
        remove a server by name
        
        @param label: name of the server
        @type label: str
        """
        return self._remove_from_section('server',label)
        
    #}
    #{ Compute
    
    def add_compute(self, compute=None, **kwargs):
        """
        Add a new compute ParameterSet
        
        @param compute: compute ParameterSet
        @type compute:  None or ParameterSet
        """
        if compute is None:
            # Take defaults from backend
            compute = parameters.ParameterSet(context='compute')
            
        for k,v in kwargs.items():
            compute.set_value(k,v)
            
        self.sections['compute'].append(compute)

    def get_compute(self,label=None,all=False,ignore_errors=False):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section(lablel,"compute")
        
    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        return self._remove_from_section('compute',label)

        
    #}
    #{ Fitting

    def add_fitting(self,fitting=None,**kwargs):
        """
        Add a new fitting ParameterSet
        
        @param fitting: fitting ParameterSet
        @type fitting:  None, or ParameterSet
        @param copy: whether to return deepcopy
        @type copy: bool
        """
        context = kwargs.pop('context', 'fitting:pymc')
        
        if fitting is None:
            fitting = parameters.ParameterSet(context=context)
            # here override from defaults in fitting.cfg
            
        for k,v in kwargs.items():
            fitting.set_value(k,v)
            
        self.sections['fitting'].append(fitting)

    def get_fitting(self,label=None):
        """
        Get a fitting ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_section(label,"fitting")
        
    def remove_fitting(self,label):
        """
        Remove a given fitting ParameterSet
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        return self._remove_from_section('fitting',label)
        
    #}
    #{ Loading and saving

        
    def save(self, basedir=None):
        """
        save all settings in the class to .cfg files in basedir
        
        @param basedir: base directory, or None to save to initialized basedir
        @type basedir: str or None
        """
        for section in self.sections.keys():
            self.dump_cfg(section, basedir)
        
    def dump_cfg(self, section, basedir=None):
        """
        save the settings of a single section to its .cfg files in basedir
        
        to save all sections, use save()
        
        @param section: name of the section ('server','compute',etc)
        @type section: str
        @param basedir: base directory, or None to save to initialized basedir
        @type basedir: str or None
        """
        if basedir is None:
            basedir = self.basedir
        else:
            basedir = os.path.expanduser(basedir)
            
        config = ConfigParser.ConfigParser()
        
        for ps in self._get_by_section(section=section, all=True):
            label = ps.get_value('label')
            # here label is the ConfigParser 'section'
            if section == 'server':
                for subsection in ['server','mpi']:
                    if ps.settings[subsection] is not None: #mpi might be None
                        self._add_to_config(config,section,'{}@{}'.format(subsection,ps.settings['server'].get_value('label')),ps.settings[subsection])
            else:
                self._add_to_config(config,section,label,ps)
                
        with open(os.path.join(basedir,'{}.cfg'.format(section)), 'wb') as configfile:
            config.write(configfile)
            
    def _add_to_config(self, config, section, label, ps):
        """
        add a new 'section' (label) to the ConfigParser class
        """
        # here label is the ConfigParser 'section'
        config.add_section(label)
        config.set(label,'context',ps.context)
        for key,value in ps.items():
            if not (section=='fitting' and key=='feedback'):
                # don't want to store feedback info in usersettings
                # and we can't anyways since its a dictionary and will fail to parse
                config.set(label,key,value)
                
    def load(self, basedir=None):
        """
        load all settings in the class from .cfg files in basedir.
        
        Initializing the settings class automatically calls this, there 
        is no need to manually call load() unless manual changes have been
        made to the .cfg files.
        
        @param basedir: base directory, or None to save to initialized basedir
        @type basedir: str or None
        """
        if basedir is None:
            basedir = self.basedir
        else:
            basedir = os.path.expanduser(basedir)
            
        # here we'll initialize all the sections and their defaults
        self.sections = OrderedDict()

        # for each of the sections we will try to load from the cfg file
        # if this fails or the file does not exist, then we will apply
        # basic defaults for that section
        # This means that any new-user defaults need to be set in this
        # function.
        # If you want to roll back any of your preferences to defaults,
        # simply delete the .cfg file and reload usersettings
        
        self.sections['server'] = []
        if not self.load_cfg('server', basedir):
            pass
        
        self.sections['compute'] = []
        if not self.load_cfg('compute', basedir):
            self.add_compute(label='preview',refl=False,heating=False,eclipse_alg='binary',subdiv_num=1)
            self.add_compute(label='detailed',ltt=True,eclipse_alg='binary',beaming_alg='local')
            
        self.sections['fitting'] = []
        if not self.load_cfg('fitting', basedir):
            self.add_fitting(context='fitting:grid',label='grid')
            self.add_fitting(context='fitting:minuit',label='minuit')
            self.add_fitting(context='fitting:lmfit',label='lmfit')
            self.add_fitting(context='fitting:lmfit:leastsq',label='lmfit:leastsq')
            self.add_fitting(context='fitting:lmfit:nelder',label='lmfit:nelder')
            self.add_fitting(context='fitting:emcee',label='emcee')
            self.add_fitting(context='fitting:pymc',label='pymc')
        
        self.sections['gui'] = [] # currently assumes only 1 entry
        if not self.load_cfg('gui', basedir):
            self._add_to_section('gui', parameters.ParameterSet(context='gui'))
            
        self.sections['logger'] = [] # currently assumes only 1 entry
        if not self.load_cfg('logger', basedir):
            self._add_to_section('logger', parameters.ParameterSet(context='logger'))
            
        # now apply the logger settings
        # if logger settings are changed, either reload the usersettings
        # or call apply_logger()
        self.apply_logger()
        
        self._build_trunk()
    
    def load_cfg(self, section, basedir=None):
        """
        load the settings of a single section to its .cfg files in basedir
        
        to load the settings of all sections, simply initialize a new settings class
        or call load()
        
        @param section: name of the section ('server','compute',etc)
        @type section: str
        @param basedir: base directory, or None to save to initialized basedir
        @type basedir: str or None
        """
        if basedir is None:
            basedir = self.basedir
        else:
            basedir = os.path.expanduser(basedir)

        # Build the filename and check if it exists. 
        # If not, exit nicely and return False
        config_filename = os.path.join(basedir,'{}.cfg'.format(section))
        if not os.path.isfile(config_filename):
            return False
            
        config = ConfigParser.ConfigParser()
        config.optionxform = str # make sure the options are case sensitive
            
        with open(config_filename) as config_file:
            config.readfp(config_file)
        
        for label in config.sections():
            # ConfigParser 'section' is our label
            items = config.items(label)
            items_dict = {key:value for key,value in items}
            
            if section=='server':
                # context already covered by context, so its a bit ambiguous
                label = label.split('@')[1]
                #~ items_dict['label'] = label
                context = items_dict['context']
            
            # now let's initialize the PS based on context and label
            ps = parameters.ParameterSet(context=items_dict.pop('context'))
            for key,value in items_dict.items():
                # set remaining values in PS
                ps.set_value(key, value)
            
            # and now add the ps to the section in usersettings
            if section=='server':
                # then we need to retrieve and add to a server class or 
                # create a new one
                server = self.get_server(label)
                if server is None:
                    server = Server()
                    server.settings[context] = ps
                    self._add_to_section(section, server)
                else:
                    server.settings[context] = ps
                
            else:
                self._add_to_section(section, ps)
            
        return True
        
    #}
