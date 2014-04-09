import os
import pickle
import ConfigParser
from collections import OrderedDict
from fnmatch import fnmatch
import copy
from server import Server
from phoebe.utils import utils
from phoebe.parameters import parameters

class Container(object):
    """
    Class that controlls accessing sections and parametersets
    
    This class in inherited by both the bundle and usersettings
    """
    
    def __init__(self):
        self.sections = OrderedDict()
        
    ## act like a dictionary
    def keys(self):
        return self.sections.keys()
        
    def values(self):
        return self.sections.values()
        
    def items(self):
        return self.sections.items()
        
    ## generic functions to get non-system parametersets
    def _return_from_dict(self,dictionary,all=False,ignore_errors=False):
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
            if len(dictionary)==0:
                if ignore_errors:
                    return None
                raise ValueError("no results found: set ignore_errors to return None")
                #~ raise ValueError('parameter {} with constraints "{}" nowhere found in system'.format(qualifier,"@".join(structure_info)))
                #~ return None
            elif len(dictionary)>1:
                if ignore_errors:
                    return dictionary.values()[0]
                raise ValueError("more than one result was returned from the search: either constrain search, set all=True, or ignore_errors=True")    
            else:
                return dictionary.values()[0]
        
    def _get_from_section_single(self,section):
        """
        retrieve a parameterset by section when only one is expected
        ie. system, gui, logger
        """
        items = self._get_from_section(section, search=None, search_by=None,
                                all=True, ignore_usersettings=True).values()
        # NOTE: usersettings will always be ignored since search_by == None
        if len(items) == 0:
            raise ValueError("ERROR: no {} attached".format(section))
            return None
        if len(items) > 1:
            raise ValueError("ERROR: more than 1 {} attached".format(section))
        return items[0]

                
    def _get_from_section(self,section,search=None,search_by='label',all=False,ignore_usersettings=False,ignore_errors=False):
        """
        retrieve a parameterset (or similar object) by section and label (optional)
        if the section is also in the defaults set by usersettings, 
        then those results will be included but overridden by those
        in the bundle
        
        this function should be called by any get_* function that gets an item 
        from one of the lists in self.sections
        
        @param section: name of the section (key of self.sections)
        @type section: str
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        @param all: whether to return a single item or all in a dictionary
        @type all: bool
        @param ignore_usersettings: whether to ignore defaults in usersettings (default: False)
        @type ignore_usersettings: bool
        """
        # We'll start with an empty dictionary, fill it, and then convert
        # to the requested format
        items = OrderedDict()
        
        #~ print "***", section, section in self.sections.keys(), ignore_usersettings, search_by
        
        # First we'll get the items from the bundle
        # If a duplicate is found in usersettings, the bundle version will trump it
        if section in self.sections.keys():
            for ps in self.sections[section]:
                #~ if search is None or ps.get_value(search_by)==search:
                if search is None or fnmatch(ps.get_value(search_by), search):
                    # if search_by is None then we want to return everything
                    # NOTE: in this case usersettings will be ignored
                    if search_by is not None:
                        try:
                            key = ps.get_value(search_by)
                        except AttributeError:
                            key = len(items)
                    else:
                        key = len(items)
                    
                    items[key] = ps

        if not ignore_usersettings and search_by is not None:
            # Now let's check the defaults in usersettings
            usersettings = self.get_usersettings().sections
            if section in usersettings.keys():
                for ps in usersettings[section]:
                    #~ if (search is None or ps.get_value(search_by)==search) and ps.get_value(search_by) not in items.keys():
                    if (search is None or fnmatch(ps.get_value(search_by), search)) and ps.get_value(search_by) not in items.keys():
                        # Then these defaults exist in the usersettings but 
                        # are not (yet) overridden by the bundle.
                        #
                        # In this case, we need to make a deepcopy and save it
                        # to the bundle (since it could be edited here).
                        # This is the version that will be returned in this 
                        # and any future retrieval attempts.
                        #
                        # In order to return to the usersettings default,
                        # the user needs to remove the bundle version, or 
                        # access directly from usersettings (bundle.get_usersettings().get_...).
                        #
                        # NOTE: in the case of things that have defaults in
                        # usersettings but not in bundle by default (ie logger, servers, etc)
                        # this will still create a new copy (for consistency)

                        psc = copy.deepcopy(ps)
                        items[psc.get_value(search_by)] = psc
                        # now we add the copy to the bundle
                        self._add_to_section(section,psc)
                    
        # and now return in the requested format
        return self._return_from_dict(items,all,ignore_errors)

    def _remove_from_section(self,section,search,search_by='label'):
        """
        remove a parameterset from by section and label
        
        this will not affect any defaults set in usersettings - so this
        function can be called to 'reset to user defaults'
        
        this function should be called by any remove_* function that gets an item 
        from one of the lists in self.sections
        
        @param section: name of the section (key of self.sections)
        @type section: str
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        """
        if search is None:    return None
        return self.sections[section].remove(self._get_from_section(section,search,search_by))
        
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
                for label,ps in self._get_from_section(section, all=True).items():
                    if ps is not None:
                        txt += "\n============ {}:{} ============\n".format(section,label)
                        txt += ps.to_string()
            else:
                ps = self.sections[section]
                if ps is not None:
                    txt += "\n============ {} ============\n".format(section)
                    txt += self.sections[section].to_string()
                
        return txt
        
    def _get_from_section(self,section,search=None,search_by='label',all=False,ignore_usersettings=False):
        return super(Settings, self)._get_from_section(section=section, search=search,
                            search_by=search_by, all=all,
                            ignore_usersettings=True)
                
    #{ General Parameters
    
    def get_logger(self):
        """
        retrieve the logger options ParameterSet
        
        @return: logger ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_section_single('logger')
        
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
        return self._get_from_section_single('gui')
        
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
        
    def get_server(self,label=None,all=False):
        """
        get a server by name
        
        @param label: name of the server
        @type label: str
        @return: server
        @rtype: Server
        """
        return self._get_from_section('server',label,all=all)
        
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

    def get_compute(self,label=None,all=False):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_section('compute',label,all=all)

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

    def get_fitting(self,label=None,all=False):
        """
        Get a fitting ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_section('fitting',label,all=all)
        
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
        
        #~ print "***", section
        for label,ps in self._get_from_section(section, all=True).items():
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
