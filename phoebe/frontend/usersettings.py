import os
import pickle
try:
    import ConfigParser
except ImportError: # for Python3
    import configparser as ConfigParser
from collections import OrderedDict
import copy
import readline
from phoebe.frontend.server import Server
from phoebe.utils import utils
from phoebe.parameters import parameters
from phoebe.backend import universe
from phoebe.frontend.common import Container, rebuild_trunk
from phoebe.frontend import phcompleter

class Settings(Container):
    """
    Class representing settings for phoebe on a particular machine.
    
    [FUTURE]
    
    These settings are loaded each time a bundle is loaded or initialized.
    Anything stored here is machine/user dependent so that you can load
    a bundle sent to you by someone else and still have access to your
    preferences on your machine.
    
    Included in these settings are:
    
        1. servers::
            (not yet supported)
            parameters that define options for running computations, either
            locally or on a remote server, and with or without mpi
            
        2. compute::
            default options for different compute options.  These compute
            options will be available from any bundle you load and can be
            overridden at the bundle level.
            
        3. fitting::
            (not yet supported)
            default options for different fitting options.  These fitting
            options will be available from any bundle you load and can be
            overridden at the bundle level.
            
        4. logger::
            options for the error logger.  This logger will be automatically
            setup whenever the settings are loaded.  You can override this
            by initializing your own logger afterwards, or by changing the
            settings and calling settings.restart_logger().
            
        4. gui::
            (not yet supported)
            options for the default layout of the gui.
            
    Each of these 'sections' of the usersettings will be stored in its
    own .cfg ascii file, which can either be edited manually or through
    the interface in this class.  To write all the settings in the class
    to these files, call settings.save()
    """
    @rebuild_trunk
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
        
        # set tab completer
        readline.set_completer(phcompleter.Completer().complete)
        readline.set_completer_delims(' \t\n`~!#$%^&*)-=+]{}\\|;:,<>/?')
        readline.parse_and_bind("tab: complete")

        
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
    
    def get_logger(self, label='default_logger'):
        """
        retrieve the logger options ParameterSet
        
        @return: logger ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_search(label, section='logger', kind='ParameterSet', ignore_errors=True)
        
    def restart_logger(self, label='default_logger', **kwargs):
        """
        [FUTURE]
        
        restart the logger
        
        any additional arguments passed will temporarily override the settings
        in the stored logger.  These can include style, clevel, flevel, filename, filemode
        
        @param label: the label of the logger (will default if not provided)
        @type label: str
        """
        lps = self.get_logger(label).copy()
        
        # now temporarily override with any values passed through kwargs    
        for k,v in kwargs.items():
            #if k in options.keys(): # otherwise nonexisting kwargs can be given
            lps.set_value(k,v)
            
        logger = utils.get_basic_logger(style=lps.get_value('style'),
                                    clevel=lps.get_value('clevel'),
                                    flevel=lps.get_value('flevel'),
                                    filename=None if lps.get_value('filename')=='None' else lps.get_value('filename'),
                                    filemode=lps.get_value('filemode'))
        
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
    
    @rebuild_trunk
    def remove_server(self,label):
        """
        remove a server by name
        
        @param label: name of the server
        @type label: str
        """
        return self._remove_from_section('server',label)
        
    #}
    #{ Compute
    
    @rebuild_trunk
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
        return self._get_by_section(label,"compute")
    
    @rebuild_trunk
    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        return self._remove_from_section('compute',label)

        
    #}
    #{ Fitting

    @rebuild_trunk
    def add_fitting(self,fitting=None,**kwargs):
        """
        Add a new fitting ParameterSet
        
        @param fitting: fitting ParameterSet
        @type fitting:  None, or ParameterSet
        @param copy: whether to return deepcopy
        @type copy: bool
        """
        context = kwargs.pop('context', 'fitting:emcee')
        
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
    
    @rebuild_trunk
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
        devel = False
        
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
        
        if devel:
            self.sections['server'] = []
            if not self.load_cfg('server', basedir):
                pass
        
        self.sections['compute'] = []
        if not self.load_cfg('compute', basedir):
            self.add_compute(label='preview',refl=False,heating=False,eclipse_alg='binary',subdiv_num=1)
            self.add_compute(label='detailed',ltt=True,eclipse_alg='binary',boosting_alg='local')
        
        #~ if devel:
        if True:
            self.sections['fitting'] = []
            if not self.load_cfg('fitting', basedir):
                self.add_fitting(context='fitting:lmfit', label='lmfit', computelabel='preview')
                self.add_fitting(context='fitting:emcee', label='emcee', computelabel='preview')

        if devel:
            self.sections['gui'] = [] # currently assumes only 1 entry
            if not self.load_cfg('gui', basedir):
                self._add_to_section('gui', parameters.ParameterSet(context='gui'))
            
        self.sections['logger'] = [] # currently assumes only 1 entry
        if not self.load_cfg('logger', basedir):
            self._add_to_section('logger', parameters.ParameterSet(context='logger'))
            
        # now apply the logger settings
        # if logger settings are changed, either reload the usersettings
        # or call restart_logger()
        self.restart_logger()
        
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
