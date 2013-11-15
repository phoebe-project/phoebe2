import os
import pickle
import ConfigParser
from server import Server
from phoebe.parameters import parameters

class Settings(object):
    def __init__(self):
        self.default_preferences = {'panel_params': True, 'panel_fitting': True,\
            'panel_datasets': True, 'panel_system': False,\
            'panel_versions': False, 'panel_python': False,\
            'pyinterp_tutsys': True, \
            'pyinterp_tutplots': True, 'pyinterp_tutsettings': True,\
            'pyinterp_thread_on': True, 'pyinterp_thread_off': False,\
            'pyinterp_startup_default': 'import phoebe\nfrom phoebe.frontend.bundle import Bundle, load\nfrom phoebe.parameters import parameters, create, tools\nfrom phoebe.io import parsers\nfrom phoebe.utils import utils\nfrom phoebe.frontend import usersettings\nsettings = usersettings.load()',
            'pyinterp_startup_custom': 'import numpy as np\nlogger = utils.get_basic_logger(clevel=\'WARNING\')', \
            'plugins': {'keplereb': False, 'example': False},\
            }
            
        self.preferences = {}
        self.servers = []
        self.compute_options = []
        self.fitting_options = []
        
        self.add_compute(label='Compute')
        self.add_compute(label='Preview')
        
        self.add_fitting(context='fitting:grid',label='grid')
        self.add_fitting(context='fitting:minuit',label='minuit')
        self.add_fitting(context='fitting:lmfit',label='lmfit')
        self.add_fitting(context='fitting:lmfit:leastsq',label='lmfit:leastsq')
        self.add_fitting(context='fitting:lmfit:nelder',label='lmfit:nelder')
        self.add_fitting(context='fitting:emcee',label='emcee')
        self.add_fitting(context='fitting:pymc',label='pymc')
        
    def get_value(self,key):
        return self.preferences[key] if key in self.preferences.keys() else self.default_preferences[key]
        
    def set_value(self,key,value):
        self.preferences[key] = value
        
    #{Servers
        
    def get_server(self,label=None):
        """
        get a server by name
        
        @param label: name of the server
        @type label: str
        @return: server
        @rtype: Server
        """
        servers = {s.get_value('label'): s for s in self.servers}
        
        if label in servers.keys():
            return servers[label]
        return servers
        
    def add_server(self,label,mpi=None,server=None,server_dir=None,server_script=None,mount_dir=None):
        """
        add a new server
        
        @param label: name to refer to the server inside settings/bundle
        @type label: str
        @param mpi: the mpi options to use for this server
        @type mpi: ParameterSet with context 'mpi'
        @param server: the location of the server (used for ssh - so username@servername if necessary), or None if local
        @type server: str
        @param server_dir: directory on the server to copy files and run scripts
        @type server_dir: str
        @param server_script: location on the server of a script to run (ie. to setup a virtual environment) before running phoebe
        @type server_script: str
        @param mount_dir: local mounted location of server:server_dir, or None if local
        @type mount_dir: str or None
        """
        self.servers.append(Server(mpi,label,server,server_dir,server_script,mount_dir))
        
    def remove_server(self,label):
        """
        remove a server by name
        
        @param label: name of the server
        @type label: str
        """
        
        return self.servers.pop(self.servers.index(self.get_server(label)))
        
    #}
    #{ Compute
    
    def add_compute(self,compute=None,**kwargs):
        """
        Add a new compute ParameterSet
        
        @param compute: compute ParameterSet
        @type compute:  None or ParameterSet
        """
        if compute is None:
            compute = parameters.ParameterSet(context='compute')
        for k,v in kwargs.items():
            compute.set_value(k,v)
            
        self.compute_options.append(compute)

    def get_compute(self,label=None,copy=False):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        # create a dictionary with key as the label
        compute_options = {co.get_value('label'): co for co in self.compute_options}
        
        if label is None:
            return compute_options
        elif label in compute_options.keys():
            return compute_options[label]
        else:
            return None
        
    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        if label is None:    return None
        return self.compute_options.pop(self.compute_options.index(self.get_compute(label))) 
        
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
        context = kwargs.pop('context') if 'context' in kwargs.keys() else 'fitting:pymc'
        if fitting is None:
            fitting = parameters.ParameterSet(context=context)
        for k,v in kwargs.items():
            fitting.set_value(k,v)
            
        self.fitting_options.append(fitting)

    def get_fitting(self,label=None):
        """
        Get a fitting ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        # create a dictionary with key as the label
        fitting_options = {fo.get_value('label'): fo for fo in self.fitting_options}
        
        if label is None:
            return fitting_options
        elif label in fitting_options.keys():
            return fitting_options[label]
        else:
            return None
        
    def remove_fitting(self,label):
        """
        Remove a given fitting ParameterSet
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        if label is None:    return None
        return self.fitting_options.pop(self.fitting_options.index(self.get_fitting(label)))     
        
    #}

    def save(self,filename=None):
        """
        save usersettings to a file
        
        @param filename: save to a given filename, or None for default (~/.phoebe/preferences)
        @type filename: str
        """
        
        if filename is None:
            filename = os.path.expanduser('~/.phoebe/preferences')

        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()
        
def load(filename=None):
    """
    load usersettings from a file
    
    @param filename: load from a given filename, or None for default (~/.phoebe/preferences)
    @type filename: str
    """
    if filename is None:
        filename = os.path.expanduser('~/.phoebe/preferences')

    if not os.path.isfile(filename):
        return Settings()

    ff = open(filename,'r')
    settings = pickle.load(ff)
    ff.close()
    
    return settings


def load_configfile(name, section=None, pararameter_sets=None):
    """
    Load settings from a configuration file.
    
    ``name`` is the name of the configuration file (e.g. `servers` for
    `servers.cfg`).
    
    If you don't give a section, you will get the configparser object back.
    
    If you give a section name, it will return a dictionary with the contents
    of that section. Both the key and value will be strings.
    
    If you give a section and a list of parameterSets, then the contents will 
    be smart-parsed to the parameterSets. If a key is not in any of the
    parametersets, it will be ignored. If a key is in both, in will be stored
    in both. If it is in only one of them, then of course it will be put in 
    that one. In this case, nothing will be returned, but the parameterSets
    will have changed!
    
    We will add something that does a smart parsing if you give parameter_sets:
    we want 'true' and 'True' and '1' all to be evaluated to True. ConfigParser
    can do that, it should only see if the parameterSet needs a boolean or float
    or whatever.
    """
    basedir = os.path.abspath(os.path.dirname(__file__))
    settings = ConfigParser.ConfigParser()
    settings.optionxform = str # make sure the options are case sensitive
    
    with open(os.path.join(basedir,'{}.cfg'.format(name))) as config_file:
        settings.readfp(config_file)
    
    # If nothing else is given, we just return the parser object
    if section is None:
        return settings
    
    # If no parameterSets are given, a dictionary of the section is returned
    if parameter_sets is None:
        items = settings.items(server)
        items_dict = {key:value for key,value in items}
        return items_dict
    
    # Else, we have parameterSets!
    for key, value in settings.items(server):
        for ps in parameter_sets:
            if key in ps:
                ps[key] = value
    
