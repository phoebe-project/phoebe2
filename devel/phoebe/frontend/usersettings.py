import os
import pickle
import ConfigParser
from server import Server
from phoebe.utils import utils
from phoebe.parameters import parameters
from phoebe.utils.utils import get_basic_logger

class Settings(object):
    def __init__(self):
        
        # settings consists of different sections - each of which is
        # either a parameterset or a list of parametersets
        
        # here we'll initialize all the sections and their defaults
        self.settings = {}
        
        self.settings['servers'] = []
        
        self.settings['compute'] = []
        self.add_compute(label='Compute')
        self.add_compute(label='Preview')
        
        self.settings['fitting'] = []
        self.add_fitting(context='fitting:grid',label='grid')
        self.add_fitting(context='fitting:minuit',label='minuit')
        self.add_fitting(context='fitting:lmfit',label='lmfit')
        self.add_fitting(context='fitting:lmfit:leastsq',label='lmfit:leastsq')
        self.add_fitting(context='fitting:lmfit:nelder',label='lmfit:nelder')
        self.add_fitting(context='fitting:emcee',label='emcee')
        self.add_fitting(context='fitting:pymc',label='pymc')
        
        self.settings['gui'] = parameters.ParameterSet(context='gui')
        
        self.settings['logger'] = parameters.ParameterSet(context='logger')
        
    #~ def __str__(self):
        #~ pass
        
    def _get_from_list(self,listname,label):
        """
        retrieve a parameterset from a list by label
        """
        items = {ps.get_value('label'): ps for ps in self.settings[listname]}
        
        if label is None:
            return items
        elif label in items:
            return items[label]
        else:
            return None
            
    def _remove_from_list(self,listname,label):
        """
        remove a parameterset from a list by label
        """
        if label is None:    return None
        return self.settings[listname].pop(self.settings[listname].index(self._get_from_list(listname,label))) 
        
    #{ General Parameters
        
    def get_ps(self,section=None):
        """
        get a parameterset by section
        
        @param section: setting section (gui, logger, etc)
        @type section: str
        @return: parameterset
        @rtype: ParameterSet
        """
        if section is None:
            return self.settings
        return self.settings[section]
        
    def get_parameter(self,qualifier,section=None):
        """
        Retrieve a parameter from the settings
        If section is not provided and there are more than one object in 
        settings containing a parameter with the same name, this will
        return an error and ask you to provide a valid section
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param section: setting section (gui, logger, etc)
        @type objref: str
        @return: Parameter corresponding to the qualifier
        @rtype: Parameter
        """
        if section is None:
            categories = [c for c in self.settings.keys() if not isinstance(self.settings[c],list)]
        else:
            categories = [section]
        
        return_params = []
        return_categories = []
        
        for section in categories:
            ps = self.get_ps(section)
            if qualifier in ps.keys():
                return_params.append(ps.get_parameter(qualifier))
                return_categories.append(section)
                
        if len(return_params) > 1:
            raise ValueError("parameter '{}' is ambiguous, please provide one of the following for section:\n{}".format(qualifier,'\n'.join(["\t'%s'" % c for c in return_categories])))

        elif len(return_params)==0:
            raise ValueError("parameter '{}' was not found in any of the objects in the system".format(qualifier))
            
        return return_params[0]
        
    def get_value(self,qualifier,section=None):
        """
        Retrieve the value of a parameter from the settings
        If section is not provided and there are more than one object in 
        settings containing a parameter with the same name, this will
        return an error and ask you to provide a valid section
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param section: setting section (gui, logger, etc)
        @type objref: str
        @return: value of the Parameter corresponding to the qualifier
        @rtype: (depends on the parameter)
        """
        param = self.get_parameter(qualifier,section)
        return param.get_value()
                    
    def set_value(self,qualifier,value,section=None):
        """
        Retrieve the value of a parameter from the settings
        If section is not provided and there are more than one object in 
        settings containing a parameter with the same name, this will
        return an error and ask you to provide a valid section
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param value: the new value for the parameter
        @type value: (depends on parameter)
        @param section: setting section (gui, logger, etc)
        @type objref: str
        """
        param = self.get_parameter(qualifier,section)
        param.set_value(value)
        
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
        self.settings['servers'].append(Server(label,mpi,**kwargs))
        
    def get_server(self,label=None):
        """
        get a server by name
        
        @param label: name of the server
        @type label: str
        @return: server
        @rtype: Server
        """
        return self._get_from_list('servers',label)
        
    def remove_server(self,label):
        """
        remove a server by name
        
        @param label: name of the server
        @type label: str
        """
        return self._remove_from_list('servers',label)
        
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
            
        self.settings['compute'].append(compute)

    def get_compute(self,label=None):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_list('compute',label)

    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        return self._remove_from_list('compute',label)

        
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
            
        self.settings['fitting'].append(fitting)

    def get_fitting(self,label=None):
        """
        Get a fitting ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_list('fitting',label)
        
    def remove_fitting(self,label):
        """
        Remove a given fitting ParameterSet
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        return self._remove_from_list('fitting',label)
        
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
    
    # and apply necessary things (ie. logger)
    lps = settings.get_ps('logger')
    logger = get_basic_logger(style=lps.get_value('style'),clevel=lps.get_value('clevel'),
        flevel=lps.get_value('flevel'),filename=None if lps.get_value('filename')=='None' else lps.get_value('filename'),
        filemode=lps.get_value('filemode'))
    
    return settings


def load_configfile(name, section=None, parameter_sets=None, basedir=None):
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
    if basedir is None:
        basedir = os.path.abspath(os.path.dirname(__file__))
    else:
        basedir = os.path.expanduser(basedir)
    
    # Build the filename and check if it exists. If not, exit nicely and return
    # False
    config_filename = os.path.join(basedir,'{}.cfg'.format(name))
    if not os.path.isfile(config_filename):
        return False
    
    settings = ConfigParser.ConfigParser()
    settings.optionxform = str # make sure the options are case sensitive
    
    with open(config_filename) as config_file:
        settings.readfp(config_file)
    
    # If nothing else is given, we just return the parser object
    if section is None:
        return settings
    
    # If no parameterSets are given, a dictionary of the section is returned
    if parameter_sets is None:
        items = settings.items(section)
        items_dict = {key:value for key,value in items}
        return items_dict
    
    # Else, we have parameterSets!
    for key, value in settings.items(section):
        for ps in parameter_sets:
            if key in ps:
                ps[key] = value
    
    # In the case the preference file existed, we return True
    return True
    

def get_user_logger(name='default_logger'):
    """
    Set up a logger with properties loaded from the user preferences.
    """
    settings = load_configfile('logging', section=name, basedir='~/.phoebe/preferences')
    filename = settings['filename']
    if filename.lower() == 'none':
        filename = None
    logger = utils.get_basic_logger(style=settings['style'],
                                    clevel=settings['console_level'],
                                    flevel=settings['file_level'],
                                    filemode=settings['filemode'],
                                    filename=filename)
    return logger
