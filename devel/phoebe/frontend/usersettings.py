import os
import pickle
import ConfigParser
from server import Server

class Settings(object):
    def __init__(self):
        self.default_preferences = {'panel_params': True, 'panel_fitting': True,\
            'panel_datasets': True, 'panel_system': False,\
            'panel_versions': False, 'panel_python': False,\
            'pyinterp_enabled': True, 'pyinterp_tutsys': True, \
            'pyinterp_tutplots': True, 'pyinterp_tutsettings': True,\
            'pyinterp_thread_on': True, 'pyinterp_thread_off': False,\
            'pyinterp_startup_default': 'import phoebe\nfrom phoebe.frontend.bundle import Bundle, load\nfrom phoebe.parameters import parameters, create, tools\nfrom phoebe.io import parsers\nfrom phoebe.utils import utils\nfrom phoebe.frontend import usersettings\nsettings = usersettings.load()',
            'pyinterp_startup_custom': 'import numpy as np\nlogger = utils.get_basic_logger(clevel=\'WARNING\')', \
            'plugins': {'keplereb': False, 'example': False},\
            
            'mpi_np': 4, 'mpi_byslot': True,\
            
            'use_server_on_compute': False,\
            'use_server_on_preview': False,\
            'use_server_on_fitting': False, 
            }
            
        self.preferences = {}
        self.servers = {}
        
    def get_setting(self,key):
        return self.preferences[key] if key in self.preferences.keys() else self.default_preferences[key]
        
    def set_setting(self,key,value):
        self.preferences[key] = value
        
    def get_server(self,label=None):
        return self.servers[label]
        
    def add_server(self,label,mpi,server=None,server_dir=None,server_script=None,mount_dir=None):
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
        self.servers[label] = Server(mpi,label,server,server_dir,server_script,mount_dir)

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


def load_from_configfile_tryout(server):
    """
    Load settings from a configuration file.
    
    Example usage:
    
        >>> mpi, serversettings = load_from_configfile_tryout('clusty.ast.villanova.edu')
    """
    import phoebe
    basedir = os.path.abspath(os.path.dirname(__file__))
    settings = ConfigParser.ConfigParser()
    settings.optionxform = str # make sure the options are case sensitive
    
    with open(os.path.join(basedir,'servers.cfg')) as config_file:
        settings.readfp(config_file)
    
    items = settings.items(server)
    items_dict = {key:value for key,value in items}
    context = items_dict.pop('type')
    mpi = phoebe.PS(context=context)
    
    for key in items_dict.keys():
        if key in mpi:
            mpi[key] = items_dict.pop(key)
    
    return mpi, items_dict
    
