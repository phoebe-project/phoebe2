import os
import pickle
#~ from client import Server

class Settings(object):
    def __init__(self):
        self.default_preferences = {'panel_params': True, 'panel_fitting': True,\
            'panel_datasets': True, 'panel_system': False,\
            'panel_versions': False, 'panel_python': False,\
            'pyinterp_enabled': True, 'pyinterp_tutsys': True, \
            'pyinterp_tutplots': True, 'pyinterp_tutsettings': True,\
            'pyinterp_thread_on': True, 'pyinterp_thread_off': False,\
            'pyinterp_startup_default': 'import phoebe\nfrom phoebe.backend.bundle import Bundle, load\nfrom phoebe.parameters import parameters, create, tools\nfrom phoebe.io import parsers\nfrom phoebe.utils import utils\nfrom phoebe.frontend import usersettings\nsettings = usersettings.load()',
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
        
    def get_server(self,servername):
        return self.servers[servername]
        
    def add_server(self,servername,**kwargs):
        #~ self.servers[servername] = Server(kwargs)
        pass

    def save(self,filename=None):
        
        if filename is None:
            filename = os.path.expanduser('~/.phoebe/preferences')

        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()
        
def load(filename=None):
    if filename is None:
        filename = os.path.expanduser('~/.phoebe/preferences')

    if not os.path.isfile(filename):
        return Settings()

    ff = open(filename,'r')
    settings = pickle.load(ff)
    ff.close()
    
    return settings
