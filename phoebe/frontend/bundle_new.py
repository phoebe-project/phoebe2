"""
Top level class of Phoebe.
"""
import pickle
import logging
import functools
import inspect
import numpy as np
from collections import OrderedDict
from datetime import datetime
from time import sleep
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image

from phoebe.utils import callbacks, utils, plotlib, coordinates, config
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.parameters import create
from phoebe.backend import fitting, observatory, plotting
from phoebe.backend import universe
from phoebe.io import parsers
from phoebe.dynamics import keplerorbit
from phoebe.frontend import usersettings
from phoebe.frontend.figures import Axes

logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())

def run_on_server(fctn):
    """
    Parse usersettings to determine whether to run a function locally
    or submit it to a server
        
    """
    @functools.wraps(fctn)
    def parse(bundle,*args,**kwargs):
        """
        """
        callargs = inspect.getcallargs(fctn,bundle,*args,**kwargs)
        dump = callargs.pop('self')
        callargstr = ','.join(["%s=%s" % (key, "\'%s\'" % callargs[key] if isinstance(callargs[key],str) else callargs[key]) for key in callargs.keys()])

        servername = kwargs['server'] if 'server' in kwargs.keys() else None

        is_server = kwargs.pop('is_server',False)

        if servername is not False and servername is not None and not is_server:
            server =  bundle.get_server(servername)
            if server.is_external():
                if server.check_status():
                    # prepare job
                    mount_dir = server.server_ps.get_value('mount_dir')
                    server_dir = server.server_ps.get_value('server_dir')
                    
                    logger.info('copying bundle to {}'.format(mount_dir))
                    timestr = str(datetime.now()).replace(' ','_')
                    in_f = '%s.bundle.in.phoebe' % timestr
                    out_f = '%s.bundle.out.phoebe' % timestr
                    script_f = '%s.py' % timestr
                    bundle.save(os.path.join(mount_dir,in_f),save_usersettings=True) # might have a problem here if threaded!!
                    
                    logger.info('creating script to run on {}'.format(servername))
                    f = open(os.path.join(mount_dir,script_f),'w')
                    f.write("#!/usr/bin/python\n")
                    f.write("try:\n")
                    f.write("\timport os\n")
                    f.write("\tfrom phoebe.frontend.bundle import load\n")
                    f.write("\tbundle = load('%s',load_usersettings=False)\n" % os.path.join(server_dir,in_f))
                    f.write("\tbundle.%s(%s,is_server=True)\n" % (fctn.func_name, callargstr))
                    f.write("\tbundle.save('%s')\n" % (os.path.join(server_dir,out_f)))
                    f.write("except:\n")
                    f.write("\tos.system(\"echo 'failed' > %s.status\")\n" % (script_f))
                    f.close()
                    
                    # create job and submit
                    logger.info('running script on {}'.format(servername))
                    server.run_script_external(script_f)
                    
                    # lock the bundle and store information about the job
                    # so it can be retrieved later
                    # the bundle returned from the server will not have this lock - so we won't have to manually reset it
                    bundle.lock = {'locked': True, 'server': servername, 'script': script_f, 'command': "bundle.%s(%s)\n" % (fctn.func_name, callargstr), 'files': [in_f, out_f, script_f], 'rfile': out_f}
                                        
                    return

                else:
                    logger.warning('{} server not available'.format(servername))
                    return

        # run locally by calling the original function
        return fctn(bundle, *args, **kwargs)
    
    return parse

class Bundle(object):
    """
    Class representing a collection of systems and stuff related to it.
    
    What is the Bundle?
    
    What main properties does it have? (rough sketch of structure)
        - a Body (can also be BodyBag), called :envvar:`system` in this context.
        - ...
        - ...
        
    How do you initialize it? Examples and/or refer to :py:func:`set_system`.
        
    What methods does it have?
    
    **Input/output**
    
    **Setting and getting system parameters**
    
    **Setting and getting computational parameters**
    
    **Setting and getting fit parameters**
    
    **Getting results**
    
    **History and GUI functionality**
    
    
    """
    def __init__(self,system=None):
        """
        Initialize a Bundle.

        For all the different possibilities to set a system, see :py:func:`Bundle.set_system`.
        """
        self.sections = OrderedDict()
        
        self.sections['system'] = [None] # only 1
        self.sections['compute'] = []
        self.sections['fitting'] = []
        self.sections['feedback'] = []
        self.sections['version'] = []
        self.sections['axes'] = []
        self.sections['meshview'] = [parameters.ParameterSet(context='plotting:mesh')] # only 1
        self.sections['orbitview'] = [parameters.ParameterSet(context='plotting:orbit')] # only 1
        
        self.select_time = None
        
        self.signals = {}
        self.attached_signals = []
        self.attached_signals_system = [] #these will be purged when making copies of the system and can be restored through set_system
        
        self.set_usersettings()
        
        self.lock = {'locked': False, 'server': '', 'script': '', 'command': '', 'files': [], 'rfile': None}

        self.filename = None
        
        self.settings = {}
        self.settings['add_version_on_compute'] = False
        self.settings['add_feedback_on_fitting'] = False
        self.settings['update_mesh_on_select_time'] = False
        
        self.set_system(system) # will handle all signals, etc
        
    ## string representation
    def __str__(self):
        return self.to_string()
        
    def to_string(self):
        txt = ""
        txt += "{} compute options\n".format(len(self.get_compute(return_type='list')))
        txt += "{} fitting options\n".format(len(self.get_fitting(return_type='list')))
        txt += "{} axes\n".format(len(self.get_axes(return_type='list')))
        txt += "============ System ============\n"
        txt += self.list()
        
        return txt
        
    ## act like a dictionary
    def keys(self):
        return self.sections.keys()
        
    def values(self):
        return self.sections.values()
        
    def items(self):
        return self.sections.items()
        
    ## generic functions to get non-system parametersets
    def _return_from_dict(self,dictionary,return_type):
        """
        this function takes a dictionary from a searching function (get_ps)
        along with the desired return_type ('single','dict','list') and
        returns in the correct format
        """
        if return_type=='dict':
            return dictionary
        elif return_type=='list':
            return dictionary.values()
        elif return_type=='single':
            if len(dictionary) > 1:
                raise ValueError("search resulted in more than one result: modify search or change return_type to 'list' or 'dict'")
            elif len(dictionary)==1: #then only one match
                return dictionary.values()[0]
            else: #then no results
                return None

    def _get_from_section(self,section,search=None,search_by='label',return_type='single'):
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
        @param return_type: 'single', 'list', 'dict'
        @type return_type: str
        """
        
        # We'll start with an empty dictionary, fill it, and then convert
        # to the requested format
        items = OrderedDict()
        
        # First we'll get the items from the bundle
        # If a duplicate is found in usersettings, the bundle version will trump it
        if section in self.sections.keys():
            for ps in self.sections[section]:
                if search is None or ps.get_value(search_by)==search:
                    items[ps.get_value(search_by)] = ps

        # Now let's check the defaults in usersettings
        usersettings = self.get_usersettings().settings
        if section in usersettings.keys():
            for ps in usersettings[section]:
                if (search is None or ps.get_value(search_by)==search) and search not in items.keys():
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
                    self._add_to_section(section,psc)
        
        # and now return in the requested format
        self._return_from_dict(items,return_type)

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
        if label is None:    return None
        #~ return self.sections[section].pop(self.sections[section].index(self._get_from_section(section,search,search_by))) 
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
        if section not in self.sections.keys()
            self.sections[section] = []
        self.sections[section].append(ps)
        
    #{ Settings
    def set_setting(self,key,value):
        """
        Set a bundle-level setting
        
        @param key: the name of the setting
        @type key: string
        @param value: the value of the setting
        @type value: string, float, boolean
        """
        self.settings[key] = value
        
    def get_setting(self,key):
        """
        Get the value for a bundle-level setting by name
        
        @param key: the name of the setting
        @type key: string
        """
        return self.settings[key]
        
    def set_usersettings(self,settings=None):
        """
        Load user settings into the bundle
        
        These settings are not saved with the bundle, but are removed and
        reloaded everytime the bundle is loaded or this function is called.
        
        @param settings: the settings (or none to load from default file)
        @type settings: string
        """
        if settings is None or isinstance(settings,str):
            settings = usersettings.load(settings)
        # else we assume settings is already the correct type
        self.usersettings = settings
        
    def get_usersettings(self,key=None):
        """
        Return the user settings class
        
        These settings are not saved with the bundle, but are removed and
        reloaded everytime the bundle is loaded or set_usersettings is called.
        
        @param key: name of the setting, or none to return the class
        @type key: string or None
        """
        if key is None:
            return self.usersettings
        else:
            return self.get_usersettings().get_value(key)
            
    def get_server(self,servername,return_type='list'):
        """
        Return a server by name
        
        Note that this is merely a shortcut to bundle.get_usersettings().get_server()
        The server settings are stored in the usersettings and are not kept with the bundle
        
        @param servername: name of the server
        @type servername: string
        """
        #~ return self.get_usersettings().get_server(servername)
        return self._get_from_section('servers',servername,return_type)
        
    #}    
    #{ System
    
    def set_system(self,system=None):
        """
        Change or set the system.
        
        Possibilities:
        
        1. If :envvar:`system` is a Body, then that body will be set as the system
        2. If :envvar:`system` is a string, the following options exist:
            - the string represents a Phoebe pickle file containing a Body; that
              one will be set
            - the string represents a Phoebe pickle file containing a Bundle;
              the system from that bundle will be set (to load a complete Bundle,
              use :py:func`load`
            - the string represents a Phoebe Legacy file, then it will be parsed
              to a system
            - the string represents a WD lcin file, then it will be parsed to
              a system
            - the string represents a system from the library (or spectral type),
              then the library will create the system

        @param system: the new system
        @type system: Body or str
        """
        # Possibly we initialized an empty Bundle
        if system is None:
            self.sections['system'] = [None]
            return None
        # Or a real system
        elif isinstance(system, universe.Body):
            self.sections['system'] = [system]
        elif isinstance(system, list) or isinstance(system, tuple):
            self.sections['system'] = [create.system(system)]
        # Or we could've given a filename
        else:
            
            # Try to guess the file type (if it is a file)
            if os.path.isfile(system):
                file_type, contents = guess_filetype(system)
            
                if file_type in ['phoebe_legacy', 'wd', 'pickle_body']:
                    system = contents
                elif file_type == 'pickle_bundle':
                    system = contents.get_system()
        
            # As a last resort, we pass it on to 'body_from_string' in the
            # create module:
            else:
                system = create.body_from_string(system)
            
            self.sections['system'] = [system]
            
        if self.get_system() is None:
            return
        
        # initialize uptodate
        self.get_uptodate() = False
        
        # connect signals
        self.attach_system_signals()
        
        # check to see if in versions, and if so set versions_curr_i
        versions_sys = [v.get_system() for v in self.get_version(return_type='list')]
        
        if system in versions_sys:
            i = versions_sys.index(system)
        else:
            i = None
        self.versions_curr_i = i
        
    def get_system(self):
        """
        Return the system.
        
        @return: the attached system
        @rtype: Body or BodyBag
        """
        return self._get_from_section('system')
                       
    def list(self,summary=None,*args):
        """
        List with indices all the ParameterSets that are available.
        Simply a shortcut to bundle.get_system().list(...)
        """
        return self.get_system().list(summary,*args)
        
    def clear_synthetic(self):
        """
        Clear all synthetic datasets
        Simply a shortcut to bundle.get_system().clear_synthetic()
        """
        return self.get_system().clear_synthetic()
        
    def set_time(self,time):
        """
        Set the time of a system, taking compute options into account.
        
        Shortcut to bundle.get_system().set_time() which insures fix_mesh
        is called first if any data was recently attached
        
        TODO: we should take advantage of the compute options here in the
              bundle. "set_time" doesn't compute reflection effects, we should
              do that here.
        
        @param time: time
        @type time: float
        """
        
        try:
            self.get_system().set_time(time)
        except:
            self.get_system().fix_mesh()
            self.get_system().set_time(time)
        
    def get_uptodate(self):
        """
        Check whether the synthetic model is uptodate
        
        If any parameters in the system have changed since the latest
        run_compute this will return False.
        If not, this will return the label of the latest compute options

        @return: uptodate
        @rtype: bool or str
        """
        return self.get_system().uptodate
            
    #}
    #{ Parameters/ParameterSets
    
       
    def get_ps(self, name=None, context=None, return_type='single'):
        pass
        
    def get_ps2(self, name, return_type='first'):
        """
        Retrieve a ParameterSet(s) from the system
        
        Undocumented
        """
        # Put the hierarchical structure in a list, that's easier to reference
        structure_info = []
        
        # Extract the info on the name and structure info. The name is always
        # first
        name = name.split('@')
        if len(name)>1:
            structure_info = name[1:]
        name = name[0]
                 
        # Reverse the structure info, that's easier to handle
        structure_info = structure_info[::-1]
        
        # First we'll loop through matching parametersets and gather all
        # parameters that match the qualifier
        found = []
        
        # You can always give top level system information if you desire
        if structure_info and structure_info[0] == self.system.get_label():
            start_index = 1
        else:
            start_index = 0
        
        # Now walk recursively over all parameters in the system, keeping track
        # of the history
        for path, val in self.system.walk_all(path_as_string=False):
            
            # Only if structure info is given, we need to do some work
            if structure_info:
                
                # We should look for the first structure information before
                # advancing to deeper levels; as long as we don't encounter
                # that reference/label/context, we can't look for the next one
                index = start_index
                
                # Look down the tree structure
                for level in path:
                    print type(level), type(val)
                    # but only if we still have structure information
                    if index < len(structure_info):
                        
                        # We don't now the name of this level yet, but we'll
                        # figure it out
                        name_of_this_level = None
                        
                        # If this level is a Body, we check it's label
                        if isinstance(level, universe.Body):
                            name_of_this_level = level.get_label()
                            
                        # If it is a ParameterSet, we'll try to match the label,
                        # reference or context
                        elif isinstance(level, parameters.ParameterSet):
                            if 'ref' in level:
                                name_of_this_level = level['ref']
                            elif 'label' in level:
                                name_of_this_level = level['label']
                            if name_of_this_level != structure_info[index]:
                                name_of_this_level = level.get_context()
                            
                            context = level.get_context()
                            ref = level['ref'] if 'ref' in level else None
                            label = level['label'] if 'label' in level else None
                            print '---',context,ref,label
                            
                        # The walk iterator could also give us 'lcsyn' or something
                        # the like, it apparently doesn't walk over these PSsets
                        # themselves -- but no problem, this works
                        elif isinstance(level, str):
                            name_of_this_level = level
                            
                        # We're on the right track and can advance to find the
                        # next label in the structure!
                        if name_of_this_level == structure_info[index]:
                            index += 1
                
                # Keep looking if we didn't exhaust all specifications yet.
                # If we're at the end already, this will avoid finding a
                # variable at all (which is what we want)
                if index < len(structure_info):
                    continue
            
            # Now did we find it?
            if isinstance(val, parameters.ParameterSet):
                context = val.get_context()
                ref = val['ref'] if 'ref' in val else None
                label = val['label'] if 'label' in val else None
                print context,ref,label
                if name in [context, ref, label] and not val in found:
                    found.append(val)
                
                    
        if len(found) == 0:
            raise ValueError('parameterSet {} with constraints "{}" nowhere found in system'.format(name,"@".join(structure_info)))
        elif return_type == 'single' and len(found)>1:
            raise ValueError("more than one parameterSet was returned from the search: either constrain search or set return_type='all'")
        elif return_type in ['single', 'first']:
            return found[0]
        else:
            return found
            
            
    def get_parameter(self, qualifier, return_type='first'):
        """
        Smart retrieval of a Parameter(s) from the system.

        If :envvar:`qualifier` is the sole occurrence of a parameter in this
        Bundle, there is no confusion and that parameter will be returned.
        If there is another occurrence, then the behaviour depends on the value
        of :envvar:`return_type`:
        
            - :envvar:`return_type='first'`: the first occurrence will be returned
            - :envvar:`return_type='single'`: a ValueError is raised if multiple occurrences exit
            - :envvar:`return_type='all'`: a list of all occurrences will be returned
       
        You can specify which qualifier you want with the :envvar:`@` operator.
        This operator allows you to hierarchically specify which parameter you
        mean. The general syntax is:
        
        :envvar:`<qualifier>@<label/ref/context>@<label/ref/context>`
        
        You can repeat as many :envvar:`@` operators as you want, as long as
        it they are hierarchically ordered, with the top level label of the
        system **last**.
        
        Examples::
        
            bundle.get_parameter('teff')
            bundle.get_parameter('teff@star') # star context
            bundle.get_parameter('teff@Vega') # name of the Star
            
            bundle.get_parameter('teff@primary') # if there is Body named primary
            bundle.get_parameter('teff@primary@V380_Cyg')
            
            bundle.get_parameter('time@lcsyn') # no confusion if there is only one lc
            bundle.get_parameter('flux@my_hipparcos_lightcurve')
        
        @param qualifier: qualifier of the parameter, or None to search all
        @type qualifier: str or None
        @param return_type: 'first', 'single', 'all'
        @type return_type: str
        @return: Parameter or list
        @rtype: Parameter or list
        """
        # Put the hierarchical structure in a list, that's easier to reference
        structure_info = []
        
        # Extract the info on the qualifier and structure info. The qualifier
        # is always first
        qualifier = qualifier.split('@')
        if len(qualifier)>1:
            structure_info = qualifier[1:]
        qualifier = qualifier[0]
                 
        # Reverse the structure info, that's easier to handle
        structure_info = structure_info[::-1]
        
        # First we'll loop through matching parametersets and gather all
        # parameters that match the qualifier
        found = []
        
        # You can always give top level system information if you desire
        if structure_info and structure_info[0] == self.system.get_label():
            start_index = 1
        else:
            start_index = 0
        
        # Now walk recursively over all parameters in the system, keeping track
        # of the history
        for path, val in self.system.walk_all(path_as_string=False):
            
            # Only if structure info is given, we need to do some work
            if structure_info:
                
                # We should look for the first structure information before
                # advancing to deeper levels; as long as we don't encounter
                # that reference/label/context, we can't look for the next one
                index = start_index
                
                # Look down the tree structure
                for level in path:
                    
                    # but only if we still have structure information
                    if index < len(structure_info):
                        
                        # We don't now the name of this level yet, but we'll
                        # figure it out
                        name_of_this_level = None
                        
                        # If this level is a Body, we check it's label
                        if isinstance(level, universe.Body):
                            name_of_this_level = level.get_label()
                            
                        # If it is a ParameterSet, we'll try to match the label,
                        # reference or context
                        elif isinstance(level, parameters.ParameterSet):
                            if 'ref' in level:
                                name_of_this_level = level['ref']
                            elif 'label' in level:
                                name_of_this_level = level['label']
                            if name_of_this_level != structure_info[index]:
                                name_of_this_level = level.get_context()
                        
                        # The walk iterator could also give us 'lcsyn' or something
                        # the like, it apparently doesn't walk over these PSsets
                        # themselves -- but no problem, this works
                        elif isinstance(level, str):
                            name_of_this_level = level
                            
                        # We're on the right track and can advance to find the
                        # next label in the structure!
                        if name_of_this_level == structure_info[index]:
                            index += 1
                
                # Keep looking if we didn't exhaust all specifications yet.
                # If we're at the end already, this will avoid finding a
                # variable at all (which is what we want)
                if index < len(structure_info):
                    continue
                
                
    def get_value(self, qualifier, return_type='first'):
        """
        Get the value from a Parameter(s) in the system
        
        @param qualifier: qualifier of the parameter, or None to search all
        @type qualifier: str or None
        @param name: label or ref of ps, or None to search all
        @type name: str or list orNone
        @param context: context of ps, or None to search all
        @type context: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: value of the parameter
        @rtype: depends on parameter type
        """
        par = self.get_parameter(qualifier, return_type=return_type)
        return par.get_value()
    
        
    def set_value(self, qualifier, value, *args, **kwargs):
        """
        Set the value of a Parameter(s) in the system
        
        @param qualifier: qualifier of the parameter
        @type qualifier: str
        @param value: new value of the parameter
        @type value: depends on parameter type
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param context: context of ps, or None to search all
        @type context: str or list or None
        @param accept_all: if True, will set value for all parameters that return from the search, otherwise will raise an Error if more than one is returned from the search
        @type accept_all: bool
        """
        apply_to = kwargs.pop('apply_to', 'first')
        if kwargs:
            raise SyntaxError("set_value does not take extra keyword arguments")
        
        params = self.get_parameter(qualifier, name, return_type=apply_to)
        
        if apply_to in ['first', 'single']:
            params.set_value(value, *args)
        elif apply_to in ['all']:
            for param in params:
                param.set_value(value, *args)
        else:
            raise ValueError("Cannot interpret argument apply_to='{}'".format(apply_to))
            
    def get_adjust(self,qualifier=None,name=None,context=None,return_type='single'):
        pass
            
    def set_adjust(self,qualifier,adjust,name=None,context=None,accept_all=False):
        pass
        
    #}
    #{ Objects
    def get_object(self, name):
        # return the Body/BodyBag from the system hierarchy
        pass
        
    def get_children(self, name):
        # return list of children for self.get_object(name)
        obj = self.get_object(name)
        if hasattr(obj,'bodies'):
            return [b.bodies[0] if hasattr(b,'bodies') else b for b in obj.bodies]
        else:
            return []
        
    def get_parent(self, name):
        # return the parent of self.get_object(name)
        return self.get_object(name).get_parent()
        
    #}  
    #{ Versions
    def add_version(self,name=None):
        """
        Add the current snapshot of the system as a new version entry
        Generally this is best to be handled by setting add_version=True in bundle.run_compute
        
        @param name: name of the version (defaults to current timestamp)
        @type name: str        
        """
        # purge any signals attached to system before copying
        self.purge_signals()
        
        # create copy of self.system and save to version
        system = self.get_system().copy()
        version = Version(system)
        date_created = datetime.now()
        version.set_value('date_created') = date_created
        version.set_value('name') = name if name is not None else str(date_created)
        
        self._add_to_section('version',version)

        # reattach signals to the system
        self.attach_system_signals()
        
    def get_version(self,search=None,search_by='name',return_type='single'):
        """
        Retrieve a stored version by one of its keys
        
        example:
        bundle.get_version('teff 4500')
        bundle.get_version(-2) # will go back 2 from the current version
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        @return: version (get system from version.get_system())
        @rtype: Version
        """
        if isinstance(version,int): #then easy to return from list
            return self._get_from_section('version',return_type='list')[self.versions_curr_i+version]
            
        return self._get_from_section('version',search,search_by,return_type=return_type)
           
    def restore_version(self,search,search_by='name'):
        """
        Restore a system version to be the current working system
        This should be used instead of bundle.set_system(bundle.get_version(...))
        
        See bundle.get_version() for syntax examples
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        """
        
        # retrieve the desired system
        version = self.get_version(search,search_by)
        system = copy.deepcopy(version.get_system())
        
        # set the current system
        # set_system attempts to find the version and reset versions_curr_i
        self.set_system(system)
        
    def remove_version(self,search,search_by='name'):
        """
        Permanently delete a stored version.
        This will not affect the current system.
        
        See bundle.get_version() for syntax examples
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        """

        self._remove_from_section('version',search,search_by)
        
        
    def rename_version(self,search,newname,search_by='name'):
        """
        Rename a currently existing version
        
        @param search: value to search by (depending on search_by)
        @type search: str
        @param newname: new name for the version
        @type newname: str
        @param search_by: key to search by (defaults to label)
        @type search_by: str      
        """
        
        version = self.get_version(search,search_by)
        version['name'] = newname
        
    #}
    #{ Datasets
    def _attach_datasets(self, output):
        """
        attach datasets and pbdeps from parsing file or creating synthetic datasets
        
        output is a dictionary with object names as keys and lists of both
        datasets and pbdeps as the values {objectname: [[ds1,ds2],[ps1,ps2]]}
        
        this is called from bundle.load_data and bundle.create_syn
        and should not be called on its own        
        """
        
        for objectlabel in output.keys():      
            # get the correct component (body)
            comp = self.get_object(objectlabel)
            
            # unpack all datasets and parametersets
            dss = output[objectlabel][0] # obs
            pss = output[objectlabel][1] # pbdep
            
            # attach pbdeps *before* obs (or will throw error and ref will not be correct)
            # pbdeps need to be attached to bodies, not body bags
            # so if comp is a body bag, attach it to all its predecessors
            if hasattr(comp, 'get_bodies'):
                for body in comp.get_bodies():
                    for ps in pss:
                        body.add_pbdeps(ps)
            else:
                for ps in pss:
                    comp.add_pbdeps(ps)

            # obs get attached to the requested object
            for ds in dss:

                #~ ds.load()
                comp.add_obs(ds)
    
    
    def load_data(self, category, filename, passband=None, columns=None,
                  components=None, ref=None, scale=False, offset=False):
        """
        import data from a file, create multiple DataSets, load data,
        and add to corresponding bodies
        
        @param category: category (lc, rv, sp, etv)
        @type category: str
        @param filename: filename
        @type filename: str
        @param passband: passband
        @type passband: str
        @param columns: list of columns in file
        @type columns: list of strings
        @param components: component for each column in file
        @type components: list of bodies
        @param ref: name for ref for all returned datasets
        @type ref: str    
        """
        
        if category == 'rv':
            output = datasets.parse_rv(filename, columns=columns,
                                       components=components, full_output=True,
                                       **{'passband':passband, 'ref': ref})
        elif category == 'lc':
            output = datasets.parse_lc(filename, columns=columns,
                                       components=components, full_output=True,
                                       **{'passband':passband, 'ref': ref})
        elif category == 'etv':
            output = datasets.parse_etv(filename, columns=columns,
                                        components=components, full_output=True,
                                        **{'passband':passband, 'ref': ref})
        
        elif category == 'sp':
            output = datasets.parse_sp(filename, columns=columns,
                                       components=components, full_output=True,
                                       **{'passband':passband, 'ref': ref})
        
        elif category == 'sed':
            output = datasets.parse_phot(filename, columns=columns,
                  group=filename, group_kwargs=dict(scale=scale, offset=offset),
                  full_output=True)
        #elif category == 'pl':
        #    output = datasets.parse_plprof(filename, columns=columns,
        #                               components=components, full_output=True,
        #                               **{'passband':passband, 'ref': ref})
        else:
            output = None
            print("only lc, rv, etv, and sp currently implemented")
        
        if output is not None:
            self._attach_datasets(output)
                       
    def create_syn(self, category='lc', times=None, components=None,
                   ref=None, **pbkwargs):
        """
        create and attach 'empty' datasets with no actual data but rather
        to provide times to compute the model

        additional keyword arguments can be provided and will be sent to the pbdeps
        (passband, atm, ld_func, ld_coeffs, etc)
        
        @param category: 'lc', 'rv', 'sp', 'etv', 'if', 'pl'
        @type category: str
        @param times: list of times
        @type times: list
        @param columns: list of columns in file
        @type columns: list of strings
        @param components: component for each column in file
        @type components: list of bodies
        @param ref: name for ref for all returned datasets
        @type ref: str    
        """
        # create pbdeps and attach to the necessary object
        # this function will be used for creating pbdeps without loading an actual file ie. creating a synthetic model only
        # times will need to be provided by the compute options (auto will not load times out of a pbdep)
        
        # Modified functionality from datasets.parse_header
        
        # What DataSet subclass do we need? We can derive it from the category    
        if not category in config.dataset_class:
            dataset_class = DataSet
        else:
            dataset_class = getattr(datasets, config.dataset_class[category])

        if components is None:
            # then attempt to make smart prediction
            if category == 'lc':
                # then top-level
                components = [self.get_system_structure(flat=True)[0]]
                logger.warning('components not provided - assuming {}'.format(components))
            else:
                logger.error('create_syn failed: components need to be provided')
                return

        output = {}
        for component in components:
            ds = dataset_class(context=category+'obs', time=times, ref=ref)
            
            # For the "dep" parameterSet, we'll use defaults derived from the
            # Body instead of the normal defaults. This should give the least
            # surprise to users: it is more logical to have default atmosphere,
            # LD-laws etc the same as the Body, then the "uniform" or "blackbody"
            # stuff that is in the defaults. So, if a parameter is in the
            # main body parameterSet, then we'll take that as a default value,
            # but it can be overriden by pbkwargs
            main_parset = self.get_ps(component)
            pb = parameters.ParameterSet(context=category+'dep', ref=ref)
            for key in pb:
                # derive default
                if key in main_parset:
                    default_value = main_parset[key]
                    # but only set it if not overridden
                    pb[key] = pbkwargs.get(key, default_value)
                else:
                    if key in pbkwargs.keys():
                        pb[key] = pbkwargs.get(key)
                        
            output[component] = [[ds],[pb]]

        self._attach_datasets(output)
        
    def get_syn(self,objref=None,dataref=None,force_dict=False):
        pass

    def get_dep(self,objref=None,dataref=None,force_dict=False):
        pass
        
    def get_obs(self, objref=None, dataref=None, force_dict=False):
        pass

    def enable_obs(self,dataref=None):
        pass

    def disable_obs(self,dataref=None):
        pass

    def adjust_obs(self,dataref=None,l3=None,pblum=None):
        pass

    def remove_data(self,dataref):
        """
        @param ref: ref (name) of the dataset
        @type ref: str
        """

        # disable any plotoptions that use this dataset
        for axes in self.axes:
            for pl in axes.get_plot():
                if pl.get_value('dataref')==dataref:
                    pl.set_value('active',False)
        
        # remove all obs attached to any object in the system
        for obj in self.get_system_structure(return_type='obj',flat=True):
            obj.remove_obs(refs=[dataref])
            if hasattr(obj, 'remove_pbdeps'): #TODO otherwise: warning 'BinaryRocheStar' has no attribute 'remove_pbdeps'
                obj.remove_pbdeps(refs=[dataref]) 

        return
        
    #}
    
    #{ Compute
    def add_compute(self,ps=None,**kwargs):
        """
        Add a new compute ParameterSet
        
        @param ps: compute ParameterSet
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
            
    def get_compute(self,label=None,return_type):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_section('compute',label,return_type=return_type)
        
    def remove_compute(self,label):
        """
        Remove a given compute ParameterSet
        
        @param label: name of compute ParameterSet
        @type label: str
        """
        return self._remove_from_section('compute',label)
    
    @run_on_server
    def run_compute(self,label=None,anim=False,add_version=None,server=None,**kwargs):
        """
        Convenience function to run observatory.observe
        
        @param label: name of one of the compute ParameterSets stored in bundle
        @type label: str
        @param anim: basename for animation, or False - will use settings in bundle.get_meshview()
        @type anim: False or str
        @param add_version: whether to save a snapshot of the system after compute is complete
        @type add_version: bool or str (which will become the version's name if provided)
        @param server: name of server to run on, or False to run locally (will override usersettings)
        @type server: string
        """
        system = self.get_system()
        
        if add_version is None:
            add_version = self.settings['add_version_on_compute']
            
        self.purge_signals(self.attached_signals_system)
        
        # clear all previous models and create new model
        system.clear_synthetic()

        self.system.set_time(0)
        
        # get compute options
        if label is None:
            options = parameters.ParameterSet(context='compute')
        else:
            options = self.get_compute(label)
        
        # get server options
        if server is not None:
            server = self.get_server(server)
            mpi = server.mpi_ps
        else:
            mpi = None
        
        if options['time']=='auto' and anim==False:
            #~ observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
            system.compute(mpi=mpi,**options)
        else:
            im_extra_func_kwargs = {key: value for key,value in self.get_meshview().items()}
            observatory.observe(system,options['time'],lc=True,rv=True,sp=True,pl=True,
                extra_func=[observatory.ef_binary_image] if anim!=False else [],
                extra_func_kwargs=[self.get_meshview()] if anim!=False else [],
                mpi=mpi,**options
                )
                
        if anim!=False:
            for ext in ['.gif','.avi']:
                plotlib.make_movie('ef_binary_image*.png',output='{}{}'.format(anim,ext),cleanup=ext=='.avi')
            
        system.uptodate = label
        
        if add_version is not False:
            self.add_version(name=None if add_version==True else add_version)

        self.attach_system_signals()

    #}
            
    #{ Fitting
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
            
    def get_fitting(self,label=None,return_type='single'):
        """
        Get a fitting ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        return self._get_from_section('fitting',label,return_type=return_type)

    def remove_fitting(self,label):
        """
        Remove a given fitting ParameterSet
        
        @param label: name of fitting ParameterSet
        @type label: str
        """
        self._remove_from_section('fitting',label)
        
    @run_on_server
    def run_fitting(self,computelabel,fittinglabel,add_feedback=None,server=None,**kwargs):
        """
        Run fitting for a given fitting ParameterSet
        and store the feedback
        
        @param computelabel: name of compute ParameterSet
        @param computelabel: str
        @param fittinglabel: name of fitting ParameterSet
        @type fittinglabel: str
        @param server: name of server to run on, or False to force locally (will override usersettings)
        @type server: string
        """
        if add_feedback is None:
            add_feedback = self.settings['add_feedback_on_fitting']
    
        if server is not None:
            server = self.get_server(server)
            mpi = server.mpi_ps
        else:
            mpi = None
        
        if isinstance(fittinglabel,str):
            fitparams = self.get_fitting(fittinglabel)
        else:
            # assume fitting ps
            fitparams = fittinglabel
            
        feedback = fitting.run(self.get_system(), params=self.get_compute(computelabel), fitparams=fitparams, mpi=mpi)
        
        if add_feedback:
            self.add_feedback(feedback)
        return feedback
    
    def add_feedback(self,ps,alias=None):
        """
        Add fitting results to the bundle.
        
        @param ps: results from the fitting
        @type ps: ParameterSet
        @param alias: alias name for this feedback (optional, defaults to datecreated string)
        @type alias: str
        """
       
        date_created = datetime.now()

        feedback = Feedback(ps)
        feedback.set_value('date_created', date_created)
        feedback.set_value('alias', alias if alias is not None else str(date_created))
        
        self._add_to_section('feedback',feedback)
        
    def get_feedback(self,search,search_by='label',return_type='single'):
        """
        Retrieve a stored feedback by one of its keys
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        """
        return self._get_from_section('feedback',search,search_by,return_type=return_type)
        
    def remove_feedback(self,search,search_by='label'):
        """
        Permanently delete a stored feedback.
        This will not affect the current system.
        
        See bundle.get_feedback() for syntax examples
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        """
        self._remove_from_section('feedback',search,search_by

    def rename_feedback(self,old_alias,new_alias):
        """
        Rename (the alias of) a currently existing feedback
        
        @param old_alias: the current alias of the feedback
        @type old_alias: str
        @param new_alias: the new alias of the feedback
        @type new_alias: str
        """
        raise NotImplementedError
        ps = self.get_feedback(old_alias,'alias')
        ps.set_value('alias',new_alias)
        
    def accept_feedback(self,search,search_by='label'):
        """
        Accept fitting results and apply to system
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        """
        fitting.accept_fit(self.get_system(),self.get_feedback(search,search_by).get_ps())
        
    def continue_mcmc(self,search,search_by='label',add_feedback=None,server=None,extra_iter=10):
        """
        Continue an MCMC chain.
        
        If you don't provide a label, the last MCMC run will be continued.
        
        @param search: value to search by (depending on search_by)
        @type search: str or None
        @param search_by: key to search by (defaults to label)
        @type search_by: str
        @param extra_iter: extra number of iterations
        @type extra_iter: int
        """
        raise NotImplementedError
        
        fitparams = self.get_feedback(search,search_by).get_ps()
        if fitparams.context.split(':')[-1] in ['pymc','emcee']:
            fitparams['iter'] += extra_iter
            
            self.run_fitting(computelabel,fitparams,add_feedback,server)
            
            feedback = fitting.run_emcee(self.system,params=self.compute,
                                fitparams=fitparams,mpi=self.mpi)
            break

        
        #~ if label is not None:
            #~ allfitparams = [self.get_feedback(feedback,by)]
        #~ else:
            #~ allfitparams = self.feedbacks.values()[::-1]
        #~ #-- take the last fitting ParameterSet that represents an mcmc
        #~ for fitparams in allfitparams:
            #~ if fitparams.context.split(':')[-1] in ['pymc','emcee']:
                #~ fitparams['iter'] += extra_iter
                #~ feedback = fitting.run_emcee(self.system,params=self.compute,
                                    #~ fitparams=fitparams,mpi=self.mpi)
                #~ break
    #}

    #{ Figures
    def plot_obs(self,dataref,objref=None,**kwargs):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param objref: label of the object
        @type objref: str
        """
        dss = self.get_obs(dataref=dataref,objref=objref,force_dict=True).values()
        if len(dss) > 1:
            logger.warning('more than one obs exists with this dataref, provide objref to ensure correct obs is used')
        ds = dss[0]
        typ = ds.context[:-3]
        
        if typ=='lc':
            plotting.plot_lcobs(self.system,ref=dataref,**kwargs)
        elif typ=='rv':
            plotting.plot_rvobs(self.system,ref=dataref,**kwargs)
        elif typ=='sp':
            plotting.plot_spobs(self.system,ref=dataref,**kwargs)
        elif typ=='if':
            plotting.plot_ifobs(self.system,ref=dataref,**kwargs)
        elif typ=='etv':
            plotting.plot_etvobs(self.system,ref=dataref,**kwargs)
        
    def plot_syn(self,dataref,objref=None,**kwargs):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param objref: label of the object
        @type objref: str
        """
        dss = self.get_syn(dataref=dataref,objref=objref,force_dict=True).values()
        if len(dss) > 1:
            logger.warning('more than one syn exists with this dataref, provide objref to ensure correct syn is used')
        ds = dss[0]
        typ = ds.context[:-3]
        
        if typ=='lc':
            plotting.plot_lcsyn(self.system,ref=dataref,**kwargs)
        elif typ=='rv':
            plotting.plot_rvsyn(self.system,ref=dataref,**kwargs)
        elif typ=='sp':
            plotting.plot_spsyn(self.system,ref=dataref,**kwargs)
        elif typ=='if':
            plotting.plot_ifsyn(self.system,ref=dataref,**kwargs)
        elif typ=='etv':
            plotting.plot_etvsyn(self.system,ref=dataref,**kwargs)        
            
    def plot_residuals(self,dataref,objref=None,**kwargs):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param objref: label of the object
        @type objref: str
        """
        dss = self.get_obs(dataref=dataref,objref=objref,force_dict=True).values()
        if len(dss) > 1:
            logger.warning('more than one obs exists with this dataref, provide objref to ensure correct obs is used')
        ds = dss[0]
        typ = ds.context[:-3]
        
        if typ=='lc':
            plotting.plot_lcres(self.system,ref=dataref,**kwargs)
        elif typ=='rv':
            plotting.plot_rvres(self.system,ref=dataref,**kwargs)
        elif typ=='sp':
            plotting.plot_spres(self.system,ref=dataref,**kwargs)
        elif typ=='if':
            plotting.plot_ifres(self.system,ref=dataref,**kwargs)
        elif typ=='etv':
            plotting.plot_etvres(self.system,ref=dataref,**kwargs)    
    
    def get_axes(self,ident=None):
        """
        Return an axes or list of axes that matches index OR title
        
        @param ident: index or title of the desired axes
        @type ident: int or str
        @return: axes
        @rtype: plotting.Axes
        """
        if isinstance(ident,int): 
            #then we need to return all in list and take index
            return self._get_from_section('axes',return_type='list')[ident]
        
        return self._get_from_section('axes',ident,'title')
        
    def add_axes(self,axes=None,**kwargs):
        """
        Add a new axes with a set of plotoptions
        
        kwargs will be applied to axesoptions ParameterSet
        it is suggested to at least intialize with kwargs for category and title
        
        @param axes: a axes to be plotted on a single axis
        @type axes: frontend.figures.Axes()
        @param title: (kwarg) name for the current plot - used to reference axes and as physical title
        @type title: str
        @param category: (kwarg) type of plot (lc,rv,etc)
        @type title: str
        """
        if axes is None:
            axes = Axes()
        for key in kwargs.keys():
            axes.set_value(key, kwargs[key])
            
        self._add_to_section('axes',axes)
        
    def remove_axes(self,ident):
        """
        Removes all axes with a given index or title
        
        @param ident: index or title of the axes to be removed
        @type ident: int or str
        """
        if isinstance(ident,int): 
            #then we need to return all in list and take index
            return self._get_from_section('axes',return_type='list')[ident]
        
        return self._get_from_section('axes',ident,'title')
                                
    def plot_axes(self,ident,mplfig=None,mplaxes=None,location=None):
        """
        Create a defined axes
        
        essentially a shortcut to bundle.get_axes(label).plot(...)
        
        @param ident: index or title of the axes
        @type ident: int or str
        @param mplfig: the matplotlib figure to add the axes to, if none is given one will be created
        @type mplfig: plt.Figure()
        @param mplaxes: the matplotlib axes to plot to (overrides mplfig)
        @type mplaxes: plt.axes.Axes()
        @param location: the location on the figure to add the axes
        @type location: str or tuple  
        """
        axes = self.get_axes(ident)
        axes.plot(self,mplfig=mplfig,mplaxes=mplaxes,location=location)
        
    def save_axes(self,ident,filename=None):
        """
        Save an axes to an image
        
        @param ident: index or title of the axes
        @type ident: int or str
        @param filename: name of desired output image
        @type filename: str
        """
        axes = self.get_axes(ident)
        if filename is None:
            filename = "{}.png".format(axes.get_value('title').replace(' ','_'))
        axes.savefig(self, filename)

    def anim_axes(self,ident,nframes=100,fps=24,outfile='anim',**kwargs):
        """
        Animate an axes on top of a meshplot
        
        @param ident: index or title of the axes
        @type ident: int or str
        @param nframes: number of frames in the gif (timestep)
        @type nframes: int
        @param fps: number of frames per second in the gif
        @type fps: int
        @param outfile: basename for output file [outfile.gif]
        @type outfile: str
        """

        axes = self.get_axes(ident)

        # if the axes is zoomed, use those limits
        tmin, tmax = axes.get_value('xlim')
        fmin, fmax = axes.get_value('ylim')
        
        # for now lets cheat - we should really check for all plots in axes.get_plot()
        plot = axes.get_plot(0)
        ds = axes.get_dataset(plot, self).asarray()

        if tmin is None or tmax is None:
            # need to get limits from datasets
            # check if 'phase' in axes.get_value('xaxis') 
            tmin, tmax = ds['time'].min(), ds['time'].max()
        
        if fmin is None or fmax is None:
            # again cheating - would probably need to check type of ds 
            # as this will probably only currently work for lc
            fmin, fmax = ds['flux'].min(), ds['flux'].max()

        times = np.linspace(tmin,tmax,nframes)
        
        # now get the limits for the meshplot so that the system
        # never goes out of limit during this time
        # TODO this function really only works well for binaries
        xmin, xmax, ymin, ymax = self._get_meshview_limits(times)
        
        figsize=kwargs.pop('figsize',10)
        dpi=kwargs.pop('dpi',80)
        figsize=(figsize,int(abs(ymin-ymax)/abs(xmin-xmax)*figsize))
        
        for i,time in enumerate(times):
            # set the time for the meshview and the selector
            self.set_select_time(time)

            plt.cla()

            mplfig = plt.figure(figsize=figsize, dpi=dpi)
            plt.gca().set_axis_off()
            self.plot_meshview(mplfig=mplfig,lims=(xmin,xmax,ymin,ymax))
            plt.twinx(plt.gca())
            plt.twiny(plt.gca())
            mplaxes = plt.gca()
            axes.plot(self,mplaxes=mplaxes)
            mplaxes.set_axis_off()
            mplaxes.set_ylim(fmin,fmax)
            mplfig.sel_axes.set_ylim(fmin,fmax)
            
            plt.savefig('gif_tmp_{:04d}.png'.format(i))
            
        for ext in ['.gif','.avi']:
            plotlib.make_movie('gif_tmp*.png',output='{}{}'.format(outfile,ext),fps=fps,cleanup=ext=='.avi')        
        
    def set_select_time(self,time=None):
        """
        Set the highlighted time to be shown in all plots
        
        set to None to not draw a highlighted time
        
        @param time: the time to highlight
        @type time: float or None
        """
        self.select_time = time
        #~ self.system.set_time(time)
        
    def get_meshview(self):
        """
        
        """
        return self._get_from_section('meshview')
        
    def _get_meshview_limits(self,times):
        # get size of system during these times for scaling image
        system = self.get_system()
        if hasattr(system, '__len__'):
            orbit = system[0].params['orbit']
            star1 = system[0]
            star2 = system[1]
        else:
            orbit = system.params['orbit']
            star1 = system
            star2 = None
        period = orbit['period']
        orbit1 = keplerorbit.get_binary_orbit(times, orbit, component='primary')[0]
        orbit2 = keplerorbit.get_binary_orbit(times, orbit, component='secondary')[0]
        # What's the radius of the stars?
        r1 = coordinates.norm(star1.mesh['_o_center'], axis=1).mean()
        if star2 is not None:
            r2 = coordinates.norm(star2.mesh['_o_center'], axis=1).mean()
        else:
            r2 = r1
        # Compute the limits
        xmin = min(orbit1[0].min(),orbit2[0].min())
        xmax = max(orbit1[0].max(),orbit2[0].max())
        ymin = min(orbit1[1].min(),orbit2[1].min())
        ymax = max(orbit1[1].max(),orbit2[1].max())
        xmin = xmin - 1.1*max(r1,r2)
        xmax = xmax + 1.1*max(r1,r2)
        ymin = ymin - 1.1*max(r1,r2)
        ymax = ymax + 1.1*max(r1,r2)
        
        return xmin, xmax, ymin, ymax
        
    def plot_meshview(self,mplfig=None,mplaxes=None,meshviewoptions=None,lims=None):
        """
        Creates a mesh plot using the saved options if not overridden
        
        @param mplfig: the matplotlib figure to add the axes to, if none is given one will be created
        @type mplfig: plt.Figure()
        @param mplaxes: the matplotlib axes to plot to (overrides mplfig)
        @type mplaxes: plt.axes.Axes()
        @param meshviewoptions: the options for the mesh, will default to saved options
        @type meshviewoptions: ParameterSet
        """
        if self.select_time is not None:
            self.set_time(self.select_time)
        
        po = self.get_meshview() if meshviewoptions is None else meshviewoptions

        if mplaxes is not None:
            axes = mplaxes
            mplfig = plt.gcf()
        elif mplfig is None:
            axes = plt.axes([0,0,1,1],aspect='equal',axisbg=po['background'])
            mplfig = plt.gcf()
        else:
            axes = mplfig.add_subplot(111,aspect='equal',axisbg=po['background'])
        
        mplfig.set_facecolor(po['background'])
        mplfig.set_edgecolor(po['background'])
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        #~ cmap = po['cmap'] if po['cmap'] is not 'None' else None
        slims, vrange, p = observatory.image(self.get_system(), ref=po['ref'], context=po['context'], select=po['select'], background=po['background'], ax=axes)
        
        #~ if po['contours']:
            #~ observatory.contour(self.system, select='longitude', colors='k', linewidths=2, linestyles='-')
            #~ observatory.contour(self.system, select='latitude', colors='k', linewidths=2, linestyles='-')
        
        if lims is None:
            axes.set_xlim(slims['xlim'])
            axes.set_ylim(slims['ylim'])       
        else: #apply supplied lims (likely from _get_meshview_limits)
            axes.set_xlim(lims[0],lims[1])
            axes.set_ylim(lims[2],lims[3])
        
    def get_orbitview(self):
        """
        
        """
        return self._get_from_section('orbitview')
        
    def plot_orbitview(self,mplfig=None,mplaxes=None,orbitviewoptions=None):
        """
        
        """
        if mplfig is None:
            if mplaxes is None: # no axes provided
                axes = plt.axes()
            else: # use provided axes
                axes = mplaxes
            
        else:
            axes = mplfig.add_subplot(111)
            
        po = self.get_orbitview() if orbitviewoptions is None else orbitviewoptions
            
        if po['data_times']:
            computeparams = parameters.ParameterSet(context='compute') #assume default (auto)
            observatory.extract_times_and_refs(self.get_system(),computeparams)
            times_data = computeparams['time']
        else:
            times_data = []
        
        top_orbit = self.get_orbit(self.get_system_structure(flat=True)[0])
        bottom_orbit = self.get_orbit(self.get_system_structure()[-1][0])
        if po['times'] == 'auto':
            times_full = np.arange(top_orbit.get_value('t0'),top_orbit.get_value('t0')+top_orbit.get_value('period'),bottom_orbit.get_value('period')/20.)
        else:
            times_full = po['times']
            
        for obj in self.get_system().get_bodies():
            orbits, components = obj.get_orbits()
            
            for times,marker in zip([times_full,times_data],['-','x']):
                if len(times):
                    pos, vel, t = keplerorbit.get_barycentric_hierarchical_orbit(times, orbits, components)
                
                    positions = ['x','y','z']
                    velocities = ['vx','vy','vz']
                    if po['xaxis'] in positions:
                        x = pos[positions.index(po['xaxis'])]
                    elif po['xaxis'] in velocities:
                        x = vel[positions.index(po['xaxis'])]
                    else: # time
                        x = t
                    if po['yaxis'] in positions:
                        y = pos[positions.index(po['yaxis'])]
                    elif po['yaxis'] in positions:
                        y = vel[positions.index(po['yaxis'])]
                    else: # time
                        y = t
                        
                    axes.plot(x,y,marker)
        axes.set_xlabel(po['xaxis'])
        axes.set_ylabel(po['yaxis'])
        
    #}
    
    #{Server
    def server_job_status(self):
        """
        check whether the job is finished on the server and ready for 
        the new bundle to be reloaded
        """
        server = self.get_server(self.lock['server'])
        script = self.lock['script']
        
        if server is not None:
            return server.check_script_status(script)
        else:
            return 'no job'
        
    def server_cancel(self):
        """
        unlock the bundle and remove information about the running job
        NOTE: this will not actually kill the job on the server, but will remove files that have already been created
        """
        server = self.get_server(self.lock['server'])
        script = self.lock['script']
        lock_files = self.lock['files']
        
        mount_dir = server.server_ps.get_value('mount_dir')
        
        if server.check_mount():
            logger.info('cleaning files in {}'.format(mount_dir))
            for fname in lock_files:
                try:
                    os.remove(os.path.join(mount_dir,fname))
                except:
                    pass
            
        self.lock = {'locked': False, 'server': '', 'script': '', 'command': '', 'files': [], 'rfile': None}
            
    def server_loop(self):
        """
        enter a loop to check the status of a job on a server and load
        the results when finished.
        
        You can always kill this loop and reenter (even after saving and loading the bundle)        
        """
        server = self.get_server(self.lock['server'])
        script = self.lock['script']
        
        servername = server.server_ps.get_value('label')
        
        while True:
            logger.info('checking {} server'.format(servername))
            if self.server_job_status()=='complete':
                self.server_get_results()
                return
            sleep(5)
    
    def server_get_results(self):
        """
        reload the bundle from a finished job on the server
        """
        
        server = self.get_server(self.lock['server'])
        script = self.lock['script']
        lock_files = self.lock['files']
        
        if self.server_job_status()!='complete':
            return False

        mount_dir = server.server_ps.get_value('mount_dir')
        
        logger.info('retrieving updated bundle from {}'.format(mount_dir))
        self_new = load(os.path.join(mount_dir,self.lock['rfile']))

        # reassign self_new -> self
        # (no one said it had to be pretty)
        for attr in ["__class__","__dict__"]:
                setattr(self, attr, getattr(self_new, attr))

        # alternatively we could just set the system, but 
        # then we lose flexibility in adjusting things outside
        # of bundle.system
        #~ bundle.set_system(bundle_new.get_system()) # note: anything changed outside system will be lost

        # cleanup files
        logger.info('cleaning files in {}'.format(mount_dir))
        for fname in lock_files:
            os.remove(os.path.join(mount_dir,fname))
        os.remove(os.path.join(mount_dir,'%s.status' % script))
        os.remove(os.path.join(mount_dir,'%s.log' % script))
        
        return
    
    #}
        
        
    #{ Pool
    #}
    
    #{ Attached Signals
    def attach_signal(self,param,funcname,callbackfunc,*args):
        """
        Attaches a callback signal and keeps list of attached signals
        
        @param param: the object to attach something to
        @type param: some class
        @param funcname: name of the class's method to add a callback to
        @type funcname: str
        @param callbackfunc: the callback function
        @type callbackfunc: callable function
        """
        system = self.get_system()
        
        # for some reason system.signals is becoming an 'instance' and 
        # is giving the error that it is not iterable
        # for now this will get around that, until we can find the source of the problem
        if system is not None and not isinstance(systemsignals, dict):
            #~ print "*system.signals not dict"
            system.signals = {}
        callbacks.attach_signal(param,funcname,callbackfunc,*args)
        self.attached_signals.append(param)
        
    def purge_signals(self,signals=None):
        """
        Purges all signals created through Bundle.attach_signal()
        
        @param signals: a list of signals to purge
        @type signals: list
        """
        
        if signals is None:
            signals = self.attached_signals
        #~ print "* purge_signals", signals
        for param in signals:
            callbacks.purge_signals(param)
        if signals == self.attached_signals:
            self.attached_signals = []
        elif signals == self.attached_signals_system:
            self.attached_signals_system = []
        #~ elif signals == self.attached_signals + self.attached_signals_system:
            #~ self.attached_signals = []
            #~ self.attached_signals_system = []
            
    def attach_system_signals(self):
        """
        this function attaches signals to:
            - set_value (for all parameters)
            - load_data/remove_data
            - enable_obs/disable_obs/adjust_obs
            
        when any of these signals are emitted, _on_param_changed will be called
        """

        self.purge_signals(self.attached_signals_system) # this will also clear the list
        for ps in [self.get_ps(label) for label in self.get_system_structure(return_type='label',flat=True)]+self.sections['compute']:
            self._attach_set_value_signals(ps)
        
        # these might already be attached?
        self.attach_signal(self,'load_data',self._on_param_changed)
        self.attach_signal(self,'remove_data',self._on_param_changed)
        self.attach_signal(self,'enable_obs',self._on_param_changed)
        self.attach_signal(self,'disable_obs',self._on_param_changed)
        self.attach_signal(self,'adjust_obs',self._on_param_changed)
        #~ self.attach_signal(self,'restore_version',self._on_param_changed)
        
    def _attach_set_value_signals(self,ps):
        for key in ps.keys():
            param = ps.get_parameter(key)
            self.attach_signal(param,'set_value',self._on_param_changed,ps)
            self.attached_signals_system.append(param)

    def _on_param_changed(self,param,ps=None):
        """
        this function is called whenever a signal is emitted that was attached
        in attach_system_signals
        """
        system = self.get_system()
        
        if ps is not None and ps.context == 'compute': # then we only want to set the changed compute to uptodate
            if system.uptodate is not False and self.get_compute(system.uptodate) == ps:
                system.uptodate=False
        else:
            system.uptodate = False
            
    #}
    
    #{ Saving
    def copy(self):
        """
        Copy this instance.
        """
        return copy.deepcopy(self)
    
    def save(self,filename=None,purge_signals=True,save_usersettings=False):
        """
        Save a class to an file.
        Will automatically purge all signals attached through bundle
        
        @param filename: path to save the bundle (or None to use same as last save)
        @type filename: str
        """
        if filename is None and self.filename is None:
            logger.error('save failed: need to provide filename')
            return
            
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        
        if purge_signals:
            #~ self_copy = self.copy()
            self.purge_signals()
            
        # remove user settings
        if not save_usersettings:
            settings = self.usersettings
            self.usersettings = None
        
        # pickle
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved bundle to file {} (pickle)'.format(filename))
        
        # reset user settings
        if not save_usersettings:
            self.usersettings = settings
        
        # call set_system so that purged (internal) signals are reconnected
        self.set_system(self.get_system())
    
    #}
    
class Version(object):
    """ 
    this class is essentially a glorified dictionary set to act like
    a parameterset that can hold a system and multiple keywords that can
    be used to search for it
    """
    def __init__(self,system):
        self.sections = {}
        self.sections['system'] = system
        
    def get_system(self):
        return self.sections['system']
        
    def get_value(self,key):
        return self.sections[key]
        
    def set_value(self,key,value):
        self.sections[key] = value
    
class Feedback(object):
    """ 
    this class is essentially a glorified dictionary set to act like
    a parameterset that can hold a feedback PS and multiple keywords that can
    be used to search for it
    """
    def __init__(self,ps):
        self.sections = {}
        self.sections['ps'] = ps
        
    def get_ps(self):
        return self.sections['ps']
        
    def get_value(self,key):
        return self.sections[key]
        
    def set_value(self,key,value):
        self.sections[key] = value
    
def load(filename, load_usersettings=True):
    """
    Load a class from a file.
    
    @param filename: filename of a Body or Bundle pickle file
    @type filename: str
    @param load_usersettings: flag to load custom user settings
    @type load_usersettings: bool
    @return: Bundle saved in file
    @rtype: Bundle
    """
    
    # First: is this thing a file?
    if os.path.isfile(filename):
        
        # If it is a file, try to unpickle it:   
        try:
            with open(filename, 'r') as open_file:
                contents = pickle.load(open_file)
        except:
            raise IOError(("Cannot load file {}: probably not "
                           "a Bundle file").format(filename))
        
        # If we can unpickle it, check if it is a Body(Bag) or a bundle. If it
        # is a Body(Bag), create a new bundle and set the system
        if isinstance(contents, universe.Body):
            bundle = Bundle(system=contents)
            
        # If it a bundle, we don't need to initiate it anymore
        elif isinstance(contents, Bundle):
            bundle = contents
            # for set_system to update all signals, etc
            bundle.set_system(bundle.system)
        
        # Else, we could load it, but we don't know what to do with it
        else:
            raise IOError(("Cannot load file {}: unrecognized contents "
                           "(probably not a Bundle file)").format(filename))
    else:
        raise IOError("Cannot load file {}: it does not exist".format(filename))
    
    # Load this users settings into the bundle
    if load_usersettings:
        bundle.set_usersettings()
    
    # That's it!
    return bundle


def guess_filetype(filename):
    """
    Guess what kind of file `filename` is and return the contents if possible.
    
    Possibilities and return values:
    
    1. Phoebe2.0 pickled Body: envvar:`file_type='pickle_body', contents=<phoebe.backend.Body>`
    2. Phoebe2.0 pickled Bundle: envvar:`file_type='pickle_bundle', contents=<phoebe.frontend.Bundle>`
    3. Phoebe Legacy file: envvar:`file_type='phoebe_legacy', contents=<phoebe.frontend.Body>`
    4. Wilson-Devinney lcin file: envvar:`file_type='wd', contents=<phoebe.frontend.Body>`
    5. Other existing loadable pickle file: envvar:`file_type='unknown', contents=<custom_class>`
    6. Other existing file: envvar:`file_type='unknown', contents=None`
    7. Nonexisting file: IOError
    
    """
    file_type = 'unknown'
    contents = None
    
    # First: is this thing a file?
    if os.path.isfile(filename):
        
        # If it is a file, try to unpickle it:   
        try:
            with open(filename, 'r') as open_file:
                contents = pickle.load(open_file)
            file_type = 'pickle'
        except AttributeError:
            logger.info("Probably old pickle file")
        except:
            pass
        
        # If the file is not a pickle file, it could be a Phoebe legacy file?
        if contents is None:
            
            try:
                contents = parsers.legacy_to_phoebe(filename, create_body=True,
                                                mesh='marching')
                file_type = 'phoebe_legacy'
            except IOError:
                pass
        
        # If it's not a pickle file nor a legacy file, is it a WD lcin file?
        if contents is None:
            
            try:
                contents = parsers.wd_to_phoebe(filename, mesh='marching',
                                                create_body=True)
                file_type = 'wd'
            except:
                contents = None
        
        # If we unpickled it, check if it is a Body(Bag) or a bundle.
        if file_type == 'pickle' and isinstance(contents, universe.Body):
            file_type += '_body'
            
        # If it a bundle, we don't need to initiate it anymore
        elif file_type == 'pickle' and isinstance(contents, Bundle):
            file_type += '_bundle'
        
        # If we don't know the filetype by now, we don't know it at all
    else:
        raise IOError(("Cannot guess type of file {}: "
                      "it does not exist").format(filename))
    
    return file_type, contents
