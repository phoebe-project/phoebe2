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

#~ def check_lock(fctn):
    #~ @functools.wraps(fctn)
    #~ def check(bundle,*args,**kwargs):
        #~ if bundle.lock['locked']:
            #~ # then raise warning and don't call function
            #~ logger.warning('there is a job running on {}: any changes made before receiving results may be lost'.format(bundle.lock['server']))
        #~ 
        #~ # call requested function
        #~ return fctn(bundle, *args, **kwargs)
    #~ return check

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
        #-- prepare 
        self.versions = [] #list of dictionaries
        self.versions_curr_i = None
        self.compute_options = []
        self.fitting_options = []
        self.feedbacks = [] #list of dictionaries
        self.axes = []
        
        self.select_time = None
        self.plot_meshviewoptions = parameters.ParameterSet(context='plotting:mesh')
        self.plot_orbitviewoptions = parameters.ParameterSet(context='plotting:orbit')
        
        self.pool = OrderedDict()
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
        
    def __str__(self):
        return self.to_string()
        
    def to_string(self):
        txt = ""
        txt += "{} compute options\n".format(len(self.compute_options))
        txt += "{} fitting options\n".format(len(self.fitting_options))
        txt += "{} axes\n".format(len(self.axes))
        txt += "============ System ============\n"
        txt += self.list()
        
        return txt
        
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
            return self.usersettings.get_value(key)
            
    def get_server(self,servername):
        """
        Return a server by name
        
        Note that this is merely a shortcut to bundle.get_usersettings().get_server()
        The server settings are stored in the usersettings and are not kept with the bundle
        
        @param servername: name of the server
        @type servername: string
        """
        return self.get_usersettings().get_server(servername)
        
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
            self.system = None
            return None
        # Or a real system
        elif isinstance(system, universe.Body):
            self.system = system
        elif isinstance(system, list) or isinstance(system, tuple):
            self.system = create.system(system)
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
            
            self.system = system
            
        if self.system is None:
            return
        
        # initialize uptodate
        self.system.uptodate = False
        
        # connect signals
        self.attach_system_signals()
        
        # check to see if in versions, and if so set versions_curr_i
        versions = [v['system'] for v in self.versions]
        if system in versions:
            i = [v['system'] for v in self.versions].index(system)
        else:
            i = None
        self.versions_curr_i = i
        
    def attach_system_signals(self):
        #~ print "* attach_system_signals"

        self.purge_signals(self.attached_signals_system) # this will also clear the list
        #~ self.attached_signals_system = []
        for ps in [self.get_ps(label) for label in self.get_system_structure(return_type='label',flat=True)]+self.compute_options:
            self._attach_set_value_signals(ps)
        
        # these might already be attached?
        self.attach_signal(self,'load_data',self._on_param_changed)
        self.attach_signal(self,'enable_obs',self._on_param_changed)
        self.attach_signal(self,'disable_obs',self._on_param_changed)
        self.attach_signal(self,'adjust_obs',self._on_param_changed)
        self.attach_signal(self,'restore_version',self._on_param_changed)
        
    def _attach_set_value_signals(self,ps):
        #~ print "* attaching", ps.context
        for key in ps.keys():
            #~ print "*** attaching signal %s:%s" % (label,key)
            param = ps.get_parameter(key)
            self.attach_signal(param,'set_value',self._on_param_changed,ps)
            self.attached_signals_system.append(param)

    def _on_param_changed(self,param,ps=None):
        if ps is not None and ps.context == 'compute': # then we only want to set the changed compute to uptodate
            if self.system.uptodate is not False and self.get_compute(self.system.uptodate) == ps:
                self.system.uptodate=False
        else:
            self.system.uptodate = False
    
    def get_system(self):
        """
        Return the system.
        
        @return: the attached system
        @rtype: Body or BodyBag
        """
        return self.system      
                
    def get_system_structure(self,return_type='label',flat=False,**kwargs):
        """
        Get the structure of the system below any bodybag in a variety of formats
        
        @param return_type: list of types to return including label,obj,ps,nchild,mask
        @type return_type: str or list of strings
        @param flat: whether to flatten to a 1d list
        @type flat: bool
        @return: the system structure
        @rtype: list or list of lists        
        """
        all_types = ['obj','ps','nchild','mask','label']
        
        # create empty list for all types, later we'll decide which to return
        struc = {}
        for typ in all_types:
            struc[typ]=[]
        
        if 'old_mask' in kwargs.keys() and 'mask' in return_type:
            # old_mask should be passed as a tuple of two flattened lists
            # the first list should be parametersets
            # the second list should be the old mask (booleans)
            # if 'mask' is in return_types, and this info is given
            # then any matching ps from the original ps will retain its old bool
            # any new items will have True in the mask
            
            # to find any new items added to the system structure
            # pass the old flattened ps output and a list of False of the same length
            
            old_mask = kwargs['old_mask']
            old_struclabel = old_mask[0]
            old_strucmask = old_mask[1]
            
        else:
            old_struclabel = [] # new mask will always be True
                
        if 'top_level' in kwargs.keys():
            item = kwargs.pop('top_level') # we don't want it in kwargs for the recursive call
        else:
            item = self.system
            
        struc['obj'].append(item)
        itemlabel = self.get_label(item)
        struc['label'].append(itemlabel)
        struc['ps'].append(self.get_ps(item))
        
        # label,ps,nchild are different whether item is body or bodybag
        if hasattr(item, 'bodies'):
            struc['nchild'].append('2') # should not be so strict
        else:
            struc['nchild'].append('0')
            
        if itemlabel in old_struclabel: #then apply previous bool from mask
            struc['mask'].append(old_strucmask[old_struclabel.index(itemlabel)])
        else:
            struc['mask'].append(True)

        # recursively loop to get hierarchical structure
        children = self.get_children(item)
        if len(children) > 1:
            for typ in all_types:
                struc[typ].append([])
        for child in children:
            new = self.get_system_structure(return_type=all_types,flat=flat,top_level=child,**kwargs)
            for i,typ in enumerate(all_types):
                struc[typ][-1]+=new[i]

        if isinstance(return_type, list):
            return [list(utils.traverse(struc[rtype])) if flat else struc[rtype] for rtype in return_type]
        else: #then just one passed, so return a single list
            rtype = return_type
            return list(utils.traverse(struc[rtype])) if flat else struc[rtype]
    
    def get_object(self, objectname=None, force_dict=False):
        """
        search for an object inside the system structure and return it if found
        this will return the Body or BodyBag
        to get the ParameterSet see get_ps, get_component, and get_orbit
        
        @param objectname: label of the desired object
        @type objectname: str, Body, or BodyBag
        @param bodybag: the bodybag to search under (will default to system)
        @type bodybag: BodyBag
        @return: the object or dictionary of objects
        @rtype: ParameterSet or OrderedDict
        """
        #this should return a Body or BodyBag
        if objectname is not None and not isinstance(objectname,str): #then return whatever is sent (probably the body or bodybag)
            if force_dict:
                return OrderedDict([(self.get_label(objectname),objectname)])
            return objectname
            
        names, objects = self.get_system_structure(return_type=['label','obj'],flat=True)
        
        # if the objectname is '__nolabel__', then it comes from the parsers,
        # and it means there is no label given to this particular object. If so,
        # it can only mean that there is only one object in the system (or it
        # is the top object that is meant)
        if objectname == '__nolabel__':
            objectname = names[0]
        
        if objectname is not None:
            if force_dict:
                return OrderedDict([(objectname,objects[names.index(objectname)])])
            return objects[names.index(objectname)]
        else:
            return OrderedDict([(n,o) for n,o in zip(names,objects)])
        
    def list(self,summary=None,*args):
        """
        List with indices all the ParameterSets that are available.
        Simply a shortcut to bundle.get_system().list(...)
        """
        return self.system.list(summary,*args)
        
    def clear_synthetic(self):
        """
        Clear all synthetic datasets
        Simply a shortcut to bundle.get_system().clear_synthetic()
        """
        return self.system.clear_synthetic()
        
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
        
        self.system.fix_mesh()
        self.system.set_time(time)
        
    def get_uptodate(self):
        """
        Check whether the synthetic model is uptodate
        
        If any parameters in the system have changed since the latest
        run_compute this will return False.
        If not, this will return the label of the latest compute options

        @return: uptodate
        @rtype: bool or str
        """
        if isinstance(self.system.uptodate,str) or isinstance(self.system.uptodate,bool):
            return self.system.uptodate
        else:
            return False
        
    def get_label(self,obj):
        """
        Get the label/name for any object (Body or BodyBag)
        
        @param obj: the object
        @type obj: Body or BodyBag
        @return: the label/name
        @rtype: str        
        """
        if isinstance(obj,str): #then probably already name, and return
            return obj
        
        objectname = None
        if hasattr(obj,'bodies'): #then bodybag
            #search for orbit in the children bodies
            for item in obj.bodies: # should be the same for all of them, but we'll search all anyways
                #NOTE: this may fail if you have different orbits for each component
                if 'orbit' in item.params.keys():
                    objectname = item.params['orbit']['label']
            return objectname
        else: #then hopefully body
            return obj.get_label()
    
    def change_label(self,oldlabel,newlabel):
        """
        not implemented yet
        """
        raise NotImplementedError
        
        # need to handle changing label and references from children objects, etc
        
        return
        
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
        
    def get_ps(self, name=None, context=None, return_type='single'):
        """
        Retrieve a ParameterSet(s) from the system
        
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param context: context of ps, or None to search all
        @type context: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: ParameterSet or OrderedDict
        @rtype: ParameterSet or OrderedDict
        """
        matching_ps = OrderedDict()
        
        if isinstance(name, str):
            name = [name]
            
        if isinstance(context, str):
            context = [context]
        
        for ps in self.get_system().walk():
            # Is walk recursive? -- pieterdegroote: yes
            # Why can't I find meshes? -- pieterdegroote: because it walks over all .params entries only (there are more general "walkers" attached to body or bodybag if you need them)
            ps_name = None
            if 'ref' in ps:
                ps_name = ps.get_value('ref')
                #~ ps_name = '{}:{}'.format(ps_name,ps.get_value('ref')) # ds are coming before the first orbit, otherwise this would work
                # TODO - there are likely going to be duplicates overwritten here
                # so we want ps_name = "{}:{}".format(parent_name,ref)
                # we would then need to do intelligent string matching between this and the provided name (ie if name='None:mylc')
                # get_obs can then simply call get_ps('objref:dataref',['lcobs','rvobs','spobs','etvobs'],return_type)
                # -- pieterdegroote: I like the idea, since it's simple for the user
                #                    and doesn't require a lot of nomenclature
                #                    however we can't use label:ref, because
                #                    some parametersets already have this ':'
                #                    perhaps we should reserve another sign, e.g. '#'
                #                    (we need to take something that people don't use
                #                    in filenames often, since filenames can be
                #                    easy references for datasets
                
            elif 'label' in ps:
                ps_name = ps.get_value('label')
            #~ else:
                # are there other cases??
                #~ print ps.keys()
            if ps_name is not None and (context is None or ps.context in context) and (name is None or ps_name in name):
                matching_ps[ps_name] = ps
                
        return self._return_from_dict(matching_ps,return_type)
            
    def get_parameter(self, qualifier, return_type='first'):
        """
        Retrieve a Parameter(s) from the system
        
        @param qualifier: qualifier of the parameter, or None to search all
        @type qualifier: str or None
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param context: context of ps, or None to search all
        @type context: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: Parameter or OrderedDict
        @rtype: Parameter or OrderedDict
        """
        matching_param = OrderedDict()
        structure_info = []
        
        qualifier = qualifier.split('@')
        if len(qualifier)>1:
            structure_info = qualifier[1:]
        qualifier = qualifier[0]
                 
        structure_info = structure_info[::-1]
        
        # first we'll loop through matching parametersets
        # and gather all parameters that match the qualifier
        found = []
        
        if structure_info and structure_info[0] == self.system.get_label():
            start_index = 1
        else:
            start_index = 0
        
        for path, val in self.system.walk_all(path_as_string=False):
            if structure_info:
                index = start_index
                for level in path:
                    if index < len(structure_info):
                        name_of_this_level = None
                        if isinstance(level, universe.Body):
                            name_of_this_level = level.get_label()
                        elif isinstance(level, parameters.ParameterSet):
                            if 'ref' in level:
                                name_of_this_level = level['ref']
                            elif 'label' in level:
                                name_of_this_level = level['label']
                            if name_of_this_level != structure_info[-1]:
                                name_of_this_level = level.get_context()
                        # We're on the right track!
                        if name_of_this_level == structure_info[index]:
                            index += 1
                if index < len(structure_info):
                    continue

            if isinstance(val, parameters.Parameter):
                if val.get_qualifier() == qualifier and not val in found:
                    found.append(val)
                    
        if len(found) == 0:
            raise ValueError('parameter {} with constraints "{}" nowhere found in system'.format(qualifier,"@".join(structure_info)))
        elif return_type == 'single' and len(found)>1:
            raise ValueError("more than one parameter was returned from the search: either constrain search or set return_type='all'")
        elif return_type in ['single', 'first']:
            return found[0]
        else:
            return found
        
        
    def get_value(self,qualifier=None,name=None,context=None,return_type='single'):
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
        matching_value = OrderedDict()
        
        for name,param in self.get_parameter(qualifier,name,context,return_type='dict').items():
            matching_value[name] = param.get_value()
            
        return self._return_from_dict(matching_value,return_type)
        
    def set_value(self,qualifier,value,name=None,context=None,accept_all=False):
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
        
        params = self.get_parameter(qualifier,name,context,return_type='list')
        
        if len(params) > 1 and not accept_all:
            raise ValueError("more than one parameter was returned from the search: either constrain search or set accept_all=True")
        
        for param in params:
            param.set_value(qualifier,value)
            
    def get_adjust(self,qualifier=None,name=None,context=None,return_type='single'):
        """
        Retrieve whether a Parameter(s) in the system is marked for adjustment
        
        @param qualifier: qualifier of the parameter, or None to search all
        @type qualifier: str or None
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param context: context of ps, or None to search all
        @type context: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: whether the parameter is set for adjustment
        @rtype: bool
        """
        matching_value = OrderedDict()
        
        for name,param in self.get_parameter(qualifier,name,context,return_type='dict').items():
            matching_value[name] = param.get_adjust()
            
        return self._return_from_dict(matching_value,return_type)
            
    def set_adjust(self,qualifier,adjust,name=None,context=None,accept_all=False):
        """
        Set whether a Parameter(s) in the system is marked for adjustment
        
        @param qualifier: qualifier of the parameter
        @type qualifier: str
        @param adjust: whether to set the parameter for adjustment
        @type adjust: bool
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param context: context of ps, or None to search all
        @type context: str or list or None
        @param accept_all: if True, will set value for all parameters that return from the search, otherwise will raise an Error if more than one is returned from the search
        @type accept_all: bool
        """
        
        params = self.get_parameter(qualifier,name,context,return_type='list')
        
        if len(params) > 1 and not accept_all:
            raise ValueError("more than one parameter was returned from the search: either constrain search or set accept_all=True")
        
        for param in params:
            param.set_adjust(qualifier,adjust)
            
    def get_orbit(self,name=None,return_type='single'):
        """
        Retrieve an orbit ParameterSet(s) from the system
        
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: ParameterSet or OrderedDict
        @rtype: ParameterSet or OrderedDict
        """
        return self.get_ps(name,context=['orbit'],return_type=return_type)

    def get_component(self,name=None,return_type='single'):
        """
        Retrieve a component ParameterSet(s) from the system
        
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: ParameterSet or OrderedDict
        @rtype: ParameterSet or OrderedDict
        """
        # TODO - need to provide all the different contexts here
        return self.get_ps(name,context=['BinaryRocheStar'],return_type=return_type)
        
    def get_mesh(self,name=None,return_type='single'):
        """
        Retrieve a mesh ParameterSet(s) from the system
        
        @param name: label or ref of ps, or None to search all
        @type name: str or list or None
        @param return_type: 'single', 'dict', 'list'
        @type return_type: str
        @return: ParameterSet or OrderedDict
        @rtype: ParameterSet or OrderedDict
        """
        return self.get_ps(name,context=['mesh:marching','mesh:disk'],return_type=return_type)
        
    def NEW_get_object(self):
        # does get_object needs to be rewritten?
        
        # this function is supposed to return the object itself instead of the PS
        # so BodyBag or Body or BinaryStar, etc
        pass
        
    def NEW_get_mesh(self):
        # cannot currently get meshes from get_ps?
        # so we either need to make it so you can (preferably)
        # or have another function for getting meshes (probably get the object and then find its mesh)
        pass
        
    def NEW_get_system_structure(self):
        # do we need to rewrite this?
        # get_system_structure was written before we had list(), so maybe
        # this functionality is no longer needed
        # HOWEVER, I may have lied a bit about things not being in here
        # for the GUI - so if we remove this from the bundle I still need
        # a working version to use in the GUI (but it can be in the GUI code)
        pass
    
    def get_mesh(self,objectname=None):
        """
        retrieve the ParameterSet for a mesh by name
        
        @param objectname: label of the desired object
        @type objectname: str or Body
        @return: the ParameterSet of the mesh
        @rtype: ParameterSet
        """
        objects = self.get_object(objectname,force_dict=True)

        meshes = OrderedDict([(name,obj.params['mesh']) for name,obj in objects.items() if 'mesh' in obj.params.keys()])

        if objectname is None:
            return meshes
        else:
            return meshes[objectname]
    
    def get_parent(self,objectname,return_type='obj'):
        """
        retrieve the parent of an item in a hierarchical structure
        
        @param objectname: label of the child object
        @type objectname: str
        @param return_type: what to return ('obj','str','ps','mesh')
        @type return_type: str
        @return: the parent
        @rtype: defaults to object or whatever specified by return_type        
        """
        return self._object_to_type(self.get_object(objectname).get_parent(),return_type)
        # TODO I'm concerned about what the parent of a BodyBag is returning
    
    def get_children(self,objectname,return_type='obj'):
        """
        retrieve the children of an item in a hierarchical structure
        
        @param objectname: label of the parent object
        @type objecname: str
        @param return_type: what to return ('obj','str','ps','mesh')
        @type return_type: str
        @return: list of children objects
        @rtype: defaults to object or whatever specified by return_type
        """
        obj = self.get_object(objectname)
        if hasattr(obj,'bodies'):
            return self._object_to_type([b.bodies[0] if hasattr(b,'bodies') else b for b in self.get_object(objectname).bodies])
            #~ return self._object_to_type(obj.get_children()) # still throws an error 
        else:
            return []

    def remove_item(self,objectname):
        """
        remove an item and all its children from the system
        
        @param objectname: label of the item to be removed
        @type objectname: str
        """
        obj = self.get_object(objectname)
        oldparent = self.get_parent(objectname)

        # remove reference? delete object?
        raise NotImplementedError
        return obj
    
    def insert_parent(self,objectname,parent):
        """
        add a parent to an existing item in the hierarchical system structure
        
        @param objectname: label of the child item
        @type objectname: str
        @param parent: the new parent
        @type parent: BodyBag        
        """
        raise NotImplementedError
        return
        
        obj = self.get_object(objectname)
        oldparent = self.get_parent(objectname)
        
        # parent should have oldparent as its parent and obj as a child
        
        # this isn't going to work because we can't initialize a BodyBag with no arguments/children
        parent.append(obj)
        oldparent.append(parent)
        
    def insert_child(self,objectname,child):
        """
        add a child to an existing item in the hierarchical system structure
        
        @param objectname: label of the parent item
        @type objectname: str
        @param child: the new child
        @type parent: Body
        """
        self.get_object(objectname).append(child)
        
    def _object_to_type(self,obj,return_type='obj'):
        """
        
        
        """
        
        # handle sending a single object or list
        if not isinstance(obj, list):
            return_lst = False
            obj = [obj]
        else:
            return_lst = True
            
        # get the correct type (in a list)
        if return_type == 'label':
            return_ = [self.get_label(o) for o in obj]
        elif return_type in ['ps','orbit','component']:
            return_ = [self.get_ps(o) for o in obj]
        elif return_type == 'mesh':
            return_ = [self.get_mesh(o) for o in obj]
            
        else:
            return_ = obj
        
        # return list or single item (same as input)
        if return_lst:
            return return_
        else:
            return return_[0]
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
        system = self.system.copy()
        version = {}
        version['system'] = system
        version['date_created'] = datetime.now()
        version['name'] = name if name is not None else str(version['date_created'])
        
        self.versions.append(version)
        
        # reattach signals to the system
        self.attach_system_signals()
        
    def get_version(self,version=None,by='name',return_type='system'):
        """
        Retrieve a stored system version by one of its keys
        
        version can either be one of the keys (date_create, name) or
        the amount to increment from the current version
        
        example:
        bundle.get_version('teff 4500')
        bundle.get_version(-2) # will go back 2 from the current version
        
        @param version: the key of the version, or index increment
        @type version: str or int
        @param by: what key to search by (defaults to name)
        @type by: str
        @return: system
        @rtype: Body or BodyBag
        """
        if isinstance(version,int): #then easy to return from list
            return self.versions[self.versions_curr_i+version]['system']
        
        # create a dictionary with key defined by by and values which are the systems
        versions = {v[by]: i if return_type=='i' else v[return_type] for i,v in enumerate(self.versions)}
        
        if version is None:
            return versions
        else:
            return versions[version]
           
    def restore_version(self,version,by='name'):
        """
        Restore a system version to be the current working system
        This should be used instead of bundle.set_system(bundle.get_version(...))
        
        See bundle.get_version() for syntax examples
        
        @param version: the key of the version, or index increment
        @type version: str or int
        @param by: what key to search by (defaults to name)
        @type by: str
        """
        
        # retrieve the desired system
        system = self.get_version(version,by=by)
        
        # set the current system
        # set_system attempts to find the version and reset versions_curr_i
        self.set_system(system)
        
    def remove_version(self,version,by='name'):
        """
        Permanently delete a stored version.
        This will not affect the current system.
        
        See bundle.get_version() for syntax examples
        
        @param version: the key of the version, or index increment
        @type version: str or int
        @param by: what key to search by (defaults to name)
        @type by: str
        """

        i = self.get_version(version,by=by,return_type='i')
        return self.versions.pop(i)
        
        
    def rename_version(self,version,newname):
        """
        Rename a currently existing version
        
        @param version: the system or name of the version that you want to edit
        @type version: version or str
        @param newname: the new name
        @type newname: str        
        """
        key = 'name' if isinstance(version,str) else 'system'
        # get index where this system is in self.versions
        i = [v[key] for v in self.versions].index(version)
        # change the name value
        self.versions[i]['name']=newname

    #}
    #{ Datasets
    def _get_data_ps(self, kind, objref, dataref, force_dict):
        """
        Retrieve a parameterSet of type dep, obs or syn.
        
        Used by :py:func:`Bundle.get_obs`, :py:func:`Bundle.get_syn`, and
        :py:func:`Bundle.get_dep`
        
        <Q>: why are objref, dataref and force_dict not None by default?
        
        @param kind: 'obs', 'syn', or 'dep'
        @type kind: str
        @param objref: name of the object the dataset is attached to
        @type objref: str
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @return: dep ParameterSet
        @rtype: ParameterSet
        """
        
        if objref is None:
            # then search the whole system
            return_ = {}
            for objref in self.get_system_structure(return_type='label',flat=True):
                parsets = self._get_data_ps(kind,objref,dataref=dataref,force_dict=True)
                for parset in parsets.values():
                    if parset not in return_:
                        return_['%s:%s' % (objref, parset.get_value('ref'))] = parset
                        
            if len(return_) == 1 and not force_dict:
                return return_.values()[0]
            else:
                return return_
            
        # now we assume we're dealing with a single object
        obj = self.get_object(objref)
        
        if dataref is not None:
            # then search for the dataref by name/index
            if kind == 'syn':
                parset = obj.get_synthetic(ref=dataref, cumulative=True)
            else:
                parset = obj.get_parset(type=kind, ref=dataref)[0]
            
            if parset != None and parset != []:
                if force_dict:
                    return OrderedDict([('%s:%s' % (objref, parset.get_value('ref')), parset)])
                else:
                    return parset
            return OrderedDict()
        
        else:
            # then loop through indices until there are none left
            return_ = []
            if kind in obj.params.keys():
                for typ in obj.params[kind]:
                    for ref in obj.params[kind][typ]:
                        return_.append(obj.params[kind][typ][ref])
            
            if len(return_)==1 and not force_dict:
                return return_[0]
            else:
                return {'%s:%s' % (objref, r.get_value('ref')): r for r in return_}
            
            
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
        
    def get_obs(self, objref=None, dataref=None, force_dict=False):
        """
        get an observables dataset by the object its attached to and its label
        if objectname and ref are given, this will return a single dataset
        if either or both are not given, this will return a list of all datasets matching the search
        
        @param objref: name of the object the dataset is attached to
        @type objref: str
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @return: dataset
        @rtype: ParameterSet
        """
        
        return self._get_data_ps('obs',objref,dataref,force_dict)
        
    def enable_obs(self,dataref=None):
        """
        Include observations in the fitting process by enabling them.
        
        @param dataref: ref (name) of the dataset
        @type dataref: str
        """
        for obs in self.get_obs(dataref=dataref, force_dict=True).values():
            obs.set_enabled(True)
            
               
    def disable_obs(self,dataref=None):
        """
        Exclude observations from the fitting process by disabling them.
        
        @param dataref: ref (name) of the dataset
        @type dataref: str
        """
        for obs in self.get_obs(dataref=dataref, force_dict=True).values():
            obs.set_enabled(False)
        
        
    def adjust_obs(self,dataref=None,l3=None,pblum=None):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param l3: whether l3 should be marked for adjustment
        @type l3: bool or None
        @param pblum: whether pblum should be marked for adjustment
        @type pblum: bool or None
        """
        
        for obs in self.get_obs(dataref=dataref,force_dict=True).values():
            if l3 is not None:
                obs.set_adjust('l3',l3)
            if pblum is not None:
                obs.set_adjust('pblum',pblum)
        return
        
        
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
            
            
    def get_syn(self,objref=None,dataref=None,force_dict=False):
        """
        get a synthetic dataset by the object its attached to and its label
        if objref and dataref are given, this will return a single dataset
        if either or both are not give, this will return a list of all datasets matching the search
        
        @param objref: name of the object the dataset is attached to
        @type objref: str
        @param dataref: dataref (name) of the dataset
        @type dataref: str
        @return: dataset
        @rtype: ParameterSet
        """
        
        return self._get_data_ps('syn',objref,dataref,force_dict)
                
    def get_dep(self,objref=None,dataref=None,force_dict=False):
        """
        get a dep by the object its attached to and its label
        if objectname and ref are given, this will return a single dataset
        if either or both are not give, this will return a list of all datasets matching the search
        
        @param objref: name of the object the dataset is attached to
        @type objref: str
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @return: dep ParameterSet
        @rtype: ParameterSet
        """
        
        return self._get_data_ps('pbdep',objref,dataref,force_dict)
        
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

        self._attach_set_value_signals(compute)
            
    def get_compute(self,label=None):
        """
        Get a compute ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: compute ParameterSet
        @rtype: ParameterSet
        """
        # create a dictionary with key as the label
        compute_options = self.usersettings.get_compute()
        # bundle compute options override those in usersettings
        for co in self.compute_options:
            compute_options[co.get_value('label')] = co
        
        if label is None:
            return compute_options
        elif label in compute_options.keys():
            co = compute_options[label]
            if co not in self.compute_options:
                # then this came from usersettings - so we need to copy to bundle
                # and return the new version.  From this point on, this version will
                # be returned and used even if usersettings is changed.
                # To return to the default options, remove from the bundle
                # by calling bundle.remove_compute(label)
                co_return = copy.deepcopy(co)
                self.add_compute(co_return)
                return co_return
            else:
                # then we're return from bundle already
                return co
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
        if add_version is None:
            add_version = self.settings['add_version_on_compute']
            
        self.purge_signals(self.attached_signals_system)
        
        # clear all previous models and create new model
        self.system.clear_synthetic()

        try:
            self.system.set_time(0)
        except:
            self.system.fix_mesh()
        
        if label is None:
            options = parameters.ParameterSet(context='compute')
        else:
            if label not in self.get_compute():
                return KeyError
            options = self.get_compute(label)
        
        if server is not None:
            server = self.get_server(server)
            mpi = server.mpi_ps
        else:
            mpi = None
        
        if options['time']=='auto' and anim==False:
            #~ observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
            self.system.compute(mpi=mpi,**options)
        else:
            im_extra_func_kwargs = {key: value for key,value in self.get_meshview().items()}
            observatory.observe(self.system,options['time'],lc=True,rv=True,sp=True,pl=True,
                extra_func=[observatory.ef_binary_image] if anim!=False else [],
                extra_func_kwargs=[self.get_meshview()] if anim!=False else [],
                mpi=mpi,**options
                )
                
        if anim!=False:
            for ext in ['.gif','.avi']:
                plotlib.make_movie('ef_binary_image*.png',output='{}{}'.format(anim,ext),cleanup=ext=='.avi')
            
        self.system.uptodate = label
        
        if add_version is not False:
            self.add_version(name=None if add_version==True else add_version)

        self.attach_system_signals()

    #}
            
    #{ Fitting
    def add_fitting(self,fitting=None,**kwargs):
        """
        Add a new fitting ParameterSet
        
        @param fitting: fitting ParameterSet
        @type fitting:  None, or ParameterSet
        """
        context = kwargs.pop('context') if 'context' in kwargs.keys() else 'fitting:pymc'
        if fitting is None:
            fitting = parameters.ParameterSet(context=context)
        for k,v in kwargs.items():
            fitting.set_value(k,v)
            
        self.fitting_options.append(fitting)

        self._attach_set_value_signals(fitting)
            
    def get_fitting(self,label=None):
        """
        Get a fitting ParameterSet by name
        
        @param label: name of ParameterSet
        @type label: str
        @return: fitting ParameterSet
        @rtype: ParameterSet
        """
        # create a dictionary with key as the label
        fitting_options = self.usersettings.get_fitting()
        # bundle fitting options override those in usersettings
        for co in self.fitting_options:
            fitting_options[co.get_value('label')] = co
        
        if label is None:
            return fitting_options
        elif label in fitting_options.keys():
            fo = fitting_options[label]
            if fo not in self.fitting_options:
                # then this came from usersettings - so we need to copy to bundle
                # and return the new version.  From this point on, this version will
                # be returned and used even if usersettings is changed.
                # To return to the default options, remove from the bundle
                # by calling bundle.remove_fitting(label)
                fo_return = copy.deepcopy(fo)
                self.add_fitting(fo_return)
                return fo_return
            else:
                # then we're return from bundle already
                return fo
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
        
        feedback = fitting.run(self.system, params=self.get_compute(computelabel), fitparams=self.get_fitting(fittinglabel), mpi=mpi)
        
        if add_feedback:
            self.add_feedback(feedback)
        return feedback
    
    def add_feedback(self,feedback,name=None):
        """
        Add fitting results to the bundle.
        
        @param feedback: results from the fitting
        @type feedback: ParameterSet
        """
        #-- if we've nothing to add, then just quit
        if feedback is None: return None
        #-- feedbacks should be a list, not one result:
        if not isinstance(feedback,list):
            feedback = [feedback]
        #-- then add the results to the bundle.
        for f in feedback:
            fd = {}
            fd['feedback'] = f
            fd['date_created'] = datetime.now()
            fd['name'] = name if name is not None else str(fd['date_created'])
            fd['label'] = f['label'] #original label of the feedback
            
            self.feedbacks.append(fd)
    
          
    def get_feedback(self,feedback=None,by='name',return_type='feedback'):
        """
        Retrieve a stored feedback by one of its keys
        
        @param feedback: the key of the feedback, or index increment
        @type feedback: str or int
        @param by: what key to search by (defaults to name)
        @type by: str
        @return: feedback
        @rtype: ParameterSet
        """
        if len(self.feedbacks)==0:
            return None
            
        if isinstance(feedback,int): #then easy to return from list
            return self.feedbacks[feedback]['feedback']
        
        # create a dictionary with key defined by by and values which are the systems
        feedbacks = {v[by]: i if return_type=='i' else v[return_type] for i,v in enumerate(self.feedbacks)}
        
        if feedback is None:
            return feedbacks
        else:
            return feedbacks[feedback]
            
    def remove_feedback(self,feedback,by='name'):
        """
        Permanently delete a stored feedback.
        This will not affect the current system.
        
        See bundle.get_feedback() for syntax examples
        
        @param feedback: the key of the feedback, or index increment
        @type feedback: str or int
        @param by: what key to search by (defaults to name)
        @type by: str
        """

        i = self.get_feedback(feedback,by=by,return_type='i')
        return self.feedbacks.pop(i)
        
        
    def rename_feedback(self,feedback,newname):
        """
        Rename a currently existing feedback
        
        @param feedback: the feedback or name of the feedback that you want to edit
        @type feedback: feedback or str
        @param newname: the new name
        @type newname: str        
        """
        key = 'name' if isinstance(feedback,str) else 'system'
        # get index where this system is in self.feedbacks
        i = [v[key] for v in self.feedbacks].index(feedback)
        # change the name value
        self.feedbacks[i]['name']=newname
               
    def accept_feedback(self,feedback,by='name'):
        """
        Accept fitting results and apply to system
        
        @param feedback: name of the feedback ParameterSet to accept
        @type label: str
        @param by: key to search for feedback (see get_feedback)
        @type by: str
        """
        fitting.accept_fit(self.system,self.get_feedback(feedback,by))
        
    def continue_mcmc(self,feedback=None,by='name',extra_iter=10):
        """
        Continue an MCMC chain.
        
        If you don't provide a label, the last MCMC run will be continued.
        
        @param feedback: name of the MCMC ParameterSet to continue
        @type label: str
        @param by: key to search for feedback (see get_feedback)
        @type by: str
        @param extra_iter: extra number of iterations
        @type extra_iter: int
        """
        if label is not None:
            allfitparams = [self.get_feedback(feedback,by)]
        else:
            allfitparams = self.feedbacks.values()[::-1]
        #-- take the last fitting ParameterSet that represents an mcmc
        for fitparams in allfitparams:
            if fitparams.context.split(':')[-1] in ['pymc','emcee']:
                fitparams['iter'] += extra_iter
                feedback = fitting.run_emcee(self.system,params=self.compute,
                                    fitparams=fitparams,mpi=self.mpi)
                break
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
        axes = OrderedDict([(ax.get_value('title'), ax) for ax in self.axes])
        
        if ident is None:
            return axes
        elif isinstance(ident,str):
            return axes[ident]
        else:
            return axes.values()[ident]
        
    def add_axes(self,axes=None,**kwargs):
        """
        Add a new axes with a set of plotoptions
        
        kwargs will be applied to axesoptions ParameterSet
        it is suggested to at least intialize with kwargs for category and title
        
        @param axes: a axes to be plotted on a single axis
        @type axes:
        @param title: (kwarg) name for the current plot - used to reference axes and as physical title
        @type title: str
        @param category: (kwarg) type of plot (lc,rv,etc)
        @type title: str
        """
        if axes is None:
            axes = Axes()
        for key in kwargs.keys():
            axes.set_value(key, kwargs[key])
        self.axes.append(axes)
        
    def remove_axes(self,ident):
        """
        Removes all axes with a given index or title
        
        @param ident: index or title of the axes to be removed
        @type ident: int or str
        """
        return self.axes.pop(self.axes.index(self.get_axes(ident)))
                                
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
        return self.plot_meshviewoptions
        
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
            self.system.set_time(self.select_time)
        
        po = self.plot_meshviewoptions if meshviewoptions is None else meshviewoptions

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
        slims, vrange, p = observatory.image(self.system, ref=po['ref'], context=po['context'], select=po['select'], background=po['background'], ax=axes)
        
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
        return self.plot_orbitviewoptions
        
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
            
        po = self.plot_orbitviewoptions if orbitviewoptions is None else orbitviewoptions
            
        if po['data_times']:
            computeparams = parameters.ParameterSet(context='compute') #assume default (auto)
            observatory.extract_times_and_refs(self.system,computeparams)
            times_data = computeparams['time']
        else:
            times_data = []
        
        top_orbit = self.get_orbit(self.get_system_structure(flat=True)[0])
        bottom_orbit = self.get_orbit(self.get_system_structure()[-1][0])
        if po['times'] == 'auto':
            times_full = np.arange(top_orbit.get_value('t0'),top_orbit.get_value('t0')+top_orbit.get_value('period'),bottom_orbit.get_value('period')/20.)
        else:
            times_full = po['times']
            
        for obj in self.system.get_bodies():
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
        
        # for some reason system.signals is becoming an 'instance' and 
        # is giving the error that it is not iterable
        # for now this will get around that, until we can find the source of the problem
        if self.system is not None and not isinstance(self.system.signals, dict):
            #~ print "*system.signals not dict"
            self.system.signals = {}
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
        self.set_system(self.system)
    
    #}
    
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
