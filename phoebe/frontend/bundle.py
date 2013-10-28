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
import matplotlib.pyplot as plt
import copy

from phoebe.utils import callbacks, utils
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.backend import fitting, observatory, plotting
from phoebe.dynamics import keplerorbit
from phoebe.frontend import usersettings

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
        if fctn.func_name == 'run_fitting':
            fctn_type = 'fitting'
        elif fctn.func_name == 'run_compute':
            fctn_type = 'preview' if args[0] == 'Preview' else 'compute'
            
        servername = kwargs.pop('server', None)
        
        callargs = inspect.getcallargs(fctn,bundle,*args,**kwargs)
        dump = callargs.pop('self')
        callargstr = ','.join(["%s=%s" % (key, "\'%s\'" % callargs[key] if isinstance(callargs[key],str) else callargs[key]) for key in callargs.keys()])

        
        if servername is None:
            # then we need to determine which system based on preferences
            servername = bundle.get_usersettings().get_setting('use_server_on_%s' % (fctn_type))
            
            # TODO allow for option to search for available or by priority?
            
        if servername is not False:
            server =  bundle.get_server(servername)
            if server.check_connection():
                logger.warning("running on servers not yet supported ({})".format(servername))
                # prepare job
                bundle.save('bundle.job.tmp') # might have a problem here if threaded
                
                f = open('script.job.tmp','w')
                f.write("#!/usr/bin/python")
                f.write("from phoebe.frontend.bundle import load")
                f.write("bundle = load('bundle.job.tmp')")
                f.write("bundle.%s(%s)" % (fctn.func_name, callargstr))
                f.write("bundle.save('bundle.return.job.tmp')")
                f.close()
                
                # create job and submit
                job = Job(fctn_type,script_file='script.job.tmp',aux_files=['bundle.job.tmp'],return_files=['bundle.return.job.tmp'])
                bundle.current_job = job
                job.start(server)
                
                # either lock bundle or loop to check progress

            else:
                logger.warning('{} server not available, running locally'.format(servername))

        # run locally by calling the original function
        return fctn(bundle, *args, **kwargs)
    
    return parse

class Bundle(object):
    """
    Class representing a collection of systems and stuff related to it.
    """
    def __init__(self,system=None,mpi=None,fitting=None,axes=None,compute=None):
        """
        Initialize a Bundle.
        
        You don't have to give anything, but you can. Afterwards, you can
        always add more.
        """
        #-- prepare 
        self.versions = [] #list of dictionaries
        self.versions_curr_i = None
        self.mpi = mpi
        self.compute = OrderedDict()
        self.fitting = OrderedDict()
        self.feedbacks = [] #list of dictionaries
        self.figs = OrderedDict()
        
        self.select_time = None
        self.plot_meshviewoptions = parameters.ParameterSet(context='plotting:mesh')
        self.plot_orbitviewoptions = parameters.ParameterSet(context='plotting:orbit')
        
        self.pool = OrderedDict()
        self.signals = {}
        self.attached_signals = []
        self.attached_signals_system = [] #these will be purged when making copies of the system and can be restored through set_system
        
        self.add_fitting(fitting)
        self.axes = []
        if isinstance(axes, list):
            self.axes = axes
        if isinstance(axes, dict):
            for key in axes.keys():
                self.add_axes(axes[key], key)
        self.add_compute(compute)
        
        self.set_usersettings()
        
        self.current_job = None
        
        self.settings = {}
        self.settings['add_version_on_compute'] = False
        self.settings['add_feedback_on_fitting'] = False
        self.settings['update_mesh_on_select_time'] = False
        
        self.set_system(system) # will handle all signals, etc
        
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
        Get the value for a setting by name
        
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
            return self.usersettings.get_setting(key)
            
    def get_server(self,servername):
        """
        Return a server by name
        
        Note that this is merely a shortcut to bundle.get_usersettings().get_server()
        The server settings are stored in the usersettings and are not kept with the bundle
        
        @param servername: name of the server
        @type servername: string
        """
        return self.get_usersettings().get_server(servername)
        
    def set_mpi(self,mpi):
        self.mpi = False if mpi is None else mpi
        
    def get_mpi(self):
        return self.mpi
    #}    
    
    #{ System
    def set_system(self,system=None):
        """
        Change the system

        @param system: the new system
        @type system: System
        """
        self.system = system 
        if system is None:  return None
        
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
        for ps in [self.get_ps(label) for label in self.get_system_structure(return_type='label',flat=True)]+self.compute.values():
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
            if self.system.uptodate is not False and self.compute[self.system.uptodate] == ps:
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
    
    def get_object(self,objectname):
        """
        search for an object inside the system structure and return it if found
        this will return the Body or BodyBag
        to get the ParameterSet see get_ps, get_component, and get_orbit
        
        @param objectname: label of the desired object
        @type objectname: str, Body, or BodyBag
        @param bodybag: the bodybag to search under (will default to system)
        @type bodybag: BodyBag
        @return: the object
        @rtype: ParameterSet
        """
        #this should return a Body or BodyBag
        
        if not isinstance(objectname,str): #then return whatever is sent (probably the body or bodybag)
            return objectname
            
        names, objects = self.get_system_structure(return_type=['label','obj'],flat=True)
        return objects[names.index(objectname)]
        
    def list(self,summary=None,*args):
        """
        List with indices all the parameterSets that are available.
        Simply a shortcut to bundle.get_system().list(...)
        """
        return self.system.list(summary,*args)
        
    def set_time(self,time):
        """
        Shortcut to bundle.get_system().set_time() which insures fix_mesh
        is called first if any data was recently attached
        
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
            
    def get_ps(self,objectname):
        """
        retrieve the ParameterSet for a component or orbit
        this is the same as calling get_orbit or get_component, except that this tries to predict the type first
        
        @param objectname: label of the desired object
        @type objectname: str, Body, or BodyBag
        @return: the ParameterSet of the component
        @rtype: ParameterSet
        """
        if isinstance(objectname,str):
            obj = self.get_object(objectname)
        else:
            obj = objectname
        if hasattr(obj,'bodies'):
            return self.get_orbit(obj)
        else:
            return self.get_component(obj)
        
    def get_component(self,objectname):
        """
        retrieve the ParameterSet for a component by name
        
        @param objectname: label of the desired object
        @type objectname: str or Body
        @return: the ParameterSet of the component
        @rtype: ParameterSet
        """
        # get_object already allows passing object, so we don't have to check to see if str
        params = self.get_object(objectname=objectname).params
        if 'component' in params.keys():
            return params['component']
        elif 'star' in params.keys():
            return params['star']
        return None
    
    def get_orbit(self,objectname):
        """
        retrieve the ParameterSet for a orbit by name
        
        @param objectname: label of the desired object
        @type objectname: str or BodyBag
        @return: the ParameterSet of the orbit
        @rtype: ParameterSet
        """
        if not isinstance(objectname,str):
            objectname = self.get_label(objectname)
        
        # for orbits we have to be more clever
        for path,item in self.system.walk_all():
            if path[-1] == 'orbit' and item['label']==objectname:
                return item
        return None
        
    def get_mesh(self,objectname):
        """
        retrieve the ParameterSet for a mesh by name
        
        @param objectname: label of the desired object
        @type objectname: str or Body
        @return: the ParameterSet of the mesh
        @rtype: ParameterSet
        """
        if isinstance(objectname,str):
            obj = self.get_object(objectname)
        else:
            obj = objectname
            
        return obj.params['mesh']
               
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
            
    def get_parameter(self,qualifier,objref=None):
        """
        Retrieve a parameter from the system
        If objref is not provided and there are more than one object in 
        they system containing a parameter with the same name, this will
        return an error and ask you to provide a valid objref
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param objref: label of the object
        @type objref: str
        @return: Parameter corresponding to the qualifier
        @rtype: Parameter
        """
    
        objrefs = self.get_system_structure(flat=True) if objref is None else [objref]
        
        return_params = []
        return_objrefs = []
        
        for objref in objrefs:
            ps = self.get_ps(objref)
            if qualifier in ps.keys():
                return_params.append(ps.get_parameter(qualifier))
                return_objrefs.append(objref)
                
        if len(return_params) > 1:
            raise ValueError("parameter '{}' is ambiguous, please provide one of the following for objref:\n{}".format(qualifier,'\n'.join(["\t'%s'" % ref for ref in return_objrefs])))

        elif len(return_params)==0:
            raise ValueError("parameter '{}' was not found in any of the objects in the system".format(qualifier))
            
        return return_params[0]
        
    def get_value(self,qualifier,objref=None):
        """
        Retrieve the value from a parameter from the system
        This is identical to bundle.get_parameter(qualifier,objref).get_value()
        
        If objref is not provided and there are more than one object in 
        they system containing a parameter with the same name, this will
        return an error and ask you to provide a valid objref
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param objref: label of the object
        @type objref: str
        @return: value of the Parameter corresponding to the qualifier
        @rtype: (depends on the parameter)
        """
        
        param = self.get_parameter(qualifier,objref)
        return param.get_value()
        
    def set_value(self,qualifier,value,objref=None):
        """
        Set the value of a parameter from the system
        This is identical to bundle.get_parameter(qualifier,objref).set_value(value)
        
        If objref is not provided and there are more than one object in 
        they system containing a parameter with the same name, this will
        return an error and ask you to provide a valid objref
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param value: the new value for the parameter
        @type value: (depends on parameter)
        @param objref: label of the object
        @type objref: str
        """
        
        param = self.get_parameter(qualifier,objref)
        param.set_value(value)
        
    def set_adjust(self,qualifier,adjust=True,objref=None):
        """
        Set adjust for a parameter from the system
        This is identical to bundle.get_parameter(qualifier,objref).set_adjust(adjust)
        
        If objref is not provided and there are more than one object in 
        they system containing a parameter with the same name, this will
        return an error and ask you to provide a valid objref
        
        @param qualifier: name or alias of the variable
        @type qualifier: str
        @param value: the new value for the parameter
        @type value: (depends on parameter)
        @param objref: label of the object
        @type objref: str
        """
        
        param = self.get_parameter(qualifier,objref)
        param.set_adjust(adjust)
        
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
    
    
    #{ Loading Data
    def load_data(self,context,filename,passband=None,columns=None,components=None,ref=None):
        """
        import data from a file, create multiple DataSets, load data,
        and add to corresponding bodies
        
        @param context: context
        @type context: str
        @param filename: filename
        @type filename: str
        @param passband: passband
        @type passband: str
        @param columns: list of columns in file
        @type columns: list of strings
        @param components: component for each column in file
        @type components: list of bodies
        @param name: name for ref for all returned datasets (currently UNUSED)
        @type name: str    
        """
        
        if 'rv' in context:
            output = datasets.parse_rv(filename,columns=columns,components=components,full_output=True,**{'passband':passband, 'ref': ref})
        elif 'lc' in context:
            output = datasets.parse_lc(filename,columns=columns,components=components,full_output=True,**{'passband':passband, 'ref': ref})
        elif 'etv' in context:
            output = datasets.parse_etv(filename,columns=columns,components=components,full_output=True,**{'passband':passband, 'ref': ref})
        elif 'sp' in context:
            output = datasets.parse_sp(filename,columns=columns,components=components,full_output=True,**{'passband':passband, 'ref': ref})
        else:
            print("only lc, rv, etv, and sp currently implemented")
        
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
                
    def create_syn(self,context,passband=None,components=None,ref=None):
        """
        
        """
        # create pbdeps and attach to the necessary object
        # this function will be used for creating pbdeps without loading an actual file ie. creating a synthetic model only
        # times will need to be provided by the compute options (auto will not load times out of a pbdep)
        
        raise NotImplementedError
    
    def add_obs(self,objectname,dataset):
        """
        attach dataset to an object
        
        @param objectname: name of the object to attach the dataset to
        @type objectname: str
        @param dataset: the dataset
        @type dataset: parameterSet
        """
        obj = self.get_object(objectname)
        
        typ = dataset.context[-3:]
        if typ=='dep':
            obj.add_pbdeps(dataset)
        elif typ=='obs':
            obj.add_obs(ds)
        
    def get_obs(self,objref=None,dataref=None):
        """
        get an observables dataset by the object its attached to and its label
        if objectname and ref are given, this will return a single dataset
        if either or both are not give, this will return a list of all datasets matching the search
        
        @param objref: name of the object the dataset is attached to
        @type objref: str
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @return: dataset
        @rtype: parameterSet
        """
        if objref is None:
            # then search the whole system
            return_ = []
            for objname in self.get_system_structure(return_type='obj',flat=True):
                parsets = self.get_obs(objname,dataref=dataref)
                for parset in parsets:
                    if parset not in return_:
                        return_.append(parset)
            return return_
            
        # now we assume we're dealing with a single object
        obj = self.get_object(objref)
        
        if dataref is not None:
            # then search for the dataref by name/index
            parset = obj.get_parset(type='obs',ref=dataref)
            if parset != (None, None):
                return [parset[0]]
            return []
        else:
            # then loop through indices until there are none left
            return_ = []
            for typ in obj.params['obs']:
                for ref in obj.params['obs'][typ]:
                    return_.append(obj.params['obs'][typ][ref])
            return return_

    def enable_obs(self,dataref=None):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        """
        for obs in self.get_obs(dataref=dataref):
            obs.enabled=True
        return
               
    def disable_obs(self,dataref=None):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        """
        
        for obs in self.get_obs(dataref=dataref):
            obs.enabled=False
        return
        
    def adjust_obs(self,dataref=None,l3=None,pblum=None):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param l3: whether l3 should be marked for adjustment
        @type l3: bool or None
        @param pblum: whether pblum should be marked for adjustment
        @type pblum: bool or None
        """
        
        for obs in self.get_obs(dataref=dataref):
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
            for pl in axes.plots:
                if pl.get_value('dataref')==dataref:
                    pl.set_value('active',False)
        
        # remove all obs attached to any object in the system
        for obj in self.get_system_structure(return_type='obj',flat=True):
            obj.remove_obs(refs=[dataref])
            if hasattr(obj, 'remove_pbdeps'): #TODO otherwise: warning 'BinaryRocheStar' has no attribute 'remove_pbdeps'
                obj.remove_pbdeps(refs=[dataref]) 

        return
            
    def get_syn(self,objref=None,dataref=None):
        """
        get a synthetic dataset by the object its attached to and its label
        if objref and dataref are given, this will return a single dataset
        if either or both are not give, this will return a list of all datasets matching the search
        
        @param objref: name of the object the dataset is attached to
        @type objref: str
        @param dataref: dataref (name) of the dataset
        @type dataref: str
        @return: dataset
        @rtype: parameterSet
        """
        if objref is None:
            # then search the whole system
            return_ = []
            for obj in self.get_system_structure(return_type='obj',flat=True):
                #~ print "*** a", self.get_label(obj)
                parsets = self.get_syn(obj,dataref=dataref)
                for parset in parsets:
                    if parset not in return_:
                        return_.append(parset)
            return return_
            
        # now we assume we're dealing with a single object
        obj = self.get_object(objref)
        
        if dataref is not None:
            # then search for the dataref by name/index
            parset = obj.get_synthetic(ref=dataref, cumulative=True)
            if parset != None:
                return [parset]
            return []
        else:
            # then loop through indices until there are none left
            return_ = []
            for typ in obj.params['obs']:
                for ref in obj.params['obs'][typ]:
                    return_.append(obj.params['obs'][typ][ref])
            return return_
        
    #}
    
    #{ Compute
    def add_compute(self,compute=None,label=None):
        """
        Add a new compute parameterSet
        
        @param compute: compute parameterSet
        @type compute:  None, parameterSet or list of ParameterSets
        @param label: name of parameterSet
        @type label: None, str, or list of strs
        """
        if compute is None:
            if label is None:
                return
            compute = parameters.ParameterSet(context='compute')
        if not isinstance(compute,list):
            compute = [compute]
            label = [label]
        for i,c in enumerate(compute):
            if label[i] is not None:
                name = label[i]
            else:
                name = c['label']
            self.compute[name] = c
            
        self._attach_set_value_signals(self.compute[name])
            
    def get_compute(self,label):
        """
        Get a compute parameterSet by name
        
        @param label: name of parameterSet
        @type label: str
        @return: compute parameterSet
        @rtype: parameterSet
        """
        if label in self.compute.keys():
            return self.compute[label]
        else:
            return None
        
    def remove_compute(self,label):
        """
        Remove a given compute parameterSet
        
        @param label: name of compute parameterSet
        @type label: str
        """
        if label is None:    return None
        self.compute.pop(label)        
    
    @run_on_server
    def run_compute(self,label=None,mpi=True,im=False,add_version=None,server=None):
        """
        Convenience function to run observatory.observe
        
        @param label: name of one of the compute parameterSets stored in bundle
        @type label: str
        @param mpi: whether to use mpi (will use stored options)
        @type mpi: bool
        @param im: whether to create images at each timestamp, if None will use value from settings
        @type im: bool or None
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

        #~ self.system.fix_mesh()
        #~ try:
            #~ self.system.set_time(0)
        #~ except:
            #~ pass
        try:
            self.system.set_time(0)
        except:
            self.system.fix_mesh()
        
        if label is None:
            options = parameters.ParameterSet(context='compute')
        else:
            if label not in self.compute:
                return KeyError
            options = self.compute[label]

        
        if options['time']=='auto' and not im:
            #~ observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
            self.system.compute(mpi=self.mpi if mpi else None,**options)
        else:
            observatory.observe(self.system,options['time'],lc=True,rv=True,sp=True,pl=True,
                extra_func=[observatory.ef_binary_image] if im else [],
                extra_func_kwargs=[dict(select='teff',cmap='blackbody')] if im else [],
                mpi=self.mpi if mpi else None,**options
                )
                
        self.system.uptodate = label
        
        if add_version is not False:
            self.add_version(name=None if add_version==True else add_version)

        self.attach_system_signals()




    #}
            
    #{ Fitting
    def add_fitting(self,fitting,label=None):
        """
        Add a new fitting parameterSet
        
        @param fitting: fitting parameterSet
        @type fitting:  None, parameterSet or list of ParameterSets
        @param label: name of parameterSet
        @type label: None, str, or list of strs
        """
        if fitting is None: return None
        if not isinstance(fitting,list):
            fitting = [fitting]
            label = [label]
        for i,f in enumerate(fitting):
            if label[i] is not None:
                name = label[i]
            else:
                name = f['label']
            self.fitting[name] = f
            
    def get_fitting(self,label):
        """
        Get a fitting parameterSet by name
        
        @param label: name of parameterSet
        @type label: str
        @return: fitting parameterSet
        @rtype: parameterSet
        """
        if label in self.fitting.keys():
            return self.fitting[label]
        else:
            return None
        
    def remove_fitting(self,label):
        """
        Remove a given fitting parameterSet
        
        @param label: name of fitting parameterSet
        @type label: str
        """
        if label is None:    return None
        self.fitting.pop(label)        
        
    @run_on_server
    def run_fitting(self,computelabel,fittinglabel,mpi=True,add_feedback=None,server=None):
        """
        Run fitting for a given fitting parameterSet
        and store the feedback
        
        @param computelabel: name of compute parameterSet
        @param computelabel: str
        @param fittinglabel: name of fitting parameterSet
        @type fittinglabel: str
        @param mpi: whether mpi is enabled (will use stored options)
        @type mpi: bool
        @param server: name of server to run on, or False to force locally (will override usersettings)
        @type server: string
        """
        if add_feedback is None:
            add_feedback = self.settings['add_feedback_on_fitting']
        
        feedback = fitting.run(self.system, params=self.get_compute(computelabel), fitparams=self.get_fitting(fittinglabel), mpi=self.mpi if mpi else None)
        
        if add_feedback:
            self.add_feedback(feedback)
        return feedback
    
    def add_feedback(self,feedback,name=None):
        """
        Add fitting results to the bundle.
        
        @param feedback: results from the fitting
        @type feedback: parameterSet
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
        @rtype: parameterSet
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
        
        @param feedback: name of the feedback parameterSet to accept
        @type label: str
        @param by: key to search for feedback (see get_feedback)
        @type by: str
        """
        fitting.accept_fit(self.system,self.get_feedback(feedback,by))
        
    def continue_mcmc(self,feedback=None,by='name',extra_iter=10):
        """
        Continue an MCMC chain.
        
        If you don't provide a label, the last MCMC run will be continued.
        
        @param feedback: name of the MCMC parameterSet to continue
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
        #-- take the last fitting parameterSet that represents an mcmc
        for fitparams in allfitparams:
            if fitparams.context.split(':')[-1] in ['pymc','emcee']:
                fitparams['iter'] += extra_iter
                feedback = fitting.run_emcee(self.system,params=self.compute,
                                    fitparams=fitparams,mpi=self.mpi)
                break
    #}

    #{ Figures
    def get_axes(self,ident=None):
        """
        Return an axes or list of axes that matches index OR title
        
        @param ident: index or title of the desired axes
        @type i: int or str
        @return: axes
        @rtype: plotting.Axes
        """
        if ident is None:
            return self.axes

        if isinstance(ident, int):
            return self.axes[ident]
        elif isinstance(ident, str):
            for ax in self.axes:
                if ax.get_value('title')==ident:
                    return ax
        
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
        if isinstance(ident, int):
            return self.axes.pop(ident)
        elif isinstance(ident, str):
            for i,ax in reversed(self.axes):
                if ax.get_value('title')==ident:
                    self.axes.pop(i)
        
                                
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
        self.get_axes(ident).plot(self,mplfig=mplfig,mplaxes=mplaxes,location=location)
        
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
        
    def plot_meshview(self,mplfig=None,mplaxes=None,meshviewoptions=None):
        """
        Creates a mesh plot using the saved options if not overridden
        
        @param mplfig: the matplotlib figure to add the axes to, if none is given one will be created
        @type mplfig: plt.Figure()
        @param mplaxes: the matplotlib axes to plot to (overrides mplfig)
        @type mplaxes: plt.axes.Axes()
        @param meshviewoptions: the options for the mesh, will default to saved options
        @type meshviewoptions: parameterSet
        """
        if self.select_time is not None:
            self.system.set_time(self.select_time)
        
        if mplfig is None:
            if mplaxes is None: # no axes provided
                axes = plt.axes()
            else: # use provided axes
                axes = mplaxes
            
        else:
            axes = mplfig.add_subplot(111)
        
        po = self.plot_meshviewoptions if meshviewoptions is None else meshviewoptions
        #~ print "***", po['cmap'] if po['cmap'] is not 'None' else None
        lims, vrange, p = observatory.image(self.system, ref=po['ref'], context=po['context'], select=po['select'], background=po['background'], ax=axes)
        axes.set_xlim(lims['xlim'])
        axes.set_ylim(lims['ylim'])       
        
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
    
    def save(self,filename,purge_signals=True):
        """
        Save a class to an file.
        Will automatically purge all signals attached through bundle
        """
        if purge_signals:
            #~ self_copy = self.copy()
            self.purge_signals()
            
        # remove user settings
        settings = self.usersettings
        self.usersettings = None
        
        # pickle
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved bundle to file {} (pickle)'.format(filename))
        
        # reset user settings
        self.usersettings = settings
        
        # call set_system so that purged (internal) signals are reconnected
        self.set_system(self.system)
    
    #}
    
class Axes(object):
    """
    Class representing a collection of plot commands for a single axes
    """
    def __init__(self,**kwargs):
        """
        Initialize an axes

        all kwargs will be added to the plotting:axes ParameterSet
        it is suggested to at least initialize with a category (lc,rv,etc) and title
        """
        #~ super(Axes, self).__init__()
        
        self.axesoptions = parameters.ParameterSet(context="plotting:axes")
        #~ self.mplaxes = {} # mplfig as keys, data axes as values
        #~ self.mplaxes_sel = {} # mplfig as keys, overlay as values
        self.plots = []
        
        self.phased = False
        self.period = None
        
        for key in kwargs.keys():
            self.set_value(key, kwargs[key])
            
    def keys(self):
        return self.axesoptions.keys()
        
    def values(self):
        return self.axesoptions.values()
        
    def add_plot(self,plotoptions=None,**kwargs):
        """
        Add a new plot command to this axes
        
        kwargs will be applied to plotoptions ParameterSet
        
        @param plotoptions: options for the new plot
        @type plotoptions: ParameterSet
        """
        if plotoptions is None:
            plotoptions = parameters.ParameterSet(context="plotting:plot")
        for key in kwargs.keys():
            plotoptions.set_value(key, kwargs[key])
        self.plots.append(plotoptions)
        
    def remove_plot(self,i):
        """
        Remove (permanently) the plot at the given index from the axes
        To deactivate the plot but keep the settings use get_plot(i).set_value('active',False)
        
        @param i: index of plot
        @type i: int
        @return: the remove plotoptions
        @rtype: ParameterSet
        """
        return self.plots.pop(i)
        
    def get_plot(self,i=None):
        """
        Return a given plot by index
        
        @param i: index of plot, will return list if None given
        @type i: int
        @return: the desired plotoptions
        @rtype: ParameterSet        
        """
        if i is None:
            return self.plots
        else:
            return self.plots[i]
            
    def get_dataset(self,plot,bundle):
        """
        return the dataset attached to a given plotting parameterset
        
        @param plot: the plotoptions or index of the plot
        @type plot: plotoptions or int
        @param bundle: the bundle
        @type bundle: Bundle
        @return: the dataset attached to the plotoptions
        @rtype: ParameterSet
        """
        if plot in self.plots: #then we already have the plot
            plotoptions = plot
        else:
            plotoptions = self.get_plot(plot)
            
        obj = bundle.get_object(plotoptions['objref'])
        if plotoptions['type'][-3:]=='syn' and hasattr(obj,'bodies'):
            dataset = obj.get_synthetic(ref=plotoptions['dataref'])
        else:
            dataset,ref = obj.get_parset(type=plotoptions['type'][-3:], context=plotoptions['type'], ref=plotoptions['dataref'])
        return dataset
        
    def get_value(self,key):
        """
        Get a value from the axesoptions
        Same as axes.axesoptions.get_value()
        
        @param key: the name of the parameter
        @type key: str
        @return: the parameter
        """
        return self.axesoptions.get_value(key)
    
    def set_value(self,key,value):
        """
        Set a value in the axesoptions
        Same as axes.axesoptions.set_value()
        
        @param key: the name of the parameter
        @type key: str
        @param value: the new value
        """
        self.axesoptions.set_value(key,value)

    def savefig(self,system,fname):
        """
        Save the plot to a file
        
        @parameter system: the phoebe system
        @type system: System
        @parameter fname: filename of output image
        @type fname: str
        """
        if self.mplaxes is None:
            self.plot(system)
        self.mplaxes.savefig(fname)
        return
            
    def plot(self,bundle,mplfig=None,mplaxes=None,location=None,*args,**kwargs):
        """
        Plot all the children plots on a single axes
        
        @parameter bundle: the phoebe bundle
        @type bundle: Bundle
        @parameter mplfig: the matplotlib figure to add the axes to, if none is give one will be created
        @type mplfig: plt.Figure()
        @parameter mplaxes: the matplotlib axes to plot to (overrides mplfig, axesoptions will not apply)
        @type mplaxes: plt.axes.Axes()
        @parameter location: the location on the figure to add the axes
        @type location: str or tuple        
        """
        
        system = bundle.get_system()
        
        # get options for axes
        ao = {}
        for key in self.axesoptions.keys():
            if key not in ['location', 'active', 'category', 'xaxis', 'yaxis']:
                ao[key] = self.axesoptions.get_value(key)
                
        # override anything set from kwargs
        for key in kwargs:
            ao[key]=kwargs[key]

            
        # control auto options
        xaxis, yaxis = self.axesoptions['xaxis'], self.axesoptions['yaxis']
        if xaxis == 'auto':
            xaxis = 'time'
        if yaxis == 'auto':
            if self.axesoptions['category'] == 'lc':
                yaxis = 'flux'
            elif self.axesoptions['category'] == 'rv':
                yaxis = 'rv'
            elif self.axesoptions['category'] == 'sp':
                yaxis = 'wavelength'
            elif self.axesoptions['category'] == 'etv':
                yaxis = 'ETV'
        if ao['xlabel'] == 'auto':
            ao['xlabel'] = xaxis
        if ao['ylabel'] == 'auto':
            ao['ylabel'] = yaxis
        if ao['xlim'] == (None, None):
            ao.pop('xlim')
        if ao['ylim'] == (None, None):
            ao.pop('ylim')
        if ao['xticks'] == ['auto']:
            ao.pop('xticks')
        if ao['yticks'] == ['auto']:
            ao.pop('yticks')
        if ao['xticklabels'] == ['auto']:
            ao.pop('xticklabels')
        if ao['yticklabels'] == ['auto']:
            ao.pop('yticklabels')
            
        # add the axes to the figure
        if location is None:
            location = self.axesoptions['location'] # location not added in ao
            
        # of course we may be trying to plot to a different figure
        # so we'll override this later if that is the case 
        # just know that self.mplaxes is just to predict which axes
        # to use and not necessarily the correct axes
        
        if mplfig is None:
            if location == 'auto':  # then just plot to an axes
                if mplaxes is None: # no axes provided
                    axes = plt.axes(**ao)
                else: # use provided axes
                    axes = mplaxes
            else:
                mplfig = plt.Figure()
            
        if location != 'auto':
            if isinstance(location, str):
                axes = mplfig.add_subplot(location,**ao)
            else:
                axes = mplfig.add_subplot(location[0],location[1],location[2],**ao)
        else:
            if mplfig is not None:
                axes = mplfig.add_subplot(111,**ao)
        
        # get phasing information
        xaxis = self.axesoptions.get_value('xaxis')
        phased = xaxis.split(':')[0]=='phase'
        
        # now loop through individual plot commands
        for plotoptions in self.plots:
            #~ print "***", plotoptions['dataref'], plotoptions['objref'], plotoptions['type'], plotoptions['active']
            if not plotoptions['active']:
                continue

            # copied functionality from bundle.get_object (since we don't want to import bundle)
            obj = bundle.get_object(plotoptions['objref'])
            
            if plotoptions['type'][-3:]=='obs':
                dataset = bundle.get_obs(objref=plotoptions['objref'],dataref=plotoptions['dataref'])[0]
            else:
                dataset = bundle.get_syn(objref=plotoptions['objref'],dataref=plotoptions['dataref'])[0]
                
            if len(dataset['time'])==0: # then empty dataset
                continue
                
            po = {}
            for key in plotoptions.keys():
                if key == 'errorbars':
                    errorbars = plotoptions.get_value(key)
                if key not in ['dataref', 'objref', 'type', 'active','errorbars']:
                    po[key] = plotoptions.get_value(key)
                    
            #if linestyle has not been set, make decision based on type
            if po['linestyle'] == 'auto':
                po['linestyle'] = 'None' if plotoptions['type'][-3:] == 'obs' else '-'
            if po['marker'] == 'auto':
                po['marker'] = 'None' if plotoptions['type'][-3:] == 'syn' else '.'
            #if color has not been set, make decision based on type
            if po['color'] == 'auto':
                po['color'] = 'k' if plotoptions['type'][-3:] == 'obs' else 'r' 
                
            # remove other autos
            for key in po.keys():
                if po[key]=='auto':
                    po.pop(key)
           
            if phased and len(xaxis.split(':')) > 1:
                #~ orbit = self._get_orbit(xaxis.split(':')[1],system) 
                orbit = bundle.get_orbit(xaxis.split(':')[1])
            elif hasattr(obj,'params') and 'orbit' in obj.params.keys():
                orbit = obj.params['orbit']
            else:
                #~ orbit = self._get_orbit(plotoptions['objref'],system)
                orbit = bundle.get_orbit(plotoptions['objref'])
            period = orbit.get_value('period')
            self.period = period
            self.phased = phased
            #~ print "** period", period
            #~ print "** phased", phased     
            #~ print "** type", plotoptions['type']
            
            # call appropriate plotting command
            if plotoptions['type']=='lcobs':
                artists,obs = plotting.plot_lcobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='lcsyn':
                artists,obs,pblum,l3 = plotting.plot_lcsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            elif plotoptions['type']=='rvobs':
                artists,obs = plotting.plot_rvobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='rvsyn':
                artists,obs,l3 = plotting.plot_rvsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            elif plotoptions['type']=='etvobs':
                artists,obs = plotting.plot_etvobs(obj, ref=plotoptions['dataref'], ax=axes, errorbars=errorbars, phased=phased, period=period, **po)
            elif plotoptions['type']=='etvsyn':
                artists,obs = plotting.plot_etvsyn(obj, ref=plotoptions['dataref'], ax=axes, phased=phased, period=period, **po)
            else:
                artists,obs = [],[]
        
        mplfig.data_axes = axes
        mplfig.sel_axes = axes.twinx()

        self.plot_select_time(bundle.select_time,mplfig=mplfig)
                
    def plot_select_time(self,time,mplfig):
        mplaxes_sel = mplfig.sel_axes
        mplaxes_sel.cla()
        mplaxes_sel.set_yticks([])
            
        if time is not None:
            if self.phased:
                time = (time % self.period) / self.period
            
            xlim = mplfig.data_axes.get_xlim()
            if time < xlim[1] and time > xlim[0]:
                mplaxes_sel.axvline(time, color='r')

    
def load(filename):
    """
    Load a class from a file.
    
    @return: Bundle saved in file
    @rtype: Bundle
    """
    ff = open(filename,'r')
    bundle = pickle.load(ff)
    ff.close()
    # for set_system to update all signals, etc
    bundle.set_system(bundle.system)
    # load this users settings into the bundle
    bundle.set_usersettings()
    return bundle
