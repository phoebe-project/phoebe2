"""
Top level class of Phoebe.
"""
import pickle
import logging
import numpy as np
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import copy

from phoebe.utils import callbacks, utils
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.backend import fitting, observatory, plotting

logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())

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
        self.feedback = OrderedDict()
        self.figs = OrderedDict()
        
        self.pool = OrderedDict()
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
        
        self.settings = {'add_version_on_compute': False}
        
        self.set_system(system) # will handle all signals, etc
        
    #{ Settings
    def set_setting(self,key,value):
        self.settings[key] = value
        
    def get_setting(self,key):
        return self.settings[key]
        
    def load_settings(self,filename):
        raise NotImplementedError  
        
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
        #~ print "*** bundle.set_system"
        self.system = system 
        if system is None:  return None
        
        # check to see if in versions, and if so set versions_curr_i
        versions = [v['system'] for v in self.versions]
        if system in versions:
            i = [v['system'] for v in self.versions].index(system)
        else:
            i = None
        self.versions_curr_i = i
        
        # connect signals
        self.attached_signals_system = []
        for label in self.get_system_structure(return_type='label',flat=True):
            ps = self.get_ps(label)
            for key in ps.keys():
                #~ print "*** attaching signal %s:%s" % (label,key)
                param = ps.get_parameter(key)
                self.attach_signal(param,'set_value',self._on_param_changed)
                #~ if param not in self.attached_signals_system:
                self.attached_signals_system.append(param)
        
        # these might already be attached?
        self.attach_signal(self,'load_data',self._on_param_changed)
        self.attach_signal(self,'enable_obs',self._on_param_changed)
        self.attach_signal(self,'disable_obs',self._on_param_changed)
                
    def _on_param_changed(self,param):
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

        
    #}  
    
    #{ Versions
    
    def add_version(self,name=None):
        """
        Add the current snapshot of the system as a new version entry
        Generally this is best to be handled by setting add_version=True in bundle.run_compute
        
        @param name: name of the version (defaults to current timestamp)
        @type name: str        
        """
        system = self.get_system()
        # purge any signals attached to system before copying
        callbacks.purge_signals(system)
        system.signals={}
        self.purge_signals(self.attached_signals_system)
        version = {'system':system.copy()} #this will probably fail if there are signals attached
        version['date_created'] = datetime.now()
        version['name'] = name if name is not None else str(version['date_created'])
        
        self.versions.append(version)
        
        # call set_system to reattach signals and set versions_curr_i
        self.set_system(system)
        
        # update versions_curr_i
        #~ self.versions_curr_i = len(self.versions)-1


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
        else:
            print("only lc and rv currently implemented")
        
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
            parset,i = 0,0
            while parset != (None,None):
                parset = obj.get_parset(type='obs',ref=i)
                i+=1
                if parset != (None,None):
                    return_.append(parset[0])
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
            parset,i = 0,0
            while parset != [] and parset != None:
                parset = obj.get_synthetic(ref=i, cumulative=True)
                i+=1
                if parset != [] and parset != None:
                    return_.append(parset)
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
    
    def run_compute(self,label=None,mpi=True,im=False,add_version=None):
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
        """
        if add_version is None:
            add_version = self.settings['add_version_on_compute']
        
        #~ self.system.fix_mesh()
        try:
            self.system.set_time(0)
        except:
            pass
        self.system.fix_mesh()
        
        if label is None:
            options = parameters.ParameterSet(context='compute')
        else:
            if label not in self.compute:
                return KeyError
            options = self.compute[label]
        # clear all previous models and create new model
        self.system.clear_synthetic()
        
        if options['time']=='auto' and not im:
            observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
        else:
            observatory.observe(self.system,options['time'],lc=True,rv=True,sp=True,pl=True,
                extra_func=[observatory.ef_binary_image] if im else [],
                extra_func_kwargs=[dict(select='teff',cmap='blackbody')] if im else [],
                mpi=self.mpi if mpi else None,**options
                )
                
        if add_version is not False:
            self.add_version(name=None if add_version==True else add_version)
            
        self.system.uptodate = label

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
        
    def run_fitting(self,computelabel,fittinglabel,mpi=True):
        """
        Run fitting for a given fitting parameterSet
        and store the feedback
        
        @param computelabel: name of compute parameterSet
        @param computelabel: str
        @param fittinglabel: name of fitting parameterSet
        @type fittinglabel: str
        @param mpi: whether mpi is enabled (will use stored options)
        @type mpi: bool
        """
        feedback = fitting.run(self.system, params=self.get_compute(computelabel), fitparams=self.get_fitting(fittinglabel), mpi=self.mpi if mpi else None)
        self.add_feedback(feedback)
    
    def add_feedback(self,feedback):
        """
        Add fitting results to the bundle.
        
        @param feedback: results from the fitting
        @type feedback: None, parameterSet or list of ParameterSets
        """
        #-- if we've nothing to add, then just quit
        if feedback is None: return None
        #-- feedbacks should be a list, not one result:
        if not isinstance(feedback,list):
            feedback = [feedback]
        #-- then add the results to the bundle.
        for f in feedback:
            label = f['label']
            self.feedback[label] = f
    
    def get_feedback(self,label):
        """
        Get fitting results by name
        
        @param label: name of the fitting results
        @type label: str
        @return: a feedback parameter set
        @rtype: parameterSet
        """
        if label in self.feedback.keys():
            return self.feedback[label]
        else:
            return None
               
    def remove_feedback(self,label):
        """
        Remove a given fitting feedback
        
        @param label: name of the fitting results
        @type label: str
        """
        self.feedback.pop(label)
    
    def accept_feedback(self,label):
        """
        Accept fitting results and apply to system
        
        @param label: name of the fitting results
        @type label: str
        """
        fitting.accept_fit(self.system,self.feedback[label])
        
    def continue_mcmc(self,label=None,extra_iter=10):
        """
        Continue an MCMC chain.
        
        If you don't provide a label, the last MCMC run will be continued.
        
        @param label: label of the MCMC parameterSet to continue
        @type label: str
        @param extra_iter: extra number of iterations
        @type extra_iter: int
        """
        if label is not None:
            allfitparams = [self.feedback[label]]
        else:
            allfitparams = self.feedback.values()[::-1]
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
            axes = plotting.Axes()
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
        @parameter mplfig: the matplotlib figure to add the axes to, if none is give one will be created
        @type mplfig: plt.Figure()
        @parameter mplaxes: the matplotlib axes to plot to (overrides mplfig)
        @type mplaxes: plt.axes.Axes()
        @parameter location: the location on the figure to add the axes
        @type location: str or tuple  
        """
        self.get_axes(ident).plot(self.get_system(),mplfig=mplfig,mplaxes=mplaxes,location=location)
        
    def plot_orbit(self,times=None):
        """
        
        """
        raise NotImplementedError
        
        # create times list if not given
        if times is None:
            times = np.linspace(self.system.get_value('t0'),self.system.get_value('t0')+self.system.get_value('period'),100)
        
        # create hierarchical dictionary for passing to get_hierarchical_orbits
        
        
        # get orbits
        
        
        # plot
        for body in output:
            plt.plot(body[0][0], body[0][1])
        plt.show()
        
        
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
        for param in signals:
            callbacks.purge_signals(param)
        if signals == self.attached_signals:
            self.attached_signals = []
            
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
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved bundle to file {} (pickle)'.format(filename))
        
        # call set_system so that purged (internal) signals are reconnected
        self.set_system(self.system)
    
    #}
    
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
    return bundle
