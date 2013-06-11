"""
Top level class of Phoebe.
"""
import pickle
import logging
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

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
        self.system = system
        self.mpi = mpi
        self.compute = OrderedDict()
        self.fitting = OrderedDict()
        self.feedback = OrderedDict()
        self.figs = OrderedDict()
        
        self.pool = OrderedDict()
        self.attached_signals = []
        
        self.add_fitting(fitting)
        self.axes = []
        if isinstance(axes, list):
            self.axes = axes
        if isinstance(axes, dict):
            for key in axes.keys():
                self.add_axes(axes[key], key)
        self.add_compute(compute)
        
    #{ System
    def set_system(self,system):
        """
        Change the system

        @param system: the new system
        @type system: System
        """
        if system is None:  return None
        self.system = system 
        
    def get_system(self):
        """
        Return the system.
        
        @return: the attached system
        @rtype: Body
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
        all_types = ['label','obj','ps','nchild','mask']
        
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
        
        # label,ps,nchild are different whether item is body or bodybag
        if hasattr(item, 'bodies'):
            itemps = self.get_orbit(item)
            struc['ps'].append(itemps)
            struc['nchild'].append('2') # should not be so strict
        else:
            itemps = self.get_component(item)
            struc['ps'].append(itemps)
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
            new = self.get_system_structure(return_type=['label','obj','ps','nchild','mask'],flat=flat,top_level=child,**kwargs)
            #~ for i in range(len(new)):
                #~ if len(new[i]) == 1:
                    #~ new[i] = new[i][0]
            for i,typ in enumerate(all_types):
                #~ struc[typ][-1].append(new[i])
                struc[typ][-1]+=new[i]
                #~ struc[typ].append(new[i])
                
        #~ print "**", struc['label']
            
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
        if 'star' in params.keys():
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
               
    def get_parent(self,objectname):
        """
        retrieve the parent of an item in a hierarchical structure
        
        @param objectname: label of the child object
        @type objectname: str
        @return: the parent bodybag
        @rtype: ParameterSet        
        """
        raise NotImplementedError
        return None
        
    def get_children(self,objectname):
        """
        retrieve the children of an item in a hierarchical structure
        
        @param objectname: label of the parent object
        @type objecname: str
        @return: list of children objects
        @rtype: list of Bodies/BodyBags
        """
        obj = self.get_object(objectname)
        if hasattr(obj,'bodies'):
            return self.get_object(objectname).bodies
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
        
    #}  
    
    #{ Loading Data
    def load_data(self,context,filename,passband=None,columns=None,components=None,name=None):
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
        
        if context=='rvobs':
            output = datasets.parse_rv(filename,columns=columns,components=components,full_output=True,**{'passband':passband, 'ref': name})
        elif context=='lcobs':
            output = datasets.parse_lc(filename,columns=columns,components=components,full_output=True,**{'passband':passband, 'ref': name})
        else:
            print("only lc and rv currently implemented")
        
        for objectlabel in output.keys():      
            # get the correct component (body)
            comp = self.get_object(objectlabel)
            
            # unpack all datasets and parametersets
            dss = output[objectlabel][0]
            pss = output[objectlabel][1]
            
            # attach
            for ds in dss:    
                #~ ds.load()
                comp.add_obs(ds)
            for ps in pss:
                comp.add_pbdeps(ps)
                
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
                
    def get_obs(self,objectname=None,ref=None):
        """
        get an observables dataset by the object its attached to and its label
        if objectname and ref are given, this will return a single dataset
        if either or both are not give, this will return a list of all datasets matching the search
        
        @param objectname: name of the object the dataset is attached to
        @type objectname: str
        @param ref: ref (name) of the dataset
        @type ref: str
        @return: dataset
        @rtype: parameterSet
        """
        if objectname is None:
            # then search the whole system
            return_ = []
            for objname in self.get_system_structure(return_type='obj',flat=True):
                parsets = self.get_obs(objname,ref=ref)
                for parset in parsets:
                    if parset not in return_:
                        return_.append(parset)
            return return_
            
        # now we assume we're dealing with a single object
        obj = self.get_object(objectname)
        
        if ref is not None:
            # then search for the ref by name/index
            parset = obj.get_parset(type='obs',ref=ref)
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
            
    def get_syn(self,objectname=None,ref=None):
        """
        get a synthetic dataset by the object its attached to and its label
        if objectname and ref are given, this will return a single dataset
        if either or both are not give, this will return a list of all datasets matching the search
        
        @param objectname: name of the object the dataset is attached to
        @type objectname: str
        @param ref: ref (name) of the dataset
        @type ref: str
        @return: dataset
        @rtype: parameterSet
        """
        counter1 = 0 
        counter2 = 0 
        if objectname is None:
            # then search the whole system
            return_ = []
            for objname in self.get_system_structure(return_type='obj',flat=True):
                #~ counter1+=1
                #~ print("Counter1 = {}".format(counter1))
                parsets = self.get_syn(objname,ref=ref)
                for parset in parsets:
                    #~ counter2 +=1
                    #~ print("Counter1 = {} - Counter2 = {}".format(counter1,counter2))
                    if parset not in return_:
                        return_.append(parset)
            return return_
            
        # now we assume we're dealing with a single object
        obj = self.get_object(objectname)
        
        if ref is not None:
            # then search for the ref by name/index
            #~ print ref
            parset = obj.get_synthetic(ref=ref)
            if parset != None:
                return [parset]
            return []
        else:
            # then loop through indices until there are none left
            return_ = []
            parset,i = 0,0
            while parset != [] and parset != None:
                parset = obj.get_synthetic(ref=i)
                i+=1
                if parset != [] and parset != None:
                    return_.append(parset)
            return return_
        
    #}
    
    #{ Compute
    def add_compute(self,compute,label=None):
        """
        Add a new compute parameterSet
        
        @param compute: compute parameterSet
        @type compute:  None, parameterSet or list of ParameterSets
        @param label: name of parameterSet
        @type label: None, str, or list of strs
        """
        if compute is None: return None
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
        return self.compute[label]
        
    def remove_compute(self,label):
        """
        Remove a given compute parameterSet
        
        @param label: name of compute parameterSet
        @type label: str
        """
        if label is None:    return None
        self.compute.pop(label)        
    
    def run_compute(self,label=None,mpi=True):
        """
        Convenience function to run observatory.observe
        
        @param label: name of one of the compute parameterSets stored in bundle
        @type label: str
        @param mpi: whether to use mpi (will use stored options)
        @type mpi: bool
        """
        self.system.fix_mesh()
        self.system.fix_mesh()
        if label is None:
            options = parameters.ParameterSet(context='compute')
        else:
            if label not in self.compute:
                return KeyError
            options = self.compute[label]
        # clear all previous models and create new model
        self.system.clear_synthetic() 
        #~ observatory.observe(self.system,lc=True,rv=True,sp=True,pl=True,im=True,mpi=self.mpi if mpi else None,**options)
        observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)

        # dataset = body.get_synthetic(type='lcsyn',ref=lcsyn['ref'])
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
        return self.fitting[label]
        
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
        return self.feedback[label]
               
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
    def save(self,filename):
        """
        Save a class to an file.
        Will automatically purge all signals attached through bundle
        """
        self.purge_signals()
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved bundle to file {} (pickle)'.format(filename))
    
    #}
    
def load(filename):
    """
    Load a class from a file.
    
    @return: Bundle saved in file
    @rtype: Bundle
    """
    ff = open(filename,'r')
    myclass = pickle.load(ff)
    ff.close()
    return myclass
