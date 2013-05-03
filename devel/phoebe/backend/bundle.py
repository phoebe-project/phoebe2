"""
Top level class of Phoebe.
"""
import pickle
import logging
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from phoebe.utils import callbacks
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.backend import fitting
from phoebe.backend import observatory

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
        self.figs = OrderedDict()
        
        self.pool = OrderedDict()
        self.attached_signals = []
        
        self.add_fitting(fitting)
        self.axes = {}
        if isinstance(axes, list):
            self.add_axes(axes)
        if isinstance(axes, dict):
            self.axes = axes
        self.add_compute(compute)
        
    #{ System        
    def get_system(self):
        """
        Return the system.
        
        @return: the attached system
        @rtype: Body
        """
        return self.system
        
    def set_system(self,system):
        """
        Change the system

        @param system: the new system
        @type system: System
        """
        if system is None:  return None
        self.system = system 
                          
    def get_object(self,objectname):
        """
        search for an object inside the system structure
        
        @param objectname: label or __name__ for the desired object
        @type objectname: str
        @return: the object
        @rtype: ParameterSet
        """
        #~ for item,obj in self.system.walk_all():
            #~ if hasattr(obj,'__name__'):
                #~ if obj.__name__ == objectname:
                    #~ return obj
            #~ try:    
                #~ if obj.params['component']['label'] == objectname:
                    #~ return obj
                #~ if obj.params['orbit']['label'] == objectname:
                    #~ return obj.params['orbit']
            #~ except AttributeError:
                #~ pass
        #~ if hasattr(self.system, '__name__'):
            #~ if self.system.__name__ == objectname:
                #~ return self.system
        #~ return None
        for path,item in self.system.walk_all():
            if path[-1] == objectname:
                return item
        return None
        
    def get_component(self,objectname):
        """
        retrieve the ParameterSet for a component by name
        
        @param objectname: label or __name__ for the desired object
        @type objectname: str
        @return: the ParameterSet of the component
        @rtype: ParameterSet
        """
        return self.get_object(objectname=objectname).params['component']
    
    def get_orbit(self,objectname):
        """
        retrieve the ParameterSet for a orbit by name
        
        @param objectname: label or __name__ for the desired object
        @type objectname: str
        @return: the ParameterSet of the orbit
        @rtype: ParameterSet
        """
        return self.get_object(objectname=objectname).params['orbit']
        
    def get_mesh(self,objectname):
        """
        retrieve the ParameterSet for a mesh by name
        
        @param objectname: label or __name__ for the desired object
        @type objectname: str
        @return: the ParameterSet of the mesh
        @rtype: ParameterSet
        """
        return self.get_object(objectname=objectname).params['mesh']
        
    def get_parent(self,objectname):
        """
        retrieve the parent of an item in a hierarchical structure
        
        @param objectname: label of the child object
        @type objectname: str
        @return: the parent bodybag
        @rtype: ParameterSet        
        """
        
        # allow for passing the object itself instead of the label
        if isinstance(objectname, str):
            obj = self.get_object(objectname)
        else:
            obj = objectname
            
        # walk through all items, when we find a match return the item one above
        walk = list(self.system.walk_all())
        for i,(path,item) in enumerate(walk):
            if item == obj:
                if len(path) <=2: # then we won't be able to access system so just return it
                    return self.system
                else:
                    return walk[i-1][1] 
                    
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
        
    def add_parent(self,objectname,parent):
        """
        add a parent to an existing item in the hierarchical system structure
        
        @param objectname: label of the child item
        @type objectname: str
        @param parent: the new parent
        @type parent: BodyBag        
        """
        obj = self.get_object(objectname)
        oldparent = self.get_parent(objectname)
        
        # this isn't going to work because we can't initialize a BodyBag with no arguments/children
        parent.append(obj)
        oldparent.append(parent)
        
    def add_child(self,objectname,child):
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
        else:
            print "only RV is currently supported"
            return
        
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
        
        @param plotname: name of compute parameterSet
        @type plotname: str
        """
        if label is None:    return None
        self.compute.pop(label)        
    
    def observe(self,label=None,mpi=False):
        """
        Convenience function to run observatory.observe
        
        @param label: name of one of the compute parameterSets stored in bundle
        @type label: str
        @param mpi: mpi parameterSet
        @type mpi: None or parameterSet
        """
        if label is None:
            options = parameters.ParameterSet(context='compute')
        else:
            if label not in self.compute:
                return KeyError
            options = self.compute[label]
        # clear all previous models and create new model
        self.system.clear_synthetic() 
        observatory.observe(self.system,times=np.arange(0,100,20.),lc=True,rv=True,sp=True,pl=True,im=True,mpi=self.mpi if mpi else None,**options)

        # dataset = body.get_synthetic(type='lcsyn',ref=lcsyn['ref'])
    #}
            
    #{ Fitting
    def add_fitting(self,fitresults):
        """
        Add fitting results to the bundle.
        
        @param fitresults: results from the fitting
        @type fitresults: None, parameterSet or list of ParameterSets
        """
        #-- if we've nothing to add, then just quit
        if fitresults is None: return None
        #-- fitresults should be a list, not one result:
        if not isinstance(fitresults,list):
            fitresults = [fitresults]
        #-- then add the results to the bundle.
        for fitresult in fitresults:
            label = fitresult['label']
            self.fitting[label] = fitresult
    
    def get_fitting(self,label):
        """
        """
        return None
        
    def remove_fitting(self,label):
        """
        """
        self.fitting.pop(label)
    
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
            allfitparams = [self.fitting[label]]
        else:
            allfitparams = self.fitting.values()[::-1]
        #-- take the last fitting parameterSet that represents an mcmc
        for fitparams in allfitparams:
            if fitparams.context.split(':')[-1] in ['pymc','emcee']:
                fitparams['iter'] += extra_iter
                feedback = fitting.run_emcee(self.system,params=self.compute,
                                    fitparams=fitparams,mpi=self.mpi)
                break
    #}

    #{ Figures
    def get_axes(self,label=None,index=None):
        """
        Return an axes
        
        @param label: the name for the desired axes
        @type label: str
        @return: axes
        @rtype:
        """
        if label is None:
            return self.axes
        else:
            if index is None:
                return self.axes[label]
            else:
                return self.axes[label][index]
        
        
    def add_axes(self,axes,label='default'):
        """
        Add a new axes with a set of plotoptions
        
        @param axes: a axes to be plotted on a single axis
        @type axes:
        @param label: name for the current plot
        @type label: str
        """
        if label in self.axes.keys():
            logger.error("cannot add new axes: \'{}\' alreadly in axes".format(label))
            return None
        self.axes[label]=axes
        
    def remove_axes(self,label):
        """
        Remove a given axes
        
        @param label: name of axes
        @type label: str
        """
        self.axes.pop(label)
        
    def rename_axes(self,oldlabel,newlabel):
        """
        Rename a given axes
        
        @param oldlabel: previous label of axes
        @type oldlabel: str
        @param newlabel: new label of axes
        @type newlabel: str
        """
        self.axes[newlabel] = self.axes.pop(oldlabel)
                                
    def plot_axes(self,label='default',mplaxes=None,*args,**kwargs):
        """
        Create a defined axes
        
        @param label: label of the axes
        @type label: str
        @param axes: matplotlib axes to use for plotting
        @type axes
        """
        self.get_axes(label).plot(self.get_system(),mplaxes,args,kwargs)
        
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
