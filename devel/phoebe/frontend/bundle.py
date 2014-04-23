"""
Top level class of Phoebe.

You can run Phoebe2 in Phoebe1 compatibility mode ('legacy' mode), if you start
from a legacy parameter file.

>>> mybundle = Bundle('legacy.phoebe')
>>> mybundle.run_compute(label='from_legacy')
>>> print(mybundle.get_logp())

>>> mybundle.plot_obs('lightcurve_0', fmt='ko')
>>> mybundle.plot_obs('primaryrv_0', fmt='ko')
>>> mybundle.plot_obs('secondaryrv_0', fmt='ro')

>>> mybundle.plot_syn('lightcurve_0', 'k-', lw=2)
>>> mybundle.plot_syn('primaryrv_0', 'k-', lw=2)
>>> mybundle.plot_syn('secondaryrv_0', 'r-', lw=2)

Basic class:

.. autosummary::

    Bundle
    
Helper functions and classes:

.. autosummary::

    Version
    Feedback
    run_on_server
    load
    guess_filetype
    
"""
import pickle
import logging
import functools
import inspect
import textwrap
import numpy as np
from collections import OrderedDict
from datetime import datetime
from time import sleep
import matplotlib.pyplot as plt
import copy
import os
import re
import readline
import json
from phoebe.utils import callbacks, utils, plotlib, coordinates, config
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.parameters import create
from phoebe.backend import fitting, observatory, plotting
from phoebe.backend import universe
from phoebe.atmospheres import limbdark
from phoebe.io import parsers
from phoebe.dynamics import keplerorbit
from phoebe.frontend.usersettings import Settings
from phoebe.frontend.common import Container, rebuild_trunk
from phoebe.frontend.figures import Axes
import phcompleter

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
        Real parser.
        """
        # first we need to reconstruct the arguments passed to the original function
        callargs = inspect.getcallargs(fctn,bundle,*args,**kwargs)
        dump = callargs.pop('self')
        callargstr = ','.join(["%s=%s" % (key, "\'%s\'" % callargs[key]\
            if isinstance(callargs[key],str) else callargs[key]) for key in callargs.keys()])

        # determine if the function is supposed to be run on a server
        servername = kwargs['server'] if 'server' in kwargs.keys() else None

        # is_server is a kwarg that will be True in the script running on a server
        # itself - this just simply avoids the server resending to itself, without
        # having to remove server from the kwargs
        is_server = kwargs.pop('is_server',False)

        # now, if this function is supposed to run on a server, let's prepare
        # the script and files and submit the job.  Otherwise we'll just return
        # with the original function as called.
        if servername is not False and servername is not None and not is_server:
            # first we need to retrieve the requested server using get_server
            server =  bundle.get_server(servername)
            
            # servers can be local (only have mpi settings) - in these cases we
            # don't need to submit a script, and the original function can handle
            # retrieving and applying the mpi settings
            if server.is_external():
                
                # now we know that we are being asked to run on an external server,
                # so let's do a quick check to see if the server seems to be online
                # and responding.  If it isn't we'll provide an error message and abort
                if server.check_status():
                    
                    # The external server seems to be responding, so now we need to
                    # prepare all files and the script to run on the server
                    mount_dir = server.server_ps.get_value('mount_dir')
                    server_dir = server.server_ps.get_value('server_dir')
                    
                    # Copy the bundle to the mounted directory, without removing
                    # usersettings (so that the server can have access to everything)
                    logger.info('copying bundle to {}'.format(mount_dir))
                    timestr = str(datetime.now()).replace(' ','_')
                    in_f = '%s.bundle.in.phoebe' % timestr
                    out_f = '%s.bundle.out.phoebe' % timestr
                    script_f = '%s.py' % timestr
                    bundle.save(os.path.join(mount_dir,in_f),save_usersettings=True) # might have a problem here if threaded!!
                    
                    # Now we write a script file to the mounted directory which will
                    # load the saved bundle file, call the original function with the
                    # same arguments, and save the bundle to a new file
                    #
                    # we'll also provide some status updates so that we can see when
                    # the script has started/failed/completed
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
    

        

def walk(mybundle):
    # do we still use this???
    
    for val,path in utils.traverse_memory(mybundle,
                                     list_types=(Bundle, universe.Body, list,tuple),
                                     dict_types=(dict, ),
                                     parset_types=(parameters.ParameterSet, ),
                                     get_label=(universe.Body, ),
                                     get_context=(parameters.ParameterSet, ),
                                     skip=()):
            
            # First one is always root
            path[0] = str(mybundle.__class__.__name__)
            
            
            # All is left is to return it
            
            yield path, val
    
    

class Bundle(Container):
    """
    Class representing a collection of systems and stuff related to it.
    
    You can initiate a bundle in different ways:
    
        1. Via a PHOEBE 2.0 file in JSON format::
        
            mybundle = Bundle()
            mybundle.save('newbundle.phoebe')
            mybundle = Bundle('newbundle.phoebe')
    
        2. Via a Body or BodyBag::
        
            mysystem = phoebe.create.from_library('V380_Cyg', create_body=True)
            mybundle = Bundle(mysystem)
        
        3. Via the library::
        
            mybundle = Bundle('V380_Cyg')
            
        4. Via a pickled system::
        
            mysystem = phoebe.create.from_library('V380_Cyg', create_body=True)
            mysystem.save('mysystem.pck')
            mybundle = Bundle('mysystem.pck')
        
        5. Via a pickled Bundle::
        
            mybundle = Bundle('V380_Cyg')
            mybundle.save('V380_Cyg.bpck')
            mybundle = Bundle('V380_Cyg.bpck')
        
        6. Via a Phoebe Legacy ASCII parameter file::
        
            mybundle = Bundle('legacy.phoebe')
    
    For more details, see :py:func:`set_system`.
    
    The interaction with a Bundle is much alike interaction with a Python
    dictionary. The following functionality is implemented and behaves as
    expected::
            
            period = mybundle['period'] # returns the value of the period if it exists, raises error if 'period' does not exist
            period = mybundle.get('period') # returns the value of the period if it exists, else returns None
            period = mybundle.get('period', default_value) # returns the value of the period if it exists, else returns default_value (whatever it is)
            keys = mybundle.keys() # returns a list of available keys
            values = mybundle.values() # returns a list of values
            
    
    **Interface**
    
    .. autosummary::
    
        phoebe.frontend.common.Container.set_value
        phoebe.frontend.common.Container.get_value
        load_data
        create_data
        plot_obs
        plot_syn
        
    **What is the Bundle?**
    
    The Bundle aims at providing a user-friendly interface to a Body or BodyBag,
    such that parameters can easily be queried or changed, data added, results
    plotted and observations computed. It does not contain any implementation of
    physics; that is all done at the Body level.
    
    **Structure of the Bundle**
    
    
    A Bundle contains:
    
        - a Body (or BodyBag), called :envvar:`system` in this context.
        - a list of compute options which can be stored and used to compute observables.
        - a list of figure axes options which can be used to recreate plots with the same options.
        
    **Outline of methods**
    
    **Input/output**
    
    .. autosummary::
        
        to_string
        load
        save
    
    **Getting system parameters**
    
    .. autosummary::
    
        phoebe.frontend.common.Container.get_ps
        phoebe.frontend.common.Container.get_parameter
        phoebe.frontend.common.Container.get_value
    
    **Attaching data**
    
    .. autosummary::
    
        load_data
        create_data
    
    **Setting and getting computational parameters**
    
    .. autosummary::
    
        phoebe.frontend.common.Container.get_compute
        phoebe.frontend.common.Container.add_compute
        phoebe.frontend.common.Container.remove_compute
        
    **Getting results**
    
    .. autosummary::
    
        get_obs
        get_syn
        
    **Plotting results**
        
    .. autosummary::
    
        plot_obs
        plot_syn
    
    """        
    def __init__(self, system=None, remove_dataref=False):
        """
        Initialize a Bundle.

        For all the different possibilities to set a system, see :py:func:`Bundle.set_system`.
        """
        
        # self.sections is an ordered dictionary containing lists of 
        # ParameterSets (or ParameterSet-like objects)

        super(Bundle, self).__init__()
        
        self.sections['system'] = [None] # only 1
        self.sections['compute'] = []
        #~ self.sections['fitting'] = []
        #~ self.sections['feedback'] = []
        #~ self.sections['version'] = []
        #~ self.sections['axes'] = []
        #~ self.sections['meshview'] = [parameters.ParameterSet(context='plotting:mesh')] # only 1
        #~ self.sections['orbitview'] = [parameters.ParameterSet(context='plotting:orbit')] # only 1
        
        # self.select_time controls at which time to draw the meshview and
        # the 'selector' on axes (if enabled)
        self.select_time = None
        
        # we need to keep track of all attached signals so that they can 
        # be purged before pickling the bundle, and can be restored
        self.signals = {}
        self.attached_signals = []
        self.attached_signals_system = [] #these will be purged when making copies of the system and can be restored through set_system
        
        # Now we load a copy of the usersettings into self.usersettings
        # Note that by default these will be removed from the bundle before
        # saving, and reimported when the bundle is loaded
        self.set_usersettings()
        
        # self.lock simply keeps track of whether the bundle is waiting
        # on a job to complete on a server.  If a lock is in place, it can
        # manually be removed by self.server_cancel()
        self.lock = {'locked': False, 'server': '', 'script': '', 'command': '', 'files': [], 'rfile': None}

        # Let's keep track of the filename whenever saving the bundle -
        # if self.save() is called without a filename but we have one in
        # memory, we'll try to save to that location.
        self.filename = None
        
        # self.settings are bundle-level settings that control defaults
        # for bundle functions.  These do not belong in either individual
        # ParameterSets or the usersettings
        self.settings = {}
        self.settings['add_version_on_compute'] = False
        self.settings['add_feedback_on_fitting'] = False
        self.settings['update_mesh_on_select_time'] = False
        
        # Next we'll set the system, which will parse the string sent
        # to init and will handle attaching all necessary signals
        if system is not None:
            # then for now (even though its hacky), we'll initialize
            # everything by setting the default first
            self.set_system()
        self.set_system(system, remove_dataref=remove_dataref)
        
        # Lastly, make sure all atmosphere tables are registered
        atms = self._get_by_search('atm', kind='Parameter', all=True, ignore_errors=True)
        ldcoeffs = self._get_by_search('ld_coeffs', kind='Parameter', all=True, ignore_errors=True)
        for atm in atms:
            limbdark.register_atm_table(atm.get_value())
        for ldcoeff in ldcoeffs:
            limbdark.register_atm_table(ldcoeff.get_value())
        
        # set tab completer
        readline.set_completer(phcompleter.Completer().complete)
        readline.set_completer_delims(' \t\n`~!#$%^&*)-=+]{}\\|;:,<>/?')
        readline.parse_and_bind("tab: complete")

    def _loop_through_container(self, return_type='twigs'):
        
        # we need to override the Container._loop_through_container to 
        # also search through usersettings and copy any items that do 
        # not exist here yet
        
        # first get items from bundle
        return_items = super(Bundle, self)._loop_through_container(do_sectionlevel=False)
        bundle_twigs = [ri['twig'] for ri in return_items]
        #~ bundle_unique_labels = [ri['twig'] for ri in return_items]
        
        # then get items from usersettings, checking each twig to see if there is a duplicate
        # with those found from the bundle.  If so - the bundle version trumps.  If not - 
        # we need to make a copy to the bundle and return that version
        
        for ri in self.get_usersettings()._loop_through_container():
            if ri['twig'] not in bundle_twigs:
                # then we need to make the copy
                
                if ri['section'] in ['compute','fitting']:
                    if ri['kind']=='OrderedDict':
                        # then this is at the section-level, and these
                        # will be rebuilt for the bundle later, so let's
                        # ignore for now
                        ri = None
                    else:
                        item_copy = ri['item'].copy()
                        ri['item'] = item_copy

                        ri['container'] = self.__class__.__name__
                        ri['twig_full'] = "{}@{}".format(ri['twig'],ri['container'])
                        
                        # now we need to attach to the correct place in the bundle
                        if isinstance(item_copy, parameters.ParameterSet):
                            self.sections[ri['section']].append(item_copy)
                
                if ri is not None:
                    return_items.append(ri) 
                    
        # now that new items have been copied, we need to redo things at the section level
        return_items += super(Bundle, self)._loop_through_container(do_pslevel=False)
        
        return return_items
        
    ## string representation
    def __str__(self):
        return self.to_string()
        
    def to_string(self):
        
        # Make sure to not print out all array variables
        old_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=8)
        # TODO: expand this to be generic across all sections
        txt = ""
        txt += "============ Compute ============\n"
        computes = self._get_dict_of_section('compute').values()
        for icomp in computes:
            mystring = []
            for par in icomp:
                mystring.append("{}={}".format(par,icomp.get_parameter(par).to_str()))
                if icomp.get_parameter(par).has_unit():
                    mystring[-1] += ' {}'.format(icomp.get_parameter(par).get_unit())
            mystring = ', '.join(mystring)
            txt += "\n".join(textwrap.wrap(mystring, initial_indent='', subsequent_indent=7*' ', width=79))
            txt += "\n"
        txt += "============ Other ============\n"
        if len(self._get_dict_of_section("fitting")):
            txt += "{} fitting options\n".format(len(self._get_dict_of_section("fitting")))
        if len(self._get_dict_of_section("axes")):
            txt += "{} axes\n".format(len(self._get_dict_of_section("axes")))
        txt += "============ System ============\n"
        txt += self.list(summary='full')
        
        # Default printoption
        np.set_printoptions(threshold=old_threshold)
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
        
    def set_usersettings(self,basedir=None):
        """
        Load user settings into the bundle
        
        These settings are not saved with the bundle, but are removed and
        reloaded everytime the bundle is loaded or this function is called.
        
        @param basedir: location of cfg files (or none to load from default location)
        @type basedir: string
        """
        if basedir is None or isinstance(basedir,str):
            settings = Settings()
        else: # if settings are passed instead of basedir - this should be fixed
            settings = basedir
            
        # else we assume settings is already the correct type
        self.usersettings = settings
        
    def get_usersettings(self):
        """
        Return the user settings class
        
        These settings are not saved with the bundle, but are removed and
        reloaded everytime the bundle is loaded or set_usersettings is called.
        """
        return self.usersettings
            
    def get_server(self,label=None):
        """
        Return a server by name
        
        @param servername: name of the server
        @type servername: string
        """
        return self._get_by_section(label,"server")
        
    #}    
    #{ System
    @rebuild_trunk
    def set_system(self, system=None, remove_dataref=False):
        """
        Change or set the system.
        
        Possibilities:
        
        1. If :envvar:`system` is a Body, then that body will be set as the system
        2. If :envvar:`system` is a string, the following options exist:
            - the string represents a Phoebe pickle file containing a Body; that
              one will be set
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
            #~ self.sections['system'] = [None]
            #~ return None
            library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../parameters/library/')
            system = os.path.join(library_dir, 'defaults.phoebe')
        elif system is False:
            self.system = None
            return
            
        # Or a real system
        file_type = None
        
        if isinstance(system, universe.Body):
            self.sections['system'] = [system]
        elif isinstance(system, list) or isinstance(system, tuple):
            self.sections['system'] = [create.system(system)]
        
        # Or we could've given a filename
        else:
            if not os.path.isfile(system):
                # then let's see if the filename exists in the library
                library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../parameters/library/')
                if os.path.isfile(os.path.join(library_dir, system)):
                    system = os.path.join(library_dir, system)
            # Try to guess the file type (if it is a file)
            if os.path.isfile(system):
                try:
                    f = open(system, 'r')
                    load_dict = json.load(f)
                    f.close()
                
                except ValueError:
                    file_type, contents = guess_filetype(system)
                    if file_type in ['wd', 'pickle_body']:
                        system = contents
                    elif file_type == 'phoebe_legacy':
                        system = contents[0]
                        if contents[1].get_value('label') in [c.get_value('label') for c in self.sections['compute']]:
                            self.remove_compute(contents[1].get_value('label'))
                        self.sections['compute'].append(contents[1])
                    elif file_type == 'pickle_bundle':
                        system = contents.get_system()
               
                else:
                    self._load_json(system)
                    file_type = 'json'                    

        
            # As a last resort, we pass it on to 'body_from_string' in the
            # create module:
            else:
                system = create.body_from_string(system)
            
            if file_type is not 'json': 
                # if it was json, then we've already attached the system
                self.sections['system'] = [system]
         
        system = self.get_system()
        if system is None:
            raise IOError('Initalisation failed: file/system not found')
       
        # Clear references if necessary:
        if remove_dataref is not False:
            if remove_dataref is True:
                remove_dataref = None
            system.remove_ref(remove_dataref)
            
        # initialize uptodate
        #~ print system.get_label()
        system.uptodate = False
        
        # connect signals
        #self.attach_system_signals()
        
    def get_system(self):
        """
        Return the system.
        
        @return: the attached system
        @rtype: Body or BodyBag
        """
        # we have to handle system slightly differently since building
        # the trunk requires calling this function
        return self.sections['system'][0]
        #~ return self._get_by_search(section='system', ignore_errors=True)
    
    def summary(self, objref=None):
        """
        Make a summary of the system, or any object in the system
        """
        bund_str = ""
        computes = self._get_dict_of_section('compute')
        #~ print computes
        if len(computes):
            bund_str+= "* Compute: " + ", ".join(computes.keys()) + '\n'
        #fittings = self._get_dict_of_section("fitting")
        #if len(fittings):
        #    bund_str+= "Fitting: " + ", ".join(fittings.keys()) + '\n'
        #axes = self._get_dict_of_section("axes")
        #if len(axes):
        #    bund_str+= "Axes: " + ", ".join(axes.keys()) + '\n'
        system_str = self.get_object(objref).list(summary='cursory')
        return system_str + '\n\n' + bund_str
        
    def tree(self, objref=None):
        """
        Make a summary of the system
        """
        return self.to_string()
        #return self.get_object(objref).list(summary='full')
    
    def list(self, summary=None, *args):
        """
        List with indices all the ParameterSets that are available.
        
        Simply a shortcut to :py:func:`bundle.get_system().list(...) <phoebe.backend.universe.Body.list>`.
        """
        return self.get_system().list(summary,*args)
        
    def clear_synthetic(self):
        """
        Clear all synthetic datasets
        Simply a shortcut to bundle.get_system().clear_synthetic()
        """
        return self.get_system().clear_synthetic()
        
    def set_time(self, time, label=None, server=None, **kwargs):
        """
        Set the time of a system, taking compute options into account.
                
        @param time: time
        @type time: float
        """
        system = self.get_system()
        
        # clear all previous models and create new model
        system.clear_synthetic()

        # <pieterdegroote> Necessary?
        system.set_time(0)
        
        # get compute options, handling 'default' if label==None
        options = self.get_compute(label, create_default=True).copy()
        
        # get server options
        # <kyle> this is dangerous and won't always work (if server is not local)
        if server is not None:
            server = self.get_server(server)
            mpi = server.mpi_ps
        else:
            mpi = kwargs.pop('mpi', None)
        
        options['time'] = [time]
        options['types'] = ['lc']
        options['refs'] = ['all']
        options['samprate'] = [[0]]
        system.compute(mpi=mpi, **options)
                
        #~ system.uptodate = label
        #~ self.attach_system_signals()
        
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
            
    def get_logp(self, dataset=None):
        """
        Retrieve the log probability of a collection of (or all) datasets.
        """
        
        # Q <pieterdegroote>: should we check first for system.uptodate?
        # Perhaps the run_compute didn't work out?
        
        if dataset is not None:
            # First disable/enabled correct datasets
            old_state = []
            location = 0
        
            for obs in self.get_system().walk_type(type='obs'):
                old_state.append(obs.get_enabled())
                this_state = False
                if dataset == obs['ref'] or dataset == obs.get_context():
                    if index is None or index == location:
                        this_state = True
                    location += 1
                obs.set_enabled(this_state)
        
        # Then compute statistics
        logf, chi2, n_data = self.get_system().get_logp(include_priors=True)
        
        # Then reset the enable property
        if dataset is not None:
            for obs, state in zip(self.get_system().walk_type(type='obs'), old_state):
                obs.set_enabled(state)
        
        return -0.5*sum(chi2)
    
    #}
    #{ Objects
    def get_object(self, objref=None):
        # handle objref if twig was given instead
        objref = objref.split('@')[0] if objref is not None else None
        # return the Body/BodyBag from the system hierarchy
        system = self.get_system()
        if objref is None or system.get_label() == objref or objref == '__nolabel__':
            this_child = system
        else:
            for child in system.walk_bodies():
                if child.get_label() == objref:
                    this_child = child
                    break
            else:
                raise ValueError("Object {} not found".format(objref))
        return this_child
            
        
    def get_children(self, objref=None):
        # handle objref if twig was given instead
        objref = objref.split('@')[0] if objref is not None else None
        # return list of children for self.get_object(objref)
        obj = self.get_object(objref)
        if hasattr(obj,'bodies'):
            #return [b.bodies[0] if hasattr(b,'bodies') else b for b in obj.bodies]
            return obj.bodies
        else:
            return []
        
    def get_parent(self, objref):
        # return the parent of self.get_object(objref)
        return self.get_object(objref).get_parent()
        
    def get_orbitps(self, objref=None):
        """
        retrieve the orbit ParameterSet that belongs to a given object
        
        @param objref: name of the object the mesh is attached to
        @type objref: str or None
        @param return_type: 'single','all'
        @type return_type: str
        """
        #~ qualifier = 'orbit'
        #~ if objref is not None:
            #~ qualifier += '{}{}'.format("@",objref)
        #~ return self.get_ps(qualifier)
        
        return self._get_by_search(label=objref, kind='ParameterSet', context='orbit')
        
    def get_meshps(self, objref=None):
        """
        retrieve the mesh ParameterSet that belongs to a given object
        
        @param objref: name of the object the mesh is attached to
        @type objref: str or None
        @param return_type: 'single','all'
        @type return_type: str
        """
        #~ qualifier = 'mesh*'
        #~ if objref is not None:
            #~ qualifier += '{}{}'.format("@",objref)
        #~ return self.get_ps(qualifier)
        
        return self._get_by_search(label=objref, kind='ParameterSet', context='mesh*')
        
    #}  
    #{ Datasets
    @rebuild_trunk
    def _attach_datasets(self, output, skip_defaults_from_body=True):
        """
        attach datasets and pbdeps from parsing file or creating synthetic datasets
        
        output is a dictionary with object names as keys and lists of both
        datasets and pbdeps as the values {objectname: [[ds1,ds2],[ps1,ps2]]}
        
        this is called from bundle.load_data and bundle.create_data
        and should not be called on its own   
        
        If ``skip_defaults_from_body`` is True, none of the parameters in the
        pbdep will be changed, and they will be added "as is". Else,
        ``skip_defaults_from_body`` needs to be a list of parameter names that
        **cannot** be changed. Any other key in the pbdep that is available in the
        main body parameterSet (e.g. atm, ld_coeffs...) but not in the
        ``skip_defaults_from_body`` list, will have the value taken from the main
        body. This, way, defaults from the body can be easily transferred.
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
                    
                    # get the main parameterSet:
                    main_parset = body.params.values()[0]
                    
                    for ps in pss:
                        
                        # Override defaults: those are all the keys that are
                        # available in both the pbdep and the main parset, and
                        # are not listed in skip_defaults_from_body
                        take_defaults = None
                        if skip_defaults_from_body is not True:
                            take_defaults = (set(ps.keys()) & set(main_parset.keys())) - set(skip_defaults_from_body)
                            for key in take_defaults:
                                if ps[key] != main_parset[key]:
                                    ps[key] = main_parset[key]
                            # and in case of overwriting existing one
                            take_defaults = set(ps.keys()) - set(skip_defaults_from_body)
                        body.add_pbdeps(ps.copy(), take_defaults=take_defaults)
                        
            else:
                # get the main parameterSet:
                main_parset = comp.params.values()[0]
                
                for ps in pss:
                    
                    # Override defaults: those are all the keys that are
                    # available in both the pbdep and the main parset, and
                    # are not listed in skip_defaults_from_body
                    take_defaults = None
                    if skip_defaults_from_body is not True:
                        take_defaults = (set(ps.keys()) & set(main_parset.keys())) - set(skip_defaults_from_body)
                        for key in take_defaults:
                            ps[key] = main_parset[key]
                    comp.add_pbdeps(ps.copy(), take_defaults=take_defaults)
            
            # obs get attached to the requested object
            for ds in dss:
                #~ ds.load()
                ds.estimate_sigma(from_col=None, force=False)
                comp.add_obs(ds)

        # Initialize the mesh after adding stuff (i.e. add columns ld_new_ref...
        self.get_system().init_mesh()
        
    
    #rebuild_trunk done by _attach_datasets
    def load_data(self, category, filename, passband=None, columns=None,
                  objref=None, units={}, dataref=None, scale=False, offset=False):
        """
        Add data from a file.
        
        Create multiple DataSets, load data,
        and add to corresponding bodies
        
        Special case here is "sed", which parses a list of snapshot multicolour
        photometry to different lcs. The will be grouped by ``filename``.
        
        @param category: category (lc, rv, sp, sed, etv)
        @type category: str
        @param filename: filename
        @type filename: str
        @param passband: passband
        @type passband: str
        @param columns: list of columns in file
        @type columns: list of strings
        @param objref: component for each column in file
        @type objref: list of strings (labels of the bodies)
        @param units: provide any non-default units
        @type units: dict
        @param dataref: name for ref for all returned datasets
        @type dataref: str    
        """
        
        if category == 'rv':
            output = datasets.parse_rv(filename, columns=columns,
                                       components=objref, units=units, 
                                       full_output=True,
                                       **{'passband':passband, 'ref': dataref})
        elif category == 'lc':
            output = datasets.parse_lc(filename, columns=columns,
                                       components=objref, full_output=True,
                                       **{'passband':passband, 'ref': dataref})
        elif category == 'etv':
            output = datasets.parse_etv(filename, columns=columns,
                                        components=objref, units=units,
                                        full_output=True,
                                        **{'passband':passband, 'ref': dataref})
        
        elif category == 'sp':
            output = datasets.parse_sp(filename, columns=columns,
                                       components=objref, units=units,
                                       full_output=True,
                                       **{'passband':passband, 'ref': dataref})
        
        elif category == 'sed':
            output = datasets.parse_phot(filename, columns=columns,
                  units=units,
                  group=filename, group_kwargs=dict(scale=scale, offset=offset),
                  full_output=True)
        #elif category == 'pl':
        #    output = datasets.parse_plprof(filename, columns=columns,
        #                               components=objref, full_output=True,
        #                               **{'passband':passband, 'ref': ref})
        else:
            output = None
            print("only lc, rv, etv, sed, and sp currently implemented")
        
        if output is not None:
            self._attach_datasets(output, skip_defaults_from_body=dict())
                       
    #rebuild_trunk done by _attach_datasets
    def create_data(self, category='lc', objref=None, dataref=None, **kwargs):
        """
        Create and attach empty data templates to compute the model.

        Additional keyword arguments contain information for the actual data
        template (cf. the "columns" in a data file) as well as for the passband
        dependable (pbdep) description of the dataset (optional, e.g. ``passband``,
        ``atm``, ``ld_func``, ``ld_coeffs``, etc). For any parameter that is not
        explicitly set, the defaults from each component are used, instead of the
        Phoebe2 defaults. For example when adding a light curve, pbdeps are added
        to each component, and the ``atm``, ``ld_func`` and ``ld_coeffs`` are
        taken from the component (i.e. the bolometric parameters) unless explicitly
        overriden.
        
        Unique references are added automatically if none are provided by the
        user (via :envvar:`dataref`). Instead of the backend-popular UUID system,
        the bundle implements a more readable system of unique references: the
        first light curve that is added is named 'lc01', and similarly for other
        categories. If the dataset with the reference already exists, 'lc02' is
        tried and so on.
        
        **Light curves (default)**
        
        Light curves are typically added to the entire system, as the combined
        light from all components is observed.
        
        For a list of available parameters, see :ref:`lcdep <parlabel-phoebe-lcdep>`
        and :ref:`lcobs <parlabel-phoebe-lcobs>`.
        
        >>> time = np.linspace(0, 10.33, 101)
        >>> bundle.create_data(time=time, passband='GENEVA.V')
        
        or in phase space (phase space will probably not work for anything but
        light curves and radial velocities):
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.create_data(phase=phase, passband='GENEVA.V')
        
        **Radial velocity curves**
        
        Radial velocities are typically added to the separate components, since
        they are determined from disentangled spectra.
        
        For a list of available parameters, see :ref:`rvdep <parlabel-phoebe-rvdep>`
        and :ref:`rvobs <parlabel-phoebe-rvobs>`.
        
        >>> time = np.linspace(0, 10.33, 101)
        >>> bundle.create_data(category='rv', objref='primary', time=time)
        >>> bundle.create_data(category='rv', objref='secondary', time=time)
        
        **Spectra**
        
        Spectra are typically added to the separate components, allthough they
        could as well be added to the entire system.
        
        For a list of available parameters, see :ref:`spdep <parlabel-phoebe-spdep>`
        and :ref:`spobs <parlabel-phoebe-spobs>`.
        
        >>> time = np.linspace(-0.5, 0.5, 11)
        >>> wavelength = np.linspace(454.8, 455.2, 500)
        >>> bundle.create_data(category='sp', objref='primary', time=time, wavelength=wavelength)
        
        or to add to the entire system:
        
        >>> bundle.create_data(time=time, wavelength=wavelength)
        
        **Interferometry**
        
        Interferometry is typically added to the entire system.
        
        For a list of available parameters, see :ref:`ifdep <parlabel-phoebe-ifdep>`
        and :ref:`ifobs <parlabel-phoebe-ifobs>`.
        
        >>> time = 0.1 * np.ones(101)
        >>> ucoord = np.linspace(0, 200, 101)
        >>> vcoord = np.zeros(101)
        >>> bundle.create_data(category='if', time=time, ucoord=ucoord, vcoord=vcoord)
        
        One extra option for interferometry is to set the keyword :envvar:`images`
        to a string, e.g.:
        
        >>> bundle.create_data(category='if', images='debug', time=time, ucoord=ucoord, vcoord=vcoord)
        
        This will generate plots of the system on the sky with the projected
        baseline orientation (as well as some other info), but will also
        write out an image with the summed profile (*_prof.png) and the rotated
        image (to correspond to the baseline orientation, (*_rot.png). Lastly,
        a FITS file is output that contains the image, for use in other programs.
        This way, you have all the tools available for debugging and to check
        if things go as expected.
        
        **And then what?**
        
        After creating synthetic datasets, you'll probably want to move on to
        functions such as 
        
        - :py:func:`Bundle.run_compute`
        - :py:func:`Bundle.plot_syn`
        
        @param category: 'lc', 'rv', 'sp', 'etv', 'if', 'pl'
        @type category: str
        @param objref: component for each column in file
        @type objref: None, str, list of str or list of bodies
        @param ref: name for ref for all returned datasets
        @type ref: str    
        """
        # create pbdeps and attach to the necessary object
        # this function will be used for creating pbdeps without loading an
        # actual file ie. creating a synthetic model only times will need to be
        # provided by the compute options (auto will not load times out of a pbdep)
        
        # Modified functionality from datasets.parse_header
        
        # What DataSet subclass do we need? We can derive it from the category.
        # This can be LCDataSet, RVDataSet etc.. If the category is not
        # recognised, we'll add the generic "DataSet".
        if not category in config.dataset_class:
            dataset_class = DataSet
        else:
            dataset_class = getattr(datasets, config.dataset_class[category])
    
        # Suppose the user did not specifiy the object to attach anything to
        if objref is None:
            # then attempt to make smart prediction
            if category in ['lc','if','sp']:
                # then top-level
                components = [self.get_system()]
                #logger.warning('components not provided - assuming {}'.format([comp.get_label() for comp in components]))
            else:
                logger.error('create_data failed: components need to be provided')
                return
        # is component just one string?
        elif isinstance(objref, str):
            components = [self.get_object(objref)]
        # is component a list of strings?
        elif isinstance(objref[0], str):
            components = [self.get_object(iobjref) for iobjref in objref]
        # perhaps component is a list of bodies, that's just fine then
        else:
            components = objref
        
        # We need a reference to link the pb and ds together.
        if dataref is None:
            # If the reference is None, suggest one. We'll add it as "lc01"
            # if no "lc01" exist, otherwise "lc02" etc (unless "lc02" already exists)
            existing_refs = self.get_system().get_refs(category=category)
            id_number = len(existing_refs)+1
            dataref = category + '{:02d}'.format(id_number)
            while dataref in existing_refs:
                id_number += 1
                dataref = category + '{:02d}'.format(len(existing_refs)+1)
        
        # Create template pb and ds:
        ds = dataset_class(context=category+'obs', ref=dataref)
        pb = parameters.ParameterSet(context=category+'dep', ref=dataref)
        
        # Split up the kwargs in extra arguments for the dataset and pbdep. We
        # are not filling in the pbdep parameters yet, because we're gonna use
        # smart defaults (see below). If a parameter is in both parameterSets,
        # the pbdep gets preference (e.g. for pblum, l3).
        pbkwargs = {}
        for key in kwargs:
            if key in pb:
                pbkwargs[key] = kwargs[key]
            elif key in ds:
                # Make sure time and phase are not both given, and that either
                # time or phase ends up in the defined columns (this is important
                # for the code that unphases)
                if key == 'time' and 'phase' in kwargs:
                    raise ValueError("You need to give either 'time' or 'phase', not both")
                elif key == 'phase' and 'time' in ds['columns']:
                    columns = ds['columns']
                    columns[columns.index('time')] = 'phase'
                    ds['columns'] = columns
                # and fill in    
                ds[key] = kwargs[key]
                
            else:
                raise ValueError("Parameter '{}' not found in obs/dep".format(key))        
        
        # Special treatment of oversampling rate and exposure time: if they are
        # single numbers, we need to convert them in arraya as long as as the
        # times
        for expand_key in ['samprate', 'exptime']:
            if expand_key in ds and not ds[expand_key].shape:
                ds[expand_key] = len(ds) * [ds[expand_key]]
        
        # check if all columns have the same length as time (or phase). There
        # a few exceptions: wavelength can be the same for all times for example
        reference_length = len(ds['time']) if 'time' in ds['columns'] else len(ds['phase'])
        ignore_columns = set(['wavelength'])
        for col in (set(ds['columns']) - ignore_columns):
            if not (len(ds[col])==0 or len(ds[col])==reference_length):
                raise ValueError("Length of column {} in dataset {} does not equal length of time/phase column".format(col, dataref))
        
        output = {}
        skip_defaults_from_body = pbkwargs.keys()
        for component in components:
            pb = parameters.ParameterSet(context=category+'dep', ref=dataref, **pbkwargs)
            output[component.get_label()] = [[ds],[pb]]
        self._attach_datasets(output, skip_defaults_from_body=skip_defaults_from_body)
    
    
    def get_syn(self, dataref=0, category=None, objref=None, all=False, ignore_errors=False):
        """
        Get synthetic
        
        """
        dss = OrderedDict()
        system = self.get_system()
        
        # Smart defaults:
        objref_orig = objref
        # For a light curve, it makes most sense to ask for the top level by
        # default
        if category == 'lc' and objref is None:
            objref = system.get_label()
            
        try:
            iterate_all_my_bodies = system.walk_bodies()
        except AttributeError:
            iterate_all_my_bodies = [system]
        
        for body in iterate_all_my_bodies:
            this_objref = body.get_label()
            
            if objref is None or this_objref == objref:
                ds = body.get_synthetic(ref=dataref, cumulative=True)
                
                if ds is not None and ds != []\
                    and (category is None or ds.context[:-3]==category)\
                    and not (objref_orig is None and category is None and ds.context[:-3]=='lc' and this_objref != system.get_label()):
                    
                    dss['{}{}{}'.format(ds['ref'],"@",this_objref)] = ds
                
                
        return self._return_from_dict(dss,all,ignore_errors)
                    

    def get_dep(self, objref=None, dataref=None, return_type='single'):
        pass
        
    def get_obs(self, objref=None, dataref=None, all=False, ignore_errors=False):
        """
        Get observations
        
        """
        dss = OrderedDict()
        system = self.get_system()
        
        try:
            iterate_all_my_bodies = system.walk_bodies()
        except AttributeError:
            iterate_all_my_bodies = [system]
        
        for body in iterate_all_my_bodies:
            this_objref = body.get_label()
            if objref is None or this_objref == objref:
                for obstype in body.params['obs']:
                    for this_dataref in body.params['obs'][obstype]:
                        if dataref is None or dataref==this_dataref:
                            ds = body.params['obs'][obstype][this_dataref]
                            dss['{}{}{}{}{}'.format(this_dataref,"@",obstype,"@",this_objref)] = ds
                            
        return self._return_from_dict(dss,all,ignore_errors)
        
    def enable_obs(self, dataref=None, objref=None):
        """
        Enable observations from being included in the fitting procedure.
        
        If you set :envvar:`dataref=None`, then all datasets will be disabled.
        """
        system = self.get_system()
        
        try:
            iterate_all_my_bodies = system.walk_bodies()
        except AttributeError:
            iterate_all_my_bodies = [system]
        
        for body in iterate_all_my_bodies:
            this_objref = body.get_label()
            if objref is None or this_objref == objref:
                for obstype in body.params['obs']:
                    if dataref is None:
                        for idataref in body.params['obs'][obstype]:
                            body.params['obs'][obstype][idataref].set_enabled(True)
                            logger.info("Enabled {} '{}'".format(obstype, idataref))
                    elif dataref in body.params['obs'][obstype]:
                        body.params['obs'][obstype][dataref].set_enabled(True)
                        logger.info("Enabled {} '{}'".format(obstype, dataref))


    def disable_obs(self, dataref=None, objref=None):
        """
        Disable observations from being included in the fitting procedure.
        
        If you set :envvar:`dataref=None`, then all datasets will be disabled.
        """
        system = self.get_system()
        
        try:
            iterate_all_my_bodies = system.walk_bodies()
        except AttributeError:
            iterate_all_my_bodies = [system]
        
        for body in iterate_all_my_bodies:
            this_objref = body.get_label()
            if objref is None or this_objref == objref:
                for obstype in body.params['obs']:
                    if dataref is None:
                        for idataref in body.params['obs'][obstype]:
                            body.params['obs'][obstype][idataref].set_enabled(False)
                            logger.info("Disabled {} '{}'".format(obstype, idataref))
                    elif dataref in body.params['obs'][obstype]:
                        body.params['obs'][obstype][dataref].set_enabled(False)
                        logger.info("Disabled {} '{}'".format(obstype, dataref))

    def reload_obs(self, dataref=None):
        """
        reload a dataset from its source file
        
        @param dataref: ref (name) of the dataset (or None for all)
        @type dataref: str or None
        """
        
        dss = self.get_obs(dataref=dataref,all=True).values()
        for ds in dss:
            ds.load()
    
    @rebuild_trunk
    def remove_data(self, dataref):
        """
        @param ref: ref (name) of the dataset
        @type ref: str
        """

        # disable any plotoptions that use this dataset
        for axes in self.get_axes(all=True).values():
            for pl in axes.get_plot().values():
                if pl.get_value('dataref')==dataref:
                    pl.set_value('active',False)
        
        # remove all obs attached to any object in the system
        for obj in self.get_system().walk_bodies():
            obj.remove_obs(refs=[dataref])
            if hasattr(obj, 'remove_pbdeps'): #TODO otherwise: warning 'BinaryRocheStar' has no attribute 'remove_pbdeps'
                obj.remove_pbdeps(refs=[dataref]) 

        return
        
    #}
    
    #{ Compute
    
    @rebuild_trunk
    @run_on_server
    def run_compute(self, label=None, objref=None, anim=False, server=None, **kwargs):
    #~ def run_compute(self,label=None,anim=False,add_version=None,server=None,**kwargs):
        """
        Convenience function to run observatory.observe
        
        @param label: name of one of the compute ParameterSets stored in bundle
        @type label: str
        @param objref: name of the top-level object used when observing
        @type objref: str
        @param anim: basename for animation, or False - will use settings in bundle.get_meshview()
        @type anim: False or str
        @param server: name of server to run on, or False to run locally (will override usersettings)
        @type server: string
        """
        system = self.get_system()
        obj = self.get_object(objref)
        #~ if add_version is None:
            #~ add_version = self.settings['add_version_on_compute']
            
        self.purge_signals(self.attached_signals_system)
        
        # clear all previous models and create new model
        system.clear_synthetic()

        # <pieterdegroote> I comment the following line, I don't think
        # it is necessary?
        #system.set_time(0)
        
        # get compute options, handling 'default' if label==None
        options = self.get_compute(label, create_default=True).copy()

        mpi = kwargs.pop('mpi', None)
        
        # get server options
        if server is not None:
            server = self.get_server(server)
            mpi = server.mpi_ps
            
        # now temporarily override with any values passed through kwargs    
        for k,v in kwargs.items():
            #if k in options.keys(): # otherwise nonexisting kwargs can be given
            options.set_value(k,v)
        
        
        # Q <pieterdegroote>: should we first set system.uptodate to False and
        # then try/except the computations? Though we should keep track of
        # why things don't work out.. how to deal with out-of-grid interpolation
        # etc...
        
        if options['time'] == 'auto' and anim == False:
            #~ observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
            obj.compute(mpi=mpi, **options)
        else:
            im_extra_func_kwargs = {key: value for key,value in self.get_meshview().items()}
            observatory.observe(obj,options['time'],lc=True,rv=True,sp=True,pl=True,
                extra_func=[observatory.ef_binary_image] if anim!=False else [],
                extra_func_kwargs=[self.get_meshview()] if anim!=False else [],
                mpi=mpi,**options
                )
        
        if anim != False:
            for ext in ['.gif','.avi']:
                plotlib.make_movie('ef_binary_image*.png',output='{}{}'.format(anim,ext),cleanup=ext=='.avi')
            
        system.uptodate = label
        
        #~ if add_version is not False:
            #~ self.add_version(name=None if add_version==True else add_version)

        self.attach_system_signals()
        
    #}
            
    #{ Fitting
    @rebuild_trunk
    @run_on_server
    def run_fitting(self,computelabel=None,fittinglabel=None,add_feedback=None,accept_feedback=False,server=None,**kwargs):
        """
        Run fitting for a given fitting ParameterSet
        and store the feedback
        
        @param computelabel: name of compute ParameterSet
        @param computelabel: str
        @param fittinglabel: name of fitting ParameterSet
        @type fittinglabel: str
        @param add_feedback: label to store the feedback under (retrieve with get_feedback)
        @type add_feedback: str or None
        @param accept_feedback: whether to automatically accept the feedback into the system
        @type accept_feedback: bool
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
        
        # get fitting params
        if fittinglabel is None:
            fittingoptions = parameters.ParameterSet(context='fitting')
        else:
            fittingoptions = self.get_fitting(fittinglabel).copy()
         
        # get compute params
        if computelabel is None:
            computeoptions = parameters.ParameterSet(context='compute')
        else:
            computeoptions = self.get_compute(label).copy()

        # now temporarily override with any values passed through kwargs    
        for k,v in kwargs.items():
            if k in options.keys():
                options.set_value(k,v)
            
        # now temporarily override with any values passed through kwargs    
        for k,v in kwargs.items():
            if k in fittingoptions.keys():
                fittingoptions.set_value(k,v)
            elif k in computeoptions.keys():
                computeoptions.set_value(k,v)
        
        # here, we should disable those obs that have no flux/rv/etc.., i.e.
        # the ones that were added just for exploration purposes. We should
        # keep track of those and then re-enstate them to their original
        # value afterwards (the user could also have disabled a dataset)
        # <some code>
        
        # Run the fitting for real
        feedback = fitting.run(self.get_system(), params=computeoptions, fitparams=fittingoptions, mpi=mpi)
        
        if add_feedback:
            self.add_feedback(feedback)
        
        # Then re-instate the status of the obs without flux/rv/etc..
        # <some code>
        
        if accept_feedback:
            fitting.accept_fit(self.get_system(),feedback)
            
        return feedback
    #}

    #{ Figures
    def plot_obs(self, dataref, *args, **kwargs):
        """
        Make a plot of the attached observations.
        
        The arguments are passed to the appropriate functions in :py:mod:`plotting`.
        
        Example usage::
            
            bundle.plot_obs('mylc')
            bundle.plot_obs('mylc', objref='secondary')
            bundle.plot_obs('mylc', fmt='ko-', objref='secondary')
            bundle.plot_obs('mylc', fmt='ko-', label='my legend label', objref='secondary')
        
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param objref: label of the object
        @type objref: str
        """
        objref = kwargs.pop('objref', None)
        
        dss = self.get_obs(dataref=dataref, objref=objref, all=True)
        if len(dss) > 1:
            logger.warning('more than one obs exists with this dataref, provide objref to ensure correct obs is used')
        elif not len(dss):
            raise ValueError("dataref '{}' not found for plotting".format(dataref))
        
        # Get the obs DataSet and retrieve its context
        ds = dss.values()[0]
        #obj = self.get_object(dss.keys()[0].split('@')[2])
        obj = self.get_object(re.split('@', dss.keys()[0])[2])
        context = ds.get_context()
        
        # Now pass everything to the correct plotting function
        kwargs['ref'] = dataref
        getattr(plotting, 'plot_{}'.format(context))(obj, *args, **kwargs)
        

        
    def plot_syn(self, dataref, *args, **kwargs):
        """
        Plot simulated/computed observations.
        
        Extra args and kwargs are passed to the corresponding plotting function
        in :py:mod:`phoebe.backend.plotting` (which passes most on to matplotlib), except
        the optional keyword argument :envvar:`objref`. The latter allows you to
        plot the synthetic computations for only one component. Although this is
        mainly useful for radial velocity computations, it also allows you to
        plot the light curves or spectra of the individual components, even
        though the computations are done for the entire system (this will not
        work for interferometry yet).
        
        Example usage:
        
        >>> mybundle.plot_syn('lc01', 'r-', lw=2) # first light curve added via 'create_data'
        >>> mybundle.plot_syn('lc01', 'r-', lw=2, scale=None)
        
        >>> mybundle.plot_syn('if01', 'k-') # first interferometry added via 'create_data'
        >>> mybundle.plot_syn('if01', 'k-', y='vis2') # first interferometry added via 'create_data'
        
        More information on arguments and keyword arguments:
        
        - :py:func:`phoebe.backend.plotting.plot_lcsyn`
        - :py:func:`phoebe.backend.plotting.plot_rvsyn`
        - :py:func:`phoebe.backend.plotting.plot_spsyn_as_profile`
        - :py:func:`phoebe.backend.plotting.plot_ifsyn`
        
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param objref: label of the object
        @type objref: str
        """
        objref = kwargs.pop('objref', None)
        
        dss = self.get_syn(dataref=dataref, objref=objref, all=True)
        if len(dss) > 1:
            logger.info('Retrieving synthetic computations with objref={}'.format(dss.keys()[0]))
        elif not len(dss):
            raise ValueError("dataref '{}' not found for plotting".format(dataref))
        # Get the obs DataSet and retrieve its context
        ds = dss.values()[0]
        #obj = self.get_object(dss.keys()[0].split('@')[-1])
        obj = self.get_object(re.split('@', dss.keys()[0])[-1])
        context = ds.get_context()
        
        # Now pass everything to the correct plotting function
        kwargs['ref'] = dataref
        getattr(plotting, 'plot_{}'.format(context))(obj, *args, **kwargs)

            
    def plot_residuals(self,dataref,objref=None,**kwargs):
        """
        @param dataref: ref (name) of the dataset
        @type dataref: str
        @param objref: label of the object
        @type objref: str
        """
        objref = kwargs.pop('objref', None)
        
        dss = self.get_obs(dataref=dataref, objref=objref, all=True)
        if len(dss) > 1:
            logger.warning('more than one obs exists with this dataref, provide objref to ensure correct obs is used')
        elif not len(dss):
            raise ValueError("dataref '{}' not found for plotting".format(dataref))
        
        # Get the obs DataSet and retrieve its context
        ds = dss.values()[0]
        #obj = self.get_object(dss.keys()[0].split('@')[2])
        obj = self.get_object(re.split('@', dss.keys()[0])[2])
        context = ds.get_context()
        
        # Now pass everything to the correct plotting function
        kwargs['ref'] = dataref
        getattr(plotting, 'plot_{}res'.format(context))(obj, *args, **kwargs)
    
    def write_syn(self, dataref, output_file, objref=None):
        dss = self.get_syn(dataref=dataref, objref=objref, all=True)
        if len(dss) > 1:
            logger.warning('more than one syn exists with this dataref, provide objref to ensure correct syn is used')
        elif not len(dss):
            raise ValueError("dataref '{}' not found for writing".format(dataref))
        
        # Get the obs DataSet and write to a file
        ds = dss[0]
        ds.save(output_file)
    
    def get_axes(self,ident=None):
        """
        Return an axes or list of axes that matches index OR title
        
        @param ident: index or title of the desired axes
        @type ident: int or str
        @return: axes
        @rtype: frontend.figures.Axes
        """
        if isinstance(ident,int): 
            return self._get_dict_of_section('axes', kind='Container').values()[ident]
        
        return self._get_by_section(section='axes', kind='Container', label=ident)
        
    @rebuild_trunk
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
            
        if 'title' not in kwargs.keys():
            kwargs['title'] = "myaxes{}".format(int(len(self._get_dict_of_section('axes'))+1))
            
        for key in kwargs.keys():
            axes.set_value(key, kwargs[key])
            
        self._add_to_section('axes',axes)
    
    @rebuild_trunk
    def remove_axes(self,ident):
        """
        Removes all axes with a given index or title
        
        @param ident: index or title of the axes to be removed
        @type ident: int or str
        """
        if isinstance(ident,int): 
            self.sections['axes'].pop(ident)

        else:
            axes = self.get_axes(ident)
            self.sections['axes'].remove(axes)
        
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
        
    def get_meshview(self,label=None):
        """
        
        """
        return self._get_by_section(label,"meshview")
      
        
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
        else:
            self.set_time(0)
        
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
        
    def get_orbitview(self,label=None):
        """
        
        """
        # TODO: fix this so we can set defaults in usersettings
        # (currently can't with search_by = None)
        return self._get_by_section(label,'orbitview')
        
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
        
        orbits = self._get_by_search(context='orbit', kind='ParameterSet', all=True)
        periods = np.array([o.get_value('period') for o in orbits])
        top_orbit = orbits[periods.argmax()]
        bottom_orbit = orbits[periods.argmin()]
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
        # of bundle.get_system()
        #~ bundle.set_system(bundle_new.get_system()) # note: anything changed outside system will be lost

        # cleanup files
        logger.info('cleaning files in {}'.format(mount_dir))
        for fname in lock_files:
            os.remove(os.path.join(mount_dir,fname))
        os.remove(os.path.join(mount_dir,'%s.status' % script))
        os.remove(os.path.join(mount_dir,'%s.log' % script))
        
        return
    
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
        if system is not None and not isinstance(system.signals, dict):
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
        # get_system_structure is not implemented yet
        
        # these might already be attached?
        self.attach_signal(self,'load_data',self._on_param_changed)
        self.attach_signal(self,'remove_data',self._on_param_changed)
        self.attach_signal(self,'enable_obs',self._on_param_changed)
        self.attach_signal(self,'disable_obs',self._on_param_changed)
        #~ self.attach_signal(self,'adjust_obs',self._on_param_changed)
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
    
    def save(self, filename):
        self._save_json(filename)
    
    def save_pickle(self,filename=None,purge_signals=True,save_usersettings=False):
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
    
    def check(self, qualifier=None, index=0):
        """
        Check if a system is OK.
        
        What 'OK' is, depends on a lot of stuff. Typically this function can be
        used to do some sanity checks when fitting, such that impossible systems
        can be avoided.
        
        We check if a parameter (or all) has a finite log likelihood.
        
        If ``qualifier=None``, all parameters with priors are checked. If any is
        found to be outside of bounds, ``False`` is returned. Any other parameter,
        even the ones without priors, are checked for their limits. If any is
        outside of the limits, ``False`` is returned. If no parameters are
        outside of their priors and/or limits, ``True`` is returned.
        
        We preprocess the system first.
        """
        
        self.get_system().preprocess()
        
        if qualifier is not None:
            par = self.get_parameter(qualifier, all=True).values()[index]
            return -np.isinf(par.get_logp())
        
        else:
            
            already_checked = []
            system = self.get_system()
            were_still_OK = True
            
            for path, val in system.walk_all():
                
                if not were_still_OK:
                    continue
                
                # If it's not a parameter don't bother
                if not isinstance(val, parameters.Parameter):
                    continue
                
                # If we've already checked this parameter, don't bother
                if val in already_checked:
                    continue
                
                # If the value has zero probability, we're not OK!
                if val.has_prior() and np.isinf(val.get_logp()):
                    were_still_OK = False
                    continue
                
                # If the value is outside of the limits (if it has any), we are
                # not OK!
                if not val.is_inside_limits():
                    were_still_OK = False
                    continue
                    
                # Remember we checked this one
                already_checked.append(val)
            
            return were_still_OK

            
        
    def updateLD(self):
        """
        Update limbdarkening coefficients according to local quantities.
        """
        atm_types = self.get_parameter('atm', all=True).values()
        ld_coeffs = self.get_parameter('ld_coeffs', all=True).values()
        for atm_type, ld_coeff in zip(atm_types, ld_coeffs):
            ld_coeff.set_value(atm_type)
    
    def set_beaming(self, on=True):
        """
        Include/exclude the boosting effect.
        """
        self.set_value('beaming', on, apply_to='all')
        
    def set_ltt(self, on=True):
        """
        Include/exclude light-time travel effects.
        """
        self.set_value('ltt', on, apply_to='all')
    
    def set_heating(self, on=True):
        """
        Include/exclude heating effects.
        """
        self.set_value('heating', on, apply_to='all')
        
    def set_reflection(self, on=True):
        """
        Include/exclude reflection effects.
        """
        self.set_value('refl', on, apply_to='all')
    
    def set_gray_scattering(self, on=True):
        """
        Force gray scattering.
        """
        system = self.get_system()
        if on:
            system.add_preprocess('gray_scattering')
        else:
            system.remove_preprocess('gray_scattering')
            
    
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
    file_type, contents = guess_filetype(filename)
    
    if file_type == 'pickle_bundle':
        bundle = contents
    elif file_type == 'pickle_body':
        bundle = Bundle(system=contents)
    elif file_type == 'phoebe_legacy':
        bundle = Bundle(system=contents[0])
        bundle.add_compute(contents[1])
    elif file_type == 'wd':
        bundle = Bundle(system=contents)
    
    # Load this users settings into the bundle
    if load_usersettings:
        bundle.set_usersettings()
    
    logger.info("Loaded contents of {}-file from {} into a Bundle".format(file_type, filename))
    
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
            
        # If the file is not a pickle file or json, it could be a Phoebe legacy file?
        if contents is None:
            
            try:
                contents = parsers.legacy_to_phoebe2(filename)
                file_type = 'phoebe_legacy'
            except IOError:
                pass
            except TypeError:
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
