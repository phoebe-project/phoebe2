"""
Top level interface to Phoebe2.

The Bundle aims at providing a user-friendly interface to a Body or BodyBag,
such that parameters can easily be queried or changed, data added, results
plotted and observations computed. It does not contain any implementation of
physics; that is the responsibility of the backend and the associated
library.

**Phoebe1 compatibility**
    
Phoebe2 can be used in a Phoebe1 compatibility mode ('legacy' mode), which is
most easily when you start immediately from a legacy parameter file:
    
    >>> mybundle = Bundle('legacy.phoebe')

When a Bundle is loaded this way, computational options are added automatically
to the Bundle to mimick the physics that is available in Phoebe1. These options
are collected in a ParameterSet of context 'compute' with label ``legacy``.
The most important parameters are listed below (some unimportant ones have been
excluded so if you try this, you'll see more parameters)

    >>> print(mybundle['legacy'])
              label legacy        --   label for the compute options               
            heating True          --   Allow irradiators to heat other Bodies      
               refl False         --   Allow irradiated Bodies to reflect light    
           refl_num 1             --   Number of reflections                       
                ltt False         --   Correct for light time travel effects       
         subdiv_num 3             --   Number of subdivisions                      
        eclipse_alg binary        --   Type of eclipse algorithm                   
        beaming_alg none          --   Type of beaming algorithm                   
    irradiation_alg point_source  --   Type of irradiation algorithm

The irradiation algorithm and reflection iterations will be adjust to match the
value in the loaded parameter file.

Once data is added, you could run Phoebe2 while mimicking the physics in Phoebe1
via::

>>> mybundle.run_compute(label='legacy')
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
from phoebe.parameters import definitions
from phoebe.parameters import datasets
from phoebe.parameters import create
from phoebe.parameters import tools
from phoebe.backend import fitting, observatory, plotting
from phoebe.backend import universe
from phoebe.atmospheres import limbdark
from phoebe.io import parsers
from phoebe.dynamics import keplerorbit
from phoebe.frontend.usersettings import Settings
from phoebe.frontend.common import Container, rebuild_trunk
from phoebe.frontend.figures import Axes
from phoebe.units import conversions
import phcompleter

logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())

def run_on_server(fctn):
    """
    Parse usersettings to determine whether to run a function locally
    or submit it to a server
    
    [FUTURE]
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
    
     
    
    

class Bundle(Container):
    """
    Class representing a collection of systems and stuff related to it.
    
    **Initialization**
    
    You can initiate a bundle in different ways:
    
      1. Using the default binary parameters::
      
          mybundle = Bundle()
          
      2. Via a PHOEBE 2.0 file in JSON format::
      
          mybundle = Bundle('newbundle.json')
    
      3. Via a Phoebe Legacy ASCII parameter file::
      
          mybundle = Bundle('legacy.phoebe')
    
      4. Via a Body or BodyBag::
      
          mysystem = phoebe.create.from_library('V380_Cyg', create_body=True)
          mybundle = Bundle(mysystem)
      
      5. Via the predefined systems in the library::
      
          mybundle = Bundle('V380_Cyg')            
        
    For more details, see :py:func:`set_system`.
    
    **Interface**
    
    The interaction with a Bundle is much alike interaction with a Python
    dictionary. The following functionality is implemented and behaves as
    expected::
            
            # return the value of the period if it exists, raises error if 'period' does not exist
            period = mybundle['period']
            
            # set the value of the period if it exists, raises error otherwise
            mybundle['period'] = 5.0
            
            # return the value of the period if it exists, else returns None
            period = mybundle.get('period')
            
            # return the value of the period if it exists, else returns default_value (whatever it is)
            period = mybundle.get('period', default_value)
            
            # returns a list of available keys
            keys = mybundle.keys()
            
            # returns a list of values
            values = mybundle.values()
            
            # iterate over the keys in the Bundle
            for key in mybundle:
                print(key, mybundle[key])
    
    .. important::
    
        *keys* are referred to as *twigs* in the context of Bundles. They behave
        much like dictionary keys, but are much more flexible to account for the
        plethora of available parameters (and possible duplicates!) in the
        Bundle. For example, both stars in a binary need atmosphere tables in
        order to compute their bolometric luminosities. This is done via the
        parameter named ``atm``, but since it is present in both components,
        you need to ``@`` operator to be more specific:
        
            >>> mybundle['atm@primary'] = 'kurucz'
            >>> mybundle['atm@secondary'] = 'blackbody'
             
    
    Inherited from :py:class:`common.Container <phoebe.frontend.common.Container>`:
    
    .. autosummary::
    
        phoebe.frontend.common.Container.get_value
        phoebe.frontend.common.Container.get_value_all
        phoebe.frontend.common.Container.get_ps
        phoebe.frontend.common.Container.get_parameter
        phoebe.frontend.common.Container.get_adjust
        phoebe.frontend.common.Container.get_prior
        phoebe.frontend.common.Container.set_value
        phoebe.frontend.common.Container.set_value_all
        phoebe.frontend.common.Container.set_ps
        phoebe.frontend.common.Container.attach_ps
        phoebe.frontend.common.Container.twigs
        phoebe.frontend.common.Container.search
    
    Defined within the Bundle:        
        
    .. autosummary::    
        
        Bundle.set_system
        
        Bundle.run_compute
        
        Bundle.lc_fromfile
        Bundle.lc_fromarrays
        Bundle.lc_fromexisting
        Bundle.rv_fromfile
        Bundle.rv_fromarrays
        Bundle.rv_fromexisting
        
        Bundle.plot_obs
        Bundle.plot_syn
    
    
    **Printing information**
    
    An overview of all the Parameters and observations loaded in a Bundle can
    be printed to the screen easily using::
    
        mybundle = phoebe.Bundle()
        print(mybundle)
        
    Extra functions are available that return informational strings
    
    .. autosummary::
    
        Bundle.summary
        Bundle.info
        
    **Structure of the Bundle**
        
    A Bundle contains:
    
        - a Body (or BodyBag), called :envvar:`system` in this context.
        - a list of compute options which can be stored and used to compute observables.
        
        
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

    def _loop_through_container(self):
        """
        [FUTURE]

        override container defaults to also loop through the usersettings
        and copy as necessary

        This function loops through the container and returns a list of dictionaries
        (called the "trunk").
        
        This function is called by _build_trunk and the rebuild_trunk decorator,
        and shouldn't need to be called elsewhere        
        """
        
        # we need to override the Container._loop_through_container to 
        # also search through usersettings and copy any items that do 
        # not exist here yet
        
        # first get items from bundle
        return_items = super(Bundle, self)._loop_through_container(do_sectionlevel=False)
        bundle_twigs = [ri['twig'] for ri in return_items]
        
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
        """
        Returns the string representation of the bundle
        
        :return: string representation
        :rtype: str
        """
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
        self._build_trunk()
        
    def get_usersettings(self):
        """
        Return the user settings class
        
        These settings are not saved with the bundle, but are removed and
        reloaded everytime the bundle is loaded or set_usersettings is called.
        """
        return self.usersettings
        
    def save_usersettings(self, basedir=None):
        """
        save all usersettings in to .cfg files in basedir
        if not provided, basedir will default to ~/.phoebe (recommended)
        
        
        @param basedir: base directory, or None to save to initialized basedir
        @type basedir: str or None
        """
        self.get_usersettings().save()
        
    def restart_logger(self, label='default_logger', **kwargs):
        """
        [FUTURE]
        
        restart the logger
        
        any additional arguments passed will temporarily override the settings
        in the stored logger.  These can include style, clevel, flevel, filename, filemode
        
        @param label: the label of the logger (will default if not provided)
        @type label: str
        """
        
        self.get_usersettings().restart_logger(label, **kwargs)
        
        
            
    def get_server(self,label=None):
        """
        Return a server by name
        
        [FUTURE]
        
        @param servername: name of the server
        @type servername: string
        """
        return self._get_by_section(label,"server")
        
    #}    
    #{ System
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
        
        With :envvar:`remove_dataref`, you can either choose to remove a
        specific dataset (if it is a string with the datareference), remove all
        data (if it is ``True``) or keep them as they are (if ``False``, the
        default).
        
        :param system: the new system
        :type system: Body or str
        :param remove_dataref: remove any/all datasets or keep them
        :type remove_dataref: ``False``, ``None`` or a string
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
        self._build_trunk()
        
    def get_system(self):
        """
        Return the system.
        
        :return: the attached system
        :rtype: Body or BodyBag
        """
        # we have to handle system slightly differently since building
        # the trunk requires calling this function
        return self.sections['system'][0]
        #~ return self._get_by_search(section='system', ignore_errors=True)
    
    def summary(self, objref=None):
        """
        Make a summary of the hierarchy of the system (or any object in it).
        
        :param objref: object reference
        :type objref: str or None (defaults to the whole system)
        :return: summary string
        :rtype: str
        """
        bund_str = ""
        computes = self._get_dict_of_section('compute')
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
        
    def tree(self):
        """
        Return a summary of the system.
        
        :return: string representation of the system
        :rtype: str
        """
        return self.to_string()
    
    def list(self, summary=None, *args):
        """
        List with indices all the ParameterSets that are available.
        
        Simply a shortcut to :py:func:`bundle.get_system().list(...) <universe.Body.list>`.
        See that function for more info on the arguments.
        """
        return self.get_system().list(summary,*args)
    
    
    def clear_syn(self):
        """
        Clear all synthetic datasets.
        
        Simply a shortcut to :py:func:`bundle.get_system().clear_synthetic() <phoebe.backend.universe.Body.clear_synthetic>`
        """
        self.get_system().clear_synthetic()
        self._build_trunk()
        
    def set_time(self, time, label=None, server=None, **kwargs):
        """
        Set the time of a system, taking compute options into account.
                
        [FUTURE]
        
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
        
        [FUTURE]
        
        @return: uptodate
        @rtype: bool or str
        """
        return self.get_system().uptodate
            
    #}
    #{ Parameters/ParameterSets
            
    def get_logp(self, dataset=None):
        """
        Retrieve the log probability of a collection of (or all) datasets.
        
        [FUTURE]
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
    def _get_object(self, objref=None):
        """
        Twig-free version of get_object.
        
        This version of get_object that does not use twigs... this must remain
        so that we can call it while building the trunk

        :param objref: Body's label or twig
        :type objref: str
        :return: the Body corresponding to the label
        :rtype: Body
        :raises ValueError: when objref is not present
        """
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
    
    def get_object(self, twig=None):
        """
        Retrieve a Body or BodyBag from the system
        
        if twig is None, will return the system itself
        
        :param twig: the twig/twiglet to use when searching
        :type twig: str
        :return: the object
        :rtype: Body
        """
        if twig is None:
            return self.get_system()
        return self._get_by_search(twig, kind='Body')
        
    def get_children(self, twig=None):
        """
        Retrieve the direct children of a Body or BodyBag from the system
        
        :param twig: the twig/twiglet to use when searching
        :type twig: str
        :return: the children
        :rtype: list of bodies
        """
        obj = self.get_object(twig)
        if hasattr(obj, 'bodies'):
            #return [b.bodies[0] if hasattr(b,'bodies') else b for b in obj.bodies]
            return obj.bodies
        else:
            return []
        
    def get_parent(self, twig=None):
        """
        Retrieve the direct parent of a Body or BodyBag from the system
        
        :param twig: the twig/twiglet to use when searching
        :type twig: str
        :return: the children
        :rtype: Body
        """
        return self.get_object(twig).get_parent()
        
        
    def get_orbitps(self, twig=None):
        """
        [FUTURE]
        
        retrieve the orbit ParameterSet that belongs to a given BodyBag
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @return: the orbit PS
        @rtype: ParameterSet
        """
        # TODO: handle default if twig is None
        return self._get_by_search('orbit@{}'.format(twig), kind='ParameterSet', context='orbit')
        
        
    def get_meshps(self, twig=None):
        """
        [FUTURE]

        retrieve the mesh ParameterSet that belongs to a given component
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @return: the mesh PS
        @rtype: ParameterSet
        """
        return self._get_by_search('mesh@{}'.format(twig), kind='ParameterSet', context='mesh*')
    
    
    def set_main_period(self, period=None, objref=None):
        """
        Set the main period of the system.
        
        Any parameter that is used to set the period needs to be of dimensions
        of time or frequency, such that the conversion to time is
        straightforward.
        
        [FUTURE]
        
        :param period: twig of the Parameter that represent the main period
        :type period: str (twig)
        :raises ValueError: if period is not of time or frequency dimensions
        """
        # Get the object to set the main period of, and get the parameter
        obj = self.get_object(objref)
        
        if period is not None:
            period = self._get_by_search(period, kind='Parameter')
        
            # Check unit type:
            unit_type = conversions.get_type(period.get_unit())
            if not unit_type in ['time', 'frequency']:
                raise ValueError('Cannot set period of the system to {} which has units of {} (needs to be time or frequency)'.format(period, unit_type))
            obj.set_period(period=period)
        
    #}  
    #{ Datasets
    def _attach_datasets(self, output, skip_defaults_from_body=True):
        """
        attach datasets and pbdeps from parsing file or creating synthetic datasets
        
        output is a dictionary with object names as keys and lists of both
        datasets and pbdeps as the values {objectname: [[ds1,ds2],[ps1,ps2]]}
        
        this is called from bundle.load_data and bundle.data_fromarrays
        and should not be called on its own   
        
        If ``skip_defaults_from_body`` is True, none of the parameters in the
        pbdep will be changed, and they will be added "as is". Else,
        ``skip_defaults_from_body`` needs to be a list of parameter names that
        **cannot** be changed. Any other key in the pbdep that is available in the
        main body parameterSet (e.g. atm, ld_coeffs...) but not in the
        ``skip_defaults_from_body`` list, will have the value taken from the main
        body. This, way, defaults from the body can be easily transferred.
        
        [FUTURE]
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
        self._build_trunk()
    
    #rebuild_trunk done by _attach_datasets
    def data_fromfile(self, filename, category='lc', objref=None, dataref=None,
                      columns=None, units=None, **kwargs):
        """
        Add data from a file.
        
        Create multiple DataSets, load data, and add to corresponding bodies
        
        Special case here is "sed", which parses a list of snapshot multicolour
        photometry to different lcs. They will be grouped by ``filename``.
        
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
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        """
        if units is None:
            units = {}
            
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
        
        # Individual cases
        if category == 'rv':
            output = datasets.parse_rv(filename, columns=columns,
                                       components=objref, units=units, 
                                       full_output=True, **kwargs)
        elif category == 'lc':
            output = datasets.parse_lc(filename, columns=columns, units=units,
                                       components=objref, full_output=True,
                                       ref=dataref, **kwargs)
            
            # if no componets (objref) was given, then we assume it's the system!
            for lbl in output:
                if lbl == '__nolabel__':
                    output[self.get_system().get_label()] = output.pop('__nolabel__')
                    
            
        elif category == 'etv':
            output = datasets.parse_etv(filename, columns=columns,
                                        components=objref, units=units,
                                        full_output=True, **kwargs)
        
        elif category == 'sp':
            output = datasets.parse_sp(filename, columns=columns,
                                       components=objref, units=units,
                                       full_output=True,
                                       **{'passband':passband, 'ref': dataref})
        
        elif category == 'sed':
            scale, offset = kwargs.pop('scale', True), kwargs.pop('offset', False)
            output = datasets.parse_phot(filename, columns=columns,
                  units=units, group=filename,
                  group_kwargs=dict(scale=scale, offset=offset),
                  full_output=True, **kwargs)
        #elif category == 'pl':
        #    output = datasets.parse_plprof(filename, columns=columns,
        #                               components=objref, full_output=True,
        #                               **{'passband':passband, 'ref': ref})
        else:
            output = None
            print("only lc, rv, etv, sed, and sp currently implemented")
        
        if output is not None:
            self._attach_datasets(output, skip_defaults_from_body=kwargs.keys())
            return dataref
                       
    #rebuild_trunk done by _attach_datasets
    def data_fromarrays(self, category='lc', objref=None, dataref=None,
                        **kwargs):
        """
        Create and attach data templates to compute the model.

        Additional keyword arguments contain information for the actual data
        template (cf. the "columns" in a data file) as well as for the passband
        dependable (pbdep) description of the dataset (optional, e.g.
        ``passband``, ``atm``, ``ld_func``, ``ld_coeffs``, etc). For any
        parameter that is not explicitly set, the defaults from each component
        are used, instead of the Phoebe2 defaults. For example when adding a
        light curve, pbdeps are added to each component, and the ``atm``,
        ``ld_func`` and ``ld_coeffs`` are taken from the component (i.e. the
        bolometric parameters) unless explicitly overriden.
        
        Unique references are added automatically if none are provided by the
        user (via :envvar:`dataref`). Instead of the backend-popular UUID
        system, the bundle implements a more readable system of unique
        references: the first light curve that is added is named 'lc01', and
        similarly for other categories. If the dataset with the reference
        already exists, 'lc02' is tried and so on.
        
        **Light curves (default)**
        
        Light curves are typically added to the entire system, as the combined
        light from all components is observed.
        
        For a list of available parameters, see :ref:`lcdep <parlabel-phoebe-lcdep>`
        and :ref:`lcobs <parlabel-phoebe-lcobs>`.
        
        >>> time = np.linspace(0, 10.33, 101)
        >>> bundle.data_fromarrays(time=time, passband='GENEVA.V')
        
        or in phase space (phase space will probably not work for anything but
        light curves and radial velocities):
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.data_fromarrays(phase=phase, passband='GENEVA.V')
        
        **Radial velocity curves**
        
        Radial velocities are typically added to the separate components, since
        they are determined from disentangled spectra.
        
        For a list of available parameters, see :ref:`rvdep <parlabel-phoebe-rvdep>`
        and :ref:`rvobs <parlabel-phoebe-rvobs>`.
        
        >>> time = np.linspace(0, 10.33, 101)
        >>> bundle.data_fromarrays(category='rv', objref='primary', time=time)
        >>> bundle.data_fromarrays(category='rv', objref='secondary', time=time)
        
        **Spectra**
        
        Spectra are typically added to the separate components, although they
        could as well be added to the entire system.
        
        For a list of available parameters, see :ref:`spdep <parlabel-phoebe-spdep>`
        and :ref:`spobs <parlabel-phoebe-spobs>`.
        
        >>> time = np.linspace(-0.5, 0.5, 11)
        >>> wavelength = np.linspace(454.8, 455.2, 500)
        >>> bundle.data_fromarrays(category='sp', objref='primary', time=time, wavelength=wavelength)
        
        or to add to the entire system:
        
        >>> bundle.data_fromarrays(time=time, wavelength=wavelength)
        
        **Interferometry**
        
        Interferometry is typically added to the entire system.
        
        For a list of available parameters, see :ref:`ifdep <parlabel-phoebe-ifdep>`
        and :ref:`ifobs <parlabel-phoebe-ifobs>`.
        
        >>> time = 0.1 * np.ones(101)
        >>> ucoord = np.linspace(0, 200, 101)
        >>> vcoord = np.zeros(101)
        >>> bundle.data_fromarrays(category='if', time=time, ucoord=ucoord, vcoord=vcoord)
        
        One extra option for interferometry is to set the keyword :envvar:`images`
        to a string, e.g.:
        
        >>> bundle.data_fromarrays(category='if', images='debug', time=time, ucoord=ucoord, vcoord=vcoord)
        
        This will generate plots of the system on the sky with the projected
        baseline orientation (as well as some other info), but will also
        write out an image with the summed profile (_prof.png) and the rotated
        image (to correspond to the baseline orientation, (_rot.png). Lastly,
        a FITS file is output that contains the image, for use in other programs.
        This way, you have all the tools available for debugging and to check
        if things go as expected.
        
        **And then what?**
        
        After creating synthetic datasets, you'll probably want to move on to
        functions such as 
        
        - :py:func:`Bundle.run_compute`
        - :py:func:`Bundle.plot_syn`
        
        :param category: one of 'lc', 'rv', 'sp', 'etv', 'if', 'pl'
        :type category: str
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises ValueError: if :envvar:`category` is not recognised.
        :raises ValueError: if :envvar:`time` and :envvar:`phase` are both given
        :raises KeyError: if any keyword argument is not recognised as obs/dep Parameter
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
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
                logger.error('data_fromarrays failed: components need to be provided')
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
        return dataref
    
    def data_fromexisting(self, to_dataref, from_dataref=None, category=None,
                          **kwargs):
        """
        Duplicate existing data to a new set with a different dataref.
        
        This can be useful if you want to change little things and want to
        examine the difference between computing options easily, or if you
        want to compute an existing set onto a higher time resolution etc.
        
        Any extra kwargs are copied to any pbdep or obs where the key is
        present.
        
        All *pbdeps* and *obs* with dataref :envvar:`from_dataref` will be
        duplicated into a *pbdep* with dataref :envvar:`to_dataref`.
        
        [FUTURE]
        
        :param category: category of the data to look for. If none are given,
        all types of data will be examined. This only has a real influence on
        the default value of :envvar:`from_dataref`.
        :type category: str, one of ``lc``, ``rv``...
        :raises KeyError: if :envvar:`to_dataref` already exists
        :raises KeyError: if :envvar:`from_dataref` is not given and there is
        either no or more than one dataset present.
        :raises ValueError: if keyword arguments are set that could not be
        processed.
        """
        # we keep track of every kwarg that has been used, to make sure
        # everything has found a place
        processed_kwargs = []
        
        # if no dataref is given, just take the only reference we have. If there
        # is more than one, this is ambiguous so we raise a KeyError
        if from_dataref is None:
            existing_refs = self.get_system().get_refs(category=category)
            if len(existing_refs) != 1:
                # build custom message depending on what is available
                if len(existing_refs) == 0:
                    msg = "no data present"
                else:
                    msg = ("more than one available. Please specify 'from_data"
                      "ref' to be any of {}").format(", ".join(existing_refs))
                raise KeyError(("Cannot figure out which dataref to copy from. "
                    "No 'from_dataref' was given and there is {}.").format(msg))
            
            from_dataref = existing_refs[0]
        
        # Walk through the system looking for deps, syns or obs
        for path, item in self.get_system().walk_all(path_as_string=False):
            if item == from_dataref:
                # now there's two possibilities: either we're at the root
                # parameterSet, or we have a Parameter with value dataref.
                # We're only interested in the former
                if isinstance(item, str):
                    # where are we at?
                    the_ordered_dict = path[-3]
                    the_context = path[-2]
                    
                    # check if the new data ref exists
                    if to_dataref in the_ordered_dict[the_context]:
                        raise KeyError(("Cannot add data from existing. "
                              "There already exists a {} with to_dataref "
                              "{}").format(the_context, to_dataref))
                    
                    # get the existing PS and copy it into a new one
                    existing = the_ordered_dict[the_context][from_dataref]
                    new = existing.copy()
                    the_ordered_dict[the_context][to_dataref] = new
                    
                    # update the reference, and any other kwargs that might
                    # be given. Remember if we added kwarg, in the end we'll
                    # check if everything found a place
                    new['ref'] = to_dataref
                    for key in kwargs:
                        if key in new:
                            new[key] = kwargs[key]
                            processed_kwargs.append(key)
                    
                    # Make sure to clear the synthetic
                    if the_context[-3:] == 'syn':
                        new.clear()
        
        # check if all kwargs found a place
        processed_kwargs = set(processed_kwargs)
        unprocessed_kwargs = []
        for key in kwargs:
            if not key in processed_kwargs:
                unprocessed_kwargs.append(key)
        if unprocessed_kwargs:
            raise ValueError(("Unprocessed arguments to *_fromexisting: "
                              "{}").format(", ".join(unprocessed_kwargs)))
            
                
        
    
    def lc_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      flux=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None):
        """
        Create and attach light curve templates to compute the model.
        
        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        A unique data reference (:envvar:`dataref`) is added automatically if
        none is provided by the user. A readable system of unique references is
        applied: the first light curve that is added is named ``lc01`` unless
        that reference already exists, in which case ``lc02`` is tried and so
        on. On the other hand, if a :envvar:`dataref` is given at it already
        exists, it's settings are overwritten.
        
        If no :envvar:`objref` is given, the light curve is added to the total
        system, and not to a separate component. That is probably fine in almost
        all cases, since you observe the whole system simultaneously and not the
        components separately.
        
        Note that you cannot add :envvar:`times` and :envvar:`phases` simultaneously.
        
        **Example usage**
        
        It doesn't make much sense to leave the time array empty, since then
        nothing will be computed. Thus, the minimal function call that makes
        sense is something like:
        
        >>> bundle = phoebe.Bundle()
        >>> bundle.lc_fromarrays(time=np.linspace(0, 10.33, 101))
        
        or in phase space (phase space will probably not work for anything but
        light curves and radial velocities):
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.lc_fromarrays(phase=phase, passband='GENEVA.V')
        
        With many more details:
        
        >>> bundle.lc_fromarrays(phase=phase, samprate=5, exptime=20.,
        ...     passband='GENEVA.V', atm='kurucz', ld_func='claret', 
        ...     ld_coeffs='kurucz')
        
        For a list of acceptable values for each parameter, see
        :ref:`lcdep <parlabel-phoebe-lcdep>` and
        :ref:`lcobs <parlabel-phoebe-lcobs>`.
        
        In general, :envvar:`time`, :envvar:`flux`, :envvar:`phase`,
        :envvar:`sigma`, :envvar:`flag`, :envvar:`weight, :envvar:`exptime` and
        :envvar:`samprate` should all be arrays of equal length (unless left to
        ``None``).
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises ValueError: if :envvar:`time` and :envvar:`phase` are both given
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
        
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        # We can pass everything now to the main function
        return self.data_fromarrays(category='lc', **set_kwargs)
    
    
    def lc_fromfile(self, filename, objref=None, dataref=None, columns=None,
                      units=None, offset=None, scale=None, atm=None,
                      ld_func=None, ld_coeffs=None, passband=None, pblum=None,
                      l3=None, alb=None, beaming=None, scattering=None):
        """
        Add a lightcurve from a file.
        
        The data contained in :envvar:`filename` will be loaded to the object
        with object reference :envvar:`objref` (if ``None``, defaults to the whole
        system), and will have data reference :envvar:`dataref`. If no
        :envvar:`dataref` is is given, a unique one is generated: the first 
        light curve that is added is named 'lc01', and if that one already
        exists, 'lc02' is tried and so on.
        
        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        See :py:func:`phoebe.parameters.datasets.parse_lc` for more information
        on file formats.
        
        **Example usage**
        
        A plain file can loaded via::
        
        >>> bundle.lc_fromarrays('myfile.lc')
        
        Note that extra parameters can be given in the file itself, but can
        also be overriden in the function call:
        
        >>> bundle.lc_fromarrays('myfile.lc', atm='kurucz')
        
        For a list of acceptable values for each parameter, see
        :ref:`lcdep <parlabel-phoebe-lcdep>` and
        :ref:`lcobs <parlabel-phoebe-lcobs>`.
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        # We can pass everything now to the main function
        return self.data_fromfile(category='lc', **set_kwargs)
    
    
    def lc_fromexisting(to_dataref, from_dataref=None, time=None, phase=None,
                      flux=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None):
        """
        Duplicate an existing light curve to a new one with a different dataref.
        
        This can be useful if you want to change little things and want to
        examine the difference between computing options easily, or if you
        want to compute an existing light curve onto a higher time resolution
        etc.
        
        Any extra kwargs are copied to any lcdep or lcobs where the key is
        present.
        
        All *lcdeps* and *lcobs* with dataref :envvar:`from_dataref` will be
        duplicated into an *lcdep* with dataref :envvar:`to_dataref`.
        
        For a list of available parameters, see :ref:`lcdep <parlabel-phoebe-lcdep>`
        and :ref:`lcobs <parlabel-phoebe-lcobs>`.
        
        :raises KeyError: if :envvar:`to_dataref` already exists
        :raises KeyError: if :envvar:`from_dataref` is not given and there is
         either no or more than one light curve present.
        :raises ValueError: if keyword arguments are set that could not be
         processed.
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        self.data_fromexisting(to_dataref, category='lc', **set_kwargs)
        
    def rv_fromarrays(self, objref, dataref=None, time=None, phase=None,
                      rv=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None):
        """
        Create and attach radial velocity curve templates to compute the model.
        
        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        A unique data reference (:envvar:`dataref`) is added automatically if
        none is provided by the user. A readable system of unique references is
        applied: the first rv curve that is added is named ``rv01`` unless
        that reference already exists, in which case ``rv02`` is tried and so
        on. On the other hand, if a :envvar:`dataref` is given at it already
        exists, it's settings are overwritten.
        
        Note that you cannot add :envvar:`times` and `phases` simultaneously.
        
        **Example usage**
        
        It doesn't make much sense to leave the time array empty, since then
        nothing will be computed. Thus, the minimal function call that makes
        sense is something like:
        
        >>> bundle.rv_fromarrays(time=np.linspace(0, 10.33, 101))
        
        or in phase space (phase space will probably not work for anything but
        light curves and radial velocities):
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.rv_fromarrays(phase=phase, passband='GENEVA.V')
        
        With many more details:
        
        >>> bundle.rv_fromarrays(phase=phase, samprate=5, exptime=20.,
        ...     passband='GENEVA.V', atm='kurucz', ld_func='claret', 
        ...     ld_coeffs='kurucz')
        
        For a list of acceptable values for each parameter, see
        :ref:`rvdep <parlabel-phoebe-rvdep>` and
        :ref:`rvobs <parlabel-phoebe-rvobs>`.
        
        In general, :envvar:`time`, :envvar:`flux`, :envvar:`phase`,
        :envvar:`sigma`, :envvar:`flag`, :envvar:`weight, :envvar:`exptime` and
        :envvar:`samprate` should all be arrays of equal length (unless left to
        ``None``).
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises ValueError: if :envvar:`time` and :envvar:`phase` are both given
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
        """
        if objref == self.get_system().get_label():
            raise ValueError("Cannot add RV to the system, only to the components")
        
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key not in ['self','objref']}
        
        # We can pass everything now to the main function
        return self.data_fromarrays(category='rv', objref=objref, **set_kwargs)
    
    
    def rv_fromfile(self, filename, objref=None, dataref=None, columns=None,
                      units=None, offset=None, scale=None, atm=None,
                      ld_func=None, ld_coeffs=None, passband=None, pblum=None,
                      l3=None, alb=None, beaming=None, scattering=None):
        """
        Add a radial velocity curve from a file.
        
        The data contained in :envvar:`filename` will be loaded to the object
        with object reference :envvar:`objref`, and will have data reference
        :envvar:`dataref`. If no :envvar:`dataref` is is given, a unique one is
        generated: the first radial velocity curve that is added is named 'rv01',
        and if that one already exists, 'rv02' is tried and so on.
        
        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        See :py:func:`phoebe.parameters.datasets.parse_rv` for more information
        on file formats.
        
        **Example usage**
        
        A plain file can loaded via::
        
        >>> bundle.rv_fromarrays('myfile.rv')
        
        Note that extra parameters can be given in the file itself, but can
        also be overriden in the function call:
        
        >>> bundle.rv_fromarrays('myfile.rv', atm='kurucz')
        
        For a list of acceptable values for each parameter, see
        :ref:`rvdep <parlabel-phoebe-rvdep>` and
        :ref:`rvobs <parlabel-phoebe-rvobs>`.
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        # We can pass everything now to the main function
        return self.data_fromfile(category='rv', **set_kwargs)
    
    
    def rv_fromexisting(to_dataref, from_dataref=None, time=None, phase=None,
                      rv=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None):
        """
        Duplicate an existing radial velocity curve to a new one with a different dataref.
        
        This can be useful if you want to change little things and want to
        examine the difference between computing options easily, or if you
        want to compute an existing radial velocity curve onto a higher time
        resolution etc.
        
        Any extra kwargs are copied to any rvdep or rvobs where the key is
        present.
        
        All *rvdeps* and *rvobs* with dataref :envvar:`from_dataref` will be
        duplicated into an *rvdep* with dataref :envvar:`to_dataref`.
        
        :raises KeyError: if :envvar:`to_dataref` already exists
        :raises KeyError: if :envvar:`from_dataref` is not given and there is
        either no or more than one radial velocity curve present.
        :raises ValueError: if keyword arguments are set that could not be
        processed.
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        self.data_fromexisting(to_dataref,  category='rv', **set_kwargs)
    
    def sed_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                       passband=None, flux=None, sigma=None, unit=None,
                       scale=False, offset=False, **kwargs):
        """
        Create and attach SED templates to compute the model.
        
        A spectral energy distribution (SED) is nothing more than a collection
        of absolutely calibrated lcs in different passbands. The given arrays
        of times, flux etc.. therefore need to be all arrays of the same lenght,
        similarly as for :py:func:`Bundle.lc_fromarrays`. One extra array needs
        to be given, i.e. a list of passbands via :envvar:`passband`.
        Optionally, a list of units can be added (i.e. the supplied fluxes can
        have different units, e.g. mag, erg/s/cm2/AA, Jy...).
        
        Extra keyword arguments are all passed to :py:func:`Bundle.lc_fromarrays`.
        That means that each lc attached will have the same set of atmosphere
        tables, limb darkening coefficients etc. If they need to be different
        for particular lcs, these need to be changed manually afterwards.
        
        Each added light curve will be named ``<dataref>_<passband>``, so they
        can be accessed using that twig.
        
        Note that many SED measurements are not recorded in time. They still
        need to be given in Phoebe2 anyway, e.g. all zeros.
        
        **Example usage**
        
        Initiate a Bundle:
        
        >>> vega = phoebe.Bundle('Vega')
        
        Create the required/optional arrays
        
        >>> passbands = ['JOHNSON.V', 'JOHNSON.U', '2MASS.J', '2MASS.H', '2MASS.KS']
        >>> flux = [0.033, 0.026, -0.177, -0.029, 0.129]
        >>> sigma = [0.012, 0.014, 0.206, 0.146, 0.186]
        >>> unit = ['mag', 'mag', 'mag', 'mag', 'mag']
        >>> time = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        And add them to the Bundle.
        
        >>> x.sed_fromarrays(dataref='mysed', passband=passbands, time=time, flux=flux,
                 sigma=sigma, unit=unit)
        
        [FUTURE]
        """
        # We need a reference to link the pb and ds together.
        if dataref is None:
            # If the reference is None, suggest one. We'll add it as "lc01"
            # if no "sed01" exist, otherwise "sed02" etc (unless "sed02"
            # already exists)
            existing_refs = self.get_system().get_refs(category='lc')
            id_number = len(existing_refs)+1
            dataref = 'sed{:02d}'.format(id_number)
            while dataref in existing_refs:
                id_number += 1
                dataref = 'sed{:02d}'.format(len(existing_refs)+1)            
        
        # group data per passband:
        passbands = np.asarray(passband)
        unique_passbands = np.unique(passbands)
    
        # Convert fluxes to the right units
        if unit is not None:
            if sigma is None:
                flux = np.array([conversions.convert(iunit, 'W/m3', iflux,\
                          passband=ipassband) for iunit, iflux, ipassband \
                              in zip(unit, flux, passbands)])
            else:
                flux, sigma = np.array([conversions.convert(iunit, 'W/m3',
                          iflux, isigma, passband=ipassband) for iunit, iflux,\
                              ipassband, isigma in zip(unit, flux, \
                                  passbands, sigma)]).T
    
        # Group per passband
        split_up = ['time', 'phase', 'flux', 'sigma']
        added_datarefs = []
        for unique_passband in unique_passbands:
            this_group = (passbands == unique_passband)
            this_kwargs = kwargs.copy()
            this_dataref = dataref + '_' + unique_passband
            
            # Split given arrays per passband
            for variable in split_up:
                if locals()[variable] is not None:
                    this_kwargs[variable] = np.array(locals()[variable])[this_group]
            
            # And add these as a light curve
            added = self.lc_fromarrays(objref=objref, dataref=this_dataref,
                                       passband=unique_passband, **this_kwargs)
            
            added_datarefs.append(added)
            
        # Group the observations, but first collect them all
        this_object = self.get_object(objref)
        obs = [this_object.get_obs(category='lc', ref=iref) \
                     for iref in added_datarefs]
        
        tools.group(obs, dataref, scale=scale, offset=offset)
        
        return dataref
    
    def sed_fromfile(self, filename, objref=None, dataref=None, columns=None,
                      units=None, offset=None, scale=None):
        """
        Add SED templates from a file.
        
        [FUTURE]
        """
        # We need a reference to link the pb and ds together.
        if dataref is None:
            # If the reference is None, suggest one. We'll add it as "lc01"
            # if no "sed01" exist, otherwise "sed02" etc (unless "sed02"
            # already exists)
            existing_refs = self.get_system().get_refs(category='lc')
            id_number = len(existing_refs)+1
            dataref = 'sed{:02d}'.format(id_number)
            while dataref in existing_refs:
                id_number += 1
                dataref = 'sed{:02d}'.format(len(existing_refs)+1)            
        
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        # We can pass everything now to the main function
        return self.data_fromfile(category='sed', **set_kwargs)
    
    def add_parameter(self, twig, replaces=None, value=None):
        """
        Add a new parameter to the set of parameters.
        
        The value of the new parameter can either be derived from the existing
        ones, or replaces one of the existing ones.
        
        If we want the replacable parameter to not show up in __str__, we need
        to rewrite the string representation part. Otherwise the backend needs
        to deal with 'hidden' stuff, and these are already implemented by the
        frontend.
        
        [FUTURE]
        """
        # Does the parameter already exist?
        param = self._get_by_search(twig, kind='Parameter', ignore_errors=True,
                                    return_trunk_item=True)
        
        # If the parameter does not exist, we need to create a new parameter
        # and add it in the right place in the tree.
        twig_split = twig.split('@')
        
        # Figure out what the parameter name is
        qualifier = twig_split[0]
        
        if replaces is None:
            replaces = qualifier
        if value is None:
            value = 1.0
        
        # If the parameter does not exist yet, there's some work to do
        if param is None:
            # Get all the info on this parameter
            info = definitions.rels['binary'][qualifier]
            
            # Figure out which level (i.e. ParameterSet) to add it to
            in_level_as = info.pop('in_level_as')
            twig_rest = '@'.join([in_level_as] + twig_split[1:])
            item = self._get_by_search(twig_rest, kind='Parameter',
                                         return_trunk_item=True)
            
            # And add it
            pset = item['path'][-2]
            pset.add(info)
            
            param = pset.get_parameter(qualifier)
            param.set_replaces(replaces)
            
        # In any case we need to set the 'replaces' attribute and the value    
        param.set_replaces(replaces)
        param.set_value(value)
            
        # add the preprocessing thing
        system = self.get_system()
        system.add_preprocess('binary_custom_variables')
        self._build_trunk()    
    
    
    def get_datarefs(self, objref=None, category=None, per_category=False):
        """
        Return all the datarefs, or only those of a certain category.
        
        [FUTURE]
        """
        return self.get_object(objref).get_refs(category=category,
                                                per_category=per_category)
    
    def get_lc_datarefs(self, objref=None):
        """
        Return all datarefs of category lc.
        
        [FUTURE]
        """
        return self.get_datarefs(objref=objref, category='lc')
    
    def get_rv_datarefs(self, objref=None):
        """
        Return all datarefs of category rv.
        
        [FUTURE]
        """
        return self.get_datarefs(objref=objref, category='rv')
    
    def get_syn(self, twig=None):
        """
        Get the synthetic parameterset for an observation
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @return: the observations DataSet
        @rtype: DataSet
        """
        return self._get_by_search(twig, context='*syn', class_name='*DataSet')

    def get_dep(self, twig=None):
        """
        Get observations dependables
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @return: the observations dependables ParameterSet
        @rtype: ParameterSet
        """
        return self._get_by_search(twig, context='*dep', class_name='ParameterSet')
        
    def get_obs(self, twig=None):
        """
        Get observations
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @return: the observations DataSet
        @rtype: DataSet
        """
        return self._get_by_search(twig, context='*obs', class_name='*DataSet')
        
    def enable_data(self, dataref=None, category=None, enabled=True):
        """
        Enable a dataset so that it will be considered during run_compute
        
        If the category is given and :envvar:`dataref=None`, the dataset to
        disable must be unique, i.e. there can be only one. Otherwise, a
        ValueError will be raised.
        
        @param dataref: reference of the dataset
        @type dataref: str
        @param category: the category of the dataset ('lc', 'rv', etc)
        @type category: str
        @param enabled: whether to enable (True) or disable (False)
        @type enabled: bool
        :raises ValueError: when the dataref is ambiguous, or is None
        :raises ValueEror: if dataref is None and no category is given
        :raises KeyError: when dataref is not available
        """
        dataref = self._process_dataref(dataref, category)
        
        if dataref is not None:
            system = self.get_system()
            try:
                iterate_all_my_bodies = system.walk_bodies()
            except AttributeError:
                iterate_all_my_bodies = [system]
            
            for body in iterate_all_my_bodies:
                this_objref = body.get_label()
                #~ if objref is None or this_objref == objref:
                if True:
                    for obstype in body.params['obs']:
                        if dataref is None:
                            for idataref in body.params['obs'][obstype]:
                                body.params['obs'][obstype][idataref].set_enabled(enabled)
                                logger.info("{} {} '{}'".format('Enabled' if enabled else 'Disabled', obstype, idataref))
                        elif dataref in body.params['obs'][obstype]:
                            body.params['obs'][obstype][dataref].set_enabled(enabled)
                            logger.info("{} {} '{}'".format('Enabled' if enabled else 'Disabled', obstype, dataref))

        
    def disable_data(self, dataref=None, category=None):
        """
        Disable a dataset so that it will not be considered during run_compute
        
        See :py:func:`Bundle.enable_data` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        @param category: the category of the dataset ('lc', 'rv', etc)
        @type category: str
        :raises ValueError: when the dataref is ambiguous, or is None and no category is given.
        """
        self.enable_data(dataref, category, enabled=False)
        
    def enable_lc(self, dataref=None):
        """
        Enable an LC dataset so that it will be considered during run_compute
        
        If no dataref is given and there is only one light curve added, there
        is no ambiguity and that one will be enabled.
        
        @param dataref: reference of the dataset
        @type dataref: str
        :raises ValueError: when the dataref is ambiguous
        :raises KeyError: when dataref does not exist
        """
        self.enable_data(dataref, 'lc', True)
        
    def disable_lc(self, dataref=None):
        """
        Disable an LC dataset so that it will not be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info.
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'lc', False)
        
    def enable_rv(self, dataref=None):
        """
        Enable an RV dataset so that it will be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'rv', True)
        
    def disable_rv(self, dataref=None):
        """
        Disable an RV dataset so that it will not be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'rv', False)
        
    def reload_obs(self, twig=None):
        """
        [FUTURE]

        reload a dataset from its source file
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        """
        self.get_obs(twig).load()
        
        #~ dss = self.get_obs(dataref=dataref,all=True).values()
        #~ for ds in dss:
            #~ ds.load()
    
    def _process_dataref(self, dataref, category=None):
        """
        [FUTURE]
        
        this function handles checking if a dataref passed to a function 
        (eg. remove_data, enable_data, etc) is unique to a single category
        
        this function also handles determining a default if dataref is None
        """
        if dataref is None:
            # then see if there is only one entry with this category
            # and if so, default to it
            if category is None: 
                category = '*'
            dss = self._get_by_search(dataref, 
                    context = ['{}obs'.format(category),'{}syn'.format(category),'{}dep'.format(category)], 
                    kind = 'ParameterSet', all = True, ignore_errors = True)
            datarefs = []
            for ds in dss:
                if ds['ref'] not in datarefs:
                    datarefs.append(ds['ref'])
            if len(datarefs)==1:
                # this is our default!
                return datarefs[0]
            elif len(datarefs)==0:
                raise ValueError("no datasets found")
                # no default
                return None
            else:
                raise ValueError("more than one dataset: must provide dataref")
                # no default
                return None
        else:
            # confirm its (always) the correct category
            # *technically* if it isn't always correct, we should just remove the correct ones
            #dss = self._get_by_search(dataref, context=['*obs','*syn','*dep'], kind='ParameterSet', all=True, ignore_errors=True)
            dss = self._get_by_search(dataref, context='*dep', kind='ParameterSet', all=True, ignore_errors=True)
            if not len(dss):
                raise KeyError("No dataref found matching {}".format(dataref))
            for ds in dss:
                if category is None:
                    # then instead we're just making sure that all are of the same type
                    category = ds.context[:-3]
                if ds.context[:-3] != category:
                    raise ValueError("{} not always of category {}".format(dataref, category))
                    # forbid this dataref
                    return None
            
            # we've survived, this dataref is allowed
            return dataref


    def remove_data(self, dataref=None, category=None):
        """
        remove a dataset (and all its obs, syn, dep) from the system
        
        @param dataref: reference of the dataset
        @type dataref: str
        @param category: the category of the dataset ('lc', 'rv', etc)
        @type category: str
        """
        
        dataref = self._process_dataref(dataref, category)
        
        if dataref is not None:
            # disable any plotoptions that use this dataset
            #~ for axes in self.get_axes(all=True).values():
                #~ for pl in axes.get_plot().values():
                    #~ if pl.get_value('dataref')==dataref:
                        #~ pl.set_value('active',False)
            
            # remove all obs attached to any object in the system
            for obj in self.get_system().walk_bodies():
                obj.remove_obs(refs=[dataref])
                if hasattr(obj, 'remove_pbdeps'): #TODO otherwise: warning 'BinaryRocheStar' has no attribute 'remove_pbdeps'
                    obj.remove_pbdeps(refs=[dataref]) 
            
            self._build_trunk()
            return
    
    def remove_lc(self, dataref=None):
        """
        remove an LC dataset (and all its obs, syn, dep) from the system
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.remove_data(dataref, 'lc')


    def remove_rv(self, dataref=None):
        """
        remove an RV dataset (and all its obs, syn, dep) from the system
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.remove_data(dataref, 'rv')
        
        
        
        
    #}
    #{ Compute
    
    @rebuild_trunk
    @run_on_server
    def run_compute(self, label=None, objref=None, animate=False, **kwargs):
    #~ def run_compute(self,label=None,anim=False,add_version=None,server=None,**kwargs):
        """
        Perform calculations to mirror any enabled attached observations.
        
        Main arguments: :envvar:`label`, :envvar:`objref`, :envvar:`anim`.
        
        Extra keyword arguments are passed to the
        :ref:`compute <parlabel-phoebe-compute>` ParameterSet.
        
        **Example usage**
        
        The minimal setup is::
        
            >>> mybundle = phoebe.Bundle()
            >>> dataref = mybundle.lc_fromarrays(phase=[0, 0.5, 1.0])
            >>> mybundle.run_compute()
        
        After which you can plot the results via::
        
            >>> mybundle.plot_syn(dataref)
        
        **Keyword 'label'**
        
        Different compute options can be added via
        :py:func:`Bundle.add_compute() <phoebe.frontend.common.Container.add_compute>`,
        where each of these ParameterSets have a
        :envvar:`label`. If :envvar:`label` is given and that compute option is
        present, those options will be used. If no :envvar:`label` is given, a
        default set of compute options is created on the fly. The used set of
        options is returned but also stored for later reference. You can access
        it via the ``default`` label in the bundle::
            
            >>> mybundle.run_compute()
        
        and at any point you can query:
        
            >>> options = mybundle['default@compute']
            
        If you want to store new options before hand for later usage you can
        issue:
        
            >>> mybundle.add_compute(label='no_heating', heating=False)
            >>> options = mybundle.run_compute(label='no_heating')
           
        **Keyword 'objref'**
        
        If :envvar:`objref` is given, the computations are only performed on
        that object. This is only advised for introspection. Beware that the
        synthetics will no longer mirror the observations of the entire system,
        but only those of the specified object.
        
        .. warning::
            
            1. Even if you only compute the light curve of the secondary in a
               binary system, the system geometry is still determined by the entire
               system. Thus, eclipses will occur if the secondary gets eclipsed!
               If you don't want this behaviour, either turn of eclipse computations
               entirely (via :envvar:`eclipse_alg='none'`) or create a new
               binary system with the other component removed.
            
            2. This function only computes synthetics of objects that have
               observations. If observations are added to the system level and
               :envvar:`run_compute` is run on a component where there are no
               observations attached to, a :envvar:`ValueError` will be raised.
        
        **Keyword 'animate'**
        
        It is possible to animate the computations for visual inspection of the
        system geometry. The different valid options for setting
        :envvar:`animate` are given in
        :py:func:`observatory.compute <phoebe.backend.observatory.compute>`,
        but here are two straightforward examples::
        
            >>> mybundle.run_compute(animate=True)
            >>> mybundle.run_compute(animate='lc')
        
        .. warning::
        
            1. Animations are only supported on computers/backends that support the
               animation capabilities of matplotlib (likely excluding Macs).
            
            2. Animations will not work in interactive mode in ipython (i.e. when
               started as ``ipython --pylab``.
        
        **Extra keyword arguments**
        
        Any extra keyword arguments are passed on to the ``compute``
        ParameterSet.
        
        This frontend function wraps the backend function
        :py:func:`observatory.compute <phoebe.backend.observatory.compute>`.
        
        :param label: name of one of the compute ParameterSets stored in bundle
        :type label: str
        :param objref: name of the top-level object used when observing
        :type objref: str
        :param anim: whether to animate the computations
        :type anim: False or str
        :return: used compute options
        :rtype: ParameterSet
        :raises ValueError: if there are no observations (or only empty ones) attached to the object.
        :raises KeyError: if a label of a compute options is set that was not added before
        """
        server = None # server support deferred 
        system = self.get_system()
        system.fix_mesh()
        obj = self.get_object(objref) if objref is not None else system
        #~ if add_version is None:
            #~ add_version = self.settings['add_version_on_compute']
                
        self.purge_signals(self.attached_signals_system)
            
        # clear all previous models and create new model
        system.clear_synthetic()
        
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
        if options['time'] == 'auto':
            #~ observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
            if mpi is not None and animate:
                raise ValueError("You cannot animate and use MPI simultaneously")
            elif mpi is not None:
                obj.compute(mpi=mpi, **options)
            else:
                obj.compute(animate=animate, **options)
            
        #else:
            #im_extra_func_kwargs = {key: value for key,value in self.get_meshview().items()}
            #observatory.observe(obj,options['time'],lc=True,rv=True,sp=True,pl=True,
                #extra_func=[observatory.ef_binary_image] if anim!=False else [],
                #extra_func_kwargs=[self.get_meshview()] if anim!=False else [],
                #mpi=mpi,**options
                #)
        
        #if anim != False:
            #for ext in ['.gif','.avi']:
                #plotlib.make_movie('ef_binary_image*.png',output='{}{}'.format(anim,ext),cleanup=ext=='.avi')
            
        system.uptodate = label
        
        #~ if add_version is not False:
            #~ self.add_version(name=None if add_version==True else add_version)

        self.attach_system_signals()
        
        return options
        
    #}
            
    #{ Fitting
    @rebuild_trunk
    @run_on_server
    def run_fitting(self,computelabel=None,fittinglabel=None,add_feedback=None,accept_feedback=False,server=None,**kwargs):
        """
        Run fitting for a given fitting ParameterSet
        and store the feedback
        
        [FUTURE]
        
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
            fittingoptions = parameters.ParameterSet(context='fitting:lmfit')
        else:
            fittingoptions = self.get_fitting(fittinglabel).copy()
         
        # get compute params
        if computelabel is None:
            computeoptions = parameters.ParameterSet(context='compute')
        else:
            computeoptions = self.get_compute(computelabel).copy()

        # now temporarily override with any values passed through kwargs    
        #for k,v in kwargs.items():
        #    if k in options.keys():
        #        options.set_value(k,v)
            
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
    
    def set_syn_as_obs(self, dataref, sigma=0.01):
        """
        Set synthetic computations as if they were really observed.
        
        This can be handy to experiment with the fitting routines.
        
        [FUTURE]
        """
        syn = self._get_by_search(dataref, context='*syn', class_name='*DataSet')
        obs = self._get_by_search(dataref, context='*obs', class_name='*DataSet')
        
        if obs.get_context()[:-3] == 'lc':
            if np.isscalar(sigma):
                sigma = sigma*np.median(syn['flux'])*np.ones(len(syn))
            obs['flux'] = syn['flux'] + np.random.normal(size=len(obs), scale=sigma)
            obs['sigma'] = sigma
        
        elif obs.get_context()[:-3] == 'rv':
            if np.isscalar(sigma):
                sigma = sigma*np.median(syn['rv'])
            obs['rv'] = syn['rv'] + np.random.normal(size=len(obs), scale=sigma)
            obs['sigma'] = sigma
            
    
    #{ Figures
    def plot_obs(self, twig=None, **kwargs):
        """
        Make a plot of the attached observations (wraps pyplot.errorbar).
        
        This function is designed to behave like matplotlib's
        `plt.errorbar() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_
        function, with additional options.
        
        Thus, all kwargs (there are no args) are passed on to matplotlib's
        `plt.errorbars() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_,
        except:
    
            - :envvar:`phased=False`: decide whether to phase the data or not.
              The default is ``True`` when the observations are phased. You can
              unphase them in that case by setting :envvar:`phased=False`
              explicitly. This setting is trumped by :envvar:`x_unit` (see
              below).
            - :envvar:`repeat=0`: handy if you are actually fitting a phase
              curve, and you want to repeat the phase curve a couple of times.
            - :envvar:`x_unit=None`: allows you to override the default units
              for the x-axis. If you plot times, you can set the unit to any
              time unit (days (``d``), seconds (``s``), years (``yr``) etc.). If
              you plot in phase, you can switch from cycle (``cy``) to radians
              (``rad``). This setting trumps :envvar:`phased`: if the x-unit is
              of type phase, the data will be phased and if they are time, they
              will be in time units.
            - :envvar:`y_unit=None`: allows you to override the default units
              for the y-axis. Allowable values depend on the type of
              observations.
            - :envvar:`ax=plt.gca()`: the axes to plot on. Defaults to current
              active axes.
    
        Some of matplotlib's defaults are overriden. If you do not specify any
        of the following keywords, they will take the values:
    
            - :envvar:`label`: the label for the legend defaults to
              ``<ref> (obs)``. If you don't want a label for this curve, set
              :envvar:`label='_nolegend_'`.
            - :envvar:`yerr`: defaults to the uncertainties from the obs if they
              are available.
        
        The DataSet that is returned is a copy of the original DataSet, but with the
        units of the columns the same as the ones plotted.
    
        **Example usage**
            
        Suppose you have the following setup::
        
            bundle = phoebe.Bundle()
            bundle.lc_fromarrays(dataref='mylc', time=np.linspace(0, 1, 100),
            ...                  flux=np.random.normal(size=100))
            bundle.rv_fromarrays(dataref='myrv', objref='secondary',
            ...                  phase=np.linspace(0, 1, 100),
            ...                  rv=np.random.normal(size=100),
            ...                  sigma=np.ones(100))
       
       Then you can plot these observations with any of the following commands::
            
            bundle.plot_obs('mylc')
            bundle.plot_obs('mylc', phased=True)
            bundle.plot_obs('mylc', phased=True, repeat=1)
            bundle.plot_obs('myrv@secondary')
            bundle.plot_obs('myrv@secondary', fmt='ko-')
            plt.legend()
            bundle.plot_obs('myrv@secondary', fmt='ko-', label='my legend label')
            plt.legend()
            bundle.plot_obs('myrv@secondary', fmt='ko-', x_unit='s', y_unit='nRsol/d')
        
        For more explanations and a list of defaults for each type of
        observaitons, see:
        
            - :py:func:`plot_lcobs <phoebe.backend.plotting.plot_lcobs>`: for light curve plots
            - :py:func:`plot_rvobs <phoebe.backend.plotting.plot_rvobs>`: for radial velocity plots
        
        The arguments are passed to the appropriate functions in
        :py:mod:`plotting`.
        
        For more info on :envvar:`kwargs`, see the
        pyplot documentation on `plt.errorbars() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_,
        
        :param twig: the twig/twiglet to use when searching
        :type twig: str
        :return: observations used for plotting
        :rtype: DataSet
        :raises IOError: when observations are not available
        :raises ValueError: when x/y unit is not allowed
        :raises ValueError: when y-axis not available (flux, rv...)
        """
        # Retrieve the obs DataSet and the object it belongs to
        dsti = self._get_by_search(twig, context='*obs', class_name='*DataSet',
                                   return_trunk_item=True)
        ds = dsti['item']
        obj = self.get_object(dsti['label'])
        context = ds.get_context()
        
        # Do we need automatic/custom xlabel, ylabel and/or title? We need to
        # pop the kwargs here because they cannot be passed to the lower level
        # plotting function
        xlabel = kwargs.pop('xlabel', '_auto_')
        ylabel = kwargs.pop('ylabel', '_auto_')
        title = kwargs.pop('title', '_auto_')
        
        # Now pass everything to the correct plotting function in the backend
        kwargs['ref'] = ds['ref']
        output = getattr(plotting, 'plot_{}'.format(context))(obj, **kwargs)
        
        # Now take care of figure decorations
        fig_decs = output[2]
        artists = output[0]
        obs = output[1]
        
        # The x-label
        if xlabel == '_auto_':
            plt.xlabel(r'{} ({})'.format(fig_decs[0][0], fig_decs[1][0]))
        elif xlabel:
            plt.xlabel(xlabel)
        
        # The y-label
        if ylabel == '_auto_':
            plt.ylabel(r'{} ({})'.format(fig_decs[0][1], fig_decs[1][1]))
        elif ylabel:
            plt.ylabel(ylabel)
        
        # The plot title
        if title == '_auto_':
            plt.title('{}'.format(config.nice_names[context[:-3]]))
        elif title:
            plt.title(title)        
        
        logger.info("Plotted {} vs {} of {}({})".format(fig_decs[0][0],
                                   fig_decs[0][1], context, ds['ref']))
        
        return obs

        
    def plot_syn(self, twig=None, *args, **kwargs):
        """
        Plot simulated/computed observations (wraps pyplot.plot).
        
        This function is designed to behave like matplotlib's
        `plt.plot() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_
        function, with additional options.
        
        Thus, all args and kwargs are passed on to matplotlib's
        `plt.plot() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_,
        except:
        
            - :envvar:`ref=0`: the reference of the lc to plot
            - :envvar:`phased=False`: decide whether to phase the data or not. If
              there are observations corresponding to :envvar:`ref`, the default
              is ``True`` when those are phased. The setting is overridden
              completely by ``x_unit`` (see below).
            - :envvar:`repeat=0`: handy if you are actually fitting a phase curve,
              and you want to repeat the phase curve a couple of times.
            - :envvar:`x_unit=None`: allows you to override the default units for
              the x-axis. If you plot times, you can set the unit to any time unit
              (days (``d``), seconds (``s``), years (``yr``) etc.). If you plot
              in phase, you switch from cycle (``cy``) to radians (``rad``). The
              :envvar:`x_unit` setting has preference over the :envvar:`phased`
              flag: if :envvar:`phased=True` but :envvar:`x_unit='s'`, then still
              the plot will be made in time, not in phase.
            - :envvar:`y_unit=None`: allows you to override the default units for
              the y-axis.
            - :envvar:`scale='obs'`: correct synthetics for ``scale`` and ``offset``
              from the observations. If ``obs``, they will effectively be scaled to
              the level/units of the observations (if that was specified in the
              computations as least). If you want to plot the synthetics in the
              model units, set ``scale=None``.
            - :envvar:`ax=plt.gca()`: the axes to plot on. Defaults to current
              active axes.
        
        Some of matplotlib's defaults are overriden. If you do not specify any of
        the following keywords, they will take the values:
        
            - :envvar:`label`: the label for the legend defaults to ``<ref> (syn)``.
              If you don't want a label for this curve, set :envvar:`label=_nolegend_`.
        
            
        Example usage:
        
        >>> mybundle.plot_syn('lc01', 'r-', lw=2) # first light curve added via 'data_fromarrays'
        >>> mybundle.plot_syn('rv01@primary', 'r-', lw=2, scale=None)
        
        >>> mybundle.plot_syn('if01', 'k-') # first interferometry added via 'data_fromarrays'
        >>> mybundle.plot_syn('if01', 'k-', y='vis2') # first interferometry added via 'data_fromarrays'
        
        More information on arguments and keyword arguments:
        
        - :py:func:`phoebe.backend.plotting.plot_lcsyn`
        - :py:func:`phoebe.backend.plotting.plot_rvsyn`
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        """
        # Retrieve the obs DataSet and the object it belongs to
        dsti = self._get_by_search(twig, context='*syn', class_name='*DataSet',
                                   return_trunk_item=True)
        ds = dsti['item']
        obj = self.get_object(dsti['label'])
        context = ds.get_context()
        
        # Do we need automatic/custom xlabel, ylabel and/or title? We need to
        # pop the kwargs here because they cannot be passed to the lower level
        # plotting function
        xlabel = kwargs.pop('xlabel', '_auto_')
        ylabel = kwargs.pop('ylabel', '_auto_')
        title = kwargs.pop('title', '_auto_')
        
        # Now pass everything to the correct plotting function
        kwargs['ref'] = ds['ref']
        output = getattr(plotting, 'plot_{}'.format(context))(obj, *args, **kwargs)
        syn = output[1]
        fig_decs = output[2]
        
        # The x-label
        if xlabel == '_auto_':
            plt.xlabel(r'{} ({})'.format(fig_decs[0][0], fig_decs[1][0]))
        elif xlabel:
            plt.xlabel(xlabel)
        
        # The y-label
        if ylabel == '_auto_':
            plt.ylabel(r'{} ({})'.format(fig_decs[0][1], fig_decs[1][1]))
        elif ylabel:
            plt.ylabel(ylabel)
        
        # The plot title
        if title == '_auto_':
            plt.title('{}'.format(config.nice_names[context[:-3]]))
        elif title:
            plt.title(title)
            
        return syn
        
    def plot_residuals(self, twig=None, **kwargs):
        """
        Plot the residuals between computed and observed for a given dataset
        
        [FUTURE]
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        """
        # Retrieve the obs DataSet and the object it belongs to
        dsti = self._get_by_search(twig, context='*syn', class_name='*DataSet',
                                   return_trunk_item=True)
        ds = dsti['item']
        obj = self.get_object(dsti['label'])
        category = ds.get_context()[:-3]
        
        # Do we need automatic/custom xlabel, ylabel and/or title? We need to
        # pop the kwargs here because they cannot be passed to the lower level
        # plotting function
        xlabel = kwargs.pop('xlabel', '_auto_')
        ylabel = kwargs.pop('ylabel', '_auto_')
        title = kwargs.pop('title', '_auto_')
        
        # Now pass everything to the correct plotting function
        kwargs['ref'] = ds['ref']
        output = getattr(plotting, 'plot_{}res'.format(category))(obj, **kwargs)
        obs, syn = output[1]
        fig_decs = output[2]
        
        # The x-label
        if xlabel == '_auto_':
            plt.xlabel(r'{} ({})'.format(fig_decs[0][0], fig_decs[1][0]))
        elif xlabel:
            plt.xlabel(xlabel)
        
        # The y-label
        if ylabel == '_auto_':
            plt.ylabel(r'{} ({})'.format(fig_decs[0][1], fig_decs[1][1]))
        elif ylabel:
            plt.ylabel(ylabel)
        
        # The plot title
        if title == '_auto_':
            plt.title('{}'.format(config.nice_names[category]))
        elif title:
            plt.title(title)
            
        return obs, syn
    
    
    def plot_prior(self, twig=None, **kwargs):
        """
        Plot a prior.
        
        [FUTURE]
        """
        prior = self.get_prior(twig)
        prior.plot(**kwargs)
        
        
    
    def write_syn(self, twig, output_file):
        """
        [FUTURE]
        
        Export the contents of a synthetic parameterset to a file
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @param output_file: path and filename of the exported file
        @type output_file: str
        """
        ds = self.get_syn(twig)
        ds.save(output_file)
        
    def get_axes(self,ident=None):
        """
        Return an axes or list of axes that matches index OR title
        
        [FUTURE]
        
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        
        @param time: the time to highlight
        @type time: float or None
        """
        self.select_time = time
        #~ self.system.set_time(time)
        
    def get_meshview(self,label=None):
        """
        [FUTURE]
        """
        return self._get_by_section(label,"meshview")
      
        
    def _get_meshview_limits(self,times):
        """
        [FUTURE]
        """
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
        
    def plot_mesh(self, objref=None, label=None, dataref=None, time=None, phase=None,
                  select='proj', cmap=None, vmin=None, vmax=None, size=800,
                  dpi=80, background=None, savefig=False,
                  with_partial_as_half=False, **kwargs):
        """
        Plot the mesh at a particular time or phase.
        
        This function has a lot of different use cases, which are all explained
        below.
        
        **Plotting the mesh at an arbitrary time or phase point**
        
        If you want to plot the mesh at a particular phase, give
        :envvar:`time` or :envvar:`phase` as a single float (but not both!):
        
            >>> mybundle.plot_mesh(time=0.12)
            >>> mybundle.plot_mesh(phase=0.25, select='teff')
            >>> mybundle.plot_mesh(phase=0.25, objref='secondary', dataref='lc01')
        
        You can use this function to plot the mesh of the entire system, or just
        one component (eclipses will show in projected light!). Any scalar
        quantity that is present in the mesh (effective temperature, logg ...)
        can be used as color values. For some of these quantities, there are
        smart defaults for the color maps.
        
        
        **Plotting the current status of the mesh after running computations**
        
        If you've just ran :py:func`Bundle.run_compute` for some observations,
        and you want to see what the mesh looks like after the last calculations
        have been performed, then this is what you need to do::
        
            >>> mybundle.plot_mesh()
            >>> mybundle.plot_mesh(objref='primary')
            >>> mybundle.plot_mesh(select='teff')
            >>> mybundle.plot_mesh(dataref='lc01')
        
        Giving a :envvar:`dataref` and/or setting :envvar:`select='proj'` plots
        the mesh coloured with the projected flux of that dataref.
        
        .. warning::
            
            1. It is strongly advised only to use this function this way when
               only one set of observations has been added: otherwise it is not
               guarenteed that the dataref has been computed for the last time
               point.
            
            2. Although it can sometimes be convenient to use this function
               plainly without arguments or with just the the dataref after
               computations, you need to be careful for the dataref that is used
               to make the plot. The dataref will show up in the logger
               information.
        
        :param objref: object/system label of which you want to plot the mesh.
         The default means the top level system.
        :type objref: str
        :param label: compute label which you want to use to calculate the mesh
         and its properties. The default means the ``default`` set.
        :type label: str
        """
        if dataref is not None:
            # Return just one pbdep, we only need the reference and context
            deps = self._get_by_search(dataref, context='*dep',
                            class_name='ParameterSet', all=True)[0]
            ref = deps['ref']
            context = deps.get_context()
            category = context[:-3]
            kwargs.setdefault('ref', ref)
            kwargs.setdefault('context', context)
        else:
            category = 'lc'
            
        # Set the configuration to the correct time/phase, but only when one
        # (and only one) of them is given.
        if time is not None and phase is not None:
            raise ValueError("You cannot set both time and phase to zero, please choose one")
        elif phase is not None:
            period, t0, shift = self.get_system().get_period()
            time = phase * period + t0
        
        # Observe the system with the right computations
        if time is not None:
            options = self.get_compute(label, create_default=True).copy()
            observatory.observe(self.get_system(), [time], lc=category=='lc',
                                rv=category=='rv', sp=category=='sp',
                                pl=category=='pl', save_result=False, **options)
        
        # Get the object and make an image.
        self.get_object(objref).plot2D(select=select, cmap=cmap, vmin=vmin,
                     vmax=vmax, size=size, dpi=dpi, background=background,
                     savefig=savefig, with_partial_as_half=with_partial_as_half,
                     **kwargs)
        
        
        
    def plot_meshview(self,mplfig=None,mplaxes=None,meshviewoptions=None,lims=None):
        """
        Creates a mesh plot using the saved options if not overridden
        
        [FUTURE]
        
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
        [FUTURE]
        """
        # TODO: fix this so we can set defaults in usersettings
        # (currently can't with search_by = None)
        return self._get_by_section(label,'orbitview')
        
    def plot_orbitview(self,mplfig=None,mplaxes=None,orbitviewoptions=None):
        """
        [FUTURE]
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
        
        [FUTURE]
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
        
        [FUTURE]
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
        
        [FUTURE]
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
        
        [FUTURE]
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        
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
        
        [FUTURE]
        """

        self.purge_signals(self.attached_signals_system) # this will also clear the list
        # get_system_structure is not implemented yet
        
        # these might already be attached?
        self.attach_signal(self,'data_fromfile',self._on_param_changed)
        self.attach_signal(self,'remove_data',self._on_param_changed)
        self.attach_signal(self,'enable_data',self._on_param_changed)
        self.attach_signal(self,'disable_data',self._on_param_changed)
        #~ self.attach_signal(self,'adjust_obs',self._on_param_changed)
        #~ self.attach_signal(self,'restore_version',self._on_param_changed)
        
    def _attach_set_value_signals(self,ps):
        """
        [FUTURE]
        """
        for key in ps.keys():
            param = ps.get_parameter(key)
            self.attach_signal(param,'set_value',self._on_param_changed,ps)
            self.attached_signals_system.append(param)
    
    def _on_param_changed(self,param,ps=None):
        """
        this function is called whenever a signal is emitted that was attached
        in attach_system_signals
        
        [FUTURE]
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
        
        [FUTURE]
        """
        return copy.deepcopy(self)
    
    def save(self, filename):
        """
        Save the bundle into a json-formatted ascii file
        
        @param filename: path to save the bundle
        @type filename: str
        """
        self._save_json(filename)
    
    def save_pickle(self,filename=None,purge_signals=True,save_usersettings=False):
        """
        Save a class to an file.
        Will automatically purge all signals attached through bundle
        
        [FUTURE]
        
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
        
        [FUTURE]
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
        
        [FUTURE]
        """
        atm_types = self.get_parameter('atm', all=True).values()
        ld_coeffs = self.get_parameter('ld_coeffs', all=True).values()
        for atm_type, ld_coeff in zip(atm_types, ld_coeffs):
            ld_coeff.set_value(atm_type)
    
    def set_beaming(self, on=True):
        """
        Include/exclude the boosting effect.
        
        [FUTURE]
        """
        self.set_value('beaming', on, apply_to='all')
        
    def set_ltt(self, on=True):
        """
        Include/exclude light-time travel effects.
        
        [FUTURE]
        """
        self.set_value('ltt', on, apply_to='all')
    
    def set_heating(self, on=True):
        """
        Include/exclude heating effects.
        
        [FUTURE]
        """
        self.set_value('heating', on, apply_to='all')
        
    def set_reflection(self, on=True):
        """
        Include/exclude reflection effects.
        
        [FUTURE]
        """
        self.set_value('refl', on, apply_to='all')
    
    def set_gray_scattering(self, on=True):
        """
        Force gray scattering.
        
        [FUTURE]
        """
        system = self.get_system()
        if on:
            system.add_preprocess('gray_scattering')
        else:
            system.remove_preprocess('gray_scattering')
            
    
def load(filename, load_usersettings=True):
    """
    Load a class from a file.
    
    [FUTURE]
    
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
    
    [FUTURE]
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


def info():
    frames = {}
    for par in parameters.defs.defs:
        for frame in par['frame']:
            if not frame in ['phoebe','wd']: continue
            if frame not in frames:
                if isinstance(par['context'],list):
                    frames[frame]+= par['context']
                else:
                    frames[frame] = [par['context']]
            elif not par['context'] in frames[frame]:
                if isinstance(par['context'],list):
                    frames[frame]+= par['context']
                else:
                    frames[frame].append(par['context'])
    contexts = sorted(list(set(frames['phoebe'])))
    # Remove some experimental stuff
    ignore = 'analytical:binary', 'derived', 'gui', 'logger', 'plotting:axes',\
             'plotting:mesh', 'plotting:orbit', 'plotting:plot',\
             'plotting:selector', 'point_source', 'pssyn', 'server', 'root',\
             'circ_orbit'
    for ign in ignore:
        contexts.remove(ign)
    
    # Sort according to category/physical
    contexts_obs = []
    contexts_syn = []
    contexts_dep = []
    contexts_phy = []
    contexts_cpt = []
    
    for context in contexts:
        if context[-3:] == 'obs':
            contexts_obs.append(context)
        elif context[-3:] == 'syn':
            contexts_syn.append(context)
        elif context[-3:] == 'dep':
            contexts_dep.append(context)
        elif context.split(':')[0] in ['compute', 'fitting', 'mpi']:
            contexts_cpt.append(context)
        else:
            contexts_phy.append(context)
    
    
    def emphasize(text):
        return '\033[1m\033[4m' + text + '\033[m'
    def italicize(text):
        return '\x1B[3m' + text + '\033[m'
    
    summary = ['List of ParameterSets:\n========================\n']
    summary.append("Create a ParameterSet via:\n   >>> myps = phoebe.ParameterSet('<name>')")
    summary.append("Print with:\n   >>> print(myps)")
    summary.append("Info on a parameter:\n   >>> myps.info('<qualifier>')")
    summary.append('\n')
    
    summary.append(emphasize('Physical contexts:'))
    
    # first things with subcontexts
    current_context = None
    for context in contexts_phy:
        if ':' in context:
            this_context = context.split(':')[0]
            if this_context != current_context:
                if current_context is not None:
                    summary[-1] = "\n".join(textwrap.wrap(summary[-1][:-2], initial_indent='',
                                                          subsequent_indent=' '*23,
                                                          width=79))
                current_context = this_context
                summary.append('{:25s}'.format(italicize(this_context))+' --> ')
            summary[-1] += context + ', '
    
    # then without
    summary.append('{:25s}'.format(italicize('other'))+' --> ')
    for context in contexts_phy:
        if not ':' in context:
            summary[-1] += context + ", "
    summary[-1] = "\n".join(textwrap.wrap(summary[-1][:-2], initial_indent='',
                                                          subsequent_indent=' '*23,
                                                          width=79))
    
    # Obs, deps and syns
    summary.append('\n')
    summary.append(emphasize('Observables:'))
    summary.append("\n".join(textwrap.wrap(italicize('dep: ')+", ".join(contexts_dep),
                                                   initial_indent='',
                                                   subsequent_indent=' '*5,
                                                   width=79)))
    summary.append("\n".join(textwrap.wrap(italicize('obs: ')+", ".join(contexts_obs),
                                                   initial_indent='',
                                                   subsequent_indent=' '*5,
                                                   width=79)))
    summary.append("\n".join(textwrap.wrap(italicize('syn: ')+", ".join(contexts_syn),
                                                   initial_indent='',
                                                   subsequent_indent=' '*5,
                                                   width=79)))
    
    # Computables:
    summary.append('\n')
    summary.append(emphasize('Computations and numerics:'))
    summary.append("\n".join(textwrap.wrap(", ".join(contexts_cpt),
                                                   initial_indent='',
                                                   subsequent_indent=' ',
                                                   width=79)))
    
    print("\n".join(summary))
    #frames_contexts = []
    #for frame in sorted(frames.keys()):
        #for context in sorted(frames[frame]):
            #if frame+context in frames_contexts: continue
            #frames_contexts.append(frame+context)
            #parset = parameters.ParameterSet(frame=frame,context=context)
            #if 'label' in parset:
                #parset['label'] = 'mylbl'
            #if 'ref' in parset:
                #parset['ref'] = 'myref'
            #if 'c1label' in parset:
                #parset['c1label'] = 'primlbl'
            #if 'c2label' in parset:
                #parset['c2label'] = 'secnlbl'
            
            #print parset
