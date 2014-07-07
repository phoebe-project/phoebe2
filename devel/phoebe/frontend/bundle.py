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
from phoebe.parameters import feedback as mod_feedback
from phoebe.backend import fitting, observatory, plotting
from phoebe.backend import universe
from phoebe.atmospheres import limbdark
from phoebe.io import parsers
from phoebe.dynamics import keplerorbit
from phoebe.frontend.plotting import Axes, Figure
from phoebe.frontend.usersettings import Settings
from phoebe.frontend.common import Container, rebuild_trunk
#~ from phoebe.frontend.figures import Axes
from phoebe.units import conversions
from phoebe.frontend import phcompleter
from phoebe.frontend import stringreps

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
             
    
    Accessing and changing parameters via twigs:
    
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
        phoebe.frontend.common.Container.set_adjust
        phoebe.frontend.common.Container.set_prior
        phoebe.frontend.common.Container.attach_ps
        phoebe.frontend.common.Container.twigs
        phoebe.frontend.common.Container.search
    
    Adding and handling data:        
        
    .. autosummary::    
        
        Bundle.set_system
        
        Bundle.run_compute
        
        Bundle.lc_fromfile
        Bundle.lc_fromarrays
        Bundle.lc_fromexisting
        Bundle.rv_fromfile
        Bundle.rv_fromarrays
        Bundle.rv_fromexisting
        Bundle.sed_fromfile
        Bundle.sed_fromarrays
        Bundle.sp_fromfile
        Bundle.sp_fromarrays
        Bundle.if_fromfile
        Bundle.if_fromarrays
        
                
        Bundle.disable_lc
        Bundle.disable_rv
    
    Computations and fitting:
    
    .. autosummary::
    
        Bundle.run_compute
        Bundle.run_fitting
    
    High-level plotting functionality:
    
    .. autosummary::
    
        Bundle.plot_obs
        Bundle.plot_syn
        Bundle.plot_residuals
        Bundle.plot_mesh

        
    Convenience functions:
    
    .. autosummary::
        
        Bundle.get_datarefs
        Bundle.get_lc_datarefs
        Bundle.get_rv_datarefs
        Bundle.get_system
        Bundle.get_object
        Bundle.get_orbit
        
    
    
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
        self.sections['fitting'] = []
        self.sections['dataset'] = []
        self.sections['feedback'] = []
        #~ self.sections['version'] = []
        self.sections['figure'] = []
        self.sections['axes'] = []
        self.sections['plot'] = []
        #~ self.sections['meshview'] = [parameters.ParameterSet(context='plotting:mesh')] # only 1
        #~ self.sections['orbitview'] = [parameters.ParameterSet(context='plotting:orbit')] # only 1
        
        self.current_axes = None
        self.current_figure = None
        self.currently_plotted = [] # this will hold (axesref, plotref) pairs of items already plotted on-the-fly
        
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
                            # The following check is added to make old(er)
                            # Bundle files still loadable.
                            if ri['section'] in self.sections:
                                self.sections[ri['section']].append(item_copy)
                            else:
                                logger.error('Unable to load information from section {}'.format(ri['section']))
                
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
        return stringreps.to_str(self)
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
        
        
            
    def get_server(self, label=None):
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
        elif isinstance(system, dict):
            self._from_dict(system)
            file_type = 'json'
        
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
        
        This is wrapper around :py:func:`Body.get_logp <phoebe.backend.universe.Body.get_logp>`
        that deals with enabled/disabled datasets, possibly on the fly. The
        probability computation include the distributions of the priors on the
        parameters.
        
        To get the :math:`\chi^2`, simply do:
        
        .. math::
        
            \chi^2 = -2\log p
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
        if twig is None or twig == '__nolabel__':
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
        
        
    def get_orbit(self, objref, time=None, phase=None, length_unit='Rsol',
                  velocity_unit='km/s', time_unit='d', observer_position=None):
        """
        Retrieve the position, velocity, barycentric and proper time of an object.
        
        This function returns 3 quantities: the position of the object of the
        orbit (by default in solar radii, change via :envvar:`length_unit`) for
        each time point, the velocity of the object (by default in km/s, change
        via :envvar:`velocity_unit`) and the proper time of the object (by
        default in days, change via :envvar:`time_unit`).
        
        The coordinate frame is chosen such that for each tuple (X, Y, Z), X and
        Y are in the plane of the sky, and Z is the radial direction pointing
        away from the observer. Negative velocities are towards the observer.
        Negative Z coordinates are closer to the observer than positive Z
        coordinates. When :envvar:`length_unit` is really length (as opposed to
        angular, see below), then the X, Y and Z coordinates are relative to the
        center of mass of the system.
        
        Length units can also be given in angle (rad, deg, mas), in which case
        the angular coordinates are returned for the X and Y coordinates and the
        Z coordinate is absolute (i.e. including the distance to the object). Be
        sure to set the distance to a realistic value!
        
        **Example usage**::
        
            mybundle = Bundle()
            position, velocity, bary_time, proper_time = mybundle.get_orbit('primary')
        
            # On-sky orbit:
            plt.plot(position[0], position[1], 'k-')
            
            # Radial velocity:
            plt.plot(bary_time, velocity[2], 'r-')
        
        
        
        :param objref: name of the object to retrieve the orbit from
        :type objref: str
        :param time: time array to compute the orbit on. If none is given, the
         time array will cover one orbital period of the outer orbit, with a time
         resolution of at least 100 points per inner orbital period. Time array
         has to be in days, but can be converted in the output.
        :type time: array
        :return: position (solar radii), velocity (km/s), barycentric times (d), proper times (d)
        :rtype: 3-tuple, 3-tuple, array, array
        """
        # Get the Body to compute the orbit of
        body = self.get_object(objref)
        
        # Get a list of all orbits the component belongs to, and if it's the
        # primary or secondary in each of them
        orbits, components = body.get_orbits()
        
        # If no times are given, construct a time array that has the length
        # of the outermost orbit, and a resolution to sample the inner most
        # orbit with at least 100 points
        if time is None:
            period_outer = orbits[-1]['period']
            t0 = orbits[-1]['t0']
            
            if phase is None:
                period_inner = orbits[0]['period']
                t_step = period_inner / 100.
                time = np.arange(t0, t0+period_outer, t_step)
            else:
                time = t0 + phase*period_outer
        
        pos, vel, proper_time = keplerorbit.get_barycentric_hierarchical_orbit(time,
                                                             orbits, components)
        
        # Correct for systemic velocity (we didn't reverse the z-axis yet!)
        globs = self.get_system().get_globals()
        if globs is not None:
            vgamma = globs['vgamma']
            vel[-1] = vel[-1] - conversions.convert('km/s', 'Rsol/d', vgamma)
        
        # Convert to correct units. If positional units are angles rather than
        # length, we need to first convert the true coordinates to spherical
        # coordinates
        if not observer_position and conversions.get_type(length_unit) == 'length':
            pos = [conversions.convert('Rsol', length_unit, i) for i in pos]
            
            # Switch direction of Z coords
            pos[2] = -1*pos[2]
            
        # Angular position wrt solar system barycentre
        elif not observer_position:
            position = body.get_globals(context='position')
            distance = position.get_value_with_unit('distance')
            origin = (position.get_value('ra', 'rad'), position.get_value('dec', 'rad'))
            
            # Switch direction of Z coords
            pos = np.array(pos)
            pos[2] = -1*pos[2]
            
            # Convert true coordinates to spherical ones. Then pos is actually
            # ra/dec
            pos = list(keplerorbit.truecoords_to_spherical(np.array(pos).T, distance=distance,
                                                origin=origin, units=length_unit))
            
            # Take proper motions into account
            pmdec = position.get_value('pmdec', 'rad/d')
            pmra = position.get_value('pmra', 'rad/d')
            pos[1] = pos[1] + pmdec*time
            pos[0] = pos[0] + pmra*time/np.cos(pos[1])
        
        # Angular position wrt observer coordinates
        else:
            raise NotImplementedError
            
        vel = [conversions.convert('Rsol/d', velocity_unit, i) for i in vel]
        
        proper_time = conversions.convert('d', time_unit, proper_time)
        time = conversions.convert('d', time_unit, time)
        
        # Switch direction of radial velocities and Z coords
        vel[2] = -1*vel[2]
        
        return tuple(pos), tuple(vel), time, proper_time
        
        
    def get_orbitps(self, twig=None):
        """
        Retrieve the orbit ParameterSet that belongs to a given BodyBag
        
        [FUTURE]
        
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
                #ds.estimate_sigma(force=False)
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
            
        # In some cases, we can have subcategories of categories. For example
        # "sp" can be given as a timeseries or a snapshort. They are treated
        # the same in the backend, but they need different parse functions
        if len(category.split(':')) > 1:
            category, subcategory = category.split(':')
        else:
            subcategory = None
        
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
                                       ref=dataref, 
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
            if subcategory is None:
                output = datasets.parse_spec_timeseries(filename, columns=columns,
                                       components=objref, units=units,
                                       full_output=True, ref=dataref,
                                       **kwargs)
            # Then this is a shapshot
            else:
                output = datasets.parse_spec_as_lprof(filename, columns=columns,
                                       components=objref, units=units,
                                       full_output=True, ref=dataref,
                                       **kwargs)
        
        elif category == 'sed':
            scale, offset = kwargs.pop('adjust_scale', False), kwargs.pop('adjust_offset', False)
            output = datasets.parse_phot(filename, columns=columns,
                  units=units, group=filename, 
                  group_kwargs=dict(scale=scale, offset=offset),
                  full_output=True, **kwargs)
            
        elif category == 'if':
            if subcategory == 'oifits':
                output = datasets.parse_oifits(filename, full_output=True,
                                               ref=dataref, **kwargs)
        #elif category == 'pl':
        #    output = datasets.parse_plprof(filename, columns=columns,
        #                               components=objref, full_output=True,
        #                               **{'passband':passband, 'ref': ref})
        else:
            output = None
            print("only lc, rv, etv, sed, and sp currently implemented")
            raise NotImplementedError
        
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
        
        # Initialize the mesh after adding stuff (i.e. add columns ld_new_ref...
        self.get_system().init_mesh()
        self._build_trunk()
            
                
        
    
    def lc_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      flux=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None, method=None):
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
        
        or in phase space:
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.lc_fromarrays(phase=phase, passband='GENEVA.V')
        
        With many more details:
        
        >>> bundle.lc_fromarrays(phase=phase, samprate=5, exptime=20.,
        ...     passband='GENEVA.V', atm='kurucz', ld_func='claret', 
        ...     ld_coeffs='kurucz')
        
        .. note:: More information
        
            - For a list of acceptable values for each parameter, see
              :ref:`lcdep <parlabel-phoebe-lcdep>` and
              :ref:`lcobs <parlabel-phoebe-lcobs>`.
            - In general, :envvar:`time`, :envvar:`flux`, :envvar:`phase`,
              :envvar:`sigma`, :envvar:`flag`, :envvar:`weight`, :envvar:`exptime` and
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
                      l3=None, alb=None, beaming=None, scattering=None,
                      method=None):
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
        
        **Example usage**
        
        A plain file can loaded via::
        
        >>> bundle.lc_fromfile('myfile.lc')
        
        Note that extra parameters can be given in the file itself, but can
        also be overriden in the function call:
        
        >>> bundle.lc_fromfile('myfile.lc', atm='kurucz')
        
        If you have a non-standard file (non-default column order or non-default
        units), you have some liberty in specifying the file-format here. You
        can specify the order of the :envvar:`columns` (a list) and the
        :envvar:`units` of each column (dict). You can skip columns by giving an
        empty string ``''``:
        
        >>> bundle.lc_fromfile('myfile.lc', columns=['flux', 'time', 'sigma'])
        >>> bundle.lc_fromfile('myfile.lc', columns=['time', 'mag', 'sigma'])
        >>> bundle.lc_fromfile('myfile.lc', columns=['sigma', 'phase', 'mag'])
        >>> bundle.lc_fromfile('myfile.lc', columns=['', 'phase', 'mag'])
        >>> bundle.lc_fromfile('myfile.lc', units=dict(time='s'))
        >>> bundle.lc_fromfile('myfile.lc', units=dict(time='s', flux='erg/s/cm2/AA'))
        >>> bundle.lc_fromfile('myfile.lc', columns=['time', 'flux'], units=dict(time='s', flux='mag'))
        
        Note that
        
        >>> bundle.lc_fromfile('myfile.lc', columns=['time', 'mag']))
        
        is actually a shortcut to
        
        >>> bundle.lc_fromfile('myfile.lc', columns=['time', 'flux'], units=dict(flux='mag'))
        
        .. note:: More information
        
            - For a list of acceptable values for each parameter, see
              :ref:`lcdep <parlabel-phoebe-lcdep>` and
              :ref:`lcobs <parlabel-phoebe-lcobs>`.
            - For more information on file formats, see
              :py:func:`phoebe.parameters.datasets.parse_lc`.
        
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast
         to the Parameter
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        # We can pass everything now to the main function
        return self.data_fromfile(category='lc', **set_kwargs)
    
    
    def lc_fromexisting(self, to_dataref, from_dataref=None, time=None, phase=None,
                      flux=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None, method=None):
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
                  if set_kwargs[key] is not None and key not in ['self', 'to_dataref']}
        
        self.data_fromexisting(to_dataref, category='lc', **set_kwargs)
        
    def rv_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      rv=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      method=None, atm=None, ld_func=None, ld_coeffs=None,
                      passband=None, pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None):
        """
        Create and attach radial velocity curve templates to compute the model.
        
        In contrast to py:func:`lc_fromarrays <phoebe.frontend.bundle.Bundle.lc_fromarrays`,
        this function will probably
        always be called with a specific :envvar:`objref`. While light curves
        typically encompass the whole system (and are thus added to the whole
        system by default), the radial velocities curves belong to a given
        component. Therefore, make sure to always supply :envvar:`objref`.
        
        An extra keyword is :envvar:`method`, which can take the values
        ``flux-weighted`` (default) or ``dynamical``. In the later case, no
        mesh is computed for the component, but the Kepler-predictions of the
        (hierarhical) orbit are computed. This is much faster, but includes no
        Rossiter-McLaughlin effect.
        
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
        
        >>> bundle.rv_fromarrays('primary', time=np.linspace(0, 10.33, 101))
        
        or in phase space (phase space will probably not work for anything but
        light curves and radial velocities):
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.rv_fromarrays('primary', phase=phase, passband='GENEVA.V')
        
        With many more details:
        
        >>> bundle.rv_fromarrays('primary', phase=phase, samprate=5, exptime=20.,
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
        # we're a little careful when it comes to binaries:
        if hasattr(self.get_system(), 'bodies') and len(self.get_system())>1:
            if objref is None:
                raise ValueError(("In binary or multiple systems, you are "
                                  "required to specify the component to which "
                                  "you want to add rv data (via objref)"))
            if objref == self.get_system().get_label():
                raise ValueError("Cannot add RV to the system, only to the components. Please specify 'objref'.")
        # But other system configuration can have a smart default
        elif objref is None:
            objref = self.get_system().get_label()
        
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
                      l3=None, alb=None, beaming=None, scattering=None,
                      method=None):
        """
        Add a radial velocity curve from a file.
        
        The data contained in :envvar:`filename` will be loaded to the object
        with object reference :envvar:`objref`, and will have data reference
        :envvar:`dataref`. If no :envvar:`dataref` is is given, a unique one is
        generated: the first radial velocity curve that is added is named 'rv01',
        and if that one already exists, 'rv02' is tried and so on.
        
        An extra keyword is :envvar:`method`, which can take the values
        ``flux-weighted`` (default) or ``dynamical``. In the later case, no
        mesh is computed for the component, but the Kepler-predictions of the
        (hierarhical) orbit are computed. This is much faster, but includes no
        Rossiter-McLaughlin effect.
        
        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        **Example usage**
        
        A plain file can loaded via::
        
        >>> bundle.rv_fromfile('myfile.rv', 'primary')
        
        Note that extra parameters can be given in the file itself, but can
        also be overriden in the function call:
        
        >>> bundle.rv_fromfile('myfile.rv', 'primary', atm='kurucz')
        
        If your radial velocity measurements of several components are in one
        file (say time, rv of primary, rv of secondary, sigma of primary rv,
        sigma of secondary rv), you could easily do:
        
        >>> bundle.rv_fromfile('myfile.rv', 'primary', columns=['time', 'rv', '', 'sigma'])
        >>> bundle.rv_fromfile('myfile.rv', 'secondary', columns=['time', '', 'rv', '', 'sigma'])
        
        .. note:: More information
        
            - For a list of acceptable values for each parameter, see
              :ref:`rvdep <parlabel-phoebe-rvdep>` and
              :ref:`rvobs <parlabel-phoebe-rvobs>`.
            - For more information on file formats, see
              :py:func:`phoebe.parameters.datasets.parse_rv`.
        
        
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
        """# we're a little careful when it comes to binaries:
        if hasattr(self.get_system(), 'bodies') and len(self.get_system())>1:
            if objref is None:
                raise ValueError(("In binary or multiple systems, you are "
                                  "required to specify the component to which "
                                  "you want to add rv data (via objref)"))
            if objref == self.get_system().get_label():
                raise ValueError("Cannot add RV to the system, only to the components. Please specify 'objref'.")
            
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
                       scale=None, offset=None, auto_scale=False,
                       auto_offset=False, **kwargs):
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
        if passband is None:
            raise ValueError("Passband is required")
        
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
        
        tools.group(obs, dataref, scale=auto_scale, offset=auto_offset)
        
        return dataref
    
    def sed_fromfile(self, filename, objref=None, dataref=None, columns=None,
                      units=None, offset=None, scale=None, adjust_scale=None,
                      adjust_offset=None):
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
    
    def sp_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      wavelength=None, flux=None, continuum=None, sigma=None,
                      flag=None, R_input=None, offset=None, scale=None,
                      profile=None, R=None, vmicro=None, depth=None, atm=None,
                      ld_func=None, ld_coeffs=None, passband=None, pblum=None,
                      l3=None, alb=None, beaming=None):
        """
        Create and attach spectral templates to compute the model.
        
        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        A unique data reference (:envvar:`dataref`) is added automatically if
        none is provided by the user. A readable system of unique references is
        applied: the first spectrum that is added is named ``sp01`` unless
        that reference already exists, in which case ``sp02`` is tried and so
        on. On the other hand, if a :envvar:`dataref` is given at it already
        exists, it's settings are overwritten.
        
        If no :envvar:`objref` is given, the spectrum is added to the total
        system, and not to a separate component. That is not want you want when
        you're modeling disentangled spectra.
        
        Note that you cannot add :envvar:`times` and :envvar:`phases` simultaneously.
        
        **Example usage**
        
        It doesn't make much sense to leave the time or wavelength array empty,
        since then nothing will be computed. Thus, the minimal function call
        that makes sense is something like:
        
        >>> bundle = phoebe.Bundle()
        >>> bundle.sp_fromarrays(time=np.linspace(0, 10.33, 101),
        ...                      wavelength=np.linspace(399, 401, 500))
        
        or in phase space:
        
        >>> wavelength = np.linspace(399, 401, 500)
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.sp_fromarrays(wavelength=wavelenth, phase=phase, passband='GENEVA.V')
        
        With many more details:
        
        >>> bundle.sp_fromarrays(wavelength=wavelenth, phase=phase, samprate=5,
        ...     exptime=20., passband='GENEVA.V', atm='kurucz',
        ...     ld_func='claret',  ld_coeffs='kurucz')
        
        For a list of acceptable values for each parameter, see
        :ref:`spdep <parlabel-phoebe-spdep>` and
        :ref:`spobs <parlabel-phoebe-spobs>`.
        
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
        return self.data_fromarrays(category='sp', **set_kwargs)
    
    
    def sp_fromfile(self, filename, objref=None, time=None,
                      clambda=None, wrange=None,
                      dataref=None, snapshot=False, columns=None,
                      units=None, offset=None, scale=None, atm=None,
                      R_input=None, vmacro=None, vmicro=None, depth=None,
                      profile=None, alphaT=None,
                      ld_func=None, ld_coeffs=None, passband=None, pblum=None,
                      l3=None, alb=None, beaming=None, scattering=None):
        """
        Add spectral templates from a file.
        
        [FUTURE]
        
        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :param snapshot: filetype of the spectra: is it a timeseries or a snapshot?
        :type snapshot: bool
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast to the Parameter
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key not in ['self','snapshot']}
        
        # Determine whether it is a spectral timeseries of snapshot
        category = 'sp:ss' if snapshot else 'sp'
        
        # We can pass everything now to the main function
        return self.data_fromfile(category=category, **set_kwargs)
        
    
    def if_fromfile(self, filename, objref=None, dataref=None,
                    include_closure_phase=False, include_triple_amplitude=False,
                    include_eff_wave=True, 
                    atm=None, ld_func=None, ld_coeffs=None, passband=None,
                    pblum=None, l3=None, bandwidth_smearing=None,
                    bandwidth_subdiv=None,  alb=None,
                    beaming=None, scattering=None):
        """
        Add interferometry data from an OIFITS file.
        
        The data contained in :envvar:`filename` will be loaded to the object
        with object reference :envvar:`objref` (if ``None``, defaults to the
        whole system), and will have data reference :envvar:`dataref`. If no
        :envvar:`dataref` is is given, a unique one is generated: the first 
        interferometric dataset that is added is named 'if01', and if that one
        already exists, 'if02' is tried and so on.
        
        By default, only the visibility is loaded from the OIFITS file. if you
        want to include closure phases and/or triple amplitudes, you need to
        set :envvar:`include_closure_phase` and/or :envvar:`include_triple_amplitude`
        explicitly.
        
        The effective wavelengths from the OIFITS file are loaded by default,
        but if you want to use the effective wavelength from the passband to
        convert baselines to spatial frequencies, you need to exclude them via
        :envvar:`include_eff_wave=False`. Setting the :envvar:`bandwidth_smearing`
        to `simple` or `detailed` will make the parameter obsolete since the
        spatial frequencies will be computed in a different way.
        
        For any other parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.
        
        **Example usage**
        
        An OIFITS file can loaded via::
        
        >>> bundle.if_fromfile('myfile.fits')
        
        Extra parameters can overriden in the function call:
        
        >>> bundle.if_fromfile('myfile.fits', atm='kurucz')
        
        .. note:: More information
        
            - For a list of acceptable values for each parameter, see
              :ref:`lcdep <parlabel-phoebe-ifdep>` and
              :ref:`lcobs <parlabel-phoebe-ifobs>`.
            - For more information on file formats, see
              :py:func:`phoebe.parameters.datasets.parse_oifits`.
        
        
        :param objref: component to add data to
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast
         to the Parameter
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}
        
        # We can pass everything now to the main function
        return self.data_fromfile(category='if:oifits', **set_kwargs)
    
    def if_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      ucoord=None, vcoord=None, vis2=None, sigma_vis2=None,
                      vphase=None, sigma_vphase=None,
                      eff_wave=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      atm=None, ld_func=None, ld_coeffs=None, passband=None,
                      pblum=None, l3=None, bandwidth_smearing=None,
                      bandwidth_subdiv=None,alb=None, beaming=None,
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
        applied: the first interferometric dataset that is added is named
        ``if01`` unless that reference already exists, in which case ``if02``
        is tried and so on. On the other hand, if a :envvar:`dataref` is given
        at it already exists, it's settings are overwritten.
        
        If no :envvar:`objref` is given, the interferometry is added to the
        total system, and not to a separate component. That is probably fine in
        almost all cases, since you observe the whole system simultaneously and
        not the components separately.
        
        Note that you cannot add :envvar:`times` and :envvar:`phases` simultaneously.
        
        **Example usage**
        
        It doesn't make much sense to leave the time array empty, since then
        nothing will be computed. You are also required to give U and V
        coordinates.
        Thus, the minimal function call that makes
        sense is something like:
        
        >>> bundle = phoebe.Bundle()
        >>> bundle.if_fromarrays(time=np.linspace(0, 10.33, 101))
        
        or in phase space:
        
        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.if_fromarrays(phase=phase, passband='GENEVA.V')
                
        .. note:: More information
        
            - For a list of acceptable values for each parameter, see
              :ref:`lcdep <parlabel-phoebe-ifdep>` and
              :ref:`lcobs <parlabel-phoebe-ifobs>`.
        
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
        return self.data_fromarrays(category='if', **set_kwargs)
    
    
    def if_fromexisting(self, to_dataref, from_dataref=None,
                    remove_closure_phase=False, remove_triple_amplitude=False,
                    remove_eff_wave=False, time=None, phase=None, ucoord=None,
                    vcoord=None, vis2=None, sigma_vis2=None, eff_wave=None,
                    atm=None, ld_func=None, ld_coeffs=None, passband=None,
                    pblum=None, l3=None, bandwidth_smearing=None,
                    bandwidth_subdiv=None,  alb=None,
                    beaming=None, scattering=None):
        """
        Duplicate an existing interferometry set to a new one with a different dataref.
                
        See :py:func:`Bundle.lc_fromexisting` for more info.
        
        Additionally, you can easiyl remove closure phases, triple amplitudes
        and/or effective wavelength, if you ever so wish.
        
        :param objref: component to add data to
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :return: dataref of added observations (perhaps it was autogenerated!)
        :rtype: str
        :raises TypeError: if a keyword is given but the value cannot be cast
         to the Parameter
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()
        
        # filter the arguments according to not being "None" nor being "self"
        ignore = ['self', 'to_dataref', 'remove_closure_phase',
                  'remove_triple_amplitude', 'remove_eff_wave']
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key not in ignore}
        
        # We can pass everything now to the main function
        out = self.data_fromexisting(to_dataref, category='if', **set_kwargs)
        
        # Remove closure phases, triple amplitudes or effective wavelengths
        # if necessary
        obs = self.get_obs(to_dataref)
        if remove_closure_phase:
            obs.remove('closure_phase')
        if remove_triple_amplitude:
            obs.remove('triple_ampl')
        if remove_eff_wave:
            obs.remove('eff_wave')
        
        return out
    
    
    def add_parameter(self, twig, replaces=None, value=None):
        """
        Add a new parameter to the set of parameters as a combination of others.
        
        The value of the new parameter can either be derived from the existing
        ones, or replaces one of the existing ones. It is thus not possible to
        simply add an extra parameter; i.e. the number of total free parameters
        must stay the same.
        
        Explanation of the two scenarios:
        
        **1. Adding a new parameter without replacing an existing one**
        
        For example, you want to add ``asini`` as a parameter but want to keep
        ``sma`` and ``incl`` as free parameters in the fit:
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.add_parameter('asini@orbit')
        >>> mybundle['asini']
        9.848077530129958
        
        Then, you can still change ``sma`` and ``incl``:
        
        >>> mybundle['sma'] = 20.0
        >>> mybundle['incl'] = 85.0, 'deg'
        
        and then ``asini`` is updated automatically:
        
        >>> x['asini']
        19.923893961843316
        
        However you are not allowed to change the parameter of ``asini``
        manually, because the code does not know if it needs to update ``incl``
        or ``sma`` to keep consistency:
        
        >>> x['asini'] = 10.0
        ValueError: Cannot change value of parameter 'asini', it is derived from other parameters

        **2. Adding a new parameter to replace an existing one**
        
        >>> mybundle = phoebe.Bundle()
        >>> mybundle.add_parameter('asini@orbit', replaces='sma')
        >>> mybundle['asini']
        9.848077530129958
        
        Then, you can change ``asini`` and ``incl``:
        
        >>> mybundle['asini'] = 10.0
        >>> mybundle['incl'] = 85.0, 'deg'
        
        and ``sma`` is updated automatically:
        
        >>> x['sma']
        10.038198375429241
        
        However you are not allowed to change the parameter of ``sma``
        manually, because the code does not know if it needs to update ``asini``
        or ``incl`` to keep consistency:
        
        >>> x['sma'] = 10.0
        ValueError: Cannot change value of parameter 'sma', it is derived from other parameters

        **Non-exhaustive list of possible additions**
        
        - ``asini@orbit``: projected system semi-major axis (:py:func:`more info <phoebe.parameters.tools.add_asini>`)
        - ``ecosw@orbit``: eccentricity times cosine of argument of periastron,
          automatically adds also ``esinw`` (:py:func:`more info <phoebe.parameters.tools.add_esinw_ecosw>`)
        - ``theta_eff@orbit``: Effective misalignment parameter (:py:func:`more info <phoebe.parameters.tools.add_theta_eff>`)
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
        
        # Special cases:
        if qualifier in ['ecosw', 'esinw']:
            qualifier = 'esinw_ecosw'
        
        # If the parameter does not exist yet, there's some work to do: we need
        # to figure out where to add it, and we need to create it
        
        # There are two ways we can add it: the easy way is if the parameter
        # can be added via the "tools" module
        if hasattr(tools, 'add_{}'.format(qualifier)):
            
            # Retrieve the parameterSet to add it to
            twig_rest = '@'.join(twig_split[1:])
            item = self._get_by_search(twig_rest, kind='ParameterSet',
                                       return_trunk_item=True)
            
            # If the 'replaces' is a twig, make sure it's a valid one
            if replaces is not None:
                replaces_param = self.get_parameter(replaces)
                
                if replaces_param.get_context() != item['item'].get_context():
                    raise ValueError("The parameter {} cannot replace {} because it is not in the same ParameterSet".format(twig, replaces))
                    
                # Set the replaced parameter to be hidden or replaced
                #replaces_param.set_hidden(True)
                replaces_param.set_replaced_by(None)
                replaces_qualifier = replaces_param.get_qualifier()
            else:
                replaces_qualifier = None
                    
            # get the function that is responsible for adding this parameter
            add_function = getattr(tools, 'add_{}'.format(qualifier))
            argspecs = inspect.getargspec(add_function)[0]
            # build the args for this function (only add the value and/or derive
            # argument if that is allowed by the function)
            add_args = [item['item']]
            if qualifier in argspecs:
                add_args += [value]
            elif value is not None:
                raise ValueError("The value of parameter '{}' can not be set explicitly, it can only be derived".format(qualifier))
            if 'derive' in argspecs:
                add_args += [replaces_qualifier]
            elif replaces_qualifier is not None:
                raise ValueError("Parameter '{}' can only be derived itself, it cannot be used to derive '{}'".format(qualifier, replaces_qualifier))
                    
            params = add_function(*add_args)
            
            if replaces is None:
                for param in params:
                    param.set_replaced_by(True)
            else:
                for param in params:
                    param.set_replaced_by(None)
                replaces_param.set_replaced_by(params)
            
            logger.info("Added {} to ParameterSet with context {}".format(qualifier, item['item'].get_context()))
            
            self._build_trunk()
            
            
            return None
        
        elif param is None:
            
            # If this parameter does not replaces any other, it is derived itself
            if replaces is None:
                replaces = qualifier    
            
            # Get all the info on this parameter
            info = definitions.rels['binary'][qualifier].copy()
            
            # Figure out which level (i.e. ParameterSet) to add it to
            in_level_as = info.pop('in_level_as')
            
            # If we already have a level here, it's easy-peasy
            if in_level_as[:2] != '__':
                twig_rest = '@'.join([in_level_as] + twig_split[1:])
                item = self._get_by_search(twig_rest, kind='Parameter',
                                         return_trunk_item=True)
            elif in_level_as == '__system__':
                system = self.get_system()
                system.attach_ps()
            
            # And add it
            pset = item['path'][-2]
            pset.add(info)
            
            param = pset.get_parameter(qualifier)
            #param.set_replaces(replaces)
            
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
        try:
            dataref = self._process_dataref(dataref, category)
        except:
            dataref = None
        
        
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
    
    def enable_sp(self, dataref=None):
        """
        Enable an SP dataset so that it will be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'sp', True)
        
    def disable_sp(self, dataref=None):
        """
        Disable an SP dataset so that it will not be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'sp', False)
        
    def enable_sed(self, dataref=None):
        """
        Enable LC datasets belonging to an sed so that it will be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        system = self.get_system()
        all_lc_refs = system.get_refs(category='lc')
        all_lc_refs = [ref for ref in all_lc_refs if dataref in ref]
        for dataref in all_lc_refs:
            self.enable_data(dataref, 'lc', True)
        
    def disable_sed(self, dataref=None):
        """
        Enable LC datasets belonging to an sed so that it will not be considered during run_compute
        
        See :py:func:`Bundle.enable_lc` for more info
        
        @param dataref: reference of the dataset
        @type dataref: str
        """
        system = self.get_system()
        all_lc_refs = system.get_refs(category='lc')
        all_lc_refs = [ref for ref in all_lc_refs if dataref in ref]
        for dataref in all_lc_refs:
            self.enable_data(dataref, 'lc', False)
        
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
                # Next line doesn't seem to work, so I short-cutted a return value
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
        system.reset_and_clear()
        #system.clear_synthetic()
        
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
            try:
                options.set_value(k,v)
            except AttributeError:
                raise ValueError("run_compute does not accept keyword '{}'".format(k))
        
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
    def run_fitting(self, fittinglabel='lmfit', computelabel=None,
                    add_feedback=True, accept_feedback=True, server=None,
                    mpi=None, **kwargs):
        """
        Run fitting for a given fitting ParameterSet and store the feedback
        
        [FUTURE]
        
        @param computelabel: name of compute ParameterSet
        @param computelabel: str
        @param fittinglabel: name of fitting ParameterSet
        @type fittinglabel: str
        @param add_feedback: flag to store the feedback (retrieve with get_feedback)
        @type add_feedback: bool
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
            mpi = mpi
        
        # get fitting params
        fittingoptions = self.get_fitting(fittinglabel).copy()
        
        # get compute params
        if computelabel is None:
            computelabel = fittingoptions['computelabel']
        
        # Make sure that the fittingoptions refer to the correct computelabel
        computeoptions = self.get_compute(computelabel).copy()
        fittingoptions['computelabel'] = computelabel
            
        # now temporarily override with any values passed through kwargs    
        for k,v in kwargs.items():
            if k in fittingoptions.keys():
                fittingoptions.set_value(k,v)
            elif k in computeoptions.keys():
                computeoptions.set_value(k,v)
        
        # Check if everything is OK (parameters are inside limits and/or priors)
        passed, errors = self.check(return_errors=True)
        if not passed:
            raise ValueError(("Some parameters are outside of reasonable limits or "
                              "prior bounds: {}").format(", ".join(errors)))
        
        # Remember the initial values of the adjustable parameters, we'll reset
        # them later:
        init_values = [par.get_value() for par in self.get_system().get_adjustable_parameters()]
        
        # here, we should disable those obs that have no flux/rv/etc.., i.e.
        # the ones that were added just for exploration purposes. We should
        # keep track of those and then re-enstate them to their original
        # value afterwards (the user could also have disabled a dataset)
        # <some code>
        logger.warning("Fit options:\n{:s}".format(fittingoptions))
        logger.warning("Compute options:\n{:s}".format(computeoptions))
        
        # Run the fitting for real
        feedback = fitting.run(self.get_system(), params=computeoptions,
                               fitparams=fittingoptions, mpi=mpi)
        
        # Reset the parameters to their initial values
        for par, val in zip(self.get_system().get_adjustable_parameters(), init_values):
            par.set_value(val)
        
        if add_feedback:
            # Create a Feedback class and add it to the feedback section with
            # the same label as the fittingoptions
            subcontext = fittingoptions.get_context().split(':')[1]
            class_name = 'Feedback' + subcontext.title()
            feedback = getattr(mod_feedback, class_name)(*feedback, init=self,
                                 fitting=fittingoptions, compute=computeoptions)
            
            # Make sure not to duplicate entries
            existing_fb = [fb.get_label() for fb in self.sections['feedback']]
            if feedback.get_label() in existing_fb:
                self.sections['feedback'][existing_fb.index(feedback.get_label())] = feedback
            else:
                self._add_to_section('feedback', feedback)
            logger.info(("You can access the feedback from the fitting '{}' ) "
                         "with the twig '{}@feedback'".format(fittinglabel, fittinglabel)))
        
        # Then re-instate the status of the obs without flux/rv/etc..
        # <some code>
        
        # Accept the feedback: set/reset the variables to their fitted values
        # or their initial values, and in any case recompute the system such
        # that the synthetics are up-to-date with the parameters
        self.accept_feedback(fittingoptions['label']+'@feedback',
                             recompute=True, revert=(not accept_feedback))
            
        return feedback
    
    def feedback_fromfile(self, feedback_file, fittinglabel, accept_feedback=True):
        """
        Add fitting feedback from a file.
        
        [FUTURE]
        """
        fittingoptions = self.get_fitting(fittinglabel).copy()
        computeoptions = self.get_compute(fittingoptions['computelabel']).copy()
        
        # Remember the initial values of the adjustable parameters, we'll reset
        # them later:
        init_values = [par.get_value() for par in self.get_system().get_adjustable_parameters()]
        
        # Create a Feedback class and add it to the feedback section with
        # the same label as the fittingoptions
        subcontext = fittingoptions.get_context().split(':')[1]
        class_name = 'Feedback' + subcontext.title()
        feedback = getattr(mod_feedback, class_name)(feedback_file, init=self,
                             fitting=fittingoptions, compute=computeoptions)
        
        # Make sure not to duplicate entries
        existing_fb = [fb.get_label() for fb in self.sections['feedback']]
        if feedback.get_label() in existing_fb:
            self.sections['feedback'][existing_fb.index(feedback.get_label())] = feedback
        else:
            self._add_to_section('feedback', feedback)
        logger.info(("You can access the feedback from the fitting '{}' ) "
                     "with the twig '{}@feedback'".format(fittinglabel, fittinglabel)))
        
        # Accept the feedback: set/reset the variables to their fitted values
        # or their initial values, and in any case recompute the system such
        # that the synthetics are up-to-date with the parameters
        self.accept_feedback(fittingoptions['label']+'@feedback',
                             recompute=True, revert=(not accept_feedback))
        
        return feedback
        
    
    def accept_feedback(self, twig, revert=False, recompute=True):
        """
        Change parameters with those resulting from fitting.
        
        [FUTURE]
        """
        # Retrieve the correct feedback
        feedback = self._get_by_search(twig, kind='Feedback')
        feedback.apply_to(self.get_system(), revert=revert)
        
        # If we need to recompute, recompute with the specified label
        if recompute:
            computelabel = feedback.get_computelabel()
            self.run_compute(label=computelabel)
            self._build_trunk()
        
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
    def show(self, twig=None, attach=None, **kwargs):
        """
        [FUTURE]
        
        Draw and show a figure, axes, or plot that is attached to the bundle.
        
        If :envvar:`twig` is None, then the current figure will be plotted.
        All kwargs are passed to :py:func`Bundle.draw`
        
        :param twig: twig pointing the the figure, axes, or plot options
        :type twig: str or None
        :param attach: whether to attach these options to the bundle
        :type attach: bool
        """
        ax_or_fig = self.draw(twig, attach=attach, **kwargs)
        plt.show()
        
    def savefig(self, twig=None, filename=None, attach=None, **kwargs):
        """
        [FUTURE]

        Draw and save a figure, axes, or plot that is attached to the bundle
        to file.
        
        If :envvar:`twig` is None, then the current figure will be plotted.
        All kwargs are passed to :py:func`Bundle.draw`
        
        :param twig: twig pointing the the figure, axes, or plot options
        :type twig: str or None
        :param attach: whether to attach these options to the bundle
        :type attach: bool
        :param filename: filename for the resulting image
        :type filename: str
        """
        ax_or_fig = self.draw(twig, attach=attach **kwargs)
        plt.savefig(filename)
        
    def draw(self, twig=None, attach=None, **kwargs):
        """
        [FUTURE]
        
        Draw a figure, axes, or plot that is attached to the bundle.
        
        If :envvar:`twig` is None, then the current figure will be plotted.
        
        If you do not provide a value for :envvar:`attach`, it will default to True
        if you provide a value for :envvar:`twig`, or False if you do not.  In this 
        way, simply calling draw() with no arguments will not retain a copy in the 
        bundle - you must either explicitly provide a twig or set attach to True.
        
        If you don't provide :envvar:`fig` or :envvar:`ax`, then the the 
        figure/axes will be drawn to plt.gcf() or plt.gca()
        
        If you provide :envvar:`fig` but :envvar:`twig` points to a plot or 
        axes instead of a figure, then the axes will be drawn to fig.gca()
        
        :param twig: twig pointing the the figure, axes, or plot options
        :type twig: str or None
        :param attach: whether to attach these options to the bundle
        :type attach: bool
        :param ax: mpl axes used for plotting (optional, overrides :envvar:`fig`)
        :type ax: mpl.Axes
        :param fig: mpl figure used for plotting (optional)
        :type fig: mpl.Figure
        :param clf: whether to call plt.clf() before plotting
        :type clf: bool
        :param cla: whether to call plt.cal() before plotting
        :type cla: bool
        """
        if attach is None:
            attach = twig is not None
        
        if twig is None:
            ps = self.get_figure()
        else:
            ps = self._get_by_search(twig, section=['plot','axes','figure'])
        
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        clf = kwargs.pop('clf', False)
        cla = kwargs.pop('cla', False)
        
        ref = ps.get_value('ref')
        level = ps.context.split(':')[1].split('_')[0] # will be plot, axes, or figure
        
        if level=='figure':
            if fig is None:
                fig = plt.gcf()
            if clf:
                fig.clf()
            
            ret = self.draw_figure(ref, fig=fig, **kwargs)
            
            if not attach:
                self.remove_figure(ref if ref is not None else self.current_figure)
        
        elif level=='axes':
            if ax is None:
                if fig is None:
                    ax = plt.gca()
                else:
                    ax = fig.gca()
            if cla:
                ax.cla()
        
            ret = self.draw_axes(ref, ax=ax, **kwargs)
            
            if not attach:
                self.remove_axes(ref if ref is not None else self.current_axes)
            
        else:
            if ax is None:
                if fig is None:
                    ax = plt.gca()
                else:
                    ax = fig.gca()
            if cla:
                ax.cla()
            
            ret = self.draw_plot(ref, ax=ax, **kwargs)
            
            if not attach:
                self.remove_plot(ref)
            
        
        return ret

    def get_figure(self, figref=None):
        """
        [FUTURE]
        
        Get the ParameterSet object for a figure attached to the bundle
        
        If :envvar:`figref` is None, then the current figure will be retrieved 
        if one exists.  If one does not exist, an empty will first be created
        and attached to the bundle.
        
        :param figref: ref to the figure
        :type figref: str or None
        :return: the figure
        :rtype: frontend.plotting.Figure (ParameterSet)
        """
        # TODO: allow passing kwargs
        
        if figref is None:
            figref = self._handle_current_figure()
        else:
            figref = figref.split('@')[0]
            
        self.current_figure = figref
        
        fig_ps = self._get_by_section(figref, "figure", kind=None)
        
        if self.current_axes is None and len(fig_ps['axesrefs']):
            self.current_axes = fig_ps['axesrefs'][-1]
        
        return fig_ps
    
    def add_figure(self, figref=None, **kwargs):
        """
        [FUTURE]
        
        Add a new figure to the bundle, and set it as the current figure 
        for any subsequent plotting calls
        
        If :envvar:`figref` is None, a default will be created
        
        Any kwargs will get passed to the plotting:figure ParameterSet.
        
        :param figref: ref for the figure
        :type figref: str or None
        :return: the figref
        :rtype: str
        """
        
        if figref is None:
            figref = "fig{:02}".format(len(self.sections['figure'])+1)
        fig = Figure(self, ref=figref, **kwargs)

        for k,v in kwargs.items():
            fig.set_value(k,v)
            
        if figref not in self._get_dict_of_section('figure').keys():
            self._add_to_section('figure',fig) #calls rebuild trunk
        
        self.current_figure = figref
        self.current_axes = None
        
        return self.current_figure
    
    @rebuild_trunk
    def remove_figure(self, figref=None):
        """
        [FUTURE]
        
        Remove a figure and all of its children axes (and their children plots)
        that are not referenced elsewhere.
        
        :param figref: ref of the figure
        :type figref: str
        """
        if figref is None:
            figref = self.current_figure
            
        fig_ps = self._get_by_section(figref, "figure", kind=None)
        axesrefs = fig_ps['axesrefs']
        self.sections['figure'].remove(fig_ps)
        
        for axesref in axesrefs:
            if all([axesref not in ps['axesrefs'] for ps in self.sections['figure']]):
                #~ print "*** remove_figure: removing axes", axesref
                self.remove_axes(axesref)
            #~ else:
                #~ print "*** remove_figure: keeping axes", axesref
        
    def _handle_current_figure(self, figref=None):
        """
        [FUTURE]
        
        similar to _handle_current_axes - except perhaps we shouldn't be creating figures by default,
        but only creating up to the axes level unless the user specifically requests a figure to be made?
        """
        if not self.current_figure or (figref is not None and figref not in self._get_dict_of_section('figure').keys()):
            figref = self.add_figure(figref=figref)
        if figref is not None:
            self.current_figure = figref
            
        return self.current_figure
        
    def draw_figure(self, figref=None, fig=None, **kwargs):
        """
        [FUTURE]
        
        Draw a figure that is attached to the bundle.
        
        If :envvar:`figref` is None, then the current figure will be plotted.
        
        If you don't provide :envvar:`fig` then the figure will be drawn to plt.gcf().
        
        :param figref: ref of the figure
        :type figref: str or None
        :param fig: mpl figure used for plotting (optional)
        :type fig: mpl.Figure
        """
        # TODO move clf option from self.draw to here

        if fig is None:
            fig = plt.gcf()
        
        live_plot = kwargs.pop('live_plot', False)
            
        fig_ps = self.get_figure(figref)
        
        axes_ret, plot_ret = {}, {}
        for (axesref,axesloc,axessharex,axessharey) in zip(fig_ps['axesrefs'], fig_ps['axeslocs'], fig_ps['axessharex'], fig_ps['axessharey']):
            # need to handle logic for live-plotting
            axes_ps = self.get_axes(axesref)
            plotrefs = axes_ps['plotrefs']
            # the following line will check (if we're live-plotting) to 
            # see if all the plots in this axes have already been drawn,
            # and if they have, then we have no need to make a new subplot
            # or call the plotting function
            if not (live_plot and all([(axesref.split('@')[0], plotref.split('@')[0]) in self.currently_plotted for plotref in plotrefs])):
                # the following line will default to the current mpl ax object if we've already
                # created this subplot - TODO this might cause issues if switching back to a previous axesref
                if live_plot and axesref.split('@')[0] in [p[0] for p in self.currently_plotted]:
                    #~ print "*** plot_figure resetting ax to None"
                    ax = None # will default to plt.gca() - this could still cause some odd things
                    xaxis_loc, yaxis_loc = None, None
                else:
                    axes_existing_refs = [a.split('@')[0] for a in axes_ret.keys()]
                    #~ print "***", axesref, axes_ps.get_value('sharex'), axes_ps.get_value('sharey'), axes_existing_refs
                    
                    # handle axes sharing
                    # we also need to check to see if we're in the same location, and if so
                    # smartly move the location of the axesticks and labels
                    axessharex = axessharex.split('@')[0]
                    axessharey = axessharey.split('@')[0]
                    if axessharex != '_auto_' and axessharex in axes_existing_refs:
                        ind = axes_existing_refs.index(axessharex)
                        sharex = axes_ret.values()[ind]
                        if axesloc in [fig_ps.get_loc(axes_ret.keys()[ind])]: 
                            # then we're drawing on the same location, so need to move the axes
                            yaxis_loc = 'right'
                    else:
                        sharex = None
                        yaxis_loc = None # just ignore, defaults will take over
                    if axessharey != '_auto_' and axessharey in axes_existing_refs:
                        ind = axes_existing_refs.index(axessharey)
                        sharey = axes_ret.values()[ind]
                        #~ print "*** draw_figure move xaxis_loc?", axesloc, fig_ps.get_loc(axes_ret.keys()[ind])
                        if axesloc in [fig_ps.get_loc(axes_ret.keys()[ind])]: 
                            # then we're drawing on the same location, so need to move the axes
                            xaxis_loc = 'top'
                    else:
                        sharey = None
                        xaxis_loc = None # just ignore, defaults will take over
                        
                    #~ print "***", axesref, sharex, sharey
                        
                    ax = fig.add_subplot(axesloc[0],axesloc[1],axesloc[2],sharex=sharex,sharey=sharey)
                    # we must set the current axes so that subsequent live-plotting calls will
                    # go here unless overriden by the user
                    plt.sca(ax)

                #~ print "*** draw_figure calling draw_axes", figref, axesref, axesloc
                axes_ret_new, plot_ret_new = self.draw_axes(axesref, ax=ax, live_plot=live_plot)
                
                #~ print "*** draw_figure", xaxis_loc, yaxis_loc
                
                if xaxis_loc == 'top':
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")
                    #~ ax.xaxis.set_offset_position('top')
                    ax.yaxis.set_visible(False)
                    ax.patch.set_visible(False)
                if yaxis_loc == 'right':
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    #~ ax.yaxis.set_offset_position('right')
                    ax.xaxis.set_visible(False)
                    ax.patch.set_visible(False)
                
                for k,v in axes_ret_new.items():
                    axes_ret[k] = v
                for k,v in plot_ret_new.items():
                    plot_ret[k] = v
            #~ else:
                #~ print "*** plot_figure skipping axesref {} entirely".format(axesref)
            
        if not live_plot:
            self.current_figure = None
            self.current_axes = None
            self.currently_plotted = []
            
        return ({figref: fig}, axes_ret, plot_ret)
        
    
    #}
    
    #{ Axes
    def get_axes(self, axesref=None):
        """
        [FUTURE]
        
        Get the ParameterSet object for an axes attached to the bundle
        
        If :envvar:`axesref` is None, then the current axes will be retrieved 
        if one exists.  If one does not exist, an empty will first be created
        and attached to the bundle as well as the current figure.
        
        :param axesref: ref to the axes
        :type axesref: str or None
        :return: the axes
        :rtype: frontend.plotting.Axes (ParameterSet)
        """
        # TODO allow passing kwargs
        
        if axesref is None:
            axesref, figref = self._handle_current_axes()
        else:
            axesref = axesref.split('@')[0]
        self.current_axes = axesref
        return self._get_by_section(axesref, "axes", kind=None)
        
    def add_axes(self, axesref=None, figref=None, loc=(1,1,1), sharex='_auto_', sharey='_auto_', **kwargs):
        """
        [FUTURE]
        
        Add a new axes to the bundle, and set it as the current axes
        for any subsequent plotting calls.
        
        If :envvar:`axesref` is None, a default will be created.
        
        If :envvar:`figref` is None, the axes will be attached to the current figure.
        If :envvar:`figref` points to an existing figure, the axes will be attached
        to that figure and it will become the current figure.
        If :envvar:`figure` does not point to an existing figure, the figure will
        be created, attached, and will become the current figure.
        
        Any kwargs will get passed to the plotting:axes ParameterSet.
        
        :param axesref: ref for the axes
        :type axesref: str
        :param figref: ref to a parent figure
        :type figref: str or None
        :param loc: location in the figure to attach the axes (see matplotlib.figure.Figure.add_subplot)
        :type loc: tuple with length 3
        :return: (axesref, figref)
        :rtype: tuple of strings
        """
        #~ axesref = kwargs.pop('axesref', None)
        #~ figref = kwargs.pop('figref', None)
        add_to_section = True
        if axesref is None or axesref not in self._get_dict_of_section('axes').keys():
            if axesref is None:
                axesref = "axes{:02}".format(len(self.sections['axes'])+1)
            axes_ps = Axes(self, ref=axesref, **kwargs)
        else:
            add_to_section = False
            axes_ps = self.get_axes(axesref)

        for k,v in kwargs.items():
            axes_ps.set_value(k,v)

        if add_to_section:
            self._add_to_section('axes',axes_ps)
        
        # now attach it to a figure
        figref = self._handle_current_figure(figref=figref)
        if loc in [None, '_auto_']: # just in case (this is necessary and used from defaults in other functions)
            loc = (1,1,1)
        self.get_figure(figref).add_axes(axesref, loc, sharex, sharey) # calls rebuild trunk and will also set self.current_figure

        self.current_axes = axesref
        
        return self.current_axes, self.current_figure

    @rebuild_trunk
    def remove_axes(self, axesref=None):
        """
        [FUTURE]
        
        Remove an axes and all of its children plots that are not referenced
        by other axes.
        
        :param axesref: ref of the axes
        :type axesref: str
        """
        axes_ps = self._get_by_section(axesref, "axes", kind=None)
        plotrefs = axes_ps['plotrefs']

        # we also need to check all figures, and remove this entry from any axesrefs
        # as well as any sharex or sharey
        for figref,fig_ps in self._get_dict_of_section('figure').items():
            fig_ps.remove_axes(axesref)
        
        self.sections['axes'].remove(axes_ps)

        for plotref in axes_ps['plotrefs']:
            if all([plotref not in ps['plotrefs'] for ps in self.sections['axes']]):
                self.remove_plot(plotref)            
        
    def draw_axes(self, axesref=None, ax=None, **kwargs):
        """
        [FUTURE]
        
        Draw an axes that is attached to the bundle.
        
        If :envvar:`axesref` is None, then the current axes will be plotted.
        
        If you don't provide :envvar:`ax` then the figure will be drawn to plt.gca().
        
        :param axesref: ref of the axes
        :type axesref: str or None
        :param ax: mpl axes used for plotting (optional, overrides :envvar:`fig`)
        :type ax: mpl.Axes
        """
        # TODO move cla here from self.draw

        axes_ps = self.get_axes(axesref) #also sets to current
        axesref = self.current_axes
        
        if ax is None:
            ax = plt.gca()
            
        live_plot = kwargs.pop('live_plot', False)
 
        # right now all axes options are being set for /each/ draw_plot
        # call, which will work, but is overkill.  The reason for this is
        # so that draw_plot(plot_label) is smart enough to handle also setting
        # the axes options for the one single call
        
        plot_ret = {}
        for plotref in axes_ps['plotrefs']:
            # handle logic for live-plotting - don't duplicate something that has already been called
            #~ print "*** draw_axes", live_plot, (axesref.split('@')[0], plotref.split('@')[0]), self.currently_plotted
            if not (live_plot and (axesref.split('@')[0], plotref.split('@')[0]) in self.currently_plotted):
                #~ print "*** draw_axes CALLING draw_plot for", axesref, plotref
                plot_ret_new = self.draw_plot(plotref, axesref=axesref, ax=ax, live_plot=live_plot)
            
                for k,v in plot_ret_new.items():
                    plot_ret[k] = v
            #~ else:
                #~ print "*** draw_axes skipping draw_plot for", axesref, plotref
            
        if not live_plot:
            self.current_axes = None
            #~ self.current_figure = None
            self.currently_plotted = []
            
        return ({axesref: ax}, plot_ret)
            
    def _handle_current_axes(self, axesref=None, figref=None, loc=None, sharex='_auto_', sharey='_auto_', x_unit='_auto_', y_unit='_auto_'):
        """
        [FUTURE]
        this function should be called whenever the user calls a plotting function
        
        it will check to see if the current axes is defined, and if it isn't it will
        create a new axes instance with an intelligent default label
        
        the created plotting parameter set should then be added later by self.get_axes(label).add_plot(plot_twig)
        """
        
        #~ print "***", self.current_axes, axesref, figref, loc, sharex, sharey
        
        fig_ps = self.get_figure(figref)
        
        #~ figaxesrefs = fig_ps.get_value('axesrefs')
        #~ figaxesrefs_ref = [ar.split('@')[0] for ar in figaxesrefs]
        #~ figaxeslocs = fig_ps.get_value('axeslocs')
        
        if self.current_axes is not None:
            curr_axes_ps = self.get_axes(self.current_axes)
            curr_axes_loc = fig_ps.get_loc(self.current_axes)
        else:
            curr_axes_ps = None
            curr_axes_loc = None

        #~ print "*** _handle_current_axes", axesref, self.current_axes, loc, curr_axes_loc, loc in [None, curr_axes_loc]

        if not self.current_axes \
                or (axesref is not None and axesref not in self._get_dict_of_section('axes')) \
                or (loc is not None and loc not in fig_ps.get_value('axeslocs')):
            # no existing figure
            # or new axesref for this figure
            # or new loc for this figure (note: cannot retrieve by loc)
            #~ print "*** _handle_current_axes calling add_axes", axesref, loc, sharex, sharey
            axesref, figref = self.add_axes(axesref=axesref, figref=figref, loc=loc, sharex=sharex, sharey=sharey) # other stuff can be handled externally (eg by _handle_plotting_call)

        # handle units
        elif axesref is None and loc in [None, curr_axes_loc]:
            # we do not allow axesref==self.current_axes in this case, because then we'd be overriding the axesref
            
            # TODO: we need to be smarter here if any of the units are _auto_ - they may be referring to
            # different datasets with different default units, so we need to check to see what _auto_
            # actually refers to and see if that matches... :/
            add_axes = True
            if x_unit != curr_axes_ps.get_value('x_unit') and y_unit != curr_axes_ps.get_value('y_unit'):
                #~ print "*** _handle_current_axes units caseA"
                sharex = '_auto_'
                sharey = '_auto_'
                loc = curr_axes_loc
                # axes locations (moving to top & right) is handled in draw_figure
            elif x_unit != curr_axes_ps.get_value('x_unit'):
                #~ print "*** _handle_current_axes units caseB"
                sharex = '_auto_'
                sharey = self.current_axes
                loc = curr_axes_loc
                # axes locations (moving to top) is handled in draw_figure
            elif y_unit != curr_axes_ps.get_value('y_unit'):
                #~ print "*** _handle_current_axes units caseC"
                sharex = self.current_axes
                sharey = '_auto_'
                loc = curr_axes_loc
                # axes locations (moving to right) is handled in draw_figure
            else:
                #~ print "*** _handle_current_axes units caseD"
                #~ sharex = '_auto_'
                #~ sharey = '_auto_'
                #~ loc = loc
                add_axes = False
                
            if add_axes:
                #~ print "*** _handle_current_axes calling add_axes", axesref, loc, sharex, sharey
                axesref, figref = self.add_axes(axesref=axesref, figref=figref, loc=loc, x_unit=x_unit, y_unit=y_unit, sharex=sharex, sharey=sharey)
                

        
        if axesref is not None:
            self.current_axes = axesref
            
        return self.current_axes, self.current_figure
        

    #}

    #{ Plots
    def _handle_plotting_call(self, func_name, dsti=None, **kwargs):
        """
        [FUTURE]
        this function should be called by any plot_* function.
        
        It essentially handles storing all of the arguments passed to the 
        function inside a parameterset, attaches it to the bundle, and attaches
        it to the list of plots of the current axes.
        """
        plotref = kwargs.pop('plotref', None)
        axesref = kwargs.pop('axesref', None)
        figref = kwargs.pop('figref', None)
        
        if plotref is None:

            if func_name in ['plot_mesh', 'plot_custom']:
                plotref_base = func_name.split('_')[1]
            else:
                # then we need an intelligent default
                plotref_base = "_".join([dsti['ref'], dsti['context'], dsti['label']])
                
            plotref = plotref_base

            # change the ref if it already exists in the bundle
            existing_plotrefs = [pi['ref'] for pi in self.sections['plot']]
            i=1
            while plotref in existing_plotrefs:
                i+=1
                plotref = "{}_{:02}".format(plotref_base, i)


        loc = kwargs.pop('loc', None)
        sharex = kwargs.pop('sharex','_auto_')
        sharey = kwargs.pop('sharex','_auto_')
        figref = self._handle_current_figure(figref=figref)
        axesref, figref = self._handle_current_axes(axesref=axesref, figref=figref, loc=loc, sharex=sharex, sharey=sharey, x_unit=kwargs.get('x_unit','_auto_'), y_unit=kwargs.get('y_unit','_auto_'))
        axes_ps = self.get_axes(axesref)
        fig_ps = self.get_figure(figref)

        # TODO: plot_obs needs to be generalized to pull from the function called
        kwargs['datatwig'] = dsti.pop('twig',None) if dsti is not None else None
        ps = parameters.ParameterSet(context='plotting:{}'.format(func_name), ref=plotref)
        for k,v in kwargs.items():
            # try plot_ps first, then axes_ps and fig_ps
            if k in ps.keys():
                ps.set_value(k,v)
            elif k in axes_ps.keys():
                if k not in ['plotrefs']:
                    axes_ps.set_value(k,v)
            elif k in fig_ps.keys():
                if k not in ['axesrefs','axeslocs','axessharex','axessharey']:
                    fig_ps.set_value(k,v)
            elif k in ['label']:
                raise KeyError("parameter with qualifier {} forbidden".format(k))
            else:
                if isinstance(v, float):
                    _cast_type = float
                    _repr = '%f'
                elif isinstance(v, int):
                    _cast_type = int
                    _repr = '%d'
                elif isinstance(v, bool):
                    _cast_type = 'make_bool'
                    _repr = ''
                else:
                    _cast_type = str
                    _repr = '%s'
                
                new_parameter = parameters.Parameter(qualifier=k,
                        description = 'user added parameter: see matplotlib',
                        repr = _repr,
                        cast_type = _cast_type,
                        value = v)
                ps.add(new_parameter, with_qualifier=k)
        
        self._add_to_section('plot', ps)

        axes_ps.add_plot(plotref)
        
        return plotref, axesref, figref
        
    def get_plot(self, plotref):
        """
        [FUTURE]
        
        Get the ParameterSet object for a plot attached to the bundle
        
        :param plotref: ref to the axes
        :type plotref: str
        :return: the plot
        :rtype: ParameterSet
        """
        # TODO allow plotref=None ?
        # TODO allow passing kwargs ?
        plotref = plotref.split('@')[0]
        return self._get_by_section(plotref,"plot",kind=None)
        
    @rebuild_trunk
    def remove_plot(self, plotref=None):
        """
        [FUTURE]
        
        Remove a plot from the bundle.
        
        This does not check to make sure the plot is not referenced in
        any existing axes, so use with caution.
        
        :param plotref: ref of the plot
        :type plotref: str
        """
        plot_ps = self._get_by_section(plotref, "plot", kind=None)
        
        # we also need to check all axes, and remove this entry from any plotrefs
        #~ plotref_ref = plotref.split('@')[0]
        for axesref,axes_ps in self._get_dict_of_section('axes').items():
            axes_ps.remove_plot(plotref)
            
        self.sections['plot'].remove(plot_ps)
        
    def add_plot(self, plotref=None, axesref=None, figref=None, loc=None, **kwargs):
        """
        [FUTURE]
        
        Add an existing plot (created through plot_obs, plot_syn, plot_residuals, plot_mesh, etc)
        to an axes and figure in the bundle.

        If :envvar:`loc` points to a location in the :envvar:`figref` figure
        that is empty, a new axes will be created
                
        If :envvar:`axesref` is None and :envvar:`loc` is None, the plot will be attached to the current axes.
        If :envvar:`axesref` points to an existing axes, the plot will be attached 
        to that axes and it will become the current axes.
        If :envvar:`axesref` does not point to an existing axes, the axes will be
        created, attached, and will become the current axes.
        
        If :envvar:`figref` is None, the axes will be attached to the current figure.
        If :envvar:`figref` points to an existing figure, the axes will be attached
        to that figure and it will become the current figure.
        If :envvar:`figure` does not point to an existing figure, the figure will
        be created, attached, and will become the current figure.
        
        Any kwargs will get passed to the :envvar:`plotref` ParameterSet.
        
        :param plotref: ref of the plot
        :type plotref: str
        :param axesref: ref to a parent axes
        :type axesref: str
        :param figref: ref to a parent figure
        :type figref: str or None
        :param loc: location in the figure to attach the axes (see matplotlib.figure.Figure.add_subplot)
        :type loc: tuple with length 3
        :return: (axesref, figref)
        :rtype: tuple of strings
        """
        
        figref = self._handle_current_figure(figref)
        axesref, figref = self._handle_current_axes(axesref, figref, loc=loc)
        
        axes_ps = self.get_axes(axesref)
        axes_ps.add_plot(plotref)
        plot_ps = self.get_plot(plotref)
        
        for k,v in kwargs.items():
            if k in axes_ps.keys():
                axes_ps.set_value(k,v)
            else:
                plot_ps.set_value(k,v)
                
        return plotref, axesref, figref

        
    def draw_plot(self, plotref, axesref=None, ax=None, **kwargs):
        """
        [FUTURE]
        
        this function should make all the calls to mpl, and can either 
        be called by the user for a previously created plot, or by the 
        functions that created the plot initially.
        
        this function /technically/ draws this one single plot on the current axes - 
        we need to be careful about weird behavior here, or maybe we need to 
        be more clever and try to find (one of?) its parent axes if the plot_ps is already attached
        """
        plkwargs = {k:v for k,v in self.get_plot(plotref).items() if v is not ''}
        
        if ax is None:
            ax = plt.gca()
        
        axes_ps = self.get_axes(axesref)
        # will default to gca if axesref is None
        # in this case we may get unexpected behavior - either odd plotrefs
        # that we previously set or overriding axes options that were _auto_
        
        # this plot needs to be attached as a member of the axes if it is not
        #~ if plotref not in axes_ps['plotrefs']:
            #~ axes_ps.add_plot(plotref)
            
        plot_fctn = self.get_plot(plotref).context.split(':')[1]

        # Retrieve the obs/syn and the object it belongs to
        datatwig = plkwargs.pop('datatwig', None)

        # when passing to mpl, plotref will be used for legends so we'll override that with the ref of the plot_ps
        plkwargs['label'] = plkwargs.pop('ref')

        # and we need to pass the mpl axes
        if plot_fctn not in ['plot_custom']:
            plkwargs['ax'] = ax
        
        if datatwig is not None and plot_fctn not in ['plot_custom']:
            dsti = self._get_by_search(datatwig, return_trunk_item=True)
            ds = dsti['item']
            obj = self.get_object(dsti['label'])
            context = ds.get_context()
        
            # when passing to mpl the backend plotting function needs ref to be the dataset ref
            plkwargs['ref'] = ds['ref']
        
        axkwargs = {}
        for key in ['x_unit', 'y_unit', 'phased', 'xlabel', 'ylabel', 'title']:
            value = plkwargs.pop(key, None)

            # if provided by the plotting call, rewrite the value stored
            # in the axes_ps
            if value is not None:
                axes_ps.set_value(key, value)

            # either way, retrieve the value from axes_ps and handle defaults
            if plot_fctn not in ['plot_custom']:
                if key in ['x_unit', 'y_unit']:
                    if axes_ps.get_value(key) not in ['_auto_', u'_auto_']:
                        plkwargs[key] = axes_ps.get_value(key) 
                
                elif key in ['phased']:
                    if 'x_unit' not in plkwargs.keys() or axes_ps.get_value(key) or value is not None:
                        # TODO: the if statement above is hideous
                        plkwargs[key] = axes_ps.get_value(key)
                    # else it won't exist in plkwargs and will use the backend defaults
                
                elif key in ['xlabel', 'ylabel', 'title']:
                    # these don't get passed to the plotting call
                    # rather if they are not overriden here, they will receive
                    # there defaults from the output of the plotting call
                    pass
                    
        
        #~ if 'x_unit' in plkwargs.keys():
            #~ print '***', plkwargs['x_unit']
        #~ if 'y_unit' in plkwargs.keys():
            #~ print '***', plkwargs['y_unit']
                    
        # handle _auto_
        for key, value in plkwargs.items():
            if value=='_auto_':
                dump = plkwargs.pop(key)
       
        if plot_fctn in ['plot_obs', 'plot_syn']:
            output = getattr(plotting, 'plot_{}'.format(context))(obj, **plkwargs)
            
            artists = output[0]
            ds_ret = output[1]
            fig_decs = output[2]

        elif plot_fctn in ['plot_residuals']:
            category = context[:-3]
            output = getattr(plotting, 'plot_{}res'.format(category))(obj, **plkwargs)
            
            artists = output[0]
            ds_ret = output[1]
            fig_decs = output[2]
            
        elif plot_fctn in ['plot_mesh']:
            select = plkwargs.pop('select', None)
            cmap = plkwargs.pop('cmap', None)
            vmin = plkwargs.pop('vmin', None)
            vmax = plkwargs.pop('vmax', None)
            size = plkwargs.pop('size', None)
            dpi = plkwargs.pop('dpi', None)
            background = plkwargs.pop('background', None)
            savefig = plkwargs.pop('savefig', False)
            with_partial_as_half = plkwargs.pop('with_partial_as_half', False)
            time = plkwargs.pop('time', None)
            phase = plkwargs.pop('phase', None)
            compute_label = plkwargs.pop('compute_label', None)
            objref = plkwargs.pop('objref', None)
            category = context[:-3] if context is not None else 'lc'
            
            # get rid of unused kwargs
            for k in ['x_unit', 'y_unit', 'phased', 'label']:
                dump = plkwargs.pop(k, None)
            
            # Set the configuration to the correct time/phase, but only when one
            # (and only one) of them is given.
            if time is not None and phase is not None:
                raise ValueError("You cannot set both time and phase, please choose one")
            elif phase is not None:
                period, t0, shift = self.get_system().get_period()
                time = phase * period + t0
            
            # Observe the system with the right computations
            if time is not None:
                options = self.get_compute(compute_label, create_default=True).copy()
                observatory.observe(self.get_system(), [time], lc=category=='lc',
                                    rv=category=='rv', sp=category=='sp',
                                    pl=category=='pl', save_result=False, **options)
            
            # Get the object and make an image.
            self.get_object(objref).plot2D(select=select, cmap=cmap, vmin=vmin,
                         vmax=vmax, size=size, dpi=dpi, background=background,
                         savefig=savefig, with_partial_as_half=with_partial_as_half,
                         **plkwargs)
                         
            fig_decs = [['changeme','changeme'],['changeme','changeme']]
            artists = None

            
        elif plot_fctn in ['plot_custom']:
            function = plkwargs.pop('function', None)
            args = plkwargs.pop('args', [])
            if hasattr(ax, function):
                output = getattr(ax, function)(*args, **plkwargs)
                
            else:
                logger.error("{} not an available function for plt.axes.Axes".format(function))
                return

            # fake things for the logger 
            # TODO handle this better
            fig_decs = [['changeme','changeme'],['changeme','changeme']]
            artists = None
            context = 'changeme'                
            ds = {'ref': 'changeme'}
            
        else:
            logger.error("non-recognized plot type: {}".format(plot_fctn))
            return
        
        # automatically set axes_ps plotrefs if they're not already
        # The x-label
        if axes_ps.get_value('xlabel') in ['_auto_', u'_auto_']:
            axes_ps.set_value('xlabel', r'{} ({})'.format(fig_decs[0][0], fig_decs[1][0]))
        
        ax.set_xlabel(axes_ps.get_value('xlabel'))
        
        # The y-label
        if axes_ps.get_value('ylabel') in ['_auto_', u'_auto_']:
            axes_ps.set_value('ylabel', r'{} ({})'.format(fig_decs[0][1], fig_decs[1][1]))

        ax.set_ylabel(axes_ps.get_value('ylabel'))
        
        # The plot title
        if axes_ps.get_value('title') in ['_auto_', u'_auto_']:
            axes_ps.set_value('title', '{}'.format('mesh' if plot_fctn=='plot_mesh' else config.nice_names[context[:-3]]))
        
        ax.set_title(axes_ps.get_value('title'))  
        
        # The limits
        if axes_ps.get_value('xlim') not in [(None,None), '_auto_', u'_auto_']:
            ax.set_xlim(axes_ps.get_value('xlim'))
        if axes_ps.get_value('ylim') not in [(None,None), '_auto_', u'_auto_']:
            ax.set_ylim(axes_ps.get_value('ylim'))
            
        # The ticks
        #~ if axes_ps.get_value('xticks') != ['_auto_', u'_auto_']:
            #~ ax.set_xticks(axes_ps.get_value('xticks')
        #~ if axes_ps.get_value('yticks') != ['_auto_', u'_auto_']:
            #~ ax.set_xticks(axes_ps.get_value('yticks')
        #~ if axes_ps.get_value('xticklabels') != ['_auto_', u'_auto_']:
            #~ ax.set_xticklabels(axes_ps.get_value('xticklabels')
        #~ if axes_ps.get_value('yticklabels') != ['_auto_', u'_auto_']:
            #~ ax.set_xticklabels(axes_ps.get_value('yticklabels')
        
        logger.info("Plotted {} vs {} of {}({})".format(fig_decs[0][0],
                                   fig_decs[0][1], context, ds['ref']))
                                   
        #~ return ds_ret
        return {plotref: artists}


    
    def new_plot_obs(self, twig=None, **kwargs):
        """
        Make a plot of the attached observations (wraps pyplot.errorbar).
        
        This function is designed to behave like matplotlib's
        `plt.errorbar() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_
        function, with additional options.
        
        Thus, all kwargs (there are no args) are passed on to matplotlib's
        `plt.errorbars() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar>`_,
        except:
    
            - :envvar:`repeat=0`: handy if you are actually fitting a phase
              curve, and you want to repeat the phase curve a couple of times.
            - :envvar:`x_unit=None`: allows you to override the default units
              for the x-axis. If you plot times, you can set the unit to any
              time unit (days (``d``), seconds (``s``), years (``yr``) etc.). If
              you plot in phase, you can switch from cycle (``cy``) to radians
              (``rad``).
            - :envvar:`y_unit=None`: allows you to override the default units
              for the y-axis. Allowable values depend on the type of
              observations.
            - :envvar:`ax=plt.gca()`: the axes to plot on. Defaults to current
              active axes.
    
        Some of matplotlib's defaults are overriden. If you do not specify any
        of the following keywords, they will take the values:
    
            - :envvar:`ref`: the ref of the stored plotoptions.  This is the value
              that will also be passed on to matplotlib for the label of this curve
              in legends.
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
        
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        
        plotref, axesref, figref = self._handle_plotting_call('plot_obs', dsti, **kwargs)

        # now call the command to plot
        if ax is None:
            output = self.draw_figure(figref, fig=fig, live_plot=True, attach=True)
        else:
            output = self.draw_axes(axesref, ax=ax, live_plot=True, attach=True)
        
        self.currently_plotted.append((axesref.split('@')[0], plotref.split('@')[0]))
        
        #~ return None, (plotref, axesref, figref)
        return output, (plotref, axesref, figref)
        #~ return obs

        
    def new_plot_syn(self, twig=None, *args, **kwargs):
        """
        Plot simulated/computed observations (wraps pyplot.plot).
        
        This function is designed to behave like matplotlib's
        `plt.plot() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_
        function, with additional options.
        
        Thus, all args and kwargs are passed on to matplotlib's
        `plt.plot() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_,
        except:
        
            - :envvar:`dataref=0`: the reference of the lc to plot
            - :envvar:`repeat=0`: handy if you are actually fitting a phase curve,
              and you want to repeat the phase curve a couple of times.
            - :envvar:`x_unit=None`: allows you to override the default units for
              the x-axis. If you plot times, you can set the unit to any time unit
              (days (``d``), seconds (``s``), years (``yr``) etc.). If you plot
              in phase, you switch from cycle (``cy``) to radians (``rad``). 
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
        
            - :envvar:`ref`: the ref of the stored plotoptions.  This is the value
              that will also be passed on to matplotlib for the label of this curve
              in legends.
        
            
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
        
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        
        plotref, axesref, figref = self._handle_plotting_call('plot_syn', dsti, **kwargs)

        # now call the command to plot
        if ax is None:
            output = self.draw_figure(figref, fig=fig, live_plot=True, attach=True)
        else:
            output = self.draw_axes(axesref, ax=ax, live_plot=True, attach=True)
        
        self.currently_plotted.append((axesref.split('@')[0], plotref.split('@')[0]))
        
        #~ return None, (plotref, axesref, figref)
        return output, (plotref, axesref, figref)
        #~ return obs
        
    def new_plot_residuals(self, twig=None, **kwargs):
        """
        Plot the residuals between computed and observed for a given dataset
        
        [FUTURE]
        
        @param twig: the twig/twiglet to use when searching
        @type twig: str
        """
        # Retrieve the obs DataSet and the object it belongs to
        dsti = self._get_by_search(twig, context='*syn', class_name='*DataSet',
                                   return_trunk_item=True)
        
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        
        plotref, axesref, figref = self._handle_plotting_call('plot_residuals', dsti, **kwargs)

        # now call the command to plot
        if ax is None:
            output = self.draw_figure(figref, fig=fig, live_plot=True, attach=True)
        else:
            output = self.draw_axes(axesref, ax=ax, live_plot=True, attach=True)
        
        self.currently_plotted.append((axesref.split('@')[0], plotref.split('@')[0]))
        
        #~ return None, (plotref, axesref, figref)
        return output, (plotref, axesref, figref)
        #~ return obs
        
    def new_plot_custom(self, function, args=None, **kwargs):
        """
        [FUTURE]
        
        **VERY EXPERIMENTAL**
        Add a custom call through matplotlib and attach it to the bundle
        
        accepts function, args, and kwargs
        
        :param function: name of the matplotlib function to call, must be an attribute of matplotlib.axes.Axes
        :type function: str
        :param args: args to pass to the function
        """
        if not hasattr(plt.gca(), function):
            logger.error("{} not an available function for plt.axes.Axes".format(function))
            
        kwargs['function'] = function
        kwargs['args'] = args
        
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        
        plotref, axesref, figref = self._handle_plotting_call('plot_custom', None, **kwargs)
        
        # now call the command to plot
        if ax is None:
            output = self.draw_figure(figref, fig=fig, live_plot=True, attach=True)
        else:
            output = self.draw_axes(axesref, ax=ax, live_plot=True, attach=True)
        
        self.currently_plotted.append((axesref.split('@')[0], plotref.split('@')[0]))
        
        #~ return None, (plotref, axesref, figref)
        return output, (plotref, axesref, figref)
        #~ return obs

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
        observations, see:
        
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
        try:
            output = getattr(plotting, 'plot_{}'.format(context))(obj, *args, **kwargs)
        except ValueError:
            logger.warning("Cannot plot synthetics {}: no calculations found".format(kwargs['ref']))
            return None
        except IndexError:
            logger.warning("Cannot plot synthetics {}: no calculations found".format(kwargs['ref']))
            return None
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
        try:
            output = getattr(plotting, 'plot_{}res'.format(category))(obj, **kwargs)
        except ValueError:
            logger.warning("Cannot plot residuals {}: no calculations found".format(kwargs['ref']))
            return None
        res = output[1]
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
            
        return res
    
    
    def plot_prior(self, twig=None, **kwargs):
        """
        Plot a prior.
        
        [FUTURE]
        """
        prior = self.get_prior(twig)
        prior.plot(**kwargs)
        
        
    
    def write_syn(self, dataref, output_file, use_user_units=True, fmt='%.18e',
                  delimiter=' ', newline='\n', footer=''):
        """
        Write synthetic datasets to an ASCII file.
        
        By default, this function writes out the model in the same units as the
        data (:envvar:`use_user_units=True`). It will then also attempt to use
        the same columns as the observations were given in. It is possible to
        override these settings and write out the synthetic data in the internal
        model units of Phoebe. In that case, set :envvar:`use_user_units=False`
        
        Extra keyword arguments come from ``np.savetxt``, though headers are not
        supported as they are auto-generated to give information on column names
        and units.
        
        [FUTURE]
        
        Export the contents of a synthetic parameterset to a file
        
        :param dataref: the dataref of the dataset to write out
        :type dataref: str
        :param output_file: path and filename of the exported file
        :type output_file: str
        """
        # Retrieve synthetics and observations
        this_syn = self.get_syn(dataref)
        this_obs = self.get_obs(dataref)
        
        # Which columns to write out?
        user_columns = this_obs['user_columns'] and use_user_units
        columns = this_obs['user_columns'] if user_columns else this_obs['columns']
        
        # Make sure to consider all synthetic columns (e.g. flux might not have
        # been given in the obs just for simulation purposes)
        columns = this_syn['columns'] + [col for col in columns if not col in this_syn['columns']]
        
        # Filter out column names that do not exist in the synthetics
        columns = [col for col in columns if col in this_syn and len(this_syn[col])==len(this_syn)]
        
        # Which units to use? Start with default ones and override with user
        # given values
        units = [this_syn.get_unit(col) if this_syn.has_unit(col) else '--' \
                                                             for col in columns]
        
        if use_user_units and this_obs['user_units']:
            for col in this_obs['user_units']:
                units[columns.index(col)] = this_obs['user_units'][col]
        
        # Create header
        header = [" ".join(columns)]
        header+= [" ".join(units)]
        header = "\n".join(header)
        
        # Create data
        data = []
        
        # We might need the passband for some conversions
        try:
            passband = self.get_value_all('passband@{}'.format(dataref)).values()[0]
        except KeyError:
            passband = None
        
        for col, unit in zip(columns, units):
            this_col_data = this_syn[col]
            
            # Convert to right units if this column has units and their not
            # already the correct one
            if unit != '--':
                this_col_unit = this_syn.get_unit(col)
                if this_col_unit != unit:
                    this_col_data = conversions.convert(this_col_unit, unit,
                                               this_col_data, passband=passband)
            
            data.append(this_col_data)
        
        # Write out file
        np.savetxt(output_file, np.column_stack(data), header=header,
                   footer=footer, comments='# ', fmt=fmt, delimiter=delimiter,
                   newline=newline)
        
        # Report to the user
        info = ", ".join(['{} ({})'.format(col, unit) for col, unit in zip(columns, units)])
        logger.info("Wrote columns {} to file '{}'".format(info, output_file))
        
        
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
        
    def new_plot_mesh(self, objref=None, label=None, dataref=None, time=None, phase=None,
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
            dsti = self._get_by_search(dataref, context='*dep',
                            class_name='ParameterSet', all=True, return_trunk_item=True)[0]
            deps = dsti['item']
            #~ ref = deps['ref']
            context = deps.get_context()
            #~ category = context[:-3]
            #~ kwargs.setdefault('dataref', ref)
            kwargs.setdefault('context', context)
        else:
            # meh, let's fake the information we need for the plotting call
            dsti = {'context': 'lcdep', 'ref': 'lc', 'label': self.get_system().get_label()}

        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        
        kwargs['compute_label'] = label
        kwargs['objref'] = objref
        plotref, axesref, figref = self._handle_plotting_call('plot_mesh', dsti, **kwargs)

        # now call the command to plot
        if ax is None:
            output = self.draw_figure(figref, fig=fig, live_plot=True, attach=True)
        else:
            output = self.draw_axes(axesref, ax=ax, live_plot=True, attach=True)
        
        self.currently_plotted.append((axesref.split('@')[0], plotref.split('@')[0]))
        
        #~ return None, (plotref, axesref, figref)
        return output, (plotref, axesref, figref)
        #~ return obs
        
        
    def plot_mesh(self, objref=None, label=None, dataref=None, time=None, phase=None,
                  select='teff', cmap=None, vmin=None, vmax=None, size=800,
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
        
        If you've just ran :py:func:`Bundle.run_compute` for some observations,
        and you want to see what the mesh looks like after the last calculations
        have been performed, then this is what you need to do::
        
            >>> mybundle.plot_mesh()
            >>> mybundle.plot_mesh(objref='primary')
            >>> mybundle.plot_mesh(select='teff')
            >>> mybundle.plot_mesh(dataref='lc01')
        
        Giving a :envvar:`dataref` and/or setting :envvar:`select='proj'` plots
        the mesh coloured with the projected flux of that dataref.
        
        .. warning::
            
            1. If you have no observations attached, you cannot use ``select='proj'``
               because there will be no flux computed. You can then only plot
               mesh quantities like ``teff``, ``logg`` etc.
            
            2. It is strongly advised only to use this function this way when
               only one set of observations has been added: otherwise it is not
               guarenteed that the dataref has been computed for the last time
               point.
            
            3. Although it can sometimes be convenient to use this function
               plainly without arguments you need to be careful for the
               :envvar:`dataref` that is used to make the plot. The dataref
               will show up in the logger information.
        
        Extra keyword arguments and the return values are explained
        in :py:func:`phoebe.backend.observatory.image`.
        
        :param objref: object/system label of which you want to plot the mesh.
         The default means the top level system.
        :type objref: str
        :param label: compute label which you want to use to calculate the mesh
         and its properties. The default means the ``default`` set.
        :type label: str
        :return: figure properties, artist properties, patch collection
        :rtype: dict, dict, matplotlib patch collection
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
            kwargs.setdefault('ref', '__bol')
            
        # Set the configuration to the correct time/phase, but only when one
        # (and only one) of them is given.
        if time is not None and phase is not None:
            raise ValueError("You cannot set both time and phase to zero, please choose one")
        elif phase is not None:
            period, t0, shift = self.get_system().get_period()
            if np.isinf(period):
                time = t0
            else:
                time = phase * period + t0
        
        # Observe the system with the right computations
        if time is not None:
            options = self.get_compute(label, create_default=True).copy()
            observatory.observe(self.get_system(), [time], lc=category=='lc',
                                rv=category=='rv', sp=category=='sp',
                                pl=category=='pl', save_result=False, **options)
        
        # Get the object and make an image.
        try:
            out = self.get_object(objref).plot2D(select=select, cmap=cmap, vmin=vmin,
                     vmax=vmax, size=size, dpi=dpi, background=background,
                     savefig=savefig, with_partial_as_half=with_partial_as_half,
                     **kwargs)
        except ValueError:
            # Most likely, the user did not add any data, did not run_compute
            # before or did not give a time and phase explicitly here...
            if time is None and not len(self.get_system().mesh):
                raise ValueError("Insufficient information to plot mesh: please specify a phase or time")
            else:
                raise
        return out
        
        
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
    
    def check(self, return_errors=False):
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
        
        return self.get_system().check(return_errors=return_errors)

            
        
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
        if ign in contexts:
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
