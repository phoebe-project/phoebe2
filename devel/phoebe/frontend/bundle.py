"""
Top level interface to Phoebe2.

The Bundle aims at providing a user-friendly interface to a Body or BodyBag,
such that parameters can easily be queried or changed, data added, results
plotted and observations computed. It does not contain any implementation of
physics; that is the responsibility of the backend and the associated
library.

**Initialisation**

Simply initialize a Bundle with default parameters via::

    >>> b = Bundle()

Or use other predefined systems as a starting point::

    >>> b = Bundle('binary')
    >>> b = Bundle('overcontact')
    >>> b = Bundle('hierarchical_triple')
    >>> b = Bundle('binary_pulsating_primary')
    >>> b = Bundle('pulsating_star')
    >>> b = Bundle('sun')
    >>> b = Bundle('vega')

For a complete list, see the documentation of the :py:mod:`create <phoebe.parameters.create>`
module (section *Generic systems* and *specific targets*).

**Phoebe1 compatibility**

Phoebe2 can be used in a Phoebe1 compatibility mode ('legacy' mode), which is
most easily when you start immediately from a legacy parameter file:

    >>> b = Bundle('legacy.phoebe')

When a Bundle is loaded this way, computational options are added automatically
to the Bundle to mimick the physics that is available in Phoebe1. These options
are collected in a ParameterSet of context 'compute' with label ``legacy``.
The most important parameters are listed below (some unimportant ones have been
excluded so if you try this, you'll see more parameters)

    >>> print(b['legacy'])
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

>>> b.run_compute(label='legacy')
>>> print(b.get_logp())

>>> b.plot_obs('lightcurve_0', fmt='ko')
>>> b.plot_obs('primaryrv_0', fmt='ko')
>>> b.plot_obs('secondaryrv_0', fmt='ro')

>>> b.plot_syn('lightcurve_0', 'k-', lw=2)
>>> b.plot_syn('primaryrv_0', 'k-', lw=2)
>>> b.plot_syn('secondaryrv_0', 'r-', lw=2)

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
from matplotlib import collections
from mpl_toolkits.mplot3d import art3d, Axes3D
import tempfile
import copy
import os
import sys
import re
import readline
import json
import marshal, types
from phoebe.utils import callbacks, utils, plotlib, coordinates, config
from phoebe.parameters import parameters
from phoebe.parameters import definitions
from phoebe.parameters import datasets
from phoebe.parameters import create
from phoebe.parameters import tools
from phoebe.parameters import feedback as mod_feedback
from phoebe.backend import fitting, observatory
from phoebe.backend import  plotting as beplotting
from phoebe.backend import universe
from phoebe.backend import decorators
from phoebe.atmospheres import passbands
from phoebe.atmospheres import limbdark
from phoebe.io import parsers
from phoebe.io import ascii
from phoebe.dynamics import keplerorbit
from phoebe.frontend.usersettings import Settings
from phoebe.frontend.common import Container, rebuild_trunk
from phoebe.frontend import sample
from phoebe.units import conversions
from phoebe.frontend import phcompleter
from phoebe.frontend import stringreps
from phoebe.frontend import plotting

logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())

class Bundle(Container):
    """
    Class representing a collection of systems and stuff related to it.
    
    Input/output:
    
    .. autosummary::
    
        Bundle.__init__
        Bundle.set_system
        Bundle.save_pickle

    Accessing and changing parameters via twigs:

    .. autosummary::

        phoebe.frontend.common.Container.get_value
        phoebe.frontend.common.Container.get_value_all
        phoebe.frontend.common.Container.set_value
        phoebe.frontend.common.Container.set_value_all
        phoebe.frontend.common.Container.get_unit
        phoebe.frontend.common.Container.set_unit
        phoebe.frontend.common.Container.set_unit_all
        phoebe.frontend.common.Container.get_ps
        phoebe.frontend.common.Container.set_ps
        phoebe.frontend.common.Container.get_parameter
        phoebe.frontend.common.Container.get_adjust
        phoebe.frontend.common.Container.set_adjust
        phoebe.frontend.common.Container.get_prior
        phoebe.frontend.common.Container.set_prior
        phoebe.frontend.common.Container.remove_prior
        phoebe.frontend.common.Container.get_posterior
        phoebe.frontend.common.Container.remove_posterior


        phoebe.frontend.common.Container.attach_ps
        Bundle.add_parameter
        phoebe.frontend.common.Container.twigs
        phoebe.frontend.common.Container.search

    Adding and handling data:

    .. autosummary::

        Bundle.lc_fromfile
        Bundle.lc_fromarrays
        Bundle.lc_fromexisting
        Bundle.rv_fromfile
        Bundle.rv_fromarrays
        Bundle.rv_fromexisting
        Bundle.sed_fromfile
        Bundle.sed_fromarrays
        Bundle.sed_fromexisting
        Bundle.sp_fromfile
        Bundle.sp_fromarrays
        Bundle.sp_fromexisting
        Bundle.if_fromfile
        Bundle.if_fromarrays
        Bundle.if_fromexisting
        Bundle.etv_fromfile
        Bundle.etv_fromarrays
        Bundle.etv_fromexisting

        Bundle.disable_data
        Bundle.enable_data
        Bundle.disable_lc
        Bundle.enable_lc
        Bundle.disable_rv
        Bundle.enable_rv
        Bundle.disable_sp
        Bundle.enable_sp
        Bundle.disable_sed
        Bundle.enable_sed
        Bundle.disable_etv
        Bundle.enable_etv
               
        Bundle.write_syn

    Computations and fitting:

    .. autosummary::

        phoebe.frontend.common.Container.add_mpi
        phoebe.frontend.common.Container.add_compute
        Bundle.run_compute
        phoebe.frontend.common.Container.add_fitting
        Bundle.run_fitting

    High-level plotting functionality:

    .. autosummary::

        Bundle.plot_obs
        Bundle.plot_syn
        Bundle.plot_residuals
        Bundle.plot_mesh
        
        Bundle.attach_plot_obs
        Bundle.attach_plot_syn
        Bundle.attach_plot_mesh
        
        Bundle.draw
        Bundle.draw_figure
        Bundle.draw_axes
        Bundle.draw_plot

    Convenience functions:

    .. autosummary::

        Bundle.get_datarefs
        Bundle.get_lc_datarefs
        Bundle.get_rv_datarefs
        Bundle.get_obs
        Bundle.get_dep
        Bundle.get_syn
        Bundle.get_system
        Bundle.get_object
        Bundle.get_orbitps
        Bundle.get_orbit
        Bundle.get_children
        Bundle.get_parent
        Bundle.get_meshps
        Bundle.get_mesh
        Bundle.get_adjustable_parameters



    **Printing information**

    An overview of all the Parameters and observations loaded in a Bundle can
    be printed to the screen easily using::

        b = phoebe.Bundle()
        print(b)

    Extra functions are available that return informational strings

    .. autosummary::

        Bundle.summary
        phoebe.frontend.common.Container.info


    **Initialization**

    You can initiate a bundle in different ways:

      1. Using the default binary parameters::

          b = Bundle()

      2. Via a Phoebe Legacy ASCII parameter file::

          b = Bundle('legacy.phoebe')

      3. Via a Body or BodyBag::

          mysystem = phoebe.create.from_library('V380_Cyg', create_body=True)
          b = Bundle(mysystem)

      4. Via the predefined systems in the library::

          b = Bundle('V380_Cyg')

    For more details, see :py:func:`set_system`.

    **Interface**

    The interaction with a Bundle is much alike interaction with a Python
    dictionary. The following functionality is implemented and behaves as
    expected::

            # return the value of the period if it exists, raises error if 'period' does not exist
            period = b['period']

            # set the value of the period if it exists, raises error otherwise
            b['period'] = 5.0

            # return the value of the period if it exists, else returns None
            period = b.get('period')

            # return the value of the period if it exists, else returns default_value (whatever it is)
            period = b.get('period', default_value)

            # returns a list of available keys
            keys = b.keys()

            # returns a list of values
            values = b.values()

            # iterate over the keys in the Bundle
            for key in b:
                print(key, b[key])

    .. important::

        *keys* are referred to as *twigs* in the context of Bundles. They behave
        much like dictionary keys, but are much more flexible to account for the
        plethora of available parameters (and possible duplicates!) in the
        Bundle. For example, both stars in a binary need atmosphere tables in
        order to compute their bolometric luminosities. This is done via the
        parameter named ``atm``, but since it is present in both components,
        you need to ``@`` operator to be more specific:

            >>> b['atm@primary'] = 'kurucz'
            >>> b['atm@secondary'] = 'blackbody'




    **Structure of the Bundle**

    A Bundle contains:

        - a Body (or BodyBag), called :envvar:`system` in this context.
        - a list of compute options which can be stored and used to compute observables.
        - a list of fitting options which can be stored and used for fitting observables.
        - a list of MPI options which can be stored and used during either compute or fitting
        - a list of options which can be used to reproduce plots and figures


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
        self.sections['mpi'] = []
        self.sections['dataset'] = []
        self.sections['feedback'] = []
        self.sections['figure'] = []
        self.sections['axes'] = []
        self.sections['plot'] = []

        self.current_axes = None
        self.current_figure = None
        self.currently_plotted = [] # this will hold (axesref, plotref) pairs of items already plotted on-the-fly

        # Now we load a copy of the usersettings into self.usersettings
        # Note that by default these will be removed from the bundle before
        # saving, and reimported when the bundle is loaded
        self.set_usersettings()

        # Let's keep track of the filename whenever saving the bundle -
        # if self.save() is called without a filename but we have one in
        # memory, we'll try to save to that location.
        self.filename = None

        # Next we'll set the system, which will parse the string sent
        # to init
        #~ if system is not None:
            # then for now (even though its hacky), we'll initialize
            # everything by setting the default first
            #~ self.set_system()
        self.system = None
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
        readline.set_completer_delims(' \t\n`~!#$%^&*)-=+]{}\\|;,<>/?')  # removed ':' since its used for subcontexts
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
                        self.sections = contents.sections.copy()

                else:
                    if not self.system:
                        self.set_system()
                    self._load_json(system)
                    file_type = 'json'

            else:
                try:
                #~ if True:
                    if not self.system:
                        self.set_system()
                    self._from_dict(system)
                except:
                    # As a last resort, we pass it on to 'body_from_string' in the
                    # create module:
                    system = create.body_from_string(system)
                else:
                    file_type = 'json'

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
        
    def hierarchy(self, change_labels=False):
        """
        [FUTURE]
        
        Show a string representation of the hierarchy.  Use change_labels=True
        to change the labels of all objects to a consistent format.
        
        The string representation shows the labels of all of the components in the
        system (not orbits/BodyBags).  The number of hyphens separating any two entries
        depicts the period - with more hyphens being longer periods.  The components
        in a single orbit are ordered based on mass ratio - with the more massive
        component coming first.
        
        :param change_labels: whether to change labels across the system
        :type change_label: bool
        :return: string representation of the hierarchy of the system
        :rtype: str
        """
        def get_kind(thing):
            if thing.__class__.__name__ in ['Star', 'BinaryRocheStar', 'BinaryStar', 'PulsatingBinaryRocheStar']:
                kind = 'star'
            elif thing.__class__.__name__ in ['Disk']:
                kind = 'disk'
            else:
                kind = 'component'
                
            return kind
            
        def get_sublabel(label):
            """
            get the capital letters at the end of a newly-created label
            
            eg. if label='orbitBC' return 'BC'
            """
            return re.findall('[A-Z][A-Z]*', label)[0]
        
        
        system = self.get_system()
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        comp_i = 0
        
        # handle both cases - top-level is bodybag or not
        if hasattr(system, 'bodies'):
            comps = []
            orbs = [system.get_period()[0]]
            orb_labels = [system.get_label()]
        else:
            orbs = []
            orb_labels = []
            if change_labels:
                kind = get_kind(system)
                self['label@'+system.get_label()] = kind+letters[comp_i]           
                comp_i += 1
            comps = [system.get_label()]
            
        for loc, thing in system.walk_all():
            if isinstance(thing, universe.Body):
                if isinstance(thing, universe.BodyBag):
                    orbs.append(thing.get_children()[0].get_period()[0])
                    orb_labels.append(thing.get_label())
                else:
                    kind = get_kind(thing)
                     
                    if change_labels:
                        self['label@'+thing.get_label()] = kind+letters[comp_i]
                        comp_i += 1 
                    comps.append(thing.get_label())
                    
        orbs_sort = orbs[:] # copy
        orbs_sort.sort()
        
        if len(orbs_sort):
            hierarchy = ''.join([c+' '+'-'*(orbs_sort.index(o)+1)+' ' for c,o in zip(comps[:-1], orbs)]+[comps[-1]])
        else:
            hierarchy = ' '.join(c for c in comps)
            
        # change orbit labels
        if change_labels:
            orb_labels.reverse()
            for orb_label in orb_labels:
                self['label@'+orb_label] = 'orbit'+get_sublabel(self['orbit@'+orb_label]['c1label'])+get_sublabel(self['orbit@'+orb_label]['c2label'])
            
        return hierarchy

    def clear_syn(self):
        """
        Clear all synthetic datasets.

        Simply a shortcut to :py:func:`bundle.get_system().clear_synthetic() <phoebe.backend.universe.Body.clear_synthetic>`
        """
        self.get_system().clear_synthetic()
        self._build_trunk()

    def set_time(self, time=0.0, computelabel=False, **kwargs):
        """
        [FUTURE]

        Update the mesh of the entire system to a specific time.

        To set the system to a specific phase do:
        >>> bundle.set_time(time=phoebe.to_time(phase, bundle.get_ephem(objref)))

        If you'd like your datasets to be computed at the single time and
        fill the values in the mesh, either set computelabel to True or None
        (to use default compute options) or the label of a compute PS stored
        in the bundle.  (See :py:func:`run_compute`)

        @param time: time (ignored if phase provided)
        @type time: float
        @param computelabel: computelabel (or False to skip computation
                and only update the mesh
        @type computelabel: bool or string or None
        """

        system = self.get_system()

        # reset will force the mesh to recompute - we'll do this always
        # to be conservative, but it is really only necessary if something
        # has changed the equipotentials (q, e, omega, etc)
        system.reset()


        if computelabel is False:
            # calling system.set_time() will now for the mesh to be computed
            # from scratch
            system.set_time(time)

        else:
            options = self.get_compute(label, create_default=True).copy()
            mpi = kwargs.pop('mpi', None)

            # now temporarily override with any values passed through kwargs
            for k,v in kwargs.items():
                #if k in options.keys(): # otherwise nonexisting kwargs can be given
                try:
                    options.set_value(k,v)
                except AttributeError:
                    raise ValueError("run_compute does not accept keyword '{}'".format(k))


            raise NotImplementedError

    #}
    #{ Parameters/ParameterSets

    def get_logp(self, dataset=None, usercosts=None):
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

        if dataset is not None:
            # First disable/enabled correct datasets
            old_state = []
            location = 0

            for obs in self.get_system().walk_type(type='obs'):
                old_state.append(obs.get_enabled())
                this_state = False
                if dataset == obs['ref'] or dataset == obs.get_context():
                    #~ if index is None or index == location:
                    if True:
                        this_state = True
                    location += 1
                obs.set_enabled(this_state)

        # Then compute statistics
        logf, chi2, n_data = self.get_system().get_logp(include_priors=True, usercosts=usercosts)

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
        if twig is None or twig == '__nolabel__' or twig == '':
            return self.get_system()
        #~ return self._get_by_search(twig, kind='Body')
        return self._get_by_search(twig, context='object', method='robust_notfirst')

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

            b = Bundle()
            position, velocity, bary_time, proper_time = b.get_orbit('primary')

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
                time = np.arange(t0, t0+period_outer+t_step, t_step)
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
        # pos[2] is already flipped above

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
        #~ return self._get_by_search('orbit@{}'.format(twig), kind='ParameterSet', context='orbit')
        return self._get_by_search(twig, kind='ParameterSet', context='orbit', method='robust_notfirst')


    def get_mesh(self, twig=None):
        """
        [FUTURE]

        Retrieve the mesh (np record array) for a given object

        If no twig is provided, the mesh for the entire system will be returned

        @param twig: the twig/twiglet of the object
        @type twig: str or None
        @return: mesh
        @rtype: np record array
        """
        return self.get_object(twig=twig).get_mesh()


    def get_meshps(self, twig=None):
        """
        [FUTURE]

        retrieve the mesh ParameterSet that belongs to a given component

        @param twig: the twig/twiglet to use when searching
        @type twig: str
        @return: the mesh PS
        @rtype: ParameterSet
        """
        #~ return self._get_by_search('mesh@{}'.format(twig), kind='ParameterSet', context='mesh*')
        return self._get_by_search(twig, kind='ParameterSet', context='mesh*', method='robust_notfirst')

    def get_ephem(self, objref=None):
        """
        [FUTURE]

        Get the ephemeris of an object in the system.  Not objref should
        be the object containing that ephemeris.  Asking for the ephemeris
        of a star will return (computing from syncpar if necessary) the
        rotation period of the star.  Asking for the ephemeris of an inner-
        binary in an hierarchical system will return the period of the inner-
        binary, NOT the period of the inner-binary in the outer-orbit.

        The ephemeris of a star returns the rotation period and t0 of its
        parent orbit (if any).

        The ephemeris of an orbit returns its period, t0, phshift, and dpdt.

        If objref is None, this will return for the top-level of the system

        @param objref: the object whose *child* orbit contains the ephemeris
        @type objref: str
        @return: period, t0, (phshift, dpdt)
        @rtype: dict
        """

        ephem = {}

        if objref is None:
            objref = self.get_system().get_label()

        # first check if we're an orbit - that's the easiest case
        orb_ps = self._get_by_search(label=objref, kind='ParameterSet',
                      context='orbit', all=True, ignore_errors=True)

        if len(orb_ps):
            for k in ['period', 't0', 'phshift', 'dpdt']:
                ephem[k] = orb_ps[0].get_value(k)
            return ephem

        # not an orbit - let's check for a component
        comp_ps = self._get_by_search(label=objref, kind='ParameterSet',
                      context='component', all=True, ignore_errors=True)

        if len(comp_ps):
            logger.warning("retrieving rotational period of {}".format(objref))
            # let's see if it has a parent orbit
            period, t0, shift = self.get_object(objref).get_period()
            # we'll ignore the period, but will use the t0
            ephem['t0'] = t0

            if 'rotperiod' in comp_ps[0].keys():
                ephem['period'] = comp_ps[0].get_value('rotperiod')
            elif 'syncpar' in comp_ps[0].keys():
                # then let's compute rotation period from the orbital
                # period and synchronicity
                ephem['period'] = period/comp_ps[0].get_value('syncpar')

        return ephem

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
            
    #~ def convert_to_stars(self, twig=None):
        #~ """
        #~ [FUTURE]
        #~ 
        #~ twig should be the twig of the parent orbit of two binary roche stars that 
        #~ you want to convert to stars
        #~ """
        #~ 
        #~ brss = self.get_children(twig)
        #~ orbit = self.get_orbit(twig)
        #~ 
        #~ if not all([isinstance(brs, phoebe.BinaryRocheStar) for brs in brss]):
            #~ logger.error("all children of {} must be of type BinaryRocheStar".format(twig))
            #~ return False
        #~ if not len(brss)==2:
            #~ logger.error("{} must have 2 children".format(twig))
            #~ return False
            #~ 
        #~ s1, s2, orbit = berts_function(brs[0], brs[1], orbit)
        #~ 
        #~ # now we need to replace and possibly rebuild the trunk
        #~ 
        #~ 
        #~ raise NotImplementedError
        
    #~ def convert_to_binaryrochestars(self, twig=None):
        #~ """
        #~ [FUTURE]
        #~ 
        #~ twig should be the twig of the parent orbit of two binary roche stars that 
        #~ you want to convert to stars
        #~ """
        #~ 
        #~ ss = self.get_children(twig)
        #~ orbit = self.get_orbit(twig)
        #~ 
        #~ if not all([isinstance(s, phoebe.Star) for s in ss]):
            #~ logger.error("all children of {} must be of type Star".format(twig))
            #~ return False
        #~ if not len(brss)==2:
            #~ logger.error("{} must have 2 children".format(twig))
            #~ return False        
#~ 
        #~ raise NotImplementedError
        

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

                # if there is a time column in the dataset, sort according to
                # time
                if 'time' in ds and not np.all(np.diff(ds['time'])>=0):
                    logger.warning("The observations are not sorted in time -- sorting now")
                    ds.sort()

                comp.add_obs(ds)

        # Initialize the mesh after adding stuff (i.e. add columns ld_new_ref...
        self.get_system().init_mesh()
        self._build_trunk()

    def write_fromfile(self, filename, outfilename, category='lc', objref=None, dataref=None,
                          columns=None, units=None, **kwargs):

        """
        [FUTURE] - IN DEVELOPMENT
        """

        # the plan here is to have a function which can take a file and
        # user passed kwargs and write the file in a phoebe-parsable format
        # so that it can then be attached to this (or in theory another)
        # bundle with just the filename

        #~ output = self._parse_fromfile(filename, category, objref, dataref, columns, units)
        #~ return output
        #~ (columns,components,datatypes,units,ncols),(pbdep,dataset) = stuff


        (p_columns,p_components,p_datatypes,p_units,p_ncols),(pbdep,dataset) = datasets.parse_header(filename)

        # deal with args
        if columns is None:
            columns = p_columns
        if objref is None:
            components = p_components
        else:
            components = objref
        if units is None:
            units = p_units

        # deal with kwargs
        if dataref is not None:
            pbdep['ref'] = dataref
        for k,v in kwargs.items():
            if k in pbdep.keys():
                pbdep[k] = v

        # filter to only used columns
        column_inds = [i for i,col in enumerate(columns) if col is not None]

        f = open(outfilename, 'w')
        f.write('#-----------------\n')
        # begin header

        # pbdep entries
        for k,v in pbdep.items():
            f.write('# {} = {}\n'.format(k,v))

        # column info
        f.write('# NAME {}\n'.format(" ".join([columns[i] for i in column_inds])))
        #~ f.write('# UNIT {}\n'.format())
        #~ f.write('# COMPONENT {}\n'.format())


        f.write('#-----------------\n')
        # end header

        # data
        data_rows = np.loadtxt(filename)
        for i,row in enumerate(data_rows):
            f.write('{}\n'.format('\t'.join([str(row[i]) for i in column_inds])))

        f.close()

        return outfilename

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
                                        full_output=True, ref=dataref, **kwargs)

        elif category == 'sp':
            if subcategory is None:
                try:
                    output = datasets.parse_spec_timeseries(filename, columns=columns,
                                       components=objref, units=units,
                                       full_output=True, ref=dataref,
                                       **kwargs)
                except IOError:
                    raise IOError("Either the file '{}' does not exist or you've specified a snapshot spectrum as a timeseries (set snapshot=True in sp_fromfile)".format(filename))
            # Then this is a shapshot
            else:
                output = datasets.parse_spec_as_lprof(filename, columns=columns,
                                       components=objref, units=units,
                                       full_output=True, ref=dataref,
                                       **kwargs)

        elif category == 'sed':
            scale, offset = kwargs.pop('adjust_scale', False), kwargs.pop('adjust_offset', False)
            output = datasets.parse_phot(filename, columns=columns,
                  units=units, group=dataref,
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
                        estimate_sigma=True,
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

        :param category: one of 'lc', 'rv', 'sp', 'etv', 'if', 'pl', 'orb'
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
            if category in ['lc','if','sp', 'pl']:
                # then top-level
                components = [self.get_system()]
                #logger.warning('components not provided - assuming {}'.format([comp.get_label() for comp in components]))
            else:
                logger.error('data_fromarrays failed: components need to be provided via the objref argument')
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
            
        dep_cols = {'lc': 'flux', 'rv': 'rv', 'sp': 'flux', 'etv': 'etv'}
        if estimate_sigma and category in dep_cols.keys() and ('sigma' not in ds['columns'] or len(ds['sigma'])==0) and len(ds[dep_cols[category]]):
            
            logger.warning('sigmas not provided, estimating from provided column: {}.  To prevent this, set estimate_sigma=False'.format(dep_cols[category]))

            ds.estimate_sigma(from_col=dep_cols[category], to_col='sigma')
            ds['columns'] = ds['columns'] + ['sigma']

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

                    
                    # handle situations in which time or phase are being switched
                    if 'columns' not in kwargs.keys() and 'columns' in new.keys() and 'columns' in existing.keys():
                        if kwargs.get('phase', None) is not None and kwargs.get('time', None) is None:
                            # then switch 'time' in columns to 'phase'
                            tmp = copy.deepcopy(existing['columns'])
                            tmp[tmp.index('time')] = 'phase'
                            new['columns'] = tmp
                            new['time'] = []
                            
                            processed_kwargs.append('columns')
                            
                        elif kwargs.get('time', None) is not None and kwargs.get('phase', None) is None:
                            # then switch 'phase' in columns to 'time'
                            tmp = copy.deepcopy(existing['columns'])
                            tmp[tmp.index('phase')] = 'time'
                            new['columns'] = tmp
                            new['phase'] = []
                            
                            processed_kwargs.append('columns')

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
                      scattering=None, method=None, estimate_sigma=True):
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
        :param estimate_sigma: whether to estimate sigmas if not provided
        :type estimate_sigma: bool
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

        >>> bundle.lc_fromfile('myfile.lc', passband='JOHNSON.V')

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
                  if set_kwargs[key] is not None and key not in ['self', 'from_dataref', 'to_dataref']}

        self.data_fromexisting(to_dataref, from_dataref, category='lc', **set_kwargs)

    def rv_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      rv=None, sigma=None, flag=None, weight=None,
                      exptime=None, samprate=None, offset=None, scale=None,
                      method=None, atm=None, ld_func=None, ld_coeffs=None,
                      passband=None, pblum=None, l3=None, alb=None, beaming=None,
                      scattering=None, estimate_sigma=True):
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
        :param estimate_sigma: whether to estimate sigmas if not provided
        :type estimate_sigma: bool
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


    def rv_fromexisting(self, to_dataref, from_dataref=None, time=None, phase=None,
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
                  if set_kwargs[key] is not None and key not in ['self', 'from_dataref', 'to_dataref']}

        self.data_fromexisting(to_dataref, from_dataref, category='rv', **set_kwargs)

    def etv_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      etv=None, sigma=None, flag=None, weight=None, estimate_sigma=True):
        """
        [FUTURE]

        Create and attach eclipse timing templates to compute the model.

        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults. For example, the :envvar:`atm`,
        :envvar:`ld_func` and :envvar:`ld_coeffs` arguments are taken from the
        component (which reflect the bolometric properties) unless explicitly
        overriden.

        A unique data reference (:envvar:`dataref`) is added automatically if
        none is provided by the user. A readable system of unique references is
        applied: the first etv that is added is named ``etv01`` unless
        that reference already exists, in which case ``etv02`` is tried and so
        on. On the other hand, if a :envvar:`dataref` is given at it already
        exists, it's settings are overwritten.

        If no :envvar:`objref` is given, the etv is added to the total
        system, and not to a separate component. That is probably fine in almost
        all cases, since you observe the whole system simultaneously and not the
        components separately.

        Note that you cannot add :envvar:`times` and :envvar:`phases` simultaneously.

        **Example usage**

        It doesn't make much sense to leave the time array empty, since then
        nothing will be computed. Thus, the minimal function call that makes
        sense is something like:

        >>> bundle = phoebe.Bundle()
        >>> bundle.etv_fromarrays(time=np.linspace(0, 10.33, 101))

        or in phase space:

        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.etv_fromarrays(phase=phase, passband='GENEVA.V')

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
        :param estimate_sigma: whether to estimate sigmas if not provided
        :type estimate_sigma: bool
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
        return self.data_fromarrays(category='etv', **set_kwargs)


    def etv_fromfile(self, filename, objref=None, dataref=None, columns=None,
                      units=None, offset=None, scale=None):
        """
        [FUTURE]

        Add an eclipse timing from a file.

        The data contained in :envvar:`filename` will be loaded to the object
        with object reference :envvar:`objref` (if ``None``, defaults to the whole
        system), and will have data reference :envvar:`dataref`. If no
        :envvar:`dataref` is is given, a unique one is generated: the first
        etv that is added is named 'etv01', and if that one already
        exists, 'etv02' is tried and so on.

        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults.

        **Example usage**

        A plain file can loaded via::

        >>> bundle.etv_fromfile('myfile.etv')

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
        return self.data_fromfile(category='etv', **set_kwargs)


    def etv_fromexisting(self, to_dataref, from_dataref=None, time=None, phase=None,
                      etv=None, sigma=None, flag=None, weight=None):
        """
        [FUTURE]

        Duplicate an existing etv to a new one with a different dataref.

        This can be useful if you want to change little things and want to
        examine the difference between computing options easily, or if you
        want to compute an existing etv onto a higher time resolution
        etc.

        Any extra kwargs are copied to any etvdep or etvobs where the key is
        present.

        All *etvdeps* and *etvobs* with dataref :envvar:`from_dataref` will be
        duplicated into an *etvdep* with dataref :envvar:`to_dataref`.

        For a list of available parameters, see :ref:`etvdep <parlabel-phoebe-etvdep>`
        and :ref:`etvobs <parlabel-phoebe-etvobs>`.

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
                  if set_kwargs[key] is not None and key not in ['self', 'from_dataref', 'to_dataref']}

        self.data_fromexisting(to_dataref, from_dataref, category='etv', **set_kwargs)


    def sed_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                       passband=None, flux=None, sigma=None, unit=None,
                       scale=None, offset=None, auto_scale=False,
                       auto_offset=False, estimate_sigma=True, **kwargs):
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
        
        :param estimate_sigma: whether to estimate sigmas if not provided
        :type estimate_sigma: bool
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
                      vgamma_offset=None, profile=None, R=None, vmicro=None,
                      depth=None, atm=None, ld_func=None, ld_coeffs=None,
                      passband=None, pblum=None, l3=None, alb=None,
                      beaming=None, estimate_sigma=True):
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

        [FUTURE]

        :param objref: component for each column in file
        :type objref: None, str, list of str or list of bodies
        :param dataref: name for ref for all returned datasets
        :type dataref: str
        :param estimate_sigma: whether to estimate sigmas if not provided
        :type estimate_sigma: bool
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
                      clambda=None, wrange=None, vgamma_offset=None,
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

    def pl_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      wavelength=None, flux=None, continuum=None, sigma=None,
                      V=None, sigma_V=None, Q=None, sigma_Q=None, U=None,
                      sigma_U=None,
                      flag=None, R_input=None, offset=None, scale=None,
                      vgamma_offset=None, profile=None, R=None, vmicro=None,
                      depth=None, atm=None, ld_func=None, ld_coeffs=None,
                      passband=None, pblum=None, l3=None, alb=None,
                      beaming=None, estimate_sigma=True):
        """
        Create and attach spectrapolarimetry templates to compute the model.

        See :py:func:`sp_fromarrays` for more information.
        """
        # retrieve the arguments with which this function is called
        set_kwargs, posargs = utils.arguments()

        # filter the arguments according to not being "None" nor being "self"
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key != 'self'}

        # We can pass everything now to the main function
        return self.data_fromarrays(category='pl', **set_kwargs)



    def if_fromfile(self, filename, objref=None, dataref=None,
                    include_closure_phase=False, include_triple_amplitude=False,
                    include_eff_wave=True,
                    atm=None, ld_func=None, ld_coeffs=None, passband=None,
                    pblum=None, l3=None, bandwidth_smearing=None,
                    bandwidth_subdiv=None,  alb=None,
                    beaming=None, scattering=None, estimate_sigma=True):
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
        :param estimate_sigma: whether to estimate sigmas if not provided
        :type estimate_sigma: bool
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
              :ref:`ifdep <parlabel-phoebe-ifdep>` and
              :ref:`ifobs <parlabel-phoebe-ifobs>`.

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

        Additionally, you can easily remove closure phases, triple amplitudes
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
        ignore = ['self', 'to_dataref', 'from_dataref', 'remove_closure_phase',
                  'remove_triple_amplitude', 'remove_eff_wave']
        set_kwargs = {key:set_kwargs[key] for key in set_kwargs \
                  if set_kwargs[key] is not None and key not in ignore}

        # We can pass everything now to the main function
        out = self.data_fromexisting(to_dataref, from_dataref, category='if', **set_kwargs)

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


    def orb_fromarrays(self, objref=None, dataref=None, time=None, phase=None,
                      x=None, y=None, z=None, vx=None, vy=None, vz=None):
        """
        [FUTURE]

        Create and attach orbit positions to the model.

        A unique data reference (:envvar:`dataref`) is added automatically if
        none is provided by the user. A readable system of unique references is
        applied: the first orb that is added is named ``orb01`` unless
        that reference already exists, in which case ``orb02`` is tried and so
        on. On the other hand, if a :envvar:`dataref` is given at it already
        exists, it's settings are overwritten.

        If no :envvar:`objref` is given, the orb is added to the total
        system, and not to a separate component. 

        Note that you cannot add :envvar:`times` and :envvar:`phases` simultaneously.

        **Example usage**

        It doesn't make much sense to leave the time array empty, since then
        nothing will be computed. Thus, the minimal function call that makes
        sense is something like:

        >>> bundle = phoebe.Bundle()
        >>> bundle.orb_fromarrays(time=np.linspace(0, 10.33, 101))

        or in phase space:

        >>> phase = np.linspace(-0.5, 0.5, 101)
        >>> bundle.orb_fromarrays(phase=phase)

        .. note:: More information

            - For a list of acceptable values for each parameter, see
              :ref:`orbobs <parlabel-phoebe-orbobs>`.
            - In general, :envvar:`time`, :envvar:`x`, :envvar:`y`,
              :envvar:`z`, :envvar:`vx`, :envvar:`vy`, and
              :envvar:`vz` should all be arrays of equal length (unless left to
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
        return self.data_fromarrays(category='orb', **set_kwargs)


    def orb_fromfile(self, filename, objref=None, dataref=None, columns=None,
                      units=None, offset=None, scale=None):
        """
        [FUTURE]

        Add orbit positions from a file.

        The data contained in :envvar:`filename` will be loaded to the object
        with object reference :envvar:`objref` (if ``None``, defaults to the whole
        system), and will have data reference :envvar:`dataref`. If no
        :envvar:`dataref` is is given, a unique one is generated: the first
        orb that is added is named 'orb01', and if that one already
        exists, 'orb02' is tried and so on.

        For any parameter that is not explicitly set (i.e. not left equal to
        ``None``), the defaults from each component in the system are used
        instead of the Phoebe2 defaults.

        **Example usage**

        A plain file can loaded via::

        >>> bundle.orb_fromfile('myfile.orb')

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
        return self.data_fromfile(category='orb', **set_kwargs)


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

        >>> b = phoebe.Bundle()
        >>> b.add_parameter('asini@orbit')
        >>> b['asini']
        9.848077530129958

        Then, you can still change ``sma`` and ``incl``:

        >>> b['sma'] = 20.0
        >>> b['incl'] = 85.0, 'deg'

        and then ``asini`` is updated automatically:

        >>> x['asini']
        19.923893961843316

        However you are not allowed to change the parameter of ``asini``
        manually, because the code does not know if it needs to update ``incl``
        or ``sma`` to keep consistency:

        >>> x['asini'] = 10.0
        ValueError: Cannot change value of parameter 'asini', it is derived from other parameters

        **2. Adding a new parameter to replace an existing one**

        >>> b = phoebe.Bundle()
        >>> b.add_parameter('asini@orbit', replaces='sma')
        >>> b['asini']
        9.848077530129958

        Then, you can change ``asini`` and ``incl``:

        >>> b['asini'] = 10.0
        >>> b['incl'] = 85.0, 'deg'

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
            elif replaces is not False:
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

    def get_etv_datarefs(self, objref=None):
        """
        Return all datarefs of category etv.

        [FUTURE]
        """
        return self.get_datarefs(objref=objref, category='etv')

    def get_sp_datarefs(self, objref=None):
        """
        Return all datarefs of category sp.

        [FUTURE]
        """
        return self.get_datarefs(objref=objref, category='sp')

    def get_syn(self, twig=None):
        """
        Get the synthetic parameterset for an observation

        @param twig: the twig/twiglet used for searching
        @type twig: str
        @return: the observations DataSet
        @rtype: DataSet
        """
        #~ return self._get_by_search(ref=dataref, context='*syn', class_name='*DataSet')
        return self._get_by_search(twig=twig, context='*syn', class_name='*DataSet', method='robust_notfirst')

    def get_dep(self, twig=None):
        """
        Get observations dependables

        @param twig: the twig/twiglet used for searching
        @type twig: str
        @return: the observations dependables ParameterSet
        @rtype: ParameterSet
        """
        #~ return self._get_by_search(ref=dataref, context='*dep', class_name='ParameterSet')
        return self._get_by_search(twig=twig, context='*dep', class_name='ParameterSet', method='robust_notfirst')

    def get_obs(self, twig=None):
        """
        Get observations

        @param twig: the twig/twiglet used for searching
        @type twig: str
        @return: the observations DataSet
        @rtype: DataSet
        """
        #~ return self._get_by_search(ref=dataref, context='*obs', class_name='*DataSet')
        return self._get_by_search(twig=twig, context='*obs', class_name='*DataSet', method='robust_notfirst')
              
    def get_enabled_obs(self, twig=None):
        """
        [FUTURE]
        
        alias for get_enabled_data
        """
        return self.get_enabled_data(twig=twig)

    def get_enabled_data(self, twig=None):
        """
        [FUTURE]
        
        Return the twigs of all datasets (obs) currently enabled
        
        :param twig: the search twig/twiglet (or None to show all)
        :type twig: str
        :return: dictionary of twig/parameter pairs
        :rtype: dict
        """      
        # TODO: should this be per-dataref or per-dataref/label (ie should primary@rv01 and secondary@rv01 be separate items - how does this work for enable/disable)?
        datasets = {}
        for ti in self._get_by_search(twig=twig, kind='ParameterSet', context='*obs', all=True, ignore_errors=True, return_trunk_item=True):
            if ti['item'].enabled:
                datasets[ti['twig']] = ti['item']
        return datasets

    @rebuild_trunk
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

        for ds in self._get_by_search(twig=dataref, context='*obs', class_name='*DataSet', method='robust_notfirst', all=True, ignore_errors=True, return_trunk_item=True):
            ds['item'].set_enabled(enabled)
            logger.info("{} {} '{}'".format('Enabled' if enabled else 'Disabled', ds['context'], ds['twig']))
        

        #~ system = self.get_system()
        #~ try:
        #~     iterate_all_my_bodies = system.walk_bodies()
        #~ except AttributeError:
        #~     iterate_all_my_bodies = [system]

        #~ system = self.get_system()
        #~ try:
            #~ iterate_all_my_bodies = system.walk_bodies()
        #~ except AttributeError:
            #~ iterate_all_my_bodies = [system]
#~ 
        #~ for body in iterate_all_my_bodies:
            #~ this_objref = body.get_label()
            #~ for obstype in body.params['obs']:
                #~ if dataref is None:
                    #~ for idataref in body.params['obs'][obstype]:
                        #~ body.params['obs'][obstype][idataref].set_enabled(enabled)
                        #~ logger.info("{} {} '{}'".format('Enabled' if enabled else 'Disabled', obstype, idataref))
                #~ elif dataref in body.params['obs'][obstype]:
                    #~ body.params['obs'][obstype][dataref].set_enabled(enabled)
                    #~ logger.info("{} {} '{}'".format('Enabled' if enabled else 'Disabled', obstype, dataref))
                    
    def set_enabled(self, twig, value=True):
        """
        Set whether a dataset is enabled (alias of enable_data)
        
        [FUTURE]
        
        :param twig: the search twig
        :type twig: str
        :param value: adjust
        :type value: bool
        """
        return self.enable_data(dataref=twig, enabled=value)

    @rebuild_trunk
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

    def enable_etv(self, dataref=None):
        """
        Enable an ETV dataset so that it will be considered during run_compute

        See :py:func:`Bundle.enable_lc` for more info

        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'etv', True)

    def disable_etv(self, dataref=None):
        """
        Disable an ETV dataset so that it will not be considered during run_compute

        See :py:func:`Bundle.enable_etv` for more info

        @param dataref: reference of the dataset
        @type dataref: str
        """
        self.enable_data(dataref, 'etv', False)


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
                    kind = 'ParameterSet', all = True, ignore_errors = True, method='robust_notfirst')

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
            dss = self._get_by_search(dataref, context='*dep', kind='ParameterSet', all=True, ignore_errors=True, method='robust_notfirst')
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
    def run_compute(self, label=None, objref=None, animate=False, **kwargs):
        """
        Perform calculations to mirror any enabled attached observations.

        Main arguments: :envvar:`label`, :envvar:`objref`, :envvar:`anim`.

        Extra keyword arguments are passed to the
        :ref:`compute <parlabel-phoebe-compute>` ParameterSet.

        **Example usage**

        The minimal setup is::

            >>> b = phoebe.Bundle()
            >>> dataref = b.lc_fromarrays(phase=[0, 0.5, 1.0])
            >>> b.run_compute()

        After which you can plot the results via::

            >>> b.plot_syn(dataref)
            
        If you'd like to run parallelized computations (by time step),
        you must also provide an :envvar:`mpi` parameterSet.
        These should be added to the bundle via :py:func:`add_mpi <phoebe.frontend.container.add_mpi>`,
        and then the label can be set in several places.  When calling run_compute,
        the mpi options will be applied  based on the following priority:
        
        - keyword argument sent to run_compute
        - :envvar:`mpilabel` set in the compute options

        **Keyword 'label'**

        Different compute options can be added via
        :py:func:`Bundle.add_compute() <phoebe.frontend.common.Container.add_compute>`,
        where each of these ParameterSets have a
        :envvar:`label`. If :envvar:`label` is given and that compute option is
        present, those options will be used. If no :envvar:`label` is given, a
        default set of compute options is created on the fly. The used set of
        options is returned but also stored for later reference. You can access
        it via the ``default`` label in the bundle::

            >>> b.run_compute()

        and at any point you can query:

            >>> options = b['default@compute']

        If you want to store new options before hand for later usage you can
        issue:

            >>> b.add_compute(label='no_heating', heating=False)
            >>> options = b.run_compute(label='no_heating')

        **Keyword 'objref'**

        If :envvar:`objref` is given, the computations are only performed on
        that object. This is only advised for introspection. Beware that the
        synthetics will no longer mirror the observations of the entire system,
        but only those of the specified object.

        .. warning::

            1. Even if you only compute the light curve of the secondary in a
               binary system, the system geometry is still determined by the entire
               If you don't want this behaviour, either turn off eclipse computations
               system. Thus, eclipses will occur if the secondary gets eclipsed!
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

            >>> b.run_compute(animate=True)
            >>> b.run_compute(animate='lc')

        .. warning::

            1. Animations are only supported on computers/backends that support the
               animation capabilities of matplotlib (likely excluding Macs).

            2. Animations will not work in interactive mode in ipython (i.e. when
               started as ``ipython --pylab``.

        **Note on keyword arguments**
        
        Any additional keyword arguments sent to run_fitting will *temporarily*
        override values in matching parameters in any of the options used during
        fitting, with qualifiers matched in the following order:
        
        - compute options
        - mpi options
        
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
        system = self.get_system()
        system.fix_mesh()

        obj = self.get_object(objref) if objref is not None else system

        # clear all previous models and create new model
        system.reset_and_clear()

        # get compute options, handling 'default' if label==None
        computeoptions = self.get_compute(label, create_default=True).copy()
        mpilabel = kwargs.pop('mpilabel', computeoptions.get_value('mpilabel') if 'mpilabel' in computeoptions.keys() else None)
        if mpilabel in [None, 'None', '']:
            mpilabel = None
        if mpilabel in [None, 'None', '']:
            mpioptions = None
        else:
            mpioptions = self.get_mpi(mpilabel).copy()

        # now temporarily override with any values passed through kwargs
        for k,v in kwargs.items():
            if k in computeoptions.keys(): # otherwise nonexisting kwargs can be given
                computeoptions.set_value(k,v)
            elif k in mpioptions.keys():
                mpioptions.set_value(k,v)
            else:
                raise ValueError("run_compute does not accept keyword '{}'".format(k))


        # use phoebe2 backend
        if 'time' not in computeoptions.keys() or computeoptions['time'] == 'auto':
            #~ observatory.compute(self.system,mpi=self.mpi if mpi else None,**options)
            if mpioptions is not None and animate:
                raise ValueError("You cannot animate and use MPI simultaneously")
            elif mpioptions is not None:
                obj.compute(mpi=mpioptions, params=computeoptions)
            else:
                obj.compute(animate=animate, params=computeoptions)
        else:
            raise ValueError("time must be set to 'auto' in compute options")
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
            
        return computeoptions

    @rebuild_trunk
    def run_sample(self, label=None, objref=None, sample_from='posterior', samples=10, **kwargs):
        """
        [FUTURE] - and EXPERIMENTAL

        Draw values from parameters that are set for adjustment, compute observables,
        and fill the synthetic datasets with the average and sigma for all
        of these samples.

        Values will be drawn from parameters which are set for adjustment
        and have priors available (see :py:func:`bundle.get_adjustable_parameters`).

        Currently MPI options will be applied per-computation (ie the computations
        are parallelized per-time rather than per-sample).

        Plotting the resulting synthetics are not automatically handled
        by plot_syn, but are by attach_plot_syn.

        >>> bundle.run_sample('preview', samples=20)
        >>> bundle.attach_plot_syn('lc01', figref='fig01')
        >>> bundle.draw('fig01')

        **Extra keyword arguments**

        Any extra keyword arguments are passed on to the ``compute``
        ParameterSet and then the ```mpi``` ParameterSet (if applicable).

        :param label: name of one of the compute ParameterSets stored in bundle
        :type label: str
        :param objref: name of the top-level object used when observing
        :type objref: str
        :param sample_from: whether to sample from priors or posteriors
        :type sample_from: str (one of 'prior', 'posterior')
        :param samples: number of samples to compute
        :type samples: int
        """

        system = self.get_system()
        system.fix_mesh()

        obj = self.get_object(objref) if objref is not None else system

        # get compute options, handling 'default' if label==None
        computeoptions = self.get_compute(label, create_default=True).copy()
        mpilabel = kwargs.pop('mpilabel', computeoptions.get_value('mpilabel'))
        if mpilabel in [None, 'None', '']:
            mpilabel = None
        if mpilabel in [None, 'None', '']:
            mpioptions = None
        else:
            mpioptions = self.get_mpi(mpilabel).copy()

        # now temporarily override with any values passed through kwargs
        for k,v in kwargs.items():
            if k in computeoptions.keys(): # otherwise nonexisting kwargs can be given
                computeoptions.set_value(k,v)
            elif k in mpioptions.keys():
                mpioptions.set_value(k,v)
            else:
                raise ValueError("run_sample does not accept keyword '{}'".format(k))

        # pickle the bundle and computeoptions to send through MPI
        if not mpioptions or not 'directory' in mpioptions or not mpioptions['directory']:
            direc = os.getcwd()
        else:
            direc = mpi['directory']

        bundle_file = tempfile.NamedTemporaryFile(delete=False, dir=direc)
        self.clear_syn()
        pickle.dump(self,bundle_file)
        bundle_file.close()

        compute_file = tempfile.NamedTemporaryFile(delete=False, dir=direc)
        pickle.dump(computeoptions,compute_file)
        compute_file.close()

        # start samples
        if mpioptions is not None:
            # Create arguments to run emceerun_backend.py
            sample_logger_level = 'WARNING'
            objref = self.get_system().get_label() if objref is None else objref
            args = " ".join([bundle_file.name, compute_file.name, objref, sample_from, str(samples), sample_logger_level])

            sample.mpi = True
            flag, mpitype = decorators.construct_mpirun_command(script='sample.py',
                                                      mpirun_par=mpioptions, args=args,
                                                      script_dir='frontend')

            # If something went wrong, we can exit nicely here, the traceback
            # should be printed at the end of the MPI process
            if flag:
                sys.exit(1)

        else:
            sample.mpi = False
            sample.run(bundle_file.name, compute_file.name, objref, sample_from, samples)

        # cleanup temporary files
        os.unlink(bundle_file.name)
        os.unlink(compute_file.name)

        # try loading resulting file
        sample_file = os.path.join(direc, computeoptions['label'] + '.sample.dat')

        if not os.path.isfile(sample_file):
            raise RuntimeError("Could not produce sample file {}, something must have seriously gone wrong during sample run".format(sample_file))

        f = open(sample_file,'r')
        samples = json.load(f)
        f.close()

        for twig, syns in samples['syns'].items():
            # syns is a dictionary with x, xk, y, yk, sigma (optional)

            ds = self.get(twig, hidden=True)
            ds[syns['xk']] = syns['x']
            ds[syns['yk']] = syns['y']

            if 'sigma' in syns.keys():
                ds['sigma'] = syns['sigma']
                # we need to add 'sigma' to columns so that saving the
                # synthetic file will include this column, but also
                # so that clear_syn will clear the sigmas\
                if 'sigma' not in ds['columns']:
                    ds['columns'].append('sigma')

        # now rebuild the trunk so that the faked datasets are summed correctly
        self._build_trunk()

        return samples['hist']

    #}

    #{ Fitting
    @rebuild_trunk
    def run_fitting(self, label='lmfit', add_feedback=True, accept_feedback=True,
                    usercosts=None, **kwargs):
        """
        Run fitting for a given fitting ParameterSet and store the feedback

        **Prerequisites**

        First of all, you need to have *observations* added to your system and
        have at least one of them *enabled*.

        Before you can run any fitting, you need to define *priors* on the
        parameters that you want to include in the probability calculations, and
        set those parameters that you want to include in the normal fitting
        process to be *adjustable*. A limited number of parameters can be
        estimated using a direct (linear) fit. You can mark these by setting
        them to be adjustable, but not define a prior. Thus, there are
        **3 types of parameters**:

        - Parameters you want to vary by virtue of the fitting algorithm. You
          need to define a prior and set them to be adjustable, e.g.::

          >>> b.set_prior('incl', distribution='uniform', lower=80, upper=100)
          >>> b.set_adjust('incl')

        - Parameters you want to estimated using a direct fitting approach. Set
          them to be adjustable, but do not define a prior::

          >>> b.set_adjust('scale@lc01@lcobs')

        - Parameters you want to include in the probability calculations, but
          not fit directly. Only define a prior, but do not mark them for
          adjustment For example suppose you have prior information on the mass
          of the primary component in a binary system::

          >>> b.add_parameter('mass1@orbit')
          >>> b.set_prior('mass1@orbit', distribution='normal', mu=1.2, sigma=0.1)

        .. warning::

            The fitting algorithms are very strict on priors and extreme limits
            on parameters. Before a fitting algorithm is run, a :py:func:`check <phoebe.frontend.bundle.Bundle.check>` is performed to check if all
            values are within the prior limits (attribute :envvar:`prior` of
            a Parameter) and the extreme limits (attributes :envvar:`llim` and
            :envvar:`ulim` of the Parameters). Phoebe2 will notify you if any
            of the checks did not pass. You can adjust any of intervals through
            :py:func:`Parameter.set_prior <phoebe.parameters.parameters.Parameter.set_prior>`
            or :py:func:`Parameter.set_limits <phoebe.parameters.parameters.Parameter.set_limits>`.


        **Setting up fitting and compute options**

        First you need to decide the fitting *context*, i.e. which fitting
        scheme or algorithm you want to use. Every fitting algorithm has
        different options to set, e.g. the number of iterations in an MCMC chain,
        or details on the algorithm that needs to be used. Because every fitting
        algorithm needs to iterate the computations of the system (and evaluate
        it to choose a new set of parameters), it also needs to know the compute
        options (e.g. take reflection effects into account etc.).
        Finally you need to supply a *label* to the fitting options, for easy
        future referencing::

            >>> b.add_fitting(context='fitting:emcee', computelabel='preview',
                                     iters=100, walkers=10, label='my_mcmc')

        You can add more than one fitting option, as long as you don't duplicate
        the labels::

            >>> b.add_fitting(context='fitting:lmfit', computelabel='preview',
                                     method='nelder', label='simplex_method')
            >>> b.add_fitting(context='fitting:lmfit', computelabel='preview',
                                     method='leastsq', label='levenberg-marquardt')


        You can easily print out all the options via::

        >>> print(b['my_mcmc@fitting'])

        **Running the fitter**

        You can run the fitter simply by issuing

        >>> feedback = b.run_fitting(label='my_mcmc')

        When run like this, the results from the fitting procedure will
        automatically be added to system and the best model will be set as the
        current model. You can change that behaviour via the :envvar:`add_feedback`
        and :envvar:`accept_feedback` arguments when calling this function.

        Some fitting algorithms accept an additional :envvar:`mpi` parameterSet.
        These should be added to the bundle via :py:func:`add_mpi <phoebe.frontend.container.add_mpi>`,
        and then the label can be set in several places.  When calling run_fitting,
        the mpi options will be applied  based on the following priority:
        
        - keyword argument sent to run_fitting
        - :envvar:`mpilabel` set in the fitting options
        - :envvar:`mpilabel` set in the compute options used during fitting
        
        The fitter returns a :py:class:`Feedback <phoebe.parameters.feedback.Feedback>`
        object, that contains a summary of the fitting results. You can simply
        print or access the feedback via the Bundle::

        >>> print(feedback)
        >>> print(b['my_mcmc@feedback'])
        
        **Note on keyword arguments**
        
        Any additional keyword arguments sent to run_fitting will *temporarily*
        override values in matching parameters in any of the options used during
        fitting, with qualifiers matched in the following order:
        
        - fitting options
        - compute options
        - mpi options
        
        As an example, sending :envvar:`mpilabel` as a keyword argument, will
        temporarily change the :envvar:`mpilabel` parameter in the fitting
        options, and will therefore parallelize at the fitting/iteration
        level rather than on the compute/time step level.

        **More details on emcee**

        Probably the most general, but also most slowest fitting method is the
        Monte Carlo Markov Chain method. Under the hood, Phoebe2 uses the
        *emcee* Python package to this end. Several options from the *fitting:emcee*
        context are noteworthy here:

        - :envvar:`iters`: number of iterations to run. You can set this to an
          incredibly high number; you can interrupt the chain or assess the
          current state at any point because the MCMC chain is incrementally
          saved to a file. It is recommended to pickle your Bundle right before
          running the fitting algorithm. If you do, you can in a separate script
          or Python terminal monitor the chain like::

          >>> b = phoebe.load('mypickle.pck')
          >>> b.feedback_fromfile('my_mcmc.mcmc_chain.dat')
          >>> b['my_mcmc@feedback'].plot_logp()

          You can also at any point restart a previous (perhaps
          interrupted) chain (see :envvar:`incremental`)
        - :envvar:`walkers`: number of different MCMC chains to run simultaneously.
          The emcee package requires at least 2 times the number of free parameters,
          but recommends much more (as many as feasible).
        - :envvar:`init_from`: the walkers have to start from some point. Either
          the starting points are drawn randomly from the priors (:envvar:`init_from='prior'`),
          from the posteriors (:envvar:`init_from='posterior'`), or from the
          last state of the previous run (:envvar:`init_from='previous_run'`).
          When you draw randomly from priors or posteriors, there is some
          checking performed if all parameters are valid, e.g. if you set the
          morphology to be detached, it checks whether all walkers start from
          such a scenario regardless of the actual priors on the potentials.
        - :envvar:`incremental`: add the results to the previous computation or
          not. You can continue the previous chain *and* resample from the
          posteriors or priors if you wish (via :envvar:`init_from`). Suppose you
          ran a chain like::

          >>> b.add_fitting(context='fitting:emcee', init_from='prior', label='my_mcmc', computelabel='preview')
          >>> feedback = b.run_fitting(fittinglabel='my_mcmc')

          Then, you could just continue these computations via::

          >>> b['incremental@my_mcmc@fitting'] = True
          >>> b['init_from@my_mcmc@fitting'] = 'previous_run'
          >>> feedback = b.run_fitting(label='my_mcmc')

          Alternatively, you can resample multivariate normals from the previous
          posteriors to continue the chain, e.g. after clipping walkers with
          low probability and/or after a burn-in period::

          >>> b['my_mcmc@feedback'].modify_chain(lnproblim=-40, burnin=10)
          >>> b.accept_feedback('my_mcmc')
          >>> b['incremental@my_mcmc@fitting'] = True
          >>> b['init_from@my_mcmc@fitting'] = 'posteriors'
          >>> feedback = b.run_fitting(label='my_mcmc')

          Quality control and convergence monitoring can be done via::

          >>> b['my_mcmc@feedback'].plot_logp()
          >>> b['my_mcmc@feedback'].plot_summary()


        [FUTURE]

        @param label: name of fitting ParameterSet
        @type label: str
        @param add_feedback: flag to store the feedback (retrieve with get_feedback)
        @type add_feedback: bool
        @param accept_feedback: whether to automatically accept the feedback into the system
        @type accept_feedback: bool
        """

         # get fitting params
        fittingoptions = self.get_fitting(label).copy()

        # get compute params
        computelabel = kwargs.pop('computelabel', fittingoptions.get_value('computelabel'))
        computeoptions = self.get_compute(computelabel).copy()

        # Make sure that the fittingoptions refer to the correct computelabel
        fittingoptions['computelabel'] = computelabel

        # get mpi params
        mpilabel = kwargs.pop('mpilabel', None)
        if mpilabel is None and 'mpilabel' in fittingoptions.keys():
            mpilabel = fittingoptions['mpilabel']
        if mpilabel in [None, 'None', ''] and 'mpilabel' in computeoptions.keys():
            mpilabel = computeoptions['mpilabel']
        if mpilabel in [None, 'None', '']:
            mpioptions = None
        else:
            mpioptions = self.get_mpi(mpilabel).copy()

        # Make sure that the fittingoptions refer to the correct mpilabel
        fittingoptions['mpilabel'] = '' if mpilabel is None else mpilabel

        # now temporarily override with any values passed through kwargs
        for k,v in kwargs.items():
            if k in fittingoptions.keys():
                fittingoptions.set_value(k,v)
            elif k in computeoptions.keys():
                computeoptions.set_value(k,v)
            elif mpioptions and k in mpioptions.keys():
                mpioptions.set_value(k,v)
            else:
                raise ValueError("run_fitting does not accept keyword '{}'".format(k))

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
        logger.warning("MPI options:\n{:s}".format(mpioptions))

        # Run the fitting for real
        feedback = fitting.run(self.get_system(), params=computeoptions,
                            fitparams=fittingoptions, mpi=mpioptions,
                            usercosts=usercosts)

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
                "with the twig '{}@feedback'".format(label, label)))

        # Then re-instate the status of the obs without flux/rv/etc..
        # <some code>
        # Accept the feedback: set/reset the variables to their fitted values
        # or their initial values, and in any case recompute the system such
        # that the synthetics are up-to-date with the parameters
        self.accept_feedback(fittingoptions['label']+'@feedback',
                    recompute=True, revert=(not accept_feedback))

        return feedback

    def feedback_fromfile(self, feedback_file, fittinglabel=None,
                          accept_feedback=True, ongoing=False, **kwargs):
        """
        Add fitting feedback from a file.

        Keyword arguments get passed on to the initialization of the feedback class.

        For emcee these can include:
        - lnproblim
        - burnin
        - thin

        [FUTURE]
        """
        if fittinglabel is None:
            fittinglabel = os.path.basename(feedback_file).split('.mcmc_chain.dat')[0]

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
                             fitting=fittingoptions, compute=computeoptions,
                             ongoing=ongoing, **kwargs)

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
                             recompute=accept_feedback, revert=(not accept_feedback))

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

    def write_syn(self, dataref, output_file=None, use_user_units=True,
                  include_obs=True, include_header=True, fmt='%25.18e',
                  delimiter=' ', newline='\n', footer='', simulate=False):
        """
        Write synthetic datasets to an ASCII file.

        By default, this function writes out the model in the same units as the
        data (:envvar:`use_user_units=True`), which are included as well
        (:envvar:`include_obs=True`). It will then also attempt to use
        the same columns as the observations were given in. It is possible to
        override these settings and write out the synthetic data in the internal
        model units of Phoebe. In that case, set :envvar:`use_user_units=False`

        Extra keyword arguments come from ``np.savetxt``, though headers are not
        supported as they are auto-generated to give information on column names
        and units.

        This probably only works for lc and rv right now, and perhaps if.
        The data structure for sp is too complicated (wavelength + time as
        independent variables) to be automatically handled like this. Also
        sed is a bit difficult.

        [FUTURE]

        :param dataref: the dataref of the dataset to write out
        :type dataref: str
        :param output_file: path and filename of the exported file
        :type output_file: str
        """
        # Retrieve synthetics and observations
        this_syn = self.get_syn(dataref)
        this_obs = self.get_obs(dataref)

        # Which columns to write out? First check if we user units are requested
        # and available. If they are not, use the columns from the obs
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

        # Create data
        data = []

        # We might need the passband for some conversions
        try:
            passband = self.get_value_all('passband@{}'.format(dataref)).values()[0]
        except KeyError:
            passband = None

        # Synthetic columns
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

        # Observed columns
        if include_obs:
            data_obs = []
            columns_obs = []
            units_obs = []
            # Collect them but put them in the right units
            for col in this_obs['columns']:
                this_col_data = this_obs[col]
                # Skip if data array length does not match
                if not len(this_col_data) == len(this_syn):
                    continue
                this_col_par = this_obs.get_parameter(col)
                this_col_unit = this_col_par.get_unit() if this_col_par.has_unit() else None
                if this_col_unit and user_columns and col in this_obs['user_units']:
                    unit = this_obs['user_units'][col]
                    if unit != this_col_unit:
                        this_col_data = conversions.convert(this_col_unit, unit,
                                                this_col_data, passband=passband)
                    this_col_unit = unit

                # Remember the names
                data_obs.append(this_col_data)
                columns_obs.append(col+'(OBS)')
                units_obs.append(this_col_unit if this_col_unit else '--')

            # Add them to the previous results
            data += data_obs
            columns += columns_obs
            units += units_obs

        # Create header
        if include_header:
            header = [" ".join(['{:>25s}'.format(col) for col in columns])]
            header+= [" ".join(['{:>25s}'.format(unit) for unit in units])]
            # strip first two spaces from first entry to make space for the '# '
            # character later
            header[0] = header[0][2:]
            header[1] = header[1][2:]
            header = "\n".join(header)
        else:
            header = ''

        if output_file is None:
            output_file = '.'.join([dataref, this_syn.get_context()[:-3]])

        # Write out file
        if not simulate:
            np.savetxt(output_file, np.column_stack(data), header=header,
                   footer=footer, comments='# ', fmt=fmt, delimiter=delimiter,
                   newline=newline)

            # Report to the user
            info = ", ".join(['{} ({})'.format(col, unit) for col, unit in zip(columns, units)])
            logger.info("Wrote columns {} to file '{}'".format(info, output_file))
        else:
            return np.column_stack(data), columns, units, passband


    def write_sedsyn(self, group_name, output_file=None,
                  include_obs=True, include_header=True, fmt='%25.18e',
                  delimiter=' ', newline='\n', footer='', simulate=False):
        """
        Special case for writing SED output to a file.

        User units not possible because that could complicate things: each
        measurement can be given in different units!

        [FUTURE]
        """
        system = self.get_system()
        all_lc_refs = system.get_refs(category='lc')
        all_lc_refs = [ref for ref in all_lc_refs if ref[:len(group_name)] == group_name]

        if output_file is None:
            output_file = '.'.join([group_name,'sed'])

        # Simulate the writing of each file separately, and append the results
        alldata = []
        for ii, dataref in enumerate(all_lc_refs):
            data, cols, units, passband = self.write_syn(dataref, output_file,
                         include_obs=include_obs,
                         include_header=(ii==0), fmt=fmt, delimiter=delimiter,
                         newline=newline, footer=footer, simulate=True)

            # Add extra cols: effective wavelength + bandpass
            eff_wave = passbands.get_response(passband, full_output=True)[2]['WAVLEFF']
            eff_wave = np.ones(data.shape[0])*eff_wave
            data = np.hstack([data, eff_wave[:,None]])
            alldata.append(data)

        if include_header:
            header = [" ".join(['{:>25s}'.format(col) for col in cols+['wavleff']])]
            header+= [" ".join(['{:>25s}'.format(unit) for unit in units+['AA']])]
            # strip first two spaces from first entry to make space for the '# '
            # character later
            header[0] = header[0][2:]
            header[1] = header[1][2:]
            header = "\n".join(header)
        else:
            header = ''

        print np.vstack(alldata).shape
        np.savetxt(output_file, np.vstack(alldata), header=header,
                   footer=footer, comments='# ', fmt=fmt, delimiter=delimiter,
                   newline=newline)

        # Report to the user
        info = ", ".join(['{} ({})'.format(col, unit) for col, unit in zip(cols, units)])
        logger.info("Wrote columns {} to file '{}'".format(info, output_file))


    #}

    #{ Live-Plotting
    
    def plot_obs(self, dataref, time=None, ax=None, **kwargs):
        """
        [FUTURE]
        
        :param dataref: twig that points to the dataset
        :type dataref: str
        :param time: current time (used for uncover, highlight, scroll)
        :type time: float
        :param ax: matplotlib axes (will use plt.gca() by default)
        :type ax: matplotlib.Axes
        :param fmt: matplotlib format (eg 'k-')
        :type fmt: str
        :param uncover: only plot data up to the current time (time must be passed)
        :type uncover: bool
        :param highlight: draw a marker at the current time (time must be passed)
        :type highlight: bool
        :param highlight_fmt: matplotlibformat for time if highlight==True
        :type highlight_fmt: str
        :param highlight_ms: matplotlib markersize for time if highlight==True
        :type highlight_ms: int
        :param scroll: whether to overrid xlim and scroll based on current time (time must be passed)
        :type scroll: bool
        :param scroll_xlim: the xlims to provide relative to the current time if scroll==True (time must be passed)
        :type scroll_xlim: tuple
        
        """
        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = plotting.obs(self, t=time, dataref=dataref, **kwargs)
        
        plot_ps = parameters.ParameterSet(context='plotting:plot')
        axes_ps = parameters.ParameterSet(context='plotting:axes')
        
        plot_ps, axes_ps, dump = self._plotting_set_defaults(func_kwargs_defaults, plot_ps, axes_ps, fig_ps=None, override=False) # should override be True?
        
        ax = plt.gca() if ax is None else ax
        
        return self._call_mpl(ax, mpl_func, mpl_args, mpl_kwargs, plot_ps, axes_ps, time)


    def plot_syn(self, dataref, time=None, ax=None, **kwargs):
        """
        [FUTURE]
        
        :param dataref: twig that points to the dataset
        :type dataref: str
        :param time: current time (used for uncover, highlight, scroll)
        :type time: float
        :param ax: matplotlib axes (will use plt.gca() by default)
        :type ax: matplotlib.Axes
        :param fmt: matplotlib format (eg 'k-')
        :type fmt: str
        :param uncover: only plot data up to the current time (time must be passed)
        :type uncover: bool
        :param highlight: draw a marker at the current time (time must be passed)
        :type highlight: bool
        :param highlight_fmt: matplotlibformat for time if highlight==True
        :type highlight_fmt: str
        :param highlight_ms: matplotlib markersize for time if highlight==True
        :type highlight_ms: int
        :param scroll: whether to overrid xlim and scroll based on current time (time must be passed)
        :type scroll: bool
        :param scroll_xlim: the xlims to provide relative to the current time if scroll==True (time must be passed)
        :type scroll_xlim: tuple
        """
        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = plotting.syn(self, t=time, dataref=dataref, **kwargs)
        
        plot_ps = parameters.ParameterSet(context='plotting:plot')
        axes_ps = parameters.ParameterSet(context='plotting:axes')
        
        plot_ps, axes_ps, dump = self._plotting_set_defaults(func_kwargs_defaults, plot_ps, axes_ps, fig_ps=None, override=False) # should override be True?
        
        ax = plt.gca() if ax is None else ax
        
        return self._call_mpl(ax, mpl_func, mpl_args, mpl_kwargs, plot_ps, axes_ps, time)

    def plot_mesh(self, dataref, time=None, ax=None, **kwargs):
        """
        [FUTURE]
        
        NOTE: If you set projection to be 3d, ax or plt.gca() must be a 3d axes.  You can add a 3d axes with ax=plt.gcf().add_subplot(projection='3d')
        
        :param dataref: twig that points to the dataset to get values
        :type dataref: str
        :param time: time to use to compute the mesh
        :type time: float
        :param ax: matplotlib axes (will use plt.gca() by default)
        :type ax: matplotlib.Axes
        :param objref: twig that points to the object to plot (defaults to entire system)
        :type objref: str
        :param select: key in the mesh to use for color (ie 'rv', 'teff') or an array with same length/size as the mesh
        :type select: str or np.array
        :param cmap: colormap to use (must be a valid matplotlib colormap).  If not provided, defaults will be used based on 'select'
        :type cmap: str or pylab.cm instance
        :param vmin: lower limit for the select array used for the colormap, np.nan for auto
        :type vmin: float or np.nan
        :param vmax: upper limit for the select array used for the colormap, np.nan for auto
        :type vmax: float or np.nan
        :param projection: '2d' or '3d' projection.  Must use 3d axes to use 3d projection.
        :type projection: str ('2d' or '3d')
        :param zlim: limits used on zaxis (only used if projection=='3d')
        :type zlim: tuple
        :param zunit: unit to plot on the zaxis (only used if projection=='3d')
        :type zunit: str
        :param zlabel: label on the zaxis (only used if projection=='3d') 
        :type zlabel: str
        :param azim: azimuthal orientation (only used if projection=='3d')
        :type azim: float
        :param elev: elevation orientation (only used if projection=='3d')
        :type elev: float
        """
        
        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = plotting.mesh(self, t=time, dataref=dataref, **kwargs)
        
        plot_ps = parameters.ParameterSet(context='plotting:plot')
        axes_ps = parameters.ParameterSet(context='plotting:axes')
        
        plot_ps, axes_ps, dump = self._plotting_set_defaults(func_kwargs_defaults, plot_ps, axes_ps, fig_ps=None, override=True)
        
        ax = plt.gca() if ax is None else ax

        return self._call_mpl(ax, mpl_func, mpl_args, mpl_kwargs, plot_ps, axes_ps, time)


    def plot_orbit(self, objref, time=None, ax=None, **kwargs):
        """
        [FUTURE]
        
        
        NOTE: If you set projection to be 3d, ax or plt.gca() must be a 3d axes.  You can add a 3d axes with ax=plt.gcf().add_subplot(projection='3d')

        :param objref: twig that points to the object to plot
        :type objref: str
        :param time: current time (not list of times)
        :type time: float
        :param ax: matplotlib axes (will use plt.gca() by default)
        :type ax: matplotlib.Axes
        :param fmt: matplotlib format (eg 'k-')
        :type fmt: str
        :param highlight: draw a marker at the current time (time must be passed during draw call)
        :type highlight: bool
        :param highlight_fmt: matplotlibformat for time if highlight==True
        :type highlight_fmt: str
        :param highlight_ms: matplotlib markersize for time if highlight==True
        :type highlight_ms: int
        :param projection: '2d' or '3d' projection.  Must use 3d axes to use 3d projection.
        :type projection: str ('2d' or '3d')
        :param zlim: limits used on zaxis (only used if projection=='3d')
        :type zlim: tuple
        :param zunit: unit to plot on the zaxis (only used if projection=='3d')
        :type zunit: str
        :param zlabel: label on the zaxis (only used if projection=='3d') 
        :type zlabel: str
        :param azim: azimuthal orientation (only used if projection=='3d')
        :type azim: float
        :param elev: elevation orientation (only used if projection=='3d')
        :type elev: float
        """
        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = plotting.orbit(self, t=time, objref=objref, **kwargs)
        
        plot_ps = parameters.ParameterSet(context='plotting:plot')
        axes_ps = parameters.ParameterSet(context='plotting:axes')
        
        plot_ps, axes_ps, dump = self._plotting_set_defaults(func_kwargs_defaults, plot_ps, axes_ps, fig_ps=None, override=False) # should override be True?
        
        ax = plt.gca() if ax is None else ax
        
        return self._call_mpl(ax, mpl_func, mpl_args, mpl_kwargs, plot_ps, axes_ps, time)


    def old_plot_obs(self, twig=None, **kwargs):
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
                                   return_trunk_item=True, all=True)

        # It's possible that we need to plot an SED: in that case, we have
        # more than one dataset but they are all light curves that are grouped
        if len(dsti) > 1:
            # Check if they are all lightcurves and collect group name
            groups = []
            for idsti in dsti:
                correct_category = idsti['item'].get_context()[:-3] == 'lc'
                is_grouped = 'group' in idsti['item']
                if not correct_category or not is_grouped:
                    # raise the correct error:
                    self._get_by_search(twig, context='*obs', class_name='*DataSet',
                                   return_trunk_item=True)
                else:
                    groups.append(idsti['item']['group'])
            # check if everything belongs to the same group
            if not len(set(groups)) == 1:
                raise KeyError("More than one SED group found matching twig '{}', please be more specific".format(twig))

            obj = self.get_object(dsti[0]['label'])
            context = 'sed' + dsti[0]['item'].get_context()[-3:]
            ds = dict(ref=groups[0])
        else:
            dsti = dsti[0]
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
        output = getattr(beplotting, 'plot_{}'.format(context))(obj, **kwargs)

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


    def old_plot_syn(self, twig=None, *args, **kwargs):
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

        >>> b.plot_syn('lc01', 'r-', lw=2) # first light curve added via 'data_fromarrays'
        >>> b.plot_syn('rv01@primary', 'r-', lw=2, scale=None)

        >>> b.plot_syn('if01', 'k-') # first interferometry added via 'data_fromarrays'
        >>> b.plot_syn('if01', 'k-', y='vis2') # first interferometry added via 'data_fromarrays'

        More information on arguments and keyword arguments:

        - :py:func:`phoebe.backend.plotting.plot_lcsyn`
        - :py:func:`phoebe.backend.plotting.plot_rvsyn`

        @param twig: the twig/twiglet to use when searching
        @type twig: str
        """
        # Retrieve the obs DataSet and the object it belongs to
        dsti = self._get_by_search(twig, context='*syn', class_name='*DataSet',
                                   return_trunk_item=True, all=True)

        # It's possible that we need to plot an SED: in that case, we have
        # more than one dataset but they are all light curves that are grouped
        if len(dsti) > 1:
            # retrieve obs for sed grouping.
            dstiobs = self._get_by_search(twig, context='*obs', class_name='*DataSet',
                                   return_trunk_item=True, all=True)

            # Check if they are all lightcurves and collect group name
            groups = []
            if not len(dsti) == len(dstiobs):
                raise ValueError("Cannot plot synthetics of SED '{}'".format(twig))

            for idsti, jdsti in zip(dsti, dstiobs):
                correct_category = idsti['item'].get_context()[:-3] == 'lc'
                is_grouped = 'group' in jdsti['item']
                if not correct_category or not is_grouped:
                    # raise the correct error:
                    self._get_by_search(twig, context='*syn', class_name='*DataSet',
                                   return_trunk_item=True)
                else:
                    groups.append(jdsti['item']['group'])
            # check if everything belongs to the same group
            if not len(set(groups)) == 1:
                raise KeyError("More than one SED group found matching twig '{}', please be more specific".format(twig))

            obj = self.get_object(dsti[0]['label'])
            context = 'sed' + dsti[0]['item'].get_context()[-3:]
            ds = dict(ref=groups[0])
        else:
            dsti = dsti[0]
            ds = dsti['item']
            obj = self.get_object(dsti['label'])
            context = ds.get_context()

        # For the pl context we just use the sp context
        if context[:-3] == 'pl':
            context = 'sp' + context[-3:]
            kwargs['category'] = 'pl'

        # Do we need automatic/custom xlabel, ylabel and/or title? We need to
        # pop the kwargs here because they cannot be passed to the lower level
        # plotting function
        xlabel = kwargs.pop('xlabel', '_auto_')
        ylabel = kwargs.pop('ylabel', '_auto_')
        title = kwargs.pop('title', '_auto_')

        # Now pass everything to the correct plotting function
        kwargs['ref'] = ds['ref']
        try:
            output = getattr(beplotting, 'plot_{}'.format(context))(obj, *args, **kwargs)
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

    def old_plot_residuals(self, twig=None, **kwargs):
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
            output = getattr(beplotting, 'plot_{}res'.format(category))(obj, **kwargs)
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

    def old_plot_mesh(self, objref=None, label=None, dataref=None, time=None, phase=None,
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

            >>> b.plot_mesh(time=0.12)
            >>> b.plot_mesh(phase=0.25, select='teff')
            >>> b.plot_mesh(phase=0.25, objref='secondary', dataref='lc01')

        You can use this function to plot the mesh of the entire system, or just
        one component (eclipses will show in projected light!). Any scalar
        quantity that is present in the mesh (effective temperature, logg ...)
        can be used as color values. For some of these quantities, there are
        smart defaults for the color maps.

        **Plotting the current status of the mesh after running computations**

        If you've just ran :py:func:`Bundle.run_compute` for some observations,
        and you want to see what the mesh looks like after the last calculations
        have been performed, then this is what you need to do::

            >>> b.plot_mesh()
            >>> b.plot_mesh(objref='primary')
            >>> b.plot_mesh(select='teff')
            >>> b.plot_mesh(dataref='lc01')

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
                                pl=category=='pl', ifm=category=='if',
                                save_result=False, **options)

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



    def plot_prior(self, twig=None, **kwargs):
        """
        Plot a prior.

        [FUTURE]
        """
        prior = self.get_prior(twig)
        prior.plot(**kwargs)


    #{ Plots
    def _plotting_set_defaults(self, kwargs, plot_ps, axes_ps, fig_ps=None, override=False):
        """
        [FUTURE]

        helper function that takes a input dictionary (kwargs) and sets those
        as defaults across the different plotting PSs with priority being given
        in the following order:

        1. plot_ps
        2. axes_ps
        3. fig_ps
        4. new parameter in plot_ps
        """
        for k,v in kwargs.items():
            value = v['value'] if isinstance(v,dict) else v
            
            # try plot_ps first, then axes_ps and fig_ps
            if k in plot_ps.keys():
                if plot_ps.get_value(k)=='_auto_' or override:
                    plot_ps.set_value(k,value)
            elif k in axes_ps.keys():
                if k not in ['plotrefs']:
                    if axes_ps.get_value(k)=='_auto_' or override:
                        axes_ps.set_value(k,value)
            elif fig_ps is not None and k in fig_ps.keys():
                if k not in ['axesrefs','axeslocs','axessharex','axessharey']:
                    if fig_ps.get_value(k)=='_auto_' or override:
                        fig_ps.set_value(k,value)
            elif k in ['label']:
                #~ raise KeyError("parameter with qualifier {} forbidden".format(k))
                pass
            else:
                if (isinstance(v,dict) and v.get('cast_type',False)=='float') or isinstance(v, float):
                    _cast_type = float
                    _repr = '%f'
                elif (isinstance(v,dict) and v.get('cast_type',False)=='int') or isinstance(v, int):
                    _cast_type = int
                    _repr = '%d'
                elif (isinstance(v,dict) and v.get('cast_type',False) in ['bool', 'make_bool']) or isinstance(v, bool):
                    _cast_type = 'make_bool'
                    _repr = ''
                elif (isinstance(v,dict) and v.get('cast_type',False) in ['list', 'return_string_or_list']) or isinstance(v, list) or isinstance(v, tuple):
                    _cast_type = 'return_string_or_list'
                    _repr = '%s'
                elif (isinstance(v,dict) and v.get('cast_type',False)=='choose' and v.get('choices',False)):
                    _cast_type = 'choose'
                    _repr = '%s'
                else:
                    _cast_type = str
                    _repr = '%s'

                new_parameter = parameters.Parameter(qualifier=k,
                        description = v['description'] if isinstance(v,dict) and 'description' in v.keys() else 'func added this parameter but failed to provide a description',
                        repr = _repr,
                        cast_type = _cast_type,
                        value = value,
                        write_protected = v['write_protected'] if isinstance(v,dict) and 'write_protected' in v.keys() else False)

                if _cast_type == 'choose':
                    new_parameter.choices = v.get('choices')


                ps = v.get('ps', 'plot') if isinstance(v,dict) else 'plot'
                if ps == 'figure':
                    fig_ps.add(new_parameter, with_qualifier=k)
                elif ps == 'axes':
                    axes_ps.add(new_parameter, with_qualifier=k)
                else:
                    plot_ps.add(new_parameter, with_qualifier=k)

        return plot_ps, axes_ps, fig_ps

    def _run_plot_process_func(self, func_str, time=None, func_kwargs={}):
        if func_str[:19] == 'marshaled function:':
            func_str = func_str[19:]
            func_code = marshal.loads(func_str)
            func = types.FunctionType(func_code, globals(), "_tmp_plot_func")

        else:
            func = getattr(plotting, func_str)

        # run the function with the args
        #func_args_list = [func_args] if isinstance(func_args, str) else list(func_args) if func_args is not None else []
        func_args = [self, time]
        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = func(*func_args, **func_kwargs)
        return mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults

    def attach_plot(self, twigs=None, func=None, plotref=None, axesref=None, figref=None, axesloc=None, **kwargs):
        """
        [FUTURE]

        This is the most generic of functions that allows you to attach a
        plotting "recipe" to the bundle.  By doing so, you will simply be
        allowed to re-draw figures or plots after changing your model.

        The calling signature of this function may seem odd - but gives this
        function its power and generality.  Several wrapping functions exist
        which allow you to do the most common plotting tasks through more
        specialized calling signatures:

        :py:func:`Bundle.attach_plot_obs`
        :py:func:`Bundle.attach_plot_syn`
        :py:func:`Bundle.attach_plot_mesh`

        At the most basic level, this function stores the arguments to
        send to either an matplotlib function (that must be an attribute
        of ax) or a preprocessing function defined by func.

        Below is a list of simple examples that all accomplish plotting
        an lcobs and lcsyn (with the dataref lc01):

        >>> from phoebe.frontend import plotting

        >>> b.attach_plot('lc01', func=plotting.obs)
        >>> b.attach_plot('lc01', func=plotting.syn)
        or
        >>> b.attach_plot('lc01@lcobs')
        >>> b.attach_plot('lc01@lcsyn')
        or
        >>> b.attach_plot(('time@lc01@lcobs', 'flux@lc01@lcobs'))
        >>> b.attach_plot(('time@lc01@lcsyn', 'flux@lc01@lcsyn'))

        Here you'll notice that the first argument is either a single twig,
        or a list of twigs.  If func is not defined, we try to determine
        which processing function should be used based on the context of the
        twig(s).  Either way, these twig(s) are passed as args to the func,
        which then returns the matplotlib function, args, and defaults.  These
        values are then stored in the plot, axes, and figure ParameterSets
        for recreation.

        To draw a plot you must call draw, draw_plot, draw_axes, or draw_figure:

        >>> import matplotlib.pyplot as plt
        >>> b.attach_plot('lc01@lcobs', fmt='k.', figref='myfig')
        >>> b.draw('myfig')
        >>> plt.show()

        If you need something more flexible than the functions provided
        in phoebe.frontend.plotting, your function must follow the format:

        def my_processing_func(bundle, time, *args, **kwargs):
            # insert magic here
            return mpl_func_name, args_for_mpl_func, kwargs_for_mpl_func, dictionary_of_defaults

        where the input args are provided by twigs from attach_plot
        and kwargs are any other keyword arguments sent through attach_plot.
        Using the bundle that is passed as the first argument and these args (which
        must be strings, preferably twigs), you can do whatever you'd like
        so long as your return three items: the string of the matplotlib function to
        call (must be an attribute of ax not plt), the arguments to send to
        that function, the kwargs to send to that function, and a dictionary of default kwargs.
        These default kwargs will be sent to the following places in this order:

            1. plot parameterset (existing parameters)
            2. axes parameterset
            3. figure parameterset
            4. plot parameterset (new parameters)

        Note that kwargs_for_mpl_func above will directly be sent to the function.
        If these are options that you would like to change later through the parameterset,
        you should instead send through dictionary_of_defaults and they should find their
        way in the plot parameterset.

        :param twigs: twigs that points to a dataset or array 
        :type twigs: list of twigs (strings), or single twig (string), or None
        :param func: a function that processes args and return (x,y) or (x,y,z)
        :type func: function
        :param plotref: the reference to assign to the stored plot parameterset
        :type plotref: str
        :param axesref: the reference of a new or existing axes parameterset
        :type axesref: str
        :param figref: the reference of a new or existing figure parameterset
        :type figref: str
        :param axesloc: the matplotlib subplot location to add the axes to the figure
        :type axesloc: list of 3 ints
        """
        # handle the correct type for twigs since we'll be passing it as *args
        if twigs is None:
            twigs = []
        elif not isinstance(twigs, str):
            twigs = list(twigs) # in case it was a tuple

        # determine plotref if not provided
        if plotref is None:
            plotref = "plot01"
            existing_plotrefs = [pi['ref'] for pi in self.sections['plot']]
            i=1
            while plotref in existing_plotrefs:
                i+=1
                plotref = "plot{:02}".format(i)

        # let's try to guess the correct func to use if necessary
        # and also set twigs args to the correct kwargs
        if isinstance(twigs, str):
            # the following should raise an error if not a unique match
            if func in ['syn', plotting.syn]:
                context = '*syn'
            elif func in ['obs', plotting.obs, 'residuals', plotting.residuals]:
                context = '*obs'
            else:
                context = None
            
            dsti = self._get_by_search(twigs, class_name='*DataSet', context=context,
                            return_trunk_item=True, all=False, method='robust_notfirst')

            # update args to be full twig
            twigs = '{}@{}'.format(dsti['label'], dsti['ref'])   # TODO: check that this is correct
            

            # get context for guessing func
            context = dsti['context']

            typ = context[-3:]

            if func is None:
                if typ=='obs':
                    func = 'obs'
                elif typ=='syn':
                    func = 'syn'
                else:
                    raise ValueError("could not determine function to parse args")
                
            # update the kwargs
            if 'dataref' not in kwargs.keys():
                kwargs['dataref'] = twigs

        elif len(twigs):
            # then we assume we have args=(x,y) or (x,y,z), so let's try to guess
            # from the context of y

            # let's update all twigs to full version
            for i,t in enumerate(twigs):
                ti = self._get_by_search(t, return_trunk_item=True, all=False)
                twigs[i] = ti['twig']

            if len(twigs)==2:
                func = 'xy' if func is None else func
                for i,k in enumerate(['x','y']):
                    if k not in kwargs.keys():
                        kwargs[k] = twigs[i]
            elif len(twigs)==3:
                func = 'xyz' if func is None else func
                for i,k in enumerate(['x','y','z']):
                    if k not in kwargs.keys():
                        kwargs[k] = twigs[i]            
            else:
                raise ValueError("could not determine function to parse args")

        # func is now either passed by the user or guessed automatically above
        func_str = func if isinstance(func, str) else func.func_name

        if func_str not in dir(plotting):
            # we need to store as a str in the PS, so let's marshal
            func_str = "marshaled function:"+marshal.dumps(func.func_code)

        # now let's run func to get the default_kwargs to add to plot_ref, etc
        # and also to make sure that it runs before allowing to attach the PS
        time = kwargs.pop('time', None)
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)

        # we'll set up the axes based on kwargs - if they aren't provided
        # and remain at auto, then we won't split into a new axes_ps anyways

        sharex = kwargs.pop('sharex','_auto_')
        sharey = kwargs.pop('sharex','_auto_')
        xunit = kwargs.get('xunit','_auto_')
        yunit = kwargs.get('yunit','_auto_')

        # at this point, all of kwargs passed by the user have either been
        # popped before the previous call, or processed, so we should be
        # using func_kwargs_defaults from now on

        figref = self._handle_current_figure(figref=figref)
        axesref, figref = self._handle_current_axes(axesref=axesref, figref=figref,
                axesloc=axesloc, sharex=sharex, sharey=sharey, xunit=xunit, yunit=yunit)

        axes_ps = self.get_axes(axesref)
        fig_ps = self.get_figure(figref)

        # if axes_ps already existed then we want to pass some of those arguments
        # on as well so they aren't reset
        # and we even need some (like xunit, yunit) from axes_ps
        for k in ['phased', 'xlim', 'ylim', 'xunit', 'yunit', 'zunit', 'projection', 'background', 'scroll', 'scroll_xlim']:
            if k in axes_ps.keys() and (type(axes_ps[k])==bool or '_auto_' not in axes_ps[k]):
                kwargs.setdefault(k, axes_ps[k])

        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = self._run_plot_process_func(func_str, time, func_kwargs=kwargs)

        # create plot_ps and retrieve existing axes_ps and fig_ps
        plot_ps = parameters.ParameterSet(context='plotting:plot', func=func_str, ref=plotref)


        # we don't need mpl_func, mpl_args, or mpl_kwargs - they'll be regenerated
        # by the processing function when draw is called, but we *do* want
        # to store func_kwargs_defaults in the plot_ps, axes_ps, or fig_ps
        plot_ps, axes_ps, fig_ps = self._plotting_set_defaults(func_kwargs_defaults, plot_ps, axes_ps, fig_ps, override=True)

        logger.info("Attaching {}@plot to {}@axes and {}@figure".format(plotref, axes_ps['ref'], fig_ps['ref']))
        self._add_to_section('plot', plot_ps)

        axes_ps.add_plot(plotref)

        return plotref, axesref, figref

    def attach_plot_obs(self, dataref, **kwargs):
        """
        [FUTURE]

        see :py:func:`Bundle.attach_plot`
        
        specific (optional) keyword arguments for attach_plot_obs include:
        
        :param dataref: twig that points to the dataset
        :type dataref: str
        :param fmt: matplotlib format (eg 'k-')
        :type fmt: str
        :param uncover: only plot data up to the current time (time must be passed during draw call)
        :type uncover: bool
        :param highlight: draw a marker at the current time (time must be passed during draw call)
        :type highlight: bool
        :param highlight_fmt: matplotlibformat for time if highlight==True
        :type highlight_fmt: str
        :param highlight_ms: matplotlib markersize for time if highlight==True
        :type highlight_ms: int
        :param scroll: whether to overrid xlim and scroll based on current time (time must be passed during draw call)
        :type scroll: bool
        :param scroll_xlim: the xlims to provide relative to the current time if scroll==True (time must be passed during the draw call)
        :type scroll_xlim: tuple
        """
        return self.attach_plot(func=plotting.obs, dataref=dataref, **kwargs)

    def attach_plot_syn(self, dataref, **kwargs):
        """
        [FUTURE]

        see :py:func:`Bundle.attach_plot`

        specific (optional) keyword arguments for attach_plot_syn include:
        
        :param dataref: twig that points to the dataset
        :type dataref: str
        :param fmt: matplotlib format (eg 'k-')
        :type fmt: str
        :param uncover: only plot data up to the current time (time must be passed during draw call)
        :type uncover: bool
        :param highlight: draw a marker at the current time (time must be passed during draw call)
        :type highlight: bool
        :param highlight_fmt: matplotlibformat for time if highlight==True
        :type highlight_fmt: str
        :param highlight_ms: matplotlib markersize for time if highlight==True
        :type highlight_ms: int
        :param scroll: whether to overrid xlim and scroll based on current time (time must be passed during draw call)
        :type scroll: bool
        :param scroll_xlim: the xlims to provide relative to the current time if scroll==True (time must be passed during the draw call)
        :type scroll_xlim: tuple
        """
        return self.attach_plot(func=plotting.syn, dataref=dataref, **kwargs)

    def attach_plot_mesh(self, objref=None, dataref=None, **kwargs):
        """
        [FUTURE]
        
        see :py:func:`Bundle.attach_plot`
        
        specific (optional) keyword arguments for attach_plot_mesh include:
        
        :param dataref: twig that points to the dataset to get values
        :type dataref: str
        :param objref: twig that points to the object to plot (defaults to entire system)
        :type objref: str
        :param select: key in the mesh to use for color (ie 'rv', 'teff') or an array with same length/size as the mesh
        :type select: str or np.array
        :param cmap: colormap to use (must be a valid matplotlib colormap).  If not provided, defaults will be used based on 'select'
        :type cmap: str or pylab.cm instance
        :param vmin: lower limit for the select array used for the colormap, np.nan for auto
        :type vmin: float or np.nan
        :param vmax: upper limit for the select array used for the colormap, np.nan for auto
        :type vmax: float or np.nan
        :param projection: '2d' or '3d' projection
        :type projection: str ('2d' or '3d')
        :param zlim: limits used on zaxis (only used if projection=='3d')
        :type zlim: tuple
        :param zunit: unit to plot on the zaxis (only used if projection=='3d')
        :type zunit: str
        :param zlabel: label on the zaxis (only used if projection=='3d') 
        :type zlabel: str
        :param azim: azimuthal orientation (only used if projection=='3d')
        :type azim: float
        :param elev: elevation orientation (only used if projection=='3d')
        :type elev: float
        """
        return self.attach_plot(func=plotting.mesh, objref=objref, dataref=dataref, **kwargs)

    def attach_plot_orbit(self, objref=None, **kwargs):
        """
        [FUTURE]
        
        see :py:func:`Bundle.attach_plot`
        
        specific (optional) keyword arguments for attach_plot_orbit include:
        
        :param objref: twig that points to the object to plot (defaults to entire system)
        :type objref: str
        :param fmt: matplotlib format (eg 'k-')
        :type fmt: str
        :param highlight: draw a marker at the current time (time must be passed during draw call)
        :type highlight: bool
        :param highlight_fmt: matplotlibformat for time if highlight==True
        :type highlight_fmt: str
        :param highlight_ms: matplotlib markersize for time if highlight==True
        :type highlight_ms: int
        :param projection: '2d' or '3d' projection
        :type projection: str ('2d' or '3d')
        :param zlim: limits used on zaxis (only used if projection=='3d')
        :type zlim: tuple
        :param zunit: unit to plot on the zaxis (only used if projection=='3d')
        :type zunit: str
        :param zlabel: label on the zaxis (only used if projection=='3d') 
        :type zlabel: str
        :param azim: azimuthal orientation (only used if projection=='3d')
        :type azim: float
        :param elev: elevation orientation (only used if projection=='3d')
        :type elev: float
        """
        # TODO: add dataref as kwarg above that points to a orbsyn
        return self.attach_plot(func=plotting.orbit, objref=objref, **kwargs)

    def attach_plot_mplcmd(self, funcname, args=(), **kwargs):
        """
        [FUTURE]
        
        see :py:func:`Bundle.attach_plot`
        
        specific (optional) keyword arguments for attach_plot_mplcmd include:
        """
        #TODO: update this to take kwargs instead of args
        return self.attach_plot(func=plotting.mplcommand, mplfunc=funcname, mplargs=args, **kwargs)

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
        # TODO allow passing kwargs and return a copy with those changes?
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

    def add_plot(self, plotref=None, axesref=None, figref=None, axesloc=None, **kwargs):
        """
        [FUTURE]

        Add an existing plot (created through attach_plot, attach_funcplot, attach_mplcmd etc)
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
        :param axesloc: location in the figure to attach the axes (see matplotlib.figure.Figure.add_subplot)
        :type axesloc: tuple with length 3
        :return: (axesref, figref)
        :rtype: tuple of strings
        """

        figref = self._handle_current_figure(figref)
        axesref, figref = self._handle_current_axes(axesref, figref, axesloc=axesloc)

        axes_ps = self.get_axes(axesref)
        axes_ps.add_plot(plotref)
        plot_ps = self.get_plot(plotref)

        for k,v in kwargs.items():
            if k in axes_ps.keys():
                axes_ps.set_value(k,v)
            else:
                plot_ps.set_value(k,v)

        return plotref, axesref, figref


    def draw_plot(self, plotref, time=None, axesref=None, ax=None, **kwargs):
        """
        [FUTURE]

        """
        logger.info('Drawing {}'.format(plotref))
        if ax is None:
            ax = plt.gca()

        # we'll get a copy of the plot_ps and axes_ps since we may be making temporary changes
        plot_ps = self.get_plot(plotref).copy()
        axes_ps = self.get_axes(axesref).copy()
        axesref = axes_ps['ref']

        # this plot needs to be attached as a member of the axes if it is not
        #~ if plotref not in axes_ps['plotrefs']:
            #~ axes_ps.add_plot(plotref)

        # retrieve the function from the plot PS
        func_str = plot_ps.get_value('func')
        #func_args = plot_ps.get_value('twigs')

        # and now we'll overwrite (temporarily) from any user-sent
        # kwargs.  These trump any values set in the PS or defaults
        # set by the function
        plot_ps, axes_ps, dump = self._plotting_set_defaults(kwargs, plot_ps, axes_ps, fig_ps=None, override=True)

        func_kwargs = {}

        # now from the plot_ps we need to build kwargs to pass to the processing func
        for k,v in plot_ps.items():
            if v not in ['', '_auto_']:
                func_kwargs[k] = v
        # and we even need some (like xunit, yunit) from axes_ps
        for k in ['phased', 'xunit', 'yunit', 'zunit', 'projection', 'background']:
            if k not in axes_ps.keys():
                continue
            v = axes_ps[k]
            if k not in kwargs and v not in ['', '_auto_']:
                # note: here we're overriding func_kwargs but not user-sent kwargs
                func_kwargs[k] = v

        # and remove items that we don't want to pass
        dump = func_kwargs.pop('func', None)
        #dump = func_kwargs.pop('twigs', None)

        # the dataref becomes mpl's label (used for legends)
        func_kwargs['label'] = func_kwargs.pop('ref')

        mpl_func, mpl_args, mpl_kwargs, func_kwargs_defaults = self._run_plot_process_func(func_str, time, func_kwargs)

        # func_kwargs_defaults may also apply to axes_ps, so let's update the PSs anyways
        plot_ps, axes_ps, dump = self._plotting_set_defaults(func_kwargs_defaults, plot_ps, axes_ps, fig_ps=None, override=False) # should override be True?

        ax = self._call_mpl(ax, mpl_func, mpl_args, mpl_kwargs, plot_ps, axes_ps, time)
        
        # TODO: fix this return statement
        return {plotref: []}
        
        
    def _call_mpl(self, ax, mpl_func, mpl_args, mpl_kwargs, plot_ps, axes_ps, time=None):
        
        # and now let's finally call the matplotlib function(s)
        if isinstance(mpl_func, str):
            # then we just have one call, but for generality let's list everything
            mpl_func = [mpl_func]
            mpl_args = [mpl_args]
            mpl_kwargs = [mpl_kwargs]

        for m_func, m_args, m_kwargs in zip(mpl_func, mpl_args, mpl_kwargs):
            # check to make sure all kwargs in mpl_kwargs are allowed

            if hasattr(ax, m_func):
                line2D_attrs = ['aa', 'agg', 'alpha', 'animated', 'antialiased', 'axes', 'c', 'children', 'clip', 'clip', 'clip', 'color', 'contains', 'dash', 'dash', 'data', 'drawstyle', 'figure', 'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'ls', 'lw', 'marker', 'markeredgecolor', 'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt', 'markersize', 'markevery', 'mec', 'mew', 'mfc', 'mfcalt', 'ms', 'path', 'picker', 'pickradius', 'rasterized', 'snap', 'solid', 'solid', 'transform', 'transformed', 'url', 'visible', 'window', 'xdata', 'xydata', 'ydata', 'zorder']
                line2D_attrs += ['verticalalignment', 'horizontalalignment']
                line2D_attrs.append('fmt')
                m_kwargs = {k:v for k,v in m_kwargs.items() if k in line2D_attrs}
                if m_func == 'plot' and 'fmt' in m_kwargs:
                    # manually move fmt from kwargs to args
                    m_args = list(m_args) # cannot append to tuple
                    m_args.append(m_kwargs.pop('fmt'))
                output = getattr(ax, m_func)(*m_args, **m_kwargs)
                
            elif hasattr(collections, m_func):
                #~ print "*", m_func, m_args, m_kwargs

                #background = m_kwargs.pop('background', 'k')
                zs = m_kwargs.pop('zs', None)

                p = getattr(collections, m_func)(*m_args, **m_kwargs)

                ax.add_collection(p)

                #ax.set_axis_bgcolor(background)
                ax.set_aspect('equal')
                ax.set_xlim(-10,10)  # TODO: this should be applied by func_kwargs_defaults
                ax.set_ylim(-10,10)  # TODO: this should be applied by func_kwargs_defaults

            elif hasattr(art3d, m_func):

                #background = m_kwargs.pop('background', 'k')

                p = getattr(art3d, m_func)(*m_args, **m_kwargs)

                ax.add_collection(p)

                #ax.set_axis_bgcolor(background)
                ax.set_aspect('equal')
                ax.set_xlim(-10,10)  # TODO: this should be applied by func_kwargs_defaults
                ax.set_ylim(-10,10)  # TODO: this should be applied by func_kwargs_defaults
                ax.set_zlim(-10,10)  # TODO: this should be applied by func_kwargs_defaults

            else:
                logger.error("could not call mpl function: {}".format(m_func))

        # now we need to make any necessary changes to the axes
        # TODO: make this more automated by checking getattr(ax, 'set_'+key)?
        # TODO: we also need to check getattr when setting new values in the PSs (in _plotting_set_defaults)
        # TODO: this gets really repetitive - can we move to plot_axes?
        if axes_ps.get_value('xlabel') != '_auto_':
            ax.set_xlabel(axes_ps.get_value('xlabel'))
        if axes_ps.get_value('ylabel') != '_auto_':
            ax.set_ylabel(axes_ps.get_value('ylabel'))
        if axes_ps.get_value('title') != '_auto_':
            ax.set_title(axes_ps.get_value('title'))

        if axes_ps.get_value('xlim') not in [(None,None), '_auto_', u'_auto_']:
            ax.set_xlim(axes_ps.get_value('xlim'))
        if axes_ps.get_value('ylim') not in [(None,None), '_auto_', u'_auto_']:
            ax.set_ylim(axes_ps.get_value('ylim'))
            
        if axes_ps.get('aspect', None)=='equal':
            ax.set_aspect('equal')

        # override any xlimits if the axes_ps scroll is set and we were passed a time
        if 'scroll' in axes_ps.keys() and axes_ps.get_value('scroll') and time:
            sxlim = axes_ps.get_value('scroll_xlim')
            ax.set_xlim(time+sxlim[0], time+sxlim[1])


        # handle z-things if this has a zaxes (projection=='3d')
        if hasattr(ax, 'zaxis'):
            ax.view_init(elev=axes_ps.get_value('elev'), azim=axes_ps.get_value('azim'))

            if not ax.zaxis_inverted():
                # We need to flip the zaxis to make this left-handed to match the
                # convention of -z and -vz pointing towards the observer.
                # Unfortunately, this seems to sometimes lose the tick and ticklabels
                ax.invert_zaxis()

            if axes_ps.get_value('zlabel') != '_auto_':
                ax.set_zlabel(axes_ps.get_value('zlabel'))

            if axes_ps.get_value('zlim') not in [(None,None), '_auto_', u'_auto_']:
                ax.set_zlim3d(axes_ps.get_value('zlim'))

                
        return ax


    def add_axes_to_figure(self, figref, axesref, axesloc):
        """
        [FUTURE]
        """
        # TODO: handle new figref
        fig = self.get_figure(figref)

        axesrefs = fig.get_value('axesrefs')
        axeslocs = fig.get_value('axeslocs')

        axesrefs.append(axesref)
        axeslocs.append(axesloc)

        fig.set_value('axesrefs', axesrefs)
        fig.set_value('axeslocs', axeslocs)


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

    def add_axes(self, axesref=None, figref=None, axesloc=(1,1,1), sharex='_auto_', sharey='_auto_', **kwargs):
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
            axes_ps = plotting.Axes(self, ref=axesref, **kwargs)
        else:
            add_to_section = False
            axes_ps = self.get_axes(axesref)

        for k,v in kwargs.items():
            axes_ps.set_value(k,v)

        if add_to_section:
            self._add_to_section('axes',axes_ps)

        # now attach it to a figure
        figref = self._handle_current_figure(figref=figref)
        if axesloc in [None, '_auto_']: # just in case (this is necessary and used from defaults in other functions)
            axesloc = (1,1,1)
        self.get_figure(figref).add_axes(axesref, axesloc, sharex, sharey) # calls rebuild trunk and will also set self.current_figure

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

    def draw_axes(self, axesref=None, time=None, ax=None, **kwargs):
        """
        [FUTURE]

        Draw an axes that is attached to the bundle.

        If :envvar:`axesref` is None, then the current axes will be plotted.

        If you don't provide :envvar:`ax` then the figure will be drawn to plt.gca().
        Note: you must provide an axes with the correct projection (for mesh and orbit plots).
        If you want this handled for you, use draw_figure instead and the axes will automatically
        be created with the correct projections.

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

        # right now all axes options are being set for /each/ draw_plot
        # call, which will work, but is overkill.  The reason for this is
        # so that draw_plot(plot_label) is smart enough to handle also setting
        # the axes options for the one single call

        plot_ret = {}
        for plotref in axes_ps['plotrefs']:
            plot_ret_new = self.draw_plot(plotref, time=time, axesref=axesref, ax=ax, **kwargs)

            for k,v in plot_ret_new.items():
                plot_ret[k] = v

        self.current_axes = None
        #~ self.current_figure = None
        self.currently_plotted = []

        return ({axesref: ax}, plot_ret)

    def _handle_current_axes(self, axesref=None, figref=None, axesloc=None, sharex='_auto_', sharey='_auto_', xunit='_auto_', yunit='_auto_'):
        """
        [FUTURE]
        this function should be called whenever the user calls a plotting function

        it will check to see if the current axes is defined, and if it isn't it will
        create a new axes instance with an intelligent default label

        the created plotting parameter set should then be added later by self.get_axes(label).add_plot(plot_twig)
        """

        fig_ps = self.get_figure(figref)

        if self.current_axes is not None:
            curr_axes_ps = self.get_axes(self.current_axes)
            curr_axes_loc = fig_ps.get_loc(self.current_axes)
        else:
            curr_axes_ps = None
            curr_axes_loc = None

        if not self.current_axes \
                or (axesref is not None and axesref not in self._get_dict_of_section('axes')) \
                or (axesloc is not None and axesloc not in fig_ps.get_value('axeslocs')):
            # no existing figure
            # or new axesref for this figure
            # or new loc for this figure (note: cannot retrieve by loc)
            #~ print "*** _handle_current_axes calling add_axes", axesref, loc, sharex, sharey
            axesref, figref = self.add_axes(axesref=axesref, figref=figref, axesloc=axesloc, sharex=sharex, sharey=sharey) # other stuff can be handled externally

        # handle units
        elif axesref is None and axesloc in [None, curr_axes_loc]:
            # we do not allow axesref==self.current_axes in this case, because then we'd be overriding the axesref

            # TODO: we need to be smarter here if any of the units are _auto_ - they may be referring to
            # different datasets with different default units, so we need to check to see what _auto_
            # actually refers to and see if that matches... :/
            add_axes = True
            if (xunit != curr_axes_ps.get_value('xunit') and xunit != '_auto_') and (yunit != curr_axes_ps.get_value('yunit') and yunit != '_auto_'):
                #~ print "*** _handle_current_axes units caseA"
                sharex = '_auto_'
                sharey = '_auto_'
                axesloc = curr_axes_loc
                # axes locations (moving to top & right) is handled in draw_figure
            elif xunit != curr_axes_ps.get_value('xunit') and xunit != '_auto_':
                #~ print "*** _handle_current_axes units caseB"
                sharex = '_auto_'
                sharey = self.current_axes
                axesloc = curr_axes_loc
                # axes locations (moving to top) is handled in draw_figure
            elif yunit != curr_axes_ps.get_value('yunit') and yunit != '_auto_':
                #~ print "*** _handle_current_axes units caseC"
                sharex = self.current_axes
                sharey = '_auto_'
                axesloc = curr_axes_loc
                # axes locations (moving to right) is handled in draw_figure
            else:
                #~ print "*** _handle_current_axes units caseD"
                #~ sharex = '_auto_'
                #~ sharey = '_auto_'
                #~ loc = loc
                add_axes = False

            if add_axes:
                #~ print "*** _handle_current_axes calling add_axes", axesref, loc, sharex, sharey
                axesref, figref = self.add_axes(axesref=axesref, figref=figref, axesloc=axesloc, xunit=xunit, yunit=yunit, sharex=sharex, sharey=sharey)



        if axesref is not None:
            self.current_axes = axesref

        return self.current_axes, self.current_figure

    #}

    #{ Figures

    def draw(self, twig=None, time=None, fname=None, **kwargs):
        """
        [FUTURE]

        Draw a figure, axes, or plot that is attached to the bundle.

        If :envvar:`twig` is None, then the current figure will be plotted.

        If you don't provide :envvar:`fig` or :envvar:`ax`, then the the
        figure/axes will be drawn to plt.gcf() or plt.gca()

        If you provide :envvar:`fig` but :envvar:`twig` points to a plot or
        axes instead of a figure, then the axes will be drawn to fig.gca()

        :param twig: twig pointing the the figure, axes, or plot options
        :type twig: str or None
        :param time: time to use during all plotting call (single time or list)
        :type time: float or list of floats
        :param ax: mpl axes used for plotting (optional, overrides :envvar:`fig`)
        :type ax: mpl.Axes
        :param fig: mpl figure used for plotting (optional)
        :type fig: mpl.Figure
        :param clf: whether to call plt.clf() before plotting
        :type clf: bool
        :param cla: whether to call plt.cal() before plotting
        :type cla: bool
        :param tight_layout: whether to call plt.gcf().tight_layout() after plotting
        :type tight_layout: bool
        :param fname: filename to save figure, or None to return mpl object
        :type fname: str or None
        :param fps: frames per second to use when creating animation (must pass length of times and a valid filename type)
        :type fps: int
        """

        fps = int(kwargs.pop('fps', 20))
        if isinstance(time, parameters.Parameter):
            time = time.get_value()

        if not isinstance(time, float) and hasattr(time, '__iter__'):
            tight_layout = kwargs.pop('tight_layout', False)
            for i,t in enumerate(time):
                logger.info("drawing at time: %f", t)
                ret = self.draw(twig=twig, time=t, fname='_anim_tmp_{:08d}.png'.format(i) if fname is not None else None, tight_layout=tight_layout and i==0, **kwargs)
                plt.clf()
            
            if fname is not None:
                sleep(5) # just to make sure the last frame finished rendering
                plotlib.make_movie('_anim_tmp_*.png', fps=fps, output=fname, cleanup=True)
                
                return fname
            
            return ret

        if twig is None:
            ps = self.get_figure()
        else:
            ps = self._get_by_search(twig, section=['plot','axes','figure'])

        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        clf = kwargs.pop('clf', False)
        cla = kwargs.pop('cla', False)
        tight_layout = kwargs.pop('tight_layout', False)

        ref = ps.get_value('ref')
        level = ps.context.split(':')[1].split('_')[0] # will be plot, axes, or figure

        if level=='figure':
            if fig is None:
                fig = plt.gcf()
            if clf:
                fig.clf()

            ret = self.draw_figure(ref, time=time, fig=fig, **kwargs)

        elif level=='axes':
            if ax is None:
                if fig is None:
                    ax = plt.gca()
                else:
                    ax = fig.gca()
            if cla:
                ax.cla()

            ret = self.draw_axes(ref, time=time, ax=ax, **kwargs)

        else:
            if ax is None:
                if fig is None:
                    ax = plt.gca()
                else:
                    ax = fig.gca()
            if cla:
                ax.cla()

            ret = self.draw_plot(ref, time=time, ax=ax, **kwargs)

        if tight_layout:
            plt.gcf().tight_layout()
            
        if fname is not None:
            plt.gcf().savefig(fname)
            ret = fname

        return ret

    def draw_figure(self, figref=None, time=None, fig=None, **kwargs):
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

        fig_ps = self.get_figure(figref)

        axes_ret, plot_ret = {}, {}
        for (axesref,axesloc,axessharex,axessharey) in zip(fig_ps['axesrefs'], fig_ps['axeslocs'], fig_ps['axessharex'], fig_ps['axessharey']):
            # need to handle logic for live-plotting
            axes_ps = self.get_axes(axesref)
            plotrefs = axes_ps['plotrefs']

            axes_existing_refs = [a.split('@')[0] for a in axes_ret.keys()]
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

            projection = axes_ps.get('projection', None)
            if projection=='2d':
                projection = None
            ax = fig.add_subplot(axesloc[0],axesloc[1],axesloc[2],sharex=sharex,sharey=sharey,projection=projection)
            # we must set the current axes so that subsequent live-plotting calls will
            # go here unless overriden by the user
            plt.sca(ax)

            axes_ret_new, plot_ret_new = self.draw_axes(axesref, time=time, ax=ax)

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

        self.current_figure = None
        self.current_axes = None
        self.currently_plotted = []

        return ({figref: fig}, axes_ret, plot_ret)

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
        fig = plotting.Figure(self, ref=figref, **kwargs)

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
        Save the bundle class to a file as a pickle
        
        In the future, this method will hopefully write to a json-formatted ascii file.

        @param filename: path to save the bundle
        @type filename: str
        """
        self.save_pickle(filename)
        #~ self._save_json(filename)

    def save_pickle(self,filename=None,save_usersettings=False):
        """
        Save a class to an file.

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

        # remove user settings
        if not save_usersettings:
            settings = self.usersettings
            self.usersettings = None

        trunk = self.trunk
        self._purge_trunk()

        # pickle
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()
        logger.info('Saved bundle to file {} (pickle)'.format(filename))

        # reset user settings
        if not save_usersettings:
            self.usersettings = settings

        self.trunk = trunk

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
             'plotting:selector', 'point_source', 'pssyn', 'root',\
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
