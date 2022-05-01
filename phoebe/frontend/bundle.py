import sys
import subprocess
import os

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

import re
import json
import atexit
import time
from datetime import datetime
from distutils.version import StrictVersion
from copy import deepcopy as _deepcopy
import pickle as _pickle
from inspect import getsource as _getsource

from scipy.optimize import curve_fit as cfit


# PHOEBE
# ParameterSet, Parameter, FloatParameter, send_if_client, etc
from phoebe.parameters import *
from phoebe.parameters import hierarchy as _hierarchy
from phoebe.parameters import system as _system
from phoebe.parameters import component as _component
from phoebe.parameters import setting as _setting
from phoebe.parameters import dataset as _dataset
from phoebe.parameters import compute as _compute
from phoebe.parameters import solver as _solver
from phoebe.parameters import constraint as _constraint
from phoebe.parameters import feature as _feature
from phoebe.parameters import figure as _figure
from phoebe.parameters.parameters import _uniqueid, _clientid, _return_ps, _extract_index_from_string, _corner_twig, _corner_label
from phoebe.backend import backends, mesh
from phoebe.backend import universe as _universe
from phoebe.solverbackends import solverbackends as _solverbackends
from phoebe.distortions import roche
from phoebe.frontend import io
from phoebe.atmospheres.passbands import list_installed_passbands, list_online_passbands, get_passband, update_passband, _timestamp_to_dt
from phoebe.dependencies import distl as _distl
from phoebe.utils import _bytes, parse_json, _get_masked_times, _get_masked_compute_times
from phoebe import helpers as _helpers
import libphoebe

from phoebe import u
from phoebe import conf, mpi
from phoebe import __version__

import logging
logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())

from io import IOBase

_bundle_cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_bundles'))+'/'

_skip_filter_checks = {'check_default': False, 'check_visible': False}

# Attempt imports for client requirements
try:
    """
    requirements for client mode:
    pip install "python-socketio[client]"
    """

    import requests
    from urllib.request import urlopen as _urlopen
    from urllib.error import URLError

    import socketio # https://python-socketio.readthedocs.io/en/latest/client.html

except ImportError:
    _can_client = False
else:
    _can_client = True

try:
    import celerite as _celerite
except ImportError:
    logger.warning("celerite not installed: only required for gaussian processes")
    _use_celerite = False
else:
    _use_celerite = True


def _get_add_func(mod, func, return_none_if_not_found=False):
    if isinstance(func, str) and "." in func:
        # allow recursive submodule access
        # example: mod=solver, func='samplers.emcee'
        if hasattr(mod, func.split('.')[0]):
            return _get_add_func(getattr(mod, func.split('.')[0]), ".".join(func.split('.')[1:]), return_none_if_not_found=return_none_if_not_found)
        else:
            func = None

    if isinstance(func, str) and hasattr(mod, func):
        func = getattr(mod, func)



    if hasattr(func, '__call__'):
        return func
    elif return_none_if_not_found:
        return None
    else:
        raise ValueError("could not find callable function in {}.{}"
                         .format(mod, func))

def _is_equiv_array_or_float(value1, value2):
    if hasattr(value1, '__iter__'):
        return np.all(value1==value2)
    else:
        return value1==value2


class RunChecksItem(object):
    def __init__(self, b, message, param_uniqueids=[], fail=True, affects_methods=[]):
        self._b = b
        self._message = message
        self._fail = fail
        self._affects_methods = [affects_methods] if isinstance(affects_methods, str) else affects_methods
        self._param_uniqueids = [uid.uniqueid if isinstance(uid, Parameter) else uid for uid in param_uniqueids]

    def __repr__(self):
        return "<RunChecksItem level={} message={} affects_methods={} parameters: {}>".format(self.level, self.message, ",".join(self.affects_methods), len(self._param_uniqueids))

    def __str__(self):
        n_affected_parameters = len(self._param_uniqueids)
        return "{}: {} ({} affected parameter{}, affecting {})".format(self.level, self.message, n_affected_parameters, "s" if n_affected_parameters else "", ",".join(self.affects_methods))

    def to_dict(self):
        """
        Return a dictionary representation of the RunChecksItem.

        See also:
        * <phoebe.frontend.bundle.RunChecksItem.message>
        * <phoebe.frontend.bundle.RunChecksItem.fail>
        * <phoebe.frontend.bundle.RunChecksItem.level>
        * <phoebe.frontend.bundle.RunChecksItem.parameters>

        Returns
        ----------
        * (dict) with keys: 'message', 'level', 'fail', 'parameters'.
        """
        return dict(message=self.message,
                    level=self.level,
                    fail=self.fail,
                    parameters={p.twig: p.uniqueid for p in self.parameters.to_list()})

    @property
    def message(self):
        """
        Access the message of the warning/error.

        See also:
        * <phoebe.frontend.bundle.RunChecksItem.to_dict>

        Returns
        ---------
        * (str) the warning/error message.
        """
        return self._message

    @property
    def fail(self):
        """
        Access whether this item will cause the checks to fail (is an error
        instead of a warning).

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.passed>
        * <phoebe.frontend.bundle.RunChecksItem.level>
        * <phoebe.frontend.bundle.RunChecksItem.affects_methods>
        * <phoebe.frontend.bundle.RunChecksItem.to_dict>

        Returns
        ---------
        * (bool) whether this item is an error that will cause the checks to fail.
        """
        return self._fail

    @property
    def affects_methods(self):
        """
        Access which bundle methods the warning/error will affect.  If <phoebe.frontend.bundle.RunChecksItem.fail>
        is True, this will then raise an error when attempting to call that method.

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.fail>

        """
        return self._affects_methods

    @property
    def level(self):
        """
        Access whether this item is an 'ERROR' or 'WARNING'.

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.passed>
        * <phoebe.frontend.bundle.RunChecksItem.fail>
        * <phoebe.frontend.bundle.RunChecksItem.to_dict>

        Returns
        ---------
        * (str) either "ERROR" or "WARNING"
        """
        return "ERROR" if self.fail else "WARNING"

    @property
    def parameters(self):
        """
        Access the parameters that are suggested by the warning/error to address
        the underlying issue.

        See also:
        * <phoebe.frontend.bundle.RunChecksItem.to_dict>

        Returns
        ----------
        * <phoebe.parameters.ParameterSet> of parameters
        """
        return self._b.filter(uniqueid=self._param_uniqueids,
                              check_visible=False,
                              check_default=False,
                              check_advanced=False,
                              check_single=False)




class RunChecksReport(object):
    def __init__(self, items=[]):
        # need to force a copy here otherwise we'll soft copy against previous
        # instances and get duplicates
        self._items = items[:]

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return "<RunChecksReport {} items: status={}>".format(len(self.items), self.status)

    def __str__(self):
        """String representation for the ParameterSet."""
        return "Run Checks Report: {}\n".format(self.status) + "\n".join([str(i) for i in self.items])

    @property
    def passed(self):
        """
        Return whether the checks are passing (as opposed to failing).

        Note that warnings items will not be included when determining if the
        checks fail.

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.status>
        * <phoebe.frontend.bundle.RunChecksItem.fail>
        * <phoebe.frontend.bundle.RunChecksItem.level>
        * <phoebe.frontend.bundle.RunChecksReport.get_items>

        Returns
        ----------
        * (bool) whether any failing items are included in the report.
        """
        return len(self.get_items(fail=True))==0

    @property
    def status(self):
        """
        Return whether the report results in a status of 'PASS', 'WARNING',
        or 'FAIL'.

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.passed>

        Returns
        ------------
        * (str) either 'PASS', 'WARNING', or 'FAIL'
        """
        return "PASS" if len(self.items)==0 else "WARNING" if self.passed else "FAIL"

    @property
    def items(self):
        """
        Access the underlying <phoebe.frontend.bundle.RunChecksItem> items.

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.get_items>

        Returns
        ---------
        * (list) list of <phoebe.frontend.bundle.RunChecksItem> objects.
        """
        return self._items

    def add_item(self, b, message, param_uniqueids=[], fail=True, affects_methods=[]):
        """
        Add a new <phoebe.frontend.bundle.RunChecksItem> to this report.
        Generally this should not be done manually, but is handled internally
        by <phoebe.frontend.bundle.Bundle.run_checks>,
        <phoebe.frontend.bundle.Bundle.run_checks_compute>,
        <phoebe.frontend.bundle.Bundle.run_checks_solver>, or
        <phoebe.frontend.bundle.Bundle.run_checks_solution>.

        Arguments
        -----------
        * `b` (Bundle): the <phoebe.frontend.bundle.Bundle> object
        * `message` (string): the message of the new item.  See
            <phoebe.frontend.bundle.RunChecksItem.message>.
        * `param_uniqueids` (list): list of uniqueids of parameters.
            See <phoebe.frontend.bundle.RunChecksItem.parameters>.
        * `fail` (bool, optional, default=True): whether the item should cause
            the report to have a failing status.  See
            <phoebe.frontend.bundle.RunChecksItem.fail> and
            <phoebe.frontend.bundle.RunChecksReport.status>.
        * `affects_methods` (list or string, optional, default=[]): which
            bundle methods (run_compute, run_solver, adopt_solution) the
            warning/error will affect
        """
        self._items.append(RunChecksItem(b, message, param_uniqueids, fail, affects_methods))

    def get_items(self, fail=None):
        """
        Access the underlying <phoebe.frontend.bundle.RunChecksItem> items,
        with optional ability to filter by the `fail` argument of each item.

        See also:
        * <phoebe.frontend.bundle.RunChecksReport.items>

        Arguments
        ----------
        * `fail` (bool or None, optional, default=None): filter for items
            with a particular value for `fail` or None to return all.
            See <phoebe.frontend.bundle.RunChecksItem.fail>.

        Returns
        ---------
        * (list) list of <phoebe.frontend.bundle.RunChecksItem> objects.
        """
        if fail is None:
            return self.items

        return [i for i in self.items if i.fail==fail]


class Bundle(ParameterSet):
    """Main container class for PHOEBE 2.

    The `Bundle` is the main object in PHOEBE 2 which is used to store
    and filter all available system parameters as well as handling attaching
    datasets, running models, and accessing synthetic data.

    The Bundle is simply a glorified <phoebe.parameters.ParameterSet>. In fact,
    filtering on a Bundle gives you a ParameterSet (and filtering on a
    ParameterSet gives you another ParameterSet).  The only difference is that
    most "actions" are only available at the Bundle level (as they need to access /all/
    parameters).

    Make sure to also see the documentation and methods for
    * <phoebe.parameters.ParameterSet>
    * <phoebe.parameters.Parameter>
    * <phoebe.parameters.FloatParameter>
    * <phoebe.parameters.parameters.FloatArrayParameter>

    To initialize a new bundle, see:
    * <phoebe.parameters.ParameterSet.open>
    * <phoebe.frontend.bundle.Bundle.from_legacy>
    * <phoebe.frontend.bundle.Bundle.default_binary>
    * <phoebe.frontend.bundle.Bundle.default_star>

    To save or export a bundle, see:
    * <phoebe.frontend.bundle.Bundle.save>
    * <phoebe.frontend.bundle.Bundle.export_legacy>

    To filter parameters and set values, see:
    * <phoebe.parameters.ParameterSet.filter>
    * <phoebe.parameters.ParameterSet.get_value>
    * <phoebe.parameters.ParameterSet.set_value>

    To deal with datasets, see:
    * <phoebe.frontend.bundle.Bundle.add_dataset>
    * <phoebe.frontend.bundle.Bundle.get_dataset>
    * <phoebe.frontend.bundle.Bundle.rename_dataset>
    * <phoebe.frontend.bundle.Bundle.remove_dataset>
    * <phoebe.frontend.bundle.Bundle.enable_dataset>
    * <phoebe.frontend.bundle.Bundle.disable_dataset>

    To compute forward models, see:
    * <phoebe.frontend.bundle.Bundle.add_compute>
    * <phoebe.frontend.bundle.Bundle.get_compute>
    * <phoebe.frontend.bundle.Bundle.rename_compute>
    * <phoebe.frontend.bundle.Bundle.remove_compute>
    * <phoebe.frontend.bundle.Bundle.run_compute>
    * <phoebe.frontend.bundle.Bundle.get_model>
    * <phoebe.frontend.bundle.Bundle.rename_model>
    * <phoebe.frontend.bundle.Bundle.remove_model>

    To deal with figures and plotting, see:
    * <phoebe.parameters.ParameterSet.plot>
    * <phoebe.frontend.bundle.Bundle.add_figure>
    * <phoebe.frontend.bundle.Bundle.get_figure>
    * <phoebe.frontend.bundle.Bundle.rename_figure>
    * <phoebe.frontend.bundle.Bundle.remove_figure>
    * <phoebe.frontend.bundle.Bundle.run_figure>

    To run solver backends, see:
    * <phoebe.frontend.bundle.Bundle.add_solver>
    * <phoebe.frontend.bundle.Bundle.get_solver>
    * <phoebe.frontend.bundle.Bundle.rename_solver>
    * <phoebe.frontend.bundle.Bundle.remove_solver>
    * <phoebe.frontend.bundle.Bundle.run_solver>
    * <phoebe.frontend.bundle.Bundle.get_solution>
    * <phoebe.frontend.bundle.Bundle.rename_solution>
    * <phoebe.frontend.bundle.Bundle.remove_solution>

    """

    def __init__(self, params=None, check_version=False):
        """Initialize a new Bundle.

        Initializing a new bundle without a constructor is possible, but not
        advised.  It is suggested that you use one of the constructors below.

        Available constructors:
        * <phoebe.frontend.bundle.Bundle.open>
        * <phoebe.frontend.bundle.Bundle.from_legacy>
        * <phoebe.frontend.bundle.Bundle.default_binary>
        * <phoebe.frontend.bundle.Bundle.default_star>

        Arguments
        ---------
        * `params` (list, optional): list of <phoebe.parameters.Parameter>
            objects to create the Bundle


        Returns
        --------
        * an instantiated Bundle object
        """
        # for some reason I do not understand at all, defaulting params=[] will
        # fail for successive inits.  So instead we'll default to None and then
        # switch to an empty array here.
        if params is None:
            params = []

        self._params = []

        # set to be not a client by default
        self._is_client = False
        self._client_allow_disconnect = False
        self._waiting_on_server = False
        self._server_changes = None
        self._server_secret = None  # if not None, will attempt to send kill signal atexit and as_client=False
        self._server_clients = []

        self._within_solver = False

        super(Bundle, self).__init__(params=params)


        # flags for handling functionality not available to files imported from
        # older version of PHOEBE.
        self._import_before_v211 = False

        # since this is a subclass of PS, some things try to access the bundle
        # by self._bundle, in this case we just need to fake that to refer to
        # self
        self._bundle = self
        self._hierarchy_param = None

        self._af_figure = None

        # handle delayed constraints when interactive mode is off
        self._delayed_constraints = []
        self._failed_constraints = []

        if not len(params):
            # add position (only 1 allowed and required)
            params, constraints = _system.system()
            self._attach_params(params, context='system')
            for constraint in constraints:
                self.add_constraint(*constraint)

            # add default settings (only 1 allowed and required)
            self._attach_params(_setting.settings(), context='setting')

            # set a blank hierarchy to start
            self.set_hierarchy(_hierarchy.blank)

            # add necessary figure options
            self._attach_params(_figure._new_bundle(), context='figure')

        else:
            for param in self._params:
                param._bundle = self

            try:
                self._hierarchy_param = self.get_parameter(qualifier='hierarchy', context='system')
            except ValueError:
                # possibly this is a bundle without a hierarchy
                self.set_hierarchy(_hierarchy.blank)

        # if loading something with constraints, we need to update the
        # bookkeeping so the parameters are aware of how they're constrained
        for constraint in self.filter(context='constraint', check_visible=False, check_default=False).to_list():
            constraint._update_bookkeeping()

        self._mplcolorcyclers = {k: _figure.MPLPropCycler('color', _figure._mplcolors[1:] if k in ['component', 'model'] else _figure._mplcolors) for k in ['default', 'component', 'dataset', 'model']}
        self._mplmarkercyclers = {k: _figure.MPLPropCycler('marker', _figure._mplmarkers) for k in ['default', 'component', 'dataset', 'model']}
        self._mpllinestylecyclers = {k: _figure.MPLPropCycler('linestyle', _figure._mpllinestyles) for k in ['default', 'component', 'dataset', 'model']}

    @classmethod
    def open(cls, filename, import_from_older=True, import_from_newer=False):
        """
        For convenience, this function is available at the top-level as
        <phoebe.open> or <phoebe.load> as well as
        <phoebe.frontend.bundle.Bundle.open>.

        Open a new bundle.

        Open a bundle from a JSON-formatted PHOEBE 2 file.
        This is a constructor so should be called as:

        ```py
        b = Bundle.open('test.phoebe')
        ```

        If opening a bundle from an older version of PHOEBE, this will attempt
        to make any necessary migrations.  Enable a logger at 'warning' (or higher)
        level to see messages regarding these migrations.  To enable a logger,
        see <phoebe.logger>.

        See also:
        * <phoebe.parameters.ParameterSet.open>
        * <phoebe.parameters.Parameter.open>

        Arguments
        ----------
        * `filename` (string or file object): relative or full path to the file
            or an opened python file object.  Alternatively, pass a list of
            parameter dictionaries to be loaded directly (use carefully).
        * `import_from_older` (bool, optional, default=True): whether to allow
            importing bundles that were created with an older minor relase
            of PHOEBE into the current version.  If True, enable the logger
            (at warning level or higher) to see messages.  If False, an error will
            be raised.  Generally, this should be a safe import operation as we
            try to handle migrating previous versions.
        * `import_from_newer` (bool, optional, default=False): whether to allow
            importing bundles that were created with a newer minor release
            of PHOEBE into the current installed version.  If True, enable the
            logger (at warning level or higher) to see messages.  If False, an
            error will be raised.  This is off by default as we cannot guarantee
            support with future changes to the code.

        Returns
        ---------
        * an instantiated <phoebe.frontend.bundle.Bundle> object

        Raises
        ---------
        * RuntimeError: if the version of the imported file fails to load according
            to `import_from_older` or `import_from_newer`.
        """
        def _ps_dict(ps, include_constrained=True):
            return {p.qualifier: p.get_quantity() if hasattr(p, 'get_quantity') else p.get_value() for p in ps.to_list() if (include_constrained or not p.is_constraint)}

        if io._is_file(filename):
            f = filename
        elif isinstance(filename, str):
            filename = os.path.expanduser(filename)
            logger.debug("importing from {}".format(filename))
            f = open(filename, 'r')
        elif isinstance(filename, list):
            # we'll handle later
            pass
        else:
            raise TypeError("filename must be string or file object, got {}".format(type(filename)))

        if isinstance(filename, list):
            data = filename
        else:
            data = json.load(f, object_pairs_hook=parse_json)
            f.close()

        b = cls(data)

        version = b.get_value(qualifier='phoebe_version', check_default=False, check_visible=False)
        phoebe_version_import = StrictVersion(version if version != 'devel' else '2.3.0')
        phoebe_version_this = StrictVersion(__version__ if __version__ != 'devel' else '2.3.0')

        logger.debug("importing from PHOEBE v {} into v {}".format(phoebe_version_import, phoebe_version_this))

        # update the entry in the PS, so if this is saved again it will have the new version
        b.set_value(qualifier='phoebe_version', value=__version__, check_default=False, check_visible=False, ignore_readonly=True)

        if phoebe_version_import == phoebe_version_this:
            return b
        elif phoebe_version_import > phoebe_version_this:
            if not import_from_newer:
                raise RuntimeError("The file/bundle is from a newer version of PHOEBE ({}) than installed ({}).  Consider updating or attempt importing by passing import_from_newer=True.".format(phoebe_version_import, phoebe_version_this))
            warning = "importing from a newer version ({}) of PHOEBE, this may or may not work, consider updating".format(phoebe_version_import)
            print("WARNING: {}".format(warning))
            logger.warning(warning)
            return b
        elif not import_from_older:
            raise RuntimeError("The file/bundle is from an older version of PHOEBE ({}) than installed ({}). Attempt importing by passing import_from_older=True.".format(phoebe_version_import, phoebe_version_this))

        if phoebe_version_import < StrictVersion("2.1.0"):
            logger.warning("importing from an older version ({}) of PHOEBE into version {}".format(phoebe_version_import, phoebe_version_this))

            # rpole -> requiv: https://github.com/phoebe-project/phoebe2/pull/300
            dict_stars = {}
            for star in b.hierarchy.get_stars():
                ps_star = b.filter(context='component', component=star)
                dict_stars[star] = _ps_dict(ps_star)

                # TODO: actually do the translation
                rpole = dict_stars[star].pop('rpole', 1.0*u.solRad).to(u.solRad).value
                # PHOEBE 2.0 didn't have syncpar for contacts
                if len(b.filter(qualifier='syncpar', component=star)):
                    F = b.get_value(qualifier='syncpar', component=star, context='component')
                else:
                    F = 1.0
                parent_orbit = b.hierarchy.get_parent_of(star)
                component = b.hierarchy.get_primary_or_secondary(star, return_ind=True)
                sma = b.get_value(qualifier='sma', component=parent_orbit, context='component', unit=u.solRad)
                q = b.get_value(qualifier='q', component=parent_orbit, context='component')
                d = 1 - b.get_value(qualifier='ecc', component=parent_orbit)

                logger.info("roche.rpole_to_requiv_aligned(rpole={}, sma={}, q={}, F={}, d={}, component={})".format(rpole, sma, q, F, d, component))
                dict_stars[star]['requiv'] = roche.rpole_to_requiv_aligned(rpole, sma, q, F, d, component=component)

                b.remove_component(star)


            for star, dict_star in dict_stars.items():
                logger.info("attempting to update component='{}' to new version requirements".format(star))
                b.add_component('star', component=star, check_label=False, **dict_star)


            dict_envs = {}
            for env in b.hierarchy.get_envelopes():
                ps_env = b.filter(context='component', component=env)
                dict_envs[env] = _ps_dict(ps_env)
                b.remove_component(env)

            for env, dict_env in dict_envs.items():
                logger.info("attempting to update component='{}' to new version requirements".format(env))
                b.add_component('envelope', component=env, check_label=False, **dict_env)

                # TODO: this probably will fail once more than one contacts are
                # supported, but will never need that for 2.0->2.1 since
                # multiples aren't supported (yet) call b.set_hierarchy() to

                # reset all hieararchy-dependent constraints (including
                # pot<->requiv)
                b.set_hierarchy()

                primary = b.hierarchy.get_stars()[0]
                b.flip_constraint('pot', component=env, solve_for='requiv@{}'.format(primary), check_nan=False)
                b.set_value(qualifier='pot', component=env, context='component', value=dict_env['pot'])
                b.flip_constraint('requiv', component=primary, solve_for='pot', check_nan=False)

            # reset all hieararchy-dependent constraints
            b.set_hierarchy()

            # mesh datasets: https://github.com/phoebe-project/phoebe2/pull/261, https://github.com/phoebe-project/phoebe2/pull/300
            for dataset in b.filter(context='dataset', kind='mesh').datasets:
                logger.info("attempting to update mesh dataset='{}' to new version requirements".format(dataset))
                ps_mesh = b.filter(context='dataset', kind='mesh', dataset=dataset)
                dict_mesh = _ps_dict(ps_mesh)
                # NOTE: we will not remove (or update) the dataset from any existing models
                b.remove_dataset(dataset, context=['dataset', 'constraint', 'compute'])
                if len(b.filter(dataset=dataset, context='model')):
                    logger.warning("existing model for dataset='{}' models={} will not be removed, but likely will not work with new plotting updates".format(dataset, b.filter(dataset=dataset, context='model').models))

                b.add_dataset('mesh', dataset=dataset, check_label=False, **dict_mesh)

            # vgamma definition: https://github.com/phoebe-project/phoebe2/issues/234
            logger.info("updating vgamma to new version requirements")
            b.set_value(qualifier='vgamma', value=-1*b.get_value(qualifier='vgamma'))

            # remove phshift parameter: https://github.com/phoebe-project/phoebe2/commit/1fa3a4e1c0f8d80502101e1b1e750f5fb14115cb
            logger.info("removing any phshift parameters for new version requirements")
            b.remove_parameters_all(qualifier='phshift')

            # colon -> long: https://github.com/phoebe-project/phoebe2/issues/211
            logger.info("removing any colon parameters for new version requirements")
            b.remove_parameters_all(qualifier='colon')

            # make sure constraints are updated according to conf.interactive_constraints
            b.run_delayed_constraints()

        if phoebe_version_import < StrictVersion("2.1.2"):
            b._import_before_v211 = True
            warning = "importing from an older version ({}) of PHOEBE which did not support constraints in solar units.  All constraints will remain in SI, but calling set_hierarchy will likely fail.".format(phoebe_version_import)
            print("WARNING: {}".format(warning))
            logger.warning(warning)

        if phoebe_version_import < StrictVersion("2.2.0"):
            warning = "importing from an older version ({}) of PHOEBE which did not support compute_times, ld_mode/ld_coeffs_source, pblum_mode, l3_mode, etc... all datasets will be migrated to include all new options.  This may take some time.  Please check all values.".format(phoebe_version_import)
            # print("WARNING: {}".format(warning))
            logger.warning(warning)

            def existing_value(param):
                if param.qualifier == 'l3':
                    q = param.get_quantity()
                    try:
                        return q.to(u.W/u.m**2)
                    except:
                        # older versions had the unit incorrect, so let's just assume u.W/u.m**3 meant u.W/u.m**2
                        return q.to(u.W/u.m**3).value * u.W/u.m**2
                if param.qualifier == 'ld_func':
                    if param.value == 'interp':
                        return 'logarithmic'
                    else:
                        return param.value
                else:
                    return param.get_quantity() if hasattr(param, 'get_quantity') else param.get_value()

            # TESS:default has been renamed to TESS:T
            # Tycho:BT has been renamed to Tycho:B
            # Tycho:VT has been renamed to Tycho:V
            pb_map = {'TESS:default': 'TESS:T', 'Tycho:BT': 'Tycho:B', 'Tycho:VT': 'Tycho:V'}
            for param in b.filter(qualifier='passband', **_skip_filter_checks).to_list():
                old_value = param.get_value()
                if old_value in pb_map.keys():
                    new_value = pb_map.get(old_value)
                    logger.warning("migrating passband='{}' to passband='{}'".format(old_value, new_value))
                    param.set_value(new_value)

            existing_values_settings = {p.qualifier: p.get_value() for p in b.filter(context='setting').to_list()}
            b.remove_parameters_all(context='setting', **_skip_filter_checks)
            b._attach_params(_setting.settings(**existing_values_settings), context='setting')

            # overwriting the datasets during migration will clear the model, so
            # let's save a copy and re-attach it after
            ps_model = b.filter(context='model', check_visible=False, check_default=False)

            existing_values_per_ds = {}
            for ds in b.filter(context='dataset').datasets:
                # NOTE: before 2.2.0, contexts included *_syn and *_dep, so
                # we need to be aware of that in this block of logic.

                # TODO: migrate pblum_ref to pblum_mode = 'decoupled' or pblum_mode = 'dataset-constrained' and set pblum_component?
                ds_kind = b.get_dataset(ds).exclude(kind=["*_syn", "*_dep"]).kind
                existing_values = {}

                if ds_kind == 'lc':
                    # handle pblum_ref -> pblum_mode/pblum_component
                    if len(b.filter(qualifier='pblum_ref', value='self', context='dataset', dataset=ds, check_visible=False, check_default=False)) == 2:
                        existing_values['pblum_mode'] == 'decoupled'
                    else:
                        existing_values['pblum_mode'] = 'component-coupled'
                        existing_values['pblum_component'] = b.filter(qualifier='pblum_ref', context='dataset', dataset=ds, check_visible=False).exclude(value='self', check_visible=False).get_parameter(check_visible=True).component


                for qualifier in b.filter(context='dataset', dataset=ds, **_skip_filter_checks).qualifiers:
                    if qualifier in ['pblum_ref']:
                        # already handled these above
                        continue
                    ps = b.filter(qualifier=qualifier, context='dataset', dataset=ds, **_skip_filter_checks)
                    if len(ps.to_list()) > 1:
                        existing_values[qualifier] = {}
                        for param in ps.to_list():
                            existing_values[qualifier]["{}@{}".format(param.time, param.component) if param.time is not None else param.component] = existing_value(param)
                            if qualifier=='ld_func':
                                if 'ld_mode' not in existing_values.keys():
                                    existing_values['ld_mode'] = {}
                                existing_values['ld_mode']["{}@{}".format(param.time, param.component) if param.time is not None else param.component] = 'interp' if param.value == 'interp' else 'manual'

                    else:
                        param = b.get_parameter(qualifier=qualifier, context='dataset', dataset=ds, **_skip_filter_checks)
                        existing_values[qualifier] = existing_value(param)
                        if qualifier=='ld_func':
                            existing_values['ld_mode']["{}@{}".format(param.time, param.component) if param.time is not None else param.component] = 'interp' if param.value == 'interp' else 'manual'

                if ds_kind in ['lp']:
                    # then we need to pass the times from the attribute instead of parameter
                    existing_values['times'] = b.filter(context='dataset', dataset=ds, **_skip_filter_checks).times

                existing_values['kind'] = ds_kind

                existing_values_per_ds[ds] = existing_values
                b.remove_dataset(dataset=ds)

            for ds, existing_values in existing_values_per_ds.items():
                ds_kind = existing_values.pop('kind')
                logger.warning("migrating '{}' {} dataset.".format(ds, ds_kind))
                logger.debug("applying existing values to {} dataset: {}".format(ds, existing_values))
                b.add_dataset(ds_kind, dataset=ds, overwrite=True, **existing_values)

            for component in b.filter(context='component', **_skip_filter_checks).components:
                existing_values = {p.qualifier: p.get_value() for p in b.filter(context='component', component=component, **_skip_filter_checks).to_list()}
                logger.warning("migrating '{}' component".format(component))
                logger.debug("applying existing values to {} component: {}".format(component, existing_values))
                b.add_component(kind=b.get_component(component=component, check_visible=False).kind, component=component, overwrite=True, **existing_values)

            # make sure constraints all attach
            b.set_hierarchy()

            logger.debug("restoring previous models")
            b._attach_params(ps_model, context='model')

        if phoebe_version_import < StrictVersion("2.3.0"):
            warning = "importing from an older version ({}) of PHOEBE which did not support sample_from, etc... all compute options will be migrated to include all new options.  Additionally, extinction parameters will be moved from the dataset to system context.  This may take some time.  Please check all values.".format(phoebe_version_import)

            b.remove_parameters_all(qualifier='log_history', **_skip_filter_checks)

            # new settings parameters were added for run_checks_*
            logger.warning("updating all parameters in setting context")
            existing_values_settings = {p.qualifier: p.get_value() for p in b.filter(context='setting').to_list()}
            b.remove_parameters_all(context='setting', **_skip_filter_checks)
            b._attach_params(_setting.settings(**existing_values_settings), context='setting')

            # new mean_anom parameter in orbits and updated descriptions in star parameters
            for component in b.filter(context='component', **_skip_filter_checks).components:
                existing_values = {p.qualifier: p.get_value() for p in b.filter(context='component', component=component, **_skip_filter_checks).to_list()}
                logger.warning("migrating '{}' component".format(component))
                logger.debug("applying existing values to {} component: {}".format(component, existing_values))
                b.add_component(kind=b.get_component(component=component, check_visible=False).kind, component=component, overwrite=True, **existing_values)

            # update logg constraints (now in solar units due to bug with MPI handling converting solMass to SI)
            for logg_constraint in b.filter(qualifier='logg', context='constraint', **_skip_filter_checks).to_list():
                component = logg_constraint.component
                logger.warning("re-creating logg constraint for component='{}' to be in solar instead of SI units".format(component))
                b.remove_constraint(uniqueid=logg_constraint.uniqueid)
                b.add_constraint('logg', component=component)

            for compute in b.filter(context='compute').computes:
                logger.info("attempting to update compute='{}' to new version requirements".format(compute))
                ps_compute = b.filter(context='compute', compute=compute, **_skip_filter_checks)
                compute_kind = ps_compute.kind
                dict_compute = _ps_dict(ps_compute)
                # NOTE: we will not remove (or update) the dataset from any existing models
                b.remove_compute(compute, context=['compute'])
                b.add_compute(compute_kind, compute=compute, check_label=False, **dict_compute)

            # all datasets need to be rebuilt to handle compute_phases_t0 -> phases_t0
            # and add mask_phases and solver_times support
            for dataset in b.filter(context='dataset').datasets:
                logger.info("attempting to update dataset='{}' to new version requirements".format(dataset))
                ps_ds = b.filter(context='dataset', dataset=dataset, **_skip_filter_checks)
                ds_kind = ps_ds.kind
                dict_ds = _ps_dict(ps_ds, include_constrained=False)
                if 'compute_phases_t0' in dict_ds.keys():
                    dict_ds['phases_t0'] = dict_ds.pop('compute_phases_t0')
                b.remove_dataset(dataset, context=['dataset', 'constraint'])
                b.add_dataset(ds_kind, dataset=dataset, check_label=False, **dict_ds)

            # extinction parameters were moved from dataset to system, so we'll just add the new parameters
            system, constraints = _system.system()
            b._attach_params([p for p in system.to_list() if p.qualifier in ['ebv', 'Av', 'Rv']], context='system')

            Avs = list(set([Av_param.get_value() for Av_param in b.filter(qualifier='Av', context='dataset', **_skip_filter_checks).to_list()]))
            if len(Avs):
                if len(Avs) > 1:
                    logger.warning("PHOEBE no longer supports multiple values for Av, adopting Av={}".format(Avs[0]))
                b.set_value(qualifier='Av', context='system', value=Avs[0], **_skip_filter_checks)
            Rvs = list(set([Rv_param.get_value() for Rv_param in b.filter(qualifier='Rv', context='dataset', **_skip_filter_checks).to_list()]))
            if len(Rvs):
                if len(Rvs) > 1:
                    logger.warning("PHOEBE no longer supports multiple values for Rv, adopting Rv={}".format(Rvs[0]))
                b.set_value(qualifier='Rv', context='system', value=Rvs[0], **_skip_filter_checks)

            b.remove_parameters_all(qualifier=['ebv', 'Av', 'Rv'], context='dataset', **_skip_filter_checks)
            b.remove_parameters_all(constraint_func='extinction', context='constraint', **_skip_filter_checks)

            for constraint in constraints:
                # there were no constraints before in the system context
                b.add_constraint(*constraint)

            # call set_hierarchy to force asini@component constraints (comp_asini) to be built
            b.set_hierarchy()

        elif phoebe_version_import < StrictVersion("2.3.25"):
            # elif here since the if above already call set_hierarchy and we want to avoid doing that twice since its expensive

            # call set_hierarchy to force mass constraints to be rebuilt
            b.set_hierarchy()

        b.run_all_constraints()
        return b



    @classmethod
    def from_server(cls, bundleid, server='http://localhost:5555',
                    as_client=True):
        """
        Load a bundle from a phoebe server.  This is a constructor so should be
        called as:

        ```py
        b = Bundle.from_server('asdf', as_client=False)
        ```

        See also:
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.as_client>
        * <phoebe.frontend.bundle.Bundle.is_client>

        Arguments
        ----------
        * `bundleid` (string): the identifier given to the bundle by the
            server.
        * `server` (string, optional, default='http://localhost:5555'): the
            host (and port) of the server.
        * `as_client` (bool, optional, default=True):  whether to attach in
            client mode.  If True, `server` will be passed to
            <phoebe.frontend.bundle.Bundle.as_client> as `as_client`.
        """
        # TODO: support default cases from server?

        if server[:4] != "http":
            server = "http://"+server
        url = "{}/json_bundle/{}".format(server, bundleid)
        logger.info("downloading bundle from {}".format(url))
        r = requests.get(url, timeout=5)
        rjson = r.json()

        if not rjson['data']['success']:
            raise ValueError("server error: {}".format(rjson['data'].get('error', 'unknown error')))

        b = cls(rjson['data']['bundle'])

        if as_client:
            b.as_client(as_client=server,
                        bundleid=rjson['meta']['bundleid'])

            logger.warning("This bundle is in client mode, meaning all computations will be handled by the server at {}.  To disable client mode, call as_client(False) or in the future pass as_client=False to from_server".format(server))

        return b

    @classmethod
    def from_legacy(cls, filename, add_compute_legacy=True, add_compute_phoebe=True,
                   ignore_errors=False, passband_map={}):
        """
        For convenience, this function is available at the top-level as
        <phoebe.from_legacy> as well as <phoebe.frontend.bundle.Bundle.from_legacy>.

        Load a bundle from a PHOEBE 1.0 Legacy file.

        This is a constructor so should be called as:

        ```py
        b = Bundle.from_legacy('myfile.phoebe')
        ```

        See also:
        * <phoebe.parameters.compute.legacy>

        Arguments
        ------------
        * `filename` (string or file object): relative or full path to the file
            or an opened python file object.  NOTE: if passing a file object,
            referenced data files will be ignored.  If wanting to load referenced
            data files, pass the location of the file so that relative paths
            to other files can be correctly parsed.
        * `add_compute_legacy` (bool, optional, default=True): whether to add
            a set of compute options for the legacy backend.  See also
            <phoebe.frontend.bundle.Bundle.add_compute> and
            <phoebe.parameters.compute.legacy> to add manually after.
        * `add_compute_phoebe` (bool, optional, default=True): whether to add
            a set of compute options for the phoebe backend.  See also
            <phoebe.frontend.bundle.Bundle.add_compute> and
            <phoebe.parameters.compute.phoebe> to add manually after.
        * `ignore_errors` (bool, optional, default=False): whether to ignore any
            import errors and include instead as a warning in the logger.
        * `passband_map` (dict, optional, default={}): dictionary to map passbands
            from the value in the legacy file to the corresponding value  in PHOEBE.

        Returns
        ---------
        * an instantiated <phoebe.frontend.bundle.Bundle> object.
        """
        logger.warning("importing from legacy is experimental until official 1.0 release")
        return io.load_legacy(filename, add_compute_legacy, add_compute_phoebe,
                              ignore_errors=ignore_errors, passband_map=passband_map)

    @classmethod
    def default_star(cls, starA='starA', force_build=False):
        """
        For convenience, this function is available at the top-level as
        <phoebe.default_star> as well as <phoebe.frontend.bundle.Bundle.default_star>.

        Load a bundle with a default single star as the system.

        sun

        This is a constructor, so should be called as:

        ```py
        b = Bundle.default_binary()
        ```

        Arguments
        -----------
        * `starA` (string, optional, default='starA'): the label to be set for
            starA.
        * `force_build` (bool, optional, default=False): whether to force building
            the bundle from scratch.  If False, pre-cached files will be loaded
            whenever possible to save time.

        Returns
        -----------
        * an instantiated <phoebe.frontend.bundle.Bundle> object.
        """
        if not force_build and not conf.devel:
            b = cls.open(os.path.join(_bundle_cache_dir, 'default_star.bundle'))

            if starA != 'starA':
                b.rename_component('starA', starA)

            b._update_atm_choices()

            return b

        b = cls()
        # IMPORTANT NOTE: if changing any of the defaults for a new release,
        # make sure to update the cached files (see frontend/default_bundles
        # directory for script to update all cached bundles)
        b.add_star(component=starA, color='blue')
        b.set_hierarchy(_hierarchy.component(b[starA]))
        b.add_compute(distortion_method='rotstar', irrad_method='none')
        return b

    @classmethod
    def default_binary(cls, starA='primary', starB='secondary', orbit='binary',
                       semidetached=False,
                       contact_binary=False, force_build=False):
        """
        For convenience, this function is available at the top-level as
        <phoebe.default_binary> as well as
        <phoebe.frontend.bundle.Bundle.default_binary>.

        Load a bundle with a default binary as the system.

        primary - secondary

        This is a constructor, so should be called as:

        ```py
        b = Bundle.default_binary()
        ```

        Arguments
        -----------
        * `starA` (string, optional, default='primary'): the label to be set for
            the primary component.
        * `starB` (string, optional, default='secondary'): the label to be set for
            the secondary component.
        * `orbit` (string, optional, default='binary'): the label to be set for
            the binary component.
        * `semidetached` (string or bool, optional, default=False): component
            to apply a semidetached constraint.  If False, system will be detached.
            If True, both components will have semidetached constraints (a
            double-contact system).  `contact_binary` must be False.
        * `contact_binary` (bool, optional, default=False): whether to also
            add an envelope (with component='contact_envelope') and set the
            hierarchy to a contact binary system.  `semidetached` must be False.
        * `force_build` (bool, optional, default=False): whether to force building
            the bundle from scratch.  If False, pre-cached files will be loaded
            whenever possible to save time.

        Returns
        -----------
        * an instantiated <phoebe.frontend.bundle.Bundle> object.

        Raises
        -----------
        * ValueError: if at least one of `semidetached` and `contact_binary` are
            not False.
        """
        if semidetached and contact_binary:
            raise ValueError("at least one of semidetached or binary must be False")

        if not force_build and not conf.devel:
            if contact_binary:
                b = cls.open(os.path.join(_bundle_cache_dir, 'default_contact_binary.bundle'))
            else:
                b = cls.open(os.path.join(_bundle_cache_dir, 'default_binary.bundle'))

            secondary = 'secondary'
            if starA != 'primary':
                if starA == 'secondary':
                    secondary = 'temp_secondary'
                    b.rename_component('secondary', secondary)
                b.rename_component('primary', starA)
            if starB != 'secondary':
                b.rename_component(secondary, starB)
            if orbit != 'binary':
                b.rename_component('binary', orbit)

            if semidetached == starA or semidetached is True:
                b.add_constraint('semidetached', component=starA)
            if semidetached == starB or semidetached is True:
                b.add_constraint('semidetached', component=starB)

            if semidetached:
                # then we need to run the constraint
                b.run_delayed_constraints()

            b._update_atm_choices()

            return b

        b = cls()
        # IMPORTANT NOTE: if changing any of the defaults for a new release,
        # make sure to update the cached files (see frontend/default_bundles
        # directory for script to update all cached bundles)
        if contact_binary:
            orbit_defaults = {'sma': 3.35, 'period': 0.5}
            star_defaults = {'requiv': 1.5}
        else:
            orbit_defaults = {'sma': 5.3, 'period': 1.0}
            star_defaults = {'requiv': 1.0}
        b.add_star(component=starA, color='blue', **star_defaults)
        b.add_star(component=starB, color='orange', **star_defaults)
        b.add_orbit(component=orbit, **orbit_defaults)
        if contact_binary:
            b.add_component('envelope', component='contact_envelope')
            b.set_hierarchy(_hierarchy.binaryorbit,
                            b[orbit],
                            b[starA],
                            b[starB],
                            b['contact_envelope'])
        else:
            b.set_hierarchy(_hierarchy.binaryorbit,
                            b[orbit],
                            b[starA],
                            b[starB])

            if semidetached == starA or semidetached is True:
                b.add_constraint('semidetached', component=starA)
            if semidetached == starB or semidetached is True:
                b.add_constraint('semidetached', component=starB)


        b.add_compute()

        return b

    @classmethod
    def default_contact_binary(cls, *args, **kwargs):
        """
        For convenience, this function is available at the top-level as
        <phoebe.default_contact_binary> as well as
        <phoebe.frontend.bundle.Bundle.default_contact_binary>.

        This is a shortcut to <phoebe.frontend.bundle.Bundle.default_binary>
        but with `contact_binary` set to True.
        """
        return cls.default_binary(contact_binary=True, *args, **kwargs)

    @classmethod
    def default_triple(cls, inner_as_primary=True, inner_as_overcontact=False,
                       starA='starA', starB='starB', starC='starC',
                       inner='inner', outer='outer',
                       contact_envelope='contact_envelope'):
        """
        For convenience, this function is available at the top-level as
        <phoebe.default_triple> as well as
        <phoebe.frontend.bundle.Bundle.default_triple>.

        Load a bundle with a default triple system.

        Set inner_as_primary based on what hierarchical configuration you want.

        `inner_as_primary = True`:

        starA - starB -- starC

        `inner_as_primary = False`:

        starC -- starA - starB

        This is a constructor, so should be called as:

        ```py
        b = Bundle.default_triple_primary()
        ```

        Arguments
        -----------


        Returns
        -------------
        * an instantiated <phoebe.frontend.bundle.Bundle> object.
        """
        if not conf.devel:
            raise NotImplementedError("'default_triple' not officially supported for this release.  Enable developer mode to test.")

        b = cls()
        b.add_star(component=starA, color='blue')
        b.add_star(component=starB, color='orange')
        b.add_star(component=starC, color='green')
        b.add_orbit(component=inner, period=1)
        b.add_orbit(component=outer, period=10)

        if inner_as_overcontact:
            b.add_envelope(component=contact_envelope)
            inner_hier = _hierarchy.binaryorbit(b[inner],
                                           b[starA],
                                           b[starB],
                                           b[contact_envelope])
        else:
            inner_hier = _hierarchy.binaryorbit(b[inner], b[starA], b[starB])

        if inner_as_primary:
            hierstring = _hierarchy.binaryorbit(b[outer], inner_hier, b[starC])
        else:
            hierstring = _hierarchy.binaryorbit(b[outer], b[starC], inner_hier)
        b.set_hierarchy(hierstring)

        b.add_constraint(constraint.keplers_third_law_hierarchical,
                         outer, inner)

        # TODO: does this constraint need to be rebuilt when things change?
        # (ie in set_hierarchy)

        b.add_compute()

        return b

    def save(self, filename, compact=False, incl_uniqueid=True):
        """
        Save the bundle to a JSON-formatted ASCII file.  This will run failed
        and delayed constraints and raise an error if they fail.

        See also:
        * <phoebe.parameters.ParameterSet.save>
        * <phoebe.parameters.Parameter.save>

        Arguments
        ------------
        * `filename` (string): relative or full path to the file
        * `compact` (bool, optional, default=False): whether to use compact
            file-formatting (may be quicker to save/load, but not as easily readable)

        Returns
        -------------
        * the filename (string)
        """
        if not incl_uniqueid:
            logger.warning("saving without uniqueids could cause issues with solutions, use with caution")
        # TODO: add option for clear_models, clear_solution
        # NOTE: PS.save will handle os.path.expanduser
        self.run_delayed_constraints()
        self.run_failed_constraints()
        return super(Bundle, self).save(filename, incl_uniqueid=incl_uniqueid,
                                        compact=compact)

    def export_legacy(self, filename, compute=None, skip_checks=False):
        """
        Export the Bundle to a file readable by PHOEBE legacy.

        See also:
        * <phoebe.parameters.compute.legacy>

        Arguments
        -----------
        * `filename` (string): relative or full path to the file
        * `compute` (string, optional, default=None): label of the compute options
            to use while exporting.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_compute> before exporting.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.

        Returns
        ------------
        * the filename (string)
        """
        logger.warning("exporting to legacy is experimental until official 1.0 release")
        self.run_delayed_constraints()

        if not skip_checks:
            report = self.run_checks_compute(compute=compute, allow_skip_constraints=False,
                                             raise_logger_warning=True, raise_error=True,
                                             run_checks_system=True)

        filename = os.path.expanduser(filename)
        legacy_dict = io.pass_to_legacy(self, compute=compute)

        return io.write_legacy_file(legacy_dict, filename)

        #return io.pass_to_legacy(self, filename, compute=compute)


    def _test_server(self, server='http://localhost:5555', wait_for_server=False):
        """
        [NOT IMPLEMENTED]
        """
        try:
            resp = _urlopen("{}/info".format(server))
        except URLError:
            test_passed = False
        else:
            resp = json.loads(resp.read())
            test_passed = resp['data']['success']

        if not test_passed:
            if wait_for_server:
                time.sleep(0.5)
                return self._test_server(server=server, wait_for_server=wait_for_server)

            return False

        return test_passed

    def _on_socket_connect(self, *args):
        """
        [NOT IMPLEMENTED]
        """
        logger.info("connected to server")
        self._server_clients = [_clientid]

    def _on_socket_disconnect(self, *args):
        """
        [NOT IMPLEMENTED]
        test
        """
        logger.warning("disconnected from server")
        if self.is_client and self._client_allow_disconnect:
            logger.warning("exiting client mode")

            self._socketio.disconnect()
            self._bundleid = None
            self._is_client = False
            self._client_allow_disconnect = False

            self._server_clients = []

    def _on_socket_push_updates(self, resp):
        """
        [NOT IMPLEMENTED]
        """
        # TODO: check to make sure resp['meta']['bundleid']==bundleid ?
        # TODO: handle added parameters
        # TODO: handle removed (isDeleted) parameters
        logger.debug('_on_socket_push_updates resp={}'.format(resp))

        requestid = resp.pop('requestid', None)

        server_changes = ParameterSet([])

        # resp['data'] = {'success': True/False, 'parameters': {uniqueid: {context: 'blah', value: 'blah', ...}}}
        added_new_params = False
        for uniqueid, paramdict in resp['parameters'].items():
            if uniqueid in self.uniqueids:
                param = self.get_parameter(uniqueid=uniqueid, check_visible=False, check_default=False, check_advanced=False)
                for attr, value in paramdict.items():
                    if hasattr(param, "_{}".format(attr)):
                        logger.debug("updates from server: setting {}@{}={}".
                                    format(attr, param.twig, value))

                        # we cannot call param.set_value because that will
                        # emit another signal to the server.  So we do need
                        # to hardcode some special cases here
                        if isinstance(value, dict):
                            if 'nparray' in value.keys():
                                value = nparray.from_json(value)

                        if attr == 'value' and hasattr(param, 'default_unit') and param.__class__.__name__ != "ConstraintParameter":
                            if 'default_unit' in paramdict.keys():
                                unit = u.Unit(paramdict.get('default_unit'))
                            else:
                                unit = param.default_unit
                            if isinstance(unit, str):
                                unit = u.Unit(unit)
                            value = value * unit

                        setattr(param, "_{}".format(attr), value)

                    server_changes += [param]
            else:
                server_changes += self._attach_param_from_server(paramdict)
                added_new_params = True

        if added_new_params:
            # we skipped processing copy_for to avoid copying before all parameters
            # were loaded, so we should do that now.
            self._check_copy_for()

        if len(resp.get('removed_parameters', [])):
            # print("*** removed_parameters", resp.get('removed_parameters'))
            server_changes += self.remove_parameters_all(uniqueid=resp.get('removed_parameters'), **_skip_filter_checks)

        if requestid == self._waiting_on_server:
            self._waiting_on_server = False

        self._server_changes = server_changes

    def _on_socket_push_error(self, resp):
        # TODO: check to make sure resp['meta']['bundleid']==bundleid ?
        requestid = resp.pop('requestid', None)
        if requestid == self._waiting_on_server:
            self._waiting_on_server = False

        raise ValueError("error from server: {}".format(resp.get('error', 'error message not provided')))

    def _on_socket_push_registeredclients(self, resp):
        self._server_clients = resp.get('connected_clients', [])

    def _attach_param_from_server(self, item):
        """
        """
        if isinstance(item, list):
            ret_ = []
            for itemi in item:
                ret_ += [self._attach_param_from_server(itemi)]
            return ret_
        else:
            # then we need to add a new parameter
            d = item
            param = parameters.parameter_from_json(d, bundle=self)

            metawargs = {}
            self._attach_params([param], check_copy_for=False, **metawargs)
            return [param]

    def _deregister_client(self, bundleid=None):
        # called at python exit as well as b.as_client(False)

        if self._socketio is None:
            return

        if self._server_secret is not None:
            logger.warning("attempting to kill child server process at {}".format(self.is_client))
            self._socketio.emit('kill', {'clientid': _clientid, 'secret': self._server_secret})
            self._server_secret = None
        else:
            logger.info("deregistering {} client from {}".format(_clientid, self.is_client))
            self._socketio.emit('deregister client', {'clientid': _clientid, 'bundleid': None})
            if bundleid is not None:
                self._socketio.emit('deregister client', {'clientid': _clientid, 'bundleid': bundleid})


        self._socketio.disconnect()
        self._socketio = None

    def as_client(self, as_client='http://localhost:5555',
                  bundleid=None, wait_for_server=False, reconnection_attempts=None,
                  blocking=False):
        """
        Enter (or exit) client mode.  Client mode allows a server (either running
        locally on localhost or on an external machine) to host the bundle
        and handle all computations.  Any changes to parameters, etc, are then
        transmitted to each of the attached clients so that they remain in
        sync.

        PHOEBE will first try a connection with `as_client`.  If an instance of
        PHOEBE server is not found, `as_client` includes 'localhost', and
        the port is open, then the server will be launched automatically as a
        child process of this thread (it will attempt to be killed when
        calling `as_client(as_client=False)` or when closing python) - but
        is not always successful and the thread may need to be killed manually.

        See also:
        * <phoebe.frontend.bundle.Bundle.from_server>
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.ui_figures>
        * <phoebe.frontend.bundle.Bundle.is_client>

        Arguments
        -----------
        * `as_client` (bool or string, optional, default='localhost:5555'):
            If a string: will attach to the server at the provided URL.  If
            the server does not exist but is at localhost, one will attempt
            to be launched as a child process and will be closed when exiting
            or leaving client-mode.  If True, will default to the above
            with 'localhost:5555'.  If False, will disconnect from the existing
            connection and exit client mode.
        * `bundleid` (string, optional, default=None): if provided and the
            bundleid is available from the given server, that bundle will be
            downloaded and attached.  If provided but bundleid is not available
            from the server, the current bundle will be uploaded and assigned
            the given bundleid.  If not provided, the current bundle will be
            uploaded and assigned a random bundleid.
        * `wait_for_server` (bool, optional, default=False): whether to wait
            for the server to be started externally rather than attempt to
            launch a child-process or raise an error.
        * `reconnection_attempts` (int or None, optional, default=None): number
            of reconnection attempts to allow before disonnnecting automatically.
        * `blocking` (bool, optional, default=False): whether to enter client-mode
            synchronously (will not return until the connection is closed) at
            which point the bundle will no longer be in client mode.
            This is used internally by <phoebe.parameters.ParameterSet.ui>,
            but should be overridden with caution.


        Raises
        ---------
        * ImportError: if required dependencies for client mode are not met.
        * ValueError: if the server at `server` is not running or reachable.
        * ValueError: if the server returns an error.
        """
        if as_client:
            if not _can_client:
                raise ImportError("dependencies to support client mode not met - see docs")

            if as_client is True:
                server = 'localhost:5555'
            else:
                server = as_client

            if 'http' not in server[:5]:
                server = 'http://'+server

            server_running = self._test_server(server=server,
                                               wait_for_server=wait_for_server)
            if not server_running:
                if 'localhost' not in server:
                    raise ValueError("server {} is not running and does not include 'localhost' so will not be launched automatically".format(server))

                try:
                    port = int(server.split(':')[-1])
                except:
                    raise ValueError("could not detect port from server='{}' when attempting to launch.  server should have the format 'localhost:5500'")

                self._server_secret = _uniqueid(6)
                cmd = 'phoebe-server --port {} --parent {} --secret {} &'.format(port, _clientid, self._server_secret)
                logger.info("system command: {}".format(cmd))
                os.system(cmd)

                server_running = self._test_server(server=server,
                                                   wait_for_server=True)

            self._socketio = socketio.Client(reconnection_delay=0.1, reconnection_attempts=0 if reconnection_attempts is None else reconnection_attempts)
            self._socketio.on('connect', self._on_socket_connect)
            self._socketio.on('disconnect', self._on_socket_disconnect)
            self._socketio.connect(server)

            if bundleid is not None:
                rj = requests.get("{}/info".format(server)).json()
                if bundleid in rj['data']['clients_per_bundle'].keys():
                    upload = False
                else:
                    upload = True
            else:
                upload = True

            if upload:
                upload_url = "{}/open_bundle/load:phoebe2".format(server)
                logger.info("uploading bundle to server {}".format(upload_url))
                data = json.dumps({'json': self.to_json(incl_uniqueid=True), 'bundleid': bundleid})
                rj = requests.post(upload_url, data=data, timeout=5).json()
                if rj['data']['success']:
                    bundleid = rj['data']['bundleid']
                else:
                    raise ValueError("server error: {}".format(rj['data'].get('error', 'unknown error')))

            self._socketio.emit('register client', {'clientid': _clientid, 'bundleid': bundleid})

            self._socketio.on('{}:changes:python'.format(bundleid), self._on_socket_push_updates)
            self._socketio.on('{}:errors:python'.format(bundleid), self._on_socket_push_error)
            self._socketio.on('{}:registeredclients'.format(bundleid), self._on_socket_push_registeredclients)

            self._bundleid = bundleid

            self._is_client = server

            atexit.register(self._deregister_client)

            logger.info("connected as client {} to server at {}".
                        format(_clientid, server))

            if blocking:
                self._socketio.wait()
                self.as_client(as_client=False)

        else:
            logger.warning("This bundle is now permanently detached from the instance on the server and will not receive future updates.  To start a client in sync with the version on the server or other clients currently subscribed, you must instantiate a new bundle with Bundle.from_server.")

            if hasattr(self, '_socketio') and self._socketio is not None:
                self._deregister_client(bundleid=self._bundleid)

            self._bundleid = None
            self._is_client = False
            self._server_clients = []

    @property
    def is_client(self):
        """
        Whether the <phoebe.frontend.bundle.Bundle> is currently in client-mode.

        See also:
        * <phoebe.frontend.bundle.Bundle.from_server>
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.as_client>

        Returns
        ---------
        * False if the bundle is not in client mode, otherwise the URL of the server.
        """
        return self._is_client

    def __repr__(self):
        # filter to handle any visibility checks, etc
        return self.filter().__repr__().replace('ParameterSet', 'PHOEBE Bundle')

    def __str__(self):
        # filter to handle any visibility checks, etc
        return self.filter().__str__().replace('ParameterSet', 'PHOEBE Bundle')

    def _default_label(self, base, context, **kwargs):
        """
        Determine a default label given a base label and the passed kwargs

        this simply counts the current number of matches on metawargs and
        appends that number to the base

        :parameter str base: the base string for the label
        :parameter str context: name of the context (where the label is going)
        :parameter **kwargs: the kwargs to run a filter on.  The returned label
            will be "{}{:02d}".format(base, number_of_results_with_kwargs+1)
        :return: label
        """

        kwargs['context'] = context
        params = len(getattr(self.filter(check_visible=False,**kwargs), '{}s'.format(context)))

        return "{}{:02d}".format(base, params+1)

    def _rename_label(self, tag, old_value, new_value, overwrite=False):
        if overwrite and new_value in getattr(self, '{}s'.format(tag)):
            getattr(self, 'remove_{}'.format(tag))(new_value)
        else:
            self._check_label(new_value)

        affected_params = []

        for param in self.filter(check_visible=False, check_default=False, **{tag: old_value}).to_list():
            setattr(param, '_{}'.format(tag), new_value)
            affected_params.append(param)
        for param in self.filter(context='constraint', check_visible=False, check_default=False).to_list():
            for k, v in param.constraint_kwargs.items():
                if v == old_value:
                    param._constraint_kwargs[k] = new_value
                    affected_params.append(param)


        if tag=='dataset':
            for param in self.filter(qualifier='include_times', check_visible=False, check_default=False).to_list():
                old_param_value = param._value
                new_param_value = [v.replace('@{}'.format(old_value), '@{}'.format(new_value)) for v in old_param_value]
                param._value = new_param_value
                affected_params.append(param)

            for param in self.filter(qualifier='lc', context='solution', check_visible=False, check_default=False).to_list():
                if param._value == old_value:
                    param._value = new_value
                    affected_params.append(param)

            affected_params += self._handle_dataset_selectparams(rename={old_value: new_value}, return_changes=True)
            affected_params += self._handle_figure_time_source_params(rename={old_value: new_value}, return_changes=True)

        elif tag=='component':
            affected_params += self._handle_component_selectparams(rename={old_value: new_value}, return_changes=True)
            affected_params += self._handle_pblum_defaults(rename={old_value: new_value}, return_changes=True)

            for param in self.filter(qualifier='orbit', context='solution', check_visible=False, check_default=False).to_list():
                if param._value == old_value:
                    param._value = new_value
                    affected_params.append(param)

        elif tag=='compute':
            affected_params += self._handle_compute_selectparams(rename={old_value: new_value}, return_changes=True)

        elif tag=='model':
            affected_params += self._handle_model_selectparams(rename={old_value: new_value}, return_changes=True)

        elif tag=='distribution':
            affected_params += self._handle_distribution_selectparams(rename={old_value: new_value}, return_changes=True)
            affected_params += self._handle_computesamplefrom_selectparams(rename={old_value: new_value}, return_changes=True)

        elif tag=='solver':
            affected_params += self._handle_solver_selectparams(rename={old_value: new_value}, return_changes=True)

        elif tag=='figure':
            affected_params += self._handle_figure_selectparams(rename={old_value: new_value}, return_changes=True)

        elif tag=='solution':
            affected_params += self._handle_computesamplefrom_selectparams(rename={old_value: new_value}, return_changes=True)
            affected_params += self._handle_solution_selectparams(rename={old_value: new_value}, return_changes=True)

        return affected_params

    def get_setting(self, twig=None, **kwargs):
        """
        Filter in the 'setting' context

        See also:
        * <phoebe.parameters.ParameterSet.filter_or_get>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> or <phoebe.parameters.Parameter> object.
        """
        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'setting'
        return self.filter_or_get(**kwargs)

    def _update_atm_choices(self):
        # affected_params = []
        for param in self.filter(qualifier='atm', kind='phoebe',
                                 check_visible=False, check_default=False).to_list():
            param._choices = _compute._atm_choices
            # affected_params.append(param)

        # return affected_params

    def _handle_pblum_defaults(self, rename={}, return_changes=False):
        """
        """
        logger.debug("calling _handle_pblum_defaults")

        affected_params = []
        changed_params = self.run_delayed_constraints()

        hier = self.get_hierarchy()
        # Handle choice parameters that need components as choices
        # meshablerefs = hier.get_meshables()  # TODO: consider for overcontacts
        starrefs = hier.get_stars()  # TODO: consider for overcontacts
        datasetrefs = self.filter(qualifier='pblum_mode', check_visible=False).datasets

        for param in self.filter(qualifier='pblum_dataset',
                                 context='dataset',
                                 check_visible=False,
                                 check_default=False).to_list():

            param._choices = [ds for ds in datasetrefs if ds!=param.dataset]

            if not len(param._choices):
                param._choices = ['']
                param.set_value('')
                if return_changes:
                    affected_params.append(param)

            elif param.value == '':
                param.set_value(param._choices[0])
                if return_changes:
                    affected_params.append(param)

            else:
                changed = param.handle_choice_rename(**rename)




        for param in self.filter(qualifier='pblum_component',
                                 context='dataset',
                                 check_visible=False,
                                 check_default=False).to_list():

            param._choices = [s for s in starrefs if s!=param.component]

            if param.value == '':
                param.set_value(starrefs[0])
            else:
                changed = param.handle_choice_rename(**rename)

            if return_changes:
                affected_params.append(param)

        return affected_params

    def _handle_dataset_selectparams(self, rename={}, return_changes=False):
        """
        """
        logger.debug("calling _handle_dataset_selectparams")

        affected_params = []
        changed_param = self.run_delayed_constraints()

        dss_ps = self.filter(context='dataset', **_skip_filter_checks)

        pbdep_datasets = dss_ps.filter(kind=_dataset._pbdep_columns.keys(),
                                       **_skip_filter_checks).datasets

        pbdep_columns = _dataset._mesh_columns[:] # force deepcopy
        for pbdep_dataset in pbdep_datasets:
            pbdep_kind = dss_ps.filter(dataset=pbdep_dataset,
                                       kind=_dataset._pbdep_columns.keys(),
                                       **_skip_filter_checks).kind

            pbdep_columns += ["{}@{}".format(column, pbdep_dataset) for column in _dataset._pbdep_columns[pbdep_kind]]

        time_datasets = dss_ps.exclude(kind='mesh').datasets

        t0s = ["{}@{}".format(p.qualifier, p.component) for p in self.filter(qualifier='t0*', context=['component'], **_skip_filter_checks).to_list()]
        t0s += ["t0@system"]

        for param in dss_ps.filter(qualifier='columns', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and pbdep_columns != param._choices:
                choices_changed = True
            param._choices = pbdep_columns
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        for param in dss_ps.filter(qualifier='include_times', **_skip_filter_checks).to_list():

            # NOTE: existing value is updated in change_component
            choices_changed = False
            if return_changes and time_datasets+t0s != param._choices:
                choices_changed = True
            param._choices = time_datasets + t0s
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        for param in self.filter(context='figure', qualifier='datasets', **_skip_filter_checks).to_list():
            ds_same_kind = self.filter(context='dataset', kind=param.kind).datasets

            choices_changed = False
            if return_changes and ds_same_kind != param._choices:
                choices_changed = True
            param._choices = ds_same_kind
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        lcchoices = self.filter(context='dataset', kind='lc', **_skip_filter_checks).datasets
        for param in self.filter(qualifier='lc_datasets', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and lcchoices != param._choices:
                choices_changed = True
            param._choices = lcchoices
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        rvchoices = self.filter(context='dataset', kind='rv', **_skip_filter_checks).datasets
        for param in self.filter(qualifier='rv_datasets', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and rvchoices != param._choices:
                choices_changed = True
            param._choices = rvchoices
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_figure_time_source_params(self, rename={}, return_changes=False):
        affected_params = []

        t0s = ["{}@{}".format(p.qualifier, p.component) for p in self.filter(qualifier='t0*', context=['component'], **_skip_filter_checks).to_list()]
        t0s += ["t0@system"]

        # here we have to use context='dataset' otherwise pb-dependent parameters
        # with context='model', kind='mesh' will show up
        valid_datasets = self.filter(context='dataset', kind=['mesh', 'lp'], **_skip_filter_checks).datasets

        mesh_times = []
        lp_times = []
        mesh_lp_times = []
        for t in self.filter(context='model', kind='mesh').times:
            mesh_times.append('{} ({})'.format(t, ', '.join(ds for ds in self.filter(context='model', time=t, **_skip_filter_checks).datasets if ds in valid_datasets)))
        for t in self.filter(context='model', kind='lp').times:
            lp_times.append('{} ({})'.format(t, ', '.join(ds for ds in self.filter(context='model', time=t, **_skip_filter_checks).datasets if ds in valid_datasets)))
        for t in self.filter(context='model').times:
            mesh_lp_times.append('{} ({})'.format(t, ', '.join(ds for ds in self.filter(context='model', time=t, **_skip_filter_checks).datasets if ds in valid_datasets)))

        for param in self.filter(context='figure', qualifier=['default_time_source', 'time_source'], **_skip_filter_checks).to_list():


            if param.qualifier == 'default_time_source':
                choices = ['None', 'manual'] + t0s + mesh_lp_times
            elif param.kind == 'mesh':
                choices = ['default'] + mesh_times
            elif param.kind == 'lp':
                choices = ['default'] + lp_times
            else:
                choices = ['None', 'default', 'manual'] + t0s + mesh_lp_times

            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                if '(' in param._value:
                    # then its likely just the () part changed, so let's find the
                    # matching item
                    for choice in choices:
                        if choice.split(' (')[0] == param._value.split(' (')[0]:
                            param._value = choice
                            break
                    else:
                        # no match found
                        param._value = 'None' if param.qualifier=='default_time_source' else 'default'

                else:
                    param._value = 'None' if param.qualifier=='default_time_source' else 'default'
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_figure_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []
        changed_params = self.run_delayed_constraints()

        figures = self.filter(context='figure', **_skip_filter_checks).figures

        for param in self.filter(qualifier='run_checks_figure', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and figures != param._choices:
                choices_changed = True
            param._choices = figures

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_compute_choiceparams(self, return_changes=False):
        affected_params = []

        choices = self.filter(context='compute', **_skip_filter_checks).computes

        for param in self.filter(qualifier='compute', context='solver', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                if param._value == 'None' and len(choices):
                    param._value = choices[0]
                else:
                    param._value = 'None'
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params


    def _handle_compute_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []
        changed_params = self.run_delayed_constraints()

        computes = self.filter(context='compute', **_skip_filter_checks).computes

        for param in self.filter(qualifier='run_checks_compute', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and computes != param._choices:
                choices_changed = True
            param._choices = computes

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_component_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []
        changed_params = self.run_delayed_constraints()

        for param in self.filter(context='figure', qualifier='components', check_default=False, check_visible=False).to_list():
            if param.kind in ['mesh', 'orb']:
                # then we won't have a times array, so we'll have to hardcode the options
                c_same_kind = self.hierarchy.get_meshables()
            elif param.kind in ['lp']:
                c_same_kind = self.filter(qualifier='wavelengths', context='dataset', kind=param.kind, check_visible=False).components
            else:
                c_same_kind = self.filter(qualifier='times', context='dataset', kind=param.kind, check_visible=False).components

            choices_changed = False
            if return_changes and c_same_kind != param._choices:
                choices_changed = True
            param._choices = c_same_kind
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_model_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []
        changed_params = self.run_delayed_constraints()

        for param in self.filter(context='figure', qualifier='models', check_default=False, check_visible=False).to_list():
            ml_same_kind = self.filter(context='model', kind=param.kind).models

            choices_changed = False
            if return_changes and ml_same_kind != param._choices:
                choices_changed = True
            param._choices = ml_same_kind
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_meshcolor_choiceparams(self, return_changes=False):
        """
        """
        affected_params = []
        # changed_params = self.run_delayed_constraints()

        ignore = ['xyz_elements', 'uvw_elements', 'xyz_normals', 'uvw_normals', 'times']

        # we'll cheat by checking in the dataset context to avoid getting the
        # pb-dependent entries with kind='mesh'
        mesh_datasets = self.filter(context='dataset', kind='mesh', **_skip_filter_checks).datasets

        choices = ['None']
        for p in self.filter(context='model', kind='mesh', **_skip_filter_checks).exclude(qualifier=ignore, **_skip_filter_checks).to_list():
            item = p.qualifier if p.dataset in mesh_datasets else '{}@{}'.format(p.qualifier, p.dataset)
            if item not in choices:
                choices.append(item)

        for param in self.filter(context='figure', qualifier=['fc_column', 'ec_column'], **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                param._value = 'None'
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_distribution_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []

        choices = self.distributions

        # NOTE: sample_from@compute currently handled in _handle_computesamplefrom_selectparams
        params = self.filter(context='solver', qualifier=['init_from', 'priors', 'bounds', 'sample_from'], **_skip_filter_checks).to_list()
        distribution_set_choices = ['manual'] + ['{}@{}'.format(p.qualifier, getattr(p, p.context)) for p in params]
        params += self.filter(context='figure', kind='distribution_collection', qualifier='distributions', **_skip_filter_checks).to_list()

        for param in params:
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        for param in self.filter(context='figure', kind='distribution_collection', qualifier='distribution_set', **_skip_filter_checks).to_list():
            # NOTE: these are technically ChoiceParameters
            choices_changed = False
            if return_changes and distribution_set_choices != param._choices:
                choices_changed = True
            param._choices = distribution_set_choices

            if param._value not in distribution_set_choices:
                changed = True
                param._value = 'manual'
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_computesamplefrom_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []

        choices = self.distributions + self.solutions

        for param in self.filter(context='compute', qualifier=['sample_from'], **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_fitparameters_selecttwigparams(self, rename={}, return_changes=False):
        affected_params = []

        params = self.filter(context='solver', qualifier=['fit_parameters'], **_skip_filter_checks).to_list()
        if not len(params):
            return affected_params

        # TODO: should we also check to make sure p.component in [None]+self.hierarchy.get_components()?  If so, we'll need to call this method in set_hierarchy as well.
        choices = []
        for p in self.get_adjustable_parameters(exclude_constrained=False, check_visible=False).to_list():
            if p.__class__.__name__ == 'FloatParameter':
                choices.append(p.twig)
            elif p.__class__.__name__ == 'FloatArrayParameter':
                for i in range(len(p.get_value())):
                    choices.append(p.qualifier+'['+str(i)+']'+'@'+'@'.join(p.twig.split('@')[1:]))
            else:
                raise NotImplementedError()
        # choices = self.get_adjustable_parameters(exclude_constrained=False, check_visible=False).twigs
        for param in params:
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_orbit_choiceparams(self, return_changes=False):
        affected_params = []

        choices = self.filter(context='component', kind='orbit', **_skip_filter_checks).components

        for param in self.filter(qualifier='orbit', context='solver', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                if param._value == 'None' and len(choices):
                    param._value = choices[0]
                else:
                    param._value = self.hierarchy.get_top()
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_component_choiceparams(self, return_changes=False):
        affected_params = []

        # currently assuming we want a component with period (which is the case for lc_periodogram component parameter)
        choices = self.filter(context='component', qualifier='period', **_skip_filter_checks).components

        for param in self.filter(qualifier='component', context='solver', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                param._value = self.hierarchy.get_top()
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_solution_choiceparams(self, return_changes=False):
        affected_params = []

        # currently hardcoded to emcee only
        choices = ['None'] + self.filter(context='solution', kind='emcee', **_skip_filter_checks).solutions

        for param in self.filter(qualifier='continue_from', context='solver', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                # if param._value == 'None' and len(choices):
                #     param._value = choices[0]
                # else:
                param._value = 'None'
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        for param in self.filter(qualifier='solution', context='figure', **_skip_filter_checks).to_list():
            choices = self.filter(context='solution', kind=param.kind, **_skip_filter_checks).solutions

            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                if param._value == '' and len(choices):
                    param._value = choices[0]
                else:
                    param._value = ''
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_solution_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []

        solutions = self.filter(context='solution', **_skip_filter_checks).solutions

        for param in self.filter(qualifier='run_checks_solution', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and solutions != param._choices:
                choices_changed = True
            param._choices = solutions

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_solver_choiceparams(self, return_changes=False):
        affected_params = []

        for param in self.filter(qualifier='solver', context='figure', **_skip_filter_checks).to_list():
            choices = self.filter(context='solver', kind=param.kind, **_skip_filter_checks).solvers

            choices_changed = False
            if return_changes and choices != param._choices:
                choices_changed = True
            param._choices = choices

            if param._value not in choices:
                changed = True
                if param._value == '' and len(choices):
                    param._value = choices[0]
                else:
                    param._value = ''
            else:
                changed = False

            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_solver_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []

        solvers = self.filter(context='solver', **_skip_filter_checks).solvers

        for param in self.filter(qualifier='run_checks_solver', **_skip_filter_checks).to_list():
            choices_changed = False
            if return_changes and solvers != param._choices:
                choices_changed = True
            param._choices = solvers

            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def set_hierarchy(self, *args, **kwargs):
        """
        Set the hierarchy of the system, and recreate/rerun all necessary
        constraints (can be slow).

        For a list of all constraints that are automatically set based on the
        hierarchy, see <phoebe.frontend.bundle.Bundle.add_constraint>.

        See the built-in functions for building hierarchy reprentations:
        * <phoebe.parameters.hierarchy>
        * <phoebe.parameters.hierarchy.binaryorbit>
        * <phoebe.parameters.hierarchy.component>

        See the following tutorials:
        * [building a system](/docs/latest/tutorials/building_a_system)

        Arguments
        -----------
        * `*args`: positional arguments can be any one of the following:
            * valid string representation of the hierarchy
            * callable function (possibly in <phoebe.parameters.hierarchy>)
                followed by arguments that return a valid string representation
                of the hierarchy.
        * `value` (str, optional, only used if no positional arguments provided):
            * valid string representation of the hierarchy
        * `**kwargs`: IGNORED
        """

        if self._import_before_v211:
            raise ValueError("This bundle was created before constraints in solar units were supported and therefore cannot call set_hierarchy.  Either downgrade PHOEBE or re-create this system from scratch if you need to change the hierarchy.")

        # need to run any constraints since some may be deleted and rebuilt
        changed_params = self.run_delayed_constraints()


        _old_param = self.get_hierarchy()

        if len(args) == 1 and isinstance(args[0], str):
            repr_ = args[0]
            kind = None

        elif len(args) == 0:
            if 'value' in kwargs.keys():
                repr_ = kwargs['value']
                kind = None
            else:
                repr_ = self.get_hierarchy().get_value()
                kind = None

        else:
            func = _get_add_func(hierarchy, args[0])
            func_args = args[1:]

            repr_ = func(*func_args)

            kind = func.__name__

        hier_param = HierarchyParameter(value=repr_,
                                        description='Hierarchy representation')

        self.remove_parameters_all(qualifier='hierarchy', context='system')

        metawargs = {'context': 'system'}
        self._attach_params([hier_param], **metawargs)

        # cache hierarchy param so we don't need to do a filter everytime we
        # want to access it in is_visible, etc
        self._hierarchy_param = hier_param

        self._handle_pblum_defaults()
        # self._handle_dataset_selectparams()

        # Handle inter-PS constraints
        starrefs = hier_param.get_stars()
        hier_envelopes = hier_param.get_envelopes()

        # user_interactive_constraints = conf.interactive_constraints
        # conf.interactive_constraints_off()
        for component in [env for env in self.filter(kind='envelope', **_skip_filter_checks).components if env not in hier_envelopes]:
            for constraint_param in self.filter(context='constraint', component=component, **_skip_filter_checks).to_list():
                logger.debug("removing {} constraint (envelope no longer in hierarchy)".format(constraint_param.twig))
                self.remove_constraint(constraint_func=constraint_param.constraint_func, component=component, **_skip_filter_checks)

        for component in hier_envelopes:
            # we need two of the three [comp_env] + self.hierarchy.get_siblings_of(comp_env) to have constraints
            logger.debug('re-creating requiv constraints')
            existing_requiv_constraints = self.filter(constraint_func='requiv_to_pot',
                                                      component=[component]+self.hierarchy.get_siblings_of(component),
                                                      **_skip_filter_checks)
            if len(existing_requiv_constraints) == 2:
                # do we need to rebuild these?
                continue
            elif len(existing_requiv_constraints)==0:
                for component_requiv in self.hierarchy.get_siblings_of(component):
                    pot_parameter = self.get_parameter(qualifier='pot', component=self.hierarchy.get_envelope_of(component_requiv), context='component', **_skip_filter_checks)
                    requiv_parameter = self.get_parameter(qualifier='requiv', component=component_requiv, context='component', **_skip_filter_checks)
                    if len(pot_parameter.constrained_by):
                        solve_for = requiv_parameter.uniquetwig
                    else:
                        solve_for = pot_parameter.uniquetwig

                    self.add_constraint(constraint.requiv_to_pot, component_requiv,
                                        constraint=self._default_label('requiv_to_pot', context='constraint'),
                                        solve_for=solve_for)
            else:
                raise NotImplementedError("expected 0 or 2 existing requiv_to_pot constraints")

            logger.debug('re-creating fillout_factor (contact) constraint for {}'.format(component))
            if len(self.filter(context='constraint',
                               constraint_func='fillout_factor',
                               component=component,
                               **_skip_filter_checks)):
                constraint_param = self.get_constraint(constraint_func='fillout_factor',
                                                       component=component,
                                                       **_skip_filter_checks)
                self.remove_constraint(constraint_func='fillout_factor',
                                       component=component,
                                       **_skip_filter_checks)
                self.add_constraint(constraint.fillout_factor, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.fillout_factor, component,
                                    constraint=self._default_label('fillout_factor', context='constraint'))

            logger.debug('re-creating pot_min (contact) constraint for {}'.format(component))
            if len(self.filter(context='constraint',
                               constraint_func='potential_contact_min',
                               component=component,
                               **_skip_filter_checks)):
                constraint_param = self.get_constraint(constraint_func='potential_contact_min',
                                                       component=component,
                                                       **_skip_filter_checks)
                self.remove_constraint(constraint_func='potential_contact_min',
                                       component=component,
                                       **_skip_filter_checks)
                self.add_constraint(constraint.potential_contact_min, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.potential_contact_min, component,
                                    constraint=self._default_label('pot_min', context='constraint'))

            logger.debug('re-creating pot_max (contact) constraint for {}'.format(component))
            if len(self.filter(context='constraint',
                               constraint_func='potential_contact_max',
                               component=component,
                               **_skip_filter_checks)):
                constraint_param = self.get_constraint(constraint_func='potential_contact_max',
                                                       component=component,
                                                       **_skip_filter_checks)
                self.remove_constraint(constraint_func='potential_contact_max',
                                       component=component,
                                       **_skip_filter_checks)
                self.add_constraint(constraint.potential_contact_max, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.potential_contact_max, component,
                                    constraint=self._default_label('pot_max', context='constraint'))

        for component in self.hierarchy.get_stars():
            if len(starrefs)==1:
                logger.debug('re-creating requiv_single_max (single star) constraint for {}'.format(component))
                if len(self.filter(context='constraint',
                                   constraint_func='requiv_single_max',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='requiv_single_max',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='requiv_single_max',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.requiv_single_max, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.requiv_single_max, component,
                                        constraint=self._default_label('requiv_max', context='constraint'))
            else:
                logger.debug('re-creating mass constraint for {}'.format(component))
                # TODO: will this cause problems if the constraint has been flipped?
                if len(self.filter(context='constraint',
                                   constraint_func='mass',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='mass',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='mass',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.mass, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.mass, component,
                                        constraint=self._default_label('mass', context='constraint'))


                logger.debug('re-creating comp_sma constraint for {}'.format(component))
                # TODO: will this cause problems if the constraint has been flipped?
                if len(self.filter(context='constraint',
                                   constraint_func='comp_sma',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='comp_sma',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='comp_sma',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.comp_sma, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.comp_sma, component,
                                        constraint=self._default_label('comp_sma', context='constraint'))

                logger.debug('re-creating comp_asini constraint for {}'.format(component))
                # TODO: will this cause problems if the constraint has been flipped?
                if len(self.filter(context='constraint',
                                   constraint_func='comp_asini',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='comp_asini',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='comp_asini',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.comp_asini, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.comp_asini, component,
                                        constraint=self._default_label('comp_asini', context='constraint'))

                logger.debug('re-creating rotation_period constraint for {}'.format(component))
                # TODO: will this cause problems if the constraint has been flipped?
                if len(self.filter(context='constraint',
                                   constraint_func='rotation_period',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='rotation_period',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='rotation_period',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.rotation_period, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.rotation_period, component,
                                        constraint=self._default_label('rotation_period', context='constraint'))

                logger.debug('re-creating pitch constraint for {}'.format(component))
                # TODO: will this cause problems if the constraint has been flipped?
                # TODO: what if the user disabled/removed this constraint?
                if len(self.filter(context='constraint',
                                   constraint_func='pitch',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='pitch',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='pitch',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.pitch, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.pitch, component,
                                        constraint=self._default_label('pitch', context='constraint'))

                logger.debug('re-creating yaw constraint for {}'.format(component))
                # TODO: will this cause problems if the constraint has been flipped?
                # TODO: what if the user disabled/removed this constraint?
                if len(self.filter(context='constraint',
                                   constraint_func='yaw',
                                   component=component,
                                   **_skip_filter_checks)):
                    constraint_param = self.get_constraint(constraint_func='yaw',
                                                           component=component,
                                                           **_skip_filter_checks)
                    self.remove_constraint(constraint_func='yaw',
                                           component=component,
                                           **_skip_filter_checks)
                    self.add_constraint(constraint.yaw, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.yaw, component,
                                        constraint=self._default_label('yaw', context='constraint'))


                if self.hierarchy.is_contact_binary(component):
                    # then we're in a contact binary and need to create pot<->requiv constraints
                    # NOTE: pot_min and pot_max are handled above at the envelope level
                    logger.debug('re-creating requiv_detached_max (contact) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_detached_max',
                                       component=component,
                                       **_skip_filter_checks)):
                        # then we're changing from detached to contact so should remove the detached constraint first
                        self.remove_constraint(constraint_func='requiv_detached_max',
                                               component=component,
                                               **_skip_filter_checks)

                    logger.debug('re-creating requiv_contact_max (contact) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_contact_max',
                                       component=component,
                                       **_skip_filter_checks)):
                        constraint_param = self.get_constraint(constraint_func='requiv_contact_max',
                                                               component=component,
                                                               **_skip_filter_checks)
                        self.remove_constraint(constraint_func='requiv_contact_max',
                                               component=component,
                                               **_skip_filter_checks)
                        self.add_constraint(constraint.requiv_contact_max, component,
                                            solve_for=constraint_param.constrained_parameter.uniquetwig,
                                            constraint=constraint_param.constraint)
                    else:
                        self.add_constraint(constraint.requiv_contact_max, component,
                                            constraint=self._default_label('requiv_max', context='constraint'))

                    logger.debug('re-creating requiv_contact_min (contact) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_contact_min',
                                       component=component,
                                       **_skip_filter_checks)):
                        constraint_param = self.get_constraint(constraint_func='requiv_contact_min',
                                                               component=component,
                                                               **_skip_filter_checks)
                        self.remove_constraint(constraint_func='requiv_contact_min',
                                               component=component,
                                               **_skip_filter_checks)
                        self.add_constraint(constraint.requiv_contact_min, component,
                                            solve_for=constraint_param.constrained_parameter.uniquetwig,
                                            constraint=constraint_param.constraint)
                    else:
                        self.add_constraint(constraint.requiv_contact_min, component,
                                            constraint=self._default_label('requiv_min', context='constraint'))

                else:
                    # then we're in a detached/semi-detached system
                    # let's make sure we remove any requiv_to_pot constraints
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_to_pot',
                                       component=component,
                                       **_skip_filter_checks)):
                        self.remove_constraint(constraint_func='requiv_to_pot',
                                               component=component,
                                               **_skip_filter_checks)

                    logger.debug('re-creating requiv_max (detached) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_contact_max',
                                       component=component,
                                       **_skip_filter_checks)):
                        # then we're changing from contact to detached so should remove the detached constraint first
                        self.remove_constraint(constraint_func='requiv_contact_max',
                                               component=component,
                                               **_skip_filter_checks)

                    logger.debug('re-creating requiv_detached_max (detached) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_detached_max',
                                       component=component,
                                       **_skip_filter_checks)):
                        constraint_param = self.get_constraint(constraint_func='requiv_detached_max',
                                                               component=component,
                                                               **_skip_filter_checks)
                        self.remove_constraint(constraint_func='requiv_detached_max',
                                               component=component,
                                               **_skip_filter_checks)
                        self.add_constraint(constraint.requiv_detached_max, component,
                                            solve_for=constraint_param.constrained_parameter.uniquetwig,
                                            constraint=constraint_param.constraint)
                    else:
                        self.add_constraint(constraint.requiv_detached_max, component,
                                            constraint=self._default_label('requiv_max', context='constraint'))



        # rebuild compute_phases in case the top-level orbit has changged
        # TODO: do we ever want to keep this fixed?
        # TODO: do we need to handle the component tag of compute_* parameters?
        # for constraint_param in self.filter(constraint_func='compute_phases', context='constraint').to_list():
        #     logger.debug('re-creating compute_phases constraint {}'.format(constraint_param.dataset))
        #
        #     self.remove_constraint(uniqueid=constraint_param.uniqueid)
        #     self.add_constraint(constraint.compute_phases, dataset=constraint.dataset,
        #                         solve_for=constraint_param.constrained_parameter.uniquetwig,
        #                         constraint=constraint_param.constraint)

        # if user_interactive_constraints:
            # conf.interactive_constraints_on()
            # self.run_delayed_constraints()


        redo_kwargs = {k: v for k, v in hier_param.to_dict().items()
                       if v not in [None, ''] and
                       k not in ['uniqueid', 'uniquetwig', 'twig',
                                 'Class', 'context', 'qualifier',
                                 'description']}
        if _old_param is None:
            # this will fake the undo-ability to raise an error saying it
            # cannot be undone
            undo_kwargs = {'uniqueid': None}
        else:
            undo_kwargs = {k: v for k, v in _old_param.to_dict().items()
                           if v not in [None, ''] and
                           k not in ['uniqueid', 'uniquetwig', 'twig',
                                     'Class', 'context', 'qualifier',
                                     'description']}

        return

    def get_adjustable_parameters(self, exclude_constrained=True, check_visible=True):
        """
        Return a <phoebe.parameters.ParameterSet> of parameters that are
        current adjustable (ie. by a solver).

        Arguments
        ----------
        * `exclude_constrained` (bool, optional, default=True): whether to exclude
            constrained parameters.  This should be `True` if looking for parameters
            that are directly adjustable, but `False` if looking for parameters
            that can have priors placed on them or that could be adjusted if the
            appropriate constraint(s) were flipped.
        * `check_visible` (bool, optional, default=True): whether to check the
            visibility of the parameters (and therefore exclude parameters that
            are not visible).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of parameters
        """
        # TODO: OPTIMIZE profile if its cheaper to do check_visible/default in the filter or in the list comprehension

        # parameters that can be fitted are only in the component or dataset context,
        # must be float parameters and must not be constrained (and must be visible)
        excluded_qualifiers = ['times', 'sigmas', 'fluxes', 'rvs', 'wavelengths', 'flux_densities']
        excluded_qualifiers += ['compute_times', 'compute_phases']
        excluded_qualifiers += ['ra', 'dec', 't0']
        ps = self.filter(context=['component', 'dataset', 'system', 'feature'], check_visible=check_visible, check_default=True)
        return ParameterSet([p for p in ps.to_list() if p.__class__.__name__ in ['FloatParameter', 'FloatArrayParameter'] and not p.readonly and p.qualifier not in excluded_qualifiers and (not exclude_constrained or not len(p.constrained_by))])


    def get_system(self, twig=None, **kwargs):
        """
        Filter in the 'system' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'system'
        return self.filter(**kwargs)

    @property
    def hierarchy(self):
        """
        Property shortcut to <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        Returns
        --------
        * the <phoebe.parameters.HierarchyParameter> or None (if no hierarchy exists)
        """
        return self.get_hierarchy()

    def get_hierarchy(self):
        """
        Get the hierarchy parameter.

        See <phoebe.parameters.HierarchyParameter>, including:
        * <phoebe.parameters.HierarchyParameter.get_components>
        * <phoebe.parameters.HierarchyParameter.get_top>
        * <phoebe.parameters.HierarchyParameter.get_stars>
        * <phoebe.parameters.HierarchyParameter.get_envelopes>
        * <phoebe.parameters.HierarchyParameter.get_orbits>

        Returns
        --------
        * the <phoebe.parameters.HierarchyParameter> or None (if no hierarchy exists)
        """
        return self._hierarchy_param

    def _kwargs_checks(self, kwargs,
                       additional_allowed_keys=[],
                       additional_forbidden_keys=[],
                       warning_only=False,
                       ps=None):
        """
        """
        if ps is None:
            # then check against the entire bundle
            ps = self

        if not len(kwargs.items()):
            return

        allowed_keys = self.qualifiers +\
                        parameters._meta_fields_filter +\
                        ['skip_checks', 'check_default', 'check_visible', 'do_create_fig_params'] +\
                        additional_allowed_keys

        for k in additional_forbidden_keys:
            allowed_keys.remove(k)

        for key,value in kwargs.items():
            if key not in allowed_keys:
                msg = "'{}' not a recognized kwarg".format(key)
                if warning_only:
                    logger.warning(msg)
                else:
                    raise KeyError(msg)

            if isinstance(value, dict):
                for k,v in value.items():
                    self._kwargs_checks({'{}@{}'.format(key, k): v},
                                        additional_allowed_keys=additional_allowed_keys+['{}@{}'.format(key, k)],
                                        additional_forbidden_keys=additional_forbidden_keys,
                                        warning_only=warning_only,
                                        ps=ps
                                        )

                continue

            for param in ps.filter(qualifier=key).to_list():
                if hasattr(param, 'valid_selection'):
                    if not param.valid_selection(value):
                        msg = "{}={} not valid with choices={}".format(key, value, param.choices)
                        if warning_only:
                            logger.warning(msg)
                        else:
                            raise ValueError(msg)
                elif hasattr(param, 'choices'):
                    if value not in param.choices:
                        msg = "{}={} not one of {}".format(key, value, param.choices)
                        if warning_only:
                            logger.warning(msg)
                        else:
                            raise ValueError(msg)

                if hasattr(param, '_check_value'):
                    try:
                        value = param._check_value(value)
                    except:
                        msg = "'{}' not valid for {}".format(value, key)
                        if warning_only:
                            logger.warning(msg)
                        else:
                            raise ValueError(msg)

                elif hasattr(param, '_check_type'):
                    # NOTE: _check_value already called _check_type, thus the elif
                    try:
                        value = param._check_type(value)
                    except:
                        msg = "'{}' not valid for {}".format(value, key)
                        if warning_only:
                            logger.warning(msg)
                        else:
                            raise TypeError(msg)

    def _run_checks_warning_error(self, report, raise_logger_warning, raise_error):
        if raise_logger_warning:
            for item in report.items:
                # passed is either False (failed) or None (raise Warning)
                msg = item.message
                if item.fail:
                    if len(item.affects_methods):
                        msg += "  If not addressed, this warning will continue to be raised and will throw an error at {}.".format(",".join([m for m in item.affects_methods if m!='system']))
                    else:
                        msg += "  If not addressed, this warning will continue to be raised may eventually throw an error."
                logger.warning(msg)

        if raise_error:
            if not report.passed:
                raise ValueError("failed to pass checks\n{}".format(report))

    def run_checks(self, raise_logger_warning=False, raise_error=False, **kwargs):
        """
        Run all available checks.

        This calls and appends the output from each of the following, as applicable:
        * <phoebe.frontend.bundle.Bundle.run_checks_system>
        * <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * <phoebe.frontend.bundle.Bundle.run_checks_solver>
        * <phoebe.frontend.bundle.Bundle.run_checks_solution>
        * <phoebe.frontend.bundle.Bundle.run_checks_figure>

        Arguments
        -----------
        * `compute` (string or list of strings, optional, default=None): the
            compute options to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_compute@setting' parameter
            will be used (which defaults to all available compute options).
        * `solver` (string or list of strings, optional, default=None): the
            solver options to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_solver@setting' parameter
            will be used (which defaults to all available solver options).
        * `solution` (string or list of strings, optional, default=None): the
            solutions to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_solution@setting' parameter
            will be used (which defaults to no solutions, if not set).
        * `figure` (string or list of strings, optional, default=None): the
            figures to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_figure@setting' parameter
            will be used (which defaults to no figures, if not set).
        * `allow_skip_constraints` (bool, optional, default=False): whether
            to allow skipping running delayed constraints if interactive
            constraints are disabled.  See <phoebe.interactive_constraints_off>.
        * `raise_logger_warning` (bool, optional, default=False): whether to
            raise any errors/warnings in the logger (with level of warning).
        * `raise_error` (bool, optional, default=False): whether to raise an
            error if the report has a status of failed.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (<phoebe.frontend.bundle.RunChecksReport>) object containing all
            errors/warnings.  Print the returned object to see all messages.
            See also: <phoebe.frontend.bundle.RunChecksReport.passed>,
             <phoebe.frontend.bundle.RunChecksReport.items>, and
             <phoebe.frontend.bundle.RunChecksItem.message>.
        """
        report = self.run_checks_system(raise_logger_warning=False, raise_error=False, **kwargs)
        report = self.run_checks_compute(run_checks_system=False, raise_logger_warning=False, raise_error=False, report=report, **kwargs)
        # TODO: passing run_checks_compute=False is to try to avoid duplicates with the call above,
        # but could get us into trouble if compute@solver references a compute that is
        # skipped by the run_checks_compute@setting or kwargs.get('compute').
        report = self.run_checks_solver(run_checks_compute=False, raise_logger_warning=False, raise_error=False, report=report, **kwargs)
        report = self.run_checks_solution(raise_logger_warning=False, raise_error=False, report=report, **kwargs)
        report = self.run_checks_figure(raise_logger_warning=False, raise_error=False, report=report, **kwargs)

        self._run_checks_warning_error(report, raise_logger_warning, raise_error)

        return report


    def run_checks_system(self, raise_logger_warning=False, raise_error=False,
                          compute=None, solver=None, solution=None, figure=None,
                          **kwargs):
        """
        Check to see whether the system is expected to be computable.

        This is called by default for each set_value but will only raise a
        logger warning if fails.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_checks>
        * <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * <phoebe.frontend.bundle.Bundle.run_checks_solver>
        * <phoebe.frontend.bundle.Bundle.run_checks_solution>
        * <phoebe.frontend.bundle.Bundle.run_checks_figure>

        Arguments
        -----------
        * `allow_skip_constraints` (bool, optional, default=False): whether
            to allow skipping running delayed constraints if interactive
            constraints are disabled.  See <phoebe.interactive_constraints_off>.
        * `raise_logger_warning` (bool, optional, default=False): whether to
            raise any errors/warnings in the logger (with level of warning).
        * `raise_error` (bool, optional, default=False): whether to raise an
            error if the report has a status of failed.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (<phoebe.frontend.bundle.RunChecksReport>) object containing all
            errors/warnings.  Print the returned object to see all messages.
            See also: <phoebe.frontend.bundle.RunChecksReport.passed>,
             <phoebe.frontend.bundle.RunChecksReport.items>, and
             <phoebe.frontend.bundle.RunChecksItem.message>.
        """

        # make sure all constraints have been run
        if conf.interactive_constraints or not kwargs.pop('allow_skip_constraints', False):
            changed_params = self.run_delayed_constraints()

        report = kwargs.pop('report', RunChecksReport())


        hier = self.hierarchy
        if hier is None:
            return report

        hier_stars = hier.get_stars()
        hier_meshables = hier.get_meshables()
        hier_orbits = hier.get_orbits()

        for component in hier_stars:
            kind = hier.get_kind_of(component) # shouldn't this always be 'star'?
            comp_ps = self.get_component(component=component, **_skip_filter_checks)

            if not len(comp_ps):
                report.add_item(b,
                                "component '{}' in the hierarchy is not in the bundle".format(component)
                                [hier],
                                True, ['system', 'run_compute'])

            parent = hier.get_parent_of(component)
            parent_ps = self.get_component(component=parent, **_skip_filter_checks)
            if kind in ['star']:
                if self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **_skip_filter_checks) >= 10000 and self.get_value(qualifier='ld_mode_bol', component=component, context='component', **_skip_filter_checks) == 'lookup':
                    report.add_item(self,
                                    "ld_mode_bol of 'lookup' uses a bolometric passband which is not reliable for hot stars.  Consider using ld_mode_bol of manual and providing ld_coeffs instead.",
                                    [self.get_parameter(qualifier='teff', component=component, context='component'),
                                     self.get_parameter(qualifier='ld_mode_bol', component=component, context='component')],
                                    False, ['system', 'run_compute'])


                # contact systems MUST by synchronous
                if hier.is_contact_binary(component):
                    if self.get_value(qualifier='syncpar', component=component, context='component', **_skip_filter_checks) != 1.0:
                        report.add_item(self,
                                        "contact binaries must be synchronous, but syncpar@{}!=1".format(component),
                                        [self.get_parameter(qualifier='syncpar', component=component, context='component', **_skip_filter_checks)],
                                        True, ['system', 'run_compute'])

                    if self.get_value(qualifier='ecc', component=parent, context='component', **_skip_filter_checks) != 0.0:
                        # TODO: this can result in duplicate entries in the report
                        report.add_item(self,
                                        "contact binaries must be circular, but ecc@{}!=0".format(parent),
                                        [self.get_parameter(qualifier='ecc', component=parent, context='component', **_skip_filter_checks)],
                                        True, ['system', 'run_compute'])

                    if self.get_value(qualifier='pitch', component=component, context='component', **_skip_filter_checks) != 0.0:
                        report.add_item(self,
                                        'contact binaries must be aligned, but pitch@{}!=0.  Try b.set_value(qualifier=\'pitch\', component=\'{}\' value=0.0, check_visible=False) to align.'.format(component, component),
                                        [self.get_parameter(qualifier='pitch', component=component, context='component', **_skip_filter_checks)],
                                        True, ['system', 'run_compute'])

                    if self.get_value(qualifier='yaw', component=component, context='component', **_skip_filter_checks) != 0.0:
                        report.add_item(self,
                                        'contact binaries must be aligned, but yaw@{}!=0.  Try b.set_value(qualifier=\'yaw\', component=\'{}\', value=0.0, check_visible=False) to align.'.format(component, component),
                                        [self.get_parameter(qualifier='yaw', component=component, context='component', **_skip_filter_checks)],
                                        True, ['system', 'run_compute'])

                # MUST NOT be overflowing at PERIASTRON (d=1-ecc, etheta=0)

                requiv = comp_ps.get_value(qualifier='requiv', unit=u.solRad, **_skip_filter_checks)
                requiv_max = comp_ps.get_value(qualifier='requiv_max', unit=u.solRad, **_skip_filter_checks)



                if hier.is_contact_binary(component):
                    requiv_min = comp_ps.get_value(qualifier='requiv_min')

                    if np.isnan(requiv) or requiv > requiv_max:
                        report.add_item(self,
                                        '{} is overflowing at L2/L3 (requiv={}, requiv_min={}, requiv_max={})'.format(component, requiv, requiv_min, requiv_max),
                                        [comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks),
                                         comp_ps.get_parameter(qualifier='requiv_max', **_skip_filter_checks),
                                         parent_ps.get_parameter(qualifier='sma', **_skip_filter_checks)],
                                        True, ['system', 'run_compute'])

                    if np.isnan(requiv) or requiv <= requiv_min:
                        report.add_item(self,
                                        '{} is underflowing at L1 and not a contact system (requiv={}, requiv_min={}, requiv_max={})'.format(component, requiv, requiv_min, requiv_max),
                                        [comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks),
                                         comp_ps.get_parameter(qualifier='requiv_min', **_skip_filter_checks),
                                         parent_ps.get_parameter(qualifier='sma', **_skip_filter_checks)],
                                        True, ['system', 'run_compute'])

                    elif requiv <= requiv_min * 1.001:
                        report.add_item(self,
                                        'requiv@{} is too close to requiv_min (within 0.1% of critical).  Use detached/semidetached model instead.'.format(component),
                                        [comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks),
                                         comp_ps.get_parameter(qualifier='requiv_min', **_skip_filter_checks),
                                         parent_ps.get_parameter(qualifier='sma', **_skip_filter_checks),
                                         hier],
                                        True, ['system', 'run_compute'])

                else:
                    if requiv > requiv_max:
                        if parent:
                            params = [comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks),
                                     comp_ps.get_parameter(qualifier='requiv_max', **_skip_filter_checks),
                                     parent_ps.get_parameter(qualifier='sma', **_skip_filter_checks)]

                            if parent_ps.get_value(qualifier='ecc', **_skip_filter_checks) > 0.0:
                                params += [parent_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)]

                            if len(self.filter(kind='envelope', context='component', **_skip_filter_checks)):
                                params += [hier]

                            report.add_item(self,
                                            '{} is overflowing at periastron (requiv={}, requiv_max={}).  Use contact model if overflowing is desired.'.format(component, requiv, requiv_max),
                                            params,
                                            True, ['system', 'run_compute'])
                        else:
                            params = comp_ps.filter(qualifier=['requiv', 'requiv_max', 'mass', 'period'], **_skip_filter_checks)

                            report.add_item(self,
                                            '{} is beyond critical rotation (requiv={}, requiv_max={}).'.format(component, requiv, requiv_max),
                                            params,
                                            True, ['system', 'run_compute'])

            else:
                raise NotImplementedError("checks not implemented for type '{}'".format(kind))

        # we also need to make sure that stars don't overlap each other
        # so we'll check for each pair of stars (see issue #70 on github)
        # TODO: rewrite overlap checks
        for orbitref in []: #hier.get_orbits():
            if len(hier.get_children_of(orbitref)) == 2:
                q = self.get_value(qualifier='q', component=orbitref, context='component', **_skip_filter_checks)
                ecc = self.get_value(qualifier='ecc', component=orbitref, context='component', **_skip_filter_checks)

                starrefs = hier.get_children_of(orbitref)
                if hier.get_kind_of(starrefs[0]) != 'star' or hier.get_kind_of(starrefs[1]) != 'star':
                    # print "***", hier.get_kind_of(starrefs[0]), hier.get_kind_of(starrefs[1])
                    continue
                if self.get_value(qualifier='pitch', component=starrefs[0], **_skip_filter_checks)!=0.0 or \
                        self.get_value(qualifier='pitch', component=starrefs[1], **_skip_filter_checks)!=0.0 or \
                        self.get_value(qualifier='yaw', component=starrefs[0], **_skip_filter_checks)!=0.0 or \
                        self.get_value(qualifier='yaw', component=starrefs[1], **_skip_filter_checks)!=0.0:

                    # we cannot run this test for misaligned cases
                   continue

                comp0 = hier.get_primary_or_secondary(starrefs[0], return_ind=True)
                comp1 = hier.get_primary_or_secondary(starrefs[1], return_ind=True)
                q0 = roche.q_for_component(q, comp0)
                q1 = roche.q_for_component(q, comp1)

                F0 = self.get_value(qualifier='syncpar', component=starrefs[0], context='component', **_skip_filter_checks)
                F1 = self.get_value(qualifier='syncpar', component=starrefs[1], context='component', **_skip_filter_checks)

                pot0 = self.get_value(qualifier='pot', component=starrefs[0], context='component', **_skip_filter_checks)
                pot0 = roche.pot_for_component(pot0, q0, comp0)

                pot1 = self.get_value(qualifier='pot', component=starrefs[1], context='component', **_skip_filter_checks)
                pot1 = roche.pot_for_component(pot1, q1, comp1)

                xrange0 = libphoebe.roche_xrange(q0, F0, 1.0-ecc, pot0+1e-6, choice=0)
                xrange1 = libphoebe.roche_xrange(q1, F1, 1.0-ecc, pot1+1e-6, choice=0)

                if xrange0[1]+xrange1[1] > 1.0-ecc:
                    report.add_item(self,
                                    'components in {} are overlapping at periastron (change ecc@{}, syncpar@{}, or syncpar@{}).'.format(orbitref, orbitref, starrefs[0], starrefs[1]),
                                    [ecc, F0, F1],
                                    True, ['system', 'run_compute'])


        # forbid pblum_mode='dataset-coupled' if no other valid datasets
        # forbid pblum_mode='dataset-coupled' with a dataset which is scaled to data or to another that is in-turn color-coupled
        for param in self.filter(qualifier='pblum_mode', value='dataset-coupled', **_skip_filter_checks).to_list():
            coupled_to = self.get_value(qualifier='pblum_dataset', dataset=param.dataset, **_skip_filter_checks)
            if coupled_to == '':
                coupled_to = None
                if param.is_visible:
                    report.add_item(self,
                                    "cannot set pblum_mode@{}='dataset-coupled' when there are no other valid datasets.  Change pblum_mode or add another dataset.".format(param.dataset),
                                    [param],
                                    True, ['system', 'run_compute'])

            pblum_mode = self.get_value(qualifier='pblum_mode', dataset=coupled_to, **_skip_filter_checks)
            if pblum_mode =='dataset-coupled':
                report.add_item(self,
                                "cannot set pblum_dataset@{}='{}' as that dataset has pblum_mode@{}='dataset-coupled'.  Perhaps set to '{}' instead.".format(param.dataset, coupled_to, coupled_to, self.get_value(qualifier='pblum_dataset', dataset=coupled_to, context='dataset', **_skip_filter_checks)),
                                [param,
                                self.get_parameter(qualifier='pblum_mode', dataset=coupled_to, **_skip_filter_checks)],
                                True, ['system', 'run_compute'])

        # require any pblum_mode == 'dataset-scaled' to have accompanying data
        for param in self.filter(qualifier='pblum_mode', value='dataset-scaled', **_skip_filter_checks).to_list():
            if not len(self.get_value(qualifier='fluxes', dataset=param.dataset, context='dataset', **_skip_filter_checks)):
                report.add_item(self,
                                "fluxes@{} cannot be empty if pblum_mode@{}='dataset-scaled'".format(param.dataset, param.dataset),
                                [param,
                                self.get_parameter(qualifier='fluxes', dataset=param.dataset, context='dataset', **_skip_filter_checks)],
                                True, ['system', 'run_compute'])

            # also check to make sure that we'll be able to handle the interpolation in time if the system is time-dependent
            if self.hierarchy.is_time_dependent(consider_gaussian_process=False):
                compute_times = self.get_value(qualifier='compute_times', dataset=param.dataset, context='dataset', **_skip_filter_checks)
                times = self.get_value(qualifier='times', dataset=param.dataset, context='dataset', **_skip_filter_checks)
                if len(times) and len(compute_times) and (min(times) < min(compute_times) or max(times) > max(compute_times)):

                    params = [self.get_parameter(qualifier='pblum_mode', dataset=param.dataset, **_skip_filter_checks),
                              self.get_parameter(qualifier='times', dataset=param.dataset, context='dataset', **_skip_filter_checks),
                              self.get_parameter(qualifier='compute_times', dataset=param.dataset, context='dataset', **_skip_filter_checks)]

                    msg = "'compute_times@{}' must cover full range of 'times@{}', for time-dependent systems with pblum_mode@{}='dataset-scaled'.".format(param.dataset, param.dataset, param.dataset)
                    if len(self.get_parameter(qualifier='compute_phases', dataset=param.dataset, context='dataset', **_skip_filter_checks).constrains):
                        msg += " Consider flipping the 'compute_phases' constraint and providing 'compute_times' instead."
                        params += [self.get_parameter(qualifier='compute_phases', dataset=param.dataset, context='dataset', **_skip_filter_checks),
                                   self.get_constraint(qualifier='compute_times', dataset=param.dataset, **_skip_filter_checks)]

                    report.add_item(self,
                                    msg,
                                    params,
                                    True, ['system', 'run_compute'])

        # tests for lengths of fluxes, rvs, etc vs times (and fluxes vs wavelengths for spectral datasets)
        for param in self.filter(qualifier=['times', 'fluxes', 'rvs', 'sigmas', 'wavelengths', 'flux_densities'], context='dataset', **_skip_filter_checks).to_list():
            shape = param.get_value().shape
            if len(shape) > 1:
                report.add_item(self,
                                "{}@{} must be a flat array, got shape {}".format(param.qualifier, param.dataset, shape),
                                [param],
                                True, ['system', 'run_compute'])

            if param.qualifier in ['fluxes', 'rvs', 'sigmas'] and shape[0] > 0 and shape[0] != self.get_value(qualifier='times', dataset=param.dataset, component=param.component, context='dataset', **_skip_filter_checks).shape[0]:
                tparam = self.get_parameter(qualifier='times', dataset=param.dataset, component=param.component, context='dataset', **_skip_filter_checks)
                report.add_item(self,
                                "{} must be of same length as {}".format(param.twig, tparam.twig),
                                [param, tparam],
                                True, ['system', 'run_compute'])

            if param.qualifier in ['flux_densities'] and shape[0] > 0 and shape[0] != self.get_value(qualifier='wavelengths', dataset=param.dataset, component=param.component,  context='dataset', **_skip_filter_checks).shape[0]:
                # NOTE: flux_densities is time-dependent, but wavelengths is not
                wparam = self.get_parameter(qualifier='wavelengths', dataset=param.dataset, component=param.component, context='dataset', **_skip_filter_checks)
                report.add_item(self,
                                "{}@{}@{} must be of same length as {}@{}".format(param.twig, wparam.twig),
                                [param, wparam],
                                True, ['system', 'run_compute'])



        try:
            self.run_failed_constraints()
        except:
            report.add_item(self,
                            "constraints {} failed to run.  Address errors and try again.  Call run_failed_constraints to see the tracebacks.".format([p.twig for p in self.filter(uniqueid=self._failed_constraints).to_list()]),
                            self.filter(uniqueid=self._failed_constraints).to_list(),
                            True, ['system', 'run_compute'])



        #### WARNINGS ONLY ####
        # let's check teff vs gravb_bol and irrad_frac_refl_bol
        for component in hier_stars:
            teff = self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **_skip_filter_checks)
            gravb_bol = self.get_value(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)

            if teff >= 8000. and gravb_bol < 0.9:
                report.add_item(self,
                                "'{}' probably has a radiative atm (teff={:.0f}K>8000K), for which gravb_bol>=0.9 might be a better approx than gravb_bol={:.2f}.".format(component, teff, gravb_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)],
                                False, ['system', 'run_compute'])
            elif teff <= 6600. and gravb_bol >= 0.9:
                report.add_item(self,
                                "'{}' probably has a convective atm (teff={:.0f}K<6600K), for which gravb_bol<0.9 (suggestion: 0.32) might be a better approx than gravb_bol={:.2f}.".format(component, teff, gravb_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)],
                                False, ['system', 'run_compute'])
            elif (teff > 6600 and teff < 8000) and gravb_bol < 0.32 or gravb_bol > 1.00:
                report.add_item(self,
                                "'{}' has intermittent temperature (6600K<teff={:.0f}K<8000K), gravb_bol might be better between 0.32-1.00 than gravb_bol={:.2f}.".format(component, teff, gravb_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)],
                                False, ['system', 'run_compute'])

        for component in hier_stars:
            teff = self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **_skip_filter_checks)
            irrad_frac_refl_bol = self.get_value(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)

            if teff >= 8000. and irrad_frac_refl_bol < 0.8:
                report.add_item(self,
                                "'{}' probably has a radiative atm (teff={:.0f}K>=8000K), for which irrad_frac_refl_bol>0.8 (suggestion: 1.0) might be a better approx than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)],
                                False, ['system', 'run_compute'])
            elif teff <= 6600. and irrad_frac_refl_bol >= 0.75:
                report.add_item(self,
                                "'{}' probably has a convective atm (teff={:.0f}K<=6600K), for which irrad_frac_refl_bol<0.75 (suggestion: 0.6) might be a better approx than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)],
                                False, ['system', 'run_compute'])
            elif (teff > 6600. and teff < 8000) and irrad_frac_refl_bol < 0.6:
                report.add_item(self,
                                "'{}' has intermittent temperature (6600K<teff={:.0f}K<8000K), irrad_frac_refl_bol might be better between 0.6-1.00 than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)],
                                False, ['system', 'run_compute'])

        # warning if any t0_supconj is more than 10 cycles from t0@system if time dependent
        if hier.is_time_dependent():
            t0_system = self.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)
            for param in self.filter(qualifier='t0_supconj', component=hier_orbits, context='component', **_skip_filter_checks).to_list():
                norbital_cycles = abs(param.get_value(unit=u.d) - t0_system)  / self.get_value(qualifier='period', component=param.component, context='component', unit=u.d, **_skip_filter_checks)
                if norbital_cycles > 10:
                    report.add_item(self,
                                    "{}@{} is ~{} orbital cycles from t0@system, which could cause precision issues for time-dependent systems".format(param.qualifier, param.component, int(norbital_cycles)),
                                    [param, self.get_parameter(qualifier='t0', context='system', **_skip_filter_checks)],
                                    False, ['system', 'run_compute'])

        # TODO: add other checks
        # - make sure all ETV components are legal
        # - check for conflict between dynamics_method and mesh_method (?)

        self._run_checks_warning_error(report, raise_logger_warning, raise_error)

        return report

    def run_checks_compute(self, compute=None, solver=None, solution=None, figure=None,
                         raise_logger_warning=False, raise_error=False, run_checks_system=True, **kwargs):
        """
        Check to see whether the system is expected to be computable.

        This is called by default for each set_value but will only raise a
        logger warning if fails.  This is also called immediately when calling
        <phoebe.frontend.bundle.Bundle.run_compute>.

        kwargs are passed to override currently set values as if they were
        sent to <phoebe.frontend.bundle.Bundle.run_compute>.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_checks>
        * <phoebe.frontend.bundle.Bundle.run_checks_system>
        * <phoebe.frontend.bundle.Bundle.run_checks_solver>
        * <phoebe.frontend.bundle.Bundle.run_checks_solution>
        * <phoebe.frontend.bundle.Bundle.run_checks_figure>

        Arguments
        -----------
        * `compute` (string or list of strings, optional, default=None): the
            compute options to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_compute@setting' parameter
            will be used (which defaults to all available compute options).
        * `run_checks_system` (bool, optional, default=True): whether to also
            call (and include the output from) <phoebe.frontend.bundle.run_checks_system>.
        * `allow_skip_constraints` (bool, optional, default=False): whether
            to allow skipping running delayed constraints if interactive
            constraints are disabled.  See <phoebe.interactive_constraints_off>.
        * `raise_logger_warning` (bool, optional, default=False): whether to
            raise any errors/warnings in the logger (with level of warning).
        * `raise_error` (bool, optional, default=False): whether to raise an
            error if the report has a status of failed.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (<phoebe.frontend.bundle.RunChecksReport>) object containing all
            errors/warnings.  Print the returned object to see all messages.
            See also: <phoebe.frontend.bundle.RunChecksReport.passed>,
             <phoebe.frontend.bundle.RunChecksReport.items>, and
             <phoebe.frontend.bundle.RunChecksItem.message>.
        """
        kwargs = _deepcopy(kwargs)

        # make sure all constraints have been run
        if conf.interactive_constraints or not kwargs.pop('allow_skip_constraints', False):
            changed_params = self.run_delayed_constraints()

        report = kwargs.pop('report', RunChecksReport())
        addl_parameters = kwargs.pop('addl_parameters', [])

        if run_checks_system:
            report = self.run_checks_system(raise_logger_warning=False, raise_error=False, report=report, **kwargs)

        run_checks_compute = self.get_value(qualifier='run_checks_compute', context='setting', default='*', expand=True, **_skip_filter_checks)
        if compute is None:
            computes = run_checks_compute
            addl_parameters += [self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)]
        else:
            computes = compute
            if isinstance(computes, str):
                computes = [computes]

        for compute in computes:
            if compute not in self.computes:
                raise ValueError("compute='{}' not found".format(compute))

            if compute not in run_checks_compute:
                report.add_item(self,
                                "compute='{}' is not included in run_checks_compute@setting, so will not raise interactive warnings".format(compute),
                                [self.get_parameter(qualifier='run_checks_compute', context='setting', check_visible=False, check_default=False)],
                                False
                                )

        hier = self.hierarchy
        if hier is None:
            return report

        hier_stars = hier.get_stars()
        hier_meshables = hier.get_meshables()


        kwargs.setdefault('check_visible', False)
        kwargs.setdefault('check_default', False)

        # run passband checks
        all_pbs = list_passbands(full_dict=True)
        online_pbs = list_online_passbands(full_dict=True)

        pb_needs_ext = self.get_value(qualifier='ebv', context='system', **_skip_filter_checks) != 0

        for pbparam in self.filter(qualifier='passband', **_skip_filter_checks).to_list():

            # we include this in the loop so that we get the most recent dict
            # if a previous passband had to be updated
            installed_pbs = list_installed_passbands(full_dict=True)

            pb = pbparam.get_value()

            pb_needs_Inorm = True
            pb_needs_Imu = True
            pb_needs_ld = True #np.any([p.get_value()!='interp' for p in self.filter(qualifier='ld_mode', dataset=pbparam.dataset, context='dataset', **_skip_filter_checks).to_list()])
            pb_needs_ldint = True

            missing_pb_content = []

            if pb_needs_ext and pb in ['Stromgren:u', 'Johnson:U', 'SDSS:u', 'SDSS:uprime']:
                # need to check for bugfix in coefficients from 2.3.4 release
                installed_timestamp = installed_pbs.get(pb, {}).get('timestamp', None)
                if _timestamp_to_dt(installed_timestamp) < _timestamp_to_dt("Mon Nov 2 00:00:00 2020"):
                    report.add_item(self,
                                    "'{}' passband ({}) with extinction needs to be updated for fixed UV extinction coefficients.  Run phoebe.list_passband_online_history('{}') to get a list of available changes and phoebe.update_passband('{}') or phoebe.update_all_passbands() to update.".format(pb, pbparam.twig, pb, pb),
                                    [pbparam, self.get_parameter(qualifier='ebv', context='system', **_skip_filter_checks)],
                                    True, 'run_compute')

            # NOTE: atms are not attached to datasets, but per-compute and per-component
            for atmparam in self.filter(qualifier='atm', kind='phoebe', compute=computes, **_skip_filter_checks).to_list() + self.filter(qualifier='ld_coeffs_source').to_list():

                # check to make sure passband supports the selected atm
                atm = atmparam.get_value(**_skip_filter_checks)
                if atmparam.qualifier == 'ld_coeffs_source' and atm == 'auto':
                    # this might get us in trouble, we might have to reproduce
                    # the auto logic here to make sure we have the necessary
                    # tables for ld lookup
                    # let's at least make sure we have the necessary ck2004 axes
                    atm = 'ck2004'

                if atm not in installed_pbs.get(pb, {}).get('atms', []):
                    if atm in online_pbs.get(pb, {}).get('atms', []):
                        missing_pb_content += ['{}:Inorm'.format(atm)]
                    else:
                        report.add_item(self,
                                        "'{}' passband ({}) does not support atm='{}' ({}).".format(pb, pbparam.twig, atm, atmparam.twig),
                                        [pbparam, atmparam],
                                        True)

                for check,content in [(pb_needs_Inorm, '{}:Inorm'.format(atm)),
                                      (pb_needs_Imu and atm not in ['extern_planckint', 'extern_atmx', 'blackbody'], '{}:Imu'.format(atm)),
                                      (pb_needs_ld and atm not in ['extern_planckint', 'extern_atmx', 'blackbody'], '{}:ld'.format(atm)),
                                      (pb_needs_ldint and atm not in ['extern_planckint', 'extern_atmx', 'blackbody'], '{}:ldint'.format(atm)),
                                      (pb_needs_ext, '{}:ext'.format(atm)),
                                      ]:
                    if not check: continue

                    if content not in installed_pbs.get(pb, {}).get('content', []):
                        if content in online_pbs.get(pb, {}).get('content', []):
                            missing_pb_content += [content]
                        else:
                            report.add_item(self,
                                            "'{}' passband ({}) does not support {} ({}).".format(pb, pbparam.twig, content, atmparam.twig),
                                            [pbparam, atmparam],
                                            True, 'run_compute')


            # remove any duplicates
            missing_pb_content = list(set(missing_pb_content))
            if len(missing_pb_content):
                installed_timestamp = installed_pbs.get(pb, {}).get('timestamp', None)
                online_timestamp = online_pbs.get(pb, {}).get('timestamp', None)
                if pb not in installed_pbs.keys():
                    # then there is no local version of the passband, so we'll
                    # download the latest version
                    logger.warning("downloading and installing {} passband with content={}".format(pb, missing_pb_content))
                    try:
                        download_passband(pb, content=missing_pb_content)
                    except IOError:
                        report.add_item(self,
                                        'Attempt to download {} passband failed.  Check internet connection, wait for tables.phoebe-project.org to come back online, or try another passband.'.format(pb),
                                        [pbparam],
                                        True, 'run_compute')
                elif conf.update_passband_ignore_version or installed_timestamp == online_timestamp:
                    # NOTE: because of the bug in https://github.com/phoebe-project/phoebe2/issues/585
                    # and https://github.com/phoebe-project/phoebe2/pull/411, we'll
                    # compare the strings directly instead of converting to datetime objects
                    # (since we don't need > logic)
                    if installed_timestamp == online_timestamp:
                        # then the same version already exists locally, so we
                        # can safely update to get the new content
                        logger.warning("updating installed {} passband (with matching online timestamp) to include content={}".format(pb, missing_pb_content))
                    else:
                        # then a DIFFERENT version exists locally than available
                        # online, but the update_passband_ignore_version allows
                        # us to update automatically.
                        logger.warning("updating installed {} passband (ignoring timestamp mismatch) to include content={}".format(pb, missing_pb_content))

                    try:
                        update_passband(pb, content=missing_pb_content)
                    except IOError:
                        report.add_item(self,
                                        'Attempt to update {} passband for the {} tables failed.  Check internet connection, wait for tables.phoebe-project.org to come back online, or try another passband.'.format(pb, missing_pb_content),
                                        [pbparam],
                                        True, 'run_compute')
                else:
                    report.add_item(self,
                                    'installed passband "{}" is missing the following tables: {}. The available online version ({}) is newer than the installed version ({}), so will not be updated automatically.  Call phoebe.update_passband("{}", content={}) or phoebe.update_all_passbands() to update to the latest version.  Set phoebe.update_passband_ignore_version_on() to ignore version mismatches and update automatically.'.format(pb, missing_pb_content, installed_timestamp, online_timestamp, pb, atm, missing_pb_content),
                                    [pbparam],
                                    True, 'run_compute')

        # check length of ld_coeffs vs ld_func and ld_func vs atm
        def ld_coeffs_len(ld_func, ld_coeffs):
            # current choices for ld_func are:
            # ['uniform', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power', 'claret', 'hillen', 'prsa']
            expected_lengths = {'linear': 1, 'logarithmic': 2, 'square_root': 2, 'quadratic': 2, 'power': 4}

            if ld_coeffs is None or len(ld_coeffs) == expected_lengths.get(ld_func):
                return True,
            else:
                return False, "ld_coeffs={} wrong length (expecting length {} instead of {}) for ld_func='{}'.".format(ld_coeffs, expected_lengths.get(ld_func), len(ld_coeffs), ld_func)

        irrad_enabled = kwargs.get('irrad_method', True) != 'none' and np.any([p.get_value()!='none' for p in self.filter(qualifier='irrad_method', compute=computes, **kwargs).to_list()])
        for component in hier_stars:
            if irrad_enabled:
                # first check ld_coeffs_bol vs ld_func_bol
                ld_mode = self.get_value(qualifier='ld_mode_bol', component=component, context='component', **_skip_filter_checks)
                ld_func = str(self.get_value(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks))
                ld_coeffs_source = self.get_value(qualifier='ld_coeffs_source_bol', component=component, context='component', **_skip_filter_checks)
                ld_coeffs = self.get_value(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)

                if np.any(np.isnan(ld_coeffs)):
                    if ld_mode == 'lookup':
                        report.add_item(self,
                                        'ld_mode_bol=\'lookup\' resulted in nans for ld_coeffs_bol.  Check system parameters to be within grids or change ld_mode_bol to \'manual\' and provide ld_coeffs_bol',
                                        [self.get_parameter(qualifier='ld_mode_bol', component=component, context='component', **_skip_filter_checks),
                                        self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                        self.get_parameter(qualifier='logg', component=component, context='component', **_skip_filter_checks),
                                        self.get_parameter(qualifier='abun', component=component, context='component', **_skip_filter_checks)
                                        ],
                                        True, 'run_compute')
                    elif ld_mode == 'manual':
                        report.add_item(self,
                                        'nans in ld_coeffs_bol are forbidden',
                                        [self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)],
                                        True, 'run_compute')
                    else:
                        # if interp, then the previously set value won't be used anyways, so we'll ignore nans
                        pass

                if ld_mode == 'lookup':
                    if ld_coeffs_source != 'auto' and ld_coeffs_source not in all_pbs.get('Bolometric:900-40000', {}).get('atms_ld', []):
                        report.add_item(self,
                                        'Bolometric:900-40000 does not support ld_coeffs_source_bol={}.  Either change ld_coeffs_source_bol@{}@component or ld_mode_bol@{}@component'.format(pb, ld_coeffs_source, component, component),
                                        [self.get_parameter(qualifier='ld_coeffs_source_bol', component=component, context='component', **_skip_filter_checks),
                                         self.get_parameter(qualifier='ld_mode_bol', component=component, context='component', **_skip_filter_checks)
                                        ],
                                        True, 'run_compute')
                elif ld_mode == 'manual':

                    check = ld_coeffs_len(ld_func, ld_coeffs)
                    if not check[0]:
                        report.add_item(self,
                                        check[1],
                                        [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                         self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)
                                        ],
                                        True, 'run_compute')

                    else:
                        check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=False)
                        if not check:
                            report.add_item(self,
                                            'ld_coeffs_bol={} not compatible for ld_func_bol=\'{}\'.'.format(ld_coeffs, ld_func),
                                            [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                             self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)
                                            ],
                                            True, 'run_compute')

                        else:
                            # only need to do the strict check if the non-strict checks passes
                            check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=True)
                            if not check:
                                report.add_item(self,
                                                'ld_coeffs_bol={} result in limb-brightening which is not allowed for irradiation.'.format(ld_coeffs),
                                                [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                                 self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)
                                                ],
                                                True, 'run_compute')

                for compute in computes:
                    if self.get_compute(compute, **_skip_filter_checks).kind in ['legacy'] and ld_func not in ['linear', 'logarithmic', 'square_root']:
                        report.add_item(self,
                                        "ld_func_bol='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', or 'square_root'.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                        [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                         self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)]+addl_parameters,
                                        True, 'run_compute')
                    # other compute backends ignore bolometric limb-darkening


            for dataset in self.filter(context='dataset', kind=['lc', 'rv'], check_default=True).datasets:
                if dataset=='_default':
                    continue
                dataset_ps = self.get_dataset(dataset=dataset, check_visible=False)

                ld_mode = dataset_ps.get_value(qualifier='ld_mode', component=component, **_skip_filter_checks)
                ld_func = dataset_ps.get_value(qualifier='ld_func', component=component, **_skip_filter_checks)
                ld_coeffs_source = dataset_ps.get_value(qualifier='ld_coeffs_source', component=component, **_skip_filter_checks)
                ld_coeffs = dataset_ps.get_value(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                pb = dataset_ps.get_value(qualifier='passband', **kwargs)

                if np.any(np.isnan(ld_coeffs)):
                    if ld_mode == 'lookup':
                        report.add_item(self,
                                        'ld_mode=\'lookup\' resulted in nans for ld_coeffs.  Check system parameters to be within grids or change ld_mode to \'manual\' and provide ld_coeffs',
                                        [dataset_ps.get_parameter(qualifier='ld_mode', component=component, **kwargs),
                                        self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                        self.get_parameter(qualifier='logg', component=component, context='component', **_skip_filter_checks),
                                        self.get_parameter(qualifier='abun', component=component, context='component', **_skip_filter_checks)
                                        ],
                                        True, 'run_compute')
                    elif ld_mode == 'manual':
                        report.add_item(self,
                                        'nans in ld_coeffs are forbidden',
                                        [dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)],
                                        True, 'run_compute')
                    else:
                        # if interp, then the previously set value won't be used anyways, so we'll ignore nans
                        pass

                if ld_mode == 'interp':
                    for compute in computes:
                        # TODO: should we ignore if the dataset is disabled?
                        compute_kind = self.get_compute(compute=compute, **_skip_filter_checks).kind
                        if compute_kind != 'phoebe':
                            report.add_item(self,
                                            "ld_mode='interp' not supported by '{}' backend used by compute='{}'.  Change ld_mode@{}@{}.".format(compute_kind, compute, component, dataset),
                                            [dataset_ps.get_parameter(qualifier='ld_mode', component=component, **_skip_filter_checks)
                                            ]+addl_parameters,
                                            True, 'run_compute')
                        else:
                            atm = self.get_value(qualifier='atm', component=component, compute=compute, context='compute', atm=kwargs.get('atm', None), **_skip_filter_checks)
                            if atm not in ['ck2004', 'phoenix']:
                                if 'ck2004' in self.get_parameter(qualifier='atm', component=component, compute=compute, context='compute', atm=kwargs.get('atm', None), **_skip_filter_checks).choices:
                                    report.add_item(self,
                                                    "ld_mode='interp' not supported by atm='{}'.  Either change atm@{}@{} or ld_mode@{}@{}.".format(atm, component, compute, component, dataset),
                                                    [dataset_ps.get_parameter(qualifier='ld_mode', component=component, **_skip_filter_checks),
                                                     self.get_parameter(qualifier='atm', component=component, compute=compute, context='compute', **_skip_filter_checks)
                                                    ]+addl_parameters,
                                                    True, 'run_compute')
                                else:
                                    report.add_item(self,
                                                    "ld_mode='interp' not supported by '{}' backend used by compute='{}'.  Change ld_mode@{}@{} or use a backend that supports atm='ck2004'.".format(self.get_compute(compute).kind, compute, component, dataset),
                                                    [dataset_ps.get_parameter(qualifier='ld_mode', component=component, **_skip_filter_checks)
                                                    ]+addl_parameters,
                                                    True, 'run_compute')


                elif ld_mode == 'lookup':
                    if ld_coeffs_source != 'auto' and ld_coeffs_source not in all_pbs.get(pb, {}).get('atms_ld', []) :
                        report.add_item(self,
                                        'passband={} does not support ld_coeffs_source={}.  Either change ld_coeffs_source@{}@{} or ld_mode@{}@{}'.format(pb, ld_coeffs_source, component, dataset, component, dataset),
                                        [dataset_ps.get_parameter(qualifier='ld_coeffs_source', component=component, **_skip_filter_checks),
                                         dataset_ps.get_parameter(qualifier='ld_mode', component=component, **_skip_filter_checks)
                                        ],
                                        True, 'run_compute')


                elif ld_mode == 'manual':
                    check = ld_coeffs_len(ld_func, ld_coeffs)
                    if not check[0]:
                        report.add_item(self,
                                        check[1],
                                        [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                         dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                                        ],
                                        True, 'run_compute')

                    else:
                        check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=False)
                        if not check:
                            report.add_item(self,
                                            'ld_coeffs={} not compatible for ld_func=\'{}\'.'.format(ld_coeffs, ld_func),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                             dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                                            ],
                                            True, 'run_compute')

                        else:
                            # only need to do the strict check if the non-strict checks passes
                            check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=True)
                            if not check:
                                report.add_item(self,
                                                'ld_coeffs={} result in limb-brightening.  Use with caution.'.format(ld_coeffs),
                                                [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                                 dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                                                 ],
                                                 False, 'run_compute')

                else:
                    raise NotImplementedError("checks for ld_mode='{}' not implemented".format(ld_mode))

                if ld_mode in ['lookup', 'manual']:
                    for compute in computes:
                        compute_kind = self.get_compute(compute, **_skip_filter_checks).kind
                        if compute_kind in ['legacy'] and ld_func not in ['linear', 'logarithmic', 'square_root']:
                            report.add_item(self,
                                            "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', or 'square_root'.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks)]+addl_parameters,
                                            True, 'run_compute')

                        if compute_kind in ['ellc'] and ld_func not in ['linear', 'logarithmic', 'square_root', 'quadratic', 'power']:
                            report.add_item(self,
                                            "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', 'quadratic', or 'square_root' or power.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks)]+addl_parameters,
                                            True, 'run_compute')

                        if compute_kind in ['jktebop'] and ld_func not in ['linear', 'logarithmic', 'square_root', 'quadratic']:
                            report.add_item(self,
                                            "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', 'quadratic', or 'square_root'.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks)]+addl_parameters,
                                            True, 'run_compute')

                atm = self.get_value(qualifier='atm', component=component, compute=compute, context='compute', atm=kwargs.get('atm', None), **_skip_filter_checks)
                pblum_method = self.get_value(qualifier='pblum_method', compute=compute, context='compute', pblum_method=kwargs.get('pblum_method', None), default='phoebe', **_skip_filter_checks)
                if atm=='blackbody' and pblum_method=='stefan-boltzmann':
                    report.add_item(self,
                                    "pblum_method@{}='stefan-boltzmann' not supported with atm@{}='blackbody'".format(compute, component),
                                    self.filter(qualifier='atm', component=component, compute=compute, context='compute', **_skip_filter_checks)+
                                    self.filter(qualifier='pblum_method', compute=compute, context='compute', **_skip_filter_checks),
                                    True, 'run_compute')


        def _get_proj_area(comp):
            if self.hierarchy.get_kind_of(comp)=='envelope':
                return np.sum([_get_proj_area(c) for c in self.hierarchy.get_siblings_of(comp)])
            else:
                return np.pi*self.get_value(qualifier='requiv', component=comp, context='component', unit='solRad', **_skip_filter_checks)**2

        def _get_surf_area(comp):
            if self.hierarchy.get_kind_of(comp)=='envelope':
                return np.sum([_get_surf_area(c) for c in self.hierarchy.get_siblings_of(comp)])
            else:
                return 4*np.pi*self.get_value(qualifier='requiv', component=comp, context='component', unit='solRad', **_skip_filter_checks)**2


        for compute in computes:
            compute_kind = self.get_compute(compute=compute, **_skip_filter_checks).kind

            gps = self.filter(kind='gaussian_process', context='feature', **_skip_filter_checks).features
            compute_enabled_gps = self.filter(qualifier='enabled', feature=gps, value=True, **_skip_filter_checks).features
            compute_enabled_datasets = self.filter(qualifier='enabled', dataset=self.datasets, value=True, **_skip_filter_checks).datasets
            compute_enabled_datasets_with_gps = [ds for ds in self.filter(qualifier='enabed', feature=gps, value=True, **_skip_filter_checks).datasets if ds in compute_enabled_datasets]

            # per-compute hierarchy checks
            if len(self.hierarchy.get_envelopes()):
                if compute_kind not in ['phoebe', 'legacy']:
                    report.add_item(self,
                                    "{} (compute='{}') does not support contact systems".format(compute_kind, compute),
                                    [self.hierarchy
                                     ]+addl_parameters,
                                     True, 'run_compute')
            if len(self.hierarchy.get_stars()) == 1:
                if compute_kind not in ['phoebe']:
                    report.add_item(self,
                                    "{} (compute='{}') does not support single star systems".format(compute_kind, compute),
                                    [self.hierarchy
                                     ]+addl_parameters,
                                     True, 'run_compute')
            elif len(self.hierarchy.get_stars()) > 2:
                if compute_kind not in []:
                    report.add_item(self,
                                    "{} (compute='{}') does not support multiple systems".format(compute_kind, compute),
                                    [self.hierarchy
                                     ]+addl_parameters,
                                     True, 'run_compute')

            # sample_from and solution checks
            # check if any parameter is in sample_from but is constrained
            # NOTE: similar logic exists for init_from in run_checks_solver

            # distribution checks
            sample_from = self.get_value(qualifier='sample_from', compute=compute, context='compute', sample_from=kwargs.get('sample_from', None), default=[], expand=True, **_skip_filter_checks)
            for dist_or_solution in sample_from:
                if dist_or_solution in self.distributions:
                    for distribution_param in self.filter(distribution=dist_or_solution, context='distribution', **_skip_filter_checks).to_list():
                        ref_param = distribution_param.get_referenced_parameter()
                        if len(ref_param.constrained_by):
                            # we'll raise an error if a delta distribution (i.e. probably not from a sampler)
                            # but only a warning for other distributions
                            error = distribution_param.get_value().__class__.__name__ in ['Delta']
                            if error:
                                msg = "{} is constrained, but is included in sample_from='{}' as a Delta function.  Flip constraint to include in sampling or remove from sample_from.".format(ref_param.twig, dist_or_solution)
                            else:
                                msg = "{} is constrained, so will be ignored by sample_from='{}'.  Flip constraint to include in sampling.".format(ref_param.twig, dist_or_solution)

                            report.add_item(self,
                                            msg,
                                            [distribution_param,
                                            ref_param,
                                            ref_param.is_constraint,
                                            self.get_parameter(qualifier='sample_from', compute=compute, context='compute', **_skip_filter_checks)
                                            ]+addl_parameters,
                                            error, 'run_compute')

                elif dist_or_solution in self.solutions:
                    solution_ps = self.get_solution(solution=dist_or_solution, **_skip_filter_checks)
                    fitted_uniqueids = solution_ps.get_value(qualifier='fitted_uniqueids', **_skip_filter_checks)
                    adopt_parameters = solution_ps.get_value(qualifier='adopt_parameters', **_skip_filter_checks)
                    fitted_ps = self.filter(uniqueid=[str(u) for u in fitted_uniqueids], **_skip_filter_checks)
                    for param in fitted_ps.filter(twig=adopt_parameters, **_skip_filter_checks).to_list():
                        if len(param.constrained_by):
                            # we'll raise an error if not a sampler (i.e. if values would be adopted by default)
                            # but only a warning for samplers (i.e. distributions would be adopted by default)
                            error = solution_ps.kind not in ['emcee', 'dynesty']
                            if error:
                                msg = "{} is constrained, but is included in sample_from='{}'.  Flip constraint to include in sampling, remove '{}' from sample_from, or remove '{}' from adopt_parameters.".format(param.twig, dist_or_solution, dist_or_solution, param.twig)
                            else:
                                msg = "{} is constrained, so the distribution be ignored by sample_from='{}'.  Flip constraint to include in sampling, or remove '{}' from sample_from or '{}' from adopt_parameters to remove warning.".format(param.twig, dist_or_solution, dist_solution, param.twig)

                            report.add_item(self,
                                            msg,
                                            [param,
                                             param.is_constraint,
                                             solution_ps.get_parameter(qualifier='adopt_parameters', **_skip_filter_checks),
                                             self.get_parameter(qualifier='sample_from', compute=compute, context='compute', **_skip_filter_checks)
                                             ]+addl_parameters,
                                             error, 'run_compute')
                else:
                    raise ValueError("{} could not be found in distributions or solutions".format(dist_or_solution))

            # check for time-dependency issues with GPs
            if len(compute_enabled_gps):
                # then if we're using compute_times/phases, compute_times must cover the range of the dataset times
                for dataset in compute_enabled_datasets_with_gps:
                    compute_times = self.get_value(qualifier='compute_times', dataset=dataset, context='dataset', unit=u.d, **_skip_filter_checks)
                    if len(compute_times):
                        for time_param in self.filter(qualifier='times', dataset=dataset, context='dataset', check_visible=True).to_list():
                            gp_warning = True
                            if self.hierarchy.is_time_dependent(consider_gaussian_process=False):
                                ds_times = time_param.get_value(unit=u.d)

                                if min(ds_times) < min(compute_times) or max(ds_times) > max(compute_times):
                                    gp_warning = False
                                    report.add_item(self,
                                                    "compute_times must cover full range of times for {} in order to include gaussian processes".format("@".join([time_param.dataset, time_param.component] if time_param.component is not None else [time_param.dataset])),
                                                    [self.get_parameter(qualifier='compute_times', dataset=dataset, context='dataset', **_skip_filter_checks),
                                                     self.get_parameter(qualifier='compute_phases', dataset=dataset, context='dataset', **_skip_filter_checks),
                                                     time_param]+self.filter(qualifier='enabled', feature=compute_enabled_gps, **_skip_filter_checks).to_list()+addl_parameters,
                                                     True, 'run_compute')
                            if gp_warning:
                                # then raise a warning to tell that the resulting model will be at different times
                                report.add_item(self,
                                                "underlying model will be computed at compute_times for {} but exposed at dataset times in order to include gaussian processes".format("@".join([time_param.dataset, time_param.component] if time_param.component is not None else [time_param.dataset])),
                                                [self.get_parameter(qualifier='compute_times', dataset=dataset, context='dataset', **_skip_filter_checks),
                                                 self.get_parameter(qualifier='compute_phases', dataset=dataset, context='dataset', **_skip_filter_checks),
                                                 time_param]+self.filter(qualifier='enabled', feature=compute_enabled_gps, **_skip_filter_checks).to_list()+addl_parameters,
                                                 False, 'run_compute')

                    ds_ps = self.get_dataset(dataset=dataset, **_skip_filter_checks)
                    xqualifier = {'lp': 'wavelength'}.get(ds_ps.kind, 'times')
                    yqualifier = {'lp': 'flux_densities', 'rv': 'rvs', 'lc': 'fluxes'}.get(ds_ps.kind)
                    # we'll loop over components (for RVs or LPs, for example)
                    if ds_ps.kind in ['lc']:
                        ds_comps = [None]
                    else:
                        ds_comps = ds_ps.filter(qualifier=xqualifier, check_visible=True).components
                    for ds_comp in ds_comps:
                        ds_x = ds_ps.get_value(qualifier=xqualifier, component=ds_comp, **_skip_filter_checks)
                        ds_y = ds_ps.get_value(qualifier=yqualifier, component=ds_comp, **_skip_filter_checks)
                        ds_sigmas = ds_ps.get_value(qualifier='sigmas', component=ds_comp, **_skip_filter_checks)
                        # NOTE: if we're supporting GPs on RVs, we should only require at least ONE component to have len(ds_x)
                        if len(ds_sigmas) != len(ds_x) or len(ds_y) != len(ds_x) or (ds_ps.kind in ['lc'] and not len(ds_x)):
                            report.add_item(self,
                                            "gaussian process requires observational data and sigmas",
                                            ds_ps.filter(qualifier=[xqualifier, yqualifier, 'sigmas'], component=ds_comp, **_skip_filter_checks).to_list()+
                                            self.filter(qualifier='enabled', feature=compute_enabled_gps, compute=compute, **_skip_filter_checks).to_list()+
                                            addl_parameters,
                                            True, 'run_compute')


            # 2.2 disables support for boosting.  The boosting parameter in 2.2 only has 'none' as an option, but
            # importing a bundle from old releases may still have 'linear' as an option, so we'll check here
            if compute_kind in ['phoebe'] and self.get_value(qualifier='boosting_method', compute=compute, boosting_method=kwargs.get('boosting_method', None), **_skip_filter_checks) != 'none':
                report.add_item(self,
                                "support for beaming/boosting has been removed from PHOEBE 2.2.  Set boosting_method to 'none'.",
                                [self.get_parameter(qualifier='boosting_method', compute=compute, boosting_method=kwargs.get('boosting_method', None), **_skip_filter_checks)
                                ]+addl_parameters,
                                True, 'run_compute')

            # misalignment checks
            if compute_kind != 'phoebe' and np.any([p.get_value() != 0 for p in self.filter(qualifier=['pitch', 'yaw'], context='component', **_skip_filter_checks).to_list()]):
                # then we have a misaligned system in an alternate backend
                if compute_kind == 'ellc':
                    if np.all([p.get_value(distortion_method=kwargs.get('distortion_method')) == 'sphere' for p in self.filter(qualifier='distortion_method', compute=compute, context='compute', **_skip_filter_checks).to_list()]):
                        # then misalignment is supported, but we'll raise a warning that it only handles RM in RVs
                        report.add_item(self,
                                        "ellc (compute='{}') only considers misalginment for the Rossiter-McLaughlin contribution to RVs".format(compute),
                                        self.filter(qualifier=['pitch', 'yaw'], context='component', **_skip_filter_checks).to_list()+addl_parameters,
                                        False, 'run_compute')
                    else:
                        report.add_item(self,
                                        "ellc (compute='{}') only supports misalignment (for Rossiter-McLaughlin contribution to RVs) with distortion_method='sphere'".format(compute),
                                        self.filter(qualifier=['pitch', 'yaw'], context='component', **_skip_filter_checks).to_list()+
                                        self.filter(qualifier='distortion_method', compute=compute, context='compute', **_skip_filter_checks).to_list()+addl_parameters,
                                        True, 'run_compute')
                else:
                    report.add_item(self,
                                    "compute='{}' with kind {} does not support misalignment".format(compute, compute_kind),
                                    self.filter(qualifier=['pitch', 'yaw'], context='component', **_skip_filter_checks).to_list()+
                                    self.filter(qualifier='distortion_method', compute=compute, context='compute', **_skip_filter_checks).to_list()+addl_parameters,
                                    True, 'run_compute')

            # mesh-consistency checks
            mesh_methods = [p.get_value(mesh_method=kwargs.get('mesh_method', None)) for p in self.filter(qualifier='mesh_method', compute=compute, force_ps=True, check_default=True, check_visible=False).to_list()]
            if 'wd' in mesh_methods:
                if len(set(mesh_methods)) > 1:
                    report.add_item(self,
                                    "all (or no) components must use mesh_method='wd'.",
                                    self.filter(qualifier='mesh_method', compute=compute, force_ps=True, check_default=True, check_visible=False).to_list()+addl_parameters,
                                    True, 'run_compute')

            # estimate if any body is smaller than any other body's triangles, using a spherical assumption
            if compute_kind=='phoebe' and 'wd' not in mesh_methods:
                eclipse_method = self.get_value(qualifier='eclipse_method', compute=compute, eclipse_method=kwargs.get('eclipse_method', None), **_skip_filter_checks)
                if eclipse_method == 'only_horizon':
                    # no need to check triangle sizes
                    continue

                areas = {comp: _get_proj_area(comp) for comp in hier_meshables}
                triangle_areas = {comp: _get_surf_area(comp)/self.get_value(qualifier='ntriangles', component=comp, compute=compute, **_skip_filter_checks) for comp in hier_meshables}
                if max(triangle_areas.values()) > 5*min(areas.values()):
                    if max(triangle_areas.values()) > 2*min(areas.values()):
                        offending_components = [comp for comp in triangle_areas.keys() if triangle_areas[comp] > 2*min(areas.values())]
                        smallest_components = [comp for comp in areas.keys() if areas[comp] == min(areas.values())]
                        report.add_item(self,
                                        "triangles on {} may be larger than the entire bodies of {}, resulting in inaccurate eclipse detection.  Check values for requiv of {} and/or ntriangles of {}.  If your system is known to NOT eclipse, you can set eclipse_method to 'only_horizon' to circumvent this check.".format(offending_components, smallest_components, smallest_components, offending_components),
                                        self.filter(qualifier='requiv', component=smallest_components, **_skip_filter_checks).to_list()+
                                        self.filter(qualifier='ntriangles', component=offending_components, compute=compute, **_skip_filter_checks).to_list()+[
                                        self.get_parameter(qualifier='eclipse_method', compute=compute, **_skip_filter_checks)
                                        ]+addl_parameters,
                                        True, 'run_compute')

                    else:
                        # only raise a warning
                        offending_components = [comp for comp in triangle_areas.keys() if triangle_areas[comp] > 5*min(areas.values())]
                        smallest_components = [comp for comp in areas.keys() if areas[comp] == min(areas.values())]
                        report.add_item(self,
                                        "triangles on {} are nearly the size of the entire bodies of {}, resulting in inaccurate eclipse detection.  Check values for requiv of {} and/or ntriangles of {}.  If your system is known to NOT eclipse, you can set eclipse_method to 'only_horizon' to circumvent this check.".format(offending_components, smallest_components, smallest_components, offending_components),
                                        self.filter(qualifier='requiv', component=smallest_components).to_list()+
                                        self.filter(qualifier='ntriangles', component=offending_components, compute=compute, **_skip_filter_checks).to_list()+[
                                        self.get_parameter(qualifier='eclipse_method', compute=compute, eclipse_method=kwargs.get('eclipse_method', None), **_skip_filter_checks)
                                        ]+addl_parameters,
                                        False, 'run_compute')

            # ellc-specific checks
            if compute_kind == 'ellc':
                irrad_method = self.get_value(qualifier='irrad_method', compute=compute, context='compute', **_skip_filter_checks)
                rv_datasets = self.filter(kind='rv', context='dataset', **_skip_filter_checks).datasets
                rv_datasets_enabled = self.filter(qualifier='enabled', dataset=rv_datasets, compute=compute, context='compute', value=True, **_skip_filter_checks).datasets
                if irrad_method != 'none' and len(rv_datasets_enabled):
                    # then we can't allow albedos with flux-weighted RVs
                    offending_components = []
                    offending_datasets = []
                    for dataset in rv_datasets_enabled:
                        for component in hier_stars:
                            rv_method = self.get_value(qualifier='rv_method', compute=compute, component=component, dataset=dataset, context='compute', **_skip_filter_checks)
                            if rv_method != 'dynamical' and self.get_value(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks) > 0:
                                if component not in offending_components:
                                    offending_components.append(component)
                                if dataset not in offending_datasets:
                                    offending_datasets.append(dataset)

                    if len(offending_components) and len(offending_datasets):
                        report.add_item(self,
                                        "ellc does not support irradiation with flux-weighted RVs.  Disable irradiation, use dynamical RVs, or set irrad_frac_refl_bol to 0.",
                                        self.filter(qualifier='irrad_method', compute=compute, context='compute', **_skip_filter_checks).to_list()+
                                        self.filter(qualifier='rv_method', compute=compute, component=offending_components, dataset=offending_datasets, context='compute', **_skip_filter_checks).to_list()+
                                        self.filter(qualifier='enabled', kind='rv', compute=compute, dataset=offending_datasets, context='compute', value=True, **_skip_filter_checks).to_list()+
                                        self.filter(qualifier='irrad_frac_refl_bol', component=offending_components, context='component', **_skip_filter_checks).to_list()+
                                        addl_parameters,
                                        True, 'run_compute')

                dpdt_non_zero = [p for p in self.filter(qualifier='dpdt', context='component', **_skip_filter_checks).to_list() if p.get_value() != 0]
                if len(dpdt_non_zero):
                    report.add_item(self,
                                    "ellc does not support orbital period time-derivative",
                                    dpdt_non_zero+addl_parameters,
                                    False, 'run_compute')

            # jktebop-specific checks
            if compute_kind == 'jktebop':
                requiv_max_limit = self.get_value(qualifier='requiv_max_limit', compute=compute, context='compute', requiv_max_limit=kwargs.get('requiv_max_limit', None), **_skip_filter_checks)
                for component in hier_stars:
                    requiv = self.get_value(qualifier='requiv', component=component, context='component', unit=u.solRad, **_skip_filter_checks)
                    requiv_max = self.get_value(qualifier='requiv_max', component=component, context='component', unit=u.solRad, **_skip_filter_checks)

                    if requiv > requiv_max_limit * requiv_max:
                        report.add_item(self,
                                        "requiv@{} ({}) > requiv_max_limit ({}) * requiv_max ({}): past user-set limit for allowed distortion for jktebop (compute='{}')".format(component, requiv, requiv_max_limit, requiv_max, compute),
                                        self.filter(qualifier='requiv_max_limit', compute=compute, context='compute', **_skip_filter_checks).to_list()+
                                        self.filter(qualifier=['requiv', 'requiv_max'], component=component, context='component', **_skip_filter_checks).to_list()+
                                        self.filter(qualifier=['sma'], component=self.hierarchy.get_parent_of(component), context='component', **_skip_filter_checks).to_list()+
                                        addl_parameters,
                                        True, 'run_compute')

                dperdt_non_zero = [p for p in self.filter(qualifier='dperdt', context='component', **_skip_filter_checks).to_list() if p.get_value() != 0]
                if len(dperdt_non_zero):
                    report.add_item(self,
                                    "jktebop does not support apsidal motion",
                                    dperdt_non_zero+addl_parameters,
                                    False, 'run_compute')

                dpdt_non_zero = [p for p in self.filter(qualifier='dpdt', context='component', **_skip_filter_checks).to_list() if p.get_value() != 0]
                if len(dpdt_non_zero):
                    report.add_item(self,
                                    "jktebop does not support orbital period time-derivative",
                                    dpdt_non_zero+addl_parameters,
                                    False, 'run_compute')

        # dependency checks
        if not _use_celerite and len(self.filter(context='feature', kind='gaussian_process').features):
            report.add_item(self,
                            "Gaussian process features attached, but celerite dependency not installed",
                            [],
                            True, 'run_compute')



        self._run_checks_warning_error(report, raise_logger_warning, raise_error)

        return report

    def run_checks_solver(self, solver=None, compute=None, solution=None, figure=None,
                          raise_logger_warning=False, raise_error=False, **kwargs):
        """
        Check to for any expected errors/warnings to <phoebe.frontend.bundle.Bundle.run_solver>.

        This is called by default for each set_value but will only raise a
        logger warning if fails.  This is also called immediately when calling
        <phoebe.frontend.bundle.Bundle.run_solver>.

        If the solver requires a forward model to be compute, <phoebe.frontend.bundle.Bundle.run_checks_compute>
        will also be called with the applicable `compute` if `run_checks_compute`
        is True.

        kwargs are passed to override currently set values as if they were
        sent to <phoebe.frontend.bundle.Bundle.run_solver>.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_checks>
        * <phoebe.frontend.bundle.Bundle.run_checks_system>
        * <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * <phoebe.frontend.bundle.Bundle.run_checks_solution>
        * <phoebe.frontend.bundle.Bundle.run_checks_figure>

        Arguments
        -----------
        * `solver` (string or list of strings, optional, default=None): the
            solver options to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_solver@setting' parameter
            will be used (which defaults to all available solver options).
        * `run_checks_compute` (bool, optional, default=True): whether to also
            call <phoebe.frontend.bundle.run_checks_compute> on any `compute`
            listed in the solver options in `solver`.
        * `allow_skip_constraints` (bool, optional, default=False): whether
            to allow skipping running delayed constraints if interactive
            constraints are disabled.  See <phoebe.interactive_constraints_off>.
        * `raise_logger_warning` (bool, optional, default=False): whether to
            raise any errors/warnings in the logger (with level of warning).
        * `raise_error` (bool, optional, default=False): whether to raise an
            error if the report has a status of failed.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (<phoebe.frontend.bundle.RunChecksReport>) object containing all
            errors/warnings.  Print the returned object to see all messages.
            See also: <phoebe.frontend.bundle.RunChecksReport.passed>,
             <phoebe.frontend.bundle.RunChecksReport.items>, and
             <phoebe.frontend.bundle.RunChecksItem.message>.
        """

        # make sure all constraints have been run
        if conf.interactive_constraints or not kwargs.pop('allow_skip_constraints', False):
            changed_params = self.run_delayed_constraints()

        report = kwargs.pop('report', RunChecksReport())
        addl_parameters = kwargs.pop('addl_parameters', [])

        run_checks_solver = self.get_value(qualifier='run_checks_solver', context='setting', default='*', expand=True, **_skip_filter_checks)
        if solver is None:
            solvers = run_checks_solver
            addl_parameters += [self.get_parameter(qualifier='run_checks_solver', context='setting', **_skip_filter_checks)]
        else:
            solvers = solver
            if isinstance(solvers, str):
                solvers = [solvers]

        for solver in solvers:
            if solver not in self.solvers:
                raise ValueError("solver='{}' not found".format(solver))

            if solver not in run_checks_solver:
                report.add_item(self,
                                "solver='{}' is not included in run_checks_solver@setting, so will not raise interactive warnings".format(solver),
                                [self.get_parameter(qualifier='run_checks_solver', context='setting', check_visible=False, check_default=False)],
                                False
                                )

        is_single = len(self.hierarchy.get_stars()) == 1
        is_cb = len(self.hierarchy.get_envelopes()) > 0

        for solver in solvers:
            solver_ps = self.get_solver(solver=solver, **_skip_filter_checks)
            solver_kind = solver_ps.kind

            if is_single and solver_kind in ['lc_geometry', 'ebai', 'rv_geometry']:
                report.add_item(self,
                                "{} does not support single stars".format(solver_kind),
                                [self.hierarchy]+addl_parameters,
                                True, 'run_solver')

            elif is_cb and solver_kind in ['lc_geometry', 'ebai']:
                report.add_item(self,
                                "{} does not support contact binaries".format(solver_kind),
                                [self.hierarchy]+addl_parameters,
                                True, 'run_solver')



            if 'compute' in solver_ps.qualifiers:
                # NOTE: we can't pass compute as a kwarg to get_value or it will be used as a filter instead... which means technically we can't be sure compute is in self.computes
                compute = kwargs.get('compute', solver_ps.get_value(qualifier='compute', **_skip_filter_checks))
                if kwargs.get('run_checks_compute', True):
                    if compute not in self.computes:
                        raise ValueError("compute='{}' not in computes".format(compute))
                    # TODO: do we need to append (only if report was sent as a kwarg)
                    report = self.run_checks_compute(compute=compute, raise_logger_warning=False, raise_error=False, report=report, addl_parameters=[solver_ps.get_parameter(qualifier='compute', **_skip_filter_checks)])

                # test to make sure solver_times will cover the full dataset for time-dependent systems
                if self.hierarchy.is_time_dependent(consider_gaussian_process=True):
                    for dataset in self.filter(qualifier='enabled', compute=compute, context='compute', value=True, **_skip_filter_checks).datasets:
                        solver_times = self.get_value(qualifier='solver_times', dataset=dataset, context='dataset', **_skip_filter_checks)
                        if solver_times == 'times':
                            continue

                        for param in self.filter(qualifier='times', dataset=dataset, context='dataset', **_skip_filter_checks).to_list():
                            component = param.component
                            compute_times = self.get_value(qualifier='compute_times', dataset=param.dataset, context='dataset', **_skip_filter_checks)
                            times = self.get_value(qualifier='times', dataset=param.dataset, component=param.component, context='dataset', **_skip_filter_checks)

                            if len(times) and len(compute_times) and (min(times) < min(compute_times) or max(times) > max(compute_times)):

                                params = [self.get_parameter(qualifier='solver_times', dataset=dataset, context='dataset', **_skip_filter_checks),
                                          self.get_parameter(qualifier='times', dataset=dataset, component=component, context='dataset', **_skip_filter_checks),
                                          self.get_parameter(qualifier='compute_times', dataset=dataset, context='dataset', **_skip_filter_checks)]

                                msg = "'compute_times@{}' must cover full range of 'times@{}', for time-dependent systems with solver_times@{}='{}'.".format(dataset, dataset, dataset, solver_times)
                                if len(self.get_parameter(qualifier='compute_phases', dataset=dataset, context='dataset', **_skip_filter_checks).constrains):
                                    msg += " Consider flipping the 'compute_phases' constraint and providing 'compute_times' instead."
                                    params += [self.get_parameter(qualifier='compute_phases', dataset=dataset, context='dataset', **_skip_filter_checks),
                                               self.get_constraint(qualifier='compute_times', dataset=dataset, **_skip_filter_checks)]

                                report.add_item(self,
                                                msg,
                                                params,
                                                True, ['run_solver'])

                # dataset column checks
                lc_datasets = self.filter(dataset=self.filter(qualifier='enabled', value=True, compute=compute, context='compute', **_skip_filter_checks).datasets, kind='lc', context='dataset', **_skip_filter_checks).datasets
                rv_datasets = self.filter(dataset=self.filter(qualifier='enabled', value=True, compute=compute, context='compute', **_skip_filter_checks).datasets, kind='rv', context='dataset', **_skip_filter_checks).datasets

                for dataset in lc_datasets+rv_datasets:
                    for time_param in self.filter(qualifier='times', dataset=dataset, context='dataset', **_skip_filter_checks).to_list():
                        component = time_param.component
                        times = time_param.get_value()

                        if np.any(np.isnan(times)):
                            report.add_item(self,
                                            "times cannot contain any nans",
                                            self.filter(qualifier=['times'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                            +addl_parameters,
                                            True, 'run_solver')

                        sigmas = self.get_value(qualifier='sigmas', dataset=dataset, component=component, context='dataset', **_skip_filter_checks)

                        if time_param.kind == 'lc':
                            fluxes = self.get_value(qualifier='fluxes', dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                            if np.any(np.isnan(fluxes)):
                                report.add_item(self,
                                                "fluxes cannot contain any nans",
                                                self.filter(qualifier=['fluxes'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                                +addl_parameters,
                                                True, 'run_solver')

                        elif time_param.kind == 'rv':
                            rvs = self.get_value(qualifier='rvs', dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                            if np.any(np.isnan(rvs)):
                                report.add_item(self,
                                                "rvs cannot contain any nans",
                                                self.filter(qualifier=['rvs'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                                +addl_parameters,
                                                True, 'run_solver')

                        if np.any(np.isnan(sigmas)):
                            report.add_item(self,
                                            "sigmas cannot contain any nans",
                                            self.filter(qualifier=['sigmas'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                            +addl_parameters,
                                            True, 'run_solver')

                        if np.any(sigmas==0):
                            report.add_item(self,
                                            "sigmas cannot contain zeros",
                                            self.filter(qualifier=['sigmas'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                            +addl_parameters,
                                            True, 'run_solver')



            if 'lc_datasets' in solver_ps.qualifiers:
                lc_datasets = solver_ps.get_value(qualifier='lc_datasets', lc_datasets=kwargs.get('lc_datasets', None), expand=True, **_skip_filter_checks)
                if not len(lc_datasets):
                    report.add_item(self,
                                    "no valid datasets in lc_datasets",
                                    [solver_ps.get_parameter(qualifier='lc_datasets', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    True, 'run_solver')

                for dataset in lc_datasets:
                    component = None
                    sigmas = self.get_value(qualifier='sigmas', dataset=dataset, component=component, context='dataset', **_skip_filter_checks)

                    if np.any(np.isnan(sigmas)):
                        report.add_item(self,
                                        "sigmas cannot contain any nans",
                                        self.filter(qualifier=['sigmas'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                        +addl_parameters,
                                        True, 'run_solver')

                    if np.any(sigmas==0):
                        report.add_item(self,
                                        "sigmas cannot contain zeros",
                                        self.filter(qualifier=['sigmas'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                        +addl_parameters,
                                        True, 'run_solver')


            elif 'compute' in solver_ps.qualifiers:
                lc_datasets = self.filter(dataset=self.filter(qualifier='enabled', value=True, compute=compute, context='compute', **_skip_filter_checks).datasets, kind='lc', context='dataset', **_skip_filter_checks).datasets
            else:
                lc_datasets = self.filter(kind='lc', context='dataset', **_skip_filter_checks).datasets

            if 'rv_datasets' in solver_ps.qualifiers:
                rv_datasets = solver_ps.get_value(qualifier='rv_datasets', rv_datasets=kwargs.get('rv_datasets', None), expand=True, **_skip_filter_checks)
                if not len(rv_datasets):
                    report.add_item(self,
                                    "no valid datasets in rv_datasets",
                                    [solver_ps.get_parameter(qualifier='rv_datasets', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    True, 'run_solver')

                for dataset in rv_datasets:
                    for time_param in self.filter(qualifier='times', dataset=dataset, context='dataset', **_skip_filter_checks).to_list():
                        component = time_param.component
                        if not len(time_param.get_value()):
                            continue
                        sigmas = self.get_value(qualifier='sigmas', dataset=dataset, component=component, context='dataset', **_skip_filter_checks)

                        if np.any(np.isnan(sigmas)):
                            report.add_item(self,
                                            "sigmas cannot contain any nans",
                                            self.filter(qualifier=['sigmas'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                            +addl_parameters,
                                            True, 'run_solver')

                        if np.any(sigmas==0):
                            report.add_item(self,
                                            "sigmas cannot contain zeros",
                                            self.filter(qualifier=['sigmas'], dataset=dataset, component=component, context='dataset', **_skip_filter_checks)
                                            +addl_parameters,
                                            True, 'run_solver')

            elif 'compute' in solver_ps.qualifiers:
                rv_datasets = self.filter(dataset=self.filter(qualifier='enabled', value=True, compute=compute, context='compute', **_skip_filter_checks).datasets, kind='rv', context='dataset', **_skip_filter_checks).datasets
            else:
                rv_datasets = self.filter(kind='rv', context='dataset', **_skip_filter_checks).datasets

            adjustable_parameters = self.get_adjustable_parameters(exclude_constrained=False, check_visible=False)

            if 'fit_parameters' in solver_ps.qualifiers:
                fit_parameters = solver_ps.get_value(qualifier='fit_parameters', fit_parameters=kwargs.get('fit_parameters', None), expand=True, **_skip_filter_checks)
                if not len(fit_parameters):
                    report.add_item(self,
                                    "no valid parameters in fit_parameters",
                                    [solver_ps.get_parameter(qualifier='fit_parameters', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    True, 'run_solver')

                for twig in fit_parameters:
                    twig, index = _extract_index_from_string(twig)
                    fit_parameter = adjustable_parameters.get_parameter(twig=twig, **_skip_filter_checks)
                    if index is not None:
                        if fit_parameter.__class__.__name__ != 'FloatArrayParameter':
                            report.add_item(self,
                                            "fit_parameters entry {} does not accept index".format(twig),
                                            [solver_ps.get_parameter(qualifier='fit_parameters', **_skip_filter_checks)
                                            ]+addl_parameters,
                                            True, 'run_solver')

                        elif index >= len(fit_parameter.get_value()):
                            report.add_item(self,
                                            "fit_parameters entry {} with length {} index {} out-of-bounds".format(twig, len(fit_parameter.get_value()), index),
                                            [solver_ps.get_parameter(qualifier='fit_parameters', **_skip_filter_checks)
                                            ]+addl_parameters,
                                            True, 'run_solver')

                    if len(fit_parameter.constrained_by):
                        report.add_item(self,
                                        "fit_parameters contains the constrained parameter '{}'".format(twig),
                                        [solver_ps.get_parameter(qualifier='fit_parameters', **_skip_filter_checks),
                                         fit_parameter.is_constraint
                                        ]+addl_parameters,
                                        True, 'run_solver')

                    if not fit_parameter.is_visible:
                        report.add_item(self,
                                        "fit_parameters contains the invisible parameter '{}'".format(twig),
                                        [solver_ps.get_parameter(qualifier='fit_parameters', **_skip_filter_checks)]
                                        +fit_parameter.visible_if_parameters.filter(check_visible=True).to_list()
                                        +addl_parameters,
                                         True, 'run_solver')


                fit_ps = adjustable_parameters.filter(twig=fit_parameters, **_skip_filter_checks)


            # need check_visible in case hidden by continue_from
            elif 'init_from' in solver_ps.qualifiers:
                continue_from = solver_ps.get_value(qualifier='continue_from', continue_from=kwargs.get('continue_from', None), default='None', **_skip_filter_checks)
                if continue_from.lower() != 'none':
                    _, init_from_uniqueids = self.get_distribution_collection(solution=continue_from, keys='uniqueid', return_dc=False)
                else:
                    _, init_from_uniqueids = self.get_distribution_collection(kwargs.get('init_from', 'init_from@{}'.format(solver)), keys='uniqueid', return_dc=False)

                if not len(init_from_uniqueids):
                    report.add_item(self,
                                    "no valid distributions in init_from",
                                    [solver_ps.get_parameter(qualifier='init_from', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    True, 'run_solver')

                fit_ps = adjustable_parameters.filter(uniqueid=init_from_uniqueids, **_skip_filter_checks)

            else:
                fit_ps = None


            if solver_kind in ['emcee'] and solver_ps.get_value(qualifier='continue_from', continue_from=kwargs.get('continue_from', None), **_skip_filter_checks) == 'None':
                # check to make sure twice as many params as walkers
                nwalkers = solver_ps.get_value(qualifier='nwalkers', nwalkers=kwargs.get('nwalkers', None), **_skip_filter_checks)

                # init_from_uniqueids should already be calculated above in call to get_distribution_collection
                if nwalkers < 2*len(init_from_uniqueids):
                    # TODO: double check this logic
                    report.add_item(self,
                                    "nwalkers must be at least 2*init_from = {}".format(2*len(init_from_uniqueids)),
                                    [solver_ps.get_parameter(qualifier='nwalkers', **_skip_filter_checks),
                                     solver_ps.get_parameter(qualifier='init_from', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    True, 'run_solver')

            if solver_kind in ['emcee', 'dynesty']:
                offending_parameters = self.filter(qualifier='pblum_mode', dataset=lc_datasets+rv_datasets, value='dataset-scaled', **_skip_filter_checks)
                if len(offending_parameters.to_list()):
                    report.add_item(self,
                                    "sampling with dataset-scaled can cause unintended issues.  Consider using component-coupled and marginalizing over pblum",
                                    offending_parameters.to_list()+
                                    [solver_ps.get_parameter(qualifier='priors' if solver in ['dynesty'] else 'init_from', **_skip_filter_checks)]+
                                    addl_parameters,
                                    False, 'run_solver')



            init_from = self.get_value(qualifier='init_from', solver=solver, context='solver', init_from=kwargs.get('init_from', None), default=[], expand=True, **_skip_filter_checks)
            for dist_or_solution in init_from:
                if dist_or_solution in self.distributions:
                    for distribution_param in self.filter(distribution=dist_or_solution, context='distribution', **_skip_filter_checks).to_list():
                        ref_param = distribution_param.get_referenced_parameter()
                        if len(ref_param.constrained_by):
                            # we'll raise an error if a delta distribution (i.e. probably not from a sampler)
                            # but only a warning for other distributions

                            msg = "{} is constrained, so cannot be included in init_from='{}'.  Flip constraint to include in sampling, or remove '{}' from init_from.".format(ref_param.twig, dist_or_solution, dist_or_solution)

                            report.add_item(self,
                                            msg,
                                            [distribution_param,
                                            ref_param,
                                            ref_param.is_constraint,
                                            self.get_parameter(qualifier='init_from', solver=solver, context='solver', **_skip_filter_checks)
                                            ]+addl_parameters,
                                            True, 'run_solver')

                        if not ref_param.is_visible:
                            report.add_item(self,
                                            "{} is not a visible parameter, so cannot be included in init_from='{}'.".format(ref_param.twig, dist_or_solution),
                                            [solver_ps.get_parameter(qualifier='init_from', **_skip_filter_checks)]
                                            +ref_param.visible_if_parameters.filter(check_visible=True).to_list()
                                            +addl_parameters,
                                             True, 'run_solver')

                        _, index = _extract_index_from_string(distribution_param.qualifier)
                        if index:
                            if index >= len(ref_param.get_value()):
                                report.add_item(self,
                                                "{}@{} in init_from@{} references an index ({}) that is out of range for {}".format(distribution_param.qualifier, dist_or_solution, solver, index, ref_param.twig),
                                                [solver_ps.get_parameter(qualifier='init_from', **_skip_filter_checks)]
                                                +[ref_param, distribution_param]
                                                +addl_parameters,
                                                True, 'run_solver')

                elif dist_or_solution in self.solutions:
                    solution_ps = self.get_solution(solution=dist_or_solution, **_skip_filter_checks)
                    fitted_uniqueids = solution_ps.get_value(qualifier='fitted_uniqueids', **_skip_filter_checks)
                    adopt_parameters = solution_ps.get_value(qualifier='adopt_parameters', **_skip_filter_checks)
                    fitted_ps = self.filter(uniqueid=[str(u) for u in fitted_uniqueids], **_skip_filter_checks)
                    for param in fitted_ps.filter(twig=adopt_parameters, **_skip_filter_checks).to_list():
                        if len(param.constrained_by):
                            # we'll raise an error if not a sampler (i.e. if values would be adopted by default)
                            # but only a warning for samplers (i.e. distributions would be adopted by default)

                            msg = "{} is constrained, so cannot be included in init_from='{}'.  Flip constraint to include in sampling, or remove '{}' from init_from.".format(param.twig, dist_or_solution, dist_solution)

                            report.add_item(self,
                                            msg,
                                            [param,
                                             param.is_constraint,
                                             self.get_parameter(qualifier='init_from', solver=solver, context='solver', **_skip_filter_checks)
                                             ]+addl_parameters,
                                             True, 'run_solver')

                        if not ref_param.is_visible:
                            report.add_item(self,
                                            "{} is not a visible parameter, so cannot be included in init_from='{}'.".format(ref_param.twig, dist_or_solution),
                                            [solver_ps.get_parameter(qualifier='init_from', **_skip_filter_checks)]
                                            +ref_param.visible_if_parameters.filter(check_visible=True).to_list()
                                            +addl_parameters,
                                             True, 'run_solver')

                else:
                    raise ValueError("{} could not be found in distributions or solutions".format(dist_or_solution))

            priors = self.get_value(qualifier='priors', solver=solver, context='solver', init_from=kwargs.get('priors', None), default=[], expand=True, **_skip_filter_checks)
            for dist in priors:
                for distribution_param in self.filter(distribution=dist, context='distribution', **_skip_filter_checks).to_list():
                    ref_param = distribution_param.get_referenced_parameter()

                    _, index = _extract_index_from_string(distribution_param.qualifier)
                    if index:
                        if index >= len(ref_param.get_value()):
                            report.add_item(self,
                                            "{}@{} in priors@{} references an index ({}) that is out of range for {}".format(distribution_param.qualifier, dist, solver, index, ref_param.twig),
                                            [solver_ps.get_parameter(qualifier='priors', **_skip_filter_checks)]
                                            +[ref_param, distribution_param]
                                            +addl_parameters,
                                            True, 'run_solver')


            offending_parameters = []
            for dist_or_solution in priors:
                if dist_or_solution in self.distributions:
                    for distribution_param in self.filter(distribution=dist_or_solution, context='distribution', **_skip_filter_checks).to_list():
                        if 'Around' in distribution_param.get_value().__class__.__name__:
                            offending_parameters.append(distribution_param)

            if len(offending_parameters):
                report.add_item(self,
                                "priors@{} includes \"around\" distributions.  Note that the central values of these distributions will update to the current face-values of the parameters (use with caution for priors)".format(solver),
                                [solver_ps.get_parameter(qualifier='priors', **_skip_filter_checks)]
                                +offending_parameters
                                +addl_parameters,
                                False, 'run_solver')


            ## warning if fitting a parameter that affects phasing but mask_phases is enabled
            if fit_ps is not None:
                fit_parameters_ephemeris = fit_ps.filter(qualifier=['period', 'per0', 't0*'], context='component', component=self.hierarchy.get_top(), **_skip_filter_checks)
                if len(fit_parameters_ephemeris):
                    offending_datasets = []
                    for dataset in lc_datasets + rv_datasets:
                        if len(self.get_value(qualifier='mask_phases', dataset=dataset, context='dataset', **_skip_filter_checks)):
                            offending_datasets.append(dataset)

                    if len(offending_datasets):
                        report.add_item(self,
                                        "fit_parameters contains a parameter ({}) that affects phasing which could cause issues with mask_phases".format(fit_parameters_ephemeris.qualifiers),
                                        self.filter(qualifier='mask_phases', dataset=offending_datasets, context='dataset', **_skip_filter_checks).to_list()
                                        +[solver_ps.get_parameter(qualifier=['fit_parameters', 'init_from'], **_skip_filter_checks)]
                                        +addl_parameters,
                                        False, 'run_solver')

            ## warning if abusing stefan-boltzmann
            if fit_ps is not None and 'compute' in solver_ps.qualifiers:
                if self.get_value(qualifier='pblum_method', compute=compute, context='compute', default='none', **_skip_filter_checks) == 'stefan-boltzmann':
                    fit_parameters_pblum_sb = fit_ps.filter(qualifier='pblum', dataset=lc_datasets+rv_datasets, **_skip_filter_checks)

                    if len(fit_parameters_pblum_sb):
                        report.add_item(self,
                                        "pblum_method=stefan-boltzmann is an approximation, fitting for pblum may not be reliable.  Consider removing from {} or setting pblum_method='phoebe' (more expensive).".format('fit_parameters' if 'fit_parameters' in solver_ps.qualifiers else 'init_from'),
                                        self.filter(qualifier='pblum_method', compute=compute, value='stefan-boltzmann', **_skip_filter_checks)
                                        +[solver_ps.get_parameter(qualifier=['fit_parameters', 'init_from'], **_skip_filter_checks)]
                                        +addl_parameters,
                                        False, 'run_solver')

        self._run_checks_warning_error(report, raise_logger_warning, raise_error)

        return report

    def run_checks_solution(self, solution=None, compute=None, solver=None, figure=None,
                            raise_logger_warning=False, raise_error=False, **kwargs):
        """
        Check to for any expected errors/warnings to <phoebe.frontend.bundle.Bundle.adopt_solution>.

        This is called by default for each set_value but will only raise a
        logger warning if fails.  This is also called immediately when calling
        <phoebe.frontend.bundle.Bundle.adopt_solution>.

        kwargs are passed to override currently set values as if they were
        sent to <phoebe.frontend.bundle.Bundle.adopt_solution>.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_checks>
        * <phoebe.frontend.bundle.Bundle.run_checks_system>
        * <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * <phoebe.frontend.bundle.Bundle.run_checks_solver>
        * <phoebe.frontend.bundle.Bundle.run_checks_figure>

        Arguments
        -----------
        * `solution` (string or list of strings, optional, default=None): the
            solution to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_solution@setting' parameter
            will be used (which defaults to no solutions, if not set).
        * `run_checks_compute` (bool, optional, default=True): whether to also
            call <phoebe.frontend.bundle.run_checks_compute> on any `compute`
            listed in the solution in `solution`.
        * `allow_skip_constraints` (bool, optional, default=False): whether
            to allow skipping running delayed constraints if interactive
            constraints are disabled.  See <phoebe.interactive_constraints_off>.
        * `raise_logger_warning` (bool, optional, default=False): whether to
            raise any errors/warnings in the logger (with level of warning).
        * `raise_error` (bool, optional, default=False): whether to raise an
            error if the report has a status of failed.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (<phoebe.frontend.bundle.RunChecksReport>) object containing all
            errors/warnings.  Print the returned object to see all messages.
            See also: <phoebe.frontend.bundle.RunChecksReport.passed>,
             <phoebe.frontend.bundle.RunChecksReport.items>, and
             <phoebe.frontend.bundle.RunChecksItem.message>.
        """

        # make sure all constraints have been run
        if conf.interactive_constraints or not kwargs.pop('allow_skip_constraints', False):
            changed_params = self.run_delayed_constraints()

        report = kwargs.pop('report', RunChecksReport())
        addl_parameters = kwargs.pop('addl_parameters', [])

        run_checks_solution = self.get_value(qualifier='run_checks_solution', context='setting', default='*', expand=True, **_skip_filter_checks)
        if solution is None:
            solutions = run_checks_solution
            addl_parameters += [self.get_parameter(qualifier='run_checks_solution', context='setting', **_skip_filter_checks)]
        else:
            solutions = solution
            if isinstance(solutions, str):
                solutions = [solutions]

        for solution in solutions:
            if solution not in self.solutions:
                raise ValueError("solution='{}' not found".format(solution))

            if solution not in run_checks_solution:
                report.add_item(self,
                                "solution='{}' is not included in run_checks_solution@setting, so will not raise interactive warnings".format(solution),
                                [self.get_parameter(qualifier='run_checks_solution', context='setting', check_visible=False, check_default=False)],
                                False
                                )

        # kwargs.setdefault('check_visible', False)
        # kwargs.setdefault('check_default', False)

        for solution in solutions:
            solution_ps = self.get_solution(solution=solution)
            solution_kind = solution_ps.kind

            adopt_values = solution_ps.get_value(qualifier='adopt_values', adopt_values=kwargs.get('adopt_values', None), **_skip_filter_checks)
            if adopt_values:
                adopt_inds, adopt_uniqueids = self._get_adopt_inds_uniqueids(solution_ps, **kwargs)

                # NOTE: samplers won't have fitted_values so this will default to the empty list
                fitted_values = solution_ps.get_value(qualifier='fitted_values', default=[], **_skip_filter_checks)
                # NOTE: the following list-comprehension is necessary because fitted_values may not be an array of floats/nans
                if len(fitted_values) and np.any([not isinstance(v,list) and np.isnan(v) for v in fitted_values[adopt_inds]]):
                    report.add_item(self,
                                    "at least one parameter in adopt_parameters includes nan in fitted_values",
                                    [solution_ps.get_parameter(qualifier='adopt_parameters', **_skip_filter_checks),
                                     solution_ps.get_parameter(qualifier='fitted_values', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    True, 'adopt_solution')

                if not kwargs.get('trial_run', False):
                    for adopt_uniqueid in adopt_uniqueids:
                        adopt_param = self.get_parameter(uniqueid=adopt_uniqueid.split('[')[0], **_skip_filter_checks)
                        if len(adopt_param.constrained_by):
                            constrained_by_ps = ParameterSet(adopt_param.constrained_by)
                            validsolvefor = [v for v in _constraint._validsolvefor.get(adopt_param.is_constraint.constraint_func, []) if adopt_param.qualifier not in v]
                            if len(validsolvefor) == 1:
                                solve_for = constrained_by_ps.get_parameter(twig=validsolvefor[0], **_skip_filter_checks)

                                report.add_item(self,
                                                "{} is currently constrained but will temporarily flip to solve_for='{}'".format(adopt_param.twig, solve_for.twig),
                                                [solution_ps.get_parameter(qualifier='adopt_parameters', **_skip_filter_checks),
                                                 adopt_param.is_constraint
                                                ]+addl_parameters,
                                                False, 'adopt_solution')

                            else:
                                report.add_item(self,
                                                "{} is currently constrained but cannot automatically temporarily flip as solve_for has several options ({}).  Flip the constraint manually first, set adopt_values=False, or remove {} from adopt_parameters.".format(adopt_param.twig, ", ".join([p.twig for p in adopt_param.constrained_by]), adopt_param.twig),
                                                [solution_ps.get_parameter(qualifier='adopt_parameters', **_skip_filter_checks),
                                                 solution_ps.get_parameter(qualifier='adopt_values', **_skip_filter_checks),
                                                 adopt_param.is_constraint
                                                ]+addl_parameters,
                                                True, 'adopt_solution')

        self._run_checks_warning_error(report, raise_logger_warning, raise_error)

        return report


    def run_checks_figure(self, figure=None, compute=None, solver=None, solution=None,
                          raise_logger_warning=False, raise_error=False, **kwargs):
        """
        Check to see whether the system is expected to be computable.

        This is called by default for each set_value but will only raise a
        logger warning if fails.  This is also called immediately when calling
        <phoebe.frontend.bundle.Bundle.run_compute>.

        kwargs are passed to override currently set values as if they were
        sent to <phoebe.frontend.bundle.Bundle.run_compute>.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_checks>
        * <phoebe.frontend.bundle.Bundle.run_checks_system>
        * <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * <phoebe.frontend.bundle.Bundle.run_checks_solver>
        * <phoebe.frontend.bundle.Bundle.run_checks_solution>

        Arguments
        -----------
        * `figure` (string or list of strings, optional, default=None): the
            figure options to use  when running checks.  If None (or not provided),
            the figure options in the 'run_checks_figure@setting' parameter
            will be used (which defaults to no figures, if not set).
        * `allow_skip_constraints` (bool, optional, default=False): whether
            to allow skipping running delayed constraints if interactive
            constraints are disabled.  See <phoebe.interactive_constraints_off>.
        * `raise_logger_warning` (bool, optional, default=False): whether to
            raise any errors/warnings in the logger (with level of warning).
        * `raise_error` (bool, optional, default=False): whether to raise an
            error if the report has a status of failed.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (<phoebe.frontend.bundle.RunChecksReport>) object containing all
            errors/warnings.  Print the returned object to see all messages.
            See also: <phoebe.frontend.bundle.RunChecksReport.passed>,
             <phoebe.frontend.bundle.RunChecksReport.items>, and
             <phoebe.frontend.bundle.RunChecksItem.message>.
        """

        # make sure all constraints have been run
        if conf.interactive_constraints or not kwargs.pop('allow_skip_constraints', False):
            changed_params = self.run_delayed_constraints()

        report = kwargs.pop('report', RunChecksReport())
        addl_parameters = kwargs.pop('addl_parameters', [])

        run_checks_figure = self.get_value(qualifier='run_checks_figure', context='setting', default='*', expand=True, **_skip_filter_checks)
        if figure is None:
            figures = run_checks_figure
            addl_parameters += [self.get_parameter(qualifier='run_checks_figure', context='setting', **_skip_filter_checks)]
        else:
            figures = figure
            if isinstance(figures, str):
                figures = [figures]

        for figure in figures:
            if figure not in self.figures:
                raise ValueError("figure='{}' not found".format(figure))

            if figure not in run_checks_figure:
                report.add_item(self,
                                "figure='{}' is not included in run_checks_figure@setting, so will not raise interactive warnings".format(figure),
                                [self.get_parameter(qualifier='run_checks_figure', context='setting', check_visible=False, check_default=False)],
                                False
                                )

        for param in self.filter(context='figure', qualifier='*lim', **_skip_filter_checks).to_list():
            if len(param.get_value()) != 2 and param.is_visible:
                parent_ps = param.get_parent_ps()
                if '{}_mode' in parent_ps.qualifiers:
                    mode_param = param.get_parent_ps().get_parameter(qualifier='{}_mode'.format(param.qualifier), check_visible=True)
                    affected_params = [param, mode_param]
                else:
                    # fclim_mode does not exist for fc='None'
                    affected_params = [param]

                report.add_item(self,
                                "{} does not have length of 2 - will be ignored and {}_mode will revert to 'auto'".format(param.twig, param.qualifier),
                                affected_params+addl_parameters,
                                False, 'run_figure')


        for figure in self.figures:
            if 'x' in self.filter(figure=figure, context='figure', **_skip_filter_checks).qualifiers:
                x = self.get_value(qualifier='x', figure=figure, context='figure', **_skip_filter_checks)
                y = self.get_value(qualifier='y', figure=figure, context='figure', **_skip_filter_checks)
                if (x in ['xs', 'ys', 'zs'] and y in ['us', 'vs', 'ws']) or (x in ['us', 'vs', 'ws'] and y in ['xs', 'ys', 'zs']):
                    report.add_item(self,
                                    "cannot mix xyz and uvw coordinates in {} figure".format(figure),
                                    [self.get_parameter(qualifier='x', figure=figure, context='figure', **_skip_filter_checks),
                                     self.get_parameter(qualifier='y', figure=figure, context='figure', **_skip_filter_checks)
                                    ]+addl_parameters,
                                    False, 'run_figure')

        self._run_checks_warning_error(report, raise_logger_warning, raise_error)

        return report

    def references(self, compute=None, dataset=None, solver=None):
        """
        Provides a list of used references from the given bundle based on the
        current parameter values and attached datasets/compute options.

        This list is not necessarily complete, but can be useful to find
        publications for various features/models used as well as to make sure
        appropriate references are being cited/acknowledged.  The returned
        dictionary includes a list for each entry why its being included.

        Included citations:
        * PHOEBE release papers based on physics included in the model (for
            example: the 2.1 release paper will be suggested if there is a
            line profile dataset or misalignment in the system).
        * Atmosphere table citations, when available/applicable.
        * Passband table citations, when available/applicable.
        * Dependency (astropy, numpy, etc) citations, when available/applicable.
        * Alternate compute and/or solver backends, when applicable.

        Arguments
        ------------
        * `compute` (string or list of strings, optional, default=None): only
            consider a single (or list of) compute options.  If None or not
            provided, will default to all attached compute options.
        * `dataset` (string or list of strings, optional, default=None): only
            consider a single (or list of) datasets.  If None or not provided,
            will default to all attached datasets.
        * `solver` (string or list of strings, optional, default=None): only
            consider a single (or list of) solver options.  If None or not
            provided, will default to all attached solver options.

        Returns
        ----------
        * (dict): dictionary with keys being the reference name and values as a
            dictionary with information about that reference: including a
            url if applicable and a list of detected uses within the current
            <phoebe.frontend.bundle.Bundle>.
        """

        if compute is None:
            computes = self.computes
        elif isinstance(compute, str):
            computes = [compute]
        elif isinstance(compute, list):
            computes = compute
        else:
            raise TypeError("compute must be type None, string, or list")

        if dataset is None:
            datasets = self.datasets
        elif isinstance(dataset, str):
            datasets = [dataset]
        elif isinstance(dataset, list):
            datasets = dataset
        else:
            raise TypeError("dataset must be type None, string, or list")

        if solver is None:
            solvers = self.solvers
        elif isinstance(solver, str):
            solvers = [solver]
        elif isinstance(solver, list):
            solvers = solver
        else:
            raise TypeError("solver must be of type None, string, or list")


        # ref: url pairs
        citation_urls = {'Prsa & Zwitter (2005)': 'https://ui.adsabs.harvard.edu/abs/2005ApJ...628..426P',
                         'Prsa et al. (2016)': 'https://ui.adsabs.harvard.edu/abs/2016ApJS..227...29P',
                         'Horvat et al. (2018)': 'https://ui.adsabs.harvard.edu/abs/2016ApJS..227...29P',
                         'Jones et al. (2020)': 'https://ui.adsabs.harvard.edu/abs/2020ApJS..247...63J',
                         'Conroy et al. (2020)': 'https://ui.adsabs.harvard.edu/abs/2020ApJS..250...34C',
                         'Castelli & Kurucz (2004)': 'https://ui.adsabs.harvard.edu/abs/2004astro.ph..5087C',
                         'Husser et al. (2013)': 'https://ui.adsabs.harvard.edu/abs/2013A&A...553A...6H',
                         'numpy/scipy': 'https://www.scipy.org/citing.html',
                         'astropy': 'https://www.astropy.org/acknowledging.html',
                         'jktebop': 'http://www.astro.keele.ac.uk/jkt/codes/jktebop.html',
                         'Carter et al. (2011)': 'https://ui.adsabs.harvard.edu/abs/2011Sci...331..562C',
                         'Andras (2012)': 'https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.1630P',
                         'Maxted (2016)': 'https://ui.adsabs.harvard.edu/abs/2016A%26A...591A.111M',
                         'Foreman-Mackey et al. (2013)': 'https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F',
                         'Speagle (2020)': 'https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S',
                         'Skilling (2004)': 'https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S',
                         'Skilling (2006)': 'https://projecteuclid.org/euclid.ba/1340370944',
                         'Foreman-Mackey et al. (2017)': 'https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F',
                         'Prsa et al. (2008)': 'https://ui.adsabs.harvard.edu/abs/2008ApJ...687..542P'
                        }

        # ref: [reasons] pairs
        recs = {}
        def _add_reason(recs, ref, reason):
            if ref not in recs.keys():
                recs[ref] = []
            if reason not in recs[ref]:
                recs[ref].append(reason)
            return recs

        recs = _add_reason(recs, 'Prsa et al. (2016)', 'general PHOEBE 2 framework')

        # check for backends
        for compute in computes:
            if self.get_compute(compute).kind == 'phoebe':
                recs = _add_reason(recs, 'Prsa et al. (2016)', 'PHOEBE 2 compute backend')
            elif self.get_compute(compute).kind == 'legacy':
                recs = _add_reason(recs, 'Prsa & Zwitter (2005)', 'PHOEBE 1 (legacy) compute backend')
                # TODO: include Wilson & Devinney?
            elif self.get_compute(compute).kind == 'jktebop':
                recs = _add_reason(recs, 'jktebop', 'jktebop compute backend')
            elif self.get_compute(compute).kind == 'photodynam':
                recs = _add_reason(recs, 'Carter et al. (2011)', 'photodynam compute backend')
                recs = _add_reason(recs, 'Andras (2012)', 'photodynam compute backend')
            elif self.get_compute(compute).kind == 'ellc':
                recs = _add_reason(recs, 'Maxted (2016)', 'ellc compute backend')

        if len(solvers):
            recs = _add_reason(recs, 'Conroy et al. (2020)', 'general inverse problem framework in PHOEBE')

        for solver in solvers:
            solver_kind = self.get_solver(solver).kind
            # estimators
            if solver_kind in ['lc_periodogram', 'rv_periodogram']:
                recs = _add_reason(recs, 'astropy', 'astropy.timeseries for periodograms')
            elif solver_kind in ['lc_geometry', 'rv_geometry']:
                recs = _add_reason(recs, 'Conroy et al. (2020)', '{} solver'.format(solver_kind))
            elif solver_kind == 'ebai':
                recs = _add_reason(recs, 'Prsa et al. (2008)', 'ebai solver backend')
            # optimizers
            elif solver_kind in ['nelder_mead', 'powell', 'cg']:
                recs = _add_reason(recs, 'numpy/scipy', '{} solver uses scipy.optimize'.format(solver_kind))
            # samplers
            elif solver_kind == 'emcee':
                recs = _add_reason(recs, 'Foreman-Mackey et al. (2013)', 'emcee solver backend')
            elif solver_kind == 'dynesty':
                recs = _add_reason(recs, 'Speagle (2020)', 'dynesty solver backend')
                recs = _add_reason(recs, 'Skilling (2004)', 'nested sampling: dynesty solver backend')
                recs = _add_reason(recs, 'Skilling (2006)', 'nested sampling: dynesty solver backend')



        # check for presence of datasets that require PHOEBE releases
        for dataset in datasets:
            if ['lp'] in self.get_dataset(dataset).kinds:
                recs = _add_reason(recs, 'Horvat et al. (2018)', 'support for line profile datasets')

        # check for any enabled physics that requires specific PHOEBE releases
        for component in self.hierarchy.get_stars():
            if self.get_value(qualifier='pitch', component=component, context='component') != 0. or self.get_value(qualifier='yaw', component=component, context='component') != 0.:
                recs = _add_reason(recs, 'Horvat et al. (2018)', 'support for misaligned system')
        if self.get_value(qualifier='ebv', context='system', **_skip_filter_checks) > 0:
            recs = _add_reason(recs, 'Jones et al. (2020)', 'support for interstellar extinction')

        # provide any references from passband tables
        for pb_param in self.filter(qualifier='passband', dataset=datasets, component=self.hierarchy.get_stars()).to_list():
            pbname = pb_param.get_value()
            pb = get_passband(pb)
            if pb.reference is not None:
                recs = _add_reason(recs, pb.reference, '{} passband'.format(pbname))

        # provide any references from atms
        for atm_param in self.filter(qualifier='atm', component=self.hierarchy.get_stars(), compute=computes).to_list():
            atmname = atm_param.get_value()
            if atmname == 'ck2004':
                recs = _add_reason(recs, 'Castelli & Kurucz (2004)', 'ck2004 atmosphere tables')
            elif atmname == 'phoenix':
                recs = _add_reason(recs, 'Husser et al. (2013)', 'phoenix atmosphere tables')
            elif atmname in ['extern_planckint', 'extern_atmx']:
                recs = _add_reason(recs, 'Prsa & Zwitter (2005)', '{} atmosphere tables'.format(atmname))

        for atm_param in self.filter(qualifier='ld_coeffs_source', component=self.hierarchy.get_stars()).to_list():
            atmname = atm_param.get_value()
            if atmname == 'ck2004':
                recs = _add_reason(recs, 'Castelli & Kurucz (2004)', 'ck2004 atmosphere tables for limb-darkening interpolation')
            elif atmname == 'phoenix':
                recs = _add_reason(recs, 'Husser et al. (2013)', 'phoenix atmosphere tables for limb-darkening interpolation')

        # provide any references from features
        if len(self.filter(context='feature', kind='gaussian_process').features):
            recs = _add_reason(recs, 'Foreman-Mackey et al. (2017)', 'celerite for gaussian processes')

        # provide references from dependencies
        recs = _add_reason(recs, 'numpy/scipy', 'numpy/scipy dependency within PHOEBE')
        recs = _add_reason(recs, 'astropy', 'astropy dependency within PHOEBE')

        return {r: {'url': citation_urls.get(r, None), 'uses': v} for r,v in recs.items()}

    @send_if_client
    def add_feature(self, kind, component=None, dataset=None,
                    return_changes= False, **kwargs):
        """
        Add a new feature (spot, gaussian process, etc) to a component or
        dataset in the system.  If not
        provided, `feature` (the name of the new feature) will be created
        for you and can be accessed by the `feature` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        ```py
        b.add_feature(feature.spot, component='mystar')
        ```

        or

        ```py
        b.add_feature('spot', 'mystar', colat=90)
        ```

        Available kinds can be found in <phoebe.parameters.feature> or by calling
        <phoebe.list_available_features> and include:
        * <phoebe.parameters.feature.spot>
        * <phoebe.parameters.feature.gaussian_process>

        See the entries above to see the valid kinds for `component` and `dataset`
        based on the type of feature.  An error will be raised if the passed value
        for `component` and/or `dataset` are not allowed by the type of feature
        with kind `kind`.

        Arguments
        -----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts only default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.compute> module.
        * `component` (string, optional): name of the component to attach the
            feature.  Required for features that must be attached to a component.
        * `dataset` (string, optional): name of the dataset to attach the feature.
            Required for features that must be attached to a dataset.
        * `feature` (string, optional): name of the newly-created feature.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing feature with the same `feature` tag.  If False,
            an error will be raised.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added


        Raises
        ----------
        * NotImplementedError: if a required constraint is not implemented.
        * ValueError: if `component` is required but is not provided or is of
            the wrong kind.
        * ValueError: if `dataset` is required but it not provided or is of the
            wrong kind.
        """
        func = _get_add_func(_feature, kind)

        if kwargs.get('feature', False) is None:
            # then we want to apply the default below, so let's pop for now
            _ = kwargs.pop('feature')

        kwargs.setdefault('feature',
                          self._default_label(func.__name__,
                                              **{'context': 'feature',
                                                 'kind': func.__name__}))

        self._check_label(kwargs['feature'], allow_overwrite=kwargs.get('overwrite', False))

        if component is not None:
            if component not in self.components:
                raise ValueError("component '{}' not one of {}".format(component, self.components))

            component_kind = self.filter(component=component, context='component').kind
        else:
            component_kind = None

        if not _feature._component_allowed_for_feature(func.__name__, component_kind):
            raise ValueError("{} does not support component with kind {}".format(func.__name__, component_kind))

        if dataset is not None:
            if dataset not in self.datasets:
                raise ValueError("dataset '{}' not one of {}".format(dataset, self.datasets))

            dataset_kind = self.filter(dataset=dataset, context='dataset').kind
        else:
            dataset_kind = None

        if not _feature._dataset_allowed_for_feature(func.__name__, dataset_kind):
            raise ValueError("{} does not support dataset with kind {}".format(func.__name__, dataset_kind))

        params, constraints = func(**kwargs)

        metawargs = {'context': 'feature',
                     'component': component,
                     'dataset': dataset,
                     'feature': kwargs['feature'],
                     'kind': func.__name__}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_feature(feature=kwargs['feature'], during_overwrite=True)
            # check the label again, just in case kwargs['feature'] belongs to
            # something other than feature
            self._check_label(kwargs['feature'], allow_overwrite=False)

        self._attach_params(params, **metawargs)
        # attach params called _check_copy_for, but only on it's own parameterset
        self._check_copy_for()

        for constraint in constraints:
            self.add_constraint(*constraint)

        ret_ps = self.filter(feature=kwargs['feature'], **_skip_filter_checks)

        ret_changes = []
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)

        if kwargs.get('overwrite', False) and return_changes:
            ret_ps += overwrite_ps

        if return_changes:
            ret_ps += ret_changes

        return _return_ps(self, ret_ps)

    def get_feature(self, feature=None, **kwargs):
        """
        Filter in the 'feature' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `feature`: (string, optional, default=None): the name of the feature
        * `**kwargs`: any other tags to do the filtering (excluding feature and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if feature is not None:
            kwargs['feature'] = feature
            if feature not in self.features:
                raise ValueError("feature='{}' not found".format(feature))
        kwargs['context'] = 'feature'
        return self.filter(**kwargs)

    @send_if_client
    def remove_feature(self, feature=None, return_changes=False, **kwargs):
        """
        Remove a 'feature' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `feature` (string, optional): the label of the feature to be removed.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: feature, qualifier.

        Returns
        -----------
        * ParameterSet of removed parameters

        Raises
        --------
        * ValueError: if `feature` is not provided AND no `kwargs` are provided.
        """
        self._kwargs_checks(kwargs, ['during_overwrite'])

        # Let's avoid deleting ALL features from the matching contexts
        if feature is None and not len(kwargs.items()):
            raise ValueError("must provide some value to filter for features")

        kwargs['feature'] = feature

        # Let's avoid the possibility of deleting a single parameter
        kwargs['qualifier'] = None

        # Let's also avoid the possibility of accidentally deleting system
        # parameters, etc
        kwargs.setdefault('context', ['feature', 'compute'])

        ret_ps = self.remove_parameters_all(**kwargs)

        ret_changes = []
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)
        if return_changes:
            ret_ps += ret_changes

        return ret_ps

    def remove_features_all(self, return_changes=False):
        """
        Remove all features from the bundle.  To remove a single feature, see
        <phoebe.frontend.bundle.Bundle.remove_feature>.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for feature in self.features:
            removed_ps += self.remove_feature(feature=feature, return_changes=return_changes)
        return removed_ps

    @send_if_client
    def rename_feature(self, old_feature, new_feature,
                       overwrite=False, return_changes=False):
        """
        Change the label of a feature attached to the Bundle.

        Arguments
        ----------
        * `old_feature` (string): current label of the feature (must exist)
        * `new_feature` (string): the desired new label of the feature
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_feature` is forbidden or already exists.
        """
        # TODO: raise error if old_feature not found?
        self._rename_label('feature', old_feature, new_feature, overwrite)

        ret_ps = self.filter(feature=new_feature)

        ret_changes = []
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)
        if return_changes:
            ret_ps += ret_changes

        return ret_ps


    def enable_feature(self, feature=None, **kwargs):
        """
        Enable a `feature`.  Features that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_solver .

        If `compute` is not provided, the dataset will be enabled across all
        compute options.

        Note that not all `compute` backends support all types of features.
        Unsupported features do not have 'enabled' parameters, and therefore
        cannot be enabled or disabled.

        See also:
        * <phoebe.frontend.bundle.Bundle.disable_feature>

        Arguments
        -----------
        * `feature` (string, optional): name of the feature
        * `**kwargs`:  any other tags to do the filter
            (except feature or context)

        Returns
        ---------
        * a <phoebe.parameters.ParameterSet> object of the enabled feature
        """
        kwargs['context'] = 'compute'
        kwargs['feature'] = feature
        kwargs['qualifier'] = 'enabled'
        self.set_value_all(value=True, **kwargs)

        return self.get_feature(feature=feature)


    def disable_feature(self, feature=None, **kwargs):
        """
        Disable a `feature`.  Features that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_solver.

        If `compute` is not provided, the dataset will be disabled across all
        compute options.

        Note that not all `compute` backends support all types of features.
        Unsupported features do not have 'enabled' parameters, and therefore
        cannot be enabled or disabled.

        See also:
        * <phoebe.frontend.bundle.Bundle.enable_feature>

        Arguments
        -----------
        * `feature` (string, optional): name of the feature
        * `**kwargs`:  any other tags to do the filter
            (except feature or context)

        Returns
        ---------
        * a <phoebe.parameters.ParameterSet> object of the disabled feature
        """
        kwargs['context'] = 'compute'
        kwargs['feature'] = feature
        kwargs['qualifier'] = 'enabled'
        self.set_value_all(value=False, **kwargs)

        return self.get_feature(feature=feature)

    def add_spot(self, component=None, feature=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_feature> but with kind='spot'.

        For details on the resulting parameters, see <phoebe.parameters.feature.spot>.
        """
        if component is None:
            if len(self.hierarchy.get_stars())==1:
                component = self.hierarchy.get_stars()[0]
            else:
                raise ValueError("must provide component for spot")

        kwargs.setdefault('component', component)
        kwargs.setdefault('feature', feature)
        return self.add_feature('spot', **kwargs)

    def get_spot(self, feature=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.get_feature> but with kind='spot'.

        Arguments
        ----------
        * `feature`: (string, optional, default=None): the name of the feature
        * `**kwargs`: any other tags to do the filtering (excluding feature, kind, and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        kwargs.setdefault('kind', 'spot')
        return self.get_feature(feature, **kwargs)

    def rename_spot(self, old_feature, new_feature,
                    overwrite=False, return_changes=False):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.rename_feature> but with kind='spot'.
        """
        return self.rename_feature(old_feature, new_feature, overwrite=overwrite, return_changes=return_changes)

    def add_gaussian_process(self, dataset=None, feature=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_feature> but with kind='gaussian_process'.

        For details on the resulting parameters, see <phoebe.parameters.feature.gaussian_process>.
        """
        if dataset is None:
            if len(self.datasets)==1:
                dataset = self.datasets[0]
            else:
                raise ValueError("must provide dataset for gaussian_process")

        kwargs.setdefault('dataset', dataset)
        kwargs.setdefault('feature', feature)
        return self.add_feature('gaussian_process', **kwargs)

    def get_gaussian_process(self, feature=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.get_feature> but with kind='gaussian_process'.

        Arguments
        ----------
        * `feature`: (string, optional, default=None): the name of the feature
        * `**kwargs`: any other tags to do the filtering (excluding feature, kind, and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        kwargs.setdefault('kind', 'gaussian_process')
        return self.get_feature(feature, **kwargs)

    def rename_gaussian_process(self, old_feature, new_feature, overwrite=False, return_changes=False):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.rename_feature> but with kind='gaussian_process'.
        """
        return self.rename_feature(old_feature, new_feature, overwrite=overwrite, return_changes=return_changes)

    @send_if_client
    def add_component(self, kind, return_changes=False, **kwargs):
        """
        Add a new component (star or orbit) to the system.  If not provided,
        `component` (the name of the new star or orbit) will be created for
        you and can be accessed by the `component` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        ```py
        b.add_component(component.star)
        ```

        or

        ```py
        b.add_component('orbit', period=2.5)
        ```

        Available kinds can be found in <phoebe.parameters.component> or by calling
        <phoebe.list_available_components> and include:
        * <phoebe.parameters.component.star>
        * <phoebe.parameters.component.orbit>
        * <phoebe.parameters.component.envelope>

        Arguments
        ----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts only default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.compute> module.
        * `component` (string, optional): name of the newly-created component.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing component with the same `component` tag.  If False,
            an error will be raised.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added
            or changed.

        Raises
        ----------
        * NotImplementedError: if a required constraint is not implemented.
        """

        func = _get_add_func(_component, kind)
        kind = func.__name__

        if kwargs.get('component', False) is None:
            # then we want to apply the default below, so let's pop for now
            _ = kwargs.pop('component')

        kwargs.setdefault('component',
                          self._default_label(kind,
                                              **{'context': 'component',
                                                 'kind': kind}))

        if kwargs.pop('check_label', True):
            self._check_label(kwargs['component'], allow_overwrite=kwargs.get('overwrite', False))

        params, constraints = func(**kwargs)


        metawargs = {'context': 'component',
                     'component': kwargs['component'],
                     'kind': kind}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_component(component=kwargs['component'], during_overwrite=True)
            # check the label again, just in case kwargs['component'] belongs to
            # something other than component
            self.exclude(component=kwargs['component'])._check_label(kwargs['component'], allow_overwrite=False)

        self._attach_params(params, **metawargs)
        # attach params called _check_copy_for, but only on it's own parameterset
        self._check_copy_for()

        for constraint in constraints:
            self.add_constraint(*constraint)

        # Figure options for this dataset
        fig_params = _figure._add_component(self, kind=kind, **kwargs)

        fig_metawargs = {'context': 'figure',
                         'kind': kind,
                         'component': kwargs['component']}
        self._attach_params(fig_params, **fig_metawargs)


        # TODO: include figure params in returned PS?
        ret_ps = self.get_component(check_visible=False, check_default=False, **metawargs)

        ret_changes = []
        ret_changes += self._handle_component_selectparams(return_changes=return_changes)
        ret_changes += self._handle_pblum_defaults(return_changes=return_changes)
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)
        ret_changes += self._handle_orbit_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_component_choiceparams(return_changes=return_changes)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs,
                            additional_allowed_keys=['overwrite'],
                            warning_only=True, ps=ret_ps)

        if kwargs.get('overwrite', False) and return_changes:
            ret_changes += overwrite_ps

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_component(self, component=None, **kwargs):
        """
        Filter in the 'component' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `component`: (string, optional, default=None): the name of the component
        * `**kwargs`: any other tags to do the filtering (excluding component and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if component is not None:
            kwargs['component'] = component
            if component not in self.components:
                raise ValueError("component='{}' not found".format(component))
        kwargs['context'] = 'component'
        return self.filter(**kwargs)

    @send_if_client
    def remove_component(self, component, return_changes=False, **kwargs):
        """
        Remove a 'component' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `component` (string): the label of the component to be removed.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: component, context

        Returns
        -----------
        * ParameterSet of removed or changed parameters
        """
        # NOTE: run_checks will check if an entry is in the hierarchy but has no parameters
        kwargs['component'] = component
        # NOTE: we do not remove from 'model' by default
        kwargs['context'] = ['component', 'constraint', 'dataset', 'compute', 'figure']
        ret_ps =  self.remove_parameters_all(**kwargs)

        ret_changes = []
        if not kwargs.get('during_overwrite', False):
            ret_changes += self._handle_component_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    @send_if_client
    def rename_component(self, old_component, new_component, return_changes=False):
        """
        Change the label of a component attached to the Bundle.

        Note: `overwrite` is not supported for `rename_component`

        Arguments
        ----------
        * `old_component` (string): current label of the component (must exist)
        * `new_component` (string): the desired new label of the component
            (must not yet exist)

        Returns
        --------
        * <phoebe.parameters.ParameterSet> of any parameters that were changed.

        Raises
        --------
        * ValueError: if the value of `new_component` is forbidden or already exists.
        """
        # TODO: raise error if old_component not found?

        # even though _rename_tag will call _check_label again, we should
        # do it first so that we can raise any errors BEFORE we start messing
        # with the hierarchy
        self._check_label(new_component)
        # changing hierarchy must be called first since it needs to access
        # the kind of old_component
        if len([c for c in self.components if new_component in c]):
            logger.warning("hierarchy may not update correctly with new component")
        self.hierarchy.rename_component(old_component, new_component)

        # NOTE: _handle_component_selectparams and _handle_pblum_defaults
        # is handled by _rename_label
        ret_params = self._rename_label('component', old_component, new_component, overwrite=False)
        self.hierarchy._update_cache()
        ret_params += [self.hierarchy]

        return ParameterSet(ret_params)

    def add_orbit(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_component> but with kind='orbit'.
        """
        kwargs.setdefault('component', component)
        return self.add_component('orbit', **kwargs)

    def get_orbit(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.get_component> but with kind='star'.

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `component`: (string, optional, default=None): the name of the component
        * `**kwargs`: any other tags to do the filtering (excluding component, kind, and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        kwargs.setdefault('kind', 'orbit')
        return self.get_component(component, **kwargs)

    def rename_orbit(self, old_orbit, new_orbit,
                     overwrite=False, return_changes=False):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.rename_component> but with kind='star'.
        """
        return self.rename_component(old_orbit, new_orbit, overwrite=overwrite, return_changes=return_changes)

    def add_star(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_component> but with kind='star'.
        """
        kwargs.setdefault('component', component)
        return self.add_component('star', **kwargs)

    def get_star(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.get_component> but with kind='star'

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `comopnent`: (string, optional, default=None): the name of the component
        * `**kwargs`: any other tags to do the filtering (excluding component, kind, and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        kwargs.setdefault('kind', 'star')
        return self.get_component(component, **kwargs)

    def rename_star(self, old_star, new_star, overwrite=False, return_changes=False):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.rename_component> but with kind='star'.
        """
        return self.rename_component(old_star, new_star, overwrite=overwrite, return_changes=return_changes)

    def remove_star(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.remove_component> but with kind='star'.
        """
        kwargs.setdefault('kind', 'star')
        return self.remove_component(component, **kwargs)

    def add_envelope(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_component> but with kind='envelope'.
        """
        kwargs.setdefault('component', component)
        return self.add_component('envelope', **kwargs)

    def get_envelope(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.get_component> but with kind='envelope'.

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `component`: (string, optional, default=None): the name of the component
        * `**kwargs`: any other tags to do the filtering (excluding component, kind, and context)

        Returns
        ----------
        * a <phoebe.parameters.ParameterSet> object.
        """
        kwargs.setdefault('kind', 'envelope')
        return self.get_component(component, **kwargs)

    def rename_envelope(self, old_envelope, new_envelope, overwrite=False, return_changes=False):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.rename_component> but with kind='envelope'
        """
        return self.rename_component(old_envelope, new_envelope, overwrite=overwrite, return_changes=return_changes)

    def remove_envelope(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.remove_component> but with kind='envelope'.
        """
        kwargs.setdefault('kind', 'envelope')
        return self.remove_component(component, **kwargs)

    def get_ephemeris(self, component=None, period='period', t0='t0_supconj', **kwargs):
        """
        Get the ephemeris of a component (star or orbit).

        NOTE: support for `shift` and `phshift` was removed as of version 2.1.
        Please pass `t0` instead.

        Arguments
        ---------------
        * `component` (str, optional): name of the component.  If not given,
            component will default to the top-most level of the current
            hierarchy.  See <phoebe.parameters.HierarchyParameter.get_top>.
        * `period` (str or float, optional, default='period'): qualifier of the parameter
            to be used for t0.  For orbits, can either be 'period' or 'period_sidereal'.
            For stars, must be 'period'.
        * `t0` (str or float, optional, default='t0_supconj'): qualifier of the parameter
            to be used for t0.  Must be 't0' for 't0@system' or a valid qualifier
            (eg. 't0_supconj', 't0_perpass', 't0_ref' for binary orbits.)
            For single stars, `t0` will be used if a float or integer, otherwise
            t0@system will be used, ignoring the passed value of `t0`.
        * `**kwargs`: any value passed through kwargs will override the
            ephemeris retrieved by component (ie period, t0, dpdt).
            Note: be careful about units - input values will not be converted.

        Returns
        -----------
        * (dict): dictionary containing period, t0 (t0_supconj if orbit),
            dpdt (as applicable)

        Raises
        ---------
        * ValueError: if `shift` is passed to `**kwargs`.
        * NotImplementedError: if the component kind is not recognized or supported.
        """

        if component is None:
            component = self.hierarchy.get_top()

        if kwargs.get('shift', False):
            raise ValueError("support for phshift was removed as of 2.1.  Please pass t0 instead.")

        ret = {}

        ps = self.filter(component=component, context='component', **_skip_filter_checks)

        if isinstance(period, str):
            ret['period'] = ps.get_value(qualifier=period, unit=u.d, **_skip_filter_checks)
        elif isinstance(period, float) or isinstance(period, int):
            ret['period'] = period
        else:
            raise ValueError("period must be a string (qualifier) or float")

        if ps.kind in ['orbit']:
            # TODO: ability to pass period to grab period_sidereal instead?
            if isinstance(t0, str):
                if t0 == 't0':
                    ret['t0'] = self.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)
                else:
                    ret['t0'] = ps.get_value(qualifier=t0, unit=u.d, **_skip_filter_checks)
            elif isinstance(t0, float) or isinstance(t0, int):
                ret['t0'] = t0
            else:
                raise ValueError("t0 must be string (qualifier) or float")
            ret['dpdt'] = ps.get_value(qualifier='dpdt', unit=u.d/u.d)

        elif ps.kind in ['star']:
            if isinstance(t0, float) or isinstance(t0, int):
                ret['t0'] = t0
            else:
                ret['t0'] = self.get_value('t0', context='system', unit=u.d, **_skip_filter_checks)
        else:
            raise NotImplementedError

        for k,v in kwargs.items():
            if k=='dpdt' and isinstance(v, str):
                if v.lower() == 'none':
                    # for example, passing dpdt = 'none' (via figure)
                    v = 0.0
                elif v == 'dpdt':
                    continue
                else:
                    raise ValueError("dpdt={} not implemented".format(v))
            ret[k] = v

        return ret

    def to_phase(self, time, component=None, period='period', t0='t0_supconj', **kwargs):
        """
        Get the phase(s) of a time(s) for a given ephemeris.

        The definition of time-to-phase used here is:
        ```
        if dpdt != 0:
            phase = np.mod(1./dpdt * np.log(1 + dpdt/period*(time-t0)), 1.0)
        else:
            phase = np.mod((time-t0)/period, 1.0)
        ```

        See also:
        * <phoebe.frontend.bundle.Bundle.to_time>
        * <phoebe.frontend.bundle.Bundle.get_ephemeris>.

        Arguments
        -----------
        * `time` (float/list/array): time to convert to phases (should be in
            same system/units as t0s)
        * `component` (str, optional): component for which to get the ephemeris.
            If not given, component will default to the top-most level of the
            current hierarchy.  See <phoebe.parameters.HierarchyParameter.get_top>.
        * `period` (str or float, optional, default='period'): qualifier of the parameter
            to be used for t0.  For orbits, can either be 'period' or 'period_sidereal'.
            For stars, must be 'period'.
        * `t0` (str or float, optional, default='t0_supconj'): qualifier of the parameter
            to be used for t0 ('t0_supconj', 't0_perpass', 't0_ref'), passed
            to <phoebe.frontend.bundle.Bundle.get_ephemeris>.
        * `**kwargs`: any value passed through kwargs will override the
            ephemeris retrieved by component (ie period, t0, dpdt).
            Note: be careful about units - input values will not be converted.

        Returns:
        ----------
        * (float/array) phases in same type as input times (except lists become arrays).

        Raises
        ---------
        * ValueError: if `shift` is passed to `**kwargs`.
        * NotImplementedError: if the component kind is not recognized or supported.
        """

        if kwargs.get('shift', False):
            raise ValueError("support for phshift was removed as of 2.1.  Please pass t0 instead.")

        ephem = self.get_ephemeris(component=component, period=period, t0=t0, **kwargs)

        if isinstance(time, list):
            time = np.array(time)
        elif isinstance(time, Parameter):
            time = time.get_value(u.d)
        elif isinstance(time, str):
            time = self.get_value(time, u.d)

        t0 = ephem.get('t0', 0.0)
        period = ephem.get('period', 1.0)
        dpdt = ephem.get('dpdt', 0.0)


        # if changing this, also see parameters.constraint.time_ephem
        # and phoebe.constraints.builtin.times_to_phases
        # and update docstring above
        if dpdt != 0:
            phase = np.mod(1./dpdt * np.log(1 + dpdt/period*(time-t0)), 1.0)
        else:
            phase = np.mod((time-t0)/period, 1.0)

        if isinstance(phase, float):
            if phase > 0.5:
                phase -= 1
        else:
            # then should be an array
            phase[phase > 0.5] -= 1

        return phase

    def to_phases(self, *args, **kwargs):
        """
        Alias to <phoebe.frontend.bundle.Bundle.to_phase>.
        """
        return self.to_phase(*args, **kwargs)

    def to_time(self, phase, component=None, period='period', t0='t0_supconj', **kwargs):
        """
        Get the time(s) of a phase(s) for a given ephemeris.

        The definition of phase-to-time used here is:
        ```
        if dpdt != 0:
            time = t0 + period/dpdt*(np.exp(dpdt*(phase))-1.0)
        else:
            time = t0 + (phase)*period
        ```

        See also:
        * <phoebe.frontend.bundle.Bundle.to_phase>
        * <phoebe.frontend.bundle.Bundle.get_ephemeris>.

        Arguments
        -----------
        * `phase` (float/list/array): phase to convert to times (should be in
            same system/units as t0s)
        * `component` (str, optional): component for which to get the ephemeris.
            If not given, component will default to the top-most level of the
            current hierarchy.  See <phoebe.parameters.HierarchyParameter.get_top>.
        * `period` (str or float, optional, default='period'): qualifier of the parameter
            to be used for t0.  For orbits, can either be 'period' or 'period_sidereal'.
            For stars, must be 'period'.
        * `t0` (str or float, optional, default='t0_supconj'): qualifier of the parameter
            to be used for t0 ('t0_supconj', 't0_perpass', 't0_ref'), passed
            to <phoebe.frontend.bundle.Bundle.get_ephemeris>.
        * `**kwargs`: any value passed through kwargs will override the
            ephemeris retrieved by component (ie period, t0, dpdt).
            Note: be careful about units - input values will not be converted.

        Returns
        ----------
        * (float/array) times in same type as input phases (except lists become arrays).

        Raises
        ---------
        * ValueError: if `shift` is passed to `**kwargs`.
        * NotImplementedError: if the component kind is not recognized or supported.
        """

        if kwargs.get('shift', False):
            raise ValueError("support for phshift was removed as of 2.1.  Please pass t0 instead.")

        ephem = self.get_ephemeris(component=component, period=period, t0=t0, **kwargs)

        if isinstance(phase, list):
            phase = np.array(phase)

        t0 = ephem.get('t0', 0.0)
        period = ephem.get('period', 1.0)
        dpdt = ephem.get('dpdt', 0.0)

        # if changing this, also see parameters.constraint.time_ephem
        # and phoebe.constraints.builtin.phases_to_times
        # and update docstring above
        if dpdt != 0:
            time = t0 + period/dpdt*(np.exp(dpdt*(phase))-1.0)
        else:
            time = t0 + (phase)*period

        return time

    def to_times(self, *args, **kwargs):
        """
        Alias to <phoebe.frontend.bundle.Bundle.to_time>.
        """
        return self.to_time(*args, **kwargs)

    @send_if_client
    def add_dataset(self, kind, component=None, return_changes=False, **kwargs):
        """
        Add a new dataset to the bundle.  If not provided,
        `dataset` (the name of the new dataset) will be created for
        you and can be accessed by the `dataset` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        For light curves, the light curve will be generated for the entire system.

        For radial velocities, you need to provide a list of components
        for which values should be computed.

        Available kinds can be found in <phoebe.parameters.dataset> or by calling
        <phoebe.list_available_datasets> and include:
        * <phoebe.parameters.dataset.lc>
        * <phoebe.parameters.dataset.rv>
        * <phoebe.parameters.dataset.lp>
        * <phoebe.parameters.dataset.orb>
        * <phoebe.parameters.dataset.mesh>

        The value of `component` will default as follows:
        * lc: defaults to `None` meaning the light curve is computed
            for the entire system.  This is the only valid option.
        * mesh: defaults to `None` meaning all components will be exposed.
            This is the only valid option.
        * rv or orb: defaults to the stars in the hierarchy.  See also
            <phoebe.parameters.HierarchyParameter.get_stars>.  Optionally,
            you can override this by providing a subset of the stars in the
            hierarchy.
        * lp: defaults to the top-level of the hierarchy (typically an orbit).
            See also <phoebe.parameters.HierarchyParameter.get_top>.  The
            exposed line-profile is then the combined line profile of all
            children components.  Optionally, you can override this by providing
            a subset (or single entry) of the stars or orbits in the hierarchy.

        Additional keyword arguments (`**kwargs`) will be applied to the resulting
        parameters, whenever possible.  See <phoebe.parameters.ParameterSet.set_value>
        for changing the values of a <phoebe.parameters.Parameter> after it has
        been attached.

        The following formats are acceptable, when applicable:

        * when passed as a single key-value pair (`times = [0,1,2,3]`), the passed
            value will be applied to all parameters with qualifier of 'times',
            including any with component = '_default' (in which case the value
            will be copied to new parameters whenver a new component is added
            to the system).
        * when passed as a single key-value pair (`times = [0, 1, 2, 3]`), **but**
            `component` (or `components`) is also passed (`component = ['primary']`),
            the passed value will be applied to all parameters with qualifier
            of 'times' and one of the passed components.  In this case, component
            = '_default' will not be included, so the value will not be copied
            to new parameters whenever a new component is added.
        * when passed as a dictionary (`times = {'primary': [0,1], 'secondary': [0,1,2]}`),
            separate values will be applied to parameters based on the component
            provided (eg. for different times/rvs per-component for RV datasets)
            or any general twig filter (eg. for different flux_densities per-time
            and per-component: `flux_densities = {'0.00@primary': [...], ...}`).
            Note that component = '_default' will only be set if it is included
            in the dictionary.

        Arguments
        ----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts only default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.compute> module.
        * `component` (list, optional): a list of components for which to compute
            the observables.  For light curves this should be left at None to always
            compute the light curve for the entire system.  See above for the
            valid options for `component` and how it will default if not provided
            based on the value of `kind` as well as how it affects the application
            of any passed values to `**kwargs`.
        * `dataset` (string, optional): name of the newly-created dataset.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing dataset with the same `dataset` tag.  If False,
            an error will be raised if a dataset already exists with the same name.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).  See examples
            above for acceptable formats.

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added


        Raises
        ----------
        * ValueError: if a dataset with the provided `dataset` already exists,
            but `overwrite` is not set to True.
        * NotImplementedError: if a required constraint is not implemented.
        """

        sing_plural = {}
        sing_plural['time'] = 'times'
        sing_plural['flux'] = 'fluxes'
        sing_plural['sigma'] = 'sigmas'
        sing_plural['rv'] = 'rvs'

        func = _get_add_func(_dataset, kind.lower()
                             if isinstance(kind, str)
                             else kind)


        # remove if None
        if kwargs.get('dataset', False) is None:
            _ = kwargs.pop('dataset')

        kwargs.setdefault('dataset',
                          self._default_label(func.__name__,
                                              **{'context': 'dataset',
                                                 'kind': func.__name__}))


        if kwargs.pop('check_label', True):
            self._check_label(kwargs['dataset'], allow_overwrite=kwargs.get('overwrite', False))

        kind = func.__name__

        # Let's remember if the user passed components or if they were automatically assigned
        user_provided_components = component or kwargs.get('components', False)

        if kind == 'lc':
            allowed_components = [None]
            default_components = allowed_components
        elif kind in ['rv', 'orb']:
            allowed_components = self.hierarchy.get_stars() # + self.hierarchy.get_orbits()
            default_components = self.hierarchy.get_stars()
            # TODO: how are we going to handle overcontacts dynamical vs flux-weighted
        elif kind in ['mesh']:
            # allowed_components = self.hierarchy.get_meshables()
            allowed_components = [None]
            # allowed_components = self.hierarchy.get_stars()
            # TODO: how will this work when changing hierarchy to add/remove the common envelope?
            default_components = allowed_components
        elif kind in ['etv']:
            hier = self.hierarchy
            stars = hier.get_stars()
            # only include components in which the sibling is also a star that
            # means that the companion in a triple cannot be timed, because how
            # do we know who it's eclipsing?
            allowed_components = [s for s in stars if hier.get_sibling_of(s) in stars]
            default_components = allowed_components
        elif kind in ['lp']:
            # TODO: need to think about what this should be for contacts...
            allowed_components = self.hierarchy.get_stars() + self.hierarchy.get_orbits()
            default_components = [self.hierarchy.get_top()]

        else:
            allowed_components = [None]
            default_components = [None]

        # Let's handle the case where the user accidentally sends components
        # instead of component
        if kwargs.get('components', None) and component is None:
            logger.warning("assuming you meant 'component' instead of 'components'")
            components = kwargs.pop('components')
        else:
            components = component

        if isinstance(components, str):
            components = [components]
        elif hasattr(components, '__iter__'):
            components = components
        elif components is None:
            components = default_components
        else:
            raise NotImplementedError

        # Let's handle the case where the user accidentally sends singular
        # instead of plural (since we used to have this)
        # TODO: use parameter._singular_to_plural?
        for singular, plural in sing_plural.items():
            if kwargs.get(singular, None) is not None and kwargs.get(plural, None) is None:
                logger.warning("assuming you meant '{}' instead of '{}'".format(plural, singular))
                kwargs[plural] = kwargs.pop(singular)

        if not np.all([component in allowed_components
                       for component in components]):
            raise ValueError("'{}' not a recognized/allowable component".format(component))

        ds_metawargs = {'context': 'dataset',
                         'kind': kind,
                         'dataset': kwargs['dataset']}

        if kind in ['lp']:
            # then times needs to be passed now to duplicate and tag the Parameters
            # correctly
            ds_kwargs = {'times': kwargs.pop('times', [])}
        else:
            ds_kwargs = {}

        # temporarily disable interactive_checks, check_default, and check_visible
        conf_interactive_checks = conf.interactive_checks
        if conf_interactive_checks:
            logger.debug("temporarily disabling interactive_checks")
            conf._interactive_checks = False

        params, constraints = func(dataset=kwargs['dataset'], component_top=self.hierarchy.get_top(), **ds_kwargs)

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_dataset(dataset=kwargs['dataset'], during_overwrite=True)
            # check the label again, just in case kwargs['dataset'] belongs to
            # something other than dataset
            self._check_label(kwargs['dataset'], allow_overwrite=False)

        self._attach_params(params, **ds_metawargs)

        for constraint in constraints:
            self.add_constraint(*constraint)


        # Figure options for this dataset
        if kind not in ['mesh']:
            fig_params = _figure._add_dataset(self, **kwargs)

            fig_metawargs = {'context': 'figure',
                             'kind': kind,
                             'dataset': kwargs['dataset']}
            self._attach_params(fig_params, **fig_metawargs)

        else:
            fig_params = None



        if self.get_value(qualifier='auto_add_figure', context='setting', auto_add_figure=kwargs.get('auto_add_figure', None), **_skip_filter_checks) and kind not in self.filter(context='figure', check_visible=False, check_default=False).exclude(figure=[None], check_visible=False, check_default=False).kinds:
            # then we don't have a figure for this kind yet
            logger.info("calling add_figure(kind='dataset.{}') since auto_add_figure@setting=True".format(kind))
            new_fig_params = self.add_figure(kind='dataset.{}'.format(kind))
        else:
            new_fig_params = None


        # Now we need to apply any kwargs sent by the user.  See the API docs
        # above for more details and make sure to update there if the options here
        # change.  There are a few scenarios (and each kwargs could fall into different ones):
        # times = [0,1,2]
        #    in this case, we want to apply time across all of the components that
        #    are applicable for this dataset kind AND to _default so that any
        #    future components added to the system are copied appropriately
        # times = [0,1,2], components=['primary', 'secondary']
        #    in this case, we want to apply the value for time across components
        #    but time@_default should remain empty (it will not copy for components
        #    added in the future)
        # times = {'primary': [0,1], 'secondary': [0,1,2]}
        #    here, regardless of the components, we want to apply these to their
        #    individually requested parameters.  We won't touch _default unless
        #    its included in the dictionary

        # this needs to happen before kwargs get applied so that the default
        # values can be overridden by the supplied kwargs
        self._handle_pblum_defaults()
        self._handle_dataset_selectparams()

        if 'compute_phases' in kwargs.keys():
            if 'compute_times' in kwargs.keys():
                self.remove_dataset(dataset=kwargs['dataset'])
                raise ValueError("cannot provide both 'compute_phases' and 'compute_times'. Dataset has not been added.")
            elif kind in ['mesh', 'orb'] and 'times' in kwargs.keys():
                self.remove_dataset(dataset=kwargs['dataset'])
                raise ValueError("cannot provide both 'compute_phases' and 'compute_times' for a {} dataset. Dataset has not been added.".format(kind))
            else:
                # then we must flip the constraint
                # TODO: this will probably break with triple support - we'll need to handle the multiple orbit components by accepting the dictionary.
                # For now we'll assume the component is top-level binary
                self.flip_constraint('compute_phases', component=self.hierarchy.get_top(), dataset=kwargs['dataset'], solve_for='compute_times')

        if kind in ['mesh','orb'] and 'times' in kwargs.keys():
            # we already checked and would have raised an error if compute_phases
            # was provided, but we still need to handle compute_times
            if 'compute_times' in kwargs.keys():
                self.remove_dataset(dataset=kwargs['dataset'])
                raise ValueError("cannot provide both 'compute_times' and 'times' (which would write to 'compute_times') for a {} dataset.  Dataset has not been added.".format(kind))

            # if we're this far, the user passed times, but not compute_times/phases
            logger.warning("{} dataset uses 'compute_times' instead of 'times', applying value sent as 'times' to 'compute_times'.".format(kind))
            kwargs['compute_times'] = kwargs.pop('times')

        if 'pblum_mode' in kwargs.keys():
            # we need to set this first so that pblum visibilities are set
            # before we enter the loop

            v = kwargs.pop('pblum_mode')
            k = 'pblum_mode'
            components_ = None

            # we shouldn't need to worry about a dictionary here since there
            # are no components, but let's just check and raise an error if it is.
            if isinstance(v, dict):
                raise TypeError("pblum_mode cannot be passed as a dictionary")

            try:
                self.set_value_all(qualifier=k,
                                   dataset=kwargs['dataset'],
                                   component=components_,
                                   value=v,
                                   check_visible=False,
                                   ignore_none=True)
            except Exception as err:
                self.remove_dataset(dataset=kwargs['dataset'])
                raise ValueError("could not set value for {}={} with error: '{}'. Dataset has not been added".format(k, value, str(err)))


        for k, v in kwargs.items():
            if k in ['dataset']:
                pass
            elif isinstance(v, dict):
                for component_or_twig, value in v.items():
                    ps = self.filter(qualifier=k,
                                     dataset=kwargs['dataset'],
                                     check_visible=False,
                                     check_default=False,
                                     ignore_none=True)

                    if component_or_twig in ps.components:
                        component = component_or_twig
                        logger.debug("setting value of dataset parameter: qualifier={}, dataset={}, component={}, value={}".format(k, kwargs['dataset'], component, value))
                        try:
                            self.set_value_all(qualifier=k,
                                               dataset=kwargs['dataset'],
                                               component=component,
                                               value=value,
                                               check_visible=False,
                                               check_default=False,
                                               ignore_none=True)
                        except Exception as err:
                            self.remove_dataset(dataset=kwargs['dataset'])
                            raise ValueError("could not set value for {}={} with error: '{}'. Dataset has not been added".format(k, value, str(err)))
                    elif len(ps.filter(component_or_twig, check_visible=False, check_default=False).to_list()) >= 1:
                        twig = component_or_twig
                        logger.debug("setting value of dataset parameter: qualifier={}, twig={}, component={}, value={}".format(k, kwargs['dataset'], twig, value))
                        try:
                            self.set_value_all(twig,
                                               qualifier=k,
                                               dataset=kwargs['dataset'],
                                               value=value,
                                               check_visible=False,
                                               check_default=False,
                                               ignore_none=True)
                        except Exception as err:
                            self.remove_dataset(dataset=kwargs['dataset'])
                            raise ValueError("could not set value for {}={} with error: '{}'. Dataset has not been added".format(k, value, str(err)))
                    else:
                        self.remove_dataset(dataset=kwargs['dataset'])
                        raise ValueError("could not set value for {}={}.  {} did not match either a component or general filter.  Dataset has not been added".format(k, value, component_or_twig))

            else:
                # for dataset kinds that include passband dependent AND
                # independent parameters, we need to carefully default on
                # what component to use when passing the defaults
                check_visible = False
                if kind in ['rv', 'lp'] and k in ['ld_func', 'ld_coeffs',
                                                  'passband', 'intens_weighting',
                                                  'profile_rest', 'profile_func', 'profile_sv']:
                    # passband-dependent parameters do not have
                    # assigned components
                    components_ = None
                elif k in ['compute_times']:
                    components_ = None
                elif k in ['compute_phases']:
                    components_ = self.hierarchy.get_top()
                elif k in ['pblum']:
                    check_visible = True

                    components_ = self.hierarchy.get_stars()+['_default']
                elif components == [None]:
                    components_ = None
                elif user_provided_components:
                    components_ = components
                else:
                    components_ = components+['_default']

                logger.debug("setting value of dataset parameter: qualifier={}, dataset={}, component={}, value={}".format(k, kwargs['dataset'], components_, v))
                try:
                    self.set_value_all(qualifier=k,
                                       dataset=kwargs['dataset'],
                                       component=components_,
                                       value=v,
                                       check_visible=check_visible,
                                       ignore_none=True)
                except Exception as err:
                    self.remove_dataset(dataset=kwargs['dataset'])
                    if conf_interactive_checks:
                        logger.debug("reenabling interactive_checks")
                        conf._interactive_checks = True
                        self.run_checks(raise_logger_warning=True)

                    raise ValueError("could not set value for {}={} with error: '{}'. Dataset has not been added.".format(k, v, str(err)))


        def _to_safe_value(v):
            if isinstance(v, nparray.ndarray):
                return v.to_json()
            elif isinstance(v, dict):
                return {k: _to_safe_value(v) for k,v in v.items()}
            else:
                return v

        if conf_interactive_checks:
            logger.debug("reenabling interactive_checks")
            conf._interactive_checks = True
            self.run_checks(raise_logger_warning=True)

        ret_ps = self.filter(dataset=kwargs['dataset'], **_skip_filter_checks)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite'], warning_only=True, ps=ret_ps)

        if new_fig_params is not None:
            ret_ps += new_fig_params

        if kwargs.get('overwrite', False) and return_changes:
            ret_ps += overwrite_ps

        ret_changes = []
        ret_changes += self._handle_dataset_selectparams(return_changes=return_changes)
        ret_changes += self._handle_figure_time_source_params(return_changes=return_changes)
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_dataset(self, dataset=None, **kwargs):
        """
        Filter in the 'dataset' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `dataset`: (string, optional, default=None): the name of the dataset
        * `**kwargs`: any other tags to do the filtering (excluding dataset and context)

        Returns
        --------
        * a <phoebe.parameters.ParameterSet> object.
        """
        if dataset is not None:
            kwargs['dataset'] = dataset
            if dataset not in self.datasets:
                raise ValueError("dataset='{}' not found".format(dataset))

        kwargs['context'] = 'dataset'
        if 'kind' in kwargs.keys():
            # since we switched how dataset kinds are named, let's just
            # automatically handle switching to lowercase
            kwargs['kind'] = kwargs['kind'].lower()
        return self.filter(**kwargs)

    @send_if_client
    def remove_dataset(self, dataset=None, return_changes=False, **kwargs):
        """
        Remove a 'dataset' from the Bundle.

        This removes all matching Parameters from the dataset, model, and
        constraint contexts (by default if the context tag is not provided).

        You must provide some sort of filter or this will raise an Error (so
        that all Parameters are not accidentally removed).

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `dataset` (string, optional): the label of the dataset to be removed.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: dataset, qualifier.

        Returns
        -----------
        * ParameterSet of removed or changed parameters

        Raises
        --------
        * ValueError: if `dataset` is not provided AND no `kwargs` are provided.
        """

        self._kwargs_checks(kwargs, ['during_overwrite'])

        # Let's avoid deleting ALL parameters from the matching contexts
        if dataset is None and not len(kwargs.items()):
            raise ValueError("must provide some value to filter for datasets")

        if dataset is None:
            # then let's find the list of datasets that match the filter,
            # we'll then use dataset to do the removing.  This avoids leaving
            # pararameters behind that don't specifically match the filter
            # (ie if kind is passed as 'rv' we still want to remove parameters
            # with datasets that are RVs but belong to a different kind in
            # another context like compute)
            dataset = self.filter(**kwargs).datasets
            kwargs['kind'] = None


        kwargs['dataset'] = dataset
        # Let's avoid the possibility of deleting a single parameter
        kwargs['qualifier'] = None
        # Let's also avoid the possibility of accidentally deleting system
        # parameters, etc
        kwargs.setdefault('context', ['dataset', 'model', 'constraint', 'compute', 'figure'])

        ret_ps = self.remove_parameters_all(**kwargs)
        # not really sure why we need to call this twice, but it seems to do
        # the trick
        ret_ps += self.remove_parameters_all(**kwargs)

        ret_changes = []
        if not kwargs.get('during_overwrite', False):
            ret_changes += self._handle_dataset_selectparams(return_changes=return_changes)
            ret_changes += self._handle_figure_time_source_params(return_changes=return_changes)
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)


        if self.get_value(qualifier='auto_remove_figure', context='setting'):
            # then we don't have a figure for this kind yet
            for param in self.filter(qualifier='datasets', context='figure', kind=ret_ps.kind, **_skip_filter_checks).to_list():
                if not len(param.choices):
                    logger.info("calling remove_figure(figure='{}') since auto_remove_figure@setting=True".format(param.figure))
                    ret_changes += self.remove_figure(figure=param.figure, return_changes=return_changes).to_list()


        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def remove_datasets_all(self, return_changes=False):
        """
        Remove all datasets from the bundle.  To remove a single dataset see
        <phoebe.frontend.bundle.Bundle.remove_dataset>.

        Arguments
        ----------
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for dataset in self.datasets:
            removed_ps += self.remove_dataset(dataset=dataset, return_changes=return_changes)

        return removed_ps

    @send_if_client
    def rename_dataset(self, old_dataset, new_dataset,
                       overwrite=False, return_changes=False):
        """
        Change the label of a dataset attached to the Bundle.

        Arguments
        ----------
        * `old_dataset` (string): current label of the dataset (must exist)
        * `new_dataset` (string): the desired new label of the dataset
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_dataset` is forbidden or already exists.
        """
        # TODO: raise error if old_component not found?
        self._rename_label('dataset', old_dataset, new_dataset, overwrite)

        ret_ps = self.filter(dataset=new_dataset)

        ret_changes = []
        ret_changes += self._handle_dataset_selectparams(return_changes=return_changes)
        ret_changes += self._handle_figure_time_source_params(return_changes=return_changes)
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)

        for param in self.filter(context='solution', qualifier='lc', **_skip_filter_checks):
            if param.get_value() == old_value:
                param.set_value(new_value)
                ret_changes += [param]

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)


    def enable_dataset(self, dataset=None, **kwargs):
        """
        Enable a `dataset`.  Datasets that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_solver.

        If `compute` is not provided, the dataset will be enabled across all
        compute options.

        Note that not all `compute` backends support all types of datasets.
        Unsupported datasets do not have 'enabled' parameters, and therefore
        cannot be enabled or disabled.

        See also:
        * <phoebe.frontend.bundle.Bundle.diable_dataset>

        Arguments
        -----------
        * `dataset` (string, optional): name of the dataset
        * `**kwargs`:  any other tags to do the filter
            (except dataset or context)

        Returns
        ---------
        * a <phoebe.parameters.ParameterSet> object of the enabled dataset
        """
        if 'kind' in kwargs.keys():
            # we'll disable this for now as kind could either refer to dataset
            # (which the enabled parameter is not tagged with) or compute
            raise ValueError("cannot pass kind to enable_dataset")

        kwargs['context'] = 'compute'
        kwargs['dataset'] = dataset
        kwargs['qualifier'] = 'enabled'
        self.set_value_all(value=True, **kwargs)

        return self.get_dataset(dataset=dataset)

    def disable_dataset(self, dataset=None, **kwargs):
        """
        Disable a `dataset`.  Datasets that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_solver.

        If `compute` is not provided, the dataset will be disabled across all
        compute options.

        Note that not all `compute` backends support all types of datasets.
        Unsupported datasets do not have 'enabled' parameters, and therefore
        cannot be enabled or disabled.

        See also:
        * <phoebe.frontend.bundle.Bundle.enable_dataset>

        Arguments
        -----------
        * `dataset` (string, optional): name of the dataset
        * `**kwargs`:  any other tags to do the filter
            (except dataset or context)

        Returns
        ---------
        * a <phoebe.parameters.ParameterSet> object of the disabled dataset
        """
        if 'kind' in kwargs.keys():
            # we'll disable this for now as kind could either refer to dataset
            # (which the enabled parameter is not tagged with) or compute
            raise ValueError("cannot pass kind to disable_dataset")

        kwargs['context'] = 'compute'
        kwargs['dataset'] = dataset
        kwargs['qualifier'] = 'enabled'
        self.set_value_all(value=False, **kwargs)

        return self.get_dataset(dataset=dataset)

    @send_if_client
    def add_constraint(self, *args, **kwargs):
        """
        Add a <phoebe.parameters.ConstraintParameter> to the
        <phoebe.frontend.bundle.Bundle>.

        See also:
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        For a list of optional built-in constraints, see <phoebe.parameters.constraint>
        including:
        * <phoebe.parameters.constraint.semidetached>
        * <phoebe.parameters.constraint.requivfrac>
        * <phoebe.parameters.constraint.requivratio>
        * <phoebe.parameters.constraint.requivsumfrac>
        * <phoebe.parameters.constraint.impact_param>
        * <phoebe.parameters.constraint.teffratio>
        * <phoebe.parameters.constraint.parallax>

        The following are automatically included for all orbits, during
        <phoebe.frontend.bundle.Bundle.add_component> for a
        <phoebe.parameters.component.orbit>:
        * <phoebe.parameters.constraint.asini>
        * <phoebe.parameters.constraint.ecosw>
        * <phoebe.parameters.constraint.esinw>
        * <phoebe.parameters.constraint.period_anom>
        * <phoebe.parameters.constraint.t0_perpass_supconj>
        * <phoebe.parameters.constraint.t0_ref_supconj>
        * <phoebe.parameters.constraint.mean_anom>
        * <phoebe.parameters.constraint.freq>

        The following are automatically included for all stars, during
        <phoebe.frontend.bundle.Bundle.add_component> for a
        <phoebe.parameters.component.star>:
        * <phoebe.parameters.constraint.freq>
        * <phoebe.parameters.constraint.irrad_frac>
        * <phoebe.parameters.constraint.logg>

        The following are automatically included for all applicable datasets,
        during <phoebe.frontend.bundle.Bundle.add_dataset>:
        * <phoebe.parameters.constraint.compute_times>
        * <phoebe.parameters.constraint.extinction>

        Additionally, some constraints are automatically handled by the hierarchy in
        <phoebe.frontend.bundle.Bundle.set_hierarchy> or when loading a default
        system.  The following are automatically included for a
        <phoebe.frontend.bundle.Bundle.default_binary>:
        * <phoebe.parameters.constraint.mass>
        * <phoebe.parameters.constraint.comp_sma>
        * <phoebe.parameters.constraint.comp_asini>
        * <phoebe.parameters.constraint.rotation_period> (detached only)
        * <phoebe.parameters.constraint.pitch> (detached only)
        * <phoebe.parameters.constraint.yaw> (detached only)
        * <phoebe.parameters.constraint.requiv_detached_max> (detached only)
        * <phoebe.parameters.constraint.potential_contact_min> (contact only)
        * <phoebe.parameters.constraint.potential_contact_max> (contact only)
        * <phoebe.parameters.constraint.requiv_contact_min> (contact only)
        * <phoebe.parameters.constraint.requiv_contact_max> (contact only)
        * <phoebe.parameters.constraint.fillout_factor> (contact only)
        * <phoebe.parameters.constraint.requiv_to_pot> (contact only)

        To add a custom constraint, pass the left-hand side (as a <phoebe.parameters.FloatParameter>)
        and the right-hand side (as a <phoebe.parameters.ConstraintParameter>).
        For example:

        ```
        lhs = b.get_parameter(qualifier='teff', component='secondary')
        rhs = 0.6 * b.get_parameter(qualifier='teff', component='primary')
        b.add_constraint(lhs, rhs)
        ```

        Arguments
        ------------
        * `*args`: positional arguments can be any one of the following:
            * lhs (left-hand side parameter) and rhs (right-hand side parameter or
                ConstraintParameter) of a custom constraint.
            * valid string representation of a constraint
            * callable function (possibly in <phoebe.parameters.constraint>)
                followed by arguments that return a valid string representation
                of the constraint.
        * `kind` (string, optional): kind of the constraint function to find in
            <phoebe.parameters.constraint>
        * `func` (string, optional): func of the constraint to find in
            <phoebe.parameters.constraint>
        * `constraint_func` (string, optional): constraint_func of the constraint
            to find in <phoebe.parameters.constraint>
        * `solve_for` (string, optional): twig/qualifier in the constraint to solve
            for.  See also <phoebe.frontend.bundle.Bundle.flip_constraint>.

        Returns
        ---------
        * a <phoebe.parameters.ParameterSet> of the created constraint.
        """
        # TODO: be smart enough to take kwargs (especially for undoing a
        # remove_constraint) for kind, value (expression),

        redo_kwargs = _deepcopy(kwargs)

        if len(args) == 1 and \
                isinstance(args[0], str) and \
                not _get_add_func(_constraint, args[0],
                                  return_none_if_not_found=True):
            # then only the expression has been passed,
            # we just need to pass it on to constraints.custom

            func = constraint.custom
            func_args = args

        elif len(args) == 2 and \
                all([isinstance(arg, Parameter) or
                     isinstance(arg, ConstraintParameter) for arg in args]):
            # then we have 2 constraint expressions

            func = constraint.custom
            func_args = args

        elif len(args) == 0:
            # then everything is passed through kwargs
            if 'kind' in kwargs.keys():
                func = _get_add_func(_constraint, kwargs['kind'])
            elif 'func' in kwargs.keys():
                func = _get_add_func(_constraint, kwargs['func'])
            elif 'constraint_func' in kwargs.keys():
                func = _get_add_func(_constraint, kwargs['constraint_func'])
            else:
                func = constraint.custom

            func_args = []

            # constraint_param = ConstraintParameter(self, **kwargs)

        else:
            # then we've been passed the function in constraints and its
            # arguments

            func = _get_add_func(_constraint, args[0])
            func_args = args[1:]

        # although we could pass solve_for IF the parameter already exists,
        # we'll just manually flip after to ensure it already does
        if 'solve_for' in kwargs.keys():
            try:
                kwargs['solve_for'] = self.get_parameter(kwargs['solve_for'], context=['component', 'dataset', 'model', 'system'], **_skip_filter_checks)
            except:
                solve_for = kwargs.pop('solve_for', None)
            else:
                solve_for = None
        else:
            solve_for = None


        lhs, rhs, addl_vars, constraint_kwargs = func(self, *func_args, **{k:v for k,v in kwargs.items() if k not in ['constraint']})
        # NOTE that any component parameters required have already been
        # created by this point

        constraint_param = ConstraintParameter(self,
                                               qualifier=lhs.qualifier,
                                               component=lhs.component,
                                               dataset=lhs.dataset,
                                               feature=lhs.feature,
                                               kind=lhs.kind,
                                               model=lhs.model,
                                               constraint_func=func.__name__,
                                               constraint_kwargs=constraint_kwargs,
                                               addl_vars=addl_vars,
                                               in_solar_units=func.__name__ not in constraint.list_of_constraints_requiring_si,
                                               value=rhs,
                                               default_unit=lhs.default_unit,
                                               description='expression that determines the constraint')


        newly_constrained_param = constraint_param.get_constrained_parameter()
        check_kwargs = {k:v for k,v in newly_constrained_param.meta.items() if k not in ['context', 'twig', 'uniquetwig']}
        check_kwargs['context'] = 'constraint'
        check_kwargs['check_visible'] = False
        if len(self._bundle.filter(**check_kwargs)):
            logger.debug("'{}' is constrained by {}".format(newly_constrained_param.twig, self._bundle.filter(**check_kwargs).twigs))
            raise ValueError("'{}' is already constrained".format(newly_constrained_param.twig))

        metawargs = {'context': 'constraint',
                     'kind': func.__name__}

        params = ParameterSet([constraint_param])
        constraint_param._update_bookkeeping()
        self._attach_params(params, **metawargs)

        if solve_for is not None:
            self.flip_constraint(uniqueid=constraint_param.uniqueid, solve_for=solve_for)

        # we should run it now to make sure everything is in-sync
        if conf.interactive_constraints:
            self.run_constraint(uniqueid=constraint_param.uniqueid, skip_kwargs_checks=True)
        else:
            self._delayed_constraints.append(constraint_param.uniqueid)

        return params
        # return self.get_constraint(**metawargs)

    def get_constraint(self, twig=None, **kwargs):
        """
        Filter in the 'constraint' context

        See also:
        * <phoebe.parameters.ParameterSet.get>
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns
        ---------
        * a <phoebe.parameters.Parameter> object.
        """
        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'constraint'
        return self.get(**kwargs)

    @send_if_client
    def remove_constraint(self, twig=None, return_changes=False, **kwargs):
        """
        Remove a 'constraint' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        Arguments
        ----------
        * `twig` (string, optional): twig to filter for the constraint.  The
            name of the constraint function (from <phoebe.parameters.constraint>)
            can also be passed.  For example: 'semidetached'.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: context, twig.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        # let's run delayed constraints first to ensure that we get the same
        # results in interactive and non-interactive modes as well as to make
        # sure we don't have delayed constraints for the constraint we're
        #  about to remove.  This could perhaps be optimized by searching
        #  for this/these constraints and only running/removing those, but
        #  probably isn't worth the savings.
        changed_params = self.run_delayed_constraints()

        kwargs['twig'] = twig
        redo_kwargs = _deepcopy(kwargs)

        kwargs['context'] = 'constraint'

        # we'll get the constraint so that we can undo the bookkeeping
        # and also reproduce an undo command
        try:
            constraint = self.get_parameter(**kwargs)
        except ValueError:
            # then perhaps the first passed argument was the constraint_func instead
            kwargs['constraint_func'] = kwargs.pop('twig')
            constraint = self.get_parameter(**kwargs)

        # undo parameter bookkeeping
        constraint._remove_bookkeeping()

        # and finally remove it
        removed_param = self.remove_parameter(**kwargs)


        undo_kwargs = {k: v for k, v in constraint.to_dict().items()
                       if v is not None and
                       k not in ['uniqueid', 'uniquetwig', 'twig',
                                 'Class', 'context']}

        return removed_param

    @send_if_client
    def flip_constraint(self, twig=None, solve_for=None, **kwargs):
        """
        Flip an existing constraint to solve for a different parameter.

        See also:
        * <phoebe.frontend.bundle.Bundle.flip_constraints_all>
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to filter the constraint.
            This (along with `**kwargs`) must resolve to a single constraint or
            else a ValueError will be raised.  See
            <phoebe.frontend.bundle.Bundle.flip_constraints_all> for flipping
            multiple constraints at once.
        * `solve_for` (string or Parameter, optional, default=None): twig or
            <phoebe.parameters.Parameter> object of the new parameter for which
            this constraint should constrain (solve for).
        * `**kwargs`: additional kwargs to use for filtering.

        Returns
        ---------
        * The <phoebe.parameters.ConstraintParameter>.

        Raises
        --------
        * ValueError: if the constraint cannot be flipped because one of the
            dependent parameters is currently nan.
        * ValueError: if cannot resolve to a single constraint.  See
            <phoebe.frontend.bundle.Bundle.flip_constraints_all> for flipping
            multiple constraints at once.
        """
        self._kwargs_checks(kwargs, additional_allowed_keys=['check_nan'])

        kwargs['twig'] = twig
        # kwargs['check_default'] = False
        # kwargs['check_visible'] = False
        redo_kwargs = _deepcopy(kwargs)
        undo_kwargs = _deepcopy(kwargs)

        changed_params = self.run_delayed_constraints()

        param = self.get_constraint(**kwargs)

        def _check_nan(value):
            if isinstance(value, np.ndarray):
                return np.any(np.isnan(value))
            else:
                return np.isnan(value)

        if kwargs.pop('check_nan', True) and np.any([_check_nan(p.get_value()) for p in param.vars.to_list() if hasattr(p, 'get_quantity')]):
            raise ValueError("cannot flip constraint while the value of {} is nan".format([p.twig for p in param.vars.to_list() if np.isnan(p.get_value())]))

        if solve_for is None:
            return param
        if isinstance(solve_for, Parameter):
            solve_for = solve_for.uniquetwig

        redo_kwargs['solve_for'] = solve_for
        undo_kwargs['solve_for'] = param.constrained_parameter.uniquetwig

        logger.info("flipping constraint '{}' to solve for '{}'".format(param.uniquetwig, solve_for))
        param.flip_for(solve_for, from_bundle_flip_constraint=True)

        # TODO: include this in the return for the UI?
        self._handle_fitparameters_selecttwigparams(return_changes=False)

        try:
            result = self.run_constraint(uniqueid=param.uniqueid, skip_kwargs_checks=True)
        except Exception as e:
            if param.uniqueid not in self._failed_constraints:
                self._failed_constraints.append(param.uniqueid)

                message_prefix = "Constraint '{}' raised the following error while flipping to solve for '{}'.  Consider flipping the constraint back or changing the value of one of {} until the constraint succeeds.  Original error: ".format(param.twig, solve_for, [p.twig for p in param.vars.to_list()])

                logger.error(message_prefix + str(e))

        return param

    def flip_constraints_all(self, twig=None, solve_for=None, **kwargs):
        """
        Flip multiple existing constraints to solve for a different parameter.

        See also:
        * <phoebe.frontend.bundle.Bundle.flip_constraint>
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to filter the constraint.
            If multiple constraints are returned from this and `**kwargs`, each
            will attempt to be flipped for the same value of `solve_for`
        * `solve_for` (string or Parameter, optional, default=None): twig or
            <phoebe.parameters.Parameter> object of the new parameter for which
            this constraint should constrain (solve for).
        * `**kwargs`: additional kwargs to use for filtering.

        Returns
        ---------
        * The <phoebe.parameters.ConstraintParameter>.

        Raises
        --------
        * ValueError: if the constraint cannot be flipped because one of the
            dependent parameters is currently nan.
        """
        self._kwargs_checks(kwargs, additional_allowed_keys=['check_nan'])

        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'constraint'

        constraints = self.filter(**kwargs)

        for constraint in constraints.to_list():
            self.flip_constraint(uniqueid=constraint.uniqueid, solve_for=solve_for)

    def run_constraint(self, twig=None, return_parameter=False, suppress_error=True, **kwargs):
        """
        Run a given 'constraint' now and set the value of the constrained
        parameter.  In general, there shouldn't be any need to manually
        call this - constraints should automatically be run whenever a
        dependent parameter's value is change.

        If interactive constraints are disabled via <phoebe.interactive_constraints_off>,
        then you can manually call this method or <phoebe.frontend.bundle.Bundle.run_delayed_constraints>
        to manually update the constraint value.

        See also:
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        Arguments
        -------------
        * `twig` (string, optional, default=None): twig to filter for the constraint
        * `return_parameter` (bool, optional, default=False): whether to
            return the constrained <phoebe.parameters.Parameter> (otherwise will
            return the resulting value).
        * `suppress_error` (bool, optional, default=True): if True, any errors
            while running the constraint will be availble via the logger at the
            'error' level and can be re-attempted via
            <phoebe.frontend.bundle.Bundle.run_failed_constraints>.  If False,
            any errors will be raised immediately.
        * `**kwargs`:  any other tags to do the filter (except twig or context)

        Returns
        -----------
        * (float or units.Quantity or <phoebe.parameters.Parameter) the resulting
            value of the constraint.  Or if `return_parameter=True`: then the
            <phoebe.parameters.Parameter> object itself.
        """
        if not kwargs.get('skip_kwargs_checks', False):
            self._kwargs_checks(kwargs)

        kwargs['twig'] = twig
        kwargs['context'] = 'constraint'
        # kwargs['qualifier'] = 'expression'
        kwargs['check_visible'] = False
        kwargs['check_default'] = False
        # print "***", kwargs
        expression_param = self.get_parameter(**kwargs)
        logger.debug("bundle.run_constraint {}".format(expression_param.twig))


        kwargs = {}
        kwargs['twig'] = None
        # TODO: this might not be the case, we just know its not in constraint
        kwargs['qualifier'] = expression_param.qualifier
        kwargs['component'] = expression_param.component
        kwargs['dataset'] = expression_param.dataset
        kwargs['feature'] = expression_param.feature
        kwargs['context'] = ['system']
        if kwargs['component'] is not None:
            kwargs['context'] += ['component']
        if kwargs['dataset'] is not None:
            kwargs['context'] += ['dataset']
        if kwargs['feature'] is not None:
            kwargs['context'] += ['feature']

        kwargs['check_visible'] = False
        kwargs['check_default'] = False
        constrained_param = self.get_parameter(**kwargs)

        try:
            result = expression_param.get_result(suppress_error=False)
        except Exception as e:
            if expression_param.uniqueid not in self._failed_constraints:
                self._failed_constraints.append(expression_param.uniqueid)
                new = True
            else:
                new = False

            message_prefix = "Constraint '{}' raised the following error while attempting to solve for '{}'.  Consider flipping the constraint or changing the value of one of {} until the constraint succeeds.  Original error: ".format(expression_param.twig, constrained_param.twig, [p.twig for p in expression_param.vars.to_list()])

            if suppress_error:
                if new:
                    logger.error(message_prefix + str(e))
                result = None
            else:
                if len(e.args) >= 1:
                    e.args = (message_prefix + str(e),) + e.args[1:]
                raise

        # we won't bother checking for arrays (we'd have to do np.all),
        # but for floats, let's only set the value if the value has changed.
        if not isinstance(result, float) or result != constrained_param.get_value():
            logger.debug("setting '{}'={} from '{}' constraint".format(constrained_param.uniquetwig, result, expression_param.uniquetwig))
            try:
                constrained_param.set_value(result, from_constraint=True, force=True)
            except Exception as e:
                if expression_param.uniqueid not in self._failed_constraints:
                    self._failed_constraints.append(expression_param.uniqueid)
                    new = True
                else:
                    new = False

                message_prefix = "Constraint '{}' raised the following error while setting the value of '{}'.  Original error: ".format(expression_param.twig, constrained_param.twig)

                if suppress_error:
                    if new:
                        logger.error(message_prefix + str(e))
                else:
                    if len(e.args) >= 1:
                        e.args = (message_prefix + str(e),) + e.args[1:]
                    raise


        if return_parameter:
            return constrained_param
        else:
            return result

    def run_delayed_constraints(self):
        """
        Manually run any delayed constraints.  In general, there shouldn't be any need to manually
        call this - constraints should automatically be run whenever a
        dependent parameter's value is change.

        If interactive constraints are disabled via <phoebe.interactive_constraints_off>,
        then you can manually call this method or <phoebe.frontend.bundle.Bundle.run_constraint>
        to manually update the constraint value.

        See also:
        * <phoebe.interactive_constraints_on>
        * <phoebe.interactive_constraints_off>
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>

        Returns
        ---------
        * (list): list of changed <phoebe.parameters.Parameter> objects.

        """
        changes = []
        delayed_constraints = self._delayed_constraints
        self._delayed_constraints = []
        for constraint_id in delayed_constraints:
            param = self.run_constraint(uniqueid=constraint_id, return_parameter=True, skip_kwargs_checks=True)
            if param not in changes:
                changes.append(param)
        if len(self._delayed_constraints):
            # some of the calls above may have delayed even more constraints,
            # we must keep calling recursively until they're all cleared
            changes += self.run_delayed_constraints()

        return changes

    def run_failed_constraints(self):
        """
        Attempt to rerun all failed constraints that may be preventing
        <phoebe.frontend.bundle.Bundle.run_checks> from succeeding.
        """
        changes = []
        failed_constraints = self._failed_constraints
        self._failed_constraints = []
        for constraint_id in failed_constraints:
            logger.debug("run_failed_constraints: {}".format(constraint_id))
            param = self.run_constraint(uniqueid=constraint_id, return_parameter=True, skip_kwargs_checks=True, suppress_error=False)
            if param not in changes:
                changes.append(param)
        return changes

    def run_all_constraints(self):
        """
        Run all constraints.  May be necessary if a previous bundle was saved while
        there were failed/delayed constraints, since those are not stored.
        """
        changes = []
        for constraint_id in [p.uniqueid for p in self.filter(context='constraint', **_skip_filter_checks).to_list()]:
            previous_value = self.get_parameter(uniqueid=constraint_id, **_skip_filter_checks).constrained_parameter.value
            param = self.run_constraint(uniqueid=constraint_id, return_parameter=True, skip_kwargs_checks=True, suppress_error=False)
            if param not in changes and not _is_equiv_array_or_float(param.value, previous_value):
                changes.append(param)
        return changes


    def _add_single_distribution(self, twig=None, value=None, return_changes=False, **kwargs):
        """
        Add a distribution to an existing or new `distribution`, tagged to reference an existing
        parameter.  Unlike other `add_` methods in the bundle, this does not
        add a ParameterSet from a specific kind, but rather adds a **single**
        <phoebe.parameters.DistributionParameter> to the bundle in context='distribution',
        tagged according to `twig` and `**kwargs`.

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig pointing to an existing
            parameter to reference with this distribution.  If provided, this
            (along with `**kwargs`) must point to a single parameter.
        * `value` (distl Distribution object, optional, default=None): the
            distribution to be applied to the created <phoebe.parameters.DistributionParameter>.
            If not provided, will be a delta function around the current value
            of the referenced parameter.
        * `distribution` (string, optional): name of the distribution set.  If already
            existing in the bundle, then the referenced parameter must not already
            have an attached distribution.
        * `**kwargs`: tags to filter for a matching parameter (and to tag the
            new <phoebe.parameters.DistributionParameter>).  This (along with `twig`)
            must point to a single parameter.

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all newly created parameters.

        Raises
        --------
        * ValueError: if `twig` and `**kwargs` do not resolve to a single parameter
        """

        if isinstance(twig, Parameter):
            ref_params = [twig]
            index_orig = None
        else:
            twig, index_orig = _extract_index_from_string(twig)
            if kwargs.get('uniqueid'):
                kwargs['uniqueid'], index_uniqueid = _extract_index_from_string(kwargs.get('uniqueid'))
                if index_orig is not None and index_uniqueid != index_orig:
                    raise ValueError("conflicting indices found!")
                elif index_uniqueid is not None:
                    index_orig = index_uniqueid

            ref_params = self.get_adjustable_parameters(exclude_constrained=False).filter(twig=twig, check_visible=False, **{k:v for k,v in kwargs.items() if k not in ['distribution']}).to_list()

        dist_params = []
        overwrite_ps = ParameterSet([])
        for ref_param in ref_params:
            if value is None:
                value_ = _distl.delta(ref_param.get_value())
            else:
                value_ = value

            metawargs = {'context': 'distribution',
                         'distribution': kwargs['distribution']}
            for k,v in ref_param.meta.items():
                if k in parameters._contexts:
                    metawargs.setdefault(k,v)

            if ref_param.__class__.__name__ == 'FloatArrayParameter' and index_orig is None:
                # then we need to iterate over the length
                indexes = range(len(ref_param.get_value()))
            else:
                indexes = [index_orig]

            for index in indexes:
                dist_param_qualifier = ref_param.qualifier if index is None else '{}[{}]'.format(ref_param.qualifier, index)
                # note: we'll always deepcopy here to avoid any linking between
                # user-provided distributions (unless provided as a multivariate)
                dist_param = DistributionParameter(bundle=self,
                                                   qualifier=dist_param_qualifier,
                                                   value=value_.deepcopy(),
                                                   description='distribution for the referenced parameter',
                                                   **metawargs)

                dist_param_existing_ps = self.filter(qualifier=dist_param_qualifier, check_visible=False, check_default=False, **metawargs)
                if len(dist_param_existing_ps):
                    if kwargs.get('overwrite_individual', False):
                        overwrite_ps += self.remove_parameters_all(uniqueid=dist_param_existing_ps.uniqueids)
                    else:
                        raise ValueError("parameter is already referenced by distribution = '{}'".format(kwargs['distribution']))

                self._attach_params([dist_param], **metawargs)
                dist_params += [dist_param]

        if not len(dist_params):
            return ParameterSet([])


        ret_ps = ParameterSet(dist_params)

        if kwargs.get('overwrite_individual', False) and return_changes:
            ret_ps += overwrite_ps

        return _return_ps(self, ret_ps)


    @send_if_client
    def add_distribution(self, arg1=None, value=None, return_changes=False, **kwargs):
        """
        Add one or multiple <phoebe.parameters.DistributionParameter>
        to a new or existing `distribution`.

        Note: the first positional argument, `arg1`, can either be a dictionary,
        a list of dictionaries, or a single twig (in which case `value` must
        also be provided).
        For example:

        ```
        b.add_distribution({'teff@primary': phoebe.uniform(5000,6000), 'incl@binary': phoebe.uniform(80,90)}, distribution='dist01')
        ```

        or

        ```
        b.add_distribution([{'qualifier': 'teff', 'component': 'primary', 'value': phoebe.uniform(5000,6000)},
                       {'qualifier': 'incl', 'component': 'binary', 'value': phoebe.uniform(80,90)}],
                       distribution='dist01')
        ```

        or

        ```
        b.add_distribution('teff@primary', phoebe.uniform(5000,6000), distribution='dist01')
        b.add_distribution('incl', phoebe.uniform(80,90), component='binary', distribution='dist01')
        ```

        Note also that the values (whether provided in the dictionary/dictionaries
        or to `value`) can either be [distl Distribution objects](https://distl.readthedocs.io/en/latest/api/) or a tuple with
        length 2, in which case the tuple is adopted as a [uniform distribution](https://distl.readthedocs.io/en/latest/api/Uniform/),
        or None in which case a [delta distribution](https://distl.readthedocs.io/en/latest/api/Delta/) is adopted around the current value.

        Distribution objects can easily be created from top-level convenience functions:
        * <phoebe.uniform>
        * <phoebe.uniform_around>
        * <phoebe.gaussian>
        * <phoebe.gaussian_around>
        * <phoebe.delta>
        * <phoebe.delta_around>

        Any "around" distribution will react so that the "central value" of the
        distribution is updated as the face-value of the referenced parameter
        updates.


        See also:
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.frontend.bundle.Bundle.rename_distribution>
        * <phoebe.frontend.bundle.Bundle.remove_distribution>
        * <phoebe.frontend.bundle.Bundle.get_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.sample_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.plot_distribution_collection>

        Arguments
        -----------
        * `arg1` (dictionary, list of dictionaries, or string, optional, default=None):
            See above for valid formats/examples.
        * `value` (distl object or tuple, optional, default=None): required
            if `arg1` is a twig/string.  Otherwise will be used as a default
            in the `arg1` dictionary/dictionaries when the `value` is not provided
            or None.  If `arg1` is a dictionary/dictionaries and the distribution/value
            is always provided, then `value` will silently be ignored.
        * `distribution` (string, optional): name of the new distribution set.  If not,
            provided or None, one will be created automatically.
        * `allow_multiple_matches` (bool, optional, default=False): whether to
            allow each entry to be attached to multiple parameters.  If True,
            the `value` (distribution) will be copied and applied to each parameter
            that matches the filter.  If False and a filter results in more than
            one parameter, a ValueError will be raised.
        * `overwrite_individual` (bool, optional, default=False): overwrite any
            existing distributions tagged with `distribution`, but leave other
            existing distributions in place.
        * `overwrite_all` (bool, optional, default=False): if `distribution`
            already exists, remove all existing distributions first.  See
            <phoebe.frontend.bundle.Bundle.remove_distribution>.
        * `return_changes` (bool, optional, default=False): whether to include
            all changed parameters in the returned ParameterSet, including any
            removed by `overwrite_individual` or `overwrite_all`.
        * `**kwargs`: tags to filter for a matching parameter (and to tag the
            new <phoebe.parameters.DistributionParameter>).  This (along with `twig`)
            must point to a single parameter.  Any filtering parameters sent at
            kwargs will be applied as defaults to any not applied in `arg1`.

        Returns
        ---------
        * ParameterSet of newly-added parameters (and changed parameters if
            `return_changes` is True).  To see all parameters tagged with `distribution`,
            see <phoebe.frontend.bundle.Bundle.get_distribution>.

        Raises
        ---------
        * ValueError: if any filter results in multiple valid parameters but
            `allow_multiple_matches` is not True.
        * ValueError: if any filter results in zero valid parameters.
        * TypeError: if `arg1` is not of a supported format.  See examples above.
        * ValueError: if `overwrite` is passed as a kwarg.  Use `overwrite_all`
            or `overwrite_individual` instead.
        * ValueError: if `overwrite_all` AND `overwrite_individual` are both True.
        """
        kwargs.setdefault('distribution',
                          self._default_label('dists',
                                              **{'context': 'distribution'}))

        if kwargs.pop('check_label', True):
            # for now we will do allow_overwrite... we'll check that logic later
            self._check_label(kwargs['distribution'], allow_overwrite=True)

        if isinstance(arg1, dict):
            dist_dicts = [{'twig': k, 'value': v} for k,v in arg1.items()]
        elif isinstance(arg1, list):
            dist_dicts = arg1
        elif isinstance(arg1, str):
            if value is None:
                raise ValueError("value must be provided if first argument is a twig")
            dist_dicts = [{'twig': arg1, 'value': value}]
        elif arg1 is None:
            dist_dicts = [{'value': value}]  # all kwargs will still be applied
        else:
            raise TypeError("first argument must be a dictionary, list of dictionaries, or a twig")

        if 'overwrite' in kwargs.keys():
            raise ValueError("add_distribution accepts overwrite_individual or overwrite_all as arguments, not overwrite")

        if kwargs.get('overwrite_all', False) and kwargs.get('overwrite_individual', False):
            raise ValueError("cannot use both overwrite_all and overwrite_individual")

        if kwargs.get('overwrite_all', False):
            overwrite_ps = self.remove_distribution(distribution=kwargs['distribution'], during_overwrite=True)
            # check the label again, just in case kwargs['distribution'] belongs to
            # something other than component
            self.exclude(distribution=kwargs['distribution'])._check_label(kwargs['distribution'], allow_overwrite=False)
        else:
            overwrite_ps = None


        adjustable_params_ps = self.get_adjustable_parameters(exclude_constrained=False)

        # then we need to check for any conflicts FIRST, before adding any distributions
        already_exists = []  # list of twigs
        no_matches = []
        multiple_matches = []
        for dist_dict in dist_dicts:
            for k,v in kwargs.items():
                if k in ['uniqueid'] + list(self.meta.keys()):
                    dist_dict.setdefault(k, v)

            ref_params = adjustable_params_ps.filter(check_visible=False, **{k:v for k,v in dist_dict.items() if k not in ['distribution', 'value']}).to_list()
            if len(ref_params) == 0:
                no_matches += [dist_dict]
            elif len(ref_params) > 1 and not kwargs.get('allow_multiple_matches', False):
                multiple_matches += [dist_dict]

            if kwargs['distribution'] in self.distributions and not kwargs.get('overwrite_individual', False):
                f = self.get_distribution(**{k:v for k,v in dist_dict.items() if k not in ['value']})
                if len(f.to_list()):
                    already_exists += f.twigs

        if len(no_matches):
            raise ValueError("{} result in zero matches to existing parameters - cannot add distributions".format(no_matches))

        if len(multiple_matches):
            # TODO: disable this check with an option?  It technically will work within _add_single_distribution
            raise ValueError("{} result in multiple matches.  To attach a copy of the distribution to each of the matching parameters, pass allow_multiple_matches=True".format(multiple_matches))

        if len(already_exists):
            raise ValueError("{} already exist{}.  Use a new distribution label, pass overwrite_individual=True to overwrite just these entries, or pass overwrite_all=True to first remove all parameters with distribution='{}'".format(already_exists, 's' if len(already_exists) == 1 else '', kwargs['distribution']))

        ret_ps = ParameterSet([])
        for dist_dict in dist_dicts:
            if not isinstance(dist_dict, dict):
                raise TypeError("each item in values must be a dictionary")
            dist_dict.setdefault('value', value)

            for k,v in kwargs.items():
                if k in ['uniqueid'] + list(self.meta.keys()):
                    # NOTE: this will also pass distribution
                    dist_dict.setdefault(k, v)

            ret_ps += self._add_single_distribution(overwrite_individual=kwargs.get('overwrite_individual', False), return_changes=return_changes, **dist_dict)

        if kwargs.get('overwrite_all', False) and return_changes:
            ret_ps += overwrite_ps

        # if self.get_value(qualifier='auto_add_figure', context='setting', auto_add_figure=kwargs.get('auto_add_figure', None), **_skip_filter_checks) and 'distribution_collection' not in self.filter(context='figure', **_skip_filter_checks).kinds:
        #     # then we don't have a figure for this kind yet
        #     logger.info("calling add_figure(kind='distribution.distribution_collection') since auto_add_figure@setting=True")
        #     new_fig_params = self.add_figure(kind='distribution.distribution_collection', distributions=[kwargs['distribution']])
        #     ret_ps += new_fig_params
        # else:
        #     new_fig_params = None

        ret_changes = []
        ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_computesamplefrom_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_distribution(self, distribution=None, **kwargs):
        """
        Filter in the 'distribution' context.

        Note that this returns a ParameterSet of <phoebe.parameters.DistributionParameter>
        objects.  The distribution objects themselves can then be accessed
        via <phoebe.parameters.DistributionParameter.get_value>.  To access
        distribution objects of constrained parameters propaged through constraints,
        use <phoebe.parameters.FloatParameter.get_distribution> instead.

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.frontend.bundle.Bundle.add_distribution>
        * <phoebe.frontend.bundle.Bundle.rename_distribution>
        * <phoebe.frontend.bundle.Bundle.remove_distribution>
        * <phoebe.frontend.bundle.Bundle.get_distribution_collection>

        Arguments
        ----------
        * `distribution` (string, optional, default=None): the name of the distribution
        * `**kwargs`: any other tags to do the filtering (excluding distribution and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if distribution is not None:
            kwargs['distribution'] = distribution
            if distribution not in self.distributions:
                raise ValueError("distribution='{}' not found".format(distribution))

        kwargs['context'] = 'distribution'
        return self.filter(**kwargs)

    @send_if_client
    def rename_distribution(self, old_distribution, new_distribution,
                       overwrite=False, return_changes=False):
        """
        Change the label of a distribution attached to the Bundle.

        See also:
        * <phoebe.frontend.bundle.Bundle.add_distribution>
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.frontend.bundle.Bundle.remove_distribution>

        Arguments
        ----------
        * `old_distribution` (string): current label of the distribution (must exist)
        * `new_distribution` (string): the desired new label of the distribution
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed distribution

        Raises
        --------
        * ValueError: if the value of `new_distribution` is forbidden or already exists.
        """
        # TODO: raise error if old_distribution not found?
        self._rename_label('distribution', old_distribution, new_distribution, overwrite)

        ret_ps = self.filter(distribution=new_distribution)

        ret_changes = []
        ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_computesamplefrom_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    @send_if_client
    def remove_distribution(self, distribution, return_changes=False, **kwargs):
        """
        Remove a distribution from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        * <phoebe.frontend.bundle.Bundle.add_distribution>
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.frontend.bundle.Bundle.rename_distribution>

        Arguments
        ----------
        * `distribution` (string): the label of the distribution to be removed.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: distribution, context

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        kwargs['distribution'] = distribution
        kwargs['context'] = 'distribution'
        ret_ps = self.remove_parameters_all(**kwargs)

        ret_changes = []
        if not kwargs.get('during_overwrite', False):
            ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
            ret_changes += self._handle_computesamplefrom_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def _distribution_collection_defaults(self, twig=None, **kwargs):
        filter_kwargs = {k:v for k,v in kwargs.items() if k in parameters._meta_fields_filter}
        kwargs = {k:v for k,v in kwargs.items() if k not in parameters._meta_fields_filter}
        # print("*** _distribution_collection_defaults twig={} kwargs={}, filter_kwargs={}".format(twig, kwargs, filter_kwargs))

        if twig is None and not len(filter_kwargs.keys()):
            filter_kwargs['context'] = 'distribution'

        if not isinstance(twig, list):
            ps = self.filter(context=['distribution', 'solution', 'solver', 'compute'], **_skip_filter_checks).filter(twig=twig, **filter_kwargs)

            if ps.context == 'compute':
                if ps.qualifier not in ['sample_from']:
                    raise ValueError("twig and kwargs must point to a single parameter in the compute options (e.g. qualifier='sample_from')")

                kwargs.setdefault('include_constrained', False)
                kwargs.setdefault('to_univariates', False)
                kwargs.setdefault('combine', self.get_value(qualifier='{}_combine'.format(ps.qualifier), check_visible=False, check_default=False, **{k:v for k,v in ps.meta.items() if k not in ['qualifier']}))
                return self._distribution_collection_defaults(ps.get_value(expand=True, **{ps.qualifier: kwargs.get(ps.qualifier, None)}), **kwargs)

            elif ps.context == 'solver':
                # then we must point to a SINGLE parameter
                if ps.qualifier not in ['init_from', 'priors']:
                    raise ValueError("twig and kwargs must point to a single parameter in the solver options (e.g. qualifier='priors')")
                kind = ps.kind

                if kind in ['emcee']:

                    if ps.qualifier in ['priors']:
                        kwargs.setdefault('include_constrained', True)
                        kwargs.setdefault('to_univariates', False)
                    elif ps.qualifier in ['init_from']:
                        kwargs.setdefault('include_constrained', False)
                        kwargs.setdefault('to_univariates', False)
                    else:
                        raise NotImplementedError("get_distribution_collection for solver kind='{}' and qualifier='{}' not implemented".format(kind, ps.qualifier))

                elif kind in ['dynesty']:
                    if ps.qualifier in ['priors']:
                        # TODO: need to support flattening to univariates
                        kwargs.setdefault('include_constrained', False)
                        kwargs.setdefault('to_univariates', True)
                    else:
                        raise NotImplementedError("get_distribution_collection for solver kind='{}' and qualifier='{}' not implemented".format(kind, ps.qualifier))

                elif kind in ['differential_evolution']:
                    if ps.qualifier in ['bounds']:
                        kwargs.setdefault('include_constrained', True)
                        kwargs.setdefault('to_univariates', True)
                        kwargs.setdefault('to_uniforms', self.get_value('{}_sigma'.format(ps.qualifier), check_visible=False, check_default=False, **{k:v for k,v in ps.meta.items() if k not in ['qualifier']}))

                else:
                    raise NotImplementedError("get_distribution_collection for solver kind='{}' not implemented".format(kind))

                kwargs.setdefault('combine', self.get_value(qualifier='{}_combine'.format(ps.qualifier), check_visible=False, check_default=False, **{k:v for k,v in ps.meta.items() if k not in ['qualifier']}))
                return self._distribution_collection_defaults(ps.get_value(expand=True, **{ps.qualifier: kwargs.get(ps.qualifier, None)}), **kwargs)

            twig = [twig]

        filters = []

        for twigi in twig:
            ps = self.filter(context=['distribution', 'solution', 'solver', 'compute'], **_skip_filter_checks).filter(twig=twigi, **filter_kwargs)
            for context in ps.contexts:
                if context == 'distribution':
                    if filter_kwargs.get('distribution', None) is not None and len(filter_kwargs.get('distribution')) == len(ps.distributions):
                        # then respect the passed order
                        filters += [{'distribution': d} for d in filter_kwargs.get('distribution')]
                    else:
                        filters += [{'distribution': d} for d in ps.distributions]

                elif context=='solution':
                    if filter_kwargs.get('solution', None) is not None and len(filter_kwargs.get('solution')) == len(ps.solutions):
                        # then respect the passed order
                        filters += [{'solution': s} for s in filter_kwargs.get('solution')]
                    else:
                        filters += [{'solution': s} for s in ps.solutions]
                    kwargs.setdefault('include_constrained', True)

                else:
                    raise ValueError("twig and kwargs must point to the solution or distribution context, got context='{}'".format(context))

        combine = kwargs.get('combine', 'first')
        include_constrained = kwargs.get('include_constrained', False)
        to_univariates = kwargs.get('to_univariates', False)
        to_uniforms = kwargs.get('to_uniforms', False)

        if to_uniforms and not to_univariates:
            raise ValueError("to_univariates must be True in order to use to_uniforms")

        return filters, combine, include_constrained, to_univariates, to_uniforms

    def get_distribution_collection(self, twig=None,
                                    keys='twig', set_labels=True,
                                    parameters=None,
                                    **kwargs):
        """
        Combine multiple distribution objects into a single
        [distl.DistributionCollection](https://distl.readthedocs.io/en/latest/api/DistributionCollection/)

        See also:
        * <phoebe.frontend.bundle.Bundle.sample_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.uncertainties_from_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.calculate_lnp>
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.parameters.FloatParameter.get_distribution>

        Arguments
        -------------
        * `twig`: (string, optional, default=None): twig to use for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet (eg. init_from or priors),
            a ParameterSet of distribution parameters, or a solution ParameterSet
            that supports multivariate distributions (eg. sampler.emcee or sampler.dynesty).
        * `combine`: (str, optional) how to combine multiple distributions for the same parameter.
            first: ignore duplicate entries and take the first entry.
            and: combine duplicate entries via AND logic, dropping covariances.
            or: combine duplicate entries via OR logic, dropping covariances.'
            Defaults to 'first' if `twig` and `**kwargs` point to distributions,
            otherwise will default to the value of the relevant parameter in the
            solver options.
        * `include_constrained` (bool, optional): whether to
            include constrained parameters.  Defaults to False if `twig` and
            `**kwargs` point to distributions, otherwise will default to the
            value necessary for the solver backend.
        * `to_univariates` (bool, optional): whether to convert any multivariate
            distributions to univariates before adding to the collection.  Defaults
            to False if `twig` and `**kwargs` point to distributions, otherwise
            will default to the value necessary for the solver backend.
        * `to_uniforms` (bool or int, optional): whether to convert all distributions
            to uniforms (and if an int, then what sigma to adopt for non-uniform
            distributions).  Defaults to False if `twig` and `**kwargs` point to
            distributions, otherwise will default to the value necessary for the
            solver backend.
        * `keys` (string, optional, default='twig'): attribute to use for the
            second returned object ('twig', 'qualifier', 'uniqueid').  NOTE: the
            attributes will be called on the referenced parameter, not the distribution parameter.
            See <phoebe.parameters.DistributionParameter.get_referenced_parameter>
            and <phoebe.parameters.FloatParameter.get_distribution>.
        * `set_labels` (bool, optional, default=True): set the labels of the
            distribution objects to be the twigs of the referenced parameters.
        * `parameters` (list, dict, or string, optional, default=None): if provided,
            then `parameters` will be passed as a filter to the available adjustable
            parameters (<phoebe.frontend.bundle.Bundle.get_adjustable_parameters>
            with `exclude_constrained=False`), and these parameters will be exposed
            in the resulting DistributionCollection (excluding any entries not
            matching the filter, and propagating any additional entries through
            constraints).  An error may be raised if any matching parameters
            are not included in the original DistributionCollection or available
            through propagated constraints.
        * `**kwargs`: additional keyword arguments are used for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet, or a ParameterSet of distribution
            parameters.  If pointing to a solution ParameterSet, `**kwargs` can
            also be used to override `distributions_convert` and `distributions_bins`,
            as well as any other solution parameters (eg. `burnin`, `thin`, etc).

        Returns
        ------------
        * distl.DistributionCollection, list of `keys`
        """
        def _to_dist(dist, to_univariates=False, to_uniform=False):
            if isinstance(dist, _distl.BaseAroundGenerator):
                # freeze to the current value
                dist = dist()
            if to_univariates:
                if hasattr(dist, 'to_univariate'):
                    if not raised['univariate']:
                        logger.warning("covariances for {} will be dropped and all distributions converted to univariates".format(dist.label))
                        raised['univariate'] = True
                    dist = dist.to_univariate()
            if to_uniform:
                if hasattr(dist, 'to_uniform'):
                    if not raised['uniform']:
                        logger.warning("all non-uniform distributions in {} will be converted to uniforms by adopting sigma={}".format(dist.label, int(to_uniforms)))
                        raised['uniform'] = True
                    dist = dist.to_uniform(sigma=int(to_uniform))

            return dist

        def _get_key(ref_param, keys, index=None):
            k = getattr(ref_param, keys)
            if index is not None:
                if keys=='twig':
                    k = '{}[{}]@{}'.format(k.split('@')[0], index, '@'.join(k.split('@')[1:]))
                else:
                    k += '[{}]'.format(index)
            return k

        if parameters is not None:
            parameters_indices = None
            if isinstance(parameters, ParameterSet):
                pass
            elif isinstance(parameters, dict):
                parameters = self.get_adjustable_parameters(exclude_constrained=False).filter(**parameters)
            else:
                parameters, parameters_indices = _extract_index_from_string(parameters)
                parameters = self.get_adjustable_parameters(exclude_constrained=False).filter(parameters)

            parameters_uniqueids = parameters.uniqueids #parameters_uniqueids_indices[:,0]
            parameters_uniqueids_with_indices = ["{}[{}]".format(uniqueid, index) for uniqueid,index in zip(parameters_uniqueids, parameters_indices)]

            # now we'll get all AVAILABLE distributions that could match...
            # any remaining items will need to be propagated or return a delta distribution
            available_dc, available_uniqueids = self.get_distribution_collection(twig=twig, keys='uniqueid', set_labels=set_labels, parameters=None, allow_non_dc=False, **{k:v for k,v in kwargs.items() if k not in ['allow_non_dc', 'set_labels']})
            available_uniqueids, available_indices = _extract_index_from_string(available_uniqueids)
            available_uniqueids_with_indices = ["{}[{}]".format(uniqueid, index) for uniqueid,index in zip(available_uniqueids, available_indices)]

            # first filter through the distributions already in dc
            ret_dists = [available_dc.dists[i] for i,uniqueid_with_index in enumerate(available_uniqueids_with_indices) if uniqueid_with_index in parameters_uniqueids_with_indices]
            ret_keys = [_get_key(self.get_parameter(uniqueid=uniqueid_with_index.split('[')[0], **_skip_filter_checks), keys, index) for uniqueid_with_index, index in zip(available_uniqueids_with_indices, available_indices) if uniqueid_with_index in parameters_uniqueids_with_indices]

            # now we need to get any that weren't included in dc
            new_params = [self.get_parameter(uniqueid=uniqueid_with_index.split('[')[0], **_skip_filter_checks) for uniqueid_with_index in parameters_uniqueids_with_indices if uniqueid_with_index not in available_uniqueids_with_indices]
            new_indices = [index for index, uniqueid_with_index in zip(parameters_indices, parameters_uniqueids_with_indices) if uniqueid_with_index not in available_uniqueids_with_indices]
            ret_dists += [param.get_distribution(distribution=available_dc, distribution_uniqueids=available_uniqueids, delta_if_none=True) for param in new_params]
            ret_keys += [_get_key(param, keys, index) for param,index in zip(new_params,new_indices)]
            # TODO: do we need to set labels on the newly added dists?

            if kwargs.get('return_dc', True):
                dc = _distl.DistributionCollection(*ret_dists)
            else:
                dc = None

            return dc, ret_keys



        if 'distribution_filters' not in kwargs.keys():
            distribution_filters, combine, include_constrained, to_univariates, to_uniforms = self._distribution_collection_defaults(twig=twig, **kwargs)
        else:
            # INTERNAL USE ONLY, probably
            distribution_filters = kwargs.get('distribution_filters')
            combine = kwargs.get('combine', 'first')
            include_constrained = kwargs.get('include_constrained', True)
            to_univariates = kwargs.get('to_univariates', False)
            to_uniforms = kwargs.get('to_uniforms', False)

        # NOTE: in python3 we could do this with booleans and nonlocal variables,
        # but for python2 support we can only fake it by mutating a dictionary.
        # https://stackoverflow.com/questions/3190706/nonlocal-keyword-in-python-2-x
        raised = {'univariate': False, 'uniform': False}

        ret_dists = []
        ret_keys = []
        uid_dist_dict = {}
        uniqueids = []
        for dist_filter in distribution_filters:
            # TODO: if * in list, need to expand (currently forbidden with error in get_distribution)
            if 'solution' in dist_filter.keys():
                # print("*** get_distribution_collection solution dist_filter={}".format(dist_filter))
                solution_ps = self.get_solution(solution=dist_filter['solution'], **_skip_filter_checks)
                solver_kind = solution_ps.kind

                adopt_inds, adopt_uniqueids = self._get_adopt_inds_uniqueids(solution_ps, **kwargs)

                if not len(adopt_inds):
                    raise ValueError('no parameters selected by adopt_parameters')

                fitted_units = solution_ps.get_value(qualifier='fitted_units', **_skip_filter_checks)

                if solver_kind == 'emcee':
                    lnprobabilities = solution_ps.get_value(qualifier='lnprobabilities', **_skip_filter_checks)
                    samples = solution_ps.get_value(qualifier='samples', **_skip_filter_checks)

                    burnin = solution_ps.get_value(qualifier='burnin', burnin=kwargs.get('burnin', None), **_skip_filter_checks)
                    thin = solution_ps.get_value(qualifier='thin', thin=kwargs.get('thin', None), **_skip_filter_checks)
                    lnprob_cutoff = solution_ps.get_value(qualifier='lnprob_cutoff', lnprob_cutoff=kwargs.get('lnprob_cutoff', None), **_skip_filter_checks)

                    lnprobabilities, samples = _helpers.process_mcmc_chains(lnprobabilities, samples, burnin, thin, lnprob_cutoff, adopt_inds)
                    weights = None

                elif solver_kind == 'dynesty':
                    samples = solution_ps.get_value(qualifier='samples', **_skip_filter_checks)
                    logwt = solution_ps.get_value(qualifier='logwt', **_skip_filter_checks)
                    logz = solution_ps.get_value(qualifier='logz', **_skip_filter_checks)

                    samples = samples[:, adopt_inds]

                    weights = np.exp(logwt - logz[-1])

                else:
                    # then this is an estimator or optimizer, so we just want Delta
                    # distributions around 'fitted_values'

                    fitted_values = solution_ps.get_value(qualifier='fitted_values', **_skip_filter_checks)

                    for fitted_value, fitted_unit, fitted_uniqueid in zip(fitted_values[adopt_inds], fitted_units[adopt_inds], adopt_uniqueids):
                        param = self.get_parameter(uniqueid=fitted_uniqueid, **_skip_filter_checks)
                        _, index = _extract_index_from_string(fitted_uniqueid)
                        ret_keys += [_get_key(param, keys, index)]
                        if kwargs.get('return_dc', True):
                            ret_dists += [_distl.delta(fitted_value, unit=fitted_unit, label=_corner_twig(param, use_tex=False), label_latex=_corner_twig(param, use_tex=True))]

                    # skip all converting?
                    continue


                distributions_convert = solution_ps.get_value(qualifier='distributions_convert', distributions_convert=kwargs.get('distributions_convert', None), **_skip_filter_checks)
                distributions_bins = solution_ps.get_value(qualifier='distributions_bins', distributions_bins=kwargs.get('distributions_bins', None), **_skip_filter_checks)

                adopt_uniqueids_with_indexes = [_extract_index_from_string(uid) for uid in adopt_uniqueids]
                labels = [_corner_twig(self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks), use_tex=False, index=index) for uniqueid, index in adopt_uniqueids_with_indexes]
                labels_latex = [_corner_twig(self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks), use_tex=True, index=index) for uniqueid, index in adopt_uniqueids_with_indexes]
                dist_samples = _distl.mvsamples(samples,
                                                weights=weights,
                                                units=[u.Unit(unit) for unit in fitted_units[adopt_inds]],
                                                labels=labels,
                                                labels_latex=labels_latex,
                                                wrap_ats=None)

                if distributions_convert == 'mvsamples':
                    dist = dist_samples
                elif distributions_convert == 'mvhistogram':
                    dist = dist_samples.to_mvhistogram(bins=distributions_bins)
                elif distributions_convert == 'mvgaussian':
                    dist = dist_samples.to_mvgaussian(allow_singular=True)
                elif distributions_convert == 'samples':
                    dist = _distl.DistributionCollection(*[dist_samples.to_univariate(dim) for dim in dist_samples.labels])
                elif distributions_convert == 'histogram':
                    dist = _distl.DistributionCollection(*[dist_samples.to_univariate(dim).to_histogram(bins=distributions_bins) for dim in dist_samples.labels])
                elif distributions_convert == 'gaussian':
                    dist = _distl.DistributionCollection(*[dist_samples.to_univariate(dim).to_gaussian() for dim in dist_samples.labels])
                else:
                    raise NotImplementedError("distributions_convert='{}' not supported".format(distributions_convert))

                ret_keys += [_get_key(self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks), keys, index) for uniqueid, index in zip(*_extract_index_from_string(adopt_uniqueids))]

                if len(distribution_filters) == 1 and kwargs.get('allow_non_dc', True):
                    # then try to avoid slicing since we don't have to combine with anything else
                    return dist, ret_keys

                if kwargs.get('return_dc', True):
                    ret_dists += [dist.slice(label) for label in labels]


            elif 'distribution' in dist_filter.keys():
                # print("*** get_distribution_collection distribution dist_filter={}".format(dist_filter))
                dist_ps = self.get_distribution(distribution=dist_filter['distribution'], **_skip_filter_checks)
                for dist_param in dist_ps.to_list():
                    qualifier, index = _extract_index_from_string(dist_param.qualifier)
                    ref_param = dist_param.get_referenced_parameter()
                    uid = ref_param.uniqueid if index is None else '{}[{}]'.format(ref_param.uniqueid, index)
                    if not include_constrained and len(ref_param.constrained_by):
                        continue
                    if uid not in uniqueids:
                        k = _get_key(ref_param, keys, index)
                        if k in uid_dist_dict.keys():
                            raise ValueError("keys='{}' does not result in unique entries for each item".format(keys))

                        uid_dist_dict[uid] = _to_dist(dist_param.get_value(), to_univariates, to_uniforms)
                        uniqueids.append(uid)
                        ret_keys.append(k)
                    elif combine.lower() == 'first':
                        logger.warning("ignoring distribution on {} with distribution='{}' as distribution existed on an earlier distribution which takes precedence.".format(ref_param.twig, dist_filter['distribution']))
                        continue
                    elif combine.lower() == 'and':
                        dist_obj = dist_param.get_value()
                        old_dists = uid_dist_dict[uid].dists if isinstance(uid_dist_dict[uid], _distl._distl.Composite) else [_to_dist(uid_dist_dict[uid], True, to_uniforms)]
                        new_dist = _to_dist(dist_obj, True, to_uniforms)
                        combined_dist = _distl._distl.Composite('__and__', old_dists + [new_dist])
                        uid_dist_dict[uid] = combined_dist
                    elif combine.lower() == 'or':
                        dist_obj = dist_param.get_value()
                        old_dists = uid_dist_dict[uid].dists if isinstance(uid_dist_dict[uid], _distl._distl.Composite) else [_to_dist(uid_dist_dict[uid], True, to_uniforms)]
                        new_dist = _to_dist(dist_obj, True, to_uniforms)
                        combined_dist = _distl._distl.Composite('__or__', old_dists + [new_dist])
                        uid_dist_dict[uid] = combined_dist
                    else:
                        raise NotImplementedError("combine='{}' not supported".format(combine))

                    if set_labels:
                        uid_dist_dict[uid].label = "@".join([getattr(ref_param, k) for k in ['qualifier', 'component', 'dataset'] if getattr(ref_param, k) is not None])

                        if index is not None:
                            uid_dist_dict[uid].label = '{}[{}]@{}'.format(uid_dist_dict[uid].label.split('@')[0], index, '@'.join(uid_dist_dict[uid].label.split('@')[1:]))
                            uid_dist_dict[uid].label_latex = ref_param.latextwig.replace("$", "")+"[{}]".format(index) if ref_param._latexfmt is not None else None
                        else:
                            uid_dist_dict[uid].label_latex =  ref_param.latextwig.replace("$", "") if ref_param._latexfmt is not None else None



            else:
                raise NotImplementedError("could not parse filter for distribution {}".format(dist_filter))

        if kwargs.get('return_dc', True):
            ret_dists += [uid_dist_dict.get(uid) for uid in uniqueids]

            dc = _distl.DistributionCollection(*ret_dists)
        else:
            dc = None

        return dc, ret_keys

    def sample_distribution_collection(self, twig=None, sample_size=None,
                                       as_quantity=False,
                                       set_value=False, keys='twig',
                                       parameters=None,
                                        **kwargs):
        """
        Sample from a [distl.DistributionCollection](https://distl.readthedocs.io/en/latest/api/DistributionCollection/).
        Note that distributions attached to constrained parameters will be
        ignored (but constrained values will be updated if `set_value` is True).

        All values will be in the current <phoebe.parameters.FloatParameter.default_unit>
        of the respective parameters, not the units of the underlying <phoebe.parameters.DistributionParameter>.
        Pass `as_quantity=True` to access the quantity objects in original units.

        See also:
        * <phoebe.frontend.bundle.Bundle.get_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.plot_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.uncertainties_from_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.calculate_lnp>
        * <phoebe.frontend.bundle.Bundle.get_distribution>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): twig to use for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet, or a ParameterSet of distribution
            parameters.
        * `sample_size` (int, optional, default=None): number of samples to draw from
            each distribution.  Note that this must be None if `set_value` is
            set to True. **NOTE**: prior to 2.3.25, this argument was name `N`.
        * `combine`: (str, optional) how to combine multiple distributions for the same parameter.
            first: ignore duplicate entries and take the first entry.
            and: combine duplicate entries via AND logic, dropping covariances.
            or: combine duplicate entries via OR logic, dropping covariances.'
            Defaults to 'first' if `twig` and `**kwargs` point to distributions,
            otherwise will default to the value of the relevant parameter in the
            solver options.
        * `include_constrained` (bool, optional): whether to
            include constrained parameters.  Defaults to False if `twig` and
            `**kwargs` point to distributions, otherwise will default to the
            value necessary for the solver backend.
        * `to_univariates` (bool, optional): whether to convert any multivariate
            distributions to univariates before adding to the collection.  Defaults
            to False if `twig` and `**kwargs` point to distributions, otherwise
            will default to the value necessary for the solver backend.
        * `to_uniforms` (bool or int, optional): whether to convert all distributions
            to uniforms (and if an int, then what sigma to adopt for non-uniform
            distributions).  Defaults to False if `twig` and `**kwargs` point to
            distributions, otherwise will default to the value necessary for the
            solver backend.
        *  `as_quantity` (bool, optional, default=False): expose values as quantities
            instead of floats.  If True, quanitities will be exposed in the units
            of the underlying distribution.  If False, floats will be converted
            to the current units of the referenced parameter.
        * `set_value` (bool, optional, default=False): whether to adopt the
            sampled values for all relevant parameters.  Note that `N` must
            be None and `include_constrained` must be False.
        * `keys` (string, optional, default='twig'): attribute to use for dictionary
            keys ('twig', 'qualifier', 'uniqueid').  Only applicable if
            `set_value` is False.
        * `parameters` (list, dict, or string, optional, default=None): if provided,
            then `parameters` will be passed as a filter to the available adjustable
            parameters (<phoebe.frontend.bundle.Bundle.get_adjustable_parameters>
            with `exclude_constrained=False`), and these parameters will be exposed
            in the resulting DistributionCollection (excluding any entries not
            matching the filter, and propagating any additional entries through
            constraints).  An error may be raised if any matching parameters
            are not included in the original DistributionCollection or available
            through propagated constraints.
        * `**kwargs`: additional keyword arguments are used for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet, or a ParameterSet of distribution
            parameters.

        Returns
        --------
        * (dict or ParameterSet): dictionary of `keys`, value pairs if `set_value`
            is False.  ParameterSet of changed Parameters (including those by
            constraints) if `set_value` is True.

        Raises
        -------
        * ValueError: if `set_value` is True and `N` is not None (as parameters
            cannot be set to multiple values)
        * ValueError: if `set_value` is True and `include_constrained` is True
            (as parameters that are constrained cannot adopt the sampled values)
        """
        # backwards compatibility before change from N to sample_size
        N = kwargs.pop('N', None)
        if N is not None:
            if sample_size is not None:
                raise ValueError("cannot pass both N and sample_size (sample_size replaces N)")
            sample_size = N

        if sample_size is not None and set_value:
            raise ValueError("cannot use set_value and sample_size together")

        if 'distribution_filters' not in kwargs.keys():
            distribution_filters, combine, include_constrained, to_univariates, to_uniforms = self._distribution_collection_defaults(twig=twig, **kwargs)
        else:
            # INTERNAL USE ONLY, probably
            distribution_filters = kwargs.get('distribution_filters')
            combine = kwargs.get('combine', 'first')
            include_constrained = kwargs.get('include_constrained', True)
            to_univariates = kwargs.get('to_univariates', False)
            to_uniforms = kwargs.get('to_uniforms', False)

        if include_constrained and set_value:
            raise ValueError("cannot use include_constrained=True and set_value together")

        if set_value:
            user_interactive_constraints = conf.interactive_constraints
            conf.interactive_constraints_off(suppress_warning=True)

        dc, uniqueids = self.get_distribution_collection(distribution_filters=distribution_filters,
                                                         combine=combine,
                                                         include_constrained=include_constrained,
                                                         to_univariates=to_univariates,
                                                         to_uniforms=to_uniforms,
                                                         keys='uniqueid',
                                                         parameters=parameters,
                                                         allow_non_dc=False)

        if isinstance(dc, _distl._distl.DistributionCollection) and np.all([isinstance(dist, _distl._distl.Delta) for dist in dc.dists]):
            if sample_size is not None and sample_size > 1:
                logger.warning("all distributions are delta, using sample_size=1 instead of sample_size={}".format(sample_size))
                N = 1

        sampled_values = dc.sample(size=sample_size).T

        ret = {}
        changed_params = []
        for sampled_value, uniqueid, unit in zip(sampled_values, uniqueids, [dist.unit for dist in dc.dists]):
            uniqueid, index = _extract_index_from_string(uniqueid)
            ref_param = self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)

            if set_value:
                if index is None:
                    ref_param.set_value(sampled_value, unit=unit)
                else:
                    ref_param.set_index_value(index, sampled_value, unit=unit)

                changed_params.append(ref_param)
            else:
                k = getattr(ref_param, keys)
                if index is not None:
                    k += '[{}]'.format(index)
                if as_quantity:
                    # this is the actual problem
                    ret[k] = sampled_value * unit
                else:
                    ret[k] = (sampled_value * unit).to(ref_param.default_unit).value


        if set_value:
            changed_params += self.run_delayed_constraints()
            if user_interactive_constraints:
                conf.interactive_constraints_on()
            return _return_ps(self, ParameterSet(changed_params))
        else:
            # ret is a dictionary
            return ret

    def plot_distribution_collection(self, twig=None,
                                    set_labels=True,
                                    parameters=None,
                                    show=False,
                                    **kwargs):

        """
        Calls plot on the first returned argument from <phoebe.frontend.bundle.Bundle.get_distribution_collection>

        See also:
        * <phoebe.frontend.bundle.Bundle.get_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.sample_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.uncertainties_from_distribution_collection>
        * <phoebe.parameters.ParameterSet.plot>

        Arguments
        -----------
        * `twig`: (string, optional, default=None): twig to use for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet (eg. init_from or priors),
            a ParameterSet of distribution parameters, or a solution ParameterSet
            that supports multivariate distributions (eg. sampler.emcee or sampler.dynesty).
        * `combine`: (str, optional) how to combine multiple distributions for the same parameter.
            first: ignore duplicate entries and take the first entry.
            and: combine duplicate entries via AND logic, dropping covariances.
            or: combine duplicate entries via OR logic, dropping covariances.'
            Defaults to 'first' if `twig` and `**kwargs` point to distributions,
            otherwise will default to the value of the relevant parameter in the
            solver options.
        * `include_constrained` (bool, optional): whether to
            include constrained parameters.  Defaults to False if `twig` and
            `**kwargs` point to distributions, otherwise will default to the
            value necessary for the solver backend.
        * `to_univariates` (bool, optional): whether to convert any multivariate
            distributions to univariates before adding to the collection.  Defaults
            to False if `twig` and `**kwargs` point to distributions, otherwise
            will default to the value necessary for the solver backend.
        * `to_uniforms` (bool or int, optional): whether to convert all distributions
            to uniforms (and if an int, then what sigma to adopt for non-uniform
            distributions).  Defaults to False if `twig` and `**kwargs` point to
            distributions, otherwise will default to the value necessary for the
            solver backend.
        * `set_labels` (bool, optional, default=True): set the labels of the
            distribution objects to be the twigs of the referenced parameters.
        * `parameters` (list, dict, or string, optional, default=None): if provided,
            then `parameters` will be passed as a filter to the available adjustable
            parameters (<phoebe.frontend.bundle.Bundle.get_adjustable_parameters>
            with `exclude_constrained=False`), and these parameters will be exposed
            in the resulting DistributionCollection (excluding any entries not
            matching the filter, and propagating any additional entries through
            constraints).  An error may be raised if any matching parameters
            are not included in the original DistributionCollection or available
            through propagated constraints.
        * `plot_uncertainties` (bool or list, optional, default=True): whether
            to plot uncertainties (as contours on 2D plots, vertical lines
            on histograms, and in the axes titles).  If True, defaults to `[1,2,3]`.
            The first value in the list is used for the histogram and title,
            with the full list being passed to the 2D contours.  So to plot
            1-, 2-, and 3-sigma uncertainties in the contours but quote 3-sigma
            uncertainties in the title and histograms, pass `[3,1,2]`.
        * `sample_size` (int, optional, default=None): number of samples to draw for
            the underlying distribution.  Defaults to 1e5 for most cases, or 1e3
            for expensive function calls.  If propagating through non-analytic
            constraints, setting a lower `sample_size` will significantly speed up
            plotting time.  Passed to distl as `size` argument
        * `show` (boolean, optional, default=False): whether to call show on the
            resulting figure object
        * `**kwargs`: all additional keyword arguments are passed directly to
            <phoebe.frontend.bundle.Bundle.get_distribution_collection>.

        Returns
        ----------
        * matplotlib figure object
        """
        plot_kwargs = {}
        for k in list(kwargs.keys()):
            if k in ['plot_uncertainties']:
                plot_kwargs[k] = kwargs.pop(k)
            elif k == 'sample_size':
                plot_kwargs['size'] = kwargs.pop('sample_size')
        dc, _ = self.get_distribution_collection(twig=twig, set_labels=set_labels, keys='uniqueid', parameters=parameters, **kwargs)
        return dc.plot(show=show, **plot_kwargs)

    def uncertainties_from_distribution_collection(self, twig=None,
                                                   parameters=None,
                                                   sigma=1, tex=False,
                                                    **kwargs):

        """
        Get (asymmetric) uncertainties for all parameters in a distribution collection
        by first sampling the underlying distribution object(s) 1 million times.

        See [distl.DistributionCollection.uncertainties](https://distl.readthedocs.io/en/latest/api/DistributionCollection.uncertainties/)
        for more details.

        See also:
        * <phoebe.frontend.bundle.Bundle.get_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.sample_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.plot_distribution_collection>

        Arguments
        -----------
        * `twig`: (string, optional, default=None): twig to use for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet (eg. init_from or priors),
            a ParameterSet of distribution parameters, or a solution ParameterSet
            that supports multivariate distributions (eg. sampler.emcee or sampler.dynesty).
        * `sigma` (int, optional, default=1): which sigma level to expose.
        * `tex` (bool, optional, default=False): whether to expose a latex
            formatted string instead of triplets.
        * `combine`: (str, optional) how to combine multiple distributions for the same parameter.
            first: ignore duplicate entries and take the first entry.
            and: combine duplicate entries via AND logic, dropping covariances.
            or: combine duplicate entries via OR logic, dropping covariances.'
            Defaults to 'first' if `twig` and `**kwargs` point to distributions,
            otherwise will default to the value of the relevant parameter in the
            solver options.
        * `include_constrained` (bool, optional): whether to
            include constrained parameters.  Defaults to False if `twig` and
            `**kwargs` point to distributions, otherwise will default to the
            value necessary for the solver backend.
        * `to_univariates` (bool, optional): whether to convert any multivariate
            distributions to univariates before adding to the collection.  Defaults
            to False if `twig` and `**kwargs` point to distributions, otherwise
            will default to the value necessary for the solver backend.
        * `to_uniforms` (bool or int, optional): whether to convert all distributions
            to uniforms (and if an int, then what sigma to adopt for non-uniform
            distributions).  Defaults to False if `twig` and `**kwargs` point to
            distributions, otherwise will default to the value necessary for the
            solver backend.
        * `parameters` (list, dict, or string, optional, default=None): if provided,
            then `parameters` will be passed as a filter to the available adjustable
            parameters (<phoebe.frontend.bundle.Bundle.get_adjustable_parameters>
            with `exclude_constrained=False`), and these parameters will be exposed
            in the resulting DistributionCollection (excluding any entries not
            matching the filter, and propagating any additional entries through
            constraints).  An error may be raised if any matching parameters
            are not included in the original DistributionCollection or available
            through propagated constraints.
        * `**kwargs`: all additional keyword arguments are passed directly to
            <phoebe.frontend.bundle.Bundle.get_distribution_collection>.

        Returns
        ----------
        * (distl.Latex object): with methods as_string and as_latex
        """
        return self.get_distribution_collection(twig=twig, parameters=parameters, set_labels=kwargs.pop('set_labels', True), **kwargs)[0].uncertainties(sigma=sigma, tex=tex)

    def calculate_lnp(self, twig=None,
                      **kwargs):
        """
        Compute the log-probability between a distribution collection
        (see <phoebe.frontend.bundle.Bundle.get_distribution_collection>
        and the face-values of the corresponding parameters.  If the
        distribution collection corresponds to 'priors', then this is effectively
        lnpriors, and if the distribution collection refers to 'posteriors'
        (or a solution), then this is effectively lnposteriors.

        This will attempt to compute the log-probability respecting covariances,
        but will fallback on dropping covariances if necessary, with a message
        raise in the error <phoebe.logger>.

        Only parameters (or distribution parameters) included in the ParameterSet
        (after filtering with `**kwargs`) will be included in the summed
        log-probability.

        See also:
        * <phoebe.parameters.ParameterSet.calculate_lnlikelihood>
        * <phoebe.frontend.bundle.Bundle.get_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.sample_distribution_collection>
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.parameters.DistributionParameter.lnp>

        Arguments
        -----------
        * `twig`: (string, optional, default=None): twig to use for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet (eg. init_from or priors),
            a ParameterSet of distribution parameters, or a solution ParameterSet
            that supports multivariate distributions (eg. sampler.emcee or sampler.dynesty).
        * `combine`: (str, optional) how to combine multiple distributions for the same parameter.
            first: ignore duplicate entries and take the first entry.
            and: combine duplicate entries via AND logic, dropping covariances.
            or: combine duplicate entries via OR logic, dropping covariances.'
            Defaults to 'first' if `twig` and `**kwargs` point to distributions,
            otherwise will default to the value of the relevant parameter in the
            solver options.
        * `include_constrained` (bool, optional, default=True): whether to
            include constrained parameters.
        * `to_univariates` (bool, optional): whether to convert any multivariate
            distributions to univariates before adding to the collection.  Defaults
            to False if `twig` and `**kwargs` point to distributions, otherwise
            will default to the value necessary for the solver backend.
        * `to_uniforms` (bool or int, optional): whether to convert all distributions
            to uniforms (and if an int, then what sigma to adopt for non-uniform
            distributions).  Defaults to False if `twig` and `**kwargs` point to
            distributions, otherwise will default to the value necessary for the
            solver backend.
        * `**kwargs`: additional keyword arguments are used for filtering.
            `twig` and `**kwargs` must result in either a single supported
            parameter in a solver ParameterSet, or a ParameterSet of distribution
            parameters.  If pointing to a solution ParameterSet, `**kwargs` can
            also be used to override `distributions_convert` and `distributions_bins`,
            as well as any other solution parameters (eg. `burnin`, `thin`, etc).


        Returns
        -----------
        * (float) log-prior value
        """
        # TODO: should we require run_checks to pass?
        self.run_delayed_constraints()
        self.run_failed_constraints()

        # TODO: check to see if dist_param references a constrained parameter,
        # and if so, raise a warning if all other parameters in the constraint
        # also have attached distributions?

        kwargs['keys'] = 'uniqueid'
        kwargs.setdefault('include_constrained', True)
        dc, uniqueids = self.get_distribution_collection(twig=twig,
                                                         **kwargs)

        # uniqueids needs to correspond to dc.dists_unpacked, not dc.dists
        if len(dc.dists_unpacked) == len(uniqueids):
            values = [self.get_value(uniqueid=uid, unit=dist.unit, **_skip_filter_checks) for uid, dist in zip(uniqueids, dc.dists_unpacked)]
        elif len(dc.dists) == len(uniqueids):
            values = [self.get_value(uniqueid=uid, unit=dist.unit, **_skip_filter_checks) for uid, dist in zip(uniqueids, dc.dists)]
        else:
            ps = self.exclude(context=['distribution', 'constraint'], **_skip_filter_checks)
            values = [ps.get_value(twig=dist.label, unit=dist.unit, **_skip_filter_checks) for dist in dc.dists_unpacked]

        try:
            return dc.logpdf(values, as_univariates=False)
        except Exception as e:
            logger.error("calculate_pdf({}, **{}) failed with as_univariates=False, falling back on as_univariates=True (covariances will be dropped in any multivariate distributions).  Original error: {}".format(twig, kwargs, e))
            return dc.logpdf(values, as_univariates=True)

    @send_if_client
    def add_figure(self, kind, return_changes=False, **kwargs):
        """
        Add a new figure to the bundle.  If not provided,
        figure` (the name of the new figure) will be created for
        you and can be accessed by the `figure` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        ```py
        b.add_figure(figure.dataset.lc)
        ```

        or

        ```py
        b.add_figure('lc', x='phases')
        ```

        Available kinds can be found in <phoebe.parameters.figure> and include:
        * <phoebe.parameters.figure.dataset.lc>
        * <phoebe.parameters.figure.dataset.rv>
        * <phoebe.parameters.figure.dataset.lp>
        * <phoebe.parameters.figure.dataset.orb>
        * <phoebe.parameters.figure.dataset.mesh>
        * <phoebe.parameters.distribution.distribution_collection>
        * <phoebe.parameters.solution.lc_periodogram>
        * <phoebe.parameters.solution.rv_periodogram>
        * <phoebe.parameters.solution.lc_geometry>
        * <phoebe.parameters.solution.rv_geometry>
        * <phoebe.parameters.solution.ebai>
        * <phoebe.parameters.solution.emcee>
        * <phoebe.parameters.solution.dynesty>

        See also:
        * <phoebe.frontend.bundle.Bundle.get_figure>
        * <phoebe.frontend.bundle.Bundle.remove_figure>
        * <phoebe.frontend.bundle.Bundle.rename_figure>
        * <phoebe.frontend.bundle.Bundle.run_figure>
        * <phoebe.list_available_figures>

        Arguments
        ----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts the bundle and default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.figure> module.
        * `figure` (string, optional): name of the newly-created figure.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing figure with the same `figure` tag.  If False,
            an error will be raised.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added
        """

        func = _get_add_func(_figure, kind)
        kind = func.__name__

        if kwargs.get('figure', False) is None:
            # then we want to apply the default below, so let's pop for now
            _ = kwargs.pop('figure')


        default_label_base = {'distribution_collection': 'dc'}.get(kind, kind)
        kwargs.setdefault('figure',
                          self._default_label(default_label_base+'fig',
                                              **{'context': 'figure',
                                                 'kind': kind}))

        if kwargs.pop('check_label', True):
            self._check_label(kwargs['figure'], allow_overwrite=kwargs.get('overwrite', False))

        # NOTE: we won't pass kwargs since some choices need to be updated.
        # Instead we'll apply kwargs after calling all self._handle_*
        params = func(self)


        metawargs = {'context': 'figure',
                     'figure': kwargs['figure'],
                     'kind': kind}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_figure(figure=kwargs['figure'], during_overwrite=True)
            # check the label again, just in case kwargs['figure'] belongs to
            # something other than component
            self.exclude(figure=kwargs['figure'])._check_label(kwargs['figure'], allow_overwrite=False)
        else:
            removed_ps = None

        self._attach_params(params, **metawargs)
        # attach params called _check_copy_for, but only on it's own parameterset
        # self._check_copy_for()

        # for constraint in constraints:
            # self.add_constraint(*constraint)

        ret_ps = self.filter(figure=kwargs['figure'], check_visible=False, check_default=False)

        ret_changes = []
        ret_changes += self._handle_figure_selectparams(return_changes=return_changes)
        ret_changes += self._handle_dataset_selectparams(return_changes=return_changes)
        ret_changes += self._handle_model_selectparams(return_changes=return_changes)
        ret_changes += self._handle_component_selectparams(return_changes=return_changes)
        ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_meshcolor_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_solution_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_solution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_solver_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_solver_selectparams(return_changes=return_changes)
        ret_changes += self._handle_figure_time_source_params(return_changes=return_changes)

        # now set parameters that needed updated choices
        qualifiers = ret_ps.qualifiers
        for k,v in kwargs.items():
            if k in qualifiers:
                try:
                    ret_ps.set_value_all(qualifier=k, value=v, **_skip_filter_checks)
                except:
                    self.remove_figure(figure=kwargs['figure'])
                    raise
            # TODO: else raise warning?

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs,
                            additional_allowed_keys=['overwrite'],
                            warning_only=True, ps=ret_ps)

        if kwargs.get('overwrite', False) and return_changes:
            ret_ps += overwrite_ps

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_figure(self, figure=None, **kwargs):
        """
        Filter in the 'figure' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.frontend.bundle.Bundle.add_figure>
        * <phoebe.frontend.bundle.Bundle.remove_figure>
        * <phoebe.frontend.bundle.Bundle.rename_figure>
        * <phoebe.frontend.bundle.Bundle.run_figure>

        Arguments
        ----------
        * `figure`: (string, optional, default=None): the name of the figure
        * `**kwargs`: any other tags to do the filtering (excluding figure and context)

        Returns
        ----------
        * a <phoebe.parameters.ParameterSet> object.
        """
        if figure is not None:
            kwargs['figure'] = figure
            if figure not in self.figures:
                raise ValueError("figure='{}' not found".format(figure))

        kwargs['context'] = 'figure'
        ret_ps = self.filter(**kwargs).exclude(figure=[None])

        if len(ret_ps.figures) == 0:
            raise ValueError("no figures matched: {}".format(kwargs))
        elif len(ret_ps.figures) > 1:
            raise ValueError("more than one figure matched: {}".format(kwargs))

        return _return_ps(self, ret_ps)

    @send_if_client
    def remove_figure(self, figure, return_changes=False, **kwargs):
        """
        Remove a 'figure' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        * <phoebe.frontend.bundle.Bundle.add_figure>
        * <phoebe.frontend.bundle.Bundle.get_figure>
        * <phoebe.frontend.bundle.Bundle.rename_figure>
        * <phoebe.frontend.bundle.Bundle.run_figure>

        Arguments
        ----------
        * `figure` (string): the label of the figure to be removed.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: figure, context

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        kwargs['figure'] = figure
        kwargs['context'] = 'figure'
        ret_ps = self.remove_parameters_all(**kwargs)

        ret_changes = []
        if not kwargs.get('during_overwrite', False):
            ret_changes += self._handle_figure_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    @send_if_client
    def rename_figure(self, old_figure, new_figure,
                      overwrite=False, return_changes=False):
        """
        Change the label of a figure attached to the Bundle.

        See also:
        * <phoebe.frontend.bundle.Bundle.add_figure>
        * <phoebe.frontend.bundle.Bundle.get_figure>
        * <phoebe.frontend.bundle.Bundle.remove_figure>
        * <phoebe.frontend.bundle.Bundle.run_figure>

        Arguments
        ----------
        * `old_figure` (string): current label of the figure (must exist)
        * `new_figure` (string): the desired new label of the figure
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed figure

        Raises
        --------
        * ValueError: if the value of `new_figure` is forbidden or already exists.
        """
        # TODO: raise error if old_figure not found?
        self._rename_label('figure', old_figure, new_figure, overwrite)

        return self.filter(figure=new_figure)

    def run_figure(self, figure=None, **kwargs):
        """
        Plot a figure for a set of figure options attached to the bundle.

        For plotting without the help of figure options, see
        <phoebe.parameters.ParameterSet.plot>.

        In general, `run_figure` is useful for creating simple plots with
        consistent defaults for styling across datasets/components/etc,
        when plotting from a UI, or when wanting to save plotting options
        along with the bundle rather than in a script.  `plot` is more
        more flexible, allows for multiple subplots and advanced positioning,
        and is less clumsy if plotting from the python frontend.

        See also:
        * <phoebe.frontend.bundle.Bundle.ui_figures>
        * <phoebe.frontend.bundle.Bundle.add_figure>
        * <phoebe.frontend.bundle.Bundle.get_figure>
        * <phoebe.frontend.bundle.Bundle.remove_figure>
        * <phoebe.frontend.bundle.Bundle.rename_figure>
        * <phoebe.frontend.bundle.Bundle.run_checks_figure>

        Arguments
        -----------
        * `figure` (string or list, optional): name of the figure(s) options to use.
            If not provided or None, run_figure will run all attached figures
            in subplots, as necessary.
        * `time` (float, optional): time to use for plotting/animating.  This will
            filter on time for any applicable dataset (i.e. meshes, line profiles),
            will be used for highlighting/uncovering based on the passed value
            to `highlight` and `uncover`.  Use `times` to set the individual
            frames when animating with `animate=True`
        * `times` (list/array, optional): times to use for animating.  If
            `animate` is not True, a warning will be raised in the logger.  If
            `animate` is True, and neither `times` nor `time` is passed,
            then the animation will cycle over the tagged times of the model
            datasets (i.e. if mesh or lp datasets exist), or the computed
            times otherwise.
        * `save` (string, optional, default=False): filename to save the
            figure (or False to not save).
        * `show` (bool, optional, default=False): whether to show the plot
        * `animate` (bool, optional, default=False): whether to animate the figure.
        * `interval` (int, optional, default=100): time in ms between each
            frame in the animation.  Applicable only if `animate` is True.
        * `**kwargs`: all additional keyword arguments will be used to override
            parameters in the figure options or passed along to
            <phoebe.parameters.ParameterSet.plot>.  See the API docs for
            <phoebe.parameters.ParameterSet.plot> for an exhaustive list
            of plotting options.  If `figure` is passed as a list (or as None and
            multiple figures exist in the bundle), then kwargs can be passed
            as dictionaries with keys of the individual `figure` labels, which
            will be applied to individual `run_figure` calls when matching.
            For example: `b.run_figure(figure=['lcfig01', 'rvfig01'], axpos={'lcfig01': 121, 'rvfig01', 122}, show=True)`

        Returns
        -----------
        * (autofig figure, matplotlib figure) - the output from the call to
            <phoebe.parameters.ParameterSet.plot>


        Raises
        ----------
        * ValueError: if `figure` is not provided but is required.

        """
        if figure is None:
            figure = self.figures

        if not kwargs.get('skip_checks', False):
            self.run_checks_figure(figure=figure,
                                   raise_logger_warning=True, raise_error=True,
                                   **kwargs)


        qmap = {'color': 'c'}

        if isinstance(figure, list) or isinstance(figure, tuple):
            figures = figure
            show = kwargs.pop('show', False)
            save = kwargs.pop('save', False)
            animate = kwargs.pop('animate', False)
            for figure in figures:
                self.run_figure(figure=figure, **{k: v.get(figure) if isinstance(v, dict) and figure in v.keys() else v for k,v in kwargs.items()})

            return self._show_or_save(save, show, animate, **kwargs)


        fig_ps = self.get_figure(figure=figure, **{k:v for k,v in kwargs.items() if k not in ['show', 'save', 'animate']})
        # errors are currently included in get_figure, so don't need to repeat here
        # if len(fig_ps.figures) == 0:
        #     raise ValueError("no figure found")
        # elif len(fig_ps.figures) > 1:
        #     raise ValueError("more than one figure found")

        kwargs['check_default'] = False
        kwargs['check_visible'] = False

        if fig_ps.kind in self.filter(context='dataset', **_skip_filter_checks).kinds:
            ds_kind = fig_ps.kind
            kwargs['kind'] = ds_kind
            ds_same_kind = self.filter(context='dataset', kind=ds_kind, **_skip_filter_checks).datasets
            ml_same_kind = self.filter(context='model', kind=ds_kind, **_skip_filter_checks).models
            comp_same_kind = self.filter(context=['dataset', 'model'], kind=ds_kind, **_skip_filter_checks).components

            kwargs.setdefault('kind', ds_kind)
            if 'contexts' in fig_ps.qualifiers:
                kwargs.setdefault('context', fig_ps.get_value(qualifier='contexts', expand=True, **_skip_filter_checks))
            else:
                kwargs['context'] = 'model'


            if 'datasets' in fig_ps.qualifiers:
                kwargs.setdefault('dataset', fig_ps.get_value(qualifier='datasets', expand=True, **_skip_filter_checks))
            if 'models' in fig_ps.qualifiers:
                kwargs.setdefault('model', [None] + fig_ps.get_value(qualifier='models', expand=True, **_skip_filter_checks))
            if 'components' in fig_ps.qualifiers:
                kwargs.setdefault('component', fig_ps.get_value(qualifier='components', expand=True, **_skip_filter_checks))

            kwargs.setdefault('legend', fig_ps.get_value(qualifier='legend', **_skip_filter_checks))

            for q in ['draw_sidebars', 'uncover', 'highlight', 'period', 't0']:
                if q in fig_ps.qualifiers:
                    kwargs.setdefault(q, fig_ps.get_value(qualifier=q, **_skip_filter_checks))

            if 'time_source' in fig_ps.qualifiers:
                time_source = fig_ps.get_value(qualifier='time_source', **_skip_filter_checks)
                if time_source == 'default':
                    time_source = self.get_value(qualifier='default_time_source', context='figure', **_skip_filter_checks)
                    if time_source == 'manual':
                        kwargs.setdefault('time', self.get_value(qualifier='default_time', context='figure', **_skip_filter_checks))
                    elif time_source == 'None':
                        # then we don't do anything
                        pass
                    elif ' (' in time_source:
                        kwargs.setdefault('time', float(time_source.split(' ')[0]))
                    else:
                        # probably a t0 of some sort, which we can pass directly as the string
                        kwargs.setdefault('time', time_source)

                elif time_source == 'manual':
                    kwargs.setdefault('time', fig_ps.get_value(qualifier='time', **_skip_filter_checks))
                elif time_source == 'None':
                    # then we don't do anything
                    pass
                elif ' (' in time_source:
                    kwargs.setdefault('time', float(time_source.split(' ')[0]))
                else:
                    # probably a t0 of some sort, which we can pass directly as the string
                    kwargs.setdefault('time', time_source)

            for d in ['x', 'y', 'fc', 'ec'] if ds_kind == 'mesh' else ['x', 'y']:
                if d not in ['fc', 'ec']:
                    # fc and ec are handled later because they have different options
                    kwargs.setdefault(d, fig_ps.get_value(qualifier=d, **_skip_filter_checks))

                if kwargs.get('{}label_source'.format(d), fig_ps.get_value(qualifier='{}label_source'.format(d), **_skip_filter_checks))=='manual':
                    kwargs.setdefault('{}label'.format(d), fig_ps.get_value(qualifier='{}label'.format(d), **_skip_filter_checks))

                if kwargs.get('{}unit_source'.format(d), fig_ps.get_value(qualifier='{}unit_source'.format(d), **_skip_filter_checks))=='manual':
                    kwargs.setdefault('{}unit'.format(d), fig_ps.get_value(qualifier='{}unit'.format(d), **_skip_filter_checks))

                if kwargs.get('{}lim_source'.format(d), fig_ps.get_value(qualifier='{}lim_source'.format(d), **_skip_filter_checks))=='manual':
                    lim = fig_ps.get_value(qualifier='{}lim'.format(d), **_skip_filter_checks)
                    if len(lim)==2:
                        kwargs.setdefault('{}lim'.format(d), lim)
                    else:
                        logger.warning("ignoring {}lim, must have length 2".format(lim))


            # if ds_kind in ['mesh', 'lp']:
                # kwargs.setdefault('time', fig_ps.get_value(qualifier='times', expand=True, **_skip_filter_checks))

                # if 'times' in kwargs.keys():
                    # logger.warning("")
                    # kwargs['time'] = kwargs.pop('times')


            if ds_kind in ['mesh']:
                for q in ['fc', 'ec']:
                    source = fig_ps.get_value(qualifier=q+'_source', **_skip_filter_checks)
                    if source == 'column':
                        kwargs[q] = fig_ps.get_value(qualifier=q+'_column', **_skip_filter_checks)

                        cmap_source = fig_ps.get_value(qualifier=q+'map_source', **_skip_filter_checks)
                        if cmap_source == 'manual':
                            kwargs[q+'map'] = fig_ps.get_value(qualifier=q+'map', **_skip_filter_checks)

                    elif source == 'manual':
                        kwargs[q] = fig_ps.get_value(qualifier=q, **_skip_filter_checks)
                    elif source == 'face':
                        kwargs[q] = 'face'
                    elif source == 'component':
                        kwargs[q] = {c: self.get_value(qualifier='color', component=c, context='figure', **_skip_filter_checks) for c in comp_same_kind if c in self.hierarchy.get_meshables()}
                    elif source == 'model':
                        kwargs[q] = {ml: self.get_value(qualifier='color', model=ml, context='figure', **_skip_filter_checks) for ml in ml_same_kind}

                    if kwargs[q] == 'None':
                        kwargs[q] = None

            else:
                for q in ['linestyle', 'marker', 'color']:
                    if q not in kwargs.keys():
                        if q == 'marker':
                            # don't apply markers to models
                            suff = '@dataset'
                        elif q == 'linestyle':
                            suff = '@model'
                        else:
                            suff = ''

                        source = kwargs.get('{}_source'.format(q), fig_ps.get_value(qualifier='{}_source'.format(q), **_skip_filter_checks))
                        if source == 'manual':
                            if q == 'marker':
                                kwargs[q] = {'dataset': fig_ps.get_value(qualifier=q, **_skip_filter_checks)}
                            elif q == 'linestyle':
                                kwargs[q] = {'model': fig_ps.get_value(qualifier=q, **_skip_filter_checks)}
                            else:
                                kwargs[qmap.get(q,q)] = fig_ps.get_value(qualifier=q, **_skip_filter_checks)
                        elif source == 'dataset':
                            kwargs[qmap.get(q,q)] = {ds+suff: self.get_value(qualifier=q, dataset=ds, context='figure', **_skip_filter_checks) for ds in ds_same_kind}
                        elif source == 'model':
                            kwargs[qmap.get(q,q)] = {ml+suff: self.get_value(qualifier=q, model=ml, context='figure', **_skip_filter_checks) for ml in ml_same_kind}
                        elif source == 'component':
                            kwargs[qmap.get(q,q)] = {}
                            for c in comp_same_kind:
                                try:
                                    kwargs[qmap.get(q,q)][c+suff] = self.get_value(qualifier=q, component=c, context='figure', **_skip_filter_checks)
                                except ValueError:
                                    # RVs will include orbits in comp_same kind, but we can safely skip those
                                    pass
                        else:
                            raise NotImplementedError("{}_source of {} not supported".format(q, source))


        elif fig_ps.kind in ['distribution_collection']:
            distribution_set = fig_ps.get_value(qualifier='distribution_set', distribution_sets=kwargs.get('distribution_sets', None), **_skip_filter_checks)
            if distribution_set == 'manual':
                kwargs['context'] = 'distribution'

                kwargs.setdefault('distribution', fig_ps.get_value(qualifier='distributions', distributions=kwargs.get('distributions', None), **_skip_filter_checks))
                if not len(kwargs.get('distribution')):
                    logger.warning("distributions not set, cannot plot")
                    return None, None

                kwargs['to_uniforms'] = fig_ps.get_value(qualifier='to_uniforms_sigma', to_uniforms_sigma=kwargs.get('to_uniforms_sigma', None), **_skip_filter_checks) if fig_ps.get_value(qualifier='to_uniforms', to_uniforms=kwargs.get('to_uniforms', None), **_skip_filter_checks) else False
                kwargs['to_univariates'] = True if kwargs['to_uniforms'] else fig_ps.get_value(qualifier='to_univariates', to_univariates=kwargs.get('to_univariates', None), **_skip_filter_checks)

                for k in fig_ps.qualifiers:
                    if k in ['distributions', 'to_uniforms', 'to_univariates']:
                        continue
                    kwargs.setdefault(k, fig_ps.get_value(qualifier=k, **_skip_filter_checks))
            else:
                # distribution_sets should be something like priors@emcee@solver, sample_from@phoebe01@compute, etc
                kwargs['twig'] = distribution_set

        elif 'solver' in fig_ps.qualifiers:
            kwargs['context'] = 'solver'
            solver = fig_ps.get_value(qualifier='solver', solver=kwargs.get('solver', None), **_skip_filter_checks)
            kwargs['solver'] = solver
            distribution = fig_ps.get_value(qualifier='distribution', distribution=kwargs.get('distribution', None), **_skip_filter_checks)
            kwargs['distribution_twig'] = '{}@{}'.format(distribution, solver)

        elif 'solution' in fig_ps.qualifiers:
            kwargs['context'] = 'solution'

            kwargs.setdefault('solution', fig_ps.get_value(qualifier='solution', **_skip_filter_checks))
            if not len(kwargs.get('solution')):
                logger.warning("solution not set, cannot plot")
                return None, None

            for k in fig_ps.qualifiers:
                if k in ['solution']:
                    continue
                kwargs.setdefault(k, fig_ps.get_value(qualifier=k, **_skip_filter_checks))

        else:
            raise ValueError("nothing found to plot")

        kwargs.setdefault('tight_layout', True)
        logger.info("calling plot(**{})".format(kwargs))
        return self.plot(**kwargs)

    def ui_figures(self, web_client=None, blocking=None):
        """
        Open an interactive user-interface for all figures in the Bundle.

        See <phoebe.parameters.ParameterSet.ui> for more details on the
        behavior of `blocking`, `web_client`, and Jupyter notebook support.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_figure>
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.as_client>
        * <phoebe.frontend.bundle.Bundle.is_client>

        Arguments
        -----------
        * `web_client` (bool or string, optional, default=False):
            If not provided or None, this will default to the values in the
            settings for `web_client` and `web_client_url`.
            If True, a web-client will be preferred over a desktop-client and
            will default to using the settings for `web_client_url`.
            If False, will use the desktop-client instead of a web-client.
            If a string, the string will be used as the url for the web-client.
            Note that if using a web-client, the bundle must already be
            in client mode.  See <phoebe.frontend.bundle.Bundle.is_client>
            and <phoebe.frontend.bundle.Bundle.as_client>.
        * `blocking` (bool, optional, default=None): whether the clal to the
            UI should be blocking (wait for the client to close/disconnect)
            before continuing the python-thread or not.  If not provided or
            None, will default to True if not currently in client-mode
            (see <phoebe.frontend.bundle.Bundle.is_client> and
            <phoebe.frontend.bundle.Bundle.as_client>) or False otherwise.

        Returns
        ----------
        * `url` (string): the opened URL (will attempt to launch in the system
            webbrowser)

        Raises
        -----------
        * ValueError: if the <phoebe.parameters.ParameterSet> is not attached
            to a parent <phoebe.frontend.bundle.Bundle>.
        * ValueError: if `web_client` is provided but the <phoebe.frontend.bundle.Bundle>
            is not in client mode (see <phoebe.frontend.bundle.Bundle.is_client>
            and <phoebe.frontend.bundle.Bundle.as_client>)
        """

        return self._launch_ui(web_client, 'figures', blocking=blocking)


    def compute_ld_coeffs(self, compute=None, set_value=False, **kwargs):
        """
        Compute the interpolated limb darkening coefficients.

        This method is only for convenience and will be recomputed internally
        within <phoebe.frontend.bundle.Bundle.run_compute> for all backends
        that require per-star limb-darkening coefficients.  Note that the default
        <phoebe.parameters.compute.phoebe> backend will instead interpolate
        limb-darkening coefficients **per-element**.

        Coefficients will only be interpolated/returned for those where `ld_mode`
        (or `ld_mode_bol`)  is 'lookup'.  The values of the `ld_coeffs`
        (or `ld_coeffs_bol`) parameter will be returned for cases where `ld_mode`
        is 'manual'.  Cases where `ld_mode` is 'interp' will not be included in
        the output.

        Note:
        * for backends without `atm` compute options, 'ck2004' will be used.

        Arguments
        ------------
        * `compute` (string, optional, default=None): label of the compute
            options (not required if only one is attached to the bundle).
        * `component` (string or list of strings, optional): label of the
            component(s) requested. If not provided, will be provided for all
            components in the hierarchy.
        * `dataset` (string or list of strings, optional): label of the
            dataset(s) requested.  If not provided, will be provided for all
            datasets attached to the bundle.  Include 'bol' to include
            bolometric (irradiation-only) quantities from `ld_func_bol`.
        * `set_value` (bool, optional, default=False): apply the interpolated
            values to the respective `ld_coeffs`/`ld_coeffs_bol` parameters
            (even if not currently visible).
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_compute> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `**kwargs`: any additional kwargs are sent to override compute options.

        Returns
        ----------
        * (dict) computed ld_coeffs in a dictionary with keys formatted as
            ld_coeffs@component@dataset and the ld_coeffs as values (arrays with
            appropriate length given the respective value of `ld_func`).
        """

        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        compute_ps = self.get_compute(compute, **_skip_filter_checks)
        # we'll add 'bol' to the list of default datasets... but only if bolometric is needed for irradiation
        needs_bol = compute_ps.get_value(qualifier='irrad_method', irrad_method=kwargs.get('irrad_method', None), default='none', **_skip_filter_checks) != 'none'

        datasets = kwargs.pop('dataset', self.datasets + ['bol'] if needs_bol else self.datasets)
        components = kwargs.pop('component', self.components)

        # don't allow things like model='mymodel', etc
        forbidden_keys = parameters._meta_fields_filter
        if not kwargs.get('skip_checks', False):
            self._kwargs_checks(kwargs, additional_allowed_keys=['skip_checks', 'overwrite'], additional_forbidden_keys=forbidden_keys)

        if not kwargs.get('skip_checks', False):
            report = self.run_checks_compute(compute=compute, allow_skip_constraints=False,
                                             raise_logger_warning=True, raise_error=True,
                                             run_checks_system=True,
                                             **kwargs)

        ld_coeffs_ret = {}
        ldcs_params = self.filter(qualifier='ld_coeffs_source', dataset=datasets, component=components, check_visible=False).to_list()
        if 'bol' in datasets:
            ldcs_params += self.filter(qualifier='ld_coeffs_source_bol', component=components, check_visible=False).to_list()

        for ldcs_param in ldcs_params:
            is_bol = ldcs_param.context == 'component'
            bol_suffix = '_bol' if is_bol else ''

            ld_mode = self.get_value(qualifier='ld_mode{}'.format(bol_suffix), dataset=ldcs_param.dataset, component=ldcs_param.component, check_visible=False)
            if ld_mode == 'interp':
                logger.debug("skipping computing ld_coeffs{} for {}@{} because ld_mode{}='interp'".format(bol_suffix, ldcs_param.dataset, ldcs_param.component, bol_suffix))
            elif ld_mode == 'manual':
                ld_coeffs_manual = self.get_value(qualifier='ld_coeffs{}'.format(bol_suffix), dataset=ldcs_param.dataset, component=ldcs_param.component, context='component' if is_bol else 'dataset', **_skip_filter_checks)
                ld_coeffs_ret["{}@{}@{}".format('ld_coeffs{}'.format(bol_suffix), ldcs_param.component, 'component' if is_bol else ldcs_param.dataset)] = ld_coeffs_manual
                continue
            elif ld_mode == 'lookup':
                ldcs = ldcs_param.get_value(**_skip_filter_checks)
                ld_func = self.get_value(qualifier='ld_func{}'.format(bol_suffix), dataset=ldcs_param.dataset, component=ldcs_param.component, context='component' if is_bol else 'dataset', **_skip_filter_checks)
                if is_bol:
                    passband = 'Bolometric:900-40000'
                else:
                    passband = self.get_value(qualifier='passband', dataset=ldcs_param.dataset, context='dataset', **_skip_filter_checks)

                atm = self.get_value(qualifier='atm', compute=compute, component=ldcs_param.component, default='ck2004', atm=kwargs.get('atm', None), **_skip_filter_checks)

                if ldcs == 'auto':
                    if atm in ['extern_atmx', 'extern_planckint', 'blackbody']:
                        ldcs = 'ck2004'
                    else:
                        ldcs = atm

                pb = get_passband(passband, content='{}:ld'.format(ldcs))
                teff = self.get_value(qualifier='teff', component=ldcs_param.component, context='component', unit='K', **_skip_filter_checks)
                logg = self.get_value(qualifier='logg', component=ldcs_param.component, context='component', **_skip_filter_checks)
                abun = self.get_value(qualifier='abun', component=ldcs_param.component, context='component', **_skip_filter_checks)
                if is_bol:
                    photon_weighted = False
                else:
                    photon_weighted = self.get_value(qualifier='intens_weighting', dataset=ldcs_param.dataset, context='dataset', check_visible=False) == 'photon'
                logger.info("{} ld_coeffs lookup for dataset='{}' component='{}' passband='{}' from ld_coeffs_source='{}'".format(ld_func, ldcs_param.dataset, ldcs_param.component, passband, ldcs))
                logger.debug("pb.interpole_ld_coeffs(teff={} logg={}, abun={}, ld_coeffs={} ld_func={} photon_weighted={})".format(teff, logg, abun, ldcs, ld_func, photon_weighted))
                try:
                    ld_coeffs = pb.interpolate_ldcoeffs(teff, logg, abun, ldcs, ld_func, photon_weighted)
                except ValueError as err:
                    if str(err).split(":")[0] == 'Atmosphere parameters out of bounds':
                        # let's override with a more helpful error message
                        logger.warning(str(err))
                        raise ValueError("Could not lookup ld_coeffs for {}.  Try changing ld_coeffs_source{} to a table that covers a sufficient range of values or set ld_mode{} to 'manual' and manually provide coefficients via ld_coeffs{}. Enable 'warning' logger to see out-of-bound arrays.".format(ldcs_param.twig, bol_suffix, bol_suffix, bol_suffix))
                    else:
                        raise err

                # NOTE: these may return nans... if so, run_checks will handle the error

                logger.info("interpolated {} ld_coeffs{}={}".format(ld_func, bol_suffix, ld_coeffs))

                ld_coeffs_ret["ld_coeffs{}@{}@{}".format(bol_suffix, ldcs_param.component, 'component' if is_bol else ldcs_param.dataset)] = ld_coeffs
                if set_value:
                    self.set_value(qualifier='ld_coeffs{}'.format(bol_suffix), component=ldcs_param.component, dataset=ldcs_param.dataset, context='component' if is_bol else 'dataset', check_visible=False, value=ld_coeffs)
            else:
                raise NotImplementedError("compute_ld_coeffs not implemented for ld_mode{}='{}'".format(bol_suffix, ld_mode))

        return ld_coeffs_ret

    def _compute_intrinsic_system_at_t0(self, compute=None, datasets=None, compute_l3=False, compute_l3_frac=False, compute_extrinsic=False, **kwargs):
        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        compute_kind = self.get_compute(compute).kind

        if compute_kind not in ['phoebe']:
            # then we'll override the compute options distortion_method and always use roche
            # as phoebe may not support all the same distortion_methods for these backends
            kwargs.setdefault('distortion_method', 'roche')

            atm_backend = {component: self.get_value(qualifier='atm', component=component, compute=compute, atm=kwargs.get('atm', kwargs.get('atms', {}).get(component, None)), default='ck2004', **_skip_filter_checks) for component in self.hierarchy.get_stars()}
            kwargs.setdefault('atm', atm_backend)

        # temporarily disable interactive_checks, check_default, and check_visible
        conf_interactive_checks = conf.interactive_checks
        if conf_interactive_checks:
            logger.debug("temporarily disabling interactive_checks")
            conf._interactive_checks = False


        def restore_conf():
            # restore user-set interactive checks
            if conf_interactive_checks:
                logger.debug("restoring interactive_checks={}".format(conf_interactive_checks))
                conf._interactive_checks = conf_interactive_checks


        system_compute = compute if compute_kind=='phoebe' else None
        logger.debug("creating system with compute={} kwargs={}".format(system_compute, kwargs))
        try:
            system = backends.PhoebeBackend()._compute_intrinsic_system_at_t0(self, system_compute, datasets=datasets, reset=False, lc_only=False, **kwargs)
        except Exception as err:
            restore_conf()
            raise

        restore_conf()

        return system

    def compute_l3s(self, compute=None, use_pbfluxes={},
                   set_value=False, **kwargs):
        """
        Compute third lights (`l3`) that will be applied to the system from
        fractional third light (`l3_frac`) and vice-versa by assuming that the
        total system flux (`pbflux`) is equivalent to the sum of the extrinsic (including
        any enabled irradiation and features) passband luminosities
        at t0 divided by 4*pi.  To see how passband luminosities are computed,
        see <phoebe.frontend.bundle.Bundle.compute_pblums>.

        This method is only for convenience and will be recomputed internally
        within <phoebe.frontend.bundle.Bundle.run_compute>.

        Arguments
        ------------
        * `compute` (string, optional, default=None): label of the compute
            options (not required if only one is attached to the bundle).
        * `dataset` (string or list of strings, optional): label of the
            dataset(s) requested.  If not provided, will be provided for all
            datasets in which an `l3_mode` Parameter exists.
        * `use_pbfluxes` (dictionary, optional): dictionary of dataset-total
            passband fluxes (in W/m**2) to use when converting between `l3` and
            `l3_flux`.  For any dataset not included in the dictionary, the pblums
            will be computed and adopted.  See also <phoebe.frontend.bundle.Bundle.compute_pblums>.
        * `set_value` (bool, optional, default=False): apply the computed
            values to the respective `l3` or `l3_frac` parameters (even if not
            currently visible).
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_compute> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `**kwargs`: any additional kwargs are sent to override compute options
            and are passed to <phoebe.frontend.bundle.Bundle.compute_pblums> if
            necessary.

        Returns
        ----------
        * (dict) computed l3s in a dictionary with keys formatted as
            l3@dataset or l3_frac@dataset and the l3 (as quantity objects
            with units of W/m**2) or l3_frac (as unitless floats).
        """
        logger.debug("b.compute_l3s")

        datasets = kwargs.pop('dataset', self.filter('l3_mode', check_visible=True).datasets)
        if isinstance(datasets, str):
            datasets = [datasets]



        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        self.run_delayed_constraints()

        datasets_need_pbflux = [d for d in datasets if d not in use_pbfluxes.keys()]
        if len(datasets_need_pbflux):
            _, _, _, _, compute_pblums_pbfluxes = self.compute_pblums(compute=compute, dataset=datasets_need_pbflux, ret_structured_dicts=True, **kwargs)
            for dataset in datasets_need_pbflux:
                use_pbfluxes[dataset] = compute_pblums_pbfluxes.get(dataset)

        elif not kwargs.get('skip_checks', False):
            report = self.run_checks_compute(compute=compute, allow_skip_constraints=False,
                                             raise_logger_warning=True, raise_error=True,
                                             run_checks_system=True,
                                             **kwargs)

            # don't allow things like model='mymodel', etc
            if not kwargs.get('skip_checks', False):
                forbidden_keys = parameters._meta_fields_filter
                compute_ps = self.get_compute(compute, **_skip_filter_checks)
                self._kwargs_checks(kwargs, additional_allowed_keys=['system', 'skip_checks', 'ret_structured_dicts', 'pblum_method']+compute_ps.qualifiers, additional_forbidden_keys=forbidden_keys)

        ret_structured_dicts = kwargs.get('ret_structured_dicts', False)
        l3s = {}
        for dataset in datasets:
            l3_mode = self.get_value(qualifier='l3_mode', context='dataset', dataset=dataset, **_skip_filter_checks)
            if l3_mode == 'flux':
                l3_flux = self.get_value(qualifier='l3', context='dataset', dataset=dataset, unit=u.W/u.m**2, **_skip_filter_checks)
                # pbflux could be 0.0 for the distortion_method='none' case
                l3_frac = l3_flux / (l3_flux + use_pbfluxes.get(dataset)) if use_pbfluxes.get(dataset) != 0.0 else 0.0
                if ret_structured_dicts:
                    l3s[dataset] = l3_flux
                else:
                    l3s['l3_frac@{}'.format(dataset)] = l3_frac
                if set_value:
                    self.set_value(qualifier='l3_frac', context='dataset', dataset=dataset, check_visible=False, value=l3_frac)

            elif l3_mode == 'fraction':
                l3_frac = self.get_value(qualifier='l3_frac', context='dataset', dataset=dataset, **_skip_filter_checks)
                l3_flux = (l3_frac)/(1-l3_frac) * use_pbfluxes.get(dataset)

                if ret_structured_dicts:
                    l3s[dataset] = l3_flux
                else:
                    l3s['l3@{}'.format(dataset)] = l3_flux

                if set_value:
                    self.set_value(qualifier='l3', context='dataset', dataset=dataset, check_visible=False, value=l3_flux)

            else:
                raise NotImplementedError("l3_mode='{}' not supported.".format(l3_mode))

        return l3s

    def compute_pblums(self, compute=None, model=None, pblum=True, pblum_abs=False,
                       pblum_scale=False, pbflux=False,
                       set_value=False, **kwargs):
        """
        Compute the passband luminosities that will be applied to the system,
        following all coupling, etc, as well as all relevant compute options
        (ntriangles, distortion_method, etc).
        The exposed passband luminosities (and any coupling) are computed at
        t0@system.

        Any `dataset` which does not support pblum scaling (rv or lp dataset,
        for example), will have their absolute intensities exposed.

        Additionally, an estimate for the total fluxes `pbflux`
        can optionally be computed.  These will also be computed at t0@system,
        under the spherical assumption where `pbflux = sum(pblum / (4 pi))`.
        The total flux from a light curve can then be estimated as `pbflux / d^2 + l3`

        For any dataset with `pblum_mode='dataset-scaled'` or `pblum_mode='dataset-coupled'`
        where `pblum_dataset` references a dataset-scaled dataset, `pblum`,
        `pblum_scale`, and `pbflux` are excluded from the output (but `pblum_abs`
        can be exposed), unless `model` is provided (see below) in which case
        the scaling factor in the model will be adopted to translate from absolute
        to relative units.

        Note about eclipses: `pbflux` estimates will not include
        any eclipsing or ellipsoidal effects (even if an eclipse occurs at time
        `t0`) as they are estimated directly from the luminosities under the
        spherical assumption.

        Note about contact systems: `component` defaults to
        <phoebe.parameters.HierarchyParameter.get_stars> +
        <phoebe.parameters.HierarchyParameter.get_envelopes>.  Under-the-hood,
        PHOEBE uses individual star luminosities to handle scaling, and the
        expose luminosity for envelopes is just the sum of its individual components.
        Note that these are then susceptible to the way in which the components are split in
        the neck - so a contact system consisting of two identical "stars" may
        return slightly different luminosities for the individual sub-components.
        These values should converge if you increase ntriangles in the compute
        options.

        Note about boosting: as boosting is an aspect-dependent effect that
        does not affect normal intensities, boosting will not be included
        in any of the returned values, including `pbflux_ext` due to the
        approximation of flux explained above.

        This method is only for convenience and will be recomputed internally
        within <phoebe.frontend.bundle.Bundle.run_compute> as needed.
        Alternatively, you can create a mesh dataset
        (see <phoebe.frontend.bundle.Bundle.add_dataset>
        and <phoebe.parameters.dataset.mesh>) and request any specific pblum to
        be exposed (per-time).

        Note:
        * for backends without `mesh_method` compute options, the most appropriate
            method will be chosen.  'roche' will be used whenever applicable,
            otherwise 'sphere' will be used.

        Arguments
        ------------
        * `compute` (string, optional, default=None): label of the compute
            options (not required if only one is attached to the bundle).
        * `model` (string, optional, default=None): label of the model to use
            for scaling absolute luminosities for any cases where
            `pblum_mode='dataset-scaled'`.  If not provided, entries
            using 'dataset-scaled' will be excluded from the output.
        * `pblum` (bool, optional, default=True): whether to include
            intrinsic (excluding irradiation & features) pblums.  These
            will be exposed in the returned dictionary as pblum@component@dataset.
        * `pblum_abs` (bool, optional, default=True): whether to include
            absolute intrinsic (excluding irradiation & features) pblums.  These
            will be exposed in the returned dictionary as pblum_abs@component@dataset.
        * `pblum_scale` (bool, optional, default=True): whether to include
            the scaling factor between absolute and scaled pblums.  These
            will be exposed in the returned dictionary as pblum_scale@component@dataset.
        * `pbflux` (bool, optional, default=False): whether to include
            intrinsic per-system passband fluxes (before including third light
            or distance).  These will be exposed as pbflux@dataset.
            Note: this will sum over all components, regardless of `component`.
        * `component` (string or list of strings, optional): label of the
            component(s) requested. If not provided, will default to all stars
            and envelopes in the hierarchy (see
            <phoebe.parameters.HierarchyParameter.get_stars> and
            <phoebe.parameters.HierarchyParameter.get_envelopes>).
        * `dataset` (string or list of strings, optional): label of the
            dataset(s) requested.  If not provided, will be provided for all
            passband-dependent datasets attached to the bundle.  Those without
            a pblum_mode parameter (eg. rv or lp datasets) will be computed
            in absolute luminosities.  Note that any valid entries in `dataset`
            with pblum_mode='dataset-scaled' will be ommitted from the output
            without raising an error (but will raise a <phoebe.logger> warning,
            if enabled).
        * `set_value` (bool, optional, default=False): apply the computed
            values to the respective `pblum` parameters (even if not
            currently visible).  This is often used internally to handle
            various options for pblum_mode for alternate backends that require
            passband luminosities or surface brightnesses as input, but is not
            ever required to be called manually.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_compute> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `**kwargs`: any additional kwargs are sent to override compute options.

        Returns
        ----------
        * (dict) computed pblums in a dictionary with keys formatted as
            pblum@component@dataset (for intrinsic pblums) and the pblums
            as values (as quantity objects with default units of W).

        Raises
        ----------
        * ValueError: if `compute` needs to be provided but is not.
        * ValueError: if any value in `dataset` points to a dataset that is not
            passband-dependent (eg. a mesh or orb dataset) or is not a valid
            dataset attached to the bundle'.
        * ValueError: if any value in `component` is not a valid star or envelope
            in the hierarchy.
        * ValueError: if the system fails to pass
            <phoebe.frontend.bundle.Bundle.run_checks_compute>.
        """
        logger.debug("b.compute_pblums")

        datasets = kwargs.pop('dataset', self.filter(qualifier='passband').datasets)
        if isinstance(datasets, str):
            datasets = [datasets]

        valid_components = self.hierarchy.get_stars()+self.hierarchy.get_envelopes()
        if 'component' in kwargs.keys():
            components = kwargs.pop('component')

            if isinstance(components, str):
                components = [components]

            not_valid = [c not in valid_components for c in components]
            if np.any(not_valid):
                raise ValueError("{} are not valid components.  Must be in {}".format([c for c,nv in zip(components, not_valid) if nv], valid_components))

        else:
            components = valid_components

        # check to make sure value of passed compute is valid
        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        compute_ps = self.get_compute(compute=compute, **_skip_filter_checks)
        # NOTE: this is flipped so that stefan-boltzmann can manually be used even if the compute-options have kind='phoebe' and don't have that choice
        pblum_method = kwargs.pop('pblum_method', compute_ps.get_value(qualifier='pblum_method', default='phoebe', **_skip_filter_checks))
        t0 = self.get_value(qualifier='t0', context='system', unit=u.d, t0=kwargs.pop('t0', None), **_skip_filter_checks)

        # don't allow things like model='mymodel', etc
        forbidden_keys = parameters._meta_fields_filter
        if not kwargs.get('skip_checks', False):
            self._kwargs_checks(kwargs, additional_allowed_keys=['system', 'skip_checks', 'ret_structured_dicts', 'overwrite', 'pblum_mode', 'pblum_method']+compute_ps.qualifiers, additional_forbidden_keys=forbidden_keys)

        self.run_delayed_constraints()

        # make sure we pass system checks
        if not kwargs.get('skip_checks', False):
            report = self.run_checks_compute(compute=compute_ps.compute,
                                             allow_skip_constraints=False,
                                             raise_logger_warning=True, raise_error=True,
                                             run_checks_system=True,
                                             pblum_method=pblum_method, # already popped from kwargs
                                             **kwargs)

        # determine datasets which need intensities computed and check to make
        # sure all passed datasets are passband-dependent
        pblum_datasets = datasets
        for dataset in datasets:
            if not len(self.filter(qualifier='passband', dataset=dataset, **_skip_filter_checks)):
                if dataset not in self.datasets:
                    raise ValueError("dataset '{}' is not a valid dataset attached to the bundle".format(dataset))
                raise ValueError("dataset '{}' is not passband-dependent".format(dataset))
            for pblum_ref_param in self.filter(qualifier='pblum_dataset', dataset=dataset, check_visible=True).to_list():
                ref_dataset = pblum_ref_param.get_value(**_skip_filter_checks)
                if ref_dataset in self.datasets and ref_dataset not in pblum_datasets:
                    # then we need to compute the system at this dataset too,
                    # even though it isn't requested to be returned
                    pblum_datasets.append(ref_dataset)

        atms = {}
        # note here that we aren't including the envelopes as they don't have atm parameters
        for component in self.hierarchy.get_stars():
            atm = compute_ps.get_value(qualifier='atm', component=component, atm=kwargs.get('atm', None), **_skip_filter_checks)
            if atm == 'extern_planckint':
                atm = 'blackbody'
            elif atm == 'extern_atmx':
                atm = 'ck2004'

            atms[component] = atm

        # preparation depending on method before looping over datasets/components
        if pblum_method == 'phoebe':
            # we'll need to make sure we've done any necessary interpolation if
            # any ld_bol or ld_mode_bol are set to 'lookup'.
            if not kwargs.get('skip_compute_ld_coeffs', False):
                self.compute_ld_coeffs(compute=compute, set_value=True, skip_checks=True, **{k:v for k,v in kwargs.items() if k not in ['ret_structured_dicts', 'pblum_mode', 'pblum_method', 'skip_checks']})
            # TODO: make sure this accepts all compute parameter overrides (distortion_method, etc)
            system = kwargs.get('system', self._compute_intrinsic_system_at_t0(compute=compute, datasets=pblum_datasets, atms=atms, **kwargs))
            logger.debug("computing observables with ignore_effects=True for {}".format(pblum_datasets))
            system.populate_observables(t0, ['lc'], pblum_datasets, ignore_effects=True)
        elif pblum_method == 'stefan-boltzmann':
            requivs = {component: self.get_value(qualifier='requiv', component=component, context='component', unit='m', **_skip_filter_checks) for component in valid_components}
            teffs = {component: self.get_value(qualifier='teff', component=component, context='component', unit='K', **_skip_filter_checks) for component in valid_components}
            loggs = {component: self.get_value(qualifier='logg', component=component, context='component', **_skip_filter_checks) for component in valid_components}
            abuns = {component: self.get_value(qualifier='abun', component=component, context='component', **_skip_filter_checks) for component in valid_components}

            system = None

        else:
            raise ValueError("pblum_method='{}' not supported".format(pblum_method))

        ret_structured_dicts = kwargs.get('ret_structured_dicts', False)
        ret = {}

        # pblum_*: {dataset: {component: value}}
        pblums_abs = {dataset: {} for dataset in datasets}
        pblums_scale = {dataset: {} for dataset in datasets}
        pblums_rel = {dataset: {} for dataset in datasets}
        # pbfluxes: {datasets: pbflux}
        pbfluxes = {}


        # first we'll determine the absolute luminosities per-dataset, per-component
        # by using pblum_method
        for dataset in datasets:
            if pblum_method == 'phoebe':

                system_items = {}
                for component, item in system.items():
                    system_items[component] = item
                    if hasattr(item, '_halves'):
                        # then we also want to iterate over the envelope halves
                        for half in item._halves:
                            system_items[half.component] = half

                for component, star in system_items.items():
                    if component not in valid_components:
                        continue

                    pblums_abs[dataset][component] = float(star.compute_luminosity(dataset, scaled=False))

            elif pblum_method == 'stefan-boltzmann':
                for component in valid_components:
                    passband = self.get_value(qualifier='passband', dataset=dataset, context='dataset', **_skip_filter_checks)
                    ld_mode = self.get_value(qualifier='ld_mode', component=component, dataset=dataset, context='dataset', **_skip_filter_checks)
                    intens_weighting = self.get_value(qualifier='intens_weighting', dataset=dataset, context='dataset', **_skip_filter_checks)
                    if ld_mode == 'manual':
                        ld_func = self.get_value(qualifier='ld_func', component=component, dataset=dataset, context='dataset', **_skip_filter_checks)
                        ld_coeffs = self.get_value(qualifier='ld_coeffs', component=component, dataset=dataset, context='dataset', **_skip_filter_checks)
                    elif ld_mode == 'lookup':
                        ld_func = self.get_value(qualifier='ld_func', component=component, dataset=dataset, context='dataset', **_skip_filter_checks)
                        # TODO: can we optimize this or have some kwarg if this has already been done?
                        if not kwargs.get('skip_compute_ld_coeffs', False):
                            self.compute_ld_coeffs(compute=compute, dataset=dataset, set_value=True, skip_checks=True, **{k:v for k,v in kwargs.items() if k not in ['ret_structured_dicts', 'pblum_mode', 'pblum_method', 'skip_checks']})
                        ld_coeffs = self.get_value(qualifier='ld_coeffs', component=component, dataset=dataset, context='dataset', **_skip_filter_checks)
                    else:
                        ld_func = 'interp'
                        ld_coeffs = None

                    if atms[component] == 'blackbody' and ld_mode!='manual':
                        raise NotImplementedError("pblum_method='stefan-boltzmann' not currently implemented for atm='blackbody' unless ld_mode='manual'")

                    required_content = ['{}:Inorm'.format(atms[component])]
                    if atms[component] != 'blackbody':
                        required_content += ['{}:ldint'.format(atms[component])]
                    pb = get_passband(passband, content=required_content)

                    # TODO: why is Inorm returning an array when passing all floats but ldint isn't??
                    try:
                        Inorm = pb.Inorm(Teff=teffs[component], logg=loggs[component],
                                         abun=abuns[component], atm=atms[component],
                                         ldatm=atms[component],
                                         ldint=None, ld_func=ld_func, ld_coeffs=ld_coeffs,
                                         photon_weighted=intens_weighting=='photon')[0]
                    except ValueError as err:
                        if str(err).split(":")[0] == 'Atmosphere parameters out of bounds':
                            # let's override with a more helpful error message
                            logger.warning(str(err))
                            raise ValueError("compute_pblums failed with pblum_method='{}', atm='{}', ld_mode='{}' with an atmosphere out-of-bounds error when querying for Inorm. Enable 'warning' logger to see out-of-bound arrays.".format(pblum_method, atms[component], ld_mode))
                        else:
                            raise err

                    try:
                        ldint = pb.ldint(Teff=teffs[component], logg=loggs[component],
                                         abun=abuns[component],
                                         ldatm=atms[component], ld_func=ld_func, ld_coeffs=ld_coeffs,
                                         photon_weighted=intens_weighting=='photon')
                    except ValueError as err:
                        if str(err).split(":")[0] == 'Atmosphere parameters out of bounds':
                            # let's override with a more helpful error message
                            logger.warning(str(err))
                            raise ValueError("compute_pblums failed with pblum_method='{}', atm='{}', ld_mode='{}' with an atmosphere out-of-bounds error when querying for ldint. Enable 'warning' logger to see out-of-bound arrays.".format(pblum_method, atms[component], ld_mode))
                        else:
                            raise err


                    if intens_weighting=='photon':
                        ptfarea = pb.ptf_photon_area/pb.h/pb.c
                    else:
                        ptfarea = pb.ptf_area

                    logger.info("estimating pblum for {}@{} using atm='{}' and stefan-boltzmann approximation".format(dataset, component, atm))
                    # requiv in m, Inorm in W/m**3, ldint unitless, ptfarea in m -> pblum_abs in W
                    pblums_abs[dataset][component] = 4 * np.pi * requivs[component]**2 * Inorm * ldint * ptfarea

            else:
                raise ValueError("pblum_method='{}' not supported".format(pblum_method))


        # now based on pblum_mode, we'll determine the necessary scaling factors
        # and therefore relative pblums
        pblum_scale_copy_ds = {}
        for dataset in datasets:
            ds = self.get_dataset(dataset=dataset, **_skip_filter_checks)
            pblum_mode = ds.get_value(qualifier='pblum_mode', pblum_mode=kwargs.get('pblum_mode', None), default='absolute', **_skip_filter_checks)

            if pblum_mode == 'decoupled':
                for component in valid_components:
                    if component=='_default':
                        continue

                    # then we want the pblum defined in the dataset, so the
                    # scale must be the requested pblum over the absolute value
                    # that was passed (which was likely either computed through
                    # a mesh or estimated using Stefan-Boltzmann/spherical
                    # approximation)
                    pblum = ds.get_value(qualifier='pblum', unit=u.W, component=component, **_skip_filter_checks)
                    pblums_scale[dataset][component] = pblum / pblums_abs[dataset][component] if pblums_abs[dataset][component] != 0.0 else 0.0

            elif pblum_mode == 'component-coupled':
                # now for each component we need to store the scaling factor between
                # absolute and relative intensities
                pblum_scale_copy_comp = {}
                pblum_component = ds.get_value(qualifier='pblum_component', **_skip_filter_checks)
                for component in valid_components:
                    if component=='_default':
                        continue
                    if pblum_component==component:
                        # then we do the same as in the decoupled case
                        # for this component
                        pblum = ds.get_value(qualifier='pblum', unit=u.W, component=component, **_skip_filter_checks)
                        pblums_scale[dataset][component] = pblum / pblums_abs[dataset][component] if pblums_abs[dataset][component] != 0.0 else 0.0
                    else:
                        # then this component wants to copy the scale from another component
                        # in the system.  We'll just store this now so that we make sure the
                        # component we're copying from has a chance to compute its scale
                        # first.
                        pblum_scale_copy_comp[component] = pblum_component

                # now let's copy all the scales for those that are just referencing another component
                for comp, comp_copy in pblum_scale_copy_comp.items():
                    pblums_scale[dataset][comp] = pblums_scale[dataset][comp_copy]

            elif pblum_mode == 'dataset-coupled':
                pblum_ref = ds.get_value(qualifier='pblum_dataset', **_skip_filter_checks)
                # similarly to the component-coupled case, we'll store
                # the referenced dataset and apply the scalings to the
                # dictionary once outside of the dataset loop.
                pblum_scale_copy_ds[dataset] = pblum_ref

            elif pblum_mode in ['dataset-scaled', 'absolute']:
                # even those these will default to 1.0, we'll set them in the dictionary
                # so the resulting pblums are available to b.compute_pblums()
                # we'll include logic later to exclude pblum, pblum_scale,
                # and pbfluxes from the user output
                for comp in valid_components:
                    pblums_scale[dataset][comp] = 1.0

            else:
                raise NotImplementedError("pblum_mode='{}' not supported".format(pblum_mode))


            for ds, ds_copy in pblum_scale_copy_ds.items():
                for component in valid_components:
                    pblums_scale[ds][component] = pblums_scale[ds_copy][component]

        # finally, we'll loop through the datasets again to apply the scales to
        # determine the relative pblums, compute pbfluxes, and expose/set whatever
        # was requested
        for dataset in datasets:
            pblum_mode = self.get_value(qualifier='pblum_mode', dataset=dataset, pblum_mode=kwargs.get('pblum_mode', None), default='absolute', **_skip_filter_checks)
            if pblum_mode == 'dataset-scaled':
                ds_scaled = True
            elif pblum_mode == 'dataset-coupled':
                coupled_to = self.get_value(qualifier='pblum_dataset', dataset=dataset, **_skip_filter_checks)
                if self.get_value(qualifier='pblum_mode', dataset=coupled_to, **_skip_filter_checks) == 'dataset-scaled':
                    ds_scaled = True
                else:
                    ds_scaled = False
            else:
                ds_scaled = False

            pbflux_this_dataset = 0.0
            for component in valid_components:
                if ds_scaled and model is not None:
                    flux_scale = self.get_value(qualifier='flux_scale', dataset=dataset, model=model, context='model', **_skip_filter_checks)
                    pblum_rel = pblums_abs[dataset][component] * flux_scale
                else:
                    pblum_rel = pblums_abs[dataset][component] * pblums_scale[dataset].get(component, 1.0)

                pblums_rel[dataset][component] = pblum_rel

                if set_value:
                    self.set_value(qualifier='pblum', component=component, dataset=dataset, context='dataset', value=pblum_rel*u.W, **_skip_filter_checks)

                if not ret_structured_dicts and component in components:
                    if pblum and (not ds_scaled or model is not None):
                        ret["{}@{}@{}".format('pblum', component, dataset)] = pblum_rel*u.W
                    if pblum_scale and (not ds_scaled or model is not None):
                        ret["{}@{}@{}".format('pblum_scale', component, dataset)] = pblums_scale[dataset].get(component, 1.0)
                    if pblum_abs:
                        ret["{}@{}@{}".format('pblum_abs', component, dataset)] = pblums_abs[dataset][component]*u.W

                if self.hierarchy.get_kind_of(component) != 'envelope':
                    # don't want to double count
                    pbflux_this_dataset += pblum_rel / (4*np.pi)

            if set_value:
                self.set_value(qualifier='pbflux', dataset=dataset, context='dataset', value=pbflux_this_dataset*u.W/u.m**2, **_skip_filter_checks)

            if pbflux and not ret_structured_dicts and (not ds_scaled or model is not None):
                ret["{}@{}".format('pbflux', dataset)] = pbflux_this_dataset*u.W/u.m**2
            elif ret_structured_dicts:
                pbfluxes[dataset] = pbflux_this_dataset

        if ret_structured_dicts:
            # this is an internal output used by run_compute, generally not requested by the user
            if system is not None:
                system.reset(force_recompute_instantaneous=True)
            return system, pblums_abs, pblums_scale, pblums_rel, pbfluxes

        # users will see the twig dictionaries with the exposed values based on
        # sent booleans
        return ret

    @send_if_client
    def add_compute(self, kind='phoebe', return_changes=False, **kwargs):
        """
        Add a set of computeoptions for a given backend to the bundle.
        The label (`compute`) can then be sent to <phoebe.frontend.bundle.Bundle.run_compute>.

        If not provided, `compute` will be created for you and can be
        accessed by the `compute` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        Available kinds can be found in <phoebe.parameters.compute> or by calling
        <phoebe.list_available_computes> and include:
        * <phoebe.parameters.compute.phoebe>
        * <phoebe.parameters.compute.legacy>
        * <phoebe.parameters.compute.ellc>
        * <phoebe.parameters.compute.jktebop>

        Arguments
        ----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts only default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.compute> module.
        * `compute` (string, optional): name of the newly-created compute options.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing set of compute options with the same `compute` tag.  If False,
            an error will be raised.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added

        Raises
        --------
        * NotImplementedError: if a required constraint is not implemented
        """
        func = _get_add_func(_compute, kind)

        # remove if None
        if kwargs.get('compute', False) is None:
            _ = kwargs.pop('compute')

        kwargs.setdefault('compute',
                          self._default_label(func.__name__,
                                              **{'context': 'compute',
                                                 'kind': func.__name__}))

        if kwargs.get('check_label', True):
            self._check_label(kwargs['compute'], allow_overwrite=kwargs.get('overwrite', False))

        sample_from = kwargs.pop('sample_from', None)
        params = func(**kwargs)
        # TODO: similar kwargs logic as in add_dataset (option to pass dict to
        # apply to different components this would be more complicated here if
        # allowing to also pass to different datasets

        metawargs = {'context': 'compute',
                     'kind': func.__name__,
                     'compute': kwargs['compute']}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_compute(compute=kwargs['compute'], during_overwrite=True)
            # check the label again, just in case kwargs['compute'] belongs to
            # something other than compute
            self.exclude(context=['model', 'solution'], **_skip_filter_checks)._check_label(kwargs['compute'], allow_overwrite=False)

        logger.info("adding {} '{}' compute to bundle".format(metawargs['kind'], metawargs['compute']))
        self._attach_params(params, **metawargs)

        if kind=='phoebe' and 'ntriangles' not in kwargs.keys():
            # the default for ntriangles in compute.py is 1500, we want 3000 for an envelope
            for envelope in self.hierarchy.get_envelopes():
                self.set_value(qualifier='ntriangles', compute=kwargs['compute'], component=envelope, value=3000, check_visible=False)

        ret_ps = self.get_compute(check_visible=False, check_default=False, **metawargs)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite'], warning_only=True, ps=ret_ps)

        ret_changes = []
        ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_computesamplefrom_selectparams(return_changes=return_changes)
        if sample_from is not None:
            ret_ps.set_value_all(qualifier='sample_from', value=sample_from, **_skip_filter_checks)

        ret_changes += self._handle_compute_selectparams(return_changes=return_changes)
        ret_changes += self._handle_compute_choiceparams(return_changes=return_changes)

        if kwargs.get('overwrite', False) and return_changes:
            ret_ps += overwrite_ps

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_compute(self, compute=None, **kwargs):
        """
        Filter in the 'compute' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `compute`: (string, optional, default=None): the name of the compute options
        * `**kwargs`: any other tags to do the filtering (excluding compute and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if compute is not None:
            kwargs['compute'] = compute
            if compute not in self.computes:
                raise ValueError("compute='{}' not found".format(compute))

        kwargs['context'] = 'compute'
        return self.filter(**kwargs)

    @send_if_client
    def remove_compute(self, compute, return_changes=False, **kwargs):
        """
        Remove a 'compute' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `compute` (string): the label of the compute options to be removed.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: context, compute.

        Returns
        -----------
        * ParameterSet of removed or changed parameters
        """
        kwargs['compute'] = compute
        kwargs['context'] = 'compute'
        ret_ps = self.remove_parameters_all(**kwargs)

        ret_changes = []
        if not kwargs.get('during_overwrite', False):
            ret_changes += self._handle_compute_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def remove_computes_all(self, return_changes=False):
        """
        Remove all compute options from the bundle.  To remove a single set
        of compute options see <phoebe.frontend.bundle.Bundle.remove_compute>.

        Arguments
        -----------
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for compute in self.computes:
            removed_ps += self.remove_compute(compute, return_changes=return_changes)
        return removed_ps

    @send_if_client
    def rename_compute(self, old_compute, new_compute,
                       overwrite=False, return_changes=False):
        """
        Change the label of compute options attached to the Bundle.

        Arguments
        ----------
        * `old_compute` (string): current label of the compute options (must exist)
        * `new_compute` (string): the desired new label of the compute options
            (must not yet exist unless `overwite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_compute` is forbidden or already exists.
        """
        # TODO: raise error if old_compute not found?
        self._rename_label('compute', old_compute, new_compute, overwrite)

        return self.filter(compute=new_compute)

    def _prepare_compute(self, compute, model, dataset, **kwargs):
        """
        """
        # protomesh and pbmesh were supported kwargs in 2.0.x but are no longer
        # so let's raise an error if they're passed here
        if 'protomesh' in kwargs.keys():
            raise ValueError("protomesh is no longer a valid option")
        if 'pbmesh' in kwargs.keys():
            raise ValueError("pbmesh is no longer a valid option")

        if model is None:
            model = 'latest'

        self._check_label(model, allow_overwrite=kwargs.get('overwrite', model=='latest'))

        overwrite_ps = None

        if model in self.models and kwargs.get('overwrite', model=='latest'):
            # NOTE: default (instead of detached_job=) is correct here
            if self.get_value(qualifier='detached_job', model=model, context='model', default='loaded') not in ['loaded', 'error', 'killed']:
                raise ValueError("model '{}' cannot be overwritten until it is complete and loaded.".format(model))
            if model=='latest':
                logger.warning("overwriting model: {}".format(model))
            else:
                logger.info("overwriting model: {}".format(model))

            do_create_fig_params = kwargs.pop('do_create_fig_params', False)

            overwrite_ps = self.remove_model(model=model, remove_figure_params=do_create_fig_params, during_overwrite=True)

            if model!='latest':
                # check the label again, just in case model belongs to something
                # other than model/figure
                self.exclude(context='figure')._check_label(model, allow_overwrite=False)

        else:
            do_create_fig_params = kwargs.pop('do_create_fig_params', True)


        # handle case where compute is not provided
        if compute is None:
            computes = self.get_compute(check_default=False, check_visible=False, **kwargs).computes
            if len(computes)==0:
                # NOTE: this doesn't take **kwargs since we want those to be
                # temporarily overriden as is the case when the compute options
                # are already attached
                self.add_compute()
                computes = self.computes
                # now len(computes) should be 1 and will trigger the next
                # if statement

            if len(computes)==1:
                compute = computes[0]
            elif len(computes)>1:
                raise ValueError("must provide label of compute options since more than one are attached.  The following were found: {}".format(self.computes))

        # handle the ability to send multiple compute options/backends - here
        # we'll just always send a list of compute options
        if isinstance(compute, str):
            computes = [compute]
        else:
            computes = compute

        # if interactive mode was ever off, let's make sure all constraints
        # have been run before running system checks or computing the model
        changed_params = self.run_delayed_constraints()

        if 'solution' in kwargs.keys():
            if kwargs.get('sample_from', None) is not None or np.any([len(p.get_value()) for p in self.filter(qualifier='sample_from', compute=computes, **_skip_filter_checks).to_list()]) :
                raise ValueError("cannot apply both solution and sample_from")
            else:
                logger.warning("applying passed solution ({}) to sample_from".format(kwargs.get('solution')))
                if 'sample_num' not in kwargs.keys() and not self.get_value(qualifier='adopt_distributions', solution=kwargs.get('solution'), default=False, **_skip_filter_checks):
                    logger.warning("defaulting sample_num=1 since adopt_distributions@{}=False".format(kwargs.get('solution')))
                    kwargs['sample_num'] = 1

                kwargs['sample_from'] = kwargs.pop('solution')

        # any kwargs that were used just to filter for get_compute should  be
        # removed so that they aren't passed on to all future get_value(...
        # **kwargs) calls
        for compute_ in computes:
            computes_ps = self.get_compute(compute=compute_, kind=kwargs.get('kind'), **_skip_filter_checks)
            for k in parameters._meta_fields_filter:
                if k in kwargs.keys():
                    dump = kwargs.pop(k)

        # we'll wait to here to run kwargs and system checks so that
        # add_compute is already called if necessary
        allowed_kwargs = ['skip_checks', 'jobid', 'overwrite', 'max_computations', 'in_export_script', 'out_fname', 'solution', 'progressbar']
        if conf.devel:
            allowed_kwargs += ['mesh_init_phi']
        self._kwargs_checks(kwargs, allowed_kwargs, ps=computes_ps)

        if not kwargs.get('skip_checks', False):
            report = self.run_checks_compute(compute=computes, allow_skip_constraints=False,
                                             raise_logger_warning=True, raise_error=True,
                                             run_checks_system=True,
                                             **kwargs)

        # let's first make sure that there is no duplication of enabled datasets
        datasets = []
        # compute_ so we don't write over compute which we need if detach=True
        for compute_ in computes:
            # TODO: filter by value instead of if statement once implemented
            if dataset is None:
                for enabled_param in self.filter(qualifier='enabled',
                                                 compute=compute_,
                                                 context='compute',
                                                 check_visible=False).to_list():
                    if enabled_param.feature is None and enabled_param.get_value():
                        item = (enabled_param.dataset, enabled_param.component)
                        if item in datasets:
                            raise ValueError("dataset {}@{} is enabled in multiple compute options".format(item[0], item[1]))
                        datasets.append(item)
            elif isinstance(dataset, list) or isinstance(dataset, tuple) or isinstance(dataset, str):
                datasets += self.filter(dataset=dataset, context='dataset', **_skip_filter_checks).datasets
            elif isinstance(dataset, dict):
                datasets += self.filter(dataset=dataset.get(compute_, []), context='dataset', **_skip_filter_checks).datasets


        if not len(datasets):
            raise ValueError("cannot run forward model without any enabled datasets")

        return model, computes, datasets, do_create_fig_params, changed_params, overwrite_ps, kwargs


    def _write_export_compute_script(self, script_fname, out_fname, compute, model, dataset, do_create_fig_params, import_from_older, log_level, kwargs):
        """
        """
        f = open(script_fname, 'w')
        f.write("import os; os.environ['PHOEBE_ENABLE_PLOTTING'] = 'FALSE'; os.environ['PHOEBE_ENABLE_SYMPY'] = 'FALSE';\n")
        f.write("import phoebe; import json\n")
        if log_level is not None:
            f.write("phoebe.logger('{}')\n".format(log_level))
        # TODO: can we skip other models
        # or datasets (except times and only for run_compute but not run_solver)
        exclude_contexts = ['model', 'figure', 'constraint', 'solver']
        sample_from = self.get_value(qualifier='sample_from', compute=compute, sample_from=kwargs.get('sample_from', None), default=[], expand=True)
        exclude_distributions = [dist for dist in self.distributions if dist not in sample_from]
        exclude_solutions = [sol for sol in self.solutions if sol not in sample_from]
        # we need to include uniqueids if needing to apply the solution during sample_from
        incl_uniqueid = len(exclude_solutions) != len(self.solutions)
        f.write("bdict = json.loads(\"\"\"{}\"\"\", object_pairs_hook=phoebe.utils.parse_json)\n".format(json.dumps(self.exclude(context=exclude_contexts, **_skip_filter_checks).exclude(distribution=exclude_distributions, **_skip_filter_checks).exclude(solution=exclude_solutions, **_skip_filter_checks).to_json(incl_uniqueid=incl_uniqueid, exclude=['description', 'advanced', 'readonly', 'copy_for', 'latexfmt', 'labels_latex', 'label_latex']))))
        f.write("b = phoebe.open(bdict, import_from_older={})\n".format(import_from_older))
        # TODO: make sure this works with multiple computes
        compute_kwargs = list(kwargs.items())+[('compute', compute), ('model', str(model)), ('dataset', dataset), ('do_create_fig_params', do_create_fig_params)]
        compute_kwargs_string = ','.join(["{}={}".format(k,"\'{}\'".format(str(v)) if isinstance(v, str) else v) for k,v in compute_kwargs])
        # as the return from run_compute just does a filter on model=model,
        # model_ps here should include any created figure parameters

        if out_fname is not None:
            f.write("model_ps = b.run_compute(out_fname='{}', in_export_script=True, {})\n".format(out_fname, compute_kwargs_string))
            f.write("b.filter(context='model', model=model_ps.model, check_visible=False).save('{}', incl_uniqueid=True)\n".format(out_fname))
        else:
            f.write("import sys\n")
            f.write("model_ps = b.run_compute(out_fname=sys.argv[0]+'.out', in_export_script=True, {})\n".format(compute_kwargs_string))
            f.write("b.filter(context='model', model=model_ps.model, check_visible=False).save(sys.argv[0]+'.out', incl_uniqueid=True)\n")
            out_fname = script_fname+'.out'

        f.write("\n# NOTE: this script only includes parameters needed to call the requested run_compute, edit manually with caution!\n")
        f.close()

        return script_fname, out_fname

    def export_compute(self, script_fname, out_fname=None,
                       compute=None, model=None, dataset=None,
                       pause=False, log_level=None,
                       import_from_older=False, **kwargs):
        """
        Export a script to call run_compute externally (in a different thread
        or on a different machine).  To automatically detach to a different
        thread and load the results, see <phoebe.frontend.bundle.Bundle.run_compute>
        with `detach=True`.

        After running the resulting `script_fname`, `out_fname` will be created,
        which will contain a ParameterSet of the model parameters.  To attach
        that model to this bundle, see <phoebe.frontend.bundle.Bundle.import_model>.

        Arguments
        ------------
        * `script_fname` (string): the filename of the python script to be generated.
        * `out_fname` (string, optional): the filename of the output file that `script_fname`
            will write when executed.  Once executed, pass this filename to
            <phoebe.frontend.bundle.Bundle.import_model> to load the resulting
            model.  If not provided, the script will automatically export
            to `script_fname`.out (where the filename is determined at runtime,
            so if you rename the script exported here, the resulting filename
            will reflect that change and be appended with '.out').
        * `compute` (string, optional): name of the compute options to use.
            If not provided or None, run_compute will use an existing set of
            attached compute options if only 1 exists.  If more than 1 exist,
            then compute becomes a required argument.  If no compute options
            exist, then this will use default options and create and attach
            a new set of compute options with a default label.
        * `model` (string, optional): name of the resulting model.  If not
            provided this will default to 'latest'.  NOTE: existing models
            with the same name will be overwritten depending on the value
            of `overwrite` (see below).   See also
            <phoebe.frontend.bundle.Bundle.rename_model> to rename a model after
            creation.
        * `dataset` (list, dict, or string, optional, default=None): filter for which datasets
            should be computed.  If provided as a dictionary, keys should be compute
            labels provided in `compute`.  If None, will use the `enabled` parameters in the
            `compute` options.  If not None, will override all `enabled` parameters.
        * `pause` (bool, optional, default=False): whether to raise an input
            with instructions for running the exported script and calling
            <phoebe.frontend.bundle.Bundle.import_model>.  Particularly
            useful if running in an interactive notebook or a script.
        * `log_level` (string, optional, default=None): `clevel` to set in the
            logger in the exported script.  See <phoebe.logger>.
        * `import_from_older` (boolean, optional, default=False): whether to allow
            the script to run on a newer version of PHOEBE.  If True and executing
            the outputed script (`script_fname`) on a newer version of PHOEBE,
            the bundle will attempt to migrate to the newer version.  If False,
            an error will be raised when attempting to run the script.  See
            also: <phoebe.frontend.bundle.Bundle.open>.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_compute> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `**kwargs`:: any values in the compute options to temporarily
            override for this single compute run (parameter values will revert
            after run_compute is finished).

        Returns
        -----------
        * `script_fname`, `out_fname`.  Where running `script_fname` will result
          in the model being written to `out_fname`.

        """
        model, computes, datasets, do_create_fig_params, changed_params, overwrite_ps, kwargs = self._prepare_compute(compute, model, dataset, **kwargs)
        script_fname, out_fname = self._write_export_compute_script(script_fname, out_fname, computes, model, dataset, do_create_fig_params, import_from_older, log_level, kwargs)

        if pause:
            input("* optional:  call b.save(...) to save the bundle to disk, you can then safely close the active python session and recover the bundle with phoebe.load(...)\n"+
                  "* run {} (within mpirun or on an external machine, if desired)\n".format(script_fname)+
                  "* once completed, copy {} to this directory, if necessary\n".format(out_fname)+
                  "* press enter to exit this pause\n"+
                  "* call b.import_model('{}')\n".format(out_fname)+
                  "\n(press enter to continue)")

        return script_fname, out_fname

    def _run_compute_changes(self, ret_ps, return_changes=False, do_create_fig_params=False, removed=False, during_overwrite=False, **kwargs):
        ret_changes = []

        # Figure options for this model
        # Since auto_add_figure is applied to the DATASET, we will always add model-dependent options
        if do_create_fig_params and not removed and ret_ps.model not in self.filter(context='figure', **_skip_filter_checks).models:
            fig_params = _figure._run_compute(self, **kwargs)

            fig_metawargs = {'context': 'figure',
                             'model': ret_ps.model}
            self._attach_params(fig_params, check_copy_for=False, **fig_metawargs)
            ret_changes += fig_params.to_list()

        ret_changes += self._handle_model_selectparams(return_changes=return_changes)
        ret_changes += self._handle_meshcolor_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_figure_time_source_params(return_changes=return_changes)
        return ret_changes

    @send_if_client
    def run_compute(self, compute=None, model=None, solver=None,
                    detach=False,
                    dataset=None, times=None,
                    return_changes=False, **kwargs):
        """
        Run a forward model of the system on the enabled dataset(s) using
        a specified set of compute options.

        To attach and set custom values for compute options, including choosing
        which backend to use, see:
        * <phoebe.frontend.bundle.Bundle.add_compute>

        To define the dataset types and times at which the model should be
        computed see:
        * <phoebe.frontend.bundle.Bundle.add_dataset>

        To disable or enable existing datasets see:
        * <phoebe.frontend.bundle.Bundle.enable_dataset>
        * <phoebe.frontend.bundle.Bundle.disable_dataset>

        See also:
        * <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * <phoebe.mpi_on>
        * <phoebe.mpi_off>

        Arguments
        ------------
        * `compute` (string, optional): name of the compute options to use.
            If not provided or None, run_compute will use an existing set of
            attached compute options if only 1 exists.  If more than 1 exist,
            then compute becomes a required argument.  If no compute options
            exist, then this will use default options and create and attach
            a new set of compute options with a default label.
        * `model` (string, optional): name of the resulting model.  If not
            provided this will default to 'latest'.  NOTE: existing models
            with the same name will be overwritten depending on the value
            of `overwrite` (see below).   See also
            <phoebe.frontend.bundle.Bundle.rename_model> to rename a model after
            creation.
        * `solver` (string, optional): name of the solver options to use to
            extract compute options and use `solver_times`
            (see <phoebe.frontend.bundle.Bundle.parse_solver_times>) unless
            `times` is also passed.  `compute` must be None (not passed) or an
            error will be raised.
        * `detach` (bool, optional, default=False, EXPERIMENTAL):
            whether to detach from the computation run,
            or wait for computations to complete.  If detach is True, see
            <phoebe.frontend.bundle.Bundle.get_model> and
            <phoebe.parameters.JobParameter>
            for details on how to check the job status and retrieve the results.
        * `dataset` (list, dict, or string, optional, default=None): filter for which datasets
            should be computed.  If provided as a dictionary, keys should be compute
            labels provided in `compute`.  If None, will use the `enabled` parameters in the
            `compute` options.  If not None, will override all `enabled` parameters.
        * `times` (list, optional, EXPERIMENTAL): override the times at which to compute the model.
            NOTE: this only (temporarily) replaces the time array for datasets
            with times provided (ie empty time arrays are still ignored).  So if
            you attach a rv to a single component, the model will still only
            compute for that single component.  ALSO NOTE: this option is ignored
            if `detach=True` (at least for now).
        * `overwrite` (boolean, optional, default=model=='latest'): whether to overwrite
            an existing model with the same `model` tag.  If False,
            an error will be raised.  This defaults to True if `model` is not provided
            or is 'latest', otherwise it will default to False.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_compute> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `max_computations` (int, optional, default=None): maximum
            number of computations to allow.  If more are detected, an error
            will be raised before the backend begins computations.
        * `progressbar` (bool, optional): whether to show a progressbar.  If not
            provided or none, will default to <phoebe.progressbars_on> or
            <phoebe.progressbars_off>.  Progressbars require `tqdm` to be installed
            (will silently ignore if not installed).
        * `**kwargs`:: any values in the compute options to temporarily
            override for this single compute run (parameter values will revert
            after run_compute is finished)

        Returns
        ----------
        * a <phoebe.parameters.ParameterSet> of the newly-created model
            containing the synthetic data.

        Raises
        --------
        * ValueError: if passing `protomesh` or `pbmesh` as these were removed in 2.1
        * ValueError: if `compute` must be provided but is not.
        * ValueError: if the system fails to pass checks.  See also
            <phoebe.frontend.bundle.Bundle.run_checks_compute>
        * ValueError: if any given dataset is enabled in more than one set of
            compute options sent to run_compute.
        """
        # NOTE: if we're already in client mode, we'll never get here in the client
        # there detach is handled slightly differently (see parameters._send_if_client)
        if isinstance(detach, str):
            # then we want to temporarily go in to client mode
            raise NotImplementedError("detach currently must be a bool")
            self.as_client(as_client=detach)
            ret_ = self.run_compute(compute=compute, model=model, solver=solver, dataset=dataset, times=times, return_changes=return_changes, **kwargs)
            self.as_client(as_client=False)
            return _return_ps(self, ret_)

        if solver is not None and compute is not None:
            raise ValueError("cannot provide both solver and compute")

        if solver is not None:
            if not kwargs.get('skip_checks', False):
                report = self.run_checks_solver(solver=solver, run_checks_compute=False,
                                                allow_skip_constraints=False,
                                                raise_logger_warning=True, raise_error=True)

            solver_ps = self.get_solver(solver=solver)
            if 'compute' not in solver_ps.qualifiers:
                raise ValueError("solver='{}' does not contain compute parameter, pass compute instead of solver".format(solver))
            compute = solver_ps.get_value(qualifier='compute')
            if times is None:
                # then override with the parsed solver times (this will be a dictionary
                # with datasets as keys and arrays of times as values)
                times = self.parse_solver_times(return_as_dict=True, set_compute_times=False)

        if isinstance(times, float) or isinstance(times, int):
            times = [times]

        if isinstance(times, dict):
            # make sure keys are datasets
            for k,v in times.items():
                if k not in self.datasets:
                    raise ValueError("keys in times dictionary must be valid datasets")
                if isinstance(v, float) or isinstance(v, int):
                    times[k] = [v]

        # NOTE: _prepare_compute calls run_checks_compute and will handle raising
        # any necessary errors
        model, computes, datasets, do_create_fig_params, changed_params, overwrite_ps, kwargs = self._prepare_compute(compute, model, dataset, **kwargs)
        _ = kwargs.pop('do_create_fig_params', None)

        kwargs.setdefault('progressbar', conf.progressbars)

        # now if we're supposed to detach we'll just prepare the job for submission
        # either in another subprocess or through some queuing system
        if detach and mpi.within_mpirun:
            logger.warning("cannot detach when within mpirun, ignoring")
            detach = False

        if (detach or mpi.enabled) and not mpi.within_mpirun:
            if detach:
                logger.warning("detach support is EXPERIMENTAL")

            if times is not None:
                # TODO: support overriding times with detached - issue here is
                # that it isn't necessarilly trivially to send this array
                # through the script.  May need to convert to list first to
                # avoid needing to import numpy?
                logger.warning("overriding time is not supported within detach - ignoring")

            if kwargs.get('max_computations', None) is not None:
                # then we need to estimate computations in advance so we can
                # raise an error immediately
                logger.info("estimating number of computations to ensure not over max_computations={}".format(kwargs['max_computations']))
                for compute in computes:
                    computeparams = self.get_compute(compute=compute)

                    if not computeparams.kind:
                        raise KeyError("could not recognize backend from compute: {}".format(compute))
                    compute_class = getattr(backends, '{}Backend'.format(computeparams.kind.title()))
                    out = compute_class().get_packet_and_syns(self, compute, times=times, **kwargs)

            # we'll track everything through the model name as well as
            # a random string, to avoid any conflicts
            jobid = kwargs.get('jobid', parameters._uniqueid())

            # we'll build a python script that can replicate this bundle as it
            # is now, run compute, and then save the resulting model
            script_fname = "_{}.py".format(jobid)
            out_fname = "_{}.out".format(jobid)
            err_fname = "_{}.err".format(jobid)
            script_fname, out_fname = self._write_export_compute_script(script_fname, out_fname, compute, model, dataset, do_create_fig_params, False, None, kwargs)

            script_fname = os.path.abspath(script_fname)
            cmd = mpi.detach_cmd.format(script_fname)
            # TODO: would be nice to catch errors caused by the detached script...
            # but that would probably need to be the responsibility of the
            # jobparam to return a failed status and message.
            # Unfortunately right now an error just results in the job hanging.
            f = open(err_fname, 'w')
            subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=f)
            f.close()

            # create model parameter and attach (and then return that instead of None)
            job_param = JobParameter(self,
                                     location=os.path.dirname(script_fname),
                                     status_method='exists',
                                     retrieve_method='local',
                                     uniqueid=jobid)

            metawargs = {'context': 'model', 'model': model}
            self._attach_params([job_param], check_copy_for=False, **metawargs)

            if isinstance(detach, str):
                self.save(detach)

            if not detach:
                return job_param.attach()
            else:
                logger.info("detaching from run_compute.  Call get_parameter(model='{}').attach() to re-attach".format(model))

            # TODO: make sure the figureparams are returned when attaching

            # return self.get_model(model)
            ret_changes = []
            ret_changes += self._handle_model_selectparams(return_changes=return_changes)
            # return self.filter(model=model, check_visible=False, check_default=False)

            if kwargs.get('overwrite', model=='latest') and return_changes and overwrite_ps is not None:
                return ParameterSet([job_param]) + overwrite_ps + ret_changes

            if return_changes:
                return ParameterSet([job_param]) + ret_changes
            return job_param

        # from here on, we do not need to detach
        # temporarily disable interactive_checks, check_default, and check_visible
        conf_interactive_checks = conf.interactive_checks
        if conf_interactive_checks:
            logger.debug("temporarily disabling interactive_checks")
            conf._interactive_checks = False

        conf_interactive_constraints = conf.interactive_constraints
        if conf_interactive_constraints:
            logger.debug("temporarily disabling interactive_constraints")
            conf._interactive_constraints = False

        def restore_conf():
            # restore user-set interactive checks
            if conf_interactive_checks:
                logger.debug("restoring interactive_checks={}".format(conf_interactive_checks))
                conf._interactive_checks = conf_interactive_checks

            if conf_interactive_constraints:
                logger.debug("restoring interactive_constraints={}".format(conf_interactive_constraints))
                conf._interactive_constraints = conf_interactive_constraints
                self.run_delayed_constraints()

        ret_changes = []

        try:
            for compute in computes:

                computeparams = self.get_compute(compute=compute)

                if not computeparams.kind:
                    raise KeyError("could not recognize backend from compute: {}".format(compute))

                metawargs = {'compute': compute, 'model': model, 'context': 'model'}  # dataset, component, etc will be set by the compute_func

                # TODO: consolidate this into _prepare_compute and simplify the logic
                # make sure to test again compute_multiple tutorial
                dataset_this_compute = datasets
                # remove any that are disabled
                if dataset_this_compute is None:
                    dataset_this_compute = computeparams.filter(qualifier='enabled', value=True, **_skip_filter_checks).datasets
                else:
                    dataset_this_compute = [ds[0] if isinstance(ds, tuple) else ds for ds in dataset_this_compute if computeparams.get_value(qualifier='enabled', dataset=ds[0] if isinstance(ds, tuple) else ds, default=False, **_skip_filter_checks)]

                # if sampling is enabled then we need to pass things off now
                # to the sampler.  The sampler will then make handle parallelization
                # and per-sample calls to run_compute.
                sample_from = computeparams.get_value(qualifier='sample_from', expand=True, sample_from=kwargs.pop('sample_from', None), **_skip_filter_checks)
                if len(sample_from):
                    params = backends.SampleOverModel().run(self, computeparams.compute,
                                                            dataset=dataset_this_compute,
                                                            times=times,
                                                            sample_from=sample_from,
                                                            **kwargs)

                    self._attach_params(params, check_copy_for=False, **metawargs)

                    # continue to the next iteration of the computes for-loop.
                    # Any dataset-scaling etc, will be handled within each
                    # individual model run within the sampler.
                    continue

                # we now need to handle any computations of ld_coeffs, pblums, l3s, etc
                # TODO: skip lookups for phoebe, skip non-supported ld_func for photodynam, etc
                # TODO: have this return a dictionary like pblums/l3s that we can pass on to the backend?

                # we need to check both for enabled but also passed via dataset kwarg
                ds_kinds_enabled = self.filter(dataset=dataset_this_compute, context='dataset', **_skip_filter_checks).kinds
                if 'lc' in ds_kinds_enabled or 'rv' in ds_kinds_enabled or 'lp' in ds_kinds_enabled:
                    logger.info("run_compute: computing necessary ld_coeffs, pblums, l3s")
                    self.compute_ld_coeffs(compute=compute, skip_checks=True, set_value=True, **{k:v for k,v in kwargs.items() if k in computeparams.qualifiers})
                    # NOTE that if pblum_method != 'phoebe', then system will be None
                    # otherwise the system will be create which we can pass on to the backend
                    # the phoebe backend can then skip initializing the system at least on the master proc
                    # (workers will need to recreate the mesh)
                    system, pblums_abs, pblums_scale, pblums_rel, pbfluxes = self.compute_pblums(compute=compute, ret_structured_dicts=True, skip_checks=True, **{k:v for k,v in kwargs.items() if k in computeparams.qualifiers})
                    l3s = self.compute_l3s(compute=compute, use_pbfluxes=pbfluxes, ret_structured_dicts=True, skip_checks=True, skip_compute_ld_coeffs=True, **{k:v for k,v in kwargs.items() if k in computeparams.qualifiers})
                else:
                    system = None
                    pblums_scale = {}
                    pblums_rel = {}
                    l3s = {}

                logger.info("run_compute: calling {} backend to create '{}' model".format(computeparams.kind, model))
                if mpi.within_mpirun:
                    logger.info("run_compute: within mpirun with nprocs={}".format(mpi.nprocs))
                compute_class = getattr(backends, '{}Backend'.format(computeparams.kind.title()))
                if computeparams.kind == 'phoebe':
                    kwargs['system'] = system
                    kwargs['pblums_scale'] = pblums_scale
                elif computeparams.kind in ['legacy', 'jktebop', 'ellc']:
                    # legacy uses pblums directly
                    # jktebop, ellc use pblums for sbratio if decoupled, otherwise will ignore and use teffs
                    kwargs['pblums'] = pblums_rel

                ml_params = compute_class().run(self, computeparams.compute,
                                                dataset=dataset_this_compute,
                                                times=times,
                                                **kwargs)

                ml_addl_params = []

                # ml_params contain the raw synthetic model from the respective
                # compute backend.  Now we need to do any post-processing that
                # can act on any results (exposure-time, flux-scaling, GPs, etc)

                # average over any exposure times before attaching parameters
                for ds in ml_params.datasets:
                    # not all dataset-types currently support exposure times.
                    # Once they do, this ugly if statement can be removed
                    if len(self.filter(dataset=ds, qualifier='exptime', **_skip_filter_checks)):
                        exptime = self.get_value(qualifier='exptime', dataset=ds, context='dataset', unit=u.d, **_skip_filter_checks)
                        if exptime > 0:
                            logger.info("handling fti for dataset='{}'".format(ds))
                            if self.get_value(qualifier='fti_method', dataset=ds, compute=compute, context='compute', fti_method=kwargs.get('fti_method', None), **_skip_filter_checks)=='oversample':
                                times_ds = self.get_value(qualifier='compute_times', dataset=ds, context='dataset', **_skip_filter_checks)
                                if not len(times_ds):
                                    times_ds = self.get_value(qualifier='times', dataset=ds, context='dataset', **_skip_filter_checks)
                                # exptime = self.get_value(qualifier='exptime', dataset=ds, context='dataset', unit=u.d)
                                fti_oversample = self.get_value(qualifier='fti_oversample', dataset=ds, compute=compute, context='compute', fti_oversample=kwargs.get('fti_oversample', None), **_skip_filter_checks)
                                # NOTE: this is hardcoded for LCs which is the
                                # only dataset that currently supports oversampling,
                                # but this will need to be generalized if/when
                                # we expand that support to other dataset kinds
                                fluxes = np.zeros(times_ds.shape)

                                # the oversampled times and fluxes will be
                                # sorted according to times this may cause
                                # exposures to "overlap" each other, so we'll
                                # later need to determine which times (and
                                # therefore fluxes) belong to which datapoint
                                times_oversampled_sorted = ml_params.get_value(qualifier='times', dataset=ds, **_skip_filter_checks)
                                fluxes_oversampled = ml_params.get_value(qualifier='fluxes', dataset=ds, **_skip_filter_checks)

                                for i,t in enumerate(times_ds):
                                    # rebuild the unsorted oversampled times - see backends._extract_from_bundle_by_time
                                    # TODO: try to optimize this by having these indices returned by the backend itself
                                    times_oversampled_this = np.linspace(t-exptime/2., t+exptime/2., fti_oversample)
                                    sample_inds = np.searchsorted(times_oversampled_sorted, times_oversampled_this)

                                    fluxes[i] = np.mean(fluxes_oversampled[sample_inds])

                                ml_params.set_value(qualifier='times', dataset=ds, value=times_ds, ignore_readonly=True, **_skip_filter_checks)
                                ml_params.set_value(qualifier='fluxes', dataset=ds, value=fluxes, ignore_readonly=True, **_skip_filter_checks)

                # handle scaling to absolute fluxes as necessary for alternate backends
                # NOTE: this must happen BEFORE dataset-scaling as that scaling assumes absolute fluxes
                for flux_param in ml_params.filter(qualifier='fluxes', kind='lc', **_skip_filter_checks).to_list():
                    fluxes = flux_param.get_value(unit=u.W/u.m**2)
                    if computeparams.kind not in ['phoebe', 'legacy']:
                        # then we need to scale the "normalized" fluxes to pbflux first
                        fluxes *= pbfluxes.get(flux_param.dataset)
                        # otherwise fluxes are already correctly scaled by passing
                        # relative pblums or pblums_scale to the respective backend

                        flux_param.set_value(fluxes, ignore_readonly=True)

                # handle flux scaling for any pblum_mode == 'dataset-scaled'
                # or for any dataset in which pblum_mode == 'dataset-coupled' and pblum_dataset points to a 'dataset-scaled' dataset
                datasets_dsscaled = []
                coupled_datasets = self.filter(qualifier='pblum_mode', dataset=ml_params.datasets, value='dataset-coupled', **_skip_filter_checks).datasets
                for pblum_mode_param in self.filter(qualifier='pblum_mode', dataset=ml_params.datasets, value='dataset-scaled', **_skip_filter_checks).to_list():
                    this_dsscale_datasets = [pblum_mode_param.dataset] + self.filter(qualifier='pblum_dataset', dataset=coupled_datasets, value=pblum_mode_param.dataset, **_skip_filter_checks).datasets
                    # keep track of all datasets that are scaled so we don't do distance/l3 corrections later
                    datasets_dsscaled += this_dsscale_datasets
                    logger.info("rescaling fluxes to data for dataset={}".format(this_dsscale_datasets))

                    ds_fluxess = np.array([])
                    ds_sigmass = np.array([])
                    l3_fluxes = np.array([])
                    l3_fracs = np.array([])
                    l3_pblum_abs_sums = np.array([])
                    model_fluxess_interp = np.array([])

                    for dataset in this_dsscale_datasets:
                        ds_obs = self.get_dataset(dataset, **_skip_filter_checks)
                        ds_times = ds_obs.get_value(qualifier='times')

                        l3_mode = ds_obs.get_value(qualifier='l3_mode', **_skip_filter_checks)
                        if l3_mode == 'flux':
                            l3_flux = ds_obs.get_value(qualifier='l3', unit=u.W/u.m**2, **_skip_filter_checks)
                            l3_fluxes = np.append(l3_fluxes, np.full_like(ds_times, fill_value=l3_flux))
                            l3_fracs = np.append(l3_fracs, np.zeros_like(ds_times))
                            l3_pblum_abs_sums = np.append(l3_pblum_abs_sums, np.zeros_like(ds_times))
                        else:
                            l3_frac = ds_obs.get_value(qualifier='l3_frac', **_skip_filter_checks)
                            l3_fluxes = np.append(l3_fluxes, np.zeros_like(ds_times))
                            l3_fracs = np.append(l3_fracs, np.full_like(ds_times, fill_value=l3_frac))
                            l3_pblum_abs_sums = np.append(l3_pblum_abs_sums, np.full_like(ds_times, fill_value=np.sum(list(pblums_abs.get(dataset).values()))))

                        ds_fluxes = ds_obs.get_value(qualifier='fluxes', unit=u.W/u.m**2, **_skip_filter_checks)
                        ds_fluxess = np.append(ds_fluxess, ds_fluxes)
                        ds_sigmas = ds_obs.get_value(qualifier='sigmas', **_skip_filter_checks)
                        if len(ds_sigmas):
                            ds_sigmass = np.append(ds_sigmass, ds_sigmas)
                        else:
                            sigma_est = 0.001*ds_fluxes.mean()
                            logger.warning("dataset-scaling: adopting sigmas={} for dataset='{}'".format(sigma_est, dataset))
                            ds_sigmass = np.append(ds_sigmass, sigma_est*np.ones(len(ds_fluxes)))

                        ml_ds = ml_params.filter(dataset=dataset, **_skip_filter_checks)
                        model_fluxes_interp = ml_ds.get_parameter(qualifier='fluxes', dataset=dataset, **_skip_filter_checks).interp_value(times=ds_times, parent_ps=ml_ds, bundle=self, consider_gaussian_process=False)
                        model_fluxess_interp = np.append(model_fluxess_interp, model_fluxes_interp)

                    scale_factor_approx = np.median(ds_fluxess / model_fluxess_interp)

                    def _scale_fluxes(fluxes, scale_factor, l3_frac, l3_pblum_abs_sum, l3_flux):
                        # note: l3_frac or l3_flux will be zero, based on which is provided
                        return scale_factor * (fluxes + l3_frac/(1-l3_frac) * l3_pblum_abs_sum) + l3_flux

                    def _scale_fluxes_cfit(fluxes, scale_factor):
                        # use values in this namespace rather than passing directly
                        return _scale_fluxes(fluxes, scale_factor, l3_fracs, l3_pblum_abs_sums, l3_fluxes)

                    logger.debug("calling curve_fit with estimated scale_factor={}".format(scale_factor_approx))
                    popt, pcov = cfit(_scale_fluxes_cfit, model_fluxess_interp, ds_fluxess, p0=(scale_factor_approx), sigma=ds_sigmass)
                    scale_factor = popt[0]

                    for flux_param in ml_params.filter(qualifier='fluxes', dataset=this_dsscale_datasets, **_skip_filter_checks).to_list():
                        logger.debug("applying scale_factor={} to fluxes@{}".format(scale_factor, flux_param.dataset))

                        ds_obs = self.get_dataset(dataset=flux_param.dataset, **_skip_filter_checks)
                        l3_mode = ds_obs.get_value(qualifier='l3_mode', **_skip_filter_checks)
                        # this time we can pass floats instead of arrays since only
                        # one will apply to this single dataset
                        if l3_mode == 'flux':
                            l3_flux = ds_obs.get_value(qualifier='l3', unit=u.W/u.m**2, **_skip_filter_checks)
                            l3_frac = 0.0
                            l3_pblum_abs_sum = 0.0
                        else:
                            l3_frac = ds_obs.get_value(qualifier='l3_frac', **_skip_filter_checks)
                            l3_flux = 0.0
                            l3_pblum_abs_sum = np.sum(list(pblums_abs.get(dataset).values()))

                        syn_fluxes = _scale_fluxes(flux_param.get_value(unit=u.W/u.m**2), scale_factor, l3_frac, l3_pblum_abs_sum, l3_flux)

                        flux_param.set_value(qualifier='fluxes', value=syn_fluxes, ignore_readonly=True)

                        ml_addl_params += [FloatParameter(qualifier='flux_scale', dataset=dataset, value=scale_factor, readonly=True, default_unit=u.dimensionless_unscaled, description='scaling applied to fluxes (intensities/luminosities) due to dataset-scaling')]

                        for mesh_param in ml_params.filter(kind='mesh', **_skip_filter_checks).to_list():
                            if mesh_param.qualifier in ['intensities', 'abs_intensities', 'normal_intensities', 'abs_normal_intensities', 'pblum_ext']:
                                logger.debug("applying scale_factor={} to {} parameter in mesh".format(scale_factor, mesh_param.qualifier))
                                mesh_param.set_value(mesh_param.get_value()*scale_factor, ignore_readonly=True)

                # handle flux scaling based on distance and l3
                # NOTE: this must happen AFTER dataset scaling
                distance = self.get_value(qualifier='distance', context='system', unit=u.m, **_skip_filter_checks)
                for flux_param in ml_params.filter(qualifier='fluxes', kind='lc', **_skip_filter_checks).to_list():
                    dataset = flux_param.dataset
                    if dataset in datasets_dsscaled:
                        # then we already handle the scaling (including l3)
                        # above in dataset-scaling
                        continue

                    fluxes = flux_param.get_value(unit=u.W/u.m**2)
                    fluxes = fluxes/distance**2 + l3s.get(dataset)

                    flux_param.set_value(fluxes, ignore_readonly=True)

                # handle vgamma and rv_offset
                vgamma = self.get_value(qualifier='vgamma', context='system', unit=u.km/u.s, **_skip_filter_checks)
                for rv_param in ml_params.filter(qualifier='rvs', kind='rv', **_skip_filter_checks).to_list():
                    dataset = rv_param.dataset
                    component = rv_param.component

                    rv_offset = self.get_value(qualifier='rv_offset', dataset=dataset, component=component, context='dataset', default=0.0, unit=u.km/u.s, **_skip_filter_checks)

                    if computeparams.kind in ['phoebe', 'legacy']:
                        # we'll use native vgamma so ltte, etc, can be handled appropriately
                        rv_param.set_value(rv_param.get_value(unit=u.km/u.s)+rv_offset, ignore_readonly=True)
                    else:
                        rv_param.set_value(rv_param.get_value(unit=u.km/u.s)+vgamma+rv_offset, ignore_readonly=True)

                ml_addl_params += [StringParameter(qualifier='comments', value=kwargs.get('comments', computeparams.get_value(qualifier='comments', default='', **_skip_filter_checks)), description='User-provided comments for this model.  Feel free to place any notes here.')]
                self._attach_params(ml_params+ml_addl_params, check_copy_for=False, **metawargs)

                model_ps = self.get_model(model=model, **_skip_filter_checks)

                # add any GPs (gaussian processes) to the returned model
                # NOTE: this has to happen after _attach_params as it uses
                # several bundle methods that act on the model
                enabled_features = self.filter(qualifier='enabled', compute=compute, context='compute', value=True, **_skip_filter_checks).features

                for ds in model_ps.datasets:
                    gp_features = self.filter(feature=enabled_features, dataset=ds, kind='gaussian_process', **_skip_filter_checks).features
                    if len(gp_features):
                        # NOTE: this is already in run_checks_compute, so this error
                        # should never be raised
                        if not _use_celerite:
                            raise ImportError("gaussian processes require celerite to be installed")

                        # NOTE: only those exposed in feature.gaussian_process
                        # will be available to the user (we don't allow jitter, for example)
                        gp_kernel_classes = {'matern32': _celerite.terms.Matern32Term,
                                              'sho': _celerite.terms.SHOTerm,
                                              'jitter': _celerite.terms.JitterTerm}

                        # build the celerite GP object from the enabled GP features attached to this dataset
                        gp_kernels = []
                        for gp in gp_features:
                            gp_ps = self.filter(feature=gp, context='feature', **_skip_filter_checks)
                            kind = gp_ps.get_value(qualifier='kernel', **_skip_filter_checks)

                            kwargs = {p.qualifier: p.value for p in gp_ps.exclude(qualifier=['kernel', 'enabled']).to_list() if p.is_visible}
                            gp_kernels.append(gp_kernel_classes.get(kind)(**kwargs))

                        if len(gp_kernels) == 1:
                            gp_kernel = _celerite.GP(gp_kernels[0])
                        else:
                            gp_kernel = _celerite.GP(_celerite.terms.TermSum(*gp_kernels))


                        ds_ps = self.get_dataset(dataset=ds, **_skip_filter_checks)
                        xqualifier = {'lp': 'wavelength'}.get(ds_ps.kind, 'times')
                        yqualifier = {'lp': 'flux_densities', 'rv': 'rvs', 'lc': 'fluxes'}.get(ds_ps.kind)
                        # we'll loop over components (for RVs or LPs, for example)
                        if ds_ps.kind in ['lc']:
                            ds_comps = [None]
                        else:
                            ds_comps = ds_ps.filter(qualifier=xqualifier, check_visible=True).components
                        for ds_comp in ds_comps:
                            ds_x = ds_ps.get_value(qualifier=xqualifier, component=ds_comp, **_skip_filter_checks)
                            model_x = model_ps.get_value(qualifier=xqualifier, dataset=ds, component=ds_comp, **_skip_filter_checks)
                            ds_sigmas = ds_ps.get_value(qualifier='sigmas', component=ds_comp, **_skip_filter_checks)
                            # TODO: do we need to inflate sigmas by lnf?
                            if len(ds_sigmas) != len(ds_x):
                                raise ValueError("gaussian_process requires sigma of same length as {}".format(xqualifier))
                            gp_kernel.compute(ds_x, ds_sigmas, check_sorted=True)

                            residuals, model_y_dstimes = self.calculate_residuals(model=model, dataset=ds, component=ds_comp, return_interp_model=True, as_quantity=False, consider_gaussian_process=False)
                            gp_y = gp_kernel.predict(residuals, ds_x, return_cov=False)
                            model_y = model_ps.get_quantity(qualifier=yqualifier, dataset=ds, component=ds_comp, **_skip_filter_checks)

                            # store just the GP component in the model PS as well
                            gp_param = FloatArrayParameter(qualifier='gps', value=gp_y, default_unit=model_y.unit, readonly=True, description='GP contribution to the model {}'.format(yqualifier))
                            y_nogp_param = FloatArrayParameter(qualifier='{}_nogps'.format(yqualifier), value=model_y_dstimes, default_unit=model_y.unit, readonly=True, description='{} before adding gps'.format(yqualifier))
                            if not np.all(ds_x == model_x):
                                logger.warning("model for dataset='{}' resampled at dataset times when adding GPs".format(ds))
                                model_ps.set_value(qualifier=xqualifier, dataset=ds, component=ds_comp, value=ds_x, ignore_readonly=True, **_skip_filter_checks)

                            self._attach_params([gp_param, y_nogp_param], check_copy_for=False, **metawargs)

                            # update the model to include the GP contribution
                            model_ps.set_value(qualifier=yqualifier, value=model_y_dstimes+gp_y, dataset=ds, component=ds_comp, ignore_readonly=True, **_skip_filter_checks)

        except Exception as err:
            restore_conf()
            raise

        restore_conf()

        ret_ps = self.filter(model=model, context=None if return_changes else 'model', **_skip_filter_checks)
        ret_changes += self._run_compute_changes(ret_ps,
                                                 return_changes=return_changes,
                                                 do_create_fig_params=do_create_fig_params,
                                                 **kwargs)

        if kwargs.get('overwrite', model=='latest') and return_changes and overwrite_ps is not None:
            ret_ps += overwrite_ps

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_model(self, model=None, **kwargs):
        """
        Filter in the 'model' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `model`: (string, optional, default=None): the name of the model
        * `**kwargs`: any other tags to do the filtering (excluding model and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if model is not None:
            kwargs['model'] = model
            if model not in self.models:
                raise ValueError("model='{}' not found".format(model))

        kwargs['context'] = 'model'
        return self.filter(**kwargs)

    def rerun_model(self, model=None, **kwargs):
        """
        Rerun run_compute for a given model.  This simply retrieves the current
        compute parameters given the same compute label used to create the original
        model.  This does not, therefore, necessarily ensure that the exact
        same compute options are used.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_compute>

        Arguments
        ------------
        * `model` (string, optional): label of the model (will be overwritten)
        * `**kwargs`: all keyword arguments are passed directly to
            <phoebe.frontend.bundle.Bundle.run_compute>

        Returns
        ------------
        * the output from <phoebe.frontend.bundle.Bundle.run_compute>
        """
        model_ps = self.get_model(model=model)

        compute = model_ps.compute
        kwargs.setdefault('compute', compute)

        return self.run_compute(model=model, **kwargs)

    def import_model(self, fname, model=None, overwrite=False, return_changes=False):
        """
        Import and attach a model from a file.

        Generally this file will be the output after running a script generated
        by <phoebe.frontend.bundle.Bundle.export_compute>.  This is NOT necessary
        to be called if generating a model directly from
        <phoebe.frontend.bundle.Bundle.run_compute>.

        See also:
        * <phoebe.frontend.bundle.Bundle.export_compute>

        Arguments
        ------------
        * `fname` (string): the path to the file containing the model.  Likely
            `out_fname` from <phoebe.frontend.bundle.Bundle.export_compute>.
            Alternatively, this can be the json of the model.  Must be
            able to be parsed by <phoebe.parameters.ParameterSet.open>.
        * `model` (string, optional): the name of the model to be attached
            to the Bundle.  If not provided, the model will be adopted from
            the tags in the file.
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.

        Returns
        -----------
        * ParameterSet of added and changed parameters
        """
        result_ps = ParameterSet.open(fname)
        metawargs = {}
        if model is None:
            model = result_ps.model
        metawargs['model'] = model

        ret_changes = []
        new_uniqueids = False
        if model in self.models:
            if overwrite:
                ret_changes += self.remove_model(model, during_overwrite=True, return_changes=return_changes).to_list()
                new_uniqueids = True
            else:
                raise ValueError("model '{}' already exists.  Use different name or pass overwrite=True".format(model))

        self._attach_params(result_ps, override_tags=True, new_uniqueids=new_uniqueids, **metawargs)

        ret_ps = self.get_model(model=model if model is not None else result_ps.models)
        ret_changes += self._run_compute_changes(ret_ps,
                                                return_changes=return_changes,
                                                do_create_fig_params=False)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    @send_if_client
    def remove_model(self, model, return_changes=False, **kwargs):
        """
        Remove a 'model' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `model` (string): the label of the model to be removed.
        * `remove_figure_params` (bool, optional): whether to also remove
            figure options tagged with `model`.  If not provided, will default
            to false if `model` is 'latest', otherwise will default to True.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: model, context.

        Returns
        -----------
        * ParameterSet of removed or changed parameters
        """
        remove_figure_params = kwargs.pop('remove_figure_params', model!='latest')

        kwargs['model'] = model
        kwargs['context'] = ['model', 'figure'] if remove_figure_params else 'model'
        ret_ps = self.remove_parameters_all(**kwargs)

        ret_changes = self._run_compute_changes(ret_ps,
                                                return_changes=return_changes,
                                                do_create_fig_params=False,
                                                removed=True,
                                                **kwargs)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def remove_models_all(self, return_changes=False):
        """
        Remove all models from the bundle.  To remove a single model see
        <phoebe.frontend.bundle.Bundle.remove_model>.

        Arguments
        -------------
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for model in self.models:
            removed_ps += self.remove_model(model=model, return_changes=return_changes)
        return removed_ps

    @send_if_client
    def rename_model(self, old_model, new_model,
                     overwrite=False, return_changes=False):
        """
        Change the label of a model attached to the Bundle.

        Arguments
        ----------
        * `old_model` (string): current label of the model (must exist)
        * `new_model` (string): the desired new label of the model
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed model

        Raises
        --------
        * ValueError: if the value of `new_model` is forbidden or already exists.
        """
        # TODO: raise error if old_feature not found?
        self._rename_label('model', old_model, new_model, overwrite)

        ret_ps = self.filter(model=new_model)

        ret_changes = []
        ret_changes += self._handle_model_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    @send_if_client
    def attach_job(self, twig=None, wait=True, sleep=5, cleanup=True,
                   return_changes=False, **kwargs):
        """
        Attach the results from an existing <phoebe.parameters.JobParameter>.

        Jobs are created when passing `detach=True` to
        <phoebe.frontend.bundle.Bundle.run_compute> or
        <phoebe.frontend.bundle.Bundle.run_solver>.

        See also:
        * <phoebe.frontend.bundle.Bundle.kill_job>
        * <phoebe.parameters.JobParameter.attach>

        Arguments
        ------------
        * `twig` (string, optional): twig to use for filtering for the JobParameter.
        * `wait` (bool, optional, default=True): whether to enter a loop to wait
            for results if the Job is not yet complete.
        * `sleep` (int, optional, default=5): number of seconds to wait in the loop.
            Only applicable if `wait` is True.
        * `cleanup` (bool, optional, default=True): whether to delete any
            temporary files created by the Job.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: any additional keyword arguments are sent to filter for the
            Job parameters.  Between `twig` and `**kwargs`, a single parameter
            with qualifier of 'detached_job' must be found.

        Returns
        -----------
        * (<phoebe.parameters.ParameterSet>): ParameterSet of the newly attached
            Parameters.
        """
        kwargs['qualifier'] = 'detached_job'
        return self.get_parameter(twig=twig, **kwargs).attach(wait=wait, sleep=sleep, cleanup=cleanup, return_changes=return_changes)

    def kill_job(self, twig=None, cleanup=True,
                   return_changes=False, **kwargs):
        """
        Send a termination signal to the external job referenced by an existing
        <phoebe.parameters.JobParameter>.

        Jobs are created when passing `detach=True` to
        <phoebe.frontend.bundle.Bundle.run_compute> or
        <phoebe.frontend.bundle.Bundle.run_solver>.

        See also:
        * <phoebe.frontend.bundle.Bundle.attach_job>
        * <phoebe.parameters.JobParameter.kill>

        Arguments
        ------------
        * `twig` (string, optional): twig to use for filtering for the JobParameter.
        * `cleanup` (bool, optional, default=True): whether to delete any
            temporary files created by the Job.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: any additional keyword arguments are sent to filter for the
            Job parameters.  Between `twig` and `**kwargs`, a single parameter
            with qualifier of 'detached_job' must be found.

        Returns
        -----------
        * (<phoebe.parameters.ParameterSet>): ParameterSet of the newly attached
            Parameters.
        """
        kwargs['qualifier'] = 'detached_job'
        return self.get_parameter(twig=twig, **kwargs).kill(cleanup=cleanup, return_changes=return_changes)

    @send_if_client
    def add_solver(self, kind, return_changes=False, **kwargs):
        """
        Add a set of solver options for a given backend to the bundle.
        The label (`solver`) can then be sent to <phoebe.frontend.bundle.Bundle.run_solver>.

        If not provided, `solver` will be created for you and can be
        accessed by the `solver` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        Available kinds can be found in <phoebe.parameters.solver> or by calling
        <phoebe.list_available_solvers> and include:
        * <phoebe.parameters.solver.sampler.emcee>
        * <phoebe.parameters.solver.optimizer.nelder_mead>

        Arguments
        ----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts only default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.solver> module.
        * `solver` (string, optional): name of the newly-created solver options.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing set of solver options with the same `solver` tag.  If False,
            an error will be raised.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added

        Raises
        --------
        * NotImplementedError: if a required constraint is not implemented
        """
        func = _get_add_func(_solver, kind)

        # remove if None
        if kwargs.get('solver', False) is None:
            _ = kwargs.pop('solver')

        kwargs.setdefault('solver',
                          self._default_label(func.__name__,
                                              **{'context': 'solver',
                                                 'kind': func.__name__}))

        self._check_label(kwargs['solver'], allow_overwrite=kwargs.get('overwrite', False))

        ## add any necessary constraints needed by the solver
        solver_kind = func.__name__
        if solver_kind in ['ebai']:
            for orbit in self.hierarchy.get_orbits():
                orbit_ps = self.get_orbit(component=orbit, **_skip_filter_checks)
                for constraint in ['teffratio', 'requivsumfrac']:
                    if constraint not in orbit_ps.qualifiers:
                        logger.warning("adding {} constraint to {} orbit (needed for {} solver)".format(constraint, orbit, solver_kind))
                        self.add_constraint(constraint, component=orbit)


        # NOTE: we don't pass kwargs here since so many require the choices
        # to be populated.  Instead, we loop through kwargs and set the values
        # later
        params = func()
        # TODO: similar kwargs logic as in add_dataset (option to pass dict to
        # apply to different components this would be more complicated here if
        # allowing to also pass to different datasets

        metawargs = {'context': 'solver',
                     'kind': func.__name__,
                     'solver': kwargs['solver']}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_solver(solver=kwargs['solver'], auto_remove_figure=False, during_overwrite=True)
            # check the label again, just in case kwargs['solver'] belongs to
            # something other than solver
            self.exclude(context='solution', **_skip_filter_checks)._check_label(kwargs['solver'], allow_overwrite=False)

        logger.info("adding {} '{}' solver to bundle".format(metawargs['kind'], metawargs['solver']))
        self._attach_params(params, **metawargs)

        # TODO: OPTIMIZE only trigger those necessary based on the solver-backend
        ret_changes = []
        ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_compute_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_solver_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_solver_selectparams(return_changes=return_changes)
        ret_changes += self._handle_fitparameters_selecttwigparams(return_changes=return_changes)
        ret_changes += self._handle_dataset_selectparams(return_changes=return_changes)
        ret_changes += self._handle_orbit_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_component_choiceparams(return_changes=return_changes)

        ret_ps = self.get_solver(check_visible=False, check_default=False, **metawargs)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite'], warning_only=True, ps=ret_ps)

        if kwargs.get('overwrite', False) and return_changes:
            ret_ps += overwrite_ps

        # now set parameters that needed updated choices
        qualifiers = ret_ps.qualifiers
        for k,v in kwargs.items():
            if k in qualifiers:
                ret_ps.set_value_all(qualifier=k, value=v, **_skip_filter_checks)
            # TODO: else raise warning?

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_solver(self, solver=None, **kwargs):
        """
        Filter in the 'solver' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `solver`: (string, optional, default=None): the name of the solver options
        * `**kwargs`: any other tags to do the filtering (excluding solver and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if solver is not None:
            kwargs['solver'] = solver
            if solver not in self.solvers:
                raise ValueError("solver='{}' not found".format(solver))
        kwargs['context'] = 'solver'
        return self.filter(**kwargs)

    @send_if_client
    def remove_solver(self, solver, return_changes=False, **kwargs):
        """
        Remove a 'solver' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `solver` (string): the label of the solver options to be removed.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: context, solver.

        Returns
        -----------
        * ParameterSet of removed or changed parameters
        """
        kwargs['solver'] = solver
        kwargs['context'] = 'solver'
        ret_ps = self.remove_parameters_all(**kwargs)

        ret_changes = []

        auto_remove_figure = self.get_value(qualifier='auto_remove_figure', context='setting', auto_remove_figure=kwargs.get('auto_remove_figure', None), default=False, **_skip_filter_checks)
        if auto_remove_figure:
            for param in self.filter(qualifier='solver', context='figure', kind=ret_ps.kind, **_skip_filter_checks).to_list():
                if param.get_value() == solver:
                    ret_changes += self.remove_figure(param.figure, return_changes=return_changes).to_list()

        ret_changes += self._handle_distribution_selectparams(return_changes=return_changes)
        if not kwargs.get('during_overwrite', False):
            ret_changes += self._handle_solver_choiceparams(return_changes=return_changes)
            ret_changes += self._handle_solver_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def remove_solvers_all(self, return_changes=False):
        """
        Remove all solver options from the bundle.  To remove a single set
        of solver options see <phoebe.frontend.bundle.Bundle.remove_solver>.

        Arguments
        ------------
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for solver in self.solvers:
            removed_ps += self.remove_solver(solver, return_changes=return_changes)
        return removed_ps

    @send_if_client
    def rename_solver(self, old_solver, new_solver,
                      overwrite=False, return_changes=False):
        """
        Change the label of solver options attached to the Bundle.

        Arguments
        ----------
        * `old_solver` (string): current label of the solver options (must exist)
        * `new_solver` (string): the desired new label of the solver options
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_solver` is forbidden or already exists.
        """
        # TODO: raise error if old_solver not found?
        self._rename_label('solver', old_solver, new_solver, overwrite)

        ret_ps = self.filter(solver=new_solver)

        ret_changes = []
        ret_changes += self._handle_solver_choiceparams(return_changes=return_changes)
        ret_changes += self._handle_solver_selectparams(return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)


    def _prepare_solver(self, solver, solution, **kwargs):
        """
        """

        if solution is None:
            solution = 'latest'

        self._check_label(solution, allow_overwrite=kwargs.get('overwrite', solution=='latest'))

        # handle case where solver is not provided
        if solver is None:
            solvers = self.get_solver(check_default=False, check_visible=False, **{k:v for k,v in kwargs.items() if k in ['kind', 'solver']}).solvers
            if len(solvers)==0:
                raise ValueError("no solvers attached.  Call add_solver first")
            if len(solvers)==1:
                solver = solvers[0]
            elif len(solvers)>1:
                raise ValueError("must provide label of solver options since more than one are attached.  The following were found: {}".format(self.solvers))


        solver_ps = self.get_solver(solver=solver, kind=kwargs.get('kind'), **_skip_filter_checks)
        if solver_ps is None:
            raise ValueError("could not find solver with solver={} kwargs={}".format(solver, kwargs))

        if 'compute' in solver_ps.qualifiers:
            compute = kwargs.pop('compute', solver_ps.get_value(qualifier='compute', **_skip_filter_checks))
            compute_ps = self.get_compute(compute=compute, **_skip_filter_checks)

            if len(compute_ps.computes) > 1:
                raise ValueError("more than one set of compute options attached, must provide compute")
            elif not len(compute_ps.computes):
                raise ValueError("no compute options found")

            compute = compute_ps.compute

        else:
            compute = kwargs.pop('compute', None)
            compute_ps = self.get_compute(compute=compute)

        # we'll wait to here to run kwargs and system checks so that
        # add_compute is already called if necessary
        allowed_kwargs = ['skip_checks', 'jobid', 'overwrite', 'max_computations', 'in_export_script', 'out_fname', 'solution', 'progressbar', 'custom_lnprobability_callable']
        if conf.devel:
            allowed_kwargs += ['mesh_init_phi']
        self._kwargs_checks(kwargs, allowed_kwargs, ps=solver_ps.copy()+compute_ps)

        if not kwargs.get('skip_checks', False):
            report = self.run_checks_solver(solver=solver, run_checks_compute=True,
                                            allow_skip_constraints=False,
                                            raise_logger_warning=True, raise_error=True,
                                            **kwargs)

        return solver, solution, compute, solver_ps


    def _write_export_solver_script(self, script_fname, out_fname, solver, solution, autocontinue, import_from_older, log_level, kwargs):
        """
        """
        f = open(script_fname, 'w')
        f.write("import os; os.environ['PHOEBE_ENABLE_PLOTTING'] = 'FALSE'; os.environ['PHOEBE_ENABLE_SYMPY'] = 'FALSE';\n")
        f.write("import phoebe; import json\n")
        if log_level is not None:
            f.write("phoebe.logger('{}')\n".format(log_level))
        # TODO: can we skip other models
        # or datasets (except times and only for run_compute but not run_solver)
        exclude_contexts = ['model', 'figure']
        continue_from = self.get_value(qualifier='continue_from', solver=solver, continue_from=kwargs.get('continue_from', None), default='')
        exclude_solutions = [sol for sol in self.solutions if sol!=continue_from]
        exclude_solvers = [s for s in self.solvers if s!=solver]
        solver_ps = self.get_solver(solver=solver, **_skip_filter_checks)
        needed_distributions_qualifiers = ['init_from', 'priors', 'bounds']
        needed_distributions = list(np.concatenate([solver_ps.get_value(qualifier=q, check_visible=False, default=[], **{q: kwargs.get(q, None)}) for q in needed_distributions_qualifiers]))
        exclude_distributions = [d for d in self.distributions if d not in needed_distributions]
        if 'continue_from_ps' in kwargs.keys():
            b = self.copy()
            b._attach_params(kwargs.pop('continue_from_ps').exclude(qualifier='detached_job', **_skip_filter_checks).copy())
        else:
            b = self

        f.write("bdict = json.loads(\"\"\"{}\"\"\", object_pairs_hook=phoebe.utils.parse_json)\n".format(json.dumps(b.exclude(context=exclude_contexts, **_skip_filter_checks).exclude(solution=exclude_solutions, **_skip_filter_checks).exclude(solver=exclude_solvers, **_skip_filter_checks).exclude(distribution=exclude_distributions, **_skip_filter_checks).to_json(incl_uniqueid=True, exclude=['description', 'advanced', 'readonly', 'copy_for', 'latexfmt', 'labels_latex', 'label_latex']))))
        f.write("b = phoebe.open(bdict, import_from_older={})\n".format(import_from_older))

        custom_lnprobability_callable = kwargs.get('custom_lnprobability_callable', None)
        if custom_lnprobability_callable is not None:
            code = _getsource(custom_lnprobability_callable)
            f.write(code)
            kwargs['custom_lnprobability_callable'] = custom_lnprobability_callable.__name__

        solver_kwargs = list(kwargs.items())+[('solver', solver), ('solution', str(solution))]
        solver_kwargs_string = ','.join(["{}={}".format(k,"\'{}\'".format(str(v)) if isinstance(v, str) and k!='custom_lnprobability_callable' else v) for k,v in solver_kwargs])


        if out_fname is None:
            f.write("import sys\n")
            f.write("out_fname=sys.argv[0]+'.out'\n")
            out_fname = script_fname+'.out'
        else:
            f.write("out_fname='{}'\n".format(out_fname))

        if autocontinue:
            if 'continue_from' not in self.get_solver(solver=solver).qualifiers:
                raise ValueError("continue_from is not a parameter in solver='{}', cannot use autocontinue".format(solver))
            f.write("if os.path.isfile(out_fname):\n")
            f.write("    b.import_solution(out_fname, solution='progress', overwrite=True)\n")
            f.write("    b.set_value(qualifier='continue_from', solver='{}', value='progress')\n".format(solver))
            f.write("elif os.path.isfile(out_fname+'.progress'):\n")
            f.write("    b.import_solution(out_fname+'.progress', solution='progress', overwrite=True)\n")
            f.write("    b.set_value(qualifier='continue_from', solver='{}', value='progress')\n".format(solver))

        f.write("solution_ps = b.run_solver(out_fname=out_fname, {})\n".format(solver_kwargs_string))
        f.write("b.filter(context='solution', solution=solution_ps.solution, check_visible=False).save(out_fname, incl_uniqueid=True)\n")

        f.write("\n# NOTE: this script only includes parameters needed to call the requested run_solver, edit manually with caution!\n")
        f.close()

        return script_fname, out_fname

    def export_solver(self, script_fname, out_fname=None,
                      solver=None, solution=None,
                      pause=False,
                      autocontinue=False,
                      import_from_older=False,
                      log_level=None,
                      **kwargs):
        """
        Export a script to call run_solver externally (in a different thread
        or on a different machine).

        After running the resulting `script_fname`, `out_fname` will be created,
        which will contain a ParameterSet of the solution parameters.  To attach
        that solution to this bundle, see <phoebe.frontend.bundle.Bundle.import_solution>.

        Arguments
        ------------
        * `script_fname` (string): the filename of the python script to be generated.
        * `out_fname` (string, optional): the filename of the output file that `script_fname`
            will write when executed.  Once executed, pass this filename to
            <phoebe.frontend.bundle.Bundle.import_model> to load the resulting
            model.  If not provided, the script will automatically export
            to `script_fname`.out (where the filename is determined at runtime,
            so if you rename the script exported here, the resulting filename
            will reflect that change and be appended with '.out').
        * `solver` (string, optional, default=None):
        * `solution` (string, optional, default=None):
        * `pause` (bool, optional, default=False): whether to raise an input
            with instructions for running the exported script and calling
            <phoebe.frontend.bundle.Bundle.import_solution>.  Particularly
            useful if running in an interactive notebook or a script.
        * `autocontinue` (bool, optional, default=False): override `continue_from`
            in `solver` to continue from `out_fname` (or `script_fname`.out or
            .progress files) if those files exist.  This is useful to set to True
            and then resubmit the same script if not converged (although care should
            be taken to ensure multiple scripts aren't reading/writing from the
            same filenames).  `continue_from` must be a parameter in `solver` options,
            or an error will be raised if `autocontinue=True`
        * `import_from_older` (boolean, optional, default=False): whether to allow
            the script to run on a newer version of PHOEBE.  If True and executing
            the outputed script (`script_fname`) on a newer version of PHOEBE,
            the bundle will attempt to migrate to the newer version.  If False,
            an error will be raised when attempting to run the script.  See
            also: <phoebe.frontend.bundle.Bundle.open>.
        * `log_level` (string, optional, default=None): `clevel` to set in the
            logger in the exported script.  See <phoebe.logger>.
        * `custom_lnprobability_callable` (callable, optional, default=None):
            custom callable function which takes the following arguments:
            `b, model, lnpriors, priors, priors_combine` and returns the lnprobability
            to override the built-in lnprobability of <phoebe.frontend.bundle.Bundle.calculate_lnp> (on priors)
            + <phoebe.parameters.ParameterSet.calculate_lnlikelihood>.  For
            optimizers that minimize, the negative returned values will be minimized.
            NOTE: if defined in an interactive session and inspect.getsource fails,
            this will raise an error.
        * `**kwargs`:: any values in the solver or compute options to temporarily
            override for this single solver run (parameter values will revert
            after run_solver is finished).

        Returns
        -----------
        * `script_fname`, `out_fname`.  Where running `script_fname` will result
          in the model being written to `out_fname`.

        """
        solver, solution, compute, solver_ps = self._prepare_solver(solver, solution, **kwargs)
        script_fname, out_fname = self._write_export_solver_script(script_fname, out_fname, solver, solution, autocontinue, import_from_older, log_level, kwargs)

        if pause:
            input("* optional:  call b.save(...) to save the bundle to disk, you can then safely close the active python session and recover the bundle with phoebe.load(...)\n"+
                  "* run {} (within mpirun or on an external machine, if desired)\n".format(script_fname)+
                  "* once completed, copy {} to this directory, if necessary\n".format(out_fname)+
                  "* press enter to exit this pause\n"+
                  "* call b.import_solution('{}')\n".format(out_fname)+
                  "\n(press enter to continue)")

        return script_fname, out_fname


    def _run_solver_changes(self, ret_ps, return_changes=False, removed=False, auto_add_figure=None, auto_remove_figure=None, during_overwrite=False):
        """
        """
        ret_changes = []

        auto_add_figure = self.get_value(qualifier='auto_add_figure', context='setting', auto_add_figure=auto_add_figure, default=False, **_skip_filter_checks)
        auto_remove_figure = self.get_value(qualifier='auto_remove_figure', context='setting', auto_remove_figure=auto_remove_figure, default=False, **_skip_filter_checks)

        def _figure_match(solution, kinds):
            for p in self.filter(qualifier='solution', context='figure', kind=kinds, **_skip_filter_checks).to_list():
                # check to see if there is a solver match or all options removed (in which case probably from an overwrite=True)
                if p.get_value() == solution or not len(p.choices):
                    return True
            return False

        if auto_add_figure and not removed and not _figure_match(ret_ps.solution, ret_ps.kinds):
            # then we don't have a figure for this kind yet
            logger.info("calling add_figure(kind='solution.{}') since auto_add_figure@setting=True".format(ret_ps.kind))
            try:
                new_fig_params = self.add_figure(kind='solution.{}'.format(ret_ps.kind), return_changes=return_changes)
            except ValueError:
                # not all solution types have corresponding figures
                logger.info("add_figure failed")
                pass
            else:
                ret_changes += new_fig_params.to_list()
                if not during_overwrite:
                    ret_changes += self._handle_solution_choiceparams(return_changes=return_changes)
                    ret_changes += self._handle_solution_selectparams(return_changes=return_changes)
                new_fig_params.set_value_all(qualifier='solution', context='figure', value=ret_ps.solution)
        elif auto_remove_figure and removed:
            for param in self.filter(qualifier='solution', context='figure', kind=ret_ps.kind, **_skip_filter_checks).to_list():
                if param.get_value() == ret_ps.solution:
                    ret_changes += self.remove_figure(param.figure, return_changes=return_changes, during_overwrite=during_overwrite).to_list()


        if not during_overwrite:
            ret_changes += self._handle_solution_choiceparams(return_changes=return_changes)
            ret_changes += self._handle_solution_selectparams(return_changes=return_changes)
        ret_changes += self._handle_computesamplefrom_selectparams(return_changes=return_changes)

        return ret_changes



    def parse_solver_times(self, return_as_dict=True, set_compute_times=False):
        """
        Parse what times will be used within <phoebe.frontend.bundle.Bundle.run_solver>
        (for any optimizer or sampler that requires a forward-model)
        or when passing `solver` to <phoebe.frontend.bundle.Bundle.run_compute>,
        based on the value of `solver_times`, `times`, `compute_times`, `mask_enabled`,
        `mask_phases`, and <phoebe.parameters.HierarchyParameter.is_time_dependent>.

        Note: this is not necessary to call manually as it will be called and
        handled automatically within <phoebe.frontend.bundle.Bundle.run_solver>
        or <phoebe.frontend.bundle.Bundle.run_compute>.  This method only exposes
        the same logic to diagnose the influence of these options on the computed
        times.

        Overview of logic:
        * if `solver_times='times'` but not `mask_enabled`: returns the dataset-times
            (concatenating over components as necessary).
        * if `solver_times='times'` and `mask_enabled`: returns the masked
            dataset-times (concatenating over components as necessary), according
            to `mask_phases`.
        * if `solver_times='compute_times'` but not `mask_enabled`: returns
            the `compute_times`
        * if `solver_times='compute_times'` and `mask_enabled` but not time-dependent:
            returns the values in `compute_phases` (converted to times) such that
            each entry in the masked dataset-times (concatenating over components
            as necessary) is surrounded by the nearest entries in `compute_phases`.
        * if `solver_times='compute_times'` and `mask_enabled` and time-dependent:
            returns the values in `compute_times` such that each entry in the
            masked dataset-times is surrounded by the nearest entries in `compute_times`.
        * if `solver_times='auto'`: determines the arrays for the corresponding
            situations for both `solver_times='times'` and `'compute_times'` and
            returns whichever is the shorter array.

        Note that this logic is currently independent per-dataset, with no consideration
        of time-savings by generating synthetic models at the same times across
        different datasets.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_solver>
        * <phoebe.frontend.bundle.Bundle.run_compute>
        * <phoebe.parameters.HierarchyParameter.is_time_dependent>

        Arguments
        -------------
        * `return_as_dict` (bool, optional, default=True): whether to return
            the parsed times as they will be applied as a dictionary of
            dataset-list pairs.
        * `set_compute_times` (bool, optional, default=False): whether to set
            the values of the corresponding `compute_times` (and `compute_phases`)
            parameters within the Bundle.  Whenever using the dataset times,
            this will set the `compute_times` to an empty list rather than
            copying the array (whereas `return_as_dict` exposes the underlying array).

        Returns
        ------------
        * (dict) of dataset-list pairs, if `return_as_dict` otherwise `None`

        """
        compute_times_per_ds = {}

        is_time_dependent = self.hierarchy.is_time_dependent()

        # TODO: change this to have times = {dataset: new_compute_times} which will be passed to all run_compute calls
        # so that this is also callable from the frontend for the user or by passing solver to run_compute

        # handle solver_times
        for param in self.filter(qualifier='solver_times', **_skip_filter_checks).to_list():
            # TODO: skip if this dataset is disabled for compute?
            # TODO: any change in logic for time-dependent systems?

            solver_times = param.get_value()
            ds_ps = self.get_dataset(dataset=param.dataset, **_skip_filter_checks)
            mask_enabled = ds_ps.get_value(qualifier='mask_enabled', default=False, **_skip_filter_checks)
            if mask_enabled:
                mask_phases = ds_ps.get_value(qualifier='mask_phases', **_skip_filter_checks)
                mask_t0 = ds_ps.get_value(qualifier='phases_t0', **_skip_filter_checks)
            else:
                mask_phases = None
                mask_t0 = 't0_supconj'

            new_compute_times = None

            if solver_times == 'auto':
                masked_times, times, phases = _get_masked_times(self, param.dataset, mask_phases, mask_t0, return_times_phases=True)
                masked_compute_times = _get_masked_compute_times(self, param.dataset, mask_phases, mask_t0,
                                                                 is_time_dependent=is_time_dependent,
                                                                 times=times, phases=phases)

                if len(masked_times) < len(masked_compute_times):
                    if mask_enabled and len(mask_phases):
                        logger.info("solver_times=auto (dataset={}): using masked times (len: {}) instead of compute_times (len: {})".format(param.dataset, len(masked_times), len(masked_compute_times)))
                        new_compute_times = masked_times
                    else:
                        logger.info("solver_times=auto (dataset={}): using dataset times (len: {}) instead of compute_times (len: {})".format(param.dataset, len(masked_times), len(masked_compute_times)))
                        new_compute_times = [] # set to empty list which will force run_compute to use dataset-times
                else:
                    if mask_enabled and len(mask_phases):
                        if is_time_dependent:
                            logger.info("solver_times=auto (dataset={}): using filtered compute_times (len: {}) surrounding masked times instead of times (len: {})".format(param.dataset, len(masked_compute_times), len(masked_times)))
                        else:
                            logger.info("solver_times=auto (dataset={}): using filtered compute_phases (len: {}) surrounding masked phases instead of times (len: {})".format(param.dataset, len(masked_compute_times), len(masked_times)))
                        new_compute_times = masked_compute_times
                    else:
                        logger.info("solver_times=auto (dataset={}): using user-provided compute_times (len: {}) instead of times (len: {})".format(param.dataset, len(masked_compute_times), len(times)))
                        new_compute_times = None  # leave at user-set compute_times


            elif solver_times == 'times':
                if mask_enabled and len(mask_phases):
                    logger.info("solver_times=times (dataset={}): using masked dataset times".format(param.dataset))
                    masked_times = _get_masked_times(self, param.dataset, mask_phases, mask_t0, return_times_phases=False)
                    new_compute_times = masked_times
                else:
                    logger.info("solver_times=times (dataset={}): using dataset times".format(param.dataset))
                    masked_times = None # if return_as_dict then we'll have to re-access dataset times
                    new_compute_times = [] # set to empty list which will force run_compute to use dataset-times

            elif solver_times == 'compute_times':
                if mask_enabled and len(mask_phases):
                    if is_time_dependent:
                        logger.info("solver_times=compute_times (dataset={}): using filtered compute_times surrounding masked times".format(param.dataset))
                    else:
                        logger.info("solver_times=compute_times (dataset={}): using filtered compute_phases surrounding masked phases".format(param.dataset))
                    masked_compute_times =  _get_masked_compute_times(self, param.dataset, mask_phases, mask_t0,
                                                                   is_time_dependent=is_time_dependent)
                    new_compute_times = masked_compute_times
                else:
                    logger.info("solver_times=compute_times (dataset={}): using user-provided compute_times".format(param.dataset))
                    masked_compute_times = None # if return_as_dict then we'll have to re-access compute_times
                    new_compute_times = None  # leave at user-set compute_times
            else:
                raise NotImplementedError("solver_times='{}' not implemented".format(solver_times))

            if return_as_dict:
                if new_compute_times == []:
                    if masked_times is None:
                        compute_times_per_ds[param.dataset] = _get_masked_times(self, param.dataset, [], 0.0, return_times_phases=False)
                    else:
                        compute_times_per_ds[param.dataset] = masked_times
                elif new_compute_times is None:
                    if masked_compute_times is None:
                        compute_times_per_ds[param.dataset] = ds_ps.get_value(qualifier='compute_times', unit=u.d, **_skip_filter_checks)
                    else:
                        compute_times_per_ds[param.dataset] = masked_compute_times
                else:
                    compute_times_per_ds[param.dataset] = new_compute_times

            if set_compute_times and new_compute_times is not None:
                if ds_ps.get_parameter(qualifier='compute_times', **_skip_filter_checks).is_constraint is not None:
                    # this is in the deepcopied bundle, so we can overwrite compute_times directly
                    self.flip_constraint(qualifier='compute_times', dataset=ds_ps.dataset, solve_for='compute_phases')
                logger.info("solver_times={} (dataset={}): setting compute_times".format(solver_times, param.dataset))
                ds_ps.set_value_all(qualifier='compute_times', value=new_compute_times, **_skip_filter_checks)

        if return_as_dict:
            return compute_times_per_ds
        else:
            return


    @send_if_client
    def run_solver(self, solver=None, solution=None, detach=False, return_changes=False, **kwargs):
        """
        Run a forward model of the system on the enabled dataset(s) using
        a specified set of solver options.

        To attach and set custom values for solver options, including choosing
        which backend to use, see:
        * <phoebe.frontend.bundle.Bundle.add_solver>

        To attach and set custom values for compute options, including choosing
        which backend to use, see:
        * <phoebe.frontend.bundle.Bundle.add_compute>

        To define the dataset types and times at which the model(s) should be
        computed see:
        * <phoebe.frontend.bundle.Bundle.add_dataset>

        To disable or enable existing datasets see:
        * <phoebe.frontend.bundle.Bundle.enable_dataset>
        * <phoebe.frontend.bundle.Bundle.disable_dataset>

        See also:
        * <phoebe.frontend.bundle.Bundle.add_solver>
        * <phoebe.frontend.bundle.Bundle.get_solver>
        * <phoebe.frontend.bundle.Bundle.get_solution>
        * <phoebe.frontend.bundle.Bundle.adopt_solution>
        * <phoebe.frontend.bundle.Bundle.parse_solver_times>
        * <phoebe.mpi_on>
        * <phoebe.mpi_off>

        Arguments
        ------------
        * `solver` (string, optional): name of the solver options to use.
            If not provided or None, run_solver will use an existing set of
            attached solver options if only 1 exists.  If more than 1 exist,
            then solver becomes a required argument.  If no solver options
            exist, an error will be raised.
        * `solution` (string, optional): name of the resulting solution.  If not
            provided this will default to 'latest'.  NOTE: existing solutions
            with the same name will be overwritten depending on the value
            of `overwrite` (see below).   See also
            <phoebe.frontend.bundle.Bundle.rename_solution> to rename a solution after
            creation.
        * `detach` (bool, optional, default=False, EXPERIMENTAL):
            whether to detach from the solver run,
            or wait for computations to complete.  If detach is True, see
            <phoebe.frontend.bundle.Bundle.get_solution> and
            <phoebe.parameters.JobParameter>
            for details on how to check the job status and retrieve the results.
        * `overwrite` (boolean, optional, default=solution=='latest'): whether to overwrite
            an existing model with the same `model` tag.  If False,
            an error will be raised.  This defaults to True if `model` is not provided
            or is 'latest', otherwise it will default to False.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet, including
            the removed parameters due to `overwrite`.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks_solver>.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `custom_lnprobability_callable` (callable, optional, default=None):
            custom callable function which takes the following arguments:
            `b, model, lnpriors, priors, priors_combine` and returns the lnprobability
            to override the built-in lnprobability of <phoebe.frontend.bundle.Bundle.calculate_lnp> (on priors)
            + <phoebe.parameters.ParameterSet.calculate_lnlikelihood>.  For
            optimizers that minimize, the negative returned values will be minimized.
            NOTE: if defined in an interactive session, passing `custom_lnprobability_callable`
            may throw an error if `detach=True`.
        * `progressbar` (bool, optional): whether to show a progressbar.  If not
            provided or none, will default to <phoebe.progressbars_on> or
            <phoebe.progressbars_off>.  Progressbars require `tqdm` to be installed
            (will silently ignore if not installed).
        * `**kwargs`: any values in the solver or compute options to temporarily
            override for this single solver run (parameter values will revert
            after run_solver is finished)

        Returns
        ----------
        * a <phoebe.parameters.ParameterSet> of the newly-created solver solution.

        """
        if isinstance(detach, str):
            # then we want to temporarily go in to client mode
            raise NotImplementedError("detach currently must be a bool")
            self.as_client(as_client=detach)
            self.run_solver(solver=solver, solution=solution, **kwargs)
            self.as_client(as_client=False)
            return self.get_solution(solution=solution)

        kwargs.setdefault('progressbar', conf.progressbars)

        solver, solution, compute, solver_ps = self._prepare_solver(solver, solution, **kwargs)

        if not kwargs.get('skip_checks', False):
            self.run_checks_solver(solver=solver_ps.solver,
                                   raise_logger_warning=True, raise_error=True,
                                   **kwargs)

        # temporarily disable interactive_checks, check_default, and check_visible
        conf_interactive_checks = conf.interactive_checks
        if conf_interactive_checks:
            logger.debug("temporarily disabling interactive_checks")
            conf._interactive_checks = False

        conf_interactive_constraints = conf.interactive_constraints
        if conf_interactive_constraints:
            logger.debug("temporarily disabling interactive_constraints")
            conf._interactive_constraints = False


        def restore_conf():
            # restore user-set interactive checks
            if conf_interactive_checks:
                logger.debug("restoring interactive_checks={}".format(conf_interactive_checks))
                conf._interactive_checks = conf_interactive_checks

            if conf_interactive_constraints:
                logger.debug("restoring interactive_constraints={}".format(conf_interactive_constraints))
                conf._interactive_constraints = conf_interactive_constraints
                self.run_delayed_constraints()

        if kwargs.get('overwrite', solution=='latest') and solution in self.solutions:
            # NOTE: default (instead of detached_job=) is correct here
            if self.get_value(qualifier='detached_job', solution=solution, context='solution', default='loaded') not in ['loaded', 'error', 'killed']:
                raise ValueError("solution '{}' cannot be overwritten until it is complete and loaded.".format(solution))
            if solution=='latest':
                logger.warning("overwriting solution: {}".format(solution))
            else:
                logger.info("overwriting solution: {}".format(solution))

            overwrite_ps = self.remove_solution(solution=solution, auto_remove_figure=False, during_overwrite=True)

            # for solver backends that allow continuing, we need to keep and pass
            # the deleted PS if it matches continue_from
            if solver_ps.get_value(qualifier='continue_from', continue_from=kwargs.get('continue_from', None), default='None') == overwrite_ps.solution:
                kwargs['continue_from_ps'] = overwrite_ps

            # check the label again, just in case solution belongs to
            # something other than solution (technically could belong to model if 'latest')
            if solution!='latest':
                self._check_label(solution, allow_overwrite=False)
        else:
            overwrite_ps = None


        # now if we're supposed to detach we'll just prepare the job for submission
        # either in another subprocess or through some queuing system
        if detach and mpi.within_mpirun:
            logger.warning("cannot detach when within mpirun, ignoring")
            detach = False

        if (detach or mpi.enabled) and not mpi.within_mpirun:
            if detach:
                logger.warning("detach support is EXPERIMENTAL")

            # if kwargs.get('max_computations', None) is not None:
            #     # then we need to estimate computations in advance so we can
            #     # raise an error immediately
            #     logger.info("estimating number of computations to ensure not over max_computations={}".format(kwargs['max_computations']))
            #     for compute in computes:
            #         computeparams = self.get_compute(compute=compute)
            #
            #         if not computeparams.kind:
            #             raise KeyError("could not recognize backend from compute: {}".format(compute))
            #         compute_class = getattr(backends, '{}Backend'.format(computeparams.kind.title()))
            #         out = compute_class().get_packet_and_syns(self, compute, times=times, **kwargs)

            # we'll track everything through the solution name as well as
            # a random string, to avoid any conflicts
            jobid = kwargs.get('jobid', parameters._uniqueid())

            script_fname = "_{}.py".format(jobid)
            out_fname = "_{}.out".format(jobid)
            err_fname = "_{}.err".format(jobid)
            kill_fname = "_{}.kill".format(jobid)
            script_fname, out_fname = self._write_export_solver_script(script_fname, out_fname, solver, solution, False, False, None, kwargs)

            script_fname = os.path.abspath(script_fname)
            cmd = mpi.detach_cmd.format(script_fname)
            # TODO: would be nice to catch errors caused by the detached script...
            # but that would probably need to be the responsibility of the
            # jobparam to return a failed status and message.
            # Unfortunately right now an error just results in the job hanging.
            f = open(err_fname, 'w')
            subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=f)
            f.close()

            # create model parameter and attach (and then return that instead of None)
            job_param = JobParameter(self,
                                     location=os.path.dirname(script_fname),
                                     status_method='exists',
                                     retrieve_method='local',
                                     uniqueid=jobid)

            metawargs = {'context': 'solution', 'solution': solution}
            self._attach_params([job_param], check_copy_for=False, **metawargs)

            if isinstance(detach, str):
                self.save(detach)

            restore_conf()

            if not detach:
                return job_param.attach()
            else:
                logger.info("detaching from run_solver.  Call get_parameter(solution='{}').attach() to re-attach".format(solution))


            if kwargs.get('overwrite', solution=='latest') and return_changes and overwrite_ps is not None:
                return ParameterSet([job_param]) + overwrite_ps

            return job_param


        solver_class = getattr(_solverbackends, '{}Backend'.format(solver_ps.kind.title()))
        params = solver_class().run(self, solver_ps.solver, compute, solution=solution, **{k:v for k,v in kwargs.items() if k not in ['compute']})
        metawargs = {'context': 'solution',
                     'solver': solver_ps.solver,
                     'compute': compute,
                     'kind': solver_ps.kind,
                     'solution': solution}


        comment_param = StringParameter(qualifier='comments', value=kwargs.get('comments', solver_ps.get_value(qualifier='comments', default='', **_skip_filter_checks)), description='User-provided comments for this solution.  Feel free to place any notes here.')

        self._attach_params(params+[comment_param], check_copy_for=False, **metawargs)

        restore_conf()

        ret_ps = self.get_solution(solution=solution)
        ret_changes = self._run_solver_changes(ret_ps, return_changes=return_changes)

        if kwargs.get('overwrite', solution=='latest') and return_changes and overwrite_ps is not None:
            ret_ps += overwrite_ps


        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def _get_adopt_inds_uniqueids(self, solution_ps, **kwargs):

        adopt_parameters = solution_ps.get_value(qualifier='adopt_parameters', adopt_parameters=kwargs.get('adopt_parameters', kwargs.get('parameters', None)), expand=True, **_skip_filter_checks)
        fitted_uniqueids = solution_ps.get_value(qualifier='fitted_uniqueids', **_skip_filter_checks).tolist()
        fitted_twigs = solution_ps.get_value(qualifier='fitted_twigs', **_skip_filter_checks)
        # NOTE: all of these could have twig[index] notation

        b_uniqueids = self.uniqueids

        adoptable_ps = self.get_adjustable_parameters(exclude_constrained=False) + self.filter(qualifier='mask_phases', context='dataset', **_skip_filter_checks)
        if np.all([uniqueid.split('[')[0] in b_uniqueids for uniqueid in fitted_uniqueids]):
            fitted_ps = adoptable_ps.filter(uniqueid=[uniqueid.split('[')[0] for uniqueid in fitted_uniqueids], **_skip_filter_checks)
        else:
            logger.warning("not all uniqueids in fitted_uniqueids@{}@solution are still valid.  Falling back on twigs.  Save and load same bundle to prevent this extra cost.".format(solution_ps.solution))
            fitted_ps = adoptable_ps.filter(twig=[t.split('[')[0] for t in fitted_twigs.tolist()], **_skip_filter_checks)
            fitted_uniqueids = [fitted_ps.get_parameter(twig=fitted_twig, **_skip_filter_checks).uniqueid for fitted_twig in fitted_twigs]

        adopt_uniqueids = []
        for adopt_twig_orig in adopt_parameters:
            adopt_twig, index = _extract_index_from_string(adopt_twig_orig)
            fitted_ps_filtered = fitted_ps.filter(twig=adopt_twig, **_skip_filter_checks)
            if len(fitted_ps_filtered) == 1:
                puid = fitted_ps_filtered.get_parameter(**_skip_filter_checks).uniqueid
                adopt_uniqueids.append(puid if index is None else puid+'[{}]'.format(index))
            elif len(fitted_ps_filtered) > 1:
                raise ValueError("multiple valid matches found for adopt_parameter='{}'".format(adopt_twig))

        adopt_inds = [fitted_uniqueids.index(uniqueid) for uniqueid in adopt_uniqueids]

        # adopt_inds (index of the parameter in all fitted_* lists)
        # adopt_uniqueids (uniqueid to find the parameter, including [index])
        return adopt_inds, adopt_uniqueids

    def adopt_solution(self, solution=None,
                       adopt_parameters=None, adopt_distributions=None, adopt_values=None,
                       trial_run=False,
                       remove_solution=False, return_changes=False,  **kwargs):
        """

        Arguments
        ------------
        * `solution` (string, optional, default=None): label of the solution
            to adopt.  Must be provided if more than one are attached to the
            bundle.
        * `adopt_parameters` (list of strings, optional, default=None): which
            of the parameters exposed by the solution should be adopted.  If
            not provided or None, will default to the value of the `adopt_parameters`
            parameter in the solution.  If provided, twig matching will be done
            between the provided list and those in the `fitted_twigs` parameter
            in the solution.
        * `adopt_distributions` (bool, optional, default=None): Whether to
            adopt the distribution(s) from the solution.  If not provided or
            None, will default to the value of the `adopt_distributions`
            parameter in the solution.
        * `adopt_values` (bool, optional, default=None): whether to adopt the
            face-values from the solution.  If not provided or None, will default
            to the value of the `adopt_values` parameter in the solution.
        * `trial_run` (bool, optional, default=False): if set to True, the
            values in the bundle will not be set and distributions will not be
            attached, but the returned ParameterSet will show the proposed changes.
            This ParameterSet will be a copy of the relevant Parameters, and
            will no longer be attached to the Bundle.  Note that if `adopt_values`
            is True, the output will no longer contain values changed via constraints
            as the Parameters are no longer attached to the Bundle.  If
            `adopt_distributions` is True, `distribution_overwrite_all` will
            still apply permanently (under-the-hood the distributions are still
            attached but then removed before returning the copy).
        * `distribution` (string, optional, default=None): applicable only
            if `adopt_distributions=True` (or None and the `adopt_distributions`
            parameter in the solution is True).  Note that if `distribution`
            already exists in the Bundle, you must pass `distribution_overwrite_all=True`
            (support for appending to an existing distribution` is not allowed).
        * `distribution_overwrite_all` (bool, optional, default=False): whether
            to overwrite if `distribution` already exists.
        * `remove_solution` (bool, optional, default=False): whether to remove
            the `solution` once successfully adopted.  See <phoebe.frontend.bundle.Bundle.remove_solution>.
            Note that this will be permanent, even if `trial_run` is True.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        ----------

        Raises
        -----------
        * ValueError: if `distribution` is provided but the referenced `solution`
            does not expose distributions.
        """
        # make sure we don't pass distribution to the filter
        kwargs.setdefault('distribution',
                          self._default_label('dists',
                                              **{'context': 'distribution'}))
        distribution = kwargs.pop('distribution')
        distribution_overwrite_all = kwargs.pop('distribution_overwrite_all', False)

        solution_ps = self.get_solution(solution=solution, **kwargs)
        solver_kind = solution_ps.kind
        if solver_kind is None:
            raise ValueError("could not find solution='{}'".format(solution))

        if not (adopt_distributions is None or isinstance(adopt_distributions, bool)):
            raise TypeError("adopt_distributions must be None or bool.  To set the label of the resulting distribution, use distribution instead.")
        adopt_distributions = solution_ps.get_value(qualifier='adopt_distributions', adopt_distributions=adopt_distributions, **_skip_filter_checks)
        adopt_values = solution_ps.get_value(qualifier='adopt_values', adopt_values=adopt_values, **_skip_filter_checks)

        ret_changes = ParameterSet([])
        if adopt_distributions:
            if distribution in self.distributions:
                if distribution_overwrite_all:
                    ret_changes += self.remove_distribution(distribution=distribution)
                elif trial_run:
                    raise ValueError("distribution='{}' already exists.  Use a different label or pass distribution_overwrite_all=True (note that the existing distribution will be permanently removed even though trial_run=True)".format(distribution))
                else:
                    raise ValueError("distribution='{}' already exists.  Use a different label or pass distribution_overwrite_all=True".format(distribution))
            if distribution is not None and kwargs.pop('check_label', True):
                # for now we will do allow_overwrite... we'll check that logic later
                self._check_label(distribution, allow_overwrite=False)

        if not (adopt_distributions or adopt_values):
            raise ValueError('either adopt_distributions or adopt_values must be True for adopt_solution to do anything.')


        adopt_inds, adopt_uniqueids = self._get_adopt_inds_uniqueids(solution_ps, adopt_parameters=adopt_parameters)
        # NOTE: adopt_uniqueids now includes [index] notation, if applicable
        if not len(adopt_inds):
            raise ValueError('no (valid) parameters selected by adopt_parameters')

        fitted_units = solution_ps.get_value(qualifier='fitted_units', **_skip_filter_checks)

        user_interactive_constraints = conf.interactive_constraints
        conf.interactive_constraints_off(suppress_warning=True)

        changed_params = []
        constraint_revert_flip = {}

        if not kwargs.get('skip_checks', False):
            self.run_checks_solution(solution=solution_ps.solution,
                                     raise_logger_warning=True, raise_error=True,
                                     adopt_parameters=adopt_parameters,
                                     adopt_distributions=adopt_distributions,
                                     adopt_values=adopt_values,
                                     trial_run=trial_run,
                                     **kwargs)

        if adopt_values and not trial_run:
            # check to make sure no constraint issues
            for uniqueid in adopt_uniqueids:
                uniqueid, index = _extract_index_from_string(uniqueid)
                param = self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)
                if len(param.constrained_by):
                    constrained_by_ps = ParameterSet(param.constrained_by)
                    validsolvefor = [v for v in _constraint._validsolvefor.get(param.is_constraint.constraint_func, []) if param.qualifier not in v]
                    if len(validsolvefor) == 1:
                        solve_for = constrained_by_ps.get_parameter(twig=validsolvefor[0], **_skip_filter_checks)
                        constraint_revert_flip[solve_for.uniqueid] = param.uniqueid
                    else:
                        # NOTE: this case really should already have been handled and raised by run_checks_solution
                        # but since we have to create the mapping for parameters that can be handled automatically
                        # we might as well double check in case skip_checks=True
                        raise ValueError("cannot adopt value for {} as it is constrained by multiple parameters: {}.  Flip the constraint manually first, or remove {} from adopt_parameters.".format(param.twig, ", ".join([p.twig for p in param.constrained_by]), param.twig))

            for solve_for_uniqueid, constrained_uniqueid in constraint_revert_flip.items():
                logger.warning("temporarily flipping {} to solve for {}".format(self.get_parameter(uniqueid=constrained_uniqueid, **_skip_filter_checks).twig, self.get_parameter(uniqueid=solve_for_uniqueid, **_skip_filter_checks).twig))
                self.get_parameter(uniqueid=constrained_uniqueid, **_skip_filter_checks).is_constraint.flip_for(uniqueid=solve_for_uniqueid)

        if solver_kind in ['emcee', 'dynesty']:
            dist, _ = self.get_distribution_collection(solution=solution, context='solution', **{k:v for k,v in kwargs.items() if k in solution_ps.qualifiers})

            for i, uniqueid_orig in enumerate(adopt_uniqueids):
                uniqueid, index = _extract_index_from_string(uniqueid_orig)
                if adopt_distributions:
                    ps = self.add_distribution(uniqueid=uniqueid_orig, value=dist.slice(i), distribution=distribution, auto_add_figure=kwargs.get('auto_add_figure', None), check_label=distribution is not None)
                if adopt_values:
                    param = self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)
                    # TODO: what to do if constrained?
                    if trial_run:
                        if param.twig in [p.twig for p in changed_params]:
                            # then we've already made the copy, so let's grab it
                            param = changed_params[[p.twig for p in changed_params].index(param.twig)]
                        else:
                            param = param.copy()
                            changed_params.append(param)
                    else:
                        changed_params.append(param)

                    if index is None:
                        # NOTE: .median() is necesary over np.median() since its acting on a distl object
                        param.set_value(value=dist.slice(i).median(), unit=dist.slice(i).unit)
                    else:
                        param.set_index_value(index=index, value=dist.slice(i).median(), unit=dist.slice(i).unit)

        else:
            fitted_values = solution_ps.get_value(qualifier='fitted_values', **_skip_filter_checks)

            if solver_kind in ['lc_periodogram', 'rv_periodogram']:
                # only the period should be in fitted_values... if that changes,
                # we'll need more complex logic here
                fitted_values = fitted_values * solution_ps.get_value(qualifier='period_factor', period_factor=kwargs.get('period_factor', None), **_skip_filter_checks)

            if solver_kind in ['lc_geometry']:

                adopt_qualifiers = [self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks).qualifier for uniqueid in adopt_uniqueids]
                if 'mask_phases' in adopt_qualifiers and 't0_supconj' in adopt_qualifiers:
                    # then we need to shift mask phases by the phase-shift introduced by the change in t0_supconj
                    logger.info("shifting mask_phases by phase-shift caused by change in t0_supconj")

                    # we'll be making changes in memory and don't want to affect the parameter itself
                    fitted_values = _deepcopy(fitted_values)

                    mask_phases_ind = adopt_qualifiers.index('mask_phases')
                    t0_supconj_ind = adopt_qualifiers.index('t0_supconj')

                    t0_supconj_old = self.get_value(uniqueid=adopt_uniqueids[t0_supconj_ind], unit=u.d, **_skip_filter_checks)
                    t0_supconj_new = fitted_values[t0_supconj_ind]

                    phase_shift = self.to_phase(t0_supconj_new) - self.to_phase(t0_supconj_old)

                    fitted_values[mask_phases_ind] = [ph-phase_shift for ph in [ecl_ph for ecl_ph in fitted_values[mask_phases_ind]]]

            for uniqueid, value, unit in zip(adopt_uniqueids, fitted_values[adopt_inds], fitted_units[adopt_inds]):
                uniqueid, index = _extract_index_from_string(uniqueid)
                if adopt_distributions:
                    dist = _distl.delta(value, unit=unit)
                    ps = self.add_distribution(uniqueid=uniqueid, value=dist, distribution=distribution, auto_add_figure=kwargs.get('auto_add_figure', None), check_label=distribution is not None)
                if adopt_values:
                    param = self.get_parameter(uniqueid=uniqueid, **_skip_filter_checks)
                    if trial_run:
                        # NOTE: we can't compare uniqueids here since the copy will create a new one
                        if param.twig in [p.twig for p in changed_params]:
                            # then we've already made the copy, so let's grab it
                            param = changed_params[[p.twig for p in changed_params].index(param.twig)]
                        else:
                            param = param.copy()
                            changed_params.append(param)
                    else:
                        changed_params.append(param)

                    if index is not None:
                        param.set_index_value(index, value, unit=unit, force=trial_run)
                    else:
                        param.set_value(value, unit=unit, force=trial_run)


        changed_params += self.run_delayed_constraints()
        if user_interactive_constraints:
            conf.interactive_constraints_on()

        for constrained_uniqueid, solve_for_uniqueid in constraint_revert_flip.items():
            logger.warning("reverting {} to solve for {}".format(self.get_parameter(uniqueid=constrained_uniqueid, **_skip_filter_checks).twig, self.get_parameter(uniqueid=solve_for_uniqueid, **_skip_filter_checks).twig))
            self.get_parameter(uniqueid=constrained_uniqueid, **_skip_filter_checks).is_constraint.flip_for(uniqueid=solve_for_uniqueid)

        ret_ps = ParameterSet([])
        if adopt_distributions:
            # TODO: do we want to only return newly added distributions?
            dist_ps = self.get_distribution(distribution=distribution)
            if trial_run:
                ret_ps += dist_ps.copy()
                self.remove_distribution(distribution=distribution)
            else:
                ret_ps += dist_ps
        if adopt_values:
            ret_ps += changed_params

        if remove_solution:
            # TODO: add to the return if return_changes
            ret_changes += self.remove_solution(solution=solution)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def get_solution(self, solution=None, **kwargs):
        """
        Filter in the 'solution' context

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `solution`: (string, optional, default=None): the name of the solution
        * `**kwargs`: any other tags to do the filtering (excluding solution and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        if solution is not None:
            kwargs['solution'] = solution
            if solution not in self.solutions:
                raise ValueError("solution='{}' not found".format(solution))
        kwargs['context'] = 'solution'
        return self.filter(**kwargs)

    def rerun_solution(self, solution=None, **kwargs):
        """
        Rerun run_solver for a given solution.  This simply retrieves the current
        solver/compute parameters given the same solver/compute label used to
        create the original solution.  This does not, therefore, necessarily
        ensure that the exact same solver/compute options are used.

        See also:
        * <phoebe.frontend.bundle.Bundle.run_solver>

        Arguments
        ------------
        * `solution` (string, optional): label of the solution (will be overwritten)
        * `**kwargs`: all keyword arguments are passed directly to
            <phoebe.frontend.bundle.Bundle.run_compute>

        Returns
        ------------
        * the output from <phoebe.frontend.bundle.Bundle.run_solver>
        """
        solution_ps = self.get_solution(solution=solution)

        solver = solution_ps.solver
        kwargs.setdefault('solver', solver)

        compute = solution_ps.compute
        kwargs.setdefault('compute', compute)

        return self.run_solver(solution=solution, **kwargs)

    def import_solution(self, fname, solution=None, overwrite=False, return_changes=False):
        """
        Import and attach a solution from a file.

        Generally this file will be the output after running a script generated
        by <phoebe.frontend.bundle.Bundle.export_solver>.  This is NOT necessary
        to be called if generating a solution directly from
        <phoebe.frontend.bundle.Bundle.run_solver>.

        See also:
        * <phoebe.frontend.bundle.Bundle.export_solver>

        Arguments
        ------------
        * `fname` (string): the path to the file containing the solution.  Likely
            `out_fname` from <phoebe.frontend.bundle.Bundle.export_compute>.
            Alternatively, this can be the json of the solution.  Must be
            able to be parsed by <phoebe.parameters.ParameterSet.open>.
        * `solution` (string, optional): the name of the solution to be attached
            to the Bundle.  If not provided, the solution will be adopted from
            the tags in the file.
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        -----------
        * ParameterSet of added and changed parameters
        """
        result_ps = ParameterSet.open(fname)
        if 'progress' in result_ps.qualifiers:
            progress = result_ps.get_value(qualifier='progress', **_skip_filter_checks)
            value = 'progress:{}%'.format(np.round(progress, 2)) if progress < 100 else 'completed'
            job_param = StringParameter(qualifier='imported_job',
                                        value=value,
                                        readonly=True,
                                        description='imported solution')

            job_param._context = 'solution'
            job_param._solution = result_ps.solution
            job_param._compute = result_ps.compute

            result_ps += [job_param]



        metawargs = {}
        if solution is None:
            solution = result_ps.solution

        metawargs['solution'] = solution

        ret_changes = []
        new_uniqueids = False
        if solution in self.solutions:
            if overwrite:
                ret_changes += self.remove_solution(solution, during_overwrite=True, return_changes=return_changes).to_list()
                new_uniqueids = True
            else:
                raise ValueError("solution '{}' already exists.  Use different name or pass overwrite=True".format(solution))

        self._attach_params(result_ps, override_tags=True, new_uniqueids=new_uniqueids, **metawargs)

        ret_ps = self.get_solution(solution=solution if solution is not None else result_ps.solutions)

        # attempt to map fitted_twigs -> fitted_uniqueids if not all match now, to prevent having to continuously repeat
        fitted_uniqueids = ret_ps.get_value(qualifier='fitted_uniqueids', **_skip_filter_checks)
        b_uniqueids = self.uniqueids
        if not np.all([u in b_uniqueids for u in fitted_uniqueids]):
            _, fitted_uniquieds = self._get_adopt_inds_uniqueids(ret_ps, adopt_parameters='*')
            ret_ps.set_value(qualifier='fitted_uniqueids', value=fitted_uniquieds, ignore_readonly=True, **_skip_filter_checks)

        ret_changes += self._run_solver_changes(ret_ps, return_changes=return_changes)

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    @send_if_client
    def remove_solution(self, solution, return_changes=False, **kwargs):
        """
        Remove a 'solution' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `solution` (string): the label of the solution to be removed.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: solution, context.

        Returns
        -----------
        * ParameterSet of removed or changed parameters
        """
        kwargs['solution'] = solution
        kwargs['context'] = 'solution'
        ret_ps = self.remove_parameters_all(**kwargs)
        ret_changes = self._run_solver_changes(ret_ps,
                                               return_changes=return_changes,
                                               removed=True,
                                               auto_remove_figure=kwargs.get('auto_remove_figure', None),
                                               during_overwrite=kwargs.get('during_overwrite', False))

        if return_changes:
            ret_ps += ret_changes
        return _return_ps(self, ret_ps)

    def remove_solutions_all(self, return_changes=False, **kwargs):
        """
        Remove all solutions from the bundle.  To remove a single solution see
        <phoebe.frontend.bundle.Bundle.remove_solution>.

        Arguments
        -----------
        * `remove_figure_params` (bool, optional): whether to also remove
            figure options tagged with `solution`.  If not provided, will default
            to false if `solution` is 'latest', otherwise will default to True.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for solution in self.solutions:
            removed_ps += self.remove_solution(solution=solution, return_changes=return_changes, **kwargs)
        return removed_ps

    @send_if_client
    def rename_solution(self, old_solution, new_solution,
                        overwrite=False, return_changes=False):
        """
        Change the label of a solution attached to the Bundle.

        Arguments
        ----------
        * `old_solution` (string): current label of the solution (must exist)
        * `new_solution` (string): the desired new label of the solution
            (must not yet exist, unless `overwrite=True`)
        * `overwrite` (bool, optional, default=False): overwrite the existing
            entry if it exists.

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed solution

        Raises
        --------
        * ValueError: if the value of `new_solution` is forbidden or already exists.
        """
        # TODO: raise error if old_solution not found?
        self._rename_label('solution', old_solution, new_solution, overwrite)

        ret_ps = self.filter(solution=new_solution)

        return _return_ps(self, ret_ps)
