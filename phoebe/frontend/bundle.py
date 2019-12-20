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
from datetime import datetime
from distutils.version import StrictVersion

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
from phoebe.parameters import constraint as _constraint
from phoebe.parameters import feature as _feature
from phoebe.parameters import figure as _figure
from phoebe.parameters.parameters import _uniqueid
from phoebe.backend import backends, mesh
from phoebe.distortions import roche
from phoebe.frontend import io
from phoebe.atmospheres.passbands import list_installed_passbands, list_online_passbands, get_passband, update_passband, _timestamp_to_dt
from phoebe.utils import _bytes, parse_json
import libphoebe

from phoebe import u
from phoebe import conf, mpi
from phoebe import __version__

import logging
logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())

if sys.version_info[0] == 3:
  unicode = str
  from io import IOBase

_bundle_cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_bundles'))+'/'

_clientid = 'python-'+_uniqueid(5)

_skip_filter_checks = {'check_default': False, 'check_visible': False}

# Attempt imports for client requirements
try:
    import requests
    import urllib2
    from socketIO_client import SocketIO, BaseNamespace
    #  pip install -U socketIO-client
except ImportError:
    _can_client = False
else:
    _can_client = True


def _get_add_func(mod, func, return_none_if_not_found=False):
    if isinstance(func, unicode):
        func = str(func)

    if isinstance(func, str) and hasattr(mod, func):
        func = getattr(mod, func)

    if hasattr(func, '__call__'):
        return func
    elif return_none_if_not_found:
        return None
    else:
        raise ValueError("could not find callable function in {}.{}"
                         .format(mod, func))


class RunChecksItem(object):
    def __init__(self, b, message, param_uniqueids=[], fail=True):
        self._b = b
        self._message = message
        self._fail = fail
        self._param_uniqueids = [uid.uniqueid if isinstance(uid, Parameter) else uid for uid in param_uniqueids]

    def __repr__(self):
        return "<RunChecksItem level={} message={} parameters: {}>".format(self.level, self.message, len(self._param_uniqueids))

    def __str__(self):
        n_affected_parameters = len(self._param_uniqueids)
        return "{}: {} ({} affected parameter{})".format(self.level, self.message, n_affected_parameters, "s" if n_affected_parameters else "")

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
        * <phoebe.frontend.bundle.RunChecksItem.to_dict>

        Returns
        ---------
        * (bool) whether this item is an error that will cause the checks to fail.
        """
        return self._fail

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

    def add_item(self, b, message, param_uniqueids=[], fail=True):
        """
        Add a new <phoebe.frontend.bundle.RunChecksItem> to this report.
        Generally this should not be done manually, but is handled internally
        by <phoebe.frontend.bundle.Bundle.run_checks>.

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
        """
        self._items.append(RunChecksItem(b, message, param_uniqueids, fail))

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

        # set to be not a client by default
        self._is_client = False
        self._last_client_update = None
        self._lock = False

        # handle delayed constraints when interactive mode is off
        self._delayed_constraints = []
        self._failed_constraints = []

        if not len(params):
            # add position (only 1 allowed and required)
            self._attach_params(_system.system(), context='system')

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

        self._mplcolorcycler = _figure.MPLPropCycler('color', _figure._mplcolors)
        self._mplmarkercycler = _figure.MPLPropCycler('marker', _figure._mplmarkers)
        self._mpllinestylecycler = _figure.MPLPropCycler('linestyle', _figure._mpllinestyles)

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
        if io._is_file(filename):
            f = filename
        elif isinstance(filename, str) or isinstance(filename, unicode):
            filename = os.path.expanduser(filename)
            logger.debug("importing from {}".format(filename))
            f = open(filename, 'r')
        elif isinstance(filename, list):
            # we'll handle later
            pass
        else:
            raise TypeError("filename must be string, unicode, or file object, got {}".format(type(filename)))

        if isinstance(filename, list):
            data = filename
        else:
            data = json.load(f, object_pairs_hook=parse_json)
            f.close()

        b = cls(data)

        version = b.get_value(qualifier='phoebe_version', check_default=False, check_visible=False)
        phoebe_version_import = StrictVersion(version if version != 'devel' else '2.2.0')
        phoebe_version_this = StrictVersion(__version__ if __version__ != 'devel' else '2.2.0')

        logger.debug("importing from PHOEBE v {} into v {}".format(phoebe_version_import, phoebe_version_this))

        # update the entry in the PS, so if this is saved again it will have the new version
        b.set_value(qualifier='phoebe_version', value=__version__, check_default=False, check_visible=False)

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
            b.remove_parameters_all(context='setting')
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
                        existing_values['pblum_component'] = b.filter(qualifier='pblum_ref', context='dataset', dataset=ds, check_visible=False).exclude(value='self', check_visible=False).get_parameter(check_visible=False).component


                for qualifier in b.filter(context='dataset', dataset=ds, check_visible=False, check_default=False).qualifiers:
                    if qualifier in ['pblum_ref']:
                        # already handled these above
                        continue
                    ps = b.filter(qualifier=qualifier, context='dataset', dataset=ds, check_visible=False)
                    if len(ps.to_list()) > 1:
                        existing_values[qualifier] = {}
                        for param in ps.to_list():
                            existing_values[qualifier]["{}@{}".format(param.time, param.component) if param.time is not None else param.component] = existing_value(param)
                            if qualifier=='ld_func':
                                if 'ld_mode' not in existing_values.keys():
                                    existing_values['ld_mode'] = {}
                                existing_values['ld_mode']["{}@{}".format(param.time, param.component) if param.time is not None else param.component] = 'interp' if param.value == 'interp' else 'manual'

                    else:
                        param = b.get_parameter(qualifier=qualifier, context='dataset', dataset=ds, check_visible=False, check_default=False)
                        existing_values[qualifier] = existing_value(param)
                        if qualifier=='ld_func':
                            existing_values['ld_mode']["{}@{}".format(param.time, param.component) if param.time is not None else param.component] = 'interp' if param.value == 'interp' else 'manual'

                if ds_kind in ['lp']:
                    # then we need to pass the times from the attribute instead of parameter
                    existing_values['times'] = b.filter(context='dataset', dataset=ds, check_visible=False, check_default=False).times

                existing_values['kind'] = ds_kind

                existing_values_per_ds[ds] = existing_values
                b.remove_dataset(dataset=ds)

            for ds, existing_values in existing_values_per_ds.items():
                ds_kind = existing_values.pop('kind')
                logger.warning("migrating '{}' {} dataset.".format(ds, ds_kind))
                logger.debug("applying existing values to {} dataset: {}".format(ds, existing_values))
                b.add_dataset(ds_kind, dataset=ds, overwrite=True, **existing_values)

            for component in b.filter(context='component', kind='star').components:
                existing_values = {p.qualifier: p.get_value() for p in b.filter(context='component', component=component).to_list()}
                logger.warning("migrating '{}' component".format(component))
                logger.debug("applying existing values to {} component: {}".format(component, existing_values))
                b.add_component(kind='star', component=component, overwrite=True, **existing_values)

            # make sure constraints all attach
            b.set_hierarchy()

            logger.debug("restoring previous models")
            b._attach_params(ps_model, context='model')

        if phoebe_version_import < StrictVersion("2.1.2"):
            b._import_before_v211 = True
            warning = "importing from an older version ({}) of PHOEBE which did not support constraints in solar units.  All constraints will remain in SI, but calling set_hierarchy will likely fail.".format(phoebe_version_import)
            print("WARNING: {}".format(warning))
            logger.warning(warning)

        if phoebe_version_import < StrictVersion("2.1.0"):
            logger.warning("importing from an older version ({}) of PHOEBE into version {}".format(phoebe_version_import, phoebe_version_this))

            def _ps_dict(ps):
                return {p.qualifier: p.get_quantity() if hasattr(p, 'get_quantity') else p.get_value() for p in ps.to_list()}

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
        * <phoebe.frontend.bundle.Bundle.client_update>

        Arguments
        ----------
        * `bundleid` (string): the identifier given to the bundle by the
            server.
        * `server` (string, optional, default='http://localhost:5555'): the
            host (and port) of the server.
        * `as_client` (bool, optional, default=True):  whether to attach in
            client mode.  See <phoebe.frontend.bundle.Bundle.as_client>.
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
            b.as_client(as_client, server=server,
                        bundleid=rjson['meta']['bundleid'],
                        start_if_fail=False)

            logger.warning("This bundle is in client mode, meaning all computations will be handled by the server at {}.  To disable client mode, call as_client(False) or in the future pass as_client=False to from_server".format(server))

        return b

    @classmethod
    def from_legacy(cls, filename, add_compute_legacy=True, add_compute_phoebe=True):
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
            <phoebe.parameters.compute.phoebe> to add manually after

        Returns
        ---------
        * an instantiated <phoebe.frontend.bundle.Bundle> object.
        """
        logger.warning("importing from legacy is experimental until official 1.0 release")
        return io.load_legacy(filename, add_compute_legacy, add_compute_phoebe)

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
                b.rename_component('binary', 'orbit')

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
        b.add_star(component=starB, color='red', **star_defaults)
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
        b.add_star(component=starB, color='red')
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

    def save(self, filename, clear_history=True, incl_uniqueid=False,
             compact=False):
        """
        Save the bundle to a JSON-formatted ASCII file.

        See also:
        * <phoebe.parameters.ParameterSet.save>
        * <phoebe.parameters.Parameter.save>

        Arguments
        ------------
        * `filename` (string): relative or full path to the file
        * `clear_history` (bool, optional, default=True): whether to clear
            history log items before saving.
        * `incl_uniqueid` (bool, optional, default=False): whether to including
            uniqueids in the file (only needed if its necessary to maintain the
            uniqueids when reloading)
        * `compact` (bool, optional, default=False): whether to use compact
            file-formatting (may be quicker to save/load, but not as easily readable)

        Returns
        -------------
        * the filename (string)
        """
        if clear_history:
            # TODO: let's not actually clear history,
            # but rather skip the context when saving
            self.remove_history()

        # TODO: add option for clear_models, clear_feedback
        # NOTE: PS.save will handle os.path.expanduser
        return super(Bundle, self).save(filename, incl_uniqueid=incl_uniqueid,
                                        compact=compact)

    def export_legacy(self, filename, compute=None, skip_checks=False):
        """
        Export the Bundle to a file readable by PHOEBE legacy.

        NEW IN PHOEBE 2.2.

        See also:
        * <phoebe.parameters.compute.legacy>

        Arguments
        -----------
        * `filename` (string): relative or full path to the file
        * `compute` (string, optional, default=None): label of the compute options
            to use while exporting.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks> before exporting.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.

        Returns
        ------------
        * the filename (string)
        """
        logger.warning("exporting to legacy is experimental until official 1.0 release")
        self.run_delayed_constraints()

        if not skip_checks:
            report = self.run_checks(compute=compute, allow_skip_constraints=False,
                                     raise_logger_warning=True, raise_error=True)

        filename = os.path.expanduser(filename)
        return io.pass_to_legacy(self, filename, compute=compute)


    def _test_server(self, server='http://localhost:5555', start_if_fail=True):
        """
        [NOT IMPLEMENTED]
        """
        try:
            resp = urllib2.urlopen("{}/info".format(server))
        except urllib2.URLError:
            test_passed = False
        else:
            resp = json.loads(resp.read())
            test_passed = resp['data']['success']

        if not test_passed and \
                start_if_fail and \
                'localhost' in re.sub(r'[\/\:]', ' ', server).split():
            raise NotImplementedError("start_if_fail not yet supported - manually start server")
            return False

        return test_passed

    def _on_socket_connect(self, *args):
        """
        [NOT IMPLEMENTED]
        """
        logger.info("connected to server")

    def _on_socket_disconnect(self, *args):
        """
        [NOT IMPLEMENTED]
        test
        """
        logger.warning("disconnected from server")

    def _on_socket_push_updates(self, resp):
        """
        [NOT IMPLEMENTED]
        """
        # TODO: check to make sure resp['meta']['bundleid']==bundleid ?
        # TODO: handle added parameters
        # TODO: handle removed (isDeleted) parameters

        # resp['data'] = {'success': True/False, 'parameters': {uniqueid: {context: 'blah', value: 'blah', ...}}}
        # print("*** _on_socket_push_updates resp={}".format(resp))
        for uniqueid, paramdict in resp['parameters'].items():
            # print("*** _on_socket_push_updates uniquide in uniqueids={}, paramdict={}".format(uniqueid in self.uniqueids, paramdict))
            if uniqueid in self.uniqueids:
                param = self.get_parameter(uniqueid=uniqueid, check_visible=False, check_default=False, check_advanced=False)
                for attr, value in paramdict.items():
                    if hasattr(param, "_{}".format(attr)):
                        logger.info("updates from server: setting {}@{}={}".
                                    format(attr, param.twig, value))

                        # we cannot call param.set_value because that will
                        # emit another signal to the server.  So we do need
                        # to hardcode some special cases here
                        if isinstance(value, dict):
                            if 'nparray' in value.keys():
                                value = nparray.from_json(value)

                        if attr == 'value' and hasattr(param, 'default_unit'):
                            if 'default_unit' in paramdict.keys():
                                unit = u.Unit(paramdict.get('default_unit'))
                            else:
                                unit = param.default_unit
                            value = value * unit

                        setattr(param, "_{}".format(attr), value)
            else:
                self._attach_param_from_server(paramdict)


    def _attach_param_from_server(self, item):
        """
        [NOT IMPLEMENTED]
        """
        if isinstance(item, list):
            for itemi in item:
                self._attach_param_from_server(itemi)
        else:
            # then we need to add a new parameter
            d = item

            print("*** _attach_param_from_server d={}".format(d))

            d['Class'] = d.pop('type')
            for attr, value in d.pop('attributes', {}).items():
                d[attr] = value
            for tag, value in d.pop('meta', {}).items():
                d[tag] = value

            _dump = d.pop('readonly', None)
            _dump = d.pop('valuestr', None)
            _dump = d.pop('twig', None)
            _dump = d.pop('valueunit', None)  # TODO: may need to account for unit?

            # print "*** _attach_param_from_server", d
            param = parameters.parameter_from_json(d, bundle=self)

            metawargs = {}
            self._attach_params([param], **metawargs)

    def _deregister_client(self, bundleid=None):
        if self._socketio is None:
            return

        logger.info("deregistering {} client from {}".format(_clientid, self.is_client))
        self._socketio.emit('deregister client', {'clientid': _clientid, 'bundleid': None})
        if bundleid is not None:
            self._socketio.emit('deregister client', {'clientid': _clientid, 'bundleid': bundleid})
        self._socketio.disconnect()
        self._socketio = None

    def as_client(self, as_client=True, server='http://localhost:5555',
                  bundleid=None, start_if_fail=True):
        """
        Enter (or exit) client mode.

        See also:
        * <phoebe.frontend.bundle.Bundle.from_server>
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.is_client>
        * <phoebe.frontend.bundle.Bundle.client_update>

        Arguments
        -----------
        * `as_client` (bool, optional, default=True): whether to enter (True)
            or exit (False) client mode.
        * `server` (string, optional, default='http://localhost:5555'): the URL
            location (including port, if necessary) to find the phoebe-server.
        * `bundleid` (string, optional, default=None): if provided and the
            bundleid is available from the given server, that bundle will be
            downloaded and attached.  If provided but bundleid is not available
            from the server, the current bundle will be uploaded and assigned
            the given bundleid.  If not provided, the current bundle will be
            uploaded and assigned a random bundleid.
        * `start_if_fail` (bool, optional, default=True): NOT CURRENTLY IMPLEMENTED

        Raises
        ---------
        * ImportError: if required dependencies for client mode are not met.
        * ValueError: if the server at `server` is not running or reachable.
        * ValueError: if the server returns an error.
        """
        if not conf.devel:
            raise NotImplementedError("'as_client' not officially supported for this release.  Enable developer mode to test.")

        if as_client:
            if not _can_client:
                raise ImportError("dependencies to support client mode not met - see docs")

            server_running = self._test_server(server=server,
                                               start_if_fail=start_if_fail)
            if not server_running:
                raise ValueError("server {} is not running".format(server))

            server_split = server.split('://')[-1].split(':')
            host = ':'.join(server_split[:-1]) if len(server_split) > 1 else server_split[0]
            port = int(float(server_split[-1])) if len(server_split) > 1 else None
            self._socketio = SocketIO(host, port, BaseNamespace)
            self._socketio.on('connect', self._on_socket_connect)
            self._socketio.on('disconnect', self._on_socket_disconnect)

            if bundleid is not None:
                rj = requests.get("{}/info".format(server)).json()
                if bundleid in rj['data']['clients_per_bundle'].keys():
                    upload = False
                else:
                    upload = True
            else:
                upload = True

            if upload:
                upload_url = "{}/open_bundle".format(server)
                logger.info("uploading bundle to server {}".format(upload_url))
                data = json.dumps({'json': self.to_json(incl_uniqueid=True), 'bundleid': bundleid})
                rj = requests.post(upload_url, data=data, timeout=5).json()
                if rj['data']['success']:
                    bundleid = rj['data']['bundleid']
                else:
                    raise ValueError("server error: {}".format(rj['data'].get('error', 'unknown error')))

            self._socketio.emit('register client', {'clientid': _clientid, 'bundleid': bundleid})

            self._socketio.on('{}:changes:python'.format(bundleid), self._on_socket_push_updates)

            self._bundleid = bundleid

            self._is_client = server

            atexit.register(self._deregister_client)

            logger.info("connected as client {} to server at {}:{}".
                        format(_clientid, host, port))

        else:
            logger.warning("This bundle is now permanently detached from the instance on the server and will not receive future updates.  To start a client in sync with the version on the server or other clients currently subscribed, you must instantiate a new bundle with Bundle.from_server.")

            if hasattr(self, '_socketio') and self._socketio is not None:
                self._deregister_client(bundleid=self._bundleid)

            self._bundleid = None
            self._is_client = False

    @property
    def is_client(self):
        """
        See also:
        * <phoebe.frontend.bundle.Bundle.from_server>
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.as_client>
        * <phoebe.frontend.bundle.Bundle.client_update>

        Returns
        ---------
        * False if the bundle is not in client mode, otherwise the URL of the server.
        """
        return self._is_client

    def client_update(self):
        """
        Check for updates from the server and update the client.  In general,
        it should not be necessary to call this manually.

        See also:
        * <phoebe.frontend.bundle.Bundle.from_server>
        * <phoebe.parameters.ParameterSet.ui>
        * <phoebe.frontend.bundle.Bundle.as_client>
        * <phoebe.frontend.bundle.Bundle.is_client>

        """
        if not conf.devel:
            raise NotImplementedError("'client_update' not officially supported for this release.  Enable developer mode to test.")

        if not self.is_client:
            raise ValueError("Bundle is not in client mode, cannot update")

        logger.info("updating client...")
        # wait briefly to pickup any missed messages, which should then fire
        # the corresponding callbacks and update the bundle
        self._socketio.wait(seconds=0.1)
        self._last_client_update = datetime.now()

    def __repr__(self):
        return super(Bundle, self).__repr__().replace('ParameterSet', 'PHOEBE Bundle')

    def __str__(self):
        return super(Bundle, self).__str__().replace('ParameterSet', 'PHOEBE Bundle')

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

    def _rename_label(self, tag, old_value, new_value):
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

            affected_params += self._handle_dataset_selectparams(rename={old_value: new_value}, return_changes=True)
            affected_params += self._handle_figure_time_source_params(rename={old_value: new_value}, return_changes=True)

        elif tag=='component':
            affected_params += self._handle_component_selectparams(rename={old_value: new_value}, return_changes=True)
            affected_params += self._handle_pblum_defaults(rename={old_value: new_value}, return_changes=True)

        elif tag=='compute':
            affected_params += self._handle_compute_selectparams(rename={old_value: new_value}, return_changes=True)

        elif tag=='model':
            affected_params += self._handle_model_selectparams(rename={old_value: new_value}, return_changes=True)

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

    def _add_history(self, redo_func, redo_kwargs, undo_func, undo_kwargs,
                     **kwargs):
        """
        Add a new log (undo/redoable) to this history context.

        Arguments
        -----------
        * `redo_func` (str): function to redo the action, must be a
            method of <phoebe.frontend.bundle.Bundle>
        * `redo_kwargs` (dict):  kwargs to pass to the redo_func.  Each
            item must be serializable (float or str, not objects)
        * `undo_func` (str): function to undo the action, must be a
            method of <phoebe.frontend.bundle.Bundle>
        * `undo_kwargs` (dict): kwargs to pass to the undo_func.  Each
            item must be serializable (float or str, not objects)
        * `history` (string, optional): label of the history parameter

        Raises
        -------
        * ValueError: if the label for this history item is forbidden or
            already exists
        """
        if not self.history_enabled:
            return

        param = HistoryParameter(self, redo_func, redo_kwargs,
                                 undo_func, undo_kwargs)

        metawargs = {'context': 'history',
                     'history': kwargs.get('history', self._default_label('hist', **{'context': 'history'}))}

        self._check_label(metawargs['history'])

        self._attach_params([param], **metawargs)

    @property
    def history(self):
        """
        Property as a shortcut to <phoebe.frontend.bundle.Bundle.get_history>

        You can toggle whether history is recorded using:
        * <phoebe.frontend.bundle.Bundle.enable_history>
        * <phoebe.frontend.bundle.Bundle.disable_history>
        """

        return self.get_history()

    def get_history(self, i=None):
        """
        Get a history item by index.

        You can toggle whether history is recorded using:
        * <phoebe.frontend.bundle.Bundle.enable_history>
        * <phoebe.frontend.bundle.Bundle.disable_history>

        Arguments
        ----------
        * `i` (integer, optional, default=None): integer for indexing (can be
            positive or negative).  If i is None or not provided, the entire list
            of history items will be returned

        Returns
        ----------
        * <phoebe.parameters.Parameter> if `i` is an int, or
            <phoebe.parameters.ParameterSet> if `i` is None (or not provided).

        Raises
        -------
        * ValueError: if no history items have been recorded.
        """
        ps = self.filter(context='history')
        # if not len(ps):
        #    raise ValueError("no history recorded")

        if i is not None:
            return ps.to_list()[i]
        else:
            return ps  # TODO: reverse the order?

    def remove_history(self, i=None):
        """
        Remove a history item from the bundle by index.

        You can toggle whether history is recorded using:
        * <phoebe.frontend.bundle.Bundle.enable_history>
        * <phoebe.frontend.bundle.Bundle.disable_history>


        Arguments
        ----------
        * `i` (integer, optional, default=None): integer for indexing (can be
            positive or negative).  If i is None or not provided, the entire list
            of history items will be removed.

        Returns
        -----------
        * ParameterSet of removed parameters

        Raises
        -------
        * ValueError: if no history items have been recorded.
        """
        if i is None:
            return_ = self.remove_parameters_all(context='history')
        else:
            param = self.get_history(i=i)
            return_ = self.remove_parameter(uniqueid=param.uniqueid)

        # let's not add_history for this one...
        return return_

    @property
    def history_enabled(self):
        """
        Property as a shortcut to `b.get_setting('log_history).get_value()``.

        You can toggle whether history is recorded using:
        * <phoebe.frontend.bundle.Bundle.enable_history>
        * <phoebe.frontend.bundle.Bundle.disable_history>

        Returns
        ------
        * (bool) whether logging of history items (undo/redo) is enabled.
        """
        return self.get_setting('log_history').get_value()\
            if len(self.get_setting())\
            else False

    def enable_history(self):
        """
        Enable logging history items (undo/redo).

        You can check wither history is enabled using
        <phoebe.frontend.bundle.Bundle.history_enabled>.

        Shortcut to `b.get_setting('log_history').set_value(True)`
        """
        self.get_setting('log_history').set_value(True)

    def disable_history(self):
        """
        Disable logging history items (undo/redo)

        You can check wither history is enabled using
        <phoebe.frontend.bundle.Bundle.history_enabled>.

        Shortcut to `b.get_setting('log_history').set_value(False)`
        """
        self.get_setting('log_history').set_value(False)

    def undo(self, i=-1):
        """
        Undo an item in the history logs

        Arguments
        ----------
        * `i` (integer, optional, default=-1): integer for indexing (can be
            positive or negative).

        Raises
        ----------
        * ValueError: if no history items have been recorded
        """

        _history_enabled = self.history_enabled
        param = self.get_history(i)
        self.disable_history()
        param.undo()
        # TODO: do we really want to remove this?  then what's the point of redo?
        self.remove_parameter(uniqueid=param.uniqueid)
        if _history_enabled:
            self.enable_history()

    def redo(self, i=-1):
        """
        Redo an item in the history logs

        Arguments
        ----------
        * `i` (integer, optional, default=-1): integer for indexing (can be
            positive or negative).

        Raises
        ----------
        * ValueError: if no history items have been recorded
        """
        _history_enabled = self.history_enabled
        param = self.get_history(i)
        self.disable_history()
        param.redo()
        self.remove_parameter(uniqueid=param.uniqueid)
        if _history_enabled:
            self.enable_history()

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

        dss_ps = self.filter(context='dataset', check_default=False, check_visible=False)

        pbdep_datasets = dss_ps.filter(kind=_dataset._pbdep_columns.keys(),
                                       check_default=False, check_visible=False).datasets

        pbdep_columns = _dataset._mesh_columns[:] # force deepcopy
        for pbdep_dataset in pbdep_datasets:
            pbdep_kind = dss_ps.filter(dataset=pbdep_dataset,
                                       kind=_dataset._pbdep_columns.keys(),
                                       check_default=False, check_visible=False).kind

            pbdep_columns += ["{}@{}".format(column, pbdep_dataset) for column in _dataset._pbdep_columns[pbdep_kind]]

        time_datasets = dss_ps.exclude(kind='mesh').datasets

        t0s = ["{}@{}".format(p.qualifier, p.component) for p in self.filter(qualifier='t0*', context=['component']).to_list()]
        t0s += ["t0@system"]

        for param in dss_ps.filter(qualifier='columns', check_default=False, check_visible=False).to_list():
            choices_changed = False
            if return_changes and pbdep_columns != param._choices:
                choices_changed = True
            param._choices = pbdep_columns
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        for param in dss_ps.filter(qualifier='include_times', check_default=False, check_visible=False).to_list():

            # NOTE: existing value is updated in change_component
            choices_changed = False
            if return_changes and time_datasets+t0s != param._choices:
                choices_changed = True
            param._choices = time_datasets + t0s
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        for param in self.filter(context='figure', qualifier='datasets', check_default=False, check_visible=False).to_list():
            ds_same_kind = self.filter(context='dataset', kind=param.kind).datasets

            choices_changed = False
            if return_changes and ds_same_kind != param._choices:
                choices_changed = True
            param._choices = ds_same_kind
            changed = param.handle_choice_rename(remove_not_valid=True, **rename)
            if return_changes and (changed or choices_changed):
                affected_params.append(param)

        return affected_params

    def _handle_figure_time_source_params(self, rename={}, return_changes=False):
        affected_params = []

        t0s = ["{}@{}".format(p.qualifier, p.component) for p in self.filter(qualifier='t0*', context=['component']).to_list()]
        t0s += ["t0@system"]

        # here we have to use context='dataset' otherwise pb-dependent parameters
        # with context='model', kind='mesh' will show up
        valid_datasets = self.filter(context='dataset', kind=['mesh', 'lp'], check_visible=False).datasets

        mesh_times = []
        lp_times = []
        mesh_lp_times = []
        for t in self.filter(context='model', kind='mesh').times:
            mesh_times.append('{} ({})'.format(t, ', '.join(ds for ds in self.filter(context='model', time=t).datasets if ds in valid_datasets)))
        for t in self.filter(context='model', kind='lp').times:
            lp_times.append('{} ({})'.format(t, ', '.join(ds for ds in self.filter(context='model', time=t).datasets if ds in valid_datasets)))
        for t in self.filter(context='model').times:
            mesh_lp_times.append('{} ({})'.format(t, ', '.join(ds for ds in self.filter(context='model', time=t).datasets if ds in valid_datasets)))

        for param in self.filter(context='figure', qualifier=['default_time_source', 'time_source'], check_default=False, check_visible=False).to_list():


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

    def _handle_compute_selectparams(self, rename={}, return_changes=False):
        """
        """
        affected_params = []
        changed_params = self.run_delayed_constraints()

        computes = self.filter(context='compute', check_default=False, check_visible=False).computes

        for param in self.filter(qualifier='run_checks_compute', check_default=False, check_visible=False).to_list():
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
        mesh_datasets = self.filter(context='dataset', kind='mesh').datasets

        choices = ['None']
        for p in self.filter(context='model', kind='mesh', check_visible=False).exclude(qualifier=ignore, check_visible=False).to_list():
            item = p.qualifier if p.dataset in mesh_datasets else '{}@{}'.format(p.qualifier, p.dataset)
            if item not in choices:
                choices.append(item)

        for param in self.filter(context='figure', qualifier=['fc_column', 'ec_column'], check_default=False, check_visible=False).to_list():
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

    def set_hierarchy(self, *args, **kwargs):
        """
        Set the hierarchy of the system, and recreate/rerun all necessary
        constraints (can be slow).

        For a list of all constraints that are automatically set based on the
        hierarchy, see <phoebe.frontend.bundle.Bundle.add_constraint>.

        See the built-in functions for building hierarchy reprentations:
        * <phoebe.parmaeters.hierarchy>
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

            if sys.version_info[0] == 3:
              kind = func.__name__
            else:
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
                pass
                # we'll do the potential constraint either way
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
        self._add_history(redo_func='set_hierarchy',
                          redo_kwargs=redo_kwargs,
                          undo_func='set_hierarchy',
                          undo_kwargs=undo_kwargs)

        return

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

    def run_checks(self, raise_logger_warning=False, raise_error=False, **kwargs):
        """
        Check to see whether the system is expected to be computable.

        This is called by default for each set_value but will only raise a
        logger warning if fails.  This is also called immediately when calling
        <phoebe.frontend.bundle.Bundle.run_compute>.

        kwargs are passed to override currently set values as if they were
        sent to <phoebe.frontend.bundle.Bundle.run_compute>.

        Arguments
        -----------
        * `compute` (string or list of strings, optional, default=None): the
            compute options to use  when running checks.  If None (or not provided),
            the compute options in the 'run_checks_compute@setting' parameter
            will be used (which defaults to all available compute options).
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

        report = RunChecksReport()

        try:
            run_checks_compute = self.get_value(qualifier='run_checks_compute', context='setting', check_visible=False, check_default=False, expand=True)
        except ValueError:
            run_checks_compute = []
        computes = kwargs.pop('compute', run_checks_compute)
        if computes is None:
            computes = self.computes
        else:
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

        kwargs.setdefault('check_visible', False)
        kwargs.setdefault('check_default', False)

        hier = self.hierarchy
        if hier is None:
            return report

        hier_stars = hier.get_stars()
        hier_meshables = hier.get_meshables()

        for component in hier_stars:
            kind = hier.get_kind_of(component) # shouldn't this always be 'star'?
            comp_ps = self.get_component(component, **_skip_filter_checks)

            if not len(comp_ps):
                report.add_item(b,
                                "component '{}' in the hierarchy is not in the bundle".format(component)
                                [hier],
                                True)

            parent = hier.get_parent_of(component)
            parent_ps = self.get_component(parent, **_skip_filter_checks)
            if kind in ['star']:
                if self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **kwargs) >= 10000 and self.get_value(qualifier='ld_mode_bol', component=component, context='component') == 'lookup':
                    report.add_item(self,
                                    "ld_mode_bol of 'lookup' uses a bolometric passband which is not reliable for hot stars.  Consider using ld_mode_bol of manual and providing ld_coeffs instead.",
                                    [self.get_parameter(qualifier='teff', component=component, context='component'),
                                     self.get_parameter(qualifier='ld_mode_bol', component=component, context='component')],
                                    False
                                    )


                # ignore the single star case
                if parent:
                    # contact systems MUST by synchronous
                    if hier.is_contact_binary(component):
                        if self.get_value(qualifier='syncpar', component=component, context='component', **_skip_filter_checks) != 1.0:
                            report.add_item(self,
                                            "contact binaries must be synchronous, but syncpar@{}!=1".format(component),
                                            [self.get_parameter(qualifier='syncpar', component=component, context='component', **_skip_filter_checks)],
                                            True)

                        if self.get_value(qualifier='ecc', component=parent, context='component', **_skip_filter_checks) != 0.0:
                            # TODO: this can result in duplicate entries in the report
                            report.add_item(self,
                                            "contact binaries must be circular, but ecc@{}!=0".format(parent),
                                            [self.get_parameter(qualifier='ecc', component=parent, context='component', **_skip_filter_checks)],
                                            True)

                        if self.get_value(qualifier='pitch', component=component, context='component', **_skip_filter_checks) != 0.0:
                            report.add_item(self,
                                            'contact binaries must be aligned, but pitch@{}!=0.  Try b.set_value(qualifier=\'pitch\', component=\'{}\' value=0.0, check_visible=False) to align.'.format(component, component),
                                            [self.get_parameter(qualifier='pitch', component=component, context='component', **_skip_filter_checks)],
                                            True)

                        if self.get_value(qualifier='yaw', component=component, context='component', **_skip_filter_checks) != 0.0:
                            report.add_item(self,
                                            'contact binaries must be aligned, but yaw@{}!=0.  Try b.set_value(qualifier=\'yaw\', component=\'{}\', value=0.0, check_visible=False) to align.'.format(component, component),
                                            [self.get_parameter(qualifier='yaw', component=component, context='component', **_skip_filter_checks)],
                                            True)

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
                                            True)

                        if np.isnan(requiv) or requiv <= requiv_min:
                            report.add_item(self,
                                            '{} is underflowing at L1 and not a contact system (requiv={}, requiv_min={}, requiv_max={})'.format(component, requiv, requiv_min, requiv_max),
                                            [comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks),
                                             comp_ps.get_parameter(qualifier='requiv_min', **_skip_filter_checks),
                                             parent_ps.get_parameter(qualifier='sma', **_skip_filter_checks)],
                                            True)

                        elif requiv <= requiv_min * 1.001:
                            report.add_item(self,
                                            'requiv@{} is too close to requiv_min (within 0.1% of critical).  Use detached/semidetached model instead.'.format(component),
                                            [comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks),
                                             comp_ps.get_parameter(qualifier='requiv_min', **_skip_filter_checks),
                                             parent_ps.get_parameter(qualifier='sma', **_skip_filter_checks),
                                             hier],
                                            True)

                    else:
                        if requiv > requiv_max:
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
                                            True)

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
                                    True)

        # run passband checks
        all_pbs = list_passbands(full_dict=True)
        online_pbs = list_online_passbands(full_dict=True)

        for pbparam in self.filter(qualifier='passband', **_skip_filter_checks).to_list():

            # we include this in the loop so that we get the most recent dict
            # if a previous passband had to be updated
            installed_pbs = list_installed_passbands(full_dict=True)

            pb = pbparam.get_value()

            pb_needs_Inorm = True
            pb_needs_Imu = True
            pb_needs_ld = True #np.any([p.get_value()!='interp' for p in self.filter(qualifier='ld_mode', dataset=pbparam.dataset, context='dataset', **_skip_filter_checks).to_list()])
            pb_needs_ldint = True
            pb_needs_ext = self.get_value(qualifier='ebv', dataset=pbparam.dataset, context='dataset', **_skip_filter_checks)

            missing_pb_content = []

            # NOTE: atms are not attached to datasets, but per-compute and per-component
            for atmparam in self.filter(qualifier='atm', kind='phoebe', **_skip_filter_checks).to_list():

                # check to make sure passband supports the selected atm
                atm = atmparam.get_value(**_skip_filter_checks)
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
                                            True)


            # remove any duplicates
            missing_pb_content = list(set(missing_pb_content))
            if len(missing_pb_content):
                installed_timestamp = installed_pbs.get(pb, {}).get('timestamp', None)
                online_timestamp = online_pbs.get(pb, {}).get('timestamp', None)
                if pb not in installed_pbs.keys():
                    logger.warning("downloading and installing {} passband with content={}".format(pb, missing_pb_content))
                    try:
                        download_passband(pb, content=missing_pb_content)
                    except IOError:
                        report.add_item(self,
                                        'Attempt to download {} passband failed.  Check internet connection, wait for tables.phoebe-project.org to come back online, or try another passband.'.format(pb),
                                        [pbparam],
                                        True)
                elif _timestamp_to_dt(installed_timestamp) == _timestamp_to_dt(online_timestamp):
                    logger.warning("updating installed {} passband (with matching online timestamp) to include content={}".format(pb, missing_pb_content))
                    try:
                        update_passband(pb, content=missing_pb_content)
                    except IOError:
                        report.add_item(self,
                                        'Attempt to update {} passband for the {} tables failed.  Check internet connection, wait for tables.phoebe-project.org to come back online, or try another passband.'.format(pb, missing_pb_content),
                                        [pbparam],
                                        True)
                else:
                    report.add_item(self,
                                    'installed passband "{}" is missing the following tables: {}. The available online version ({}) is newer than the installed version ({}), so will not be updated automatically.  Call phoebe.update_passband("{}", content={}) or phoebe.update_all_passbands() to update to the latest version.'.format(pb, missing_pb_content, installed_timestamp, online_timestamp, pb, atm, missing_pb_content),
                                    [pbparam],
                                    True)

        # check length of ld_coeffs vs ld_func and ld_func vs atm
        def ld_coeffs_len(ld_func, ld_coeffs):
            # current choices for ld_func are:
            # ['uniform', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power', 'claret', 'hillen', 'prsa']
            if ld_func in ['linear'] and (ld_coeffs is None or len(ld_coeffs)==1):
                return True,
            elif ld_func in ['logarithmic', 'square_root', 'quadratic'] and (ld_coeffs is None or len(ld_coeffs)==2):
                return True,
            elif ld_func in ['power'] and (ld_coeffs is None or len(ld_coeffs)==4):
                return True,
            else:
                return False, "ld_coeffs={} wrong length for ld_func='{}'.".format(ld_coeffs, ld_func)

        irrad_enabled = kwargs.get('irrad_method', True) != 'none' and np.any([p.get_value()!='none' for p in self.filter(qualifier='irrad_method', compute=computes, **kwargs).to_list()])
        for component in hier_stars:
            if not irrad_enabled:
                continue
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
                                    True)
                elif ld_mode == 'manual':
                    report.add_item(self,
                                    'nans in ld_coeffs_bol are forbidden',
                                    [self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)],
                                    True)
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
                                    True)
            elif ld_mode == 'manual':

                check = ld_coeffs_len(ld_func, ld_coeffs)
                if not check[0]:
                    report.add_item(self,
                                    check[1],
                                    [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                     self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)
                                    ],
                                    True)


                check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=False)
                if not check:
                    report.add_item(self,
                                    'ld_coeffs_bol={} not compatible for ld_func_bol=\'{}\'.'.format(ld_coeffs, ld_func),
                                    [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                     self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)
                                    ],
                                    True)

                else:
                    # only need to do the strict check if the non-strict checks passes
                    check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=True)
                    if not check:
                        report.add_item(self,
                                        'ld_coeffs_bol={} result in limb-brightening.  Use with caution.'.format(ld_coeffs),
                                        [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                         self.get_parameter(qualifier='ld_coeffs_bol', component=component, context='component', **_skip_filter_checks)
                                        ],
                                        True)

            for compute in computes:
                if self.get_compute(compute, **_skip_filter_checks).kind in ['legacy'] and ld_func not in ['linear', 'logarithmic', 'square_root']:
                    report.add_item(self,
                                    "ld_func_bol='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', or 'square_root'.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                    [self.get_parameter(qualifier='ld_func_bol', component=component, context='component', **_skip_filter_checks),
                                     self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)],
                                    True)


            for dataset in self.filter(context='dataset', kind=['lc', 'rv'], check_default=True).datasets:
                if dataset=='_default':
                    # just in case conf.check_default = False
                    continue
                dataset_ps = self.get_dataset(dataset=dataset, check_visible=False)

                ld_mode = dataset_ps.get_value(qualifier='ld_mode', component=component, **_skip_filter_checks)
                # cast to string to ensure not a unicode since we're passing to libphoebe
                ld_func = str(dataset_ps.get_value(qualifier='ld_func', component=component, **_skip_filter_checks))
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
                                        True)
                    elif ld_mode == 'manual':
                        report.add_item(self,
                                        'nans in ld_coeffs are forbidden',
                                        [dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)],
                                        True)
                    else:
                        # if interp, then the previously set value won't be used anyways, so we'll ignore nans
                        pass

                if ld_mode == 'interp':
                    for compute in computes:
                        # TODO: should we ignore if the dataset is disabled?
                        try:
                            atm = self.get_value(qualifier='atm', component=component, compute=compute, context='compute', atm=kwargs.get('atm', None), **_skip_filter_checks)
                        except ValueError:
                            # not all backends have atm as a parameter/option
                            continue
                        else:
                            if atm not in ['ck2004', 'phoenix']:
                                if 'ck2004' in self.get_parameter(qualifier='atm', component=component, compute=compute, context='compute', atm=kwargs.get('atm', None), **_skip_filter_checks).choices:
                                    report.add_item(self,
                                                    "ld_mode='interp' not supported by atm='{}'.  Either change atm@{}@{} or ld_mode@{}@{}.".format(atm, component, compute, component, dataset),
                                                    [dataset_ps.get_parameter(qualifier='ld_mode', component=component, **_skip_filter_checks),
                                                     self.get_parameter(qualifier='atm', component=component, compute=compute, context='compute', **_skip_filter_checks),
                                                     self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)
                                                    ],
                                                    True)
                                else:
                                    report.add_item(self,
                                                    "ld_mode='interp' not supported by '{}' backend used by compute='{}'.  Change ld_mode@{}@{} or use a backend that supports atm='ck2004'.".format(self.get_compute(compute).kind, compute, component, dataset),
                                                    [dataset_ps.get_parameter(qualifier='ld_mode', component=component, **_skip_filter_checks),
                                                     self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)
                                                    ],
                                                    True)


                elif ld_mode == 'lookup':
                    if ld_coeffs_source != 'auto' and ld_coeffs_source not in all_pbs.get(pb, {}).get('atms_ld', []) :
                        report.add_item(self,
                                        'passband={} does not support ld_coeffs_source={}.  Either change ld_coeffs_source@{}@{} or ld_mode@{}@{}'.format(pb, ld_coeffs_source, component, dataset, component, dataset),
                                        [dataset_ps.get_parameter(qualifier='ld_coeffs_source', component=component, **_skip_filter_checks),
                                         dataset_ps.get_parameter(qualifier='ld_mode', component=component **_skip_filter_checks)
                                        ],
                                        True)


                elif ld_mode == 'manual':
                    check = ld_coeffs_len(ld_func, ld_coeffs)
                    if not check[0]:
                        report.add_item(self,
                                        check[1],
                                        [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                         dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                                        ],
                                        True)

                    check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=False)
                    if not check:
                        report.add_item(self,
                                        'ld_coeffs={} not compatible for ld_func=\'{}\'.'.format(ld_coeffs, ld_func),
                                        [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                         dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                                        ],
                                        True)

                    else:
                        # only need to do the strict check if the non-strict checks passes
                        check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs), strict=True)
                        if not check:
                            report.add_item(self,
                                            'ld_coeffs={} result in limb-brightening.  Use with caution.'.format(ld_coeffs),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                             dataset_ps.get_parameter(qualifier='ld_coeffs', component=component, **_skip_filter_checks)
                                             ],
                                             False)

                else:
                    raise NotImplementedError("checks for ld_mode='{}' not implemented".format(ld_mode))

                if ld_mode in ['lookup', 'manual']:
                    for compute in computes:
                        compute_kind = self.get_compute(compute, **_skip_filter_checks).kind
                        if compute_kind in ['legacy'] and ld_func not in ['linear', 'logarithmic', 'square_root']:
                            report.add_item(self,
                                            "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', or 'square_root'.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                             self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)],
                                            True)

                        if compute_kind in ['jktebop'] and ld_func not in ['linear', 'logarithmic', 'square_root', 'quadratic']:
                            report.add_item(self,
                                            "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', 'quadratic', or 'square_root'.".format(ld_func, self.get_compute(compute, **_skip_filter_checks).kind, compute),
                                            [dataset_ps.get_parameter(qualifier='ld_func', component=component, **_skip_filter_checks),
                                             self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)],
                                            True)


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

            # 2.2 disables support for boosting.  The boosting parameter in 2.2 only has 'none' as an option, but
            # importing a bundle from old releases may still have 'linear' as an option, so we'll check here
            if compute_kind in ['phoebe'] and self.get_value(qualifier='boosting_method', compute=compute, boosting_method=kwargs.get('boosting_method', None), **_skip_filter_checks) != 'none':
                report.add_item(self,
                                "support for beaming/boosting has been removed from PHOEBE 2.2.  Set boosting_method to 'none'.",
                                [self.get_parameter(qualifier='boosting_method', compute=compute, boosting_method=kwargs.get('boosting_method', None), **_skip_filter_checks)],
                                True)

            # mesh-consistency checks
            mesh_methods = [p.get_value(mesh_method=kwargs.get('mesh_method', None)) for p in self.filter(qualifier='mesh_method', compute=compute, force_ps=True, check_default=True, check_visible=False).to_list()]
            if 'wd' in mesh_methods:
                if len(set(mesh_methods)) > 1:
                    report.add_item(self,
                                    "all (or no) components must use mesh_method='wd'.",
                                    self.filter(qualifier='mesh_method', compute=compute, force_ps=True, check_default=True, check_visible=False).to_list()+[self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)],
                                    True)

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
                                        self.get_parameter(qualifier='eclipse_method', compute=compute, **_skip_filter_checks),
                                        self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)],
                                        True)

                    else:
                        # only raise a warning
                        offending_components = [comp for comp in triangle_areas.keys() if triangle_areas[comp] > 5*min(areas.values())]
                        smallest_components = [comp for comp in areas.keys() if areas[comp] == min(areas.values())]
                        report.add_item(self,
                                        "triangles on {} are nearly the size of the entire bodies of {}, resulting in inaccurate eclipse detection.  Check values for requiv of {} and/or ntriangles of {}.  If your system is known to NOT eclipse, you can set eclipse_method to 'only_horizon' to circumvent this check.".format(offending_components, smallest_components, smallest_components, offending_components),
                                        self.filter(qualifier='requiv', component=smallest_components).to_list(),
                                        self.get_parameter(qualifier='ntriangles', component=offending_components, compute=compute, **_skip_filter_checks).to_list()+[
                                        self.get_parameter(qualifier='eclipse_method', compute=compute, eclipse_method=kwargs.get('eclipse_method', None), **_skip_filter_checks).
                                        self.get_parameter(qualifier='run_checks_compute', context='setting', **_skip_filter_checks)],
                                        False)

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
                                    True)

            pblum_mode = self.get_value(qualifier='pblum_mode', dataset=coupled_to, **_skip_filter_checks)
            if pblum_mode in ['dataset-scaled', 'dataset-coupled']:
                report.add_item(self,
                                "cannot set pblum_dataset@{}='{}' as that dataset has pblum_mode@{}='{}'".format(param.dataset, coupled_to, coupled_to, pblum_mode),
                                [param,
                                self.get_parameter(qualifier='pblum_mode', dataset=coupled_to, **_skip_filter_checks)],
                                True)

        # require any pblum_mode == 'dataset-scaled' to have accompanying data
        for param in self.filter(qualifier='pblum_mode', value='dataset-scaled', **_skip_filter_checks).to_list():
            if not len(self.get_value(qualifier='fluxes', dataset=param.dataset, context='dataset', **_skip_filter_checks)):
                report.add_item(self,
                                "fluxes@{} cannot be empty if pblum_mode@{}='dataset-scaled'".format(param.dataset, param.dataset),
                                [param,
                                self.get_parameter(qualifier='fluxes', dataset=param.dataset, context='dataset', **_skip_filter_checks)],
                                True)

            # also check to make sure that we'll be able to handle the interpolation in time if the system is time-dependent
            if self.hierarchy.is_time_dependent():
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
                                    True)

        # tests for lengths of fluxes, rvs, etc vs times (and fluxes vs wavelengths for spectral datasets)
        for param in self.filter(qualifier=['times', 'fluxes', 'rvs', 'sigmas', 'wavelengths', 'flux_densities'], context='dataset', **_skip_filter_checks).to_list():
            shape = param.get_value().shape
            if len(shape) > 1:
                report.add_item(self,
                                "{}@{} must be a flat array, got shape {}".format(param.qualifier, param.dataset, shape),
                                [param],
                                True)

            if param.qualifier in ['fluxes', 'rvs', 'sigmas'] and shape[0] > 0 and shape[0] != self.get_value(qualifier='times', dataset=param.dataset, component=param.component, context='dataset', **_skip_filter_checks).shape[0]:
                tparam = self.get_parameter(qualifier='times', dataset=param.dataset, component=param.component, context='dataset', **_skip_filter_checks)
                report.add_item(self,
                                "{} must be of same length as {}".format(param.twig, tparam.twig),
                                [param, tparam],
                                True)

            if param.qualifier in ['flux_densities'] and shape[0] > 0 and shape[0] != self.get_value(qualifier='wavelengths', dataset=param.dataset, component=param.component, time=param.time, context='dataset', **_skip_filter_checks).shape[0]:
                wparam = self.get_parameter(qualifier='wavelengths', dataset=param.dataset, component=param.component, time=param.time, context='dataset', **_skip_filter_checks)
                report.add_item(self,
                                "{}@{}@{} must be of same length as {}@{}".format(param.twig, wparam.twig),
                                [param, wparam],
                                True)



        try:
            self.run_failed_constraints()
        except:
            report.add_item(self,
                            "constraints {} failed to run.  Address errors and try again.  Call run_failed_constraints to see the tracebacks.".format([p.twig for p in self.filter(uniqueid=self._failed_constraints).to_list()]),
                            self.filter(uniqueid=self._failed_constraints).to_list(),
                            True)


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
                                False)
            elif teff <= 6600. and gravb_bol >= 0.9:
                report.add_item(self,
                                "'{}' probably has a convective atm (teff={:.0f}K<6600K), for which gravb_bol<0.9 (suggestion: 0.32) might be a better approx than gravb_bol={:.2f}.".format(component, teff, gravb_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)],
                                False)
            elif (teff > 6600 and teff < 8000) and gravb_bol < 0.32 or gravb_bol > 1.00:
                report.add_item(self,
                                "'{}' has intermittent temperature (6600K<teff={:.0f}K<8000K), gravb_bol might be better between 0.32-1.00 than gravb_bol={:.2f}.".format(component, teff, gravb_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)],
                                False)

        for component in hier_stars:
            teff = self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **_skip_filter_checks)
            irrad_frac_refl_bol = self.get_value(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)

            if teff >= 8000. and irrad_frac_refl_bol < 0.8:
                report.add_item(self,
                                "'{}' probably has a radiative atm (teff={:.0f}K>=8000K), for which irrad_frac_refl_bol>0.8 (suggestion: 1.0) might be a better approx than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)],
                                False)
            elif teff <= 6600. and irrad_frac_refl_bol >= 0.75:
                report.add_item(self,
                                "'{}' probably has a convective atm (teff={:.0f}K<=6600K), for which irrad_frac_refl_bol<0.75 (suggestion: 0.6) might be a better approx than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)],
                                False)
            elif (teff > 6600. and teff < 8000) and irrad_frac_refl_bol < 0.6:
                report.add_item(self,
                                "'{}' has intermittent temperature (6600K<teff={:.0f}K<8000K), irrad_frac_refl_bol might be better between 0.6-1.00 than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol),
                                [self.get_parameter(qualifier='teff', component=component, context='component', **_skip_filter_checks),
                                 self.get_parameter(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)],
                                False)

        # TODO: add other checks
        # - make sure all ETV components are legal
        # - check for conflict between dynamics_method and mesh_method (?)


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
                                affected_params,
                                False)


        for figure in self.figures:
            x = self.get_value(qualifier='x', figure=figure, context='figure', **_skip_filter_checks)
            y = self.get_value(qualifier='y', figure=figure, context='figure', **_skip_filter_checks)
            if (x in ['xs', 'ys', 'zs'] and y in ['us', 'vs', 'ws']) or (x in ['us', 'vs', 'ws'] and y in ['xs', 'ys', 'zs']):
                report.add_item(self,
                                "cannot mix xyz and uvw coordinates in {} figure".format(figure),
                                [self.get_parameter(qualifier='x', figure=figure, context='figure', **_skip_filter_checks),
                                 self.get_parameter(qualifier='y', figure=figure, context='figure', **_skip_filter_checks)
                                ],
                                False)

        if raise_logger_warning:
            for item in report.items:
                # passed is either False (failed) or None (raise Warning)
                msg = item.message
                if item.fail:
                    msg += "  If not addressed, this warning will continue to be raised and will throw an error at run_compute."
                logger.warning(msg)

        if raise_error:
            if not report.passed:
                raise ValueError("system failed to pass checks\n{}".format(report))

        return report

    def references(self, compute=None, dataset=None):
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
        * Alternate backends, when applicable.

        Arguments
        ------------
        * `compute` (string or list of strings, optional, default=None): only
            consider a single (or list of) compute options.  If None or not
            provided, will default to all attached compute options.
        * `dataset` (string or list of strings, optional, default=None): only
            consider a single (or list of) datasets.  If None or not provided,
            will default to all attached datasets.

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


        # ref: url pairs
        citation_urls = {'Prsa & Zwitter (2005)': 'https://ui.adsabs.harvard.edu/?#abs/2005ApJ...628..426P',
                         'Prsa et al. (2016)': 'https://ui.adsabs.harvard.edu/?#abs/2016ApJS..227...29P',
                         'Horvat et al. (2018)': 'https://ui.adsabs.harvard.edu/?#abs/2016ApJS..227...29P',
                         'Jones et al. (2020, submitted)': 'https://ui.adsabs.harvard.edu/abs/2019arXiv191209474J',
                         'Castelli & Kurucz (2004)': 'https://ui.adsabs.harvard.edu/#abs/2004astro.ph..5087C',
                         'Husser et al. (2013)': 'https://ui.adsabs.harvard.edu/#abs/2013A&A...553A...6H',
                         'numpy/scipy': 'https://www.scipy.org/citing.html',
                         'astropy': 'https://www.astropy.org/acknowledging.html',
                         'jktebop': 'http://www.astro.keele.ac.uk/jkt/codes/jktebop.html',
                         'Carter et al. (2011)': 'https://ui.adsabs.harvard.edu/abs/2011Sci...331..562C',
                         'Andras (2012)': 'https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.1630P',
                         'Maxted (2016)': 'https://ui.adsabs.harvard.edu/abs/2016A%26A...591A.111M',
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
                recs = _add_reason(recs, 'Prsa et al. (2016)', 'PHOEBE 2 backend')
            elif self.get_compute(compute).kind == 'legacy':
                recs = _add_reason(recs, 'Prsa & Zwitter (2005)', 'PHOEBE 1 (legacy) backend')
                # TODO: include Wilson & Devinney?
            elif self.get_compute(compute).kind == 'jktebop':
                recs = _add_reason(recs, 'jktebop', 'jktebop backend')
            elif self.get_compute(compute).kind == 'photodynam':
                recs = _add_reason(recs, 'Carter et al. (2011)', 'photodynam backend')
                recs = _add_reason(recs, 'Andras (2012)', 'photodynam backend')
            elif self.get_compute(compute).kind == 'ellc':
                recs = _add_response(recs, 'Maxted (2016)', 'ellc backend')


        # check for presence of datasets that require PHOEBE releases
        for dataset in datasets:
            if ['lp'] in self.get_dataset(dataset).kinds:
                recs = _add_reason(recs, 'Horvat et al. (2018)', 'support for line profile datasets')

        # check for any enabled physics that requires specific PHOEBE releases
        for component in self.hierarchy.get_stars():
            if self.get_value(qualifier='pitch', component=component, context='component') != 0. or self.get_value(qualifier='yaw', component=component, context='component') != 0.:
                recs = _add_reason(recs, 'Horvat et al. (2018)', 'support for misaligned system')
        for ebv_param in self.filter(qualifier='ebv', context='dataset'):
            if ebv_param.get_value() > 0:
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

        # provide references from dependencies
        recs = _add_reason(recs, 'numpy/scipy', 'numpy/scipy dependency within PHOEBE')
        recs = _add_reason(recs, 'astropy', 'astropy dependency within PHOEBE')

        return {r: {'url': citation_urls.get(r, None), 'uses': v} for r,v in recs.items()}


    def add_feature(self, kind, component=None, **kwargs):
        """
        Add a new feature (spot, etc) to a component in the system.  If not
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

        Arguments
        -----------
        * `kind` (string): function to call that returns a
             <phoebe.parameters.ParameterSet> or list of
             <phoebe.parameters.Parameter> objects.  This must either be a
             callable function that accepts only default values, or the name
             of a function (as a string) that can be found in the
             <phoebe.parameters.compute> module.
        * `component` (string, optional): name of the component to attach the
            feature.  Note: only optional if only a single possibility otherwise.
        * `feature` (string, optional): name of the newly-created feature.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing feature with the same `feature` tag.  If False,
            an error will be raised.
        * `return_overwrite` (boolean, optional, default=False): whether to include
            removed parameters due to `overwrite` in the returned ParameterSet.
            Only applicable if `overwrite` is True.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added


        Raises
        ----------
        * NotImplementedError: if a required constraint is not implemented.
        * ValueError: if `component` is required but is not provided.
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

        if component is None:
            stars = self.hierarchy.get_meshables()
            if len(stars) == 1:
                component = stars[0]
            else:
                raise ValueError("must provide component")

        if component not in self.components:
            raise ValueError('component not recognized')

        component_kind = self.filter(component=component, context='component').kind
        if not _feature._component_allowed_for_feature(func.__name__, component_kind):
            raise ValueError("{} does not support component with kind {}".format(func.__name__, component_kind))

        params, constraints = func(**kwargs)

        metawargs = {'context': 'feature',
                     'component': component,
                     'feature': kwargs['feature'],
                     'kind': func.__name__}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_feature(feature=kwargs['feature'])
            # check the label again, just in case kwargs['feature'] belongs to
            # something other than feature
            self._check_label(kwargs['feature'], allow_overwrite=False)

        self._attach_params(params, **metawargs)
        # attach params called _check_copy_for, but only on it's own parameterset
        self._check_copy_for()

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.__name__
        self._add_history(redo_func='add_feature',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_feature',
                          undo_kwargs={'feature': kwargs['feature']})

        for constraint in constraints:
            self.add_constraint(*constraint)

        #return params
        # NOTE: we need to call get_ in order to make sure all metawargs are applied
        ret_ps = self.filter(feature=kwargs['feature'], **_skip_filter_checks)

        if kwargs.get('overwrite', False) and kwargs.get('return_overwrite', False):
            ret_ps += overwrite_ps

        return ret_ps

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
        kwargs['context'] = 'feature'
        return self.filter(**kwargs)

    def rename_feature(self, old_feature, new_feature):
        """
        Rename a 'feature' in the bundle

        :parameter old_feature: current label for the feature
        :parameter new_feature: new label for the feature

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed feature
        """
        for param in self.filter(feature=old_feature).to_list():
            param._feature = new_feature

        redo_kwargs = {'old_feature': old_feature, 'new_feature': new_feature}
        undo_kwargs = {'old_feature': new_feature, 'new_feature': old_feature}

        self._add_history(redo_func='rename_feature',
                          redo_kwargs=redo_kwargs,
                          undo_func='rename_feature',
                          undo_kwargs=undo_kwargs)

        return self.filter(feature=new_feature)

    def remove_feature(self, feature=None, **kwargs):
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
        self._kwargs_checks(kwargs)

        # Let's avoid deleting ALL features from the matching contexts
        if feature is None and not len(kwargs.items()):
            raise ValueError("must provide some value to filter for features")

        kwargs['feature'] = feature

        # Let's avoid the possibility of deleting a single parameter
        kwargs['qualifier'] = None

        # Let's also avoid the possibility of accidentally deleting system
        # parameters, etc
        kwargs.setdefault('context', ['feature', 'compute'])

        removed_ps = self.remove_parameters_all(**kwargs)

        self._add_history(redo_func='remove_feature',
                          redo_kwargs=kwargs,
                          undo_func=None,
                          undo_kwargs={})

        return removed_ps

    def remove_features_all(self):
        """
        Remove all features from the bundle.  To remove a single feature, see
        <phoebe.frontend.bundle.Bundle.remove_feature>.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for feature in self.features:
            removed_ps += self.remove_feature(feature=feature)
        return removed_ps

    def rename_feature(self, old_feature, new_feature):
        """
        Change the label of a feature attached to the Bundle.

        Arguments
        ----------
        * `old_feature` (string): current label of the feature (must exist)
        * `new_feature` (string): the desired new label of the feature
            (must not yet exist)

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_feature` is forbidden or already exists.
        """
        # TODO: raise error if old_feature not found?

        self._check_label(new_feature)
        self._rename_label('feature', old_feature, new_feature)

        return self.filter(feature=new_feature)


    def enable_feature(self, feature=None, **kwargs):
        """
        Enable a `feature`.  Features that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_fitting (once supported).

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

        self._add_history(redo_func='enable_feature',
                          redo_kwargs={'feature': feature},
                          undo_func='disable_feature',
                          undo_kwargs={'feature': feature})

        return self.get_feature(feature=feature)


    def disable_feature(self, feature=None, **kwargs):
        """
        Disable a `feature`.  Features that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_fitting (once supported).

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

        self._add_history(redo_func='disable_feature',
                          redo_kwargs={'feature': feature},
                          undo_func='enable_feature',
                          undo_kwargs={'feature': feature})

        return self.get_feature(feature=feature)

    def add_spot(self, component=None, feature=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_feature> but with kind='spot'.
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

    def rename_spot(self, old_feature, new_feature):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.remove_feature> but with kind='spot'.
        """
        kwargs.setdefault('kind', 'spot')
        return self.remove_feature(feature, **kwargs)

    @send_if_client
    def add_component(self, kind, **kwargs):
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
        * <phoebe.parmaeters.component.orbit>
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
        * `return_overwrite` (boolean, optional, default=False): whether to include
            removed parameters due to `overwrite` in the returned ParameterSet.
            Only applicable if `overwrite` is True.
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

        if sys.version_info[0] == 3:
          fname = func.__name__
        else:
          fname = func.__name__


        if kwargs.get('component', False) is None:
            # then we want to apply the default below, so let's pop for now
            _ = kwargs.pop('component')

        kwargs.setdefault('component',
                          self._default_label(fname,
                                              **{'context': 'component',
                                                 'kind': fname}))

        if kwargs.pop('check_label', True):
            self._check_label(kwargs['component'], allow_overwrite=kwargs.get('overwrite', False))

        params, constraints = func(**kwargs)


        metawargs = {'context': 'component',
                     'component': kwargs['component'],
                     'kind': fname}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_component(component=kwargs['component'])
            # check the label again, just in case kwargs['component'] belongs to
            # something other than component
            self.exclude(component=kwargs['component'])._check_label(kwargs['component'], allow_overwrite=False)

        self._attach_params(params, **metawargs)
        # attach params called _check_copy_for, but only on it's own parameterset
        self._check_copy_for()

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = fname
        self._add_history(redo_func='add_component',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_component',
                          undo_kwargs={'component': kwargs['component']})

        for constraint in constraints:
            self.add_constraint(*constraint)

        # Figure options for this dataset
        if kind in ['star']:
            fig_params = _figure._add_component(self, **kwargs)

            fig_metawargs = {'context': 'figure',
                             'kind': kind,
                             'component': kwargs['component']}
            self._attach_params(fig_params, **fig_metawargs)


        # TODO: include figure params in returned PS?
        ret_ps = self.get_component(check_visible=False, check_default=False, **metawargs)

        ret_ps += ParameterSet(self._handle_component_selectparams(return_changes=True))
        ret_ps += ParameterSet(self._handle_pblum_defaults(return_changes=True))

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs,
                            additional_allowed_keys=['overwrite', 'return_overwrite'],
                            warning_only=True, ps=ret_ps)

        if kwargs.get('overwrite', False) and kwargs.get('return_overwrite', False):
            ret_ps += overwrite_ps

        return ret_ps

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
        kwargs['context'] = 'component'
        return self.filter(**kwargs)

    def remove_component(self, component, **kwargs):
        """
        Remove a 'component' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `component` (string): the label of the component to be removed.
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
        ret_ps += ParameterSet(self._handle_component_selectparams(return_changes=True))
        return ret_ps

    def rename_component(self, old_component, new_component):
        """
        Change the label of a component attached to the Bundle.

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
        ret_params = self._rename_label('component', old_component, new_component)
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

    def rename_orbit(self, old_orbit, new_orbit):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.remove_component> but with kind='star'.
        """
        kwargs.setdefault('kind', 'orbit')
        return self.remove_component(component, **kwargs)

    def add_star(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.add_component> but with kind='star'.
        """
        kwargs.setdefault('component', component)
        return self.add_component('star', **kwargs)

    def get_star(self, component=None, **kwargs):
        """
        Shortcut to :meth:`get_component` but with kind='star'

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

    def rename_star(self, old_star, new_star):
        """
        Shortcut to :meth:`rename_component`
        """
        return self.rename_component(old_star, new_star)

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

    def rename_envelope(self, old_envelope, new_envelope):
        """
        Shortcut to :meth:`rename_component`
        """
        return self.rename_component(old_envelope, new_envelope)

    def remove_envelope(self, component=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.remove_component> but with kind='envelope'.
        """
        kwargs.setdefault('kind', 'envelope')
        return self.remove_component(component, **kwargs)

    def get_ephemeris(self, component=None, t0='t0_supconj', **kwargs):
        """
        Get the ephemeris of a component (star or orbit).

        NOTE: support for `shift` and `phshift` was removed as of version 2.1.
        Please pass `t0` instead.

        Arguments
        ---------------
        * `component` (str, optional): name of the component.  If not given,
            component will default to the top-most level of the current
            hierarchy.  See <phoebe.parameters.HierarchyParameter.get_top>.
        * `t0` (str, optional, default='t0_supconj'): qualifier of the parameter
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

        ps = self.filter(component=component, context='component')

        if ps.kind in ['orbit']:
            ret['period'] = ps.get_value(qualifier='period', unit=u.d)
            if isinstance(t0, str):
                if t0 == 't0':
                    ret['t0'] = self.get_value(qualifier='t0', context='system', unit=u.d)
                else:
                    ret['t0'] = ps.get_value(qualifier=t0, unit=u.d)
            elif isinstance(t0, float) or isinstance(t0, int):
                ret['t0'] = t0
            else:
                raise ValueError("t0 must be string (qualifier) or float")
            ret['dpdt'] = ps.get_value(qualifier='dpdt', unit=u.d/u.d)
        elif ps.kind in ['star']:
            # TODO: consider renaming period to prot
            ret['period'] = ps.get_value(qualifier='period', unit=u.d)
            if isinstance(t0, float) or isinstance(t0, int):
                ret['t0'] = t0
            else:
                ret['t0'] = self.get_value('t0', context='system', unit=u.d)
        else:
            raise NotImplementedError

        for k,v in kwargs.items():
            ret[k] = v

        return ret

    def to_phase(self, time, component=None, t0='t0_supconj', **kwargs):
        """
        Get the phase(s) of a time(s) for a given ephemeris.

        See also: <phoebe.frontend.bundle.Bundle.get_ephemeris>.

        Arguments
        -----------
        * `time` (float/list/array): time to convert to phases (should be in
            same system/units as t0s)
        * `component` (str, optional): component for which to get the ephemeris.
            If not given, component will default to the top-most level of the
            current hierarchy.  See <phoebe.parameters.HierarchyParameter.get_top>.
        * `t0` (str, optional, default='t0_supconj'): qualifier of the parameter
            to be used for t0
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

        ephem = self.get_ephemeris(component=component, t0=t0, **kwargs)

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
        if dpdt != 0:
            phase = np.mod(1./dpdt * np.log(period + dpdt*(time-t0)), 1.0)
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

    def to_time(self, phase, component=None, t0='t0_supconj', **kwargs):
        """
        Get the time(s) of a phase(s) for a given ephemeris.

        See also: <phoebe.frontend.bundle.Bundle.get_ephemeris>.

        Arguments
        -----------
        * `phase` (float/list/array): phase to convert to times (should be in
            same system/units as t0s)
        * `component` (str, optional): component for which to get the ephemeris.
            If not given, component will default to the top-most level of the
            current hierarchy.  See <phoebe.parameters.HierarchyParameter.get_top>.
        * `t0` (str, optional, default='t0_supconj'): qualifier of the parameter
            to be used for t0
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

        ephem = self.get_ephemeris(component=component, t0=t0, **kwargs)

        if isinstance(phase, list):
            phase = np.array(phase)

        t0 = ephem.get('t0', 0.0)
        period = ephem.get('period', 1.0)
        dpdt = ephem.get('dpdt', 0.0)

        # if changing this, also see parameters.constraint.time_ephem
        # and phoebe.constraints.builtin.phases_to_times
        if dpdt != 0:
            time = t0 + 1./dpdt*(np.exp(dpdt*(phase))-period)
        else:
            time = t0 + (phase)*period

        return time

    def to_times(self, *args, **kwargs):
        """
        Alias to <phoebe.frontend.bundle.Bundle.to_time>.
        """
        return self.to_time(*args, **kwargs)

    @send_if_client
    def add_dataset(self, kind, component=None, **kwargs):
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
        * `return_overwrite` (boolean, optional, default=False): whether to include
            removed parameters due to `overwrite` in the returned ParameterSet.
            Only applicable if `overwrite` is True.
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

        if not isinstance(kwargs['dataset'], str):
            # if dataset is a unicode, that conflicts with copy-for
            # TODO: this really should be replaced with a more elegant handling
            # of unicode within parameters.ParameterSet._check_copy_for
            kwargs['dataset'] = str(kwargs['dataset'])

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
            overwrite_ps = self.remove_dataset(dataset=kwargs['dataset'])
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



        if self.get_value(qualifier='auto_add_figure', context='setting') and kind not in self.filter(context='figure', check_visible=False, check_default=False).exclude(figure=[None], check_visible=False, check_default=False).kinds:
            # then we don't have a figure for this kind yet
            logger.info("calling add_figure(kind='{}') since auto_add_figure@setting=True".format(kind))
            new_fig_params = self.add_figure(kind=kind)
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

        redo_kwargs = deepcopy({k:_to_safe_value(v) for k,v in kwargs.items()})
        redo_kwargs['func'] = func.__name__
        self._add_history(redo_func='add_dataset',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_dataset',
                          undo_kwargs={'dataset': kwargs['dataset']})


        ret_ps = self.filter(dataset=kwargs['dataset'], **_skip_filter_checks)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite', 'return_overwrite'], warning_only=True, ps=ret_ps)

        if new_fig_params is not None:
            ret_ps += new_fig_params

        if kwargs.get('overwrite', False) and kwargs.get('return_overwrite', False):
            ret_ps += overwrite_ps

        return ret_ps

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

        kwargs['context'] = 'dataset'
        if 'kind' in kwargs.keys():
            # since we switched how dataset kinds are named, let's just
            # automatically handle switching to lowercase
            kwargs['kind'] = kwargs['kind'].lower()
        return self.filter(**kwargs)

    def remove_dataset(self, dataset=None, **kwargs):
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

        self._kwargs_checks(kwargs)

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

        ret_ps += ParameterSet(self._handle_dataset_selectparams(return_changes=True))
        # the dataset could have been removed from an existing model which changes options
        # for time_source params if it was a mesh or lp
        ret_ps += ParameterSet(self._handle_figure_time_source_params(return_changes=True))

        # TODO: check to make sure that trying to undo this
        # will raise an error saying this is not undo-able
        self._add_history(redo_func='remove_dataset',
                          redo_kwargs={'dataset': dataset},
                          undo_func=None,
                          undo_kwargs={})

        return ret_ps

    def remove_datasets_all(self):
        """
        Remove all datasets from the bundle.  To remove a single dataset see
        <phoebe.frontend.bundle.Bundle.remove_dataset>.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for dataset in self.datasets:
            removed_ps += self.remove_dataset(dataset=dataset)

        return removed_ps

    def rename_dataset(self, old_dataset, new_dataset):
        """
        Change the label of a dataset attached to the Bundle.

        Arguments
        ----------
        * `old_dataset` (string): current label of the dataset (must exist)
        * `new_dataset` (string): the desired new label of the dataset
            (must not yet exist)

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_dataset` is forbidden or already exists.
        """
        # TODO: raise error if old_component not found?

        self._check_label(new_dataset)
        self._rename_label('dataset', old_dataset, new_dataset)

        ret_ps = self.filter(dataset=new_dataset)

        ret_ps += ParameterSet(self._handle_dataset_selectparams(return_changes=True))
        # Only needed if it was a mesh or lp
        ret_ps += ParameterSet(self._handle_figure_time_source_params(return_changes=True))

        return ret_ps


    def enable_dataset(self, dataset=None, **kwargs):
        """
        Enable a `dataset`.  Datasets that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_fitting (once supported).

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
        kwargs['context'] = 'compute'
        kwargs['dataset'] = dataset
        kwargs['qualifier'] = 'enabled'
        self.set_value_all(value=True, **kwargs)

        self._add_history(redo_func='enable_dataset',
                          redo_kwargs={'dataset': dataset},
                          undo_func='disable_dataset',
                          undo_kwargs={'dataset': dataset})

        return self.get_dataset(dataset=dataset)

    def disable_dataset(self, dataset=None, **kwargs):
        """
        Disable a `dataset`.  Datasets that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_fitting (once supported).

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
        kwargs['context'] = 'compute'
        kwargs['dataset'] = dataset
        kwargs['qualifier'] = 'enabled'
        self.set_value_all(value=False, **kwargs)

        self._add_history(redo_func='disable_dataset',
                          redo_kwargs={'dataset': dataset},
                          undo_func='enable_dataset',
                          undo_kwargs={'dataset': dataset})

        return self.get_dataset(dataset=dataset)

    def add_parameter(self):
        """
        [NOT IMPLEMENTED]

        Add a new parameter to the bundle

        :raises NotImplementedError: because it isn't
        """
        # TODO: don't forget add_history
        raise NotImplementedError

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

        The following are automatically included for all orbits, during
        <phoebe.frontend.bundle.Bundle.add_component> for a
        <phoebe.parameters.component.orbit>:
        * <phoebe.parameters.constraint.asini>
        * <phoebe.parameters.constraint.ecosw>
        * <phoebe.parameters.constraint.esinw>
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

        Arguments
        ------------
        * `*args`: positional arguments can be any one of the following:
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

        redo_kwargs = deepcopy(kwargs)

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

        if 'solve_for' in kwargs.keys():
            # solve_for is a twig, we need to pass the parameter
            kwargs['solve_for'] = self.get_parameter(kwargs['solve_for'], context=['component', 'dataset', 'model'], check_visible=False)

        lhs, rhs, addl_vars, constraint_kwargs = func(self, *func_args, **kwargs)
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

        redo_kwargs['func'] = func.__name__

        self._add_history(redo_func='add_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_constraint',
                          undo_kwargs={'uniqueid': constraint_param.uniqueid})

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

    def remove_constraint(self, twig=None, **kwargs):
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
        redo_kwargs = deepcopy(kwargs)

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
        self._add_history(redo_func='remove_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='add_constraint',
                          undo_kwargs=undo_kwargs)

        return removed_param


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
        redo_kwargs = deepcopy(kwargs)
        undo_kwargs = deepcopy(kwargs)

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
        param.flip_for(solve_for)

        try:
            result = self.run_constraint(uniqueid=param.uniqueid, skip_kwargs_checks=True)
        except Exception as e:
            if param.uniqueid not in self._failed_constraints:
                self._failed_constraints.append(param.uniqueid)

                message_prefix = "Constraint '{}' raised the following error while flipping to solve for '{}'.  Consider flipping the constraint back or changing the value of one of {} until the constraint succeeds.  Original error: ".format(param.twig, solve_for, [p.twig for p in param.vars.to_list()])

                logger.error(message_prefix + str(e))

        self._add_history(redo_func='flip_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='flip_constraint',
                          undo_kwargs=undo_kwargs)

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
        kwargs['context'] = []
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
                constrained_param.set_value(result, force=True)
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


    @send_if_client
    def add_figure(self, kind, **kwargs):
        """
        Add a new figure to the bundle.  If not provided,
        figure` (the name of the new figure) will be created for
        you and can be accessed by the `figure` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        ```py
        b.add_figure(figure.lc)
        ```

        or

        ```py
        b.add_figure('lc', x='phases')
        ```

        Available kinds can be found in <phoebe.parameters.figure> and include:
        * <phoebe.parameters.figure.lc>
        * <phoebe.parameters.figure.rv>
        * <phoebe.parameters.figure.lp>
        * <phoebe.parameters.figure.orb>
        * <phoebe.parameters.figure.mesh>

        See also:
        * <phoebe.frontend.bundle.Bundle.get_figure>
        * <phoebe.frontend.bundle.Bundle.remove_figure>
        * <phoebe.frontend.bundle.Bundle.rename_figure>
        * <phoebe.frontend.bundle.Bundle.run_figure>

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
            an existing component with the same `figure` tag.  If False,
            an error will be raised.
        * `return_overwrite` (boolean, optional, default=False): whether to include
            removed parameters due to `overwrite` in the returned ParameterSet.
            Only applicable if `overwrite` is True.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added
        """

        func = _get_add_func(_figure, kind)

        if sys.version_info[0] == 3:
          fname = func.__name__
        else:
          fname = func.__name__


        if kwargs.get('figure', False) is None:
            # then we want to apply the default below, so let's pop for now
            _ = kwargs.pop('figure')

        kwargs.setdefault('figure',
                          self._default_label(fname+'fig',
                                              **{'context': 'figure',
                                                 'kind': fname}))

        if kwargs.pop('check_label', True):
            self._check_label(kwargs['figure'], allow_overwrite=kwargs.get('overwrite', False))

        params = func(self, **kwargs)


        metawargs = {'context': 'figure',
                     'figure': kwargs['figure'],
                     'kind': fname}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_figure(figure=kwargs['figure'])
            # check the label again, just in case kwargs['figure'] belongs to
            # something other than component
            self.exclude(figure=kwargs['figure'])._check_label(kwargs['figure'], allow_overwrite=False)
        else:
            removed_ps = None

        self._attach_params(params, **metawargs)
        # attach params called _check_copy_for, but only on it's own parameterset
        # self._check_copy_for()

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = fname
        self._add_history(redo_func='add_figure',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_figure',
                          undo_kwargs={'figure': kwargs['figure']})

        # for constraint in constraints:
            # self.add_constraint(*constraint)

        ret_ps = self.filter(figure=kwargs['figure'], check_visible=False, check_default=False)

        self._handle_dataset_selectparams()
        self._handle_model_selectparams()
        self._handle_component_selectparams()
        self._handle_meshcolor_choiceparams()
        self._handle_figure_time_source_params()

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs,
                            additional_allowed_keys=['overwrite', 'return_overwrite'],
                            warning_only=True, ps=ret_ps)

        if kwargs.get('overwrite', False) and kwargs.get('return_overwrite', False):
            ret_ps += overwrite_ps

        return ret_ps

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

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        kwargs['figure'] = figure
        kwargs['context'] = 'figure'
        ret_ps = self.filter(**kwargs).exclude(figure=[None])

        if len(ret_ps.figures) == 0:
            raise ValueError("no figures matched: {}".format(kwargs))
        elif len(ret_ps.figures) > 1:
            raise ValueError("more than one figure matched: {}".format(kwargs))

        return ret_ps

    def remove_figure(self, figure, **kwargs):
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
        return self.remove_parameters_all(**kwargs)

    def rename_figure(self, old_figure, new_figure):
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
            (must not yet exist)

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed figure

        Raises
        --------
        * ValueError: if the value of `new_figure` is forbidden or already exists.
        """
        # TODO: raise error if old_figure not found?

        self._check_label(new_figure)
        self._rename_label('figure', old_figure, new_figure)

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
        * <phoebe.frontend.bundle.Bundle.add_figure>
        * <phoebe.frontend.bundle.Bundle.get_figure>
        * <phoebe.frontend.bundle.Bundle.remove_figure>
        * <phoebe.frontend.bundle.Bundle.rename_figure>

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


        qmap = {'color': 'c'}

        if isinstance(figure, list) or isinstance(figure, tuple):
            figures = figure
            show = kwargs.pop('show', False)
            save = kwargs.pop('save', False)
            animate = kwargs.pop('animate', False)
            for figure in figures:
                self.run_figure(figure=figure, **{k: v.get(figure) if isinstance(v, dict) and figure in v.keys() else v for k,v in kwargs.items()})

            return self._show_or_save(save, show, animate, **kwargs)


        fig_ps = self.get_figure(figure=figure, **kwargs)
        if len(fig_ps.figures) == 0:
            raise ValueError("no figure found")
        elif len(fig_ps.figures) > 1:
            raise ValueError("more than one figure found")

        kwargs['check_default'] = False
        kwargs['check_visible'] = False

        ds_kind = fig_ps.kind
        ds_same_kind = self.filter(context='dataset', kind=ds_kind).datasets
        ml_same_kind = self.filter(context='model', kind=ds_kind).models
        comp_same_kind = self.filter(context=['dataset', 'model'], kind=ds_kind).components

        kwargs.setdefault('kind', ds_kind)
        if 'contexts' in fig_ps.qualifiers:
            kwargs.setdefault('context', fig_ps.get_value(qualifier='contexts', expand=True, **_skip_filter_checks))
        else:
            kwargs['context'] = 'model'
        kwargs.setdefault('dataset', fig_ps.get_value(qualifier='datasets', expand=True, **_skip_filter_checks))
        kwargs.setdefault('model', [None] + fig_ps.get_value(qualifier='models', expand=True, **_skip_filter_checks))
        if 'components' in fig_ps.qualifiers:
            kwargs.setdefault('component', fig_ps.get_value(qualifier='components', expand=True, **_skip_filter_checks))
        kwargs.setdefault('legend', fig_ps.get_value(qualifier='legend', **_skip_filter_checks))

        for q in ['draw_sidebars', 'uncover', 'highlight']:
            if q in fig_ps.qualifiers:
                kwargs.setdefault(q, fig_ps.get_value(qualifier=q, **_skip_filter_checks))

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



        kwargs.setdefault('tight_layout', True)
        logger.info("calling plot(**{})".format(kwargs))
        return self.plot(**kwargs)


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
            <phoebe.frontend.bundle.Bundle.run_checks> before computing the model.
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

        compute_ps = self.get_compute(compute)
        # we'll add 'bol' to the list of default datasets... but only if bolometric is needed for irradiation
        needs_bol = 'irrad_method' in compute_ps.qualifiers and compute_ps.get_value(qualifier='irrad_method', irrad_method=kwargs.get('irrad_method', None), **_skip_filter_checks) != 'none'

        datasets = kwargs.pop('dataset', self.datasets + ['bol'] if needs_bol else self.datasets)
        components = kwargs.pop('component', self.components)

        # don't allow things like model='mymodel', etc
        forbidden_keys = parameters._meta_fields_filter
        if not kwargs.get('skip_checks', False):
            self._kwargs_checks(kwargs, additional_allowed_keys=['skip_checks', 'overwrite'], additional_forbidden_keys=forbidden_keys)

        if not kwargs.get('skip_checks', False):
            report = self.run_checks(compute=compute, allow_skip_constraints=False,
                                     raise_logger_warning=True, raise_error=True,
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
                ld_coeffs_ret["{}@{}@{}".format('ld_coeffs{}'.format(bol_suffix), ldcs_param.component, 'component' if is_bol else ldcs_param.dataset)] = self.get_value(qualifier='ld_coeffs{}'.format(bol_suffix), dataset=ldcs_param.dataset, component=ldcs_param.component, check_visible=False)
                continue
            elif ld_mode == 'lookup':
                ldcs = ldcs_param.get_value(check_visible=False)
                ld_func = self.get_value(qualifier='ld_func{}'.format(bol_suffix), dataset=ldcs_param.dataset, component=ldcs_param.component, check_visible=False)
                if is_bol:
                    passband = 'Bolometric:900-40000'
                else:
                    passband = self.get_value(qualifier='passband', dataset=ldcs_param.dataset, check_visible=False)

                try:
                    atm = self.get_value(qualifier='atm', compute=compute, component=ldcs_param.component, check_visible=False)
                except ValueError:
                    # not all backends have atm as an option
                    logger.warning("backend compute='{}' has no 'atm' option: falling back on ck2004 for ld_coeffs{} interpolation".format(compute, bol_suffix))
                    atm = 'ck2004'

                if ldcs == 'auto':
                    if atm in ['extern_atmx', 'extern_planckint', 'blackbody']:
                        ldcs = 'ck2004'
                    else:
                        ldcs = atm

                pb = get_passband(passband, content='{}:ld'.format(ldcs))
                teff = self.get_value(qualifier='teff', component=ldcs_param.component, context='component', unit='K', check_visible=False)
                logg = self.get_value(qualifier='logg', component=ldcs_param.component, context='component', check_visible=False)
                abun = self.get_value(qualifier='abun', component=ldcs_param.component, context='component', check_visible=False)
                if is_bol:
                    photon_weighted = False
                else:
                    photon_weighted = self.get_value(qualifier='intens_weighting', dataset=ldcs_param.dataset, context='dataset', check_visible=False) == 'photon'
                logger.info("interpolating {} ld_coeffs for dataset='{}' component='{}' passband='{}' from ld_coeffs_source='{}'".format(ld_func, ldcs_param.dataset, ldcs_param.component, passband, ldcs))
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
                    self.set_value(qualifier='ld_coeffs{}'.format(bol_suffix), component=ldcs_param.component, dataset=ldcs_param.dataset, check_visible=False, value=ld_coeffs)
            else:
                raise NotImplementedError("compute_ld_coeffs not implemented for ld_mode{}='{}'".format(bol_suffix, ld_mode))

        return ld_coeffs_ret

    def _compute_system(self, compute=None, datasets=None, compute_l3=False, compute_l3_frac=False, compute_extrinsic=False, **kwargs):
        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        compute_kind = self.get_compute(compute).kind

        if compute_kind in ['legacy']:
            kwargs.setdefault('distortion_method', 'roche')
        elif compute_kind in ['jktebop']:
            kwargs.setdefault('distortion_method', 'sphere')

        # temporarily disable interactive_checks, check_default, and check_visible
        conf_interactive_checks = conf.interactive_checks
        if conf_interactive_checks:
            logger.debug("temporarily disabling interactive_checks")
            conf._interactive_checks = False

        conf_check_default = conf.check_default
        if conf_check_default:
            logger.debug("temporarily disabling check_default")
            conf.check_default_off()

        conf_check_visible = conf.check_visible
        if conf_check_visible:
            logger.debug("temporarily disabling check_visible")
            conf.check_visible_off()

        def restore_conf():
            # restore user-set interactive checks
            if conf_interactive_checks:
                logger.debug("restoring interactive_checks={}".format(conf_interactive_checks))
                conf._interactive_checks = conf_interactive_checks

            if conf_check_visible:
                logger.debug("restoring check_visible")
                conf.check_visible_on()

            if conf_check_default:
                logger.debug("restoring check_default")
                conf.check_default_on()

        system_compute = compute if compute_kind=='phoebe' else None
        logger.debug("creating system with compute={} kwargs={}".format(system_compute, kwargs))
        try:
            system = backends.PhoebeBackend()._create_system_and_compute_pblums(self, system_compute, datasets=datasets, compute_l3=compute_l3, compute_l3_frac=compute_l3_frac, compute_extrinsic=compute_extrinsic, reset=False, lc_only=False, **kwargs)
        except Exception as err:
            restore_conf()
            raise

        restore_conf()

        return system

    def compute_l3s(self, compute=None, set_value=False, **kwargs):
        """
        Compute third lights (`l3`) that will be applied to the system from
        fractional third light (`l3_frac`) and vice-versa by assuming that the
        total system flux is equivalent to the sum of the extrinsic (including
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
        * `set_value` (bool, optional, default=False): apply the computed
            values to the respective `l3` or `l3_frac` parameters (even if not
            currently visible).
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `**kwargs`: any additional kwargs are sent to override compute options.

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

        # don't allow things like model='mymodel', etc
        if not kwargs.get('skip_checks', False):
            forbidden_keys = parameters._meta_fields_filter
            self._kwargs_checks(kwargs, additional_allowed_keys=['system', 'skip_checks'], additional_forbidden_keys=forbidden_keys)

        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        if not kwargs.get('skip_checks', False):
            report = self.run_checks(compute=compute, allow_skip_constraints=False,
                                     raise_logger_warning=True, raise_error=True,
                                     **kwargs)

        system = kwargs.get('system', self._compute_system(compute=compute, datasets=datasets, compute_l3=True, compute_l3_frac=True, **kwargs))

        l3s = {}
        for dataset in datasets:
            l3_mode = self.get_value(qualifier='l3_mode', dataset=dataset)
            if l3_mode == 'flux':
                l3_frac = system.l3s[dataset]['frac']
                l3s['l3_frac@{}'.format(dataset)] = l3_frac
                if set_value:
                    self.set_value(qualifier='l3_frac', dataset=dataset, check_visible=False, value=l3_frac)

            elif l3_mode == 'fraction':
                l3_flux = system.l3s[dataset]['flux'] * u.W / u.m**2
                l3s['l3@{}'.format(dataset)] = l3_flux

                if set_value:
                    self.set_value(qualifier='l3', dataset=dataset, check_visible=False, value=l3_flux)

            else:
                raise NotImplementedError("l3_mode='{}' not supported.".format(l3_mode))

        return l3s

    def compute_pblums(self, compute=None, pblum=True, pblum_ext=True,
                       pbflux=False, pbflux_ext=False,
                       set_value=False, **kwargs):
        """
        Compute the passband luminosities that will be applied to the system,
        following all coupling, etc, as well as all relevant compute options
        (ntriangles, distortion_method, etc), third light, and distance.
        The exposed passband luminosities (and any coupling) are computed at
        t0@system.

        This method allows for computing both intrinsic and extrinsic luminosities.
        Note that pblum scaling is computed (and applied to flux scaling) based
        on intrinsic luminosities (`pblum`).

        Any `dataset` which does not support pblum scaling (rv or lp dataset,
        for example), will have their absolute intensities exposed.

        Note that luminosities cannot be exposed for any dataset in which
        `pblum_mode` is 'dataset-scaled' as the entire light curve must be
        computed prior to scaling.  These will be excluded from the output
        without error, but with a warning message in the <phoebe.logger>, if
        enabled.

        Additionally, an estimate for the total fluxes `pbflux` and `pbflux_ext`
        can optionally be computed.  These will also be computed at t0@system,
        under the spherical assumption where `pbflux = sum(pblum / (4 pi)) + l3`
        or `pbflux_ext = sum(pblum_ext / (4 pi)) + l3`.  Note that in either case,
        the translation from `l3_frac` to `l3` (when necessary) will include
        extrinsic effects.  See also <phoebe.frontend.bundle.Bundle.compute_l3s>.

        Note about eclipses: `pbflux` and `pbflux_ext` estimates will not include
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
        * for backends without `atm` compute options, 'ck2004' will be used.
        * for backends without `mesh_method` compute options, the most appropriate
            method will be chosen.  'roche' will be used whenever applicable,
            otherwise 'sphere' will be used.

        Arguments
        ------------
        * `compute` (string, optional, default=None): label of the compute
            options (not required if only one is attached to the bundle).
        * `pblum` (bool, optional, default=True): whether to include
            intrinsic (excluding irradiation & features) pblums.  These
            will be exposed in the returned dictionary as pblum@component@dataset.
        * `pblum_ext` (bool, optional, default=True): whether to include
            extrinsic (irradiation & features) pblums.  These will
            be exposed as pblum_ext@component@dataset.
        * `pbflux` (bool, optional, default=False): whether to include
            intrinsic per-system passband fluxes.  These include third-light
            (from the l3 or l3_frac parameter), but are estimated based
            on intrinsic `pblum`.  These will be exposed as pbflux@dataset.
        * `pbflux_ext` (bool, optional, default=False): whether to include
            extrinsic per-system passband fluxes.  These include third-light
            (from the l3 or l3_frac parameter), and are estimated based on
            extrinsic `pblum_ext`.  These will be exposed as pbflux_ext@dataset.
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
            currently visible).  Note that extrinsic values (`pblum_ext` and
            `pbflux_ext`) are not input parameters to the
            model, so are not set.  This is often used internally to handle
            various options for pblum_mode for alternate backends that require
            passband luminosities or surface brightnesses as input, but is not
            ever required to be called manually.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `**kwargs`: any additional kwargs are sent to override compute options.

        Returns
        ----------
        * (dict) computed pblums in a dictionary with keys formatted as
            pblum@component@dataset (for intrinsic pblums) or
            pblum_ext@component@dataset (for extrinsic pblums) and the pblums
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
            <phoebe.frontend.bundle.Bundle.run_checks>.
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

        # don't allow things like model='mymodel', etc
        forbidden_keys = parameters._meta_fields_filter
        if not kwargs.get('skip_checks', False):
            self._kwargs_checks(kwargs, additional_allowed_keys=['system', 'skip_checks', 'overwrite'], additional_forbidden_keys=forbidden_keys)

        # check to make sure value of passed compute is valid
        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        # make sure we pass system checks
        if not kwargs.get('skip_checks', False):
            report = self.run_checks(compute=compute, allow_skip_constraints=False,
                                     raise_logger_warning=True, raise_error=True,
                                     **kwargs)

        # determine datasets which need intensities computed and check to make
        # sure all passed datasets are passband-dependent
        pblum_datasets = datasets
        for dataset in datasets:
            if not len(self.filter(qualifier='passband', dataset=dataset)):
                if dataset not in self.datasets:
                    raise ValueError("dataset '{}' is not a valid dataset attached to the bundle".format(dataset))
                raise ValueError("dataset '{}' is not passband-dependent".format(dataset))
            for pblum_ref_param in self.filter(qualifier='pblum_dataset', dataset=dataset).to_list():
                ref_dataset = pblum_ref_param.get_value()
                if ref_dataset in self.datasets and ref_dataset not in pblum_datasets:
                    # then we need to compute the system at this dataset too,
                    # even though it isn't requested to be returned
                    pblum_datasets.append(ref_dataset)

            ds_kind = self.get_dataset(dataset=dataset, check_visible=False).kind
            if ds_kind == 'lc' and self.get_value(qualifier='pblum_mode', dataset=dataset, check_visible=False) == 'dataset-scaled':
                logger.warning("cannot expose pblum for dataset={} with pblum_mode@{}='dataset-scaled'".format(dataset, dataset))
                pblum_datasets.remove(dataset)

        t0 = self.get_value(qualifier='t0', context='system', unit=u.d)

        # we'll need to make sure we've done any necessary interpolation if
        # any ld_bol or ld_mode_bol are set to 'lookup'.
        self.compute_ld_coeffs(compute=compute, set_value=True, **kwargs)

        ret = {}
        l3s = None
        for compute_extrinsic in [True, False]:
            # we need to compute the extrinsic case if we're requesting pblum_ext
            # or pbflux_ext or if we're requesting pbflux but l3s need to be
            # converted (as those need to be translated with extrinsic enabled)
            if compute_extrinsic and not (pblum_ext or pbflux_ext or (pbflux and len(self.filter(qualifier='l3_mode', dataset=datasets, value='fraction')))):
                continue
            if not compute_extrinsic and not (pblum or pbflux):
                continue

            # TODO: can we prevent rebuilding the entire system the second time if both intrinsic and extrinsic are True?
            compute_l3 = compute_extrinsic and (pbflux_ext or pbflux)
            logger.debug("b._compute_system(compute={}, datasets={}, compute_l3={}, compute_extrinsic={}, kwargs={})".format(compute, pblum_datasets, compute_l3, compute_extrinsic, kwargs))
            system = kwargs.get('system', self._compute_system(compute=compute, datasets=pblum_datasets, compute_l3=compute_l3, compute_extrinsic=compute_extrinsic, **kwargs))

            if compute_l3:
                # these needed to be computed with compute_extrinsic=True even for pbflux instrinsic
                l3s = {dataset: system.l3s[dataset]['flux'] for dataset in datasets} # in u.W/u.m**2

            for dataset in datasets:
                pbflux_this_dataset = 0

                # TODO: can we get away with skipping this in some cases?  If we
                # skipped the compute_extrinsic=True case, then we should
                # already have these with ignore_effects=True from computing the
                # scaling
                # Technically we only need to do this if compute_extrinsic as of
                # right now, since there is nothing in _populate_lc which
                # affects Inorms (though boosting affects Imus).
                logger.debug("computing observables with ignore_effects={} for {}".format(not compute_extrinsic, dataset))
                system.populate_observables(t0, ['lc'], [dataset],
                                            ignore_effects=not compute_extrinsic)


                system_items = {}
                for component, item in system.items():
                    system_items[component] = item
                    if hasattr(item, '_halves'):
                        # then we also want to iterate over the envelope halves
                        for half in item._halves:
                            system_items[half.component] = half

                for component, star in system_items.items():
                    if component not in components:
                        continue

                    pblum = float(star.compute_luminosity(dataset))
                    pbflux_this_dataset += pblum / (4*np.pi)

                    if (compute_extrinsic and pblum_ext) or (not compute_extrinsic and pblum):
                        if not compute_extrinsic and set_value:
                            self.set_value(qualifier='pblum', component=component, dataset=dataset, context='dataset', check_visible=False, value=pblum*u.W)

                        ret["{}@{}@{}".format('pblum_ext' if compute_extrinsic else 'pblum', component, dataset)] = pblum*u.W

                if (compute_extrinsic and pbflux_ext) or (not compute_extrinsic and pbflux):

                    pbflux_this_dataset /= self.get_value(qualifier='distance', context='system', unit=u.m)**2

                    if l3s is None:
                        # then we didn't need to compute l3s, so we can pull straight from the parameter
                        pbflux_this_dataset += self.get_value(qualifier='l3', dataset=dataset, context='dataset', unit=u.W/u.m**2)
                    else:
                        pbflux_this_dataset += l3s[dataset]


                    if not compute_extrinsic and set_value:
                        self.set_value(qualifier='pbflux', dataset=dataset, context='dataset', check_visible=False, value=pbflux_this_dataset*u.W/u.m**2)

                    ret["{}@{}".format('pbflux_ext' if compute_extrinsic else 'pbflux', dataset)] = pbflux_this_dataset*u.W/u.m**2

        return ret

    def _compute_necessary_values(self, computeparams, **kwargs):
        # we'll manually disable skip_checks anyways to avoid them being done twice
        _ = kwargs.pop('backend', None)
        _ = kwargs.pop('skip_checks', None)
        compute = computeparams.compute

        if computeparams.kind == 'phoebe' and computeparams.get_value(qualifier='irrad_method', **_skip_filter_checks) !='none':
            # then all we need to do is handle any ld_mode_bol=='lookup'
            self.compute_ld_coeffs(compute, dataset=['bol'], set_value=True, skip_checks=True)
            return

        enabled_datasets = computeparams.filter(qualifier='enabled', value=True).datasets
        # handle any limb-darkening interpolation
        dataset_compute_ld_coeffs = self.filter(dataset=enabled_datasets, qualifier='ld_coeffs_source').exclude(value='none').datasets
        if computeparams.kind == 'photodynam':
            # then we're ignoring anything that isn't quadratic anyways
            dataset_compute_ld_coeffs = self.filter(dataset=dataset_compute_ld_coeffs, qualifier='ld_func', value='quadratic').datasets

        if len(dataset_compute_ld_coeffs):
            logger.warning("{} does not natively support interpolating ld coefficients.  These will be interpolated by PHOEBE 2 and then passed to {}.".format(computeparams.kind, computeparams.kind))
            logger.debug("calling compute_ld_coeffs(compute={}, dataset={}, set_value=True, skip_checks=True, **{})".format(dataset_compute_ld_coeffs, compute, kwargs))
            self.compute_ld_coeffs(compute, dataset=dataset_compute_ld_coeffs, set_value=True, skip_checks=True, **kwargs)

        # handle any necessary pblum computations
        allowed_pblum_modes = ['decoupled', 'component-coupled'] if computeparams.kind == 'legacy' else ['decoupled']
        dataset_compute_pblums = self.filter(dataset=enabled_datasets, qualifier='pblum_mode').exclude(value=allowed_pblum_modes).datasets
        if len(dataset_compute_pblums):
            logger.warning("{} does not natively support pblum_mode={}.  pblum values will be computed by PHOEBE 2 and then passed to {}.".format(computeparams.kind, [p.get_value() for p in self.filter(qualifier='pblum_mode').exclude(value=allowed_pblum_modes).to_list()], computeparams.kind))
            logger.debug("calling compute_pblums(compute={}, dataset={}, pblum=True, pblum_ext=False, pbflux=True, pbflux_ext=False, set_value=True, skip_checks=True, **{})".format(compute, dataset_compute_pblums, kwargs))
            self.compute_pblums(compute, dataset=dataset_compute_pblums, pblum=True, pblum_ext=False, pbflux=True, pbflux_ext=False, set_value=True, skip_checks=True, **kwargs)

        # handle any necessary l3 computations
        if computeparams.kind == 'ellc':
            dataset_compute_l3s = self.filter(dataset=enabled_datasets, qualifier='l3_mode', value='flux').datasets
        else:
            dataset_compute_l3s = self.filter(dataset=enabled_datasets, qualifier='l3_mode', value='fraction').datasets
        if computeparams.kind == 'legacy':
            # legacy support either mode, but all must be the same
            l3_modes = [p.value for p in self.filter(qualifier='l3_mode').to_list()]
            if len(list(set(l3_modes))) <= 1:
                dataset_compute_l3s = []

        if len(dataset_compute_l3s):
            if computeparams.kind == 'legacy':
                logger.warning("{} does not natively support mixed values for l3_mode.  l3 values will be computed by PHOEBE 2 and then passed to {}.".format(computeparams.kind, computeparams.kind))
            elif computeparams.kind == 'ellc':
                logger.warning("{} does not natively support l3_mode='flux'.  l3_frac values will be computed by PHOEBE 2 and then passed to {}.".format(computeparams.kind, computeparams.kind))
            else:
                logger.warning("{} does not natively support l3_mode='fraction'.  l3 values will be computed by PHOEBE 2 and then passed to {}.".format(computeparams.kind, computeparams.kind))
            logger.debug("calling compute_l3s(compute={}, dataset={}, set_value=True, skip_checks=True, **{})".format(compute, dataset_compute_l3s, kwargs))
            self.compute_l3s(compute, dataset=dataset_compute_l3s, set_value=True, skip_checks=True, **kwargs)


    @send_if_client
    def add_compute(self, kind='phoebe', **kwargs):
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
        * `return_overwrite` (boolean, optional, default=False): whether to include
            removed parameters due to `overwrite` in the returned ParameterSet.
            Only applicable if `overwrite` is True.
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

        self._check_label(kwargs['compute'], allow_overwrite=kwargs.get('overwrite', False))

        params = func(**kwargs)
        # TODO: similar kwargs logic as in add_dataset (option to pass dict to
        # apply to different components this would be more complicated here if
        # allowing to also pass to different datasets

        metawargs = {'context': 'compute',
                     'kind': func.__name__,
                     'compute': kwargs['compute']}

        if kwargs.get('overwrite', False):
            overwrite_ps = self.remove_compute(compute=kwargs['compute'])
            # check the label again, just in case kwargs['compute'] belongs to
            # something other than compute
            self._check_label(kwargs['compute'], allow_overwrite=False)

        logger.info("adding {} '{}' compute to bundle".format(metawargs['kind'], metawargs['compute']))
        self._attach_params(params, **metawargs)

        if kind=='phoebe' and 'ntriangles' not in kwargs.keys():
            # the default for ntriangles in compute.py is 1500, we want 3000 for an envelope
            for envelope in self.hierarchy.get_envelopes():
                self.set_value(qualifier='ntriangles', compute=kwargs['compute'], component=envelope, value=3000, check_visible=False)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.__name__
        self._add_history(redo_func='add_compute',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_compute',
                          undo_kwargs={'compute': kwargs['compute']})


        ret_ps = self.get_compute(check_visible=False, check_default=False, **metawargs)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite', 'return_overwrite'], warning_only=True, ps=ret_ps)

        self._handle_compute_selectparams()

        if kwargs.get('overwrite', False) and kwargs.get('return_overwrite', False):
            ret_ps += overwrite_ps

        return ret_ps

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
        kwargs['context'] = 'compute'
        return self.filter(**kwargs)

    def remove_compute(self, compute, **kwargs):
        """
        Remove a 'compute' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `compute` (string): the label of the compute options to be removed.
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
        ret_ps += ParameterSet(self._handle_compute_selectparams(return_changes=True))
        return ret_ps

    def remove_computes_all(self):
        """
        Remove all compute options from the bundle.  To remove a single set
        of compute options see <phoebe.frontend.bundle.Bundle.remove_compute>.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for compute in self.computes:
            removed_ps += self.remove_compute(compute)
        return removed_ps

    def rename_compute(self, old_compute, new_compute):
        """
        Change the label of compute options attached to the Bundle.

        Arguments
        ----------
        * `old_compute` (string): current label of the compute options (must exist)
        * `new_compute` (string): the desired new label of the compute options
            (must not yet exist)

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed dataset

        Raises
        --------
        * ValueError: if the value of `new_compute` is forbidden or already exists.
        """
        # TODO: raise error if old_compute not found?

        self._check_label(new_compute)
        self._rename_label('compute', old_compute, new_compute)

        return self.filter(compute=new_compute)

    def _prepare_compute(self, compute, model, **kwargs):
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
            if self.get_value(qualifier='detached_job', model=model, context='model', default='loaded') != 'loaded':
                raise ValueError("model '{}' cannot be overwritten until it is complete and loaded.")
            if model=='latest':
                logger.warning("overwriting model: {}".format(model))
            else:
                logger.info("overwriting model: {}".format(model))

            do_create_fig_params = kwargs.get('do_create_fig_params', False)

            overwrite_ps = self.remove_model(model, remove_figure_params=do_create_fig_params)
            # check the label again, just in case model belongs to something
            # other than model/figure

            self.exclude(context='figure')._check_label(model, allow_overwrite=False)

        else:
            do_create_fig_params = kwargs.get('do_create_fig_params', True)


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
        if isinstance(compute, unicode):
            compute = str(compute)

        if isinstance(compute, str):
            computes = [compute]
        else:
            computes = compute

        # if interactive mode was ever off, let's make sure all constraints
        # have been run before running system checks or computing the model
        changed_params = self.run_delayed_constraints()

        # any kwargs that were used just to filter for get_compute should  be
        # removed so that they aren't passed on to all future get_value(...
        # **kwargs) calls
        computes_ps = self.get_compute(compute=compute, **kwargs)
        for k in parameters._meta_fields_filter:
            if k in kwargs.keys():
                dump = kwargs.pop(k)

        # we'll wait to here to run kwargs and system checks so that
        # add_compute is already called if necessary
        allowed_kwargs = ['skip_checks', 'jobid', 'overwrite', 'return_overwrite', 'max_computations']
        if conf.devel:
            allowed_kwargs += ['mesh_init_phi']
        self._kwargs_checks(kwargs, allowed_kwargs, ps=computes_ps)

        if not kwargs.get('skip_checks', False):
            report = self.run_checks(compute=compute, allow_skip_constraints=False,
                                     raise_logger_warning=True, raise_error=True,
                                     **kwargs)

        # let's first make sure that there is no duplication of enabled datasets
        datasets = []
        # compute_ so we don't write over compute which we need if detach=True
        for compute_ in computes:
            # TODO: filter by value instead of if statement once implemented
            for enabled_param in self.filter(qualifier='enabled',
                                             compute=compute_,
                                             context='compute',
                                             check_visible=False).to_list():
                if enabled_param.feature is None and enabled_param.get_value():
                    item = (enabled_param.dataset, enabled_param.component)
                    if item in datasets:
                        raise ValueError("dataset {}@{} is enabled in multiple compute options".format(item[0], item[1]))
                    datasets.append(item)


        return model, computes, datasets, do_create_fig_params, changed_params, overwrite_ps


    def _write_export_compute_script(self, script_fname, out_fname, compute, model, do_create_fig_params, import_from_older, kwargs):
        """
        """
        f = open(script_fname, 'w')
        f.write("import os; os.environ['PHOEBE_ENABLE_PLOTTING'] = 'FALSE'; os.environ['PHOEBE_ENABLE_SYMPY'] = 'FALSE'; os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE';\n")
        f.write("import phoebe; import json\n")
        # TODO: can we skip the history context?  And maybe even other models
        # or datasets (except times and only for run_compute but not run_fitting)
        f.write("bdict = json.loads(\"\"\"{}\"\"\", object_pairs_hook=phoebe.utils.parse_json)\n".format(json.dumps(self.exclude(context=['model', 'figure', 'constraint'], **_skip_filter_checks).to_json(exclude=['description', 'advanced']))))
        f.write("b = phoebe.open(bdict, import_from_older={})\n".format(import_from_older))
        # TODO: make sure this works with multiple computes
        compute_kwargs = list(kwargs.items())+[('compute', compute), ('model', str(model)), ('do_create_fig_params', do_create_fig_params)]
        compute_kwargs_string = ','.join(["{}={}".format(k,"\'{}\'".format(str(v)) if (isinstance(v, str) or isinstance(v, unicode)) else v) for k,v in compute_kwargs])
        f.write("model_ps = b.run_compute({})\n".format(compute_kwargs_string))
        # as the return from run_compute just does a filter on model=model,
        # model_ps here should include any created figure parameters
        if out_fname is not None:
            f.write("model_ps.save('{}', incl_uniqueid=True)\n".format(out_fname))
        else:
            f.write("import sys\n")
            f.write("model_ps.save(sys.argv[0]+'.out', incl_uniqueid=True)\n")

        f.close()

        return script_fname, out_fname

    def export_compute(self, script_fname, out_fname=None,
                       compute=None, model=None,
                       import_from_older=False, **kwargs):
        """
        NEW in PHOEBE 2.2

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
        * `import_from_older` (boolean, optional, default=False): whether to allow
            the script to run on a newer version of PHOEBE.  If True and executing
            the outputed script (`script_fname`) on a newer version of PHOEBE,
            the bundle will attempt to migrate to the newer version.  If False,
            an error will be raised when attempting to run the script.  See
            also: <phoebe.frontend.bundle.Bundle.open>.
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks> before computing the model.
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
        model, computes, datasets, do_create_fig_params, changed_params, overwrite_ps = self._prepare_compute(compute, model, **kwargs)
        script_fname, out_fname = self._write_export_compute_script(script_fname, out_fname, compute, model, do_create_fig_params, import_from_older, kwargs)
        return script_fname, out_fname


    @send_if_client
    def run_compute(self, compute=None, model=None, detach=False,
                    times=None, **kwargs):
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
        * `detach` (bool, optional, default=False, EXPERIMENTAL):
            whether to detach from the computation run,
            or wait for computations to complete.  If detach is True, see
            <phoebe.frontend.bundle.Bundle.get_model> and
            <phoebe.parameters.JobParameter>
            for details on how to check the job status and retrieve the results.
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
        * `return_overwrite` (boolean, optional, default=False): whether to include
            removed parameters due to `overwrite` in the returned ParameterSet.
            Only applicable if `overwrite` is True (or defaults to True if
            `model` is 'latest').
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
        * `max_computations` (int, optional, default=None): maximum
            number of computations to allow.  If more are detected, an error
            will be raised before the backend begins computations.
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
            <phoebe.frontend.bundle.Bundle.run_checks>
        * ValueError: if any given dataset is enabled in more than one set of
            compute options sent to run_compute.
        """
        if isinstance(detach, str) or isinstance(detach, unicode):
            # then we want to temporarily go in to client mode
            raise NotImplementedError("detach currently must be a bool")
            self.as_client(server=detach)
            self.run_compute(compute=compute, model=model, times=times, **kwargs)
            self.as_client(False)
            return self.get_model(model)

        if isinstance(times, float) or isinstance(times, int):
            times = [times]

        model, computes, datasets, do_create_fig_params, changed_params, overwrite_ps = self._prepare_compute(compute, model, **kwargs)

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
            script_fname, out_fname = self._write_export_compute_script(script_fname, out_fname, compute, model, do_create_fig_params, False, kwargs)

            script_fname = os.path.abspath(script_fname)
            cmd = mpi.detach_cmd.format(script_fname)
            # TODO: would be nice to catch errors caused by the detached script...
            # but that would probably need to be the responsibility of the
            # jobparam to return a failed status and message.
            # Unfortunately right now an error just results in the job hanging.
            f = open(err_fname, 'w')
            subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=f)

            # create model parameter and attach (and then return that instead of None)
            job_param = JobParameter(self,
                                     location=os.path.dirname(script_fname),
                                     status_method='exists',
                                     retrieve_method='local',
                                     uniqueid=jobid)

            metawargs = {'context': 'model', 'model': model}
            self._attach_params([job_param], **metawargs)

            if isinstance(detach, str):
                self.save(detach)

            if not detach:
                return job_param.attach()
            else:
                logger.info("detaching from run_compute.  Call get_model('{}').attach() to re-attach".format(model))

            # TODO: make sure the figureparams are returned when attaching

            # return self.get_model(model)
            self._handle_model_selectparams()
            # return self.filter(model=model, check_visible=False, check_default=False)

            if kwargs.get('overwrite', model=='latest') and kwargs.get('return_overwrite', False) and overwrite_ps is not None:
                return ParameterSet([job_param]) + overwrite_ps

            return job_param

        # temporarily disable interactive_checks, check_default, and check_visible
        conf_interactive_checks = conf.interactive_checks
        if conf_interactive_checks:
            logger.debug("temporarily disabling interactive_checks")
            conf._interactive_checks = False

        conf_check_default = conf.check_default
        if conf_check_default:
            logger.debug("temporarily disabling check_default")
            conf.check_default_off()

        conf_check_visible = conf.check_visible
        if conf_check_visible:
            logger.debug("temporarily disabling check_visible")
            conf.check_visible_off()

        def restore_conf():
            # restore user-set interactive checks
            if conf_interactive_checks:
                logger.debug("restoring interactive_checks={}".format(conf_interactive_checks))
                conf._interactive_checks = conf_interactive_checks

            if conf_check_visible:
                logger.debug("restoring check_visible")
                conf.check_visible_on()

            if conf_check_default:
                logger.debug("restoring check_default")
                conf.check_default_on()


        try:
            for compute in computes:

                computeparams = self.get_compute(compute=compute)

                if not computeparams.kind:
                    raise KeyError("could not recognize backend from compute: {}".format(compute))

                logger.info("running {} backend to create '{}' model".format(computeparams.kind, model))
                compute_class = getattr(backends, '{}Backend'.format(computeparams.kind.title()))
                # compute_func = getattr(backends, computeparams.kind)

                metawargs = {'compute': compute, 'model': model, 'context': 'model'}  # dataset, component, etc will be set by the compute_func

                params = compute_class().run(self, compute, times=times, **kwargs)


                # average over any exposure times before attaching parameters
                if computeparams.kind == 'phoebe':
                    # TODO: we could eventually do this for all backends - we would
                    # just need to copy the computeoption parameters into each backend's
                    # compute PS, and include similar logic for oversampling that is
                    # currently in backends._extract_info_from_bundle_by_time into
                    # backends._extract_info_from_bundle_by_dataset.  We'd also
                    # need to make sure that exptime is not being passed to any
                    # alternate backend - and ALWAYS handle it here
                    for dataset in params.datasets:
                        # not all dataset-types currently support exposure times.
                        # Once they do, this ugly if statement can be removed
                        if len(self.filter(dataset=dataset, qualifier='exptime')):
                            exptime = self.get_value(qualifier='exptime', dataset=dataset, context='dataset', unit=u.d)
                            if exptime > 0:
                                logger.info("handling fti for dataset='{}'".format(dataset))
                                if self.get_value(qualifier='fti_method', dataset=dataset, compute=compute, context='compute', **kwargs)=='oversample':
                                    times_ds = self.get_value(qualifier='compute_times', dataset=dataset, context='dataset')
                                    if not len(times_ds):
                                        times_ds = self.get_value(qualifier='times', dataset=dataset, context='dataset')
                                    # exptime = self.get_value(qualifier='exptime', dataset=dataset, context='dataset', unit=u.d)
                                    fti_oversample = self.get_value(qualifier='fti_oversample', dataset=dataset, compute=compute, context='compute', check_visible=False, **kwargs)
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
                                    times_oversampled_sorted = params.get_value(qualifier='times', dataset=dataset)
                                    fluxes_oversampled = params.get_value(qualifier='fluxes', dataset=dataset)

                                    for i,t in enumerate(times_ds):
                                        # rebuild the unsorted oversampled times - see backends._extract_from_bundle_by_time
                                        # TODO: try to optimize this by having these indices returned by the backend itself
                                        times_oversampled_this = np.linspace(t-exptime/2., t+exptime/2., fti_oversample)
                                        sample_inds = np.searchsorted(times_oversampled_sorted, times_oversampled_this)

                                        fluxes[i] = np.mean(fluxes_oversampled[sample_inds])

                                    params.set_value(qualifier='times', dataset=dataset, value=times_ds)
                                    params.set_value(qualifier='fluxes', dataset=dataset, value=fluxes)


                self._attach_params(params, check_copy_for=False, **metawargs)

                def _scale_fluxes(model_fluxes, scale_factor):
                    return model_fluxes * scale_factor

                # scale fluxes whenever pblum_mode = 'dataset-scaled'
                for param in self.filter(qualifier='pblum_mode', value='dataset-scaled').to_list():
                    if not self.get_value(qualifier='enabled', compute=compute, dataset=param.dataset):
                        continue

                    logger.info("rescaling fluxes to data for dataset='{}'".format(param.dataset))
                    ds_obs = self.get_dataset(param.dataset, check_visible=False)
                    ds_times = ds_obs.get_value(qualifier='times')
                    ds_fluxes = ds_obs.get_value(qualifier='fluxes')
                    ds_sigmas = ds_obs.get_value(qualifier='sigmas')

                    ds_model = self.get_model(model, dataset=param.dataset, check_visible=False)
                    model_fluxes = ds_model.get_value(qualifier='fluxes')
                    model_fluxes_interp = ds_model.get_parameter(qualifier='fluxes').interp_value(times=ds_times)
                    scale_factor_approx = np.median(ds_fluxes / model_fluxes_interp)

                    # TODO: can we skip this if sigmas don't exist?
                    logger.debug("calling curve_fit with estimated scale_factor={}".format(scale_factor_approx))
                    popt, pcov = cfit(_scale_fluxes, model_fluxes_interp, ds_fluxes, p0=(scale_factor_approx), sigma=ds_sigmas if len(ds_sigmas) else None)
                    scale_factor = popt[0]

                    logger.debug("applying scale_factor={} to fluxes@{}".format(scale_factor, param.dataset))
                    ds_model.set_value(qualifier='fluxes', value=model_fluxes*scale_factor)

                    for param in ds_model.filter(kind='mesh').to_list():
                        if param.qualifier in ['intensities', 'abs_intensities', 'normal_intensities', 'abs_normal_intensities', 'pblum_ext']:
                            logger.debug("applying scale_factor={} to {} parameter in mesh".format(scale_factor, param.qualifier))
                            param.set_value(param.get_value() * scale_factor)

            # Figure options for this model
            if do_create_fig_params:
                fig_params = _figure._run_compute(self, **kwargs)

                fig_metawargs = {'context': 'figure',
                                 'model': model}
                self._attach_params(fig_params, **fig_metawargs)

            redo_kwargs = deepcopy(kwargs)
            redo_kwargs['compute'] = computes if len(computes)>1 else computes[0]
            redo_kwargs['model'] = model

            self._add_history(redo_func='run_compute',
                              redo_kwargs=redo_kwargs,
                              undo_func='remove_model',
                              undo_kwargs={'model': model})

        except Exception as err:
            restore_conf()
            raise

        restore_conf()

        ret_ps = self.filter(model=model, **_skip_filter_checks)

        # TODO: should we add these to the output?
        self._handle_model_selectparams()
        self._handle_meshcolor_choiceparams()
        self._handle_figure_time_source_params()

        if kwargs.get('overwrite', model=='latest') and kwargs.get('return_overwrite', False) and overwrite_ps is not None:
            ret_ps += overwrite_ps

        return ret_ps

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

    def import_model(self, fname, model=None):
        """
        Import and attach a model from a file.

        NEW IN PHOEBE 2.2

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

        Returns
        -----------
        * ParameterSet of added and changed parameters
        """
        result_ps = ParameterSet.open(fname)
        metawargs = {}
        if model is not None:
            metawargs['model'] = model
        self._attach_params(result_ps, override_tags=True, **metawargs)

        changed_params = self._handle_model_selectparams(return_changes=True)

        return ParameterSet(changed_params) + self.get_model(model=model if model is not None else result_ps.models)

    def remove_model(self, model, **kwargs):
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
        ret_ps += ParameterSet(self._handle_model_selectparams(return_changes=True))
        if remove_figure_params:
            ret_ps += ParameterSet(self._handle_meshcolor_choiceparams(return_changes=True))
        return ret_ps

    def remove_models_all(self):
        """
        Remove all models from the bundle.  To remove a single model see
        <phoebe.frontend.bundle.Bundle.remove_model>.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        removed_ps = ParameterSet()
        for model in self.models:
            removed_ps += self.remove_model(model=model)
        return removed_ps

    def rename_model(self, old_model, new_model):
        """
        Change the label of a model attached to the Bundle.

        Arguments
        ----------
        * `old_model` (string): current label of the model (must exist)
        * `new_model` (string): the desired new label of the model
            (must not yet exist)

        Returns
        --------
        * <phoebe.parameters.ParameterSet> the renamed model

        Raises
        --------
        * ValueError: if the value of `new_model` is forbidden or already exists.
        """
        # TODO: raise error if old_feature not found?

        self._check_label(new_model)
        self._rename_label('model', old_model, new_model)

        ret_ps = self.filter(model=new_model)

        ret_ps += ParameterSet(self._handle_model_selectparams())

        return ret_ps

    def attach_job(self, twig=None, wait=True, sleep=5, cleanup=True, **kwargs):
        """
        Attach the results from an existing <phoebe.parameters.JobParameter>.

        Jobs are created when passing `detach=True` to
        <phoebe.frontend.bundle.Bundle.run_compute>.

        See also:
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
        * `**kwargs`: any additional keyword arguments are sent to filter for the
            Job parameters.  Between `twig` and `**kwargs`, a single parameter
            with qualifier of 'detached_job' must be found.

        Returns
        -----------
        * (<phoebe.parameters.ParameterSet>): ParameterSet of the newly attached
            Parameters.
        """
        kwargs['qualifier'] = 'detached_job'
        return self.get_parameter(twig=twig, **kwargs).attach(wait=wait, sleep=sleep, cleanup=cleanup)
