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
from datetime import datetime
from distutils.version import StrictVersion

from scipy.optimize import curve_fit as cfit


# PHOEBE
# ParameterSet, Parameter, FloatParameter, send_if_client, etc
from phoebe.parameters import *
from phoebe.parameters import hierarchy as _hierarchy
from phoebe.parameters import system as _system
from phoebe.parameters import setting as _setting
from phoebe.parameters import dataset as _dataset
from phoebe.parameters import compute as _compute
from phoebe.parameters import constraint as _constraint
from phoebe.parameters import feature as _feature
from phoebe.backend import backends, mesh
from phoebe.distortions import roche
from phoebe.frontend import io
from phoebe.atmospheres.passbands import list_installed_passbands, list_online_passbands, get_passband, _timestamp_to_dt
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

_bundle_cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_bundles'))+'/'

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
    if isinstance(func, str) and hasattr(mod, func):
        func = getattr(mod, func)

    if hasattr(func, '__call__'):
        return func
    elif return_none_if_not_found:
        return None
    else:
        raise ValueError("could not find callable function in {}.{}"
                         .format(mod, func))


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

    To filter parameters and set values, see:
    * <phoebe.parameters.ParameterSet.filter>
    * <phoebe.parameters.ParameterSet.get_value>
    * <phoebe.parameters.ParameterSet.set_value>

    To deal with datasets, see:
    * <phoebe.frontend.bundle.Bundle.add_dataset>
    * <phoebe.frontend.bundle.Bundle.get_dataset>
    * <phoebe.frontend.bundle.Bundle.remove_dataset>
    * <phoebe.frontend.bundle.Bundle.enable_dataset>
    * <phoebe.frontend.bundle.Bundle.disable_dataset>

    To compute forward models, see:
    * <phoebe.frontend.bundle.Bundle.add_compute>
    * <phoebe.frontend.bundle.Bundle.get_compute>
    * <phoebe.frontend.bundle.Bundle.run_compute>
    * <phoebe.frontend.bundle.Bundle.get_model>

    To plot observations or synthetic datasets, see:
    * <phoebe.parameters.ParameterSet.plot>

    """

    def __init__(self, params=None):
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

        self._figure = None

        # set to be not a client by default
        self._is_client = False
        self._last_client_update = None

        # handle delayed constraints when interactive mode is off
        self._delayed_constraints = []

        if not len(params):
            # add position (only 1 allowed and required)
            self._attach_params(_system.system(), context='system')

            # add default settings (only 1 allowed and required)
            self._attach_params(_setting.settings(), context='setting')

            # set a blank hierarchy to start
            self.set_hierarchy(_hierarchy.blank)

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
        for constraint in self.filter(context='constraint').to_list():
            constraint._update_bookkeeping()

        # TODO: is this the correct place to do this? is blank hierarchy still
        # ok for loading from file??

    @classmethod
    def open(cls, filename):
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

        See also:
        * <phoebe.parameters.ParameterSet.open>
        * <phoebe.parameters.Parameter.open>

        Arguments
        ----------
        * `filename` (string): relative or full path to the file

        Returns
        ---------
        * an instantiated <phoebe.frontend.bundle.Bundle> object
        """
        filename = os.path.expanduser(filename)
        logger.debug("importing from {}".format(filename))
        f = open(filename, 'r')
        data = json.load(f, object_pairs_hook=parse_json)
        f.close()
        b = cls(data)

        version = b.get_value('phoebe_version')
        phoebe_version_import = StrictVersion(version if version != 'devel' else '2.1.2')
        phoebe_version_this = StrictVersion(__version__ if __version__ != 'devel' else '2.1.2')

        logger.debug("importing from PHOEBE v {} into v {}".format(phoebe_version_import, phoebe_version_this))

        # update the entry in the PS, so if this is saved again it will have the new version
        b.set_value('phoebe_version', __version__)

        if phoebe_version_import == phoebe_version_this:
            return b
        elif phoebe_version_import > phoebe_version_this:
            warning = "importing from a newer version ({}) of PHOEBE, this may or may not work, consider updating".format(phoebe_version_import)
            print("WARNING: {}".format(warning))
            logger.warning(warning)
            return b

        if phoebe_version_import < StrictVersion("2.1.2"):
            b._import_before_v211 = True
            warning = "Importing from an older version ({}) of PHOEBE which did not support constraints in solar units.  All constraints will remain in SI, but calling set_hierarchy will likely fail.".format(phoebe_version_import)
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
                if len(b.filter('syncpar', component=star)):
                    F = b.get_value('syncpar', component=star, context='component')
                else:
                    F = 1.0
                parent_orbit = b.hierarchy.get_parent_of(star)
                component = b.hierarchy.get_primary_or_secondary(star, return_ind=True)
                sma = b.get_value('sma', component=parent_orbit, context='component', unit=u.solRad)
                q = b.get_value('q', component=parent_orbit, context='component')
                d = 1 - b.get_value('ecc', component=parent_orbit)

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
                b.set_value('pot', component=env, context='component', value=dict_env['pot'])
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
            b.set_value('vgamma', -1*b.get_value('vgamma'))

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
        """Load a new bundle from a server.

        [NOT SUPPORTED]

        Load a bundle from a phoebe server.  This is a constructor so should be
        called as:

        ```py
        b = Bundle.from_server('asdf', as_client=False)
        ```

        Arguments
        ----------
        * `bundleid` (string): the identifier given to the bundle by the
            server
        * `server` (string): the host (and port) of the server
        * `as_client` (bool, optional, default=True):  whether to attach in
            client mode
        """
        if not conf.devel:
            raise NotImplementedError("'from_server' not officially supported for this release.  Enable developer mode to test.")

        # TODO: run test message on server, if localhost and fails, attempt to
        # launch?
        url = "{}/{}/json".format(server, bundleid)
        logger.info("downloading bundle from {}".format(url))
        r = requests.get(url, timeout=5)
        rjson = r.json()

        b = cls(rjson['data'])

        if as_client:
            b.as_client(as_client, server=server,
                        bundleid=rjson['meta']['bundleid'])

            logger.warning("This bundle is in client mode, meaning all\
            computations will be handled by the server at {}.  To disable\
            client mode, call as_client(False) or in the future pass\
            as_client=False to from_server".format(server))

        return b

    @classmethod
    def from_catalog(cls, identifier):
        """Load a new bundle from the phoebe catalog.

        [NOT SUPPORTED]

        Load a bundle from the online catalog.  This is a constructor
        so should be called as:

        ```py
        b = Bundle.from_catalog(identifier)
        ```

        Arguments
        ----------
        * `identifier` (string): identifier of the object in the catalog

        Returns
        ----------
        * instantiated <phoebe.frontend.bundle.Bundle> object.
        """
        raise NotImplementedError
        # TODO: pull from online catalog and pass arguments needed to cls
        # (__init__) or cls.open (defined in PS.open)

        return cls()

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
        * `filename` (string): relative or full path to the file
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
        filename = os.path.expanduser(filename)
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

            return b

        b = cls()
        # IMPORTANT NOTE: if changing any of the defaults for a new release,
        # make sure to update the cached files (see frontend/default_bundles
        # directory for script to update all cached bundles)
        b.add_star(component=starA)
        b.set_hierarchy(_hierarchy.component(b[starA]))
        b.add_compute(distortion_method='rotstar', irrad_method='none')
        return b

    @classmethod
    def default_binary(cls, starA='primary', starB='secondary', orbit='binary',
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
        * `contact_binary` (bool, optional, default=False): whether to also
            add an envelope (with component='contact_envelope') and set the
            hierarchy to a contact binary system.
        * `force_build` (bool, optional, default=False): whether to force building
            the bundle from scratch.  If False, pre-cached files will be loaded
            whenever possible to save time.

        Returns
        -----------
        * an instantiated <phoebe.frontend.bundle.Bundle> object.
        """
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
        b.add_star(component=starA, **star_defaults)
        b.add_star(component=starB, **star_defaults)
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

        b.add_compute()

        return b


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
        b.add_star(component=starA)
        b.add_star(component=starB)
        b.add_star(component=starC)
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
            passed, msg = self.run_checks(compute=compute)
            if passed is None:
                # then just raise a warning
                logger.warning(msg)
            if passed is False:
                # then raise an error
                raise ValueError("system failed to pass checks: {}".format(msg))

        filename = os.path.expanduser(filename)
        return io.pass_to_legacy(self, filename, compute=compute)


    def _test_server(self, server='http://localhost:5555', start_if_fail=True):
        """
        [NOT IMPLEMENTED]
        """
        try:
            resp = urllib2.urlopen("{}/test".format(server))
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

        for item in resp['data']:
            if item['id'] in self.uniqueids:
                # then we're updating something in the parameter (or deleting)
                param = self.get_parameter(uniqueid=item['id'])
                for attr, value in item['attributes'].items():
                    if hasattr(param, "_{}".format(attr)):
                        logger.info("updates from server: setting {}@{}={}".
                                    format(attr, param.twig, value))
                        setattr(param, "_{}".format(attr), value)
            else:
                self._attach_param_from_server(item)

    def _attach_param_from_server(self, item):
        """
        [NOT IMPLEMENTED]
        """
        if isinstance(item, list):
            for itemi in item:
                self._attach_param_from_server(itemi)
        else:
            # then we need to add a new parameter
            d = item['attributes']
            d['uniqueid'] = item['id']
            param = parameters.parameter_from_json(d, bundle=self)

            metawargs = {}
            self._attach_params([param], **metawargs)

    def as_client(self, as_client=True, server='http://localhost:5555',
                  bundleid=None):
        """
        [NOT IMPLEMENTED]
        """
        if as_client:
            if not _can_client:
                raise ImportError("dependencies to support client mode not met - see docs")

            server_running = self._test_server(server=server,
                                               start_if_fail=True)
            if not server_running:
                raise ValueError("server {} is not running".format(server))

            server_split = server.split(':')
            host = ':'.join(server_split[:-1])
            port = int(float(server_split[-1] if len(server_split) else 8000))
            self._socketio = SocketIO(host, port, BaseNamespace)
            self._socketio.on('connect', self._on_socket_connect)
            self._socketio.on('disconnect', self._on_socket_disconnect)

            self._socketio.on('push updates', self._on_socket_push_updates)

            if not bundleid:
                upload_url = "{}/upload".format(server)
                logger.info("uploading bundle to server {}".format(upload_url))
                data = json.dumps(self.to_json(incl_uniqueid=True))
                r = requests.post(upload_url, data=data, timeout=5)
                bundleid = r.json()['meta']['bundleid']

            self._socketio.emit('subscribe bundle', {'bundleid': bundleid})

            self._bundleid = bundleid

            self._is_client = server
            logger.info("connected as client to server at {}:{}".
                        format(host, port))

        else:
            logger.warning("This bundle is now permanently detached from the instance\
                on the server and will not receive future updates.  To start a client\
                in sync with the version on the server or other clients currently \
                subscribed, you must instantiate a new bundle with Bundle.from_server.")

            if hasattr(self, '_socketIO') and self._socketIO is not None:
                self._socketio.emit('unsubscribe bundle', {'bundleid': bundleid})
                self._socketIO.disconnect()
                self._socketIO = None

            self._bundleid = None
            self._is_client = False

    @property
    def is_client(self):
        return self._is_client

    def client_update(self):
        """
        [NOT IMPLEMENTED]
        """
        if not self.is_client:
            raise ValueError("Bundle is not in client mode, cannot update")

        logger.info("updating client...")
        # wait briefly to pickup any missed messages, which should then fire
        # the corresponding callbacks and update the bundle
        self._socketio.wait(seconds=1)
        self._last_client_update = datetime.now()

    def __repr__(self):
        return super(Bundle, self).__repr__().replace('ParameterSet', 'PHOEBE Bundle')

    def __str__(self):
        return_ = ''
        for context in ['system', 'component', 'dataset', 'constraint',
                        'compute', 'model', 'fitting', 'feedback', 'plugin']:
            return_ += '{}:\n'.format(context.upper())
            return_ += "\n".join(self.filter(context=context).to_dict().keys())
            return_ += '\n\n'

        return return_

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

        for param in self.filter(check_visible=False, check_default=False, **{tag: old_value}).to_list():
            setattr(param, '_{}'.format(tag), new_value)
        for param in self.filter(context='constraint', check_visible=False, check_default=False).to_list():
            for k, v in param.constraint_kwargs.items():
                if v == old_value:
                    param._constraint_kwargs[k] = new_value
        for param in self.filter(qualifier='include_times', check_visible=False, check_default=False).to_list():
            old_param_value = param._value
            new_param_value = [v.replace('@{}'.format(old_value), '@{}'.format(new_value)) for v in old_param_value]
            param._value = new_param_value

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
        Add a new log (undo/redoable) to this history contextself.

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

        Raises
        -------
        * ValueError: if no history items have been recorded.
        """
        if i is None:
            self.remove_parameters_all(context='history')
        else:
            param = self.get_history(i=i)
            self.remove_parameter(uniqueid=param.uniqueid)

        # let's not add_history for this one...

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

    def _handle_pblum_defaults(self):
        """
        """
        logger.debug("calling _handle_pblum_defaults")

        changed_params = self.run_delayed_constraints()

        hier = self.get_hierarchy()
        # Handle choice parameters that need components as choices
        # meshablerefs = hier.get_meshables()  # TODO: consider for overcontacts
        starrefs = hier.get_stars()  # TODO: consider for overcontacts
        datasetrefs = self.filter(qualifier='pblum_mode', check_visible=False).datasets

        for param in self.filter(qualifier='pblum_ref',
                                 context='dataset',
                                 check_visible=False,
                                 check_default=False).to_list():

            if param.component is None:
                param._choices = [ds for ds in datasetrefs if ds!=param.dataset]

                if param.value == '' and len(param._choices):
                    param.set_value(param._choices[0])

                if not len(param._choices):
                    param._choices = ['']

            else:
                param._choices = ['self'] + [s for s in starrefs if s!=param.component]

                if param.value == '':
                    # then this was the default from the parameter itself, so we
                    # want to set it to be pblum of its the "primary" star
                    if param.component == starrefs[0]:
                        param.set_value('self')
                    else:
                        param.set_value(starrefs[0])

    def _handle_dataset_selectparams(self):
        """
        """
        logger.debug("calling _handle_dataset_selectparams")

        changed_param = self.run_delayed_constraints()

        pbdep_datasets = self.filter(context='dataset',
                                     kind=_dataset._pbdep_columns.keys()).datasets

        pbdep_columns = _dataset._mesh_columns[:] # force deepcopy
        for pbdep_dataset in pbdep_datasets:
            pbdep_kind = self.filter(context='dataset',
                                     dataset=pbdep_dataset,
                                     kind=_dataset._pbdep_columns.keys()).kind

            pbdep_columns += ["{}@{}".format(column, pbdep_dataset) for column in _dataset._pbdep_columns[pbdep_kind]]

        time_datasets = (self.filter(context='dataset')-
                         self.filter(context='dataset', kind='mesh')).datasets

        t0s = ["{}@{}".format(p.qualifier, p.component) for p in self.filter(qualifier='t0*', context=['component']).to_list()]
        t0s += ["t0@system"]

        for param in self.filter(qualifier='columns',
                                 context='dataset').to_list():

            param._choices = pbdep_columns
            param.remove_not_valid_selections()

        for param in self.filter(qualifier='include_times',
                                 context='dataset').to_list():

            # NOTE: existing value is updated in change_component
            param._choices = time_datasets + t0s
            param.remove_not_valid_selections()


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

        # user_interactive_constraints = conf.interactive_constraints
        # conf.interactive_constraints_off()

        for component in self.hierarchy.get_envelopes():
            # we need two of the three [comp_env] + self.hierarchy.get_siblings_of(comp_env) to have constraints
            logger.debug('re-creating requiv constraints')
            existing_requiv_constraints = self.filter(constraint_func='requiv_to_pot', component=[component]+self.hierarchy.get_siblings_of(component))
            if len(existing_requiv_constraints) == 2:
                # do we need to rebuild these?
                continue
            elif len(existing_requiv_constraints)==0:
                for component_requiv in self.hierarchy.get_siblings_of(component):
                    pot_parameter = self.get_parameter(qualifier='pot', component=self.hierarchy.get_envelope_of(component_requiv), context='component')
                    requiv_parameter = self.get_parameter(qualifier='requiv', component=component_requiv, context='component')
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
                               component=component)):
                constraint_param = self.get_constraint(constraint_func='fillout_factor',
                                                       component=component)
                self.remove_constraint(constraint_func='fillout_factor',
                                       component=component)
                self.add_constraint(constraint.fillout_factor, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.fillout_factor, component,
                                    constraint=self._default_label('fillout_factor', context='constraint'))

            logger.debug('re-creating pot_min (contact) constraint for {}'.format(component))
            if len(self.filter(context='constraint',
                               constraint_func='potential_contact_min',
                               component=component)):
                constraint_param = self.get_constraint(constraint_func='potential_contact_min',
                                                       component=component)
                self.remove_constraint(constraint_func='potential_contact_min',
                                       component=component)
                self.add_constraint(constraint.potential_contact_min, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.potential_contact_min, component,
                                    constraint=self._default_label('pot_min', context='constraint'))

            logger.debug('re-creating pot_max (contact) constraint for {}'.format(component))
            if len(self.filter(context='constraint',
                               constraint_func='potential_contact_max',
                               component=component)):
                constraint_param = self.get_constraint(constraint_func='potential_contact_max',
                                                       component=component)
                self.remove_constraint(constraint_func='potential_contact_max',
                                       component=component)
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
                                component=component)):
                    constraint_param = self.get_constraint(constraint_func='mass',
                                                           component=component)
                    self.remove_constraint(constraint_func='mass',
                                           component=component)
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
                                   component=component)):
                    constraint_param = self.get_constraint(constraint_func='comp_sma',
                                                           component=component)
                    self.remove_constraint(constraint_func='comp_sma',
                                           component=component)
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
                                   component=component)):
                    constraint_param = self.get_constraint(constraint_func='rotation_period',
                                                           component=component)
                    self.remove_constraint(constraint_func='rotation_period',
                                           component=component)
                    self.add_constraint(constraint.rotation_period, component,
                                        solve_for=constraint_param.constrained_parameter.uniquetwig,
                                        constraint=constraint_param.constraint)
                else:
                    self.add_constraint(constraint.rotation_period, component,
                                        constraint=self._default_label('rotation_period', context='constraint'))

                if self.hierarchy.is_contact_binary(component):
                    # then we're in a contact binary and need to create pot<->requiv constraints
                    # NOTE: pot_min and pot_max are handled above at the envelope level
                    logger.debug('re-creating requiv_detached_max (contact) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_detached_max',
                                       component=component)):
                        # then we're changing from detached to contact so should remove the detached constraint first
                        self.remove_constraint(constraint_func='requiv_detached_max', component=component)

                    logger.debug('re-creating requiv_contact_max (contact) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_contact_max',
                                       component=component)):
                        constraint_param = self.get_constraint(constraint_func='requiv_contact_max',
                                                               component=component)
                        self.remove_constraint(constraint_func='requiv_contact_max',
                                               component=component)
                        self.add_constraint(constraint.requiv_contact_max, component,
                                            solve_for=constraint_param.constrained_parameter.uniquetwig,
                                            constraint=constraint_param.constraint)
                    else:
                        self.add_constraint(constraint.requiv_contact_max, component,
                                            constraint=self._default_label('requiv_max', context='constraint'))

                    logger.debug('re-creating requiv_contact_min (contact) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_contact_min',
                                       component=component)):
                        constraint_param = self.get_constraint(constraint_func='requiv_contact_min',
                                                               component=component)
                        self.remove_constraint(constraint_func='requiv_contact_min',
                                               component=component)
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
                                       component=component)):
                        self.remove_constraint(constraint_func='requiv_to_pot', component=component)

                    logger.debug('re-creating requiv_max (detached) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_contact_max',
                                       component=component)):
                        # then we're changing from contact to detached so should remove the detached constraint first
                        self.remove_constraint(constraint_func='requiv_contact_max', component=component)

                    logger.debug('re-creating requiv_detached_max (detached) constraint for {}'.format(component))
                    if len(self.filter(context='constraint',
                                       constraint_func='requiv_detached_max',
                                       component=component)):
                        constraint_param = self.get_constraint(constraint_func='requiv_detached_max',
                                                               component=component)
                        self.remove_constraint(constraint_func='requiv_detached_max',
                                               component=component)
                        self.add_constraint(constraint.requiv_detached_max, component,
                                            solve_for=constraint_param.constrained_parameter.uniquetwig,
                                            constraint=constraint_param.constraint)
                    else:
                        self.add_constraint(constraint.requiv_detached_max, component,
                                            constraint=self._default_label('requiv_max', context='constraint'))

                    logger.debug('re-creating pitch constraint for {}'.format(component))
                    # TODO: will this cause problems if the constraint has been flipped?
                    # TODO: what if the user disabled/removed this constraint?
                    if len(self.filter(context='constraint',
                                       constraint_func='pitch',
                                       component=component)):
                        constraint_param = self.get_constraint(constraint_func='pitch',
                                                               component=component)
                        self.remove_constraint(constraint_func='pitch',
                                               component=component)
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
                                    component=component)):
                        constraint_param = self.get_constraint(constraint_func='yaw',
                                                               component=component)
                        self.remove_constraint(constraint_func='yaw',
                                               component=component)
                        self.add_constraint(constraint.yaw, component,
                                            solve_for=constraint_param.constrained_parameter.uniquetwig,
                                            constraint=constraint_param.constraint)
                    else:
                        self.add_constraint(constraint.yaw, component,
                                            constraint=self._default_label('yaw', context='constraint'))

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

    def _kwargs_checks(self, kwargs, additional_allowed_keys=[],
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
                        ['skip_checks', 'check_default', 'check_visible'] +\
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
                                        warning_only=warning_only
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

    def run_checks(self, **kwargs):
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
            all available compute options will be considered.
        * `**kwargs`: overrides for any parameter (given as qualifier=value pairs)

        Returns
        ----------
        * (bool, str) whether the checks passed or failed and a message describing
            the FIRST failure (if applicable).
        """

        # make sure all constraints have been run
        changed_params = self.run_delayed_constraints()

        computes = kwargs.pop('compute', self.computes)
        if computes is None:
            computes = self.computes
        if isinstance(computes, str):
            computes = [computes]

        hier = self.hierarchy
        if hier is None:
            return True, ''
        for component in hier.get_stars():
            kind = hier.get_kind_of(component)
            comp_ps = self.get_component(component)

            if not len(comp_ps):
                return False, "component '{}' in the hierarchy is not in the bundle".format(component)

            parent = hier.get_parent_of(component)
            parent_ps = self.get_component(parent)
            if kind in ['star']:
                # ignore the single star case
                if parent:
                    # contact systems MUST by synchronous
                    if hier.is_contact_binary(component):
                        if self.get_value(qualifier='syncpar', component=component, context='component', **kwargs) != 1.0:
                            return False,\
                                'contact binaries must by synchronous, but syncpar@{}!=1'.format(component)

                        if self.get_value(qualifier='ecc', component=parent, context='component', **kwargs) != 0.0:
                            return False,\
                                'contact binaries must by circular, but ecc@{}!=0'.format(component)

                        if self.get_value(qualifier='pitch', component=component, context='component', **kwargs) != 0.0:
                            return False,\
                                'contact binaries must be aligned, but pitch@{}!=0'.format(component)

                        if self.get_value(qualifier='yaw', component=component, context='component', **kwargs) != 0.0:
                            return False,\
                                'contact binaries must be aligned, but yaw@{}!=0'.format(component)

                    # MUST NOT be overflowing at PERIASTRON (d=1-ecc, etheta=0)

                    requiv = comp_ps.get_value('requiv', unit=u.solRad, **kwargs)
                    requiv_max = comp_ps.get_value('requiv_max', unit=u.solRad, **kwargs)



                    if hier.is_contact_binary(component):
                        if np.isnan(requiv) or requiv > requiv_max:
                            return False,\
                                '{} is overflowing at L2/L3 (requiv={}, requiv_max={})'.format(component, requiv, requiv_max)


                        requiv_min = comp_ps.get_value('requiv_min')

                        if np.isnan(requiv) or requiv <= requiv_min:
                            return False,\
                                 '{} is underflowing at L1 and not a contact system (requiv={}, requiv_min={})'.format(component, requiv, requiv_min)
                        elif requiv <= requiv_min * 1.001:
                            return False,\
                                'requiv@{} is too close to requiv_min (within 0.1% of critical).  Use detached/semidetached model instead.'.format(component)

                    else:
                        if requiv > requiv_max:
                            return False,\
                                '{} is overflowing at periastron (requiv={}, requiv_max={})'.format(component, requiv, requiv_max)

            else:
                raise NotImplementedError("checks not implemented for type '{}'".format(kind))

        # we also need to make sure that stars don't overlap each other
        # so we'll check for each pair of stars (see issue #70 on github)
        # TODO: rewrite overlap checks
        for orbitref in []: #hier.get_orbits():
            if len(hier.get_children_of(orbitref)) == 2:
                q = self.get_value(qualifier='q', component=orbitref, context='component', **kwargs)
                ecc = self.get_value(qualifier='ecc', component=orbitref, context='component', **kwargs)

                starrefs = hier.get_children_of(orbitref)
                if hier.get_kind_of(starrefs[0]) != 'star' or hier.get_kind_of(starrefs[1]) != 'star':
                    # print "***", hier.get_kind_of(starrefs[0]), hier.get_kind_of(starrefs[1])
                    continue
                if self.get_value(qualifier='pitch', component=starrefs[0])!=0.0 or \
                        self.get_value(qualifier='pitch', component=starrefs[1])!=0.0 or \
                        self.get_value(qualifier='yaw', component=starrefs[0])!=0.0 or \
                        self.get_value(qualifier='yaw', component=starrefs[1])!=0.0:

                    # we cannot run this test for misaligned cases
                   continue

                comp0 = hier.get_primary_or_secondary(starrefs[0], return_ind=True)
                comp1 = hier.get_primary_or_secondary(starrefs[1], return_ind=True)
                q0 = roche.q_for_component(q, comp0)
                q1 = roche.q_for_component(q, comp1)

                F0 = self.get_value(qualifier='syncpar', component=starrefs[0], context='component', **kwargs)
                F1 = self.get_value(qualifier='syncpar', component=starrefs[1], context='component', **kwargs)

                pot0 = self.get_value(qualifier='pot', component=starrefs[0], context='component', **kwargs)
                pot0 = roche.pot_for_component(pot0, q0, comp0)

                pot1 = self.get_value(qualifier='pot', component=starrefs[1], context='component', **kwargs)
                pot1 = roche.pot_for_component(pot1, q1, comp1)

                xrange0 = libphoebe.roche_xrange(q0, F0, 1.0-ecc, pot0+1e-6, choice=0)
                xrange1 = libphoebe.roche_xrange(q1, F1, 1.0-ecc, pot1+1e-6, choice=0)

                if xrange0[1]+xrange1[1] > 1.0-ecc:
                    return False,\
                        'components in {} are overlapping at periastron (change ecc@{}, syncpar@{}, or syncpar@{}).'.format(orbitref, orbitref, starrefs[0], starrefs[1])

        # run passband checks
        all_pbs = list_passbands(full_dict=True)
        installed_pbs = list_installed_passbands(full_dict=True)
        online_pbs = list_online_passbands(full_dict=True)
        for pbparam in self.filter(qualifier='passband').to_list():
            pb = pbparam.get_value()
            pbatms = installed_pbs[pb]['atms']
            # NOTE: atms are not attached to datasets, but per-compute and per-component
            # check to make sure passband supports the selected atm
            for atmparam in self.filter(qualifier='atm', kind='phoebe').to_list():
                atm = atmparam.get_value()
                if atm not in pbatms:
                    return False, "'{}' passband ({}) does not support atm='{}' ({}).".format(pb, pbparam.twig, atm, atmparam.twig)

            # check to see if passband timestamp is recent enough for reddening, etc.
            if False: # if reddening is non-zero: and also update timestamp to the release of extinction-ready passbands
                if installed_pbs[pb]['timestamp'] is None or _timestamp_to_dt(installed_pbs[pb]['timestamp']) < _timestamp_to_dt("Wed Jan 25 12:00:00 2019"):
                    return False,\
                        'installed passband "{}" does not support reddening/extinction.  Call phoebe.download_passband("{}") or phoebe.update_all_passbands() to update to the latest version.'.format(pb, pb)


        # check length of ld_coeffs vs ld_func and ld_func vs atm
        def ld_coeffs_len(ld_func, ld_coeffs):
            # current choices for ld_func are:
            # ['interp', 'uniform', 'linear', 'logarithmic', 'quadratic', 'square_root', 'power', 'claret', 'hillen', 'prsa']
            if ld_func == 'interp':
                return True,
            elif ld_func in ['linear'] and (ld_coeffs is None or len(ld_coeffs)==1):
                return True,
            elif ld_func in ['logarithmic', 'square_root', 'quadratic'] and (ld_coeffs is None or len(ld_coeffs)==2):
                return True,
            elif ld_func in ['power'] and (ld_coeffs is None or len(ld_coeffs)==4):
                return True,
            else:
                return False, "ld_coeffs={} wrong length for ld_func='{}'.".format(ld_coeffs, ld_func)

        for component in self.hierarchy.get_stars():
            # first check ld_coeffs_bol vs ld_func_bol
            ld_func = str(self.get_value(qualifier='ld_func_bol', component=component, context='component', check_visible=False, **kwargs))
            ld_coeffs = self.get_value(qualifier='ld_coeffs_bol', component=component, context='component', check_visible=False, **kwargs)
            check = ld_coeffs_len(ld_func, ld_coeffs)
            if not check[0]:
                return check

            if ld_func != 'interp':
                check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs))
                if not check:
                    return False, 'ld_coeffs_bol={} not compatible for ld_func_bol=\'{}\'.'.format(ld_coeffs, ld_func)

                for compute in computes:
                    if self.get_compute(compute).kind in ['legacy'] and ld_func not in ['linear', 'logarithmic', 'square_root']:
                        return False, "ld_func_bol='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', or 'square_root'.".format(ld_func, self.get_compute(compute).kind, compute)

            for dataset in self.datasets:
                if dataset=='_default' or self.get_dataset(dataset=dataset, kind='*dep').kind not in ['lc_dep', 'rv_dep']:
                    continue
                ld_func = str(self.get_value(qualifier='ld_func', dataset=dataset, component=component, context='dataset', **kwargs))
                ld_coeffs_source = self.get_value(qualifier='ld_coeffs_source', dataset=dataset, component=component, context='dataset', check_visible=False, **kwargs)
                ld_coeffs = self.get_value(qualifier='ld_coeffs', dataset=dataset, component=component, context='dataset', check_visible=False, **kwargs)
                pb = self.get_value(qualifier='passband', dataset=dataset, context='dataset', check_visible=False, **kwargs)

                if ld_func != 'interp':
                    if ld_coeffs_source not in ['none', 'auto']:
                        if ld_coeffs_source not in all_pbs[pb]['atms_ld']:
                            return False, 'passband={} does not support ld_coeffs_source={}'.format(pb, ld_coeffs_source)

                    elif ld_coeffs_source == 'none':
                        check = ld_coeffs_len(ld_func, ld_coeffs)
                        if not check[0]:
                            return check

                        check = libphoebe.ld_check(_bytes(ld_func), np.asarray(ld_coeffs))
                        if not check:
                            return False, 'ld_coeffs={} not compatible for ld_func=\'{}\'.'.format(ld_coeffs, ld_func)

                        for compute in computes:
                            if self.get_compute(compute).kind in ['legacy'] and ld_func not in ['linear', 'logarithmic', 'square_root']:
                                return False, "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', or 'square_root'.".format(ld_func, self.get_compute(compute).kind, compute)
                            if self.get_compute(compute).kind in ['jktebop'] and ld_func not in ['linear', 'logarithmic', 'square_root', 'quadratic']:
                                return False, "ld_func='{}' not supported by '{}' backend used by compute='{}'.  Use 'linear', 'logarithmic', 'quadratic', or 'square_root'.".format(ld_func, self.get_compute(compute).kind, compute)


                if ld_func=='interp':
                    for compute in computes:
                        # TODO: should we ignore if the dataset is disabled?
                        try:
                            atm = self.get_value(qualifier='atm', component=component, compute=compute, context='compute', **kwargs)
                        except ValueError:
                            # not all backends have atm as a parameter/option
                            continue
                        else:
                            if atm != 'ck2004' and atm != 'phoenix':
                                if 'ck2004' in self.get_parameter(qualifier='atm', component=component, compute=compute, context='compute', **kwargs).choices:
                                    return False, "ld_func='interp' not supported by atm='{}'.  Either change atm@{}@{} or ld_func@{}@{}.".format(atm, component, compute, component, dataset)
                                else:
                                    return False, "ld_func='interp' not supported by '{}' backend used by compute='{}'.  Change ld_func@{}@{} or use a backend that supports atm='ck2004'.".format(self.get_compute(compute).kind, compute, component, dataset)

        for compute in computes:
            compute_kind = self.get_compute(compute=compute).kind

            # mesh-consistency checks
            mesh_methods = [p.get_value() for p in self.filter(qualifier='mesh_method', compute=compute, force_ps=True).to_list()]
            if 'wd' in mesh_methods:
                if len(set(mesh_methods)) > 1:
                    return False, "all (or none) components must use mesh_method='wd'."

            # l3_mode checks
            if compute_kind in ['legacy']:
                enabled_datasets = self.filter(qualifier='enabled', value=True, compute=compute, force_ps=True).datasets
                l3_modes = [p.get_value() for p in self.filter('l3_mode', dataset=enabled_datasets, force_ps=True).to_list()]
                if len(set(l3_modes)) > 1:
                    return False, "{} backend (compute='{}') requires all values of 'l3_mode' (for enabled datasets) to be the same.".format(compute_kind, compute)



        # forbid color-coupling with a dataset which is scaled to data or to another that is in-turn color-coupled
        for param in self.filter(qualifier='pblum_mode', value='color coupled').to_list():
            coupled_to = self.get_value(qualifier='pblum_ref', dataset=param.dataset)
            if coupled_to == '':
                continue
            pblum_mode = self.get_value(qualifier='pblum_mode', dataset=coupled_to)
            if pblum_mode in ['scale to data', 'color coupled']:
                return False, "cannot set pblum_ref@{}='{}' as that dataset has pblum_mode@{}='{}'".format(param.dataset, coupled_to, coupled_to, pblum_mode)

        # require any pblum_mode == 'scale to data' to have accompanying data
        for param in self.filter(qualifier='pblum_mode', value='scale to data').to_list():
            if not len(self.get_value(qualifier='fluxes', dataset=param.dataset, context='dataset')):
                return False, "fluxes@{} cannot be empty if pblum_mode@{}='scale to data'".format(param.dataset, param.dataset)

        ### TODO: add tests for lengths of fluxes, rvs, etc vs times (and fluxes vs wavelengths for spectral datasets)

        #### WARNINGS ONLY ####
        # let's check teff vs gravb_bol and irrad_frac_refl_bol
        for component in self.hierarchy.get_stars():
            teff = self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **kwargs)
            gravb_bol = self.get_value(qualifier='gravb_bol', component=component, context='component', **kwargs)

            if teff >= 8000. and gravb_bol < 0.9:
                return None, "'{}' probably has a radiative atm (teff={:.0f}K>8000K), for which gravb_bol=1.00 might be a better approx than gravb_bol={:.2f}.".format(component, teff, gravb_bol)
            elif teff <= 6600. and gravb_bol >= 0.9:
                return None, "'{}' probably has a convective atm (teff={:.0f}K<6600K), for which gravb_bol=0.32 might be a better approx than gravb_bol={:.2f}.".format(component, teff, gravb_bol)
            elif gravb_bol < 0.32 or gravb_bol > 1.00:
                return None, "'{}' has intermittent temperature (6600K<teff={:.0f}K<8000K), gravb_bol might be better between 0.32-1.00 than gravb_bol={:.2f}.".format(component, teff, gravb_bol)

        for component in self.hierarchy.get_stars():
            teff = self.get_value(qualifier='teff', component=component, context='component', unit=u.K, **kwargs)
            irrad_frac_refl_bol = self.get_value(qualifier='irrad_frac_refl_bol', component=component, context='component', **kwargs)

            if teff >= 8000. and irrad_frac_refl_bol < 0.8:
                return None, "'{}' probably has a radiative atm (teff={:.0f}K>8000K), for which irrad_frac_refl_bol=1.00 might be a better approx than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol)
            elif teff <= 6600. and irrad_frac_refl_bol >= 0.75:
                return None, "'{}' probably has a convective atm (teff={:.0f}K<6600K), for which irrad_frac_refl_bol=0.6 might be a better approx than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol)
            elif irrad_frac_refl_bol < 0.6:
                return None, "'{}' has intermittent temperature (6600K<teff={:.0f}K<8000K), irrad_frac_refl_bol might be better between 0.6-1.00 than irrad_frac_refl_bol={:.2f}.".format(component, teff, irrad_frac_refl_bol)

        # TODO: add other checks
        # - make sure all ETV components are legal
        # - check for conflict between dynamics_method and mesh_method (?)

        # we've survived all tests
        return True, ''

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

        Available kinds can be found in <phoebe.parameters.feature> and include:
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
            self.remove_feature(feature=kwargs['feature'])
            # check the label again, just in case kwargs['feature'] belongs to
            # something other than feature
            self._check_label(kwargs['feature'], allow_overwrite=False)

        self._attach_params(params, **metawargs)

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
        return self.get_feature(**metawargs)

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
        kwargs.setdefault('context', ['feature'])

        self.remove_parameters_all(**kwargs)

        self._add_history(redo_func='remove_feature',
                          redo_kwargs=kwargs,
                          undo_func=None,
                          undo_kwargs={})

        return

    def remove_features_all(self):
        """
        Remove all features from the bundle.  To remove a single feature, see
        <phoebe.frontend.bundle.Bundle.remove_feature>.
        """
        for feature in self.features:
            self.remove_feature(feature=feature)

    def rename_feature(self, old_feature, new_feature):
        """
        Change the label of a feature attached to the Bundle.

        Arguments
        ----------
        * `old_feature` (string): current label of the feature (must exist)
        * `new_feature` (string): the desired new label of the feature
            (must not yet exist)

        Raises
        --------
        * ValueError: if the value of `new_feature` is forbidden or already exists.
        """
        # TODO: raise error if old_feature not found?

        self._check_label(new_feature)
        self._rename_label('feature', old_feature, new_feature)

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

    def remove_spot(self, feature=None, **kwargs):
        """
        Shortcut to <phoebe.frontend.bundle.Bundle.remove_feature> but with kind='spot'.
        """
        kwargs.setdefault('kind', 'spot')
        return self.remove_feature(feature, **kwargs)

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

        Available kinds can be found in <phoebe.parameters.component> and include:
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
        * `component` (string, optional): name of the newly-created feature.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing component with the same `component` tag.  If False,
            an error will be raised.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added


        Raises
        ----------
        * NotImplementedError: if a required constraint is not implemented.
        """

        func = _get_add_func(component, kind)

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
            self.remove_component(component=kwargs['component'])
            # check the label again, just in case kwargs['component'] belongs to
            # something other than component
            self._check_label(kwargs['component'], allow_overwrite=False)

        self._attach_params(params, **metawargs)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = fname
        self._add_history(redo_func='add_component',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_component',
                          undo_kwargs={'component': kwargs['component']})

        for constraint in constraints:
            self.add_constraint(*constraint)

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, warning_only=True)

        # return params
        return self.get_component(**metawargs)

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
        """
        # NOTE: run_checks will check if an entry is in the hierarchy but has no parameters
        kwargs['component'] = component
        # NOTE: we do not remove from 'model' by default
        kwargs['context'] = ['component', 'constraint', 'dataset', 'compute']
        self.remove_parameters_all(**kwargs)

    def rename_component(self, old_component, new_component):
        """
        Change the label of a component attached to the Bundle.

        Arguments
        ----------
        * `old_component` (string): current label of the component (must exist)
        * `new_component` (string): the desired new label of the component
            (must not yet exist)

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

        self._rename_label('component', old_component, new_component)

        self._handle_dataset_selectparams()

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

    def remove_orbit(self, component=None, **kwargs):
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
            to be used for t0
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
                ret['t0'] = ps.get_value(qualifier=t0, unit=u.d)
            elif isinstance(t0, float) or isinstance(t0, int):
                ret['t0'] = t0
            else:
                raise ValueError("t0 must be string (qualifier) or float")
            ret['dpdt'] = ps.get_value(qualifier='dpdt', unit=u.d/u.d)
        elif ps.kind in ['star']:
            # TODO: consider renaming period to prot
            ret['period'] = ps.get_value(qualifier='period', unit=u.d)
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
        if dpdt != 0:
            time = t0 + 1./dpdt*(np.exp(dpdt*(phase))-period)
        else:
            time = t0 + (phase)*period

        return time

    def add_dataset(self, kind, component=None, **kwargs):
        """
        Add a new dataset to the bundle.  If not provided,
        `dataset` (the name of the new dataset) will be created for
        you and can be accessed by the `dataset` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        For light curves, the light curve will be generated for the entire system.

        For radial velocities, you need to provide a list of components
        for which values should be computed.

        Available kinds can be found in <phoebe.parameters.dataset> and include:
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
            based on the value of `kind`.
        * `dataset` (string, optional): name of the newly-created feature.
        * `overwrite` (boolean, optional, default=False): whether to overwrite
            an existing dataset with the same `dataset` tag.  If False,
            an error will be raised.
        * `**kwargs`: default values for any of the newly-created parameters
            (passed directly to the matched callabled function).

        Returns
        ---------
        * <phoebe.parameters.ParameterSet> of all parameters that have been added


        Raises
        ----------
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

        obs_metawargs = {'context': 'dataset',
                         'kind': kind,
                         'dataset': kwargs['dataset']}

        if kind in ['lp']:
            # then times needs to be passed now to duplicate and tag the Parameters
            # correctly
            obs_kwargs = {'times': kwargs.pop('times', [])}
        else:
            obs_kwargs = {}

        obs_params, constraints = func(dataset=kwargs['dataset'], component_top=self.hierarchy.get_top(), **obs_kwargs)

        if kwargs.get('overwrite', False):
            self.remove_dataset(dataset=kwargs['dataset'])
            # check the label again, just in case kwargs['dataset'] belongs to
            # something other than dataset
            self._check_label(kwargs['dataset'], allow_overwrite=False)

        self._attach_params(obs_params, **obs_metawargs)

        for constraint in constraints:
            # TODO: tricky thing here will be copying the constraints
            self.add_constraint(*constraint)

        dep_func = _get_add_func(_dataset, "{}_dep".format(kind))
        dep_metawargs = {'context': 'dataset',
                         'kind': '{}_dep'.format(kind),
                         'dataset': kwargs['dataset']}
        dep_params = dep_func()
        self._attach_params(dep_params, **dep_metawargs)

        # Now we need to apply any kwargs sent by the user.  There are a few
        # scenarios (and each kwargs could fall into different ones):
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
            else:
                # then we must flip the constraint
                # TODO: this will probably break with triple support - we'll need to handle the multiple orbit components by accepting the dictionary.
                # For now we'll assume the component is top-level binary
                self.flip_constraint('compute_phases', component=self.hierarchy.get_top(), dataset=kwargs['dataset'], solve_for='compute_times')

        for k, v in kwargs.items():
            if isinstance(v, dict):
                for component, value in v.items():
                    logger.debug("setting value of dataset parameter: qualifier={}, dataset={}, component={}, value={}".format(k, kwargs['dataset'], component, value))
                    try:
                        self.set_value_all(qualifier=k,
                                           dataset=kwargs['dataset'],
                                           component=component,
                                           value=value,
                                           check_visible=False,
                                           ignore_none=True)
                    except Exception as err:
                        self.remove_dataset(dataset=kwargs['dataset'])
                        raise ValueError("could not set value for {}={} with error: '{}'. Dataset has not been added".format(k, value, err.message))

            elif k in ['dataset']:
                pass
            else:
                # for dataset kinds that include passband dependent AND
                # independent parameters, we need to carefully default on
                # what component to use when passing the defaults
                if kind in ['rv', 'lp'] and k in ['ld_func', 'ld_coeffs',
                                                  'passband', 'intens_weighting',
                                                  'profile_rest', 'profile_func', 'profile_sv']:
                    # passband-dependent (ie lc_dep) parameters do not have
                    # assigned components
                    components_ = None
                elif k in ['compute_times', 'compute_phases']:
                    components_ = self.hierarchy.get_top()
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
                                       check_visible=False,
                                       ignore_none=True)
                except Exception as err:
                    self.remove_dataset(dataset=kwargs['dataset'])
                    raise ValueError("could not set value for {}={} with error: '{}'. Dataset has not been added.".format(k, v, err.message))


        def _to_safe_value(v):
            if isinstance(v, nparray.ndarray):
                return v.to_json()
            elif isinstance(v, dict):
                return {k: _to_safe_value(v) for k,v in v.items()}
            else:
                return v

        redo_kwargs = deepcopy({k:_to_safe_value(v) for k,v in kwargs.items()})
        redo_kwargs['func'] = func.__name__
        self._add_history(redo_func='add_dataset',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_dataset',
                          undo_kwargs={'dataset': kwargs['dataset']})

        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite'], warning_only=True)

        return self.filter(dataset=kwargs['dataset'])

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

        Raises
        --------
        * ValueError: if `dataset` is not provided AND no `kwargs` are provided.
        """

        self._kwargs_checks(kwargs)

        # Let's avoid deleting ALL parameters from the matching contexts
        if dataset is None and not len(kwargs.items()):
            raise ValueError("must provide some value to filter for datasets")

        # let's handle deps if kind was passed
        kind = kwargs.get('kind', None)

        if kind is not None:
            if isinstance(kind, str):
                kind = [kind]
            kind_deps = []
            for kind_i in kind:
                dep = '{}_dep'.format(kind_i)
                if dep not in kind:
                    kind_deps.append(dep)
            kind = kind + kind_deps
        kwargs['kind'] = kind


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
        kwargs.setdefault('context', ['dataset', 'model', 'constraint', 'compute'])

        # ps = self.filter(**kwargs)
        # logger.info('removing {} parameters (this is not undoable)'.\
        #             format(len(ps)))

        # print "*** kwargs", kwargs, len(ps)
        self.remove_parameters_all(**kwargs)
        # not really sure why we need to call this twice, but it seems to do
        # the trick
        self.remove_parameters_all(**kwargs)

        self._handle_dataset_selectparams()

        # TODO: check to make sure that trying to undo this
        # will raise an error saying this is not undo-able
        self._add_history(redo_func='remove_dataset',
                          redo_kwargs={'dataset': dataset},
                          undo_func=None,
                          undo_kwargs={})

        return

    def remove_datasets_all(self):
        """
        Remove all datasets from the bundle.  To remove a single dataset see
        <phoebe.frontend.bundle.Bundle.remove_dataset>.
        """
        for dataset in self.datasets:
            self.remove_dataset(dataset=dataset)

    def rename_dataset(self, old_dataset, new_dataset):
        """
        Change the label of a dataset attached to the Bundle.

        Arguments
        ----------
        * `old_dataset` (string): current label of the dataset (must exist)
        * `new_dataset` (string): the desired new label of the dataset
            (must not yet exist)

        Raises
        --------
        * ValueError: if the value of `new_dataset` is forbidden or already exists.
        """
        # TODO: raise error if old_component not found?

        self._check_label(new_dataset)
        self._rename_label('dataset', old_dataset, new_dataset)
        self._handle_dataset_selectparams()


    def enable_dataset(self, dataset=None, **kwargs):
        """
        Enable a `dataset`.  Datasets that are enabled will be computed
        during <phoebe.frontend.bundle.Bundle.run_compute> and included in the cost function
        during run_fitting (once supported).

        If `compute` is not provided, the dataset will be enabled across all
        compute options.

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
        * <phoebe.parameters.constraint.logg>

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
            kwargs['solve_for'] = self.get_parameter(kwargs['solve_for'], context=['component', 'dataset', 'model'])

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
        if len(self._bundle.filter(**check_kwargs)):
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
        * `twig` (string, optional): twig to filter for the constraint.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: context, twig.
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
        constraint = self.get_parameter(**kwargs)

        # undo parameter bookkeeping
        constraint._remove_bookkeeping()

        # and finally remove it
        self.remove_parameter(**kwargs)


        undo_kwargs = {k: v for k, v in constraint.to_dict().items()
                       if v is not None and
                       k not in ['uniqueid', 'uniquetwig', 'twig',
                                 'Class', 'context']}
        self._add_history(redo_func='remove_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='add_constraint',
                          undo_kwargs=undo_kwargs)


    def flip_constraint(self, twig=None, solve_for=None, **kwargs):
        """
        Flip an existing constraint to solve for a different parameter.

        See also:
        * <phoebe.frontend.bundle.Bundle.add_constraint>
        * <phoebe.frontend.bundle.Bundle.get_constraint>
        * <phoebe.frontend.bundle.Bundle.remove_constraint>
        * <phoebe.frontend.bundle.Bundle.run_constraint>
        * <phoebe.frontend.bundle.Bundle.run_delayed_constraints>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to filter the constraint
        * `solve_for` (string or Parameter, optional, default=None): twig or
            <phoebe.parameters.Parameter> object of the new parameter for which
            this constraint should constrain (solve for).

        Returns
        ---------
        * The <phoebe.parameters.ConstraintParameter>.
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

        if kwargs.pop('check_nan', True) and np.any([_check_nan(p.get_value()) for p in param.vars.to_list()]):
            raise ValueError("cannot flip constraint while the value of {} is nan".format([p.twig for p in param.vars.to_list() if np.isnan(p.get_value())]))

        if solve_for is None:
            return param
        if isinstance(solve_for, Parameter):
            solve_for = solve_for.uniquetwig

        redo_kwargs['solve_for'] = solve_for
        undo_kwargs['solve_for'] = param.constrained_parameter.uniquetwig

        logger.info("flipping constraint '{}' to solve for '{}'".format(param.uniquetwig, solve_for))
        param.flip_for(solve_for)

        result = self.run_constraint(uniqueid=param.uniqueid, skip_kwargs_checks=True)

        self._add_history(redo_func='flip_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='flip_constraint',
                          undo_kwargs=undo_kwargs)

        return param

    def run_constraint(self, twig=None, return_parameter=False, **kwargs):
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

        result = expression_param.result

        if result != constrained_param.get_value():
            logger.debug("setting '{}'={} from '{}' constraint".format(constrained_param.uniquetwig, result, expression_param.uniquetwig))
            constrained_param.set_value(result, force=True, run_constraints=True)

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
        for constraint_id in self._delayed_constraints:
            param = self.run_constraint(uniqueid=constraint_id, return_parameter=True, skip_kwargs_checks=True)
            if param not in changes:
                changes.append(param)
        self._delayed_constraints = []
        return changes

    def compute_ld_coeffs(self, compute=None, set_value=False, **kwargs):
        """
        Compute the interpolated limb darkening coefficients.

        This method is only for convenience and will be recomputed internally
        within <phoebe.frontend.bundle.Bundle.run_compute> for all backends
        that require per-star limb-darkening coefficients.  Note that the default
        <phoebe.parameters.compute.phoebe> backend will instead interpolate
        limb-darkening coefficients **per-element**.

        Coefficients will only be interpolated/returned for those where `ld_func`
        is not 'interp' and `ld_coeffs_source` is not 'none'.  The values of
        the `ld_coeffs` parameter will be returned for cases where `ld_func` is
        not `interp` but `ld_coeffs_source` is 'none'.  Cases where `ld_func` is
        'interp' will not be included in the output.

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
            datasets attached to the bundle.
        * `set_value` (bool, optional, default=False): apply the interpolated
            values to the respective `ld_coeffs` parameters (even if not
            currently visible).
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

        datasets = kwargs.pop('dataset', self.datasets)
        components = kwargs.pop('component', self.components)

        # don't allow things like model='mymodel', etc
        forbidden_keys = parameters._meta_fields_filter
        self._kwargs_checks(kwargs, additional_allowed_keys=['skip_checks'], additional_forbidden_keys=forbidden_keys)

        if compute is None:
            if len(self.computes)==1:
                compute = self.computes[0]
            else:
                raise ValueError("must provide compute")
        if not isinstance(compute, str):
            raise TypeError("compute must be a single value (string)")

        if not kwargs.get('skip_checks', False):
            passed, msg = self.run_checks(compute=compute, **kwargs)
            if passed is None:
                # then just raise a warning
                logger.warning(msg)
            if passed is False:
                # then raise an error
                raise ValueError("system failed to pass checks: {}".format(msg))

        ld_coeffs_ret = {}
        for ldcs_param in self.filter(qualifier='ld_coeffs_source', dataset=datasets, component=components).to_list():
            ldcs = ldcs_param.get_value()
            if ldcs == 'none':
                ld_coeffs_ret["ld_coeffs@{}@{}".format(ldcs_param.component, ldcs_param.dataset)] = self.get_value(qualifier='ld_coeffs', dataset=ldcs_param.dataset, component=ldcs_param.component, check_visible=False)

                continue

            if ldcs=='auto':
                try:
                    atm = self.get_value(qualifier='atm', compute=compute, component=ldcs_param.component)
                except ValueError:
                    # not all backends have atm as an option
                    logger.warning("backend compute='{}' has no 'atm' option: falling back on ck2004 for ld_coeffs interpolation".format(compute))
                    atm = 'ck2004'

                if atm in ['extern_atmx', 'extern_planckint']:
                    ldcs = 'ck2004'
                else:
                    ldcs = atm

            passband = self.get_value(qualifier='passband', dataset=ldcs_param.dataset)
            ld_func = self.get_value(qualifier='ld_func', dataset=ldcs_param.dataset, component=ldcs_param.component)

            if ld_func == 'interp':
                # really shouldn't happen as the ld_coeffs_source parameter should not be visible
                # and so shouldn't be included in the loop
                raise ValueError("cannot interpolating ld_coeffs for ld_func='interp'")

            logger.info("interpolating {} ld_coeffs for dataset='{}' component='{}' passband='{}' from ld_coeffs_source='{}'".format(ld_func, ldcs_param.dataset, ldcs_param.component, passband, ldcs))
            pb = get_passband(passband)
            teff = self.get_value(qualifier='teff', component=ldcs_param.component, context='component', unit='K')
            logg = self.get_value(qualifier='logg', component=ldcs_param.component, context='component')
            abun = self.get_value(qualifier='abun', component=ldcs_param.component, context='component')
            photon_weighted = self.get_value(qualifier='intens_weighting', dataset=ldcs_param.dataset, context='dataset') == 'photon'
            ld_coeffs = pb.interpolate_ldcoeffs(teff, logg, abun, ldcs, ld_func, photon_weighted)

            logger.info("interpolated {} ld_coeffs={}".format(ld_func, ld_coeffs))

            ld_coeffs_ret["ld_coeffs@{}@{}".format(ldcs_param.component, ldcs_param.dataset)] = ld_coeffs
            if set_value:
                self.set_value(qualifier='ld_coeffs', component=ldcs_param.component, dataset=ldcs_param.dataset, check_visible=False, value=ld_coeffs)

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

        system_compute = compute if compute_kind=='phoebe' else None
        logger.debug("creating system with compute={} kwargs={}".format(system_compute, kwargs))
        return backends.PhoebeBackend()._create_system_and_compute_pblums(self, system_compute, datasets=datasets, compute_l3=compute_l3, compute_l3_frac=compute_l3_frac, compute_extrinsic=compute_extrinsic, reset=False, **kwargs)

    def compute_l3s(self, compute=None, set_value=False, **kwargs):
        """
        Compute third lights (`l3`) that will be applied to the system from
        fractional third light (`l3_frac`) and vice-versa by assuming that the
        total system flux is equivalent to the sum of the extrinsic (including
        any enabled irradiation and features) passband luminosities
        at t0 divided by 4*pi.  To see how passband luminosities are computed,
        see <phoebe.frontend.bundle.Bundle.compute_pblums>.

        Note: this can only be computed for datasets in which `l3_mode` is set
        to 'fraction of total light' instead of 'flux'.  When this is the case,
        the `l3_frac` parameter takes place of the `l3` parameter.  This method
        simply provides a convenience function for exposing the third light
        that will be adopted in units of flux.

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
        # TODO: consider a b.compute_total/system_fluxes or a b.compute_total_fluxes_to_pblums
        logger.debug("b.compute_l3s")

        datasets = kwargs.pop('dataset', self.filter('l3_mode', check_visible=True).datasets)
        if isinstance(datasets, str):
            datasets = [datasets]

        # don't allow things like model='mymodel', etc
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
            passed, msg = self.run_checks(compute=compute, **kwargs)
            if passed is None:
                # then just raise a warning
                logger.warning(msg)
            if passed is False:
                # then raise an error
                raise ValueError("system failed to pass checks: {}".format(msg))

        system = kwargs.get('system', self._compute_system(compute=compute, datasets=datasets, compute_l3=True, compute_l3_frac=True, **kwargs))

        l3s = {}
        for dataset in datasets:
            l3_mode = self.get_value('l3_mode', dataset=dataset)
            if l3_mode == 'flux':
                l3_frac = system.l3s[dataset]['frac']
                l3s['l3_frac@{}'.format(dataset)] = l3_frac
                if set_value:
                    self.set_value('l3_frac', dataset=dataset, check_visible=False, value=l3_frac)

            elif l3_mode == 'fraction of total light':
                l3_flux = system.l3s[dataset]['flux'] * u.W / u.m**2
                l3s['l3@{}'.format(dataset)] = l3_flux

                if set_value:
                    self.set_value('l3', dataset=dataset, check_visible=False, value=l3_flux)

            else:
                raise NotImplementedError("l3_mode='{}' not supported.".format(l3_mode))

        return l3s

    def compute_pblums(self, compute=None, intrinsic=True, extrinsic=True, **kwargs):
        """
        Compute the passband luminosities that will be applied to the system,
        following all coupling, etc, as well as all relevant compute options
        (ntriangles, distortion_method, etc).  The exposed passband luminosities
        (and any coupling) are computed at t0@system.

        This method allows for computing both intrinsic and extrinsic luminosities.
        Note that pblum scaling is computed (and applied to flux scaling) based
        on intrinsic luminosities.

        Note that luminosities cannot be exposed for any dataset in which
        `pblum_mode` is 'scale to data' as the entire light curve must be
        computed prior to scaling.

        This method is only for convenience and will be recomputed internally
        within <phoebe.frontend.bundle.Bundle.run_compute>.  Alternatively, you
        can create a mesh dataset (see <phoebe.frontend.bundle.Bundle.add_dataset>
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
        * `intrinsic` (bool, optional, default=True): whether to include
            intrinsic (excluding irradiation & features) pblums.  These
            will be exposed in the returned dictionary as pblum@...
        * `extrinsic` (bool, optional, default=False): whether to include
            extrinsic (irradiation & features) pblums.  These will be exposed
            as pblum_ext@...
        * `component` (string or list of strings, optional): label of the
            component(s) requested. If not provided, will be provided for all
            components in the hierarchy.
        * `dataset` (string or list of strings, optional): label of the
            dataset(s) requested.  If not provided, will be provided for all
            datasets attached to the bundle.
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
        """
        logger.debug("b.compute_pblums")

        datasets = kwargs.pop('dataset', self.datasets)
        if isinstance(datasets, str):
            datasets = [datasets]
        components = kwargs.pop('component', self.components)
        if isinstance(components, str):
            components = [components]

        # don't allow things like model='mymodel', etc
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
            passed, msg = self.run_checks(compute=compute, **kwargs)
            if passed is None:
                # then just raise a warning
                logger.warning(msg)
            if passed is False:
                # then raise an error
                raise ValueError("system failed to pass checks: {}".format(msg))

        pblums = {}
        for compute_extrinsic in [False, True]:
            if compute_extrinsic and not extrinsic:
                continue
            if not compute_extrinsic and not intrinsic:
                continue

            # TODO: can we prevent rebuilding the entire system the second time if both intrinsic and extrinsic are True?
            system = kwargs.get('system', self._compute_system(compute=compute, datasets=datasets, compute_l3=False, compute_extrinsic=compute_extrinsic, **kwargs))

            t0 = self.get_value('t0', context='system', unit=u.d)
            for component, star in system.items():
                if component not in components:
                    continue
                for dataset in star._pblum_scale.keys():
                    if dataset not in datasets:
                        continue
                    if compute_extrinsic:
                        logger.debug("computing (extrinsic) observables for {}".format(dataset))
                        system.populate_observables(t0, ['lc'], [dataset],
                                                    ignore_effects=False)

                    pblums["{}@{}@{}".format('pblum_ext' if compute_extrinsic else 'pblum', component, dataset)] = float(star.compute_luminosity(dataset)) * u.W

        return pblums

    def add_compute(self, kind='phoebe', **kwargs):
        """
        Add a set of computeoptions for a given backend to the bundle.
        The label (`compute`) can then be sent to <phoebe.frontend.bundle.Bundle.run_compute>.

        If not provided, `compute` will be created for you and can be
        accessed by the `compute` attribute of the returned
        <phoebe.parameters.ParameterSet>.

        Available kinds can be found in <phoebe.parameters.compute> and include:
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
            self.remove_compute(compute=kwargs['compute'])
            # check the label again, just in case kwargs['compute'] belongs to
            # something other than compute
            self._check_label(kwargs['compute'], allow_overwrite=False)

        logger.info("adding {} '{}' compute to bundle".format(metawargs['kind'], metawargs['compute']))
        self._attach_params(params, **metawargs)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.__name__
        self._add_history(redo_func='add_compute',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_compute',
                          undo_kwargs={'compute': kwargs['compute']})


        # since we've already processed (so that we can get the new qualifiers),
        # we'll only raise a warning
        self._kwargs_checks(kwargs, ['overwrite'], warning_only=True)

        return self.get_compute(**metawargs)

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
        """
        kwargs['compute'] = compute
        kwargs['context'] = 'compute'
        self.remove_parameters_all(**kwargs)

    def remove_computes_all(self):
        """
        Remove all compute options from the bundle.  To remove a single set
        of compute options see <phoebe.frontend.bundle.Bundle.remove_compute>.
        """
        for compute in self.computes:
            self.remove_compute(compute)

    def rename_compute(self, old_compute, new_compute):
        """
        Change the label of compute options attached to the Bundle.

        Arguments
        ----------
        * `old_compute` (string): current label of the compute options (must exist)
        * `new_compute` (string): the desired new label of the compute options
            (must not yet exist)

        Raises
        --------
        * ValueError: if the value of `new_compute` is forbidden or already exists.
        """
        # TODO: raise error if old_compute not found?

        self._check_label(new_compute)
        self._rename_label('compute', old_compute, new_compute)


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
            with the same name will be overwritten - including 'latest'.
            See also <phoebe.frontend.bundle.Bundle.rename_model> to rename
            a model after creation.
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
        * `skip_checks` (bool, optional, default=False): whether to skip calling
            <phoebe.frontend.bundle.Bundle.run_checks> before computing the model.
            NOTE: some unexpected errors could occur for systems which do not
            pass checks.
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
        if isinstance(detach, str):
            # then we want to temporarily go in to client mode
            self.as_client(server=detach)
            self.run_compute(compute=compute, model=model, time=time, **kwargs)
            self.as_client(False)
            return self.get_model(model)

        # protomesh and pbmesh were supported kwargs in 2.0.x but are no longer
        # so let's raise an error if they're passed here
        if 'protomesh' in kwargs.keys():
            raise ValueError("protomesh is no longer a valid option")
        if 'pbmesh' in kwargs.keys():
            raise ValueError("pbmesh is no longer a valid option")

        if model is None:
            model = 'latest'

        self._check_label(model, allow_overwrite=kwargs.get('overwrite', model=='latest'))

        if model in self.models and kwargs.get('overwrite', model=='latest'):
            logger.warning("overwriting model: {}".format(model))
            self.remove_model(model)
            # check the label again, just in case model belongs to something
            # other than model
            self._check_label(model, allow_overwrite=False)

        if isinstance(times, float) or isinstance(times, int):
            times = [times]

        # handle case where compute is not provided
        if compute is None:
            computes = self.get_compute(**kwargs).computes
            if len(computes)==0:
                # NOTE: this doesn't take **kwargs since we want those to be
                # temporarily overriden as is the case when the compute options
                # are already attached
                self.add_compute()
                computes = self.get_compute().computes
                # now len(computes) should be 1 and will trigger the next
                # if statement

            if len(computes)==1:
                compute = computes[0]
            elif len(computes)>1:
                raise ValueError("must provide label of compute options since more than one are attached")

        # handle the ability to send multiple compute options/backends - here
        # we'll just always send a list of compute options
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
        self._kwargs_checks(kwargs, ['skip_checks', 'jobid', 'overwrite'], ps=computes_ps)

        if not kwargs.get('skip_checks', False):
            passed, msg = self.run_checks(compute=computes, **kwargs)
            if passed is None:
                # then just raise a warning
                logger.warning(msg)
            if passed is False:
                # then raise an error
                raise ValueError("system failed to pass checks: {}".format(msg))

        # let's first make sure that there is no duplication of enabled datasets
        datasets = []
        # compute_ so we don't write over compute which we need if detach=True
        for compute_ in computes:
            # TODO: filter by value instead of if statement once implemented
            for enabled_param in self.filter(qualifier='enabled',
                                             compute=compute_,
                                             context='compute').to_list():
                if enabled_param.get_value():
                    item = (enabled_param.dataset, enabled_param.component)
                    if item in datasets:
                        raise ValueError("dataset {}@{} is enabled in multiple compute options".format(item[0], item[1]))
                    datasets.append(item)

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

            # we'll track everything through the model name as well as
            # a random string, to avoid any conflicts
            jobid = kwargs.get('jobid', parameters._uniqueid())

            # we'll build a python script that can replicate this bundle as it
            # is now, run compute, and then save the resulting model
            script_fname = "_{}.py".format(jobid)
            f = open(script_fname, 'w')
            f.write("import os; os.environ['PHOEBE_ENABLE_PLOTTING'] = 'FALSE'; os.environ['PHOEBE_ENABLE_SYMPY'] = 'FALSE'; os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE';\n")
            f.write("import phoebe; import json\n")
            # TODO: can we skip the history context?  And maybe even other models
            # or datasets (except times and only for run_compute but not run_fitting)
            f.write("bdict = json.loads(\"\"\"{}\"\"\", object_pairs_hook=phoebe.utils.parse_json)\n".format(json.dumps(self.to_json())))
            f.write("b = phoebe.Bundle(bdict)\n")
            # TODO: make sure this works with multiple computes
            compute_kwargs = list(kwargs.items())+[('compute', compute), ('model', model)]
            compute_kwargs_string = ','.join(["{}={}".format(k,"\'{}\'".format(v) if isinstance(v, str) else v) for k,v in compute_kwargs])
            f.write("model_ps = b.run_compute({})\n".format(compute_kwargs_string))
            f.write("model_ps.save('_{}.out', incl_uniqueid=True)\n".format(jobid))
            f.close()

            script_fname = os.path.abspath(script_fname)
            cmd = mpi.detach_cmd.format(script_fname)
            # TODO: would be nice to catch errors caused by the detached script...
            # but that would probably need to be the responsibility of the
            # jobparam to return a failed status and message
            subprocess.call(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)

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

            # return self.get_model(model)
            return job_param

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
                                times_oversampled_sorted = params.get_value('times', dataset=dataset)
                                fluxes_oversampled = params.get_value('fluxes', dataset=dataset)

                                for i,t in enumerate(times_ds):
                                    # rebuild the unsorted oversampled times - see backends._extract_from_bundle_by_time
                                    # TODO: try to optimize this by having these indices returned by the backend itself
                                    times_oversampled_this = np.linspace(t-exptime/2., t+exptime/2., fti_oversample)
                                    sample_inds = np.searchsorted(times_oversampled_sorted, times_oversampled_this)

                                    fluxes[i] = np.mean(fluxes_oversampled[sample_inds])

                                params.set_value(qualifier='times', dataset=dataset, value=times_ds)
                                params.set_value(qualifier='fluxes', dataset=dataset, value=fluxes)


            self._attach_params(params, **metawargs)

        def _scale_fluxes(model_fluxes, scale_factor):
            return model_fluxes * scale_factor

        # scale fluxes whenever pblum_mode = 'scale to data'
        for param in self.filter(qualifier='pblum_mode', value='scale to data').to_list():
            logger.debug("rescaling fluxes to data for dataset='{}'".format(param.dataset))
            ds_times = self.get_dataset(param.dataset).get_value('times')
            ds_fluxes = self.get_dataset(param.dataset).get_value('fluxes')
            ds_sigmas = self.get_dataset(param.dataset).get_value('sigmas')

            model_fluxes = self.get_model(model).get_value('fluxes')
            model_fluxes_interp = self.get_model(model).get_parameter('fluxes').interp_value(times=ds_times)
            scale_factor_approx = np.median(ds_fluxes / model_fluxes_interp)

            # TODO: can we skip this if sigmas don't exist?
            logger.debug("calling curve_fit with estimated scale_factor={}".format(scale_factor_approx))
            popt, pcov = cfit(_scale_fluxes, model_fluxes_interp, ds_fluxes, p0=(scale_factor_approx), sigma=ds_sigmas if len(ds_sigmas) else None)
            scale_factor = popt[0]

            logger.debug("applying scale_factor={} to fluxes@{}".format(scale_factor, param.dataset))
            self.get_model(model).set_value('fluxes', dataset=param.dataset, value=model_fluxes*scale_factor)

            for param in self.get_model(model, dataset=param.dataset, kind='mesh').to_list():
                if param.qualifier in ['intensities', 'abs_intensities', 'normal_intensities', 'abs_normal_intensities', 'pblum_ext']:
                    logger.debug("applying scale_factor={} to {} parameter in mesh".format(scale_factor, param.qualifier))
                    param.set_value(param.get_value() * scale_factor)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['compute'] = computes if len(computes)>1 else computes[0]
        redo_kwargs['model'] = model

        self._add_history(redo_func='run_compute',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_model',
                          undo_kwargs={'model': model})

        return self.get_model(model)

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

    def remove_model(self, model, **kwargs):
        """
        Remove a 'model' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>

        Arguments
        ----------
        * `model` (string): the label of the model to be removed.
        * `**kwargs`: other filter arguments to be sent to
            <phoebe.parameters.ParameterSet.remove_parameters_all>.  The following
            will be ignored: model, context.
        """
        kwargs['model'] = model
        kwargs['context'] = 'model'
        self.remove_parameters_all(**kwargs)

    def remove_models_all(self):
        """
        Remove all models from the bundle.  To remove a single model see
        <phoebe.frontend.bundle.Bundle.remove_model>.
        """
        for model in self.models:
            self.remove_model(model=model)

    def rename_model(self, old_model, new_model):
        """
        Change the label of a model attached to the Bundle.

        Arguments
        ----------
        * `old_model` (string): current label of the model (must exist)
        * `new_model` (string): the desired new label of the model
            (must not yet exist)

        Raises
        --------
        * ValueError: if the value of `new_model` is forbidden or already exists.
        """
        # TODO: raise error if old_feature not found?

        self._check_label(new_model)
        self._rename_label('model', old_model, new_model)

    # TODO: ability to copy a posterior to a prior or have a prior reference an attached posterior (for drawing in fitting)
    def add_prior(self, twig=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """

        raise NotImplementedError

        param = self.get_parameter(twig=twig, **kwargs)
        # TODO: make sure param is a float parameter?

        func = _get_add_func(_distributions, 'prior')

        # TODO: send smart defaults for priors based on limits of parameter
        params = func(**kwargs)

        metawargs = {k: v for k, v in params.meta.items()
                     if k not in ['uniqueid', 'uniquetwig', 'twig']}
        metawargs['context'] = 'prior'

        logger.info("adding prior on '{}' parameter".format(param.uniquetwig))
        self._attach_params(params, **metawargs)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.__name__
        self._add_history(redo_func='add_prior',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_prior',
                          undo_kwargs={'twig': param.uniquetwig})

        # return params
        return self.get_prior(**metawargs)

    def get_prior(self, twig=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        raise NotImplementedError
        kwargs['context'] = 'prior'
        return self.filter(twig=twig, **kwargs)

    def enable_prior(self, twig=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # instead of set_adjust(True)
        raise NotImplementedError

    def disable_prior(self, twig=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # instead of set_adjust(False)
        raise NotImplementedError

    def draw_from_prior(self, twig=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError

    def remove_prior(self, twig=None, **kwargs):
        """
        Remove a 'prior' from the bundleself.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    def add_fitting(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    def get_fitting(self, fitting=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        raise NotImplementedError
        if fitting is not None:
            kwargs['fitting'] = fitting
        kwargs['context'] = 'fitting'
        return self.filter(**kwargs)

    def remove_fitting(self):
        """
        [NOT IMPLEMENTED]

        Remove a 'fitting' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    def run_fitting(self, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # TODO: kwargs override temporarily (fittingparams.get_value(qualifier, **kwargs))

        # TODO: don't forget add_history (probably not undoable)
        raise NotImplementedError

    def get_posterior(self, twig=None, feedback=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        raise NotImplementedError
        kwargs['context'] = 'posterior'
        return self.filter(twig=twig, **kwargs)

    def draw_from_posterior(self, twig=None, feedback=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError

    def remove_posterior(self, twig=None, feedback=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        Remove a 'posterior' from the bundleself.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    # TODO: make feedback work more like models above.  Maybe we could even
    # submit a job and detach, loading the results later.  See notes and tasks
    # about re-working fitting interface.

    def add_feedback(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # TODO: don't forget to add_history
        raise NotImplementedError

    def get_feedback(self, feedback=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        raise NotImplementedError
        if feedback is not None:
            kwargs['feedback'] = feedback
        kwargs['context'] = 'feedback'
        return self.filter(**kwargs)

    def remove_feedback(self, feedback=None):
        """
        [NOT IMPLEMENTED]

        Remove a 'feedback' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        """
        # TODO: don't forget add_history
        raise NotImplementedError
        self.remove_parameters_all(context='feedback', feedback=feedback)

    def add_plugin(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # TODO: don't forget to add_history
        raise NotImplementedError

    def get_plugin(self):
        """
        [NOT IMPLEMENTED]

        See also:
        * <phoebe.parameters.ParameterSet.filter>

        Arguments
        ----------
        * `twig`: (string, optional, default=None): the twig used for filtering
        * `**kwargs`: any other tags to do the filtering (excluding twig and context)

        Returns:
        * a <phoebe.parameters.ParameterSet> object.
        """
        raise NotImplementedError

    def remove_plugin(self):
        """
        [NOT IMPLEMENTED]

        Remove a 'plugin' from the bundle.

        See also:
        * <phoebe.parameters.ParameterSet.remove_parameters_all>
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    def run_plugin(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError
