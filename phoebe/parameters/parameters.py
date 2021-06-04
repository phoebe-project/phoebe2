"""Parameters and ParameterSets.

General logic for all Parameters and ParameterSets which makeup the overall
framework of the PHOEBE 2.0 frontend.
"""

from phoebe.constraints.expression import ConstraintVar
# from phoebe.constraints import builtin
from phoebe.parameters.twighelpers import _uniqueid_to_uniquetwig
from phoebe.parameters.twighelpers import _twig_to_uniqueid
from phoebe.frontend import tabcomplete
from phoebe.dependencies import nparray, distl
from phoebe.utils import parse_json, phase_mask_inds
from phoebe import helpers as _helpers

import sys
import random
import string
import functools
import itertools
import re
import sys
import os
import difflib
import time
import types
import tempfile
import subprocess
from collections import OrderedDict
from fnmatch import fnmatch
from copy import deepcopy as _deepcopy
import readline
import numpy as np

import json
# try:
#     import ujson
# except ImportError:
#     _can_ujson = False
# else:
#     _can_ujson = True

# ujson is currently causing issues loading in mesh data (and/or large files)
_can_ujson = False

import webbrowser
from datetime import datetime

if os.getenv('PHOEBE_ENABLE_EXTERNAL_JOBS', 'FALSE').upper() == 'TRUE':
    try:
        import requests
    except ImportError:
        _can_requests = False
    else:
        _can_requests = True
else:
    _can_requests = False

# things needed to be imported at top-level for constraints to solve:
from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt

from phoebe import u
from phoebe import conf
from phoebe import list_passbands, list_installed_passbands, list_online_passbands, download_passband

if os.getenv('PHOEBE_ENABLE_SYMPY', 'FALSE').upper() == 'TRUE':
    try:
        import sympy
    except ImportError:
        _use_sympy = False
    else:
        _use_sympy = True
else:
    _use_sympy = False

_use_sympy = False
_is_server = False

if os.getenv('PHOEBE_ENABLE_PLOTTING', 'TRUE').upper() == 'TRUE':
    try:
        from phoebe.dependencies import autofig
    except (ImportError, TypeError):
        _use_autofig = False
        _phoebecolorsdict = {}
    else:
        _use_autofig = True
        # add the PHOEBE palette to autofig.cyclers
        black = (19./255, 19./255, 19./255) #131313
        blue = (43./255, 113./255, 177./255) #2B71B1
        orange = (255./255, 112./255, 47./255) #FF702F
        yellow = (255./255, 205./255, 47./255) #FFCD2F
        green = (34./255,183./255,127./255) #22B77F
        fuchsia = (237./255,49./255,112./255) #ED3170
        _phoebecolors = [black, blue, orange, green, yellow, fuchsia]
        _phoebecolorsdict = {'black': "#131313",
                             'blue': "#2B71B1",
                             'orange': "#FF702F",
                             'green': "#22B77F",
                             'red': '#F92E3D',
                             'purple': '#6D2EB8',
                             'pink': "#ED3170",
                             'yellow': "#FFCD2F"}

        ofr=6
        for i in [5,3,1,2,4]:
            _phoebecolors += [tuple((np.array(basecolor)+np.array([0,i/ofr,i/ofr]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
            _phoebecolors += [tuple((np.array(basecolor)+np.array([i/ofr,0,i/ofr]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
            _phoebecolors += [tuple((np.array(basecolor)+np.array([i/ofr,i/ofr,0]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
            _phoebecolors += [tuple((np.array(basecolor)+np.array([0,i/ofr,0]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
            _phoebecolors += [tuple((np.array(basecolor)+np.array([0,0,i/ofr]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
            _phoebecolors += [tuple((np.array(basecolor)+np.array([i/ofr,0,0]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
            _phoebecolors += [tuple((np.array(basecolor)+np.array([i/ofr,i/ofr,i/ofr]))%1) for basecolor in [blue, orange, green, yellow, fuchsia]]
        autofig.cyclers._mplcolors = _phoebecolors + autofig.cyclers._mplcolors

    try:
        import corner
    except (ImportError, TypeError):
        _use_corner = False
    else:
        _use_corner = True

    try:
        from dynesty import plotting as dyplot
    except (ImportError, TypeError):
        _use_dyplot = False
    else:
        _use_dyplot = True

else:
    _use_autofig = False
    _use_corner = False
    _use_dyplot = False



import logging
logger = logging.getLogger("PARAMETERS")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

_parameter_class_that_require_bundle = ['TwigParameter',
                                        'ConstraintParameter', 'DistributionParameter',
                                        'JobParameter']

_meta_fields_twig = ['time', 'qualifier', 'feature', 'component',
                     'dataset', 'constraint', 'distribution', 'compute', 'model',
                     'solver', 'solution', 'figure', 'kind',
                     'context']

_meta_fields_all = _meta_fields_twig + ['twig', 'uniquetwig', 'uniqueid']
_meta_fields_filter = _meta_fields_all + ['constraint_func', 'value']

_contexts = ['system', 'component', 'feature',
             'dataset', 'constraint', 'distribution', 'compute', 'model',
             'solver', 'solution', 'figure', 'setting']

# define a list of default_forbidden labels
# an individual ParameterSet may build on this list with components, datasets,
# etc for labels
# components and datasets should also forbid this list
_forbidden_labels = _deepcopy(_meta_fields_all)

# forbid all "contexts", although should already be in _meta_fields_all
_forbidden_labels += _contexts

#
_forbidden_labels += ['True', 'False', 'true', 'false', 'None', 'none', 'null']

# forbid all "methods"
_forbidden_labels += ['value', 'adjust', 'default_unit',
                      'quantity',
                      'unit', 'timederiv', 'visible_if', 'description', 'result', 'advanced', 'readonly', 'latexfmt']

# forbid some random things
_forbidden_labels += ['protomesh', 'pbmesh']
_forbidden_labels += ['bol']

# forbid all kinds
_forbidden_labels += ['lc', 'rv', 'lp', 'sp', 'orb', 'mesh']
_forbidden_labels += ['star', 'orbit', 'envelope']
_forbidden_labels += ['spot', 'pulsation']
_forbidden_labels += ['phoebe', 'legacy', 'jktebop', 'photodynam', 'ellc']



# we also want to forbid any possible qualifiers
# from system:
_forbidden_labels += ['t0', 'ra', 'dec', 'epoch', 'distance', 'parallax', 'vgamma', 'hierarchy',
                     'Rv', 'Av', 'ebv', 'extinction']

# from setting:
_forbidden_labels += ['phoebe_version', 'dict_filter',
                      'dict_set_all', 'run_checks_compute', 'run_checks_solver',
                      'run_checks_solution', 'run_checks_figure',
                      'auto_add_figure', 'auto_remove_figure', 'web_client', 'web_client_url']

# from component
_forbidden_labels += ['requiv', 'requiv_max', 'requiv_min', 'teff', 'abun', 'logg',
                      'fillout_factor', 'pot_min', 'pot_max',
                      'period_anom',
                      'syncpar', 'period', 'pitch', 'yaw', 'incl', 'long_an',
                      'gravb_bol', 'irrad_frac_refl_bol', 'irrad_frac_lost_bol',
                      'ld_mode_bol', 'ld_func_bol',
                      'ld_coeffs_source_bol', 'ld_coeffs_bol',
                      'mass', 'dpdt', 'per0',
                      'dperdt', 'ecc', 'deccdt', 't0_perpass', 't0_supconj',
                      't0_ref', 'mean_anom', 'q', 'sma', 'asini', 'ecosw', 'esinw',
                      'teffratio', 'requivratio', 'requivsumfrac'
                      ]

# from dataset:
_forbidden_labels += ['times', 'fluxes', 'sigmas', 'sigmas_lnf',
                     'compute_times', 'compute_phases', 'compute_phases_t0',
                     'phases_period', 'phases_dpdt', 'phases_t0', 'mask_enabled', 'mask_phases',
                     'solver_times', 'expose_samples', 'expose_failed',
                     'ld_mode', 'ld_func', 'ld_coeffs', 'ld_coeffs_source',
                     'passband', 'intens_weighting',
                     'pblum_mode', 'pblum_ref', 'pblum', 'pbflux',
                     'pblum_dataset', 'pblum_component',
                     'l3_mode', 'l3', 'l3_frac',
                     'exptime', 'rvs', 'wavelengths', 'rv_offset',
                     'flux_densities', 'profile_func', 'profile_rest', 'profile_sv',
                     'Ns', 'time_ecls', 'time_ephems', 'etvs',
                     'us', 'vs', 'ws', 'vus', 'vvs', 'vws',
                     'include_times', 'columns', 'coordinates',
                     'uvw_elements', 'xyz_elements',
                     'pot', 'rpole', 'volume',
                     'xs', 'ys', 'zs', 'vxs', 'vys', 'vzs',
                     'nxs', 'nys', 'nzs', 'nus', 'nvs', 'nws',
                     'areas', 'rs', 'rprojs', 'loggs', 'teffs', 'mus',
                     'visible_centroids', 'visibilities',
                     'intensities', 'abs_intensities',
                     'normal_intensities', 'abs_normal_intensities',
                     'boost_factors', 'ldint', 'ptfarea',
                     'pblum', 'pblum_ext', 'abs_pblum', 'abs_pblum_ext']


# from compute:
_forbidden_labels += ['enabled', 'dynamics_method', 'ltte', 'comments',
                      'gr', 'stepsize', 'integrator',
                      'irrad_method', 'boosting_method', 'mesh_method', 'distortion_method',
                      'ntriangles', 'rv_grav',
                      'mesh_offset', 'mesh_init_phi', 'horizon_method', 'eclipse_method',
                      'atm', 'lc_method', 'rv_method', 'fti_method', 'fti_oversample',
                      'pblum_method', 'requiv_max_limit',
                      'etv_method', 'etv_tol',
                      'gridsize', 'refl_num', 'ie',
                      'stepsize', 'orbiterror', 'ringsize',
                      'exact_grav', 'grid', 'hf',
                      'sample_from', 'sample_from_combine', 'sample_num', 'sample_mode'
                      ]

# from solver:
_forbidden_labels += ['nwalkers', 'niters', 'priors', 'init_from',
                      'lc_datasets', 'rv_datasets', 'lc_combine',
                      'phase_bin', 'phase_nbins',
                      'algorithm', 'duration', 'minimum_n_cycles', 'frequency_factor',
                      'samples_per_peak', 'nyquist_factor',
                      't0_near_times', 'sample_periods', 'sample_frequencies', 'objective',
                      'expose_lnlikelihoods', 'expose_lnprobabilities', 'fit_parameters', 'initial_values',
                      'expose_model', 'gtol', 'norm', 'xtol', 'ftol',
                      'priors_combine', 'maxiter', 'maxfev', 'adaptive',
                      'xatol', 'fatol', 'bounds', 'bounds_combine', 'bounds_sigma',
                      'strategy', 'popsize', 'continue_from', 'init_from_combine',
                      'burnin_factor', 'thin_factor', 'progress_every_niters',
                      'nlive', 'maxcall', 'lc_geometry', 'rv_geometry', 'lc_periodogram', 'rv_periodogram', 'ebai',
                      'nelder_mead', 'differential_evolution', 'cg', 'powell', 'emcee', 'dynesty']

# from solution:
_forbidden_labels += ['primary_width', 'secondary_width',
                      'primary_phase', 'secondary_phase',
                      'primary_depth', 'secondary_depth',
                      'eclipse_edges',
                      'fitted_uniqueids', 'fitted_twigs', 'fitted_values', 'fitted_units',
                      'adopt_parameters', 'adopt_distributions', 'distributions_convert', 'distributions_bins',
                      'failed_samples', 'lnprobabilities', 'acceptance_fractions',
                      'autocorr_time', 'burnin', 'thin', 'lnprob_cutoff',
                      'progress',
                      'period_factor', 'power',
                      'nlive', 'niter', 'ncall', 'eff', 'samples', 'samples_id', 'samples_it', 'samples_u',
                      'logwt', 'logl', 'logvol', 'logz', 'logzerr', 'information', 'bound', 'bounds',
                      'bound_iter', 'samples_bound', 'scale',
                      'message', 'nfev', 'niter', 'success', 'initial_values',
                      'initial_lnlikelihood', 'fitted_lnlikelihood']


# from feature:
_forbidden_labels += ['colat', 'long', 'radius', 'relteff',
                      'radamp', 'freq', 'l', 'm', 'teffext',
                      'spot', 'gaussian_process', 'pulsation',
                      'kernel', 'log_S0', 'log_Q', 'log_rho',
                      'log_omega0', 'log_sigma', 'eps'
                      ]

# from figure:
_forbidden_labels += ['datasets', 'models', 'components', 'contexts',
                      'x', 'y', 'z',
                      'color_source', 'color', 'c_source', 'c',
                      'marker_source', 'marker',
                      'linestyle_source', 'linestyle',
                      'xlabel_source', 'xlabel', 'ylabel_source', 'ylabel',
                      'xunit_source', 'xunit', 'yunit_source', 'yunit',
                      'xlim_source', 'xlim', 'ylim_source', 'ylim',
                      'fc_source', 'fc_column', 'fc', 'fclim_source', 'fclim', 'fcunit_source', 'fcunit', 'fclabel_source', 'fclabel',
                      'fcmap_source', 'fcmap',
                      'ec_source', 'ec_column', 'ec', 'eclim_source', 'eclim', 'ecunit_source', 'ecunit', 'eclabel_source', 'eclabel',
                      'ecmap_source', 'ecmap',
                      'default_time_source', 'default_time', 'time_source', 'time',
                      'uncover', 'highlight', 'draw_sidebars',
                      'latex_repr',
                      'legend']

# ? and * used for wildcards in twigs
_twig_delims = ' \t\n`~!#$%^&)-=+]{}\\|;,<>/:'


_singular_to_plural = {'time': 'times', 'phase': 'phases', 'flux': 'fluxes', 'sigma': 'sigmas',
                       'rv': 'rvs', 'flux_density': 'flux_densities',
                       'time_ecl': 'time_ecls', 'time_ephem': 'time_ephems', 'N': 'Ns',
                       'x': 'xs', 'y': 'ys', 'z': 'zs', 'vx': 'vxs', 'vy': 'vys',
                       'vz': 'vzs', 'nx': 'nxs', 'ny': 'nys', 'nz': 'nzs',
                       'u': 'us', 'v': 'vs', 'w': 'ws', 'vu': 'vus', 'vv': 'vvs',
                       'vw': 'vws', 'nu': 'nus', 'nv': 'nvs', 'nw': 'nws',
                       'cosbeta': 'cosbetas', 'logg': 'loggs', 'teff': 'teffs',
                       'r': 'rs', 'rproj': 'rprojs', 'mu': 'mus',
                       'visibility': 'visibilities'}
_plural_to_singular = {v:k for k,v in _singular_to_plural.items()}

def _singular_to_plural_get(k):
    return _singular_to_plural.get(k, k)

def _plural_to_singular_get(k):
    return _plural_to_singular.get(k, k)

def _return_ps(b, ps):
    """set the _filter of the ps to be the uniqueids and return"""
    if isinstance(ps, list):
        ps = ParameterSet(ps)
    if ps._bundle is None:
        ps._bundle = b
    if not len(ps._filter.keys()):
        ps._filter = {'uniqueid': ps.uniqueids}
    return ps

def send_if_client(fctn):
    """Intercept and send to the server if bundle is in client mode."""
    @functools.wraps(fctn)
    def _send_if_client(self, *args, **kwargs):
        fctn_map = {'set_quantity': 'set_value',
                    'set_value': 'set_value',
                    'set_default_unit': 'set_default_unit',
                    'flip_for': 'flip_constraint'}
        b = self._bundle
        if b is not None and hasattr(b, 'is_client') and b.is_client:
            # TODO: self._filter???
            # TODO: args???
            requestid = _uniqueid(6)
            self._bundle._waiting_on_server = requestid

            method = fctn_map.get(fctn.__name__, 'bundle_method')
            # NOTE: the deepcopy is necessary here so we don't overwrite self._filter
            d = self._filter.copy() if hasattr(self, '_filter') \
                else {'uniqueid': self.uniqueid}
            d['bundleid'] = b._bundleid
            d['requestid'] = requestid
            d['args'] = args
            if method == 'bundle_method':
                d['method'] = fctn.__name__
            for k, v in kwargs.items():
                if hasattr(v, 'to_json'):
                    v = v.to_json()
                d[k] = v

            if d.get('method', None) in ['run_compute', 'run_solver']:
                detach = d.pop('detach', False)

            logger.info('emitting {} ({}) to server'.format(method, d))

            b._socketio.emit(method, d)

            while self._bundle._waiting_on_server:
                # print("waiting on server for method: {}, bundleid: {}, requestid: {}".format(d.get('method', None), self._bundle._bundleid, self._bundle._waiting_on_server))
                time.sleep(0.2)

            ret_ = self._bundle._server_changes
            self._bundle._server_changes = None

            if d.get('method', None) in ['run_compute', 'run_solver'] and not detach:
                # then we need to sit in a poll loop until the job returns as completed
                # otherwise the user will have to call b.attach_job manually?  Or will the results just come in once done?
                # should be the single job parameter
                ret_ += self._bundle.attach_job(uniqueid=ParameterSet([p for p in ret_.to_list() if p._bundle is not None]).get_parameter(qualifier='detached_job', **_skip_filter_checks).uniqueid)

            if isinstance(ret_, ParameterSet) and not len(ret_._filter.keys()):
                ret_ = _return_ps(self._bundle, ret_)

            return ret_

        else:
            return fctn(self, *args, **kwargs)

    return _send_if_client


def _uniqueid(n=30):
    """Return a unique string with length n.

    :parameter int N: number of character in the uniqueid
    :return: the uniqueid
    :rtype: str
    """
    return ''.join(random.SystemRandom().choice(
                   string.ascii_uppercase + string.ascii_lowercase)
                   for _ in range(n))

_clientid = 'python-'+_uniqueid(5)

def _is_unit(unit):
    return isinstance(unit, u.Unit) or isinstance(unit, u.CompositeUnit) or isinstance(unit, u.IrreducibleUnit)

def _value_for_constraint(item, constraintparam=None):
    if getattr(item, 'keep_in_solar_units', False):
        # for example, constants defined in the constraint itself can have
        # this attribute added to force them to stay in solar units in the constraint
        return item.value
    elif constraintparam is not None and constraintparam.in_solar_units:
        return u.to_solar(item).value
    else:
        return item.si.value

def _extract_index_from_string(s):
    if s is None:
        return s, None
    if isinstance(s, list):
        all = [_extract_index_from_string(si) for si in s]
        return [a[0] for a in all], [a[1] for a in all]
    index = None
    if '[' in s and ']' in s:
        ind0 = s.index('[')
        ind1 = s.index(']')
        index = int(float(s[ind0+1:ind1]))
        s = s[:ind0] + s[ind1+1:]
    if '[' in s or ']' in s:
        raise ValueError("could not succesfully extract single index")

    return s, index

def _corner_twig(param, use_tex=True, index=None):
    if use_tex and param._latexfmt is not None:
        if index is None:
            return param.latextwig
        else:
            return "$"+param.latextwig.replace("$", "")+"[{}]".format(index)+"$"
    if param.context == 'system':
        return param.qualifier
    else:
        if index is None:
            return '{}@{}'.format(param.qualifier, getattr(param, param.context))
        else:
            return '{}[{}]@{}'.format(param.qualifier, index, getattr(param, param.context))

def _corner_label(param, index=None, use_tex=True):
    if param.default_unit.to_string():
        return '{} [{}]'.format(_corner_twig(param, use_tex=use_tex, index=index), param.default_unit)
    return _corner_twig(param, use_tex=use_tex, index=index)


def parameter_from_json(dictionary, bundle=None):
    """Load a single parameter from a JSON dictionary.

    Arguments
    ----------
    * `parameter` (dict): the dictionarry containing the parameter information
    * `bundle` (<phoebe.frontend.bundle.Bundle>, optional): the bundle object
        that the parameter will be attached to

    Returns
    --------
    * an instantiated <phoebe.parameters.Parameter> object>
    """
    if isinstance(dictionary, str):
        dictionary = json.loads(dictionary, object_pairs_hook=parse_json)

    classname = dictionary.get('Class')

    if classname not in _parameter_class_that_require_bundle:
        bundle = None

    # now let's do some dirty magic and get the actual classitself
    # from THIS module.  __name__ is a string to lookup this module
    # from the sys.modules dictionary
    cls = getattr(sys.modules[__name__], classname)

    return cls._from_json(bundle, **dictionary)

def _instance_in(obj, *types):
    for typ in types:
        if isinstance(obj, typ):
            return True

    return False

def _fnmatch(to_this, expression_or_string):
    if isinstance(expression_or_string, str) and ('*' in expression_or_string or '?' in expression_or_string):
        return fnmatch(to_this, expression_or_string)
    else:
        return expression_or_string == to_this

class JupyterUI(object):
    def __init__(self, url):
        self.url = url
    def __repr__(self):
        return self.url
    def _repr_html_(self):
        return "<iframe src=\"{}\" style='width:100%;min-height:400px'></iframe>".format(self.url)

class ParameterSetInfo(dict):
    def __init__(self, ps, attribute):
        super(dict, self).__init__()
        self._attribute = attribute

        for qualifier in ps.qualifiers:
            entries_this_qualifier = {}
            for param in ps.filter(qualifier=qualifier, check_visible=False, check_default=False).to_list():
                if not hasattr(param, attribute):
                    continue

                value = str(getattr(param, attribute))
                if value not in entries_this_qualifier.keys():
                    entries_this_qualifier[value] = []

                entries_this_qualifier[value].append(param.uniqueid)

            if not len(entries_this_qualifier):
                # then the hasattr hasn't returned anything, so we don't need to
                # do anything for this qualifier
                continue


            if len(entries_this_qualifier) == 1:
                self[qualifier] = value
            else:
                for value, uniqueids in entries_this_qualifier.items():
                    self[ps.filter(uniqueid=uniqueids, check_visible=False, check_default=False).common_twig] = value

    def __repr__(self):
        return "<ParameterSetInfo (qualifier/twig: {}): {}>".format(self._attribute, {k:v for k,v in self.items()})

    def __str__(self):
        """String representation for the ParameterSet."""
        if len(self.keys()):
            param_info = "\n".join("{:>32}: {}".format(k,v) for k,v in sorted(self.items()))
        else:
            param_info = "NO PARAMETERS"

        return "ParameterSetInfo: (qualfier/twig: {})\n".format(self._attribute)+param_info



class ParameterSet(object):
    """ParameterSet.

    The ParameterSet is an abstract list of Parameters which can then be
    filtered into another ParameterSet or Parameter by filtering on set tags of
    the Parameter or on "twig" notation (a single string using '@' symbols to
    separate these same tags).
    """

    def __init__(self, params=[]):
        """Initialize a new ParameterSet.

        Arguments
        ---------
        * `params` (list, optional, default=[]): list of
            <phoebe.parameters.Parameter> objects.

        Returns:
        --------
        * an instantiated <phoebe.parameters.ParameterSet>.
        """
        self._bundle = None
        self._filter = {}

        if isinstance(params, str):
            params = json.loads(params)

        if len(params) and not isinstance(params[0], Parameter):
            # then attempt to load as if json
            self._params = []
            for param_dict in params:
                self._params += [parameter_from_json(param_dict, self)]

            for param in self._params:
                param._bundle = self
        else:
            self._params = params

        self._qualifier = None
        self._time = None
        self._component = None
        self._dataset = None
        self._figure = None
        self._constraint = None
        self._distribution = None
        self._compute = None
        self._model = None
        self._solver = None
        self._solution = None
        # self._plugin = None
        self._kind = None
        self._context = None

        # just as a dummy, this'll be filled and handled by to_dict()
        self._next_field = 'key'

        self._set_meta()

        # force an update to _next_field
        self.to_dict(skip_return=True)

        # set tab completer
        readline.set_completer(tabcomplete.Completer().complete)
        readline.set_completer_delims(_twig_delims)
        readline.parse_and_bind("tab: complete")

    def __repr__(self):
        """Representation for the ParameterSet."""
        self.to_dict()  # <-- to force an update to _next_field
        if len(self._params):
            if len(self.keys()) and self.keys()[0] is not None:
                return "<ParameterSet: {} parameters | {}s: {}>"\
                    .format(len(self._params),
                            self._next_field,
                            ', '.join(self.keys()))
            else:
                return "<ParameterSet: {} parameters>"\
                    .format(len(self._params))
        else:
            return "<ParameterSet: EMPTY>"

    def __str__(self):
        """String representation for the ParameterSet."""
        if len(self._params):
            param_info = "\n".join([p.to_string_short() for p in self._params])
        else:
            param_info = "NO PARAMETERS"

        return "ParameterSet: {} parameters\n".format(len(self._params))+param_info

    def __lt__(self, other):
        raise NotImplementedError("comparison operators with ParameterSets are not supported")

    def __le__(self, other):
        raise NotImplementedError("comparison operators with ParameterSets are not supported")

    def __gt__(self, other):
        raise NotImplementedError("comparison operators with ParameterSets are not supported")

    def __ge__(self, other):
        raise NotImplementedError("comparison operators with ParameterSets are not supported")

    def __eq__(self, other):
        raise NotImplementedError("comparison operators with ParameterSets are not supported")

    def __ne__(self, other):
        raise NotImplementedError("comparison operators with ParameterSets are not supported")

    def copy(self):
        """
        Deepcopy the <<class>>.
        """
        return _deepcopy(self)

    def __copy__(self):
        return self.copy()

    @property
    def info(self):
        """
        Shortcut to <phoebe.parameters.ParameterSet.get_info> with the default
        arguments.
        """
        return self.get_info()

    def get_info(self, attribute='description', **kwargs):
        """
        Access any available attribute across the ParameterSet.  This returns
        a dictionary-like object where keys are the qualifier or twig
        and values are according to the value passed to `attribute`.  Any
        entries that can be merged (because they have the same value) will
        be into a single entry.  Any that cannot, will show the shortest
        common twig of all parameters that apply to that entry.  Parameters
        without the requested `attribute` will omitted (non-FloatParameters will
        be excluded if `attribute` is 'default_unit', for example).

        See also:
        * <phoebe.parameters.ParameterSet.info>

        Arguments
        -------------
        * `attribute` (string, optional, default='description'): attribute
            to access for each parameter.  This will be the values in the
            returned dictionary object.
        * `**kwargs`: additional keyword arguments are first sent to
            <phoebe.parameters.ParameterSet.filter>.

        Returns
        -----------
        * a dictionary-like object that is subclassed to provide a nice
          representation when printed to the screen.
        """
        if len(kwargs.items()):
            return self.filter(**kwargs).get_info(attribute)
        return ParameterSetInfo(self, attribute)

    @property
    def meta(self):
        """Dictionary of all meta-tags.

        This is a shortcut to the <phoebe.parameters.ParameterSet.get_meta> method.
        See <phoebe.parameters.ParameterSet.get_meta> for the ability to ignore
        certain keys.

        See all the meta-tag properties that are shared by ALL Parameters. If a
        given value is 'None', that means that it is not shared among ALL
        Parameters.  To see the different values among the Parameters, you can
        access that attribute.

        For example: if `ps.meta['context'] == None`, you can see all values
        through `ps.contexts`.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        Returns
        ----------
        * (dict) an ordered dictionary of all tag properties
        """
        return self.get_meta()

    def get_meta(self, ignore=['uniqueid']):
        """Dictionary of all meta-tags, with option to ignore certain tags.

        See all the meta-tag properties that are shared by ALL Parameters.
        If a given value is 'None', that means that it is not shared
        among ALL Parameters.  To see the different values among the
        Parameters, you can access that attribute.

        For example: if `ps.meta['context'] == None`, you can see all values
        through `ps.contexts`.

        See also:
        * <phoebe.parameters.ParameterSet.meta>
        * <phoebe.parameters.ParameterSet.tags>
        * <phoebe.parameters.Parameter.get_meta>

        Arguments
        -----------
        * `ignore` (list, optional, default=['uniqueid']): list of keys to exclude
            from the returned dictionary.

        Returns
        ----------
        * (dict) an ordered dictionary of all tag properties
        """
        return OrderedDict([(k, getattr(self, k))
                            for k in _meta_fields_twig
                            if k not in ignore])

    def set_meta(self, **kwargs):
        """Set the value of tags for all Parameters in this ParameterSet.

        Arguments
        -----------
        * `**kwargs`: tag value pairs to be set for all <phoebe.parameters.Parameter>
            objects in this <phoebe.parameters.ParameterSet>
        """
        for param in self.to_list():
            for k, v in kwargs.items():
                # Here we'll set the attributes (_context, _qualifier, etc)
                if getattr(param, '_{}'.format(k)) is None:
                    setattr(param, '_{}'.format(k), v)

    def _options_for_tag(self, tag, include_default=True):
        # keys_for_this_field = set([getattr(p, tag)
        #                            for p in self.to_list()
        #                            if getattr(p, tag) is not None])

        # especially as the PS gets larger, this is actually somewhat cheaper
        # than building the large list and taking the set.
        keys_for_this_field = []
        for p in self.to_list():
            key = getattr(p, tag)
            if key is not None and key not in keys_for_this_field and (include_default or key!='_default'):
                keys_for_this_field.append(key)

        return keys_for_this_field

    @property
    def tags(self):
        """Returns a dictionary that lists all available tags that can be used
        for further filtering.

        See also:
        * <phoebe.parameters.ParameterSet.meta>
        * <phoebe.parameters.Parameter.meta>

        Will include entries from the plural attributes:
        * <phoebe.parameters.ParameterSet.contexts>
        * <phoebe.parameters.ParameterSet.kinds>
        * <phoebe.parameters.ParameterSet.models>
        * <phoebe.parameters.ParameterSet.computes>
        * <phoebe.parameters.ParameterSet.constraints>
        * <phoebe.parameters.ParameterSet.datasets>
        * <phoebe.parameters.ParameterSet.components>
        * <phoebe.parameters.ParameterSet.features>
        * <phoebe.parameters.ParameterSet.times>
        * <phoebe.parameters.ParameterSet.qualifiers>

        Returns
        ----------
        * (dict) a dictionary of all plural tag attributes.
        """
        ret = {}
        for typ in _meta_fields_twig:
            if typ in ['uniqueid', 'twig', 'uniquetwig']:
                continue

            k = '{}s'.format(typ)
            ret[k] = getattr(self, k)

        return ret

    @property
    def uniqueids(self):
        """Return a list of all uniqueids in this ParameterSet.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        There is no singular version for uniqueid for a ParameterSet.  At the
        <phoebe.parameters.Parameter>, see <phoebe.parameters.Parameter.uniqueid>.

        Returns
        --------
        * (list) a list of all uniqueids for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return [p.uniqueid for p in self.to_list()]

    @property
    def twigs(self):
        """Return a list of all twigs in this ParameterSet.

        See also:
        * <phoebe.parameters.ParameterSet.common_twig>
        * <phoebe.parameters.ParameterSet.tags>

        There is no singular version for twig for a ParameterSet.  At the
        <phoebe.parameters.Parameter>, see <phoebe.parameters.Parameter.twig>.

        Returns
        --------
        * (list) a list of all twigs for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return [p.twig for p in self.to_list()]

    @property
    def common_twig(self):
        """
        The twig that is common between all items in this ParameterSet.
        This twig gives a single string which can point back to this ParameterSet
        (but may include other entries as well).

        See also:
        * <phoebe.parameters.ParameterSet.twigs>

        Returns
        -----------
        * (string) common twig of Parameters in the ParameterSet
        """
        meta = self.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig'])
        return "@".join([getattr(self, k) for k in _meta_fields_twig if meta.get(k) is not None])

    @property
    def qualifier(self):
        """Return the value for qualifier if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.qualifiers>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.qualifier>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._qualifier

    @property
    def qualifiers(self):
        """Return a list of all qualifiers in this ParameterSet.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.qualifier>

        Returns
        --------
        * (list) a list of all qualifiers for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('qualifier')

    @property
    def time(self):
        """Return the value for time if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.times>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.time>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return str(self._time) if self._time is not None else None

    @property
    def times(self):
        """Return a list of all the times of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.time>

        Returns
        --------
        * (list) a list of all times for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('time')

    @property
    def feature(self):
        """Return the value for feature if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.features>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.feature>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._feature

    @property
    def features(self):
        """Return a list of all this features of teh Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.feature>

        Returns
        --------
        * (list) a list of all features for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('feature', include_default=False)


    # @property
    # def properties(self):
    #     """Return a list of all the properties of the Parameters.
    #
    #     :return: list of strings
    #     """
    #     return self.to_dict(field='feature').keys()

    @property
    def component(self):
        """Return the value for component if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.components>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.component>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._component

    @property
    def components(self):
        """Return a list of all the components of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.component>

        Returns
        --------
        * (list) a list of all components for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('component', include_default=False)

    @property
    def dataset(self):
        """Return the value for dataset if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.datasets>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.dataset>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._dataset

    @property
    def datasets(self):
        """Return a list of all the datasets of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.dataset>

        Returns
        --------
        * (list) a list of all datasets for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('dataset', include_default=False)

    @property
    def constraint(self):
        """Return the value for constraint if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.constraints>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.constraint>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._constraint

    @property
    def constraints(self):
        """Return a list of all the constraints of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.constraint>

        Returns
        --------
        * (list) a list of all constraints for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('constraint')

    @property
    def distribution(self):
        """Return the value for distribution if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.distributions>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.distribution>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._distribution

    @property
    def distributions(self):
        """Return a list of all the distributions of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.distribution>

        Returns
        --------
        * (list) a list of all constraints for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('distribution')

    @property
    def compute(self):
        """Return the value for compute if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.computes>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.compute>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._compute

    @property
    def computes(self):
        """Return a list of all the computes of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.compute>

        Returns
        --------
        * (list) a list of all computes for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('compute')

    @property
    def model(self):
        """Return the value for model if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.models>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.model>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._model

    @property
    def models(self):
        """Return a list of all the models of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.model>

        Returns
        --------
        * (list) a list of all models for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('model')

    @property
    def figure(self):
        """Return the value for figure if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.figures>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.figure>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._figure

    @property
    def figures(self):
        """Return a list of all the figures of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.figure>

        Returns
        --------
        * (list) a list of all figures for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('figure')

    @property
    def solver(self):
        """Return the value for solver if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.solvers>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.solver>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._solver

    @property
    def solvers(self):
        """Return a list of all the solvers of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.solver>

        Returns
        --------
        * (list) a list of all solvers for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('solver')

    @property
    def solution(self):
        """Return the value for solution if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.solutions>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.solution>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._solution

    @property
    def solutions(self):
        """Return a list of all the solutions of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.solution>

        Returns
        --------
        * (list) a list of all solutions for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('solution')

    @property
    def kind(self):
        """Return the value for kind if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.kinds>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.kind>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._kind

    @property
    def kinds(self):
        """Return a list of all the kinds of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.kind>

        Returns
        --------
        * (list) a list of all kinds for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('kind')

    @property
    def context(self):
        """Return the value for context if shared by ALL Parameters.

        If the value is not shared by ALL, then None will be returned.  To see
        all the qualifiers of all parameters, see <phoebe.parameters.ParameterSet.contexts>.

        To see the value of a single <phoebe.parameters.Parameter> object, see
        <phoebe.parameters.Parameter.context>.

        Returns
        --------
        (string or None) the value if shared by ALL <phoebe.parameters.Parameter>
            objects in the <phoebe.parmaters.ParameterSet>, otherwise None
        """
        return self._context

    @property
    def contexts(self):
        """Return a list of all the contexts of the Parameters.

        See also:
        * <phoebe.parameters.ParameterSet.tags>

        For the singular version, see:
        * <phoebe.parameters.ParameterSet.context>

        Returns
        --------
        * (list) a list of all contexts for each <phoebe.parameters.Parameter>
            in this <phoebe.parameters.ParameterSet>
        """
        return self._options_for_tag('context')

    def _set_meta(self):
        """
        set the meta fields of the ParameterSet as those that are shared
        by ALL parameters in the ParameterSet.  For any fields that are
        not
        """
        # we want to set meta-fields that are shared by ALL params in the PS
        for field in _meta_fields_twig:
            keys_for_this_field = self._options_for_tag(field)

            if len(keys_for_this_field)==1:
                setattr(self, '_'+field, keys_for_this_field[0])
            else:
                setattr(self, '_'+field, None)

    def _uniquetags(self, param_or_twig, force_levels=['qualifier'], exclude_levels=[]):
        """
        get the least unique twig for the parameter given by twig that
        will return this single result for THIS PS

        :parameter str twig: a twig that will return a single Parameter from
                THIS PS
        :parameter list force_levels: (optional) a list of "levels"
            (eg. context) that should be included whether or not they are
            necessary.  'context' will be appended unless `incl_context` is disabled.
        :return: the unique twig
        :rtype: str
        """
        for_this_param = param_or_twig if isinstance(param_or_twig, Parameter) else self.get_parameter(twig, check_default=False, check_visible=False)

        metawargs = {}

        # NOTE: self.contexts is INCREDIBLY expensive
        # if len(self.contexts) and 'context' not in force_levels:
        if 'context' not in force_levels:
            # then let's force context to be included
            force_levels.append('context')

        for k in force_levels:
            if k in exclude_levels:
                continue
            metawargs[k] = getattr(for_this_param, k)

        prev_count = len(self)
        # just to fake in case no metawargs are passed at all
        ps_for_this_search = []
        for k in _meta_fields_twig:
            if k in exclude_levels or k in force_levels:
                continue

            metawargs[k] = getattr(for_this_param, k)
            if getattr(for_this_param, k) is None:
                continue

            ps_for_this_search = self.filter(check_visible=False, **metawargs)

            if len(ps_for_this_search) < prev_count and k not in force_levels:
                prev_count = len(ps_for_this_search)
            elif k not in force_levels:
                # this didn't help us
                metawargs[k] = None

        if len(ps_for_this_search) != 1:
            # TODO: after fixing regex in twig (t0type vs t0)
            # change this to raise Error instead of return
            return {k:v for k,v in for_this_param.tags.items() if len(k)}

        # now we go in the other direction and try to remove each to make sure
        # the count goes up
        for k in _meta_fields_twig:
            if metawargs.get(k, None) is None or k in force_levels:
                continue

            ps_for_this_search = self.filter(check_visible=False,
                                             **{ki: metawargs[ki]
                                                for ki in _meta_fields_twig
                                                if ki != k and ki in metawargs.keys()})

            if len(ps_for_this_search) == 1:
                # then we didn't need to use this tag
                metawargs[k] = None

        # and lastly, we make sure that the tag corresponding to the context
        # is present
        context = for_this_param.context
        if hasattr(for_this_param, context):
            metawargs[context] = getattr(for_this_param, context)

        return {k:v for k,v in metawargs.items() if v is not None}

    def _uniquetwig(self, param_or_twig, force_levels=['qualifier'], exclude_levels=[]):
        """
        get the least unique twig for the parameter given by twig that
        will return this single result for THIS PS

        :parameter str twig: a twig that will return a single Parameter from
                THIS PS
        :parameter list force_levels: (optional) a list of "levels"
            (eg. context) that should be included whether or not they are
            necessary.  'context' will be appended unless `incl_context` is disabled.
        :return: the unique twig
        :rtype: str
        """
        metawargs = self._uniquetags(param_or_twig, force_levels=force_levels, exclude_levels=exclude_levels)


        return "@".join([metawargs[k]
                         for k in _meta_fields_twig
                         if metawargs.get(k, None) is not None])

    def _attach_params(self, params, check_copy_for=True, override_tags=False, new_uniqueids=False, overwrite=False, return_changes=False, **kwargs):
        """Attach a list of parameters (or ParameterSet) to this ParameterSet.

        :parameter list params: list of parameters, or ParameterSet
        :parameter **kwargs: attributes to set for each parameter (ie tags)
        """
        lst = params.to_list() if isinstance(params, ParameterSet) else params
        ps = params if isinstance(params, ParameterSet) else ParameterSet(params)
        ret_changes = []
        for param in lst:
            param._bundle = self

            if new_uniqueids:
                param._uniqueid = _uniqueid()

            for k, v in kwargs.items():
                # Here we'll set the attributes (_context, _qualifier, etc)
                if k in ['check_default', 'check_visible']: continue
                if getattr(param, '_{}'.format(k)) is None or override_tags:
                    setattr(param, '_{}'.format(k), v)

            if overwrite:
                ret_changes += self.remove_parameters_all(**{k:v for k,v in param.meta.items() if k not in ['uniqueid', 'twig', 'uniquetwig']}).to_list()
            self._params.append(param)

        if check_copy_for:
            self._check_copy_for()

        if ret_changes:
            return ParameterSet(ret_changes)
        return []

    def _check_copy_for(self):
        """Check the value of copy_for and make appropriate copies."""
        if not self._bundle:
            return

        # read the following at your own risk - I just wrote it and it still
        # confuses me and baffles me that it works
        pss = {}
        for param in self.to_list():
            if param.copy_for:
                # copy_for tells us how to filter and what set of attributes
                # needs a copy of this parameter
                #
                # copy_for = {'kind': ['star', 'disk', 'custombody'], 'component': '*'}
                # means that this should exist for each component (since that has a wildcard) which
                # has a kind in [star, disk, custombody]
                #
                # copy_for = {'kind': ['rv'], 'component': '*', 'dataset': '*'}
                # or
                # copy_for = {'component': {}, 'dataset': {'kind': 'rv'}}
                # means that this should exist for each component/dataset pair with the
                # rv kind
                #
                # copy_for = {'component': {'kind': 'star'}, 'dataset': {'kind': 'rv'}}
                # means that this should exist for each component/dataset pair
                # in which the component has kind='star' and dataset has kind='rv'


                attrs = [k for k,v in param.copy_for.items() if '*' in v or isinstance(v, dict)]
                # attrs is a list of the attributes for which we need a copy of
                # this parameter for any pair

                def force_list(v):
                    if isinstance(v, list):
                        return v
                    elif v=='*':
                        return v
                    else:
                        return [v]

                filter_ = {}
                for k,v in param.copy_for.items():
                    if isinstance(v,dict):
                        for dk,dv in v.items():
                            if dk in filter_.keys():
                                filter_[dk] += force_list(dv)
                            else:
                                filter_[dk] = force_list(dv)
                    else:
                        filter_[k] = force_list(v)

                # making this filter call repeatedly is expensive, so since
                # we're filtering for the same thing multiple times, let's
                # cache the filter by the json string of the filter dictionary
                filter_json = json.dumps(filter_)
                if filter_json in pss.keys():
                    ps = pss.get(filter_json)
                else:
                    ps = self.filter(check_visible=False,
                                     check_default=False,
                                     check_advanced=False,
                                     check_single=False,
                                     force_ps=True, **filter_)
                    pss[filter_json] = ps

                metawargs = {k:v for k,v in ps.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig']).items() if v is not None and k in attrs}
                # print("*** check_copy_for {} attrs={} filter_={}, metawargs={}".format(param.copy_for, attrs, filter_, metawargs))

                for k,v in param.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig']).items():
                    if k not in attrs:
                        metawargs[k] = v
                # metawargs is a list of the shared tags that will be used to filter for
                # existing parameters so that we know whether they already exist or
                # still need to be created

                # logger.debug("_check_copy_for {}: attrs={}".format(param.twig, attrs))
                for attrvalues in itertools.product(*(getattr(ps, '{}s'.format(attr)) for attr in attrs)):
                    # logger.debug("_check_copy_for {}: attrvalues={}".format(param.twig, attrvalues))
                    # for each attrs[i] (ie component), attrvalues[i] (star01)
                    # we need to look for this parameter, and if it does not exist
                    # then create it by copying param

                    valid = True

                    for attr, attrvalue in zip(attrs, attrvalues):
                        #if attrvalue=='_default' and not getattr(param, attr):
                        #    print "SKIPPING", attr, attrvalue
                        #    continue

                        # make sure valid from the copy_for dictionary
                        if isinstance(param.copy_for[attr], dict):
                            filter_ = {k:v for k,v in param.copy_for[attr].items()}
                            filter_[attr] = attrvalue
                            if not len(ps.filter(check_visible=False, check_default=False, check_advanced=False, check_single=False, force_ps=True, **filter_)):
                                valid = False

                        metawargs[attr] = attrvalue

                    # logger.debug("_check_copy_for {}: metawargs={}".format(param.twig, metawargs))
                    if valid and not len(self._bundle.filter(check_visible=False, check_default=False, **metawargs)):
                        # then we need to make a new copy
                        logger.debug("copying '{}' parameter for {}".format(param.qualifier, {attr: attrvalue for attr, attrvalue in zip(attrs, attrvalues)}))

                        newparam = param.copy()

                        for attr, attrvalue in zip(attrs, attrvalues):
                            setattr(newparam, '_{}'.format(attr), attrvalue)

                        newparam._copy_for = False
                        if newparam._visible_if and newparam._visible_if.lower() == 'false':
                            newparam._visible_if = None
                        newparam._bundle = self._bundle

                        self._params.append(newparam)


                    # Now we need to handle copying constraints.  This can't be
                    # in the previous if statement because the parameters can be
                    # copied before constraints are ever attached.
                    if valid and hasattr(param, 'is_constraint') and param.is_constraint:

                        param_constraint = param.is_constraint

                        copied_param = self._bundle.get_parameter(check_visible=False,
                                                                  check_default=False,
                                                                  check_advanced=False,
                                                                  check_single=False,
                                                                  **metawargs)

                        if not copied_param.is_constraint:
                            constraint_kwargs = param_constraint.constraint_kwargs.copy()
                            for attr, attrvalue in zip(attrs, attrvalues):
                                if attr in constraint_kwargs.keys():
                                    constraint_kwargs[attr] = attrvalue

                            logger.debug("copying constraint '{}' parameter for {}".format(param_constraint.constraint_func, {attr: attrvalue for attr, attrvalue in zip(attrs, attrvalues)}))
                            self.add_constraint(func=param_constraint.constraint_func, **constraint_kwargs)

        return

    def _check_label(self, label, allow_overwrite=False):
        """Check to see if the label is allowed."""

        if not isinstance(label, str):
            label = str(label)

        if label.lower() in _forbidden_labels:
            raise ValueError("'{}' is forbidden to be used as a label"
                             .format(label))
        if not re.match("^[a-z,A-Z,0-9,_]*$", label):
            raise ValueError("label '{}' is forbidden - only alphabetic, numeric, and '_' characters are allowed in labels".format(label))
        if len(self.filter(twig=label, check_visible=False)) and not allow_overwrite:
            raise ValueError("label '{}' is already in use.  Remove first or pass overwrite=True, if available.".format(label))
        if label[0] in ['_']:
            raise ValueError("first character of label is a forbidden character")


    def __add__(self, other):
        """Adding 2 PSs returns a new PS with items that are in either."""
        if isinstance(other, Parameter):
            other = ParameterSet([other])
        elif isinstance(other, list):
            other = ParameterSet(other)

        if isinstance(other, ParameterSet):
            # NOTE: used to have the following but doesn't work in python3
            # because the Parameters aren't hashable:
            # return ParameterSet(list(set(self._params+other._params)))
            lst = self._params
            for p in other._params:
                if p not in lst:
                    lst.append(p)

            ps = ParameterSet(lst)
            ps._bundle = self._bundle
            return ps
        else:
            raise NotImplementedError

    def __sub__(self, other):
        """Subtracting 2 PSs returns a new PS with items in the first but not second."""

        if isinstance(other, Parameter):
            other = ParameterSet([other])
        elif isinstance(other, list):
            other = ParameterSet(other)

        if isinstance(other, ParameterSet):
            ps = ParameterSet([p for p in self._params if p not in other._params])
            ps._bundle = self._bundle
            return ps
        else:
            raise NotImplementedError

    def __mul__(self, other):
        """
        multiplying 2 PSs should return a new PS with items that are in both (ie AND logic)

        this is the same as chaining filter calls
        """
        if isinstance(other, Parameter):
            other = ParameterSet([other])

        if isinstance(other, ParameterSet):
            ps = ParameterSet([p for p in self._params if p in other._params])
            ps._bundle = self._bundle
            return ps
        else:
            raise NotImplementedError

    @classmethod
    def open(cls, filename):
        """
        Open a ParameterSet from a JSON-formatted file.
        This is a constructor so should be called as:

        ```py
        ps = ParameterSet.open('test.json')
        ```

        See also:
        * <phoebe.parameters.Parameter.open>
        * <phoebe.frontend.bundle.Bundle.open>

        Arguments
        ---------
        * `filename` (string): relative or full path to the file.  Alternatively,
            this can be the json string itself or a list of dictionaries (the
            unpacked json).

        Returns
        ---------
        * an instantiated <phoebe.parameters.ParameterSet> object
        """
        if isinstance(filename, list):
            data = filename
        elif isinstance(filename, str) and "{" in filename:
            data = json.loads(filename)
        else:
            filename = os.path.expanduser(filename)
            with open(filename, 'r') as f:
                if _can_ujson:
                    # NOTE: this will not parse the unicode.  Bundle.open always calls
                    # json instead of ujson for this reason.
                    data = ujson.load(f)
                else:
                    data = json.load(f, object_pairs_hook=parse_json)

        return cls(data)

    def save(self, filename, incl_uniqueid=False, compact=False, sort_by_context=True):
        """
        Save the ParameterSet to a JSON-formatted ASCII file.

        See also:
        * <phoebe.parameters.Parameter.save>
        * <phoebe.frontend.bundle.Bundle.save>

        Arguments
        ----------
        * `filename` (string): relative or full path to the file
        * `incl_uniqueid` (bool, optional, default=False): whether to include
            uniqueids in the file (only needed if its necessary to maintain the
            uniqueids when reloading)
        * `compact` (bool, optional, default=False): whether to use compact
            file-formatting (may be quicker to save/load, but not as easily readable)

        Returns
        --------
        * (string) filename
        """
        filename = os.path.expanduser(filename)
        f = open(filename, 'w')
        if compact:
            if _can_ujson:
                ujson.dump(self.to_json(incl_uniqueid=incl_uniqueid, sort_by_context=sort_by_context),
                           f, sort_keys=False, indent=0)
            else:
                logger.warning("for faster compact saving, install ujson")
                json.dump(self.to_json(incl_uniqueid=incl_uniqueid, sort_by_context=sort_by_context),
                          f, sort_keys=False, indent=0)
        else:
            json.dump(self.to_json(incl_uniqueid=incl_uniqueid, sort_by_context=sort_by_context),
                      f, sort_keys=True, indent=0, separators=(',', ': '))
        f.close()

        return filename

    def _launch_ui(self, web_client, action, filter={}, full_ui=False, blocking=None):

        def filteritem(k, v):
            if isinstance(v, list):
                return v
            else:
                return [v]

        if len(filter.items()):
            querystr = "&".join(["{}={}".format(k, filteritem(k, v))
                                 for k, v in filter.items()])

        else:
            querystr = ""

        was_client = self._bundle.is_client

        # let's handle some defaults.
        # First determine if we're in jupyter, ipython, or python
        try:
            ipython_class = get_ipython().__class__.__name__
        except NameError:
            ipython_class = None

        if ipython_class == 'ZMQInteractiveShell':
            is_jupyter = True
        else:
            is_jupyter = False

        if is_jupyter:
            if self._bundle.is_client:
                querystr += "&disconnectButton=disconnect"
            else:
                querystr += "&disconnectButton=continue"

        # now if full_ui was passed to auto, then we want to default to False
        # for Jupyter, but to True if no filter otherwise
        if full_ui == 'auto':
            if is_jupyter:
                full_ui = False
            else:
                full_ui = len(self._filter.items()) == 0

        if len(self._filter.items()) or not full_ui:
            # then we want to make sure we're not filtering out things that should be in the filter
            # but if we're launching the full_ui UNFILTERED, then default to as if it were opened
            querystr += "&advanced=['is_advanced','is_single','is_constraint']"

        # and now disable the passed action from the parent method if full_ui
        # is (still) true
        action = action if not full_ui else None

        if web_client is None:
            web_client = self._bundle.get_value(qualifier='web_client', context='setting', default=False, **_skip_filter_checks)
        if web_client is True:
            web_client = self._bundle.get_value(qualifier='web_client_url', context='setting', default='ui.phoebe-project.org', **_skip_filter_checks)

        # TODO: expose options for advanced filters (or include everything by default)

        if web_client or (is_jupyter and not full_ui):
            if not web_client:
                # then we want to launch the UI inline.  We can't do this from
                # the installed client directly, BUT since the server is on
                # localhost, it will be serving the static web-version of the
                # desktop-client to self._bundle.is_client/ui
                if not self._bundle.is_client:
                    blocking = blocking if blocking is not None else True
                    if blocking:
                        logger.info("(temporarily) entering client mode")
                    else:
                        logger.info("entering client mode for non-blocking UI.  Must manually call b.as_client(False) to exit client mode.")
                    self._bundle.as_client()
                else:
                    blocking = blocking if blocking is not None else False

                web_client = self._bundle.is_client
                is_static_file = True

            else:
                is_static_file = False
                blocking = blocking if blocking is not None else False

                # then we must be in client mode already, if not, we'll raise an error
                # note that we do allow not in client-mode for the case of jupyter
                # as we will launch the server on localhost and manage it
                if not self._bundle.is_client:
                    # TODO: could allow passing as_client to PS.ui() to launch in one line...
                    # self._bundle.as_client(as_client=as_client, bundleid=bundleid, wait_for_server=True, reconnection_attempts=3, blocking=False)
                    raise ValueError("bundle must be in client mode (see b.as_client) before launching a web_client.")

            if web_client is True:
                web_client = 'http://ui.phoebe-project.org'

            if 'http' not in web_client[:5]:
                web_client = 'http://'+web_client

            if action:
                url = "{}/{}/{}/{}?{}".format(web_client, self._bundle.is_client.strip("http://"), self._bundle._bundleid, action, querystr)
            else:
                url = "{}/{}/{}?{}".format(web_client, self._bundle.is_client.strip("http://"), self._bundle._bundleid, querystr)


            if is_jupyter:
                jui = JupyterUI(url)
                if blocking:
                    from IPython.display import display, HTML
                    display(HTML(jui._repr_html_()))

                    # first wait to make sure the jupyter client connects
                    while len(self._bundle._server_clients) < 2:
                        logger.debug("blocking: waiting for jupyter client to connect", self._bundle._server_clients)
                        time.sleep(0.1)

                    # now the server is connect to at least the jupyter client and python-client
                    while len(self._bundle._server_clients) > 1:
                        logger.debug("blocking: waiting for jupyter client to disconnect", self._bundle._server_clients)
                        time.sleep(0.5)

                    if not was_client:
                        logger.info("leaving client mode")
                        self._bundle.as_client(False)
                else:
                    return jui
            else:
                # the is_jupyter case will be
                logger.info("opening {} in browser".format(url))
                webbrowser.open(url)
                return url


        else:
            # then we'll attempt to launch the desktop app on this machine
            cmd = 'phoebe'
            if self._bundle.is_client:
                blocking = blocking if blocking is not None else False
            if not self._bundle.is_client:
                blocking = blocking if blocking is not None else True
                if blocking:
                    logger.info("(temporarily) entering client mode")
                else:
                    logger.info("entering client mode for non-blocking UI.  Must manually call b.as_client(False) to exit client mode.")
                self._bundle.as_client()

            # then we're attaching the UI to an already existing instance on an already running server
            cmd += ' -s {} -b {}'.format(self._bundle.is_client.strip('http://'), self._bundle._bundleid)
            cmd += ' --skip-child-server'
            cmd += ' --disable-bundle-change'

            if querystr:
                cmd += ' -f \"{}\"'.format(querystr.replace(' ', ''))

            if action:
                cmd += ' -a {}'.format(action)

            # then we want to launch the UI in a separate thread
            cmd += ' --noWarnOnClose'

            if not blocking:
                cmd += ' &'

            logger.info("system call: "+cmd)
            # NOTE: this will block if not blocking
            os.system(cmd)

            if not was_client and blocking:
                logger.info("leaving client mode")
                self._bundle.as_client(False)


    def ui(self, web_client=None, full_ui=None, blocking=None, **kwargs):
        """
        Open an interactive user-interface for the ParameterSet.

        If the bundle is in client mode (see <phoebe.frontend.bundle.Bundle.is_client>
        and <phoebe.frontend.bundle.Bundle.as_client>) then the UI will open
        asynchronously or non-blocking (allowing you to interact from
        both python and the UI simultaneously).  Otherwise the UI will open
        synchronously by blocking the thread until the UI is closed and the
        bundle will continue outside of client mode.  To override this default
        behavior, see `blocking`. Note that if the bundle is not currently in
        client-mode but `blocking` is manually set to False, then the bundle
        will remain in client mode until manually passing `False` to
        <phoebe.frontend.bundle.Bundle.as_client>

        If the UI is not installed, pass a URL to `web_client`
        (ie. http://ui.phoebe-project.org) to launch the web-client in the
        default system browser. Note that this requires the bundle to already be in client mode.
        Call <phoebe.frontend.bundle.Bundle.as_client> first to use `web_client`.

        In Jupyter notebooks, the UI will be shown in-line with a button to disconnect
        that instance of the client (if not blocking) or to disconnect and "continue"
        the notebook execution if blocking.

        To more information or to install the desktop-client, see
        http://phoebe-project.org/clients

        See also:
        * <phoebe.frontend.bundle.Bundle.ui_figures>
        * <phoebe.frontend.bundle.Bundle.from_server>
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
        * `full_ui` (bool, optional, default=None): whether to show the entire
            bundle or just the filtered ParameterSet.  If not provided, will
            default to True if acting on the Bundle, or False if acting on
            a filtered ParameterSet.  If in Jupyter, will default to False
            always, and override to True will launch the full client (not in-line)
        * `blocking` (bool, optional, default=None): whether the clal to the
            UI should be blocking (wait for the client to close/disconnect)
            before continuing the python-thread or not.  If not provided or
            None, will default to True if not currently in client-mode
            (see <phoebe.frontend.bundle.Bundle.is_client> and
            <phoebe.frontend.bundle.Bundle.as_client>) or False otherwise.
        * `**kwargs`: additional kwargs will be sent to
            <phoebe.parameters.ParameterSet.filter>.

        Returns
        ----------
        * if `web_client`: `url` (string): the opened URL (will attempt to launch in the system
            webbrowser)

        Raises
        -----------
        * ValueError: if the <phoebe.parameters.ParameterSet> is not attached
            to a parent <phoebe.frontend.bundle.Bundle>.
        * ValueError: if `web_client` is provided but the <phoebe.frontend.bundle.Bundle>
            is not in client mode (see <phoebe.frontend.bundle.Bundle.is_client>
            and <phoebe.frontend.bundle.Bundle.as_client>)
        """
        if self._bundle is None:
            raise ValueError("cannot call ui on a ParameterSet not attached to a Bundle")


        if len(kwargs):
            return self.filter(**kwargs).ui(web_client=web_client, full_ui=full_ui, blocking=blocking)

        if full_ui is None:
            # default to True for the full bundle, False for a filtered PS
            full_ui = 'auto'

        action = 'ps'
        return self._launch_ui(web_client, action, self._filter, full_ui, blocking)

    def to_list(self, **kwargs):
        """
        Convert the <phoebe.parameters.ParameterSet> to a list of
        <phoebe.parameters.Parameter> objects.

        Arguments
        ---------
        * `**kwargs`: filter arguments sent to
            <phoebe.parameters.ParameterSet.filter>

        Returns
        --------
        * (list) list of <phoebe.parameter.Parameter> objects
        """
        if kwargs:
            return self.filter(**kwargs).to_list()
        return self._params

    def tolist(self, **kwargs):
        """
        Alias of <phoebe.parameters.ParameterSet.to_list>

        Arguments
        ---------
        * `**kwargs`: filter arguments sent to
            <phoebe.parameters.ParameterSet.filter>

        Returns
        --------
        * (list) list of <phoebe.parameter.Parameter> objects
        """
        return self.to_list(**kwargs)

    def to_list_of_dicts(self, **kwargs):
        """
        Convert the <phoebe.parameters.ParameterSet> to a list of the dictionary
        representation of each <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.Parameter.to_dict>

        Arguments
        ----------
        * `**kwargs`: filter arguments sent to
            <phoebe.parameters.ParameterSet.filter>

        Returns
        --------
        * (list of dicts) list of dictionaries, with each entry in the list
            representing a single <phoebe.parameters.Parameter> object converted
            to a dictionary via <phoebe.parameters.Parameter.to_dict>.
        """
        if kwargs:
            return self.filter(**kwargs).to_list_of_dicts()
        return [param.to_dict() for param in self._params]

    # @property
    # def __dict__(self):
    #     """Dictionary representation of a ParameterSet."""
    #     return self.to_dict()

    def to_flat_dict(self, **kwargs):
        """
        Convert the <phoebe.parameters.ParameterSet> to a flat dictionary, with
        keys being uniquetwigs to access the parameter and values being the
        <phoebe.parameters.Parameter> objects themselves.

        See also:
        * <phoebe.parameters.Parameter.uniquetwig>

        Arguments
        ----------
        * `**kwargs`: filter arguments sent to
            <phoebe.parameters.ParameterSet.filter>

        Returns
        --------
        * (dict) uniquetwig: Parameter pairs.
        """
        if kwargs:
            return self.filter(**kwargs).to_flat_dict()
        return {param.uniquetwig: param for param in self._params}

    def to_dict(self, field=None, include_none=False, **kwargs):
        """
        Convert the <phoebe.parameters.ParameterSet> to a structured (nested)
        dictionary to allow traversing the structure from the bottom up.

        See also:
        * <phoebe.parameters.ParameterSet.to_json>
        * <phoebe.parameters.ParameterSet.keys>
        * <phoebe.parameters.ParameterSet.values>
        * <phoebe.parameters.ParameterSet.items>

        Arguments
        ----------
        * `field` (string, optional, default=None): build the dictionary with
            keys at a given level/field.  Can be any of the keys in
            <phoebe.parameters.ParameterSet.meta>.  If None, the keys will be
            the lowest level in which Parameters have different values.

        Returns
        ---------
        * (dict) dictionary of <phoebe.parameters.ParameterSet> or
            <phoebe.parameters.Parameter> objects.
        """
        # skip_return is used internally when we want to call this just to update
        # self._next field, but don't want to waste time on the actual dictionary
        # comprehension
        skip_return = kwargs.pop('skip_return', False)

        if kwargs:
            return self.filter(**kwargs).to_dict(field=field)

        if field is not None:
            keys_for_this_field = self._options_for_tag(field)
            if skip_return: return

            d =  {k: self.filter(check_visible=False, **{field: k}) for k in keys_for_this_field}
            if include_none:
                d_None = ParameterSet([p for p in self.to_list() if getattr(p, field) is None])
                if len(d_None):
                    d[None] = d_None

            return d

        # we want to find the first level (from the bottom) in which filtering
        # further would shorten the list (ie there are more than one unique
        # item for that field)

        # so let's go through the fields in reverse (context up to (but not
        # including) qualifier)
        for field in reversed(_meta_fields_twig[1:]):
            # then get the unique keys in this field among the params in this
            # PS
            keys_for_this_field = set([getattr(p, field)
                                       for p in self.to_list()
                                       if getattr(p, field) is not None])
            # and if there are more than one, then return a dictionary with
            # those keys and the ParameterSet of the matching items
            if len(keys_for_this_field) > 1:
                self._next_field = field
                if skip_return: return
                return {k: self.filter(check_visible=False, **{field: k})
                        for k in keys_for_this_field}

        # if we've survived, then we're at the bottom and only have times or
        # qualifier left
        if self.context in ['hierarchy']:
            self._next_field = 'qualifier'
            if skip_return: return
            return {param.qualifier: param for param in self._params}
        else:
            self._next_field = 'time'
            if skip_return: return
            return {param.time: param for param in self._params}

    def keys(self):
        """
        Return the keys from <phoebe.parameters.ParameterSet.to_dict>

        Returns
        ---------
        * (list) list of strings
        """
        return list(self.to_dict().keys())

    def values(self):
        """
        Return the values from <phoebe.parameters.ParameterSet.to_dict>

        Returns
        -------
        * (list) list of <phoebe.paramters.ParameterSet> or
            <phoebe.parameters.Parameter> objects.
        """
        return self.to_dict().values()

    def items(self):
        """
        Returns the items (key, value pairs) from
        <phoebe.parameters.ParameterSet.to_dict>.

        :return: string, :class:`Parameter` or :class:`ParameterSet` pairs
        """
        return self.to_dict().items()

    def set(self, key, value, **kwargs):
        """
        Set the value of a <phoebe.parameters.Parameter> in the
        <phoebe.parameters.ParameterSet>.

        If <phoebe.parameters.ParameterSet.get> with the same value for
        `key`/`twig` and `**kwargs` would retrieve a single Parameter,
        this will set the value of that parameter.

        Or you can provide 'value@...' or 'default_unit@...', etc
        to specify what attribute to set.

        Arguments
        -----------
        * `key` (string): the twig (called key here to be analagous to a python
            dictionary) used for filtering.
        * `value` (valid value for the matching Parameter): value to set
        * `**kwargs`: other filter parameters

        Returns
        --------
        * (float/array/string/etc): the value of the <phoebe.parameters.Parameter>
            after setting the value (including converting units if applicable).
        """
        twig = key

        method = None
        mkwargs = {}
        if 'index' in kwargs.keys():
            mkwargs['index'] = kwargs.pop('index')
        twig, index = _extract_index_from_string(twig)
        twigsplit = re.findall(r"[\w']+", twig)
        if twigsplit[0] == 'value':
            twig = '@'.join(twigsplit[1:])
            if index is not None:
                method = 'set_index_value'
                mkwargs = {'index': index}
            else:
                method = 'set_value'
            mkwargs['value'] = value

        elif twigsplit[0] == 'quantity':
            twig = '@'.join(twigsplit[1:])
            if index is not None:
                raise ValueError("index notation not supported for {}".format(twigsplit[0]))
                # method = 'set_index_quantity'
                # mkwargs = {'index': index}
            else:
                method = 'set_quantity'

            mkwargs['value'] = value


        elif twigsplit[0] in ['unit', 'default_unit']:
            twig = '@'.join(twigsplit[1:])
            if index is not None:
                raise ValueError("index notation not supported for {}".format(twigsplit[0]))
            method = 'set_default_unit'

            mkwargs['unit'] = value

        elif twigsplit[0] in ['timederiv']:
            twig = '@'.join(twigsplit[1:])
            if index is not None:
                raise ValueError("index notation not supported for {}".format(twigsplit[0]))
            method = 'set_timederiv'
        elif twigsplit[0] in ['description']:
            raise KeyError("cannot set {} of {}".format(twigsplit[0], '@'.join(twigsplit[1:])))


        if self._bundle is not None and self._bundle.get_setting('dict_set_all').get_value() and len(self.filter(twig=twig, **kwargs)) > 1:
            # then we need to loop through all the returned parameters and call set on them
            for param in self.filter(twig=twig, **kwargs).to_list():
                self.set('{}@{}'.format(method, param.twig) if method is not None else param.twig, **mkwargs)
        else:

            if method is None:
                return self.set_value(twig=twig, value=value, **mkwargs)
            else:
                param = self.get_parameter(twig=twig, **kwargs)

                return getattr(param, method)(**mkwargs)

    def __getitem__(self, key):
        """
        """
        if self._bundle is not None:
            kwargs = self._bundle.get_value(qualifier='dict_filter',
                                            context='setting',
                                            default={})
        else:
            kwargs = {}

        if isinstance(key, int):
            return self.filter(**kwargs).to_list()[key]

        return self.filter_or_get(twig=key, **kwargs)

    def __setitem__(self, twig, value):
        """
        """
        if self._bundle is not None:
            kwargs = self._bundle.get_setting('dict_filter').get_value()
        else:
            kwargs = {}

        self.set(twig, value, allow_value_as_first_arg=False, **kwargs)

    def __contains__(self, twig):
        """
        """
        # may not be an exact match with to_dict().keys()
        if isinstance(twig, Parameter):
            return len(self.filter(uniqueid=twig.uniqueid))
        elif isinstance(twig, str):
            return len(self.filter(twig=twig))
        else:
            raise NotImplementedError("cannot check contains on type {}".format(type(twig)))

    def __len__(self):
        """
        """
        return len(self._params)

    def __iter__(self):
        """
        """
        return iter(self.to_dict())

    def to_json(self, incl_uniqueid=False, incl_none=False, exclude=[], sort_by_context=True):
        """
        Convert the <phoebe.parameters.ParameterSet> to a json-compatible
        object.

        The resulting object will be a list, with one entry per-Parameter
        being the json representation of that Parameter from
        <phoebe.parameters.Parameter.to_json>.

        See also:
        * <phoebe.parameters.Parameter.to_json>
        * <phoebe.parameters.ParameterSet.to_dict>
        * <phoebe.parameters.ParameterSet.save>

        Arguments
        --------
        * `incl_uniqueid` (bool, optional, default=False): whether to include
            uniqueids in the file (only needed if its necessary to maintain the
            uniqueids when reloading)
        * `incl_none` (bool, optional, default=False): whether to include tags
            whose values are None.
        * `exclude` (list, optional, default=[]): tags to exclude when saving.

        Returns
        -----------
        * (list of dicts)
        """
        lst = []
        if sort_by_context:
            for context in _contexts:
                lst += [v.to_json(incl_uniqueid=incl_uniqueid, incl_none=incl_none, exclude=exclude)
                        for v in self.filter(context=context,
                                             check_visible=False,
                                             check_default=False).to_list()]
        else:
            lst = [v.to_json(incl_uniqueid=incl_uniqueid, exclude=exclude) for v in self.to_list()]
        return lst
        # return {k: v.to_json() for k,v in self.to_flat_dict().items()}

    def export_arrays(self, fname,
                      delimiter=' ',header='', footer='',
                      comments='# ', encoding=None,
                      **kwargs):
        """
        Export arrays from <phoebe.parameters.Parameter.FloatArrayParameter>
        parameters to a file via `np.savetxt`.

        Each parameter will have its array values as a column in the output
        file in a format that can be reloaded manually with `np.loadtxt`.

        Note: all parameters must be FloatArrayParameters and have the same
        shape.


        Arguments
        ------------
        * `fname` (string or file object): passed to np.savetxt.
            If the filename ends in .gz, the file is automatically saved in
            compressed gzip format. loadtxt understands gzipped files
            transparently.
        * `delimiter` (string, optional, default=' '): passed to np.savetxt.
            String or character separating columns.
        * `header` (string, optional): The header will automatically be appended
            with the twigs of the parameters making up the columns and then
            passed to np.savetxt.
            String that will be written at the beginning of the file.
        * `footer` (string, optional): passed to np.savetxt.
            String that will be written at the end of the file.
        * `comments` (string, optional, default='#'): passed to np.savetxt.
            String that will be prepended to the `header` and `footer` strings,
            to mark them as comments.
        * `encoding` (None or string, optional, default=None): passed to np.savetxt.
            Encoding used to encode the outputfile. Does not apply to output
            streams. If the encoding is something other than ‘bytes’ or ‘latin1’
            you will not be able to load the file in NumPy versions < 1.14.
            Default is ‘latin1’.
        * `**kwargs`: all additional keyword arguments will be sent to
            <phoebe.parameters.ParameterSet.filter>.  The filter must result
            in all <phoebe.parameters.Parameter.FloatArrayParameter> objects
            with the same length, otherwise an error will be raised.


        Returns
        -----------
        * (string or file object) `fname`

        Raises
        -----------
        * TypeError: if not all parameters are of type
            <phoebe.parameters.Parameter.FloatArrayParameter> or no parameters
            are included in the filter.
        """
        if len(kwargs):
            return self.filter(**kwargs).export_arrays(fname)

        if not len(self.to_list()):
            raise TypeError("no parameters to be exported")

        for param in self.to_list():
            if param.__class__.__name__ != 'FloatArrayParameter':
                raise TypeError("all parameters must be of type FloatArrayParameter")

        X = np.array([param.get_value() for param in self.to_list()]).T

        header += delimiter.join([param.uniquetwig for param in self.to_list()])

        np.savetxt(fname, X, delimiter=delimiter,
                   header=header, footer=footer, comments=comments,
                   encoding=encoding)

        return fname


    def filter(self, twig=None, check_visible=True, check_default=True,
               check_advanced=False, check_single=False, **kwargs):
        """
        Filter the <phoebe.parameters.ParameterSet> based on the meta-tags of the
        children <phoebe.parameters.Parameter> objects and return another
        <phoebe.parameters.ParameterSet>.

        Because another ParameterSet is returned, these filter calls are
        chainable.

        ```py
        b.filter(context='component').filter(component='starA')
        ```

        * `check_advanced` (bool, optional, default=False): whether to exclude parameters which
            are considered "advanced".
        * `check_single` (bool, optional, default=False): whether to exclude ChoiceParameters
            with only a single choice.
        See also:
        * <phoebe.parameters.ParameterSet.filter_or_get>
        * <phoebe.parameters.ParameterSet.exclude>
        * <phoebe.parameters.ParameterSet.get>
        * <phoebe.parameters.ParameterSet.get_parameter>
        * <phoebe.parameters.ParameterSet.get_or_create>

        Arguments
        -----------
        * `twig` (str, optional, default=None): the search twig - essentially a single
            string with any delimiter (ie '@') that will be parsed
            into any of the meta-tags.  Example: instead of
            `b.filter(context='component', component='starA')`, you
            could do `b.filter('starA@component')`.
        * `check_visible` (bool, optional, default=True): whether to hide invisible
            parameters.  These are usually parameters that do not
            play a role unless the value of another parameter meets
            some condition.
        * `check_default` (bool, optional, default=True): whether to exclude parameters which
            have a _default tag (these are parameters which solely exist
            to provide defaults for when new parameters or datasets are
            added and the parameter needs to be copied appropriately).
        * `check_advanced` (bool, optional, default=False): whether to exclude parameters which
            are considered "advanced".
        * `check_single` (bool, optional, default=False): whether to exclude ChoiceParameters
            with only a single choice.
        * `**kwargs`:  meta-tags to search (ie. 'context', 'component',
            'model', etc).  See <phoebe.parameters.ParameterSet.meta>
            for all possible options.

        Returns
        ----------
        * the resulting <phoebe.parameters.ParameterSet>.
        """
        kwargs['check_visible'] = check_visible
        kwargs['check_default'] = check_default
        kwargs['check_advanced'] = check_advanced
        kwargs['check_single'] = check_single
        kwargs['force_ps'] = True
        return self.filter_or_get(twig=twig, **kwargs)

    def get(self, twig=None, check_visible=True, check_default=True,
            check_advanced=False, check_single=False, **kwargs):
        """
        Get a single <phoebe.parameters.Parameter> from this
        <phoebe.parameters.ParameterSet>.  This works exactly the
        same as <phoebe.parameters.ParameterSet.filter> except there must be only
        a single result, and the Parameter itself is returned instead of a
        ParameterSet.

        This is identical to <phoebe.parameters.ParameterSet.get_parameter>

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.parameters.ParameterSet.filter_or_get>
        * <phoebe.parameters.ParameterSet.exclude>
        * <phoebe.parameters.ParameterSet.get_or_create>

        Arguments
        -----------
        * `twig` (str, optional, default=None): the search twig - essentially a single
            string with any delimiter (ie '@') that will be parsed
            into any of the meta-tags.  Example: instead of
            `b.filter(context='component', component='starA')`, you
            could do `b.filter('starA@component')`.
        * `check_visible` (bool, optional, default=True): whether to hide invisible
            parameters.  These are usually parameters that do not
            play a role unless the value of another parameter meets
            some condition.
        * `check_default` (bool, optional, default=True): whether to exclude parameters which
            have a _default tag (these are parameters which solely exist
            to provide defaults for when new parameters or datasets are
            added and the parameter needs to be copied appropriately).
        * `check_advanced` (bool, optional, default=False): whether to exclude parameters which
            are considered "advanced".
        * `check_single` (bool, optional, default=False): whether to exclude ChoiceParameters
            with only a single choice.
        * `**kwargs`:  meta-tags to search (ie. 'context', 'component',
            'model', etc).  See <phoebe.parameters.ParameterSet.meta>
            for all possible options.

        Returns
        --------
        * the resulting <phoebe.parameters.Parameter>.

        Raises
        -------
        * ValueError: if either 0 or more than 1 results are found
            matching the search.
        """
        kwargs['check_visible'] = check_visible
        kwargs['check_default'] = check_default
        kwargs['check_advanced'] = check_advanced
        kwargs['check_single'] = check_single
        # print "***", kwargs
        ps = self.filter(twig=twig, **kwargs)
        if not len(ps):
            # TODO: custom exception?
            raise ValueError("0 results found for twig: '{}', {}".format(twig, kwargs))
        elif len(ps) != 1:
            # TODO: custom exception?
            raise ValueError("{} results found: {}".format(len(ps), ps.twigs))
        else:
            # then only 1 item, so return the parameter
            return ps._params[0]

    def filter_or_get(self, twig=None, autocomplete=False, force_ps=False,
                      check_visible=True, check_default=True,
                      check_advanced=False, check_single=False, **kwargs):
        """

        Filter the <phoebe.parameters.ParameterSet> based on the meta-tags of its
        Parameters and return another <phoebe.parameters.ParameterSet> unless there is
        exactly 1 result, in which case the <phoebe.parameters.Parameter> itself is
        returned (set `force_ps=True` to avoid this from happening or call
        <phoebe.parameters.ParameterSet.filter> instead).

        In the case when another <phoebe.parameters.ParameterSet> is returned, these
        calls are chainable.

        ```py
        b.filter_or_get(context='component').filter_or_get(component='starA')
        ```

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.parameters.ParameterSet.exclude>
        * <phoebe.parameters.ParameterSet.get>
        * <phoebe.parameters.ParameterSet.get_parameter>
        * <phoebe.parameters.ParameterSet.get_or_create>

        Arguments
        -----------
        * `twig` (str, optional, default=None): the search twig - essentially a single
            string with any delimiter (ie '@') that will be parsed
            into any of the meta-tags.  Example: instead of
            `b.filter(context='component', component='starA')`, you
            could do `b.filter('starA@component')`.
        * `check_visible` (bool, optional, default=True): whether to hide invisible
            parameters.  These are usually parameters that do not
            play a role unless the value of another parameter meets
            some condition.
        * `check_default` (bool, optional, default=True): whether to exclude parameters which
            have a _default tag (these are parameters which solely exist
            to provide defaults for when new parameters or datasets are
            added and the parameter needs to be copied appropriately).
        * `check_advanced` (bool, optional, default=False): whether to exclude parameters which
            are considered "advanced".
        * `check_single` (bool, optional, default=False): whether to exclude ChoiceParameters
            with only a single choice.
        * `force_ps` (bool, optional, default=False): whether to force a
            <phoebe.parameters.ParameterSet> to be returned, even if more than
            1 result (see also: <phoebe.parameters.ParameterSet.filter>)
        * `**kwargs`:  meta-tags to search (ie. 'context', 'component',
            'model', etc).  See <phoebe.parameters.ParameterSet.meta>
            for all possible options.

        Returns
        ----------
        * the resulting <phoebe.parameters.Parameter> object if the length
            of the results is exactly 1 and `force_ps=False`, otherwise the
            resulting <phoebe.parameters.ParameterSet>.
        """
        def _return(params, force_ps, method=None, index=None):
            if len(params) == 1 and not force_ps:
                # then just return the parameter itself
                if method is None:
                    return params[0]
                elif index is not None:
                    return getattr(params[0], method)()[index]
                else:
                    return getattr(params[0], method)()

            elif method is not None:
                raise ValueError("{} results found, could not call {}".format(len(params), method))

            # TODO: handle returning 0 results better

            ps = ParameterSet(params)
            ps._bundle = self._bundle
            ps._filter = self._filter.copy()
            for k, v in kwargs.items():
                if k in _meta_fields_filter:
                    ps._filter[k] = v
            if twig is not None and not isinstance(twig, list):
                # try to guess necessary additions to filter
                twigsplit = twig.split('@')
                for attr in _meta_fields_twig:
                    tag = getattr(ps, attr)
                    if tag in twigsplit:
                        ps._filter[attr] = tag
            return ps

        if self._bundle is None:
            # then override check_default to False - its annoying when building
            # a ParameterSet say by calling datasets.lc() and having half
            # of the Parameters hidden by this switch
            check_default = False

        if isinstance(twig, list):
            params = []
            for t in twig:
                params += self.filter(twig=t, check_visible=check_visible, check_default=check_default, check_advanced=check_advanced, check_single=check_single, **kwargs).to_list()
            return _return(params, force_ps)

        if not (twig is None or isinstance(twig, str)):
            raise TypeError("first argument (twig) must be of type str or None, got {}".format(type(twig)))

        if kwargs.get('component', None) == '_default' or\
                kwargs.get('dataset', None) == '_default' or\
                kwargs.get('uniqueid', None) is not None or\
                kwargs.get('uniquetwig', None) is not None:
            # then override the default for check_default and make sure to
            # return a result
            check_default = False

        if kwargs.get('uniqueid', None) is not None:
            check_visible = False
            kwargs['uniqueid'], _ = _extract_index_from_string(kwargs.get('uniqueid'))

        time = kwargs.get('time', None)
        if hasattr(time, '__iter__') and not isinstance(time, str):
            # then we should loop here rather than forcing complicated logic
            # below
            kwargs['twig'] = twig
            kwargs['autocomplete'] = autocomplete
            kwargs['force_ps'] = force_ps
            kwargs['check_visible'] = check_visible
            kwargs['check_default'] = check_default
            kwargs['check_advanced'] = check_advanced
            kwargs['check_single'] = check_single
            return_ = ParameterSet()
            for t in time:
                kwargs['time'] = t
                return_ += self.filter_or_get(**kwargs)
            return return_

        params = self.to_list()

        def string_to_time(string):
            try:
                return float(string)
            except ValueError:
                # allow for passing a twig that needs to resolve a float (ie. 't0_supconj')
                if self._bundle is None:
                    return self.get_value(string, context=['system', 'component'], check_default=False, check_visible=False)
                else:
                    return self._bundle.get_value(string, context=['system', 'component'], check_default=False, check_visible=False)

        # TODO: replace with key,value in kwargs.items()... unless there was
        # some reason that won't work?
        for key in kwargs.keys():
            if len(params) and \
                    key in _meta_fields_filter and \
                    kwargs[key] is not None:

                params = [pi for pi in params if (hasattr(pi,key) and getattr(pi,key) is not None or isinstance(kwargs[key], list) and None in kwargs[key]) and
                    (getattr(pi,key) is kwargs[key] or
                    (isinstance(kwargs[key],list) and getattr(pi,key) in kwargs[key]) or
                    (isinstance(kwargs[key],list) and np.any([_fnmatch(getattr(pi,key),keyi) for keyi in kwargs[key]])) or
                    (isinstance(kwargs[key],str) and isinstance(getattr(pi,key),str) and _fnmatch(getattr(pi,key),kwargs[key])) or
                    (key=='kind' and isinstance(kwargs[key],str) and getattr(pi,key).lower()==kwargs[key].lower()) or
                    (key=='kind' and hasattr(kwargs[key],'__iter__') and getattr(pi,key).lower() in [k.lower() for k in kwargs[key]]) or
                    (key=='time' and abs(float(getattr(pi,key))-string_to_time(kwargs[key]))<1e-6))]
                    #(key=='time' and abs(float(getattr(pi,key))-float(kwargs[key]))<=abs(np.array([p._time for p in params])-float(kwargs[key]))))]

        # handle hiding _default (cheaper than visible_if so let's do first)
        if check_default:
            params = [pi for pi in params if pi.component != '_default' and pi.dataset != '_default' and pi.feature != '_default']

        # handle visible_if
        if check_visible:
            params = [pi for pi in params if pi.is_visible]

        # handle hiding advanced parameters
        if check_advanced:
            params = [pi for pi in params if not pi.advanced]

        # handle hiding choice parameters with a single option
        if check_single:
            params = [pi for pi in params if not hasattr(pi, 'choices') or len(pi.choices) > 1]

        if isinstance(twig, int):
            # then act as a list index
            return params[twig]

        # TODO: handle isinstance(twig, float) as passing time?

        # TODO: smarter error if trying to slice on a PS instead of value (ie
        # b['value@blah'][::8] where b['value@blah'] is returning an empty PS
        # and ::8 is being passed here as twig)

        # now do twig matching
        method = None
        mindex = None
        if twig is not None:
            _user_twig = _deepcopy(twig)

            twig, index = _extract_index_from_string(twig)

            twigsplit = twig.split('@')
            if twigsplit[0] == 'value':
                twig = '@'.join(twigsplit[1:])
                method = 'get_value'
                mindex = index
            elif twigsplit[0] == 'quantity':
                twig = '@'.join(twigsplit[1:])
                method = 'get_quantity'
                mindex = index
            elif twigsplit[0] in ['unit', 'default_unit']:
                twig = '@'.join(twigsplit[1:])
                method = 'get_default_unit'
            elif twigsplit[0] in ['timederiv']:
                twig = '@'.join(twigsplit[1:])
                method = 'get_timederiv'
            elif twigsplit[0] == 'description':
                twig = '@'.join(twigsplit[1:])
                method = 'get_description'
            elif twigsplit[0] == 'choices':
                twig = '@'.join(twigsplit[1:])
                method = 'get_choices'
            elif twigsplit[0] == 'result':
                twig = '@'.join(twigsplit[1:])
                method = 'get_result'

            # twigsplit = re.findall(r"[\w']+", twig)
            twigsplit = twig.split('@')

            if autocomplete:
                # then we want to do matching based on all but the
                # last item in the twig and then try to autocomplete
                # based on the last item
                if re.findall(r"[^\w]", _user_twig[-1]):
                    # then we will autocomplete on an empty entry
                    twigautocomplete = ''
                else:
                    # the last item in twig is incomplete
                    twigautocomplete = twigsplit[-1]
                    twigsplit = twigsplit[:-1]

            def _twigmatch(twig, ti, index):
                if index is None:
                    return ti in twig.split('@')
                else:
                    return ti in twig.split('@') or '{}[{}]'.format(ti, index) in twig.split('@')

            for ti in twigsplit:
                # TODO: need to fix repeating twigs (ie
                # period@period@period@period still matches and causes problems
                # with the tabcomplete)
                params = [pi for pi in params if _twigmatch(pi.twig, ti, index) or _fnmatch(pi.twig, ti)]

            if autocomplete:
                # we want to provide options for what twigautomplete
                # could be to produce matches
                options = []
                for pi in params:
                    for twiglet in pi.twig.split('@'):
                        if twigautocomplete == twiglet[:len(twigautocomplete)]:
                            if len(twigautocomplete):
                                completed_twig = _user_twig.replace(twigautocomplete, twiglet)
                            else:
                                completed_twig = _user_twig + twiglet
                            if completed_twig not in options:
                                options.append(completed_twig)
                return options

        return _return(params, force_ps, method, mindex)

    def exclude(self, twig=None, check_visible=True, check_default=True, **kwargs):
        """
        Exclude the results from this filter from the current
        <phoebe.parameters.ParameterSet>.

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.parameters.ParameterSet.filter_or_get>
        * <phoebe.parameters.ParameterSet.get>
        * <phoebe.parameters.ParameterSet.get_parameter>
        * <phoebe.parameters.ParameterSet.get_or_create>

        Arguments
        -----------
        * `twig` (str, optional, default=None): the search twig - essentially a single
            string with any delimiter (ie '@') that will be parsed
            into any of the meta-tags.  Example: instead of
            `b.filter(context='component', component='starA')`, you
            could do `b.filter('starA@component')`.
        * `check_visible` (bool, optional, default=True): whether to hide invisible
            parameters.  These are usually parameters that do not
            play a role unless the value of another parameter meets
            some condition.
        * `check_default` (bool, optional, default=True): whether to exclude parameters which
            have a _default tag (these are parameters which solely exist
            to provide defaults for when new parameters or datasets are
            added and the parameter needs to be copied appropriately).
            Defaults to True.
        * `**kwargs`:  meta-tags to search (ie. 'context', 'component',
            'model', etc).  See <phoebe.parameters.ParameterSet.meta>
            for all possible options.

        Returns
        ----------
        * the resulting <phoebe.parameters.ParameterSet>.
        """
        return self - self.filter(twig=twig,
                                  check_visible=check_visible,
                                  check_default=check_default,
                                  **kwargs)

    def get_parameter(self, twig=None, **kwargs):
        """
        Get a <phoebe.parameters.Parameter> from this
        <phoebe.parameters.ParameterSet>.  This is identical to
        <phoebe.parameters.ParameterSet.get>.

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.parameters.ParameterSet.filter_or_get>
        * <phoebe.parameters.ParameterSet.exclude>
        * <phoebe.parameters.ParameterSet.get_or_create>

        Arguments
        -----------
        * `twig` (str, optional, default=None): the search twig - essentially a single
            string with any delimiter (ie '@') that will be parsed
            into any of the meta-tags.  Example: instead of
            `b.filter(context='component', component='starA')`, you
            could do `b.filter('starA@component')`.
        * `check_visible` (bool, optional, default=True): whether to hide invisible
            parameters.  These are usually parameters that do not
            play a role unless the value of another parameter meets
            some condition.
        * `check_default` (bool, optional, default=True): whether to exclude parameters which
            have a _default tag (these are parameters which solely exist
            to provide defaults for when new parameters or datasets are
            added and the parameter needs to be copied appropriately).
            Defaults to True.
        * `**kwargs`:  meta-tags to search (ie. 'context', 'component',
            'model', etc).  See <phoebe.parameters.ParameterSet.meta>
            for all possible options.

        Returns
        --------
        * the resulting <phoebe.parameters.Parameter>.

        Raises
        -------
        * ValueError: if either 0 or more than 1 results are found
            matching the search.
        """
        return self.get(twig=twig, **kwargs)

    def get_or_create(self, qualifier, new_parameter, attach_to_bundle=False, **kwargs):
        """
        Get a <phoebe.parameters.Parameter> from the
        <phoebe.parameters.ParameterSet>. If it does not exist,
        create and attach it.

        Note: running this on a ParameterSet that is NOT a
        <phoebe.frontend.bundle.Bundle>,
        will NOT add the Parameter to the Bundle, but only the temporary
        ParameterSet, unless `attach_to_bundle` is set to True and the bundle
        can be found.

        See also:
        * <phoebe.parameters.ParameterSet.filter>
        * <phoebe.parameters.ParameterSet.filter_or_get>
        * <phoebe.parameters.ParameterSet.exclude>
        * <phoebe.parameters.ParameterSet.get>
        * <phoebe.parameters.ParameterSet.get_parameter>

        Arguments
        ----------
        * `qualifier` (string): the qualifier of the Parameter.
            **NOTE**: this must be a qualifier, not a twig.
        * `new_parameter`: (<phoebe.parameters.Parameter>): the parameter to
            attach if no result is found.
        * `attach_to_bundle` (bool, optional, default=False): whether to attach
            the added parameter (if created) to the bundle.
        * `**kwargs`: meta-tags to use when filtering, including `check_visible` and
            `check_default`.  See <phoebe.parameters.ParameterSet.filter_or_get>.

        Returns
        ---------
        * (<phoebe.parameters.Parameter, bool): the Parameter object (either
            from filtering or newly created) and a boolean telling whether the
            Parameter was created or not.

        Raises
        -----------
        * ValueError: if more than 1 result was found using the filter criteria.
        """
        ps = self.filter_or_get(qualifier=qualifier, **kwargs)
        if isinstance(ps, Parameter):
            return ps, False
        elif len(ps):
            # TODO: custom exception?
            raise ValueError("more than 1 result was found")
        else:
            logger.debug("creating and attaching new parameter: {}".format(new_parameter.qualifier))

            if attach_to_bundle:
                self._bundle._attach_params(ParameterSet([new_parameter]), **kwargs)
                return self._bundle.get_parameter(uniqueid=new_parameter.uniqueid), True
            else:
                self._attach_params(ParameterSet([new_parameter]), **kwargs)
                return self.get_parameter(uniqueid=new_parameter.uniqueid), True

    def _remove_parameter(self, param):
        """
        Remove a Parameter from the ParameterSet

        :parameter param: the :class:`Parameter` object to be removed
        :type param: :class:`Parameter`
        """
        # TODO: check to see if protected (required by a current constraint or
        # by a backend)
        param._bundle = None
        self._params = [p for p in self._params if p.uniqueid != param.uniqueid]

    def remove_parameter(self, twig=None, **kwargs):
        """
        Remove a <phoebe.parameters.Parameter> from the
        <phoebe.parameters.ParameterSet>.

        Note: removing Parameters from a ParameterSet will not remove
        them from any parent ParameterSets
        (including the <phoebe.fontend.bundle.Bundle>).

        Arguments
        --------
        * `twig` (string, optional, default=None): the twig to search for the
            parameter (see <phoebe.parameters.ParameterSet.get>)
        * `**kwargs`: meta-tags to use when filtering, including `check_visible` and
            `check_default`.  See <phoebe.parameters.ParameterSet.get>.

        Returns
        -----------
        * the removed <phoebe.parameters.Parameter>.

        Raises
        ------
        * ValueError: if 0 or more than 1 results are found using the
                provided filter criteria.
        """
        param = self.get(twig=twig, **kwargs)

        self._remove_parameter(param)

    def remove_parameters_all(self, twig=None, **kwargs):
        """
        Remove all <phoebe.parameters.Parameter> objects that match the filter
        from the <phoebe.parameters.ParameterSet>.

        Any Parameter that would be included in the resulting ParameterSet
        from a <phoebe.parameters.ParameterSet.filter> call with the same
        arguments will be removed from this ParameterSet.

        Note: removing Parameters from a ParameterSet will not remove
        them from any parent ParameterSets
        (including the <phoebe.frontend.bundle.Bundle>)

        Arguments
        --------
        * `twig` (string, optional, default=None): the twig to search for the
            parameter (see <phoebe.parameters.ParameterSet.get>)
        * `**kwargs`: meta-tags to use when filtering, including `check_visible` and
            `check_default` which will all default to False if not provided.
            See <phoebe.parameters.ParameterSet.filter>.

        Returns
        -----------
        * ParameterSet of removed parameters
        """
        kwargs.setdefault('check_visible', False)
        kwargs.setdefault('check_default', False)
        kwargs.setdefault('check_single', False)
        kwargs.setdefault('check_advanced', False)
        params = self.filter(twig=twig, **kwargs)

        for param in params.to_list():
            param._bundle = None

        removed_ids = [p.uniqueid for p in params.to_list()]
        self._params = [p for p in self._params if p.uniqueid not in removed_ids]

        return params

    def get_quantity(self, twig=None, unit=None,
                     default=None, t=None, **kwargs):
        """
        Get the quantity of a <phoebe.parameters.Parameter> in this
        <phoebe.parameters.ParameterSet>.

        Note: this only works for Parameter objects with a `get_quantity` method.
        These include:
        * <phoebe.parameters.FloatParameter> (see <phoebe.parameters.FloatParameter.get_quantity>)
        * <phoebe.parameters.FloatArrayParameter>

        See also:
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameter.  See <phoebe.parameters.ParameterSet.get_parameter>.
        * `unit` (string or unit, optional, default=None): unit to convert the
            quantity.  If not provided or None, will use the default unit.
            'SI' or 'solar' are also allowed values which will then be determined
            based on the physical type of the default unit.  See
            <phoebe.parameters.ParameterSet.get_default_unit>.
        * `default` (quantity, optional, default=None): value to return if
            no results are returned by <phoebe.parameters.ParameterSet.get_parameter>
            given the value of `twig` and `**kwargs`.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter>.

        Returns
        --------
        * an astropy quantity object
        """
        # TODO: for time derivatives will need to use t instead of time (time
        # gets passed to twig filtering)

        if default is not None:
            # then we need to do a filter first to see if parameter exists
            if not len(self.filter(twig=twig, **kwargs)):
                return default

        param = self.get_parameter(twig=twig, **kwargs)

        if param.qualifier in kwargs.keys():
            # then we have an "override" value that was passed, and we should
            # just return that.
            # Example b.get_value('teff', teff=6000) returns 6000
            return kwargs.get(param.qualifier)

        return param.get_quantity(unit=unit, t=t)

    def set_quantity(self, twig=None, value=None, **kwargs):
        """
        Set the quantity of a <phoebe.parameters.Parameter> in the
        <phoebe.parameters.ParameterSet>.

        Note: this only works for Parameter objects with a `set_quantity` method.
        These include:
        * <phoebe.parameters.FloatParameter> (see <phoebe.parameters.FloatParameter>)
        * <phoebe.parameters.FloatArrayParameter>

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameter.  See <phoebe.parameters.ParameterSet.get_parameter>.
        * `value` (quantity, optional, default=None): quantity to set for the
            matched Parameter.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter> and `set_quantity`.

        Raises
        --------
        * ValueError: if a unique match could not be found via
            <phoebe.parameters.ParameterSet.get_parameter>
        """
        # TODO: handle twig having parameter key (value@, default_unit@, adjust@, etc)
        # TODO: does this return anything (update the docstring)?
        return self.get_parameter(twig=twig, **kwargs).set_quantity(value=value, **kwargs)

    def get_value(self, twig=None, unit=None, default=None, t=None, **kwargs):
        """
        Get the value of a <phoebe.parameters.Parameter> in this
        <phoebe.parameters.ParameterSet>.

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameter.  See <phoebe.parameters.ParameterSet.get_parameter>.
        * `unit` (string or unit, optional, default=None): unit to convert the
            value.  If not provided or None, will use the default unit.
            'SI' or 'solar' are also allowed values which will then be determined
            based on the physical type of the default unit. See
            <phoebe.parameters.ParameterSet.get_default_unit>. `unit` will
            be ignored for Parameters that do not store quantities.
        * `draw_from` (string, optional, default=None): distribution-tag to
            draw from (if applicable).  See <phoebe.parameters.FloatParameter.get_quantity>
            or <phoebe.parameters.FloatParameter.get_value> for more information.
        * `default` (quantity, optional, default=None): value to return if
            no results are returned by <phoebe.parameters.ParameterSet.get_parameter>
            given the value of `twig` and `**kwargs`.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter>.

        Returns
        --------
        * (float/array/string) the value of the filtered
            <phoebe.parameters.Parameter>.
        """
        # TODO: for time derivatives will need to use t instead of time (time
        # gets passed to twig filtering)

        if default is not None:
            # then we need to do a filter first to see if parameter exists
            if not len(self.filter(twig=twig, **kwargs)):
                return default

        twig, index = _extract_index_from_string(twig)
        if kwargs.get('qualifier', None):
            kwargs['qualifier'], index = _extract_index_from_string(kwargs.get('qualifier'))
        if kwargs.get('uniqueid', None):
            kwargs['uniqueid'], index = _extract_index_from_string(kwargs.get('uniqueid'))

        param = self.get_parameter(twig=twig, **kwargs)

        # if hasattr(param, 'default_unit'):
        # This breaks for constraint parameters
        if isinstance(param, FloatParameter) or\
                isinstance(param,FloatArrayParameter):

            if index is not None:
                if isinstance(param, FloatArrayParameter):
                    return param.get_value(unit=unit, t=t, **kwargs)[index]
                else:
                    raise ValueError("indices only supported for FloatArrayParameter")
            else:
                return param.get_value(unit=unit, t=t, **kwargs)


        if index is not None:
            raise ValueError("indices only supported for FloatArrayParameter")

        return param.get_value(**kwargs)

    def set_value(self, twig=None, value=None, **kwargs):
        """
        Set the value of a <phoebe.parameters.Parameter> in this
        <phoebe.parameters.ParameterSet>.

        Note: setting the value of a Parameter in a ParameterSet WILL
        change that Parameter across any parent ParameterSets (including
        the <phoebe.frontend.bundle.Bundle>).

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameter.  See <phoebe.parameters.ParameterSet.get_parameter>.
        * `value` (optional, default=None): valid value to set for the
            matched Parameter.
        * `index` (int, optional): only applicable for
            <phoebe.parameters.FloatArrayParameter>.  Passing `index` will call
            <phoebe.parameters.FloatArrayParameter.set_index_value> and pass
            `index` instead of <phoebe.parameters.FloatArrayParameter.set_value>.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter> and
            <phoebe.parameters.Parameter.set_value>.

        Raises
        --------
        * ValueError: if a unique match could not be found via
            <phoebe.parameters.ParameterSet.get_parameter>
        """
        # TODO: handle twig having parameter key (value@, default_unit@, adjust@, etc)
        # TODO: does this return anything (update the docstring)?
        if twig is not None and value is None and kwargs.get('allow_value_as_first_arg', True):
            # then try to support value as the first argument if no matches with twigs
            if not isinstance(twig, str):
                value = twig
                twig = None

            elif not len(self.filter(twig=twig, check_default=False, **kwargs)):
                value = twig
                twig = None

        if "index" in kwargs.keys():
            return self.get_parameter(twig=twig,
                                      **kwargs).set_index_value(value=value,
                                                                **kwargs)

        if "time" in kwargs.keys():
            if not len(self.filter(**kwargs)):
                # then let's try filtering without time and seeing if we get a
                # FloatArrayParameter so that we can use set_index_value instead
                time = kwargs.pop("time")

                param = self.get_parameter(twig=twig, **kwargs)
                if not isinstance(param, FloatArrayParameter):
                    raise TypeError

                # TODO: do we need to be more clever about time qualifier for
                # ETV datasets? TODO: is this robust enough... this won't search
                # for times outside the existing ParameterSet.  We could also
                # try param.get_parent_ps().get_parameter('time'), but this
                # won't work when outside the bundle (which is used within
                # backends.py to set fluxes, etc) print "***
                # get_parameter(qualifier='times', **kwargs)", {k:v for k,v in
                # kwargs.items() if k not in ['qualifier']}
                time_param = self.get_parameter(qualifier='times', **{k:v for k,v in kwargs.items() if k not in ['qualifier']})
                index = np.where(time_param.get_value()==time)[0]

                return param.set_index_value(value=value, index=index, **kwargs)

        return self.get_parameter(twig=twig,
                                  **kwargs).set_value(value=value,
                                                      **kwargs)

    def set_values_all(self, *args, **kwargs):
        """
        Alias to <phoebe.parameters.ParameterSet.set_value_all>
        """
        return self.set_value_all(*args, **kwargs)

    def set_value_all(self, twig=None, value=None, check_default=False, **kwargs):
        """
        Set the value of all returned <phoebe.parameters.Parameter> objects
        in this <phoebe.parameters.ParameterSet>.

        Any Parameter that would be included in the resulting ParameterSet
        from a <phoebe.parameters.ParametSet.filter> call with the same arguments
        will have their value set.

        Note: setting the value of a Parameter in a ParameterSet WILL
        change that Parameter across any parent ParameterSets (including
        the <phoebe.frontend.bundle.Bundle>)

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameters.  See <phoebe.parameters.ParameterSet.filter>.
        * `value` (optional, default=None): valid value to set for each
            matched Parameter.
        * `index` (int, optional): only applicable for
            <phoebe.parameters.FloatArrayParameter>.  Passing `index` will call
            <phoebe.parameters.FloatArrayParameter.set_index_value> and pass
            `index` instead of <phoebe.parameters.FloatArrayParameter.set_value>.
        * `ignore_none` (bool, optional, default=False): if `ignore_none=True`,
            no error will be raised if the filter returns 0 results.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter> and
            <phoebe.parameters.Parameter.set_value>.

        Raises
        -------
        * ValueError: if the <phoebe.parameters.ParameterSet.filter> call with
            the given `twig` and `**kwargs` returns 0 results.  This error
            is ignored if `ignore_none=True`.
        """
        if twig is not None and value is None:
            # then try to support value as the first argument if no matches with twigs
            if not isinstance(twig, str):
                value = twig
                twig = None

            elif not len(self.filter(twig=twig, check_default=check_default, **kwargs)):
                value = twig
                twig = None

        params = self.filter(twig=twig,
                             check_default=check_default,
                             **kwargs).to_list()

        if not kwargs.pop('ignore_none', False) and not len(params):
            raise ValueError("no parameters found")

        for param in params:
            if "index" in kwargs.keys():
                return self.get_parameter(twig=twig,
                                          **kwargs).set_index_value(value=value,
                                                                    **kwargs)
            param.set_value(value=value, **kwargs)

    def get_default_unit(self, twig=None, **kwargs):
        """
        Get the default unit for a <phoebe.parameters.Parameter> in the
        <phoebe.parameters.ParameterSet>.

        Note: this only works for Parameter objects with a `get_default_unit` method.
        These include:
        * <phoebe.parameters.FloatParameter.get_default_unit>
        * <phoebe.parameters.FloatArrayParameter.get_default_unit>

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.set_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>
        """
        return self.get_parameter(twig=twig, **kwargs).get_default_unit()

    def set_default_unit(self, twig=None, unit=None, **kwargs):
        """
        Set the default unit for a <phoebe.parameters.Parameter> in the
        <phoebe.parameters.ParameterSet>.

        Note: setting the default_unit of a Parameter in a ParameterSet WILL
        change that Parameter across any parent ParameterSets (including
        the <phoebe.frontend.bundle.Bundle>).

        Note: this only works for Parameter objects with a `set_default_unit` method.
        These include:
        * <phoebe.parameters.FloatParameter.set_default_unit>
        * <phoebe.parameters.FloatArrayParameter.set_default_unit>

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit_all>

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameter.  See <phoebe.parameters.ParameterSet.get_parameter>.
        * `unit` (unit, optional, default=None): valid unit to set for the
            matched Parameter.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter> and
            <phoebe.parameters.Parameter.set_value>.

        Raises
        --------
        * ValueError: if a unique match could not be found via
            <phoebe.parameters.ParameterSet.get_parameter>
        """
        if twig is not None and unit is None:
            # then try to support value as the first argument if no matches with twigs
            if isinstance(unit, u.Unit) or not isinstance(twig, str):
                unit = twig
                twig = None

            elif not len(self.filter(twig=twig, check_default=check_default, **kwargs)):
                unit = twig
                twig = None

        return self.get_parameter(twig=twig, **kwargs).set_default_unit(unit)

    def set_default_units_all(self, *args, **kwargs):
        """
        Alias to <phoebe.parameters.ParameterSet.set_default_unit_all>
        """
        return self.set_default_unit_all(*args, **kwargs)

    def set_default_unit_all(self, twig=None, unit=None, **kwargs):
        """
        Set the default unit for all <phoebe.parameters.Parameter> objects in the
        <phoebe.parameters.ParameterSet>.

        Any Parameter that would be included in the resulting ParameterSet
        from a <phoebe.parameters.ParametSet.filter> call with the same arguments
        will have their default_unit set.

        Note: setting the default_unit of a Parameter in a ParameterSet WILL
        change that Parameter across any parent ParameterSets (including
        the <phoebe.frontend.bundle.Bundle>).

        Note: this only works for Parameter objects with a `set_default_unit` method.
        These include:
        * <phoebe.parameters.FloatParameter.set_default_unit>
        * <phoebe.parameters.FloatArrayParameter.set_default_unit>

        See also:
        * <phoebe.parameters.ParameterSet.get_quantity>
        * <phoebe.parameters.ParameterSet.set_quantity>
        * <phoebe.parameters.ParameterSet.get_value>
        * <phoebe.parameters.ParameterSet.set_value>
        * <phoebe.parameters.ParameterSet.set_value_all>
        * <phoebe.parameters.ParameterSet.get_default_unit>
        * <phoebe.parameters.ParameterSet.set_default_unit>


        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameters.  See <phoebe.parameters.ParameterSet.filter>.
        * `unit` (unit, optional, default=None): valid unit to set for each
            matched Parameter.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter> and
            `set_default_unit`.

        Returns
        ----------
        * <phoebe.parameters.ParameterSet> of the changed Parameters.
        """
        # TODO: add support for ignore_none as per set_value_all
        if twig is not None and unit is None:
            # then try to support value as the first argument if no matches with twigs
            if isinstance(unit, u.Unit) or not isinstance(twig, str):
                unit = twig
                twig = None

            elif not len(self.filter(twig=twig, **kwargs)):
                unit = twig
                twig = None

        ps = self.filter(twig=twig, **kwargs)
        for param in ps.to_list():
            param.set_default_unit(unit)

        return ps

    def get_description(self, twig=None, **kwargs):
        """
        Get the description of a <phoebe.parameters.Parameter> in the
        <phoebe.parameters.ParameterSet>.

        This is simply a shortcut to <phoebe.parameters.ParameterSet.get_parameter>
        and <phoebe.parameters.Parameter.get_description>.

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to be used to access
            the Parameter.  See <phoebe.parameters.ParameterSet.get_parameter>.
        * `**kwargs`: filter options to be passed along to
            <phoebe.parameters.ParameterSet.get_parameter>.

        Returns
        --------
        * (string) the description of the filtered
            <phoebe.parameters.Parameter>.
        """
        return self.get_parameter(twig=twig, **kwargs).get_description()

    def calculate_residuals(self, model=None, dataset=None, component=None,
                            consider_gaussian_process=True,
                            as_quantity=True, return_interp_model=False,
                            mask_enabled=None, mask_phases=None):
        """
        Compute residuals between the observed values in a dataset and the
        corresponding model.

        Currently supports the following datasets:
        * <phoebe.parameters.dataset.lc>
        * <phoebe.parameters.dataset.rv>

        If necessary (due to the `compute_times`/`compute_phases` parameters
        or a change in the dataset `times` since the model was computed),
        interpolation will be handled, in time-space if possible, and in
        phase-space otherwise. See
        <phoebe.parameters.FloatArrayParameter.interp_value>.

        See also:
        * <phoebe.parameters.ParameterSet.calculate_chi2>
        * <phoebe.parameters.ParameterSet.calculate_lnlikelihood>

        Arguments
        -----------
        * `model` (string, optional, default=None): model to compare against
            observations.  Required if more than one model exist.
        * `dataset` (string, optional, default=None): dataset for comparison.
            Required if more than one dataset exist.
        * `component` (string, optional, default=None): component for comparison.
            Required only if more than one component exist in the dataset (for
            RVs, for example)
        * `consider_gaussian_process` (bool, optional, defult=True): whether
            to consider a system with gaussian process(es) as time-dependent
            for any required interpolation.
        * `as_quantity` (bool, default=True): whether to return a quantity object.
        * `return_interp_model` (bool, default=False): whether to also return
            the interpolated model used to compute the residuals.
        * `mask_enabled` (bool, optional, default=None): whether to enable
            masking on the dataset(s).  If None or not provided, will default to
            the values set in the dataset(s).
        * `mask_phases` (list of tuples, optional, default=None): phase masks
            to apply if `mask_enabled = True`.  If None or not provided, will
            default to the values set in the dataset(s).


        Returns
        -----------
        * (array) array of residuals with same length as the times array of the
            corresponding dataset.  If `return_interp_model = True`, a second
            array will be returned corresponding to the interpolated values of
            the model with the same length.

        Raises
        ----------
        * ValueError: if the provided filter options (`model`, `dataset`,
            `component`) do not result in a single parameter for comparison.
        * NotImplementedError: if the dataset kind is not supported for residuals.
        """
        if dataset is not None and not isinstance(dataset, str):
            raise TypeError("model must be of type string or None")

        if not len(self.filter(context='dataset', **_skip_filter_checks).datasets):
            dataset_ps = self._bundle.get_dataset(dataset=dataset, **_skip_filter_checks)
        else:
            dataset_ps = self.filter(dataset=dataset, context='dataset', **_skip_filter_checks)

        if dataset is not None and dataset not in dataset_ps.datasets:
            raise ValueError("dataset '{}' not found".format(dataset))

        dataset_kind = dataset_ps.kind

        if model is not None and not isinstance(model, str):
            raise TypeError("model must be of type string or None")

        if not len(self.filter(context='model', **_skip_filter_checks).models):
            model_ps = self._bundle.filter(model=model, context='model', dataset=dataset, component=component, **_skip_filter_checks)
        else:
            model_ps = self.filter(model=model, context='model', dataset=dataset, component=component, **_skip_filter_checks)

        if model is not None and model not in model_ps.models:
            raise ValueError("model '{}' not found".format(model))

        if dataset_kind == 'lc':
            qualifier = 'fluxes'
        elif dataset_kind == 'rv':
            qualifier = 'rvs'
        else:
            # TODO: lp compared for a given time interpolating in wavelength?
            # NOTE: add to documentation if adding support for other datasets

            # TODO: anything not supported here should be excluded from the
            # bexcl in fitting in solverbackends._bsolver
            raise NotImplementedError("calculate_residuals not implemented for dataset with kind='{}' (model={}, dataset={}, component={})".format(dataset_kind, model, dataset, component))


        dataset_param_ps = dataset_ps.filter(qualifier=qualifier, component=component, **_skip_filter_checks)
        if len(dataset_param_ps.to_list()) > 1:
            raise ValueError("filter (dataset={}, qualifier={}, component={}) resulted in more than one parameter".format(dataset_ps.dataset, qualifier, component))
        elif len(dataset_param_ps.to_list()) == 0:
            raise ValueError("filter (dataset={}, qualifier={}, component={}) resulted in no parameters".format(dataset_ps.dataset, qualifier, component))
        else:
            dataset_param = dataset_param_ps.to_list()[0]
        dataset_param = dataset_ps.get_parameter(qualifier=qualifier, component=component, **_skip_filter_checks)
        model_param = model_ps.get_parameter(qualifier=qualifier, **_skip_filter_checks)

        # TODO: do we need to worry about conflicting units?
        # NOTE: this should automatically handle interpolating in phases, if necessary
        times = dataset_ps.get_value(qualifier='times', component=component, **_skip_filter_checks)
        if not len(times):
            residuals = np.array([])
            interp_model = np.array([])

            if as_quantity:
                residuals *= dataset_param.default_unit
                interp_model *= dataset_param.default_unit

            if return_interp_model:
                return residuals, interp_model
            else:
                return residuals

        if not len(dataset_param.get_value()) == len(times):
            if len(dataset_param.get_value())==0:
                # then the dataset was empty, so let's just return an empty array
                if return_interp_model:
                    if as_quantity:
                        return np.asarray([]) * dataset_param.default_unit, np.asarray([]) * dataset_param.default_unit
                    else:
                        return np.asarray([]), np.asarray([])
                else:
                    if as_quantity:
                        return np.asarray([]) * dataset_param.default_unit
                    else:
                        return np.asarray([])
            else:
                raise ValueError("{}@{}@{} and {}@{}@{} do not have the same length, cannot compute residuals".format(qualifier, component, dataset, 'times', component, dataset))

        mask_enabled = dataset_ps.get_value(qualifier='mask_enabled', default=False, mask_enabled=mask_enabled, **_skip_filter_checks)
        if mask_enabled:
            mask_phases = dataset_ps.get_value(qualifier='mask_phases', mask_phases=mask_phases, **_skip_filter_checks)
            mask_period = dataset_ps.get_value(qualifier='phases_period', default='period', **_skip_filter_checks)
            mask_dpdt = dataset_ps.get_value(qualifier='phases_dpdt', default='dpdt', **_skip_filter_checks)
            mask_t0 = dataset_ps.get_value(qualifier='phases_t0', **_skip_filter_checks)
            if len(mask_phases):
                phases = self._bundle.to_phase(times, period=mask_period, dpdt=mask_dpdt, t0=mask_t0)

                inds = phase_mask_inds(phases, mask_phases)

                times = times[inds]


        if dataset_param.default_unit != model_param.default_unit:
            raise ValueError("model and dataset do not have the same default_unit, cannot interpolate")

        model_interp = model_param.interp_value(times=times, consider_gaussian_process=consider_gaussian_process)
        residuals = np.asarray(dataset_param.interp_value(times=times, consider_gaussian_process=consider_gaussian_process) - model_interp)

        if as_quantity:
            if return_interp_model:
                return residuals * dataset_param.default_unit, model_interp * dataset_param.default_unit
            else:
                return residuals * dataset_param.default_unit
        else:
            if return_interp_model:
                return residuals, model_interp
            else:
                return residuals

    def calculate_chi2(self, model=None, dataset=None, component=None,
                       consider_gaussian_process=True,
                       mask_enabled=None, mask_phases=None):
        """
        Compute the chi2 between a model and the observed values in the dataset(s).

        Currently supports the following datasets:
        * <phoebe.parameters.dataset.lc>
        * <phoebe.parameters.dataset.rv>

        If necessary (due to the `compute_times`/`compute_phases` parameters
        or a change in the dataset `times` since the model was computed),
        interpolation will be handled, in time-space if possible, and in
        phase-space otherwise. See
        <phoebe.parameters.FloatArrayParameter.interp_value>.

        Residuals per-dataset for the given model are computed by
        <phoebe.parameters.ParameterSet.calculate_residuals>.  The returned
        chi2 value is then the sum over the chi2 of each dataset, where each
        dataset's chi2 value is computed as the sum of squares of residuals
        over the squares of sigmas (if available).

        If `sigmas_lnf` is not -inf (default value), then the following term
        is added to the squares of sigmas:

        `interpolated_model**2 * np.exp(2 * sigmas_lnf)`


        See also:
        * <phoebe.parameters.ParameterSet.calculate_residuals>
        * <phoebe.parameters.ParameterSet.calculate_lnlikelihood>
        * <phoebe.frontend.bundle.Bundle.calculate_lnp>

        Arguments
        -----------
        * `model` (string, optional, default=None): model to compare against
            observations.  Required if more than one model exist.
        * `dataset` (string or list, optional, default=None): dataset(s) for comparison.
            Will sum over chi2 values of all datasets that match the filter.  So
            if not provided, will default to all datasets exposed in the model.
        * `component` (string or list, optional, default=None): component(s) for
            comparison.  Required only if more than one component exist in the
            dataset (for RVs, for example) and not all should be included in
            the chi2
        * `consider_gaussian_process` (bool, optional, default=True): whether
            to consider a system with gaussian process(es) as time-dependent
        * `mask_enabled` (bool, optional, default=None): whether to enable
            masking on the dataset(s).  If None or not provided, will default to
            the values set in the dataset(s).
        * `mask_phases` (list of tuples, optional, default=None): phase masks
            to apply if `mask_enabled = True`.  If None or not provided, will
            default to the values set in the dataset(s).

        Returns
        -----------
        * (float) chi2 value

        Raises
        ----------
        * NotImplementedError: if the dataset kind is not supported for residuals.
        """

        chi2 = 0

        if model is not None and not isinstance(model, str):
            raise TypeError("model must be of type string or None")

        if not len(self.filter(context='model', **_skip_filter_checks).models):
            model_ps = self._bundle.filter(model=model, context='model', dataset=dataset, component=component, **_skip_filter_checks)
        else:
            model_ps = self.filter(model=model, context='model', dataset=dataset, component=component, **_skip_filter_checks)

        if model is not None and model not in model_ps.models:
            raise ValueError("model '{}' not found".format(model))


        for ds in model_ps.datasets:
            ds_comps = model_ps.filter(dataset=ds, **_skip_filter_checks).components
            if not len(ds_comps):
                ds_comps = [None]

            for ds_comp in ds_comps:
                residuals, model_interp = self.calculate_residuals(model=model, dataset=ds, component=ds_comp,
                                                                   return_interp_model=True,
                                                                   consider_gaussian_process=consider_gaussian_process,
                                                                   mask_enabled=mask_enabled, mask_phases=mask_phases,
                                                                   as_quantity=True)
                ds_ps = self._bundle.get_dataset(dataset=ds, **_skip_filter_checks)
                sigmas = ds_ps.get_value(qualifier='sigmas', component=ds_comp, unit=residuals.unit, **_skip_filter_checks)

                mask_enabled = ds_ps.get_value(qualifier='mask_enabled', default=False, mask_enabled=mask_enabled, **_skip_filter_checks)
                if mask_enabled:
                    mask_phases = ds_ps.get_value(qualifier='mask_phases', mask_phases=mask_phases, **_skip_filter_checks)
                    mask_period = ds_ps.get_value(qualifier='phases_period', default='period', **_skip_filter_checks)
                    mask_dpdt = ds_ps.get_value(qualifier='phases_dpdt', default='dpdt', **_skip_filter_checks)
                    mask_t0 = ds_ps.get_value(qualifier='phases_t0', **_skip_filter_checks)
                    if len(mask_phases):
                        times = ds_ps.get_value(qualifier='times', component=ds_comp, unit=u.d, **_skip_filter_checks)
                        phases = self._bundle.to_phase(times, period=mask_period, dpdt=mask_dpdt, t0=mask_t0)

                        inds = phase_mask_inds(phases, mask_phases)

                        sigmas = sigmas[inds]


                sigmas_lnf = ds_ps.get_value(qualifier='sigmas_lnf', component=ds_comp, default=-np.inf, **_skip_filter_checks)

                if len(sigmas):
                    sigmas2 = sigmas**2
                    if sigmas_lnf != -np.inf:
                        sigmas2 += model_interp.value ** 2 * np.exp(2 * sigmas_lnf)

                    chi2 += np.sum((residuals.value**2 / sigmas2) + np.log(sigmas2))
                else:
                    chi2 += np.sum(residuals.value**2)

        return chi2

    def calculate_lnlikelihood(self, model=None, dataset=None, component=None, consider_gaussian_process=True):
        """
        Compute the log-likelihood between a model and the observed values in the dataset(s).

        Currently supports the following datasets:
        * <phoebe.parameters.dataset.lc>
        * <phoebe.parameters.dataset.rv>

        This returns -0.5 * chi2 (see <phoebe.parameters.ParameterSet.calculate_chi2>)

        See also:
        * <phoebe.parameters.ParameterSet.calculate_residuals>
        * <phoebe.parameters.ParameterSet.calculate_chi2>
        * <phoebe.frontend.bundle.Bundle.calculate_lnp>

        Arguments
        -----------
        * `model` (string, optional, default=None): model to compare against
            observations.  Required if more than one model exist.
        * `dataset` (string or list, optional, default=None): dataset(s) for comparison.
            Will sum over chi2 values of all datasets that match the filter.  So
            if not provided, will default to all datasets exposed in the model.
        * `component` (string or list, optional, default=None): component(s) for
            comparison.  Required only if more than one component exist in the
            dataset (for RVs, for example) and not all should be included in
            the chi2
        * `consider_gaussian_process` (bool, optional, default=True): whether
            to consider a system with gaussian process(es) as time-dependent

        Returns
        -----------
        * (float) log-likelihood value

        Raises
        ----------
        * NotImplementedError: if the dataset kind is not supported for residuals.
        """

        return -0.5 * self.calculate_chi2(model, dataset, component, consider_gaussian_process=consider_gaussian_process)

    def _unpack_plotting_kwargs(self, animate=False, **kwargs):

        def _handle_additional_calls(ps, kwargss):
            for kwargs in kwargss:
                additional_calls = kwargs.pop('additional_calls', [])
                if additional_calls:
                    ps = additional_calls.pop('ps', ps)
                    kw_additional = {k:v for k,v in kwargs.items() if k not in ['autofig_method', 'i', 'iqualifier', 'x', 'xqualifier', 'xerror', 'y', 'yqualifier', 'yerror', 'z', 'zqualifier', 'zerror']}
                    for k,v in additional_calls.items():
                        kw_additional[k] = v
                    # print("*** kw_additional", kw_additional)
                    kwargss += ps._unpack_plotting_kwargs(animate=animate, **kw_additional)
            return kwargss

        # We now need to unpack if the contents within kwargs contain multiple
        # contexts/datasets/kinds/components/etc.
        # the dataset tag can appear in the compute context as well, so if the
        # context tag isn't in kwargs, let's default it to dataset or model
        # print("**************************************************")
        # print("*** kwargs['context'] (provided)", kwargs.get('context'))
        # print("*** _filter['context']", self._filter.get('context'))
        if 'context' not in kwargs.keys() and 'context' not in self._filter.keys():
            provided_tags = list(self._filter.keys()) + list(kwargs.keys())
            # print("*** getting default contexts, provided_tags=", provided_tags)
            default_contexts = []
            if 'style' in kwargs.keys():
                default_contexts += ['solution']
            # if we have a tag-filter (either before or within .plot), we want to
            # include that context.  For example:
            # b.filter(model='mymodel').plot(solution='mysolution')
            # should include contexts ['model', 'solution']
            default_contexts += [context for context in self.contexts if (context in provided_tags or 'twig' in kwargs.keys()) and context in ['dataset', 'compute', 'model', 'distribution', 'solver', 'solution']]

            if not len(default_contexts):
                # then there were no tag filters so we'll default to model
                # (and therefore dataset)
                default_contexts += ['model']

            if 'model' in default_contexts and 'dataset' not in default_contexts:
                # then we also want to include context='dataset'
                # NOTE: this will not be the case if context was explicitly
                # provided by the user
                default_contexts += ['dataset']

            if 'dataset' in default_contexts and 'model' not in default_contexts:
                default_contexts += ['model']

            kwargs.setdefault('context', default_contexts)

        if isinstance(kwargs.get('context'), str):
            kwargs['context'] = [kwargs['context']]

        # print("*** kwargs['context'] (after defaults)", kwargs.get('context'))
        # print("*** before filter self.contexts", self.contexts)
        # print("*** kwargs", kwargs)
        filter_kwargs = {}
        for k in list(self.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig']).keys())+['twig']:
            if k in ['time']:
                # time handled later
                continue
            filter_kwargs[k] = kwargs.pop(k, None)

        # any items that we remove from filter_kwargs need to be replaced in kwargs
        for k,v in filter_kwargs.items():
            if not (filter_kwargs.get('context', []) is None or k not in filter_kwargs.get('context', [])):
                if k is not None:
                    kwargs[k] = filter_kwargs[k]
                filter_kwargs[k] = None

        # print("*** filter_kwargs", filter_kwargs)
        # print("*** kwargs", kwargs)
        filter_kwargs_this = {k:v for k,v in filter_kwargs.items()}
        if self.context == 'dataset':
            _ = filter_kwargs_this.pop('model')
        # print("*** applying filter", filter_kwargs_this)
        ps = self.filter(check_visible=False, **filter_kwargs_this).exclude(qualifier=['compute_times', 'compute_phases', 'compute_phases_t0', 'phases_t0', 'mask_phases'], check_visible=False)

        if 'time' in kwargs.keys() and ps.kind in ['mesh', 'lp']:
            ps = ps.filter(time=kwargs.get('time'), check_visible=False)

        # If ps returns more than one dataset/model/component, then we need to
        # loop and plot all.  This will automatically rotate through colors
        # (unless the user provided color in a kwarg).  To choose individual
        # styling, the user must make individual calls and provide the styling
        # options as kwargs.

        # we'll return a list of dictionaries, with each dictionary prepared
        # to pass directly to autofig
        return_ = []

        # print("*** after filter ps.contexts", ps.contexts)
        # print("*** after filter ps.tags", ps.tags)
        if len(ps.contexts) > 1:
            for context in ps.contexts:
                # print("*** context loop, context={}".format(context))
                filter_ = {'context': context, context: kwargs.get(context, filter_kwargs.get(context, None))}
                if context == 'model':
                    filter_['dataset'] = filter_kwargs.get('dataset', None)
                # print("*** context loop, applying filter", filter_)
                this_return = ps.filter(check_visible=False, **filter_)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        # if len(ps.distributions)>1:
        #     for distribution in ps.distributions:
        #         this_return = ps.filter(check_visible=False, distribution=distribution)._unpack_plotting_kwargs(animate=animate, **kwargs)
        #         return_ += this_return
        #     return _handle_additional_calls(ps, return_)

        if ps.context=='compute' and len(ps.computes)>1:
            for compute in ps.filter(compute=kwargs.get('compute', filter_kwargs.get('compute', None)), **_skip_filter_checks).computes:
                this_return = ps.filter(check_visible=False, compute=compute)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        elif ps.context=='solver' and len(ps.solvers)>1:
            for solver in ps.filter(solver=kwargs.get('solver', filter_kwargs.get('solver', None)), **_skip_filter_checks).solvers:
                this_return = ps.filter(check_visible=False, solver=solver)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        elif ps.context=='solution' and len(ps.solutions)>1:
            for solution in ps.filter(solution=kwargs.get('solution', filter_kwargs.get('solution', None)), **_skip_filter_checks).solutions:
                # print("*** solution loop, solution={}".format(solution))
                this_return = ps.filter(check_visible=False, solution=solution)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        elif ps.context in ['dataset', 'model'] and len(ps.datasets)>1 and not (ps.context=='dataset' and ps.kind in ['mesh', 'orb']):
            # print("*** entering dataset loop, filter_kwargs['dataset']={}, kwargs['dataset']={}".format(filter_kwargs.get('dataset', None), kwargs.get('dataset', None)))
            for dataset in ps.filter(dataset=kwargs.get('dataset', filter_kwargs.get('dataset', None)), **_skip_filter_checks).datasets:
                # print("*** dataset loop, context={}, dataset={}".format(ps.context, dataset))
                # print("*** dataset loop filter_kwargs['model']={}, kwargs['model']={}".format(filter_kwargs.get('model', None), kwargs.get('model', None)))
                filter_model = kwargs.get('model', filter_kwargs.get('model', None))
                if filter_model is not None and dataset not in self._bundle.filter(model=filter_model, context='model', **_skip_filter_checks).datasets:
                    # in cases where a dataset is disabled for a certain model, we shouldn't
                    # try to plot the observations when model was provided
                    continue
                if ps.kind=='mesh' and self._bundle.get_dataset(dataset=dataset).kind != ps.kind and kwargs.get('x', None) in [None, 'us', 'vs', 'ws', 'xs', 'ys', 'zs'] and kwargs.get('y', None) in [None, 'us', 'vs', 'ws', 'xs', 'ys', 'zs']:
                    # avoid a y vs x plot for meshes with lc-columns
                    continue
                this_return = ps.filter(check_visible=False, dataset=dataset)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        # If we are asking to plot a dataset that also shows up in columns in
        # the mesh, then remove the mesh kind.  In other words: mesh stuff
        # will only be plotted if mesh is the only kind in the filter.
        pskinds = ps.filter(kind=kwargs.get('kind', filter_kwargs.get('kind', None))).kinds
        if len(pskinds) > 1 and 'mesh' in pskinds:
            pskinds.remove('mesh')

        if len(ps.kinds) > 1:
            for kind in pskinds:
                # print("*** kind loop, kind={}".format(kind))
                this_return = ps.filter(kind=kind, check_visible=False)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        if len(ps.models) > 1: # and ps.context=='model'
            # we'll filter by filter_kwargs again in case it wasn't filtered above for being in default_contexts
            for model in ps.filter(model=kwargs.get('model', filter_kwargs.get('model', None)), **_skip_filter_checks).models:
                # TODO: change linestyle for models instead of color?
                # print("*** model loop, model={}".format(model))
                this_return = ps.filter(check_visible=False, model=model)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        if len(ps.times) > 1 and kwargs.get('x', None) not in ['time', 'times'] and kwargs.get('y', None) not in ['time', 'times'] and kwargs.get('z', None) not in ['time', 'times']:
            # only meshes, lp, spectra, etc will be able to iterate over times
            for time in ps.times:
                this_return = ps.filter(check_visible=False, time=time)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)

        if len(ps.components) > 1 and ps.context in ['model', 'dataset'] and ps.kind not in ['lc']:
            # lc has per-component passband-dependent parameters in the dataset which are not plottable
            return_ = []
            for component in ps.filter(component=kwargs.get('component', filter_kwargs.get('component', None)), **_skip_filter_checks).exclude(qualifier=['*_phases', 'phases_*'], **_skip_filter_checks).components:
                # print("*** component loop, component={}".format(component))
                this_return = ps.filter(check_visible=False, component=component)._unpack_plotting_kwargs(animate=animate, **kwargs)
                return_ += this_return
            return _handle_additional_calls(ps, return_)


        if ps.kind in ['mesh', 'orb'] and \
                ps.context == 'dataset':
            # nothing to plot here... at least for now
            return []

        if ps.kind in ['lp'] and not len(ps.filter(qualifier='flux_densities', **_skip_filter_checks)):
            # then maybe we're in the dataset where just compute_times is defined
            return []

        if not len(ps):
            return []

        # Now that we've looped over everything, we can assume that we are dealing
        # with a SINGLE call.  We need to prepare kwargs so that it can be passed
        # to autofig.plot or autofig.mesh


        #### SUPPORT DICTIONARIES IN KWARGS
        # like color={'primary': 'red', 'secondary': 'blue'} or
        # linestyle={'rv01': 'solid', 'rv02': 'dashed'}
        # here we need to filter any kwargs that are dictionaries if they match
        # the current ps
        for k,v in kwargs.copy().items():
            if isinstance(v, dict) and 'kwargs' not in k:
                # overwrite kwargs[k] based on any match in v
                match = None
                for kk,vv in v.items():
                    meta = ps.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig'])
                    # support twigs as well as wildcards in the dictionary keys
                    # for example: color={'lc*': 'blue', 'primary@rv*': 'green'}
                    # this will likely be a little expensive, but we only do it
                    # in the case where a dictionary is passed.
                    logger.debug("_unpack_plotting_kwargs: trying to find match for dictionary {}={} in kwargs against meta={}.  match={}".format(k,v,meta,match))
                    if np.all([np.any([_fnmatch(mv, kksplit) for mv in meta.values() if mv is not None]) for kksplit in kk.split('@')]):
                        if match is not None:
                            raise ValueError("dictionary {}={} is not unique for {}".format(k,v, meta))
                        match = vv


                if match is not None:
                    kwargs[k] = match
                else:
                    # remove from the dictionary and fallback on defaults
                    _dump = kwargs.pop(k)

        #### ALIASES
        if 'color' in kwargs.keys() and 'colors' not in kwargs.keys() and 'c' not in kwargs.keys():
            logger.warning("assuming you meant 'c' instead of 'color'")
            kwargs['c'] = kwargs.pop('color')
        elif 'colors' in kwargs.keys() and 'c' not in kwargs.keys():
            logger.warning("assuming you meant 'c' instead of 'colors'")
            kwargs['c'] = kwargs.pop('colors')
        if 'facecolor' in kwargs.keys() and 'facecolors' not in kwargs.keys() and 'fc' not in kwargs.keys():
            logger.warning("assuming you meant 'fc' instead of 'facecolor'")
            kwargs['fc'] = kwargs.pop('facecolor')
        elif 'facecolors' in kwargs.keys() and 'fc' not in kwargs.keys():
            logger.warning("assuming you meant 'fc' instead of 'facecolors'")
            kwargs['fc'] = kwargs.pop('facecolors')
        if 'edgecolor' in kwargs.keys() and 'edgecolors' not in kwargs.keys() and 'ec' not in kwargs.keys():
            logger.warning("assuming you meant 'ec' instead of 'edgecolor'")
            kwargs['ec'] = kwargs.pop('edgecolor')
        elif 'edgecolors' in kwargs.keys() and 'ec' not in kwargs.keys():
            logger.warning("assuming you meant 'ec' instead of 'edgecolors'")
            kwargs['ec'] = kwargs.pop('edgecolors')

        for k in ['c', 'fc', 'ec']:
            if k in kwargs.keys():
                kwargs[k] = _phoebecolorsdict.get(kwargs[k], kwargs[k])

        for d in ['x', 'y', 'z']:
            if '{}error'.format(d) not in kwargs.keys():
                if '{}errors'.format(d) in kwargs.keys():
                    logger.warning("assuming you meant '{}error' instead of '{}errors'".format(d,d))
                    kwargs['{}error'.format(d)] = kwargs.pop('{}errors'.format(d))

        def _handle_mask(ps, array, **kwargs):
            mask_enabled = ps.get_value(qualifier='mask_enabled', mask_enabled=kwargs.get('mask_enabled', None), default=False, **_skip_filter_checks)
            if not mask_enabled:
                return array

            # mask_phases and phases_t0 was excluded from the filter to avoid
            # looping over the components they're attached to, so we'll need
            # to re-filter for the entire dataset first
            ps_ds = ps._bundle.get_dataset(dataset=ps.dataset, **_skip_filter_checks)
            mask_phases = ps_ds.get_value(qualifier='mask_phases', mask_phases=kwargs.get('mask_phases', None), **_skip_filter_checks)
            if not len(mask_phases):
                return array

            mask_period = ps_ds.get_value(qualifier='phases_period', default='period', phases_period=kwargs.get('phases_period', None), **_skip_filter_checks)
            mask_dpdt = ps_ds.get_value(qualifier='phases_dpdt', default='dpdt', phases_dpdt=kwargs.get('phases_dpdt', None), **_skip_filter_checks)
            mask_t0 = ps_ds.get_value(qualifier='phases_t0', phases_t0=kwargs.get('phases_t0', None), **_skip_filter_checks)

            times = ps.get_value(qualifier='times', unit=u.d, **_skip_filter_checks)
            phases = ps._bundle.to_phase(times, period=mask_period, dpdt=mask_dpdt, t0=mask_t0)

            return array[phase_mask_inds(phases, mask_phases)]


        def _kwargs_fill_dimension(kwargs, direction, ps):
            # kwargs[direction] is currently one of the following:
            # * twig/qualifier
            # * array/float
            # * string (applicable for color dimensions)
            #
            # if kwargs[direction] is a twig, then we need to change the
            # entry in the dictionary to be the data-array itself

            current_value = kwargs.get(direction, None)

            #### RETRIEVE DATA ARRAYS
            if isinstance(current_value, str):
                if ps.kind != 'mesh' and direction in ['fc', 'ec']:
                    logger.warning("fc and ec are not allowable for dataset={} with kind={}, ignoring {}={}".format(ps.dataset, ps.kind, direction, current_value))
                    _dump = kwargs.pop(direction)
                    return kwargs

                elif current_value in ['None', 'none']:
                    return kwargs

                elif '@' in current_value or current_value in ps.qualifiers or \
                        (current_value in ['xs', 'ys', 'zs'] and 'xyz_elements' in ps.qualifiers) or \
                        (current_value in ['us', 'vs', 'ws'] and 'uvw_elements' in ps.qualifiers):

                    if kwargs['autofig_method'] == 'mesh' and current_value in ['xs', 'ys', 'zs']:
                        # then we actually need to unpack from the xyz_elements
                        verts = ps.get_quantity(qualifier='xyz_elements', **_skip_filter_checks)
                        if not verts.shape[0]:
                            return None
                        array_value = verts.value[:, :, ['xs', 'ys', 'zs'].index(current_value)] * verts.unit

                        if direction == 'z':
                            try:
                                norms = ps.get_quantity(qualifier='xyz_normals', **_skip_filter_checks)
                            except ValueError:
                                # if importing from 2.1, uvw_elements may exist, but uvw_normals won't
                                array_value_norms = None
                            else:
                                array_value_norms = norms.value[:, ['xs', 'ys', 'zs'].index(current_value)]
                                # TODO: flip if necessary for a right-handed axes?  (currently the z-values aren't flipped)
                                # if
                                    # array_value_norms *= -1
                            kwargs['{}normals'.format(direction)] = array_value_norms

                    elif kwargs['autofig_method'] == 'mesh' and current_value in ['us', 'vs', 'ws']:
                        # then we actually need to unpack from the uvw_elements
                        verts = ps.get_quantity(qualifier='uvw_elements', **_skip_filter_checks)
                        if not verts.shape[0]:
                            return None
                        array_value = verts.value[:, :, ['us', 'vs', 'ws'].index(current_value)] * verts.unit

                        if direction == 'z':
                            try:
                                norms = ps.get_quantity(qualifier='uvw_normals', **_skip_filter_checks)
                            except ValueError:
                                # if importing from 2.1, uvw_elements may exist, but uvw_normals won't
                                array_value_norms = None
                            else:
                                array_value_norms = norms.value[:, ['us', 'vs', 'ws'].index(current_value)]
                                # TODO: flip if necessary for a right-handed axes?  (currently the z-values aren't flipped)
                                # if
                                    # array_value_norms *= -1
                            kwargs['{}normals'.format(direction)] = array_value_norms

                    elif current_value in ['time', 'times'] and 'residuals' in kwargs.values():
                        # then we actually need to pull the times from the dataset instead of the model since the length may not match
                        ds_ps = ps._bundle.get_dataset(dataset=ps.dataset, **_skip_filter_checks)
                        array_value = _handle_mask(ds_ps, ds_ps.get_quantity(qualifier='times', component=ps.component, **_skip_filter_checks), **kwargs)

                    else:
                        if '@' in current_value:
                            # then we need to remove the dataset from the filter
                            psf = self._bundle.filter(**{k:v for k,v in ps.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig']).items() if k!='dataset'})
                        else:
                            psf = ps

                        psff = psf.filter(twig=current_value, **_skip_filter_checks)
                        if len(psff)==1:
                            array_value = psff.get_quantity(**_skip_filter_checks)
                        elif len(psff.times) > 1 and psff.get_value(time=psff.times[0], **_skip_filter_checks):
                            # then we'll assume we have something like volume vs times.  If not, then there may be a length mismatch issue later
                            unit = psff.get_quantity(time=psff.times[0], **_skip_filter_checks).unit
                            array_value = np.array([psff.get_quantity(time=time, **_skip_filter_checks).to(unit).value for time in psff.times])*unit
                        else:
                            raise ValueError("could not find Parameter for {} in {}".format(current_value, psf.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig'])))

                        array_value = _handle_mask(psf, array_value, **kwargs)

                    kwargs[direction] = array_value

                    if ps.context == 'dataset' and current_value in sigmas_avail:
                        # then let's see if there are errors
                        errorkey = '{}error'.format(direction)
                        errors = kwargs.get(errorkey, None)
                        if isinstance(errors, np.ndarray) or isinstance(errors, float) or isinstance(errors, int):
                            kwargs[errorkey] = errors
                        elif isinstance(errors, str):
                            errors = _handle_mask(ps, ps.get_quantity(kwargs.get(errorkey), **_skip_filter_checks), **kwargs)
                            kwargs[errorkey] = errors
                        else:
                            sigmas = _handle_mask(ps, ps.get_quantity(qualifier='sigmas', **_skip_filter_checks), **kwargs)
                            if len(sigmas):
                                kwargs.setdefault(errorkey, sigmas)

                    # now let's set the label for the dimension from the qualifier/twig
                    kwargs.setdefault('{}label'.format(direction), _plural_to_singular_get(current_value))

                    # we'll also keep the qualifier around - autofig doesn't use this
                    # but we'll keep it so we can set some defaults
                    kwargs['{}qualifier'.format(direction)] = current_value

                    return kwargs

                elif current_value in ['time', 'times'] and len(ps.times):
                    kwargs[direction] = sorted([float(t) for t in ps.times])
                    kwargs['{}qualifier'] = None
                    return kwargs

                elif current_value in ['wavelengths'] and ps.time is not None:
                    # these are not tagged with the time, so we need to find them
                    full_dataset_meta = ps.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig', 'qualifier', 'time'])
                    full_dataset_ps = ps._bundle.filter(check_visible=False, **full_dataset_meta)
                    candidate_params = full_dataset_ps.filter(qualifier=current_value, **_skip_filter_checks)
                    if len(candidate_params) == 1:
                        kwargs[direction] = candidate_params.get_quantity()
                        kwargs.setdefault('{}label'.format(direction), _plural_to_singular_get(current_value))
                        kwargs['{}qualifier'.format(direction)] = current_value
                        return kwargs
                    elif len(candidate_params) > 1:
                        raise ValueError("could not find single match for {}={}, found: {}".format(direction, current_value, candidate_params.twigs))
                    else:
                        # then len(candidate_params) == 0
                        raise ValueError("could not find a match for {}={}".format(direction, current_value))

                elif current_value.split(':')[0] in ['phase', 'phases']:
                    component_phase = current_value.split(':')[1] \
                                        if len(current_value.split(':')) > 1 \
                                        else None


                    if 'residuals' in kwargs.values():
                        # then we actually need to pull the times from the dataset instead of the model since the length may not match
                        ds_ps = ps._bundle.get_dataset(dataset=ps.dataset, **_skip_filter_checks)
                        times = ds_ps.get_value(qualifier='times', component=ps.component, **_skip_filter_checks)
                        times = _handle_mask(ds_ps, times, **kwargs)
                    elif ps.kind == 'etvs':
                        times = ps.get_value(qualifier='time_ecls', unit=u.d, **_skip_filter_checks)
                        times = _handle_mask(ps, times, **kwargs)
                    else:
                        times = ps.get_value(qualifier='times', unit=u.d, **_skip_filter_checks)
                        times = _handle_mask(ps, times, **kwargs)

                    kwargs[direction] = self._bundle.to_phase(times, component=component_phase, period=kwargs.get('period', 'period'), t0=kwargs.get('t0', 't0_supconj'), dpdt=kwargs.get('dpdt', 'dpdt')) * u.dimensionless_unscaled

                    kwargs.setdefault('{}label'.format(direction), 'phase:{}'.format(component_phase) if component_phase is not None else 'phase')

                    kwargs['{}qualifier'.format(direction)] = current_value

                    # and we'll set the linebreak so that decreasing phase breaks any lines (allowing for phase wrapping)
                    kwargs.setdefault('linebreak', '{}-'.format(direction))

                    return kwargs
                elif current_value.split('_')[-1] in ['gps', 'nogps']:
                    if ps.model is None:
                        logger.info("skipping {} for dataset".format(current_value))
                        return {}
                    kwargs['{}qualifier'.format(direction)] = current_value
                    return kwargs

                elif current_value in ['residuals_spread']:
                    if ps.model is None:
                        logger.info("skipping residuals_spread for dataset")
                        return {}

                    if '-sigma' in self._bundle.get_value(qualifier='sample_mode', model=ps.model, context='model', default='none', **_skip_filter_checks):
                        # NOTE: this probably needs to be interpolated
                        kwargs[direction] = ps.get_quantity(qualifier=['fluxes', 'rvs'], model=ps.model, dataset=ps.dataset, component=ps.component, context='model', **_skip_filter_checks)
                        kwargs[direction] -= kwargs[direction][1]
                        kwargs.setdefault('{}label'.format(direction), '{} residuals'.format({'lc': 'flux', 'rv': 'rv'}.get(ps.kind, '')))
                        kwargs['{}qualifier'.format(direction)] = 'residuals'
                        # try to place on top of data/error bars since transparency will be applied
                        kwargs.setdefault('z', 1)
                        return kwargs
                    else:
                        return {}

                elif current_value in ['residuals']:
                    if ps.model is None:
                        logger.info("skipping residuals for dataset")
                        return {}

                    if '-sigma' in self._bundle.get_value(qualifier='sample_mode', model=ps.model, context='model', default='none', **_skip_filter_checks):
                        # TODO: if we ever use this for anything else, then we'll need to make it a list instead and append new items
                        # kwargs['additional_calls'] = {'y': 'residuals_spread', 'ps': ps, **{k:v for k,v in kwargs.items() if k in ['x']}} # not python2 safe :-(

                        if kwargs.get('xqualifier', 'times') in ['time', 'times']:
                            sample_x = self._bundle.get_quantity(qualifier='times', model=ps.model, component=ps.component, dataset=ps.dataset, context='model', **_skip_filter_checks)
                        elif kwargs.get('xqualifier', 'times') in ['phase', 'phases']:
                            sample_times = self._bundle.get_value(qualifier='times', model=ps.model, component=ps.component, dataset=ps.dataset, context='model', unit=u.d, **_skip_filter_checks)
                            # TODO: should this to_phase take t0/period?
                            sample_x = self._bundle.to_phase(sample_times) * u.dimensionless_unscaled
                        else:
                            raise NotImplementedError("cannot plot residuals from the sampled model with x='{}'".format(kwargs.get('xqualifier')))
                        kwargs['additional_calls'] = {'y': 'residuals_spread', 'ps': ps, 'x': sample_x}

                    # we're currently within the MODEL context
                    # NOTE: calculate_residuals will already handle masking
                    kwargs[direction] = ps.calculate_residuals(model=ps.model,
                                                               dataset=ps.dataset,
                                                               component=ps.component,
                                                               as_quantity=True,
                                                               mask_enabled=kwargs.get('mask_enabled', None),
                                                               mask_phases=kwargs.get('mask_phases', None))

                    kwargs.setdefault('{}label'.format(direction), '{} residuals'.format({'lc': 'flux', 'rv': 'rv'}.get(ps.kind, '')))
                    kwargs['{}qualifier'.format(direction)] = current_value
                    kwargs.setdefault('linestyle', 'none')
                    kwargs.setdefault('marker', '+')

                    # now let's see if there are errors
                    errorkey = '{}error'.format(direction)
                    errors = kwargs.get(errorkey, None)
                    if isinstance(errors, np.ndarray) or isinstance(errors, float) or isinstance(errors, int):
                        kwargs[errorkey] = errors
                    elif isinstance(errors, str):
                        ds_ps = self._bundle.get_dataset(ps.dataset, **_skip_filter_checks)
                        errors = ds_ps.get_quantity(qualifier=kwargs.get(errorkey), context='dataset', **_skip_filter_checks)
                        kwargs[errorkey] = _handle_mask(ds_ps, errors, **kwargs)
                    else:
                        ds_ps = self._bundle.get_dataset(ps.dataset, **_skip_filter_checks)
                        sigmas = ds_ps.get_quantity(qualifier='sigmas', component=ps.component, context='dataset', **_skip_filter_checks)
                        sigmas = _handle_mask(ds_ps, sigmas, **kwargs)
                        if len(sigmas):
                            kwargs.setdefault(errorkey, sigmas)

                    return kwargs

                elif direction in ['c', 'fc', 'ec']:
                    # then there is the possibility of referring to a column
                    # that technnically is attached to a different dataset in
                    # the same mesh (e.g. rvs@rv01 inside kind=mesh).  Let's
                    # check for that first.

                    if ps.kind == 'mesh' and ps._bundle is not None:
                        full_mesh_meta = ps.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig', 'qualifier', 'dataset'])
                        full_mesh_ps = ps._bundle.filter(check_visible=False, **full_mesh_meta)
                        candidate_params = full_mesh_ps.filter(current_value, **_skip_filter_checks)
                        if len(candidate_params) == 1:
                            kwargs[direction] = candidate_params.get_quantity()
                            kwargs.setdefault('{}label'.format(direction), _plural_to_singular_get(current_value))
                            kwargs['{}qualifier'.format(direction)] = current_value
                            return kwargs
                        elif len(candidate_params) > 1:
                            raise ValueError("could not find single match for {}={}, found: {}".format(direction, current_value, candidate_params.twigs))
                        elif current_value in autofig.cyclers._mplcolors:
                            # no need to raise a warning, this is a valid color
                            pass
                        else:
                            # maybe a hex or anything not in the cycler? or should we raise an error instead?
                            logger.warning("could not find Parameter match for {}={} at time={}, assuming named color".format(direction, current_value, full_mesh_meta['time']))


                    # Nothing has been found, so we'll assume the string is
                    # the name of a color.  If the color isn't accepted by
                    # autofig then autofig will raise an error listing the
                    # list of allowed colors.
                    return kwargs

                else:
                    raise ValueError("could not recognize '{}' for {} direction in dataset='{}', ps.meta={}".format(current_value, direction, ps.dataset, ps.meta))

            elif _instance_in(current_value, np.ndarray, list, tuple, float, int):
                # then leave it as-is
                return kwargs
            elif current_value is None:
                return kwargs
            else:
                raise NotImplementedError


        #### DIRECTION DEFAULTS
        # define defaults for directions based on ps.kind
        if ps.kind == 'mesh':
            # TODO: check to make sure axes will be right-handed?
            # first determine from any passed values if we're in xyz or uvw
            # (do not allow mixing between roche and POS)
            detected_qualifiers = [kwargs[af_direction] for af_direction in ['x', 'y', 'z'] if af_direction in kwargs.keys()]
            if len(detected_qualifiers):
                coordinate_systems = set(['uvw' if detected_qualifier in ['us', 'vs', 'ws'] else 'xyz' for detected_qualifier in detected_qualifiers if detected_qualifier in ['us', 'vs', 'ws', 'xs', 'ys', 'zs']])


                if len(coordinate_systems) == 1:
                    coordinates = ['xs', 'ys', 'zs'] if list(coordinate_systems)[0] == 'xyz' else ['us', 'vs', 'ws']
                elif len(coordinate_systems) > 1:
                    # then we're mixing roche and POS
                    raise ValueError("cannot mix xyz (roche) and uvw (pos) coordinates while plotting")
                else:
                    # then len(coordinate_system) == 0
                    coordinates = ['us', 'vs', 'ws']

            elif 'uvw_elements' in ps.qualifiers:
                coordinates = ['us', 'vs', 'ws']
            elif 'xyz_elements' in ps.qualifiers:
                coordinates = ['xs', 'ys', 'zs']
            else:
                # then we're doing a scatter plot
                coordinates = []


            defaults = {}
            # first we need to know if any of the af_directions are set to
            # something other than cartesian by the user (in which case we need
            # to check for the parameter's existence before defaulting and use
            # scatter instead of mesh plot)
            mesh_all_cartesian = True
            for af_direction in ['x', 'y', 'z']:
                if kwargs.get(af_direction, None) not in [None] + coordinates:
                    mesh_all_cartesian = False

            # now we need to loop again and set any missing defaults
            for af_direction in ['x', 'y', 'z']:
                if af_direction in kwargs.keys():
                    # then default doesn't matter, but we'll set it at what it is
                    defaults[af_direction] = kwargs[af_direction]

                    if kwargs[af_direction] in coordinates:
                        # the provided qualifier could be something else (ie teffs)
                        # in which case we'll end up doing a scatter instead of
                        # a mesh plot

                        # now we'll remove from coordinates still available
                        coordinates.remove(kwargs[af_direction])
                elif len(coordinates):
                    # we'll take the first entry remaining in coordinates
                    coordinate = coordinates.pop(0)

                    # if mesh_all_cartesian then we're doing a mesh plot
                    # and know that we have xyz/uvw_elements available.
                    # Otherwise, we need to check and only apply the default
                    # if that parameter (xs, ys, zs, us, vs, ws) is available.
                    # Either way, we've removed this from the coordinates
                    # list so the next direction will fill from the next available
                    if mesh_all_cartesian or coordinate in ps.qualifiers:
                        defaults[af_direction] = coordinate

                else:
                    # then we need defaults for a scatter plot
                    mesh_all_cartesian = False

                    # for now we'll just go based on the order of the qualifiers
                    # but we probably could be a little smarter here, especially
                    # if the user overrides a dimension to make sure we don't
                    # repeat, etc.
                    if af_direction == 'z':
                        # otherwise for 2d scatter plots this just gets
                        # prohibitively expensive
                        defaults['z'] = 0.0
                    else:
                        qualifiers_avail = [q for q in ps.qualifiers if q != 'times']
                        index = ['x', 'y'].index(af_direction)
                        if not len(qualifiers_avail):
                            raise ValueError("cannot plot mesh with no columns")

                        if index > len(qualifiers_avail) - 1:
                            index = len(qualifiers_avail) - 1

                        defaults[af_direction] = qualifiers_avail[index]


            # since we'll be selecting from the time tag, we need a non-zero tolerance
            kwargs.setdefault('itol', 1e-6)

            if mesh_all_cartesian:
                # then we'll be doing a mesh plot, so set some reasonable defaults

                # units will have handled this in POS (uvw) coordinates, but not
                # Roche (xyz) as those are unitless
                kwargs.setdefault('equal_aspect', True)
                kwargs.setdefault('pad_aspect', not animate)

                # we want the wireframe by default
                kwargs.setdefault('ec', 'black')
                kwargs.setdefault('fc', 'white')

                # by default, we'll exclude the back if fc is not 'none'
                if kwargs.get('fc') != 'none':
                    kwargs.setdefault('exclude_back', True)

            else:
                # then even though the scatter may be rs vs cartesian with same
                # units, let's default to disabling equal aspect ratio
                kwargs.setdefault('equal_aspect', False)

            sigmas_avail = []

        elif ps.kind == 'orb':
            # similar logic to meshes above, except we only have uvw
            coordinates = ['us', 'vs', 'ws']

            defaults = {}
            for af_direction in ['x', 'y', 'z']:
                if af_direction in kwargs.keys():
                    # then default doesn't matter, but we'll set it at what it is
                    defaults[af_direction] = kwargs[af_direction]

                    if kwargs[af_direction] in coordinates:
                        # the provided qualifier could be something else (ie teffs)
                        # in which case we'll end up doing a scatter instead of
                        # a mesh plot

                        # now we'll remove from coordinates still available
                        coordinates.remove(kwargs[af_direction])
                else:
                    # we'll take the first entry remaining in coordinates
                    defaults[af_direction] = coordinates.pop(0)

            if kwargs.get('projection', None) != '3d':
                defaults['z'] = 0

            sigmas_avail = []
        elif ps.kind == 'lc':
            defaults = {'x': 'times',
                        'y': 'fluxes',
                        'z': 0}
            sigmas_avail = ['fluxes']
        elif ps.kind == 'rv':
            defaults = {'x': 'times',
                        'y': 'rvs',
                        'z': 0}
            sigmas_avail = ['rvs']
        elif ps.kind == 'lp':
            defaults = {'x': 'wavelengths',
                        'y': 'flux_densities',
                        'z': ps._bundle.hierarchy.get_components().index(ps.component if ps.component is not None else ps._bundle.hierarchy.get_top())}
            sigmas_avail = ['flux_densities']

            # since we'll be selecting from the time tag, we need a non-zero tolerance
            kwargs.setdefault('itol', 1e-6)

            # if animating or drawing at a single time, we want to show only
            # the selected item, not all and then highlight the selected item
            kwargs.setdefault('highlight_linestyle', kwargs.get('linestyle', 'solid'))
            kwargs.setdefault('highlight_marker', 'None')
            kwargs.setdefault('highlight_size', kwargs.get('size', 0.02))  # this matches the default in autofig for call._sizes
            kwargs.setdefault('uncover', True)
            kwargs.setdefault('trail', 0)

        elif ps.kind == 'etv':
            defaults = {'x': 'time_ecls',
                        'y': 'etvs',
                        'z': 0}
            sigmas_avail = ['etvs']
        elif ps.kind in ['emcee', 'dynesty', 'lc_periodogram', 'rv_periodogram', 'lc_geometry', 'rv_geometry', 'ebai']:
            pass
            # handled below
        elif ps.context in ['distribution']:
            pass
            # handled below
        elif ps.context in ['solver', 'solution']:
            # ignore non-implemented solver/solution parameters
            return []
        elif ps.context in ['compute']:
            if 'sample_from' in ps.qualifiers:
                pass
            else:
                return []
        else:
            logger.debug("could not find plotting defaults for ps.meta: {}, ps.twigs: {}".format(ps.meta, ps.twigs))
            raise NotImplementedError("defaults for kind {} (dataset: {}) not yet implemented".format(ps.kind, ps.dataset))

        #### DETERMINE AUTOFIG PLOT TYPE
        # NOTE: this must be done before calling _kwargs_fill_dimension below
        cartesian = ['xs', 'ys', 'zs', 'us', 'vs', 'ws']
        if ps.context == 'model' and kwargs.get('style', None) in ['corner', 'failed']:
            kwargs['plot_package'] = 'corner'
            kwargs['data'] = ps.get_value(qualifier='samples', default=[], **_skip_filter_checks)


            # TODO: use units from fitted_units instead of parameter?

            try:
                params_uniqueids_and_indices = [_extract_index_from_string(uid) for uid in ps.get_value(qualifier='sampled_uniqueids', **_skip_filter_checks)]
                param_list = [self._bundle.get_parameter(uniqueid=uniqueid, **_skip_filter_checks) for uniqueid, index in params_uniqueids_and_indices]
                kwargs['labels'] = [_corner_label(param, uid_and_index[1]) for param, uid_and_index in zip(param_list, param_uniqueids_and_indices)]
            except:
                logger.warning("could not match to sampled_uniqueids, falling back on sampled_twigs")
                params_twigs_and_indices = [_extract_index_from_string(twig) for twig in ps.get_value(qualifier='sampled_twigs', **_skip_filter_checks)]
                param_list = [self._bundle.get_parameter(twig=twig, **_skip_filter_checks) for twig, index in params_twigs_and_indices]
                kwargs['labels'] = [_corner_label(param, twig_and_index[1]) for param, twig_and_index in zip(param_list, param_twigs_and_indices)]

            if kwargs.get('style') == 'failed':
                kwargs.setdefault('plot_uncertainties', False)
                kwargs['failed_samples'] = ps.get_value(qualifier='failed_samples', default={}, **_skip_filter_checks)

            return (kwargs,)
        elif ps.context == 'distribution':
            kwargs['plot_package'] = 'distl'
            kwargs.setdefault('distribution', ps.distribution)
            kwargs['dc'], _ = self._bundle.get_distribution_collection(context='distribution', **{k:v for k,v in kwargs.items() if k in ['distribution', 'combine', 'include_constrained', 'to_univariates', 'to_uniforms', 'parameters', 'plot_uncertainties']})
            return (kwargs,)
        elif ps.context == 'solver':
            kwargs['plot_package'] = 'distl'
            kwargs['dc'], _ = self._bundle.get_distribution_collection(twig=kwargs.get('distribution_twig', 'priors@{}'.format(ps.solver)))
            return (kwargs,)
        elif ps.context == 'compute':
            if not len(ps.get_value(qualifier='sample_from', expand=True, **_skip_filter_checks)):
                return []
            kwargs['plot_package'] = 'distl'
            kwargs['dc'], _ = self._bundle.get_distribution_collection(twig=kwargs.get('distribution_twig', 'sample_from@{}'.format(ps.compute)))
            return (kwargs,)
        elif ps.kind in ['lc_periodogram', 'rv_periodogram']:
            kwargs['plot_package'] = 'autofig'
            kwargs['x'] = ps.get_quantity(qualifier='period', **_skip_filter_checks)
            kwargs['xlabel'] = 'period'
            kwargs['y'] = ps.get_value(qualifier='power', **_skip_filter_checks)
            kwargs['ylabel'] = 'power'

            kwargs.setdefault('marker', 'None')
            # kwargs.setdefault('linestyle', 'solid')

            axvline_kwargs = {'plot_package': 'autofig', 'autofig_method': 'plot'}
            axvline_kwargs['x'] = ps.get_value(qualifier='fitted_values', **_skip_filter_checks)[0] * ps.get_value(qualifier='period_factor', period_factor=kwargs.get('period_factor', None), **_skip_filter_checks) * u.d
            axvline_kwargs['linestyle'] = 'dashed'
            axvline_kwargs['axvline'] = True # to avoid the empty y ignore in plot


            return (kwargs, axvline_kwargs)

        elif ps.kind == 'lc_geometry':
            # lc = ps.get_value(qualifier='lc', **_skip_filter_checks)
            orbit = ps.get_value(qualifier='orbit', **_skip_filter_checks)
            primary, secondary = self._bundle.hierarchy.get_children_of(orbit)
            # phases = self._bundle.to_phase(self._bundle.get_value(qualifier='times', dataset=lc, context='dataset', **_skip_filter_checks))
            # fluxes = self._bundle.get_value(qualifier='fluxes', dataset=lc, context='dataset', **_skip_filter_checks)
            phases = ps.get_value(qualifier='input_phases', **_skip_filter_checks)
            fluxes = ps.get_value(qualifier='input_fluxes', **_skip_filter_checks)
            sigmas = ps.get_value(qualifier='input_sigmas', **_skip_filter_checks)
            kwargs['plot_package'] = 'autofig'
            kwargs['autofig_method'] = 'plot'
            kwargs['x'] = phases
            kwargs['xlabel'] = 'phase'
            kwargs['y'] = fluxes
            kwargs['yerror'] = sigmas if len(sigmas) else None
            kwargs['ylabel'] = 'flux (normalized)'
            kwargs['marker'] = '.'
            kwargs['linestyle'] = 'None'
            kwargs['c'] = 'gray'

            def _phase_wrap(phase):
                if phase < -0.5:
                    phase += 1
                return phase

            addl_kwargss = []

            analytic_phases = ps.get_value(qualifier='analytic_phases', defualt=None, **_skip_filter_checks)
            if analytic_phases is not None:
                analytic_fluxes_dict = ps.get_value(qualifier='analytic_fluxes', **_skip_filter_checks)
                analytic_best_model = ps.get_value(qualifier='analytic_best_model', **_skip_filter_checks)
                analytic_fluxes = analytic_fluxes_dict[analytic_best_model]
                addl_kwargss += [{'plot_package': 'autofig', 'autofig_method': 'plot', 'x': analytic_phases, 'y': analytic_fluxes, 'marker': 'None', 'c': 'k', 'linestyle': 'solid', 's': 0.04, 'label': analytic_best_model}]

            ecl_edges = ps.get_value(qualifier='eclipse_edges', **_skip_filter_checks)

            pcolor = self._bundle.get_value(qualifier='color', component=primary, default='blue', **_skip_filter_checks)
            addl_kwargss += [{'plot_package': 'autofig', 'autofig_method': 'plot', 'axvline': True, 'x': [_phase_wrap(phase)], 'xlabel': 'phase', 'c': pcolor, 'linestyle': 'dashed'} for phase in ecl_edges[:2]]
            addl_kwargss += [{'plot_package': 'autofig', 'autofig_method': 'plot', 'axvline': True, 'x': [_phase_wrap(ps.get_value(qualifier='primary_phase', **_skip_filter_checks))], 'xlabel': 'phase', 'c': pcolor, 'label': 'primary ({}) eclipse'.format(primary) if primary!='primary' else 'primary eclipse', 'linestyle': 'solid'}]

            scolor = self._bundle.get_value(qualifier='color', component=secondary, default='orange', **_skip_filter_checks)
            addl_kwargss += [{'plot_package': 'autofig', 'autofig_method': 'plot', 'axvline': True, 'x': [_phase_wrap(phase)], 'xlabel': 'phase', 'c': scolor, 'linestyle': 'dashed'} for phase in ecl_edges[2:]]
            addl_kwargss += [{'plot_package': 'autofig', 'autofig_method': 'plot', 'axvline': True, 'x': [_phase_wrap(ps.get_value(qualifier='secondary_phase', **_skip_filter_checks))], 'xlabel': 'phase', 'c': scolor, 'label': 'secondary ({}) eclipse'.format(secondary) if secondary!='secondary' else 'secondary eclipse', 'linestyle': 'solid'}]


            # for model, analytic_fluxes in analytic_fluxes_dict.items():
            #     addl_kwargss += [{'plot_package': 'autofig', 'autofig_method': 'plot', 'x': phases, 'y': analytic_fluxes, 'z': 1, 'marker': 'None', 'c': None if model==analytic_best_model else 'k', 'linestyle': 'solid' if model==analytic_best_model else 'dotted', 's': 0.04 if model==analytic_best_model else 0.02, 'label': model}]



            return [kwargs] + addl_kwargss

        elif ps.kind == 'rv_geometry':
            orbit = ps.get_value(qualifier='orbit', **_skip_filter_checks)
            primary, secondary = self._bundle.hierarchy.get_children_of(orbit)

            kwargs['xlabel'] = 'phase'
            kwargs['ylabel'] = 'RVs'
            kwargs['yunit'] = 'km/s'
            kwargs['plot_package'] = 'autofig'
            kwargs['autofig_method'] = 'plot'

            kwargss = [_deepcopy(kwargs), _deepcopy(kwargs), _deepcopy(kwargs), _deepcopy(kwargs)]
            for i,comp in enumerate([primary, secondary]):
                phases = ps.get_value(qualifier='input_phases', component=comp, **_skip_filter_checks)
                input_rvs = ps.get_value(qualifier='input_rvs', component=comp, unit='km/s', **_skip_filter_checks)
                input_sigmas = ps.get_value(qualifier='input_sigmas', component=comp, **_skip_filter_checks)

                analytic_phases = ps.get_value(qualifier='analytic_phases', default=[], **_skip_filter_checks)
                analytic_rvs = ps.get_value(qualifier='analytic_rvs', component=comp, default=[], unit='km/s', **_skip_filter_checks)

                kwargss[i]['x'] = phases
                kwargss[i+2]['x'] = analytic_phases
                kwargss[i]['y'] = input_rvs
                kwargss[i]['yerror'] = input_sigmas if len(input_sigmas) else None
                kwargss[i+2]['y'] = analytic_rvs
                kwargss[i]['marker'] = '.'
                kwargss[i+2]['marker'] = 'None'
                kwargss[i]['linestyle'] = 'None'
                kwargss[i+2]['linestyle'] = 'solid'
                kwargss[i]['color'] = 'gray'
                kwargss[i+2]['color'] = self._bundle.get_value(qualifier='color', component=comp, default=['blue', 'orange'][i])

            return kwargss

        elif ps.kind == 'ebai':
            orbit = ps.get_value(qualifier='orbit', **_skip_filter_checks)
            primary, secondary = self._bundle.hierarchy.get_children_of(orbit)

            input_phases = ps.get_value(qualifier='input_phases', **_skip_filter_checks)
            input_fluxes = ps.get_value(qualifier='input_fluxes', **_skip_filter_checks)
            input_sigmas = ps.get_value(qualifier='input_sigmas', **_skip_filter_checks)
            ebai_phases = ps.get_value(qualifier='ebai_phases', **_skip_filter_checks)
            ebai_fluxes = ps.get_value(qualifier='ebai_fluxes', **_skip_filter_checks)

            kwargs['plot_package'] = 'autofig'
            kwargs['autofig_method'] = 'plot'
            kwargs['xlabel'] = 'phase'
            kwargs['ylabel'] = 'flux (normalized)'

            kwargss = [_deepcopy(kwargs), _deepcopy(kwargs)]

            kwargss[0]['x'] = input_phases
            kwargss[1]['x'] = ebai_phases
            kwargss[0]['y'] = input_fluxes
            kwargss[1]['y'] = ebai_fluxes
            kwargss[0]['yerror'] = input_sigmas if len(input_sigmas) else None
            kwargss[0]['z'] = 0.0
            kwargss[1]['z'] = 1.0  # force ebai model on top of data
            kwargss[0]['marker'] = '.'
            kwargss[1]['marker'] = '+'
            kwargss[0]['linestyle'] = 'None'
            kwargss[1]['linestyle'] = 'solid'
            kwargss[0]['color'] = 'gray'
            kwargss[1]['color'] = 'blue'
            kwargss[0]['label'] = 'observations'
            kwargss[1]['label'] = '2 gaussian model'

            return kwargss

        elif ps.kind == 'dynesty':
            kwargs.setdefault('style', 'corner')

            adopt_inds, adopt_uniqueids = self._bundle._get_adopt_inds_uniqueids(ps, **kwargs)

            style = kwargs.get('style')
            if not isinstance(style, str):
                raise ValueError("style must be a (single) string for dynesty")

            if style != 'corner':
                kwargs['results'] = _helpers.get_dynesty_object_from_solution(ps._bundle, ps.solution, adopt_parameters=kwargs.get('adopt_parameters'))

            if style in ['corner', 'failed']:
                kwargs['plot_package'] = 'distl'
                if 'parameters' in kwargs.keys() and style=='failed':
                    raise ValueError("cannot currently plot failed_samples while providing parameters.  Pass or set adopt_parameters to plot a subset of available parameters")
                if style=='failed' and len(adopt_inds) < 2:
                    raise ValueError("cannot plot failed_samples with < 2 parameters")

                kwargs['dc'], _ = ps._bundle.get_distribution_collection(solution=ps.solution, **{k:v for k,v in kwargs.items() if k in ['distributions_convert', 'distributions_bins', 'parameters']})

                if style=='failed':
                    kwargs.setdefault('plot_uncertainties', False)
                    kwargs['failed_samples'] = {k: np.asarray(v)[:,adopt_inds] for k,v in ps.get_value(qualifier='failed_samples', **_skip_filter_checks).items()}

                return_ += [kwargs]

            elif style == 'trace':
                kwargs['plot_package'] = 'dynesty'
                kwargs['dynesty_method'] = 'traceplot'
            elif style == 'run':
                kwargs['plot_package'] = 'dynesty'
                kwargs['dynesty_method'] = 'runplot'


            else:
                raise ValueError("dynesty plots with style='{}' not recognized".format(kwargs.get('style')))
            return (kwargs,)
        elif ps.kind == 'emcee':
            kwargs.setdefault('style', ['trace', 'lnprobability'])

            adopt_inds, adopt_uniqueids = self._bundle._get_adopt_inds_uniqueids(ps, **kwargs)

            burnin = ps.get_value(qualifier='burnin', burnin=kwargs.get('burnin', None), **_skip_filter_checks)
            thin = ps.get_value(qualifier='thin', thin=kwargs.get('thin', None), **_skip_filter_checks)
            lnprob_cutoff = ps.get_value(qualifier='lnprob_cutoff', lnprob_cutoff=kwargs.get('lnprob_cutoff', None), **_skip_filter_checks)

            lnprobabilities = ps.get_value(qualifier='lnprobabilities', **_skip_filter_checks)
            samples = ps.get_value(qualifier='samples', **_skip_filter_checks)

            styles = kwargs.get('style')
            if isinstance(styles, str):
                styles = [styles]

            return_ = []
            for style in styles:
                kwargs = _deepcopy(kwargs)

                if style in ['corner', 'failed']:
                    kwargs['plot_package'] = 'distl'
                    if 'parameters' in kwargs.keys() and style=='failed':
                        raise ValueError("cannot currently plot failed_samples while providing parameters.  Pass or set adopt_parameters to plot a subset of available parameters")
                    if style=='failed' and len(adopt_inds) < 2:
                        raise ValueError("cannot plot failed_samples with < 2 parameters")

                    kwargs['dc'], _ = ps._bundle.get_distribution_collection(solution=ps.solution,
                                                                            **{k:v for k,v in kwargs.items() if k in ['burnin', 'thin', 'lnprob_cutoff', 'distributions_convert', 'distributions_bins', 'parameters', 'adopt_parameters']})

                    if style=='failed':
                        kwargs.setdefault('plot_uncertainties', False)
                        kwargs['failed_samples'] = {k: np.asarray(v)[:,adopt_inds] for k,v in ps.get_value(qualifier='failed_samples', **_skip_filter_checks).items()}

                    return_ += [kwargs]

                elif style in ['lnprobability', 'lnprob', 'lnprobabilities']:
                    kwargs['plot_package'] = 'autofig'
                    kwargs['autofig_method'] = 'plot'
                    kwargs.setdefault('marker', 'None')
                    kwargs.setdefault('linestyle', 'solid')

                    lnprobabilities_proc, samples_proc = _helpers.process_mcmc_chains(lnprobabilities, samples, burnin, thin, -np.inf, adopt_inds, flatten=False)

                    # we'll be editing items in the array, so we need to make a deepcopy first
                    lnprobabilities_proc = _deepcopy(lnprobabilities_proc)
                    lnprobabilities_proc[lnprobabilities_proc < lnprob_cutoff] = np.nan

                    for lnp in lnprobabilities_proc.T:
                        if not np.any(np.isfinite(lnp)):
                            continue

                        if np.all(np.isnan(lnp)):
                            continue

                        kwargs = _deepcopy(kwargs)
                        kwargs['x'] = np.arange(len(lnp), dtype=float)*thin+burnin
                        kwargs['xlabel'] = 'iteration (burnin={}, thin={})'.format(burnin, thin)
                        kwargs['y'] = lnp
                        kwargs['ylabel'] = 'lnprobability' if lnprob_cutoff==-np.inf else 'lnprobability (lnprob_cutoff={})'.format(lnprob_cutoff)
                        return_ += [kwargs]

                elif style in ['trace', 'walks']:
                    kwargs['plot_package'] = 'autofig'
                    if 'parameters' in kwargs.keys():
                        raise ValueError("cannot currently plot {} while providing parameters.  Pass or set adopt_parameters to plot a subset of available parameters".format(style))
                    kwargs['autofig_method'] = 'plot'
                    kwargs.setdefault('marker', 'None')
                    kwargs.setdefault('linestyle', 'solid')

                    fitted_uniqueids = self._bundle.get_value(qualifier='fitted_uniqueids', context='solution', solution=ps.solution, **_skip_filter_checks)
                    # fitted_twigs = self._bundle.get_value(qualifier='fitted_twigs', context='solution', solution=ps.solution, **_skip_filter_checks)
                    fitted_units = self._bundle.get_value(qualifier='fitted_units', context='solution', solution=ps.solution, **_skip_filter_checks)
                    fitted_ps = self._bundle.filter(uniqueid=list(adopt_uniqueids), **_skip_filter_checks)
                    lnprobabilities_proc, samples_proc = _helpers.process_mcmc_chains(lnprobabilities, samples, burnin, thin, lnprob_cutoff, adopt_inds, flatten=False)

                    # samples [niters, nwalkers, parameter]
                    # allow user override of which parameter(s) to include
                    # but in order to handle the possibility of indexes in array parameters
                    # we need to find the matches in adopt_uniqueids which includes the index
                    if kwargs.get('y', None):
                        y = kwargs.get('y')
                        if isinstance(ys, str):
                            ys = [ys]

                        # ys are currently assumed to twigs (with or without indices)
                        # we need a list of uniqueids, including indices when necessary
                        def _uniqueids_for_y(fitted_ps, twig=None):
                            y, index = _extract_index_from_string(y)
                            p = fitted_ps.get_parameter(twig=y, **_skip_filter_checks)
                            if index is None:
                                if p.__class__.__name__ == 'FloatArrayParameter':
                                    return ['{}[{}]'.format(p.uniqueid, i) for i in range(len(p.get_value()))]
                                else:
                                    return [p.uniqueid]
                            else:
                                return ['{}[{}]'.format(p.uniqueid, index)]

                        plot_uniqueids = []
                        for y in ys:
                            plot_uniqueids += _uniqueids_for_y(fitted_ps, y)

                    else:
                        plot_uniqueids = adopt_uniqueids

                    for plot_uniqueid in plot_uniqueids:
                        parameter_ind = list(adopt_uniqueids).index(plot_uniqueid)
                        _, index = _extract_index_from_string(plot_uniqueid)
                        yparam = fitted_ps.get_parameter(uniqueid=plot_uniqueid, **_skip_filter_checks)

                        for walker_ind in range(samples_proc.shape[1]):
                            kwargs = _deepcopy(kwargs)

                            # this needs to be the unflattened version
                            samples_y = samples_proc[:, walker_ind, parameter_ind]

                            kwargs['x'] = np.arange(len(samples_y), dtype=float)*thin+burnin
                            kwargs['xlabel'] = 'iteration (burnin={}, thin={}, lnprob_cutoff={})'.format(burnin, thin, lnprob_cutoff)

                            kwargs['y'] = samples_y
                            kwargs['ylabel'] = _corner_label(yparam, index=index)
                            # TODO: use fitted_units instead?
                            kwargs['yunit'] = fitted_units[parameter_ind]
                            return_ += [kwargs]
                else:
                    raise NotImplementedError()

            return return_

        elif ps.kind == 'mesh':
            if mesh_all_cartesian:
                kwargs['autofig_method'] = 'mesh'
            else:
                kwargs['autofig_method'] = 'plot'

            if self.time is not None:
                kwargs['i'] = float(self.time) * u.d
        else:
            kwargs['autofig_method'] = 'plot'  # may use 'fill_between' later

        #### GET DATA ARRAY FOR EACH AUTOFIG "DIRECTION"
        for af_direction in ['x', 'y', 'z', 'c', 's', 'fc', 'ec']:
            # set the array and dimension label
            # logger.debug("af_direction={}, kwargs={}, defaults={}".format(af_direction, kwargs, defaults))
            if af_direction not in kwargs.keys() and af_direction in defaults.keys():
                # don't want to use setdefault here because we don't want an
                # entry if the af_direction is not in either dict
                kwargs[af_direction] = defaults[af_direction]

            # logger.debug("_kwargs_fill_dimension {} {} {}".format(kwargs, af_direction, ps.twigs))
            kwargs = _kwargs_fill_dimension(kwargs, af_direction, ps)
            if af_direction == 'y' and len(kwargs.get('y', np.asarray([])).shape) > 1:
                if '-sigma' in self._bundle.get_value(qualifier='sample_mode', context='model', model=ps.model, default='none', **_skip_filter_checks):
                    kwargs['autofig_method'] = 'fill_between'
                    kwargs['y'] = np.asarray(kwargs['y'].value).T * kwargs['y'].unit
                    kwargs['yunit'] = kwargs['y'].unit
                    kwargs['xunit'] = kwargs['x'].unit
            if kwargs is None:
                # cannot plot
                logger.warning("cannot plot {}-dimension of {}@{}, skipping".format(af_direction, ps.component, ps.dataset))
                return []

        #### HANDLE AUTOFIG'S INDENPENDENT VARIABLE DIRECTION (i)
        # try to find 'times' in the cartesian dimensions:
        if 'phases' not in [_singular_to_plural_get(kwargs['{}qualifier'.format(af_direction)].split(':')[0]) for af_direction in ['x', 'y', 'z'] if isinstance(kwargs.get('{}qualifier'.format(af_direction), None), str)]:
            iqualifier_default = 'times'
        elif kwargs.get('yqualifier') == 'residuals' and kwargs.get('additional_calls', {}).get('y', None) == 'residuals_spread':
            # then we won't be using linestyle by default and we want to be
            # compatible with an axis that does residuals_spread
            iqualifier_default= 'times'
        elif self._bundle.hierarchy.is_time_dependent():
            if 'i' not in kwargs.keys():
                logger.warning("defaulting to i='times' to plot in time-order because system is time_dependent.  Pass i='phases' to override.")
            iqualifier_default = 'times'
        else:
            iqualifier_default = 'phases'

        iqualifier = kwargs.pop('i', iqualifier_default)
        for af_direction in ['x', 'y', 'z']:
            if ps.kind != 'mesh' and (kwargs.get('{}label'.format(af_direction), None) in ['times', 'time_ecls'] if iqualifier=='times' else [iqualifier]):
                kwargs['i'] = af_direction
                kwargs['iqualifier'] = None
                break
        else:
            # then we didn't find a match, so we'll either pass the time
            # (for a mesh) or times array (otherwise)
            if ps.time is not None:
                # a single mesh will pass just that single time on as the
                # independent variable/direction
                if iqualifier=='times':
                    kwargs['i'] = float(ps.time) * u.d
                    kwargs['iqualifier'] = 'ps.times'
                elif _instance_in(iqualifier, float, u.Quantity):
                    kwargs['i'] = iqualifier
                    kwargs['iqualifier'] = iqualifier
                elif isinstance(iqualifier, str) and iqualifier.split(':')[0] == 'phases':
                    # TODO: need to test this
                    component = iqualifier.split(':')[1] if len(iqualifier.split(':')) > 1 else None
                    # TODO: take t0 and period strings
                    kwargs['i'] = self._bundle.to_phase(float(ps.time), component=component)
                    kwargs['iqualifier'] = iqualifier
                else:
                    raise NotImplementedError
            elif ps.kind == 'etv':
                if iqualfier=='times':
                    kwargs['i'] = ps.get_quantity(qualifier='time_ecls', **_skip_filter_checks)
                    kwargs['iqualifier'] = 'time_ecls'
                elif iqualifier.split(':')[0] == 'phases':
                    # TODO: need to test this
                    icomponent = iqualifier.split(':')[1] if len(iqualifier.split(':')) > 1 else None
                    kwargs['i'] = self._bundle.to_phase(ps.get_quantity(qualifier='time_ecls'), component=icomponent, **_skip_filter_checks)
                    kwargs['iqualifier'] = iqualifier
                else:
                    raise NotImplementedError
            else:
                if iqualifier=='times':
                    kwargs['i'] = _handle_mask(ps, ps.get_quantity(qualifier='times', **_skip_filter_checks), **kwargs)
                    kwargs['iqualifier'] = 'times'
                elif iqualifier.split(':')[0] == 'phases':
                    # TODO: need to test this
                    icomponent = iqualifier.split(':')[1] if len(iqualifier.split(':')) > 1 else None
                    times = _handle_mask(ps, ps.get_quantity(qualifier='times', **_skip_filter_checks), **kwargs)
                    # TODO: take period and t0 strings
                    kwargs['i'] = self._bundle.to_phase(times, component=icomponent)
                    kwargs['iqualifier'] = iqualifier
                else:
                    raise NotImplementedError

        #### STYLE DEFAULTS
        # set defaults for marker/linestyle depending on whether this is
        # observational or synthetic data
        if ps.context == 'dataset':
            kwargs.setdefault('linestyle', 'none')
        elif ps.context == 'model':
            if ps.kind == 'mesh' and kwargs['autofig_method'] == 'plot':
                kwargs.setdefault('marker', '^')
                kwargs.setdefault('linestyle', 'none')
            else:
                kwargs.setdefault('marker', 'none')

        # set defaults for colormap and symmetric limits
        for af_direction in ['c', 'fc', 'ec']:
            qualifier = kwargs.get('{}qualifier'.format(af_direction), '').split('@')[0]
            if qualifier in ['rvs']:
                kwargs.setdefault('{}map'.format(af_direction), 'RdBu_r')
                if kwargs['{}map'.format(af_direction)] == 'RdBu_r':
                    # only apply symmetric default if taking the colormap default
                    kwargs.setdefault('{}lim'.format(af_direction), 'symmetric')
            elif qualifier in ['vxs', 'vys', 'vzs', 'vus', 'vvs', 'vws']:
                kwargs.setdefault('{}map'.format(af_direction), 'RdBu')
                if kwargs['{}map'.format(af_direction)] == 'RdBu':
                    # only apply symmetric default if taking the colormap default
                    kwargs.setdefault('{}lim'.format(af_direction), 'symmetric')
                kwargs.setdefault('{}lim'.format(af_direction), 'symmetric')
            elif qualifier in ['teffs']:
                kwargs.setdefault('{}map'.format(af_direction), 'afmhot')
            elif qualifier in ['loggs']:
                kwargs.setdefault('{}map'.format(af_direction), 'gnuplot')
            elif qualifier in ['visibilities']:
                kwargs.setdefault('{}map'.format(af_direction), 'RdYlGn')
                kwargs.setdefault('{}lim'.format(af_direction), (0,1))

        #### LABEL FOR LEGENDS
        attrs = ['component', 'dataset']
        if ps._bundle is not None and len(ps._bundle.models) > 1:
            attrs += ['model']
        default_label = '@'.join([getattr(ps, attr) for attr in attrs if getattr(ps, attr) is not None])
        kwargs.setdefault('label', default_label)

        return (kwargs,)

    def gcf(self):
        """
        Get the active current autofig Figure.

        See also:
        * <phoebe.parameters.ParameterSet.plot>
        * <phoebe.parameters.ParameterSet.show>
        * <phoebe.parameters.ParameterSet.savefig>
        * <phoebe.parameters.ParameterSet.clf>
        """
        if self._bundle is None:
            return autofig.gcf()

        if self._bundle._af_figure is None:
            self._bundle._af_figure = autofig.Figure()

        return self._bundle._af_figure

    def clf(self):
        """
        Clear/reset the active current autofig Figure.

        See also:
        * <phoebe.parameters.ParameterSet.plot>
        * <phoebe.parameters.ParameterSet.show>
        * <phoebe.parameters.ParameterSet.savefig>
        * <phoebe.parameters.ParameterSet.gcf>
        """
        if self._bundle is None:
            raise ValueError("could not find parent Bundle object")

        self._bundle._af_figure = None

    def plot(self, twig=None, **kwargs):
        """
        High-level wrapper around matplotlib that uses
        [autofig 1.1.0](https://autofig.readthedocs.io/en/1.2.0)
        under-the-hood for automated figure and animation production.

        For an even higher-level interface allowing to interactively set and
        save plotting options see:
        * <phoebe.frontend.bundle.Bundle.add_figure>
        * <phoebe.frontend.bundle.Bundle.run_figure>

        In general, `run_figure` is useful for creating simple plots with
        consistent defaults for styling across datasets/components/etc,
        when plotting from a UI, or when wanting to save plotting options
        along with the bundle rather than in a script.  `plot` is more
        more flexible, allows for multiple subplots and advanced positioning,
        and is less clumsy if plotting from the python frontend.

        See also:
        * <phoebe.parameters.ParameterSet.show>
        * <phoebe.parameters.ParameterSet.savefig>
        * <phoebe.parameters.ParameterSet.gcf>
        * <phoebe.parameters.ParameterSet.clf>

        All keyword arguments also support passing dictionaries.  In this case,
        they are applied to any resulting plotting call in which the dictionary
        matches (including support for wildcards) to the tags of the respective
        ParameterSet.  For example:

        ```
        plot(c={'primary@rv*': 'blue', 'secondary@rv*': 'red'})
        ```

        Note: not all options are listed below.  See the
        [autofig](https://autofig.readthedocs.io/en/1.2.0/)
        tutorials and documentation for more options which are passed along
        via `**kwargs`.

        Arguments
        ----------
        * `twig` (string, optional, default=None): twig to use for filtering
            prior to plotting.  See <phoebe.parameters.ParameterSet.filter>
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
        * `period` (string/float, optional): qualifier/twig or float of the period that
            should be used for phasing, if applicable.  If provided as a string,
            `b.get_value(period)` needs to provide a valid float.  This is used
            if `phase`/`phases` provided instead of `time`/`times` as well as
            if 'phases' is set as any direction (`x`, `y`, `z`, etc).
            Passed directly to <phoebe.frontend.bundle.Bundle.to_phase>.
        * `dpdt` (string/float, optional): qualifier/twig or float of the dpdt that
            should be used for phasing, if applicable.  If provided as a string,
            `b.get_value(dpdt)` needs to provide a valid float.  This is used
            if `phase`/`phases` provided instead of `time`/`times` as well as
            if 'phases' is set as any direction (`x`, `y`, `z`, etc).
            Passed directly to <phoebe.frontend.bundle.Bundle.to_phase>.
        * `t0` (string/float, optional): qualifier/twig or float of the t0 that
            should be used for phasing, if applicable.  If provided as a string,
            `b.get_value(t0)` needs to provide a valid float.  This is used
            if `phase`/`phases` provided instead of `time`/`times` as well as
            if 'phases' is set as any direction (`x`, `y`, `z`, etc).
            Passed directly to <phoebe.frontend.bundle.Bundle.to_phase>.
        * `phase` (float, optional): phase to use for plotting/animating.  This
            will convert to `time` using the current ephemeris via
            <phoebe.frontend.bundle.Bundle.to_time> along with the passed value
            of `t0`.  If `time` and `phase` are both provided, an error will be
            raised.  Note: if a dataset uses compute_phases_t0 that differs
            from `t0`, this may result in a different mapping between
            `phase` and `time`.
        * `phases` (list/array, optional): phases to use for animating.  This
            will convert to `times` using the current ephemeris via
            <phoebe.frontend.bundle.Bundle.to_time> along with the passed
            value of `t0`.  If `times` and `phases` are both provided, an error
            will be raised.  Note: if a dataset uses compute_phases_t0 that differs
            from `t0`, this may result in a different mapping between
            `phase` and `time`.

        * `x` (string/float/array, optional): qualifier/twig of the array to plot on the
            x-axis (will default based on the dataset-kind if not provided).
            With the exception of phase, `b.get_value(x)` needs to provide a
            valid float or array.  To plot phase along the x-axis, pass
            `x='phases'` or `x='phases:[component]'`.  This will use the ephemeris
            from <phoebe.frontend.bundle.Bundle.get_ephemeris>(component) if
            possible to phase the applicable times array.
        * `y` (string/float/array, optional): qualifier/twig of the array to plot on the
            y-axis (will default based on the dataset-kind if not provided).  To
            plot residuals along the y-axis, pass `y='residuals'`.  This will
            call <phoebe.frontend.bundle.Bundle.calculate_residuals> for the given
            dataset/model.
        * `z` (string/float/array, optional): qualifier/twig of the array to plot on the
            z-axis.  By default, this will just order the points on a 2D plot.
            To plot in 3D, also pass `projection='3d'`.
        * `s` (strong/float/array, optional): qualifier/twig of the array to use
            for size.  See the [autofig tutorial on size](https://autofig.readthedocs.io/en/1.2.0/tutorials/size_modes/)
            for more information.
        * `smode` (string, optional): mode for handling size (`s`).  See the
            [autofig tutorial on size mode](https://autofig.readthedocs.io/en/1.2.0/tutorials/size_modes/)
            for more information.
        * `c` (string/float/array, optional): qualifier/twig of the array to use
            for color.
        * `fc` (string/float/array, optional): qualifier/twig of the array to use
            for facecolor (only applicable for mesh plots).
        * `ec` (string/float/array, optional): qualifier/twig of the array to use
            for edgecolor (only applicable for mesh plots).   To disable plotting
            edges, use `ec='none'`.  To plot edges in the same colors as the face,
            use `ec='face'` (not supported if `projection='3d'`).

        * `i` (string, optional, default='phases' or 'times'): qualifier/twig to
            use for the independent variable.  In the vast majority of cases,
            using the default is sufficient.  `i` will default to 'times' unless
            'phases' is plotted along `x`, `y`, or `z`.  If 'phases' is plotted,
            then `i` will still default to 'times' if the system is time-dependent,
            according to <phoebe.parameters.HierarchyParameter.is_time_dependent>
            (note that this is determined based on current values of the relevant
            parameters, not neccessarily those when the model was computed),
            otherwise will default to 'phases'.  If `x` is 'phases' or ('phases:[component]'),
            then setting `i` to phases will sort and connect the points in
            phase-order, whereas if set to `times` they will be sorted and connected
            in time-order, with linebreaks when needed for phase-wrapping.
            See also the [autofig tutorial on a looping independent variable](https://autofig.readthedocs.io/en/1.2.0/gallery/looping_indep/).

        * `xerror` (string/float/array, optional): qualifier/twig of the array to plot as
            x-errors (will default based on `x` if not provided).  Pass None to
            disable plotting xerrors.
        * `yerror` (string/float/array, optional): qualifier/twig of the array to plot as
            y-errors (will default based on `y` if not provided).  Pass None to
            disable plotting yerrors.
        * `zerror` (string/float/array, optional): qualifier/twig of the array to plot as
            z-errors (will default based on `z` if not provided).  Pass None to
            disable plotting zerrors.

        * `xunit` (string/unit, optional): unit to plot on the x-axis (will
            default on `x` if not provided).
        * `yunit` (string/unit, optional): unit to plot on the y-axis (will
            default on `y` if not provided).
        * `zunit` (string/unit, optional): unit to plot on the z-axis (will
            default on `z` if not provided).
        * `cunit` (string/unit, optional): unit to plot on the color-axis (will
            default on `c` if not provided).
        * `fcunit` (string/unit, optional): unit to plot on the facecolor-axis (will
            default on `fc` if not provided, only applicable for mesh plots).
        * `ecunit` (string/unit, optional): unit to plot on the edgecolor-axis (will
            default on `ec` if not provided, only applicable for mesh plots).

        * `xlabel` (string, optional): label for the x-axis (will default on `x`
            if not provided, but will not set if the axes already has an xlabel).
        * `ylabel` (string, optional): label for the y-axis (will default on `y`
            if not provided, but will not set if the axes already has an ylabel).
        * `zlabel` (string, optional): label for the z-axis (will default on `z`
            if not provided, but will not set if the axes already has an zlabel).
        * `slabel` (string, optional): label for the size-axis (will default on `s`
            if not provided, but will not set if the axes already has an slabel).
        * `clabel` (string, optional): label for the color-axis (will default on `c`
            if not provided, but will not set if the axes already has an clabel).
        * `fclabel` (string, optional): label for the facecolor-axis (will default on `fc`
            if not provided, but will not set if the axes already has an fclabel,
            only applicable for mesh plots).
        * `eclabel` (string, optional): label for the edgecolor-axis (will default on `ec`
            if not provided, but will not set if the axes already has an eclabel,
            only applicable for mesh plots).

        * `xlim` (tuple/string, optional): limits for the x-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.
        * `ylim` (tuple/string, optional): limits for the y-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.
        * `zlim` (tuple/string, optional): limits for the z-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.
        * `slim` (tuple/string, optional): limits for the size-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.
        * `clim` (tuple/string, optional): limits for the color-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.
        * `fclim` (tuple/string, optional): limits for the facecolor-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.
        * `eclim` (tuple/string, optional): limits for the edgecolor-axis (will default on
            data if not provided).  See [autofig tutorial on limits](https://autofig.readthedocs.io/en/1.2.0/tutorials/limits/)
            for more information/choices.

        * `fcmap` (string, optional): colormap to use for the facecolor-axis (will default on
            the type of data passed to `fc` if not provided, only applicable for mesh plots).
            See the [matplotlib colormap reference](https://matplotlib.org/3.1.0/gallery/color/colormap_reference.html)
            for a list of options (may vary based on installed version of matplotlib).
        * `ecmap` (string, optional): colormap to use for the edgecolor-axis (will default on
            the type of data passed to `ec` if not provided, only applicable for mesh plots).
            See the [matplotlib colormap reference](https://matplotlib.org/3.1.0/gallery/color/colormap_reference.html)
            for a list of options (may vary based on installed version of matplotlib).

        * `smode` (string, optional): size mode.  See the [autofig tutorial on sizes](https://autofig.readthedocs.io/en/1.2.0/tutorials/size_modes/)
            for more information.

        * `highlight` (bool, optional, default=True): whether to highlight at the
            current time.  Only applicable if `time` or `times` provided.
        * `highlight_marker` (string, optional): marker to use for highlighting.
            Only applicable if `highlight=True` and `time` or `times` provided.
        * `highlight_color` (string, optional): color to use for highlighting.
            Only applicable if `highlight=True` and `time` or `times` provided.
        * `highlight_size` (int, optional): size to use for highlighting.
            Only applicable if `highlight=True` and `time` or `times` provided.

        * `uncover` (bool, optional): whether to uncover data based on the current
            time.  Only applicable if `time` or `times` provided.
        * `trail` (bool or float, optional): whether trail is enabled.
            If a float, then a value between 0 and 1 indicating the fractional
            length of the trail.  Defaults to 0 for mesh and lineprofiles and False
            otherwise.  Only applicable if `times` or `times` provided.

        * `legend` (bool, optional, default=False): whether to draw a legend for
            this axes.
        * `legend_kwargs` (dict, optional):  keyword arguments (position,
            formatting, etc) to be passed on to [plt.legend](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html)

        * `fig` (matplotlib figure, optional): figure to use for plotting.  If
            not provided, will use `plt.gcf()`.  Ignored unless `save`, `show`,
            or `animate`.

        * `save` (string, optional, default=False): filename to save the
            figure (or False to not save).
        * `show` (bool, optional, default=False): whether to show the plot

        * `animate` (bool, optional, default=False): whether to animate the figure.
        * `interval` (int, optional, default=100): time in ms between each
            frame in the animation.  Applicable only if `animate` is True.
        * `animate_callback` (callable, optional, default=None): Function which
            takes the matplotlib figure object and will be called at each frame
            within the animation.

        * `equal_aspect` (optional): whether to force the aspect ratio of the
            axes to be equal.  If not provided, this will default to True if
            all directions (i.e. `x` and `y` for `projection='2d'` or `x`,
            `y`, and `z` for '3d') are positions and of the same units, or
            False otherwise.
        * `pad_aspect` (optional): whether to achieve the equal aspect ratio
            by padding the limits instead of whitespace around the axes.  Only
            applicable if `equal_aspect` is True.  If not provided, this will
            default to True unless `animate` is True, in which case it will
            default to False (as autofig cannot currently handle `pad_aspect`)
            in animations.

        * `projection` (string, optional, default='2d'): whether to plot
            on a 2d or 3d axes.  If '3d', the orientation of the axes will
            be provided by `azim` and `elev` (see [autofig tutorial on 3d](https://autofig.readthedocs.io/en/1.2.0/tutorials/3d/))
        * `azim` (float or list, optional): azimuth to use when `projection`
            is '3d'.  If `animate` is True, then a tuple or list will allow
            rotating the axes throughout the animation (see [autofig tutorial on 3d](https://autofig.readthedocs.io/en/1.2.0/tutorials/3d/))
        * `elev` (float or list, optional): elevation to use when `projection`
            is '3d'.  If `animate` is True, then a tuple or list will allow
            rotating the axes throughout the animation (see [autofig tutorial on 3d](https://autofig.readthedocs.io/en/1.2.0/tutorials/3d/))
        * `exclude_back` (bool, optional): whether to exclude plotting the back
            of meshes when in '2d' projections.  Defaults to True if `fc` is
            not 'none' (otherwise defaults to False so that you can "see through"
            the star).

        * `draw_sidebars` (bool, optional, default=False): whether to include
            any applicable sidebars (colorbar, sizebar, etc).
        * `draw_title` (bool, optional, default=False): whether to draw axes
            titles.
        * `subplot_grid` (tuple, optional, default=None): override the subplot
            grid used (see [autofig tutorial on subplots](https://autofig.readthedocs.io/en/1.2.0/tutorials/subplot_positioning/)
            for more details).

        * `save_kwargs` (dict, optional): any kwargs necessary to pass on to
            save (only applicable if `animate=True`).  On many systems,
            it may be necessary to pass `save_kwargs={'writer': 'imagemagick'}`.

        * `**kwargs`: additional keyword arguments are sent along to [autofig](https://autofig.readthedocs.io/en/1.2.0/).

        Returns
        --------
        * (autofig figure, matplotlib figure)

        Raises
        ------------
        * ValueError: if both `time` and `phase` or `times` and `phases` are passed.
        * ValueError: if the resulting figure is empty.
        """
        if not _use_autofig:
            if os.getenv('PHOEBE_ENABLE_PLOTTING', 'TRUE').upper() != 'TRUE':
                raise ImportError("cannot plot because PHOEBE_ENABLE_PLOTTING environment variable is disabled")
            else:
                raise ImportError("autofig not imported, cannot plot")

        # since we used the args trick above, all other options have to be in kwargs
        fig = kwargs.pop('fig', None)
        save = kwargs.pop('save', False)
        show = kwargs.pop('show', False)
        tight_layout = kwargs.pop('tight_layout', False)
        draw_sidebars = kwargs.pop('draw_sidebars', False)
        draw_title = kwargs.pop('draw_title', False)
        subplot_grid = kwargs.pop('subplot_grid', None)
        animate = kwargs.pop('animate', False)
        animate_callback = kwargs.pop('animate_callback', None)

        if kwargs.get('projection', '2d') == '3d' and kwargs.get('ec', None) =='face':
            raise ValueError("projection='3d' and ec='face' do not work together.  Consider ec='none' instead.")

        if 'phase' in kwargs.keys():
            if 'time' in kwargs.keys():
                raise ValueError("cannot pass both time and phase")

            period = kwargs.get('period', 'period')
            dpdt = kwargs.get('dpdt', 'dpdt')
            t0 = kwargs.get('t0', 't0_supconj')
            logger.info("converting from phase to time with period={}, dpdt={}, t0={}".format(period, dpdt, t0))
            kwargs['time'] = self._bundle.to_time(kwargs.pop('phase'), period=period, dpdt=dpdt, t0=t0)

        if 'phases' in kwargs.keys():
            if 'times' in kwargs.keys():
                raise ValueError("cannot pass both times and phases")

            period = kwargs.get('period', 'period')
            dpdt = kwargs.get('dpdt', 'dpdt')
            t0 = kwargs.get('t0', 't0_supconj')
            logger.info("converting from phases to times with period={}, dpdt={}, t0={}".format(period, dpdt, t0))
            kwargs['times'] = self._bundle.to_time(kwargs.pop('phases'), period=period, dpdt=dpdt, t0=t0)

        if 'times' in kwargs.keys() and not animate:
            if kwargs.get('time', None) is not None:
                logger.warning("ignoring 'times' in favor of 'time'")
            else:
                logger.warning("assuming you meant 'time' instead of 'times' since animate=False")
                kwargs['time'] = kwargs.pop('times')
        elif 'time' in kwargs.keys() and animate:
            if kwargs.get('times', None) is not None:
                logger.warning("value passed for time will still be used for filtering, despite 'times' being passed.")
            else:
                logger.warning("value passed for time will still be used for filtering, but will also be assumed as 'times' since animate=True.")
                kwargs['times'] = kwargs['time']

        time = kwargs.get('time', None)  # don't pop since time may be used for filtering


        if twig is not None:
            kwargs['twig'] = twig

        def _plot_failed_samples(mplfig, failed_samples):
            mplaxes = mplfig.axes

            for msgi, (msg, samples) in enumerate(failed_samples.items()):
                samples = np.asarray(samples)
                color = _phoebecolors[msgi+1]

                # print(msg, samples.shape)
                for axi, ax in enumerate(mplaxes):
                    axix = int(axi % sqrt(len(mplaxes)))
                    axiy = int(axi / sqrt(len(mplaxes)))
                    if axix < axiy:
                        # print("axix: {}, axiy: {}, samples[:,axix].shape: {}, samples[:,axiy].shape: {}".format(axix, axiy, samples[:,axix].shape, samples[:,axiy].shape))
                        ax.plot(samples[:,axix], samples[:,axiy], marker='x', linestyle='none', color=color, label=msg)

            # may need to reset the axes limits that were defined by corner
            xlims = {}
            for axi, ax in enumerate(mplaxes):
                axix = int(axi % sqrt(len(mplaxes)))
                axiy = int(axi / sqrt(len(mplaxes)))
                if axix < axiy:
                    ax.autoscale(enable=True, tight=True)
                    if axix not in xlims.keys():
                        xlims[axix] = ax.get_xlim()
                    if axiy not in xlims.keys():
                        xlims[axiy] = ax.get_ylim()

            # and one final loop to apply the same xlims to the hists on the diagonal
            for axi, ax in enumerate(mplaxes):
                axix = int(axi % sqrt(len(mplaxes)))
                axiy = int(axi / sqrt(len(mplaxes)))
                if axix==axiy:
                    ax.set_xlim(xlims[axix])

            # and now attempt to draw a legend in an intelligent location in the upper-right of the figure
            if len(mplaxes) > 1:
                mplaxes[int(sqrt(len(mplaxes)))].legend(loc='lower left', bbox_to_anchor=(2.1, 0.1))
            else:
                raise ValueError("cannot plot failed samples with only one axes")

            return mplfig

        try:
            if self._filter and self._bundle is not None:
                # then make sure the filter hasn't already removed hidden parameters we might need!
                ps = self._bundle.filter(check_visible=False, check_default=False, **self._filter)
            else:
                ps = self
            plot_kwargss = ps._unpack_plotting_kwargs(animate=animate, **kwargs)
            # print("*** plot_kwargss", plot_kwargss)
            # this loop handles any of the automatically-generated
            # multiple plotting calls, passing each on to autofig
            for plot_kwargs in plot_kwargss:
                plot_package = plot_kwargs.pop('plot_package', 'autofig')
                if plot_package == 'corner':
                    if not _use_corner:
                        raise ImportError("corner not imported, cannot plot")
                    if len(plot_kwargss) > 1:
                        raise ValueError("corner plots not supported with other axes")

                    mplfig = corner.corner(plot_kwargs['data'], labels=plot_kwargs.get('labels', None))

                    if 'failed_samples' in plot_kwargs.keys():
                        mplfig = _plot_failed_samples(mplfig, plot_kwargs.get('failed_samples', {}))

                    if save:
                        mplfig.savefig(save)

                    return None, mplfig

                elif plot_package == 'distl':
                    if not _use_corner:
                        raise ImportError("corner not imported, cannot plot")
                    if len(plot_kwargss) > 1:
                        # TODO: could we just return multiple figure instances?
                        raise ValueError("corner plots not supported with other axes.  Adjust the filter to include only a single distribution (including from compute or solver contexts), or to exclude all distributions.")

                    mplfig = plot_kwargs['dc'].plot(show=show, **plot_kwargs)

                    if 'failed_samples' in plot_kwargs.keys():
                        mplfig = _plot_failed_samples(mplfig, plot_kwargs.get('failed_samples', {}))

                    if save:
                        mplfig.savefig(save)

                    return None, mplfig

                elif plot_package == 'dynesty':
                    if not _use_dyplot:
                        raise ImportError("dynesty not imported, cannot plot")
                    if len(plot_kwargss) > 1:
                        # TODO: could we just return multiple figure instances?
                        raise ValueError("dynesty plots not supported with other axes.  Adjust the filter to include only a single dynesty plot or to exclude all dynesty plots.")

                    style = plot_kwargs.pop('style')
                    dynesty_method = plot_kwargs.pop('dynesty_method')
                    func = getattr(dyplot, dynesty_method)
                    mplfig, mplaxes = func(**{k:v for k,v in plot_kwargs.items() if k in ['results']})
                    if save:
                        mplfig.savefig(save)

                    return None, mplfig

                elif plot_package == 'autofig':
                    y = plot_kwargs.get('y', [])
                    axvline = plot_kwargs.pop('axvline', False)
                    if axvline or (isinstance(y, u.Quantity) and isinstance(y.value, float)) or (hasattr(y, 'value') and isinstance(y.value, float)):
                        pass
                    elif not len(y):
                        # a dataset without observational data, for example
                        continue

                    autofig_method = plot_kwargs.pop('autofig_method', 'plot')
                    # we kept the qualifiers around so we could do some default-logic,
                    # but it isn't necessary to pass them on to autofig.
                    dump = kwargs.pop('qualifier', None)
                    func = getattr(self.gcf(), autofig_method)

                    if autofig_method == 'plot' and len(np.asarray(plot_kwargs.get('y', [])).shape) > 1:
                        # then we want to loop over the y index
                        for y in plot_kwargs.get('y'):
                            func(y=y, **{k:v for k,v in plot_kwargs.items() if k!='y'})
                    else:
                        logger.info("calling autofig.{}({})".format(autofig_method, ", ".join(["{}={}".format(k,v if not isinstance(v, np.ndarray) else "<data ({} unit={})>".format(v.shape, v.unit if hasattr(v, 'unit') else None)) for k,v in plot_kwargs.items()])))

                        func(**plot_kwargs)
                else:
                    raise ValueError("plot_package={} not recognized".format(plot_package))

        except Exception as err:
            raise


        if save or show or animate:
            # NOTE: time, times, will all be included in kwargs
            try:
                return self._show_or_save(save, show, animate,
                                          fig=fig,
                                          draw_sidebars=draw_sidebars,
                                          draw_title=draw_title,
                                          tight_layout=tight_layout,
                                          subplot_grid=subplot_grid,
                                          animate_callback=animate_callback,
                                          **kwargs)
            except Exception as err:
                self.clf()
                raise
        else:
            afig = self.gcf()
            if not len(afig.axes):
                # try to detect common causes and provide useful messages
                if (kwargs.get('x', None) in ['xs', 'ys', 'zs'] and kwargs.get('y', None) in ['us', 'vs', 'ws']) or (kwargs.get('x', None) in ['us', 'vs', 'ws'] and kwargs.get('y', None) in ['xs', 'ys', 'zs']):
                    raise ValueError("cannot mix xyz and uvw coordinates when plotting")

                raise ValueError("Nothing could be found to plot.  Check all arguments.")

            fig = None

            return afig, fig

    def _show_or_save(self, save, show, animate,
                      fig=None,
                      draw_sidebars=True,
                      draw_title=True,
                      tight_layout=False,
                      subplot_grid=None,
                      **kwargs):
        """
        Draw/animate and show and/or save a autofig plot
        """
        if animate and not show and not save:
            logger.warning("setting show to True since animate=True and save not provided")
            show = True

        if animate:
            # prefer times over time
            times = kwargs.get('times', kwargs.get('time', None))
            save_kwargs = kwargs.get('save_kwargs', {})
            interval = kwargs.get('interval', 100)
            animate_callback = kwargs.get('animate_callback', None)

            if times is None:
                # then let's try to get all SYNTHETIC times
                # it would be nice to only do ENABLED, but then we have to worry about compute
                # it would also be nice to worry about models... but then you should filter first
                times_attr = []
                times_computed = []
                for dataset in self.datasets:
                    ps = self.filter(dataset=dataset, context='model')
                    if len(ps.times):
                        # for the case of meshes/spectra
                        times_attr += [float(t) for t in ps.times]
                    else:
                        for param in ps.filter(qualifier='times').to_list():
                            times_computed += list(param.get_value())

                if len(times_attr):
                    logger.info("no times were providing, so defaulting to animate over all tagged times")
                    times = sorted(list(set(times_attr)))
                else:
                    logger.info("no times were provided, so defaulting to animate over all computed times in the model")
                    times = sorted(list(set(times_computed)))

            logger.info("calling autofig.animate(i={}, draw_sidebars={}, draw_title={}, tight_layout={}, interval={}, save={}, show={}, save_kwargs={})".format(times, draw_sidebars, draw_title, tight_layout, interval, save, show, save_kwargs))

            mplanim = self.gcf().animate(i=times,
                                         draw_sidebars=draw_sidebars,
                                         draw_title=draw_title,
                                         tight_layout=tight_layout,
                                         subplot_grid=subplot_grid,
                                         animate_callback=animate_callback,
                                         interval=interval,
                                         fig=fig,
                                         save=save,
                                         show=show,
                                         save_kwargs=save_kwargs)

            afig = self.gcf()
            if not len(afig.axes):
                raise ValueError("Nothing could be found to plot.  Check all arguments.")

            # clear the autofig figure
            self.clf()

            return afig, mplanim

        else:
            time = kwargs.get('time', None)

            if isinstance(time, str):
                # TODO: need to expand this whole logic to be the same as include_times in backends.py
                time = self._bundle.get_value(time, context=['component', 'system'], check_visible=False)

            # plotting doesn't currently support highlighting at multiple times
            # if isinstance(time, list) or isinstance(time, tuple):
            #     user_time = time
            #     time = []
            #     for t in user_time:
            #         if isinstance(t, str):
            #             new_time = self.get_value(t, context=['component', 'system'], check_visible=False)
            #             if isinstance(new_time, np.ndarray):
            #                 for nt in new_time:
            #                     time.append(nt)
            #             else:
            #                 time.append(new_time)
            #         else:
            #             time.append(t)

            afig = self.gcf()
            if not len(afig.axes):
                raise ValueError("Nothing could be found to plot.  Check all arguments.")


            logger.info("calling autofig.draw(i={}, draw_sidebars={}, draw_title={}, tight_layout={}, save={}, show={})".format(time, draw_sidebars, draw_title, tight_layout, save, show))
            fig = afig.draw(i=time,
                            draw_sidebars=draw_sidebars,
                            draw_title=draw_title,
                            tight_layout=tight_layout,
                            subplot_grid=subplot_grid,
                            fig=fig,
                            save=save, show=show)

            # clear the figure so next call will start over and future shows will work
            self.clf()

            return afig, fig


    def show(self, **kwargs):
        """
        Draw and show the plot.

        See also:
        * <phoebe.parameters.ParameterSet.plot>
        * <phoebe.parameters.ParameterSet.savefig>
        * <phoebe.parameters.ParameterSet.gcf>
        * <phoebe.parameters.ParameterSet.clf>

        Arguments
        ----------
        * `show` (bool, optional, default=True): whether to show the plot
        * `save` (False/string, optional, default=False): filename to save the
            figure (or False to not save).
        * `animate` (bool, optional, default=False): whether to animate the figure.
        * `fig` (matplotlib figure, optional): figure to use for plotting.  If
            not provided, will use plt.gcf().  Ignored unless `save`, `show`,
            or `animate`.
        * `draw_sidebars` (bool, optional, default=True): whether to include
            any applicable sidebars (colorbar, sizebar, etc).
        * `draw_title` (bool, optional, default=True): whether to draw axes
            titles.
        * `subplot_grid` (tuple, optional, default=None): override the subplot
            grid used (see [autofig tutorial on subplots](https://github.com/kecnry/autofig/blob/1.0.0/tutorials/subplot_positioning.ipynb)
            for more details).
        * `time` (float, optional): time to use for plotting/animating.
        * `times` (list/array, optional): times to use for animating (will
            override any value sent to `time`).
        * `save_kwargs` (dict, optional): any kwargs necessary to pass on to
            save (only applicable if `animate=True`).

        Returns
        --------
        * (autofig figure, matplotlib figure)
        """
        kwargs.setdefault('show', True)
        kwargs.setdefault('save', False)
        kwargs.setdefault('animate', False)
        return self._show_or_save(**kwargs)

    def savefig(self, filename, **kwargs):
        """
        Draw and save the plot.

        See also:
        * <phoebe.parameters.ParameterSet.plot>
        * <phoebe.parameters.ParameterSet.show>
        * <phoebe.parameters.ParameterSet.gcf>
        * <phoebe.parameters.ParameterSet.clf>

        Arguments
        ----------
        * `save` (string): filename to save the figure (or False to not save).
        * `show` (bool, optional, default=False): whether to show the plot
        * `animate` (bool, optional, default=False): whether to animate the figure.
        * `fig` (matplotlib figure, optional): figure to use for plotting.  If
            not provided, will use plt.gcf().  Ignored unless `save`, `show`,
            or `animate`.
        * `draw_sidebars` (bool, optional, default=True): whether to include
            any applicable sidebars (colorbar, sizebar, etc).
        * `draw_title` (bool, optional, default=True): whether to draw axes
            titles.
        * `subplot_grid` (tuple, optional, default=None): override the subplot
            grid used (see [autofig tutorial on subplots](https://github.com/kecnry/autofig/blob/1.0.0/tutorials/subplot_positioning.ipynb)
            for more details).
        * `time` (float, optional): time to use for plotting/animating.
        * `times` (list/array, optional): times to use for animating (will
            override any value sent to `time`).
        * `save_kwargs` (dict, optional): any kwargs necessary to pass on to
            save (only applicable if `animate=True`).

        Returns
        --------
        * (autofig figure, matplotlib figure)
        """
        filename = os.path.expanduser(filename)
        kwargs.setdefault('show', False)
        kwargs.setdefault('save', filename)
        kwargs.setdefault('animate', False)
        return self._show_or_save(**kwargs)

class Parameter(object):
    def __init__(self, qualifier, value=None, description='', **kwargs):
        """
        This is a generic class for a Parameter.  Any Parameter that
        will actually be usable will be a subclass of this class.

        Parameters are the base of PHOEBE and hold, at the minimum,
        the value of the parameter, a description, and meta-tags
        which are used to collect and filter a list of Parameters
        inside a ParameterSet.

        Some subclasses of Parameter can add additional methods
        or attributes.  For example :class:`FloatParameter` handles
        converting units and storing a default_unit.


        Any subclass of Parameter must (at the minimum):
        - method for get_value
        - method for set_value,
        - call to set_value in the overload of __init__
        - self._dict_fields_other defined in __init__
        - self._dict_fields = _meta_fields_all + self._dict_fields_other in __init__

        Arguments
        ------------
        * `value`: value to initialize the parameter
        * `description` (string, optional): description of the parameter
        * `bundle` (<phoebe.frontend.bundle.Bundle>, optional): parent bundle
            object.
        * `uniqueid` (string, optional): uniqueid for the parameter (suggested to leave blank
            and a random string will be generated)
        * `time` (string/float, optional): value for the time tag
        * `feature` (string, optional): label for the feature tag
        * `component` (string, optional): label for the component tag
        * `dataset` (string, optional): label for the dataset tag
        * `figure` (string, optional): label for the figure tag
        * `constraint` (string, optional): label for the constraint tag
        * `compute` (string, optional): label for the compute tag
        * `model` (string, optional): label for the model tag
        * `solver` (string, optional): label for the solver tag
        * `solution` (string, optional): label for the solution tag
        * `kind` (string, optional): label for the kind tag
        * `context` (string, optional): label for the context tag
        * `copy_for` (dictionary/False, optional, default=False): dictionary of
            filter arguments for which this parameter must be copied (use with caution)
        * `visible_if` (string, optional): string to check the value of another
            parameter holding the same meta-tags (except qualifier) to determine
            whether this parameter is visible and therefore shown in filters
            (example: `visible_if='otherqualifier:True'`).  See also
            <phoebe.parameters.Parameter.is_visible>
        """

        uniqueid = str(kwargs.get('uniqueid', _uniqueid()))
        bundle = kwargs.get('bundle', None)

        self._in_constraints = []   # labels of constraints that have this parameter in the expression
        self._is_constraint = None  # label of the constraint that defines the value of this parameter

        self._description = description
        self._readonly = kwargs.get('readonly', False)
        self._advanced = kwargs.get('advanced', False)
        self._bundle = bundle
        self._value = None

        # Meta-data
        self.set_uniqueid(uniqueid)
        self._qualifier = qualifier
        self._time = kwargs.get('time', None)
        self._feature = kwargs.get('feature', None)
        self._component = kwargs.get('component', None)
        self._dataset = kwargs.get('dataset', None)
        self._figure = kwargs.get('figure', None)
        self._constraint = kwargs.get('constraint', None)
        self._distribution = kwargs.get('distribution', None)
        self._compute = kwargs.get('compute', None)
        self._model = kwargs.get('model', None)
        self._solver = kwargs.get('solver', None)
        self._solution = kwargs.get('solution', None)
        # self._plugin = kwargs.get('plugin', None)
        self._kind = kwargs.get('kind', None)
        self._context = kwargs.get('context', None)

        self._latexfmt = kwargs.get('latexfmt', None)

        # set whether new 'copies' of this parameter need to be created when
        # new objects (body components, not orbits) or datasets are added to
        # the bundle.
        self._copy_for = kwargs.get('copy_for', False)

        self._visible_if = kwargs.get('visible_if', None)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

        # loading from json can result in unicodes instead of strings - this then
        # causes problems with a lot of isinstances and string-matching.
        for attr in _meta_fields_twig + self._dict_fields_other:
            attr = '_{}'.format(attr)
            val = getattr(self, attr)

            #if attr == '_copy_for' and isinstance(self._copy_for, str):
            #    print "***", self._copy_for
            #    self._copy_for = json.loads(self._copy_for)

    @classmethod
    def _from_json(cls, bundle=None, **kwargs):
        """
        """
        # this is a class method to initialize any subclassed Parameter by classname
        # will almost always call through parameter_from_json

        # TODO: is this even necessary?  for most cases we can probably just call __init__
        # TODO: this will surely break for those that require bundle as the first arg
        if bundle is not None:
            return cls(bundle, **kwargs)
        else:
            return cls(**kwargs)

    def __repr__(self):
        """
        """
        if isinstance(self._value, nparray.ndarray):
            quantity = self._value
        elif hasattr(self, 'quantity'):
            quantity = self.get_quantity()
        else:
            quantity = self.get_value()

        if hasattr(self, 'constraint') and self.constraint is not None:
            return "<Parameter: {}={} (constrained) | keys: {}>".format(self.qualifier, quantity.__repr__() if isinstance(quantity, distl._distl.BaseDistlObject) else quantity, ', '.join(self._dict_fields_other))
        else:
            return "<Parameter: {}={} | keys: {}>".format(self.qualifier, quantity.__repr__() if isinstance(quantity, distl._distl.BaseDistlObject) else quantity, ', '.join(self._dict_fields_other))

    def __str__(self):
        """
        """
        if isinstance(self._value, nparray.ndarray):
            quantity = self._value
        elif hasattr(self, 'quantity'):
            quantity = self.get_quantity()
        else:
            quantity = self.get_value()

        str_ = "{}: {}\n".format("Parameter", self.uniquetwig)
        str_ += "{:>32}: {}\n".format("Qualifier", self.qualifier)
        str_ += "{:>32}: {}\n".format("Description", self.description)
        str_ += "{:>32}: {}\n".format("Value", quantity.__repr__() if isinstance(quantity, distl._distl.BaseDistlObject) else quantity)

        if hasattr(self, 'choices'):
            str_ += "{:>32}: {}\n".format("Choices", ", ".join(self.choices))
        if hasattr(self, 'constrained_by'):
            str_ += "{:>32}: {}\n".format("Constrained by", ", ".join([p.uniquetwig for p in self.constrained_by]) if self.constrained_by is not None else 'None')
        if hasattr(self, 'constrains'):
            str_ += "{:>32}: {}\n".format("Constrains", ", ".join([p.uniquetwig for p in self.constrains]) if len(self.constrains) else 'None')
        if hasattr(self, 'related_to'):
            str_ += "{:>32}: {}\n".format("Related to", ", ".join([p.uniquetwig for p in self.related_to]) if len(self.related_to) else 'None')
        if self.visible_if is not None:
            str_ += "{:>32}: {}\n".format("Only visible if", self.visible_if)

        return str_

    def __len__(self):
        """
        since this may be returned from a filter, fake to say there is only 1 result
        """
        return 1

    def __comp__(self, other, comp):
        if isinstance(other, float) or isinstance(other, int):
            return getattr(self.get_value(), comp)(other)
        elif isinstance(other, u.Quantity):
            return getattr(self.get_quantity(), comp)(other)
        elif isinstance(other, str) and isinstance(self.get_value(), str) and comp in ['__eq__', '__ne__']:
            return getattr(self.get_value(), comp)(other)
        elif isinstance(other, tuple) and len(other)==2 and (isinstance(other[0], float) or isinstance(other[0], int)) and isinstance(other[1], str):
            return self.__comp__(other[0]*u.Unit(other[1]), comp)
        else:
            raise NotImplementedError("cannot compare between {} and {}".format(self.__class__.__name__, type(other)))


    def __lt__(self, other):
        return self.__comp__(other, '__lt__')

    def __le__(self, other):
        return self.__comp__(other, '__le__')

    def __gt__(self, other):
        return self.__comp__(other, '__gt__')

    def __ge__(self, other):
        return self.__comp__(other, '__ge__')

    def __eq__(self, other):
        """
        """
        if other is None:
            return False

        if not isinstance(other, Parameter):
            return self.__comp__(other, '__eq__')

        return self.uniqueid == other.uniqueid

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        """
        Deepcopy the <phoebe.parameters.Parameter> (with a new uniqueid).
        All other tags will remain the same... so some other tag should be
        changed before attaching back to a <phoebe.parameters.ParameterSet> or
        <phoebe.frontend.bundle.Bundle>.

        See also:
        * <phoebe.parameters.Parameter.uniqueid>

        Returns
        ---------
        * (<phoebe.parameters.Parameter>): the copied Parameter object
        """
        s = self.to_json()

        if self.__class__.__name__ in _parameter_class_that_require_bundle:
            cpy = parameter_from_json(s, bundle=self._bundle)
        else:
            cpy = parameter_from_json(s)

        cpy.set_uniqueid(_uniqueid())
        cpy._bundle = None
        return cpy

    def to_string(self):
        """
        Return the string representation of the <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.Parameter.to_string_short>

        Returns
        -------
        * (str): the string representation
        """
        return self.__str__()

    def to_string_short(self):
        """
        Return a short/abreviated string representation of the
        <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.Parameter.to_string>

        Returns
        --------
        * (str): the string representation
        """
        if hasattr(self, 'constrained_by') and len(self.constrained_by) > 0:
            prefix = 'C '
        elif self.readonly:
            prefix = 'R '
        else:
            prefix = '  '

        quantity = self.get_quantity() if hasattr(self, 'quantity') else self.get_value()
        return "{} {:>30}: {}".format(prefix, self.uniquetwig_trunc, quantity.__repr__() if isinstance(quantity, distl._distl.BaseDistlObject) else quantity)

    # @property
    # def __dict__(self):
    #     """
    #     """
    # return self.to_dict()

    def to_dict(self):
        """
        Return the dictionary representation of the <phoebe.parameters.Parameter>.

        Returns
        -------
        * (dict): the dictionary representation of the Parameter.
        """
        # including uniquetwig for everything can be VERY SLOW, so let's not
        # include that in the dictionary
        d =  {k: getattr(self,k) for k in self._dict_fields if k not in ['uniquetwig'] and (k not in ['readonly', 'advanced'] or getattr(self,k))}
        d['Class'] = self.__class__.__name__
        return d

    def __getitem__(self, key):
        """
        """
        return self.to_dict()[key]

    def __setitem__(self, key, value):
        """
        """
        # TODO: don't allow changing things like visible_if or description here?
        raise NotImplementedError

    @classmethod
    def open(cls, filename):
        """
        Open a Parameter from a JSON-formatted file.
        This is a constructor so should be called as:

        ```py
        param = Parameter.open('test.json')
        ```

        See also:
        * <phoebe.parameters.ParameterSet.open>
        * <phoebe.frontend.bundle.Bundle.open>

        Arguments
        ---------
        * `filename` (string): relative or full path to the file.  Alternatively,
            this can be the json string itself or a dictionary (the
            unpacked json).

        Returns
        -------
        * (<phoebe.parameters.Parameter): the inistantiated Parameter object.
        """
        if isinstance(filename, dict):
            data = filename
        elif isinstance(filename, str) and "{" in filename:
            data = json.loads(filename)
        else:
            filename = os.path.expanduser(filename)
            f = open(filename, 'r')
            data = json.load(f, object_pairs_hook=parse_json)
            f.close()
        return cls(data)

    def save(self, filename, incl_uniqueid=False):
        """
        Save the Parameter to a JSON-formatted ASCII file

        See also:
        * <phoebe.parameters.ParameterSet.save>
        * <phoebe.frontend.bundle.Bundle.save>

        Arguments
        ----------
        * `filename` (string): relative or full path to the file
        * `incl_uniqueid` (bool, optional, default=False): whether to include
            uniqueids in the file (only needed if its necessary to maintain the
            uniqueids when reloading)

        Returns
        --------
        * (string) filename
        """
        filename = os.path.expanduser(filename)
        f = open(filename, 'w')
        json.dump(self.to_json(incl_uniqueid=incl_uniqueid),
                  f, sort_keys=True, indent=0, separators=(',', ': '))
        f.close()

        return filename

    def to_json(self, incl_uniqueid=False, incl_none=False, exclude=[]):
        """
        Convert the <phoebe.parameters.Parameter> to a json-compatible
        object.

        See also:
        * <phoebe.parameters.ParameterSet.to_json>
        * <phoebe.parameters.Parameter.to_dict>
        * <phoebe.parameters.Parameter.save>

        Arguments
        --------
        * `incl_uniqueid` (bool, optional, default=False): whether to include
            uniqueids in the file (only needed if its necessary to maintain the
            uniqueids when reloading)
        * `incl_none` (bool, optional, default=False): whether to include tags
            whose values are None.
        * `exclude` (list, optional, default=[]): tags to exclude when saving.

        Returns
        -----------
        * (dict)
        """
        def _parse(k, v):
            """
            """
            if k=='value':
                if isinstance(self._value, nparray.ndarray):
                    if self._value.unit is not None and hasattr(self, 'default_unit'):
                        v = self._value.to(self.default_unit).to_dict()
                    else:
                        v = self._value.to_dict()
                elif isinstance(self._value, distl.BaseDistlObject):
                    v = self._value.to_dict(exclude=exclude)
                elif isinstance(v, dict):
                    v = {dk: _parse(k, dv) for dk,dv in v.items()}

                if isinstance(v, u.Quantity):
                    v = self.get_value() # force to be in default units
                if isinstance(v, np.ndarray):
                    # can handle N-dim arrays
                    v = v.tolist()
                if _is_unit(v):
                    v = str(v.to_string())
                return v
            elif k=='limits':
                return [vi.value if hasattr(vi, 'value') else vi for vi in v]
            elif k=='required_shape':
                return v.tolist() if v is not None else None
            elif v is None:
                return v
            elif isinstance(v, str):
                return v
            elif isinstance(v, dict):
                return v
            elif isinstance(v, float) or isinstance(v, int) or isinstance(v, list):
                return v
            elif _is_unit(v):
                return str(v.to_string())
            else:
                try:
                    return str(v)
                except:
                    raise NotImplementedError("could not parse {} of '{}' to json".format(k, self.uniquetwig))

        return {k: _parse(k, v) for k,v in self.to_dict().items() if ((v is not None or incl_none) and k not in ['twig', 'uniquetwig', 'quantity']+exclude and (k!='uniqueid' or incl_uniqueid or self.qualifier=='detached_job'))}

    @property
    def attributes(self):
        """
        Return a list of the attributes of this <phoebe.parameters.Parameter>.

        Returns
        -------
        * (list)
        """
        return self._dict_fields_other

    def get_attributes(self):
        """
        Return a list of the attributes of this <phoebe.parameters.Parameter>.
        This is simply a shortcut to <phoebe.parameters.Parameter.attributes>.

        Returns
        --------
        * (list)
        """
        return self.attributes

    @property
    def meta(self):
        """
        See all the meta-tag properties for this <phoebe.parameters.Parameter>.

        See <phoebe.parameters.Parameter.get_meta> for the ability to ignore
        certain keys.

        Returns
        -------
        * (dict) an ordered dictionary of all tag properties.
        """
        return self.get_meta()

    def get_meta(self, ignore=['uniqueid']):
        """
        See all the meta-tag properties for this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.Parameter.meta>
        * <phoebe.parameters.ParameterSet.get_meta>

        Arguments
        ---------
        * `ignore` (list, optional, default=['uniqueid']): list of keys to
            exclude from the returned dictionary

        Returns
        ----------
        * (dict) an ordered dictionary of tag properties
        """
        return OrderedDict([(k, getattr(self, k)) for k in _meta_fields_all if k not in ignore])

    @property
    def tags(self):
        """
        Returns a dictionary that lists all available tags.

        See also:
        * <phoebe.parameters.ParameterSet.tags>
        * <phoebe.parameters.Parameter.meta>

        Will include entries from the singular attributes:
        * <phoebe.parameters.Parameter.context>
        * <phoebe.parameters.Parameter.kind>
        * <phoebe.parameters.Parameter.model>
        * <phoebe.parameters.Parameter.compute>
        * <phoebe.parameters.Parameter.constraint>
        * <phoebe.parameters.Parameter.dataset>
        * <phoebe.parameters.Parameter.component>
        * <phoebe.parameters.Parameter.feature>
        * <phoebe.parameters.Parameter.time>
        * <phoebe.parameters.Parameter.qualifier>

        Returns
        ----------
        * (dict) a dictionary of all singular tag attributes.
        """
        return self.get_meta(ignore=['uniqueid', 'twig', 'uniquetwig'])

    @property
    def uniquetags(self):
        """
        Determine the minimal required filter tags which will point
        to this single <phoebe.parameters.Parameter> in the parent
        <phoebe.frontend.bundle.Bundle>.

        See <phoebe.parameters.Parameter.get_uniquetwig>
        for the ability to pass a <phoebe.parameters.ParameterSet>.

        See also:
        * <phoebe.parameters.Parameter.tags>
        * <phoebe.parameters.Parameter.uniquetwig>

        Returns
        --------
        * (dict) dictionary of tags
        """
        return self.get_uniquetags()


    def get_uniquetags(self, ps=None, force_levels=['qualifier'], exclude_levels=[]):
        """
        Determine the minimal required filter tags which will point
        to this single <phoebe.parameters.Parameter> in a given parent
        <phoebe.parameters.ParameterSet>.

        See also:
        * <phoebe.parameters.Parameter.tags>
        * <phoebe.parameters.Parameter.uniquetwig>

        Arguments
        ----------
        * `ps` (<phoebe.parameters.ParameterSet>, optional): ParameterSet
            in which the returned uniquetwig will point to this Parameter.
            If not provided or None this will default to the parent
            <phoebe.frontend.bundle.Bundle>, if available.
        * `force_levels` (list, optional, default=['qualifier']): levels to
            always include in the returned twig.  In addition, the attribute
            corresponding to the context of the parameter as well as the
            context itself will ALWAYS be included (unless in `exclude_levels`).
        * `exclude_levels` (bool, optional, default=True): levels to exclude
            from the twig (takes precedence over `force_levels`)

        Returns
        --------
        * (dict) dictionary of tags
        """

        if ps is None:
            ps = self._bundle

        if ps is None:
            return self.tags

        return ps._uniquetags(self, force_levels=force_levels, exclude_levels=exclude_levels)

    @property
    def readonly(self):
        """
        Whether the parameter is readonly.  To force setting the value, pass
        `ignore_readonly=True` to <<class>.set_value>.
        """
        return self._readonly

    @property
    def advanced(self):
        """
        Whether the parameter is considered an advanced parameter
        """
        return self._advanced

    @property
    def qualifier(self):
        """
        Return the qualifier of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.qualifier>
        * <phoebe.parameters.ParameterSet.qualifiers>

        Returns
        -------
        * (str) the qualifier tag of this Parameter.
        """
        return self._qualifier

    @property
    def time(self):
        """
        Return the time of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.time>
        * <phoebe.parameters.ParameterSet.times>

        Returns
        -------
        * (str) the time tag of this Parameter.
        """
        # need to force formatting because of the different way numpy.float64 is
        # handled before numpy 1.14.  See https://github.com/phoebe-project/phoebe2/issues/247
        return '{:09f}'.format(float(self._time)) if self._time is not None else None

    @property
    def feature(self):
        """
        Return the feature of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.feature>
        * <phoebe.parameters.ParameterSet.features>

        Returns
        -------
        * (str) the feature tag of this Parameter.
        """
        return self._feature

    @property
    def component(self):
        """
        Return the component of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.component>
        * <phoebe.parameters.ParameterSet.components>

        Returns
        -------
        * (str) the component tag of this Parameter.
        """
        return self._component

    @property
    def dataset(self):
        """
        Return the dataset of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.dataset>
        * <phoebe.parameters.ParameterSet.datasets>

        Returns
        -------
        * (str) the dataset tag of this Parameter.
        """
        return self._dataset

    @property
    def constraint(self):
        """
        Return the constraint of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.constraint>
        * <phoebe.parameters.ParameterSet.constraints>

        Returns
        -------
        * (str) the constraint tag of this Parameter.
        """
        return self._constraint

    @property
    def distribution(self):
        """
        Return the distribution of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.distribution>
        * <phoebe.parameters.ParameterSet.distributions>

        Returns
        -------
        * (str) the distribution tag of this Parameter.
        """
        return self._distribution

    @property
    def compute(self):
        """
        Return the compute of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.compute>
        * <phoebe.parameters.ParameterSet.computes>

        Returns
        -------
        * (str) the compute tag of this Parameter.
        """
        return self._compute

    @property
    def model(self):
        """
        Return the model of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.model>
        * <phoebe.parameters.ParameterSet.models>

        Returns
        -------
        * (str) the model tag of this Parameter.
        """
        return self._model

    @property
    def figure(self):
        """
        Return the figure of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.figure>
        * <phoebe.parameters.ParameterSet.figures>

        Returns
        -------
        * (str) the figure tag of this Parameter.
        """
        return self._figure

    @property
    def solver(self):
        """
        Return the solver of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.solver>
        * <phoebe.parameters.ParameterSet.solvers>

        Returns
        -------
        * (str) the solver tag of this Parameter.
        """
        return self._solver

    @property
    def solution(self):
        """
        Return the solution of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.solution>
        * <phoebe.parameters.ParameterSet.solutions>

        Returns
        -------
        * (str) the solution tag of this Parameter.
        """
        return self._solution

    @property
    def kind(self):
        """
        Return the kind of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.kind>
        * <phoebe.parameters.ParameterSet.kinds>

        Returns
        -------
        * (str) the kind tag of this Parameter.
        """
        return self._kind

    @property
    def context(self):
        """
        Return the context of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.context>
        * <phoebe.parameters.ParameterSet.contexts>

        Returns
        -------
        * (str) the context tag of this Parameter.
        """
        return self._context

    @property
    def uniqueid(self):
        """
        Return the uniqueid of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.ParameterSet.uniequids>

        Returns
        -------
        * (str) the uniqueid of this Parameter.
        """
        return self._uniqueid

    @property
    def uniquetwig_trunc(self):
        """
        Return the uniquetwig but truncated if necessary to be <=12 characters.

        See also:
        * <phoebe.parameters.Parameter.uniquetwig>
        * <phoebe.parameters.Parameter.twig>

        Returns
        --------
        * (str) the uniquetwig, truncated to 12 characters
        """
        uniquetwig = self.uniquetwig
        if len(uniquetwig) > 30:
            return uniquetwig[:27]+'...'
        else:
            return uniquetwig


    @property
    def uniquetwig(self):
        """
        Determine the shortest (more-or-less) twig which will point
        to this single <phoebe.parameters.Parameter> in the parent
        <phoebe.frontend.bundle.Bundle>.

        See <phoebe.parameters.Parameter.get_uniquetwig>
        for the ability to pass a <phoebe.parameters.ParameterSet>.

        See also:
        * <phoebe.parameters.Parameter.twig>
        * <phoebe.parameters.Parameter.uniquetwig_trunc>

        Returns
        --------
        * (str) uniquetwig
        """
        return self.get_uniquetwig()


    def get_uniquetwig(self, ps=None, force_levels=['qualifier'], exclude_levels=[]):
        """
        Determine the shortest (more-or-less) twig which will point
        to this single <phoebe.parameters.Parameter> in a given parent
        <phoebe.parameters.ParameterSet>.

        See also:
        * <phoebe.parameters.Parameter.twig>
        * <phoebe.parameters.Parameter.uniquetwig_trunc>

        Arguments
        ----------
        * `ps` (<phoebe.parameters.ParameterSet>, optional): ParameterSet
            in which the returned uniquetwig will point to this Parameter.
            If not provided or None this will default to the parent
            <phoebe.frontend.bundle.Bundle>, if available.
        * `force_levels` (list, optional, default=['qualifier']): levels to
            always include in the returned twig.  In addition, the attribute
            corresponding to the context of the parameter as well as the
            context itself will ALWAYS be included (unless in `exclude_levels`).
        * `exclude_levels` (bool, optional, default=True): levels to exclude
            from the twig (takes precedence over `force_levels`)

        Returns
        --------
        * (str) uniquetwig
        """

        if ps is None:
            ps = self._bundle

        if ps is None:
            return self.twig

        return ps._uniquetwig(self, force_levels=force_levels, exclude_levels=exclude_levels)

    @property
    def twig(self):
        """
        The twig of a <phoebe.parameters.Parameter> is a single string with the
        individual <phoebe.parameters.Parameter.meta> tags separated by '@' symbols.
        This twig gives a single string which can point back to this Parameter.

        See also:
        * <phoebe.parameters.Parameter.uniquetwig>
        * <phoebe.parameters.ParameterSet.twigs>

        Returns
        --------
        * (str): the full twig of this Parameter.
        """
        return "@".join([getattr(self, k) for k in _meta_fields_twig if getattr(self, k) is not None])

    @property
    def latexfmt(self):
        """
        """
        return self._latexfmt

    @property
    def latextwig(self):
        """
        The latex representation of the parameter name/tags.  Will default to
        <phoebe.parameters.uniquetwig> if a latex representation does not exist.

        See also:
        * <phoebe.parameters.Parameter.qualifier>
        * <phoebe.parameters.Parameter.uniquetwig>
        """
        if self._latexfmt is not None:
            d = self.meta
            if d.get('component', None) not in [None, '_default']:
                parent = self._bundle.hierarchy.get_parent_of(d.get('component'))
                children = self._bundle.hierarchy.get_children_of(d.get('component'))

                latex_reprs = {}
                for p in self._bundle.filter(qualifier='latex_repr', context='figure', **_skip_filter_checks).to_list():
                    if not len(p.get_value()):
                        continue
                    if p.component is not None:
                        latex_reprs[p.component] = p.get_value()
                    elif p.dataset is not None:
                        latex_reprs[p.dataset] = p.get_value()
                    elif p.feature is not None:
                        latex_reprs[p.feature] = p.get_value()

                if d.get('component', None) not in [None, '_default']:
                    d['component'] = latex_reprs.get(d.get('component'), d.get('component'))
                    if parent is not None:
                        d['parent'] = latex_reprs.get(parent, parent)
                    for i,child in enumerate(children):
                        d['children{}'.format(i)] = latex_reprs.get(child, child)

                if d.get('dataset', None) not in [None, '_default']:
                    d['dataset'] = latex_reprs.get(d.get('dataset'), d.get('dataset'))

                if d.get('feature', None) not in [None, '_default']:
                    d['feature'] = latex_reprs.get(d.get('feature'), d.get('feature'))

            return r'$'+self._latexfmt.format(**d)+'$'
        else:
            return self.uniquetwig

    @property
    def visible_if(self):
        """
        Return the `visible_if` expression for this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.Parameter.is_visible>
        * <phoebe.parameters.Parameter.visible_if_parameters>

        Returns
        --------
        * (str): the `visible_if` expression for this Parameter
        """
        return self._visible_if

    @property
    def visible_if_parameters(self):
        """
        Return the parameters affecting the visibility of this <phoebe.parameters.Parameter>.

        See also:
        * <phoebe.parameters.Parameter.visible_if>
        * <phoebe.parameters.Parameters.is_visible>

        Returns
        ----------
        * <phoebe.parameters.ParameterSet>
        """
        parameter_uids = []

        for visible_if in self.visible_if.replace(',','||').split('||'):
            if visible_if.lower() == 'false':
                continue

            # otherwise we need to find the parameter we're referencing and check its value
            remove_metawargs = []
            while visible_if[0] == '[':
                remove_metawargs.append(visible_if[1:].split(']')[0])
                visible_if = ']'.join(visible_if[1:].split(']')[1:])

            qualifier, value = visible_if.split(':')

            if 'hierarchy.' in qualifier:
                # TODO: set specific syntax (hierarchy.get_meshables:2)
                # then this needs to do some logic on the hierarchy
                parameter_uids += [self._bundle.hierarchy.uniqueid]

            else:
                # the parameter needs to have all the same meta data except qualifier
                # TODO: switch this to use self.get_parent_ps ?
                metawargs = {k:v for k,v in self.get_meta(ignore=['twig', 'uniquetwig', 'uniqueid']+remove_metawargs).items() if v is not None}
                metawargs['qualifier'] = qualifier

                # this call is quite expensive and bloats every get_parameter(check_visible=True)
                param = self._bundle.get_parameter(check_visible=False,
                                                   check_default=False,
                                                   check_advanced=False,
                                                   check_single=False,
                                                   **metawargs)

                parameter_uids += [param.uniqueid]

        return self._bundle.filter(uniqueid=parameter_uids, **_skip_filter_checks)

    @property
    def is_visible(self, visible_if=None):
        """
        Execute the `visible_if` expression for this <phoebe.parameters.Parameter>
        and determine whether it is currently visible in the parent
        <phoebe.parameters.ParameterSet>.

        If `False`, <phoebe.parameters.ParameterSet.filter> calls must have
        `check_visible=False` or else this Parameter will be excluded.

        See also:
        * <phoebe.parameters.Parameter.visible_if>
        * <phoebe.parameters.Parameter.visible_if_parameters>

        Returns
        --------
        * (bool):  whether this parameter is currently visible
        """
        return self._is_visible()


    def _is_visible(self, visible_if=None):
        """
        Execute the `visible_if` expression for this <phoebe.parameters.Parameter>
        and determine whether it is currently visible in the parent
        <phoebe.parameters.ParameterSet>.

        If `False`, <phoebe.parameters.ParameterSet.filter> calls must have
        `check_visible=False` or else this Parameter will be excluded.

        See also:
        * <phoebe.parameters.Parameter.visible_if>

        Arguments
        -----------
        * `visible_if` (string or list, optional, default=None): expression to
            use to compute visibility.  If None or not provided, will default
            to <phoebe.parameters.Parameter.visible_if>.

        Returns
        --------
        * (bool):  whether this parameter is currently visible
        """
        def is_visible_single(visible_if):
            # visible_if syntax:
            # * [ignore,these]qualifier:value
            # * [ignore,these]qualifier:<tag>
            # print("is_visible_single {}".format(visible_if))

            if visible_if.lower() == 'false':
                return False

            # otherwise we need to find the parameter we're referencing and check its value
            remove_metawargs = []
            while visible_if[0] == '[':
                remove_metawargs.append(visible_if[1:].split(']')[0])
                visible_if = ']'.join(visible_if[1:].split(']')[1:])

            qualifier, value = visible_if.split(':')

            if 'hierarchy.' in qualifier:
                # TODO: set specific syntax (hierarchy.get_meshables:2)
                # then this needs to do some logic on the hierarchy
                hier = self._bundle.hierarchy
                if not hier or not len(hier.get_value()):
                    # then hierarchy hasn't been set yet, so we can't do any
                    # of these tests
                    return True

                method = qualifier.split('.')[1]

                if value in ['true', 'True']:
                    value = True
                elif value in ['false', 'False']:
                    value = False

                return getattr(hier, method)(self.component) == value

            else:

                # the parameter needs to have all the same meta data except qualifier
                # TODO: switch this to use self.get_parent_ps ?
                metawargs = {k:v for k,v in self.get_meta(ignore=['twig', 'uniquetwig', 'uniqueid']+remove_metawargs).items() if v is not None}
                metawargs['qualifier'] = qualifier
                # metawargs['twig'] = None
                # metawargs['uniquetwig'] = None
                # metawargs['uniqueid'] = None
                # if metawargs.get('component', None) == '_default':
                    # metawargs['component'] = None

                try:
                    # this call is quite expensive and bloats every get_parameter(check_visible=True)
                    param = self._bundle.get_parameter(check_visible=False,
                                                       check_default=False,
                                                       check_advanced=False,
                                                       check_single=False,
                                                       **metawargs)
                except ValueError:
                    # let's not let this hold us up - sometimes this can happen when copying
                    # parameters (from copy_for) in order that the visible_if parameter
                    # happens later
                    logger.debug("parameter not found when trying to determine is_visible for {}: {}".format(self.twig, metawargs))
                    return True

                #~ print "***", qualifier, param.qualifier, param.get_value(), value

                if isinstance(param, BoolParameter):
                    if value in ['true', 'True']:
                        value = True
                    elif value in ['false', 'False']:
                        value = False


                if isinstance(value, str) and value[0] in ['!', '~']:
                    pvalue = param.get_value()
                    if isinstance(pvalue, float):
                        return pvalue != float(value[1:])
                    elif isinstance(pvalue, int):
                        return pvalue != int(float(value[1:]))
                    return pvalue != value[1:]
                elif isinstance(value, str) and "|" in value:
                    return param.get_value() in value.split("|")
                elif value=='<notempty>':
                    return len(param.get_value(expand=True)) > 0
                elif value=='<plural>':
                    return len(param.get_value(expand=True)) > 1
                elif value=='<empty>':
                    return len(param.get_value(expand=True)) == 0
                elif isinstance(value, str) and value[0] == '<' and value[-1] == '>':
                    return param.get_value() == getattr(self, value[1:-1])
                else:
                    return param.get_value() == value

        if visible_if is None:
            visible_if = self.visible_if

        if visible_if is None:
            return True

        if not self._bundle:
            # then we may not be able to do the check, for now let's just return True
            return True


        # if isinstance(visible_if, list) or isinstance(visible_if, tuple):
            # return np.any([self.is_visible(vi) for vi in visible_if])

        # syntax:
        # * visible_if = 'condition1,condition2||condition3' (where '||' is or ',' is and)
        return np.any([np.all([is_visible_single(visible_if_ii) for visible_if_ii in visible_if_i.split(',')]) for visible_if_i in visible_if.split('||')])



    @property
    def copy_for(self):
        """
        Return the `copy_for` expression for this <phoebe.parameters.Parameter>.

        This expression determines which new components and datasets should
        receive a copy of this Parameter.

        Returns:
        * (dict) the `copy_for` expression for this Parameter
        """
        return self._copy_for


    @property
    def description(self):
        """
        Return the `description` of the <phoebe.parameters.Parameter>.  The
        description is a slightly longer explanation of the Parameter qualifier.

        See also:
        * <phoebe.parameters.Parameter.get_description>
        * <phoebe.parameters.ParameterSet.get_description>
        * <phoebe.parameters.Parameter.qualifier>

        Returns
        --------
        * (str) the description
        """
        return self._description

    def get_description(self):
        """
        Return the `description` of the <phoebe.parameters.Parameter>.  The
        description is a slightly longer explanation of the Parameter qualifier.

        See also:
        * <phoebe.parameters.Parameter.description>
        * <phoebe.parameters.ParameterSet.get_description>
        * <phoebe.parameters.Parameter.qualifier>

        Returns
        --------
        * (str) the description
        """
        return self._description

    @property
    def value(self):
        """
        Return the value of the <phoebe.parameters.Parameter>.  For more options,
        including units when applicable, use the appropriate `get_value`
        method instead:

        * <phoebe.parameters.FloatParameter.get_value>
        * <phoebe.parameters.FloatArrayParameter.get_value>
        * <phoebe.parameters.HierarchyParameter.get_value>
        * <phoebe.parameters.IntParameter.get_value>
        * <phoebe.parameters.BoolParameter.get_value>
        * <phoebe.parameters.ChoiceParameter.get_value>
        * <phoebe.parameters.SelectParameter.get_value>
        * <phoebe.parameters.ConstraintParameter.get_value>

        Returns
        ---------
        * (float/int/string/bool): the current value of the Parameter.
        """

        return self.get_value()

    def get_parent_ps(self):
        """
        Return a <phoebe.parameters.ParameterSet> of all Parameters in the same
        <phoebe.frontend.bundle.Bundle> which share the same
        meta-tags (except qualifier, twig, uniquetwig).

        See also:
        * <phoebe.parameters.Parameter.meta>

        Returns
        ----------
        * (<phoebe.parameters.ParameterSet>): the parent ParameterSet.
        """
        if self._bundle is None:
            return None

        metawargs = {k:v for k,v in self.meta.items() if k not in ['qualifier', 'twig', 'uniquetwig']}

        return self._bundle.filter(check_visible=False, check_default=False, **metawargs)

    #~ @property
    #~ def constraint(self):
        #~ """
        #~ returns the label of the constraint that constrains this parameter
        #~
        #~ you can then access all of the parameters of the constraint via bundle.get_constraint(label)
        #~ """
        #~ return self.constraint_expression.uniquetwig

    @property
    def is_constraint(self):
        """
        Returns the <phoebe.parameters.ConstraintParameter> that constrains
        this parameter.  If this <phoebe.parameters.Parameter>] is not
        constrained, this will return None.

        See also:
        * <phoebe.parameters.FloatParameter.constrained_by>
        * <phoebe.parameters.FloatParameter.in_constraints>
        * <phoebe.parameters.FloatParameter.constrains>
        * <phoebe.parameters.FloatParameter.related_to>

        Returns
        -------
        * (None or <phoebe.parameters.ConstraintParameter)
        """
        if self._is_constraint is None:
            return None
        return self._bundle.get_parameter(context='constraint', uniqueid=self._is_constraint, check_visible=False)

    @property
    def constrained_by(self):
        """
        Returns a list of <phoebe.parameters.Parameter> objects that constrain
        this <phoebe.parameters.FloatParameter>.

        See also:
        * <phoebe.parameters.FloatParameter.is_constraint>
        * <phoebe.parameters.FloatParameter.in_constraints>
        * <phoebe.parameters.FloatParameter.constrains>
        * <phoebe.parameters.FloatParameter.related_to>

        Returns
        -------
        * (list of <phoebe.parameters.Parameter>)
        """
        if self._is_constraint is None:
            return []
        params = []
        uniqueids = []
        for var in self.is_constraint._vars:
            param = var.get_parameter()
            if param.uniqueid != self.uniqueid and param.uniqueid not in uniqueids:
                params.append(param)
                uniqueids.append(param.uniqueid)
        return params

    #~ @property
    #~ def in_constraints(self):
        #~ """
        #~ returns a list the labels of the constraints in which this parameter constrains another
        #~
        #~ you can then access all of the parameters of a given constraint via bundle.get_constraint(constraint)
        #~ """
        #~ return [param.uniquetwig for param in self.in_constraints_expressions]

    @property
    def in_constraints(self):
        """
        Returns a list of the expressions in which this
        <phoebe.parameters.FloatParameter> constrains other Parameters.

        See also:
        * <phoebe.parameters.FloatParameter.is_constraint>
        * <phoebe.parameters.FloatParameter.constrained_by>
        * <phoebe.parameters.FloatParameter.constrains>
        * <phoebe.parameters.FloatParameter.related_to>

        Returns
        -------
        * (list of expressions)
        """
        expressions = []
        for uniqueid in self._in_constraints:
            expressions.append(self._bundle.get_parameter(context='constraint', uniqueid=uniqueid, check_visible=False))
        return expressions

    @property
    def constrains(self):
        """
        Returns a list of Parameters that are directly constrained by this
         <phoebe.parameters.FloatParameter>.

        See also:
        * <phoebe.parameters.FloatParameter.constrains_indirect>
        * <phoebe.parameters.FloatParameter.is_constraint>
        * <phoebe.parameters.FloatParameter.constrained_by>
        * <phoebe.parameters.FloatParameter.in_constraints>
        * <phoebe.parameters.FloatParameter.related_to>

         Returns
         -------
         * (list of Parameters)
        """
        params = []
        for constraint in self.in_constraints:
            for var in constraint._vars:
                param = var.get_parameter()
                if param.component == constraint.component and param.qualifier == constraint.qualifier:
                    if param not in params and param.uniqueid != self.uniqueid:
                        params.append(param)
        return params

    @property
    def constrains_indirect(self):
        """
        Returns a list of Parameters that are directly or indirectly constrained by this
         <phoebe.parameters.FloatParameter>.

        See also:
        * <phoebe.parameters.FloatParameter.constrains>
        * <phoebe.parameters.FloatParameter.is_constraint>
        * <phoebe.parameters.FloatParameter.constrained_by>
        * <phoebe.parameters.FloatParameter.in_constraints>
        * <phoebe.parameters.FloatParameter.related_to>

         Returns
         -------
         * (list of Parameters)
        """
        params = self.constrains
        for param in params:
            for p in param.constrains_indirect:
                if p not in params:
                    params.append(p)
        return params

    @property
    def related_to(self):
        """
        Returns a list of all parameters that are either constrained by or
        constrain this parameter.

        See also:
        * <phoebe.parameters.FloatParameter.is_constraint>
        * <phoebe.parameters.FloatParameter.constrained_by>
        * <phoebe.parameters.FloatParameter.in_constraints>
        * <phoebe.parameters.FloatParameter.constrains>

         Returns
         -------
         * (list of Parameters)
        """
        params = []
        constraints = self.in_constraints
        if self.is_constraint is not None:
            constraints.append(self.is_constraint)

        for constraint in constraints:
            for var in constraint._vars:
                param = var.get_parameter()
                if param not in params and param.uniqueid != self.uniqueid:
                    params.append(param)

        return params

    def to_constraint(self):
        """
        Convert this <phoebe.parameters.Parameter> to a
        <phoebe.parameters.ConstraintParameter>.

        **NOTE**: this is an advanced functionality: use with caution.

        Returns
        --------
        * (<phoebe.parameters.ConstraintParameter): the ConstraintParameter
        """
        return ConstraintParameter(self._bundle, "{%s}" % self.uniquetwig)

    def __math__(self, other, symbol, mathfunc):
        """
        """


        try:
            if isinstance(other, ConstraintParameter):
                # print "*** __math__", self.quantity, mathfunc, other.result, other.expr
                return ConstraintParameter(self._bundle, "{%s} %s (%s)" % (self.uniquetwig, symbol, other.expr), default_unit=(getattr(self.quantity, mathfunc)(other.result).unit))
            elif isinstance(other, Parameter):

                # we need to do some tricks here since the math could fail if doing
                # math on  arrays of different lengths (ie if one is empty)
                # So instead, we'll just multiply with 1.0 floats if we can get the
                # unit from the quantity.

                self_quantity = self.quantity
                other_quantity = other.quantity

                if hasattr(self_quantity, 'unit'):
                    self_quantity = 1.0 * self_quantity.unit
                if hasattr(other_quantity, 'unit'):
                    other_quantity = 1.0 * other_quantity.unit

                default_unit = getattr(self_quantity, mathfunc)(other_quantity).unit
                return ConstraintParameter(self._bundle, "{%s} %s {%s}" % (self.uniquetwig, symbol, other.uniquetwig), default_unit=default_unit)
            elif isinstance(other, u.Quantity):
                return ConstraintParameter(self._bundle, "{%s} %s %0.30f" % (self.uniquetwig, symbol, _value_for_constraint(other)), default_unit=(getattr(self.quantity, mathfunc)(other).unit))
            elif isinstance(other, float) or isinstance(other, int):
                if symbol in ['+', '-'] and hasattr(self, 'default_unit'):
                    # assume same units as self (NOTE: NOT NECESSARILY SI) if addition or subtraction
                    other = float(other)*self.default_unit
                else:
                    # assume dimensionless
                    other = float(other)*u.dimensionless_unscaled
                return ConstraintParameter(self._bundle, "{%s} %s %f" % (self.uniquetwig, symbol, _value_for_constraint(other)), default_unit=(getattr(self.quantity, mathfunc)(other).unit))
            elif isinstance(other, u.Unit) and mathfunc=='__mul__':
                return self.quantity*other
            else:
                raise NotImplementedError("math with type {} not supported".format(type(other)))
        except ValueError:
            raise ValueError("constraint math failed: make sure you're using astropy 1.0+")

    def __rmath__(self, other, symbol, mathfunc):
        """
        """
        try:
            if isinstance(other, ConstraintParameter):
                return ConstraintParameter(self._bundle, "(%s) %s {%s}" % (other.expr, symbol, self.uniquetwig), default_unit=(getattr(self.quantity, mathfunc)(other.result).unit))
            elif isinstance(other, Parameter):
                return ConstraintParameter(self._bundle, "{%s} %s {%s}" % (other.uniquetwig, symbol, self.uniquetwig), default_unit=(getattr(self.quantity, mathfunc)(other.quantity).unit))
            elif isinstance(other, u.Quantity):
                return ConstraintParameter(self._bundle, "%0.30f %s {%s}" % (_value_for_constraint(other), symbol, self.uniquetwig), default_unit=(getattr(self.quantity, mathfunc)(other).unit))
            elif isinstance(other, float) or isinstance(other, int):
                if symbol in ['+', '-'] and hasattr(self, 'default_unit'):
                    # assume same units as self if addition or subtraction
                    other = float(other)*self.default_unit
                else:
                    # assume dimensionless
                    other = float(other)*u.dimensionless_unscaled
                return ConstraintParameter(self._bundle, "%f %s {%s}" % (_value_for_constraint(other), symbol, self.uniquetwig), default_unit=(getattr(self.quantity, mathfunc)(other).unit))
            elif isinstance(other, u.Unit) and mathfunc=='__mul__':
                return self.quantity*other
            else:
                raise NotImplementedError("math with type {} not supported".format(type(other)))
        except ValueError:
            raise ValueError("constraint math failed: make sure you're using astropy 1.0+")

    def __add__(self, other):
        """
        """
        return self.__math__(other, '+', '__add__')

    def __radd__(self, other):
        """
        """
        return self.__rmath__(other, '+', '__radd__')

    def __sub__(self, other):
        """
        """
        return self.__math__(other, '-', '__sub__')

    def __rsub__(self, other):
        """
        """
        return self.__rmath__(other, '-', '__rsub__')

    def __mul__(self, other):
        """
        """
        return self.__math__(other, '*', '__mul__')

    def __rmul__(self, other):
        """
        """
        return self.__rmath__(other, '*', '__rmul__')

    def __div__(self, other):
        """
        """
        return self.__math__(other, '/', '__div__')

    def __rdiv__(self, other):
        """
        """
        return self.__rmath__(other, '/', '__rdiv__')

    def __truediv__(self, other):
        """
        """
        # NOTE: only used in python3
        return self.__math__(other, '/', '__truediv__')

    def __rtruediv__(self, other):
        """
        """
        # note only used in python3
        return self.__rmath__(other, '/', '__rtruediv__')

    def __pow__(self, other):
        """
        """
        return self.__math__(other, '**', '__pow__')

    def __rpow__(self, other):
        """
        """
        return self.__rmath__(other, '**', '__rpow__')

    def __mod__(self, other):
        """
        """
        return self.__math__(other, '%', '__mod__')

    def __rmod__(self, other):
        """
        """
        return self.__rmath__(other, '%', '__rmod__')

    def set_uniqueid(self, uniqueid):
        """
        Set the `uniqueid` of this <phoebe.parameters.Parameter>.
        There is no real need for a user to call this unless there is some
        conflict or they manually want to set the uniqueids.

        NOTE: this does not check for conflicts, and having two parameters
        without the same uniqueid (not really unique anymore is it) will
        surely cause unexpected results.  Use with caution.

        See also:
        * <phoebe.parameters.Parameter.uniqueid>

        Arguments
        ---------
        * `uniqueid` (string): the new uniqueid
        """
        # TODO: check to make sure uniqueid is valid (is actually unique within self._bundle and won't cause problems with constraints, etc)
        self._uniqueid = uniqueid

    def get_value(self, *args, **kwargs):
        """
        This method should be overriden by any subclass of
        <phoebe.parameters.Parameter>.
        Please see the individual classes documentation:

        * <phoebe.parameters.FloatParameter.get_value>
        * <phoebe.parameters.FloatArrayParameter.get_value>
        * <phoebe.parameters.HierarchyParameter.get_value>
        * <phoebe.parameters.IntParameter.get_value>
        * <phoebe.parameters.BoolParameter.get_value>
        * <phoebe.parameters.ChoiceParameter.get_value>
        * <phoebe.parameters.SelectParameter.get_value>
        * <phoebe.parameters.ConstraintParameter.get_value>

        If subclassing, this method needs to:
        * cast to the correct type/units, handling defaults

        Raises
        -------
        * NoteImplemmentedError: because this must be subclassed
        """
        if self.qualifier in kwargs.keys():
            # then we have an "override" value that was passed, and we should
            # just return that.
            # Example teff_param.get_value('teff', teff=6000) returns 6000
            return kwargs.get(self.qualifier)
        return None

    def set_value(self, *args, **kwargs):
        """
        This method should be overriden by any subclass of Parameter, and should
        be decorated with the @send_if_client decorator
        Please see the individual classes for documentation:

        * <phoebe.parameters.FloatParameter.set_value>
        * <phoebe.parameters.FloatArrayParameter.set_value>
        * <phoebe.parameters.HierarchyParameter.set_value>
        * <phoebe.parameters.IntParameter.set_value>
        * <phoebe.parameters.BoolParameter.set_value>
        * <phoebe.parameters.ChoiceParameter.set_value>
        * <phoebe.parameters.SelectParameter.set_value>
        * <phoebe.parameters.ConstraintParameter.set_value>

        If subclassing, this method needs to:
        * check the inputs for the correct format/agreement/cast_type
        * make sure that converting back to default_unit will work (if applicable)
        * make sure that in choices (if a choose)
        * make sure that not out of limits
        * make sure that not out of prior ??

        Raises
        -------
        * NotImplementedError: because this must be subclassed
        """
        raise NotImplementedError # <--- leave this in place, should be subclassed

    def _readonly_check(self, **kwargs):
        if 'ignore_readonly' in kwargs.keys():
            return
        if self.readonly:
            raise ValueError("Parameter is read-only.  Pass ignore_readonly=True to force setting value (use with caution).")


class StringParameter(Parameter):
    """
    Parameter that accepts any string for the value
    """
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(StringParameter, self).__init__(*args, **kwargs)

        self.set_value(kwargs.get('value', ''), ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.StringParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (string) the current or overridden value of the Parameter
        """
        default = super(StringParameter, self).get_value(**kwargs)
        if default is not None: return default
        return str(self._value)

    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.StringParameter>.

        Arguments
        ----------
        * `value` (string): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to the correct type
            or is not a valid value for the Parameter.
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(value)

        try:
            value = str(value)
        except:
            raise ValueError("could not cast value to string")
        else:
            self._value = value

class TwigParameter(Parameter):
    # TODO: change to RefParameter?
    """
    Parameter that handles referencing any other *parameter* by twig (must exist)
    This stores the uniqueid but will display as the current uniquetwig for that item
    """
    def __init__(self, bundle, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(TwigParameter, self).__init__(*args, **kwargs)

        # usually its the bundle's job to attach param._bundle after the
        # creation of a parameter.  But in this case, having access to the
        # bundle is necessary in order to intialize and set the value
        self._bundle = bundle

        self.set_value(kwargs.get('value', ''), ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def get_parameter(self):
        """
        Return the parameter that the `value` is referencing

        Returns
        --------
        * (<phoebe.parameters.Parameter>)
        """
        return self._bundle.get_parameter(uniqueid=self._value)

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.TwigParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (string) the current or overridden value of the Parameter
        """
        # self._value is the uniqueid of the parameter.  So we need to
        # retrieve that parameter, but display the current uniquetwig
        # to the user
        # print "*** TwigParameter.get_value self._value: {}".format(self._value)
        default = super(TwigParameter, self).get_value(**kwargs)
        if default is not None: return default
        if self._value is None:
            return None
        return _uniqueid_to_uniquetwig(self._bundle, self._value)


    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.StringParameter>.

        Arguments
        ----------
        * `value` (string): the new value of the Parameter.
        * `**kwargs`: passed on to filter to find the Parameter

        Raises
        ---------
        * ValueError: if `value` could not be converted to the correct type
            or is not a valid value for the Parameter.
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        # first make sure only returns one results
        if self._bundle is None:
            raise ValueError("TwigParameters must be attached from the bundle, and cannot be standalone")

        value = str(value)

        # NOTE: this means that in all saving of bundles, we MUST keep the uniqueid and retain them when re-opening
        value = _twig_to_uniqueid(self._bundle, value, **kwargs)
        self._value = value


class ChoiceParameter(Parameter):
    """
    Parameter in which the value has to match one of the pre-defined choices
    """
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(ChoiceParameter, self).__init__(*args, **kwargs)

        self._choices = kwargs.get('choices', [''])

        self.set_value(kwargs.get('value', ''), ignore_readonly=True, allow_not_in_choices=True)

        self._dict_fields_other = ['description', 'choices', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    @property
    def choices(self):
        """
        Return the valid list of choices.

        This is identical to: <phoebe.parameters.ChoiceParameter.get_choices>

        Returns
        ---------
        * (list) list of valid choices
        """
        return self._choices

    def get_choices(self):
        """
        Return the valid list of choices.

        This is identical to: <phoebe.parameters.ChoiceParameter.choices>

        Returns
        ---------
        * (list) list of valid choices
        """
        return self._choices

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.ChoiceParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.
            Note: the provided value is not checked against the valid set
            of choices (<phoebe.parameters.ChoiceParameter.choices>).

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (string) the current or overridden value of the Parameter
        """
        default = super(ChoiceParameter, self).get_value(**kwargs)
        if default is not None: return default
        return str(self._value)

    @send_if_client
    def set_value(self, value, run_checks=None, run_constraints=None, **kwargs):
        """
        Set the current value of the <phoebe.parameters.ChoiceParameter>.

        Arguments
        ----------
        * `value` (string): the new value of the Parameter.
        * `run_checks` (bool, optional): whether to call
            <phoebe.frontend.bundle.Bundle.run_checks> after setting the value.
            If `None`, the value in `phoebe.conf.interactive_checks` will be used.
            This will not raise an error, but will cause a warning in the logger
            if the new value will cause the system to fail checks.
        * `run_constraints` whether to run any necessary constraints after setting
            the value.  If `None`, the value in `phoebe.conf.interactive_constraints`
            will be used.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to a string.
        * ValueError: if `value` is not one of
            <phoebe.parameters.ChoiceParameter.choices>
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        try:
            value = str(value)
        except:
            raise ValueError("could not cast value to string")

        if self.qualifier=='passband':
            if value not in self.choices:
                self._choices = list_passbands(refresh=True)

        if value not in self.choices and not kwargs.get('allow_not_in_choices', False):
            raise ValueError("value for {} must be one of {}, not '{}'".format(self.uniquetwig, self.choices, value))

        # NOTE: downloading passbands from online is now handled by run_checks

        self._value = value


        if run_constraints is None:
            run_constraints = conf.interactive_constraints

        if run_constraints:
            if len(self._in_constraints):
                logger.debug("changing value of {} triggers {} constraints".format(self.twig, [c.twig for c in self.in_constraints]))
            for constraint_id in self._in_constraints:
                self._bundle.run_constraint(uniqueid=constraint_id, skip_kwargs_checks=True, run_constraints=run_constraints)
        else:
            # then we want to delay running constraints... so we need to track
            # which ones need to be run once requested
            if len(self._in_constraints):
                logger.debug("changing value of {} triggers delayed constraints {}".format(self.twig, [c.twig for c in self.in_constraints]))
            for constraint_id in self._in_constraints:
                if constraint_id not in self._bundle._delayed_constraints:
                    self._bundle._delayed_constraints.append(constraint_id)

        # run_checks if requested (default)
        if run_checks is None:
            run_checks = conf.interactive_checks
        if run_checks and self._bundle:
            report = self._bundle.run_checks(allow_skip_constraints=True, raise_logger_warning=True)


    def handle_choice_rename(self, **rename):
        """
        Update the value according to a set of renames.

        Arguments
        ---------------
        * `**rename`: all pairs are renamed from the keys to the values.

        Returns
        ------------
        * bool: whether the value has been changed due to `rename`.

        Raises
        -------------
        * ValueError: if the current value cannot be mapped to a value in
            <phoebe.parameters.ChoiceParameter.choices>.
        """
        current_value = self.get_value()

        value = rename.get(current_value, current_value)

        if current_value == value:
            return False

        if value in self.choices:
            self.set_value(value, ignore_readonly=True)
            return True
        else:
            raise ValueError("could not set value to a valid entry in choices: {}".format(self.choices))

class SelectParameter(Parameter):
    """
    Parameter in which the value is a list of pre-defined choices
    """
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(SelectParameter, self).__init__(*args, **kwargs)

        self._choices = kwargs.get('choices', [])

        self.set_value(kwargs.get('value', []), ignore_readonly=True)

        self._dict_fields_other = ['description', 'choices', 'value', 'visible_if', 'readonly', 'copy_for', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    @property
    def choices(self):
        """
        Return the valid list of choices.

        This is identical to: <phoebe.parameters.SelectParameter.get_choices>

        Returns
        ---------
        * (list) list of valid choices
        """
        return self._choices

    def get_choices(self):
        """
        Return the valid list of choices.

        This is identical to: <phoebe.parameters.SelectParameter.choices>

        Returns
        ---------
        * (list) list of valid choices
        """
        return self._choices

    def valid_selection(self, value):
        """
        Determine if `value` is valid given the current value of
        <phoebe.parameters.SelectParameter.choices>.

        In order to be valid, each item in the list `value` can be one of the
        items in the list of or match with at least one item by allowing for
        '*' and '?' wildcards.  Wildcard matching is done via the fnmatch
        python package.

        See also:
        * <phoebe.parameters.SelectParameter.remove_not_valid_selections>
        * <phoebe.parameters.SelectParameter.expand_value>

        Arguments
        ----------
        * `value` (string or list): the value to test against the list of choices

        Returns
        --------
        * (bool): whether `value` is valid given the choices.
        """
        if isinstance(value, list):
            return np.all([self.valid_selection(v) for v in value])

        if value in self.choices:
            return True

        if value == '*':
            return True

        # allow for wildcards
        for choice in self.choices:
            if _fnmatch(choice, value):
                return True

        return False

    def get_value(self, expand=False, **kwargs):
        """
        Get the current value of the <phoebe.parameters.SelectParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        See also:
        * <phoebe.parameters.SelectParameter.expand_value>

        Arguments
        ----------
        * `expand` (bool, optional, default=False): whether to expand any
            wildcards in the stored value against the valid choices (see
            <phoebe.parameters.SelectParameter.choices>)
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (list) the current or overridden value of the Parameter
        """
        if expand:
            return self.expand_value(**kwargs)

        default = super(SelectParameter, self).get_value(**kwargs)
        if default is not None:
            if isinstance(default, str):
                return [default]
            return default
        return self._value

    def expand_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.SelectParameter>.

        This is simply a shortcut to <phoebe.parameters.SelectParameter.get_value>
        but passing `expand=True`.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        See also:
        * <phoebe.parameters.SelectParameter.valid_selection>
        * <phoebe.parameters.SelectParameter.remove_not_valid_selections>

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (list) the current or overridden value of the Parameter
        """
        selection = []
        for v in self.get_value(**kwargs):
            for choice in self.choices:
                if v==choice and choice not in selection and len(choice):
                    selection.append(choice)
                elif _fnmatch(choice, v) and choice not in selection and len(choice):
                    selection.append(choice)

        return selection

    @send_if_client
    def set_value(self, value, run_checks=None, **kwargs):
        """
        Set the current value of the <phoebe.parameters.SelectParameter>.

        `value` must be valid according to
        <phoebe.parameters.SelectParameter.valid_selection>, otherwise a
        ValueError will be raised.

        Arguments
        ----------
        * `value` (string): the new value of the Parameter.
        * `run_checks` (bool, optional): whether to call
            <phoebe.frontend.bundle.Bundle.run_checks> after setting the value.
            If `None`, the value in `phoebe.conf.interactive_checks` will be used.
            This will not raise an error, but will cause a warning in the logger
            if the new value will cause the system to fail checks.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to the correct type
        * ValueError: if `value` is not valid for the current choices in
            <phoebe.parameters.SelectParameter.choices>.
            See also <phoebe.parameters.SelectParameter.valid_selection>
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            raise TypeError("value must be a list of strings, received {}".format(type(value)))

        try:
            value = [str(v) for v in value]
        except:
            raise ValueError("could not cast to list of strings")

        invalid_selections = []
        for v in value:
            if not self.valid_selection(v):
                invalid_selections.append(v)

        if len(invalid_selections):
            raise ValueError("{} are not valid selections.  Choices: {}".format(invalid_selections, self.choices))

        self._value = value

        # run_checks if requested (default)
        if run_checks is None:
            run_checks = conf.interactive_checks
        if run_checks and self._bundle:
            report = self._bundle.run_checks(allow_skip_constraints=True, raise_logger_warning=True)


    def handle_choice_rename(self, remove_not_valid=False, **rename):
        """
        Update the value according to a set of renames.

        Arguments
        ---------------
        * `remove_not_valid` (bool, optional, default=False): whether to allow
            for invalid selections but remove them by calling
            <phoebe.parameters.SelectParameter.remove_not_valid_selections>.
        * `**rename`: all pairs are renamed from the keys to the values.

        Raises
        -------------
        * ValueError: if any of the renamed items fails to pass
            <phoebe.parameters.SelectParameter.is_valid_selection>.
        """
        value = [rename.get(v, v) for v in self.get_value()]
        changed = len(rename.keys())

        if remove_not_valid:
            value_orig = value
            value = [v for v in value if self.valid_selection(v)]
            self.set_value(value, run_checks=False, ignore_readonly=True)
            return changed or len(value_orig) != len(value)

        else:
            if np.any([not self.is_valid_selection(v) for v in value]):
                raise ValueError("not all are valid after renaming")

            self.set_value(value, run_checks=False, ignore_readonly=True)
            return changed


    def remove_not_valid_selections(self):
        """
        Update the value to remove any that are (no longer) valid.  This
        should not need to be called manually, but is often called internally
        when components or datasets are removed from the
        <phoebe.frontend.bundle.Bundle>.

        See also:
        * <phoebe.parameters.SelectParameter.valid_selection>
        * <phoebe.parameters.SelectParameter.expand_value>
        * <phoebe.parameters.SelectParameter.set_value>
        """
        value = [v for v in self.get_value() if self.valid_selection(v)]
        changed = len(value) != len(self.get_value())
        self.set_value(value, run_checks=False, ignore_readonly=True)
        return changed

    def __add__(self, other):
        if isinstance(other, str):
            other = [other]

        if not isinstance(other, list):
            return super(SelectParameter, self).__add__(self, other)

        # then we have a list, so we want to append to the existing value
        return list(set(self.get_value()+other))

    def __sub__(self, other):
        if isinstance(other, str):
            other = [other]

        if not isinstance(other, list):
            return super(SelectParameter, self).__sub__(self, other)

        return [v for v in self.get_value() if v not in other]

class SelectTwigParameter(SelectParameter):
    @staticmethod
    def _match_twig(value, valueindex, choice):
        choice, choiceindex = _extract_index_from_string(choice)
        if '@' in value:
            value = value.split('@')
        if '@' in choice:
            choice = choice.split('@')
        return np.all([vs in choice for vs in value]) and (valueindex is None or valueindex == choiceindex)

    def valid_selection(self, value):
        """
        Determine if `value` is valid given the current value of
        <phoebe.parameters.SelectTwigParameter.choices>.

        In order to be valid, each item in the list `value` can be one of the
        items in the list of or match with at least one item by allowing for
        '*' and '?' wildcards.  Wildcard matching is done via the fnmatch
        python package.

        See also:
        * <phoebe.parameters.SelectTwigParameter.remove_not_valid_selections>
        * <phoebe.parameters.SelectTwigParameter.expand_value>

        Arguments
        ----------
        * `value` (string or list): the value to test against the list of choices

        Returns
        --------
        * (bool): whether `value` is valid given the choices.
        """
        if isinstance(value, list):
            return np.all([self.valid_selection(v) for v in value])

        if super(SelectTwigParameter, self).valid_selection(value):
            return True

        value, index = _extract_index_from_string(value)

        twigsplit = value.split('@')

        # need to do special twig matching
        for choice in self.choices:
            if self._match_twig(twigsplit, index, choice):
                return True

        return False

    def expand_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.SelectTwigParameter>.

        This is simply a shortcut to <phoebe.parameters.SelectTwigParameter.get_value>
        but passing `expand=True`.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        See also:
        * <phoebe.parameters.SelectParameter.valid_selection>
        * <phoebe.parameters.SelectParameter.remove_not_valid_selections>

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (list) the current or overridden value of the Parameter
        """

        selection = []
        for v in self.get_value(**kwargs):
            v, index = _extract_index_from_string(v)
            vsplit = v.split('@')
            for choice in self.choices:
                if v==choice and choice not in selection and len(choice):
                    selection.append(choice)
                elif _fnmatch(choice, v) and choice not in selection and len(choice):
                    selection.append(choice)
                elif self._match_twig(vsplit, index, choice) and choice not in selection and len(choice):
                    selection.append(choice)


        return selection


class BoolParameter(Parameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(BoolParameter, self).__init__(*args, **kwargs)

        self.set_value(kwargs.get('value', True), ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.BoolParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (bool) the current or overridden value of the Parameter
        """
        default = super(BoolParameter, self).get_value(**kwargs)
        if default is not None: return default
        return self._value

    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.BoolParameter>.

        If not a boolean, `value` is casted as follows:
        * 'false', 'False', '0' -> `False`
        * default python casting (0->`False`, other numbers->`True`, strings with length->`True`)

        Arguments
        ----------
        * `value` (bool): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to a boolean
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        if value in ['false', 'False', '0']:
            value = False

        try:
            value = bool(value)
        except:
            raise ValueError("could not cast value to boolean")
        else:
            self._value = value

class UnitParameter(ChoiceParameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(UnitParameter, self).__init__(*args, **kwargs)

        value = kwargs.get('value')
        value = self._check_type(value)
        self._value = value

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def _check_type(self, value):
        if isinstance(value, u.Unit) or isinstance(value, u.CompositeUnit) or isinstance(value, u.IrreducibleUnit):
            value = value.to_string()
            if value == '':
                return 'dimensionless'
            else:
                return value

        if value in ['', 'dimensionless']:
            return 'dimensionless'

        if isinstance(value, str) :
            try:
                value = u.Unit(str(value))
            except:
                raise ValueError("{} not supported Unit".format(value))
            else:
                return value.to_string()

        return value

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.UnitParameter>.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (string) the current or overridden value of the Parameter
        """
        return u.Unit(self._value) if self._value not in [None, '', 'dimensionless'] else u.dimensionless_unscaled

    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.UnitParameter>.

        Arguments
        ----------
        * `value` (Unit or string): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to a unit.
        * ValueError: if `value` cannot be mapped to one of
            <phoebe.parameters.UnitParameter.choices>
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        value = self._check_type(value)

        if value not in self.choices and not kwargs.get('allow_not_in_choices', False):
            # TODO: see if same physical type and allow if so?

            raise ValueError("value for {} must be one of {}, not '{}'".format(self.uniquetwig, self.choices, value))

        self._value = value


class DictParameter(Parameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(DictParameter, self).__init__(*args, **kwargs)

        self.set_value(kwargs.get('value', {}), ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.DictParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (dict) the current or overridden value of the Parameter
        """
        default = super(DictParameter, self).get_value(**kwargs)
        if default is not None: return default
        return self._value

    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.DictParameter>.

        Arguments
        ----------
        * `value` (dict): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to the correct type
            or is not a valid value for the Parameter.
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        try:
            value = dict(value)
        except:
            raise ValueError("could not cast value to dictionary")
        else:
            self._value = value


class IntParameter(Parameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(IntParameter, self).__init__(*args, **kwargs)

        limits = kwargs.get('limits', (None, None))
        self.set_limits(limits)

        self.set_value(kwargs.get('value', 1), ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'limits', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    @property
    def limits(self):
        """
        Return the current valid limits for the <phoebe.parameters.IntParameter>.

        This is identical to <phoebe.parameters.IntParameter.get_limits>.

        See also:
        * <phoebe.parameters.IntParameter.set_limits>
        * <phoebe.parameters.IntParameter.within_limits>

        Returns
        --------
        * (tuple): the current limits, where `None` means no lower/upper limits.
        """
        return self._limits

    def get_limits(self):
        """
        Return the current valid limits for the <phoebe.parameters.IntParameter>.

        This is identical to <phoebe.parameters.IntParameter.limits>.

        See also:
        * <phoebe.parameters.IntParameter.set_limits>
        * <phoebe.parameters.IntParameter.within_limits>

        Returns
        --------
        * (tuple): the current limits, where `None` means no lower/upper limits.
        """
        return self.limits

    def set_limits(self, limits=(None, None)):
        """
        Set the limits for the <phoebe.parameters.IntParameter>.

        See also:
        * <phoebe.parameters.IntParameter.get_limits>
        * <phoebe.parameters.IntParameter.within_limits>

        Arguments
        ----------
        * `limits` (tuple, optional, default=(None, None)): new limits
            formatted as (`lower`, `upper`) where either value can be `None`
            (interpretted as no lower/upper limits).
        """
        if not len(limits)==2:
            raise ValueError("limits must be in the format: (min, max)")

        if None not in limits and limits[1] < limits[0]:
            raise ValueError("lower limits must be less than upper limit")

        limits = list(limits)

        self._limits = limits

    def within_limits(self, value):
        """
        Check whether a value falls within the set limits.

        See also:
        * <phoebe.parameters.IntParameter.get_limits>
        * <phoebe.parameters.IntParameter.set_limits>

        Arguments
        --------
        * `value` (int): the value to check against the current limits.

        Returns
        --------
        * (bool): whether `value` is valid according to the limits.
        """

        return (self.limits[0] is None or value >= self.limits[0]) and (self.limits[1] is None or value <= self.limits[1])

    def _check_value(self, value):
        if isinstance(value, str):
            value = float(value)

        try:
            value = int(value)
        except:
            raise ValueError("could not cast value to integer")
        else:

            # make sure the value is within the limits
            if not self.within_limits(value):
                raise ValueError("value of {}={} not within limits of {}".format(self.qualifier, value, self.limits))

        return value

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.IntParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (int) the current or overridden value of the Parameter
        """
        default = super(IntParameter, self).get_value(**kwargs)
        if default is not None: return default
        return self._value

    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.IntParameter>.

        See also:
        * <phoebe.parameters.IntParameter.get_limits>
        * <phoebe.parameters.IntParameter.set_limits>
        * <phoebe.parameters.IntParameter.within_limits>

        Arguments
        ----------
        * `value` (int): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to an integer
        * ValueError: if `value` is outside the limits.  See:
            <phoebe.parameters.IntParameter.get_limits> and
            <phoebe.parameters.IntParameter.within_limits>
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        value = self._check_value(value)

        self._value = value


class DistributionParameter(Parameter):
    def __init__(self, bundle, value, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>

        additional options:
        * `default_unit`
        """
        super(DistributionParameter, self).__init__(**kwargs)

        self._bundle = bundle
        # also have to set all attributes before calling set_value so that
        # get_referenced_parameter works
        for k,v in kwargs.items():
            if hasattr(self, '_{}'.format(k)):
                setattr(self, '_{}'.format(k), v)

        self.set_value(value, ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def get_referenced_parameter(self):
        """
        Access the referenced parameter from the Bundle

        Returns
        ----------
        * <phoebe.parameters.Parameter> object
        """
        qualifier, index = _extract_index_from_string(self.qualifier)
        return self._bundle.exclude(context=['distribution', 'constraint'],
                                    check_visible=False).get_parameter(qualifier=qualifier,
                                          check_visible=False,
                                          **{k:v for k,v in self.meta.items() if k in _contexts and k not in ['context', 'distribution']})

    def lnp(self, value=None):
        """
        Return the log probability of drawing a value of the referenced
        parameter <phoebe.parameters.DistributionParameter.get_referenced_parameter>
        from the distribution.

        See also:
        * <phoebe.frontend.bundle.Bundle.calculate_lnp>

        Arguments
        ------------
        * `value` (float or quantity, optional, default=None): value at which
            to compute the probability.  If None, the quantity of the
            referenced parameter will be used.  If units are not provided, will
            assumed to be in the units of the distribution.

        Returns
        --------
        * (float): log probability

        Raises
        ----------
        * TypeError: if `value` is not of type None, quantity, float, or int.
        """
        if value is None:
            param_quantity = self.get_referenced_parameter().get_quantity()
        elif isinstance(value, u.Quantity):
            param_quantity = value
        elif isinstance(value, float) or isinstance(value, int):
            param_quantity = value * self.get_value().unit
        else:
            raise TypeError("value must be of type None, Quantity, float, or int")

        return self.get_value().logpdf(param_quantity.value, unit=param_quantity.unit)

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.DistributionParameter>
        """
        default = super(DistributionParameter, self).get_value(**kwargs)
        if default is not None: return default
        dist = self._value

        if isinstance(dist, distl.BaseAroundGenerator) and not self._bundle._within_solver:
            if dist.unit is not None:
                value = self.get_referenced_parameter().get_quantity().to(dist.unit).value
            else:
                value = self.get_referenced_parameter().get_value()

            qualifier, index = _extract_index_from_string(self.qualifier)
            if index is None:
                dist.value = value
            else:
                if len(value) <= index:
                    dist.value = np.nan
                else:
                    dist.value = value[index]

        return dist

    def _check_value(self, value):
        if isinstance(value, distl.BaseDistlObject):
            return value
        elif isinstance(value, dict) and 'distl' in value.keys():
            # then we're loading the JSON version of an nparray object
            return distl.from_dict(value)
        elif (isinstance(value, tuple) or isinstance(value, list)) and len(value)==2:
            # assume uniform
            return distl.uniform(*value)
        else:
            raise TypeError("must be a distl Distribution object, got {} (type: {})".format(value, type(value)))

    @send_if_client
    def set_value(self, value, force=False, run_checks=None, run_constraints=None, **kwargs):
        """
        Set the current value of the <phoebe.parameters.DistributionParameter>.

        Arguments
        -------------
        * `value` (distl distribution object): the distribution object
        * `run_checks` (bool, optional): whether to call
            <phoebe.frontend.bundle.Bundle.run_checks> after setting the value.
            If `None`, the value in `phoebe.conf.interactive_checks` will be used.
            This will not raise an error, but will cause a warning in the logger
            if the new value will cause the system to fail checks.
        * `run_constraints` whether to run any necessary constraints after setting
            the value.  If `None`, the value in `phoebe.conf.interactive_constraints`
            will be used.
        * `**kwargs`: IGNORED
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())
        value = self._check_value(value)

        ref_param = self.get_referenced_parameter()
        if value.unit is None:
            value.unit = ref_param.default_unit
        else:
            try:
                value.unit.to(ref_param.default_unit)
            except:
                raise ValueError("units of {} on distribution not compatible with units of {} on parameter".format(value.unit, ref_param.default_unit))

        # TODO: apply the label, but use dist.__repr__ for Parameter and ParameterSet
        # displays of the value (distl falls back on {label} when available)
        # value.label = ref_param.get_uniquetwig(self._bundle, exclude_levels=['context'])

        self._value = value


    def set_property(self, **kwargs):
        """
        Set any property of the underlying [distl](https://distl.readthedocs.io)
        object.

        Example:
        ```py
        param.set_value(loc=10, scale=5)
        ```

        Arguments
        ----------
        * `**kwargs`: properties to be set on the underlying distl object.
        """
        for property, value in kwargs.items():
            setattr(self._value, property, value)

    def plot(self, **kwargs):
        """
        Plot both the analytic distribution function as well as a sampled
        histogram from the distribution.  Requires matplotlib to be installed.

        This is simply a shortcut to [distl.BaseDistribution.plot](https://distl.readthedocs.io/en/latest/api/BaseDistribution.plot/)

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        return self.get_value().plot(**kwargs)

class FloatParameter(Parameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>

        additional options:
        * `default_unit`
        """
        super(FloatParameter, self).__init__(*args, **kwargs)

        default_unit = kwargs.get('default_unit', None)
        self.set_default_unit(default_unit)

        limits = kwargs.get('limits', (None, None))
        self.set_limits(limits)

        unit = kwargs.get('unit', None)  # will default to default_unit in set_value

        if isinstance(unit, str):
          unit = u.Unit(unit)


        timederiv = kwargs.get('timederiv', None)
        self.set_timederiv(timederiv)

        self.set_value(kwargs.get('value', ''), unit, ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'quantity', 'default_unit', 'limits', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt'] # TODO: add adjust?  or is that a different subclass?
        if conf.devel:
            # NOTE: this check will take place when CREATING the parameter,
            # so toggling devel after won't affect whether timederiv is included
            # in string representations.
            self._dict_fields_other += ['timederiv']

        self._dict_fields = _meta_fields_all + self._dict_fields_other

    @property
    def default_unit(self):
        """
        Return the default unit for the <phoebe.parameters.FloatParameter>.

        This is identical to <phoebe.parameters.FloatParameter.get_default_unit>.

        See also:
        * <phoebe.parameters.FloatParameter.set_default_unit>

        Returns
        --------
        * (unit): the current default units.
        """
        return self._default_unit

    def get_default_unit(self):
        """
        Return the default unit for the <phoebe.parameters.FloatParameter>.

        This is identical to <phoebe.parameters.FloatParameter.default_unit>.

        See also:
        * <phoebe.parameters.FloatParameter.set_default_unit>

        Returns
        --------
        * (unit): the current default units.
        """
        return self.default_unit

    def set_default_unit(self, unit):
        """
        Set the default unit for the <phoebe.parameters.FloatParameter>.

        See also:
        * <phoebe.parameters.FloatParameter.get_default_unit>

        Arguments
        --------
        * `unit` (unit or valid string): the desired new units.  If the Parameter
            currently has default units, then the new units must be compatible
            with the current units

        Raises
        -------
        * Error: if the new and current units are incompatible.
        """
        # TODO: add to docstring documentation about what happens (does the value convert, etc)
        # TODO: check to make sure isinstance(unit, astropy.u.Unit)
        # TODO: check to make sure can convert from current default unit (if exists)
        if isinstance(unit, str):
          unit = u.Unit(unit)
        elif unit is None:
            unit = u.dimensionless_unscaled

        if not _is_unit(unit):
            raise TypeError("unit must be a Unit")

        if hasattr(self, '_default_unit') and self._default_unit is not None:
            # we won't use a try except here so that the error comes from astropy
            check_convert = self._default_unit.to(unit)

        self._default_unit = unit

    @property
    def limits(self):
        """
        Return the current valid limits for the <phoebe.parameters.FloatParameter>.

        This is identical to <phoebe.parameters.FloatParameter.get_limits>.

        See also:
        * <phoebe.parameters.FloatParameter.set_limits>
        * <phoebe.parameters.FloatParameter.within_limits>

        Returns
        --------
        * (tuple): the current limits, where `None` means no lower/upper limits.
        """
        return self._limits

    def get_limits(self):
        """
        Return the current valid limits for the <phoebe.parameters.FloatParameter>.

        This is identical to <phoebe.parameters.FloatParameter.get_limits>.

        See also:
        * <phoebe.parameters.FloatParameter.set_limits>
        * <phoebe.parameters.FloatParameter.within_limits>

        Returns
        --------
        * (tuple): the current limits, where `None` means no lower/upper limits.
        """
        return self.limits

    def set_limits(self, limits=(None, None)):
        """
        Set the limits for the <phoebe.parameters.FloatParameter>.

        See also:
        * <phoebe.parameters.FloatParameter.get_limits>
        * <phoebe.parameters.FloatParameter.within_limits>

        Arguments
        ----------
        * `limits` (tuple, optional, default=(None, None)): new limits
            formatted as (`lower`, `upper`) where either value can be `None`
            (interpretted as no lower/upper limits).  If the individual values
            are floats (not quantities), they'll be assumed to be in the default
            units of the Parameter (see
            <phoebe.parameters.FloatParameter.get_default_unit> and
            <phoebe.parameters.FloatParameter.set_default_unit>)
        """
        if not len(limits)==2:
            raise ValueError("limits must be in the format: (min, max)")

        limits = list(limits)
        for i in range(2):
            # first convert to float if integer
            if isinstance(limits[i], int):
                limits[i] = float(limits[i])

            # now convert to quantity using default unit if value was float or int
            if isinstance(limits[i], float):
                limits[i] = limits[i] * self.default_unit

        if None not in limits and limits[1] < limits[0]:
            raise ValueError("lower limits must be less than upper limit")

        self._limits = limits

    def within_limits(self, value):
        """
        Check whether a value falls within the set limits.

        See also:
        * <phoebe.parameters.FloatParameter.get_limits>
        * <phoebe.parameters.FloatParameter.set_limits>

        Arguments
        --------
        * `value` (float/quantity): the value to check against the current
            limits.  If `value` is a float, it is assume to have the same
            units as the default units (see
            <phoebe.parameters.FloatParameter.get_default_unit> and
            <phoebe.parameters.FloatParameter.set_default_unit>).

        Returns
        --------
        * (bool): whether `value` is valid according to the limits.
        """

        if isinstance(value, int) or isinstance(value, float):
            value = value * self.default_unit

        return (self.limits[0] is None or value >= self.limits[0]) and (self.limits[1] is None or value <= self.limits[1])

    @property
    def timederiv(self):
        return self._timederiv

    @property
    def quantity(self):
        """
        Shortcut to <phoebe.parameters.FloatParameter.get_quantity>
        """
        return self.get_quantity()

    def get_timederiv(self):
        """
        """
        return self._timederiv

    def set_timederiv(self, timederiv):
        """
        """
        self._timederiv = timederiv

    @property
    def in_distributions(self):
        """
        List the distribution tags of the distributions attached to this parameters

        Returns
        ----------
        * (list of strings)
        """
        return self._bundle.filter(context='distribution', qualifier=self.qualifier,
                                   check_visible=False, check_default=False,
                                   **{k:v for k,v in self.meta.items() if k in _contexts and k not in ['context', 'distribution']}).distributions

    def add_distribution(self, value):
        """
        Add a distribution to the bundle attached to this Parameter.

        See also:
        * <phoebe.frontend.bundle.Bundle.add_distribution>
        * <phoebe.frontend.bundle.Bundle.add_dist>
        * <phoebe.parameters.FloatParameter.get_distribution>
        * <phoebe.parameters.FloatParameter.sample_distribution>

        Arguments
        ------------
        * `value` (distl Distribution object, optional, default=None): the
            distribution to be applied to the created <phoebe.parameters.DistributionParameter>.
            If not provided, will be a delta function around the current value
            of the referenced parameter.
        """
        if self._bundle is None:
            raise ValueError("parameter must be attached to a Bundle to call add_distribution")

        self._bundle.add_distribution(twig=self, value=value)

    def get_distribution_parameters(self, distribution=None, follow_constraints=True):
        """
        Get the distribution parameter(s) corresponding to `distribution`.

        See also:
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.frontend.bundle.Bundle.get_dist>
        * <phoebe.parameters.FloatParameter.get_distribution>
        * <phoebe.parameters.FloatParameter.sample_distribution>
        * <phoebe.parameters.FloatParameter.add_distribution>

        Arguments
        -------------
        * `distribution` (string list or None, optional, default=None): distribution
            to use when filtering.  If None, will default to <phoebe.parameters.FloatParameter.in_distributions>
        * `follow_constraints` (bool, optional, default=True): whether to include
            the distributions of parameters in the constrained parameter.  Only
            applicable if this parameter is currently constrained.  See also
            <phoebe.parameters.FloatParameter.is_constrained> and
            <phoebe.parameters.FloatParameter.constrained_by>.

        Returns
        ----------
        * <phoebe.parameters.ParameterSet> of distribution parameters.
        """
        if distribution is None:
            direct_distribution = self.in_distributions
        else:
            direct_distribution = distribution

        direct_ps =  self._bundle.filter(qualifier=self.qualifier,
                                         distribution=direct_distribution,
                                         context='distribution',
                                         check_visible=False,
                                         **{k:v for k,v in self.meta.items() if k in _contexts and k not in ['context', 'distribution']})

        indirect_params = []
        if follow_constraints and len(self.constrained_by):
            # then this is a constrained parameter, so we want to propagate
            # any distributions through the constraint and return a CompositeDistribution
            # instead.
            for constraining_param in self.constrained_by:
                indirect_params += self._bundle.filter(qualifier=constraining_param.qualifier,
                                                       distribution=distribution if distribution is not None else constraining_param.in_distributions,
                                                       context='distribution',
                                                       check_visible=False,
                                                       **{k:v for k,v in constraining_param.meta.items() if k in _contexts and k not in ['context', 'distribution']}).to_list()

        return direct_ps + indirect_params


    def get_distribution(self, distribution=None, follow_constraints=True,
                              resolve_around_distributions=False,
                              distribution_uniqueids=None,
                              delta_if_none=False):
        """
        Access the distribution object corresponding to this parameter
        tagged with distribution=`distribution`.  To access the
        <phoebe.parameters.DistributionParameter> itself, see
        <phoebe.frontend.bundle.Bundle.get_distribution>.

        If this parameter is a constrained parameter, and any of the parameters
        involved in the constraint have distributions attached with
        distribution=`distribution`, a distribution object will be exposed
        that is propagated through the constraint (whether or not a
        <phoebe.parameters.DistributionParameter> exists).  A warning will
        be raised in the <phoebe.logger> if a distribution does exist but
        the propaged distribution is to be returned instead.  To disable this
        behavior, set `follow_constraints` to False.

        See also:
        * <phoebe.frontend.bundle.Bundle.get_distribution>
        * <phoebe.frontend.bundle.Bundle.get_dist>
        * <phoebe.parameters.FloatParameter.get_distribution_parameter>
        * <phoebe.parameters.FloatParameter.sample_distribution>
        * <phoebe.parameters.FloatParameter.add_distribution>

        Arguments
        ----------
        * `distribution` (string or DistributionCollection, optional, default=None):
            distribution tag of the <phoebe.parameters.DistributionParameter>.
            Required if more than one are available.  Alternatively, a
            DistributionCollection (from <phoebe.frontend.bundle.Bundle.get_distribution_collection>)
            can be passed to `distribution` along with the uniqueids (pass `keys='uniqueids'`)
            passed to `distribution_uniqueids`.
        * `follow_constraints` (bool, optional, default=True): whether to propagate
            distributions through constraints if this parameter is constrained.
            If False, the distribution directly attached to the parameter
            will be exposed instead.
        * `resolve_around_distributions` (bool, optional, default=False): resolve
            any "around" distributions to the current face-value.
        * `distribution_uniqueids` (list of str, optional, default=None): if
            `distribution` is a DistributionCollection object, providing the uniqueids
            (from <phoebe.frontend.bundle.Bundle.get_distribution_collection>)
            is necessary to slice appropriately.  If `distribution` is not a
            DistributionCollection, `distribution_uniqueids` is ignored.
        * `delta_if_none` (bool, optional, default=False): whether to return
            a delta distribution around the parameter face-value if no distribution
            is found.

        Returns
        ----------
        * Distribution object (not the parameter, see
            <phoebe.frontend.bundle.Bundle.get_distribution> to access the
            <phoebe.parameters.DistributionParameter>)

        Raises
        ----------
        * ValueError: if no valid distribution can be found.
        """
        if self._bundle is None:
            raise ValueError("parameter must be attached to a Bundle to call get_distribution")

        if not isinstance(distribution, str) and not isinstance(distribution, distl._distl.BaseDistlObject) and distribution is not None:
            if isinstance(distribution, list):
                raise NotImplementedError()
            else:
                raise TypeError("distribution must be of type string, DistributionCollection, or None, got {}".format(type(distribution)))

        if isinstance(distribution, str) and distribution not in self._bundle.distributions:
            # then maybe pointing to a solution, etc
            distribution, distribution_uniqueids = self._bundle.get_distribution_collection(distribution, keys='uniqueid', allow_non_dc=False)

        dist = None
        # print("*** {}.get_distribution len(self.constrained_by)={}".format(self.twig, len(self.constrained_by)))
        if follow_constraints and len(self.constrained_by):
            # then this is a constrained parameter, so we want to propagate
            # any distributions through the constraint and return a CompositeDistribution
            # instead.

            if isinstance(distribution, str) and len(self._bundle.filter(qualifier=self.qualifier,
                                                     distribution=distribution,
                                                     context='distribution',
                                                     check_visible=False,
                                                     **{k:v for k,v in self.meta.items() if k in _contexts and k not in ['context', 'distribution']})):

                logger.warning("{} is constrained but also has a distribution attached with distribution='{}'.  Returning the distribution propagated through the constraint instead (pass follow_constraints=False to disable this behavior).".format(self.twig, distribution))

            # constraint_expr = self.is_constraint.get_value()
            if distribution is None:
                if len(self._bundle.distributions) > 1:
                    raise ValueError("must provide label of distribution")
                elif len(self._bundle.distributions) == 1:
                    distribution = self._bundle.distributions[0]
                else:
                    raise ValueError("no distributions found attached to bundle")

            if isinstance(distribution, distl._distl.DistributionCollection):
                if distribution_uniqueids is None:
                    raise ValueError("must provide distribution_uniqueids if distribution is a DistributionCollection")
                # attach the uniqueids so they're available within get_result
                # for subsequent calls here
                distribution.distribution_uniqueids = distribution_uniqueids

            dist = self.is_constraint.get_result(use_distribution=distribution, distribution_uniqueids=distribution_uniqueids)

            if not isinstance(dist, distl._distl.BaseDistlObject):
                # then the constraint returned a value, which means none of the
                # constraining parameters had matching distributions... so we'll
                # fallback on the distribution directly attached to this parameter
                # instead
                dist = None

        if dist is None:
            if isinstance(distribution, str):
                try:
                    dist = self._bundle.get_parameter(qualifier=self.qualifier,
                                                      distribution=distribution,
                                                      context='distribution',
                                                      check_visible=False,
                                                      **{k:v for k,v in self.meta.items() if k in _contexts and k not in ['context', 'distribution']}).get_value()
                except ValueError:
                    dist = None
            elif isinstance(distribution, distl._distl.DistributionCollection):
                if distribution_uniqueids is None:
                    if hasattr(distribution, 'distribution_uniqueids'):
                        # will this be a problem when jsoned internally by constraints?
                        distribution_uniqueids = distribution.distribution_uniqueids
                    else:
                        raise ValueError("must provide distribution_uniqueids if distribution is a DistributionCollection")

                if self.uniqueid in distribution_uniqueids:
                    dist = distribution.dists[distribution_uniqueids.index(self.uniqueid)]
                else:
                    dist = None


        if dist is None:
            if delta_if_none:
                dist = distl.delta(self.get_value())
            else:
                return None

        if isinstance(dist, distl.BaseAroundGenerator):
            if resolve_around_distributions:
                dist = dist.__call__(self.get_value())
            else:
                dist.value = self.get_value()

        if hasattr(self, self.context):
            dist.label = '{}@{}'.format(self.qualifier, getattr(self, self.context))
        else:
            dist.label = '{}@{}'.format(self.qualifier, self.context)
        if self._latexfmt is not None:
            dist.label_latex = self.latextwig.replace("$", "")

        return dist

    def sample_distribution(self, distribution=None, follow_constraints=True,
                            seed=None, set_value=False):
        """
        Sample from the distribution attached to this parameter (and optionally
        adopt the sampled value).

        See also:
        * <phoebe.parameters.FloatParameter.get_distribution>
        * <phoebe.parameters.FloatParameter.add_distribution>

        Arguments
        ------------
        * `distribution` (string, optional, default=None): distribution tag
            of the <phoebe.parameters.DistributionParameter>.  Required if
            more than one are available.
        * `follow_constraints` (bool, optional, default=True): whether to propagate
            distributions through constraints if this parameter is constrained.
            If False, the distribution directly attached to the parameter
            will be exposed instead.  See <phoebe.parameters.FloatParameter.get_distribution>
            for more details.
        * `seed` (int, optional, default=None): seed to use when randomly
            drawing from the distribution.
        * `set_value` (bool, optional, default=False): whether to adopt the
            sampled value.

        Returns
        --------
        * (float): the sampled value


        Raises
        ----------
        * ValueError: if no valid distribution can be found.
        """
        dist = self.get_distribution(distribution, follow_constraints=follow_constraints)
        value = dist.sample(seed=seed)
        if set_value:
            self.set_value(value, ignore_readonly=True)
        return value

    def get_value(self, unit=None, t=None,
                  **kwargs):
        """
        Get the current value of the <phoebe.parameters.FloatParameter> or
        <phoebe.parameters.FloatArrayParameter>.

        This is identical to <phoebe.parameters.FloatParameter.get_quantity>
        and is just included to match the method names of most other Parameter
        types.  See the documentation of <phoebe.parameters.FloatParameter.get_quantity>
        for full details.
        """
        default = super(FloatParameter, self).get_value(**kwargs)
        if default is not None: return self._check_type(default)
        quantity = self.get_quantity(unit=unit, t=t,
                                     **kwargs)
        if hasattr(quantity, 'value'):
            return quantity.value
        else:
            return quantity

    def get_quantity(self, unit=None, t=None,
                     **kwargs):
        """
        Get the current quantity of the <phoebe.parameters.FloatParameter> or
        <phoebe.parameters.FloatArrayParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        See also:
        * <phoebe.parameters.FloatParameter.get_quantity>

        Arguments
        ----------
        * `unit` (unit or string, optional, default=None): unit to convert the
            value.  If not provided, will use the default unit (see
            <phoebe.parameters.FloatParameter.default_unit>
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (float/array) the current or overridden value of the Parameter
        """
        default = super(FloatParameter, self).get_value(**kwargs) # <- note this is calling get_value on the Parameter object
        if default is not None:
            value = self._check_type(default)
            if isinstance(default, u.Quantity):
                return value
            return value * self.default_unit
        else:
            value = self._value

        if isinstance(value, nparray.ndarray):
            if value.unit is not None:
                value = value.quantity
            else:
                value = value.array

        if t is not None:
            raise NotImplementedError("timederiv is currently disabled until it can be tested thoroughly")

        if t is not None and self.is_constraint is not None:
            # TODO: is this a risk for an infinite loop?
            value = self.is_constraint.get_result(t=t)

        if t is not None and self.timederiv is not None:
            # check to see if value came from a constraint - and if so, we will
            # need to re-evaluate that constraint at t=t.


            parent_ps = self.get_parent_ps()
            deriv = parent_ps.get_value(self.timederiv, unit=self.default_unit/u.d)
            # t0 = parent_ps.get_value(qualifier='t0_values', unit=u.d)
            t0 = self._bundle.get_value(qualifier='t0', context='system', unit=u.d)

            # if time has been provided without units, we assume the same units as t0
            if not hasattr(time, 'value'):
                # time = time * parent_ps.get_parameter(qualifier='t0_values').default_unit
                time = time * self._bundle.get_value(qualifier='t0', context='system').default_unit

            # print "***", value, deriv, time, t0
            value = value + deriv*(time-t0)

        if unit is None:
            unit = self.default_unit

        # TODO: check to see if this is still necessary
        if isinstance(unit, str):
            if unit == 'solar':
                unit = u._physical_types_to_solar.get(u._get_physical_type(self.default_unit))
            elif unit in ['si', 'SI']:
                unit = u._physical_types_to_si.get(u._get_physical_type(self.default_unit))
            else:
                # we need to do this to make sure we get PHOEBE's version of
                # the unit instead of astropy's
                unit = u.Unit(unit)

        # TODO: catch astropy units and convert to PHOEBE's?

        if unit is None or value is None:
            return value
        else:
            # NOTE: astropy will raise an error if units not compatible
            return value.to(unit)

    def _check_value(self, value, unit=None):
        if isinstance(value, tuple) and (len(value) !=2 or isinstance(value[1], float) or isinstance(value[1], int)):
            # allow passing tuples (this could be a FloatArrayParameter - if it isn't
            # then this array will fail _check_type below)
            value = np.asarray(value)
        # accept tuples (ie 1.2, 'rad') from dictionary access
        if isinstance(value, tuple) and unit is None:
            value, unit = value
        if isinstance(value, str):
            if len(value.strip().split(' ')) == 2 and unit is None and self.__class__.__name__ == 'FloatParameter':
                # support value unit as string
                valuesplit = value.strip().split(' ')
                value = float(valuesplit[0])
                unit = valuesplit[1]

            elif "," in value and self.__class__.__name__ == 'FloatArrayParameter':
                try:
                    value = json.loads(value)
                    # we'll take it from here in the dict section below
                except:
                    value = np.asarray([float(v) for v in value.split(',') if len(v)])

            else:
                value = float(value)

        if isinstance(value, dict) and 'nparray' in value.keys():
            # then we're loading the JSON version of an nparray object
            value = nparray.from_dict(value)

        return self._check_type(value), unit

    def _check_type(self, value):
        # we do this separately so that FloatArrayParameter can keep this set_value
        # and just subclass _check_type
        if isinstance(value, u.Quantity):
            if not (isinstance(value.value, float) or isinstance(value.value, int)):
                raise ValueError("value could not be cast to float")

        elif not (isinstance(value, float) or isinstance(value, int)):
            # TODO: probably need to change this to be flexible with all the cast_types
            raise ValueError("value ({}) could not be cast to float".format(value))

        return value

    #@send_if_client is on the called set_quantity
    def set_value(self, value, unit=None, force=False, run_checks=None, run_constraints=None, **kwargs):
        """
        Set the current value/quantity of the <phoebe.parameters.FloatParameter>.

        This is identical to <phoebe.parameters.FloatParameter.set_quantity>
        and is just included to match the method names of most other Parameter
        types.  See the documentation of <phoebe.parameters.FloatParameter.set_quantity>
        for full details.

        See also:
        * <phoebe.parameters.FloatParameter.set_quantity>
        * <phoebe.parameters.FloatParameter.get_limits>
        * <phoebe.parameters.FloatParameter.set_limits>
        * <phoebe.parameters.FloatParameter.within_limits>
        """
        return self.set_quantity(value=value, unit=unit, force=force,
                                 run_checks=run_checks, run_constraints=run_constraints,
                                 **kwargs)

    @send_if_client
    def set_quantity(self, value, unit=None, force=False, run_checks=None, run_constraints=None, **kwargs):
        """
        Set the current value/quantity of the <phoebe.parameters.FloatParameter>
        or <phoebe.parameters.FloatArrayParameter>.

        Units can either be passed by providing a Quantity object to `value`
        OR by passing a unit object (or valid string representation) to `unit`.
        If units are provided with both but do not agree, an error will be raised.

        See also:
        * <phoebe.parameters.FloatParameter.set_value>
        * <phoebe.parameters.FloatParameter.get_limits>
        * <phoebe.parameters.FloatParameter.set_limits>
        * <phoebe.parameters.FloatParameter.within_limits>

        Arguments
        ----------
        * `value` (float/quantity): the new value of the Parameter.
        * `unit` (unit or valid string, optional, default=None): the unit in
            which `value` is provided.  If not provided or None, it is assumed
            that `value` is in the default units (see <phoebe.parameters.FloatParameter.default_unit>
            and <phoebe.parameters.FloatParameter.set_default_unit>).
        * `force` (bool, optional, default=False, EXPERIMENTAL): override
            and set the value of a constrained Parameter.
        * `run_checks` (bool, optional): whether to call
            <phoebe.frontend.bundle.Bundle.run_checks> after setting the value.
            If `None`, the value in `phoebe.conf.interactive_checks` will be used.
            This will not raise an error, but will cause a warning in the logger
            if the new value will cause the system to fail checks.
        * `run_constraints` whether to run any necessary constraints after setting
            the value.  If `None`, the value in `phoebe.conf.interactive_constraints`
            will be used.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to a float/quantity.
        * ValueError: if the units of `value` and `unit` are in disagreement
        * ValueError: if the provided units are not compatible with the
            default units.
        * ValueError: if `value` is outside the limits.  See:
            <phoebe.parameters.FloatParameter.get_limits> and
            <phoebe.parameters.FloatParameter.within_limits>
        """
        self._readonly_check(**kwargs)

        _orig_quantity = _deepcopy(self.get_quantity())

        if len(self.constrained_by) and not force:
            raise ValueError("cannot change the value of a constrained parameter.  This parameter is constrained by '{}'".format(', '.join([p.uniquetwig for p in self.constrained_by])))

        # if 'time' in kwargs.keys() and isinstance(self, FloatArrayParameter):
        #     # then find the correct index and set by index instead
        #     time_param = self._bundle

        value, unit = self._check_value(value, unit)

        if isinstance(unit, str):
            # print "*** converting string to unit"
            unit = u.Unit(unit)  # should raise error if not a recognized unit
        elif unit is not None and not _is_unit(unit):
            raise TypeError("unit must be an phoebe.u.Unit or None, got {}".format(unit))

        # check to make sure value and unit don't clash
        if isinstance(value, u.Quantity) or (isinstance(value, nparray.ndarray) and value.unit is not None):
            if unit is not None:
                # check to make sure they're the same
                if value.unit != unit:
                    raise ValueError("value and unit do not agree")

        elif unit is not None:
            # print "*** converting value to quantity"
            value = value * unit

        elif self.default_unit is not None:
            value = value * self.default_unit

        # handle wrapping for angle measurements
        if value is not None and value.unit.physical_type == 'angle':
            # NOTE: this may fail for nparray types
            if value > (360*u.deg) or value < (0*u.deg):
                if self._bundle is not None and self._bundle._within_solver and not kwargs.get('from_constraint', False):
                    if abs(value.to(u.deg).value - self._value.to(u.deg).value) > 180:
                        raise ValueError("value further than 180 deg from {}".format(self._value.to(u.deg).value))
                value = value % (360*u.deg)
                logger.warning("wrapping value of {} to {}".format(self.qualifier, value))

        # make sure the value is within the limits, if this isn't an array or nan
        if ((isinstance(value, float) and not np.isnan(value))
            or (isinstance(value, u.Quantity) and ((isinstance(value.value, float) and not np.isnan(value.value))
                                                   or isinstance(value.value, np.ndarray) and not np.any(np.isnan(value.value))))) and not self.within_limits(value):
            raise ValueError("value of {}={} not within limits of {}".format(self.qualifier, value, self.limits))


        # make sure we can convert back to the default_unit
        try:
            if self.default_unit is not None and value is not None:
                test = value.to(self.default_unit)
        except u.core.UnitsError:
            raise ValueError("cannot convert provided unit ({}) to default unit ({})".format(value.unit, self.default_unit))
        except:
            self._value = value
        else:
            self._value = value

        if run_constraints is None:
            run_constraints = conf.interactive_constraints

        if _orig_quantity is not None and self.__class__.__name__ == 'FloatParameter' and abs(_orig_quantity - value).value < 1e-12:
            logger.debug("value of {} didn't change within 1e-12, skipping triggering of constraints".format(self.twig))
        elif run_constraints:
            if len(self._in_constraints):
                logger.debug("changing value of {} triggers {} constraints".format(self.twig, [c.twig for c in self.in_constraints]))
            for constraint_id in self._in_constraints:
                self._bundle.run_constraint(uniqueid=constraint_id, skip_kwargs_checks=True, run_constraints=run_constraints)
        else:
            # then we want to delay running constraints... so we need to track
            # which ones need to be run once requested
            if len(self._in_constraints):
                logger.debug("changing value of {} triggers delayed constraints {}".format(self.twig, [c.twig for c in self.in_constraints]))
            for constraint_id in self._in_constraints:
                if constraint_id not in self._bundle._delayed_constraints:
                    self._bundle._delayed_constraints.append(constraint_id)

        # run_checks if requested (default)
        if run_checks is None:
            run_checks = conf.interactive_checks
        if run_checks and self._bundle:
            report = self._bundle.run_checks(allow_skip_constraints=True, raise_logger_warning=True)

        # make any necessary updates to choices
        # skip_update_choices (as a hidden kwarg) exists so that the server can
        # handle this externally and return the changes to the clients
        if self._bundle is not None and not kwargs.get('skip_update_choices', False):
            if self.qualifier in ['ld_coeffs', 'ld_coeffs_bol']:
                self._bundle._handle_fitparameters_selecttwigparams()

class FloatArrayParameter(FloatParameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>

        Additional arguments
        ---------------------
        """
        required_shape = kwargs.pop('required_shape', None)
        if isinstance(required_shape, int):
            required_shape = [required_shape]
        self._required_shape = np.asarray(required_shape) if required_shape is not None else None

        super(FloatArrayParameter, self).__init__(*args, **kwargs)

        # NOTE: default_unit and value handled in FloatParameter.__init__()

        self._dict_fields_other = ['description', 'value', 'default_unit', 'visible_if', 'required_shape', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def __repr__(self):
        """
        FloatArrayParameter needs to "truncate" the array by temporarily
        overriding np.set_printoptions
        """
        opt = np.get_printoptions()
        # <Parameter:_qualifier= takes 13+len(qualifier) characters
        np.set_printoptions(threshold=8, edgeitems=3, linewidth=opt['linewidth']-(13+len(self.qualifier)))
        repr_ = super(FloatArrayParameter, self).__repr__()
        np.set_printoptions(**opt)
        return repr_

    def __str__(self):
        """
        FloatArrayParameter needs to "truncate" the array by temporarily
        overriding np.set_printoptions
        """
        opt = np.get_printoptions()
        # Value:_ takes 7 characters
        np.set_printoptions(threshold=8, edgeitems=3, linewidth=opt['linewidth']-7)
        str_ = super(FloatArrayParameter, self).__str__()
        np.set_printoptions(**opt)
        return str_

    @property
    def required_shape(self):
        return self._required_shape

    def to_string_short(self):
        """
        Short abbreviated string representation of the
        <phoebe.parameters.FloatArrayParameter>.

        See also:
        * <phoebe.parameters.Parameter.to_string>

        Returns
        --------
        * (str)
        """
        opt = np.get_printoptions()
        np.set_printoptions(threshold=8, edgeitems=3, linewidth=opt['linewidth']-len(self.uniquetwig)-2)
        str_ = super(FloatArrayParameter, self).to_string_short()
        np.set_printoptions(**opt)
        return str_

    def interp_value(self, unit=None, component=None,
                     period='period', dpdt='dpdt', t0='t0_supconj',
                     consider_gaussian_process=True, **kwargs):
        """
        Interpolate to find the value in THIS array given a value from
        ANOTHER array in the SAME parent <phoebe.parameters.ParameterSet>
        (see <phoebe.parameters.Parameter.get_parent_ps>).

        This currently only supports simple 1D linear interpolation (via
        `numpy.interp`) and does no checks to make sure you're interpolating
        with respect to an independent parameter - so use with caution.

        ```py
        print this_param.get_parent_ps().qualifiers
        'other_qualifier' in this_param.get_parent_ps().qualifiers
        True
        this_param.interp_value(other_qualifier=5)
        ```

        where other_qualifier must be in ParentPS.qualifiers
        AND must point to another <phoebe.parameters.FloatArrayParameter>.

        Example:

        ```py
        b['fluxes@lc01@model'].interp_value(times=10.2)
        ```

        The only exception is when interpolating in phase-space, in which
        case the 'times' qualifier must be found in the ParentPS.  Interpolating
        in phase-space is only allowed if there are no time derivatives present
        in the system.  This can be checked with
        <phoebe.parameters.HierarchyParameter.is_time_dependent>(`consider_gaussian_process=consider_gaussian_process`).
        To interpolate in phases:

        ```
        b['fluxes@lc01@model'].interp_value(phases=0.5)
        ```

        Additionally, when interpolating in time but the time is outside the
        available range, phase-interpolation will automatically be attempted,
        with a warning raised via the <phoebe.logger>.

        Note: if this parameter is a model parameter where `sample_mode`
        was 'all', an error will be raised.  If `sample_mode` was 'n-sigma',
        then the median will be used when determining residuals.

        See also:
        * <phoebe.parameters.FloatArrayParameter.interp_quantity>

        Arguments
        ----------
        * `unit` (string or unit, optional, default=None): units to convert
            the *returned* value.  If not provided or None, will return in the
            default_units of the referenced parameter.  **NOTE**: to provide
            units on the *passed* value, you must send a quantity object (see
            `**kwargs` below).
        * `component` (string, optional, default=None): if interpolating in phases,
            `component` will be passed along to
            <phoebe.frontend.bundle.Bundle.to_phase>.
        * `period` (string/float, optional, default='period'): if interpolating
            in phases, `period` will be passed along to
            <phoebe.frontend.bundle.Bundle.to_phase>.
        * `dpdt` (string/float, optional, default='dpdt'): if interpolating in
            phases, `dpdt` will be passed along to
            <phoebe.frontend.bundle.Bundle.to_phase>.
        * `t0` (string/float, optional, default='t0_supconj'): if interpolating
            in phases, `t0` will be passed along to
             <phoebe.frontend.bundle.Bundle.to_phase>.
        * `consider_gaussian_process` (bool, optional, defult=True): whether
            to consider a system with gaussian process(es) as time-dependent.
        * `**kwargs`: see examples above, must provide a single
            qualifier-value pair to use for interpolation.  In most cases
            this will probably be time=value or wavelength=value.  If the value
            is provided as a quantity object, it will be converted to the default
            units of the referenced parameter prior to interpolation (enable
            a 'warning' <phoebe.logger> for conversion messages)

        Returns
        --------
        * (float or array) the interpolated value in value of `unit` if provided,
            or the <phoebe.parameters.FloatParameter.default_unit> of the
            referenced <phoebe.parameters.FloatArrayParameter>.  To return
            a quantity instead, see
            <phoebe.parameters.FloatArrayParameter.interp_quantity>.

        Raises
        --------
        * KeyError: if more than one qualifier is passed.
        * KeyError: if no qualifier is passed that belongs to the
            parent <phoebe.parameters.ParameterSet>.
        * KeyError: if the qualifier does not point to another
            <phoebe.parameters.FloatArrayParameter>.
        """
        # TODO: add support for non-linear interpolation (probably would need to use scipy)?

        return_quantity = kwargs.pop('return_quantity', False)
        parent_ps = kwargs.pop('parent_ps', self.get_parent_ps())
        bundle = kwargs.pop('bundle', self._bundle)

        if len(kwargs.keys()) > 1:
            raise KeyError("interp_value only takes a single qualifier-value pair, got other kwargs: {}".format(list(kwargs.keys())))

        qualifier, qualifier_interp_value = list(kwargs.items())[0]

        if qualifier in _singular_to_plural.keys():
            logger.warning("assuming {} instead of {}".format(_singular_to_plural.get(qualifier), qualifier))
            qualifier = _singular_to_plural.get(qualifier)

        if isinstance(qualifier_interp_value, str):
            # then assume its a twig and try to resolve
            # for example: time='t0_supconj'
            qualifier_interp_value = bundle.get_value(qualifier=qualifier_interp_value, context=['system', 'component'], **_skip_filter_checks)


        if qualifier not in parent_ps.qualifiers and not (qualifier=='phases' and 'times' in parent_ps.qualifiers):
            raise KeyError("'{}' not valid qualifier (must be one of {})".format(qualifier, parent_ps.qualifiers))

        if isinstance(qualifier_interp_value, u.Quantity):
            default_unit = parent_ps.get_parameter(qualifier=qualifier, **_skip_filter_checks).default_unit
            logger.warning("converting from provided quantity with units {} to default units ({}) of {}".format(qualifier_interp_value.unit, default_unit, qualifier))
            qualifier_interp_value = qualifier_interp_value.to(default_unit).value

        if qualifier=='times':
            # TODO: do we need to worry about units here?
            times = parent_ps.get_value(qualifier='times', **_skip_filter_checks)
            if np.any(qualifier_interp_value < times.min()) or np.any(qualifier_interp_value > times.max()):
                qualifier_interp_value_time = qualifier_interp_value
                qualifier = 'phases'
                qualifier_interp_value = bundle.to_phase(qualifier_interp_value_time, component=component, period=period, dpdt=dpdt, t0=t0)

                qualifier_interp_value_time_str = "({} -> {})".format(min(qualifier_interp_value_time), max(qualifier_interp_value_time)) if hasattr(qualifier_interp_value_time, '__iter__') else qualifier_interp_value_time
                qualifier_interp_value_str = "({} -> {})".format(min(qualifier_interp_value), max(qualifier_interp_value)) if hasattr(qualifier_interp_value, '__iter__') else qualifier_interp_value
                logger.warning("times={} outside of interpolation limits ({} -> {}), attempting to interpolate at phases={}".format(qualifier_interp_value_time_str, times.min(), times.max(), qualifier_interp_value_str))

        self_value = self.get_value()
        if len(self_value.shape) > 1:
            # then we need to check to see if this is in a model with sample_mode set
            if self.context != 'model':
                raise NotImplementedError("only 1D arrays supported unless in context='model' with sample_mode='n-sigma'")
            # do we want bundle or parent_ps here (for the case where doing scaling from run_compute)
            sample_mode = bundle.get_value(qualifier='sample_mode', context='model', model=self.model, default='none', **_skip_filter_checks)
            if '-sigma' in sample_mode:
                logger.warning("using median for interpolation for sample_mode='{}'".format(sample_mode))
                self_value = self_value[1]
            elif sample_mode == 'all' and self_value.shape[0] == 1:
                # then sample_num = 1, possibly from an optimizer solution
                self_value = self_value[0]
            else:
                raise NotImplementedError("iterpolation not supported for sample_mode='{}'".format(sample_mode))

        if qualifier=='phases':
            if bundle.hierarchy.is_time_dependent(consider_gaussian_process=consider_gaussian_process):
                raise ValueError("cannot interpolate in phase for time-dependent systems")

            times = parent_ps.get_value(qualifier='times', **_skip_filter_checks)
            phases = bundle.to_phase(times, component=component, period=period, dpdt=dpdt, t0=t0)

            sort = phases.argsort()

            value = np.interp(qualifier_interp_value, phases[sort], self_value[sort])

        else:

            qualifier_parameter = parent_ps.get(qualifier=qualifier)

            if not isinstance(qualifier_parameter, FloatArrayParameter):
                raise KeyError("'{}' does not point to a FloatArrayParameter".format(qualifier))

            qualifier_value = qualifier_parameter.get_value()
            sort = qualifier_value.argsort()

            value = np.interp(qualifier_interp_value, qualifier_value[sort], self_value[sort])

        if unit is not None:
            if return_quantity:
                return value*qualifier_parameter.default_unit.to(unit)
            else:
                return (value*qualifier_parameter.default_unit).to(unit).value
        else:
            if return_quantity:
                return value*qualifier_parameter.default_unit
            else:
                return value

    def interp_quantity(self, unit=None, **kwargs):
        """
        Interpolate to find the value in THIS array given a value from
        ANOTHER array in the SAME parent <phoebe.parameters.ParameterSet>
        (see <phoebe.parameters.Parameter.get_parent_ps>).

        See <phoebe.parameters.FloatArrayParameter.interp_value> for examples,
        this method calls interp_value and then returns the quantity object
        instead of the array.

        See also:
        * <phoebe.parameters.FloatArrayParameter.interp_value>

        Arguments
        ----------
        * `unit` (string or unit, optional, default=None): units to convert
            the *returned* value.  If not provided or None, will return in the
            default_units of the referenced parameter.  **NOTE**: to provide
            units on the *passed* value, you must send a quantity object (see
            `**kwargs` below).
        * `component` (string, optional): if interpolating in phases, `component`
            will be passed along to <phoebe.frontend.bundle.Bundle.to_phase>.
        * `period` (string/float, optional, default='period'): if interpolating
            in phases, `period` will be passed along to
            <phoebe.frontend.bundle.Bundle.to_phase>.
        * `t0` (string/float, optional): if interpolating in phases, `t0` will
            be passed along to <phoebe.frontend.bundle.Bundle.to_phase>.
        * `**kwargs`: see examples above, must provide a single
            qualifier-value pair to use for interpolation.  In most cases
            this will probably be time=value or wavelength=value.  If the value
            is provided as a quantity object, it will be converted to the default
            units of the referenced parameter prior to interpolation (enable
            a 'warning' <phoebe.logger> for conversion messages)

        Returns
        --------
        * (quantity) the interpolated value in value of `unit` if provided, or
            the <phoebe.parameters.FloatParameter.default_unit> of the
            referenced <phoebe.parameters.FloatArrayParameter>.  To return
            a float or array instead of a quantity object, see
            <phoebe.parameters.FloatArrayParameter.interp_value>.

        Raises
        --------
        * KeyError: if more than one qualifier is passed.
        * KeyError: if no qualifier is passed that belongs to the
            parent <phoebe.parameters.ParameterSet>.
        * KeyError: if the qualifier does not point to another
            <phoebe.parameters.FloatArrayParameter>.
        """

        return self.interp_value(unit=unit, return_quantity=True, **kwargs)

    def append(self, value, ignore_readonly=False):
        """
        Append a value to the end of the array.

        Arguments
        ---------
        * `value` (float): the float to append to the end of the current array
        """
        # check units
        if isinstance(value, u.Quantity):
            value = value.to(self.default_unit).value

        if isinstance(value, nparray.ndarray):
            value = value.to_array()

        new_value = np.append(self.get_value(), value) * self.default_unit
        self.set_value(new_value, ignore_readonly=ignore_readonly)

    def set_index_value(self, index, value, **kwargs):
        """
        Set the value of the array at a given index.

        Arguments
        -----------
        * `index` (int): the index of the value to be replaced
        * `value` (float): the value to be replaced
        * `**kwargs`: IGNORED
        """
        if isinstance(value, u.Quantity):
            value = value.to(self.default_unit).value
        elif isinstance(value, str):
            value = float(value)
        #else:
            #value = value*self.default_unit
        lst =self.get_value()#.value
        lst[index] = value
        self.set_value(lst, ignore_readonly=kwargs.get('ignore_readonly', False))

    def __add__(self, other):
        if not (isinstance(other, list) or isinstance(other, np.ndarray)):
            return super(FloatArrayParameter, self).__add__(other)

        # then we have a list, so we want to append to the existing value
        return np.append(self.get_value(), np.asarray(other))

    def __sub__(self, other):
        if not (isinstance(other, list) or isinstance(other, np.ndarray)):
            return super(FloatArrayParameter, self).__add__(other)

        # then we have a list, so we want to append to the existing value
        return np.array([v for v in self.get_value() if v not in other])

    # def set_value_at_time(self, time, value, **kwargs):
    #     """
    #     """
    #     parent_ps = self.get_parent_ps()
    #     times_param = parent_ps.get_parameter(qualifier='times')
    #     index = np.where(times_param.get_value()==time)[0][0]
    #
    #     self.set_index_value(index, value, **kwargs)


    #~ def at_time(self, time):
        #~ """
        #~ looks for a parameter with qualifier time that shares all the same meta data and
        #~ """
        #~ raise NotImplementedError

    def _check_type(self, value):
        """
        """
        if isinstance(value, u.Quantity):
            if isinstance(value.value, float) or isinstance(value.value, int):
                value = np.array([value.value])*value.unit

        # if isinstance(value, str):
            # value = np.fromstring(value)

        elif isinstance(value, list) or isinstance(value, tuple):
            value = np.asarray(value)

        elif isinstance(value, float) or isinstance(value, int):
            value = np.array([value])

        elif isinstance(value, dict) and 'nparray' in value.keys():
            value = nparray.from_dict(value)

        elif not (isinstance(value, list) or isinstance(value, tuple) or isinstance(value, np.ndarray) or isinstance(value, nparray.ndarray)):
            # TODO: probably need to change this to be flexible with all the cast_types
            raise TypeError("value '{}' ({}) could not be cast to array".format(value, type(value)))

        if len(value) and self.required_shape is not None:
            if len(value.shape) != len(self.required_shape):
                raise TypeError("value must have {} dimensions (value.shape={}, required_shape={})".format(len(self.required_shape), value.shape, self.required_shape))

            for i, ilength in enumerate(self.required_shape):
                if ilength == 0 or ilength is None:
                    continue
                if value.shape[i] != ilength:
                    raise TypeError("dimension {} must have length {} (value.shape={}, required_shape={})".format(i, ilength, value.shape, self.required_shape))

        return value

    def set_property(self, **kwargs):
        """
        Set any property of the underlying [nparray](https://github.com/kecnry/nparray/tree/1.0.0)
        object.

        Example:
        ```py
        param.set_value(start=10, stop=20)
        ```

        Arguments
        ----------
        * `**kwargs`: properties to be set on the underlying nparray object.

        Raises
        -------
        * ValueError: if the value is not an nparray object.
        """
        if not isinstance(self._value, nparray.ndarray):
            raise ValueError("value is not a nparray object")

        for property, value in kwargs.items():
            setattr(self._value, property, value)


class ArrayParameter(Parameter):
    def __init__(self, *args, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        super(ArrayParameter, self).__init__(*args, **kwargs)

        self.set_value(kwargs.get('value', []), ignore_readonly=True)

        self._dict_fields_other = ['description', 'value', 'visible_if', 'copy_for', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def append(self, value):
        """
        Append a value to the end of the array.

        Arguments
        ---------
        * `value`: the float to append to the end of the current array
        """
        if isinstance(value, nparray.ndarray):
            value = value.to_array()

        new_value = np.append(self.get_value(), value)
        self.set_value(new_value)

    #~ def at_time(self, time):
        #~ """
        #~ looks for a parameter with qualifier time that shares all the same meta data and
        #~ """
        #~ raise NotImplementedError

    def get_value(self, **kwargs):
        """
        Get the current value of the <phoebe.parameters.ArrayParameter>.

        **default/override values**: if passing a keyword argument with the same
            name as the Parameter qualifier (see
            <phoebe.parameters.Parameter.qualifier>), then the value passed
            to that keyword argument will be returned **instead of** the current
            value of the Parameter.  This is mostly used internally when
            wishing to override values sent to
            <phoebe.frontend.bundle.Bundle.run_compute>, for example.

        Arguments
        ----------
        * `**kwargs`: passing a keyword argument that matches the qualifier
            of the Parameter, will return that value instead of the stored value.
            See above for how default values are treated.

        Returns
        --------
        * (np array) the current or overridden value of the Parameter
        """
        default = super(ArrayParameter, self).get_value(**kwargs)
        if default is not None: return default

        if isinstance(self._value, nparray.ndarray):
            return self._value.to_array()
        else:
            return self._value

    @send_if_client
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.ArrayParameter>.

        Arguments
        ----------
        * `value` (Array): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to the correct type
            or is not a valid value for the Parameter.
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self._value)
        self._value = np.array(value)


class HierarchyParameter(StringParameter):
    def __init__(self, value, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        dump = kwargs.pop('qualifier', None)
        kwargs.setdefault('advanced', True)
        super(HierarchyParameter, self).__init__(qualifier='hierarchy', value=value, **kwargs)

    def __repr__(self):
        return "<HierarchyParameter: {}>".format(self.get_value())

    def __str__(self):
        #~ return self.get_value().replace('(', '\n\t').replace(')', '\n')
        #~ def _print_item(item, tab, str_):
            #~ if isinstance(item, list):
                #~ tab += 1
                #~ for child in item:
                    #~ str_ += _print_item(child, tab, str_)
            #~ else:
                #~ return str_ + '\n' + '\t'*tab + item
        #~
        #~ str_ = ''
        #~ for item in self._parse_repr():
            #~ tab = 0
            #~ str_ += _print_item(str(item), tab, '')
#~
        #~ return str_
        if not len(self.get_value()):
            return 'NO HIERARCHY'
        else:
            return json.dumps(self._parse_repr(), indent=4).replace(',','').replace('[','').replace(']','').replace('"', '').replace('\n\n','\n')

    @send_if_client
    def set_value(self, value, update_cache=True, **kwargs):
        """
        Set the current value of the <phoebe.parameters.HierarchyParameter>.

        Arguments
        ----------
        * `value` (string): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to a string.
        """
        self._readonly_check(**kwargs)

        # TODO: check to make sure valid

        _orig_value = _deepcopy(self.get_value())

        try:
            value = str(value)
        except:
            raise ValueError("cannot cast to string")
        else:
            self._value = value

        if update_cache:
            self._update_cache()

    def _clear_cache(self):
        """
        """
        self._is_binary = {}
        self._is_contact_binary = {}
        self._meshables = []

    def _update_cache(self):
        """
        """
        # update cache for is_binary and is_contact_binary
        self._clear_cache()
        if self._bundle is not None:
            self._meshables = self._compute_meshables()

            # for comp in self.get_components():
            for comp in self._bundle.components:
                if comp == '_default':
                    continue
                self._is_binary[comp] = self._compute_is_binary(comp)
                self._is_contact_binary[comp] = self._compute_is_contact_binary(comp)


    def _parse_repr(self):
         """
         turn something like "orbit:outer(orbit:inner(star:starA, star:starB), star:starC)"
         into ['orbit:outer', ['orbit:inner', ['star:starA', 'star:starB'], 'star:starC']]
         """

         repr_ = self.get_value()
         repr_str = '["{}"]'.format(repr_.replace(', ', '", "').replace('(', '", ["').replace(')', '"]')).replace(']"', '"]').replace('""', '"').replace(']"', ']')
         return json.loads(repr_str)

    def _recurse_find_trace(self, structure, item, trace=[]):
        """
        given a nested structure from _parse_repr and find the trace route to get to item
        """

        try:
            i = structure.index(item)
        except ValueError:
            for j,substructure in enumerate(structure):
                if isinstance(substructure, list):
                    return self._recurse_find_trace(substructure, item, trace+[j])
        else:
            return trace+[i]

    def _get_by_trace(self, structure, trace):
        """
        retrieve an item from the nested structure from _parse_repr given a trace (probably modified from _recurse_find_trace)
        """
        for i in trace:
            structure = structure[i]

        return structure

    def _get_structure_and_trace(self, component):
        """
        """
        obj = self._bundle.filter(component=component, context='component', check_visible=False)
        our_item = '{}:{}'.format(obj.kind, component)


        repr_ = self.get_value()
        structure = self._parse_repr()

        trace = self._recurse_find_trace(structure, our_item)

        return structure, trace, our_item

    def rename_component(self, old_component, new_component):
        """
        Swap a component in the <phoebe.parameters.HierarchyParameter>.

        Note that this does NOT update component tags within the
        <phoebe.parametes.ParameterSet> or <phoebe.frontend.bundle.Bundle>.
        To change the name of a component, use
        <phoebe.frontend.bundle.Bundle.rename_component> instead.

        If calling this manually, make sure to update all other tags
        or components and update the cache of the hierarchy.

        Arguments
        ----------
        * `old_component` (string): the current name of the component in the
            hierarchy
        * `new_component` (string): the replaced component
        """
        kind = self.get_kind_of(old_component)
        value = self.get_value()
        # TODO: this could still cause issues if the names of components are
        # contained in other components (ie starA, starAB)
        value = value.replace("{}:{}".format(kind, old_component), "{}:{}".format(kind, new_component))
        # delay updating cache until after the bundle
        # has had a chance to also change its component tags
        self.set_value(value, update_cache=False)


    def get_components(self):
        """
        Return a list of all components in the <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_top>
        * <phoebe.parameters.HierarchyParameter.get_stars>
        * <phoebe.parameters.HierarchyParameter.get_envelopes>
        * <phoebe.parameters.HierarchyParameter.get_orbits>
        * <phoebe.parameters.HierarchyParameter.get_meshables>

        Returns
        -------
        * (list of strings)
        """
        l = re.findall(r"[\w']+", self.get_value())
        return l[1::2]

    def get_top(self):
        """
        Return the top-level component in the <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_components>
        * <phoebe.parameters.HierarchyParameter.get_stars>
        * <phoebe.parameters.HierarchyParameter.get_envelopes>
        * <phoebe.parameters.HierarchyParameter.get_orbits>
        * <phoebe.parameters.HierarchyParameter.get_meshables>

        Returns
        -------
        * (string)
        """
        return str(self._parse_repr()[0].split(':')[1])

    def get_stars(self):
        """
        Return a list of all components with kind='star' in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_components>
        * <phoebe.parameters.HierarchyParameter.get_top>
        * <phoebe.parameters.HierarchyParameter.get_envelopes>
        * <phoebe.parameters.HierarchyParameter.get_orbits>
        * <phoebe.parameters.HierarchyParameter.get_meshables>

        Returns
        -------
        * (list of strings)
        """
        l = re.findall(r"[\w']+", self.get_value())
        # now search for indices of star and take the next entry from this flat list
        return [l[i+1] for i,s in enumerate(l) if s=='star']

    def get_envelopes(self):
        """
        Return a list of all components with kind='envelope' in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_components>
        * <phoebe.parameters.HierarchyParameter.get_top>
        * <phoebe.parameters.HierarchyParameter.get_stars>
        * <phoebe.parameters.HierarchyParameter.get_orbits>
        * <phoebe.parameters.HierarchyParameter.get_meshables>

        Returns
        -------
        * (list of strings)
        """
        l = re.findall(r"[\w']+", self.get_value())
        # now search for indices of star and take the next entry from this flat list
        return [l[i+1] for i,s in enumerate(l) if s=='envelope']

    def get_orbits(self):
        """
        Return a list of all components with kind='orbit' in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_components>
        * <phoebe.parameters.HierarchyParameter.get_top>
        * <phoebe.parameters.HierarchyParameter.get_stars>
        * <phoebe.parameters.HierarchyParameter.get_envelopes>
        * <phoebe.parameters.HierarchyParameter.get_meshables>

        Returns
        -------
        * (list of strings)
        """
        #~ l = re.findall(r"[\w']+", self.get_value())
        # now search for indices of orbit and take the next entry from this flat list
        #~ return [l[i+1] for i,s in enumerate(l) if s=='orbit']
        orbits = []
        for star in self.get_stars():
            parent = self.get_parent_of(star)
            if parent not in orbits and parent!='component' and parent is not None:
                orbits.append(parent)
        return orbits

    def _compute_meshables(self):
        l = re.findall(r"[\w']+", self.get_value())
        # now search for indices of star and take the next entry from this flat list
        meshables = [l[i+1] for i,s in enumerate(l) if s in ['star', 'envelope']]

        # now we want to remove any star which has a sibling envelope
        has_sibling_envelope = []
        for item in meshables:
            if self.get_sibling_of(item, kind='envelope'):
                has_sibling_envelope.append(item)

        return [m for m in meshables if m not in has_sibling_envelope]

    def get_meshables(self):
        """
        Return a list of all components that are meshable (generally stars,
        but also handles the envelope for a contact binary)
        in the <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.is_meshable>
        * <phoebe.parameters.HierarchyParameter.get_components>
        * <phoebe.parameters.HierarchyParameter.get_top>
        * <phoebe.parameters.HierarchyParameter.get_stars>
        * <phoebe.parameters.HierarchyParameter.get_envelopes>
        * <phoebe.parameters.HierarchyParameter.get_orbits>

        Returns
        -------
        * (list of strings)
        """
        if not len(self._meshables):
            self._update_cache()

        return self._meshables

    def is_meshable(self, component):
        """
        Determine if `component` is one of
        <phoebe.parameters.HierarchyParameter.get_meshables>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_meshables>

        Arguments
        ------------
        * `component` (string): the name of the component to check.

        Returns
        ----------
        * (bool)
        """
        return component in self.get_meshables()

    def get_parent_of(self, component):
        """
        Get the parent of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.

        Returns
        ---------
        * (string)
        """
        # example:
        # - self.get_value(): "orbit:outer(orbit:inner(star:starA, star:starB), star:starC)"
        # - component: "starA"
        # - needs to return "inner"

        if component is None:
            return self.get_top()


        structure, trace, item = self._get_structure_and_trace(component)
        # trace points us to our_item at self._get_by_trace(structure, trace)
        # so to get the parent, if our trace is [1,1,0] we want to use [1, 0] which is trace[:-2]+[trace[-2]-1]


        #~ print "***", trace
        if len(trace)<=1:
            return None

        return str(self._get_by_trace(structure, trace[:-2]+[trace[-2]-1]).split(':')[-1])

    def get_sibling_of(self, component, kind=None):
        """
        Get the sibling of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        If there is more than one sibling, the first result will be returned.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.
        * `kind` (string, optional): filter to match the kind of the component.

        Returns
        ---------
        * (string)
        """
        siblings = self.get_siblings_of(component, kind=kind)
        if not len(siblings):
            return None
        else:
            return siblings[0]


    def get_siblings_of(self, component, kind=None):
        """
        Get the siblings of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.
        * `kind` (string, optional): filter to match the kind of the component.

        Returns
        ---------
        * (list of strings)
        """

        structure, trace, item = self._get_structure_and_trace(component)
        #item_kind, item_label = item.split(':')

        parent_label = self.get_parent_of(component)
        siblings = self.get_children_of(parent_label, kind=kind)

        #self_ind = siblings.index(component)
        if component in siblings:
            siblings.remove(component)

        if not len(siblings):
            return []
        else:
            return siblings

    def get_envelope_of(self, component):
        """
        Get the parent-envelope of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.

        Returns
        ---------
        * (string)
        """
        envelopes = self.get_siblings_of(component, 'envelope')
        if not len(envelopes):
            return []
        else:
            return envelopes[0]

    def get_stars_of_sibling_of(self, component):
        """
        Get the stars under the sibling of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        This is the same as <phoebe.parameters.Hierarchy.get_sibling_of> except
        if a sibling is in an orbit, this will recursively follow the tree to
        return a list of all stars under that orbit.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.

        Returns
        ---------
        * (string)
        """
        sibling = self.get_sibling_of(component)

        if sibling in self.get_stars():
            return sibling

        stars = [child for child in self.get_stars_of_children_of(sibling)]

        # TODO: do we need to make sure there aren't duplicates?
        # return list(set(stars))

        return stars


    def get_children_of(self, component, kind=None):
        """
        Get the children of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.
        * `kind` (string, optional): filter to match the kind of the component.

        Returns
        ---------
        * (list of strings)
        """

        structure, trace, item = self._get_structure_and_trace(component)
        item_kind, item_label = item.split(':')

        if isinstance(kind, str):
            kind = [kind]

        if item_kind not in ['orbit']:
            # return None
            return []
        else:
            items = self._get_by_trace(structure, trace[:-1]+[trace[-1]+1])
            # we want to ignore suborbits
            #return [str(ch.split(':')[-1]) for ch in items if isinstance(ch, unicode)]
            return [str(ch.split(':')[-1]) for ch in items if isinstance(ch, str) and (kind is None or ch.split(':')[0] in kind)]

    def get_stars_of_children_of(self, component):
        """
        Get the stars under the children of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        This is the same as <phoebe.parameters.Hierarchy.get_children_of> except
        if any of the children is in an orbit, this will recursively follow the tree to
        return a list of all stars under that orbit.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>
        * <phoebe.parameters.HierarchyParameter.get_child_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.

        Returns
        ---------
        * (string)
        """

        stars = self.get_stars()
        orbits = self.get_orbits()
        stars_children = []

        for child in self.get_children_of(component):
            if child in stars:
                stars_children.append(child)
            elif child in orbits:
                stars_children += self.get_stars_of_children_of(child)
            else:
                # maybe an envelope or eventually spot, ring, etc
                pass

        return stars_children



    def get_child_of(self, component, ind, kind=None):
        """
        Get the child (by index) of a component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        See also:
        * <phoebe.parameters.HierarchyParameter.get_parent_of>
        * <phoebe.parameters.HierarchyParameter.get_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_siblings_of>
        * <phoebe.parameters.HierarchyParameter.get_envelope_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_sibling_of>
        * <phoebe.parameters.HierarchyParameter.get_children_of>
        * <phoebe.parameters.HierarchyParameter.get_stars_of_children_of>

        Arguments
        ----------
        * `component` (string): the name of the component under which to search.
        * `ind` (int): the index of the child to return (starting at 0)
        * `kind` (string, optional): filter to match the kind of the component.

        Returns
        ---------
        * (string)
        """
        children = self.get_children_of(component, kind=kind)
        if children is None:
            return None
        else:
            return children[ind]


    def get_primary_or_secondary(self, component, return_ind=False):
        """
        Return whether a given component is the 'primary' or 'secondary'
        component in its parent orbit, according to the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
        <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        Arguments
        ----------
        * `component` (string): the name of the component.
        * `return_ind` (bool, optional, default=False): if `True`, this
            will return `0` instead of `'primary'` and `1` instead of
            `'secondary'`.

        Returns
        --------
        * (string or int): either 'primary'/'secondary' or 0/1 depending on the
            value of `return_ind`.
        """
        parent = self.get_parent_of(component)
        if parent is None:
            # then this is a single component, not in a binary
            return 'primary'

        children_of_parent = self.get_children_of(parent)

        ind = children_of_parent.index(component)

        if ind > 1:
            return None

        if return_ind:
            return ind + 1

        return ['primary', 'secondary'][ind]

    def get_kind_of(self, component):
        """
        Return the kind of a given component in the
        <phoebe.parameters.HierarchyParameter>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        Arguments
        ----------
        * `component` (string): the name of the component.

        Returns
        --------
        * (string): the kind (star, orbit, envelope, etc) of the component
        """
        structure, trace, item = self._get_structure_and_trace(component)
        item_kind, item_label = item.split(':')

        return item_kind


    def _compute_is_contact_binary(self, component):
        """
        """
        if 'envelope' not in self.get_value():
            return False

        if component not in self.get_components():
            # TODO: this can probably at least check to see if is itself
            # an envelope?
            return False

        return self.get_kind_of(component)=='envelope' or (self.get_sibling_of(component, kind='envelope') is not None)

    def is_contact_binary(self, component):
        """
        Return whether a given component is part of a contact binary,
        according to the <phoebe.parameters.HierarchyPararameter>.
        This is especially useful for <phoebe.parameters.ConstraintParameter>.

        This is done by checking whether any of the component's siblings is
        an envelope.  See <phoebe.parameters.HierarchyParameter.get_siblings_of>
        and <phoebe.parameters.HierarchyParameter.get_kind_of>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        Arguments
        ----------
        * `component` (string): the name of the component.

        Returns
        --------
        * (bool): whether the given component is part of a contact binary.
        """
        if component not in self._is_contact_binary.keys():
            self._update_cache()

        return self._is_contact_binary.get(component)

    def _compute_is_binary(self, component):
        """
        """
        if component not in self.get_components():
            # TODO: is this the best fallback?
            return True

        if len(self.get_stars())==1:
            return False

        return self.get_kind_of(self.get_parent_of(component))=='orbit'

    def is_binary(self, component):
        """
        Return whether a given component is part of a binary system,
        according to the <phoebe.parameters.HierarchyPararameter>.
        This is especially useful for <phoebe.parameters.ConstraintParameter>.

        This is done by checking whether the component's parent is an orbit.
        See <phoebe.parameters.HierarchyParameter.get_parent_of> and
        <phoebe.parameters.HierarchyParameter.get_kind_of>.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        Arguments
        ----------
        * `component` (string): the name of the component.

        Returns
        --------
        * (bool): whether the given component is part of a contact binary.
        """
        if component not in self._is_binary.keys():
            self._update_cache()

        return self._is_binary.get(component)

    def is_misaligned(self):
        """
        Return whether the system is misaligned.

        Returns
        ---------
        * (bool): whether the system is misaligned.
        """
        for component in self.get_stars():
            if self._bundle.get_value(qualifier='pitch', component=component, context='component') != 0:
                return True
            if self._bundle.get_value(qualifier='yaw', component=component, context='component') != 0:
                return True

        return False

    def is_time_dependent(self, consider_gaussian_process=True):
        """
        Return whether the system has any time-dependent parameters (other than
        phase-dependence).

        This will return True if any of the following conditions are met:
        * `dpdt` is non-zero
        * `dperdt` is non-zero
        * a feature (eg. spot) is attached to an asynchronous star (with
            non-unity value for `syncpar`).
        * a gaussian_process feature is attached to any dataset, unless
            `consider_gaussian_process` is False.

        To access the HierarchyParameter from the Bundle, see
         <phoebe.frontend.bundle.Bundle.get_hierarchy>.

        Arguments
        ---------
        * `consider_gaussian_process` (bool, optional, defult=True): whether
            to consider a system with gaussian process(es) as time-dependent

        Returns
        ---------
        * (bool): whether the system is time-dependent
        """
        orbits = self.get_orbits()
        for orbit in orbits:
            if self._bundle.get_value(qualifier='dpdt', component=orbit, context='component') != 0:
                return True
            if self._bundle.get_value(qualifier='dperdt', component=orbit, context='component') != 0:
                return True
            # if conf.devel and self._bundle.get_value(qualifier='deccdt', component=orbit, context='component') != 0:
            #     return True

        if len(orbits):
            for component in self.get_stars():
                if self._bundle.get_value(qualifier='syncpar', component=component, context='component') != 1 and len(self._bundle.filter(context='feature', component=component)):
                    # spots on asynchronous stars
                    return True

        # TODO: allow passing compute to do only enabled features attached to enabled datasets?
        if consider_gaussian_process and len(self._bundle.filter(kind='gaussian_process', context='feature', **_skip_filter_checks).features):
            return True

        return False


class ConstraintParameter(Parameter):
    """
    One side of a constraint (not an equality)

    qualifier: constrained parameter
    value: expression
    """
    def __init__(self, bundle, value, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        # the super call is popping default_unit, so we'll access it first
        default_unit_kwargs = kwargs.get('default_unit', None)
        super(ConstraintParameter, self).__init__(qualifier=kwargs.pop('qualifier', None), value=value, description=kwargs.pop('description', 'constraint'), **kwargs)

        # usually its the bundle's job to attach param._bundle after the
        # creation of a parameter.  But in this case, having access to the
        # bundle is necessary in order to intialize and set the value
        self._bundle = bundle
        if isinstance(value, ConstraintParameter):
            if default_unit_kwargs is None:
                default_unit = value.result.unit
            else:
                default_unit = default_unit_kwargs
            value = value.get_value()

        else:
            default_unit = kwargs.get('default_unit', u.dimensionless_unscaled)

        if 'constraint_addl_vars' in kwargs.keys():
            self._addl_vars = [ConstraintVar(bundle, twig) for twig in kwargs.get('constraint_addl_vars', [])]
        else:
            self._addl_vars = [ConstraintVar(bundle, v.twig) for v in kwargs.get('addl_vars', [])]
        self._vars = self._addl_vars
        self._var_params = None
        self._addl_var_params = None
        self._constraint_func = kwargs.get('constraint_func', None)
        self._constraint_kwargs = kwargs.get('constraint_kwargs', {})
        self._in_solar_units = kwargs.get('in_solar_units', False)
        self.set_value(value, ignore_readonly=True)
        self.set_default_unit(default_unit)
        self._dict_fields_other = ['description', 'value', 'default_unit', 'constraint_func', 'constraint_kwargs', 'constraint_addl_vars', 'in_solar_units', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    @property
    def is_visible(self):
        """
        Return whether the <phoebe.parameters.ConstraintParameter> is visible
        by checking on the visibility of the constrained parameter.

        See:
        * <phoebe.parameters.ConstraintParameter.constrained_parameter>
        * <phoebe.parameters.Parameter.is_visible>
        * <phoebe.parameters.Parameter.visible_if>

        Returns
        -------
        * (bool)
        """
        return self.constrained_parameter.is_visible

    @property
    def constraint_func(self):
        """
        Access the constraint_func tag of this
        <phoebe.parameters.ConstraintParameter>.

        Returns
        -------
        * (str) the constraint_func tag of this Parameter.
        """
        return self._constraint_func

    @property
    def constraint_kwargs(self):
        """
        Access the keyword arguments sent to the constraint.

        Returns
        ------
        * (dict)
        """
        return self._constraint_kwargs

    @property
    def constraint_addl_vars(self):
        return [v.twig for v in self.addl_vars.to_list()]

    @property
    def in_solar_units(self):
        """
        """
        return self._in_solar_units

    @property
    def vars(self):
        """
        Return all the variables in this <phoebe.parameters.ConstraintParameter>
        as a <phoebe.parameters.ParameterSet>.

        Return
        ------
        * (<phoebe.parameters.ParameterSet>): ParameterSet of all variables in
            the expression for this constraint.
        """
        # cache _var_params
        if self._var_params is None:
            self._var_params = ParameterSet([var.get_parameter() for var in self._vars])
        return self._var_params

    @property
    def addl_vars(self):
        """
        return all the additional (those that may be needed if flipped) variables in a PS
        """
        # cache _var_params
        if self._addl_var_params is None:
            self._addl_var_params = ParameterSet([var.get_parameter() for var in self._addl_vars])
        return self._addl_var_params

    def _get_var(self, param=None, **kwargs):
        if not isinstance(param, Parameter):
            if isinstance(param, str) and 'twig' not in kwargs.keys():
                kwargs['twig'] = param

            param = self.get_parameter(**kwargs)

        varids = [var.unique_label for var in self._vars]
        if param.uniqueid not in varids:
            varids = [var.unique_label for var in self._addl_vars]
            if param.uniqueid not in varids:
                raise KeyError("{} was not found in expression".format(param.uniquetwig))
            return self._addl_vars[varids.index(param.uniqueid)]
        return self._vars[varids.index(param.uniqueid)]



    def _parse_expr(self, expr):

        # the expression currently has twigs between curly braces,
        # we need to extract these and turn each into a ConstraintVar
        # so that we actually store the uniqueid of the parameter,
        # but always display the /current/ uniquetwig in the expression

        vars_ = []
        lbls = re.findall(r'\{.[^{}]*\}', expr)

        for lbl in lbls:
            twig = lbl.replace('{', '').replace('}', '')
            #~ print "ConstraintParameter._parse_expr lbl: {}, twig: {}".format(lbl, twig)
            var = ConstraintVar(self._bundle, twig)

            # TODO: if var.is_param, we need to make it read-only and required by this constraint, etc

            vars_.append(var)
            expr = expr.replace(lbl, var.safe_label)

        if self.qualifier:
            #~ print "***", self._bundle.__repr__(), self.qualifier, self.component
            ps = self._bundle.exclude(context='constraint', **_skip_filter_checks).filter(qualifier=self.qualifier, component=self.component, dataset=self.dataset, feature=self.feature, model=self.model, **_skip_filter_checks)
            if len(ps) == 1:
                constrained_parameter = ps.get_parameter(**_skip_filter_checks)
            else:
                raise KeyError("could not find single match for {} (found {})".format({'qualifier': self.qualifier, 'component': self.component, 'dataset': self.dataset, 'feature': self.feature, 'model': self.model}, ps.twigs))


            var = ConstraintVar(self._bundle, constrained_parameter.twig)
            vars_.append(var)

        return expr, vars_

    @property
    def constrained_parameter(self):
        """
        Access the <phoebe.parameters.Parameter> that is constrained (i.e.
        solved for) by this <phoebe.parameters.ConstraintParameter>.

        This is identical to
        <phoebe.parameters.ConstraintParameter.get_constrained_parameter>.

        See also:
        * <phoebe.parameters.ConstraintParameter.flip_for>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>

        Returns
        -------
        * (<phoebe.parameters>parameter>)
        """
        # try:
        if True:
            return self.get_constrained_parameter()
        # except: # TODO exception type
            # return None

    def get_constrained_parameter(self):
        """
        Access the <phoebe.parameters.Parameter> that is constrained (i.e.
        solved for) by this <phoebe.parameters.ConstraintParameter>.

        This is identical to
        <phoebe.parameters.ConstraintParameter.constrained_parameter>.

        See also:
        * <phoebe.parameters.ConstraintParameter.flip_for>
        * <phoebe.frontend.bundle.Bundle.flip_constraint>

        Returns
        -------
        * (<phoebe.parameters.Parameter>)
        """
        return self.get_parameter(qualifier=self.qualifier, component=self.component, dataset=self.dataset, check_visible=False)

    def get_parameter(self, twig=None, **kwargs):
        """
        Access one of the <phoebe.parameters.Parameter> object that is a variable
        in the <phoebe.parameters.ConstraintParameter>.

        **NOTE**: if the filtering results in more than one result, the first
        will be taken instead of raising an error.

        Arguments
        ----------
        * `twig` (string, optional): twig to use for filtering.  See
            <phoebe.parameters.ParameterSet.get_parameter>
        * `**kwargs`: other tags used for filtering.  See
            <phoebe.parameters.ParameterSet.get_parameter>

        Returns
        --------
        * (<phoebe.parameters.Parameter>)

        Raises
        -------
            * KeyError: if the filtering results in 0 matches
        """
        kwargs['twig'] = twig
        kwargs.setdefault('check_default', False)
        kwargs.setdefault('check_visible', False)
        kwargs.setdefault('check_advanced', False)
        kwargs.setdefault('check_single', False)
        vars = self.vars + self.addl_vars
        ps = vars.filter(**kwargs)
        if len(ps)==1:
            return ps.get(check_visible=False, check_default=False,
                          check_advanced=False, check_single=False)
        elif len(ps) > 1:
            # TODO: is this safe?  Some constraints may have a parameter listed
            # twice, so we can do this then, but maybe should check to make sure
            # all items have the same uniqueid?  Maybe check len(ps.uniqueids)?
            return ps.to_list()[0]
        else:
            if self._bundle is not None:
                logger.debug("ConstraintParameter.get_parameter: reverting to filtering on bundle, could not {} find in {}".format(kwargs, vars.twigs))
                kwargs['context'] = [c for c in self._bundle.contexts if c!='constraint']
                return self._bundle.get_parameter(**kwargs)
            raise ValueError("no result found for {} in bundle after checking in {}".format(kwargs, vars.twigs))

    @property
    def default_unit(self):
        """
        Return the default unit for the <phoebe.parameters.ConstraintParameter>.

        This is identical to <phoebe.parameters.ConstraintParameter.get_default_unit>.

        See also:
        * <phoebe.parameters.ConstraintParameter.set_default_unit>

        Returns
        --------
        * (unit): the current default units.
        """
        return self._default_unit

    def get_default_unit(self):
        """
        Return the default unit for the <phoebe.parameters.ConstraintParameter>.

        This is identical to <phoebe.parameters.ConstraintParameter.default_unit>.

        See also:
        * <phoebe.parameters.ConstraintParameter.set_default_unit>

        Returns
        --------
        * (unit): the current default units.
        """
        return self.default_unit

    def set_default_unit(self, unit):
        """
        Set the default unit for the <phoebe.parameters.ConstraintParameter>.

        See also:
        * <phoebe.parameters.ConstraintParameter.get_default_unit>

        Arguments
        --------
        * `unit` (unit or valid string): the desired new units.  If the Parameter
            currently has default units, then the new units must be compatible
            with the current units

        Raises
        -------
        * Error: if the new and current units are incompatible.
        """
        # TODO: check to make sure can convert from current default unit (if exists)
        if isinstance(unit, str):
            unit = u.Unit(unit)

        if not _is_unit(unit):
            raise TypeError("unit must be a Unit")

        if hasattr(self, '_default_unit') and self._default_unit is not None:
            # we won't use a try except here so that the error comes from astropy
            check_convert = self._default_unit.to(unit)


        self._default_unit = unit

    #@send_if_client   # TODO: this breaks
    def set_value(self, value, **kwargs):
        """
        Set the current value of the <phoebe.parameters.ConstraintParameter>.

        Arguments
        ----------
        * `value` (string): the new value of the Parameter.
        * `**kwargs`: IGNORED

        Raises
        ---------
        * ValueError: if `value` could not be converted to a string.
        * Error: if `value` could not be parsed into a valid constraint expression.
        """
        self._readonly_check(**kwargs)

        _orig_value = _deepcopy(self.get_value())

        if self._bundle is None:
            raise ValueError("ConstraintParameters must be attached from the bundle, and cannot be standalone")
        value = str(value)
        # if the user wants to see the expression, we'll replace all
        # var.safe_label with var.curly_label
        self._value, self._vars = self._parse_expr(value)
        # reset the cached version of the PS - will be recomputed on next request
        self._var_params = None
        self._addl_var_params = None
        #~ print "***", self.uniquetwig, self.uniqueid

    def _update_bookkeeping(self):
        # do bookkeeping on parameters
        self._remove_bookkeeping()
        # logger.debug("ConstraintParameter {} _update_bookkeeping".format(self.twig))
        for param in self.vars.to_list():
            if param.qualifier == self.qualifier and param.component == self.component and param.dataset == self.dataset:
                # then this is the currently constrained parameter
                param._is_constraint = self.uniqueid
                if self.uniqueid in param._in_constraints:
                    param._in_constraints.remove(self.uniqueid)
            else:
                # then this is a constraining parameter
                if self.uniqueid not in param._in_constraints:
                    param._in_constraints.append(self.uniqueid)

        for param in self.addl_vars.to_list():
            if param.qualifier == self.qualifier and param.component == self.component and param.dataset == self.dataset:
                # then this is the currently constrained parameter
                param._is_constraint = self.uniqueid

                if self.uniqueid in param._in_constraints:
                    param._in_constraints.remove(self.uniqueid)

    def _remove_bookkeeping(self):
        # logger.debug("ConstraintParameter {} _remove_bookkepping".format(self.twig))
        vars = self.vars + self.addl_vars
        for param in vars.to_list():
            if hasattr(param, '_is_constraint') and param._is_constraint == self.uniqueid:
                param._is_constraint = None
            if self.uniqueid in param._in_constraints:
                logger.debug("removing {} from {}.in_constraints".format(self.twig, param.twig))
                param._in_constraints.remove(self.uniqueid)

    @property
    def expr(self):
        """
        Return the expression of the <phoebe.parameters.ConstraintParameter>.
        This is just a shortcut to
        <phoebe.parameters.ConstraintParameter.get_value>.

        Returns
        -------
        * (float)
        """
        return self.get_value()

    def get_value(self):
        """
        Return the expression/value of the
        <phoebe.parameters.ConstraintParameter>.

        Returns
        -------
        * (float)
        """
        # for access to the sympy-safe expr, just use self._expr
        expr = self._value
        if expr is not None:
            vars = self._vars + self._addl_vars
            for var in vars:
                # update to current unique twig
                var.update_user_label()  # update curly label
                #~ print "***", expr, var.safe_label, var.curly_label
                expr = expr.replace(str(var.safe_label), str(var.curly_label))

        return expr

    def __repr__(self):
        expr = "{} ({})".format(self.expr, "solar units" if self.in_solar_units else "SI")
        if self.qualifier is not None:
            lhs = '{'+self.get_constrained_parameter().uniquetwig+'}'
            return "<ConstraintParameter: {} = {} => {}>".format(lhs, expr, self.result)
        else:
            return "<ConstraintParameter: {} => {}>".format(expr, self.result)

    def __str__(self):
        return "Constrains (qualifier): {}\nExpression in {} (value): {}\nCurrent Result (result): {}".format(self.qualifier, 'solar units' if self.in_solar_units else 'SI', self.expr, self.result)

    def __math__(self, other, symbol, mathfunc):
        #~ print "*** ConstraintParameter.__math__ other.type", type(other)
        if isinstance(other, ConstraintParameter):
            #~ print "*** ConstraintParameter.__math__", symbol, self.result, other.result
            return ConstraintParameter(self._bundle, "(%s) %s (%s)" % (self.expr, symbol, other.expr), default_unit=(getattr(self.result, mathfunc)(other.result).unit))
        elif isinstance(other, Parameter):
            return ConstraintParameter(self._bundle, "(%s) %s {%s}" % (self.expr, symbol, other.uniquetwig), default_unit=(getattr(self.result, mathfunc)(other.quantity).unit))
        elif isinstance(other, u.Quantity):
            #print "***", other, type(other), isinstance(other, ConstraintParameter)
            return ConstraintParameter(self._bundle, "(%s) %s %0.30f" % (self.expr, symbol, _value_for_constraint(other, self)), default_unit=(getattr(self.result, mathfunc)(other).unit))
        elif isinstance(other, float) or isinstance(other, int):
            if symbol in ['+', '-']:
                # assume same units as self (NOTE: NOT NECESSARILY SI) if addition or subtraction
                other = float(other)*self.default_unit
            else:
                # assume dimensionless
                other = float(other)*u.dimensionless_unscaled
            return ConstraintParameter(self._bundle, "(%s) %s %f" % (self.expr, symbol, _value_for_constraint(other, self)), default_unit=(getattr(self.result, mathfunc)(other).unit))
        elif isinstance(other, str):
            return ConstraintParameter(self._bundle, "(%s) %s %s" % (self.expr, symbol, other), default_unit=(getattr(self.result, mathfunc)(eval(other)).unit))
        elif _is_unit(other) and mathfunc=='__mul__':
            # here we'll fake the unit to become a quantity so that we still return a ConstraintParameter
            return self*(1*other)
        else:
            raise NotImplementedError("math using {} with type {} not supported".format(mathfunc, type(other)))

    def __rmath__(self, other, symbol, mathfunc):
        #~ print "*** ConstraintParameter.__rmath__ other.type", type(other)
        if isinstance(other, ConstraintParameter):
            #~ print "*** ConstraintParameter.__math__", symbol, self.result, other.result
            return ConstraintParameter(self._bundle, "(%s) %s (%s)" % (other.expr, symbol, self.expr), default_unit=(getattr(self.result, mathfunc)(other.result).unit))
        elif isinstance(other, Parameter):
            return ConstraintParameter(self._bundle, "{%s} %s (%s)" % (other.uniquetwig, symbol, self.expr), default_unit=(getattr(self.result, mathfunc)(other.quantity).unit))
        elif isinstance(other, u.Quantity):
            #~ print "*** rmath", other, type(other)
            return ConstraintParameter(self._bundle, "%0.30f %s (%s)" % (_value_for_constraint(other, self), symbol, self.expr), default_unit=(getattr(self.result, mathfunc)(other).unit))
        elif isinstance(other, float) or isinstance(other, int):
            if symbol in ['+', '-']:
                # assume same units as self if addition or subtraction
                other = float(other)*self.default_unit
            else:
                # assume dimensionless
                other = float(other)*u.dimensionless_unscaled
            return ConstraintParameter(self._bundle, "%f %s (%s)" % (_value_for_constraint(other, self), symbol, self.expr), default_unit=(getattr(self.result, mathfunc)(other).unit))
        elif isinstance(other, str):
            return ConstraintParameter(self._bundle, "%s %s (%s)" % (other, symbol, self.expr), default_unit=(getattr(self.result, mathfunc)(eval(other)).unit))
        elif _is_unit(other) and mathfunc=='__mul__':
            # here we'll fake the unit to become a quantity so that we still return a ConstraintParameter
            return self*(1*other)
        else:
            raise NotImplementedError("math using {} with type {} not supported".format(mathfunc, type(other)))

    def __add__(self, other):
        return self.__math__(other, '+', '__add__')

    def __radd__(self, other):
        return self.__rmath__(other, '+', '__radd__')

    def __sub__(self, other):
        return self.__math__(other, '-', '__sub__')

    def __rsub__(self, other):
        return self.__math__(other, '-', '__rsub__')

    def __mul__(self, other):
        return self.__math__(other, '*', '__mul__')

    def __rmul__(self, other):
        return self.__rmath__(other, '*', '__rmul__')

    def __div__(self, other):
        return self.__math__(other, '/', '__div__')

    def __rdiv__(self, other):
        return self.__rmath__(other, '/', '__rdiv__')

    def __pow__(self, other):
        return self.__math__(other, '**', '__pow__')

    @property
    def result(self):
        """
        Get the current value (as a quantity) of the result of the expression
        of this <phoebe.parameters.ConstraintParameter>.

        This is identical to <phoebe.parameters.ConstraintParameter.get_result>.

        Returns
        --------
        * (quantity): the current result of evaluating the constraint expression.
        """
        return self.get_result()

    def get_result(self, t=None, use_distribution=None,
                   suppress_error=True, distribution_uniqueids=None):
        """
        Get the current value (as a quantity) of the result of the expression
        of this <phoebe.parameters.ConstraintParameter>.

        This is identical to <phoebe.parameters.ConstraintParameter.result>.

        Arguments
        -----------
        * `t` (int or float, optional, default=None): time at which to compute the
            result of the constraint.
        * `use_distribution` (str or None or DistrubutionCollection, optional, default=None):
            label of the distribution collection to propagate through the constraints.
            In general, its easiest to pass `parameters` to <phoebe.frontend.bundle.get_distribution_collection>,
            <phoebe.frontend.bundle.plot_distribution_collection>, etc.
        * `suppress_error` (bool, optional, default=True):
        * `distribution_uniqueids` (list, optional, default=None): must be provided
            if `use_distribution` is a DistributionCollection.

        Returns
        --------
        * (quantity): the current result of evaluating the constraint expression.
        """
        # TODO: optimize this:
        # almost half the time is being spent on self.get_value() of which most is spent on var.update_user_label
        # second culprit is converting everything to si
        # third culprit is the dictionary comprehensions

        # in theory, it would be nice to prepare this list at the module import
        # level, but that causes an infinite loop in the imports, so we'll
        # do a re-import here.  If this causes significant lag, it may be worth
        # trying to resolve the infinite loop.
        from phoebe.constraints import builtin
        _constraint_builtin_funcs = [f for f in dir(builtin) if isinstance(getattr(builtin, f), types.FunctionType)]
        _constraint_math_funcs = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2', 'sqrt', 'log10']

        def eq_needs_builtin(eq, include_math=True):
            builtin_funcs =  _constraint_builtin_funcs + _constraint_math_funcs if include_math else _constraint_builtin_funcs
            for func in builtin_funcs:
                if "{}(".format(func) in eq:
                    #print "*** eq_needs_builtin", func
                    return True
            return False

        def get_values(vars, safe_label=True, string_safe_arrays=False, use_distribution=None, needs_builtin=False):
            # use np.float64 so that dividing by zero will result in a
            # np.inf
            def _single_value(quantity, string_safe_arrays=False):
                if isinstance(quantity, u.Quantity):
                    if self.in_solar_units:
                        v = np.float64(u.to_solar(quantity).value)
                    else:
                        v = np.float64(quantity.si.value)

                    if isinstance(v, np.ndarray) and string_safe_arrays:
                        v = v.tolist()
                    return v
                elif isinstance(quantity, distl.BaseDistlObject):
                    if self.in_solar_units:
                        v = quantity.to_solar(strip_units=True)
                    else:
                        v = quantity.to_si(strip_units=True)
                    return v
                elif isinstance(quantity, str):
                    return '\'{}\''.format(quantity)
                else:
                    return quantity

            def _value(var, string_safe_arrays=False, use_distribution=None, needs_builtin=False):
                param = var.get_parameter()

                if use_distribution and param != self.constrained_parameter:
                    # print("\n\n*** {}.get_result param={}".format(self.twig, param.twig))
                    dist = param.get_distribution(use_distribution, distribution_uniqueids=distribution_uniqueids, follow_constraints=True)

                    # print("*** dist={}".format(dist))
                    if dist is not None:
                        if needs_builtin:
                            return _single_value(dist)
                        else:
                            # will we need to force distribution_uniqueids to be included in the json?
                            return "distl_from_json('{}')".format(_single_value(dist).to_json(export_func_as_path=True, exclude=['label_latex', 'labels_latex']))

                if param != self.constrained_parameter:
                    return _single_value(var.get_quantity(t=t), string_safe_arrays)
                else:
                    return _single_value(var.get_quantity(), string_safe_arrays)

            return {var.safe_label if safe_label else var.user_label: _value(var, string_safe_arrays, use_distribution, needs_builtin) for var in vars}

        eq = self.get_value()

        if _use_sympy and not eq_needs_builtin(eq) and not use_distribution:
            values = get_values(self._vars+self._addl_vars, safe_label=True)
            values['I'] = 1 # CHEATING MAGIC
            # just to be safe, let's reinitialize the sympy vars
            for v in self._vars:
                #~ print "creating sympy var: ", v.safe_label
                sympy.var(str(v.safe_label), positive=True)

            # print "***", self._value, values
            eq = sympy.N(self._value, 30)
            # print "***", self._value, values, eq.subs(values), eq.subs(values).evalf(15)
            value = float(eq.subs(values).evalf(15))
        else:
            # order here matters - self.get_value() will update the user_labels
            # to be the current unique twigs
            #print "***", eq, values
            # if use_distribution:
            #     values = get_values(self._vars+self._addl_vars, safe_label=False, string_safe_arrays=True, use_distribution=use_distribution)
            #     print("***", values)
            #     return
            needs_builtin_or_math = eq_needs_builtin(eq)
            if needs_builtin_or_math or use_distribution:
                # the else (which works for np arrays) does not work for the built-in funcs
                # this means that we can't currently support the built-in funcs WITH arrays
                needs_builtin = eq_needs_builtin(eq, include_math=False)

                # cannot do from builtin import *
                for func in _constraint_builtin_funcs + _constraint_math_funcs:
                    # I should be shot for doing this...
                    # in order for eval to work, the builtin functions need
                    # to be imported at the top-level, but I don't really want
                    # to do from builtin import * (and even if I did, python
                    # yells at me for doing that), so instead we'll add them
                    # to the locals dictionary.
                    locals()[func] = getattr(builtin, func)

                # if eq.split('(')[0] in ['times_to_phases', 'phases_to_times']:
                    # these require passing the bundle
                    # values['b'] = self._bundle

                values = get_values([v for v in self._vars+self._addl_vars if v.user_label in eq], safe_label=False, string_safe_arrays=True, use_distribution=use_distribution, needs_builtin=needs_builtin)

                if needs_builtin and use_distribution:
                    # need to parse {} in eq and get values in correct order as args (including non {}, like 1)
                    # need to access callable func from eq
                    funcname = eq.split("(")[0]
                    argnames = eq.split("(")[1].split(")")[0].split(", ")
                    args = []
                    for argname in argnames:
                        if argname[0] == "{":
                            args.append(values.get(argname[1:-1]))
                        else:
                            args.append(float(argname) if "." in argname else int(argname))

                    hist_samples = None
                    vectorized = False
                    if 'pot' in funcname or 'fillout_factor' in funcname or 'requiv_L1' in funcname:
                        # these are particularly expensive, so we'll only use 1000 samples in the underlying histogram by default
                        hist_samples = 1000
                        vectorized = False
                    if funcname[:2] == 't0':
                        vectorized = True
                    value = distl.function(locals().get(funcname), args, vectorized=vectorized, hist_samples=hist_samples)
                else:
                    # print("\n\n\n*** eval eq={} values={}".format(eq, values))
                    value = eval(eq.format(**values))

                if value is None:
                    if suppress_error:
                        value = np.nan
                        logger.error("{} constraint returned None".format(self.twig))
                    else:
                        raise ValueError("constraint returned None")
                else:
                    if use_distribution is None:
                        try:
                            value = float(value)
                        except TypeError as err:
                            try:
                                value = np.asarray(value)
                            except:
                                if suppress_error:
                                    value = np.nan
                                    logger.error("{} constraint raised the following error: {}".format(self.twig, str(err)))
                                else:
                                    raise
                        except ValueError as err:
                            if suppress_error:
                                value = np.nan
                                logger.error("{} constraint raised the following error: {}".format(self.twig, str(err)))
                            else:
                                raise



            else:
                # the following works for np arrays

                # TODO: cannot leave this as it stupidly expensive... so constraints need to return addl_vars or similar
                # vars = [ConstraintVar(self._bundle, twig) for twig in self._bundle.filter(context=['component', 'system', 'dataset']).twigs]
                # values = get_values(vars, safe_label=True)

                values = get_values(self._vars+self._addl_vars, safe_label=True)


                # if any of the arrays are empty (except the one we're filling)
                # then we want to return an empty array as well (the math would fail)
                arrays_filled = True
                for var in self._vars:
                    var_value = var.get_value()
                    #print "***", self.twig, self.constrained_parameter.twig, var.user_label, var_value, isinstance(var_value, np.ndarray), var.unique_label != self.constrained_parameter.uniqueid
                    # if self.qualifier is None then this isn't attached to solve anything yet, so we don't need to worry about checking to see if the var is the constrained parameter
                    if isinstance(var_value, np.ndarray) and len(var_value)==0 and (var.unique_label != self.constrained_parameter.uniqueid or self.qualifier is None):
                        #print "*** found empty array", self.constrainted_parameter.twig, var.safe_label, var_value
                        arrays_filled = False
                        #break  # out of the for loop

                if arrays_filled:
                    #print "*** else else", self._value, values
                    #print "***", _use_sympy, self._value, value
                    value = eval(self._value, values)
                else:
                    #print "*** EMPTY ARRAY FROM CONSTRAINT"
                    value = np.array([])

        #~ return value

        # let's assume the math was correct to give SI and we want units stored in self.default_units

        if self.default_unit is not None:
            if isinstance(value, distl.BaseDistlObject):
                if value.unit is None:
                    if self.in_solar_units:
                        value.unit = distl._distl._physical_types_to_solar.get(self.default_unit.physical_type)
                    else:
                        value.unit = distl._distl._physical_types_to_si.get(self.default_unit.physical_type)

                # TODO: should we skip this when calling internally and ask for it directly in solar/si?
                value = value.to(self.default_unit)
            else:
                if self.in_solar_units:
                    convert_scale = u.to_solar(self.default_unit)
                else:
                    convert_scale = self.default_unit.to_system(u.si)[0].scale
                #value = float(value/convert_scale) * self.default_unit
                value = value/convert_scale * self.default_unit


        return value

    @send_if_client
    def flip_for(self, twig=None, expression=None, **kwargs):
        """
        Flip the constraint expression to solve for for any of the parameters
        in the expression.

        The filtering (with `twig` and `**kwargs`) must find a single match
        among the Parameters in the expression.  See
        <phoebe.parameters.ConstraintParameter.vars> and
        <phoebe.parameters.ConstraintParameter.get_parameter>.

        See also:
        * <phoebe.frontend.bundle.Bundle.flip_constraint>

        Arguments
        ----------
        * `twig` (string, optional): the twig of the Parameter to constraint (solve_for).
        * `expression` (string, optional): provide the new expression.  If not
            provided, the expression will be pulled from the constraint func
            if possible, or solved for analytically if sympy is installed.
        * `**kwargs`: tags to be used for filtering for the newly constrained
            Parameter.
        """

        _orig_expression = self.get_value()

        # try to get the parameter from the bundle
        kwargs['twig'] = twig
        newly_constrained_var = self._get_var(**kwargs)
        newly_constrained_param = self.get_parameter(**kwargs)

        check_kwargs = newly_constrained_param.get_meta(ignore=['uniqueid', 'uniquetwig', 'twig', 'context'])
        check_kwargs['context'] = 'constraint'
        if len(self._bundle.filter(**check_kwargs)) and not kwargs.get('force', False):
            raise ValueError("'{}' is already constrained".format(newly_constrained_param.twig))

        currently_constrained_var = self._get_var(qualifier=self.qualifier, component=self.component)
        currently_constrained_param = currently_constrained_var.get_parameter() # or self.constrained_parameter

        addl_vars = []

        # cannot be at the top, or will cause circular import
        from . import constraint
        if self.constraint_func is not None and hasattr(constraint, self.constraint_func):
            # then let's see if the method is capable of resolving for use
            # try:
            if True:
                # TODO: this is not nearly general enough, each method takes different arguments
                # and getting solve_for as newly_constrained_param.qualifier

                lhs, rhs, addl_vars, constraint_kwargs = getattr(constraint, self.constraint_func)(self._bundle, solve_for=newly_constrained_param, **self.constraint_kwargs)
            # except NotImplementedError:
            #     pass
            # else:
                # TODO: this needs to be smarter and match to self._get_var().user_label instead of the current uniquetwig

                expression = rhs._value # safe expression
                #~ print "*** flip by recalling method success!", expression

        # print "***", lhs._value, rhs._value

        if expression is not None:
            expression = expression

        elif _use_sympy:


            eq_safe = "({}) - {}".format(self._value, currently_constrained_var.safe_label)

            #~ print "*** solving {} for {}".format(eq_safe, newly_constrained_var.safe_label)

            expression = sympy.solve(eq_safe, newly_constrained_var.safe_label)[0]

            #~ print "*** solution: {}".format(expression)

        else:
            # TODO: ability for built-in constraints to flip themselves
            # we could access self.kind and re-call that with a new solve_for option?
            raise ValueError("must either have sympy installed or provide a new expression")

        self._qualifier = newly_constrained_param.qualifier
        self._component = newly_constrained_param.component
        self._kind = newly_constrained_param.kind

        # self._value, self._vars = self._parse_expr(rhs)
        # self.set_value(rhs)

        if len(addl_vars):
            # then the vars may have changed (esinw,ecosw, for example)
            vars_ = []
            var_safe_labels = []
            # technically addl_vars probably hasn't changed... but let's recompute to be safe
            # self._addl_vars = [ConstraintVar(self._bundle, v.twig) for v in addl_vars]

            for var in self._vars + self._addl_vars:
                var_safe_label = var.safe_label
                if var_safe_label in expression and var_safe_label not in var_safe_labels:
                    vars_.append(var)
                    var_safe_labels.append(var_safe_label)
            self._vars = vars_

            # and we'll reset the cached version of the parameters
            self._var_params = None
            self._addl_var_params = None

        self._value = str(expression)


        #self.set_value(str(expression))
        # reset the default_unit so that set_default_unit doesn't complain
        # about incompatible units
        self._default_unit = None
        self.set_default_unit(newly_constrained_param.default_unit)

        self._update_bookkeeping()

        if self._bundle is not None and not kwargs.get('from_flip_bundle_constraint', False):
            self._bundle._handle_fitparameters_selecttwigparams(return_changes=False)


class JobParameter(Parameter):
    """
    Parameter that tracks a submitted job (detached
    <phoebe.frontend.bundle.Bundle.run_compute>, for example)
    """
    def __init__(self, b, location, status_method, retrieve_method, server_status=None, **kwargs):
        """
        see <phoebe.parameters.Parameter.__init__>
        """
        _qualifier = kwargs.pop('qualifier', None)
        kwargs.setdefault('readonly', True)
        super(JobParameter, self).__init__(qualifier='detached_job', **kwargs)

        self._bundle = b
        self._server_status = server_status
        self._location = location
        self._status_method = status_method
        self._retrieve_method = retrieve_method
        self._value = 'unknown'
        #self._randstr = randstr

        # TODO: may need to be more clever once remote servers are supported
        self._script_fname = os.path.join(location, '_{}.py'.format(self.uniqueid))
        self._results_fname = os.path.join(location, '_{}.out'.format(self.uniqueid))
        self._err_fname = os.path.join(location, '_{}.err'.format(self.uniqueid))
        self._kill_fname = self._results_fname + '.kill'

        # TODO: add a description?

        self._dict_fields_other = ['description', 'value', 'server_status', 'location', 'status_method', 'retrieve_method', 'uniqueid', 'readonly', 'advanced', 'latexfmt']
        self._dict_fields = _meta_fields_all + self._dict_fields_other

    def __str__(self):
        """
        """
        # TODO: implement a nice(r) string representation
        return "qualifier: {}\nstatus: {}".format(self.qualifier, self.status)

    def get_value(self, **kwargs):
        """
        JobParameter doesn't really have a value, but for the sake of Parameter
        representations, we'll provide the current status.

        Also see:
            * <phoebe.parameters.JobParameter.status>
            * <phoebe.parameters.JobParameter.attach>
            * <phoebe.parameters.JobParameter.location>
            * <phoebe.parameters.JobParameter.server_status>
            * <phoebe.parameters.JobParameter.status_method>
            * <phoebe.parameters.JobParameter.retrieve_method>

        """
        return self.status

    def set_value(self, *args, **kwargs):
        """
        <phoebe.parameters.JobParameter> is read-only.  Calling set_value
        will raise an Error.

        Raises
        --------
        * NotImplementedError: because this never will be
        """

        raise NotImplementedError("JobParameter is a read-only parameter.  Call status or attach()")

    @property
    def server_status(self):
        """
        Access the status of the remote server, if applicable.

        Returns
        -----------
        * (str)
        """
        return self._server_status

    @property
    def location(self):
        """
        Access the location of the remote server, if applicable.

        Returns
        ----------
        * (str)
        """
        return self._location

    @property
    def status_method(self):
        """
        Access the method for determining the status of the Job.

        Returns
        ---------
        * (str)
        """
        return self._status_method

    @property
    def retrieve_method(self):
        """
        Access the method for retrieving the results from the Job, once completed.

        Returns
        -----------
        * (str)
        """
        return self._retrieve_method

    @property
    def status(self):
        """
        Access the status of the Job.  This is just a property shortcut to
        <phoebe.parameters.JobParameter.get_status>.

        Returns
        ---------
        * (str): the current status of the Job.

        Raises
        ------------
        * NotImplementedError: if status isn't implemented for the given <phoebe.parameters.JobParameter.status_method>
        """
        return self.get_status()

    def get_status(self):
        """
        Access the status of the Job.

        Returns
        ---------
        * (str): the current status of the Job.

        Raises
        ------------
        * ImportError: if the requests module is not installed - this is
            required to handle detached Jobs.
        * ValueError: if the status of the Job cannot be determined.
        * NotImplementedError: if status isn't implemented for the given
            <phoebe.parameters.JobParameter.status_method>.
        """
        if self._value == 'loaded':
            status = 'loaded'

        elif not _is_server and self._bundle is not None and self._server_status is not None:
            if not _can_requests:
                raise ImportError("requests module required for external jobs")

            raise NotImplementedError()
            # if self._value in ['complete']:
            #     # then we have no need to bother checking again
            #     status = self._value
            # else:
            #     url = self._server_status
            #     logger.info("checking job status on server from {}".format(url))
            #     # "{}/{}/parameters/{}".format(server, bundleid, self.uniqueid)
            #     r = requests.get(url, timeout=5)
            #     try:
            #         rjson = r.json()
            #     except ValueError:
            #         # TODO: better exception here - perhaps look for the status code from the response?
            #         status = self._value
            #     else:
            #         status = rjson['data']['attributes']['value']

        else:

            if self.status_method == 'exists':
                if self._value in ['error', 'killed', 'loaded']:
                    # then error was already detected and we've already done cleanup
                    status = self._value
                elif os.path.isfile(self._results_fname):
                    status = 'complete'
                elif os.path.isfile(self._kill_fname):
                    status = 'killed'
                elif os.path.isfile(self._results_fname+'.progress'):
                    status = 'progress'
                elif os.path.isfile(self._err_fname) and os.stat(self._err_fname).st_size > 0:
                    # some warnings from other packages can be set to stderr
                    # so we need to make sure the last line is actually from
                    # raising an error.
                    ferr = open(self._err_fname, 'r')
                    msg = ferr.readlines()[-1]
                    ferr.close()
                    if 'Error' in msg.split()[0]:
                        status = 'error'
                    else:
                        status = 'running'
                else:
                    status = 'running'
            else:
                raise NotImplementedError

        # here we'll set the value to be the latest CHECKED status for the sake
        # of exporting to JSON and updating the status for clients.  get_value
        # will still call status so that it will return the CURRENT value.
        self._value = status
        return status

    def _retrieve_results(self):
        """
        [NOT IMPLEMENTED]
        """
        # now the file with the model should be retrievable from self._result_fname
        if 'progress' in self._value:
            fname = self._results_fname + '.progress'
        else:
            fname = self._results_fname

        try:
            ret_ps = ParameterSet.open(fname)
        except Exception as err:
            if 'progress' in self._value:
                return None
            else:
                raise
        else:
            return ret_ps

    def _cleanup(self):
        try:
            os.remove(self._script_fname)
        except: pass
        try:
            os.remove(self._results_fname)
        except: pass
        try:
            os.remove(self._err_fname)
        except: pass
        try:
            os.remove(self._results_fname+".progress")
        except: pass
        try:
            os.remove(self._kill_fname)
        except: pass

    def attach(self, wait=True, sleep=5, cleanup=True, return_changes=False):
        """
        Attach the results from a <phoebe.parameters.JobParameter> to the
        <phoebe.frontend.bundle.Bundle>.  If the status is not yet reported as
        complete, this will loop every `sleep` seconds until it is.

        Arguments
        ---------
        * `wait` (bool, optional, default=True): whether to wait until the job
            is complete.
        * `sleep` (int, optional, default=5): number of seconds to sleep between
            status checks.  See <phoebe.parameters.JobParameter.get_status>.
            Only applicable if `wait` is True.
        * `cleanup` (bool, optional, default=True): whether to delete any
            temporary files once the results are loaded.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        ---------
        * ParameterSet of newly attached parameters (if attached or already
            loaded) or this Parameter with an updated status if `wait` is False
            and the Job is not completed.

        Raises
        -----------
        * ValueError: if not attached to a <phoebe.frontend.bundle.Bundle> object.
        """
        if not self._bundle:
            raise ValueError("can only attach a job if attached to a bundle")

        status = self.get_status()
        if not wait and status not in ['complete', 'error', 'progress']:
            if status in ['loaded']:
                logger.info("job already loaded")
                if self.context == 'model':
                    return self._bundle.get_model(self.model)
                elif self.context == 'solution':
                    return self._bundle.get_solution(self.solution)
                else:
                    raise NotImplementedError("attaching for context='{}' not implemented".format(self.context))
            else:
                logger.info("current status: {}, check again or use wait=True".format(status))
                return self

        if wait:
            while self.get_status() not in ['complete', 'loaded', 'error']:
                # TODO: any way we can not make 2 calls to self.status here?
                logger.info("current status: {}, trying again in {}s".format(self.get_status(), sleep))
                time.sleep(sleep)

        if self._server_status is not None and not _is_server:
            if not _can_requests:
                raise ImportError("requests module required for external jobs")

            raise NotImplementedError()
            # # then we are no longer attached as a client to this bundle on
            # # the server, so we need to just pull the results manually
            # url = self._server_status
            # logger.info("pulling job results from server from {}".format(url))
            # # "{}/{}/parameters/{}".format(server, bundleid, self.uniqueid)
            # r = requests.get(url, timeout=5)
            # rjson = r.json()
            #
            # # status should already be complete because of while loop above,
            # # but could always check the following:
            # # rjson['value']['attributes']['value'] == 'complete'
            #
            # # TODO: server needs to sideload results once complete
            # newparams = rjson['included']
            # self._bundle._attach_param_from_server(newparams)

        elif self.status == 'error':
            lines = ferr.readlines()
            ferr.close()

            if cleanup:
                self._cleanup()

            self._value = 'error'

            print("ERROR: full error message: {}".format(lines))
            logger.error("full error message: {}".format(lines))
            raise RuntimeError("job failed with error: {}".format(lines[-1]))
        else:
            logger.info("current status: {}, pulling job results".format(self.status))
            ret_ps = self._retrieve_results()

            if ret_ps is None and 'progress' in self._value:
                # then we just want to update the progress value from the progress-file
                # but don't have anything to actually load
                f = open(self._results_fname + '.progress', 'r')
                progress_str = f.readlines()[0]
                f.close()

                try:
                    progress = np.round(float(progress_str.strip()), 2)
                except:
                    return ParameterSet([])
                else:
                    self._value = 'progress:{}%'.format(progress)
                    return ParameterSet([self])


            # now we need to attach ret_ps to self._bundle
            # TODO: is creating metawargs here necessary?  Shouldn't the params already be tagged?
            if self.context == 'model':
                metawargs = {'compute': str(ret_ps.compute), 'model': str(ret_ps.model), 'context': 'model'}
            elif self.context == 'solution':
                metawargs = {'solver': str(ret_ps.solver), 'solution': str(ret_ps.solution), 'context': 'solution'}
            else:
                raise NotImplementedError("attaching for context='{}' not implemented".format(self.context))

            if 'progress' in self._value:
                if ret_ps.get_value(qualifier='progress', default=0, **_skip_filter_checks) == self._bundle.get_value(qualifier='progress', default=100, check_visible=False, check_advanced=False, **metawargs):
                    # then we have nothing new to load, so let's not bother attaching and overwriting with the exact same thing
                    return ParameterSet([])
                elif 'progress' in ret_ps.qualifiers:
                    self._value = 'progress:{}%'.format(np.round(ret_ps.get_value(qualifier='progress'), 2))


            ret_changes = self._bundle._attach_params(ret_ps, overwrite=True, return_changes=return_changes, **metawargs)
            if return_changes:
                ret_changes += [self]

            if cleanup and self._value in ['complete', 'loaded', 'error', 'killed']:
                self._cleanup()

            if 'progress' not in self._value:
                self._value = 'loaded'

            if self.context == 'model':
                # TODO: check logic for do_create_fig_params
                ret_changes += self._bundle._run_compute_changes(ret_ps, return_changes=return_changes, do_create_fig_params=True)

            elif self.context == 'solution':
                ret_changes += self._bundle._run_solver_changes(ret_ps, return_changes=return_changes)

            else:
                raise NotImplementedError("attaching for context='{}' not implemented".format(self.context))

            if return_changes:
                return ret_ps + ret_changes

            return ret_ps

    def kill(self, cleanup=True, return_changes=False):
        """
        Send a termination signal to the external thread running a
        <phoebe.parameters.JobParameter>

        Arguments
        ---------
        * `cleanup` (bool, optional, default=True): whether to wait for the
            thread to terminate and then call <phoebe.parameters.JobParameter.attach>
            with `cleanup=True` and `wait=True`.
        * `return_changes` (bool, optional, default=False): whether to include
            changed/removed parameters in the returned ParameterSet.

        Returns
        ---------


        """
        f = open(self._kill_fname, 'w')
        f.write('kill')
        f.close()
        if cleanup:
            return self.attach(wait=True, cleanup=True)
