import subprocess
import os
import re
import json
from datetime import datetime

# PHOEBE
# ParameterSet, Parameter, FloatParameter, send_if_client, etc
from phoebe.parameters import *
from phoebe.parameters import hierarchy as _hierarchy
from phoebe.parameters import system as _system
from phoebe.parameters import setting as _setting
from phoebe.parameters import dataset as _dataset
from phoebe.parameters import compute as _compute
from phoebe.parameters import constraint as _constraint
from phoebe.backend import backends
from phoebe.frontend import io
import libphoebe

from phoebe import u

import logging
logger = logging.getLogger("BUNDLE")
logger.addHandler(logging.NullHandler())


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

    The :class:`Bundle` is the main object in PHOEBE 2 which is used to store
    and filter all available system parameters as well as handling attaching
    datasets, running models, and accessing synthetic data.

    The Bundle is simply a glorified
    :class:`phoebe.parameters.parameters.ParameterSet`. In fact, filtering on
    a Bundle gives you a ParameterSet (and filtering on a ParameterSet gives
    you another ParameterSet).  The only difference is that most "actions" are
    only available at the Bundle level (as they need to access /all/
    parameters).

    Make sure to also see the documentation and methods for  *
    :class:`phoebe.parameters.parameters.ParameterSet` *
    :class:`phoebe.parameters.parameters.Parameter` *
    :class:`phoebe.parameters.parameters.FloatParameter` *
    :class:`phoebe.parameters.parameters.ArrayParameter`

    To initialize a new bundle, see:
        * :meth:`open`
        * :meth:`from_legacy`
        * :meth:`default_binary`

    To filter parameters and set values, see:
        * :meth:`phoebe.parameters.parameters.ParameterSet.filter`
        * :meth:`phoebe.parameters.parameters.ParameterSet.get_value`
        * :meth:`phoebe.parameters.parameters.ParameterSet.set_value`

    To deal with datasets, see:
        * :meth:`add_dataset`
        * :meth:`get_dataset`
        * :meth:`remove_dataset`
        * :meth:`enable_dataset`
        * :meth:`disable_dataset`

    To compute forward models, see:
        * :meth:`add_compute`
        * :meth:`get_compute`
        * :meth:`run_compute`
        * :meth:`get_model`

    To plot observations or synthetic datasets, see:
        * :meth:`phoebe.parameters.parameters.ParameterSet.plot`

    """

    def __init__(self, params=None):
        """Initialize a new Bundle.

        Initializing a new bundle without a constructor is possible, but not
        advised.  It is suggested that you use one of the constructors below.

        Available constructors:
            * :meth:`open`
            * :meth:`from_legacy`
            * :meth:`default_binary`

        :param list parameters: list of
            :class:`phoebe.parameters.parameters.Parameter` to create the
            Bundle (optional)
        :return: instantiated :class:`Bundle` object
        """
        # for some reason I do not understand at all, defaulting params=[] will
        # fail for successive inits.  So instead we'll default to None and then
        # switch to an empty array here.
        if params is None:
            params = []

        self._params = []
        super(Bundle, self).__init__(params=params)

        # since this is a subclass of PS, some things try to access the bundle
        # by self._bundle, in this case we just need to fake that to refer to
        # self
        self._bundle = self

        # set to be not a client by default
        self._is_client = False
        self._last_client_update = None

        if not len(params):
            # add position (only 1 allowed and required)
            self._attach_params(_system.system(), context='system')

            # add default settings (only 1 allowed and required)
            self._attach_params(_setting.settings(), context='setting')

            # set a blank hierarchy to start
            self.set_hierarchy(_hierarchy.blank)

        # if loading something with constraints, we need to update the
        # bookkeeping so the parameters are aware of how they're constrained
        for constraint in self.filter(context='constraint').to_list():
            constraint._update_bookkeeping()

        # TODO: is this the correct place to do this? is blank hierarchy still
        # ok for loading from file??

    @classmethod
    def open(cls, filename):
        """Open a new bundle.

        Open a bundle from a JSON-formatted PHOEBE 2.0 (beta) file.
        This is a constructor so should be called as:


        >>> b = Bundle.open('test.phoebe')


        :parameter str filename: relative or full path to the file
        :return: instantiated :class:`Bundle` object
        """
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        return cls(data)

    @classmethod
    def from_server(cls, bundleid, server='http://localhost:5555',
                    as_client=True):
        """Load a new bundle from a server.

        [NOT IMPLEMENTED]

        Load a bundle from a phoebe server.  This is a constructor so should be
        called as:

        >>> b = Bundle.from_server('asdf', as_client=False)

        :parameter str bundleid: the identifier given to the bundle by the
            server
        :parameter str server: the host (and port) of the server
        :parameter bool as_client: whether to attach in client mode
            (default: True)
        """
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

        [NOTIMPLEMENTED]

        Load a bundle from the online catalog.  This is a constructor
        so should be called as:

        >>> b = Bundle.from_catalog(identifier)

        :parameter str identifier: identifier of the object in the catalog
        :return: instantiated :class:`Bundle` object
        :raises NotImplementedError: because this isn't implemented yet
        """
        raise NotImplementedError
        # TODO: pull from online catalog and pass arguments needed to cls
        # (__init__) or cls.open (defined in PS.open)

        return cls()

    @classmethod
    def from_legacy(cls, filename):
        """Load a bundle from a PHOEBE 1.0 Legacy file.

        This is a constructor so should be called as:

        >>> b = Bundle.from_legacy('myfile.phoebe')

        :parameter str filename: relative or full path to the file
        :return: instantiated :class:`Bundle` object
        """
        return io.load_legacy(filename)

    @classmethod
    def default_binary(cls, overcontact=False):
        """Load a bundle with a default binary as the system.

        primary - secondary

        This is a constructor, so should be called as:

        >>> b = Bundle.default_binary()

        :return: instantiated :class:`Bundle` object
        """
        b = cls()
        b.add_component('star', component='primary')
        b.add_component('star', component='secondary')
        b.add_component('orbit', component='binary')
        if overcontact:
            b.add_component('envelope', component='common_envelope')
            b.set_hierarchy(_hierarchy.binaryorbit,
                            b['binary'],
                            b['primary'],
                            b['secondary'],
                            b['common_envelope'])
        else:
            b.set_hierarchy(_hierarchy.binaryorbit,
                            b['binary'],
                            b['primary'],
                            b['secondary'])

        return b

    @classmethod
    def default_triple(cls, inner_as_primary=True, inner_as_overcontact=False):
        """Load a bundle with a default triple system.

        Set inner_as_primary based on what hierarchical configuration you want.

        inner_as_primary = True:

        starA - starB -- starC

        inner_as_primary = False:

        starC -- starA - starB

        This is a constructor, so should be called as:

        >>> b = Bundle.default_triple_primary()

        :parameter bool inner_as_primary: whether the inner-binary should be
            the primary component of the outer-orbit
        :return: instantiated :class:`Bundle` object
        """
        b = cls()
        b.add_component('star', component='starA')
        b.add_component('star', component='starB')
        b.add_component('star', component='starC')
        b.add_component('orbit', component='inner', period=1)
        b.add_component('orbit', component='outer', period=10)

        if inner_as_overcontact:
            b.add_component('envelope', component='common_envelope')
            inner = _hierarchy.binaryorbit(b['inner'],
                                           b['starA'],
                                           b['starB'],
                                           b['common_envelope'])
        else:
            inner = _hierarchy.binaryorbit(b['inner'], b['starA'], b['starB'])

        if inner_as_primary:
            hierstring = _hierarchy.binaryorbit(b['outer'], inner, b['starC'])
        else:
            hierstring = _hierarchy.binaryorbit(b['outer'], b['starC'], inner)
        b.set_hierarchy(hierstring)

        b.add_constraint(constraint.keplers_third_law_hierarchical,
                         'outer', 'inner')

        # TODO: does this constraint need to be rebuilt when things change?
        # (ie in set_hierarchy)

        return b

    def save(self, filename, clear_history=True, incl_uniqueid=False):
        """Save the bundle to a JSON-formatted ASCII file.

        :parameter str filename: relative or full path to the file
        :parameter bool clear_history: whether to clear history log
            items before saving (default: True)
        :parameter bool incl_uniqueid: whether to including uniqueids in the
            file (only needed if its necessary to maintain the uniqueids when
            reloading)
        :return: the filename
        """
        if clear_history:
            # TODO: let's not actually clear history,
            # but rather skip the context when saving
            self.remove_history()

        # TODO: add option for clear_models, clear_feedback

        return super(Bundle, self).save(filename, incl_uniqueid=incl_uniqueid)

    def export_legacy(self, filename):
        """
        TODO: add docs
        """

        return io.pass_to_legacy(self, filename)


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
        params = len(getattr(self.filter(**kwargs), '{}s'.format(context)))

        return "{}{:02d}".format(base, params+1)

    def change_component(self, old_component, new_component):
        """
        Change the label of a component attached to the Bundle

        :parameter str old_component: the current name of the component
            (must exist)
        :parameter str new_component: the desired new name of the component
            (must not exist)
        :return: None
        :raises ValueError: if the new_component is forbidden
        """
        # TODO: raise error if old_component not found?

        self._check_label(new_component)
        for param in self.filter(component=old_component).to_list():
            param._component = new_component
        for param in self.filter(context='constraint').to_list():
            for k, v in param.constraint_kwargs:
                if v == old_component:
                    param._constraint_kwargs[k] = new_component

    def get_setting(self, twig=None, **kwargs):
        """
        Filter in the 'setting' context

        :parameter str twig: the twig used for filtering
        :parameter **kwargs: any other tags to do the filter (except tag or
            context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'setting'
        return self.filter_or_get(**kwargs)

    def _add_history(self, redo_func, redo_kwargs, undo_func, undo_kwargs,
                     **kwargs):
        """
        Add a new log (undo/redoable) to this history context

        :parameter str redo_func: function to redo the action, must be a
            method of :class:`Bundle`
        :parameter dict redo_kwargs: kwargs to pass to the redo_func.  Each
            item must be serializable (float or str, not objects)
        :parameter str undo_func: function to undo the action, must be a
            method of :class:`Bundle`
        :parameter dict undo_kwargs: kwargs to pass to the undo_func.  Each
            item must be serializable (float or str, not objects)
        :parameter str history: label of the history parameter
        :raises ValueError: if the label for this history item is forbidden or
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
        Property as a shortcut to :meth:`get_history`

        You can toggle whether history is recorded using
            * :meth:`enable_history`
            * :meth:`disable_history`
        """

        return self.get_history()

    def get_history(self, i=None):
        """
        Get a history item by index.

        You can toggle whether history is recorded using
            * :meth:`enable_history`
            * :meth:`disable_history`

        :parameter int i: integer for indexing (can be positive or
            negative).  If i is None or not provided, the entire list
            of history items will be returned
        :return: :class:`phoebe.parameters.parameters.Parameter` if i is
            an int, or :class:`phoebe.parameters.parameters.ParameterSet` if i
            is not provided
        :raises ValueError: if no history items have been recorded.
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

        You can toggle whether history is recorded using
            * :meth:`enable_history`
            * :meth:`disable_history`


        :parameter int i: integer for indexing (can be positive or
            negative).  If i is None or not provided, the entire list
            of history items will be removed
        :raises ValueError: if no history items have been recorded.
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
        Property as a shortcut to b.get_setting('log_history).get_value().

        You can toggle whether history is recorded using
            * :meth:`enable_history`
            * :func:`disable_history`

        :return: whether logging of history items (undo/redo) is enabled.
        :rtype: bool
        """
        return self.get_setting('log_history').get_value()\
            if len(self.get_setting())\
            else False

    def enable_history(self):
        """
        Enable logging history items (undo/redo).

        You can check wither history is enabled using :meth:`history_enabled`.

        Shortcut to b.get_setting('log_history').set_value(True)
        """
        self.get_setting('log_history').set_value(True)

    def disable_history(self):
        """
        Disable logging history items (undo/redo)

        You can check wither history is enabled using :meth:`history_enabled`.

        Shortcut to b.get_setting('log_history').set_value(False)
        """
        self.get_setting('log_history').set_value(False)

    def undo(self, i=-1):
        """
        Undo an item in the history logs

        :parameter int i: integer for indexing (can be positive or
            negative).  Defaults to -1 if not provided (the latest
            recorded history item)
        :raises ValueError: if no history items have been recorded
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

        :parameter int i: integer for indexing (can be positive or
            negative).  Defaults to -1 if not provided (the latest
            recorded history item)
        :raises ValueError: if no history items have been recorded
        """
        _history_enabled = self.history_enabled
        param = self.get_history(i)
        self.disable_history()
        param.redo()
        self.remove_parameter(uniqueid=param.uniqueid)
        if _history_enabled:
            self.enable_history()

    def set_hierarchy(self, *args, **kwargs):
        """
        Set the hierarchy of the system.

        See tutorial on building a system.

        TODO: provide documentation
        args can be
        - string representation (preferably passed through hierarchy already)
        - func and strings/PSs/params to pass to function
        """

        _old_param = self.get_hierarchy()

        if len(args) == 1 and isinstance(args[0], str):
            repr_ = args[0]
            method = None

        elif len(args) == 0:
            if 'value' in kwargs.keys():
                repr_ = kwargs['value']
                method = None
            else:
                # TODO: raise warning?
                # return self.remove_parameter(context='hierarchy')
                repr_ = self.get_hierarchy().get_value()
                method = None

        else:
            func = _get_add_func(hierarchy, args[0])
            func_args = args[1:]

            repr_ = func(*func_args)

            method = func.func_name

        hier_param = HierarchyParameter(value=repr_,
                                        description='Hierarchy representation')

        self.remove_parameters_all(qualifier='hierarchy', context='system')

        metawargs = {'context': 'system'}
        self._attach_params([hier_param], **metawargs)

        # Handle choice parameters that need components as choices
        starrefs = hier_param.get_stars()  # TODO: consider for overcontacts
        for param in self.filter(qualifier='pbscale',
                                 context='dataset').to_list():
            param._choices = ['pblum'] + starrefs
            if param.value == '':
                # then this was the default from the parameter itself, so we
                # want to set it to be pblum if its the "primary" star, and
                # otherwise point to the primary star
                if param.component == starrefs[0]:
                    param.set_value('pblum')
                else:
                    param.set_value(starrefs[0])

        # Handle inter-PS constraints
        for component in self.hierarchy.get_stars():
            logger.info('re-creating mass constraint for {}'.format(component))
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

            logger.info('re-creating comp_sma constraint for {}'.format(component))
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

            logger.info('re-creating rotation_period constraint for {}'.format(component))
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

            logger.info('re-creating incl_aligned constraint for {}'.format(component))
            # TODO: will this cause problems if the constraint has been flipped?
            # TODO: what if the user disabled/removed this constraint?
            if len(self.filter(context='constraint',
                               constraint_func='incl_aligned',
                            component=component)):
                constraint_param = self.get_constraint(constraint_func='incl_aligned',
                                                       component=component)
                self.remove_constraint(constraint_func='incl_aligned',
                                       component=component)
                self.add_constraint(constraint.incl_aligned, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.incl_aligned, component,
                                    constraint=self._default_label('incl_aligned', context='constraint'))

            logger.info('re-creating potential constraint for {}'.format(component))
            # TODO: will this cause problems if the constraint has been flipped?
            if len(self.filter(context='constraint',
                               constraint_func='potential',
                               component=component)):
                constraint_param = self.get_constraint(constraint_func='potential',
                                                       component=component)
                self.remove_constraint(constraint_func='potential',
                                       component=component)
                self.add_constraint(constraint.potential, component,
                                    solve_for=constraint_param.constrained_parameter.uniquetwig,
                                    constraint=constraint_param.constraint)
            else:
                self.add_constraint(constraint.potential, component,
                                    constraint=self._default_label('potential', context='constraint'))


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

        :parameter str twig: twig to use for filtering
        :parameter **kwargs: any other tags to do the filter
            (except twig or context)

        :return: :class:`phoebe.parameters.parameters.Parameter` or
            :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'system'
        return self.filter(**kwargs)

    @property
    def hierarchy(self):
        """
        Property shortcut to :meth:`get_hierarchy`

        :return: the hierarcy :class:`phoebe.parameters.parameters.Parameter`
            or None (if no hierarchy exists)
        """
        return self.get_hierarchy()

    def get_hierarchy(self):
        """
        Get the hierarchy parameter

        :return: the hierarcy :class:`phoebe.parameters.parameters.Parameter`
            or None (if no hierarchy exists)
        """
        try:
            return self.get_parameter(qualifier='hierarchy', context='system')
        except ValueError:
            return None

    def run_checks(self):
        """
        Check to see whether the system is expected to be computable.

        This is called by default for each set_value but will only raise a
        logger warning if fails.  This is also called immediately when calling
        :meth:`run_compute`.

        :return: True if passed, False if failed and a message
        """

        hier = self.hierarchy
        for component in hier.get_meshables():
            kind = hier.get_kind_of(component)
            comp_ps = self.get_component(component)
            parent_ps = self.get_component(hier.get_parent_of(component))
            if kind in ['star']:
                # MUST NOT be overflowing at PERIASTRON (1-ecc)
                # TODO: implement this check based of fillout factor or crit_pots constrained parameter?
                # TODO: only do this if distortion_method == 'roche'
                pot = comp_ps.get_value('pot')
                q = parent_ps.get_value('q')
                # TODO: do we need to invert q for secondary or not (pot is defined as if this was primary)
                if hier.get_primary_or_secondary(component) == 'secondary':
                    q = 1./q
                F = comp_ps.get_value('syncpar')
                d = 1 - parent_ps.get_value('ecc')

                # TODO: this needs to be generalized once other potentials are supported
                critical_pots = libphoebe.roche_critical_potential(q, F, d)

                if pot < critical_pots[0] or\
                        pot < critical_pots[1] or\
                        pot < critical_pots[2]:
                    return False,\
                        '{} is overflowing at periastron ({:.02f}, {:.02f}, {:.02f})'.format(component, *critical_pots)
            elif kind in ['envelope']:
                # MUST be overflowing at APASTRON (1+ecc)
                # TODO: implement this check based of fillout factor or crit_pots constrained parameter
                # TODO: only do this if distortion_method == 'roche' (which probably will be required for envelope?)
                pot = comp_ps.get_value('pot')
                q = parent_ps.get_value('q')
                # NOTE: pot for envelope will always be as if primary, so no need to invert
                F = 1.0
                # NOTE: syncpar is fixed at 1.0 for envelopes

                # TODO: this is technically cheating since our pot is defined at periastron.
                # We'll either need to transform the pot (using volume conservation??) or
                # force OCs to be in circular orbits, in which case this test can be done at
                # periastron as well
                d = 1 + parent_ps.get_value('ecc')
                critical_pots = libphoebe.roche_critical_potential(q, F, d)

                if pot > critical_pots[0]:
                    return False,\
                        '{} is not overflowing L1 at apastron'.format(component)

                # BUT MUST NOT be overflowing L2 or L3 at periastron
                d = 1 - parent_ps.get_value('ecc')
                critical_pots = libphoebe.roche_critical_potential(q, F, d)

                if pot < critical_pots[1] or pot < critical_pots[2]:
                    return False,\
                        '{} is overflowing L2 or L3 at periastron'.format(component)

            else:
                raise NotImplementedError("checks not implemented for type '{}'".format(kind))

        # TODO: add other checks
        # - make sure all ETV components are legal
        # - check for conflict between dynamics_method and mesh_method (?)

        # we've survived all tests
        return True, ''

    def add_component(self, method, **kwargs):
        """
        Add a new component (star or orbit) to the system.  If not provided,
        'component' (the name of the new star or orbit) will be created for
        you and can be accessed by the 'component' attribute of the returned
        ParameterSet.

        >>> b.add_component(component.star)

        or

        >>> b.add_component('orbit', period=2.5)

        Available methods include:
            * :func:`phoebe.parameters.component.star`
            * :func:`phoebe.parameters.component.orbit`

        :parameter method: function to call that returns a
            ParameterSet or list of parameters.  This must either be
            a callable function that accepts nothing but default
            values, or the name of a function (as a string) that can
            be found in the :mod:`phoebe.parameters.component` module
            (ie. 'star', 'orbit')
        :type method: str or callable
        :parameter str component: (optional) name of the newly-created
            component
        :parameter **kwargs: default values for any of the newly-created
            parameters
        :return: :class:`phoebe.parameters.parameters.ParameterSet` of
            all parameters that have been added
        :raises NotImplementedError: if required constraint is not implemented
        """

        func = _get_add_func(component, method)

        kwargs.setdefault('component',
                          self._default_label(func.func_name,
                                              **{'context': 'component',
                                                 'method': func.func_name}))

        self._check_label(kwargs['component'])

        params, constraints = func(**kwargs)

        metawargs = {'context': 'component',
                     'component': kwargs['component'],
                     'method': func.func_name}

        self._attach_params(params, **metawargs)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.func_name
        self._add_history(redo_func='add_component',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_component',
                          undo_kwargs={'component': kwargs['component']})

        for constraint in constraints:
            self.add_constraint(*constraint)

        return params

    def get_component(self, component=None, **kwargs):
        """
        Filter in the 'component' context

        :parameter str component: name of the component (optional)
        :parameter **kwargs: any other tags to do the filter
            (except component or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if component is not None:
            kwargs['component'] = component
        kwargs['context'] = 'component'
        return self.filter(**kwargs)

    def remove_component(self, component=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        Remove a 'component' from the bundle

        :raises NotImplementedError: because this isn't implemented yet
        """
        # TODO: don't forget to add_history
        # TODO: make sure also removes and handles the percomponent parameters correctly (ie maxpoints@phoebe@compute)
        raise NotImplementedError

    def add_orbit(self, component=None, **kwargs):
        """
        Shortcut to :meth:`add_component` but with method='orbit'
        """
        kwargs.setdefault('component', component)
        return self.add_component('orbit', **kwargs)

    def get_orbit(self, component=None, **kwargs):
        """
        Shortcut to :meth:`get_component` but with method='star'
        """
        kwargs.setdefault('method', 'orbit')
        return self.get_component(component, **kwargs)

    def remove_orbit(self, component=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        Shortcut to :meth:`remove_component` but with method='star'
        """
        kwargs.setdefault('method', 'orbit')
        return self.remove_component(component, **kwargs)

    def add_star(self, component=None, **kwargs):
        """
        Shortcut to :meth:`add_component` but with method='star'
        """
        kwargs.setdefault('component', component)
        return self.add_component('star', **kwargs)

    def get_star(self, component=None, **kwargs):
        """
        Shortcut to :meth:`get_component` but with method='star'
        """
        kwargs.setdefault('method', 'star')
        return self.get_component(component, **kwargs)

    def remove_star(self, component=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        Shortcut to :meth:`remove_component` but with method='star'
        """
        kwargs.setdefault('method', 'star')
        return self.remove_component(component, **kwargs)

    def add_envelope(self, component=None, **kwargs):
        """
        [NOT SUPPORTED]

        Shortcut to :meth:`add_component` but with method='envelope'
        """
        kwargs.setdefault('component', component)
        return self.add_component('envelope', **kwargs)

    def get_envelope(self, component=None, **kwargs):
        """
        [NOT SUPPORTED]

        Shortcut to :meth:`get_component` but with method='envelope'
        """
        kwargs.setdefault('method', 'envelope')
        return self.get_component(component, **kwargs)

    def remove_envelope(self, component=None, **kwargs):
        """
        [NOT SUPPORTED]
        [NOT IMPLEMENTED]

        Shortcut to :meth:`remove_component` but with method='envelope'
        """
        kwargs.setdefault('method', 'envelope')
        return self.remove_component(component, **kwargs)

    def get_ephemeris(self, component=None, t0='t0_supconj', **kwargs):
        """
        Get the ephemeris of a component (star or orbit)

        :parameter str component: name of the component.  If not given,
            component will default to the top-most level of the current
            hierarchy
        :parameter **kwargs: any value passed through kwargs will override the
            ephemeris retrieved by component (ie period, t0, phshift, dpdt).
            Note: be careful about units - input values will not be converted.
        :return: dictionary containing period, t0 (t0_supconj if orbit),
            phshift, dpdt (as applicable)
        :rtype: dict
        """

        if component is None:
            component = self.hierarchy.get_top()

        ret = {}

        ps = self.filter(component=component, context='component')

        if ps.method in ['orbit']:
            ret['period'] = ps.get_value(qualifier='period', unit=u.d)
            ret['t0'] = ps.get_value(qualifier=t0, unit=u.d)
            ret['phshift'] = ps.get_value(qualifier='phshift')
            ret['dpdt'] = ps.get_value(qualifier='dpdt', unit=u.d/u.d)
        elif ps.method in ['star']:
            ret['period'] = ps.get_value(qualifier='period', unit=u.d)
        else:
            raise NotImplementedError

        for k,v in kwargs.items():
            ret[k] = v

        return ret

    def to_phase(self, time, component=None, t0='t0_supconj', **kwargs):
        """
        Get the phase(s) of a time(s) for a given ephemeris

        :parameter time: time to convert to phases (should be in same system
            as t0s)
        :type time: float, list, or array
        :parameter str component: component for which to get the ephemeris.
            If not given, component will default to the top-most level of the
            current hierarchy
        :parameter **kwargs: any value passed through kwargs will override the
            ephemeris retrieved by component (ie period, t0, phshift, dpdt).
            Note: be careful about units - input values will not be converted.
        :return: phase (float) or phases (array)
        """

        ephem = self.get_ephemeris(component=component, t0=t0, **kwargs)

        if isinstance(time, list):
            time = np.array(time)
        elif isinstance(time, Parameter):
            time = time.get_value(u.d)
        elif isinstance(time, str):
            time = self.get_value(time, u.d)

        t0 = ephem.get('t0', 0.0)
        phshift = ephem.get('phshift', 0.0)
        period = ephem.get('period', 1.0)
        dpdt = ephem.get('dpdt', 0.0)

        return ((time - t0) % (period + dpdt * (time - t0))) / \
            (period + dpdt * (time - t0)) + phshift

    def to_time(self, phase, component=None, t0='t0_supconj', **kwargs):
        """
        Get the time(s) of a phase(s) for a given ephemeris

        :parameter phase: phase to convert to times (should be in
            same system as t0s)
        :type phase: float, list, or array
    `   :parameter str component: component for which to get the ephemeris.
            If not given, component will default to the top-most level of the
            current hierarchy
        :parameter **kwargs: any value passed through kwargs will override the
            ephemeris retrieved by component (ie period, t0, phshift, dpdt).
            Note: be careful about units - input values will not be converted.
        :return: time (float) or times (array)
        """

        ephem = self.get_ephemeris(component=component, t0=t0, **kwargs)

        if isinstance(phase, list):
            phase = np.array(phase)

        t0 = ephem.get('t0', 0.0)
        phshift = ephem.get('phshift', 0.0)
        period = ephem.get('period', 1.0)
        dpdt = ephem.get('dpdt', 0.0)

        # t = t0 + (phase - phshift) * (period + dpdt(t - t0))

        # if changing this, also see parameters.constraint.time_ephem
        return t0 + ((phase - phshift) * period) / (1 - (phase - phshift) * dpdt)

    def add_dataset(self, method, component=None, **kwargs):
        """
        Add a new dataset to the bundle.  If not provided,
        'dataset' (the name of the new dataset) will be created for
        you and can be accessed by the 'dataset' attribute of the returned
        ParameterSet.

        For light curves, if you do not provide a value for 'component',
        the light curve will be generated for the entire system.

        For radial velocities, you need to provide a list of components
        for which values should be computed.

        Available methods include:
            * :func:`phoebe.parameters.dataset.lc`
            * :func:`phoebe.parameters.dataset.rv`
            * :func:`phoebe.parameters.dataset.etv`
            * :func:`phoebe.parameters.dataset.orb`
            * :func:`phoebe.parameters.dataset.mesh`

        :parameter method: function to call that returns a
            ParameterSet or list of parameters.  This must either be
            a callable function that accepts nothing but default
            values, or the name of a function (as a string) that can
            be found in the :mod:`phoebe.parameters.dataset` module
        :type method: str or callable
        :parameter component: a list of
            components for which to compute the observables.  For
            light curves this should be left at None to always compute
            the light curve for the entire system.  For most other
            types, you need to provide at least one component.
        :type component: str or list of strings or None
        :parameter str dataset: (optional) name of the newly-created dataset
        :parameter **kwargs: default values for any of the newly-created
            parameters
        :return: :class:`phoebe.parameters.parameters.ParameterSet` of
            all parameters that have been added
        :raises NotImplementedError: if required constraint is not implemented
        """
        func = _get_add_func(_dataset, method.lower()
                             if isinstance(method, str)
                             else method)

        kwargs.setdefault('dataset',
                          self._default_label(func.func_name,
                                              **{'context': 'dataset',
                                                 'method': func.func_name.upper()}))

        self._check_label(kwargs['dataset'])

        method = func.func_name.upper()

        # Let's remember if the user passed components or if they were automatically assigned
        user_provided_components = component or kwargs.get('components', False)

        if method == 'LC':
            allowed_components = [None]
        elif method in ['RV', 'ORB']:
            allowed_components = self.hierarchy.get_stars()
            # TODO: how are we going to handle overcontacts dynamical vs flux-weighted
        elif method in ['MESH']:
            # allowed_components = self.hierarchy.get_meshables()
            allowed_components = [None]
            # allowed_components = self.hierarchy.get_stars()
            # TODO: how will this work when changing hierarchy to add/remove the common envelope?
        elif method in ['ETV']:
            hier = self.hierarchy
            stars = hier.get_stars()
            # only include components in which the sibling is also a star that
            # means that the companion in a triple cannot be timed, because how
            # do we know who it's eclipsing?
            allowed_components = [s for s in stars if hier.get_sibling_of(s) in stars]
        else:
            allowed_components = [None]

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
            components = allowed_components
        else:
            raise NotImplementedError

        # Let's handle the case where the user accidentally sends times instead
        # of time
        if kwargs.get('times', None) and not kwargs.get('time', None):
            logger.warning("assuming you meant 'time' instead of 'times'")
            kwargs['time'] = kwargs.pop('times')

        if not np.all([component in allowed_components
                       for component in components]):
            raise ValueError("'{}' not a recognized component".format(component))

        obs_metawargs = {'context': 'dataset',
                         'method': method,
                         'dataset': kwargs['dataset']}
        obs_params, constraints = func()
        self._attach_params(obs_params, **obs_metawargs)

        for constraint in constraints:
            # TODO: tricky thing here will be copying the constraints
            self.add_constraint(*constraint)

        dep_func = _get_add_func(_dataset, "{}_dep".format(method.lower()))
        dep_metawargs = {'context': 'dataset',
                         'method': '{}_dep'.format(method.upper()),
                         'dataset': kwargs['dataset']}
        dep_params = dep_func()
        self._attach_params(dep_params, **dep_metawargs)

        # Now we need to apply any kwargs sent by the user.  There are a few
        # scenarios (and each kwargs could fall into different ones):
        # time = [0,1,2]
        #    in this case, we want to apply time across all of the components that
        #    are applicable for this dataset method AND to _default so that any
        #    future components added to the system are copied appropriately
        # time = [0,1,2], components=['primary', 'secondary']
        #    in this case, we want to apply the value for time across components
        #    but time@_default should remain empty (it will not copy for components
        #    added in the future)
        # time = {'primary': [0,1], 'secondary': [0,1,2]}
        #    here, regardless of the components, we want to apply these to their
        #    individually requested parameters.  We won't touch _default unless
        #    its included in the dictionary

        # set default for times - this way the time array for "attached"
        # components will not be empty
        kwargs.setdefault('time', [0.])


        # pbscale/pblum defaults depend on the hierarchy, calling set_hierarchy
        # will handle these
        # TODO: should this happen before kwargs?
        # TODO: can we avoid rebuilding ALL the constraints when we call this?
        self.set_hierarchy()

        for k, v in kwargs.items():
            if isinstance(v, dict):
                for component, value in v.items():
                    self.set_value_all(qualifier=k,
                                       dataset=kwargs['dataset'],
                                       component=component,
                                       value=value,
                                       check_relevant=False)
            else:
                if components == [None]:
                    components_ = None
                elif user_provided_components:
                    components_ = components
                else:
                    components_ = components+['_default']

                self.set_value_all(qualifier=k,
                                   dataset=kwargs['dataset'],
                                   component=components_,
                                   value=v,
                                   check_relevant=False)



        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.func_name
        self._add_history(redo_func='add_dataset',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_dataset',
                          undo_kwargs={'dataset': kwargs['dataset']})

        return self.filter(dataset=kwargs['dataset'])

    def get_dataset(self, dataset=None, **kwargs):
        """
        Filter in the 'dataset' context

        :parameter str dataset: name of the dataset (optional)
        :parameter **kwargs: any other tags to do the filter
            (except dataset or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if dataset is not None:
            kwargs['dataset'] = dataset

        kwargs['context'] = 'dataset'
        if 'method' in kwargs.keys():
            # since we deal with dataset methods differently (always uppercase)
            # let's just give the user a break if they provided lowercase
            kwargs['method'] = kwargs['method'].upper().replace('DEP', 'dep')
        return self.filter(**kwargs)

    def remove_dataset(self, dataset=None, **kwargs):
        """ Remove a dataset from the Bundle.

        This removes all matching Parameters from the dataset, model, and
        constraint contexts (by default if the context tag is not provided).

        You must provide some sort of filter or this will raise an Error (so
        that all Parameters are not accidentally removed).

        :parameter str dataset: name of the dataset
        :parameter **kwargs: any other tags to do the filter (except qualifier
            and dataset)
        :raises ValueError: if no filter is provided
        """
        # Let's avoid deleting ALL parameters from the matching contexts
        if dataset is None and not len(kwargs.items()):
            raise ValueError("must provide some value to filter for datasets")

        kwargs['dataset'] = dataset
        # Let's avoid the possibility of deleting a single parameter
        kwargs['qualifier'] = None
        # Let's also avoid the possibility of accidentally deleting system
        # parameters, etc
        kwargs.setdefault('context', ['dataset', 'model', 'constraint'])
        # and lastly, let's handle deps if method was passed
        method = kwargs.get('method', None)

        if method is not None:
            if isinstance(method, str):
                method = [method]
            method_deps = []
            for meth in method:
                dep = '{}_dep'.format(meth)
                if dep not in method:
                    method_deps.append(dep)
            method = method + method_deps
        kwargs['method'] = method

        # ps = self.filter(**kwargs)
        # logger.info('removing {} parameters (this is not undoable)'.\
        #             format(len(ps)))

        # print "*** kwargs", kwargs, len(ps)
        self.remove_parameters_all(**kwargs)
        # not really sure why we need to call this twice, but it seems to do
        # the trick
        self.remove_parameters_all(**kwargs)

        # TODO: check to make sure that trying to undo this
        # will raise an error saying this is not undo-able
        self._add_history(redo_func='remove_dataset',
                          redo_kwargs={'dataset': dataset},
                          undo_func=None,
                          undo_kwargs={})

        return

    def enable_dataset(self, dataset=None, **kwargs):
        """
        Enable a 'dataset'.  Datasets that are enabled will be computed
        during :meth:`run_compute` and included in the cost function
        during :meth:`run_fitting`.

        :parameter str dataset: name of the dataset
        :parameter **kwargs: any other tags to do the filter
            (except dataset or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
            of the enabled dataset
        """
        kwargs['context'] = 'dataset'
        kwargs['dataset'] = dataset
        kwargs['qualifier'] = 'enabled'
        self.set_value(value=True, **kwargs)

        self._add_history(redo_func='enable_dataset',
                          redo_kwargs={'dataset': dataset},
                          undo_func='disable_dataset',
                          undo_kwargs={'dataset': dataset})

        return self.get_dataset(dataset=dataset)

    def disable_dataset(self, dataset=None, **kwargs):
        """
        Disable a 'dataset'.  Datasets that are enabled will be computed
        during :meth:`run_compute` and included in the cost function
        during :meth:`run_fitting`.

        :parameter str dataset: name of the dataset
        :parameter **kwargs: any other tags to do the filter
            (except dataset or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
            of the disabled dataset
        """
        kwargs['context'] = 'dataset'
        kwargs['dataset'] = dataset
        kwargs['qualifier'] = 'enabled'
        self.set_value(value=False, **kwargs)

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
        TODO: add documentation

        args can be string representation (length 1)
        func and strings to pass to function
        """
        # TODO: be smart enough to take kwargs (especially for undoing a
        # remove_constraint) for method, value (expression),

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
            if 'method' in kwargs.keys():
                func = _get_add_func(_constraint, kwargs['method'])
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
            kwargs['solve_for'] = self.get_parameter(kwargs['solve_for'])

        lhs, rhs, constraint_kwargs = func(self, *func_args, **kwargs)
        # NOTE that any component parameters required have already been
        # created by this point

        constraint_param = ConstraintParameter(self,
                                               qualifier=lhs.qualifier,
                                               component=lhs.component,
                                               dataset=lhs.dataset,
                                               method=lhs.method,
                                               model=lhs.model,
                                               constraint_func=func.__name__,
                                               constraint_kwargs=constraint_kwargs,
                                               value=rhs,
                                               default_unit=lhs.default_unit,
                                               description='expression that determines the constraint')

        metawargs = {'context': 'constraint',
                     'method': func.func_name}

        params = ParameterSet([constraint_param])
        constraint_param._update_bookkeeping()
        self._attach_params(params, **metawargs)

        redo_kwargs['func'] = func.func_name

        self._add_history(redo_func='add_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_constraint',
                          undo_kwargs={'uniqueid': constraint_param.uniqueid})

        # we should run it now to make sure everything is in-sync
        self.run_constraint(uniqueid=constraint_param.uniqueid)

        return params

    def get_constraint(self, twig=None, **kwargs):
        """
        Filter in the 'constraint' context

        :parameter str constraint: name of the constraint (optional)
        :parameter **kwargs: any other tags to do the filter
            (except constraint or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if twig is not None:
            kwargs['twig'] = twig
        kwargs['context'] = 'constraint'
        return self.get(**kwargs)

    def remove_constraint(self, twig=None, **kwargs):
        """
        Remove a 'constraint' from the bundle

        :parameter str twig: twig to filter for the constraint
        :parameter **kwargs: any other tags to do the filter
            (except twig or context)
        """
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
        Flip an existing constraint to solve for a different parameter

        :parameter str twig: twig to filter the constraint
        :parameter solve_for: twig or actual parameter object of the new
            parameter which this constraint should constraint (solve for).
        :type solve_for: str or :class:`phoebe.parameters.parameters.Parameter
        :parameter **kwargs: any other tags to do the filter
            (except twig or context)

        """
        kwargs['twig'] = twig
        redo_kwargs = deepcopy(kwargs)
        undo_kwargs = deepcopy(kwargs)

        param = self.get_constraint(**kwargs)

        if solve_for is None:
            return param
        if isinstance(solve_for, Parameter):
            solve_for = solve_for.uniquetwig

        redo_kwargs['solve_for'] = solve_for
        undo_kwargs['solve_for'] = param.constrained_parameter.uniquetwig

        logger.info("flipping constraint '{}' to solve for '{}'".format(param.uniquetwig, solve_for))
        param.flip_for(solve_for)

        result = self.run_constraint(uniqueid=param.uniqueid)

        self._add_history(redo_func='flip_constraint',
                          redo_kwargs=redo_kwargs,
                          undo_func='flip_constraint',
                          undo_kwargs=undo_kwargs)

        return param

    def run_constraint(self, twig=None, **kwargs):
        """
        Run a given 'constraint' now and set the value of the constrained
        parameter.  In general, there shouldn't be any need to manually
        call this - constraints should automatically be run whenever a
        dependent parameter's value is change.

        :parameter str twig: twig to filter for the constraint
        :parameter **kwargs: any other tags to do the filter
            (except twig or context)
        :return: the resulting value of the constraint
        :rtype: float or units.Quantity
        """
        kwargs['twig'] = twig
        kwargs['context'] = 'constraint'
        # kwargs['qualifier'] = 'expression'
        expression_param = self.get_parameter(**kwargs)

        kwargs = {}
        kwargs['twig'] = None
        # TODO: this might not be the case, we just know its not in constraint
        kwargs['context'] = ['component', 'dataset']
        kwargs['qualifier'] = expression_param.qualifier
        kwargs['component'] = expression_param.component
        kwargs['dataset'] = expression_param.dataset
        constrained_param = self.get_parameter(**kwargs)

        result = expression_param.result

        constrained_param.set_value(result, force=True)

        logger.info("setting '{}'={} from '{}' constraint".format(constrained_param.uniquetwig, result, expression_param.uniquetwig))

        return result

    def add_compute(self, method=compute.phoebe, **kwargs):
        """
        Add a set of computeoptions for a given backend to the bundle.
        The label ('compute') can then be sent to :meth:`run_compute`.

        If not provided, 'compute' will be created for you and can be
        accessed by the 'compute' attribute of the returned
        ParameterSet.

        Available methods include:
            * :func:`phoebe.parameters.compute.phoebe`
            * :func:`phoebe.parameters.compute.legacy`
            * :func:`phoebe.parameters.compute.photodynam`
            * :func:`phoebe.parameters.compute.jktebop`

        :parameter method: function to call that returns a
            ParameterSet or list of parameters.  This must either be
            a callable function that accepts nothing but default
            values, or the name of a function (as a string) that can
            be found in the :mod:`phoebe.parameters.compute` module
        :type method: str or callable
        :parameter str compute: (optional) name of the newly-created
            compute optins
        :parameter **kwargs: default values for any of the newly-created
            parameters
        :return: :class:`phoebe.parameters.parameters.ParameterSet` of
            all parameters that have been added
        :raises NotImplementedError: if required constraint is not implemented
        """
        func = _get_add_func(_compute, method)

        kwargs.setdefault('compute',
                          self._default_label(func.func_name,
                                              **{'context': 'compute',
                                                 'method': func.func_name}))

        self._check_label(kwargs['compute'])

        params = func(**kwargs)
        # TODO: similar kwargs logic as in add_dataset (option to pass dict to
        # apply to different components this would be more complicated here if
        # allowing to also pass to different datasets

        metawargs = {'context': 'compute',
                     'method': func.func_name,
                     'compute': kwargs['compute']}

        logger.info("adding {} '{}' compute to bundle".format(metawargs['method'], metawargs['compute']))
        self._attach_params(params, **metawargs)

        redo_kwargs = deepcopy(kwargs)
        redo_kwargs['func'] = func.func_name
        self._add_history(redo_func='add_compute',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_compute',
                          undo_kwargs={'compute': kwargs['compute']})

        return params

    def get_compute(self, compute=None, **kwargs):
        """
        Filter in the 'compute' context

        :parameter str compute: name of the compute options (optional)
        :parameter **kwargs: any other tags to do the filter
            (except compute or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if compute is not None:
            kwargs['compute'] = compute
        kwargs['context'] = 'compute'
        return self.filter(**kwargs)

    def remove_compute(self, compute, **kwargs):
        """
        [NOT IMPLEMENTED]
        Remove a 'constraint' from the bundle

        :parameter str twig: twig to filter for the compute options
        :parameter **kwargs: any other tags to do the filter
            (except twig or context)
        :raise NotImplementedError: because it isn't
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    @send_if_client
    def run_compute(self, compute=None, model=None, detach=False,
                    animate=False, time=None, **kwargs):
        """
        Run a forward model of the system on the enabled dataset using
        a specified set of compute options.

        To attach and set custom values for compute options, including choosing
        which backend to use, see:
            * :meth:`add_compute`

        To define the dataset types and times at which the model should be
        computed see:
            * :meth:`add_dataset`

        To disable or enable existing datasets see:
            * :meth:`enable_dataset`
            * :meth:`disable_dataset`

        :parameter str compute: (optional) name of the compute options to use.
            If not provided or None, run_compute will use an existing set of
            attached compute options if only 1 exists.  If more than 1 exist,
            then compute becomes a required argument.  If no compute options
            exist, then this will use default options and create and attach
            a new set of compute options with a default label.
        :parameter str model: (optional) name of the resulting model.  If not
            provided this will default to 'tmpmodel'.  NOTE: existing models
            with the same name will be overwritten - including 'tmpmodel'
        :parameter bool datach: [EXPERIMENTAL] whether to detach from the computation run,
            or wait for computations to complete.  If detach is True, see
            :meth:`get_model` and :meth:`phoebe.parameters.parameters.JobParameter`
            for details on how to check the job status and retrieve the results.
            Alternatively, you can provide the server location (host and port) as
            a string to detach and the bundle will temporarily enter client mode,
            submit the job to the server, and leave client mode.  The resulting
            :meth:`phoebe.parameters.parameters.JobParameter` will then contain
            the necessary information to pull the results from the server at anytime
            in the future.
        :parameter animate: [EXPERIMENTAL] information to send to :meth:`animate`
            while the synthetics are being built.  If not False (in which case
            live animation will not be done), animate should be a dictionary or
            list of dictionaries and a new frame will be displayed and plotted
            as they are computed.  This really only makes sense for backends
            that compute per-time rather than per-dataset.  Note: animation
            may significantly slow down the time of run_compute, especially
            for a large number of time-points or if meshes are being stored/plotted.
            Also note: animate will obviously be ignored if detach=True, this
            isn't magic.  NOTE: fixed_limits are not supported from run_compute,
            axes limits will be updated each frame, but all colorlimits will
            be determined per-frame and will not be constant across the animation.
        :parameter list time: [EXPERIMENTAL] override the times at which to compute the model.
            NOTE: this only (temporarily) replaces the time array for datasets
            with times provided (ie empty time arrays are still ignored).  So if
            you attach a RV to a single component, the model will still only
            compute for that single component.  ALSO NOTE: this option is ignored
            if detach=True (at least for now).
        :parameter **kwargs: any values in the compute options to temporarily
            override for this single compute run (parameter values will revert
            after run_compute is finished)
        :return: :class:`phoebe.parameters.parameters.ParameterSet` of the
            newly-created model containing the synthetic data.
        """
        if isinstance(detach, str):
            # then we want to temporarily go in to client mode
            self.as_client(server=detach)
            self.run_compute(compute=compute, model=model, time=time, **kwargs)
            self.as_client(False)
            return self.get_model(model)

        if model is None:
            model = 'tmpmodel'

        passed, msg = self.run_checks()
        if not passed and not kwargs.get('skip_checks', False):
            raise ValueError("system failed to pass checks: {}".format(msg))

        if model in self.models:
            logger.warning("overwriting model: {}".format(model))
            self.remove_model(model)

        self._check_label(model)

        if isinstance(time, float) or isinstance(time, int):
            time = [time]

        # handle case where compute is not provided
        if compute is None:
            computes = self.get_compute().computes
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
        if detach:
            logger.warning("detach support is EXPERIMENTAL")

            if time is not None:
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
            f.write("import phoebe; import json\n")
            # TODO: can we skip the history context?  And maybe even other models
            # or datasets (except times and only for run_compute but not run_fitting)
            f.write("bdict = json.loads(\"\"\"{}\"\"\")\n".format(json.dumps(self.to_json())))
            f.write("b = phoebe.Bundle(bdict)\n")
            # TODO: make sure this works with multiple computes
            f.write("model_ps = b.run_compute(compute='{}', model='{}')\n".format(compute, model))  # TODO: support other kwargs
            f.write("model_ps.save('_{}.out', incl_uniqueid=True)\n".format(jobid))
            f.close()

            script_fname = os.path.abspath(script_fname)
            cmd = 'python {} &>/dev/null &'.format(script_fname)
            subprocess.call(cmd, shell=True)

            # create model parameter and attach (and then return that instead of None)
            job_param = JobParameter(self,
                                     location=os.path.dirname(script_fname),
                                     status_method='exists',
                                     retrieve_method='local',
                                     uniqueid=jobid)

            metawargs = {'context': 'model', 'model': model}
            self._attach_params([job_param], **metawargs)

            logger.info("detaching from run_compute.  Call get_model('{}').attach() to re-attach".format(model))

            if isinstance(detach, str):
                self.save(detach)

            # return self.get_model(model)
            return job_param

        for compute in computes:

            computeparams = self.get_compute(compute=compute)

            if not computeparams.method:
                raise KeyError("could not recognize backend from compute: {}".format(compute))

            logger.info("running {} backend to create '{}' model".format(computeparams.method, model))
            compute_func = getattr(backends, computeparams.method)

            metawargs = {'compute': compute, 'model': model, 'context': 'model'}  # dataset, component, etc will be set by the compute_func

            if animate:
                # handle setting defaults from kwargs to each plotting call
                compute_generator = compute_func(self, compute, as_generator=True, time=time, **kwargs)

                # In order to build the first frame and initialize the animation,
                # we'll iterate the generator once (ie compute the first time-step)
                ps_tmp, time = next(compute_generator)
                ps_tmp.set_meta(**metawargs) # TODO: is this necessary?

                # Now we'll initialize the figure and send the generator to the
                # animator.  The animator will then handle looping through the
                # rest of the generator to finish computing the synthetic
                # model.
                plotting_args = parameters._parse_plotting_args(animate)
                for plot_args in plotting_args:
                    # live-plotting doesn't support highlight (because time
                    # array is already filled and interpolation will fail)

                    # TODO: make this work to be defaulted to True (current
                    # problem is that time array is prepopulated)
                    plot_args['highlight'] = False
                    # plot_args['uncover'] = False

                anim, ao = mpl_animate.animate(self-self.filter(context='model'),
                                               init_ps=self-self.filter(context='model')+ps_tmp,
                                               init_time=time,
                                               frames=compute_generator,
                                               fixed_limits=False,
                                               plotting_args=plotting_args,
                                               metawargs=metawargs)

                plt.show()

                # NOTE: this will not finish if the mpl window is closed before
                # all times are filled

                # TODO: can we make sure the generator is finished, and if it
                # isn't complete the loop rather than access ao.latest_frame?
                # Or alternatively, if we didn't just copy the time array we
                # could leave an "incomplete" model that wouldn't fail future
                # plotting

                # assuming the animation was allowed to complete, the ao object
                # holds the last yielded parameters and time.  Let's take the
                # params here so that they can be attached to the bundle.
                params, last_time = ao.latest_frame

            else:
                # comma in the following line is necessary because compute_func
                # is /technically/ a generator (it yields instead of returns)
                params, = compute_func(self, compute, time=time, **kwargs)

            self._attach_params(params, **metawargs)

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

        :parameter str model: name of the model (optional)
        :parameter **kwargs: any other tags to do the filter
            (except model or context)
        :return: :class:`phoebe.parameters.parameters.ParameterSet`
        """
        if model is not None:
            kwargs['model'] = model
        kwargs['context'] = 'model'
        return self.filter(**kwargs)

    def remove_model(self, model, **kwargs):
        """
        Remove a 'model' from the bundle

        :parameter str twig: twig to filter for the model
        :parameter **kwargs: any other tags to do the filter
            (except twig or context)
        """
        kwargs['model'] = model
        kwargs['context'] = 'model'
        self.remove_parameters_all(**kwargs)

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
        redo_kwargs['func'] = func.func_name
        self._add_history(redo_func='add_prior',
                          redo_kwargs=redo_kwargs,
                          undo_func='remove_prior',
                          undo_kwargs={'twig': param.uniquetwig})

        return params

    def get_prior(self, twig=None, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
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
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
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

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError
        if fitting is not None:
            kwargs['fitting'] = fitting
        kwargs['context'] = 'fitting'
        return self.filter(**kwargs)

    def remove_fitting(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
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

        :raises NotImplementedError: because it isn't
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

        :raises NotImplementedError: because it isn't
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

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError
        if feedback is not None:
            kwargs['feedback'] = feedback
        kwargs['context'] = 'feedback'
        return self.filter(**kwargs)

    def remove_feedback(self, feedback=None):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
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

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError

    def remove_plugin(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # TODO: don't forget add_history
        raise NotImplementedError

    def run_plugin(self):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        raise NotImplementedError
