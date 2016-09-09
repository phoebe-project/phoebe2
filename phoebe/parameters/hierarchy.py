
from phoebe.parameters import *

def _to_component(obj, allow_hierarchy=True):
    """
    takes either a string, ParameterSet, Parameter, or the string representation (component or nested hierarchy)
    """

    def _to_str(obj):
        return "{}:{}".format(obj.kind, obj.component)

    if isinstance(obj, str):
        # TODO: check to see if valid?, use allow_hierarchy
        # TODO: when passed labels this is going to give the wrong thing, but that might be fixable in the HierarchyParameter set_value check
        #~ raise NotImplementedError # need to get object from bundle and pass to _to_str
        return obj
    elif isinstance(obj, ParameterSet):
        # TODO: be smarter about this and don't assume only 1 will be returned
        if 'repr' in obj.to_flat_dict().keys():
            return obj.get_value(qualifier='repr')

        else:
            # make sure we only have things in the 'component' context
            obj = obj.filter(context='component')
            return _to_str(obj)

    elif isinstance(obj, Parameter):
        if obj.qualifier == 'repr':
            return obj.get_value()
        else:
            return _to_str(obj)

    else:
        raise NotImplementedError("could not parse {}".format(obj))


def binaryorbit(orbit, comp1, comp2, envelope=None):
    """
    Build the string representation of a hierarchy containing a binary
    orbit with 2 components.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.set_hierarchy`

    :parameter comp1: an existing hierarchy string, Parameter, or ParameterSet
    :parameter comp2: an existing hierarchy string, Parameter, or ParameterSet
    :return: the string representation of the hierarchy
    """

    if envelope:
        return '{}({}, {}, {})'.format(_to_component(orbit, False), _to_component(comp1), _to_component(comp2), _to_component(envelope, False))
    else:
        return '{}({}, {})'.format(_to_component(orbit, False), _to_component(comp1), _to_component(comp2))

def component(*args):
    """
    Build the string representation of a hierarchy that groups multiple
    objects without a parent orbit (ie a disk around a planet).

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.set_hierarchy`

    :parameter *args:
    :return: the string representation of the hierarchy
    """

    return 'component({})'.format(', '.join([_to_component(arg) for arg in args]))

def blank(*args):
    """
    Build the string representation of a blank hierarchy.

    Generally, this will be used as an input to :func:`Bundle.set_hierarchy`

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.set_hierarchy`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: the string representation of the hierarchy
    """
    return ''

