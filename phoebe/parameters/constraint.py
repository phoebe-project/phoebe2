import numpy as np
#from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt

from phoebe.parameters.parameters import ParameterSet, ConstraintParameter, FloatParameter
from phoebe.constraints.expression import ConstraintVar
from phoebe import u, c

import logging
logger = logging.getLogger("CONSTRAINT")
logger.addHandler(logging.NullHandler())

list_of_constraints_requiring_si = []

_skip_filter_checks = {'check_default': False, 'check_visible': False}

_validsolvefor = {}

def _get_system_ps(b, item, context='component'):
    """
    parses the input arg (either twig or PS) to retrieve the actual parametersets
    """
    # TODO: make this a decorator?
    if isinstance(item, list) and len(item)==1:
        item = item[0]

    if isinstance(item, ParameterSet):
        return item.filter(context=context, **_skip_filter_checks)
    elif isinstance(item, str):
        return b.filter(item, context=context, **_skip_filter_checks)
    else:
        logger.debug("_get_system_ps got {}".format(item))
        raise NotImplementedError("_get_system_ps does not support item with type: {}".format(type(item)))

#{ Mathematical expressions

# these all return a constraint expression... everything else in this module is meant
# to be used through b.add_constraint()

def _get_expr(param):
    """
    """
    if hasattr(param, 'expr'):
        return param.expr
    elif hasattr(param, 'uniquetwig'):
        return "{%s}" % param.uniquetwig
    else:
        raise NotImplementedError("could not build constraint expression with type {}".format(type(param)))

def sin(param):
    """
    Allows using the sin function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "sin({})".format(_get_expr(param)))

def cos(param):
    """
    Allows using the cos function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "cos({})".format(_get_expr(param)))

def tan(param):
    """
    Allows using the tan function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "tan({})".format(_get_expr(param)))

def arcsin(param):
    """
    Allows using the arcsin function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "arcsin({})".format(_get_expr(param)))

def arccos(param):
    """
    Allows using the arccos function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    # print "***", "arccos({})".format(_get_expr(param))
    return ConstraintParameter(param._bundle, "arccos({})".format(_get_expr(param)))

def arctan(param):
    """
    Allows using the arctan function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "arctan({})".format(_get_expr(param)))

def arctan2(param1, param2):
    """
    Allows using the arctan2 function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param1._bundle, "arctan2({}, {})".format(_get_expr(param1), _get_expr(param2)))

def arctan2(param1, param2):
    """
    Allows using the arctan2 function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param1._bundle, "arctan2({}, {})".format(_get_expr(param1), _get_expr(param2)))

def abs(param):
    """
    Allows using the abs (absolute value) function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "abs({})".format(_get_expr(param)))

def sqrt(param):
    """
    Allows using the sqrt (square root) function in a constraint

    Arguments
    ------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    ---------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "sqrt({})".format(_get_expr(param)))

def log10(param):
    """
    Allows using the log10 function in a constraint

    Arguments
    ----------------
    * `param` (<phoebe.parameters.Parameter>)

    Returns
    -----------
    * (<phoebe.parameters.ConstraintParameter>)
    """
    return ConstraintParameter(param._bundle, "log10({})".format(_get_expr(param)))

#}
#{ Built-in functions (see phoebe.constraints.builtin for actual functions)
def roche_requiv_L1(q, syncpar, ecc, sma, incl_star, long_an_star, incl_orb, long_an_orb, compno=1):
    return ConstraintParameter(q._bundle, "requiv_L1(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q, syncpar, ecc, sma, incl_star, long_an_star, incl_orb, long_an_orb)]), compno))

def roche_requiv_contact_L1(q, sma, compno=1):
    return ConstraintParameter(q._bundle, "requiv_contact_L1(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q, sma)]), compno))

def roche_requiv_contact_L23(q, sma, compno=1):
    return ConstraintParameter(q._bundle, "requiv_contact_L23(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q, sma)]), compno))

def roche_potential_contact_L1(q):
    return ConstraintParameter(q._bundle, "potential_contact_L1({})".format(_get_expr(q)))

def roche_potential_contact_L23(q):
    return ConstraintParameter(q._bundle, "potential_contact_L23({})".format(_get_expr(q)))

def roche_pot_to_fillout_factor(q, pot):
    return ConstraintParameter(q._bundle, "pot_to_fillout_factor({}, {})".format(_get_expr(q), _get_expr(pot)))

def roche_fillout_factor_to_pot(q, fillout_factor):
    return ConstraintParameter(q._bundle, "fillout_factor_to_pot({}, {})".format(_get_expr(q), _get_expr(fillout_factor)))

def requiv_to_pot_contact(requiv, q, sma, compno=1):
    return ConstraintParameter(requiv._bundle, "requiv_to_pot_contact({}, {}, {}, {})".format(_get_expr(requiv), _get_expr(q), _get_expr(sma), compno))

def pot_to_requiv_contact(pot, q, sma, compno=1):
    return ConstraintParameter(pot._bundle, "pot_to_requiv_contact({}, {}, {}, {})".format(_get_expr(pot), _get_expr(q), _get_expr(sma), compno))

def esinw2per0(ecc, esinw):
    return ConstraintParameter(ecc._bundle, "esinw2per0({}, {})".format(_get_expr(ecc), _get_expr(esinw)))

def ecosw2per0(ecc, ecosw):
    return ConstraintParameter(ecc._bundle, "ecosw2per0({}, {})".format(_get_expr(ecc), _get_expr(ecosw)))

def esinw2ecc(esinw, per0):
    """
    TODO: add documentation
    """
    return ConstraintParameter(esinw._bundle, "esinw2ecc({}, {})".format(_get_expr(esinw), _get_expr(per0)))

def ecosw2ecc(ecosw, per0):
    """
    TODO: add documentation
    """
    return ConstraintParameter(ecosw._bundle, "ecosw2ecc({}, {})".format(_get_expr(ecosw), _get_expr(per0)))

def t0_perpass_to_supconj(t0_perpass, period, ecc, per0, dpdt, dperdt, t0):
    return ConstraintParameter(t0_perpass._bundle, "t0_perpass_to_supconj({}, {}, {}, {}, {}, {}, {})".format(_get_expr(t0_perpass), _get_expr(period), _get_expr(ecc), _get_expr(per0), _get_expr(dpdt), _get_expr(dperdt), _get_expr(t0)))

def t0_supconj_to_perpass(t0_supconj, period, ecc, per0, dpdt, dperdt, t0):
    return ConstraintParameter(t0_supconj._bundle, "t0_supconj_to_perpass({}, {}, {}, {}, {}, {}, {})".format(_get_expr(t0_supconj), _get_expr(period), _get_expr(ecc), _get_expr(per0), _get_expr(dpdt), _get_expr(dperdt), _get_expr(t0)))

def t0_ref_to_supconj(t0_ref, period, ecc, per0, dpdt, dperdt, t0):
    return ConstraintParameter(t0_ref._bundle, "t0_ref_to_supconj({}, {}, {}, {}, {}, {}, {})".format(_get_expr(t0_ref), _get_expr(period), _get_expr(ecc), _get_expr(per0), _get_expr(dpdt), _get_expr(dperdt), _get_expr(t0)))

def t0_supconj_to_ref(t0_supconj, period, ecc, per0, dpdt, dperdt, t0):
    return ConstraintParameter(t0_supconj._bundle, "t0_supconj_to_ref({}, {}, {}, {}, {}, {}, {})".format(_get_expr(t0_supconj), _get_expr(period), _get_expr(ecc), _get_expr(per0), _get_expr(dpdt), _get_expr(dperdt), _get_expr(t0)))

def _times_to_phases(times, period_choice, period, period_anom, phases_dpdt, dpdt, t0_choice, t0_supconj, t0_perpass, t0_ref):
    return ConstraintParameter(times._bundle, "times_to_phases({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(_get_expr(times), _get_expr(period_choice), _get_expr(period), _get_expr(period_anom), _get_expr(phases_dpdt), _get_expr(dpdt), _get_expr(t0_choice), _get_expr(t0_supconj), _get_expr(t0_perpass), _get_expr(t0_ref)))

def _phases_to_times(phases, period_choice, period, period_anom, phases_dpdt, dpdt, t0_choice, t0_supconj, t0_perpass, t0_ref):
    return ConstraintParameter(phases._bundle, "phases_to_times({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(_get_expr(phases), _get_expr(period_choice), _get_expr(period), _get_expr(period_anom), _get_expr(phases_dpdt), _get_expr(dpdt), _get_expr(t0_choice), _get_expr(t0_supconj), _get_expr(t0_perpass), _get_expr(t0_ref)))

#{ Custom constraints

def custom(b, *args, **kwargs):
    """
    [NOT IMPLEMENTED]

    args can be
    - 2 ConstraintParameters (or parameters which need param.to_constraint()) (lhs, rhs)
    - 2 parsable strings (lhs, rhs)
    - single parsable string (lhs, rhs = args[0].split('=')

    :raise NotImplementedError: because it isn't
    """

    raise NotImplementedError("custom constraints not yet supported")

    # TODO: handle parsing different types of input

    # create parameters
    #~ params = []
    # TODO: fix this to also accept constraint objects for lhs
    #~ params += [TwigParameter(b, qualifier='solve_for', value=lhs.uniquetwig, description='which parameter should be constrained by the others')]
    #~ params += [ConstraintParameter(b, qualifier='expression', value=rhs, description='expression that determines the constraint')]

    #~ return ParameterSet(params)

#}
#{ Intra-orbit constraints

_validsolvefor['asini'] = ['asini', 'incl', 'sma']
def asini(b, orbit, solve_for=None):
    """
    Create a constraint for asini in an orbit.

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('asini', orbit='binary')`, where `orbit` is one of
     <phoebe.parameters.HierarchyParameter.get_orbits>.

    If any of the required parameters ('asini', 'sma', 'incl') do not
    exist in the orbit, they will be created.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'asini' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'sma' or 'incl')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    orbit_ps = _get_system_ps(b, orbit)

    # We want to get the parameters in THIS orbit, but calling through
    # the bundle in case we need to create it.
    # To do that, we need to know the search parameters to get items from this PS.
    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    # Now we'll define the parameters in case they don't exist and need to be created
    sma_def = FloatParameter(qualifier='sma', latexfmt=r'a_\mathrm{{ {component} }}', value=8.0, default_unit=u.solRad, description='Semi major axis')
    incl_def = FloatParameter(qualifier='incl', latexfmt=r'a_\mathrm{{ {component} }}', value=90.0, default_unit=u.deg, description='Orbital inclination angle')
    asini_def = FloatParameter(qualifier='asini', latexfmt=r'a_\mathrm{{ {component} }} \sin i_\mathrm{{ {component} }}', value=8.0, default_unit=u.solRad, description='Projected semi major axis')

    # And now call get_or_create on the bundle
    sma, created = b.get_or_create('sma', sma_def, **metawargs)
    incl, created = b.get_or_create('incl', incl_def, **metawargs)
    asini, created = b.get_or_create('asini', asini_def, **metawargs)

    if solve_for in [None, asini]:
        lhs = asini
        rhs = sma * sin(incl)

    elif solve_for == sma:
        lhs = sma
        rhs = asini / sin(incl)

    elif solve_for == incl:
        lhs = incl
        rhs = arcsin(asini/sma)

    else:
        raise NotImplementedError

    #- return lhs, rhs, args_as_pss
    return lhs, rhs, [], {'orbit': orbit}

_validsolvefor['esinw'] = ['esinw', 'ecc', 'per0', 'ecosw']
def esinw(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for esinw in an orbit.

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint>  as
     `b.add_constraint('esinw', orbit='binary')`, where `orbit` is one of
     <phoebe.parameters.HierarchyParameter.get_orbits>.

    If 'esinw' does not exist in the orbit, it will be created

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'esinw' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'ecc' or 'per0')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    orbit_ps = _get_system_ps(b, orbit)

    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    esinw_def = FloatParameter(qualifier='esinw', latexfmt=r'e_\mathrm{{ {component} }} \sin \omega_0', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times sin of argument of periastron')
    esinw, created = b.get_or_create('esinw', esinw_def, **metawargs)

    ecosw_def = FloatParameter(qualifier='ecosw', latexfmt=r'e_\mathrm{{ {component} }} \cos \omega_0', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times cos of argument of periastron')
    ecosw, ecosw_created = b.get_or_create('ecosw', ecosw_def, **metawargs)

    ecosw_constrained = kwargs.get('ecosw_constrained', len(ecosw.constrained_by) > 0)
    logger.debug("esinw constraint: solve_for={}, ecosw_constrained={}, ecosw_created={}".format(solve_for.qualifier if solve_for is not None else "None", ecosw_constrained, ecosw_created))

    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)

    if solve_for in [None, esinw]:
        lhs = esinw
        rhs = ecc * sin(per0)
        if not ecosw_created and not ecosw_constrained:
            if per0.is_constraint and per0.is_constraint.constraint_func != 'esinw':
                per0.is_constraint.constraint_kwargs['esinw_constrained'] = True
                per0.is_constraint.flip_for('per0', force=True)
            elif ecc.is_constraint and ecc.is_constraint.constraint_func != 'esinw':
                ecc.is_constraint.constraint_kwargs['esinw_constrained'] = True
                ecc.is_constraint.flip_for('ecc', force=True)

    elif solve_for == ecc:
        lhs = ecc
        if ecosw_constrained:
            # cannot just do esinw/sin(per0) because sin(per0) may be zero
            rhs = esinw2ecc(esinw, per0)
        else:
            rhs = (esinw**2 + ecosw**2)**0.5
            # the other constraint needs to also follow the alternate equations
            if per0.is_constraint and 'esinw_constrained' not in per0.is_constraint.constraint_kwargs.keys():
                logger.debug("esinw constraint: attempting to also flip per0 constraint")
                per0.is_constraint.constraint_kwargs['esinw_constrained'] = False
                per0.is_constraint.flip_for('per0', force=True)

    elif solve_for == per0:
        lhs = per0
        if ecosw_constrained:
            # cannot just do arcsin because ecc may be zero
            rhs = esinw2per0(ecc, esinw)
        else:
            rhs = arctan2(esinw, ecosw)
            # the other constraint needs to also follow the alternate equations
            if ecc.is_constraint and 'esinw_constrained' not in ecc.is_constraint.constraint_kwargs.keys():
                logger.debug("esinw constraint: attempting to also flip ecc constraint")
                ecc.is_constraint.constraint_kwargs['esinw_constrained'] = False
                ecc.is_constraint.flip_for('ecc', force=True)
    elif solve_for == ecosw:
        raise NotImplementedError("cannot solve this constraint for 'ecosw' since it was originally 'esinw'")
    else:
        raise NotImplementedError

    return lhs, rhs, [esinw, ecosw, ecc, per0], {'orbit': orbit}

_validsolvefor['ecosw'] = ['ecosw', 'ecc', 'per0', 'esinw']
def ecosw(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for ecosw in an orbit.

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('ecosw', orbit='binary')`, where `orbit` is one of
     <phoebe.parameters.HierarchyParameter.get_orbits>.

    If 'ecosw' does not exist in the orbit, it will be created

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'ecosw' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'ecc' or 'per0')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    orbit_ps = _get_system_ps(b, orbit)

    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    ecosw_def = FloatParameter(qualifier='ecosw', latexfmt=r'e_\mathrm{{ {component} }} \cos \omega_0', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times cos of argument of periastron')
    ecosw, created = b.get_or_create('ecosw', ecosw_def, **metawargs)

    esinw_def = FloatParameter(qualifier='esinw', latexfmt=r'e_\mathrm{{ {component} }} \sin \omega_0', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times sin of argument of periastron')
    esinw, esinw_created = b.get_or_create('esinw', esinw_def, **metawargs)

    esinw_constrained = kwargs.get('esinw_constrained', len(esinw.constrained_by) > 0)
    logger.debug("ecosw constraint: solve_for={}, esinw_constrained={}, esinw_created={}".format(solve_for.qualifier if solve_for is not None else "None", esinw_constrained, esinw_created))

    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)

    if solve_for in [None, ecosw]:
        lhs = ecosw
        rhs = ecc * cos(per0)
        if not esinw_created and not esinw_constrained:
            if per0.is_constraint and per0.is_constraint.constraint_func != 'ecosw':
                per0.is_constraint.constraint_kwargs['ecosw_constrained'] = True
                per0.is_constraint.flip_for('per0', force=True)
            elif ecc.is_constraint and ecc.is_constraint.constraint_func != 'ecosw':
                ecc.is_constraint.constraint_kwargs['ecosw_constrained'] = True
                ecc.is_constraint.flip_for('ecc', force=True)

    elif solve_for == ecc:
        lhs = ecc
        if esinw_constrained:
            # cannot just do ecosw/cos(per0) because cos(per0) may be zero
            rhs = ecosw2ecc(ecosw, per0)
        else:
            rhs = (esinw**2 + ecosw**2)**0.5
            # the other constraint needs to also follow the alternate equations
            if per0.is_constraint and 'ecosw_constrained' not in per0.is_constraint.constraint_kwargs.keys():
                logger.debug("ecosw constraint: attempting to also flip per0 constraint")
                per0.is_constraint.constraint_kwargs['ecosw_constrained'] = False
                per0.is_constraint.flip_for('per0', force=True)

    elif solve_for == per0:
        lhs = per0
        if esinw_constrained:
            # cannot just do arccos because ecc may be 0
            rhs = ecosw2per0(ecc, ecosw)
        else:
            rhs = arctan2(esinw, ecosw)
            # the other constraint needs to also follow the alternate equations
            if ecc.is_constraint and 'ecosw_constrained' not in ecc.is_constraint.constraint_kwargs.keys():
                logger.debug("ecosw constraint: attempting to also flip per0 constraint")
                ecc.is_constraint.constraint_kwargs['ecosw_constrained'] = False
                ecc.is_constraint.flip_for('ecc', force=True)
    elif solve_for == esinw:
        raise NotImplementedError("cannot solve this constraint for 'esinw' since it was originally 'ecosw'")
    else:
        raise NotImplementedError

    return lhs, rhs, [esinw, ecosw, ecc, per0], {'orbit': orbit}

_validsolvefor['t0_perpass_supconj'] = ['t0_perpass', 't0_supconj']
def t0_perpass_supconj(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for t0_perpass in an orbit - allowing translating between
    t0_perpass and t0_supconj.

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('t0_perpass_supconj', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'to_supconj' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 't0_perpass', 'period', 'ecc',
        or 'per0')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    orbit_ps = _get_system_ps(b, orbit)

    # by default both t0s exist in an orbit, so we don't have to worry about creating either
    t0_perpass = orbit_ps.get_parameter(qualifier='t0_perpass', **_skip_filter_checks)
    t0_supconj = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)
    period = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    dpdt = orbit_ps.get_parameter(qualifier='dpdt', **_skip_filter_checks)
    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
    dperdt = orbit_ps.get_parameter(qualifier='dperdt', **_skip_filter_checks)
    t0 = b.get_parameter(qualifier='t0', context='system', **_skip_filter_checks)

    if solve_for in [None, t0_perpass]:
        lhs = t0_perpass
        rhs = t0_supconj_to_perpass(t0_supconj, period, ecc, per0, dpdt, dperdt, t0)

    elif solve_for == t0_supconj:
        lhs = t0_supconj
        rhs = t0_perpass_to_supconj(t0_perpass, period, ecc, per0, dpdt, dperdt, t0)



    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

def t0(*args, **kwargs):
    """
    shortcut to <phoebe.parameters.constraint.t0_perpass_supconj> for backwards
    compatibility.
    """
    return t0_perpass_supconj(*args, **kwargs)

_validsolvefor['t0_ref_supconj'] = ['t0_ref', 't0_supconj']
def t0_ref_supconj(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for t0_ref in an orbit - allowing translating between
    t0_ref and t0_supconj.

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('t0_ref_supconj', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        't0_supconj' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 't0_ref', 'period', 'ecc', or 'per0')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    orbit_ps = _get_system_ps(b, orbit)

    # by default both t0s exist in an orbit, so we don't have to worry about creating either
    t0_ref = orbit_ps.get_parameter(qualifier='t0_ref', **_skip_filter_checks)
    t0_supconj = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)
    period = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    dpdt = orbit_ps.get_parameter(qualifier='dpdt', **_skip_filter_checks)
    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
    dperdt = orbit_ps.get_parameter(qualifier='dperdt', **_skip_filter_checks)
    t0 = b.get_parameter(qualifier='t0', context='system', **_skip_filter_checks)

    if solve_for in [None, t0_ref]:
        lhs = t0_ref
        rhs = t0_supconj_to_ref(t0_supconj, period, ecc, per0, dpdt, dperdt, t0)

    elif solve_for == t0_supconj:
        lhs = t0_supconj
        rhs = t0_ref_to_supconj(t0_ref, period, ecc, per0, dpdt, dperdt, t0)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

_validsolvefor['period_anom'] = ['period', 'period_anom']
def period_anom(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for period_anom in an orbit - allowing translating between
    period (sidereal) and period_anom (anomalistic).

    This constraint uses the following linear approximation:

    `period_sidereal = period_anomalistic * (1 - period_sidereal * dperdt/(2pi))`

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('period_anom', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'period_anom' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'period')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    orbit_ps = _get_system_ps(b, orbit)

    period_sid = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    period_anom = orbit_ps.get_parameter(qualifier='period_anom', **_skip_filter_checks)
    dperdt = orbit_ps.get_parameter(qualifier='dperdt', **_skip_filter_checks)

    if solve_for in [None, period_anom]:
        lhs = period_anom
        # rhs = period_sidereal_to_anomalistic(period_sidereal, dperdt)

        # period_sidereal = period_anomalistic * (1 - period_sidereal * dperdt/(2pi))
        # solving for period_anomalistic gives us:
        rhs = period_sid / (-1*period_sid * dperdt/(2*np.pi*u.rad) + 1*u.dimensionless_unscaled)
    elif solve_for in [period_sid]:
        lhs = period_sid
        # rhs = period_anomalistic_to_sidereal(period, dperdt)

        # period_sidereal = period_anomalistic * (1 - period_sidereal * dperdt/(2pi))
        # solving for period_sidereal gives us:
        rhs = period_anom / (period_anom * dperdt/(2*np.pi*u.rad) + 1*u.dimensionless_unscaled)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

_validsolvefor['mean_anom'] = ['mean_anom', 't0_perpass']
def mean_anom(b, orbit, solve_for=None, **kwargs):
    """

    This constraint is automatically included for all orbits, during
    <phoebe.frontend.bundle.Bundle.add_component> for a
    <phoebe.parameters.component.orbit>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('mean_anom', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.

     **NOTE**: this constraint does not account for any time derivatives in
     orbital elements (dpdt, dperdt, etc).

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'mean_anom' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 't0_perpass', 'period', or 't0')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    orbit_ps = _get_system_ps(b, orbit)

    mean_anom = orbit_ps.get_parameter(qualifier='mean_anom', **_skip_filter_checks)
    t0_perpass = orbit_ps.get_parameter(qualifier='t0_perpass', **_skip_filter_checks)
    period = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    dpdt = orbit_ps.get_parameter(qualifier='dpdt', **_skip_filter_checks)
    t0 = b.get_parameter(qualifier='t0', context='system', **_skip_filter_checks)

    if solve_for in [None, mean_anom]:
        lhs = mean_anom
        rhs = 2 * np.pi * (t0 - t0_perpass) / period
    elif solve_for in [t0_perpass]:
        lhs = t0_perpass
        rhs = t0 - (mean_anom*period)/(2*np.pi*u.rad)
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

def _true_anom_to_phase(true_anom, period, ecc, per0):
    """
    TODO: add documentation
    """
    phshift = 0

    mean_anom = true_anom - (ecc*sin(true_anom))*u.deg

    Phi = (mean_anom + per0) / (360*u.deg) - 1./4

    # phase = Phi - (phshift - 0.25 + per0/(360*u.deg)) * period
    phase = (Phi*u.d - (phshift - 0.25 + per0/(360*u.deg)) * period)*(u.cycle/u.d)

    return phase

_validsolvefor['ph_supconj'] = ['ph_supconj']
def ph_supconj(b, orbit, solve_for=None, **kwargs):
    """
    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('ph_supconj', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.
    """
    orbit_ps = _get_system_ps(b, orbit)

    # metawargs = orbit_ps.meta
    #metawargs.pop('qualifier')

    # t0_ph0 and phshift both exist by default, so we don't have to worry about creating either
    # t0_ph0 = orbit_ps.get_parameter(qualifier='t0_ph0')
    # phshift = orbit_ps.get_parameter(qualifier='phshift')
    ph_supconj = orbit_ps.get_parameter(qualifier='ph_supconj', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    period = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)

    # true_anom_supconj = pi/2 - per0
    # mean_anom_supconj = true_anom_supconj - ecc*sin(true_anom_supconj)
    # ph_supconj = (mean_anom_supconj + per0) / (2 * pi) - 1/4

    if solve_for in [None, ph_supconj]:
        lhs = ph_supconj

        #true_anom_supconj = np.pi/2*u.rad - per0
        true_anom_supconj = -1*(per0 - 360*u.deg)

        rhs = _true_anom_to_phase(true_anom_supconj, period, ecc, per0)

    #elif solve_for in [per0]:
    #    raise NotImplementedError("phshift constraint does not support solving for per0 yet")
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

_validsolvefor['ph_infconj'] = ['ph_infconj']
def ph_infconj(b, orbit, solve_for=None, **kwargs):
    """
    This constraint is automatically added for binary systems via
    <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('ph_infconj', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.
    """
    orbit_ps = _get_system_ps(b, orbit)

    ph_infconj = orbit_ps.get_parameter(qualifier='ph_infconj', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    period = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)

    if solve_for in [None, ph_infconj]:
        lhs = ph_infconj

        #true_anom_infconj = 3*np.pi/2 - per0
        # true_anom_infconj = (3*90)*u.deg - per0  # TODO: fix math to allow this
        true_anom_infconj = -1*(per0 - (3*90)*u.deg)

        rhs = _true_anom_to_phase(true_anom_infconj, period, ecc, per0)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

_validsolvefor['ph_perpass'] = ['ph_perpass']
def ph_perpass(b, orbit, solve_for=None, **kwargs):
    """
    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('ph_perpass', orbit='binary')`, where `orbit` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits>.
    """
    orbit_ps = _get_system_ps(b, orbit)

    ph_perpass = orbit_ps.get_parameter(qualifier='ph_perpass', **_skip_filter_checks)
    per0 = orbit_ps.get_parameter(qualifier='per0', **_skip_filter_checks)
    ecc = orbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    period = orbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)

    if solve_for in [None, ph_perpass]:
        lhs = ph_perpass

        # true_anom_per0 = (per0 - pi/2) / (2*pi)
        true_anom_per0 = (per0 - 90*u.deg) / (360)

        rhs = _true_anom_to_phase(true_anom_per0, period, ecc, per0)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}




_validsolvefor['freq'] = ['freq', 'period']
def freq(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for frequency (either orbital or rotational) given a period.

    ```
    freq = 2 * pi / period
    ```

    This constraint is automatically included for all <phoebe.parameters.component.star>
    and <phoebe.parameters.component.orbit> components via
    <phoebe.frontend.bundle.Bundle.add_component>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('freq', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_orbits> or
     <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or star in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'freq' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'period').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    component_ps = _get_system_ps(b, component)

    #metawargs = component_ps.meta
    #metawargs.pop('qualifier')

    period = component_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    freq = component_ps.get_parameter(qualifier='freq', **_skip_filter_checks)

    if solve_for in [None, freq]:
        lhs = freq
        rhs = 2 * np.pi / period

    elif solve_for == period:
        lhs = period
        rhs = 2 * np.pi / freq

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

#}
#{ Inter-orbit constraints

def keplers_third_law_hierarchical(b, orbit1, orbit2, solve_for=None, **kwargs):
    """
    TODO: add documentation

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint>.
    """

    hier = b.hierarchy

    orbit1_ps = _get_system_ps(b, orbit1)
    orbit2_ps = _get_system_ps(b, orbit2)

    sma1 = orbit1_ps.get_parameter(qualifier='sma', **_skip_filter_checks)
    sma2 = orbit2_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    q1 = orbit1_ps.get_parameter(qualifier='q', **_skip_filter_checks)
    q2 = orbit2_ps.get_parameter(qualifier='q', **_skip_filter_checks)

    period1 = orbit1_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    period2 = orbit2_ps.get_parameter(qualifier='period', **_skip_filter_checks)

    # NOTE: orbit1 is the outer, so we need to check orbit2... which will
    # be the OPPOSITE component as that of the mass we're solving for
    if hier.get_primary_or_secondary(orbit2_ps.component) == 'primary':
        qthing1 = 1.0+q1
    else:
        qthing1 = 1.0+1./q1

    if solve_for in [None, sma1]:
        lhs = sma1
        rhs = (sma2**3 * qthing1 * period1**2/period2**2)**"(1./3)"
    else:
        # TODO: add other options to solve_for
        raise NotImplementedError

    return lhs, rhs, [], {'orbit1': orbit1, 'orbit2': orbit2}

#}
#{ Intra-component constraints

_validsolvefor['irrad_frac'] = ['irrad_frac_lost_bol', 'irrad_frac_refl_bol']
def irrad_frac(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to ensure that energy is conserved and all incident
    light is accounted for.

    This constraint is automatically included for all
    <phoebe.parameters.component.star> during
    <phoebe.frontend.bundle.Bundle.add_component>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('irrad_frac', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'irrad_frac_lost_bol' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'irrad_frac_refl_bol').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    comp_ps = b.get_component(component=component)

    irrad_frac_refl_bol = comp_ps.get_parameter(qualifier='irrad_frac_refl_bol', **_skip_filter_checks)
    irrad_frac_lost_bol = comp_ps.get_parameter(qualifier='irrad_frac_lost_bol', **_skip_filter_checks)

    if solve_for in [irrad_frac_lost_bol, None]:
        lhs = irrad_frac_lost_bol
        rhs = 1.0 - irrad_frac_refl_bol
    elif solve_for in [irrad_frac_refl_bol]:
        lhs = irrad_frac_refl_bol
        rhs = 1.0 - irrad_frac_lost_bol
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['semidetached'] = ['requiv']
def semidetached(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to force requiv to be semidetached.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('semidetached', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requiv' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'requiv_max').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    comp_ps = b.get_component(component=component, **_skip_filter_checks)

    requiv = comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    requiv_critical = comp_ps.get_parameter(qualifier='requiv_max', **_skip_filter_checks)

    if solve_for in [requiv, None]:
        lhs = requiv
        rhs = 1.0*requiv_critical
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['logg'] = ['logg', 'requiv', 'mass']
def logg(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for logg at requiv for a star.

    Note that the constant includes G in solar units and then a conversion
    factor from solar to cgs.

    This constraint is automatically included for all
    <phoebe.parameters.component.star> during
    <phoebe.frontend.bundle.Bundle.add_component>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('logg', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'logg' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'mass', 'requiv').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    comp_ps = b.get_component(component=component, **_skip_filter_checks)

    requiv = comp_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    mass = comp_ps.get_parameter(qualifier='mass', **_skip_filter_checks)

    metawargs = comp_ps.meta
    metawargs.pop('qualifier')
    logg_def = FloatParameter(qualifier='logg', latexfmt=r'\mathrm{log}g_\mathrm{{ {component} }}', value=1.0, default_unit=u.dimensionless_unscaled, description='logg at requiv')
    logg, created = b.get_or_create('logg', logg_def, **metawargs)

    # logg needs to be in cgs, but we'll handle all quantities in solar
    G = c.G.to('solRad3/(solMass d2)').value
    solar_to_cgs = (1*u.solRad/u.d**2).to(u.cm/u.s**2).value


    if solve_for in [logg, None]:
        lhs = logg
        rhs = log10(mass / requiv**2 * G * solar_to_cgs)
    elif solve_for in [requiv]:
        lhs = requiv
        rhs = sqrt((mass*G * solar_to_cgs)/10**logg)
    elif solve_for in [mass]:
        lhs = mass
        rhs = requiv**2 * 10**logg / ( G * solar_to_cgs)
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

# def vsini(b, component, solve_for=None, **kwargs):
#     """
#     Create a constraint for vsini at requiv for a star.
#
#     See also:
#     * <phoebe.parameters.constraint.vrot>
#
#     This is usually passed as an argument to
#      <phoebe.frontend.bundle.Bundle.add_constraint>.
#
#     Arguments
#     -----------
#     * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
#     * `component` (string): the label of the component in which this
#         constraint should be built.
#     * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
#         'vsini' should not be the derived/constrained parameter, provide which
#         other parameter should be derived (ie 'incl', 'freq', 'requiv').
#
#     Returns
#     ----------
#     * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
#         lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
#         that were passed to this function)
#
#     Raises
#     --------
#     * NotImplementedError: if the value of `solve_for` is not implemented.
#     """
#     comp_ps = b.get_component(component=component)
#
#     requiv = comp_ps.get_parameter(qualifier='requiv')
#     freq = comp_ps.get_parameter(qualifier='freq')
#     incl = comp_ps.get_parameter(qualifier='incl')
#
#     metawargs = comp_ps.meta
#     metawargs.pop('qualifier')
#     vsini_def = FloatParameter(qualifier='vsini', value=1.0, default_unit=u.km/u.s, description='vsini at requiv')
#     vsini, created = b.get_or_create('vsini', vsini_def, **metawargs)
#
#
#     if solve_for in [vsini, None]:
#         lhs = vsini
#         rhs = requiv * freq / sin(incl)
#     elif solve_for in [freq]:
#         # will likely need to flip freq constraint for period first
#         lhs = freq
#         rhs = vsini / (requiv * sin(incl))
#     else:
#         raise NotImplementedError
#
#     return lhs, rhs, [], {'component': component}

# def vrot(b, component, solve_for=None, **kwargs):
#     """
#     Create a constraint for vrot at requiv for a star.
#
#     See also:
#     * <phoebe.parameters.constraint.vsini>
#
#     This is usually passed as an argument to
#      <phoebe.frontend.bundle.Bundle.add_constraint>.
#
#     Arguments
#     -----------
#     * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
#     * `component` (string): the label of the component in which this
#         constraint should be built.
#     * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
#         'vrot' should not be the derived/constrained parameter, provide which
#         other parameter should be derived (ie 'freq', 'requiv').
#
#     Returns
#     ----------
#     * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
#         lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
#         that were passed to this function)
#
#     Raises
#     --------
#     * NotImplementedError: if the value of `solve_for` is not implemented.
#     """
#     comp_ps = b.get_component(component=component)
#
#     requiv = comp_ps.get_parameter(qualifier='requiv')
#     freq = comp_ps.get_parameter(qualifier='freq')
#
#     metawargs = comp_ps.meta
#     metawargs.pop('qualifier')
#     vrot_def = FloatParameter(qualifier='vrot', value=1.0, default_unit=u.km/u.s, description='vrot at requiv')
#     vrot, created = b.get_or_create('vrot', vrot_def, **metawargs)
#
#
#     if solve_for in [vrot, None]:
#         lhs = vrot
#         rhs = requiv * freq
#     elif solve_for in [freq]:
#         # will likely need to flip freq constraint for period first
#         lhs = freq
#         rhs = vrot / requiv
#     else:
#         raise NotImplementedError
#
#     return lhs, rhs, [], {'component': component}

#}
#{ Inter-component constraints

_validsolvefor['teffratio'] = ['teffratio', 'teff@hier.children_of(orbit)[0]', 'teff@hier.children_of(orbit)[1]']
def teffratio(b, orbit=None, solve_for=None, **kwargs):
    """
    Create a constraint to for the teff ratio between two stars in the same orbit.
    Defined as teffratio = teff@comp2 / teff@comp1, where comp1 and comp2 are
    determined from the primary and secondary components of the orbit `orbit`.

    This is usually passed as an argument to
    <phoebe.frontend.bundle.Bundle.add_constraint> as
    `b.add_constraint('teffratio', orbit='binary')`, where
    `orbit` is one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (phoebe.frontend.bundle.Bundle): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should be built.
        Optional if only one orbit exists in the hierarchy.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'teffratio' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'teff@...').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    -------------
    * ValueError: if `orbit` is not provided, but more than one orbit exists
        in the hierarchy.
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    # TODO: do we need to rebuild this if the hierarchy changes???
    hier = b.hierarchy

    if orbit is None:
        orbits = hier.get_orbits()
        if len(orbits)==1:
            orbit = orbits[0]
        else:
            raise ValueError("must provide orbit since more than one orbit present in the hierarchy")

    comp1, comp2 = hier.get_stars_of_children_of(orbit)

    comp1_ps = b.get_component(component=comp1, **_skip_filter_checks)
    comp2_ps = b.get_component(component=comp2, **_skip_filter_checks)

    teffratio_def = FloatParameter(qualifier='teffratio',  latexfmt=r'T_\mathrm{{ eff, {children1} }} / T_\mathrm{{ eff, {children0} }}', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='ratio between effective temperatures of children stars')
    teffratio, created = b.get_or_create('teffratio', teffratio_def, kind='orbit', component=orbit, context='component')

    teff1 = comp1_ps.get_parameter(qualifier='teff', **_skip_filter_checks)
    teff2 = comp2_ps.get_parameter(qualifier='teff', **_skip_filter_checks)

    if solve_for in [teffratio, None]:
        lhs = teffratio
        rhs = teff2/teff1
    elif solve_for in [teff1]:
        lhs = teff1
        rhs = teff2 / teffratio
    elif solve_for in [teff2]:
        lhs = teff2
        rhs = teffratio * teff1
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}


_validsolvefor['requivratio'] = ['requivratio', 'requiv@hier.children_of(orbit)[0]', 'requiv@hier.children_of(orbit)[1]']
def requivratio(b, orbit=None, solve_for=None, **kwargs):
    """
    Create a constraint to for the requiv ratio between two stars in the same orbit.
    Defined as requivratio = requiv@comp2 / requiv@comp1, where comp1 and comp2 are
    determined from the primary and secondary components of the orbit `orbit`.

    This is usually passed as an argument to
    <phoebe.frontend.bundle.Bundle.add_constraint> as
    `b.add_constraint('requivratio', orbit='binary')`, where
    `orbit` is one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (phoebe.frontend.bundle.Bundle): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should be built.
        Optional if only one orbit exists in the hierarchy.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requivratio' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'requiv@...').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    -------------
    * ValueError: if `orbit` is not provided, but more than one orbit exists
        in the hierarchy.
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    # TODO: do we need to rebuild this if the hierarchy changes???
    hier = b.hierarchy

    if orbit is None:
        orbits = hier.get_orbits()
        if len(orbits)==1:
            orbit = orbits[0]
        else:
            raise ValueError("must provide orbit since more than one orbit present in the hierarchy")

    comp1, comp2 = hier.get_stars_of_children_of(orbit)

    orbit_ps = b.get_component(component=orbit, **_skip_filter_checks)
    comp1_ps = b.get_component(component=comp1, **_skip_filter_checks)
    comp2_ps = b.get_component(component=comp2, **_skip_filter_checks)

    requiv1 = comp1_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    requiv2 = comp2_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)

    sma = orbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    requivratio_def = FloatParameter(qualifier='requivratio', latexfmt=r'R_\mathrm{{ equiv, {children1} }} / R_\mathrm{{ equiv, {children0} }}', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='ratio between equivalent radii of children stars')
    requivratio, requivratio_created = b.get_or_create('requivratio', requivratio_def, kind='orbit', component=orbit, context='component')

    requivsumfrac_exists = 'requivsumfrac' in orbit_ps.qualifiers
    if requivsumfrac_exists:
        requivsumfrac = orbit_ps.get_parameter(qualifier='requivsumfrac', **_skip_filter_checks)
        requivsumfrac_constrained = kwargs.get('requivsumfrac_constrained', len(requivsumfrac.constrained_by) > 0)
        params = [requivratio, requivsumfrac, requiv1, requiv2, sma]

        if requivsumfrac.is_constraint is not None and requivratio not in requivsumfrac.is_constraint.addl_vars:
            requivsumfrac.is_constraint._addl_vars.append(ConstraintVar(b, requivratio.twig))

    else:
        requivsumfrac = None
        requivsumfrac_constrained = True
        params = [requivratio, requiv1, requiv2, sma]

    if solve_for in [requivratio, None]:
        lhs = requivratio
        rhs = requiv2/requiv1
        if not requivsumfrac_constrained:
            if requiv1.is_constraint:
                requiv1.is_constraint.constraint_kwargs['requivratio_constrained'] = True
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)
            elif requiv2.is_constraint:
                requiv2.is_constraint.constraint_kwargs['requivratio_constrained'] = True
                requiv2.is_constraint.flip_for('requiv@'.format(requiv2.component), force=True)

    elif solve_for in [requiv1]:
        lhs = requiv1
        if requivsumfrac_constrained:
            rhs = requiv2 / requivratio
        else:
            rhs = (requivsumfrac * sma) / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv2.is_constraint and 'requivratio_constrained' not in requiv2.is_constraint.constraint_kwargs.keys():
                requiv2.is_constraint.constraint_kwargs['requivratio_constrained'] = False
                requiv2.is_constraint.flip_for('requiv@{}'.format(requiv2.component), force=True)

    elif solve_for in [requiv2]:
        lhs = requiv2
        if requivsumfrac_constrained:
            rhs = requivratio * requiv1
        else:
            rhs = (requivratio * requivsumfrac * sma) / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv1.is_constraint and 'requivratio_constrained' not in requiv1.is_constraint.constraint_kwargs.keys():
                requiv1.is_constraint.constraint_kwargs['requivratio_constrained'] = False
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)
    elif solve_for == requivsumfrac:
        raise NotImplementedError("cannot solve this constraint for 'requivsumfrac' since it was originally 'requivratio'")
    else:
        raise NotImplementedError

    return lhs, rhs, params, {'orbit': orbit}

_validsolvefor['requivsumfrac'] = ['requivsumfrac', 'sma', 'requiv@hier.children_of(orbit)[0]', 'requiv@hier.children_of(orbit)[1]']
def requivsumfrac(b, orbit=None, solve_for=None, **kwargs):
    """
    Create a constraint to for the requiv sum of two stars in the same orbit
    normalized to the semi major axis.
    Defined as requivsumfrac = (requiv@comp2 + requiv@comp1)/sma, where comp1 and comp2 are
    determined from the primary and secondary components of the orbit `orbit`.

    This is usually passed as an argument to
    <phoebe.frontend.bundle.Bundle.add_constraint> as
    `b.add_constraint('requivsumfrac', orbit='binary')`, where
    `orbit` is one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (phoebe.frontend.bundle.Bundle): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should be built.
        Optional if only one orbit exists in the hierarchy.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requivsumfrac' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'requiv@...').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    -------------
    * ValueError: if `orbit` is not provided, but more than one orbit exists
        in the hierarchy.
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    # TODO: do we need to rebuild this if the hierarchy changes???
    hier = b.hierarchy

    if orbit is None:
        orbits = hier.get_orbits()
        if len(orbits)==1:
            orbit = orbits[0]
        else:
            raise ValueError("must provide orbit since more than one orbit present in the hierarchy")

    comp1, comp2 = hier.get_stars_of_children_of(orbit)

    orbit_ps = b.get_component(component=orbit, **_skip_filter_checks)
    comp1_ps = b.get_component(component=comp1, **_skip_filter_checks)
    comp2_ps = b.get_component(component=comp2, **_skip_filter_checks)

    requiv1 = comp1_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    requiv2 = comp2_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    sma = orbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    requivsumfrac_def = FloatParameter(qualifier='requivsumfrac', latexfmt=r'(R_\mathrm{{ equiv, {children0} }} + R_\mathrm{{ equiv, {children1} }}) / a_\mathrm{{ {component} }}', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='sum of fractional equivalent radii of children stars')
    requivsumfrac, requivsumfrac_created = b.get_or_create('requivsumfrac', requivsumfrac_def, kind='orbit', component=orbit, context='component')

    requivratio_exists = 'requivratio' in orbit_ps.qualifiers
    if requivratio_exists:
        requivratio = orbit_ps.get_parameter(qualifier='requivratio', **_skip_filter_checks)
        requivratio_constrained = kwargs.get('requivratio_constrained', len(requivratio.constrained_by) > 0)
        params = [requivratio, requivsumfrac, requiv1, requiv2, sma]

        if requivratio.is_constraint is not None and requivsumfrac not in requivratio.is_constraint.addl_vars:
            requivratio.is_constraint._addl_vars.append(ConstraintVar(b, requivsumfrac.twig))
    else:
        requivratio = None
        requivratio_constrained = True
        params = [requivsumfrac, requiv1, requiv2, sma]


    if solve_for in [requivsumfrac, None]:
        lhs = requivsumfrac
        rhs = (requiv1 + requiv2)/sma
        if requivratio_exists and not requivratio_constrained:
            if requiv1.is_constraint:
                requiv1.is_constraint.constraint_kwargs['requivsumfrac_constrained'] = True
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)
            elif requiv2.is_constraint:
                requiv2.is_constraint.constraint_kwargs['requivsumfrac_constrained'] = True
                requiv2.is_constraint.flip_for('requiv@'.format(requiv2.component), force=True)

    elif solve_for in [sma]:
        lhs = sma
        rhs = (requiv1 + requiv2) / requivsumfrac

    elif solve_for in [requiv1]:
        lhs = requiv1
        if requivratio_constrained:
            rhs = requivsumfrac * sma - requiv2
        else:
            rhs = (requivsumfrac * sma) / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv2.is_constraint and 'requivsum_constrained' not in requiv2.is_constraint.constraint_kwargs.keys():
                requiv2.is_constraint.constraint_kwargs['requivsumfrac_constrained'] = False
                requiv2.is_constraint.flip_for('requiv@{}'.format(requiv2.component), force=True)

    elif solve_for in [requiv2]:
        lhs = requiv2
        if requivratio_constrained:
            rhs = requivsumfrac * sma - requiv1
        else:
            rhs = (requivratio * requivsumfrac * sma) / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv1.is_constraint and 'requivsumfrac_constrained' not in requiv1.is_constraint.constraint_kwargs.keys():
                requiv1.is_constraint.constraint_kwargs['requivsumfrac_constrained'] = False
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)

    elif solve_for == requivratio:
        raise NotImplementedError("cannot solve this constraint for 'requivratio' since it was originally 'requivsum'")
    else:
        raise NotImplementedError


    return lhs, rhs, params, {'orbit': orbit}

#}
#{ Orbit-component constraints

_validsolvefor['mass'] = ['mass', 'sma', 'period', 'q']
def mass(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the mass of a star based on Kepler's third
    law from its parent orbit.

    This constraint is automatically created and attached for all stars
    in binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('mass', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'mass' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'period', 'sma', 'q').  Note:
        you cannot solve_for 'period' and 'sma' in the same orbit as the solution
        will not be unique.

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    # TODO: optimize this - this is currently by far the most expensive constraint (due mostly to the parameter multiplication)

    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for mass requires hierarchy")

    component_ps = _get_system_ps(b, component)

    sibling = hier.get_sibling_of(component)
    sibling_ps = _get_system_ps(b, sibling)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    mass = component_ps.get_parameter(qualifier='mass', **_skip_filter_checks)
    mass_sibling = sibling_ps.get_parameter(qualifier='mass', **_skip_filter_checks)

    # we need to find the constraint attached to the other component... but we
    # don't know who is constrained, or whether it belongs to the sibling or parent
    # orbit, so we'll have to do a bit of digging.
    mass_constraint_sibling = None
    for p in b.filter(constraint_func='mass', component=[parentorbit, sibling], context='constraint', **_skip_filter_checks).to_list():
        if p.constraint_kwargs['component'] == sibling:
            mass_constraint_sibling = p
            break
    if mass_constraint_sibling is not None:
        sibling_solve_for = mass_constraint_sibling.qualifier
        logger.debug("constraint.mass for component='{}': sibling ('{}') is solved for '{}'".format(component, sibling, sibling_solve_for))
    else:
        # this could happen when we build the first constraint, before the second has been built
        sibling_solve_for = None

    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)
    # NOTE: sidereal period
    period = parentorbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)

    G = c.G.to('solRad3 / (solMass d2)')
    G.keep_in_solar_units = True

    if hier.get_primary_or_secondary(component) == 'primary':
        qthing = 1.0+q
    else:
        qthing = 1.0+1./q

    if solve_for in [None, mass]:
        lhs = mass
        rhs = (4*np.pi**2 * sma**3 ) / (period**2 * qthing * G)

    elif solve_for==sma:
        if sibling_solve_for in ['period', 'sma']:
            raise ValueError("cannot solve for '{}' when sibling ('{}') is solved for '{}'".format(solve_for.twig, sibling, sibling_solve_for))
        lhs = sma
        rhs = ((mass * period**2 * qthing * G)/(4 * np.pi**2))**"(1./3)"

    elif solve_for==period:
        if sibling_solve_for in ['period', 'sma']:
            raise ValueError("cannot solve for '{}' when sibling ('{}') is solved for '{}'".format(solve_for.twig, sibling, sibling_solve_for))
        lhs = period
        rhs = ((4 * np.pi**2 * sma**3)/(mass * qthing * G))**"(1./2)"

    elif solve_for==q:
        lhs = q

        if hier.get_primary_or_secondary(component) == 'primary':
            rhs = mass_sibling / mass
        else:
            rhs = mass / mass_sibling

        # qthing = (4*np.pi**2 * sma**3 ) / (period**2 * mass * G)
        # if hier.get_primary_or_secondary(component) == 'primary':
        #     rhs = qthing - 1.0
        # else:
        #     rhs = 1 / (qthing - 1.0)

    else:
        raise NotImplementedError

    return lhs, rhs, [mass_sibling, period, sma, q], {'component': component}


    # ecosw_def = FloatParameter(qualifier='ecosw', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times cos of argument of periastron')
    # ecosw, ecosw_created = b.get_or_create('ecosw', ecosw_def, **metawargs)
    #
    # ecosw_constrained = kwargs.get('ecosw_constrained', len(ecosw.constrained_by) > 0)
    # logger.debug("esinw constraint: solve_for={}, ecosw_constrained={}, ecosw_created={}".format(solve_for.qualifier if solve_for is not None else "None", ecosw_constrained, ecosw_created))
    #
    # ecc = b.get_parameter(qualifier='ecc', **metawargs)
    # per0 = b.get_parameter(qualifier='per0', **metawargs)
    #
    # if solve_for in [None, esinw]:
    #     lhs = esinw
    #     rhs = ecc * sin(per0)
    #     if not ecosw_created and not ecosw_constrained:
    #         if per0.is_constraint:
    #             per0.is_constraint.constraint_kwargs['esinw_constrained'] = True
    #             per0.is_constraint.flip_for('per0', force=True)
    #         elif ecc.is_constraint:
    #             ecc.is_constraint.constraint_kwargs['esinw_constrained'] = True
    #             ecc.is_constraint.flip_for('ecc', force=True)
    #
    # elif solve_for == ecc:
    #     lhs = ecc
    #     if ecosw_constrained:
    #         rhs = esinw / sin(per0)
    #     else:
    #         rhs = (esinw**2 + ecosw**2)**0.5
    #         # the other constraint needs to also follow the alternate equations
    #         if per0.is_constraint and 'esinw_constrained' not in per0.is_constraint.constraint_kwargs.keys():
    #             logger.debug("esinw constraint: attempting to also flip per0 constraint")
    #             per0.is_constraint.constraint_kwargs['esinw_constrained'] = False
    #             per0.is_constraint.flip_for('per0', force=True)
    #
    # elif solve_for == per0:
    #     lhs = per0
    #     if ecosw_constrained:
    #         # cannot just do arcsin because ecc may be zero
    #         rhs = esinw2per0(ecc, esinw)
    #     else:
    #         rhs = arctan2(esinw, ecosw)
    #         # the other constraint needs to also follow the alternate equations
    #         if ecc.is_constraint and 'esinw_constrained' not in ecc.is_constraint.constraint_kwargs.keys():
    #             logger.debug("esinw constraint: attempting to also flip ecc constraint")
    #             ecc.is_constraint.constraint_kwargs['esinw_constrained'] = False
    #             ecc.is_constraint.flip_for('ecc', force=True)
    # elif solve_for == ecosw:
    #     raise NotImplementedError("cannot solve this constraint for 'ecosw' since it was originally 'esinw'")
    # else:
    #     raise NotImplementedError
    #
    # return lhs, rhs, [esinw, ecosw, ecc, per0], {'orbit': orbit}

_validsolvefor['comp_sma'] = ['sma@orbit', 'incl@orbit']
def comp_sma(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the star's semi-major axes WITHIN its
    parent orbit.  This is NOT the same as the semi-major axes OF
    the parent orbit

    This constraint is automatically created and attached for all stars
    in binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('comp_sma', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    If 'sma' does not exist in the component, it will be created

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'sma@star' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q', 'sma@orbit').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for comp_sma requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    metawargs = component_ps.meta
    metawargs.pop('qualifier')
    compsma_def = FloatParameter(qualifier='sma', latexfmt=r'a_\mathrm{{ {component} }}', value=4.0, default_unit=u.solRad, advanced=True, description='Semi major axis of the component in the orbit')
    compsma, created = b.get_or_create('sma', compsma_def, **metawargs)

    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)

    # NOTE: similar logic is also in dynamics.keplerian.dynamics_from_bundle to
    # handle nested hierarchical orbits.  If changing any of the logic here,
    # it should be changed there as well.

    if hier.get_primary_or_secondary(component) == 'primary':
        qthing = (1. + 1./q)
    else:
        qthing = (1. + q)


    if solve_for in [None, compsma]:
        lhs = compsma
        rhs = sma / qthing

    elif solve_for == sma:
        lhs = sma
        rhs = compsma * qthing

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['comp_asini'] = ['asini@star', 'sma@orbit']
def comp_asini(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the star's projected semi-major axes WITHIN its
    parent orbit.

    This constraint is automatically created and attached for all stars
    in binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('comp_asini', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    If 'asini' does not exist in the component, it will be created

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'asini@star' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'sma@orbit').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for comp_sma requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    metawargs = component_ps.meta
    metawargs.pop('qualifier')
    compasini_def = FloatParameter(qualifier='asini', latexfmt=r'a_\mathrm{{ {component} }} \sin i_\mathrm{{ {parent} }}',  value=4.0, default_unit=u.solRad, advanced=True, description='Projected semi major axis of the component in the orbit')
    compasini, created = b.get_or_create('asini', compasini_def, **metawargs)

    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)
    incl = parentorbit_ps.get_parameter(qualifier='incl', **_skip_filter_checks)

    # NOTE: similar logic is also in dynamics.keplerian.dynamics_from_bundle to
    # handle nested hierarchical orbits.  If changing any of the logic here,
    # it should be changed there as well.

    if hier.get_primary_or_secondary(component) == 'primary':
        qthing = (1. + 1./q)
    else:
        qthing = (1. + q)


    if solve_for in [None, compasini]:
        lhs = compasini
        rhs = sma * sin(incl) / qthing

    elif solve_for == sma:
        lhs = sma
        rhs = compasini / sin(incl) * qthing

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['requivfrac'] = ['requivfrac@star', 'requiv@star', 'sma@orbit']
def requivfrac(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the star's fractional equivalent radius.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('requivfrac', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    If 'requivfrac' does not exist in the component, it will be created

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requivfrac@star' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'requiv@star' 'sma@orbit').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for comp_sma requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    metawargs = component_ps.meta
    metawargs.pop('qualifier')
    requivfrac_def = FloatParameter(qualifier='requivfrac', latexfmt=r'R_\mathrm{{ {component} }} / a_\mathrm{{ {parent} }}', value=1.0, default_unit=u.solRad, advanced=True, description='Fractional equivalent radius')
    requivfrac, created = b.get_or_create('requivfrac', requivfrac_def, **metawargs)

    requiv = component_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    if solve_for in [None, requivfrac]:
        lhs = requivfrac
        rhs = requiv / sma

    elif solve_for == requiv:
        lhs = requiv
        rhs = requivfrac * sma

    elif solve_for == sma:
        lhs = sma
        rhs = requiv / requivfrac

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['requiv_detached_max'] = ['requiv_max']
def requiv_detached_max(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L1) value of
    requiv.

    This constraint is automatically created and attached for all stars
    in detached binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('requiv_detached_max', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requiv_max' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q', 'syncpar', 'ecc', 'sma'
        'incl@star', 'long_an@star', 'incl@orbit', 'long_an@orbit').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_detached_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)


    if parentorbit == 'component':
        raise ValueError("cannot constrain requiv_detached_max for single star")

    parentorbit_ps = _get_system_ps(b, parentorbit)

    requiv_max = component_ps.get_parameter(qualifier='requiv_max', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)
    syncpar = component_ps.get_parameter(qualifier='syncpar', **_skip_filter_checks)
    ecc = parentorbit_ps.get_parameter(qualifier='ecc', **_skip_filter_checks)
    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)
    incl_star = component_ps.get_parameter(qualifier='incl', **_skip_filter_checks)
    long_an_star = component_ps.get_parameter(qualifier='long_an', **_skip_filter_checks)
    incl_orbit = parentorbit_ps.get_parameter(qualifier='incl', **_skip_filter_checks)
    long_an_orbit = parentorbit_ps.get_parameter(qualifier='long_an', **_skip_filter_checks)

    if solve_for in [None, requiv_max]:
        lhs = requiv_max

        rhs = roche_requiv_L1(q, syncpar, ecc, sma,
                              incl_star, long_an_star,
                              incl_orbit, long_an_orbit,
                              hier.get_primary_or_secondary(component, return_ind=True))
    else:
        raise NotImplementedError("requiv_detached_max can only be solved for requiv_max")

    return lhs, rhs, [], {'component': component}

_validsolvefor['potential_contact_min'] = ['pot_min']
def potential_contact_min(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L23) value of
    potential at which a constact will underflow.  This will only be used
    for contacts for pot_min.

    This constraint is automatically created and attached for all stars
    in contact binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('potential_contact_min', component='common_envelope')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_envelopes>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'pot_min' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_min requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot_min = component_ps.get_parameter(qualifier='pot_min', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)

    if solve_for in [None, pot_min]:
        lhs = pot_min

        rhs = roche_potential_contact_L23(q)
    else:
        raise NotImplementedError("potential_contact_min can only be solved for requiv_min")

    return lhs, rhs, [], {'component': component}

_validsolvefor['potential_contact_max'] = ['pot_max']
def potential_contact_max(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L1) value of
    potential at which a constact will underflow.  This will only be used
    for contacts for pot_min.

    This constraint is automatically created and attached for all stars
    in contact binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('potential_contact_min', component='common_envelope')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_envelopes>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'pot_max' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot_max = component_ps.get_parameter(qualifier='pot_max', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)

    if solve_for in [None, pot_max]:
        lhs = pot_max

        rhs = roche_potential_contact_L1(q)
    else:
        raise NotImplementedError("potential_contact_max can only be solved for requiv_max")

    return lhs, rhs, [], {'component': component}

_validsolvefor['requiv_contact_min'] = ['requiv_min']
def requiv_contact_min(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L1) value of
    requiv at which a constact will underflow.  This will only be used
    for contacts for requiv_min.

    This constraint is automatically created and attached for all stars
    in contact binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('requiv_contact_min', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requiv_min' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q', 'sma').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_min requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    requiv_min = component_ps.get_parameter(qualifier='requiv_min', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)
    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    if solve_for in [None, requiv_min]:
        lhs = requiv_min

        rhs = roche_requiv_contact_L1(q, sma, hier.get_primary_or_secondary(component, return_ind=True))
    else:
        raise NotImplementedError("requiv_contact_min can only be solved for requiv_min")

    return lhs, rhs, [], {'component': component}

_validsolvefor['requiv_contact_max'] = ['requiv_max']
def requiv_contact_max(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L2/3) value of
    requiv at which a constact will overflow.  This will only be used
    for contacts for requiv_max.

    This constraint is automatically created and attached for all stars
    in contact binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('requiv_contact_min', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requiv_max' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q', 'sma').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    requiv_max = component_ps.get_parameter(qualifier='requiv_max', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)
    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    if solve_for in [None, requiv_max]:
        lhs = requiv_max

        rhs = roche_requiv_contact_L23(q, sma, hier.get_primary_or_secondary(component, return_ind=True))
    else:
        raise NotImplementedError("requiv_contact_max can only be solved for requiv_max")

    return lhs, rhs, [], {'component': component}

_validsolvefor['fillout_factor'] = ['fillout_factor', 'pot']
def fillout_factor(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the fillout factor of a contact envelope.

    This constraint is automatically created and attached for all stars
    in contact binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('potential_contact_min', component='common_envelope')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_envelopes>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'fillout_factor' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'pot', 'q').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot = component_ps.get_parameter(qualifier='pot', **_skip_filter_checks)
    fillout_factor = component_ps.get_parameter(qualifier='fillout_factor', **_skip_filter_checks)
    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)

    if solve_for in [None, fillout_factor]:
        lhs = fillout_factor

        rhs = roche_pot_to_fillout_factor(q, pot)
    elif solve_for in [pot]:
        lhs = pot

        rhs = roche_fillout_factor_to_pot(q, fillout_factor)
    else:
        raise NotImplementedError("fillout_factor can not be solved for {}".format(solve_for))

    return lhs, rhs, [], {'component': component}

_validsolvefor['rotation_period'] = ['period@star', 'syncpar', 'period@orbit']
def rotation_period(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the rotation period of a star given its orbital
    period and synchronicity parameters.

    This constraint is automatically created and attached for all stars
    in detached binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('rotation_period', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'period@star' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'period@orbit', 'syncpar').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for comp_sma requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    period_star = component_ps.get_parameter(qualifier='period', **_skip_filter_checks)
    syncpar_star = component_ps.get_parameter(qualifier='syncpar', **_skip_filter_checks)

    # NOTE: sidereal period
    period_orbit = parentorbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)

    if solve_for in [None, period_star]:
        lhs = period_star
        rhs = period_orbit / syncpar_star

    elif solve_for == syncpar_star:
        lhs = syncpar_star
        rhs = period_orbit / period_star

    elif solve_for == period_orbit:
        lhs = period_orbit
        rhs = syncpar_star * period_star

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['pitch'] = ['incl@star', 'incl@orbit', 'pitch']
def pitch(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the inclination of a star relative to its parent orbit.

    This constraint is automatically created and attached for all stars
    in detached binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('pitch', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'pitch' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'incl@star', 'incl@orbit').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for pitch requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    incl_comp = component_ps.get_parameter(qualifier='incl', **_skip_filter_checks)
    pitch_comp = component_ps.get_parameter(qualifier='pitch', **_skip_filter_checks)
    incl_orb = parentorbit_ps.get_parameter(qualifier='incl', **_skip_filter_checks)

    if solve_for in [None, incl_comp]:
        lhs = incl_comp
        rhs = incl_orb + pitch_comp

    elif solve_for == incl_orb:
        lhs = incl_orb
        rhs = incl_comp - pitch_comp

    elif solve_for == pitch_comp:
        lhs = pitch_comp
        rhs = incl_comp - incl_orb

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

_validsolvefor['yaw'] = ['long_an@star', 'long_an@orbit', 'yaw']
def yaw(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the inclination of a star relative to its parent orbit.

    This constraint is automatically created and attached for all stars
    in detached binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('yaw', component='primary')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'yaw' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'long_an@star', 'long_an@orbit').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for yaw requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    long_an_comp = component_ps.get_parameter(qualifier='long_an', **_skip_filter_checks)
    yaw_comp = component_ps.get_parameter(qualifier='yaw', **_skip_filter_checks)
    long_an_orb = parentorbit_ps.get_parameter(qualifier='long_an', **_skip_filter_checks)

    if solve_for in [None, long_an_comp]:
        lhs = long_an_comp
        rhs = long_an_orb + yaw_comp

    elif solve_for == long_an_orb:
        lhs = long_an_orb
        rhs = long_an_comp - yaw_comp

    elif solve_for == yaw_comp:
        lhs = yaw_comp
        rhs = long_an_comp - long_an_orb

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}


#}
#{ Data constraints

def passband_ratio(b, *args, **kwargs):
    """
    ability to constraint pblum ratios (for colors).

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint>.

    :raises NotImplementedError: because it isn't, yet
    """
    raise NotImplementedError

#}
#{ Dataset constraints
_validsolvefor['compute_phases'] = ['compute_phases', 'compute_times']
def compute_phases(b, component, dataset, solve_for=None, **kwargs):
    """
    Create a constraint for the translation between compute_phases and compute_times.

    This constraint is automatically created and attached for all datasets
    via <phoebe.frontend.bundle.Bundle.add_dataset>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('compute_phases', component=b.hierarchy.get_top(), dataset='dataset')`.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which the
        `period` should be found.
    * `dataset` (string): the label of the dataset in which to find the
        `compute_times` and `compute_phases` parameters.
    * `solve_for` (<phoebe.parameters.Parameter, optional, default=None): if
        'compute_phases' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'compute_times').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """
    ds = b.get_dataset(dataset, check_default=False, check_visible=False)
    compute_times = ds.get_parameter(qualifier='compute_times', **_skip_filter_checks)
    compute_phases = ds.get_parameter(qualifier='compute_phases', component=component, **_skip_filter_checks)
    period = b.get_parameter(qualifier='period', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)

    if len(b.hierarchy.get_stars()) == 1:
        # then for the single star case we always use t0@system and have no dpdt
        t0_system = b.get_parameter(qualifier='t0', context='system', **_skip_filter_checks)

        if solve_for in [None, compute_phases]:
            lhs = compute_phases
            rhs = ((compute_times - t0_system) / period) % 1
        elif solve_for in [compute_times]:
            lhs = compute_times
            rhs = compute_phases * period + t0_system
        else:
            raise NotImplementedError


    else:
        try:
            period_anom = b.get_parameter(qualifier='period_anom', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)
        except ValueError:
            # we need to handle the backward compatibility case where period_anom does not yet exit (probably calling this DURING migration)
            if 'period_anom' not in b.qualifiers:
                logger.warning("compute_phases constraint falling back on period (sidereal)")
                period_anom = b.get_parameter(qualifier='period', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)
            else:
                raise

        phases_period = ds.get_parameter(qualifier='phases_period', component=component, **_skip_filter_checks)
        phases_dpdt = ds.get_parameter(qualifier='phases_dpdt', component=component, **_skip_filter_checks)
        phases_t0 = ds.get_parameter(qualifier='phases_t0', component=component, **_skip_filter_checks)
        t0_supconj = b.get_parameter(qualifier='t0_supconj', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)
        t0_perpass = b.get_parameter(qualifier='t0_perpass', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)
        t0_ref = b.get_parameter(qualifier='t0_ref', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)
        dpdt = b.get_parameter(qualifier='dpdt', component=component if component!='_default' else b.hierarchy.get_top(), context='component', **_skip_filter_checks)

        if solve_for in [None, compute_phases]:
            lhs = compute_phases
            rhs = _times_to_phases(compute_times, phases_period, period, period_anom, phases_dpdt, dpdt, phases_t0, t0_supconj, t0_perpass, t0_ref)
        elif solve_for in [compute_times]:
            lhs = compute_times
            rhs = _phases_to_times(compute_phases, phases_period, period, period_anom, phases_dpdt, dpdt, phases_t0, t0_supconj, t0_perpass, t0_ref)
        else:
            raise NotImplementedError

    return lhs, rhs, [], {'component': component, 'dataset': dataset}

# System constraints

_validsolvefor['extinction'] = ['ebv', 'Av', 'Rv']
def extinction(b, solve_for=None, **kwargs):
    """
    Create a constraint for the translation between ebv, Av, and Rv.

    This constraint is automatically created and attached for all systems.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('extinction')`.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `solve_for` (<phoebe.parameters.Parameter, optional, default=None): if
        'ebv' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'Av', 'Rv').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    # Rv =Av/ebv
    system_ps = b.filter(context='system', **_skip_filter_checks)
    ebv = system_ps.get_parameter(qualifier='ebv', **_skip_filter_checks)
    Av = system_ps.get_parameter(qualifier='Av', **_skip_filter_checks)
    Rv = system_ps.get_parameter(qualifier='Rv', **_skip_filter_checks)


    if solve_for in [None, ebv]:
        lhs = ebv
        rhs = Av / Rv
    elif solve_for in [Av]:
        lhs = Av
        rhs = Rv * Av
    elif solve_for in [Rv]:
        lhs = Rv
        # NOTE: could result in infinity
        rhs = Av / ebv
    else:
        raise NotImplementedError

    return lhs, rhs, [], {}

_validsolvefor['parallax'] = ['distance', 'parallax']
def parallax(b, solve_for=None, **kwargs):
    """
    Create a constraint for the translation between distance and parallax.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('parallax')`.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `solve_for` (<phoebe.parameters.Parameter, optional, default=None): if
        'parallax' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'distance').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    # Rv =Av/ebv
    system_ps = b.filter(context='system', **_skip_filter_checks)
    distance = system_ps.get_parameter(qualifier='distance', **_skip_filter_checks)

    parallax_def = FloatParameter(qualifier='parallax', latexfmt=r'\pi', value=1.0, default_unit=u.arcsec, description='Parallax')

    # And now call get_or_create on the bundle
    metawargs = system_ps.meta
    metawargs.pop('qualifier')
    parallax, created = b.get_or_create('parallax', parallax_def, **metawargs)

    # NOTE: parallax here is in radians and distance in solRad (solar units)
    if solve_for in [None, parallax]:
        lhs = parallax
        rhs = (1*u.arcsec.to(u.rad)/1*u.pc.to(u.solRad))/distance
    elif solve_for in [distance]:
        lhs = distance
        rhs = (1*u.pc.to(u.solRad)/1*u.arcsec.to(u.rad))/parallax
    else:
        raise NotImplementedError

    return lhs, rhs, [], {}

# _validsolvefor['time_ephem'] = ['time_ephem']
# def time_ephem(b, component, dataset, solve_for=None, **kwargs):
#     """
#     use the ephemeris of component to predict the expected times of eclipse (used
#         in the ETV dataset)
#
#     This is usually passed as an argument to
#      <phoebe.frontend.bundle.Bundle.add_constraint>.
#     """
#     hier = b.get_hierarchy()
#     if not len(hier.get_value()):
#         # TODO: change to custom error type to catch in bundle.add_component
#         # TODO: check whether the problem is 0 hierarchies or more than 1
#         raise NotImplementedError("constraint for time_ecl requires hierarchy")
#
#     if component=='_default':
#         # need to do this so that the constraint won't fail before being copied
#         parentorbit = hier.get_top()
#     else:
#         parentorbit = hier.get_parent_of(component)
#
#     parentorbit_ps = _get_system_ps(b, parentorbit)
#
#     filterwargs = _skip_filter_checks
#     if component is not None:
#         filterwargs['component'] = component
#     if dataset is not None:
#         filterwargs['dataset'] = dataset
#
#     time_ephem = b.get_parameter(qualifier='time_ephems', **filterwargs)
#     t0 = parentorbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)  # TODO: make sure t0_supconj makes sense here
#     period = parentorbit_ps.get_parameter(qualifier='period', **_skip_filter_checks)
#     phshift = parentorbit_ps.get_parameter(qualifier='phshift', **_skip_filter_checks)
#     dpdt = parentorbit_ps.get_parameter(qualifier='dpdt', **_skip_filter_checks)
#     esinw_ = parentorbit_ps.get_parameter(qualifier='esinw', **_skip_filter_checks)
#
#     N = b.get_parameter(qualifier='Ns', **filterwargs)
#
#     if solve_for in [None, time_ephem]:
#
#         # TODO: N is always an int, but we want to include the expected phase of eclipse (ie N+ph_ecl) based on which component and esinw/ecosw
#         # then we can have bundle.add_component automatically default to add all components instead of just the primary
#
#         # same as Bundle.to_time except phase can be > 1
#         lhs = time_ephem
#         # we have to do a trick here since dpdt is in sec/yr and floats are
#         # assumed to have the same unit during subtraction or addition.
#         one = 1.0*(u.s/u.s)
#         if component!='_default' and hier.get_primary_or_secondary(component)=='secondary':
#             # TODO: make sure this constraint updates if the hierarchy changes?
#             N = N + 0.5 + esinw_  # TODO: check this
#         rhs = t0 + ((N - phshift) * period) / (-1 * (N - phshift) * dpdt + one)
#         #rhs = (N-phshift)*period
#     else:
#         raise NotImplementedError
#
#     return lhs, rhs, [], {'component': component, 'dataset': dataset}
#
# def etv(b, component, dataset, solve_for=None, **kwargs):
#     """
#     compute the ETV column from the time_ephem and time_ecl columns (used in the
#         ETV dataset).
#
#     This is usually passed as an argument to
#      <phoebe.frontend.bundle.Bundle.add_constraint>.
#     """
#
#     time_ephem = b.get_parameter(qualifier='time_ephems', component=component, dataset=dataset, context=['dataset', 'model'])  # need to provide context to avoid getting the constraint
#     time_ecl = b.get_parameter(qualifier='time_ecls', component=component, dataset=dataset)
#     etv = b.get_parameter(qualifier='etvs', component=component, dataset=dataset)
#
#     if solve_for in [None, etv]:
#         lhs = etv
#         rhs = time_ecl - time_ephem
#     else:
#         raise NotImplementedError
#
#     return lhs, rhs, [], {'component': component, 'dataset': dataset}

#}

def requiv_to_pot(b, component, solve_for=None, **kwargs):
    """
    This constraint is automatically created and attached for all stars
    in contact binary orbits via <phoebe.frontend.bundle.Bundle.set_hierarchy>.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('requiv_to_pot', component='common_envelope')`, where `component` is
     one of <phoebe.parameters.HierarchyParameter.get_envelopes> or
     <phoebe.parameters.HierarchyParameter.get_stars>.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `component` (string): the label of the orbit or component in which this
        constraint should be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'pot' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'q', 'sma', 'requiv').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    hier = b.get_hierarchy()
    parentorbit = hier.get_parent_of(component)

    parentorbit_ps = _get_system_ps(b, parentorbit)

    if hier.get_kind_of(component) == 'envelope':
        raise NotImplementedError
        # envelope_ps = _get_system_ps(b, component)
        # component_ps = _get_system_ps(b, hier.get)
    else:
        component_ps = _get_system_ps(b, component)
        envelope_ps = _get_system_ps(b, hier.get_envelope_of(component))

    q = parentorbit_ps.get_parameter(qualifier='q', **_skip_filter_checks)
    sma = parentorbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)

    # assuming component is always primary or secondary and never envelope
    pot = envelope_ps.get_parameter(qualifier='pot', **_skip_filter_checks)
    requiv = component_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)

    compno = hier.get_primary_or_secondary(component, return_ind=True)

    if solve_for in [None, requiv]:
        lhs = requiv
        rhs = pot_to_requiv_contact(pot, q, sma, compno)
    elif solve_for == pot:
        lhs = pot
        rhs = requiv_to_pot_contact(requiv, q, sma, compno)
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}



_validsolvefor['impact_param'] = ['incl', 'impact_param']
def impact_param(b, orbit=None, solve_for=None, **kwargs):
    """
    Create a constraint between the impact parameter and inclination in an orbit.

    This is usually passed as an argument to
     <phoebe.frontend.bundle.Bundle.add_constraint> as
     `b.add_constraint('impact_param', orbit='binary')`, where `orbit` is one of
     <phoebe.parameters.HierarchyParameter.get_orbits>.

    If 'impact_param' does not exist in the orbit, it will be created.

    Arguments
    -----------
    * `b` (<phoebe.frontend.bundle.Bundle>): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should
        be built.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'impact_param' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'incl')

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list):
        lhs (Parameter), rhs (ConstraintParameter), addl_params (list of additional
        parameters that may be included in the constraint), kwargs (dict of
        keyword arguments that were passed to this function).

    Raises
    --------
    * NotImplementedError: if the value of `solve_for` is not implemented.
    """

    hier = b.hierarchy

    if orbit is None:
        orbits = hier.get_orbits()
        if len(orbits)==1:
            orbit = orbits[0]
        else:
            raise ValueError("must provide orbit since more than one orbit present in the hierarchy")


    orbit_ps = _get_system_ps(b, orbit)

    # We want to get the parameters in THIS orbit, but calling through
    # the bundle in case we need to create it.
    # To do that, we need to know the search parameters to get items from this PS.
    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    # Now we'll define the parameters in case they don't exist and need to be created
    impactparam_def = FloatParameter(qualifier='impact_param', latexfmt=r'x_\mathrm{im}', value=0., default_unit=u.dimensionless_unscaled, limits=[-2, 2], description='Impact parameter of the orbit')

    # And now call get_or_create on the bundle
    impactparam, impactparam_created = b.get_or_create('impact_param', impactparam_def, **metawargs)
    comp1, comp2 = hier.get_stars_of_children_of(orbit)
    comp1_ps = b.get_component(component=comp1, **_skip_filter_checks)
    comp2_ps = b.get_component(component=comp2, **_skip_filter_checks)

    requiv1 = comp1_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    requiv2 = comp2_ps.get_parameter(qualifier='requiv', **_skip_filter_checks)
    sma = orbit_ps.get_parameter(qualifier='sma', **_skip_filter_checks)
    requivsumfrac = (requiv1 + requiv2)/sma

    incl = orbit_ps.get_parameter(qualifier='incl', **_skip_filter_checks)
    esinw = orbit_ps.get_parameter(qualifier='esinw', **_skip_filter_checks)
    ecosw = orbit_ps.get_parameter(qualifier='ecosw', **_skip_filter_checks)


    if solve_for in [None, impactparam]:
        lhs = impactparam
        rhs = cos(incl)/requivsumfrac * (1-esinw**2-ecosw**2)/(1+esinw)

    elif solve_for == incl:
        lhs = incl
        rhs = arccos(impactparam*requivsumfrac*(1+esinw)/(1-esinw**2-ecosw**2))

    else:
        raise NotImplementedError

    #- return lhs, rhs, args_as_pss
    return lhs, rhs, [], {'orbit': orbit}
