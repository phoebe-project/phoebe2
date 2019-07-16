import numpy as np
#from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt

from phoebe.parameters import *
from phoebe import u, c

import logging
logger = logging.getLogger("CONSTRAINT")
logger.addHandler(logging.NullHandler())

list_of_constraints_requiring_si = []


def _get_system_ps(b, item, context='component'):
    """
    parses the input arg (either twig or PS) to retrieve the actual parametersets
    """
    # TODO: make this a decorator?
    if isinstance(item, list) and len(item)==1:
        item = item[0]

    if isinstance(item, ParameterSet):
        return item.filter(context=context, check_visible=False)
    elif isinstance(item, str):
        return b.filter(item, context=context, check_visible=False)
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

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "sin({})".format(_get_expr(param)))

def cos(param):
    """
    Allows using the cos function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "cos({})".format(_get_expr(param)))

def tan(param):
    """
    Allows using the tan function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "tan({})".format(_get_expr(param)))

def arcsin(param):
    """
    Allows using the arcsin function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "arcsin({})".format(_get_expr(param)))

def arccos(param):
    """
    Allows using the arccos function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    # print "***", "arccos({})".format(_get_expr(param))
    return ConstraintParameter(param._bundle, "arccos({})".format(_get_expr(param)))

def arctan(param):
    """
    Allows using the arctan function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "arctan({})".format(_get_expr(param)))

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

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "abs({})".format(_get_expr(param)))

def sqrt(param):
    """
    Allows using the sqrt (square root) function in a constraint

    :parameter param: the :class:`phoebe.parameters.parameters.Parameter`
    :returns: the :class:`phoebe.parameters.parameters.ConstraintParameter`
    """
    return ConstraintParameter(param._bundle, "sqrt({})".format(_get_expr(param)))

#}
#{ Built-in functions (see phoebe.constraints.builtin for actual functions)
def roche_requiv_L1(q, syncpar, ecc, sma, incl_star, long_an_star, incl_orb, long_an_orb, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(q._bundle, "requiv_L1(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q, syncpar, ecc, sma, incl_star, long_an_star, incl_orb, long_an_orb)]), compno))

def roche_requiv_contact_L1(q, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(q._bundle, "requiv_contact_L1(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q, sma)]), compno))

def roche_requiv_contact_L23(q, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(q._bundle, "requiv_contact_L23(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q, sma)]), compno))

def roche_potential_contact_L1(q):
    """
    """
    return ConstraintParameter(q._bundle, "potential_contact_L1({})".format(_get_expr(q)))

def roche_potential_contact_L23(q):
    """
    """
    return ConstraintParameter(q._bundle, "potential_contact_L23({})".format(_get_expr(q)))

def roche_pot_to_fillout_factor(q, pot):
    """
    """
    return ConstraintParameter(q._bundle, "pot_to_fillout_factor({}, {})".format(_get_expr(q), _get_expr(pot)))

def roche_fillout_factor_to_pot(q, fillout_factor):
    """
    """
    return ConstraintParameter(q._bundle, "fillout_factor_to_pot({}, {})".format(_get_expr(q), _get_expr(fillout_factor)))

def requiv_to_pot_contact(requiv, q, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(requiv._bundle, "requiv_to_pot_contact({}, {}, {}, {})".format(_get_expr(requiv), _get_expr(q), _get_expr(sma), compno))

def pot_to_requiv_contact(pot, q, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(pot._bundle, "pot_to_requiv_contact({}, {}, {}, {})".format(_get_expr(pot), _get_expr(q), _get_expr(sma), compno))

def esinw2per0(ecc, esinw):
    """
    TODO: add documentation
    """
    return ConstraintParameter(ecc._bundle, "esinw2per0({}, {})".format(_get_expr(ecc), _get_expr(esinw)))

def ecosw2per0(ecc, ecosw):
    """
    TODO: add documentation
    """
    # print "***", "ecosw2per0({}, {})".format(_get_expr(ecc), _get_expr(ecosw))
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

def t0_perpass_to_supconj(t0_perpass, period, ecc, per0):
    """
    TODO: add documentation
    """
    return ConstraintParameter(t0_perpass._bundle, "t0_perpass_to_supconj({}, {}, {}, {})".format(_get_expr(t0_perpass), _get_expr(period), _get_expr(ecc), _get_expr(per0)))

def t0_supconj_to_perpass(t0_supconj, period, ecc, per0):
    """
    TODO: add documentation
    """
    return ConstraintParameter(t0_supconj._bundle, "t0_supconj_to_perpass({}, {}, {}, {})".format(_get_expr(t0_supconj), _get_expr(period), _get_expr(ecc), _get_expr(per0)))

def t0_ref_to_supconj(t0_ref, period, ecc, per0):
    """
    TODO: add documentation
    """
    return ConstraintParameter(t0_ref._bundle, "t0_ref_to_supconj({}, {}, {}, {})".format(_get_expr(t0_ref), _get_expr(period), _get_expr(ecc), _get_expr(per0)))

def t0_supconj_to_ref(t0_supconj, period, ecc, per0):
    """
    TODO: add documentation
    """
    return ConstraintParameter(t0_supconj._bundle, "t0_supconj_to_ref({}, {}, {}, {})".format(_get_expr(t0_supconj), _get_expr(period), _get_expr(ecc), _get_expr(per0)))

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

def asini(b, orbit, solve_for=None):
    """
    Create a constraint for asini in an orbit.

    If any of the required parameters ('asini', 'sma', 'incl') do not
    exist in the orbit, they will be created.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str orbit: the label of the orbit in which this
        constraint should be built
    :parameter str solve_for:  if 'asini' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'sma' or 'incl')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """

    orbit_ps = _get_system_ps(b, orbit)

    # We want to get the parameters in THIS orbit, but calling through
    # the bundle in case we need to create it.
    # To do that, we need to know the search parameters to get items from this PS.
    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    # Now we'll define the parameters in case they don't exist and need to be created
    sma_def = FloatParameter(qualifier='sma', value=8.0, default_unit=u.solRad, description='Semi major axis')
    incl_def = FloatParameter(qualifier='incl', value=90.0, default_unit=u.deg, description='Orbital inclination angle')
    asini_def = FloatParameter(qualifier='asini', value=8.0, default_unit=u.solRad, description='Projected semi major axis')

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

def esinw(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for esinw in an orbit.

    If 'esinw' does not exist in the orbit, it will be created

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str orbit: the label of the orbit in which this
        constraint should be built
    :parameter str solve_for:  if 'esinw' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'ecc', 'per0')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    orbit_ps = _get_system_ps(b, orbit)

    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    esinw_def = FloatParameter(qualifier='esinw', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times sin of argument of periastron')
    esinw, created = b.get_or_create('esinw', esinw_def, **metawargs)

    ecosw_def = FloatParameter(qualifier='ecosw', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times cos of argument of periastron')
    ecosw, ecosw_created = b.get_or_create('ecosw', ecosw_def, **metawargs)

    ecosw_constrained = kwargs.get('ecosw_constrained', len(ecosw.constrained_by) > 0)
    # print("~~~esinw constraint: solve_for={}, ecosw_constrained={}".format(solve_for.qualifier if solve_for is not None else "None", ecosw_constrained))

    ecc = b.get_parameter(qualifier='ecc', **metawargs)
    per0 = b.get_parameter(qualifier='per0', **metawargs)

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
                # print("~~~esinw constraint: attempting to also flip per0 constraint")
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
                # print("~~~esinw constraint: attempting to also flip ecc constraint")
                ecc.is_constraint.constraint_kwargs['esinw_constrained'] = False
                ecc.is_constraint.flip_for('ecc', force=True)
    elif solve_for == ecosw:
        raise NotImplementedError("cannot solve this constraint for 'ecosw' since it was originally 'esinw'")
    else:
        raise NotImplementedError

    return lhs, rhs, [esinw, ecosw, ecc, per0], {'orbit': orbit}

def ecosw(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for ecosw in an orbit.

    If 'ecosw' does not exist in the orbit, it will be created

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str orbit: the label of the orbit in which this
        constraint should be built
    :parameter str solve_for:  if 'ecosw' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'ecc' or 'per0')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    orbit_ps = _get_system_ps(b, orbit)

    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    ecosw_def = FloatParameter(qualifier='ecosw', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times cos of argument of periastron')
    ecosw, created = b.get_or_create('ecosw', ecosw_def, **metawargs)

    esinw_def = FloatParameter(qualifier='esinw', value=0.0, default_unit=u.dimensionless_unscaled, limits=(-1.0,1.0), description='Eccentricity times sin of argument of periastron')
    esinw, esinw_created = b.get_or_create('esinw', esinw_def, **metawargs)

    esinw_constrained = kwargs.get('esinw_constrained', len(esinw.constrained_by) > 0)
    # print("~~~ecosw constraint: solve_for={}, esinw_constrained={}".format(solve_for.qualifier if solve_for is not None else "None", esinw_constrained))

    ecc = b.get_parameter(qualifier='ecc', **metawargs)
    per0 = b.get_parameter(qualifier='per0', **metawargs)

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
                # print("~~~ecosw constraint: attempting to also flip per0 constraint")
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
                # print("~~~ecosw constraint: attempting to also flip per0 constraint")
                ecc.is_constraint.constraint_kwargs['ecosw_constrained'] = False
                ecc.is_constraint.flip_for('ecc', force=True)
    elif solve_for == esinw:
        raise NotImplementedError("cannot solve this constraint for 'esinw' since it was originally 'ecosw'")
    else:
        raise NotImplementedError

    return lhs, rhs, [esinw, ecosw, ecc, per0], {'orbit': orbit}

def t0_perpass_supconj(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for t0_perpass in an orbit - allowing translating between
    t0_perpass and t0_supconj.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str orbit: the label of the orbit in which this
        constraint should be built
    :parameter str solve_for:  if 't0_perpass' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 't0_supconj', 'per0', 'period')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """

    orbit_ps = _get_system_ps(b, orbit)

    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    # by default both t0s exist in an orbit, so we don't have to worry about creating either
    t0_perpass = b.get_parameter(qualifier='t0_perpass', **metawargs)
    t0_supconj = b.get_parameter(qualifier='t0_supconj', **metawargs)
    period = b.get_parameter(qualifier='period', **metawargs)
    ecc = b.get_parameter(qualifier='ecc', **metawargs)
    per0 = b.get_parameter(qualifier='per0', **metawargs)

    if solve_for in [None, t0_perpass]:
        lhs = t0_perpass
        rhs = t0_supconj_to_perpass(t0_supconj, period, ecc, per0)

    elif solve_for == t0_supconj:
        lhs = t0_supconj
        rhs = t0_perpass_to_supconj(t0_perpass, period, ecc, per0)



    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

def t0(*args, **kwargs):
    """
    shortcut to t0_perpass for backwards compatibility
    """
    return t0_perpass_supconj(*args, **kwargs)

def t0_ref_supconj(b, orbit, solve_for=None, **kwargs):
    """
    Create a constraint for t0_ref in an orbit - allowing translating between
    t0_ref and t0_supconj.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str orbit: the label of the orbit in which this
        constraint should be built
    :parameter str solve_for:  if 't0_ref' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 't0_supconj', 'per0', 'period')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """

    orbit_ps = _get_system_ps(b, orbit)

    metawargs = orbit_ps.meta
    metawargs.pop('qualifier')

    # by default both t0s exist in an orbit, so we don't have to worry about creating either
    t0_ref = b.get_parameter(qualifier='t0_ref', **metawargs)
    t0_supconj = b.get_parameter(qualifier='t0_supconj', **metawargs)
    period = b.get_parameter(qualifier='period', **metawargs)
    ecc = b.get_parameter(qualifier='ecc', **metawargs)
    per0 = b.get_parameter(qualifier='per0', **metawargs)

    if solve_for in [None, t0_ref]:
        lhs = t0_ref
        rhs = t0_supconj_to_ref(t0_supconj, period, ecc, per0)

    elif solve_for == t0_supconj:
        lhs = t0_supconj
        rhs = t0_ref_to_supconj(t0_ref, period, ecc, per0)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}


def mean_anom(b, orbit, solve_for=None, **kwargs):
    """
    """

    orbit_ps = _get_system_ps(b, orbit)

    mean_anom = orbit_ps.get_parameter(qualifier='mean_anom')
    t0_perpass = orbit_ps.get_parameter(qualifier='t0_perpass')
    period = orbit_ps.get_parameter(qualifier='period')
    time0 = b.get_parameter(qualifier='t0', context='system')

    if solve_for in [None, mean_anom]:
        lhs = mean_anom
        rhs = 2 * np.pi * (time0 - t0_perpass) / period
    elif solve_for in [t0_perpass]:
        lhs = t0_perpass
        rhs = time0 - (mean_anom*period)/(2*np.pi*u.rad)
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

def ph_supconj(b, orbit, solve_for=None, **kwargs):
    """
    TODO: add documentation
    """
    orbit_ps = _get_system_ps(b, orbit)

    # metawargs = orbit_ps.meta
    #metawargs.pop('qualifier')

    # t0_ph0 and phshift both exist by default, so we don't have to worry about creating either
    # t0_ph0 = orbit_ps.get_parameter(qualifier='t0_ph0')
    # phshift = orbit_ps.get_parameter(qualifier='phshift')
    ph_supconj = orbit_ps.get_parameter(qualifier='ph_supconj')
    per0 = orbit_ps.get_parameter(qualifier='per0')
    ecc = orbit_ps.get_parameter(qualifier='ecc')
    period = orbit_ps.get_parameter(qualifier='period')

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

def ph_infconj(b, orbit, solve_for=None, **kwargs):
    """
    TODO: add documentation
    """
    orbit_ps = _get_system_ps(b, orbit)

    ph_infconj = orbit_ps.get_parameter(qualifier='ph_infconj')
    per0 = orbit_ps.get_parameter(qualifier='per0')
    ecc = orbit_ps.get_parameter(qualifier='ecc')
    period = orbit_ps.get_parameter(qualifier='period')

    if solve_for in [None, ph_infconj]:
        lhs = ph_infconj

        #true_anom_infconj = 3*np.pi/2 - per0
        # true_anom_infconj = (3*90)*u.deg - per0  # TODO: fix math to allow this
        true_anom_infconj = -1*(per0 - (3*90)*u.deg)

        rhs = _true_anom_to_phase(true_anom_infconj, period, ecc, per0)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}

def ph_perpass(b, orbit, solve_for=None, **kwargs):
    """
    TODO: add documentation
    """
    orbit_ps = _get_system_ps(b, orbit)

    ph_perpass = orbit_ps.get_parameter(qualifier='ph_perpass')
    per0 = orbit_ps.get_parameter(qualifier='per0')
    ecc = orbit_ps.get_parameter(qualifier='ecc')
    period = orbit_ps.get_parameter(qualifier='period')

    if solve_for in [None, ph_perpass]:
        lhs = ph_perpass

        # true_anom_per0 = (per0 - pi/2) / (2*pi)
        true_anom_per0 = (per0 - 90*u.deg) / (360)

        rhs = _true_anom_to_phase(true_anom_per0, period, ecc, per0)

    else:
        raise NotImplementedError

    return lhs, rhs, [], {'orbit': orbit}





def freq(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for frequency (either orbital or rotational) given a period.

    freq = 2 * pi / period

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the orbit or component in which this
        constraint should be built
    :parameter str solve_for:  if 'freq' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'period')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """

    component_ps = _get_system_ps(b, component)

    #metawargs = component_ps.meta
    #metawargs.pop('qualifier')

    period = component_ps.get_parameter(qualifier='period', check_visible=False)
    freq = component_ps.get_parameter(qualifier='freq', check_visible=False)

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
    """

    hier = b.hierarchy

    orbit1_ps = _get_system_ps(b, orbit1)
    orbit2_ps = _get_system_ps(b, orbit2)

    sma1 = orbit1_ps.get_parameter(qualifier='sma')
    sma2 = orbit2_ps.get_parameter(qualifier='sma')

    q1 = orbit1_ps.get_parameter(qualifier='q')
    q2 = orbit2_ps.get_parameter(qualifier='q')

    period1 = orbit1_ps.get_parameter(qualifier='period')
    period2 = orbit2_ps.get_parameter(qualifier='period')

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

def irrad_frac(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to ensure that energy is conserved and all incident
    light is accounted for.
    """

    comp_ps = b.get_component(component=component)

    irrad_frac_refl_bol = comp_ps.get_parameter(qualifier='irrad_frac_refl_bol')
    irrad_frac_lost_bol = comp_ps.get_parameter(qualifier='irrad_frac_lost_bol')

    if solve_for in [irrad_frac_lost_bol, None]:
        lhs = irrad_frac_lost_bol
        rhs = 1.0 - irrad_frac_refl_bol
    elif solve_for in [irrad_frac_refl_bol]:
        lhs = irrad_frac_refl_bol
        rhs = 1.0 - irrad_frac_lost_bol
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}

def semidetached(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to force requiv to be semidetached
    """
    comp_ps = b.get_component(component=component)

    requiv = comp_ps.get_parameter(qualifier='requiv')
    requiv_critical = comp_ps.get_parameter(qualifier='requiv_max')

    if solve_for in [requiv, None]:
        lhs = requiv
        rhs = 1.0*requiv_critical
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component}


#}
#{ Inter-component constraints

def teffratio(b, orbit=None, solve_for=None, **kwargs):
    """
    Introduced in 2.1.7

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
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list): lhs (Parameter), rhs (ConstraintParameter), args (list of arguments that were passed to this function)

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

    comp1_ps = b.get_component(component=comp1)
    comp2_ps = b.get_component(component=comp2)

    teffratio_def = FloatParameter(qualifier='teffratio', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='ratio between effective temperatures of children stars')
    teffratio, created = b.get_or_create('teffratio', teffratio_def, component=orbit, context='component')

    teff1 = comp1_ps.get_parameter(qualifier='teff')
    teff2 = comp2_ps.get_parameter(qualifier='teff')

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



def requivratio(b, orbit=None, solve_for=None, **kwargs):
    """
    Introduced in 2.1.7

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
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list): lhs (Parameter), rhs (ConstraintParameter), args (list of arguments that were passed to this function)

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

    comp1_ps = b.get_component(component=comp1)
    comp2_ps = b.get_component(component=comp2)

    requiv1 = comp1_ps.get_parameter(qualifier='requiv')
    requiv2 = comp2_ps.get_parameter(qualifier='requiv')

    requivratio_def = FloatParameter(qualifier='requivratio', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='ratio between equivalent radii of children stars')
    requivratio, requivratio_created = b.get_or_create('requivratio', requivratio_def, component=orbit, context='component')

    requivsum_def = FloatParameter(qualifier='requivsum', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='sum of equivalent radii of children stars')
    requivsum, requivsum_created = b.get_or_create('requivsum', requivsum_def, component=orbit, context='component')

    requivsum_constrained = kwargs.get('requivsum_constrained', len(requivsum.constrained_by) > 0)

    if solve_for in [requivratio, None]:
        lhs = requivratio
        rhs = requiv2/requiv1
        if not requivsum_created and not requivsum_constrained:
            if requiv1.is_constraint:
                requiv1.is_constraint.constraint_kwargs['requivratio_constrained'] = True
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)
            elif requiv2.is_constraint:
                requiv2.is_constraint.constraint_kwargs['requivratio_constrained'] = True
                requiv2.is_constraint.flip_for('requiv@'.format(requiv2.component), force=True)

    elif solve_for in [requiv1]:
        lhs = requiv1
        if requivsum_constrained:
            rhs = requiv2 / requivratio
        else:
            rhs = requivsum / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv2.is_constraint and 'requivratio_constrained' not in requiv2.is_constraint.constraint_kwargs.keys():
                requiv2.is_constraint.constraint_kwargs['requivratio_constrained'] = False
                requiv2.is_constraint.flip_for('requiv@{}'.format(requiv2.component), force=True)

    elif solve_for in [requiv2]:
        lhs = requiv2
        if requivsum_constrained:
            rhs = requivratio * requiv1
        else:
            rhs = (requivratio * requivsum) / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv1.is_constraint and 'requivratio_constrained' not in requiv1.is_constraint.constraint_kwargs.keys():
                requiv1.is_constraint.constraint_kwargs['requivratio_constrained'] = False
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)
    elif solve_for == requivsum:
        raise NotImplementedError("cannot solve this constraint for 'requivsum' since it was originally 'requivratio'")
    else:
        raise NotImplementedError


    return lhs, rhs, [requivratio, requivsum, requiv1, requiv2], {'orbit': orbit}

def requivsum(b, orbit=None, solve_for=None, **kwargs):
    """
    Introduced in 2.1.7

    Create a constraint to for the requiv sum of two stars in the same orbit.
    Defined as requivsum = requiv@comp2 / requiv@comp1, where comp1 and comp2 are
    determined from the primary and secondary components of the orbit `orbit`.

    This is usually passed as an argument to
    <phoebe.frontend.bundle.Bundle.add_constraint> as
    `b.add_constraint('requivsum', orbit='binary')`, where
    `orbit` is one of <phoebe.parameters.HierarchyParameter.get_orbits>.

    Arguments
    -----------
    * `b` (phoebe.frontend.bundle.Bundle): the Bundle
    * `orbit` (string): the label of the orbit in which this constraint should be built.
        Optional if only one orbit exists in the hierarchy.
    * `solve_for` (<phoebe.parameters.Parameter>, optional, default=None): if
        'requivsum' should not be the derived/constrained parameter, provide which
        other parameter should be derived (ie 'requiv@...').

    Returns
    ----------
    * (<phoebe.parameters.Parameter>, <phoebe.parameters.ConstraintParameter>, list): lhs (Parameter), rhs (ConstraintParameter), args (list of arguments that were passed to this function)

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

    comp1_ps = b.get_component(component=comp1)
    comp2_ps = b.get_component(component=comp2)

    requiv1 = comp1_ps.get_parameter(qualifier='requiv')
    requiv2 = comp2_ps.get_parameter(qualifier='requiv')

    requivratio_def = FloatParameter(qualifier='requivratio', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='ratio between equivalent radii of children stars')
    requivratio, requivratio_created = b.get_or_create('requivratio', requivratio_def, component=orbit, context='component')

    requivsum_def = FloatParameter(qualifier='requivsum', value=1.0, default_unit=u.dimensionless_unscaled, limits=[0, None], description='sum of equivalent radii of children stars')
    requivsum, requivsum_created = b.get_or_create('requivsum', requivsum_def, component=orbit, context='component')

    requivratio_constrained = kwargs.get('requivratio_constrained', len(requivratio.constrained_by) > 0)

    if solve_for in [requivsum, None]:
        lhs = requivsum
        rhs = requiv1 + requiv2
        if not requivratio_created and not requivratio_constrained:
            if requiv1.is_constraint:
                requiv1.is_constraint.constraint_kwargs['requivsum_constrained'] = True
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)
            elif requiv2.is_constraint:
                requiv2.is_constraint.constraint_kwargs['requivsum_constrained'] = True
                requiv2.is_constraint.flip_for('requiv@'.format(requiv2.component), force=True)

    elif solve_for in [requiv1]:
        lhs = requiv1
        if requivratio_constrained:
            rhs = requivsum - requiv2
        else:
            rhs = requivsum / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv2.is_constraint and 'requivsum_constrained' not in requiv2.is_constraint.constraint_kwargs.keys():
                requiv2.is_constraint.constraint_kwargs['requivsum_constrained'] = False
                requiv2.is_constraint.flip_for('requiv@{}'.format(requiv2.component), force=True)

    elif solve_for in [requiv2]:
        lhs = requiv2
        if requivratio_constrained:
            rhs = requivsum - requiv1
        else:
            rhs = (requivratio * requivsum) / (requivratio + 1)
            # the other constraint needs to also follow the alternate equations
            if requiv1.is_constraint and 'requivsum_constrained' not in requiv1.is_constraint.constraint_kwargs.keys():
                requiv1.is_constraint.constraint_kwargs['requivsum_constrained'] = False
                requiv1.is_constraint.flip_for('requiv@{}'.format(requiv1.component), force=True)

    elif solve_for == requivratio:
        raise NotImplementedError("cannot solve this constraint for 'requivratio' since it was originally 'requivsum'")
    else:
        raise NotImplementedError


    return lhs, rhs, [requivratio, requivsum, requiv1, requiv2], {'orbit': orbit}

#}
#{ Orbit-component constraints


def mass(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the mass of a star based on Kepler's third
    law from its parent orbit.

    If 'mass' does not exist in the component, it will be created

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'mass' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'q', sma', 'period')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    :raises NotImplementedError: if the hierarchy is not found
    :raises NotImplementedError: if the value of solve_for is not yet implemented
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

    mass = component_ps.get_parameter('mass')
    mass_sibling = sibling_ps.get_parameter('mass')

    # we need to find the constraint attached to the other component... but we
    # don't know who is constrained, or whether it belongs to the sibling or parent
    # orbit, so we'll have to do a bit of digging.
    mass_constraint_sibling = None
    for p in b.filter(constraint_func='mass', component=[parentorbit, sibling], context='constraint').to_list():
        if p.constraint_kwargs['component'] == sibling:
            mass_constraint_sibling = p
            break
    if mass_constraint_sibling is not None:
        sibling_solve_for = mass_constraint_sibling.qualifier
        logger.debug("constraint.mass for component='{}': sibling ('{}') is solved for '{}'".format(component, sibling, sibling_solve_for))
    else:
        # this could happen when we build the first constraint, before the second has been built
        sibling_solve_for = None

    sma = parentorbit_ps.get_parameter(qualifier='sma')
    period = parentorbit_ps.get_parameter(qualifier='period')
    q = parentorbit_ps.get_parameter(qualifier='q')

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


def comp_sma(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the star's semi-major axes WITHIN its
    parent orbit.  This is NOT the same as the semi-major axes OF
    the parent orbit

    If 'sma' does not exist in the component, it will be created

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'sma@star' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'sma@orbit', 'q')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
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
    compsma_def = FloatParameter(qualifier='sma', value=4.0, default_unit=u.solRad, description='Semi major axis of the component in the orbit')
    compsma, created = b.get_or_create('sma', compsma_def, **metawargs)

    metawargs = parentorbit_ps.meta
    metawargs.pop('qualifier')
    sma = b.get_parameter(qualifier='sma', **metawargs)
    q = b.get_parameter(qualifier='q', **metawargs)

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


def requiv_detached_max(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L1) value of
    requiv.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'requiv_max' should not be the derived/constrained
        parameter, provide which other parameter should be derived
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
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

    requiv_max = component_ps.get_parameter(qualifier='requiv_max')
    q = parentorbit_ps.get_parameter(qualifier='q')
    syncpar = component_ps.get_parameter(qualifier='syncpar')
    ecc = parentorbit_ps.get_parameter(qualifier='ecc')
    sma = parentorbit_ps.get_parameter(qualifier='sma')
    incl_star = component_ps.get_parameter(qualifier='incl')
    long_an_star = component_ps.get_parameter(qualifier='long_an')
    incl_orbit = parentorbit_ps.get_parameter(qualifier='incl')
    long_an_orbit = parentorbit_ps.get_parameter(qualifier='long_an')

    if solve_for in [None, requiv_max]:
        lhs = requiv_max

        rhs = roche_requiv_L1(q, syncpar, ecc, sma,
                              incl_star, long_an_star,
                              incl_orbit, long_an_orbit,
                              hier.get_primary_or_secondary(component, return_ind=True))
    else:
        raise NotImplementedError("requiv_detached_max can only be solved for requiv_max")

    return lhs, rhs, [], {'component': component}

def potential_contact_min(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L23) value of
    potential at which a constact will underflow.  This will only be used
    for contacts for pot_min

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'pot_min' should not be the derived/constrained
        parameter, provide which other parameter should be derived
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_min requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot_min = component_ps.get_parameter(qualifier='pot_min')
    q = parentorbit_ps.get_parameter(qualifier='q')

    if solve_for in [None, pot_min]:
        lhs = pot_min

        rhs = roche_potential_contact_L23(q)
    else:
        raise NotImplementedError("potential_contact_min can only be solved for requiv_min")

    return lhs, rhs, [], {'component': component}

def potential_contact_max(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L1) value of
    potential at which a constact will underflow.  This will only be used
    for contacts for pot_min

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'pot_max' should not be the derived/constrained
        parameter, provide which other parameter should be derived
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot_max = component_ps.get_parameter(qualifier='pot_max')
    q = parentorbit_ps.get_parameter(qualifier='q')

    if solve_for in [None, pot_max]:
        lhs = pot_max

        rhs = roche_potential_contact_L1(q)
    else:
        raise NotImplementedError("potential_contact_max can only be solved for requiv_max")

    return lhs, rhs, [], {'component': component}

def requiv_contact_min(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L1) value of
    requiv at which a constact will underflow.  This will only be used
    for contacts for requiv_min

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'requiv_max' should not be the derived/constrained
        parameter, provide which other parameter should be derived
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_min requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    requiv_min = component_ps.get_parameter(qualifier='requiv_min')
    q = parentorbit_ps.get_parameter(qualifier='q')
    sma = parentorbit_ps.get_parameter(qualifier='sma')

    if solve_for in [None, requiv_min]:
        lhs = requiv_min

        rhs = roche_requiv_contact_L1(q, sma, hier.get_primary_or_secondary(component, return_ind=True))
    else:
        raise NotImplementedError("requiv_contact_min can only be solved for requiv_min")

    return lhs, rhs, [], {'component': component}

def requiv_contact_max(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the critical (at L2/3) value of
    requiv at which a constact will overflow.  This will only be used
    for contacts for requiv_max

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'requiv_max' should not be the derived/constrained
        parameter, provide which other parameter should be derived
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    requiv_max = component_ps.get_parameter(qualifier='requiv_max')
    q = parentorbit_ps.get_parameter(qualifier='q')
    sma = parentorbit_ps.get_parameter(qualifier='sma')

    if solve_for in [None, requiv_max]:
        lhs = requiv_max

        rhs = roche_requiv_contact_L23(q, sma, hier.get_primary_or_secondary(component, return_ind=True))
    else:
        raise NotImplementedError("requiv_contact_max can only be solved for requiv_max")

    return lhs, rhs, [], {'component': component}

def fillout_factor(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to determine the fillout factor of a contact envelope.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'requiv_max' should not be the derived/constrained
        parameter, provide which other parameter should be derived
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for requiv_contact_max requires hierarchy")


    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot = component_ps.get_parameter(qualifier='pot')
    fillout_factor = component_ps.get_parameter(qualifier='fillout_factor')
    q = parentorbit_ps.get_parameter(qualifier='q')

    if solve_for in [None, fillout_factor]:
        lhs = fillout_factor

        rhs = roche_pot_to_fillout_factor(q, pot)
    elif solve_for in [pot]:
        lhs = pot

        rhs = roche_fillout_factor_to_pot(q, fillout_factor)
    else:
        raise NotImplementedError("fillout_factor can not be solved for {}".format(solve_for))

    return lhs, rhs, [], {'component': component}

def rotation_period(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the rotation period of a star given its orbital
    period and synchronicity parameters.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'period@star' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'syncpar@star', 'period@orbit')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
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
    period_star = b.get_parameter(qualifier='period', **metawargs)
    syncpar_star = b.get_parameter(qualifier='syncpar', **metawargs)


    metawargs = parentorbit_ps.meta
    metawargs.pop('qualifier')
    period_orbit = b.get_parameter(qualifier='period', **metawargs)

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

def pitch(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the inclination of a star relative to its parent orbit

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'incl@star' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'incl@orbit', 'pitch@star')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """

    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for pitch requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    incl_comp = component_ps.get_parameter(qualifier='incl', check_visible=False)
    pitch_comp = component_ps.get_parameter(qualifier='pitch', check_visible=False)
    incl_orb = parentorbit_ps.get_parameter(qualifier='incl', check_visible=False)

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

def yaw(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the inclination of a star relative to its parent orbit

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'long_an@star' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'long_an@orbit', 'yaw@star')
    :returns: lhs (Parameter), rhs (ConstraintParameter), args (list of arguments
        that were passed to this function)
    """

    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for yaw requires hierarchy")

    component_ps = _get_system_ps(b, component)

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    long_an_comp = component_ps.get_parameter(qualifier='long_an', check_visible=False)
    yaw_comp = component_ps.get_parameter(qualifier='yaw', check_visible=False)
    long_an_orb = parentorbit_ps.get_parameter(qualifier='long_an', check_visible=False)

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
    ability to constraint pblum ratios (for colors)

    :raises NotImplementedError: because it isn't, yet
    """
    raise NotImplementedError

#}
#{ Dataset constraints

def time_ephem(b, component, dataset, solve_for=None, **kwargs):
    """
    use the ephemeris of component to predict the expected times of eclipse (used
        in the ETV dataset)
    """
    hier = b.get_hierarchy()
    if not len(hier.get_value()):
        # TODO: change to custom error type to catch in bundle.add_component
        # TODO: check whether the problem is 0 hierarchies or more than 1
        raise NotImplementedError("constraint for time_ecl requires hierarchy")

    if component=='_default':
        # need to do this so that the constraint won't fail before being copied
        parentorbit = hier.get_top()
    else:
        parentorbit = hier.get_parent_of(component)

    parentorbit_ps = _get_system_ps(b, parentorbit)

    filterwargs = {}
    if component is not None:
        filterwargs['component'] = component
    if dataset is not None:
        filterwargs['dataset'] = dataset

    time_ephem = b.get_parameter(qualifier='time_ephems', **filterwargs)
    t0 = parentorbit_ps.get_parameter(qualifier='t0_supconj')  # TODO: make sure t0_supconj makes sense here
    period = parentorbit_ps.get_parameter(qualifier='period')
    phshift = parentorbit_ps.get_parameter(qualifier='phshift')
    dpdt = parentorbit_ps.get_parameter(qualifier='dpdt')
    esinw_ = parentorbit_ps.get_parameter(qualifier='esinw')

    N = b.get_parameter(qualifier='Ns', **filterwargs)

    if solve_for in [None, time_ephem]:

        # TODO: N is always an int, but we want to include the expected phase of eclipse (ie N+ph_ecl) based on which component and esinw/ecosw
        # then we can have bundle.add_component automatically default to add all components instead of just the primary

        # same as Bundle.to_time except phase can be > 1
        lhs = time_ephem
        # we have to do a trick here since dpdt is in sec/yr and floats are
        # assumed to have the same unit during subtraction or addition.
        one = 1.0*(u.s/u.s)
        if component!='_default' and hier.get_primary_or_secondary(component)=='secondary':
            # TODO: make sure this constraint updates if the hierarchy changes?
            N = N + 0.5 + esinw_  # TODO: check this
        rhs = t0 + ((N - phshift) * period) / (-1 * (N - phshift) * dpdt + one)
        #rhs = (N-phshift)*period
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component, 'dataset': dataset}

def etv(b, component, dataset, solve_for=None, **kwargs):
    """
    compute the ETV column from the time_ephem and time_ecl columns (used in the
        ETV dataset)
    """

    time_ephem = b.get_parameter(qualifier='time_ephems', component=component, dataset=dataset, context=['dataset', 'model'])  # need to provide context to avoid getting the constraint
    time_ecl = b.get_parameter(qualifier='time_ecls', component=component, dataset=dataset)
    etv = b.get_parameter(qualifier='etvs', component=component, dataset=dataset)

    if solve_for in [None, etv]:
        lhs = etv
        rhs = time_ecl - time_ephem
    else:
        raise NotImplementedError

    return lhs, rhs, [], {'component': component, 'dataset': dataset}

#}

def requiv_to_pot(b, component, solve_for=None, **kwargs):

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

    q = parentorbit_ps.get_parameter(qualifier='q')
    sma = parentorbit_ps.get_parameter(qualifier='sma')

    # assuming component is always primary or secondary and never envelope
    pot = envelope_ps.get_parameter(qualifier='pot')
    requiv = component_ps.get_parameter(qualifier='requiv')

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
