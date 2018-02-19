
import numpy as np
#from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt

from phoebe.parameters import *
from phoebe import u, c

import logging
logger = logging.getLogger("CONSTRAINT")
logger.addHandler(logging.NullHandler())


def _get_system_ps(b, item, context='component'):
    """
    parses the input arg (either twig or PS) to retrieve the actual parametersets
    """
    # TODO: make this a decorator?

    if isinstance(item, ParameterSet):
        return item
    elif isinstance(item, str):
        return b.filter(item, context=context, check_visible=False)
    else:
        raise NotImplementedError("do not support item with type: {}".format(type(item)))

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
def rocherpole2potential(rpole, q, e, syncpar, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(rpole._bundle, "rocherpole2potential(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (rpole,q,e,syncpar,sma)]), compno))


def rochepotential2rpole(pot, q, e, syncpar, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(pot._bundle, "rochepotential2rpole(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (pot,q,e,syncpar,sma)]), compno))

def rotstarrpole2potential(rpole, rotfreq):
    """
    TODO: add documentation
    """
    return ConstraintParameter(rpole._bundle, "rotstarrpole2potential(%s)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (rpole, rotfreq)])))

def rotstarpotential2rpole(pot, rotfreq):
    """
    TODO: add documentation
    """
    return ConstraintParameter(pot._bundle, "rotstarpotential2rpole(%s)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (pot, rotfreq)])))

def rochecriticalL12potential(q, e, syncpar, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(q._bundle, "rochecriticalL12potential(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q,e,syncpar)]), compno))

def rochecriticalL12rpole(q, e, syncpar, sma, compno=1):
    """
    TODO: add documentation
    """
    return ConstraintParameter(q._bundle, "rochecriticalL12rpole(%s, %d)" % (", ".join(["{%s}" % (param.uniquetwig if hasattr(param, 'uniquetwig') else param.expr) for param in (q,e,syncpar,sma)]), compno))

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



#}
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
    return lhs, rhs, {'orbit': orbit}

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

    ecc = b.get_parameter(qualifier='ecc', **metawargs)
    per0 = b.get_parameter(qualifier='per0', **metawargs)

    if solve_for in [None, esinw]:
        lhs = esinw
        rhs = ecc * sin(per0)
    elif solve_for == ecc:
        lhs = ecc
        rhs = esinw / sin(per0)
    elif solve_for == per0:
        lhs = per0
        #rhs = arcsin(esinw/ecc)
        rhs = esinw2per0(ecc, esinw)

    else:
        raise NotImplementedError

    return lhs, rhs, {'orbit': orbit}

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

    ecc = b.get_parameter(qualifier='ecc', **metawargs)
    per0 = b.get_parameter(qualifier='per0', **metawargs)

    if solve_for in [None, ecosw]:
        lhs = ecosw
        rhs = ecc * cos(per0)

    elif solve_for == ecc:
        lhs = ecc
        rhs = ecosw / cos(per0)
    elif solve_for == per0:
        lhs = per0
        #rhs = arccos(ecosw/ecc)
        rhs = ecosw2per0(ecc, ecosw)
    else:
        raise NotImplementedError

    return lhs, rhs, {'orbit': orbit}

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

    return lhs, rhs, {'orbit': orbit}

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

    return lhs, rhs, {'orbit': orbit}


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

    return lhs, rhs, {'orbit': orbit}

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

    return lhs, rhs, {'orbit': orbit}

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

    return lhs, rhs, {'orbit': orbit}

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

    return lhs, rhs, {'orbit': orbit}





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

    period = component_ps.get_parameter(qualifier='period')
    freq = component_ps.get_parameter(qualifier='freq')

    if solve_for in [None, freq]:
        lhs = freq
        rhs = 2 * np.pi / period

    elif solve_for == period:
        lhs = period
        rhs = freq / (2 * np.pi)

    else:
        raise NotImplementedError

    return lhs, rhs, {'component': component}

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
        rhs = (sma2**3 * qthing1 * period1**2/period2**2)**(1./3)
    else:
        # TODO: add other options to solve_for
        raise NotImplementedError

    return lhs, rhs, {'orbit1': orbit1, 'orbit2': orbit2}

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

    return lhs, rhs, {'component': component}

def reflredist(b, component, solve_for=None, **kwargs):
    """
    Create a constraint to ensure that all reflected light is considered under
    the available redistribution schemes
    """
    comp_ps = b.get_component(component=component)

    frac_refl_noredist_bol = comp_ps.get_parameter('frac_refl_noredist_bol')
    frac_refl_localredist_bol = comp_ps.get_parameter('frac_refl_localredist_bol')
    frac_refl_horizredist_bol = comp_ps.get_parameter('frac_refl_horizredist_bol')
    frac_refl_globalredist_bol = comp_ps.get_parameter('frac_refl_globalredist_bol')

    if solve_for in [frac_refl_noredist_bol, None]:
        lhs = frac_refl_noredist_bol
        rhs = 1.0 - frac_refl_localredist_bol - frac_refl_horizredist_bol - frac_refl_globalredist_bol
    else:
        raise NotImplementedError

    return lhs, rhs, {'component': component}


#}
#{ Inter-component constraints

def teffratio(b, comp1, comp2, **kwargs):
    """
    :raises NotImplementedError: because this isn't yet
    """
    raise NotImplementedError

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

    parentorbit = hier.get_parent_of(component)
    parentorbit_ps = _get_system_ps(b, parentorbit)

    metawargs = component_ps.meta
    metawargs.pop('qualifier')
    mass_def = FloatParameter(qualifier='mass', value=1.0, default_unit=u.solMass, description='Mass')
    mass, created = b.get_or_create('mass', mass_def, **metawargs)

    metawargs = parentorbit_ps.meta
    metawargs.pop('qualifier')
    sma = b.get_parameter(qualifier='sma', **metawargs)
    period = b.get_parameter(qualifier='period', **metawargs)
    q = b.get_parameter(qualifier='q', **metawargs)

    G = c.G.to('solRad3 / (solMass d2)')

    if hier.get_primary_or_secondary(component) == 'primary':
        qthing = 1.0+q
    else:
        qthing = 1.0+1./q

    if solve_for in [None, mass]:

        lhs = mass
        rhs = (4*np.pi**2 * sma**3 ) / (period**2 * qthing * G)

    elif solve_for==sma:

        lhs = sma
        rhs = ((mass * period**2 * qthing * G)/(4 * np.pi**2))**(1./3)

    elif solve_for==period:

        lhs = period
        rhs = ((4 * np.pi**2 * sma**3)/(mass * qthing * G))**(1./2)

    elif solve_for==q:
        # TODO: implement this so that one mass can be solved for sma and the
        # other can be solved for q.  The tricky thing is that we actually
        # have qthing here... so we'll probably need to handle the primary
        # vs secondary case separately.
        raise NotImplementedError

    else:
        # TODO: solve for other options
        raise NotImplementedError

    return lhs, rhs, {'component': component}


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

    return lhs, rhs, {'component': component}

def potential(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the potential of a star.

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'pot' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'rpole')
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


    if parentorbit == 'component':
        # then single star (rotstar) case
        pot = component_ps.get_parameter(qualifier='pot')
        rpole = component_ps.get_parameter(qualifier='rpole')
        rotfreq = component_ps.get_parameter(qualifier='freq')

        if solve_for in [None, pot]:
            lhs = pot
            rhs = rotstarrpole2potential(rpole, rotfreq)
        elif solve_for == rpole:
            lhs = rpole
            rhs = rotstarpotential2rpole(pot, rotfreq)
        else:
            raise NotImplementedError
    else:
        # then binary (roche) case

        parentorbit_ps = _get_system_ps(b, parentorbit)

        # metawargs = component_ps.meta
        # metawargs.pop('qualifier')

        pot = component_ps.get_parameter(qualifier='pot')
        rpole = component_ps.get_parameter(qualifier='rpole')
        syncpar = component_ps.get_parameter(qualifier='syncpar')

        sma = parentorbit_ps.get_parameter(qualifier='sma')
        q = parentorbit_ps.get_parameter(qualifier='q')
        ecc = parentorbit_ps.get_parameter(qualifier='ecc')

        if solve_for in [None, pot]:
            lhs = pot
            # Eq 3.20 from PHOEBE scientific reference
            # delta = separation / a
            # at periastron: separation = a(1-e)
            # so delta at periastron = (1-e)

            # TODO: this needs to include syncpar
            # TODO: this probably should care about primary vs secondary (flip q?)

            # rhs = 1./(rpole/sma) + q / ((1-ecc)**2+(rpole/sma)**2)**0.5

            compno = {'primary': 1, 'secondary': 2}
            rhs = rocherpole2potential(rpole, q, ecc, syncpar, sma, compno[hier.get_primary_or_secondary(component)])
        elif solve_for == rpole:
            lhs = rpole
            compno = {'primary': 1, 'secondary': 2}
            rhs = rochepotential2rpole(pot, q, ecc, syncpar, sma, compno[hier.get_primary_or_secondary(component)])
        else:
            raise NotImplementedError

    return lhs, rhs, {'component': component}

def critical_potential(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the potential of a star to match the critical
    potential at L1

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'pot' should not be the derived/constrained
        parameter, provide which other parameter should be derived
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


    if parentorbit == 'component':
        raise ValueError("cannot constrain critical potential for single star")

    parentorbit_ps = _get_system_ps(b, parentorbit)

    pot = component_ps.get_parameter(qualifier='pot')
    syncpar = component_ps.get_parameter(qualifier='syncpar')
    q = parentorbit_ps.get_parameter(qualifier='q')
    ecc = parentorbit_ps.get_parameter(qualifier='ecc')

    if solve_for in [None, pot]:
        lhs = pot

        compno = {'primary': 1, 'secondary': 2}
        rhs = rochecriticalL12potential(q, ecc, syncpar, compno[hier.get_primary_or_secondary(component)])
    else:
        raise NotImplementedError

    return lhs, rhs, {'component': component}

def critical_rpole(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the rpole of a star to match the critical
    rpole at L1

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'rpole' should not be the derived/constrained
        parameter, provide which other parameter should be derived
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


    if parentorbit == 'component':
        raise ValueError("cannot constrain critical rpole for single star")

    parentorbit_ps = _get_system_ps(b, parentorbit)

    rpole = component_ps.get_parameter(qualifier='rpole')
    syncpar = component_ps.get_parameter(qualifier='syncpar')
    q = parentorbit_ps.get_parameter(qualifier='q')
    ecc = parentorbit_ps.get_parameter(qualifier='ecc')
    sma = parentorbit_ps.get_parameter(qualifier='sma')

    if solve_for in [None, rpole]:
        lhs = rpole

        compno = {'primary': 1, 'secondary': 2}
        rhs = rochecriticalL12rpole(q, ecc, syncpar, sma, compno[hier.get_primary_or_secondary(component)])
    else:
        raise NotImplementedError

    return lhs, rhs, {'component': component}


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

    return lhs, rhs, {'component': component}

def incl_aligned(b, component, solve_for=None, **kwargs):
    """
    Create a constraint for the inclination of a star to be the same as its
    parent orbit (ie aligned).

    :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
    :parameter str component: the label of the star in which this
        constraint should be built
    :parameter str solve_for:  if 'incl@star' should not be the derived/constrained
        parameter, provide which other parameter should be derived
        (ie 'incl@orbit')
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

    incl_comp = component_ps.get_parameter(qualifier='incl')
    incl_orb = parentorbit_ps.get_parameter(qualifier='incl')

    if solve_for in [None, incl_comp]:
        lhs = incl_comp
        rhs = incl_orb.to_constraint()

    elif solve_for == incl_orb:
        lhs = incl_orb
        rhs = incl_comp.to_constraint()

    else:
        raise NotImplementedError

    return lhs, rhs, {'component': component}

#}
#{ Feature constraints

def colon_deprecation(b, feature, solve_for=None, **kwargs):
    feature_ps = _get_system_ps(b, feature, context='feature')

    longitude = feature_ps.get_parameter(qualifier='long')
    colon = feature_ps.get_parameter(qualifier='colon')

    if solve_for in [None, colon]:
        lhs = colon
        rhs = 1*longitude
    else:
        lhs = longitude
        rhs = 1*colon

    return lhs, rhs, {'feature': feature}

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

    return lhs, rhs, {'component': component, 'dataset': dataset}

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

    return lhs, rhs, {'component': component, 'dataset': dataset}

#}
