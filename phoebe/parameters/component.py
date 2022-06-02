
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u
from phoebe import conf

# Each of these functions should take component, **kwargs as arguments and
# return a ParameterSet of parameters to attach to the bundle (meta-data will
# be handled by add_component) and a list of constraints - each being a tuple
# with the function in constraint.py and any arguments necessary to pass to that function)
# constraints are then responsible for creating and attaching the derived parameters
# Example: orbit calls constraint.asini with component as an argument, constraints.asini
# then creates the asini parameter, attaches it to the same orbit, and attaches the constraint

# Note: this means that cross-PS constraints should not be added by default as we do not
# know the hierarchy here

from phoebe.parameters.dataset import _ld_func_choices, _ld_coeffs_source_choices

def orbit(component, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a new orbit.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_component>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_component>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    The following constraints are returned, and will automtically be applied
    if attaching to the <phoebe.frontend.bundle.Bundle> via
    <phoebe.frontend.bundle.Bundle.add_component>:
    * <phoebe.parameters.constraint.asini>
    * <phoebe.parameters.constraint.ecosw>
    * <phoebe.parameters.constraint.esinw>
    * <phoebe.parameters.constraint.t0_perpass_supconj>
    * <phoebe.parameters.constraint.t0_ref_supconj>
    * <phoebe.parameters.constraint.mean_anom>
    * <phoebe.parameters.constraint.freq>

    In addition, some constraints are created automatically by <phoebe.frontend.bundle.Bundle.set_hierarchy>.
    For a list of these, see <phoebe.frontend.bundle.Bundle.add_constraint>.

    Arguments
    ----------
    * `period` (float/quantity, optional): Orbital period (defined at t0@system,
        sidereal: wrt the sky)
    * `period_anom` (float/quantity, optional): Anomalistic orbital period (defined
        at t0@system, anomalistic: time between two successive periastron passages).
    * `freq` (float/quantity, optional): Orbital frequency (sidereal).
    * `dpdt` (float/quantity, optional): Time derivative orbital period (anomalistic),
        where `period` is defined at t0@system.
    * `per0` (float/quantity, optional): Argument of periastron (defined at time t0@system)
    * `dperdt` (float/quantity, optional): Time derivative of argument of periastron,
        where `per0` is defined at t0@system
    * `ecc` (float, optional): eccentricity
    * `t0_perpass` (float/quantity, optional): zeropoint date at periastron passage of the
        primary component.
    * `t0_supconj` (float/quantity, optional): zeropoint date at superior conjunction of
        the primary component.
    * `t0_ref` (float/quantity, optional): zeropoint date at reference point for the
        primary component.
    * `mean_anom` (float/quantity, optional): mean anomaly.
    * `incl` (float/quantity, optional): orbital inclination.
    * `q` (float, optional): mass ratio.
    * `sma` (float/quantity, optional): semi-major axis of the orbit.
    * `long_an` (float/quantity, optional): longitude of the ascending node.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """
    params = []

    params += [FloatParameter(qualifier='period', latexfmt=r'P_\mathrm{{ {component} }}', timederiv='dpdt', value=kwargs.get('period', 1.0), default_unit=u.d, limits=(1e-6,None), description='Orbital period (defined at t0@system, sidereal: wrt the sky)')]
    params += [FloatParameter(qualifier='period_anom', visible_if='dperdt:!0.0', latexfmt=r'P_\mathrm{{anom, {component} }}', value=kwargs.get('period_anom', 1.0), default_unit=u.d, limit=(1e-6,None), description='Anomalistic orbital period (defined at t0@system, anomalistic: time between two successive periastron passages)')]
    params += [FloatParameter(qualifier='freq', latexfmt=r'f_\mathrm{{ {component} }}', value=kwargs.get('freq', 2*np.pi/3.0), default_unit=u.rad/u.d, advanced=True, description='Orbital frequency (sidereal)')]
    params += [FloatParameter(qualifier='dpdt', latexfmt=r'\dot{{P}}_\mathrm{{ {component} }}',  value=kwargs.get('dpdt', 0.0), default_unit=u.s/u.yr, advanced=True, description='Time derivative of orbital period (anomalistic), where period is defined at t0@system')]
    params += [FloatParameter(qualifier='per0', latexfmt=r'\omega_{{0, \mathrm{{ {component} }} }}', timederiv='dperdt', value=kwargs.get('per0', 0.0), default_unit=u.deg, description='Argument of periastron (defined at time t0@system)')]
    params += [FloatParameter(qualifier='dperdt', latexfmt=r'\dot{{\omega}}_{{0, \mathrm{{ {component} }}}}', value=kwargs.get('dperdt', 0.0), default_unit=u.deg/u.yr, advanced=True, description='Time derivative of argument of periastron, where per0 is defined at t0@system')]
    params += [FloatParameter(qualifier='ecc', latexfmt=r'e_\mathrm{{ {component} }}', timederiv='deccdt', value=kwargs.get('ecc', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,0.999999), description='Eccentricity')]
    # if conf.devel:
        # NOTE: if adding this back in, will need to update the t0_* constraints in builtin.py and re-enable in parameters.HierarchyParameter.is_time_dependent
        # params += [FloatParameter(qualifier='deccdt', value=kwargs.get('deccdt', 0.0), default_unit=u.dimensionless_unscaled/u.d, advanced=True, description='Eccentricity change')]
    params += [FloatParameter(qualifier='t0_perpass', latexfmt=r't_{{0, \mathrm{{ perpass }}, \mathrm{{ {component} }} }}', value=kwargs.get('t0_perpass', 0.0), default_unit=u.d, description='Zeropoint date at periastron passage of the primary component')]
    params += [FloatParameter(qualifier='t0_supconj', latexfmt=r't_{{0, \mathrm{{ supconj }}, \mathrm{{ {component} }} }}', value=kwargs.get('t0_supconj', 0.0), default_unit=u.d, description='Zeropoint date at superior conjunction of the primary component')]
    params += [FloatParameter(qualifier='t0_ref', latexfmt=r't_{{0, _\mathrm{{ ref }}, \mathrm{{ {component} }} }}', value=kwargs.get('t0_ref', 0.0), default_unit=u.d, description='Zeropoint date at reference point for the primary component')]
    params += [FloatParameter(qualifier='mean_anom', value=kwargs.get('mean_anom', 0.0), default_unit=u.deg, advanced=True, description='Mean anomaly at t0@system')]
    #params += [FloatParameter(qualifier='ph_perpass', value=kwargs.get('ph_perpass', 0.0), default_unit=u.cycle, description='Phase at periastron passage')]
    #params += [FloatParameter(qualifier='ph_supconj', value=kwargs.get('ph_supconj', 0.0), default_unit=u.cycle, description='Phase at superior conjunction')]
    #params += [FloatParameter(qualifier='ph_infconj', value=kwargs.get('ph_infconj', 0.0), default_unit=u.cycle, description='Phase at inferior conjunction')]
    #params += [FloatParameter(qualifier='t0_ph0', value=kwargs.get('t0_ph0', 0.0), default_unit=u.d, description='Zeropoint to anchor at phase=0.0')] # TODO: d vs JD
    # NOTE: the limits on inclination are from 0-180 so that the definition of superior conjunction doesn't flip
    params += [FloatParameter(qualifier='incl', latexfmt=r'i_\mathrm{{ {component} }}', timederiv='dincldt', value=kwargs.get('incl', 90.0), limits=(0.0, 180.0), default_unit=u.deg, description='Orbital inclination angle')]
    # params += [FloatParameter(qualifier='dincldt', value=kwargs.get('dincldt', 0.0), default_unit=u.deg/u.yr, description="Inclination change")]
    params += [FloatParameter(qualifier='q', latexfmt=r'q_\mathrm{{ {component} }}', value=kwargs.get('q', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Mass ratio')]
    params += [FloatParameter(qualifier='sma', latexfmt=r'a_\mathrm{{ {component} }}', value=kwargs.get('sma', 5.3), default_unit=u.solRad, limits=(0.0,None), description='Semi-major axis of the orbit (defined at time t0@system)')]
    params += [FloatParameter(qualifier='long_an', latexfmt=r'\Omega_\mathrm{{ {component} }}', value=kwargs.get('long_an', 0.0), default_unit=u.deg, description='Longitude of the ascending node')]

    constraints = []
    constraints += [(constraint.asini, component)]
    constraints += [(constraint.t0_perpass_supconj, component)]
    constraints += [(constraint.t0_ref_supconj, component)]
    constraints += [(constraint.period_anom, component)]
    constraints += [(constraint.mean_anom, component)]
    #constraints += [(constraint.ph_perpass, component)]
    #constraints += [(constraint.ph_supconj, component)]
    #constraints += [(constraint.ph_infconj, component)]
    constraints += [(constraint.ecosw, component)]
    constraints += [(constraint.esinw, component)]
    constraints += [(constraint.freq, component)]

    return ParameterSet(params), constraints


def star(component, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a new star.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_component>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_component>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    The following constraints are returned, and will automtically be applied
    if attaching to the <phoebe.frontend.bundle.Bundle> via
    <phoebe.frontend.bundle.Bundle.add_component>:
    * <phoebe.parameters.constraint.freq>
    * <phoebe.parameters.constraint.irrad_frac>
    * <phoebe.parameters.constraint.logg>

    In addition, some constraints are created automatically by <phoebe.frontend.bundle.Bundle.set_hierarchy>.
    For a list of these, see <phoebe.frontend.bundle.Bundle.add_constraint>.

    Arguments
    ----------
    * `requiv` (float/quantity, optional): equivalent radius.
    * `requiv_max` (float/quantity, optional): critical (maximum) value of the
        equivalent radius for the given morphology.
    * `requiv_min` (float/quantity, optional): critical (minimum) value of the
        equivalent radius for the given morphology.
    * `teff` (float/quantity, optional): mean effective temperature.
    * `abun` (float, optional): abundance/metallicity
    * `syncpar` (float, optional): synchronicity parameter.
    * `period` (float/quantity, optional): rotation period (wrt the sky).
    * `freq` (float/quantity, optional): rotation frequency (wrt the sky).
    * `pitch` (float/quantity, optional): pitch of the stellar rotation axis wrt
        the orbital inclination.
    * `yaw` (float/quantity, optional): yaw of the stellar rotation axis wrt
        the orbital longitdue of ascending node.
    * `incl` (float/quantity, optional): inclination of the stellar rotation axis
    * `long_an` (float/quantity, optional): longitude of the ascending node (ie.
        equator) of the star.
    * `gravb_bol` (float, optional): bolometric gravity brightening.
    * `irrad_frac_refl_bol` (float, optional): ratio of incident
        bolometric light that is used for reflection (heating without
        redistribution).
    * `irrad_frac_refl_lost` (float, optional): ratio of incident
        bolometric light that is lost/ignored.
    * `ld_mode_bol` (string, optional, default='lookup'): mode to use for handling
        bolometric limb-darkening.  Note that unlike passband limb-darkening,
        'lookup' will apply a single set of coefficients per-component instead
        of per-element.
    * `ld_func_bol` (string, optional): bolometric limb-darkening model.
    * `ld_coeffs_source_bol` (string, optional, default='auto'): source for
        bolometric limb-darkening coefficients ('auto' to interpolate from the
        applicable table according to the 'atm' parameter, or the name of a
        specific atmosphere table).  Only applicable if `ld_mode_bol` is
        'lookup'.
    * `ld_coeffs_bol` (list/array, optional): bolometric limb-darkening
        coefficients.  Only applicable if `ld_mode_bol` is 'manual'.
    * `mass` (float/quantity, optional): mass of the star.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [FloatParameter(qualifier='requiv', latexfmt=r'R_{{ \mathrm{{ equiv }}, \mathrm{{ {component} }} }}', value=kwargs.get('requiv', 1.0), default_unit=u.solRad, limits=(1e-6,None), description='Equivalent radius')]
    params += [FloatParameter(qualifier='requiv_max', latexfmt=r'R_{{ \mathrm{{ equiv }}, \mathrm{{ max }}, \mathrm{{ {component} }} }}', value=kwargs.get('requiv_max', 10.0), default_unit=u.solRad, limits=(0.0, None), description='Critical (maximum) value of the equivalent radius for the given morphology')]
    params += [FloatParameter(qualifier='requiv_min', latexfmt=r'R_{{ \mathrm{{ equiv }}, \mathrm{{ min }}, \mathrm{{ {component} }} }}', visible_if='hierarchy.is_contact_binary:True', value=kwargs.get('requiv_min', 0.1), default_unit=u.solRad, limits=(0.0, None), description='Critical (minimum) value of the equivalent radius for the given morphology')]
    params += [FloatParameter(qualifier='teff', latexfmt=r'T_{{ \mathrm{{ eff }}, \mathrm{{ {component} }} }}', value=kwargs.get('teff', 6000.), default_unit=u.K, limits=(300.0,None), description='Mean effective temperature')]
    params += [FloatParameter(qualifier='abun', visible_if='hierarchy.is_contact_binary:False', value=kwargs.get('abun', 0.), default_unit=u.dimensionless_unscaled, description='Abundance/Metallicity')]   # TODO: correct units??? check if log or not? (logabun = 0)

    params += [FloatParameter(qualifier='logg', latexfmt=r'\mathrm{{log}}g_\mathrm{{ {component} }}', value=1.0, default_unit=u.dimensionless_unscaled, description='logg at requiv')]

    params += [FloatParameter(qualifier='syncpar',  latexfmt=r'F_\mathrm{{ {component} }}', visible_if='hierarchy.is_binary:True', value=kwargs.get('syncpar', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Synchronicity parameter')]
    params += [FloatParameter(qualifier='period',  latexfmt=r'P_\mathrm{{ {component} }}', value=kwargs.get('period', 1.0), default_unit=u.d, limits=(1e-6,None), advanced=True, description='Rotation period (wrt the sky)')]
    params += [FloatParameter(qualifier='freq',  latexfmt=r'f_\mathrm{{ {component} }}', value=kwargs.get('freq', 2*np.pi), default_unit=u.rad/u.d, limits=(0.0,None), advanced=True, description='Rotation frequency (wrt the sky)')]

    params += [FloatParameter(qualifier='pitch', visible_if='hierarchy.is_contact_binary:False,hierarchy.is_binary:True', value=kwargs.get('pitch', 0), default_unit=u.deg, advanced=True, description='Pitch of the stellar rotation axis wrt the orbital inclination')]
    params += [FloatParameter(qualifier='yaw', visible_if='hierarchy.is_contact_binary:False,hierarchy.is_binary:True', value=kwargs.get('yaw', 0), default_unit=u.deg, advanced=True, description='Yaw of the stellar rotation axis wrt the orbital longitude of ascending node')]

    params += [FloatParameter(qualifier='incl', latexfmt=r'i_\mathrm{{ {component} }}', visible_if='hierarchy.is_contact_binary:False', value=kwargs.get('incl', 90), default_unit=u.deg, advanced=True, description='Inclination of the stellar rotation axis')]
    params += [FloatParameter(qualifier='long_an', visible_if='hierarchy.is_contact_binary:False', value=kwargs.get('long_an', 0.0), default_unit=u.deg, advanced=True, description='Longitude of the ascending node (ie. equator) of the star')]

    # params += [FloatParameter(qualifier='vsini', value=kwargs.get('vsini', 1), default_unit=u.km/u.s, description='Projected maximum rotational velocity')]

    # params += [ChoiceParameter(qualifier='gravblaw_bol', value=kwargs.get('gravblaw_bol', 'zeipel'), choices=['zeipel', 'espinosa', 'claret'], description='Gravity brightening law')]

    # params += [FloatParameter(qualifier='gravb_bol', visible_if='hierarchy.is_contact_binary:False', value=kwargs.get('gravb_bol', 0.32), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='Bolometric gravity brightening')]
    params += [FloatParameter(qualifier='gravb_bol', latexfmt=r'\beta_{{ \mathrm{{bol}}, \mathrm{{ {component} }} }}', value=kwargs.get('gravb_bol', 0.32), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='Bolometric gravity brightening')]

    # also see constraint below
    params += [FloatParameter(qualifier='irrad_frac_refl_bol', latexfmt=r'A_{{ \mathrm{{bol}}, \mathrm{{ {component} }} }}', value=kwargs.get('irrad_frac_refl_bol', 0.6), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of incident bolometric light that is used for reflection/irradiation (heating without redistribution)')]
    params += [FloatParameter(qualifier='irrad_frac_lost_bol', value=kwargs.get('irrad_frac_lost_bol', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0, 1.0), advanced=True, description='ratio of incident bolometric light that is lost/ignored')]

    params += [ChoiceParameter(qualifier='ld_mode_bol',
                               value=kwargs.get('ld_mode_bol', 'lookup'), choices=['lookup', 'manual'],
                               description='Mode to use for bolometric limb-darkening (used only for irradiation).')]

    params += [ChoiceParameter(qualifier='ld_func_bol',
                              value=kwargs.get('ld_func_bol', 'logarithmic'),
                              choices=_ld_func_choices,
                              description='Bolometric limb darkening model (used only for irradiation).')]

    params += [ChoiceParameter(visible_if='ld_mode_bol:lookup', qualifier='ld_coeffs_source_bol',
                               value=kwargs.get('ld_coeffs_source_bol', 'auto'), choices=_ld_coeffs_source_choices,
                               advanced=True,
                               description='Source for bolometric limb darkening coefficients (used only for irradiation; \'auto\' to interpolate from the applicable table according to the \'atm\' parameter, or the name of a specific atmosphere table)')]


    params += [FloatArrayParameter(visible_if='ld_mode_bol:manual', qualifier='ld_coeffs_bol',
                                   latexfmt=r'\mathrm{{ ldc }}_\mathrm{{ bol, {component} }}',
                                   value=kwargs.get('ld_coeffs_bol', [0.5, 0.5]),
                                   default_unit=u.dimensionless_unscaled,
                                   required_shape=[None],
                                   description='Bolometric limb darkening coefficients (used only for irradiation).')]

    params += [FloatParameter(qualifier='mass', latexfmt=r'M_\mathrm{{ {component} }}', value=kwargs.get('mass', 1.0), default_unit=u.solMass, description='Mass')]

    constraints = []
    # constraints handled by set_hierarchy:
    # - requiv_detached_max
    # - mass
    # - comp_sma
    # - asini
    # - rotation_period
    # - pitch
    # - yaw
    # - teffratio
    # - requivratio
    # - requivsumfrac


    constraints += [(constraint.freq, component)]
    constraints += [(constraint.logg, component)]
    constraints += [(constraint.irrad_frac, component)]

    return ParameterSet(params), constraints

def envelope(component, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a new envelope.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_component>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_component>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    In addition, some constraints are created automatically by <phoebe.frontend.bundle.Bundle.set_hierarchy>.
    For a list of these, see <phoebe.frontend.bundle.Bundle.add_constraint>.

    Arguments
    ----------
    * `abun` (float, optional): abundance/metallicity.
    * `fillout_factor` (float, optional): fillout-factor of the envelope.
    * `pot` (float, optional): potential of the envelope.
    * `pot_min` (float, optional): critical (minimum) value of the potential to
        remain a contact.
    * `pot_max` (float, optional): critical (maximum) value of the potential to
        remain a contact.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """
    params = []

    params += [FloatParameter(qualifier='abun', value=kwargs.get('abun', 0.), default_unit=u.dimensionless_unscaled, description='Abundance/Metallicity')]   # TODO: correct units??? check if log or not? (logabun = 0)
    # params += [FloatParameter(qualifier='gravb_bol', value=kwargs.get('gravb_bol', 0.32), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='Bolometric gravity brightening')]


    # params += [FloatParameter(qualifier='frac_refl_bol', value=kwargs.get('frac_refl_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of incident bolometric light that is used for reflection (heating without redistribution)')]
    # params += [FloatParameter(qualifier='frac_heat_bol', value=kwargs.get('frac_heat_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of incident bolometric light that is used for heating')]
    # params += [FloatParameter(qualifier='frac_scatt_bol', value=kwargs.get('frac_scatt_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of bolometric incident light that is scattered')]
    # params += [FloatParameter(qualifier='frac_lost_bol', value=kwargs.get('frac_lost_bol', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0, 1.0), description='ratio of incident bolometric light that is lost/ignored')]


    params += [FloatParameter(qualifier='fillout_factor', latexfmt=r'\mathrm{{FF}}_\mathrm{{ {component} }}', value=kwargs.get('fillout_factor', 0.5), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='Fillout-factor of the envelope')]
    params += [FloatParameter(qualifier='pot', latexfmt=r'\Omega_\mathrm{{ {component} }}', value=kwargs.get('pot', 3.5), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Potential of the envelope (from the primary component\'s reference)')]
    params += [FloatParameter(qualifier='pot_min', latexfmt=r'\Omega_\mathrm{{ min,  {component} }}', value=kwargs.get('pot_min', 3.5), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Critical (minimum) value of the potential to remain a contact')]
    params += [FloatParameter(qualifier='pot_max', latexfmt=r'\Omega_\mathrm{{ max, {component} }}', value=kwargs.get('pot_max', 3.5), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Critical (maximum) value of the potential to remain a contact')]
    # params += [FloatParameter(qualifier='intens_coeff1', value=kwargs.get('intens_coeff1', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    # params += [FloatParameter(qualifier='intens_coeff2', value=kwargs.get('intens_coeff2', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    # params += [FloatParameter(qualifier='intens_coeff3', value=kwargs.get('intens_coeff3', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    # params += [FloatParameter(qualifier='intens_coeff4', value=kwargs.get('intens_coeff4', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    # params += [FloatParameter(qualifier='intens_coeff5', value=kwargs.get('intens_coeff5', 1.0), default_unit=u.dimensionless_unscaled, description='')]

    # params += [ChoiceParameter(qualifier='ld_func_bol', value=kwargs.get('ld_func_bol', 'logarithmic'), choices=_ld_func_choices_no_interp, description='Bolometric limb darkening model')]
    # params += [FloatArrayParameter(qualifier='ld_coeffs_bol', value=kwargs.get('ld_coeffs_bol', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, description='Bolometric limb darkening coefficients')]

    constraints = []

    # constraints handled by set hierarchy:
    # potential_contact_min/max
    # requiv_contact_min/max

    return ParameterSet(params), constraints


# del deepcopy
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
