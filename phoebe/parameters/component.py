
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

_ld_func_choices_no_interp = ['linear', 'logarithmic', 'quadratic', 'square_root', 'power']

def orbit(component, **kwargs):
    """
    Create parameters for a new orbit.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_component`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    params = []

    #~ params += [ObjrefParameter(value=component)]
    params += [FloatParameter(qualifier='period', timederiv='dpdt', value=kwargs.get('period', 3.0), default_unit=u.d, limits=(0.0,None), description='Orbital period')]
    params += [FloatParameter(qualifier='freq', value=kwargs.get('freq', 2*np.pi/3.0), default_unit=u.rad/u.d, description='Orbital frequency')]
    params += [FloatParameter(qualifier='dpdt', value=kwargs.get('dpdt', 0.0), default_unit=u.s/u.yr, description='Period change')]
    params += [FloatParameter(qualifier='per0', timederiv='dperdt', value=kwargs.get('per0', 0.0), default_unit=u.deg, description='Periastron')]
    params += [FloatParameter(qualifier='dperdt', value=kwargs.get('dperdt', 0.0), default_unit=u.deg/u.yr, description='Periastron change')]
    params += [FloatParameter(qualifier='ecc', timederiv='deccdt', value=kwargs.get('ecc', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='Eccentricity')]
    if conf.devel:
        params += [FloatParameter(qualifier='deccdt', value=kwargs.get('deccdt', 0.0), default_unit=u.dimensionless_unscaled/u.d, description='Eccentricity change')]
    params += [FloatParameter(qualifier='t0_perpass', value=kwargs.get('t0_perpass', 0.0), default_unit=u.d, description='Zeropoint date at periastron passage')]  # TODO: d vs JD
    params += [FloatParameter(qualifier='t0_supconj', value=kwargs.get('t0_supconj', 0.0), default_unit=u.d, description='Zeropoint date at superior conjunction')]  # TODO: d vs JD
    params += [FloatParameter(qualifier='mean_anom', value=kwargs.get('mean_anom', 0.0), default_unit=u.deg, description='Mean anomaly')]
    #params += [FloatParameter(qualifier='ph_perpass', value=kwargs.get('ph_perpass', 0.0), default_unit=u.cycle, description='Phase at periastron passage')]
    #params += [FloatParameter(qualifier='ph_supconj', value=kwargs.get('ph_supconj', 0.0), default_unit=u.cycle, description='Phase at superior conjunction')]
    #params += [FloatParameter(qualifier='ph_infconj', value=kwargs.get('ph_infconj', 0.0), default_unit=u.cycle, description='Phase at inferior conjunction')]
    #params += [FloatParameter(qualifier='t0_ph0', value=kwargs.get('t0_ph0', 0.0), default_unit=u.d, description='Zeropoint to anchor at phase=0.0')] # TODO: d vs JD
    params += [FloatParameter(qualifier='incl', timederiv='dincldt', value=kwargs.get('incl', 90.0), default_unit=u.deg, description='Orbital inclination angle')]
    # params += [FloatParameter(qualifier='dincldt', value=kwargs.get('dincldt', 0.0), default_unit=u.deg/u.yr, description="Inclination change")]
    params += [FloatParameter(qualifier='phshift', value=kwargs.get('phshift', 0.0), default_unit=u.dimensionless_unscaled, description='Phase shift')]
    params += [FloatParameter(qualifier='q', value=kwargs.get('q', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Mass ratio')]
    params += [FloatParameter(qualifier='sma', value=kwargs.get('sma', 8.0), default_unit=u.solRad, limits=(0.0,None), description='Semi major axis of the orbit')]
    params += [FloatParameter(qualifier='long_an', value=kwargs.get('long_an', 0.0), default_unit=u.deg, description='Longitude of the ascending node')]

    constraints = []
    constraints += [(constraint.asini, component)]
    constraints += [(constraint.t0, component)]
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
    Create parameters for a new star.

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_component`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """

    params = []

    #~ params += [ObjrefParameter(value=component)]
    params += [FloatParameter(qualifier='rpole', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('rpole', 1.0), default_unit=u.solRad, limits=(0.0,None), description='Polar radius at periastron')]
    params += [FloatParameter(qualifier='pot', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('pot', 4.0), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Potential at periastron')]   # TODO: correct units???
    params += [FloatParameter(qualifier='teff', value=kwargs.get('teff', 10000.), default_unit=u.K, limits=(0.0,None), description='Mean effective temperature')]
    params += [FloatParameter(qualifier='abun', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('abun', 0.), default_unit=u.dimensionless_unscaled, description='Metallicity')]   # TODO: correct units??? check if log or not? (logabun = 0)

    params += [FloatParameter(qualifier='syncpar', visible_if='hierarchy.is_overcontact:False,hierarchy.is_binary:True', value=kwargs.get('syncpar', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Synchronicity parameter')]
    params += [FloatParameter(qualifier='period', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('period', 1.0), default_unit=u.d, limits=(0.0,None), description='Rotation period')]
    params += [FloatParameter(qualifier='freq', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('freq', 2*np.pi), default_unit=u.rad/u.d, limits=(0.0,None), description='Rotation frequency')]

    params += [FloatParameter(qualifier='incl', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('incl', 90), default_unit=u.deg, description='Inclination of the stellar rotation axis')]
    # params += [FloatParameter(qualifier='pitch', value=kwargs.get('pitch', 90), default_unit=u.deg, description='Pitch of the stellar rotation axis')]
    # params += [FloatParameter(qualifier='yaw', value=kwargs.get('yaw', 0), default_unit=u.deg, description='Yaw of the stellar rotation axis')]
    # params += [FloatParameter(qualifier='vsini', value=kwargs.get('vsini', 1), default_unit=u.km/u.s, description='Projected maximum rotational velocity')]

    # params += [ChoiceParameter(qualifier='gravblaw_bol', value=kwargs.get('gravblaw_bol', 'zeipel'), choices=['zeipel', 'espinosa', 'claret'], description='Gravity brightening law')]

    params += [FloatParameter(qualifier='gravb_bol', value=kwargs.get('gravb_bol', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='Bolometric gravity brightening')]

    # params += [FloatParameter(qualifier='frac_bol', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('frac_bol', 0.4), default_unit=u.dimensionless_unscaled, description='Bolometric albedo (1-alb heating, alb reflected)')]
    params += [FloatParameter(qualifier='frac_refl_bol', value=kwargs.get('frac_refl_bol', 0.6), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of incident bolometric light that is used for reflection (heating without redistribution)')]

    # if conf.devel:
        # also see constraint below
        # these currently don't do anything until libphoebe's reflection will take these values
        # params += [FloatParameter(qualifier='frac_refl_noredist_bol', value=kwargs.get('frac_refl_noredist_bol', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of frac_refl_bol that is not redistributed')]
        # params += [FloatParameter(qualifier='frac_refl_localredist_bol', value=kwargs.get('frac_refl_localredist_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of frac_refl_bol that is locally redistributed')]
        # params += [FloatParameter(qualifier='frac_refl_horizredist_bol', value=kwargs.get('frac_refl_horizredist_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of frac_refl_bol that is horizontally (along constant latitude) redistributed')]
        # params += [FloatParameter(qualifier='frac_refl_globalredist_bol', value=kwargs.get('frac_refl_globalredist_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of frac_refl_bol that is globally redistributed')]


    # params += [FloatParameter(qualifier='frac_scatt_bol', visible_if='hierarchy.is_overcontact:False', value=kwargs.get('frac_scatt_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of bolometric incident light that is scattered')]
    params += [FloatParameter(qualifier='frac_lost_bol', value=kwargs.get('frac_lost_bol', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0, 1.0), description='ratio of incident bolometric light that is lost/ignored')]

    # params += [FloatParameter(qualifier='redist', value=kwargs.get('redist', 0.0), unit=u.dimensionless_unscaled, description='Global redist par (1-redist) local heating, redist global heating')]
    # params += [FloatParameter(qualifier='redisth', value=kwargs.get('redisth', 0.0), unit=u.dimensionless_unscaled, description='Horizontal redist par (redisth/redist) horizontally spread')]

    # TODO: allow for 'interp' as choice, make default, and set visible_if for ld_coeffs_bol (see ld_coeffs in dataset.py)
    params += [ChoiceParameter(qualifier='ld_func_bol', value=kwargs.get('ld_func_bol', 'logarithmic'), choices=_ld_func_choices_no_interp, description='Bolometric limb darkening model')]
    params += [FloatArrayParameter(qualifier='ld_coeffs_bol', value=kwargs.get('ld_coeffs_bol', [0.5, 0.5]), default_unit=u.dimensionless_unscaled, description='Bolometric limb darkening coefficients')]


    params += [FloatParameter(qualifier='mass', value=kwargs.get('mass', 1.0), default_unit=u.solMass, description='Mass')]


    # TODO: add others or move to a bol_dep (in which case create the bol_dep now)?

    constraints = []
    # constraints handled by set_hierarchy:
    # - potential
    # - mass
    # - comp_sma
    # - rotation_period
    # - incl_aligned

    constraints += [(constraint.freq, component)]
    constraints += [(constraint.refl, component)]
    # if conf.devel:
        # constraints += [(constraint.reflredist, component)]


    return ParameterSet(params), constraints

def envelope(component, **kwargs):
    """
    Create parameters for an envelope (usually will be attached to two stars solRad
        that they can share a common-envelope)

    Generally, this will be used as an input to the kind argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_component`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    if not conf.devel:
        raise NotImplementedError("'envelope' component (ie overcontacts) not officially supported for this release.  Enable developer mode to test.")


    params = []

    params += [FloatParameter(qualifier='abun', value=kwargs.get('abun', 0.), default_unit=u.dimensionless_unscaled, description='Metallicity')]   # TODO: correct units??? check if log or not? (logabun = 0)

    # params += [FloatParameter(qualifier='frac_refl_bol', value=kwargs.get('frac_refl_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of incident bolometric light that is used for reflection (heating without redistribution)')]
    # params += [FloatParameter(qualifier='frac_heat_bol', value=kwargs.get('frac_heat_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of incident bolometric light that is used for heating')]
    # params += [FloatParameter(qualifier='frac_scatt_bol', value=kwargs.get('frac_scatt_bol', 0.0), default_unit=u.dimensionless_unscaled, limits=(0.0,1.0), description='ratio of bolometric incident light that is scattered')]
    # params += [FloatParameter(qualifier='frac_lost_bol', value=kwargs.get('frac_lost_bol', 1.0), default_unit=u.dimensionless_unscaled, limits=(0.0, 1.0), description='ratio of incident bolometric light that is lost/ignored')]


    params += [FloatParameter(qualifier='pot', value=kwargs.get('pot', 3.5), default_unit=u.dimensionless_unscaled, limits=(0.0,None), description='Potential at periastron (in the primary component\'s reference frame')]   # TODO: correct units???
    params += [FloatParameter(qualifier='intens_coeff1', value=kwargs.get('intens_coeff1', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    params += [FloatParameter(qualifier='intens_coeff2', value=kwargs.get('intens_coeff2', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    params += [FloatParameter(qualifier='intens_coeff3', value=kwargs.get('intens_coeff3', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    params += [FloatParameter(qualifier='intens_coeff4', value=kwargs.get('intens_coeff4', 1.0), default_unit=u.dimensionless_unscaled, description='')]
    params += [FloatParameter(qualifier='intens_coeff5', value=kwargs.get('intens_coeff5', 1.0), default_unit=u.dimensionless_unscaled, description='')]


    constraints = []

    return ParameterSet(params), constraints
