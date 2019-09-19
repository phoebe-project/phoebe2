

from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _component_allowed_for_feature(feature_kind, component_kind):
    _allowed = {}
    _allowed['spot'] = ['star', 'envelope']
    _allowed['pulsation'] = ['star', 'envelope']

    return component_kind in _allowed[feature_kind]

def spot(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a spot feature.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `colat` (float/quantity, optional): colatitude of the center of the spot
        wrt spin axis.
    * `long` (float/quantity, optional): longitude of the center of the spot wrt
        spin axis.
    * `radius` (float/quantity, optional): angular radius of the spot.
    * `relteff` (float/quantity, optional): temperature of the spot relative
        to the intrinsic temperature.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [FloatParameter(qualifier="colat", value=kwargs.get('colat', 0.0), default_unit=u.deg, description='Colatitude of the center of the spot wrt spin axis')]
    params += [FloatParameter(qualifier="long", value=kwargs.get('long', 0.0), default_unit=u.deg, description='Longitude of the center of the spot wrt spin axis')]
    params += [FloatParameter(qualifier='radius', value=kwargs.get('radius', 1.0), default_unit=u.deg, description='Angular radius of the spot')]
    # params += [FloatParameter(qualifier='area', value=kwargs.get('area', 1.0), default_unit=u.solRad, description='Surface area of the spot')]

    params += [FloatParameter(qualifier='relteff', value=kwargs.get('relteff', 1.0), limits=(0.,None), default_unit=u.dimensionless_unscaled, description='Temperature of the spot relative to the intrinsic temperature')]
    # params += [FloatParameter(qualifier='teff', value=kwargs.get('teff', 10000), default_unit=u.K, description='Temperature of the spot')]

    constraints = []

    return ParameterSet(params), constraints


# del deepcopy
# del _component_allowed_for_feature
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
