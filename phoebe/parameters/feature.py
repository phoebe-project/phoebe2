
from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u

def spot(**kwargs):
    """
    Create parameters for a spot

    Generally, this will be used as input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_feature`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.Parameters`s
    """

    params = []

    params += [FloatParameter(qualifier="colat", value=kwargs.get('colat', 0.0), default_unit=u.deg, description='Colatitude of the center of the spot')]
    params += [FloatParameter(qualifier="colon", value=kwargs.get('colon', 0.0), default_unit=u.deg, description='Colongitude of the center of the spot')]
    params += [FloatParameter(qualifier='radius', value=kwargs.get('radius', 1.0), default_unit=u.deg, description='Angular radius of the spot')]
    # params += [FloatParameter(qualifier='area', value=kwargs.get('area', 1.0), default_unit=u.solRad, description='Surface area of the spot')]

    params += [FloatParameter(qualifier='relteff', value=kwargs.get('relteff', 1.0), default_unit=u.dimensionless_unscaled, description='Temperature of the spot relative to the intrinsic temperature')]
    # params += [FloatParameter(qualifier='teff', value=kwargs.get('teff', 10000), deafault_unit=u.K, description='Temperature of the spot')]

    constraints = []

    return ParameterSet(params), constraints
