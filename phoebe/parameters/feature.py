

from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u
from phoebe import conf

def spot(**kwargs):
    """
    Create parameters for a spot

    Generally, this will be used as input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_feature`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet`
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

def pulsation(**kwargs):
    """
    Create parameters for a pulsation feature

    Generally, this will be used as input to the method argument in
    :meth:`phoebe.frontend.bundle.Bundle.add_feature`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet`
    """
    if not conf.devel:
        raise NotImplementedError("'pulsation' feature not officially supported for this release.  Enable developer mode to test.")


    params = []

    params += [FloatParameter(qualifier='radamp', value=kwargs.get('radamp', 0.1), default_unit=u.dimensionless_unscaled, description='Relative radial amplitude of the pulsations')]
    params += [FloatParameter(qualifier='freq', value=kwargs.get('freq', 1.0), default_unit=u.d**-1, description='Frequency of the pulsations')]
    params += [FloatParameter(qualifier='l', value=kwargs.get('l', 1.0), default_unit=u.dimensionless_unscaled, description='Non-radial degree l')]
    params += [FloatParameter(qualifier='m', value=kwargs.get('m', 1.0), default_unit=u.dimensionless_unscaled, description='Azimuthal order m')]
    params += [BoolParameter(qualifier='teffext', value=kwargs.get('teffext', False), description='Switch to denote whether Teffs are provided by the external code')]


    constraints = []

    return ParameterSet(params), constraints
