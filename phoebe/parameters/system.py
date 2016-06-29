
from phoebe.parameters import *
from phoebe import u


def system(**kwargs):
    """
    Generally, this will automatically be added to a newly initialized
    :class:`phoebe.frontend.bundle.Bundle`

    :parameter **kwargs: defaults for the values of any of the parameters
    :return: a :class:`phoebe.parameters.parameters.ParameterSet` of all newly
        created :class:`phoebe.parameters.parameters.Parameter`s
    """
    params = []

    params += [FloatParameter(qualifier='t0', value=kwargs.get('t0', 0.0), default_unit=u.d, description='Time at which all values are provided')]

    # TODO: re-enable these once they're incorporated into orbits (dynamics) correctly.
    params += [FloatParameter(qualifier='ra', value=kwargs.get('ra', 0.0), default_unit=u.deg, description='Right ascension')]
    params += [FloatParameter(qualifier='dec', value=kwargs.get('dec', 0.0), default_unit=u.deg, description='Declination')]
    params += [StringParameter(qualifier='epoch', value=kwargs.get('epoch', 'J2000'), description='Epoch of coordinates')]
    #params += [FloatParameter(qualifier='pmra', value=kwargs.get('pmra', 0.0), default_unit=u.mas/u.yr, description='Proper motion in right ascension')]
    #params += [FloatParameter(qualifier='pmdec', value=kwargs.get('pmdec', 0.0), default_unit=u.mas/u.yr, description='Proper motion in declination')]

    params += [FloatParameter(qualifier='distance', value=kwargs.get('distance', 1.0), default_unit=u.m, description='Distance to the system')]
    params += [FloatParameter(qualifier='vgamma', value=kwargs.get('vgamma', 0.0), default_unit=u.km/u.s, description='Systemic velocity')]

    return ParameterSet(params)
