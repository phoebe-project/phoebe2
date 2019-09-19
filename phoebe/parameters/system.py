
from phoebe.parameters import *
from phoebe import u

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def system(**kwargs):
    """
    Generally, this will automatically be added to a newly initialized
    <phoebe.frontend.bundle.Bundle>.

    Arguments
    -----------
    * `t0` (float/quantity, optional): time at which all values are defined.
    * `ra` (float/quantity, optional): right ascension.
    * `dec` (float/quantity, optiona): declination.
    * `epoch` (string, optional): epoch of `ra` and `dec`.
    * `distance` (float/quantity, optional): distance to the system.
    * `vgamma` (float/quantity, optional): systemic velocity.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet) ParameterSet of all created Parameters.
    """
    params = []

    params += [FloatParameter(qualifier='t0', value=kwargs.get('t0', 0.0), default_unit=u.d, description='Time at which all values are provided')]

    # TODO: re-enable these once they're incorporated into orbits (dynamics) correctly.
    params += [FloatParameter(qualifier='ra', value=kwargs.get('ra', 0.0), default_unit=u.deg, advanced=True, description='Right ascension')]
    params += [FloatParameter(qualifier='dec', value=kwargs.get('dec', 0.0), default_unit=u.deg, advanced=True, description='Declination')]
    # params += [StringParameter(qualifier='epoch', value=kwargs.get('epoch', 'J2000'), advanced=True, description='Epoch of coordinates')]
    #params += [FloatParameter(qualifier='pmra', value=kwargs.get('pmra', 0.0), default_unit=u.mas/u.yr, description='Proper motion in right ascension')]
    #params += [FloatParameter(qualifier='pmdec', value=kwargs.get('pmdec', 0.0), default_unit=u.mas/u.yr, description='Proper motion in declination')]

    params += [FloatParameter(qualifier='distance', value=kwargs.get('distance', 1.0), default_unit=u.m, description='Distance to the system')]
    params += [FloatParameter(qualifier='vgamma', value=kwargs.get('vgamma', 0.0), default_unit=u.km/u.s, description='Systemic velocity (in the direction of positive RV or negative vz)')]

    return ParameterSet(params)
