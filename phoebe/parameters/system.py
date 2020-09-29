
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
    * `vgamma` (float/quantity, optional): Constant barycentric systemic velocity
        (in the direction of positive RV or negative vz)
    * `ebv` (float, optional, default=0): extinction E(B-V).
    * `Av` (float, optional, default=0): extinction Av.
    * `Rv` (float, optional, default=3.1): extinction law parameter.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet) ParameterSet of all created Parameters.
    """
    params = []
    constraints = []

    params += [FloatParameter(qualifier='t0', latexfmt=r't_0', value=kwargs.get('t0', 0.0), default_unit=u.d, description='Time at which all values are provided.  For values with time-derivatives, this defines their zero-point.')]

    # TODO: re-enable these once they're incorporated into orbits (dynamics) correctly.
    params += [FloatParameter(qualifier='ra', latexfmt=r'\alpha', value=kwargs.get('ra', 0.0), default_unit=u.deg, advanced=True, description='Right ascension')]
    params += [FloatParameter(qualifier='dec', latexfmt=r'\delta', value=kwargs.get('dec', 0.0), default_unit=u.deg, advanced=True, description='Declination')]
    # params += [StringParameter(qualifier='epoch', value=kwargs.get('epoch', 'J2000'), advanced=True, description='Epoch of coordinates')]
    #params += [FloatParameter(qualifier='pmra', value=kwargs.get('pmra', 0.0), default_unit=u.mas/u.yr, description='Proper motion in right ascension')]
    #params += [FloatParameter(qualifier='pmdec', value=kwargs.get('pmdec', 0.0), default_unit=u.mas/u.yr, description='Proper motion in declination')]

    params += [FloatParameter(qualifier='distance', latexfmt=r'd', value=kwargs.get('distance', 1.0), default_unit=u.m, description='Distance to the system')]
    params += [FloatParameter(qualifier='vgamma', latexfmt=r'v_\gamma', value=kwargs.get('vgamma', 0.0), default_unit=u.km/u.s, description='Constant barycentric systemic velocity (in the direction of positive RV or negative vz)')]

    params += [FloatParameter(qualifier='ebv', latexfmt=r'E(B-V)', value=kwargs.get('ebv', 0.0), default_unit=u.dimensionless_unscaled, limits=(None, None), description='Extinction E(B-V)')]
    params += [FloatParameter(qualifier='Av', latexfmt=r'A_v', value=kwargs.get('Av', 0.0), default_unit=u.dimensionless_unscaled, limits=(None, None), description='Extinction Av')]
    params += [FloatParameter(qualifier='Rv', latexfmt=r'R_v', value=kwargs.get('Rv', 3.1), default_unit=u.dimensionless_unscaled, limits=(None, None), description='Extinction law parameter')]
    constraints +=[(constraint.extinction,)]

    return ParameterSet(params), constraints
