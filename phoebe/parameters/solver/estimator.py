
from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def lc_eclipse_geometry(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    light curve eclipse geometry esimator.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('estimator.lc_eclipse_geometry')
    b.run_solver(kind='lc_eclipse_geometry')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [ChoiceParameter(qualifier='lc', value=kwargs.get('lc', ''), choices=[''], description='Light curve dataset to use to extract eclipse geometry')]
    params += [ChoiceParameter(qualifier='orbit', value=kwargs.get('orbit', ''), choices=[''], description='Orbit to use for phasing the light curve referenced in the dataset parameter')]

    params += [BoolParameter(qualifier='t0_near_times', value=kwargs.get('t0_near_times', True), description='Whether the returned value for t0_supconj should be forced to be in the range of the referenced observations.')]

    return ParameterSet(params)

def bls_period(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    BLS (box least squares) period estimator using astropy.  See
    https://docs.astropy.org/en/stable/timeseries/bls.html for more details.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('estimator.bls_period')
    b.run_solver(kind='bls_period')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [ChoiceParameter(qualifier='lc', value=kwargs.get('lc', ''), choices=[''], description='Light curve dataset to use to run the BLS algorithm')]
    params += [ChoiceParameter(qualifier='component', value=kwargs.get('component', ''), choices=[''], description='Component to apply the found period')]

    params += [ChoiceParameter(qualifier='sample_mode', value=kwargs.get('sample_mode', 'auto'), choices=['auto', 'manual'], description='Whether to automatically determine sampling periods or set manually')]
    params += [FloatArrayParameter(visible_if='sample_mode:manual', qualifier='sample_periods', value=kwargs.get('sample_periods', []), default_unit=u.d, description='Manual period grid for sampling the periodogram')]

    params += [ChoiceParameter(qualifier='objective', value=kwargs.get('objective', 'likelihood'), choices=['likelihood', 'snr'], description='Objective to use for the periodogram.  See https://docs.astropy.org/en/stable/timeseries/bls.html#objectives')]

    return ParameterSet(params)
