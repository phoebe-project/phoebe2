
from phoebe.parameters import *
from phoebe import conf
from phoebe import geomspace as _geomspace

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

def periodogram(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    periodogram period estimator using astropy.  See
    https://docs.astropy.org/en/stable/timeseries/bls.html or
    https://docs.astropy.org/en/stable/timeseries/lombscargle.html for more details.

    NOTE: this requires astropy 3.2+, which in turn requires python 3.  If these
    requirements are not met, an error will be raised when attempting to call
    <phoebe.frontend.bundle.Bundle.run_solver>.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('estimator.periodogram')
    b.run_solver(kind='periodogram')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [ChoiceParameter(qualifier='algorithm', value=kwargs.get('algorithm', 'bls'), choices=['bls', 'ls'], description='Algorithm to use to create the periodogram.  bls: BoxLeastSquares, ls: LombScargle.')]

    params += [ChoiceParameter(qualifier='lc', value=kwargs.get('lc', ''), choices=[''], description='Light curve dataset to use to run the BLS algorithm')]
    params += [ChoiceParameter(qualifier='component', value=kwargs.get('component', ''), choices=[''], description='Component to apply the found period')]

    params += [ChoiceParameter(qualifier='sample_mode', value=kwargs.get('sample_mode', 'auto'), choices=['auto', 'manual'], description='Whether to automatically determine sampling periods/frequencies or set manually')]

    # params += [FloatArrayParameter(visible_if='sample_mode:manual,algorithm:bls', qualifier='sample_periods', value=kwargs.get('sample_periods', []), default_unit=u.d, description='Manual period grid for sampling the periodogram')]
    # params += [FloatArrayParameter(visible_if='sample_mode:manual,algorithm:ls', qualifier='sample_frequencies', value=kwargs.get('sample_frequencies', []), default_unit=u.d**-1, description='Manual frequency grid for sampling the periodogram')]
    params += [FloatArrayParameter(visible_if='sample_mode:manual', qualifier='sample_periods', value=kwargs.get('sample_periods', []), default_unit=u.d, description='Manual period grid for sampling the periodogram.  Note: if algorithm is \'ls\', these will be converted to frequencies and will be more efficient if sampled evenly in frequency space (consider using phoebe.invspace instead of phoebe.linspace).')]

    # BLS-specific options
    params += [FloatArrayParameter(visible_if='algorithm:bls', qualifier='duration', value=kwargs.get('duration', _geomspace(0.01,0.3,5)), default_unit=u.dimensionless_unscaled, description='The set of durations (in phase-space) that will be considered.  See https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html')]
    params += [ChoiceParameter(visible_if='algorithm:bls', qualifier='objective', value=kwargs.get('objective', 'likelihood'), choices=['likelihood', 'snr'], description='Objective to use for the periodogram.  See https://docs.astropy.org/en/stable/timeseries/bls.html#objectives')]
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:bls', qualifier='minimum_n_cycles', value=kwargs.get('minimum_n_cycles', 3), advanced=True, limits=(1,None), description='Minimum number of cycles/eclipses.  This is passed directly to autopower as \'minimum_n_transit\'. See https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.autopower')]


    # LS-specific options
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:ls', qualifier='samples_per_peak', value=kwargs.get('samples_per_peak', 5), advanced=True, limits=(1,None), description='The approximate number of desired samples across the typical peak.  This is passed directly to autopower. See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower')]
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:ls', qualifier='nyquist_factor', value=kwargs.get('nyquist_factor', 5), advanced=True, limits=(1,None), description='The multiple of the average nyquist frequency used to choose the maximum frequency.  This is passed directly to autopower. See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower')]

    return ParameterSet(params)