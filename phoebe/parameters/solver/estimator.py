
from phoebe.parameters import *
from phoebe import conf
from phoebe import geomspace as _geomspace
from phoebe import linspace as _linspace

from distutils.version import StrictVersion
import numpy as np


### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _comments_params(**kwargs):
    """
    """
    params = []

    params += [StringParameter(qualifier='comments', value=kwargs.get('comments', ''), description='User-provided comments for these solver-options.  Feel free to place any notes here - if not overridden, they will be copied to any resulting solutions.')]
    return params


def lc_periodogram(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    light curve periodogram period estimator using astropy.  See
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
    b.add_solver('estimator.lc_periodogram')
    b.run_solver(kind='lc_periodogram')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)

    params += [ChoiceParameter(qualifier='algorithm', value=kwargs.get('algorithm', 'bls'), choices=['bls', 'ls'], description='Algorithm to use to create the periodogram.  bls: BoxLeastSquares, ls: LombScargle.')]

    params += [ChoiceParameter(qualifier='lc', value=kwargs.get('lc', ''), choices=[''], description='Light curve dataset to use to run the periodogram algorithm')]
    params += [ChoiceParameter(qualifier='component', value=kwargs.get('component', ''), choices=[''], description='Component to apply the found period')]

    params += [ChoiceParameter(qualifier='sample_mode', value=kwargs.get('sample_mode', 'auto'), choices=['auto', 'manual'], description='Whether to automatically determine sampling periods/frequencies or set manually')]

    params += [FloatArrayParameter(visible_if='sample_mode:manual', qualifier='sample_periods', value=kwargs.get('sample_periods', []), default_unit=u.d, description='Manual period grid for sampling the periodogram.  Note: if algorithm is \'ls\', these will be converted to frequencies and will be more efficient if sampled evenly in frequency space (consider using phoebe.invspace instead of phoebe.linspace).')]

    # BLS-specific options
    params += [FloatArrayParameter(visible_if='algorithm:bls', qualifier='duration', value=kwargs.get('duration', _geomspace(0.01,0.3,5) if StrictVersion(np.__version__) >= StrictVersion("1.13") else _linspace(0.01, 0.3, 0.5)), default_unit=u.dimensionless_unscaled, description='The set of durations (in phase-space) that will be considered.  See https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html')]
    params += [ChoiceParameter(visible_if='algorithm:bls', qualifier='objective', value=kwargs.get('objective', 'likelihood'), choices=['likelihood', 'snr'], description='Objective to use for the periodogram.  See https://docs.astropy.org/en/stable/timeseries/bls.html#objectives')]
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:bls', qualifier='minimum_n_cycles', value=kwargs.get('minimum_n_cycles', 3), advanced=True, limits=(1,None), description='Minimum number of cycles/eclipses.  This is passed directly to autopower as \'minimum_n_transit\'. See https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.autopower')]


    # LS-specific options
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:ls', qualifier='samples_per_peak', value=kwargs.get('samples_per_peak', 10), advanced=True, limits=(1,None), description='The approximate number of desired samples across the typical peak.  This is passed directly to autopower. See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower')]
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:ls', qualifier='nyquist_factor', value=kwargs.get('nyquist_factor', 5), advanced=True, limits=(1,None), description='The multiple of the average nyquist frequency used to choose the maximum frequency.  This is passed directly to autopower. See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower')]

    return ParameterSet(params)

def rv_periodogram(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    radial velocity periodogram period estimator using astropy.  See
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
    b.add_solver('estimator.rv_periodogram')
    b.run_solver(kind='rv_periodogram')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)

    params += [ChoiceParameter(qualifier='algorithm', value=kwargs.get('algorithm', 'ls'), choices=['ls'], description='Algorithm to use to create the periodogram.  ls: LombScargle.')]

    params += [ChoiceParameter(qualifier='rv', value=kwargs.get('rv', ''), choices=[''], description='Radial velocity dataset to use to run the periodgram algorithm')]
    params += [ChoiceParameter(qualifier='component', value=kwargs.get('component', ''), choices=[''], description='Component to apply the found period')]

    params += [ChoiceParameter(qualifier='sample_mode', value=kwargs.get('sample_mode', 'auto'), choices=['auto', 'manual'], description='Whether to automatically determine sampling periods/frequencies or set manually')]

    params += [FloatArrayParameter(visible_if='sample_mode:manual', qualifier='sample_periods', value=kwargs.get('sample_periods', []), default_unit=u.d, description='Manual period grid for sampling the periodogram.  Note: if algorithm is \'ls\', these will be converted to frequencies and will be more efficient if sampled evenly in frequency space (consider using phoebe.invspace instead of phoebe.linspace).')]

    # LS-specific options
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:ls', qualifier='samples_per_peak', value=kwargs.get('samples_per_peak', 10), advanced=True, limits=(1,None), description='The approximate number of desired samples across the typical peak.  This is passed directly to autopower. See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower')]
    params += [IntParameter(visible_if='sample_mode:auto,algorithm:ls', qualifier='nyquist_factor', value=kwargs.get('nyquist_factor', 5), advanced=True, limits=(1,None), description='The multiple of the average nyquist frequency used to choose the maximum frequency.  This is passed directly to autopower. See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower')]

    return ParameterSet(params)


def lc_geometry(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    light curve geometry esimator.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('estimator.lc_geometry')
    b.run_solver(kind='lc_geometry')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)

    params += [ChoiceParameter(qualifier='lc', value=kwargs.get('lc', ''), choices=[''], description='Light curve dataset to use to extract eclipse geometry')]
    params += [ChoiceParameter(qualifier='orbit', value=kwargs.get('orbit', ''), choices=[''], description='Orbit to use for phasing the light curve referenced in the dataset parameter')]

    params += [BoolParameter(qualifier='t0_near_times', value=kwargs.get('t0_near_times', True), description='Whether the returned value for t0_supconj should be forced to be in the range of the referenced observations.')]

    return ParameterSet(params)

def rv_geometry(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    radial velocity geometry esimator.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('estimator.rv_geometry')
    b.run_solver(kind='rv_geometry')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)

    params += [ChoiceParameter(qualifier='rv', value=kwargs.get('rv', ''), choices=[''], description='Radial velocity dataset to use to extract eclipse geometry')]
    params += [ChoiceParameter(qualifier='orbit', value=kwargs.get('orbit', ''), choices=[''], description='Orbit to use for estimating orbital parameters')]

    # params += [BoolParameter(qualifier='t0_near_times', value=kwargs.get('t0_near_times', True), description='Whether the returned value for t0_supconj should be forced to be in the range of the referenced observations.')]

    return ParameterSet(params)
