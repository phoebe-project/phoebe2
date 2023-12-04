
from phoebe.parameters import *


### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _comments_params(**kwargs):
    """
    """
    params = []

    params += [StringParameter(qualifier='comments', value=kwargs.get('comments', ''), description='User-provided comments for these solver-options.  Feel free to place any notes here - if not overridden, they will be copied to any resulting solutions.')]
    return params

def _server_params(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='use_server', value=kwargs.get('use_server', 'none'), choices=['none'], description='Server to use when running the solver (or "none" to run locally).')]
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

    The input light curve datasets (`lc_datasets`) are each normalized
    according to `lc_combine` and then combined.
    These combined data are then sent to the respective periodgram
    `algorithm` and the resulting period
    corresponding to the strongest peak is proposed as an adopted value.  In
    addition, the periodgram itself is exposed in the solution and available
    for plotting via <phoebe.parameters.ParameterSet.plot>.

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
    * `algorithm` (string, optional, default='bls'): algorithm to use to create
        the periodogram.  bls: BoxLeastSquares, ls: LombScargle.
    * `lc_datasets` (string or list, optional, default='*'): Light curve
        dataset(s) to use to run the periodogram algorithm.
    * `lc_combine` (string, optional, default='median'): How to normalize each
        light curve prior to combining.
    * `component` (string, optional, default=top-level orbit): Component to
        apply the found period.
    * `sample_mode` (string, optional, default='auto'): Whether to automatically
        determine sampling periods/frequencies ('auto') or set manually ('manual').
    * `sample_periods` (array, optional, default=[]): only applicable if
        `sample_mode` is 'manual'.  Manual period grid for sampling the periodogram.
        Note: if `algorithm` is 'ls', these will be converted to frequencies and
        will be more efficient if sampled evenly in frequency space (consider
        using <phoebe.invspace> instead of <phoebe.linspace>).
    * `duration` (array, optional, default=geomspace(0.01,0.3,5)): only applicable
        if `algorithm` is 'bls'.  The set of durations (in phase-space) that
        will be considered.  See
        https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html
    * `objective` (string, optional, default='likelihood'): only applicable if
        `algorithm` is 'bls'.  Objective to use for the periodogram.  See
        https://docs.astropy.org/en/stable/timeseries/bls.html#objectives
    * `minimum_n_cycles` (int, optional, default=3): only applicable if
        `algorithm` is 'bls' and `sample_mode` is 'auto'.  Minimum number of
        cycles/eclipses.  This is passed directly to autopower as
        'minimum_n_transit'. See
        https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html#astropy.timeseries.BoxLeastSquares.autopower
    * `samples_per_peak` (int, optional, default=10): only applicable if
        `algorithm` is 'ls' and `sample_mode` is 'auto'.  The approximate number
        of desired samples across the typical peak.  This is passed directly to
        autopower. See
        https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower
    * `nyquist_factor` (int, optional, default=5): only applicable if
        `algorithm` is 'ls' and `sample_mode` is 'auto'.  The multiple of the
        average nyquist frequency used to choose the maximum frequency.  This is
        passed directly to autopower. See
        https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='algorithm', value=kwargs.get('algorithm', 'bls'), choices=['bls', 'ls'], description='Algorithm to use to create the periodogram.  bls: BoxLeastSquares, ls: LombScargle.')]

    params += [SelectParameter(qualifier='lc_datasets', value=kwargs.get('lc_datasets', '*'), choices=[], description='Light curve dataset(s) to use to run the periodogram algorithm')]
    params += [ChoiceParameter(visible_if='lc_datasets:<plural>', qualifier='lc_combine', value=kwargs.get('lc_combine', 'median'), choices=['median', 'max'], advanced=True, description='How to normalize each light curve prior to combining.')]

    params += [ChoiceParameter(qualifier='component', value=kwargs.get('component', ''), choices=[''], description='Component to apply the found period')]

    params += [ChoiceParameter(qualifier='sample_mode', value=kwargs.get('sample_mode', 'auto'), choices=['auto', 'manual'], description='Whether to automatically determine sampling periods/frequencies or set manually')]

    params += [FloatArrayParameter(visible_if='sample_mode:manual', qualifier='sample_periods', value=kwargs.get('sample_periods', []), default_unit=u.d, description='Manual period grid for sampling the periodogram.  Note: if algorithm is \'ls\', these will be converted to frequencies and will be more efficient if sampled evenly in frequency space (consider using phoebe.invspace instead of phoebe.linspace).')]

    # BLS-specific options
    params += [FloatArrayParameter(visible_if='algorithm:bls', qualifier='duration', value=kwargs.get('duration', [0.1]), default_unit=u.d, description='The set of durations (in time-space) that will be considered.  See https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html')]
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

    The input radial velocity datasets (`rv_datasets`) are combined and then
    normalized by the absolute maximum value for the primary and secondary star
    independently, with the secondary then mirrored.  These combined data are
    then sent to the respective periodgram `algorithm` and the resulting period
    corresponding to the strongest peak is proposed as an adopted value.  In
    addition, the periodgram itself is exposed in the solution and available
    for plotting via <phoebe.parameters.ParameterSet.plot>.

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
    * `algorithm` (string, optional, default='bls'): algorithm to use to create
        the periodogram.  ls: LombScargle.
    * `rv_datasets` (string or list, optional, default='*'): Radial velocity
        dataset(s) to use to run the periodogram algorithm.
    * `component` (string, optional, default=top-level orbit): Component to
        apply the found period.
    * `sample_mode` (string, optional, default='auto'): Whether to automatically
        determine sampling periods/frequencies ('auto') or set manually ('manual').
    * `sample_periods` (array, optional, default=[]): only applicable if
        `sample_mode` is 'manual'.  Manual period grid for sampling the periodogram.
        Note: if `algorithm` is 'ls', these will be converted to frequencies and
        will be more efficient if sampled evenly in frequency space (consider
        using <phoebe.invspace> instead of <phoebe.linspace>).
    * `samples_per_peak` (int, optional, default=10): only applicable if
        `algorithm` is 'ls' and `sample_mode` is 'auto'.  The approximate number
        of desired samples across the typical peak.  This is passed directly to
        autopower. See
        https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower
    * `nyquist_factor` (int, optional, default=5): only applicable if
        `algorithm` is 'ls' and `sample_mode` is 'auto'.  The multiple of the
        average nyquist frequency used to choose the maximum frequency.  This is
        passed directly to autopower. See
        https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower



    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='algorithm', value=kwargs.get('algorithm', 'ls'), choices=['ls'], description='Algorithm to use to create the periodogram.  ls: LombScargle.')]

    params += [SelectParameter(qualifier='rv_datasets', value=kwargs.get('rv_datasets', '*'), choices=[], description='Radial velocity dataset(s) to use to run the periodgram algorithm')]

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

    The input light curve datasets (`lc_datasets`) are each normalized
    according to `lc_combine` and then combined.
    These combined data are then fitted with a 2-gaussian model
    which is used to help determine phases of eclipse minima, ingress, and
    egress.  These are then used to estimate and propose values for `ecc`, `per0`,
    `t0_supconj` for the corresponding `orbit` as well as `mask_phases` (not included in `adopt_parameters`
    by default).  If `expose_model` is True, the 2-gaussian model and the phases of minima,
    ingress, and egress are exposed in the solution and available for
    plotting with <phoebe.parameters.ParameterSet.plot>.

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
    * `lc_datasets` (string or list, optional, default='*'): Light curve
        dataset(s) to use to extract eclipse geometry
    * `lc_combine` (string, optional, default='median'): How to normalize each
        light curve prior to combining.
    * `phase_bin` (bool, optional, default=True): Bin the input observations (
        see `phase_nbins`) if more than 2*phase_nbins.  NOTE: input observational
        sigmas will be ignored during binning and replaced by per-bin standard
        deviations if possible, or ignored entirely otherwise.
    * `phase_nbins` (int, optional, default=500): Number of bins to use during
        phase binning input observations
        (will only be applied if len(times) > 2*`phase_nbins`).  Only applicable
        if `phase_bin` is True.
    * `orbit` (string, optional, default=top-level orbit): Orbit to use for
        phasing the light curve referenced in the `lc_datasets` parameter
    * `analytical_model` (string, optional, default='two-gaussian): Analytical
        model to fit the light curve with ('two-gaussian' or 'polyfit').
    * `t0_near_times` (bool, optional, default=True): Whether the returned value
        for t0_supconj should be forced to be in the range of the referenced
        observations.
    * `expose_model` (bool, optional, default=True): Whether to expose the
        2-gaussian analytical models in the solution

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [SelectParameter(qualifier='lc_datasets', value=kwargs.get('lc_datasets', '*'), choices=[], description='Light curve dataset(s) to use to extract eclipse geometry')]
    params += [ChoiceParameter(visible_if='lc_datasets:<plural>', qualifier='lc_combine', value=kwargs.get('lc_combine', 'median'), choices=['median', 'max'], advanced=True, description='How to normalize each light curve prior to combining.')]

    params += [BoolParameter(qualifier='phase_bin', value=kwargs.get('phase_bin', True), description='Bin the input observations (see phase_nbins) if more than 2*phase_nbins.  NOTE: input observational sigmas will be ignored during binning and replaced by per-bin standard deviations if possible, or ignored entirely otherwise.')]
    params += [IntParameter(qualifier='phase_nbins', visible_if='phase_bin:True', value=kwargs.get('phase_nbins', 500), limits=(100,None), description='Number of bins to use during phase binning input observations (will only be applied if len(times) > 2*phase_nbins)')]

    params += [ChoiceParameter(qualifier='orbit', value=kwargs.get('orbit', ''), choices=[''], description='Orbit to use for phasing the light curve referenced in the lc_datasets parameter')]
    params += [ChoiceParameter(qualifier='analytical_model', value=kwargs.get('analytical_model', 'two-gaussian'), choices=['two-gaussian', 'polyfit'], description='Analytical model to fit the light curve with.')]
    params += [BoolParameter(qualifier='interactive', value=kwargs.get('interactive', False), description='Whether to open results in interactive mode for manual adjustment.')]

    params += [BoolParameter(qualifier='t0_near_times', value=kwargs.get('t0_near_times', True), description='Whether the returned value for t0_supconj should be forced to be in the range of the referenced observations.')]
    params += [BoolParameter(qualifier='expose_model', value=kwargs.get('expose_model', True), description='Whether to expose the 2-gaussian analytical models in the solution')]

    return ParameterSet(params)

def rv_geometry(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    radial velocity geometry esimator.

    The input radial velocity datasets (`rv_datasets`) are combined without
    normalization.  These combined data are then used to estimate the
    semi-amplitude and `t0_supconj` which are then used to fit a Keplerian
    orbit using least-squares.  If RVs are available for both components,
    this results in proposed values for `t0_supconj`,
    `q`, `asini`, `ecc`, and `per0` for the corresponding `orbit` and `vgamma`
    of the system .  If RVs are only available for one of the components, then
    `q` is excluded, and the proposed `asini` is for the stellar component instead
    of the binary orbit.
    If `expose_model` is True, the analytical Keplerian RVs are exposed in the
    solution and available for
    plotting with <phoebe.parameters.ParameterSet.plot>.


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
    * `rv_datasets` (string or list, optional, default='*'): Radial velocity
        dataset(s) to use to extract RV geometry.
    * `phase_bin` (bool, optional, default=True): Bin the input observations (
        see `phase_nbins`) if more than 2*phase_nbins.  NOTE: input observational
        sigmas will be ignored during binning and replaced by per-bin standard
        deviations if possible, or ignored entirely otherwise.
    * `phase_nbins` (int, optional, default=500): Number of bins to use during
        phase binning input observations
        (will only be applied if len(times) > 2*`phase_nbins`).  Only applicable
        if `phase_bin` is True.
    * `orbit` (string, optional, default=top-level orbit): Orbit to use for
        estimating orbital parameters.
    * `expose_model` (bool, optional, default=True): Whether to expose the
        Keplerian analytical models in the solution.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [SelectParameter(qualifier='rv_datasets', value=kwargs.get('rv_datasets', '*'), choices=[], description='Radial velocity dataset(s) to use to extract RV geometry')]
    params += [ChoiceParameter(qualifier='orbit', value=kwargs.get('orbit', ''), choices=[''], description='Orbit to use for estimating orbital parameters')]

    params += [BoolParameter(qualifier='phase_bin', value=kwargs.get('phase_bin', True), description='Bin the input observations (see phase_nbins) if more than 2*phase_nbins.  NOTE: input observational sigmas will be ignored during binning and replaced by per-bin standard deviations if possible, or ignored entirely otherwise.')]
    params += [IntParameter(qualifier='phase_nbins', visible_if='phase_bin:True', value=kwargs.get('phase_nbins', 500), limits=(100,None), description='Number of bins to use during phase binning input observations (will only be applied if len(times) > 2*phase_nbins)')]

    # params += [BoolParameter(qualifier='t0_near_times', value=kwargs.get('t0_near_times', True), description='Whether the returned value for t0_supconj should be forced to be in the range of the referenced observations.')]

    params += [BoolParameter(qualifier='expose_model', value=kwargs.get('expose_model', True), description='Whether to expose the Keplerian analytical models in the solution')]


    return ParameterSet(params)


def ebai(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    ebai artificial neural network solver.

    This solver requires scikit-learn to be installed if using the `knn` method.
    To install scikit-learn, see https://scikit-learn.org/stable/install.html.

    When using this solver, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2008ApJ...687..542P (if ebai_method = `mlp`)
    * http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html (if ebai_method = `knn`)

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    The input light curve datasets (`lc_datasets`) are each normalized
    according to `lc_combine`, combined and
    fitted with an analytical model (two-Gaussian for contact binaries and
    detached if ebai_method=`mlp`, polyfit for detached with ebai_method=`knn`),
    which is then itself normalized and used as input to `ebai`.
    Any necessary phase-shift required to ensure the primary is at a phase of 0 is used
    to provide the proposed value for `t0_supconj`.  The normalized model is then sent
    through the pre-trained `ebai` model, resulting in proposed values for
    `teffratio`, `requivsumfrac`, `esinw`, `ecosw`, and `incl` for detached systems,
    and `teffratio`, `q`, `fillout_factor` and `incl` for contact systems.

    Note that the `mlp` network only supports detached systems and will return
    all nans and a logger warning if either eclipse from the 2 gaussian model has
    a width greater than 0.25 (in phase-space). Use ebai_method = `knn` instead.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('estimator.ebai')
    b.run_solver(kind='ebai')
    ```

    Arguments
    ----------
    * `lc_datasets` (string or list, optional, default='*'): Light curve
        dataset(s) to pass to ebai.
    * `lc_combine` (string, optional, default='median'): How to normalize each
        light curve prior to combining.
    * `phase_bin` (bool, optional, default=True): Bin the input observations (
        see `phase_nbins`) if more than 2*phase_nbins.  NOTE: input observational
        sigmas will be ignored during binning and replaced by per-bin standard
        deviations if possible, or ignored entirely otherwise.
    * `phase_nbins` (int, optional, default=500): Number of bins to use during
        phase binning input observations
        (will only be applied if len(times) > 2*`phase_nbins`).  Only applicable
        if `phase_bin` is True.
    * `ebai_method` (str, optional, default='knn'): EBAI method to use. If 'knn',
        a train scikit-learn KNeighborsRegressor will be used. If 'mlp', a custom
        neural network will be used instead (only applicable to detached systems.)
    * `orbit` (string, optional, default=top-level orbit): Orbit to use for
        phasing the light curve referenced in the `lc_datasets` parameter

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [SelectParameter(qualifier='lc_datasets', value=kwargs.get('lc_datasets', '*'), choices=[], description='Light curve dataset(s) to pass to ebai')]
    params += [ChoiceParameter(visible_if='lc_datasets:<plural>', qualifier='lc_combine', value=kwargs.get('lc_combine', 'median'), choices=['median', 'max'], advanced=True, description='How to normalize each light curve prior to combining.')]

    params += [BoolParameter(qualifier='phase_bin', value=kwargs.get('phase_bin', True), description='Bin the input observations (see phase_nbins) if more than 2*phase_nbins.  NOTE: input observational sigmas will be ignored during binning and replaced by per-bin standard deviations if possible, or ignored entirely otherwise.')]
    params += [IntParameter(qualifier='phase_nbins', visible_if='phase_bin:True', value=kwargs.get('phase_nbins', 500), limits=(100,None), description='Number of bins to use during phase binning input observations (will only be applied if len(times) > 2*phase_nbins)')]
    params += [ChoiceParameter(qualifier='ebai_method', value=kwargs.get('ebai_method', 'knn'), choices=['knn', 'mlp'], description='Choice of machine learning model to use for prediction. knn uses a trained sklearn kNeighborsRegressor, while mlp uses a trained neural network.')]
    params += [ChoiceParameter(qualifier='orbit', value=kwargs.get('orbit', ''), choices=[''], description='Orbit to use for phasing the light curve referenced in the lc_datasets parameter')]

    return ParameterSet(params)
