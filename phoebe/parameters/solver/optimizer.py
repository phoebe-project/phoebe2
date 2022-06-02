
from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _comments_params(**kwargs):
    """
    """
    params = []

    params += [StringParameter(qualifier='comments', value=kwargs.get('comments', ''), description='User-provided comments for these solver-options.  Feel free to place any notes here - if not overridden, they will be copied to any resulting solutions.')]
    return params

def _server_params(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='use_server', value=kwargs.get('use_server', 'compute'), choices=['none', 'compute'], description='Server to use when running the solver (or "none" to run locally).  If "compute", will use the server settings in the referenced compute options.')]
    return params

def nelder_mead(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    scipy.optimize.minimize(method='nelder-mead') backend.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('optimizer.nelder_mead')
    b.run_solver(kind='nelder_mead')
    ```

    Parallelization support: nelder_mead does not support parallelization.  If
    within mpi, parallelization will be handled at the compute-level (per-time)
    for the <phoebe.parameters.compute.phoebe> backend.

    Arguments
    ----------
    * `compute` (string, optional): compute options to use for the forward
        model.
    * `expose_lnprobabilities` (bool, optional, default=False): whether to expose
        the initial and final lnprobabilities in the solution (will result in 2
        additional forward model calls)
    * `continue_from` (string, optional, default='none'): continue the optimization
        run from an existing solution by starting each parameter at its final
        position in the solution.
    * `fit_parameters` (list, optional, default=[]): parameters (as twigs) to
        optimize. Only applicable if `continue_from` is 'None'.
    * `initial_values` (dict, optional, default={}): twig-value pairs to
        (optionally) override the current values in the bundle.  Any items not
        in `fit_parameters` will be silently ignored.  Only applicable if
        `continue_from` is 'None'.
    * `priors` (list, optional, default=[]): distribution(s) to use for priors
        (constrained and unconstrained parameters will be included, covariances
        will be respected except for distributions merge via `priors_combine`).
    * `priors_combine` (str, optional, default='and'): Method to use to combine
        multiple distributions from priors for the same parameter.
        first: ignore duplicate entries and take the first in the priors parameter.
        and: combine duplicate entries via AND logic, dropping covariances.
        or: combine duplicate entries via OR logic, dropping covariances.
    * `maxiter` (int, optional, default=1e6): passed directly to
        scipy.optimize.minimize.  Maximum allowed number of iterations.
    * `adaptive` (bool, optional, default=False): passed directly to
        scipy.optimize.minimize.  Adapt algorithm parameters to dimensionality
        of problem. Useful for high-dimensional minimization
    * `xatol` (float, optional, default=1e-4): passed directly to
        scipy.optimize.minimize.  Absolute error in xopt (input parameters)
        between iterations that is acceptable for convergence.
    * `fatol` (float, optional, default=1e-4): passed directly to
        scipy.optimize.minimize.  Absolute error in func(xopt)
        (lnlikelihood + lnp(priors)) between iterations that is acceptable for
        convergence.
    * `progress_every_niters` (int, optional, default=0): Save the progress of
        the solution every n iterations.  The solution can only be recovered
        from an early termination by loading the bundle from a saved file and
        then calling <phoebe.frontend.bundle.Bundle.import_solution>(filename).
        The filename of the saved file will default to solution.ps.progress within
        <phoebe.frontend.bundle.Bundle.run_solver>, or the output filename provided
        to <phoebe.frontend.bundle.Bundle.export_solver> suffixed with .progress.
        If using detach=True within run_solver, attach job will load the progress
        and allow re-attaching until the job is completed.  If 0 will not save
        and will only return after completion.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]
    params += [BoolParameter(qualifier='expose_lnprobabilities', value=kwargs.get('expose_lnprobabilities', False), description='whether to expose the initial and final lnprobabilities in the solution (will result in 2 additional forward model calls)')]

    params += [ChoiceParameter(qualifier='continue_from', value=kwargs.get('continue_from', 'None'), choices=['None'], description='continue the optimization run from an existing solution by starting each parameter at its final position in the solution.')]

    params += [SelectTwigParameter(visible_if='continue_from:None', qualifier='fit_parameters', value=kwargs.get('fit_parameters', []), choices=[], description='parameters (as twigs) to optimize')]
    params += [DictParameter(visible_if='continue_from:None', qualifier='initial_values', value=kwargs.get('initial_values', {}), description='twig-value pairs to (optionally) override the current values in the bundle.  Any items not in fit_parameters will be silently ignored.')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (constrained and unconstrained parameters will be included, covariances will be respected except for distributions merge via priors_combine)')]
    params += [ChoiceParameter(visible_if='priors:<plural>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.  first: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 1e6), limits=[1,1e12], description='passed directly to scipy.optimize.minimize.  Maximum allowed number of iterations.')]
    params += [BoolParameter(qualifier='adaptive', value=kwargs.get('adaptive', False), description='passed directly to scipy.optimize.minimize.  Adapt algorithm parameters to dimensionality of problem. Useful for high-dimensional minimization')]

    params += [FloatParameter(qualifier='xatol', value=kwargs.get('xatol', 1e-4), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Absolute error in xopt (input parameters) between iterations that is acceptable for convergence.')]
    params += [FloatParameter(qualifier='fatol', value=kwargs.get('fatol', 1e-4), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Absolute error in func(xopt) (lnlikelihood + lnp(priors)) between iterations that is acceptable for convergence.')]

    params += [IntParameter(qualifier='progress_every_niters', value=kwargs.get('progress_every_niters', 0), limits=(0,1e6), description='save the progress of the solution every n iterations.  The solution can only be recovered from an early termination by loading the bundle from a saved file and then calling b.import_solution(filename).  The filename of the saved file will default to solution.ps.progress within run_solver, or the output filename provided to export_solver suffixed with .progress.  If using detach=True within run_solver, attach job will load the progress and allow re-attaching until the job is completed.  If 0 will not save and will only return after completion.')]

    return ParameterSet(params)

def powell(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    scipy.optimize.minimize(method='powell') backend.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('optimizer.powell')
    b.run_solver(kind='powell')
    ```

    Parallelization support: powell does not support parallelization.  If
    within mpi, parallelization will be handled at the compute-level (per-time)
    for the <phoebe.parameters.compute.phoebe> backend.

    Arguments
    ----------
    * `compute` (string, optional): compute options to use for the forward
        model.
    * `expose_lnprobabilities` (bool, optional, default=False): whether to expose
        the initial and final lnprobabilities in the solution (will result in 2
        additional forward model calls)
    * `continue_from` (string, optional, default='none'): continue the optimization
        run from an existing solution  by starting each parameter at its final
        position in the solution.
    * `fit_parameters` (list, optional, default=[]): parameters (as twigs) to
        optimize.  Only applicable if `continue_from` is 'None'.
    * `initial_values` (dict, optional, default={}): twig-value pairs to
        (optionally) override the current values in the bundle.  Any items not
        in `fit_parameters` will be silently ignored.  Only applicable if
        `continue_from` is 'None'.
    * `priors` (list, optional, default=[]): distribution(s) to use for priors
        (constrained and unconstrained parameters will be included, covariances
        will be respected except for distributions merge via `priors_combine`).
    * `priors_combine` (str, optional, default='and'): Method to use to combine
        multiple distributions from priors for the same parameter.
        first: ignore duplicate entries and take the first in the priors parameter.
        and: combine duplicate entries via AND logic, dropping covariances.
        or: combine duplicate entries via OR logic, dropping covariances.
    * `maxiter` (int, optional, default=1e6): passed directly to
        scipy.optimize.minimize.  Maximum allowed number of iterations.
    * `xtol` (float, optional, default=1e-4): passed directly to
        scipy.optimize.minimize.  Relative error in xopt (input parameters)
        between iterations that is acceptable for convergence.
    * `ftol` (float, optional, default=1e-4): passed directly to
        scipy.optimize.minimize.  Relative error in func(xopt)
        (lnlikelihood + lnp(priors)) between iterations that is acceptable for
        convergence.
    * `progress_every_niters` (int, optional, default=0): Save the progress of
        the solution every n iterations.  The solution can only be recovered
        from an early termination by loading the bundle from a saved file and
        then calling <phoebe.frontend.bundle.Bundle.import_solution>(filename).
        The filename of the saved file will default to solution.ps.progress within
        <phoebe.frontend.bundle.Bundle.run_solver>, or the output filename provided
        to <phoebe.frontend.bundle.Bundle.export_solver> suffixed with .progress.
        If using detach=True within run_solver, attach job will load the progress
        and allow re-attaching until the job is completed.  If 0 will not save
        and will only return after completion.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]
    params += [BoolParameter(qualifier='expose_lnprobabilities', value=kwargs.get('expose_lnprobabilities', False), description='whether to expose the initial and final lnprobabilities in the solution (will result in 2 additional forward model calls)')]

    params += [ChoiceParameter(qualifier='continue_from', value=kwargs.get('continue_from', 'None'), choices=['None'], description='continue the optimization run from an existing solution by starting each parameter at its final position in the solution.')]

    params += [SelectTwigParameter(visible_if='continue_from:None', qualifier='fit_parameters', value=kwargs.get('fit_parameters', []), choices=[], description='parameters to optimize')]
    params += [DictParameter(visible_if='continue_from:None', qualifier='initial_values', value=kwargs.get('initial_values', {}), description='twig-value pairs to (optionally) override the current values in the bundle.  Any items not in fit_parameters will be silently ignored.')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (constrained and unconstrained parameters will be included, covariances will be respected except for distributions merge via priors_combine)')]
    params += [ChoiceParameter(visible_if='priors:<plural>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.  irst: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 1e6), limits=[1,1e12], description='passed directly to scipy.optimize.minimize.  Maximum allowed number of iterations.')]

    params += [FloatParameter(qualifier='xtol', value=kwargs.get('xtol', 1e-4), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Relative error in xopt (input parameters) between iterations that is acceptable for convergence.')]
    params += [FloatParameter(qualifier='ftol', value=kwargs.get('ftol', 1e-4), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Relative error in func(xopt) (lnlikelihood + lnp(priors)) between iterations that is acceptable for convergence.')]

    params += [IntParameter(qualifier='progress_every_niters', value=kwargs.get('progress_every_niters', 0), limits=(0,1e6), description='save the progress of the solution every n iterations.  The solution can only be recovered from an early termination by loading the bundle from a saved file and then calling b.import_solution(filename).  The filename of the saved file will default to solution.ps.progress within run_solver, or the output filename provided to export_solver suffixed with .progress.  If using detach=True within run_solver, attach job will load the progress and allow re-attaching until the job is completed.  If 0 will not save and will only return after completion.')]


    return ParameterSet(params)

def cg(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    scipy.optimize.minimize(method='cg') "conjugate gradient" backend.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('optimizer.cg')
    b.run_solver(kind='cg')
    ```

    Parallelization support: cg does not support parallelization.  If
    within mpi, parallelization will be handled at the compute-level (per-time)
    for the <phoebe.parameters.compute.phoebe> backend.

    Arguments
    ----------
    * `compute` (string, optional): compute options to use for the forward
        model.
    * `expose_lnprobabilities` (bool, optional, default=False): whether to expose
        the initial and final lnprobabilities in the solution (will result in 2
        additional forward model calls)
    * `continue_from` (string, optional, default='none'): continue the optimization
        run from an existing solution by starting each parameter at its final
        position in the solution.
    * `fit_parameters` (list, optional, default=[]): parameters (as twigs) to
        optimize.  Only applicable if `continue_from` is 'None'.
    * `initial_values` (dict, optional, default={}): twig-value pairs to
        (optionally) override the current values in the bundle.  Any items not
        in `fit_parameters` will be silently ignored.  Only applicable if
        `continue_from` is 'None'.
    * `priors` (list, optional, default=[]): distribution(s) to use for priors
        (constrained and unconstrained parameters will be included, covariances
        will be respected except for distributions merge via `priors_combine`).
    * `priors_combine` (str, optional, default='and'): Method to use to combine
        multiple distributions from priors for the same parameter.
        first: ignore duplicate entries and take the first in the priors parameter.
        and: combine duplicate entries via AND logic, dropping covariances.
        or: combine duplicate entries via OR logic, dropping covariances.
    * `maxiter` (int, optional, default=1e6): passed directly to
        scipy.optimize.minimize.  Maximum allowed number of iterations.
    * `gtol` (float, optional, default=1e-5): passed directly to
        scipy.optimize.minimize.  Gradient norm must be less than gtol before successful termination.
    * `norm` (float, optional, default=np.inf): passed directly to
        scipy.optimize.minimize.  Order of norm (Inf is max, -Inf is min).
    * `progress_every_niters` (int, optional, default=0): Save the progress of
        the solution every n iterations.  The solution can only be recovered
        from an early termination by loading the bundle from a saved file and
        then calling <phoebe.frontend.bundle.Bundle.import_solution>(filename).
        The filename of the saved file will default to solution.ps.progress within
        <phoebe.frontend.bundle.Bundle.run_solver>, or the output filename provided
        to <phoebe.frontend.bundle.Bundle.export_solver> suffixed with .progress.
        If using detach=True within run_solver, attach job will load the progress
        and allow re-attaching until the job is completed.  If 0 will not save
        and will only return after completion.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]
    params += [BoolParameter(qualifier='expose_lnprobabilities', value=kwargs.get('expose_lnprobabilities', False), description='whether to expose the initial and final lnprobabilities in the solution (will result in 2 additional forward model calls)')]

    params += [ChoiceParameter(qualifier='continue_from', value=kwargs.get('continue_from', 'None'), choices=['None'], description='continue the optimization run from an existing solution by starting each parameter at its final position in the solution.')]

    params += [SelectTwigParameter(visible_if='continue_from:None', qualifier='fit_parameters', value=kwargs.get('fit_parameters', []), choices=[], description='parameters to optimize')]
    params += [DictParameter(visible_if='continue_from:None', qualifier='initial_values', value=kwargs.get('initial_values', {}), description='twig-value pairs to (optionally) override the current values in the bundle.  Any items not in fit_parameters will be silently ignored.')]

    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors (constrained and unconstrained parameters will be included, covariances will be respected except for distributions merge via priors_combine)')]
    params += [ChoiceParameter(visible_if='priors:<plural>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.  irst: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 1e6), limits=[1,1e12], description='passed directly to scipy.optimize.minimize.  Maximum allowed number of iterations.')]

    params += [FloatParameter(qualifier='gtol', value=kwargs.get('gtol', 1e-5), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Gradient norm must be less than gtol before successful termination.')]
    params += [FloatParameter(qualifier='norm', value=kwargs.get('norm', np.inf), limits=[None,None], description='passed directly to scipy.optimize.minimize.  Order of norm (Inf is max, -Inf is min).')]

    params += [IntParameter(qualifier='progress_every_niters', value=kwargs.get('progress_every_niters', 0), limits=(0,1e6), description='save the progress of the solution every n iterations.  The solution can only be recovered from an early termination by loading the bundle from a saved file and then calling b.import_solution(filename).  The filename of the saved file will default to solution.ps.progress within run_solver, or the output filename provided to export_solver suffixed with .progress.  If using detach=True within run_solver, attach job will load the progress and allow re-attaching until the job is completed.  If 0 will not save and will only return after completion.')]

    return ParameterSet(params)

def differential_evolution(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    scipy.optimize.differential_evolution() backend.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('optimizer.differential_evolution')
    b.run_solver(kind='differential_evolution')
    ```

    Parallelization support: differential_evolution supports both MPI and multiprocessing, always
    at the solver-level (per-model).


    Arguments
    ----------
    * `compute` (string, optional): compute options to use for the forward
        model.
    * `expose_lnprobabilities` (bool, optional, default=False): whether to expose
        the initial and final lnprobabilities in the solution (will result in 2
        additional forward model calls)
    * `continue_from` (string, optional, default='none'): continue the optimization
        run from an existing solution by starting each parameter at its final
        position in the solution.
    * `fit_parameters` (list, optional, default=[]): parameters (as twigs) to
        optimize.  Only applicable if `continue_from` is 'None'.
    * `bounds` (list, optional, default=[]): uniform priors as bounds on the parameters.
    * `maxiter` (int, optional, default=1e6): passed directly to
        scipy.optimize.differential_evolution.  Maximum allowed number of iterations.
    * `strategy` (str, optional, default='best1bin'): passed directly to
        scipy.optimize.differential_evolution.  The differential evolution strategy to use.
        Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
    * `popsize` (int, optional, default=8): passed directly to
        scipy.optimize.differential_evolution. A multiplier for setting the total
        population size.
    * `recombination` (float, optional, default=0.7): passed directly to
        scipy.optimize.differential_evolution. The recombination constant.
    * `tol` (float, optional, default=0.01): passed directly to
        scipy.optimize.differential_evolution. Relative tolerance for convergence.
    * `atol` (float, optional, default=0.0): passed directly to
        scipy.optimize.differential_evolution. Absolute tolerance for convergence.
    * `polish` (bool, optional, default=True): passed directly to
        scipy.optimize.differential_evolution. If True, then `scipy.optimize.minimize`
        with the `L-BFGS-B` method is used to polish the best population member at the end,
        which can improve the minimization slightly.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]
    params += [BoolParameter(qualifier='expose_lnprobabilities', value=kwargs.get('expose_lnprobabilities', False), description='whether to expose the initial and final lnprobabilities in the solution (will result in 2 additional forward model calls)')]

    params += [SelectTwigParameter(qualifier='fit_parameters', value=kwargs.get('fit_parameters', []), choices=[], description='parameters to optimize')]

    params += [SelectParameter(qualifier='bounds', value=kwargs.get('bounds', []), choices=[], description='distribution(s) to use for bounds.  For any non-uniform distributions, bounds will be adopted based on the bounds_sigma parameter.  Only those in fit_parameters will be considered.  Any in fit_parameters but not in bounds will use the limits on the parameter itself as bounds.  Any covariances will be ignored.')]
    params += [ChoiceParameter(visible_if='bounds:<plural>', qualifier='bounds_combine', value=kwargs.get('bounds_combine', 'first'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from bounds for the same parameter.  irst: ignore duplicate entries and take the first in the bounds parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances..')]
    params += [FloatParameter(visible_if='bounds:<notempty>', qualifier='bounds_sigma', value=kwargs.get('bounds_sigma', 3), limits=(0,10), default_units=u.dimensionless_unscaled, description='sigma-level to use when converting non-uniform distributions for bounds to uniform bounds')]

    # params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors')]
    # params += [ChoiceParameter(visible_if='priors:<plural>', qualifier='priors_combine', value=kwargs.get('priors_combine', 'and'), choices=['first', 'and', 'or'], description='Method to use to combine multiple distributions from priors for the same parameter.  irst: ignore duplicate entries and take the first in the priors parameter. and: combine duplicate entries via AND logic, dropping covariances.  or: combine duplicate entries via OR logic, dropping covariances.')]

    strategy_choices = ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp',
                        'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin',
                        'rand2bin', 'rand1bin']
    params += [ChoiceParameter(qualifier='strategy', value=kwargs.get('strategy', 'best1bin'), choices=strategy_choices, description='passed directly to scipy.optimize.differential_evolution.')]

    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 1e6), limits=[1,1e12], description='passed directly to scipy.optimize.differential_evolution.  The maximum number of generations over which the entire population is evolved. The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * len(x)')]
    params += [IntParameter(qualifier='popsize', value=kwargs.get('popsize', 8), limits=[1,1e4], description='passed directly to scipy.optimize.differential_evolution.  A multiplier for setting the total population size. The population has popsize * len(x) individuals (unless the initial population is supplied via the init keyword)')]
    params += [FloatParameter(qualifier='recombination', value=kwargs.get('recombination', 0.7), limits=[0,1], description='passed directly to scipy.optimize.differential_evolution. The recombination constant.')]
    params += [FloatParameter(qualifier='tol', value=kwargs.get('tol', 0.01), limits=[0, None], description='passed directly to scipy.optimize.differential_evolution. Relative tolerance for convergence.')]
    params += [FloatParameter(qualifier='atol', value=kwargs.get('atol', 0.0), limits=[0, None], description='passed directly to scipy.optimize.differential_evolution. Absolute tolerance for convergence.')]
    params += [BoolParameter(qualifier='polish', value=kwargs.get('polish', True), description='passed directly to scipy.optimize.differential_evolution. If True (default), then scipy.optimize.minimize with the L-BFGS-B method is used to polish the best population member at the end, which can improve the minimization slightly.')]
    # TODO: expose mutation, seed, init
    # params += [IntParameter(qualifier='progress_every_niters', value=kwargs.get('progress_every_niters', 0), limits=(0,1e6), description='save the progress of the solution every n iterations.  The solution can only be recovered from an early termination by loading the bundle from a saved file and then calling b.import_solution(filename).  The filename of the saved file will default to solution.ps.progress within run_solver, or the output filename provided to export_solver suffixed with .progress.  If using detach=True within run_solver, attach job will load the progress and allow re-attaching until the job is completed.  If 0 will not save and will only return after completion.')]

    return ParameterSet(params)

def differential_corrections(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for solver options for the
    differential-corrections backend.

    This only acts on LC and RV datasets.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_solver>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_solver>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_solver('optimizer.differential_corrections')
    b.run_solver(kind='differential_corrections')
    ```

    Parallelization support: differential_corrections does not support parallelization.  If
    within mpi, parallelization will be handled at the compute-level (per-time)
    for the <phoebe.parameters.compute.phoebe> backend.

    Arguments
    ----------
    * `compute` (string, optional): compute options to use for the forward
        model.
    * `expose_lnprobabilities` (bool, optional, default=False): whether to expose
        the initial and final lnprobabilities in the solution (will result in 1
        additional forward model call).  Note that since `differential_corrections`
        does not support priors, this is effectively the lnlikelihood.
    * `continue_from` (string, optional, default='none'): continue the optimization
        run from an existing solution by starting each parameter at its final
        position in the solution.
    * `fit_parameters` (list, optional, default=[]): parameters (as twigs) to
        optimize. Only applicable if `continue_from` is 'None'.
    * `initial_values` (dict, optional, default={}): twig-value pairs to
        (optionally) override the current values in the bundle.  Any items not
        in `fit_parameters` will be silently ignored.  Only applicable if
        `continue_from` is 'None'.
    * `steps` (dict, optional, default={}): twig-step pairs to set the
        stepsize for the differential corrections.  Any items not in
        `fit_parameters` will be silently ignored.
    * `deriv_method` (str, optional, default='symmetric'):  Whether to use
        'symmetric' or 'asymmetric' method for determining derivatives.


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = _comments_params(**kwargs)
    params += _server_params(**kwargs)

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]
    params += [BoolParameter(qualifier='expose_lnprobabilities', value=kwargs.get('expose_lnprobabilities', False), description='whether to expose the initial and final lnprobabilities in the solution (will result in 1 additional forward model call)')]

    params += [ChoiceParameter(qualifier='continue_from', value=kwargs.get('continue_from', 'None'), choices=['None'], description='continue the optimization run from an existing solution by starting each parameter at its final position in the solution.')]

    params += [SelectTwigParameter(visible_if='continue_from:None', qualifier='fit_parameters', value=kwargs.get('fit_parameters', []), choices=[], description='parameters (as twigs) to optimize')]
    params += [DictParameter(visible_if='continue_from:None', qualifier='initial_values', value=kwargs.get('initial_values', {}), description='twig-value pairs to (optionally) override the current values in the bundle.  Any items not in fit_parameters will be silently ignored.')]
    params += [DictParameter(qualifier='steps', value=kwargs.get('steps', {}), description='twig-value pairs to set the stepsize for differential corrections.  Any items not in fit_parameters will be silently ignored.')]

    params += [ChoiceParameter(qualifier='deriv_method', value=kwargs.get('deriv_method', 'symmetric'), choices=['symmetric', 'asymmetric'], description='Method to use for determining parameter derivatives.')]

    return ParameterSet(params)
