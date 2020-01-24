
from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def nelder_mead(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for fitting options for the
    scipy.optimize.minimize(method='nelder-mead') backend.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_fitting>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_fitting>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_fitting('optimize.nelder_mead')
    b.run_fitting(kind='nelder_mead')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [ChoiceParameter(qualifier='compute', value=kwargs.get('compute', 'None'), choices=['None'], description='compute options to use for forward model')]
    params += [SelectParameter(qualifier='init_from', value=kwargs.get('init_from', []), choices=[], description='distribution(s) to draw from for initial guess (all UNCONSTRAINED parameters with attached distributions will be sampled/fitted)')]
    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution(s) to use for priors')]

    params += [IntParameter(qualifier='maxiter', value=kwargs.get('maxiter', 1e6), limits=[1,1e12], description='passed directly to scipy.optimize.minimize.  Maximum allowed number of iterations.')]
    params += [IntParameter(qualifier='maxfev', value=kwargs.get('maxfev', 1e6), limits=[1,1e12], description='passed directly to scipy.optimize.minimize.  Maximum allowed number of function evaluations (forward models).')]
    params += [BoolParameter(qualifier='adaptive', value=kwargs.get('adaptive', False), description='passed directly to scipy.optimize.minimize.  Adapt algorithm parameters to dimensionality of problem. Useful for high-dimensional minimization')]

    params += [FloatParameter(qualifier='xatol', value=kwargs.get('xatol', 1e-4), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Absolute error in xopt (input parameters) between iterations that is acceptable for convergence.')]
    params += [FloatParameter(qualifier='fatol', value=kwargs.get('fatol', 1e-4), limits=[1e-12,None], description='passed directly to scipy.optimize.minimize.  Absolute error in func(xopt) (lnlikelihood + lnp(priors)) between iterations that is acceptable for convergence.')]

    return ParameterSet(params)
