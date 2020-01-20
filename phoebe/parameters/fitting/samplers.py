
import numpy as np

from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def emcee(**kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for fitting options for the
    emcee backend.  To use this backend, you must have emcee 3.0+ installed.

    To install emcee, see https://emcee.readthedocs.io

    If using this backend for fitting, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_fitting>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_fitting>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    For example:

    ```py
    b.add_fitting('emcee')
    b.run_fitting(kind='emcee')
    ```

    Arguments
    ----------

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """
    params = []

    params += [IntParameter(qualifier='nwalkers', value=kwargs.get('nwalkers', 16), limits=(1,1e5), description='number of walkers')]
    params += [IntParameter(qualifier='niters', value=kwargs.get('niters', 100), limits=(1,1e12), description='number of iterations')]

    params += [SelectParameter(qualifier='init_from', value=kwargs.get('init_from', []), choices=[], description='distribution to initialize samples from (all UNCONSTRAINED parameters with attached distributions will be sampled/fitted)')]
    params += [SelectParameter(qualifier='priors', value=kwargs.get('priors', []), choices=[], description='distribution to use for priors')]

    return ParameterSet(params)
