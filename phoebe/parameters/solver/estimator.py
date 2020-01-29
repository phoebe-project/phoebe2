
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

    return ParameterSet(params)
