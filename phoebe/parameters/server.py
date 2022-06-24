

from phoebe.parameters import *
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def remoteslurm(server, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a remoteslurm server.

    The server referenced by `crimpl_name` must be configured on the local
    machine with [crimpl](http://crimpl.readthedocs.io).

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_server>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_server>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `crimpl_name` (string, optional): Name of server saved in crimpl.  Must be
        available on the local machine.  See docs for more details.
    * `use_conda` (bool, optional, default=True): Whether to use a conda environment
        on the server.  Jobs will fail if conda is not installed on the remote
        server if `use_conda=True`.
    * `conda_env` (string, optional, default='default'): Name of conda
        environment on remote machine to run jobs.  Will be created and
        necessary deps installed (if `install_deps=True`) if does not exist.
        Only applicable if `use_conda=True`.
    * `isolate_env` (bool, optional, default=False): Whether to clone the
        conda_env environment per-job.  Only applicable if `use_conda=True`.
    * `nprocs` (int, optional, default=4): Number of processors to allocate to
        each job
    * `use_mpi` (bool, optional, default=True): Whether to use mpi on the remote
       machine
    * `install_deps` (bool, optional, default=True): Whether to ensure required
        dependencies are installed in conda_env on the remote machine (adds some
        overhead)
    * `slurm_job_name` (string, optional): Job name to use within slurm on the
        remote machine
    * `walltime` (float, optional, default=0.5): Walltime to allocate to
        each job.  Default units are in hours.
    * `mail_user` (string, optional): Email to have slurm notify about events
        matching mail_type
    * `mail_type` (list, optional, default=['END', 'FAIL']): Scenarios in which
        to request slurm to notify mail_user by email
    * `addl_slurm_kwargs` (dict, optional): additional kwargs
        to pass to slurm.  Entries will be prepended to `script` as
        "#SBATCH -<k> <v>" or "#SBATCH --<k>=<v>" depending on whether the
        key (`k`) is a single character or multiple characters, respectively.
        NEW IN PHOEBE 2.4.3.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """

    params = []

    params += [StringParameter(qualifier="crimpl_name", value=kwargs.get('crimpl_name', ''), description='Name of server saved in crimpl.  Must be available on the local machine.  See docs for more details.')]
    params += [BoolParameter(qualifier='use_conda', value=kwargs.get('use_conda', True), description='Whether to use a conda environment on the server.')]
    params += [StringParameter(visible_if='use_conda:true', qualifier='conda_env', value=kwargs.get('conda_env', 'default'), description='Name of conda environment on remote machine to run jobs.  Will be created if does not exist.')]
    params += [BoolParameter(visible_if='use_conda:true', qualifier='isolate_env', value=kwargs.get('isolate_env', False), advanced=True, description='Whether to clone the conda_env environment per-job.')]
    params += [IntParameter(qualifier='nprocs', value=kwargs.get('nprocs', 4), description='Number of processors to allocate to each job')]
    params += [BoolParameter(qualifier='use_mpi', value=kwargs.get('use_mpi', True), description='Whether to use mpi on the remote machine')]
    params += [BoolParameter(qualifier='install_deps', value=kwargs.get('install_deps', True), description='Whether to ensure required dependencies are installed in conda_env on the remote machine (adds some overhead)')]

    params += [StringParameter(qualifier='slurm_job_name', value=kwargs.get('slurm_job_name', ''), description='Jobname to use within slurm on the remote machine (optional)')]
    params += [FloatParameter(qualifier='walltime', value=kwargs.get('walltime', 0.5), default_unit=u.hr, limits=(0,None), description='Walltime to allocate to each job')]
    params += [StringParameter(qualifier='mail_user', value=kwargs.get('mail_user', ''), description='Email to have slurm notify about events matching mail_type')]
    params += [SelectParameter(qualifier='mail_type', visible_if='mail_user:<notempty>', value=kwargs.get('mail_type', ['END', 'FAIL']), choices=['BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'], description='Scenarios in which to request slurm to notify mail_user by email')]
    params += [DictParameter(qualifier='addl_slurm_kwargs', value=kwargs.get('addl_slurm_kwargs', {}), description='List of addition slurm arguments.  Keys with single characters will be passed to slurm as "#SBATCH -<key> <value>".  Keys with multiple characters will be passed to slurm as "#SBATCH --<key>=<value>"')]

    return ParameterSet(params)

def localthread(server, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a localthread server.

    The server referenced by `crimpl_name` must be configured on the local
    machine with [crimpl](http://crimpl.readthedocs.io).

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_server>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_server>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Arguments
    ----------
    * `crimpl_name` (string, optional): Name of server saved in crimpl.  Must be
        available on the local machine.  See docs for more details.
    * `use_conda` (bool, optional, default=False): Whether to use a conda environment
        for jobs running in the server directory.
    * `conda_env` (string, optional, default='default'): Name of conda
        environmentto run jobs.  Will be created and necessary deps installed
        (if `install_deps=True`) if does not exist.  Only applicable if `use_conda=True`.
    * `isolate_env` (bool, optional, default=False): Whether to clone the
        conda_env environment per-job.  Only applicable if `use_conda=True`.
    * `use_mpi` (bool, optional, default=True): Whether to use mpi on the remote
       machine
    * `nprocs` (int, optional, default=4): Number of processors to use
        within MPI.  Applicable only if `use_mpi=True`.
    * `install_deps` (bool, optional, default=True): Whether to ensure required
        dependencies are installed in conda_env on the remote machine (adds some
        overhead)

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects.
    """

    params = []

    params += [StringParameter(qualifier='crimpl_name', value=kwargs.get('crimpl_name', ''), description='Name of server saved in crimpl.  If empty, a crimpl local thread server will be created in a \'crimpl\' subdirectory in the current working directory.')]
    params += [BoolParameter(qualifier='use_conda', value=kwargs.get('use_conda', False), description='Whether to use a conda environment on the server.')]
    params += [StringParameter(visible_if='use_conda:true', qualifier='conda_env', value=kwargs.get('conda_env', 'default'), description='Name of conda environment on remote machine to run jobs.  Will be created if does not exist.')]
    params += [BoolParameter(visible_if='use_conda:true', qualifier='isolate_env', value=kwargs.get('isolate_env', False), advanced=True, description='Whether to clone the conda_env environment per-job.')]
    params += [BoolParameter(qualifier='use_mpi', value=kwargs.get('use_mpi', True), description='Whether to use mpi on the remote machine')]
    params += [IntParameter(visible_if='use_mpi:true', qualifier='nprocs', value=kwargs.get('nprocs', 4), description='Number of processors to use within MPI')]
    params += [BoolParameter(qualifier='install_deps', value=kwargs.get('install_deps', True), description='Whether to ensure required dependencies are installed in conda_env on the remote machine (adds some overhead)')]

    return ParameterSet(params)
