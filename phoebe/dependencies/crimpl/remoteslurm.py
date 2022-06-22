
# from time import sleep as _sleep
import os as _os
import subprocess as _subprocess

from . import common as _common
from . import remotethread as _remotethread

class RemoteSlurmJob(_remotethread.RemoteThreadJob):
    def __init__(self, server=None,
                 job_name=None,
                 conda_env=None, isolate_env=False,
                 nprocs=4,
                 slurm_id=None, connect_to_existing=None):
        """
        Create and submit a job on a <RemoteSlurmServer>.

        Under-the-hood, this creates a subdirectory in <RemoteSlurmServer.directory>
        based on the provided or assigned `job_name`.  All submitted scripts/files
        (through either <RemoteSlurmJob.run_script> or <RemoteSlurmJob.submit_script>)
        are copied to and run in this directory.

        Arguments
        -------------
        * `server` (<RemoteSlurmServer>, optional, default=None): server to
            use when running the job.  If `server` is not provided, `host` must
            be provided.
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <RemoteSlurmJob.job_name>.  This `job_name` will
            be necessary to reconnect to a previously submitted job.
        * `conda_env` (string or None, optional, default=None): name of
            the conda environment to use for the job or False to not use a
            conda environment.  If not passed or None, will default to 'default'
            if conda is installed on the server or to False otherwise.
        * `isolate_env` (bool, optional, default=False): whether to clone
            the `conda_env` for use in this job.  If True, any setup/installation
            done by this job will not affect the original environment and
            will not affect other jobs.  Note that the environment is cloned
            (and therefore isolated) at the first call to <<class>.run_script>
            or <<class>.submit_script>.  Setup in the parent environment can
            be done at the server level, but requires passing `conda_env`.
            Will raise an error if `isolate_env=True` and `conda_env=False`.
        * `nprocs` (int, optional, default=4): default number of procs to use
            when calling <RemoteSlurmJob.submit_script>
        * `slurm_id` (int, optional, default=None): internal id of the remote
            slurm job.  If unknown, this will be determined automatically.
            Do **NOT** set `slurm_id` for a new <RemoteSlurmJob> instance.
        * `connect_to_existing` (bool, optional, default=None): NOT YET IMPLEMENTED
        """
        if slurm_id is not None and not isinstance(slurm_id, int):
            raise TypeError("slurm_id must be of type int")
        # TODO: check if slurm_id and job_name are in agreement? or is this handled by the super call below?
        self._slurm_id = slurm_id


        if connect_to_existing is None:
            if job_name is None:
                connect_to_existing = False
            else:
                connect_to_existing = True

        job_matches = [j for j in server.existing_jobs if j == job_name or job_name is None]

        if connect_to_existing:
            if len(job_matches) == 1:
                job_name = job_matches[0]
            elif len(job_matches) > 1:
                raise ValueError("{} jobs found on {} server.  Provide job_name or create a new job".format(len(job_matches), server.server_name))
            else:
                raise ValueError("no job could be found with job_name={} on {} server".format(job_name, server.server_name))
        else:
            if job_name is None:
                job_name = _common._new_job_name()
            elif len(job_matches):
                raise ValueError("job_name={} already exists on {} server".format(job_name, server.server_name))

        self._nprocs = nprocs

        super().__init__(server, job_name,
                         conda_env=conda_env,
                         isolate_env=isolate_env,
                         connect_to_existing=connect_to_existing)

    @property
    def nprocs(self):
        """
        Default number of processors to use when calling <<class>.submit_script>.

        Returns
        ---------
        * (int)
        """
        return self._nprocs

    @property
    def slurm_id(self):
        """
        Access the internal remote id of the Slurm scheduler on the remote server.

        Returns
        ----------
        * (int) slurm id
        """
        if self._slurm_id is None:
            # attempt to get slurm id from server
            try:
                out = self.server._run_server_cmd("cat {}".format(_os.path.join(self.remote_directory, "crimpl_slurm_id")))
                self._slurm_id = int(float(out))
            except:
                raise ValueError("No job has been submitted, call submit_script")

        return self._slurm_id

    @property
    def squeue(self):
        """
        Run and return the results from calling `squeue` on the remote server for
        this job's <RemoteSlurmJob.slurm_id>.

        Returns
        -----------
        * (string)
        """
        return self.server._run_server_cmd("squeue -j {}".format(self.slurm_id))

    @property
    def job_status(self):
        """
        Return the status of the job by calling and parsing the output of
        <RemoteSlurmJob.squeue>.

        If the job is no longer available in the queue, it is assumed to have
        completed (although in reality, it may have failed or been canceled).

        Returns
        -----------
        * (string): one of not-submitted, pending, running, canceled, failed, complete, unknown
        """
        if not self._job_submitted:
            return 'not-submitted'

        try:
            out = self.squeue
        except:
            # then we'll revert to checking the status file below
            out = ""

        if len(out.split("\n")) < 2:
            # then no longer in the queue, so we'll rely on the status file

            try:
                response = self.server._run_server_cmd("cat {}".format(_os.path.join(self.remote_directory, "crimpl-job.status")))
            except _subprocess.CalledProcessError:
                return 'unknown'

            if response == 'running':
                # then it started, but is no longer running according to slurm
                return 'failed'

            return response

        status = out.split("\n")[1].split()[4]
        # options for status from man squeue
        # BF  BOOT_FAIL       Job terminated due to launch failure, typically due to a hardware failure (e.g. unable to boot the node or block and the job can not be requeued).
        # CA  CANCELLED       Job was explicitly cancelled by the user or system administrator.  The job may or may not have been initiated.
        # CD  COMPLETED       Job has terminated all processes on all nodes with an exit code of zero.
        # CF  CONFIGURING     Job has been allocated resources, but are waiting for them to become ready for use (e.g. booting).
        # CG  COMPLETING      Job is in the process of completing. Some processes on some nodes may still be active.
        # DL  DEADLINE        Job terminated on deadline.
        # F   FAILED          Job terminated with non-zero exit code or other failure condition.
        # NF  NODE_FAIL       Job terminated due to failure of one or more allocated nodes.
        # OOM OUT_OF_MEMORY   Job experienced out of memory error.
        # PD  PENDING         Job is awaiting resource allocation.
        # PR  PREEMPTED       Job terminated due to preemption.
        # R   RUNNING         Job currently has an allocation.
        # RD  RESV_DEL_HOLD   Job is held.
        # RF  REQUEUE_FED     Job is being requeued by a federation.
        # RH  REQUEUE_HOLD    Held job is being requeued.
        # RQ  REQUEUED        Completing job is being requeued.
        # RS  RESIZING        Job is about to change size.
        # RV  REVOKED         Sibling was removed from cluster due to other cluster starting the job.
        # SI  SIGNALING       Job is being signaled.
        # SE  SPECIAL_EXIT    The job was requeued in a special state. This state can be set by users, typically in EpilogSlurmctld, if the job has terminated with a particular exit value.
        # SO  STAGE_OUT       Job is staging out files.
        # ST  STOPPED         Job has an allocation, but execution has been stopped with SIGSTOP signal.  CPUS have been retained by this job.
        # S   SUSPENDED       Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
        # TO  TIMEOUT         Job terminated upon reaching its time limit.


        if status in ['R', 'CG']:
            return 'running'
        elif status in ['CD']:
            return 'complete'
        elif status in ['CA']:
            return 'canceled'
        elif status in ['F', 'DL', 'NF', 'OOM', 'RF', 'RV', 'SE', 'ST', 'S', 'TO']:
            return 'failed'
        elif status in ['PD']:
            return 'pending'

        return status

    def kill_job(self):
        """
        Kill a job by calling `scancel` on the remote server for
        this job's <RemoteSlurmJob.slurm_id>.

        Returns
        -----------
        * (string)
        """
        return self.server._run_server_cmd("scancel {}".format(self.slurm_id))

    def run_script(self, script, files=[], trial_run=False):
        """
        Run a script on the server in the <<class>.conda_env>,
        and wait for it to complete.

        This is useful for short installation/setup scripts that do not belong
        in the scheduled job.

        The resulting `script` and `files` are copied to <RemoteSlurmJob.remote_directory>
        on the remote server and then `script` is executed via ssh.

        See <RemoteSlurmJob.submit_script> to submit a script via the slurm scheduler
        and leave running in the background on the server.

        Arguments
        ----------------
        * `script` (string or list): shell script to run on the remote server,
            including any necessary installation steps.  Note that the script
            can call any other scripts in `files`.  If a string, must be the
            path of a valid file which will be copied to the server.  If a list,
            must be a list of commands (i.e. a newline will be placed between
            each item in the list and sent as a single script to the server).
        * `files` (list, optional, default=[]): list of paths to additional files
            to copy to the server required in order to successfully execute
            `script`.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed.


        Returns
        ------------
        * None

        Raises
        ------------
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        return super().run_script(script, files=files, trial_run=trial_run)

    def submit_script(self, script, files=[],
                      slurm_job_name=None,
                      nprocs=None,
                      walltime='2-00:00:00',
                      mail_type='END,FAIL',
                      mail_user=None,
                      addl_slurm_kwargs={},
                      ignore_files=[],
                      wait_for_job_status=False,
                      trial_run=False):
        """
        Submit a script to the server in the <<class>.conda_env>.

        This will copy `script` (modified with the provided slurm options) and
        `files` to <RemoteSlurmJob.remote_directory> on the remote server and
        submit the script to the slurm scheduler.  To check on its status,
        see <RemoteSlurmJob.job_status>.

        Additional slurm customization (not included in the keyword arguments
        listed below) can be included in the beginning of the script.

        To check on any expected output files, call <RemoteSlurmJob.check_output>.

        See <RemoteSlurmJob.run_script> to run a script and wait for it to complete.

        Arguments
        ----------------
        * `script` (string or list): shell script to run on the remote server,
            including any necessary installation steps.  Note that the script
            can call any other scripts in `files`.  If a string, must be the
            path of a valid file which will be copied to the server.  If a list,
            must be a list of commands (i.e. a newline will be placed between
            each item in the list and sent as a single script to the server).
        * `files` (list, optional, default=[]): list of paths to additional files
            to copy to the server required in order to successfully execute
            `script`.
        * `slurm_job_name` (string, optional, default=None): name of the job within slurm.
            Prepended to `script` as "#SBATCH -J jobname".  Defaults to
            <RemoteSlurmJob.job_name>.
        * `nprocs` (int, optional, default=None): number of processors to run the
            job.  Prepended to `script` as "#SBATCH -n nprocs".  If None, will
            default to the `nprocs` set when creating the <RemoteSlurmJob> instance.
            See <RemoteSlurmJob.nprocs>.
        * `walltime` (string, optional, default='2-00:00:00'): maximum walltime
            to schedule the job.  Prepended to `script` as "#SBATCH -t walltime".
        * `mail_type` (string, optional, default='END,FAIL'): conditions to notify
            by email to `mail_user`.  Prepended to `script` as "#SBATCH --mail_user=mail_user".
        * `mail_user` (string, optional, default=None): email to send notifications.
            If not provided or None, will default to the value in <RemoteSlurmServer.mail_user>.
            Prepended to `script` as "#SBATCH --mail_user=mail_user"
        * `addl_slurm_kwargs` (dict, optional, default={}): additional kwargs
            to pass to slurm.  Entries will be prepended to `script` as
            "#SBATCH -<k> <v>" or "#SBATCH --<k>=<v>" depending on whether the
            key (`k`) is a single character or multiple characters, respectively.
        * `ignore_files` (list, optional, default=[]): list of filenames on the
            remote server to ignore when calling <<class>.check_output>
        * `wait_for_job_status` (bool or string or list, optional, default=False):
            Whether to wait for a specific job_status.  If True, will default to
            'complete'.  See also <RemoteSlurmJob.wait_for_job_status>.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed.

        Returns
        ------------
        * <RemoteSlurmJob>

        Raises
        ------------
        * ValueError: if a script has already been submitted within this
            <RemoteSlurmJob> instance.
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        if self._slurm_id is not None:
            raise ValueError("a job is already submitted.")

        if nprocs is None:
            nprocs = self.nprocs

        if "crimpl_submit_script.sh" in self.ls:
            raise ValueError("job already submitted.  Create a new job or call resubmit_job")

        cmds = self.server._submit_script_cmds(script, files, ignore_files,
                                               use_scheduler='slurm',
                                               directory=self.remote_directory,
                                               conda_env=self.conda_env,
                                               isolate_env=self.isolate_env,
                                               job_name=slurm_job_name if slurm_job_name is not None and len(slurm_job_name) else self.job_name,
                                               terminate_on_complete=False,
                                               use_nohup=False,
                                               install_conda=False,
                                               nprocs=nprocs,
                                               walltime=walltime,
                                               mail_type=mail_type,
                                               mail_user=mail_user if mail_user is not None else self.server.mail_user,
                                               **addl_slurm_kwargs)

        if trial_run:
            return cmds

        for cmd in cmds:
            if cmd is None: continue
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)

            try:
                out = self.server._run_server_cmd(cmd)
            except _subprocess.CalledProcessError as e:
                if addl_slurm_kwargs:
                    raise ValueError(f"failed to submit to scheduler, addl_slurm_kwargs may be invalid.  Original error: {e.output}")
                raise ValueError(f"failed to submit to scheduler.  Original error: {e.output}")

            if "sbatch" in cmd:
                self._slurm_id = out.split(' ')[-1]

                # leave record of slurm id in the remote directory
                self.server._run_server_cmd("echo {} > {}".format(self._slurm_id, _os.path.join(self.remote_directory, "crimpl_slurm_id")))


        self._job_submitted = True
        self._input_files = None

        if wait_for_job_status:
            self.wait_for_job_status(wait_for_job_status)

        return self

    def resubmit_script(self):
        """
        Resubmit an existing job script if <<class>.job_status> one of: complete,
        failed, killed.
        """
        status = self.job_status
        if status not in ['complete', 'failed', 'killed']:
            raise ValueError("cannot resubmit script with job_status='{}'".format(status))

        remote_script = _os.path.join(self.remote_directory, _os.path.basename("crimpl_submit_script.sh"))
        out = self.server._run_server_cmd("sbatch {remote_script}".format(remote_script=remote_script))
        self._slurm_id = out.split(' ')[-1]

        # leave record of (NEW) slurm id in the remote directory
        self.server._run_server_cmd("echo {} > {}".format(self._slurm_id, _os.path.join(self.remote_directory, "crimpl_slurm_id")))




class RemoteSlurmServer(_remotethread.RemoteThreadServer):
    _JobClass = RemoteSlurmJob
    def __init__(self, host, directory='~/crimpl', ssh='ssh', scp='scp',
                 mail_user=None, server_name=None):
        """
        Connect to a remote server running a Slurm scheduler.

        To create a new job, use <RemoteSlurmServer.create_job> or to connect
        to a previously created job, use <RemoteSlurmServer.get_job>.

        Arguments
        -----------
        * `host` (string): host of the remote server.  Must be passwordless ssh-able.
            See <RemoteSlurmServer.host>
        * `directory` (string, optional, default='~/crimpl'): root directory of all
            jobs to run on the remote server.  The directory will be created
            if it does not already exist.
        * `ssh` (string, optional, default='ssh'): command (and any arguments in
            addition to `host`) to ssh to the remote server.
        * `scp` (string, optional, default='scp'): command (and any arguments)
            to copy files to the remote server.
        * `mail_user` (string, optional, default=None): email to send notifications.
            If not provided or None, will default to the value in <RemoteSlurmServer.mail_user>.
            Prepended to `script` as "#SBATCH --mail_user=mail_user"
        * `server_name` (string): name to assign to the server.  If not provided,
            will be adopted automatically from `host` and available from
            <RemoteSlurmServer.server_name>.
        """
        super().__init__(host, directory, ssh, scp)
        self.mail_user = mail_user
        self._dict_keys += ['mail_user']

    @property
    def mail_user(self):
        """
        Default email to send notification from the slurm scheduler when calling
        <RemoteSlurmServer.submit_job> or <RemoteSlurmJob.submit_script>.
        """
        return self._mail_user

    @mail_user.setter
    def mail_user(self, mail_user):
        if mail_user is None:
            self._mail_user = None
            return

        if not isinstance(mail_user, str):
            raise TypeError("mail_user must be a string or None")

        if "@" not in mail_user:
            raise ValueError("mail_user must be a valid email address (with an @ symbol)")

        self._mail_user = mail_user

    @property
    def squeue(self):
        """
        Run and return the output of `squeue` on the server (for all jobs).

        To run for a single job, see <RemoteSlurmJob.squeue>.

        Returns
        -----------
        * (string)
        """
        return self._run_server_cmd("squeue")

    @property
    def sinfo(self):
        """
        Run and return the output of `sinfo` on the server (for all jobs).

        Returns
        -----------
        * (string)
        """
        return self._run_server_cmd("sinfo")

    def create_job(self, job_name=None,
                   conda_env=None, isolate_env=False,
                   nprocs=4):
        """
        Create a child <RemoteSlurmJob> instance.

        Arguments
        -----------
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <RemoteSlurmJob.job_name>.  This `job_name` will
            be necessary to reconnect to a previously submitted job.
        * `conda_env` (string or None, optional, default=None): name of
            the conda environment to use for the job or False to not use a
            conda environment.  If not passed or None, will default to 'default'
            if conda is installed on the server or to False otherwise.
        * `isolate_env` (bool, optional, default=False): whether to clone
            the `conda_env` for use in this job.  If True, any setup/installation
            done by this job will not affect the original environment and
            will not affect other jobs.  Note that the environment is cloned
            (and therefore isolated) at the first call to <<class>.run_script>
            or <<class>.submit_script>.  Setup in the parent environment can
            be done at the server level, but requires passing `conda_env`.
            Will raise an error if `isolate_env=True` and `conda_env=False`.
        * `nprocs` (int, optional, default=4): default number of procs to use
            when calling <RemoteSlurmJob.submit_job>

        Returns
        ---------
        * <RemoteSlurmJob>
        """
        return self._JobClass(server=self, job_name=job_name,
                              conda_env=conda_env,
                              isolate_env=isolate_env,
                              nprocs=nprocs, connect_to_existing=False)

    def submit_job(self, script, files=[],
                   job_name=None, slurm_job_name=None,
                   conda_env=None, isolate_env=False,
                   nprocs=4,
                   walltime='2-00:00:00',
                   mail_type='END,FAIL',
                   mail_user=None,
                   addl_slurm_kwargs={},
                   ignore_files=[],
                   wait_for_job_status=False,
                   trial_run=False):
        """
        Shortcut to <RemoteSlurmServer.create_job> followed by <RemoteSlurmJob.submit_script>.

        Arguments
        --------------
        * `script`: passed to <RemoteSlurmJob.submit_script>
        * `files`: passed to <RemoteSlurmJob.submit_script>
        * `job_name`: passed to <RemoteSlurmServer.create_job>
        * `slurm_job_name`: passed to <RemoteSlurmJob.submit_script>
        * `conda_env`: passed to <RemoteSlurmServer.create_job>
        * `isolate_env`: passed to <RemoteSlurmServer.create_job>
        * `nprocs`: passed to <RemoteSlurmServer.create_job>
        * `walltime`: passed to <RemoteSlurmJob.submit_script>
        * `mail_type`: passed to <RemoteSlurmJob.submit_script>
        * `mail_user`: passed to <RemoteSlurmJob.submit_script>
        * `addl_slurm_kwargs': pass to <RemoteSlurmJob.submit_script>`
        * `ignore_files`: passed to <RemoteSlurmJob.submit_script>
        * `wait_for_job_status`: passed to <RemoteSlurmJob.submit_script>
        * `trial_run`: passed to <RemoteSlurmJob.submit_script>

        Returns
        --------------
        * <RemoteSlurmJob>
        """
        j = self.create_job(job_name=job_name,
                            conda_env=conda_env,
                            isolate_env=isolate_env,
                            nprocs=nprocs)

        return j.submit_script(script, files=files,
                               slurm_job_name=slurm_job_name,
                               walltime=walltime,
                               mail_type=mail_type,
                               mail_user=mail_user,
                               addl_slurm_kwargs=addl_slurm_kwargs,
                               ignore_files=ignore_files,
                               wait_for_job_status=wait_for_job_status,
                               trial_run=trial_run)

    def run_script(self, script, files=[], conda_env=None, trial_run=False):
        """
        Run a script on the server in the `conda_env`, and wait for it to complete.

        The files are copied and executed in <RemoteSlurmServer.directory> directly
        (whereas <RemoteSlurmJob> scripts are executed in subdirectories).

        This is useful for short installation/setup scripts that do not belong
        in the scheduled job.

        The resulting `script` and `files` are copied to <RemoteSlurmServer.directory>
        on the remote server and then `script` is executed via ssh.

        Arguments
        ----------------
        * `script` (string or list): shell script to run on the remote server,
            including any necessary installation steps.  Note that the script
            can call any other scripts in `files`.  If a string, must be the
            path of a valid file which will be copied to the server.  If a list,
            must be a list of commands (i.e. a newline will be placed between
            each item in the list and sent as a single script to the server).
        * `files` (list, optional, default=[]): list of paths to additional files
            to copy to the server required in order to successfully execute
            `script`.
        * `conda_env` (string or None, optional, default=None): name of
            the conda environment to run the script or False to not use a
            conda environment.  If not passed or None, will default to 'default'
            if conda is installed on the server or to False otherwise.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed.


        Returns
        ------------
        * None

        Raises
        ------------
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        return super().run_script(script, files=files, conda_env=conda_env, trial_run=trial_run)
