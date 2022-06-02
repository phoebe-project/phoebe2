
# from time import sleep as _sleep
import os as _os
import subprocess as _subprocess

from . import common as _common

class RemoteThreadJob(_common.ServerJob):
    def __init__(self, server=None,
                 job_name=None,
                 conda_env=None, isolate_env=False,
                 nprocs=None,
                 connect_to_existing=None):
        """
        Create and submit a job on a <RemoteThreadServer>.

        Under-the-hood, this creates a subdirectory in <RemoteThreadServer.directory>
        based on the provided or assigned `job_name`.  All submitted scripts/files
        (through either <RemoteThreadJob.run_script> or <RemoteThreadJob.submit_script>)
        are copied to and run in this directory.

        Using multiple processors is left to the user to pass the appropriate `mpirun`
        command (for example) to <RemoteThreadJob.run_script> or <RemoteThreadJob.submit_script>.

        Arguments
        -------------
        * `server` (<RemoteThreadServer>, optional, default=None): server to
            use when running the job.  If `server` is not provided, `host` must
            be provided.
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <RemoteThreadJob.job_name>.  This `job_name` will
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
        * `connect_to_existing` (bool, optional, default=None): NOT YET IMPLEMENTED
        """
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

        super().__init__(server, job_name,
                         conda_env=conda_env,
                         isolate_env=isolate_env,
                         job_submitted=connect_to_existing)

    @property
    def _pid(self):
        """
        """
        if self.job_status != 'running':
            raise ValueError("job is not currently running")

        return self.server._run_server_cmd("cat {}".format(_os.path.join(self.remote_directory, "crimpl-nohup.pid")))

    def kill_job(self):
        """
        Kill a job by terminating the process by <LocalThreadJob.pid>

        Returns
        -----------
        * (string)
        """
        self.server._run_server_cmd("kill -9 {}".format(self._pid))
        self.server._run_server_cmd("echo \'killed\' > {}/crimpl-job.status".format(self.remote_directory))

    def run_script(self, script, files=[], trial_run=False):
        """
        Run a script on the server in the <<class>.conda_env>,
        and wait for it to complete.

        This is useful for short installation/setup scripts that do not belong
        in the scheduled job.

        The resulting `script` and `files` are copied to <RemoteThreadJob.remote_directory>
        on the remote server and then `script` is executed via ssh.

        See <RemoteThreadJob.submit_script> to submit a script
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
        cmds = self.server._submit_script_cmds(script, files, [],
                                               use_scheduler=False,
                                               directory=self.remote_directory,
                                               conda_env=self.conda_env,
                                               isolate_env=self.isolate_env,
                                               job_name=None,
                                               terminate_on_complete=False,
                                               use_nohup=False,
                                               install_conda=False)
        if trial_run:
            return cmds

        for cmd in cmds:
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)
            self.server._run_server_cmd(cmd)

        return

    def submit_script(self, script, files=[],
                      nprocs=None,
                      ignore_files=[],
                      wait_for_job_status=False,
                      trial_run=False):
        """
        Submit a script to the server in the <<class>.conda_env>.

        This will copy `script` (modified with the provided slurm options) and
        `files` to <RemoteThreadJob.remote_directory> on the remote server and
        run the script in a thread (without a scheduler).  To check on its status,
        see <RemoteThreadJob.job_status>.

        To check on any expected output files, call <RemoteThreadJob.check_output>.

        See <RemoteThreadJob.run_script> to run a script and wait for it to complete.

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
        * `ignore_files` (list, optional, default=[]): list of filenames on the
            remote server to ignore when calling <<class>.check_output>
        * `wait_for_job_status` (bool or string or list, optional, default=False):
            Whether to wait for a specific job_status.  If True, will default to
            'complete'.  See also <RemoteThreadJob.wait_for_job_status>.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed.

        Returns
        ------------
        * <RemoteThreadJob>

        Raises
        ------------
        * ValueError: if a script has already been submitted within this
            <RemoteThreadJob> instance.
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        if "crimpl_submit_script.sh" in self.ls:
            raise ValueError("job already submitted.  Create a new job or call resubmit_job")

        cmds = self.server._submit_script_cmds(script, files, ignore_files,
                                               use_scheduler=False,
                                               directory=self.remote_directory,
                                               conda_env=self.conda_env,
                                               isolate_env=self.isolate_env,
                                               job_name=self.job_name,
                                               use_nohup=True,
                                               install_conda=False)

        if trial_run:
            return cmds

        for cmd in cmds:
            if cmd is None: continue
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)

            if cmd is None: continue
            self.server._run_server_cmd(cmd, detach="nohup" in cmd)

        self._job_submitted = True
        self._input_files = None

        if wait_for_job_status:
            self.wait_for_job_status(wait_for_job_status)

        return self

        if self._slurm_id is not None:
            raise ValueError("a job is already submitted.")

    def resubmit_script(self):
        """
        Resubmit an existing job script if <<class>.job_status> one of: complete,
        failed, killed.
        """
        status = self.job_status
        if status not in ['complete', 'failed', 'killed']:
            raise ValueError("cannot resubmit script with job_status='{}'".format(status))

        self.server._run_server_cmd("cd {directory}; nohup bash {remote_script} &".format(directory=self.remote_directory,
                                                                                          remote_script='crimpl_submit_script.sh'))





class RemoteThreadServer(_common.SSHServer):
    _JobClass = RemoteThreadJob
    def __init__(self, host, directory='~/crimpl', ssh='ssh', scp='scp',
                 server_name=None):
        """
        Connect to a remote server running jobs in threads (no scheduler).

        To create a new job, use <RemoteThreadServer.create_job> or to connect
        to a previously created job, use <RemoteThreadServer.get_job>.

        Arguments
        -----------
        * `host` (string): host of the remote server.  Must be passwordless ssh-able.
            See <RemoteThreadServer.host>
        * `directory` (string, optional, default='~/crimpl'): root directory of all
            jobs to run on the remote server.  The directory will be created
            if it does not already exist.
        * `ssh` (string, optional, default='ssh'): command (and any arguments in
            addition to `host`) to ssh to the remote server.
        * `scp` (string, optional, default='scp'): command (and any arguments)
            to copy files to the remote server.
        * `server_name` (string): name to assign to the server.  If not provided,
            will be adopted automatically from `host` and available from
            <RemoteThreadServer.server_name>.
        """
        self._host = host

        if server_name is None:
            server_name = host.split("@")[-1]

        self._ssh = ssh
        self._scp = scp
        self._server_name = server_name

        super().__init__(directory)
        self._dict_keys = ['host', 'directory', 'ssh', 'scp']

    @property
    def ssh(self):
        """
        """
        return self._ssh

    @property
    def scp(self):
        """
        """
        return self._scp

    @property
    def host(self):
        """
        host of the remote machine.  Should be passwordless ssh-able for the current user

        Returns
        ---------
        * (string)
        """
        return self._host

    @property
    def _ssh_cmd(self):
        """
        ssh command to the server

        Returns
        ----------
        * (string)
        """

        return "{} {}".format(self.ssh, self.host)

    @property
    def scp_cmd_to(self):
        """
        scp command to copy files to the server.

        Returns
        ----------
        * (string): command with "{}" placeholders for `local_path` and `server_path`.
        """

        return "%s {local_path} %s:{server_path}" % (self.scp, self.host)

    @property
    def scp_cmd_from(self):
        """
        scp command to copy files from the server.

        Returns
        ----------
        * (string): command with "{}" placeholders for `server_path` and `local_path`.
        """

        return "%s %s:{server_path} {local_path}" % (self.scp, self.host)

    def create_job(self, job_name=None,
                   conda_env=None, isolate_env=False):
        """
        Create a child <RemoteThreadJob> instance.

        Arguments
        -----------
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <RemoteThreadJob.job_name>.  This `job_name` will
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


        Returns
        ---------
        * <RemoteThreadJob>
        """
        return self._JobClass(server=self, job_name=job_name,
                              conda_env=conda_env,
                              isolate_env=isolate_env,
                              connect_to_existing=False)

    def submit_job(self, script, files=[],
                   job_name=None,
                   conda_env=None, isolate_env=False,
                   ignore_files=[],
                   wait_for_job_status=False,
                   trial_run=False):
        """
        Shortcut to <RemoteThreadServer.create_job> followed by <RemoteThreadJob.submit_script>.

        Arguments
        --------------
        * `script`: passed to <RemoteThreadJob.submit_script>
        * `files`: passed to <RemoteThreadJob.submit_script>
        * `conda_env`: passed to <RemoteThreadServer.create_job>
        * `isolate_env`: passed to <RemoteThreadServer.create_job>
        * `ignore_files`: passed to <RemoteThreadJob.submit_script>
        * `wait_for_job_status`: passed to <RemoteThreadJob.submit_script>
        * `trial_run`: passed to <RemoteThreadJob.submit_script>

        Returns
        --------------
        * <RemoteThreadJob>
        """
        j = self.create_job(job_name=job_name,
                            conda_env=conda_env,
                            isolate_env=isolate_env)

        return j.submit_script(script, files=files,
                               ignore_files=ignore_files,
                               wait_for_job_status=wait_for_job_status,
                               trial_run=trial_run)

    def run_script(self, script, files=[], conda_env=None, trial_run=False):
        """
        Run a script on the server in the `conda_env`, and wait for it to complete.

        The files are copied and executed in <RemoteThreadServer.directory> directly
        (whereas <RemoteThreadJob> scripts are executed in subdirectories).

        This is useful for short installation/setup scripts that do not belong
        in the scheduled job.

        The resulting `script` and `files` are copied to <RemoteThreadServer.directory>
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
        cmds = self._submit_script_cmds(script, files, [],
                                        use_scheduler=False,
                                        directory=self.directory,
                                        conda_env=conda_env,
                                        isolate_env=False,
                                        job_name=None,
                                        terminate_on_complete=False,
                                        use_nohup=False,
                                        install_conda=False)

        if trial_run:
            return cmds

        for cmd in cmds:
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)
            self._run_server_cmd(cmd)

        return
