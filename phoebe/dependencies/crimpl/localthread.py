
# from time import sleep as _sleep
import os as _os
import subprocess as _subprocess

from . import common as _common

class LocalThreadJob(_common.ServerJob):
    def __init__(self, server=None,
                 job_name=None,
                 conda_env=None, isolate_env=False,
                 connect_to_existing=None):
        """
        Create and submit a job on a <LocalThreadServer>.

        Under-the-hood, this creates a subdirectory in <LocalThreadServer.directory>
        based on the provided or assigned `job_name`.  All submitted scripts/files
        (through either <LocalThreadJob.run_script> or <LocalThreadJob.submit_script>)
        are copied to and run in this directory.

        Arguments
        -------------
        * `server` (<LocalThreadServer>, optional, default=None): server to
            use when running the job.
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <LocalThreadJob.job_name>.  This `job_name` will
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

        # run ls on

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
    def remote_directory(self):
        """
        Access the **job** subdirectory location in the server directory.

        Returns
        ----------
        * (string)
        """
        return _os.path.join(self.server.directory, "crimpl-job-{}".format(self.job_name))

    @property
    def pid(self):
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
        self.server._run_server_cmd("kill -9 {}".format(self.pid))
        self.server._run_server_cmd("echo \'killed\' > {}/crimpl-job.status".format(self.remote_directory))

    def run_script(self, script, files=[], trial_run=False):
        """
        Run a script on the server in the <<class>.conda_env>,
        and wait for it to complete.

        This is useful for short installation/setup scripts that do not belong
        in the scheduled job.

        The resulting `script` and `files` are copied to <LocalThreadJob.remote_directory>
        in the server directory and then `script` is executed.

        See <LocalThreadJob.submit_script> to submit a script via the slurm scheduler
        and leave running in the background on the server.

        Arguments
        ----------------
        * `script` (string or list): shell script to run in the server directory,
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
                                               use_nohup=False,
                                               install_conda=False)
        if trial_run:
            return cmds

        for cmd in cmds:
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)
            _common._run_cmd(cmd)

        return

    def submit_script(self, script, files=[],
                      ignore_files=[],
                      wait_for_job_status=False,
                      trial_run=False):
        """
        Submit a script to the server in the <<class>.conda_env>.

        This will copy `script` and `files` to <LocalThreadJob.remote_directory>
        in the server directory and run in a thread.
        To check on its status, see <LocalThreadJob.job_status>.

        To check on any expected output files, call <LocalThreadJob.check_output>.

        See <LocalThreadJob.run_script> to run a script and wait for it to complete.

        Arguments
        ----------------
        * `script` (string or list): shell script to run in the server directory,
            including any necessary installation steps.  Note that the script
            can call any other scripts in `files`.  If a string, must be the
            path of a valid file which will be copied to the server.  If a list,
            must be a list of commands (i.e. a newline will be placed between
            each item in the list and sent as a single script to the server).
        * `files` (list, optional, default=[]): list of paths to additional files
            to copy to the server required in order to successfully execute
            `script`.
        * `ignore_files` (list, optional, default=[]): list of filenames in the
            server directory to ignore when calling <<class>.check_output>
        * `wait_for_job_status` (bool or string or list, optional, default=False):
            Whether to wait for a specific job_status.  If True, will default to
            'complete'.  See also <LocalThreadJob.wait_for_job_status>.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed.

        Returns
        ------------
        * <LocalThreadJob>

        Raises
        ------------
        * ValueError: if a script has already been submitted within this
            <LocalThreadJob> instance.
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
            _common._run_cmd(cmd, detach="nohup" in cmd)

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

        self.server._run_server_cmd("cd {directory}; nohup bash {remote_script} &".format(directory=self.remote_directory,
                                                                                          remote_script='crimpl_submit_script.sh'))




class LocalThreadServer(_common.Server):
    _JobClass = LocalThreadJob
    def __init__(self, directory='~/crimpl', server_name=None):
        """
        Run scripts and jobs in threads in an isolated directory on the local machine.

        To create a new job, use <LocalThreadServer.create_job> or to connect
        to a previously created job, use <LocalThreadServer.get_job>.

        Arguments
        -----------
        * `directory` (string, optional, default='~/crimpl'): root directory of all
            jobs to run in the server directory.  The directory will be created
            if it does not already exist.
        * `server_name` (string): name to assign to the server.  If not provided,
            will be adopted automatically from `host` and available from
            <LocalThreadServer.server_name>.
        """
        if server_name is None:
            server_name = "local:{}".format(_os.path.basename(directory.strip("/")))

        self._server_name = server_name

        super().__init__(directory)
        self._dict_keys = ['directory']

    def _run_server_cmd(self, cmd, exportpath=None):
        if exportpath is None:
            exportpath = 'conda' in cmd or 'crimpl_script.sh' in cmd

        if exportpath:
            cmd = "source {}/exportpath.sh; {}".format(self.directory, cmd)
        else:
            cmd = cmd

        return _common._run_cmd(cmd)

    @property
    def directory(self):
        return _os.path.abspath(_os.path.expanduser(self._directory))

    @property
    def scp_cmd_to(self):
        """
        scp (cp in this case) command to copy files to the server directory.

        Returns
        ----------
        * (string): command with "{}" placeholders for `local_path` and `server_path`.
        """

        return "cp {local_path} {server_path}"

    @property
    def scp_cmd_from(self):
        """
        scp (cp in this case) command to copy files from the server directory.

        Returns
        ----------
        * (string): command with "{}" placeholders for `server_path` and `local_path`.
        """

        return "cp {server_path} {local_path}"

    @property
    def ssh_cmd(self):
        """
        ssh command to the server

        Returns
        ----------
        * (string): command with "{}" placeholders for the command to run in the server directory.
        """
        return "source %s/exportpath.sh; {}" % (self.directory)

    @property
    def ls(self):
        """
        Run and return the output of `ls` on the server directory (for all jobs).

        Returns
        -----------
        * (string)
        """
        return self.server._run_server_cmd("ls")

    def create_job(self, job_name=None,
                   conda_env=None, isolate_env=False):
        """
        Create a child <LocalThreadJob> instance.

        Arguments
        -----------
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <LocalThreadJob.job_name>.  This `job_name` will
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
        * <LocalThreadJob>
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
        Shortcut to <LocalThreadServer.create_job> followed by <LocalThreadJob.submit_script>.

        Arguments
        --------------
        * `script`: passed to <LocalThreadJob.submit_script>
        * `files`: passed to <LocalThreadJob.submit_script>
        * `job_name`: passed to <LocalThreadServer.create_job>
        * `conda_env`: passed to <LocalThreadServer.create_job>
        * `isolate_env`: passed to <LocalThreadServer.create_job>
        * `ignore_files`: passed to <LocalThreadJob.submit_script>
        * `wait_for_job_status`: passed to <LocalThreadJob.submit_script>
        * `trial_run`: passed to <LocalThreadJob.submit_script>

        Returns
        --------------
        * <LocalThreadJob>
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
        Run a script in the server directory in the `conda_env`, and wait for it to complete.

        The files are copied and executed in <LocalThreadServer.directory> directly
        (whereas <LocalThreadJob> scripts are executed in subdirectories).

        This is useful for short installation/setup scripts that do not belong
        in the scheduled job.

        The resulting `script` and `files` are copied to <LocalThreadServer.directory>
        and then `script` is executed.

        Arguments
        ----------------
        * `script` (string or list): shell script to run in the server directory,
            including any necessary installation steps.  Note that the script
            can call any other scripts in `files`.  If a string, must be the
            path of a valid file which will be copied to the server.  If a list,
            must be a list of commands (i.e. a newline will be placed between
            each item in the list and sent as a single script to the server).
        * `files` (list, optional, default=[]): list of paths to additional files
            to copy to the server directory required in order to successfully execute
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
                                        use_nohup=False,
                                        install_conda=False)

        if trial_run:
            return cmds

        for cmd in cmds:
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)
            _common._run_cmd(cmd)

        return
