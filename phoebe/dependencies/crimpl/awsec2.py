
from . import common as _common

from time import sleep as _sleep
import os as _os
import subprocess as _subprocess


try:
    import boto3 as _boto3
    _ec2_resource = _boto3.resource('ec2')
    _ec2_client = _boto3.client('ec2')
except ImportError:
    _boto3_installed = False
else:
    _boto3_installed = True


def _get_ec2_instance_type(nprocs):
    if nprocs < 2:
        return "t2.micro", 2
    elif nprocs < 4:
        return "t2.medium", 4
    elif nprocs < 8:
        return "t2.xlarge", 8
    elif nprocs < 16:
        return "t3.2xlarge", 16
    elif nprocs < 36:
        return "c4.4xlarge", 36
    elif nprocs < 58:
        return "c5.9xlarge", 58
    elif nprocs < 72:
        return "c5.18xlarge", 72
    elif nprocs == 96:
        return "c5.24xlarge", 96
    else:
        raise ValueError("no known instanceType for nprocs={}".format(nprocs))


# TODO:
# def list_aws_ec2_instances():

def list_awsec2_instances():
    """
    list all AWS EC2 instances (either <AWSEC2Server> or <AWSEC2Job>) managed by crimpl.

    Returns
    -----------
    * (dict): dictionary with instanceIds as keys and a dictionary as values
    """
    def _instance_to_dict(instance_dict):
        d = {tag['Key']: tag['Value'] for tag in instance_dict['Tags']}
        d['instanceId'] = instance_dict['InstanceId']
        d['state'] = instance_dict['State']['Name']
        if 'PublicIpAddress' in instance_dict.keys():
            d['ip'] = instance_dict['PublicIpAddress']
        return d

    def _get_instances(response):
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append(instance)
        return instances

    response = _ec2_client.describe_instances(Filters=[{'Name': 'tag-key', 'Values': ['crimpl.version']}])
    return {instance_dict['InstanceId']: _instance_to_dict(instance_dict) for instance_dict in _get_instances(response) if instance_dict['State']['Name'] != 'terminated'}

def list_awsec2_volumes():
    """
    list all AWS EC2 server volumes (for a <AWSEC2Server> and any of its created
    <AWSEC2Job>) managed by crimpl.

    Returns
    -----------
    * (dict): dictionary with volumeIds as keys and a dictionary as values
    """
    def _volume_to_dict(volume_dict):
        d = {tag['Key']: tag['Value'] for tag in volume_dict['Tags']}
        d['volumeId'] = volume_dict['VolumeId']
        return d

    response = _ec2_client.describe_volumes(Filters=[{'Name': 'tag-key', 'Values': ['crimpl.version']}])
    return {volume_dict['VolumeId']: _volume_to_dict(volume_dict) for volume_dict in response['Volumes']}

def delete_awsec2_volume(volumeId):
    """
    Manually delete an AWS EC2 volume by `volumeId`.

    Usually this will be done via <AWSEC2Server.delete_volume>.

    Arguments
    -----------
    * `volumeId` (string): AWS volumeId.  See <list_awsec2_volumes>.
    """
    _ec2_client.delete_volume(VolumeId=volumeId)

def delete_all_awsec2_volumes():
    """
    Manually delete all AWS EC2 volumes managed by crimpl.
    """
    for volumeId in list_awsec2_volumes().keys():
        delete_awsec2_volume(volumeId)

def terminate_awsec2_instance(instanceId):
    """
    Manually terminate an AWS EC2 instance by `instanceId`.

    Usually this will be done via <AWSEC2Server.terminate> or <AWSEC2Job.terminate>.

    Arguments
    ----------
    * `instanceId` (string): AWS EC2 instanceId.  See <list_awsec2_instances>.
    """
    _ec2_client.terminate_instances(InstanceIds=[instanceId])

def terminate_all_awsec2_instances():
    """
    Manually terminate all AWS EC2 instances managed by crimpl.
    """
    for instanceId in list_awsec2_instances().keys():
        terminate_awsec2_instance(instanceId)

class AWSEC2Job(_common.ServerJob):
    def __init__(self, server, job_name=None,
                 conda_env=None, isolate_env=False,
                 connect_to_existing=None,
                 nprocs=None, InstanceType=None,
                 ImageId='ami-03d315ad33b9d49c4', username='ubuntu',
                 start=False):
        """


        Arguments
        -------------
        * `server`
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <RemoteSlurmJob.job_name>.  This `job_name` will
            be necessary to reconnect to a previously submitted job.
        * `conda_env` (string or None, optional, default=None): name of
            the conda environment to use for the job, or None to use the
            'default' environment stored in the server crimpl directory.
        * `isolate_env` (bool, optional, default=False): whether to clone
            the `conda_env` for use in this job.  If True, any setup/installation
            done by this job will not affect the original environment and
            will not affect other jobs.  Note that the environment is cloned
            (and therefore isolated) at the first call to <<class>.run_script>
            or <<class>.submit_script>.  Setup in the parent environment can
            be done at the server level, but requires passing `conda_env`.
        * `connect_to_existing` (bool, optional, default=None): NOT YET IMPLEMENTED
        * `nprocs`
        * `InstanceType`
        * `ImageId`
        * `username`
        * `start`
        """
        if not _boto3_installed:
            raise ImportError("boto3 and \"aws config\" required for {}".format(self.__class__.__name__))

        # TODO: think about where nprocs should be defined.  We need it **BY**
        # the first call to start, but it might be nice to allow it to be passed
        # to run/submit_script to be consistent with RemoteSlurmJob.  Or we could
        # require nprocs for RemoteSlurmJob on init.
        # So crimpl.RemoteSlurm('terra').new_job(nprocs=...).submit_job(...)

        if connect_to_existing is None:
            if job_name is None:
                connect_to_existing = False
            else:
                connect_to_existing = True

        # TODO: try to use self.existing_jobs (which requires ssh) if a server is currently running?
        # otherwise we have to worry here about completed/cancelled/failed jobs not showing up in the list
        # in the case of a completed/cancelled/failed job with no running instance, we still want to allow connect_to_existing, we just want to set instanceId to None to a new one is created instead
        job_matches_running = [d for d in list_awsec2_instances().values() if d['crimpl.level'] == 'job' and d['crimpl.server_name'] == server.server_name and (d['crimpl.job_name'] == job_name or job_name is None)]

        if connect_to_existing:
            if len(job_matches_running) == 1:
                instanceId = job_matches_running[0]['instanceId']
                job_name = job_matches_running[0]['crimpl.job_name']
            elif len(job_matches_running) > 1:
                raise ValueError("{} running jobs found on {} server.  Provide job_name or create a new job".format(len(job_matches_running), server.server_name))
            else:
                job_matches_all = [j for j in self.existing_jobs if j==job_name or job_name is None]
                if len(job_matches_all) == 1:
                    instanceId = None
                    job_name = job_name
                else:
                    raise ValueError("{} total matching jobs found on {} server.  Please provide job_name or create a new job".format(len(job_matches_all), server.server_name))
        else:
            instanceId = None
            if job_name is None:
                job_name = _common._new_job_name()
            elif len(job_matches_running):
                raise ValueError("job_name={} already exists on {} server".format(job_name, server.server_name))

            # technically we should check this, but that requires launching the instance... can we instead check it before creating the directory?
            # else:
                # job_matches_all = [j for j in self.existing_jobs if j==job_name or job_name is None]
                # if len(job_matches_all):
                    # raise ValueError("job_name={} already exists on {} server".format(job_name, server.server_name))


        self._instanceId = instanceId
        self._username = username

        if nprocs is not None:
            if InstanceType is not None:
                raise ValueError("cannot provide both nprocs and instanceType")
            InstanceType, nprocs = _get_ec2_instance_type(nprocs=nprocs)

        self._nprocs = nprocs

        super().__init__(server, job_name,
                         conda_env=conda_env,
                         isolate_env=isolate_env,
                         job_submitted=connect_to_existing)


        self._ec2_init_kwargs = {'InstanceType': InstanceType,
                                 'ImageId': ImageId,
                                 'KeyName': self.server._ec2_init_kwargs.get('KeyName'),
                                 'SubnetId': self.server._ec2_init_kwargs.get('SubnetId'),
                                 'SecurityGroupIds': self.server._ec2_init_kwargs.get('SecurityGroupIds'),
                                 'MaxCount': 1,
                                 'MinCount': 1,
                                 'InstanceInitiatedShutdownBehavior': 'terminate'}

        if start:
            return self.start()  # wait is always True

    def __repr__(self):
        return "<AWSEC2Job job_name={}, instanceId={}>".format(self.job_name, self.instanceId)

    @property
    def nprocs(self):
        """
        Number of processors available on the **job** EC2 instance.

        Returns
        ---------
        * (int)
        """
        return self._nprocs

    @property
    def state(self):
        """
        Current state of the **job** EC2 instance.  Can be one of:
        * pending
        * running
        * shutting-down
        * terminated
        * stopping
        * stopped

        See also:

        * <AWSEC2Server.state>

        Returns
        -----------
        * (string)
        """
        if self._instanceId is None:
            return 'not-started'

        # pending, running, shutting-down, terminated, stopping, stopped
        return self._instance.state['Name']

    @property
    def instanceId(self):
        """
        instanceId of the **job** EC2 instance.

        Returns
        -----------
        * (string)
        """
        return self._instanceId

    @property
    def _instance(self):
        return _ec2_resource.Instance(self._instanceId)

    @property
    def username(self):
        """
        Username on the **job** EC2 instance.

        Returns
        ------------
        * (string)
        """
        return self._username

    @property
    def ip(self):
        """
        Public IP address of the **job** EC2 instance.

        Returns
        ------------
        * (string)
        """
        return self._instance.public_ip_address

    def wait_for_state(self, state='running', error_if=[], sleeptime=0.5):
        """
        Wait for the **job** EC2 instance to reach a specified state.

        Arguments
        ----------
        * `state` (string or list, optional, default='running'): state or states
            to exit the wait loop successfully.
        * `error_if` (string or list, optional, default=[]): state or states
            to exit the wait loop and raise an error.
        * `sleeptime` (float, optional, default): seconds to wait between
            successive state checks.

        Returns
        ----------
        * (string) <AWSEC2Job.state>
        """
        if isinstance(state, string):
            state = [state]

        if isinstance(error_if, string):
            error_if = [error_if]

        while True:
            curr_state = self.state
            if curr_state in state:
                break
            if curr_state in error_if:
                raise ValueError("state={}".format(curr_state))

            _sleep(sleeptime)

        return state

    def start(self, wait=True):
        """
        Start the **job** EC2 instance.

        A running EC2 instance charges per CPU-second.  See AWS pricing for more details.

        Note that <AWSEC2.submit_script> will automatically start the instance
        if not already manually started.

        Arguments
        ------------
        * `wait` (bool, optional, default=True): `wait` is required to be True
            in order to attach the **server** volume and is only exposed as a
            kwarg for consistent calling signature (the passed value will be ignored)

        Return
        --------
        * (string) <AWSEC2Job.state>
        """
        state = self.state
        if state in ['running', 'pending']:
            return
        elif state in ['terminated', 'shutting-down', 'stopping']:
            raise ValueError("cannot start: current state is {}".format(state))

        server_state = self.server.state
        if server_state not in ['not-started', 'terminated']:
            print("# crimpl: terminating server EC2 instance before starting job EC2 instance...")
            self.server.terminate(wait=True)

        if self.instanceId is None:
            ec2_init_kwargs = self._ec2_init_kwargs.copy()
            ec2_init_kwargs['TagSpecifications'] = [{'ResourceType': 'instance', 'Tags': [{'Key': 'crimpl.level', 'Value': 'job'}, {'Key': 'crimpl.job_name', 'Value': self.job_name}, {'Key': 'crimpl.server_name', 'Value': self.server.server_name}, {'Key': 'crimpl.version', 'Value': _common.__version__}]}]
            response = _ec2_client.run_instances(**ec2_init_kwargs)
            self._instanceId = response['Instances'][0]['InstanceId']
        else:
            response = _ec2_client.start_instances(InstanceIds=[self.instanceId], DryRun=False)

        print("# crimpl: waiting for job EC2 instance to start...")
        self._instance.wait_until_running()

        # attach the volume
        print("# crimpl: attaching server volume {} to job EC2 instance...".format(self.server.volumeId))
        try:
            response = _ec2_client.attach_volume(Device='/dev/sdh',
                                                 InstanceId=self.instanceId,
                                                 VolumeId=self.server.volumeId)
        except Exception as e:
            print("# crimpl: attaching server volume failed, stopping EC2 instance...")
            self.stop()
            raise e

        print("# crimpl: waiting 30s for initialization checks to complete...")
        _sleep(30)

        print("# crimpl: mounting server volume on job EC2 instance...")
        cmd = self.server.ssh_cmd.format("sudo mkdir crimpl_server; {mkfs_cmd}sudo mount /dev/xvdh crimpl_server; sudo chown {username} crimpl_server; sudo chgrp {username} crimpl_server".format(username=self.username, mkfs_cmd="sudo mkfs -t xfs /dev/xvdh; " if self.server._volume_needs_format else ""))

        while True:
            try:
                _common._run_cmd(cmd)
            except _subprocess.CalledProcessError:
                print("# crimpl: ssh call to mount server failed, waiting another 10s and trying again")
                _sleep(10)
            else:
                break

        print("# crimpl: initializing conda on job EC2 instance...")
        _common._run_cmd("conda init")

        self.server._volume_needs_format = False

        return self.state

    def stop(self, wait=True):
        """
        Stop the **job** EC2 instance.

        Once stopped, the EC2 instance can be restarted via <AWSEC2Job.start>.

        A stopped EC2 instance still results in charges for the storage, but no longer
        charges for the CPU time.  See AWS pricing for more details.

        Arguments
        -------------
        * `wait` (bool, optional, default=True): whether to wait for the server
            to reach a stopped <AWSEC2Job.state>.

        Return
        --------
        * (string) <AWSEC2Job.state>
        """
        response = _ec2_client.stop_instances(InstanceIds=[self.instanceId], DryRun=False)
        if wait:
            print("# crimpl: waiting for job EC2 instance to stop...")
            self._instance.wait_until_stopped()
        return self.state

    def terminate(self, wait=True):
        """
        Terminate the **job** EC2 instance.

        Once terminated, the EC2 instance cannot be restarted, but will no longer
        result in charges.  See also <AWSEC2Job.stop> and AWS pricing for more details.

        Arguments
        -------------
        * `wait` (bool, optional, default=True): whether to wait for the server
            to reach a terminated <AWSEC2Job.state>.

        Return
        --------
        * (string) <AWSEC2Job.state>
        """
        response = _ec2_client.terminate_instances(InstanceIds=[self.instanceId], DryRun=False)
        if wait:
            print("# crimpl: waiting for job EC2 instance to terminate...")
            self._instance.wait_until_terminated()
        return self.state


    @property
    def job_status(self):
        """
        Return the status of the job.

        If a job has been submitted, but the EC2 instance no longer exists or has
        been stopped or terminated, then the job is assumed to have completed
        (although in reality it may have failed or the EC2 instance terminated
        manually).

        If the EC2 instance is running, then the status file in the job-directory
        is checked and will return either 'running' or 'complete'.


        Returns
        -----------
        * (string): one of not-submitted, running, complete, unknown
        """
        if not self._job_submitted:
            return 'not-submitted'

        # first handle the case where we're not connected to an EC2 instance
        if self._instanceId is None:
            # then we should have connected if one existed already.  In this case
            # the instance is already terminated (job_status: complete/failed)
            # since we know the job has been submitted
            return 'complete'

        # if we've made it this far, then there is an existing EC2 instance
        # assigned to this job.  Let's make some assumptions based on its state
        state = self.state
        if state in ['terminated', 'stopped', 'stopping', 'shutting-down']:
            # then since the job has been submitted, we'll assume complete/failed
            return 'complete'
        elif state in ['pending']:
            # then the server is starting back up?  In theory this means it was
            # already stopped/terminated
            return 'complete'
        else:
            # then we have an instance and we can check its state via ssh
            try:
                response = self.server._run_ssh_cmd("cat {}".format(_os.path.join(self.remote_directory, "crimpl-job.status")))
            except _subprocess.CalledProcessError:
                return 'unknown'

            return response

    def kill_job(self, delete_volume=False):
        """
        Kill a job by terminating the EC2 instance.

        Arguments
        -----------
        * `delete_volume` (bool, optional, default=False): whether to delete the volume
        """
        if delete_volume:
            return self.server.delete_volume()
        return self.server.terminate()

    def run_script(self, script, files=[], trial_run=False):
        """
        Run a script on the **job** server in the <<class>.conda_env>,
        and wait for it to complete.

        See <AWSEC2Job.submit_script> to submit a script to leave running in the background.

        For scripts that only require one processor and may take some time (i.e.
        installation and setup script), consider using <AWSEC2Server.submit_script>
        before initializing the **job** EC2 instance.

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
            that would be sent to the server are returned but not executed
            (and the server is not started automatically - so these may include
            an <ip> placeholder).


        Returns
        ------------
        * None

        Raises
        ------------
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        if self.state != 'running' and not trial_run:
            self.start()  # wait is always True

        cmds = self.server._submit_script_cmds(script, files, [],
                                               use_slurm=False,
                                               directory=self.remote_directory,
                                               conda_env=self.conda_env,
                                               isolate_env=self.isolate_env,
                                               job_name=None,
                                               terminate_on_complete=False,
                                               use_nohup=False,
                                               install_conda=True)

        if trial_run:
            return cmds

        for cmd in cmds:
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)
            _common._run_cmd(cmd)

        return

    def submit_script(self, script, files=[], terminate_on_complete=True,
                      ignore_files=[],
                      wait_for_job_status=False,
                      trial_run=False):
        """
        Submit a script to the server.

        This will call <AWSEC2Job.start> and wait for
        the server to intialize if it is not already running.  Once running,
        `script` and `files` are copied to the server, and `script` is executed
        in a screen session at which point this method will return.

        To check on any expected output files, call <AWSEC2Job.check_output>.

        See <AWSEC2Job.run_script> to run a script and wait for it to complete.

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
        * `terminate_on_complete` (bool, optional, default=True): whether to terminate
            the EC2 instance once `script` has completed.  This is useful for
            long jobs where you may not immediately be able to pull the results
            to minimize costs.  In this case, the <AWSEC2Job.server> EC2 instance
            will be restarted when calling <AWSEC2Job.check_output> with access
            to the same storage volume.
        * `ignore_files` (list, optional, default=[]): list of filenames on the
            remote server to ignore when calling <<class>.check_output>
        * `wait_for_job_status` (bool or string or list, optional, default=False):
            Whether to wait for a specific job_status.  If True, will default to
            'complete'.  See also <AWSEC2Job.wait_for_job_status>.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed
            (and the server is not started automatically - so these may include
            an <ip> placeholder).


        Returns
        ------------
        * <AWSEC2Job>

        Raises
        ------------
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        if self.state != 'running' and not trial_run:
            self.start() # wait is always True

        cmds = self.server._submit_script_cmds(script, files, ignore_files,
                                               use_slurm=False,
                                               directory=self.remote_directory,
                                               conda_env=self.conda_env,
                                               isolate_env=self.isolate_env,
                                               job_name=self.job_name,
                                               terminate_on_complete=terminate_on_complete,
                                               use_nohup=True,
                                               install_conda=True)

        if trial_run:
            return cmds

        for cmd in cmds:
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

        if self.state != 'running' and not trial_run:
            self.start() # wait is always True

        # TODO: discriminate between run_script and submit_script filenames and don't allow multiple calls to submit_script
        self.server._run_ssh_cmd("cd {directory}; nohup bash {remote_script} &".format(directory=self.remote_directory,
                                                                                      remote_script='crimpl_script.sh'))


    def check_output(self, server_path=None, local_path="./",
                     terminate_if_server_started=False):
        """
        Attempt to copy a file(s) back from the remote server.

        Arguments
        -----------
        * `server_path` (string or list or None, optional, default=None): path(s)
            (relative to `directory`) on the server of the file(s) to retrieve.
            If not provided or None, will default to <<class>.output_files>.
            See also: <<class>.ls> or <<class>.job_files> for a full list of
            available files on the remote server.
        * `local_path` (string, optional, default="./"): local path to copy
            the retrieved file.
        * `terminate_if_server_started` (bool, optional, default=False): whether
            the server EC2 instance should immediately be terminated if it
            was started in order to retrieve the files from the volume.
            The server EC2 instance can manually be terminated via
            <AWSEC2Server.terminate>.

        Returns
        ----------
        * None

        """

        did_restart = False
        if self.state != 'running' and self.server.state != 'running':
            self.server.start()  # wait is always True
            did_restart = True

        super().check_output(server_path, local_path)

        if did_restart and terminate_if_server_started:
            self.server.terminate()

class AWSEC2Server(_common.Server):
    _JobClass = AWSEC2Job
    def __init__(self, server_name=None, volumeId=None,
                       instanceId=None,
                       KeyFile=None, KeyName=None,
                       SubnetId=None, SecurityGroupId=None):
        """
        Connect to an existing <AWSEC2Server> by `volumeId`.

        To create a new server, use <AWSEC2Server.new> instead.

        * `server_name` (string, optional, default=None): internal name of the
            **existing** server to retrieve.  To create a new server, call
            <AWSEC2Server.new> instead.  Either `server_name` or `volumeId` must
            be provided.
        * `volumeId` (string, optional, default=None): AWS internal `volumeId`
            of the shared AWS EC2 volume instance.  Either `server_name` or
            `volumeId` must be provided.
        * `instanceId` (string, optional, default=None): AWS internal `instanceId`
            of the **server** EC2 instance.  If not provided, will be determined
            from `server_name` and/or `volumeId`.
        * `KeyFile` (string, required, default=None): path to the KeyFile
        * `KeyName` (string, optional, default=None): AWS internal name corresponding
            to `KeyFile`.  If not provided, will be assumed to be `basename(KeyFile).split(.)[0]`.
        * `SubnetId` (string, required, default=None):
        * `SecurityGroupId` (string, required, default=None):

        Returns
        ------------
        * <AWSEC2Server>
        """
        if not _boto3_installed:
            raise ImportError("boto3 and \"aws config\" required for {}".format(self.__class__.__name__))

        if volumeId is None and server_name is None:
            raise ValueError("volumeId or server_name required.  To generate a new server, use new instead of __init__")

        sdicts = list_awsec2_volumes()
        server_match = [s for s in sdicts.values() if (s['volumeId']==volumeId or volumeId is None) and (s['crimpl.server_name']==server_name or server_name is None)]
        if len(server_match) == 1:
            server_name = server_match[0]['crimpl.server_name']
            volumeId = server_match[0]['volumeId']
        else:
            raise ValueError("{} existing EC2 volumes matched with server_name={} and volumeId={}.  Use new instead of __init__ to generate a new server.".format(len(server_match), server_name, volumeId))

        self._server_name = server_name
        self._volumeId = volumeId

        if instanceId is None:
            # try to find a matching instance
            server_match = [s for s in list_awsec2_instances().values() if s['crimpl.server_name'] == server_name]
            if len(server_match):
                instanceId = server_match[0]['instanceId']
            # otherwise we can leave at None and a new 1-proc EC2 instance will be instantiated and instanceId assigned
        else:
            server_match = [s for s in list_awsec2_instances().values() if s['crimpl.server_name']==server_name and s['instanceId']==instanceId]
            if len(server_match) != 1:
                raise ValueError("{} existing EC2 instances matched with server_name={} and instanceId={}".format(server_name, instanceId))

        self._instanceId = instanceId

        if KeyFile is None:
            raise ValueError("KeyFile must not be None")
        if not _os.path.isfile(_os.path.expanduser(KeyFile)):
            raise ValueError("KeyFile does not point to a file")

        self._KeyFile = KeyFile

        if KeyName is None:
            KeyName = _os.path.basename(KeyFile).split(".")[0]

        self._ec2_init_kwargs = {'KeyName': KeyName,
                                 'SubnetId': SubnetId,
                                 'SecurityGroupIds': [SecurityGroupId],
                                 'MaxCount': 1,
                                 'MinCount': 1}

        self._ImageId = 'ami-03d315ad33b9d49c4'
        self._username = "ubuntu"

        self._volume_needs_format = False

        # directory here is fixed at the point of the mounted volume
        super().__init__(directory="~/crimpl_server")

        self._dict_keys = ['server_name', 'VolumeId', 'instanceId', 'KeyFile', 'KeyName', 'SubnetId', 'SecurityGroupId']

    @classmethod
    def load(cls, name):
        # TODO: support loading from saved cache by name
        raise NotImplementedError()

    @classmethod
    def new(cls, server_name=None, volumeSize=4,
            KeyFile=None, KeyName=None,
            SubnetId=None, SecurityGroupId=None):
        """
        Create a new <AWSEC2Server> instance.

        This creates a blank AWS EC2 volume to be shared among both **server**
        and **job** AWS EC2 instances with `volumeSize` storage.

        Arguments
        -----------
        * `server_name` (string, optional, default=None): internal name to assign
            to the server.  If not provided, will be assigned automatically and
            available from <AWSEC2Server.server_name>.  Once created, the <AWSEC2Server>
            object can then be retrieved by name via <AWSEC2Server.__init__>.
        * `volumeSize` (int, optional, default=4): size, in GiB of the shared
            volume to create.  Once created, the volume begins accruing charges.
            See AWS documentation for pricing details.
        * `KeyFile` (string, required, default=None): path to the KeyFile
        * `KeyName` (string, optional, default=None): AWS internal name corresponding
            to `KeyFile`.  If not provided, will be assumed to be `basename(KeyFile).split(.)[0]`.
        * `SubnetId` (string, required, default=None):
        * `SecurityGroupId` (string, required, default=None):

        Returns
        ----------
        * <AWSEC2Server>
        """


        # Multi-Attach is available in the following Regions only:
        # For io2 volumes—us-east-2, eu-central-1, and ap-south-1.
        # For io1 volumes—us-east-1, us-west-2, eu-west-1, and ap-northeast-2.

        # iops: The number of I/O operations per second (IOPS).
        # The following are the supported values for each volume type:
        # io1 : 100-64,000 IOPS
        # io2 : 100-64,000 IOPS
        # response = _ec2_client.create_volume(Size=volumeSize,
        #                                      VolumeType='io1',
        #                                      Iops=100,  # TODO: expose
        #                                      MultiAttachEnabled=True,
        #                                      AvailabilityZone='us-east-1a')  # TODO: expose


        if server_name is None:
            # create a new server name
            server_names = [d['crimpl.server_name'] for d in list_awsec2_volumes()]
            server_name = 'awsec2{:02d}'.format(len(server_names)+1)

        volume_init_kwargs = {'Size': volumeSize,
                              'VolumeType': 'gp2',
                              'AvailabilityZone': 'us-east-1a'}  # TODO: expose
        volume_init_kwargs['TagSpecifications'] = [{'ResourceType': 'volume', 'Tags': [{'Key': 'crimpl.level', 'Value': 'server-volume'},
                                                                                       {'Key': 'crimpl.server_name', 'Value': server_name},
                                                                                       {'Key': 'crimpl.version', 'Value': _common.__version__}]}]

        response = _ec2_client.create_volume(**volume_init_kwargs)

        volumeId = response.get('VolumeId')

        self = cls(server_name=server_name, volumeId=volumeId,
                   KeyFile=KeyFile, KeyName=KeyName,
                   SubnetId=SubnetId, SecurityGroupId=SecurityGroupId)

        self._volume_needs_format = True
        return self

    def __repr__(self):
        return "<AWSEC2Server server_name={} volumeId={} instanceId={}>".format(self.server_name, self.volumeId, self.instanceId)

    @property
    def server_name(self):
        """
        internal name of the server.

        Returns
        ----------
        * (string)
        """
        return self._server_name

    @property
    def volumeId(self):
        """
        AWS internal volumeId for the **server** volume

        Returns
        ---------
        * (string)
        """
        return self._volumeId

    @property
    def _volume(self):
        return _ec2_resource.Volume(self._volumeId)

    def delete_volume(self, terminate_instances=True):
        """
        Delete the AWS EC2 **server** volume.  Once deleted, servers and jobs
        will no longer be accessible and a new <AWSEC2Server> instance must be created
        for additional submissions.

        If `terminate_instances` is True (as it is by default), any EC2 instances
        in <crimpl.list_awsec2_instances> with this <AWSEC2Server.server_name>
        will be terminated first.

        Arguments
        ----------
        * `terminate_instances` (bool, optional, default=True): whether to first
            check for any non-terminated **server** or **job** EC2 instances
            and terminate them.

        Returns
        -----------
        * None
        """
        if terminate_instances:
            matching_ec2s = [d for d in list_awsec2_instances().values() if d['crimpl.server_name'] == self.server_name]
            matching_instances = [d['instanceId'] for d in matching_ec2s]
            if len(matching_instances):
                print("# crimpl: terminating ec2 instances {}...".format(matching_instances))
                _ec2_client.terminate_instances(InstanceIds=matching_instances)

                print("# crimpl: waiting for ec2 instances to terminate...")
                for instanceId in matching_instances:
                    while _ec2_resource.Instance(instanceId).state['Name'] != 'terminated':
                        _sleep(0.5)

        # TODO: check status of server, MUST be terminated
        elif self._instanceId is not None and self.state != 'terminated':
            raise ValueError("server must be terminated before deleting volume")

        # TODO: other checks?  Make sure NO instance is attached to the volume and raise helpful errors
        print("# crimpl: deleting volume {}...".format(self.volumeId))
        self._volume.delete()
        self._volumeId = None

    @property
    def instanceId(self):
        """
        AWS internal instanceId of the **server** EC2 instance.

        Returns
        -----------
        * (string)
        """
        return self._instanceId

    @property
    def _instance(self):
        return _ec2_resource.Instance(self._instanceId)

    @property
    def state(self):
        """
        Current state of the **server** EC2 instance.  Can be one of:
        * pending
        * running
        * shutting-down
        * terminated
        * stopping
        * stopped

        See also:

        <AWSEC2Job.state>

        Returns
        -----------
        * (string)
        """
        if self._instanceId is None:
            return 'not-started'

        # pending, running, shutting-down, terminated, stopping, stopped
        return self._instance.state['Name']


    @property
    def username(self):
        """
        Username on the **server** EC2 instance.

        Returns
        ------------
        * (string)
        """
        return self._username

    @property
    def ip(self):
        """
        Public IP address of the **server** EC2 instance, if its running, or one
        of its children **job** EC2 instances, if available.

        Returns
        ------------
        * (string)
        """
        if self._instanceId is None or self.state != 'running':
            # then try to fetch another valid instanceId that IS running
            matching_ec2s = [d for d in list_awsec2_instances().values() if d['crimpl.server_name'] == self.server_name]
            # now we want one that is RUNNING
            for matching_ec2 in matching_ec2s:
                instance = _ec2_resource.Instance(matching_ec2['instanceId'])
                if instance.state['Name'] == 'running':
                    return instance.public_ip_address

        return self._instance.public_ip_address


    @property
    def _ssh_cmd(self):
        """
        ssh command to the **server** EC2 instance (or a child **job** EC2 instance
        if the **server** EC2 instance is not running).

        Returns
        ----------
        * (string): If the server is not yet started and <AWSEC2Server.ip> is not available,
            the ip will be replaced with {ip}
        """
        try:
            ip = self.ip
        except:
            ip = "{ip}"

        return "ssh -o \"StrictHostKeyChecking no\" -i {} {}@{}".format(self._KeyFile, self.username, ip)

    @property
    def scp_cmd_to(self):
        """
        scp command to copy files to the **server** EC2 instance (or a child **job** EC2 instance
        if the **server** EC2 instance is not running).

        Returns
        ----------
        * (string): command with "{}" placeholders for `local_path` and `server_path`.
            If the server is not yet started and <AWSEC2Server.ip> is not available,
            the ip will be replaced with {ip}
        """
        try:
            ip = self.ip
        except:
            ip = "{ip}"

        return "scp -i %s {local_path} %s@%s:{server_path}" % (self._KeyFile, self.username, ip)

    @property
    def scp_cmd_from(self):
        """
        scp command to copy files from the **server** EC2 instance (or a child **job** EC2 instance
        if the **server** EC2 instance is not running).

        Returns
        ----------
        * (string): command with "{}" placeholders for `server_path` and `local_path`.
            If the server is not yet started and <AWSEC2Server.ip> is not available,
            the ip will be replaced with {ip}
        """
        try:
            ip = self.ip
        except:
            ip = "{ip}"

        return "scp -i %s %s@%s:{server_path} {local_path}" % (self._KeyFile, self.username, ip)

    def create_job(self, job_name=None,
                   conda_env=None, isolate_env=False,
                   nprocs=4,
                   InstanceType=None,
                   ImageId='ami-03d315ad33b9d49c4', username='ubuntu',
                   start=False):
        """
        Create a child <AWSEC2Job> instance.

        Arguments
        -----------
        * `job_name` (string, optional, default=None): name for this job instance.
            If not provided, one will be created from the current datetime and
            accessible through <AWSEC2Job.job_name>.  This `job_name` will
            be necessary to reconnect to a previously submitted job.
        * `conda_env` (string or None, optional, default=None): name of
            the conda environment to use for the job, or None to use the
            'default' environment stored in the server crimpl directory.
        * `isolate_env` (bool, optional, default=False): whether to clone
            the `conda_env` for use in this job.  If True, any setup/installation
            done by this job will not affect the original environment and
            will not affect other jobs.  Note that the environment is cloned
            (and therefore isolated) at the first call to <<class>.run_script>
            or <<class>.submit_script>.  Setup in the parent environment can
            be done at the server level, but requires passing `conda_env`.
        * `nprocs` (int, optional, default=4): number of processors for the
            **job** EC2 instance.  The `InstanceType` will be determined and
            `nprocs` will be rounded up to the next available instance meeting
            those available requirements.
        * `InstanceType` (string, optional, default=None):
        * `ImageId` (string, optional, default=None):  ImageId of the **job**
            EC2 instance.  If None or not provided, will default to the same
            as the **server** EC2 instance (Ubuntu 20.04).
        * `username` (string, optional, default='ubuntu'): username required
            to log in to the **job** EC2 instance.  If None or not provided,
            will default to <AWSEC2Server.username>.
        * `start` (bool, optional, default=False): whether to immediately start
            the **job** EC2 instance.

        Returns
        ----------
        * <AWSEC2Job>
        """
        return self._JobClass(server=self, job_name=job_name,
                              conda_env=conda_env,
                              isolate_env=isolate_env,
                              nprocs=nprocs, InstanceType=InstanceType,
                              ImageId=self._ImageId if ImageId is None else ImageId,
                              username=self.username if username is None else username,
                              start=start, connect_to_existing=False)

    def submit_job(self, script, files=[],
                   job_name=None,
                   conda_env=None, isolate_env=False,
                   nprocs=4,
                   terminate_on_complete=True,
                   ignore_files=[],
                   wait_for_job_status=False,
                   trial_run=False
                   ):
        """
        Shortcut to <AWSEC2Server.create_job> followed by <AWSEC2Job.submit_script>.

        Arguments
        --------------
        * `script`: passed to <AWSEC2Job.submit_script>
        * `files`: passed to <AWSEC2Job.submit_script>
        * `job_name`: passed to <AWSEC2Server.create_job>
        * `conda_env`: passed to <AWSEC2Server.create_job>
        * `isolate_env`: passed to <AWSEC2Server.create_job>
        * `nprocs`: passed to <AWSEC2Server.create_job>
        * `terminate_on_complete`: passed to <AWSEC2Job.submit_script>
        * `ignore_files`: passed to <AWSEC2Job.submit_script>
        * `wait_for_job_status`: passed to <AWSEC2Job.submit_script>
        * `trial_run`: passed to <AWSEC2Job.submit_script>

        Returns
        --------------
        * <AWSEC2Job>
        """
        j = self.create_job(job_name=job_name,
                            conda_env=conda_env,
                            isolate_env=isolate_env,
                            nprocs=nprocs,
                            InstanceType=None,
                            ImageId='ami-03d315ad33b9d49c4', username='ubuntu',
                            start=False)

        return j.submit_script(script, files=files,
                               terminate_on_complete=terminate_on_complete,
                               ignore_files=ignore_files,
                               wait_for_job_status=wait_for_job_status,
                               trial_run=trial_run)


    def wait_for_state(self, state='running', sleeptime=0.5):
        """
        Wait for the **server** EC2 instance to reach a specified state.

        Arguments
        ----------
        * `state` (string or list, optional, default='running'): state or states
            to exit the wait loop successfully.
        * `error_if` (string or list, optional, default=[]): state or states
            to exit the wait loop and raise an error.
        * `sleeptime` (float, optional, default=5): number of seconds to wait
            between successive EC2 state checks.

        Returns
        ----------
        * (string) <AWSEC2Server.state>
        """
        if isinstance(state, str):
            state = [state]

        if isinstance(error_if, string):
            error_if = [error_if]

        while True:
            curr_state = self.state
            if curr_state in state:
                break
            if curr_state in error_if:
                raise ValueError("state={}".format(curr_state))

            _sleep(sleeptime)

        return state

    def start(self, wait=True):
        """
        Start the **server** EC2 instance.

        A running EC2 instance charges per CPU-second.  See AWS pricing for more details.

        Arguments
        ------------
        * `wait` (bool, optional, default=True): `wait` is required to be True
            in order to attach the **server** volume and is only exposed as a
            kwarg for consistent calling signature (the passed value will be ignored)

        Return
        --------
        * (string) <AWSEC2Server.state>
        """
        state = self.state
        if state in ['running', 'pending']:
            return
        elif state in ['terminated', 'shutting-down', 'stopping']:
            raise ValueError("cannot start: current state is {}".format(state))

        if self.instanceId is None:
            ec2_init_kwargs = self._ec2_init_kwargs.copy()
            ec2_init_kwargs['InstanceType'], nprocs = _get_ec2_instance_type(nprocs=1)
            ec2_init_kwargs['ImageId'] = self._ImageId # Ubuntu 20.04 ('ami-03d315ad33b9d49c4') - provide option for this?
            ec2_init_kwargs['TagSpecifications'] = [{'ResourceType': 'instance', 'Tags': [{'Key': 'crimpl.level', 'Value': 'server'}, {'Key': 'crimpl.server_name', 'Value': self.server_name}, {'Key': 'crimpl.volumeId', 'Value': self.volumeId}, {'Key': 'crimpl.version', 'Value': _common.__version__}]}]
            response = _ec2_client.run_instances(**ec2_init_kwargs)
            self._instanceId = response['Instances'][0]['InstanceId']
        else:
            response = _ec2_client.start_instances(InstanceIds=[self.instanceId])

        self._username = "ubuntu" # assumed - provide option for this?

        print("# crimpl: waiting for server EC2 instance to start before attaching server volume...")
        self._instance.wait_until_running()

        # attach the volume
        print("# crimpl: attaching server volume {}...".format(self.volumeId))


        # TODO: test if already attached anywhere else (possibly an instance stopped via sudo shutdown) and detach.
        # Or we could change shutdown behavior on the workers to terminate which should release them

        try:
            response = _ec2_client.attach_volume(Device='/dev/sdh',
                                                 InstanceId=self.instanceId,
                                                 VolumeId=self.volumeId)
        except Exception as e:
            print("# crimpl: attaching server volume failed, stopping EC2 instance...")
            self.stop()
            raise e

        print("# crimpl: waiting 30s for initialization checks to complete...")
        _sleep(30)

        print("# crimpl: mounting volume on server EC2 instance...")
        cmd = self.ssh_cmd.format("sudo mkdir crimpl_server; {mkfs_cmd}sudo mount /dev/xvdh crimpl_server; sudo chown {username} crimpl_server; sudo chgrp {username} crimpl_server".format(username=self.username, mkfs_cmd="sudo mkfs -t xfs /dev/xvdh; " if self._volume_needs_format else ""))
        _common._run_cmd(cmd)
        self._volume_needs_format = False

        return self.state

    def stop(self, wait=True):
        """
        Stop the **server** EC2 instance.

        Once stopped, the EC2 instance can be restarted via <AWSEC2Server.start>, including
        by creating a new <AWSEC2Server> instance with the correct <AWSEC2Server.instanceId>.

        A stopped EC2 instance still results in charges for the storage, but no longer
        charges for the CPU time.  See AWS pricing for more details.

        Arguments
        -------------
        * `wait` (bool, optional, default=True): whether to wait for the server
            to reach a stopped <AWSEC2Server.state>.

        Return
        --------
        * (string) <AWSEC2Server.state>
        """
        response = _ec2_client.stop_instances(InstanceIds=[self.instanceId], DryRun=False)
        if wait:
            print("# crimpl: waiting for server EC2 instance to stop...")
            self._instance.wait_until_stopped()
        return self.state

    def terminate(self, wait=True, delete_volume=False):
        """
        Terminate the **server** EC2 instance.

        Once terminated, the EC2 instance cannot be restarted, but will no longer
        result in charges.  See also <AWSEC2Server.stop> and AWS pricing for more details.

        Arguments
        -------------
        * `wait` (bool, optional, default=True): whether to wait for the server
            to reach a terminated <AWSEC2Server.state>.

        Return
        --------
        * (string) <AWSEC2Server.state>
        """
        response = _ec2_client.terminate_instances(InstanceIds=[self.instanceId], DryRun=False)
        if wait or delete_volume:
            print("# crimpl: waiting for server EC2 instance to terminate...")
            self._instance.wait_until_terminated()

        self._instanceId = None

        if delete_volume:
            _sleep(2)
            self.delete_volume()

        return self.state

    def run_script(self, script, files=[], conda_env=None, trial_run=False):
        """
        Run a script on the **server** EC2 instance (single processor) in the
        `conda_env`, and wait for it to complete.

        The files are copied and executed in <AWSEC2Server.directory> directly
        (whereas <AWSEC2Job> scripts are executed in subdirectories).

        This is useful for installation scripts, setting up virtual environments,
        etc, as the **server** EC2 instance only runs a single processor.  Once
        complete, setup a job with <AWSEC2Server.create_job> or <AWSEC2Server.get_job>
        to submit the compute job on more resources (via <AWSEC2Job.run_script>
        or <AWSEC2.submit_script>).

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
        * `conda_env` (string or None): name of the conda environment to
            run the script, or None to use the 'default' environment stored in
            the server crimpl directory.
        * `trial_run` (bool, optional, default=False): if True, the commands
            that would be sent to the server are returned but not executed
            (and the server is not started automatically - so these may include
            an <ip> placeholder).


        Returns
        ------------
        * None

        Raises
        ------------
        * TypeError: if `script` or `files` are not valid types.
        * ValueError: if the files referened by `script` or `files` are not valid.
        """
        if self.state != 'running' and not trial_run:
            self.start()  # wait is always True

        cmds = self._submit_script_cmds(script, files, [],
                                        use_slurm=False,
                                        directory=self.directory,
                                        conda_env=conda_env,
                                        isolate_env=False,
                                        job_name=None,
                                        terminate_on_complete=False,
                                        use_nohup=False,
                                        install_conda=True)

        if trial_run:
            return cmds

        for cmd in cmds:
            # TODO: get around need to add IP to known hosts (either by
            # expecting and answering yes, or by looking into subnet options)
            _common._run_cmd(cmd)

        return
