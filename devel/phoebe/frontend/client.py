#!/usr/bin/python

import os
from glob import glob
from uuid import uuid4
from time import sleep
import json

class Client(object):  # will be merged with bundle
    def __init__(self):
        self.servers = {}
        self.jobs = {}
        
    def add_server(self,server_name,mount_location,mpi=None):
        """
        add a new server instance
        
        @param server_name: internal label for the server
        @type server_name: str
        @param mount_location: local path to mount location of the server
        @type mount_location: str
        @param mpi: name of mpi to use on server (must be available on the server) - not supported yet
        @type mpi: str
        """
        server = Server(server_name,mount_location,mpi)
        self.servers[server_name] = server
        
    def get_server(self,server_name):
        """
        retrieve server by name
        
        @param server_name: internal label for the server
        @type server_name: str
        @return: server
        @rtype: object
        """
        return self.servers[server_name]
        
    def add_job(self,job_name,script_file,aux_files=[],return_files='all',**kwargs):
        """
        add a new job instance
        
        @param job_name: internal label for the job
        @type job_name: str
        @param script_file: path to the script to run on the server
        @type script_file: str
        @param aux_files: list of files to copy to server
        @type aux_files: list of str
        @param return_files: names of files to retrieve when finished
        @type return_files: list of strings or 'all'
        """
        job = Job(job_name,script_file,aux_files,return_files,**kwargs)
        self.jobs[job_name] = job
        
    def get_job(self,job_name):
        """
        retrieve job by name
        
        @param job_name: internal label for the job
        @type job_name: str
        @return: job
        @rtype: object
        """
        
        return self.jobs[job_name]
        
    def run_job_on_server(self,job_name,server_name):
        """
        shortcut to run an existing job on an existing server
        
        @param job_name: internal label for the job
        @type job_name: str
        @param server_name: internatl label for the server
        @type server_name: str
        """
        job = self.jobs[job_name]
        server = self.servers[server_name]
                
        job.start(server)
        
    
class Server(object):
    # server must be mounted locally (sshfs)
    # possibly add options for scp/ftp/webserver eventually
    
    def __init__(self,name,mount_location,mpi=None):
        self.name = name
        self.mount_location = mount_location
        self.mpi = mpi 
        
        #mpi should be a string that is setup on the daemon
        #the script will probably be responsible for pulling the mpi settings from the server and applying them
    
    def check_connection(self):
        """
        returns whether the server is mounted as expected
        """
        return os.path.exists(self.mount_location) and os.path.isdir(self.mount_location)
        
    def get_existing_job(self,dirname):
        pass
        
    def add_script_to_queue(self,task):
        """
        
        """
        if not self.check_connection():
            return False
        
        script_file = os.path.basename(task.script_file)
        
        f = open(os.path.join(self.mount_location,'server.tasks','%s.queue' % task.dirname),'w')
        f.write(json.dumps(task.as_dict()))
        f.close()
        
    def copy_file_to_server(self,dirname,fname):
        """
        """
        if not self.check_connection():
            return False
        # make sure formatted correctly and create directory if doesn't exist
        dirname = os.path.join(self.mount_location,dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        # copy file
        command = 'cp %s %s' % (fname, os.path.join(dirname,fname))
        #~ print "*", command
        os.system(command)
        
    def copy_file_from_server(self,dirname,fname):
        """
        """
        if not self.check_connection():
            return False
            
        dirname = os.path.join(self.mount_location,dirname)
        
        command = 'cp %s %s' % (os.path.join(dirname,fname),fname)
        #~ print "*", command
        os.system(command)
        
    def get_status(self):
        """
        """
        # return status of server (off, not mounted, etc)
        raise NotImplementedError
        return status
        
    def get_queue(self,status=None):
        """
        """
        if not self.check_connection():
            return False
        
        queue = {}

        for fname in glob(os.path.join(self.mount_location,'server.tasks','*.*')):
            d, s = os.path.basename(fname).split('.')
            if s not in queue.keys():
                queue[s]=[]
            queue[s].append(d)
            
        return queue if status is None else queue[status]
        
    def get_job_status(self,job):
        """
        return the status of a job on this server
        
        @param job: dirname of job or job object
        @type job: str or job
        @return: status
        @rtype: str
        """
        if not self.check_connection():
            return False
            
        if isinstance(job,str):
            dirname = job
        else:
            dirname = job.dirname
            
        
        fname = glob(os.path.join(self.mount_location,'server.tasks','%s.*' % (dirname)))[0]
        
        return os.path.basename(fname).split('.')[1]
        
    def task_update(self,task):
        """
        """
        #COPIED FROM DAEMON with '..' changed to self.mount_location
        
        oldname = glob(os.path.join(self.mount_location,'server.tasks','%s.*' % (task['dirname'])))[0] #could be running or queue
        newname = os.path.join(self.mount_location,'server.tasks','%s.%s' % (task['dirname'],task['status']))
        os.rename(oldname,newname)
        
        f = open(newname,'w')
        f.write(json.dumps(task))
        f.close()
        
        
class Job(object):
    def __init__(self,name,script_file,aux_files=[],return_files='all',dirname=None):
        self.name = name
        self.script_file = script_file
        self.aux_files = aux_files
        self.return_files = aux_files if return_files == 'all' else return_files
        self.dirname = dirname if dirname is not None else str(uuid4())
        
        self.server = None
        
    def as_dict(self):
        """
        dictionary representation
        """
        d = {}
        d['name'] = self.name
        d['script_file'] = self.script_file
        d['dirname'] = self.dirname
        return d

    def start(self,server,dirname=None):
        """
        start the job on a given server
        
        @param server: server 
        @type server: server object
        """
        self.server = server
        if not self.server.check_connection():
            return False
        
        self.server = server
        all_files = self.aux_files[:]
        all_files.append(self.script_file)
        for f in all_files:
            server.copy_file_to_server(self.dirname,f)
        
        server.add_script_to_queue(self)
        
    def get_status(self):
        """
        get the status of this job
        
        @return: status
        @rtype: str
        """
        
        if self.server is None:
            return 'no server'
        else:
            return self.server.get_job_status(self)
        
    def retrieve_results(self):
        """
        copy back all the files in return_files
        will only be successful if job.get_status()=='complete'
        
        @return: success
        @rtype: bool
        """
        if not self.server.check_connection():
            return False
        
        if self.server.get_job_status(self.dirname) != 'complete':
            return False
            
        # copy back required files
        for f in self.return_files:
            self.server.copy_file_from_server(self.dirname,f)

        # update task to delete on next pass
        task = self.as_dict()
        task['status']='pending_delete'
        self.server.task_update(task)

        return True
        
    def wait_for_results(self,wait=5):
        """
        continue to try to retrieve results until successful
        
        @param wait: wait time between attempts in seconds
        @type wait: int
        @return: success
        @rtype: bool
        """
        complete = False
        while not complete:
            complete = self.retrieve_results()
            sleep(wait)
        return True
 

    def cancel(self):
        raise NotImplementedError
        
