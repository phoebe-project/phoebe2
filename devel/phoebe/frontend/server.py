#!/usr/bin/python

import os

class Server(object):
    def __init__(self,mpi,server=None,server_dir='~/',mount_location=None):
        """
        @param mpi: the mpi options to use for this server
        @type mpi: ParameterSet with context 'mpi'
        @param server: the location of the server (used for ssh - so username@servername if necessary), or None if local
        @type server: str
        @param server_dir: directory on the server to copy files and run scripts
        @type server_dir: str
        @param mount_location: local mounted location of server:server_dir, or None if local
        @type mount_location: str or None
        """
        self.mpi = mpi
        self.server = server
        self.server_dir = server_dir
        self.mount_location = mount_location # local path to server:server_dir
        
    def is_external(self):
        """
        """
        return self.server is not None
        
    def is_local(self):
        """
        """
        return self.server is None
        
    def check_connection(self):
        """
        checks whether the server is local or, if external, whether the mount location exists
        """
        return self.is_local() or (os.path.exists(self.mount_location) and os.path.isdir(self.mount_location))

    def run_script_external(self, script):
        """
        run an existing script on the server remotely
        
        @param script: filename of the script on the server (server:server_dir/script) 
        @type script: str
        """
        if self.is_local():
            print 'server is local'
            return
        
        # files and script should already have been copied/saved to self.mount_location
        # the user or bundle is responsible for this
        os.system("ssh %s 'python %s/%s'" % (self.server,self.server_dir,script))
