#!/usr/bin/python

import os
from phoebe.parameters import parameters

class Server(object):
    def __init__(self,mpi,label=None,server=None,server_dir='~/',server_script='',mount_dir=None):
        """
        @param mpi: the mpi options to use for this server
        @type mpi: ParameterSet with context 'mpi'
        @param server: the location of the server (used for ssh - so username@servername if necessary), or None if local
        @type server: str
        @param server_dir: directory on the server to copy files and run scripts
        @type server_dir: str
        @param server_script: location on the server of a script to run (ie. to setup a virtual environment) before running phoebe
        @type server_script: str
        @param mount_dir: local mounted location of server:server_dir, or None if local
        @type mount_dir: str or None
        """
        self.mpi_ps = mpi
        if server is not None:
            self.server_ps = parameters.ParameterSet(context='server',label=label,server=server,server_dir=server_dir,server_script=server_script,mount_dir=mount_dir)
        else:
            self.server_ps = None

    def set_value(self,qualifier,value):
        
        if self.mpi_ps is not None and qualifier in self.mpi_ps.keys():
            self.mpi_ps.set_value(qualifier,value)
        elif self.server_ps is not None and qualifier in self.server_ps.keys():
            self.server_ps.set_value(qualifier,value)
        else:
            raise IOError('parameter not understood')
            
    def get_value(self,qualifier):
        
        if self.mpi_ps is not None and qualifier in self.mpi_ps.keys():
            return self.mpi_ps.get_value(qualifier)
        elif self.server_ps is not None and qualifier in self.server_ps.keys():
            return self.server_ps.get_value(qualifier)
        else:
            raise IOError('parameter not understood')
            
    def is_external(self):
        """
        checks whether this server is on an external machine (not is_local())
        """
        return self.server_ps is not None
        
    def is_local(self):
        """
        check whether this server is on the local machine
        """
        return self.server_ps is None
        
    def check_connection(self):
        """
        checks whether the server is local or, if external, whether the mount location exists
        """
        return self.is_local() or (os.path.exists(self.server_ps.get_value('mount_dir')) and os.path.isdir(self.server_ps.get_value('mount_dir')))

    def run_script_external(self, script):
        """
        run an existing script on the server remotely
        
        @param script: filename of the script on the server (server:server_dir/script) 
        @type script: str
        """
        if self.is_local():
            print 'server is local'
            return
        
        # files and script should already have been copied/saved to self.mount_dir
        # the user or bundle is responsible for this
        server = self.server_ps.get_value('server')
        server_dir = self.server_ps.get_value('server_dir')
        server_script = self.server_ps.get_value('server_script')
        
        command = ''
        if server_script != '':
            command += '%s &&' % server_script
        command += 'python %s' % (os.path.join(server_dir,script))
        os.system("ssh %s '%s'" % (server,command))
