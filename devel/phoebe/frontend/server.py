#!/usr/bin/python

import os
from datetime import datetime
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
        self.server_ps = parameters.ParameterSet(context='server',label=label,server=server,server_dir=server_dir,server_script=server_script,mount_dir=mount_dir)

        local = self.is_local()
        self.last_known_status = {'mount': local, 'ping': local, 'test': local, 'status': local, 'phoebe_version': 'unknown'}

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
        return self.server_ps.get_value('server') != 'None'
        
    def is_local(self):
        """
        check whether this server is on the local machine
        """
        return self.server_ps.get_value('server') == 'None'
        
    def check_mount(self):
        """
        checks whether the server is local or, if external, whether the mount location exists
        """
        success = self.is_local() or (os.path.exists(self.server_ps.get_value('mount_dir')) and os.path.isdir(self.server_ps.get_value('mount_dir')))
        self.last_known_status['mount'] = success
        return success
        
    def check_ping(self):
        """
        checks whether the server is local or, if external, whether the ping to the server is successful
        """
        if self.is_local():
            return True
        pingdata = os.popen('ping -c 1 %s' % (self.server_ps.get_value('server'))).readlines()
        # Extract the ping time
        if len(pingdata) < 2:
            # Failed to find a DNS resolution or route
            success = False
        else:
            index = pingdata[1].find('time=')
            if index == -1:
                # Ping failed or timed-out
                success = False
            else:
                success = True
                
        self.last_known_status['ping'] = success
        return success
        
    def check_test(self):
        """
        checks whether the server is local or, if external, whether a test command can be run and retrieved from the server
        """
        if self.is_local():
            return True
        if not self.check_mount() and self.check_ping():
            return False
        timestr = str(datetime.now()).replace(' ','_')
        command = "touch %s" % os.path.join(self.server_ps.get_value('server_dir') ,"test.%s" % timestr)
        #~ print "*** ssh %s '%s'" % (self.server_ps.get_value('server'),command)
        os.system("ssh %s '%s'" % (self.server_ps.get_value('server'),command))
        
        fname = os.path.join(self.server_ps.get_value('mount_dir'), 'test.%s' % timestr)
        success = os.path.exists(fname)
        
        if success:
            os.remove(fname)
        self.last_known_status['test'] = success
        return success
        
    def check_phoebe_version(self):
        """
        
        """
        return 'unknown'
        
    def check_status(self,full_output=False):
        """
        runs check_mount, check_ping, check_test, and check_phoebe_version
        this also sets server.last_known_status
        """
        
        mount = self.check_mount()
        self.last_known_status['mount'] = mount
        ping = self.check_ping()
        self.last_known_status['ping'] = ping
        test = False if not (mount and ping) else self.check_test()
        self.last_known_status['test'] = test
        self.last_known_status['phoebe_version'] = self.check_phoebe_version()
        status = mount and ping and test
        self.last_known_status['status'] = status
        
        if full_output:
            return self.last_known_status
        else:
            return status
        
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
        script = os.path.join(server_dir,script)
        
        command = ''
        if server_script != '':
            command += "%s && " % server_script
        # change status to running
        command += "nohup echo 'running' > %s.status" % (script)
        # script with nohup, redirecting stderr and stdout to [script].log
        command += "&& nohup python %s > %s.log 2>&1 " % (script,script)
        # when script is complete, create new file to notify client
        command += "&& nohup echo 'complete' > %s.status" % (script)
        # run command in background
        #~ print ("call: ssh %s '%s' &" % (server,command))
        os.system("ssh %s '%s' &" % (server,command))
        return

    def check_script_status(self, script):
        """
        
        """
        if self.is_local():
            print 'server is local'
            return 
                    
        fname = os.path.join(self.server_ps.get_value('mount_dir'),'%s.status' % script)
        if not os.path.exists(fname):
            return False
            
        f = open(fname, 'r')
        status = f.readline().strip()
        f.close()
        return status
