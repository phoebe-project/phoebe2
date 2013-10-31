#!/usr/bin/python

from glob import glob
from time import sleep
import os
import json

class Daemon(object):
    # we'll currently assume this is running in the directory that is mounted
    
    def __init__(self,cwd=None,sleep=5):
        # cwd needs to be same directory that is mount_location for client
        # this directory should have all the subdirectories and server.tasks in it
        
        self.sleep = sleep
        
        self.cwd = os.getcwd() if cwd is None else cwd #needs to be the 
        
        # check if tasks directory exists
        if not os.path.exists('./server.tasks'):
            os.makedirs('./server.tasks')
        
    def run(self):
        while True:
            for task in self.get_tasks_in_queue(status='queue'):
                print "server running:", task
                try:
                    os.chdir(os.path.join(self.cwd,task['dirname']))
                    task['status'] = 'running'
                    self.task_update(task) # to say running
                    os.system('python %s' % task['script_file'])
                except:
                    task['status'] = 'failed'
                else:
                    task['status'] = 'complete'
                self.task_update(task) # to say failed or complete
                
            self.remove_pending_delete()
            sleep(self.sleep)


    def get_tasks_in_queue(self,status='queue'):
    
        tasks = []
        for fname in glob(os.path.join(self.cwd,'server.tasks','*.%s' % status)):
            f=open(fname,'r')
            tasks.append(json.loads(f.read()))
            f.close()
            
        return tasks
        
    def task_update(self,task):
        
        oldname = glob(os.path.join(self.cwd,'server.tasks','%s.*' % (task['dirname'])))[0] #could be running or queue
        newname = os.path.join(self.cwd,'server.tasks','%s.%s' % (task['dirname'],task['status']))
        os.rename(oldname,newname)
        
        f = open(newname,'w')
        f.write(json.dumps(task))
        f.close()
        
    def remove_pending_delete(self):
        
        tasks = self.get_tasks_in_queue('pending_delete')
        
        for task in tasks:
            # remove files from subdir
            print "server removing", task
            os.system('rm -r %s' % os.path.join(self.cwd,task['dirname']))
        
            # change status of task file
            task['status']='deleted'
            self.task_update(task)
        
        
daemon = Daemon()
daemon.run()
