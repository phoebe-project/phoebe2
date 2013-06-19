"""
MPI interface to the observatory.
"""
import numpy as np
from mpi4py import MPI
import pickle
import cPickle
import sys
import os
import time
from phoebe.backend import observatory
from phoebe.backend import universe
from phoebe.utils import utils

def update_progress(progress, width=80):
    """
    Displays or updates a console progress bar
    
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    """
    barLength = max(1, width-20-20) # Modify this to change the length of the progress bar
    
    progress = float(progress)
    if progress < 0:
        progress = 0
        status = "{:20s}".format("First time point...")+"\r"
    # At the end, report and let the progressbar disappear
    elif progress > 1:
        progress = 1
        status = "{:20s}".format('Done...')+"\r" + " "*(20+20+barLength)+"\r"
    elif progress <= 0.25:
        status = "{:20s}".format("Warming up...")
    elif progress <= 0.50:
        status = "{:20s}".format("Going steady...")
    elif progress <= 0.75:
        status = "{:20s}".format("Wait for it...")
    elif progress <= 1.00:
        status = "{:20s}".format("Almost there...")    
    else:
        status = "{:20s}".format("Running...")
    block = int(round(barLength*progress))
    text = "MPIRUN: [{0}] {1:7.3f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    text = "\r" + text[-width:]
    sys.stdout.write(text)
    sys.stdout.flush()

if __name__=="__main__":
    comm = MPI.COMM_WORLD

    # arbitrary tag numbers:
    TAG_REQ = 41
    TAG_DATA = 42
    TAG_RES = 43

    myrank = comm.Get_rank()
    nprocs = comm.Get_size()

    #logger = utils.get_basic_logger(filename='worker{}.log'.format(myrank),flevel='INFO')

    # Which function do we wich to run?
    function = sys.argv[1]

    if myrank == 0:
        
        try:
            # Load the system that we want to compute
            system = universe.load(sys.argv[2])
            # Load the arguments to the function
            with open(sys.argv[3],'r') as ff: args = cPickle.load(ff)
            # Load the keyword arguments to the function
            with open(sys.argv[4],'r') as ff: kwargs = cPickle.load(ff)
        finally:
            #-- Clean up pickle files once they are loaded:
            os.unlink(sys.argv[2])
            os.unlink(sys.argv[3])
            os.unlink(sys.argv[4])
        
        res = []


        # Derive at which phases this system needs to be computed
        params = kwargs.pop('params')
        observatory.extract_times_and_refs(system,params,tol=1e-6)
        dates = params['time']
        labels = params['refs']
        types = params['types']
        
        # This is the manager: we set the time of the system first, so that
        # the mesh gets created. This system will be distributed over the nodes,
        # so the workers have less overhead.
        if params['refl']:
            system.prepare_reflection(ref='all')
            system.fix_mesh()
        update_progress(-1)#print("MPIrun: calculating at first time point")
        system.set_time(params['time'][0])
        
        
        # Instead of computing one phase per worker, we'll let each worker do
        # a few. We're conservative on the number of phases each worker should do,
        # just to make sure that that if one worker is a bit lazy, the rest doesn't
        # have to wait for too long...: e.g. if there are 8 phases left and there
        # are 4 workers, we only take 1 phase point for each worker.
        olength = len(dates)
        while len(dates):
            #-- print some diagnostics to the user.
            #print("MPIrun: starting at {:.3f}% of total run".format((olength-len(dates))/float(olength)*100))
            N = len(dates)
            take = max(N/(2*nprocs),1)
            do_dates,dates = dates[:take],dates[take:]
            do_labels,labels = labels[:take],labels[take:]
            do_types,types = types[:take],types[take:]
            # Pass on the subset of dates/labels/types to compute
            kwargs['params'] = params.copy()
            kwargs['params']['time'] = do_dates
            kwargs['params']['refs'] = do_labels
            kwargs['params']['types'] = do_types
            
            # Wait for a free worker:
            node = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQ)
            
            # Send the phase and the system to that worker.
            packet = {'system':system, 'args':args, 'kwargs':kwargs, 'continue': True}
            comm.send(packet, node, tag=TAG_DATA)

            # Store the results asynchronously:
            res.append(comm.irecv(bytearray(500000), node, tag=TAG_RES))
            
            update_progress((olength-len(dates))/float(olength))
            
        
        update_progress(1.1)#print("MPIrun: finished")
        
        packet = {'continue': False}
        for i in range(1,nprocs):
            node = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQ)
            comm.send(packet, node, tag=TAG_DATA)
        
        # Collect all the results: these are 'empty' Bodies containing only the
        # results
        output = []
        for i in range(len(res)):
            #res[i].wait()
            done, val = res[i].test()
            output.append(val)
        
        # Now merge the results with the first Body, and save that one to a
        # pickle.
        output = universe.merge_results(output)
        output.save(sys.argv[2])
        
        
    else:
        # This is the worker.
        while True:
            # Send the work request:
            comm.send(myrank, 0, tag=TAG_REQ)
            
            # Receive the work order:
            packet = comm.recv (tag=TAG_DATA)
            if packet['continue'] == False:
                break
            # Do the work:
            getattr(observatory,function)(packet['system'],*packet['args'],**packet['kwargs'])
            
            # Send the results back to the manager:
            the_result = universe.keep_only_results(packet['system'])
            #print "BYTESIZE",sys.getsizeof(cPickle.dumps(the_result))
            comm.send(the_result, 0, tag=TAG_RES)
